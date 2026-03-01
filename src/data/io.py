"""通用資料匯出/匯入模組。

支援將任意 ORM 資料表匯出為 CSV/Parquet，
以及從 CSV/Parquet 匯入資料（含欄位驗證 + upsert）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import func, inspect, select
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

from src.data.database import get_session, init_db
from src.data.schema import (
    Announcement,
    BacktestResult,
    DailyPrice,
    DiscoveryRecord,
    Dividend,
    FinancialStatement,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
    PortfolioBacktestResult,
    PortfolioTrade,
    StockInfo,
    StockValuation,
    TechnicalIndicator,
    Trade,
)

logger = logging.getLogger(__name__)

# ── 資料表名稱 → ORM Model 映射 ──────────────────────────────

TABLE_REGISTRY: dict[str, type] = {
    "daily_price": DailyPrice,
    "institutional_investor": InstitutionalInvestor,
    "margin_trading": MarginTrading,
    "monthly_revenue": MonthlyRevenue,
    "stock_valuation": StockValuation,
    "dividend": Dividend,
    "technical_indicator": TechnicalIndicator,
    "announcement": Announcement,
    "financial_statement": FinancialStatement,
    "backtest_result": BacktestResult,
    "trade": Trade,
    "portfolio_backtest_result": PortfolioBacktestResult,
    "stock_info": StockInfo,
    "portfolio_trade": PortfolioTrade,
    "discovery_record": DiscoveryRecord,
}

# ── 資料表 → 衝突鍵（用於 upsert import）────────────────────

CONFLICT_KEYS: dict[str, list[str]] = {
    "daily_price": ["stock_id", "date"],
    "institutional_investor": ["stock_id", "date", "name"],
    "margin_trading": ["stock_id", "date"],
    "monthly_revenue": ["stock_id", "date"],
    "stock_valuation": ["stock_id", "date"],
    "dividend": ["stock_id", "date"],
    "technical_indicator": ["stock_id", "date", "name"],
    "announcement": ["stock_id", "date", "seq"],
    "financial_statement": ["stock_id", "date"],
    "stock_info": ["stock_id"],
    "discovery_record": ["scan_date", "mode", "stock_id"],
    # 以下表無 UniqueConstraint，使用主鍵
    "backtest_result": ["id"],
    "trade": ["id"],
    "portfolio_backtest_result": ["id"],
    "portfolio_trade": ["id"],
}


def _get_column_names(model: type) -> list[str]:
    """取得 ORM model 的所有欄位名稱。"""
    mapper = inspect(model)
    return [col.key for col in mapper.column_attrs]


def _get_required_columns(model: type) -> list[str]:
    """取得不可為 NULL 且非自增主鍵的欄位名稱。"""
    mapper = inspect(model)
    required = []
    for col_attr in mapper.column_attrs:
        col = col_attr.columns[0]
        if col.primary_key and col.autoincrement:
            continue
        if not col.nullable and col.default is None and col.server_default is None:
            required.append(col_attr.key)
    return required


# ── 匯出 ─────────────────────────────────────────────────────


def list_tables() -> list[dict]:
    """列出所有支援的資料表及筆數。

    Returns:
        list[dict]: [{"table": "daily_price", "count": 12345}, ...]
    """
    init_db()
    result = []
    with get_session() as session:
        for table_name, model in TABLE_REGISTRY.items():
            count = session.execute(select(func.count()).select_from(model)).scalar()
            result.append({"table": table_name, "count": count})
    return result


def export_table(
    table_name: str,
    output_path: str | None = None,
    fmt: str = "csv",
    stocks: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> int:
    """匯出指定資料表。

    Args:
        table_name: 資料表名稱（必須在 TABLE_REGISTRY 中）。
        output_path: 輸出路徑。None 時自動產生 data/export/<table>.<fmt>。
        fmt: 輸出格式，"csv" 或 "parquet"。
        stocks: 篩選股票代號列表（僅限有 stock_id 欄位的表）。
        start_date: 起始日期 YYYY-MM-DD（僅限有 date 欄位的表）。
        end_date: 結束日期 YYYY-MM-DD。

    Returns:
        匯出筆數。

    Raises:
        ValueError: 表名不存在或格式不支援。
    """
    if table_name not in TABLE_REGISTRY:
        raise ValueError(f"不支援的資料表: {table_name}（可用: {', '.join(TABLE_REGISTRY.keys())}）")
    if fmt not in ("csv", "parquet"):
        raise ValueError(f"不支援的格式: {fmt}（可用: csv, parquet）")

    model = TABLE_REGISTRY[table_name]
    columns = _get_column_names(model)

    # 排除自增主鍵 id
    export_columns = [c for c in columns if c != "id"]

    init_db()
    stmt = select(model)

    # 股票篩選
    if stocks and hasattr(model, "stock_id"):
        stmt = stmt.where(model.stock_id.in_(stocks))

    # 日期篩選（偵測 date 或 scan_date 欄位）
    date_col = None
    if hasattr(model, "date"):
        date_col = model.date
    elif hasattr(model, "scan_date"):
        date_col = model.scan_date

    if start_date and date_col is not None:
        stmt = stmt.where(date_col >= start_date)
    if end_date and date_col is not None:
        stmt = stmt.where(date_col <= end_date)

    with get_session() as session:
        rows = session.execute(stmt).scalars().all()
        if not rows:
            return 0

        records = []
        for row in rows:
            records.append({col: getattr(row, col) for col in export_columns})

    df = pd.DataFrame(records)

    # 輸出路徑
    if output_path is None:
        out_dir = Path("data/export")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{table_name}.{fmt}")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        try:
            df.to_parquet(output_path, index=False)
        except ImportError:
            raise ValueError("匯出 Parquet 需安裝 pyarrow：pip install pyarrow")

    logger.info("匯出 %s: %d 筆 → %s", table_name, len(df), output_path)
    return len(df)


# ── 匯入 ─────────────────────────────────────────────────────


def validate_import(table_name: str, df: pd.DataFrame) -> list[str]:
    """驗證 DataFrame 欄位是否符合目標表。

    Args:
        table_name: 資料表名稱。
        df: 待匯入的 DataFrame。

    Returns:
        錯誤訊息列表（空 = 通過驗證）。
    """
    if table_name not in TABLE_REGISTRY:
        return [f"不支援的資料表: {table_name}"]

    errors: list[str] = []
    model = TABLE_REGISTRY[table_name]
    required = _get_required_columns(model)

    # 檢查必要欄位
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"缺少必要欄位: {', '.join(missing)}")

    if errors:
        return errors

    # 檢查日期欄位可解析
    all_columns = _get_column_names(model)
    mapper = inspect(model)
    for col_attr in mapper.column_attrs:
        col_name = col_attr.key
        if col_name not in df.columns:
            continue
        col_type = col_attr.columns[0].type
        if isinstance(col_type, (type(DailyPrice.date.type),)) or col_type.__class__.__name__ == "Date":
            # 嘗試解析日期
            try:
                pd.to_datetime(df[col_name])
            except (ValueError, TypeError):
                errors.append(f"欄位 {col_name} 包含無法解析的日期值")

    return errors


def import_table(
    table_name: str,
    source_path: str,
    dry_run: bool = False,
) -> int:
    """從 CSV/Parquet 匯入資料到指定表。

    Args:
        table_name: 目標資料表名稱。
        source_path: 來源檔案路徑（.csv 或 .parquet）。
        dry_run: True 時僅驗證不寫入。

    Returns:
        匯入（或驗證通過的）筆數。

    Raises:
        ValueError: 表名不存在、檔案格式不支援、驗證失敗。
        FileNotFoundError: 來源檔案不存在。
    """
    if table_name not in TABLE_REGISTRY:
        raise ValueError(f"不支援的資料表: {table_name}（可用: {', '.join(TABLE_REGISTRY.keys())}）")

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案: {source_path}")

    # 讀取資料
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(source_path, encoding="utf-8-sig")
    elif suffix in (".parquet", ".pq"):
        try:
            df = pd.read_parquet(source_path)
        except ImportError:
            raise ValueError("讀取 Parquet 需安裝 pyarrow：pip install pyarrow")
    else:
        raise ValueError(f"不支援的檔案格式: {suffix}（可用: .csv, .parquet）")

    if df.empty:
        return 0

    # 驗證
    errors = validate_import(table_name, df)
    if errors:
        raise ValueError("資料驗證失敗:\n" + "\n".join(f"  - {e}" for e in errors))

    # 過濾至目標表的有效欄位（移除 id 和多餘欄位）
    model = TABLE_REGISTRY[table_name]
    valid_columns = [c for c in _get_column_names(model) if c != "id"]
    df = df[[c for c in valid_columns if c in df.columns]]

    # 日期欄位轉換
    mapper = inspect(model)
    for col_attr in mapper.column_attrs:
        col_name = col_attr.key
        if col_name not in df.columns:
            continue
        col_type = col_attr.columns[0].type
        if col_type.__class__.__name__ == "Date":
            df[col_name] = pd.to_datetime(df[col_name]).dt.date
        elif col_type.__class__.__name__ == "DateTime":
            df[col_name] = pd.to_datetime(df[col_name])

    if dry_run:
        logger.info("Dry run: %s 驗證通過，%d 筆資料", table_name, len(df))
        return len(df)

    # 寫入 DB
    init_db()
    conflict_keys = CONFLICT_KEYS[table_name]
    records = df.to_dict("records")
    batch_size = 80

    with get_session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            stmt = sqlite_upsert(model).values(batch)
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_keys)
            session.execute(stmt)
        session.commit()

    logger.info("匯入 %s: %d 筆 ← %s", table_name, len(records), source_path)
    return len(records)
