"""SQLite Schema 遷移輔助 — 為既有資料庫新增欄位。

使用 ALTER TABLE ADD COLUMN，欄位已存在則跳過。
新表由 SQLAlchemy create_all 自動建立。
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from src.data.database import get_session, init_db

logger = logging.getLogger(__name__)

# (table_name, column_name, column_type)
MIGRATIONS: list[tuple[str, str, str]] = [
    ("backtest_result", "sortino_ratio", "REAL"),
    ("backtest_result", "calmar_ratio", "REAL"),
    ("backtest_result", "var_95", "REAL"),
    ("backtest_result", "cvar_95", "REAL"),
    ("backtest_result", "profit_factor", "REAL"),
    ("trade", "exit_reason", "VARCHAR(20)"),
    ("discovery_record", "entry_price", "REAL"),
    ("discovery_record", "stop_loss", "REAL"),
    ("discovery_record", "take_profit", "REAL"),
    ("discovery_record", "entry_trigger", "VARCHAR(100)"),
    ("discovery_record", "valid_until", "DATE"),
    ("trade", "stop_price", "REAL"),
    ("trade", "target_price", "REAL"),
    ("announcement", "event_type", "VARCHAR(20)"),
    # P3: WatchEntry 移動止損欄位
    ("watch_entry", "trailing_stop_enabled", "INTEGER DEFAULT 0"),  # SQLite 用 INTEGER 表示 bool
    ("watch_entry", "trailing_atr_multiplier", "REAL"),
    ("watch_entry", "highest_price_since_entry", "REAL"),
    # P1: Discover 籌碼因子層級透明度
    ("discovery_record", "chip_tier", "VARCHAR(5)"),
    # Universe Filter: StockInfo 新增有價證券類型欄位（stock/etf/warrant/preferred/None）
    ("stock_info", "security_type", "VARCHAR(20)"),
    # 雙窗口流動性確認：DailyFeature 新增 20 日均成交金額（防短暫放量誘多）
    ("daily_feature", "turnover_ma20", "REAL"),
]


def run_migrations() -> list[str]:
    """執行所有 schema 遷移，回傳成功新增的欄位清單。"""
    # 先確保新表已建立
    init_db()

    added: list[str] = []

    with get_session() as session:
        for table, column, col_type in MIGRATIONS:
            # 檢查欄位是否已存在
            result = session.execute(text(f"PRAGMA table_info({table})"))
            existing_cols = {row[1] for row in result.fetchall()}

            if column in existing_cols:
                logger.debug("欄位已存在: %s.%s", table, column)
                continue

            stmt = f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
            session.execute(text(stmt))
            session.commit()
            added.append(f"{table}.{column}")
            logger.info("新增欄位: %s.%s (%s)", table, column, col_type)

    return added
