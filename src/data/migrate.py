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
    # 概念熱度加成（Stage 3.3b）
    ("discovery_record", "concept_bonus", "REAL"),
    # 隔日沖大戶偵測欄位
    ("discovery_record", "daytrade_penalty", "REAL"),
    ("discovery_record", "daytrade_tags", "VARCHAR(200)"),
    # 輪動持倉止損價持久化（進場時鎖定，不隨 discover 每日更新）
    ("rotation_position", "stop_loss", "REAL"),
    # 籌碼層級變化稽核（如 "8F→7F"）
    ("discovery_record", "chip_tier_change", "VARCHAR(20)"),
    # Universe Filter 強化：DailyFeature 新增相對流動性 + 20 日最高價
    ("daily_feature", "turnover_ratio_5d_20d", "REAL"),
    ("daily_feature", "high_20d", "REAL"),
]

# Phase 2 效能優化：複合索引加速頻繁查詢
# (index_name, table_name, columns)
INDEX_MIGRATIONS: list[tuple[str, str, str]] = [
    ("ix_daily_price_stock_date", "daily_price", "stock_id, date"),
    ("ix_institutional_stock_date", "institutional_investor", "stock_id, date"),
    ("ix_broker_trade_stock_date", "broker_trade", "stock_id, date"),
    ("ix_securities_lending_stock_date", "securities_lending", "stock_id, date"),
    ("ix_daily_feature_stock_date", "daily_feature", "stock_id, date"),
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

        # 複合索引遷移（CREATE INDEX IF NOT EXISTS）
        for idx_name, table, columns in INDEX_MIGRATIONS:
            try:
                session.execute(text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({columns})"))
                session.commit()
                logger.debug("索引已確保存在: %s ON %s(%s)", idx_name, table, columns)
            except Exception:
                logger.debug("索引建立跳過: %s（可能已存在或表不存在）", idx_name)

    return added
