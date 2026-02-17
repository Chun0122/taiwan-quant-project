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
