"""簡易排程器 — 使用 schedule 函式庫（需前景執行）。

daily_sync_job() delegate 給 main.py cmd_morning_routine()，
確保排程流程與 CLI morning-routine 完全一致。
"""

from __future__ import annotations

import argparse
import logging
import time

import schedule

logger = logging.getLogger(__name__)


def daily_sync_job() -> None:
    """每日同步任務 — delegate 給 morning-routine（Step 0~7 + Discord 摘要）。"""
    logger.info("=" * 60)
    logger.info("開始每日自動同步（morning-routine）")
    logger.info("=" * 60)

    try:
        from main import cmd_morning_routine

        cmd_morning_routine(
            argparse.Namespace(
                dry_run=False,
                skip_sync=False,
                top=20,
                notify=True,
            )
        )
        logger.info("morning-routine 完成")
    except Exception:
        logger.exception("morning-routine 執行失敗")


def weekly_holding_job() -> None:
    """每週四同步大戶持股分級週資料。"""
    logger.info("=" * 60)
    logger.info("開始每週大戶持股分級同步")
    logger.info("=" * 60)

    try:
        from src.data.pipeline import sync_holding_distribution

        sync_holding_distribution(weeks=4)
        logger.info("大戶持股分級同步完成")
    except Exception:
        logger.exception("大戶持股分級同步失敗")


def run_scheduler() -> None:
    """啟動排程器（阻塞式）。

    每日任務：23:00 執行 morning-routine（含 sync/sbl/broker/discover all/alert/watch/revenue/anomaly）
    週四任務：23:30 執行 sync-holding（TDCC 大戶持股週資料）
    """
    schedule.every().day.at("23:00").do(daily_sync_job)
    schedule.every().thursday.at("23:30").do(weekly_holding_job)

    print("排程器已啟動")
    print("  每日 23:00  — morning-routine（Step 0~7 + Discord 摘要）")
    print("  每週四 23:30 — sync-holding（大戶持股週資料）")
    print("按 Ctrl+C 停止")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n排程器已停止")
