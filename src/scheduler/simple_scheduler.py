"""簡易排程器 — 使用 schedule 函式庫（需前景執行）。"""

from __future__ import annotations

import logging
import time

import schedule

from src.data.pipeline import sync_watchlist, sync_indicators, sync_taiex_index

logger = logging.getLogger(__name__)


def daily_sync_job() -> None:
    """每日同步與計算任務。"""
    logger.info("=" * 60)
    logger.info("開始每日自動同步")
    logger.info("=" * 60)

    try:
        sync_taiex_index()
        sync_watchlist()
        sync_indicators()
        logger.info("每日同步完成")
    except Exception:
        logger.exception("每日同步失敗")


def run_scheduler() -> None:
    """啟動排程器（阻塞式，每日 23:00 執行）。"""
    schedule.every().day.at("23:00").do(daily_sync_job)

    print("排程器已啟動，每日 23:00 自動同步")
    print("按 Ctrl+C 停止")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n排程器已停止")
