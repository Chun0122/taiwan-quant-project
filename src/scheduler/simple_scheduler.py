"""簡易排程器 — 使用 schedule 函式庫（需前景執行）。"""

from __future__ import annotations

import logging
import time

import schedule

from src.data.pipeline import sync_watchlist, sync_indicators, sync_taiex_index

logger = logging.getLogger(__name__)


def daily_sync_job() -> None:
    """每日同步、計算、篩選、通知任務。"""
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
        return

    # 同步完成後執行篩選 + 通知
    try:
        from src.screener.engine import MultiFactorScreener
        from src.notification.line_notify import send_scan_results
        from src.config import settings

        logger.info("開始每日自動篩選")
        screener = MultiFactorScreener()
        results = screener.scan()

        if not results.empty:
            logger.info("篩選完成，找到 %d 檔符合條件", len(results))
            if settings.discord.webhook_url and settings.discord.enabled:
                send_scan_results(results)
                logger.info("Discord 通知已發送")
        else:
            logger.info("篩選完成，無符合條件的股票")
    except Exception:
        logger.exception("每日篩選/通知失敗")


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
