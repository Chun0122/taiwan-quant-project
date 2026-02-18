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

    # 每日選股報告
    try:
        from src.report.engine import DailyReportEngine
        from src.report.formatter import format_daily_report
        from src.notification.line_notify import send_message
        from src.config import settings

        logger.info("開始生成每日選股報告")
        engine = DailyReportEngine(ml_enabled=False)
        report_df = engine.run()

        if not report_df.empty:
            logger.info("每日報告完成，共 %d 檔", len(report_df))
            if settings.discord.webhook_url and settings.discord.enabled:
                msgs = format_daily_report(report_df, top_n=10)
                for msg in msgs:
                    send_message(msg)
                logger.info("每日報告 Discord 通知已發送")
        else:
            logger.info("每日報告：無資料")
    except Exception:
        logger.exception("每日報告生成/通知失敗")

    # 全市場選股掃描
    try:
        from src.data.pipeline import sync_market_data, sync_stock_info
        from src.discovery.scanner import MarketScanner
        from src.report.formatter import format_discovery_report
        from src.notification.line_notify import send_message
        from src.config import settings

        logger.info("開始全市場選股掃描")
        sync_stock_info(force_refresh=False)
        sync_market_data(days=3)

        scanner = MarketScanner()
        result = scanner.run()

        if not result.rankings.empty:
            logger.info("全市場掃描完成，共 %d 支候選", len(result.rankings))
            if settings.discord.webhook_url and settings.discord.enabled:
                msgs = format_discovery_report(result, top_n=20)
                for msg in msgs:
                    send_message(msg)
                logger.info("全市場掃描 Discord 通知已發送")
        else:
            logger.info("全市場掃描：無符合條件的股票")
    except Exception:
        logger.exception("全市場掃描/通知失敗")


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
