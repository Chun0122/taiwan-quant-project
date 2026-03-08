"""簡易排程器 — 使用 schedule 函式庫（需前景執行）。"""

from __future__ import annotations

import logging
import time

import schedule

from src.data.pipeline import sync_indicators, sync_taiex_index, sync_watchlist

logger = logging.getLogger(__name__)


def daily_sync_job() -> None:
    """每日同步、計算、篩選、通知任務。"""
    logger.info("=" * 60)
    logger.info("開始每日自動同步")
    logger.info("=" * 60)

    # Step 1: 同步日K + TAIEX
    try:
        sync_taiex_index()
        sync_watchlist()
        logger.info("日K + TAIEX 同步完成")
    except Exception:
        logger.exception("日K 同步失敗")
        return

    # Step 2: 同步借券賣出（全市場，前一交易日）
    try:
        from src.data.pipeline import sync_sbl_all_market

        logger.info("開始同步借券賣出資料")
        sync_sbl_all_market(days=3)
        logger.info("借券賣出同步完成")
    except Exception:
        logger.exception("借券賣出同步失敗，繼續後續流程")

    # Step 3: 同步分點進出（上次 discover 推薦標的 + watchlist）
    try:
        from sqlalchemy import func, select

        from src.data.database import get_effective_watchlist, get_session
        from src.data.pipeline import sync_broker_trades
        from src.data.schema import DiscoveryRecord

        with get_session() as session:
            latest_date = session.execute(select(func.max(DiscoveryRecord.scan_date))).scalar()
            discover_stocks: list[str] = []
            if latest_date:
                rows = session.execute(
                    select(DiscoveryRecord.stock_id).where(DiscoveryRecord.scan_date == latest_date).distinct()
                ).all()
                discover_stocks = [r[0] for r in rows]

        watchlist = get_effective_watchlist()
        broker_stocks = list(set(watchlist + discover_stocks))

        logger.info("開始同步分點資料（%d 支股票）", len(broker_stocks))
        sync_broker_trades(stock_ids=broker_stocks, days=5)
        logger.info("分點資料同步完成")
    except Exception:
        logger.exception("分點資料同步失敗，繼續後續流程")

    # Step 4: 重算技術指標
    try:
        sync_indicators()
        logger.info("技術指標重算完成")
    except Exception:
        logger.exception("技術指標計算失敗")
        return

    # Step 5: watchlist 多因子篩選
    try:
        from src.config import settings
        from src.notification.line_notify import send_scan_results
        from src.screener.engine import MultiFactorScreener

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
        from src.config import settings
        from src.notification.line_notify import send_message
        from src.report.engine import DailyReportEngine
        from src.report.formatter import format_daily_report

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

    # Step 6: 全市場選股掃描（SBL + 分點資料已就緒）
    try:
        from src.config import settings
        from src.data.pipeline import sync_market_data, sync_stock_info
        from src.discovery.scanner import MarketScanner
        from src.notification.line_notify import send_message
        from src.report.formatter import format_discovery_report

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

    每日任務：23:00 執行（sync → sync-sbl → sync-broker → compute → scan → discover）
    週四任務：23:30 執行（sync-holding，TWSE 大戶資料通常週四更新）
    """
    schedule.every().day.at("23:00").do(daily_sync_job)
    schedule.every().thursday.at("23:30").do(weekly_holding_job)

    print("排程器已啟動")
    print("  每日 23:00  — sync / sync-sbl / sync-broker / compute / scan / discover")
    print("  每週四 23:30 — sync-holding（大戶持股週資料）")
    print("按 Ctrl+C 停止")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n排程器已停止")
