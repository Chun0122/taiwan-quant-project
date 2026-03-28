"""CLI 資料同步子命令 — sync / compute / sync-mops / sync-revenue / sync-financial /
sync-info / sync-features / sync-holding / sync-vix / sync-sbl / sync-broker。"""

from __future__ import annotations

import argparse

from src.cli.helpers import init_db, read_stocks_from_file  # noqa: A004
from src.cli.helpers import safe_print as print


def cmd_sync(args: argparse.Namespace) -> None:
    """執行資料同步。"""
    from src.data.pipeline import sync_taiex_index, sync_watchlist

    # 同步 TAIEX 指數（預設啟用）
    taiex_count = sync_taiex_index(start_date=args.start, end_date=args.end)
    print(f"\n  TAIEX 加權指數: {taiex_count} 筆")

    stocks = args.stocks if args.stocks else None
    results = sync_watchlist(watchlist=stocks, start_date=args.start, end_date=args.end)

    print("\n" + "=" * 60)
    print("同步結果摘要")
    print("=" * 60)
    for stock_id, counts in results.items():
        if "error" in counts:
            print(f"  {stock_id}: 失敗")
        else:
            parts = [
                f"日K={counts['daily_price']}",
                f"法人={counts['institutional']}",
                f"融資融券={counts['margin']}",
                f"營收={counts.get('revenue', 0)}",
                f"股利={counts.get('dividend', 0)}",
                f"財報={counts.get('financial', 0)}",
            ]
            print(f"  {stock_id}: {', '.join(parts)}")


def cmd_compute(args: argparse.Namespace) -> None:
    """計算技術指標。"""
    from src.data.pipeline import sync_indicators

    stocks = args.stocks if args.stocks else None
    results = sync_indicators(watchlist=stocks)

    print("\n" + "=" * 60)
    print("指標計算結果摘要")
    print("=" * 60)
    for stock_id, count in results.items():
        if count < 0:
            print(f"  {stock_id}: 失敗")
        else:
            print(f"  {stock_id}: {count:,} 筆指標")


def cmd_sync_mops(args: argparse.Namespace) -> None:
    """同步 MOPS 重大訊息公告。"""
    from src.data.pipeline import sync_mops_announcements

    print("同步 MOPS 最新重大訊息...")
    count = sync_mops_announcements()
    print(f"\nMOPS 公告同步完成: {count:,} 筆")

    # 顯示情緒分布統計
    if count > 0:
        from sqlalchemy import func, select

        from src.data.database import get_session
        from src.data.schema import Announcement

        with get_session() as session:
            dist = session.execute(select(Announcement.sentiment, func.count()).group_by(Announcement.sentiment)).all()

        sentiment_labels = {1: "正面", 0: "中性", -1: "負面"}
        print("\n情緒分布:")
        for sentiment, cnt in sorted(dist, key=lambda x: x[0], reverse=True):
            label = sentiment_labels.get(sentiment, str(sentiment))
            print(f"  {label}: {cnt:,} 筆")


def cmd_sync_revenue(args: argparse.Namespace) -> None:
    """從 MOPS 同步全市場月營收。"""
    from src.data.pipeline import sync_mops_revenue

    months = getattr(args, "months", 1)
    print(f"從 MOPS 同步全市場月營收（最近 {months} 個月）...")
    count = sync_mops_revenue(months=months)
    print(f"\n月營收同步完成: {count:,} 筆")

    if count > 0:
        from sqlalchemy import func, select

        from src.data.database import get_session
        from src.data.schema import MonthlyRevenue

        with get_session() as session:
            total = session.execute(select(func.count()).select_from(MonthlyRevenue)).scalar()
            distinct_stocks = session.execute(select(func.count(func.distinct(MonthlyRevenue.stock_id)))).scalar()

        print(f"月營收資料庫: {total:,} 筆（{distinct_stocks:,} 支股票）")


def cmd_sync_financial(args: argparse.Namespace) -> None:
    """同步財報資料（季報損益表 + 資產負債表 + 現金流量表）。"""
    from src.data.pipeline import sync_financial_statements

    stocks = args.stocks if args.stocks else None
    quarters = getattr(args, "quarters", 4)
    print(f"同步財報資料（最近 {quarters} 季）...")
    count = sync_financial_statements(watchlist=stocks, quarters=quarters)
    print(f"\n財報同步完成: {count:,} 筆")

    if count > 0:
        from sqlalchemy import func, select

        from src.data.database import get_session
        from src.data.schema import FinancialStatement

        with get_session() as session:
            total = session.execute(select(func.count()).select_from(FinancialStatement)).scalar()
            distinct_stocks = session.execute(select(func.count(func.distinct(FinancialStatement.stock_id)))).scalar()
            max_date = session.execute(select(func.max(FinancialStatement.date))).scalar()

        print(f"財報資料庫: {total:,} 筆（{distinct_stocks:,} 支股票，最新至 {max_date}）")


def cmd_sync_info(args: argparse.Namespace) -> None:
    """同步全市場股票基本資料（產業分類、上市/上櫃別）到 stock_info 表。"""
    from src.data.pipeline import sync_stock_info

    init_db()
    force = getattr(args, "force", False)
    print("正在同步全市場股票基本資料（StockInfo）...")
    count = sync_stock_info(force_refresh=force)
    if count > 0:
        from sqlalchemy import func, select

        from src.data.database import get_session
        from src.data.schema import StockInfo

        with get_session() as session:
            total = session.execute(select(func.count()).select_from(StockInfo)).scalar()
            max_updated = session.execute(select(func.max(StockInfo.updated_at))).scalar()

        print(f"\nStockInfo 同步完成: 本次更新 {count:,} 筆")
        print(f"資料庫合計: {total:,} 支股票，最後更新 {max_updated}")
    else:
        print("\nStockInfo 無需更新（DB 已有資料，使用 --force 強制重新同步）")


def cmd_sync_features(args: argparse.Namespace) -> None:
    """計算並寫入全市場 DailyFeature（Feature Store），供 UniverseFilter Stage 2/3 使用。"""
    from src.data.pipeline import compute_and_store_daily_features

    init_db()
    days = getattr(args, "days", 90)
    print(f"正在計算全市場 DailyFeature（回溯 {days} 天）...")
    count = compute_and_store_daily_features(lookback_days=days)
    print(f"DailyFeature 寫入完成: {count:,} 筆")


def cmd_sync_holding(args: argparse.Namespace) -> None:
    """同步 watchlist 大戶持股分級資料（週資料，FinMind TaiwanStockHoldingSharesPer）。"""
    from src.data.pipeline import sync_holding_distribution

    stocks = args.stocks if args.stocks else None
    weeks = getattr(args, "weeks", 4)
    print(f"同步大戶持股分級資料（最近 {weeks} 週）...")
    count = sync_holding_distribution(watchlist=stocks, weeks=weeks)
    print(f"\n持股分級同步完成: {count:,} 筆")

    if count > 0:
        from sqlalchemy import func, select

        from src.data.database import get_session
        from src.data.schema import HoldingDistribution

        with get_session() as session:
            total = session.execute(select(func.count()).select_from(HoldingDistribution)).scalar()
            distinct_stocks = session.execute(select(func.count(func.distinct(HoldingDistribution.stock_id)))).scalar()
            max_date = session.execute(select(func.max(HoldingDistribution.date))).scalar()

        print(f"持股分級資料庫: {total:,} 筆（{distinct_stocks:,} 支股票，最新至 {max_date}）")


def cmd_sync_vix(args: argparse.Namespace) -> None:
    """同步 VIX 波動率指數（台灣 + 美國）。"""
    from src.data.pipeline import sync_taiwan_vix, sync_us_vix

    print("同步台灣 VIX 波動率指數...")
    try:
        tw_count = sync_taiwan_vix()
        print(f"台灣 VIX 同步完成：{tw_count:,} 筆")
    except (ConnectionError, OSError, ValueError, KeyError) as e:
        print(f"台灣 VIX 同步失敗（不影響流程）: {e}")
        tw_count = 0

    print("同步美國 VIX (CBOE ^VIX)...")
    try:
        us_count = sync_us_vix()
        print(f"美國 VIX 同步完成：{us_count:,} 筆")
    except (ConnectionError, OSError, ValueError, KeyError) as e:
        print(f"美國 VIX 同步失敗（不影響流程）: {e}")
        us_count = 0

    print(f"\nVIX 同步總計：台灣 {tw_count:,} 筆 + 美國 {us_count:,} 筆")


def cmd_sync_sbl(args: argparse.Namespace) -> None:
    """同步 TWSE 全市場借券賣出彙總資料（TWT96U）。"""
    from src.data.pipeline import sync_sbl_all_market

    days = getattr(args, "days", 3)
    print(f"同步全市場借券賣出資料（最近 {days} 個交易日）...")
    count = sync_sbl_all_market(days=days)
    print(f"\n借券資料同步完成: {count:,} 筆")

    if count > 0:
        from sqlalchemy import func, select

        from src.data.database import get_session
        from src.data.schema import SecuritiesLending

        with get_session() as session:
            total = session.execute(select(func.count()).select_from(SecuritiesLending)).scalar()
            distinct_stocks = session.execute(select(func.count(func.distinct(SecuritiesLending.stock_id)))).scalar()
            max_date = session.execute(select(func.max(SecuritiesLending.date))).scalar()

        print(f"借券資料庫: {total:,} 筆（{distinct_stocks:,} 支股票，最新至 {max_date}）")


def cmd_sync_broker(args: argparse.Namespace) -> None:
    """同步分點交易資料（FinMind TaiwanStockTradingDailyReport）。

    --watchlist-bootstrap：一次性對所有 watchlist 股票逐日補齊歷史分點資料。
      DJ 端點每次呼叫僅回傳期間彙整（date=end），因此改為逐日查詢（start=d, end=d），
      使每個交易日產生獨立記錄，達到 Smart Broker 8F 所需的 min_trading_days=20。
      預設補齊最近 120 個交易日（半年），建議首次部署或新增 watchlist 股票後執行。
    --from-file：從文字/CSV 檔案讀取股票代號清單（可與 --watchlist-bootstrap 合用）。
    """
    from src.data.pipeline import sync_broker_bootstrap, sync_broker_trades

    stock_ids = args.stocks if args.stocks else None
    days = getattr(args, "days", 5)

    # --from-file：從外部檔案讀取股票清單，合併 --stocks
    from_file_path = getattr(args, "from_file", None)
    if from_file_path:
        file_stocks = read_stocks_from_file(from_file_path)
        stock_ids = list(dict.fromkeys((stock_ids or []) + file_stocks))
        print(f"從檔案 {from_file_path} 讀入 {len(file_stocks)} 支股票")

    # --watchlist-bootstrap：逐日補齊 watchlist 股票的分點歷史（啟用 8F）
    if getattr(args, "watchlist_bootstrap", False):
        from src.data.database import get_effective_watchlist

        # 若使用者未明確指定 --days（預設為 5），bootstrap 改用 120 天
        bootstrap_days = days if days != 5 else 120
        watchlist = get_effective_watchlist()
        target_ids = stock_ids if stock_ids else watchlist
        print(f"[Bootstrap] 對 {len(target_ids)} 支股票逐日補齊最近 {bootstrap_days} 個交易日分點歷史...")
        print(
            f"  預估時間：{len(target_ids)} 支 × {bootstrap_days} 天 × 3s ≈ {len(target_ids) * bootstrap_days * 3 // 60} 分鐘"
        )
        print("  （已存在的日期自動跳過，可中斷後重跑）")
        count = sync_broker_bootstrap(stock_ids=target_ids, days=bootstrap_days)
        print(f"\n分點 Bootstrap 完成: {count:,} 筆（{bootstrap_days} 個交易日）")
        print("  若筆數 > 0，watchlist 股票已可使用 Smart Broker 8F 評分")
        return

    if getattr(args, "from_discover", False):
        from sqlalchemy import func, select

        from src.data.database import get_session
        from src.data.schema import DiscoveryRecord

        with get_session() as session:
            latest_date = session.execute(select(func.max(DiscoveryRecord.scan_date))).scalar()
            if latest_date:
                rows = session.execute(
                    select(DiscoveryRecord.stock_id).where(DiscoveryRecord.scan_date == latest_date).distinct()
                ).all()
                discover_stocks = [r[0] for r in rows]
                from src.data.database import get_effective_watchlist

                watchlist = get_effective_watchlist()
                # 排除已在 watchlist 中的股票（morning-routine 2a 已同步）
                non_watchlist = [s for s in discover_stocks if s not in watchlist]
                stock_ids = list(set((stock_ids or []) + non_watchlist))
                if stock_ids:
                    print(f"從最近 discover（{latest_date}）補抓非 watchlist 股票 {len(stock_ids)} 支")
                else:
                    print(f"從最近 discover（{latest_date}）無需補抓（全在 watchlist 中）")
                    return

    print(f"同步分點交易資料（最近 {days} 日）...")
    count = sync_broker_trades(stock_ids=stock_ids, days=days)
    print(f"\n分點資料同步完成: {count:,} 筆")


def cmd_alert_check(args: argparse.Namespace) -> None:
    """掃描近期 MOPS 重大事件警報（法說會、財報、高關注公告）。

    查詢 Announcement 表，篩選指定天數內的非一般性事件，
    以事件類型分組顯示，並可選擇推播 Discord。
    """
    from sqlalchemy import and_, select

    from src.data.database import get_session
    from src.data.database import init_db as db_init

    init_db()
    db_init()

    days = getattr(args, "days", 7)
    event_types = getattr(args, "types", None)  # None = 顯示全部非 general
    stocks = getattr(args, "stocks", None)
    notify = getattr(args, "notify", False)

    from datetime import date, timedelta

    from src.data.schema import Announcement

    since = date.today() - timedelta(days=days)

    with get_session() as session:
        conditions = [Announcement.date >= since]
        if event_types:
            conditions.append(Announcement.event_type.in_(event_types))
        else:
            conditions.append(Announcement.event_type != "general")
        if stocks:
            conditions.append(Announcement.stock_id.in_(stocks))

        rows = session.execute(
            select(
                Announcement.date,
                Announcement.stock_id,
                Announcement.event_type,
                Announcement.sentiment,
                Announcement.subject,
            )
            .where(and_(*conditions))
            .order_by(Announcement.date.desc(), Announcement.event_type)
        ).all()

    if not rows:
        print(f"最近 {days} 天內無特殊事件公告")
        return

    # 分類顯示
    _EVENT_LABELS = {
        "earnings_call": "[法說] 法說會",
        "investor_day": "[投日] 投資人日",
        "filing": "[財報] 財報發布",
        "revenue": "[營收] 營收公告",
        "governance_change": "[治理] 董監改選",
        "buyback": "[庫藏] 庫藏股",
        "general": "[公告] 一般公告",
    }
    _SENTIMENT_LABELS = {1: "▲", 0: "─", -1: "▼"}

    print(f"\n=== MOPS 重大事件警報（近 {days} 天）共 {len(rows)} 筆 ===\n")
    current_type = None
    lines = []
    for row in rows:
        if row.event_type != current_type:
            current_type = row.event_type
            label = _EVENT_LABELS.get(current_type, current_type)
            print(f"【{label}】")
            lines.append(f"【{label}】")
        sentiment_sym = _SENTIMENT_LABELS.get(row.sentiment, "─")
        line = f"  {row.date} [{row.stock_id}] {sentiment_sym} {row.subject[:60]}"
        print(line)
        lines.append(line)

    if notify:
        from src.notification.line_notify import send_message

        msg = f"📡 MOPS 事件警報（近 {days} 天，共 {len(rows)} 筆）\n" + "\n".join(lines[:40])
        ok = send_message(msg)
        print(f"\nDiscord 通知: {'成功' if ok else '失敗（請確認 Webhook 設定）'}")
