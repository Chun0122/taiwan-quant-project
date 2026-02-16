"""台股量化投資系統 — 主程式入口。

Usage:
    # 同步預設關注清單
    python main.py sync

    # 同步指定股票
    python main.py sync --stocks 2330 2317

    # 指定日期範圍
    python main.py sync --start 2023-01-01 --end 2024-12-31

    # 計算技術指標
    python main.py compute

    # 執行回測
    python main.py backtest --stock 2330 --strategy sma_cross

    # 查詢已入庫的資料概況
    python main.py status
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import settings


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.logging.level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_sync(args: argparse.Namespace) -> None:
    """執行資料同步。"""
    from src.data.pipeline import sync_watchlist

    stocks = args.stocks if args.stocks else None
    results = sync_watchlist(
        watchlist=stocks, start_date=args.start, end_date=args.end
    )

    print("\n" + "=" * 60)
    print("同步結果摘要")
    print("=" * 60)
    for stock_id, counts in results.items():
        if "error" in counts:
            print(f"  {stock_id}: 失敗")
        else:
            print(
                f"  {stock_id}: 日K={counts['daily_price']} 筆, "
                f"法人={counts['institutional']} 筆, "
                f"融資融券={counts['margin']} 筆"
            )


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


def cmd_backtest(args: argparse.Namespace) -> None:
    """執行回測。"""
    from datetime import date

    from src.data.database import init_db
    from src.data.pipeline import save_backtest_result
    from src.strategy import STRATEGY_REGISTRY
    from src.backtest.engine import BacktestEngine

    if args.strategy not in STRATEGY_REGISTRY:
        print(f"未知策略: {args.strategy}")
        print(f"可用策略: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)

    init_db()

    start = args.start or settings.fetcher.default_start_date
    end = args.end or date.today().isoformat()

    strategy_cls = STRATEGY_REGISTRY[args.strategy]
    strategy = strategy_cls(stock_id=args.stock, start_date=start, end_date=end)

    engine = BacktestEngine(strategy)
    result = engine.run()

    # 存入 DB
    bt_id = save_backtest_result(result)

    # 印出摘要
    print("\n" + "=" * 60)
    print(f"回測結果 — {result.strategy_name} | {result.stock_id}")
    print("=" * 60)
    print(f"  期間:         {result.start_date} ~ {result.end_date}")
    print(f"  初始資金:     {result.initial_capital:>14,.0f}")
    print(f"  最終資金:     {result.final_capital:>14,.2f}")
    print(f"  總報酬率:     {result.total_return:>13.2f}%")
    print(f"  年化報酬率:   {result.annual_return:>13.2f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio or 'N/A':>13}")
    print(f"  最大回撤:     {result.max_drawdown:>13.2f}%")
    print(f"  勝率:         {result.win_rate or 'N/A':>13}%")
    print(f"  交易次數:     {result.total_trades:>13}")
    print(f"  (結果已儲存, id={bt_id})")


def cmd_status(args: argparse.Namespace) -> None:
    """顯示資料庫概況。"""
    from sqlalchemy import func, select

    from src.data.database import get_session, init_db
    from src.data.schema import (
        DailyPrice, InstitutionalInvestor, MarginTrading, TechnicalIndicator,
        BacktestResult,
    )

    init_db()

    with get_session() as session:
        for model, label in [
            (DailyPrice, "日K線"),
            (InstitutionalInvestor, "三大法人"),
            (MarginTrading, "融資融券"),
            (TechnicalIndicator, "技術指標"),
        ]:
            total = session.execute(
                select(func.count()).select_from(model)
            ).scalar()
            stocks = session.execute(
                select(func.count(func.distinct(model.stock_id)))
            ).scalar()
            min_date = session.execute(
                select(func.min(model.date))
            ).scalar()
            max_date = session.execute(
                select(func.max(model.date))
            ).scalar()

            print(f"[{label}] {total:,} 筆 | {stocks} 檔股票 | {min_date} ~ {max_date}")

        # 額外顯示各指標名稱的筆數
        indicator_counts = session.execute(
            select(TechnicalIndicator.name, func.count())
            .group_by(TechnicalIndicator.name)
            .order_by(TechnicalIndicator.name)
        ).all()
        if indicator_counts:
            print("\n  指標明細:")
            for name, cnt in indicator_counts:
                print(f"    {name:15s} {cnt:>8,} 筆")

        # 回測結果摘要
        bt_count = session.execute(
            select(func.count()).select_from(BacktestResult)
        ).scalar()
        if bt_count:
            print(f"\n[回測紀錄] {bt_count} 筆")
            rows = session.execute(
                select(BacktestResult).order_by(BacktestResult.id.desc()).limit(5)
            ).scalars().all()
            for r in rows:
                print(
                    f"  #{r.id} {r.stock_id} {r.strategy_name} | "
                    f"報酬={r.total_return:+.2f}% | MDD={r.max_drawdown:.2f}% | "
                    f"交易={r.total_trades}次"
                )


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="台股量化投資系統")
    subparsers = parser.add_subparsers(dest="command")

    # sync 子命令
    sp_sync = subparsers.add_parser("sync", help="同步股票資料")
    sp_sync.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_sync.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_sync.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")

    # compute 子命令
    sp_compute = subparsers.add_parser("compute", help="計算技術指標")
    sp_compute.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")

    # backtest 子命令
    sp_bt = subparsers.add_parser("backtest", help="執行回測")
    sp_bt.add_argument("--stock", required=True, help="股票代號")
    sp_bt.add_argument("--strategy", required=True, help="策略名稱 (sma_cross, rsi_threshold)")
    sp_bt.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_bt.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")

    # status 子命令
    subparsers.add_parser("status", help="顯示資料庫概況")

    args = parser.parse_args()

    if args.command == "sync":
        cmd_sync(args)
    elif args.command == "compute":
        cmd_compute(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
