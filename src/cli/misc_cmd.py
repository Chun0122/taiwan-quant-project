"""CLI 雜項子命令 — dashboard / optimize / schedule / status / scan / notify /
report / strategy-rank / industry / migrate / validate / export / import-data。"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.cli.helpers import init_db
from src.cli.helpers import safe_print as print
from src.config import settings


def cmd_dashboard() -> None:
    """啟動 Streamlit 儀表板。"""
    import subprocess

    from src.config import PROJECT_ROOT

    app_path = PROJECT_ROOT / "src" / "visualization" / "app.py"
    print("啟動儀表板: http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], cwd=str(PROJECT_ROOT))


def cmd_optimize(args: argparse.Namespace) -> None:
    """執行參數優化（Grid Search）。"""
    from datetime import date

    from src.optimization.grid_search import GridSearchOptimizer

    init_db()

    start = args.start or settings.fetcher.default_start_date
    end = args.end or date.today().isoformat()

    optimizer = GridSearchOptimizer(
        strategy_name=args.strategy,
        stock_id=args.stock,
        start_date=start,
        end_date=end,
    )

    results = optimizer.run()
    optimizer.print_top_n(results, n=args.top_n)

    if args.export:
        optimizer.export_to_csv(results, args.export)


def _resolve_schedule_mode(mode: str) -> str:
    """解析排程模式：auto 時依平台自動選擇。"""
    if mode != "auto":
        return mode
    import sys as _sys

    return "macos" if _sys.platform == "darwin" else "windows"


def cmd_schedule(args: argparse.Namespace) -> None:
    """設定排程任務。"""
    mode = _resolve_schedule_mode(args.mode)

    if mode == "simple":
        from src.scheduler.simple_scheduler import run_scheduler

        run_scheduler()
    elif mode == "macos":
        from src.scheduler.launchd_task import generate_scripts

        generate_scripts()
    elif mode == "windows":
        from src.scheduler.windows_task import generate_scripts

        generate_scripts()


def cmd_status(args: argparse.Namespace) -> None:
    """顯示資料庫概況。"""
    from sqlalchemy import func, select

    from src.data.database import get_session
    from src.data.schema import (
        BacktestResult,
        DailyPrice,
        Dividend,
        FinancialStatement,
        InstitutionalInvestor,
        MarginTrading,
        MonthlyRevenue,
        PortfolioBacktestResult,
        TechnicalIndicator,
    )

    init_db()

    with get_session() as session:
        for model, label in [
            (DailyPrice, "日K線"),
            (InstitutionalInvestor, "三大法人"),
            (MarginTrading, "融資融券"),
            (MonthlyRevenue, "月營收"),
            (Dividend, "股利"),
            (FinancialStatement, "財報"),
            (TechnicalIndicator, "技術指標"),
        ]:
            total = session.execute(select(func.count()).select_from(model)).scalar()
            stocks = session.execute(select(func.count(func.distinct(model.stock_id)))).scalar()
            min_date = session.execute(select(func.min(model.date))).scalar()
            max_date = session.execute(select(func.max(model.date))).scalar()

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
        bt_count = session.execute(select(func.count()).select_from(BacktestResult)).scalar()
        if bt_count:
            print(f"\n[回測紀錄] {bt_count} 筆")
            rows = session.execute(select(BacktestResult).order_by(BacktestResult.id.desc()).limit(5)).scalars().all()
            for r in rows:
                print(
                    f"  #{r.id} {r.stock_id} {r.strategy_name} | "
                    f"報酬={r.total_return:+.2f}% | MDD={r.max_drawdown:.2f}% | "
                    f"交易={r.total_trades}次"
                )

        # 投資組合回測摘要
        pbt_count = session.execute(select(func.count()).select_from(PortfolioBacktestResult)).scalar()
        if pbt_count:
            print(f"\n[投資組合回測] {pbt_count} 筆")
            rows = (
                session.execute(select(PortfolioBacktestResult).order_by(PortfolioBacktestResult.id.desc()).limit(5))
                .scalars()
                .all()
            )
            for r in rows:
                print(
                    f"  #{r.id} [{r.stock_ids}] {r.strategy_name} | "
                    f"報酬={r.total_return:+.2f}% | MDD={r.max_drawdown:.2f}% | "
                    f"交易={r.total_trades}次"
                )


def cmd_scan(args: argparse.Namespace) -> None:
    """執行多因子選股篩選。"""

    from src.screener.engine import MultiFactorScreener

    init_db()

    stocks = args.stocks if args.stocks else None
    screener = MultiFactorScreener(watchlist=stocks, lookback_days=args.lookback)

    print("正在掃描股票...")
    if args.conditions:
        results = screener.scan_with_conditions(args.conditions, require_all=True)
    else:
        results = screener.scan()

    if results.empty:
        print("無符合條件的股票")
        return

    # 顯示結果
    print(f"\n{'=' * 70}")
    print(f"篩選結果 — 共 {len(results)} 檔")
    print(f"{'=' * 70}")

    display_cols = ["stock_id", "close", "factor_score"]
    optional = ["rsi_14", "foreign_net", "yoy_growth"]
    for col in optional:
        if col in results.columns:
            display_cols.append(col)

    print(results[[c for c in display_cols if c in results.columns]].to_string(index=False))

    # 匯出 CSV
    if args.export:
        results.to_csv(args.export, index=False)
        print(f"\n結果已匯出至: {args.export}")

    # 發送 Discord 通知
    if args.notify:
        from src.notification.line_notify import send_scan_results

        ok = send_scan_results(results)
        if ok:
            print("Discord 通知已發送")
        else:
            print("Discord 通知發送失敗（請確認 webhook_url 設定）")


def cmd_notify(args: argparse.Namespace) -> None:
    """發送 Discord Webhook 測試訊息。"""
    from src.notification.line_notify import send_message

    ok = send_message(args.message)
    if ok:
        print("Discord 通知發送成功")
    else:
        print("Discord 通知發送失敗（請確認 config/settings.yaml 的 discord.webhook_url 設定）")


def cmd_report(args: argparse.Namespace) -> None:
    """執行每日選股報告。"""

    from src.report.engine import DailyReportEngine

    init_db()

    stocks = args.stocks if args.stocks else None
    engine = DailyReportEngine(
        watchlist=stocks,
        lookback_days=5,
        ml_enabled=not args.no_ml,
    )

    print("正在計算四維度評分...")
    df = engine.run()

    if df.empty:
        print("無資料可生成報告")
        return

    # 顯示結果
    display = df.head(args.top)
    print(f"\n{'=' * 75}")
    print(f"每日選股報告 — 前 {min(args.top, len(df))} 名（共 {len(df)} 檔）")
    print(f"{'=' * 75}")
    print(
        f"{'#':>3}  {'代號':>6}  {'收盤':>8}  {'綜合':>6}  {'技術':>6}  {'籌碼':>6}  "
        f"{'基本':>6}  {'ML':>6}  {'RSI':>5}  {'外資':>10}  {'YoY':>7}"
    )
    print(f"{'─' * 75}")

    for _, row in display.iterrows():
        rsi = f"{row['rsi']:.0f}" if pd.notna(row.get("rsi")) else "N/A"
        foreign = f"{row['foreign_net']:>10,.0f}" if pd.notna(row.get("foreign_net")) else "       N/A"
        yoy = f"{row['yoy_growth']:.1f}%" if pd.notna(row.get("yoy_growth")) else "   N/A"
        print(
            f"{int(row['rank']):>3}  {row['stock_id']:>6}  {row['close']:>8.1f}  "
            f"{row['composite_score']:>6.3f}  {row['technical_score']:>6.3f}  "
            f"{row['chip_score']:>6.3f}  {row['fundamental_score']:>6.3f}  "
            f"{row['ml_score']:>6.3f}  {rsi:>5}  {foreign}  {yoy:>7}"
        )

    if args.export:
        df.to_csv(args.export, index=False)
        print(f"\n結果已匯出至: {args.export}")

    if args.notify:
        from src.notification.line_notify import send_message
        from src.report.formatter import format_daily_report

        msgs = format_daily_report(df, top_n=args.top)
        for msg in msgs:
            send_message(msg)
        print("Discord 通知已發送")


def cmd_strategy_rank(args: argparse.Namespace) -> None:
    """執行策略回測排名。"""

    from src.strategy_rank.engine import StrategyRankEngine

    init_db()

    stocks = args.stocks if args.stocks else None
    strategies = args.strategies if args.strategies else None

    engine = StrategyRankEngine(
        watchlist=stocks,
        strategy_names=strategies,
        metric=args.metric,
        start_date=args.start,
        end_date=args.end,
        min_trades=args.min_trades,
    )

    print("正在執行批次回測...")
    df = engine.run()
    engine.print_summary(df, top_n=20)

    if args.export and not df.empty:
        df.to_csv(args.export, index=False)
        print(f"\n結果已匯出至: {args.export}")

    if args.notify and not df.empty:
        from src.notification.line_notify import send_message
        from src.report.formatter import format_strategy_rank

        msg = format_strategy_rank(df, metric=args.metric)
        send_message(msg)
        print("Discord 通知已發送")


def cmd_industry(args: argparse.Namespace) -> None:
    """執行產業輪動分析。"""

    from src.data.pipeline import sync_stock_info
    from src.industry.analyzer import IndustryRotationAnalyzer

    init_db()

    # 同步 StockInfo
    if args.refresh:
        print("正在同步股票基本資料...")
        count = sync_stock_info(force_refresh=True)
        print(f"已同步 {count} 筆")
    else:
        sync_stock_info(force_refresh=False)

    stocks = args.stocks if args.stocks else None
    analyzer = IndustryRotationAnalyzer(
        watchlist=stocks,
        lookback_days=args.lookback,
        momentum_days=args.momentum,
    )

    print("正在分析產業輪動...")
    sector_df = analyzer.rank_sectors()

    if sector_df.empty:
        print("無法計算產業排名（資料不足）")
        return

    # 顯示產業排名
    display = sector_df.head(args.top_sectors)
    print(f"\n{'=' * 70}")
    print(f"產業輪動分析 — 前 {min(args.top_sectors, len(sector_df))} 名產業")
    print(f"{'=' * 70}")
    print(f"{'#':>3}  {'產業':<14}  {'綜合':>6}  {'法人':>6}  {'動能':>6}  {'淨買超':>14}  {'漲幅':>8}")
    print(f"{'─' * 70}")

    for _, row in display.iterrows():
        total_net = row.get("total_net", 0)
        avg_ret = row.get("avg_return_pct", 0)
        print(
            f"{int(row['rank']):>3}  {str(row['industry']):<14}  "
            f"{row['sector_score']:>6.3f}  "
            f"{row['institutional_score']:>6.3f}  "
            f"{row['momentum_score']:>6.3f}  "
            f"{total_net:>14,.0f}  {avg_ret:>7.2f}%"
        )

    # 精選個股
    top_stocks = analyzer.top_stocks_from_hot_sectors(sector_df, top_sectors=args.top_sectors, top_n=args.top)
    if not top_stocks.empty:
        print(f"\n{'─' * 70}")
        print("熱門產業精選個股")
        print(f"{'─' * 70}")
        for ind in top_stocks["industry"].unique():
            sector_stocks = top_stocks[top_stocks["industry"] == ind]
            print(f"\n  [{ind}]")
            for _, sr in sector_stocks.iterrows():
                name = sr.get("stock_name", "")
                foreign = sr.get("foreign_net_sum", 0)
                print(f"    {sr['stock_id']} {name:<8}  收盤={sr['close']:>8.1f}  外資淨買超={foreign:>12,.0f}")

    if args.notify:
        from src.notification.line_notify import send_message
        from src.report.formatter import format_industry_report

        msgs = format_industry_report(sector_df, top_stocks, top_n=args.top_sectors)
        for msg in msgs:
            send_message(msg)
        print("\nDiscord 通知已發送")


def cmd_migrate(args: argparse.Namespace) -> None:
    """執行 DB schema 遷移。"""
    from src.data.migrate import run_migrations

    added = run_migrations()
    if added:
        print(f"遷移完成，新增 {len(added)} 個欄位:")
        for col in added:
            print(f"  + {col}")
    else:
        print("資料庫已是最新，無需遷移")


def cmd_validate(args: argparse.Namespace) -> None:
    """執行資料品質檢查。"""
    from src.data.validator import export_issues_csv, print_validation_report, run_validation

    stocks = args.stocks if args.stocks else None
    report = run_validation(
        stock_ids=stocks,
        gap_threshold=args.gap_threshold,
        streak_threshold=args.streak_threshold,
        check_freshness=not args.no_freshness,
    )

    print_validation_report(report)

    if args.export:
        export_issues_csv(report, args.export)


def cmd_export(args: argparse.Namespace) -> None:
    """匯出資料表為 CSV/Parquet。"""

    from src.data.io import TABLE_REGISTRY, export_table, list_tables

    init_db()

    # --list 模式：列出所有表及筆數
    if args.list:
        tables = list_tables()
        print("可匯出的資料表：")
        print(f"{'資料表':<30} {'筆數':>10}")
        print("-" * 42)
        for t in tables:
            print(f"{t['table']:<30} {t['count']:>10,}")
        return

    if not args.table:
        print("錯誤：請指定資料表名稱，或使用 --list 查看所有表")
        print(f"可用資料表: {', '.join(TABLE_REGISTRY.keys())}")
        return

    count = export_table(
        table_name=args.table,
        output_path=args.output,
        fmt=args.format,
        stocks=args.stocks,
        start_date=args.start,
        end_date=args.end,
    )

    if count == 0:
        print("無資料可匯出（表為空或篩選條件無符合資料）")
    else:
        output = args.output or f"data/export/{args.table}.{args.format}"
        print(f"結果已匯出至: {output}（共 {count:,} 筆）")


def cmd_import_data(args: argparse.Namespace) -> None:
    """從 CSV/Parquet 匯入資料。"""

    from src.data.io import import_table

    init_db()

    try:
        count = import_table(
            table_name=args.table,
            source_path=args.source,
            dry_run=args.dry_run,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"錯誤：{e}")
        return

    if args.dry_run:
        print(f"驗證通過：{count:,} 筆資料（dry-run 模式，未寫入）")
    elif count == 0:
        print("無資料可匯入（檔案為空）")
    else:
        print(f"匯入完成：{count:,} 筆 -> {args.table}（重複資料自動略過）")
