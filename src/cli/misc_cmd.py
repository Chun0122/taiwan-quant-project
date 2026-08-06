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


def cmd_backfill_history(args: argparse.Namespace) -> None:
    """回補歷史全市場資料（B1① PIT 研究環境的資料前提）。

    長時間作業（2020 起約 1,200 個平日、數小時）。可隨時 Ctrl-C 中止，
    重跑會自動從缺口續行——進度以 DB 現況判定，不另存進度檔。
    """
    from datetime import date as _date

    from src.data.pipeline import (
        backfill_daily_features,
        backfill_market_history,
        backfill_valuation_history,
        sync_delisting_info,
    )

    start = _date.fromisoformat(args.start)
    end = _date.fromisoformat(args.end) if getattr(args, "end", None) else None
    datasets = tuple(s.strip() for s in (args.datasets or "price,institutional,margin").split(",") if s.strip())

    if getattr(args, "valuation_only", False):
        markets = tuple(
            s.strip() for s in (getattr(args, "valuation_markets", None) or "twse,tpex").split(",") if s.strip()
        )
        print(f"只回補 stock_valuation（§6.5 #20）：{start} ~ {end or '今日'}　市場={','.join(markets)}")
        print("  上市＝TWSE BWIBBU_d 每日全市場；上櫃＝FinMind 逐股（TPEX 估值端點已下架）")
        if args.dry_run:
            print("[dry-run] 僅估算，不實際抓取\n")
        vr = backfill_valuation_history(start, end, markets=markets, dry_run=args.dry_run)
        if args.dry_run:
            print("dry-run 結束——上方 log 已列出待補量與預估時間")
            return
        print("\n估值回補完成：")
        print(f"  上市 交易日 {vr['twse_days']:>6}　筆數 {vr['twse_rows']:>8}")
        print(f"  上櫃 股票數 {vr['tpex_stocks']:>6}　筆數 {vr['tpex_rows']:>8}")
        print(f"  已跳過      {vr['skipped_days']:>6} 日 / {vr['skipped_stocks']:>5} 檔（DB 已有）")
        if vr.get("quota_exhausted"):
            print("\n⚠ FinMind 配額用盡，上櫃部分未跑完——配額恢復後重跑本指令即可從缺口續行")
        return

    # 先同步下市清單：倖存者偏差修正的前提（知道哪些股票何時下市）
    if not args.skip_delisting:
        print("同步下市清單（倖存者偏差修正）...")
        n = sync_delisting_info()
        print(f"  stock_info 更新 {n} 筆\n")

    if args.features_only:
        print(f"只回補 DailyFeature（B1②）：{start} ~ {end or '今日'}")
        fr = backfill_daily_features(start, end, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"\nDailyFeature 回補完成：{fr['dates']} 日 / {fr['rows']} 筆（已有 {fr['skipped_dates']} 日）")
        return

    print(f"回補範圍：{start} ~ {end or '今日'}　dataset={','.join(datasets)}")
    print("  （已達全市場覆蓋的日期會自動跳過；中斷後重跑會從缺口續行）")
    if args.dry_run:
        print("[dry-run] 僅估算，不實際抓取\n")

    result = backfill_market_history(start, end, datasets=datasets, dry_run=args.dry_run)

    if args.dry_run:
        print("dry-run 結束——上方 log 已列出待補日數與預估時間")
        return
    print("\n回補完成：")
    print(f"  交易日     {result['trading_days']:>6}")
    print(f"  日K線     {result['daily_price']:>6} 筆")
    print(f"  三大法人   {result['institutional']:>6} 筆")
    print(f"  融資融券   {result['margin']:>6} 筆")
    print(f"  已跳過     {result['skipped']:>6} 日（DB 已有 / 週末）")

    if args.with_features:
        print("\n接著回補 DailyFeature（B1②）...")
        fr = backfill_daily_features(start, end)
        print(f"  DailyFeature {fr['dates']} 日 / {fr['rows']} 筆")


def cmd_pit_replay(args: argparse.Namespace) -> None:
    """PIT 歷史重放（B1④）——在歷史日重跑 scanner 並評估前瞻報酬。

    唯讀：不寫入 DiscoveryRecord / CandidateFactorLog / universe_stat_log，
    regime 亦不推進狀態機。
    """
    from datetime import date as _date

    from src.discovery.pit_replay import (
        compute_forward_returns,
        replay_scan,
        sample_replay_dates,
        summarize_replays,
    )

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())

    if args.date:
        dates = [_date.fromisoformat(args.date)]
    else:
        start = _date.fromisoformat(args.start)
        end = _date.fromisoformat(args.end) if args.end else _date.today()
        dates = sample_replay_dates(start, end, args.every)
        if not dates:
            print("指定區間內無具備全市場資料的交易日——請先執行 backfill-history")
            return

    est_min = len(dates) * 90 / 60
    print(f"PIT 重放：mode={args.mode}　{len(dates)} 個基準日　預估 {est_min:.0f} 分鐘")
    print(f"  前瞻窗口：{', '.join(f'{h}d' for h in horizons)}　（唯讀，不寫入任何 live 資料表）\n")
    if args.dry_run:
        print("  " + ", ".join(str(d) for d in dates[:10]) + (" ..." if len(dates) > 10 else ""))
        return

    collected = []
    n_no_data = 0
    n_no_picks = 0
    print(f"{'基準日':<12}{'regime':<10}{'掃描':>6}{'粗篩':>6}{'產出':>6}   前瞻均報酬")
    print("-" * 68)
    for d in dates:
        try:
            res = replay_scan(args.mode, d, top_n=args.top)
        except Exception as exc:  # noqa: BLE001 — 單日失敗不中斷整批
            print(f"{str(d):<12}重放失敗：{exc}")
            continue
        # §6.5 #21b：資料缺席的日子**不計入彙總**——此時的選股（若有）來自退化後的
        # 漏斗，把它平均進去等於用別的東西的報酬去描述這個模式
        if res.verdict == "no_data":
            n_no_data += 1
            note = res.coverage.describe() if res.coverage else "資料缺席"
            print(
                f"{str(d):<12}{res.regime:<10}{res.total_stocks:>6}{res.after_coarse:>6}"
                f"{res.n_picks:>6}   ⚠ {note}（不計入）"
            )
            continue
        if res.picks.empty:
            n_no_picks += 1
            print(f"{str(d):<12}{res.regime:<10}{res.total_stocks:>6}{res.after_coarse:>6}{0:>6}   （無選股）")
            continue
        withfwd = compute_forward_returns(res.picks, d, horizons)
        collected.append(withfwd)
        summary = "  ".join(
            f"{h}d={withfwd[f'fwd_{h}d'].mean():+.2f}%" if withfwd[f"fwd_{h}d"].notna().any() else f"{h}d=—"
            for h in horizons
        )
        print(f"{str(d):<12}{res.regime:<10}{res.total_stocks:>6}{res.after_coarse:>6}{res.n_picks:>6}   {summary}")

    n_valid = len(collected) + n_no_picks
    print(f"\n基準日分類：可採信 {n_valid}（有選股 {len(collected)}／無選股 {n_no_picks}）　資料缺席 {n_no_data}")
    if n_no_data:
        print(f"  ⚠ {n_no_data} 個基準日的輸入資料不足，已排除——產能率與報酬皆以可採信的 {n_valid} 日為母體")

    if not collected:
        print("\n無任何選股結果可彙總")
        return

    print(f"\n{'=' * 68}")
    print(f"彙總（{args.mode}，{len(collected)} 個有效基準日；產能率 {len(collected) / max(n_valid, 1):.0%}）")
    print(f"{'=' * 68}")
    summary_df = summarize_replays(collected, horizons)
    print(f"{'窗口':<8}{'樣本':>7}{'平均':>9}{'中位':>9}{'勝率':>8}{'最佳':>9}{'最差':>9}")
    print("-" * 60)
    for _, r in summary_df.iterrows():
        print(
            f"{r['horizon']:<8}{int(r['n']):>7}{r['avg_return']:>8.2f}%{r['median']:>8.2f}%"
            f"{r['win_rate']:>7.1%}{r['best']:>8.2f}%{r['worst']:>8.2f}%"
        )

    if args.export:
        import pandas as _pd

        _pd.concat(collected, ignore_index=True).to_csv(args.export, index=False)
        print(f"\n明細已匯出：{args.export}")


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
