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

    # 加停損停利
    python main.py backtest --stock 2330 --strategy sma_cross --stop-loss 5 --take-profit 15

    # 固定比例部位
    python main.py backtest --stock 2330 --strategy rsi_threshold --sizing fixed_fraction --fraction 0.3

    # 投資組合回測
    python main.py backtest --stocks 2330 2317 2454 --strategy sma_cross --stop-loss 5

    # 多因子策略回測
    python main.py backtest --stock 2330 --strategy multi_factor

    # 啟動視覺化儀表板
    python main.py dashboard

    # 參數優化
    python main.py optimize --stock 2330 --strategy sma_cross

    # 設定排程
    python main.py schedule --mode windows

    # 查詢已入庫的資料概況
    python main.py status

    # 多因子選股篩選
    python main.py scan
    python main.py scan --export scan_results.csv
    python main.py scan --notify

    # Discord 通知
    python main.py notify --message "測試訊息"

    # Walk-Forward 驗證（ML 策略防過擬合）
    python main.py walk-forward --stock 2330 --strategy ml_random_forest
    python main.py walk-forward --stock 2330 --strategy ml_xgboost --train-window 504 --test-window 126

    # 每日選股報告
    python main.py report --top 10
    python main.py report --no-ml --notify
    python main.py report --export daily_report.csv

    # 策略回測排名
    python main.py strategy-rank --metric sharpe
    python main.py strategy-rank --strategies sma_cross rsi_threshold --stocks 2330 2317

    # 產業輪動分析
    python main.py industry --refresh --top-sectors 5
    python main.py industry --notify

    # 全市場選股掃描
    python main.py discover
    python main.py discover --top 30 --min-price 50
    python main.py discover --skip-sync --top 10
    python main.py discover --export picks.csv --notify
    python main.py discover all --skip-sync --top 20
    python main.py discover all --skip-sync --min-appearances 2
    python main.py discover all --skip-sync --export compare.csv

    # Discover 推薦績效回測
    python main.py discover-backtest --mode momentum
    python main.py discover-backtest --mode swing --days 5,10,20,60
    python main.py discover-backtest --mode value --top 10
    python main.py discover-backtest --mode momentum --start 2025-06-01 --end 2025-12-31
    python main.py discover-backtest --mode momentum --export result.csv

    # 同步財報資料
    python main.py sync-financial                    # 同步 watchlist 財報（預設最近 4 季）
    python main.py sync-financial --stocks 2330 2317 # 指定股票
    python main.py sync-financial --quarters 8       # 最近 8 季

    # DB 遷移
    python main.py migrate

    # 資料品質檢查
    python main.py validate
    python main.py validate --stocks 2330 2317
    python main.py validate --gap-threshold 3 --streak-threshold 3
    python main.py validate --no-freshness
    python main.py validate --export issues.csv
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.config import settings


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.logging.level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def _build_risk_config(args: argparse.Namespace):
    """從 CLI 參數建立 RiskConfig。"""
    from src.backtest.engine import RiskConfig

    return RiskConfig(
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        trailing_stop_pct=args.trailing_stop,
        position_sizing=args.sizing,
        fixed_fraction=args.fraction,
    )


def cmd_backtest(args: argparse.Namespace) -> None:
    """執行回測（單股或投資組合）。"""
    from datetime import date

    from src.backtest.engine import BacktestEngine
    from src.data.database import init_db
    from src.data.pipeline import save_backtest_result, save_portfolio_result
    from src.strategy import STRATEGY_REGISTRY

    if args.strategy not in STRATEGY_REGISTRY:
        print(f"未知策略: {args.strategy}")
        print(f"可用策略: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)

    init_db()

    start = args.start or settings.fetcher.default_start_date
    end = args.end or date.today().isoformat()

    risk_config = _build_risk_config(args)
    strategy_cls = STRATEGY_REGISTRY[args.strategy]

    # --- 投資組合回測 ---
    if args.stocks:
        from src.backtest.portfolio import PortfolioBacktestEngine, PortfolioConfig

        adj_div = getattr(args, "adjust_dividend", False)
        strategies = [
            strategy_cls(stock_id=sid, start_date=start, end_date=end, adjust_dividend=adj_div) for sid in args.stocks
        ]

        portfolio_config = PortfolioConfig(allocation_method=args.allocation)

        engine = PortfolioBacktestEngine(
            strategies=strategies,
            risk_config=risk_config,
            portfolio_config=portfolio_config,
        )
        result = engine.run()

        # 存入 DB
        bt_id = save_portfolio_result(result)

        # 印出摘要
        print("\n" + "=" * 60)
        print(f"投資組合回測結果 — {result.strategy_name}")
        print(f"  股票: {', '.join(result.stock_ids)}")
        print("=" * 60)
        print(f"  期間:         {result.start_date} ~ {result.end_date}")
        print(f"  配置方式:     {result.allocation_method}")
        print(f"  初始資金:     {result.initial_capital:>14,.0f}")
        print(f"  最終資金:     {result.final_capital:>14,.2f}")
        print(f"  總報酬率:     {result.total_return:>13.2f}%")
        print(f"  年化報酬率:   {result.annual_return:>13.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio or 'N/A':>13}")
        print(f"  Sortino Ratio:{result.sortino_ratio or 'N/A':>13}")
        print(f"  最大回撤:     {result.max_drawdown:>13.2f}%")
        print(f"  Calmar Ratio: {result.calmar_ratio or 'N/A':>13}")
        print(f"  勝率:         {result.win_rate or 'N/A':>13}%")
        print(f"  交易次數:     {result.total_trades:>13}")
        print(f"  VaR (95%):    {result.var_95 or 'N/A':>13}")
        print(f"  CVaR (95%):   {result.cvar_95 or 'N/A':>13}")
        print(f"  Profit Factor:{result.profit_factor or 'N/A':>13}")

        if result.per_stock_returns:
            print("\n  個股報酬貢獻:")
            for sid, ret in result.per_stock_returns.items():
                print(f"    {sid}: {ret:+.2f}%")

        print(f"  (結果已儲存, portfolio_id={bt_id})")
        return

    # --- 單股回測 ---
    if not args.stock:
        print("請指定 --stock 或 --stocks")
        sys.exit(1)

    adj_div = getattr(args, "adjust_dividend", False)
    strategy = strategy_cls(stock_id=args.stock, start_date=start, end_date=end, adjust_dividend=adj_div)

    engine = BacktestEngine(strategy, risk_config=risk_config)
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
    if result.benchmark_return is not None:
        print(f"  基準報酬率:   {result.benchmark_return:>13.2f}%  (Buy & Hold)")
        alpha = result.total_return - result.benchmark_return
        print(f"  超額報酬:     {alpha:>13.2f}%")
    print(f"  年化報酬率:   {result.annual_return:>13.2f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio or 'N/A':>13}")
    print(f"  Sortino Ratio:{result.sortino_ratio or 'N/A':>13}")
    print(f"  最大回撤:     {result.max_drawdown:>13.2f}%")
    print(f"  Calmar Ratio: {result.calmar_ratio or 'N/A':>13}")
    print(f"  勝率:         {result.win_rate or 'N/A':>13}%")
    print(f"  交易次數:     {result.total_trades:>13}")
    print(f"  VaR (95%):    {result.var_95 or 'N/A':>13}")
    print(f"  CVaR (95%):   {result.cvar_95 or 'N/A':>13}")
    print(f"  Profit Factor:{result.profit_factor or 'N/A':>13}")
    print(f"  (結果已儲存, id={bt_id})")


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

    from src.data.database import init_db
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


def cmd_schedule(args: argparse.Namespace) -> None:
    """設定排程任務。"""
    if args.mode == "simple":
        from src.scheduler.simple_scheduler import run_scheduler

        run_scheduler()
    elif args.mode == "windows":
        from src.scheduler.windows_task import generate_scripts

        generate_scripts()


def cmd_status(args: argparse.Namespace) -> None:
    """顯示資料庫概況。"""
    from sqlalchemy import func, select

    from src.data.database import get_session, init_db
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
    from src.data.database import init_db
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


def cmd_walk_forward(args: argparse.Namespace) -> None:
    """執行 Walk-Forward 滾動驗證。"""
    from datetime import date

    from src.backtest.walk_forward import WalkForwardEngine
    from src.data.database import init_db
    from src.strategy import STRATEGY_REGISTRY

    if args.strategy not in STRATEGY_REGISTRY:
        print(f"未知策略: {args.strategy}")
        print(f"可用策略: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)

    init_db()

    start = args.start or settings.fetcher.default_start_date
    end = args.end or date.today().isoformat()
    strategy_cls = STRATEGY_REGISTRY[args.strategy]

    # 收集策略參數
    strategy_params: dict = {}
    if args.lookback:
        strategy_params["lookback"] = args.lookback
    if args.forward_days:
        strategy_params["forward_days"] = args.forward_days
    if args.threshold:
        strategy_params["threshold"] = args.threshold
    if args.train_ratio:
        strategy_params["train_ratio"] = args.train_ratio

    # 除權息還原
    adj_div = getattr(args, "adjust_dividend", False)
    if adj_div:
        strategy_params["adjust_dividend"] = True

    # 對 ML 策略，從策略名自動設定 model_type
    if args.strategy.startswith("ml_"):
        model_type = args.strategy.replace("ml_", "")
        strategy_params.setdefault("model_type", model_type)

    risk_config = _build_risk_config(args)

    print(f"Walk-Forward 驗證: {args.strategy} | {args.stock}")
    print(f"  期間: {start} ~ {end}")
    print(f"  訓練窗口: {args.train_window} 日 | 測試窗口: {args.test_window} 日 | 步進: {args.step_size} 日")
    if strategy_params:
        print(f"  策略參數: {strategy_params}")

    try:
        engine = WalkForwardEngine(
            strategy_cls=strategy_cls,
            stock_id=args.stock,
            start_date=start,
            end_date=end,
            train_window=args.train_window,
            test_window=args.test_window,
            step_size=args.step_size,
            risk_config=risk_config,
            strategy_params=strategy_params,
        )
        result = engine.run()
    except ValueError as e:
        print(f"\n錯誤: {e}")
        sys.exit(1)

    # 印出結果
    print("\n" + "=" * 60)
    print(f"Walk-Forward 驗證結果 — {result.strategy_name} | {result.stock_id}")
    print("=" * 60)
    print(f"  期間:         {result.start_date} ~ {result.end_date}")
    print(f"  Fold 數:      {result.total_folds}")
    print(f"  總報酬率:     {result.total_return:>13.2f}%")
    print(f"  年化報酬率:   {result.annual_return:>13.2f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio or 'N/A':>13}")
    print(f"  最大回撤:     {result.max_drawdown:>13.2f}%")
    print(f"  勝率:         {result.win_rate or 'N/A':>13}%")
    print(f"  交易次數:     {result.total_trades:>13}")
    print(f"  Profit Factor:{result.profit_factor or 'N/A':>13}")

    # 各 Fold 摘要
    if result.folds:
        print(f"\n{'─' * 60}")
        print(f"  {'Fold':>4}  {'訓練期':^23}  {'測試期':^23}  {'報酬':>7}  {'交易':>4}")
        print(f"{'─' * 60}")
        for f in result.folds:
            print(
                f"  {f.fold_idx:>4}  {f.train_start}~{f.train_end}  "
                f"{f.test_start}~{f.test_end}  {f.total_return:>6.2f}%  {f.trades:>4}"
            )


def cmd_report(args: argparse.Namespace) -> None:
    """執行每日選股報告。"""
    from src.data.database import init_db
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
    from src.data.database import init_db
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
    from src.data.database import init_db
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


def cmd_discover(args: argparse.Namespace) -> None:
    """執行全市場選股掃描（momentum / swing / value 模式）。"""
    from src.data.database import init_db
    from src.data.pipeline import sync_market_data, sync_stock_info
    from src.discovery.scanner import DividendScanner, GrowthScanner, MomentumScanner, SwingScanner, ValueScanner

    init_db()

    mode = getattr(args, "mode", "momentum") or "momentum"

    if mode == "all":
        _cmd_discover_all(args)
        return

    mode_label = {
        "momentum": "Momentum 短線動能",
        "swing": "Swing 中期波段",
        "value": "Value 價值修復",
        "dividend": "Dividend 高息存股",
        "growth": "Growth 高成長",
    }[mode]

    # swing 模式自動擴展同步天數
    sync_days = args.sync_days
    if mode == "swing" and sync_days < 80:
        sync_days = 80
        print(f"  [Swing 模式] 自動擴展同步天數至 {sync_days} 天（SMA60 / 60日動能需要）")

    # 同步全市場資料（除非 --skip-sync）
    if not args.skip_sync:
        print("正在同步股票基本資料...")
        sync_stock_info(force_refresh=False)
        print("正在同步全市場資料（TWSE/TPEX 官方資料）...")
        counts = sync_market_data(days=sync_days, max_stocks=args.max_stocks)
        print(
            f"  日K線: {counts['daily_price']:,} 筆 | "
            f"法人: {counts['institutional']:,} 筆 | "
            f"融資融券: {counts['margin']:,} 筆"
        )

    # 選擇 scanner
    scanner_map = {
        "momentum": MomentumScanner,
        "swing": SwingScanner,
        "value": ValueScanner,
        "dividend": DividendScanner,
        "growth": GrowthScanner,
    }
    ScannerClass = scanner_map[mode]
    print(f"正在掃描全市場 [{mode_label}]...")
    scanner = ScannerClass(
        min_price=args.min_price,
        max_price=args.max_price,
        min_volume=args.min_volume,
        top_n_results=args.top,
    )
    result = scanner.run()

    if result.rankings.empty:
        print("無符合條件的股票")
        return

    # 顯示結果
    display = result.rankings.head(args.top)
    print(f"\n{'=' * 80}")
    print(
        f"全市場選股掃描 [{mode_label}] — "
        f"掃描 {result.total_stocks} 支 → 粗篩 {result.after_coarse} 支 → Top {len(display)}"
    )
    print(f"{'=' * 80}")
    print(
        f"{'#':>3}  {'代號':>6} {'名稱':<8}  {'收盤':>8}  {'綜合':>6}  "
        f"{'技術':>6}  {'籌碼':>6}  {'基本':>6}  {'產業加成':>6}  {'產業':<10}"
    )
    print(f"{'─' * 86}")

    for _, row in display.iterrows():
        name = str(row.get("stock_name", ""))[:8]
        industry = str(row.get("industry_category", ""))[:10]
        sector_bonus = row.get("sector_bonus", 0.0)
        if pd.isna(sector_bonus):
            sector_bonus = 0.0
        print(
            f"{int(row['rank']):>3}  {row['stock_id']:>6} {name:<8}  "
            f"{row['close']:>8.1f}  {row['composite_score']:>6.3f}  "
            f"{row['technical_score']:>6.3f}  {row['chip_score']:>6.3f}  "
            f"{row['fundamental_score']:>6.3f}  {sector_bonus:>+6.1%}  {industry:<10}"
        )

    # 產業分布
    if result.sector_summary is not None and not result.sector_summary.empty:
        print(f"\n{'─' * 40}")
        print("產業分布")
        for _, sr in result.sector_summary.head(8).iterrows():
            print(f"  {sr['industry']:<14} {int(sr['count']):>3} 支  (均分 {sr['avg_score']:.3f})")

    # 進出場建議（Top 5）
    ee_cols = ["entry_price", "stop_loss", "take_profit", "entry_trigger", "valid_until"]
    if all(c in result.rankings.columns for c in ee_cols):
        print(f"\n{'─' * 80}")
        print("[ 進出場建議 ]")
        print(f"{'#':>3}  {'代號':>6}  {'進場價':>8}  {'止損':>12}  {'止利':>12}  {'觸發條件':<18}  {'有效至'}")
        print(f"{'─' * 80}")
        for _, row in result.rankings.head(5).iterrows():
            ep = row.get("entry_price")
            sl = row.get("stop_loss")
            tp = row.get("take_profit")
            sl_str = f"{sl:.1f}({(sl - ep) / ep:+.1%})" if pd.notna(sl) and pd.notna(ep) and ep > 0 else "—"
            tp_str = f"{tp:.1f}({(tp - ep) / ep:+.1%})" if pd.notna(tp) and pd.notna(ep) and ep > 0 else "—"
            ep_str = f"{ep:.1f}" if pd.notna(ep) else "—"
            trigger = str(row.get("entry_trigger", "")) or "—"
            valid = str(row.get("valid_until", "")) or "—"
            print(
                f"{int(row['rank']):>3}  {row['stock_id']:>6}  "
                f"{ep_str:>8}  {sl_str:>12}  {tp_str:>12}  {trigger:<18}  {valid}"
            )

    # 儲存推薦記錄到 DB
    _save_discovery_records(result, mode, scanner)

    # 歷史比較
    if args.compare:
        _show_discovery_comparison(mode, result)

    # 匯出 CSV
    if args.export:
        result.rankings.to_csv(args.export, index=False)
        print(f"\n結果已匯出至: {args.export}")

    # Discord 通知
    if args.notify:
        from src.notification.line_notify import send_message
        from src.report.formatter import format_discovery_report

        msgs = format_discovery_report(result, top_n=args.top)
        for msg in msgs:
            send_message(msg)
        print("Discord 通知已發送")


def _build_cross_comparison(results: dict, top_n: int) -> "pd.DataFrame":
    """建立五模式交叉比較表。

    Args:
        results: dict[mode_key → ScanResult]（只含成功的模式）
        top_n: 每個模式取前 N 名納入比較

    Returns:
        DataFrame，欄位：stock_id, stock_name, close, appearances, best_rank,
        momentum_rank, swing_rank, value_rank, dividend_rank, growth_rank,
        industry_category
    """
    mode_cols = ["momentum", "swing", "value", "dividend", "growth"]

    # 收集各模式的 stock → rank 映射
    mode_data: dict[str, dict] = {}
    stock_meta: dict[str, dict] = {}  # stock_id → {name, close, industry}

    for mode_key, result in results.items():
        if result is None or result.rankings.empty:
            mode_data[mode_key] = {}
            continue
        subset = result.rankings.head(top_n)
        ranks = {}
        for _, row in subset.iterrows():
            sid = row["stock_id"]
            ranks[sid] = int(row["rank"])
            if sid not in stock_meta:
                stock_meta[sid] = {
                    "stock_name": str(row.get("stock_name", "")),
                    "close": float(row.get("close", 0.0)),
                    "industry_category": str(row.get("industry_category", "")),
                }
        mode_data[mode_key] = ranks

    if not stock_meta:
        return pd.DataFrame()

    rows = []
    for sid, meta in stock_meta.items():
        row = {"stock_id": sid, **meta}
        appearances = 0
        best_rank = 9999
        for col in mode_cols:
            rank_val = mode_data.get(col, {}).get(sid)
            row[f"{col}_rank"] = rank_val
            if rank_val is not None:
                appearances += 1
                if rank_val < best_rank:
                    best_rank = rank_val
        row["appearances"] = appearances
        row["best_rank"] = best_rank if best_rank < 9999 else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["appearances", "best_rank"], ascending=[False, True]).reset_index(drop=True)
    return df


def _calc_rsi14_from_series(closes: "pd.Series") -> float:
    """從收盤價序列計算 RSI14（取最後一個有效值）。

    使用 Wilder's smoothing（alpha=1/14 EWM）。
    長度不足 15 時回傳 50.0（中性值）。

    Args:
        closes: 收盤價序列（pd.Series，已按日期升序排列）

    Returns:
        RSI14 值（0.0 ~ 100.0），資料不足時回傳 50.0
    """
    if len(closes) < 15:
        return 50.0

    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    last_gain = float(avg_gain.iloc[-1])
    last_loss = float(avg_loss.iloc[-1])

    if last_loss == 0.0:
        return 100.0 if last_gain > 0 else 50.0

    rs = last_gain / last_loss
    return round(100.0 - 100.0 / (1.0 + rs), 2)


def _assess_timing(
    rsi14: float,
    close: float,
    sma20: float,
    atr_pct: float,
    regime: str,
) -> str:
    """評估單股進場時機（純函數）。

    綜合 RSI14 超買/超賣位置、均線相對位置、ATR 波動率、市場 Regime，
    輸出中文時機評估字串。

    Args:
        rsi14:    最新 RSI14 值（0.0 ~ 100.0）
        close:    最新收盤價
        sma20:    SMA20 值
        atr_pct:  ATR14 / close（波動率比例）
        regime:   市場狀態 "bull" | "bear" | "sideways"

    Returns:
        中文時機評估字串
    """
    above_sma = sma20 > 0 and close > sma20 * 1.005

    # 波動率修飾符
    if atr_pct < 0.015:
        vol_tag = "，低波動"
    elif atr_pct > 0.04:
        vol_tag = "，高波動謹慎"
    else:
        vol_tag = ""

    # 決策矩陣
    if rsi14 >= 70:
        timing = "謹慎觀望：RSI 超買，追高風險高"
    elif rsi14 <= 30:
        if regime in ("bull", "sideways"):
            timing = "潛在反彈：RSI 超賣，留意止損"
        else:
            timing = "下跌趨勢中超賣，等待企穩訊號"
    elif above_sma and regime == "bull":
        if rsi14 >= 55:
            timing = "積極做多：動能強勁 + 趨勢向上"
        else:
            timing = "順勢佈局：趨勢向上，動能待確認"
    elif above_sma and regime == "sideways":
        timing = "區間上軌，注意壓力，設好止損"
    elif regime == "bear":
        timing = "空頭環境，建議觀望或嚴守止損"
    else:
        timing = "等待訊號：尚未站上均線"

    return timing + vol_tag


def _cmd_discover_all(args: argparse.Namespace) -> None:
    """執行五個 Scanner 並輸出多模式綜合比較表。"""
    import datetime

    from src.data.database import init_db
    from src.data.pipeline import sync_market_data, sync_stock_info
    from src.discovery.scanner import DividendScanner, GrowthScanner, MomentumScanner, SwingScanner, ValueScanner

    init_db()

    # Swing 模式需要至少 80 天資料
    sync_days = max(args.sync_days, 80)

    if not args.skip_sync:
        print("正在同步股票基本資料...")
        sync_stock_info(force_refresh=False)
        print(f"正在同步全市場資料（{sync_days} 天）...")
        counts = sync_market_data(days=sync_days, max_stocks=args.max_stocks)
        print(
            f"  日K線: {counts['daily_price']:,} 筆 | "
            f"法人: {counts['institutional']:,} 筆 | "
            f"融資融券: {counts['margin']:,} 筆"
        )

    scanner_classes = {
        "momentum": MomentumScanner,
        "swing": SwingScanner,
        "value": ValueScanner,
        "dividend": DividendScanner,
        "growth": GrowthScanner,
    }
    mode_labels = {
        "momentum": "動能",
        "swing": "波段",
        "value": "價值",
        "dividend": "高息",
        "growth": "成長",
    }

    results: dict = {}
    scan_summaries: list[str] = []

    for mode_key, ScannerClass in scanner_classes.items():
        label = mode_labels[mode_key]
        print(f"正在掃描 [{label}]...", end="", flush=True)
        scanner = ScannerClass(
            min_price=args.min_price,
            max_price=args.max_price,
            min_volume=args.min_volume,
            top_n_results=args.top,
        )
        result = scanner.run()
        results[mode_key] = result

        if result.rankings.empty:
            print(" (無符合條件的股票)")
            scan_summaries.append(
                f"  {label:<4} 掃描 {result.total_stocks:,} 支 → 粗篩 {result.after_coarse} 支 → 0 筆"
            )
        else:
            actual = len(result.rankings.head(args.top))
            print(f" → Top {actual}")
            scan_summaries.append(
                f"  {label:<4} 掃描 {result.total_stocks:,} 支 → 粗篩 {result.after_coarse} 支 → Top {actual}"
            )
            _save_discovery_records(result, mode_key, scanner)

    # 建立比較表
    df = _build_cross_comparison(results, args.top)

    # 套用 --min-appearances 篩選
    min_app = getattr(args, "min_appearances", 1)
    if min_app > 1 and not df.empty:
        df = df[df["appearances"] >= min_app].reset_index(drop=True)

    today = datetime.date.today().strftime("%Y-%m-%d")
    n_modes = sum(1 for r in results.values() if r is not None and not r.rankings.empty)

    print(f"\n{'═' * 82}")
    print(f"多模式綜合比較 — {today}  ｜  掃描 {n_modes} 個模式  ×  Top {args.top}")
    print(f"{'═' * 82}")

    if df.empty:
        print(f"  （無出現在 {min_app}+ 個模式的股票）")
    else:
        print(
            f"{'出現':>5}  {'代號':>6} {'名稱':<8}  {'收盤':>8}  "
            f"{'動能':>5} {'波段':>5} {'價值':>5} {'高息':>5} {'成長':>5}  {'產業':<12}"
        )
        print(f"{'─' * 82}")

        def _fmt_rank(val) -> str:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "  —  "
            return f"#{int(val):<4}"

        for _, row in df.iterrows():
            app = int(row["appearances"])
            star = f"★×{app}" if app < 5 else "★×5!"
            name = str(row.get("stock_name", ""))[:8]
            industry = str(row.get("industry_category", ""))[:12]
            close_val = row.get("close", 0.0)
            close_str = f"{close_val:>8.1f}" if pd.notna(close_val) else "       —"
            print(
                f"{star:>5}  {row['stock_id']:>6} {name:<8}  {close_str}  "
                f"{_fmt_rank(row.get('momentum_rank')):>5} "
                f"{_fmt_rank(row.get('swing_rank')):>5} "
                f"{_fmt_rank(row.get('value_rank')):>5} "
                f"{_fmt_rank(row.get('dividend_rank')):>5} "
                f"{_fmt_rank(row.get('growth_rank')):>5}  {industry:<12}"
            )

    print(f"{'─' * 82}")
    print("\n各模式掃描摘要：")
    for s in scan_summaries:
        print(s)

    # CSV 匯出
    if args.export and not df.empty:
        df.to_csv(args.export, index=False)
        print(f"\n結果已匯出至: {args.export}")

    # Discord 通知
    if args.notify and not df.empty:
        from src.notification.line_notify import send_message

        top10 = df[df["appearances"] >= max(2, min_app)].head(10) if len(df) > 0 else df.head(10)
        lines = [f"**多模式綜合選股** ({today})", f"掃描 {n_modes} 個模式 × Top {args.top}", ""]
        for _, row in top10.iterrows():
            name = str(row.get("stock_name", ""))[:8]
            app = int(row["appearances"])
            lines.append(f"{'★' * app} {row['stock_id']} {name} (出現{app}模式)")
        send_message("\n".join(lines))
        print("Discord 通知已發送")


def _format_suggest_discord(
    stock_id: str,
    stock_name: str,
    today: object,
    close: float,
    sma20: float,
    rsi14: float,
    atr_str: str,
    regime_zh: str,
    taiex_close: float,
    entry_price: float,
    sl_str: str,
    tp_str: str,
    rr_str: str,
    trigger: str,
    timing: str,
    valid_until: object,
) -> str:
    """將 suggest 結果格式化為 Discord 訊息（純函數，≤ 2000 字元）。

    Args:
        stock_id:    股票代號
        stock_name:  股票名稱
        today:       分析日期
        close:       最新收盤價
        sma20:       SMA20 值
        rsi14:       RSI14 值
        atr_str:     ATR14 的格式化字串（含百分比）
        regime_zh:   中文市場狀態（多頭/空頭/盤整）
        taiex_close: 加權指數最新收盤
        entry_price: 進場參考價
        sl_str:      止損價格式化字串（含百分比）
        tp_str:      目標價格式化字串（含百分比）
        rr_str:      風險報酬比字串
        trigger:     進場觸發條件
        timing:      時機評估字串
        valid_until: 建議有效日期

    Returns:
        格式化字串（已截斷至 2000 字元）
    """
    sep = "─" * 40
    lines = [
        f"**進出場建議 — {stock_id} {stock_name}**",
        f"分析日期：{today}  ｜  市場：{regime_zh}（TAIEX {taiex_close:,.0f}）",
        "```",
        f"收盤  ：{close:.2f}   SMA20：{sma20:.2f}   RSI：{rsi14:.1f}",
        f"ATR14 ：{atr_str}",
        sep,
        f"進場參考：{entry_price:.2f}",
        f"止  損  ：{sl_str}",
        f"目  標  ：{tp_str}",
        f"風險報酬：{rr_str}",
        sep,
        f"觸發條件：{trigger}",
        f"時機評估：{timing}",
        f"有效至  ：{valid_until}",
        "```",
    ]
    return "\n".join(lines)[:2000]


def _save_discovery_records(result, mode: str, scanner) -> None:
    """將 discover 推薦結果存入 DB（供歷史追蹤用）。"""
    from src.data.database import get_session
    from src.data.schema import DiscoveryRecord

    if result.rankings.empty:
        return

    regime = getattr(scanner, "regime", "sideways")
    scan_date = result.scan_date

    records = []
    for _, row in result.rankings.iterrows():
        records.append(
            DiscoveryRecord(
                scan_date=scan_date,
                mode=mode,
                rank=int(row["rank"]),
                stock_id=row["stock_id"],
                stock_name=str(row.get("stock_name", "")) or None,
                close=float(row["close"]),
                composite_score=float(row["composite_score"]),
                technical_score=float(row.get("technical_score", 0.5)),
                chip_score=float(row.get("chip_score", 0.5)),
                fundamental_score=float(row.get("fundamental_score", 0.5)),
                news_score=float(row.get("news_score", 0.5)),
                sector_bonus=float(row.get("sector_bonus", 0.0)) if pd.notna(row.get("sector_bonus")) else 0.0,
                industry_category=str(row.get("industry_category", "")) or None,
                regime=regime,
                total_stocks=result.total_stocks,
                after_coarse=result.after_coarse,
                entry_price=float(row.get("entry_price")) if pd.notna(row.get("entry_price")) else None,
                stop_loss=float(row.get("stop_loss")) if pd.notna(row.get("stop_loss")) else None,
                take_profit=float(row.get("take_profit")) if pd.notna(row.get("take_profit")) else None,
                entry_trigger=str(row.get("entry_trigger", "")) or None,
                valid_until=row.get("valid_until") if pd.notna(row.get("valid_until")) else None,
            )
        )

    with get_session() as session:
        # 先刪除同日同模式的舊記錄（重跑時覆蓋）
        session.query(DiscoveryRecord).filter(
            DiscoveryRecord.scan_date == scan_date,
            DiscoveryRecord.mode == mode,
        ).delete()
        session.add_all(records)
        session.commit()

    print(f"\n推薦記錄已存入 DB（{len(records)} 筆，{scan_date} {mode}）")


def _show_discovery_comparison(mode: str, current_result) -> None:
    """顯示本次推薦與上次推薦的差異（新進/退出/排名變化）。"""
    from sqlalchemy import func, select

    from src.data.database import get_session
    from src.data.schema import DiscoveryRecord

    if current_result.rankings.empty:
        return

    scan_date = current_result.scan_date

    with get_session() as session:
        # 找到上次掃描日期（同模式、日期 < 今天）
        prev_date_row = session.execute(
            select(func.max(DiscoveryRecord.scan_date)).where(
                DiscoveryRecord.mode == mode,
                DiscoveryRecord.scan_date < scan_date,
            )
        ).scalar()

        if prev_date_row is None:
            print(f"\n{'─' * 40}")
            print("歷史比較：無前次記錄可供比較")
            return

        prev_date = prev_date_row

        # 載入上次記錄
        prev_rows = session.execute(
            select(
                DiscoveryRecord.stock_id,
                DiscoveryRecord.stock_name,
                DiscoveryRecord.rank,
                DiscoveryRecord.composite_score,
            ).where(
                DiscoveryRecord.scan_date == prev_date,
                DiscoveryRecord.mode == mode,
            )
        ).all()

    if not prev_rows:
        print(f"\n{'─' * 40}")
        print("歷史比較：無前次記錄可供比較")
        return

    prev_df = pd.DataFrame(prev_rows, columns=["stock_id", "stock_name", "rank", "composite_score"])
    prev_ids = set(prev_df["stock_id"])
    prev_rank = dict(zip(prev_df["stock_id"], prev_df["rank"]))

    curr_ids = set(current_result.rankings["stock_id"])
    curr_rank = dict(zip(current_result.rankings["stock_id"], current_result.rankings["rank"]))

    new_entries = curr_ids - prev_ids
    exits = prev_ids - curr_ids
    stayed = curr_ids & prev_ids

    print(f"\n{'─' * 50}")
    print(f"歷史比較（vs {prev_date}）")
    print(f"{'─' * 50}")

    # 新進
    if new_entries:
        print(f"\n  新進（{len(new_entries)} 支）：")
        for sid in sorted(new_entries, key=lambda s: curr_rank[s]):
            row = current_result.rankings[current_result.rankings["stock_id"] == sid].iloc[0]
            name = str(row.get("stock_name", ""))[:6]
            print(f"    + #{curr_rank[sid]:>2}  {sid:>6} {name:<6}  綜合 {row['composite_score']:.3f}")

    # 退出
    if exits:
        print(f"\n  退出（{len(exits)} 支）：")
        for sid in sorted(exits, key=lambda s: prev_rank[s]):
            prev_row = prev_df[prev_df["stock_id"] == sid].iloc[0]
            name = str(prev_row.get("stock_name", ""))[:6]
            print(f"    - 前#{prev_rank[sid]:>2}  {sid:>6} {name:<6}")

    # 排名變化（僅顯示變動 >= 3 名的）
    rank_changes = []
    for sid in stayed:
        delta = prev_rank[sid] - curr_rank[sid]  # 正=上升, 負=下降
        if abs(delta) >= 3:
            rank_changes.append((sid, curr_rank[sid], delta))

    if rank_changes:
        rank_changes.sort(key=lambda x: -x[2])  # 上升最多的排前面
        print("\n  排名變動（變動 >= 3 名）：")
        for sid, curr_r, delta in rank_changes:
            arrow = "↑" if delta > 0 else "↓"
            row = current_result.rankings[current_result.rankings["stock_id"] == sid].iloc[0]
            name = str(row.get("stock_name", ""))[:6]
            print(f"    {arrow} #{curr_r:>2} ({delta:>+3})  {sid:>6} {name:<6}  綜合 {row['composite_score']:.3f}")

    if not new_entries and not exits and not rank_changes:
        print("  推薦清單無顯著變化")


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


def cmd_discover_backtest(args: argparse.Namespace) -> None:
    """評估 Discover 推薦的歷史績效。"""
    from src.data.database import init_db
    from src.discovery.performance import DiscoveryPerformance, print_performance_report

    init_db()

    holding_days = [int(d) for d in args.days.split(",")]

    perf = DiscoveryPerformance(
        mode=args.mode,
        holding_days=holding_days,
        top_n=args.top,
        start_date=args.start,
        end_date=args.end,
    )

    print(f"正在計算 {args.mode} 推薦績效...")
    result = perf.evaluate()

    print_performance_report(result, args.mode, args.start, args.end)

    # 匯出 CSV
    if args.export and not result["detail"].empty:
        result["detail"].to_csv(args.export, index=False, encoding="utf-8-sig")
        print(f"\n明細已匯出至: {args.export}")


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


def cmd_suggest(args: argparse.Namespace) -> None:
    """單股進出場建議。

    從 DB 讀取最近 60 日日K線，計算 ATR14 / SMA20 / RSI14，
    偵測市場 Regime，輸出進場區間、止損、目標價與時機評估。
    """
    import datetime

    import pandas as pd
    from sqlalchemy import select

    from src.data.database import get_session, init_db
    from src.data.schema import DailyPrice, StockInfo
    from src.discovery.scanner import _calc_atr14
    from src.regime.detector import MarketRegimeDetector

    stock_id: str = args.stock_id
    init_db()

    # ── 1. 從 DB 載入最近 60 日日K（倒序取 60 筆，再反轉為升序）──────
    with get_session() as session:
        rows = (
            session.execute(
                select(DailyPrice).where(DailyPrice.stock_id == stock_id).order_by(DailyPrice.date.desc()).limit(60)
            )
            .scalars()
            .all()
        )

        info_row = session.execute(select(StockInfo.stock_name).where(StockInfo.stock_id == stock_id)).scalar()

    if not rows:
        print(f"錯誤：找不到股票 {stock_id} 的日K線資料（請先執行 sync）")
        sys.exit(1)

    stock_name: str = info_row or stock_id

    df = (
        pd.DataFrame(
            [
                {
                    "date": r.date,
                    "open": float(r.open),
                    "high": float(r.high),
                    "low": float(r.low),
                    "close": float(r.close),
                    "volume": float(r.volume),
                }
                for r in reversed(rows)
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ── 2. 計算 ATR14、SMA20、RSI14 ────────────────────────────────
    atr14 = _calc_atr14(df)
    close = float(df["close"].iloc[-1])
    sma20 = float(df["close"].tail(20).mean()) if len(df) >= 20 else close
    rsi14 = _calc_rsi14_from_series(df["close"])

    # ── 3. 偵測市場 Regime ─────────────────────────────────────────
    try:
        regime_info = MarketRegimeDetector().detect()
        regime: str = regime_info["regime"]
        taiex_close: float = float(regime_info["taiex_close"])
    except Exception:
        regime = "sideways"
        taiex_close = 0.0

    # ── 4. 計算進出場數字 ──────────────────────────────────────────
    today = datetime.date.today()
    valid_until = (pd.Timestamp(today) + pd.offsets.BDay(5)).date()

    entry_price = round(close, 2)
    atr_pct = atr14 / close if close > 0 else 0.0

    if atr14 > 0:
        stop_loss = round(close - 1.5 * atr14, 2)
        take_profit = round(close + 3.0 * atr14, 2)
        risk_pct = (entry_price - stop_loss) / entry_price * 100
        reward_pct = (take_profit - entry_price) / entry_price * 100
        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0.0
        sl_str = f"{stop_loss:.2f}（-{risk_pct:.1f}%）"
        tp_str = f"{take_profit:.2f}（+{reward_pct:.1f}%）"
        rr_str = f"1 : {rr_ratio:.1f}"
        atr_str = f"{atr14:.2f}（{atr_pct:.1%}）"
    else:
        stop_loss = None
        take_profit = None
        sl_str = "—"
        tp_str = "—"
        rr_str = "—"
        atr_str = "—"

    # ── 5. 計算 entry_trigger（與 scanner.py 邏輯一致）────────────
    if sma20 > 0:
        if close > sma20 * 1.01:
            trigger = "站上均線"
        elif close >= sma20 * 0.99:
            trigger = "貼近均線"
        else:
            trigger = "均線下方，等待確認"
    else:
        trigger = "均線下方，等待確認"

    if atr_pct < 0.02:
        trigger += "，低波動"
    elif atr_pct > 0.04:
        trigger += "，高波動謹慎"

    # ── 6. 時機評估 ────────────────────────────────────────────────
    timing = _assess_timing(rsi14, close, sma20, atr_pct, regime)

    # ── 7. 輸出 CLI ────────────────────────────────────────────────
    regime_zh = {"bull": "多頭", "bear": "空頭", "sideways": "盤整"}.get(regime, regime)
    taiex_str = f"TAIEX {taiex_close:,.0f}" if taiex_close > 0 else ""

    sep60 = "═" * 60
    sep_thin = "─" * 60
    print(f"\n{sep60}")
    print(f"  單股進出場建議  ｜  {stock_id} {stock_name}")
    print(sep60)
    print(f"  分析日期  ：{today}")
    print(f"  最新收盤  ：{close:.2f}")
    print(f"  SMA20    ：{sma20:.2f}")
    print(f"  RSI14    ：{rsi14:.1f}")
    print(f"  ATR14    ：{atr_str}")
    print(sep_thin)
    taiex_part = f"（{taiex_str}）" if taiex_str else ""
    print(f"  市場 Regime ：{regime_zh}{taiex_part}")
    print(sep_thin)
    print(f"  進場參考價  ：{entry_price:.2f}")
    print(f"  止 損 價   ：{sl_str}")
    print(f"  目 標 價   ：{tp_str}")
    print(f"  風 險 報 酬 ：{rr_str}")
    print(sep_thin)
    print(f"  進場觸發  ：{trigger}")
    print(f"  時機評估  ：{timing}")
    print(f"  建議有效至：{valid_until}")
    print(f"{sep60}\n")

    # ── 8. Discord 通知（--notify）────────────────────────────────
    if args.notify:
        from src.notification.line_notify import send_message

        msg = _format_suggest_discord(
            stock_id=stock_id,
            stock_name=stock_name,
            today=today,
            close=close,
            sma20=sma20,
            rsi14=rsi14,
            atr_str=atr_str,
            regime_zh=regime_zh,
            taiex_close=taiex_close,
            entry_price=entry_price,
            sl_str=sl_str,
            tp_str=tp_str,
            rr_str=rr_str,
            trigger=trigger,
            timing=timing,
            valid_until=valid_until,
        )
        ok = send_message(msg)
        print("Discord 通知已發送" if ok else "Discord 通知失敗")


def cmd_export(args: argparse.Namespace) -> None:
    """匯出資料表為 CSV/Parquet。"""
    from src.data.database import init_db
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
    from src.data.database import init_db
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
        print(f"匯入完成：{count:,} 筆 → {args.table}（重複資料自動略過）")


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="台股量化投資系統")
    subparsers = parser.add_subparsers(dest="command")

    # sync 子命令
    sp_sync = subparsers.add_parser("sync", help="同步股票資料")
    sp_sync.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_sync.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_sync.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")
    sp_sync.add_argument("--taiex", action="store_true", help="同步加權指數（現預設啟用）")

    # compute 子命令
    sp_compute = subparsers.add_parser("compute", help="計算技術指標")
    sp_compute.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")

    # backtest 子命令
    sp_bt = subparsers.add_parser("backtest", help="執行回測")
    sp_bt.add_argument("--stock", default=None, help="股票代號（單股回測）")
    sp_bt.add_argument("--stocks", nargs="+", default=None, help="多支股票代號（投資組合回測）")
    sp_bt.add_argument("--strategy", required=True, help="策略名稱 (sma_cross, rsi_threshold, ...)")
    sp_bt.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_bt.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")
    # 風險管理參數
    sp_bt.add_argument("--stop-loss", type=float, default=None, help="停損百分比 (例: 5.0 = -5%%)")
    sp_bt.add_argument("--take-profit", type=float, default=None, help="停利百分比 (例: 15.0 = +15%%)")
    sp_bt.add_argument("--trailing-stop", type=float, default=None, help="移動停損百分比 (例: 8.0)")
    sp_bt.add_argument(
        "--sizing", default="all_in", choices=["all_in", "fixed_fraction", "kelly", "atr"], help="部位大小計算方式"
    )
    sp_bt.add_argument("--fraction", type=float, default=1.0, help="fixed_fraction 比例 (0.0~1.0)")
    sp_bt.add_argument(
        "--allocation",
        default="equal_weight",
        choices=["equal_weight", "custom", "risk_parity", "mean_variance"],
        help="投資組合配置方式 (equal_weight/custom/risk_parity/mean_variance)",
    )
    sp_bt.add_argument(
        "--adjust-dividend", action="store_true", default=False, help="啟用除權息還原（回溯調整價格 + 股利入帳）"
    )

    # dashboard 子命令
    subparsers.add_parser("dashboard", help="啟動視覺化儀表板")

    # optimize 子命令
    sp_opt = subparsers.add_parser("optimize", help="參數優化（Grid Search）")
    sp_opt.add_argument("--stock", required=True, help="股票代號")
    sp_opt.add_argument("--strategy", required=True, help="策略名稱")
    sp_opt.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_opt.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")
    sp_opt.add_argument("--top-n", type=int, default=10, help="顯示前 N 名結果 (預設 10)")
    sp_opt.add_argument("--export", default=None, help="匯出 CSV 路徑")

    # schedule 子命令
    sp_sched = subparsers.add_parser("schedule", help="設定自動排程")
    sp_sched.add_argument(
        "--mode",
        choices=["simple", "windows"],
        default="windows",
        help="排程模式: simple=前景執行, windows=產生 Task Scheduler 腳本",
    )

    # scan 子命令
    sp_scan = subparsers.add_parser("scan", help="多因子選股篩選")
    sp_scan.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_scan.add_argument("--conditions", nargs="+", help="篩選條件（因子名稱）")
    sp_scan.add_argument("--lookback", type=int, default=5, help="回溯天數 (預設 5)")
    sp_scan.add_argument("--export", default=None, help="匯出 CSV 路徑")
    sp_scan.add_argument("--notify", action="store_true", help="將結果發送 Discord 通知")

    # notify 子命令
    sp_notify = subparsers.add_parser("notify", help="發送 Discord Webhook 訊息")
    sp_notify.add_argument("--message", required=True, help="訊息內容")

    # walk-forward 子命令
    sp_wf = subparsers.add_parser("walk-forward", help="Walk-Forward 滾動驗證")
    sp_wf.add_argument("--stock", required=True, help="股票代號")
    sp_wf.add_argument("--strategy", required=True, help="策略名稱 (ml_random_forest, ml_xgboost, ...)")
    sp_wf.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_wf.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")
    sp_wf.add_argument("--train-window", type=int, default=252, help="訓練窗口天數 (預設 252)")
    sp_wf.add_argument("--test-window", type=int, default=63, help="測試窗口天數 (預設 63)")
    sp_wf.add_argument("--step-size", type=int, default=63, help="步進大小 (預設 63)")
    sp_wf.add_argument("--lookback", type=int, default=None, help="ML 回溯天數")
    sp_wf.add_argument("--forward-days", type=int, default=None, help="ML 預測天數")
    sp_wf.add_argument("--threshold", type=float, default=None, help="ML 訊號門檻")
    sp_wf.add_argument("--train-ratio", type=float, default=None, help="ML 訓練比例")
    # 風險管理參數（複用）
    sp_wf.add_argument("--stop-loss", type=float, default=None, help="停損百分比")
    sp_wf.add_argument("--take-profit", type=float, default=None, help="停利百分比")
    sp_wf.add_argument("--trailing-stop", type=float, default=None, help="移動停損百分比")
    sp_wf.add_argument(
        "--sizing", default="all_in", choices=["all_in", "fixed_fraction", "kelly", "atr"], help="部位大小計算方式"
    )
    sp_wf.add_argument("--fraction", type=float, default=1.0, help="fixed_fraction 比例")
    sp_wf.add_argument(
        "--adjust-dividend", action="store_true", default=False, help="啟用除權息還原（回溯調整價格 + 股利入帳）"
    )

    # report 子命令
    sp_report = subparsers.add_parser("report", help="每日選股報告")
    sp_report.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_report.add_argument("--top", type=int, default=10, help="顯示前 N 名 (預設 10)")
    sp_report.add_argument("--no-ml", action="store_true", help="跳過 ML 評分（較快）")
    sp_report.add_argument("--export", default=None, help="匯出 CSV 路徑")
    sp_report.add_argument("--notify", action="store_true", help="發送 Discord 通知")

    # strategy-rank 子命令
    sp_sr = subparsers.add_parser("strategy-rank", help="策略回測排名")
    sp_sr.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_sr.add_argument("--strategies", nargs="+", help="策略名稱（預設 6 個快速策略）")
    sp_sr.add_argument("--metric", default="sharpe", help="排名指標 (sharpe/total_return/win_rate/annual_return)")
    sp_sr.add_argument("--start", default=None, help="回測起始日期 (YYYY-MM-DD)")
    sp_sr.add_argument("--end", default=None, help="回測結束日期 (YYYY-MM-DD)")
    sp_sr.add_argument("--min-trades", type=int, default=3, help="最少交易次數 (預設 3)")
    sp_sr.add_argument("--export", default=None, help="匯出 CSV 路徑")
    sp_sr.add_argument("--notify", action="store_true", help="發送 Discord 通知")

    # industry 子命令
    sp_ind = subparsers.add_parser("industry", help="產業輪動分析")
    sp_ind.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_ind.add_argument("--refresh", action="store_true", help="強制重新抓取 StockInfo")
    sp_ind.add_argument("--top-sectors", type=int, default=5, help="顯示前 N 名產業 (預設 5)")
    sp_ind.add_argument("--top", type=int, default=5, help="每產業顯示前 N 支股票 (預設 5)")
    sp_ind.add_argument("--lookback", type=int, default=20, help="法人流量回溯天數 (預設 20)")
    sp_ind.add_argument("--momentum", type=int, default=60, help="價格動能回溯天數 (預設 60)")
    sp_ind.add_argument("--notify", action="store_true", help="發送 Discord 通知")

    # discover 子命令
    sp_disc = subparsers.add_parser("discover", help="全市場選股掃描 (momentum/swing/value/dividend/growth)")
    sp_disc.add_argument(
        "mode",
        nargs="?",
        default="momentum",
        choices=["momentum", "swing", "value", "dividend", "growth", "all"],
        help="掃描模式：momentum=短線動能, swing=中期波段, value=價值修復, dividend=高息存股, growth=高成長, all=五模式綜合比較 (預設 momentum)",
    )
    sp_disc.add_argument("--top", type=int, default=20, help="顯示前 N 名 (預設 20)")
    sp_disc.add_argument("--min-price", type=float, default=10, help="最低股價 (預設 10)")
    sp_disc.add_argument("--max-price", type=float, default=2000, help="最高股價 (預設 2000)")
    sp_disc.add_argument("--min-volume", type=int, default=500_000, help="最低成交量 (預設 500000)")
    sp_disc.add_argument("--sync-days", type=int, default=3, help="同步最近幾個交易日 (預設 3)")
    sp_disc.add_argument("--max-stocks", type=int, default=200, help="備案逐股抓取上限 (預設 200)")
    sp_disc.add_argument("--skip-sync", action="store_true", help="跳過全市場資料同步")
    sp_disc.add_argument("--export", default=None, help="匯出 CSV 路徑")
    sp_disc.add_argument("--notify", action="store_true", help="發送 Discord 通知")
    sp_disc.add_argument("--compare", action="store_true", help="顯示與上次推薦的差異比較")
    sp_disc.add_argument(
        "--min-appearances",
        type=int,
        default=1,
        help="[all 模式] 只顯示出現在 N 個以上模式的股票（預設 1 = 全部顯示）",
    )

    # discover-backtest 子命令
    sp_db = subparsers.add_parser("discover-backtest", help="評估 Discover 推薦的歷史績效")
    sp_db.add_argument(
        "--mode", required=True, choices=["momentum", "swing", "value", "dividend", "growth"], help="掃描模式"
    )
    sp_db.add_argument("--days", default="5,10,20", help="持有天數（逗號分隔，預設 5,10,20）")
    sp_db.add_argument("--top", type=int, default=None, help="只計算每次掃描前 N 名的績效")
    sp_db.add_argument("--start", default=None, help="掃描日期範圍起始 (YYYY-MM-DD)")
    sp_db.add_argument("--end", default=None, help="掃描日期範圍結束 (YYYY-MM-DD)")
    sp_db.add_argument("--export", default=None, help="匯出明細 CSV 路徑")

    # sync-mops 子命令
    sp_mops = subparsers.add_parser("sync-mops", help="同步 MOPS 最新重大訊息公告")

    # sync-revenue 子命令
    sp_rev = subparsers.add_parser("sync-revenue", help="從 MOPS 同步全市場月營收（上市+上櫃）")
    sp_rev.add_argument("--months", type=int, default=1, help="同步最近幾個月（預設 1）")

    # sync-financial 子命令
    sp_fin = subparsers.add_parser("sync-financial", help="同步財報資料（季報損益表+資產負債表+現金流量表）")
    sp_fin.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_fin.add_argument("--quarters", type=int, default=4, help="同步最近幾季（預設 4）")

    # validate 子命令
    sp_val = subparsers.add_parser("validate", help="資料品質檢查（缺漏、異常值、新鮮度）")
    sp_val.add_argument("--stocks", nargs="+", help="指定股票代號（預設檢查全部）")
    sp_val.add_argument("--gap-threshold", type=int, default=5, help="缺漏營業日門檻（預設 5）")
    sp_val.add_argument("--streak-threshold", type=int, default=5, help="連續漲跌停天數門檻（預設 5）")
    sp_val.add_argument("--no-freshness", action="store_true", help="跳過資料新鮮度檢查")
    sp_val.add_argument("--export", default=None, help="匯出問題清單 CSV 路徑")

    # export 子命令
    sp_exp = subparsers.add_parser("export", help="匯出資料表為 CSV/Parquet")
    sp_exp.add_argument("table", nargs="?", default=None, help="資料表名稱")
    sp_exp.add_argument("-o", "--output", default=None, help="輸出檔案路徑")
    sp_exp.add_argument("--format", default="csv", choices=["csv", "parquet"], help="輸出格式 (預設 csv)")
    sp_exp.add_argument("--stocks", nargs="+", help="篩選股票代號")
    sp_exp.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_exp.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")
    sp_exp.add_argument("--list", action="store_true", help="列出所有可匯出的資料表及筆數")

    # import-data 子命令
    sp_imp = subparsers.add_parser("import-data", help="從 CSV/Parquet 匯入資料")
    sp_imp.add_argument("table", help="目標資料表名稱")
    sp_imp.add_argument("source", help="來源檔案路徑 (.csv 或 .parquet)")
    sp_imp.add_argument("--dry-run", action="store_true", help="僅驗證資料格式，不實際寫入")

    # suggest 子命令
    sp_suggest = subparsers.add_parser("suggest", help="單股進出場建議（ATR14 + SMA20 + RSI14 + Regime）")
    sp_suggest.add_argument("stock_id", help="股票代號（例：2330）")
    sp_suggest.add_argument("--notify", action="store_true", help="發送 Discord 通知")

    # status 子命令
    subparsers.add_parser("status", help="顯示資料庫概況")

    # migrate 子命令
    subparsers.add_parser("migrate", help="執行資料庫 schema 遷移")

    args = parser.parse_args()

    if args.command == "sync":
        cmd_sync(args)
    elif args.command == "compute":
        cmd_compute(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "dashboard":
        cmd_dashboard()
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "schedule":
        cmd_schedule(args)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "notify":
        cmd_notify(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "walk-forward":
        cmd_walk_forward(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "strategy-rank":
        cmd_strategy_rank(args)
    elif args.command == "industry":
        cmd_industry(args)
    elif args.command == "discover":
        cmd_discover(args)
    elif args.command == "discover-backtest":
        cmd_discover_backtest(args)
    elif args.command == "sync-mops":
        cmd_sync_mops(args)
    elif args.command == "sync-revenue":
        cmd_sync_revenue(args)
    elif args.command == "sync-financial":
        cmd_sync_financial(args)
    elif args.command == "migrate":
        cmd_migrate(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "import-data":
        cmd_import_data(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "suggest":
        cmd_suggest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
