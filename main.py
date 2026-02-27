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

    # DB 遷移
    python main.py migrate
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

    # 同步 TAIEX 指數
    if args.taiex:
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
    from src.discovery.scanner import MomentumScanner, SwingScanner, ValueScanner

    init_db()

    mode = getattr(args, "mode", "momentum") or "momentum"
    mode_label = {
        "momentum": "Momentum 短線動能",
        "swing": "Swing 中期波段",
        "value": "Value 價值修復",
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
    scanner_map = {"momentum": MomentumScanner, "swing": SwingScanner, "value": ValueScanner}
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
            dist = session.execute(
                select(Announcement.sentiment, func.count()).group_by(Announcement.sentiment)
            ).all()

        sentiment_labels = {1: "正面", 0: "中性", -1: "負面"}
        print("\n情緒分布:")
        for sentiment, cnt in sorted(dist, key=lambda x: x[0], reverse=True):
            label = sentiment_labels.get(sentiment, str(sentiment))
            print(f"  {label}: {cnt:,} 筆")


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


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="台股量化投資系統")
    subparsers = parser.add_subparsers(dest="command")

    # sync 子命令
    sp_sync = subparsers.add_parser("sync", help="同步股票資料")
    sp_sync.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_sync.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_sync.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")
    sp_sync.add_argument("--taiex", action="store_true", help="同步加權指數")

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
        "--allocation", default="equal_weight", choices=["equal_weight", "custom"], help="投資組合配置方式"
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
    sp_disc = subparsers.add_parser("discover", help="全市場選股掃描 (momentum/swing)")
    sp_disc.add_argument(
        "mode",
        nargs="?",
        default="momentum",
        choices=["momentum", "swing", "value"],
        help="掃描模式：momentum=短線動能, swing=中期波段, value=價值修復 (預設 momentum)",
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

    # sync-mops 子命令
    sp_mops = subparsers.add_parser("sync-mops", help="同步 MOPS 最新重大訊息公告")

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
    elif args.command == "sync-mops":
        cmd_sync_mops(args)
    elif args.command == "migrate":
        cmd_migrate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
