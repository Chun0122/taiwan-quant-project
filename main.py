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

    # 每日早晨例行流程（一鍵執行）
    python main.py morning-routine --notify         # 完整流程 + Discord 摘要
    python main.py morning-routine --skip-sync --notify  # 跳過借券/分點同步（資料已新鮮時）
    python main.py morning-routine --dry-run        # 預覽步驟與摘要（不實際執行）
    python main.py morning-routine --top 30 --notify     # discover Top 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import pandas as pd

from src.config import settings
from src.features.indicators import calc_rsi14_from_series as _calc_rsi14_from_series
from src.notification.line_notify import format_suggest_discord as _format_suggest_discord


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.logging.level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _init_db() -> None:
    """共用 DB 初始化，避免各 cmd_ 函數重複 lazy import + 呼叫。"""
    from src.data.database import init_db

    init_db()


def _ensure_sync_market_data(sync_days: int, args: argparse.Namespace) -> None:
    """共用全市場資料同步流程（cmd_discover / _cmd_discover_all 共用）。

    依序執行：stock_info → TAIEX → daily_price + institutional + margin，並印出筆數。
    若 args.skip_sync 為 True 則直接跳過。
    """
    if args.skip_sync:
        return

    from src.data.pipeline import sync_market_data, sync_stock_info, sync_taiex_index

    print("正在同步股票基本資料...")
    sync_stock_info(force_refresh=False)
    print("正在同步 TAIEX 加權指數（Regime 偵測用）...")
    sync_taiex_index()
    print(f"正在同步全市場資料（{sync_days} 天，TWSE/TPEX 官方資料）...")
    counts = sync_market_data(days=sync_days, max_stocks=args.max_stocks)
    print(
        f"  日K線: {counts['daily_price']:,} 筆 | "
        f"法人: {counts['institutional']:,} 筆 | "
        f"融資融券: {counts['margin']:,} 筆"
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


def _print_attribution(attr: object) -> None:
    """印出五因子歸因分析摘要表。"""
    from src.backtest.attribution import FactorAttributionResult

    if not isinstance(attr, FactorAttributionResult) or attr.n_trades == 0:
        print("  因子歸因：交易筆數不足（至少需 3 筆）")
        return

    print(f"\n因子歸因分析（{attr.n_trades} 筆交易）")
    print(f"{'因子':<16} {'相關係數':>8}  解讀")
    print("─" * 40)
    for fname, label in attr.factor_labels.items():
        corr = attr.correlations.get(fname)
        if corr is None:
            print(f"  {label:<14} {'N/A':>8}  資料不足")
        else:
            if corr > 0.2:
                interpretation = "正向貢獻"
            elif corr < -0.2:
                interpretation = "負向貢獻"
            else:
                interpretation = "中性"
            print(f"  {label:<14} {corr:>+8.3f}  {interpretation}")
    print()


def cmd_backtest(args: argparse.Namespace) -> None:
    """執行回測（單股或投資組合）。"""
    from datetime import date

    from src.backtest.engine import BacktestEngine
    from src.data.pipeline import save_backtest_result, save_portfolio_result
    from src.strategy import STRATEGY_REGISTRY

    if args.strategy not in STRATEGY_REGISTRY:
        print(f"未知策略: {args.strategy}")
        print(f"可用策略: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)

    _init_db()

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

    # --- 因子歸因分析 ---
    if getattr(args, "attribution", False):
        from src.backtest.attribution import FactorAttribution

        data = strategy._data
        if data is not None and not data.empty:
            attr = FactorAttribution().compute(result, data)
            _print_attribution(attr)
        else:
            print("  因子歸因：無法取得策略資料")


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

    _init_db()

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

    _init_db()

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

    _init_db()

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
    from src.strategy import STRATEGY_REGISTRY

    if args.strategy not in STRATEGY_REGISTRY:
        print(f"未知策略: {args.strategy}")
        print(f"可用策略: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)

    _init_db()

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

    from src.report.engine import DailyReportEngine

    _init_db()

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

    _init_db()

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

    _init_db()

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

    from src.discovery.scanner import DividendScanner, GrowthScanner, MomentumScanner, SwingScanner, ValueScanner

    _init_db()

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

    # 各模式自動擴展同步天數至 Scanner 所需的 lookback_days
    sync_days = args.sync_days
    if mode == "swing" and sync_days < 80:
        sync_days = 80
        print(f"  [Swing 模式] 自動擴展同步天數至 {sync_days} 天（SMA60 / 60日動能需要）")
    elif mode in ("momentum", "value", "dividend", "growth") and sync_days < 25:
        sync_days = 25
        print(f"  [{mode_label}] 自動擴展同步天數至 {sync_days} 天（ATR14 / SMA20 需要）")

    _ensure_sync_market_data(sync_days, args)

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
        weekly_confirm=getattr(args, "weekly_confirm", False),
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


def _compute_watch_status(
    entry_price: float,  # noqa: ARG001
    stop_loss: float | None,
    take_profit: float | None,
    valid_until: date | None,
    latest_price: float | None,
    today: date,
) -> str:
    """根據最新價格與到期日計算持倉狀態（純函數，不寫 DB）。

    優先級：止損 > 止利 > 過期 > active

    Args:
        entry_price: 進場價（保留供未來擴充，目前未使用）
        stop_loss:   止損價，None 表示未設定
        take_profit: 目標價，None 表示未設定
        valid_until: 有效期限，None 表示永不過期
        latest_price:最新收盤價，None 表示查無資料
        today:       今日日期

    Returns:
        "stopped_loss" | "taken_profit" | "expired" | "active"
    """
    if latest_price is not None:
        if stop_loss is not None and latest_price <= stop_loss:
            return "stopped_loss"
        if take_profit is not None and latest_price >= take_profit:
            return "taken_profit"
    if valid_until is not None and today > valid_until:
        return "expired"
    return "active"


def _compute_trailing_stop(highest_price: float, atr14: float, multiplier: float) -> float:
    """計算移動止損價（純函數）。

    公式：stop = highest_price - atr14 * multiplier
    僅在外部確認 new_stop > current_stop 時才更新（只升不降）。

    Args:
        highest_price: 進場後追蹤的最高收盤價
        atr14:        最近 14 日平均真實波幅
        multiplier:   ATR 倍數（如 1.5）

    Returns:
        移動止損價，四捨五入至小數點後兩位。
    """
    return round(highest_price - atr14 * multiplier, 2)


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

    from src.discovery.scanner import DividendScanner, GrowthScanner, MomentumScanner, SwingScanner, ValueScanner

    _init_db()

    # Swing 模式需要至少 80 天資料
    sync_days = max(args.sync_days, 80)
    _ensure_sync_market_data(sync_days, args)

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
            weekly_confirm=getattr(args, "weekly_confirm", False),
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


def cmd_sync_info(args: argparse.Namespace) -> None:
    """同步全市場股票基本資料（產業分類、上市/上櫃別）到 stock_info 表。"""
    from src.data.pipeline import sync_stock_info

    _init_db()
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
    """同步分點交易資料（FinMind TaiwanStockTradingDailyReport）。"""
    from src.data.pipeline import sync_broker_trades

    stock_ids = args.stocks if args.stocks else None
    days = getattr(args, "days", 5)

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
                stock_ids = list(set((stock_ids or watchlist) + discover_stocks))
                print(f"從最近 discover（{latest_date}）補抓 {len(discover_stocks)} 支，合計 {len(stock_ids)} 支")

    print(f"同步分點交易資料（最近 {days} 日）...")
    count = sync_broker_trades(stock_ids=stock_ids, days=days)
    print(f"\n分點資料同步完成: {count:,} 筆")


def cmd_alert_check(args: argparse.Namespace) -> None:
    """掃描近期 MOPS 重大事件警報（法說會、財報、高關注公告）。

    查詢 Announcement 表，篩選指定天數內的非一般性事件，
    以事件類型分組顯示，並可選擇推播 Discord。
    """
    from sqlalchemy import and_, select

    from src.data.database import get_session, init_db
    from src.data.schema import Announcement

    _init_db()
    init_db()

    days = getattr(args, "days", 7)
    event_types = getattr(args, "types", None)  # None = 顯示全部非 general
    stocks = getattr(args, "stocks", None)
    notify = getattr(args, "notify", False)

    from datetime import date, timedelta

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
        "earnings_call": "📣 法說會",
        "investor_day": "🏢 投資人日",
        "filing": "📋 財報發布",
        "revenue": "💰 營收公告",
        "general": "📰 一般公告",
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


def _compute_revenue_scan(
    watchlist: list[str],
    min_yoy: float,
    min_margin_improve: float,
) -> "pd.DataFrame":
    """掃描 watchlist 中 YoY 高成長 + 毛利率改善的個股（純函數）。

    Args:
        watchlist:           要掃描的股票代號清單
        min_yoy:             最低 YoY 門檻（%，例 10.0）
        min_margin_improve:  毛利率 QoQ 最低改善幅度（百分點，例 0.0 = 正向即可）

    Returns:
        DataFrame 含欄位：stock_id, yoy_growth, mom_growth, gross_margin,
                          margin_change, revenue_rank
    """
    import pandas as pd
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import FinancialStatement, MonthlyRevenue

    # ── 月營收（取每支股票最新一筆）──────────────────────────────────
    with get_session() as session:
        rev_rows = session.execute(
            select(
                MonthlyRevenue.stock_id,
                MonthlyRevenue.date,
                MonthlyRevenue.revenue,
                MonthlyRevenue.yoy_growth,
                MonthlyRevenue.mom_growth,
            ).where(MonthlyRevenue.stock_id.in_(watchlist))
        ).all()

    if not rev_rows:
        return pd.DataFrame()

    df_rev = pd.DataFrame(rev_rows, columns=["stock_id", "date", "revenue", "yoy_growth", "mom_growth"])
    # 每支股票只取最新一筆
    df_rev = df_rev.sort_values("date").groupby("stock_id", sort=False).last().reset_index()

    # ── 財報毛利率（取最新兩季，計算 QoQ 趨勢）─────────────────────
    with get_session() as session:
        fin_rows = session.execute(
            select(
                FinancialStatement.stock_id,
                FinancialStatement.date,
                FinancialStatement.gross_margin,
            )
            .where(FinancialStatement.stock_id.in_(watchlist))
            .order_by(FinancialStatement.stock_id, FinancialStatement.date.desc())
        ).all()

    df_fin = pd.DataFrame(fin_rows, columns=["stock_id", "date", "gross_margin"])
    df_fin = df_fin.dropna(subset=["gross_margin"])

    # 計算毛利率 QoQ 變化（最新季 - 前一季）
    margin_rows = []
    for sid, grp in df_fin.groupby("stock_id", sort=False):
        grp = grp.sort_values("date", ascending=False)
        latest_gm = grp["gross_margin"].iloc[0] if len(grp) >= 1 else None
        prev_gm = grp["gross_margin"].iloc[1] if len(grp) >= 2 else None
        margin_change = (latest_gm - prev_gm) if (latest_gm is not None and prev_gm is not None) else None
        margin_rows.append({"stock_id": sid, "gross_margin": latest_gm, "margin_change": margin_change})

    df_margin = (
        pd.DataFrame(margin_rows)
        if margin_rows
        else pd.DataFrame(columns=["stock_id", "gross_margin", "margin_change"])
    )

    # ── 合併 + 篩選 ───────────────────────────────────────────────────
    df = df_rev.merge(df_margin, on="stock_id", how="left")
    df["yoy_growth"] = pd.to_numeric(df["yoy_growth"], errors="coerce")
    df["mom_growth"] = pd.to_numeric(df["mom_growth"], errors="coerce")

    # 條件篩選
    mask = df["yoy_growth"] >= min_yoy
    if min_margin_improve is not None:
        mask_margin = df["margin_change"].isna() | (df["margin_change"] >= min_margin_improve)
        mask = mask & mask_margin

    df = df[mask].copy()
    if df.empty:
        return df

    # 排名分數：YoY 70% + 毛利率改善 30%
    df["yoy_rank"] = df["yoy_growth"].rank(pct=True)
    df["margin_rank"] = df["margin_change"].fillna(0).rank(pct=True)
    df["revenue_rank"] = df["yoy_rank"] * 0.70 + df["margin_rank"] * 0.30
    df = df.sort_values("revenue_rank", ascending=False)

    return df[["stock_id", "yoy_growth", "mom_growth", "gross_margin", "margin_change", "revenue_rank"]]


# ═══════════════════════════════════════════════════════════════════════════
# P5 籌碼異動警報 — 四個純函數（零 mock 可測試）
# ═══════════════════════════════════════════════════════════════════════════


def detect_volume_spike(
    df_price: "pd.DataFrame",
    lookback: int = 10,
    threshold: float = 2.0,
) -> "pd.DataFrame":
    """量能暴增偵測：今日量 > 近 lookback 天均量 × threshold（純函數）。

    輸入欄位: stock_id, date, volume（股）
    輸出欄位: stock_id, today_vol, avg_vol, vol_ratio
    需至少 2 天資料（1 天歷史 + 1 天今日）；不足回傳空 DataFrame。
    """
    import pandas as pd

    empty = pd.DataFrame(columns=["stock_id", "today_vol", "avg_vol", "vol_ratio"])
    if df_price.empty:
        return empty

    results = []
    for stock_id, grp in df_price.groupby("stock_id"):
        grp = grp.sort_values("date")
        latest_date = grp["date"].max()
        today_row = grp[grp["date"] == latest_date]
        hist = grp[grp["date"] < latest_date].tail(lookback)
        if hist.empty or today_row.empty:
            continue
        today_vol = int(today_row["volume"].iloc[0])
        avg_vol = float(hist["volume"].mean())
        if avg_vol <= 0:
            continue
        ratio = today_vol / avg_vol
        if ratio >= threshold:
            results.append(
                {
                    "stock_id": stock_id,
                    "today_vol": today_vol,
                    "avg_vol": round(avg_vol),
                    "vol_ratio": round(ratio, 2),
                }
            )

    if not results:
        return empty
    return pd.DataFrame(results).sort_values("vol_ratio", ascending=False).reset_index(drop=True)


def detect_institutional_buy(
    df_inst: "pd.DataFrame",
    threshold: float = 3_000_000,
) -> "pd.DataFrame":
    """外資大買超偵測：最新日外資 net > threshold（股）（純函數）。

    輸入欄位: stock_id, date, name, net
    name 用 str.contains("外資") 篩選（容納多種命名格式）。
    輸出欄位: stock_id, inst_net（股）
    """
    import pandas as pd

    empty = pd.DataFrame(columns=["stock_id", "inst_net"])
    if df_inst.empty:
        return empty

    foreign = df_inst[df_inst["name"].str.contains("外資", na=False)].copy()
    if foreign.empty:
        return empty

    latest_date = foreign["date"].max()
    today_foreign = foreign[foreign["date"] == latest_date]

    # 同一股票可能有多筆外資記錄（外資 + 外資自營商），合計
    summed = today_foreign.groupby("stock_id")["net"].sum().reset_index()
    summed.columns = ["stock_id", "inst_net"]

    result = summed[summed["inst_net"] > threshold]
    return result.sort_values("inst_net", ascending=False).reset_index(drop=True)


def detect_sbl_spike(
    df_sbl: "pd.DataFrame",
    lookback: int = 10,
    sigma: float = 2.0,
) -> "pd.DataFrame":
    """借券賣出激增偵測：最新日 sbl_change > mean + sigma × std（純函數）。

    需至少 3 筆歷史資料才計算（含今日），不足回傳空 DataFrame。
    只偵測 sbl_change > 0（借券增加）的情況。
    輸入欄位: stock_id, date, sbl_change
    輸出欄位: stock_id, sbl_change, sbl_mean, sbl_std
    """
    import pandas as pd

    empty = pd.DataFrame(columns=["stock_id", "sbl_change", "sbl_mean", "sbl_std"])
    if df_sbl.empty:
        return empty

    results = []
    for stock_id, grp in df_sbl.groupby("stock_id"):
        grp = grp.sort_values("date")
        if len(grp) < 3:
            continue
        latest_date = grp["date"].max()
        today_row = grp[grp["date"] == latest_date]
        if today_row.empty:
            continue
        today_change = today_row["sbl_change"].iloc[0]
        if pd.isna(today_change):
            continue
        today_change = float(today_change)

        hist_changes = grp[grp["date"] < latest_date]["sbl_change"].dropna()
        if len(hist_changes) < 2:
            continue
        mean_c = float(hist_changes.tail(lookback).mean())
        std_c = float(hist_changes.tail(lookback).std())
        if std_c <= 0:
            continue
        z = (today_change - mean_c) / std_c
        if today_change > 0 and z >= sigma:
            results.append(
                {
                    "stock_id": stock_id,
                    "sbl_change": int(today_change),
                    "sbl_mean": round(mean_c, 1),
                    "sbl_std": round(std_c, 1),
                }
            )

    if not results:
        return empty
    return pd.DataFrame(results).sort_values("sbl_change", ascending=False).reset_index(drop=True)


def detect_broker_concentration(
    df_broker: "pd.DataFrame",
    hhi_threshold: float = 0.4,
) -> "pd.DataFrame":
    """主力分點集中買進：最新日 HHI(淨買超分點) > hhi_threshold AND 總淨買 > 0（純函數）。

    輸入欄位: stock_id, date, broker_id, buy, sell
    輸出欄位: stock_id, broker_hhi, net_buy_total（股）
    """
    import pandas as pd

    empty = pd.DataFrame(columns=["stock_id", "broker_hhi", "net_buy_total"])
    if df_broker.empty:
        return empty

    latest_date = df_broker["date"].max()
    today = df_broker[df_broker["date"] == latest_date].copy()
    today["net"] = (today["buy"].fillna(0) - today["sell"].fillna(0)).astype(int)

    results = []
    for stock_id, grp in today.groupby("stock_id"):
        net_buy_total = int(grp["net"].sum())
        if net_buy_total <= 0:
            continue
        buyers = grp[grp["net"] > 0]
        if buyers.empty:
            continue
        total = buyers["net"].sum()
        shares = buyers["net"] / total
        hhi = float((shares**2).sum())
        if hhi >= hhi_threshold:
            results.append(
                {
                    "stock_id": stock_id,
                    "broker_hhi": round(hhi, 3),
                    "net_buy_total": net_buy_total,
                }
            )

    if not results:
        return empty
    return pd.DataFrame(results).sort_values("broker_hhi", ascending=False).reset_index(drop=True)


def _compute_anomaly_scan(
    watchlist: list[str],
    lookback: int = 10,
    vol_mult: float = 2.0,
    inst_threshold: float = 3_000_000,
    sbl_sigma: float = 2.0,
    hhi_threshold: float = 0.4,
) -> "dict[str, pd.DataFrame]":
    """從 DB 讀取四類資料，呼叫四個純函數，回傳異常偵測結果（純函數）。

    Keys: "volume_spike", "inst_buy", "sbl_spike", "broker_conc"
    各值為 DataFrame；無資料時為空 DataFrame。
    """
    import datetime

    import pandas as pd
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import BrokerTrade, DailyPrice, InstitutionalInvestor, SecuritiesLending

    cutoff = datetime.date.today() - datetime.timedelta(days=lookback + 5)
    inst_cutoff = datetime.date.today() - datetime.timedelta(days=5)

    # ── A. 量能暴增 ──────────────────────────────────────────────────
    try:
        with get_session() as session:
            price_rows = session.execute(
                select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.volume).where(
                    DailyPrice.stock_id.in_(watchlist),
                    DailyPrice.date >= cutoff,
                )
            ).all()
        df_price = pd.DataFrame(price_rows, columns=["stock_id", "date", "volume"]) if price_rows else pd.DataFrame()
    except Exception:
        df_price = pd.DataFrame()

    # ── B. 外資大買超 ─────────────────────────────────────────────────
    try:
        with get_session() as session:
            inst_rows = session.execute(
                select(
                    InstitutionalInvestor.stock_id,
                    InstitutionalInvestor.date,
                    InstitutionalInvestor.name,
                    InstitutionalInvestor.net,
                ).where(
                    InstitutionalInvestor.stock_id.in_(watchlist),
                    InstitutionalInvestor.date >= inst_cutoff,
                )
            ).all()
        df_inst = pd.DataFrame(inst_rows, columns=["stock_id", "date", "name", "net"]) if inst_rows else pd.DataFrame()
    except Exception:
        df_inst = pd.DataFrame()

    # ── C. 借券賣出激增 ───────────────────────────────────────────────
    try:
        with get_session() as session:
            sbl_rows = session.execute(
                select(
                    SecuritiesLending.stock_id,
                    SecuritiesLending.date,
                    SecuritiesLending.sbl_change,
                ).where(
                    SecuritiesLending.stock_id.in_(watchlist),
                    SecuritiesLending.date >= cutoff,
                )
            ).all()
        df_sbl = pd.DataFrame(sbl_rows, columns=["stock_id", "date", "sbl_change"]) if sbl_rows else pd.DataFrame()
    except Exception:
        df_sbl = pd.DataFrame()

    # ── D. 主力分點集中買進 ───────────────────────────────────────────
    try:
        with get_session() as session:
            broker_rows = session.execute(
                select(
                    BrokerTrade.stock_id,
                    BrokerTrade.date,
                    BrokerTrade.broker_id,
                    BrokerTrade.buy,
                    BrokerTrade.sell,
                ).where(
                    BrokerTrade.stock_id.in_(watchlist),
                    BrokerTrade.date >= inst_cutoff,
                )
            ).all()
        df_broker = (
            pd.DataFrame(broker_rows, columns=["stock_id", "date", "broker_id", "buy", "sell"])
            if broker_rows
            else pd.DataFrame()
        )
    except Exception:
        df_broker = pd.DataFrame()

    return {
        "volume_spike": detect_volume_spike(df_price, lookback=lookback, threshold=vol_mult),
        "inst_buy": detect_institutional_buy(df_inst, threshold=inst_threshold),
        "sbl_spike": detect_sbl_spike(df_sbl, lookback=lookback, sigma=sbl_sigma),
        "broker_conc": detect_broker_concentration(df_broker, hhi_threshold=hhi_threshold),
    }


def cmd_anomaly_scan(args: argparse.Namespace) -> None:
    """掃描 watchlist 中成交量/籌碼異動的即時警報。

    偵測四類異常：量能暴增、外資大買超、借券賣出激增、主力分點集中買進。
    資料直接從 DB 讀取（需先執行 sync / sync-sbl / sync-broker）。
    """
    import datetime

    _init_db()

    from src.data.database import get_effective_watchlist

    watchlist = args.stocks if args.stocks else get_effective_watchlist()
    lookback: int = getattr(args, "lookback", 10)
    vol_mult: float = getattr(args, "vol_mult", 2.0)
    inst_threshold: float = getattr(args, "inst_threshold", 3_000_000)
    sbl_sigma: float = getattr(args, "sbl_sigma", 2.0)
    hhi_threshold: float = getattr(args, "hhi_threshold", 0.4)
    notify: bool = getattr(args, "notify", False)

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    print(f"\n掃描 {len(watchlist)} 支股票的籌碼異動（{today_str}）...")

    results = _compute_anomaly_scan(
        watchlist=watchlist,
        lookback=lookback,
        vol_mult=vol_mult,
        inst_threshold=inst_threshold,
        sbl_sigma=sbl_sigma,
        hhi_threshold=hhi_threshold,
    )

    df_vol = results["volume_spike"]
    df_inst = results["inst_buy"]
    df_sbl = results["sbl_spike"]
    df_broker = results["broker_conc"]

    total = len(df_vol) + len(df_inst) + len(df_sbl) + len(df_broker)
    print(f"\n=== 籌碼異動警報（{today_str}，共 {total} 筆）===")

    if not df_vol.empty:
        print(f"\n【📊 量能暴增】（今日量 > {lookback}MA × {vol_mult}x，共 {len(df_vol)} 支）")
        for _, row in df_vol.iterrows():
            today_lot = int(row["today_vol"]) // 1000
            avg_lot = int(row["avg_vol"]) // 1000
            print(f"  {row['stock_id']}  今日量 {today_lot:,} 張  均量 {avg_lot:,} 張  倍率 {row['vol_ratio']:.2f}x")
    else:
        print(f"\n【📊 量能暴增】無（門檻: > {lookback}MA × {vol_mult}x）")

    if not df_inst.empty:
        thresh_lot = int(inst_threshold) // 1000
        print(f"\n【🏦 外資大買超】（淨買超 > {thresh_lot:,} 張，共 {len(df_inst)} 支）")
        for _, row in df_inst.iterrows():
            lots = int(row["inst_net"]) // 1000
            print(f"  {row['stock_id']}  外資淨買 +{lots:,} 張（+{int(row['inst_net']):,} 股）")
    else:
        print(f"\n【🏦 外資大買超】無（門檻: > {int(inst_threshold) // 1000:,} 張）")

    if not df_sbl.empty:
        print(f"\n【🔴 借券賣出激增】（sbl_change > mean + {sbl_sigma}σ，共 {len(df_sbl)} 支）")
        for _, row in df_sbl.iterrows():
            chg_lot = int(row["sbl_change"]) // 1000
            print(
                f"  {row['stock_id']}  借券增加 +{chg_lot:,} 張（均值 {row['sbl_mean']:.0f}  std {row['sbl_std']:.0f}）"
            )
    else:
        print(f"\n【🔴 借券賣出激增】無（門檻: > mean + {sbl_sigma}σ）")

    if not df_broker.empty:
        print(f"\n【🎯 主力分點集中買進】（HHI > {hhi_threshold:.2f}，共 {len(df_broker)} 支）")
        for _, row in df_broker.iterrows():
            net_lot = int(row["net_buy_total"]) // 1000
            print(f"  {row['stock_id']}  HHI={row['broker_hhi']:.3f}  淨買超 +{net_lot:,} 張")
    else:
        print(f"\n【🎯 主力分點集中買進】無（門檻: HHI > {hhi_threshold:.2f}）")

    if notify:
        from src.notification.line_notify import send_message

        lines = [f"📡 **籌碼異動警報** ({today_str})，共 {total} 筆"]
        if not df_vol.empty:
            lines.append(f"\n📊 量能暴增 ({len(df_vol)} 支)")
            for _, row in df_vol.head(5).iterrows():
                lots = int(row["today_vol"]) // 1000
                lines.append(f"  {row['stock_id']} 今日 {lots:,}張 倍率 {row['vol_ratio']:.1f}x")
        if not df_inst.empty:
            lines.append(f"\n🏦 外資大買超 ({len(df_inst)} 支)")
            for _, row in df_inst.head(5).iterrows():
                lots = int(row["inst_net"]) // 1000
                lines.append(f"  {row['stock_id']} +{lots:,}張")
        if not df_sbl.empty:
            lines.append(f"\n🔴 借券激增 ({len(df_sbl)} 支)")
            for _, row in df_sbl.head(3).iterrows():
                lots = int(row["sbl_change"]) // 1000
                lines.append(f"  {row['stock_id']} +{lots:,}張")
        if not df_broker.empty:
            lines.append(f"\n🎯 主力集中買進 ({len(df_broker)} 支)")
            for _, row in df_broker.head(3).iterrows():
                net_lot = int(row["net_buy_total"]) // 1000
                lines.append(f"  {row['stock_id']} HHI={row['broker_hhi']:.2f} +{net_lot:,}張")
        if total == 0:
            lines.append("（今日無異常訊號）")
        ok = send_message("\n".join(lines[:40]))
        print(f"\nDiscord 通知: {'成功' if ok else '失敗（請確認 Webhook 設定）'}")


def cmd_revenue_scan(args: argparse.Namespace) -> None:
    """掃描 watchlist 中 YoY 高成長 + 毛利率改善的個股。

    資料直接從 DB 讀取（需先執行 sync-revenue + sync-financial）。
    """
    import pandas as pd

    _init_db()

    from src.data.database import get_effective_watchlist

    watchlist = args.stocks if args.stocks else get_effective_watchlist()
    top_n = getattr(args, "top", 20)
    min_yoy = getattr(args, "min_yoy", 10.0)
    min_margin = getattr(args, "min_margin_improve", 0.0)
    notify = getattr(args, "notify", False)

    print(f"掃描 {len(watchlist)} 支股票（YoY ≥ {min_yoy}%，毛利率 QoQ ≥ {min_margin:.1f} pp）...")

    df = _compute_revenue_scan(watchlist, min_yoy=min_yoy, min_margin_improve=min_margin)

    if df.empty:
        print("無符合條件的個股")
        return

    df_show = df.head(top_n).copy()
    print(f"\n=== 營收成長掃描結果 Top {min(top_n, len(df_show))} ===\n")
    print(f"{'排名':>4}  {'股票':>6}  {'YoY%':>7}  {'MoM%':>7}  {'毛利率%':>8}  {'毛利率 QoQ':>10}")
    print("-" * 55)
    for rank, (_, row) in enumerate(df_show.iterrows(), 1):
        yoy = f"{row['yoy_growth']:+.1f}%" if pd.notna(row["yoy_growth"]) else "  N/A "
        mom = f"{row['mom_growth']:+.1f}%" if pd.notna(row["mom_growth"]) else "  N/A "
        gm = f"{row['gross_margin']:.1f}%" if pd.notna(row["gross_margin"]) else "  N/A "
        mc = f"{row['margin_change']:+.1f}pp" if pd.notna(row["margin_change"]) else "   N/A  "
        print(f"{rank:>4}  {row['stock_id']:>6}  {yoy:>7}  {mom:>7}  {gm:>8}  {mc:>10}")

    print(f"\n共 {len(df)} 支符合條件（{len(watchlist)} 支中）")

    if notify:
        from src.notification.line_notify import send_message

        lines = [f"📈 營收成長掃描（YoY≥{min_yoy:.0f}%，共 {len(df)} 支）"]
        for rank, (_, row) in enumerate(df_show.iterrows(), 1):
            yoy = f"{row['yoy_growth']:+.1f}%" if pd.notna(row["yoy_growth"]) else "N/A"
            mc = f"{row['margin_change']:+.1f}pp" if pd.notna(row["margin_change"]) else "N/A"
            lines.append(f"#{rank} {row['stock_id']} YoY={yoy} 毛利率QoQ={mc}")
        ok = send_message("\n".join(lines[:30]))
        print(f"\nDiscord 通知: {'成功' if ok else '失敗（請確認 Webhook 設定）'}")


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

    from src.discovery.performance import DiscoveryPerformance, print_performance_report

    _init_db()

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

    from src.data.database import get_session
    from src.data.schema import DailyPrice, StockInfo
    from src.discovery.scanner import _calc_atr14
    from src.regime.detector import MarketRegimeDetector

    stock_id: str = args.stock_id
    _init_db()

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


def _watch_add(args: argparse.Namespace) -> None:
    """watch add：新增持倉監控記錄。"""
    import datetime

    import pandas as pd
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import DailyPrice, DiscoveryRecord, StockInfo, WatchEntry
    from src.discovery.scanner import _calc_atr14

    stock_id: str = args.stock_id
    today = datetime.date.today()

    with get_session() as session:
        stock_name_row = session.execute(select(StockInfo.stock_name).where(StockInfo.stock_id == stock_id)).scalar()
    stock_name: str = stock_name_row or stock_id

    if args.from_discover:
        mode_src: str = args.from_discover
        with get_session() as session:
            rec = (
                session.execute(
                    select(DiscoveryRecord)
                    .where(DiscoveryRecord.stock_id == stock_id)
                    .where(DiscoveryRecord.mode == mode_src)
                    .order_by(DiscoveryRecord.scan_date.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
        if rec is None:
            print(f"錯誤：找不到 {stock_id} 在 {mode_src} 模式的推薦記錄")
            return
        entry_price_val = float(args.price) if args.price else (rec.entry_price or rec.close)
        stop_loss_val = float(args.stop) if args.stop else rec.stop_loss
        take_profit_val = float(args.target) if args.target else rec.take_profit
        entry_trigger_val = rec.entry_trigger
        valid_until_val = rec.valid_until
        source_val = "discover"
        mode_val: str | None = mode_src

    else:
        with get_session() as session:
            rows = (
                session.execute(
                    select(DailyPrice).where(DailyPrice.stock_id == stock_id).order_by(DailyPrice.date.desc()).limit(30)
                )
                .scalars()
                .all()
            )
        if not rows:
            print(f"錯誤：找不到 {stock_id} 的日K線資料（請先執行 sync）")
            return

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

        close = float(df["close"].iloc[-1])
        atr14 = _calc_atr14(df)
        sma20 = float(df["close"].tail(20).mean()) if len(df) >= 20 else close
        atr_pct = atr14 / close if close > 0 else 0.0

        entry_price_val = round(float(args.price), 2) if args.price else round(close, 2)
        stop_loss_val = (
            round(float(args.stop), 2)
            if args.stop
            else (round(entry_price_val - 1.5 * atr14, 2) if atr14 > 0 else None)
        )
        take_profit_val = (
            round(float(args.target), 2)
            if args.target
            else (round(entry_price_val + 3.0 * atr14, 2) if atr14 > 0 else None)
        )

        if sma20 > 0:
            if close > sma20 * 1.01:
                entry_trigger_val: str | None = "站上均線"
            elif close >= sma20 * 0.99:
                entry_trigger_val = "貼近均線"
            else:
                entry_trigger_val = "均線下方，等待確認"
        else:
            entry_trigger_val = None

        if entry_trigger_val and atr_pct > 0:
            if atr_pct < 0.02:
                entry_trigger_val += "，低波動"
            elif atr_pct > 0.04:
                entry_trigger_val += "，高波動謹慎"

        valid_until_val = (pd.Timestamp(today) + pd.offsets.BDay(5)).date()
        source_val = "manual"
        mode_val = None

    # ── 移動止損參數 ────────────────────────────────────────────────
    trailing_enabled: bool = getattr(args, "trailing", False)
    trailing_mult: float = float(getattr(args, "trailing_multiplier", 1.5) or 1.5)

    entry = WatchEntry(
        stock_id=stock_id,
        stock_name=stock_name,
        entry_date=today,
        entry_price=entry_price_val,
        stop_loss=stop_loss_val,
        take_profit=take_profit_val,
        quantity=int(args.qty) if args.qty else None,
        source=source_val,
        mode=mode_val,
        entry_trigger=entry_trigger_val,
        valid_until=valid_until_val,
        status="active",
        notes=args.notes or None,
        trailing_stop_enabled=trailing_enabled,
        trailing_atr_multiplier=trailing_mult if trailing_enabled else None,
        highest_price_since_entry=entry_price_val if trailing_enabled else None,
    )

    with get_session() as session:
        session.add(entry)
        session.commit()
        entry_id = entry.id

    sl_str = f"{stop_loss_val:.2f}" if stop_loss_val else "—"
    tp_str = f"{take_profit_val:.2f}" if take_profit_val else "—"
    trailing_str = f"  [移動止損 ×{trailing_mult}]" if trailing_enabled else ""
    print(f"\n已加入持倉監控 #{entry_id}：{stock_id} {stock_name}{trailing_str}")
    print(f"  進場價：{entry_price_val:.2f}  止損：{sl_str}  目標：{tp_str}")
    if valid_until_val:
        print(f"  有效至：{valid_until_val}  來源：{source_val}")
    print()


def _watch_list(args: argparse.Namespace) -> None:
    """watch list：列出持倉記錄。"""
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import WatchEntry

    status_filter: str = args.status
    with get_session() as session:
        q = select(WatchEntry).order_by(WatchEntry.entry_date.desc())
        if status_filter != "all":
            q = q.where(WatchEntry.status == status_filter)
        entries = session.execute(q).scalars().all()

    if not entries:
        label = "任何" if status_filter == "all" else status_filter
        print(f"目前沒有 {label} 狀態的持倉記錄。使用 `watch add <stock_id>` 新增。")
        return

    STATUS_ZH = {
        "active": "🟢 持倉中",
        "stopped_loss": "🔴 止損",
        "taken_profit": "🟡 止利",
        "expired": "⚫ 過期",
        "closed": "⚪ 已平倉",
    }

    print(
        f"\n{'ID':>4}  {'代號':<8} {'名稱':<12} {'進場日':<12} {'進場價':>8} {'止損':>8} {'目標':>8}  {'類型':<6}  {'狀態'}"
    )
    print("─" * 83)
    for e in entries:
        sl_s = f"{e.stop_loss:.2f}" if e.stop_loss else "  —"
        tp_s = f"{e.take_profit:.2f}" if e.take_profit else "  —"
        st_s = STATUS_ZH.get(e.status, e.status)
        name_s = (e.stock_name or "")[:10]
        # 移動止損標記（[T] = Trailing）
        type_s = f"[T×{e.trailing_atr_multiplier:.1f}]" if e.trailing_stop_enabled else "靜態"
        print(
            f"{e.id:>4}  {e.stock_id:<8} {name_s:<12} {str(e.entry_date):<12} {e.entry_price:>8.2f} {sl_s:>8} {tp_s:>8}  {type_s:<6}  {st_s}"
        )
    print()


def _watch_close(args: argparse.Namespace) -> None:
    """watch close：平倉持倉記錄。"""
    import datetime

    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import WatchEntry

    entry_id_arg: int = args.entry_id
    close_price_arg: float | None = float(args.price) if args.price else None
    close_date_today = datetime.date.today()

    with get_session() as session:
        entry_obj = session.execute(select(WatchEntry).where(WatchEntry.id == entry_id_arg)).scalars().first()
        if entry_obj is None:
            print(f"錯誤：找不到 ID={entry_id_arg} 的持倉記錄")
            return
        entry_obj.status = "closed"
        entry_obj.close_date = close_date_today
        entry_obj.close_price = close_price_arg
        session.commit()

    pnl_str = ""
    if close_price_arg and entry_obj.entry_price:
        pnl_pct = (close_price_arg - entry_obj.entry_price) / entry_obj.entry_price * 100
        pnl_str = f"  損益：{pnl_pct:+.2f}%"
    print(f"\n已平倉 #{entry_id_arg} {entry_obj.stock_id}（{close_date_today}）{pnl_str}\n")


def _watch_update_status() -> None:
    """watch update-status：批次更新止損/止利/過期狀態（含移動止損）。

    對 trailing_stop_enabled=True 的持倉，先依最新收盤價更新
    highest_price_since_entry 與 stop_loss（只升不降），
    再統一檢查止損/止利/過期狀態。
    """
    import datetime

    import pandas as pd
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import DailyPrice, WatchEntry
    from src.discovery.scanner import _calc_atr14

    today = datetime.date.today()

    with get_session() as session:
        active_entries = session.execute(select(WatchEntry).where(WatchEntry.status == "active")).scalars().all()

    if not active_entries:
        print("目前沒有 active 持倉，無需更新。")
        return

    stock_ids = list({e.stock_id for e in active_entries})

    # ── 1. 取得最新收盤價 ─────────────────────────────────────────────
    latest_prices: dict[str, float] = {}
    with get_session() as session:
        for sid in stock_ids:
            row = session.execute(
                select(DailyPrice.close).where(DailyPrice.stock_id == sid).order_by(DailyPrice.date.desc()).limit(1)
            ).scalar()
            if row is not None:
                latest_prices[sid] = float(row)

    # ── 2. 預先計算有移動止損的股票 ATR14 ────────────────────────────
    trailing_ids = {e.stock_id for e in active_entries if e.trailing_stop_enabled}
    atr14_cache: dict[str, float] = {}
    if trailing_ids:
        with get_session() as session:
            for sid in trailing_ids:
                rows = (
                    session.execute(
                        select(DailyPrice).where(DailyPrice.stock_id == sid).order_by(DailyPrice.date.desc()).limit(30)
                    )
                    .scalars()
                    .all()
                )
                if rows:
                    df = (
                        pd.DataFrame(
                            [
                                {
                                    "date": r.date,
                                    "high": float(r.high),
                                    "low": float(r.low),
                                    "close": float(r.close),
                                }
                                for r in reversed(rows)
                            ]
                        )
                        .sort_values("date")
                        .reset_index(drop=True)
                    )
                    atr14_cache[sid] = _calc_atr14(df)

    # ── 3. 批次更新（移動止損 → 狀態判斷）────────────────────────────
    updated = 0
    trailing_updated = 0
    with get_session() as session:
        for e in active_entries:
            latest_price = latest_prices.get(e.stock_id)
            effective_stop = e.stop_loss  # 預設使用目前止損

            # ── 移動止損更新（只升不降）──────────────────────────
            if e.trailing_stop_enabled and latest_price is not None:
                atr14 = atr14_cache.get(e.stock_id, 0.0)
                if atr14 > 0:
                    mult = e.trailing_atr_multiplier or 1.5
                    curr_highest = e.highest_price_since_entry or e.entry_price
                    new_highest = max(curr_highest, latest_price)
                    new_stop = _compute_trailing_stop(new_highest, atr14, mult)

                    # 只在移動止損往上移動時才更新
                    if new_stop > (effective_stop or 0.0):
                        obj = session.execute(select(WatchEntry).where(WatchEntry.id == e.id)).scalars().first()
                        if obj:
                            obj.highest_price_since_entry = new_highest
                            obj.stop_loss = new_stop
                            effective_stop = new_stop
                            trailing_updated += 1

            # ── 狀態判斷（使用更新後的 effective_stop）───────────
            new_status = _compute_watch_status(
                entry_price=e.entry_price,
                stop_loss=effective_stop,
                take_profit=e.take_profit,
                valid_until=e.valid_until,
                latest_price=latest_price,
                today=today,
            )
            if new_status != "active":
                obj = session.execute(select(WatchEntry).where(WatchEntry.id == e.id)).scalars().first()
                if obj:
                    obj.status = new_status
                    updated += 1
        session.commit()

    trailing_msg = f"移動止損更新 {trailing_updated} 筆，" if trailing_ids else ""
    print(f"\n更新完成：{len(active_entries)} 筆持倉，{trailing_msg}觸發狀態變更 {updated} 筆。\n")


def cmd_watchlist(args: argparse.Namespace) -> None:
    """觀察清單管理（add / remove / list / import）。

    DB-based watchlist 取代 settings.yaml watchlist：
    - add：新增股票至 DB watchlist
    - remove：從 DB watchlist 移除股票
    - list：列出 DB watchlist 清單
    - import：從 settings.yaml 一次性匯入所有股票
    """
    import datetime

    from sqlalchemy import select

    from src.config import settings
    from src.data.database import get_effective_watchlist, get_session, init_db
    from src.data.schema import Watchlist

    init_db()
    action: str | None = getattr(args, "wl_action", None)

    if action == "list" or action is None:
        with get_session() as session:
            rows = session.execute(select(Watchlist).order_by(Watchlist.added_date, Watchlist.stock_id)).scalars().all()
        if not rows:
            # DB 為空時顯示 YAML fallback
            yaml_wl = list(settings.fetcher.watchlist)
            print(f"DB watchlist 為空，目前使用 settings.yaml 清單（{len(yaml_wl)} 支）：")
            for sid in yaml_wl:
                print(f"  {sid}")
            print("\n使用 'watchlist import' 將 YAML 清單匯入 DB，或用 'watchlist add <stock_id>' 逐筆新增。")
            return
        print(f"{'股票ID':<8} {'股票名稱':<14} {'加入日期':<12} 備註")
        print("-" * 55)
        for row in rows:
            print(f"{row.stock_id:<8} {row.stock_name or '':<14} {str(row.added_date):<12} {row.note or ''}")
        print(f"\n共 {len(rows)} 支")

    elif action == "add":
        stock_id: str = args.stock_id
        with get_session() as session:
            existing = session.execute(select(Watchlist).where(Watchlist.stock_id == stock_id)).scalar_one_or_none()
            if existing:
                print(f"⚠️  {stock_id} 已在觀察清單中（加入日期：{existing.added_date}）")
                return
            session.add(
                Watchlist(
                    stock_id=stock_id,
                    stock_name=getattr(args, "name", None),
                    added_date=datetime.date.today(),
                    note=getattr(args, "note", None),
                )
            )
            session.commit()
        print(f"✅ 已新增 {stock_id} 至觀察清單")
        print(f"   目前有效 watchlist：{len(get_effective_watchlist())} 支")

    elif action == "remove":
        stock_id = args.stock_id
        with get_session() as session:
            existing = session.execute(select(Watchlist).where(Watchlist.stock_id == stock_id)).scalar_one_or_none()
            if not existing:
                print(f"⚠️  {stock_id} 不在觀察清單中")
                return
            session.delete(existing)
            session.commit()
        print(f"✅ 已從觀察清單移除 {stock_id}")
        print(f"   目前有效 watchlist：{len(get_effective_watchlist())} 支")

    elif action == "import":
        yaml_watchlist = list(settings.fetcher.watchlist)
        added = 0
        skipped = 0
        with get_session() as session:
            for sid in yaml_watchlist:
                existing = session.execute(select(Watchlist).where(Watchlist.stock_id == sid)).scalar_one_or_none()
                if existing:
                    skipped += 1
                else:
                    session.add(
                        Watchlist(
                            stock_id=sid,
                            added_date=datetime.date.today(),
                        )
                    )
                    added += 1
            session.commit()
        print(f"✅ 從 settings.yaml 匯入完成：新增 {added} 支，已存在 {skipped} 支跳過")
        print(f"   DB watchlist 共 {added + skipped} 支")

    else:
        print(f"未知動作：{action}。可用動作：add / remove / list / import")


def cmd_watch(args: argparse.Namespace) -> None:
    """持倉監控管理（add / list / close / update-status）。"""
    _init_db()
    action: str = args.action
    if action == "add":
        _watch_add(args)
    elif action == "list":
        _watch_list(args)
    elif action == "close":
        _watch_close(args)
    elif action == "update-status":
        _watch_update_status()
    else:
        print(f"未知動作：{action}。可用動作：add / list / close / update-status")


def cmd_export(args: argparse.Namespace) -> None:
    """匯出資料表為 CSV/Parquet。"""

    from src.data.io import TABLE_REGISTRY, export_table, list_tables

    _init_db()

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

    _init_db()

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


def _build_morning_discord_summary(today_str: str, top_n: int) -> str:
    """建立早晨例行報告的 Discord 訊息摘要。

    查詢今日 DiscoveryRecord、近3日 Announcement、以及 WatchEntry 狀態，
    組合成一則 Discord 推播訊息（≤ 1900 字元）。
    """
    import datetime
    from collections import defaultdict

    from sqlalchemy import and_, func, select

    from src.data.database import get_session
    from src.data.schema import Announcement, DiscoveryRecord, WatchEntry

    lines: list[str] = [f"🌅 **早晨例行報告** ({today_str})", ""]
    today = datetime.date.fromisoformat(today_str)

    # ── 1. 多模式選股（今日 DiscoveryRecord，出現 2+ 模式）──────────────
    with get_session() as session:
        disc_rows = session.execute(
            select(
                DiscoveryRecord.stock_id,
                DiscoveryRecord.stock_name,
                DiscoveryRecord.mode,
                DiscoveryRecord.rank,
            ).where(DiscoveryRecord.scan_date == today)
        ).all()

    if disc_rows:
        mode_labels = {"momentum": "動", "swing": "波", "value": "值", "dividend": "息", "growth": "長"}
        stock_modes: dict = defaultdict(list)
        stock_names: dict = {}
        for r in disc_rows:
            stock_modes[r.stock_id].append((r.mode, r.rank))
            stock_names[r.stock_id] = r.stock_name or r.stock_id

        multi = {sid: modes for sid, modes in stock_modes.items() if len(modes) >= 2}
        multi_sorted = sorted(multi.items(), key=lambda x: -len(x[1]))

        if multi_sorted:
            lines.append(f"📊 **多模式選股** (出現 2+ 模式，共 {len(multi)} 支)")
            for sid, modes in multi_sorted[:5]:
                name = str(stock_names.get(sid) or "")[:6]
                mode_str = " ".join(f"{mode_labels.get(m, '?')}#{r}" for m, r in sorted(modes, key=lambda x: x[1]))
                lines.append(f"  {'★' * len(modes)} {sid} {name} ({mode_str})")
            if len(multi) > 5:
                lines.append(f"  …共 {len(multi)} 支")
            lines.append("")
        else:
            lines.append(
                f"📊 **多模式選股**：今日無出現 2+ 模式的股票（共掃描 {len(set(r.stock_id for r in disc_rows))} 支）"
            )
            lines.append("")
    else:
        lines.append("📊 **多模式選股**：今日無掃描記錄（請確認 discover all 是否已執行）")
        lines.append("")

    # ── 2. 重大事件（近3日，非 general）────────────────────────────────
    since = today - datetime.timedelta(days=3)
    alert_rows = []
    try:
        with get_session() as session:
            alert_rows = session.execute(
                select(
                    Announcement.date,
                    Announcement.stock_id,
                    Announcement.event_type,
                    Announcement.subject,
                )
                .where(and_(Announcement.date >= since, Announcement.event_type != "general"))
                .order_by(Announcement.date.desc())
                .limit(10)
            ).all()
    except Exception:
        # event_type 欄位可能尚未 migrate，跳過重大事件區塊
        pass

    _EVENT_SHORT = {
        "earnings_call": "📣法說",
        "investor_day": "🏢投資日",
        "filing": "📋財報",
        "revenue": "💰營收",
    }
    if alert_rows:
        lines.append(f"📣 **重大事件** (近3日，{len(alert_rows)} 件)")
        for r in alert_rows[:4]:
            label = _EVENT_SHORT.get(r.event_type, r.event_type)
            subj = str(r.subject or "")[:20]
            lines.append(f"  {r.date} {r.stock_id} {label} {subj}")
        if len(alert_rows) > 4:
            lines.append(f"  …共 {len(alert_rows)} 件")
        lines.append("")

    # ── 3. 持倉監控狀態 ──────────────────────────────────────────────
    with get_session() as session:
        watch_counts: dict[str, int] = {}
        for st in ("active", "stopped_loss", "taken_profit", "expired"):
            cnt = session.execute(select(func.count()).select_from(WatchEntry).where(WatchEntry.status == st)).scalar()
            watch_counts[st] = cnt or 0

    total_watch = sum(watch_counts.values())
    if total_watch > 0:
        parts = []
        if watch_counts["active"]:
            parts.append(f"監控中 {watch_counts['active']} 支")
        if watch_counts["stopped_loss"]:
            parts.append(f"⛔止損 {watch_counts['stopped_loss']} 支")
        if watch_counts["taken_profit"]:
            parts.append(f"✅止利 {watch_counts['taken_profit']} 支")
        if watch_counts["expired"]:
            parts.append(f"⏰過期 {watch_counts['expired']} 支")
        lines.append(f"👁 **持倉監控**：{' | '.join(parts)}")
        lines.append("")

    # ── 4. 籌碼異動警報（快速摘要）──────────────────────────────────
    try:
        from src.config import settings as _cfg

        _anomaly = _compute_anomaly_scan(_cfg.fetcher.watchlist)
        _total_anomaly = sum(len(df) for df in _anomaly.values())
        if _total_anomaly > 0:
            parts_a = []
            if not _anomaly["volume_spike"].empty:
                sids = ", ".join(_anomaly["volume_spike"]["stock_id"].head(3).tolist())
                parts_a.append(f"📊量增{len(_anomaly['volume_spike'])}({sids})")
            if not _anomaly["inst_buy"].empty:
                sids = ", ".join(_anomaly["inst_buy"]["stock_id"].head(3).tolist())
                parts_a.append(f"🏦外資{len(_anomaly['inst_buy'])}({sids})")
            if not _anomaly["sbl_spike"].empty:
                sids = ", ".join(_anomaly["sbl_spike"]["stock_id"].head(3).tolist())
                parts_a.append(f"🔴借券{len(_anomaly['sbl_spike'])}({sids})")
            if not _anomaly["broker_conc"].empty:
                sids = ", ".join(_anomaly["broker_conc"]["stock_id"].head(3).tolist())
                parts_a.append(f"🎯主力{len(_anomaly['broker_conc'])}({sids})")
            lines.append(f"🚨 **籌碼異動** ({_total_anomaly}筆)  " + "  ".join(parts_a))
            lines.append("")
    except Exception:
        pass

    msg = "\n".join(lines)
    # Discord 單訊息上限 2000 字元，保留緩衝
    return msg[:1900] if len(msg) > 1900 else msg


def cmd_morning_routine(args: argparse.Namespace) -> None:
    """每日早晨例行流程。

    依序執行：
      Step 1  sync-sbl            同步全市場借券賣出（TWSE TWT96U，3日）
      Step 2  sync-broker         補抓 discover 推薦分點資料（5日）
      Step 3  discover all        五模式全市場掃描（--skip-sync，不重複同步）
      Step 4  alert-check         MOPS 重大事件警報（近3日）
      Step 5  watch update-status 批次更新持倉止損/止利/過期狀態
      Step 6  revenue-scan        高成長掃描（YoY≥10%，Top 5）
      最終     Discord 推播綜合摘要（需加 --notify）

    Flags:
      --dry-run     只顯示步驟與摘要，不執行任何操作
      --skip-sync   跳過 Step 1–2（借券/分點同步），適合資料已新鮮時使用
      --top N       discover 的 Top N（預設 20）
      --notify      執行完畢後推播 Discord 摘要
    """
    import datetime

    _init_db()

    dry_run: bool = getattr(args, "dry_run", False)
    skip_sync: bool = getattr(args, "skip_sync", False)
    top_n: int = getattr(args, "top", 20)
    notify: bool = getattr(args, "notify", False)

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    TOTAL = 7

    def _step(n: int, title: str) -> None:
        print(f"\n{'═' * 64}")
        print(f"  [Step {n}/{TOTAL}] {title}")
        print(f"{'═' * 64}")

    def _skip(reason: str) -> None:
        print(f"  >> 跳過（{reason}）")

    # ── Step 1: sync-sbl ─────────────────────────────────────────
    _step(1, "同步借券賣出資料（sync-sbl --days 3）")
    if dry_run or skip_sync:
        _skip("dry-run" if dry_run else "--skip-sync")
    else:
        cmd_sync_sbl(argparse.Namespace(days=3))

    # ── Step 2: sync-broker --from-discover ──────────────────────
    _step(2, "同步分點交易資料（sync-broker --from-discover --days 5）")
    if dry_run or skip_sync:
        _skip("dry-run" if dry_run else "--skip-sync")
    else:
        cmd_sync_broker(argparse.Namespace(stocks=None, days=5, from_discover=True))

    # ── Step 3: discover all --skip-sync ─────────────────────────
    _step(3, f"五模式全市場掃描（discover all --skip-sync --top {top_n}）")
    if dry_run:
        _skip("dry-run")
    else:
        _cmd_discover_all(
            argparse.Namespace(
                skip_sync=True,  # 不重複同步市場資料
                sync_days=30,
                top=top_n,
                min_price=10.0,
                max_price=None,
                min_volume=1000,
                max_stocks=None,
                min_appearances=1,
                export=None,
                notify=False,  # 統一由 morning-routine 推播
            )
        )

    # ── Step 4: alert-check --days 3 ─────────────────────────────
    _step(4, "掃描近期重大事件（alert-check --days 3）")
    if dry_run:
        _skip("dry-run")
    else:
        cmd_alert_check(argparse.Namespace(days=3, types=None, stocks=None, notify=False))

    # ── Step 5: watch update-status ──────────────────────────────
    _step(5, "更新持倉狀態（watch update-status）")
    if dry_run:
        _skip("dry-run")
    else:
        _watch_update_status()

    # ── Step 6: revenue-scan --top 5 ─────────────────────────────
    _step(6, "高成長掃描（revenue-scan --min-yoy 10 --top 5）")
    if dry_run:
        _skip("dry-run")
    else:
        cmd_revenue_scan(argparse.Namespace(stocks=None, top=5, min_yoy=10.0, min_margin_improve=0.0, notify=False))

    # ── Step 7: anomaly-scan ──────────────────────────────────────
    _step(7, "籌碼異動掃描（anomaly-scan）")
    if dry_run:
        _skip("dry-run")
    else:
        cmd_anomaly_scan(
            argparse.Namespace(
                stocks=None,
                lookback=10,
                vol_mult=2.0,
                inst_threshold=3_000_000,
                sbl_sigma=2.0,
                hhi_threshold=0.4,
                notify=False,  # 統一由 morning-routine 推播
            )
        )

    # ── 完成提示 ─────────────────────────────────────────────────
    suffix = "（dry-run 模式，未執行任何操作）" if dry_run else ""
    print(f"\n{'═' * 64}")
    print(f"  [完成] 早晨例行流程完成！{suffix}")
    print(f"{'═' * 64}\n")

    # ── Discord 摘要推播（或 dry-run 預覽）────────────────────────
    if notify or dry_run:
        msg = _build_morning_discord_summary(today_str, top_n)
        if dry_run:
            # 使用 UTF-8 輸出，繞過 Windows cp950 對 emoji 的限制
            print("-- Discord Summary Preview (dry-run) --")
            sys.stdout.flush()
            sys.stdout.buffer.write(msg.encode("utf-8"))
            sys.stdout.buffer.write(b"\n--\n")
            sys.stdout.buffer.flush()
        else:
            from src.notification.line_notify import send_message

            ok = send_message(msg)
            print(f"Discord 摘要通知: {'成功' if ok else '失敗（請確認 Webhook 設定）'}")


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
    sp_bt.add_argument("--attribution", action="store_true", default=False, help="回測結束後計算五因子歸因分析")

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
    sp_disc.add_argument(
        "--weekly-confirm",
        action="store_true",
        default=False,
        help="啟用週線多時框確認（週線多頭 +5%%，週線空頭 -5%%，預設關閉）",
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

    # sync-info 子命令
    sp_info = subparsers.add_parser("sync-info", help="同步全市場股票基本資料（產業分類 + 上市/上櫃別）")
    sp_info.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="強制重新同步（即使 DB 已有資料，預設跳過）",
    )

    # sync-holding 子命令
    sp_hold = subparsers.add_parser("sync-holding", help="同步大戶持股分級資料（週資料）")
    sp_hold.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_hold.add_argument("--weeks", type=int, default=4, help="同步最近幾週（預設 4）")

    # sync-sbl 子命令
    sp_sbl = subparsers.add_parser("sync-sbl", help="同步全市場借券賣出資料（TWSE TWT96U）")
    sp_sbl.add_argument("--days", type=int, default=3, help="同步最近幾個交易日（預設 3）")

    # sync-broker 子命令
    sp_broker = subparsers.add_parser("sync-broker", help="同步分點交易資料（FinMind TaiwanStockTradingDailyReport）")
    sp_broker.add_argument("--stocks", nargs="+", help="指定股票代號（預設使用 watchlist）")
    sp_broker.add_argument("--days", type=int, default=5, help="同步最近幾個交易日（預設 5）")
    sp_broker.add_argument("--from-discover", action="store_true", help="補抓最近一次 discover 推薦結果的分點資料")

    # alert-check 子命令
    sp_alert = subparsers.add_parser("alert-check", help="掃描近期 MOPS 重大事件警報（法說會/財報/月營收）")
    sp_alert.add_argument("--days", type=int, default=7, help="查詢最近幾天（預設 7）")
    sp_alert.add_argument(
        "--types",
        nargs="+",
        choices=["earnings_call", "investor_day", "filing", "revenue"],
        help="篩選事件類型（預設全部）",
    )
    sp_alert.add_argument("--stocks", nargs="+", help="指定股票代號（預設全部）")
    sp_alert.add_argument("--notify", action="store_true", help="推播 Discord")

    # revenue-scan 子命令
    sp_rscan = subparsers.add_parser("revenue-scan", help="掃描 watchlist 營收高成長 + 毛利率改善個股")
    sp_rscan.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_rscan.add_argument("--top", type=int, default=20, help="顯示前 N 支（預設 20）")
    sp_rscan.add_argument("--min-yoy", type=float, default=10.0, help="最低 YoY 門檻 %%（預設 10.0）")
    sp_rscan.add_argument("--min-margin-improve", type=float, default=0.0, help="毛利率 QoQ 最低改善 pp（預設 0.0）")
    sp_rscan.add_argument("--notify", action="store_true", help="推播 Discord")

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

    # watchlist 子命令（DB-based 觀察清單管理）
    sp_wlcmd = subparsers.add_parser("watchlist", help="DB-based 觀察清單管理（新增/移除/列出/從YAML匯入）")
    wlcmd_sub = sp_wlcmd.add_subparsers(dest="wl_action")

    # watchlist list
    wlcmd_sub.add_parser("list", help="列出 DB watchlist 清單")

    # watchlist add
    sp_wla = wlcmd_sub.add_parser("add", help="新增股票至 DB watchlist")
    sp_wla.add_argument("stock_id", help="股票代號（例：2330）")
    sp_wla.add_argument("--name", help="股票名稱（例：台積電）")
    sp_wla.add_argument("--note", help="備註")

    # watchlist remove
    sp_wlr = wlcmd_sub.add_parser("remove", help="從 DB watchlist 移除股票")
    sp_wlr.add_argument("stock_id", help="股票代號")

    # watchlist import（從 settings.yaml 一次性匯入）
    wlcmd_sub.add_parser("import", help="從 settings.yaml watchlist 一次性匯入所有股票至 DB")

    # watch 子命令
    sp_watch = subparsers.add_parser("watch", help="持倉監控管理（新增/列出/平倉/更新狀態）")
    watch_sub = sp_watch.add_subparsers(dest="action")

    # watch add
    sp_wa = watch_sub.add_parser("add", help="新增持倉監控")
    sp_wa.add_argument("stock_id", help="股票代號（例：2330）")
    sp_wa.add_argument("--price", type=float, default=None, help="進場價（預設使用最新收盤）")
    sp_wa.add_argument("--stop", type=float, default=None, help="止損價（預設 entry - 1.5×ATR14）")
    sp_wa.add_argument("--target", type=float, default=None, help="目標價（預設 entry + 3.0×ATR14）")
    sp_wa.add_argument("--qty", type=int, default=None, help="股數")
    sp_wa.add_argument(
        "--from-discover",
        metavar="MODE",
        default=None,
        help="從最新 discover 推薦記錄匯入（MODE: momentum/swing/value/dividend/growth）",
    )
    sp_wa.add_argument("--trailing", action="store_true", help="啟用移動止損（隨最高價自動上移止損位置）")
    sp_wa.add_argument(
        "--trailing-multiplier",
        type=float,
        default=1.5,
        metavar="MULT",
        help="移動止損 ATR 倍數（預設 1.5，即止損 = 最高價 - 1.5×ATR14）",
    )
    sp_wa.add_argument("--notes", default=None, help="備註")

    # watch list
    sp_wl = watch_sub.add_parser("list", help="列出持倉")
    sp_wl.add_argument(
        "--status",
        default="active",
        choices=["active", "stopped_loss", "taken_profit", "expired", "closed", "all"],
        help="篩選狀態（預設 active）",
    )

    # watch close
    sp_wc = watch_sub.add_parser("close", help="平倉（標記 closed）")
    sp_wc.add_argument("entry_id", type=int, help="持倉 ID（由 watch list 查詢）")
    sp_wc.add_argument("--price", type=float, default=None, help="平倉價格")

    # watch update-status
    watch_sub.add_parser("update-status", help="批次更新持倉狀態（比對最新收盤價自動標記止損/止利/過期）")

    # anomaly-scan 子命令
    sp_anomaly = subparsers.add_parser(
        "anomaly-scan", help="掃描 watchlist 成交量/籌碼異動警報（量能暴增/外資大買超/借券激增/主力集中）"
    )
    sp_anomaly.add_argument("--stocks", nargs="+", help="指定股票代號（預設使用 watchlist）")
    sp_anomaly.add_argument("--lookback", type=int, default=10, help="計算均量/均值的天數（預設 10）")
    sp_anomaly.add_argument("--vol-mult", type=float, default=2.0, dest="vol_mult", help="量能倍數門檻（預設 2.0）")
    sp_anomaly.add_argument(
        "--inst-threshold",
        type=float,
        default=3_000_000,
        dest="inst_threshold",
        help="外資淨買超股數門檻（預設 3,000,000 股 = 3,000 張）",
    )
    sp_anomaly.add_argument(
        "--sbl-sigma", type=float, default=2.0, dest="sbl_sigma", help="借券激增標準差倍數（預設 2.0σ）"
    )
    sp_anomaly.add_argument(
        "--hhi-threshold",
        type=float,
        default=0.4,
        dest="hhi_threshold",
        help="主力分點集中度 HHI 門檻（預設 0.4）",
    )
    sp_anomaly.add_argument("--notify", action="store_true", help="推播 Discord 通知")

    # morning-routine 子命令
    sp_mr = subparsers.add_parser(
        "morning-routine",
        help="每日早晨例行流程（sync-sbl → sync-broker → discover all → alert-check → watch update-status → revenue-scan → anomaly-scan → Discord 摘要）",
    )
    sp_mr.add_argument(
        "--dry-run",
        action="store_true",
        help="只顯示各步驟與摘要預覽，不實際執行",
    )
    sp_mr.add_argument(
        "--skip-sync",
        action="store_true",
        help="跳過 Step 1–2（借券/分點同步），適合資料已是最新時使用",
    )
    sp_mr.add_argument(
        "--top",
        type=int,
        default=20,
        help="discover all 的 Top N（預設 20）",
    )
    sp_mr.add_argument(
        "--notify",
        action="store_true",
        help="流程完成後推播 Discord 摘要",
    )

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
    elif args.command == "sync-info":
        cmd_sync_info(args)
    elif args.command == "sync-holding":
        cmd_sync_holding(args)
    elif args.command == "sync-sbl":
        cmd_sync_sbl(args)
    elif args.command == "sync-broker":
        cmd_sync_broker(args)
    elif args.command == "alert-check":
        cmd_alert_check(args)
    elif args.command == "revenue-scan":
        cmd_revenue_scan(args)
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
    elif args.command == "anomaly-scan":
        cmd_anomaly_scan(args)
    elif args.command == "morning-routine":
        cmd_morning_routine(args)
    elif args.command == "watchlist":
        cmd_watchlist(args)
    elif args.command == "watch":
        if not args.action:
            sp_watch.print_help()
        else:
            cmd_watch(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
