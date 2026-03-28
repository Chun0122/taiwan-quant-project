"""CLI 回測子命令 — backtest / walk-forward + 輔助函數。"""

from __future__ import annotations

import argparse
import sys

from src.cli.helpers import init_db  # noqa: A004
from src.cli.helpers import safe_print as print
from src.config import settings

EXIT_REASON_LABELS: dict[str, str] = {
    "signal": "策略訊號",
    "stop_loss": "停損",
    "take_profit": "停利",
    "trailing_stop": "移動停損",
    "force_close": "到期平倉",
}


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


def _print_trade_stats(trades: list) -> None:
    """印出交易明細統計。"""
    from src.backtest.metrics import compute_trade_stats

    stats = compute_trade_stats(trades)
    if stats["holding_days_avg"] is None and not stats["exit_reason_counts"]:
        return

    print(f"\n{'─' * 60}")
    print("交易統計")
    print(f"{'─' * 60}")

    if stats["holding_days_avg"] is not None:
        print(f"  平均持倉天數: {stats['holding_days_avg']:>10}")
        print(f"  中位數持倉:   {stats['holding_days_median']:>10}")
        print(f"  最短 / 最長:  {stats['holding_days_min']:>4} / {stats['holding_days_max']} 天")

    if stats["avg_win_return"] is not None:
        print(f"  獲利平均報酬: {stats['avg_win_return']:>+10.2f}%")
    if stats["avg_loss_return"] is not None:
        print(f"  虧損平均報酬: {stats['avg_loss_return']:>+10.2f}%")
    if stats["avg_win_pnl"] is not None:
        print(f"  獲利平均損益: {stats['avg_win_pnl']:>+14,.2f}")
    if stats["avg_loss_pnl"] is not None:
        print(f"  虧損平均損益: {stats['avg_loss_pnl']:>+14,.2f}")

    if stats["max_consecutive_wins"] > 0 or stats["max_consecutive_losses"] > 0:
        print(f"  最大連勝:     {stats['max_consecutive_wins']:>10}")
        print(f"  最大連敗:     {stats['max_consecutive_losses']:>10}")

    if stats["exit_reason_counts"]:
        print("  出場原因:")
        for reason, count in sorted(stats["exit_reason_counts"].items(), key=lambda x: -x[1]):
            label = EXIT_REASON_LABELS.get(reason, reason)
            print(f"    {label:<12} {count:>4} 筆")


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

        # 交易統計
        if result.trades:
            _print_trade_stats(result.trades)

        print(f"\n  (結果已儲存, portfolio_id={bt_id})")

        # 交易明細匯出
        if getattr(args, "export_trades", None) and result.trades:
            from src.backtest.metrics import export_trades

            path = export_trades(result.trades, args.export_trades)
            print(f"  交易明細已匯出至: {path}（{len(result.trades)} 筆）")
        return

    # --- 單股回測 ---
    if not args.stock:
        print("請指定 --stock 或 --stocks")
        sys.exit(1)

    adj_div = getattr(args, "adjust_dividend", False)

    # ML 策略額外參數（Phase C）
    ml_kwargs: dict = {}
    if args.strategy.startswith("ml_"):
        if getattr(args, "shap", False):
            ml_kwargs["use_shap"] = True
        if getattr(args, "optuna", False):
            ml_kwargs["use_optuna"] = True
        if getattr(args, "feature_selection", False):
            ml_kwargs["use_shap"] = True
            ml_kwargs["feature_selection"] = True

    strategy = strategy_cls(stock_id=args.stock, start_date=start, end_date=end, adjust_dividend=adj_div, **ml_kwargs)

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

    # 交易統計
    if result.trades:
        _print_trade_stats(result.trades)

    print(f"\n  (結果已儲存, id={bt_id})")

    # --- 交易明細匯出 ---
    if getattr(args, "export_trades", None) and result.trades:
        from src.backtest.metrics import export_trades

        path = export_trades(result.trades, args.export_trades, stock_id=result.stock_id)
        print(f"  交易明細已匯出至: {path}（{len(result.trades)} 筆）")

    # --- 因子歸因分析 ---
    if getattr(args, "attribution", False):
        from src.backtest.attribution import FactorAttribution

        data = strategy._data
        if data is not None and not data.empty:
            attr = FactorAttribution().compute(result, data)
            _print_attribution(attr)
        else:
            print("  因子歸因：無法取得策略資料")

    # --- SHAP 特徵重要性輸出（Phase C4） ---
    if getattr(args, "shap", False) and hasattr(strategy, "last_shap_importances"):
        shap_imp = strategy.last_shap_importances
        if shap_imp:
            print("\n  🔍 SHAP 特徵重要性 Top-10:")
            print(f"  {'排名':<4} {'特徵名稱':<28} {'重要性':>10}")
            print("  " + "-" * 44)
            for rank, (feat, imp) in enumerate(list(shap_imp.items())[:10], 1):
                print(f"  {rank:<4} {feat:<28} {imp:>10.4f}")
        else:
            print("\n  🔍 SHAP：無法計算特徵重要性（可能 shap 未安裝或模型不支援）")

    # --- Optuna 調優結果輸出（Phase C3） ---
    if getattr(args, "optuna", False) and hasattr(strategy, "last_tune_result"):
        tune = strategy.last_tune_result
        if tune and tune.get("best_params"):
            print(f"\n  ⚡ Optuna 調優（{tune['n_trials']} trials）:")
            print(f"     最佳 CV 分數: {tune['best_score']:.2%}")
            for k, v in tune["best_params"].items():
                print(f"     {k}: {v}")

    # --- CV 結果輸出（Phase C2） ---
    if hasattr(strategy, "last_cv_result") and strategy.last_cv_result:
        cv = strategy.last_cv_result
        if cv["n_splits"] > 0:
            print(f"\n  📊 TimeSeriesSplit CV ({cv['n_splits']}-fold):")
            print(f"     平均準確率: {cv['mean_accuracy']:.2%} ± {cv['std_accuracy']:.2%}")


def cmd_walk_forward(args: argparse.Namespace) -> None:
    """執行 Walk-Forward 滾動驗證。"""
    from datetime import date

    from src.backtest.walk_forward import WalkForwardEngine
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
    print(f"  Sortino Ratio:{result.sortino_ratio or 'N/A':>13}")
    print(f"  Calmar Ratio: {result.calmar_ratio or 'N/A':>13}")
    print(f"  VaR (95%):    {result.var_95 or 'N/A':>13}")
    print(f"  CVaR (95%):   {result.cvar_95 or 'N/A':>13}")
    print(f"  Profit Factor:{result.profit_factor or 'N/A':>13}")

    # 交易統計
    if result.all_trades:
        _print_trade_stats(result.all_trades)

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

    # 交易明細匯出
    if getattr(args, "export_trades", None) and result.all_trades:
        from src.backtest.metrics import export_trades

        path = export_trades(result.all_trades, args.export_trades, stock_id=result.stock_id)
        print(f"\n  交易明細已匯出至: {path}（{len(result.all_trades)} 筆）")
