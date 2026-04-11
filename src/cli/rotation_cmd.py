"""CLI 輪動組合子命令 — rotation create/update/status/history/backtest/管理。"""

from __future__ import annotations

import argparse

from src.cli.helpers import init_db
from src.cli.helpers import safe_print as print


def _rotation_update_all(regime: str | None = None) -> None:
    """更新所有 active 的輪動組合（供 morning-routine 呼叫）。

    Parameters
    ----------
    regime : str | None
        目前市場狀態，傳遞給 RotationManager.update() 用於 Crisis 硬阻擋。
    """
    from src.portfolio.manager import RotationManager

    portfolios = RotationManager.list_portfolios()
    active = [p for p in portfolios if p["status"] == "active"]
    if not active:
        print("  無 active 的輪動組合，跳過。")
        return
    for p in active:
        print(f"\n  ── 更新 [{p['name']}] ({p['mode']}, N={p['max_positions']}, {p['holding_days']}d) ──")
        mgr = RotationManager(p["name"])
        actions = mgr.update(regime=regime)
        if actions:
            sold = len(actions.to_sell)
            bought = len(actions.to_buy)
            renewed = len(actions.renewed)
            held = len(actions.to_hold)
            print(f"    賣出={sold}, 買入={bought}, 續持={renewed}, 保持={held}")
        else:
            print("    無動作（無排名資料或已暫停）")


def cmd_rotation(args: argparse.Namespace) -> None:
    """輪動組合部位控制系統。"""
    init_db()
    action = getattr(args, "action", None)
    if not action:
        print("使用方式: python main.py rotation {create|update|status|history|backtest|list|pause|resume|delete}")
        return

    from src.portfolio.manager import RotationManager

    if action == "create":
        name = args.name
        mode = args.mode
        max_pos = args.max_positions
        hold_days = args.holding_days
        capital = args.capital
        allow_renewal = not getattr(args, "no_renewal", False)

        valid_modes = ("momentum", "swing", "value", "dividend", "growth", "all")
        if mode not in valid_modes:
            print(f"錯誤: --mode 必須為 {valid_modes} 之一")
            return

        try:
            portfolio = RotationManager.create_portfolio(
                name=name,
                mode=mode,
                max_positions=max_pos,
                holding_days=hold_days,
                capital=capital,
                allow_renewal=allow_renewal,
            )
            print(f"已建立輪動組合: {portfolio.name}")
            print(f"  模式: {mode} | 持股上限: {max_pos} | 持有天數: {hold_days}")
            print(f"  初始資金: {capital:,.0f} | 續持: {'是' if allow_renewal else '否'}")
        except (ValueError, KeyError) as e:
            print(f"建立失敗: {e}")

    elif action == "update":
        name = getattr(args, "name", None)
        update_all = getattr(args, "all", False)

        if update_all:
            portfolios = RotationManager.list_portfolios()
            active = [p for p in portfolios if p["status"] == "active"]
            if not active:
                print("無 active 的輪動組合")
                return
            for p in active:
                print(f"\n── 更新 [{p['name']}] ──")
                mgr = RotationManager(p["name"])
                actions = mgr.update()
                if actions:
                    _print_rotation_actions(p["name"], actions)
        elif name:
            mgr = RotationManager(name)
            actions = mgr.update()
            if actions:
                _print_rotation_actions(name, actions)
            elif actions is None:
                print(f"找不到組合或已暫停: {name}")
        else:
            print("請指定 --name 或 --all")

    elif action == "status":
        name = getattr(args, "name", None)
        show_all = getattr(args, "all", False)

        if show_all:
            portfolios = RotationManager.list_portfolios()
            if not portfolios:
                print("尚無輪動組合")
                return
            print(f"\n{'名稱':<20s} {'模式':<8s} {'持股':<6s} {'天數':<6s} {'資金':>14s} {'狀態':<8s}")
            print("─" * 68)
            for p in portfolios:
                print(
                    f"{p['name']:<20s} {p['mode']:<8s} "
                    f"{p['max_positions']:<6d} {p['holding_days']:<6d} "
                    f"{p['current_capital']:>14,.0f} {p['status']:<8s}"
                )
        elif name:
            mgr = RotationManager(name)
            status = mgr.get_status()
            if status is None:
                print(f"找不到組合: {name}")
                return
            _print_rotation_status(status)
        else:
            print("請指定 --name 或 --all")

    elif action == "history":
        name = args.name
        limit = getattr(args, "limit", 30)
        mgr = RotationManager(name)
        df = mgr.get_history(limit=limit)
        if df.empty:
            print(f"[{name}] 無已平倉交易記錄")
            return
        print(f"\n[{name}] 最近 {limit} 筆已平倉交易：")
        print(df.to_string(index=False))

    elif action == "backtest":
        from datetime import date as date_type

        name = getattr(args, "name", None) or "__adhoc__"
        start = date_type.fromisoformat(args.start)
        end = date_type.fromisoformat(args.end)

        mode = getattr(args, "mode", None)
        max_pos = getattr(args, "max_positions", None)
        hold_days = getattr(args, "holding_days", None)
        capital = getattr(args, "capital", None)

        mgr = RotationManager(name)
        result = mgr.backtest(
            start_date=start,
            end_date=end,
            mode=mode,
            max_positions=max_pos,
            holding_days=hold_days,
            capital=capital,
        )

        _print_rotation_backtest(result)

        # 匯出每日持倉快照
        export_pos = getattr(args, "export_positions", None)
        if export_pos and result.daily_positions:
            import pandas as pd

            df_pos = pd.DataFrame(result.daily_positions)
            df_pos.to_csv(export_pos, index=False, encoding="utf-8-sig")
            print(f"\n每日持倉快照已匯出: {export_pos}（{len(df_pos)} 筆）")

    elif action == "list":
        portfolios = RotationManager.list_portfolios()
        if not portfolios:
            print("尚無輪動組合")
            return
        print(f"\n{'名稱':<20s} {'模式':<8s} {'持股':<6s} {'天數':<6s} {'續持':<6s} {'資金':>14s} {'狀態':<8s}")
        print("─" * 74)
        for p in portfolios:
            renewal = "是" if p["allow_renewal"] else "否"
            print(
                f"{p['name']:<20s} {p['mode']:<8s} "
                f"{p['max_positions']:<6d} {p['holding_days']:<6d} {renewal:<6s} "
                f"{p['current_capital']:>14,.0f} {p['status']:<8s}"
            )

    elif action == "pause":
        mgr = RotationManager(args.name)
        if mgr.pause():
            print(f"已暫停: {args.name}")
        else:
            print(f"找不到組合: {args.name}")

    elif action == "resume":
        mgr = RotationManager(args.name)
        if mgr.resume():
            print(f"已恢復: {args.name}")
        else:
            print(f"找不到組合: {args.name}")

    elif action == "delete":
        mgr = RotationManager(args.name)
        if mgr.delete():
            print(f"已刪除: {args.name}")
        else:
            print(f"找不到組合: {args.name}")


def _print_rotation_actions(name: str, actions) -> None:
    """列印 rotation 更新結果。"""
    print(f"\n[{name}] Rotation 更新結果：")
    if actions.to_sell:
        print("  賣出:")
        for s in actions.to_sell:
            print(f"    {s['stock_id']} ({s.get('reason', '')}) @ {s.get('exit_price', 'N/A')}")
    if actions.renewed:
        print("  續持:")
        for r in actions.renewed:
            print(f"    {r['stock_id']} → 延長至 {r.get('new_planned_exit_date', 'N/A')}")
    if actions.to_buy:
        print("  買入:")
        for b in actions.to_buy:
            print(f"    {b['stock_id']} (#{b['rank']}) @ {b['entry_price']} × {b['shares']}股")
    if actions.to_hold:
        print(f"  保持: {len(actions.to_hold)} 筆持倉")


def _print_rotation_status(status: dict) -> None:
    """列印組合狀態。"""
    print(f"\n{'═' * 60}")
    print(f"  輪動組合: {status['name']}  ({status['mode']})")
    print(f"{'═' * 60}")
    print(
        f"  持股上限: {status['max_positions']} | 持有天數: {status['holding_days']} | 續持: {'是' if status['allow_renewal'] else '否'}"
    )
    print(f"  初始資金: {status['initial_capital']:>14,.0f}")
    print(f"  目前資產: {status['current_capital']:>14,.0f}  ({status['total_return_pct']:+.2%})")
    print(f"  現金部位: {status['current_cash']:>14,.0f}")
    print(f"  持倉市值: {status['total_market_value']:>14,.0f}")
    print(f"  未實現損益: {status['total_unrealized_pnl']:>12,.0f}")
    print(f"  狀態: {status['status']} | 更新: {status['updated_at']}")

    holdings = status.get("holdings", [])
    if holdings:
        print(f"\n  {'股票':<8s} {'進場價':>8s} {'現價':>8s} {'股數':>8s} {'損益':>10s} {'報酬率':>8s}")
        print(f"  {'─' * 54}")
        for h in holdings:
            print(
                f"  {h['stock_id']:<8s} {h['entry_price']:>8.1f} {h['current_price']:>8.1f} "
                f"{h['shares']:>8d} {h['unrealized_pnl']:>10,.0f} {h['unrealized_pct']:>+8.2%}"
            )
    else:
        print("\n  （無持倉）")


def _print_rotation_backtest(result) -> None:
    """列印回測結果。"""
    config = result.config
    metrics = result.metrics

    if not metrics:
        print("回測無結果（可能無 DiscoveryRecord 或交易日資料）")
        return

    print(f"\n{'═' * 60}")
    print("  Rotation 回測結果")
    print(f"{'═' * 60}")
    print(f"  模式: {config.get('mode')} | 持股: {config.get('max_positions')} | 天數: {config.get('holding_days')}")
    print(f"  期間: {config.get('start_date')} ~ {config.get('end_date')}")
    print(f"  初始資金: {config.get('capital', 0):,.0f}")
    print(f"  續持: {'是' if config.get('allow_renewal') else '否'}")
    print()
    print(f"  總報酬:     {metrics.get('total_return', 0):>+8.2%}")
    print(f"  年化報酬:   {metrics.get('annual_return', 0):>+8.2%}")
    print(f"  最大回撤:   {metrics.get('max_drawdown', 0):>8.2%}")
    print(f"  Sharpe:     {metrics.get('sharpe_ratio', 0):>8.4f}")
    if metrics.get("sortino_ratio") is not None:
        print(f"  Sortino:    {metrics['sortino_ratio']:>8.4f}")
    if metrics.get("calmar_ratio") is not None:
        print(f"  Calmar:     {metrics['calmar_ratio']:>8.4f}")
    if metrics.get("profit_factor") is not None:
        print(f"  盈虧比:     {metrics['profit_factor']:>8.4f}")
    if metrics.get("var_95") is not None:
        print(f"  VaR(95%):   {metrics['var_95']:>+8.4f}%")
    print(f"  交易次數:   {metrics.get('total_trades', 0):>8d}")
    print(f"  勝率:       {metrics.get('win_rate', 0):>8.2%}")
    print(f"  平均報酬:   {metrics.get('avg_return_per_trade', 0):>+8.4f}")
    print(f"  平均獲利:   {metrics.get('avg_win', 0):>+8.4f}")
    print(f"  平均虧損:   {metrics.get('avg_loss', 0):>+8.4f}")
    print(f"  最終資金:   {metrics.get('final_capital', 0):>14,.0f}")
    print(f"  交易天數:   {metrics.get('trading_days', 0):>8d}")
    # TAIEX Benchmark + Alpha
    if metrics.get("benchmark_return") is not None:
        bm = metrics["benchmark_return"]
        alpha = metrics.get("total_return", 0) * 100 - bm
        print(f"  TAIEX 同期: {bm:>+8.2f}%")
        print(f"  Alpha:      {alpha:>+8.2f}%")
    # 成本歸因
    if metrics.get("total_cost") is not None:
        print(f"  交易成本:   {metrics['total_cost']:>14,.0f}")
        print(f"  成本拖累:   {metrics.get('cost_drag_pct', 0):>8.4f}%")
