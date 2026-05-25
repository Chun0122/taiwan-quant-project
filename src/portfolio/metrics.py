"""Rotation 純計算指標（P2 任務 14 phase 1：從 manager.py 抽出）。

無 DB / IO 副作用，方便單元測試。由 RotationManager 與 backtest 路徑共用。
"""

from __future__ import annotations


def compute_cost_metrics(
    commission: float,
    tax: float,
    slippage: float,
    turnover_value: float,
    capital: float,
) -> dict:
    """計算成本拆解指標（純函式，可獨立單元測試）。

    回傳欄位：
      total_commission / total_tax / total_slippage_cost: 各項累計金額
      total_cost: 三項加總
      cost_drag_pct: 總成本佔初始資金 % （手續費+稅+滑價 / capital × 100）
      commission_pct / tax_pct / slippage_pct: 各項佔初始資金 %（總和 == cost_drag_pct）
      turnover_value: 累計名目交易額（買賣雙邊計入）
      turnover_ratio: turnover / capital
      cost_per_turnover_bps: 每元周轉成本（bps），跨期間/策略可比較指標

    不變式：
      total_cost == total_commission + total_tax + total_slippage_cost
      cost_drag_pct ≈ commission_pct + tax_pct + slippage_pct（rounding 誤差 < 0.001%）
      turnover_value == 0 → cost_per_turnover_bps == 0（避免除零）
    """
    total_cost = commission + tax + slippage
    return {
        "total_commission": round(commission, 2),
        "total_tax": round(tax, 2),
        "total_slippage_cost": round(slippage, 2),
        "total_cost": round(total_cost, 2),
        "cost_drag_pct": round(total_cost / capital * 100, 4) if capital > 0 else 0,
        "commission_pct": round(commission / capital * 100, 4) if capital > 0 else 0,
        "tax_pct": round(tax / capital * 100, 4) if capital > 0 else 0,
        "slippage_pct": round(slippage / capital * 100, 4) if capital > 0 else 0,
        "turnover_value": round(turnover_value, 2),
        "turnover_ratio": round(turnover_value / capital, 4) if capital > 0 else 0,
        "cost_per_turnover_bps": round(total_cost / turnover_value * 10000, 2) if turnover_value > 0 else 0,
    }


def compute_benchmark_alpha_fields(
    today_bm_close: float | None,
    prev_bm_close: float | None,
    base_bm_close: float | None,
    portfolio_cum_return: float | None,
) -> tuple[float | None, float | None, float | None]:
    """純函數：回傳 (benchmark_return_pct, benchmark_cum_return_pct, alpha_cum_pct)。

    供 `_write_daily_snapshot` 與 `backfill_snapshot_benchmark_alpha` 共用，
    確保兩條路徑的計算邏輯一致（audit S4 純函數化建議）。
    任一輸入為 None / 0 / 非正值即回傳 None，不擲例外。
    """
    benchmark_return_pct: float | None = None
    if today_bm_close is not None and prev_bm_close is not None and prev_bm_close > 0:
        benchmark_return_pct = (today_bm_close - prev_bm_close) / prev_bm_close

    benchmark_cum_return_pct: float | None = None
    if today_bm_close is not None and base_bm_close is not None and base_bm_close > 0:
        benchmark_cum_return_pct = (today_bm_close - base_bm_close) / base_bm_close

    alpha_cum_pct: float | None = None
    if benchmark_cum_return_pct is not None and portfolio_cum_return is not None:
        alpha_cum_pct = portfolio_cum_return - benchmark_cum_return_pct

    return benchmark_return_pct, benchmark_cum_return_pct, alpha_cum_pct
