"""Rotation audit 純函數（rotation-audit CLI 的計算核心）。

支援可重複的修復前後 / 期間對比審計（對應 logs/audit_20260529/REPORT.md 的
A/B trade stats + benchmark alpha 分解 + 訊號穩定性 Jaccard）。

全為純函數（無 DB / IO），DB 查詢在 cli/audit_cmd.py 完成後餵入。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class TradeStats:
    """單一期間 closed trade 統計。"""

    n_closed: int = 0
    win_pct: float | None = None
    avg_return_pct: float | None = None
    total_pnl: float = 0.0
    stop_loss_count: int = 0
    stop_loss_pct: float | None = None


def compute_trade_stats(trades: list[dict]) -> TradeStats:
    """從 closed trade dict list 計算統計（純函數）。

    Parameters
    ----------
    trades : list of dict，每筆需含 keys: return_pct, pnl, exit_reason
        （return_pct 為小數，如 0.05 = +5%；本函數輸出 win_pct/sl_pct 為百分比）
    """
    n = len(trades)
    if n == 0:
        return TradeStats()

    wins = sum(1 for t in trades if (t.get("return_pct") or 0) > 0)
    sl = sum(1 for t in trades if t.get("exit_reason") == "stop_loss")
    returns = [t.get("return_pct") for t in trades if t.get("return_pct") is not None]
    pnls = [t.get("pnl") or 0.0 for t in trades]

    return TradeStats(
        n_closed=n,
        win_pct=round(100.0 * wins / n, 1),
        avg_return_pct=round(100.0 * statistics.fmean(returns), 2) if returns else None,
        total_pnl=round(sum(pnls), 0),
        stop_loss_count=sl,
        stop_loss_pct=round(100.0 * sl / n, 1),
    )


@dataclass
class AlphaDelta:
    """期間 benchmark alpha 增量（snapshot-based）。"""

    portfolio_name: str
    cap_start: float | None
    cap_end: float | None
    portfolio_return_pct: float | None  # (cap_end - cap_start) / cap_start × 100
    alpha_start_pct: float | None
    alpha_end_pct: float | None
    alpha_delta_pp: float | None  # alpha_end - alpha_start（百分點）
    benchmark_delta_pp: float | None  # benchmark_cum 變化（百分點）


def compute_alpha_delta(
    portfolio_name: str,
    snap_start: dict | None,
    snap_end: dict | None,
) -> AlphaDelta:
    """從期初 / 期末兩筆 snapshot 計算 alpha 增量（純函數）。

    snap dict 需含：total_capital, alpha_cum_pct, benchmark_cum_return_pct
    （alpha_cum_pct / benchmark_cum_return_pct 為小數；本函數輸出為百分點 pp）。
    任一 snapshot 為 None → 對應欄位 None（不擲例外）。
    """
    cap_start = snap_start.get("total_capital") if snap_start else None
    cap_end = snap_end.get("total_capital") if snap_end else None

    port_ret: float | None = None
    if cap_start and cap_end and cap_start > 0:
        port_ret = round((cap_end - cap_start) / cap_start * 100, 2)

    a_start = snap_start.get("alpha_cum_pct") if snap_start else None
    a_end = snap_end.get("alpha_cum_pct") if snap_end else None
    alpha_delta: float | None = None
    if a_start is not None and a_end is not None:
        alpha_delta = round((a_end - a_start) * 100, 2)

    b_start = snap_start.get("benchmark_cum_return_pct") if snap_start else None
    b_end = snap_end.get("benchmark_cum_return_pct") if snap_end else None
    bm_delta: float | None = None
    if b_start is not None and b_end is not None:
        bm_delta = round((b_end - b_start) * 100, 2)

    return AlphaDelta(
        portfolio_name=portfolio_name,
        cap_start=cap_start,
        cap_end=cap_end,
        portfolio_return_pct=port_ret,
        alpha_start_pct=round(a_start * 100, 2) if a_start is not None else None,
        alpha_end_pct=round(a_end * 100, 2) if a_end is not None else None,
        alpha_delta_pp=alpha_delta,
        benchmark_delta_pp=bm_delta,
    )


@dataclass
class JaccardStability:
    """連續日對的 top-N 重疊穩定性。"""

    pairs: list[dict] = field(default_factory=list)  # [{day1, day2, overlap, union, jaccard}]
    mean_jaccard: float | None = None
    median_jaccard: float | None = None
    min_jaccard: float | None = None
    max_jaccard: float | None = None


def compute_jaccard_stability(daily_sets: list[tuple[str, set[str]]]) -> JaccardStability:
    """計算相鄰掃描日 top-N 集合的 Jaccard 重疊序列（純函數）。

    Parameters
    ----------
    daily_sets : list of (date_str, set_of_stock_ids)，須已按日期排序。

    Jaccard = |交集| / |聯集|。空聯集視為 0。
    """
    if len(daily_sets) < 2:
        return JaccardStability()

    pairs: list[dict] = []
    jaccards: list[float] = []
    for i in range(1, len(daily_sets)):
        d1, s1 = daily_sets[i - 1]
        d2, s2 = daily_sets[i]
        inter = s1 & s2
        union = s1 | s2
        j = len(inter) / len(union) if union else 0.0
        jaccards.append(j)
        pairs.append(
            {
                "day1": d1,
                "day2": d2,
                "overlap": len(inter),
                "union": len(union),
                "jaccard": round(j, 3),
            }
        )

    return JaccardStability(
        pairs=pairs,
        mean_jaccard=round(statistics.fmean(jaccards), 3),
        median_jaccard=round(statistics.median(jaccards), 3),
        min_jaccard=round(min(jaccards), 3),
        max_jaccard=round(max(jaccards), 3),
    )
