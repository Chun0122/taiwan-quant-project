"""因子消融測試（Factor Ablation）— 量化各因子對選股結果的邊際貢獻。

兩層消融：
  1. 維度級（dimension）：逐一歸零 technical / chip / fundamental / news，
     比較排名位移與 Spearman ρ。
  2. 子因子級（sub-factor）：在單一維度內逐一移除子因子，
     觀察維度分數變化。

設計為純函數，不直接碰 DB——接收 scanner.run() 的結果進行分析。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ──────────────────────────────────────────────────────────
#  資料結構
# ──────────────────────────────────────────────────────────


@dataclass
class DimensionAblationResult:
    """單一維度消融的結果。"""

    removed_dimension: str
    original_weights: dict[str, float]
    ablated_weights: dict[str, float]
    rank_correlation: float  # Spearman ρ vs baseline
    mean_rank_shift: float  # 平均排名位移（絕對值）
    max_rank_shift: int  # 最大排名位移
    stocks_dropped: list[str]  # 基線有但消融後掉出 top_n 的股票
    stocks_added: list[str]  # 消融後新進 top_n 的股票
    top5_changes: list[dict]  # 前 5 名的變動明細


@dataclass
class SubFactorAblationResult:
    """單一子因子消融的結果。"""

    dimension: str
    removed_factor: str
    score_correlation: float  # 維度分數 Spearman ρ vs baseline
    mean_score_change: float  # 平均分數變化
    top_movers: list[dict]  # 排名變動最大的股票


@dataclass
class AblationReport:
    """完整消融報告。"""

    mode: str
    regime: str
    baseline_top_n: int
    dimension_results: list[DimensionAblationResult] = field(default_factory=list)
    sub_factor_results: list[SubFactorAblationResult] = field(default_factory=list)


# ──────────────────────────────────────────────────────────
#  維度級消融（純函數）
# ──────────────────────────────────────────────────────────


def redistribute_weights(
    weights: dict[str, float],
    remove_key: str,
) -> dict[str, float]:
    """移除一個維度後，將其權重按比例分配給剩餘維度。

    >>> redistribute_weights({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}, "a")
    {'b': 0.5, 'c': 0.333..., 'd': 0.166...}
    """
    if remove_key not in weights:
        return dict(weights)

    remaining = {k: v for k, v in weights.items() if k != remove_key}
    total = sum(remaining.values())
    if total <= 0:
        # 極端情況：只有一個維度
        return {k: 1.0 / len(remaining) for k in remaining} if remaining else {}
    return {k: v / total for k, v in remaining.items()}


def recompute_composite(
    scored_df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """用給定權重重新計算 composite_score。

    scored_df 需含 {key}_score 欄位。
    """
    composite = pd.Series(0.0, index=scored_df.index)
    for key, weight in weights.items():
        col = f"{key}_score"
        if col in scored_df.columns:
            composite += pd.to_numeric(scored_df[col], errors="coerce").fillna(0.5) * weight
    return composite


def run_dimension_ablation(
    scored_df: pd.DataFrame,
    baseline_weights: dict[str, float],
    top_n: int = 20,
) -> list[DimensionAblationResult]:
    """逐一移除每個維度，比較排名變化。

    Args:
        scored_df: scanner._score_candidates() 輸出，含 stock_id + *_score 欄位
        baseline_weights: 原始 regime 權重（如 {"technical": 0.4, "chip": 0.4, ...}）
        top_n: 比較前 N 名

    Returns:
        每個維度一筆 DimensionAblationResult
    """
    if scored_df.empty or not baseline_weights:
        return []

    # 基線排名
    baseline_scores = recompute_composite(scored_df, baseline_weights)
    df = scored_df.copy()
    df["_baseline_score"] = baseline_scores
    df = df.sort_values("_baseline_score", ascending=False).reset_index(drop=True)
    df["_baseline_rank"] = range(1, len(df) + 1)
    baseline_top = set(df.head(top_n)["stock_id"])

    results = []
    for dim in baseline_weights:
        col = f"{dim}_score"
        if col not in scored_df.columns:
            continue

        ablated_w = redistribute_weights(baseline_weights, dim)
        ablated_scores = recompute_composite(scored_df, ablated_w)

        df["_ablated_score"] = ablated_scores
        df = df.sort_values("_ablated_score", ascending=False).reset_index(drop=True)
        df["_ablated_rank"] = range(1, len(df) + 1)

        # 排名映射
        rank_map = df.set_index("stock_id")[["_baseline_rank", "_ablated_rank"]]
        rank_shifts = (rank_map["_ablated_rank"] - rank_map["_baseline_rank"]).abs()

        # Spearman ρ
        valid = rank_map.dropna()
        if len(valid) >= 3:
            rho, _ = spearmanr(valid["_baseline_rank"], valid["_ablated_rank"])
        else:
            rho = float("nan")

        ablated_top = set(df.head(top_n)["stock_id"])
        dropped = sorted(baseline_top - ablated_top)
        added = sorted(ablated_top - baseline_top)

        # 前 5 名變動
        top5_changes = []
        for _, row in df.head(5).iterrows():
            sid = row["stock_id"]
            bl_rank = int(rank_map.loc[sid, "_baseline_rank"]) if sid in rank_map.index else 0
            ab_rank = int(rank_map.loc[sid, "_ablated_rank"]) if sid in rank_map.index else 0
            top5_changes.append(
                {
                    "stock_id": sid,
                    "baseline_rank": bl_rank,
                    "ablated_rank": ab_rank,
                    "shift": ab_rank - bl_rank,
                }
            )

        results.append(
            DimensionAblationResult(
                removed_dimension=dim,
                original_weights=dict(baseline_weights),
                ablated_weights=ablated_w,
                rank_correlation=round(rho, 4) if not np.isnan(rho) else 0.0,
                mean_rank_shift=round(float(rank_shifts.mean()), 2),
                max_rank_shift=int(rank_shifts.max()),
                stocks_dropped=dropped,
                stocks_added=added,
                top5_changes=top5_changes,
            )
        )

        # 恢復排序
        df = df.sort_values("_baseline_rank").reset_index(drop=True)

    return results


# ──────────────────────────────────────────────────────────
#  子因子級消融（純函數）
# ──────────────────────────────────────────────────────────

# 各維度的子因子前綴映射
DIMENSION_PREFIXES: dict[str, str] = {
    "technical": "tech_",
    "chip": "chip_",
    "fundamental": "fund_",
    "news": "news_",
}


def run_sub_factor_ablation(
    sub_factor_df: pd.DataFrame,
    dimension: str,
) -> list[SubFactorAblationResult]:
    """在指定維度內逐一移除子因子，比較維度分數變化。

    Args:
        sub_factor_df: scanner.get_sub_factor_df() 輸出，含 stock_id + 子因子 rank 欄位
        dimension: 要分析的維度（technical / chip）

    Returns:
        每個子因子一筆 SubFactorAblationResult
    """
    if sub_factor_df.empty:
        return []

    prefix = DIMENSION_PREFIXES.get(dimension, f"{dimension}_")
    factor_cols = [c for c in sub_factor_df.columns if c.startswith(prefix)]

    if len(factor_cols) < 2:
        return []

    # 基線：所有子因子等權平均
    baseline_score = sub_factor_df[factor_cols].mean(axis=1)

    results = []
    for remove_col in factor_cols:
        remaining = [c for c in factor_cols if c != remove_col]
        ablated_score = sub_factor_df[remaining].mean(axis=1)

        # Spearman ρ
        valid_mask = baseline_score.notna() & ablated_score.notna()
        if valid_mask.sum() >= 3:
            rho, _ = spearmanr(baseline_score[valid_mask], ablated_score[valid_mask])
        else:
            rho = float("nan")

        score_diff = ablated_score - baseline_score

        # 排名變動最大者
        baseline_ranks = baseline_score.rank(ascending=False)
        ablated_ranks = ablated_score.rank(ascending=False)
        rank_shift = (ablated_ranks - baseline_ranks).abs()

        top_movers = []
        if "stock_id" in sub_factor_df.columns:
            top_idx = rank_shift.nlargest(3).index
            for idx in top_idx:
                top_movers.append(
                    {
                        "stock_id": sub_factor_df.loc[idx, "stock_id"],
                        "baseline_rank": int(baseline_ranks.loc[idx]),
                        "ablated_rank": int(ablated_ranks.loc[idx]),
                        "shift": int(ablated_ranks.loc[idx] - baseline_ranks.loc[idx]),
                    }
                )

        results.append(
            SubFactorAblationResult(
                dimension=dimension,
                removed_factor=remove_col,
                score_correlation=round(rho, 4) if not np.isnan(rho) else 0.0,
                mean_score_change=round(float(score_diff.abs().mean()), 4),
                top_movers=top_movers,
            )
        )

    return results


# ──────────────────────────────────────────────────────────
#  歷史績效消融（結合 DiscoveryRecord）
# ──────────────────────────────────────────────────────────


def compute_ablation_performance(
    records_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    baseline_weights: dict[str, float],
    holding_days: int = 5,
    top_n: int = 20,
    select_ratio: float = 0.5,
) -> pd.DataFrame:
    """用歷史推薦記錄，比較消融後的假設績效差異。

    v2 修正：baseline 與消融均以 select_ratio（預設 50%）cutoff 取 top-N，
    確保消融後不同的分數排序會導致不同的股票被選入/排除。
    原 v1 問題：每日推薦數 ≈ top_n，消融後全選名單不變。

    records_df 需含：scan_date, stock_id, close,
                     technical_score, chip_score, fundamental_score, news_score
    prices_df 需含：stock_id, date, close

    Returns:
        DataFrame(removed_dimension, win_rate, avg_return, baseline_win_rate, baseline_avg_return,
                  win_rate_delta, avg_return_delta, selection_overlap)
    """
    if records_df.empty or prices_df.empty:
        return pd.DataFrame()

    score_cols = [f"{k}_score" for k in baseline_weights if f"{k}_score" in records_df.columns]
    if not score_cols:
        return pd.DataFrame()

    # 計算每筆推薦的 N 日報酬
    records_with_return = _attach_forward_returns(records_df, prices_df, holding_days)
    if records_with_return.empty:
        return pd.DataFrame()

    ret_col = f"return_{holding_days}d"

    # 用原始權重計算 baseline composite score
    records_with_return = records_with_return.copy()
    records_with_return["_baseline_score"] = recompute_composite(records_with_return, baseline_weights)

    # Baseline: 每日 top select_ratio 選股
    baseline_returns = []
    baseline_selections: dict = {}  # scan_date → set(stock_id)
    for scan_date, group in records_with_return.groupby("scan_date"):
        n = max(1, int(len(group) * select_ratio))
        n = min(n, top_n)
        top = group.nlargest(n, "_baseline_score")
        valid = top[ret_col].dropna()
        baseline_returns.extend(valid.tolist())
        baseline_selections[scan_date] = set(top["stock_id"].tolist())

    if not baseline_returns:
        return pd.DataFrame()

    baseline_s = pd.Series(baseline_returns)
    baseline_wr = float((baseline_s > 0).mean())
    baseline_avg = float(baseline_s.mean())

    results = []

    # baseline 自身
    results.append(
        {
            "removed_dimension": "(none — baseline)",
            "win_rate": baseline_wr,
            "avg_return": baseline_avg,
            "baseline_win_rate": baseline_wr,
            "baseline_avg_return": baseline_avg,
            "win_rate_delta": 0.0,
            "avg_return_delta": 0.0,
            "selection_overlap": 1.0,
        }
    )

    # 逐維度消融
    for dim in baseline_weights:
        ablated_w = redistribute_weights(baseline_weights, dim)

        ablated_returns = []
        overlap_ratios = []
        for scan_date, group in records_with_return.groupby("scan_date"):
            n = max(1, int(len(group) * select_ratio))
            n = min(n, top_n)
            ablated_score = recompute_composite(group, ablated_w)
            group = group.copy()
            group["_ablated_score"] = ablated_score
            top = group.nlargest(n, "_ablated_score")
            valid = top[ret_col].dropna()
            ablated_returns.extend(valid.tolist())

            # 計算選股重疊率
            ablated_set = set(top["stock_id"].tolist())
            baseline_set = baseline_selections.get(scan_date, set())
            if baseline_set:
                overlap = len(ablated_set & baseline_set) / len(baseline_set)
                overlap_ratios.append(overlap)

        if not ablated_returns:
            continue

        ablated_s = pd.Series(ablated_returns)
        abl_wr = float((ablated_s > 0).mean())
        abl_avg = float(ablated_s.mean())
        avg_overlap = float(np.mean(overlap_ratios)) if overlap_ratios else 1.0

        results.append(
            {
                "removed_dimension": dim,
                "win_rate": abl_wr,
                "avg_return": abl_avg,
                "baseline_win_rate": baseline_wr,
                "baseline_avg_return": baseline_avg,
                "win_rate_delta": round(abl_wr - baseline_wr, 4),
                "avg_return_delta": round(abl_avg - baseline_avg, 4),
                "selection_overlap": round(avg_overlap, 4),
            }
        )

    return pd.DataFrame(results)


def _attach_forward_returns(
    records_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    holding_days: int,
) -> pd.DataFrame:
    """為每筆推薦附上 N 日遠期報酬。"""
    results = []
    for _, rec in records_df.iterrows():
        scan_d = rec["scan_date"]
        sid = rec["stock_id"]
        entry = rec["close"]
        if entry is None or entry <= 0:
            continue

        future = (
            prices_df[(prices_df["stock_id"] == sid) & (prices_df["date"] > scan_d)]
            .sort_values("date")
            .head(holding_days)
        )

        ret = None
        if len(future) >= holding_days:
            exit_close = float(future.iloc[-1]["close"])
            ret = (exit_close - entry) / entry

        row = rec.to_dict()
        row[f"return_{holding_days}d"] = ret
        results.append(row)

    return pd.DataFrame(results) if results else pd.DataFrame()


# ──────────────────────────────────────────────────────────
#  影響力分級
# ──────────────────────────────────────────────────────────


def _classify_dimension_impact(rho: float) -> str:
    """維度級消融分級。

    維度 ρ 跨度大（-0.05 ~ 0.95）：移除整個維度對排名變動明顯。
    四級門檻：< 0.50「極高」、< 0.85「高」、< 0.95「中」、其餘「低」。
    """
    if rho < 0.50:
        return "極高"
    if rho < 0.85:
        return "高"
    if rho < 0.95:
        return "中"
    return "低"


def _classify_subfactor_impact(rho: float, mean_shift: float) -> str:
    """子因子級消融分級（雙指標）。

    子因子 ρ 跨度小（多落在 0.90~0.99），單看 ρ 易全歸「低」。
    輔以 `mean_shift`（分數絕對偏移）加強判斷：
      - rho < 0.90 或 mean_shift > 0.06 → 高
      - rho < 0.95 或 mean_shift > 0.04 → 中
      - rho < 0.98 或 mean_shift > 0.02 → 低
      - 其餘 → 微
    """
    if rho < 0.90 or mean_shift > 0.06:
        return "高"
    if rho < 0.95 or mean_shift > 0.04:
        return "中"
    if rho < 0.98 or mean_shift > 0.02:
        return "低"
    return "微"


# ──────────────────────────────────────────────────────────
#  報告格式化
# ──────────────────────────────────────────────────────────


def format_ablation_report(report: AblationReport) -> str:
    """將消融報告格式化為 console 輸出文字。"""
    lines: list[str] = []
    lines.append(f"\n{'=' * 75}")
    lines.append(f"因子消融測試報告 [{report.mode}] — Regime: {report.regime}")
    lines.append(f"基線 Top {report.baseline_top_n} 名")
    lines.append(f"{'=' * 75}")

    # 維度級
    if report.dimension_results:
        lines.append(f"\n{'─' * 70}")
        lines.append("維度級消融（逐一移除維度，比較排名變化）")
        lines.append(f"{'─' * 70}")
        lines.append(
            f"{'移除維度':<15}  {'Spearman ρ':>10}  {'平均位移':>8}  {'最大位移':>8}  {'掉出':>4}  {'新增':>4}"
        )

        for r in report.dimension_results:
            lines.append(
                f"{r.removed_dimension:<15}  {r.rank_correlation:>10.4f}  "
                f"{r.mean_rank_shift:>8.1f}  {r.max_rank_shift:>8}  "
                f"{len(r.stocks_dropped):>4}  {len(r.stocks_added):>4}"
            )

        # 影響力排序（ρ 越低 = 影響力越大）
        sorted_dims = sorted(report.dimension_results, key=lambda x: x.rank_correlation)
        lines.append("\n  影響力排序（ρ 越低 = 該維度對排名影響越大）：")
        for i, r in enumerate(sorted_dims, 1):
            impact = _classify_dimension_impact(r.rank_correlation)
            lines.append(f"    {i}. {r.removed_dimension}（ρ={r.rank_correlation:.4f}，影響 {impact}）")

        # 前 5 名異動明細
        for r in report.dimension_results:
            if r.stocks_dropped or r.stocks_added:
                lines.append(f"\n  移除 [{r.removed_dimension}] 後 Top5 變化：")
                for c in r.top5_changes:
                    delta = c["shift"]
                    arrow = "↑" if delta < 0 else "↓" if delta > 0 else "—"
                    lines.append(
                        f"    #{c['ablated_rank']:>2}（原 #{c['baseline_rank']:>2} {arrow}{abs(delta):>2}） {c['stock_id']}"
                    )

    # 子因子級
    if report.sub_factor_results:
        dims_seen = set()
        for r in report.sub_factor_results:
            if r.dimension not in dims_seen:
                dims_seen.add(r.dimension)
                lines.append(f"\n{'─' * 70}")
                lines.append(f"子因子消融 [{r.dimension}]（逐一移除子因子，比較維度分數）")
                lines.append(f"{'─' * 70}")
                lines.append(f"{'移除因子':<25}  {'分數 ρ':>8}  {'平均偏移':>8}  {'影響':>4}")

            impact = _classify_subfactor_impact(r.score_correlation, r.mean_score_change)
            lines.append(
                f"{r.removed_factor:<25}  {r.score_correlation:>8.4f}  {r.mean_score_change:>8.4f}  {impact:>4}"
            )

    if not report.dimension_results and not report.sub_factor_results:
        lines.append("\n  無足夠資料進行消融分析")

    lines.append("")
    return "\n".join(lines)
