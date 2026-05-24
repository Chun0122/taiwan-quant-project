"""跨模式 mode score 相關性研究（P2 任務 13）。

5 個 discover 模式（momentum/swing/value/dividend/growth）各自對股票算
composite_score。本模組量化：

1. **跨模式相關性**（compute_cross_mode_correlation）：對每個 scan_date 做
   cross-sectional Spearman corr（同一天各股的 mode-A score vs mode-B score），
   再對日期平均。高相關 = 模式冗餘；低/負相關 = 模式互補。

2. **模式重疊**（compute_mode_overlap）：mode 兩兩共同推薦的股票數（每日平均）。

用途（audit）：
  - 若 momentum 與 growth corr=0.9 → 兩者選股高度重疊，'all' 模式 quota 可能浪費
  - 若 value 與 momentum corr<0 → 兩者天然對沖，組合分散效果佳

純函數設計（與 DB 解耦）：load_mode_scores 撈資料，compute_* 接受 DataFrame。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import DiscoveryRecord

logger = logging.getLogger(__name__)

MODES: tuple[str, ...] = ("momentum", "swing", "value", "dividend", "growth")

# 每日每對 mode 至少需 N 檔共同股票才計入當日 corr（避免少樣本雜訊）
DEFAULT_MIN_PAIRS = 5


def load_mode_scores(lookback_days: int = 60, end_date: date | None = None) -> pd.DataFrame:
    """從 DiscoveryRecord 載入 long-format composite_score。

    Returns DataFrame: scan_date / mode / stock_id / composite_score
    """
    end_date = end_date or date.today()
    start_date = end_date - timedelta(days=lookback_days)

    with get_session() as session:
        rows = session.execute(
            select(
                DiscoveryRecord.scan_date,
                DiscoveryRecord.mode,
                DiscoveryRecord.stock_id,
                DiscoveryRecord.composite_score,
            ).where(
                DiscoveryRecord.scan_date >= start_date,
                DiscoveryRecord.scan_date <= end_date,
            )
        ).all()

    return pd.DataFrame(
        [
            {
                "scan_date": r.scan_date,
                "mode": r.mode,
                "stock_id": r.stock_id,
                "composite_score": r.composite_score,
            }
            for r in rows
        ]
    )


def compute_cross_mode_correlation(
    df: pd.DataFrame,
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> pd.DataFrame:
    """每日 cross-sectional Spearman corr 跨模式，再對日期平均（純函數）。

    對每個 scan_date：
      pivot 成 stock × mode 的 composite_score 矩陣，
      對每對 (mode_a, mode_b) 取兩者都有 score 的股票算 Spearman corr。
    僅在當日共同股票數 >= min_pairs 且兩者 std>0 時計入。
    回傳對稱矩陣（index/columns 為出現過的 modes，對角線=1.0，缺資料對=NaN）。
    """
    if df.empty:
        return pd.DataFrame()

    present_modes = [m for m in MODES if m in set(df["mode"].unique())]
    if len(present_modes) < 2:
        return pd.DataFrame()

    pair_corrs: dict[tuple[str, str], list[float]] = defaultdict(list)

    for _scan_date, day_df in df.groupby("scan_date"):
        pivot = day_df.pivot_table(index="stock_id", columns="mode", values="composite_score")
        for i, a in enumerate(present_modes):
            for b in present_modes[i + 1 :]:
                if a not in pivot.columns or b not in pivot.columns:
                    continue
                pair = pivot[[a, b]].dropna()
                if len(pair) < min_pairs:
                    continue
                if pair[a].std() == 0 or pair[b].std() == 0:
                    continue
                c = pair[a].corr(pair[b], method="spearman")
                if not np.isnan(c):
                    pair_corrs[(a, b)].append(float(c))

    matrix = pd.DataFrame(index=present_modes, columns=present_modes, dtype=float)
    for m in present_modes:
        matrix.loc[m, m] = 1.0
    for (a, b), corrs in pair_corrs.items():
        avg = float(np.mean(corrs))
        matrix.loc[a, b] = avg
        matrix.loc[b, a] = avg

    return matrix


def compute_mode_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """mode 兩兩共同推薦股票數（每日平均，純函數）。

    對每個 scan_date 計算各 mode pair 的共同股票數，再對日期平均。
    回傳對稱矩陣（對角線=該 mode 每日平均推薦數）。
    """
    if df.empty:
        return pd.DataFrame()

    present_modes = [m for m in MODES if m in set(df["mode"].unique())]
    if not present_modes:
        return pd.DataFrame()

    pair_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    self_counts: dict[str, list[int]] = defaultdict(list)

    for _scan_date, day_df in df.groupby("scan_date"):
        mode_sids = {m: set(day_df[day_df["mode"] == m]["stock_id"]) for m in present_modes}
        for m in present_modes:
            self_counts[m].append(len(mode_sids[m]))
        for i, a in enumerate(present_modes):
            for b in present_modes[i + 1 :]:
                shared = len(mode_sids[a] & mode_sids[b])
                pair_counts[(a, b)].append(shared)

    matrix = pd.DataFrame(index=present_modes, columns=present_modes, dtype=float)
    for m in present_modes:
        matrix.loc[m, m] = float(np.mean(self_counts[m])) if self_counts[m] else 0.0
    for (a, b), counts in pair_counts.items():
        avg = float(np.mean(counts)) if counts else 0.0
        matrix.loc[a, b] = avg
        matrix.loc[b, a] = avg

    return matrix


def format_cross_mode_report(
    corr_matrix: pd.DataFrame,
    overlap_matrix: pd.DataFrame,
    *,
    lookback_days: int,
    n_scan_dates: int,
) -> str:
    """組成 console 報告字串。"""
    lines: list[str] = []
    lines.append(f"\n{'═' * 64}")
    lines.append(f"  跨模式 Score 相關性研究（lookback={lookback_days}d, {n_scan_dates} 個掃描日）")
    lines.append(f"{'═' * 64}")

    if corr_matrix.empty:
        lines.append("  資料不足（需 ≥2 個模式且每日共同股票 ≥ min_pairs）")
        return "\n".join(lines)

    modes = list(corr_matrix.index)
    header = "          " + "".join(f"{m[:6]:>9s}" for m in modes)
    lines.append("\n  ── Spearman 相關性矩陣（per-date 平均）──")
    lines.append(header)
    for a in modes:
        cells = []
        for b in modes:
            v = corr_matrix.loc[a, b]
            cells.append("    nan  " if pd.isna(v) else f"{v:>+8.3f} ")
        lines.append(f"  {a[:8]:<8s}" + "".join(cells))

    # 高相關 / 互補對警示
    lines.append("\n  ── 解讀 ──")
    flagged = False
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            v = corr_matrix.loc[a, b]
            if pd.isna(v):
                continue
            if v >= 0.7:
                lines.append(f"  🔴 {a} ↔ {b} 高度冗餘 (corr={v:+.3f}) — 'all' quota 可能浪費")
                flagged = True
            elif v <= -0.3:
                lines.append(f"  🟢 {a} ↔ {b} 互補對沖 (corr={v:+.3f}) — 組合分散效果佳")
                flagged = True
    if not flagged:
        lines.append("  （無顯著高相關 ≥0.7 或互補 ≤-0.3 的模式對）")

    if not overlap_matrix.empty:
        lines.append("\n  ── 每日平均共同推薦股票數 ──")
        lines.append(header)
        for a in modes:
            cells = []
            for b in modes:
                v = overlap_matrix.loc[a, b]
                cells.append("    nan  " if pd.isna(v) else f"{v:>8.1f} ")
            lines.append(f"  {a[:8]:<8s}" + "".join(cells))

    return "\n".join(lines)
