"""IC-aware 分數翻轉（compute_ic_aware_score_transform）單元測試。

測試覆蓋：
  - IC 正向 → 原樣保留
  - IC 負向 → score 翻轉為 1 - score
  - |IC| < threshold → 歸零為 0.5（雜訊中性化）
  - 樣本不足 → 保留原始（cold-start 保護）
  - 空輸入 / 缺欄位 / 數值防護
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.discovery.scanner._functions import (
    IC_DAMPEN_WEIGHT_MULT,
    compute_ic_aware_score_transform,
)


def _candidates(**score_cols: list[float]) -> pd.DataFrame:
    n = len(next(iter(score_cols.values())))
    return pd.DataFrame({"stock_id": [f"S{i}" for i in range(n)], **score_cols})


def _ic_row(factor: str, ic: float, n: int = 100) -> dict:
    return {"factor": factor, "ic": ic, "evaluable_count": n, "direction": "x"}


def test_positive_ic_keeps_scores():
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", 0.10)])
    out, actions = compute_ic_aware_score_transform(cands, ic_df)
    assert actions["news_score"] == "kept"
    assert list(out["news_score"]) == [0.2, 0.5, 0.8]


def test_negative_ic_flips_scores():
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", -0.0736)])  # 實測值
    out, actions = compute_ic_aware_score_transform(cands, ic_df)
    assert actions["news_score"] == "flipped"
    # 1 - score
    assert np.allclose(out["news_score"].tolist(), [0.8, 0.5, 0.2])


def test_weak_ic_neutralizes_to_mid():
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", 0.01)])  # |IC| < 0.02 threshold
    out, actions = compute_ic_aware_score_transform(cands, ic_df)
    assert actions["news_score"] == "neutralized"
    assert (out["news_score"] == 0.5).all()


def test_low_samples_preserves_scores():
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", -0.1, n=10)])  # n < min_samples=50
    out, actions = compute_ic_aware_score_transform(cands, ic_df)
    assert actions["news_score"] == "kept_low_samples"
    assert list(out["news_score"]) == [0.2, 0.5, 0.8]


def test_multiple_factors_independent():
    cands = _candidates(
        technical_score=[0.3, 0.6, 0.9],
        news_score=[0.2, 0.5, 0.8],
        chip_score=[0.1, 0.4, 0.7],
    )
    ic_df = pd.DataFrame(
        [
            _ic_row("technical_score", 0.08),  # kept
            _ic_row("news_score", -0.1),  # flipped
            _ic_row("chip_score", 0.005),  # neutralized
        ]
    )
    out, actions = compute_ic_aware_score_transform(cands, ic_df)
    assert actions == {
        "technical_score": "kept",
        "news_score": "flipped",
        "chip_score": "neutralized",
    }
    assert list(out["technical_score"]) == [0.3, 0.6, 0.9]
    assert np.allclose(out["news_score"].tolist(), [0.8, 0.5, 0.2])
    assert (out["chip_score"] == 0.5).all()


def test_flip_clipped_to_unit_interval():
    # 輸入分數超出 [0,1]（防禦性測試：理論上不應發生）
    cands = _candidates(news_score=[-0.1, 0.5, 1.2])
    ic_df = pd.DataFrame([_ic_row("news_score", -0.1)])
    out, _ = compute_ic_aware_score_transform(cands, ic_df)
    # 翻轉後 clip 到 [0, 1]
    assert (out["news_score"] >= 0.0).all()
    assert (out["news_score"] <= 1.0).all()


def test_empty_ic_df_noop():
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    out, actions = compute_ic_aware_score_transform(cands, pd.DataFrame())
    assert actions == {}
    assert list(out["news_score"]) == [0.2, 0.5, 0.8]


def test_missing_column_skipped():
    cands = _candidates(technical_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", -0.1)])  # 欄位不存在
    out, actions = compute_ic_aware_score_transform(cands, ic_df)
    assert "news_score" not in actions
    assert list(out["technical_score"]) == [0.2, 0.5, 0.8]


def test_nan_ic_skipped():
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", float("nan"))])
    out, actions = compute_ic_aware_score_transform(cands, ic_df)
    assert "news_score" not in actions
    assert list(out["news_score"]) == [0.2, 0.5, 0.8]


def test_returns_copy_not_mutation():
    """確保原 candidates 不被修改（purity 保證）。"""
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", -0.1)])
    _ = compute_ic_aware_score_transform(cands, ic_df)
    assert list(cands["news_score"]) == [0.2, 0.5, 0.8]  # 未被 mutated


def test_threshold_boundary():
    """邊界測試：IC == ±threshold 應視為中性（不翻、不保留）。"""
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", 0.02)])  # 正好在邊界
    out, actions = compute_ic_aware_score_transform(cands, ic_df)
    assert actions["news_score"] == "neutralized"  # 邊界視為雜訊
    ic_df_neg = pd.DataFrame([_ic_row("news_score", -0.02)])
    out_neg, actions_neg = compute_ic_aware_score_transform(cands, ic_df_neg)
    assert actions_neg["news_score"] == "neutralized"


# ──────────────────────────────────────────────────────────────────────
# Dampen 模式測試（IC_DAMPEN=1 啟用，弱 IC 改為降權而非歸 0.5）
# ──────────────────────────────────────────────────────────────────────


def test_dampen_mode_weak_ic_keeps_scores():
    """dampen_mode=True：|IC|<threshold 時，分數保留原值，action='dampen'。"""
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", 0.01)])
    out, actions = compute_ic_aware_score_transform(cands, ic_df, dampen_mode=True)
    assert actions["news_score"] == "dampen"
    # 與 baseline 模式對照：分數保留原值，不歸 0.5
    assert list(out["news_score"]) == [0.2, 0.5, 0.8]


def test_dampen_mode_does_not_affect_strong_positive_ic():
    """dampen_mode=True：IC > threshold 時行為與 baseline 一致（kept）。"""
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", 0.10)])
    out, actions = compute_ic_aware_score_transform(cands, ic_df, dampen_mode=True)
    assert actions["news_score"] == "kept"
    assert list(out["news_score"]) == [0.2, 0.5, 0.8]


def test_dampen_mode_does_not_affect_negative_ic_flip():
    """dampen_mode=True：IC < -threshold 時行為與 baseline 一致（flipped）。"""
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", -0.10)])
    out, actions = compute_ic_aware_score_transform(cands, ic_df, dampen_mode=True)
    assert actions["news_score"] == "flipped"
    assert np.allclose(out["news_score"].tolist(), [0.8, 0.5, 0.2])


def test_dampen_mode_low_samples_still_kept():
    """dampen_mode=True：樣本不足時仍是 kept_low_samples，不誤觸發 dampen。"""
    cands = _candidates(news_score=[0.2, 0.5, 0.8])
    ic_df = pd.DataFrame([_ic_row("news_score", 0.01, n=10)])
    out, actions = compute_ic_aware_score_transform(cands, ic_df, dampen_mode=True)
    assert actions["news_score"] == "kept_low_samples"
    assert list(out["news_score"]) == [0.2, 0.5, 0.8]


def test_ic_dampen_weight_mult_constant():
    """常數應為 0.25，與 _base.py 套用 weight×0.25 邏輯一致。"""
    assert IC_DAMPEN_WEIGHT_MULT == 0.25
