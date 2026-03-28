"""ML 特徵工程 — 從 OHLCV + 技術指標建構機器學習特徵矩陣。

Phase C1 擴充：跨截面特徵（需外部傳入）、交互特徵、lag 特徵。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C1: 特徵工程核心
# ---------------------------------------------------------------------------


def build_ml_features(
    data: pd.DataFrame,
    lookback: int = 20,
    forward_days: int = 5,
    *,
    sector_ranks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """建構 ML 特徵矩陣。

    Args:
        data: 含 OHLCV + 技術指標的寬表 DataFrame（index=日期）
        lookback: 回溯天數（用於滾動特徵）
        forward_days: 預測目標天數（未來 N 天報酬）
        sector_ranks: 跨截面特徵 DataFrame（index=日期），可選欄位：
            rsi_rank, inst_flow_zscore 等。

    Returns:
        含特徵 + 標籤的 DataFrame，已移除 NaN 列
    """
    df = data.copy()

    # ------------------------------------------------------------------ #
    #  動量特徵
    # ------------------------------------------------------------------ #
    for period in [1, 5, 10, 20]:
        df[f"return_{period}d"] = df["close"].pct_change(period)

    # ------------------------------------------------------------------ #
    #  均線比值特徵
    # ------------------------------------------------------------------ #
    if "sma_5" in df.columns and "sma_20" in df.columns:
        df["sma_ratio_5_20"] = df["sma_5"] / df["sma_20"]
    if "sma_10" in df.columns and "sma_60" in df.columns:
        df["sma_ratio_10_60"] = df["sma_10"] / df["sma_60"]

    # 價格相對於 SMA20 的位置
    if "sma_20" in df.columns:
        df["price_vs_sma20"] = df["close"] / df["sma_20"] - 1

    # ------------------------------------------------------------------ #
    #  波動度特徵
    # ------------------------------------------------------------------ #
    daily_ret = df["close"].pct_change()
    df["volatility_10"] = daily_ret.rolling(10).std()
    df["volatility_20"] = daily_ret.rolling(20).std()

    # ------------------------------------------------------------------ #
    #  量價特徵
    # ------------------------------------------------------------------ #
    df["volume_ratio_5d"] = df["volume"] / df["volume"].rolling(5).mean()
    df["volume_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()

    # ------------------------------------------------------------------ #
    #  布林通道位置
    # ------------------------------------------------------------------ #
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = np.where(
            bb_range > 0,
            (df["close"] - df["bb_lower"]) / bb_range,
            0.5,
        )

    # ------------------------------------------------------------------ #
    #  價格區間位置（lookback 天內的高低點位置）
    # ------------------------------------------------------------------ #
    rolling_high = df["high"].rolling(lookback).max()
    rolling_low = df["low"].rolling(lookback).min()
    hl_range = rolling_high - rolling_low
    df["price_position"] = np.where(
        hl_range > 0,
        (df["close"] - rolling_low) / hl_range,
        0.5,
    )

    # ------------------------------------------------------------------ #
    #  直接使用已有技術指標
    # ------------------------------------------------------------------ #
    # rsi_14, macd, macd_signal, macd_hist 已在 data 中（若已 compute）

    # MACD 柱狀圖變化率
    if "macd_hist" in df.columns:
        df["macd_hist_diff"] = df["macd_hist"].diff()

    # ------------------------------------------------------------------ #
    #  C1 擴充：交互特徵
    # ------------------------------------------------------------------ #
    # 量價共振 = volume_ratio × momentum
    if "volume_ratio_5d" in df.columns and "return_5d" in df.columns:
        df["vol_price_synergy"] = df["volume_ratio_5d"] * df["return_5d"]

    # RSI × 波動率：超買/超賣在高波動環境更有意義
    if "rsi_14" in df.columns and "volatility_20" in df.columns:
        df["rsi_vol_interaction"] = df["rsi_14"] * df["volatility_20"]

    # ADX × 動量：趨勢強度確認的動量
    if "adx_14" in df.columns and "return_10d" in df.columns:
        df["adx_momentum"] = df["adx_14"] * df["return_10d"]

    # ------------------------------------------------------------------ #
    #  C1 擴充：Lag 特徵（5/10/20 日 lag）
    # ------------------------------------------------------------------ #
    for lag in [5, 10, 20]:
        df[f"return_1d_lag{lag}"] = df["return_1d"].shift(lag) if "return_1d" in df.columns else np.nan
        df[f"volume_ratio_5d_lag{lag}"] = (
            df["volume_ratio_5d"].shift(lag) if "volume_ratio_5d" in df.columns else np.nan
        )

    # ------------------------------------------------------------------ #
    #  C1 擴充：跨截面特徵（由外部傳入）
    # ------------------------------------------------------------------ #
    if sector_ranks is not None and not sector_ranks.empty:
        # 合併跨截面特徵（產業內 RSI rank, 法人淨買超 z-score 等）
        for col in sector_ranks.columns:
            if col not in df.columns:
                df = df.join(sector_ranks[[col]], how="left")

    # ------------------------------------------------------------------ #
    #  標籤：未來 N 天報酬 > 0 → 1，否則 0
    # ------------------------------------------------------------------ #
    df["future_return"] = df["close"].shift(-forward_days) / df["close"] - 1
    df["label"] = (df["future_return"] > 0).astype(int)

    # 移除 NaN
    df = df.dropna()

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """取得特徵欄位名稱（排除 OHLCV、標籤等非特徵欄位）。"""
    exclude = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "future_return",
        "label",
        "turnover",
        "spread",
    }
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# C5: SHAP-based 特徵篩選純函數
# ---------------------------------------------------------------------------


def select_features_by_importance(
    feature_names: list[str],
    shap_importances: dict[str, float],
    drop_ratio: float = 0.2,
) -> list[str]:
    """根據 SHAP 重要性篩選特徵，移除最不重要的 bottom N%。

    Parameters
    ----------
    feature_names : list[str]
        所有特徵名稱。
    shap_importances : dict[str, float]
        {feature_name: mean(|SHAP value|)} 重要性分數。
    drop_ratio : float
        移除比例（預設 0.2 = bottom 20%）。

    Returns
    -------
    list[str]
        篩選後的特徵名稱（已排除低重要性特徵）。
    """
    if not feature_names or not shap_importances:
        return list(feature_names)

    # 以 SHAP 重要性排序（升序）
    scored = [(f, shap_importances.get(f, 0.0)) for f in feature_names]
    scored.sort(key=lambda x: x[1])

    n_drop = max(1, int(len(scored) * drop_ratio))
    if n_drop >= len(scored):
        # 不要丟掉全部
        return list(feature_names)

    dropped = {f for f, _ in scored[:n_drop]}
    selected = [f for f in feature_names if f not in dropped]

    logger.info(
        "特徵篩選：%d → %d（移除 %d 個低重要性特徵：%s）",
        len(feature_names),
        len(selected),
        len(dropped),
        ", ".join(sorted(dropped)),
    )
    return selected


def compute_shap_importances(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """從 SHAP values 計算各特徵的平均絕對重要性。

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values 矩陣（n_samples × n_features）。
    feature_names : list[str]
        特徵名稱。

    Returns
    -------
    dict[str, float]
        {feature_name: mean(|SHAP value|)}，按重要性降序。
    """
    if shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names):
        return {}

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    result = {name: round(float(val), 6) for name, val in zip(feature_names, mean_abs)}
    # 降序排列
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
