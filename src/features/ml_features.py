"""ML 特徵工程 — 從 OHLCV + 技術指標建構機器學習特徵矩陣。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_ml_features(data: pd.DataFrame, lookback: int = 20, forward_days: int = 5) -> pd.DataFrame:
    """建構 ML 特徵矩陣。

    Args:
        data: 含 OHLCV + 技術指標的寬表 DataFrame（index=日期）
        lookback: 回溯天數（用於滾動特徵）
        forward_days: 預測目標天數（未來 N 天報酬）

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
        "open", "high", "low", "close", "volume",
        "future_return", "label",
        "turnover", "spread",
    }
    return [c for c in df.columns if c not in exclude]
