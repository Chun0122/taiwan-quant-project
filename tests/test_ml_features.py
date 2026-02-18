"""測試 src/features/ml_features.py — ML 特徵工程純函數。"""

import numpy as np
import pandas as pd

from src.features.ml_features import build_ml_features, get_feature_columns


def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    """建立 n 天 OHLCV + 基本技術指標的 DataFrame。"""
    dates = pd.bdate_range("2024-01-01", periods=n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.random.randint(500_000, 2_000_000, size=n),
        },
        index=dates.date,
    )

    # 加入技術指標欄位
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_60"] = df["close"].rolling(60).mean()
    df["bb_upper"] = df["sma_20"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["sma_20"] - 2 * df["close"].rolling(20).std()
    df["rsi_14"] = 50.0  # 簡化
    df["macd_hist"] = np.random.randn(n) * 0.1
    return df


class TestBuildMlFeatures:
    def test_output_columns_present(self):
        df = _make_ohlcv(60)
        result = build_ml_features(df)
        expected_cols = [
            "return_1d",
            "return_5d",
            "return_10d",
            "return_20d",
            "volatility_10",
            "volatility_20",
            "volume_ratio_5d",
            "volume_ratio_20d",
            "price_position",
            "label",
            "future_return",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_sma_ratio_columns(self):
        df = _make_ohlcv(60)
        result = build_ml_features(df)
        assert "sma_ratio_5_20" in result.columns
        assert "price_vs_sma20" in result.columns

    def test_bb_position_range(self):
        df = _make_ohlcv(80)
        result = build_ml_features(df, forward_days=3)
        assert "bb_position" in result.columns
        if not result.empty:
            assert result["bb_position"].min() >= -0.5
            assert result["bb_position"].max() <= 1.5

    def test_label_is_binary(self):
        df = _make_ohlcv(60)
        result = build_ml_features(df)
        assert set(result["label"].unique()).issubset({0, 1})

    def test_no_nan_in_output(self):
        df = _make_ohlcv(60)
        result = build_ml_features(df)
        assert result.isna().sum().sum() == 0

    def test_macd_hist_diff(self):
        df = _make_ohlcv(60)
        result = build_ml_features(df)
        assert "macd_hist_diff" in result.columns


class TestGetFeatureColumns:
    def test_excludes_ohlcv_and_label(self):
        df = _make_ohlcv(60)
        result = build_ml_features(df)
        features = get_feature_columns(result)
        excluded = {"open", "high", "low", "close", "volume", "future_return", "label"}
        for col in excluded:
            assert col not in features

    def test_includes_derived_features(self):
        df = _make_ohlcv(60)
        result = build_ml_features(df)
        features = get_feature_columns(result)
        assert "return_1d" in features
        assert "volatility_10" in features
        assert "volume_ratio_5d" in features
