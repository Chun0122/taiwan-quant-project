"""測試 src/features/ml_features.py — ML 特徵工程純函數。"""

import numpy as np
import pandas as pd

from src.features.ml_features import (
    build_ml_features,
    compute_shap_importances,
    get_feature_columns,
    select_features_by_importance,
)


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
    df["adx_14"] = 25.0 + np.random.randn(n) * 5
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


# ---------------------------------------------------------------------------
# C1: 新增特徵測試
# ---------------------------------------------------------------------------


class TestInteractionFeatures:
    """C1: 交互特徵測試。"""

    def test_vol_price_synergy(self):
        df = _make_ohlcv(80)
        result = build_ml_features(df)
        assert "vol_price_synergy" in result.columns

    def test_rsi_vol_interaction(self):
        df = _make_ohlcv(80)
        result = build_ml_features(df)
        assert "rsi_vol_interaction" in result.columns

    def test_adx_momentum(self):
        df = _make_ohlcv(80)
        result = build_ml_features(df)
        assert "adx_momentum" in result.columns

    def test_interaction_no_nan(self):
        df = _make_ohlcv(80)
        result = build_ml_features(df)
        for col in ["vol_price_synergy", "rsi_vol_interaction", "adx_momentum"]:
            if col in result.columns:
                assert result[col].isna().sum() == 0


class TestLagFeatures:
    """C1: Lag 特徵測試。"""

    def test_lag_columns_present(self):
        df = _make_ohlcv(80)
        result = build_ml_features(df)
        for lag in [5, 10, 20]:
            assert f"return_1d_lag{lag}" in result.columns
            assert f"volume_ratio_5d_lag{lag}" in result.columns

    def test_lag_values_shifted(self):
        """Lag 特徵應該是對應原始特徵的延遲版本。"""
        df = _make_ohlcv(80)
        result = build_ml_features(df)
        # 在 dropna 後，lag5 的值應該存在且不全為 0
        assert not result["return_1d_lag5"].eq(0).all()

    def test_lag_no_nan(self):
        df = _make_ohlcv(80)
        result = build_ml_features(df)
        for lag in [5, 10, 20]:
            assert result[f"return_1d_lag{lag}"].isna().sum() == 0


class TestSectorRanks:
    """C1: 跨截面特徵測試。"""

    def test_sector_ranks_merged(self):
        df = _make_ohlcv(80)
        # 建立跨截面排名 DataFrame
        ranks = pd.DataFrame(
            {"rsi_rank": np.linspace(0.2, 0.8, len(df))},
            index=df.index,
        )
        result = build_ml_features(df, sector_ranks=ranks)
        assert "rsi_rank" in result.columns
        assert result["rsi_rank"].isna().sum() == 0

    def test_empty_sector_ranks_ignored(self):
        df = _make_ohlcv(80)
        result1 = build_ml_features(df)
        result2 = build_ml_features(df, sector_ranks=pd.DataFrame())
        assert len(result1.columns) == len(result2.columns)

    def test_none_sector_ranks_ignored(self):
        df = _make_ohlcv(80)
        result = build_ml_features(df, sector_ranks=None)
        assert not result.empty


class TestFeatureCount:
    """C1: 驗證擴充後的特徵數量增加。"""

    def test_more_features_than_before(self):
        """擴充後特徵數應大於原始的 ~15 個。"""
        df = _make_ohlcv(80)
        result = build_ml_features(df)
        features = get_feature_columns(result)
        # 原始 ~15 特徵 + 交互 3 + lag 6 = ~24
        assert len(features) >= 20


# ---------------------------------------------------------------------------
# C5: 特徵篩選測試
# ---------------------------------------------------------------------------


class TestSelectFeaturesByImportance:
    """C5: SHAP-based 特徵篩選。"""

    def test_drops_bottom_20_pct(self):
        features = ["a", "b", "c", "d", "e"]
        importances = {"a": 0.5, "b": 0.1, "c": 0.3, "d": 0.05, "e": 0.4}
        selected = select_features_by_importance(features, importances, drop_ratio=0.2)
        # 5 × 0.2 = 1，應移除 1 個最不重要的（d=0.05）
        assert "d" not in selected
        assert len(selected) == 4

    def test_empty_features(self):
        selected = select_features_by_importance([], {}, drop_ratio=0.2)
        assert selected == []

    def test_empty_importances(self):
        features = ["a", "b", "c"]
        selected = select_features_by_importance(features, {}, drop_ratio=0.2)
        assert selected == features

    def test_does_not_drop_all(self):
        """即使 drop_ratio=1.0，也不應丟掉全部。"""
        features = ["a", "b"]
        importances = {"a": 0.5, "b": 0.1}
        selected = select_features_by_importance(features, importances, drop_ratio=1.0)
        assert selected == features  # n_drop >= len => 不丟

    def test_preserves_order(self):
        features = ["x", "y", "z", "w"]
        importances = {"x": 0.8, "y": 0.01, "z": 0.6, "w": 0.4}
        selected = select_features_by_importance(features, importances, drop_ratio=0.25)
        # 移除 y（最不重要）
        assert selected == ["x", "z", "w"]


class TestComputeShapImportances:
    """C5: SHAP 重要性計算純函數。"""

    def test_basic(self):
        shap_values = np.array([[0.1, -0.3], [0.2, 0.4], [-0.1, 0.5]])
        features = ["feat_a", "feat_b"]
        result = compute_shap_importances(shap_values, features)
        assert "feat_a" in result
        assert "feat_b" in result
        # feat_b 的 mean(|SHAP|) 較大，應排在前面
        keys = list(result.keys())
        assert keys[0] == "feat_b"

    def test_wrong_shape(self):
        shap_values = np.array([0.1, 0.2, 0.3])  # 1D
        result = compute_shap_importances(shap_values, ["a", "b", "c"])
        assert result == {}

    def test_mismatched_columns(self):
        shap_values = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = compute_shap_importances(shap_values, ["a"])  # 只有 1 個名稱但 2 列
        assert result == {}

    def test_sorted_descending(self):
        shap_values = np.array([[0.1, 0.5, 0.3]])
        features = ["low", "high", "mid"]
        result = compute_shap_importances(shap_values, features)
        values = list(result.values())
        assert values == sorted(values, reverse=True)
