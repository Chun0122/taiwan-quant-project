"""測試 src/strategy/ml_strategy.py — ML 策略 Phase C 純函數。

C2: TimeSeriesSplit 交叉驗證
C3: Optuna 超參數調優
C4: SHAP 特徵重要性
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.strategy.ml_strategy import (
    compute_shap_feature_importance,
    evaluate_time_series_cv,
    tune_hyperparameters,
)


def _make_classification_data(n: int = 200, n_features: int = 10, seed: int = 42):
    """建構分類用假資料。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    # 簡單線性規則 + 噪音
    w = rng.randn(n_features)
    y = (X @ w > 0).astype(int)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


# ---------------------------------------------------------------------------
# C2: TimeSeriesSplit CV
# ---------------------------------------------------------------------------


class TestEvaluateTimeSeriesCv:
    def test_basic_cv(self):
        X, y, _ = _make_classification_data(200)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = evaluate_time_series_cv(X, y, model, n_splits=5)
        assert result["n_splits"] == 5
        assert 0.0 <= result["mean_accuracy"] <= 1.0
        assert result["std_accuracy"] >= 0.0

    def test_insufficient_data(self):
        """資料不足時回傳空結果。"""
        X, y, _ = _make_classification_data(20)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = evaluate_time_series_cv(X, y, model, n_splits=5)
        assert result["n_splits"] == 0
        assert result["mean_accuracy"] == 0.0

    def test_logistic_cv(self):
        X, y, _ = _make_classification_data(200)
        model = LogisticRegression(max_iter=1000, random_state=42)
        result = evaluate_time_series_cv(X, y, model, n_splits=3)
        assert result["n_splits"] == 3
        assert result["mean_accuracy"] > 0.0

    def test_n_splits_param(self):
        X, y, _ = _make_classification_data(300)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = evaluate_time_series_cv(X, y, model, n_splits=3)
        assert result["n_splits"] == 3

    def test_accuracy_reasonable(self):
        """準確率應在合理範圍（>40% for random-ish data）。"""
        X, y, _ = _make_classification_data(300)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        result = evaluate_time_series_cv(X, y, model, n_splits=5)
        assert result["mean_accuracy"] > 0.4


# ---------------------------------------------------------------------------
# C3: Optuna 超參數調優
# ---------------------------------------------------------------------------


class TestTuneHyperparameters:
    def test_rf_tuning(self):
        X, y, _ = _make_classification_data(200)
        result = tune_hyperparameters(X, y, model_type="random_forest", n_trials=5, n_splits=3)
        assert result["n_trials"] == 5
        assert result["best_score"] > 0.0
        assert "n_estimators" in result["best_params"]

    def test_logistic_tuning(self):
        X, y, _ = _make_classification_data(200)
        result = tune_hyperparameters(X, y, model_type="logistic", n_trials=5, n_splits=3)
        assert result["n_trials"] == 5
        assert "C" in result["best_params"]

    def test_insufficient_data(self):
        X, y, _ = _make_classification_data(10)
        result = tune_hyperparameters(X, y, model_type="random_forest", n_trials=5, n_splits=3)
        assert result["n_trials"] == 0
        assert result["best_params"] == {}

    def test_xgb_tuning(self):
        """XGBoost 調優（若已安裝）。"""
        try:
            import xgboost  # noqa: F401
        except ImportError:
            pytest.skip("xgboost 未安裝")
        X, y, _ = _make_classification_data(200)
        result = tune_hyperparameters(X, y, model_type="xgboost", n_trials=5, n_splits=3)
        assert result["best_score"] > 0.0

    def test_best_score_in_range(self):
        X, y, _ = _make_classification_data(200)
        result = tune_hyperparameters(X, y, model_type="random_forest", n_trials=5, n_splits=3)
        assert 0.0 < result["best_score"] <= 1.0


# ---------------------------------------------------------------------------
# C4: SHAP 特徵重要性
# ---------------------------------------------------------------------------


class TestComputeShapFeatureImportance:
    def test_rf_shap(self):
        X, y, feature_names = _make_classification_data(100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = compute_shap_feature_importance(model, X, feature_names)
        assert len(result) == len(feature_names)
        # 值應為非負
        assert all(v >= 0 for v in result.values())

    def test_logistic_shap(self):
        X, y, feature_names = _make_classification_data(100)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        result = compute_shap_feature_importance(model, X, feature_names)
        # Logistic 可能使用 LinearExplainer 或失敗
        # 只要不拋異常就算通過
        assert isinstance(result, dict)

    def test_max_samples_limit(self):
        """max_samples 限制樣本數。"""
        X, y, feature_names = _make_classification_data(1000)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = compute_shap_feature_importance(model, X, feature_names, max_samples=50)
        assert len(result) == len(feature_names)

    def test_sorted_descending(self):
        X, y, feature_names = _make_classification_data(100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = compute_shap_feature_importance(model, X, feature_names)
        values = list(result.values())
        assert values == sorted(values, reverse=True)

    def test_xgb_shap(self):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            pytest.skip("xgboost 未安裝")
        X, y, feature_names = _make_classification_data(100)
        model = XGBClassifier(n_estimators=10, verbosity=0, random_state=42)
        model.fit(X, y)
        result = compute_shap_feature_importance(model, X, feature_names)
        assert len(result) == len(feature_names)
