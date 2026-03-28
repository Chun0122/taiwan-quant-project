"""機器學習策略 — 使用監督式學習預測漲跌方向產生交易訊號。

支援模型：
- random_forest: Random Forest（穩健，不易過擬合）
- xgboost: XGBoost（準確度高）
- logistic: Logistic Regression（簡單基線，可解釋）

訓練方式：
以 train_ratio 前段資料訓練，後段資料產生訊號。
預測機率 > threshold → 買入，< (1-threshold) → 賣出。

Phase C 增強：
- C2: TimeSeriesSplit 交叉驗證（取代簡單 train/test split）
- C3: Optuna 超參數調優
- C4: SHAP 特徵重要性分析
- C5: 基於 SHAP 的特徵篩選
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.features.ml_features import (
    build_ml_features,
    compute_shap_importances,
    get_feature_columns,
    select_features_by_importance,
)
from src.strategy.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C2: TimeSeriesSplit 交叉驗證純函數
# ---------------------------------------------------------------------------


def evaluate_time_series_cv(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    n_splits: int = 5,
) -> dict[str, float]:
    """使用 TimeSeriesSplit 進行交叉驗證。

    Parameters
    ----------
    X : np.ndarray
        特徵矩陣。
    y : np.ndarray
        標籤。
    model : sklearn estimator
        分類器實例。
    n_splits : int
        折數（預設 5）。

    Returns
    -------
    dict[str, float]
        {"mean_accuracy", "std_accuracy", "n_splits"} 交叉驗證結果。
    """
    if len(X) < n_splits * 10:
        # 資料不足以進行有意義的 CV
        return {"mean_accuracy": 0.0, "std_accuracy": 0.0, "n_splits": 0}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy")

    return {
        "mean_accuracy": round(float(np.mean(scores)), 4),
        "std_accuracy": round(float(np.std(scores)), 4),
        "n_splits": n_splits,
    }


# ---------------------------------------------------------------------------
# C3: Optuna 超參數調優純函數
# ---------------------------------------------------------------------------


def _rf_objective(trial, X: np.ndarray, y: np.ndarray, n_splits: int = 3) -> float:
    """Random Forest Optuna objective。"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy")
    return float(np.mean(scores))


def _xgb_objective(trial, X: np.ndarray, y: np.ndarray, n_splits: int = 3) -> float:
    """XGBoost Optuna objective。"""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return 0.0

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy")
    return float(np.mean(scores))


def _logistic_objective(trial, X: np.ndarray, y: np.ndarray, n_splits: int = 3) -> float:
    """Logistic Regression Optuna objective。"""
    params = {
        "C": trial.suggest_float("C", 0.01, 10.0, log=True),
        "max_iter": 1000,
        "random_state": 42,
    }
    model = LogisticRegression(**params)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy")
    return float(np.mean(scores))


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    n_trials: int = 30,
    n_splits: int = 3,
) -> dict[str, Any]:
    """使用 Optuna 調優超參數。

    Parameters
    ----------
    X, y : np.ndarray
        訓練資料。
    model_type : str
        模型類型。
    n_trials : int
        調優次數（預設 30）。
    n_splits : int
        CV 折數。

    Returns
    -------
    dict
        {"best_params": {...}, "best_score": float, "n_trials": int}
    """
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("optuna 未安裝，使用預設超參數")
        return {"best_params": {}, "best_score": 0.0, "n_trials": 0}

    if len(X) < n_splits * 10:
        return {"best_params": {}, "best_score": 0.0, "n_trials": 0}

    objectives = {
        "random_forest": _rf_objective,
        "xgboost": _xgb_objective,
        "logistic": _logistic_objective,
    }
    objective_fn = objectives.get(model_type, _rf_objective)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_fn(trial, X, y, n_splits),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    return {
        "best_params": study.best_params,
        "best_score": round(study.best_value, 4),
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# C4: SHAP 特徵重要性
# ---------------------------------------------------------------------------


def compute_shap_feature_importance(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 500,
) -> dict[str, float]:
    """使用 SHAP TreeExplainer 計算特徵重要性。

    Parameters
    ----------
    model : fitted sklearn/xgb model
        已訓練的模型。
    X : np.ndarray
        特徵矩陣（用於 SHAP 計算）。
    feature_names : list[str]
        特徵名稱。
    max_samples : int
        計算 SHAP 的最大樣本數（避免過慢）。

    Returns
    -------
    dict[str, float]
        {feature_name: mean(|SHAP|)} 按重要性降序排列。
        若 SHAP 未安裝或不支援該模型，回傳空 dict。
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap 未安裝，無法計算特徵重要性")
        return {}

    # 限制樣本數以避免計算過慢
    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    try:
        # 嘗試 TreeExplainer（RF/XGB）
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # 二分類：shap_values 格式因 SHAP 版本而異
        if isinstance(shap_values, list):
            # 舊版 SHAP: list of ndarray，取 class=1
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # 新版 SHAP (>=0.44): 3D array (n_samples, n_features, n_classes)，取 class=1
            sv = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
        else:
            sv = shap_values

        return compute_shap_importances(sv, feature_names)

    except Exception:
        # fallback: 嘗試 LinearExplainer（Logistic）
        try:
            masker = shap.maskers.Independent(X_sample)
            explainer = shap.LinearExplainer(model, masker)
            shap_values = explainer.shap_values(X_sample)
            return compute_shap_importances(shap_values, feature_names)
        except Exception:
            logger.debug("SHAP 計算失敗，跳過特徵重要性分析")
            return {}


# ---------------------------------------------------------------------------
# MLStrategy 主體
# ---------------------------------------------------------------------------


class MLStrategy(Strategy):
    """機器學習策略。"""

    def __init__(
        self,
        stock_id: str,
        start_date: str,
        end_date: str,
        model_type: str = "random_forest",
        lookback: int = 20,
        train_ratio: float = 0.7,
        threshold: float = 0.6,
        forward_days: int = 5,
        adjust_dividend: bool = False,
        *,
        use_optuna: bool = False,
        optuna_trials: int = 30,
        use_shap: bool = False,
        feature_selection: bool = False,
        feature_drop_ratio: float = 0.2,
    ) -> None:
        super().__init__(stock_id, start_date, end_date, adjust_dividend=adjust_dividend)
        self.model_type = model_type
        self.lookback = lookback
        self.train_ratio = train_ratio
        self.threshold = threshold
        self.forward_days = forward_days
        # Phase C options
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.use_shap = use_shap
        self.feature_selection = feature_selection
        self.feature_drop_ratio = feature_drop_ratio
        # 儲存最後一次的分析結果
        self.last_cv_result: dict | None = None
        self.last_shap_importances: dict[str, float] = {}
        self.last_tune_result: dict | None = None

    @property
    def name(self) -> str:
        return f"ml_{self.model_type}"

    def _create_model(self, params: dict | None = None):
        """建立 ML 模型，可使用調優後的參數。"""
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier

                defaults = {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "use_label_encoder": False,
                    "eval_metric": "logloss",
                    "verbosity": 0,
                    "random_state": 42,
                }
                if params:
                    defaults.update(params)
                return XGBClassifier(**defaults)
            except ImportError:
                logger.warning("xgboost 未安裝，fallback 到 random_forest")
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                )

        if self.model_type == "logistic":
            defaults = {"max_iter": 1000, "random_state": 42}
            if params:
                defaults.update(params)
            return LogisticRegression(**defaults)

        # random_forest（預設）
        defaults = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            defaults.update(params)
        return RandomForestClassifier(**defaults)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """訓練模型並產生交易訊號。"""
        # 建構特徵
        df = build_ml_features(data, lookback=self.lookback, forward_days=self.forward_days)

        if len(df) < 50:
            logger.warning("[%s] 特徵資料不足 (%d 筆)，無法訓練", self.stock_id, len(df))
            return pd.Series(0, index=data.index)

        feature_cols = get_feature_columns(df)
        X = df[feature_cols].values
        y = df["label"].values

        # 切分訓練/測試
        split_idx = int(len(df) * self.train_ratio)
        if split_idx < 30:
            logger.warning("[%s] 訓練資料不足", self.stock_id)
            return pd.Series(0, index=data.index)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train = y[:split_idx]
        test_dates = df.index[split_idx:]

        # 標準化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 處理 NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # ── C3: Optuna 超參數調優 ──
        tuned_params = None
        if self.use_optuna:
            tune_result = tune_hyperparameters(
                X_train,
                y_train,
                model_type=self.model_type,
                n_trials=self.optuna_trials,
            )
            self.last_tune_result = tune_result
            if tune_result["best_params"]:
                tuned_params = tune_result["best_params"]
                logger.info(
                    "[%s] Optuna 調優完成: best_score=%.2f%%, params=%s",
                    self.stock_id,
                    tune_result["best_score"] * 100,
                    tuned_params,
                )

        # 訓練
        model = self._create_model(tuned_params)
        model.fit(X_train, y_train)

        # ── C2: TimeSeriesSplit 交叉驗證 ──
        cv_model = self._create_model(tuned_params)
        cv_result = evaluate_time_series_cv(X_train, y_train, cv_model, n_splits=5)
        self.last_cv_result = cv_result
        if cv_result["n_splits"] > 0:
            logger.info(
                "[%s] CV: mean_acc=%.2f%% ± %.2f%% (%d-fold)",
                self.stock_id,
                cv_result["mean_accuracy"] * 100,
                cv_result["std_accuracy"] * 100,
                cv_result["n_splits"],
            )

        # ── C4: SHAP 特徵重要性 ──
        if self.use_shap:
            shap_imp = compute_shap_feature_importance(model, X_train, feature_cols)
            self.last_shap_importances = shap_imp
            if shap_imp:
                top_10 = list(shap_imp.items())[:10]
                logger.info(
                    "[%s] SHAP Top-10: %s",
                    self.stock_id,
                    ", ".join(f"{k}={v:.4f}" for k, v in top_10),
                )

            # ── C5: 特徵篩選（基於 SHAP） ──
            if self.feature_selection and shap_imp:
                selected = select_features_by_importance(feature_cols, shap_imp, drop_ratio=self.feature_drop_ratio)
                if len(selected) < len(feature_cols):
                    # 用篩選後的特徵重新訓練
                    sel_indices = [feature_cols.index(f) for f in selected]
                    X_train = X_train[:, sel_indices]
                    X_test = X_test[:, sel_indices]
                    model = self._create_model(tuned_params)
                    model.fit(X_train, y_train)
                    logger.info(
                        "[%s] 特徵篩選後重新訓練：%d → %d 特徵",
                        self.stock_id,
                        len(feature_cols),
                        len(selected),
                    )
                    feature_cols = selected

        # 預測
        proba = model.predict_proba(X_test)

        # 產生訊號
        signals = pd.Series(0, index=data.index)

        for i, dt in enumerate(test_dates):
            prob_up = proba[i][1] if len(proba[i]) > 1 else proba[i][0]
            if prob_up > self.threshold:
                signals[dt] = 1
            elif prob_up < (1 - self.threshold):
                signals[dt] = -1

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y[: split_idx + len(X_test)][-len(X_test) :])
        logger.info(
            "[%s] ML 模型訓練完成: train_acc=%.2f%%, test_acc=%.2f%%, 特徵=%d",
            self.stock_id,
            train_acc * 100,
            test_acc * 100,
            len(feature_cols),
        )

        return signals
