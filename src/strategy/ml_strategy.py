"""機器學習策略 — 使用監督式學習預測漲跌方向產生交易訊號。

支援模型：
- random_forest: Random Forest（穩健，不易過擬合）
- xgboost: XGBoost（準確度高）
- logistic: Logistic Regression（簡單基線，可解釋）

訓練方式：
以 train_ratio 前段資料訓練，後段資料產生訊號。
預測機率 > threshold → 買入，< (1-threshold) → 賣出。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.ml_features import build_ml_features, get_feature_columns
from src.strategy.base import Strategy

logger = logging.getLogger(__name__)


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
    ) -> None:
        super().__init__(stock_id, start_date, end_date)
        self.model_type = model_type
        self.lookback = lookback
        self.train_ratio = train_ratio
        self.threshold = threshold
        self.forward_days = forward_days

    @property
    def name(self) -> str:
        return f"ml_{self.model_type}"

    def _create_model(self):
        """建立 ML 模型。"""
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                )
            except ImportError:
                logger.warning("xgboost 未安裝，fallback 到 random_forest")
                return RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
                )

        if self.model_type == "logistic":
            return LogisticRegression(max_iter=1000, random_state=42)

        # random_forest（預設）
        return RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
        )

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

        # 訓練
        model = self._create_model()
        model.fit(X_train, y_train)

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
        test_acc = model.score(X_test, y[:split_idx + len(X_test)][-len(X_test):])
        logger.info(
            "[%s] ML 模型訓練完成: train_acc=%.2f%%, test_acc=%.2f%%, 特徵=%d",
            self.stock_id, train_acc * 100, test_acc * 100, len(feature_cols),
        )

        return signals
