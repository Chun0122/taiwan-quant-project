"""allocator 單元測試 — risk_parity / mean_variance 權重計算。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.allocator import mean_variance_weights, risk_parity_weights


def _make_returns(
    n_days: int = 100,
    vols: list[float] | None = None,
    means: list[float] | None = None,
    stock_ids: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """建立模擬日報酬率 DataFrame。"""
    rng = np.random.RandomState(seed)
    if stock_ids is None:
        stock_ids = ["A", "B"]
    n = len(stock_ids)
    if vols is None:
        vols = [0.02] * n
    if means is None:
        means = [0.001] * n

    data = {}
    for i, sid in enumerate(stock_ids):
        data[sid] = rng.normal(means[i], vols[i], n_days)
    return pd.DataFrame(data)


# ================================================================
# Risk Parity
# ================================================================


class TestRiskParity:
    def test_equal_vol_near_equal_weight(self):
        """相同波動率的股票應得到近似等權重。"""
        returns = _make_returns(n_days=500, vols=[0.02, 0.02])
        weights = risk_parity_weights(returns)
        assert weights["A"] == pytest.approx(0.5, abs=0.05)
        assert weights["B"] == pytest.approx(0.5, abs=0.05)

    def test_different_vol_low_vol_gets_more(self):
        """高波動股票權重應較低。"""
        returns = _make_returns(n_days=500, vols=[0.01, 0.04])
        weights = risk_parity_weights(returns)
        assert weights["A"] > weights["B"]

    def test_weights_sum_to_one(self):
        """權重和 = 1。"""
        returns = _make_returns(n_days=200, stock_ids=["A", "B", "C"], vols=[0.01, 0.02, 0.03])
        weights = risk_parity_weights(returns)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(w > 0 for w in weights.values())

    def test_single_stock(self):
        """單一股票應回傳權重 1.0。"""
        returns = _make_returns(n_days=100, stock_ids=["A"], vols=[0.02])
        weights = risk_parity_weights(returns)
        assert weights["A"] == pytest.approx(1.0)


# ================================================================
# Mean-Variance
# ================================================================


class TestMeanVariance:
    def test_weights_sum_to_one(self):
        """權重和 = 1。"""
        returns = _make_returns(n_days=200, stock_ids=["A", "B", "C"], vols=[0.01, 0.02, 0.03])
        weights = mean_variance_weights(returns)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(w >= 0 for w in weights.values())

    def test_high_return_stock_gets_more(self):
        """高報酬低風險的股票應得到較高權重。"""
        returns = _make_returns(
            n_days=500,
            means=[0.005, 0.0005],
            vols=[0.01, 0.03],
        )
        weights = mean_variance_weights(returns)
        # A 有高報酬 + 低風險 → 應有更高權重
        assert weights["A"] > weights["B"]

    def test_single_stock(self):
        """單一股票應回傳權重 1.0。"""
        returns = _make_returns(n_days=100, stock_ids=["A"], vols=[0.02])
        weights = mean_variance_weights(returns)
        assert weights["A"] == pytest.approx(1.0)


# ================================================================
# Portfolio 整合（fallback 測試）
# ================================================================


class TestFallback:
    def test_fallback_insufficient_data(self):
        """資料不足（< 30 天）時 fallback 到 equal_weight。"""
        from src.backtest.engine import BacktestConfig
        from src.backtest.portfolio import PortfolioBacktestEngine, PortfolioConfig

        engine = object.__new__(PortfolioBacktestEngine)
        engine.config = BacktestConfig()
        engine.portfolio_config = PortfolioConfig(allocation_method="risk_parity")

        # 建立只有 10 天資料的 stock_data
        dates = pd.bdate_range("2024-01-01", periods=10)
        stock_data = {}
        for sid in ["A", "B"]:
            stock_data[sid] = pd.DataFrame(
                {"close": [100 + i for i in range(10)]},
                index=[d.date() for d in dates],
            )

        weights = engine._compute_weights(["A", "B"], stock_data)
        # 應 fallback 到 equal_weight
        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == pytest.approx(0.5)
