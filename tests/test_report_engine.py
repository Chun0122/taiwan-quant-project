"""報告引擎評分函數測試 — 4 個 _compute_* 純計算方法。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.report.engine import DailyReportEngine


def _make_engine():
    """建立 DailyReportEngine（不初始化 DB，只用純計算方法）。"""
    # 跳過 __init__ 的 init_db()，直接建立實例的計算方法
    engine = object.__new__(DailyReportEngine)
    engine.weights = {
        "technical": 0.30,
        "chip": 0.30,
        "fundamental": 0.20,
        "ml": 0.20,
    }
    return engine


def _make_snapshot(n=5, **overrides):
    """建立簡單的 snapshot DataFrame。"""
    dates = pd.bdate_range("2024-01-01", periods=n)
    idx = [d.date() for d in dates]
    data = {
        "close": [100.0] * n,
        "rsi_14": [50.0] * n,
        "macd": [0.0] * n,
        "macd_signal": [0.0] * n,
        "sma_20": [100.0] * n,
        "foreign_net": [0.0] * n,
        "trust_net": [0.0] * n,
        "dealer_net": [0.0] * n,
        "margin_balance": [1000.0] * n,
        "yoy_growth": [0.0] * n,
        "mom_growth": [0.0] * n,
    }
    data.update(overrides)
    return pd.DataFrame(data, index=idx)


# ================================================================
# _compute_technical_score
# ================================================================


class TestComputeTechnicalScore:
    def test_low_rsi_high_score(self):
        engine = _make_engine()
        snapshot = _make_snapshot(rsi_14=[30.0] * 5)
        score = engine._compute_technical_score(snapshot)
        # RSI=30 → rsi_score=1.0, macd=0→macd_score=0.5, close==sma20→sma_score=0
        # mean([1.0, 0.5, 0.0]) = 0.5
        assert score >= 0.4

    def test_high_rsi_low_score(self):
        engine = _make_engine()
        snapshot = _make_snapshot(rsi_14=[70.0] * 5)
        score = engine._compute_technical_score(snapshot)
        assert score < 0.5  # RSI=70 → rsi_score=0.0

    def test_macd_above_signal_boost(self):
        engine = _make_engine()
        snapshot = _make_snapshot(macd=[2.0] * 5, macd_signal=[0.0] * 5)
        score1 = engine._compute_technical_score(snapshot)

        snapshot2 = _make_snapshot(macd=[-2.0] * 5, macd_signal=[0.0] * 5)
        score2 = engine._compute_technical_score(snapshot2)
        assert score1 > score2

    def test_close_above_sma20(self):
        engine = _make_engine()
        snapshot = _make_snapshot(close=[105.0] * 5, sma_20=[100.0] * 5)
        score = engine._compute_technical_score(snapshot)
        # close > sma_20 → sma_score=1.0
        assert score > 0.3

    def test_missing_rsi_fallback(self):
        engine = _make_engine()
        snapshot = _make_snapshot()
        snapshot["rsi_14"] = [np.nan] * 5
        score = engine._compute_technical_score(snapshot)
        # 預設 RSI=50 → rsi_score=0.5
        assert 0 <= score <= 1


# ================================================================
# _compute_chip_score
# ================================================================


class TestComputeChipScore:
    def test_all_buy_high_score(self):
        engine = _make_engine()
        snapshot = _make_snapshot(
            foreign_net=[1000.0] * 5,
            trust_net=[500.0] * 5,
            dealer_net=[200.0] * 5,
        )
        score = engine._compute_chip_score(snapshot)
        assert score > 0.7

    def test_all_sell_low_score(self):
        engine = _make_engine()
        snapshot = _make_snapshot(
            foreign_net=[-1000.0] * 5,
            trust_net=[-500.0] * 5,
            dealer_net=[-200.0] * 5,
        )
        score = engine._compute_chip_score(snapshot)
        assert score < 0.3

    def test_margin_decrease_bonus(self):
        engine = _make_engine()
        # 融資餘額從 1000 減到 500，外資全買超 → 好指標
        snapshot = _make_snapshot(
            foreign_net=[1000.0] * 5,
            trust_net=[500.0] * 5,
            dealer_net=[200.0] * 5,
            margin_balance=[1000, 900, 800, 700, 500],
        )
        score = engine._compute_chip_score(snapshot)
        # foreign_ratio=1.0, inst_ratio=1.0, margin_score=1.0
        # 0.4*1.0 + 0.4*1.0 + 0.2*1.0 = 1.0
        assert score >= 0.8

    def test_empty_snapshot(self):
        engine = _make_engine()
        snapshot = pd.DataFrame()
        score = engine._compute_chip_score(snapshot)
        assert score == 0.5


# ================================================================
# _compute_fundamental_score
# ================================================================


class TestComputeFundamentalScore:
    def test_high_yoy_full_score(self):
        engine = _make_engine()
        snapshot = _make_snapshot(yoy_growth=[50.0] * 5)
        score = engine._compute_fundamental_score(snapshot)
        assert score == pytest.approx(1.0)

    def test_zero_yoy(self):
        engine = _make_engine()
        snapshot = _make_snapshot(yoy_growth=[0.0] * 5)
        score = engine._compute_fundamental_score(snapshot)
        assert score == pytest.approx(0.0)

    def test_mom_positive_bonus(self):
        engine = _make_engine()
        snapshot = _make_snapshot(yoy_growth=[25.0] * 5, mom_growth=[5.0] * 5)
        score = engine._compute_fundamental_score(snapshot)
        # yoy_score = 25/50 = 0.5, mom_bonus = 0.1 → 0.6
        assert score == pytest.approx(0.6)

    def test_nan_yoy_fallback(self):
        engine = _make_engine()
        snapshot = _make_snapshot(yoy_growth=[np.nan] * 5)
        score = engine._compute_fundamental_score(snapshot)
        assert score == pytest.approx(0.0)


# ================================================================
# _compute_composite
# ================================================================


class TestComputeComposite:
    def test_weighted_sum(self):
        engine = _make_engine()
        # tech=0.8, chip=0.6, fund=0.4, ml=0.5
        result = engine._compute_composite(0.8, 0.6, 0.4, 0.5)
        expected = 0.30 * 0.8 + 0.30 * 0.6 + 0.20 * 0.4 + 0.20 * 0.5
        assert result == pytest.approx(expected)

    def test_all_zeros(self):
        engine = _make_engine()
        assert engine._compute_composite(0, 0, 0, 0) == pytest.approx(0.0)

    def test_all_ones(self):
        engine = _make_engine()
        assert engine._compute_composite(1, 1, 1, 1) == pytest.approx(1.0)
