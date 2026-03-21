"""測試 src/entry_exit.py 共用純函數：compute_atr_stops、compute_entry_trigger、assess_timing。"""

from __future__ import annotations

import pytest

from src.entry_exit import REGIME_ATR_PARAMS, assess_timing, compute_atr_stops, compute_entry_trigger

# ── TestComputeAtrStops ──────────────────────────────────────────


class TestComputeAtrStops:
    """compute_atr_stops 純函數測試。"""

    def test_bull_regime(self) -> None:
        sl, tp = compute_atr_stops(close=100.0, atr14=2.0, regime="bull")
        assert sl == pytest.approx(100.0 - 1.5 * 2.0)  # 97.0
        assert tp == pytest.approx(100.0 + 3.5 * 2.0)  # 107.0

    def test_sideways_regime(self) -> None:
        sl, tp = compute_atr_stops(close=100.0, atr14=2.0, regime="sideways")
        assert sl == pytest.approx(100.0 - 1.5 * 2.0)  # 97.0
        assert tp == pytest.approx(100.0 + 3.0 * 2.0)  # 106.0

    def test_bear_regime(self) -> None:
        sl, tp = compute_atr_stops(close=100.0, atr14=2.0, regime="bear")
        assert sl == pytest.approx(100.0 - 1.2 * 2.0)  # 97.6
        assert tp == pytest.approx(100.0 + 2.5 * 2.0)  # 105.0

    def test_crisis_regime(self) -> None:
        sl, tp = compute_atr_stops(close=100.0, atr14=2.0, regime="crisis")
        assert sl == pytest.approx(100.0 - 1.0 * 2.0)  # 98.0
        assert tp == pytest.approx(100.0 + 1.8 * 2.0)  # 103.6

    def test_unknown_regime_falls_back_to_sideways(self) -> None:
        sl, tp = compute_atr_stops(close=100.0, atr14=2.0, regime="unknown")
        sl_sw, tp_sw = compute_atr_stops(close=100.0, atr14=2.0, regime="sideways")
        assert sl == sl_sw
        assert tp == tp_sw

    def test_atr_zero_returns_none(self) -> None:
        sl, tp = compute_atr_stops(close=100.0, atr14=0.0, regime="bull")
        assert sl is None
        assert tp is None

    def test_atr_negative_returns_none(self) -> None:
        sl, tp = compute_atr_stops(close=100.0, atr14=-1.0, regime="bull")
        assert sl is None
        assert tp is None


# ── TestComputeEntryTrigger ──────────────────────────────────────


class TestComputeEntryTrigger:
    """compute_entry_trigger 純函數測試。"""

    def test_above_sma(self) -> None:
        result = compute_entry_trigger(close=103.0, sma20=100.0, atr_pct=0.03)
        assert "站上均線" in result

    def test_near_sma(self) -> None:
        result = compute_entry_trigger(close=100.0, sma20=100.0, atr_pct=0.03)
        assert "貼近均線" in result

    def test_below_sma(self) -> None:
        result = compute_entry_trigger(close=95.0, sma20=100.0, atr_pct=0.03)
        assert "均線下方" in result

    def test_low_volatility_tag(self) -> None:
        result = compute_entry_trigger(close=100.0, sma20=100.0, atr_pct=0.01)
        assert "低波動" in result

    def test_high_volatility_tag(self) -> None:
        result = compute_entry_trigger(close=100.0, sma20=100.0, atr_pct=0.05)
        assert "高波動謹慎" in result

    def test_crisis_warning_appended(self) -> None:
        result = compute_entry_trigger(close=103.0, sma20=100.0, atr_pct=0.03, regime="crisis")
        assert "崩盤期建議降低部位規模" in result

    def test_sma20_zero_fallback(self) -> None:
        result = compute_entry_trigger(close=100.0, sma20=0.0, atr_pct=0.03)
        assert "均線下方" in result


# ── TestAssessTimingCrisis ───────────────────────────────────────


class TestAssessTimingCrisis:
    """assess_timing 針對 crisis regime 的測試。"""

    def test_crisis_rsi_oversold(self) -> None:
        # RSI ≤ 30 + crisis → 走 else 分支（非 bull/sideways）
        result = assess_timing(rsi14=25.0, close=90.0, sma20=100.0, atr_pct=0.03, regime="crisis")
        assert "企穩" in result

    def test_crisis_rsi_neutral(self) -> None:
        # 30 < RSI < 70 + crisis → 崩盤期專用分支
        result = assess_timing(rsi14=50.0, close=103.0, sma20=100.0, atr_pct=0.03, regime="crisis")
        assert "崩盤期" in result

    def test_crisis_rsi_overbought(self) -> None:
        # RSI ≥ 70 → 超買（優先於 crisis 分支）
        result = assess_timing(rsi14=75.0, close=103.0, sma20=100.0, atr_pct=0.03, regime="crisis")
        assert "超買" in result


# ── TestRegimeAtrParamsConsistency ───────────────────────────────


class TestRegimeAtrParamsConsistency:
    """確認 REGIME_ATR_PARAMS 常數的基本性質。"""

    def test_all_four_regimes_present(self) -> None:
        assert set(REGIME_ATR_PARAMS.keys()) == {"bull", "sideways", "bear", "crisis"}

    def test_all_tuples_have_two_positive_floats(self) -> None:
        for regime, (stop_m, target_m) in REGIME_ATR_PARAMS.items():
            assert stop_m > 0, f"{regime} stop_mult should be positive"
            assert target_m > 0, f"{regime} target_mult should be positive"
            assert target_m > stop_m, f"{regime} target_mult should exceed stop_mult"
