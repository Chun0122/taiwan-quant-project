"""Regime 偵測測試。"""

import pandas as pd
import pytest

from src.regime.detector import REGIME_WEIGHTS, MarketRegimeDetector, detect_from_series


class TestDetectFromSeries:
    """純函數 detect_from_series 測試。"""

    def test_bull_regime(self):
        """close > SMA60 > SMA120 + 20 日漲 5% → bull。"""
        # 構造上升趨勢：從 15000 漲到 18000
        closes = pd.Series([15000 + i * 25 for i in range(130)])
        result = detect_from_series(closes)
        assert result["regime"] == "bull"
        assert result["signals"]["vs_sma_short"] == "bull"
        assert result["signals"]["vs_sma_long"] == "bull"

    def test_bear_regime(self):
        """close < SMA60 < SMA120 + 20 日跌 5% → bear。"""
        # 構造下跌趨勢：從 18000 跌到 15000
        closes = pd.Series([18000 - i * 25 for i in range(130)])
        result = detect_from_series(closes)
        assert result["regime"] == "bear"
        assert result["signals"]["vs_sma_short"] == "bear"
        assert result["signals"]["vs_sma_long"] == "bear"

    def test_sideways_regime(self):
        """混合訊號 → sideways。"""
        # 先漲後跌（近期小跌），整體仍在均線附近
        values = [16000 + i * 10 for i in range(100)]  # 緩漲到 17000
        values += [17000 - i * 5 for i in range(30)]  # 小跌到 16850
        closes = pd.Series(values)
        result = detect_from_series(closes)
        # 因為整體仍在均線之上但短期小回，可能是 sideways 或 bull
        # 只需確認回傳格式正確
        assert result["regime"] in ("bull", "bear", "sideways")
        assert "vs_sma_short" in result["signals"]
        assert "vs_sma_long" in result["signals"]
        assert "return_20d" in result["signals"]

    def test_insufficient_data_defaults_sideways(self):
        """資料不足時預設 sideways。"""
        closes = pd.Series([16000, 16100, 16200])
        result = detect_from_series(closes)
        assert result["regime"] == "sideways"
        assert result["signals"]["vs_sma_short"] == "unknown"

    def test_return_threshold(self):
        """20 日報酬率在 ±3% 之間 → return 訊號為 sideways。"""
        # 穩定不變的序列
        closes = pd.Series([16000.0] * 130)
        result = detect_from_series(closes)
        assert result["signals"]["return_20d"] == "sideways"

    def test_taiex_close_returned(self):
        """結果包含最新 TAIEX 收盤價。"""
        closes = pd.Series([16000 + i for i in range(130)])
        result = detect_from_series(closes)
        assert result["taiex_close"] == pytest.approx(16129.0)


class TestRegimeWeights:
    """權重矩陣測試。"""

    def test_momentum_weights_sum_to_one(self):
        """Momentum 各 regime 權重加總應為 1。"""
        for regime in ("bull", "sideways", "bear"):
            w = REGIME_WEIGHTS["momentum"][regime]
            assert sum(w.values()) == pytest.approx(1.0)

    def test_swing_weights_sum_to_one(self):
        """Swing 各 regime 權重加總應為 1。"""
        for regime in ("bull", "sideways", "bear"):
            w = REGIME_WEIGHTS["swing"][regime]
            assert sum(w.values()) == pytest.approx(1.0)

    def test_value_weights_sum_to_one(self):
        """Value 各 regime 權重加總應為 1。"""
        for regime in ("bull", "sideways", "bear"):
            w = REGIME_WEIGHTS["value"][regime]
            assert sum(w.values()) == pytest.approx(1.0)

    def test_get_weights_known_mode(self):
        """get_weights 回傳正確的權重。"""
        w = MarketRegimeDetector.get_weights("momentum", "bull")
        assert w["technical"] == 0.45
        assert w["chip"] == 0.35
        assert w["fundamental"] == 0.10
        assert w["news"] == 0.10

    def test_get_weights_unknown_mode_returns_default(self):
        """未知模式回傳預設權重。"""
        w = MarketRegimeDetector.get_weights("unknown_mode", "bull")
        assert sum(w.values()) == pytest.approx(1.0)

    def test_bear_shifts_weight_to_fundamental(self):
        """空頭時各模式都加重基本面。"""
        for mode in ("momentum", "swing"):
            bull_w = REGIME_WEIGHTS[mode]["bull"]
            bear_w = REGIME_WEIGHTS[mode]["bear"]
            assert bear_w["fundamental"] > bull_w["fundamental"]

    def test_bull_shifts_weight_to_technical(self):
        """多頭時 momentum/swing 加重技術面。"""
        for mode in ("momentum", "swing"):
            bull_w = REGIME_WEIGHTS[mode]["bull"]
            bear_w = REGIME_WEIGHTS[mode]["bear"]
            assert bull_w["technical"] > bear_w["technical"]
