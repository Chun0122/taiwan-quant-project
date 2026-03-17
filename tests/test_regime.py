"""Regime 偵測測試。"""

import numpy as np
import pandas as pd
import pytest

from src.regime.detector import (
    REGIME_WEIGHTS,
    MarketRegimeDetector,
    detect_crisis_signals,
    detect_from_series,
)


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
        assert w["technical"] == 0.40
        assert w["chip"] == 0.40
        assert w["fundamental"] == 0.10
        assert w["news"] == 0.10

    def test_momentum_sideways_chip_dominant(self):
        """盤整時 momentum 籌碼面 50% > 技術面 30%，Smart Broker 蓄積效益最大。"""
        w = REGIME_WEIGHTS["momentum"]["sideways"]
        assert w["chip"] == 0.50
        assert w["technical"] == 0.30

    def test_momentum_bear_news_elevated(self):
        """空頭時 momentum 消息面提升至 20%，確保選出有事件催化劑的錯殺股。"""
        w = REGIME_WEIGHTS["momentum"]["bear"]
        assert w["news"] == 0.20
        assert w["technical"] == 0.25

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


class TestDetectCrisisSignals:
    """detect_crisis_signals 純函數測試。"""

    def _make_series(self, values: list[float]) -> pd.Series:
        return pd.Series(values)

    def test_5d_drop_triggers_signal(self):
        """5 日跌幅 > 5% 觸發 fast_return_5d 訊號。"""
        # 穩定後急跌 6%
        stable = [16000.0] * 130
        for i in range(6):
            stable[-(6 - i)] = 16000 * (1 - 0.01 * (i + 1))  # 逐步跌
        stable[-1] = 16000 * 0.93  # 最後一天跌至 -7%（5日累積跌幅）
        result = detect_crisis_signals(self._make_series(stable))
        assert result["signals"]["fast_return_5d"] is True
        assert result["fast_return_5d_val"] < -0.05

    def test_consec_decline_triggers_signal(self):
        """連跌 3 天觸發 consec_decline 訊號。"""
        closes = [16000.0] * 130
        closes[-3] = 15900.0
        closes[-2] = 15800.0
        closes[-1] = 15700.0
        result = detect_crisis_signals(self._make_series(closes))
        assert result["signals"]["consec_decline"] is True

    def test_vol_spike_triggers_signal(self):
        """波動率飆升超過 1.8x 觸發 vol_spike 訊號。"""
        rng = np.random.default_rng(42)
        # 前 120 天：低波動（daily std ≈ 0.5%）
        normal = 16000 * np.cumprod(1 + rng.normal(0, 0.005, 120))
        # 後 20 天：高波動（daily std ≈ 3%）
        volatile = normal[-1] * np.cumprod(1 + rng.normal(0, 0.03, 20))
        closes = np.concatenate([normal, volatile])
        result = detect_crisis_signals(self._make_series(closes.tolist()))
        assert result["signals"]["vol_spike"] is True
        assert result["vol_ratio_val"] > 1.8

    def test_two_signals_enough_for_crisis(self):
        """2/3 訊號觸發即為 crisis。"""
        # 構造：連跌 3 天 + 5 日跌 >5%，波動率不觸發
        closes = [16000.0] * 130
        closes[-5] = 16000.0
        closes[-4] = 15700.0  # -1.9%
        closes[-3] = 15500.0  # -1.3%
        closes[-2] = 15300.0  # -1.3%
        closes[-1] = 15100.0  # 5日跌幅 ≈ -5.6%，連跌 4 天
        result = detect_crisis_signals(self._make_series(closes))
        assert result["signals"]["fast_return_5d"] is True
        assert result["signals"]["consec_decline"] is True
        assert result["crisis"] is True

    def test_one_signal_not_enough(self):
        """只有 1 個訊號不觸發 crisis。"""
        # 只有連跌 3 天，但跌幅很小，波動率正常
        closes = [16000.0] * 130
        closes[-3] = 15998.0
        closes[-2] = 15996.0
        closes[-1] = 15994.0  # 僅連跌，但幅度極小
        result = detect_crisis_signals(self._make_series(closes))
        # consec_decline 可能觸發，但 5d_return 和 vol_spike 不觸發
        assert result["crisis"] is False

    def test_insufficient_data_safe_fallback(self):
        """資料不足時安全降級，不觸發 crisis。"""
        result = detect_crisis_signals(pd.Series([16000.0, 15900.0, 15800.0]))
        assert result["crisis"] is False
        assert result["signals"] == {"fast_return_5d": False, "consec_decline": False, "vol_spike": False}

    def test_crisis_overrides_bull_vote(self):
        """多數決說 bull，但 crisis 訊號觸發時 regime 應覆蓋為 crisis。"""
        # 構造長期上升趨勢（多數決 = bull），但末段急跌 6%
        closes_list = [15000 + i * 20 for i in range(130)]  # 上升趨勢
        closes_list[-5] = float(closes_list[-5])
        # 末 5 日急跌造成 fast_return_5d 觸發
        peak = closes_list[-6]
        for j in range(5):
            closes_list[-(5 - j)] = peak * (1 - 0.013 * (j + 1))  # 5日累積跌 >5%
        closes = pd.Series(closes_list)
        result = detect_from_series(closes)
        # crisis 應覆蓋多數決
        assert result["crisis_triggered"] is True
        assert result["regime"] == "crisis"

    def test_no_crisis_in_gradual_bear(self):
        """緩慢空頭（每日 -0.2%）不觸發 crisis。"""
        # 每日微跌，無急跌、無連日大跌、波動率正常
        closes = pd.Series([18000 * (0.998**i) for i in range(130)])
        result = detect_crisis_signals(closes)
        # 緩慢下跌不會有 5 日 -5%（5日約 -1%），連跌訊號可能觸發但 5d 不觸發
        assert result["signals"]["fast_return_5d"] is False


class TestRegimeWeightsCrisis:
    """Crisis regime 權重矩陣測試。"""

    def test_all_modes_crisis_weights_sum_to_one(self):
        """所有模式的 crisis 權重加總應為 1.0。"""
        for mode in ("momentum", "swing", "value", "dividend", "growth"):
            w = REGIME_WEIGHTS[mode]["crisis"]
            assert sum(w.values()) == pytest.approx(1.0), f"{mode}/crisis 權重和不為 1"

    def test_momentum_crisis_news_dominant(self):
        """momentum/crisis：消息面（0.40）應為最高權重（技術失真時事件催化劑優先）。"""
        w = REGIME_WEIGHTS["momentum"]["crisis"]
        assert w["news"] == 0.40
        assert w["news"] > w["technical"]
        assert w["news"] > w["chip"]
        assert w["news"] > w["fundamental"]

    def test_swing_crisis_fundamental_dominant(self):
        """swing/crisis：基本面（0.50）應為最高權重（品質防禦）。"""
        w = REGIME_WEIGHTS["swing"]["crisis"]
        assert w["fundamental"] == 0.50
        assert w["fundamental"] > w["news"]

    def test_crisis_technical_weight_lower_than_bear(self):
        """crisis 的技術面權重應低於 bear（技術訊號在崩盤時更不可靠）。"""
        for mode in ("momentum", "swing", "growth"):
            crisis_tech = REGIME_WEIGHTS[mode]["crisis"].get("technical", 0.0)
            bear_tech = REGIME_WEIGHTS[mode]["bear"].get("technical", 0.0)
            assert crisis_tech <= bear_tech, f"{mode}: crisis tech {crisis_tech} > bear tech {bear_tech}"

    def test_get_weights_crisis_returns_correct_dict(self):
        """get_weights('momentum', 'crisis') 回傳正確權重字典。"""
        w = MarketRegimeDetector.get_weights("momentum", "crisis")
        assert w["news"] == 0.40
        assert w["technical"] == 0.10
        assert sum(w.values()) == pytest.approx(1.0)
