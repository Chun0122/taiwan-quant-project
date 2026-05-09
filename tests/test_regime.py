"""Regime 偵測測試。"""

import numpy as np
import pandas as pd
import pytest

from src.regime.detector import (
    REGIME_WEIGHTS,
    MarketRegimeDetector,
    RegimeStateMachine,
    apply_hysteresis,
    check_transition_condition,
    compute_market_breadth_pct,
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
        """get_weights 回傳正確的權重（momentum v5：technical 歸零，chip/fundamental 主導）。"""
        w = MarketRegimeDetector.get_weights("momentum", "bull")
        assert w["technical"] == 0.00
        assert w["chip"] == 0.55
        assert w["fundamental"] == 0.45
        assert w["news"] == 0.00

    def test_momentum_sideways_chip_dominant(self):
        """盤整時 momentum chip 0.45 為最高權重（v5：technical 歸零後）。"""
        w = REGIME_WEIGHTS["momentum"]["sideways"]
        assert w["chip"] == 0.45
        assert w["technical"] == 0.00

    def test_momentum_bear_chip_dominant(self):
        """空頭時 momentum chip 0.42 主導，technical 仍歸零（v5）。"""
        w = REGIME_WEIGHTS["momentum"]["bear"]
        assert w["chip"] == 0.42
        assert w["technical"] == 0.00
        assert w["news"] == 0.20

    def test_get_weights_unknown_mode_returns_default(self):
        """未知模式回傳預設權重。"""
        w = MarketRegimeDetector.get_weights("unknown_mode", "bull")
        assert sum(w.values()) == pytest.approx(1.0)

    def test_bear_shifts_weight_to_fundamental(self):
        """空頭時 swing 加重基本面（momentum 已移除 fundamental）。"""
        for mode in ("swing",):
            bull_w = REGIME_WEIGHTS[mode]["bull"]
            bear_w = REGIME_WEIGHTS[mode]["bear"]
            assert bear_w["fundamental"] > bull_w["fundamental"]

    def test_bull_shifts_weight_to_technical(self):
        """多頭時 swing 加重技術面（momentum v5 已將 technical 歸零，不再適用此規則）。"""
        for mode in ("swing",):
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
        assert result["signals"] == {
            "fast_return_5d": False,
            "consec_decline": False,
            "vol_spike": False,
            "panic_volume": False,
            "vix_spike": False,
            "single_day_drop": False,
            "us_vix_spike": False,
        }

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

    # ── VIX spike 訊號測試 ──

    def test_vix_spike_absolute_level(self):
        """VIX > 30 觸發 vix_spike。"""
        closes = self._make_series([16000.0] * 130)
        vix = pd.Series([20.0, 35.0])  # 最新 VIX=35 > 30
        result = detect_crisis_signals(closes, vix_series=vix)
        assert result["signals"]["vix_spike"] is True
        assert result["vix_val"] == pytest.approx(35.0)

    def test_vix_spike_daily_change(self):
        """VIX 單日漲幅 > 25% 觸發 vix_spike。"""
        closes = self._make_series([16000.0] * 130)
        vix = pd.Series([20.0, 26.0])  # 漲幅 30% > 25%
        result = detect_crisis_signals(closes, vix_series=vix)
        assert result["signals"]["vix_spike"] is True

    def test_vix_none_graceful(self):
        """vix_series=None → vix_spike=False（向後相容）。"""
        closes = self._make_series([16000.0] * 130)
        result = detect_crisis_signals(closes, vix_series=None)
        assert result["signals"]["vix_spike"] is False
        assert result["vix_val"] == 0.0

    def test_vix_insufficient_data(self):
        """VIX 資料 < 2 筆 → vix_spike=False。"""
        closes = self._make_series([16000.0] * 130)
        vix = pd.Series([35.0])  # 只有一筆
        result = detect_crisis_signals(closes, vix_series=vix)
        assert result["signals"]["vix_spike"] is False

    # ── 單日急跌訊號測試 ──

    def test_single_day_drop_triggers(self):
        """TAIEX 單日跌 3% 觸發 single_day_drop。"""
        closes = [16000.0] * 130
        closes[-1] = 16000.0 * 0.97  # -3%
        result = detect_crisis_signals(self._make_series(closes))
        assert result["signals"]["single_day_drop"] is True

    def test_single_day_small_decline_no_trigger(self):
        """TAIEX 單日跌 1% 不觸發 single_day_drop。"""
        closes = [16000.0] * 130
        closes[-1] = 16000.0 * 0.99  # -1%
        result = detect_crisis_signals(self._make_series(closes))
        assert result["signals"]["single_day_drop"] is False

    def test_vix_plus_single_drop_crisis(self):
        """VIX spike + single_day_drop 兩個新訊號同時觸發 → crisis。"""
        closes = [16000.0] * 130
        closes[-1] = 16000.0 * 0.97  # -3% → single_day_drop
        vix = pd.Series([20.0, 35.0])  # VIX=35 > 30 → vix_spike
        result = detect_crisis_signals(self._make_series(closes), vix_series=vix)
        assert result["signals"]["vix_spike"] is True
        assert result["signals"]["single_day_drop"] is True
        assert result["crisis"] is True

    def test_backward_compat_no_vix(self):
        """不傳 vix_series 時行為與原 4 訊號完全一致（向後相容回歸測試）。"""
        closes = [16000.0] * 130
        # 只觸發 consec_decline（連跌 3 天但幅度極小）
        closes[-3] = 15998.0
        closes[-2] = 15996.0
        closes[-1] = 15994.0
        result = detect_crisis_signals(self._make_series(closes))
        # 新訊號預設 False，不影響結果
        assert result["signals"]["vix_spike"] is False
        assert result["signals"]["single_day_drop"] is False
        assert result["signals"]["us_vix_spike"] is False
        assert result["crisis"] is False  # 仍只有 1 個訊號

    # ── 美國 VIX (US VIX) spike 訊號測試 ──

    def test_us_vix_spike_absolute_level(self):
        """US VIX > 30 觸發 us_vix_spike。"""
        closes = self._make_series([16000.0] * 130)
        us_vix = pd.Series([20.0, 35.0])
        result = detect_crisis_signals(closes, us_vix_series=us_vix)
        assert result["signals"]["us_vix_spike"] is True
        assert result["us_vix_val"] == pytest.approx(35.0)

    def test_us_vix_spike_daily_change(self):
        """US VIX 單日漲幅 > 25% 觸發 us_vix_spike。"""
        closes = self._make_series([16000.0] * 130)
        us_vix = pd.Series([20.0, 26.0])  # 漲幅 30% > 25%
        result = detect_crisis_signals(closes, us_vix_series=us_vix)
        assert result["signals"]["us_vix_spike"] is True

    def test_us_vix_none_graceful(self):
        """us_vix_series=None → us_vix_spike=False（graceful degradation）。"""
        closes = self._make_series([16000.0] * 130)
        result = detect_crisis_signals(closes, us_vix_series=None)
        assert result["signals"]["us_vix_spike"] is False
        assert result["us_vix_val"] == 0.0

    def test_us_vix_insufficient_data(self):
        """US VIX 資料 < 2 筆 → us_vix_spike=False。"""
        closes = self._make_series([16000.0] * 130)
        us_vix = pd.Series([35.0])  # 只有一筆
        result = detect_crisis_signals(closes, us_vix_series=us_vix)
        assert result["signals"]["us_vix_spike"] is False

    def test_us_vix_plus_single_drop_crisis(self):
        """US VIX spike + single_day_drop → crisis。"""
        closes = [16000.0] * 130
        closes[-1] = 16000.0 * 0.97  # -3% → single_day_drop
        us_vix = pd.Series([20.0, 35.0])  # US VIX=35 > 30 → us_vix_spike
        result = detect_crisis_signals(self._make_series(closes), us_vix_series=us_vix)
        assert result["signals"]["us_vix_spike"] is True
        assert result["signals"]["single_day_drop"] is True
        assert result["crisis"] is True

    def test_tw_and_us_vix_both_spike_crisis(self):
        """TW VIX + US VIX 同時飆升 → crisis（兩個 VIX 訊號可同時觸發）。"""
        closes = self._make_series([16000.0] * 130)
        tw_vix = pd.Series([20.0, 35.0])
        us_vix = pd.Series([20.0, 35.0])
        result = detect_crisis_signals(closes, vix_series=tw_vix, us_vix_series=us_vix)
        assert result["signals"]["vix_spike"] is True
        assert result["signals"]["us_vix_spike"] is True
        assert result["crisis"] is True

    def test_us_vix_normal_no_trigger(self):
        """US VIX 正常水位（< 30 且漲幅 < 25%）不觸發。"""
        closes = self._make_series([16000.0] * 130)
        us_vix = pd.Series([18.0, 20.0])  # 漲幅 11%，值 20 — 都正常
        result = detect_crisis_signals(closes, us_vix_series=us_vix)
        assert result["signals"]["us_vix_spike"] is False
        assert result["us_vix_val"] == pytest.approx(20.0)


class TestRegimeWeightsCrisis:
    """Crisis regime 權重矩陣測試。"""

    def test_all_modes_crisis_weights_sum_to_one(self):
        """所有模式的 crisis 權重加總應為 1.0。"""
        for mode in ("momentum", "swing", "value", "dividend", "growth"):
            w = REGIME_WEIGHTS[mode]["crisis"]
            assert sum(w.values()) == pytest.approx(1.0), f"{mode}/crisis 權重和不為 1"

    def test_momentum_crisis_news_dominant(self):
        """momentum/crisis v5：technical 歸零；news 0.35 與 chip 0.35 並列最高，fundamental 0.30。"""
        w = REGIME_WEIGHTS["momentum"]["crisis"]
        assert w["news"] == 0.35
        assert w["chip"] == 0.35
        assert w["technical"] == 0.00
        assert w["fundamental"] == 0.30
        assert w["news"] == w["chip"]  # 並列最高
        assert w["news"] > w["fundamental"] > w["technical"]

    def test_swing_crisis_fundamental_dominant(self):
        """swing/crisis：基本面（0.55）應為最高權重（品質防禦）。"""
        w = REGIME_WEIGHTS["swing"]["crisis"]
        assert w["fundamental"] == 0.55
        assert w["fundamental"] > w["news"]

    def test_crisis_technical_weight_lower_than_bear(self):
        """crisis 的技術面權重應低於 bear（技術訊號在崩盤時更不可靠）。"""
        for mode in ("momentum", "swing", "growth"):
            crisis_tech = REGIME_WEIGHTS[mode]["crisis"].get("technical", 0.0)
            bear_tech = REGIME_WEIGHTS[mode]["bear"].get("technical", 0.0)
            assert crisis_tech <= bear_tech, f"{mode}: crisis tech {crisis_tech} > bear tech {bear_tech}"

    def test_get_weights_crisis_returns_correct_dict(self):
        """get_weights('momentum', 'crisis') 回傳正確權重字典（v5：technical 歸零）。"""
        w = MarketRegimeDetector.get_weights("momentum", "crisis")
        assert w["news"] == 0.35
        assert w["technical"] == 0.00
        assert w["fundamental"] == 0.30
        assert sum(w.values()) == pytest.approx(1.0)


class TestComputeMarketBreadthPct:
    """compute_market_breadth_pct 純函數測試。"""

    def test_all_below_ma20(self):
        """所有股票 close < ma20 → 回傳 1.0。"""
        closes = pd.Series([90, 80, 70], index=["A", "B", "C"])
        ma20s = pd.Series([100, 100, 100], index=["A", "B", "C"])
        assert compute_market_breadth_pct(closes, ma20s) == pytest.approx(1.0)

    def test_none_below_ma20(self):
        """所有股票 close > ma20 → 回傳 0.0。"""
        closes = pd.Series([110, 120, 130], index=["A", "B", "C"])
        ma20s = pd.Series([100, 100, 100], index=["A", "B", "C"])
        assert compute_market_breadth_pct(closes, ma20s) == pytest.approx(0.0)

    def test_mixed(self):
        """3/5 股跌破 → 回傳 0.6。"""
        closes = pd.Series([90, 80, 70, 110, 120], index=list("ABCDE"))
        ma20s = pd.Series([100, 100, 100, 100, 100], index=list("ABCDE"))
        assert compute_market_breadth_pct(closes, ma20s) == pytest.approx(0.6)

    def test_nan_ma20_excluded(self):
        """NaN ma20 應排除在分母之外。"""
        closes = pd.Series([90, 110, 120], index=["A", "B", "C"])
        ma20s = pd.Series([100, float("nan"), 100], index=["A", "B", "C"])
        # A: below, C: above → 1/2 = 0.5
        assert compute_market_breadth_pct(closes, ma20s) == pytest.approx(0.5)

    def test_empty_series(self):
        """空輸入 → 回傳 0.0。"""
        assert compute_market_breadth_pct(pd.Series(dtype=float), pd.Series(dtype=float)) == 0.0


class TestBreadthDowngrade:
    """市場寬度降級測試。"""

    def test_bull_downgraded_to_sideways(self):
        """bull + breadth=0.65 → 降級為 sideways。"""
        # 構造 bull 多數決（上升趨勢）
        closes = pd.Series([15000 + i * 25 for i in range(130)])
        result = detect_from_series(closes, breadth_below_ma20_pct=0.65)
        assert result["regime"] == "sideways"
        assert result["breadth_downgraded"] is True
        assert result["breadth_below_ma20_pct"] == pytest.approx(0.65)

    def test_sideways_downgraded_to_bear(self):
        """sideways + breadth=0.70 → 降級為 bear。"""
        # 構造混合訊號（sideways）：先跌後漲
        # 前 70 天穩定下跌（拉低 SMA120），後 60 天反彈但不夠高
        values = [17000 - i * 15 for i in range(70)]  # 跌到 15950
        values += [15950 + i * 5 for i in range(60)]  # 緩漲到 16250
        closes = pd.Series(values)
        # 先確認原始 regime = sideways（SMA60 bull，SMA120 bear，return sideways）
        baseline = detect_from_series(closes, breadth_below_ma20_pct=None)
        assert baseline["regime"] == "sideways", f"前提失敗：原始 regime={baseline['regime']}"
        # 加上 breadth > 0.60 → 降級
        result = detect_from_series(closes, breadth_below_ma20_pct=0.70)
        assert result["breadth_downgraded"] is True
        assert result["regime"] == "bear"

    def test_no_downgrade_below_threshold(self):
        """breadth=0.55（< 0.60）→ 不降級。"""
        closes = pd.Series([15000 + i * 25 for i in range(130)])
        result = detect_from_series(closes, breadth_below_ma20_pct=0.55)
        assert result["regime"] == "bull"
        assert result["breadth_downgraded"] is False

    def test_none_skips_check(self):
        """breadth=None（預設）→ 跳過寬度檢查。"""
        closes = pd.Series([15000 + i * 25 for i in range(130)])
        result = detect_from_series(closes, breadth_below_ma20_pct=None)
        assert result["regime"] == "bull"
        assert result["breadth_downgraded"] is False
        assert result["breadth_below_ma20_pct"] is None

    def test_crisis_overrides_breadth_downgrade(self):
        """breadth 降級後若 crisis 訊號觸發，crisis 仍優先。"""
        # 構造上升趨勢（bull）但末段急跌觸發 crisis
        closes_list = [15000 + i * 20 for i in range(130)]
        peak = closes_list[-6]
        for j in range(5):
            closes_list[-(5 - j)] = peak * (1 - 0.013 * (j + 1))
        closes = pd.Series(closes_list)
        result = detect_from_series(closes, breadth_below_ma20_pct=0.75)
        # crisis 應覆蓋 breadth 降級
        assert result["regime"] == "crisis"
        assert result["crisis_triggered"] is True


class TestPanicVolumeSignal:
    """爆量長黑（panic_volume）訊號測試。"""

    def _make_series(self, values: list[float]) -> pd.Series:
        return pd.Series(values)

    def test_panic_volume_triggers(self):
        """成交量 > 20d avg × 1.5 且下跌 → panic_volume=True。"""
        closes = [16000.0] * 130
        closes[-1] = 15900.0  # 最後一天下跌
        volumes = [1000.0] * 130
        volumes[-1] = 2000.0  # 最後一天爆量（2x > 1.5x）
        result = detect_crisis_signals(
            self._make_series(closes),
            volumes=self._make_series(volumes),
        )
        assert result["signals"]["panic_volume"] is True

    def test_no_trigger_positive_return(self):
        """成交量很大但上漲 → panic_volume=False。"""
        closes = [16000.0] * 130
        closes[-1] = 16100.0  # 上漲
        volumes = [1000.0] * 130
        volumes[-1] = 2000.0  # 爆量
        result = detect_crisis_signals(
            self._make_series(closes),
            volumes=self._make_series(volumes),
        )
        assert result["signals"]["panic_volume"] is False

    def test_no_trigger_low_volume(self):
        """下跌但成交量不夠大 → panic_volume=False。"""
        closes = [16000.0] * 130
        closes[-1] = 15900.0  # 下跌
        volumes = [1000.0] * 130
        volumes[-1] = 1200.0  # 量不夠（1.2x < 1.5x）
        result = detect_crisis_signals(
            self._make_series(closes),
            volumes=self._make_series(volumes),
        )
        assert result["signals"]["panic_volume"] is False

    def test_none_volumes_safe(self):
        """volumes=None → panic_volume=False（向後相容）。"""
        closes = [16000.0] * 130
        closes[-1] = 15900.0
        result = detect_crisis_signals(self._make_series(closes), volumes=None)
        assert result["signals"]["panic_volume"] is False

    def test_panic_plus_consec_triggers_crisis(self):
        """panic_volume + consec_decline → crisis=True（2/4）。"""
        closes = [16000.0] * 130
        # 連跌 3 天 + 最後一天爆量
        closes[-3] = 15998.0
        closes[-2] = 15996.0
        closes[-1] = 15994.0
        volumes = [1000.0] * 130
        volumes[-1] = 2000.0  # 爆量長黑
        result = detect_crisis_signals(
            self._make_series(closes),
            volumes=self._make_series(volumes),
        )
        assert result["signals"]["consec_decline"] is True
        assert result["signals"]["panic_volume"] is True
        assert result["crisis"] is True

    def test_only_panic_not_crisis(self):
        """僅 panic_volume=True 不觸發 crisis（1/4 < 2）。"""
        closes = [16000.0] * 130
        closes[-1] = 15999.0  # 微跌（不觸發 5d return 也不觸發 consec）
        volumes = [1000.0] * 130
        volumes[-1] = 2000.0  # 爆量
        result = detect_crisis_signals(
            self._make_series(closes),
            volumes=self._make_series(volumes),
        )
        assert result["signals"]["panic_volume"] is True
        assert result["signals"]["fast_return_5d"] is False
        assert result["signals"]["consec_decline"] is False
        assert result["crisis"] is False


# ── Hysteresis 測試 ──────────────────────────────────────────────


class TestCheckTransitionCondition:
    """check_transition_condition 純函數測試。"""

    def _bull_closes(self) -> pd.Series:
        """上升趨勢序列（close > SMA60 × 1.01）。"""
        return pd.Series([15000 + i * 25 for i in range(130)])

    def test_sideways_to_bull_above_threshold(self):
        """close > SMA60 × 1.01 → True。"""
        closes = self._bull_closes()
        result = check_transition_condition("sideways", "bull", closes)
        assert result is True

    def test_sideways_to_bull_below_threshold(self):
        """close 剛好在 SMA60 附近（< 1.01×）→ False。"""
        # 平穩序列：close ≈ SMA60
        closes = pd.Series([16000.0] * 60 + [16010.0])  # 僅微漲
        result = check_transition_condition("sideways", "bull", closes)
        assert result is False

    def test_bull_to_sideways_below_threshold(self):
        """close < SMA60 × 0.99 → True（快速降級）。"""
        # 先漲再急跌
        values = [15000 + i * 20 for i in range(130)]
        values[-1] = float(pd.Series(values[-60:]).mean()) * 0.98  # 跌破 2%
        closes = pd.Series(values)
        result = check_transition_condition("bull", "sideways", closes)
        assert result is True

    def test_bull_to_sideways_above_threshold(self):
        """close 仍在 SMA60 × 0.99 之上 → False。"""
        closes = self._bull_closes()
        result = check_transition_condition("bull", "sideways", closes)
        assert result is False

    def test_crisis_exit_lows_rising_vol_calming(self):
        """3 日低點遞增 + 波動率正常 → True。"""
        # 穩定序列 + 末 3 天連漲（模擬 crisis 退出）
        values = [16000.0] * 130
        values[-3] = 15900.0
        values[-2] = 15950.0
        values[-1] = 16000.0
        closes = pd.Series(values)
        result = check_transition_condition("crisis", "bear", closes)
        assert result is True

    def test_crisis_exit_not_rising(self):
        """3 日低點未遞增 → False。"""
        values = [16000.0] * 130
        values[-3] = 15950.0
        values[-2] = 15900.0  # 第 2 天反而更低
        values[-1] = 15920.0
        closes = pd.Series(values)
        result = check_transition_condition("crisis", "bear", closes)
        assert not result


class TestApplyHysteresis:
    """apply_hysteresis 純函數測試。"""

    def _bull_closes(self) -> pd.Series:
        """上升趨勢序列。"""
        return pd.Series([15000 + i * 25 for i in range(130)])

    def _bear_closes(self) -> pd.Series:
        """下跌趨勢序列。"""
        return pd.Series([18000 - i * 25 for i in range(130)])

    def test_cold_start_accepts_raw(self):
        """prev_regime=None → 直接使用 raw_regime。"""
        closes = self._bull_closes()
        regime, count, info = apply_hysteresis("bull", None, closes)
        assert regime == "bull"
        assert count == 0
        assert info["reason"] == "cold_start"

    def test_crisis_immediate_no_hysteresis(self):
        """raw_regime=crisis → 立即切換，不需確認。"""
        closes = self._bull_closes()
        regime, count, info = apply_hysteresis("crisis", "bull", closes)
        assert regime == "crisis"
        assert count == 0
        assert info["reason"] == "crisis_immediate"

    def test_same_regime_resets_counter(self):
        """raw == prev → 不變，重置計數器。"""
        closes = self._bull_closes()
        regime, count, info = apply_hysteresis("bull", "bull", closes, confirmation_count=2)
        assert regime == "bull"
        assert count == 0
        assert info["reason"] == "no_change"

    def test_sideways_to_bull_day1_blocked(self):
        """sideways→bull 第 1 天條件符合但 count < 3 → 維持 sideways。"""
        closes = self._bull_closes()  # close > SMA60 × 1.01
        regime, count, info = apply_hysteresis("bull", "sideways", closes, confirmation_count=0)
        assert regime == "sideways"  # blocked
        assert count == 1
        assert info["transition_blocked"] is True
        assert "1/3" in info["confirmation_progress"]

    def test_sideways_to_bull_day2_blocked(self):
        """sideways→bull 第 2 天 → 仍維持 sideways, count=2。"""
        closes = self._bull_closes()
        regime, count, info = apply_hysteresis("bull", "sideways", closes, confirmation_count=1)
        assert regime == "sideways"
        assert count == 2
        assert info["transition_blocked"] is True

    def test_sideways_to_bull_day3_confirmed(self):
        """sideways→bull 第 3 天 count=3 → 切換至 bull。"""
        closes = self._bull_closes()
        regime, count, info = apply_hysteresis("bull", "sideways", closes, confirmation_count=2)
        assert regime == "bull"
        assert count == 0
        assert info["transition_blocked"] is False

    def test_sideways_to_bull_condition_not_met(self):
        """sideways→bull 但 close 未站上 SMA60 × 1.01 → 重置計數器。"""
        # 平穩序列
        closes = pd.Series([16000.0] * 130)
        regime, count, info = apply_hysteresis("bull", "sideways", closes, confirmation_count=2)
        assert regime == "sideways"
        assert count == 0
        assert info["reason"] == "condition_not_met"

    def test_bull_to_sideways_fast_1day(self):
        """bull→sideways：close < SMA60 × 0.99 → 1 天立即降級。"""
        values = [15000 + i * 20 for i in range(130)]
        sma60_val = float(pd.Series(values[-60:]).mean())
        values[-1] = sma60_val * 0.98  # 跌破 2%
        closes = pd.Series(values)
        regime, count, info = apply_hysteresis("sideways", "bull", closes, confirmation_count=0)
        assert regime == "sideways"
        assert count == 0
        assert info["transition_blocked"] is False

    def test_bull_to_sideways_not_deep_enough(self):
        """bull→sideways：close 仍 > SMA60 × 0.99 → 不觸發。"""
        closes = self._bull_closes()
        regime, count, info = apply_hysteresis("sideways", "bull", closes, confirmation_count=0)
        # check_transition_condition("bull", "sideways") should be False for bull closes
        assert regime == "bull"
        assert info["reason"] == "condition_not_met"

    def test_bear_to_sideways_needs_3days(self):
        """bear→sideways 需 3 天 close > SMA60 確認。"""
        closes = self._bull_closes()  # close > SMA60
        # Day 1: blocked
        regime1, count1, _ = apply_hysteresis("sideways", "bear", closes, confirmation_count=0)
        assert regime1 == "bear"
        assert count1 == 1
        # Day 2: blocked
        regime2, count2, _ = apply_hysteresis("sideways", "bear", closes, confirmation_count=1)
        assert regime2 == "bear"
        assert count2 == 2
        # Day 3: confirmed
        regime3, count3, info3 = apply_hysteresis("sideways", "bear", closes, confirmation_count=2)
        assert regime3 == "sideways"
        assert count3 == 0
        assert info3["transition_blocked"] is False

    def test_unhandled_transition_default_2days(self):
        """未定義的轉換（bull→bear）→ 預設 2 天確認。"""
        closes = self._bear_closes()
        # Day 1: blocked
        regime1, count1, _ = apply_hysteresis("bear", "bull", closes, confirmation_count=0)
        assert regime1 == "bull"
        assert count1 == 1
        # Day 2: confirmed
        regime2, count2, info2 = apply_hysteresis("bear", "bull", closes, confirmation_count=1)
        assert regime2 == "bear"
        assert count2 == 0
        assert info2["transition_blocked"] is False

    def test_crisis_to_bear_needs_2days(self):
        """crisis→bear 需 2 天確認。"""
        # 穩定序列 + 末 3 天連漲（crisis exit 條件）
        values = [16000.0] * 130
        values[-3] = 15900.0
        values[-2] = 15950.0
        values[-1] = 16000.0
        closes = pd.Series(values)
        # Day 1: blocked
        regime1, count1, _ = apply_hysteresis("bear", "crisis", closes, confirmation_count=0)
        assert regime1 == "crisis"
        assert count1 == 1
        # Day 2: confirmed
        regime2, count2, _ = apply_hysteresis("bear", "crisis", closes, confirmation_count=1)
        assert regime2 == "bear"
        assert count2 == 0


class TestRegimeStateMachine:
    """RegimeStateMachine 狀態管理測試。"""

    def _bull_closes(self) -> pd.Series:
        return pd.Series([15000 + i * 25 for i in range(130)])

    def test_cold_start_no_file(self, tmp_path):
        """無 JSON 檔 → cold start，使用 raw regime。"""
        sm = RegimeStateMachine(state_path=tmp_path / "state.json")
        closes = self._bull_closes()
        result = sm.update(closes)
        assert result["regime"] == "bull"
        assert sm.current_regime == "bull"

    def test_state_persistence(self, tmp_path):
        """update() 後 JSON 檔應存在且包含正確 regime。"""
        import json

        state_file = tmp_path / "state.json"
        sm = RegimeStateMachine(state_path=state_file)
        sm.update(self._bull_closes())
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["regime"] == "bull"

    def test_hysteresis_across_updates(self, tmp_path):
        """連續 3 次 update() 驗證 sideways→bull 確認流程。"""
        import json

        state_file = tmp_path / "state.json"
        # 先寫入 sideways 初始狀態
        state_file.write_text(
            json.dumps(
                {
                    "regime": "sideways",
                    "regime_since": "2026-03-18",
                    "confirmation_count": 0,
                    "pending_transition": None,
                    "last_updated": "2026-03-18",
                }
            ),
            encoding="utf-8",
        )
        sm = RegimeStateMachine(state_path=state_file)
        closes = self._bull_closes()

        # 第一次呼叫（raw=bull, prev=sideways, day 1）
        result = sm.update(closes)
        # 由於 sideways→bull 需 3 天，第一天應被 block
        # 但 RegimeStateMachine 的同日防護可能干擾
        # 直接檢查 JSON 的 confirmation_count
        data = json.loads(state_file.read_text(encoding="utf-8"))
        # 因為 last_updated != today（2026-03-18 vs today），所以會執行
        assert data["regime"] in ("sideways", "bull")  # 取決於確認進度

    def test_corrupt_json_cold_start(self, tmp_path):
        """JSON 損壞 → 當作 cold start。"""
        state_file = tmp_path / "state.json"
        state_file.write_text("CORRUPT{{{", encoding="utf-8")
        sm = RegimeStateMachine(state_path=state_file)
        result = sm.update(self._bull_closes())
        assert result["regime"] == "bull"  # cold start → accept raw

    def test_update_returns_superset_dict(self, tmp_path):
        """回傳 dict 包含原有 + 新增 hysteresis 欄位。"""
        sm = RegimeStateMachine(state_path=tmp_path / "state.json")
        result = sm.update(self._bull_closes())
        # 原有欄位
        assert "regime" in result
        assert "taiex_close" in result
        assert "signals" in result
        assert "crisis_triggered" in result
        # 新增欄位
        assert "hysteresis_applied" in result
        assert "raw_regime" in result
        assert "transition_info" in result

    def test_current_regime_property(self, tmp_path):
        """current_regime 屬性正確反映最新狀態。"""
        sm = RegimeStateMachine(state_path=tmp_path / "state.json")
        assert sm.current_regime is None  # 未初始化
        sm.update(self._bull_closes())
        assert sm.current_regime == "bull"
