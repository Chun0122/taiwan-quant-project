"""測試 suggest 命令的純函數：_calc_rsi14_from_series、_assess_timing、_format_suggest_discord。

所有測試皆為純函數測試（零 mock），不需要 DB 或外部服務。
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from main import _assess_timing, _calc_rsi14_from_series, _format_suggest_discord

# ============================================================
# _calc_rsi14_from_series
# ============================================================


class TestCalcRsi14FromSeries:
    def test_insufficient_data_returns_50(self):
        """資料不足 15 筆時回傳中性值 50.0。"""
        closes = pd.Series([100.0] * 10)
        assert _calc_rsi14_from_series(closes) == 50.0

    def test_exactly_14_rows_returns_50(self):
        """剛好 14 筆（需要 15 筆）時仍回傳 50.0。"""
        closes = pd.Series([100.0 + i for i in range(14)])
        assert _calc_rsi14_from_series(closes) == 50.0

    def test_flat_series_returns_50(self):
        """完全持平序列（無漲跌）→ gain=loss=0 → 50.0。"""
        closes = pd.Series([100.0] * 20)
        assert _calc_rsi14_from_series(closes) == 50.0

    def test_all_up_returns_high_rsi(self):
        """持續上漲序列 RSI 應接近 100。"""
        closes = pd.Series([100.0 + i for i in range(30)])
        result = _calc_rsi14_from_series(closes)
        assert result > 80.0

    def test_all_down_returns_low_rsi(self):
        """持續下跌序列 RSI 應接近 0。"""
        closes = pd.Series([200.0 - i for i in range(30)])
        result = _calc_rsi14_from_series(closes)
        assert result < 20.0

    def test_result_in_valid_range(self):
        """結果必須在 0~100 之間。"""
        rng = pd.Series([100.0 + np.sin(i / 3) * 10 for i in range(40)])
        result = _calc_rsi14_from_series(rng)
        assert 0.0 <= result <= 100.0

    def test_exactly_15_rows_works(self):
        """恰好 15 筆資料應能正常計算（非 50.0）。"""
        closes = pd.Series([100.0 + i * 2 for i in range(15)])
        result = _calc_rsi14_from_series(closes)
        assert result != 50.0
        assert 0.0 <= result <= 100.0

    def test_mixed_up_down_mid_range(self):
        """混合漲跌序列 RSI 應落在中間範圍（30~70）。"""
        closes = pd.Series([100.0 + (i % 2) * 2 for i in range(30)])
        result = _calc_rsi14_from_series(closes)
        assert 30.0 < result < 70.0

    def test_returns_float(self):
        """回傳型別必須是 float。"""
        closes = pd.Series([100.0 + i for i in range(20)])
        result = _calc_rsi14_from_series(closes)
        assert isinstance(result, float)


# ============================================================
# _assess_timing
# ============================================================


class TestAssessTiming:
    # ── 超買邊界 ────────────────────────────────────────────

    def test_overbought_rsi70_warns(self):
        """RSI = 70 → 謹慎觀望。"""
        result = _assess_timing(rsi14=70.0, close=100.0, sma20=95.0, atr_pct=0.03, regime="bull")
        assert "謹慎" in result

    def test_overbought_rsi80_any_regime(self):
        """RSI = 80 + bear → 仍謹慎。"""
        result = _assess_timing(rsi14=80.0, close=90.0, sma20=95.0, atr_pct=0.03, regime="bear")
        assert "謹慎" in result

    # ── 超賣邊界 ────────────────────────────────────────────

    def test_oversold_bull_rebound(self):
        """RSI = 28 + bull → 潛在反彈。"""
        result = _assess_timing(rsi14=28.0, close=90.0, sma20=100.0, atr_pct=0.03, regime="bull")
        assert "反彈" in result

    def test_oversold_sideways_rebound(self):
        """RSI = 25 + sideways → 潛在反彈。"""
        result = _assess_timing(rsi14=25.0, close=90.0, sma20=100.0, atr_pct=0.03, regime="sideways")
        assert "反彈" in result

    def test_oversold_bear_wait(self):
        """RSI = 25 + bear → 不應積極做多（等待企穩）。"""
        result = _assess_timing(rsi14=25.0, close=80.0, sma20=100.0, atr_pct=0.03, regime="bear")
        assert "等待" in result or "企穩" in result

    # ── 積極做多 ─────────────────────────────────────────────

    def test_active_long_bull_above_sma_high_rsi(self):
        """RSI 60 + 站上均線（>0.5%）+ bull → 積極做多。"""
        result = _assess_timing(rsi14=60.0, close=103.0, sma20=100.0, atr_pct=0.03, regime="bull")
        assert "積極" in result

    def test_orderly_bull_above_sma_low_rsi(self):
        """RSI 50 + 站上均線 + bull → 順勢佈局。"""
        result = _assess_timing(rsi14=50.0, close=103.0, sma20=100.0, atr_pct=0.03, regime="bull")
        assert "順勢" in result

    # ── 區間觀望 ─────────────────────────────────────────────

    def test_sideways_above_sma(self):
        """站上均線 + sideways → 區間上軌。"""
        result = _assess_timing(rsi14=55.0, close=103.0, sma20=100.0, atr_pct=0.03, regime="sideways")
        assert "區間" in result

    # ── 空頭 ─────────────────────────────────────────────────

    def test_bear_regime_warns(self):
        """bear + RSI 中性 + 均線下方 → 空頭觀望。"""
        result = _assess_timing(rsi14=45.0, close=95.0, sma20=100.0, atr_pct=0.03, regime="bear")
        assert "空頭" in result

    # ── 等待訊號 ─────────────────────────────────────────────

    def test_wait_below_sma_sideways(self):
        """均線下方 + sideways + RSI 中性 → 等待訊號。"""
        result = _assess_timing(rsi14=45.0, close=95.0, sma20=100.0, atr_pct=0.03, regime="sideways")
        assert "等待" in result

    # ── 波動率修飾符 ─────────────────────────────────────────

    def test_low_volatility_appended(self):
        """atr_pct < 1.5% → 附加「低波動」。"""
        result = _assess_timing(rsi14=50.0, close=100.0, sma20=98.0, atr_pct=0.01, regime="sideways")
        assert "低波動" in result

    def test_high_volatility_appended(self):
        """atr_pct > 4% → 附加「高波動謹慎」。"""
        result = _assess_timing(rsi14=50.0, close=100.0, sma20=98.0, atr_pct=0.05, regime="bull")
        assert "高波動" in result

    def test_normal_volatility_no_tag(self):
        """atr_pct 在 1.5%~4% 之間 → 無波動率標籤。"""
        result = _assess_timing(rsi14=50.0, close=100.0, sma20=98.0, atr_pct=0.03, regime="bull")
        assert "低波動" not in result
        assert "高波動" not in result

    # ── 均線邊界 ─────────────────────────────────────────────

    def test_close_at_sma_not_above(self):
        """close == sma20 → 未達 0.5% 閾值，不視為站上均線。"""
        result = _assess_timing(rsi14=55.0, close=100.0, sma20=100.0, atr_pct=0.03, regime="bull")
        # close = sma20 → 不滿足 close > sma20 * 1.005
        assert "積極" not in result
        assert "順勢" not in result

    def test_returns_non_empty_string(self):
        """回傳值必須是非空字串。"""
        result = _assess_timing(50.0, 100.0, 100.0, 0.03, "bull")
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================
# _format_suggest_discord
# ============================================================


def _default_kwargs() -> dict:
    """提供預設測試參數。"""
    return dict(
        stock_id="2330",
        stock_name="台積電",
        today=datetime.date(2026, 3, 1),
        close=850.0,
        sma20=832.5,
        rsi14=62.3,
        atr_str="18.45（2.17%）",
        regime_zh="多頭",
        taiex_close=22350.0,
        entry_price=850.0,
        sl_str="822.33（-3.3%）",
        tp_str="905.35（+6.5%）",
        rr_str="1 : 2.0",
        trigger="站上均線",
        timing="積極做多：動能強勁 + 趨勢向上",
        valid_until=datetime.date(2026, 3, 6),
    )


class TestFormatSuggestDiscord:
    def test_returns_string(self):
        result = _format_suggest_discord(**_default_kwargs())
        assert isinstance(result, str)

    def test_under_2000_chars(self):
        result = _format_suggest_discord(**_default_kwargs())
        assert len(result) <= 2000

    def test_contains_stock_id(self):
        result = _format_suggest_discord(**_default_kwargs())
        assert "2330" in result

    def test_contains_stock_name(self):
        result = _format_suggest_discord(**_default_kwargs())
        assert "台積電" in result

    def test_contains_entry_price(self):
        result = _format_suggest_discord(**_default_kwargs())
        assert "850.00" in result

    def test_contains_regime(self):
        result = _format_suggest_discord(**_default_kwargs())
        assert "多頭" in result

    def test_contains_trigger(self):
        result = _format_suggest_discord(**_default_kwargs())
        assert "站上均線" in result

    def test_contains_timing(self):
        result = _format_suggest_discord(**_default_kwargs())
        assert "積極做多" in result

    def test_contains_valid_until(self):
        result = _format_suggest_discord(**_default_kwargs())
        assert "2026-03-06" in result

    def test_long_inputs_truncated_to_2000(self):
        """超長輸入應截斷至 2000 字元。"""
        kw = _default_kwargs()
        kw["stock_name"] = "測" * 500
        result = _format_suggest_discord(**kw)
        assert len(result) <= 2000
