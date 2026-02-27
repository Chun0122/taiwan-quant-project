"""策略 generate_signals() 測試 — 純函數，不走 DB。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_data_with_sma(n: int = 30) -> pd.DataFrame:
    """建立含 SMA 指標欄位的測試資料。"""
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = [100 + i * 0.5 for i in range(n)]
    df = pd.DataFrame(
        {"close": close, "open": close, "high": close, "low": close, "volume": 1000},
        index=[d.date() for d in dates],
    )
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    return df


def _make_golden_cross_data() -> pd.DataFrame:
    """建構一個明確的黃金交叉場景。"""
    dates = pd.bdate_range("2024-01-01", periods=5)
    idx = [d.date() for d in dates]
    df = pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104],
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "volume": [1000] * 5,
            # day 0-2: fast < slow; day 3-4: fast > slow → 黃金交叉在 day 3
            "sma_10": [98, 99, 100, 102, 103],
            "sma_20": [100, 100, 100, 100, 100],
        },
        index=idx,
    )
    return df


def _make_death_cross_data() -> pd.DataFrame:
    """建構一個明確的死亡交叉場景。"""
    dates = pd.bdate_range("2024-01-01", periods=5)
    idx = [d.date() for d in dates]
    df = pd.DataFrame(
        {
            "close": [104, 103, 102, 101, 100],
            "open": [104, 103, 102, 101, 100],
            "high": [105, 104, 103, 102, 101],
            "low": [103, 102, 101, 100, 99],
            "volume": [1000] * 5,
            # day 0-2: fast > slow; day 3-4: fast < slow → 死亡交叉在 day 3
            "sma_10": [102, 101, 100, 98, 97],
            "sma_20": [100, 100, 100, 100, 100],
        },
        index=idx,
    )
    return df


# ================================================================
# SMACrossStrategy
# ================================================================


class TestSMACrossStrategy:
    def test_golden_cross_buy(self):
        from src.strategy.sma_cross import SMACrossStrategy

        df = _make_golden_cross_data()
        s = SMACrossStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        # day 3 應該是買入訊號
        assert signals.iloc[3] == 1

    def test_death_cross_sell(self):
        from src.strategy.sma_cross import SMACrossStrategy

        df = _make_death_cross_data()
        s = SMACrossStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        # day 3 應該是賣出訊號
        assert signals.iloc[3] == -1

    def test_no_cross_zero(self):
        """沒有交叉時全為 0。"""
        from src.strategy.sma_cross import SMACrossStrategy

        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {
                "close": [100] * 5,
                "open": [100] * 5,
                "high": [101] * 5,
                "low": [99] * 5,
                "volume": [1000] * 5,
                # fast 始終 > slow，無交叉
                "sma_10": [105, 105, 105, 105, 105],
                "sma_20": [100, 100, 100, 100, 100],
            },
            index=idx,
        )
        s = SMACrossStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert (signals == 0).all()

    def test_missing_column_raises(self):
        from src.strategy.sma_cross import SMACrossStrategy

        df = pd.DataFrame({"close": [100]}, index=[pd.Timestamp("2024-01-01").date()])
        s = SMACrossStrategy("TEST", "2024-01-01", "2024-12-31")
        with pytest.raises(ValueError, match="缺少指標欄位"):
            s.generate_signals(df)

    def test_custom_periods(self):
        """自訂快慢週期。"""
        from src.strategy.sma_cross import SMACrossStrategy

        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {
                "close": [100] * 5,
                "open": [100] * 5,
                "high": [101] * 5,
                "low": [99] * 5,
                "volume": [1000] * 5,
                "sma_5": [98, 99, 100, 102, 103],
                "sma_60": [100, 100, 100, 100, 100],
            },
            index=idx,
        )
        s = SMACrossStrategy("TEST", "2024-01-01", "2024-12-31", fast=5, slow=60)
        signals = s.generate_signals(df)
        assert signals.iloc[3] == 1  # 黃金交叉


# ================================================================
# RSIThresholdStrategy
# ================================================================


class TestRSIThresholdStrategy:
    def test_oversold_breakout_buy(self):
        from src.strategy.rsi_threshold import RSIThresholdStrategy

        dates = pd.bdate_range("2024-01-01", periods=4)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {
                "close": [100] * 4,
                "open": [100] * 4,
                "high": [101] * 4,
                "low": [99] * 4,
                "volume": [1000] * 4,
                "rsi_14": [25, 28, 31, 35],  # day 2: 從 28→31 突破 30
            },
            index=idx,
        )
        s = RSIThresholdStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert signals.iloc[2] == 1

    def test_overbought_breakdown_sell(self):
        from src.strategy.rsi_threshold import RSIThresholdStrategy

        dates = pd.bdate_range("2024-01-01", periods=4)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {
                "close": [100] * 4,
                "open": [100] * 4,
                "high": [101] * 4,
                "low": [99] * 4,
                "volume": [1000] * 4,
                "rsi_14": [75, 72, 69, 65],  # day 2: 從 72→69 跌破 70
            },
            index=idx,
        )
        s = RSIThresholdStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert signals.iloc[2] == -1

    def test_mid_range_no_signal(self):
        from src.strategy.rsi_threshold import RSIThresholdStrategy

        dates = pd.bdate_range("2024-01-01", periods=4)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {
                "close": [100] * 4,
                "open": [100] * 4,
                "high": [101] * 4,
                "low": [99] * 4,
                "volume": [1000] * 4,
                "rsi_14": [50, 52, 48, 55],
            },
            index=idx,
        )
        s = RSIThresholdStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert (signals == 0).all()

    def test_missing_rsi_raises(self):
        from src.strategy.rsi_threshold import RSIThresholdStrategy

        df = pd.DataFrame({"close": [100]}, index=[pd.Timestamp("2024-01-01").date()])
        s = RSIThresholdStrategy("TEST", "2024-01-01", "2024-12-31")
        with pytest.raises(ValueError, match="缺少指標欄位"):
            s.generate_signals(df)

    def test_custom_thresholds(self):
        from src.strategy.rsi_threshold import RSIThresholdStrategy

        dates = pd.bdate_range("2024-01-01", periods=3)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {
                "close": [100] * 3,
                "open": [100] * 3,
                "high": [101] * 3,
                "low": [99] * 3,
                "volume": [1000] * 3,
                "rsi_14": [15, 20, 25],  # day 1: prev=15<=20, day 2: rsi=25>20 → 突破
            },
            index=idx,
        )
        s = RSIThresholdStrategy("TEST", "2024-01-01", "2024-12-31", oversold=20, overbought=80)
        signals = s.generate_signals(df)
        assert signals.iloc[2] == 1


# ================================================================
# BollingerBandBreakoutStrategy
# ================================================================


class TestBollingerBandBreakoutStrategy:
    def test_lower_band_bounce_buy(self):
        from src.strategy.bb_breakout import BollingerBandBreakoutStrategy

        dates = pd.bdate_range("2024-01-01", periods=4)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {
                "close": [95, 94, 97, 100],  # day 1 close<=lower, day 2 close>lower
                "open": [96, 95, 96, 99],
                "high": [97, 96, 98, 101],
                "low": [94, 93, 95, 98],
                "volume": [1000] * 4,
                "bb_upper": [110, 110, 110, 110],
                "bb_lower": [95, 95, 95, 95],
            },
            index=idx,
        )
        s = BollingerBandBreakoutStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert signals.iloc[2] == 1

    def test_upper_band_fallback_sell(self):
        from src.strategy.bb_breakout import BollingerBandBreakoutStrategy

        dates = pd.bdate_range("2024-01-01", periods=4)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {
                "close": [105, 111, 108, 105],  # day 1 close>=upper, day 2 close<upper
                "open": [104, 110, 109, 106],
                "high": [106, 112, 110, 107],
                "low": [103, 109, 107, 104],
                "volume": [1000] * 4,
                "bb_upper": [110, 110, 110, 110],
                "bb_lower": [90, 90, 90, 90],
            },
            index=idx,
        )
        s = BollingerBandBreakoutStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert signals.iloc[2] == -1

    def test_missing_bb_raises(self):
        from src.strategy.bb_breakout import BollingerBandBreakoutStrategy

        df = pd.DataFrame({"close": [100]}, index=[pd.Timestamp("2024-01-01").date()])
        s = BollingerBandBreakoutStrategy("TEST", "2024-01-01", "2024-12-31")
        with pytest.raises(ValueError, match="缺少指標欄位"):
            s.generate_signals(df)


# ================================================================
# MACDCrossStrategy
# ================================================================


class TestMACDCrossStrategy:
    def test_macd_golden_cross_buy(self):
        from src.strategy.macd_cross import MACDCrossStrategy

        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {
                "close": [100] * 5,
                "open": [100] * 5,
                "high": [101] * 5,
                "low": [99] * 5,
                "volume": [1000] * 5,
                "macd": [-2, -1, 0, 1, 2],
                "macd_signal": [0, 0, 0, 0, 0],
            },
            index=idx,
        )
        s = MACDCrossStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        # day 3: prev_diff=0→0, curr_diff=1>0 → 買入
        assert signals.iloc[3] == 1

    def test_macd_death_cross_sell(self):
        from src.strategy.macd_cross import MACDCrossStrategy

        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {
                "close": [100] * 5,
                "open": [100] * 5,
                "high": [101] * 5,
                "low": [99] * 5,
                "volume": [1000] * 5,
                "macd": [2, 1, 0, -1, -2],
                "macd_signal": [0, 0, 0, 0, 0],
            },
            index=idx,
        )
        s = MACDCrossStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        # day 3: prev_diff=0→0, curr_diff=-1<0 → 賣出
        assert signals.iloc[3] == -1

    def test_missing_macd_raises(self):
        from src.strategy.macd_cross import MACDCrossStrategy

        df = pd.DataFrame({"close": [100]}, index=[pd.Timestamp("2024-01-01").date()])
        s = MACDCrossStrategy("TEST", "2024-01-01", "2024-12-31")
        with pytest.raises(ValueError, match="缺少指標欄位"):
            s.generate_signals(df)


# ================================================================
# BuyAndHoldStrategy
# ================================================================


class TestBuyAndHoldStrategy:
    def test_first_day_buy(self):
        from src.strategy.buy_hold import BuyAndHoldStrategy

        dates = pd.bdate_range("2024-01-01", periods=10)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {"close": [100] * 10, "open": [100] * 10, "high": [101] * 10, "low": [99] * 10, "volume": [1000] * 10},
            index=idx,
        )
        s = BuyAndHoldStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert signals.iloc[0] == 1

    def test_last_day_sell(self):
        from src.strategy.buy_hold import BuyAndHoldStrategy

        dates = pd.bdate_range("2024-01-01", periods=10)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {"close": [100] * 10, "open": [100] * 10, "high": [101] * 10, "low": [99] * 10, "volume": [1000] * 10},
            index=idx,
        )
        s = BuyAndHoldStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert signals.iloc[-1] == -1

    def test_middle_days_hold(self):
        from src.strategy.buy_hold import BuyAndHoldStrategy

        dates = pd.bdate_range("2024-01-01", periods=10)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {"close": [100] * 10, "open": [100] * 10, "high": [101] * 10, "low": [99] * 10, "volume": [1000] * 10},
            index=idx,
        )
        s = BuyAndHoldStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert (signals.iloc[1:-1] == 0).all()

    def test_empty_data(self):
        from src.strategy.buy_hold import BuyAndHoldStrategy

        df = pd.DataFrame(columns=["close", "open", "high", "low", "volume"])
        s = BuyAndHoldStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert len(signals) == 0

    def test_single_day(self):
        """只有 1 天：第一天=買入，最後天=賣出 → 都是同一天。"""
        from src.strategy.buy_hold import BuyAndHoldStrategy

        idx = [pd.Timestamp("2024-01-02").date()]
        df = pd.DataFrame(
            {"close": [100], "open": [100], "high": [101], "low": [99], "volume": [1000]},
            index=idx,
        )
        s = BuyAndHoldStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        # 第一天也是最後天，最終結果看 iloc[-1] 是 -1 (後蓋前)
        assert signals.iloc[0] == -1


# ================================================================
# MultiFactorStrategy.generate_signals()
# ================================================================


class TestMultiFactorStrategy:
    def _make_data(self, rsi=50, macd_diff=0, foreign=0, yoy=0, n=5):
        dates = pd.bdate_range("2024-01-01", periods=n)
        idx = [d.date() for d in dates]
        return pd.DataFrame(
            {
                "close": [100] * n,
                "open": [100] * n,
                "high": [101] * n,
                "low": [99] * n,
                "volume": [1000] * n,
                "rsi_14": [rsi] * n,
                "macd": [macd_diff] * n,
                "macd_signal": [0] * n,
                "foreign_net": [foreign] * n,
                "trust_net": [0] * n,
                "dealer_net": [0] * n,
                "yoy_growth": [yoy] * n,
            },
            index=idx,
        )

    def test_all_bullish_buy(self):
        from src.strategy.multi_factor import MultiFactorStrategy

        df = self._make_data(rsi=25, macd_diff=5, foreign=1000, yoy=30)
        s = MultiFactorStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        # 全看多 → 加權分數 > buy_threshold → 買入
        assert (signals == 1).any()

    def test_all_bearish_sell(self):
        from src.strategy.multi_factor import MultiFactorStrategy

        df = self._make_data(rsi=75, macd_diff=-5, foreign=-1000, yoy=-10)
        s = MultiFactorStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        # 全看空 → 加權分數 < sell_threshold → 賣出
        assert (signals == -1).any()

    def test_neutral_hold(self):
        from src.strategy.multi_factor import MultiFactorStrategy

        df = self._make_data(rsi=50, macd_diff=0, foreign=0, yoy=10)
        s = MultiFactorStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        # 中性 → 持有
        assert (signals == 0).all()

    def test_rsi_factor_contribution(self):
        """只有 RSI 看多，其餘中性。"""
        from src.strategy.multi_factor import MultiFactorStrategy

        df = self._make_data(rsi=25, macd_diff=0, foreign=0, yoy=10)
        # RSI < 30 → score=+1, weight=0.2 → total=0.2 < 0.3 threshold → hold
        s = MultiFactorStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert (signals == 0).all()

    def test_custom_threshold(self):
        """自訂低門檻。"""
        from src.strategy.multi_factor import MultiFactorStrategy

        df = self._make_data(rsi=25)  # 只有 RSI 看多，score=0.2
        s = MultiFactorStrategy("TEST", "2024-01-01", "2024-12-31", buy_threshold=0.1)
        signals = s.generate_signals(df)
        assert (signals == 1).any()

    def test_no_indicator_columns(self):
        """缺少所有指標欄位時應全 hold（不崩潰）。"""
        from src.strategy.multi_factor import MultiFactorStrategy

        dates = pd.bdate_range("2024-01-01", periods=3)
        idx = [d.date() for d in dates]
        df = pd.DataFrame(
            {"close": [100] * 3, "open": [100] * 3, "high": [101] * 3, "low": [99] * 3, "volume": [1000] * 3},
            index=idx,
        )
        s = MultiFactorStrategy("TEST", "2024-01-01", "2024-12-31")
        signals = s.generate_signals(df)
        assert (signals == 0).all()


# ================================================================
# Strategy._apply_dividend_adjustment()
# ================================================================


class TestDividendAdjustment:
    def _make_strategy(self):
        from src.strategy.sma_cross import SMACrossStrategy

        return SMACrossStrategy("TEST", "2024-01-01", "2024-12-31")

    def _make_price_df(self, n=10, base=100.0):
        dates = pd.bdate_range("2024-01-01", periods=n)
        idx = [d.date() for d in dates]
        close = [base + i for i in range(n)]
        return pd.DataFrame(
            {
                "open": [c - 0.5 for c in close],
                "high": [c + 1 for c in close],
                "low": [c - 1 for c in close],
                "close": close,
                "volume": [1000] * n,
            },
            index=idx,
        )

    def test_empty_dividends_no_change(self):
        s = self._make_strategy()
        df = self._make_price_df()
        divs = pd.DataFrame(columns=["cash_dividend", "stock_dividend"])
        result = s._apply_dividend_adjustment(df, divs)
        np.testing.assert_array_equal(result["close"].values, df["close"].values)

    def test_cash_dividend_adjusts_price_down(self):
        s = self._make_strategy()
        df = self._make_price_df(n=5, base=100.0)
        ex_date = df.index[3]  # 除息日
        divs = pd.DataFrame(
            {"cash_dividend": [2.0], "stock_dividend": [0.0]},
            index=[ex_date],
        )
        result = s._apply_dividend_adjustment(df, divs)
        # ex_date 之前的收盤價應被調降
        assert result["close"].iloc[0] < df["close"].iloc[0]
        # 保留 raw_close 為原始值
        assert "raw_close" in result.columns
        assert result["raw_close"].iloc[0] == df["close"].iloc[0]

    def test_stock_dividend_adjusts_price_down(self):
        s = self._make_strategy()
        df = self._make_price_df(n=5, base=100.0)
        ex_date = df.index[3]
        divs = pd.DataFrame(
            {"cash_dividend": [0.0], "stock_dividend": [1.0]},  # 每 10 股配 1 股
            index=[ex_date],
        )
        result = s._apply_dividend_adjustment(df, divs)
        assert result["close"].iloc[0] < df["close"].iloc[0]

    def test_mixed_dividend(self):
        s = self._make_strategy()
        df = self._make_price_df(n=5, base=100.0)
        ex_date = df.index[3]
        divs = pd.DataFrame(
            {"cash_dividend": [2.0], "stock_dividend": [0.5]},
            index=[ex_date],
        )
        result = s._apply_dividend_adjustment(df, divs)
        # 應該調得更多
        assert result["close"].iloc[0] < df["close"].iloc[0]
        # raw_close 不變
        np.testing.assert_array_equal(result["raw_close"].values, df["close"].values)
