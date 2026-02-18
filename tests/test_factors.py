"""測試 src/screener/factors.py 的 8 個篩選因子。"""

import pandas as pd

from src.screener.factors import (
    foreign_net_buy,
    institutional_consecutive_buy,
    macd_golden_cross,
    price_above_sma,
    revenue_consecutive_growth,
    revenue_yoy_growth,
    rsi_oversold,
    short_squeeze_ratio,
)

# ─── RSI 超賣 ─────────────────────────────────────────────


class TestRsiOversold:
    def test_below_threshold_returns_true(self):
        df = pd.DataFrame({"rsi_14": [25.0, 29.9, 15.0]})
        result = rsi_oversold(df)
        assert result.tolist() == [True, True, True]

    def test_above_threshold_returns_false(self):
        df = pd.DataFrame({"rsi_14": [30.0, 50.0, 70.0]})
        result = rsi_oversold(df)
        assert result.tolist() == [False, False, False]

    def test_missing_column_returns_false(self):
        df = pd.DataFrame({"close": [100, 200]})
        result = rsi_oversold(df)
        assert result.tolist() == [False, False]

    def test_custom_threshold(self):
        df = pd.DataFrame({"rsi_14": [25.0, 35.0]})
        result = rsi_oversold(df, threshold=40)
        assert result.tolist() == [True, True]


# ─── MACD 黃金交叉 ────────────────────────────────────────


class TestMacdGoldenCross:
    def test_crossover_detected(self):
        df = pd.DataFrame(
            {
                "macd": [-1.0, 0.5, 1.0],
                "macd_signal": [0.0, 0.0, 0.0],
            }
        )
        result = macd_golden_cross(df)
        # 第 0 天: prev=NaN → False
        # 第 1 天: prev_diff=-1 <= 0, diff=0.5 > 0 → True
        # 第 2 天: prev_diff=0.5 > 0, diff=1.0 > 0 → False (已在上方)
        assert result.tolist() == [False, True, False]

    def test_no_crossover(self):
        df = pd.DataFrame(
            {
                "macd": [1.0, 2.0, 3.0],
                "macd_signal": [0.0, 0.0, 0.0],
            }
        )
        result = macd_golden_cross(df)
        assert not result.any()

    def test_missing_columns_returns_false(self):
        df = pd.DataFrame({"close": [100, 200]})
        result = macd_golden_cross(df)
        assert result.tolist() == [False, False]


# ─── 股價站上 SMA ──────────────────────────────────────────


class TestPriceAboveSma:
    def test_above_sma(self):
        df = pd.DataFrame({"close": [110, 90], "sma_20": [100, 100]})
        result = price_above_sma(df)
        assert result.tolist() == [True, False]

    def test_missing_column(self):
        df = pd.DataFrame({"close": [110]})
        result = price_above_sma(df)
        assert result.tolist() == [False]

    def test_custom_period(self):
        df = pd.DataFrame({"close": [110], "sma_60": [100]})
        result = price_above_sma(df, period=60)
        assert result.tolist() == [True]


# ─── 外資買超 ──────────────────────────────────────────────


class TestForeignNetBuy:
    def test_positive_net_buy(self):
        df = pd.DataFrame({"foreign_net": [1000, -500, 0]})
        result = foreign_net_buy(df)
        assert result.tolist() == [True, False, False]

    def test_missing_column(self):
        df = pd.DataFrame({"close": [100]})
        result = foreign_net_buy(df)
        assert result.tolist() == [False]


# ─── 法人連續買超 ──────────────────────────────────────────


class TestInstitutionalConsecutiveBuy:
    def test_consecutive_3_days(self):
        df = pd.DataFrame(
            {
                "foreign_net": [100, 200, 300, 400, -100],
                "trust_net": [0, 0, 0, 0, 0],
                "dealer_net": [0, 0, 0, 0, 0],
            }
        )
        result = institutional_consecutive_buy(df, days=3)
        assert result.tolist() == [False, False, True, True, False]

    def test_no_columns(self):
        df = pd.DataFrame({"close": [100, 200, 300]})
        result = institutional_consecutive_buy(df)
        assert result.tolist() == [False, False, False]

    def test_partial_columns(self):
        df = pd.DataFrame({"foreign_net": [100, 200, 300]})
        result = institutional_consecutive_buy(df, days=3)
        assert bool(result.iloc[-1]) is True


# ─── 券資比 ───────────────────────────────────────────────


class TestShortSqueezeRatio:
    def test_above_threshold(self):
        df = pd.DataFrame(
            {
                "short_balance": [300, 100],
                "margin_balance": [1000, 1000],
            }
        )
        result = short_squeeze_ratio(df, threshold=0.2)
        assert result.tolist() == [True, False]

    def test_zero_margin_returns_false(self):
        df = pd.DataFrame(
            {
                "short_balance": [300],
                "margin_balance": [0],
            }
        )
        result = short_squeeze_ratio(df)
        # margin=0 → NaN → comparison with NaN → False
        assert result.tolist() == [False]

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [100]})
        result = short_squeeze_ratio(df)
        assert result.tolist() == [False]


# ─── 營收 YoY ─────────────────────────────────────────────


class TestRevenueYoyGrowth:
    def test_above_threshold(self):
        df = pd.DataFrame({"yoy_growth": [25.0, 10.0, 50.0]})
        result = revenue_yoy_growth(df)
        assert result.tolist() == [True, False, True]

    def test_missing_column(self):
        df = pd.DataFrame({"close": [100]})
        result = revenue_yoy_growth(df)
        assert result.tolist() == [False]


# ─── 連續營收成長 ──────────────────────────────────────────


class TestRevenueConsecutiveGrowth:
    def test_3_months_growth(self):
        df = pd.DataFrame({"mom_growth": [5.0, 3.0, 2.0, 1.0, -1.0]})
        result = revenue_consecutive_growth(df, months=3)
        assert result.tolist() == [False, False, True, True, False]

    def test_missing_column(self):
        df = pd.DataFrame({"close": [100]})
        result = revenue_consecutive_growth(df)
        assert result.tolist() == [False]
