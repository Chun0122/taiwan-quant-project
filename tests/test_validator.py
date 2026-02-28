"""tests/test_validator.py — 資料品質檢查純函數測試。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.data.validator import (
    check_data_freshness,
    check_date_range_consistency,
    check_limit_streaks,
    check_missing_days,
    check_price_anomalies,
    check_zero_volume,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_df(dates, close=None, volume=None, open_=None, high=None, low=None):
    """快速建立測試用 DataFrame（index=date）。"""
    data: dict = {}
    if close is not None:
        data["close"] = close
    if volume is not None:
        data["volume"] = volume
    if open_ is not None:
        data["open"] = open_
    if high is not None:
        data["high"] = high
    if low is not None:
        data["low"] = low
    df = pd.DataFrame(data, index=pd.to_datetime(dates))
    return df


# ===========================================================================
# TestCheckMissingDays
# ===========================================================================


class TestCheckMissingDays:
    """check_missing_days 的測試。"""

    def test_no_gap(self):
        """連續營業日，無缺漏。"""
        dates = [date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)]  # 一 ~ 三
        issues = check_missing_days("2330", dates, gap_threshold=3)
        assert issues == []

    def test_weekend_ignored(self):
        """週末不算缺漏。"""
        # 週五 → 週一，中間只有週末
        dates = [date(2025, 1, 3), date(2025, 1, 6)]
        issues = check_missing_days("2330", dates, gap_threshold=2)
        assert issues == []

    def test_small_gap_under_threshold(self):
        """小缺漏低於門檻，不報告。"""
        # 週一 → 下週一，缺 4 個營業日（二三四五）
        dates = [date(2025, 1, 6), date(2025, 1, 13)]
        issues = check_missing_days("2330", dates, gap_threshold=5)
        assert issues == []

    def test_gap_exceeds_threshold(self):
        """缺漏 >= 門檻，報 error。"""
        # 1/6(一) → 1/14(二)，缺 5 個營業日
        dates = [date(2025, 1, 6), date(2025, 1, 14)]
        issues = check_missing_days("2330", dates, gap_threshold=5)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].check_name == "missing_days"
        assert issues[0].details["gap"] == 5

    def test_multiple_gaps(self):
        """多段缺漏。"""
        dates = [date(2025, 1, 2), date(2025, 1, 14), date(2025, 1, 28)]
        issues = check_missing_days("2330", dates, gap_threshold=5)
        assert len(issues) == 2

    def test_single_date(self):
        """只有一天，無法計算差距。"""
        issues = check_missing_days("2330", [date(2025, 1, 6)], gap_threshold=3)
        assert issues == []

    def test_empty_dates(self):
        """空列表。"""
        issues = check_missing_days("2330", [], gap_threshold=3)
        assert issues == []


# ===========================================================================
# TestCheckZeroVolume
# ===========================================================================


class TestCheckZeroVolume:
    """check_zero_volume 的測試。"""

    def test_normal_volume(self):
        """正常成交量，無問題。"""
        dates = pd.bdate_range("2025-01-06", periods=5)
        df = _make_df(dates, volume=[1000, 2000, 3000, 4000, 5000])
        issues = check_zero_volume("2330", df)
        assert issues == []

    def test_single_zero(self):
        """單日零成交量，warning。"""
        dates = pd.bdate_range("2025-01-06", periods=5)
        df = _make_df(dates, volume=[1000, 0, 3000, 4000, 5000])
        issues = check_zero_volume("2330", df)
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_consecutive_zero_error(self):
        """連續 3+ 天零成交量，error。"""
        dates = pd.bdate_range("2025-01-06", periods=5)
        df = _make_df(dates, volume=[1000, 0, 0, 0, 5000])
        issues = check_zero_volume("2330", df)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "3" in issues[0].description

    def test_scattered_zeros(self):
        """分散的零成交量（各自 warning）。"""
        dates = pd.bdate_range("2025-01-06", periods=5)
        df = _make_df(dates, volume=[0, 1000, 0, 1000, 0])
        issues = check_zero_volume("2330", df)
        assert len(issues) == 3
        assert all(i.severity == "warning" for i in issues)

    def test_empty_df(self):
        """空 DataFrame。"""
        df = pd.DataFrame(columns=["volume"])
        issues = check_zero_volume("2330", df)
        assert issues == []


# ===========================================================================
# TestCheckLimitStreaks
# ===========================================================================


class TestCheckLimitStreaks:
    """check_limit_streaks 的測試。"""

    def test_normal_returns(self):
        """正常報酬率，無漲跌停。"""
        dates = pd.bdate_range("2025-01-06", periods=10)
        closes = [100 + i * 0.5 for i in range(10)]
        df = _make_df(dates, close=closes)
        issues = check_limit_streaks("2330", df, streak_threshold=3)
        assert issues == []

    def test_single_limit(self):
        """單日漲停，不到門檻。"""
        dates = pd.bdate_range("2025-01-06", periods=5)
        closes = [100, 110, 111, 112, 113]  # 第二天 +10%
        df = _make_df(dates, close=closes)
        issues = check_limit_streaks("2330", df, streak_threshold=3)
        assert issues == []

    def test_consecutive_up_limit(self):
        """連續漲停 >= streak_threshold。"""
        dates = pd.bdate_range("2025-01-06", periods=6)
        # 每天漲 10%
        closes = [100]
        for _ in range(5):
            closes.append(closes[-1] * 1.10)
        df = _make_df(dates, close=closes)
        issues = check_limit_streaks("2330", df, streak_threshold=5)
        assert len(issues) == 1
        assert "漲停" in issues[0].description

    def test_consecutive_down_limit(self):
        """連續跌停 >= streak_threshold。"""
        dates = pd.bdate_range("2025-01-06", periods=6)
        closes = [100]
        for _ in range(5):
            closes.append(closes[-1] * 0.90)
        df = _make_df(dates, close=closes)
        issues = check_limit_streaks("2330", df, streak_threshold=5)
        assert len(issues) == 1
        assert "跌停" in issues[0].description

    def test_mixed_no_streak(self):
        """混合漲跌停但方向不同，不算連續。"""
        dates = pd.bdate_range("2025-01-06", periods=5)
        closes = [100, 110, 99, 109, 98]  # 交替漲跌停
        df = _make_df(dates, close=closes)
        issues = check_limit_streaks("2330", df, streak_threshold=2)
        assert issues == []

    def test_custom_threshold(self):
        """自訂門檻。"""
        dates = pd.bdate_range("2025-01-06", periods=4)
        closes = [100, 110, 121, 133.1]  # 每天 +10%
        df = _make_df(dates, close=closes)
        # streak_threshold=3，應報告
        issues = check_limit_streaks("2330", df, streak_threshold=3)
        assert len(issues) == 1


# ===========================================================================
# TestCheckPriceAnomalies
# ===========================================================================


class TestCheckPriceAnomalies:
    """check_price_anomalies 的測試。"""

    def test_normal_ohlc(self):
        """正常 OHLC，無異常。"""
        dates = pd.bdate_range("2025-01-06", periods=3)
        df = _make_df(
            dates,
            open_=[100, 101, 102],
            high=[105, 106, 107],
            low=[98, 99, 100],
            close=[103, 104, 105],
        )
        issues = check_price_anomalies("2330", df)
        assert issues == []

    def test_high_less_than_low(self):
        """high < low。"""
        dates = pd.bdate_range("2025-01-06", periods=3)
        df = _make_df(
            dates,
            open_=[100, 101, 102],
            high=[105, 98, 107],  # 第 2 天 high < low
            low=[98, 99, 100],
            close=[103, 97, 105],
        )
        issues = check_price_anomalies("2330", df)
        assert any(i.details.get("type") == "high_lt_low" for i in issues)

    def test_close_out_of_range(self):
        """close 超出 [low, high] 範圍。"""
        dates = pd.bdate_range("2025-01-06", periods=3)
        df = _make_df(
            dates,
            open_=[100, 101, 102],
            high=[105, 106, 107],
            low=[98, 99, 100],
            close=[103, 110, 105],  # 第 2 天 close > high
        )
        issues = check_price_anomalies("2330", df)
        assert any(i.details.get("type") == "close_out_of_range" for i in issues)

    def test_negative_price(self):
        """負價格。"""
        dates = pd.bdate_range("2025-01-06", periods=2)
        df = _make_df(
            dates,
            open_=[100, -5],
            high=[105, 106],
            low=[98, 99],
            close=[103, 104],
        )
        issues = check_price_anomalies("2330", df)
        assert any(i.details.get("type") == "non_positive" for i in issues)

    def test_multiple_anomalies(self):
        """多種異常同時出現。"""
        dates = pd.bdate_range("2025-01-06", periods=2)
        df = _make_df(
            dates,
            open_=[100, -5],
            high=[105, 90],  # high < low
            low=[98, 99],
            close=[103, 104],
        )
        issues = check_price_anomalies("2330", df)
        assert len(issues) >= 2


# ===========================================================================
# TestCheckDateRangeConsistency
# ===========================================================================


class TestCheckDateRangeConsistency:
    """check_date_range_consistency 的測試。"""

    def test_consistent(self):
        """各表日期範圍一致。"""
        ranges = {
            "daily_price": {"2330": (date(2020, 1, 2), date(2025, 1, 10))},
            "institutional_investor": {"2330": (date(2020, 1, 2), date(2025, 1, 10))},
        }
        issues = check_date_range_consistency(ranges)
        assert issues == []

    def test_small_diff_ok(self):
        """差距 <= 30 天，不報告。"""
        ranges = {
            "daily_price": {"2330": (date(2020, 1, 2), date(2025, 1, 31))},
            "institutional_investor": {"2330": (date(2020, 1, 2), date(2025, 1, 10))},
        }
        issues = check_date_range_consistency(ranges)
        assert issues == []

    def test_large_diff_warning(self):
        """差距 > 30 天，報 warning。"""
        ranges = {
            "daily_price": {"2330": (date(2020, 1, 2), date(2025, 6, 30))},
            "institutional_investor": {"2330": (date(2020, 1, 2), date(2025, 1, 10))},
        }
        issues = check_date_range_consistency(ranges)
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_no_base_table(self):
        """缺少 daily_price 基礎表，跳過。"""
        ranges = {
            "institutional_investor": {"2330": (date(2020, 1, 2), date(2025, 1, 10))},
        }
        issues = check_date_range_consistency(ranges)
        assert issues == []


# ===========================================================================
# TestCheckDataFreshness
# ===========================================================================


class TestCheckDataFreshness:
    """check_data_freshness 的測試。"""

    def test_fresh_data(self):
        """資料新鮮（距今 < 門檻）。"""
        ref = date(2025, 1, 10)
        ranges = {"daily_price": (date(2020, 1, 2), date(2025, 1, 8))}
        issues = check_data_freshness(ranges, ref, stale_threshold=7)
        assert issues == []

    def test_stale_data(self):
        """資料過期（距今 > 門檻）。"""
        ref = date(2025, 2, 28)
        ranges = {"daily_price": (date(2020, 1, 2), date(2025, 1, 10))}
        issues = check_data_freshness(ranges, ref, stale_threshold=7)
        assert len(issues) == 1
        assert issues[0].check_name == "data_freshness"

    def test_empty_table(self):
        """空表（max_date=None）。"""
        ref = date(2025, 1, 10)
        ranges = {"daily_price": (None, None)}
        issues = check_data_freshness(ranges, ref, stale_threshold=7)
        assert len(issues) == 1
        assert "無資料" in issues[0].description
