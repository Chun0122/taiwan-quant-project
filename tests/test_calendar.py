"""tests/test_calendar.py — TWSE 交易日行事曆測試。"""

from __future__ import annotations

from datetime import date

from src.data.calendar import (
    get_trading_days,
    has_calendar_data,
    is_trading_day,
    is_twse_holiday,
    next_trading_day,
    prev_trading_day,
)


class TestIsTradingDay:
    """is_trading_day 基本判斷。"""

    def test_regular_weekday(self):
        """一般週二 → 交易日。"""
        assert is_trading_day(date(2025, 3, 4)) is True

    def test_saturday(self):
        """週六 → 非交易日。"""
        assert is_trading_day(date(2025, 3, 1)) is False

    def test_sunday(self):
        """週日 → 非交易日。"""
        assert is_trading_day(date(2025, 3, 2)) is False

    def test_new_year(self):
        """2025 元旦 → 非交易日。"""
        assert is_trading_day(date(2025, 1, 1)) is False

    def test_lunar_new_year(self):
        """2025 春節（1/28~1/31）→ 非交易日。"""
        assert is_trading_day(date(2025, 1, 28)) is False
        assert is_trading_day(date(2025, 1, 29)) is False
        assert is_trading_day(date(2025, 1, 30)) is False
        assert is_trading_day(date(2025, 1, 31)) is False

    def test_day_after_holiday(self):
        """2025/2/3（週一）→ 春節後第一個交易日。"""
        assert is_trading_day(date(2025, 2, 3)) is True


class TestIsTwseHoliday:
    """is_twse_holiday 判斷。"""

    def test_known_holiday(self):
        """已知假日。"""
        assert is_twse_holiday(date(2025, 4, 4)) is True

    def test_not_holiday(self):
        """非假日。"""
        assert is_twse_holiday(date(2025, 3, 5)) is False

    def test_unknown_year_returns_false(self):
        """未建立年份 → False。"""
        assert is_twse_holiday(date(2030, 1, 1)) is False


class TestNextPrevTradingDay:
    """next_trading_day / prev_trading_day。"""

    def test_next_from_friday(self):
        """週五 → 下一交易日 = 週一。"""
        assert next_trading_day(date(2025, 2, 28)) == date(2025, 3, 3)
        # 2/28 是 TWSE 假日（和平紀念日），3/1=六, 3/2=日

    def test_prev_from_monday(self):
        """週一 → 前一交易日 = 上週五。"""
        assert prev_trading_day(date(2025, 3, 3)) == date(2025, 2, 27)
        # 2/28 是假日 → 2/27

    def test_next_skips_holiday(self):
        """跨過假日。"""
        # 2025/1/27(一) 除夕彈性 → 下一交易日 = 2/3(一)
        nxt = next_trading_day(date(2025, 1, 24))  # 1/24 = 週五
        assert nxt == date(2025, 2, 3)


class TestGetTradingDays:
    """get_trading_days 區間查詢。"""

    def test_one_week(self):
        """一般週 5 天。"""
        days = get_trading_days(date(2025, 3, 3), date(2025, 3, 7))
        assert len(days) == 5

    def test_holiday_week(self):
        """含假日的週。"""
        # 2025/2/24~2/28：2/28 為假日 → 4 天
        days = get_trading_days(date(2025, 2, 24), date(2025, 2, 28))
        assert date(2025, 2, 28) not in days
        assert len(days) == 4


class TestHasCalendarData:
    """has_calendar_data 年份檢查。"""

    def test_2025_exists(self):
        assert has_calendar_data(2025) is True

    def test_2026_exists(self):
        assert has_calendar_data(2026) is True

    def test_2030_not_exists(self):
        assert has_calendar_data(2030) is False
