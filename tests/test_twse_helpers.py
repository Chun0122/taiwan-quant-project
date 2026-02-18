"""測試 src/data/twse_fetcher.py 的工具函數。"""

from datetime import date

from src.data.twse_fetcher import _find_last_trading_day, _parse_number, _to_roc_date


class TestParseNumber:
    def test_comma_separated(self):
        assert _parse_number("1,234.56") == 1234.56

    def test_large_number(self):
        assert _parse_number("1,234,567,890") == 1234567890.0

    def test_dash_returns_none(self):
        assert _parse_number("--") is None

    def test_four_dashes(self):
        assert _parse_number("----") is None

    def test_empty_string(self):
        assert _parse_number("") is None

    def test_none_input(self):
        assert _parse_number(None) is None

    def test_special_strings(self):
        assert _parse_number("除權息") is None
        assert _parse_number("除權") is None
        assert _parse_number("除息") is None
        assert _parse_number("---") is None

    def test_plain_number(self):
        assert _parse_number("123.45") == 123.45

    def test_integer_string(self):
        assert _parse_number("100") == 100.0

    def test_negative_number(self):
        assert _parse_number("-5.30") == -5.30


class TestToRocDate:
    def test_standard_date(self):
        assert _to_roc_date(date(2026, 2, 18)) == "115/02/18"

    def test_year_2024(self):
        assert _to_roc_date(date(2024, 1, 1)) == "113/01/01"

    def test_year_2000(self):
        assert _to_roc_date(date(2000, 12, 31)) == "89/12/31"

    def test_single_digit_month_day(self):
        assert _to_roc_date(date(2025, 3, 5)) == "114/03/05"


class TestFindLastTradingDay:
    def test_weekday_returns_same(self):
        # 2026-02-18 is Wednesday
        result = _find_last_trading_day(date(2026, 2, 18))
        assert result == date(2026, 2, 18)

    def test_saturday_returns_friday(self):
        # 2026-02-21 is Saturday
        result = _find_last_trading_day(date(2026, 2, 21))
        assert result == date(2026, 2, 20)  # Friday

    def test_sunday_returns_friday(self):
        # 2026-02-22 is Sunday
        result = _find_last_trading_day(date(2026, 2, 22))
        assert result == date(2026, 2, 20)  # Friday

    def test_monday_returns_same(self):
        result = _find_last_trading_day(date(2026, 2, 16))
        assert result == date(2026, 2, 16)
