"""測試 src/data/twse_fetcher.py 的工具函數。"""

from datetime import date

import pandas as pd
import pytest

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


class TestFetchTwseValuationAll:
    """測試 fetch_twse_valuation_all() — TWSE BWIBBU_d 估值資料解析。"""

    def _mock_resp(self, monkeypatch, json_data):
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = json_data
        monkeypatch.setattr("src.data.twse_fetcher.requests.get", lambda *a, **kw: mock)
        monkeypatch.setattr("src.data.twse_fetcher.time.sleep", lambda x: None)

    def test_parses_valid_response(self, monkeypatch):
        """正常回傳格式解析正確（PE/DY/PB 值對應）。"""
        from src.data.twse_fetcher import fetch_twse_valuation_all

        self._mock_resp(
            monkeypatch,
            {
                "stat": "OK",
                "fields": ["證券代號", "證券名稱", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"],
                "data": [
                    ["2330", "台積電", "1.02", "113", "21.50", "6.20", "113/Q3"],
                    ["2317", "鴻海", "5.30", "113", "11.20", "1.30", "113/Q3"],
                ],
            },
        )
        df = fetch_twse_valuation_all(date(2026, 3, 5))
        assert len(df) == 2
        assert set(df.columns) >= {"stock_id", "pe_ratio", "pb_ratio", "dividend_yield", "date"}
        row = df[df["stock_id"] == "2330"].iloc[0]
        assert row["pe_ratio"] == pytest.approx(21.50)
        assert row["dividend_yield"] == pytest.approx(1.02)
        assert row["pb_ratio"] == pytest.approx(6.20)

    def test_returns_empty_on_non_trading_day(self, monkeypatch):
        """stat != 'OK' 時（假日）回傳空 DataFrame。"""
        from src.data.twse_fetcher import fetch_twse_valuation_all

        self._mock_resp(monkeypatch, {"stat": "很抱歉，找不到符合條件的資料！"})
        df = fetch_twse_valuation_all(date(2026, 3, 7))
        assert df.empty

    def test_handles_double_dash_values(self, monkeypatch):
        """'--' 空值解析為 None，但有其他欄位的股票仍保留。"""
        from src.data.twse_fetcher import fetch_twse_valuation_all

        self._mock_resp(
            monkeypatch,
            {
                "stat": "OK",
                "data": [
                    ["2330", "台積電", "--", "113", "--", "6.20", "113/Q3"],
                ],
            },
        )
        df = fetch_twse_valuation_all(date(2026, 3, 5))
        assert len(df) == 1  # 有 pb_ratio=6.20，應保留
        assert df.iloc[0]["pe_ratio"] is None or pd.isna(df.iloc[0]["pe_ratio"])
        assert df.iloc[0]["pb_ratio"] == pytest.approx(6.20)

    def test_filters_non_4digit_stock_ids(self, monkeypatch):
        """非 4 碼數字的代號（如指數）應被過濾。"""
        from src.data.twse_fetcher import fetch_twse_valuation_all

        self._mock_resp(
            monkeypatch,
            {
                "stat": "OK",
                "data": [
                    ["2330", "台積電", "1.02", "113", "21.50", "6.20", "113/Q3"],
                    ["IX0001", "發行量加權股價指數", "--", "--", "--", "--", "--"],
                    ["00878", "國泰永續高股息", "5.00", "113", "--", "2.10", "113/Q3"],
                ],
            },
        )
        df = fetch_twse_valuation_all(date(2026, 3, 5))
        # 2330: 保留；IX0001: 非純數字 4 碼，過濾；00878: 4 碼但非純數字，過濾
        assert len(df) == 1
        assert df.iloc[0]["stock_id"] == "2330"

    def test_returns_empty_on_request_failure(self, monkeypatch):
        """HTTP 請求失敗應回傳空 DataFrame（不拋例外）。"""
        from src.data.twse_fetcher import fetch_twse_valuation_all

        def _raise(*a, **kw):
            raise ConnectionError("無法連線")

        monkeypatch.setattr("src.data.twse_fetcher.requests.get", _raise)
        monkeypatch.setattr("src.data.twse_fetcher.time.sleep", lambda x: None)
        df = fetch_twse_valuation_all(date(2026, 3, 5))
        assert df.empty
