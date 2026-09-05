"""測試 src/data/fetcher.py — FinMindFetcher mock HTTP 測試。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestFinMindFetcher:
    @pytest.fixture(autouse=True)
    def _patch_settings(self, monkeypatch):
        """Mock settings 避免依賴真實 config。"""
        mock_settings = MagicMock()
        mock_settings.finmind.api_url = "https://api.finmindtrade.com/api/v4/data"
        mock_settings.finmind.api_token = "test_token"
        monkeypatch.setattr("src.data.fetcher.settings", mock_settings)

    def _make_fetcher(self):
        from src.data.fetcher import FinMindFetcher

        return FinMindFetcher(api_token="test_token")

    def test_fetch_daily_price_renames_columns(self, monkeypatch):
        fetcher = self._make_fetcher()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "msg": "success",
            "data": [
                {
                    "date": "2024-01-02",
                    "stock_id": "2330",
                    "open": 580.0,
                    "max": 585.0,
                    "min": 578.0,
                    "close": 583.0,
                    "Trading_Volume": 25000000,
                    "Trading_money": 14575000000,
                    "spread": 3.0,
                }
            ],
        }
        monkeypatch.setattr(fetcher._session, "get", lambda *a, **kw: mock_resp)
        monkeypatch.setattr("src.data.fetcher.time.sleep", lambda x: None)

        df = fetcher.fetch_daily_price("2330", "2024-01-01")
        assert "high" in df.columns
        assert "low" in df.columns
        assert "volume" in df.columns
        assert "max" not in df.columns
        assert "Trading_Volume" not in df.columns
        assert df.iloc[0]["close"] == 583.0

    def test_api_error_raises(self, monkeypatch):
        fetcher = self._make_fetcher()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"msg": "error", "status": 400}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr(fetcher._session, "get", lambda *a, **kw: mock_resp)
        monkeypatch.setattr("src.data.fetcher.time.sleep", lambda x: None)

        with pytest.raises(RuntimeError, match="FinMind API 錯誤"):
            fetcher.fetch_daily_price("2330", "2024-01-01")

    def test_empty_data_returns_empty_df(self, monkeypatch):
        fetcher = self._make_fetcher()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"msg": "success", "data": []}
        monkeypatch.setattr(fetcher._session, "get", lambda *a, **kw: mock_resp)
        monkeypatch.setattr("src.data.fetcher.time.sleep", lambda x: None)

        df = fetcher.fetch_daily_price("2330", "2024-01-01")
        assert df.empty

    def test_fetch_monthly_revenue_calculates_mom(self, monkeypatch):
        fetcher = self._make_fetcher()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "msg": "success",
            "data": [
                {
                    "date": "2024-01-10",
                    "stock_id": "2330",
                    "revenue": 200_000_000,
                    "revenue_month": 1,
                    "revenue_year": 2024,
                },
                {
                    "date": "2024-02-10",
                    "stock_id": "2330",
                    "revenue": 220_000_000,
                    "revenue_month": 2,
                    "revenue_year": 2024,
                },
                {
                    "date": "2024-03-10",
                    "stock_id": "2330",
                    "revenue": 250_000_000,
                    "revenue_month": 3,
                    "revenue_year": 2024,
                },
            ],
        }
        monkeypatch.setattr(fetcher._session, "get", lambda *a, **kw: mock_resp)
        monkeypatch.setattr("src.data.fetcher.time.sleep", lambda x: None)

        df = fetcher.fetch_monthly_revenue("2330", "2024-01-01")
        assert "mom_growth" in df.columns
        assert "yoy_growth" in df.columns
        # MoM for row 1: (220M - 200M) / 200M * 100 = 10%
        assert df.iloc[1]["mom_growth"] == pytest.approx(10.0, abs=0.1)

    def test_request_by_date_400_returns_empty(self, monkeypatch):
        fetcher = self._make_fetcher()

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"msg": "付費帳號限定"}
        monkeypatch.setattr(fetcher._session, "get", lambda *a, **kw: mock_resp)
        monkeypatch.setattr("src.data.fetcher.time.sleep", lambda x: None)

        df = fetcher._request_by_date("TaiwanStockPrice", "2024-01-01", "2024-01-31")
        assert df.empty


class TestDividendNaTHandling:
    """A-04 修復驗證：股利日期欄位中的 NaT 應正確轉為 None。"""

    @pytest.fixture(autouse=True)
    def _patch_settings(self, monkeypatch):
        mock_settings = MagicMock()
        mock_settings.finmind.api_url = "https://api.finmindtrade.com/api/v4/data"
        mock_settings.finmind.api_token = "test_token"
        monkeypatch.setattr("src.data.fetcher.settings", mock_settings)

    def _make_fetcher(self):
        from src.data.fetcher import FinMindFetcher

        return FinMindFetcher(api_token="test_token")

    def test_nat_cash_payment_date_becomes_none(self, monkeypatch):
        """cash_payment_date 為空字串或無效日期時應轉為 None，不可為 1970-01-01。"""
        from datetime import date

        fetcher = self._make_fetcher()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "msg": "success",
            "data": [
                {
                    "date": "2024-07-01",
                    "stock_id": "2330",
                    "year": 2024,
                    "cash_dividend": 3.5,
                    "stock_dividend": 0.0,
                    "cash_payment_date": "",  # 空字串 → NaT → 應轉 None
                    "announcement_date": "invalid-date",  # 無效日期 → NaT → 應轉 None
                },
                {
                    "date": "2024-01-02",
                    "stock_id": "2330",
                    "year": 2024,
                    "cash_dividend": 3.0,
                    "stock_dividend": 0.0,
                    "cash_payment_date": "2024-02-15",  # 有效日期
                    "announcement_date": "2023-12-01",
                },
            ],
        }
        monkeypatch.setattr(fetcher._session, "get", lambda *a, **kw: mock_resp)
        monkeypatch.setattr("src.data.fetcher.time.sleep", lambda x: None)

        df = fetcher.fetch_dividend("2330", "2024-01-01")

        # 第一筆：空/無效日期應為 None
        assert df.iloc[0]["cash_payment_date"] is None
        assert df.iloc[0]["announcement_date"] is None

        # 第二筆：有效日期應正確轉換
        assert df.iloc[1]["cash_payment_date"] == date(2024, 2, 15)
        assert df.iloc[1]["announcement_date"] == date(2023, 12, 1)


class TestMonthlyRevenueDateSemantics:
    """§6.6 #23：FinMind 月營收的 date 正規化與缺月安全的 MoM/YoY。"""

    @pytest.fixture(autouse=True)
    def _patch_settings(self, monkeypatch):
        mock_settings = MagicMock()
        mock_settings.finmind.api_url = "https://api.finmindtrade.com/api/v4/data"
        mock_settings.finmind.api_token = "test_token"
        monkeypatch.setattr("src.data.fetcher.settings", mock_settings)

    def _fetch(self, monkeypatch, rows):
        from src.data.fetcher import FinMindFetcher

        fetcher = FinMindFetcher(api_token="test_token")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"msg": "success", "data": rows}
        monkeypatch.setattr(fetcher._session, "get", lambda *a, **kw: mock_resp)
        monkeypatch.setattr("src.data.fetcher.time.sleep", lambda x: None)
        return fetcher.fetch_monthly_revenue("2330", "2020-01-01")

    def _row(self, year, month, revenue):
        # FinMind 原始慣例：date ＝ 次月 1 日
        nxt = (year + 1, 1) if month == 12 else (year, month + 1)
        return {
            "date": f"{nxt[0]:04d}-{nxt[1]:02d}-01",
            "stock_id": "2330",
            "revenue": revenue,
            "revenue_month": month,
            "revenue_year": year,
        }

    def test_date_normalized_to_month_end(self, monkeypatch):
        """次月 1 日必須改寫成營收月份的月底——否則與 MOPS 列同月並存。"""
        from datetime import date as _date

        df = self._fetch(monkeypatch, [self._row(2024, 1, 100), self._row(2024, 2, 110)])
        assert list(df["date"]) == [_date(2024, 1, 31), _date(2024, 2, 29)]

    def test_yoy_aligns_on_year_month_not_position(self, monkeypatch):
        """**核心回歸**：缺月時不得用位置位移當「去年同月」。

        FinMind 對未公布/停業月份會缺列，`shift(12)` 會把 12 列之前當成去年同月，
        算出張冠李戴的 YoY（且無從察覺）。
        """
        rows = [self._row(2023, m, 100) for m in range(1, 13)]
        rows.pop(5)  # 挖掉 2023-06，只剩 11 列
        rows.append(self._row(2024, 1, 150))

        df = self._fetch(monkeypatch, rows)
        latest = df[(df["revenue_year"] == 2024) & (df["revenue_month"] == 1)].iloc[0]
        assert latest["yoy_growth"] == pytest.approx(50.0), "應與 2023-01 相比 (150/100-1)"

    def test_missing_previous_month_gives_none_not_wrong_value(self, monkeypatch):
        df = self._fetch(monkeypatch, [self._row(2024, 1, 100), self._row(2024, 3, 130)])
        march = df[df["revenue_month"] == 3].iloc[0]
        assert march["mom_growth"] is None, "上月缺列時 MoM 必須是 None 而非跟 1 月比"

    def test_yoy_available_with_thirteen_months(self, monkeypatch):
        """13 個月即可算出最新一筆的 YoY（`REVENUE_FINMIND_LOOKBACK_DAYS`=430 的理由）。"""
        rows = [self._row(2023, m, 100) for m in range(1, 13)] + [self._row(2024, 1, 120)]
        df = self._fetch(monkeypatch, rows)
        assert df.iloc[-1]["yoy_growth"] == pytest.approx(20.0)


class TestQuotaStatus:
    """§6.6 #25：長跑逐股回補前的配額查詢。"""

    @pytest.fixture(autouse=True)
    def _patch_settings(self, monkeypatch):
        mock_settings = MagicMock()
        mock_settings.finmind.api_url = "https://api.finmindtrade.com/api/v4/data"
        mock_settings.finmind.api_token = "test_token"
        monkeypatch.setattr("src.data.fetcher.settings", mock_settings)

    def _fetcher(self, monkeypatch, payload, *, status=200, boom=False):
        from src.data.fetcher import FinMindFetcher

        f = FinMindFetcher(api_token="test_token")
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = payload
        resp.raise_for_status = MagicMock()

        def _get(*a, **kw):
            if boom:
                raise ConnectionError("network down")
            return resp

        monkeypatch.setattr(f._session, "get", _get)
        return f

    def test_parses_limit_and_usage(self, monkeypatch):
        f = self._fetcher(
            monkeypatch,
            {
                "msg": "success",
                "level": 1,
                "level_title": "Free",
                "api_request_limit": 600,
                "api_request_limit_hour": 600,
                "user_count": 232,
            },
        )
        q = f.fetch_quota_status()
        assert q["limit"] == 600
        assert q["used"] == 232
        assert q["remaining"] == 368
        assert q["level_title"] == "Free"

    def test_network_failure_returns_empty_not_raise(self, monkeypatch):
        """配額查詢掛掉不該讓 10 小時的回補作業中止——呼叫端會退回預設上限。"""
        f = self._fetcher(monkeypatch, {}, boom=True)
        assert f.fetch_quota_status() == {}

    def test_unexpected_payload_returns_empty(self, monkeypatch):
        f = self._fetcher(monkeypatch, {"msg": "token not valid"})
        assert f.fetch_quota_status() == {}

    def test_no_token_returns_empty(self, monkeypatch):
        from src.data.fetcher import FinMindFetcher

        mock_settings = MagicMock()
        mock_settings.finmind.api_url = "https://api.finmindtrade.com/api/v4/data"
        mock_settings.finmind.api_token = ""
        monkeypatch.setattr("src.data.fetcher.settings", mock_settings)
        assert FinMindFetcher(api_token="").fetch_quota_status() == {}
