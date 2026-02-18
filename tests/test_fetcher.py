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
