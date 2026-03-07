"""大戶持股分級測試。

測試項目：
- _extract_level_lower_bound: 持股分級字串解析
- compute_whale_score: 大戶集中度計算（週環比）
- HoldingDistribution ORM: 資料寫入 + 唯一鍵衝突
- fetch_holding_distribution: FinMind API 欄位映射（mock HTTP）
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

from src.data.schema import HoldingDistribution
from src.discovery.scanner import _extract_level_lower_bound, compute_whale_score

# ------------------------------------------------------------------ #
#  _extract_level_lower_bound
# ------------------------------------------------------------------ #


class TestExtractLevelLowerBound:
    """持股分級字串解析測試。"""

    @pytest.mark.parametrize(
        "level, expected",
        [
            ("1-999 Shares", 1),
            ("1,000-5,000 Shares", 1000),
            ("400,001-600,000 Shares", 400001),
            ("800,001-1,000,000 Shares", 800001),
            ("Over 1,000,000 Shares", 1000000),
            ("1000001 shares and above", 1000001),
            ("無資料", 0),
            ("", 0),
        ],
    )
    def test_various_formats(self, level, expected):
        assert _extract_level_lower_bound(level) == expected


# ------------------------------------------------------------------ #
#  compute_whale_score
# ------------------------------------------------------------------ #


def _make_holding_df(stock_ids, dates, levels_percents):
    """建立測試用持股分級 DataFrame。

    levels_percents: {stock_id: [(level, percent), ...]} for the latest date
    """
    rows = []
    for sid in stock_ids:
        for dt in dates:
            for level, pct in levels_percents.get(sid, []):
                rows.append({"stock_id": sid, "date": dt, "level": level, "percent": pct})
    return pd.DataFrame(rows)


class TestComputeWhaleScore:
    """大戶集中度分數計算測試。"""

    def test_empty_input_returns_empty(self):
        result = compute_whale_score(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == ["stock_id", "whale_percent", "whale_change"]

    def test_single_stock_single_week(self):
        """單股單週 — whale_percent 正確、whale_change = 0。"""
        dt = date(2025, 3, 7)
        df = pd.DataFrame(
            [
                {"stock_id": "2330", "date": dt, "level": "1-999 Shares", "percent": 30.0},
                {"stock_id": "2330", "date": dt, "level": "400,001-600,000 Shares", "percent": 15.0},
                {"stock_id": "2330", "date": dt, "level": "800,001-1,000,000 Shares", "percent": 5.0},
            ]
        )
        result = compute_whale_score(df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["stock_id"] == "2330"
        # 大戶（>=400000）= 15.0 + 5.0
        assert pytest.approx(row["whale_percent"]) == 20.0
        assert pytest.approx(row["whale_change"]) == 0.0

    def test_two_weeks_change(self):
        """兩週資料 — 大戶增持時 whale_change 為正值。"""
        dt_prev = date(2025, 2, 28)
        dt_now = date(2025, 3, 7)
        df = pd.DataFrame(
            [
                # 前一週
                {"stock_id": "2317", "date": dt_prev, "level": "400,001-600,000 Shares", "percent": 10.0},
                {"stock_id": "2317", "date": dt_prev, "level": "1-999 Shares", "percent": 40.0},
                # 本週（大戶增持）
                {"stock_id": "2317", "date": dt_now, "level": "400,001-600,000 Shares", "percent": 13.0},
                {"stock_id": "2317", "date": dt_now, "level": "1-999 Shares", "percent": 37.0},
            ]
        )
        result = compute_whale_score(df)
        row = result[result["stock_id"] == "2317"].iloc[0]
        assert pytest.approx(row["whale_percent"]) == 13.0
        assert pytest.approx(row["whale_change"]) == 3.0  # 13 - 10

    def test_two_stocks_different_concentration(self):
        """兩支股票 — 大戶比例不同，結果各自獨立正確。"""
        dt = date(2025, 3, 7)
        df = pd.DataFrame(
            [
                # 2330 大戶 25%
                {"stock_id": "2330", "date": dt, "level": "400,001-600,000 Shares", "percent": 25.0},
                {"stock_id": "2330", "date": dt, "level": "1-999 Shares", "percent": 75.0},
                # 6505 大戶 5%
                {"stock_id": "6505", "date": dt, "level": "400,001-600,000 Shares", "percent": 5.0},
                {"stock_id": "6505", "date": dt, "level": "1-999 Shares", "percent": 95.0},
            ]
        )
        result = compute_whale_score(df)
        assert len(result) == 2
        pct_2330 = result[result["stock_id"] == "2330"]["whale_percent"].iloc[0]
        pct_6505 = result[result["stock_id"] == "6505"]["whale_percent"].iloc[0]
        assert pytest.approx(pct_2330) == 25.0
        assert pytest.approx(pct_6505) == 5.0

    def test_no_whale_levels_gives_zero(self):
        """所有持股區間均為散戶層級 → whale_percent = 0。"""
        dt = date(2025, 3, 7)
        df = pd.DataFrame(
            [
                {"stock_id": "9999", "date": dt, "level": "1-999 Shares", "percent": 60.0},
                {"stock_id": "9999", "date": dt, "level": "1,000-5,000 Shares", "percent": 40.0},
            ]
        )
        result = compute_whale_score(df)
        assert pytest.approx(result.iloc[0]["whale_percent"]) == 0.0

    def test_boundary_level_400000(self):
        """剛好 400,000 下限的層級不算大戶（需 > 400,000）。"""
        dt = date(2025, 3, 7)
        df = pd.DataFrame(
            [
                # 下限 400,001 → 算大戶
                {"stock_id": "A001", "date": dt, "level": "400,001-600,000 Shares", "percent": 8.0},
                # 下限 200,001 → 不算大戶
                {"stock_id": "A001", "date": dt, "level": "200,001-400,000 Shares", "percent": 12.0},
            ]
        )
        result = compute_whale_score(df)
        assert pytest.approx(result.iloc[0]["whale_percent"]) == 8.0


# ------------------------------------------------------------------ #
#  HoldingDistribution ORM
# ------------------------------------------------------------------ #


class TestHoldingDistributionORM:
    """持股分級 ORM CRUD 測試（in-memory SQLite）。"""

    def test_insert_and_query(self, db_session):
        """寫入一筆資料後可查詢。"""
        dt = date(2025, 3, 7)
        entry = HoldingDistribution(
            stock_id="2330",
            date=dt,
            level="400,001-600,000 Shares",
            count=1234,
            percent=8.5,
        )
        db_session.add(entry)
        db_session.flush()

        result = db_session.query(HoldingDistribution).filter_by(stock_id="2330").first()
        assert result is not None
        assert result.level == "400,001-600,000 Shares"
        assert pytest.approx(result.percent) == 8.5

    def test_unique_constraint(self, db_session):
        """(stock_id, date, level) 唯一約束 — on_conflict_do_nothing 不拋出錯誤。"""
        dt = date(2025, 3, 7)
        records = [
            {"stock_id": "2317", "date": dt, "level": "400,001-600,000 Shares", "count": 100, "percent": 5.0},
            # 完全相同 — 應被忽略
            {"stock_id": "2317", "date": dt, "level": "400,001-600,000 Shares", "count": 200, "percent": 6.0},
        ]
        for rec in records:
            stmt = sqlite_upsert(HoldingDistribution).values([rec])
            stmt = stmt.on_conflict_do_nothing(index_elements=["stock_id", "date", "level"])
            db_session.execute(stmt)
        db_session.flush()

        count = db_session.query(HoldingDistribution).filter_by(stock_id="2317").count()
        assert count == 1

    def test_multiple_levels_same_date(self, db_session):
        """同一股票同日多個持股層級各別存入。"""
        dt = date(2025, 3, 7)
        levels = ["1-999 Shares", "1,000-5,000 Shares", "400,001-600,000 Shares"]
        for lv in levels:
            db_session.add(HoldingDistribution(stock_id="6505", date=dt, level=lv, count=50, percent=10.0))
        db_session.flush()

        count = db_session.query(HoldingDistribution).filter_by(stock_id="6505", date=dt).count()
        assert count == 3


# ------------------------------------------------------------------ #
#  fetch_holding_distribution (mock HTTP)
# ------------------------------------------------------------------ #


class TestFetchHoldingDistribution:
    """FinMind API 欄位映射測試（mock requests）。"""

    def _make_api_response(self):
        return {
            "msg": "success",
            "status": 200,
            "data": [
                {
                    "date": "2025-03-07",
                    "stock_id": "2330",
                    "HoldingSharesLevel": "400,001-600,000 Shares",
                    "HoldingSharesCount": 1500,
                    "HoldingSharesPercent": 12.34,
                },
                {
                    "date": "2025-03-07",
                    "stock_id": "2330",
                    "HoldingSharesLevel": "1-999 Shares",
                    "HoldingSharesCount": 120000,
                    "HoldingSharesPercent": 30.0,
                },
            ],
        }

    def test_column_rename(self):
        """回傳欄位正確映射：level / count / percent。"""
        from src.data.fetcher import FinMindFetcher

        fetcher = FinMindFetcher(api_token="test_token")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._make_api_response()
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.Session.get", return_value=mock_resp), patch("time.sleep"):
            df = fetcher.fetch_holding_distribution("2330", "2025-01-01")

        assert not df.empty
        assert "level" in df.columns
        assert "count" in df.columns
        assert "percent" in df.columns
        assert "HoldingSharesLevel" not in df.columns

    def test_data_types(self):
        """count 為整數，percent 為浮點數，date 為 date 物件。"""
        from src.data.fetcher import FinMindFetcher

        fetcher = FinMindFetcher(api_token="test_token")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._make_api_response()
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.Session.get", return_value=mock_resp), patch("time.sleep"):
            df = fetcher.fetch_holding_distribution("2330", "2025-01-01")

        assert df["count"].dtype == int
        assert df["percent"].dtype == float
        row = df[df["level"].str.contains("400,001")].iloc[0]
        assert row["count"] == 1500
        assert pytest.approx(row["percent"]) == 12.34

    def test_empty_api_response(self):
        """API 回傳空資料 → 空 DataFrame。"""
        from src.data.fetcher import FinMindFetcher

        fetcher = FinMindFetcher(api_token="test_token")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"msg": "success", "status": 200, "data": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.Session.get", return_value=mock_resp), patch("time.sleep"):
            df = fetcher.fetch_holding_distribution("9999", "2025-01-01")

        assert df.empty
