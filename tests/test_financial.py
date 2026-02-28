"""財報資料同步測試 — fetcher pivot、衍生比率計算、pipeline upsert。"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.fetcher import FinMindFetcher, compute_financial_ratios
from src.data.schema import FinancialStatement

# ------------------------------------------------------------------ #
#  compute_financial_ratios 純函數測試
# ------------------------------------------------------------------ #


class TestComputeFinancialRatios:
    """衍生比率計算（純函數）。"""

    def test_gross_margin(self):
        df = pd.DataFrame({"revenue": [1000], "gross_profit": [300]})
        result = compute_financial_ratios(df)
        assert result["gross_margin"].iloc[0] == pytest.approx(30.0)

    def test_operating_margin(self):
        df = pd.DataFrame({"revenue": [2000], "operating_income": [400]})
        result = compute_financial_ratios(df)
        assert result["operating_margin"].iloc[0] == pytest.approx(20.0)

    def test_net_margin(self):
        df = pd.DataFrame({"revenue": [5000], "net_income": [500]})
        result = compute_financial_ratios(df)
        assert result["net_margin"].iloc[0] == pytest.approx(10.0)

    def test_roe(self):
        df = pd.DataFrame({"net_income": [100], "equity": [1000]})
        result = compute_financial_ratios(df)
        assert result["roe"].iloc[0] == pytest.approx(10.0)

    def test_roa(self):
        df = pd.DataFrame({"net_income": [50], "total_assets": [2000]})
        result = compute_financial_ratios(df)
        assert result["roa"].iloc[0] == pytest.approx(2.5)

    def test_debt_ratio(self):
        df = pd.DataFrame({"total_liabilities": [600], "total_assets": [1000]})
        result = compute_financial_ratios(df)
        assert result["debt_ratio"].iloc[0] == pytest.approx(60.0)

    def test_free_cf(self):
        df = pd.DataFrame({"operating_cf": [500], "investing_cf": [-200]})
        result = compute_financial_ratios(df)
        assert result["free_cf"].iloc[0] == pytest.approx(300)

    def test_zero_revenue_returns_none(self):
        df = pd.DataFrame({"revenue": [0], "gross_profit": [100]})
        result = compute_financial_ratios(df)
        assert result["gross_margin"].iloc[0] is None

    def test_none_values_handled(self):
        df = pd.DataFrame({"revenue": [None], "gross_profit": [None]})
        result = compute_financial_ratios(df)
        assert result["gross_margin"].iloc[0] is None

    def test_all_ratios_combined(self):
        """完整的一筆財報資料，驗證所有比率同時計算。"""
        df = pd.DataFrame(
            {
                "revenue": [10000],
                "gross_profit": [3500],
                "operating_income": [2000],
                "net_income": [1500],
                "total_assets": [50000],
                "total_liabilities": [20000],
                "equity": [30000],
                "operating_cf": [3000],
                "investing_cf": [-1000],
            }
        )
        result = compute_financial_ratios(df)
        assert result["gross_margin"].iloc[0] == pytest.approx(35.0)
        assert result["operating_margin"].iloc[0] == pytest.approx(20.0)
        assert result["net_margin"].iloc[0] == pytest.approx(15.0)
        assert result["roe"].iloc[0] == pytest.approx(5.0)
        assert result["roa"].iloc[0] == pytest.approx(3.0)
        assert result["debt_ratio"].iloc[0] == pytest.approx(40.0)
        assert result["free_cf"].iloc[0] == pytest.approx(2000)


# ------------------------------------------------------------------ #
#  _pivot_eav 測試
# ------------------------------------------------------------------ #


class TestPivotEav:
    """EAV → 寬表 pivot 邏輯。"""

    def setup_method(self):
        self.fetcher = FinMindFetcher.__new__(FinMindFetcher)

    def test_pivot_income_statement(self):
        """模擬損益表 EAV 回傳，驗證 pivot 結果。"""
        eav_data = pd.DataFrame(
            {
                "date": ["2024-03-31"] * 5,
                "stock_id": ["2330"] * 5,
                "type": ["Revenue", "GrossProfit", "OperatingIncome", "IncomeAfterTaxes", "EPS"],
                "value": [100000, 30000, 20000, 15000, 3.5],
            }
        )
        result = self.fetcher._pivot_eav(eav_data, FinMindFetcher._INCOME_TYPES)

        assert len(result) == 1
        assert result["revenue"].iloc[0] == pytest.approx(100000)
        assert result["gross_profit"].iloc[0] == pytest.approx(30000)
        assert result["operating_income"].iloc[0] == pytest.approx(20000)
        assert result["net_income"].iloc[0] == pytest.approx(15000)
        assert result["eps"].iloc[0] == pytest.approx(3.5)

    def test_pivot_balance_sheet(self):
        eav_data = pd.DataFrame(
            {
                "date": ["2024-03-31"] * 3,
                "stock_id": ["2330"] * 3,
                "type": ["TotalAssets", "TotalLiabilities", "Equity"],
                "value": [500000, 200000, 300000],
            }
        )
        result = self.fetcher._pivot_eav(eav_data, FinMindFetcher._BALANCE_TYPES)

        assert len(result) == 1
        assert result["total_assets"].iloc[0] == pytest.approx(500000)
        assert result["total_liabilities"].iloc[0] == pytest.approx(200000)
        assert result["equity"].iloc[0] == pytest.approx(300000)

    def test_pivot_cashflow(self):
        eav_data = pd.DataFrame(
            {
                "date": ["2024-03-31"] * 3,
                "stock_id": ["2330"] * 3,
                "type": [
                    "CashFlowsFromOperatingActivities",
                    "CashProvidedByInvestingActivities",
                    "CashFlowsFromFinancingActivities",
                ],
                "value": [50000, -20000, -10000],
            }
        )
        result = self.fetcher._pivot_eav(eav_data, FinMindFetcher._CASHFLOW_TYPES)

        assert len(result) == 1
        assert result["operating_cf"].iloc[0] == pytest.approx(50000)
        assert result["investing_cf"].iloc[0] == pytest.approx(-20000)
        assert result["financing_cf"].iloc[0] == pytest.approx(-10000)

    def test_pivot_empty_dataframe(self):
        result = self.fetcher._pivot_eav(pd.DataFrame(), FinMindFetcher._INCOME_TYPES)
        assert result.empty

    def test_pivot_filters_unneeded_types(self):
        """不在 type_map 中的 type 應被過濾。"""
        eav_data = pd.DataFrame(
            {
                "date": ["2024-03-31"] * 3,
                "stock_id": ["2330"] * 3,
                "type": ["Revenue", "SomeOtherType", "EPS"],
                "value": [100000, 999, 3.5],
            }
        )
        result = self.fetcher._pivot_eav(eav_data, FinMindFetcher._INCOME_TYPES)

        assert len(result) == 1
        assert "revenue" in result.columns
        assert "eps" in result.columns

    def test_pivot_multiple_quarters(self):
        """多季度資料應產生多列。"""
        eav_data = pd.DataFrame(
            {
                "date": ["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"],
                "stock_id": ["2330"] * 4,
                "type": ["Revenue", "Revenue", "EPS", "EPS"],
                "value": [100000, 110000, 3.5, 4.0],
            }
        )
        result = self.fetcher._pivot_eav(eav_data, FinMindFetcher._INCOME_TYPES)
        assert len(result) == 2


# ------------------------------------------------------------------ #
#  fetch_financial_summary 整合測試（mock API）
# ------------------------------------------------------------------ #


class TestFetchFinancialSummary:
    """fetch_financial_summary 合併三表 + 計算比率。"""

    def _make_eav(self, types_values: dict, dt: str = "2024-03-31", sid: str = "2330") -> pd.DataFrame:
        rows = []
        for t, v in types_values.items():
            rows.append({"date": dt, "stock_id": sid, "type": t, "value": v})
        return pd.DataFrame(rows)

    @patch.object(FinMindFetcher, "_request")
    def test_summary_merges_three_tables(self, mock_request):
        """驗證三表合併與衍生比率計算。"""
        income_eav = self._make_eav(
            {"Revenue": 10000, "GrossProfit": 3000, "OperatingIncome": 2000, "IncomeAfterTaxes": 1500, "EPS": 3.5}
        )
        balance_eav = self._make_eav({"TotalAssets": 50000, "TotalLiabilities": 20000, "Equity": 30000})
        cashflow_eav = self._make_eav(
            {
                "CashFlowsFromOperatingActivities": 5000,
                "CashProvidedByInvestingActivities": -2000,
                "CashFlowsFromFinancingActivities": -1000,
            }
        )

        mock_request.side_effect = [income_eav, balance_eav, cashflow_eav]

        fetcher = FinMindFetcher.__new__(FinMindFetcher)
        fetcher.api_url = "http://test"
        fetcher.api_token = "test"
        fetcher._session = MagicMock()

        result = fetcher.fetch_financial_summary("2330", "2024-01-01", "2024-12-31")

        assert len(result) == 1
        row = result.iloc[0]
        assert row["stock_id"] == "2330"
        assert row["year"] == 2024
        assert row["quarter"] == 1
        assert row["revenue"] == pytest.approx(10000)
        assert row["eps"] == pytest.approx(3.5)
        assert row["total_assets"] == pytest.approx(50000)
        assert row["operating_cf"] == pytest.approx(5000)
        assert row["free_cf"] == pytest.approx(3000)
        assert row["gross_margin"] == pytest.approx(30.0)
        assert row["roe"] == pytest.approx(5.0)
        assert row["debt_ratio"] == pytest.approx(40.0)

    @patch.object(FinMindFetcher, "_request")
    def test_summary_empty_income_returns_empty(self, mock_request):
        mock_request.return_value = pd.DataFrame()

        fetcher = FinMindFetcher.__new__(FinMindFetcher)
        fetcher.api_url = "http://test"
        fetcher.api_token = "test"
        fetcher._session = MagicMock()

        result = fetcher.fetch_financial_summary("2330", "2024-01-01")
        assert result.empty


# ------------------------------------------------------------------ #
#  Pipeline upsert + DB 整合測試
# ------------------------------------------------------------------ #


@pytest.fixture(autouse=True)
def _patch_pipeline_session(db_session, monkeypatch):
    """確保 pipeline 模組使用測試 session（與 test_pipeline.py 相同模式）。"""
    import src.data.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "get_session", lambda: db_session)
    monkeypatch.setattr(pipeline_mod, "init_db", lambda: None)


class TestFinancialPipeline:
    """pipeline _upsert_financial 寫入 DB。"""

    def test_upsert_financial(self, db_session):
        """寫入財報資料到 in-memory DB，驗證讀取。"""
        from src.data.pipeline import _upsert_financial

        df = pd.DataFrame(
            [
                {
                    "stock_id": "2330",
                    "date": date(2024, 3, 31),
                    "year": 2024,
                    "quarter": 1,
                    "revenue": 100000,
                    "gross_profit": 30000,
                    "operating_income": 20000,
                    "net_income": 15000,
                    "eps": 3.5,
                    "total_assets": 50000,
                    "total_liabilities": 20000,
                    "equity": 30000,
                    "operating_cf": 5000,
                    "investing_cf": -2000,
                    "financing_cf": -1000,
                    "free_cf": 3000,
                    "gross_margin": 30.0,
                    "operating_margin": 20.0,
                    "net_margin": 15.0,
                    "roe": 5.0,
                    "roa": 3.0,
                    "debt_ratio": 40.0,
                }
            ]
        )

        count = _upsert_financial(df)
        assert count == 1

        # 驗證 DB 中的資料
        rows = db_session.query(FinancialStatement).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.stock_id == "2330"
        assert row.date == date(2024, 3, 31)
        assert row.quarter == 1
        assert row.eps == pytest.approx(3.5)
        assert row.gross_margin == pytest.approx(30.0)
        assert row.roe == pytest.approx(5.0)

    def test_upsert_conflict_do_nothing(self, db_session):
        """重複寫入同一筆（stock_id + date），不應報錯。"""
        from src.data.pipeline import _upsert_financial

        df = pd.DataFrame(
            [
                {
                    "stock_id": "2330",
                    "date": date(2024, 6, 30),
                    "year": 2024,
                    "quarter": 2,
                    "revenue": 120000,
                    "eps": 4.0,
                }
            ]
        )

        count1 = _upsert_financial(df)
        count2 = _upsert_financial(df)
        assert count1 == 1
        assert count2 == 1  # 衝突時略過，仍回傳 record 數

        rows = db_session.query(FinancialStatement).filter_by(stock_id="2330", quarter=2).all()
        assert len(rows) == 1

    def test_upsert_empty(self, db_session):
        """空 DataFrame 不寫入。"""
        from src.data.pipeline import _upsert_financial

        count = _upsert_financial(pd.DataFrame())
        assert count == 0

    def test_upsert_multiple_quarters(self, db_session):
        """多季度資料批次寫入。"""
        from src.data.pipeline import _upsert_financial

        df = pd.DataFrame(
            [
                {"stock_id": "2317", "date": date(2024, 3, 31), "year": 2024, "quarter": 1, "eps": 2.0},
                {"stock_id": "2317", "date": date(2024, 6, 30), "year": 2024, "quarter": 2, "eps": 2.5},
                {"stock_id": "2317", "date": date(2024, 9, 30), "year": 2024, "quarter": 3, "eps": 3.0},
                {"stock_id": "2317", "date": date(2024, 12, 31), "year": 2024, "quarter": 4, "eps": 3.5},
            ]
        )

        count = _upsert_financial(df)
        assert count == 4

        rows = db_session.query(FinancialStatement).filter_by(stock_id="2317").all()
        assert len(rows) == 4


# ------------------------------------------------------------------ #
#  FinancialStatement ORM 測試
# ------------------------------------------------------------------ #


class TestFinancialStatementORM:
    """ORM 模型基本測試。"""

    def test_repr(self, db_session):
        fs = FinancialStatement(
            stock_id="2330",
            date=date(2024, 3, 31),
            year=2024,
            quarter=1,
            eps=3.5,
        )
        db_session.add(fs)
        db_session.flush()

        assert "2330" in repr(fs)
        assert "Q1" in repr(fs)
        assert "3.5" in repr(fs)

    def test_unique_constraint(self, db_session):
        """同一 stock_id + date 不能重複。"""
        from sqlalchemy.exc import IntegrityError

        fs1 = FinancialStatement(stock_id="2330", date=date(2024, 3, 31), year=2024, quarter=1, eps=3.5)
        fs2 = FinancialStatement(stock_id="2330", date=date(2024, 3, 31), year=2024, quarter=1, eps=4.0)

        db_session.add(fs1)
        db_session.flush()

        db_session.add(fs2)
        with pytest.raises(IntegrityError):
            db_session.flush()
