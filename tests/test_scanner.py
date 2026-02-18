"""測試 src/discovery/scanner.py — MarketScanner 純計算方法。"""

from datetime import date

import pandas as pd
import pytest

from src.discovery.scanner import MarketScanner


@pytest.fixture()
def scanner():
    return MarketScanner(
        min_price=10,
        max_price=2000,
        min_volume=100_000,
        top_n_candidates=10,
        top_n_results=5,
    )


def _make_price_df(n_stocks: int = 20) -> pd.DataFrame:
    """建立模擬市場日 K 資料（2 天）。"""
    rows = []
    d1, d2 = date(2025, 1, 2), date(2025, 1, 3)
    for i in range(n_stocks):
        sid = f"{1000 + i}"
        close_d1 = 50 + i * 10
        close_d2 = close_d1 + (i - n_stocks // 2)
        for d, c in [(d1, close_d1), (d2, close_d2)]:
            rows.append(
                {
                    "stock_id": sid,
                    "date": d,
                    "open": c - 1,
                    "high": c + 2,
                    "low": c - 2,
                    "close": c,
                    "volume": 200_000 + i * 50_000,
                }
            )
    return pd.DataFrame(rows)


def _make_inst_df(stock_ids: list[str], target_date: date) -> pd.DataFrame:
    rows = []
    for sid in stock_ids:
        rows.append(
            {
                "stock_id": sid,
                "date": target_date,
                "name": "Foreign_Investor",
                "net": int(sid) % 3 * 1000 - 500,
            }
        )
        rows.append(
            {
                "stock_id": sid,
                "date": target_date,
                "name": "Investment_Trust",
                "net": int(sid) % 5 * 200 - 200,
            }
        )
    return pd.DataFrame(rows)


# ─── _coarse_filter ───────────────────────────────────────


class TestCoarseFilter:
    def test_filters_by_price_and_volume(self, scanner):
        df_price = _make_price_df(20)
        df_inst = pd.DataFrame()
        result = scanner._coarse_filter(df_price, df_inst)
        # All stocks should pass min_price=10, only ones with volume >= 100k
        assert len(result) <= scanner.top_n_candidates
        if not result.empty:
            assert (result["close"] >= scanner.min_price).all()
            assert (result["volume"] >= scanner.min_volume).all()

    def test_filters_out_index_stocks(self, scanner):
        df_price = _make_price_df(5)
        # Add an index-like stock with letters
        idx_row = pd.DataFrame(
            [
                {
                    "stock_id": "TAIEX",
                    "date": date(2025, 1, 3),
                    "open": 18000,
                    "high": 18100,
                    "low": 17900,
                    "close": 18000,
                    "volume": 10_000_000,
                }
            ]
        )
        df_price = pd.concat([df_price, idx_row], ignore_index=True)
        result = scanner._coarse_filter(df_price, pd.DataFrame())
        stock_ids = result["stock_id"].tolist() if not result.empty else []
        assert "TAIEX" not in stock_ids

    def test_empty_input(self, scanner):
        empty_df = pd.DataFrame(columns=["stock_id", "date", "open", "high", "low", "close", "volume"])
        result = scanner._coarse_filter(empty_df, pd.DataFrame())
        assert result.empty

    def test_with_institutional_data(self, scanner):
        df_price = _make_price_df(20)
        sids = df_price["stock_id"].unique().tolist()
        df_inst = _make_inst_df(sids, date(2025, 1, 3))
        result = scanner._coarse_filter(df_price, df_inst)
        assert "inst_rank" in result.columns
        assert "coarse_score" in result.columns


# ─── _compute_technical_scores ────────────────────────────


class TestComputeTechnicalScores:
    def test_scores_in_valid_range(self, scanner):
        df_price = _make_price_df(5)
        sids = df_price["stock_id"].unique().tolist()
        result = scanner._compute_technical_scores(sids, df_price)
        assert "technical_score" in result.columns
        assert (result["technical_score"] >= 0).all()
        assert (result["technical_score"] <= 1.0).all()

    def test_insufficient_data_gets_default(self, scanner):
        df_price = pd.DataFrame(
            [
                {
                    "stock_id": "9999",
                    "date": date(2025, 1, 3),
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "volume": 500_000,
                }
            ]
        )
        result = scanner._compute_technical_scores(["9999"], df_price)
        assert result.iloc[0]["technical_score"] == pytest.approx(0.5)


# ─── _compute_chip_scores ────────────────────────────────


class TestComputeChipScores:
    def test_empty_inst_returns_default(self, scanner):
        result = scanner._compute_chip_scores(["1000", "1001"], pd.DataFrame())
        assert len(result) == 2
        assert (result["chip_score"] == 0.5).all()

    def test_scores_in_valid_range(self, scanner):
        sids = ["1000", "1001"]
        df_inst = _make_inst_df(sids, date(2025, 1, 3))
        result = scanner._compute_chip_scores(sids, df_inst)
        assert (result["chip_score"] >= 0).all()
        assert (result["chip_score"] <= 1.0).all()


# ─── _compute_sector_summary ─────────────────────────────


class TestComputeSectorSummary:
    def test_sector_summary(self, scanner):
        rankings = pd.DataFrame(
            {
                "stock_id": ["1000", "1001", "1002"],
                "industry_category": ["半導體", "半導體", "金融"],
                "composite_score": [0.8, 0.7, 0.6],
            }
        )
        result = scanner._compute_sector_summary(rankings)
        assert "industry" in result.columns
        assert "count" in result.columns
        assert "avg_score" in result.columns
        assert len(result) == 2

    def test_empty_rankings(self, scanner):
        result = scanner._compute_sector_summary(pd.DataFrame())
        assert result.empty


# ─── _compute_fundamental_scores ─────────────────────────


class TestComputeFundamentalScores:
    def test_with_revenue_data(self, scanner):
        """排名百分位：YoY 較高者分數較高"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001", "1002"],
                "yoy_growth": [10.0, 30.0, -5.0],
                "mom_growth": [5.0, -2.0, 3.0],
            }
        )
        result = scanner._compute_fundamental_scores(["1000", "1001", "1002"], df_revenue)
        assert len(result) == 3
        scores = result.set_index("stock_id")["fundamental_score"]
        # 1001 有最高 YoY → 分數最高
        assert scores["1001"] > scores["1002"]
        # 所有分數都在 0~1 範圍
        assert (result["fundamental_score"] >= 0).all()
        assert (result["fundamental_score"] <= 1).all()

    def test_no_revenue_returns_default(self, scanner):
        """空 DF → 全部 0.5"""
        result = scanner._compute_fundamental_scores(
            ["1000", "1001"], pd.DataFrame(columns=["stock_id", "yoy_growth", "mom_growth"])
        )
        assert len(result) == 2
        assert (result["fundamental_score"] == 0.5).all()

    def test_higher_yoy_ranks_higher(self, scanner):
        """YoY 越高排名越前，分數越高"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [100.0, 5.0],
                "mom_growth": [10.0, 10.0],
            }
        )
        result = scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]

    def test_scores_spread_across_range(self, scanner):
        """多支股票時分數應分散，不會全部相同"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001", "1002", "1003"],
                "yoy_growth": [-20.0, 0.0, 15.0, 50.0],
                "mom_growth": [-5.0, 3.0, -1.0, 10.0],
            }
        )
        result = scanner._compute_fundamental_scores(["1000", "1001", "1002", "1003"], df_revenue)
        scores = result["fundamental_score"]
        # 至少有 3 個不同的分數值（分散性）
        assert scores.nunique() >= 3
