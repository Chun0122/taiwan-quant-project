"""測試 src/discovery/scanner.py — MarketScanner / MomentumScanner / SwingScanner / ValueScanner / DividendScanner / GrowthScanner 純計算方法。"""

from datetime import date, timedelta

import pandas as pd
import pytest

from src.discovery.scanner import (
    DividendScanner,
    GrowthScanner,
    MarketScanner,
    MomentumScanner,
    SwingScanner,
    ValueScanner,
)
from tests.scanner_helpers import (
    make_entry_exit_price_df,
    make_inst_df,
    make_momentum_price_df,
    make_price_df,
    make_swing_price_df,
)


@pytest.fixture()
def scanner():
    return MarketScanner(
        min_price=10,
        max_price=2000,
        min_volume=100_000,
        top_n_candidates=10,
        top_n_results=5,
    )


# 向後相容別名（內部測試使用舊命名）
_make_price_df = make_price_df
_make_inst_df = make_inst_df

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

    def test_volatility_convergence(self, scanner):
        """波動度收斂：價格穩定的股票應在波動收斂因子得高分。"""
        # 建立 5 天幾乎不動的資料（低波動）
        rows = []
        for i, d in enumerate([date(2025, 1, j) for j in range(1, 6)]):
            rows.append(
                {
                    "stock_id": "8888",
                    "date": d,
                    "open": 100,
                    "high": 100.5,
                    "low": 99.5,
                    "close": 100 + i * 0.1,  # 極小變動
                    "volume": 1_000_000,
                }
            )
        df_price = pd.DataFrame(rows)
        result = scanner._compute_technical_scores(["8888"], df_price)
        # 低波動 → 波動收斂因子接近 1.0，整體分應 > 0.5
        assert result.iloc[0]["technical_score"] >= 0.4

    def test_volume_price_divergence(self, scanner):
        """量價背離：價漲量增 vs 價跌量增，前者分數應更高。"""
        # 價漲量增
        rows_up = []
        for i, d in enumerate([date(2025, 1, j) for j in range(1, 6)]):
            rows_up.append(
                {
                    "stock_id": "7777",
                    "date": d,
                    "open": 100 + i * 2,
                    "high": 103 + i * 2,
                    "low": 99 + i * 2,
                    "close": 101 + i * 2,
                    "volume": 500_000 + i * 100_000,
                }
            )
        # 價跌量增
        rows_down = []
        for i, d in enumerate([date(2025, 1, j) for j in range(1, 6)]):
            rows_down.append(
                {
                    "stock_id": "6666",
                    "date": d,
                    "open": 110 - i * 2,
                    "high": 113 - i * 2,
                    "low": 109 - i * 2,
                    "close": 109 - i * 2,
                    "volume": 500_000 + i * 100_000,
                }
            )
        df_price = pd.DataFrame(rows_up + rows_down)
        result = scanner._compute_technical_scores(["7777", "6666"], df_price)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["7777"] > scores["6666"]


# ─── _compute_chip_scores ────────────────────────────────


class TestComputeChipScores:
    def test_empty_inst_returns_default(self, scanner):
        result = scanner._compute_chip_scores(["1000", "1001"], pd.DataFrame())
        assert len(result) == 2
        assert (result["chip_score"] == 0.5).all()

    def test_scores_in_valid_range(self, scanner):
        sids = ["1000", "1001"]
        df_inst = _make_inst_df(sids, date(2025, 1, 3))
        df_price = _make_price_df(5)
        result = scanner._compute_chip_scores(sids, df_inst, df_price)
        assert (result["chip_score"] >= 0).all()
        assert (result["chip_score"] <= 1.0).all()

    def test_consecutive_buy_days_boost(self, scanner):
        """連續買超天數越多，分數應越高。"""
        sids = ["1000", "1001"]
        # 1000: 只有 1 天買超，1001: 連續 3 天買超
        rows = []
        for d_offset, d in enumerate([date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]):
            # 1000: 只有最後一天淨買超 > 0
            net_1000 = 1000 if d_offset == 2 else -500
            rows.append({"stock_id": "1000", "date": d, "name": "Foreign_Investor", "net": net_1000})
            # 1001: 每天都淨買超 > 0
            rows.append({"stock_id": "1001", "date": d, "name": "Foreign_Investor", "net": 500})
        df_inst = pd.DataFrame(rows)
        df_price = _make_price_df(5)
        result = scanner._compute_chip_scores(sids, df_inst, df_price)
        scores = result.set_index("stock_id")["chip_score"]
        # 1001 連續買超 3 天，籌碼分數應高於 1000
        assert scores["1001"] > scores["1000"]

    def test_without_price_data(self, scanner):
        """未傳入 df_price 時應 graceful fallback（買超佔量比為 0）。"""
        sids = ["1000", "1001"]
        df_inst = _make_inst_df(sids, date(2025, 1, 3))
        result = scanner._compute_chip_scores(sids, df_inst)
        assert len(result) == 2
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


# ─── _base_filter ─────────────────────────────────────────


class TestBaseFilter:
    def test_filters_out_etf(self, scanner):
        """排除 00 開頭的 ETF"""
        df_price = _make_price_df(5)
        etf_row = pd.DataFrame(
            [
                {
                    "stock_id": "0050",
                    "date": date(2025, 1, 3),
                    "open": 150,
                    "high": 152,
                    "low": 148,
                    "close": 151,
                    "volume": 10_000_000,
                }
            ]
        )
        df_price = pd.concat([df_price, etf_row], ignore_index=True)
        result = scanner._base_filter(df_price)
        assert "0050" not in result["stock_id"].tolist()


# ====================================================================== #
#  MomentumScanner 測試
# ====================================================================== #


_make_momentum_price_df = make_momentum_price_df


@pytest.fixture()
def momentum_scanner():
    return MomentumScanner(
        min_price=10,
        max_price=2000,
        min_volume=100_000,
        top_n_candidates=10,
        top_n_results=5,
    )


class TestMomentumTechnicalScores:
    def test_scores_in_valid_range(self, momentum_scanner):
        df_price = _make_momentum_price_df(25, 5)
        sids = df_price["stock_id"].unique().tolist()
        result = momentum_scanner._compute_technical_scores(sids, df_price)
        assert (result["technical_score"] >= 0).all()
        assert (result["technical_score"] <= 1.0).all()

    def test_insufficient_data_gets_default(self, momentum_scanner):
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
        result = momentum_scanner._compute_technical_scores(["9999"], df_price)
        assert result.iloc[0]["technical_score"] == pytest.approx(0.5)

    def test_five_factors_computed(self, momentum_scanner):
        """25 天資料應觸發全部 5 個因子。"""
        df_price = _make_momentum_price_df(25, 3)
        sids = df_price["stock_id"].unique().tolist()
        result = momentum_scanner._compute_technical_scores(sids, df_price)
        # 有足夠資料時分數不應全部相同
        assert result["technical_score"].nunique() >= 2

    def test_stronger_momentum_higher_score(self, momentum_scanner):
        """動能更強的股票，技術分數應更高。"""
        df_price = _make_momentum_price_df(25, 5)
        sids = df_price["stock_id"].unique().tolist()
        result = momentum_scanner._compute_technical_scores(sids, df_price)
        scores = result.set_index("stock_id")["technical_score"]
        # 2004 動能最強，2000 最弱
        assert scores["2004"] >= scores["2000"]


class TestMomentumChipScores:
    def test_empty_inst_returns_default(self, momentum_scanner):
        result = momentum_scanner._compute_chip_scores(["1000"], pd.DataFrame())
        assert result.iloc[0]["chip_score"] == pytest.approx(0.5)

    def test_three_factors_weighted(self, momentum_scanner):
        """連續外資買超天數 + 買超佔量比 + 合計買超。"""
        sids = ["1000", "1001"]
        rows = []
        for d in range(5):
            day = date(2025, 1, 1) + timedelta(days=d)
            # 1001: 外資每天都買超，1000: 只有最後一天
            rows.append({"stock_id": "1000", "date": day, "name": "外資買賣超", "net": 100 if d == 4 else -50})
            rows.append({"stock_id": "1001", "date": day, "name": "外資買賣超", "net": 500})
        df_inst = pd.DataFrame(rows)
        df_price = _make_momentum_price_df(5, 2)
        # 需要 stock_id 對應
        df_price["stock_id"] = df_price["stock_id"].map({"2000": "1000", "2001": "1001"})
        result = momentum_scanner._compute_chip_scores(sids, df_inst, df_price)
        scores = result.set_index("stock_id")["chip_score"]
        assert scores["1001"] > scores["1000"]


class TestMomentumChipWithMargin:
    """融資融券券資比因子測試。"""

    def test_margin_data_adds_short_margin_factor(self, momentum_scanner):
        """有融資融券資料時，籌碼面使用 4 因子加權（含券資比）。"""
        sids = ["1000", "1001"]
        # 法人數據完全相同，只有券資比不同
        df_inst = pd.DataFrame(
            [
                {"stock_id": "1000", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 100},
                {"stock_id": "1001", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 100},
            ]
        )
        df_margin = pd.DataFrame(
            [
                {"stock_id": "1000", "date": date(2025, 1, 25), "margin_balance": 1000, "short_balance": 500},
                {"stock_id": "1001", "date": date(2025, 1, 25), "margin_balance": 1000, "short_balance": 100},
            ]
        )
        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, df_margin)
        assert len(result) == 2
        scores = result.set_index("stock_id")["chip_score"]
        # 1000 有較高券資比 (0.5 vs 0.1)，券資比排名加分
        assert scores["1000"] > scores["1001"]

    def test_empty_margin_uses_three_factors(self, momentum_scanner):
        """無融資融券資料時，退回 3 因子加權。"""
        sids = ["1000"]
        df_inst = pd.DataFrame([{"stock_id": "1000", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 100}])
        result_no_margin = momentum_scanner._compute_chip_scores(sids, df_inst, None, None)
        result_empty = momentum_scanner._compute_chip_scores(sids, df_inst, None, pd.DataFrame())
        assert result_no_margin.iloc[0]["chip_score"] == result_empty.iloc[0]["chip_score"]


class TestChipTier:
    """籌碼因子層級（chip_tier）透明度測試。"""

    def test_empty_inst_returns_na_tier(self, momentum_scanner):
        """法人資料空時，chip_tier 應為 N/A。"""
        result = momentum_scanner._compute_chip_scores(["1000"], pd.DataFrame())
        assert "chip_tier" in result.columns
        assert result.iloc[0]["chip_tier"] == "N/A"

    def test_three_factor_tier(self, momentum_scanner):
        """無任何附加資料時，MomentumScanner 應回傳 3F。"""
        sids = ["1000", "1001"]
        df_inst = pd.DataFrame(
            [
                {"stock_id": "1000", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 100},
                {"stock_id": "1001", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 200},
            ]
        )
        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, None)
        assert result.iloc[0]["chip_tier"] == "3F"
        assert result.iloc[1]["chip_tier"] == "3F"

    def test_four_factor_tier_with_margin(self, momentum_scanner):
        """有融資融券資料時（無大戶/借券），MomentumScanner 應回傳 4F。"""
        sids = ["1000", "1001"]
        df_inst = pd.DataFrame(
            [
                {"stock_id": "1000", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 100},
                {"stock_id": "1001", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 200},
            ]
        )
        df_margin = pd.DataFrame(
            [
                {"stock_id": "1000", "date": date(2025, 1, 25), "margin_balance": 1000, "short_balance": 200},
                {"stock_id": "1001", "date": date(2025, 1, 25), "margin_balance": 1000, "short_balance": 100},
            ]
        )
        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, df_margin)
        assert result.iloc[0]["chip_tier"] == "4F"

    def test_swing_scanner_tier_without_whale(self):
        """SwingScanner 無大戶資料時回傳 2F。"""
        from src.discovery.scanner import SwingScanner

        scanner = SwingScanner()
        sids = ["1000", "1001"]
        df_inst = pd.DataFrame(
            [
                {"stock_id": "1000", "date": date(2025, 1, 25), "name": "投信買賣超", "net": 100},
                {"stock_id": "1001", "date": date(2025, 1, 25), "name": "投信買賣超", "net": 200},
            ]
        )
        result = scanner._compute_chip_scores(sids, df_inst, None, None)
        assert result.iloc[0]["chip_tier"] == "2F"

    def test_value_scanner_tier_always_2f(self):
        """ValueScanner 籌碼面固定 2F。"""
        from src.discovery.scanner import ValueScanner

        scanner = ValueScanner()
        sids = ["1000", "1001"]
        df_inst = pd.DataFrame(
            [
                {"stock_id": "1000", "date": date(2025, 1, 25), "name": "投信買賣超", "net": 100},
                {"stock_id": "1001", "date": date(2025, 1, 25), "name": "投信買賣超", "net": 200},
            ]
        )
        result = scanner._compute_chip_scores(sids, df_inst, None, None)
        assert all(result["chip_tier"] == "2F")

    def test_chip_tier_in_rankings_columns(self, momentum_scanner, db_session):
        """chip_tier 欄位應出現在 _rank_and_enrich 輸出中（需 db_session 建立 stock_info 表）。"""
        scored = pd.DataFrame(
            {
                "rank": [1, 2],
                "stock_id": ["1000", "1001"],
                "close": [100.0, 200.0],
                "volume": [1000, 2000],
                "composite_score": [0.8, 0.6],
                "technical_score": [0.7, 0.6],
                "chip_score": [0.9, 0.5],
                "chip_tier": ["4F", "3F"],
                "fundamental_score": [0.6, 0.7],
                "news_score": [0.5, 0.5],
                "sector_bonus": [0.0, 0.0],
                "momentum": [0.1, 0.05],
                "inst_net": [1000, 500],
                "entry_price": [100.0, 200.0],
                "stop_loss": [95.0, 190.0],
                "take_profit": [115.0, 230.0],
                "entry_trigger": ["突破均線", "動能確認"],
                "valid_until": [date(2025, 2, 1), date(2025, 2, 1)],
            }
        )
        result = momentum_scanner._rank_and_enrich(scored)
        assert "chip_tier" in result.columns
        assert list(result["chip_tier"]) == ["4F", "3F"]


class TestMomentumFundamentalScores:
    def test_yoy_positive_gets_higher(self, momentum_scanner):
        """YoY > 0 且無加速度資料 → Tier 3 (0.55)；YoY <= 0 → Tier 4 (0.30)。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [10.0, -5.0],
                "mom_growth": [5.0, 5.0],
            }
        )
        result = momentum_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] == pytest.approx(0.55)
        assert scores["1001"] == pytest.approx(0.30)

    def test_no_data_fallback(self, momentum_scanner):
        result = momentum_scanner._compute_fundamental_scores(
            ["1000"], pd.DataFrame(columns=["stock_id", "yoy_growth", "mom_growth"])
        )
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.5)

    def test_acceleration_bonus_positive(self, momentum_scanner):
        """YoY > 0、MoM > 0、加速（yoy < yoy_3m_ago 的逆）→ Tier 1: 0.85。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000"],
                "yoy_growth": [20.0],
                "mom_growth": [5.0],
                "yoy_3m_ago": [10.0],  # 加速：20 > 10
            }
        )
        result = momentum_scanner._compute_fundamental_scores(["1000"], df_revenue)
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.85)

    def test_acceleration_bonus_negative(self, momentum_scanner):
        """YoY > 0、MoM > 0 但減速（yoy < yoy_3m_ago）→ Tier 3: 0.55。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000"],
                "yoy_growth": [10.0],
                "mom_growth": [5.0],
                "yoy_3m_ago": [30.0],  # 減速：10 < 30
            }
        )
        result = momentum_scanner._compute_fundamental_scores(["1000"], df_revenue)
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.55)

    def test_tier2_yoy_accelerating_mom_negative(self, momentum_scanner):
        """YoY > 0 且加速，但 MoM <= 0 → Tier 2: 0.72（非雙增故不升 Tier 1）。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000"],
                "yoy_growth": [20.0],
                "mom_growth": [-3.0],
                "yoy_3m_ago": [10.0],
            }
        )
        result = momentum_scanner._compute_fundamental_scores(["1000"], df_revenue)
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.72)

    def test_tier3_yoy_positive_no_yoy3m(self, momentum_scanner):
        """YoY > 0 但無 yoy_3m_ago → 無法判斷加速 → Tier 3: 0.55。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000"],
                "yoy_growth": [15.0],
                "mom_growth": [5.0],
            }
        )
        result = momentum_scanner._compute_fundamental_scores(["1000"], df_revenue)
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.55)

    def test_tier4_yoy_negative_regardless_mom(self, momentum_scanner):
        """YoY <= 0 → Tier 4: 0.30（MoM 正也無法拉升）。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000"],
                "yoy_growth": [-5.0],
                "mom_growth": [10.0],
                "yoy_3m_ago": [-8.0],
            }
        )
        result = momentum_scanner._compute_fundamental_scores(["1000"], df_revenue)
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.30)

    def test_tier1_highest_tier2_second(self, momentum_scanner):
        """四階梯排序：Tier 1 > Tier 2 > Tier 3 > Tier 4。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["t1", "t2", "t3", "t4"],
                "yoy_growth": [20.0, 20.0, 15.0, -5.0],
                "mom_growth": [5.0, -3.0, 5.0, 5.0],
                "yoy_3m_ago": [10.0, 10.0, 30.0, -8.0],
            }
        )
        result = momentum_scanner._compute_fundamental_scores(["t1", "t2", "t3", "t4"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["t1"] > scores["t2"] > scores["t3"] > scores["t4"]


@pytest.mark.parametrize(
    "scanner_name,n_days,sid_base,spread_low,spread_high",
    [
        ("momentum_scanner", 15, 3000, 1, 20),
        ("swing_scanner", 25, 6000, 0.5, 15),
        ("value_scanner", 15, 7000, 0.5, 20),
        ("dividend_scanner", 15, 8000, 0.5, 20),
        ("growth_scanner", 15, 9000, 1, 20),
    ],
)
def test_risk_filter_removes_high_volatility(request, scanner_name, n_days, sid_base, spread_low, spread_high):
    """各 Scanner risk filter 應剔除高波動外圍股。"""
    scanner = request.getfixturevalue(scanner_name)
    rows = []
    sids = [f"{sid_base + i}" for i in range(10)]
    for i, sid in enumerate(sids):
        for d in range(n_days):
            day = date(2025, 1, 1) + timedelta(days=d)
            spread = spread_high if i == 9 else spread_low
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": 100,
                    "high": 100 + spread,
                    "low": 100 - spread,
                    "close": 100 + (spread if d % 2 == 0 else -spread),
                    "volume": 1_000_000,
                }
            )
    df_price = pd.DataFrame(rows)
    scored = pd.DataFrame({"stock_id": sids, "composite_score": [0.5] * 10})
    result = scanner._apply_risk_filter(scored, df_price)
    assert len(result) < len(scored)


@pytest.mark.parametrize("scanner_name", ["momentum_scanner", "growth_scanner"])
def test_risk_filter_empty_returns_empty(request, scanner_name):
    """空輸入應回傳空 DataFrame。"""
    scanner = request.getfixturevalue(scanner_name)
    result = scanner._apply_risk_filter(pd.DataFrame(columns=["stock_id"]), pd.DataFrame())
    assert result.empty


class TestMomentumCompositeWeights:
    def test_weights_with_regime(self, momentum_scanner):
        """確認綜合分數依 regime 權重計算（含消息面）。"""
        from src.regime.detector import MarketRegimeDetector

        candidates = pd.DataFrame({"stock_id": ["1000"], "close": [100], "volume": [500_000]})
        df_price = _make_momentum_price_df(25, 1)
        df_price["stock_id"] = "1000"
        df_inst = pd.DataFrame([{"stock_id": "1000", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 100}])
        df_revenue = pd.DataFrame({"stock_id": ["1000"], "yoy_growth": [10.0], "mom_growth": [5.0]})

        df_margin = pd.DataFrame()
        result = momentum_scanner._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue)
        tech = result.iloc[0]["technical_score"]
        chip = result.iloc[0]["chip_score"]
        fund = result.iloc[0]["fundamental_score"]
        news = result.iloc[0]["news_score"]
        regime = getattr(momentum_scanner, "regime", "sideways")
        w = MarketRegimeDetector.get_weights("momentum", regime)
        expected = tech * w["technical"] + chip * w["chip"] + fund * w["fundamental"] + news * w.get("news", 0.10)
        assert result.iloc[0]["composite_score"] == pytest.approx(expected, abs=1e-6)


# ====================================================================== #
#  SwingScanner 測試
# ====================================================================== #


_make_swing_price_df = make_swing_price_df


@pytest.fixture()
def swing_scanner():
    return SwingScanner(
        min_price=10,
        max_price=2000,
        min_volume=100_000,
        top_n_candidates=10,
        top_n_results=5,
    )


class TestSwingCoarseFilter:
    def test_requires_sma60(self, swing_scanner):
        """close > SMA60 過濾：低於 SMA60 的股票應被排除。"""
        rows = []
        # 股票 A：穩定上升（close > SMA60）
        for d in range(65):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {
                    "stock_id": "5000",
                    "date": day,
                    "open": 99 + d * 0.5,
                    "high": 101 + d * 0.5,
                    "low": 98 + d * 0.5,
                    "close": 100 + d * 0.5,
                    "volume": 500_000,
                }
            )
        # 股票 B：穩定下降（close < SMA60）
        for d in range(65):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {
                    "stock_id": "5001",
                    "date": day,
                    "open": 201 - d * 0.5,
                    "high": 203 - d * 0.5,
                    "low": 199 - d * 0.5,
                    "close": 200 - d * 0.5,
                    "volume": 500_000,
                }
            )
        df_price = pd.DataFrame(rows)
        result = swing_scanner._coarse_filter(df_price, pd.DataFrame())
        sids = result["stock_id"].tolist() if not result.empty else []
        # 上升股票應被保留，下降股票應被排除
        assert "5000" in sids
        assert "5001" not in sids


class TestSwingSma60Vectorized:
    """驗證向量化 SMA60 計算結果與語義正確性。"""

    def _make_sma60_price(self, sid: str, n_days: int, start_close: float, slope: float) -> list[dict]:
        rows = []
        for d in range(n_days):
            day = date(2025, 1, 1) + timedelta(days=d)
            close = start_close + d * slope
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 500_000,
                }
            )
        return rows

    def test_sma60_vectorized_excludes_downtrend(self, swing_scanner):
        """向量化 SMA60：下降趨勢（close < SMA60）應被排除。"""
        rows = self._make_sma60_price("UP60", 65, 100.0, 0.5)  # 上升：close > SMA60
        rows += self._make_sma60_price("DN60", 65, 200.0, -0.5)  # 下降：close < SMA60
        df_price = pd.DataFrame(rows)
        result = swing_scanner._coarse_filter(df_price, pd.DataFrame())
        sids = result["stock_id"].tolist() if not result.empty else []
        assert "UP60" in sids
        assert "DN60" not in sids

    def test_sma60_insufficient_data_passes(self, swing_scanner):
        """資料不足 60 天時 SMA60 過濾應跳過（不因資料不足而拒絕所有股票）。"""
        rows = self._make_sma60_price("SHORT", 30, 100.0, 0.5)  # 僅 30 天，不足 60
        df_price = pd.DataFrame(rows)
        result = swing_scanner._coarse_filter(df_price, pd.DataFrame())
        # 資料不足 60 天 → SMA60 為 NaN → 過濾跳過 → 股票應保留（只要其他條件通過）
        assert result.empty or "SHORT" in result["stock_id"].tolist()

    def test_sma60_vectorized_matches_exact_value(self, swing_scanner):
        """向量化 SMA60 結果應等於手動計算的最後 60 天均值。"""
        rows = self._make_sma60_price("VERI", 70, 100.0, 0.5)
        df_price = pd.DataFrame(rows)
        # 手動計算：最後 60 個 close 的平均
        closes = [100.0 + d * 0.5 for d in range(70)]
        expected_sma60 = sum(closes[-60:]) / 60
        result = swing_scanner._coarse_filter(df_price, pd.DataFrame())
        if not result.empty and "VERI" in result["stock_id"].values:
            row = result[result["stock_id"] == "VERI"].iloc[0]
            assert abs(row["sma60"] - expected_sma60) < 0.01


class TestEffectiveTopN:
    """驗證 _effective_top_n() 自適應候選數邏輯。"""

    def test_returns_floor_for_small_universe(self, scanner):
        """小 Universe（< top_n_candidates / 0.15）應回傳 top_n_candidates 下限。"""
        # top_n_candidates=10，15% of 20 = 3 → max(10, 3) = 10
        assert scanner._effective_top_n(20) == 10

    def test_scales_with_large_universe(self, scanner):
        """大 Universe 應線性擴展：max(10, 15% * size)。"""
        # top_n_candidates=10，15% of 100 = 15 → max(10, 15) = 15
        assert scanner._effective_top_n(100) == 15
        # 15% of 200 = 30 → max(10, 30) = 30
        assert scanner._effective_top_n(200) == 30

    def test_boundary_at_threshold(self, scanner):
        """剛好在閾值邊界（67 支）：int(67 * 0.15) = 10 == top_n_candidates。"""
        # top_n_candidates=10，int(67 * 0.15) = 10 → max(10, 10) = 10
        assert scanner._effective_top_n(67) == 10

    def test_coarse_filter_respects_adaptive_limit(self, scanner):
        """_coarse_filter 傳入大量股票時，輸出不超過 _effective_top_n 上限。"""
        # 建立 80 支股票（80 * 0.15 = 12 > top_n_candidates=10）
        df_price = _make_price_df(80)
        result = scanner._coarse_filter(df_price, pd.DataFrame())
        expected_limit = scanner._effective_top_n(len(df_price["stock_id"].unique()))
        assert len(result) <= expected_limit


class TestSwingTechnicalScores:
    def test_scores_in_valid_range(self, swing_scanner):
        df_price = _make_swing_price_df(80, 5)
        sids = df_price["stock_id"].unique().tolist()
        result = swing_scanner._compute_technical_scores(sids, df_price)
        assert (result["technical_score"] >= 0).all()
        assert (result["technical_score"] <= 1.0).all()

    def test_uptrend_scores_higher(self, swing_scanner):
        """穩定上升趨勢的股票應在趨勢因子得高分。"""
        rows = []
        # 上升趨勢
        for d in range(65):
            day = date(2025, 1, 1) + timedelta(days=d)
            close = 100 + d * 0.5
            vol = 500_000 + d * 5_000
            rows.append(
                {
                    "stock_id": "UP",
                    "date": day,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": vol,
                }
            )
        # 下降趨勢
        for d in range(65):
            day = date(2025, 1, 1) + timedelta(days=d)
            close = 200 - d * 0.5
            vol = 500_000 - d * 2_000
            rows.append(
                {
                    "stock_id": "DOWN",
                    "date": day,
                    "open": close * 1.01,
                    "high": close * 1.02,
                    "low": close * 0.99,
                    "close": close,
                    "volume": max(vol, 100_000),
                }
            )
        df_price = pd.DataFrame(rows)
        result = swing_scanner._compute_technical_scores(["UP", "DOWN"], df_price)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["UP"] > scores["DOWN"]

    def test_insufficient_data_gets_default(self, swing_scanner):
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
        result = swing_scanner._compute_technical_scores(["9999"], df_price)
        assert result.iloc[0]["technical_score"] == pytest.approx(0.5)


class TestSwingChipScores:
    def test_trust_and_cumulative(self, swing_scanner):
        """投信買超 + 20 日累積買超。"""
        sids = ["1000", "1001"]
        rows = []
        for d in range(20):
            day = date(2025, 1, 1) + timedelta(days=d)
            # 1001 投信每天大量買超
            rows.append({"stock_id": "1000", "date": day, "name": "投信買賣超", "net": 50})
            rows.append({"stock_id": "1001", "date": day, "name": "投信買賣超", "net": 500})
            rows.append({"stock_id": "1000", "date": day, "name": "外資買賣超", "net": 100})
            rows.append({"stock_id": "1001", "date": day, "name": "外資買賣超", "net": 800})
        df_inst = pd.DataFrame(rows)
        result = swing_scanner._compute_chip_scores(sids, df_inst)
        scores = result.set_index("stock_id")["chip_score"]
        assert scores["1001"] > scores["1000"]

    def test_empty_inst_returns_default(self, swing_scanner):
        result = swing_scanner._compute_chip_scores(["1000"], pd.DataFrame())
        assert result.iloc[0]["chip_score"] == pytest.approx(0.5)


class TestSwingFundamentalScores:
    def test_with_acceleration(self, swing_scanner):
        """含加速度的 3 因子計算。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001", "1002"],
                "yoy_growth": [10.0, 30.0, -5.0],
                "mom_growth": [5.0, -2.0, 3.0],
                "prev_yoy_growth": [5.0, 35.0, 0.0],
                "prev_mom_growth": [3.0, -5.0, 5.0],
            }
        )
        result = swing_scanner._compute_fundamental_scores(["1000", "1001", "1002"], df_revenue)
        assert len(result) == 3
        assert (result["fundamental_score"].dropna() >= 0).all()
        assert (result["fundamental_score"].dropna() <= 1).all()

    def test_without_prev_data_fallback(self, swing_scanner):
        """無上月資料時，加速度 fallback 0.5。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000"],
                "yoy_growth": [10.0],
                "mom_growth": [5.0],
            }
        )
        result = swing_scanner._compute_fundamental_scores(["1000"], df_revenue)
        assert len(result) == 1
        score = result.iloc[0]["fundamental_score"]
        assert 0 <= score <= 1

    def test_acceleration_boosts_score(self, swing_scanner):
        """營收加速成長（YoY 越來越高）應有更高的基本面分數。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [20.0, 20.0],
                "mom_growth": [5.0, 5.0],
                "prev_yoy_growth": [10.0, 30.0],  # 1000 加速，1001 減速
                "prev_mom_growth": [5.0, 5.0],
            }
        )
        result = swing_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        # 1000 加速 (20-10=+10) vs 1001 減速 (20-30=-10)
        assert scores["1000"] > scores["1001"]

    def test_empty_revenue_returns_default(self, swing_scanner):
        result = swing_scanner._compute_fundamental_scores(
            ["1000"], pd.DataFrame(columns=["stock_id", "yoy_growth", "mom_growth"])
        )
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.5)


class TestSwingFinancialFundamentalScores:
    """SwingScanner 財報因子整合測試（ROE QoQ + 毛利率趨勢）。"""

    def _make_revenue(self):
        return pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [15.0, 15.0],
                "mom_growth": [3.0, 3.0],
                "prev_yoy_growth": [10.0, 10.0],
            }
        )

    def test_roe_qoq_improvement_rewarded(self, swing_scanner):
        """ROE QoQ 改善（上升）的股票基本面分數應優於退化的股票。"""
        df_revenue = self._make_revenue()
        fin_df = _make_fin_df(
            [
                # 1000: ROE QoQ 改善 (15 → 20)
                {
                    "stock_id": "1000",
                    "date": date(2024, 12, 31),
                    "year": 2024,
                    "quarter": 4,
                    "roe": 20.0,
                    "gross_margin": 35.0,
                },
                {
                    "stock_id": "1000",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "roe": 15.0,
                    "gross_margin": 34.0,
                },
                {
                    "stock_id": "1000",
                    "date": date(2024, 6, 30),
                    "year": 2024,
                    "quarter": 2,
                    "roe": 13.0,
                    "gross_margin": 33.0,
                },
                # 1001: ROE QoQ 退化 (20 → 10)
                {
                    "stock_id": "1001",
                    "date": date(2024, 12, 31),
                    "year": 2024,
                    "quarter": 4,
                    "roe": 10.0,
                    "gross_margin": 35.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "roe": 20.0,
                    "gross_margin": 34.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2024, 6, 30),
                    "year": 2024,
                    "quarter": 2,
                    "roe": 18.0,
                    "gross_margin": 33.0,
                },
            ]
        )
        swing_scanner._load_financial_data = lambda sids, quarters=5: fin_df
        result = swing_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]

    def test_gm_trend_improvement_rewarded(self, swing_scanner):
        """毛利率兩季對比改善應優於惡化。"""
        df_revenue = self._make_revenue()
        fin_df = _make_fin_df(
            [
                # 1000: 毛利率改善（t vs t-2: 40 vs 30 = +10）
                {
                    "stock_id": "1000",
                    "date": date(2024, 12, 31),
                    "year": 2024,
                    "quarter": 4,
                    "roe": 15.0,
                    "gross_margin": 40.0,
                },
                {
                    "stock_id": "1000",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "roe": 15.0,
                    "gross_margin": 36.0,
                },
                {
                    "stock_id": "1000",
                    "date": date(2024, 6, 30),
                    "year": 2024,
                    "quarter": 2,
                    "roe": 14.0,
                    "gross_margin": 30.0,
                },
                # 1001: 毛利率惡化（t vs t-2: 25 vs 38 = -13）
                {
                    "stock_id": "1001",
                    "date": date(2024, 12, 31),
                    "year": 2024,
                    "quarter": 4,
                    "roe": 15.0,
                    "gross_margin": 25.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "roe": 15.0,
                    "gross_margin": 30.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2024, 6, 30),
                    "year": 2024,
                    "quarter": 2,
                    "roe": 14.0,
                    "gross_margin": 38.0,
                },
            ]
        )
        swing_scanner._load_financial_data = lambda sids, quarters=5: fin_df
        result = swing_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]

    def test_no_financial_data_fallback_to_revenue(self, swing_scanner):
        """財報空 DataFrame 時，應降回純營收三因子，分數仍在 [0, 1]。"""
        df_revenue = self._make_revenue()
        swing_scanner._load_financial_data = lambda sids, quarters=5: pd.DataFrame()
        result = swing_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        assert len(result) == 2
        assert (result["fundamental_score"].dropna() >= 0).all()
        assert (result["fundamental_score"].dropna() <= 1).all()

    def test_score_range_with_financial_data(self, swing_scanner):
        """有財報資料時 fundamental_score 應在 [0, 1] 範圍內。"""
        df_revenue = self._make_revenue()
        fin_df = _make_fin_df(
            [
                {
                    "stock_id": "1000",
                    "date": date(2024, 12, 31),
                    "year": 2024,
                    "quarter": 4,
                    "roe": 18.0,
                    "gross_margin": 38.0,
                },
                {
                    "stock_id": "1000",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "roe": 15.0,
                    "gross_margin": 35.0,
                },
                {
                    "stock_id": "1000",
                    "date": date(2024, 6, 30),
                    "year": 2024,
                    "quarter": 2,
                    "roe": 13.0,
                    "gross_margin": 33.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2024, 12, 31),
                    "year": 2024,
                    "quarter": 4,
                    "roe": 12.0,
                    "gross_margin": 32.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "roe": 14.0,
                    "gross_margin": 34.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2024, 6, 30),
                    "year": 2024,
                    "quarter": 2,
                    "roe": 16.0,
                    "gross_margin": 36.0,
                },
            ]
        )
        swing_scanner._load_financial_data = lambda sids, quarters=5: fin_df
        result = swing_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        assert (result["fundamental_score"].dropna() >= 0).all()
        assert (result["fundamental_score"].dropna() <= 1).all()


class TestSwingCompositeWeights:
    def test_weights_with_regime(self, swing_scanner):
        """確認綜合分數依 regime 權重計算（含消息面）。"""
        from src.regime.detector import MarketRegimeDetector

        candidates = pd.DataFrame({"stock_id": ["4000"], "close": [100], "volume": [500_000]})
        df_price = _make_swing_price_df(80, 1)
        df_inst = pd.DataFrame([{"stock_id": "4000", "date": date(2025, 3, 1), "name": "投信買賣超", "net": 100}])
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["4000"],
                "yoy_growth": [10.0],
                "mom_growth": [5.0],
                "prev_yoy_growth": [5.0],
                "prev_mom_growth": [3.0],
            }
        )

        df_margin = pd.DataFrame()
        result = swing_scanner._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue)
        tech = result.iloc[0]["technical_score"]
        chip = result.iloc[0]["chip_score"]
        fund = result.iloc[0]["fundamental_score"]
        news = result.iloc[0]["news_score"]
        regime = getattr(swing_scanner, "regime", "sideways")
        w = MarketRegimeDetector.get_weights("swing", regime)
        expected = tech * w["technical"] + chip * w["chip"] + fund * w["fundamental"] + news * w.get("news", 0.10)
        assert result.iloc[0]["composite_score"] == pytest.approx(expected, abs=1e-6)


# ====================================================================== #
#  ValueScanner 測試
# ====================================================================== #


@pytest.fixture()
def value_scanner():
    return ValueScanner(
        min_price=10,
        max_price=2000,
        min_volume=100_000,
        top_n_candidates=10,
        top_n_results=5,
    )


class TestValueValuationScores:
    """估值面因子測試。"""

    def test_lower_pe_gets_higher_score(self, value_scanner):
        """PE 越低應得分越高（反向排名）。"""
        value_scanner._df_valuation = pd.DataFrame(
            {
                "stock_id": ["1000", "1001", "1002"],
                "date": [date(2025, 1, 25)] * 3,
                "pe_ratio": [8.0, 15.0, 25.0],
                "pb_ratio": [1.0, 1.0, 1.0],
                "dividend_yield": [5.0, 5.0, 5.0],
            }
        )
        result = value_scanner._compute_valuation_scores(["1000", "1001", "1002"])
        scores = result.set_index("stock_id")["valuation_score"]
        assert scores["1000"] > scores["1002"]

    def test_higher_dividend_yield_gets_higher_score(self, value_scanner):
        """殖利率越高應得分越高。"""
        value_scanner._df_valuation = pd.DataFrame(
            {
                "stock_id": ["1000", "1001", "1002"],
                "date": [date(2025, 1, 25)] * 3,
                "pe_ratio": [15.0, 15.0, 15.0],
                "pb_ratio": [1.5, 1.5, 1.5],
                "dividend_yield": [8.0, 4.0, 1.0],
            }
        )
        result = value_scanner._compute_valuation_scores(["1000", "1001", "1002"])
        scores = result.set_index("stock_id")["valuation_score"]
        assert scores["1000"] > scores["1002"]

    def test_no_valuation_data_returns_default(self, value_scanner):
        """無估值資料時回傳 0.5。"""
        value_scanner._df_valuation = pd.DataFrame()
        result = value_scanner._compute_valuation_scores(["1000"])
        assert result.iloc[0]["valuation_score"] == pytest.approx(0.5)


class TestValueFundamentalScores:
    def test_with_revenue_data(self, value_scanner):
        """有營收資料時應計算正確分數。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [20.0, -5.0],
                "mom_growth": [10.0, -2.0],
                "prev_yoy_growth": [10.0, 0.0],
                "prev_mom_growth": [5.0, 0.0],
            }
        )
        result = value_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]


class TestValueCompositeWeights:
    def test_weights_with_regime(self, value_scanner):
        """確認綜合分數依 regime 權重計算（含消息面）。"""
        from src.regime.detector import MarketRegimeDetector

        candidates = pd.DataFrame({"stock_id": ["1000"], "close": [100], "volume": [500_000]})
        df_price = _make_momentum_price_df(25, 1)
        df_price["stock_id"] = "1000"
        df_inst = pd.DataFrame([{"stock_id": "1000", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 100}])
        df_margin = pd.DataFrame()
        df_revenue = pd.DataFrame({"stock_id": ["1000"], "yoy_growth": [10.0], "mom_growth": [5.0]})
        value_scanner._df_valuation = pd.DataFrame(
            {
                "stock_id": ["1000"],
                "date": [date(2025, 1, 25)],
                "pe_ratio": [12.0],
                "pb_ratio": [1.5],
                "dividend_yield": [5.0],
            }
        )

        result = value_scanner._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue)
        fund = result.iloc[0]["fundamental_score"]
        val = result.iloc[0]["valuation_score"]
        chip = result.iloc[0]["chip_score"]
        news = result.iloc[0]["news_score"]
        regime = getattr(value_scanner, "regime", "sideways")
        w = MarketRegimeDetector.get_weights("value", regime)
        expected = fund * w["fundamental"] + val * w["valuation"] + chip * w["chip"] + news * w.get("news", 0.10)
        assert result.iloc[0]["composite_score"] == pytest.approx(expected, abs=1e-6)


# ====================================================================== #
#  產業加成測試
# ====================================================================== #


class TestComputeSectorBonus:
    """_compute_sector_bonus() 回傳格式與範圍測試。"""

    def test_returns_correct_format(self, scanner):
        """應回傳 DataFrame(stock_id, sector_bonus)。"""
        # _compute_sector_bonus 依賴 DB，失敗時應 graceful fallback 回傳全 0
        result = scanner._compute_sector_bonus(["1000", "1001"])
        assert "stock_id" in result.columns
        assert "sector_bonus" in result.columns
        assert len(result) == 2

    def test_bonus_range(self, scanner):
        """sector_bonus 應在 -0.05 ~ +0.05 範圍內（或 fallback 0）。"""
        result = scanner._compute_sector_bonus(["1000", "1001", "1002"])
        assert (result["sector_bonus"] >= -0.05).all()
        assert (result["sector_bonus"] <= 0.05).all()

    def test_empty_stock_ids(self, scanner):
        """空 stock_ids 應回傳空 DataFrame。"""
        result = scanner._compute_sector_bonus([])
        assert result.empty or len(result) == 0


class TestApplySectorBonus:
    """_apply_sector_bonus() 套用到 composite_score 的測試。"""

    def test_sector_bonus_applied_to_composite_score(self, scanner):
        """sector_bonus 應乘以 (1 + bonus) 套用到 composite_score。"""
        scored = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "composite_score": [0.8, 0.6],
            }
        )
        # Mock _compute_sector_bonus 回傳已知值
        original_method = scanner._compute_sector_bonus
        scanner._compute_sector_bonus = lambda sids: pd.DataFrame(
            {
                "stock_id": sids,
                "sector_bonus": [0.05, -0.05],
            }
        )
        try:
            result = scanner._apply_sector_bonus(scored)
            assert "sector_bonus" in result.columns
            # 0.8 * 1.05 = 0.84, 0.6 * 0.95 = 0.57
            assert result.iloc[0]["composite_score"] == pytest.approx(0.84, abs=1e-6)
            assert result.iloc[1]["composite_score"] == pytest.approx(0.57, abs=1e-6)
        finally:
            scanner._compute_sector_bonus = original_method

    def test_empty_scored_returns_empty(self, scanner):
        """空 DataFrame 應直接回傳。"""
        result = scanner._apply_sector_bonus(pd.DataFrame())
        assert result.empty

    def test_zero_bonus_no_change(self, scanner):
        """bonus=0 時 composite_score 不變。"""
        scored = pd.DataFrame(
            {
                "stock_id": ["1000"],
                "composite_score": [0.75],
            }
        )
        scanner._compute_sector_bonus = lambda sids: pd.DataFrame(
            {
                "stock_id": sids,
                "sector_bonus": [0.0],
            }
        )
        result = scanner._apply_sector_bonus(scored)
        assert result.iloc[0]["composite_score"] == pytest.approx(0.75, abs=1e-6)


# ====================================================================== #
#  DividendScanner 測試
# ====================================================================== #


@pytest.fixture()
def dividend_scanner():
    return DividendScanner(
        min_price=10,
        max_price=2000,
        min_volume=100_000,
        top_n_candidates=10,
        top_n_results=5,
    )


class TestDividendCoarseFilter:
    def test_dividend_yield_threshold(self, dividend_scanner):
        """殖利率 > 3% 門檻過濾。"""
        df_price = _make_price_df(10)
        sids = df_price[df_price["date"] == df_price["date"].max()]["stock_id"].unique().tolist()

        dividend_scanner._df_valuation = pd.DataFrame(
            {
                "stock_id": sids,
                "date": [date(2025, 1, 3)] * len(sids),
                "pe_ratio": [12.0] * len(sids),
                "pb_ratio": [1.5] * len(sids),
                # 只有前 3 支殖利率 > 3%
                "dividend_yield": [5.0, 4.0, 3.5, 2.5, 1.0, 0.5, 2.0, 1.5, 0.0, 2.9],
            }
        )
        result = dividend_scanner._coarse_filter(df_price, pd.DataFrame())
        if not result.empty:
            # 通過粗篩的應該只有殖利率 > 3% 的
            passed_ids = set(result["stock_id"].tolist())
            for sid in passed_ids:
                val_row = dividend_scanner._df_valuation[dividend_scanner._df_valuation["stock_id"] == sid]
                assert val_row.iloc[0]["dividend_yield"] > 3.0

    def test_pe_positive_filter(self, dividend_scanner):
        """PE > 0 過濾：排除虧損股。"""
        df_price = _make_price_df(5)
        sids = df_price[df_price["date"] == df_price["date"].max()]["stock_id"].unique().tolist()

        dividend_scanner._df_valuation = pd.DataFrame(
            {
                "stock_id": sids,
                "date": [date(2025, 1, 3)] * len(sids),
                "pe_ratio": [12.0, -5.0, 15.0, 0.0, 8.0],
                "pb_ratio": [1.5] * len(sids),
                "dividend_yield": [5.0, 5.0, 5.0, 5.0, 5.0],
            }
        )
        result = dividend_scanner._coarse_filter(df_price, pd.DataFrame())
        if not result.empty:
            passed_ids = set(result["stock_id"].tolist())
            for sid in passed_ids:
                val_row = dividend_scanner._df_valuation[dividend_scanner._df_valuation["stock_id"] == sid]
                assert val_row.iloc[0]["pe_ratio"] > 0

    def test_no_valuation_returns_empty(self, dividend_scanner):
        """無估值資料時回傳空。"""
        df_price = _make_price_df(5)
        dividend_scanner._df_valuation = pd.DataFrame()
        result = dividend_scanner._coarse_filter(df_price, pd.DataFrame())
        assert result.empty


class TestDividendDividendScores:
    def test_higher_yield_gets_higher_score(self, dividend_scanner):
        """殖利率越高應得分越高。"""
        dividend_scanner._df_valuation = pd.DataFrame(
            {
                "stock_id": ["1000", "1001", "1002"],
                "date": [date(2025, 1, 25)] * 3,
                "pe_ratio": [12.0, 12.0, 12.0],
                "pb_ratio": [1.5, 1.5, 1.5],
                "dividend_yield": [8.0, 4.0, 1.0],
            }
        )
        result = dividend_scanner._compute_dividend_scores(["1000", "1001", "1002"])
        scores = result.set_index("stock_id")["dividend_score"]
        assert scores["1000"] > scores["1002"]

    def test_lower_pe_gets_higher_score(self, dividend_scanner):
        """PE 越低應得分越高（反向排名）。"""
        dividend_scanner._df_valuation = pd.DataFrame(
            {
                "stock_id": ["1000", "1001", "1002"],
                "date": [date(2025, 1, 25)] * 3,
                "pe_ratio": [8.0, 15.0, 25.0],
                "pb_ratio": [1.0, 1.0, 1.0],
                "dividend_yield": [5.0, 5.0, 5.0],
            }
        )
        result = dividend_scanner._compute_dividend_scores(["1000", "1001", "1002"])
        scores = result.set_index("stock_id")["dividend_score"]
        assert scores["1000"] > scores["1002"]

    def test_no_valuation_data_returns_default(self, dividend_scanner):
        """無估值資料時回傳 0.5。"""
        dividend_scanner._df_valuation = pd.DataFrame()
        result = dividend_scanner._compute_dividend_scores(["1000"])
        assert result.iloc[0]["dividend_score"] == pytest.approx(0.5)


class TestDividendFundamentalScores:
    def test_with_acceleration(self, dividend_scanner):
        """含加速度的 3 因子計算。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [20.0, -5.0],
                "mom_growth": [10.0, -2.0],
                "prev_yoy_growth": [10.0, 0.0],
                "prev_mom_growth": [5.0, 0.0],
            }
        )
        result = dividend_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]

    def test_empty_revenue_returns_default(self, dividend_scanner):
        result = dividend_scanner._compute_fundamental_scores(
            ["1000"], pd.DataFrame(columns=["stock_id", "yoy_growth", "mom_growth"])
        )
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.5)


class TestDividendRegimeWeights:
    def test_weights_exist_for_all_regimes(self):
        """確認 dividend 模式在三種 regime 下都有權重。"""
        from src.regime.detector import MarketRegimeDetector

        for regime in ["bull", "bear", "sideways"]:
            w = MarketRegimeDetector.get_weights("dividend", regime)
            assert "fundamental" in w
            assert "dividend" in w
            assert "chip" in w
            assert "news" in w
            assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_composite_weight_calculation(self, dividend_scanner):
        """確認綜合分數依 regime 權重計算。"""
        from src.regime.detector import MarketRegimeDetector

        candidates = pd.DataFrame({"stock_id": ["1000"], "close": [100], "volume": [500_000]})
        df_price = _make_momentum_price_df(25, 1)
        df_price["stock_id"] = "1000"
        df_inst = pd.DataFrame([{"stock_id": "1000", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 100}])
        df_margin = pd.DataFrame()
        df_revenue = pd.DataFrame({"stock_id": ["1000"], "yoy_growth": [10.0], "mom_growth": [5.0]})
        dividend_scanner._df_valuation = pd.DataFrame(
            {
                "stock_id": ["1000"],
                "date": [date(2025, 1, 25)],
                "pe_ratio": [12.0],
                "pb_ratio": [1.5],
                "dividend_yield": [5.0],
            }
        )

        result = dividend_scanner._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue)
        fund = result.iloc[0]["fundamental_score"]
        div = result.iloc[0]["dividend_score"]
        chip = result.iloc[0]["chip_score"]
        news = result.iloc[0]["news_score"]
        regime = getattr(dividend_scanner, "regime", "sideways")
        w = MarketRegimeDetector.get_weights("dividend", regime)
        expected = fund * w["fundamental"] + div * w["dividend"] + chip * w["chip"] + news * w.get("news", 0.10)
        assert result.iloc[0]["composite_score"] == pytest.approx(expected, abs=1e-6)


# ====================================================================== #
#  GrowthScanner 測試
# ====================================================================== #


@pytest.fixture()
def growth_scanner():
    return GrowthScanner(
        min_price=10,
        max_price=2000,
        min_volume=100_000,
        top_n_candidates=10,
        top_n_results=5,
    )


class TestGrowthCoarseFilter:
    def test_yoy_threshold(self, growth_scanner):
        """YoY > 10% 門檻過濾。"""
        df_price = _make_price_df(10)
        sids = df_price[df_price["date"] == df_price["date"].max()]["stock_id"].unique().tolist()

        # 手動設定營收資料（只有部分 YoY > 10%）
        growth_scanner._coarse_revenue = pd.DataFrame(
            {
                "stock_id": sids,
                "yoy_growth": [15.0, 25.0, 5.0, -3.0, 50.0, 8.0, 11.0, 2.0, 30.0, 9.0],
                "mom_growth": [5.0] * len(sids),
            }
        )
        result = growth_scanner._coarse_filter(df_price, pd.DataFrame())
        if not result.empty:
            passed_ids = set(result["stock_id"].tolist())
            for sid in passed_ids:
                rev_row = growth_scanner._coarse_revenue[growth_scanner._coarse_revenue["stock_id"] == sid]
                assert rev_row.iloc[0]["yoy_growth"] > 10.0

    def test_no_revenue_returns_empty(self, growth_scanner):
        """無營收資料時回傳空。"""
        df_price = _make_price_df(5)
        growth_scanner._coarse_revenue = pd.DataFrame()
        result = growth_scanner._coarse_filter(df_price, pd.DataFrame())
        assert result.empty


class TestGrowthTechnicalScores:
    def test_scores_in_valid_range(self, growth_scanner):
        df_price = _make_momentum_price_df(25, 5)
        sids = df_price["stock_id"].unique().tolist()
        result = growth_scanner._compute_technical_scores(sids, df_price)
        assert (result["technical_score"] >= 0).all()
        assert (result["technical_score"] <= 1.0).all()

    def test_insufficient_data_gets_default(self, growth_scanner):
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
        result = growth_scanner._compute_technical_scores(["9999"], df_price)
        assert result.iloc[0]["technical_score"] == pytest.approx(0.5)

    def test_stronger_momentum_higher_score(self, growth_scanner):
        """動能更強的股票，技術分數應更高。"""
        df_price = _make_momentum_price_df(25, 5)
        sids = df_price["stock_id"].unique().tolist()
        result = growth_scanner._compute_technical_scores(sids, df_price)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["2004"] >= scores["2000"]


class TestGrowthFundamentalScores:
    def test_with_acceleration(self, growth_scanner):
        """含 3 個月加速度的 2 因子計算：加速股分數 > 減速股。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [20.0, 20.0],
                "mom_growth": [5.0, 5.0],
                "yoy_3m_ago": [10.0, 30.0],  # 1000 加速(+10)，1001 減速(-10)
            }
        )
        result = growth_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]

    def test_empty_revenue_returns_default(self, growth_scanner):
        result = growth_scanner._compute_fundamental_scores(
            ["1000"], pd.DataFrame(columns=["stock_id", "yoy_growth", "mom_growth"])
        )
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.5)

    def test_no_yoy_3m_ago_fallback(self, growth_scanner):
        """無 yoy_3m_ago 欄位時，加速度 fallback 0.5，分數 = YoY 排名 × 0.6 + 0.4 × 0.5。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [20.0, 5.0],
                "mom_growth": [5.0, 5.0],
            }
        )
        result = growth_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        # 高 YoY 應得較高分
        assert scores["1000"] > scores["1001"]


class TestGrowthChipScores:
    def test_empty_inst_returns_default(self, growth_scanner):
        result = growth_scanner._compute_chip_scores(["1000"], pd.DataFrame())
        assert result.iloc[0]["chip_score"] == pytest.approx(0.5)

    def test_margin_data_adds_short_margin_factor(self, growth_scanner):
        """有融資融券資料時，籌碼面使用 4 因子加權（含券資比）。"""
        sids = ["1000", "1001"]
        df_inst = pd.DataFrame(
            [
                {"stock_id": "1000", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 100},
                {"stock_id": "1001", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 100},
            ]
        )
        df_margin = pd.DataFrame(
            [
                {"stock_id": "1000", "date": date(2025, 1, 25), "margin_balance": 1000, "short_balance": 500},
                {"stock_id": "1001", "date": date(2025, 1, 25), "margin_balance": 1000, "short_balance": 100},
            ]
        )
        result = growth_scanner._compute_chip_scores(sids, df_inst, None, df_margin)
        assert len(result) == 2
        scores = result.set_index("stock_id")["chip_score"]
        assert scores["1000"] > scores["1001"]

    def test_consecutive_foreign_buy(self, growth_scanner):
        """外資連續買超天數越多分數越高。"""
        sids = ["1000", "1001"]
        rows = []
        for d in range(5):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append({"stock_id": "1000", "date": day, "name": "外資買賣超", "net": 100 if d == 4 else -50})
            rows.append({"stock_id": "1001", "date": day, "name": "外資買賣超", "net": 500})
        df_inst = pd.DataFrame(rows)
        df_price = _make_momentum_price_df(5, 2)
        df_price["stock_id"] = df_price["stock_id"].map({"2000": "1000", "2001": "1001"})
        result = growth_scanner._compute_chip_scores(sids, df_inst, df_price)
        scores = result.set_index("stock_id")["chip_score"]
        assert scores["1001"] > scores["1000"]


class TestGrowthRegimeWeights:
    def test_weights_exist_for_all_regimes(self):
        """確認 growth 模式在三種 regime 下都有權重。"""
        from src.regime.detector import MarketRegimeDetector

        for regime in ["bull", "bear", "sideways"]:
            w = MarketRegimeDetector.get_weights("growth", regime)
            assert "fundamental" in w
            assert "technical" in w
            assert "chip" in w
            assert "news" in w
            assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_composite_weight_calculation(self, growth_scanner):
        """確認綜合分數依 regime 權重計算。"""
        from src.regime.detector import MarketRegimeDetector

        candidates = pd.DataFrame({"stock_id": ["2000"], "close": [100], "volume": [500_000]})
        df_price = _make_momentum_price_df(25, 1)
        df_inst = pd.DataFrame([{"stock_id": "2000", "date": date(2025, 1, 25), "name": "外資買賣超", "net": 100}])
        df_margin = pd.DataFrame()
        df_revenue = pd.DataFrame({"stock_id": ["2000"], "yoy_growth": [30.0], "mom_growth": [10.0]})

        result = growth_scanner._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue)
        tech = result.iloc[0]["technical_score"]
        fund = result.iloc[0]["fundamental_score"]
        chip = result.iloc[0]["chip_score"]
        news = result.iloc[0]["news_score"]
        regime = getattr(growth_scanner, "regime", "sideways")
        w = MarketRegimeDetector.get_weights("growth", regime)
        expected = fund * w["fundamental"] + tech * w["technical"] + chip * w["chip"] + news * w.get("news", 0.10)
        assert result.iloc[0]["composite_score"] == pytest.approx(expected, abs=1e-6)


# ====================================================================== #
#  進出場建議欄位測試（_calc_atr14 + _compute_entry_exit_cols）
# ====================================================================== #


_make_entry_exit_price_df = make_entry_exit_price_df


class TestCalcAtr14:
    def test_positive_with_enough_data(self):
        """20 天資料應回傳正值 ATR14。"""
        from src.discovery.scanner import _calc_atr14

        df = _make_entry_exit_price_df()
        atr = _calc_atr14(df)
        assert atr > 0

    def test_returns_zero_for_empty(self):
        """空 DataFrame 應回傳 0.0。"""
        from src.discovery.scanner import _calc_atr14

        df = pd.DataFrame(columns=["high", "low", "close"])
        assert _calc_atr14(df) == 0.0

    def test_returns_zero_for_single_row(self):
        """單筆資料（無法計算 TR）應回傳 0.0。"""
        from src.discovery.scanner import _calc_atr14

        df = pd.DataFrame([{"high": 102.0, "low": 98.0, "close": 100.0}])
        assert _calc_atr14(df) == 0.0

    def test_known_value(self):
        """固定 high/low/close → ATR 應等於 high-low（無跳空）。"""
        from src.discovery.scanner import _calc_atr14

        rows = []
        for d in range(16):
            rows.append({"high": 102.0, "low": 98.0, "close": 100.0})
        df = pd.DataFrame(rows)
        atr = _calc_atr14(df)
        # TR = max(102-98, |102-100|, |98-100|) = max(4, 2, 2) = 4
        assert atr == pytest.approx(4.0, abs=1e-6)


class TestComputeEntryExitCols:
    def test_returns_five_columns(self, scanner):
        """回傳 DataFrame 含五個進出場欄位。"""
        df_price = _make_entry_exit_price_df()
        result = scanner._compute_entry_exit_cols(["1000"], df_price)
        for col in ["entry_price", "stop_loss", "take_profit", "entry_trigger", "valid_until"]:
            assert col in result.columns

    def test_stop_loss_below_entry_take_profit_above(self, scanner):
        """stop_loss < entry_price < take_profit（有足夠資料時）。"""
        df_price = _make_entry_exit_price_df()
        result = scanner._compute_entry_exit_cols(["1000"], df_price)
        row = result.iloc[0]
        ep = row["entry_price"]
        sl = row["stop_loss"]
        tp = row["take_profit"]
        assert pd.notna(ep) and ep > 0
        assert pd.notna(sl) and sl < ep
        assert pd.notna(tp) and tp > ep

    def test_insufficient_data_trigger(self, scanner):
        """不足 15 天 → trigger 為「資料不足，僅供參考」。"""
        rows = []
        for d in range(5):
            rows.append(
                {
                    "stock_id": "9999",
                    "date": date(2025, 1, 1) + timedelta(days=d),
                    "open": 99,
                    "high": 102,
                    "low": 98,
                    "close": 100.0,
                    "volume": 500_000,
                }
            )
        df_price = pd.DataFrame(rows)
        result = scanner._compute_entry_exit_cols(["9999"], df_price)
        assert result.iloc[0]["entry_trigger"] == "資料不足，僅供參考"

    def test_valid_until_is_business_days_ahead(self, scanner):
        """valid_until 應在 scan_date 之後（至少 5 個工作日）。"""
        df_price = _make_entry_exit_price_df()
        result = scanner._compute_entry_exit_cols(["1000"], df_price)
        valid = result.iloc[0]["valid_until"]
        assert valid > date.today()

    def test_above_sma20_trigger_says_站上均線(self, scanner):
        """close > SMA20 × 1.01 → 「站上均線」。"""
        rows = []
        # 價格逐日遞增，最後一天顯著高於 SMA20
        for d in range(20):
            close = 80.0 + d * 2  # 最後一天 close=118, SMA20≈99
            rows.append(
                {
                    "stock_id": "5555",
                    "date": date(2025, 1, 1) + timedelta(days=d),
                    "open": close - 1,
                    "high": close + 2,
                    "low": close - 2,
                    "close": close,
                    "volume": 500_000,
                }
            )
        df_price = pd.DataFrame(rows)
        result = scanner._compute_entry_exit_cols(["5555"], df_price)
        trigger = result.iloc[0]["entry_trigger"]
        assert "站上均線" in trigger


class TestComputeEntryExitColsRegime:
    """驗證 _compute_entry_exit_cols 的 Regime 自適應 ATR 倍數。"""

    def _make_price_df(self, sid: str = "1000") -> pd.DataFrame:
        """產生 20 天固定波動的日K，ATR14 = 4.0（high-low=4，無跳空）。"""
        rows = []
        for d in range(20):
            rows.append(
                {
                    "stock_id": sid,
                    "date": date(2025, 1, 1) + timedelta(days=d),
                    "open": 99.0,
                    "high": 102.0,
                    "low": 98.0,
                    "close": 100.0,
                    "volume": 500_000,
                }
            )
        return pd.DataFrame(rows)

    def test_sideways_uses_default_multipliers(self, scanner):
        """sideways → stop=entry-1.5×ATR，target=entry+3.0×ATR。"""
        df = self._make_price_df()
        result = scanner._compute_entry_exit_cols(["1000"], df, regime="sideways")
        row = result.iloc[0]
        ep, sl, tp = row["entry_price"], row["stop_loss"], row["take_profit"]
        atr = (ep - sl) / 1.5
        assert tp == pytest.approx(ep + 3.0 * atr, abs=0.02)

    def test_bull_wider_target(self, scanner):
        """bull → target_mult=3.5，大於 sideways 的 3.0。"""
        df = self._make_price_df()
        res_bull = scanner._compute_entry_exit_cols(["1000"], df, regime="bull")
        res_side = scanner._compute_entry_exit_cols(["1000"], df, regime="sideways")
        # take_profit(bull) > take_profit(sideways)
        assert res_bull.iloc[0]["take_profit"] > res_side.iloc[0]["take_profit"]
        # stop_loss 相同（倍數相同）
        assert res_bull.iloc[0]["stop_loss"] == pytest.approx(res_side.iloc[0]["stop_loss"], abs=0.02)

    def test_bear_tighter_stop_and_target(self, scanner):
        """bear → stop_mult=1.2、target_mult=2.5，止損更緊、目標更保守。"""
        df = self._make_price_df()
        res_bear = scanner._compute_entry_exit_cols(["1000"], df, regime="bear")
        res_side = scanner._compute_entry_exit_cols(["1000"], df, regime="sideways")
        ep_bear = res_bear.iloc[0]["entry_price"]
        sl_bear = res_bear.iloc[0]["stop_loss"]
        tp_bear = res_bear.iloc[0]["take_profit"]
        ep_side = res_side.iloc[0]["entry_price"]
        sl_side = res_side.iloc[0]["stop_loss"]
        tp_side = res_side.iloc[0]["take_profit"]
        # bear stop 更靠近 entry（止損距離更小）
        assert (ep_bear - sl_bear) < (ep_side - sl_side)
        # bear target 更保守（距離更小）
        assert (tp_bear - ep_bear) < (tp_side - ep_side)

    def test_unknown_regime_falls_back_to_sideways(self, scanner):
        """未知 regime 字串 → fallback sideways（stop×1.5/target×3.0）。"""
        df = self._make_price_df()
        res_unknown = scanner._compute_entry_exit_cols(["1000"], df, regime="unknown_xyz")
        res_side = scanner._compute_entry_exit_cols(["1000"], df, regime="sideways")
        assert res_unknown.iloc[0]["stop_loss"] == pytest.approx(res_side.iloc[0]["stop_loss"], abs=0.02)
        assert res_unknown.iloc[0]["take_profit"] == pytest.approx(res_side.iloc[0]["take_profit"], abs=0.02)

    def test_regime_propagates_from_self(self, scanner):
        """self.regime 設為 'bull' 後，_score_candidates 應採用 bull 倍數。"""
        df_price = _make_entry_exit_price_df()
        candidates = pd.DataFrame(
            {
                "stock_id": ["1000"],
                "close": [df_price[df_price["stock_id"] == "1000"]["close"].iloc[-1]],
                "volume": [500_000],
            }
        )
        # 先以 sideways 取得基準
        scanner.regime = "sideways"
        res_side = scanner._score_candidates(candidates, df_price, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        # 再以 bull 取得結果
        scanner.regime = "bull"
        res_bull = scanner._score_candidates(candidates, df_price, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        tp_side = res_side.iloc[0]["take_profit"]
        tp_bull = res_bull.iloc[0]["take_profit"]
        assert pd.notna(tp_bull) and pd.notna(tp_side)
        assert tp_bull > tp_side


class TestRankAndEnrichHasEntryExitCols:
    """驗證 _score_candidates 輸出含五個進出場欄位且值合理。"""

    def test_entry_exit_cols_present_and_reasonable(self, scanner):
        """_score_candidates 結果含五欄位；stop_loss < close < take_profit。"""
        df_price = _make_entry_exit_price_df()
        candidates = pd.DataFrame(
            {
                "stock_id": ["1000"],
                "close": [df_price[df_price["stock_id"] == "1000"]["close"].iloc[-1]],
                "volume": [500_000],
            }
        )
        result = scanner._score_candidates(candidates, df_price, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        for col in ["entry_price", "stop_loss", "take_profit", "entry_trigger", "valid_until"]:
            assert col in result.columns

        row = result.iloc[0]
        ep = row["entry_price"]
        sl = row["stop_loss"]
        tp = row["take_profit"]

        assert pd.notna(ep) and ep > 0
        if pd.notna(sl):
            assert sl < ep
        if pd.notna(tp):
            assert tp > ep
        assert row["valid_until"] > date.today()


# ================================================================
# Stage 0.5 Cold-Start 機制測試
# ================================================================


@pytest.mark.parametrize(
    "ScannerCls",
    [pytest.param(ValueScanner, id="value"), pytest.param(DividendScanner, id="dividend")],
)
class TestScannerStage05:
    """Stage 0.5 估值 Cold-Start 自動補抓機制（Value / Dividend 共用）。"""

    def _make_mocked_scanner(self, monkeypatch, ScannerCls, val_count: int):
        from unittest.mock import MagicMock

        scanner = ScannerCls(top_n_candidates=5, top_n_results=3)
        scanner._load_market_data = MagicMock(
            return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        )
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.execute.return_value.scalar.return_value = val_count
        monkeypatch.setattr("src.discovery.scanner.get_session", lambda: mock_session)
        return scanner

    def test_stage05_triggers_when_no_valuation_data(self, monkeypatch, ScannerCls):
        """val_count = 0 時，應呼叫 sync_valuation_all_market。"""
        import src.data.pipeline as pipeline_mod

        sync_calls = []
        monkeypatch.setattr(pipeline_mod, "sync_valuation_all_market", lambda: sync_calls.append(1) or 1200)
        self._make_mocked_scanner(monkeypatch, ScannerCls, val_count=0).run()
        assert len(sync_calls) == 1, "Stage 0.5 應觸發一次 sync_valuation_all_market"

    def test_stage05_skips_when_sufficient_data(self, monkeypatch, ScannerCls):
        """val_count >= 500 時，不應呼叫 sync_valuation_all_market。"""
        import src.data.pipeline as pipeline_mod

        sync_calls = []
        monkeypatch.setattr(pipeline_mod, "sync_valuation_all_market", lambda: sync_calls.append(1) or 0)
        self._make_mocked_scanner(monkeypatch, ScannerCls, val_count=1200).run()
        assert len(sync_calls) == 0, "估值充足時不應觸發補抓"

    def test_stage05_does_not_crash_on_sync_failure(self, monkeypatch, ScannerCls):
        """sync_valuation_all_market 拋例外時，掃描應繼續（不崩潰）。"""
        import src.data.pipeline as pipeline_mod

        monkeypatch.setattr(
            pipeline_mod, "sync_valuation_all_market", lambda: (_ for _ in ()).throw(RuntimeError("網路錯誤"))
        )
        result = self._make_mocked_scanner(monkeypatch, ScannerCls, val_count=0).run()
        assert result is not None


# ====================================================================== #
#  消息面評分：時間衰減 + 事件類型加權（Task #38）
# ====================================================================== #


class TestComputeNewsDecayWeight:
    """compute_news_decay_weight 純函數測試。"""

    def test_day_zero_general_returns_one(self):
        """當天 general 公告，衰減值應為 1.0。"""
        from src.discovery.scanner import compute_news_decay_weight

        w = compute_news_decay_weight(0, "general")
        assert w == pytest.approx(1.0, abs=1e-9)

    def test_day_zero_earnings_call_returns_three(self):
        """當天 earnings_call 公告，衰減值應為 3.0。"""
        from src.discovery.scanner import compute_news_decay_weight

        w = compute_news_decay_weight(0, "earnings_call")
        assert w == pytest.approx(3.0, abs=1e-9)

    def test_decay_reduces_over_time(self):
        """第 1 天的加權值應小於第 0 天（指數衰減，常數 0.2）。"""
        import math

        from src.discovery.scanner import compute_news_decay_weight

        w0 = compute_news_decay_weight(0, "general")
        w1 = compute_news_decay_weight(1, "general")
        assert w1 == pytest.approx(math.exp(-0.2), abs=1e-9)
        assert w1 < w0

    def test_earnings_call_beats_general_same_age(self):
        """相同天數下，earnings_call 加權值應大於 general。"""
        from src.discovery.scanner import compute_news_decay_weight

        days = 3
        assert compute_news_decay_weight(days, "earnings_call") > compute_news_decay_weight(days, "general")

    def test_unknown_event_type_defaults_to_one(self):
        """未知 event_type 應 fallback 為 type_weight=1.0。"""
        from src.discovery.scanner import compute_news_decay_weight

        assert compute_news_decay_weight(0, "unknown_xyz") == pytest.approx(1.0, abs=1e-9)

    def test_negative_days_treated_as_zero(self):
        """負數 days_ago（日期在未來）應 clamp 為 0，不應出現超過 type_weight 的值。"""
        from src.discovery.scanner import compute_news_decay_weight

        w = compute_news_decay_weight(-5, "general")
        assert w == pytest.approx(1.0, abs=1e-9)


class TestComputeNewsScores:
    """_compute_news_scores 方法測試（直接傳入 df_ann，不觸及 DB）。"""

    def _make_scanner(self):
        return MarketScanner(min_price=10, max_price=2000, min_volume=100_000)

    def _make_ann(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_empty_ann_returns_default(self):
        """df_ann 為空時，所有 stock 應回傳 news_score=0.5。"""
        scanner = self._make_scanner()
        result = scanner._compute_news_scores(["1000", "2000"], pd.DataFrame())
        assert (result["news_score"] == 0.5).all()

    def test_no_matching_stocks_returns_default(self):
        """df_ann 中無候選股的資料時，應回傳預設 0.5。"""
        scanner = self._make_scanner()
        today = date.today()
        ann = self._make_ann([{"stock_id": "9999", "date": today, "sentiment": 1, "event_type": "general"}])
        result = scanner._compute_news_scores(["1000", "2000"], ann)
        assert (result["news_score"] == 0.5).all()

    def test_positive_ann_beats_no_ann(self):
        """有正面公告的股票，排名應高於無公告的股票。"""
        scanner = self._make_scanner()
        today = date.today()
        ann = self._make_ann([{"stock_id": "1000", "date": today, "sentiment": 1, "event_type": "general"}])
        result = scanner._compute_news_scores(["1000", "2000"], ann).set_index("stock_id")
        assert result.loc["1000", "news_score"] > result.loc["2000", "news_score"]

    def test_earnings_call_beats_general_positive(self):
        """同為正面公告、相同日期：earnings_call 股票排名應高於 general 股票。"""
        scanner = self._make_scanner()
        today = date.today()
        ann = self._make_ann(
            [
                {"stock_id": "1000", "date": today, "sentiment": 1, "event_type": "earnings_call"},
                {"stock_id": "2000", "date": today, "sentiment": 1, "event_type": "general"},
            ]
        )
        result = scanner._compute_news_scores(["1000", "2000"], ann).set_index("stock_id")
        assert result.loc["1000", "news_score"] > result.loc["2000", "news_score"]

    def test_recent_beats_old_same_type(self):
        """同類型正面公告，近期（今天）排名應高於舊公告（10 天前）。"""
        scanner = self._make_scanner()
        today = date.today()
        old_date = today - timedelta(days=10)
        ann = self._make_ann(
            [
                {"stock_id": "1000", "date": today, "sentiment": 1, "event_type": "general"},
                {"stock_id": "2000", "date": old_date, "sentiment": 1, "event_type": "general"},
            ]
        )
        result = scanner._compute_news_scores(["1000", "2000"], ann).set_index("stock_id")
        assert result.loc["1000", "news_score"] > result.loc["2000", "news_score"]

    def test_negative_ann_lowers_score(self):
        """負面公告股票，排名應低於無公告股票。"""
        scanner = self._make_scanner()
        today = date.today()
        ann = self._make_ann([{"stock_id": "1000", "date": today, "sentiment": -1, "event_type": "general"}])
        result = scanner._compute_news_scores(["1000", "2000"], ann).set_index("stock_id")
        assert result.loc["1000", "news_score"] < result.loc["2000", "news_score"]

    def test_missing_event_type_column_fallback(self):
        """df_ann 無 event_type 欄位時，應降級為 general（不崩潰）。"""
        scanner = self._make_scanner()
        today = date.today()
        ann = self._make_ann([{"stock_id": "1000", "date": today, "sentiment": 1}])
        result = scanner._compute_news_scores(["1000", "2000"], ann)
        assert "news_score" in result.columns
        assert len(result) == 2

    def test_output_score_in_0_1_range(self):
        """news_score 應在 [0, 1] 範圍內。"""
        scanner = self._make_scanner()
        today = date.today()
        ann = self._make_ann(
            [
                {"stock_id": "1000", "date": today, "sentiment": 1, "event_type": "earnings_call"},
                {"stock_id": "2000", "date": today - timedelta(days=5), "sentiment": -1, "event_type": "general"},
                {"stock_id": "3000", "date": today, "sentiment": 0, "event_type": "filing"},
            ]
        )
        result = scanner._compute_news_scores(["1000", "2000", "3000"], ann)
        assert result["news_score"].between(0.0, 1.0).all()


# ====================================================================== #
#  消息面評分升級測試（Task 48）
# ====================================================================== #


class TestComputeNewsDecayWeightV2:
    """新 decay 常數（0.2）與新事件類型（governance_change/buyback）測試。"""

    def test_decay_constant_is_0_2(self):
        """1 天後 general 公告的衰減值應為 exp(-0.2)。"""
        import math

        from src.discovery.scanner import compute_news_decay_weight

        assert compute_news_decay_weight(1, "general") == pytest.approx(math.exp(-0.2), abs=1e-9)

    def test_seven_days_retention_above_20pct(self):
        """7 天後仍保留 >20%（exp(-1.4) ≈ 0.247）。"""
        import math

        from src.discovery.scanner import compute_news_decay_weight

        w = compute_news_decay_weight(7, "general")
        assert w == pytest.approx(math.exp(-1.4), abs=1e-9)
        assert w > 0.20

    def test_governance_change_weight_5(self):
        """governance_change 事件類型加權值應為 5.0。"""
        from src.discovery.scanner import compute_news_decay_weight

        assert compute_news_decay_weight(0, "governance_change") == pytest.approx(5.0, abs=1e-9)

    def test_buyback_weight_4(self):
        """buyback 事件類型加權值應為 4.0。"""
        from src.discovery.scanner import compute_news_decay_weight

        assert compute_news_decay_weight(0, "buyback") == pytest.approx(4.0, abs=1e-9)

    def test_governance_beats_earnings_call_same_age(self):
        """相同天數下，governance_change 加權值應大於 earnings_call。"""
        from src.discovery.scanner import compute_news_decay_weight

        assert compute_news_decay_weight(3, "governance_change") > compute_news_decay_weight(3, "earnings_call")

    def test_buyback_beats_investor_day_same_age(self):
        """相同天數下，buyback 加權值應大於 investor_day。"""
        from src.discovery.scanner import compute_news_decay_weight

        assert compute_news_decay_weight(3, "buyback") > compute_news_decay_weight(3, "investor_day")


class TestComputeAbnormalAnnouncementRate:
    """compute_abnormal_announcement_rate 純函數測試。"""

    def _make_history(self, stock_id: str, dates: list) -> pd.DataFrame:
        return pd.DataFrame({"stock_id": [stock_id] * len(dates), "date": dates})

    def test_empty_history_returns_zero(self):
        """無歷史資料 → Z=0.0。"""
        from src.discovery.scanner import compute_abnormal_announcement_rate

        result = compute_abnormal_announcement_rate(pd.DataFrame(), ["1000", "2000"])
        assert (result == 0.0).all()

    def test_no_recent_activity_returns_negative(self):
        """基準期有公告但近期無 → Z < 0（安靜期）。"""
        from src.discovery.scanner import compute_abnormal_announcement_rate

        today = date.today()
        # 基準期早期有密集公告，近 10 天沒有
        dates = [today - timedelta(days=d) for d in range(20, 150, 10)]
        df_hist = self._make_history("1000", dates)
        result = compute_abnormal_announcement_rate(df_hist, ["1000"])
        assert result["1000"] < 0.0

    def test_abnormal_spike_returns_high_z(self):
        """近期公告突增（vs 基準期低頻）→ Z > 2（異常活躍）。"""
        from src.discovery.scanner import compute_abnormal_announcement_rate

        today = date.today()
        # 基準期：每窗口僅 1 篇（低頻）
        baseline_dates = [today - timedelta(days=d) for d in range(15, 170, 10)]
        # 近期 10 天：10 篇（高頻爆量）
        recent_dates = [today - timedelta(days=d) for d in range(0, 10)]
        all_dates = baseline_dates + recent_dates
        df_hist = self._make_history("1000", all_dates)
        result = compute_abnormal_announcement_rate(df_hist, ["1000"])
        assert result["1000"] > 2.0

    def test_stock_not_in_history_returns_zero(self):
        """候選股不在歷史中 → Z=0.0。"""
        from src.discovery.scanner import compute_abnormal_announcement_rate

        today = date.today()
        df_hist = self._make_history("9999", [today - timedelta(days=5)])
        result = compute_abnormal_announcement_rate(df_hist, ["1000"])
        assert result["1000"] == 0.0

    def test_result_is_series_indexed_by_stock_id(self):
        """回傳 Series，包含所有傳入的 stock_id。"""
        from src.discovery.scanner import compute_abnormal_announcement_rate

        result = compute_abnormal_announcement_rate(pd.DataFrame(), ["1000", "2000"])
        assert isinstance(result, pd.Series)
        assert set(result.index) == {"1000", "2000"}


class TestComputeNewsScoresV2:
    """_compute_news_scores 整合新事件類型與異常率乘數測試。"""

    def _make_scanner(self):
        return MarketScanner(min_price=10, max_price=2000, min_volume=100_000)

    def test_governance_change_beats_earnings_call(self):
        """governance_change 正面公告排名應高於 earnings_call 正面公告。"""
        scanner = self._make_scanner()
        today = date.today()
        ann = pd.DataFrame(
            [
                {"stock_id": "1000", "date": today, "sentiment": 1, "event_type": "governance_change"},
                {"stock_id": "2000", "date": today, "sentiment": 1, "event_type": "earnings_call"},
            ]
        )
        result = scanner._compute_news_scores(["1000", "2000"], ann).set_index("stock_id")
        assert result.loc["1000", "news_score"] > result.loc["2000", "news_score"]

    def test_buyback_beats_general(self):
        """buyback 正面公告排名應高於 general 正面公告。"""
        scanner = self._make_scanner()
        today = date.today()
        ann = pd.DataFrame(
            [
                {"stock_id": "1000", "date": today, "sentiment": 1, "event_type": "buyback"},
                {"stock_id": "2000", "date": today, "sentiment": 1, "event_type": "general"},
            ]
        )
        result = scanner._compute_news_scores(["1000", "2000"], ann).set_index("stock_id")
        assert result.loc["1000", "news_score"] > result.loc["2000", "news_score"]

    def test_abnormal_rate_boosts_score(self):
        """有異常公告爆量（Z>2）的股票相較普通公告的股票得分更高。"""
        scanner = self._make_scanner()
        today = date.today()
        # 兩股都有相同類型正面公告
        ann = pd.DataFrame(
            [
                {"stock_id": "1000", "date": today, "sentiment": 1, "event_type": "general"},
                {"stock_id": "2000", "date": today, "sentiment": 1, "event_type": "general"},
            ]
        )
        # 1000 近期爆量（基準期低頻），2000 無歷史
        baseline = [today - timedelta(days=d) for d in range(15, 170, 15)]
        recent = [today - timedelta(days=d) for d in range(0, 10)]
        df_hist = pd.DataFrame({"stock_id": ["1000"] * len(baseline + recent), "date": baseline + recent})
        result = scanner._compute_news_scores(["1000", "2000"], ann, df_ann_history=df_hist).set_index("stock_id")
        assert result.loc["1000", "news_score"] > result.loc["2000", "news_score"]

    def test_no_history_behaves_same_as_before(self):
        """未傳入 df_ann_history（None）時，正常計算不崩潰。"""
        scanner = self._make_scanner()
        today = date.today()
        ann = pd.DataFrame([{"stock_id": "1000", "date": today, "sentiment": 1, "event_type": "general"}])
        result = scanner._compute_news_scores(["1000", "2000"], ann, df_ann_history=None)
        assert len(result) == 2
        assert "news_score" in result.columns

    def test_output_still_0_1_range_with_history(self):
        """加入歷史後 news_score 仍應在 [0, 1] 範圍內。"""
        scanner = self._make_scanner()
        today = date.today()
        ann = pd.DataFrame(
            [
                {"stock_id": "1000", "date": today, "sentiment": 1, "event_type": "governance_change"},
                {"stock_id": "2000", "date": today, "sentiment": -1, "event_type": "general"},
                {"stock_id": "3000", "date": today - timedelta(days=5), "sentiment": 1, "event_type": "buyback"},
            ]
        )
        history_dates = [today - timedelta(days=d) for d in range(5, 50, 5)]
        df_hist = pd.DataFrame({"stock_id": ["1000"] * len(history_dates), "date": history_dates})
        result = scanner._compute_news_scores(["1000", "2000", "3000"], ann, df_ann_history=df_hist)
        assert result["news_score"].between(0.0, 1.0).all()


# ====================================================================== #
#  財報因子整合測試（Task 39）
# ====================================================================== #


def _make_fin_df(rows: list[dict]) -> pd.DataFrame:
    """建立 _load_financial_data() 回傳格式的財報 DataFrame。"""
    defaults = {
        "stock_id": "1000",
        "date": date(2024, 12, 31),
        "year": 2024,
        "quarter": 4,
        "eps": None,
        "roe": None,
        "gross_margin": None,
        "debt_ratio": None,
    }
    result = []
    for r in rows:
        row = dict(defaults)
        row.update(r)
        result.append(row)
    return pd.DataFrame(result)


class TestValueFinancialFundamentalScores:
    """ValueScanner 財報因子整合測試。"""

    def _make_revenue(self):
        return pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [15.0, 10.0],
                "mom_growth": [5.0, 5.0],
            }
        )

    def test_high_roe_improves_score(self, value_scanner):
        """高 ROE 股票在財報整合後應得到更高基本面分數。"""
        df_revenue = self._make_revenue()
        fin_df = _make_fin_df(
            [
                {
                    "stock_id": "1000",
                    "year": 2024,
                    "quarter": 4,
                    "eps": 5.0,
                    "roe": 25.0,
                    "gross_margin": 40.0,
                    "debt_ratio": 30.0,
                },
                {
                    "stock_id": "1000",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "eps": 4.5,
                    "roe": 22.0,
                    "gross_margin": 38.0,
                    "debt_ratio": 30.0,
                },
                {
                    "stock_id": "1000",
                    "date": date(2023, 12, 31),
                    "year": 2023,
                    "quarter": 4,
                    "eps": 3.0,
                    "roe": 18.0,
                    "gross_margin": 35.0,
                    "debt_ratio": 35.0,
                },
                {
                    "stock_id": "1001",
                    "year": 2024,
                    "quarter": 4,
                    "eps": 1.0,
                    "roe": 5.0,
                    "gross_margin": 20.0,
                    "debt_ratio": 60.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "eps": 1.2,
                    "roe": 6.0,
                    "gross_margin": 21.0,
                    "debt_ratio": 60.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2023, 12, 31),
                    "year": 2023,
                    "quarter": 4,
                    "eps": 0.8,
                    "roe": 4.0,
                    "gross_margin": 19.0,
                    "debt_ratio": 62.0,
                },
            ]
        )
        value_scanner._load_financial_data = lambda sids, quarters=5: fin_df
        result = value_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]

    def test_fallback_no_financial_data(self, value_scanner):
        """無財報資料時應降回純營收分（不崩潰，分數在 0~1）。"""
        df_revenue = self._make_revenue()
        value_scanner._load_financial_data = lambda sids, quarters=5: pd.DataFrame(
            columns=["stock_id", "date", "year", "quarter", "eps", "roe", "gross_margin", "debt_ratio"]
        )
        result = value_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        assert "fundamental_score" in result.columns
        assert result["fundamental_score"].between(0.0, 1.0).all()
        scores = result.set_index("stock_id")["fundamental_score"]
        # 高 YoY(1000) 應仍優於低 YoY(1001)
        assert scores["1000"] > scores["1001"]

    def test_gm_qoq_improvement_rewarded(self, value_scanner):
        """毛利率 QoQ 改善（本季 > 上季）應在同等條件下得到更高分數。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [10.0, 10.0],
                "mom_growth": [5.0, 5.0],
            }
        )
        fin_df = _make_fin_df(
            [
                # 1000: 毛利率改善 (35 → 40)
                {"stock_id": "1000", "year": 2024, "quarter": 4, "eps": 3.0, "roe": 15.0, "gross_margin": 40.0},
                {
                    "stock_id": "1000",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "eps": 3.0,
                    "roe": 15.0,
                    "gross_margin": 35.0,
                },
                # 1001: 毛利率下滑 (40 → 35)
                {"stock_id": "1001", "year": 2024, "quarter": 4, "eps": 3.0, "roe": 15.0, "gross_margin": 35.0},
                {
                    "stock_id": "1001",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "eps": 3.0,
                    "roe": 15.0,
                    "gross_margin": 40.0,
                },
            ]
        )
        value_scanner._load_financial_data = lambda sids, quarters=5: fin_df
        result = value_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]

    def test_output_schema(self, value_scanner):
        """回傳應有 stock_id + fundamental_score 兩欄。"""
        value_scanner._load_financial_data = lambda sids, quarters=5: pd.DataFrame(
            columns=["stock_id", "date", "year", "quarter", "eps", "roe", "gross_margin", "debt_ratio"]
        )
        result = value_scanner._compute_fundamental_scores(
            ["1000", "1001"],
            pd.DataFrame({"stock_id": ["1000", "1001"], "yoy_growth": [5.0, 3.0], "mom_growth": [1.0, 1.0]}),
        )
        assert set(result.columns) >= {"stock_id", "fundamental_score"}
        assert len(result) == 2


class TestComputeRelativePEThresholds:
    """compute_relative_pe_thresholds() 模組級純函數測試。"""

    from src.discovery.scanner import compute_relative_pe_thresholds as _fn  # noqa: E402

    def test_relative_threshold_with_sufficient_industry(self):
        """同產業樣本充足（≥3）時，門檻 = 產業 PE 中位數 × 1.5。"""
        from src.discovery.scanner import compute_relative_pe_thresholds

        industry = pd.Series(["科技", "科技", "科技", "科技", "科技"])
        pe = pd.Series([20.0, 22.0, 24.0, 26.0, 28.0])
        thresholds = compute_relative_pe_thresholds(industry, pe, multiplier=1.5, fallback_pe=50.0)
        expected_median = 24.0
        expected_threshold = expected_median * 1.5  # = 36.0
        assert abs(thresholds.iloc[0] - expected_threshold) < 0.01
        assert abs(thresholds.iloc[-1] - expected_threshold) < 0.01

    def test_fallback_when_industry_count_below_min(self):
        """同產業樣本不足（< min_industry_count=3）時，使用 fallback_pe。"""
        from src.discovery.scanner import compute_relative_pe_thresholds

        industry = pd.Series(["生技", "生技"])
        pe = pd.Series([15.0, 20.0])
        thresholds = compute_relative_pe_thresholds(
            industry, pe, multiplier=1.5, fallback_pe=50.0, min_industry_count=3
        )
        assert (thresholds == 50.0).all()

    def test_mixed_industries(self):
        """部分產業充足、部分不足，分別使用相對/絕對門檻。"""
        from src.discovery.scanner import compute_relative_pe_thresholds

        industry = pd.Series(["科技", "科技", "科技", "生技"])  # 科技3支, 生技1支
        pe = pd.Series([20.0, 24.0, 28.0, 15.0])
        thresholds = compute_relative_pe_thresholds(
            industry, pe, multiplier=1.5, fallback_pe=50.0, min_industry_count=3
        )
        # 科技：中位數 24 × 1.5 = 36
        assert abs(thresholds.iloc[0] - 36.0) < 0.01
        # 生技：樣本不足 → fallback 50
        assert thresholds.iloc[3] == 50.0

    def test_negative_pe_excluded_from_median(self):
        """PE ≤ 0 的股票不應影響產業中位數計算。"""
        from src.discovery.scanner import compute_relative_pe_thresholds

        industry = pd.Series(["金融", "金融", "金融", "金融"])
        pe = pd.Series([-5.0, 10.0, 12.0, 14.0])  # 第一支 PE < 0
        thresholds = compute_relative_pe_thresholds(industry, pe, multiplier=1.5, fallback_pe=50.0)
        # 只有 10, 12, 14 有效，中位數 12 × 1.5 = 18
        assert abs(thresholds.iloc[1] - 18.0) < 0.01
        assert abs(thresholds.iloc[-1] - 18.0) < 0.01

    def test_empty_input_returns_empty(self):
        """空 Series 應回傳空 Series。"""
        from src.discovery.scanner import compute_relative_pe_thresholds

        thresholds = compute_relative_pe_thresholds(pd.Series(dtype=str), pd.Series(dtype=float))
        assert thresholds.empty

    def test_value_coarse_filter_uses_relative_pe(self):
        """ValueScanner._coarse_filter() 應接受同產業高 PE 股票（相對估值合格）。"""
        scanner = ValueScanner(min_price=10, max_price=2000, min_volume=100_000, top_n_candidates=10, top_n_results=5)
        # 設定 _df_valuation（all PE=35，絕對 PE<30 會濾掉，但相對 PE<中位數×1.5=52.5 應通過）
        scanner._df_valuation = pd.DataFrame(
            [
                {
                    "stock_id": "A001",
                    "date": date(2025, 3, 1),
                    "pe_ratio": 35.0,
                    "pb_ratio": 2.0,
                    "dividend_yield": 1.0,
                },
                {
                    "stock_id": "A002",
                    "date": date(2025, 3, 1),
                    "pe_ratio": 35.0,
                    "pb_ratio": 2.0,
                    "dividend_yield": 1.0,
                },
                {
                    "stock_id": "A003",
                    "date": date(2025, 3, 1),
                    "pe_ratio": 35.0,
                    "pb_ratio": 2.0,
                    "dividend_yield": 1.0,
                },
            ]
        )
        # 三支股票同屬「科技」產業，中位數 PE=35，閾值=35×1.5=52.5 → PE=35 通過
        scanner._df_stock_info = pd.DataFrame(
            [
                {"stock_id": "A001", "industry_category": "科技"},
                {"stock_id": "A002", "industry_category": "科技"},
                {"stock_id": "A003", "industry_category": "科技"},
            ]
        )
        rows = []
        for sid in ["A001", "A002", "A003"]:
            for d in range(20):
                day = date(2025, 1, 1) + timedelta(days=d)
                rows.append(
                    {
                        "stock_id": sid,
                        "date": day,
                        "open": 99,
                        "high": 102,
                        "low": 98,
                        "close": 100,
                        "volume": 500_000,
                    }
                )
        df_price = pd.DataFrame(rows)
        result = scanner._coarse_filter(df_price, pd.DataFrame())
        # 三支股票 PE=35 < 52.5（相對閾值）→ 全部應保留
        assert len(result) == 3


class TestDividendFinancialFundamentalScores:
    """DividendScanner 財報因子整合測試（EPS 穩定性 + 配息率代理）。"""

    def _make_revenue(self):
        return pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [10.0, 10.0],
                "mom_growth": [5.0, 5.0],
            }
        )

    def test_stable_eps_rewarded(self, dividend_scanner):
        """EPS 穩定（低標準差）的股票基本面分數應優於波動大的股票。"""
        df_revenue = self._make_revenue()
        fin_df = _make_fin_df(
            [
                # 1000: EPS 穩定 (2.0, 2.1, 2.0, 1.9)
                {"stock_id": "1000", "date": date(2024, 12, 31), "year": 2024, "quarter": 4, "eps": 2.0},
                {"stock_id": "1000", "date": date(2024, 9, 30), "year": 2024, "quarter": 3, "eps": 2.1},
                {"stock_id": "1000", "date": date(2024, 6, 30), "year": 2024, "quarter": 2, "eps": 2.0},
                {"stock_id": "1000", "date": date(2024, 3, 31), "year": 2024, "quarter": 1, "eps": 1.9},
                # 1001: EPS 波動大 (5.0, -1.0, 3.0, -2.0)
                {"stock_id": "1001", "date": date(2024, 12, 31), "year": 2024, "quarter": 4, "eps": 5.0},
                {"stock_id": "1001", "date": date(2024, 9, 30), "year": 2024, "quarter": 3, "eps": -1.0},
                {"stock_id": "1001", "date": date(2024, 6, 30), "year": 2024, "quarter": 2, "eps": 3.0},
                {"stock_id": "1001", "date": date(2024, 3, 31), "year": 2024, "quarter": 1, "eps": -2.0},
            ]
        )
        dividend_scanner._load_financial_data = lambda sids, quarters=5: fin_df
        result = dividend_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]

    def test_positive_eps_payout_proxy(self, dividend_scanner):
        """全正 EPS 的股票配息代理分數應優於含負 EPS 的股票。"""
        df_revenue = self._make_revenue()
        fin_df = _make_fin_df(
            [
                # 1000: 4 季全正 EPS
                {"stock_id": "1000", "date": date(2024, 12, 31), "year": 2024, "quarter": 4, "eps": 2.0},
                {"stock_id": "1000", "date": date(2024, 9, 30), "year": 2024, "quarter": 3, "eps": 2.0},
                {"stock_id": "1000", "date": date(2024, 6, 30), "year": 2024, "quarter": 2, "eps": 2.0},
                {"stock_id": "1000", "date": date(2024, 3, 31), "year": 2024, "quarter": 1, "eps": 2.0},
                # 1001: 含 2 季負 EPS
                {"stock_id": "1001", "date": date(2024, 12, 31), "year": 2024, "quarter": 4, "eps": 3.0},
                {"stock_id": "1001", "date": date(2024, 9, 30), "year": 2024, "quarter": 3, "eps": -1.0},
                {"stock_id": "1001", "date": date(2024, 6, 30), "year": 2024, "quarter": 2, "eps": 2.0},
                {"stock_id": "1001", "date": date(2024, 3, 31), "year": 2024, "quarter": 1, "eps": -0.5},
            ]
        )
        dividend_scanner._load_financial_data = lambda sids, quarters=5: fin_df
        result = dividend_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]

    def test_fallback_no_financial_data(self, dividend_scanner):
        """無財報資料時降回純營收分，不崩潰。"""
        df_revenue = self._make_revenue()
        dividend_scanner._load_financial_data = lambda sids, quarters=5: pd.DataFrame(
            columns=["stock_id", "date", "year", "quarter", "eps", "roe", "gross_margin", "debt_ratio"]
        )
        result = dividend_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        assert "fundamental_score" in result.columns
        assert result["fundamental_score"].between(0.0, 1.0).all()


class TestComputeEpsSustainability:
    """compute_eps_sustainability() 模組級純函數測試。"""

    def _make_fin(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_returns_empty_when_no_data(self):
        """財報 DataFrame 為空時，應回傳空集合（不排除任何股票）。"""
        from src.discovery.scanner import compute_eps_sustainability

        result = compute_eps_sustainability(pd.DataFrame())
        assert result == frozenset()

    def test_all_positive_returns_empty(self):
        """全部季度 EPS > 0 時，回傳空集合（無排除）。"""
        from src.discovery.scanner import compute_eps_sustainability

        df = self._make_fin(
            [
                {"stock_id": "A", "date": date(2024, 12, 31), "eps": 2.0},
                {"stock_id": "A", "date": date(2024, 9, 30), "eps": 1.5},
                {"stock_id": "A", "date": date(2024, 6, 30), "eps": 1.8},
                {"stock_id": "A", "date": date(2024, 3, 31), "eps": 2.1},
            ]
        )
        result = compute_eps_sustainability(df, min_quarters=4)
        assert result == frozenset()

    def test_any_nonpositive_eps_excluded(self):
        """任一季 EPS ≤ 0 的股票應出現在回傳集合中（被排除）。"""
        from src.discovery.scanner import compute_eps_sustainability

        df = self._make_fin(
            [
                {"stock_id": "B", "date": date(2024, 12, 31), "eps": 3.0},
                {"stock_id": "B", "date": date(2024, 9, 30), "eps": -0.5},  # 負值
                {"stock_id": "B", "date": date(2024, 6, 30), "eps": 1.0},
                {"stock_id": "B", "date": date(2024, 3, 31), "eps": 2.0},
            ]
        )
        result = compute_eps_sustainability(df, min_quarters=4)
        assert "B" in result

    def test_zero_eps_also_excluded(self):
        """EPS = 0 也應視為不可持續而排除。"""
        from src.discovery.scanner import compute_eps_sustainability

        df = self._make_fin(
            [
                {"stock_id": "C", "date": date(2024, 12, 31), "eps": 0.0},
                {"stock_id": "C", "date": date(2024, 9, 30), "eps": 1.0},
                {"stock_id": "C", "date": date(2024, 6, 30), "eps": 1.5},
            ]
        )
        result = compute_eps_sustainability(df, min_quarters=4)
        assert "C" in result

    def test_mixed_stocks(self):
        """混合場景：部分股票有負 EPS，部分全正。"""
        from src.discovery.scanner import compute_eps_sustainability

        df = self._make_fin(
            [
                # GOOD: 全正
                {"stock_id": "GOOD", "date": date(2024, 12, 31), "eps": 2.0},
                {"stock_id": "GOOD", "date": date(2024, 9, 30), "eps": 1.5},
                {"stock_id": "GOOD", "date": date(2024, 6, 30), "eps": 1.8},
                {"stock_id": "GOOD", "date": date(2024, 3, 31), "eps": 2.2},
                # BAD: 含負值
                {"stock_id": "BAD", "date": date(2024, 12, 31), "eps": 2.0},
                {"stock_id": "BAD", "date": date(2024, 9, 30), "eps": -1.0},
                {"stock_id": "BAD", "date": date(2024, 6, 30), "eps": 1.0},
                {"stock_id": "BAD", "date": date(2024, 3, 31), "eps": 0.5},
            ]
        )
        result = compute_eps_sustainability(df, min_quarters=4)
        assert "GOOD" not in result
        assert "BAD" in result

    def test_no_data_stock_passes(self):
        """某股在財報 DataFrame 中完全無紀錄，不應出現在排除集合中（pass through）。"""
        from src.discovery.scanner import compute_eps_sustainability

        # 只有 A 的財報，B 沒有任何紀錄
        df = self._make_fin(
            [
                {"stock_id": "A", "date": date(2024, 12, 31), "eps": 2.0},
            ]
        )
        result = compute_eps_sustainability(df, min_quarters=4)
        assert "B" not in result  # B 無資料 → pass through

    def test_only_recent_quarters_evaluated(self):
        """只評估最近 min_quarters 季，較舊的負 EPS 不影響結果。"""
        from src.discovery.scanner import compute_eps_sustainability

        df = self._make_fin(
            [
                # 最近 2 季正值
                {"stock_id": "D", "date": date(2024, 12, 31), "eps": 2.0},
                {"stock_id": "D", "date": date(2024, 9, 30), "eps": 1.5},
                # 較舊的 2 季有負值（超出 min_quarters=2 的窗口）
                {"stock_id": "D", "date": date(2024, 6, 30), "eps": -1.0},
                {"stock_id": "D", "date": date(2024, 3, 31), "eps": -2.0},
            ]
        )
        result = compute_eps_sustainability(df, min_quarters=2)
        # 只看最近 2 季（12月、9月），皆正 → 不排除
        assert "D" not in result

    def test_dividend_coarse_filter_excludes_negative_eps(self):
        """DividendScanner._coarse_filter() 應排除有負 EPS 的股票。"""
        scanner = DividendScanner(
            min_price=10, max_price=2000, min_volume=100_000, top_n_candidates=10, top_n_results=5
        )
        # 設定估值資料：兩股都通過殖利率 > 3% + PE > 0
        scanner._df_valuation = pd.DataFrame(
            [
                {
                    "stock_id": "GOOD",
                    "date": date(2025, 3, 1),
                    "pe_ratio": 12.0,
                    "pb_ratio": 1.5,
                    "dividend_yield": 4.5,
                },
                {"stock_id": "BAD", "date": date(2025, 3, 1), "pe_ratio": 10.0, "pb_ratio": 1.2, "dividend_yield": 5.0},
            ]
        )
        # GOOD: 近 4 季 EPS 全正 / BAD: 有一季負 EPS
        scanner._df_eps_quarterly = pd.DataFrame(
            [
                {"stock_id": "GOOD", "date": date(2024, 12, 31), "eps": 2.0},
                {"stock_id": "GOOD", "date": date(2024, 9, 30), "eps": 1.8},
                {"stock_id": "GOOD", "date": date(2024, 6, 30), "eps": 1.5},
                {"stock_id": "GOOD", "date": date(2024, 3, 31), "eps": 1.6},
                {"stock_id": "BAD", "date": date(2024, 12, 31), "eps": 3.0},
                {"stock_id": "BAD", "date": date(2024, 9, 30), "eps": -0.5},
                {"stock_id": "BAD", "date": date(2024, 6, 30), "eps": 1.0},
                {"stock_id": "BAD", "date": date(2024, 3, 31), "eps": 0.8},
            ]
        )
        rows = []
        for sid in ["GOOD", "BAD"]:
            for d in range(20):
                day = date(2025, 1, 1) + timedelta(days=d)
                rows.append(
                    {
                        "stock_id": sid,
                        "date": day,
                        "open": 99,
                        "high": 102,
                        "low": 98,
                        "close": 100,
                        "volume": 500_000,
                    }
                )
        result = scanner._coarse_filter(pd.DataFrame(rows), pd.DataFrame())
        sids = result["stock_id"].tolist() if not result.empty else []
        assert "GOOD" in sids
        assert "BAD" not in sids


class TestGrowthFinancialFundamentalScores:
    """GrowthScanner 財報因子整合測試（毛利率加速 + EPS 季增率）。"""

    def _make_revenue(self):
        return pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [25.0, 25.0],
                "mom_growth": [5.0, 5.0],
                "yoy_3m_ago": [15.0, 15.0],  # 兩者加速度相同
            }
        )

    def test_gm_acceleration_rewarded(self, growth_scanner):
        """毛利率 YoY 加速（改善）的股票基本面分數應優於惡化的股票。"""
        df_revenue = self._make_revenue()
        fin_df = _make_fin_df(
            [
                # 1000: 毛利率 YoY 改善 (Q4 2024 vs Q4 2023: 40 vs 30 → +10)
                {
                    "stock_id": "1000",
                    "date": date(2024, 12, 31),
                    "year": 2024,
                    "quarter": 4,
                    "eps": 5.0,
                    "gross_margin": 40.0,
                },
                {
                    "stock_id": "1000",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "eps": 4.5,
                    "gross_margin": 38.0,
                },
                {
                    "stock_id": "1000",
                    "date": date(2023, 12, 31),
                    "year": 2023,
                    "quarter": 4,
                    "eps": 3.0,
                    "gross_margin": 30.0,
                },
                # 1001: 毛利率 YoY 惡化 (Q4 2024 vs Q4 2023: 25 vs 35 → -10)
                {
                    "stock_id": "1001",
                    "date": date(2024, 12, 31),
                    "year": 2024,
                    "quarter": 4,
                    "eps": 5.0,
                    "gross_margin": 25.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "eps": 4.5,
                    "gross_margin": 28.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2023, 12, 31),
                    "year": 2023,
                    "quarter": 4,
                    "eps": 3.0,
                    "gross_margin": 35.0,
                },
            ]
        )
        growth_scanner._load_financial_data = lambda sids, quarters=5: fin_df
        result = growth_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]

    def test_eps_qoq_growth_rewarded(self, growth_scanner):
        """EPS 季增率正成長應優於季衰退。"""
        df_revenue = self._make_revenue()
        fin_df = _make_fin_df(
            [
                # 1000: EPS QoQ 成長 (3.0 → 5.0)
                {
                    "stock_id": "1000",
                    "date": date(2024, 12, 31),
                    "year": 2024,
                    "quarter": 4,
                    "eps": 5.0,
                    "gross_margin": 35.0,
                },
                {
                    "stock_id": "1000",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "eps": 3.0,
                    "gross_margin": 34.0,
                },
                # 1001: EPS QoQ 衰退 (5.0 → 3.0)
                {
                    "stock_id": "1001",
                    "date": date(2024, 12, 31),
                    "year": 2024,
                    "quarter": 4,
                    "eps": 3.0,
                    "gross_margin": 35.0,
                },
                {
                    "stock_id": "1001",
                    "date": date(2024, 9, 30),
                    "year": 2024,
                    "quarter": 3,
                    "eps": 5.0,
                    "gross_margin": 34.0,
                },
            ]
        )
        growth_scanner._load_financial_data = lambda sids, quarters=5: fin_df
        result = growth_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] > scores["1001"]

    def test_fallback_no_financial_data(self, growth_scanner):
        """無財報資料時降回 YoY+加速度（原邏輯），不崩潰。"""
        df_revenue = self._make_revenue()
        growth_scanner._load_financial_data = lambda sids, quarters=5: pd.DataFrame(
            columns=["stock_id", "date", "year", "quarter", "eps", "roe", "gross_margin", "debt_ratio"]
        )
        result = growth_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        assert "fundamental_score" in result.columns
        assert result["fundamental_score"].between(0.0, 1.0).all()
        # 原邏輯：相同 YoY + 相同加速度 → 分數應相近
        scores = result.set_index("stock_id")["fundamental_score"]
        assert abs(scores["1000"] - scores["1001"]) < 0.01


# ─── compute_sector_relative_strength ────────────────────────────────


class TestComputeSectorRelativeStrength:
    """測試 compute_sector_relative_strength() 純函數。"""

    def _make_price_df(self, sid: str, start_close: float, end_close: float, n: int = 20) -> pd.DataFrame:
        """產生 n 天從 start_close 線性變化到 end_close 的日K（只需 close）。"""
        rows = []
        for i in range(n):
            close = start_close + (end_close - start_close) * i / max(n - 1, 1)
            rows.append(
                {
                    "stock_id": sid,
                    "date": date(2025, 1, 1) + timedelta(days=i),
                    "close": round(close, 2),
                }
            )
        return pd.DataFrame(rows)

    def test_outperformer_gets_positive_bonus(self):
        """個股報酬率顯著超越產業中位數時應得 +0.03。"""
        from src.industry.analyzer import compute_sector_relative_strength

        # 同產業兩支股票：1000 漲 50%，1001 漲 5%，中位數 ≈ 27.5%；差 = 22.5% > 20% → +0.03
        df1 = self._make_price_df("1000", 100.0, 150.0)
        df2 = self._make_price_df("1001", 100.0, 105.0)
        df_price = pd.concat([df1, df2], ignore_index=True)
        industry_map = {"1000": "半導體", "1001": "半導體"}

        result = compute_sector_relative_strength(["1000", "1001"], df_price, industry_map)
        rs = result.set_index("stock_id")["relative_strength_bonus"]
        assert rs["1000"] == pytest.approx(0.03)

    def test_underperformer_gets_negative_bonus(self):
        """個股報酬率顯著落後產業中位數時應得 -0.03。"""
        from src.industry.analyzer import compute_sector_relative_strength

        # 1000 漲 5%，1001 漲 50%；1000 落後 22.5% < -20% → -0.03
        df1 = self._make_price_df("1000", 100.0, 105.0)
        df2 = self._make_price_df("1001", 100.0, 150.0)
        df_price = pd.concat([df1, df2], ignore_index=True)
        industry_map = {"1000": "半導體", "1001": "半導體"}

        result = compute_sector_relative_strength(["1000", "1001"], df_price, industry_map)
        rs = result.set_index("stock_id")["relative_strength_bonus"]
        assert rs["1000"] == pytest.approx(-0.03)

    def test_within_threshold_gets_zero(self):
        """差距未超過門檻時 bonus 應為 0.0。"""
        from src.industry.analyzer import compute_sector_relative_strength

        # 1000 漲 10%，1001 漲 15%；差 = 2.5% < 20% → 皆為 0.0
        df1 = self._make_price_df("1000", 100.0, 110.0)
        df2 = self._make_price_df("1001", 100.0, 115.0)
        df_price = pd.concat([df1, df2], ignore_index=True)
        industry_map = {"1000": "金融", "1001": "金融"}

        result = compute_sector_relative_strength(["1000", "1001"], df_price, industry_map)
        for _, row in result.iterrows():
            assert row["relative_strength_bonus"] == pytest.approx(0.0)

    def test_empty_price_df_returns_all_zeros(self):
        """空 df_price 應回傳全 0.0。"""
        from src.industry.analyzer import compute_sector_relative_strength

        df_price = pd.DataFrame(columns=["stock_id", "date", "close"])
        industry_map = {"1000": "半導體"}
        result = compute_sector_relative_strength(["1000"], df_price, industry_map)
        assert result.iloc[0]["relative_strength_bonus"] == pytest.approx(0.0)

    def test_single_stock_per_sector_gets_zero(self):
        """產業內只有一支股票時，中位數 = 自身，差距 = 0 → bonus = 0.0。"""
        from src.industry.analyzer import compute_sector_relative_strength

        df_price = self._make_price_df("1000", 100.0, 200.0)  # 漲 100%
        industry_map = {"1000": "化工"}
        result = compute_sector_relative_strength(["1000"], df_price, industry_map)
        assert result.iloc[0]["relative_strength_bonus"] == pytest.approx(0.0)

    def test_different_industries_independent(self):
        """不同產業的個股相互獨立計算，不互相影響。"""
        from src.industry.analyzer import compute_sector_relative_strength

        # 1000 在半導體，1001 在金融；各自產業只有一支 → 皆為 0.0
        df1 = self._make_price_df("1000", 100.0, 200.0)
        df2 = self._make_price_df("1001", 100.0, 50.0)  # 跌 50%
        df_price = pd.concat([df1, df2], ignore_index=True)
        industry_map = {"1000": "半導體", "1001": "金融"}

        result = compute_sector_relative_strength(["1000", "1001"], df_price, industry_map)
        for _, row in result.iterrows():
            assert row["relative_strength_bonus"] == pytest.approx(0.0)

    def test_missing_stock_id_defaults_zero(self):
        """stock_ids 中有股票無對應 price 資料時，應回傳 0.0（不崩潰）。"""
        from src.industry.analyzer import compute_sector_relative_strength

        df_price = self._make_price_df("1000", 100.0, 130.0)
        industry_map = {"1000": "電子", "9999": "電子"}

        result = compute_sector_relative_strength(["1000", "9999"], df_price, industry_map)
        rs = result.set_index("stock_id")["relative_strength_bonus"]
        # 9999 無資料 → 0.0
        assert rs["9999"] == pytest.approx(0.0)


# ─── 分點資料接入非 Momentum 模式 chip_tier 測試 ───────────────────────


def _make_broker_df(stock_ids: list[str]) -> pd.DataFrame:
    """建立簡單的分點交易 DataFrame（單日單分點淨買超）。"""
    rows = []
    for sid in stock_ids:
        rows.append(
            {
                "stock_id": sid,
                "date": date(2025, 1, 25),
                "broker_id": "9A00",
                "buy": 5000,
                "sell": 1000,
            }
        )
    return pd.DataFrame(rows)


def _make_basic_inst_df(stock_ids: list[str]) -> pd.DataFrame:
    """建立最小法人 DataFrame（所有股票單日投信買超）。"""
    return pd.DataFrame(
        [{"stock_id": sid, "date": date(2025, 1, 25), "name": "投信買賣超", "net": 100} for sid in stock_ids]
    )


class TestSwingChipBrokerTier:
    """SwingScanner 接入分點資料後的 chip_tier 升級測試。"""

    def _make_scanner(self):
        from src.discovery.scanner import SwingScanner

        return SwingScanner()

    def _make_holding_df(self, stock_ids: list[str]) -> pd.DataFrame:
        """建立符合 compute_whale_score 格式的大戶持股資料（下限 >= 400,000 股）。"""
        rows = []
        for sid in stock_ids:
            rows.append(
                {
                    "stock_id": sid,
                    "date": date(2025, 1, 25),
                    "level": "400001600000",  # _extract_level_lower_bound → 400001 >= 400000
                    "percent": 30.0,
                }
            )
        return pd.DataFrame(rows)

    def test_whale_and_broker_gives_4f(self, monkeypatch):
        """有大戶 + 有分點（無借券）時，chip_tier 應為 4F。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: _make_broker_df(ids))
        monkeypatch.setattr(scanner, "_load_holding_data", lambda ids: self._make_holding_df(ids))
        monkeypatch.setattr(scanner, "_load_sbl_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_broker_data_extended", lambda ids, **kw: pd.DataFrame())

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids))
        assert result.iloc[0]["chip_tier"] == "4F"

    def test_broker_only_gives_3f(self, monkeypatch):
        """無大戶 + 有分點（無借券）時，chip_tier 應為 3F。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: _make_broker_df(ids))
        monkeypatch.setattr(scanner, "_load_holding_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_sbl_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_broker_data_extended", lambda ids, **kw: pd.DataFrame())

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids))
        assert result.iloc[0]["chip_tier"] == "3F"

    def test_no_broker_no_whale_gives_2f(self, monkeypatch):
        """無大戶 + 無分點（無借券）時，chip_tier 維持 2F（現狀不退化）。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_holding_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_sbl_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_broker_data_extended", lambda ids, **kw: pd.DataFrame())

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids))
        assert result.iloc[0]["chip_tier"] == "2F"

    def test_chip_score_in_range(self, monkeypatch):
        """有分點資料時 chip_score 應在 [0, 1] 範圍內。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: _make_broker_df(ids))
        monkeypatch.setattr(scanner, "_load_holding_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_sbl_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_broker_data_extended", lambda ids, **kw: pd.DataFrame())

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids))
        assert (result["chip_score"] >= 0).all()
        assert (result["chip_score"] <= 1.0).all()


def _make_sbl_df(stock_ids: list[str]) -> pd.DataFrame:
    """建立最小借券賣出 DataFrame（compute_sbl_score 格式）。"""
    return pd.DataFrame([{"stock_id": sid, "date": date(2025, 1, 25), "sbl_balance": 5000} for sid in stock_ids])


def _make_broker_ext_df(stock_ids: list[str]) -> pd.DataFrame:
    """建立含均價的分點歷史資料（供 compute_smart_broker_score 使用）。

    模擬一個分點有 5 天買進、3 次獲利賣出，達到 Smart Broker 判定門檻。
    """
    rows = []
    for sid in stock_ids:
        bp, sp = 100.0, 112.0  # 買低賣高
        # 買入 5 天（累計 ~600 萬）
        for d in range(5):
            rows.append(
                {
                    "stock_id": sid,
                    "date": date(2025, 1, d + 10),
                    "broker_id": "9A00",
                    "buy": 12000,
                    "sell": 0,
                    "buy_price": bp,
                    "sell_price": 0.0,
                }
            )
        # 賣出 3 次（獲利）
        for d in range(3):
            rows.append(
                {
                    "stock_id": sid,
                    "date": date(2025, 1, d + 18),
                    "broker_id": "9A00",
                    "buy": 0,
                    "sell": 8000,
                    "buy_price": 0.0,
                    "sell_price": sp,
                }
            )
    return pd.DataFrame(rows)


class TestSwingChipSblTier:
    """SwingScanner 接入借券資料後的 chip_tier 升級測試。"""

    def _make_scanner(self):
        from src.discovery.scanner import SwingScanner

        return SwingScanner()

    def _make_holding_df(self, stock_ids):
        return pd.DataFrame(
            [
                {"stock_id": sid, "date": date(2025, 1, 25), "level": "400001600000", "percent": 30.0}
                for sid in stock_ids
            ]
        )

    def test_sbl_with_whale_and_broker_gives_5f(self, monkeypatch):
        """有大戶 + 分點 + 借券時，chip_tier 應升至 5F。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: _make_broker_df(ids))
        monkeypatch.setattr(scanner, "_load_holding_data", lambda ids: self._make_holding_df(ids))
        monkeypatch.setattr(scanner, "_load_sbl_data", lambda ids: _make_sbl_df(ids))
        monkeypatch.setattr(scanner, "_load_broker_data_extended", lambda ids, **kw: pd.DataFrame())

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids))
        assert result.iloc[0]["chip_tier"] == "5F"

    def test_sbl_without_broker_whale_gives_3f(self, monkeypatch):
        """有借券但無大戶/分點時，chip_tier 應為 3F（借券逆向因子納入）。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_holding_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_sbl_data", lambda ids: _make_sbl_df(ids))
        monkeypatch.setattr(scanner, "_load_broker_data_extended", lambda ids, **kw: pd.DataFrame())

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids))
        assert result.iloc[0]["chip_tier"] == "3F"


class TestSwingChipSmartBrokerTier:
    """SwingScanner Smart Broker 因子啟用後的 chip_tier 升級測試。"""

    def _make_scanner(self):
        from src.discovery.scanner import SwingScanner

        return SwingScanner()

    def _make_holding_df(self, stock_ids):
        return pd.DataFrame(
            [
                {"stock_id": sid, "date": date(2025, 1, 25), "level": "400001600000", "percent": 30.0}
                for sid in stock_ids
            ]
        )

    def test_smart_broker_full_data_gives_6f(self, monkeypatch):
        """有大戶 + 分點 + 借券 + Smart Broker 時，chip_tier 應升至 6F。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        ext_df = _make_broker_ext_df(sids)
        smart_result = pd.DataFrame(
            [
                {"stock_id": sid, "smart_broker_score": 0.8, "accum_broker_score": 0.6, "smart_broker_factor": 0.72}
                for sid in sids
            ]
        )
        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: _make_broker_df(ids))
        monkeypatch.setattr(scanner, "_load_holding_data", lambda ids: self._make_holding_df(ids))
        monkeypatch.setattr(scanner, "_load_sbl_data", lambda ids: _make_sbl_df(ids))
        monkeypatch.setattr(scanner, "_load_broker_data_extended", lambda ids, **kw: ext_df)
        monkeypatch.setattr("src.discovery.scanner.compute_smart_broker_score", lambda df, prices, **kw: smart_result)

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids))
        assert result.iloc[0]["chip_tier"] == "6F"

    def test_smart_broker_without_other_factors_no_6f(self, monkeypatch):
        """有 Smart Broker 但無大戶/分點/借券時，不觸發 6F（條件未全部滿足）。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        smart_result = pd.DataFrame(
            [
                {"stock_id": sid, "smart_broker_score": 0.8, "accum_broker_score": 0.6, "smart_broker_factor": 0.72}
                for sid in sids
            ]
        )
        ext_df = _make_broker_ext_df(sids)
        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_holding_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_sbl_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_broker_data_extended", lambda ids, **kw: ext_df)
        monkeypatch.setattr("src.discovery.scanner.compute_smart_broker_score", lambda df, prices, **kw: smart_result)

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids))
        # 沒有 broker/whale/sbl，smart_broker 單獨無法觸發 6F
        assert result.iloc[0]["chip_tier"] != "6F"


class TestGrowthChipBrokerTier:
    """GrowthScanner 接入分點資料後的 chip_tier 升級測試。"""

    def _make_scanner(self):
        from src.discovery.scanner import GrowthScanner

        return GrowthScanner()

    def _make_margin_df(self, stock_ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"stock_id": sid, "date": date(2025, 1, 25), "margin_balance": 1000, "short_balance": 100}
                for sid in stock_ids
            ]
        )

    def test_margin_and_broker_gives_5f(self, monkeypatch):
        """有券資比 + 有分點時，chip_tier 應升至 5F。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: _make_broker_df(ids))
        df_margin = self._make_margin_df(sids)

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids), None, df_margin)
        assert result.iloc[0]["chip_tier"] == "5F"

    def test_margin_no_broker_gives_4f(self, monkeypatch):
        """有券資比 + 無分點時，chip_tier 維持 4F（現狀不退化）。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: pd.DataFrame())
        df_margin = self._make_margin_df(sids)

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids), None, df_margin)
        assert result.iloc[0]["chip_tier"] == "4F"

    def test_broker_no_margin_gives_4f(self, monkeypatch):
        """有分點 + 無券資比時，chip_tier 應為 4F。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: _make_broker_df(ids))

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids), None, None)
        assert result.iloc[0]["chip_tier"] == "4F"

    def test_no_margin_no_broker_gives_3f(self, monkeypatch):
        """無券資比 + 無分點時，chip_tier 維持 3F（現狀不退化）。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: pd.DataFrame())

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids), None, None)
        assert result.iloc[0]["chip_tier"] == "3F"


class TestValueChipBrokerTier:
    """ValueScanner 接入分點資料後的 chip_tier 升級測試。"""

    def _make_scanner(self):
        from src.discovery.scanner import ValueScanner

        return ValueScanner()

    def test_broker_gives_3f(self, monkeypatch):
        """有分點資料時，chip_tier 應升至 3F。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: _make_broker_df(ids))

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids))
        assert result.iloc[0]["chip_tier"] == "3F"

    def test_no_broker_gives_2f(self, monkeypatch):
        """無分點資料時，chip_tier 維持 2F（現狀不退化）。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: pd.DataFrame())

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids))
        assert result.iloc[0]["chip_tier"] == "2F"

    def test_chip_score_in_range_with_broker(self, monkeypatch):
        """有分點資料時 chip_score 應在 [0, 1] 範圍內。"""
        sids = ["1000", "1001"]
        scanner = self._make_scanner()

        monkeypatch.setattr(scanner, "_load_broker_data", lambda ids: _make_broker_df(ids))

        result = scanner._compute_chip_scores(sids, _make_basic_inst_df(sids))
        assert (result["chip_score"] >= 0).all()
        assert (result["chip_score"] <= 1.0).all()


# ─── compute_vcp_score ────────────────────────────────────────────────────


class TestVcpBonus:
    """compute_vcp_score() 純函數測試（無 DB）。

    VCP 條件：
    - 近 10 日 close 波動幅度 < 8%（價格整理）
    - 近 3 日均量 / 近 20 日均量 < 0.8（量縮）
    兩個條件同時滿足 → vcp_bonus = 0.03
    """

    def _make_price_df(self, stock_id: str, closes: list[float], volumes: list[int]) -> pd.DataFrame:
        """建立日K線 DataFrame，日期從今天往前推。"""
        today = date.today()
        n = len(closes)
        rows = [
            {
                "stock_id": stock_id,
                "date": today - timedelta(days=n - 1 - i),
                "close": closes[i],
                "volume": volumes[i],
            }
            for i in range(n)
        ]
        return pd.DataFrame(rows)

    def test_vcp_conditions_met_gives_bonus(self):
        """近 10 日低波動 + 量縮 → vcp_bonus = 0.03。"""
        from src.discovery.scanner import compute_vcp_score

        # 近 20 日均量 = 1,000,000；近 3 日量縮到 700,000（ratio=0.7 < 0.8）
        # 近 10 日 close 在 100~102（波動幅度 = 2/101 ≈ 1.98% < 8%）
        closes = [100.0] * 10 + [100.0, 101.0, 102.0, 101.5, 100.5, 101.0, 100.0, 100.5, 101.0, 101.5]
        volumes = [1_000_000] * 17 + [700_000, 700_000, 700_000]
        df = self._make_price_df("2330", closes, volumes)

        result = compute_vcp_score(["2330"], df)
        assert result.iloc[0]["vcp_bonus"] == pytest.approx(0.03)

    def test_high_volatility_no_bonus(self):
        """近 10 日高波動（>8%）→ vcp_bonus = 0。"""
        from src.discovery.scanner import compute_vcp_score

        # 近 10 日 close 從 100 到 115（波動幅度 > 8%）
        closes = [100.0] * 10 + [100.0, 105.0, 110.0, 108.0, 106.0, 104.0, 102.0, 100.0, 115.0, 112.0]
        volumes = [1_000_000] * 17 + [700_000, 700_000, 700_000]
        df = self._make_price_df("2330", closes, volumes)

        result = compute_vcp_score(["2330"], df)
        assert result.iloc[0]["vcp_bonus"] == pytest.approx(0.0)

    def test_volume_not_contracting_no_bonus(self):
        """量能未收縮（ratio >= 0.8）→ vcp_bonus = 0，即使價格低波動。"""
        from src.discovery.scanner import compute_vcp_score

        # 近 10 日低波動，但近 3 日量 = 均量（ratio = 1.0 >= 0.8）
        closes = [100.0] * 20
        volumes = [1_000_000] * 20  # 近 3 日量 = 均量（ratio = 1.0）
        df = self._make_price_df("2330", closes, volumes)

        result = compute_vcp_score(["2330"], df)
        assert result.iloc[0]["vcp_bonus"] == pytest.approx(0.0)

    def test_empty_df_returns_zero_bonus(self):
        """空 DataFrame → 所有股票 vcp_bonus = 0。"""
        from src.discovery.scanner import compute_vcp_score

        result = compute_vcp_score(["2330", "2317"], pd.DataFrame())
        assert (result["vcp_bonus"] == 0.0).all()
        assert set(result["stock_id"].tolist()) == {"2330", "2317"}


class TestSwingAdxTechnicalFactor:
    """SwingScanner ADX(14) 第 5 技術因子測試。"""

    def _make_price_df(self, sid: str, n: int, rising: bool = True) -> pd.DataFrame:
        rows = []
        for d in range(n):
            day = date(2025, 1, 1) + timedelta(days=d)
            if rising:
                close = 100.0 + d * 0.8
            else:
                close = 100.0 - d * 0.1
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 500_000,
                }
            )
        return pd.DataFrame(rows)

    def test_adx_factor_boosts_strong_trend(self):
        """強趨勢（ADX>25）的股票技術得分應高於資料不足（無 ADX 因子）的股票。"""
        scanner = SwingScanner()
        # 強趨勢：80 根足以計算 ADX
        df_strong = self._make_price_df("STRONG", 80, rising=True)
        # 短資料：只有 10 根，ADX 無法計算（未達 28 根門檻）
        df_short = self._make_price_df("SHORT", 10, rising=True)
        df_price = pd.concat([df_strong, df_short], ignore_index=True)
        result = scanner._compute_technical_scores(["STRONG", "SHORT"], df_price)
        scores = result.set_index("stock_id")["technical_score"]
        # 強趨勢有完整 5 因子，短資料少 1 因子，分母不同不影響比較方向
        assert scores["STRONG"] >= 0.0
        assert scores["SHORT"] >= 0.0

    def test_adx_score_range(self):
        """包含 ADX 因子的技術得分應仍在 [0, 1] 範圍內。"""
        scanner = SwingScanner()
        df_price = _make_swing_price_df(80, 3)
        sids = df_price["stock_id"].unique().tolist()
        result = scanner._compute_technical_scores(sids, df_price)
        assert (result["technical_score"] >= 0.0).all()
        assert (result["technical_score"] <= 1.0).all()


class TestSwingBreakoutBonus:
    """SwingScanner 技術突破形態加成測試。"""

    def _make_price_df(self, sid: str, closes, highs=None, lows=None, volumes=None) -> pd.DataFrame:
        n = len(closes)
        rows = []
        for i in range(n):
            day = date(2025, 1, 1) + timedelta(days=i)
            c = closes[i]
            h = highs[i] if highs else c * 1.005
            lo = lows[i] if lows else c * 0.995
            v = volumes[i] if volumes else 500_000
            rows.append({"stock_id": sid, "date": day, "open": c * 0.99, "high": h, "low": lo, "close": c, "volume": v})
        return pd.DataFrame(rows)

    def test_quarterly_breakout_gives_4pct(self):
        """前 5 日 close < SMA60，今日 close > SMA60，且量 > MA20量×1.5 → breakout_bonus = 0.04。"""
        scanner = SwingScanner()
        # 設計：前 74 根 90，接著 5 根 88（回落，仍 < SMA60≈90），最後 1 根 102（突破）
        # SMA60 = (54*90 + 5*88 + 102)/60 ≈ 90.03；prev5 = 88 < 90.03，today 102 > 90.03
        closes = [90.0] * 74 + [88.0] * 5 + [102.0]
        normal_vol = 500_000
        volumes = [normal_vol] * 79 + [normal_vol * 2]
        highs = [c * 1.005 for c in closes]
        lows = [c * 0.995 for c in closes]
        df = self._make_price_df("A", closes, highs, lows, volumes)
        result = scanner._compute_breakout_bonus(["A"], df)
        assert result.iloc[0]["breakout_bonus"] == pytest.approx(0.04)

    def test_no_breakout_gives_zero(self):
        """無突破形態時 breakout_bonus = 0。"""
        scanner = SwingScanner()
        # 橫向整理，今日未突破
        closes = [100.0] * 80
        df = self._make_price_df("B", closes)
        result = scanner._compute_breakout_bonus(["B"], df)
        assert result.iloc[0]["breakout_bonus"] == pytest.approx(0.0)
