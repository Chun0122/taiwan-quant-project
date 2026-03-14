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

    def test_chip_tier_in_rankings_columns(self, momentum_scanner):
        """chip_tier 欄位應出現在 _rank_and_enrich 輸出中。"""
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
        """YoY > 0 得 0.7，YoY <= 0 得 0.3。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [10.0, -5.0],
                "mom_growth": [5.0, 5.0],
            }
        )
        result = momentum_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        assert scores["1000"] == pytest.approx(0.7)
        assert scores["1001"] == pytest.approx(0.3)

    def test_no_data_fallback(self, momentum_scanner):
        result = momentum_scanner._compute_fundamental_scores(
            ["1000"], pd.DataFrame(columns=["stock_id", "yoy_growth", "mom_growth"])
        )
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.5)


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
        """含加速度的 3 因子計算。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [20.0, 20.0],
                "mom_growth": [5.0, 5.0],
                "prev_yoy_growth": [10.0, 30.0],  # 1000 加速，1001 減速
                "prev_mom_growth": [5.0, 5.0],
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
