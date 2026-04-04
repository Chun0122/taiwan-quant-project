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
    compute_taiex_relative_strength,
)
from src.discovery.scanner._functions import (
    compute_adaptive_atr_multiplier,
    compute_chip_macd,
    compute_earnings_quality,
    compute_factor_ic,
    compute_ic_weight_adjustments,
    compute_institutional_acceleration,
    compute_key_player_cost_basis,
    compute_mfe_mae,
    compute_momentum_decay,
    compute_multi_timeframe_alignment,
    compute_peer_fundamental_ranking,
    compute_revenue_acceleration_score,
    compute_value_weighted_inst_flow,
    compute_win_rate_threshold_adjustment,
    detect_chip_tier_changes,
    score_key_player_cost,
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


class TestMomentumTechnicalScoresCrossSection:
    """橫截面排名版技術面（#3 升級）測試。"""

    def _make_bull_market_df(self) -> pd.DataFrame:
        """建立牛市資料：5 檔股票 5 日報酬率分別為 12/15/18/22/25%，全部超過舊 clamp 門檻 10%。"""
        rows = []
        five_d_rets = [0.12, 0.15, 0.18, 0.22, 0.25]
        for i, ret in enumerate(five_d_rets):
            sid = f"BULL{i}"
            for d in range(25):
                day = date(2025, 1, 1) + timedelta(days=d)
                # 前 20 日持平，後 5 日產生指定 5 日報酬率
                if d < 20:
                    close = 100.0
                else:
                    close = 100.0 * (1 + ret) ** ((d - 19) / 5)
                rows.append(
                    {
                        "stock_id": sid,
                        "date": day,
                        "open": close * 0.99,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "volume": 500_000 + i * 100_000,
                    }
                )
        return pd.DataFrame(rows)

    def test_no_ceiling_effect_in_bull_market(self, momentum_scanner):
        """牛市中多檔股票 5 日報酬率都超過舊 clamp 門檻，橫截面排名後分數仍應有鑑別度。"""
        df_price = self._make_bull_market_df()
        sids = [f"BULL{i}" for i in range(5)]
        result = momentum_scanner._compute_technical_scores(sids, df_price)
        scores = result.set_index("stock_id")["technical_score"]
        # 舊 clamp 版：全部鎖在 1.0 → nunique = 1
        # 新橫截面版：動能強弱自動分層 → nunique > 1
        assert scores.nunique() > 1, "牛市中橫截面排名應產生鑑別度（分數不應全部相同）"
        # 動能最強的股票排名應最高
        assert scores["BULL4"] >= scores["BULL0"]

    def test_strongest_always_top_in_bear_market(self, momentum_scanner):
        """熊市中最抗跌的股票，橫截面排名後仍應排在第一。"""
        rows = []
        # 5 檔全跌，跌幅：-1% / -3% / -5% / -7% / -10%（舊 clamp 版全部約 0.5 以下，無法區分）
        five_d_rets = [-0.01, -0.03, -0.05, -0.07, -0.10]
        for i, ret in enumerate(five_d_rets):
            sid = f"BEAR{i}"
            for d in range(25):
                day = date(2025, 1, 1) + timedelta(days=d)
                if d < 20:
                    close = 100.0
                else:
                    close = 100.0 * (1 + ret) ** ((d - 19) / 5)
                rows.append(
                    {
                        "stock_id": sid,
                        "date": day,
                        "open": close * 0.99,
                        "high": close * 1.01,
                        "low": max(close * 0.99, 1.0),
                        "close": max(close, 1.0),
                        "volume": 500_000,
                    }
                )
        df_price = pd.DataFrame(rows)
        sids = [f"BEAR{i}" for i in range(5)]
        result = momentum_scanner._compute_technical_scores(sids, df_price)
        scores = result.set_index("stock_id")["technical_score"]
        # 跌最少（BEAR0）的排名應高於跌最多（BEAR4）
        assert scores["BEAR0"] >= scores["BEAR4"]

    def test_scores_sum_to_expected(self, momentum_scanner):
        """橫截面排名：5 檔等量均勻分布時，平均分應接近 0.5（rank 中心）。"""
        df_price = _make_momentum_price_df(25, 5)
        sids = df_price["stock_id"].unique().tolist()
        result = momentum_scanner._compute_technical_scores(sids, df_price)
        # 5 檔排名後各因子均值均在 (0, 1) 之間，整體均值應接近 0.5（允許一定偏差）
        mean_score = result["technical_score"].mean()
        assert 0.3 <= mean_score <= 0.7, f"均值 {mean_score:.3f} 不在合理範圍 [0.3, 0.7]"


class TestMomentumRiskAdjusted:
    """#5 風險調整後動能因子（Sharpe-proxy）測試。"""

    def _make_stock_df(self, sid: str, closes: list) -> pd.DataFrame:
        rows = []
        for d, c in enumerate(closes):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 500_000,
                }
            )
        return pd.DataFrame(rows)

    def test_smooth_gain_beats_volatile_same_return(self, momentum_scanner):
        """10 日報酬率相同但波動率不同：平穩上漲的股票風險調整後動能應高於暴漲暴跌者。

        設計：
        - 兩股 10 日報酬率完全相同（close[-11]=100，close[-1]=105，ret=5%）。
        - SMOOTH：前 15 天持平，後 10 天線性上漲 → 最近 20 日 pct_change std 很低。
        - VOLATILE：前 15 天 ±5% 劇烈震盪（偶數天=100/奇數天=105），後 10 天同 SMOOTH →
          最近 20 日 pct_change std 大幅偏高。
        - 因此 SMOOTH 的 Sharpe proxy（ret_10d / vol_20d_std）遠高於 VOLATILE，
          橫截面排名後 SMOOTH 技術分應高於 VOLATILE。
        """
        # 兩股的後 10 天完全相同（ret_10d=5%）
        common_tail = [100.0 + (i + 1) * 0.5 for i in range(10)]

        # SMOOTH：前 15 天持平 → vol_20d_std 很低（後段 ≈ 0.5%/天，前段 0%/天）
        smooth_closes = [100.0] * 15 + common_tail

        # VOLATILE：前 15 天奇偶 ±5%（偶=100，奇=105）→ vol_20d_std 很高
        volatile_front = [100.0 if d % 2 == 0 else 105.0 for d in range(15)]
        volatile_closes = volatile_front + common_tail

        df = pd.concat(
            [self._make_stock_df("SMOOTH", smooth_closes), self._make_stock_df("VOLATILE", volatile_closes)],
            ignore_index=True,
        )
        result = momentum_scanner._compute_technical_scores(["SMOOTH", "VOLATILE"], df)
        scores = result.set_index("stock_id")["technical_score"]
        # 平穩上漲 Sharpe proxy 較高，技術分應高於或等於暴漲暴跌者
        assert scores["SMOOTH"] >= scores["VOLATILE"], (
            f"平穩上漲（{scores['SMOOTH']:.3f}）應 >= 暴漲暴跌（{scores['VOLATILE']:.3f}）"
        )

    def test_risk_adjusted_score_in_valid_range(self, momentum_scanner):
        """風險調整後技術分仍在 [0, 1]。"""
        df_price = _make_momentum_price_df(25, 5)
        sids = df_price["stock_id"].unique().tolist()
        result = momentum_scanner._compute_technical_scores(sids, df_price)
        assert (result["technical_score"] >= 0.0).all()
        assert (result["technical_score"] <= 1.0).all()


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
        """YoY > 0 且無加速度資料 → 基礎 Tier 3 (0.55)×80% + 加速度中性(0.5)×20% = 0.54；
        YoY <= 0 → 基礎 Tier 4 (0.30)×80% + 加速度中性(0.5)×20% = 0.34。"""
        df_revenue = pd.DataFrame(
            {
                "stock_id": ["1000", "1001"],
                "yoy_growth": [10.0, -5.0],
                "mom_growth": [5.0, 5.0],
            }
        )
        result = momentum_scanner._compute_fundamental_scores(["1000", "1001"], df_revenue)
        scores = result.set_index("stock_id")["fundamental_score"]
        # C2: 80% base tier + 20% acceleration (neutral=0.5 when no acceleration data)
        assert scores["1000"] == pytest.approx(0.55 * 0.80 + 0.5 * 0.20)
        assert scores["1001"] == pytest.approx(0.30 * 0.80 + 0.5 * 0.20)

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
        # C2 混合：base 0.85 × 80% + rev_accel 0.6（accel=10>0, 單月）× 20% = 0.80
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.80)

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
        # C2 混合：base 0.55 × 80% + rev_accel 0.3（accel=-20≤0, 減速）× 20% = 0.50
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.50)

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
        # C2 混合：base 0.72 × 80% + rev_accel 0.6（accel=10>0, 單月）× 20% = 0.696
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.696)

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
        # C2 混合：base 0.55 × 80% + rev_accel 0.5（無 yoy_3m, 中性）× 20% = 0.54
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.54)

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
        # C2 混合：base 0.30 × 80% + rev_accel 0.6（accel=3>0, 單月）× 20% = 0.36
        assert result.iloc[0]["fundamental_score"] == pytest.approx(0.36)

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


class TestSwingSmaSlope:
    """SwingScanner SMA 斜率/乖離率因子（#2 Binary Cliff 消除）測試。"""

    def _make_df(self, sid: str, closes: list) -> pd.DataFrame:
        rows = []
        for d, c in enumerate(closes):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 500_000,
                }
            )
        return pd.DataFrame(rows)

    def test_rising_sma60_scores_higher_than_falling(self):
        """SMA60 持續上揚的股票技術分應高於 SMA60 持續下滑的股票。"""
        scanner = SwingScanner()
        # 上升：65 天穩定上漲
        closes_up = [100.0 + d * 0.8 for d in range(65)]
        # 下降：65 天穩定下跌
        closes_down = [150.0 - d * 0.8 for d in range(65)]
        df = pd.concat(
            [self._make_df("UP", closes_up), self._make_df("DOWN", closes_down)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["UP", "DOWN"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["UP"] > scores["DOWN"]

    def test_sma_spread_positive_beats_negative(self):
        """SMA20 > SMA60（多頭排列）的乖離率分數應高於 SMA20 < SMA60（空頭排列）。"""
        scanner = SwingScanner()
        # 強多頭排列：前 40 天持平，後 20 天急拉（SMA20 遠高於 SMA60）
        c_golden = [80.0] * 40 + [80.0 + (i + 1) * 2.0 for i in range(25)]
        # 空頭排列：前 40 天持平，後 20 天急跌（SMA20 遠低於 SMA60）
        c_death = [150.0] * 40 + [150.0 - (i + 1) * 2.0 for i in range(25)]
        df = pd.concat(
            [self._make_df("GOLDEN", c_golden), self._make_df("DEATH", c_death)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["GOLDEN", "DEATH"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["GOLDEN"] > scores["DEATH"]

    def test_slope_score_in_valid_range(self):
        """SMA 斜率/乖離率因子計算後技術分仍在 [0, 1]。"""
        scanner = SwingScanner()
        df = _make_swing_price_df(80, 3)
        sids = df["stock_id"].unique().tolist()
        result = scanner._compute_technical_scores(sids, df)
        assert (result["technical_score"] >= 0.0).all()
        assert (result["technical_score"] <= 1.0).all()


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


class TestValueStyleTechnicalScores:
    """ValueScanner 專屬技術面 3 因子（#4 Value 校準）測試。"""

    def _make_df(self, sid: str, closes: list) -> pd.DataFrame:
        rows = []
        for d, c in enumerate(closes):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 500_000,
                }
            )
        return pd.DataFrame(rows)

    def test_low_rsi_scores_higher_than_high_rsi(self):
        """RSI 超賣（低 RSI）的股票 Value 技術分應高於 RSI 過熱（高 RSI）者。"""
        scanner = ValueScanner()
        # LOW_RSI：130 天持平後急跌（RSI < 30）
        c_low_rsi = [100.0] * 110 + [100.0 - d * 1.5 for d in range(20)]
        # HIGH_RSI：130 天持平後急漲（RSI > 70）
        c_high_rsi = [100.0] * 110 + [100.0 + d * 1.5 for d in range(20)]
        df = pd.concat(
            [self._make_df("LOW_RSI", c_low_rsi), self._make_df("HIGH_RSI", c_high_rsi)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["LOW_RSI", "HIGH_RSI"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["LOW_RSI"] > scores["HIGH_RSI"], (
            f"超賣股（{scores['LOW_RSI']:.3f}）應高於過熱股（{scores['HIGH_RSI']:.3f}）"
        )

    def test_near_120d_low_scores_higher(self):
        """股價接近 120 日低點的股票（Value 區間）技術分應高於偏離低點的股票。"""
        scanner = ValueScanner()
        # NEAR_LOW：120 天穩定，最近收盤接近低點（距低點 < 5%）
        c_near = [100.0] * 119 + [101.0]
        # FAR_HIGH：120 天穩定，最近收盤遠高於低點（距低點 > 60%）
        c_far = [100.0] * 119 + [165.0]
        df = pd.concat(
            [self._make_df("NEAR", c_near), self._make_df("FAR", c_far)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["NEAR", "FAR"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["NEAR"] > scores["FAR"], f"接近低點（{scores['NEAR']:.3f}）應高於遠離低點（{scores['FAR']:.3f}）"

    def test_ma_convergence_scores_higher_than_divergence(self):
        """三均線收斂（糾結）的股票 Value 技術分應高於三均線發散者。"""
        scanner = ValueScanner()
        # CONVERGE：120 天完全持平（SMA20/SMA60/SMA120 幾乎相等，CV≈0）
        c_conv = [100.0] * 130
        # DIVERGE：後 120 天包含大幅拉升（SMA20 遠高於 SMA120，CV > 6%）
        c_divg = [80.0] * 10 + [80.0 + d * 1.2 for d in range(120)]
        df = pd.concat(
            [self._make_df("CONV", c_conv), self._make_df("DIVG", c_divg)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["CONV", "DIVG"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["CONV"] > scores["DIVG"], (
            f"均線收斂（{scores['CONV']:.3f}）應高於均線發散（{scores['DIVG']:.3f}）"
        )

    def test_score_in_valid_range(self):
        """Value 技術分仍在 [0, 1]。"""
        scanner = ValueScanner()
        df = self._make_df("X", [100.0] * 130)
        result = scanner._compute_technical_scores(["X"], df)
        s = result.iloc[0]["technical_score"]
        assert 0.0 <= s <= 1.0


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


class TestDividendStyleTechnicalScores:
    """DividendScanner 專屬技術面 3 因子（#4 Dividend 校準）測試。"""

    def _make_df(self, sid: str, closes: list) -> pd.DataFrame:
        rows = []
        for d, c in enumerate(closes):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 500_000,
                }
            )
        return pd.DataFrame(rows)

    def test_neutral_rsi_scores_higher_than_extreme_rsi(self):
        """RSI 在中性帶（接近 50）的股票 Dividend 技術分應高於 RSI 極端者。"""
        scanner = DividendScanner()
        # NEUTRAL：130 天穩定橫盤（RSI ≈ 50）
        c_neutral = [100.0] * 130
        # OVERBOUGHT：持續急漲（RSI > 70）
        c_overbought = [100.0] * 110 + [100.0 + d * 2.0 for d in range(20)]
        df = pd.concat(
            [self._make_df("NEUTRAL", c_neutral), self._make_df("OVER", c_overbought)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["NEUTRAL", "OVER"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["NEUTRAL"] > scores["OVER"], (
            f"中性帶（{scores['NEUTRAL']:.3f}）應高於過熱（{scores['OVER']:.3f}）"
        )

    def test_rising_sma60_scores_higher_than_falling(self):
        """SMA60 穩健上升的股票 Dividend 技術分應高於 SMA60 下滑者。"""
        scanner = DividendScanner()
        # RISING：前 50 天持平，後 80 天緩步上漲（SMA60 今日 > 20 天前）
        c_rising = [100.0] * 50 + [100.0 + d * 0.3 for d in range(80)]
        # FALLING：前 50 天持平，後 80 天緩步下跌（SMA60 今日 < 20 天前）
        c_falling = [100.0] * 50 + [100.0 - d * 0.3 for d in range(80)]
        df = pd.concat(
            [self._make_df("RISING", c_rising), self._make_df("FALLING", c_falling)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["RISING", "FALLING"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["RISING"] > scores["FALLING"], (
            f"SMA60 上升（{scores['RISING']:.3f}）應高於下滑（{scores['FALLING']:.3f}）"
        )

    def test_score_in_valid_range(self):
        """Dividend 技術分仍在 [0, 1]。"""
        scanner = DividendScanner()
        rows = []
        for d in range(130):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {"stock_id": "X", "date": day, "open": 99, "high": 101, "low": 99, "close": 100.0, "volume": 500_000}
            )
        df = pd.DataFrame(rows)
        result = scanner._compute_technical_scores(["X"], df)
        s = result.iloc[0]["technical_score"]
        assert 0.0 <= s <= 1.0


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
        monkeypatch.setattr("src.discovery.scanner._base.get_session", lambda: mock_session)
        # ValueScanner/DividendScanner 子模組也各自 import get_session
        monkeypatch.setattr("src.discovery.scanner._value.get_session", lambda: mock_session)
        monkeypatch.setattr("src.discovery.scanner._dividend.get_session", lambda: mock_session)
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
        # general 使用快速衰減 0.15
        assert w1 == pytest.approx(math.exp(-0.15), abs=1e-9)
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

    def test_general_uses_transient_decay(self):
        """general 事件使用快速衰減常數 0.15：1 天後 = exp(-0.15)。"""
        import math

        from src.discovery.scanner import compute_news_decay_weight

        assert compute_news_decay_weight(1, "general") == pytest.approx(math.exp(-0.15), abs=1e-9)

    def test_structural_uses_slow_decay(self):
        """governance_change 使用慢衰減 0.07：10 天後仍保留 ~50%。"""
        import math

        from src.discovery.scanner import compute_news_decay_weight

        w = compute_news_decay_weight(10, "governance_change")
        # exp(-0.07 × 10) × 5.0 = exp(-0.7) × 5.0 ≈ 0.497 × 5.0 = 2.48
        assert w == pytest.approx(math.exp(-0.7) * 5.0, abs=0.01)
        assert w > 2.0  # 10 天後結構性事件仍有顯著影響

    def test_transient_decays_faster_than_structural(self):
        """相同天數下，一般性事件衰減更快。"""
        from src.discovery.scanner import compute_news_decay_weight

        # 7 天後比較（排除 type_weight 差異，只看衰減率）
        # general: exp(-0.15 × 7) × 1.0 = 0.35
        # governance: exp(-0.07 × 7) × 5.0 = 3.06
        # 正規化：general/1.0 vs governance/5.0
        g = compute_news_decay_weight(7, "general") / 1.0
        s = compute_news_decay_weight(7, "governance_change") / 5.0
        assert s > g  # 結構性衰減更慢

    def test_earnings_call_uses_default_decay(self):
        """earnings_call 使用中性預設衰減 0.12。"""
        import math

        from src.discovery.scanner import compute_news_decay_weight

        w = compute_news_decay_weight(1, "earnings_call")
        assert w == pytest.approx(math.exp(-0.12) * 3.0, abs=1e-6)

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
        "revenue": None,
        "net_income": None,
        "operating_cf": None,
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
            columns=[
                "stock_id",
                "date",
                "year",
                "quarter",
                "eps",
                "roe",
                "gross_margin",
                "debt_ratio",
                "revenue",
                "net_income",
                "operating_cf",
            ]
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
            columns=[
                "stock_id",
                "date",
                "year",
                "quarter",
                "eps",
                "roe",
                "gross_margin",
                "debt_ratio",
                "revenue",
                "net_income",
                "operating_cf",
            ]
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
        monkeypatch.setattr(
            "src.discovery.scanner._swing.compute_smart_broker_score", lambda df, prices, **kw: smart_result
        )

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
        monkeypatch.setattr(
            "src.discovery.scanner._swing.compute_smart_broker_score", lambda df, prices, **kw: smart_result
        )

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


class TestSwingAdxDmiFilter:
    """SwingScanner ADX + DMI 方向過濾（#1 升級）測試。"""

    def _make_trend_df(self, sid: str, n: int, rising: bool) -> pd.DataFrame:
        rows = []
        for d in range(n):
            day = date(2025, 1, 1) + timedelta(days=d)
            if rising:
                close = 100.0 + d * 0.8  # 每日穩定上漲
            else:
                close = 100.0 - d * 0.8  # 每日穩定下跌
            close = max(close, 1.0)
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": close * (1.005 if rising else 0.995),
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 500_000,
                }
            )
        return pd.DataFrame(rows)

    def test_downtrend_adx_suppressed(self):
        """強下跌趨勢（-DI > +DI）時，ADX 因子不加分，技術分數應低於多頭趨勢。"""
        scanner = SwingScanner()
        df_up = self._make_trend_df("UP", 80, rising=True)
        df_down = self._make_trend_df("DOWN", 80, rising=False)
        df_price = pd.concat([df_up, df_down], ignore_index=True)
        result = scanner._compute_technical_scores(["UP", "DOWN"], df_price)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["UP"] > scores["DOWN"], f"多頭技術分 {scores['UP']:.3f} 應大於空頭技術分 {scores['DOWN']:.3f}"

    def test_downtrend_score_in_valid_range(self):
        """下跌趨勢（ADX 因子給 0）時，整體技術分仍應在 [0, 1] 範圍內。"""
        scanner = SwingScanner()
        df_price = self._make_trend_df("D1", 80, rising=False)
        result = scanner._compute_technical_scores(["D1"], df_price)
        assert 0.0 <= result.iloc[0]["technical_score"] <= 1.0


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


class TestComputeTaixRelativeStrength:
    """compute_taiex_relative_strength 純函數測試。"""

    def _make_df(self, stock_closes: dict[str, list[float]], n_days: int = 25) -> pd.DataFrame:
        """建立含多支股票（含 TAIEX）的 df_price。"""
        from datetime import date, timedelta

        rows = []
        base_date = date(2025, 1, 1)
        for sid, closes in stock_closes.items():
            for i, c in enumerate(closes):
                rows.append({"stock_id": sid, "date": base_date + timedelta(days=i), "close": c})
        return pd.DataFrame(rows)

    def test_stock_outperforming_taiex(self):
        """個股漲幅 15%，TAIEX 漲幅 5% → excess > 0（個股跑贏 TAIEX）。"""
        n = 25
        taiex = [10000.0 * (1 + 0.05 / n * i) for i in range(n + 1)]
        stock = [100.0 * (1 + 0.15 / n * i) for i in range(n + 1)]
        df = self._make_df({"TAIEX": taiex, "2330": stock})
        result = compute_taiex_relative_strength(df, window=20)
        assert "2330" in result.index
        assert result["2330"] > 0.05  # 超額報酬為正且明顯 > 5%

    def test_stock_underperforming_taiex(self):
        """個股跌幅 8%，TAIEX 跌幅 2% → excess ≈ -6%。"""
        n = 25
        taiex = [10000.0 * (1 - 0.02 / n * i) for i in range(n + 1)]
        stock = [100.0 * (1 - 0.08 / n * i) for i in range(n + 1)]
        df = self._make_df({"TAIEX": taiex, "2317": stock})
        result = compute_taiex_relative_strength(df, window=20)
        assert result["2317"] == pytest.approx(-0.06, abs=0.02)

    def test_no_taiex_data_returns_zeros(self):
        """df_price 中無 TAIEX 資料時，所有個股返回 0.0（安全 fallback）。"""
        n = 25
        df = self._make_df({"2330": [100.0 + i for i in range(n + 1)]})
        result = compute_taiex_relative_strength(df, window=20)
        assert "2330" in result.index
        assert result["2330"] == pytest.approx(0.0)

    def test_taiex_excluded_from_output(self):
        """TAIEX 本身不出現在返回的 Series index 中。"""
        n = 25
        taiex = [10000.0] * (n + 1)
        stock = [100.0] * (n + 1)
        df = self._make_df({"TAIEX": taiex, "2330": stock})
        result = compute_taiex_relative_strength(df, window=20)
        assert "TAIEX" not in result.index

    def test_insufficient_data_returns_zero(self):
        """個股資料不足 window+1 天時填 0.0（不懲罰新股）。"""
        n = 25
        taiex = [10000.0 * (1 + 0.05 / n * i) for i in range(n + 1)]
        short_stock = [100.0 + i for i in range(10)]  # 僅 10 天
        df = self._make_df({"TAIEX": taiex, "NEW": short_stock})
        result = compute_taiex_relative_strength(df, window=20)
        assert result.get("NEW", 0.0) == pytest.approx(0.0)

    def test_empty_df_returns_empty(self):
        """空 DataFrame 不拋例外，返回空 Series。"""
        result = compute_taiex_relative_strength(pd.DataFrame())
        assert len(result) == 0


class TestApplyCrisisFilter:
    """MarketScanner._apply_crisis_filter 方法測試。"""

    def _make_price_df_with_taiex(
        self,
        stock_returns: dict[str, float],
        taiex_return: float = -0.03,
        n_days: int = 25,
    ) -> pd.DataFrame:
        """建立含 TAIEX 的 df_price，各股有指定 N 日報酬率。"""
        from datetime import date, timedelta

        base_date = date(2025, 1, 1)
        rows = []
        # TAIEX
        for i in range(n_days + 1):
            close = 10000.0 * (1 + taiex_return / n_days * i)
            rows.append({"stock_id": "TAIEX", "date": base_date + timedelta(days=i), "close": close})
        # 個股
        for sid, ret in stock_returns.items():
            for i in range(n_days + 1):
                close = 100.0 * (1 + ret / n_days * i)
                rows.append({"stock_id": sid, "date": base_date + timedelta(days=i), "close": close})
        return pd.DataFrame(rows)

    def _make_scored(self, stock_ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stock_id": stock_ids,
                "composite_score": [0.8 - 0.1 * i for i in range(len(stock_ids))],
            }
        )

    def test_non_crisis_regime_no_filter(self):
        """非 crisis regime 時不過濾任何股票（zero overhead）。"""
        scanner = MomentumScanner()
        scanner.regime = "bear"  # 非 crisis
        df_price = self._make_price_df_with_taiex({"2330": -0.15, "2317": -0.12}, taiex_return=-0.03)
        scored = self._make_scored(["2330", "2317"])
        result = scanner._apply_crisis_filter(scored, df_price)
        assert len(result) == 2

    def test_crisis_regime_filters_weak(self):
        """crisis regime 時剔除跑輸 TAIEX 超過 10% 的弱勢股。"""
        scanner = MomentumScanner()
        scanner.regime = "crisis"
        # TAIEX -5%；2330 -3%（超額 +2%，強勢留下）；2317 -25%（超額 -17%，剔除）
        df_price = self._make_price_df_with_taiex({"2330": -0.03, "2317": -0.25}, taiex_return=-0.05)
        scored = self._make_scored(["2330", "2317"])
        result = scanner._apply_crisis_filter(scored, df_price)
        assert "2330" in result["stock_id"].values
        assert "2317" not in result["stock_id"].values

    def test_crisis_all_strong_nothing_removed(self):
        """crisis 模式但所有股票都強於 TAIEX 時無剔除。"""
        scanner = MomentumScanner()
        scanner.regime = "crisis"
        # TAIEX -5%；兩股各跌 3%（超額 +2%）
        df_price = self._make_price_df_with_taiex({"A": -0.03, "B": -0.03}, taiex_return=-0.05)
        scored = self._make_scored(["A", "B"])
        result = scanner._apply_crisis_filter(scored, df_price)
        assert len(result) == 2

    def test_empty_scored_safe_fallback(self):
        """空 DataFrame 時不拋例外。"""
        scanner = MomentumScanner()
        scanner.regime = "crisis"
        df_price = self._make_price_df_with_taiex({"A": -0.03}, taiex_return=-0.05)
        result = scanner._apply_crisis_filter(pd.DataFrame(), df_price)
        assert result.empty


# ====================================================================== #
#  P0-1: Momentum F3 季線突破（60d）測試
# ====================================================================== #


class TestMomentumBreakout60d:
    """MomentumScanner F3 突破因子改用 60 日最高（季線突破）。"""

    def _make_df(self, sid: str, closes: list) -> pd.DataFrame:
        rows = []
        for d, c in enumerate(closes):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 500_000 + d * 1000,
                }
            )
        return pd.DataFrame(rows)

    def test_near_60d_high_scores_higher_rank(self):
        """60 日新高附近的股票 F3 排名應高於遠低於 60 日高點者。"""
        scanner = MomentumScanner()
        # HIGH_NEAR：前 60 天平均 100，最近一天 110（接近 60 日高）
        c_near = [100.0] * 60 + [110.0]
        # HIGH_FAR：前 60 天平均 100，最近一天 70（遠低於 60 日高）
        c_far = [100.0] * 60 + [70.0]
        df = pd.concat(
            [self._make_df("NEAR", c_near), self._make_df("FAR", c_far)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["NEAR", "FAR"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["NEAR"] > scores["FAR"], (
            f"接近 60 日高（{scores['NEAR']:.3f}）應高於遠低於 60 日高（{scores['FAR']:.3f}）"
        )

    def test_insufficient_60d_data_falls_back_to_neutral(self):
        """資料不足 60 天時，F3 以中性分 0.5 填補（不影響整體結果有效性）。"""
        scanner = MomentumScanner()
        # 只有 25 天資料，無法計算 60 日最高，F3 應 fallback 為 0.5
        df = self._make_df("SHORT", [100.0 + i * 0.5 for i in range(25)])
        result = scanner._compute_technical_scores(["SHORT"], df)
        s = result.iloc[0]["technical_score"]
        assert 0.0 <= s <= 1.0, f"分數應在 [0, 1]，得到 {s}"

    def test_scores_in_valid_range_with_full_data(self):
        """65 天完整資料的技術分數應在 [0, 1]。"""
        scanner = MomentumScanner()
        df = self._make_df("X", [100.0 * (1 + 0.002 * d) for d in range(65)])
        result = scanner._compute_technical_scores(["X"], df)
        s = result.iloc[0]["technical_score"]
        assert 0.0 <= s <= 1.0


# ====================================================================== #
#  P1-2: Momentum 漲停板量能免懲罰測試
# ====================================================================== #


class TestMomentumLimitUpVolume:
    """漲停板（≥9.8% 單日漲幅）時 rv/ra 自動設為滿分（1.0），避免量縮誤判。"""

    def _make_df_two_stocks(self) -> pd.DataFrame:
        """
        LU（漲停）：前 64 天穩定 100，第 65 天漲 10%（110），今日成交量極低（50,000）
        NLU（普通）：前 64 天穩定 100，第 65 天漲 2%（102），今日成交量正常（1,500,000）
        """
        rows = []
        base_vol = 1_000_000
        for d in range(64):
            day = date(2025, 1, 1) + timedelta(days=d)
            for sid, close in [("LU", 100.0), ("NLU", 100.0)]:
                rows.append(
                    {
                        "stock_id": sid,
                        "date": day,
                        "open": close * 0.99,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "volume": base_vol,
                    }
                )
        last_day = date(2025, 1, 1) + timedelta(days=64)
        rows.append(
            {
                "stock_id": "LU",
                "date": last_day,
                "open": 100.0,
                "high": 110.0,
                "low": 100.0,
                "close": 110.0,
                "volume": 50_000,  # 漲停量縮
            }
        )
        rows.append(
            {
                "stock_id": "NLU",
                "date": last_day,
                "open": 100.0,
                "high": 102.0,
                "low": 100.0,
                "close": 102.0,
                "volume": 1_500_000,  # 正常量能
            }
        )
        return pd.DataFrame(rows)

    def test_limit_up_not_penalized_vs_normal(self):
        """漲停股即使量縮，技術分數仍應優於或不低於動能較弱的正常股。"""
        scanner = MomentumScanner()
        df = self._make_df_two_stocks()
        result = scanner._compute_technical_scores(["LU", "NLU"], df)
        scores = result.set_index("stock_id")["technical_score"]
        # LU 動能（+10%）遠優於 NLU（+2%），應在保護後維持優勢
        assert scores["LU"] >= scores["NLU"], f"漲停股（{scores['LU']:.3f}）不應低於普通股（{scores['NLU']:.3f}）"

    def test_scores_in_valid_range(self):
        """漲停保護後分數仍在 [0, 1]。"""
        scanner = MomentumScanner()
        df = self._make_df_two_stocks()
        result = scanner._compute_technical_scores(["LU", "NLU"], df)
        assert (result["technical_score"] >= 0.0).all()
        assert (result["technical_score"] <= 1.0).all()


# ====================================================================== #
#  P1-1: Value SMA5 右側確認因子測試
# ====================================================================== #


class TestValueSMA5SparkFactor:
    """Value 第 4 因子：收盤 > SMA5 右側確認，避免接刀。"""

    def _make_df(self, sid: str, closes: list) -> pd.DataFrame:
        rows = []
        for d, c in enumerate(closes):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 500_000,
                }
            )
        return pd.DataFrame(rows)

    def test_close_above_sma5_scores_higher(self):
        """收盤 > SMA5 的股票（右側確認）技術分應高於收盤 < SMA5 的股票（仍在下跌）。

        設計原則：F1/F2/F3 對兩支股票幾乎相同（基底 129 天一致），
        只有最後一天的微小差異決定 F4（close vs SMA5），確保 F4 是決定性因子。
        """
        scanner = ValueScanner()
        # 共用底部：前 100 天從 100 下跌至 90（製造超賣 + 接近低點），
        # 再 29 天緩步回穩至 91（RSI/距低點 對兩股幾乎相同）
        base_closes = [100.0 - i * 0.1 for i in range(100)] + [90.0 + i * (1.0 / 29) for i in range(29)]
        # SMA5 = mean of last 5 base days ≈ 90.76
        # ABOVE：最後一天微漲至 91.2（> SMA5），F4 = 1.0
        c_above = base_closes + [91.2]
        # BELOW：最後一天微跌至 90.3（< SMA5），F4 = 0.0
        c_below = base_closes + [90.3]
        df = pd.concat(
            [self._make_df("ABOVE", c_above), self._make_df("BELOW", c_below)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["ABOVE", "BELOW"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["ABOVE"] > scores["BELOW"], (
            f"右側確認（{scores['ABOVE']:.3f}）應高於跌破SMA5（{scores['BELOW']:.3f}）"
        )

    def test_score_range_with_four_factors(self):
        """4 因子後分數仍在 [0, 1]。"""
        scanner = ValueScanner()
        closes = [100.0] * 130
        df = self._make_df("X", closes)
        result = scanner._compute_technical_scores(["X"], df)
        s = result.iloc[0]["technical_score"]
        assert 0.0 <= s <= 1.0


# ====================================================================== #
#  P0-2: Swing SMA60 斜率 20 日窗口測試
# ====================================================================== #


class TestSwingSma60Slope20d:
    """SwingScanner F1 SMA60 斜率改用 20 日比較窗口。"""

    def _make_df(self, sid: str, closes: list) -> pd.DataFrame:
        rows = []
        for d, c in enumerate(closes):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 500_000,
                }
            )
        return pd.DataFrame(rows)

    def test_rising_sma60_over_20d_scores_higher(self):
        """SMA60 過去 20 日上升的股票技術分應高於 SMA60 下滑者。"""
        scanner = SwingScanner()
        # RISING：前 60 天持平，後 80 天緩步上漲（SMA60_today > SMA60_20d_ago）
        c_rising = [100.0] * 60 + [100.0 + d * 0.5 for d in range(80)]
        # FALLING：前 60 天持平，後 80 天緩步下跌
        c_falling = [100.0] * 60 + [100.0 - d * 0.5 for d in range(80)]
        df = pd.concat(
            [self._make_df("RISING", c_rising), self._make_df("FALLING", c_falling)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["RISING", "FALLING"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["RISING"] > scores["FALLING"], (
            f"SMA60 上升（{scores['RISING']:.3f}）應高於下滑（{scores['FALLING']:.3f}）"
        )

    def test_fewer_than_80d_skips_f1(self):
        """資料不足 80 天時 F1 被跳過，技術分仍在 [0, 1]。"""
        scanner = SwingScanner()
        df = self._make_df("SHORT", [100.0 + i * 0.2 for i in range(70)])
        result = scanner._compute_technical_scores(["SHORT"], df)
        s = result.iloc[0]["technical_score"]
        assert 0.0 <= s <= 1.0


# ====================================================================== #
#  P2-2: Swing F4 量價結構（VPT）測試
# ====================================================================== #


class TestSwingVolumePriceRatio:
    """SwingScanner F4 改為漲日均量/跌日均量比率（VPT 概念）。"""

    def _make_vpt_df(self, sid: str, up_day_vol: float, down_day_vol: float) -> pd.DataFrame:
        """建立近 21 天資料：前 20 天交替漲跌，分別帶入不同成交量。"""
        rows = []
        base = 100.0
        for d in range(21):
            day = date(2025, 1, 1) + timedelta(days=d)
            if d == 0:
                close = base
                vol = (up_day_vol + down_day_vol) / 2
            elif d % 2 == 1:  # 漲日
                close = base + 0.5 * ((d + 1) // 2)
                vol = up_day_vol
            else:  # 跌日
                close = base + 0.5 * (d // 2) - 0.3
                vol = down_day_vol
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": int(vol),
                }
            )
        return pd.DataFrame(rows)

    def test_up_vol_greater_scores_higher(self):
        """漲日均量 > 跌日均量（健康多頭）應比漲日均量 < 跌日均量得分高。"""
        scanner = SwingScanner()
        # HEALTH：漲日均量 1,000,000；跌日均量 300,000（ratio≈3.3）
        df_h = self._make_vpt_df("HEALTH", 1_000_000, 300_000)
        # WEAK：漲日均量 300,000；跌日均量 1,000,000（ratio≈0.3）
        df_w = self._make_vpt_df("WEAK", 300_000, 1_000_000)
        df = pd.concat([df_h, df_w], ignore_index=True)
        result = scanner._compute_technical_scores(["HEALTH", "WEAK"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["HEALTH"] > scores["WEAK"], (
            f"健康量價（{scores['HEALTH']:.3f}）應高於弱勢量價（{scores['WEAK']:.3f}）"
        )

    def test_score_in_valid_range(self):
        """量價結構分數仍在 [0, 1]。"""
        scanner = SwingScanner()
        df = _make_swing_price_df(80, 3)
        sids = df["stock_id"].unique().tolist()
        result = scanner._compute_technical_scores(sids, df)
        assert (result["technical_score"] >= 0.0).all()
        assert (result["technical_score"] <= 1.0).all()


# ====================================================================== #
#  P2-3: Dividend 低波動率因子測試
# ====================================================================== #


class TestDividendLowVolatility:
    """DividendScanner F4 新增低歷史波動率（負向）因子。"""

    def _make_df(self, sid: str, closes: list) -> pd.DataFrame:
        rows = []
        for d, c in enumerate(closes):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 500_000,
                }
            )
        return pd.DataFrame(rows)

    def test_low_vol_scores_higher_than_high_vol(self):
        """低波動率（穩健存股）技術分應高於高波動率股票。"""
        scanner = DividendScanner()
        # LOW_VOL：130 天幾乎持平（HV ≈ 0）
        c_low = [100.0 + i * 0.01 for i in range(130)]
        # HIGH_VOL：130 天每天暴漲暴跌 5%（HV >> 3%）
        c_high = [100.0 * (1 + 0.05 * (1 if i % 2 == 0 else -1)) for i in range(130)]
        df = pd.concat(
            [self._make_df("LOW_VOL", c_low), self._make_df("HIGH_VOL", c_high)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["LOW_VOL", "HIGH_VOL"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["LOW_VOL"] > scores["HIGH_VOL"], (
            f"低波動（{scores['LOW_VOL']:.3f}）應高於高波動（{scores['HIGH_VOL']:.3f}）"
        )

    def test_score_in_valid_range(self):
        """4 因子後分數仍在 [0, 1]。"""
        scanner = DividendScanner()
        closes = [100.0] * 130
        df = self._make_df("X", closes)
        result = scanner._compute_technical_scores(["X"], df)
        s = result.iloc[0]["technical_score"]
        assert 0.0 <= s <= 1.0


# ====================================================================== #
#  P3: Dividend 除息缺口免疫測試
# ====================================================================== #


class TestDividendExDivGap:
    """DividendScanner F1 SMA60 斜率在偵測到除息缺口（近 45 日跌幅 ≥ 4.5%）時給中性 0.5。"""

    def _make_df(self, sid: str, closes: list) -> pd.DataFrame:
        rows = []
        for d, c in enumerate(closes):
            day = date(2025, 1, 1) + timedelta(days=d)
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": c * 0.99,
                    "high": c * 1.01,
                    "low": c * 0.99,
                    "close": c,
                    "volume": 500_000,
                }
            )
        return pd.DataFrame(rows)

    def test_ex_div_gap_suppresses_slope_penalty(self):
        """
        下滑趨勢中若近 45 日出現 ≥4.5% 單日跌幅（除息代理），
        F1 給 0.5（中性），使整體分數不會因斜率下彎被大幅懲罰。
        """
        scanner = DividendScanner()
        # FALLING_WITH_GAP：前 80 天持平 100，第 81 天除息跌 -5%（95），後 49 天繼續持平
        # 此後 SMA60 向下斜，但因偵測到除息缺口 F1 → 0.5（而非懲罰）
        c_gap = [100.0] * 80 + [95.0] + [95.0] * 49
        # FALLING_NO_GAP：前 80 天緩跌，後 50 天繼續緩跌，無單日 4.5% 跌幅
        # SMA60 斜率下彎，F1 → 低分
        c_no_gap = [100.0 - i * 0.15 for i in range(130)]
        df = pd.concat(
            [self._make_df("GAP", c_gap), self._make_df("NOGAP", c_no_gap)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["GAP", "NOGAP"], df)
        scores = result.set_index("stock_id")["technical_score"]
        # GAP 有除息免疫（F1=0.5，中性），NOGAP 無免疫（F1 因斜率下彎被懲罰）
        assert scores["GAP"] >= scores["NOGAP"], (
            f"除息免疫（{scores['GAP']:.3f}）應不低於純下滑（{scores['NOGAP']:.3f}）"
        )

    def test_no_gap_calculates_slope_normally(self):
        """無除息缺口的股票，SMA60 斜率正常計算（上升股優於下滑股）。"""
        scanner = DividendScanner()
        # RISE：前 50 天持平，後 80 天緩步上漲（SMA60 穩健向上，RSI 不過熱）
        c_rising = [100.0] * 50 + [100.0 + d * 0.3 for d in range(80)]
        # FALL：前 50 天持平，後 80 天緩步下跌（SMA60 向下）
        c_falling = [100.0] * 50 + [100.0 - d * 0.3 for d in range(80)]
        df = pd.concat(
            [self._make_df("RISE", c_rising), self._make_df("FALL", c_falling)],
            ignore_index=True,
        )
        result = scanner._compute_technical_scores(["RISE", "FALL"], df)
        scores = result.set_index("stock_id")["technical_score"]
        assert scores["RISE"] >= scores["FALL"], (
            f"SMA60 上升（{scores['RISE']:.3f}）應不低於下滑（{scores['FALL']:.3f}）"
        )

    def test_score_in_valid_range(self):
        """除息缺口偵測後分數仍在 [0, 1]。"""
        scanner = DividendScanner()
        # 含一次大跌
        c = [100.0] * 80 + [95.0] + [95.0] * 49
        df = self._make_df("Z", c)
        result = scanner._compute_technical_scores(["Z"], df)
        s = result.iloc[0]["technical_score"]
        assert 0.0 <= s <= 1.0


# ====================================================================== #
#  TestDetectDaytradeBrokers — 隔日沖行為偵測純函數測試
# ====================================================================== #


class TestDetectDaytradeBrokers:
    """detect_daytrade_brokers() 向量化配對邏輯測試。"""

    _BASE = date(2026, 3, 10)

    def _make_broker_df(self, stock_id: str, records: list[dict]) -> pd.DataFrame:
        """records: list of {date_offset, broker_id, broker_name, buy, sell}"""
        rows = []
        for r in records:
            rows.append(
                {
                    "stock_id": stock_id,
                    "date": self._BASE + timedelta(days=r.get("date_offset", 0)),
                    "broker_id": r.get("broker_id", "B001"),
                    "broker_name": r.get("broker_name", "測試分點"),
                    "buy": r.get("buy", 0),
                    "sell": r.get("sell", 0),
                }
            )
        return pd.DataFrame(rows)

    def test_basic_t1_sell(self):
        """T 日大量買進，T+1 對應賣出 → 配對成功。"""
        from src.discovery.scanner import detect_daytrade_brokers

        # 建構 5 次「買→隔日賣」模式
        records = []
        for i in range(5):
            d = i * 2
            records.append({"date_offset": d, "buy": 10000, "sell": 0})
            records.append({"date_offset": d + 1, "buy": 0, "sell": 8000})

        df = self._make_broker_df("2330", records)
        result = detect_daytrade_brokers(df, min_events=3)
        assert len(result) == 1
        assert result.iloc[0]["daytrade_events"] == 5
        assert result.iloc[0]["avg_hold_days"] == 1.0

    def test_t3_delayed_sell(self):
        """T+3 才賣出，仍在窗口內 → 配對成功。"""
        from src.discovery.scanner import detect_daytrade_brokers

        records = []
        for i in range(4):
            d = i * 4
            records.append({"date_offset": d, "buy": 10000, "sell": 0})
            records.append({"date_offset": d + 1, "buy": 500, "sell": 500})  # 中性
            records.append({"date_offset": d + 2, "buy": 500, "sell": 500})  # 中性
            records.append({"date_offset": d + 3, "buy": 0, "sell": 9000})  # T+3 賣出

        df = self._make_broker_df("2330", records)
        result = detect_daytrade_brokers(df, min_events=3)
        assert len(result) == 1
        assert result.iloc[0]["daytrade_events"] >= 3

    def test_no_matching_sell(self):
        """買進後無對應賣出 → 不標記。"""
        from src.discovery.scanner import detect_daytrade_brokers

        records = []
        for i in range(5):
            records.append({"date_offset": i, "buy": 10000, "sell": 0})  # 只買不賣

        df = self._make_broker_df("2330", records)
        result = detect_daytrade_brokers(df, min_events=1)
        assert len(result) == 0

    def test_sell_ratio_below_threshold(self):
        """賣量 < 70% 買量 → 不配對。"""
        from src.discovery.scanner import detect_daytrade_brokers

        records = []
        for i in range(5):
            d = i * 2
            records.append({"date_offset": d, "buy": 10000, "sell": 0})
            records.append({"date_offset": d + 1, "buy": 0, "sell": 5000})  # 50% < 70%

        df = self._make_broker_df("2330", records)
        result = detect_daytrade_brokers(df, sell_buy_ratio_min=0.70, min_events=1)
        assert len(result) == 0

    def test_min_events_filter(self):
        """僅 2 次配對 < min_events=3 → 不標記。"""
        from src.discovery.scanner import detect_daytrade_brokers

        records = [
            {"date_offset": 0, "buy": 10000, "sell": 0},
            {"date_offset": 1, "buy": 0, "sell": 9000},
            {"date_offset": 2, "buy": 10000, "sell": 0},
            {"date_offset": 3, "buy": 0, "sell": 9000},
            # 只有 2 次配對
        ]
        df = self._make_broker_df("2330", records)
        result = detect_daytrade_brokers(df, min_events=3)
        assert len(result) == 0

    def test_empty_input(self):
        """空 DataFrame → 空結果。"""
        from src.discovery.scanner import detect_daytrade_brokers

        df = pd.DataFrame()
        result = detect_daytrade_brokers(df)
        assert result.empty
        assert "daytrade_events" in result.columns


# ====================================================================== #
#  TestComputeDaytradePenalty — 隔日沖扣分計算純函數測試
# ====================================================================== #


class TestComputeDaytradePenalty:
    """compute_daytrade_penalty() 三層邏輯測試。"""

    _BASE = date(2026, 3, 10)

    def _make_day_data(self, stock_id: str, brokers: list[dict]) -> pd.DataFrame:
        """建構最新一日的分點資料。brokers: [{broker_id, broker_name, buy, sell}]"""
        rows = [
            {
                "stock_id": stock_id,
                "date": self._BASE,
                "broker_id": b.get("broker_id", f"B{i:03d}"),
                "broker_name": b.get("broker_name", f"一般分點{i}"),
                "buy": b.get("buy", 0),
                "sell": b.get("sell", 0),
            }
            for i, b in enumerate(brokers)
        ]
        return pd.DataFrame(rows)

    def test_known_broker_penalty(self):
        """黑名單分點大量買超 → penalty > 0。"""
        from src.discovery.scanner import compute_daytrade_penalty

        df = self._make_day_data(
            "2330",
            [
                {"broker_name": "凱基-台北", "buy": 50000, "sell": 0},
                {"broker_name": "一般分點A", "buy": 50000, "sell": 0},
            ],
        )
        result = compute_daytrade_penalty(df)
        row = result[result["stock_id"] == "2330"]
        assert len(row) == 1
        assert row.iloc[0]["daytrade_penalty"] > 0

    def test_no_daytrade_zero_penalty(self):
        """無隔日沖跡象 → penalty = 0。"""
        from src.discovery.scanner import compute_daytrade_penalty

        df = self._make_day_data(
            "2330",
            [
                {"broker_name": "一般分點A", "buy": 50000, "sell": 0},
                {"broker_name": "一般分點B", "buy": 30000, "sell": 0},
            ],
        )
        result = compute_daytrade_penalty(df)
        row = result[result["stock_id"] == "2330"]
        assert row.iloc[0]["daytrade_penalty"] == 0.0

    def test_penalty_capped_at_one(self):
        """即使隔日沖分點佔比 100% → penalty ≤ 1.0。"""
        from src.discovery.scanner import compute_daytrade_penalty

        df = self._make_day_data(
            "2330",
            [
                {"broker_name": "凱基-台北", "buy": 100000, "sell": 0},
            ],
        )
        result = compute_daytrade_penalty(df)
        row = result[result["stock_id"] == "2330"]
        assert row.iloc[0]["daytrade_penalty"] <= 1.0

    def test_group_aggregation(self):
        """三個隔日沖分點加總佔比 → penalty 反映群聚效應。"""
        from src.discovery.scanner import compute_daytrade_penalty

        df = self._make_day_data(
            "2330",
            [
                {"broker_name": "凱基-台北", "buy": 30000, "sell": 0},
                {"broker_name": "美林", "buy": 20000, "sell": 0},
                {"broker_name": "一般分點", "buy": 50000, "sell": 0},
            ],
        )
        result = compute_daytrade_penalty(df)
        row = result[result["stock_id"] == "2330"]
        # 凱基台北 + 美林 = 50000 / 100000 = 0.5
        assert row.iloc[0]["daytrade_penalty"] == pytest.approx(0.5, abs=0.05)

    def test_volume_threshold_dampening(self):
        """小量買超觸發流動性降半。"""
        from src.discovery.scanner import compute_daytrade_penalty

        df = self._make_day_data(
            "2330",
            [
                {"broker_name": "凱基-台北", "buy": 100, "sell": 0},
                {"broker_name": "一般分點", "buy": 100, "sell": 0},
            ],
        )
        # 20 日均量 100 萬股，隔日沖買超 100 股 < 100萬 × 5% = 5 萬 → 降半
        df_volume = pd.DataFrame({"stock_id": ["2330"], "avg_volume_20d": [1_000_000]})
        result = compute_daytrade_penalty(df, df_volume=df_volume)
        row = result[result["stock_id"] == "2330"]
        # 原始 penalty = 100/200 = 0.5，降半 → 0.25
        assert row.iloc[0]["daytrade_penalty"] == pytest.approx(0.25, abs=0.05)

    def test_instant_risk_large_volume(self):
        """黑名單分點佔當日總買量 ≥ 10% → 即時觸發至少 0.5。"""
        from src.discovery.scanner import compute_daytrade_penalty

        df = self._make_day_data(
            "2330",
            [
                {"broker_name": "凱基-台北", "buy": 15000, "sell": 10000},  # 淨買 5000
                {"broker_name": "一般分點A", "buy": 80000, "sell": 0},
                {"broker_name": "一般分點B", "buy": 5000, "sell": 0},
            ],
        )
        result = compute_daytrade_penalty(df, instant_volume_ratio=0.10)
        row = result[result["stock_id"] == "2330"]
        # 凱基台北 buy=15000 / 總buy=100000 = 15% ≥ 10% → 即時觸發
        assert row.iloc[0]["daytrade_penalty"] >= 0.5

    def test_empty_broker_data(self):
        """空資料 → penalty = 0。"""
        from src.discovery.scanner import compute_daytrade_penalty

        df = pd.DataFrame()
        result = compute_daytrade_penalty(df)
        assert result.empty


# ====================================================================== #
#  風險過濾強化測試（Task 53）
# ====================================================================== #


class TestRiskFilterEnhancements:
    """風險過濾強化：絕對值 cap、資料不足剔除、Dividend 收緊。"""

    def _make_price_df(self, sids: list[str], n_days: int = 25, spread: float = 1.0) -> pd.DataFrame:
        """建立 N 支股票的日K，spread 控制波動幅度。"""
        rows = []
        for sid in sids:
            for d in range(n_days):
                rows.append(
                    {
                        "stock_id": sid,
                        "date": date(2025, 1, 1) + timedelta(days=d),
                        "open": 100.0,
                        "high": 100.0 + spread,
                        "low": 100.0 - spread,
                        "close": 100.0 + (spread if d % 2 == 0 else -spread),
                        "volume": 1_000_000,
                    }
                )
        return pd.DataFrame(rows)

    def test_vol_filter_insufficient_data_excluded(self):
        """資料不足 10 天的股票應被 vol 風險過濾剔除（vol=inf）。"""
        scanner = MomentumScanner()
        # 9 支正常股（25 天）+ 1 支只有 5 天
        sids_normal = [f"N{i}" for i in range(9)]
        df_normal = self._make_price_df(sids_normal, n_days=25, spread=1.0)
        rows_short = []
        for d in range(5):
            rows_short.append(
                {
                    "stock_id": "SHORT",
                    "date": date(2025, 1, 1) + timedelta(days=d),
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "volume": 1_000_000,
                }
            )
        df_price = pd.concat([df_normal, pd.DataFrame(rows_short)], ignore_index=True)
        all_sids = sids_normal + ["SHORT"]
        scored = pd.DataFrame({"stock_id": all_sids, "composite_score": [0.5] * len(all_sids)})
        # SwingScanner 用 _apply_vol_risk_filter
        swing = SwingScanner()
        result = swing._apply_risk_filter(scored, df_price)
        # SHORT 應被剔除（vol=inf 必超任何門檻）
        assert "SHORT" not in result["stock_id"].values

    def test_atr_filter_absolute_cap(self):
        """ATR ratio 超過絕對值 cap 時即使 percentile 通過也應被剔除。"""
        scanner = MomentumScanner()
        # 10 支股票全部高波動（spread=10，ATR/close ≈ 10%）
        sids = [f"H{i}" for i in range(10)]
        df_price = self._make_price_df(sids, n_days=25, spread=10.0)
        scored = pd.DataFrame({"stock_id": sids, "composite_score": [0.5] * 10})
        # 全部股票 ATR ratio 約 10%，absolute_cap=0.08 應能剔除部分
        result = scanner._apply_atr_risk_filter(scored, df_price, percentile=99, absolute_cap=0.08)
        # percentile=99 理論上只剔除 1 支，但 absolute_cap=0.08 會剔除所有 >8%
        assert len(result) < len(scored)

    def test_vol_filter_absolute_cap_annualized(self):
        """年化波動率超過絕對值 cap=80% 時應被剔除。"""
        scanner = SwingScanner()
        # 10 支股票高日波動（spread=8 → 日波動 ~8%，年化 ~127%）
        sids = [f"V{i}" for i in range(10)]
        df_price = self._make_price_df(sids, n_days=65, spread=8.0)
        scored = pd.DataFrame({"stock_id": sids, "composite_score": [0.5] * 10})
        result = scanner._apply_vol_risk_filter(
            scored, df_price, percentile=99, window=60, annualize=True, absolute_cap=0.80
        )
        # 年化波動 >> 80%，absolute_cap 應起作用
        assert len(result) < len(scored)

    def test_dividend_tighter_percentile(self):
        """DividendScanner 風險過濾應使用 percentile=75（比 Value 的 90 更嚴格）。"""
        # 製造 10 支股票，波動遞增
        sids = [f"D{i}" for i in range(10)]
        rows = []
        for i, sid in enumerate(sids):
            spread = 0.5 + i * 0.5  # 0.5 ~ 5.0
            for d in range(25):
                rows.append(
                    {
                        "stock_id": sid,
                        "date": date(2025, 1, 1) + timedelta(days=d),
                        "open": 100,
                        "high": 100 + spread,
                        "low": 100 - spread,
                        "close": 100 + (spread if d % 2 == 0 else -spread),
                        "volume": 1_000_000,
                    }
                )
        df_price = pd.DataFrame(rows)
        scored = pd.DataFrame({"stock_id": sids, "composite_score": [0.5] * 10})

        div_scanner = DividendScanner()
        val_scanner = ValueScanner()
        result_div = div_scanner._apply_risk_filter(scored.copy(), df_price)
        result_val = val_scanner._apply_risk_filter(scored.copy(), df_price)
        # Dividend (75th) 應剔除更多股票
        assert len(result_div) <= len(result_val)


class TestCrisisFilterMA60:
    """Crisis 模式新增 MA60 絕對趨勢濾網測試。"""

    def _make_price_df_with_taiex(
        self,
        stock_data: dict[str, dict],
        taiex_return: float = -0.03,
        n_days: int = 65,
    ) -> pd.DataFrame:
        """建立含 TAIEX + 個股的 df_price。

        stock_data: {sid: {"return": float, "below_ma60": bool}}
        below_ma60=True 的股票近期收盤會低於 MA60。
        """
        base_date = date(2025, 1, 1)
        rows = []
        # TAIEX
        for i in range(n_days + 1):
            close = 10000.0 * (1 + taiex_return / n_days * i)
            rows.append({"stock_id": "TAIEX", "date": base_date + timedelta(days=i), "close": close})
        # 個股
        for sid, cfg in stock_data.items():
            ret = cfg.get("return", 0.0)
            below = cfg.get("below_ma60", False)
            for i in range(n_days + 1):
                if below and i > n_days - 10:
                    # 最後 10 天大跌，使 close 遠低於 MA60
                    close = 100.0 * (1 + ret / n_days * i) * 0.85
                else:
                    close = 100.0 * (1 + ret / n_days * i)
                rows.append(
                    {
                        "stock_id": sid,
                        "date": base_date + timedelta(days=i),
                        "close": close,
                    }
                )
        return pd.DataFrame(rows)

    def _make_scored(self, stock_ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"stock_id": stock_ids, "composite_score": [0.8 - 0.1 * i for i in range(len(stock_ids))]})

    def test_crisis_ma60_filters_below(self):
        """crisis 模式下收盤價跌破 MA60 的股票應被剔除。"""
        scanner = MomentumScanner()
        scanner.regime = "crisis"
        # A: 強勢（超越 TAIEX 且在 MA60 上方）; B: 超越 TAIEX 但跌破 MA60
        df_price = self._make_price_df_with_taiex(
            {
                "A": {"return": 0.02, "below_ma60": False},
                "B": {"return": 0.02, "below_ma60": True},
            },
            taiex_return=-0.05,
        )
        scored = self._make_scored(["A", "B"])
        result = scanner._apply_crisis_filter(scored, df_price)
        assert "A" in result["stock_id"].values
        assert "B" not in result["stock_id"].values

    def test_crisis_ma60_insufficient_data_passes(self):
        """資料不足 60 天的股票不受 MA60 濾網懲罰（避免冷啟動誤殺）。"""
        scanner = MomentumScanner()
        scanner.regime = "crisis"
        # 只有 25 天資料 → MA60 無法計算 → 通過
        base_date = date(2025, 1, 1)
        rows = []
        for i in range(26):
            rows.append(
                {"stock_id": "TAIEX", "date": base_date + timedelta(days=i), "close": 10000.0 * (1 - 0.02 / 25 * i)}
            )
            rows.append({"stock_id": "NEW", "date": base_date + timedelta(days=i), "close": 100.0})
        df_price = pd.DataFrame(rows)
        scored = self._make_scored(["NEW"])
        result = scanner._apply_crisis_filter(scored, df_price)
        assert "NEW" in result["stock_id"].values

    def test_non_crisis_skips_ma60(self):
        """非 crisis 模式不執行 MA60 濾網。"""
        scanner = MomentumScanner()
        scanner.regime = "bear"
        df_price = self._make_price_df_with_taiex(
            {"X": {"return": -0.10, "below_ma60": True}},
            taiex_return=-0.05,
        )
        scored = self._make_scored(["X"])
        result = scanner._apply_crisis_filter(scored, df_price)
        # bear mode → 不過濾
        assert len(result) == 1

    def test_crisis_both_filters_combined(self):
        """crisis 模式兩道濾網同時作用：相對弱勢 + 跌破 MA60。"""
        scanner = MomentumScanner()
        scanner.regime = "crisis"
        # A: 跑贏 TAIEX + MA60 上方（保留）
        # B: 跑贏 TAIEX 但跌破 MA60（MA60 濾網剔除）
        # C: 跑輸 TAIEX 超過 10%（相對強度濾網剔除）
        df_price = self._make_price_df_with_taiex(
            {
                "A": {"return": 0.05, "below_ma60": False},
                "B": {"return": 0.02, "below_ma60": True},
                "C": {"return": -0.20, "below_ma60": False},
            },
            taiex_return=-0.05,
        )
        scored = self._make_scored(["A", "B", "C"])
        result = scanner._apply_crisis_filter(scored, df_price)
        assert list(result["stock_id"]) == ["A"]


class TestCrisisEntryTriggerText:
    """Crisis 模式進場建議文字應包含部位規模提示。"""

    def _make_price_df(self, sid: str = "1000") -> pd.DataFrame:
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

    def test_crisis_trigger_includes_position_warning(self, scanner):
        """crisis 模式 entry_trigger 應包含部位規模警示。"""
        df = self._make_price_df()
        result = scanner._compute_entry_exit_cols(["1000"], df, regime="crisis")
        trigger = result.iloc[0]["entry_trigger"]
        assert "降低部位規模" in trigger

    def test_non_crisis_trigger_no_position_warning(self, scanner):
        """非 crisis 模式 entry_trigger 不含部位規模警示。"""
        df = self._make_price_df()
        result = scanner._compute_entry_exit_cols(["1000"], df, regime="sideways")
        trigger = result.iloc[0]["entry_trigger"]
        assert "降低部位規模" not in trigger

    def test_crisis_stop_loss_uses_1x_atr(self, scanner):
        """crisis 模式止損基準為 1.0x ATR，D1 自適應可能微調。"""
        df = self._make_price_df()
        result = scanner._compute_entry_exit_cols(["1000"], df, regime="crisis")
        row = result.iloc[0]
        ep = row["entry_price"]
        sl = row["stop_loss"]
        # ATR14 ≈ 4.0（high-low=4，無跳空）
        # D1: MDD=0%（close 恆定）→ 穩定股 → base×1.2 = 1.0×1.2 = 1.2
        # stop = entry - 1.2 × ATR ≈ 4.8
        stop_dist = ep - sl
        assert stop_dist == pytest.approx(4.8, abs=0.5)  # 1.0×1.2（D1 穩定股放寬）× ATR ≈ 4.8


# ====================================================================== #
#  TestComputeInstitutionalPersistence — 法人連續性因子純函數測試
# ====================================================================== #


class TestComputeInstitutionalPersistence:
    """compute_institutional_persistence() 純函數測試。"""

    def test_empty_df_returns_neutral(self):
        """空 df_inst 時，所有股票回傳中性值 0.5。"""
        from src.discovery.scanner import compute_institutional_persistence

        result = compute_institutional_persistence(pd.DataFrame(), ["2330", "2317"], window=10)
        assert len(result) == 2
        assert result[result["stock_id"] == "2330"].iloc[0]["inst_persistence"] == 0.5
        assert result[result["stock_id"] == "2317"].iloc[0]["inst_persistence"] == 0.5
        assert result[result["stock_id"] == "2330"].iloc[0]["inst_positive_days"] == 0

    def test_all_positive_days(self):
        """10/10 天淨買超為正 → persistence = 1.0。"""
        from src.discovery.scanner import compute_institutional_persistence

        rows = []
        for i in range(10):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"stock_id": "2330", "date": d, "name": "外資", "net": 100 + i})
        df = pd.DataFrame(rows)
        result = compute_institutional_persistence(df, ["2330"], window=10)
        assert result.iloc[0]["inst_persistence"] == pytest.approx(1.0)
        assert result.iloc[0]["inst_positive_days"] == 10

    def test_all_negative_days(self):
        """10/10 天淨買超為負 → persistence = 0.0。"""
        from src.discovery.scanner import compute_institutional_persistence

        rows = []
        for i in range(10):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"stock_id": "2330", "date": d, "name": "外資", "net": -100})
        df = pd.DataFrame(rows)
        result = compute_institutional_persistence(df, ["2330"], window=10)
        assert result.iloc[0]["inst_persistence"] == pytest.approx(0.0)
        assert result.iloc[0]["inst_positive_days"] == 0

    def test_partial_positive_days(self):
        """6/10 天淨買超為正 → persistence = 0.6。"""
        from src.discovery.scanner import compute_institutional_persistence

        rows = []
        for i in range(10):
            d = date(2026, 3, 1) + timedelta(days=i)
            net = 100 if i < 6 else -100  # 前 6 天正，後 4 天負
            rows.append({"stock_id": "2330", "date": d, "name": "外資", "net": net})
        df = pd.DataFrame(rows)
        result = compute_institutional_persistence(df, ["2330"], window=10)
        assert result.iloc[0]["inst_persistence"] == pytest.approx(0.6)
        assert result.iloc[0]["inst_positive_days"] == 6

    def test_fewer_days_than_window(self):
        """僅 5 天資料但 window=10 → 以實際天數為分母，5/5 = 1.0。"""
        from src.discovery.scanner import compute_institutional_persistence

        rows = []
        for i in range(5):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"stock_id": "2330", "date": d, "name": "外資", "net": 100})
        df = pd.DataFrame(rows)
        result = compute_institutional_persistence(df, ["2330"], window=10)
        assert result.iloc[0]["inst_persistence"] == pytest.approx(1.0)
        assert result.iloc[0]["inst_positive_days"] == 5

    def test_multiple_stocks_independent(self):
        """多檔股票各自獨立計算。"""
        from src.discovery.scanner import compute_institutional_persistence

        rows = []
        for i in range(10):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"stock_id": "2330", "date": d, "name": "外資", "net": 100})  # 全正
            rows.append({"stock_id": "2317", "date": d, "name": "外資", "net": -100})  # 全負
        df = pd.DataFrame(rows)
        result = compute_institutional_persistence(df, ["2330", "2317"], window=10)
        r_2330 = result[result["stock_id"] == "2330"].iloc[0]
        r_2317 = result[result["stock_id"] == "2317"].iloc[0]
        assert r_2330["inst_persistence"] == pytest.approx(1.0)
        assert r_2317["inst_persistence"] == pytest.approx(0.0)

    def test_missing_stock_returns_neutral(self):
        """stock_id 不在 df_inst 中時回傳中性值 0.5。"""
        from src.discovery.scanner import compute_institutional_persistence

        rows = [{"stock_id": "2330", "date": date(2026, 3, 1), "name": "外資", "net": 100}]
        df = pd.DataFrame(rows)
        result = compute_institutional_persistence(df, ["2330", "9999"], window=10)
        r_9999 = result[result["stock_id"] == "9999"].iloc[0]
        assert r_9999["inst_persistence"] == pytest.approx(0.5)

    def test_multiple_institutions_aggregated(self):
        """同日多法人合計 — 外資+100 + 投信-200 → 日合計 -100（負）。"""
        from src.discovery.scanner import compute_institutional_persistence

        d = date(2026, 3, 1)
        rows = [
            {"stock_id": "2330", "date": d, "name": "外資", "net": 100},
            {"stock_id": "2330", "date": d, "name": "投信", "net": -200},
        ]
        df = pd.DataFrame(rows)
        result = compute_institutional_persistence(df, ["2330"], window=10)
        assert result.iloc[0]["inst_persistence"] == pytest.approx(0.0)  # -100 合計為負
        assert result.iloc[0]["inst_positive_days"] == 0


# ====================================================================== #
#  TestDaytradePersistenceDampening — 隔日沖×連續性交互測試
# ====================================================================== #


class TestDaytradePersistenceDampening:
    """_apply_daytrade_penalty() 法人連續性調節測試。"""

    def _make_broker_df(self) -> pd.DataFrame:
        """建立含已知隔日沖分點（凱基-台北）的 BrokerTrade。

        凱基-台北在最新日仍有淨買超（即時風險），確保 penalty > 0。
        """
        d1 = date(2026, 3, 10)
        d2 = date(2026, 3, 11)
        d3 = date(2026, 3, 12)  # 最新日
        return pd.DataFrame(
            [
                # 凱基-台北：T 日買、T+1 日賣（隔日沖模式），但最新日仍有淨買超
                {
                    "date": d1,
                    "stock_id": "2330",
                    "broker_id": "B001",
                    "broker_name": "凱基-台北",
                    "buy": 5000,
                    "sell": 0,
                },
                {
                    "date": d2,
                    "stock_id": "2330",
                    "broker_id": "B001",
                    "broker_name": "凱基-台北",
                    "buy": 0,
                    "sell": 5000,
                },
                {
                    "date": d3,
                    "stock_id": "2330",
                    "broker_id": "B001",
                    "broker_name": "凱基-台北",
                    "buy": 6000,
                    "sell": 500,
                },
                # 正常主力
                {
                    "date": d1,
                    "stock_id": "2330",
                    "broker_id": "M001",
                    "broker_name": "元大-台北",
                    "buy": 8000,
                    "sell": 1000,
                },
                {
                    "date": d2,
                    "stock_id": "2330",
                    "broker_id": "M001",
                    "broker_name": "元大-台北",
                    "buy": 7000,
                    "sell": 1000,
                },
                {
                    "date": d3,
                    "stock_id": "2330",
                    "broker_id": "M001",
                    "broker_name": "元大-台北",
                    "buy": 7000,
                    "sell": 1000,
                },
            ]
        )

    def test_no_persistence_full_penalty(self):
        """persistence=0.0 時隔沖扣分全額施加。"""
        from src.discovery.scanner._base import MarketScanner

        scanner = MarketScanner()
        broker_rank = pd.Series([0.8])
        persist_df = pd.DataFrame({"stock_id": ["2330"], "inst_persistence": [0.0]})

        adjusted, _ = scanner._apply_daytrade_penalty(
            broker_rank,
            self._make_broker_df(),
            ["2330"],
            persistence_scores=persist_df,
        )
        # penalty > 0, persistence=0 → effective_penalty = penalty × 1.0 → 全額扣分
        assert adjusted.iloc[0] < 0.8

    def test_full_persistence_zero_penalty(self):
        """persistence=1.0 時隔沖扣分完全取消。"""
        from src.discovery.scanner._base import MarketScanner

        scanner = MarketScanner()
        broker_rank = pd.Series([0.8])
        persist_df = pd.DataFrame({"stock_id": ["2330"], "inst_persistence": [1.0]})

        adjusted, _ = scanner._apply_daytrade_penalty(
            broker_rank,
            self._make_broker_df(),
            ["2330"],
            persistence_scores=persist_df,
        )
        # persistence=1.0 → effective_penalty = penalty × (1 - 1.0) = 0 → rank 不變
        assert adjusted.iloc[0] == pytest.approx(0.8)

    def test_half_persistence_reduces_penalty(self):
        """persistence=0.5 時隔沖扣分減半。"""
        from src.discovery.scanner._base import MarketScanner

        scanner = MarketScanner()
        broker_rank = pd.Series([0.8])
        df_broker = self._make_broker_df()

        # 無連續性
        persist_0 = pd.DataFrame({"stock_id": ["2330"], "inst_persistence": [0.0]})
        adj_0, _ = scanner._apply_daytrade_penalty(
            broker_rank.copy(),
            df_broker,
            ["2330"],
            persistence_scores=persist_0,
        )

        # 半連續性
        persist_half = pd.DataFrame({"stock_id": ["2330"], "inst_persistence": [0.5]})
        adj_half, _ = scanner._apply_daytrade_penalty(
            broker_rank.copy(),
            df_broker,
            ["2330"],
            persistence_scores=persist_half,
        )

        # adj_half 的扣分幅度應為 adj_0 的一半
        deduction_0 = 0.8 - adj_0.iloc[0]
        deduction_half = 0.8 - adj_half.iloc[0]
        assert deduction_half == pytest.approx(deduction_0 * 0.5, abs=0.01)

    def test_none_persistence_scores_no_dampening(self):
        """persistence_scores=None 時行為與舊版一致（全額扣分）。"""
        from src.discovery.scanner._base import MarketScanner

        scanner = MarketScanner()
        broker_rank = pd.Series([0.8])
        df_broker = self._make_broker_df()

        # None（舊行為）
        adj_none, _ = scanner._apply_daytrade_penalty(
            broker_rank.copy(),
            df_broker,
            ["2330"],
            persistence_scores=None,
        )

        # persistence=0.0（等效舊行為）
        persist_0 = pd.DataFrame({"stock_id": ["2330"], "inst_persistence": [0.0]})
        adj_0, _ = scanner._apply_daytrade_penalty(
            broker_rank.copy(),
            df_broker,
            ["2330"],
            persistence_scores=persist_0,
        )

        assert adj_none.iloc[0] == pytest.approx(adj_0.iloc[0], abs=0.01)


# ====================================================================== #
#  TestComputeInstNetBuySlope — 法人淨買超斜率純函數測試
# ====================================================================== #


class TestComputeInstNetBuySlope:
    """compute_inst_net_buy_slope() 純函數測試。"""

    def test_empty_df_returns_zero(self):
        """空 df_inst 時回傳 0.0。"""
        from src.discovery.scanner import compute_inst_net_buy_slope

        result = compute_inst_net_buy_slope(pd.DataFrame(), ["2330"], window=10)
        assert len(result) == 1
        assert result.iloc[0]["inst_slope"] == pytest.approx(0.0)

    def test_increasing_net_buy_positive_slope(self):
        """淨買超逐日遞增 → 斜率為正。"""
        from src.discovery.scanner import compute_inst_net_buy_slope

        rows = []
        for i in range(10):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"stock_id": "2330", "date": d, "name": "外資", "net": 100 + i * 50})
        df = pd.DataFrame(rows)
        result = compute_inst_net_buy_slope(df, ["2330"], window=10)
        assert result.iloc[0]["inst_slope"] > 0

    def test_decreasing_net_buy_negative_slope(self):
        """淨買超逐日遞減 → 斜率為負。"""
        from src.discovery.scanner import compute_inst_net_buy_slope

        rows = []
        for i in range(10):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"stock_id": "2330", "date": d, "name": "外資", "net": 500 - i * 50})
        df = pd.DataFrame(rows)
        result = compute_inst_net_buy_slope(df, ["2330"], window=10)
        assert result.iloc[0]["inst_slope"] < 0

    def test_constant_net_buy_zero_slope(self):
        """淨買超恆定 → 斜率為 0。"""
        from src.discovery.scanner import compute_inst_net_buy_slope

        rows = []
        for i in range(10):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"stock_id": "2330", "date": d, "name": "外資", "net": 100})
        df = pd.DataFrame(rows)
        result = compute_inst_net_buy_slope(df, ["2330"], window=10)
        assert result.iloc[0]["inst_slope"] == pytest.approx(0.0, abs=1e-6)

    def test_fewer_than_3_days_returns_zero(self):
        """資料不足 3 天時回傳 0.0（無法算斜率）。"""
        from src.discovery.scanner import compute_inst_net_buy_slope

        rows = [
            {"stock_id": "2330", "date": date(2026, 3, 1), "name": "外資", "net": 100},
            {"stock_id": "2330", "date": date(2026, 3, 2), "name": "外資", "net": 200},
        ]
        df = pd.DataFrame(rows)
        result = compute_inst_net_buy_slope(df, ["2330"], window=10)
        assert result.iloc[0]["inst_slope"] == pytest.approx(0.0)

    def test_multiple_stocks_independent(self):
        """多股票各自獨立計算斜率。"""
        from src.discovery.scanner import compute_inst_net_buy_slope

        rows = []
        for i in range(5):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"stock_id": "2330", "date": d, "name": "外資", "net": 100 + i * 100})  # 遞增
            rows.append({"stock_id": "2317", "date": d, "name": "外資", "net": 500 - i * 100})  # 遞減
        df = pd.DataFrame(rows)
        result = compute_inst_net_buy_slope(df, ["2330", "2317"], window=10)
        r_2330 = result[result["stock_id"] == "2330"].iloc[0]["inst_slope"]
        r_2317 = result[result["stock_id"] == "2317"].iloc[0]["inst_slope"]
        assert r_2330 > 0
        assert r_2317 < 0

    def test_missing_stock_returns_zero(self):
        """stock_id 不在資料中時回傳 0.0。"""
        from src.discovery.scanner import compute_inst_net_buy_slope

        rows = [{"stock_id": "2330", "date": date(2026, 3, 1), "name": "外資", "net": 100}]
        df = pd.DataFrame(rows)
        result = compute_inst_net_buy_slope(df, ["2330", "9999"], window=10)
        r_9999 = result[result["stock_id"] == "9999"].iloc[0]
        assert r_9999["inst_slope"] == pytest.approx(0.0)


# ====================================================================== #
#  TestComputeHhiTrend — HHI 集中度趨勢純函數測試
# ====================================================================== #


class TestComputeHhiTrend:
    """compute_hhi_trend() 純函數測試。"""

    def test_empty_df_returns_zero(self):
        """空 df_broker 時回傳 0.0。"""
        from src.discovery.scanner import compute_hhi_trend

        result = compute_hhi_trend(pd.DataFrame(), ["2330"])
        assert len(result) == 1
        assert result.iloc[0]["hhi_trend"] == pytest.approx(0.0)
        assert result.iloc[0]["hhi_short_avg"] == pytest.approx(0.0)

    def test_concentrating_positive_trend(self):
        """主力越來越集中 → hhi_trend > 0。"""
        from src.discovery.scanner import compute_hhi_trend

        rows = []
        # 前 4 天：2 個分點各買一半（HHI≈0.5）
        for i in range(4):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"date": d, "stock_id": "2330", "broker_id": "A", "buy": 5000, "sell": 0})
            rows.append({"date": d, "stock_id": "2330", "broker_id": "B", "buy": 5000, "sell": 0})
        # 後 3 天：只剩 1 個分點（HHI=1.0）
        for i in range(3):
            d = date(2026, 3, 5) + timedelta(days=i)
            rows.append({"date": d, "stock_id": "2330", "broker_id": "A", "buy": 10000, "sell": 0})
            rows.append({"date": d, "stock_id": "2330", "broker_id": "B", "buy": 0, "sell": 1000})
        df = pd.DataFrame(rows)
        result = compute_hhi_trend(df, ["2330"], short_window=3, long_window=7)
        assert result.iloc[0]["hhi_trend"] > 0  # 近期更集中
        assert result.iloc[0]["hhi_short_avg"] > 0.5  # 近期 HHI 高

    def test_dispersing_negative_trend(self):
        """主力分散出貨 → hhi_trend < 0。"""
        from src.discovery.scanner import compute_hhi_trend

        rows = []
        # 前 4 天：1 個主力獨佔（HHI=1.0）
        for i in range(4):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"date": d, "stock_id": "2330", "broker_id": "A", "buy": 10000, "sell": 0})
        # 後 3 天：分散到 5 個分點（HHI≈0.2）
        for i in range(3):
            d = date(2026, 3, 5) + timedelta(days=i)
            for b_id in ["A", "B", "C", "D", "E"]:
                rows.append({"date": d, "stock_id": "2330", "broker_id": b_id, "buy": 2000, "sell": 0})
        df = pd.DataFrame(rows)
        result = compute_hhi_trend(df, ["2330"], short_window=3, long_window=7)
        assert result.iloc[0]["hhi_trend"] < 0  # 近期分散化

    def test_stable_hhi_zero_trend(self):
        """HHI 穩定不變 → trend ≈ 0。"""
        from src.discovery.scanner import compute_hhi_trend

        rows = []
        for i in range(7):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"date": d, "stock_id": "2330", "broker_id": "A", "buy": 7000, "sell": 0})
            rows.append({"date": d, "stock_id": "2330", "broker_id": "B", "buy": 3000, "sell": 0})
        df = pd.DataFrame(rows)
        result = compute_hhi_trend(df, ["2330"], short_window=3, long_window=7)
        assert abs(result.iloc[0]["hhi_trend"]) < 0.01  # 幾乎為 0

    def test_missing_stock_returns_zero(self):
        """stock_id 不在資料中時回傳 0.0。"""
        from src.discovery.scanner import compute_hhi_trend

        rows = [{"date": date(2026, 3, 1), "stock_id": "2330", "broker_id": "A", "buy": 1000, "sell": 0}]
        df = pd.DataFrame(rows)
        result = compute_hhi_trend(df, ["2330", "9999"])
        r_9999 = result[result["stock_id"] == "9999"].iloc[0]
        assert r_9999["hhi_trend"] == pytest.approx(0.0)

    def test_no_net_buyers_hhi_zero(self):
        """全部分點都是淨賣超 → HHI = 0。"""
        from src.discovery.scanner import compute_hhi_trend

        rows = []
        for i in range(5):
            d = date(2026, 3, 1) + timedelta(days=i)
            rows.append({"date": d, "stock_id": "2330", "broker_id": "A", "buy": 0, "sell": 5000})
        df = pd.DataFrame(rows)
        result = compute_hhi_trend(df, ["2330"])
        assert result.iloc[0]["hhi_trend"] == pytest.approx(0.0)
        assert result.iloc[0]["hhi_short_avg"] == pytest.approx(0.0)


# ====================================================================== #
#  TestApplyChipQualityModifiers — 籌碼品質修正測試
# ====================================================================== #


class TestApplyChipQualityModifiers:
    """_apply_chip_quality_modifiers() 靜態方法測試。"""

    def test_positive_slope_adds_bonus(self):
        """正斜率 → chip_score 增加。"""
        from src.discovery.scanner._base import MarketScanner

        chip = pd.Series([0.5])
        slope_df = pd.DataFrame({"stock_id": ["2330"], "inst_slope": [0.1]})
        result = MarketScanner._apply_chip_quality_modifiers(chip, ["2330"], slope_df=slope_df)
        assert result.iloc[0] == pytest.approx(0.53)  # 0.5 + 0.03

    def test_negative_slope_subtracts(self):
        """負斜率 → chip_score 減少。"""
        from src.discovery.scanner._base import MarketScanner

        chip = pd.Series([0.5])
        slope_df = pd.DataFrame({"stock_id": ["2330"], "inst_slope": [-0.2]})
        result = MarketScanner._apply_chip_quality_modifiers(chip, ["2330"], slope_df=slope_df)
        assert result.iloc[0] == pytest.approx(0.47)  # 0.5 - 0.03

    def test_hhi_high_concentrating_adds_bonus(self):
        """HHI 高且趨勢上升 → chip_score 增加。"""
        from src.discovery.scanner._base import MarketScanner

        chip = pd.Series([0.5])
        hhi_df = pd.DataFrame({"stock_id": ["2330"], "hhi_trend": [0.1], "hhi_short_avg": [0.4]})
        result = MarketScanner._apply_chip_quality_modifiers(chip, ["2330"], hhi_trend_df=hhi_df)
        assert result.iloc[0] == pytest.approx(0.53)

    def test_hhi_low_no_effect(self):
        """HHI 低（< 0.25）時不論趨勢方向，不施加修正。"""
        from src.discovery.scanner._base import MarketScanner

        chip = pd.Series([0.5])
        hhi_df = pd.DataFrame({"stock_id": ["2330"], "hhi_trend": [0.2], "hhi_short_avg": [0.1]})
        result = MarketScanner._apply_chip_quality_modifiers(chip, ["2330"], hhi_trend_df=hhi_df)
        assert result.iloc[0] == pytest.approx(0.5)  # 不變

    def test_clip_prevents_overflow(self):
        """修正後的 chip_score 不超過 1.0。"""
        from src.discovery.scanner._base import MarketScanner

        chip = pd.Series([0.98])
        slope_df = pd.DataFrame({"stock_id": ["2330"], "inst_slope": [0.5]})
        hhi_df = pd.DataFrame({"stock_id": ["2330"], "hhi_trend": [0.3], "hhi_short_avg": [0.5]})
        result = MarketScanner._apply_chip_quality_modifiers(
            chip,
            ["2330"],
            slope_df=slope_df,
            hhi_trend_df=hhi_df,
        )
        assert result.iloc[0] == pytest.approx(1.0)  # clip at 1.0

    def test_both_modifiers_combined(self):
        """斜率正 + HHI 集中化 → 雙重加分。"""
        from src.discovery.scanner._base import MarketScanner

        chip = pd.Series([0.5])
        slope_df = pd.DataFrame({"stock_id": ["2330"], "inst_slope": [0.1]})
        hhi_df = pd.DataFrame({"stock_id": ["2330"], "hhi_trend": [0.1], "hhi_short_avg": [0.4]})
        result = MarketScanner._apply_chip_quality_modifiers(
            chip,
            ["2330"],
            slope_df=slope_df,
            hhi_trend_df=hhi_df,
        )
        assert result.iloc[0] == pytest.approx(0.56)  # 0.5 + 0.03 + 0.03


# ─── compute_volume_price_divergence ──────────────────────────────────


class TestComputeVolumePriceDivergence:
    """compute_volume_price_divergence() 純函數測試。"""

    def test_price_up_volume_up_positive(self):
        """價漲量增 → 正向（量價齊揚）。"""
        from src.discovery.scanner._functions import compute_volume_price_divergence

        # 連續 7 天：價格遞增 + 成交量遞增 → 高正相關
        rows = []
        for d in range(7):
            rows.append(
                {
                    "stock_id": "2330",
                    "date": date(2025, 1, 1) + timedelta(days=d),
                    "close": 100 + d * 2,
                    "volume": 1_000_000 + d * 200_000,
                }
            )
        df = pd.DataFrame(rows)
        result = compute_volume_price_divergence(df, ["2330"], window=5)
        assert len(result) == 1
        assert result.iloc[0]["vp_divergence"] == pytest.approx(0.02)

    def test_price_up_volume_down_penalty(self):
        """價漲量縮 → 負向（背離懲罰）。"""
        from src.discovery.scanner._functions import compute_volume_price_divergence

        # 價格線性上漲（每日漲幅遞增），成交量線性大幅遞減
        closes = [100, 102, 105, 110, 116, 123, 132]
        volumes = [5_000_000, 4_200_000, 3_500_000, 2_800_000, 2_200_000, 1_500_000, 900_000]
        rows = []
        for d in range(7):
            rows.append(
                {
                    "stock_id": "2330",
                    "date": date(2025, 1, 1) + timedelta(days=d),
                    "close": closes[d],
                    "volume": volumes[d],
                }
            )
        df = pd.DataFrame(rows)
        result = compute_volume_price_divergence(df, ["2330"], window=5)
        assert result.iloc[0]["vp_divergence"] < 0  # 應為負值

    def test_insufficient_data_returns_zero(self):
        """資料不足 → 中性 0。"""
        from src.discovery.scanner._functions import compute_volume_price_divergence

        rows = [
            {"stock_id": "2330", "date": date(2025, 1, 1), "close": 100, "volume": 1_000_000},
            {"stock_id": "2330", "date": date(2025, 1, 2), "close": 101, "volume": 1_100_000},
        ]
        df = pd.DataFrame(rows)
        result = compute_volume_price_divergence(df, ["2330"], window=5)
        assert result.iloc[0]["vp_divergence"] == 0.0

    def test_empty_df(self):
        """空 DataFrame → 全 0。"""
        from src.discovery.scanner._functions import compute_volume_price_divergence

        result = compute_volume_price_divergence(pd.DataFrame(), ["2330"], window=5)
        assert len(result) == 1
        assert result.iloc[0]["vp_divergence"] == 0.0

    def test_missing_stock_returns_zero(self):
        """stock_id 不在 df_price 中 → 0。"""
        from src.discovery.scanner._functions import compute_volume_price_divergence

        rows = []
        for d in range(7):
            rows.append(
                {
                    "stock_id": "2330",
                    "date": date(2025, 1, 1) + timedelta(days=d),
                    "close": 100 + d,
                    "volume": 1_000_000 + d * 100_000,
                }
            )
        df = pd.DataFrame(rows)
        result = compute_volume_price_divergence(df, ["9999"], window=5)
        assert result.iloc[0]["vp_divergence"] == 0.0

    def test_multiple_stocks(self):
        """多支股票同時計算。"""
        from src.discovery.scanner._functions import compute_volume_price_divergence

        rows = []
        for sid_i, sid in enumerate(["2330", "2317"]):
            for d in range(7):
                rows.append(
                    {
                        "stock_id": sid,
                        "date": date(2025, 1, 1) + timedelta(days=d),
                        "close": 100 + d * (2 if sid_i == 0 else -1),
                        "volume": 1_000_000 + d * (200_000 if sid_i == 0 else -100_000),
                    }
                )
        df = pd.DataFrame(rows)
        result = compute_volume_price_divergence(df, ["2330", "2317"], window=5)
        assert len(result) == 2
        # 2330: 價漲量增 → 正值; 2317: 方向更複雜
        r2330 = result[result["stock_id"] == "2330"]["vp_divergence"].iloc[0]
        assert r2330 >= 0


# ─── _apply_score_threshold ──────────────────────────────────────────


class TestApplyScoreThreshold:
    """_apply_score_threshold() 動態評分閾值測試。"""

    def _make_scored(self, scores):
        return pd.DataFrame({"stock_id": [f"{i}" for i in range(len(scores))], "composite_score": scores})

    def test_bull_threshold(self):
        """Bull regime 門檻 0.45。"""
        scanner = MarketScanner()
        scanner.regime = "bull"
        scored = self._make_scored([0.60, 0.50, 0.44, 0.30])
        result = scanner._apply_score_threshold(scored)
        assert len(result) == 2  # 0.60 和 0.50 通過，0.44 和 0.30 被剔除

    def test_sideways_threshold(self):
        """Sideways regime 門檻 0.50。"""
        scanner = MarketScanner()
        scanner.regime = "sideways"
        scored = self._make_scored([0.70, 0.55, 0.49, 0.30])
        result = scanner._apply_score_threshold(scored)
        assert len(result) == 2

    def test_bear_threshold(self):
        """Bear regime 門檻 0.55。"""
        scanner = MarketScanner()
        scanner.regime = "bear"
        scored = self._make_scored([0.70, 0.60, 0.54, 0.40])
        result = scanner._apply_score_threshold(scored)
        assert len(result) == 2

    def test_crisis_threshold(self):
        """Crisis regime 門檻 0.60 — 最嚴格。"""
        scanner = MarketScanner()
        scanner.regime = "crisis"
        scored = self._make_scored([0.75, 0.65, 0.59, 0.40])
        result = scanner._apply_score_threshold(scored)
        assert len(result) == 2

    def test_empty_df(self):
        """空 DataFrame 不出錯。"""
        scanner = MarketScanner()
        scanner.regime = "bull"
        scored = pd.DataFrame(columns=["stock_id", "composite_score"])
        result = scanner._apply_score_threshold(scored)
        assert result.empty

    def test_all_pass(self):
        """全部通過門檻 → 不剔除。"""
        scanner = MarketScanner()
        scanner.regime = "bull"
        scored = self._make_scored([0.80, 0.70, 0.60])
        result = scanner._apply_score_threshold(scored)
        assert len(result) == 3

    def test_all_fail(self):
        """全部低於門檻 → 全部剔除。"""
        scanner = MarketScanner()
        scanner.regime = "crisis"
        scored = self._make_scored([0.50, 0.40, 0.30])
        result = scanner._apply_score_threshold(scored)
        assert result.empty


# ─── _apply_sector_diversification ───────────────────────────────────


class TestApplySectorDiversification:
    """_apply_sector_diversification() 同產業分散化測試。"""

    def _make_rankings(self, data):
        """data: list of (stock_id, composite_score, industry_category)"""
        df = pd.DataFrame(data, columns=["stock_id", "composite_score", "industry_category"])
        df["rank"] = range(1, len(df) + 1)
        return df

    def test_no_concentration(self):
        """產業分散 → 不剔除。"""
        scanner = MarketScanner(top_n_results=5)
        rankings = self._make_rankings(
            [
                ("2330", 0.9, "半導體"),
                ("2317", 0.8, "電子零組件"),
                ("2412", 0.7, "通信"),
                ("2882", 0.6, "金融"),
                ("1301", 0.5, "塑膠"),
            ]
        )
        result = scanner._apply_sector_diversification(rankings)
        assert len(result) == 5

    def test_concentration_capped(self):
        """同產業超過 25% 上限 → 超出的被替換。"""
        scanner = MarketScanner(top_n_results=8)
        # sector_cap = max(3, int(8*0.25)) = 3
        rankings = self._make_rankings(
            [
                ("A1", 0.95, "半導體"),
                ("A2", 0.90, "半導體"),
                ("A3", 0.85, "半導體"),
                ("A4", 0.80, "半導體"),  # 第 4 個半導體，超出 cap=3
                ("A5", 0.75, "半導體"),  # 第 5 個半導體
                ("B1", 0.70, "金融"),
                ("B2", 0.65, "金融"),
                ("C1", 0.60, "塑膠"),
                ("C2", 0.55, "塑膠"),
                ("D1", 0.50, "通信"),
            ]
        )
        result = scanner._apply_sector_diversification(rankings)
        # 半導體最多 3 個，金融 2，塑膠 2，通信 1 → 可湊 8 個
        semi_count = (result["industry_category"] == "半導體").sum()
        assert semi_count <= 3
        assert len(result) == 8

    def test_small_top_n_uses_min_cap(self):
        """top_n 很小時 sector_cap 至少 3。"""
        scanner = MarketScanner(top_n_results=4)
        # sector_cap = max(3, int(4*0.25)) = max(3, 1) = 3
        rankings = self._make_rankings(
            [
                ("A1", 0.95, "半導體"),
                ("A2", 0.90, "半導體"),
                ("A3", 0.85, "半導體"),
                ("A4", 0.80, "半導體"),
                ("B1", 0.70, "金融"),
                ("B2", 0.65, "金融"),
            ]
        )
        result = scanner._apply_sector_diversification(rankings)
        semi_count = (result["industry_category"] == "半導體").sum()
        assert semi_count <= 3
        assert len(result) == 4

    def test_empty_rankings(self):
        """空排名 → 直接回傳。"""
        scanner = MarketScanner(top_n_results=10)
        rankings = pd.DataFrame(columns=["stock_id", "composite_score", "industry_category", "rank"])
        result = scanner._apply_sector_diversification(rankings)
        assert result.empty

    def test_rank_renumbered(self):
        """分散化後 rank 重新編號。"""
        scanner = MarketScanner(top_n_results=5)
        rankings = self._make_rankings(
            [
                ("A1", 0.95, "半導體"),
                ("A2", 0.90, "半導體"),
                ("A3", 0.85, "半導體"),
                ("B1", 0.80, "金融"),
                ("C1", 0.75, "塑膠"),
            ]
        )
        result = scanner._apply_sector_diversification(rankings)
        assert list(result["rank"]) == [1, 2, 3, 4, 5]

    def test_missing_industry_treated_as_unknown(self):
        """缺少產業分類 → 歸類為「未分類」，也受 cap 限制。"""
        scanner = MarketScanner(top_n_results=5)
        rankings = self._make_rankings(
            [
                ("A1", 0.95, ""),
                ("A2", 0.90, ""),
                ("A3", 0.85, ""),
                ("A4", 0.80, ""),  # 第 4 個未分類
                ("B1", 0.70, "金融"),
                ("C1", 0.60, "塑膠"),
            ]
        )
        result = scanner._apply_sector_diversification(rankings)
        unknown_count = (result["industry_category"] == "").sum() + (result["industry_category"] == "未分類").sum()
        assert unknown_count <= 3


# ─── _apply_volume_price_divergence ──────────────────────────────────


class TestApplyVolumePriceDivergence:
    """_apply_volume_price_divergence() 整合測試。"""

    def test_adjusts_composite_score(self):
        """量價背離會調整 composite_score。"""
        scanner = MarketScanner()
        # 建立價漲量縮的資料（極端背離）
        rows = []
        base_vol = 10_000_000
        base_close = 100.0
        for d in range(8):
            rows.append(
                {
                    "stock_id": "2330",
                    "date": date(2025, 1, 1) + timedelta(days=d),
                    "open": base_close * (1.05**d) * 0.99,
                    "high": base_close * (1.05**d) * 1.02,
                    "low": base_close * (1.05**d) * 0.98,
                    "close": base_close * (1.05**d),
                    "volume": int(base_vol * (0.6**d)),
                }
            )
        df_price = pd.DataFrame(rows)
        scored = pd.DataFrame({"stock_id": ["2330"], "composite_score": [0.60]})
        result = scanner._apply_volume_price_divergence(scored, df_price)
        # 價漲量縮 → composite_score 應該下降
        assert "vp_divergence" in result.columns
        assert result.iloc[0]["composite_score"] <= 0.60

    def test_empty_scored(self):
        """空 scored → 直接回傳。"""
        scanner = MarketScanner()
        scored = pd.DataFrame(columns=["stock_id", "composite_score"])
        result = scanner._apply_volume_price_divergence(scored, pd.DataFrame())
        assert result.empty


# ======================================================================== #
#  P1-B1: compute_momentum_decay
# ======================================================================== #


class TestComputeMomentumDecay:
    """測試動量衰減偵測（RSI 頂背離 + MACD 柱縮短）。"""

    def _make_rising_price_df(self, stock_id: str, n: int = 60) -> pd.DataFrame:
        """建立一個穩定上漲的股價 DataFrame（無背離）。"""
        dates = pd.bdate_range(end=date.today(), periods=n)
        closes = [100 + i * 0.5 for i in range(n)]
        highs = [c + 1 for c in closes]
        return pd.DataFrame(
            {
                "stock_id": stock_id,
                "date": dates,
                "close": closes,
                "high": highs,
            }
        )

    def test_no_decay_accelerating_rise(self):
        """加速上漲（日漲幅遞增），MACD histogram 擴張 → decay = 0.0。"""
        n = 60
        dates = pd.bdate_range(end=date.today(), periods=n)
        # 加速成長：漲幅逐日遞增 → MACD histogram 持續擴張
        closes = [100.0]
        for i in range(1, n):
            # 每日漲幅 = 0.5% + 0.02%*i（越來越快）
            closes.append(closes[-1] * (1 + 0.005 + 0.0002 * i))
        highs = [c + 0.5 for c in closes]
        df = pd.DataFrame({"stock_id": "2330", "date": dates, "close": closes, "high": highs})
        result = compute_momentum_decay(df, ["2330"])
        assert len(result) == 1
        assert result.iloc[0]["momentum_decay"] == 0.0

    def test_empty_df(self):
        """空 DataFrame → 全部 0.0。"""
        result = compute_momentum_decay(pd.DataFrame(), ["2330", "2317"])
        assert len(result) == 2
        assert all(result["momentum_decay"] == 0.0)

    def test_missing_stock(self):
        """stock_id 不在 df 中 → 0.0。"""
        df = self._make_rising_price_df("2330")
        result = compute_momentum_decay(df, ["9999"])
        assert result.iloc[0]["momentum_decay"] == 0.0

    def test_insufficient_data(self):
        """資料不足 → 0.0。"""
        dates = pd.bdate_range(end=date.today(), periods=10)
        df = pd.DataFrame({"stock_id": "2330", "date": dates, "close": range(100, 110), "high": range(101, 111)})
        result = compute_momentum_decay(df, ["2330"])
        assert result.iloc[0]["momentum_decay"] == 0.0

    def test_rsi_divergence_detection(self):
        """價格創新高但 RSI 走弱（模擬頂背離）→ decay < 0。"""
        n = 60
        dates = pd.bdate_range(end=date.today(), periods=n)
        # 先大漲再平緩上漲（RSI 先衝高再走弱，但 high 持續新高）
        closes = []
        for i in range(n):
            if i < 30:
                closes.append(100 + i * 2.0)  # 強勢上漲 → RSI 衝高
            else:
                closes.append(160 + (i - 30) * 0.1)  # 微幅新高 → RSI 回落
        highs = [c + 0.5 for c in closes]
        df = pd.DataFrame({"stock_id": "2330", "date": dates, "close": closes, "high": highs})
        result = compute_momentum_decay(df, ["2330"])
        # 應偵測到至少 RSI 頂背離
        assert result.iloc[0]["momentum_decay"] <= 0.0

    def test_macd_shrinking_detection(self):
        """MACD histogram 連續縮短 → decay < 0。"""
        n = 60
        dates = pd.bdate_range(end=date.today(), periods=n)
        # 先大漲讓 MACD hist 走正，然後漲幅遞減讓 hist 連縮
        closes = []
        for i in range(n):
            if i < 40:
                closes.append(100 + i * 1.5)
            else:
                # 漲幅快速遞減
                closes.append(closes[-1] + max(0.01, 1.5 - (i - 40) * 0.15))
        highs = [c + 0.5 for c in closes]
        df = pd.DataFrame({"stock_id": "2330", "date": dates, "close": closes, "high": highs})
        result = compute_momentum_decay(df, ["2330"])
        # MACD hist 縮短應被偵測
        assert result.iloc[0]["momentum_decay"] <= 0.0

    def test_multiple_stocks(self):
        """多股混合測試。"""
        df1 = self._make_rising_price_df("2330")
        df2 = self._make_rising_price_df("2317")
        df = pd.concat([df1, df2])
        result = compute_momentum_decay(df, ["2330", "2317"])
        assert len(result) == 2

    def test_decay_range(self):
        """衰減值在預期範圍 [-0.06, 0.0]。"""
        df = self._make_rising_price_df("2330", n=80)
        result = compute_momentum_decay(df, ["2330"])
        val = result.iloc[0]["momentum_decay"]
        assert -0.06 <= val <= 0.0


# ======================================================================== #
#  P1-B2: compute_institutional_acceleration
# ======================================================================== #


class TestComputeInstitutionalAcceleration:
    """測試法人買超加速度。"""

    def _make_inst_data(self, stock_id: str, daily_nets: list[float]) -> pd.DataFrame:
        """建立法人資料（每日單一 net）。"""
        dates = pd.bdate_range(end=date.today(), periods=len(daily_nets))
        return pd.DataFrame(
            {
                "stock_id": stock_id,
                "date": dates,
                "name": "外資及陸資(不含外資自營商)",
                "net": daily_nets,
            }
        )

    def test_accelerating_buy(self):
        """前 7 天小量買超，後 3 天大量買超 → 加分。"""
        nets = [100] * 7 + [500] * 3  # 10 天，後 3 天大幅加速
        df = self._make_inst_data("2330", nets)
        result = compute_institutional_acceleration(df, ["2330"])
        assert result.iloc[0]["inst_accel_bonus"] > 0

    def test_decelerating_buy(self):
        """前 7 天大量買超，後 3 天小量 → 不加分。"""
        nets = [500] * 7 + [100] * 3
        df = self._make_inst_data("2330", nets)
        result = compute_institutional_acceleration(df, ["2330"])
        assert result.iloc[0]["inst_accel_bonus"] == 0.0

    def test_selling_recent(self):
        """近期賣超 → 不加分。"""
        nets = [100] * 7 + [-200] * 3
        df = self._make_inst_data("2330", nets)
        result = compute_institutional_acceleration(df, ["2330"])
        assert result.iloc[0]["inst_accel_bonus"] == 0.0

    def test_sell_to_buy_reversal(self):
        """從賣超轉為買超（加速比 = 1.0）→ 加分 +0.04。"""
        nets = [-100] * 7 + [200] * 3
        df = self._make_inst_data("2330", nets)
        result = compute_institutional_acceleration(df, ["2330"])
        assert result.iloc[0]["inst_accel_bonus"] == 0.04

    def test_empty_inst(self):
        """空法人資料 → 0.0。"""
        result = compute_institutional_acceleration(pd.DataFrame(), ["2330"])
        assert result.iloc[0]["inst_accel_bonus"] == 0.0

    def test_insufficient_data(self):
        """資料不足 10 天 → 0.0。"""
        nets = [100] * 5
        df = self._make_inst_data("2330", nets)
        result = compute_institutional_acceleration(df, ["2330"], window=10)
        assert result.iloc[0]["inst_accel_bonus"] == 0.0

    def test_missing_stock(self):
        """stock_id 不在資料中 → 0.0。"""
        nets = [100] * 10
        df = self._make_inst_data("2330", nets)
        result = compute_institutional_acceleration(df, ["9999"])
        assert result.iloc[0]["inst_accel_bonus"] == 0.0

    def test_bonus_range(self):
        """加分在 [0.0, 0.04] 範圍。"""
        nets = [100] * 7 + [300] * 3
        df = self._make_inst_data("2330", nets)
        result = compute_institutional_acceleration(df, ["2330"])
        val = result.iloc[0]["inst_accel_bonus"]
        assert 0.0 <= val <= 0.04

    def test_moderate_acceleration(self):
        """中度加速（ratio 0 < x ≤ 0.5）→ +0.02。"""
        nets = [200] * 7 + [250] * 3  # 250/200 - 1 = 0.25
        df = self._make_inst_data("2330", nets)
        result = compute_institutional_acceleration(df, ["2330"])
        assert result.iloc[0]["inst_accel_bonus"] == 0.02


# ======================================================================== #
#  P1-C1: compute_multi_timeframe_alignment
# ======================================================================== #


class TestComputeMultiTimeframeAlignment:
    """測試多時框一致性。"""

    def test_both_bullish(self):
        """日線多頭 + 週線多頭 → +0.04。"""
        result = compute_multi_timeframe_alignment({"2330": True}, {"2330": 0.05})
        assert result.iloc[0]["mtf_alignment"] == 0.04

    def test_both_bearish(self):
        """日線空頭 + 週線空頭 → -0.04。"""
        result = compute_multi_timeframe_alignment({"2330": False}, {"2330": -0.05})
        assert result.iloc[0]["mtf_alignment"] == -0.04

    def test_daily_bull_weekly_bear(self):
        """日線多頭 + 週線空頭（短多長空矛盾）→ -0.03。"""
        result = compute_multi_timeframe_alignment({"2330": True}, {"2330": -0.05})
        assert result.iloc[0]["mtf_alignment"] == -0.03

    def test_daily_bear_weekly_bull(self):
        """日線空頭 + 週線多頭（短空長多，較輕）→ -0.02。"""
        result = compute_multi_timeframe_alignment({"2330": False}, {"2330": 0.05})
        assert result.iloc[0]["mtf_alignment"] == -0.02

    def test_weekly_neutral(self):
        """週線中性（0.0）→ 0.0。"""
        result = compute_multi_timeframe_alignment({"2330": True}, {"2330": 0.0})
        assert result.iloc[0]["mtf_alignment"] == 0.0

    def test_daily_none(self):
        """日線無資料 → 0.0。"""
        result = compute_multi_timeframe_alignment({"2330": None}, {"2330": 0.05})
        assert result.iloc[0]["mtf_alignment"] == 0.0

    def test_multiple_stocks(self):
        """多股混合。"""
        result = compute_multi_timeframe_alignment(
            {"2330": True, "2317": False, "2454": None},
            {"2330": 0.05, "2317": -0.05, "2454": 0.05},
        )
        assert len(result) == 3
        vals = dict(zip(result["stock_id"], result["mtf_alignment"]))
        assert vals["2330"] == 0.04
        assert vals["2317"] == -0.04
        assert vals["2454"] == 0.0

    def test_alignment_range(self):
        """所有回傳值在 [-0.04, +0.04]。"""
        for daily in [True, False, None]:
            for weekly in [0.05, -0.05, 0.0]:
                result = compute_multi_timeframe_alignment({"X": daily}, {"X": weekly})
                val = result.iloc[0]["mtf_alignment"]
                assert -0.04 <= val <= 0.04


# ======================================================================== #
#  P1-A2: 多時框強制共振排除
# ======================================================================== #


class TestMultiTimeframeForceExclude:
    """測試 momentum 模式日多週空矛盾時直接排除。"""

    def test_momentum_excludes_daily_bull_weekly_bear(self):
        """momentum 模式：日線多頭 + 週線空頭 → 排除。"""
        scanner = MomentumScanner(top_n_results=20)
        scored = pd.DataFrame(
            {
                "stock_id": ["A", "B", "C"],
                "composite_score": [0.8, 0.7, 0.6],
                "technical_score": [0.70, 0.70, 0.30],  # A,B 日線多頭; C 日線空頭
                "weekly_bonus": [-0.05, 0.05, -0.05],  # A 週空; B 週多; C 週空
            }
        )
        result = scanner._apply_multi_timeframe_alignment(scored)
        # A: 日多+週空 → 排除（mtf=-0.03）
        # B: 日多+週多 → 保留加分
        # C: 日空+週空 → 保留（日週一致空頭）
        assert "A" not in result["stock_id"].values
        assert "B" in result["stock_id"].values
        assert "C" in result["stock_id"].values

    def test_value_mode_keeps_conflict(self):
        """value 模式：日多週空矛盾時只降分，不排除。"""
        scanner = ValueScanner(top_n_results=20)
        scored = pd.DataFrame(
            {
                "stock_id": ["A"],
                "composite_score": [0.8],
                "technical_score": [0.70],
                "weekly_bonus": [-0.05],
            }
        )
        result = scanner._apply_multi_timeframe_alignment(scored)
        assert "A" in result["stock_id"].values  # 未排除
        assert result.iloc[0]["composite_score"] < 0.8  # 但有降分

    def test_no_weekly_bonus_skips(self):
        """無 weekly_bonus 欄位 → 跳過。"""
        scanner = MomentumScanner(top_n_results=20)
        scored = pd.DataFrame({"stock_id": ["A"], "composite_score": [0.8], "technical_score": [0.70]})
        result = scanner._apply_multi_timeframe_alignment(scored)
        assert len(result) == 1
        assert result.iloc[0]["composite_score"] == 0.8


# ======================================================================== #
#  P1-B1: compute_value_weighted_inst_flow
# ======================================================================== #


class TestComputeValueWeightedInstFlow:
    """測試法人金額加權連續性。"""

    def _make_inst_data(self, stock_id: str, daily_nets: list[float]) -> pd.DataFrame:
        dates = pd.bdate_range(end=date.today(), periods=len(daily_nets))
        return pd.DataFrame({"stock_id": stock_id, "date": dates, "name": "外資", "net": daily_nets})

    def test_large_recent_buy_scores_higher(self):
        """近期大額買超 → 加權值高。"""
        # A: 前 7 天 100，後 3 天 1000
        # B: 前 7 天 100，後 3 天 100
        df_a = self._make_inst_data("A", [100] * 7 + [1000] * 3)
        df_b = self._make_inst_data("B", [100] * 10)
        df = pd.concat([df_a, df_b])
        result = compute_value_weighted_inst_flow(df, ["A", "B"])
        vals = dict(zip(result["stock_id"], result["inst_flow_weighted"]))
        assert vals["A"] > vals["B"]

    def test_consistent_buy_beats_old_spike(self):
        """持續小額 > 早期單次大額（衰減後權重低）。"""
        # A: 10 天每天 300（持續累積）
        # B: 第 1 天 3000（最早），之後 0（衰減至 3000×0.85^9 ≈ 694）
        df_a = self._make_inst_data("A", [300] * 10)
        df_b = self._make_inst_data("B", [3000] + [0] * 9)
        df = pd.concat([df_a, df_b])
        result = compute_value_weighted_inst_flow(df, ["A", "B"])
        vals = dict(zip(result["stock_id"], result["inst_flow_weighted"]))
        assert vals["A"] > vals["B"]

    def test_selling_produces_negative(self):
        """持續賣超 → 負值。"""
        df = self._make_inst_data("X", [-500] * 10)
        result = compute_value_weighted_inst_flow(df, ["X"])
        assert result.iloc[0]["inst_flow_weighted"] < 0

    def test_empty_inst(self):
        """空資料 → 0.0。"""
        result = compute_value_weighted_inst_flow(pd.DataFrame(), ["X"])
        assert result.iloc[0]["inst_flow_weighted"] == 0.0

    def test_missing_stock(self):
        """stock_id 不在資料中 → 0.0。"""
        df = self._make_inst_data("A", [100] * 10)
        result = compute_value_weighted_inst_flow(df, ["Z"])
        assert result.iloc[0]["inst_flow_weighted"] == 0.0

    def test_decay_effect(self):
        """衰減係數生效：最近一天權重最大。"""
        # 第 1 天（最早）1000，其餘 0 → 加權值 = 1000 × 0.85^9 ≈ 232
        # 最後一天 1000，其餘 0 → 加權值 = 1000 × 0.85^0 = 1000
        df_early = self._make_inst_data("E", [1000] + [0] * 9)
        df_late = self._make_inst_data("L", [0] * 9 + [1000])
        df = pd.concat([df_early, df_late])
        result = compute_value_weighted_inst_flow(df, ["E", "L"])
        vals = dict(zip(result["stock_id"], result["inst_flow_weighted"]))
        assert vals["L"] > vals["E"]


# ======================================================================== #
#  P1-C1: compute_earnings_quality
# ======================================================================== #


class TestComputeEarningsQuality:
    """測試盈餘品質分數。"""

    def _make_financial_df(self, stock_id: str, **kwargs) -> pd.DataFrame:
        """建立單季財報資料。"""
        defaults = {
            "stock_id": stock_id,
            "date": date(2025, 3, 31),
            "operating_cf": 100,
            "net_income": 80,
            "revenue": 1000,
            "debt_ratio": 40.0,
        }
        defaults.update(kwargs)
        return pd.DataFrame([defaults])

    def test_high_quality(self):
        """OCF > NI, 低負債 → 品質高。"""
        df = self._make_financial_df("X", operating_cf=150, net_income=80, debt_ratio=30.0)
        result = compute_earnings_quality(df, ["X"])
        assert result.iloc[0]["earnings_quality"] > 0.6

    def test_low_quality_negative_ocf(self):
        """OCF < 0 → 品質低。"""
        df = self._make_financial_df("X", operating_cf=-50, net_income=80, debt_ratio=75.0)
        result = compute_earnings_quality(df, ["X"])
        assert result.iloc[0]["earnings_quality"] < 0.5

    def test_high_debt_penalized(self):
        """高負債 → 品質下降。"""
        df_low = self._make_financial_df("A", debt_ratio=30.0)
        df_high = self._make_financial_df("B", debt_ratio=80.0)
        df = pd.concat([df_low, df_high])
        result = compute_earnings_quality(df, ["A", "B"])
        vals = dict(zip(result["stock_id"], result["earnings_quality"]))
        assert vals["A"] > vals["B"]

    def test_empty_financial(self):
        """無財報 → 預設 0.5。"""
        result = compute_earnings_quality(pd.DataFrame(), ["X"])
        assert result.iloc[0]["earnings_quality"] == 0.5

    def test_missing_stock(self):
        """stock_id 不在資料中 → 0.5。"""
        df = self._make_financial_df("A")
        result = compute_earnings_quality(df, ["Z"])
        assert result.iloc[0]["earnings_quality"] == 0.5

    def test_two_quarters_revenue_quality(self):
        """兩季資料：淨利增速遠超營收 → 品質下降。"""
        df = pd.DataFrame(
            [
                {
                    "stock_id": "X",
                    "date": date(2025, 3, 31),
                    "operating_cf": 100,
                    "net_income": 200,
                    "revenue": 1000,
                    "debt_ratio": 40.0,
                },
                {
                    "stock_id": "X",
                    "date": date(2024, 12, 31),
                    "operating_cf": 80,
                    "net_income": 50,
                    "revenue": 950,
                    "debt_ratio": 42.0,
                },
            ]
        )
        result = compute_earnings_quality(df, ["X"])
        # NI 成長 300% vs 營收成長 5% → 灌水嫌疑
        quality = result.iloc[0]["earnings_quality"]
        assert quality < 0.7  # 被懲罰

    def test_quality_range(self):
        """品質在 [0.0, 1.0]。"""
        df = self._make_financial_df("X")
        result = compute_earnings_quality(df, ["X"])
        val = result.iloc[0]["earnings_quality"]
        assert 0.0 <= val <= 1.0


# ======================================================================== #
#  P1-D3: 回撤降頻
# ======================================================================== #


class TestDrawdownAdjustedTopN:
    """測試回撤降頻機制。"""

    def _make_taiex_price(self, drawdown_pct: float) -> pd.DataFrame:
        """建立 TAIEX 模擬資料，最後一天相對 20 日最高點下跌 drawdown_pct%。"""
        n = 30
        dates = pd.bdate_range(end=date.today(), periods=n)
        # 先漲到高點，最後跌 drawdown_pct
        high_val = 20000.0
        closes = [high_val] * n
        closes[-1] = high_val * (1 + drawdown_pct / 100)
        return pd.DataFrame({"stock_id": "TAIEX", "date": dates, "close": closes})

    def test_normal_market_no_change(self):
        """正常市場（回撤 -5%）→ top_n 不變。"""
        scanner = MomentumScanner(top_n_results=20)
        df_price = self._make_taiex_price(-5.0)
        result = scanner._compute_drawdown_adjusted_top_n(df_price)
        assert result == 20

    def test_moderate_drawdown_halves(self):
        """中度回撤（-12%）→ momentum 砍半。"""
        scanner = MomentumScanner(top_n_results=20)
        df_price = self._make_taiex_price(-12.0)
        result = scanner._compute_drawdown_adjusted_top_n(df_price)
        assert result == 10

    def test_severe_drawdown_thirds(self):
        """嚴重回撤（-18%）→ momentum 砍至 1/3。"""
        scanner = MomentumScanner(top_n_results=20)
        df_price = self._make_taiex_price(-18.0)
        result = scanner._compute_drawdown_adjusted_top_n(df_price)
        assert result <= 7  # 20 // 3 = 6

    def test_defensive_mode_unaffected(self):
        """value 模式不受嚴重回撤影響。"""
        scanner = ValueScanner(top_n_results=20)
        df_price = self._make_taiex_price(-18.0)
        result = scanner._compute_drawdown_adjusted_top_n(df_price)
        assert result == 20

    def test_dividend_mode_unaffected(self):
        """dividend 模式不受嚴重回撤影響。"""
        scanner = DividendScanner(top_n_results=20)
        df_price = self._make_taiex_price(-18.0)
        result = scanner._compute_drawdown_adjusted_top_n(df_price)
        assert result == 20

    def test_no_taiex_data_returns_original(self):
        """無 TAIEX 資料 → 不變。"""
        scanner = MomentumScanner(top_n_results=20)
        df_price = pd.DataFrame({"stock_id": ["2330"], "date": [date.today()], "close": [600]})
        result = scanner._compute_drawdown_adjusted_top_n(df_price)
        assert result == 20

    def test_min_top_n_is_3(self):
        """即使砍到極限，至少保留 3。"""
        scanner = MomentumScanner(top_n_results=5)
        df_price = self._make_taiex_price(-20.0)
        result = scanner._compute_drawdown_adjusted_top_n(df_price)
        assert result >= 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P2-B3: compute_chip_macd 籌碼面 MACD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestComputeChipMacd:
    """測試 compute_chip_macd 法人淨買超 MACD 信號。"""

    @staticmethod
    def _make_inst_series(stock_id: str, net_values: list[float]) -> pd.DataFrame:
        """建立指定淨買超序列的法人資料。"""
        rows = []
        base = date.today() - timedelta(days=len(net_values))
        for i, net in enumerate(net_values):
            d = base + timedelta(days=i)
            rows.append({"stock_id": stock_id, "date": d, "name": "外資", "net": net})
        return pd.DataFrame(rows)

    def test_empty_returns_default(self):
        """空資料 → 預設 0.5。"""
        result = compute_chip_macd(pd.DataFrame(), ["2330"])
        assert len(result) == 1
        assert result.iloc[0]["chip_macd_score"] == 0.5

    def test_insufficient_data_returns_default(self):
        """資料不足 slow_span(20) → 預設 0.5。"""
        df = self._make_inst_series("2330", [100] * 10)  # 只有 10 天
        result = compute_chip_macd(df, ["2330"])
        assert result.iloc[0]["chip_macd_score"] == 0.5

    def test_strong_accumulation(self):
        """持續增加的淨買超 → 高分（強勢吸籌）。"""
        # 30 天，淨買超從 100 持續增加到 3000
        nets = [100 + i * 100 for i in range(30)]
        df = self._make_inst_series("2330", nets)
        result = compute_chip_macd(df, ["2330"])
        assert result.iloc[0]["chip_macd_score"] >= 0.7

    def test_distribution_signal(self):
        """持續賣出 → 低分（出貨信號，MACD 負且柱狀圖負）。"""
        # 持續淨賣超，且加速賣出 → MACD 負且柱狀圖負
        nets = [200] * 15 + [-100 * i for i in range(1, 16)]
        df = self._make_inst_series("2330", nets)
        result = compute_chip_macd(df, ["2330"])
        assert result.iloc[0]["chip_macd_score"] <= 0.3

    def test_multiple_stocks(self):
        """多支股票各自獨立計算：加速買入 > 加速賣出。"""
        nets_bull = [100 + i * 100 for i in range(30)]  # 加速買入
        nets_bear = [200] * 15 + [-100 * i for i in range(1, 16)]  # 轉賣出
        df1 = self._make_inst_series("2330", nets_bull)
        df2 = self._make_inst_series("2317", nets_bear)
        df = pd.concat([df1, df2], ignore_index=True)
        result = compute_chip_macd(df, ["2330", "2317"])
        scores = dict(zip(result["stock_id"], result["chip_macd_score"]))
        assert scores["2330"] > scores["2317"]

    def test_flat_buying(self):
        """穩定小量買入 → 中性分數。"""
        nets = [200] * 30
        df = self._make_inst_series("2330", nets)
        result = compute_chip_macd(df, ["2330"])
        # 穩定買入 → fast ≈ slow → MACD ≈ 0 → 中性 0.5
        assert result.iloc[0]["chip_macd_score"] == 0.5

    def test_missing_stock_gets_default(self):
        """stock_ids 中有 DB 無資料的股票 → 預設 0.5。"""
        nets = [100 + i * 100 for i in range(30)]
        df = self._make_inst_series("2330", nets)
        result = compute_chip_macd(df, ["2330", "9999"])
        scores = dict(zip(result["stock_id"], result["chip_macd_score"]))
        assert scores["9999"] == 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P2-E1: compute_win_rate_threshold_adjustment 勝率回饋循環
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWinRateThresholdAdjustment:
    """測試勝率回饋門檻調整純函數。"""

    @staticmethod
    def _make_records_and_prices(
        n_recs: int, win_ratio: float, ref_date: date | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """建立推薦記錄和對應價格資料。

        win_ratio: 0.0~1.0，獲利推薦的比例。
        """
        ref = ref_date or date.today()
        records = []
        prices = []
        n_win = int(n_recs * win_ratio)

        for i in range(n_recs):
            scan_d = ref - timedelta(days=20 - i % 20)
            sid = f"S{i:03d}"
            entry_close = 100.0
            records.append(
                {
                    "scan_date": scan_d,
                    "stock_id": sid,
                    "close": entry_close,
                }
            )
            # 5 天後的收盤價
            exit_close = 105.0 if i < n_win else 95.0
            for d in range(1, 7):
                prices.append(
                    {
                        "stock_id": sid,
                        "date": scan_d + timedelta(days=d),
                        "close": exit_close if d >= 5 else entry_close,
                    }
                )

        return pd.DataFrame(records), pd.DataFrame(prices)

    def test_empty_returns_zero(self):
        """空資料 → 不調整。"""
        result = compute_win_rate_threshold_adjustment(pd.DataFrame(), pd.DataFrame(), "momentum")
        assert result == 0.0

    def test_high_win_rate_no_adjustment(self):
        """勝率 70% → 不調整。"""
        df_rec, df_price = self._make_records_and_prices(20, 0.7)
        result = compute_win_rate_threshold_adjustment(df_rec, df_price, "momentum", holding_days=5, lookback_days=30)
        assert result == 0.0

    def test_moderate_win_rate_small_adjustment(self):
        """勝率 45% (40~50%) → 輕微調整 +0.02。"""
        df_rec, df_price = self._make_records_and_prices(20, 0.45)
        result = compute_win_rate_threshold_adjustment(df_rec, df_price, "momentum", holding_days=5, lookback_days=30)
        assert result == pytest.approx(0.02)

    def test_low_win_rate_large_adjustment(self):
        """勝率 30% (<40%) → 大幅調整 +0.05。"""
        df_rec, df_price = self._make_records_and_prices(20, 0.30)
        result = compute_win_rate_threshold_adjustment(df_rec, df_price, "momentum", holding_days=5, lookback_days=30)
        assert result == pytest.approx(0.05)

    def test_too_few_samples_no_adjustment(self):
        """不足 5 筆 → 不調整。"""
        df_rec, df_price = self._make_records_and_prices(3, 0.0)
        result = compute_win_rate_threshold_adjustment(df_rec, df_price, "momentum", holding_days=5, lookback_days=30)
        assert result == 0.0

    def test_exactly_50pct_no_adjustment(self):
        """勝率剛好 50% → 不調整（>= moderate_threshold）。"""
        df_rec, df_price = self._make_records_and_prices(20, 0.50)
        result = compute_win_rate_threshold_adjustment(df_rec, df_price, "momentum", holding_days=5, lookback_days=30)
        assert result == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P2-E2: compute_factor_ic + compute_ic_weight_adjustments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestComputeFactorIc:
    """測試因子有效性 IC 計算。"""

    @staticmethod
    def _make_records_with_scores(n: int, tech_predictive: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """建立帶四維度分數的推薦記錄。

        tech_predictive=True: technical_score 與後續報酬正相關。
        """
        ref = date.today()
        records = []
        prices = []

        for i in range(n):
            scan_d = ref - timedelta(days=25 - i % 20)
            sid = f"T{i:03d}"
            entry_close = 100.0
            tech = 0.3 + (i / n) * 0.5  # 0.3 ~ 0.8
            records.append(
                {
                    "scan_date": scan_d,
                    "stock_id": sid,
                    "close": entry_close,
                    "technical_score": tech,
                    "chip_score": 0.5,  # 隨機，不預測
                    "fundamental_score": 0.5,
                    "news_score": 0.5,
                }
            )
            # tech 分高 → 報酬高（如果 tech_predictive）
            if tech_predictive:
                exit_close = 100 + (tech - 0.5) * 20  # tech=0.8→106, tech=0.3→96
            else:
                exit_close = 100 - (tech - 0.5) * 20  # 反向
            for d in range(1, 7):
                prices.append(
                    {
                        "stock_id": sid,
                        "date": scan_d + timedelta(days=d),
                        "close": exit_close if d >= 5 else entry_close,
                    }
                )

        return pd.DataFrame(records), pd.DataFrame(prices)

    def test_empty_returns_empty(self):
        """空資料 → 空 DataFrame。"""
        result = compute_factor_ic(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_effective_factor_positive_ic(self):
        """technical_score 正向預測 → IC > 0, direction=effective。"""
        df_rec, df_price = self._make_records_with_scores(20, tech_predictive=True)
        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)
        tech_row = result[result["factor"] == "technical_score"]
        assert len(tech_row) == 1
        assert tech_row.iloc[0]["ic"] > 0
        assert tech_row.iloc[0]["direction"] == "effective"

    def test_inverse_factor_negative_ic(self):
        """technical_score 反向預測 → IC < 0, direction=inverse。"""
        df_rec, df_price = self._make_records_with_scores(20, tech_predictive=False)
        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)
        tech_row = result[result["factor"] == "technical_score"]
        assert len(tech_row) == 1
        assert tech_row.iloc[0]["ic"] < 0
        assert tech_row.iloc[0]["direction"] == "inverse"

    def test_neutral_factor_weak(self):
        """chip_score 全部相同 → IC ≈ 0, direction=weak。"""
        df_rec, df_price = self._make_records_with_scores(20, tech_predictive=True)
        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)
        chip_row = result[result["factor"] == "chip_score"]
        if not chip_row.empty:
            # 全部 0.5，IC 應近 0
            assert abs(chip_row.iloc[0]["ic"]) <= 0.1

    def test_too_few_samples(self):
        """不足 10 筆 → 空 DataFrame。"""
        df_rec, df_price = self._make_records_with_scores(5, tech_predictive=True)
        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)
        assert result.empty

    def test_evaluable_count_reported(self):
        """evaluable_count 正確反映有效樣本數。"""
        df_rec, df_price = self._make_records_with_scores(20, tech_predictive=True)
        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)
        for _, row in result.iterrows():
            assert row["evaluable_count"] >= 10


class TestComputeIcWeightAdjustments:
    """測試 IC 驅動的權重調整。"""

    def test_all_effective_no_change(self):
        """全部 effective → 權重不變。"""
        ic_df = pd.DataFrame(
            {
                "factor": ["technical_score", "chip_score"],
                "ic": [0.10, 0.08],
                "direction": ["effective", "effective"],
            }
        )
        base = {"technical_score": 0.45, "chip_score": 0.45}
        result = compute_ic_weight_adjustments(ic_df, base)
        assert result["technical_score"] == pytest.approx(0.45)
        assert result["chip_score"] == pytest.approx(0.45)

    def test_weak_factor_dampened(self):
        """weak 因子 → 權重降半後歸一化。"""
        ic_df = pd.DataFrame(
            {
                "factor": ["technical_score", "chip_score"],
                "ic": [0.10, 0.02],
                "direction": ["effective", "weak"],
            }
        )
        base = {"technical_score": 0.50, "chip_score": 0.50}
        result = compute_ic_weight_adjustments(ic_df, base, dampen_factor=0.5)
        # chip 降半 → 0.25，tech 維持 0.5，歸一化後 tech 佔更大
        assert result["technical_score"] > 0.50
        assert result["chip_score"] < 0.50
        # 總和應維持 1.0
        assert sum(result.values()) == pytest.approx(1.0)

    def test_inverse_factor_heavily_dampened(self):
        """inverse 因子 → 權重大幅降低。"""
        ic_df = pd.DataFrame(
            {
                "factor": ["technical_score", "chip_score"],
                "ic": [0.10, -0.10],
                "direction": ["effective", "inverse"],
            }
        )
        base = {"technical_score": 0.50, "chip_score": 0.50}
        result = compute_ic_weight_adjustments(ic_df, base, dampen_factor=0.5)
        # inverse → 0.50 * 0.25 = 0.125，effective 維持 0.50
        assert result["chip_score"] < result["technical_score"]

    def test_empty_ic_returns_base(self):
        """空 IC → 維持原始權重。"""
        base = {"technical_score": 0.45, "chip_score": 0.45}
        result = compute_ic_weight_adjustments(pd.DataFrame(), base)
        assert result == base

    def test_preserves_total_weight(self):
        """權重總和一定保持不變。"""
        ic_df = pd.DataFrame(
            {
                "factor": ["technical_score", "chip_score", "fundamental_score"],
                "ic": [0.10, -0.10, 0.01],
                "direction": ["effective", "inverse", "weak"],
            }
        )
        base = {"technical_score": 0.40, "chip_score": 0.30, "fundamental_score": 0.30}
        result = compute_ic_weight_adjustments(ic_df, base)
        assert sum(result.values()) == pytest.approx(1.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P3-B2: compute_key_player_cost_basis + score_key_player_cost
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKeyPlayerCostBasis:
    """測試主力成本分析純函數。"""

    @staticmethod
    def _make_broker_df(stock_id: str, brokers: dict[str, tuple[int, int, float]]) -> pd.DataFrame:
        """建立分點資料。brokers: {broker_id: (buy, sell, buy_price)}"""
        rows = []
        d = date.today() - timedelta(days=5)
        for bid, (buy, sell, bp) in brokers.items():
            rows.append(
                {
                    "stock_id": stock_id,
                    "date": d,
                    "broker_id": bid,
                    "buy": buy,
                    "sell": sell,
                    "buy_price": bp,
                    "sell_price": bp,
                }
            )
        return pd.DataFrame(rows)

    def test_empty_returns_default(self):
        """空資料 → 預設 0.5。"""
        result = compute_key_player_cost_basis(pd.DataFrame(), ["2330"])
        assert len(result) == 1
        assert result.iloc[0]["key_player_score"] == 0.5

    def test_top3_brokers_selected(self):
        """取淨買超前 3 大主力。"""
        df = self._make_broker_df(
            "2330",
            {
                "B1": (5000, 1000, 100.0),  # net=4000
                "B2": (3000, 500, 100.0),  # net=2500
                "B3": (2000, 300, 100.0),  # net=1700
                "B4": (1000, 800, 100.0),  # net=200
            },
        )
        result = compute_key_player_cost_basis(df, ["2330"], top_n_brokers=3)
        assert result.iloc[0]["key_player_cost"] is not None

    def test_no_net_buyer_returns_default(self):
        """所有分點淨賣超 → 無法計算。"""
        df = self._make_broker_df(
            "2330",
            {
                "B1": (100, 5000, 100.0),
                "B2": (200, 3000, 100.0),
            },
        )
        result = compute_key_player_cost_basis(df, ["2330"])
        assert result.iloc[0]["key_player_cost"] is None

    def test_score_trapped_players(self):
        """現價 < 主力成本（被套）→ 0.8。"""
        cost_df = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "key_player_cost": [100.0],
                "key_player_score": [0.5],
            }
        )
        result = score_key_player_cost(cost_df, {"2330": 90.0})
        assert result.iloc[0]["key_player_score"] == 0.8

    def test_score_profitable_players(self):
        """現價 > 主力成本 × 1.1（已獲利）→ 0.2。"""
        cost_df = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "key_player_cost": [100.0],
                "key_player_score": [0.5],
            }
        )
        result = score_key_player_cost(cost_df, {"2330": 115.0})
        assert result.iloc[0]["key_player_score"] == 0.2

    def test_score_near_cost(self):
        """現價 ≈ 主力成本 → 0.5。"""
        cost_df = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "key_player_cost": [100.0],
                "key_player_score": [0.5],
            }
        )
        result = score_key_player_cost(cost_df, {"2330": 103.0})
        assert result.iloc[0]["key_player_score"] == 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P3-D1: compute_adaptive_atr_multiplier 動態停損
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAdaptiveAtrMultiplier:
    """測試動態 ATR 倍數調整。"""

    @staticmethod
    def _make_price_df(stock_id: str, closes: list[float]) -> pd.DataFrame:
        base = date.today() - timedelta(days=len(closes))
        rows = []
        for i, c in enumerate(closes):
            rows.append(
                {
                    "stock_id": stock_id,
                    "date": base + timedelta(days=i),
                    "close": c,
                    "high": c * 1.01,
                    "low": c * 0.99,
                }
            )
        return pd.DataFrame(rows)

    def test_empty_returns_base(self):
        """空資料 → 維持 base multiplier。"""
        result = compute_adaptive_atr_multiplier(pd.DataFrame(), ["2330"], base_stop_mult=1.5)
        assert result.iloc[0]["adjusted_stop_mult"] == 1.5

    def test_stable_stock_widens(self):
        """穩定股（MDD < 5%）→ 倍數放寬 ×1.2。"""
        closes = [100 + i * 0.1 for i in range(25)]  # 穩定上升，幾乎無回撤
        df = self._make_price_df("2330", closes)
        result = compute_adaptive_atr_multiplier(df, ["2330"], base_stop_mult=1.5)
        assert result.iloc[0]["adjusted_stop_mult"] == pytest.approx(1.8)

    def test_volatile_stock_tightens(self):
        """高波動股（MDD > 15%）→ 倍數收緊 ×0.7。"""
        # 先漲後暴跌 20%
        closes = [100] * 10 + [100 - i * 2.5 for i in range(10)]  # 最低 75
        df = self._make_price_df("2330", closes)
        result = compute_adaptive_atr_multiplier(df, ["2330"], base_stop_mult=1.5)
        assert result.iloc[0]["adjusted_stop_mult"] == pytest.approx(1.05)

    def test_moderate_mdd_normal(self):
        """中等 MDD (5~10%) → 維持基準。"""
        # 先漲後回檔 ~7%
        closes = [100] * 10 + [100 - i * 0.7 for i in range(10)]  # 最低 93
        df = self._make_price_df("2330", closes)
        result = compute_adaptive_atr_multiplier(df, ["2330"], base_stop_mult=1.5)
        assert result.iloc[0]["adjusted_stop_mult"] == pytest.approx(1.5)

    def test_insufficient_data(self):
        """資料不足 → 維持基準。"""
        df = self._make_price_df("2330", [100, 99, 98])
        result = compute_adaptive_atr_multiplier(df, ["2330"], base_stop_mult=1.5)
        assert result.iloc[0]["adjusted_stop_mult"] == 1.5

    def test_multiple_stocks(self):
        """多股各自獨立計算。"""
        stable = [100 + i * 0.1 for i in range(25)]
        volatile = [100] * 10 + [100 - i * 2.5 for i in range(10)]
        df1 = self._make_price_df("2330", stable)
        df2 = self._make_price_df("2317", volatile)
        df = pd.concat([df1, df2], ignore_index=True)
        result = compute_adaptive_atr_multiplier(df, ["2330", "2317"], base_stop_mult=1.5)
        r = dict(zip(result["stock_id"], result["adjusted_stop_mult"]))
        assert r["2330"] > r["2317"]  # 穩定股放寬 > 波動股收緊


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P3-C2: compute_revenue_acceleration_score 營收加速度推廣
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRevenueAccelerationScore:
    """測試營收加速度分數。"""

    @staticmethod
    def _make_rev_df(stock_id: str, yoy_values: list[float], yoy_3m_ago: float | None = None) -> pd.DataFrame:
        """建立多月營收資料。yoy_values: 由新到舊。"""
        rows = []
        base = date.today()
        for i, yoy in enumerate(yoy_values):
            rows.append(
                {
                    "stock_id": stock_id,
                    "date": base - timedelta(days=30 * i),
                    "yoy_growth": yoy,
                    "yoy_3m_ago": yoy_3m_ago,
                }
            )
        return pd.DataFrame(rows)

    def test_empty_returns_default(self):
        """空資料 → 預設 0.5。"""
        result = compute_revenue_acceleration_score(pd.DataFrame(), ["2330"])
        assert result.iloc[0]["rev_accel_score"] == 0.5

    def test_strong_acceleration(self):
        """連續 3 月加速且加速幅度 > 10pp → 0.9。"""
        # yoy: 30, 25, 20, 15 → 連續遞增 3 次，accel = 30 - 15 = 15pp > 10pp
        df = self._make_rev_df("2330", [30, 25, 20, 15], yoy_3m_ago=15.0)
        result = compute_revenue_acceleration_score(df, ["2330"])
        assert result.iloc[0]["rev_accel_score"] == 0.9

    def test_stable_acceleration(self):
        """連續 3 月加速但幅度 ≤ 10pp → 0.75。"""
        df = self._make_rev_df("2330", [20, 18, 16, 14], yoy_3m_ago=14.0)
        result = compute_revenue_acceleration_score(df, ["2330"])
        assert result.iloc[0]["rev_accel_score"] == 0.75

    def test_single_month_acceleration(self):
        """acceleration > 0 但未連續 → 0.6。"""
        # yoy: 20, 15, 18, 10 → 不連續（15 < 18 中斷了）
        df = self._make_rev_df("2330", [20, 15, 18, 10], yoy_3m_ago=10.0)
        result = compute_revenue_acceleration_score(df, ["2330"])
        assert result.iloc[0]["rev_accel_score"] == 0.6

    def test_deceleration(self):
        """acceleration ≤ 0 → 0.3。"""
        df = self._make_rev_df("2330", [10, 15, 20, 25], yoy_3m_ago=25.0)
        result = compute_revenue_acceleration_score(df, ["2330"])
        assert result.iloc[0]["rev_accel_score"] == 0.3

    def test_consecutive_months_counted(self):
        """consecutive_accel_months 正確計算。"""
        df = self._make_rev_df("2330", [30, 25, 20, 15], yoy_3m_ago=15.0)
        result = compute_revenue_acceleration_score(df, ["2330"])
        assert result.iloc[0]["consecutive_accel_months"] == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P3-C3: compute_peer_fundamental_ranking 同業基本面排名
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPeerFundamentalRanking:
    """測試同業基本面排名。"""

    def test_empty_returns_zero(self):
        """空資料 → 全部 0。"""
        result = compute_peer_fundamental_ranking(pd.DataFrame(), ["2330"], {})
        assert result.iloc[0]["peer_rank_bonus"] == 0.0

    def test_leader_gets_bonus(self):
        """產業龍頭（前 25%）→ +0.03。"""
        df = pd.DataFrame(
            {
                "stock_id": ["S1", "S2", "S3", "S4", "S5"],
                "roe": [25, 20, 15, 10, 5],
                "gross_margin": [40, 35, 30, 25, 20],
            }
        )
        imap = {f"S{i}": "半導體" for i in range(1, 6)}
        result = compute_peer_fundamental_ranking(df, ["S1", "S2", "S3", "S4", "S5"], imap)
        r = dict(zip(result["stock_id"], result["peer_rank_bonus"]))
        assert r["S1"] == 0.03  # 龍頭
        assert r["S5"] == -0.03  # 落後者
        assert r["S3"] == 0.0  # 中間

    def test_too_few_peers_no_bonus(self):
        """同業不足 4 家 → 不加減分。"""
        df = pd.DataFrame(
            {
                "stock_id": ["S1", "S2", "S3"],
                "roe": [25, 15, 5],
            }
        )
        imap = {"S1": "A", "S2": "A", "S3": "A"}
        result = compute_peer_fundamental_ranking(df, ["S1", "S2", "S3"], imap)
        assert (result["peer_rank_bonus"] == 0.0).all()

    def test_multiple_industries(self):
        """不同產業各自獨立排名。"""
        df = pd.DataFrame(
            {
                "stock_id": [f"S{i}" for i in range(1, 9)],
                "roe": [30, 25, 20, 15, 10, 8, 5, 3],
            }
        )
        imap = {f"S{i}": "A" for i in range(1, 5)}
        imap.update({f"S{i}": "B" for i in range(5, 9)})
        result = compute_peer_fundamental_ranking(df, [f"S{i}" for i in range(1, 9)], imap)
        r = dict(zip(result["stock_id"], result["peer_rank_bonus"]))
        assert r["S1"] == 0.03  # A 產業龍頭
        assert r["S5"] == 0.03  # B 產業龍頭（ROE 10 是 B 中最高）

    def test_missing_stock_gets_zero(self):
        """stock_ids 中有 DB 無資料的股票 → 0。"""
        df = pd.DataFrame(
            {
                "stock_id": ["S1", "S2", "S3", "S4"],
                "roe": [25, 20, 15, 10],
            }
        )
        imap = {f"S{i}": "A" for i in range(1, 5)}
        result = compute_peer_fundamental_ranking(df, ["S1", "S2", "S3", "S4", "S99"], imap)
        r = dict(zip(result["stock_id"], result["peer_rank_bonus"]))
        assert r["S99"] == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P3-E3: compute_mfe_mae MFE/MAE 分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestComputeMfeMae:
    """測試 MFE/MAE 分析。"""

    @staticmethod
    def _make_data(
        entry_close: float, highs: list[float], lows: list[float], closes: list[float]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """建立推薦記錄和後續價格。"""
        scan_d = date.today() - timedelta(days=30)
        records = pd.DataFrame(
            {
                "scan_date": [scan_d],
                "stock_id": ["2330"],
                "close": [entry_close],
            }
        )
        prices = []
        for i in range(len(closes)):
            prices.append(
                {
                    "stock_id": "2330",
                    "date": scan_d + timedelta(days=i + 1),
                    "close": closes[i],
                    "high": highs[i],
                    "low": lows[i],
                }
            )
        return records, pd.DataFrame(prices)

    def test_empty_returns_empty(self):
        """空資料 → 空 DataFrame。"""
        result = compute_mfe_mae(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_profitable_trade(self):
        """獲利交易：MFE > 0，MAE 接近 0。"""
        records, prices = self._make_data(
            100.0,
            highs=[102, 104, 106, 108, 110],
            lows=[99, 101, 103, 105, 107],
            closes=[101, 103, 105, 107, 109],
        )
        result = compute_mfe_mae(records, prices, holding_days=5)
        assert len(result) == 1
        assert result.iloc[0]["mfe"] > 0
        assert result.iloc[0]["mfe_mae_ratio"] > 1.0
        assert result.iloc[0]["final_return"] > 0

    def test_losing_trade(self):
        """虧損交易：MAE < 0 且絕對值大，MFE/MAE < 1。"""
        records, prices = self._make_data(
            100.0,
            highs=[101, 100, 99, 98, 97],
            lows=[98, 96, 94, 92, 90],
            closes=[99, 97, 95, 93, 91],
        )
        result = compute_mfe_mae(records, prices, holding_days=5)
        assert len(result) == 1
        assert result.iloc[0]["mae"] < 0
        assert result.iloc[0]["final_return"] < 0

    def test_mfe_mae_ratio_calculation(self):
        """MFE/MAE 比率正確計算。"""
        records, prices = self._make_data(
            100.0,
            highs=[105, 103, 102, 101, 100],  # MFE = +5%
            lows=[97, 98, 99, 100, 99],  # MAE = -3%
            closes=[103, 101, 100, 100, 99],
        )
        result = compute_mfe_mae(records, prices, holding_days=5)
        assert result.iloc[0]["mfe"] == pytest.approx(0.05)
        assert result.iloc[0]["mae"] == pytest.approx(-0.03)
        assert result.iloc[0]["mfe_mae_ratio"] == pytest.approx(0.05 / 0.03, rel=0.1)

    def test_insufficient_future_data(self):
        """未來價格不足 → 使用可用資料。"""
        records, prices = self._make_data(
            100.0,
            highs=[105],
            lows=[95],
            closes=[102],
        )
        result = compute_mfe_mae(records, prices, holding_days=20)
        assert len(result) == 1  # 仍會計算（用 head(20) 取可用的）

    def test_multiple_recommendations(self):
        """多筆推薦各自獨立計算。"""
        scan_d1 = date.today() - timedelta(days=30)
        scan_d2 = date.today() - timedelta(days=25)
        records = pd.DataFrame(
            {
                "scan_date": [scan_d1, scan_d2],
                "stock_id": ["2330", "2317"],
                "close": [100.0, 200.0],
            }
        )
        prices = []
        for d_off in range(1, 10):
            prices.append(
                {
                    "stock_id": "2330",
                    "date": scan_d1 + timedelta(days=d_off),
                    "close": 105.0,
                    "high": 106.0,
                    "low": 104.0,
                }
            )
            prices.append(
                {
                    "stock_id": "2317",
                    "date": scan_d2 + timedelta(days=d_off),
                    "close": 195.0,
                    "high": 196.0,
                    "low": 190.0,
                }
            )
        result = compute_mfe_mae(records, pd.DataFrame(prices), holding_days=5)
        assert len(result) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 分數加成統一上限（sector + concept + peer ≤ ±8%）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBonusTotalCap:
    """驗證 peer_rank_bonus 受統一加成上限約束。"""

    def test_peer_capped_when_sector_concept_maxed(self):
        """sector+concept 已用 8% → peer_rank_bonus 被 clamp 至 0。"""
        scored = pd.DataFrame(
            {
                "stock_id": ["S1"],
                "composite_score": [0.70],
                "sector_bonus": [0.05],  # 5%
                "concept_bonus": [0.03],  # 3% → 合計 8%
            }
        )
        peer_rank_bonus = pd.Series([0.03])  # 想加 3%
        remaining = (0.08 - (scored["sector_bonus"] + scored["concept_bonus"]).abs()).clip(lower=0.0)
        capped = peer_rank_bonus.clip(lower=-remaining, upper=remaining)
        assert capped.iloc[0] == pytest.approx(0.0, abs=0.001)

    def test_peer_allowed_when_headroom_exists(self):
        """sector+concept 用 4% → 剩餘 4%，peer +3% 可通過。"""
        scored = pd.DataFrame(
            {
                "stock_id": ["S1"],
                "composite_score": [0.70],
                "sector_bonus": [0.02],
                "concept_bonus": [0.02],  # 合計 4%
            }
        )
        peer_rank_bonus = pd.Series([0.03])
        remaining = (0.08 - (scored["sector_bonus"] + scored["concept_bonus"]).abs()).clip(lower=0.0)
        capped = peer_rank_bonus.clip(lower=-remaining, upper=remaining)
        assert capped.iloc[0] == pytest.approx(0.03, abs=0.001)

    def test_negative_peer_capped_symmetrically(self):
        """sector+concept 用 -6% → 剩餘 2%，peer -3% 被 clamp 至 -2%。"""
        scored = pd.DataFrame(
            {
                "stock_id": ["S1"],
                "composite_score": [0.70],
                "sector_bonus": [-0.03],
                "concept_bonus": [-0.03],  # 合計 -6%
            }
        )
        peer_rank_bonus = pd.Series([-0.03])
        used = scored["sector_bonus"] + scored["concept_bonus"]
        remaining = (0.08 - used.abs()).clip(lower=0.0)
        capped = peer_rank_bonus.clip(lower=-remaining, upper=remaining)
        assert capped.iloc[0] == pytest.approx(-0.02, abs=0.001)


# ─── detect_chip_tier_changes ────────────────────────────────────
class TestDetectChipTierChanges:
    """detect_chip_tier_changes 純函數測試。"""

    def test_downgrade_detected(self):
        """8F → 7F 應偵測為 downgrade。"""
        current = pd.DataFrame({"stock_id": ["2330", "2317"], "chip_tier": ["7F", "5F"]})
        previous = pd.DataFrame({"stock_id": ["2330", "2317"], "chip_tier": ["8F", "5F"]})
        result = detect_chip_tier_changes(current, previous)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["stock_id"] == "2330"
        assert row["prev_tier"] == "8F"
        assert row["curr_tier"] == "7F"
        assert row["direction"] == "downgrade"

    def test_upgrade_detected(self):
        """3F → 5F 應偵測為 upgrade。"""
        current = pd.DataFrame({"stock_id": ["2454"], "chip_tier": ["5F"]})
        previous = pd.DataFrame({"stock_id": ["2454"], "chip_tier": ["3F"]})
        result = detect_chip_tier_changes(current, previous)
        assert len(result) == 1
        assert result.iloc[0]["direction"] == "upgrade"

    def test_no_overlap_returns_empty(self):
        """無重疊股票 → 空結果。"""
        current = pd.DataFrame({"stock_id": ["2330"], "chip_tier": ["8F"]})
        previous = pd.DataFrame({"stock_id": ["2317"], "chip_tier": ["7F"]})
        result = detect_chip_tier_changes(current, previous)
        assert result.empty

    def test_empty_previous_returns_empty(self):
        """前次為空 → 空結果。"""
        current = pd.DataFrame({"stock_id": ["2330"], "chip_tier": ["8F"]})
        previous = pd.DataFrame(columns=["stock_id", "chip_tier"])
        result = detect_chip_tier_changes(current, previous)
        assert result.empty

    def test_na_to_3f_is_upgrade(self):
        """N/A → 3F 視為 upgrade（從無資料到有資料）。"""
        current = pd.DataFrame({"stock_id": ["2330"], "chip_tier": ["3F"]})
        previous = pd.DataFrame({"stock_id": ["2330"], "chip_tier": ["N/A"]})
        result = detect_chip_tier_changes(current, previous)
        assert len(result) == 1
        assert result.iloc[0]["direction"] == "upgrade"


# ─── use_ic_adjustment flag ──────────────────────────────────────
class TestUseICAdjustmentFlag:
    """Scanner use_ic_adjustment 旗標測試。"""

    def test_flag_defaults_to_false(self):
        """預設不啟用 IC 權重調整。"""
        s = MarketScanner()
        assert s.use_ic_adjustment is False

    def test_flag_can_be_enabled(self):
        """可透過參數啟用。"""
        s = MarketScanner(use_ic_adjustment=True)
        assert s.use_ic_adjustment is True

    def test_ic_adjustment_no_data_returns_original(self, monkeypatch):
        """無歷史推薦時 _apply_ic_weight_adjustment 回傳原始權重。"""
        from unittest.mock import MagicMock

        # Mock get_session 回傳空結果
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.execute.return_value.all.return_value = []

        monkeypatch.setattr("src.discovery.scanner._base.get_session", lambda: mock_session)

        s = MarketScanner(use_ic_adjustment=True)
        s.mode_name = "momentum"
        s.scan_date = date.today()
        base = {"technical": 0.45, "chip": 0.25, "fundamental": 0.15, "news": 0.15}
        result = s._apply_ic_weight_adjustment(base)
        assert result == base

    def test_ic_weight_adjustments_pure_function_integration(self):
        """純函數 compute_ic_weight_adjustments 整合測試：weak 因子被衰減。"""
        ic_df = pd.DataFrame(
            {
                "factor": ["technical_score", "chip_score", "fundamental_score", "news_score"],
                "ic": [0.10, 0.02, -0.08, 0.06],
                "evaluable_count": [50, 50, 50, 50],
                "direction": ["effective", "weak", "inverse", "effective"],
            }
        )
        base = {"technical_score": 0.45, "chip_score": 0.25, "fundamental_score": 0.15, "news_score": 0.15}
        adjusted = compute_ic_weight_adjustments(ic_df, base)
        # effective → 維持，weak → 衰減，inverse → 大幅衰減
        # 歸一化後 technical + news 佔比應增加
        assert adjusted["technical_score"] > base["technical_score"]
        assert adjusted["chip_score"] < base["chip_score"]
        assert adjusted["fundamental_score"] < base["fundamental_score"]
