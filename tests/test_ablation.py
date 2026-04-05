"""因子消融測試 — ablation.py 純函數測試。"""

from datetime import date

import numpy as np
import pandas as pd

from src.discovery.ablation import (
    AblationReport,
    DimensionAblationResult,
    SubFactorAblationResult,
    compute_ablation_performance,
    format_ablation_report,
    recompute_composite,
    redistribute_weights,
    run_dimension_ablation,
    run_sub_factor_ablation,
)

# ──────────────────────────────────────────────────────────
#  redistribute_weights
# ──────────────────────────────────────────────────────────


class TestRedistributeWeights:
    def test_basic_redistribution(self):
        """移除一個維度後權重按比例分配。"""
        w = {"technical": 0.4, "chip": 0.3, "fundamental": 0.2, "news": 0.1}
        result = redistribute_weights(w, "technical")
        # 剩餘 0.6 → chip=0.3/0.6=0.5, fundamental=0.2/0.6=0.333, news=0.1/0.6=0.167
        assert "technical" not in result
        assert abs(sum(result.values()) - 1.0) < 1e-10
        assert abs(result["chip"] - 0.5) < 1e-10

    def test_remove_nonexistent(self):
        """移除不存在的維度 → 原樣返回。"""
        w = {"a": 0.6, "b": 0.4}
        result = redistribute_weights(w, "c")
        assert result == w

    def test_all_equal_weights(self):
        """等權重移除一個 → 剩餘等分。"""
        w = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        result = redistribute_weights(w, "a")
        for v in result.values():
            assert abs(v - 1 / 3) < 1e-10

    def test_sum_to_one(self):
        """任意權重移除後，剩餘應歸一。"""
        w = {"technical": 0.30, "chip": 0.40, "fundamental": 0.20, "news": 0.10}
        for key in w:
            result = redistribute_weights(w, key)
            assert abs(sum(result.values()) - 1.0) < 1e-10


# ──────────────────────────────────────────────────────────
#  recompute_composite
# ──────────────────────────────────────────────────────────


class TestRecomputeComposite:
    def test_weighted_sum(self):
        """驗證加權合成分數計算。"""
        df = pd.DataFrame(
            {
                "stock_id": ["A", "B"],
                "technical_score": [0.8, 0.2],
                "chip_score": [0.6, 0.4],
            }
        )
        weights = {"technical": 0.6, "chip": 0.4}
        result = recompute_composite(df, weights)
        # A: 0.8*0.6 + 0.6*0.4 = 0.72
        assert abs(result.iloc[0] - 0.72) < 1e-10
        # B: 0.2*0.6 + 0.4*0.4 = 0.28
        assert abs(result.iloc[1] - 0.28) < 1e-10

    def test_missing_column_ignored(self):
        """權重中有欄位但 DataFrame 缺少 → 該維度不計分。"""
        df = pd.DataFrame({"stock_id": ["A"], "technical_score": [0.8]})
        weights = {"technical": 0.5, "chip": 0.5}
        result = recompute_composite(df, weights)
        assert abs(result.iloc[0] - 0.4) < 1e-10  # only technical

    def test_nan_filled_with_05(self):
        """NaN score 應填 0.5。"""
        df = pd.DataFrame(
            {
                "technical_score": [None],
                "chip_score": [0.8],
            }
        )
        weights = {"technical": 0.5, "chip": 0.5}
        result = recompute_composite(df, weights)
        # technical=0.5*0.5 + chip=0.8*0.5 = 0.65
        assert abs(result.iloc[0] - 0.65) < 1e-10


# ──────────────────────────────────────────────────────────
#  run_dimension_ablation
# ─────────────────────────────────────────────���────────────


def _make_scored_df(n=30):
    """建構測試用 scored DataFrame。"""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "stock_id": [f"S{i:03d}" for i in range(n)],
            "technical_score": rng.uniform(0, 1, n),
            "chip_score": rng.uniform(0, 1, n),
            "fundamental_score": rng.uniform(0, 1, n),
            "news_score": rng.uniform(0, 1, n),
        }
    )


class TestDimensionAblation:
    def test_returns_one_result_per_dimension(self):
        """應為每個維度產出一筆結果。"""
        df = _make_scored_df()
        weights = {"technical": 0.3, "chip": 0.4, "fundamental": 0.2, "news": 0.1}
        results = run_dimension_ablation(df, weights, top_n=10)
        assert len(results) == 4
        dims = {r.removed_dimension for r in results}
        assert dims == {"technical", "chip", "fundamental", "news"}

    def test_high_weight_dimension_has_more_impact(self):
        """權重較高的維度被移除後，排名相關性應較低（影響較大）。"""
        df = _make_scored_df(50)
        weights = {"technical": 0.7, "chip": 0.1, "fundamental": 0.1, "news": 0.1}
        results = run_dimension_ablation(df, weights, top_n=20)

        tech_result = next(r for r in results if r.removed_dimension == "technical")
        news_result = next(r for r in results if r.removed_dimension == "news")
        # 移除 70% 權重的 technical 應比移除 10% 的 news 影響更大
        assert tech_result.rank_correlation < news_result.rank_correlation

    def test_rank_correlation_range(self):
        """Spearman ρ 應在 [-1, 1] 之間。"""
        df = _make_scored_df()
        weights = {"technical": 0.3, "chip": 0.4, "fundamental": 0.2, "news": 0.1}
        results = run_dimension_ablation(df, weights)
        for r in results:
            assert -1.0 <= r.rank_correlation <= 1.0

    def test_dropped_and_added_consistency(self):
        """掉出和新增的數量應一致（top_n 不變）。"""
        df = _make_scored_df()
        weights = {"technical": 0.3, "chip": 0.4, "fundamental": 0.2, "news": 0.1}
        results = run_dimension_ablation(df, weights, top_n=10)
        for r in results:
            assert len(r.stocks_dropped) == len(r.stocks_added)

    def test_empty_df_returns_empty(self):
        """空 DataFrame → 空結果。"""
        results = run_dimension_ablation(pd.DataFrame(), {"technical": 1.0})
        assert results == []

    def test_top5_changes_populated(self):
        """每個消融結果應有前 5 名的變動。"""
        df = _make_scored_df()
        weights = {"technical": 0.3, "chip": 0.4, "fundamental": 0.2, "news": 0.1}
        results = run_dimension_ablation(df, weights)
        for r in results:
            assert len(r.top5_changes) == 5
            for c in r.top5_changes:
                assert "stock_id" in c
                assert "baseline_rank" in c
                assert "ablated_rank" in c


# ──────────────────────────────────────────────────────────
#  run_sub_factor_ablation
# ──────────────────────────────────────────────────────────


class TestSubFactorAblation:
    def _make_sub_df(self, n=20):
        rng = np.random.RandomState(42)
        return pd.DataFrame(
            {
                "stock_id": [f"S{i:03d}" for i in range(n)],
                "tech_ret5d": rng.uniform(0, 1, n),
                "tech_ret10d": rng.uniform(0, 1, n),
                "tech_breakout60d": rng.uniform(0, 1, n),
                "tech_vol_ratio": rng.uniform(0, 1, n),
            }
        )

    def test_one_result_per_factor(self):
        """每個子因子一筆結果。"""
        df = self._make_sub_df()
        results = run_sub_factor_ablation(df, "technical")
        assert len(results) == 4
        removed = {r.removed_factor for r in results}
        assert removed == {"tech_ret5d", "tech_ret10d", "tech_breakout60d", "tech_vol_ratio"}

    def test_correlation_range(self):
        """分數相關性在 [-1, 1]。"""
        df = self._make_sub_df()
        results = run_sub_factor_ablation(df, "technical")
        for r in results:
            assert -1.0 <= r.score_correlation <= 1.0

    def test_empty_df_returns_empty(self):
        results = run_sub_factor_ablation(pd.DataFrame(), "technical")
        assert results == []

    def test_single_factor_returns_empty(self):
        """只有一個子因子無法消融。"""
        df = pd.DataFrame({"stock_id": ["A"], "tech_ret5d": [0.5]})
        results = run_sub_factor_ablation(df, "technical")
        assert results == []

    def test_top_movers_populated(self):
        """top_movers 應有內容。"""
        df = self._make_sub_df()
        results = run_sub_factor_ablation(df, "technical")
        for r in results:
            assert len(r.top_movers) <= 3
            for m in r.top_movers:
                assert "stock_id" in m


# ──────────────────────────────────────────────────────────
#  compute_ablation_performance
# ──────────────────────────────────────────────────────────


class TestAblationPerformance:
    def _make_records(self):
        """建構歷史推薦記錄。"""
        rows = []
        for d in [date(2025, 6, 2), date(2025, 6, 9)]:
            for i, sid in enumerate(["P001", "P002", "P003"], 1):
                rows.append(
                    {
                        "scan_date": d,
                        "stock_id": sid,
                        "close": 100.0,
                        "rank": i,
                        "technical_score": 0.5 + i * 0.1,
                        "chip_score": 0.5,
                        "fundamental_score": 0.5,
                        "news_score": 0.5,
                    }
                )
        return pd.DataFrame(rows)

    def _make_prices(self):
        """建構價格資料。"""
        rows = []
        for sid in ["P001", "P002", "P003"]:
            for i, d in enumerate(pd.bdate_range("2025-06-03", periods=30)):
                rows.append({"stock_id": sid, "date": d.date(), "close": 100.0 + i * 0.5})
        return pd.DataFrame(rows)

    def test_baseline_row_present(self):
        """結果應包含 baseline 行。"""
        records = self._make_records()
        prices = self._make_prices()
        weights = {"technical": 0.3, "chip": 0.4, "fundamental": 0.2, "news": 0.1}
        result = compute_ablation_performance(records, prices, weights, holding_days=5, top_n=3)
        assert not result.empty
        baseline = result[result["removed_dimension"] == "(none — baseline)"]
        assert len(baseline) == 1
        assert baseline.iloc[0]["win_rate_delta"] == 0.0

    def test_ablation_rows_for_each_dimension(self):
        """每個維度應有一筆消融績效。"""
        records = self._make_records()
        prices = self._make_prices()
        weights = {"technical": 0.3, "chip": 0.4, "fundamental": 0.2, "news": 0.1}
        result = compute_ablation_performance(records, prices, weights, holding_days=5, top_n=3)
        # baseline + 4 dimensions = 5 rows
        assert len(result) == 5

    def test_empty_records_returns_empty(self):
        result = compute_ablation_performance(pd.DataFrame(), pd.DataFrame(), {"technical": 1.0})
        assert result.empty


# ──────────────────────────────────────────────────────────
#  format_ablation_report
# ──────────────────────────────────────────────────────────


class TestFormatAblationReport:
    def test_empty_report(self):
        """空報告不應 crash。"""
        report = AblationReport(mode="Momentum", regime="bull", baseline_top_n=20)
        text = format_ablation_report(report)
        assert "Momentum" in text
        assert "無足夠資料" in text

    def test_with_dimension_results(self):
        """有維度結果時應輸出影響力排序。"""
        report = AblationReport(
            mode="Momentum",
            regime="bull",
            baseline_top_n=20,
            dimension_results=[
                DimensionAblationResult(
                    removed_dimension="technical",
                    original_weights={"technical": 0.4, "chip": 0.4},
                    ablated_weights={"chip": 1.0},
                    rank_correlation=0.75,
                    mean_rank_shift=5.2,
                    max_rank_shift=15,
                    stocks_dropped=["A"],
                    stocks_added=["B"],
                    top5_changes=[{"stock_id": "X", "baseline_rank": 1, "ablated_rank": 3, "shift": 2}],
                ),
            ],
        )
        text = format_ablation_report(report)
        assert "technical" in text
        assert "影響力排序" in text
        assert "0.7500" in text

    def test_with_sub_factor_results(self):
        """有子因子結果時應顯示子因子消融段。"""
        report = AblationReport(
            mode="Momentum",
            regime="bull",
            baseline_top_n=20,
            sub_factor_results=[
                SubFactorAblationResult(
                    dimension="technical",
                    removed_factor="tech_ret5d",
                    score_correlation=0.92,
                    mean_score_change=0.03,
                    top_movers=[],
                ),
            ],
        )
        text = format_ablation_report(report)
        assert "tech_ret5d" in text
        assert "子因子消融" in text
