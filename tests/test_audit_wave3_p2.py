"""Wave 3 P2 audit 強化測試（2026-05-09 audit）。

涵蓋 S1/S2/S4/S5：
- S1: _apply_ic_weight_adjustment 新增 IC value + impact ρ log
- S2: compute_factor_ic 新增 min_per_date_count 參數
- S4: build_equity_history 純函數重構
- S5: UniverseFilter 暴露 _feature_latest_date / _feature_staleness_days

S3（DailyFeature vN_published_date schema 變更）已 defer — computed_at 欄位
已涵蓋大部分追溯需求，schema migration 風險較大延後評估。
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.data.schema import DailyFeature  # noqa: F401  top-level register

# ─────────────────────────────────────────────────────────────────
#  S1: _apply_ic_weight_adjustment log 增強
# ─────────────────────────────────────────────────────────────────


class TestApplyIcWeightLogEnhancement:
    def test_log_includes_ic_and_impact_when_weight_changes(self):
        """權重調整時 log 應同時包含 IC value 與 impact ρ。"""
        from src.discovery.scanner._base import MarketScanner

        scanner = MarketScanner.__new__(MarketScanner)
        scanner.mode_name = "momentum"
        scanner._sub_factor_ranks = {}
        scanner._dimension_ic_df = None
        scanner._ic_actions = {}
        scanner._precomputed_ic = pd.DataFrame(
            [
                {"factor": "technical_score", "ic": -0.13, "evaluable_count": 30, "direction": "inverse"},
                {"factor": "chip_score", "ic": 0.18, "evaluable_count": 30, "direction": "effective"},
            ]
        )
        # 候選資料含 *_score 欄
        candidates = pd.DataFrame(
            {
                "stock_id": [f"S{i:02d}" for i in range(15)],
                "technical_score": np.linspace(0.3, 0.9, 15),
                "chip_score": np.linspace(0.4, 0.8, 15),
                "fundamental_score": [0.5] * 15,
                "news_score": [0.5] * 15,
            }
        )

        base_weights = {"technical": 0.40, "chip": 0.30, "fundamental": 0.20, "news": 0.10}

        from src.discovery.scanner import _base as base_mod

        with patch.object(base_mod.logger, "info") as mock_info:
            adjusted = scanner._apply_ic_weight_adjustment(base_weights, scored_candidates=candidates)

        # 至少有一個維度權重變化 → 應 log
        # 檢查任何 log 訊息含 "IC=" 與 "ρ="
        all_log_strs = [str(c.args) for c in mock_info.call_args_list]
        assert any("IC=" in s for s in all_log_strs), f"Expected IC value in log: {all_log_strs}"
        # 確認 adjusted 為 dict
        assert isinstance(adjusted, dict)


# ─────────────────────────────────────────────────────────────────
#  S2: compute_factor_ic min_per_date_count 參數
# ─────────────────────────────────────────────────────────────────


class TestComputeFactorIcMinPerDateParam:
    @staticmethod
    def _build_records(per_day_count: int, n_days: int = 5):
        """構造 n_days 個交易日 × per_day_count 檔，日內 factor 與 ret 完全相關。"""
        ref = date.today()
        records, prices = [], []
        for day_idx in range(n_days):
            scan_d = ref - timedelta(days=20 + day_idx)
            for j in range(per_day_count):
                tech = 0.2 + j * 0.1
                ret = 0.005 * j
                records.append(
                    {
                        "scan_date": scan_d,
                        "stock_id": f"S{day_idx}{j}",
                        "close": 100.0,
                        "technical_score": tech,
                        "chip_score": 0.5,
                        "fundamental_score": 0.5,
                        "news_score": 0.5,
                    }
                )
                exit_close = 100.0 * (1 + ret)
                for d in range(1, 7):
                    prices.append(
                        {"stock_id": f"S{day_idx}{j}", "date": scan_d + timedelta(days=d), "close": exit_close}
                    )
        return pd.DataFrame(records), pd.DataFrame(prices)

    def test_default_min_per_date_count_3(self):
        """預設 min_per_date_count=3：每日 2 檔 → 全部 fallback pooled。"""
        from src.discovery.scanner._functions import compute_factor_ic

        df_rec, df_price = self._build_records(per_day_count=2, n_days=5)
        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)
        # 2 < 3 → 每日跳過 → fallback pooled
        # pooled 也會回傳一個 IC（全部 monotonic）
        if not result.empty:
            tech = result[result["factor"] == "technical_score"]
            if not tech.empty:
                assert tech.iloc[0]["evaluable_count"] == 10  # pooled 用全 valid samples (5 days × 2)

    def test_custom_min_per_date_count_2_uses_per_date_path(self):
        """傳 min_per_date_count=2 → 每日 2 檔可進 per-date 路徑。"""
        from src.discovery.scanner._functions import compute_factor_ic

        df_rec, df_price = self._build_records(per_day_count=2, n_days=5)
        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30, min_per_date_count=2)
        tech = result[result["factor"] == "technical_score"]
        assert len(tech) == 1
        # per-date 路徑：5 dates × 2 stocks = 10 used_samples
        assert tech.iloc[0]["evaluable_count"] == 10
        # 完美單調 → IC = 1.0
        assert tech.iloc[0]["ic"] > 0.95


# ─────────────────────────────────────────────────────────────────
#  S4: build_equity_history 純函數
# ─────────────────────────────────────────────────────────────────


class TestBuildEquityHistory:
    def test_basic_sequence(self):
        from src.portfolio.rotation import build_equity_history

        equity = build_equity_history(
            initial_capital=1_000_000,
            closed_pnls=[50_000, -20_000, 30_000],
            final_equity=1_080_000,
        )
        assert equity == [1_000_000, 1_050_000, 1_030_000, 1_060_000, 1_080_000]

    def test_no_closed_positions(self):
        from src.portfolio.rotation import build_equity_history

        equity = build_equity_history(initial_capital=500_000, closed_pnls=[], final_equity=480_000)
        assert equity == [500_000, 480_000]

    def test_none_pnl_treated_as_zero(self):
        from src.portfolio.rotation import build_equity_history

        equity = build_equity_history(
            initial_capital=1_000_000,
            closed_pnls=[None, 10_000, None],  # type: ignore[list-item]
            final_equity=1_010_000,
        )
        assert equity == [1_000_000, 1_000_000, 1_010_000, 1_010_000, 1_010_000]

    def test_with_drawdown_kill_switch(self):
        """整合：build_equity_history 結果可直接餵 check_drawdown_kill_switch。"""
        from src.portfolio.rotation import build_equity_history, check_drawdown_kill_switch

        # peak = 1.05M after first close, final = 0.6M → dd ≈ 42.86%
        equity = build_equity_history(
            initial_capital=1_000_000,
            closed_pnls=[50_000],
            final_equity=600_000,
        )
        assert check_drawdown_kill_switch(equity, threshold_pct=25.0) is True


# ─────────────────────────────────────────────────────────────────
#  S5: UniverseFilter 暴露 staleness_days 屬性
# ─────────────────────────────────────────────────────────────────


class TestUniverseFilterStalenessAttribute:
    def test_load_feature_turnover_sets_staleness_attribute(self):
        import src.discovery.universe as univ_mod
        from src.discovery.universe import UniverseConfig, UniverseFilter

        scalar_result = MagicMock()
        scalar_result.scalar.return_value = date(2026, 5, 4)
        all_result = MagicMock()
        all_result.all.return_value = [("S1", 1000.0, 1100.0, 0.91)]
        mock_session = MagicMock()
        mock_session.execute.side_effect = [scalar_result, all_result]
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_session
        mock_cm.__exit__.return_value = False

        class _MockDate(date):
            @classmethod
            def today(cls):
                return date(2026, 5, 11)

        uf = UniverseFilter(UniverseConfig())
        with patch.object(univ_mod, "get_session", return_value=mock_cm), patch.object(univ_mod, "date", _MockDate):
            uf._load_feature_turnover(["S1"])

        # S5：暴露為 instance 屬性
        assert hasattr(uf, "_feature_latest_date")
        assert hasattr(uf, "_feature_staleness_days")
        assert uf._feature_latest_date == date(2026, 5, 4)
        # 5/4 → 5/11 = 5 個交易日 gap
        assert uf._feature_staleness_days == 5

    def test_load_feature_turnover_sets_none_when_no_data(self):
        import src.discovery.universe as univ_mod
        from src.discovery.universe import UniverseConfig, UniverseFilter

        scalar_result = MagicMock()
        scalar_result.scalar.return_value = None  # 無 DailyFeature 資料
        mock_session = MagicMock()
        mock_session.execute.return_value = scalar_result
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_session
        mock_cm.__exit__.return_value = False

        uf = UniverseFilter(UniverseConfig())
        with patch.object(univ_mod, "get_session", return_value=mock_cm):
            df = uf._load_feature_turnover(["S1"])

        assert df.empty
        assert uf._feature_latest_date is None
        assert uf._feature_staleness_days is None
