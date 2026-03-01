"""Discord 訊息格式化測試 — 4 個純函數。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.report.formatter import (
    format_daily_report,
    format_discovery_entry_exit,
    format_discovery_report,
    format_industry_report,
    format_strategy_rank,
)

# ================================================================
# format_daily_report
# ================================================================


class TestFormatDailyReport:
    def test_none_returns_default(self):
        result = format_daily_report(None)
        assert len(result) == 1
        assert "無資料" in result[0]

    def test_empty_df_returns_default(self):
        result = format_daily_report(pd.DataFrame())
        assert len(result) == 1
        assert "無資料" in result[0]

    def test_top_n_truncation(self):
        df = pd.DataFrame(
            {
                "rank": range(1, 21),
                "stock_id": [f"{i:04d}" for i in range(1, 21)],
                "close": [100.0] * 20,
                "composite_score": [0.8 - i * 0.01 for i in range(20)],
                "technical_score": [0.7] * 20,
                "chip_score": [0.6] * 20,
                "fundamental_score": [0.5] * 20,
                "ml_score": [0.4] * 20,
            }
        )
        result = format_daily_report(df, top_n=5)
        full_text = "\n".join(result)
        # 應該只包含前 5 筆
        assert "0006" not in full_text

    def test_output_is_list(self):
        df = pd.DataFrame(
            {
                "rank": [1],
                "stock_id": ["2330"],
                "close": [600.0],
                "composite_score": [0.85],
                "technical_score": [0.8],
                "chip_score": [0.7],
                "fundamental_score": [0.6],
                "ml_score": [0.5],
            }
        )
        result = format_daily_report(df)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_contains_stock_id(self):
        df = pd.DataFrame(
            {
                "rank": [1],
                "stock_id": ["2330"],
                "close": [600.0],
                "composite_score": [0.85],
                "technical_score": [0.8],
                "chip_score": [0.7],
                "fundamental_score": [0.6],
                "ml_score": [0.5],
            }
        )
        result = format_daily_report(df)
        assert "2330" in result[0]

    def test_extra_info_rsi(self):
        df = pd.DataFrame(
            {
                "rank": [1],
                "stock_id": ["2330"],
                "close": [600.0],
                "composite_score": [0.85],
                "technical_score": [0.8],
                "chip_score": [0.7],
                "fundamental_score": [0.6],
                "ml_score": [0.5],
                "rsi": [65.0],
                "foreign_net": [5000.0],
                "yoy_growth": [15.5],
            }
        )
        result = format_daily_report(df)
        full_text = "\n".join(result)
        assert "RSI" in full_text


# ================================================================
# format_strategy_rank
# ================================================================


class TestFormatStrategyRank:
    def test_none_returns_default(self):
        result = format_strategy_rank(None)
        assert "無資料" in result

    def test_empty_df_returns_default(self):
        result = format_strategy_rank(pd.DataFrame())
        assert "無資料" in result

    def test_basic_format(self):
        df = pd.DataFrame(
            {
                "rank": [1, 2],
                "stock_id": ["2330", "2317"],
                "strategy_name": ["sma_cross", "rsi_threshold"],
                "total_return": [15.5, 10.2],
                "sharpe_ratio": [1.5, 0.8],
                "max_drawdown": [-5.2, -8.1],
                "win_rate": [60.0, 55.0],
            }
        )
        result = format_strategy_rank(df, metric="sharpe")
        assert "2330" in result
        assert "sma_cross" in result
        assert "sharpe" in result

    def test_truncation_at_15(self):
        df = pd.DataFrame(
            {
                "rank": list(range(1, 21)),
                "stock_id": [f"{i:04d}" for i in range(1, 21)],
                "strategy_name": ["sma_cross"] * 20,
                "total_return": [10.0] * 20,
                "sharpe_ratio": [1.0] * 20,
                "max_drawdown": [-5.0] * 20,
                "win_rate": [50.0] * 20,
            }
        )
        result = format_strategy_rank(df)
        assert "及其他" in result

    def test_max_2000_chars(self):
        df = pd.DataFrame(
            {
                "rank": list(range(1, 21)),
                "stock_id": [f"{i:04d}" for i in range(1, 21)],
                "strategy_name": ["very_long_strategy_name_xxxx"] * 20,
                "total_return": [10.0] * 20,
                "sharpe_ratio": [1.0] * 20,
                "max_drawdown": [-5.0] * 20,
                "win_rate": [50.0] * 20,
            }
        )
        result = format_strategy_rank(df)
        assert len(result) <= 2000


# ================================================================
# format_industry_report
# ================================================================


class TestFormatIndustryReport:
    def test_none_returns_default(self):
        result = format_industry_report(None, None)
        assert len(result) == 1
        assert "無資料" in result[0]

    def test_empty_sector_df(self):
        result = format_industry_report(pd.DataFrame(), None)
        assert "無資料" in result[0]

    def test_basic_sector_ranking(self):
        sector_df = pd.DataFrame(
            {
                "rank": [1, 2],
                "industry": ["半導體", "金融"],
                "sector_score": [0.85, 0.72],
                "institutional_score": [0.9, 0.6],
                "momentum_score": [0.8, 0.7],
                "total_net": [50000, -10000],
                "avg_return_pct": [2.5, -0.3],
            }
        )
        result = format_industry_report(sector_df, None)
        assert len(result) >= 1
        assert "半導體" in result[0]

    def test_with_top_stocks(self):
        sector_df = pd.DataFrame(
            {
                "rank": [1],
                "industry": ["半導體"],
                "sector_score": [0.85],
                "institutional_score": [0.9],
                "momentum_score": [0.8],
                "total_net": [50000],
                "avg_return_pct": [2.5],
            }
        )
        stocks_df = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "stock_name": ["台積電"],
                "industry": ["半導體"],
                "close": [600.0],
                "foreign_net_sum": [10000],
            }
        )
        result = format_industry_report(sector_df, stocks_df)
        assert len(result) >= 2
        full_text = "\n".join(result)
        assert "2330" in full_text


# ================================================================
# format_discovery_report
# ================================================================


@dataclass
class MockDiscoveryResult:
    rankings: pd.DataFrame | None = None
    total_stocks: int = 0
    after_coarse: int = 0
    sector_summary: pd.DataFrame | None = None


class TestFormatDiscoveryReport:
    def test_empty_rankings(self):
        result_obj = MockDiscoveryResult(rankings=pd.DataFrame())
        result = format_discovery_report(result_obj)
        assert len(result) == 1
        assert "無符合條件" in result[0]

    def test_none_rankings(self):
        result_obj = MockDiscoveryResult(rankings=None)
        result = format_discovery_report(result_obj)
        assert "無符合條件" in result[0]

    def test_basic_format(self):
        rankings = pd.DataFrame(
            {
                "rank": [1, 2],
                "stock_id": ["2330", "2317"],
                "stock_name": ["台積電", "鴻海"],
                "close": [600.0, 100.0],
                "composite_score": [0.85, 0.72],
                "technical_score": [0.8, 0.7],
                "chip_score": [0.75, 0.65],
                "industry_category": ["半導體", "電子零組件"],
            }
        )
        result_obj = MockDiscoveryResult(
            rankings=rankings,
            total_stocks=5000,
            after_coarse=200,
        )
        result = format_discovery_report(result_obj)
        full_text = "\n".join(result)
        assert "2330" in full_text
        assert "5000" in full_text

    def test_with_sector_summary(self):
        rankings = pd.DataFrame(
            {
                "rank": [1],
                "stock_id": ["2330"],
                "stock_name": ["台積電"],
                "close": [600.0],
                "composite_score": [0.85],
                "technical_score": [0.8],
                "chip_score": [0.75],
                "industry_category": ["半導體"],
            }
        )
        sector_summary = pd.DataFrame(
            {
                "industry": ["半導體", "金融"],
                "count": [5, 3],
                "avg_score": [0.82, 0.65],
            }
        )
        result_obj = MockDiscoveryResult(
            rankings=rankings,
            total_stocks=5000,
            after_coarse=200,
            sector_summary=sector_summary,
        )
        result = format_discovery_report(result_obj)
        full_text = "\n".join(result)
        assert "產業分布" in full_text


# ================================================================
# format_discovery_entry_exit
# ================================================================


class TestFormatDiscoveryEntryExit:
    def _make_rankings_with_entry_exit(self):
        from datetime import date

        return pd.DataFrame(
            {
                "rank": [1, 2],
                "stock_id": ["2330", "2317"],
                "stock_name": ["台積電", "鴻海"],
                "close": [600.0, 100.0],
                "composite_score": [0.85, 0.72],
                "entry_price": [600.0, 100.0],
                "stop_loss": [570.0, 95.0],
                "take_profit": [660.0, 110.0],
                "entry_trigger": ["站上均線，低波動", "貼近均線"],
                "valid_until": [date(2026, 3, 8), date(2026, 3, 8)],
            }
        )

    def test_returns_string_with_entry_exit_section(self):
        rankings = self._make_rankings_with_entry_exit()
        result = format_discovery_entry_exit(rankings)
        assert isinstance(result, str)
        assert "進出場建議" in result

    def test_contains_stock_ids(self):
        rankings = self._make_rankings_with_entry_exit()
        result = format_discovery_entry_exit(rankings)
        assert "2330" in result

    def test_empty_rankings_returns_empty_string(self):
        result = format_discovery_entry_exit(pd.DataFrame())
        assert result == ""

    def test_missing_columns_returns_empty_string(self):
        """缺少進出場欄位時應回傳空字串。"""
        rankings = pd.DataFrame({"rank": [1], "stock_id": ["2330"], "close": [600.0]})
        result = format_discovery_entry_exit(rankings)
        assert result == ""

    def test_format_discovery_report_includes_entry_exit(self):
        """format_discovery_report 末尾應包含進出場建議區塊。"""
        from datetime import date

        rankings = pd.DataFrame(
            {
                "rank": [1],
                "stock_id": ["2330"],
                "stock_name": ["台積電"],
                "close": [600.0],
                "composite_score": [0.85],
                "technical_score": [0.8],
                "chip_score": [0.75],
                "industry_category": ["半導體"],
                "entry_price": [600.0],
                "stop_loss": [570.0],
                "take_profit": [660.0],
                "entry_trigger": ["站上均線，低波動"],
                "valid_until": [date(2026, 3, 8)],
            }
        )
        result_obj = MockDiscoveryResult(rankings=rankings, total_stocks=1000, after_coarse=50)
        msgs = format_discovery_report(result_obj)
        full_text = "\n".join(msgs)
        assert "進出場建議" in full_text
