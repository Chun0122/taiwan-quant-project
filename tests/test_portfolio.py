"""投資組合回測測試 — _compute_weights、_compute_metrics、整合測試。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig
from src.backtest.portfolio import (
    PortfolioBacktestEngine,
    PortfolioConfig,
    PortfolioResultData,
    PortfolioTradeRecord,
)
from src.strategy.base import Strategy

# ================================================================
# Mock Strategy（不走 DB）
# ================================================================


class MockStrategy(Strategy):
    """測試用策略 — 直接注入資料和訊號。"""

    def __init__(self, stock_id: str, data: pd.DataFrame, signals: dict):
        self.stock_id = stock_id
        self._mock_data = data
        self._mock_signals = signals
        self._dividends = None

    @property
    def name(self) -> str:
        return "mock_strategy"

    def load_data(self) -> pd.DataFrame:
        return self._mock_data

    def generate_signals(self, data: pd.DataFrame) -> dict:
        return self._mock_signals


def _make_stock_data(n=20, base=100.0, stock_id="TEST"):
    """建立 N 天的股票資料。"""
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = [base + i * 0.5 for i in range(n)]
    df = pd.DataFrame(
        {
            "open": [c - 0.3 for c in close],
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [1_000_000] * n,
        },
        index=[d.date() for d in dates],
    )
    return df


# ================================================================
# _compute_weights
# ================================================================


class TestComputeWeights:
    def _make_engine(self, portfolio_config=None):
        engine = object.__new__(PortfolioBacktestEngine)
        engine.config = BacktestConfig()
        engine.portfolio_config = portfolio_config or PortfolioConfig()
        return engine

    def test_equal_weight_two_stocks(self):
        engine = self._make_engine()
        weights = engine._compute_weights(["2330", "2317"])
        assert weights["2330"] == pytest.approx(0.5)
        assert weights["2317"] == pytest.approx(0.5)

    def test_equal_weight_three_stocks(self):
        engine = self._make_engine()
        weights = engine._compute_weights(["A", "B", "C"])
        for w in weights.values():
            assert w == pytest.approx(1 / 3)

    def test_custom_weights_normalized(self):
        config = PortfolioConfig(
            allocation_method="custom",
            weights={"A": 3.0, "B": 1.0},
        )
        engine = self._make_engine(config)
        weights = engine._compute_weights(["A", "B"])
        assert weights["A"] == pytest.approx(0.75)
        assert weights["B"] == pytest.approx(0.25)

    def test_custom_missing_stock_zero(self):
        config = PortfolioConfig(
            allocation_method="custom",
            weights={"A": 1.0},
        )
        engine = self._make_engine(config)
        weights = engine._compute_weights(["A", "B"])
        assert weights["B"] == pytest.approx(0.0)


# ================================================================
# _compute_metrics
# ================================================================


class TestComputeMetrics:
    def _make_engine(self):
        engine = object.__new__(PortfolioBacktestEngine)
        engine.config = BacktestConfig()
        return engine

    def test_total_return(self):
        engine = self._make_engine()
        equity = [1_000_000, 1_050_000, 1_100_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 6, 30))
        assert metrics["total_return"] == pytest.approx(10.0)

    def test_max_drawdown(self):
        engine = self._make_engine()
        equity = [1_000_000, 1_100_000, 900_000, 1_050_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 6, 30))
        # peak=1_100_000, trough=900_000 → MDD = 200_000/1_100_000 ≈ 18.18%
        assert metrics["max_drawdown"] == pytest.approx(18.18, abs=0.1)

    def test_win_rate(self):
        engine = self._make_engine()
        trades = [
            PortfolioTradeRecord("A", date(2024, 1, 1), 100, pnl=50),
            PortfolioTradeRecord("B", date(2024, 1, 1), 100, pnl=-30),
            PortfolioTradeRecord("A", date(2024, 1, 1), 100, pnl=20),
        ]
        metrics = engine._compute_metrics([1_000_000], trades, date(2024, 1, 1), date(2024, 6, 30))
        assert metrics["win_rate"] == pytest.approx(66.67, abs=0.1)

    def test_profit_factor(self):
        engine = self._make_engine()
        trades = [
            PortfolioTradeRecord("A", date(2024, 1, 1), 100, pnl=100),
            PortfolioTradeRecord("B", date(2024, 1, 1), 100, pnl=-50),
        ]
        metrics = engine._compute_metrics([1_000_000], trades, date(2024, 1, 1), date(2024, 6, 30))
        assert metrics["profit_factor"] == pytest.approx(2.0)

    def test_no_trades_no_win_rate(self):
        engine = self._make_engine()
        metrics = engine._compute_metrics([1_000_000], [], date(2024, 1, 1), date(2024, 6, 30))
        assert metrics["win_rate"] is None
        assert metrics["profit_factor"] is None

    def test_sharpe_ratio_calculated(self):
        engine = self._make_engine()
        # 穩定上漲的 equity curve
        equity = [1_000_000 + i * 10_000 for i in range(20)]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 6, 30))
        assert metrics["sharpe_ratio"] is not None
        assert metrics["sharpe_ratio"] > 0


# ================================================================
# 整合測試
# ================================================================


class TestPortfolioBacktestIntegration:
    def test_basic_run(self):
        """基本的多股回測。"""
        data_a = _make_stock_data(n=10, base=100)
        data_b = _make_stock_data(n=10, base=200)
        dates = list(data_a.index)

        signals_a = {dates[0]: 1, dates[-1]: -1}  # 第一天買、最後天賣
        signals_b = {dates[0]: 1, dates[-1]: -1}

        strat_a = MockStrategy("A", data_a, signals_a)
        strat_b = MockStrategy("B", data_b, signals_b)

        engine = PortfolioBacktestEngine(
            strategies=[strat_a, strat_b],
            config=BacktestConfig(initial_capital=1_000_000),
        )
        result = engine.run()

        assert isinstance(result, PortfolioResultData)
        assert len(result.stock_ids) == 2
        assert result.total_trades >= 2  # 至少各買賣一次

    def test_force_close_at_end(self):
        """結束時應強制平倉。"""
        data = _make_stock_data(n=5, base=100)
        dates = list(data.index)
        # 只買不賣 → 應在結束時強制平倉
        signals = {dates[0]: 1}

        strat = MockStrategy("A", data, signals)
        engine = PortfolioBacktestEngine(
            strategies=[strat],
            config=BacktestConfig(initial_capital=1_000_000),
        )
        result = engine.run()
        assert result.total_trades >= 1
        assert any(t.exit_reason == "force_close" for t in result.trades)

    def test_equity_curve_length(self):
        """equity_curve 長度應等於交易日數。"""
        data = _make_stock_data(n=10, base=100)
        dates = list(data.index)
        signals = {dates[0]: 1, dates[-1]: -1}
        strat = MockStrategy("A", data, signals)

        engine = PortfolioBacktestEngine(
            strategies=[strat],
            config=BacktestConfig(initial_capital=1_000_000),
        )
        result = engine.run()
        assert len(result.equity_curve) == 10
