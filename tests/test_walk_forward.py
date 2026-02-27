"""Walk-Forward 驗證引擎測試。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig
from src.backtest.walk_forward import WalkForwardEngine, WalkForwardResult
from src.strategy.base import Strategy

# ================================================================
# Mock Strategy（不走 DB）
# ================================================================


class MockWFStrategy(Strategy):
    """Walk-Forward 測試用策略。"""

    def __init__(self, stock_id: str, start_date: str, end_date: str, **kwargs):
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self._data = None
        self._dividends = None
        self.adjust_dividend = False

    @property
    def name(self) -> str:
        return "mock_wf"

    def load_data(self) -> pd.DataFrame:
        """產生 start_date~end_date 期間的假資料。"""
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        dates = pd.bdate_range(start, end)
        if len(dates) == 0:
            return pd.DataFrame()
        close = [100 + i * 0.1 for i in range(len(dates))]
        return pd.DataFrame(
            {
                "open": [c - 0.1 for c in close],
                "high": [c + 0.5 for c in close],
                "low": [c - 0.5 for c in close],
                "close": close,
                "volume": [1_000_000] * len(dates),
            },
            index=[d.date() for d in dates],
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """每 10 天交替買賣。"""
        signals = pd.Series(0, index=data.index)
        for i in range(len(data)):
            if i % 20 == 0:
                signals.iloc[i] = 1
            elif i % 20 == 10:
                signals.iloc[i] = -1
        return signals


# ================================================================
# _simulate_fold
# ================================================================


class TestSimulateFold:
    def _make_engine(self):
        engine = object.__new__(WalkForwardEngine)
        engine.config = BacktestConfig()
        return engine

    def test_buy_sell_trade(self):
        engine = self._make_engine()
        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = [d.date() for d in dates]
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "volume": [1000] * 5,
            },
            index=idx,
        )
        signals = pd.Series([1, 0, 0, 0, -1], index=idx)

        result = engine._simulate_fold(data, signals, 1_000_000)
        assert len(result["trades"]) == 1
        assert result["trades"][0].exit_reason == "signal"

    def test_force_close_at_end(self):
        engine = self._make_engine()
        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = [d.date() for d in dates]
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "volume": [1000] * 5,
            },
            index=idx,
        )
        # 只買不賣 → 強制平倉
        signals = pd.Series([1, 0, 0, 0, 0], index=idx)

        result = engine._simulate_fold(data, signals, 1_000_000)
        assert len(result["trades"]) == 1
        assert result["trades"][0].exit_reason == "force_close"

    def test_equity_curve_length(self):
        engine = self._make_engine()
        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = [d.date() for d in dates]
        data = pd.DataFrame(
            {
                "close": [100] * 5,
                "open": [100] * 5,
                "high": [101] * 5,
                "low": [99] * 5,
                "volume": [1000] * 5,
            },
            index=idx,
        )
        signals = pd.Series([0] * 5, index=idx)

        result = engine._simulate_fold(data, signals, 1_000_000)
        assert len(result["equity_curve"]) == 5

    def test_no_signal_no_trades(self):
        engine = self._make_engine()
        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = [d.date() for d in dates]
        data = pd.DataFrame(
            {
                "close": [100] * 5,
                "open": [100] * 5,
                "high": [101] * 5,
                "low": [99] * 5,
                "volume": [1000] * 5,
            },
            index=idx,
        )
        signals = pd.Series([0] * 5, index=idx)

        result = engine._simulate_fold(data, signals, 1_000_000)
        assert len(result["trades"]) == 0
        assert result["total_return"] == pytest.approx(0.0)


# ================================================================
# _compute_combined_metrics
# ================================================================


class TestComputeCombinedMetrics:
    def _make_engine(self):
        engine = object.__new__(WalkForwardEngine)
        engine.config = BacktestConfig()
        return engine

    def test_total_return(self):
        engine = self._make_engine()
        equity = [1_000_000, 1_050_000, 1_100_000]
        metrics = engine._compute_combined_metrics(equity, [], date(2024, 1, 1), date(2024, 6, 30))
        assert metrics["total_return"] == pytest.approx(10.0)

    def test_max_drawdown(self):
        engine = self._make_engine()
        equity = [1_000_000, 1_200_000, 900_000, 1_000_000]
        metrics = engine._compute_combined_metrics(equity, [], date(2024, 1, 1), date(2024, 6, 30))
        # peak=1_200_000, trough=900_000 → MDD = 25%
        assert metrics["max_drawdown"] == pytest.approx(25.0)


# ================================================================
# WalkForwardEngine.run() 整合
# ================================================================


class TestWalkForwardRun:
    def test_basic_run(self):
        engine = WalkForwardEngine(
            strategy_cls=MockWFStrategy,
            stock_id="TEST",
            start_date="2024-01-01",
            end_date="2025-12-31",
            train_window=100,
            test_window=50,
        )
        result = engine.run()
        assert isinstance(result, WalkForwardResult)
        assert result.total_folds >= 1
        assert len(result.folds) == result.total_folds

    def test_insufficient_data_raises(self):
        engine = WalkForwardEngine(
            strategy_cls=MockWFStrategy,
            stock_id="TEST",
            start_date="2024-01-01",
            end_date="2024-02-01",  # 太短
            train_window=252,
            test_window=63,
        )
        with pytest.raises(ValueError, match="資料不足"):
            engine.run()

    def test_fold_count(self):
        """驗證 fold 數量合理。"""
        engine = WalkForwardEngine(
            strategy_cls=MockWFStrategy,
            stock_id="TEST",
            start_date="2024-01-01",
            end_date="2025-12-31",
            train_window=100,
            test_window=50,
            step_size=50,
        )
        result = engine.run()
        # 約 500 交易日，(500 - 100) / 50 ≈ 8 folds
        assert result.total_folds >= 2

    def test_equity_curve_non_empty(self):
        engine = WalkForwardEngine(
            strategy_cls=MockWFStrategy,
            stock_id="TEST",
            start_date="2024-01-01",
            end_date="2025-06-30",
            train_window=100,
            test_window=50,
        )
        result = engine.run()
        assert len(result.equity_curve) > 0
