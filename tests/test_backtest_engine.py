"""測試 src/backtest/engine.py — 回測引擎計算邏輯。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    RiskConfig,
    TradeRecord,
)
from src.strategy.base import Strategy


class MockStrategy(Strategy):
    """用於測試的 mock 策略。"""

    def __init__(self, data: pd.DataFrame, signals: pd.Series):
        self._mock_data = data
        self._signals = signals
        self.stock_id = "TEST"

    @property
    def name(self) -> str:
        return "mock_strategy"

    def load_data(self) -> pd.DataFrame:
        return self._mock_data

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return self._signals


def _make_engine(
    data: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
    risk_config: RiskConfig | None = None,
) -> BacktestEngine:
    strategy = MockStrategy(data, signals)
    return BacktestEngine(strategy, config=config, risk_config=risk_config)


# ─── _compute_metrics ─────────────────────────────────────


class TestComputeMetrics:
    def test_total_return(self):
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000, 1_050_000, 1_100_000]
        trades = []
        metrics = engine._compute_metrics(equity, trades, date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["total_return"] == pytest.approx(10.0, abs=0.01)

    def test_max_drawdown(self):
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        # Peak at 1.2M, trough at 1.0M → 16.67% drawdown
        equity = [1_000_000, 1_200_000, 1_000_000, 1_100_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["max_drawdown"] == pytest.approx(16.67, abs=0.01)

    def test_win_rate(self):
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        trades = [
            TradeRecord(entry_date=date(2024, 1, 1), entry_price=100, pnl=500),
            TradeRecord(entry_date=date(2024, 2, 1), entry_price=100, pnl=-200),
            TradeRecord(entry_date=date(2024, 3, 1), entry_price=100, pnl=300),
        ]
        equity = [1_000_000, 1_000_500, 1_000_300, 1_000_600]
        metrics = engine._compute_metrics(equity, trades, date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["win_rate"] == pytest.approx(66.67, abs=0.01)

    def test_sharpe_ratio_calculated(self):
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000 + i * 1000 for i in range(100)]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["sharpe_ratio"] is not None
        assert metrics["sharpe_ratio"] > 0

    def test_no_trades_win_rate_none(self):
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["win_rate"] is None

    def test_profit_factor(self):
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        trades = [
            TradeRecord(entry_date=date(2024, 1, 1), entry_price=100, pnl=1000),
            TradeRecord(entry_date=date(2024, 2, 1), entry_price=100, pnl=-500),
        ]
        equity = [1_000_000, 1_001_000, 1_000_500]
        metrics = engine._compute_metrics(equity, trades, date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["profit_factor"] == pytest.approx(2.0, abs=0.01)


# ─── _compute_kelly_fraction ──────────────────────────────


class TestComputeKellyFraction:
    def test_insufficient_trades_returns_fallback(self):
        engine = _make_engine(pd.DataFrame(), pd.Series())
        trades = [TradeRecord(entry_date=date(2024, 1, 1), entry_price=100, pnl=100)] * 3
        result = engine._compute_kelly_fraction(trades)
        assert result == 0.1

    def test_known_kelly_value(self):
        engine = _make_engine(
            pd.DataFrame(),
            pd.Series(),
            risk_config=RiskConfig(kelly_fraction=1.0),
        )
        # 60% win rate, avg_win=200, avg_loss=100 → W/L=2
        # kelly = 0.6 - 0.4/2 = 0.4
        trades = [TradeRecord(entry_date=date(2024, 1, i + 1), entry_price=100, pnl=200) for i in range(6)] + [
            TradeRecord(entry_date=date(2024, 2, i + 1), entry_price=100, pnl=-100) for i in range(4)
        ]
        result = engine._compute_kelly_fraction(trades)
        assert result == pytest.approx(0.4, abs=0.01)

    def test_all_wins_returns_capped(self):
        engine = _make_engine(
            pd.DataFrame(),
            pd.Series(),
            risk_config=RiskConfig(kelly_fraction=1.0),
        )
        # All wins, no losses → fallback 0.1
        trades = [TradeRecord(entry_date=date(2024, 1, i + 1), entry_price=100, pnl=100) for i in range(10)]
        result = engine._compute_kelly_fraction(trades)
        assert result == 0.1


# ─── _compute_atr ─────────────────────────────────────────


class TestComputeAtr:
    def test_atr_calculation(self):
        dates = pd.bdate_range("2024-01-01", periods=20)
        data = pd.DataFrame(
            {
                "high": [110] * 20,
                "low": [90] * 20,
                "close": [100] * 20,
            },
            index=dates.date,
        )
        engine = _make_engine(data, pd.Series(), risk_config=RiskConfig(atr_period=5))
        # TR = max(H-L, |H-prevC|, |L-prevC|) = max(20, 10, 10) = 20
        atr = engine._compute_atr(data, dates[10].date())
        assert atr is not None
        assert atr == pytest.approx(20.0, abs=0.1)

    def test_insufficient_data_returns_none(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        data = pd.DataFrame(
            {
                "high": [110] * 5,
                "low": [90] * 5,
                "close": [100] * 5,
            },
            index=dates.date,
        )
        engine = _make_engine(data, pd.Series(), risk_config=RiskConfig(atr_period=14))
        atr = engine._compute_atr(data, dates[3].date())
        assert atr is None


# ─── run() 整合 ───────────────────────────────────────────


class TestBacktestRun:
    def test_buy_and_sell_produces_trade(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
                "volume": [1_000_000] * 5,
            },
            index=[d.date() for d in dates],
        )
        signals = pd.Series([1, 0, 0, -1, 0], index=data.index)

        engine = _make_engine(data, signals)
        result = engine.run()

        assert result.total_trades == 1
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "signal"
        assert result.final_capital != result.initial_capital

    def test_no_signals_no_trades(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        data = pd.DataFrame(
            {
                "open": [100] * 5,
                "high": [101] * 5,
                "low": [99] * 5,
                "close": [100] * 5,
                "volume": [1_000_000] * 5,
            },
            index=[d.date() for d in dates],
        )
        signals = pd.Series([0, 0, 0, 0, 0], index=data.index)

        engine = _make_engine(data, signals)
        result = engine.run()

        assert result.total_trades == 0
        assert result.final_capital == pytest.approx(1_000_000, abs=0.01)

    def test_force_close_at_end(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
                "volume": [1_000_000] * 3,
            },
            index=[d.date() for d in dates],
        )
        # Buy on day 1, never sell
        signals = pd.Series([1, 0, 0], index=data.index)

        engine = _make_engine(data, signals)
        result = engine.run()

        assert result.total_trades == 1
        assert result.trades[0].exit_reason == "force_close"

    def test_equity_curve_length(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        data = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [101] * 10,
                "low": [99] * 10,
                "close": [100] * 10,
                "volume": [1_000_000] * 10,
            },
            index=[d.date() for d in dates],
        )
        signals = pd.Series([0] * 10, index=data.index)

        engine = _make_engine(data, signals)
        result = engine.run()

        assert len(result.equity_curve) == 10
