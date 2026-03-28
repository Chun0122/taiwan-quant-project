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
from src.backtest.metrics import (
    compute_metrics,
    compute_trade_stats,
    export_trades,
    monte_carlo_equity,
    trades_to_dataframe,
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

        equity = [1_000_000, 1_050_000, 1_100_000]
        trades = []
        metrics = compute_metrics(equity, trades, date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["total_return"] == pytest.approx(10.0, abs=0.01)

    def test_max_drawdown(self):

        # Peak at 1.2M, trough at 1.0M → 16.67% drawdown
        equity = [1_000_000, 1_200_000, 1_000_000, 1_100_000]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["max_drawdown"] == pytest.approx(16.67, abs=0.01)

    def test_win_rate(self):

        trades = [
            TradeRecord(entry_date=date(2024, 1, 1), entry_price=100, pnl=500),
            TradeRecord(entry_date=date(2024, 2, 1), entry_price=100, pnl=-200),
            TradeRecord(entry_date=date(2024, 3, 1), entry_price=100, pnl=300),
        ]
        equity = [1_000_000, 1_000_500, 1_000_300, 1_000_600]
        metrics = compute_metrics(equity, trades, date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["win_rate"] == pytest.approx(66.67, abs=0.01)

    def test_sharpe_ratio_calculated(self):

        equity = [1_000_000 + i * 1000 for i in range(100)]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["sharpe_ratio"] is not None
        assert metrics["sharpe_ratio"] > 0

    def test_no_trades_win_rate_none(self):

        equity = [1_000_000]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["win_rate"] is None

    def test_profit_factor(self):

        trades = [
            TradeRecord(entry_date=date(2024, 1, 1), entry_price=100, pnl=1000),
            TradeRecord(entry_date=date(2024, 2, 1), entry_price=100, pnl=-500),
        ]
        equity = [1_000_000, 1_001_000, 1_000_500]
        metrics = compute_metrics(equity, trades, date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["profit_factor"] == pytest.approx(2.0, abs=0.01)

    def test_sortino_ratio_with_mixed_returns(self):
        """含漲跌天數的 equity curve 應計算出正的 Sortino Ratio。"""

        # 交替漲跌，確保有多個下跌日（neg_returns not empty，std > 0）
        equity = [1_000_000, 1_010_000, 1_005_000, 1_015_000, 1_008_000, 1_020_000, 1_012_000, 1_025_000]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["sortino_ratio"] is not None
        assert metrics["sortino_ratio"] > 0

    def test_sortino_ratio_none_when_no_down_days(self):
        """單調遞增（無下跌日）的 equity curve → sortino_ratio 為 None。"""

        equity = [1_000_000, 1_010_000, 1_020_000, 1_030_000]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["sortino_ratio"] is None

    def test_calmar_ratio_computed_with_drawdown(self):
        """有最大回撤時應計算 Calmar Ratio > 0。"""

        # Peak 1.2M → trough 0.9M → final 1.1M → max_drawdown = 25%，年化報酬 > 0
        equity = [1_000_000, 1_200_000, 900_000, 1_100_000]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["calmar_ratio"] is not None
        assert metrics["calmar_ratio"] > 0

    def test_calmar_ratio_none_when_no_drawdown(self):
        """無最大回撤（單調遞增）時 calmar_ratio 為 None。"""

        equity = [1_000_000, 1_100_000, 1_200_000]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["calmar_ratio"] is None

    def test_var_cvar_computed_with_sufficient_data(self):
        """有足夠資料點（≥20 筆日報酬）時 var_95 與 cvar_95 應不為 None。"""

        equity = [1_000_000 + i * 2000 * ((-1) ** i) for i in range(25)]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["var_95"] is not None
        assert metrics["cvar_95"] is not None

    def test_cvar_is_worse_than_or_equal_to_var(self):
        """CVaR（尾部期望損失）應 <= VaR（第 5 百分位數）。"""

        equity = [1_000_000 + i * 3000 * ((-1) ** i) for i in range(25)]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["cvar_95"] is not None
        assert metrics["var_95"] is not None
        assert metrics["cvar_95"] <= metrics["var_95"]

    def test_var_negative_for_downtrend(self):
        """持續下跌的 equity curve（≥20 筆日報酬）→ var_95 應為負值。"""

        equity = [1_000_000 - i * 10_000 for i in range(25)]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["var_95"] is not None
        assert metrics["var_95"] < 0

    def test_var_cvar_none_for_single_point(self):
        """只有 1 筆資料點時 var_95 與 cvar_95 應為 None。"""

        equity = [1_000_000]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["var_95"] is None
        assert metrics["cvar_95"] is None


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


# ─── ATR-based 止損止利 ────────────────────────────────────


class TestAtrBasedStop:
    def _make_uniform_data(self, n: int, high: float = 12.0, low: float = 10.0, close: float = 11.0) -> pd.DataFrame:
        """建立均勻 OHLCV 資料（ATR14 ≈ high - low）。"""
        dates = pd.bdate_range("2024-01-01", periods=n)
        return pd.DataFrame(
            {
                "open": [close] * n,
                "high": [high] * n,
                "low": [low] * n,
                "close": [close] * n,
                "volume": [10_000] * n,
            },
            index=[d.date() for d in dates],
        )

    def test_atr_stop_triggers_and_records_prices(self):
        """ATR-based 止損在 low 跌破止損價時觸發，TradeRecord 正確記錄 stop_price/target_price。

        資料設計：high=12, low=10, close=11 → TR=2.0, ATR14≈2.0
        進場（Day 15）：entry_price ≈ 11，stop ≈ 11 - 1.5×2 = 8.0，target ≈ 11 + 3.0×2 = 17.0
        Day 20：low 跌至 6.0 → 低於止損價 8.0 → 止損觸發
        """
        n = 30
        data = self._make_uniform_data(n)

        # Day 20：low 跌至 6.0，低於止損價（≈8.0）
        data.iloc[20, data.columns.get_loc("low")] = 6.0

        signals = pd.Series([0] * n, index=data.index)
        signals.iloc[15] = 1  # Day 15 買入（ATR14 已穩定）
        signals.iloc[29] = -1  # Day 29 賣出訊號（不會觸達，止損先觸發）

        risk_config = RiskConfig(atr_multiplier_stop=1.5, atr_multiplier_profit=3.0)
        engine = _make_engine(data, signals, risk_config=risk_config)
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "stop_loss"
        # stop_price ≈ 11 - 1.5×2 = 8.0（含滑價誤差）
        assert trade.stop_price is not None
        assert trade.stop_price == pytest.approx(8.0, abs=0.5)
        # target_price ≈ 11 + 3.0×2 = 17.0（含滑價誤差）
        assert trade.target_price is not None
        assert trade.target_price == pytest.approx(17.0, abs=0.5)

    def test_atr_take_profit_triggers(self):
        """ATR-based 止利在 high 超過目標價時觸發。

        進場（Day 15）：target ≈ 11 + 1.0×2 = 13.0
        Day 20：high 上衝至 15.0 → 高於目標價 13.0 → 止利觸發
        """
        n = 30
        data = self._make_uniform_data(n)

        # Day 20：high 上衝至 15.0，高於目標價（≈13.0）
        data.iloc[20, data.columns.get_loc("high")] = 15.0

        signals = pd.Series([0] * n, index=data.index)
        signals.iloc[15] = 1
        signals.iloc[29] = -1

        risk_config = RiskConfig(atr_multiplier_stop=2.0, atr_multiplier_profit=1.0)
        engine = _make_engine(data, signals, risk_config=risk_config)
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "take_profit"
        assert trade.target_price is not None
        assert trade.target_price == pytest.approx(13.0, abs=0.5)

    def test_percentage_stop_still_works_without_atr_multiplier(self):
        """未設定 atr_multiplier_stop 時，百分比停損行為與原有邏輯一致。"""
        n = 30
        data = self._make_uniform_data(n)

        # Day 20 low 跌至 9.0，低於 10% 停損價（≈9.9）
        data.iloc[20, data.columns.get_loc("low")] = 9.0

        signals = pd.Series([0] * n, index=data.index)
        signals.iloc[15] = 1
        signals.iloc[29] = -1

        risk_config = RiskConfig(stop_loss_pct=10.0)  # 無 ATR 乘數
        engine = _make_engine(data, signals, risk_config=risk_config)
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "stop_loss"
        # stop_price 由百分比計算，≈ 11×0.9 = 9.9
        assert trade.stop_price is not None
        assert trade.stop_price == pytest.approx(9.9, abs=0.5)
        assert trade.target_price is None  # 未設定 take_profit

    def test_no_stop_no_prices_recorded(self):
        """未設定任何停損時，stop_price 與 target_price 應為 None。"""
        n = 10
        data = self._make_uniform_data(n)
        signals = pd.Series([0] * n, index=data.index)
        signals.iloc[0] = 1
        signals.iloc[9] = -1

        engine = _make_engine(data, signals)  # 無 risk_config
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.stop_price is None
        assert trade.target_price is None


# ─── 跳空停損/停利成交價修正 ──────────────────────────────────────────────────


class TestGapDownStopLoss:
    """驗證跳空開低時停損/停利使用開盤價而非停損價。"""

    def _make_data_with_open(
        self, n: int, high: float = 12.0, low: float = 10.0, close: float = 11.0, open_price: float = 11.0
    ) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-01", periods=n)
        return pd.DataFrame(
            {
                "open": [open_price] * n,
                "high": [high] * n,
                "low": [low] * n,
                "close": [close] * n,
                "volume": [10_000] * n,
            },
            index=[d.date() for d in dates],
        )

    def test_normal_stop_uses_stop_price(self):
        """正常停損（open > stop > low）→ 以 stop_price 附近成交。"""
        n = 30
        data = self._make_data_with_open(n)
        # Day 20: open 正常(11)，low 跌破停損
        data.iloc[20, data.columns.get_loc("low")] = 6.0
        data.iloc[20, data.columns.get_loc("open")] = 11.0  # open > stop

        signals = pd.Series([0] * n, index=data.index)
        signals.iloc[15] = 1
        signals.iloc[29] = -1

        risk_config = RiskConfig(atr_multiplier_stop=1.5, atr_multiplier_profit=3.0)
        engine = _make_engine(data, signals, risk_config=risk_config)
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "stop_loss"
        # open(11) > stop(≈8)，所以用 stop_price
        assert trade.stop_price is not None
        # 成交價應接近 stop_price（含滑價）
        assert trade.exit_price < trade.stop_price * 1.01  # 扣滑價後略低

    def test_gap_down_uses_open_price(self):
        """跳空開低（open < stop）→ 以 open 成交。"""
        n = 30
        data = self._make_data_with_open(n)
        # Day 20: open 跳空開低到 5.0（低於 stop ≈ 8.0）
        data.iloc[20, data.columns.get_loc("open")] = 5.0
        data.iloc[20, data.columns.get_loc("low")] = 4.5
        data.iloc[20, data.columns.get_loc("close")] = 5.5

        signals = pd.Series([0] * n, index=data.index)
        signals.iloc[15] = 1
        signals.iloc[29] = -1

        risk_config = RiskConfig(atr_multiplier_stop=1.5, atr_multiplier_profit=3.0)
        engine = _make_engine(data, signals, risk_config=risk_config)
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "stop_loss"
        # open(5.0) < stop(≈8.0)，成交價應基於 open(5.0) 而非 stop(8.0)
        assert trade.exit_price < 5.5  # 應該接近 5.0（含滑價）

    def test_gap_up_take_profit_uses_open(self):
        """跳空開高停利（open > target）→ 以 open 成交。"""
        n = 30
        data = self._make_data_with_open(n)
        # Day 20: open 跳空開高到 20.0（高於 target ≈ 13.0）
        data.iloc[20, data.columns.get_loc("open")] = 20.0
        data.iloc[20, data.columns.get_loc("high")] = 21.0
        data.iloc[20, data.columns.get_loc("close")] = 19.0

        signals = pd.Series([0] * n, index=data.index)
        signals.iloc[15] = 1
        signals.iloc[29] = -1

        risk_config = RiskConfig(atr_multiplier_stop=2.0, atr_multiplier_profit=1.0)
        engine = _make_engine(data, signals, risk_config=risk_config)
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "take_profit"
        # open(20) > target(≈13)，成交價應基於 open(20) 而非 target(13)
        assert trade.exit_price > 15.0  # 遠高於 target

    def test_no_open_column_fallback(self):
        """無 open 欄位 → fallback 到 raw_close 邏輯（不崩潰）。"""
        n = 30
        dates = pd.bdate_range("2024-01-01", periods=n)
        data = pd.DataFrame(
            {
                "high": [12.0] * n,
                "low": [10.0] * n,
                "close": [11.0] * n,
                "volume": [10_000] * n,
            },
            index=[d.date() for d in dates],
        )
        data.iloc[20, data.columns.get_loc("low")] = 6.0

        signals = pd.Series([0] * n, index=data.index)
        signals.iloc[15] = 1
        signals.iloc[29] = -1

        risk_config = RiskConfig(atr_multiplier_stop=1.5, atr_multiplier_profit=3.0)
        engine = _make_engine(data, signals, risk_config=risk_config)
        result = engine.run()

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "stop_loss"


# ─── ATR 尺度一致性（除權息模式）───────────────────────────────────────────────


class TestAtrScaleAdjustedDividend:
    """驗證除權息模式（adjust_dividend）下，ATR 使用原始價格欄位計算，確保與 entry_price 同尺度。"""

    def _make_adjusted_data(
        self,
        n: int,
        adj_factor: float = 0.9,
    ) -> pd.DataFrame:
        """建立含 raw_* 欄位的除權息資料。

        raw 價格：high=12, low=10, close=11（ATR ≈ 2.0）
        調整後價格：乘以 adj_factor（ATR ≈ 2.0 × adj_factor）
        """
        dates = pd.bdate_range("2024-01-01", periods=n)
        raw_high = 12.0
        raw_low = 10.0
        raw_close = 11.0
        return pd.DataFrame(
            {
                "open": [raw_close * adj_factor] * n,
                "high": [raw_high * adj_factor] * n,
                "low": [raw_low * adj_factor] * n,
                "close": [raw_close * adj_factor] * n,
                "volume": [10_000] * n,
                "raw_high": [raw_high] * n,
                "raw_low": [raw_low] * n,
                "raw_close": [raw_close] * n,
            },
            index=[d.date() for d in dates],
        )

    def test_use_raw_true_returns_raw_scale_atr(self):
        """use_raw=True 時，ATR 以原始價格尺度計算（≈ raw TR，非調整後）。

        raw 價格 TR ≈ 2.0，調整後 TR ≈ 2.0 × 0.9 = 1.8。
        use_raw=True 應回傳 ≈2.0，use_raw=False 應回傳 ≈1.8。
        """
        n = 20
        data = self._make_adjusted_data(n, adj_factor=0.9)
        engine = _make_engine(data, pd.Series([0] * n, index=data.index))

        dt = data.index[15]  # idx=15 > atr_period=14，資料足夠
        atr_raw = engine._compute_atr(data, dt, use_raw=True)
        atr_adj = engine._compute_atr(data, dt, use_raw=False)

        assert atr_raw is not None
        assert atr_adj is not None
        # raw ATR ≈ 2.0（原始尺度）
        assert atr_raw == pytest.approx(2.0, abs=0.05)
        # 調整後 ATR ≈ 1.8（縮小 10%）
        assert atr_adj == pytest.approx(1.8, abs=0.05)
        # 兩者必須不同（用以確認尺度確實不一樣）
        assert atr_raw != pytest.approx(atr_adj, abs=0.05)

    def test_use_raw_false_without_raw_cols_unchanged(self):
        """無 raw_* 欄位時，use_raw=True 應自動 fallback 至調整後欄位，不報錯。"""
        n = 20
        dates = pd.bdate_range("2024-01-01", periods=n)
        data = pd.DataFrame(
            {
                "open": [11.0] * n,
                "high": [12.0] * n,
                "low": [10.0] * n,
                "close": [11.0] * n,
                "volume": [10_000] * n,
                # 無 raw_* 欄位
            },
            index=[d.date() for d in dates],
        )
        engine = _make_engine(data, pd.Series([0] * n, index=data.index))

        dt = data.index[15]
        atr_with_flag = engine._compute_atr(data, dt, use_raw=True)
        atr_without_flag = engine._compute_atr(data, dt, use_raw=False)

        # 兩者應相等（皆 fallback 至 high/low/close）
        assert atr_with_flag is not None
        assert atr_with_flag == pytest.approx(atr_without_flag, abs=1e-9)


# ─── Phase1 修復驗證：Sharpe/Sortino inf 防護 + ATR 對齊 ──────


class TestPhase1SharpeInfGuard:
    """A-03 修復驗證：equity 含零值時 Sharpe/Sortino 不應產生 inf/nan。"""

    def test_sharpe_with_zero_equity(self):
        """equity curve 含 0（爆倉）時 Sharpe 應為 None 或有限值。"""
        import math

        # 模擬爆倉：equity 跌至 0 再回升
        equity = [1_000_000, 500_000, 0, 100_000, 200_000]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        # 不應為 nan 或 inf
        if metrics["sharpe_ratio"] is not None:
            assert math.isfinite(metrics["sharpe_ratio"])

    def test_sortino_with_zero_equity(self):
        """equity curve 含 0 時 Sortino 應為 None 或有限值。"""
        import math

        equity = [1_000_000, 500_000, 0, 100_000, 200_000]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        if metrics["sortino_ratio"] is not None:
            assert math.isfinite(metrics["sortino_ratio"])

    def test_var_cvar_with_zero_equity(self):
        """equity curve 含 0 時 VaR/CVaR 應為 None 或有限值。"""
        import math

        equity = [1_000_000, 500_000, 0, 100_000, 200_000]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        if metrics["var_95"] is not None:
            assert math.isfinite(metrics["var_95"])
        if metrics["cvar_95"] is not None:
            assert math.isfinite(metrics["cvar_95"])

    def test_sharpe_normal_case_unchanged(self):
        """正常 equity curve 的 Sharpe 計算不受影響。"""

        equity = [1_000_000 + i * 1000 for i in range(100)]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["sharpe_ratio"] is not None
        assert metrics["sharpe_ratio"] > 0


class TestMetricsNumericalStability:
    """驗證 Sharpe/Sortino/VaR/CVaR 在邊界條件下的數值穩定性。"""

    def test_sharpe_near_zero_std_returns_none(self):
        """日報酬率幾乎恆定（std ~1e-16）時，Sharpe 應為 None 而非極端值。"""
        # 1000 天微幅波動：equity 每天增加 0.001 元 → std ≈ 0
        equity = [1_000_000 + i * 0.001 for i in range(100)]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        # std 極低時應回傳 None
        assert metrics["sharpe_ratio"] is None

    def test_sortino_near_zero_std_returns_none(self):
        """負向報酬率 std 接近零時，Sortino 應為 None 而非極端值。"""
        # 構造極小負向波動：每天交替 -1e-12 和 -2e-12
        base = 1_000_000
        equity = [base]
        for i in range(50):
            equity.append(equity[-1] - 1e-12 * (1 + (i % 2)))
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), base)
        assert metrics["sortino_ratio"] is None

    def test_cvar_insufficient_samples_returns_none(self):
        """日報酬率不足 20 筆時，VaR/CVaR 應為 None。"""
        # 10 天 equity → 9 筆日報酬率，低於 _MIN_SAMPLES_FOR_VAR=20
        equity = [1_000_000 + i * 5000 for i in range(10)]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 1, 10), 1_000_000)
        assert metrics["var_95"] is None
        assert metrics["cvar_95"] is None

    def test_cvar_sufficient_samples_computes(self):
        """日報酬率 ≥ 20 筆時，VaR/CVaR 應正常計算。"""
        equity = [1_000_000 + i * 5000 for i in range(30)]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 1, 30), 1_000_000)
        assert metrics["var_95"] is not None
        assert metrics["cvar_95"] is not None

    def test_sharpe_normal_case_still_works(self):
        """標準波動度下 Sharpe 仍正常計算（迴歸驗證）。"""
        import math

        equity = [1_000_000 + i * 1000 for i in range(100)]
        metrics = compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31), 1_000_000)
        assert metrics["sharpe_ratio"] is not None
        assert math.isfinite(metrics["sharpe_ratio"])
        assert metrics["sharpe_ratio"] > 0


class TestPhase1AtrAlignment:
    """A-01 修復驗證：ATR 的 prev_close 對齊正確。"""

    def test_atr_prev_close_alignment_with_varying_close(self):
        """當 close 逐日遞增時，ATR 應反映真實 TR（含 prev_close 影響）。

        設計：close 從 100 漲到 119（每天 +1），high/low 固定 ±5
        TR[i] = max(H-L, |H-prev_close|, |L-prev_close|)
            = max(10, |close+5 - (close-1)|, |close-5 - (close-1)|)
            = max(10, 6, 4) = 10
        所以 ATR(5) ≈ 10.0
        """
        n = 20
        dates = pd.bdate_range("2024-01-01", periods=n)
        closes = [100.0 + i for i in range(n)]
        data = pd.DataFrame(
            {
                "high": [c + 5 for c in closes],
                "low": [c - 5 for c in closes],
                "close": closes,
            },
            index=[d.date() for d in dates],
        )
        engine = _make_engine(data, pd.Series(), risk_config=RiskConfig(atr_period=5))
        atr = engine._compute_atr(data, dates[10].date())
        assert atr is not None
        assert atr == pytest.approx(10.0, abs=0.5)

    def test_atr_with_gap_up_reflects_prev_close(self):
        """跳空高開時，TR 應包含 |High - PrevClose| 的貢獻。

        設計：前 14 天 close=100，第 15 天跳空至 120
        最後一根 TR = max(H-L, |H-100|, |L-100|) = max(10, 25, 15) = 25
        前 13 根 TR = 10 → ATR(14) ≈ (13×10 + 25) / 14 ≈ 11.07
        """
        n = 20
        dates = pd.bdate_range("2024-01-01", periods=n)
        closes = [100.0] * n
        highs = [105.0] * n
        lows = [95.0] * n
        # 第 15 天（idx=14）跳空高開
        closes[14] = 120.0
        highs[14] = 125.0
        lows[14] = 115.0

        data = pd.DataFrame(
            {"high": highs, "low": lows, "close": closes},
            index=[d.date() for d in dates],
        )
        engine = _make_engine(data, pd.Series(), risk_config=RiskConfig(atr_period=14))
        # dt = dates[15]，ATR 視窗 = idx[1..14]
        atr = engine._compute_atr(data, dates[15].date())
        assert atr is not None
        # 含跳空的 ATR 應 > 純粹 H-L=10 的 ATR
        assert atr > 10.0


# ─── compute_trade_stats ──────────────────────────────────


class TestComputeTradeStats:
    """測試 compute_trade_stats 持倉天數與勝敗統計。"""

    def test_empty_trades(self):
        stats = compute_trade_stats([])
        assert stats["holding_days_avg"] is None
        assert stats["max_consecutive_wins"] == 0
        assert stats["exit_reason_counts"] == {}

    def test_holding_days_calculation(self):
        trades = [
            TradeRecord(entry_date=date(2024, 1, 1), entry_price=100, exit_date=date(2024, 1, 11), pnl=100),
            TradeRecord(entry_date=date(2024, 2, 1), entry_price=100, exit_date=date(2024, 2, 6), pnl=-50),
        ]
        stats = compute_trade_stats(trades)
        # 第一筆 10 天，第二筆 5 天
        assert stats["holding_days_avg"] == pytest.approx(7.5)
        assert stats["holding_days_min"] == 5
        assert stats["holding_days_max"] == 10
        assert stats["holding_days_median"] == pytest.approx(7.5)

    def test_holding_days_median_odd(self):
        """奇數筆交易的中位數。"""
        trades = [
            TradeRecord(entry_date=date(2024, 1, 1), entry_price=100, exit_date=date(2024, 1, 4), pnl=10),
            TradeRecord(entry_date=date(2024, 2, 1), entry_price=100, exit_date=date(2024, 2, 11), pnl=20),
            TradeRecord(entry_date=date(2024, 3, 1), entry_price=100, exit_date=date(2024, 3, 21), pnl=30),
        ]
        stats = compute_trade_stats(trades)
        # 3, 10, 20 天 → 中位數 = 10
        assert stats["holding_days_median"] == 10.0

    def test_win_loss_analysis(self):
        trades = [
            TradeRecord(
                entry_date=date(2024, 1, 1), entry_price=100, exit_date=date(2024, 1, 5), pnl=1000, return_pct=10.0
            ),
            TradeRecord(
                entry_date=date(2024, 2, 1), entry_price=100, exit_date=date(2024, 2, 5), pnl=-500, return_pct=-5.0
            ),
            TradeRecord(
                entry_date=date(2024, 3, 1), entry_price=100, exit_date=date(2024, 3, 5), pnl=600, return_pct=6.0
            ),
        ]
        stats = compute_trade_stats(trades)
        assert stats["avg_win_pnl"] == pytest.approx(800.0)  # (1000+600)/2
        assert stats["avg_loss_pnl"] == pytest.approx(-500.0)
        assert stats["avg_win_return"] == pytest.approx(8.0)  # (10+6)/2
        assert stats["avg_loss_return"] == pytest.approx(-5.0)

    def test_consecutive_wins_losses(self):
        trades = [
            TradeRecord(entry_date=date(2024, 1, 1), entry_price=100, pnl=100),
            TradeRecord(entry_date=date(2024, 1, 2), entry_price=100, pnl=200),
            TradeRecord(entry_date=date(2024, 1, 3), entry_price=100, pnl=300),
            TradeRecord(entry_date=date(2024, 1, 4), entry_price=100, pnl=-100),
            TradeRecord(entry_date=date(2024, 1, 5), entry_price=100, pnl=-200),
        ]
        stats = compute_trade_stats(trades)
        assert stats["max_consecutive_wins"] == 3
        assert stats["max_consecutive_losses"] == 2

    def test_exit_reason_counts(self):
        trades = [
            TradeRecord(entry_date=date(2024, 1, 1), entry_price=100, pnl=100, exit_reason="signal"),
            TradeRecord(entry_date=date(2024, 1, 2), entry_price=100, pnl=-50, exit_reason="stop_loss"),
            TradeRecord(entry_date=date(2024, 1, 3), entry_price=100, pnl=200, exit_reason="signal"),
            TradeRecord(entry_date=date(2024, 1, 4), entry_price=100, pnl=150, exit_reason="take_profit"),
        ]
        stats = compute_trade_stats(trades)
        assert stats["exit_reason_counts"]["signal"] == 2
        assert stats["exit_reason_counts"]["stop_loss"] == 1
        assert stats["exit_reason_counts"]["take_profit"] == 1

    def test_none_exit_dates(self):
        """exit_date 為 None 的交易不納入持倉天數計算。"""
        trades = [
            TradeRecord(entry_date=date(2024, 1, 1), entry_price=100, exit_date=None, pnl=100),
            TradeRecord(entry_date=date(2024, 2, 1), entry_price=100, exit_date=date(2024, 2, 11), pnl=-50),
        ]
        stats = compute_trade_stats(trades)
        # 只有一筆有效 (10 天)
        assert stats["holding_days_avg"] == pytest.approx(10.0)
        assert stats["holding_days_min"] == 10
        assert stats["holding_days_max"] == 10


class TestTradesToDataframe:
    """測試 trades_to_dataframe DataFrame 轉換。"""

    def test_basic_conversion(self):
        trades = [
            TradeRecord(
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                exit_date=date(2024, 1, 11),
                exit_price=110.0,
                shares=1000,
                pnl=10000,
                return_pct=10.0,
                exit_reason="signal",
            ),
        ]
        df = trades_to_dataframe(trades, stock_id="2330")
        assert len(df) == 1
        assert df.iloc[0]["stock_id"] == "2330"
        assert df.iloc[0]["holding_days"] == 10
        assert df.iloc[0]["pnl"] == 10000

    def test_includes_stop_target_prices(self):
        trades = [
            TradeRecord(
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                exit_date=date(2024, 1, 5),
                pnl=-500,
                exit_reason="stop_loss",
                stop_price=95.0,
                target_price=115.0,
            ),
        ]
        df = trades_to_dataframe(trades)
        assert "stop_price" in df.columns
        assert df.iloc[0]["stop_price"] == 95.0
        assert df.iloc[0]["target_price"] == 115.0

    def test_empty_trades(self):
        df = trades_to_dataframe([])
        assert df.empty

    def test_portfolio_trade_has_stock_id(self):
        """PortfolioTradeRecord 自帶 stock_id，不需外部傳入。"""
        from src.backtest.portfolio import PortfolioTradeRecord

        trades = [
            PortfolioTradeRecord(
                stock_id="2317",
                entry_date=date(2024, 1, 1),
                entry_price=80.0,
                exit_date=date(2024, 1, 21),
                pnl=5000,
                return_pct=6.25,
            ),
        ]
        df = trades_to_dataframe(trades)
        assert df.iloc[0]["stock_id"] == "2317"
        assert df.iloc[0]["holding_days"] == 20


class TestExportTrades:
    """測試 export_trades CSV 匯出。"""

    def test_export_creates_csv(self, tmp_path):
        trades = [
            TradeRecord(
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                exit_date=date(2024, 1, 11),
                exit_price=110.0,
                shares=1000,
                pnl=10000,
                return_pct=10.0,
            ),
            TradeRecord(
                entry_date=date(2024, 2, 1),
                entry_price=110.0,
                exit_date=date(2024, 2, 15),
                exit_price=105.0,
                shares=1000,
                pnl=-5000,
                return_pct=-4.55,
            ),
        ]
        csv_path = str(tmp_path / "trades.csv")
        result_path = export_trades(trades, csv_path, stock_id="2330")
        assert result_path == csv_path

        # 驗證 CSV 內容
        import pandas as pd

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "holding_days" in df.columns
        assert df.iloc[0]["holding_days"] == 10
        assert df.iloc[1]["holding_days"] == 14

    def test_export_empty_raises(self, tmp_path):
        csv_path = str(tmp_path / "empty.csv")
        with pytest.raises(ValueError, match="無交易記錄"):
            export_trades([], csv_path)


# ─── A1: 動態滑價模型 ──────────────────────────────────────


class TestDynamicSlippage:
    """測試 BacktestEngine._get_slippage() 動態滑價模型。"""

    def _make_engine_with_dynamic(self, dynamic: bool = True, impact_coeff: float = 0.5):
        config = BacktestConfig(
            dynamic_slippage=dynamic,
            slippage_impact_coeff=impact_coeff,
        )
        return _make_engine(pd.DataFrame(), pd.Series(), config=config)

    def test_fixed_slippage_when_disabled(self):
        """dynamic_slippage=False 時回傳固定滑價。"""
        engine = self._make_engine_with_dynamic(dynamic=False)
        assert engine._get_slippage(1_000_000) == BacktestConfig().slippage

    def test_dynamic_higher_for_low_volume(self):
        """低成交量股票的滑價應高於高成交量。"""
        engine = self._make_engine_with_dynamic(dynamic=True)
        slip_low = engine._get_slippage(50_000)  # 小型股
        slip_high = engine._get_slippage(30_000_000)  # TSMC 級
        assert slip_low > slip_high

    def test_dynamic_converges_to_base_for_large_volume(self):
        """超高成交量時滑價應接近 base。"""
        engine = self._make_engine_with_dynamic(dynamic=True, impact_coeff=0.5)
        base = BacktestConfig().slippage
        slip = engine._get_slippage(100_000_000)
        assert slip == pytest.approx(base, abs=0.0001)

    def test_dynamic_formula_correctness(self):
        """驗證公式 slippage = base + k / sqrt(volume)。"""
        import numpy as np

        engine = self._make_engine_with_dynamic(dynamic=True, impact_coeff=0.5)
        volume = 250_000
        expected = BacktestConfig().slippage + 0.5 / np.sqrt(volume)
        assert engine._get_slippage(volume) == pytest.approx(expected, rel=1e-6)

    def test_dynamic_zero_volume_fallback(self):
        """volume=0 時回傳固定滑價（避免除零）。"""
        engine = self._make_engine_with_dynamic(dynamic=True)
        assert engine._get_slippage(0) == BacktestConfig().slippage

    def test_dynamic_slippage_affects_backtest(self):
        """啟用動態滑價後，低量股的最終資金應低於固定滑價。"""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        data = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [105] * 10,
                "low": [95] * 10,
                "close": [100, 102, 104, 106, 108, 110, 108, 106, 104, 102],
                "volume": [10_000] * 10,  # 低量
            },
            index=dates,
        )
        signals = pd.Series([1, 0, 0, 0, -1, 0, 0, 0, 0, 0], index=dates)

        # 固定滑價
        result_fixed = _make_engine(data, signals, config=BacktestConfig(dynamic_slippage=False)).run()
        # 動態滑價
        result_dynamic = _make_engine(data, signals, config=BacktestConfig(dynamic_slippage=True)).run()

        # 動態滑價在低量下成本更高 → 最終資金更少
        assert result_dynamic.final_capital < result_fixed.final_capital


# ─── A2: 流動性約束 ──────────────────────────────────────


class TestLiquidityLimit:
    """測試 BacktestEngine._apply_liquidity_limit() 流動性約束。"""

    def _make_engine_with_limit(self, limit: float | None = 0.05):
        config = BacktestConfig(liquidity_limit=limit)
        return _make_engine(pd.DataFrame(), pd.Series(), config=config)

    def test_no_limit_returns_original(self):
        """liquidity_limit=None 時不限制。"""
        engine = self._make_engine_with_limit(limit=None)
        assert engine._apply_liquidity_limit(10_000, 100_000) == 10_000

    def test_within_limit_unchanged(self):
        """交易量在限制內時不調整。"""
        engine = self._make_engine_with_limit(limit=0.05)
        # 5% of 1M = 50,000 → 10,000 shares 不超限
        assert engine._apply_liquidity_limit(10_000, 1_000_000) == 10_000

    def test_exceeds_limit_capped(self):
        """交易量超出限制時被截斷。"""
        engine = self._make_engine_with_limit(limit=0.05)
        # 5% of 100,000 = 5,000 → 10,000 shares 超限
        assert engine._apply_liquidity_limit(10_000, 100_000) == 5_000

    def test_zero_volume_no_limit(self):
        """volume=0 時不限制（可能為資料缺失）。"""
        engine = self._make_engine_with_limit(limit=0.05)
        assert engine._apply_liquidity_limit(10_000, 0) == 10_000

    def test_liquidity_limit_reduces_position_in_backtest(self):
        """啟用流動性約束後，低量股的持倉應被限制。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        data = pd.DataFrame(
            {
                "open": [100] * 5,
                "high": [105] * 5,
                "low": [95] * 5,
                "close": [100, 105, 110, 105, 100],
                "volume": [5_000] * 5,  # 極低量
            },
            index=dates,
        )
        signals = pd.Series([1, 0, 0, -1, 0], index=dates)

        # 無限制
        result_no_limit = _make_engine(data, signals, config=BacktestConfig(liquidity_limit=None)).run()
        # 有限制（5% × 5000 = 250 股）
        result_limited = _make_engine(data, signals, config=BacktestConfig(liquidity_limit=0.05)).run()

        # 有限制時持倉量較小 → 獲利絕對值較小
        if result_no_limit.trades and result_limited.trades:
            assert result_limited.trades[0].shares < result_no_limit.trades[0].shares


# ─── A3: Monte Carlo 信賴區間 ──────────────────────────────


class TestMonteCarlo:
    """測試 monte_carlo_equity() Bootstrap Resampling。"""

    def test_empty_returns_none(self):
        """空交易序列 → 所有指標為 None。"""
        result = monte_carlo_equity([])
        assert result["total_return_p50"] is None
        assert result["n_trades"] == 0

    def test_single_trade_returns_none(self):
        """只有 1 筆交易 → 不足以做 bootstrap。"""
        result = monte_carlo_equity([5.0])
        assert result["total_return_p50"] is None
        assert result["n_trades"] == 1

    def test_positive_trades_positive_median(self):
        """全正報酬序列 → 中位數報酬應為正。"""
        returns = [3.0, 5.0, 2.0, 4.0, 6.0, 1.0, 3.5, 2.5]
        result = monte_carlo_equity(returns, seed=42)
        assert result["total_return_p50"] is not None
        assert result["total_return_p50"] > 0

    def test_confidence_interval_ordering(self):
        """P5 ≤ P50 ≤ P95 對所有指標成立。"""
        returns = [3.0, -2.0, 5.0, -1.0, 4.0, -3.0, 2.0, 1.0, -0.5, 6.0]
        result = monte_carlo_equity(returns, seed=42, n_simulations=500)
        assert result["total_return_p5"] <= result["total_return_p50"] <= result["total_return_p95"]
        assert result["max_drawdown_p5"] <= result["max_drawdown_p50"] <= result["max_drawdown_p95"]
        assert result["sharpe_p5"] <= result["sharpe_p50"] <= result["sharpe_p95"]

    def test_deterministic_with_seed(self):
        """相同 seed → 相同結果（可重現性）。"""
        returns = [2.0, -1.0, 3.0, -2.0, 4.0]
        r1 = monte_carlo_equity(returns, seed=123)
        r2 = monte_carlo_equity(returns, seed=123)
        assert r1["total_return_p50"] == r2["total_return_p50"]
        assert r1["max_drawdown_p50"] == r2["max_drawdown_p50"]

    def test_max_drawdown_non_negative(self):
        """最大回撤應 >= 0。"""
        returns = [5.0, -3.0, 2.0, -1.0, 4.0]
        result = monte_carlo_equity(returns, seed=42)
        assert result["max_drawdown_p5"] >= 0
        assert result["max_drawdown_p95"] >= 0

    def test_n_simulations_respected(self):
        """回傳的 n_simulations 應與輸入一致。"""
        returns = [1.0, 2.0, 3.0]
        result = monte_carlo_equity(returns, n_simulations=200, seed=42)
        assert result["n_simulations"] == 200
