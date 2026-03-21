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

    def test_sortino_ratio_with_mixed_returns(self):
        """含漲跌天數的 equity curve 應計算出正的 Sortino Ratio。"""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        # 交替漲跌，確保有多個下跌日（neg_returns not empty，std > 0）
        equity = [1_000_000, 1_010_000, 1_005_000, 1_015_000, 1_008_000, 1_020_000, 1_012_000, 1_025_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["sortino_ratio"] is not None
        assert metrics["sortino_ratio"] > 0

    def test_sortino_ratio_none_when_no_down_days(self):
        """單調遞增（無下跌日）的 equity curve → sortino_ratio 為 None。"""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000, 1_010_000, 1_020_000, 1_030_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["sortino_ratio"] is None

    def test_calmar_ratio_computed_with_drawdown(self):
        """有最大回撤時應計算 Calmar Ratio > 0。"""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        # Peak 1.2M → trough 0.9M → final 1.1M → max_drawdown = 25%，年化報酬 > 0
        equity = [1_000_000, 1_200_000, 900_000, 1_100_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["calmar_ratio"] is not None
        assert metrics["calmar_ratio"] > 0

    def test_calmar_ratio_none_when_no_drawdown(self):
        """無最大回撤（單調遞增）時 calmar_ratio 為 None。"""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000, 1_100_000, 1_200_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["calmar_ratio"] is None

    def test_var_cvar_computed_with_sufficient_data(self):
        """有多個資料點時 var_95 與 cvar_95 應不為 None。"""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000, 1_010_000, 1_005_000, 1_020_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["var_95"] is not None
        assert metrics["cvar_95"] is not None

    def test_cvar_is_worse_than_or_equal_to_var(self):
        """CVaR（尾部期望損失）應 <= VaR（第 5 百分位數）。"""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000, 1_010_000, 1_005_000, 1_015_000, 1_008_000, 1_020_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["cvar_95"] is not None
        assert metrics["var_95"] is not None
        assert metrics["cvar_95"] <= metrics["var_95"]

    def test_var_negative_for_downtrend(self):
        """持續下跌的 equity curve → var_95 應為負值。"""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000, 990_000, 980_000, 970_000, 960_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["var_95"] is not None
        assert metrics["var_95"] < 0

    def test_var_cvar_none_for_single_point(self):
        """只有 1 筆資料點時 var_95 與 cvar_95 應為 None。"""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
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

        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        # 模擬爆倉：equity 跌至 0 再回升
        equity = [1_000_000, 500_000, 0, 100_000, 200_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        # 不應為 nan 或 inf
        if metrics["sharpe_ratio"] is not None:
            assert math.isfinite(metrics["sharpe_ratio"])

    def test_sortino_with_zero_equity(self):
        """equity curve 含 0 時 Sortino 應為 None 或有限值。"""
        import math

        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000, 500_000, 0, 100_000, 200_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        if metrics["sortino_ratio"] is not None:
            assert math.isfinite(metrics["sortino_ratio"])

    def test_var_cvar_with_zero_equity(self):
        """equity curve 含 0 時 VaR/CVaR 應為 None 或有限值。"""
        import math

        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000, 500_000, 0, 100_000, 200_000]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        if metrics["var_95"] is not None:
            assert math.isfinite(metrics["var_95"])
        if metrics["cvar_95"] is not None:
            assert math.isfinite(metrics["cvar_95"])

    def test_sharpe_normal_case_unchanged(self):
        """正常 equity curve 的 Sharpe 計算不受影響。"""
        config = BacktestConfig(initial_capital=1_000_000)
        engine = _make_engine(pd.DataFrame(), pd.Series(), config=config)
        equity = [1_000_000 + i * 1000 for i in range(100)]
        metrics = engine._compute_metrics(equity, [], date(2024, 1, 1), date(2024, 12, 31))
        assert metrics["sharpe_ratio"] is not None
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
