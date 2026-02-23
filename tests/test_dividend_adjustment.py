"""測試除權息還原功能 — 價格調整 + 指標重算 + 回測股利入帳。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.features.indicators import compute_indicators_from_df
from src.strategy.base import Strategy

# ─── helpers ──────────────────────────────────────────────


class MockDividendStrategy(Strategy):
    """可注入 data / signals / dividends 的 mock 策略。"""

    def __init__(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        dividends: pd.DataFrame | None = None,
    ):
        self._mock_data = data
        self._signals = signals
        self._dividends = dividends
        self.stock_id = "TEST"

    @property
    def name(self) -> str:
        return "mock_dividend"

    def load_data(self) -> pd.DataFrame:
        return self._mock_data

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return self._signals


def _make_engine(
    data: pd.DataFrame,
    signals: pd.Series,
    dividends: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
) -> BacktestEngine:
    strategy = MockDividendStrategy(data, signals, dividends)
    return BacktestEngine(strategy, config=config)


# ─── _apply_dividend_adjustment ───────────────────────────


class TestApplyDividendAdjustment:
    """測試 Strategy._apply_dividend_adjustment() 價格回溯調整。"""

    def _make_strategy(self) -> Strategy:
        """建立可呼叫 _apply_dividend_adjustment 的 mock。"""
        return MockDividendStrategy(pd.DataFrame(), pd.Series())

    def test_cash_dividend_adjusts_prices_before_ex_date(self):
        """現金股利 → ex_date 前的 close 乘以 factor。"""
        dates = [date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3)]
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 95.0],
                "high": [102.0, 102.0, 97.0],
                "low": [98.0, 98.0, 93.0],
                "close": [100.0, 100.0, 95.0],
                "volume": [1000, 1000, 1000],
            },
            index=dates,
        )
        # 7/3 除權息，現金股利 5 元
        dividends = pd.DataFrame(
            {"cash_dividend": [5.0], "stock_dividend": [0.0]},
            index=[date(2024, 7, 3)],
        )

        strategy = self._make_strategy()
        result = strategy._apply_dividend_adjustment(df, dividends)

        # factor = (100 - 5) / 100 = 0.95
        assert result.loc[date(2024, 7, 1), "close"] == pytest.approx(95.0, abs=0.01)
        assert result.loc[date(2024, 7, 2), "close"] == pytest.approx(95.0, abs=0.01)
        # ex_date 當天不調整
        assert result.loc[date(2024, 7, 3), "close"] == pytest.approx(95.0, abs=0.01)
        # raw_close 保持原值
        assert result.loc[date(2024, 7, 1), "raw_close"] == pytest.approx(100.0)
        assert result.loc[date(2024, 7, 2), "raw_close"] == pytest.approx(100.0)

    def test_stock_dividend_adjusts_prices(self):
        """股票股利 → 除權調整。"""
        dates = [date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3)]
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 90.0],
                "high": [102.0, 102.0, 92.0],
                "low": [98.0, 98.0, 88.0],
                "close": [100.0, 100.0, 90.0],
                "volume": [1000, 1000, 1000],
            },
            index=dates,
        )
        # stock_dividend=1.0 → 每 10 股配 1 股 → factor 額外除以 1.1
        dividends = pd.DataFrame(
            {"cash_dividend": [0.0], "stock_dividend": [1.0]},
            index=[date(2024, 7, 3)],
        )

        strategy = self._make_strategy()
        result = strategy._apply_dividend_adjustment(df, dividends)

        # factor = (100 - 0) / 100 / (1 + 1.0/10) = 1.0 / 1.1 ≈ 0.9091
        expected = 100.0 / 1.1
        assert result.loc[date(2024, 7, 1), "close"] == pytest.approx(expected, abs=0.01)

    def test_combined_cash_and_stock_dividend(self):
        """同時有現金+股票股利。"""
        dates = [date(2024, 7, 1), date(2024, 7, 2)]
        df = pd.DataFrame(
            {
                "open": [200.0, 180.0],
                "high": [202.0, 182.0],
                "low": [198.0, 178.0],
                "close": [200.0, 180.0],
                "volume": [1000, 1000],
            },
            index=dates,
        )
        # 7/2 除權息: cash=5, stock=0.5 (每 10 股配 0.5 股)
        dividends = pd.DataFrame(
            {"cash_dividend": [5.0], "stock_dividend": [0.5]},
            index=[date(2024, 7, 2)],
        )

        strategy = self._make_strategy()
        result = strategy._apply_dividend_adjustment(df, dividends)

        # factor = (200 - 5) / 200 / (1 + 0.5/10) = 195/200 / 1.05 = 0.975 / 1.05
        factor = (200 - 5) / 200 / (1 + 0.5 / 10)
        assert result.loc[date(2024, 7, 1), "close"] == pytest.approx(200.0 * factor, abs=0.01)

    def test_empty_dividends_returns_unchanged(self):
        """無除權息資料 → 原表不變。"""
        dates = [date(2024, 7, 1), date(2024, 7, 2)]
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [98.0, 99.0],
                "close": [100.0, 101.0],
                "volume": [1000, 1000],
            },
            index=dates,
        )
        dividends = pd.DataFrame(columns=["cash_dividend", "stock_dividend"])

        strategy = self._make_strategy()
        result = strategy._apply_dividend_adjustment(df, dividends)

        # 沒有 raw_close 欄位（空 dividends 直接 return）
        assert "raw_close" not in result.columns
        assert result.loc[date(2024, 7, 1), "close"] == pytest.approx(100.0)

    def test_invalid_factor_skipped(self):
        """異常因子（factor <= 0 或 >= 1.5）會被跳過。"""
        dates = [date(2024, 7, 1), date(2024, 7, 2)]
        df = pd.DataFrame(
            {
                "open": [10.0, 5.0],
                "high": [12.0, 7.0],
                "low": [8.0, 3.0],
                "close": [10.0, 5.0],
                "volume": [1000, 1000],
            },
            index=dates,
        )
        # cash_dividend=15 > prev_close=10 → factor < 0 → skip
        dividends = pd.DataFrame(
            {"cash_dividend": [15.0], "stock_dividend": [0.0]},
            index=[date(2024, 7, 2)],
        )

        strategy = self._make_strategy()
        result = strategy._apply_dividend_adjustment(df, dividends)

        # raw_close 欄位存在（有非空 dividends），但 close 未被調整
        assert result.loc[date(2024, 7, 1), "close"] == pytest.approx(10.0)


# ─── compute_indicators_from_df ───────────────────────────


class TestComputeIndicatorsFromDf:
    """測試從 DataFrame 直接計算技術指標。"""

    def test_returns_expected_columns(self):
        """確認回傳的指標欄位。"""
        dates = pd.bdate_range("2024-01-01", periods=60)
        df = pd.DataFrame(
            {
                "close": [100 + i * 0.5 for i in range(60)],
                "high": [101 + i * 0.5 for i in range(60)],
                "low": [99 + i * 0.5 for i in range(60)],
            },
            index=dates.date,
        )

        result = compute_indicators_from_df(df)

        expected_cols = {
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_60",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_sma_values_correct(self):
        """SMA 計算值驗證。"""
        dates = pd.bdate_range("2024-01-01", periods=10)
        closes = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        df = pd.DataFrame(
            {
                "close": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
            },
            index=dates.date,
        )

        result = compute_indicators_from_df(df)

        # SMA_5 at index 4 (5th day) = avg(100,102,104,106,108) = 104
        sma5_day5 = result.iloc[4]["sma_5"]
        assert sma5_day5 == pytest.approx(104.0, abs=0.01)

    def test_index_preserved(self):
        """回傳的 DataFrame index 與輸入一致。"""
        dates = pd.bdate_range("2024-01-01", periods=30)
        df = pd.DataFrame(
            {
                "close": [100 + i for i in range(30)],
                "high": [101 + i for i in range(30)],
                "low": [99 + i for i in range(30)],
            },
            index=dates.date,
        )

        result = compute_indicators_from_df(df)
        assert list(result.index) == list(df.index)


# ─── BacktestEngine 除權息整合 ────────────────────────────


class TestBacktestDividendIntegration:
    """測試回測引擎的股利入帳邏輯。"""

    def test_cash_dividend_adds_to_capital(self):
        """持倉時除權息日 → 現金股利入帳。"""
        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = [d.date() for d in dates]
        data = pd.DataFrame(
            {
                "close": [100.0, 100.0, 95.0, 95.0, 95.0],
                "raw_close": [100.0, 100.0, 95.0, 95.0, 95.0],
                "open": [100.0, 100.0, 95.0, 95.0, 95.0],
                "high": [101.0, 101.0, 96.0, 96.0, 96.0],
                "raw_high": [101.0, 101.0, 96.0, 96.0, 96.0],
                "low": [99.0, 99.0, 94.0, 94.0, 94.0],
                "raw_low": [99.0, 99.0, 94.0, 94.0, 94.0],
                "volume": [1_000_000] * 5,
            },
            index=idx,
        )
        # 買入 day0，day2 除息 5 元，day4 賣出
        signals = pd.Series([1, 0, 0, 0, -1], index=idx)
        dividends = pd.DataFrame(
            {"cash_dividend": [5.0], "stock_dividend": [0.0]},
            index=[idx[2]],
        )

        config = BacktestConfig(
            initial_capital=1_000_000,
            commission_rate=0.0,
            tax_rate=0.0,
            slippage=0.0,
        )
        engine = _make_engine(data, signals, dividends, config)
        result = engine.run()

        # 買 10000 股 @100，capital = 0
        # day2 除息 → cash += 10000 * 5 = 50000
        # day4 賣出 @95 → revenue = 10000 * 95 = 950000
        # final = 50000 + 950000 = 1000000
        assert result.final_capital == pytest.approx(1_000_000, abs=1.0)

    def test_stock_dividend_increases_shares(self):
        """持倉時除權日 → 股票股利增加持股。"""
        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = [d.date() for d in dates]
        data = pd.DataFrame(
            {
                "close": [100.0, 100.0, 90.0, 90.0, 90.0],
                "raw_close": [100.0, 100.0, 90.0, 90.0, 90.0],
                "open": [100.0, 100.0, 90.0, 90.0, 90.0],
                "high": [101.0, 101.0, 91.0, 91.0, 91.0],
                "raw_high": [101.0, 101.0, 91.0, 91.0, 91.0],
                "low": [99.0, 99.0, 89.0, 89.0, 89.0],
                "raw_low": [99.0, 99.0, 89.0, 89.0, 89.0],
                "volume": [1_000_000] * 5,
            },
            index=idx,
        )
        # stock_dividend=1.0 → 每股配 0.1 股
        signals = pd.Series([1, 0, 0, 0, -1], index=idx)
        dividends = pd.DataFrame(
            {"cash_dividend": [0.0], "stock_dividend": [1.0]},
            index=[idx[2]],
        )

        config = BacktestConfig(
            initial_capital=1_000_000,
            commission_rate=0.0,
            tax_rate=0.0,
            slippage=0.0,
        )
        engine = _make_engine(data, signals, dividends, config)
        result = engine.run()

        # 買 10000 股 @100, capital=0
        # day2 除權 → shares = 10000 + int(10000 * 1.0 / 10) = 10000 + 1000 = 11000
        # day4 賣出 @90 → 11000 * 90 = 990000
        assert result.final_capital == pytest.approx(990_000, abs=1.0)
        assert result.total_trades >= 1

    def test_no_dividend_when_not_holding(self):
        """無持倉時除權息 → 不影響資金。"""
        dates = pd.bdate_range("2024-01-01", periods=3)
        idx = [d.date() for d in dates]
        data = pd.DataFrame(
            {
                "close": [100.0, 95.0, 95.0],
                "raw_close": [100.0, 95.0, 95.0],
                "open": [100.0, 95.0, 95.0],
                "high": [101.0, 96.0, 96.0],
                "raw_high": [101.0, 96.0, 96.0],
                "low": [99.0, 94.0, 94.0],
                "raw_low": [99.0, 94.0, 94.0],
                "volume": [1_000_000] * 3,
            },
            index=idx,
        )
        signals = pd.Series([0, 0, 0], index=idx)
        dividends = pd.DataFrame(
            {"cash_dividend": [5.0], "stock_dividend": [0.0]},
            index=[idx[1]],
        )

        config = BacktestConfig(initial_capital=1_000_000, commission_rate=0.0, tax_rate=0.0, slippage=0.0)
        engine = _make_engine(data, signals, dividends, config)
        result = engine.run()

        assert result.final_capital == pytest.approx(1_000_000, abs=0.01)

    def test_raw_close_used_for_trading(self):
        """有 raw_close 欄位時，交易使用 raw_close 而非 close。"""
        dates = pd.bdate_range("2024-01-01", periods=4)
        idx = [d.date() for d in dates]
        data = pd.DataFrame(
            {
                "close": [90.0, 91.0, 92.0, 93.0],  # 調整後價格
                "raw_close": [100.0, 101.0, 102.0, 103.0],  # 原始價格
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "raw_high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "raw_low": [99.0, 100.0, 101.0, 102.0],
                "volume": [1_000_000] * 4,
            },
            index=idx,
        )
        signals = pd.Series([1, 0, 0, -1], index=idx)

        config = BacktestConfig(
            initial_capital=1_000_000,
            commission_rate=0.0,
            tax_rate=0.0,
            slippage=0.0,
        )
        engine = _make_engine(data, signals, config=config)
        result = engine.run()

        # 用 raw_close 買 @100 → 10000 股
        # 用 raw_close 賣 @103 → 10000 * 103 = 1_030_000
        assert result.trades[0].entry_price == pytest.approx(100.0, abs=0.1)
        assert result.trades[0].exit_price == pytest.approx(103.0, abs=0.1)
