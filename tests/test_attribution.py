"""tests/test_attribution.py — 因子歸因分析純函數測試。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.backtest.attribution import FactorAttribution

# ------------------------------------------------------------------ #
#  測試輔助
# ------------------------------------------------------------------ #


def _make_price_df(n: int = 60, base_price: float = 100.0, base_vol: float = 1_000_000.0) -> pd.DataFrame:
    """建立 n 個交易日的模擬 OHLCV + RSI DataFrame（DatetimeIndex）。"""
    idx = pd.bdate_range("2024-01-01", periods=n)
    np.random.seed(42)
    close = base_price + np.cumsum(np.random.randn(n) * 2)
    close = np.maximum(close, 10.0)
    volume = base_vol + np.random.randn(n) * 100_000
    volume = np.maximum(volume, 1000.0)
    rsi = 50.0 + np.random.randn(n) * 10
    rsi = np.clip(rsi, 20, 80)
    return pd.DataFrame({"close": close, "volume": volume, "rsi_14": rsi}, index=idx)


def _make_trade(entry_idx: int, return_pct: float, data: pd.DataFrame) -> dict:
    """建立交易 dict（entry_date 對應 data.index[entry_idx]）。"""
    return {
        "entry_date": data.index[entry_idx].date(),
        "return_pct": return_pct,
    }


@dataclass
class _MockTradeRecord:
    entry_date: date
    return_pct: float
    exit_date: date = field(default_factory=date.today)
    exit_price: float = 100.0


@dataclass
class _MockBacktestResult:
    trades: list[_MockTradeRecord]


# ------------------------------------------------------------------ #
#  測試：基本計算
# ------------------------------------------------------------------ #


class TestComputeBasic:
    def test_compute_returns_result_with_five_factors(self):
        """5 筆交易應回傳含五個因子相關係數的結果。"""
        data = _make_price_df(60)
        fa = FactorAttribution()

        trades = [_MockTradeRecord(entry_date=data.index[30 + i].date(), return_pct=float(i)) for i in range(5)]
        result = fa.compute(_MockBacktestResult(trades=trades), data)

        assert result.n_trades == 5
        assert set(result.correlations.keys()) == {"momentum", "reversal", "quality", "size", "liquidity"}

    def test_compute_from_df_same_as_compute(self):
        """compute_from_df 與 compute 對相同資料應得相同結果。"""
        data = _make_price_df(60)
        fa = FactorAttribution()

        trade_records = [
            _MockTradeRecord(entry_date=data.index[25 + i].date(), return_pct=float(i * 2 - 4)) for i in range(5)
        ]
        mock_result = _MockBacktestResult(trades=trade_records)

        df_rows = [{"entry_date": t.entry_date, "return_pct": t.return_pct} for t in trade_records]
        trades_df = pd.DataFrame(df_rows)

        r1 = fa.compute(mock_result, data)
        r2 = fa.compute_from_df(trades_df, data)

        assert r1.n_trades == r2.n_trades
        for fname in FactorAttribution.FACTOR_LABELS:
            c1 = r1.correlations[fname]
            c2 = r2.correlations[fname]
            if c1 is None:
                assert c2 is None
            else:
                assert abs(c1 - c2) < 1e-9


class TestTooFewTrades:
    def test_fewer_than_min_returns_empty(self):
        """少於 MIN_TRADES 筆交易應回傳空結果（n_trades=0）。"""
        data = _make_price_df(60)
        fa = FactorAttribution()

        trades = [_MockTradeRecord(entry_date=data.index[30].date(), return_pct=5.0)]
        result = fa.compute(_MockBacktestResult(trades=trades), data)

        assert result.n_trades == 0
        assert all(v is None for v in result.correlations.values())

    def test_no_trades_returns_empty(self):
        data = _make_price_df(60)
        fa = FactorAttribution()
        result = fa.compute(_MockBacktestResult(trades=[]), data)
        assert result.n_trades == 0


class TestMissingRsi:
    def test_quality_none_when_no_rsi_column(self):
        """data 中無 RSI 欄位時，quality 因子相關係數應為 None。"""
        data = _make_price_df(60).drop(columns=["rsi_14"])
        fa = FactorAttribution()

        trades = [_MockTradeRecord(entry_date=data.index[25 + i].date(), return_pct=float(i)) for i in range(5)]
        result = fa.compute(_MockBacktestResult(trades=trades), data)

        assert result.correlations["quality"] is None
        # 其他因子不受影響
        assert result.correlations["momentum"] is not None or result.correlations["size"] is not None


# ------------------------------------------------------------------ #
#  測試：因子暴露計算純函數
# ------------------------------------------------------------------ #


class TestMomentumExposure:
    def test_momentum_20d_positive_trend(self):
        """單調遞增序列的 20 日動能應為正。"""
        close = pd.Series(range(50, 110), dtype=float)
        result = FactorAttribution._momentum(close, idx=30, lookback=20)
        assert result is not None
        assert result > 0

    def test_momentum_insufficient_history_returns_none(self):
        """歷史不足 lookback 天時應回傳 None。"""
        close = pd.Series(range(10), dtype=float)
        result = FactorAttribution._momentum(close, idx=5, lookback=20)
        assert result is None

    def test_momentum_value_correct(self):
        """驗證動能計算：(close[idx] / close[idx-lookback] - 1) × 100。"""
        close = pd.Series([100.0] * 25 + [110.0])
        result = FactorAttribution._momentum(close, idx=25, lookback=20)
        assert result is not None
        assert abs(result - 10.0) < 1e-9  # (110/100 - 1) * 100 = 10%


class TestReversalExposure:
    def test_reversal_negative_for_downtrend(self):
        """進場前 5 日下跌的 reversal 值應為負（代表超賣反彈機會）。"""
        # close 在 idx-5 = 110，idx = 90（下跌）
        vals = [100.0] * 20 + [110.0] * 5 + [90.0]
        close = pd.Series(vals)
        result = FactorAttribution._momentum(close, idx=len(vals) - 1, lookback=5)
        assert result is not None
        assert result < 0


class TestSizeExposure:
    def test_log_avg_volume_positive(self):
        """對數均量應為正值（量 > 1）。"""
        volume = pd.Series([1_000_000.0] * 30)
        result = FactorAttribution._log_avg_volume(volume, idx=25, lookback=20)
        assert result is not None
        assert abs(result - math.log(1_000_000.0)) < 0.01

    def test_log_avg_volume_none_when_no_volume(self):
        result = FactorAttribution._log_avg_volume(None, idx=25, lookback=20)
        assert result is None


class TestLiquidityExposure:
    def test_relative_volume_high_volume_day(self):
        """進場日量為均量 3 倍時，relative_volume 應約等於 3。"""
        avg_vol = 1_000_000.0
        volume = pd.Series([avg_vol] * 20 + [avg_vol * 3])
        result = FactorAttribution._relative_volume(volume, idx=20, lookback=20)
        assert result is not None
        assert abs(result - 3.0) < 0.05

    def test_relative_volume_none_when_no_volume(self):
        result = FactorAttribution._relative_volume(None, idx=10, lookback=20)
        assert result is None


class TestCorrelationDirection:
    def test_positive_correlation_when_momentum_predicts_return(self):
        """確認 Pearson 相關係數方向正確：人工建構完全正相關案例。"""
        # 建立一個 close 序列，使每筆交易的 20 日動能值各不相同
        # 交易在 idx=25,26,27,28,29,30（連續 6 天）
        # close[5..24] 各自不同，使 20 日動能有明顯差異
        n_days = 50
        # 建立 close：前 25 天從不同起點出發，後面固定 100
        close_vals = [100.0] * n_days
        # 讓 idx=25 時 close[5]=60 → momentum = (100/60-1)*100 = +66.7%
        # 讓 idx=26 時 close[6]=70 → momentum = (100/70-1)*100 = +42.9%
        # 讓 idx=27 時 close[7]=80 → momentum = (100/80-1)*100 = +25.0%
        # 讓 idx=28 時 close[8]=90 → momentum = (100/90-1)*100 = +11.1%
        # 讓 idx=29 時 close[9]=110 → momentum = (100/110-1)*100 = -9.1%
        # 讓 idx=30 時 close[10]=120 → momentum = (100/120-1)*100 = -16.7%
        for i, past_val in enumerate([60.0, 70.0, 80.0, 90.0, 110.0, 120.0]):
            close_vals[5 + i] = past_val
        close = pd.Series(close_vals, dtype=float)
        volume = pd.Series([1_000_000.0] * n_days)
        rsi = pd.Series([50.0] * n_days)
        data = pd.DataFrame(
            {"close": close.values, "volume": volume.values, "rsi_14": rsi.values},
            index=pd.bdate_range("2024-01-01", periods=n_days),
        )

        fa = FactorAttribution()

        # 報酬率與動能完全正相關（動能高 → 報酬高）
        # 動能（由高到低）：66.7, 42.9, 25.0, 11.1, -9.1, -16.7
        # 報酬率對應：  5.0,  4.0,  3.0,  2.0,   1.0,   0.0
        trade_indices = list(range(25, 31))  # 6 筆交易
        returns = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        trades = [
            _MockTradeRecord(
                entry_date=data.index[idx].date(),
                return_pct=ret,
            )
            for idx, ret in zip(trade_indices, returns)
        ]

        result = fa.compute(_MockBacktestResult(trades=trades), data)
        corr = result.correlations.get("momentum")
        assert corr is not None
        # 動能與報酬完全正相關（r ≈ 1.0）
        assert corr > 0.9
