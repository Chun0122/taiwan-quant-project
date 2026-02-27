"""技術指標計算測試 — compute_indicators_from_df() 純函數。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int, base: float = 100.0) -> pd.DataFrame:
    """建立 N 天的 OHLCV 測試資料。"""
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for i, dt in enumerate(dates):
        close = base + i * 0.5
        rows.append(
            {
                "date": dt.date(),
                "open": close - 0.3,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000 + i * 10_000,
            }
        )
    return pd.DataFrame(rows).set_index("date")


@pytest.fixture()
def df_100():
    """100 天的 OHLCV，足以計算所有指標。"""
    return _make_ohlcv(100)


@pytest.fixture()
def df_20():
    """20 天的 OHLCV。"""
    return _make_ohlcv(20)


class TestComputeIndicatorsFromDf:
    """compute_indicators_from_df() 純函數測試。"""

    def test_output_columns(self, df_100):
        from src.features.indicators import compute_indicators_from_df

        result = compute_indicators_from_df(df_100)
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
        assert expected_cols == set(result.columns)

    def test_output_length_matches_input(self, df_100):
        from src.features.indicators import compute_indicators_from_df

        result = compute_indicators_from_df(df_100)
        assert len(result) == len(df_100)

    def test_sma5_nan_start(self, df_100):
        """SMA(5) 前 4 天應為 NaN。"""
        from src.features.indicators import compute_indicators_from_df

        result = compute_indicators_from_df(df_100)
        assert result["sma_5"].isna().sum() >= 4
        assert result["sma_5"].dropna().iloc[0] == pytest.approx(df_100["close"].iloc[:5].mean(), rel=1e-3)

    def test_sma60_nan_start(self, df_100):
        """SMA(60) 前 59 天應為 NaN。"""
        from src.features.indicators import compute_indicators_from_df

        result = compute_indicators_from_df(df_100)
        assert result["sma_60"].isna().sum() >= 59

    def test_sma_correctness(self, df_100):
        """SMA(20) 數值正確。"""
        from src.features.indicators import compute_indicators_from_df

        result = compute_indicators_from_df(df_100)
        # 第 20 天（index=19）的 SMA(20) 應等於前 20 天收盤均值
        expected = df_100["close"].iloc[:20].mean()
        sma20_first = result["sma_20"].dropna().iloc[0]
        assert sma20_first == pytest.approx(expected, rel=1e-3)

    def test_rsi_range(self, df_100):
        """RSI 值應在 0~100 之間。"""
        from src.features.indicators import compute_indicators_from_df

        result = compute_indicators_from_df(df_100)
        rsi = result["rsi_14"].dropna()
        assert len(rsi) > 0
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_rsi_uptrend_high(self, df_100):
        """持續上漲序列的 RSI 應偏高（>50）。"""
        from src.features.indicators import compute_indicators_from_df

        result = compute_indicators_from_df(df_100)
        rsi_last = result["rsi_14"].dropna().iloc[-1]
        # df_100 是持續上漲的，RSI 應偏高
        assert rsi_last > 50

    def test_macd_produced(self, df_100):
        """MACD 三條線應有非 NaN 值。"""
        from src.features.indicators import compute_indicators_from_df

        result = compute_indicators_from_df(df_100)
        assert result["macd"].dropna().shape[0] > 0
        assert result["macd_signal"].dropna().shape[0] > 0
        assert result["macd_hist"].dropna().shape[0] > 0

    def test_macd_hist_equals_diff(self, df_100):
        """MACD_Hist 應等於 MACD - MACD_Signal。"""
        from src.features.indicators import compute_indicators_from_df

        result = compute_indicators_from_df(df_100)
        valid = result.dropna(subset=["macd", "macd_signal", "macd_hist"])
        diff = valid["macd"] - valid["macd_signal"]
        np.testing.assert_allclose(valid["macd_hist"].values, diff.values, atol=1e-3)

    def test_bollinger_bands_order(self, df_100):
        """Bollinger Bands: upper >= middle >= lower。"""
        from src.features.indicators import compute_indicators_from_df

        result = compute_indicators_from_df(df_100)
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        assert len(valid) > 0
        assert (valid["bb_upper"] >= valid["bb_middle"] - 1e-6).all()
        assert (valid["bb_middle"] >= valid["bb_lower"] - 1e-6).all()

    def test_bollinger_middle_equals_sma20(self, df_100):
        """BB middle 應等於 SMA(20)。"""
        from src.features.indicators import compute_indicators_from_df

        result = compute_indicators_from_df(df_100)
        valid = result.dropna(subset=["bb_middle", "sma_20"])
        np.testing.assert_allclose(valid["bb_middle"].values, valid["sma_20"].values, rtol=1e-3)

    def test_short_data_no_crash(self):
        """資料不足時不應崩潰，只是很多 NaN。"""
        from src.features.indicators import compute_indicators_from_df

        df = _make_ohlcv(5)
        result = compute_indicators_from_df(df)
        assert len(result) == 5
        # SMA(60) 全是 NaN
        assert result["sma_60"].isna().all()

    def test_single_row(self):
        """只有 1 天的資料。"""
        from src.features.indicators import compute_indicators_from_df

        df = _make_ohlcv(1)
        result = compute_indicators_from_df(df)
        assert len(result) == 1
