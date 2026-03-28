"""測試 src/visualization/charts.py — Dashboard 純計算函數。

D4: 從 charts.py 抽出的純函數測試（不依賴 Streamlit / Plotly 渲染）。
"""

from datetime import date

import pandas as pd

from src.visualization.charts import (
    compute_drawdown_series,
    simulate_equity_curve,
    transform_calendar_heatmap_data,
)

# ---------------------------------------------------------------------------
# simulate_equity_curve
# ---------------------------------------------------------------------------


class TestSimulateEquityCurve:
    def test_no_trades(self):
        """無交易時權益應等於初始資金。"""
        dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
        closes = [100.0, 101.0, 102.0]
        equity = simulate_equity_curve(dates, closes, {}, {}, initial_capital=1_000_000)
        assert len(equity) == 3
        assert all(e == 1_000_000 for e in equity)

    def test_buy_increases_equity_on_rise(self):
        """買入後股價上漲，權益應大於初始。"""
        dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
        closes = [100.0, 110.0, 120.0]
        buy = {date(2024, 1, 1): 100.0}
        equity = simulate_equity_curve(dates, closes, buy, {}, initial_capital=100_000)
        # 第3天（股價120）時權益應大於初始
        assert equity[2] > 100_000

    def test_buy_sell_roundtrip(self):
        """買賣一輪後有手續費扣除。"""
        dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
        closes = [100.0, 100.0, 100.0]
        buy = {date(2024, 1, 1): 100.0}
        sell = {date(2024, 1, 2): 100.0}
        equity = simulate_equity_curve(dates, closes, buy, sell, initial_capital=100_000)
        # 同價買賣，因手續費+稅，資金應小於初始
        assert equity[2] < 100_000

    def test_length_matches_dates(self):
        dates = [date(2024, 1, i) for i in range(1, 11)]
        closes = [100.0] * 10
        equity = simulate_equity_curve(dates, closes, {}, {})
        assert len(equity) == 10

    def test_single_day(self):
        dates = [date(2024, 1, 1)]
        closes = [100.0]
        equity = simulate_equity_curve(dates, closes, {}, {}, initial_capital=500_000)
        assert equity == [500_000]

    def test_buy_on_last_day(self):
        """最後一天買入，持倉市值應反映。"""
        dates = [date(2024, 1, 1), date(2024, 1, 2)]
        closes = [100.0, 100.0]
        buy = {date(2024, 1, 2): 100.0}
        equity = simulate_equity_curve(dates, closes, buy, {}, initial_capital=100_000)
        assert equity[0] == 100_000  # 第一天無交易
        # 第二天買入，權益應接近但因手續費略低
        assert 99_000 < equity[1] < 101_000


# ---------------------------------------------------------------------------
# compute_drawdown_series
# ---------------------------------------------------------------------------


class TestComputeDrawdownSeries:
    def test_empty(self):
        assert compute_drawdown_series([]) == []

    def test_monotone_up(self):
        """持續上漲，回撤應為 0。"""
        equity = [100, 110, 120, 130]
        dd = compute_drawdown_series(equity)
        assert all(d == 0.0 for d in dd)

    def test_simple_drawdown(self):
        """峰值 200，當前 180 → 回撤 10%。"""
        equity = [100, 200, 180]
        dd = compute_drawdown_series(equity)
        assert dd[0] == 0.0
        assert dd[1] == 0.0
        assert abs(dd[2] - 10.0) < 0.01

    def test_recovery(self):
        """回撤後恢復到新高，回撤歸零。"""
        equity = [100, 200, 150, 250]
        dd = compute_drawdown_series(equity)
        assert dd[3] == 0.0  # 新高點

    def test_length_preserved(self):
        equity = [100] * 5
        dd = compute_drawdown_series(equity)
        assert len(dd) == 5

    def test_all_same(self):
        """平盤無回撤。"""
        equity = [1000, 1000, 1000]
        dd = compute_drawdown_series(equity)
        assert all(d == 0.0 for d in dd)


# ---------------------------------------------------------------------------
# transform_calendar_heatmap_data
# ---------------------------------------------------------------------------


class TestTransformCalendarHeatmapData:
    def test_empty_df(self):
        y, x, z = transform_calendar_heatmap_data(pd.DataFrame(), "value")
        assert y == []
        assert x == []
        assert z == []

    def test_basic_transform(self):
        df = pd.DataFrame(
            {
                "scan_date": [
                    date(2024, 1, 5),
                    date(2024, 1, 12),
                    date(2024, 1, 20),
                    date(2024, 2, 3),
                ],
                "count": [10, 15, 20, 8],
            }
        )
        y, x, z = transform_calendar_heatmap_data(df, "count")
        assert len(y) >= 1  # 至少 1 個年月
        assert len(x) >= 1  # 至少 1 個週
        assert len(z) >= 1  # 至少 1 列

    def test_month_sorting(self):
        df = pd.DataFrame(
            {
                "scan_date": [
                    date(2024, 3, 1),
                    date(2024, 1, 1),
                    date(2024, 2, 1),
                ],
                "value": [30, 10, 20],
            }
        )
        y, x, z = transform_calendar_heatmap_data(df, "value")
        # 應按年月排序
        assert y == sorted(y)

    def test_week_of_month(self):
        """1號在第1週，28號在第4週。"""
        df = pd.DataFrame(
            {
                "scan_date": [date(2024, 1, 1), date(2024, 1, 28)],
                "value": [1.0, 2.0],
            }
        )
        y, x, z = transform_calendar_heatmap_data(df, "value")
        # 應有第1週和第4週
        assert "第1週" in x
        assert "第4週" in x

    def test_aggregation(self):
        """同一年月同一週的值應被平均。"""
        df = pd.DataFrame(
            {
                "scan_date": [date(2024, 1, 1), date(2024, 1, 2)],
                "value": [10.0, 20.0],
            }
        )
        y, x, z = transform_calendar_heatmap_data(df, "value")
        # 兩個日期都在 1 月第 1 週，平均為 15
        assert len(z) == 1
        assert any(v is not None and abs(v - 15.0) < 0.01 for row in z for v in row)
