"""市場總覽相關模組測試。

測試對象：
- data_loader: load_market_breadth, load_market_breadth_stats, load_top_institutional, load_taiex_history,
  load_announcements
- charts: plot_taiex_regime, plot_market_breadth_area, plot_institutional_ranking, plot_sector_treemap,
  plot_institutional_cumulative, plot_candlestick (enhanced)
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.data.schema import Announcement, DailyPrice, InstitutionalInvestor, StockInfo

# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(autouse=True)
def _patch_data_loader_session(db_session, monkeypatch):
    """確保 data_loader 模組使用測試 session。"""
    import src.visualization.data_loader as dl_mod

    @contextmanager
    def _mock_get_session():
        yield db_session

    monkeypatch.setattr(dl_mod, "get_session", _mock_get_session)
    monkeypatch.setattr(dl_mod, "init_db", lambda: None)


@pytest.fixture()
def seed_taiex(db_session):
    """植入 TAIEX 測試資料（150 筆）。"""
    base = date(2025, 1, 2)
    rows = []
    for i in range(150):
        d = base + timedelta(days=i)
        close = 18000 + i * 10
        rows.append(
            DailyPrice(
                stock_id="TAIEX",
                date=d,
                open=close - 50,
                high=close + 80,
                low=close - 80,
                close=close,
                volume=5_000_000_000,
                turnover=200_000_000_000,
                spread=10.0 if i % 3 != 2 else -10.0,
            )
        )
    db_session.add_all(rows)
    db_session.flush()


@pytest.fixture()
def seed_market_data(db_session):
    """植入多支股票日K線（含 spread）供市場廣度測試。"""
    base = date(2025, 6, 1)
    stocks = ["2330", "2317", "2454", "3008", "1301"]
    rows = []
    for i in range(30):
        d = base + timedelta(days=i)
        for j, sid in enumerate(stocks):
            spread = 1.0 if j % 3 == 0 else (-1.0 if j % 3 == 1 else 0.0)
            if i % 2 == 1:
                spread = -spread
            rows.append(
                DailyPrice(
                    stock_id=sid,
                    date=d,
                    open=100.0,
                    high=105.0,
                    low=95.0,
                    close=100.0 + spread,
                    volume=1_000_000,
                    turnover=100_000_000,
                    spread=spread,
                )
            )
    db_session.add_all(rows)
    db_session.flush()


@pytest.fixture()
def seed_institutional(db_session):
    """植入法人買賣超測試資料。"""
    base = date(2025, 6, 1)
    stocks = ["2330", "2317", "2454"]
    rows = []
    for i in range(10):
        d = base + timedelta(days=i)
        for sid in stocks:
            rows.append(
                InstitutionalInvestor(
                    stock_id=sid,
                    date=d,
                    name="Foreign_Investor",
                    buy=10000,
                    sell=5000,
                    net=5000 if sid == "2330" else -3000,
                )
            )
            rows.append(
                InstitutionalInvestor(
                    stock_id=sid,
                    date=d,
                    name="Investment_Trust",
                    buy=3000,
                    sell=1000,
                    net=2000,
                )
            )
    db_session.add_all(rows)
    # 植入 StockInfo
    for sid, name in [("2330", "台積電"), ("2317", "鴻海"), ("2454", "聯發科")]:
        db_session.add(StockInfo(stock_id=sid, stock_name=name, industry_category="半導體"))
    db_session.flush()


# ------------------------------------------------------------------ #
#  data_loader 測試
# ------------------------------------------------------------------ #


class TestLoadTaiexHistory:
    def test_returns_dataframe(self, seed_taiex):
        from src.visualization.data_loader import load_taiex_history

        df = load_taiex_history(days=30)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30
        assert "close" in df.columns
        assert "date" in df.columns

    def test_empty_when_no_data(self, db_session):
        from src.visualization.data_loader import load_taiex_history

        df = load_taiex_history(days=10)
        assert df.empty

    def test_sorted_by_date(self, seed_taiex):
        from src.visualization.data_loader import load_taiex_history

        df = load_taiex_history(days=50)
        dates = df["date"].tolist()
        assert dates == sorted(dates)


class TestLoadMarketBreadth:
    def test_returns_breadth(self, seed_market_data):
        from src.visualization.data_loader import load_market_breadth

        df = load_market_breadth(days=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(c in df.columns for c in ["rising", "falling", "flat", "total_volume"])

    def test_empty_when_no_data(self, db_session):
        from src.visualization.data_loader import load_market_breadth

        df = load_market_breadth(days=10)
        assert df.empty

    def test_excludes_taiex(self, seed_taiex, seed_market_data):
        """確保 TAIEX 不被計入漲跌家數。"""
        from src.visualization.data_loader import load_market_breadth

        df = load_market_breadth(days=5)
        if not df.empty:
            max_stocks = (df["rising"] + df["falling"] + df["flat"]).max()
            assert max_stocks <= 5


class TestLoadMarketBreadthStats:
    def test_returns_stats(self, seed_market_data):
        from src.visualization.data_loader import load_market_breadth_stats

        stats = load_market_breadth_stats(windows=[1, 5])
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 2
        assert stats.iloc[0]["window"] == "當日"

    def test_empty_when_no_data(self, db_session):
        from src.visualization.data_loader import load_market_breadth_stats

        stats = load_market_breadth_stats()
        assert stats.empty


class TestLoadTopInstitutional:
    def test_returns_ranking(self, seed_institutional):
        from src.visualization.data_loader import load_top_institutional

        df = load_top_institutional(lookback=5, top_n=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "foreign" in df.columns
        assert "trust" in df.columns
        assert "total" in df.columns
        assert "stock_name" in df.columns

    def test_empty_when_no_data(self, db_session):
        from src.visualization.data_loader import load_top_institutional

        df = load_top_institutional(lookback=5, top_n=10)
        assert df.empty

    def test_respects_top_n(self, seed_institutional):
        from src.visualization.data_loader import load_top_institutional

        df = load_top_institutional(lookback=5, top_n=2)
        assert len(df) <= 2


# ------------------------------------------------------------------ #
#  charts 測試
# ------------------------------------------------------------------ #


class TestPlotTaiexRegime:
    def test_returns_figure(self):
        from src.visualization.charts import plot_taiex_regime

        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=130),
                "open": range(100, 230),
                "high": range(101, 231),
                "low": range(99, 229),
                "close": range(100, 230),
            }
        )
        regime_info = {"regime": "bull", "signals": {}, "taiex_close": 229}
        fig = plot_taiex_regime(df, regime_info)
        assert isinstance(fig, go.Figure)

    def test_empty_data(self):
        from src.visualization.charts import plot_taiex_regime

        fig = plot_taiex_regime(pd.DataFrame(), {"regime": "sideways"})
        assert isinstance(fig, go.Figure)


class TestPlotMarketBreadthArea:
    def test_returns_figure(self):
        from src.visualization.charts import plot_market_breadth_area

        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=20),
                "rising": [500] * 20,
                "falling": [300] * 20,
                "flat": [100] * 20,
            }
        )
        fig = plot_market_breadth_area(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3

    def test_empty_data(self):
        from src.visualization.charts import plot_market_breadth_area

        fig = plot_market_breadth_area(pd.DataFrame())
        assert isinstance(fig, go.Figure)


class TestPlotInstitutionalRanking:
    def test_returns_figure(self):
        from src.visualization.charts import plot_institutional_ranking

        df = pd.DataFrame(
            {
                "stock_id": ["2330", "2317"],
                "stock_name": ["台積電", "鴻海"],
                "foreign": [50000, -30000],
                "trust": [20000, 10000],
                "dealer": [5000, -5000],
                "total": [75000, -25000],
            }
        )
        fig = plot_institutional_ranking(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3

    def test_empty_data(self):
        from src.visualization.charts import plot_institutional_ranking

        fig = plot_institutional_ranking(pd.DataFrame())
        assert isinstance(fig, go.Figure)


class TestPlotSectorTreemap:
    def test_returns_figure(self):
        from src.visualization.charts import plot_sector_treemap

        df = pd.DataFrame(
            {
                "industry": ["半導體", "金融", "電子零組件"],
                "total_net": [1000000, -500000, 300000],
                "avg_return_pct": [2.5, -1.0, 0.5],
            }
        )
        fig = plot_sector_treemap(df)
        assert isinstance(fig, go.Figure)

    def test_empty_data(self):
        from src.visualization.charts import plot_sector_treemap

        fig = plot_sector_treemap(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_filters_zero_net(self):
        from src.visualization.charts import plot_sector_treemap

        df = pd.DataFrame(
            {
                "industry": ["半導體", "金融"],
                "total_net": [0, 0],
                "avg_return_pct": [0.0, 0.0],
            }
        )
        fig = plot_sector_treemap(df)
        assert isinstance(fig, go.Figure)


# ------------------------------------------------------------------ #
#  load_announcements 測試
# ------------------------------------------------------------------ #


class TestLoadAnnouncements:
    def test_returns_empty_when_no_data(self, db_session):
        from src.visualization.data_loader import load_announcements

        df = load_announcements("2330", "2025-01-01", "2025-12-31")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "subject" in df.columns
        assert "sentiment" in df.columns

    def test_returns_df_with_announcements(self, db_session):
        from src.visualization.data_loader import load_announcements

        db_session.add(
            Announcement(
                stock_id="2330",
                date=date(2025, 6, 10),
                seq="1",
                subject="測試公告",
                sentiment=1,
                spoke_time="09:30",
            )
        )
        db_session.flush()

        df = load_announcements("2330", "2025-01-01", "2025-12-31")
        assert len(df) == 1
        assert df.iloc[0]["subject"] == "測試公告"
        assert df.iloc[0]["sentiment"] == 1
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_date_range_filter(self, db_session):
        from src.visualization.data_loader import load_announcements

        for i, d in enumerate([date(2025, 1, 1), date(2025, 6, 1), date(2025, 12, 1)]):
            db_session.add(
                Announcement(
                    stock_id="2330",
                    date=d,
                    seq=str(i),
                    subject=f"公告 {i}",
                    sentiment=0,
                )
            )
        db_session.flush()

        df = load_announcements("2330", "2025-06-01", "2025-12-01")
        assert len(df) == 2


# ------------------------------------------------------------------ #
#  plot_institutional_cumulative 測試
# ------------------------------------------------------------------ #


class TestPlotInstitutionalCumulative:
    def test_returns_figure_with_data(self):
        from src.visualization.charts import plot_institutional_cumulative

        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=10),
                "name": ["Foreign_Investor"] * 10,
                "net": [5000] * 10,
            }
        )
        fig = plot_institutional_cumulative(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # 確認是折線圖
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) > 0

    def test_empty_returns_figure(self):
        from src.visualization.charts import plot_institutional_cumulative

        fig = plot_institutional_cumulative(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_cumulative_values_increase(self):
        from src.visualization.charts import plot_institutional_cumulative

        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=5),
                "name": ["Foreign_Investor"] * 5,
                "net": [1000] * 5,
            }
        )
        fig = plot_institutional_cumulative(df)
        y_values = list(fig.data[0].y)
        # cumsum of [1000]*5 should be [1000, 2000, 3000, 4000, 5000]
        assert y_values[-1] > y_values[0]


# ------------------------------------------------------------------ #
#  plot_candlestick 增強測試
# ------------------------------------------------------------------ #


class TestPlotCandlestickEnhanced:
    @pytest.fixture()
    def sample_df(self):
        dates = pd.date_range("2025-01-01", periods=30)
        return pd.DataFrame(
            {
                "date": dates,
                "open": [100.0] * 30,
                "high": [105.0] * 30,
                "low": [95.0] * 30,
                "close": [101.0] * 30,
                "volume": [1_000_000] * 30,
                "rsi_14": [50.0] * 30,
                "macd": [0.5] * 30,
                "macd_signal": [0.3] * 30,
                "macd_hist": [0.2] * 30,
            }
        )

    def test_selected_indicators_filter(self, sample_df):
        from src.visualization.charts import plot_candlestick

        # 只勾 RSI，不勾 MACD
        fig = plot_candlestick(sample_df, selected_indicators={"rsi"})
        macd_traces = [t for t in fig.data if t.name in ("MACD", "Signal", "Histogram")]
        rsi_traces = [t for t in fig.data if t.name == "RSI"]
        assert len(macd_traces) == 0
        assert len(rsi_traces) == 1

    def test_volume_overlay_in_row1(self, sample_df):
        from src.visualization.charts import plot_candlestick

        fig = plot_candlestick(sample_df, selected_indicators=set())
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar) and t.name == "成交量"]
        assert len(bar_traces) == 1

    def test_no_indicators_single_row(self, sample_df):
        from src.visualization.charts import plot_candlestick

        # 不勾任何指標 → 只有 1 row（K線），高度為 500
        fig = plot_candlestick(sample_df, selected_indicators=set())
        assert fig.layout.height == 500

    def test_with_announcements(self, sample_df):
        from src.visualization.charts import plot_candlestick

        df_ann = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-10", "2025-01-15"]),
                "subject": ["公告1", "公告2"],
                "sentiment": [1, -1],
                "spoke_time": ["09:30", "10:00"],
            }
        )
        fig = plot_candlestick(sample_df, selected_indicators=set(), df_announcements=df_ann)
        assert isinstance(fig, go.Figure)
        # add_vline 會在 layout.shapes 中新增 shape
        assert len(fig.layout.shapes) > 0

    def test_backward_compatible_no_args(self, sample_df):
        from src.visualization.charts import plot_candlestick

        # 不傳新參數，應能正常執行（預設顯示全部指標）
        fig = plot_candlestick(sample_df)
        assert isinstance(fig, go.Figure)
        rsi_traces = [t for t in fig.data if t.name == "RSI"]
        assert len(rsi_traces) == 1
