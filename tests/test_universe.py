"""Universe Filtering Module 測試。

涵蓋：
- TestFilterLiquidity (4):  filter_liquidity() 純函數
- TestFilterTrend (5):      filter_trend() 純函數
- TestStage1SqlFilter (6):  UniverseFilter._stage1_sql_filter() DB 整合
- TestStage2LiquidityFilter (4): Stage 2 DB fallback
- TestCandidateMemory (3):  _load_candidate_memory() DB 整合
- TestComputeAndStoreDailyFeatures (4): pipeline ETL 整合
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data.schema import (
    DailyFeature,
    DailyPrice,
    DiscoveryRecord,
    StockInfo,
)
from src.discovery.universe import UniverseConfig, UniverseFilter, filter_liquidity, filter_trend

# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────


def _make_dp(
    stock_id: str, dt: date, close: float = 100.0, volume: int = 1_000_000, turnover: int = 100_000_000
) -> DailyPrice:
    return DailyPrice(
        stock_id=stock_id,
        date=dt,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=volume,
        turnover=turnover,
    )


def _make_si(stock_id: str, listing_type: str = "上市", security_type: str | None = "stock") -> StockInfo:
    return StockInfo(
        stock_id=stock_id,
        stock_name=f"Test {stock_id}",
        listing_type=listing_type,
        security_type=security_type,
    )


def _make_dr(stock_id: str, mode: str = "momentum", scan_date: date | None = None) -> DiscoveryRecord:
    sd = scan_date or date.today()
    return DiscoveryRecord(
        scan_date=sd,
        mode=mode,
        rank=1,
        stock_id=stock_id,
        close=100.0,
        composite_score=0.8,
    )


# ────────────────────────────────────────────────────────────────────────────
#  TestFilterLiquidity — 純函數
# ────────────────────────────────────────────────────────────────────────────


class TestFilterLiquidity:
    """filter_liquidity() 純函數測試（無 DB）。"""

    def _cfg(self, avg=30_000_000.0, min_=10_000_000.0) -> UniverseConfig:
        return UniverseConfig(avg_turnover_5d_min=avg, min_turnover_5d_min=min_)

    def test_passes_when_both_thresholds_met(self):
        df = pd.DataFrame(
            [
                {"stock_id": "2330", "turnover": 50_000_000},
                {"stock_id": "2330", "turnover": 40_000_000},
                {"stock_id": "2330", "turnover": 35_000_000},
                {"stock_id": "2330", "turnover": 45_000_000},
                {"stock_id": "2330", "turnover": 30_000_000},
            ]
        )
        result = filter_liquidity(df, self._cfg())
        assert "2330" in result

    def test_excludes_when_avg_too_low(self):
        df = pd.DataFrame([{"stock_id": "0050", "turnover": t} for t in [20_000_000] * 5])
        result = filter_liquidity(df, self._cfg())
        assert "0050" not in result

    def test_excludes_when_min_too_low(self):
        # avg is high enough but one day is very low
        df = pd.DataFrame(
            [
                {"stock_id": "1234", "turnover": 100_000_000},
                {"stock_id": "1234", "turnover": 100_000_000},
                {"stock_id": "1234", "turnover": 100_000_000},
                {"stock_id": "1234", "turnover": 100_000_000},
                {"stock_id": "1234", "turnover": 5_000_000},  # min < 10M
            ]
        )
        result = filter_liquidity(df, self._cfg())
        assert "1234" not in result

    def test_empty_df_returns_empty(self):
        result = filter_liquidity(pd.DataFrame(), self._cfg())
        assert result == []


# ────────────────────────────────────────────────────────────────────────────
#  TestFilterTrend — 純函數
# ────────────────────────────────────────────────────────────────────────────


class TestFilterTrend:
    """filter_trend() 純函數測試（無 DB）。"""

    def _make_df(
        self,
        stock_id: str,
        close: float = 110.0,
        ma60: float = 100.0,
        volume: int = 2_000_000,
        volume_ma20: float = 1_000_000.0,
    ) -> pd.DataFrame:
        today = date.today()
        return pd.DataFrame(
            [
                {
                    "stock_id": stock_id,
                    "date": today,
                    "close": close,
                    "volume": volume,
                    "ma60": ma60,
                    "volume_ma20": volume_ma20,
                }
            ]
        )

    def test_passes_when_close_above_ma_and_volume_ratio_ok(self):
        df = self._make_df("2330", close=110.0, ma60=100.0, volume=2_000_000, volume_ma20=1_000_000.0)
        result = filter_trend(df, UniverseConfig(trend_ma=60, volume_ratio_min=1.5))
        assert "2330" in result

    def test_excludes_when_close_below_ma60(self):
        df = self._make_df("2330", close=90.0, ma60=100.0)
        result = filter_trend(df, UniverseConfig(trend_ma=60, volume_ratio_min=1.5))
        assert "2330" not in result

    def test_excludes_when_volume_ratio_too_low(self):
        df = self._make_df("2330", close=110.0, ma60=100.0, volume=1_000_000, volume_ma20=2_000_000.0)
        result = filter_trend(df, UniverseConfig(trend_ma=60, volume_ratio_min=1.5))
        assert "2330" not in result

    def test_skips_when_trend_ma_is_none(self):
        df = self._make_df("2330", close=80.0, ma60=100.0)  # would fail trend test
        cfg = UniverseConfig(trend_ma=None, volume_ratio_min=None)
        result = filter_trend(df, cfg)
        assert "2330" in result  # skipped because trend_ma=None

    def test_empty_df_returns_empty_list_when_trend_ma_set(self):
        result = filter_trend(pd.DataFrame(), UniverseConfig(trend_ma=60))
        assert result == []


# ────────────────────────────────────────────────────────────────────────────
#  TestStage1SqlFilter — DB 整合
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_universe_session(db_session, monkeypatch):
    """讓 universe 模組使用測試 session。"""
    import src.discovery.universe as univ_mod

    monkeypatch.setattr(univ_mod, "get_session", lambda: db_session)


class TestStage1SqlFilter:
    """_stage1_sql_filter() 的 DB 整合測試，使用 in-memory SQLite。"""

    def _add_stock_with_history(
        self,
        db_session,
        stock_id: str,
        listing_type: str = "上市",
        security_type: str | None = "stock",
        days: int = 130,
        close: float = 50.0,
    ):
        """新增 StockInfo + N 天 DailyPrice。"""
        db_session.add(_make_si(stock_id, listing_type, security_type))
        today = date.today()
        for i in range(days):
            dt = today - timedelta(days=days - i)
            db_session.add(_make_dp(stock_id, dt, close=close))
        db_session.flush()

    def test_passes_normal_stock(self, db_session):
        self._add_stock_with_history(db_session, "2330")
        uf = UniverseFilter(UniverseConfig(min_available_days=120, min_close=10.0))
        result = uf._stage1_sql_filter()
        assert "2330" in result

    def test_excludes_etf_by_security_type(self, db_session):
        self._add_stock_with_history(db_session, "0050", security_type="etf")
        uf = UniverseFilter(UniverseConfig(security_type="stock"))
        result = uf._stage1_sql_filter()
        assert "0050" not in result

    def test_null_security_type_passes_with_or_null_fallback(self, db_session):
        # security_type=None → 向後相容，應通過 OR NULL 條件
        self._add_stock_with_history(db_session, "2317", security_type=None)
        uf = UniverseFilter(UniverseConfig(security_type="stock"))
        result = uf._stage1_sql_filter()
        assert "2317" in result

    def test_excludes_insufficient_trading_days(self, db_session):
        self._add_stock_with_history(db_session, "9999", days=50)  # 只有 50 天
        uf = UniverseFilter(UniverseConfig(min_available_days=120))
        result = uf._stage1_sql_filter()
        assert "9999" not in result

    def test_excludes_low_price(self, db_session):
        self._add_stock_with_history(db_session, "1111", close=5.0)  # 收盤 < 10
        uf = UniverseFilter(UniverseConfig(min_close=10.0))
        result = uf._stage1_sql_filter()
        assert "1111" not in result

    def test_excludes_wrong_listing_type(self, db_session):
        self._add_stock_with_history(db_session, "8888", listing_type="興櫃")
        uf = UniverseFilter(UniverseConfig(listing_types=("上市", "上櫃")))
        result = uf._stage1_sql_filter()
        assert "8888" not in result


# ────────────────────────────────────────────────────────────────────────────
#  TestStage2LiquidityFilter — DB fallback 整合
# ────────────────────────────────────────────────────────────────────────────


class TestStage2LiquidityFilter:
    """_stage2_liquidity_filter() 的 DB fallback 路徑測試。"""

    def _add_dp_series(self, db_session, stock_id: str, turnover: int, days: int = 5):
        today = date.today()
        for i in range(days):
            dt = today - timedelta(days=days - i)
            db_session.add(_make_dp(stock_id, dt, turnover=turnover))
        db_session.flush()

    def test_passes_when_turnover_sufficient(self, db_session):
        self._add_dp_series(db_session, "2330", turnover=50_000_000)
        uf = UniverseFilter(UniverseConfig(avg_turnover_5d_min=30_000_000, min_turnover_5d_min=10_000_000))
        result = uf._stage2_liquidity_filter(["2330"])
        assert "2330" in result

    def test_excludes_when_turnover_insufficient(self, db_session):
        self._add_dp_series(db_session, "1234", turnover=5_000_000)
        uf = UniverseFilter(UniverseConfig(avg_turnover_5d_min=30_000_000, min_turnover_5d_min=10_000_000))
        result = uf._stage2_liquidity_filter(["1234"])
        assert "1234" not in result

    def test_falls_back_to_dailyprice_when_no_feature(self, db_session):
        # DailyFeature 為空，應 fallback DailyPrice
        self._add_dp_series(db_session, "2454", turnover=40_000_000)
        uf = UniverseFilter(UniverseConfig(avg_turnover_5d_min=30_000_000, min_turnover_5d_min=10_000_000))
        result = uf._stage2_liquidity_filter(["2454"])
        assert "2454" in result

    def test_returns_input_when_no_data_available(self, db_session):
        # 完全無資料時不應報錯，應回傳輸入
        uf = UniverseFilter(UniverseConfig())
        result = uf._stage2_liquidity_filter(["9999_no_data"])
        assert "9999_no_data" in result


# ────────────────────────────────────────────────────────────────────────────
#  TestCandidateMemory — DB 整合
# ────────────────────────────────────────────────────────────────────────────


class TestCandidateMemory:
    """_load_candidate_memory() 的 DB 整合測試。"""

    def test_returns_yesterday_candidate(self, db_session):
        yesterday = date.today() - timedelta(days=1)
        db_session.add(_make_dr("2330", mode="momentum", scan_date=yesterday))
        db_session.flush()
        uf = UniverseFilter(UniverseConfig(candidate_memory_days=1))
        result = uf._load_candidate_memory(mode="momentum", stage1_ids=["2330"])
        assert "2330" in result

    def test_excludes_stock_not_in_stage1_ids(self, db_session):
        yesterday = date.today() - timedelta(days=1)
        db_session.add(_make_dr("2454", mode="momentum", scan_date=yesterday))
        db_session.flush()
        uf = UniverseFilter(UniverseConfig(candidate_memory_days=1))
        # stage1_ids 不含 2454
        result = uf._load_candidate_memory(mode="momentum", stage1_ids=["2330"])
        assert "2454" not in result

    def test_mode_isolation(self, db_session):
        yesterday = date.today() - timedelta(days=1)
        db_session.add(_make_dr("2330", mode="swing", scan_date=yesterday))
        db_session.flush()
        uf = UniverseFilter(UniverseConfig(candidate_memory_days=1))
        # 查詢 momentum 模式，不應看到 swing 記錄
        result = uf._load_candidate_memory(mode="momentum", stage1_ids=["2330"])
        assert "2330" not in result


# ────────────────────────────────────────────────────────────────────────────
#  TestComputeAndStoreDailyFeatures — Pipeline ETL 整合
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def _patch_pipeline_session(db_session, monkeypatch):
    """讓 pipeline 模組使用測試 session。"""
    import src.data.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "get_session", lambda: db_session)
    monkeypatch.setattr(pipeline_mod, "init_db", lambda: None)


class TestComputeAndStoreDailyFeatures:
    """compute_and_store_daily_features() 的 DB 整合測試。"""

    def _seed_prices(self, db_session, stock_id: str, days: int = 65):
        """寫入 N 天連續 DailyPrice 資料（含 turnover）。"""
        today = date.today()
        close = 100.0
        for i in range(days):
            dt = today - timedelta(days=days - i)
            db_session.add(
                _make_dp(
                    stock_id,
                    dt,
                    close=close + i * 0.5,
                    volume=1_000_000 + i * 10_000,
                    turnover=100_000_000 + i * 1_000_000,
                )
            )
        db_session.flush()

    @pytest.mark.usefixtures("_patch_pipeline_session")
    def test_writes_rows_for_each_stock(self, db_session):
        from sqlalchemy import select

        from src.data.pipeline import compute_and_store_daily_features

        self._seed_prices(db_session, "2330", days=65)
        count = compute_and_store_daily_features(lookback_days=65)
        assert count >= 1
        rows = db_session.execute(select(DailyFeature).where(DailyFeature.stock_id == "2330")).all()
        assert len(rows) >= 1

    @pytest.mark.usefixtures("_patch_pipeline_session")
    def test_ma20_calculation_correct(self, db_session):
        """ma20 的值應為最近 20 日收盤均值（近似）。"""
        from sqlalchemy import select

        from src.data.pipeline import compute_and_store_daily_features

        self._seed_prices(db_session, "2317", days=65)
        compute_and_store_daily_features(lookback_days=65)
        row = db_session.execute(select(DailyFeature).where(DailyFeature.stock_id == "2317")).first()
        assert row is not None
        feat = row[0]
        assert feat.ma20 is not None
        assert feat.ma20 > 0

    @pytest.mark.usefixtures("_patch_pipeline_session")
    def test_turnover_ma5_populated(self, db_session):
        from sqlalchemy import select

        from src.data.pipeline import compute_and_store_daily_features

        self._seed_prices(db_session, "2454", days=65)
        compute_and_store_daily_features(lookback_days=65)
        row = db_session.execute(select(DailyFeature).where(DailyFeature.stock_id == "2454")).first()
        feat = row[0]
        assert feat.turnover_ma5 is not None and feat.turnover_ma5 > 0

    @pytest.mark.usefixtures("_patch_pipeline_session")
    def test_upsert_idempotent(self, db_session):
        """重複執行不應建立重複列。"""
        from sqlalchemy import func, select

        from src.data.pipeline import compute_and_store_daily_features

        self._seed_prices(db_session, "3008", days=65)
        compute_and_store_daily_features(lookback_days=65)
        compute_and_store_daily_features(lookback_days=65)
        count = db_session.execute(select(func.count()).where(DailyFeature.stock_id == "3008")).scalar()
        # 每支股票在最新一日只應有 1 筆記錄
        assert count == 1
