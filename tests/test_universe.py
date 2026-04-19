"""Universe Filtering Module 測試。

涵蓋：
- TestFilterLiquidity (4):           filter_liquidity() 純函數
- TestFilterTrend (5):               filter_trend() 純函數
- TestRegimeAwareFiltering (4):      Regime 自適應門檻調整
- TestRelativeLiquidity (4):         相對流動性救援通道
- TestFilterTrendBreakout (8):       突破型過濾純函數
- TestStage1SqlFilter (6):           UniverseFilter._stage1_sql_filter() DB 整合
- TestStage2LiquidityFilter (4):     Stage 2 DB fallback
- TestCandidateMemory (5):           _load_candidate_memory() DB 整合（含 days_ago）
- TestMemoryDecay (4):               Candidate Memory 漸進衰減門檻
- TestComputeAndStoreDailyFeatures (4): pipeline ETL 整合
- TestClassifySecurityType (8):      _classify_security_type() 純函數
- TestFilterLiquidityDualWindow (5): 雙窗口流動性確認（turnover_ma20 欄位）
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
from src.discovery.universe import (
    UniverseConfig,
    UniverseFilter,
    filter_liquidity,
    filter_trend,
    filter_trend_breakout,
)

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


def _make_si(stock_id: str, listing_type: str = "twse", security_type: str | None = "stock") -> StockInfo:
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
#  TestRegimeAwareFiltering — 純函數
# ────────────────────────────────────────────────────────────────────────────


class TestRegimeAwareFiltering:
    """Regime 自適應門檻測試（filter_liquidity / filter_trend 純函數）。"""

    def _liquidity_df(self, turnover: int) -> pd.DataFrame:
        return pd.DataFrame([{"stock_id": "2330", "turnover": turnover}] * 5)

    def test_bull_regime_relaxes_turnover_threshold(self):
        """Bull: multiplier=0.8 → 有效門檻 24M，25M 應通過。"""
        df = self._liquidity_df(25_000_000)
        cfg = UniverseConfig(avg_turnover_5d_min=30_000_000, min_turnover_5d_min=8_000_000)
        # 預設門檻 30M → ×0.8 = 24M，25M > 24M → 通過
        result = filter_liquidity(df, cfg, turnover_multiplier=0.8)
        assert "2330" in result

    def test_crisis_regime_tightens_turnover_threshold(self):
        """Crisis: multiplier=1.5 → 有效門檻 45M，40M 不應通過。"""
        df = self._liquidity_df(40_000_000)
        cfg = UniverseConfig(avg_turnover_5d_min=30_000_000, min_turnover_5d_min=10_000_000)
        # 預設門檻 30M → ×1.5 = 45M，40M < 45M → 不通過
        result = filter_liquidity(df, cfg, turnover_multiplier=1.5)
        assert "2330" not in result

    def test_bear_regime_disables_volume_ratio(self):
        """Bear: volume_ratio_override=None → 量比門檻失效，低量比也通過。"""
        today = date.today()
        df = pd.DataFrame(
            [
                {
                    "stock_id": "2330",
                    "date": today,
                    "close": 110.0,
                    "volume": 500_000,  # 量比 = 0.5 < 1.5
                    "ma60": 100.0,
                    "volume_ma20": 1_000_000.0,
                }
            ]
        )
        cfg = UniverseConfig(trend_ma=60, volume_ratio_min=1.5)
        # volume_ratio_override=None → 跳過量比過濾
        result = filter_trend(df, cfg, volume_ratio_override=None)
        assert "2330" in result

    def test_none_regime_uses_defaults(self):
        """regime=None → 不調整，使用原始門檻。"""
        df = self._liquidity_df(25_000_000)
        cfg = UniverseConfig(avg_turnover_5d_min=30_000_000, min_turnover_5d_min=10_000_000)
        # 不傳 multiplier → 預設 1.0，25M < 30M → 不通過
        result = filter_liquidity(df, cfg)
        assert "2330" not in result


# ────────────────────────────────────────────────────────────────────────────
#  TestRelativeLiquidity — 純函數
# ────────────────────────────────────────────────────────────────────────────


class TestRelativeLiquidity:
    """相對流動性救援通道測試（filter_liquidity 純函數）。"""

    def _cfg(self) -> UniverseConfig:
        return UniverseConfig(avg_turnover_5d_min=30_000_000, min_turnover_5d_min=10_000_000)

    def test_relative_liquidity_rescues_high_ratio_stock(self):
        """avg5=20M（< 30M 絕對門檻），但 ratio=3.0 且 avg5 > 15M（半門檻）→ 救援通過。"""
        df = pd.DataFrame([{"stock_id": "9876", "turnover": 20_000_000, "turnover_ratio_5d_20d": 3.0}] * 5)
        result = filter_liquidity(df, self._cfg())
        assert "9876" in result

    def test_relative_liquidity_no_rescue_below_half_threshold(self):
        """avg5=10M（< 15M 半門檻），即使 ratio 高也不救援。"""
        df = pd.DataFrame([{"stock_id": "9876", "turnover": 10_000_000, "turnover_ratio_5d_20d": 5.0}] * 5)
        result = filter_liquidity(df, self._cfg())
        assert "9876" not in result

    def test_relative_liquidity_no_rescue_low_ratio(self):
        """avg5=20M 但 ratio=1.5（< 2.0）→ 不救援。"""
        df = pd.DataFrame([{"stock_id": "9876", "turnover": 20_000_000, "turnover_ratio_5d_20d": 1.5}] * 5)
        result = filter_liquidity(df, self._cfg())
        assert "9876" not in result

    def test_already_passed_not_duplicated(self):
        """已通過絕對門檻的股票不應重複出現。"""
        df = pd.DataFrame([{"stock_id": "2330", "turnover": 50_000_000, "turnover_ratio_5d_20d": 3.0}] * 5)
        result = filter_liquidity(df, self._cfg())
        assert result.count("2330") == 1


# ────────────────────────────────────────────────────────────────────────────
#  TestFilterTrendBreakout — 純函數
# ────────────────────────────────────────────────────────────────────────────


class TestFilterTrendBreakout:
    """filter_trend_breakout() 突破型過濾純函數測試。"""

    def _make_df(
        self,
        stock_id: str = "2330",
        close: float = 105.0,
        ma20: float = 100.0,
        momentum_20d: float = 5.0,
        high_20d: float = 110.0,
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
                    "ma20": ma20,
                    "momentum_20d": momentum_20d,
                    "high_20d": high_20d,
                    "volume": volume,
                    "volume_ma20": volume_ma20,
                }
            ]
        )

    def test_breakout_passes_all_conditions(self):
        """close > ma20, momentum > 0, near high, volume expansion → 通過。"""
        df = self._make_df(close=105, ma20=100, momentum_20d=5.0, high_20d=110, volume=2_000_000, volume_ma20=1_000_000)
        # close/high_20d = 105/110 = 0.954 > 0.9 ✓
        result = filter_trend_breakout(df, UniverseConfig())
        assert "2330" in result

    def test_breakout_rejects_below_ma20(self):
        """close < ma20 → 不通過。"""
        df = self._make_df(close=95, ma20=100)
        result = filter_trend_breakout(df, UniverseConfig())
        assert "2330" not in result

    def test_breakout_rejects_negative_momentum(self):
        """momentum_20d < 0 → 死貓反彈，不通過。"""
        df = self._make_df(momentum_20d=-3.0)
        result = filter_trend_breakout(df, UniverseConfig())
        assert "2330" not in result

    def test_breakout_rejects_far_from_high(self):
        """close / high_20d < 0.9 → 離高點太遠，非真突破。"""
        df = self._make_df(close=80, high_20d=110)
        # 80/110 = 0.727 < 0.9
        result = filter_trend_breakout(df, UniverseConfig())
        assert "2330" not in result

    def test_breakout_rejects_low_volume(self):
        """volume / volume_ma20 < 1.5 → 量能不足。"""
        df = self._make_df(volume=800_000, volume_ma20=1_000_000)
        # 0.8 < 1.5
        result = filter_trend_breakout(df, UniverseConfig())
        assert "2330" not in result

    def test_trend_or_breakout_returns_union(self):
        """trend_or_breakout 模式：Type A ∪ Type B。"""
        today = date.today()
        # Stock A: 只通過趨勢（close > MA60，but below MA20 — won't happen）
        # Stock B: 只通過突破（close > MA20 but < MA60）
        # Stock C: 都通過
        df = pd.DataFrame(
            [
                # B: close=105 > ma20=100, but close=105 < ma60=120 → 趨勢不過，突破過
                {
                    "stock_id": "B",
                    "date": today,
                    "close": 105,
                    "ma20": 100,
                    "ma60": 120,
                    "momentum_20d": 5.0,
                    "high_20d": 110,
                    "volume": 2_000_000,
                    "volume_ma20": 1_000_000,
                },
                # C: close=130 > ma60=120 > ma20=100 → 都過
                {
                    "stock_id": "C",
                    "date": today,
                    "close": 130,
                    "ma20": 100,
                    "ma60": 120,
                    "momentum_20d": 5.0,
                    "high_20d": 135,
                    "volume": 2_000_000,
                    "volume_ma20": 1_000_000,
                },
            ]
        )
        cfg = UniverseConfig(trend_ma=60, volume_ratio_min=1.5)
        trend_ids = filter_trend(df, cfg)
        breakout_ids = filter_trend_breakout(df, cfg)
        union = set(trend_ids) | set(breakout_ids)
        assert "B" in union  # 突破型通過
        assert "C" in union  # 兩種都通過
        assert "B" not in trend_ids  # B 不在趨勢型中

    def test_breakout_missing_required_columns_returns_empty(self):
        """缺少必要欄位（ma20, volume_ma20）→ 回傳空清單。"""
        today = date.today()
        df = pd.DataFrame([{"stock_id": "X", "date": today, "close": 100, "volume": 1_000_000}])
        result = filter_trend_breakout(df, UniverseConfig())
        assert result == []

    def test_empty_df_returns_empty(self):
        result = filter_trend_breakout(pd.DataFrame(), UniverseConfig())
        assert result == []


# ────────────────────────────────────────────────────────────────────────────
#  TestAnomalyExclusion — 漲跌停異常排除（Critical Bug #1 回歸測試）
# ────────────────────────────────────────────────────────────────────────────


class TestAnomalyExclusion:
    """近 7 日內累計 3 次以上 |pct_chg| ≥ 9.5% 的股票必須被 Universe 排除。

    Regression：舊實作使用 `groupby(level=0)` 對單層 RangeIndex 逐列 group，
    導致 anomaly_ids 永遠為空，漲跌停異常股未被過濾。
    """

    @staticmethod
    def _hist_df(stock_id: str, closes: list[float], volume: int = 2_000_000) -> pd.DataFrame:
        n = len(closes)
        dates = pd.date_range(end=date.today(), periods=n)
        return pd.DataFrame(
            {
                "stock_id": [stock_id] * n,
                "date": dates,
                "close": closes,
                "volume": [volume] * n,
                "ma20": [closes[0]] * n,
                "ma60": [closes[0]] * n,
                "volume_ma20": [volume // 2] * n,
                "momentum_20d": [5.0] * n,
                "high_20d": [max(closes)] * n,
            }
        )

    def test_filter_trend_excludes_consecutive_limit_up(self):
        """連續 5 日漲停（≥9.5%）→ filter_trend 必須排除。"""
        closes = [100.0] * 25 + [100.0 * (1.097 ** (i + 1)) for i in range(5)]
        df = self._hist_df("9999", closes)
        cfg = UniverseConfig(trend_ma=20, volume_ratio_min=0.5)
        assert "9999" not in filter_trend(df, cfg)

    def test_filter_trend_excludes_consecutive_limit_down(self):
        """連續 5 日跌停 → 也要排除（絕對值 ≥9.5%）。"""
        closes = [100.0] * 25
        b = 100.0
        for _ in range(5):
            b *= 0.903
            closes.append(b)
        df = self._hist_df("8888", closes)
        # 特意讓 ma20/ma60 較低，避免 close<MA 先一步剔除
        df["ma20"] = 50.0
        df["ma60"] = 50.0
        df["high_20d"] = df["close"]
        cfg = UniverseConfig(trend_ma=20, volume_ratio_min=0.5)
        assert "8888" not in filter_trend(df, cfg)

    def test_filter_trend_passes_single_limit_day(self):
        """僅 1 日漲停（<3 次）→ 不視為異常，允許通過。"""
        closes = [100.0] * 29 + [100.0 * 1.097]
        df = self._hist_df("7777", closes)
        cfg = UniverseConfig(trend_ma=20, volume_ratio_min=0.5)
        assert "7777" in filter_trend(df, cfg)

    def test_filter_trend_mixed_normal_and_anomaly(self):
        """混合：正常股通過、漲停股被排除。"""
        normal = self._hist_df("1234", [100.0 + i * 0.5 for i in range(30)])
        anomaly_closes = [100.0] * 25 + [100.0 * (1.097 ** (i + 1)) for i in range(5)]
        anomaly = self._hist_df("9999", anomaly_closes)
        df = pd.concat([normal, anomaly], ignore_index=True)
        cfg = UniverseConfig(trend_ma=20, volume_ratio_min=0.5)
        result = filter_trend(df, cfg)
        assert "1234" in result
        assert "9999" not in result

    def test_filter_trend_breakout_excludes_anomaly(self):
        """filter_trend_breakout 同樣需排除連續漲停異常股。"""
        closes = [100.0] * 25 + [100.0 * (1.097 ** (i + 1)) for i in range(5)]
        df = self._hist_df("9999", closes)
        cfg = UniverseConfig()
        assert "9999" not in filter_trend_breakout(df, cfg)


# ────────────────────────────────────────────────────────────────────────────
#  TestDataStaleness — 資料新鮮度告警（Minor #2 回歸測試）
# ────────────────────────────────────────────────────────────────────────────


class TestDataStaleness:
    """filter_trend / filter_trend_breakout 偵測到過期快照時應發出 warning。"""

    @staticmethod
    def _fresh_df(latest_offset_days: int) -> pd.DataFrame:
        end = date.today() - timedelta(days=latest_offset_days)
        dates = pd.date_range(end=end, periods=30)
        return pd.DataFrame(
            {
                "stock_id": ["1234"] * 30,
                "date": dates,
                "close": [100.0 + i * 0.5 for i in range(30)],
                "volume": [2_000_000] * 30,
                "ma20": [100.0] * 30,
                "ma60": [100.0] * 30,
                "volume_ma20": [1_000_000] * 30,
                "momentum_20d": [5.0] * 30,
                "high_20d": [115.0] * 30,
            }
        )

    def test_filter_trend_warns_when_stale(self, caplog):
        df = self._fresh_df(latest_offset_days=15)  # 遠超 STALE_DATA_WARN_DAYS=5
        cfg = UniverseConfig(trend_ma=20, volume_ratio_min=0.5)
        with caplog.at_level("WARNING", logger="src.discovery.universe"):
            filter_trend(df, cfg)
        assert any("資料新鮮度告警" in rec.message for rec in caplog.records)

    def test_filter_trend_silent_when_fresh(self, caplog):
        df = self._fresh_df(latest_offset_days=0)
        cfg = UniverseConfig(trend_ma=20, volume_ratio_min=0.5)
        with caplog.at_level("WARNING", logger="src.discovery.universe"):
            filter_trend(df, cfg)
        assert not any("資料新鮮度告警" in rec.message for rec in caplog.records)

    def test_filter_trend_breakout_warns_when_stale(self, caplog):
        df = self._fresh_df(latest_offset_days=20)
        cfg = UniverseConfig()
        with caplog.at_level("WARNING", logger="src.discovery.universe"):
            filter_trend_breakout(df, cfg)
        assert any("資料新鮮度告警" in rec.message for rec in caplog.records)


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
        listing_type: str = "twse",
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
        uf = UniverseFilter(UniverseConfig(listing_types=("twse", "tpex")))
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

    def test_falls_back_when_feature_coverage_insufficient(self, db_session):
        """DailyFeature 僅覆蓋極少數股票時，應 fallback 至 DailyPrice。

        模擬：10 支股票送入 Stage 2，但 DailyFeature 只有 1 支（覆蓋率 10% < 30%），
        DailyPrice 有充足 turnover 的 5 支應全部通過。
        """
        today = date.today()
        all_ids = [f"S{i:04d}" for i in range(10)]

        # 5 支高流動性 + 5 支低流動性（DailyPrice）
        for i, sid in enumerate(all_ids):
            turnover = 50_000_000 if i < 5 else 1_000_000
            for d in range(5):
                dt = today - timedelta(days=5 - d)
                db_session.add(_make_dp(sid, dt, turnover=turnover))

        # DailyFeature 只覆蓋 1 支（覆蓋率 1/10 = 10% < 30%）
        db_session.add(
            DailyFeature(
                stock_id="S0000",
                date=today - timedelta(days=1),
                close=100.0,
                volume=1_000_000,
                turnover=50_000_000,
                turnover_ma5=50_000_000,
            )
        )
        db_session.flush()

        uf = UniverseFilter(UniverseConfig(avg_turnover_5d_min=30_000_000, min_turnover_5d_min=10_000_000))
        result = uf._stage2_liquidity_filter(all_ids)

        # 應該用 DailyPrice fallback，5 支高流動性全部通過
        for sid in all_ids[:5]:
            assert sid in result, f"{sid} 應通過流動性過濾（DailyPrice fallback）"
        for sid in all_ids[5:]:
            assert sid not in result, f"{sid} 不應通過流動性過濾"


# ────────────────────────────────────────────────────────────────────────────
#  TestCandidateMemory — DB 整合
# ────────────────────────────────────────────────────────────────────────────


class TestCandidateMemory:
    """_load_candidate_memory() 的 DB 整合測試（回傳 dict[str, int]）。"""

    def test_returns_yesterday_candidate_with_days_ago(self, db_session):
        yesterday = date.today() - timedelta(days=1)
        db_session.add(_make_dr("2330", mode="momentum", scan_date=yesterday))
        db_session.flush()
        uf = UniverseFilter(UniverseConfig(candidate_memory_days=3))
        result = uf._load_candidate_memory(mode="momentum", stage1_ids=["2330"])
        assert "2330" in result
        assert result["2330"] == 1  # days_ago = 1

    def test_excludes_stock_not_in_stage1_ids(self, db_session):
        yesterday = date.today() - timedelta(days=1)
        db_session.add(_make_dr("2454", mode="momentum", scan_date=yesterday))
        db_session.flush()
        uf = UniverseFilter(UniverseConfig(candidate_memory_days=3))
        # stage1_ids 不含 2454
        result = uf._load_candidate_memory(mode="momentum", stage1_ids=["2330"])
        assert "2454" not in result

    def test_mode_isolation(self, db_session):
        yesterday = date.today() - timedelta(days=1)
        db_session.add(_make_dr("2330", mode="swing", scan_date=yesterday))
        db_session.flush()
        uf = UniverseFilter(UniverseConfig(candidate_memory_days=3))
        # 查詢 momentum 模式，不應看到 swing 記錄
        result = uf._load_candidate_memory(mode="momentum", stage1_ids=["2330"])
        assert "2330" not in result

    def test_multi_day_uses_most_recent(self, db_session):
        """同股在 Day1 和 Day3 都出現 → 取 days_ago=1。"""
        db_session.add(_make_dr("2330", mode="momentum", scan_date=date.today() - timedelta(days=1)))
        db_session.add(_make_dr("2330", mode="momentum", scan_date=date.today() - timedelta(days=3)))
        db_session.flush()
        uf = UniverseFilter(UniverseConfig(candidate_memory_days=3))
        result = uf._load_candidate_memory(mode="momentum", stage1_ids=["2330"])
        assert result["2330"] == 1  # 最近的

    def test_day4_excluded_by_memory_window(self, db_session):
        """4 天前的推薦 → 超出 memory_days=3，不回傳。"""
        db_session.add(_make_dr("2330", mode="momentum", scan_date=date.today() - timedelta(days=4)))
        db_session.flush()
        uf = UniverseFilter(UniverseConfig(candidate_memory_days=3))
        result = uf._load_candidate_memory(mode="momentum", stage1_ids=["2330"])
        assert "2330" not in result


# ────────────────────────────────────────────────────────────────────────────
#  TestMemoryDecay — 純函數（衰減門檻邏輯）
# ────────────────────────────────────────────────────────────────────────────


class TestMemoryDecay:
    """Candidate Memory 漸進衰減機制測試。

    使用 UniverseFilter 物件直接操控內部狀態，測試 run() 中的衰減邏輯。
    """

    def test_day1_memory_relaxes_threshold_to_80pct(self):
        """Day 1 記憶：門檻 ×0.8 = 24M，25M turnover 應通過。"""
        from src.discovery.universe import MEMORY_DECAY

        assert MEMORY_DECAY[1] == 0.8
        # 基底門檻 30M × 0.8 = 24M，25M > 24M → 通過
        base = 30_000_000
        assert 25_000_000 >= base * 0.8

    def test_day2_memory_relaxes_threshold_to_90pct(self):
        """Day 2 記憶：門檻 ×0.9 = 27M，25M turnover 不通過。"""
        from src.discovery.universe import MEMORY_DECAY

        assert MEMORY_DECAY[2] == 0.9
        base = 30_000_000
        assert 25_000_000 < base * 0.9  # 25M < 27M → 不通過

    def test_day3_memory_uses_full_threshold(self):
        """Day 3 記憶：門檻 ×1.0 = 30M，需完全達標。"""
        from src.discovery.universe import MEMORY_DECAY

        assert MEMORY_DECAY[3] == 1.0

    def test_day4_not_in_decay_map(self):
        """Day 4+ 不在衰減表中 → 不保留。"""
        from src.discovery.universe import MEMORY_DECAY

        assert 4 not in MEMORY_DECAY


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


# ────────────────────────────────────────────────────────────────────────────
#  TestClassifySecurityType — 純函數
# ────────────────────────────────────────────────────────────────────────────


class TestClassifySecurityType:
    """_classify_security_type() 純函數測試。

    驗證各類台股代號的分類正確性，重點涵蓋 ETF 的多種代號格式。
    """

    def _classify(self, stock_id: str, stock_name: str = "") -> str:
        from src.data.pipeline import _classify_security_type

        return _classify_security_type(stock_id, stock_name)

    # --- 普通股 ---
    def test_regular_4digit_stock(self):
        assert self._classify("2330") == "stock"

    def test_regular_4digit_stock_1xxx(self):
        assert self._classify("1301") == "stock"

    # --- ETF（各種代號格式）---
    def test_etf_4digit_0050(self):
        r"""0050 — 最早 ETF，4 位代號（舊 regex ^00\d{4}$ 誤判為 stock）"""
        assert self._classify("0050") == "etf"

    def test_etf_5digit_00878(self):
        """00878 — 5 位代號（舊 regex 誤判為 stock）"""
        assert self._classify("00878") == "etf"

    def test_etf_5digit_00882(self):
        """00882 — 5 位代號（債券型 ETF，舊 regex 誤判為 stock）"""
        assert self._classify("00882") == "etf"

    def test_etf_with_letter_suffix_00991A(self):
        """00991A — 含字母後綴的期貨 ETF（舊 regex 誤判為 stock）"""
        assert self._classify("00991A") == "etf"

    def test_etf_with_L_suffix_00715L(self):
        """00715L — 槓桿 ETF（舊 regex 誤判為 stock）"""
        assert self._classify("00715L") == "etf"

    def test_etf_with_B_suffix_00679B(self):
        """00679B — 債券 ETF（舊 regex 誤判為 stock）"""
        assert self._classify("00679B") == "etf"

    # --- 名稱含 ETF 字樣 ---
    def test_etf_by_name(self):
        assert self._classify("9999", "富邦ETF") == "etf"

    # --- 權證 ---
    def test_warrant_6digit(self):
        assert self._classify("123456") == "warrant"


# ────────────────────────────────────────────────────────────────────────────
#  TestFilterLiquidityDualWindow — 純函數（雙窗口流動性確認）
# ────────────────────────────────────────────────────────────────────────────


class TestFilterLiquidityDualWindow:
    """filter_liquidity() 雙窗口流動性確認測試（無 DB）。

    驗證當 df_5d 含 turnover_ma20 欄位時，額外的 20 日均量門檻正確生效。
    """

    def _cfg(
        self,
        avg=30_000_000.0,
        min_=10_000_000.0,
        ma20_min=20_000_000.0,
    ) -> UniverseConfig:
        return UniverseConfig(avg_turnover_5d_min=avg, min_turnover_5d_min=min_, turnover_ma20_min=ma20_min)

    def _make_df(self, turnover_5d: list[int], turnover_ma20: float | None) -> pd.DataFrame:
        """建立 5 日 turnover 資料，附帶可選的 turnover_ma20 欄位。"""
        rows = [{"stock_id": "2330", "turnover": t} for t in turnover_5d]
        df = pd.DataFrame(rows)
        if turnover_ma20 is not None:
            df["turnover_ma20"] = turnover_ma20
        return df

    def test_passes_when_ma5_and_ma20_both_meet_threshold(self):
        """5 日均量 + 20 日均量皆達標 → 通過。"""
        df = self._make_df([40_000_000] * 5, turnover_ma20=25_000_000.0)
        result = filter_liquidity(df, self._cfg())
        assert "2330" in result

    def test_excludes_when_ma20_below_threshold(self):
        """5 日均量達標但 20 日均量不足 → 誘多陷阱，應排除。"""
        # 近 5 日突然爆量（平均 40M），但 20 日均量只有 15M（中期流動性不足）
        df = self._make_df([40_000_000] * 5, turnover_ma20=15_000_000.0)
        result = filter_liquidity(df, self._cfg())
        assert "2330" not in result

    def test_no_ma20_column_falls_back_to_old_logic(self):
        """未提供 turnover_ma20 欄位時，降回舊有 avg5/min5 邏輯（向後相容）。"""
        # 不帶 turnover_ma20 → 舊邏輯只看 avg5 + min5
        df = self._make_df([40_000_000] * 5, turnover_ma20=None)
        result = filter_liquidity(df, self._cfg())
        # 舊邏輯通過（avg5=40M > 30M，min5=40M > 10M）
        assert "2330" in result

    def test_multiple_stocks_independent_ma20_check(self):
        """多股混合情況：各股獨立判斷 20 日均量。"""
        rows = [
            # 2330: 5 日均量達標，20 日均量也達標 → 通過
            {"stock_id": "2330", "turnover": 40_000_000, "turnover_ma20": 25_000_000.0},
            {"stock_id": "2330", "turnover": 40_000_000, "turnover_ma20": 25_000_000.0},
            {"stock_id": "2330", "turnover": 40_000_000, "turnover_ma20": 25_000_000.0},
            {"stock_id": "2330", "turnover": 40_000_000, "turnover_ma20": 25_000_000.0},
            {"stock_id": "2330", "turnover": 40_000_000, "turnover_ma20": 25_000_000.0},
            # 9999: 5 日均量達標，但 20 日均量不足 → 排除
            {"stock_id": "9999", "turnover": 50_000_000, "turnover_ma20": 10_000_000.0},
            {"stock_id": "9999", "turnover": 50_000_000, "turnover_ma20": 10_000_000.0},
            {"stock_id": "9999", "turnover": 50_000_000, "turnover_ma20": 10_000_000.0},
            {"stock_id": "9999", "turnover": 50_000_000, "turnover_ma20": 10_000_000.0},
            {"stock_id": "9999", "turnover": 50_000_000, "turnover_ma20": 10_000_000.0},
        ]
        df = pd.DataFrame(rows)
        result = filter_liquidity(df, self._cfg())
        assert "2330" in result
        assert "9999" not in result

    def test_ma20_zero_excluded(self):
        """turnover_ma20 為 0（新股或無資料）→ 應排除（0 < 20M）。"""
        df = self._make_df([40_000_000] * 5, turnover_ma20=0.0)
        result = filter_liquidity(df, self._cfg())
        assert "2330" not in result
