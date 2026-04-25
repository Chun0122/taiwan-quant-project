"""測試共用 market data 載入層（項目 B）。

驗證重點：
  - `load_shared_market_data` 產出的 DataFrame 欄位、cutoff 與 DB 查詢一致。
  - scanner 透過 `run(shared=...)` 走共用路徑時，`_load_market_data` 結果
    與不傳 shared（DB 路徑）逐列一致 —— 這是 B 項最重要的語意保證。
  - `slice_revenue_raw` months=1 / 2 / 4 的 pivot 行為正確。
  - `_cmd_discover_all` 只呼叫 `load_shared_market_data` 一次（非 5 次）。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data.schema import (
    DailyPrice,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
)
from src.discovery.scanner._shared_load import (
    SharedMarketData,
    load_shared_market_data,
    slice_revenue_raw,
)

# ------------------------------------------------------------------ #
#  Helpers — 以 ORM 塞入 in-memory DB
# ------------------------------------------------------------------ #


def _insert_daily_prices(session, stock_ids: list[str], start: date, n_days: int) -> None:
    for sid in stock_ids:
        for d in range(n_days):
            day = start + timedelta(days=d)
            close = 100.0 + d * 0.5 + (int(sid) % 10) * 0.1
            session.add(
                DailyPrice(
                    stock_id=sid,
                    date=day,
                    open=close - 0.5,
                    high=close + 1.0,
                    low=close - 1.0,
                    close=close,
                    volume=1_000_000 + d * 10_000,
                    turnover=int((1_000_000 + d * 10_000) * close),
                )
            )
    session.commit()


def _insert_institutional(session, stock_ids: list[str], start: date, n_days: int) -> None:
    for sid in stock_ids:
        for d in range(n_days):
            day = start + timedelta(days=d)
            session.add(
                InstitutionalInvestor(
                    stock_id=sid,
                    date=day,
                    name="Foreign_Investor",
                    buy=500,
                    sell=300,
                    net=200 + d * 10,
                )
            )
    session.commit()


def _insert_margin(session, stock_ids: list[str], start: date, n_days: int) -> None:
    for sid in stock_ids:
        for d in range(n_days):
            day = start + timedelta(days=d)
            session.add(
                MarginTrading(
                    stock_id=sid,
                    date=day,
                    margin_buy=100,
                    margin_sell=50,
                    margin_balance=10_000 + d * 100,
                    short_sell=10,
                    short_buy=5,
                    short_balance=1_000 + d,
                )
            )
    session.commit()


def _insert_monthly_revenue(session, stock_ids: list[str], n_months: int = 4) -> None:
    """為每支股票塞 n_months 筆月營收（date 為各月月底）。"""
    today = date.today()
    for sid in stock_ids:
        for m in range(n_months):
            # 往前 m 個月的月底日期
            base = today.replace(day=1) - timedelta(days=1)
            for _ in range(m):
                base = base.replace(day=1) - timedelta(days=1)
            session.add(
                MonthlyRevenue(
                    stock_id=sid,
                    date=base,
                    revenue=10_000_000 + int(sid) * 1000 + m * 100,
                    revenue_year=base.year,
                    revenue_month=base.month,
                    mom_growth=5.0 - m * 0.5,
                    yoy_growth=10.0 - m * 1.0,
                )
            )
    session.commit()


# ------------------------------------------------------------------ #
#  Tests — load_shared_market_data
# ------------------------------------------------------------------ #


class TestLoadSharedMarketData:
    def test_returns_dataclass_with_expected_fields(self, db_session):
        _insert_daily_prices(db_session, ["2330", "2317"], date.today() - timedelta(days=5), 5)
        _insert_institutional(db_session, ["2330", "2317"], date.today() - timedelta(days=5), 5)
        _insert_margin(db_session, ["2330", "2317"], date.today() - timedelta(days=5), 5)
        _insert_monthly_revenue(db_session, ["2330", "2317"], n_months=2)

        shared = load_shared_market_data(price_lookback_days=10, revenue_days=180)

        assert isinstance(shared, SharedMarketData)
        assert list(shared.df_price.columns) == [
            "stock_id",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ]
        assert list(shared.df_inst.columns) == ["stock_id", "date", "name", "net"]
        assert list(shared.df_margin.columns) == [
            "stock_id",
            "date",
            "margin_balance",
            "short_balance",
        ]
        assert list(shared.df_revenue.columns) == [
            "stock_id",
            "date",
            "yoy_growth",
            "mom_growth",
        ]
        assert shared.price_cutoff <= date.today()
        assert shared.revenue_cutoff < shared.price_cutoff  # revenue 拉更遠
        assert len(shared.df_price) == 2 * 5  # 2 股票 × 5 天

    def test_is_frozen_dataclass(self, db_session):
        shared = load_shared_market_data(price_lookback_days=10)
        with pytest.raises(Exception):  # dataclass(frozen=True) 阻止設定
            shared.df_price = pd.DataFrame()  # type: ignore

    def test_cutoff_honors_price_lookback_days(self, db_session):
        # 塞入跨越 cutoff 的資料：近 5 天 + 50 天前各 5 天
        _insert_daily_prices(db_session, ["2330"], date.today() - timedelta(days=5), 5)
        _insert_daily_prices(db_session, ["2330"], date.today() - timedelta(days=55), 5)

        shared = load_shared_market_data(price_lookback_days=10)  # cutoff = today - 20
        assert not shared.df_price.empty
        # 只保留近期 5 天（55 天前那批超出 cutoff）
        assert len(shared.df_price) == 5
        assert shared.df_price["date"].min() >= shared.price_cutoff


# ------------------------------------------------------------------ #
#  Tests — slice_revenue_raw（pivot 行為）
# ------------------------------------------------------------------ #


class TestSliceRevenueRaw:
    def _make_raw(self) -> pd.DataFrame:
        """建 2 支股票 × 4 個月的 raw rows（date desc 排序時 yoy 為 10/9/8/7）。"""
        rows = []
        for sid in ["A", "B"]:
            for m in range(4):
                rows.append(
                    {
                        "stock_id": sid,
                        "date": date(2025, 1, 1) + timedelta(days=30 * m),
                        "yoy_growth": 7.0 + m,  # m=0 → 7.0, m=3 → 10.0
                        "mom_growth": 1.0 + m * 0.5,
                    }
                )
        return pd.DataFrame(rows)

    def test_months_1_returns_latest_per_stock(self):
        raw = self._make_raw()
        out = slice_revenue_raw(raw, stock_ids=None, months=1)
        assert list(out.columns) == ["stock_id", "yoy_growth", "mom_growth"]
        assert len(out) == 2
        # 每支股票取最新 (m=3) → yoy=10.0
        assert (out["yoy_growth"] == 10.0).all()

    def test_months_2_computes_prev_growth(self):
        raw = self._make_raw()
        out = slice_revenue_raw(raw, stock_ids=None, months=2)
        assert set(out.columns) == {
            "stock_id",
            "yoy_growth",
            "mom_growth",
            "prev_yoy_growth",
            "prev_mom_growth",
        }
        assert len(out) == 2
        # 最新 m=3 (yoy=10)、上期 m=2 (yoy=9)
        assert (out["yoy_growth"] == 10.0).all()
        assert (out["prev_yoy_growth"] == 9.0).all()

    def test_months_4_computes_yoy_3m_ago(self):
        raw = self._make_raw()
        out = slice_revenue_raw(raw, stock_ids=None, months=4)
        assert "yoy_3m_ago" in out.columns
        # 最新 m=3 (yoy=10)、3 個月前 m=0 (yoy=7)
        assert (out["yoy_growth"] == 10.0).all()
        assert (out["yoy_3m_ago"] == 7.0).all()

    def test_stock_id_filter(self):
        raw = self._make_raw()
        out = slice_revenue_raw(raw, stock_ids=["A"], months=1)
        assert len(out) == 1
        assert out["stock_id"].iloc[0] == "A"

    def test_empty_input_returns_empty_with_correct_columns(self):
        empty = pd.DataFrame(columns=["stock_id", "date", "yoy_growth", "mom_growth"])
        out = slice_revenue_raw(empty, stock_ids=None, months=4)
        assert out.empty
        assert "yoy_3m_ago" in out.columns


# ------------------------------------------------------------------ #
#  Tests — Scanner 經由 shared 路徑 vs DB 路徑的一致性（B 項關鍵保證）
# ------------------------------------------------------------------ #


class TestSharedVsDBEquivalence:
    """關鍵：對相同 DB state，scanner._load_market_data() 走 shared 路徑
    vs 走 DB 路徑，df_price/df_inst/df_margin/df_revenue 必須逐列一致。"""

    def _seed(self, db_session, stock_ids=("2330", "2317", "2454")):
        start = date.today() - timedelta(days=100)
        sids = list(stock_ids)
        _insert_daily_prices(db_session, sids, start, 100)
        _insert_institutional(db_session, sids, start, 100)
        _insert_margin(db_session, sids, start, 100)
        _insert_monthly_revenue(db_session, sids, n_months=2)

    def _sort_and_reset(self, df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        if df.empty:
            return df
        return df.sort_values(keys).reset_index(drop=True)

    def test_momentum_shared_matches_db_path(self, db_session, monkeypatch):
        from src.discovery.scanner import MomentumScanner

        self._seed(db_session)

        # bypass UniverseFilter（會依賴 StockInfo 等，此處聚焦資料載入正確性）
        monkeypatch.setattr("src.discovery.scanner._base.MarketScanner._get_universe_ids", lambda self: [])

        scanner_db = MomentumScanner(min_volume=100_000, top_n_results=5, use_ic_adjustment=False)
        scanner_db._shared = None
        price_db, inst_db, margin_db, rev_db = scanner_db._load_market_data()

        shared = load_shared_market_data(price_lookback_days=80, revenue_days=180)
        scanner_shared = MomentumScanner(min_volume=100_000, top_n_results=5, use_ic_adjustment=False)
        scanner_shared._shared = shared
        price_sh, inst_sh, margin_sh, rev_sh = scanner_shared._load_market_data()

        pd.testing.assert_frame_equal(
            self._sort_and_reset(price_db, ["stock_id", "date"]),
            self._sort_and_reset(price_sh, ["stock_id", "date"]),
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            self._sort_and_reset(inst_db, ["stock_id", "date", "name"]),
            self._sort_and_reset(inst_sh, ["stock_id", "date", "name"]),
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            self._sort_and_reset(margin_db, ["stock_id", "date"]),
            self._sort_and_reset(margin_sh, ["stock_id", "date"]),
            check_like=True,
        )
        # Momentum _revenue_months=4，但測資只塞 2 個月 → 兩路徑都會給 None 欄位
        pd.testing.assert_frame_equal(
            self._sort_and_reset(rev_db, ["stock_id"]),
            self._sort_and_reset(rev_sh, ["stock_id"]),
            check_like=True,
        )

    def test_swing_shared_matches_db_path(self, db_session, monkeypatch):
        """Swing 覆寫 _load_market_data，月營收 months=2，需同樣一致。"""
        from src.discovery.scanner import SwingScanner

        self._seed(db_session)
        monkeypatch.setattr("src.discovery.scanner._base.MarketScanner._get_universe_ids", lambda self: [])

        scanner_db = SwingScanner(min_volume=100_000, top_n_results=5, use_ic_adjustment=False)
        scanner_db._shared = None
        price_db, inst_db, margin_db, rev_db = scanner_db._load_market_data()

        shared = load_shared_market_data(price_lookback_days=80, revenue_days=180)
        scanner_shared = SwingScanner(min_volume=100_000, top_n_results=5, use_ic_adjustment=False)
        scanner_shared._shared = shared
        price_sh, inst_sh, margin_sh, rev_sh = scanner_shared._load_market_data()

        pd.testing.assert_frame_equal(
            self._sort_and_reset(price_db, ["stock_id", "date"]),
            self._sort_and_reset(price_sh, ["stock_id", "date"]),
            check_like=True,
        )
        # Swing revenue months=2 → 兩路徑必須含 prev_yoy_growth / prev_mom_growth
        assert "prev_yoy_growth" in rev_sh.columns
        assert "prev_yoy_growth" in rev_db.columns


# ------------------------------------------------------------------ #
#  Tests — _cmd_discover_all 只呼叫一次 load_shared_market_data
# ------------------------------------------------------------------ #


class TestDiscoverAllSharedLoadCallCount:
    def test_load_shared_market_data_called_once(self, db_session, monkeypatch):
        """_cmd_discover_all 應只呼叫 1 次 load_shared_market_data，而非 5 次。"""
        import argparse

        from src.cli import discover_cmd as dc

        call_count = {"n": 0}

        def fake_load(price_lookback_days=80, revenue_days=180):
            call_count["n"] += 1
            return SharedMarketData(
                df_price=pd.DataFrame(
                    columns=[
                        "stock_id",
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "turnover",
                    ]
                ),
                df_inst=pd.DataFrame(columns=["stock_id", "date", "name", "net"]),
                df_margin=pd.DataFrame(columns=["stock_id", "date", "margin_balance", "short_balance"]),
                df_revenue=pd.DataFrame(columns=["stock_id", "date", "yoy_growth", "mom_growth"]),
                price_cutoff=date.today() - timedelta(days=90),
                revenue_cutoff=date.today() - timedelta(days=180),
                loaded_at=pd.Timestamp.utcnow().to_pydatetime(),
            )

        monkeypatch.setattr(dc, "load_shared_market_data", fake_load, raising=False)
        # 讓 import 成功：_cmd_discover_all 在函式內以 from ... import 的方式拿 load_shared_market_data
        monkeypatch.setattr("src.discovery.scanner._shared_load.load_shared_market_data", fake_load)
        # 跳過同步
        monkeypatch.setattr(dc, "ensure_sync_market_data", lambda *a, **kw: None)
        # 跳過所有 scanner.run（避免觸碰更多 DB / Regime detector）
        monkeypatch.setattr(
            "src.discovery.scanner._base.MarketScanner.run",
            lambda self, shared=None: _empty_discovery_result(self.mode_name),
        )
        # Growth / Value / Dividend 有自己的 run 覆寫，同樣短路
        for cls_path in (
            "src.discovery.scanner._growth.GrowthScanner.run",
            "src.discovery.scanner._value.ValueScanner.run",
            "src.discovery.scanner._dividend.DividendScanner.run",
        ):
            monkeypatch.setattr(cls_path, lambda self, shared=None: _empty_discovery_result(self.mode_name))

        args = argparse.Namespace(
            skip_sync=True,
            sync_days=80,
            top=20,
            min_price=10.0,
            max_price=2000.0,
            min_volume=500_000,
            max_stocks=None,
            min_appearances=1,
            export=None,
            notify=False,
            use_ic_adjustment=False,
            disabled_modes=[],
        )
        dc._cmd_discover_all(args)

        assert call_count["n"] == 1, (
            f"load_shared_market_data 應呼叫 1 次（共用給 5 scanner），實際 {call_count['n']} 次"
        )


def _empty_discovery_result(mode: str):
    """建立空的 DiscoveryResult 供測試短路 scanner.run。"""
    from src.discovery.scanner._functions import DiscoveryResult

    return DiscoveryResult(
        rankings=pd.DataFrame(),
        total_stocks=0,
        after_coarse=0,
        mode=mode,
    )
