"""B1① 歷史回補 + 下市清單測試（2026-08-01）。

倖存者偏差是歷史驗證最典型的系統性偏差，且方向恆定偏樂觀：`stock_info` 只描述
「今天還在市」的股票，任何歷史重放都會自動排除當時在市、後來下市的標的——
而下市股往往正是表現最差的那批。

解法有兩半，本檔各自覆蓋：
  1. **價格**：以 TWSE/TPEX 每日全市場端點回補。該端點回傳的是當日實際掛牌的
     所有股票，故 2020 年的檔案自然含有已下市標的——偏差在資料源頭就解掉。
     實測 2024-12-03 補回 5,859 檔，其中 19 檔如今已下市。
  2. **可交易性**：`stock_info.delisted_date`，供 PIT 判定「該股於 as_of 當時
     是否還在市」。

另有一個**曾實際踩到的陷阱**專門測試（`TestCoverageBasedResume`）：續跑判定
若只看「該日是否有資料」，會把 2020~2024 整整 5 年跳過——那些日期都有
`daily_price`，但每天只有 6 檔（watchlist + TAIEX）。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.constants import BACKFILL_MIN_COMMON_STOCKS
from src.data.schema import DailyPrice, StockInfo


@pytest.fixture()
def db(db_session, monkeypatch):
    """讓 pipeline 的 DB 存取全部落在單一測試 session。

    ⚠ `src/data/pipeline.py` 在 **module 層**綁定 `get_session`
    （`from src.data.database import ... get_session ...`），因此只 patch
    `src.data.database.get_session` **不會生效**——pipeline 仍用原函數，其中的
    `session.commit()` 會提交 conftest 的外層交易，導致 rollback 失效、
    session-scoped 的 `:memory:` 連線被關閉，後續測試全數噴
    `ResourceClosedError`。必須直接 patch `src.data.pipeline.get_session`。
    """
    import src.data.pipeline as pl

    class _Ctx:
        def __enter__(self):
            return db_session

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(pl, "get_session", lambda: _Ctx())
    monkeypatch.setattr(pl, "init_db", lambda: None)
    return db_session


def _seed_prices(session, d: date, n_stocks: int):
    for i in range(n_stocks):
        session.add(
            DailyPrice(
                stock_id=f"{1000 + i}",
                date=d,
                open=10.0,
                high=11.0,
                low=9.0,
                close=10.0,
                volume=1000,
                turnover=10000,
            )
        )
    session.flush()


def _stub_fetchers(monkeypatch, trading_days: set[date], n_rows: int = 600):
    """把 TWSE/TPEX 端點打樁，回傳被請求的日期清單。"""
    called: list[date] = []
    import src.data.twse_fetcher as tw

    def _price(target_date=None):
        called.append(target_date)
        if target_date not in trading_days:
            return pd.DataFrame()  # 假日
        return pd.DataFrame(
            [
                {
                    "stock_id": f"{2000 + i}",
                    "date": target_date,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0,
                    "volume": 1000,
                    "turnover": 10000,
                }
                for i in range(n_rows)
            ]
        )

    monkeypatch.setattr(tw, "fetch_market_daily_prices", _price)
    monkeypatch.setattr(tw, "fetch_market_institutional", lambda target_date=None: pd.DataFrame())
    monkeypatch.setattr(tw, "fetch_market_margin", lambda target_date=None: pd.DataFrame())
    return called


# ====================================================================== #
# A. 續跑判定必須看「全市場覆蓋」而非「日期是否存在」
# ====================================================================== #


class TestCoverageBasedResume:
    def test_thin_days_are_not_skipped(self, db, monkeypatch):
        """**核心回歸**：只有少數股票的日期不算已回補，必須重抓。

        實測 2020~2024 每個交易日都已有 daily_price 但只有 6 檔。若以「日期存在」
        判定，整整 5 年會被靜默跳過、回補什麼都不做。
        """
        from src.data.pipeline import backfill_market_history

        thin_day = date(2024, 12, 2)
        _seed_prices(db, thin_day, 6)  # watchlist 等級的稀疏資料
        called = _stub_fetchers(monkeypatch, {thin_day})

        backfill_market_history(thin_day, thin_day)
        assert thin_day in called, "只有 6 檔的日期必須被視為未回補"

    def test_full_coverage_days_are_skipped(self, db, monkeypatch):
        from src.data.pipeline import backfill_market_history

        full_day = date(2024, 12, 2)
        _seed_prices(db, full_day, BACKFILL_MIN_COMMON_STOCKS + 10)
        called = _stub_fetchers(monkeypatch, {full_day})

        res = backfill_market_history(full_day, full_day)
        assert called == [], "已達全市場覆蓋的日期不應重抓"
        assert res["trading_days"] == 0

    def test_boundary_at_threshold(self, db, monkeypatch):
        """恰好達門檻視為已覆蓋。"""
        from src.data.pipeline import backfill_market_history

        d = date(2024, 12, 2)
        _seed_prices(db, d, BACKFILL_MIN_COMMON_STOCKS)
        called = _stub_fetchers(monkeypatch, {d})
        backfill_market_history(d, d)
        assert called == []

    def test_crash_day_not_refetched(self, db, monkeypatch):
        """崩盤日總筆數少但普通股完整 → 不得重抓（否則永遠補不完）。

        實測 2025-04-07 關稅崩盤：TAIEX −9.7%、80.3% 普通股無量跌停，權證當日
        幾乎無報價 → 總筆數僅 2,922，但普通股 1,894 檔其實是完整的。
        """
        from src.data.pipeline import backfill_market_history

        d = date(2025, 4, 7)
        _seed_prices(db, d, BACKFILL_MIN_COMMON_STOCKS + 100)  # 普通股足量、無權證
        called = _stub_fetchers(monkeypatch, {d})
        backfill_market_history(d, d)
        assert called == [], "崩盤日的普通股是完整的，不應被判為未回補"

    def test_warrant_heavy_half_day_is_refetched(self, db, monkeypatch):
        """權證多但普通股只有一半 → 必須重抓（總筆數門檻會漏掉這種）。

        實測 2026-03-03：總筆數 5,795（權證多）但普通股僅 879 檔＝只有一個交易所。
        """
        from src.data.pipeline import backfill_market_history

        d = date(2026, 3, 3)
        _seed_prices(db, d, 879)  # 普通股不足
        for i in range(5000):  # 大量權證（6 碼）
            db.add(
                DailyPrice(
                    stock_id=f"7{i:05d}",
                    date=d,
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=10,
                    turnover=10,
                )
            )
        db.flush()
        called = _stub_fetchers(monkeypatch, {d})
        backfill_market_history(d, d)
        assert called == [d], "普通股不足即為半套，總筆數再多也要重抓"

    def test_fills_gap_in_middle(self, db, monkeypatch):
        """中間缺口也會被補（不只是往前延伸）。"""
        from src.data.pipeline import backfill_market_history

        days = [date(2024, 12, 2), date(2024, 12, 3), date(2024, 12, 4)]
        _seed_prices(db, days[0], BACKFILL_MIN_COMMON_STOCKS + 1)
        _seed_prices(db, days[2], BACKFILL_MIN_COMMON_STOCKS + 1)
        called = _stub_fetchers(monkeypatch, set(days))

        backfill_market_history(days[0], days[2])
        assert called == [days[1]], "只應補中間缺口"


# ====================================================================== #
# B. 範圍 / 假日 / dataset 子集 / dry-run
# ====================================================================== #


class TestBackfillBehaviour:
    def test_weekends_never_requested(self, db, monkeypatch):
        from src.data.pipeline import backfill_market_history

        # 2024-12-07 六、12-08 日
        called = _stub_fetchers(monkeypatch, set())
        backfill_market_history(date(2024, 12, 7), date(2024, 12, 8))
        assert called == []

    def test_holiday_returns_empty_and_is_not_counted(self, db, monkeypatch):
        """平日但 API 回空（假日）→ 不計為交易日，也不應中斷。"""
        from src.data.pipeline import backfill_market_history

        d = date(2024, 12, 2)
        called = _stub_fetchers(monkeypatch, set())  # 該日不在交易日集合
        res = backfill_market_history(d, d)
        assert called == [d]
        assert res["trading_days"] == 0

    def test_dry_run_does_not_fetch(self, db, monkeypatch):
        from src.data.pipeline import backfill_market_history

        called = _stub_fetchers(monkeypatch, {date(2024, 12, 2)})
        res = backfill_market_history(date(2024, 12, 2), date(2024, 12, 4), dry_run=True)
        assert called == []
        assert res["trading_days"] == 0

    def test_dataset_subset(self, db, monkeypatch):
        """--datasets price 時不應呼叫法人/融資券端點。"""
        from src.data.pipeline import backfill_market_history

        d = date(2024, 12, 2)
        _stub_fetchers(monkeypatch, {d})
        inst_called: list = []
        import src.data.twse_fetcher as tw

        monkeypatch.setattr(
            tw, "fetch_market_institutional", lambda target_date=None: inst_called.append(1) or pd.DataFrame()
        )
        backfill_market_history(d, d, datasets=("price",))
        assert inst_called == []

    def test_empty_range_is_noop(self, db, monkeypatch):
        from src.data.pipeline import backfill_market_history

        called = _stub_fetchers(monkeypatch, set())
        res = backfill_market_history(date(2024, 12, 5), date(2024, 12, 1))
        assert called == []
        assert res["trading_days"] == 0

    def test_stop_flag_aborts(self, db, monkeypatch):
        from src.data.pipeline import backfill_market_history

        days = {date(2024, 12, 2), date(2024, 12, 3), date(2024, 12, 4)}
        called = _stub_fetchers(monkeypatch, days)
        backfill_market_history(date(2024, 12, 2), date(2024, 12, 4), stop_flag=lambda: True)
        assert called == [], "stop_flag 應在第一日前即中止"


# ====================================================================== #
# C. 下市清單同步（倖存者偏差的另一半）
# ====================================================================== #


class TestDelistingSync:
    def _stub_list(self, monkeypatch, rows):
        import src.data.pipeline as pl

        class _F:
            def fetch_delisting_list(self):
                return pd.DataFrame(rows)

        monkeypatch.setattr(pl, "FinMindFetcher", lambda *a, **kw: _F())

    def test_adds_missing_stock(self, db, monkeypatch):
        from src.data.pipeline import sync_delisting_info

        self._stub_list(monkeypatch, [{"date": "2023-04-21", "stock_id": "9999", "stock_name": "已下市"}])
        assert sync_delisting_info() == 1
        row = db.query(StockInfo).filter_by(stock_id="9999").one()
        assert row.delisted_date == date(2023, 4, 21)
        assert row.security_type == "stock", "須能通過 universe SQL 過濾"

    def test_updates_existing_without_clobbering_industry(self, db, monkeypatch):
        from src.data.pipeline import sync_delisting_info

        db.add(StockInfo(stock_id="3454", stock_name="晶睿", industry_category="電子零組件業"))
        db.flush()
        self._stub_list(monkeypatch, [{"date": "2026-03-27", "stock_id": "3454", "stock_name": "晶睿"}])
        assert sync_delisting_info() == 1
        row = db.query(StockInfo).filter_by(stock_id="3454").one()
        assert row.delisted_date == date(2026, 3, 27)
        assert row.industry_category == "電子零組件業", "既有產業分類不得被覆蓋"

    def test_idempotent(self, db, monkeypatch):
        from src.data.pipeline import sync_delisting_info

        self._stub_list(monkeypatch, [{"date": "2023-04-21", "stock_id": "9999", "stock_name": "已下市"}])
        assert sync_delisting_info() == 1
        assert sync_delisting_info() == 0, "重跑不應重複更新"

    def test_empty_list_does_not_clear_existing(self, db, monkeypatch):
        """取得失敗時**不得**修改資料——否則會把全部股票誤標為未下市。"""
        from src.data.pipeline import sync_delisting_info

        db.add(StockInfo(stock_id="3454", stock_name="晶睿", delisted_date=date(2026, 3, 27)))
        db.flush()
        self._stub_list(monkeypatch, [])
        assert sync_delisting_info() == 0
        assert db.query(StockInfo).filter_by(stock_id="3454").one().delisted_date == date(2026, 3, 27)


# ====================================================================== #
# D. PIT 可交易性語意
# ====================================================================== #


class TestTradableAsOf:
    @pytest.mark.parametrize(
        "delisted,as_of,tradable",
        [
            (None, date(2022, 6, 1), True),  # 仍在市
            (date(2026, 3, 27), date(2022, 6, 1), True),  # 當時仍在市
            (date(2026, 3, 27), date(2026, 3, 26), True),  # 下市前一日
            (date(2026, 3, 27), date(2026, 3, 27), False),  # 下市當日起不可交易
            (date(2026, 3, 27), date(2026, 7, 1), False),
        ],
    )
    def test_semantics(self, delisted, as_of, tradable):
        """`delisted_date is None or delisted_date > as_of` ＝當時可交易。"""
        assert (delisted is None or delisted > as_of) is tradable

    def test_backfilled_day_contains_later_delisted_stocks(self, db, monkeypatch):
        """回補後的歷史日必須含有「當時在市、如今已下市」的股票。

        這正是倖存者偏差被解掉的證據——live 實測 2024-12-03 補回 5,859 檔，
        其中 19 檔如今已下市（6457 紘康、3202 樺晟等）。
        """
        from src.data.pipeline import backfill_market_history

        d = date(2024, 12, 2)
        db.add(StockInfo(stock_id="2000", stock_name="後來下市", delisted_date=date(2025, 6, 1)))
        db.flush()
        _stub_fetchers(monkeypatch, {d}, n_rows=BACKFILL_MIN_COMMON_STOCKS + 1)
        backfill_market_history(d, d)

        got = db.query(DailyPrice).filter_by(stock_id="2000", date=d).one_or_none()
        assert got is not None, "回補的歷史日應含當時在市、如今已下市的股票"


# ====================================================================== #
# E. 轉板誤記為下市的防線
# ====================================================================== #


class TestClearFalseDelistings:
    """FinMind `TaiwanStockDelisting` 收錄的是「從該板終止」，含**轉板**。

    實測 2026-08-03：30 檔有價量的下市股中，5236 凌陽創新於「下市日」2026-07-16
    之後仍持續正常交易（2026-08-03 成交 138,695 股）——它是上櫃轉上市，不是下市。
    誤判方向是過度保守（把可交易股票當成不可交易），比倖存者偏差安全但仍是錯的。
    """

    def _seed(self, session, sid: str, delisted: date, trading_days_after: int):
        session.add(StockInfo(stock_id=sid, stock_name=sid, delisted_date=delisted))
        for i in range(1, trading_days_after + 1):
            session.add(
                DailyPrice(
                    stock_id=sid,
                    date=delisted + timedelta(days=i),
                    open=10.0,
                    high=11.0,
                    low=9.0,
                    close=10.0,
                    volume=1000,
                    turnover=10000,
                )
            )
        session.flush()

    def test_clears_when_still_trading(self, db):
        from src.data.pipeline import clear_false_delistings

        self._seed(db, "5236", date(2026, 7, 16), trading_days_after=12)
        assert clear_false_delistings() == 1
        assert db.query(StockInfo).filter_by(stock_id="5236").one().delisted_date is None

    def test_keeps_true_delisting(self, db):
        """真正下市者（下市日後無報價）不得被清除。"""
        from src.data.pipeline import clear_false_delistings

        db.add(StockInfo(stock_id="3454", stock_name="晶睿", delisted_date=date(2026, 3, 27)))
        db.add(
            DailyPrice(
                stock_id="3454",
                date=date(2026, 3, 18),  # 下市前最後交易日
                open=10.0,
                high=11.0,
                low=9.0,
                close=10.0,
                volume=1000,
                turnover=10000,
            )
        )
        db.flush()
        assert clear_false_delistings() == 0
        assert db.query(StockInfo).filter_by(stock_id="3454").one().delisted_date == date(2026, 3, 27)

    def test_single_dirty_row_does_not_flip(self, db):
        """單筆髒資料不足以翻案——需達 min_trading_days_after 才判定誤記。"""
        from src.data.pipeline import clear_false_delistings

        self._seed(db, "9998", date(2026, 3, 27), trading_days_after=1)
        assert clear_false_delistings(min_trading_days_after=3) == 0
        assert db.query(StockInfo).filter_by(stock_id="9998").one().delisted_date is not None

    def test_threshold_boundary(self, db):
        from src.data.pipeline import clear_false_delistings

        self._seed(db, "9997", date(2026, 3, 27), trading_days_after=3)
        assert clear_false_delistings(min_trading_days_after=3) == 1

    def test_no_delisted_stocks_is_noop(self, db):
        from src.data.pipeline import clear_false_delistings

        db.add(StockInfo(stock_id="2330", stock_name="台積電"))
        db.flush()
        assert clear_false_delistings() == 0


# ====================================================================== #
# F. B1② DailyFeature 歷史化
# ====================================================================== #


class TestBackfillDailyFeatures:
    """PIT universe 過濾需要「當時那一天的特徵」，而每日路徑只寫最新一日。

    最大風險是**分批計算的邊界**：MA60 需 60 個交易日暖身，若 chunk 起點沒有
    足夠前置資料，邊界日的 ma60 會算錯。live 實測 6 個 chunk 邊界的 ma60 與
    獨立重算差異皆為 0。
    """

    def _seed(self, session, n_days: int, n_stocks: int = 3, start=date(2024, 1, 1)):
        from src.data.schema import DailyPrice

        d = start
        added = 0
        while added < n_days:
            if d.weekday() < 5:
                for i in range(n_stocks):
                    px = 100.0 + added + i
                    session.add(
                        DailyPrice(
                            stock_id=f"{1000 + i}",
                            date=d,
                            open=px,
                            high=px + 1,
                            low=px - 1,
                            close=px,
                            volume=1000,
                            turnover=10000,
                        )
                    )
                added += 1
            d += timedelta(days=1)
        session.flush()

    def test_writes_all_dates_not_only_latest(self, db, monkeypatch):
        """與每日增量路徑的關鍵差異：整個區間都要寫，不是只寫最新一日。"""
        from src.data.pipeline import backfill_daily_features
        from src.data.schema import DailyFeature

        self._seed(db, 30)
        res = backfill_daily_features(date(2024, 1, 1), date(2024, 2, 29), min_stocks_per_day=1)
        n_dates = db.query(DailyFeature.date).distinct().count()
        assert res["dates"] == n_dates > 1, "應寫入多個日期"

    def test_skips_already_computed_dates(self, db, monkeypatch):
        from src.data.pipeline import backfill_daily_features

        self._seed(db, 20)
        first = backfill_daily_features(date(2024, 1, 1), date(2024, 2, 29), min_stocks_per_day=1)
        second = backfill_daily_features(date(2024, 1, 1), date(2024, 2, 29), min_stocks_per_day=1)
        assert first["dates"] > 0
        assert second["dates"] == 0, "續跑應跳過已計算日期"
        assert second["skipped_dates"] == first["dates"]

    def test_features_are_backward_looking_only(self, db, monkeypatch):
        """PIT：D 日的 ma20 只能用 <= D 的收盤價。"""
        import pandas as pd

        from src.data.pipeline import backfill_daily_features
        from src.data.schema import DailyFeature, DailyPrice

        self._seed(db, 40, n_stocks=1)
        backfill_daily_features(date(2024, 1, 1), date(2024, 3, 31), min_stocks_per_day=1)

        px = pd.DataFrame(
            db.query(DailyPrice.date, DailyPrice.close).filter_by(stock_id="1000").order_by(DailyPrice.date).all(),
            columns=["date", "close"],
        )
        feats = db.query(DailyFeature.date, DailyFeature.ma20).filter_by(stock_id="1000").all()
        assert feats
        for d, ma20 in feats:
            if ma20 is None:
                continue
            expected = px[px["date"] <= d]["close"].tail(20).mean()
            assert abs(ma20 - expected) < 1e-9, f"{d} 的 ma20 用到了未來資料"

    def test_low_coverage_days_skipped(self, db, monkeypatch):
        """DailyPrice 覆蓋不足的日期不計算特徵（避免用殘缺資料算 rolling）。"""
        from src.data.pipeline import backfill_daily_features

        self._seed(db, 20, n_stocks=3)
        res = backfill_daily_features(date(2024, 1, 1), date(2024, 2, 29), min_stocks_per_day=1000)
        assert res["dates"] == 0

    def test_empty_range_is_noop(self, db, monkeypatch):
        from src.data.pipeline import backfill_daily_features

        res = backfill_daily_features(date(2024, 3, 1), date(2024, 1, 1))
        assert res == {"dates": 0, "rows": 0, "skipped_dates": 0}

    def test_shared_compute_function_is_used_by_both_paths(self):
        """SSOT：每日路徑與回補路徑必須呼叫同一個計算函數。

        兩邊算式若漂移，歷史特徵與今日特徵就不同質，PIT 重放得到的 universe
        便不是當時真正會產生的那個。
        """
        import inspect

        from src.data import pipeline as pl

        for fn in (pl.compute_and_store_daily_features, pl.backfill_daily_features):
            assert "compute_feature_columns" in inspect.getsource(fn), f"{fn.__name__} 未使用共用計算函數"
