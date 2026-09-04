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

    def test_recomputes_when_price_arrives_later(self, db, monkeypatch):
        """核心回歸：價量只補一半時算過特徵的日期，補齊後**必須**重算。

        2026-08-09 實測踩到——TPEX 同步逾時使該日只有上市價量，特徵算完寫入後
        日期被永久標記為已補，事後補齊上櫃價量也不再重算。11 天中招（含 3 個
        live 交易日），`daily_price` 4,400~7,300 列但 `daily_feature` 僅 1,147~1,362 列。
        後果不只重放失真：缺特徵列的股票被 `_stage2_liquidity_filter` **整批排除於
        universe 之外**（`avg5_map` 只由既有列建立），那 11 天的候選池少約四成。
        """
        from src.data.pipeline import backfill_daily_features
        from src.data.schema import DailyFeature, DailyPrice

        # 第一輪：只有 3 檔「上市」股
        self._seed(db, 20, n_stocks=3)
        first = backfill_daily_features(date(2024, 1, 1), date(2024, 2, 29), min_stocks_per_day=1)
        assert first["dates"] > 0

        # 事後補進 12 檔「上櫃」股（同一批日期），使特徵覆蓋率掉到 3/15 = 0.2
        dates = sorted({r[0] for r in db.query(DailyPrice.date).distinct().all()})
        for d in dates:
            for i in range(12):
                px = 50.0 + i
                db.add(
                    DailyPrice(
                        stock_id=f"{6000 + i}",
                        date=d,
                        open=px,
                        high=px + 1,
                        low=px - 1,
                        close=px,
                        volume=1000,
                        turnover=10000,
                    )
                )
        db.flush()

        second = backfill_daily_features(date(2024, 1, 1), date(2024, 2, 29), min_stocks_per_day=1)
        assert second["dates"] == first["dates"], "價量補齊後應重算全部日期，而非跳過"

        # 補齊後每日特徵檔數應與價量一致
        feat_per_day = {d: db.query(DailyFeature).filter(DailyFeature.date == d).count() for d in dates[-3:]}
        assert all(n == 15 for n in feat_per_day.values()), f"補齊後應每日 15 檔，實得 {feat_per_day}"

    def test_full_coverage_still_skips(self, db, monkeypatch):
        """對照組：覆蓋率已足時仍要跳過，證明上一題不是「永遠重算」。"""
        from src.data.pipeline import backfill_daily_features

        self._seed(db, 20, n_stocks=3)
        backfill_daily_features(date(2024, 1, 1), date(2024, 2, 29), min_stocks_per_day=1)
        again = backfill_daily_features(date(2024, 1, 1), date(2024, 2, 29), min_stocks_per_day=1)
        assert again["dates"] == 0

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


# ====================================================================== #
# §6.5 #20：估值歷史回補（backfill_valuation_history）
# ====================================================================== #


def _seed_valuation(session, d: date, n_stocks: int, prefix: int = 1000):
    from src.data.schema import StockValuation

    for i in range(n_stocks):
        session.add(StockValuation(stock_id=f"{prefix + i}", date=d, pe_ratio=10.0, pb_ratio=1.0, dividend_yield=3.0))
    session.flush()


class TestValuationBackfillTwse:
    """上市路徑：TWSE 每日全市場端點 + 以「當日估值檔數」判定續跑。

    §6.6 #27 起**只走 DB 認定的交易日**（該日 4 碼普通股價量檔數 ≥
    `BACKFILL_MIN_COMMON_STOCKS`），故各測試須先種當日價量才會被納入。
    """

    def test_skips_days_already_covered(self, db, monkeypatch):
        """已達 BACKFILL_MIN_VALUATION_STOCKS 的日期不得重抓。"""
        import src.data.twse_fetcher as tw
        from src.constants import BACKFILL_MIN_VALUATION_STOCKS
        from src.data.pipeline import backfill_valuation_history

        covered = date(2024, 1, 2)
        _seed_prices(db, covered, BACKFILL_MIN_COMMON_STOCKS)
        _seed_prices(db, date(2024, 1, 3), BACKFILL_MIN_COMMON_STOCKS)
        _seed_valuation(db, covered, BACKFILL_MIN_VALUATION_STOCKS)

        called: list[date] = []
        monkeypatch.setattr(tw, "fetch_twse_valuation_all", lambda d: (called.append(d), pd.DataFrame())[1])

        backfill_valuation_history(covered, date(2024, 1, 3), markets=("twse",))
        assert covered not in called, "已覆蓋日期不應重抓"
        assert date(2024, 1, 3) in called

    def test_thin_day_is_refetched(self, db, monkeypatch):
        """只有候選股估值（實測 43~150 檔）的日期必須重抓。

        這正是 §6.5 #20 要修的缺口——live 的 `sync_valuation_for_stocks` 每天只補
        候選池，若門檻設成「有無資料」，整段歷史都會被靜默跳過。
        """
        import src.data.twse_fetcher as tw
        from src.data.pipeline import backfill_valuation_history

        thin = date(2024, 1, 2)
        _seed_prices(db, thin, BACKFILL_MIN_COMMON_STOCKS)
        _seed_valuation(db, thin, 150, prefix=5000)  # live 候選股補抓的典型量

        called: list[date] = []
        monkeypatch.setattr(tw, "fetch_twse_valuation_all", lambda d: (called.append(d), pd.DataFrame())[1])

        backfill_valuation_history(thin, thin, markets=("twse",))
        assert called == [thin], "僅有候選股估值的日期必須視為未補"

    def test_writes_fetched_rows(self, db, monkeypatch):
        import src.data.twse_fetcher as tw
        from src.data.pipeline import backfill_valuation_history
        from src.data.schema import StockValuation

        d = date(2024, 1, 2)
        _seed_prices(db, d, BACKFILL_MIN_COMMON_STOCKS)
        monkeypatch.setattr(
            tw,
            "fetch_twse_valuation_all",
            lambda td: (
                pd.DataFrame(
                    [
                        {
                            "stock_id": f"{6000 + i}",
                            "date": td,
                            "pe_ratio": 12.0,
                            "pb_ratio": 1.5,
                            "dividend_yield": 4.0,
                        }
                        for i in range(5)
                    ]
                )
                if td == d
                else pd.DataFrame()
            ),
        )

        res = backfill_valuation_history(d, d, markets=("twse",))
        assert res["twse_days"] == 1
        rows = db.query(StockValuation).filter(StockValuation.date == d).all()
        assert len(rows) == 5
        assert rows[0].pe_ratio == 12.0

    def test_dry_run_fetches_nothing(self, db, monkeypatch):
        import src.data.twse_fetcher as tw
        from src.data.pipeline import backfill_valuation_history

        _seed_prices(db, date(2024, 1, 2), BACKFILL_MIN_COMMON_STOCKS)
        called: list[date] = []
        monkeypatch.setattr(tw, "fetch_twse_valuation_all", lambda d: (called.append(d), pd.DataFrame())[1])
        backfill_valuation_history(date(2024, 1, 2), date(2024, 1, 5), markets=("twse",), dry_run=True)
        assert called == []


class TestValuationHolidaySkip:
    """§6.6 #27：非交易日不得進入待補清單。"""

    def _stub(self, monkeypatch):
        import src.data.twse_fetcher as tw

        called: list[date] = []
        monkeypatch.setattr(tw, "fetch_twse_valuation_all", lambda d: (called.append(d), pd.DataFrame())[1])
        return called

    def test_weekend_and_holiday_not_requested(self, db, monkeypatch):
        """**核心回歸**：假日永遠達不到估值門檻，不濾掉就每次執行都重打。

        實測 2020–2023 有 69 天這樣的日子，資料其實零缺口，卻永遠列為待補
        （ETA 因此失真 5 倍）。
        """
        from src.data.pipeline import backfill_valuation_history

        # 2024-01-01 元旦（週一，休市）、01-06/07 週末；只有 02~05 有價量
        for d in (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)):
            _seed_prices(db, d, BACKFILL_MIN_COMMON_STOCKS)

        called = self._stub(monkeypatch)
        res = backfill_valuation_history(date(2024, 1, 1), date(2024, 1, 8), markets=("twse",))

        assert called == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
        assert date(2024, 1, 1) not in called, "元旦休市日不得請求"
        assert res["skipped_days"] == 0, "skipped_days 語意＝已在 DB 中的交易日"

    def test_calendar_gap_years_still_filtered(self, db, monkeypatch):
        """**不可改用 `calendar.is_trading_day`**：假日表只有 2025~2027。

        2021-02-11 是農曆除夕（休市）但不在 `_TWSE_HOLIDAYS`，行事曆會判為交易日。
        DB 判定不受此限——這正是本修法走 DB 而非行事曆的理由。
        """
        from src.data.calendar import is_trading_day
        from src.data.pipeline import backfill_valuation_history

        lunar_new_year_eve = date(2021, 2, 11)
        assert is_trading_day(lunar_new_year_eve), "前提：行事曆對 2021 年會誤判（假日表未涵蓋）"

        _seed_prices(db, date(2021, 2, 10), BACKFILL_MIN_COMMON_STOCKS)  # 除夕前最後交易日

        called = self._stub(monkeypatch)
        backfill_valuation_history(date(2021, 2, 10), date(2021, 2, 12), markets=("twse",))
        assert called == [date(2021, 2, 10)], "無價量的休市日不得請求"

    def test_thin_price_day_not_treated_as_trading_day(self, db, monkeypatch):
        """只有 watchlist 等級價量的日子不算交易日（與價量回補判定同源）。"""
        from src.data.pipeline import backfill_valuation_history

        _seed_prices(db, date(2024, 1, 2), 6)  # 2020~2024 的 watchlist 稀疏資料
        called = self._stub(monkeypatch)
        backfill_valuation_history(date(2024, 1, 2), date(2024, 1, 2), markets=("twse",))
        assert called == []

    def test_warns_when_no_trading_days(self, db, monkeypatch, caplog):
        """價量未回補時要明講前提不成立，而不是靜默回報「零待補」。"""
        import logging

        from src.data.pipeline import backfill_valuation_history

        called = self._stub(monkeypatch)
        with caplog.at_level(logging.WARNING):
            backfill_valuation_history(date(2024, 1, 2), date(2024, 1, 5), markets=("twse",))
        assert called == []
        assert any("查無交易日" in r.message for r in caplog.records)


class TestValuationBackfillTpex:
    """上櫃路徑：FinMind 逐股 + 以「估值日數 / 價量日數」比例判定續跑。

    TPEX 官方估值端點（`peratio_book/pera_result.php`）已下架——所有日期含當日
    皆 302 導向 `/errors`，新版 openapi 只回當日無歷史。故上櫃只能走 FinMind。
    """

    def _seed_otc(self, session, sid: str, price_days: int, val_days: int, start: date):
        from src.data.schema import StockValuation

        session.add(StockInfo(stock_id=sid, stock_name=f"櫃{sid}", listing_type="tpex", security_type="stock"))
        for i in range(price_days):
            session.add(
                DailyPrice(
                    stock_id=sid,
                    date=start + timedelta(days=i),
                    open=10.0,
                    high=11.0,
                    low=9.0,
                    close=10.0,
                    volume=1000,
                    turnover=10000,
                )
            )
        for i in range(val_days):
            session.add(
                StockValuation(
                    stock_id=sid, date=start + timedelta(days=i), pe_ratio=10.0, pb_ratio=1.0, dividend_yield=3.0
                )
            )
        session.flush()

    def _stub_finmind(self, monkeypatch, requested: list[str]):
        import src.data.pipeline as pl

        class _F:
            def fetch_per_pbr(self, sid, s, e):
                requested.append(sid)
                return pd.DataFrame(
                    [{"stock_id": sid, "date": s, "pe_ratio": 9.0, "pb_ratio": 1.1, "dividend_yield": 5.0}]
                )

        monkeypatch.setattr(pl, "FinMindFetcher", lambda *a, **kw: _F())

    def test_skips_well_covered_stock(self, db, monkeypatch):
        """估值覆蓋率達 VALUATION_COVERAGE_RATIO 的個股不得重抓。"""
        from src.data.pipeline import backfill_valuation_history

        start = date(2024, 1, 1)
        self._seed_otc(db, "6488", price_days=10, val_days=9, start=start)  # 90% ≥ 80%
        requested: list[str] = []
        self._stub_finmind(monkeypatch, requested)

        backfill_valuation_history(start, start + timedelta(days=20), markets=("tpex",))
        assert requested == []

    def test_refetches_thin_stock(self, db, monkeypatch):
        from src.data.pipeline import backfill_valuation_history

        start = date(2024, 1, 1)
        self._seed_otc(db, "6488", price_days=10, val_days=2, start=start)  # 20% < 80%
        requested: list[str] = []
        self._stub_finmind(monkeypatch, requested)

        backfill_valuation_history(start, start + timedelta(days=20), markets=("tpex",))
        assert requested == ["6488"]

    def test_stock_without_prices_is_skipped(self, db, monkeypatch):
        """區間內無價量的個股（尚未上市/早已下市）不必補，避免浪費 API 額度。"""
        from src.data.pipeline import backfill_valuation_history

        db.add(StockInfo(stock_id="9999", stock_name="無量", listing_type="tpex", security_type="stock"))
        db.flush()
        requested: list[str] = []
        self._stub_finmind(monkeypatch, requested)

        backfill_valuation_history(date(2024, 1, 1), date(2024, 1, 31), markets=("tpex",))
        assert requested == []

    def test_only_common_stock_backfilled(self, db, monkeypatch):
        """ETF/權證無本益比語意，不應消耗 FinMind 額度。"""
        from src.data.pipeline import backfill_valuation_history

        start = date(2024, 1, 1)
        self._seed_otc(db, "6488", price_days=10, val_days=0, start=start)
        # 同樣有價量、但 security_type 非 stock
        db.add(StockInfo(stock_id="0056", stock_name="高股息", listing_type="tpex", security_type="etf"))
        for i in range(10):
            db.add(
                DailyPrice(
                    stock_id="0056",
                    date=start + timedelta(days=i),
                    open=10.0,
                    high=11.0,
                    low=9.0,
                    close=10.0,
                    volume=1000,
                    turnover=10000,
                )
            )
        db.flush()
        requested: list[str] = []
        self._stub_finmind(monkeypatch, requested)

        backfill_valuation_history(start, start + timedelta(days=20), markets=("tpex",))
        assert requested == ["6488"], "ETF 不應被回補"

    def test_fetch_failure_does_not_abort_run(self, db, monkeypatch):
        """單一檔失敗不得中斷整輪——回補是長時作業，一檔壞掉全跑白費不可接受。"""
        import src.data.pipeline as pl
        from src.data.pipeline import backfill_valuation_history

        start = date(2024, 1, 1)
        for sid in ("6488", "5483"):
            self._seed_otc(db, sid, price_days=10, val_days=0, start=start)

        class _F:
            def fetch_per_pbr(self, sid, s, e):
                if sid == "6488":
                    raise RuntimeError("API 掛了")
                return pd.DataFrame(
                    [{"stock_id": sid, "date": s, "pe_ratio": 9.0, "pb_ratio": 1.1, "dividend_yield": 5.0}]
                )

        monkeypatch.setattr(pl, "FinMindFetcher", lambda *a, **kw: _F())
        res = backfill_valuation_history(start, start + timedelta(days=20), markets=("tpex",))
        assert res["tpex_stocks"] == 1, "5483 仍應被補進來"


class TestValuationBackfillCommon:
    def test_market_subset_respected(self, db, monkeypatch):
        import src.data.twse_fetcher as tw
        from src.data.pipeline import backfill_valuation_history

        called: list[date] = []
        monkeypatch.setattr(tw, "fetch_twse_valuation_all", lambda d: (called.append(d), pd.DataFrame())[1])
        res = backfill_valuation_history(date(2024, 1, 2), date(2024, 1, 3), markets=("tpex",))
        assert called == [], "markets 未含 twse 時不應打 TWSE 端點"
        assert res["twse_days"] == 0

    def test_empty_range_is_noop(self, db):
        from src.data.pipeline import backfill_valuation_history

        res = backfill_valuation_history(date(2024, 3, 1), date(2024, 1, 1))
        assert res["twse_days"] == 0 and res["tpex_stocks"] == 0


# ====================================================================== #
# §6.5 #20b：Stage 0.5 估值覆蓋閘門必須看「近期窗口」而非全表
# ====================================================================== #


class TestValuationFreshnessGate:
    """回歸測試：閘門看全表相異股票數 → 一旦歷史累積夠就永遠不再同步。

    實測 2026-07-31：全表 1,505 檔（閘門關閉）但當日僅 43 檔有估值，
    value/dividend 的 `_coarse_filter` 因而以數月前的舊 PE 評分。
    """

    def _scanner(self, monkeypatch, db_session, as_of: date):
        import src.discovery.scanner._base as base_mod
        from src.discovery.scanner import ValueScanner

        class _Ctx:
            def __enter__(self):
                return db_session

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(base_mod, "get_session", lambda: _Ctx())
        s = ValueScanner.__new__(ValueScanner)
        s.scan_date = as_of
        s._offline = False
        return s

    def _seed(self, session, d: date, n: int):
        from src.data.schema import StockValuation

        for i in range(n):
            session.add(StockValuation(stock_id=f"{1000 + i}", date=d, pe_ratio=10.0, pb_ratio=1.0, dividend_yield=3.0))
        session.flush()

    def test_stale_bulk_does_not_satisfy_gate(self, db_session, monkeypatch):
        """全表有 1,500 檔但都是半年前的 → 仍須觸發全市場同步。"""
        import src.data.pipeline as pl

        as_of = date(2026, 8, 3)
        self._seed(db_session, as_of - timedelta(days=180), 1500)  # 陳舊
        self._seed(db_session, as_of, 43)  # 當日只有候選池

        synced: list[int] = []
        monkeypatch.setattr(pl, "sync_valuation_all_market", lambda: synced.append(1) or 0)

        self._scanner(monkeypatch, db_session, as_of)._maybe_sync_valuation()
        assert synced, "近 7 日僅 43 檔，必須觸發同步"

    def test_fresh_coverage_skips_sync(self, db_session, monkeypatch):
        import src.data.pipeline as pl

        as_of = date(2026, 8, 3)
        self._seed(db_session, as_of - timedelta(days=1), 1000)

        synced: list[int] = []
        monkeypatch.setattr(pl, "sync_valuation_all_market", lambda: synced.append(1) or 0)

        self._scanner(monkeypatch, db_session, as_of)._maybe_sync_valuation()
        assert not synced, "近 7 日已有 1,000 檔，不應重複同步"

    def test_future_rows_do_not_count(self, db_session, monkeypatch):
        """PIT：`as_of` 之後的估值不得用來滿足覆蓋率（否則重放會洩題）。"""
        import src.data.pipeline as pl

        as_of = date(2026, 8, 3)
        self._seed(db_session, as_of + timedelta(days=1), 1000)  # 未來資料

        synced: list[int] = []
        monkeypatch.setattr(pl, "sync_valuation_all_market", lambda: synced.append(1) or 0)

        self._scanner(monkeypatch, db_session, as_of)._maybe_sync_valuation()
        assert synced, "未來資料不得計入覆蓋率"


class TestValuationQuotaHandling:
    """配額用盡必須立刻停手，不得像個股錯誤那樣續跑空轉。

    實測 2026-08-05 首跑：580 檔後 FinMind 回 402，其後 299 次呼叫全部瞬間失敗
    （間隔 0 秒）——純粹浪費，且真正的原因被淹沒在 299 行相同 WARNING 裡。
    """

    def _seed_otc(self, session, sid: str, start: date):
        session.add(StockInfo(stock_id=sid, stock_name=f"櫃{sid}", listing_type="tpex", security_type="stock"))
        for i in range(5):
            session.add(
                DailyPrice(
                    stock_id=sid,
                    date=start + timedelta(days=i),
                    open=10.0,
                    high=11.0,
                    low=9.0,
                    close=10.0,
                    volume=1000,
                    turnover=10000,
                )
            )
        session.flush()

    def test_quota_error_aborts_loop(self, db, monkeypatch):
        import src.data.pipeline as pl
        from src.data.pipeline import backfill_valuation_history

        start = date(2024, 1, 1)
        for sid in ("6001", "6002", "6003", "6004"):
            self._seed_otc(db, sid, start)

        attempted: list[str] = []

        class _Resp:
            status_code = 402

        class _F:
            def fetch_per_pbr(self, sid, s, e):
                attempted.append(sid)
                if sid == "6002":
                    exc = RuntimeError("402 Client Error: Payment Required")
                    exc.response = _Resp()
                    raise exc
                return pd.DataFrame(
                    [{"stock_id": sid, "date": s, "pe_ratio": 9.0, "pb_ratio": 1.1, "dividend_yield": 5.0}]
                )

        monkeypatch.setattr(pl, "FinMindFetcher", lambda *a, **kw: _F())
        res = backfill_valuation_history(start, start + timedelta(days=20), markets=("tpex",))

        assert attempted == ["6001", "6002"], "配額用盡後不應繼續呼叫"
        assert res["quota_exhausted"] == 1
        assert res["tpex_stocks"] == 1, "配額用盡前已成功的仍須計入"

    def test_ordinary_error_continues(self, db, monkeypatch):
        """對照組：非配額錯誤仍應跳過續跑（一檔壞掉不該讓整輪白費）。"""
        import src.data.pipeline as pl
        from src.data.pipeline import backfill_valuation_history

        start = date(2024, 1, 1)
        for sid in ("6001", "6002", "6003"):
            self._seed_otc(db, sid, start)

        attempted: list[str] = []

        class _F:
            def fetch_per_pbr(self, sid, s, e):
                attempted.append(sid)
                if sid == "6002":
                    raise ValueError("該股無資料")
                return pd.DataFrame(
                    [{"stock_id": sid, "date": s, "pe_ratio": 9.0, "pb_ratio": 1.1, "dividend_yield": 5.0}]
                )

        monkeypatch.setattr(pl, "FinMindFetcher", lambda *a, **kw: _F())
        res = backfill_valuation_history(start, start + timedelta(days=20), markets=("tpex",))

        assert attempted == ["6001", "6002", "6003"], "個股錯誤不應中斷整輪"
        assert res["quota_exhausted"] == 0
        assert res["tpex_stocks"] == 2

    def test_detector_recognises_status_codes(self):
        from src.data.pipeline import _is_quota_exhausted

        class _R:
            def __init__(self, c):
                self.status_code = c

        for code, expected in ((402, True), (429, True), (500, False), (404, False)):
            exc = RuntimeError("x")
            exc.response = _R(code)
            assert _is_quota_exhausted(exc) is expected, f"status {code}"
        # 只剩字串的情形
        assert _is_quota_exhausted(RuntimeError("402 Client Error: Payment Required")) is True
        assert _is_quota_exhausted(RuntimeError("connection reset")) is False


# ====================================================================== #
# G. §6.6 #23/#24 — 月營收日期語意、MOPS 續跑閘門、歷史回補
# ====================================================================== #


def _seed_revenue(session, stock_id: str, year: int, month: int, *, day: int | None = None, **kw):
    """種一筆月營收；`day=None` 走 canonical 月底，指定 day 則模擬舊 FinMind 慣例。"""
    from src.data.pit import month_end
    from src.data.schema import MonthlyRevenue

    d = month_end(year, month) if day is None else date(year, month, 1) + timedelta(days=day - 1)
    row = MonthlyRevenue(
        stock_id=stock_id,
        date=d,
        revenue=kw.pop("revenue", 1_000_000),
        revenue_year=year,
        revenue_month=month,
        mom_growth=kw.pop("mom_growth", None),
        yoy_growth=kw.pop("yoy_growth", None),
        source=kw.pop("source", None),
    )
    session.add(row)
    session.flush()
    return row


class TestRevenueDateNormalization:
    """§6.6 #23：`monthly_revenue` 的日期語意統一。"""

    def test_finmind_row_moves_to_month_end(self, db):
        from src.data.pipeline import normalize_revenue_date_semantics
        from src.data.schema import MonthlyRevenue

        # FinMind 慣例：1 月營收寫成 2/1
        _seed_revenue(db, "2330", 2024, 1, day=32)  # 2024-01-01 + 31 天 = 2024-02-01
        stats = normalize_revenue_date_semantics()

        row = db.query(MonthlyRevenue).one()
        assert row.date == date(2024, 1, 31), "次月 1 日必須改寫為營收月份的月底"
        assert row.source == "finmind", "來源標記必須依**原本**的日期慣例回填"
        assert stats["moved"] == 1

    def test_month_end_row_tagged_as_mops(self, db):
        from src.data.pipeline import normalize_revenue_date_semantics
        from src.data.schema import MonthlyRevenue

        _seed_revenue(db, "2330", 2024, 1)
        normalize_revenue_date_semantics()
        assert db.query(MonthlyRevenue).one().source == "mops"

    def test_duplicate_month_merges_keeping_mops(self, db):
        """**核心回歸**：同月雙列必須合併成一列，且保留 MOPS 的官方 YoY。

        實測 live 有 2,488 組這種重複——unique key 是 `(stock_id, date)`，
        兩套日期慣例並存時完全不衝突，`pivot_revenue_rows` 的 4 個月窗口
        因此實際只拿到 2 個月。
        """
        from src.data.pipeline import normalize_revenue_date_semantics
        from src.data.schema import MonthlyRevenue

        _seed_revenue(db, "2330", 2024, 1, revenue=500, yoy_growth=12.5)  # MOPS（月底）
        _seed_revenue(db, "2330", 2024, 1, day=32, revenue=500, mom_growth=3.0)  # FinMind（次月 1 日）

        stats = normalize_revenue_date_semantics()

        rows = db.query(MonthlyRevenue).all()
        assert len(rows) == 1, "同月只能留一列"
        assert rows[0].date == date(2024, 1, 31)
        assert rows[0].source == "mops", "衝突時保留權威來源"
        assert rows[0].yoy_growth == 12.5, "MOPS 的官方 YoY 不得被覆蓋"
        assert rows[0].mom_growth == 3.0, "被合併那筆的非空欄位要補進來"
        assert stats["merged"] == 1

    def test_is_idempotent(self, db):
        from src.data.pipeline import normalize_revenue_date_semantics
        from src.data.schema import MonthlyRevenue

        _seed_revenue(db, "2330", 2024, 1, day=32)
        _seed_revenue(db, "2454", 2024, 1)
        normalize_revenue_date_semantics()
        second = normalize_revenue_date_semantics()

        assert second == {"tagged": 0, "moved": 0, "merged": 0}, "第二次執行必須完全無動作"
        assert db.query(MonthlyRevenue).count() == 2


class TestMopsRevenueGate:
    """§6.6 #23：續跑閘門必須只數 MOPS 來源的股票數。"""

    def _stub_mops(self, monkeypatch, n_stocks: int):
        import src.data.mops_fetcher as mf

        called: list[tuple[int, int]] = []

        def _fetch(year=None, month=None):
            called.append((year, month))
            from src.data.pit import month_end

            return pd.DataFrame(
                [
                    {
                        "stock_id": f"{1000 + i}",
                        "date": month_end(year, month),
                        "revenue": 1_000_000,
                        "revenue_month": month,
                        "revenue_year": year,
                        "mom_growth": 1.0,
                        "yoy_growth": 2.0,
                    }
                    for i in range(n_stocks)
                ]
            )

        monkeypatch.setattr(mf, "fetch_mops_monthly_revenue", _fetch)
        return called

    def test_candidate_pool_rows_do_not_satisfy_gate(self, db, monkeypatch):
        """**核心回歸**：候選池逐股補抓的列再多也不算數。

        舊版數「該月全部列數 ≥500」，被 FinMind 列灌滿後全市場同步永不執行——
        實測 2026-02 的 MOPS 列因此永久停在 1 筆（候選池 1,284 筆）。
        """
        from src.constants import BACKFILL_MIN_REVENUE_STOCKS
        from src.data.pipeline import _sync_mops_revenue_month

        for i in range(BACKFILL_MIN_REVENUE_STOCKS + 100):
            _seed_revenue(db, f"9{i:03d}", 2026, 2, day=32, source="finmind")

        called = self._stub_mops(monkeypatch, 1500)
        _sync_mops_revenue_month(2026, 2)
        assert called == [(2026, 2)], "全部是 finmind 列時必須照抓"

    def test_partial_mops_month_is_retried(self, db, monkeypatch):
        """半套 MOPS（月初剛開始公布）不得被視為已完成。"""
        from src.data.pipeline import _sync_mops_revenue_month

        for i in range(498):  # 實測 2026-06 的 MOPS 覆蓋
            _seed_revenue(db, f"8{i:03d}", 2026, 6, source="mops")

        called = self._stub_mops(monkeypatch, 1700)
        _sync_mops_revenue_month(2026, 6)
        assert called == [(2026, 6)]

    def test_full_mops_month_is_skipped(self, db, monkeypatch):
        from src.constants import BACKFILL_MIN_REVENUE_STOCKS
        from src.data.pipeline import _sync_mops_revenue_month

        for i in range(BACKFILL_MIN_REVENUE_STOCKS + 10):
            _seed_revenue(db, f"7{i:04d}", 2026, 1, source="mops")

        called = self._stub_mops(monkeypatch, 1700)
        assert _sync_mops_revenue_month(2026, 1) == 0
        assert called == [], "MOPS 覆蓋已達標的月份不得重抓"

    def test_mops_upsert_overwrites_finmind_row(self, db, monkeypatch):
        """MOPS 必須覆蓋候選池先寫進來的 NULL YoY，並把 source 翻成 mops。

        否則該股永遠算不進閘門、`yoy_growth` 也永遠是 NULL（growth 粗篩要求非空）。
        """
        from src.data.pipeline import _sync_mops_revenue_month
        from src.data.schema import MonthlyRevenue

        _seed_revenue(db, "1000", 2026, 3, source="finmind", yoy_growth=None)
        self._stub_mops(monkeypatch, 3)
        _sync_mops_revenue_month(2026, 3)

        row = db.query(MonthlyRevenue).filter_by(stock_id="1000", revenue_year=2026, revenue_month=3).one()
        assert row.source == "mops"
        assert row.yoy_growth == 2.0


class TestRevenueBackfill:
    """§6.6 #24：`backfill_revenue_history` 的月份序列與續跑。"""

    def _stub(self, monkeypatch, n_stocks: int = 1700):
        return TestMopsRevenueGate()._stub_mops(monkeypatch, n_stocks)

    def test_iterates_months_across_year_boundary(self, db, monkeypatch):
        from src.data.pipeline import backfill_revenue_history

        called = self._stub(monkeypatch)
        res = backfill_revenue_history(date(2023, 11, 1), date(2024, 2, 28))

        assert called == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]
        assert res["months"] == 4

    def test_covered_months_are_skipped(self, db, monkeypatch):
        from src.constants import BACKFILL_MIN_REVENUE_STOCKS
        from src.data.pipeline import backfill_revenue_history

        for i in range(BACKFILL_MIN_REVENUE_STOCKS + 5):
            _seed_revenue(db, f"6{i:04d}", 2024, 1, source="mops")

        called = self._stub(monkeypatch)
        res = backfill_revenue_history(date(2024, 1, 1), date(2024, 2, 29))

        assert called == [(2024, 2)]
        assert res["skipped_months"] == 1

    def test_dry_run_does_not_fetch(self, db, monkeypatch):
        from src.data.pipeline import backfill_revenue_history

        called = self._stub(monkeypatch)
        res = backfill_revenue_history(date(2024, 1, 1), date(2024, 3, 31), dry_run=True)

        assert called == []
        assert res["months"] == 0

    def test_normalizes_before_fetching(self, db, monkeypatch):
        """回補前必須先正規化——否則寫進來的月底列會與舊的次月 1 日列並存。"""
        from src.data.pipeline import backfill_revenue_history
        from src.data.schema import MonthlyRevenue

        _seed_revenue(db, "2330", 2024, 1, day=32)  # 舊 FinMind 列
        self._stub(monkeypatch, n_stocks=2)  # 回補會寫 1000/1001，不含 2330
        backfill_revenue_history(date(2024, 1, 1), date(2024, 1, 31))

        rows = db.query(MonthlyRevenue).filter_by(stock_id="2330").all()
        assert len(rows) == 1 and rows[0].date == date(2024, 1, 31)


class TestGrowthStage05RevenueGate:
    """§6.6 #23：growth 的 Stage 0.5 必須看「該月覆蓋」而非全表相異股票數。"""

    def _scanner(self, as_of: date):
        from src.discovery.scanner._growth import GrowthScanner

        scanner = GrowthScanner.__new__(GrowthScanner)  # 跳過 __init__（不需要 universe/DB）
        scanner.scan_date = as_of  # `_as_of()` 讀的是 scan_date
        return scanner

    def test_syncs_latest_visible_month_not_calendar_last_month(self, monkeypatch):
        """月初尚未到 10 日時，該補的是**兩個月前**那個月（依法已公布者）。"""
        import src.data.pipeline as pl

        called: list[tuple[int, int]] = []
        monkeypatch.setattr(pl, "_sync_mops_revenue_month", lambda y, m: called.append((y, m)) or 0)

        self._scanner(date(2026, 3, 5))._prepare_before_load()
        assert called == [(2026, 1)], "3/5 尚未到 3/10，最新可見的是 1 月營收"

        called.clear()
        self._scanner(date(2026, 3, 10))._prepare_before_load()
        assert called == [(2026, 2)]

    def test_full_table_does_not_suppress_sync(self, db, monkeypatch):
        """**核心回歸**：全表已有上千檔，但當月缺席時仍必須補抓。

        舊版數 `count(distinct stock_id)` 且無日期條件，累積 ≥500 後永不觸發——
        live 因此在 2026-02 只寫進 1 筆的情況下完全沒有自癒。
        """
        import src.data.mops_fetcher as mf

        for i in range(1900):  # 全表很滿，但都是別的月份
            _seed_revenue(db, f"{1000 + i}", 2025, 12, source="mops")

        fetched: list[tuple[int, int]] = []
        monkeypatch.setattr(
            mf,
            "fetch_mops_monthly_revenue",
            lambda year=None, month=None: fetched.append((year, month)) or pd.DataFrame(),
        )

        self._scanner(date(2026, 3, 10))._prepare_before_load()
        assert fetched == [(2026, 2)], "當月無資料時必須實際打 MOPS"

    def test_covered_month_is_not_refetched(self, db, monkeypatch):
        import src.data.mops_fetcher as mf
        from src.constants import BACKFILL_MIN_REVENUE_STOCKS

        for i in range(BACKFILL_MIN_REVENUE_STOCKS + 5):
            _seed_revenue(db, f"{2000 + i}", 2026, 2, source="mops")

        fetched: list = []
        monkeypatch.setattr(
            mf,
            "fetch_mops_monthly_revenue",
            lambda year=None, month=None: fetched.append((year, month)) or pd.DataFrame(),
        )

        self._scanner(date(2026, 3, 10))._prepare_before_load()
        assert fetched == []


# ====================================================================== #
# H. §6.6 #25 — 財報回補（欄位級續跑判定 / 配額治理 / 母體）
# ====================================================================== #


def _seed_financial_stock(
    session,
    sid: str,
    *,
    n_days: int = 70,
    listing_type: str = "twse",
    security_type: str = "stock",
    turnover: float = 1_000_000.0,
    first_day: date = date(2024, 1, 2),
    step_days: int = 5,
):
    """種一檔可進財報回補母體的股票（StockInfo + 跨全年的價量）。"""
    from src.data.schema import StockInfo

    session.add(StockInfo(stock_id=sid, stock_name=sid, listing_type=listing_type, security_type=security_type))
    for i in range(n_days):
        session.add(
            DailyPrice(
                stock_id=sid,
                date=first_day + timedelta(days=i * step_days),
                open=10.0,
                high=11.0,
                low=9.0,
                close=10.0,
                volume=1000,
                turnover=turnover,
            )
        )
    session.flush()


def _seed_financial_rows(session, sid: str, quarters: list[date], *, eps=1.0, equity=100, operating_cf=50):
    """種財報列；把 equity / operating_cf 設 None 即模擬「三表只抓到損益表」的半套列。"""
    from src.data.schema import FinancialStatement

    for d in quarters:
        session.add(
            FinancialStatement(
                stock_id=sid,
                date=d,
                year=d.year,
                quarter=(d.month - 1) // 3 + 1,
                eps=eps,
                equity=equity,
                operating_cf=operating_cf,
            )
        )
    session.flush()


_Q2024 = [date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30)]  # 2024 年底前已屆申報期限的三季


def _stub_financial_fetcher(monkeypatch, *, rows_per_stock: int = 3, fail: dict | None = None):
    """打樁 FinMindFetcher：回傳被請求的股票清單，`fail[sid]` 指定要拋的例外。"""
    import src.data.pipeline as pl

    called: list[str] = []
    fail = fail or {}

    class _F:
        def fetch_quota_status(self):
            return {"level": 1, "level_title": "Free", "limit": 600, "used": 0, "remaining": 600}

        def fetch_financial_summary(self, sid, s, e):
            called.append(sid)
            if sid in fail:
                exc = fail[sid]
                raise exc() if isinstance(exc, type) else exc
            return pd.DataFrame(
                [
                    {
                        "stock_id": sid,
                        "date": _Q2024[i],
                        "year": 2024,
                        "quarter": i + 1,
                        "eps": 1.5,
                        "equity": 1000,
                        "operating_cf": 500,
                    }
                    for i in range(rows_per_stock)
                ]
            )

    monkeypatch.setattr(pl, "FinMindFetcher", lambda *a, **kw: _F())
    return called


def _quota_error(status: int = 402):
    class _R:
        status_code = status

    exc = RuntimeError("quota")
    exc.response = _R()
    return exc


class TestFinancialExpectedQuarters:
    """§6.6 #25：應有季數必須同時受「該股價量區間」與「法定申報期限」約束。"""

    def test_counts_only_published_quarters(self):
        from src.data.pipeline import _financial_expected_quarters

        # 2024 全年有價量，end=2024-12-31：Q4 年報要到 2025-03-31 才須申報 → 不算
        n = _financial_expected_quarters(date(2024, 1, 2), date(2024, 12, 30), date(2024, 1, 1), date(2024, 12, 31))
        assert n == 3

    def test_unpublished_quarter_not_required(self):
        """**核心回歸**：最近一季未到申報期限就不能算缺漏，否則該股每次都重抓。"""
        from src.data.pipeline import _financial_expected_quarters

        # end=2024-11-01：Q3（期限 11/14）尚未到 → 只剩 Q1/Q2
        n = _financial_expected_quarters(date(2024, 1, 2), date(2024, 11, 1), date(2024, 1, 1), date(2024, 11, 1))
        assert n == 2

    def test_late_listed_stock_not_required_to_cover_full_range(self):
        from src.data.pipeline import _financial_expected_quarters

        # 2024-08 才上市 → 只有 Q3 落在價量區間內
        n = _financial_expected_quarters(date(2024, 8, 1), date(2024, 12, 30), date(2024, 1, 1), date(2024, 12, 31))
        assert n == 1


class TestFinancialBackfillResume:
    """§6.6 #25：續跑判定看**欄位**不看列數。"""

    def test_half_filled_rows_are_retried(self, db, monkeypatch):
        """**核心回歸**：列數滿額但 equity/operating_cf 全 NULL 必須重抓。

        三表任一逾時時 `fetch_financial_summary` 仍會回傳只有損益表的 DataFrame，
        寫進去就是半套列——列數檢查完全看不出來，而 peer ranking 會照樣拿它排名。
        """
        from src.data.pipeline import backfill_financial_history

        _seed_financial_stock(db, "2330")
        _seed_financial_rows(db, "2330", _Q2024, equity=None, operating_cf=None)

        called = _stub_financial_fetcher(monkeypatch)
        backfill_financial_history(date(2024, 1, 1), date(2024, 12, 31), request_interval=0)
        assert called == ["2330"]

    def test_complete_stock_is_skipped(self, db, monkeypatch):
        from src.data.pipeline import backfill_financial_history

        _seed_financial_stock(db, "2330")
        _seed_financial_rows(db, "2330", _Q2024)

        called = _stub_financial_fetcher(monkeypatch)
        res = backfill_financial_history(date(2024, 1, 1), date(2024, 12, 31), request_interval=0)
        assert called == []
        assert res["skipped_stocks"] == 1

    def test_missing_stock_is_fetched(self, db, monkeypatch):
        from src.data.pipeline import backfill_financial_history

        _seed_financial_stock(db, "2330")
        called = _stub_financial_fetcher(monkeypatch)
        res = backfill_financial_history(date(2024, 1, 1), date(2024, 12, 31), request_interval=0)

        assert called == ["2330"]
        assert res["stocks"] == 1 and res["rows"] == 3

    def test_upsert_overwrites_half_filled_row(self, db, monkeypatch):
        """重抓回來的完整值必須覆蓋半套列——do_nothing 會讓續跑判定永遠自癒不了。"""
        from src.data.pipeline import backfill_financial_history
        from src.data.schema import FinancialStatement

        _seed_financial_stock(db, "2330")
        _seed_financial_rows(db, "2330", _Q2024, equity=None, operating_cf=None)

        _stub_financial_fetcher(monkeypatch)
        backfill_financial_history(date(2024, 1, 1), date(2024, 12, 31), request_interval=0)

        rows = db.query(FinancialStatement).filter_by(stock_id="2330").all()
        assert len(rows) == 3, "不得新增重複列"
        assert all(r.equity == 1000 and r.operating_cf == 500 for r in rows)


class TestFinancialBackfillUniverse:
    def test_illiquid_and_nonstock_excluded(self, db, monkeypatch):
        from src.data.pipeline import backfill_financial_history

        _seed_financial_stock(db, "2330")  # 70 個交易日 → 入選
        _seed_financial_stock(db, "0050", security_type="etf")  # ETF 無財報語意
        _seed_financial_stock(db, "9999", n_days=5)  # 交易日太少
        _seed_financial_stock(db, "5678", listing_type="emerging")  # 興櫃不補

        called = _stub_financial_fetcher(monkeypatch)
        backfill_financial_history(date(2024, 1, 1), date(2024, 12, 31), request_interval=0)
        assert called == ["2330"]

    def test_ordered_by_liquidity(self, db, monkeypatch):
        """中斷時要先補到最可能進 universe 的標的。"""
        from src.data.pipeline import backfill_financial_history

        _seed_financial_stock(db, "1111", turnover=1_000.0)
        _seed_financial_stock(db, "2222", turnover=9_000_000.0)
        _seed_financial_stock(db, "3333", turnover=500_000.0)

        called = _stub_financial_fetcher(monkeypatch)
        backfill_financial_history(date(2024, 1, 1), date(2024, 12, 31), request_interval=0)
        assert called == ["2222", "3333", "1111"]

    def test_dry_run_does_not_fetch(self, db, monkeypatch):
        from src.data.pipeline import backfill_financial_history

        _seed_financial_stock(db, "2330")
        called = _stub_financial_fetcher(monkeypatch)
        res = backfill_financial_history(date(2024, 1, 1), date(2024, 12, 31), dry_run=True, request_interval=0)
        assert called == []
        assert res["stocks"] == 0


class TestFinancialBackfillQuota:
    def test_stops_on_quota_exhaustion(self, db, monkeypatch):
        """配額用盡要立刻停手——續跑只會空轉，還會把真因淹沒在重複警告裡。"""
        from src.data.pipeline import backfill_financial_history

        _seed_financial_stock(db, "1111", turnover=900.0)
        _seed_financial_stock(db, "2222", turnover=9_000_000.0)

        called = _stub_financial_fetcher(monkeypatch, fail={"2222": _quota_error()})
        res = backfill_financial_history(date(2024, 1, 1), date(2024, 12, 31), request_interval=0)

        assert called == ["2222"], "撞到配額後不得繼續打第二檔"
        assert res["quota_exhausted"] == 1

    def test_wait_on_quota_sleeps_and_resumes(self, db, monkeypatch):
        from src.data.pipeline import backfill_financial_history

        _seed_financial_stock(db, "2330")

        attempts = {"n": 0}
        slept: list[float] = []

        import src.data.pipeline as pl

        class _F:
            def fetch_quota_status(self):
                return {}

            def fetch_financial_summary(self, sid, s, e):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise _quota_error()
                return pd.DataFrame(
                    [{"stock_id": sid, "date": _Q2024[0], "year": 2024, "quarter": 1, "eps": 1.0, "equity": 5}]
                )

        monkeypatch.setattr(pl, "FinMindFetcher", lambda *a, **kw: _F())
        monkeypatch.setattr(pl.time, "sleep", lambda s: slept.append(s))

        res = backfill_financial_history(date(2024, 1, 1), date(2024, 12, 31), request_interval=0, wait_on_quota=True)

        assert attempts["n"] == 2, "等待後必須重試同一檔"
        assert res["quota_waits"] == 1
        assert res["quota_exhausted"] == 0
        assert slept and slept[0] > 0

    def test_per_stock_error_does_not_abort(self, db, monkeypatch):
        from src.data.pipeline import backfill_financial_history

        _seed_financial_stock(db, "1111", turnover=900.0)
        _seed_financial_stock(db, "2222", turnover=9_000_000.0)

        called = _stub_financial_fetcher(monkeypatch, fail={"2222": ValueError("該股無資料")})
        res = backfill_financial_history(date(2024, 1, 1), date(2024, 12, 31), request_interval=0)

        assert called == ["2222", "1111"], "個股錯誤不應中斷整輪"
        assert res["failed_stocks"] == 1
        assert res["stocks"] == 1

    def test_interval_derived_from_quota(self, db, monkeypatch):
        """未指定 request_interval 時，節流間隔由帳號真實上限推導（3600/limit）。"""
        from src.data.pipeline import backfill_financial_history

        _seed_financial_stock(db, "2330")
        _seed_financial_rows(db, "2330", _Q2024)  # 已完整 → 不會真的 sleep

        import src.data.pipeline as pl

        captured = {}

        class _F:
            def fetch_quota_status(self):
                captured["asked"] = True
                return {"level": 1, "level_title": "Free", "limit": 600, "used": 0, "remaining": 600}

            def fetch_financial_summary(self, sid, s, e):  # pragma: no cover — 本測試不應走到
                raise AssertionError("已完整的股票不該被抓取")

        monkeypatch.setattr(pl, "FinMindFetcher", lambda *a, **kw: _F())
        backfill_financial_history(date(2024, 1, 1), date(2024, 12, 31))
        assert captured.get("asked"), "開跑前必須查配額"


class TestSecondsUntilNextHour:
    def test_returns_positive_with_buffer(self):
        from datetime import datetime

        from src.data.pipeline import _seconds_until_next_hour

        # 10:30:00 → 距 11:00 為 1800 秒，加 60 秒緩衝
        assert _seconds_until_next_hour(datetime(2026, 8, 15, 10, 30, 0)) == pytest.approx(1860.0)
        # 整點前一秒也必須是正值（不能算成 0 而連續重試）
        assert _seconds_until_next_hour(datetime(2026, 8, 15, 10, 59, 59)) == pytest.approx(61.0)
