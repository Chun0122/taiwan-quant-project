"""測試 DB 整合 — ORM + upsert，使用 in-memory SQLite。"""

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import select

from src.data.database import init_db
from src.data.pipeline import _get_last_date, _upsert_batch
from src.data.schema import DailyPrice, DiscoveryRecord, InstitutionalInvestor, StockInfo


class TestInitDb:
    def test_create_all_tables(self, db_session, in_memory_engine):
        """init_db() 應建立所有表。"""
        init_db()
        table_names = in_memory_engine.dialect.get_table_names(in_memory_engine.connect())
        assert "daily_price" in table_names
        assert "institutional_investor" in table_names
        assert "technical_indicator" in table_names
        assert "backtest_result" in table_names
        assert "stock_info" in table_names
        assert "discovery_record" in table_names


class TestUpsertBatch:
    def test_insert_new_records(self, db_session):
        df = pd.DataFrame(
            [
                {
                    "stock_id": "2330",
                    "date": date(2024, 1, 2),
                    "open": 580,
                    "high": 585,
                    "low": 578,
                    "close": 583,
                    "volume": 25_000_000,
                    "turnover": 14_575_000_000,
                    "spread": 3.0,
                },
                {
                    "stock_id": "2330",
                    "date": date(2024, 1, 3),
                    "open": 583,
                    "high": 590,
                    "low": 582,
                    "close": 588,
                    "volume": 30_000_000,
                    "turnover": 17_640_000_000,
                    "spread": 5.0,
                },
            ]
        )
        count = _upsert_batch(DailyPrice, df, ["stock_id", "date"])
        assert count == 2

        # 驗證資料寫入
        rows = db_session.execute(select(DailyPrice).where(DailyPrice.stock_id == "2330")).scalars().all()
        assert len(rows) == 2

    def test_conflict_does_nothing(self, db_session):
        df = pd.DataFrame(
            [
                {
                    "stock_id": "2317",
                    "date": date(2024, 1, 2),
                    "open": 100,
                    "high": 105,
                    "low": 98,
                    "close": 103,
                    "volume": 10_000_000,
                    "turnover": 1_030_000_000,
                    "spread": 3.0,
                },
            ]
        )
        _upsert_batch(DailyPrice, df, ["stock_id", "date"])

        # 重複寫入同筆資料（不同 close）
        df2 = pd.DataFrame(
            [
                {
                    "stock_id": "2317",
                    "date": date(2024, 1, 2),
                    "open": 100,
                    "high": 105,
                    "low": 98,
                    "close": 999,
                    "volume": 10_000_000,
                    "turnover": 1_030_000_000,
                    "spread": 3.0,
                },
            ]
        )
        _upsert_batch(DailyPrice, df2, ["stock_id", "date"])

        row = db_session.execute(
            select(DailyPrice).where(
                DailyPrice.stock_id == "2317",
                DailyPrice.date == date(2024, 1, 2),
            )
        ).scalar_one()
        # 衝突時 do_nothing → 保留原值
        assert row.close == 103

    def test_empty_dataframe(self, db_session):
        count = _upsert_batch(DailyPrice, pd.DataFrame(), ["stock_id", "date"])
        assert count == 0


class TestGetLastDate:
    def test_returns_last_date(self, db_session):
        df = pd.DataFrame(
            [
                {
                    "stock_id": "0050",
                    "date": date(2024, 1, 2),
                    "open": 150,
                    "high": 152,
                    "low": 149,
                    "close": 151,
                    "volume": 5_000_000,
                    "turnover": 755_000_000,
                    "spread": 1.0,
                },
                {
                    "stock_id": "0050",
                    "date": date(2024, 3, 15),
                    "open": 155,
                    "high": 157,
                    "low": 154,
                    "close": 156,
                    "volume": 6_000_000,
                    "turnover": 936_000_000,
                    "spread": 5.0,
                },
            ]
        )
        _upsert_batch(DailyPrice, df, ["stock_id", "date"])

        last = _get_last_date(DailyPrice, "0050")
        assert last == "2024-03-15"

    def test_no_data_returns_none(self, db_session):
        last = _get_last_date(DailyPrice, "XXXX")
        assert last is None


class TestOrmCrud:
    def test_add_and_query_stock_info(self, db_session):
        info = StockInfo(
            stock_id="2330",
            stock_name="台積電",
            industry_category="半導體業",
            listing_type="twse",
        )
        db_session.add(info)
        db_session.flush()

        row = db_session.execute(select(StockInfo).where(StockInfo.stock_id == "2330")).scalar_one()
        assert row.stock_name == "台積電"
        assert row.industry_category == "半導體業"

    def test_add_institutional_investor(self, db_session):
        rec = InstitutionalInvestor(
            stock_id="2330",
            date=date(2024, 1, 2),
            name="Foreign_Investor",
            buy=10_000_000,
            sell=5_000_000,
            net=5_000_000,
        )
        db_session.add(rec)
        db_session.flush()

        rows = (
            db_session.execute(select(InstitutionalInvestor).where(InstitutionalInvestor.stock_id == "2330"))
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].net == 5_000_000


class TestDiscoveryRecord:
    def test_insert_and_query(self, db_session):
        """DiscoveryRecord 基本 CRUD。"""
        rec = DiscoveryRecord(
            scan_date=date(2025, 6, 1),
            mode="momentum",
            rank=1,
            stock_id="2330",
            stock_name="台積電",
            close=950.0,
            composite_score=0.85,
            technical_score=0.9,
            chip_score=0.8,
            fundamental_score=0.7,
            news_score=0.6,
            sector_bonus=0.03,
            industry_category="半導體業",
            regime="bull",
            total_stocks=1800,
            after_coarse=120,
            entry_price=950.0,
            stop_loss=910.5,
            take_profit=1031.0,
            entry_trigger="站上均線，低波動",
            valid_until=date(2025, 6, 8),
        )
        db_session.add(rec)
        db_session.flush()

        rows = (
            db_session.execute(
                select(DiscoveryRecord).where(
                    DiscoveryRecord.scan_date == date(2025, 6, 1),
                    DiscoveryRecord.mode == "momentum",
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].stock_id == "2330"
        assert rows[0].composite_score == 0.85
        assert rows[0].regime == "bull"
        assert rows[0].entry_price == 950.0
        assert rows[0].stop_loss == 910.5
        assert rows[0].take_profit == 1031.0
        assert rows[0].entry_trigger == "站上均線，低波動"
        assert rows[0].valid_until == date(2025, 6, 8)

    def test_entry_exit_persists(self, db_session):
        """進出場建議欄位寫入後可正確讀取，nullable 欄位為 None 時也正常。"""
        # 有進出場資料
        rec_full = DiscoveryRecord(
            scan_date=date(2025, 7, 1),
            mode="swing",
            rank=1,
            stock_id="2317",
            close=105.0,
            composite_score=0.78,
            entry_price=105.0,
            stop_loss=99.75,
            take_profit=120.75,
            entry_trigger="貼近均線",
            valid_until=date(2025, 7, 8),
        )
        # 無進出場資料（nullable）
        rec_null = DiscoveryRecord(
            scan_date=date(2025, 7, 1),
            mode="swing",
            rank=2,
            stock_id="2454",
            close=200.0,
            composite_score=0.65,
        )
        db_session.add_all([rec_full, rec_null])
        db_session.flush()

        rows = (
            db_session.execute(
                select(DiscoveryRecord).where(
                    DiscoveryRecord.scan_date == date(2025, 7, 1),
                    DiscoveryRecord.mode == "swing",
                )
            )
            .scalars()
            .all()
        )
        by_stock = {r.stock_id: r for r in rows}

        # 完整欄位
        r1 = by_stock["2317"]
        assert r1.entry_price == pytest.approx(105.0)
        assert r1.stop_loss == pytest.approx(99.75)
        assert r1.take_profit == pytest.approx(120.75)
        assert r1.entry_trigger == "貼近均線"
        assert r1.valid_until == date(2025, 7, 8)

        # nullable 欄位為 None
        r2 = by_stock["2454"]
        assert r2.entry_price is None
        assert r2.stop_loss is None
        assert r2.valid_until is None

    def test_unique_constraint(self, db_session):
        """同日同模式同股票不可重複。"""
        from sqlalchemy.exc import IntegrityError

        rec1 = DiscoveryRecord(
            scan_date=date(2025, 6, 2),
            mode="swing",
            rank=1,
            stock_id="2317",
            close=100.0,
            composite_score=0.7,
        )
        rec2 = DiscoveryRecord(
            scan_date=date(2025, 6, 2),
            mode="swing",
            rank=2,
            stock_id="2317",
            close=100.0,
            composite_score=0.6,
        )
        db_session.add(rec1)
        db_session.flush()
        db_session.add(rec2)
        try:
            db_session.flush()
            assert False, "應該觸發 IntegrityError"
        except IntegrityError:
            db_session.rollback()

    def test_multiple_dates_for_comparison(self, db_session):
        """多日記錄查詢：可找到前一次掃描日期。"""
        from sqlalchemy import func

        for day, stock_id in [(1, "2330"), (3, "2330"), (3, "2317")]:
            db_session.add(
                DiscoveryRecord(
                    scan_date=date(2025, 6, day),
                    mode="momentum",
                    rank=1 if stock_id == "2330" else 2,
                    stock_id=stock_id,
                    close=100.0,
                    composite_score=0.8,
                )
            )
        db_session.flush()

        # 查詢 6/3 之前最近的掃描日期
        prev_date = db_session.execute(
            select(func.max(DiscoveryRecord.scan_date)).where(
                DiscoveryRecord.mode == "momentum",
                DiscoveryRecord.scan_date < date(2025, 6, 3),
            )
        ).scalar()
        assert prev_date == date(2025, 6, 1)

        # 6/3 有兩筆記錄
        rows = (
            db_session.execute(
                select(DiscoveryRecord).where(
                    DiscoveryRecord.scan_date == date(2025, 6, 3),
                    DiscoveryRecord.mode == "momentum",
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 2
