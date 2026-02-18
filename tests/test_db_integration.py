"""測試 DB 整合 — ORM + upsert，使用 in-memory SQLite。"""

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import select, func

from src.data.database import Base, init_db
from src.data.schema import DailyPrice, InstitutionalInvestor, StockInfo
from src.data.pipeline import _upsert_batch, _get_last_date


class TestInitDb:
    def test_create_all_tables(self, db_session, in_memory_engine):
        """init_db() 應建立所有表。"""
        init_db()
        table_names = in_memory_engine.dialect.get_table_names(
            in_memory_engine.connect()
        )
        assert "daily_price" in table_names
        assert "institutional_investor" in table_names
        assert "technical_indicator" in table_names
        assert "backtest_result" in table_names
        assert "stock_info" in table_names


class TestUpsertBatch:
    def test_insert_new_records(self, db_session):
        df = pd.DataFrame([
            {"stock_id": "2330", "date": date(2024, 1, 2),
             "open": 580, "high": 585, "low": 578, "close": 583,
             "volume": 25_000_000, "turnover": 14_575_000_000, "spread": 3.0},
            {"stock_id": "2330", "date": date(2024, 1, 3),
             "open": 583, "high": 590, "low": 582, "close": 588,
             "volume": 30_000_000, "turnover": 17_640_000_000, "spread": 5.0},
        ])
        count = _upsert_batch(DailyPrice, df, ["stock_id", "date"])
        assert count == 2

        # 驗證資料寫入
        rows = db_session.execute(
            select(DailyPrice).where(DailyPrice.stock_id == "2330")
        ).scalars().all()
        assert len(rows) == 2

    def test_conflict_does_nothing(self, db_session):
        df = pd.DataFrame([
            {"stock_id": "2317", "date": date(2024, 1, 2),
             "open": 100, "high": 105, "low": 98, "close": 103,
             "volume": 10_000_000, "turnover": 1_030_000_000, "spread": 3.0},
        ])
        _upsert_batch(DailyPrice, df, ["stock_id", "date"])

        # 重複寫入同筆資料（不同 close）
        df2 = pd.DataFrame([
            {"stock_id": "2317", "date": date(2024, 1, 2),
             "open": 100, "high": 105, "low": 98, "close": 999,
             "volume": 10_000_000, "turnover": 1_030_000_000, "spread": 3.0},
        ])
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
        df = pd.DataFrame([
            {"stock_id": "0050", "date": date(2024, 1, 2),
             "open": 150, "high": 152, "low": 149, "close": 151,
             "volume": 5_000_000, "turnover": 755_000_000, "spread": 1.0},
            {"stock_id": "0050", "date": date(2024, 3, 15),
             "open": 155, "high": 157, "low": 154, "close": 156,
             "volume": 6_000_000, "turnover": 936_000_000, "spread": 5.0},
        ])
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

        row = db_session.execute(
            select(StockInfo).where(StockInfo.stock_id == "2330")
        ).scalar_one()
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

        rows = db_session.execute(
            select(InstitutionalInvestor).where(
                InstitutionalInvestor.stock_id == "2330"
            )
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].net == 5_000_000
