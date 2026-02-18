"""共用 fixtures — in-memory DB、mock settings。"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data.database import Base


@pytest.fixture(scope="session")
def in_memory_engine():
    """建立 in-memory SQLite engine 並建表。"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture()
def db_session(in_memory_engine, monkeypatch):
    """每個測試獨立的 DB session（transaction rollback 隔離）。"""
    import src.data.database as db_mod

    connection = in_memory_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    monkeypatch.setattr(db_mod, "engine", in_memory_engine)
    monkeypatch.setattr(db_mod, "SessionLocal", Session)
    monkeypatch.setattr(db_mod, "get_session", lambda: session)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """建立 20 天的簡單 OHLCV 測試資料。"""
    dates = pd.bdate_range("2025-01-01", periods=20)
    base_price = 100.0
    rows = []
    for i, dt in enumerate(dates):
        close = base_price + i * 0.5
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
    df = pd.DataFrame(rows).set_index("date")
    return df
