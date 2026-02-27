"""ETL Pipeline 測試 — _get_last_date、_upsert_batch、save 函數。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import select

from src.data.schema import (
    DailyPrice,
    PortfolioBacktestResult,
    Trade,
)


@pytest.fixture(autouse=True)
def _patch_pipeline_session(db_session, monkeypatch):
    """確保 pipeline 模組使用測試 session。"""
    import src.data.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "get_session", lambda: db_session)
    monkeypatch.setattr(pipeline_mod, "init_db", lambda: None)


def _daily_price_row(**kwargs):
    """建立一筆 DailyPrice 用的完整 dict。"""
    defaults = {
        "stock_id": "2330",
        "date": date(2024, 1, 1),
        "open": 600.0,
        "high": 610.0,
        "low": 595.0,
        "close": 605.0,
        "volume": 10000,
        "turnover": 6_050_000,
    }
    defaults.update(kwargs)
    return defaults


# ================================================================
# _get_last_date
# ================================================================


class TestGetLastDate:
    def test_returns_date_when_data_exists(self, db_session):
        from src.data.pipeline import _get_last_date

        dp = DailyPrice(
            stock_id="2330",
            date=date(2024, 6, 15),
            open=600,
            high=610,
            low=595,
            close=605,
            volume=10000,
            turnover=6_050_000,
        )
        db_session.add(dp)
        db_session.flush()

        result = _get_last_date(DailyPrice, "2330")
        assert result == "2024-06-15"

    def test_returns_none_when_no_data(self, db_session):
        from src.data.pipeline import _get_last_date

        result = _get_last_date(DailyPrice, "9999")
        assert result is None


# ================================================================
# _upsert_batch
# ================================================================


class TestUpsertBatch:
    def test_insert_new_records(self, db_session):
        from src.data.pipeline import _upsert_batch

        df = pd.DataFrame(
            [
                _daily_price_row(date=date(2024, 1, 1)),
                _daily_price_row(date=date(2024, 1, 2)),
            ]
        )
        count = _upsert_batch(DailyPrice, df, ["stock_id", "date"])
        assert count == 2

    def test_conflict_skip(self, db_session):
        from src.data.pipeline import _upsert_batch

        df = pd.DataFrame([_daily_price_row()])
        _upsert_batch(DailyPrice, df, ["stock_id", "date"])
        count = _upsert_batch(DailyPrice, df, ["stock_id", "date"])
        assert count == 1  # 回傳記錄數（即使被跳過）

    def test_empty_df_returns_zero(self, db_session):
        from src.data.pipeline import _upsert_batch

        count = _upsert_batch(DailyPrice, pd.DataFrame(), ["stock_id", "date"])
        assert count == 0

    def test_batch_splitting(self, db_session):
        from src.data.pipeline import _upsert_batch

        rows = [_daily_price_row(date=date(2024, 1, 1) + pd.Timedelta(days=i)) for i in range(100)]
        df = pd.DataFrame(rows)
        count = _upsert_batch(DailyPrice, df, ["stock_id", "date"])
        assert count == 100


# ================================================================
# save_backtest_result
# ================================================================


@dataclass
class MockBacktestResult:
    stock_id: str = "2330"
    strategy_name: str = "sma_cross"
    start_date: date = date(2024, 1, 1)
    end_date: date = date(2024, 6, 30)
    initial_capital: float = 1_000_000
    final_capital: float = 1_100_000
    total_return: float = 10.0
    annual_return: float = 20.0
    sharpe_ratio: float = 1.5
    max_drawdown: float = -5.0
    win_rate: float = 60.0
    total_trades: int = 5
    benchmark_return: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    var_95: float | None = None
    cvar_95: float | None = None
    profit_factor: float | None = None
    trades: list = field(default_factory=list)


@dataclass
class MockTrade:
    entry_date: date = date(2024, 1, 10)
    entry_price: float = 600.0
    exit_date: date = date(2024, 2, 10)
    exit_price: float = 650.0
    shares: int = 100
    pnl: float = 5000.0
    return_pct: float = 8.33
    exit_reason: str = "signal"


class TestSaveBacktestResult:
    def test_save_and_return_id(self, db_session):
        from src.data.pipeline import save_backtest_result

        bt_id = save_backtest_result(MockBacktestResult(trades=[MockTrade()]))
        assert isinstance(bt_id, int)
        assert bt_id > 0

    def test_trades_saved(self, db_session):
        from src.data.pipeline import save_backtest_result

        trades = [MockTrade(), MockTrade(entry_date=date(2024, 3, 1))]
        bt_id = save_backtest_result(MockBacktestResult(trades=trades))

        saved = db_session.execute(select(Trade).where(Trade.backtest_id == bt_id)).scalars().all()
        assert len(saved) == 2

    def test_no_trades(self, db_session):
        from src.data.pipeline import save_backtest_result

        bt_id = save_backtest_result(MockBacktestResult(trades=[]))
        assert bt_id > 0


# ================================================================
# save_portfolio_result
# ================================================================


@dataclass
class MockPortfolioTrade:
    stock_id: str = "2330"
    entry_date: date = date(2024, 1, 10)
    entry_price: float = 600.0
    exit_date: date = date(2024, 2, 10)
    exit_price: float = 650.0
    shares: int = 100
    pnl: float = 5000.0
    return_pct: float = 8.33
    exit_reason: str = "signal"


@dataclass
class MockPortfolioResult:
    strategy_name: str = "sma_cross"
    stock_ids: list = field(default_factory=lambda: ["2330", "2317"])
    start_date: date = date(2024, 1, 1)
    end_date: date = date(2024, 6, 30)
    initial_capital: float = 1_000_000
    final_capital: float = 1_100_000
    total_return: float = 10.0
    annual_return: float = 20.0
    sharpe_ratio: float = 1.5
    max_drawdown: float = -5.0
    win_rate: float = 60.0
    total_trades: int = 3
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    var_95: float | None = None
    cvar_95: float | None = None
    profit_factor: float | None = None
    allocation_method: str = "equal_weight"
    trades: list = field(default_factory=list)


class TestSavePortfolioResult:
    def test_save_and_return_id(self, db_session):
        from src.data.pipeline import save_portfolio_result

        pbt_id = save_portfolio_result(MockPortfolioResult(trades=[MockPortfolioTrade()]))
        assert isinstance(pbt_id, int)
        assert pbt_id > 0

    def test_stock_ids_joined(self, db_session):
        from src.data.pipeline import save_portfolio_result

        pbt_id = save_portfolio_result(MockPortfolioResult(trades=[]))
        saved = db_session.execute(
            select(PortfolioBacktestResult).where(PortfolioBacktestResult.id == pbt_id)
        ).scalar_one()
        assert saved.stock_ids == "2330,2317"


# ================================================================
# _upsert 便利函數
# ================================================================


class TestUpsertConvenience:
    def test_upsert_daily_price(self, db_session):
        from src.data.pipeline import _upsert_daily_price

        df = pd.DataFrame([_daily_price_row(date=date(2024, 3, 1))])
        count = _upsert_daily_price(df)
        assert count == 1

    def test_upsert_announcement(self, db_session):
        from src.data.pipeline import _upsert_announcement

        df = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "date": [date(2024, 3, 1)],
                "seq": ["1"],
                "subject": ["重大訊息測試"],
                "sentiment": [0],
            }
        )
        count = _upsert_announcement(df)
        assert count == 1


# ================================================================
# sync_mops_announcements
# ================================================================


class TestSyncMopsAnnouncements:
    @patch("src.data.mops_fetcher.fetch_mops_announcements")
    def test_empty_result(self, mock_fetch, db_session):
        from src.data.pipeline import sync_mops_announcements

        mock_fetch.return_value = pd.DataFrame()
        count = sync_mops_announcements()
        assert count == 0

    @patch("src.data.mops_fetcher.fetch_mops_announcements")
    def test_with_data(self, mock_fetch, db_session):
        from src.data.pipeline import sync_mops_announcements

        mock_fetch.return_value = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "date": [date(2024, 3, 1)],
                "seq": ["1"],
                "subject": ["重大訊息"],
                "sentiment": [0],
            }
        )
        count = sync_mops_announcements()
        assert count == 1
