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
# _validate_ohlcv
# ================================================================


class TestValidateOhlcv:
    def test_empty_df_passthrough(self):
        from src.data.pipeline import _validate_ohlcv

        df = pd.DataFrame()
        assert _validate_ohlcv(df).empty

    def test_valid_row_kept(self):
        from src.data.pipeline import _validate_ohlcv

        df = pd.DataFrame([_daily_price_row()])
        assert len(_validate_ohlcv(df)) == 1

    def test_close_at_high_is_valid(self):
        """close==high（收在最高價）為合法收盤，不應被攔。"""
        from src.data.pipeline import _validate_ohlcv

        df = pd.DataFrame([_daily_price_row(high=105.4, low=102.7, close=105.4)])
        assert len(_validate_ohlcv(df)) == 1

    def test_close_at_low_is_valid(self):
        """close==low（收在最低價）為合法收盤，不應被攔。"""
        from src.data.pipeline import _validate_ohlcv

        df = pd.DataFrame([_daily_price_row(high=105.4, low=102.7, close=102.7)])
        assert len(_validate_ohlcv(df)) == 1

    def test_close_above_high_dropped(self):
        """close > high：OHLC 不一致，過濾。"""
        from src.data.pipeline import _validate_ohlcv

        df = pd.DataFrame([_daily_price_row(high=105.4, low=102.7, close=108.0)])
        assert len(_validate_ohlcv(df)) == 0

    def test_close_below_low_dropped(self):
        """close < low：OHLC 不一致，過濾。"""
        from src.data.pipeline import _validate_ohlcv

        df = pd.DataFrame([_daily_price_row(high=105.4, low=102.7, close=100.0)])
        assert len(_validate_ohlcv(df)) == 0

    def test_consistency_only_drops_offending_row(self):
        """僅過濾不一致的列，其餘保留。"""
        from src.data.pipeline import _validate_ohlcv

        df = pd.DataFrame(
            [
                _daily_price_row(stock_id="2330", high=610.0, low=595.0, close=605.0),
                _daily_price_row(stock_id="2317", high=105.4, low=102.7, close=108.0),
                _daily_price_row(stock_id="2454", high=105.4, low=102.7, close=105.4),
            ]
        )
        kept = _validate_ohlcv(df)
        assert sorted(kept["stock_id"]) == ["2330", "2454"]

    def test_missing_high_skips_close_high_check(self):
        """high 缺值時跳過 close<=high 邊界檢查（不誤刪）。"""
        from src.data.pipeline import _validate_ohlcv

        df = pd.DataFrame([_daily_price_row(high=None, low=102.7, close=108.0)])
        assert len(_validate_ohlcv(df)) == 1

    def test_high_less_than_low_dropped(self):
        """既有 high<low 檢查不受影響。"""
        from src.data.pipeline import _validate_ohlcv

        df = pd.DataFrame([_daily_price_row(high=100.0, low=105.0, close=102.0)])
        assert len(_validate_ohlcv(df)) == 0

    def test_nonpositive_close_dropped(self):
        """既有 close<=0 檢查不受影響。"""
        from src.data.pipeline import _validate_ohlcv

        df = pd.DataFrame([_daily_price_row(close=0.0)])
        assert len(_validate_ohlcv(df)) == 0


# ================================================================
# _detect_price_jumps（跳動哨兵）
# ================================================================


class TestDetectPriceJumps:
    def test_empty_df(self, db_session):
        from src.data.pipeline import _detect_price_jumps

        assert _detect_price_jumps(pd.DataFrame()) == 0

    def test_within_df_jump_flagged(self, db_session, caplog):
        """df 內同股相鄰兩日 > 門檻 → WARN 計數。"""
        import logging

        from src.data.pipeline import _detect_price_jumps

        df = pd.DataFrame(
            [
                _daily_price_row(stock_id="2330", date=date(2026, 5, 28), close=100.0),
                _daily_price_row(stock_id="2330", date=date(2026, 5, 29), close=120.0),  # +20%
            ]
        )
        with caplog.at_level(logging.WARNING):
            n = _detect_price_jumps(df)
        assert n == 1
        assert "跳動哨兵" in caplog.text

    def test_downward_jump_flagged(self, db_session):
        from src.data.pipeline import _detect_price_jumps

        df = pd.DataFrame(
            [
                _daily_price_row(stock_id="2330", date=date(2026, 5, 28), close=100.0),
                _daily_price_row(stock_id="2330", date=date(2026, 5, 29), close=85.0),  # -15%
            ]
        )
        assert _detect_price_jumps(df) == 1

    def test_normal_move_not_flagged(self, db_session):
        from src.data.pipeline import _detect_price_jumps

        df = pd.DataFrame(
            [
                _daily_price_row(stock_id="2330", date=date(2026, 5, 28), close=100.0),
                _daily_price_row(stock_id="2330", date=date(2026, 5, 29), close=105.0),  # +5%
            ]
        )
        assert _detect_price_jumps(df) == 0

    def test_0050_intraday_anomaly_not_flagged(self, db_session):
        """0050 5/29 +4.9%（±10% 內離群）不在單序列哨兵偵測範圍 —— 記錄已知限制。"""
        from src.data.pipeline import _detect_price_jumps

        df = pd.DataFrame(
            [
                _daily_price_row(stock_id="0050", date=date(2026, 5, 28), close=100.5),
                _daily_price_row(stock_id="0050", date=date(2026, 5, 29), close=105.4),  # +4.9%
            ]
        )
        assert _detect_price_jumps(df) == 0

    def test_db_prior_close_used(self, db_session):
        """單日 df（無 df 內前值）→ 回查 DB 前一交易日收盤。"""
        from src.data.pipeline import _detect_price_jumps

        db_session.add(DailyPrice(**_daily_price_row(stock_id="2317", date=date(2026, 5, 28), close=100.0)))
        db_session.flush()
        df = pd.DataFrame([_daily_price_row(stock_id="2317", date=date(2026, 5, 29), close=120.0)])  # +20%
        assert _detect_price_jumps(df) == 1

    def test_no_prior_close_skipped(self, db_session):
        """無任何前值（如新股首日）→ 不誤判。"""
        from src.data.pipeline import _detect_price_jumps

        df = pd.DataFrame([_daily_price_row(stock_id="9999", date=date(2026, 5, 29), close=120.0)])
        assert _detect_price_jumps(df) == 0

    def test_only_offending_rows_counted(self, db_session):
        """多股混合，僅計入超門檻列。"""
        from src.data.pipeline import _detect_price_jumps

        df = pd.DataFrame(
            [
                _daily_price_row(stock_id="2330", date=date(2026, 5, 28), close=100.0),
                _daily_price_row(stock_id="2330", date=date(2026, 5, 29), close=101.0),  # +1%
                _daily_price_row(stock_id="2317", date=date(2026, 5, 28), close=50.0),
                _daily_price_row(stock_id="2317", date=date(2026, 5, 29), close=65.0),  # +30%
            ]
        )
        assert _detect_price_jumps(df) == 1

    def test_custom_threshold(self, db_session):
        """threshold 可調：放寬至 25% 則 +20% 不再觸發。"""
        from src.data.pipeline import _detect_price_jumps

        df = pd.DataFrame(
            [
                _daily_price_row(stock_id="2330", date=date(2026, 5, 28), close=100.0),
                _daily_price_row(stock_id="2330", date=date(2026, 5, 29), close=120.0),  # +20%
            ]
        )
        assert _detect_price_jumps(df, threshold=0.25) == 0


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


# ================================================================
# sync_valuation_all_market
# ================================================================


class TestSyncValuationAllMarket:
    """測試 sync_valuation_all_market() — TWSE/TPEX 全市場估值同步。"""

    def test_skips_when_db_already_has_500_records(self, monkeypatch):
        """DB 已有 500+ 筆時不呼叫 fetch_market_valuation_all。"""
        import src.data.pipeline as pipeline_mod

        fetch_calls = []

        def fake_fetch(*a, **kw):
            fetch_calls.append(1)
            return pd.DataFrame()

        monkeypatch.setattr(
            "src.data.twse_fetcher.fetch_market_valuation_all",
            fake_fetch,
        )
        monkeypatch.setattr(pipeline_mod, "init_db", lambda: None)

        # 在 DB 中插入 501 筆 StockValuation 使 count >= 500
        # 使用 _find_last_trading_day() 與 pipeline 函數對齊日期（避免週末日期不符）
        from datetime import date as _date

        from src.data.database import get_session
        from src.data.schema import StockValuation
        from src.data.twse_fetcher import _find_last_trading_day

        target_date = _find_last_trading_day(_date.today())
        with get_session() as session:
            for i in range(501):
                session.execute(
                    __import__("sqlalchemy.dialects.sqlite", fromlist=["insert"])
                    .insert(StockValuation)
                    .values(stock_id=f"{1000 + i}", date=target_date, pe_ratio=15.0, pb_ratio=1.5, dividend_yield=3.0)
                    .on_conflict_do_nothing(index_elements=["stock_id", "date"])
                )
            session.commit()

        result = pipeline_mod.sync_valuation_all_market()
        assert result == 0
        assert len(fetch_calls) == 0, "已有 500+ 筆時不應呼叫 fetch"

    def test_calls_fetch_when_db_is_empty(self, monkeypatch):
        """DB 為空時應呼叫 fetch_market_valuation_all 並寫入。"""
        from datetime import date as _date

        import src.data.pipeline as pipeline_mod

        target = _date(2026, 3, 5)
        fake_df = pd.DataFrame(
            [
                {
                    "date": target,
                    "stock_id": f"{2000 + i}",
                    "pe_ratio": 15.0,
                    "pb_ratio": 1.5,
                    "dividend_yield": 3.5,
                }
                for i in range(10)
            ]
        )

        monkeypatch.setattr(
            "src.data.twse_fetcher.fetch_market_valuation_all",
            lambda *a, **kw: fake_df,
        )
        monkeypatch.setattr(pipeline_mod, "init_db", lambda: None)
        # 讓 _find_last_trading_day 直接回傳 target（跳過週末判斷）
        monkeypatch.setattr(
            "src.data.twse_fetcher._find_last_trading_day",
            lambda d: target,
        )

        result = pipeline_mod.sync_valuation_all_market()
        assert result == 10
