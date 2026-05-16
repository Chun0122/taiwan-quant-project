"""2026-05-15 sprint P1 — RotationPosition 滑價/成本三欄位實盤紀錄測試。

對應 5/29 audit 需要：實盤淨報酬 / dampen sensitivity / chip 硬阻擋 ROI 試算
皆需要 buy_slippage / sell_slippage / trade_cost 三欄位真實資料。
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select

from src.constants import COMMISSION_RATE, SLIPPAGE_RATE, TAX_RATE
from src.data.schema import RotationPortfolio, RotationPosition
from src.portfolio.manager import RotationManager


def _make_portfolio(db_session, name: str = "p_test", initial: float = 500_000.0) -> RotationPortfolio:
    p = RotationPortfolio(
        name=name,
        mode="swing",
        max_positions=3,
        holding_days=5,
        allow_renewal=False,
        initial_capital=initial,
        current_capital=initial,
        current_cash=initial,
        status="active",
    )
    db_session.add(p)
    db_session.flush()
    return p


def _make_trading_cal() -> list[date]:
    """簡單 trading_cal — 連續 30 個工作日。"""
    import pandas as pd

    return [d.date() for d in pd.bdate_range("2026-05-15", periods=30)]


# ====================================================================== #
# P1-A: schema 欄位存在 + nullable
# ====================================================================== #


class TestSchemaColumns:
    def test_position_can_be_created_without_slippage_fields(self, db_session):
        """新欄位 nullable — 既有測試/呼叫端不傳值仍能建立。"""
        p = _make_portfolio(db_session)
        pos = RotationPosition(
            portfolio_id=p.id,
            stock_id="2330",
            entry_date=date(2026, 5, 15),
            entry_price=600.0,
            entry_rank=1,
            holding_days_count=0,
            planned_exit_date=date(2026, 5, 22),
            shares=100,
            allocated_capital=60_000.0,
            status="open",
        )
        db_session.add(pos)
        db_session.flush()

        loaded = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        assert loaded.buy_slippage is None
        assert loaded.sell_slippage is None
        assert loaded.trade_cost is None

    def test_position_accepts_slippage_fields(self, db_session):
        p = _make_portfolio(db_session)
        pos = RotationPosition(
            portfolio_id=p.id,
            stock_id="2330",
            entry_date=date(2026, 5, 15),
            entry_price=600.0,
            entry_rank=1,
            holding_days_count=0,
            planned_exit_date=date(2026, 5, 22),
            shares=100,
            allocated_capital=60_000.0,
            status="open",
            buy_slippage=SLIPPAGE_RATE,
            sell_slippage=SLIPPAGE_RATE,
            trade_cost=123.45,
        )
        db_session.add(pos)
        db_session.flush()

        loaded = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        assert loaded.buy_slippage == pytest.approx(SLIPPAGE_RATE)
        assert loaded.sell_slippage == pytest.approx(SLIPPAGE_RATE)
        assert loaded.trade_cost == pytest.approx(123.45)


# ====================================================================== #
# P1-B: _execute_buy 寫入 buy_slippage + trade_cost
# ====================================================================== #


class TestExecuteBuyTracksSlippage:
    def test_buy_writes_slippage_and_trade_cost(self, db_session):
        p = _make_portfolio(db_session, initial=1_000_000.0)
        mgr = RotationManager("p_test")
        cal = _make_trading_cal()

        # 進場 100 股 @ 600
        buy = {"stock_id": "2330", "entry_price": 600.0, "shares": 100, "rank": 1, "score": 0.85}
        new_cash = mgr._execute_buy(db_session, p.id, buy, date(2026, 5, 15), cal, cash=1_000_000.0)
        db_session.flush()

        loaded = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        assert loaded.buy_slippage == pytest.approx(SLIPPAGE_RATE)
        # trade_cost 進場端 = price * shares * (COMMISSION_RATE + SLIPPAGE_RATE)
        expected = 600.0 * 100 * (COMMISSION_RATE + SLIPPAGE_RATE)
        assert loaded.trade_cost == pytest.approx(expected, abs=0.01)
        # sell 還未發生
        assert loaded.sell_slippage is None
        # cash 扣對
        assert new_cash == pytest.approx(1_000_000.0 - 600.0 * 100 - expected, abs=0.01)


# ====================================================================== #
# P1-C: _execute_sell 寫入 sell_slippage 並累加 trade_cost
# ====================================================================== #


class TestExecuteSellAccumulatesTradeCost:
    def test_sell_accumulates_trade_cost(self, db_session):
        p = _make_portfolio(db_session, initial=1_000_000.0)
        mgr = RotationManager("p_test")
        cal = _make_trading_cal()

        # 先買
        buy = {"stock_id": "2330", "entry_price": 600.0, "shares": 100, "rank": 1}
        mgr._execute_buy(db_session, p.id, buy, date(2026, 5, 15), cal, cash=1_000_000.0)
        db_session.flush()

        # 後賣 @ 650
        sell = {"stock_id": "2330", "exit_price": 650.0, "reason": "holding_expired", "days_held": 5}
        mgr._execute_sell(db_session, p.id, sell, date(2026, 5, 22), cash=0.0)
        db_session.flush()

        loaded = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        # 兩端滑價都記到
        assert loaded.sell_slippage == pytest.approx(SLIPPAGE_RATE)
        assert loaded.buy_slippage == pytest.approx(SLIPPAGE_RATE)
        # trade_cost = buy 端 + sell 端
        buy_cost = 600.0 * 100 * (COMMISSION_RATE + SLIPPAGE_RATE)
        sell_cost = 650.0 * 100 * (COMMISSION_RATE + TAX_RATE + SLIPPAGE_RATE)
        assert loaded.trade_cost == pytest.approx(round(buy_cost, 2) + round(sell_cost, 2), abs=0.05)
        # 狀態
        assert loaded.status == "closed"
        assert loaded.exit_price == pytest.approx(650.0)

    def test_sell_on_position_without_buy_trade_cost_initializes(self, db_session):
        """容錯：sell 端遇到 trade_cost=None 的舊資料，視為 0 起算。"""
        p = _make_portfolio(db_session, initial=500_000.0)
        # 手動建立一個沒有 trade_cost 的 open position（模擬 5/15 前的舊資料）
        pos = RotationPosition(
            portfolio_id=p.id,
            stock_id="2330",
            entry_date=date(2026, 5, 10),
            entry_price=600.0,
            entry_rank=1,
            holding_days_count=0,
            planned_exit_date=date(2026, 5, 17),
            shares=100,
            allocated_capital=60_000.0,
            status="open",
            buy_slippage=None,
            trade_cost=None,
        )
        db_session.add(pos)
        db_session.flush()

        mgr = RotationManager("p_test")
        sell = {"stock_id": "2330", "exit_price": 650.0, "reason": "holding_expired", "days_held": 5}
        mgr._execute_sell(db_session, p.id, sell, date(2026, 5, 17), cash=0.0)
        db_session.flush()

        loaded = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        # sell_slippage + trade_cost（只有 sell 端，因為 buy 端是 None=0）
        sell_cost = 650.0 * 100 * (COMMISSION_RATE + TAX_RATE + SLIPPAGE_RATE)
        assert loaded.sell_slippage == pytest.approx(SLIPPAGE_RATE)
        assert loaded.trade_cost == pytest.approx(round(sell_cost, 2), abs=0.05)


# ====================================================================== #
# P1-D: Migration 冪等性（schema 與 migrate 對齊）
# ====================================================================== #


class TestMigrationIdempotent:
    def test_rotation_position_migrations_listed(self):
        """3 個新欄位在 MIGRATIONS list 中。"""
        from src.data.migrate import MIGRATIONS

        cols = {(t, c) for t, c, _ in MIGRATIONS}
        assert ("rotation_position", "buy_slippage") in cols
        assert ("rotation_position", "sell_slippage") in cols
        assert ("rotation_position", "trade_cost") in cols
