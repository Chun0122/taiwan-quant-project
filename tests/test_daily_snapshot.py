"""tests/test_daily_snapshot.py — RotationDailySnapshot 寫入與讀取測試。

涵蓋：
- _write_daily_snapshot() 首日無前日 daily_return_pct 為 None
- 第二天計算正確的 daily_return_pct
- 同日重複呼叫採 update（不違反 UniqueConstraint）
- get_recent_snapshots() 限制筆數 + asc by date
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select

from src.data.schema import RotationDailySnapshot, RotationPortfolio
from src.portfolio.manager import RotationManager


@pytest.fixture()
def portfolio_with_snapshot_setup(db_session):
    """建立一個 active portfolio，回傳 (portfolio, manager)。"""
    p = RotationPortfolio(
        name="snap_test",
        mode="momentum",
        max_positions=5,
        holding_days=10,
        allow_renewal=True,
        initial_capital=1_000_000.0,
        current_capital=1_000_000.0,
        current_cash=1_000_000.0,
        status="active",
    )
    db_session.add(p)
    db_session.commit()
    return p, RotationManager("snap_test")


class TestWriteDailySnapshot:
    def test_first_day_daily_return_is_none(self, db_session, portfolio_with_snapshot_setup):
        portfolio, mgr = portfolio_with_snapshot_setup

        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 1),
            market_value=0.0,
            n_holdings=0,
            regime="bull",
        )

        rows = db_session.execute(select(RotationDailySnapshot)).scalars().all()
        assert len(rows) == 1
        assert rows[0].daily_return_pct is None
        assert rows[0].total_capital == 1_000_000.0
        assert rows[0].n_holdings == 0
        assert rows[0].regime_state == "bull"

    def test_second_day_daily_return_computed(self, db_session, portfolio_with_snapshot_setup):
        portfolio, mgr = portfolio_with_snapshot_setup
        # day 1
        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 1),
            market_value=0.0,
            n_holdings=0,
            regime="bull",
        )
        # day 2: capital +1%
        portfolio.current_capital = 1_010_000.0
        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 2),
            market_value=500_000.0,
            n_holdings=2,
            regime="bull",
        )

        rows = (
            db_session.execute(select(RotationDailySnapshot).order_by(RotationDailySnapshot.snapshot_date))
            .scalars()
            .all()
        )
        assert len(rows) == 2
        assert rows[0].daily_return_pct is None
        assert rows[1].daily_return_pct == pytest.approx(0.01, abs=1e-6)
        assert rows[1].n_holdings == 2

    def test_same_day_double_write_updates(self, db_session, portfolio_with_snapshot_setup):
        """同日重複呼叫應 update 既有列，不違反 UniqueConstraint。"""
        portfolio, mgr = portfolio_with_snapshot_setup
        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 1),
            market_value=0.0,
            n_holdings=0,
            regime="bull",
        )
        portfolio.current_capital = 1_020_000.0
        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 1),  # 同日
            market_value=300_000.0,
            n_holdings=1,
            regime="bear",  # regime 改變
        )

        rows = db_session.execute(select(RotationDailySnapshot)).scalars().all()
        assert len(rows) == 1  # 仍只有一筆
        assert rows[0].total_capital == 1_020_000.0
        assert rows[0].n_holdings == 1
        assert rows[0].regime_state == "bear"


class TestGetRecentSnapshots:
    def test_returns_asc_by_date_with_limit(self, db_session, portfolio_with_snapshot_setup):
        portfolio, mgr = portfolio_with_snapshot_setup
        # 注入 20 個 snapshot
        for i in range(20):
            d = date(2026, 4, 1) + timedelta(days=i)
            db_session.add(
                RotationDailySnapshot(
                    portfolio_name="snap_test",
                    snapshot_date=d,
                    total_capital=1_000_000.0 + i * 1000,
                    total_market_value=0.0,
                    total_cash=1_000_000.0 + i * 1000,
                    unrealized_pnl=i * 1000,
                    daily_return_pct=None if i == 0 else 0.001,
                    n_holdings=0,
                    regime_state="bull",
                )
            )
        db_session.commit()

        snaps = mgr.get_recent_snapshots(n_days=10)
        assert len(snaps) == 10
        # asc by snapshot_date
        assert snaps[0]["snapshot_date"] < snaps[-1]["snapshot_date"]
        # 取最近 10 筆 → 最後一筆是 day20 (i=19) → 4/20
        assert snaps[-1]["snapshot_date"] == date(2026, 4, 20)
        assert snaps[0]["snapshot_date"] == date(2026, 4, 11)

    def test_returns_empty_when_no_snapshots(self, db_session, portfolio_with_snapshot_setup):
        _, mgr = portfolio_with_snapshot_setup
        assert mgr.get_recent_snapshots(n_days=30) == []

    def test_isolates_by_portfolio_name(self, db_session, portfolio_with_snapshot_setup):
        """改名後不會撈到舊組合 snapshot。"""
        _, _ = portfolio_with_snapshot_setup
        # 為另一個組合 "other" 注入資料
        db_session.add(
            RotationDailySnapshot(
                portfolio_name="other",
                snapshot_date=date(2026, 5, 1),
                total_capital=999_999.0,
                total_market_value=0.0,
                total_cash=999_999.0,
                unrealized_pnl=0.0,
                n_holdings=0,
            )
        )
        db_session.commit()

        snaps = RotationManager("snap_test").get_recent_snapshots(n_days=10)
        assert snaps == []
