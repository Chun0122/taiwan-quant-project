"""RotationManager.decide() 測試（A2-2，T+1 兩段式階段 A）。

涵蓋：
  DC-A  cold start decide → 3 筆 pending buy 落庫、不建 position、現金不動
  DC-B  dry_run 不落庫
  DC-C  同日冪等 guard + force 覆蓋
  DC-D  停損觸發 → pending sell（position 仍 open，明日才成交）
  DC-E  續持即時執行（planned_exit_date 更新、無 pending order）
  DC-F  熔斷即時清倉 + 既有 pending orders 全 cancelled（刻意不走 T+1）
  DC-G  decide 冪等重寫不累加 pending orders
  DC-H  decide 的 ActionLog 刪除範圍不動同日 fill 階段的 open/close 列
  DC-I  decide 寫入當日 snapshot（今日 close MtM）
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select

from src.data.schema import (
    DailyPrice,
    DiscoveryRecord,
    RotationActionLog,
    RotationDailySnapshot,
    RotationPendingOrder,
    RotationPortfolio,
    RotationPosition,
)
from src.portfolio.manager import RotationManager


@pytest.fixture()
def patch_session(db_session, monkeypatch):
    """manager.get_session 指向 in-memory session（__exit__ no-op，沿用既有模式）。"""

    class _Ctx:
        def __init__(self, s):
            self._s = s

        def __enter__(self):
            return self._s

        def __exit__(self, *a):
            return False

    from src.portfolio import manager as mgr_module

    monkeypatch.setattr(mgr_module, "get_session", lambda: _Ctx(db_session))
    return db_session


def _seed_universe(db_session, today: date) -> None:
    for i in range(-30, 31):
        d = today + timedelta(days=i)
        db_session.add(
            DailyPrice(stock_id="TAIEX", date=d, open=23000, high=23100, low=22900, close=23050, volume=0, turnover=0.0)
        )
    for sid in ("2330", "2317", "2454", "3008", "6669", "1111"):
        db_session.add(
            DailyPrice(
                stock_id=sid,
                date=today,
                open=100,
                high=102,
                low=99,
                close=100,
                volume=10_000_000,
                turnover=1_000_000_000.0,
            )
        )
    for i, sid in enumerate(("2330", "2317", "2454", "3008", "6669")):
        db_session.add(
            DiscoveryRecord(
                scan_date=today,
                mode="momentum",
                rank=i + 1,
                stock_id=sid,
                stock_name=f"name_{sid}",
                close=100,
                composite_score=0.9 - i * 0.05,
                technical_score=0.8,
                chip_score=0.7,
                fundamental_score=0.5,
                news_score=0.4,
                regime="bull",
                entry_price=100,
                stop_loss=95,
            )
        )
    db_session.commit()


def _make_portfolio(db_session, name="dc_test", initial=1_000_000.0) -> RotationPortfolio:
    p = RotationPortfolio(
        name=name,
        mode="momentum",
        max_positions=3,
        holding_days=10,
        allow_renewal=True,
        initial_capital=initial,
        current_capital=initial,
        current_cash=initial,
        status="active",
    )
    db_session.add(p)
    db_session.commit()
    return p


def _add_position(db_session, portfolio_id, sid, today, *, days_ago, stop_loss, name="held"):
    db_session.add(
        RotationPosition(
            portfolio_id=portfolio_id,
            stock_id=sid,
            stock_name=name,
            entry_date=today - timedelta(days=days_ago),
            entry_price=100,
            entry_rank=1,
            holding_days_count=days_ago,
            planned_exit_date=today - timedelta(days=days_ago) + timedelta(days=10),
            shares=1000,
            allocated_capital=100_000,
            stop_loss=stop_loss,
            status="open",
        )
    )
    db_session.commit()


def _pending_rows(session, name="dc_test"):
    return (
        session.execute(select(RotationPendingOrder).where(RotationPendingOrder.portfolio_name == name)).scalars().all()
    )


def _log_rows(session, name="dc_test"):
    return session.execute(select(RotationActionLog).where(RotationActionLog.portfolio_name == name)).scalars().all()


class TestDecideWritesPendingOrders:
    def test_cold_start_writes_pending_buys_no_positions(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        actions = RotationManager("dc_test").decide(today=today, regime="bull")

        assert actions is not None and len(actions.to_buy) == 3
        orders = _pending_rows(patch_session)
        assert len(orders) == 3
        assert all(o.side == "buy" and o.status == "pending" for o in orders)
        assert {o.stock_id for o in orders} == {"2330", "2317", "2454"}
        assert all(o.decision_date == today and o.exec_date is None for o in orders)
        assert all(o.ref_price == 100 for o in orders)
        assert all(o.allocated_capital and o.allocated_capital > 0 for o in orders)
        assert all(o.stop_loss == 95 for o in orders)
        # 不成交：無 position、現金不動
        assert patch_session.execute(select(RotationPosition)).scalars().all() == []
        p = patch_session.execute(select(RotationPortfolio)).scalar_one()
        assert p.current_cash == 1_000_000.0
        # ActionLog 寫 pending_buy
        logs = _log_rows(patch_session)
        assert [r.action_type for r in logs].count("pending_buy") == 3

    def test_dry_run_writes_nothing(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        actions = RotationManager("dc_test").decide(today=today, regime="bull", dry_run=True)

        assert actions is not None and len(actions.to_buy) == 3
        assert _pending_rows(patch_session) == []
        assert _log_rows(patch_session) == []
        assert patch_session.execute(select(RotationDailySnapshot)).scalars().all() == []

    def test_same_day_guard_and_force(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        mgr = RotationManager("dc_test")
        assert mgr.decide(today=today, regime="bull") is not None
        assert mgr.decide(today=today, regime="bull") is None, "同日第二次 decide 應被 guard 跳過"
        assert mgr.decide(today=today, regime="bull", force=True) is not None, "force 應繞過 guard"
        # force 重跑後 pending orders 收斂不累加
        assert len(_pending_rows(patch_session)) == 3

    def test_stop_loss_becomes_pending_sell_position_stays_open(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        p = _make_portfolio(patch_session)
        # 1111 不在 rankings；stop_loss=150 > close 100 → 觸發停損
        _add_position(patch_session, p.id, "1111", today, days_ago=3, stop_loss=150)

        RotationManager("dc_test").decide(today=today, regime="bull")

        sells = [o for o in _pending_rows(patch_session) if o.side == "sell"]
        assert len(sells) == 1
        s = sells[0]
        assert s.stock_id == "1111"
        assert s.reason == "stop_loss"
        assert s.shares == 1000
        assert s.ref_price == 100  # 決策日 close
        # 明日才成交：position 今晚仍 open
        pos = patch_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "1111")).scalar_one()
        assert pos.status == "open"
        # ActionLog 記 pending_sell 且標風控
        logs = [r for r in _log_rows(patch_session) if r.action_type == "pending_sell"]
        assert len(logs) == 1 and logs[0].is_risk_exit is True

    def test_renewal_executes_immediately_no_pending(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        p = _make_portfolio(patch_session)
        # 2330 在 rankings rank 1 且已過持有期 → 續持（即時，不走 pending）
        _add_position(patch_session, p.id, "2330", today, days_ago=15, stop_loss=50)
        old_exit = (
            patch_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330"))
            .scalar_one()
            .planned_exit_date
        )

        actions = RotationManager("dc_test").decide(today=today, regime="bull")

        assert any(r["stock_id"] == "2330" for r in actions.renewed)
        pos = patch_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        assert pos.planned_exit_date > old_exit, "續持應即時延展 planned_exit_date"
        assert all(o.stock_id != "2330" for o in _pending_rows(patch_session)), "renew 不應產生 pending order"

    def test_drawdown_liquidation_immediate_and_cancels_pending(self, patch_session, monkeypatch):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "1111", today, days_ago=3, stop_loss=50)
        # 昨日殘留一筆 pending 買單 → 清算時應被 cancel
        patch_session.add(
            RotationPendingOrder(
                portfolio_name="dc_test",
                decision_date=today - timedelta(days=1),
                side="buy",
                stock_id="2330",
                shares=1000,
                ref_price=100.0,
                status="pending",
            )
        )
        patch_session.commit()

        monkeypatch.setattr("src.portfolio.manager.compute_drawdown_with_snapshots", lambda *a, **kw: 30.0)

        actions = RotationManager("dc_test").decide(today=today, regime="bull")

        assert actions is not None and len(actions.to_sell) == 1
        # 熔斷即時：position 當日 closed（刻意不走 T+1）
        pos = patch_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "1111")).scalar_one()
        assert pos.status == "closed"
        assert pos.exit_reason == "max_drawdown_liquidation"
        assert patch_session.execute(select(RotationPortfolio)).scalar_one().status == "liquidated"
        # 殘留 pending 買單全部作廢
        leftover = patch_session.execute(
            select(RotationPendingOrder).where(RotationPendingOrder.stock_id == "2330")
        ).scalar_one()
        assert leftover.status == "cancelled"

    def test_decide_preserves_same_day_fill_rows(self, patch_session):
        """D 日早上 fill 寫的 open/close 不可被晚上 decide 的冪等刪除掃掉。"""
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)
        # 模擬 fill_pending(today) 已寫入的成交紀錄
        patch_session.add(
            RotationActionLog(
                portfolio_name="dc_test", action_date=today, action_type="open", stock_id="9999", stock_name="filled"
            )
        )
        patch_session.commit()

        RotationManager("dc_test").decide(today=today, regime="bull", force=True)

        logs = _log_rows(patch_session)
        assert any(r.action_type == "open" and r.stock_id == "9999" for r in logs), "fill 階段列應被保留"
        assert [r.action_type for r in logs].count("pending_buy") == 3

    def test_decide_writes_snapshot_with_mtm(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "1111", today, days_ago=3, stop_loss=150)  # 停損倉今晚仍持有

        RotationManager("dc_test").decide(today=today, regime="bull")

        snap = patch_session.execute(select(RotationDailySnapshot)).scalar_one()
        assert snap.snapshot_date == today
        # 停損倉明日才出：今日 MtM 仍含該倉（1000 股 × close 100）
        assert snap.total_market_value == 100_000.0
        assert snap.n_holdings == 1
        assert snap.total_cash == 1_000_000.0
