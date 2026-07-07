"""RotationManager.fill_pending() 測試（A2-3，T+1 兩段式階段 B）。

涵蓋：
  FP-A  E2E：decide(D) → fill_pending(D+1) 以 open[D+1] 成交、order 標 filled
  FP-B  賣單以 open[D+1] 成交、position 關閉、現金回收
  FP-C  賣先買後（先回收現金再買）
  FP-D  停牌賣單 fallback ref_price 成交（風控賣單不凍結）
  FP-E  停牌買單 TTL 內順延、逾期取消
  FP-F  open 跳空 → 股數以 allocated_capital 對 open 重算（縮股）
  FP-G  冪等：重跑 no-op（status 轉移天然冪等）
  FP-H  dry_run 不落庫
  FP-I  守恆：cash_before + Σ賣出淨回收 − Σ買入總支出 = cash_after
        （以 execution_core 用落庫欄位重算比對，容忍浮點 epsilon）
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select

from src.data.schema import (
    DailyPrice,
    DiscoveryRecord,
    RotationActionLog,
    RotationPendingOrder,
    RotationPortfolio,
    RotationPosition,
)
from src.portfolio.execution_core import simulate_buy, simulate_sell
from src.portfolio.manager import RotationManager

D = date(2026, 6, 1)
D1 = date(2026, 6, 2)


@pytest.fixture()
def patch_session(db_session, monkeypatch):
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


def _seed(db_session, *, d1_open: float = 105.0, skip_d1_sids: set[str] | None = None) -> None:
    """TAIEX 行事曆 + 個股 D 與 D+1 兩天報價 + momentum DiscoveryRecord。"""
    skip_d1_sids = skip_d1_sids or set()
    for i in range(-30, 31):
        d = D + timedelta(days=i)
        db_session.add(
            DailyPrice(stock_id="TAIEX", date=d, open=23000, high=23100, low=22900, close=23050, volume=0, turnover=0.0)
        )
    for sid in ("2330", "2317", "2454", "1111"):
        db_session.add(
            DailyPrice(stock_id=sid, date=D, open=100, high=102, low=99, close=100, volume=10_000_000, turnover=1e9)
        )
        if sid not in skip_d1_sids:
            db_session.add(
                DailyPrice(
                    stock_id=sid,
                    date=D1,
                    open=d1_open,
                    high=d1_open + 2,
                    low=d1_open - 2,
                    close=d1_open + 1,
                    volume=10_000_000,
                    turnover=1e9,
                )
            )
    for i, sid in enumerate(("2330", "2317", "2454")):
        db_session.add(
            DiscoveryRecord(
                scan_date=D,
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


def _make_portfolio(db_session, name="fp_test", initial=1_000_000.0) -> RotationPortfolio:
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


def _add_sell_order(db_session, name="fp_test", sid="1111", ref_price=100.0, reason="stop_loss", days_held=3):
    db_session.add(
        RotationPendingOrder(
            portfolio_name=name,
            decision_date=D,
            side="sell",
            stock_id=sid,
            stock_name="held",
            shares=1000,
            ref_price=ref_price,
            days_held=days_held,
            reason=reason,
            status="pending",
        )
    )
    db_session.commit()


def _add_position(db_session, portfolio_id, sid="1111", shares=1000, buy_slippage=0.0005):
    db_session.add(
        RotationPosition(
            portfolio_id=portfolio_id,
            stock_id=sid,
            stock_name="held",
            entry_date=D - timedelta(days=3),
            entry_price=100,
            entry_rank=1,
            holding_days_count=3,
            planned_exit_date=D + timedelta(days=7),
            shares=shares,
            allocated_capital=100_000,
            stop_loss=90,
            status="open",
            buy_slippage=buy_slippage,
        )
    )
    db_session.commit()


def _orders(session, name="fp_test"):
    return (
        session.execute(select(RotationPendingOrder).where(RotationPendingOrder.portfolio_name == name)).scalars().all()
    )


class TestFillPendingBuys:
    def test_e2e_decide_then_fill_at_next_open(self, patch_session):
        _seed(patch_session, d1_open=105.0)
        _make_portfolio(patch_session)
        mgr = RotationManager("fp_test")

        mgr.decide(today=D, regime="bull")
        assert patch_session.execute(select(RotationPosition)).scalars().all() == []

        result = mgr.fill_pending(exec_day=D1)

        assert result == {"filled": 3, "cancelled": 0, "deferred": 0, "dividends": 0}
        positions = patch_session.execute(select(RotationPosition)).scalars().all()
        assert len(positions) == 3
        assert all(p.entry_price == 105.0 for p in positions), "成交價應為 open[D+1] 而非決策日 close"
        assert all(p.entry_date == D1 for p in positions)
        for o in _orders(patch_session):
            assert o.status == "filled"
            assert o.exec_date == D1
        p = patch_session.execute(select(RotationPortfolio)).scalar_one()
        assert p.current_cash < 1_000_000.0
        # fill 階段 ActionLog：open 列落在 D+1
        opens = (
            patch_session.execute(
                select(RotationActionLog).where(
                    RotationActionLog.action_date == D1, RotationActionLog.action_type == "open"
                )
            )
            .scalars()
            .all()
        )
        assert len(opens) == 3
        assert all(r.price == 105.0 for r in opens)

    def test_gap_up_reshrinks_shares_from_alloc(self, patch_session):
        """open 大幅高於決策 close → 股數 = alloc/open 重算，比決策股數少。"""
        _seed(patch_session, d1_open=130.0)
        _make_portfolio(patch_session)
        mgr = RotationManager("fp_test")
        mgr.decide(today=D, regime="bull")
        planned = {o.stock_id: o.shares for o in _orders(patch_session)}

        mgr.fill_pending(exec_day=D1)

        for pos in patch_session.execute(select(RotationPosition)).scalars().all():
            assert pos.shares < planned[pos.stock_id], "跳空 30% 後股數應縮"
            assert pos.entry_price == 130.0

    def test_suspended_buy_defers_within_ttl_then_cancels(self, patch_session):
        _seed(patch_session, skip_d1_sids={"2330", "2317", "2454"})  # D+1 起全部無報價
        _make_portfolio(patch_session)
        mgr = RotationManager("fp_test")
        mgr.decide(today=D, regime="bull")

        # D+1：TTL 內（1 個交易日）→ 順延
        result = mgr.fill_pending(exec_day=D1)
        assert result == {"filled": 0, "cancelled": 0, "deferred": 3, "dividends": 0}
        assert all(o.status == "pending" for o in _orders(patch_session))

        # D+4：已逾 2 個交易日 → 取消（行事曆每日皆交易日）
        result = mgr.fill_pending(exec_day=D + timedelta(days=4))
        assert result == {"filled": 0, "cancelled": 3, "deferred": 0, "dividends": 0}
        assert all(o.status == "cancelled" for o in _orders(patch_session))
        assert patch_session.execute(select(RotationPosition)).scalars().all() == []

    def test_rerun_is_noop(self, patch_session):
        _seed(patch_session)
        _make_portfolio(patch_session)
        mgr = RotationManager("fp_test")
        mgr.decide(today=D, regime="bull")
        mgr.fill_pending(exec_day=D1)
        cash_after = patch_session.execute(select(RotationPortfolio)).scalar_one().current_cash

        result = mgr.fill_pending(exec_day=D1)

        assert result == {"filled": 0, "cancelled": 0, "deferred": 0, "dividends": 0}
        assert patch_session.execute(select(RotationPortfolio)).scalar_one().current_cash == cash_after
        assert len(patch_session.execute(select(RotationPosition)).scalars().all()) == 3

    def test_dry_run_writes_nothing(self, patch_session):
        _seed(patch_session)
        _make_portfolio(patch_session)
        mgr = RotationManager("fp_test")
        mgr.decide(today=D, regime="bull")

        result = mgr.fill_pending(exec_day=D1, dry_run=True)

        assert result == {"filled": 3, "cancelled": 0, "deferred": 0, "dividends": 0}
        assert all(o.status == "pending" for o in _orders(patch_session))
        assert patch_session.execute(select(RotationPosition)).scalars().all() == []
        assert patch_session.execute(select(RotationPortfolio)).scalar_one().current_cash == 1_000_000.0


class TestFillPendingSells:
    def test_sell_fills_at_open_and_closes_position(self, patch_session):
        _seed(patch_session, d1_open=105.0)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id)
        _add_sell_order(patch_session)

        result = RotationManager("fp_test").fill_pending(exec_day=D1)

        assert result["filled"] == 1
        pos = patch_session.execute(select(RotationPosition)).scalar_one()
        assert pos.status == "closed"
        assert pos.exit_price == 105.0
        assert pos.exit_date == D1
        assert pos.exit_reason == "stop_loss"
        assert pos.holding_days_count == 3  # order.days_held 寫回
        pf = patch_session.execute(select(RotationPortfolio)).scalar_one()
        assert pf.current_cash > 1_000_000.0  # 回收現金

    def test_suspended_sell_falls_back_to_ref_price(self, patch_session):
        """停牌無報價 → 以決策日 close（ref_price）成交，風控賣單不凍結。"""
        _seed(patch_session, skip_d1_sids={"1111"})
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id)
        _add_sell_order(patch_session, ref_price=100.0)

        result = RotationManager("fp_test").fill_pending(exec_day=D1)

        assert result["filled"] == 1
        pos = patch_session.execute(select(RotationPosition)).scalar_one()
        assert pos.status == "closed"
        assert pos.exit_price == 100.0

    def test_sell_before_buy_funds_purchase(self, patch_session):
        """現金近零時：先賣（回收）再買——買單靠賣出回收的現金成交。"""
        _seed(patch_session, d1_open=100.0)
        p = _make_portfolio(patch_session, initial=1_000_000.0)
        p.current_cash = 1_000.0  # 幾乎滿倉
        patch_session.commit()
        _add_position(patch_session, p.id, sid="1111", shares=3000)
        _add_sell_order(patch_session, reason="holding_expired")
        # 手動加一筆買單（alloc 10 萬 > 現金 1 千，須靠賣出回收）
        patch_session.add(
            RotationPendingOrder(
                portfolio_name="fp_test",
                decision_date=D,
                side="buy",
                stock_id="2330",
                stock_name="tsmc",
                shares=1000,
                ref_price=100.0,
                allocated_capital=100_000.0,
                entry_rank=1,
                entry_score=0.9,
                stop_loss=95.0,
                status="pending",
            )
        )
        patch_session.commit()

        result = RotationManager("fp_test").fill_pending(exec_day=D1)

        assert result["filled"] == 2
        buy_pos = patch_session.execute(
            select(RotationPosition).where(RotationPosition.stock_id == "2330")
        ).scalar_one()
        assert buy_pos.status == "open" and buy_pos.shares > 0


class TestConservation:
    def test_cash_conservation_recomputed_from_db(self, patch_session):
        """守恆：cash_after = cash_before + Σ賣出淨回收 − Σ買入總支出。

        以落庫欄位（entry/exit price、shares、buy/sell slippage）用 execution_core
        重算金額，比對 portfolio.current_cash（同一份算式 → epsilon 內一致）。
        """
        _seed(patch_session, d1_open=104.0)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, sid="1111", shares=2000)
        _add_sell_order(patch_session, reason="holding_expired")
        mgr = RotationManager("fp_test")
        mgr.decide(today=D, regime="bull", force=True)
        cash_before = patch_session.execute(select(RotationPortfolio)).scalar_one().current_cash

        mgr.fill_pending(exec_day=D1)

        expected = cash_before
        for pos in patch_session.execute(select(RotationPosition)).scalars().all():
            if pos.status == "closed":
                fill = simulate_sell(
                    pos.entry_price,
                    pos.exit_price,
                    pos.shares,
                    buy_slippage=pos.buy_slippage or 0.0005,
                    sell_slippage=pos.sell_slippage,
                )
                expected += fill.proceeds
            elif pos.entry_date == D1:
                fill = simulate_buy(pos.entry_price, pos.shares, pos.buy_slippage)
                expected -= fill.buy_cost
        actual = patch_session.execute(select(RotationPortfolio)).scalar_one().current_cash
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_status_state_machine_no_other_transitions(self, patch_session):
        """全部 order 結局 ∈ {filled, cancelled, pending}；filled 必有 exec_date。"""
        _seed(patch_session, skip_d1_sids={"2454"})
        _make_portfolio(patch_session)
        mgr = RotationManager("fp_test")
        mgr.decide(today=D, regime="bull")
        mgr.fill_pending(exec_day=D1)

        for o in _orders(patch_session):
            assert o.status in ("filled", "cancelled", "pending")
            if o.status == "filled":
                assert o.exec_date == D1
            else:
                assert o.exec_date is None
