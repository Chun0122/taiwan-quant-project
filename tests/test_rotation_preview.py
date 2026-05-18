"""P2 任務 9 — rotation preview (dry_run) 測試。

涵蓋：
  P9-A update(dry_run=True) 回傳正確 RotationActions
  P9-B update(dry_run=True) 不寫入 RotationPosition 新列
  P9-C update(dry_run=True) 不寫入 RotationDailySnapshot
  P9-D update(dry_run=True) 不變更 portfolio.current_cash / current_capital
  P9-E update(dry_run=True) 回傳的 actions 與 dry_run=False 一致
  P9-F drawdown 觸發時 dry_run=True 不變更 status='liquidated'
  P9-G CLI _print_rotation_preview 輸出
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select

from src.cli.rotation_cmd import _print_rotation_preview
from src.data.schema import (
    DailyPrice,
    DiscoveryRecord,
    RotationDailySnapshot,
    RotationPortfolio,
    RotationPosition,
)
from src.portfolio.manager import RotationManager
from src.portfolio.rotation import RotationActions


@pytest.fixture()
def patch_session(db_session, monkeypatch):
    """Monkeypatch get_session to return the in-memory db_session."""
    from src.portfolio import manager as mgr_module

    class _Ctx:
        def __init__(self, s):
            self._s = s

        def __enter__(self):
            return self._s

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(mgr_module, "get_session", lambda: _Ctx(db_session))
    return db_session


def _seed_universe(db_session, today: date) -> None:
    """注入 TAIEX 行事曆 + 5 個股 DailyPrice + 5 筆 momentum DiscoveryRecord。"""
    # TAIEX 行事曆（過去 30 天 + 未來 30 天）
    for i in range(-30, 31):
        d = today + timedelta(days=i)
        db_session.add(
            DailyPrice(stock_id="TAIEX", date=d, open=23000, high=23100, low=22900, close=23050, volume=0, turnover=0.0)
        )
    # 5 個股各注入 today close
    for sid in ("2330", "2317", "2454", "3008", "6669"):
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
    # 5 筆 DiscoveryRecord
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


def _make_portfolio(db_session, name="prev_test", initial=1_000_000.0) -> RotationPortfolio:
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


# ====================================================================== #
# P9-A/B/C/D: dry_run=True 不寫入任何 DB
# ====================================================================== #


class TestDryRunNoSideEffects:
    def test_dry_run_returns_rotation_actions(self, patch_session):
        today = date(2026, 5, 17)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        mgr = RotationManager("prev_test")
        actions = mgr.update(today=today, regime="bull", dry_run=True)
        assert isinstance(actions, RotationActions)
        # cold start：應該預備買入 top 3
        assert len(actions.to_buy) == 3
        assert [b["stock_id"] for b in actions.to_buy] == ["2330", "2317", "2454"]

    def test_dry_run_does_not_create_positions(self, patch_session):
        today = date(2026, 5, 17)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        mgr = RotationManager("prev_test")
        mgr.update(today=today, regime="bull", dry_run=True)

        positions = patch_session.execute(select(RotationPosition)).scalars().all()
        assert positions == [], f"expected no positions; got {len(positions)}"

    def test_dry_run_does_not_create_snapshot(self, patch_session):
        today = date(2026, 5, 17)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        mgr = RotationManager("prev_test")
        mgr.update(today=today, regime="bull", dry_run=True)

        snaps = patch_session.execute(select(RotationDailySnapshot)).scalars().all()
        assert snaps == [], f"expected no snapshots; got {len(snaps)}"

    def test_dry_run_does_not_mutate_portfolio_cash(self, patch_session):
        today = date(2026, 5, 17)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        mgr = RotationManager("prev_test")
        mgr.update(today=today, regime="bull", dry_run=True)

        p = patch_session.execute(select(RotationPortfolio)).scalar_one()
        # current_cash 與 current_capital 應保持初始值
        assert p.current_cash == pytest.approx(1_000_000.0)
        assert p.current_capital == pytest.approx(1_000_000.0)


# ====================================================================== #
# P9-E: dry_run actions 與實際執行的 actions 一致
# ====================================================================== #


class TestDryRunMatchesReal:
    def test_actions_identical_after_rollback(self, patch_session):
        """dry_run 與 dry_run=False 應計算出相同的 to_buy（同一輸入）。"""
        today = date(2026, 5, 17)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        mgr = RotationManager("prev_test")
        preview_actions = mgr.update(today=today, regime="bull", dry_run=True)

        # 確認 preview 之後 DB 真的乾淨（即使 _execute_buy 已被呼叫但被 rollback）
        positions_after_preview = patch_session.execute(select(RotationPosition)).scalars().all()
        assert positions_after_preview == []

        # 第二次 update（實際執行）— 應產生相同的 to_buy
        actual_actions = mgr.update(today=today, regime="bull", dry_run=False)
        assert [b["stock_id"] for b in preview_actions.to_buy] == [b["stock_id"] for b in actual_actions.to_buy]


# ====================================================================== #
# P9-F: drawdown kill switch dry_run 不變更 status
# ====================================================================== #


class TestDryRunSkipsLiquidation:
    def test_drawdown_trigger_does_not_set_liquidated_in_dry_run(self, patch_session, monkeypatch):
        today = date(2026, 5, 17)
        _seed_universe(patch_session, today)
        portfolio = _make_portfolio(patch_session)

        # 注入持倉（讓 _load_open_positions 有東西）
        patch_session.add(
            RotationPosition(
                portfolio_id=portfolio.id,
                stock_id="2330",
                stock_name="台積電",
                entry_date=today - timedelta(days=5),
                entry_price=600,
                entry_rank=1,
                holding_days_count=5,
                planned_exit_date=today + timedelta(days=5),
                shares=1000,
                allocated_capital=600_000,
                stop_loss=550,
                status="open",
            )
        )
        patch_session.commit()

        # 強制 drawdown kill switch 觸發
        monkeypatch.setattr("src.portfolio.manager.check_drawdown_kill_switch", lambda *a, **kw: True)
        monkeypatch.setattr("src.portfolio.manager.compute_portfolio_drawdown", lambda *a, **kw: 30.0)

        mgr = RotationManager("prev_test")
        actions = mgr.update(today=today, regime="bull", dry_run=True)

        # dry_run 應回傳 to_sell 但不變更 status
        assert isinstance(actions, RotationActions)
        assert len(actions.to_sell) == 1

        p = patch_session.execute(select(RotationPortfolio)).scalar_one()
        assert p.status == "active"  # 未被改成 "liquidated"
        assert p.current_cash == pytest.approx(1_000_000.0)  # 未被改成全現金


# ====================================================================== #
# P9-G: CLI formatter
# ====================================================================== #


class TestPrintRotationPreview:
    def test_prints_will_buy_section(self, capsys):
        actions = RotationActions(
            to_buy=[
                {"stock_id": "2330", "rank": 1, "entry_price": 600, "shares": 100, "allocated_capital": 60_000},
            ],
        )
        _print_rotation_preview("test_pf", actions, target_date=date(2026, 5, 17))
        out = capsys.readouterr().out
        assert "Pre-Trade 預覽" in out
        assert "2330" in out
        assert "rank#1" in out
        assert "將買入" in out
        assert "DRY RUN" in out

    def test_prints_will_sell_section(self, capsys):
        actions = RotationActions(
            to_sell=[
                {"stock_id": "2330", "reason": "holding_expired", "exit_price": 650.0},
            ],
        )
        _print_rotation_preview("test_pf", actions, target_date=None)
        out = capsys.readouterr().out
        assert "將賣出" in out
        assert "holding_expired" in out
        assert "@650" in out

    def test_prints_no_actions_message_when_empty(self, capsys):
        _print_rotation_preview("test_pf", RotationActions(), target_date=date(2026, 5, 17))
        out = capsys.readouterr().out
        assert "（無動作）" in out
