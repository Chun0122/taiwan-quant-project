"""RotationActionLog 落庫 + export today_actions 測試（v4 — 「今天各 Rotation 做了什麼」）。

涵蓋：
  AL-A  update() cold start → 3 筆 open 落庫
  AL-B  dry_run=True 不寫 action log
  AL-C  同日重跑冪等（delete-then-insert，不重複）
  AL-D  stop_loss 平倉 → action_type=close, is_risk_exit=True
  AL-E  同日非風控賣出 + 買入 → switch_group 標記（換股）
  AL-F  drawdown 熔斷 → max_drawdown_liquidation 落庫且 is_risk_exit=True
  AL-G  _serialize_action_log_row 純序列化欄位
  AL-H  _build_today_actions 依日期撈 + 排序（open→close→renew→hold）
  AL-I  _build_rotations 帶出 today_actions
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select

from src.cli import export_dashboard_cmd as ed
from src.data.schema import (
    DailyPrice,
    DiscoveryRecord,
    RotationActionLog,
    RotationPortfolio,
    RotationPosition,
)
from src.portfolio.manager import RotationManager


@pytest.fixture()
def patch_session(db_session, monkeypatch):
    """把 manager 與 export_dashboard_cmd 的 get_session 都指向同一個 in-memory session。

    __exit__ 為 no-op，避免關閉共享 session（沿用 test_rotation_preview 模式）。
    """

    class _Ctx:
        def __init__(self, s):
            self._s = s

        def __enter__(self):
            return self._s

        def __exit__(self, *a):
            return False

    from src.portfolio import manager as mgr_module

    monkeypatch.setattr(mgr_module, "get_session", lambda: _Ctx(db_session))
    monkeypatch.setattr(ed, "get_session", lambda: _Ctx(db_session))
    return db_session


def _seed_universe(db_session, today: date) -> None:
    """TAIEX 行事曆 + 6 個股 today close + 5 筆 momentum DiscoveryRecord（rank 1~5）。"""
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


def _make_portfolio(db_session, name="al_test", initial=1_000_000.0) -> RotationPortfolio:
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


def _log_rows(session, name="al_test"):
    return session.execute(select(RotationActionLog).where(RotationActionLog.portfolio_name == name)).scalars().all()


# ====================================================================== #
# update() 落庫
# ====================================================================== #


class TestUpdateWritesActionLog:
    def test_cold_start_writes_three_opens(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        RotationManager("al_test").update(today=today, regime="bull")

        rows = _log_rows(patch_session)
        opens = [r for r in rows if r.action_type == "open"]
        assert len(opens) == 3
        assert {r.stock_id for r in opens} == {"2330", "2317", "2454"}
        assert all(r.action_date == today for r in opens)
        assert all(r.reason is None and r.is_risk_exit is False for r in opens)
        # cold start 無賣出 → 非換股日，switch_group 全為 None
        assert all(r.switch_group is None for r in opens)

    def test_dry_run_writes_nothing(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        RotationManager("al_test").update(today=today, regime="bull", dry_run=True)

        assert _log_rows(patch_session) == []

    def test_rerun_same_day_is_idempotent(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        mgr = RotationManager("al_test")
        mgr.update(today=today, regime="bull")
        first = len(_log_rows(patch_session))
        mgr.update(today=today, regime="bull")
        second = len(_log_rows(patch_session))

        assert first == second == 3, "同日重跑應覆寫而非累加"

    def test_stop_loss_marks_risk_exit(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        p = _make_portfolio(patch_session)
        # 1111 不在 rankings；stop_loss=150 > today close 100 → 觸發停損
        _add_position(patch_session, p.id, "1111", today, days_ago=3, stop_loss=150)

        RotationManager("al_test").update(today=today, regime="bull")

        rows = _log_rows(patch_session)
        closes = [r for r in rows if r.action_type == "close"]
        assert len(closes) == 1
        c = closes[0]
        assert c.stock_id == "1111"
        assert c.reason == "stop_loss"
        assert c.is_risk_exit is True
        assert c.switch_group is None  # 風控出場不計入換股

    def test_expired_plus_buy_assigns_switch_group(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        p = _make_portfolio(patch_session)
        # 1111 不在 Top-N 且已過持有期（15 天 > holding_days 10）→ holding_expired 賣出（非風控）
        _add_position(patch_session, p.id, "1111", today, days_ago=15, stop_loss=50)

        RotationManager("al_test").update(today=today, regime="bull")

        rows = _log_rows(patch_session)
        close = next(r for r in rows if r.action_type == "close")
        opens = [r for r in rows if r.action_type == "open"]
        assert close.reason == "holding_expired"
        assert close.is_risk_exit is False
        # 同日非風控賣出 + 買入 → 同一 switch_group
        assert close.switch_group is not None
        assert opens and all(o.switch_group == close.switch_group for o in opens)

    def test_drawdown_liquidation_logged_as_risk_exit(self, patch_session, monkeypatch):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "1111", today, days_ago=3, stop_loss=50)

        monkeypatch.setattr("src.portfolio.manager.check_drawdown_kill_switch", lambda *a, **kw: True)
        monkeypatch.setattr("src.portfolio.manager.compute_portfolio_drawdown", lambda *a, **kw: 30.0)

        RotationManager("al_test").update(today=today, regime="bull")

        rows = _log_rows(patch_session)
        assert len(rows) == 1
        c = rows[0]
        assert c.action_type == "close"
        assert c.reason == "max_drawdown_liquidation"
        assert c.is_risk_exit is True


# ====================================================================== #
# export 序列化
# ====================================================================== #


class TestSerializeAndExport:
    def test_serialize_row_fields(self):
        row = RotationActionLog(
            portfolio_name="al_test",
            action_date=date(2026, 6, 1),
            action_type="close",
            reason="stop_loss",
            is_risk_exit=True,
            switch_group=None,
            stock_id="1111",
            stock_name="held",
            shares=1000,
            price=100.0,
            entry_rank=1,
            pnl=-50000.0,
            return_pct=-5.0,
        )
        out = ed._serialize_action_log_row(row)
        assert out == {
            "action_type": "close",
            "reason": "stop_loss",
            "is_risk_exit": True,
            "switch_group": None,
            "stock_id": "1111",
            "stock_name": "held",
            "shares": 1000,
            "price": 100.0,
            "entry_rank": 1,
            "pnl": -50000.0,
            "return_pct": -5.0,
        }

    def test_build_today_actions_filters_date_and_sorts(self, patch_session):
        today = date(2026, 6, 1)
        yesterday = today - timedelta(days=1)
        # 故意亂序插入：hold → close → renew → open，並混入前一日資料
        patch_session.add_all(
            [
                RotationActionLog(portfolio_name="al_test", action_date=today, action_type="hold", stock_id="3008"),
                RotationActionLog(portfolio_name="al_test", action_date=today, action_type="close", stock_id="2317"),
                RotationActionLog(portfolio_name="al_test", action_date=today, action_type="renew", stock_id="2454"),
                RotationActionLog(portfolio_name="al_test", action_date=today, action_type="open", stock_id="2330"),
                RotationActionLog(portfolio_name="al_test", action_date=yesterday, action_type="open", stock_id="9999"),
            ]
        )
        patch_session.commit()

        result = ed._build_today_actions(today)
        assert set(result.keys()) == {"al_test"}
        types = [a["action_type"] for a in result["al_test"]]
        assert types == ["open", "close", "renew", "hold"]  # 9999（昨日）被濾掉
        # action_date 不是序列化欄位（只用於過濾，不外露）
        assert "action_date" not in result["al_test"][0]

    def test_build_rotations_includes_today_actions(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        RotationManager("al_test").update(today=today, regime="bull")

        rots = ed._build_rotations(today)
        assert len(rots) == 1
        block = rots[0]
        assert "today_actions" in block
        assert len(block["today_actions"]) == 3
        assert all(a["action_type"] == "open" for a in block["today_actions"])

    def test_build_rotations_empty_actions_when_none(self, patch_session):
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)
        # 不呼叫 update → 無 action log

        rots = ed._build_rotations(today)
        assert rots[0]["today_actions"] == []


# ====================================================================== #
# P0 止血包 #7：update() 入口同日冪等 guard
# ====================================================================== #


class TestUpdateEntryGuard:
    def test_second_run_same_day_is_skipped(self, patch_session):
        """同日第二次 update 應在入口被 guard 短路：回傳 None、持倉/現金零變化。"""
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        mgr = RotationManager("al_test")
        first_actions = mgr.update(today=today, regime="bull")
        assert first_actions is not None and len(first_actions.to_buy) == 3

        positions_before = patch_session.execute(select(RotationPosition)).scalars().all()
        cash_before = patch_session.execute(select(RotationPortfolio)).scalar_one().current_cash

        second_actions = mgr.update(today=today, regime="bull")

        assert second_actions is None, "同日第二次 update 應被冪等 guard 跳過"
        positions_after = patch_session.execute(select(RotationPosition)).scalars().all()
        assert len(positions_after) == len(positions_before)
        assert patch_session.execute(select(RotationPortfolio)).scalar_one().current_cash == cash_before

    def test_force_overrides_guard(self, patch_session):
        """force=True 繞過 guard 正常執行（人工修復情境）。"""
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        mgr = RotationManager("al_test")
        mgr.update(today=today, regime="bull")
        forced_actions = mgr.update(today=today, regime="bull", force=True)

        assert forced_actions is not None, "force=True 應繞過冪等 guard"
        # ActionLog 仍靠 delete-rebuild 收斂，不重複
        assert len(_log_rows(patch_session)) == 3

    def test_dry_run_bypasses_guard(self, patch_session):
        """dry_run 不受 guard 限制（preview 隨時可跑）。"""
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        _make_portfolio(patch_session)

        mgr = RotationManager("al_test")
        mgr.update(today=today, regime="bull")
        preview = mgr.update(today=today, regime="bull", dry_run=True)

        assert preview is not None, "dry_run 應繞過冪等 guard"

    def test_circuit_breaker_day_still_guarded(self, patch_session, monkeypatch):
        """熔斷日只寫 ActionLog 不寫 snapshot——guard 靠 ActionLog 仍應攔住第二次。"""
        today = date(2026, 6, 1)
        _seed_universe(patch_session, today)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "1111", today, days_ago=3, stop_loss=50)

        # 強制熔斷。相容兩種熔斷實作（與止血包 #3 PR 的 merge 順序解耦）：
        # #3 合併後熔斷判斷改用 compute_drawdown_with_snapshots，合併前用 check_drawdown_kill_switch
        from src.portfolio import manager as mgr_mod

        if hasattr(mgr_mod, "compute_drawdown_with_snapshots"):
            monkeypatch.setattr(mgr_mod, "compute_drawdown_with_snapshots", lambda *a, **kw: 30.0)
        else:
            monkeypatch.setattr(mgr_mod, "check_drawdown_kill_switch", lambda *a, **kw: True)
            monkeypatch.setattr(mgr_mod, "compute_portfolio_drawdown", lambda *a, **kw: 30.0)

        mgr = RotationManager("al_test")
        first = mgr.update(today=today, regime="bull")
        assert first is not None and len(first.to_sell) == 1

        # 熔斷後組合 status="liquidated"，第二次會先被 status 檢查擋掉；
        # 這裡強制改回 active 以驗證 guard 本身（僅 ActionLog、無 snapshot 的情境）
        p.status = "active"
        patch_session.commit()

        second = mgr.update(today=today, regime="bull")
        assert second is None, "熔斷日（無 snapshot）第二次 update 仍應被 ActionLog guard 攔住"
