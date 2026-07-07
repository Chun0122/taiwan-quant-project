"""A2-4：live 兩段式 T+1 的 E2E 與 live/backtest parity 測試。

涵蓋：
  TP-A  E2E：update(D) 寫 pending → update(D+1) 先以 open[D+1] 成交再做當日決策
  TP-B  Parity：同一段 discovery 史料，live 兩段式與 backtest() 的
        進場（stock_id/entry_price/shares）與期末權益逐項一致
  TP-C  t1_execution=False 研究旋鈕：同日 close 成交（舊 live 行為），
        entry_price 落在決策日 close 而非 D+1 open
  TP-D  刻意差異聲明：live 熔斷即時（見 test_rotation_decide 的
        test_drawdown_liquidation_immediate_and_cancels_pending）；backtest
        熔斷走 T+1——本檔 parity 比較不含熔斷情境
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


def _seed(db_session, *, d1_open: float = 106.0, base: date = D) -> None:
    """TAIEX 行事曆（base±10 天）+ 個股兩日報價 + base 日 DiscoveryRecord。

    base 參數讓各測試用不重疊的日期窗，避開 backtest() commit 穿透
    fixture rollback 造成的跨測試 UNIQUE 衝突（db_session 陷阱）。
    """
    b1 = base + timedelta(days=1)
    for i in range(-10, 11):
        d = base + timedelta(days=i)
        db_session.add(
            DailyPrice(stock_id="TAIEX", date=d, open=23000, high=23100, low=22900, close=23050, volume=0, turnover=0.0)
        )
    for sid in ("2330", "2317", "2454"):
        db_session.add(
            DailyPrice(stock_id=sid, date=base, open=100, high=102, low=99, close=100, volume=10_000_000, turnover=1e9)
        )
        db_session.add(
            DailyPrice(
                stock_id=sid,
                date=b1,
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
                scan_date=base,
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


def _make_portfolio(db_session, name="tp_test", initial=1_000_000.0) -> RotationPortfolio:
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


class TestTwoStageE2E:
    def test_update_d_then_d1_fills_at_open_then_decides(self, patch_session):
        _seed(patch_session)
        _make_portfolio(patch_session)
        mgr = RotationManager("tp_test")

        # D 日：決策 → 只有 pending，無持倉
        actions_d = mgr.update(today=D, regime="bull")
        assert actions_d is not None and len(actions_d.to_buy) == 3
        assert patch_session.execute(select(RotationPosition)).scalars().all() == []
        assert (
            len(
                patch_session.execute(select(RotationPendingOrder).where(RotationPendingOrder.status == "pending"))
                .scalars()
                .all()
            )
            == 3
        )

        # D+1 日：先以 open[D+1] 成交，再做 D+1 決策（沿用 D 的 rankings → 全部保持）
        actions_d1 = mgr.update(today=D1, regime="bull")

        positions = patch_session.execute(select(RotationPosition)).scalars().all()
        assert len(positions) == 3
        assert all(p.entry_price == 106.0 for p in positions), "成交價必須是 open[D+1]"
        assert all(p.entry_date == D1 for p in positions)
        assert actions_d1 is not None
        assert len(actions_d1.to_hold) == 3, "D+1 決策：新持倉全部保持"
        # ActionLog：D+1 同日有成交（open）與決策（hold）兩組
        d1_logs = (
            patch_session.execute(select(RotationActionLog).where(RotationActionLog.action_date == D1)).scalars().all()
        )
        types = sorted({r.action_type for r in d1_logs})
        assert "open" in types and "hold" in types

    def test_dry_run_full_chain_writes_nothing(self, patch_session):
        _seed(patch_session)
        _make_portfolio(patch_session)
        mgr = RotationManager("tp_test")
        mgr.update(today=D, regime="bull")

        mgr.update(today=D1, regime="bull", dry_run=True)

        # dry_run 不 fill：orders 仍 pending、無持倉
        assert patch_session.execute(select(RotationPosition)).scalars().all() == []
        orders = patch_session.execute(select(RotationPendingOrder)).scalars().all()
        assert all(o.status == "pending" for o in orders)


class TestLiveBacktestParity:
    def test_entries_and_equity_match_backtest(self, patch_session):
        """同一段史料：live 兩段式與 backtest T+1 的進場與期末權益一致。"""
        _seed(patch_session)
        _make_portfolio(patch_session)
        mgr = RotationManager("tp_test")

        # live 路徑
        mgr.update(today=D, regime="bull")
        mgr.update(today=D1, regime="bull")
        live_positions = {
            p.stock_id: (p.entry_price, p.shares)
            for p in patch_session.execute(select(RotationPosition)).scalars().all()
        }
        live_capital = patch_session.execute(select(RotationPortfolio)).scalar_one().current_capital

        # backtest 路徑（同參數、同史料）；save_result=False——落庫路徑走
        # pipeline 的另一條 engine 連線，會把測試資料永久 commit 進
        # session-scope 共用 engine（db_session fixture 陷阱），跳過
        result = RotationManager("tp_test").backtest(start_date=D, end_date=D1, save_result=False)
        bt_rows = [r for r in result.daily_positions if r["date"] == D1]
        bt_positions = {r["stock_id"]: (r["entry_price"], r["shares"]) for r in bt_rows}

        assert live_positions == bt_positions, "live 兩段式與 backtest T+1 的進場必須逐項一致"
        bt_final_equity = result.equity_curve[-1]["equity"]
        assert live_capital == pytest.approx(bt_final_equity, rel=1e-9), "期末權益（cash + close MtM）必須一致"

    def test_t1_false_knob_fills_at_decision_close(self, patch_session):
        """研究旋鈕：t1_execution=False 重現舊 live 同日 close 成交。"""
        base = D + timedelta(days=30)  # 獨立日期窗（避開前一測試 backtest commit 穿透）
        _seed(patch_session, d1_open=106.0, base=base)
        _make_portfolio(patch_session, name="tp_knob")

        result = RotationManager("tp_knob").backtest(
            start_date=base, end_date=base + timedelta(days=1), t1_execution=False, save_result=False
        )

        d_rows = [r for r in result.daily_positions if r["date"] == base]
        assert d_rows, "同日成交模式：決策日即應有持倉"
        assert all(r["entry_price"] == 100 for r in d_rows), "成交價 = 決策日 close（look-ahead 舊行為）"
