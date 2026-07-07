"""rotation live 股利會計測試（A3-3，_process_dividends via fill_pending）。

涵蓋：
  LD-A  除息日現金入帳 + 停損調整 + ActionLog(dividend) + 權益連續
  LD-B  冪等：同日重跑 fill_pending 不二次入帳（ActionLog 標記）
  LD-C  dry_run：回報事件數但不落庫（cash/停損/ActionLog 全不動）
  LD-D  配股第一版：股數不調、停損按合成因子調整、現金不動
  LD-E  parity：含除息段 live 三日兩段式與 backtest() 進場/期末權益一致

劇本（與 test_rotation_dividend_backtest 同構）：
  D0 決策(收100) → D1 開100進場(收100) → D2 除息跳空(開94收94，cash 6/股)

註：各測試用不重疊的日期窗 + 獨立 portfolio 名，避開 in-memory engine
跨測試污染（db_session fixture 陷阱，同 test_rotation_t1_parity）。
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select

from src.data.schema import (
    DailyPrice,
    DiscoveryRecord,
    Dividend,
    RotationActionLog,
    RotationPortfolio,
    RotationPosition,
)
from src.portfolio.manager import RotationManager

BASE = date(2026, 6, 1)


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


def _price(sid, d, o, h, lo, c, vol=10_000_000):
    return DailyPrice(stock_id=sid, date=d, open=o, high=h, low=lo, close=c, volume=vol, turnover=1e9)


def _seed(db_session, *, base: date, sid: str, cash_div=6.0, stock_div=0.0, d2_price=None, stop_loss=95.0):
    """三日劇本種子：TAIEX 行事曆 + 個股三日報價 + D0 DiscoveryRecord + D2 除息。"""
    d0, d1, d2 = base, base + timedelta(days=1), base + timedelta(days=2)
    for i in range(-10, 11):
        db_session.add(_price("TAIEX", base + timedelta(days=i), 23000, 23100, 22900, 23050, vol=0))
    if d2_price is None:
        d2_price = 100 - cash_div  # 跳空恰等於現金股利
    db_session.add_all(
        [
            _price(sid, d0, 100, 101, 99, 100),
            _price(sid, d1, 100, 101, 99, 100),
            _price(sid, d2, d2_price, d2_price + 1, d2_price - 0.5, d2_price),
        ]
    )
    db_session.add(
        DiscoveryRecord(
            scan_date=d0,
            mode="momentum",
            rank=1,
            stock_id=sid,
            stock_name="divlive",
            close=100,
            composite_score=0.9,
            technical_score=0.8,
            chip_score=0.7,
            fundamental_score=0.5,
            news_score=0.4,
            regime="bull",
            entry_price=100,
            stop_loss=stop_loss,
        )
    )
    db_session.add(Dividend(stock_id=sid, date=d2, year="115", cash_dividend=cash_div, stock_dividend=stock_div))
    db_session.commit()
    return d0, d1, d2


def _make_portfolio(db_session, name, initial=1_000_000.0):
    p = RotationPortfolio(
        name=name,
        mode="momentum",
        max_positions=1,
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


def _dividend_logs(session, name, d):
    return (
        session.execute(
            select(RotationActionLog).where(
                RotationActionLog.portfolio_name == name,
                RotationActionLog.action_date == d,
                RotationActionLog.action_type == "dividend",
            )
        )
        .scalars()
        .all()
    )


class TestLiveDividendAccrual:
    def test_cash_accrual_stop_adjust_and_equity_continuity(self, patch_session):
        d0, d1, d2 = _seed(patch_session, base=BASE, sid="1111")
        _make_portfolio(patch_session, "ld_a")
        mgr = RotationManager("ld_a")

        mgr.update(today=d0, regime="bull")
        mgr.update(today=d1, regime="bull")
        p = patch_session.execute(select(RotationPortfolio).where(RotationPortfolio.name == "ld_a")).scalar_one()
        pos = patch_session.execute(select(RotationPosition).where(RotationPosition.portfolio_id == p.id)).scalar_one()
        assert pos.entry_price == 100.0
        cash_d1, capital_d1, shares = p.current_cash, p.current_capital, pos.shares

        mgr.update(today=d2, regime="bull")

        # 現金入帳：shares × 6
        assert p.current_cash == pytest.approx(cash_d1 + shares * 6.0)
        # 停損調整：95 × (100−6)/100 = 89.3（close 94 > 89.3 → 不誤觸）
        assert pos.stop_loss == pytest.approx(95.0 * 0.94)
        assert pos.status == "open", "除息跳空不得誤觸停損"
        # ActionLog 審計軌跡（pnl=入帳金額、price=每股股利）
        logs = _dividend_logs(patch_session, "ld_a", d2)
        assert len(logs) == 1
        assert logs[0].pnl == pytest.approx(shares * 6.0)
        assert logs[0].price == pytest.approx(6.0)
        # 權益連續：close 94×shares + 入帳 6×shares = 100×shares（假性回撤消除）
        assert p.current_capital == pytest.approx(capital_d1, rel=1e-9)

    def test_rerun_same_day_does_not_double_accrue(self, patch_session):
        d0, d1, d2 = _seed(patch_session, base=BASE + timedelta(days=30), sid="1112")
        _make_portfolio(patch_session, "ld_b")
        mgr = RotationManager("ld_b")
        mgr.update(today=d0, regime="bull")
        mgr.update(today=d1, regime="bull")
        mgr.update(today=d2, regime="bull")
        p = patch_session.execute(select(RotationPortfolio).where(RotationPortfolio.name == "ld_b")).scalar_one()
        cash_after = p.current_cash

        result = mgr.fill_pending(exec_day=d2)

        assert result["dividends"] == 0, "同日重跑不得二次入帳"
        assert p.current_cash == pytest.approx(cash_after)
        assert len(_dividend_logs(patch_session, "ld_b", d2)) == 1

    def test_dry_run_reports_but_writes_nothing(self, patch_session):
        d0, d1, d2 = _seed(patch_session, base=BASE + timedelta(days=60), sid="1113")
        _make_portfolio(patch_session, "ld_c")
        mgr = RotationManager("ld_c")
        mgr.update(today=d0, regime="bull")
        mgr.update(today=d1, regime="bull")
        p = patch_session.execute(select(RotationPortfolio).where(RotationPortfolio.name == "ld_c")).scalar_one()
        pos = patch_session.execute(select(RotationPosition).where(RotationPosition.portfolio_id == p.id)).scalar_one()
        cash_before, stop_before = p.current_cash, pos.stop_loss

        result = mgr.fill_pending(exec_day=d2, dry_run=True)

        assert result["dividends"] == 1, "dry_run 應回報事件數"
        assert p.current_cash == pytest.approx(cash_before)
        assert pos.stop_loss == pytest.approx(stop_before)
        assert _dividend_logs(patch_session, "ld_c", d2) == []

    def test_stock_dividend_adjusts_stop_not_shares(self, patch_session):
        # 純配股 1.0/10 股：因子 = 1/1.1，D2 跳空 100/1.1 ≈ 90.909
        d0, d1, d2 = _seed(
            patch_session,
            base=BASE + timedelta(days=90),
            sid="1114",
            cash_div=0.0,
            stock_div=1.0,
            d2_price=90.909,
        )
        _make_portfolio(patch_session, "ld_d")
        mgr = RotationManager("ld_d")
        mgr.update(today=d0, regime="bull")
        mgr.update(today=d1, regime="bull")
        p = patch_session.execute(select(RotationPortfolio).where(RotationPortfolio.name == "ld_d")).scalar_one()
        pos = patch_session.execute(select(RotationPosition).where(RotationPosition.portfolio_id == p.id)).scalar_one()
        cash_before, shares_before = p.current_cash, pos.shares

        mgr.update(today=d2, regime="bull")

        assert pos.shares == shares_before, "配股第一版不調股數"
        assert pos.stop_loss == pytest.approx(95.0 / 1.1), "停損按配股因子調整"
        assert p.current_cash == pytest.approx(cash_before), "純配股無現金入帳"
        assert pos.status == "open", "配股跳空不得誤觸停損"


class TestLiveBacktestDividendParity:
    def test_live_matches_backtest_across_ex_date(self, patch_session):
        """含除息段：live 三日兩段式與 backtest() 進場與期末權益逐項一致。"""
        d0, d1, d2 = _seed(patch_session, base=BASE + timedelta(days=120), sid="1115")
        _make_portfolio(patch_session, "ld_e")
        mgr = RotationManager("ld_e")

        mgr.update(today=d0, regime="bull")
        mgr.update(today=d1, regime="bull")
        mgr.update(today=d2, regime="bull")
        p = patch_session.execute(select(RotationPortfolio).where(RotationPortfolio.name == "ld_e")).scalar_one()
        pos = patch_session.execute(select(RotationPosition).where(RotationPosition.portfolio_id == p.id)).scalar_one()
        live_entry = (pos.stock_id, pos.entry_price, pos.shares)
        live_capital = p.current_capital

        result = RotationManager("ld_e").backtest(start_date=d0, end_date=d2, save_result=False)

        bt_rows = [r for r in result.daily_positions if r["date"] == d2]
        assert len(bt_rows) == 1
        bt_entry = (bt_rows[0]["stock_id"], bt_rows[0]["entry_price"], bt_rows[0]["shares"])
        assert live_entry == bt_entry, "live 與 backtest 進場必須逐項一致"
        assert result.metrics["dividend_events"] == 1
        bt_final_equity = result.equity_curve[-1]["equity"]
        assert live_capital == pytest.approx(bt_final_equity, rel=1e-9), "含除息段期末權益必須一致"
