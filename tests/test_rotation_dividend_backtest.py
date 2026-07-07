"""rotation backtest 股利會計測試（A3-2）。

涵蓋：
  DB-A  除息日現金入帳：total_dividend_cash = shares × cash_div，計入權益
  DB-B  權益守恆：跳空恰等於股利時，除息日權益與前日連續（假性回撤消除）
  DB-C  停損不誤觸：除息跳空 close < 原 stop，但 stop 已按因子調整 → 不觸發
  DB-D  配股：股數不調（第一版），停損價按合成因子調整
  DB-E  無股利資料時行為不變（向後相容）
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select  # noqa: F401  （與同伴測試檔一致的 helper 引用）

from src.data.schema import DailyPrice, DiscoveryRecord, Dividend, RotationPortfolio
from src.portfolio.manager import RotationManager

BASE = date(2026, 6, 8)  # 週一；D0=決策、D1=進場、D2=除息
D0, D1, D2 = BASE, BASE + timedelta(days=1), BASE + timedelta(days=2)


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


def _seed(db_session, *, sid="1111", cash_div=6.0, stock_div=0.0, base=BASE, stop_loss=95.0):
    """單一標的三日劇本：D0 決策(收100) → D1 開100進場(收100) → D2 除息跳空(開94收94)。"""
    d0, d1, d2 = base, base + timedelta(days=1), base + timedelta(days=2)
    for i in range(-5, 6):
        db_session.add(_price("TAIEX", base + timedelta(days=i), 23000, 23100, 22900, 23050, vol=0))
    drop = cash_div  # 跳空恰等於現金股利
    db_session.add_all(
        [
            _price(sid, d0, 100, 101, 99, 100),
            _price(sid, d1, 100, 101, 99, 100),
            _price(sid, d2, 100 - drop, 100 - drop + 1, 100 - drop - 0.5, 100 - drop),
        ]
    )
    db_session.add(
        DiscoveryRecord(
            scan_date=d0,
            mode="momentum",
            rank=1,
            stock_id=sid,
            stock_name="divtest",
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


class TestBacktestDividendAccrual:
    def test_cash_accrual_and_equity_continuity(self, patch_session):
        _seed(patch_session)
        _make_portfolio(patch_session, "db_a")

        result = RotationManager("db_a").backtest(start_date=D0, end_date=D2, save_result=False)

        # D1 進場（open 100）；D2 除息（cash 6/股）
        pos_rows = [r for r in result.daily_positions if r["date"] == D1]
        assert pos_rows, "D1 應已進場"
        shares = pos_rows[0]["shares"]
        assert result.metrics["dividend_events"] == 1
        assert result.metrics["total_dividend_cash"] == pytest.approx(shares * 6.0)

        # 權益守恆：D2 跳空 −6 恰被入帳 +6/股 補償 → 與 D1 權益一致
        eq = {e["date"]: e["equity"] for e in result.equity_curve}
        assert eq[D2] == pytest.approx(eq[D1], rel=1e-9), "除息日權益應與前日連續（假性回撤消除）"

    def test_stop_loss_not_falsely_triggered(self, patch_session):
        """原 stop 95：除息後 close 94 < 95，未調整會誤觸；調整後 stop=95×0.94=89.3 → 不觸發。"""
        _seed(patch_session, stop_loss=95.0, cash_div=6.0)
        _make_portfolio(patch_session, "db_c")

        result = RotationManager("db_c").backtest(start_date=D0, end_date=D2, save_result=False)

        stop_trades = [t for t in result.trades if t["exit_reason"] == "stop_loss"]
        assert stop_trades == [], "除息跳空不得誤觸停損"
        # 持倉續抱至期末
        assert [r for r in result.daily_positions if r["date"] == D2], "D2 應仍持有"

    def test_stock_dividend_adjusts_stop_not_shares(self, patch_session):
        _seed(patch_session, cash_div=0.0, stock_div=1.0)  # 純配股：因子 = 1/1.1
        # 配股情境下 D2 跳空應為 100/1.1 ≈ 90.9；覆寫 D2 價格
        patch_session.query(DailyPrice).filter(DailyPrice.stock_id == "1111", DailyPrice.date == D2).delete()
        patch_session.add(_price("1111", D2, 90.9, 91.5, 90.0, 90.9))
        patch_session.commit()
        _make_portfolio(patch_session, "db_d")

        result = RotationManager("db_d").backtest(start_date=D0, end_date=D2, save_result=False)

        d1_shares = [r for r in result.daily_positions if r["date"] == D1][0]["shares"]
        d2_shares = [r for r in result.daily_positions if r["date"] == D2][0]["shares"]
        assert d1_shares == d2_shares, "第一版配股不調整股數"
        assert result.metrics["total_dividend_cash"] == 0.0
        # stop 95 × (1/1.1) ≈ 86.36 → close 90.9 不觸發
        assert [t for t in result.trades if t["exit_reason"] == "stop_loss"] == []

    def test_no_dividend_data_backward_compatible(self, patch_session):
        _seed(patch_session, cash_div=6.0)
        # 刪掉股利列 → 舊行為：跳空計為虧損、可能觸停損
        patch_session.query(Dividend).delete()
        patch_session.commit()
        _make_portfolio(patch_session, "db_e")

        result = RotationManager("db_e").backtest(start_date=D0, end_date=D2, save_result=False)

        assert result.metrics["total_dividend_cash"] == 0.0
        assert result.metrics["dividend_events"] == 0
        eq = {e["date"]: e["equity"] for e in result.equity_curve}
        assert eq[D2] < eq[D1], "無股利資料 → 跳空仍計為虧損（資料前提的已知限制）"
