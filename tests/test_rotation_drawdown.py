"""C1 修復回歸測試 — Drawdown Kill Switch 反映即時 MtM。

對應 audit 2026-05-09 P0-C1：
- 原 _compute_equity_history append portfolio.current_capital（過時值）→
  gap-down 情境下 drawdown 不觸發熔斷。
- 修復：接受 open_positions + today_prices，計算即時 MtM。

測試場景：
  T1: 初始 1,000,000 → 已平倉 1 筆 +50,000 → current_capital=1,050,000
  T2: 開倉 5 支 × 進場價 → cash=525,000 + market_value=525,000 = 1,050,000（持平）
  T3: 隔日 gap-down 30% → MtM=525,000×0.7=367,500 → equity=525,000+367,500=892,500
       真實回撤 = (1,050,000 − 892,500) / 1,050,000 ≈ 15.0%
  T4 (bug 場景): portfolio.current_capital 仍是 T2 的 1,050,000（未刷 MtM）
       原 _compute_equity_history → equity peak=1,050,000、final=1,050,000 → dd=0%
       新 _compute_equity_history（含 today_prices）→ final=892,500 → dd=15%
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from src.data.schema import RotationDailySnapshot, RotationPortfolio, RotationPosition
from src.portfolio.manager import RotationManager
from src.portfolio.rotation import (
    check_drawdown_kill_switch,
    compute_drawdown_with_snapshots,
    compute_portfolio_drawdown,
)


def _make_portfolio(
    session, *, initial=1_000_000, current_capital=1_000_000, current_cash=1_000_000
) -> RotationPortfolio:
    p = RotationPortfolio(
        name="test_dd",
        mode="momentum",
        max_positions=5,
        holding_days=5,
        allow_renewal=False,
        initial_capital=initial,
        current_capital=current_capital,
        current_cash=current_cash,
        status="active",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    session.add(p)
    session.flush()
    return p


def _add_position(
    session,
    portfolio_id: int,
    *,
    stock_id: str,
    entry_date: date,
    entry_price: float,
    shares: int,
    status: str,
    pnl: float | None = None,
    exit_date: date | None = None,
    exit_price: float | None = None,
    exit_reason: str | None = None,
) -> RotationPosition:
    pos = RotationPosition(
        portfolio_id=portfolio_id,
        stock_id=stock_id,
        entry_date=entry_date,
        entry_price=entry_price,
        entry_rank=1,
        holding_days_count=5,
        planned_exit_date=entry_date + timedelta(days=5),
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=exit_reason,
        shares=shares,
        allocated_capital=entry_price * shares,
        pnl=pnl,
        status=status,
        created_at=datetime.utcnow(),
    )
    session.add(pos)
    session.flush()
    return pos


# ─────────────────────────────────────────────────────────────────
#  C1 修復核心測試：MtM 反映 gap-down，drawdown 正確觸發熔斷
# ─────────────────────────────────────────────────────────────────


class TestComputeEquityHistoryWithMtM:
    def test_with_open_positions_reflects_intraday_mtm(self, db_session):
        """C1 修復：傳入 open_positions+today_prices 時，equity 反映即時 MtM。

        場景：T 日已建倉 5 支，今日 gap-down 30% →
        equity 應該是 cash + 市值×0.7，而非過時 current_capital。
        """
        p = _make_portfolio(
            db_session,
            initial=1_000_000,
            current_capital=1_050_000,  # 上輪結束時值（昨日）
            current_cash=525_000,
        )
        # 1 筆已平倉（+50,000）
        _add_position(
            db_session,
            p.id,
            stock_id="2330",
            entry_date=date(2025, 4, 1),
            entry_price=500.0,
            shares=1000,
            status="closed",
            pnl=50_000,
            exit_date=date(2025, 4, 5),
            exit_price=550.0,
            exit_reason="holding_expired",
        )
        # 5 支當前 open，總市值 525,000（每支 105,000）
        for sid, price in [("1101", 100), ("1102", 105), ("1103", 110), ("1104", 95), ("1105", 110)]:
            _add_position(
                db_session,
                p.id,
                stock_id=sid,
                entry_date=date(2025, 5, 5),
                entry_price=price,
                shares=1000,
                status="open",
            )

        mgr = RotationManager("test_dd")
        open_positions = mgr._load_open_positions(db_session, p.id)

        # 今日 gap-down 30%
        today_prices = {
            sid: 0.7 * price
            for sid, price in [("1101", 100), ("1102", 105), ("1103", 110), ("1104", 95), ("1105", 110)]
        }

        equity = mgr._compute_equity_history(db_session, p, open_positions=open_positions, today_prices=today_prices)

        # equity = [initial, after_close_pnl, latest_with_mtm]
        assert equity[0] == 1_000_000
        assert equity[1] == 1_050_000  # initial + 50,000 closed pnl
        # MtM = 525,000 × 0.7 = 367,500；cash = 525,000；total = 892,500
        expected_mtm_equity = 525_000 + sum(0.7 * price * 1000 for price in [100, 105, 110, 95, 110])
        assert abs(equity[-1] - expected_mtm_equity) < 1.0

        # drawdown 計算正確：peak=1,050,000，final=892,500 → dd≈15%
        dd = compute_portfolio_drawdown(equity)
        assert 14.0 < dd < 16.0

    def test_with_open_positions_kill_switch_triggers_at_25pct_dd(self, db_session):
        """C1 修復：真實 25% 回撤情境下，drawdown kill switch 應觸發。

        Bug 修復前：portfolio.current_capital 過時 → equity_history final = peak → dd=0% → 不熔斷
        Bug 修復後：今日 MtM = cash + market_value(下跌後) → 真實 dd 反映 → 熔斷
        """
        p = _make_portfolio(
            db_session,
            initial=1_000_000,
            current_capital=1_000_000,
            current_cash=200_000,
        )
        # 5 支 open，總成本 800,000（每支 160,000）
        for sid in ["1101", "1102", "1103", "1104", "1105"]:
            _add_position(
                db_session,
                p.id,
                stock_id=sid,
                entry_date=date(2025, 5, 5),
                entry_price=160.0,
                shares=1000,
                status="open",
            )

        mgr = RotationManager("test_dd")
        open_positions = mgr._load_open_positions(db_session, p.id)

        # 全部跌 50%（80,000/支）→ MtM=400,000 → cash+MtM=600,000 → dd=40%
        today_prices = {sid: 80.0 for sid in ["1101", "1102", "1103", "1104", "1105"]}

        equity_with_mtm = mgr._compute_equity_history(
            db_session, p, open_positions=open_positions, today_prices=today_prices
        )
        assert equity_with_mtm[-1] == pytest.approx(600_000, abs=1.0)
        assert check_drawdown_kill_switch(equity_with_mtm, threshold_pct=25.0) is True

        # 對照舊行為（不傳 today_prices）：用 portfolio.current_capital=1,000,000 → dd=0% → 不熔斷
        equity_old = mgr._compute_equity_history(db_session, p)
        assert equity_old[-1] == 1_000_000
        assert check_drawdown_kill_switch(equity_old, threshold_pct=25.0) is False

    def test_fallback_to_current_capital_when_no_prices(self, db_session):
        """向後相容：未提供 today_prices 時 fallback 至 portfolio.current_capital。"""
        p = _make_portfolio(
            db_session,
            initial=1_000_000,
            current_capital=950_000,
            current_cash=950_000,
        )
        _add_position(
            db_session,
            p.id,
            stock_id="2330",
            entry_date=date(2025, 4, 1),
            entry_price=500.0,
            shares=100,
            status="closed",
            pnl=-50_000,
            exit_date=date(2025, 4, 5),
            exit_price=450.0,
            exit_reason="stop_loss",
        )

        mgr = RotationManager("test_dd")
        equity = mgr._compute_equity_history(db_session, p)

        assert equity == [1_000_000, 950_000, 950_000]

    def test_with_empty_open_positions_uses_cash_only(self, db_session):
        """空持倉 + today_prices 提供時，最後 equity = current_cash + 0（MtM=0）。"""
        p = _make_portfolio(
            db_session,
            initial=1_000_000,
            current_capital=1_100_000,  # 過時
            current_cash=1_080_000,  # 真實 cash
        )
        # 已實現 +80,000
        _add_position(
            db_session,
            p.id,
            stock_id="2330",
            entry_date=date(2025, 4, 1),
            entry_price=500.0,
            shares=1000,
            status="closed",
            pnl=80_000,
            exit_date=date(2025, 4, 5),
            exit_price=580.0,
            exit_reason="holding_expired",
        )

        mgr = RotationManager("test_dd")
        equity = mgr._compute_equity_history(db_session, p, open_positions=[], today_prices={})

        # equity = [initial, after_pnl, current_cash + 0] = [1M, 1.08M, 1.08M]
        assert equity == [1_000_000, 1_080_000, 1_080_000]

    def test_missing_price_falls_back_to_entry_price(self, db_session):
        """today_prices 缺價的個股 fallback 至 entry_price（保守估值）。"""
        p = _make_portfolio(
            db_session,
            initial=1_000_000,
            current_capital=1_000_000,
            current_cash=500_000,
        )
        for sid in ["AAA", "BBB"]:
            _add_position(
                db_session,
                p.id,
                stock_id=sid,
                entry_date=date(2025, 5, 5),
                entry_price=250.0,
                shares=1000,
                status="open",
            )

        mgr = RotationManager("test_dd")
        open_positions = mgr._load_open_positions(db_session, p.id)

        # 只給 AAA 的今日價（下跌），BBB 缺價 → 用 entry_price=250
        today_prices = {"AAA": 200.0}
        equity = mgr._compute_equity_history(db_session, p, open_positions=open_positions, today_prices=today_prices)

        # AAA MtM=200,000；BBB MtM=250,000（fallback）→ total MtM=450,000
        # equity = cash 500,000 + 450,000 = 950,000
        assert equity[-1] == pytest.approx(950_000, abs=1.0)


# ─────────────────────────────────────────────────────────────────
#  P0 止血包 #3：peak 補上 snapshot MtM 序列（浮盈回吐型崩跌）
# ─────────────────────────────────────────────────────────────────


def _add_snapshot(session, name: str, snap_date: date, total_capital: float) -> None:
    session.add(
        RotationDailySnapshot(
            portfolio_name=name,
            snapshot_date=snap_date,
            total_capital=total_capital,
            total_market_value=total_capital,
            total_cash=0.0,
            unrealized_pnl=0.0,
            n_holdings=5,
        )
    )
    session.flush()


class TestComputeDrawdownWithSnapshots:
    def test_snapshot_peak_triggers_kill_switch(self):
        """核心回歸：realized peak=1.0M 但 snapshot 曾達 1.35M（浮盈未實現），
        current=0.95M → 真實 dd≈29.6% ≥ 25%。修復前（只看 equity_history）dd=5%。"""
        equity_history = [1_000_000, 1_000_000, 950_000]
        snapshot_capitals = [1_050_000, 1_200_000, 1_350_000, 1_100_000]

        dd = compute_drawdown_with_snapshots(equity_history, snapshot_capitals)

        assert dd == pytest.approx(29.63, abs=0.01)
        assert dd >= 25.0
        # 修復前行為對照：realized-only 序列只看得到 5%
        assert compute_portfolio_drawdown(equity_history) == pytest.approx(5.0, abs=0.01)

    def test_no_snapshots_falls_back_to_equity_history(self):
        """snapshot 空（缺日/新組合）→ 與現行 compute_portfolio_drawdown 完全一致。"""
        equity_history = [1_000_000, 1_100_000, 900_000]
        assert compute_drawdown_with_snapshots(equity_history, []) == compute_portfolio_drawdown(equity_history)

    def test_snapshot_gap_tolerated(self):
        """snapshot 缺中間日（熔斷日不寫/update 失敗）仍取得正確 max。"""
        equity_history = [1_000_000, 980_000]
        # 缺了最高點前後幾日，但只要高點那筆在就抓得到
        snapshot_capitals = [1_020_000, 1_300_000, 1_010_000]
        dd = compute_drawdown_with_snapshots(equity_history, snapshot_capitals)
        assert dd == pytest.approx((1_300_000 - 980_000) / 1_300_000 * 100, abs=0.01)

    def test_empty_equity_history_returns_zero(self):
        assert compute_drawdown_with_snapshots([], [1_000_000]) == 0.0
        assert compute_drawdown_with_snapshots([], []) == 0.0

    def test_nonpositive_peak_returns_zero(self):
        assert compute_drawdown_with_snapshots([0.0], [0.0]) == 0.0
        assert compute_drawdown_with_snapshots([-100.0], []) == 0.0

    def test_at_peak_returns_zero(self):
        assert compute_drawdown_with_snapshots([1_000_000, 1_200_000], [1_100_000]) == 0.0


class TestLoadSnapshotCapitals:
    def test_loads_own_portfolio_sorted(self, db_session):
        _make_portfolio(db_session)
        _add_snapshot(db_session, "test_dd", date(2026, 6, 1), 1_050_000)
        _add_snapshot(db_session, "test_dd", date(2026, 6, 3), 1_350_000)
        _add_snapshot(db_session, "test_dd", date(2026, 6, 2), 1_200_000)
        _add_snapshot(db_session, "other_pf", date(2026, 6, 2), 9_999_999)  # 他組合須隔離

        mgr = RotationManager("test_dd")
        capitals = mgr._load_snapshot_capitals(db_session)

        assert capitals == [1_050_000, 1_200_000, 1_350_000]

    def test_empty_when_no_snapshots(self, db_session):
        _make_portfolio(db_session)
        mgr = RotationManager("test_dd")
        assert mgr._load_snapshot_capitals(db_session) == []

    def test_kill_switch_end_to_end_with_snapshot_peak(self, db_session):
        """整合：realized-only equity 看不到的浮盈高點，經 snapshot 序列補回後觸發熔斷。

        場景：組合曾浮盈至 1,350,000（snapshot 有記錄、從未實現），
        之後崩跌，今日 MtM 權益 950,000 → dd≈29.6% ≥ 25% 熔斷。
        修復前：equity_history peak=1,000,000 → dd=5% 不觸發。
        """
        p = _make_portfolio(
            db_session,
            initial=1_000_000,
            current_capital=1_000_000,
            current_cash=200_000,
        )
        for sid in ["1101", "1102", "1103"]:
            _add_position(
                db_session,
                p.id,
                stock_id=sid,
                entry_date=date(2026, 5, 5),
                entry_price=300.0,
                shares=1000,
                status="open",
            )
        _add_snapshot(db_session, "test_dd", date(2026, 5, 20), 1_350_000)  # 浮盈高點

        mgr = RotationManager("test_dd")
        open_positions = mgr._load_open_positions(db_session, p.id)
        today_prices = {sid: 250.0 for sid in ["1101", "1102", "1103"]}  # MtM=750,000 → equity=950,000

        equity = mgr._compute_equity_history(db_session, p, open_positions=open_positions, today_prices=today_prices)
        capitals = mgr._load_snapshot_capitals(db_session)
        dd = compute_drawdown_with_snapshots(equity, capitals)

        assert equity[-1] == pytest.approx(950_000, abs=1.0)
        assert dd == pytest.approx(29.63, abs=0.01)
        assert dd >= 25.0
        # 修復前行為對照：不看 snapshot → 5% 不熔斷
        assert check_drawdown_kill_switch(equity, threshold_pct=25.0) is False
