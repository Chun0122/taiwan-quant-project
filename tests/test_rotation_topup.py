"""補倉（target-weight top-up）測試 — 純函數 + plan_topup + fill_pending 補倉路徑。

背景：2026-09-07 修好 `compute_rotation_actions` 的部位大小後（分母由 max_positions
改為空缺數），修復只作用在**新買入**；既有的萎縮部位要等自然換手才會重建，期間
組合以遠低於設計的曝險運行（實測 mom3_20d 7.1%、mom5_10d 22.3%）。補倉是把殘留
缺口一次補平的維護路徑。

涵蓋：
  TU-A  純函數：目標等權缺口、等比縮減、門檻、跳過無報價
  TU-B  plan_topup：寫 pending 買單 + ActionLog、非 active 拒絕、冪等、dry_run
  TU-C  fill_pending：加碼既有部位（不新開倉）、加權平均進場價、持有時鐘不變
  TU-D  fill_pending：持倉已出場的補倉單一律取消，**絕不退化成開新倉**
  TU-E  E2E：plan_topup(D) → fill_pending(D+1)，現金守恆
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select

from src.constants import (
    ACTION_TYPE_PENDING_TOPUP,
    ACTION_TYPE_TOPUP,
    PENDING_REASON_TOPUP,
)
from src.data.schema import (
    DailyPrice,
    RotationActionLog,
    RotationPendingOrder,
    RotationPortfolio,
    RotationPosition,
)
from src.portfolio.execution_core import simulate_buy
from src.portfolio.manager import RotationManager
from src.portfolio.rotation import compute_topup_orders

D = date(2026, 6, 1)
D1 = date(2026, 6, 2)


# ---------------------------------------------------------------------------
# TU-A 純函數
# ---------------------------------------------------------------------------


class TestComputeTopupOrders:
    """目標部位與 compute_rotation_actions 同式：total_capital / max_positions。"""

    def test_gap_to_target_equal_weight(self):
        # 資本 900k / N=3 → 目標 300k；持倉各 100k → 每檔缺 200k
        positions = [{"stock_id": s, "shares": 1000} for s in ("A", "B", "C")]
        prices = {"A": 100.0, "B": 100.0, "C": 100.0}
        orders = compute_topup_orders(positions, prices, 900_000.0, 3, 600_000.0)

        assert len(orders) == 3
        for o in orders:
            assert o["target_capital"] == pytest.approx(300_000.0)
            assert o["gap"] == pytest.approx(200_000.0)
            assert o["allocated_capital"] == pytest.approx(200_000.0)

    def test_positions_end_equal_weight(self):
        """行為級：補完後各部位市值應相等（這才是「等權」的意思）。"""
        positions = [
            {"stock_id": "A", "shares": 1000},  # 100k
            {"stock_id": "B", "shares": 300},  # 30k
            {"stock_id": "C", "shares": 100},  # 10k
        ]
        prices = {"A": 100.0, "B": 100.0, "C": 100.0}
        total_capital = 900_000.0
        orders = compute_topup_orders(positions, prices, total_capital, 3, 760_000.0)

        held = {p["stock_id"]: p["shares"] for p in positions}
        after = {o["stock_id"]: (held[o["stock_id"]] + o["shares"]) * prices[o["stock_id"]] for o in orders}
        assert len(after) == 3
        # 各部位市值互相接近（誤差僅來自整數股與成本內扣）
        assert max(after.values()) - min(after.values()) < 1_000

    def test_insufficient_cash_scales_proportionally(self):
        """現金不足時等比縮減，而非先來後到把前幾檔補滿、後幾檔掛零。"""
        positions = [
            {"stock_id": "A", "shares": 1000},  # 100k，缺 200k
            {"stock_id": "B", "shares": 500},  # 50k，缺 250k
        ]
        prices = {"A": 100.0, "B": 100.0}
        # 目標 300k×2；總缺口 450k，但只有 225k 現金 → scale = 0.5
        orders = compute_topup_orders(positions, prices, 900_000.0, 3, 225_000.0)

        allocs = {o["stock_id"]: o["allocated_capital"] for o in orders}
        assert allocs["A"] == pytest.approx(100_000.0)
        assert allocs["B"] == pytest.approx(125_000.0)
        assert sum(allocs.values()) == pytest.approx(225_000.0)

    def test_small_gap_below_threshold_skipped(self):
        """缺口 < 目標 × min_gap_ratio 不補，避免價格波動造成的零碎單。"""
        positions = [{"stock_id": "A", "shares": 2900}]  # 290k，目標 300k，缺 3.3%
        orders = compute_topup_orders(positions, {"A": 100.0}, 900_000.0, 3, 500_000.0)
        assert orders == []

    def test_position_at_or_above_target_skipped(self):
        positions = [{"stock_id": "A", "shares": 4000}]  # 400k > 目標 300k
        assert compute_topup_orders(positions, {"A": 100.0}, 900_000.0, 3, 500_000.0) == []

    def test_missing_price_skipped(self):
        """無報價的持倉無法定價，直接略過（不以 entry_price 猜）。"""
        positions = [{"stock_id": "A", "shares": 1000}, {"stock_id": "B", "shares": 1000}]
        orders = compute_topup_orders(positions, {"A": 100.0}, 900_000.0, 3, 500_000.0)
        assert [o["stock_id"] for o in orders] == ["A"]

    @pytest.mark.parametrize(
        ("capital", "max_positions", "cash"),
        [(0.0, 3, 500_000.0), (900_000.0, 0, 500_000.0), (900_000.0, 3, 0.0)],
    )
    def test_degenerate_inputs_return_empty(self, capital, max_positions, cash):
        positions = [{"stock_id": "A", "shares": 1000}]
        assert compute_topup_orders(positions, {"A": 100.0}, capital, max_positions, cash) == []

    def test_no_positions_returns_empty(self):
        assert compute_topup_orders([], {}, 900_000.0, 3, 500_000.0) == []


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------


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


def _seed_prices(db_session, *, d1_open: float = 100.0) -> None:
    for i in range(-30, 31):
        d = D + timedelta(days=i)
        db_session.add(
            DailyPrice(stock_id="TAIEX", date=d, open=23000, high=23100, low=22900, close=23050, volume=0, turnover=0.0)
        )
    for sid in ("2330", "2317"):
        db_session.add(
            DailyPrice(stock_id=sid, date=D, open=100, high=102, low=99, close=100, volume=50_000_000, turnover=1e9)
        )
        db_session.add(
            DailyPrice(
                stock_id=sid,
                date=D1,
                open=d1_open,
                high=d1_open + 2,
                low=d1_open - 2,
                close=d1_open + 1,
                volume=50_000_000,
                turnover=1e9,
            )
        )
    db_session.commit()


def _make_portfolio(db_session, *, name="tu_test", capital=600_000.0, cash=400_000.0, status="active"):
    p = RotationPortfolio(
        name=name,
        mode="momentum",
        max_positions=3,
        holding_days=10,
        allow_renewal=True,
        initial_capital=capital,
        current_capital=capital,
        current_cash=cash,
        status=status,
    )
    db_session.add(p)
    db_session.commit()
    return p


def _add_position(db_session, portfolio_id, sid="2330", shares=1000, entry_price=100.0):
    pos = RotationPosition(
        portfolio_id=portfolio_id,
        stock_id=sid,
        stock_name=f"name_{sid}",
        entry_date=D - timedelta(days=3),
        entry_price=entry_price,
        entry_rank=1,
        entry_score=0.9,
        holding_days_count=3,
        planned_exit_date=D + timedelta(days=7),
        shares=shares,
        allocated_capital=entry_price * shares,
        stop_loss=90.0,
        status="open",
        buy_slippage=0.0005,
        trade_cost=150.0,
    )
    db_session.add(pos)
    db_session.commit()
    return pos


def _pending(session, name="tu_test"):
    return (
        session.execute(select(RotationPendingOrder).where(RotationPendingOrder.portfolio_name == name)).scalars().all()
    )


# ---------------------------------------------------------------------------
# TU-B plan_topup
# ---------------------------------------------------------------------------


class TestPlanTopup:
    def test_writes_pending_buy_and_action_log(self, patch_session):
        _seed_prices(patch_session)
        p = _make_portfolio(patch_session)  # 資本 600k / N=3 → 目標 200k
        _add_position(patch_session, p.id, "2330", shares=1000)  # 100k，缺 100k

        summary = RotationManager("tu_test").plan_topup(decision_date=D)

        assert summary["planned"] == 1
        assert summary["notional"] == pytest.approx(100_000.0)
        assert summary["exposure_before"] == pytest.approx(1 / 6)

        orders = _pending(patch_session)
        assert len(orders) == 1
        assert orders[0].side == "buy"
        assert orders[0].reason == PENDING_REASON_TOPUP
        assert orders[0].decision_date == D
        assert orders[0].status == "pending"
        # 停損沿用既有持倉（補倉不重設停損）
        assert orders[0].stop_loss == 90.0

        logs = (
            patch_session.execute(
                select(RotationActionLog).where(RotationActionLog.action_type == ACTION_TYPE_PENDING_TOPUP)
            )
            .scalars()
            .all()
        )
        assert len(logs) == 1
        assert logs[0].stock_id == "2330"

    def test_paused_portfolio_refused(self, patch_session):
        """非 active 的組合 fill_pending 會整段跳過，掛單永遠不會成交 → 直接拒絕。"""
        _seed_prices(patch_session)
        p = _make_portfolio(patch_session, status="paused")
        _add_position(patch_session, p.id, "2330", shares=1000)

        assert RotationManager("tu_test").plan_topup(decision_date=D) is None
        assert _pending(patch_session) == []

    def test_dry_run_writes_nothing(self, patch_session):
        _seed_prices(patch_session)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "2330", shares=1000)

        summary = RotationManager("tu_test").plan_topup(decision_date=D, dry_run=True)

        assert summary["planned"] == 1
        assert _pending(patch_session) == []

    def test_rerun_replaces_previous_plan(self, patch_session):
        """冪等：重跑取代尚未成交的舊補倉單，不累積成兩份。"""
        _seed_prices(patch_session)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "2330", shares=1000)
        mgr = RotationManager("tu_test")

        mgr.plan_topup(decision_date=D)
        mgr.plan_topup(decision_date=D)

        orders = [o for o in _pending(patch_session) if o.status == "pending"]
        assert len(orders) == 1
        logs = (
            patch_session.execute(
                select(RotationActionLog).where(RotationActionLog.action_type == ACTION_TYPE_PENDING_TOPUP)
            )
            .scalars()
            .all()
        )
        assert len(logs) == 1

    def test_no_gap_plans_nothing(self, patch_session):
        _seed_prices(patch_session)
        p = _make_portfolio(patch_session, capital=600_000.0, cash=400_000.0)
        _add_position(patch_session, p.id, "2330", shares=2000)  # 200k = 目標

        summary = RotationManager("tu_test").plan_topup(decision_date=D)

        assert summary["planned"] == 0
        assert _pending(patch_session) == []


# ---------------------------------------------------------------------------
# TU-C / TU-D / TU-E fill_pending 補倉路徑
# ---------------------------------------------------------------------------


def _add_topup_order(db_session, *, name="tu_test", sid="2330", shares=1000, alloc=100_000.0, decision_date=D):
    db_session.add(
        RotationPendingOrder(
            portfolio_name=name,
            decision_date=decision_date,
            side="buy",
            stock_id=sid,
            stock_name=f"name_{sid}",
            shares=shares,
            ref_price=100.0,
            allocated_capital=alloc,
            reason=PENDING_REASON_TOPUP,
            stop_loss=90.0,
            status="pending",
        )
    )
    db_session.commit()


class TestFillTopup:
    def test_adds_to_existing_position_not_new(self, patch_session):
        _seed_prices(patch_session, d1_open=100.0)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "2330", shares=1000, entry_price=100.0)
        _add_topup_order(patch_session)

        result = RotationManager("tu_test").fill_pending(exec_day=D1)

        assert result["filled"] == 1
        positions = (
            patch_session.execute(select(RotationPosition).where(RotationPosition.portfolio_id == p.id)).scalars().all()
        )
        assert len(positions) == 1, "補倉必須加碼既有部位，不可新增第二筆 position"
        assert positions[0].shares > 1000

    def test_weighted_average_entry_price(self, patch_session):
        """加碼後 entry_price = 股數加權平均（出場 pnl 才反映真實成本）。"""
        _seed_prices(patch_session, d1_open=120.0)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "2330", shares=1000, entry_price=100.0)
        _add_topup_order(patch_session, alloc=120_000.0)

        RotationManager("tu_test").fill_pending(exec_day=D1)

        pos = patch_session.execute(select(RotationPosition)).scalars().one()
        added = pos.shares - 1000
        expected = (100.0 * 1000 + 120.0 * added) / pos.shares
        assert pos.entry_price == pytest.approx(expected)
        assert 100.0 < pos.entry_price < 120.0

    def test_holding_clock_and_stop_loss_unchanged(self, patch_session):
        """補倉不重設持有時鐘與停損——那是原始進場的屬性，移動停損是另一個決策。"""
        _seed_prices(patch_session, d1_open=100.0)
        p = _make_portfolio(patch_session)
        before = _add_position(patch_session, p.id, "2330", shares=1000)
        entry_date, planned_exit, stop_loss = before.entry_date, before.planned_exit_date, before.stop_loss
        holding_count, entry_rank = before.holding_days_count, before.entry_rank
        _add_topup_order(patch_session)

        RotationManager("tu_test").fill_pending(exec_day=D1)

        pos = patch_session.execute(select(RotationPosition)).scalars().one()
        assert pos.entry_date == entry_date
        assert pos.planned_exit_date == planned_exit
        assert pos.stop_loss == stop_loss
        assert pos.holding_days_count == holding_count
        assert pos.entry_rank == entry_rank

    def test_trade_cost_and_allocated_capital_accumulate(self, patch_session):
        _seed_prices(patch_session, d1_open=100.0)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "2330", shares=1000, entry_price=100.0)
        _add_topup_order(patch_session, alloc=100_000.0)

        RotationManager("tu_test").fill_pending(exec_day=D1)

        pos = patch_session.execute(select(RotationPosition)).scalars().one()
        assert pos.allocated_capital == pytest.approx(100_000.0 + 100_000.0)
        assert pos.trade_cost > 150.0  # 原始 150 + 本次買入成本

    def test_action_log_uses_topup_type(self, patch_session):
        """成交紀錄用 topup 而非 open——UI 與交易統計都不該把它當成新開倉。"""
        _seed_prices(patch_session, d1_open=100.0)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "2330", shares=1000)
        _add_topup_order(patch_session)

        RotationManager("tu_test").fill_pending(exec_day=D1)

        logs = (
            patch_session.execute(select(RotationActionLog).where(RotationActionLog.action_date == D1)).scalars().all()
        )
        types = {row.action_type for row in logs}
        assert ACTION_TYPE_TOPUP in types
        assert "open" not in types

    def test_cancelled_when_position_already_exited(self, patch_session):
        """持倉已不在（例如今晨停損先成交）→ 補倉單取消，絕不退化成開新倉。"""
        _seed_prices(patch_session, d1_open=100.0)
        _make_portfolio(patch_session)  # 不建持倉，模擬已出場
        _add_topup_order(patch_session)

        result = RotationManager("tu_test").fill_pending(exec_day=D1)

        assert result["filled"] == 0
        assert result["cancelled"] == 1
        assert patch_session.execute(select(RotationPosition)).scalars().all() == []
        order = _pending(patch_session)[0]
        assert order.status == "cancelled"

    def test_normal_buy_on_held_stock_still_cancelled(self, patch_session):
        """非補倉的一般買單打到既有持倉時，原有的重複防護不可鬆掉。"""
        _seed_prices(patch_session, d1_open=100.0)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "2330", shares=1000)
        patch_session.add(
            RotationPendingOrder(
                portfolio_name="tu_test",
                decision_date=D,
                side="buy",
                stock_id="2330",
                stock_name="name_2330",
                shares=1000,
                ref_price=100.0,
                allocated_capital=100_000.0,
                reason=None,
                status="pending",
            )
        )
        patch_session.commit()

        result = RotationManager("tu_test").fill_pending(exec_day=D1)

        assert result["cancelled"] == 1
        assert patch_session.execute(select(RotationPosition)).scalars().one().shares == 1000

    def test_e2e_plan_then_fill_conserves_cash(self, patch_session):
        """plan_topup(D) → fill_pending(D+1)：現金扣減 = simulate_buy 的總支出。"""
        _seed_prices(patch_session, d1_open=100.0)
        p = _make_portfolio(patch_session)
        _add_position(patch_session, p.id, "2330", shares=1000)
        cash_before = p.current_cash
        mgr = RotationManager("tu_test")

        mgr.plan_topup(decision_date=D)
        mgr.fill_pending(exec_day=D1)

        pos = patch_session.execute(select(RotationPosition)).scalars().one()
        added = pos.shares - 1000
        portfolio = (
            patch_session.execute(select(RotationPortfolio).where(RotationPortfolio.name == "tu_test")).scalars().one()
        )
        # 成交滑價由 compute_dynamic_slippage 決定；從 buy_slippage 的股數加權平均
        # 反解回本次成交的滑價，順帶驗證混合欄位本身算對
        fill_slippage = (pos.buy_slippage * pos.shares - 0.0005 * 1000) / added
        fill = simulate_buy(100.0, added, fill_slippage)
        assert cash_before - portfolio.current_cash == pytest.approx(fill.buy_cost, rel=1e-6)
