"""2026-05-16 任務 2 — rotation cost-attribution CLI 測試。

對應 5/29 audit：實盤 RotationPosition 成本歸因（手續費/交易稅/滑價）
+ 累計周轉 + bps per turnover，用於量化 alpha 拖累中成本占比。

涵蓋：
  P2-A 純函數 compute_positions_cost_attribution（多場景 + edge cases）
  P2-B RotationManager.get_cost_attribution DB 整合
"""

from __future__ import annotations

from datetime import date

import pytest

from src.constants import COMMISSION_RATE, SLIPPAGE_RATE, TAX_RATE
from src.data.schema import RotationPortfolio, RotationPosition
from src.portfolio.manager import RotationManager
from src.portfolio.rotation import (
    PositionCostAttribution,
    compute_positions_cost_attribution,
)

# ====================================================================== #
# P2-A: 純函數 compute_positions_cost_attribution
# ====================================================================== #


def _pos(
    entry_price: float,
    shares: int,
    *,
    exit_price: float | None = None,
    status: str = "closed",
    buy_slippage: float | None = None,
    sell_slippage: float | None = None,
) -> dict:
    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares": shares,
        "status": status,
        "buy_slippage": buy_slippage,
        "sell_slippage": sell_slippage,
    }


class TestComputePositionsCostAttributionPure:
    def test_empty_positions_returns_zero(self):
        r = compute_positions_cost_attribution([], portfolio_name="x", initial_capital=1_000_000.0)
        assert r.n_positions_total == 0
        assert r.commission == 0.0
        assert r.tax == 0.0
        assert r.slippage == 0.0
        assert r.total_cost == 0.0
        assert r.notional_traded == 0.0
        assert r.cost_per_turnover_bps == 0.0

    def test_single_closed_with_full_slippage_filled(self):
        # entry 100×1000=100k, exit 110×1000=110k
        positions = [_pos(100.0, 1000, exit_price=110.0, buy_slippage=0.001, sell_slippage=0.001)]
        r = compute_positions_cost_attribution(positions, portfolio_name="x", initial_capital=1_000_000.0)

        expected_commission = (100_000 + 110_000) * COMMISSION_RATE
        expected_tax = 110_000 * TAX_RATE
        expected_slippage = 100_000 * 0.001 + 110_000 * 0.001
        assert r.commission == pytest.approx(expected_commission, abs=0.5)
        assert r.tax == pytest.approx(expected_tax, abs=0.5)
        assert r.slippage == pytest.approx(expected_slippage, abs=0.5)
        assert r.notional_traded == pytest.approx(210_000.0)
        assert r.n_positions_closed == 1
        assert r.n_positions_open == 0
        assert r.n_buy_slippage_estimated == 0
        assert r.n_sell_slippage_estimated == 0

    def test_null_slippage_falls_back_to_default(self):
        """commit 649dc2c 前的歷史 position — buy/sell_slippage 為 NULL，應以 SLIPPAGE_RATE 估算。"""
        positions = [_pos(50.0, 2000, exit_price=55.0)]  # 兩端 slippage 都 None
        r = compute_positions_cost_attribution(positions, portfolio_name="x", initial_capital=1_000_000.0)

        expected_slippage = (50.0 * 2000 + 55.0 * 2000) * SLIPPAGE_RATE
        assert r.slippage == pytest.approx(expected_slippage, abs=0.5)
        assert r.n_buy_slippage_estimated == 1
        assert r.n_sell_slippage_estimated == 1

    def test_open_position_only_counts_buy_side(self):
        positions = [_pos(100.0, 500, status="open", buy_slippage=0.001)]
        r = compute_positions_cost_attribution(positions, portfolio_name="x", initial_capital=1_000_000.0)

        assert r.tax == 0.0  # 賣端未發生
        assert r.commission == pytest.approx(100.0 * 500 * COMMISSION_RATE, abs=0.5)
        assert r.slippage == pytest.approx(100.0 * 500 * 0.001, abs=0.5)
        assert r.notional_traded == pytest.approx(50_000.0)
        assert r.n_positions_open == 1
        assert r.n_positions_closed == 0
        assert r.n_sell_slippage_estimated == 0  # 沒有賣端

    def test_mixed_closed_and_open(self):
        positions = [
            _pos(100.0, 1000, exit_price=110.0),  # closed, NULL slip
            _pos(80.0, 500, status="open"),  # open, NULL slip
        ]
        r = compute_positions_cost_attribution(positions, portfolio_name="x", initial_capital=1_000_000.0)
        assert r.n_positions_total == 2
        assert r.n_positions_closed == 1
        assert r.n_positions_open == 1
        # buy 估算: 兩筆都 NULL → 2; sell 估算: 只有 closed 那筆 → 1
        assert r.n_buy_slippage_estimated == 2
        assert r.n_sell_slippage_estimated == 1
        # notional: closed round trip 210k + open buy 40k
        assert r.notional_traded == pytest.approx(250_000.0)

    def test_skip_invalid_zero_entry_price(self):
        positions = [
            _pos(0.0, 1000, exit_price=110.0),  # invalid
            _pos(100.0, 1000, exit_price=105.0),  # valid
        ]
        r = compute_positions_cost_attribution(positions, portfolio_name="x", initial_capital=1_000_000.0)
        assert r.n_positions_total == 1  # 第 1 筆被 skip
        assert r.n_positions_closed == 1

    def test_skip_zero_shares(self):
        positions = [_pos(100.0, 0, exit_price=110.0)]
        r = compute_positions_cost_attribution(positions, portfolio_name="x", initial_capital=1_000_000.0)
        assert r.n_positions_total == 0

    def test_pct_calculations(self):
        positions = [_pos(100.0, 1000, exit_price=110.0, buy_slippage=0.001, sell_slippage=0.001)]
        r = compute_positions_cost_attribution(positions, portfolio_name="x", initial_capital=1_000_000.0)
        # total_cost 約 = 210k×0.001425 + 110k×0.003 + (100k+110k)×0.001 ≈ 299 + 330 + 210 = 839
        assert r.cost_pct_of_initial == pytest.approx(r.total_cost / 1_000_000.0 * 100, abs=1e-6)
        assert r.commission_pct + r.tax_pct + r.slippage_pct == pytest.approx(r.cost_pct_of_initial, abs=1e-6)
        assert r.turnover_ratio == pytest.approx(0.21)  # 210k / 1M
        assert r.cost_per_turnover_bps == pytest.approx(r.total_cost / 210_000.0 * 10000, abs=1e-4)

    def test_zero_initial_capital_safe(self):
        """initial_capital=0 不爆 ZeroDivisionError。"""
        positions = [_pos(100.0, 1000, exit_price=110.0)]
        r = compute_positions_cost_attribution(positions, portfolio_name="x", initial_capital=0.0)
        assert r.cost_pct_of_initial == 0.0
        assert r.turnover_ratio == 0.0
        # cost_per_turnover_bps 仍可算（依 notional）
        assert r.cost_per_turnover_bps > 0

    def test_closed_without_exit_price_treated_as_open(self):
        """status=closed 但 exit_price=None（資料異常）→ 只算買端，不擲例外。"""
        positions = [_pos(100.0, 1000, exit_price=None, status="closed")]
        r = compute_positions_cost_attribution(positions, portfolio_name="x", initial_capital=1_000_000.0)
        assert r.tax == 0.0  # 賣端沒發生
        assert r.n_positions_open == 1  # 視同 open
        assert r.n_positions_closed == 0

    def test_custom_default_slippage_rate(self):
        """default_slippage_rate 可覆寫（敏感度測試用途）。"""
        positions = [_pos(100.0, 1000, exit_price=110.0)]
        r_low = compute_positions_cost_attribution(
            positions, portfolio_name="x", initial_capital=1_000_000.0, default_slippage_rate=0.0001
        )
        r_high = compute_positions_cost_attribution(
            positions, portfolio_name="x", initial_capital=1_000_000.0, default_slippage_rate=0.005
        )
        assert r_high.slippage > r_low.slippage * 30  # 50× ratio


# ====================================================================== #
# P2-B: RotationManager.get_cost_attribution（DB 整合）
# ====================================================================== #


def _make_portfolio(db_session, *, name: str = "ca_test", initial: float = 1_000_000.0) -> RotationPortfolio:
    p = RotationPortfolio(
        name=name,
        mode="momentum",
        max_positions=5,
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


def _add_position(
    db_session,
    portfolio: RotationPortfolio,
    *,
    stock_id: str,
    entry_date: date,
    entry_price: float,
    shares: int,
    exit_date: date | None = None,
    exit_price: float | None = None,
    status: str = "closed",
    buy_slippage: float | None = None,
    sell_slippage: float | None = None,
) -> RotationPosition:
    pos = RotationPosition(
        portfolio_id=portfolio.id,
        stock_id=stock_id,
        entry_date=entry_date,
        entry_price=entry_price,
        entry_rank=1,
        planned_exit_date=entry_date,
        exit_date=exit_date,
        exit_price=exit_price,
        shares=shares,
        allocated_capital=entry_price * shares,
        status=status,
        buy_slippage=buy_slippage,
        sell_slippage=sell_slippage,
    )
    db_session.add(pos)
    db_session.commit()
    return pos


class TestGetCostAttributionDB:
    def test_missing_portfolio_returns_none(self, db_session, monkeypatch):
        # 直接呼叫 mgr.get_cost_attribution，當 _load_portfolio 找不到 → None
        # 用 monkeypatch 把 get_session 換成本測試 session
        from src.portfolio import manager as mgr_module

        class _CtxSession:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(mgr_module, "get_session", lambda: _CtxSession(db_session))

        mgr = RotationManager("ghost")
        result = mgr.get_cost_attribution()
        assert result is None

    def test_aggregates_closed_positions_only_by_default(self, db_session, monkeypatch):
        from src.portfolio import manager as mgr_module

        class _CtxSession:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(mgr_module, "get_session", lambda: _CtxSession(db_session))

        p = _make_portfolio(db_session)
        _add_position(
            db_session,
            p,
            stock_id="2330",
            entry_date=date(2026, 4, 1),
            entry_price=600.0,
            shares=100,
            exit_date=date(2026, 4, 10),
            exit_price=630.0,
            status="closed",
        )
        _add_position(
            db_session,
            p,
            stock_id="2317",
            entry_date=date(2026, 4, 5),
            entry_price=100.0,
            shares=500,
            status="open",
        )

        mgr = RotationManager("ca_test")
        r = mgr.get_cost_attribution()
        assert isinstance(r, PositionCostAttribution)
        # include_open 預設 False → open 不算入
        assert r.n_positions_closed == 1
        assert r.n_positions_open == 0
        assert r.n_positions_total == 1

    def test_include_open_toggles_in_open_position(self, db_session, monkeypatch):
        from src.portfolio import manager as mgr_module

        class _CtxSession:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(mgr_module, "get_session", lambda: _CtxSession(db_session))

        p = _make_portfolio(db_session)
        _add_position(
            db_session,
            p,
            stock_id="2330",
            entry_date=date(2026, 4, 1),
            entry_price=600.0,
            shares=100,
            exit_date=date(2026, 4, 10),
            exit_price=630.0,
            status="closed",
        )
        _add_position(
            db_session,
            p,
            stock_id="2317",
            entry_date=date(2026, 4, 5),
            entry_price=100.0,
            shares=500,
            status="open",
        )

        mgr = RotationManager("ca_test")
        r = mgr.get_cost_attribution(include_open=True)
        assert r.n_positions_total == 2
        assert r.n_positions_open == 1
        assert r.n_positions_closed == 1

    def test_date_range_filter(self, db_session, monkeypatch):
        from src.portfolio import manager as mgr_module

        class _CtxSession:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(mgr_module, "get_session", lambda: _CtxSession(db_session))

        p = _make_portfolio(db_session)
        # 三筆，分別 3/15、4/1、5/1
        for d, sid in [(date(2026, 3, 15), "A"), (date(2026, 4, 1), "B"), (date(2026, 5, 1), "C")]:
            _add_position(
                db_session,
                p,
                stock_id=sid,
                entry_date=d,
                entry_price=100.0,
                shares=100,
                exit_date=d,
                exit_price=105.0,
                status="closed",
            )

        mgr = RotationManager("ca_test")
        r = mgr.get_cost_attribution(start_date=date(2026, 4, 1), end_date=date(2026, 4, 30))
        assert r.n_positions_closed == 1  # 只有 B

    def test_historical_null_slippage_uses_estimation(self, db_session, monkeypatch):
        """commit 649dc2c 前的歷史 position 應全部以 default slippage 估算（透明化計數）。"""
        from src.portfolio import manager as mgr_module

        class _CtxSession:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(mgr_module, "get_session", lambda: _CtxSession(db_session))

        p = _make_portfolio(db_session)
        # 三筆 closed 全部 buy/sell_slippage=NULL
        for d, sid in [(date(2026, 4, 1), "A"), (date(2026, 4, 5), "B"), (date(2026, 4, 10), "C")]:
            _add_position(
                db_session,
                p,
                stock_id=sid,
                entry_date=d,
                entry_price=100.0,
                shares=100,
                exit_date=d,
                exit_price=105.0,
                status="closed",
            )

        mgr = RotationManager("ca_test")
        r = mgr.get_cost_attribution()
        assert r.n_buy_slippage_estimated == 3
        assert r.n_sell_slippage_estimated == 3
        # 滑價 ≈ (100+105) × 100 × SLIPPAGE_RATE × 3 = 30.75
        assert r.slippage == pytest.approx((100.0 + 105.0) * 100 * SLIPPAGE_RATE * 3, abs=0.5)
