"""P0 #13 — rankings stale fallback 時效上限測試（2026-07-19 事故重審）。

背景：momentum 被 M2 IC 反向停用後，`_find_latest_rankings` 無時效上限，
使 mom5_10d/mg5_20d 以月餘舊排名（6/16）持續交易；composite 分支 fallback
不過濾 mode，任何模式的掃描日都會被取用。

涵蓋：
  SF-A  時效內（≤ RANKINGS_FALLBACK_MAX_TRADING_DAYS）fallback 正常取得前日排名
  SF-B  逾期 → 回空（訊號斷糧凍結）
  SF-C  composite：只在含 member mode 記錄的掃描日中找（跳過 value-only 日）
  SF-D  composite 逾期 → 回空
  SF-E  E2E decide：逾期凍結時不開新買單，風控賣出（停損）照走
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select

from src.constants import RANKINGS_FALLBACK_MAX_TRADING_DAYS
from src.data.schema import DailyPrice, DiscoveryRecord, RotationPendingOrder, RotationPortfolio, RotationPosition
from src.portfolio.manager import RotationManager

D = date(2026, 6, 15)


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


def _seed_calendar_and_prices(db_session) -> None:
    """TAIEX 行事曆（每日皆交易日）+ 個股報價。"""
    for i in range(-30, 31):
        d = D + timedelta(days=i)
        db_session.add(
            DailyPrice(stock_id="TAIEX", date=d, open=23000, high=23100, low=22900, close=23050, volume=0, turnover=0.0)
        )
    for sid in ("2330", "2317", "1111"):
        for i in range(-15, 1):
            d = D + timedelta(days=i)
            db_session.add(
                DailyPrice(stock_id=sid, date=d, open=100, high=102, low=99, close=100, volume=10_000_000, turnover=1e9)
            )
    db_session.commit()


def _add_discovery(db_session, scan_date: date, mode: str = "momentum", sids=("2330", "2317")) -> None:
    for i, sid in enumerate(sids):
        db_session.add(
            DiscoveryRecord(
                scan_date=scan_date,
                mode=mode,
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


def _make_portfolio(db_session, name="sf_test", mode="momentum") -> RotationPortfolio:
    p = RotationPortfolio(
        name=name,
        mode=mode,
        max_positions=3,
        holding_days=10,
        allow_renewal=True,
        initial_capital=1_000_000.0,
        current_capital=1_000_000.0,
        current_cash=1_000_000.0,
        status="active",
    )
    db_session.add(p)
    db_session.commit()
    return p


class TestFindLatestRankingsStaleness:
    def test_fallback_within_cap_returns_rankings(self, patch_session):
        """SF-A：前一交易日的掃描（elapsed=1 ≤ 上限）→ 正常取得排名。"""
        _seed_calendar_and_prices(patch_session)
        _make_portfolio(patch_session)
        _add_discovery(patch_session, D - timedelta(days=1))

        rankings = RotationManager("sf_test")._find_latest_rankings(patch_session, "momentum", D)

        assert [r["stock_id"] for r in rankings] == ["2330", "2317"]

    def test_fallback_beyond_cap_returns_empty(self, patch_session):
        """SF-B：逾 RANKINGS_FALLBACK_MAX_TRADING_DAYS 個交易日 → 訊號斷糧回空。"""
        _seed_calendar_and_prices(patch_session)
        _make_portfolio(patch_session)
        stale_days = RANKINGS_FALLBACK_MAX_TRADING_DAYS + 2  # 行事曆每日皆交易日
        _add_discovery(patch_session, D - timedelta(days=stale_days))

        rankings = RotationManager("sf_test")._find_latest_rankings(patch_session, "momentum", D)

        assert rankings == []

    def test_composite_skips_non_member_scan_dates(self, patch_session):
        """SF-C：mom_growth fallback 不得取 value-only 掃描日，須回溯至含 member 的日期。"""
        _seed_calendar_and_prices(patch_session)
        _make_portfolio(patch_session, mode="mom_growth")
        _add_discovery(patch_session, D - timedelta(days=1), mode="value", sids=("1111",))
        _add_discovery(patch_session, D - timedelta(days=2), mode="growth")

        rankings = RotationManager("sf_test")._find_latest_rankings(patch_session, "mom_growth", D)

        assert rankings, "應回溯至 D-2 的 growth 記錄而非 D-1 的 value-only 日"
        assert {r["stock_id"] for r in rankings} == {"2330", "2317"}

    def test_composite_beyond_cap_returns_empty(self, patch_session):
        """SF-D：member mode 記錄全數逾期 → 回空（value-only 新掃描日不得續命）。"""
        _seed_calendar_and_prices(patch_session)
        _make_portfolio(patch_session, mode="mom_growth")
        _add_discovery(patch_session, D - timedelta(days=1), mode="value", sids=("1111",))
        _add_discovery(patch_session, D - timedelta(days=RANKINGS_FALLBACK_MAX_TRADING_DAYS + 2), mode="growth")

        rankings = RotationManager("sf_test")._find_latest_rankings(patch_session, "mom_growth", D)

        assert rankings == []


class TestDecideFrozenOnStaleSignals:
    def test_decide_no_new_buys_but_risk_sells_proceed(self, patch_session):
        """SF-E：訊號斷糧凍結時 decide 不開新買單；停損穿越的持倉仍寫賣出 pending。"""
        _seed_calendar_and_prices(patch_session)
        p = _make_portfolio(patch_session)
        _add_discovery(patch_session, D - timedelta(days=RANKINGS_FALLBACK_MAX_TRADING_DAYS + 2))
        # 持倉：現價 100 已跌破停損 105 → 應觸發 stop_loss 賣出
        patch_session.add(
            RotationPosition(
                portfolio_id=p.id,
                stock_id="1111",
                stock_name="held",
                entry_date=D - timedelta(days=3),
                entry_price=120,
                entry_rank=1,
                holding_days_count=3,
                planned_exit_date=D + timedelta(days=7),
                shares=1000,
                allocated_capital=120_000,
                stop_loss=105,
                status="open",
                buy_slippage=0.0005,
            )
        )
        patch_session.commit()

        RotationManager("sf_test").decide(today=D, regime="bull")

        orders = patch_session.execute(select(RotationPendingOrder)).scalars().all()
        buys = [o for o in orders if o.side == "buy"]
        sells = [o for o in orders if o.side == "sell"]
        assert buys == [], "訊號斷糧時不得以舊排名開新買單"
        assert [o.stock_id for o in sells] == ["1111"]
        assert sells[0].reason == "stop_loss"
