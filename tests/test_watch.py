"""持倉監控單元測試 — WatchEntry ORM + _compute_watch_status 純函數。"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select

from src.data.schema import WatchEntry

# ──────────────────────────────────────────────
#  純函數測試（零 mock）
# ──────────────────────────────────────────────


def _compute(entry_price, stop_loss, take_profit, valid_until, latest_price, today):
    """代理呼叫 main._compute_watch_status（避免匯入整個 main 模組）。"""
    from main import _compute_watch_status

    return _compute_watch_status(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        valid_until=valid_until,
        latest_price=latest_price,
        today=today,
    )


def test_compute_watch_status_active():
    """價格在止損～目標之間，且未過期 → active。"""
    result = _compute(
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        valid_until=date(2099, 12, 31),
        latest_price=105.0,
        today=date(2026, 1, 1),
    )
    assert result == "active"


def test_compute_watch_status_stopped_loss():
    """最新價 <= 止損價 → stopped_loss。"""
    result = _compute(
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        valid_until=date(2099, 12, 31),
        latest_price=89.5,
        today=date(2026, 1, 1),
    )
    assert result == "stopped_loss"


def test_compute_watch_status_stopped_loss_exact():
    """最新價 == 止損價（等號邊界）→ stopped_loss。"""
    result = _compute(
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        valid_until=date(2099, 12, 31),
        latest_price=90.0,
        today=date(2026, 1, 1),
    )
    assert result == "stopped_loss"


def test_compute_watch_status_taken_profit():
    """最新價 >= 目標價 → taken_profit。"""
    result = _compute(
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        valid_until=date(2099, 12, 31),
        latest_price=125.0,
        today=date(2026, 1, 1),
    )
    assert result == "taken_profit"


def test_compute_watch_status_expired():
    """today > valid_until，且未觸及止損止利 → expired。"""
    result = _compute(
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        valid_until=date(2025, 1, 1),
        latest_price=105.0,
        today=date(2026, 3, 1),
    )
    assert result == "expired"


def test_compute_watch_status_no_levels():
    """stop/target 皆為 None，未過期 → active。"""
    result = _compute(
        entry_price=100.0,
        stop_loss=None,
        take_profit=None,
        valid_until=date(2099, 12, 31),
        latest_price=80.0,
        today=date(2026, 1, 1),
    )
    assert result == "active"


def test_compute_watch_status_stop_priority_over_expired():
    """同時觸及止損 + 過期：止損優先。"""
    result = _compute(
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        valid_until=date(2025, 1, 1),  # 已過期
        latest_price=85.0,  # 低於止損
        today=date(2026, 3, 1),
    )
    assert result == "stopped_loss"


def test_compute_watch_status_no_latest_price_expired():
    """查無最新收盤價（None）但已過期 → expired。"""
    result = _compute(
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        valid_until=date(2025, 1, 1),
        latest_price=None,
        today=date(2026, 3, 1),
    )
    assert result == "expired"


# ──────────────────────────────────────────────
#  ORM 整合測試（in-memory SQLite）
# ──────────────────────────────────────────────


def test_watch_entry_orm_create(db_session):
    """建立 WatchEntry 並 query 回來，欄位應正確。"""
    entry = WatchEntry(
        stock_id="2330",
        stock_name="台積電",
        entry_date=date(2026, 3, 1),
        entry_price=850.0,
        stop_loss=820.0,
        take_profit=900.0,
        quantity=1000,
        source="manual",
        status="active",
    )
    db_session.add(entry)
    db_session.flush()

    fetched = db_session.execute(select(WatchEntry).where(WatchEntry.stock_id == "2330")).scalars().first()

    assert fetched is not None
    assert fetched.stock_name == "台積電"
    assert fetched.entry_price == pytest.approx(850.0)
    assert fetched.stop_loss == pytest.approx(820.0)
    assert fetched.take_profit == pytest.approx(900.0)
    assert fetched.quantity == 1000
    assert fetched.source == "manual"


def test_watch_entry_default_status(db_session):
    """未指定 status 時，預設應為 'active'。"""
    entry = WatchEntry(
        stock_id="2317",
        stock_name="鴻海",
        entry_date=date(2026, 3, 1),
        entry_price=200.0,
    )
    db_session.add(entry)
    db_session.flush()

    fetched = db_session.execute(select(WatchEntry).where(WatchEntry.stock_id == "2317")).scalars().first()

    assert fetched is not None
    assert fetched.status == "active"
