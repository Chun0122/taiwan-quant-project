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
    from src.cli.watch_cmd import _compute_watch_status

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


# ──────────────────────────────────────────────
#  P3：移動止損（Trailing Stop）純函數測試
# ──────────────────────────────────────────────


def _trailing_stop(highest_price, atr14, multiplier):
    """代理呼叫 main._compute_trailing_stop。"""
    from src.cli.watch_cmd import _compute_trailing_stop

    return _compute_trailing_stop(highest_price, atr14, multiplier)


def test_compute_trailing_stop_basic():
    """stop = highest - atr14 * mult，四捨五入至小數點後兩位。"""
    result = _trailing_stop(highest_price=100.0, atr14=5.0, multiplier=1.5)
    assert result == pytest.approx(92.5)


def test_compute_trailing_stop_rounding():
    """結果應四捨五入至小數點後兩位。"""
    result = _trailing_stop(highest_price=100.0, atr14=3.333, multiplier=1.5)
    # 100 - 3.333 * 1.5 = 100 - 4.9995 = 95.0005 → 95.0
    assert result == pytest.approx(95.0, abs=0.01)


def test_compute_trailing_stop_multiplier_2():
    """ATR 倍數 2.0 的計算。"""
    result = _trailing_stop(highest_price=850.0, atr14=10.0, multiplier=2.0)
    assert result == pytest.approx(830.0)


def test_compute_trailing_stop_rises_with_price():
    """價格上漲時，移動止損應隨之上移。"""
    atr14 = 5.0
    mult = 1.5
    # 進場時最高 = 100 → stop = 92.5
    stop1 = _trailing_stop(100.0, atr14, mult)
    # 價格漲到 110 → highest = 110 → stop = 102.5
    stop2 = _trailing_stop(110.0, atr14, mult)
    assert stop2 > stop1
    assert stop2 == pytest.approx(102.5)


# ──────────────────────────────────────────────
#  P3：WatchEntry ORM 移動止損欄位測試
# ──────────────────────────────────────────────


def test_watch_entry_trailing_stop_fields(db_session):
    """WatchEntry 支援 trailing_stop 相關欄位的存取。"""
    entry = WatchEntry(
        stock_id="2330",
        stock_name="台積電",
        entry_date=date(2026, 3, 1),
        entry_price=850.0,
        stop_loss=837.5,
        take_profit=1000.0,
        trailing_stop_enabled=True,
        trailing_atr_multiplier=1.5,
        highest_price_since_entry=850.0,
    )
    db_session.add(entry)
    db_session.flush()

    fetched = db_session.execute(select(WatchEntry).where(WatchEntry.stock_id == "2330")).scalars().first()

    assert fetched is not None
    assert fetched.trailing_stop_enabled is True
    assert fetched.trailing_atr_multiplier == pytest.approx(1.5)
    assert fetched.highest_price_since_entry == pytest.approx(850.0)


def test_watch_entry_trailing_stop_default_false(db_session):
    """未指定 trailing_stop_enabled 時，預設為 False。"""
    entry = WatchEntry(
        stock_id="2454",
        entry_date=date(2026, 3, 1),
        entry_price=200.0,
    )
    db_session.add(entry)
    db_session.flush()

    fetched = db_session.execute(select(WatchEntry).where(WatchEntry.stock_id == "2454")).scalars().first()

    assert fetched is not None
    assert fetched.trailing_stop_enabled is False
    assert fetched.trailing_atr_multiplier is None
    assert fetched.highest_price_since_entry is None


def test_watch_entry_trailing_highest_price_update(db_session):
    """highest_price_since_entry 可以被更新（模擬 update-status 邏輯）。"""
    entry = WatchEntry(
        stock_id="2317",
        entry_date=date(2026, 3, 1),
        entry_price=100.0,
        trailing_stop_enabled=True,
        trailing_atr_multiplier=1.5,
        highest_price_since_entry=100.0,
    )
    db_session.add(entry)
    db_session.flush()

    # 模擬價格上漲 → 更新最高價與止損
    fetched = db_session.execute(select(WatchEntry).where(WatchEntry.stock_id == "2317")).scalars().first()
    fetched.highest_price_since_entry = 115.0
    fetched.stop_loss = _trailing_stop(115.0, 5.0, 1.5)  # = 107.5
    db_session.flush()

    updated = db_session.execute(select(WatchEntry).where(WatchEntry.stock_id == "2317")).scalars().first()
    assert updated.highest_price_since_entry == pytest.approx(115.0)
    assert updated.stop_loss == pytest.approx(107.5)


def test_trailing_stop_only_moves_up():
    """移動止損只升不降：new_stop > current_stop 才更新。"""
    # 純函數層面：只要在外部做 max(new_stop, current_stop) 即可
    current_stop = 92.5
    atr14 = 5.0
    mult = 1.5

    # 價格下跌，highest 不變 → new_stop 不超過 current_stop
    new_stop_same_high = _trailing_stop(100.0, atr14, mult)  # = 92.5（等於）
    assert new_stop_same_high <= current_stop or new_stop_same_high == pytest.approx(current_stop)

    # 價格上漲，highest 更新 → new_stop 上升
    new_stop_higher = _trailing_stop(110.0, atr14, mult)  # = 102.5
    assert new_stop_higher > current_stop
