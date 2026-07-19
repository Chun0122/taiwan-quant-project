"""P0 #14 — 臨時休市哨兵測試（_verify_data_freshness phantom 偵測）。

背景：2026-07-10 行事曆標為交易日但市場實際休市（颱風假），全市場
daily_price 為 0 筆——morning-routine 照常以陳舊 close 執行 discover 與
rotation decide，成為 7/13 重複買單事故鏈的觸發源。

涵蓋：
  PH-A  交易日 + 全市場當日筆數 < 門檻 → phantom=True（is_stale 同步 True）
  PH-B  交易日 + 筆數 ≥ 門檻 → phantom=False、新鮮度正常
  PH-C  check_phantom=False（--skip-sync 路徑）→ 跳過哨兵不誤判
  PH-D  非交易日（週末）→ 即使 0 筆也不觸發 phantom
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.cli.morning_cmd import _verify_data_freshness
from src.data.schema import DailyPrice

# 2026-06-01 = 週一，非假日
D = date(2026, 6, 1)


@pytest.fixture()
def patch_session(db_session, monkeypatch):
    class _Ctx:
        def __init__(self, s):
            self._s = s

        def __enter__(self):
            return self._s

        def __exit__(self, *a):
            return False

    from src.data import database as db_module

    monkeypatch.setattr(db_module, "get_session", lambda: _Ctx(db_session))
    return db_session


def _seed_taiex(db_session, until: date, days: int = 10) -> None:
    for i in range(days):
        d = until - timedelta(days=i)
        db_session.add(
            DailyPrice(stock_id="TAIEX", date=d, open=23000, high=23100, low=22900, close=23050, volume=0, turnover=0.0)
        )
    db_session.commit()


class TestPhantomSentinel:
    def test_trading_day_with_no_market_rows_flags_phantom(self, patch_session, capsys):
        """PH-A：TAIEX 有今日資料（新鮮度 OK）但全市場僅 1 筆 → phantom。"""
        _seed_taiex(patch_session, until=D)

        result = _verify_data_freshness(D.isoformat())

        assert result["phantom"] is True
        assert result["is_stale"] is True
        assert "臨時休市" in result["message"]
        assert "臨時休市哨兵" in capsys.readouterr().out

    def test_trading_day_with_enough_rows_is_normal(self, patch_session, monkeypatch):
        """PH-B：當日筆數 ≥ 門檻 → 不觸發，走正常新鮮度判定。"""
        import src.constants as constants

        monkeypatch.setattr(constants, "PHANTOM_TRADING_DAY_MIN_ROWS", 1)
        _seed_taiex(patch_session, until=D)

        result = _verify_data_freshness(D.isoformat())

        assert result["phantom"] is False
        assert result["is_stale"] is False
        assert result["gap_days"] == 0

    def test_check_phantom_false_skips_sentinel(self, patch_session):
        """PH-C：--skip-sync 路徑（check_phantom=False）不誤判為臨時休市。"""
        _seed_taiex(patch_session, until=D)

        result = _verify_data_freshness(D.isoformat(), check_phantom=False)

        assert result["phantom"] is False
        assert result["is_stale"] is False

    def test_weekend_never_phantom(self, patch_session):
        """PH-D：週六（非交易日）即使全市場 0 筆也不觸發 phantom。"""
        saturday = date(2026, 6, 6)
        _seed_taiex(patch_session, until=saturday - timedelta(days=1))

        result = _verify_data_freshness(saturday.isoformat())

        assert result["phantom"] is False
