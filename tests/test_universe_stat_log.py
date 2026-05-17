"""P1 任務 8 — UniverseStatLog 落庫測試。

涵蓋：
  P8-A UniverseStatLog schema 欄位
  P8-B log_universe_stats 純函數（新增 / upsert / 失敗 graceful）
  P8-C MarketScanner 整合：_get_universe_ids 寫入紀錄
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select

from src.data.schema import UniverseStatLog
from src.discovery.universe import log_universe_stats

# ====================================================================== #
# P8-A: schema 欄位
# ====================================================================== #


class TestSchemaColumns:
    def test_can_create_with_minimal_fields(self, db_session):
        row = UniverseStatLog(
            scan_date=date(2026, 5, 17),
            mode="momentum",
            total_after_sql=1500,
            total_after_liquidity=800,
            total_after_trend=300,
            from_memory=12,
            final_candidates=312,
        )
        db_session.add(row)
        db_session.commit()
        loaded = db_session.execute(select(UniverseStatLog)).scalar_one()
        assert loaded.mode == "momentum"
        assert loaded.regime is None  # nullable
        assert loaded.turnover_multiplier is None

    def test_unique_constraint_on_scan_date_mode(self, db_session):
        db_session.add(
            UniverseStatLog(
                scan_date=date(2026, 5, 17),
                mode="momentum",
                total_after_sql=1500,
                total_after_liquidity=800,
                total_after_trend=300,
                from_memory=0,
                final_candidates=300,
            )
        )
        db_session.commit()
        # 同 (scan_date, mode) 重插 → IntegrityError
        db_session.add(
            UniverseStatLog(
                scan_date=date(2026, 5, 17),
                mode="momentum",
                total_after_sql=999,
                total_after_liquidity=999,
                total_after_trend=999,
                from_memory=0,
                final_candidates=999,
            )
        )
        with pytest.raises(Exception):
            db_session.commit()
        db_session.rollback()


# ====================================================================== #
# P8-B: log_universe_stats 行為
# ====================================================================== #


class TestLogUniverseStats:
    def test_insert_new_row(self, db_session, monkeypatch):
        from src.discovery import universe as uv_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(uv_mod, "get_session", lambda: _Ctx(db_session))

        log_universe_stats(
            scan_date=date(2026, 5, 17),
            mode="momentum",
            stats={
                "total_after_sql": 1500,
                "total_after_liquidity": 800,
                "total_after_trend": 300,
                "from_memory": 12,
                "final_candidates": 312,
            },
            regime="bull",
            turnover_multiplier=0.8,
        )

        row = db_session.execute(select(UniverseStatLog)).scalar_one()
        assert row.total_after_sql == 1500
        assert row.total_after_liquidity == 800
        assert row.total_after_trend == 300
        assert row.from_memory == 12
        assert row.final_candidates == 312
        assert row.regime == "bull"
        assert row.turnover_multiplier == pytest.approx(0.8)

    def test_upsert_same_date_mode_overwrites(self, db_session, monkeypatch):
        """同 (scan_date, mode) 第二次呼叫應覆蓋舊值，不擲 IntegrityError。"""
        from src.discovery import universe as uv_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(uv_mod, "get_session", lambda: _Ctx(db_session))

        log_universe_stats(
            scan_date=date(2026, 5, 17),
            mode="momentum",
            stats={
                "total_after_sql": 100,
                "total_after_liquidity": 50,
                "total_after_trend": 20,
                "from_memory": 0,
                "final_candidates": 20,
            },
        )
        log_universe_stats(
            scan_date=date(2026, 5, 17),
            mode="momentum",
            stats={
                "total_after_sql": 200,
                "total_after_liquidity": 80,
                "total_after_trend": 30,
                "from_memory": 2,
                "final_candidates": 32,
            },
            regime="bear",
        )

        rows = db_session.execute(select(UniverseStatLog)).scalars().all()
        assert len(rows) == 1
        assert rows[0].total_after_sql == 200
        assert rows[0].final_candidates == 32
        assert rows[0].regime == "bear"

    def test_different_modes_coexist(self, db_session, monkeypatch):
        from src.discovery import universe as uv_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(uv_mod, "get_session", lambda: _Ctx(db_session))

        for mode in ("momentum", "swing", "value"):
            log_universe_stats(
                scan_date=date(2026, 5, 17),
                mode=mode,
                stats={
                    "total_after_sql": 1000,
                    "total_after_liquidity": 500,
                    "total_after_trend": 200,
                    "from_memory": 0,
                    "final_candidates": 200,
                },
            )

        rows = db_session.execute(select(UniverseStatLog).order_by(UniverseStatLog.mode)).scalars().all()
        assert {r.mode for r in rows} == {"momentum", "swing", "value"}

    def test_missing_stats_keys_default_zero(self, db_session, monkeypatch):
        """stats dict 缺鍵時不擲例外，欄位填 0。"""
        from src.discovery import universe as uv_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(uv_mod, "get_session", lambda: _Ctx(db_session))

        log_universe_stats(
            scan_date=date(2026, 5, 17),
            mode="momentum",
            stats={"total_after_sql": 100},  # 其他鍵缺
        )

        row = db_session.execute(select(UniverseStatLog)).scalar_one()
        assert row.total_after_sql == 100
        assert row.total_after_liquidity == 0
        assert row.total_after_trend == 0
        assert row.from_memory == 0
        assert row.final_candidates == 0

    def test_failure_is_graceful(self, monkeypatch, caplog):
        """get_session 拋例外時 log warning 而非 raise。"""
        from src.discovery import universe as uv_mod

        def _bad_session():
            raise RuntimeError("DB down")

        monkeypatch.setattr(uv_mod, "get_session", _bad_session)

        # 不應 raise
        log_universe_stats(
            scan_date=date(2026, 5, 17),
            mode="momentum",
            stats={
                "total_after_sql": 100,
                "total_after_liquidity": 50,
                "total_after_trend": 20,
                "from_memory": 0,
                "final_candidates": 20,
            },
        )


# ====================================================================== #
# P8-C: MarketScanner 整合（_get_universe_ids 寫入紀錄）
# ====================================================================== #


class TestMarketScannerIntegration:
    def test_get_universe_ids_writes_log_via_call_count(self, db_session, monkeypatch):
        """驗證 _get_universe_ids 會呼叫 log_universe_stats 一次（避免實跑 scanner）。"""
        from src.discovery.scanner._base import MarketScanner

        # 用最小 patched scanner — 不需要實際 DB；mock UniverseFilter.run
        captured = []

        def _fake_log(**kwargs):
            captured.append(kwargs)

        monkeypatch.setattr("src.discovery.scanner._base.log_universe_stats", _fake_log, raising=False)

        # 由於 src.discovery.scanner._base 內部是 from 進來的，
        # 我們攔截 _get_universe_ids 的執行路徑：直接 mock _universe_filter.run
        class _FakeFilter:
            def run(self, mode):
                return (
                    ["2330", "2317"],
                    {
                        "total_after_sql": 100,
                        "total_after_liquidity": 50,
                        "total_after_trend": 30,
                        "from_memory": 2,
                        "final_candidates": 32,
                    },
                )

        # 構造一個最小化 scanner（直接 instantiate MarketScanner 會跑很多 init；
        # 改成只把方法綁到一個假物件即可）
        class _StubScanner:
            mode_name = "momentum"
            scan_date = date(2026, 5, 17)
            regime = "bull"

            def __init__(self):
                self._universe_filter = _FakeFilter()
                self._universe_config = type("C", (), {"regime": None})()

            _get_universe_ids = MarketScanner._get_universe_ids

        # 由於 log_universe_stats 是在 _get_universe_ids 內 `from ... import` 拉進來的，
        # 我們需要 patch source module 而非 _base 模組
        from src.discovery import universe as uv_mod

        monkeypatch.setattr(uv_mod, "log_universe_stats", _fake_log)

        ids = _StubScanner()._get_universe_ids()
        assert ids == ["2330", "2317"]
        assert len(captured) == 1
        kw = captured[0]
        assert kw["scan_date"] == date(2026, 5, 17)
        assert kw["mode"] == "momentum"
        assert kw["regime"] == "bull"
        assert kw["stats"]["final_candidates"] == 32
