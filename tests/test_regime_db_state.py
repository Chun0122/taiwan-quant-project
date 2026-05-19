"""P2 任務 12 — RegimeStateMachine DB backend 測試。

涵蓋：
  P12-A use_db flag：state_path=None → DB；state_path=Path → JSON
  P12-B DB load：append-only history，取最新 row
  P12-C DB save：每次 update() 新增一筆紀錄（不覆蓋）
  P12-D legacy JSON 自動遷移到 DB
  P12-E DB 失敗 graceful fallback
  P12-F current_regime property DB mode
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import select

from src.data.schema import RegimeStateLog
from src.regime.detector import RegimeState, RegimeStateMachine


def _bull_closes() -> pd.Series:
    return pd.Series([15000 + i * 25 for i in range(130)])


# ====================================================================== #
# P12-A: use_db flag
# ====================================================================== #


class TestUseDbFlag:
    def test_no_state_path_defaults_to_db(self):
        sm = RegimeStateMachine()
        assert sm.use_db is True
        assert sm.state_path is None

    def test_with_state_path_uses_json(self, tmp_path):
        sm = RegimeStateMachine(state_path=tmp_path / "state.json")
        assert sm.use_db is False
        assert sm.state_path == tmp_path / "state.json"


# ====================================================================== #
# P12-B/C: DB save + load
# ====================================================================== #


class TestDbBackend:
    def test_save_appends_to_log(self, db_session, monkeypatch):
        """每次 _save_state_to_db 新增一筆，不覆蓋舊紀錄。"""
        from src.regime import detector as det_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(det_mod, "get_session", lambda: _Ctx(db_session), raising=False)
        # detector imports get_session lazily inside _save_state_to_db；改用 patch src.data.database
        import src.data.database as db_mod

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx(db_session))

        sm = RegimeStateMachine()  # DB mode
        sm._save_state(
            RegimeState(
                regime="bull",
                regime_since="2026-05-01",
                confirmation_count=0,
                pending_transition=None,
                last_updated="2026-05-01",
            )
        )
        sm._save_state(
            RegimeState(
                regime="sideways",
                regime_since="2026-05-10",
                confirmation_count=1,
                pending_transition="bull→sideways",
                last_updated="2026-05-10",
            )
        )

        rows = db_session.execute(select(RegimeStateLog).order_by(RegimeStateLog.id)).scalars().all()
        assert len(rows) == 2
        assert rows[0].regime == "bull"
        assert rows[1].regime == "sideways"
        assert rows[1].pending_transition == "bull→sideways"

    def test_load_returns_latest_by_created_at(self, db_session, monkeypatch):
        """_load_state 取 created_at 最大的那筆。"""
        import src.data.database as db_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx(db_session))

        # 注入 3 筆（id 順序 = created_at 順序，因為 default datetime.utcnow 連續呼叫）
        from datetime import datetime, timedelta

        base = datetime.utcnow()
        for i, regime in enumerate(("bull", "sideways", "bear")):
            db_session.add(
                RegimeStateLog(
                    regime=regime,
                    regime_since=f"2026-05-{i + 1:02d}",
                    confirmation_count=i,
                    pending_transition=None,
                    last_updated=f"2026-05-{i + 1:02d}",
                    created_at=base + timedelta(seconds=i),
                )
            )
        db_session.commit()

        sm = RegimeStateMachine()
        loaded = sm._load_state()
        assert loaded is not None
        assert loaded.regime == "bear"  # 最後寫入的
        assert loaded.confirmation_count == 2

    def test_load_empty_db_returns_none(self, db_session, monkeypatch):
        """DB 為空 + 無 legacy JSON → 回 None（cold start）。"""
        import src.data.database as db_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx(db_session))
        # patch _DEFAULT_STATE_PATH 指向不存在的檔，避免讀到真實 data/regime_state.json
        from src.regime import detector as det_mod

        monkeypatch.setattr(det_mod, "_DEFAULT_STATE_PATH", Path("/tmp/nonexistent_regime_state.json"))

        sm = RegimeStateMachine()
        assert sm._load_state() is None


# ====================================================================== #
# P12-D: legacy JSON 自動遷移
# ====================================================================== #


class TestLegacyJsonMigration:
    def test_db_empty_with_legacy_json_migrates(self, db_session, monkeypatch, tmp_path):
        """DB 空 + legacy JSON 存在 → 一次性遷移 + 後續走 DB。"""
        import src.data.database as db_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx(db_session))

        legacy_path = tmp_path / "regime_state.json"
        legacy_path.write_text(
            '{"regime": "bull", "regime_since": "2026-04-01", "confirmation_count": 3, '
            '"pending_transition": null, "last_updated": "2026-05-15"}',
            encoding="utf-8",
        )

        from src.regime import detector as det_mod

        monkeypatch.setattr(det_mod, "_DEFAULT_STATE_PATH", legacy_path)

        sm = RegimeStateMachine()
        loaded = sm._load_state()
        assert loaded is not None
        assert loaded.regime == "bull"
        assert loaded.regime_since == "2026-04-01"

        # 驗證 DB 已寫入一筆遷移後的紀錄
        rows = db_session.execute(select(RegimeStateLog)).scalars().all()
        assert len(rows) == 1
        assert rows[0].regime == "bull"


# ====================================================================== #
# P12-E: DB 失敗 graceful
# ====================================================================== #


class TestDbFailureGraceful:
    def test_db_query_exception_returns_none(self, monkeypatch):
        """get_session 失敗 → _load_state 回 None（cold start），不擲例外。"""
        import src.data.database as db_mod

        def _bad():
            raise RuntimeError("DB down")

        monkeypatch.setattr(db_mod, "get_session", _bad)

        # 同時 patch 預設 JSON path 避免讀真實檔
        from src.regime import detector as det_mod

        monkeypatch.setattr(det_mod, "_DEFAULT_STATE_PATH", Path("/tmp/nonexistent_x.json"))

        sm = RegimeStateMachine()
        # 第一次呼叫 _load_state_from_db 內部 try/except 應吃掉異常
        result = sm._load_state()
        assert result is None

    def test_db_write_exception_logs_warning(self, monkeypatch, caplog):
        """_save_state 失敗 → log warning 不擲例外。"""
        import src.data.database as db_mod

        def _bad():
            raise RuntimeError("DB write down")

        monkeypatch.setattr(db_mod, "get_session", _bad)

        sm = RegimeStateMachine()
        # 不應 raise
        sm._save_state(
            RegimeState(
                regime="bull",
                regime_since="2026-05-01",
                confirmation_count=0,
                pending_transition=None,
                last_updated="2026-05-01",
            )
        )


# ====================================================================== #
# P12-F: current_regime property with DB mode
# ====================================================================== #


class TestCurrentRegimeDbMode:
    def test_current_regime_reads_db(self, db_session, monkeypatch):
        import src.data.database as db_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx(db_session))

        db_session.add(
            RegimeStateLog(
                regime="bear",
                regime_since="2026-05-01",
                confirmation_count=0,
                pending_transition=None,
                last_updated="2026-05-15",
            )
        )
        db_session.commit()

        sm = RegimeStateMachine()
        assert sm.current_regime == "bear"

    def test_current_regime_none_when_db_empty(self, db_session, monkeypatch):
        import src.data.database as db_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx(db_session))
        from src.regime import detector as det_mod

        monkeypatch.setattr(det_mod, "_DEFAULT_STATE_PATH", Path("/tmp/nonexistent_y.json"))

        sm = RegimeStateMachine()
        assert sm.current_regime is None


# ====================================================================== #
# P12-G: full update() with DB end-to-end
# ====================================================================== #


class TestUpdateE2eWithDb:
    def test_update_persists_to_db(self, db_session, monkeypatch):
        """update() 走 DB backend，rows 累積。"""
        import src.data.database as db_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx(db_session))
        from src.regime import detector as det_mod

        monkeypatch.setattr(det_mod, "_DEFAULT_STATE_PATH", Path("/tmp/nonexistent_z.json"))

        sm = RegimeStateMachine()
        result = sm.update(_bull_closes())
        assert result["regime"] == "bull"

        rows = db_session.execute(select(RegimeStateLog)).scalars().all()
        assert len(rows) == 1
        assert rows[0].regime == "bull"
