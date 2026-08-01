"""P0 #15 — Regime 狀態機冪等性測試（2026-08-01）。

背景（`logs/audit_discover_20260731/REPORT.md` §3）：
`RegimeStateMachine.update()` 舊版的同日快取條件含 `self._cached_result is not None`，
而該欄位是 **per-instance**；所有呼叫端都寫成 `MarketRegimeDetector().detect()`
（每次全新實例）→ 快取從未命中 → 每次呼叫都重跑 `apply_hysteresis()` 並 append
一筆 `RegimeStateLog`。實測每日 4~15 次呼叫，導致：
  ① 同一批 discover 掃描中各模式看到不同 regime（2026-07-24 實測 sideways↔bear 來回）
  ② 模式當天跑不跑取決於它排在第幾個呼叫
  ③ hysteresis 確認計數按「呼叫次數」而非「天數」消耗 → 遲滯機制被架空

修法：改以「本次判定依據的最新資料日」(`RegimeState.data_date`) 為冪等鍵。

涵蓋：
  A. resolve_data_date 純函數（各種 index 型別）
  B. 同一資料日重複呼叫（跨實例）不推進狀態機、不寫 log
  C. **核心回歸**：3 次呼叫不得湊滿 3 天確認天數而誤觸轉換
  D. 新資料日推進一次
  E. 舊資料（data_date=None）相容：推進一次後轉為冪等
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from src.data.schema import RegimeStateLog
from src.regime.detector import RegimeState, RegimeStateMachine, resolve_data_date


class _Ctx:
    """把 conftest 的 db_session 包成 context manager（detector 用 `with get_session()`）。"""

    def __init__(self, s):
        self._s = s

    def __enter__(self):
        return self._s

    def __exit__(self, *a):
        return False


@pytest.fixture()
def db_backend(db_session, monkeypatch):
    """DB 模式的 RegimeStateMachine 環境：包裝 session + 停用 legacy JSON 遷移。"""
    import src.data.database as db_mod
    from src.regime import detector as det_mod

    monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx(db_session))
    monkeypatch.setattr(det_mod, "_DEFAULT_STATE_PATH", Path("/tmp/nonexistent_regime_idem.json"))
    return db_session


def _dated_closes(last_day: str, values: list[float]) -> pd.Series:
    """建立 date index 的收盤序列，最後一天為 last_day（往前逐日回推）。"""
    end = datetime.date.fromisoformat(last_day)
    idx = [end - datetime.timedelta(days=len(values) - 1 - i) for i in range(len(values))]
    return pd.Series(values, index=idx)


def _rising(last_day: str, n: int = 130, start: float = 15000.0, step: float = 25.0) -> pd.Series:
    """穩定上升序列 → raw regime = bull，且滿足 sideways→bull 的 close > SMA60×1.01。"""
    return _dated_closes(last_day, [start + i * step for i in range(n)])


def _seed_state(session, regime: str, data_date: str | None, confirmation_count: int = 0) -> None:
    session.add(
        RegimeStateLog(
            regime=regime,
            regime_since="2026-07-01",
            confirmation_count=confirmation_count,
            pending_transition=None,
            last_updated="2026-07-30",
            data_date=data_date,
        )
    )
    session.flush()


def _rows(session) -> list[RegimeStateLog]:
    return session.execute(select(RegimeStateLog).order_by(RegimeStateLog.id)).scalars().all()


# ====================================================================== #
# A. resolve_data_date 純函數
# ====================================================================== #


class TestResolveDataDate:
    def test_date_index(self):
        s = pd.Series([1, 2], index=[datetime.date(2026, 7, 30), datetime.date(2026, 7, 31)])
        assert resolve_data_date(s) == "2026-07-31"

    def test_datetime_index(self):
        s = pd.Series([1, 2], index=[datetime.datetime(2026, 7, 30, 9), datetime.datetime(2026, 7, 31, 13, 30)])
        assert resolve_data_date(s) == "2026-07-31"

    def test_pandas_timestamp_index(self):
        """pd.Timestamp 是 datetime 子類別 → 走 datetime 分支取 date 部分。"""
        s = pd.Series([1, 2], index=pd.to_datetime(["2026-07-30", "2026-07-31"]))
        assert resolve_data_date(s) == "2026-07-31"

    def test_iso_string_index(self):
        s = pd.Series([1, 2], index=["2026-07-30", "2026-07-31"])
        assert resolve_data_date(s) == "2026-07-31"

    def test_integer_index_returns_none(self):
        """純函數測試常用整數 index → 無法解析，退回舊行為（每次推進）。"""
        assert resolve_data_date(pd.Series([1, 2, 3])) is None

    def test_empty_series_returns_none(self):
        assert resolve_data_date(pd.Series(dtype=float)) is None

    def test_none_returns_none(self):
        assert resolve_data_date(None) is None


# ====================================================================== #
# B. 同一資料日重複呼叫 → 不推進、不寫 log
# ====================================================================== #


class TestIdempotentSameDataDate:
    def test_repeat_calls_across_instances_write_one_row(self, db_backend):
        """跨實例重複呼叫同一份資料，只寫一筆 RegimeStateLog。

        這是 P0 #15 的核心：修復前每個 scanner 各建新實例 → 每次都寫一筆。
        """
        closes = _rising("2026-07-31")

        first = RegimeStateMachine().update(closes)
        assert first["regime"] == "bull"
        assert first["state_advanced"] is True
        assert len(_rows(db_backend)) == 1

        # 模擬 morning-routine 一次 run 內的 5 個 scanner + rotation
        for _ in range(6):
            again = RegimeStateMachine().update(closes)
            assert again["regime"] == first["regime"]
            assert again["state_advanced"] is False

        assert len(_rows(db_backend)) == 1, "同一資料日不得重複寫入狀態 log"

    def test_idempotent_path_preserves_signal_fields(self, db_backend):
        """冪等路徑仍回傳完整訊號欄位（非只有 regime）。"""
        closes = _rising("2026-07-31")
        first = RegimeStateMachine().update(closes)
        cached = RegimeStateMachine().update(closes)

        assert cached["taiex_close"] == first["taiex_close"]
        assert cached["signals"] == first["signals"]
        assert cached["raw_regime"] == first["raw_regime"]
        assert cached["transition_info"]["reason"] == "idempotent_same_data_date"

    def test_data_date_persisted(self, db_backend):
        RegimeStateMachine().update(_rising("2026-07-31"))
        assert _rows(db_backend)[0].data_date == "2026-07-31"


# ====================================================================== #
# C. 核心回歸：確認天數不得被同日重複呼叫湊滿
# ====================================================================== #


class TestHysteresisNotConsumedPerCall:
    def test_three_calls_same_day_do_not_confirm_three_day_transition(self, db_backend):
        """sideways→bull 需 3 天確認；同一天呼叫 3 次不得完成轉換。

        修復前：每次呼叫 confirmation_count +1，第 3 次呼叫即 3>=3 → 誤轉為 bull，
        等於「一個早上走完三天的確認流程」。
        """
        _seed_state(db_backend, regime="sideways", data_date="2026-07-30", confirmation_count=0)
        closes = _rising("2026-07-31")

        results = [RegimeStateMachine().update(closes) for _ in range(3)]

        assert [r["regime"] for r in results] == ["sideways"] * 3, "同日重複呼叫不得湊滿確認天數"
        assert [r["state_advanced"] for r in results] == [True, False, False]

        latest = _rows(db_backend)[-1]
        assert latest.regime == "sideways"
        assert latest.confirmation_count == 1, "確認計數應只推進一次（按天，不按呼叫次數）"
        assert latest.pending_transition == "sideways→bull"

    def test_transition_confirms_across_three_data_dates(self, db_backend):
        """換成三個不同資料日，轉換仍應如設計般於第 3 天完成。"""
        _seed_state(db_backend, regime="sideways", data_date="2026-07-28", confirmation_count=0)

        regimes = [
            RegimeStateMachine().update(_rising(d))["regime"] for d in ("2026-07-29", "2026-07-30", "2026-07-31")
        ]

        assert regimes == ["sideways", "sideways", "bull"], "跨資料日的確認流程不應被破壞"


# ====================================================================== #
# D/E. 新資料日推進 + 舊資料相容
# ====================================================================== #


class TestAdvanceAndLegacy:
    def test_new_data_date_advances_once(self, db_backend):
        RegimeStateMachine().update(_rising("2026-07-30"))
        RegimeStateMachine().update(_rising("2026-07-30"))
        assert len(_rows(db_backend)) == 1

        r = RegimeStateMachine().update(_rising("2026-07-31"))
        assert r["state_advanced"] is True
        rows = _rows(db_backend)
        assert len(rows) == 2
        assert [row.data_date for row in rows] == ["2026-07-30", "2026-07-31"]

    def test_legacy_null_data_date_advances_then_becomes_idempotent(self, db_backend):
        """migration 前的舊列 data_date=NULL → 重新判定一次，之後轉為冪等。"""
        _seed_state(db_backend, regime="bull", data_date=None)
        closes = _rising("2026-07-31")

        first = RegimeStateMachine().update(closes)
        assert first["state_advanced"] is True

        second = RegimeStateMachine().update(closes)
        assert second["state_advanced"] is False
        assert len([r for r in _rows(db_backend) if r.data_date == "2026-07-31"]) == 1

    def test_unparseable_index_falls_back_to_old_behavior(self, db_backend):
        """整數 index（既有純函數測試）→ 無冪等鍵，維持每次推進的舊行為。"""
        closes = pd.Series([15000 + i * 25 for i in range(130)])

        RegimeStateMachine().update(closes)
        RegimeStateMachine().update(closes)

        rows = _rows(db_backend)
        assert len(rows) == 2
        assert all(r.data_date is None for r in rows)


# ====================================================================== #
# F. JSON backend 亦具同一保證
# ====================================================================== #


class TestJsonBackendIdempotence:
    def test_json_mode_same_data_date_does_not_advance(self, tmp_path):
        path = tmp_path / "regime_state.json"
        closes = _rising("2026-07-31")

        first = RegimeStateMachine(state_path=path).update(closes)
        assert first["state_advanced"] is True

        second = RegimeStateMachine(state_path=path).update(closes)
        assert second["state_advanced"] is False
        assert second["regime"] == first["regime"]

    def test_json_roundtrip_carries_data_date(self, tmp_path):
        path = tmp_path / "regime_state.json"
        sm = RegimeStateMachine(state_path=path)
        sm._save_state(
            RegimeState(
                regime="bear",
                regime_since="2026-07-01",
                confirmation_count=2,
                pending_transition=None,
                last_updated="2026-07-31",
                data_date="2026-07-31",
            )
        )
        loaded = RegimeStateMachine(state_path=path)._load_state()
        assert loaded is not None
        assert loaded.data_date == "2026-07-31"


# ====================================================================== #
# G. morning-routine Step 8e：同步後 regime 重解
# ====================================================================== #


class TestResolveRegimeAfterSync:
    """P0 #15 的第二部分：Step 0 在同步前執行，Step 12 不得沿用其過期 regime。"""

    def test_returns_refreshed_regime(self, monkeypatch):
        from src.cli import morning_cmd
        from src.regime import detector as det_mod

        monkeypatch.setattr(det_mod, "MarketRegimeDetector", lambda: _FakeDetector({"regime": "crisis"}))
        assert morning_cmd.resolve_regime_after_sync("bear") == "crisis"

    def test_falls_back_to_previous_on_exception(self, monkeypatch):
        """重解失敗不得讓 rotation 失去 regime（退回 Step 0 結果）。"""
        from src.cli import morning_cmd
        from src.regime import detector as det_mod

        def _boom():
            raise RuntimeError("DB 掛了")

        monkeypatch.setattr(det_mod, "MarketRegimeDetector", _boom)
        assert morning_cmd.resolve_regime_after_sync("bear") == "bear"

    def test_falls_back_when_refreshed_is_empty(self, monkeypatch):
        from src.cli import morning_cmd
        from src.regime import detector as det_mod

        monkeypatch.setattr(det_mod, "MarketRegimeDetector", lambda: _FakeDetector({"regime": None}))
        assert morning_cmd.resolve_regime_after_sync("sideways") == "sideways"

    def test_previous_none_still_takes_refreshed(self, monkeypatch):
        """Step 0 預檢失敗（None）時，同步後重解仍應補上 regime。"""
        from src.cli import morning_cmd
        from src.regime import detector as det_mod

        monkeypatch.setattr(det_mod, "MarketRegimeDetector", lambda: _FakeDetector({"regime": "bull"}))
        assert morning_cmd.resolve_regime_after_sync(None) == "bull"


class _FakeDetector:
    def __init__(self, result: dict):
        self._result = result

    def detect(self) -> dict:
        return self._result
