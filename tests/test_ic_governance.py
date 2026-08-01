"""P0 #16 / B11 最小切口 — IC 執法治理測試（2026-08-01）。

背景（`logs/audit_discover_20260731/REPORT.md` §2）：M2 自動停用的實際判準是
「rolling IC 算不算得出來」而非「因子有沒有效」：

  - 模式停掃 → `max_date` 凍結 → rolling 窗口凍結 → `iloc[-1]` 每天重讀同一份
    過期 IC 執法（實測 2026-07-31 用的是 `window_end=2026-06-27`、n=40、34 天前）
  - 記錄跨度不足 → 窗口迴圈零次 → `rolling_df` 空 → fail-open 放行
  - `docs/MASTER_PLAN.md` §3 原則 6（n≥100、跨 ≥3 掃描週）從未被程式執行

涵蓋：
  A. select_enforceable_ic 三道閘門（時效 / 窗口數 / 樣本）
  B. 決策 IC = 最近 3 窗平均（而非單一最新窗）
  C. **核心回歸**：2026-07-31 momentum 真實情境不得執法
  D. is_inverse 僅在可執法時為真
  E. resolve_rankings exclude_modes（單一 / composite）
  F. rotation 層阻擋新買入（整合）
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.constants import DISCOVERY_IC_MIN_SAMPLES, DISCOVERY_IC_MIN_WINDOWS
from src.discovery.ic_governance import (
    REASON_INSUFFICIENT_SAMPLES,
    REASON_INSUFFICIENT_WINDOWS,
    REASON_NO_WINDOWS,
    REASON_OK,
    REASON_STALE_WINDOW,
    select_enforceable_ic,
    staleness_budget_days,
)

AS_OF = date(2026, 7, 31)


def _rolling(
    factor: str = "chip_score",
    *,
    n_windows: int = 4,
    last_window_end: date = AS_OF,
    ic: float | list[float] = -0.10,
    sample_count: int | list[int] = 200,
    step_days: int = 7,
) -> pd.DataFrame:
    """建構 compute_rolling_ic() 形狀的 DataFrame（由舊至新）。"""
    ics = ic if isinstance(ic, list) else [ic] * n_windows
    ns = sample_count if isinstance(sample_count, list) else [sample_count] * n_windows
    rows = []
    for i in range(n_windows):
        offset = (n_windows - 1 - i) * step_days
        rows.append(
            {
                "window_end": last_window_end - timedelta(days=offset),
                "factor": factor,
                "ic": ics[i],
                "sample_count": ns[i],
                "direction": "effective",
            }
        )
    return pd.DataFrame(rows)


def _verdict(**kw):
    params = {"factor": "chip_score", "as_of": AS_OF, "holding_days": 5, "mode": "momentum"}
    params.update(kw)
    df = params.pop("rolling_df", None)
    return select_enforceable_ic(df, params.pop("factor"), **params)


# ====================================================================== #
# A. 三道閘門
# ====================================================================== #


class TestGates:
    def test_healthy_series_is_enforceable(self):
        v = _verdict(rolling_df=_rolling())
        assert v.enforceable is True
        assert v.reason == REASON_OK
        assert v.ic == pytest.approx(-0.10)

    def test_stale_window_blocks_enforcement(self):
        """窗口過期 → 不得執法（P0 #16 核心）。"""
        budget = staleness_budget_days(5)  # 5 + 14 = 19
        v = _verdict(rolling_df=_rolling(last_window_end=AS_OF - timedelta(days=budget + 1)))
        assert v.enforceable is False
        assert v.reason == REASON_STALE_WINDOW
        assert v.staleness_days == budget + 1
        assert v.staleness_budget_days == budget
        # 仍回報觀測到的 IC —— 可告警，只是不可行動
        assert v.ic is not None

    def test_exactly_at_budget_still_enforceable(self):
        budget = staleness_budget_days(5)
        v = _verdict(rolling_df=_rolling(last_window_end=AS_OF - timedelta(days=budget)))
        assert v.enforceable is True

    def test_staleness_budget_scales_with_holding_days(self):
        """holding=20 的模式窗口天生落後更多，預算須跟著放寬。"""
        assert staleness_budget_days(5) == 19
        assert staleness_budget_days(20) == 34
        stale_for_momentum = AS_OF - timedelta(days=25)
        assert _verdict(rolling_df=_rolling(last_window_end=stale_for_momentum)).enforceable is False
        v20 = _verdict(
            rolling_df=_rolling(factor="fundamental_score", last_window_end=stale_for_momentum),
            factor="fundamental_score",
            holding_days=20,
        )
        assert v20.enforceable is True

    def test_insufficient_windows_blocks_enforcement(self):
        v = _verdict(rolling_df=_rolling(n_windows=DISCOVERY_IC_MIN_WINDOWS - 1))
        assert v.enforceable is False
        assert v.reason == REASON_INSUFFICIENT_WINDOWS

    def test_insufficient_samples_blocks_enforcement(self):
        """取決策窗口中的最小樣本數（最保守）。"""
        v = _verdict(rolling_df=_rolling(n_windows=4, sample_count=[300, 300, 300, 60]))
        assert v.enforceable is False
        assert v.reason == REASON_INSUFFICIENT_SAMPLES
        assert v.min_sample_count == 60

    def test_samples_exactly_at_threshold_enforceable(self):
        v = _verdict(rolling_df=_rolling(sample_count=DISCOVERY_IC_MIN_SAMPLES))
        assert v.enforceable is True

    def test_empty_or_missing_rolling_df(self):
        for df in (None, pd.DataFrame()):
            v = _verdict(rolling_df=df)
            assert v.enforceable is False
            assert v.reason == REASON_NO_WINDOWS

    def test_factor_absent_from_rolling_df(self):
        v = _verdict(rolling_df=_rolling(factor="news_score"))
        assert v.enforceable is False
        assert v.reason == REASON_NO_WINDOWS


# ====================================================================== #
# B. 決策 IC = 最近 N 窗平均
# ====================================================================== #


class TestDecisionStatistic:
    def test_decision_ic_averages_last_three_windows(self):
        """單一最新窗易受小樣本漂移；取最近 3 窗平均。

        取材自 2026-07-31 swing 實況：前三窗 +0.005，最新兩窗 -0.055/-0.093，
        單窗判定 -0.093 會停用，3 窗平均 -0.048 未達 -0.05 門檻。
        """
        df = _rolling(
            factor="fundamental_score",
            n_windows=5,
            ic=[0.005, 0.005, 0.005, -0.0553, -0.0933],
            sample_count=[280, 280, 280, 200, 140],
        )
        v = select_enforceable_ic(df, "fundamental_score", as_of=AS_OF, holding_days=10, mode="swing")
        assert v.enforceable is True
        assert v.latest_window_ic == pytest.approx(-0.0933)
        assert v.ic == pytest.approx((0.005 - 0.0553 - 0.0933) / 3)
        assert v.is_inverse is False, "3 窗平均未達 -0.05，不應判定反向"

    def test_latest_window_ic_reported_separately(self):
        df = _rolling(n_windows=3, ic=[0.2, 0.0, -0.3])
        v = _verdict(rolling_df=df)
        assert v.latest_window_ic == pytest.approx(-0.3)
        assert v.ic == pytest.approx((0.2 + 0.0 - 0.3) / 3)


# ====================================================================== #
# C. 核心回歸：2026-07-31 momentum 真實情境
# ====================================================================== #


class TestMomentumRealWorldRegression:
    def test_frozen_june_window_must_not_enforce(self):
        """實測情境：停用 momentum 用的是 window_end=2026-06-27、n=40、距今 34 天。

        修復前 `iloc[-1]` 直接執法 → 每天以同一份六月觀測停用 momentum，
        而 momentum 停掃後永遠產不出新記錄 → 自鎖。三道閘門全部命中。
        """
        df = _rolling(
            n_windows=2,
            last_window_end=date(2026, 6, 27),
            ic=[-0.1273, -0.1109],
            sample_count=[96, 40],
        )
        v = select_enforceable_ic(df, "chip_score", as_of=AS_OF, holding_days=5, mode="momentum")

        assert v.enforceable is False
        assert v.is_inverse is False, "過期 IC 不得停用模式（自鎖迴路的根因）"
        assert v.reason == REASON_STALE_WINDOW
        assert v.staleness_days == 34
        # 觀測值仍在，供告警顯示
        assert v.latest_window_ic == pytest.approx(-0.1109)
        assert "34 天" in v.describe()


# ====================================================================== #
# D. is_inverse 語意
# ====================================================================== #


class TestIsInverse:
    def test_inverse_requires_enforceable(self):
        stale = _verdict(rolling_df=_rolling(ic=-0.5, last_window_end=AS_OF - timedelta(days=99)))
        assert stale.ic is not None and stale.ic < -0.05
        assert stale.is_inverse is False

    def test_inverse_true_when_all_gates_pass(self):
        assert _verdict(rolling_df=_rolling(ic=-0.20)).is_inverse is True

    def test_positive_ic_not_inverse(self):
        assert _verdict(rolling_df=_rolling(ic=0.20)).is_inverse is False

    def test_threshold_is_minus_005(self):
        """−0.05 為反向門檻：略高不算、略低才算。

        刻意不測「恰為 −0.05」——決策 IC 是多窗平均，浮點累積誤差會讓
        「四個 −0.05 的平均」變成 −0.050000000000000010，斷言等值屬脆弱測試。
        """
        assert _verdict(rolling_df=_rolling(ic=-0.049)).is_inverse is False
        assert _verdict(rolling_df=_rolling(ic=-0.051)).is_inverse is True


# ====================================================================== #
# E. resolve_rankings exclude_modes
# ====================================================================== #


class TestResolveRankingsExcludeModes:
    def _seed(self, session, scan_date: date, mode: str, sids: list[str]) -> None:
        from src.data.schema import DiscoveryRecord

        for i, sid in enumerate(sids, start=1):
            session.add(
                DiscoveryRecord(
                    scan_date=scan_date,
                    stock_id=sid,
                    stock_name=f"股{sid}",
                    mode=mode,
                    rank=i,
                    composite_score=0.9 - i * 0.01,
                    close=100.0,
                    regime="bull",
                )
            )
        session.flush()

    def test_excluded_single_mode_returns_empty(self, db_session):
        from src.portfolio.rankings import resolve_rankings

        sd = date(2026, 7, 31)
        self._seed(db_session, sd, "momentum", ["1101", "1102"])

        assert len(resolve_rankings("momentum", sd, db_session)) == 2
        assert resolve_rankings("momentum", sd, db_session, exclude_modes={"momentum"}) == []

    def test_other_mode_unaffected(self, db_session):
        from src.portfolio.rankings import resolve_rankings

        sd = date(2026, 7, 31)
        self._seed(db_session, sd, "momentum", ["1101"])
        self._seed(db_session, sd, "value", ["2201"])

        got = resolve_rankings("value", sd, db_session, exclude_modes={"momentum"})
        assert [r["stock_id"] for r in got] == ["2201"]

    def test_composite_drops_excluded_member_only(self, db_session):
        from src.portfolio.rankings import resolve_rankings

        sd = date(2026, 7, 31)
        self._seed(db_session, sd, "momentum", ["1101"])
        self._seed(db_session, sd, "growth", ["2201"])

        got = resolve_rankings("mom_growth", sd, db_session, exclude_modes={"momentum"})
        assert [r["stock_id"] for r in got] == ["2201"], "只應濾掉被擋成員，另一成員照常"

    def test_composite_all_members_excluded_returns_empty(self, db_session):
        from src.portfolio.rankings import resolve_rankings

        sd = date(2026, 7, 31)
        self._seed(db_session, sd, "momentum", ["1101"])
        self._seed(db_session, sd, "growth", ["2201"])

        assert resolve_rankings("mom_growth", sd, db_session, exclude_modes={"momentum", "growth"}) == []


# ====================================================================== #
# F. rotation 層阻擋（整合）
# ====================================================================== #


class TestRotationIcBlock:
    def test_blocked_mode_yields_no_rankings(self, db_session, monkeypatch):
        """IC 反向且可執法 → _resolve_ic_blocked_modes 回傳該模式 → 排名為空。"""
        from src.portfolio.manager import RotationManager

        mgr = RotationManager("t")
        monkeypatch.setattr(
            "src.discovery.ic_governance.blocked_modes",
            lambda session, modes, as_of: {"momentum": _verdict(rolling_df=_rolling(ic=-0.2))},
        )
        excluded = mgr._resolve_ic_blocked_modes(db_session, "momentum", date(2026, 7, 31))
        assert excluded == {"momentum"}

    def test_composite_expands_to_members(self, db_session, monkeypatch):
        from src.portfolio.manager import RotationManager

        seen: dict = {}

        def _fake(session, modes, as_of):
            seen["modes"] = list(modes)
            return {}

        monkeypatch.setattr("src.discovery.ic_governance.blocked_modes", _fake)
        RotationManager("t")._resolve_ic_blocked_modes(db_session, "mom_growth", date(2026, 7, 31))
        assert set(seen["modes"]) == {"momentum", "growth"}

    def test_failure_fails_open(self, db_session, monkeypatch):
        """判定失敗不得凍結交易（fail-open）。"""
        from src.portfolio.manager import RotationManager

        def _boom(session, modes, as_of):
            raise RuntimeError("DB 掛了")

        monkeypatch.setattr("src.discovery.ic_governance.blocked_modes", _boom)
        assert RotationManager("t")._resolve_ic_blocked_modes(db_session, "momentum", date(2026, 7, 31)) == set()

    def test_result_is_cached_per_mode_and_date(self, db_session, monkeypatch):
        from src.portfolio.manager import RotationManager

        calls = {"n": 0}

        def _counting(session, modes, as_of):
            calls["n"] += 1
            return {}

        monkeypatch.setattr("src.discovery.ic_governance.blocked_modes", _counting)
        mgr = RotationManager("t")
        for _ in range(3):
            mgr._resolve_ic_blocked_modes(db_session, "momentum", date(2026, 7, 31))
        assert calls["n"] == 1
        mgr._resolve_ic_blocked_modes(db_session, "momentum", date(2026, 8, 1))
        assert calls["n"] == 2, "不同決策日應重算"
