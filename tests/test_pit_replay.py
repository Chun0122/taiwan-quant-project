"""B1④ PIT 歷史重放測試（2026-08-04）。

重放的價值完全建立在「唯讀」與「不洩題」兩件事上：

  - **唯讀**：重放若寫進 `discovery_record` / `candidate_factor_log` /
    `universe_stat_log` / `regime_state_log`，事後就分不清哪些列是當日真實掃描、
    哪些是後來重跑的，live 資料的可信度反而被重放毀掉。
  - **不洩題**：前瞻報酬是**唯一**允許看 as_of 之後資料的地方，因為它是評分而
    非決策輸入。選股本身的任何一步看到未來，重放結果就是假的。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.discovery.pit_replay import (
    ReplayResult,
    compute_forward_returns,
    sample_replay_dates,
    summarize_replays,
)


def _picks(sids=("1101", "1102"), close=100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rank": list(range(1, len(sids) + 1)),
            "stock_id": list(sids),
            "stock_name": [f"股{s}" for s in sids],
            "close": [close] * len(sids),
            "composite_score": [0.9 - 0.1 * i for i in range(len(sids))],
        }
    )


@pytest.fixture()
def db(db_session, monkeypatch):
    import src.discovery.pit_replay as pr

    class _Ctx:
        def __enter__(self):
            return db_session

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(pr, "get_session", lambda: _Ctx())
    return db_session


def _seed(session, sid: str, start: date, closes: list[float]):
    from src.data.schema import DailyPrice

    d = start
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        session.add(DailyPrice(stock_id=sid, date=d, open=c, high=c, low=c, close=c, volume=1000, turnover=10000))
        d += timedelta(days=1)
    session.flush()


# ====================================================================== #
# A. 前瞻報酬（唯一允許看未來之處）
# ====================================================================== #


class TestForwardReturns:
    def test_uses_only_dates_after_as_of(self, db):
        """基準日當天與之前的價格不得計入前瞻報酬。"""
        as_of = date(2026, 3, 2)
        # as_of 當天 100；之後 110, 120, 130...
        _seed(db, "1101", as_of, [100.0] + [110.0, 120.0, 130.0, 140.0, 150.0])
        out = compute_forward_returns(_picks(("1101",), close=100.0), as_of, horizons=(1, 3))
        assert out.iloc[0]["fwd_1d"] == pytest.approx(10.0), "第 1 個交易日應是 110 → +10%"
        assert out.iloc[0]["fwd_3d"] == pytest.approx(30.0)

    def test_missing_future_data_is_nan(self, db):
        as_of = date(2026, 3, 2)
        _seed(db, "1101", as_of, [100.0, 110.0])  # 只有 1 天後續
        out = compute_forward_returns(_picks(("1101",)), as_of, horizons=(1, 20))
        assert out.iloc[0]["fwd_1d"] == pytest.approx(10.0)
        assert pd.isna(out.iloc[0]["fwd_20d"]), "資料不足應為 NaN 而非 0"

    def test_zero_volume_days_excluded(self, db):
        """停牌日（volume=0）不應被當成一個前瞻交易日。"""
        from src.data.schema import DailyPrice

        as_of = date(2026, 3, 2)
        db.add(
            DailyPrice(
                stock_id="1101", date=as_of + timedelta(days=1), open=1, high=1, low=1, close=1, volume=0, turnover=0
            )
        )
        db.flush()
        _seed(db, "1101", as_of + timedelta(days=2), [150.0])
        out = compute_forward_returns(_picks(("1101",), close=100.0), as_of, horizons=(1,))
        assert out.iloc[0]["fwd_1d"] == pytest.approx(50.0), "應跳過停牌日取下一個有量交易日"

    def test_empty_picks(self, db):
        assert compute_forward_returns(pd.DataFrame(), date(2026, 3, 2)).empty
        assert compute_forward_returns(None, date(2026, 3, 2)).empty

    def test_no_future_data_at_all(self, db):
        out = compute_forward_returns(_picks(("9999",)), date(2026, 3, 2), horizons=(5,))
        assert pd.isna(out.iloc[0]["fwd_5d"])


# ====================================================================== #
# B. 抽樣與彙總
# ====================================================================== #


class TestSamplingAndSummary:
    def test_sample_skips_thin_days(self, db, monkeypatch):
        """只取具備全市場覆蓋的交易日——半套日不能拿來重放。"""
        from src.constants import BACKFILL_MIN_COMMON_STOCKS
        from src.data.schema import DailyPrice

        full = date(2026, 3, 2)
        thin = date(2026, 3, 3)
        for i in range(BACKFILL_MIN_COMMON_STOCKS + 5):
            db.add(DailyPrice(stock_id=f"{1000 + i}", date=full, open=1, high=1, low=1, close=1, volume=1, turnover=1))
        for i in range(10):
            db.add(DailyPrice(stock_id=f"{1000 + i}", date=thin, open=1, high=1, low=1, close=1, volume=1, turnover=1))
        db.flush()
        got = sample_replay_dates(date(2026, 3, 1), date(2026, 3, 31), every_n_days=1)
        assert got == [full]

    def test_every_n_days_sampling(self, db):
        from src.constants import BACKFILL_MIN_COMMON_STOCKS
        from src.data.schema import DailyPrice

        days = [date(2026, 3, 2) + timedelta(days=i) for i in range(5)]
        for d in days:
            for i in range(BACKFILL_MIN_COMMON_STOCKS + 1):
                db.add(DailyPrice(stock_id=f"{1000 + i}", date=d, open=1, high=1, low=1, close=1, volume=1, turnover=1))
        db.flush()
        got = sample_replay_dates(days[0], days[-1], every_n_days=2)
        assert got == [days[0], days[2], days[4]]

    def test_summarize_computes_win_rate(self):
        frames = [
            _picks(("1101", "1102")).assign(fwd_5d=[10.0, -5.0]),
            _picks(("1103", "1104")).assign(fwd_5d=[3.0, float("nan")]),
        ]
        s = summarize_replays(frames, horizons=(5,))
        row = s.iloc[0]
        assert row["n"] == 3, "NaN 不計入樣本"
        assert row["win_rate"] == pytest.approx(2 / 3)
        assert row["best"] == pytest.approx(10.0)
        assert row["worst"] == pytest.approx(-5.0)

    def test_summarize_empty(self):
        assert summarize_replays([]).empty


# ====================================================================== #
# C. 唯讀契約
# ====================================================================== #


class TestReadOnlyContract:
    def test_replay_writes_nothing_to_live_tables(self, monkeypatch):
        """重放不得寫入任何 live 資料表。

        以原始碼契約檢查：`replay_scan` 只呼叫 `scanner.run(as_of=...)`，
        絕不呼叫 `_save_discovery_records` / `_save_candidate_pool`。
        run() 內部的 universe_stat_log 與 regime 狀態機各自有 offline/唯讀保護
        （見 `_get_universe_ids` 與 `MarketRegimeDetector.detect`）。
        """
        import inspect

        from src.discovery import pit_replay

        src = inspect.getsource(pit_replay)
        for forbidden in ("_save_discovery_records", "_save_candidate_pool", "log_universe_stats"):
            assert forbidden not in src, f"PIT 重放不得呼叫 {forbidden}"

    def test_universe_stats_not_logged_when_offline(self):
        """`_get_universe_ids` 在 offline（PIT）時提前返回，不寫 universe_stat_log。"""
        import inspect

        from src.discovery.scanner._base import MarketScanner

        src = inspect.getsource(MarketScanner._get_universe_ids)
        idx_guard = src.index("_is_offline()")
        idx_log = src.index("log_universe_stats(")
        assert idx_guard < idx_log, "offline 守門必須在寫入之前"

    def test_replay_result_shape(self):
        r = ReplayResult(as_of=date(2026, 3, 2), mode="value", regime="crisis", total_stocks=10, after_coarse=5)
        assert r.n_picks == 0
        r.picks = _picks()
        assert r.n_picks == 2
