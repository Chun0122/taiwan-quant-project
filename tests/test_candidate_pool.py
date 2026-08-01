"""B2 — 全候選池因子落庫測試（2026-08-01）。

背景（`logs/audit_discover_20260731/REPORT.md` §7.3）：實測漏斗
`1,857 →(流動性) 659 →(粗篩) 150 →(落庫) 20`，先前**只有最終 top-20 落庫**，
於是所有 IC 都算在 top-20 上（range restriction）——被評分的 150 檔只留分數
最高的 20 檔，因子變異被人為壓縮，IC 天生低估且不穩。這是 B11／E2b／E2c
三處噪音開關的共同上游。

涵蓋：
  A. 擷取時點：軟加成後、硬風控前 → 保有被硬閘門剔除者
  B. selected 標記與最終 top-N 一致
  C. 維度以真實語意落庫（不沿用 _post_score 的顯示別名）
  D. rankings 為空時**仍**落庫（M2 自鎖的資料面成因）
  E. load_candidate_factor_records 形狀符合 IC 計算期望
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import select

from src.data.schema import CandidateFactorLog
from src.discovery.scanner import DividendScanner, MomentumScanner, ValueScanner


class _Res:
    """最小 DiscoveryResult 替身（只需 _save_candidate_pool 用到的欄位）。"""

    def __init__(self, pool, scan_date=date(2026, 8, 1)):
        self.candidate_pool_df = pool
        self.scan_date = scan_date
        self.rankings = pd.DataFrame()


class _Scanner:
    regime = "crisis"


def _pool(n=5, selected_idx=(0, 1)):
    return pd.DataFrame(
        {
            "stock_id": [f"100{i}" for i in range(n)],
            "technical_score": [0.1 * i for i in range(n)],
            "chip_score": [0.5] * n,
            "fundamental_score": [0.6] * n,
            "news_score": [0.5] * n,
            "valuation_score": [0.7] * n,
            "dividend_score": [None] * n,
            "close": [100.0] * n,
            "composite_score": [0.9 - 0.1 * i for i in range(n)],
            "pool_rank": list(range(1, n + 1)),
            "selected": [i in selected_idx for i in range(n)],
        }
    )


@pytest.fixture()
def saver(db_session, monkeypatch):
    import src.data.database as db_mod

    class _Ctx:
        def __enter__(self):
            return db_session

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx())
    from src.cli.discover_cmd import _save_candidate_pool

    return _save_candidate_pool


def _rows(session, mode="value"):
    return (
        session.execute(
            select(CandidateFactorLog).where(CandidateFactorLog.mode == mode).order_by(CandidateFactorLog.pool_rank)
        )
        .scalars()
        .all()
    )


# ====================================================================== #
# A/B. 落庫涵蓋全池 + selected 標記
# ====================================================================== #


class TestPersistence:
    def test_saves_entire_pool_not_only_selected(self, db_session, saver):
        saver(_Res(_pool(n=5, selected_idx=(0, 1))), "value", _Scanner())
        rows = _rows(db_session)
        assert len(rows) == 5, "必須落庫全候選池，而非只有入選者"
        assert sum(r.selected for r in rows) == 2

    def test_selected_flag_and_pool_rank_preserved(self, db_session, saver):
        saver(_Res(_pool(n=4, selected_idx=(0, 3))), "value", _Scanner())
        rows = _rows(db_session)
        assert [r.pool_rank for r in rows] == [1, 2, 3, 4]
        assert [r.selected for r in rows] == [True, False, False, True]

    def test_rerun_replaces_same_day_same_mode(self, db_session, saver):
        saver(_Res(_pool(n=5)), "value", _Scanner())
        saver(_Res(_pool(n=3)), "value", _Scanner())
        assert len(_rows(db_session)) == 3, "重跑應覆蓋同日同模式，不累加"

    def test_other_mode_untouched(self, db_session, saver):
        saver(_Res(_pool(n=3)), "value", _Scanner())
        saver(_Res(_pool(n=2)), "momentum", _Scanner())
        assert len(_rows(db_session, "value")) == 3
        assert len(_rows(db_session, "momentum")) == 2

    def test_regime_and_provenance_recorded(self, db_session, saver):
        saver(_Res(_pool(n=2)), "value", _Scanner())
        r = _rows(db_session)[0]
        assert r.regime == "crisis"
        assert r.settings_hash, "A5 可重放：settings_hash 必須落庫"

    def test_empty_pool_is_noop(self, db_session, saver):
        saver(_Res(None), "value", _Scanner())
        saver(_Res(pd.DataFrame()), "value", _Scanner())
        assert _rows(db_session) == []

    def test_nan_scores_become_null(self, db_session, saver):
        pool = _pool(n=2)
        pool.loc[0, "chip_score"] = float("nan")
        saver(_Res(pool), "value", _Scanner())
        assert _rows(db_session)[0].chip_score is None


# ====================================================================== #
# C. 擷取時點與維度語意
# ====================================================================== #


class TestSnapshotSemantics:
    def test_snapshot_ranks_by_composite(self):
        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        scored = pd.DataFrame(
            {
                "stock_id": ["A", "B", "C"],
                "composite_score": [0.3, 0.9, 0.6],
                "technical_score": [0.1, 0.2, 0.3],
                "chip_score": [0.5, 0.5, 0.5],
                "fundamental_score": [0.5, 0.5, 0.5],
                "news_score": [0.5, 0.5, 0.5],
                "close": [10.0, 20.0, 30.0],
            }
        )
        out = s._snapshot_candidate_pool(scored)
        assert list(out["stock_id"]) == ["B", "C", "A"]
        assert list(out["pool_rank"]) == [1, 2, 3]
        assert (~out["selected"]).all(), "快照時尚未決定入選，一律 False"

    def test_snapshot_prefers_raw_technical_over_display_alias(self):
        """value/dividend 的 technical_score 是估值別名——落庫須取覆寫前的真值。

        `_post_score()` 在 composite 之後把 technical_score 覆寫為
        valuation/dividend 分數（顯示用）。實測 2026-08-01 value 模式 2615：
        DiscoveryRecord.technical_score = 0.7643 = 其 valuation_score。
        """
        s = ValueScanner(min_volume=1, use_ic_adjustment=False)
        scored = pd.DataFrame(
            {
                "stock_id": ["A"],
                "composite_score": [0.5],
                "technical_score": [0.99],  # 已被別名覆寫
                "technical_score_raw": [0.11],  # 覆寫前的真值
                "valuation_score": [0.99],
                "chip_score": [0.5],
                "fundamental_score": [0.5],
                "news_score": [0.5],
                "close": [10.0],
            }
        )
        out = s._snapshot_candidate_pool(scored)
        assert out.iloc[0]["technical_score"] == pytest.approx(0.11), "須取真實技術分"
        assert out.iloc[0]["valuation_score"] == pytest.approx(0.99), "估值另存專屬欄位"

    def test_post_score_preserves_raw_technical(self):
        for cls, extra in ((ValueScanner, "valuation_score"), (DividendScanner, "dividend_score")):
            s = cls(min_volume=1, use_ic_adjustment=False)
            df = pd.DataFrame({"stock_id": ["A"], "technical_score": [0.25], extra: [0.88]})
            out = s._post_score(df)
            assert out.iloc[0]["technical_score_raw"] == pytest.approx(0.25)
            assert out.iloc[0]["technical_score"] == pytest.approx(0.88), "顯示別名行為不變"

    def test_missing_columns_tolerated(self):
        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        out = s._snapshot_candidate_pool(pd.DataFrame({"stock_id": ["A"], "composite_score": [0.5]}))
        assert len(out) == 1
        assert out.iloc[0]["valuation_score"] is None

    def test_empty_input(self):
        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        assert s._snapshot_candidate_pool(pd.DataFrame()).empty
        assert s._snapshot_candidate_pool(None).empty


# ====================================================================== #
# D. rankings 為空時仍落庫（M2 自鎖的資料面成因）
# ====================================================================== #


class TestSavesEvenWhenNothingSelected:
    def test_pool_saved_when_all_filtered_out(self, db_session, saver):
        """硬閘門清空 top-N（如 swing 2026-07-28 剔除 88 支後落庫 0 筆）時，
        候選池**仍須落庫**——否則 IC 再也算不出來，正是 M2 自鎖的資料面成因。"""
        saver(_Res(_pool(n=6, selected_idx=())), "swing", _Scanner())
        rows = _rows(db_session, "swing")
        assert len(rows) == 6
        assert not any(r.selected for r in rows)


# ====================================================================== #
# E. IC 取數介面
# ====================================================================== #


class TestLoadForIC:
    def test_returns_ic_ready_shape(self, db_session, saver, monkeypatch):
        from src.discovery.ic_governance import load_candidate_factor_records

        saver(_Res(_pool(n=4, selected_idx=(0,))), "value", _Scanner())
        df = load_candidate_factor_records(db_session, "value", date(2026, 1, 1))
        assert len(df) == 4
        for col in ("scan_date", "stock_id", "close", "technical_score", "chip_score"):
            assert col in df.columns

    def test_selected_only_reproduces_truncated_sample(self, db_session, saver):
        """selected_only=True 等同舊 DiscoveryRecord 路徑，用於量化截斷偏差。"""
        from src.discovery.ic_governance import load_candidate_factor_records

        saver(_Res(_pool(n=5, selected_idx=(0, 1))), "value", _Scanner())
        full = load_candidate_factor_records(db_session, "value", date(2026, 1, 1))
        trunc = load_candidate_factor_records(db_session, "value", date(2026, 1, 1), selected_only=True)
        assert len(full) == 5
        assert len(trunc) == 2

    def test_empty_returns_typed_frame(self, db_session):
        from src.discovery.ic_governance import load_candidate_factor_records

        df = load_candidate_factor_records(db_session, "value", date(2026, 1, 1))
        assert df.empty
        assert "composite_score" in df.columns
