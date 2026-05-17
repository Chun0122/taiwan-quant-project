"""P1 任務 5 — 進場理由 JSON 落 RotationPosition 測試。

涵蓋：
  P5-A 純函數 _record_to_score_breakdown（含 mode='all' primary_mode 案例）
  P5-B resolve_rankings 回傳含 score_breakdown
  P5-C _resolve_all_mode_rankings 回傳含 mode_scores + primary_mode breakdown
  P5-D _execute_buy 透過 to_buy.score_breakdown 序列化 JSON 寫入 RotationPosition
  P5-E backfill_entry_score_breakdown 重建歷史 NULL 欄位
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from sqlalchemy import select

from src.data.schema import DiscoveryRecord, RotationPortfolio, RotationPosition
from src.portfolio.manager import (
    RotationManager,
    _record_to_score_breakdown,
    _resolve_all_mode_rankings,
    resolve_rankings,
)

# ====================================================================== #
# 共用 helpers
# ====================================================================== #


def _make_disc(
    db_session,
    *,
    scan_date: date,
    mode: str,
    stock_id: str,
    rank: int,
    composite: float = 0.5,
    chip: float | None = None,
    tech: float | None = None,
    fund: float | None = None,
    news: float | None = None,
    regime: str | None = "bull",
    chip_tier: str | None = None,
    concept_bonus: float | None = None,
    daytrade_penalty: float | None = None,
    close: float = 100.0,
) -> DiscoveryRecord:
    r = DiscoveryRecord(
        scan_date=scan_date,
        mode=mode,
        rank=rank,
        stock_id=stock_id,
        stock_name=f"name_{stock_id}",
        close=close,
        composite_score=composite,
        technical_score=tech,
        chip_score=chip,
        fundamental_score=fund,
        news_score=news,
        regime=regime,
        chip_tier=chip_tier,
        concept_bonus=concept_bonus,
        daytrade_penalty=daytrade_penalty,
    )
    db_session.add(r)
    db_session.flush()
    return r


def _make_portfolio(db_session, *, name: str, mode: str = "momentum") -> RotationPortfolio:
    p = RotationPortfolio(
        name=name,
        mode=mode,
        max_positions=5,
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


# ====================================================================== #
# P5-A: _record_to_score_breakdown 純函數
# ====================================================================== #


class TestRecordToScoreBreakdown:
    def test_basic_fields_serialized(self, db_session):
        r = _make_disc(
            db_session,
            scan_date=date(2026, 5, 15),
            mode="momentum",
            stock_id="2330",
            rank=1,
            composite=0.85,
            chip=0.7,
            tech=0.9,
            fund=0.6,
            news=0.4,
            chip_tier="7F",
            concept_bonus=0.05,
        )
        out = _record_to_score_breakdown(r)
        assert out["scan_date"] == "2026-05-15"
        assert out["mode"] == "momentum"
        assert out["rank"] == 1
        assert out["composite_score"] == pytest.approx(0.85)
        assert out["regime"] == "bull"
        assert out["scores"] == {"chip": 0.7, "technical": 0.9, "fundamental": 0.6, "news": 0.4}
        assert out["chip_tier"] == "7F"
        assert out["concept_bonus"] == pytest.approx(0.05)
        assert out["discovery_record_id"] == r.id
        assert "primary_mode" not in out  # 非 all 模式不帶 primary_mode

    def test_primary_mode_added_for_all_mode(self, db_session):
        r = _make_disc(db_session, scan_date=date(2026, 5, 15), mode="swing", stock_id="2330", rank=1)
        out = _record_to_score_breakdown(r, primary_mode="swing")
        assert out["primary_mode"] == "swing"

    def test_json_serializable(self, db_session):
        """輸出必須可被 json.dumps 序列化（不含 datetime 等非 JSON 型別）。"""
        r = _make_disc(db_session, scan_date=date(2026, 5, 15), mode="momentum", stock_id="2330", rank=1)
        out = _record_to_score_breakdown(r)
        s = json.dumps(out, ensure_ascii=False)
        assert "2026-05-15" in s

    def test_null_scores_preserved(self, db_session):
        """部分 score 為 None 時不擲例外，保留 None。"""
        r = _make_disc(
            db_session,
            scan_date=date(2026, 5, 15),
            mode="momentum",
            stock_id="2330",
            rank=1,
            chip=None,
            tech=None,
            fund=None,
            news=None,
        )
        out = _record_to_score_breakdown(r)
        assert out["scores"] == {"chip": None, "technical": None, "fundamental": None, "news": None}


# ====================================================================== #
# P5-B: resolve_rankings 回傳含 score_breakdown
# ====================================================================== #


class TestResolveRankingsIncludesBreakdown:
    def test_single_mode_returns_breakdown_per_row(self, db_session):
        _make_disc(db_session, scan_date=date(2026, 5, 15), mode="momentum", stock_id="2330", rank=1, composite=0.9)
        _make_disc(db_session, scan_date=date(2026, 5, 15), mode="momentum", stock_id="2317", rank=2, composite=0.8)
        db_session.commit()

        rankings = resolve_rankings("momentum", date(2026, 5, 15), db_session, top_n=10)
        assert len(rankings) == 2
        for r in rankings:
            assert "score_breakdown" in r
            assert r["score_breakdown"]["mode"] == "momentum"
            assert r["score_breakdown"]["composite_score"] is not None

    def test_no_records_returns_empty(self, db_session):
        rankings = resolve_rankings("momentum", date(2026, 5, 15), db_session, top_n=10)
        assert rankings == []


# ====================================================================== #
# P5-C: _resolve_all_mode_rankings 多模式 breakdown
# ====================================================================== #


class TestResolveAllModeRankingsBreakdown:
    def test_primary_mode_picked_from_max_score(self, db_session):
        """同一股票出現在 momentum/swing 兩個 mode，primary_mode = 較高分的那個。"""
        _make_disc(db_session, scan_date=date(2026, 5, 15), mode="momentum", stock_id="2330", rank=1, composite=0.6)
        _make_disc(db_session, scan_date=date(2026, 5, 15), mode="swing", stock_id="2330", rank=1, composite=0.9)
        db_session.commit()

        rankings = _resolve_all_mode_rankings(date(2026, 5, 15), db_session, top_n=10, per_mode_max=0)
        assert len(rankings) == 1
        r = rankings[0]
        assert r["primary_mode"] == "swing"
        bd = r["score_breakdown"]
        assert bd["mode"] == "all"
        assert bd["primary_mode"] == "swing"
        assert bd["mode_scores"] == {"momentum": 0.6, "swing": 0.9}
        assert bd["avg_score"] == pytest.approx(0.75)

    def test_includes_all_4_scoring_dimensions_from_primary(self, db_session):
        """breakdown.scores 應該是 primary mode 那筆 record 的 4 維分數。"""
        _make_disc(
            db_session,
            scan_date=date(2026, 5, 15),
            mode="swing",
            stock_id="2330",
            rank=1,
            composite=0.9,
            chip=0.7,
            tech=0.85,
            fund=0.65,
            news=0.4,
        )
        db_session.commit()

        rankings = _resolve_all_mode_rankings(date(2026, 5, 15), db_session, top_n=10, per_mode_max=0)
        bd = rankings[0]["score_breakdown"]
        assert bd["scores"]["chip"] == pytest.approx(0.7)
        assert bd["scores"]["technical"] == pytest.approx(0.85)
        assert bd["scores"]["fundamental"] == pytest.approx(0.65)
        assert bd["scores"]["news"] == pytest.approx(0.4)


# ====================================================================== #
# P5-D: _execute_buy 寫入 entry_score_breakdown_json
# ====================================================================== #


class TestExecuteBuyWritesBreakdown:
    def test_buy_with_breakdown_persists_json(self, db_session, monkeypatch):
        """to_buy 含 score_breakdown → RotationPosition.entry_score_breakdown_json 被序列化寫入。"""
        from src.portfolio import manager as mgr_module

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(mgr_module, "get_session", lambda: _Ctx(db_session))

        portfolio = _make_portfolio(db_session, name="bd_test", mode="momentum")
        # 給 TAIEX 假交易日（compute_planned_exit_date 需要）
        from src.data.schema import DailyPrice

        for i in range(20):
            db_session.add(
                DailyPrice(
                    stock_id="TAIEX",
                    date=date(2026, 5, 1 + i),
                    open=100,
                    high=100,
                    low=100,
                    close=100,
                    volume=0,
                    turnover=0,
                )
            )
        db_session.commit()

        mgr = RotationManager("bd_test")
        buy = {
            "stock_id": "2330",
            "stock_name": "台積電",
            "rank": 1,
            "score": 0.85,
            "entry_price": 600.0,
            "shares": 100,
            "allocated_capital": 60_000.0,
            "stop_loss": 580.0,
            "score_breakdown": {
                "mode": "momentum",
                "rank": 1,
                "composite_score": 0.85,
                "scores": {"chip": 0.7, "technical": 0.9, "fundamental": 0.6, "news": 0.4},
                "scan_date": "2026-05-15",
                "regime": "bull",
            },
        }
        trading_cal = [date(2026, 5, 1 + i) for i in range(20)]
        new_cash = mgr._execute_buy(db_session, portfolio.id, buy, date(2026, 5, 15), trading_cal, cash=1_000_000.0)
        db_session.commit()

        assert new_cash < 1_000_000.0
        pos = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        assert pos.entry_score_breakdown_json is not None
        parsed = json.loads(pos.entry_score_breakdown_json)
        assert parsed["mode"] == "momentum"
        assert parsed["composite_score"] == pytest.approx(0.85)
        assert parsed["scores"]["chip"] == pytest.approx(0.7)

    def test_buy_without_breakdown_writes_null(self, db_session, monkeypatch):
        """to_buy 無 score_breakdown → entry_score_breakdown_json 為 NULL（不擲例外）。"""
        from src.portfolio import manager as mgr_module

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(mgr_module, "get_session", lambda: _Ctx(db_session))

        portfolio = _make_portfolio(db_session, name="bd_test2", mode="momentum")
        from src.data.schema import DailyPrice

        for i in range(20):
            db_session.add(
                DailyPrice(
                    stock_id="TAIEX",
                    date=date(2026, 5, 1 + i),
                    open=100,
                    high=100,
                    low=100,
                    close=100,
                    volume=0,
                    turnover=0,
                )
            )
        db_session.commit()

        mgr = RotationManager("bd_test2")
        buy = {
            "stock_id": "2317",
            "stock_name": "鴻海",
            "rank": 1,
            "score": 0.7,
            "entry_price": 200.0,
            "shares": 100,
            "allocated_capital": 20_000.0,
            "stop_loss": 195.0,
            # 沒有 score_breakdown key
        }
        trading_cal = [date(2026, 5, 1 + i) for i in range(20)]
        mgr._execute_buy(db_session, portfolio.id, buy, date(2026, 5, 15), trading_cal, cash=1_000_000.0)
        db_session.commit()

        pos = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2317")).scalar_one()
        assert pos.entry_score_breakdown_json is None


# ====================================================================== #
# P5-E: backfill_entry_score_breakdown
# ====================================================================== #


class TestBackfillEntryScoreBreakdown:
    def test_backfill_fills_null_when_record_exists(self, db_session):
        portfolio = _make_portfolio(db_session, name="bf_test", mode="momentum")
        rec = _make_disc(
            db_session,
            scan_date=date(2026, 5, 10),
            mode="momentum",
            stock_id="2330",
            rank=1,
            composite=0.8,
            chip=0.6,
            tech=0.85,
            fund=0.7,
            news=0.5,
        )
        pos = RotationPosition(
            portfolio_id=portfolio.id,
            stock_id="2330",
            entry_date=date(2026, 5, 10),
            entry_price=600.0,
            entry_rank=1,
            shares=100,
            allocated_capital=60_000.0,
            planned_exit_date=date(2026, 5, 20),
            status="closed",
            entry_score_breakdown_json=None,
        )
        db_session.add(pos)
        db_session.commit()

        stats = RotationManager.backfill_entry_score_breakdown(db_session, portfolio_name="bf_test")
        assert stats["updated"] == 1
        assert stats["skipped_no_record"] == 0

        loaded = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        assert loaded.entry_score_breakdown_json is not None
        parsed = json.loads(loaded.entry_score_breakdown_json)
        assert parsed["composite_score"] == pytest.approx(0.8)
        assert parsed["discovery_record_id"] == rec.id

    def test_backfill_skips_already_filled_by_default(self, db_session):
        portfolio = _make_portfolio(db_session, name="bf_test2", mode="momentum")
        _make_disc(db_session, scan_date=date(2026, 5, 10), mode="momentum", stock_id="2330", rank=1)
        pos = RotationPosition(
            portfolio_id=portfolio.id,
            stock_id="2330",
            entry_date=date(2026, 5, 10),
            entry_price=600.0,
            entry_rank=1,
            shares=100,
            allocated_capital=60_000.0,
            planned_exit_date=date(2026, 5, 20),
            status="closed",
            entry_score_breakdown_json='{"sentinel": true}',
        )
        db_session.add(pos)
        db_session.commit()

        stats = RotationManager.backfill_entry_score_breakdown(db_session)
        assert stats["updated"] == 0
        assert stats["skipped_already_filled"] == 1

        loaded = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        assert json.loads(loaded.entry_score_breakdown_json) == {"sentinel": True}

    def test_backfill_overwrite_replaces_existing(self, db_session):
        portfolio = _make_portfolio(db_session, name="bf_test3", mode="momentum")
        _make_disc(db_session, scan_date=date(2026, 5, 10), mode="momentum", stock_id="2330", rank=1, composite=0.99)
        pos = RotationPosition(
            portfolio_id=portfolio.id,
            stock_id="2330",
            entry_date=date(2026, 5, 10),
            entry_price=600.0,
            entry_rank=1,
            shares=100,
            allocated_capital=60_000.0,
            planned_exit_date=date(2026, 5, 20),
            status="closed",
            entry_score_breakdown_json='{"sentinel": true}',
        )
        db_session.add(pos)
        db_session.commit()

        stats = RotationManager.backfill_entry_score_breakdown(db_session, overwrite=True)
        assert stats["updated"] == 1

        loaded = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        parsed = json.loads(loaded.entry_score_breakdown_json)
        assert parsed["composite_score"] == pytest.approx(0.99)
        assert "sentinel" not in parsed

    def test_backfill_skips_position_with_no_matching_record(self, db_session):
        """歷史 position 的 (entry_date, mode, stock_id) 在 DiscoveryRecord 找不到 → 跳過。"""
        portfolio = _make_portfolio(db_session, name="bf_test4", mode="momentum")
        pos = RotationPosition(
            portfolio_id=portfolio.id,
            stock_id="9999",
            entry_date=date(2026, 5, 10),
            entry_price=100.0,
            entry_rank=1,
            shares=100,
            allocated_capital=10_000.0,
            planned_exit_date=date(2026, 5, 20),
            status="closed",
            entry_score_breakdown_json=None,
        )
        db_session.add(pos)
        db_session.commit()

        stats = RotationManager.backfill_entry_score_breakdown(db_session)
        assert stats["updated"] == 0
        assert stats["skipped_no_record"] == 1

        loaded = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "9999")).scalar_one()
        assert loaded.entry_score_breakdown_json is None  # 仍為 NULL

    def test_backfill_all_mode_uses_primary_mode(self, db_session):
        """portfolio.mode='all' 時，多 mode record 應選最高分作 primary_mode。"""
        portfolio = _make_portfolio(db_session, name="bf_all", mode="all")
        _make_disc(db_session, scan_date=date(2026, 5, 10), mode="momentum", stock_id="2330", rank=1, composite=0.6)
        _make_disc(db_session, scan_date=date(2026, 5, 10), mode="swing", stock_id="2330", rank=1, composite=0.9)
        pos = RotationPosition(
            portfolio_id=portfolio.id,
            stock_id="2330",
            entry_date=date(2026, 5, 10),
            entry_price=600.0,
            entry_rank=1,
            shares=100,
            allocated_capital=60_000.0,
            planned_exit_date=date(2026, 5, 20),
            status="closed",
            entry_score_breakdown_json=None,
        )
        db_session.add(pos)
        db_session.commit()

        stats = RotationManager.backfill_entry_score_breakdown(db_session)
        assert stats["updated"] == 1

        loaded = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        parsed = json.loads(loaded.entry_score_breakdown_json)
        assert parsed["primary_mode"] == "swing"
        assert parsed["mode_scores"] == {"momentum": 0.6, "swing": 0.9}
        assert parsed["mode"] == "all"
