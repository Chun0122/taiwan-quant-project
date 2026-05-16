"""2026-05-15 sprint P2 — RotationDailySnapshot benchmark/alpha 三欄位測試。

對應 5/29 audit 需要：portfolio vs 0050 alpha 對比。
benchmark 採 0050；缺資料時三欄位皆 None。
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select

from src.data.schema import DailyPrice, RotationDailySnapshot, RotationPortfolio
from src.portfolio.manager import RotationManager


def _seed_0050(db_session, prices: dict[date, float]) -> None:
    for d, c in prices.items():
        db_session.add(
            DailyPrice(
                stock_id="0050",
                date=d,
                open=c,
                high=c,
                low=c,
                close=c,
                volume=1_000_000,
                turnover=1_000_000_000,
            )
        )
    db_session.flush()


def _make_portfolio(db_session, name: str = "alpha_test", initial: float = 1_000_000.0) -> RotationPortfolio:
    p = RotationPortfolio(
        name=name,
        mode="momentum",
        max_positions=5,
        holding_days=10,
        allow_renewal=True,
        initial_capital=initial,
        current_capital=initial,
        current_cash=initial,
        status="active",
    )
    db_session.add(p)
    db_session.commit()
    return p


# ====================================================================== #
# P2-A: schema 欄位存在 + nullable
# ====================================================================== #


class TestSchemaColumns:
    def test_snapshot_can_be_created_without_benchmark_fields(self, db_session):
        s = RotationDailySnapshot(
            portfolio_name="x",
            snapshot_date=date(2026, 5, 15),
            total_capital=1_000_000.0,
            total_market_value=0.0,
            total_cash=1_000_000.0,
            unrealized_pnl=0.0,
            n_holdings=0,
        )
        db_session.add(s)
        db_session.flush()

        loaded = db_session.execute(select(RotationDailySnapshot)).scalar_one()
        assert loaded.benchmark_return_pct is None
        assert loaded.benchmark_cum_return_pct is None
        assert loaded.alpha_cum_pct is None


# ====================================================================== #
# P2-B: benchmark/alpha 計算
# ====================================================================== #


class TestSnapshotBenchmarkAlpha:
    def test_first_snapshot_no_prev_no_daily_bm_but_has_cum(self, db_session):
        """首筆 snapshot：無 prev → benchmark_return_pct=None；但 base=self → cum=0%。"""
        _seed_0050(db_session, {date(2026, 5, 15): 100.0})
        portfolio = _make_portfolio(db_session)
        mgr = RotationManager("alpha_test")

        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 15),
            market_value=0.0,
            n_holdings=0,
            regime="bull",
        )

        s = db_session.execute(select(RotationDailySnapshot)).scalar_one()
        assert s.benchmark_return_pct is None  # 無 prev
        assert s.benchmark_cum_return_pct == pytest.approx(0.0)  # base = 當前
        assert s.alpha_cum_pct == pytest.approx(0.0)  # portfolio_cum_return = 0

    def test_second_day_benchmark_and_alpha_computed(self, db_session):
        """第二天：0050 +5%、portfolio +8% → daily_bm=+5%、cum_bm=+5%、alpha=+3%。"""
        _seed_0050(
            db_session,
            {date(2026, 5, 15): 100.0, date(2026, 5, 16): 105.0},
        )
        portfolio = _make_portfolio(db_session, initial=1_000_000.0)
        mgr = RotationManager("alpha_test")

        # day 1
        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 15),
            market_value=0.0,
            n_holdings=0,
            regime="bull",
        )
        # day 2: capital → 1.08M (+8%)
        portfolio.current_capital = 1_080_000.0
        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 16),
            market_value=500_000.0,
            n_holdings=2,
            regime="bull",
        )

        s = db_session.execute(
            select(RotationDailySnapshot).where(RotationDailySnapshot.snapshot_date == date(2026, 5, 16))
        ).scalar_one()
        assert s.benchmark_return_pct == pytest.approx(0.05)
        assert s.benchmark_cum_return_pct == pytest.approx(0.05)
        assert s.alpha_cum_pct == pytest.approx(0.03, abs=1e-6)

    def test_missing_0050_writes_none(self, db_session):
        """0050 完全缺資料：三欄位都 None，不拋例外。"""
        portfolio = _make_portfolio(db_session)
        mgr = RotationManager("alpha_test")

        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 15),
            market_value=0.0,
            n_holdings=0,
            regime="bull",
        )

        s = db_session.execute(select(RotationDailySnapshot)).scalar_one()
        assert s.benchmark_return_pct is None
        assert s.benchmark_cum_return_pct is None
        assert s.alpha_cum_pct is None

    def test_weekend_fallback_to_previous_trading_day(self, db_session):
        """snapshot 落在週末/假日：fallback 取 ≤ target_date 最近一天的 0050。"""
        # 只有 5/15 (Fri) 有 0050 資料；snapshot 寫 5/16 (Sat)
        _seed_0050(db_session, {date(2026, 5, 15): 100.0})
        portfolio = _make_portfolio(db_session)
        mgr = RotationManager("alpha_test")

        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 16),  # Sat
            market_value=0.0,
            n_holdings=0,
            regime="bull",
        )

        s = db_session.execute(select(RotationDailySnapshot)).scalar_one()
        # fallback 抓到 5/15 → base = self → cum=0
        assert s.benchmark_cum_return_pct == pytest.approx(0.0)

    def test_negative_alpha_when_portfolio_lags_benchmark(self, db_session):
        """0050 +10% 但 portfolio -2% → alpha = -12%。"""
        _seed_0050(
            db_session,
            {date(2026, 5, 15): 100.0, date(2026, 5, 22): 110.0},
        )
        portfolio = _make_portfolio(db_session, initial=1_000_000.0)
        mgr = RotationManager("alpha_test")

        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 15),
            market_value=0.0,
            n_holdings=0,
            regime="bull",
        )
        portfolio.current_capital = 980_000.0  # -2%
        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 22),
            market_value=500_000.0,
            n_holdings=2,
            regime="bull",
        )

        s = db_session.execute(
            select(RotationDailySnapshot).where(RotationDailySnapshot.snapshot_date == date(2026, 5, 22))
        ).scalar_one()
        assert s.benchmark_cum_return_pct == pytest.approx(0.10)
        assert s.alpha_cum_pct == pytest.approx(-0.12, abs=1e-6)

    def test_update_path_recomputes_benchmark(self, db_session):
        """同日重複呼叫採 update path：benchmark/alpha 也要重算。"""
        _seed_0050(db_session, {date(2026, 5, 15): 100.0})
        portfolio = _make_portfolio(db_session, initial=1_000_000.0)
        mgr = RotationManager("alpha_test")

        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 15),
            market_value=0.0,
            n_holdings=0,
            regime="bull",
        )
        # 同日 update：capital +5%
        portfolio.current_capital = 1_050_000.0
        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 15),
            market_value=500_000.0,
            n_holdings=2,
            regime="bull",
        )

        rows = db_session.execute(select(RotationDailySnapshot)).scalars().all()
        assert len(rows) == 1  # update path
        s = rows[0]
        # benchmark base=self → cum=0；portfolio_cum=+5% → alpha=+5%
        assert s.benchmark_cum_return_pct == pytest.approx(0.0)
        assert s.alpha_cum_pct == pytest.approx(0.05)


# ====================================================================== #
# P2-C: helper 函數
# ====================================================================== #


class TestBenchmarkCloseHelper:
    def test_returns_close_on_exact_date(self, db_session):
        from src.portfolio.manager import _get_benchmark_close_on_or_before

        _seed_0050(db_session, {date(2026, 5, 15): 100.5})
        result = _get_benchmark_close_on_or_before(db_session, date(2026, 5, 15))
        assert result == pytest.approx(100.5)

    def test_returns_most_recent_before_target(self, db_session):
        from src.portfolio.manager import _get_benchmark_close_on_or_before

        _seed_0050(
            db_session,
            {date(2026, 5, 13): 99.0, date(2026, 5, 14): 100.0, date(2026, 5, 15): 101.0},
        )
        # target=5/17 (週日) → 應取 5/15 最近一筆
        result = _get_benchmark_close_on_or_before(db_session, date(2026, 5, 17))
        assert result == pytest.approx(101.0)

    def test_returns_none_when_outside_lookback(self, db_session):
        from src.portfolio.manager import _get_benchmark_close_on_or_before

        _seed_0050(db_session, {date(2026, 5, 1): 100.0})
        # target=5/15，lookback=7 天，5/1 在 5/8 之前 → 找不到
        result = _get_benchmark_close_on_or_before(db_session, date(2026, 5, 15), lookback_days=7)
        assert result is None

    def test_returns_none_when_no_data(self, db_session):
        from src.portfolio.manager import _get_benchmark_close_on_or_before

        result = _get_benchmark_close_on_or_before(db_session, date(2026, 5, 15))
        assert result is None


# ====================================================================== #
# P2-D: Migration
# ====================================================================== #


class TestMigrationIdempotent:
    def test_snapshot_migrations_listed(self):
        from src.data.migrate import MIGRATIONS

        cols = {(t, c) for t, c, _ in MIGRATIONS}
        assert ("rotation_daily_snapshot", "benchmark_return_pct") in cols
        assert ("rotation_daily_snapshot", "benchmark_cum_return_pct") in cols
        assert ("rotation_daily_snapshot", "alpha_cum_pct") in cols


# ====================================================================== #
# P2-E: 純函數 compute_benchmark_alpha_fields
# ====================================================================== #


class TestComputeBenchmarkAlphaFields:
    """純函數：與 DB 解耦的計算邏輯。"""

    def test_all_inputs_present_returns_three_values(self):
        from src.portfolio.manager import compute_benchmark_alpha_fields

        bm_ret, bm_cum, alpha = compute_benchmark_alpha_fields(
            today_bm_close=105.0,
            prev_bm_close=100.0,
            base_bm_close=100.0,
            portfolio_cum_return=0.08,
        )
        assert bm_ret == pytest.approx(0.05)
        assert bm_cum == pytest.approx(0.05)
        assert alpha == pytest.approx(0.03)

    def test_missing_today_returns_all_none(self):
        from src.portfolio.manager import compute_benchmark_alpha_fields

        result = compute_benchmark_alpha_fields(None, 100.0, 100.0, 0.05)
        assert result == (None, None, None)

    def test_missing_prev_drops_daily_only(self):
        from src.portfolio.manager import compute_benchmark_alpha_fields

        bm_ret, bm_cum, alpha = compute_benchmark_alpha_fields(105.0, None, 100.0, 0.08)
        assert bm_ret is None
        assert bm_cum == pytest.approx(0.05)
        assert alpha == pytest.approx(0.03)

    def test_zero_or_negative_base_drops_cum_and_alpha(self):
        from src.portfolio.manager import compute_benchmark_alpha_fields

        bm_ret, bm_cum, alpha = compute_benchmark_alpha_fields(105.0, 100.0, 0.0, 0.08)
        assert bm_ret == pytest.approx(0.05)
        assert bm_cum is None
        assert alpha is None

    def test_missing_portfolio_cum_return_drops_alpha_only(self):
        from src.portfolio.manager import compute_benchmark_alpha_fields

        bm_ret, bm_cum, alpha = compute_benchmark_alpha_fields(105.0, 100.0, 100.0, None)
        assert bm_cum == pytest.approx(0.05)
        assert alpha is None


# ====================================================================== #
# P2-F: backfill_snapshot_benchmark_alpha — 一次性補齊歷史 NULL 列
# ====================================================================== #


class TestBackfillSnapshotBenchmarkAlpha:
    def test_backfill_fills_null_rows(self, db_session):
        """歷史 snapshot 的 alpha 三欄位為 NULL 時，backfill 後應計算正確。"""
        from src.portfolio.manager import RotationManager

        _seed_0050(
            db_session,
            {date(2026, 5, 6): 100.0, date(2026, 5, 7): 102.0, date(2026, 5, 8): 105.0},
        )
        portfolio = _make_portfolio(db_session, name="bf_test", initial=1_000_000.0)

        # 手動寫入 3 筆 snapshot — alpha 欄位全部 NULL（模擬 commit 7f13f08 前的歷史資料）
        for d, cap in [
            (date(2026, 5, 6), 1_000_000.0),
            (date(2026, 5, 7), 1_010_000.0),
            (date(2026, 5, 8), 1_080_000.0),
        ]:
            db_session.add(
                RotationDailySnapshot(
                    portfolio_name="bf_test",
                    snapshot_date=d,
                    total_capital=cap,
                    total_market_value=0.0,
                    total_cash=cap,
                    unrealized_pnl=0.0,
                    n_holdings=0,
                    daily_return_pct=None,
                )
            )
        db_session.commit()

        stats = RotationManager.backfill_snapshot_benchmark_alpha(db_session, portfolio_name="bf_test")
        assert stats["updated"] == 3
        assert stats["skipped_no_benchmark"] == 0

        rows = (
            db_session.execute(
                select(RotationDailySnapshot)
                .where(RotationDailySnapshot.portfolio_name == "bf_test")
                .order_by(RotationDailySnapshot.snapshot_date)
            )
            .scalars()
            .all()
        )
        # row 0: base = self → cum=0, daily=None（無 prev）
        assert rows[0].benchmark_return_pct is None
        assert rows[0].benchmark_cum_return_pct == pytest.approx(0.0)
        assert rows[0].alpha_cum_pct == pytest.approx(0.0)
        # row 1: prev=100, today=102 → daily=+2%, cum=+2%, port=+1% → alpha=-1%
        assert rows[1].benchmark_return_pct == pytest.approx(0.02)
        assert rows[1].benchmark_cum_return_pct == pytest.approx(0.02)
        assert rows[1].alpha_cum_pct == pytest.approx(-0.01, abs=1e-6)
        # row 2: prev=102, today=105 → daily≈+2.94%, cum=+5%, port=+8% → alpha=+3%
        assert rows[2].benchmark_cum_return_pct == pytest.approx(0.05)
        assert rows[2].alpha_cum_pct == pytest.approx(0.03, abs=1e-6)

    def test_backfill_default_skips_already_filled(self, db_session):
        """已有值的列不應被覆蓋（除非 overwrite=True）。"""
        from src.portfolio.manager import RotationManager

        _seed_0050(db_session, {date(2026, 5, 6): 100.0, date(2026, 5, 7): 110.0})
        _make_portfolio(db_session, name="bf_test", initial=1_000_000.0)
        # 第一筆已有 alpha 值（人工塞 sentinel 偵測是否被覆蓋）
        db_session.add(
            RotationDailySnapshot(
                portfolio_name="bf_test",
                snapshot_date=date(2026, 5, 6),
                total_capital=1_000_000.0,
                total_market_value=0.0,
                total_cash=1_000_000.0,
                unrealized_pnl=0.0,
                n_holdings=0,
                alpha_cum_pct=999.0,  # sentinel
            )
        )
        # 第二筆 alpha=NULL（待補）
        db_session.add(
            RotationDailySnapshot(
                portfolio_name="bf_test",
                snapshot_date=date(2026, 5, 7),
                total_capital=1_050_000.0,
                total_market_value=0.0,
                total_cash=1_050_000.0,
                unrealized_pnl=0.0,
                n_holdings=0,
            )
        )
        db_session.commit()

        stats = RotationManager.backfill_snapshot_benchmark_alpha(db_session)
        assert stats["updated"] == 1  # 只補第二筆

        rows = (
            db_session.execute(select(RotationDailySnapshot).order_by(RotationDailySnapshot.snapshot_date))
            .scalars()
            .all()
        )
        assert rows[0].alpha_cum_pct == 999.0  # 未被覆蓋
        assert rows[1].alpha_cum_pct is not None

    def test_backfill_overwrite_recomputes_all(self, db_session):
        """overwrite=True 時連已有值的列也重算。"""
        from src.portfolio.manager import RotationManager

        _seed_0050(db_session, {date(2026, 5, 6): 100.0})
        _make_portfolio(db_session, name="bf_test", initial=1_000_000.0)
        db_session.add(
            RotationDailySnapshot(
                portfolio_name="bf_test",
                snapshot_date=date(2026, 5, 6),
                total_capital=1_000_000.0,
                total_market_value=0.0,
                total_cash=1_000_000.0,
                unrealized_pnl=0.0,
                n_holdings=0,
                alpha_cum_pct=999.0,  # 錯誤 sentinel
            )
        )
        db_session.commit()

        stats = RotationManager.backfill_snapshot_benchmark_alpha(db_session, overwrite=True)
        assert stats["updated"] == 1

        row = db_session.execute(select(RotationDailySnapshot)).scalar_one()
        assert row.alpha_cum_pct == pytest.approx(0.0)  # base=self → 0

    def test_backfill_no_benchmark_skips_row(self, db_session):
        """0050 完全缺資料時，row 被歸到 skipped_no_benchmark 不擲例外。"""
        from src.portfolio.manager import RotationManager

        _make_portfolio(db_session, name="bf_test", initial=1_000_000.0)
        db_session.add(
            RotationDailySnapshot(
                portfolio_name="bf_test",
                snapshot_date=date(2026, 5, 6),
                total_capital=1_000_000.0,
                total_market_value=0.0,
                total_cash=1_000_000.0,
                unrealized_pnl=0.0,
                n_holdings=0,
            )
        )
        db_session.commit()

        stats = RotationManager.backfill_snapshot_benchmark_alpha(db_session)
        assert stats["updated"] == 0
        assert stats["skipped_no_benchmark"] == 1

        row = db_session.execute(select(RotationDailySnapshot)).scalar_one()
        assert row.alpha_cum_pct is None  # 仍維持 NULL

    def test_backfill_filters_by_portfolio_name(self, db_session):
        """指定 portfolio_name 時不影響其他 portfolio。"""
        from src.portfolio.manager import RotationManager

        _seed_0050(db_session, {date(2026, 5, 6): 100.0})
        _make_portfolio(db_session, name="bf_a", initial=1_000_000.0)
        _make_portfolio(db_session, name="bf_b", initial=1_000_000.0)
        for name in ("bf_a", "bf_b"):
            db_session.add(
                RotationDailySnapshot(
                    portfolio_name=name,
                    snapshot_date=date(2026, 5, 6),
                    total_capital=1_000_000.0,
                    total_market_value=0.0,
                    total_cash=1_000_000.0,
                    unrealized_pnl=0.0,
                    n_holdings=0,
                )
            )
        db_session.commit()

        stats = RotationManager.backfill_snapshot_benchmark_alpha(db_session, portfolio_name="bf_a")
        assert stats["updated"] == 1

        rows = (
            db_session.execute(select(RotationDailySnapshot).order_by(RotationDailySnapshot.portfolio_name))
            .scalars()
            .all()
        )
        assert rows[0].alpha_cum_pct is not None  # bf_a 被補
        assert rows[1].alpha_cum_pct is None  # bf_b 未動


# ====================================================================== #
# P2-G: get_recent_snapshots 暴露 alpha 欄位（dashboard 串接前置）
# ====================================================================== #


class TestGetRecentSnapshotsExposesAlpha:
    def test_returns_benchmark_and_alpha_fields(self, db_session):
        from src.portfolio.manager import RotationManager

        _seed_0050(db_session, {date(2026, 5, 15): 100.0, date(2026, 5, 16): 105.0})
        portfolio = _make_portfolio(db_session, name="exp_test", initial=1_000_000.0)
        mgr = RotationManager("exp_test")
        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 15),
            market_value=0.0,
            n_holdings=0,
            regime="bull",
        )
        portfolio.current_capital = 1_080_000.0
        mgr._write_daily_snapshot(
            db_session,
            portfolio=portfolio,
            snapshot_date=date(2026, 5, 16),
            market_value=500_000.0,
            n_holdings=2,
            regime="bull",
        )

        snaps = mgr.get_recent_snapshots(n_days=90)
        assert len(snaps) == 2
        for s in snaps:
            assert "benchmark_return_pct" in s
            assert "benchmark_cum_return_pct" in s
            assert "alpha_cum_pct" in s

        # 第二筆值正確
        assert snaps[1]["benchmark_cum_return_pct"] == pytest.approx(0.05)
        assert snaps[1]["alpha_cum_pct"] == pytest.approx(0.03, abs=1e-6)
