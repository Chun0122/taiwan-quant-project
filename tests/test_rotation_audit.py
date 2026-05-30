"""rotation-audit 純函數測試（src/portfolio/audit.py）。

涵蓋：
  compute_trade_stats：空 / 正常 / 全勝 / 全停損 / None return_pct
  compute_alpha_delta：完整 / 缺 snapshot / 恆等式
  compute_jaccard_stability：空 / 單日 / 多日 / 完全重疊 / 零重疊
"""

from __future__ import annotations

import argparse
from datetime import date

import pytest

from src.portfolio.audit import (
    compute_alpha_delta,
    compute_jaccard_stability,
    compute_trade_stats,
)

# ====================================================================== #
# compute_trade_stats
# ====================================================================== #


class TestComputeTradeStats:
    def test_empty(self):
        s = compute_trade_stats([])
        assert s.n_closed == 0
        assert s.win_pct is None
        assert s.avg_return_pct is None
        assert s.total_pnl == 0.0
        assert s.stop_loss_pct is None

    def test_normal_mix(self):
        trades = [
            {"return_pct": 0.10, "pnl": 10000, "exit_reason": "holding_expired"},
            {"return_pct": -0.05, "pnl": -5000, "exit_reason": "stop_loss"},
            {"return_pct": 0.02, "pnl": 2000, "exit_reason": "holding_expired"},
            {"return_pct": -0.08, "pnl": -8000, "exit_reason": "stop_loss"},
        ]
        s = compute_trade_stats(trades)
        assert s.n_closed == 4
        assert s.win_pct == 50.0
        # avg return = (10 - 5 + 2 - 8)/4 = -0.25% ; ×100 from decimals
        assert s.avg_return_pct == pytest.approx(-0.25, abs=0.01)
        assert s.total_pnl == -1000
        assert s.stop_loss_count == 2
        assert s.stop_loss_pct == 50.0

    def test_all_wins(self):
        trades = [{"return_pct": 0.05, "pnl": 5000, "exit_reason": "holding_expired"} for _ in range(3)]
        s = compute_trade_stats(trades)
        assert s.win_pct == 100.0
        assert s.stop_loss_pct == 0.0

    def test_all_stop_loss(self):
        trades = [{"return_pct": -0.07, "pnl": -7000, "exit_reason": "stop_loss"} for _ in range(5)]
        s = compute_trade_stats(trades)
        assert s.win_pct == 0.0
        assert s.stop_loss_pct == 100.0
        assert s.stop_loss_count == 5

    def test_decimal_to_percent_conversion(self):
        """return_pct 儲存為小數，輸出 avg_return_pct 應 ×100。"""
        trades = [{"return_pct": 0.1166, "pnl": 100, "exit_reason": "holding_expired"}]
        s = compute_trade_stats(trades)
        assert s.avg_return_pct == pytest.approx(11.66, abs=0.01)

    def test_none_return_pct_excluded_from_avg(self):
        trades = [
            {"return_pct": None, "pnl": 0, "exit_reason": "manual"},
            {"return_pct": 0.10, "pnl": 1000, "exit_reason": "holding_expired"},
        ]
        s = compute_trade_stats(trades)
        # avg 只計有值的那筆
        assert s.avg_return_pct == pytest.approx(10.0)
        # win% 分母為 n_closed=2（None 視為非勝）
        assert s.win_pct == 50.0


# ====================================================================== #
# compute_alpha_delta
# ====================================================================== #


class TestComputeAlphaDelta:
    def test_full_computation(self):
        start = {"total_capital": 1_000_000, "alpha_cum_pct": -0.0019, "benchmark_cum_return_pct": 0.0}
        end = {"total_capital": 1_147_941, "alpha_cum_pct": 0.1119, "benchmark_cum_return_pct": 0.0361}
        ad = compute_alpha_delta("mom5_10d", start, end)
        assert ad.portfolio_return_pct == pytest.approx(14.79, abs=0.05)
        assert ad.alpha_start_pct == pytest.approx(-0.19, abs=0.01)
        assert ad.alpha_end_pct == pytest.approx(11.19, abs=0.01)
        assert ad.alpha_delta_pp == pytest.approx(11.38, abs=0.01)
        assert ad.benchmark_delta_pp == pytest.approx(3.61, abs=0.01)

    def test_identity_port_approx_alpha_plus_benchmark(self):
        """恆等式：port Δ% ≈ alpha 增量 + bm 增量（容忍 rounding）。"""
        start = {"total_capital": 1_000_000, "alpha_cum_pct": 0.0, "benchmark_cum_return_pct": 0.0}
        end = {"total_capital": 1_120_000, "alpha_cum_pct": 0.08, "benchmark_cum_return_pct": 0.04}
        ad = compute_alpha_delta("x", start, end)
        assert ad.portfolio_return_pct == pytest.approx(12.0)
        assert ad.alpha_delta_pp == pytest.approx(8.0)
        assert ad.benchmark_delta_pp == pytest.approx(4.0)
        # 12 ≈ 8 + 4
        assert ad.portfolio_return_pct == pytest.approx(ad.alpha_delta_pp + ad.benchmark_delta_pp, abs=0.5)

    def test_missing_start_snapshot(self):
        end = {"total_capital": 1_000_000, "alpha_cum_pct": 0.05, "benchmark_cum_return_pct": 0.03}
        ad = compute_alpha_delta("x", None, end)
        assert ad.cap_start is None
        assert ad.portfolio_return_pct is None
        assert ad.alpha_delta_pp is None
        assert ad.alpha_end_pct == pytest.approx(5.0)

    def test_missing_both(self):
        ad = compute_alpha_delta("x", None, None)
        assert ad.alpha_delta_pp is None
        assert ad.portfolio_return_pct is None

    def test_none_alpha_fields(self):
        start = {"total_capital": 1_000_000, "alpha_cum_pct": None, "benchmark_cum_return_pct": None}
        end = {"total_capital": 1_050_000, "alpha_cum_pct": None, "benchmark_cum_return_pct": None}
        ad = compute_alpha_delta("x", start, end)
        assert ad.portfolio_return_pct == pytest.approx(5.0)  # cap 仍可算
        assert ad.alpha_delta_pp is None  # alpha 缺
        assert ad.benchmark_delta_pp is None

    def test_zero_start_capital_safe(self):
        start = {"total_capital": 0, "alpha_cum_pct": 0.0, "benchmark_cum_return_pct": 0.0}
        end = {"total_capital": 1000, "alpha_cum_pct": 0.05, "benchmark_cum_return_pct": 0.0}
        ad = compute_alpha_delta("x", start, end)
        assert ad.portfolio_return_pct is None  # 防除零


# ====================================================================== #
# compute_jaccard_stability
# ====================================================================== #


class TestComputeJaccardStability:
    def test_empty(self):
        j = compute_jaccard_stability([])
        assert j.pairs == []
        assert j.mean_jaccard is None

    def test_single_day(self):
        j = compute_jaccard_stability([("2026-05-09", {"2330", "2317"})])
        assert j.pairs == []
        assert j.mean_jaccard is None

    def test_full_overlap(self):
        days = [
            ("2026-05-25", {"a", "b", "c"}),
            ("2026-05-26", {"a", "b", "c"}),
        ]
        j = compute_jaccard_stability(days)
        assert len(j.pairs) == 1
        assert j.pairs[0]["jaccard"] == 1.0
        assert j.mean_jaccard == 1.0

    def test_zero_overlap(self):
        days = [
            ("2026-05-15", {"a", "b"}),
            ("2026-05-18", {"c", "d"}),
        ]
        j = compute_jaccard_stability(days)
        assert j.pairs[0]["jaccard"] == 0.0
        assert j.mean_jaccard == 0.0

    def test_partial_overlap(self):
        days = [
            ("d1", {"a", "b", "c", "d", "e"}),
            ("d2", {"a", "b", "x", "y", "z"}),  # overlap=2, union=8
        ]
        j = compute_jaccard_stability(days)
        assert j.pairs[0]["overlap"] == 2
        assert j.pairs[0]["union"] == 8
        assert j.pairs[0]["jaccard"] == pytest.approx(0.25)

    def test_multi_day_aggregates(self):
        days = [
            ("d1", {"a", "b"}),
            ("d2", {"a", "b"}),  # jaccard 1.0
            ("d3", {"c", "d"}),  # jaccard 0.0
        ]
        j = compute_jaccard_stability(days)
        assert len(j.pairs) == 2
        assert j.mean_jaccard == pytest.approx(0.5)
        assert j.max_jaccard == 1.0
        assert j.min_jaccard == 0.0
        assert j.median_jaccard == pytest.approx(0.5)


# ====================================================================== #
# CLI handler（_parse_period + cmd_rotation_audit DB 整合）
# ====================================================================== #


class TestParsePeriod:
    def test_valid(self):
        from src.cli.audit_cmd import _parse_period

        assert _parse_period("2026-04-29:2026-05-08") == (date(2026, 4, 29), date(2026, 5, 8))

    def test_none(self):
        from src.cli.audit_cmd import _parse_period

        assert _parse_period(None) is None

    def test_malformed(self):
        from src.cli.audit_cmd import _parse_period

        assert _parse_period("garbage") is None
        assert _parse_period("2026-04-29") is None


@pytest.fixture()
def patch_session(db_session, monkeypatch):
    from src.cli import audit_cmd as mod

    class _Ctx:
        def __init__(self, s):
            self._s = s

        def __enter__(self):
            return self._s

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(mod, "get_session", lambda: _Ctx(db_session))
    import src.cli.helpers as helpers

    monkeypatch.setattr(helpers, "init_db", lambda: None)
    return db_session


class TestCmdRotationAudit:
    def test_missing_period_b_returns_2(self, patch_session, capsys):
        from src.cli.audit_cmd import cmd_rotation_audit

        code = cmd_rotation_audit(
            argparse.Namespace(period_a=None, period_b=None, top=5, jaccard_mode="momentum", out=None)
        )
        assert code == 2
        assert "period-b 為必填" in capsys.readouterr().out

    def test_empty_db_graceful(self, patch_session, capsys):
        """無 portfolio → 報告標明無紀錄，exit 0。"""
        from src.cli.audit_cmd import cmd_rotation_audit

        code = cmd_rotation_audit(
            argparse.Namespace(
                period_a=None, period_b="2026-05-09:2026-05-29", top=5, jaccard_mode="momentum", out=None
            )
        )
        assert code == 0
        assert "無 rotation_portfolio 紀錄" in capsys.readouterr().out

    def test_full_report_with_seed(self, patch_session, capsys):
        """種子 portfolio + closed trades + snapshot → 報告含三大區塊。"""
        from src.cli.audit_cmd import cmd_rotation_audit
        from src.data.schema import RotationDailySnapshot, RotationPortfolio, RotationPosition

        p = RotationPortfolio(
            name="aud_test",
            mode="momentum",
            max_positions=3,
            holding_days=5,
            allow_renewal=True,
            initial_capital=1_000_000.0,
            current_capital=1_050_000.0,
            current_cash=200_000.0,
            status="active",
        )
        patch_session.add(p)
        patch_session.flush()

        # 2 closed trades in B period
        for sid, ret, pnl, reason, ed in [
            ("2330", 0.10, 10000, "holding_expired", date(2026, 5, 10)),
            ("2317", -0.06, -6000, "stop_loss", date(2026, 5, 12)),
        ]:
            patch_session.add(
                RotationPosition(
                    portfolio_id=p.id,
                    stock_id=sid,
                    entry_date=ed,
                    entry_price=100,
                    entry_rank=1,
                    holding_days_count=5,
                    planned_exit_date=date(2026, 5, 20),
                    exit_date=date(2026, 5, 18),
                    shares=1000,
                    allocated_capital=100000,
                    status="closed",
                    return_pct=ret,
                    pnl=pnl,
                    exit_reason=reason,
                )
            )
        # snapshots at B period 端點
        for d, cap, alpha, bm in [
            (date(2026, 5, 9), 1_000_000.0, 0.0, 0.0),
            (date(2026, 5, 29), 1_050_000.0, 0.02, 0.03),
        ]:
            patch_session.add(
                RotationDailySnapshot(
                    portfolio_name="aud_test",
                    snapshot_date=d,
                    total_capital=cap,
                    total_market_value=cap * 0.8,
                    total_cash=cap * 0.2,
                    n_holdings=3,
                    alpha_cum_pct=alpha,
                    benchmark_cum_return_pct=bm,
                )
            )
        patch_session.commit()

        code = cmd_rotation_audit(
            argparse.Namespace(
                period_a=None, period_b="2026-05-09:2026-05-29", top=5, jaccard_mode="momentum", out=None
            )
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Closed Trade 統計" in out
        assert "Benchmark Alpha 分解" in out
        assert "訊號穩定性" in out
        assert "aud_test" in out
        # 2 closed < 10 → 樣本不足警告
        assert "樣本不足警告" in out

    def test_writes_to_out_file(self, patch_session, tmp_path):
        from src.cli.audit_cmd import cmd_rotation_audit
        from src.data.schema import RotationPortfolio

        patch_session.add(
            RotationPortfolio(
                name="x",
                mode="momentum",
                max_positions=3,
                holding_days=5,
                allow_renewal=True,
                initial_capital=1_000_000.0,
                current_capital=1_000_000.0,
                current_cash=1_000_000.0,
                status="active",
            )
        )
        patch_session.commit()

        out_path = tmp_path / "sub" / "REPORT.md"
        code = cmd_rotation_audit(
            argparse.Namespace(
                period_a=None,
                period_b="2026-05-09:2026-05-29",
                top=5,
                jaccard_mode="momentum",
                out=str(out_path),
            )
        )
        assert code == 0
        assert out_path.exists()
        assert "Rotation Audit Report" in out_path.read_text(encoding="utf-8")
