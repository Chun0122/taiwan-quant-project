"""2026-05-16 任務 3 — baseline regression guard 測試。

對應 P0 任務 3 驗收標準：
  - data/baseline_metrics.json 含 4 active rotation × {sharpe, mdd, win_rate, alpha_cum_pct}
  - validate-baseline --tolerance X：對比當前 vs baseline，超 tolerance 退出碼 ≠ 0
  - morning-routine Step 17：失敗不阻擋但寫 Discord 警告

涵蓋：
  P3-A 純函數 compute_metrics_from_snapshots（資料不足／充足／edge cases）
  P3-B 純函數 compare_metrics（pass／sharpe regression／mdd regression／win rate／alpha／None handling）
  P3-C save/load_baseline JSON 雙向 round-trip
  P3-D CLI exit code（0=pass / 1=regression / 2=baseline missing）
  P3-E Step 17 Discord formatter
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.cli.baseline_cmd import (
    DEFAULT_TOLERANCE_DELTAS,
    PortfolioMetrics,
    RegressionFinding,
    compare_metrics,
    compute_metrics_from_snapshots,
    format_regressions_for_discord,
    load_baseline,
    save_baseline,
)

# ====================================================================== #
# P3-A: compute_metrics_from_snapshots 純函數
# ====================================================================== #


def _snap(d: date, capital: float, daily: float | None, alpha: float | None = None) -> dict:
    return {"snapshot_date": d, "total_capital": capital, "daily_return_pct": daily, "alpha_cum_pct": alpha}


class TestComputeMetricsFromSnapshots:
    def test_empty_returns_all_none(self):
        m = compute_metrics_from_snapshots([], portfolio_name="x", as_of="2026-05-15")
        assert m.snapshot_count == 0
        assert m.sharpe_ratio is None
        assert m.max_drawdown_pct is None
        assert m.win_rate_pct is None
        assert m.alpha_cum_pct is None

    def test_fewer_than_10_snapshots_skips_sharpe(self):
        """N<10 → Sharpe=None；N>=3 → MDD/win_rate 仍算。"""
        snaps = [
            _snap(date(2026, 5, 1 + i), 1_000_000 + i * 1000, 0.001 if i > 0 else None, alpha=0.05) for i in range(5)
        ]
        m = compute_metrics_from_snapshots(snaps, portfolio_name="x", as_of="2026-05-15")
        assert m.sharpe_ratio is None
        assert m.max_drawdown_pct is not None
        assert m.win_rate_pct is not None
        assert m.alpha_cum_pct == pytest.approx(0.05)

    def test_full_data_computes_all_metrics(self):
        """N=15 → 所有 metric 都算。"""
        # 報酬序列：穩定 +0.5% 上漲，第 8 天 -2% 形成 drawdown
        returns = [
            None,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            -0.02,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
        ]
        cap = 1_000_000.0
        snaps = []
        for i, r in enumerate(returns):
            if r is not None:
                cap *= 1 + r
            snaps.append(_snap(date(2026, 5, 1 + i), cap, r, alpha=0.10 + i * 0.001))
        m = compute_metrics_from_snapshots(snaps, portfolio_name="x", as_of="2026-05-15")
        assert m.sharpe_ratio is not None
        assert m.max_drawdown_pct is not None
        assert m.max_drawdown_pct > 0
        assert m.win_rate_pct is not None
        assert m.alpha_cum_pct == pytest.approx(0.10 + 14 * 0.001, abs=1e-6)

    def test_alpha_from_last_snapshot(self):
        """alpha_cum_pct 取最新一筆（asc 排序）。"""
        snaps = [
            _snap(date(2026, 5, 1), 1_000_000, None, alpha=0.05),
            _snap(date(2026, 5, 2), 1_000_000, 0.0, alpha=0.07),
            _snap(date(2026, 5, 3), 1_000_000, 0.0, alpha=0.08),
        ]
        m = compute_metrics_from_snapshots(snaps, portfolio_name="x", as_of="2026-05-15")
        assert m.alpha_cum_pct == pytest.approx(0.08)

    def test_alpha_none_when_last_snapshot_missing(self):
        snaps = [_snap(date(2026, 5, 1), 1_000_000, None, alpha=None)]
        m = compute_metrics_from_snapshots(snaps, portfolio_name="x", as_of="2026-05-15")
        assert m.alpha_cum_pct is None


# ====================================================================== #
# P3-B: compare_metrics 純函數
# ====================================================================== #


def _bm(name="x", sharpe=1.0, mdd=5.0, win=55.0, alpha=0.10) -> PortfolioMetrics:
    return PortfolioMetrics(
        portfolio_name=name,
        as_of="2026-05-01",
        snapshot_count=30,
        sharpe_ratio=sharpe,
        max_drawdown_pct=mdd,
        win_rate_pct=win,
        alpha_cum_pct=alpha,
    )


def _cm(name="x", sharpe=1.0, mdd=5.0, win=55.0, alpha=0.10) -> PortfolioMetrics:
    return PortfolioMetrics(
        portfolio_name=name,
        as_of="2026-05-15",
        snapshot_count=33,
        sharpe_ratio=sharpe,
        max_drawdown_pct=mdd,
        win_rate_pct=win,
        alpha_cum_pct=alpha,
    )


class TestCompareMetrics:
    def test_identical_metrics_no_regression(self):
        findings = compare_metrics(_bm(), _cm())
        assert all(not f.is_regression for f in findings)

    def test_sharpe_regression_triggers(self):
        """current Sharpe 比 baseline 低超過閾值（預設 0.20）。"""
        findings = compare_metrics(_bm(sharpe=1.0), _cm(sharpe=0.7))
        sharpe_finding = next(f for f in findings if f.metric == "sharpe_ratio")
        assert sharpe_finding.is_regression
        assert sharpe_finding.delta == pytest.approx(-0.3)

    def test_sharpe_within_tolerance_passes(self):
        """退化在閾值內不觸發。"""
        findings = compare_metrics(_bm(sharpe=1.0), _cm(sharpe=0.85))  # -0.15 < -0.20 → 不觸發
        sharpe_finding = next(f for f in findings if f.metric == "sharpe_ratio")
        assert not sharpe_finding.is_regression

    def test_max_drawdown_regression_triggers(self):
        """current MDD 比 baseline 高超過閾值（預設 2.0pp）。"""
        findings = compare_metrics(_bm(mdd=5.0), _cm(mdd=8.0))
        mdd_finding = next(f for f in findings if f.metric == "max_drawdown_pct")
        assert mdd_finding.is_regression
        assert mdd_finding.delta == pytest.approx(3.0)

    def test_max_drawdown_decrease_no_regression(self):
        """MDD 變小（改善）不是 regression。"""
        findings = compare_metrics(_bm(mdd=5.0), _cm(mdd=2.0))
        mdd_finding = next(f for f in findings if f.metric == "max_drawdown_pct")
        assert not mdd_finding.is_regression

    def test_win_rate_regression_triggers(self):
        findings = compare_metrics(_bm(win=55.0), _cm(win=45.0))
        f = next(f for f in findings if f.metric == "win_rate_pct")
        assert f.is_regression

    def test_alpha_regression_triggers(self):
        """alpha 退化 > 0.03（3pp）觸發。"""
        findings = compare_metrics(_bm(alpha=0.10), _cm(alpha=0.05))
        f = next(f for f in findings if f.metric == "alpha_cum_pct")
        assert f.is_regression

    def test_negative_alpha_decrease_triggers(self):
        """從 -0.05 退化到 -0.12 → delta = -0.07 → 觸發。"""
        findings = compare_metrics(_bm(alpha=-0.05), _cm(alpha=-0.12))
        f = next(f for f in findings if f.metric == "alpha_cum_pct")
        assert f.is_regression

    def test_none_metric_not_regression(self):
        """任一值為 None（資料不足）→ not regression（不足以判斷）。"""
        findings = compare_metrics(_bm(sharpe=None), _cm(sharpe=0.3))
        f = next(f for f in findings if f.metric == "sharpe_ratio")
        assert not f.is_regression
        assert "資料不足" in f.reason

    def test_tolerance_multiplier_loosens(self):
        """tolerance=2.0 → 閾值放寬 2 倍。"""
        findings_strict = compare_metrics(_bm(sharpe=1.0), _cm(sharpe=0.7), tolerance=1.0)
        findings_loose = compare_metrics(_bm(sharpe=1.0), _cm(sharpe=0.7), tolerance=2.0)
        assert next(f for f in findings_strict if f.metric == "sharpe_ratio").is_regression
        assert not next(f for f in findings_loose if f.metric == "sharpe_ratio").is_regression

    def test_tolerance_multiplier_tightens(self):
        """tolerance=0.5 → 閾值縮一半。"""
        # -0.15 在預設 0.20 內，但在 0.10 (0.20×0.5) 外
        findings_normal = compare_metrics(_bm(sharpe=1.0), _cm(sharpe=0.85), tolerance=1.0)
        findings_tight = compare_metrics(_bm(sharpe=1.0), _cm(sharpe=0.85), tolerance=0.5)
        assert not next(f for f in findings_normal if f.metric == "sharpe_ratio").is_regression
        assert next(f for f in findings_tight if f.metric == "sharpe_ratio").is_regression


# ====================================================================== #
# P3-C: save_baseline / load_baseline round-trip
# ====================================================================== #


class TestBaselineIO:
    def test_save_and_load_preserves_values(self):
        metrics = {
            "all10_5d": _bm(name="all10_5d", sharpe=1.2, mdd=4.5, win=52.0, alpha=0.08),
            "swing5_3d": _bm(name="swing5_3d", sharpe=None, mdd=2.0, win=40.0, alpha=-0.02),
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "baseline.json"
            save_baseline(metrics, path=path, git_commit="testhash")
            loaded = load_baseline(path)

        assert set(loaded.keys()) == {"all10_5d", "swing5_3d"}
        assert loaded["all10_5d"].sharpe_ratio == pytest.approx(1.2)
        assert loaded["all10_5d"].alpha_cum_pct == pytest.approx(0.08)
        assert loaded["swing5_3d"].sharpe_ratio is None
        assert loaded["swing5_3d"].alpha_cum_pct == pytest.approx(-0.02)

    def test_load_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "does_not_exist.json"
            assert load_baseline(path) == {}

    def test_save_includes_metadata_and_thresholds(self):
        metrics = {"x": _bm()}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "baseline.json"
            save_baseline(metrics, path=path, git_commit="abc1234")
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        assert raw["metadata"]["git_commit"] == "abc1234"
        assert raw["metadata"]["tolerance_deltas"] == DEFAULT_TOLERANCE_DELTAS
        assert "created_at" in raw["metadata"]


# ====================================================================== #
# P3-D: Discord formatter
# ====================================================================== #


class TestFormatRegressionsForDiscord:
    def test_empty_returns_empty_string(self):
        assert format_regressions_for_discord([]) == ""

    def test_single_regression_renders(self):
        regs = [
            RegressionFinding(
                portfolio_name="mom5_3d",
                metric="sharpe_ratio",
                baseline_value=1.0,
                current_value=0.5,
                delta=-0.5,
                threshold=0.2,
                is_regression=True,
                reason="current(+0.5000) - baseline(+1.0000) = -0.5000，閾值=-0.2000",
            )
        ]
        out = format_regressions_for_discord(regs)
        assert "Baseline Regression" in out
        assert "mom5_3d" in out
        assert "sharpe_ratio" in out

    def test_groups_by_portfolio_and_caps_at_3_per(self):
        """同 portfolio 超過 3 項只顯示前 3 項，避免爆字元。"""
        regs = [
            RegressionFinding(
                portfolio_name="X",
                metric=f"metric_{i}",
                baseline_value=0,
                current_value=-1,
                delta=-1,
                threshold=0.1,
                is_regression=True,
                reason="reason",
            )
            for i in range(5)
        ]
        out = format_regressions_for_discord(regs)
        assert out.count("metric_") == 3


# ====================================================================== #
# P3-E: validate-baseline CLI exit codes
# ====================================================================== #


class TestValidateBaselineExitCodes:
    def test_missing_baseline_returns_exit_2(self, monkeypatch, tmp_path):
        """baseline 檔不存在 → exit 2。"""
        from src.cli import baseline_cmd

        monkeypatch.setattr(baseline_cmd, "BASELINE_PATH", tmp_path / "nope.json")
        # init_db / collect 也要 monkeypatch 防止 DB 副作用
        monkeypatch.setattr(baseline_cmd, "collect_current_metrics", lambda *a, **kw: {})

        from src.cli.helpers import init_db as real_init_db  # noqa: F401

        monkeypatch.setattr("src.cli.baseline_cmd.load_baseline", lambda *a, **kw: {})
        import argparse

        # init_db 為 no-op
        import src.cli.helpers as helpers

        monkeypatch.setattr(helpers, "init_db", lambda: None)

        exit_code = baseline_cmd.cmd_validate_baseline(argparse.Namespace(tolerance=1.0, lookback_days=90, quiet=True))
        assert exit_code == 2

    def test_pass_returns_exit_0(self, monkeypatch):
        import argparse

        import src.cli.helpers as helpers
        from src.cli import baseline_cmd

        monkeypatch.setattr(helpers, "init_db", lambda: None)
        bm = _bm(name="x")
        cm = _cm(name="x")
        monkeypatch.setattr(baseline_cmd, "load_baseline", lambda *a, **kw: {"x": bm})
        monkeypatch.setattr(baseline_cmd, "collect_current_metrics", lambda *a, **kw: {"x": cm})

        exit_code = baseline_cmd.cmd_validate_baseline(argparse.Namespace(tolerance=1.0, lookback_days=90, quiet=True))
        assert exit_code == 0

    def test_regression_returns_exit_1(self, monkeypatch):
        import argparse

        import src.cli.helpers as helpers
        from src.cli import baseline_cmd

        monkeypatch.setattr(helpers, "init_db", lambda: None)
        bm = _bm(name="x", sharpe=1.0)
        cm = _cm(name="x", sharpe=0.3)  # -0.7 退化 > 0.20
        monkeypatch.setattr(baseline_cmd, "load_baseline", lambda *a, **kw: {"x": bm})
        monkeypatch.setattr(baseline_cmd, "collect_current_metrics", lambda *a, **kw: {"x": cm})

        exit_code = baseline_cmd.cmd_validate_baseline(argparse.Namespace(tolerance=1.0, lookback_days=90, quiet=True))
        assert exit_code == 1

    def test_update_baseline_requires_confirm(self, monkeypatch):
        import argparse

        import src.cli.helpers as helpers
        from src.cli import baseline_cmd

        monkeypatch.setattr(helpers, "init_db", lambda: None)
        exit_code = baseline_cmd.cmd_update_baseline(
            argparse.Namespace(confirm=False, portfolios=None, lookback_days=90)
        )
        assert exit_code == 2
