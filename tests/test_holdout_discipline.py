"""P1 任務 7 — Out-of-Sample Hold-Out 紀律測試。

涵蓋：
  P7-A compute_default_holdout_start
  P7-B assess_holdout_violation 純函數（5 個 severity 分支）
  P7-C HoldoutAudit.is_clean 判定邏輯
  P7-D DiscoveryPerformance.evaluate 回傳 holdout_audit
  P7-E print_performance_report 顯示 banner
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.discovery.performance import (
    DEFAULT_HOLDOUT_DAYS,
    DiscoveryPerformance,
    HoldoutAudit,
    _print_holdout_banner,
    assess_holdout_violation,
    compute_default_holdout_start,
    print_performance_report,
)

# ====================================================================== #
# P7-A: compute_default_holdout_start
# ====================================================================== #


class TestComputeDefaultHoldoutStart:
    def test_default_90_days_back(self):
        today = date(2026, 5, 17)
        h = compute_default_holdout_start(today=today)
        assert h == date(2026, 5, 17) - timedelta(days=DEFAULT_HOLDOUT_DAYS)

    def test_custom_days(self):
        today = date(2026, 5, 17)
        h = compute_default_holdout_start(today=today, days=30)
        assert h == date(2026, 4, 17)

    def test_today_none_uses_real_today(self):
        h1 = compute_default_holdout_start(days=90)
        # 不嚴格檢查當日（測試與 CI 跨時區），但 delta 應為 90 天
        assert (date.today() - h1).days == 90


# ====================================================================== #
# P7-B: assess_holdout_violation — 4 個 severity 分支
# ====================================================================== #


class TestAssessHoldoutViolation:
    HOLDOUT = date(2026, 2, 16)

    def test_severity_ok_when_end_before_holdout(self):
        audit = assess_holdout_violation(date(2026, 1, 1), date(2026, 2, 1), self.HOLDOUT)
        assert audit.severity == "ok"
        assert audit.in_sample_days == 32  # 1/1 ~ 2/1 inclusive
        assert audit.holdout_days == 0
        assert audit.is_clean() is True

    def test_severity_forward_when_start_at_or_after_holdout(self):
        audit = assess_holdout_violation(date(2026, 4, 1), date(2026, 5, 15), self.HOLDOUT)
        assert audit.severity == "forward"
        assert audit.in_sample_days == 0
        assert audit.holdout_days == 45  # 4/1 ~ 5/15 inclusive
        assert audit.is_clean() is True  # forward test 仍算乾淨

    def test_severity_forward_boundary_start_equals_holdout(self):
        """start == holdout_start → 視為純 forward（含當日）。"""
        audit = assess_holdout_violation(self.HOLDOUT, date(2026, 5, 15), self.HOLDOUT)
        assert audit.severity == "forward"

    def test_severity_partial_when_range_crosses_boundary(self):
        audit = assess_holdout_violation(date(2026, 1, 1), date(2026, 5, 15), self.HOLDOUT)
        assert audit.severity == "partial"
        assert audit.in_sample_days > 0
        assert audit.holdout_days > 0
        assert audit.is_clean() is False

    def test_severity_no_range_when_endpoints_missing(self):
        audit = assess_holdout_violation(None, None, self.HOLDOUT)
        assert audit.severity == "no_range"
        # 不視為乾淨（因為無法保證）
        assert audit.is_clean() is False

    def test_partial_audit_includes_suggested_split_in_message(self):
        audit = assess_holdout_violation(date(2026, 1, 1), date(2026, 5, 15), self.HOLDOUT)
        # 建議拆兩段：in-sample 到 2/15、OOS 從 2/16 起
        assert "--end 2026-02-15" in audit.message
        assert "--start 2026-02-16" in audit.message

    def test_ignored_flag_propagates(self):
        audit = assess_holdout_violation(date(2026, 4, 1), date(2026, 5, 15), self.HOLDOUT, ignored=True)
        assert audit.ignored is True
        # severity 不變
        assert audit.severity == "forward"


# ====================================================================== #
# P7-C: HoldoutAudit.is_clean 邊界
# ====================================================================== #


class TestHoldoutAuditIsClean:
    def test_ok_is_clean(self):
        a = HoldoutAudit(
            holdout_start=date(2026, 2, 16),
            backtest_start=date(2026, 1, 1),
            backtest_end=date(2026, 2, 1),
            in_sample_days=32,
            holdout_days=0,
            severity="ok",
            message="",
        )
        assert a.is_clean() is True

    def test_forward_is_clean(self):
        a = HoldoutAudit(
            holdout_start=date(2026, 2, 16),
            backtest_start=date(2026, 4, 1),
            backtest_end=date(2026, 5, 15),
            in_sample_days=0,
            holdout_days=45,
            severity="forward",
            message="",
        )
        assert a.is_clean() is True

    def test_partial_not_clean(self):
        a = HoldoutAudit(
            holdout_start=date(2026, 2, 16),
            backtest_start=date(2026, 1, 1),
            backtest_end=date(2026, 5, 15),
            in_sample_days=46,
            holdout_days=89,
            severity="partial",
            message="",
        )
        assert a.is_clean() is False

    def test_no_range_not_clean(self):
        a = HoldoutAudit(
            holdout_start=date(2026, 2, 16),
            backtest_start=None,
            backtest_end=None,
            in_sample_days=0,
            holdout_days=0,
            severity="no_range",
            message="",
        )
        assert a.is_clean() is False


# ====================================================================== #
# P7-D: DiscoveryPerformance.evaluate 整合 — holdout_audit 在 result 中
# ====================================================================== #


class TestDiscoveryPerformanceHoldoutAuditIntegration:
    def test_evaluate_returns_holdout_audit_even_when_empty(self, db_session, monkeypatch):
        """無 DiscoveryRecord 時，holdout_audit 仍必須出現在 result。"""
        from src.discovery import performance as perf_module

        # monkeypatch get_session 讓 DiscoveryPerformance 用 in-memory session
        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(perf_module, "get_session", lambda: _Ctx(db_session))

        perf = DiscoveryPerformance(
            mode="momentum",
            start_date="2026-01-01",
            end_date="2026-02-01",
            holdout_start=date(2026, 2, 16),
        )
        result = perf.evaluate()
        assert "holdout_audit" in result
        a = result["holdout_audit"]
        assert isinstance(a, HoldoutAudit)
        assert a.severity == "ok"

    def test_evaluate_holdout_audit_respects_custom_holdout_start(self, db_session, monkeypatch):
        from src.discovery import performance as perf_module

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(perf_module, "get_session", lambda: _Ctx(db_session))

        perf = DiscoveryPerformance(
            mode="momentum",
            start_date="2026-04-01",
            end_date="2026-04-30",
            holdout_start=date(2026, 5, 1),  # 自訂 holdout → 應為 ok
        )
        result = perf.evaluate()
        assert result["holdout_audit"].severity == "ok"

    def test_evaluate_ignore_holdout_propagates(self, db_session, monkeypatch):
        from src.discovery import performance as perf_module

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(perf_module, "get_session", lambda: _Ctx(db_session))

        perf = DiscoveryPerformance(
            mode="momentum",
            start_date="2026-01-01",
            end_date="2026-05-15",
            holdout_start=date(2026, 2, 16),
            ignore_holdout=True,
        )
        result = perf.evaluate()
        a = result["holdout_audit"]
        assert a.ignored is True
        # severity 不被 ignore 影響
        assert a.severity == "partial"


# ====================================================================== #
# P7-E: print_performance_report banner 輸出
# ====================================================================== #


class TestPrintHoldoutBanner:
    def test_ok_banner_contains_severity(self, capsys):
        a = assess_holdout_violation(date(2026, 1, 1), date(2026, 2, 1), date(2026, 2, 16))
        _print_holdout_banner(a)
        out = capsys.readouterr().out
        assert "Hold-Out 紀律審計" in out
        assert "ok" in out

    def test_partial_banner_shows_split_advice(self, capsys):
        a = assess_holdout_violation(date(2026, 1, 1), date(2026, 5, 15), date(2026, 2, 16))
        _print_holdout_banner(a)
        out = capsys.readouterr().out
        assert "partial" in out
        assert "--ignore-holdout" in out  # 解法提示

    def test_ignored_banner_shows_ack(self, capsys):
        a = assess_holdout_violation(date(2026, 4, 1), date(2026, 5, 15), date(2026, 2, 16), ignored=True)
        _print_holdout_banner(a)
        out = capsys.readouterr().out
        assert "已加 --ignore-holdout" in out

    def test_print_performance_report_prints_banner_when_audit_present(self, capsys):
        a = assess_holdout_violation(date(2026, 1, 1), date(2026, 2, 1), date(2026, 2, 16))
        result = {
            "summary": pd.DataFrame(),
            "by_scan": pd.DataFrame(),
            "detail": pd.DataFrame(),
            "holdout_audit": a,
        }
        print_performance_report(result, "momentum")
        out = capsys.readouterr().out
        assert "Hold-Out 紀律審計" in out
        # summary 空時不擲例外
        assert "無推薦記錄或無法計算績效" in out

    def test_print_performance_report_skips_banner_when_audit_absent(self, capsys):
        """舊版 result 無 holdout_audit key 時不擲例外（向後相容）。"""
        result = {
            "summary": pd.DataFrame(),
            "by_scan": pd.DataFrame(),
            "detail": pd.DataFrame(),
        }
        print_performance_report(result, "momentum")
        out = capsys.readouterr().out
        assert "Hold-Out 紀律審計" not in out
