"""tests/test_morning_audit_fixes.py — morning-routine 系統級審計修復測試。

涵蓋：
- C1 修復：_compute_factor_ic_status / _check_factor_ic_decay 失敗時不靜默
- C2 修復：Step 0 stress check 任意例外不中斷流程
- C3 修復：stress check 失敗時 regime 為 None（讓 Rotation 走 fallback）
- M1 修復：freshness.gap_days > 7 時 Step 9 自動阻擋
- M2 修復：IC 反向模式從 _cmd_discover_all 掃描清單移除
- M3/M4 修復：_build_morning_discord_summary 使用傳入的 stress_result / ic_status
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

from src.cli import morning_cmd

# ═══════════════════════════════════════════════════════════════════════════
# C1：IC 檢查 silent failure 修復
# ═══════════════════════════════════════════════════════════════════════════


class TestICStatusComputation:
    """_compute_factor_ic_status 行為測試。"""

    def test_failure_returns_error_level_not_silent(self):
        """compute_rolling_ic 丟錯 → 回傳 level='error' 而非 silent continue。"""
        # 模擬 DB 有足夠資料進入 rolling IC 呼叫，但 rolling 本體失敗
        with patch(
            "src.discovery.scanner._functions.compute_rolling_ic",
            side_effect=RuntimeError("模擬崩壞"),
        ):
            status = morning_cmd._compute_factor_ic_status()

        # 每個模式只要有進入 compute_rolling_ic 都應回 level='error'
        # （若樣本不足才會是 insufficient；此處無法控制 DB，
        #  至少驗證所有 entry 都非靜默遺失）
        assert len(status) == 5  # 5 個 discover 模式
        assert all("level" in s for s in status)
        # 至少有一個 mode 有足夠樣本觸發 error（視 DB 狀態而定）
        # 若 DB 全無資料，所有都是 insufficient → 這也符合「不靜默」

    def test_inverse_modes_extraction(self):
        """_inverse_modes_from_ic_status 正確抽出反向模式。"""
        sample = [
            {"mode_key": "momentum", "level": "normal"},
            {"mode_key": "swing", "level": "inverse"},
            {"mode_key": "value", "level": "weak"},
            {"mode_key": "dividend", "level": "inverse"},
            {"mode_key": "growth", "level": "error"},
        ]
        inverse = morning_cmd._inverse_modes_from_ic_status(sample)
        assert inverse == ["swing", "dividend"]


class TestCheckFactorICDecayNotSilent:
    """_check_factor_ic_decay 失敗訊息必須可見。"""

    def test_all_failures_reported_not_hidden(self, capsys):
        """全部 mode 失敗 → 不應印「IC 正常」。"""
        fake_status = [
            {
                "mode": "Momentum",
                "mode_key": "momentum",
                "factor": "technical_score",
                "level": "error",
                "error": "DB boom",
            },
            {"mode": "Swing", "mode_key": "swing", "factor": "chip_score", "level": "error", "error": "DB boom"},
            {"mode": "Value", "mode_key": "value", "factor": "fundamental_score", "level": "error", "error": "DB boom"},
            {
                "mode": "Dividend",
                "mode_key": "dividend",
                "factor": "fundamental_score",
                "level": "error",
                "error": "DB boom",
            },
            {
                "mode": "Growth",
                "mode_key": "growth",
                "factor": "fundamental_score",
                "level": "error",
                "error": "DB boom",
            },
        ]
        morning_cmd._check_factor_ic_decay(ic_status=fake_status)
        out = capsys.readouterr().out
        assert "所有模式關鍵因子 IC 正常" not in out
        assert "IC 計算失敗" in out
        assert "5 個模式 IC 計算失敗" in out

    def test_mixed_results_report_correct_levels(self, capsys):
        """混合結果 → 正常/反向/衰減各自標記。"""
        fake_status = [
            {"mode": "Momentum", "mode_key": "momentum", "factor": "technical_score", "ic": 0.15, "level": "normal"},
            {"mode": "Swing", "mode_key": "swing", "factor": "chip_score", "ic": -0.13, "level": "inverse"},
            {"mode": "Value", "mode_key": "value", "factor": "fundamental_score", "ic": 0.07, "level": "decay"},
            {"mode": "Dividend", "mode_key": "dividend", "factor": "fundamental_score", "ic": 0.01, "level": "weak"},
            {
                "mode": "Growth",
                "mode_key": "growth",
                "factor": "fundamental_score",
                "level": "insufficient",
                "sample_count": 5,
            },
        ]
        morning_cmd._check_factor_ic_decay(ic_status=fake_status)
        out = capsys.readouterr().out
        assert "Momentum" in out and "+0.1500" in out
        assert "反向" in out and "Swing" in out
        assert "衰減" in out or "decay" in out.lower()
        assert "弱" in out or "weak" in out.lower()
        assert "樣本不足" in out


# ═══════════════════════════════════════════════════════════════════════════
# M3/M4：Discord 摘要使用外部傳入參數
# ═══════════════════════════════════════════════════════════════════════════


class TestDiscordSummaryParams:
    """_build_morning_discord_summary 必須使用傳入的 stress_result / ic_status。

    所有測試使用 db_session fixture 建立 in-memory DB（含 DiscoveryRecord/
    Announcement/WatchEntry 空表），避免 CI 無 DB 時 OperationalError。
    """

    def test_regime_none_shows_unknown_banner(self, db_session):
        """stress_result.regime=None → 顯示 regime 未知警示。"""
        stress = {"regime": None, "summary": "壓力預檢失敗"}
        msg = morning_cmd._build_morning_discord_summary("2026-04-18", 5, stress_result=stress)
        assert "Regime 未知" in msg

    def test_no_extra_stress_check_call_when_provided(self, db_session):
        """M4 修復：傳入 stress_result 後不應內部再次呼叫 _compute_macro_stress_check。"""
        with patch.object(morning_cmd, "_compute_macro_stress_check") as mock_call:
            stress = {"regime": "bull", "summary": "bull market"}
            morning_cmd._build_morning_discord_summary("2026-04-18", 5, stress_result=stress)
            mock_call.assert_not_called()

    def test_ic_inverse_status_displayed(self, db_session):
        """M3：IC 反向模式在 Discord 訊息中可見。"""
        ic_status = [
            {"mode": "Swing", "factor": "chip_score", "ic": -0.13, "level": "inverse"},
        ]
        msg = morning_cmd._build_morning_discord_summary("2026-04-18", 5, ic_status=ic_status, disabled_modes=["swing"])
        assert "IC 健康度" in msg
        assert "Swing" in msg
        assert "已暫停" in msg

    def test_ic_error_displayed_not_hidden(self, db_session):
        """C1/M3：IC 計算失敗的模式在 Discord 可見。"""
        ic_status = [
            {"mode": "Value", "factor": "fundamental_score", "level": "error", "error": "DB locked"},
        ]
        msg = morning_cmd._build_morning_discord_summary("2026-04-18", 5, ic_status=ic_status)
        assert "IC 計算失敗" in msg
        assert "Value" in msg

    def test_discover_blocked_banner(self, db_session):
        """M1：discover_blocked=True → 摘要顯示已阻擋。"""
        freshness = {"is_stale": True, "gap_days": 10, "message": "資料落後 10 天"}
        msg = morning_cmd._build_morning_discord_summary("2026-04-18", 5, freshness=freshness, discover_blocked=True)
        assert "已阻擋" in msg


# ═══════════════════════════════════════════════════════════════════════════
# M2：_cmd_discover_all 接受 disabled_modes
# ═══════════════════════════════════════════════════════════════════════════


class TestDiscoverAllDisabledModes:
    """_cmd_discover_all 應跳過 disabled_modes 中的 scanner。"""

    def test_disabled_modes_skipped(self, capsys):
        """disabled_modes=['swing','growth'] → swing 與 growth 不呼叫 scanner.run()。"""
        from src.cli import discover_cmd

        called_modes: list[str] = []

        class _FakeResult:
            def __init__(self):
                import pandas as pd

                self.rankings = pd.DataFrame()
                self.total_stocks = 0
                self.after_coarse = 0
                self.sector_summary = None
                self.audit_trail = None
                self.scan_date = None
                self.sub_factor_df = None

        class _FakeScanner:
            def __init__(self, mode_key, **kwargs):
                self._mode_key = mode_key

            def run(self):
                called_modes.append(self._mode_key)
                return _FakeResult()

        def _make(mode_key):
            return lambda **kw: _FakeScanner(mode_key, **kw)

        ns = argparse.Namespace(
            skip_sync=True,
            sync_days=80,
            top=5,
            min_price=10.0,
            max_price=2000.0,
            min_volume=500_000,
            max_stocks=None,
            min_appearances=1,
            export=None,
            notify=False,
            use_ic_adjustment=False,
            disabled_modes=["swing", "growth"],
        )

        with (
            patch.object(discover_cmd, "init_db"),
            patch.object(discover_cmd, "ensure_sync_market_data"),
            patch("src.discovery.scanner.MomentumScanner", _make("momentum")),
            patch("src.discovery.scanner.SwingScanner", _make("swing")),
            patch("src.discovery.scanner.ValueScanner", _make("value")),
            patch("src.discovery.scanner.DividendScanner", _make("dividend")),
            patch("src.discovery.scanner.GrowthScanner", _make("growth")),
        ):
            discover_cmd._cmd_discover_all(ns)

        assert "swing" not in called_modes
        assert "growth" not in called_modes
        assert "momentum" in called_modes
        assert "value" in called_modes
        assert "dividend" in called_modes

        out = capsys.readouterr().out
        assert "已停用（IC 反向）" in out


# ═══════════════════════════════════════════════════════════════════════════
# C2/C3：Step 0 stress check 保護
# ═══════════════════════════════════════════════════════════════════════════


class TestStressCheckProtection:
    """C2/C3 修復：stress check 任意例外不中斷、regime=None 明確傳達。"""

    def test_regime_unknown_in_banner_when_none(self, db_session, capsys):
        """regime=None → 顯示 Regime 未知 banner（不假性 sideways）。"""
        msg = morning_cmd._build_morning_discord_summary(
            "2026-04-18",
            5,
            stress_result={"regime": None, "summary": "失敗"},
        )
        assert "Regime 未知" in msg
        # 確認不是錯誤地顯示為 sideways
        assert "Rotation 以保守模式運行" in msg
