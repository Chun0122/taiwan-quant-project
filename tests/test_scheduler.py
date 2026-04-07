"""排程模組測試 — launchd_task / windows_task / auto 偵測。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# _resolve_schedule_mode
# ---------------------------------------------------------------------------


class TestResolveScheduleMode:
    """auto 模式平台偵測。"""

    def _resolve(self, mode: str) -> str:
        from src.cli.misc_cmd import _resolve_schedule_mode

        return _resolve_schedule_mode(mode)

    def test_explicit_simple(self):
        assert self._resolve("simple") == "simple"

    def test_explicit_windows(self):
        assert self._resolve("windows") == "windows"

    def test_explicit_macos(self):
        assert self._resolve("macos") == "macos"

    def test_auto_darwin(self):
        with patch("sys.platform", "darwin"):
            assert self._resolve("auto") == "macos"

    def test_auto_win32(self):
        with patch("sys.platform", "win32"):
            assert self._resolve("auto") == "windows"


# ---------------------------------------------------------------------------
# launchd_task — 產生 .sh / .plist
# ---------------------------------------------------------------------------


class TestLaunchdTask:
    """macOS LaunchAgent 腳本產生。"""

    @pytest.fixture()
    def script_dir(self, tmp_path: Path) -> Path:
        """以 tmp_path 模擬 PROJECT_ROOT。"""
        return tmp_path

    def test_generate_daily_scripts(self, script_dir: Path):
        with patch("src.scheduler.launchd_task.PROJECT_ROOT", script_dir):
            from src.scheduler.launchd_task import generate_scripts

            sh_path, plist_path = generate_scripts()

        assert sh_path.exists()
        assert plist_path.exists()

        sh_text = sh_path.read_text(encoding="utf-8")
        assert "morning-routine --notify" in sh_text
        assert "#!/bin/bash" in sh_text
        assert "venv/bin/activate" in sh_text

        plist_text = plist_path.read_text(encoding="utf-8")
        assert "com.taiwan-quant.daily-sync" in plist_text
        assert "<key>Hour</key>" in plist_text
        assert "<integer>23</integer>" in plist_text
        assert str(sh_path) in plist_text

    def test_generate_weekly_scripts(self, script_dir: Path):
        with patch("src.scheduler.launchd_task.PROJECT_ROOT", script_dir):
            from src.scheduler.launchd_task import generate_weekly_scripts

            sh_path, plist_path = generate_weekly_scripts()

        assert sh_path.exists()
        assert plist_path.exists()

        sh_text = sh_path.read_text(encoding="utf-8")
        assert "sync-holding" in sh_text

        plist_text = plist_path.read_text(encoding="utf-8")
        assert "com.taiwan-quant.weekly-holding" in plist_text
        # Weekday 4 = Thursday
        assert "<key>Weekday</key>" in plist_text
        assert "<integer>4</integer>" in plist_text

    def test_logs_dir_created(self, script_dir: Path):
        with patch("src.scheduler.launchd_task.PROJECT_ROOT", script_dir):
            from src.scheduler.launchd_task import generate_scripts

            generate_scripts()

        assert (script_dir / "logs").is_dir()

    def test_scripts_dir_created(self, script_dir: Path):
        with patch("src.scheduler.launchd_task.PROJECT_ROOT", script_dir):
            from src.scheduler.launchd_task import generate_scripts

            generate_scripts()

        assert (script_dir / "scripts").is_dir()


# ---------------------------------------------------------------------------
# windows_task — 基本驗證（確保既有功能未受影響）
# ---------------------------------------------------------------------------


class TestWindowsTask:
    """Windows Task Scheduler 腳本產生。"""

    def test_generate_daily_scripts(self, tmp_path: Path):
        with patch("src.scheduler.windows_task.PROJECT_ROOT", tmp_path):
            from src.scheduler.windows_task import generate_scripts

            bat_path, xml_path = generate_scripts()

        assert bat_path.exists()
        assert xml_path.exists()

        bat_text = bat_path.read_text(encoding="utf-8")
        assert "morning-routine --notify" in bat_text

    def test_generate_weekly_scripts(self, tmp_path: Path):
        with patch("src.scheduler.windows_task.PROJECT_ROOT", tmp_path):
            from src.scheduler.windows_task import generate_weekly_scripts

            bat_path, xml_path = generate_weekly_scripts()

        assert bat_path.exists()
        assert xml_path.exists()

        bat_text = bat_path.read_text(encoding="utf-8")
        assert "sync-holding" in bat_text
