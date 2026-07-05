"""P0 止血包 #1 — DB 備份（backup_db + 異地副本 + 保留策略 + Step 18）測試。"""

from __future__ import annotations

import os
import time

import pytest

import src.data.database as db_mod
from src.config import BackupConfig, Settings
from src.data.database import _prune_old_backups, backup_db


@pytest.fixture()
def fake_db(tmp_path, monkeypatch):
    """假 SQLite 檔 + 隔離的 settings（offsite 指向 tmp_path/offsite）。"""
    db_path = tmp_path / "stock.db"
    db_path.write_bytes(b"fake sqlite content")

    offsite = tmp_path / "offsite"
    test_settings = Settings(
        database={"url": f"sqlite:///{db_path}"},
        backup=BackupConfig(offsite_dir=str(offsite), retention_days=7),
    )
    monkeypatch.setattr(db_mod, "settings", test_settings)

    # integrity check 走全域 engine（指向真 DB），測試中跳過
    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def execute(self, *a, **kw):
            class _R:
                @staticmethod
                def scalar():
                    return "ok"

            return _R()

    class _FakeEngine:
        @staticmethod
        def connect():
            return _FakeConn()

    monkeypatch.setattr(db_mod, "engine", _FakeEngine())
    return db_path, offsite, test_settings


class TestBackupDb:
    def test_creates_local_and_offsite_copies(self, fake_db):
        db_path, offsite, _ = fake_db

        result = backup_db()

        assert result is not None
        assert result.exists()
        assert result.parent == db_path.parent / "backups"
        assert result.suffix == ".bak"
        # 異地副本同名存在
        offsite_copy = offsite / result.name
        assert offsite_copy.exists()
        assert offsite_copy.read_bytes() == db_path.read_bytes()

    def test_empty_offsite_dir_skips_offsite(self, fake_db, monkeypatch):
        db_path, offsite, test_settings = fake_db
        test_settings.backup.offsite_dir = ""

        result = backup_db()

        assert result is not None and result.exists()
        assert not offsite.exists(), "offsite_dir 空字串時不應建立異地目錄"

    def test_offsite_failure_does_not_raise(self, fake_db, monkeypatch):
        """異地 copy 失敗（iCloud 離線等）只 warning，不影響本地備份回傳。"""
        db_path, offsite, _ = fake_db

        original_copy2 = db_mod.shutil.copy2
        calls = {"n": 0}

        def _copy2_fail_second(src, dst):
            calls["n"] += 1
            if calls["n"] >= 2:  # 第 1 次本地、第 2 次異地
                raise OSError("iCloud 離線")
            return original_copy2(src, dst)

        monkeypatch.setattr(db_mod.shutil, "copy2", _copy2_fail_second)

        result = backup_db()

        assert result is not None and result.exists(), "本地備份不受異地失敗影響"
        assert not any(offsite.glob("*.bak"))

    def test_missing_db_returns_none(self, fake_db):
        db_path, _, _ = fake_db
        db_path.unlink()
        assert backup_db() is None

    def test_retention_prunes_old_backups(self, fake_db):
        """本地與異地都套保留策略：超過 retention_days 的 .bak 被刪除。"""
        db_path, offsite, _ = fake_db
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir()
        offsite.mkdir()

        old_time = time.time() - 10 * 86400  # 10 天前 > retention 7 天
        for d in (backup_dir, offsite):
            stale = d / "stock_old.bak"
            stale.write_bytes(b"old")
            os.utime(stale, (old_time, old_time))
            fresh = d / "stock_fresh.bak"
            fresh.write_bytes(b"fresh")

        backup_db()

        assert not (backup_dir / "stock_old.bak").exists()
        assert not (offsite / "stock_old.bak").exists()
        assert (backup_dir / "stock_fresh.bak").exists()
        assert (offsite / "stock_fresh.bak").exists()


class TestPruneOldBackups:
    def test_only_bak_files_pruned(self, tmp_path):
        old_time = time.time() - 30 * 86400
        bak = tmp_path / "a.bak"
        bak.write_bytes(b"x")
        os.utime(bak, (old_time, old_time))
        other = tmp_path / "keep.txt"
        other.write_bytes(b"x")
        os.utime(other, (old_time, old_time))

        _prune_old_backups(tmp_path, retention_days=7)

        assert not bak.exists()
        assert other.exists(), "非 .bak 檔不應被清理"


class TestMorningStep18:
    def test_steps_contains_backup_step(self):
        """morning-routine _steps 應含 Step 18 DB 備份（結構測試，不整跑 routine）。"""
        import inspect

        from src.cli import morning_cmd

        src = inspect.getsource(morning_cmd)
        assert "DB 備份（本地 + 異地副本）" in src
        assert "_run_db_backup" in src

    def test_run_db_backup_prints_paths(self, fake_db, capsys):
        # morning_cmd._run_db_backup 內部 import settings/backup_db，
        # fake_db 已 patch db_mod.settings 與 engine；settings 另需 patch config 端
        import src.config as config_mod
        from src.cli.morning_cmd import _run_db_backup

        _, _, test_settings = fake_db
        orig = config_mod.settings
        config_mod.settings = test_settings
        try:
            _run_db_backup()
        finally:
            config_mod.settings = orig

        out = capsys.readouterr().out
        assert "本地備份" in out
        assert "異地副本" in out
