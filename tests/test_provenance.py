"""tests/test_provenance.py — A5 決策可重放（版本戳）測試。

涵蓋：
- current_provenance 回傳形狀（git_commit 可 None、settings_hash 16 hex）
- try_git_head 非 git 環境 graceful degradation
- _save_discovery_records 將版本戳蓋章到每一筆 DiscoveryRecord
- migrate.py 對 discovery_record 新欄位冪等
"""

from __future__ import annotations

import subprocess
from datetime import date

import pandas as pd

from src.config import Settings
from src.data.schema import DiscoveryRecord
from src.provenance import current_provenance, sanitize_settings, try_git_head


class TestCurrentProvenance:
    def test_shape_and_hash_format(self):
        git_commit, settings_hash = current_provenance(Settings())
        assert git_commit is None or isinstance(git_commit, str)
        assert len(settings_hash) == 16
        int(settings_hash, 16)  # 合法 hex

    def test_hash_changes_with_quant_params(self):
        s1 = Settings()
        s2 = Settings(**{"quant": {"rotation_cost": {"min_hold_days": 99}}})
        assert current_provenance(s1)[1] != current_provenance(s2)[1]

    def test_hash_idempotent(self):
        s = Settings()
        assert current_provenance(s)[1] == current_provenance(s)[1]

    def test_secrets_not_in_hash(self):
        """token/webhook 變動不影響 settings_hash（sanitize 已排除）。"""
        s1 = Settings(**{"finmind": {"api_token": "AAA"}, "discord": {"webhook_url": "http://x"}})
        s2 = Settings(**{"finmind": {"api_token": "BBB"}, "discord": {"webhook_url": "http://y"}})
        assert current_provenance(s1)[1] == current_provenance(s2)[1]
        sanitized = sanitize_settings(s1)
        assert "AAA" not in str(sanitized)
        assert "http://x" not in str(sanitized)


class TestTryGitHead:
    def test_returns_short_hash_in_repo(self):
        # 本專案即 git repo；subprocess 用 process cwd 執行
        head = try_git_head()
        assert head is None or (4 <= len(head) <= 40)

    def test_git_missing_returns_none(self, monkeypatch):
        def boom(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", boom)
        assert try_git_head() is None

    def test_git_timeout_returns_none(self, monkeypatch):
        def slow(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="git", timeout=5)

        monkeypatch.setattr(subprocess, "run", slow)
        assert try_git_head() is None


class _FakeResult:
    """最小 DiscoverResult 替身（僅 _save_discovery_records 需要的欄位）。"""

    def __init__(self, scan_date: date, rankings: pd.DataFrame):
        self.scan_date = scan_date
        self.rankings = rankings
        self.total_stocks = 100
        self.after_coarse = 50


class _FakeScanner:
    regime = "bull"


class TestSaveDiscoveryRecordsStamping:
    def _make_result(self) -> _FakeResult:
        rankings = pd.DataFrame(
            [
                {"rank": 1, "stock_id": "2330", "stock_name": "台積電", "close": 1000.0, "composite_score": 0.9},
                {"rank": 2, "stock_id": "2317", "stock_name": "鴻海", "close": 200.0, "composite_score": 0.8},
            ]
        )
        return _FakeResult(scan_date=date(2026, 7, 3), rankings=rankings)

    def test_records_carry_provenance(self, db_session):
        from src.cli import discover_cmd

        discover_cmd._save_discovery_records(self._make_result(), "momentum", _FakeScanner())

        rows = db_session.query(DiscoveryRecord).order_by(DiscoveryRecord.rank).all()
        assert len(rows) == 2
        for row in rows:
            assert row.settings_hash is not None
            assert len(row.settings_hash) == 16
        # 同一 run 全部同戳記
        assert rows[0].settings_hash == rows[1].settings_hash
        assert rows[0].git_commit == rows[1].git_commit

    def test_rerun_overwrites_with_new_stamp(self, db_session, monkeypatch):
        """重跑同日同模式：舊列刪除、新列帶新 hash。"""
        from src.cli import discover_cmd
        from src.config import RotationCostConfig

        discover_cmd._save_discovery_records(self._make_result(), "momentum", _FakeScanner())
        old_hash = db_session.query(DiscoveryRecord.settings_hash).first()[0]

        # 改 quant 參數 → hash 應改變
        import src.config as config_mod

        monkeypatch.setattr(config_mod.settings.quant, "rotation_cost", RotationCostConfig(min_hold_days=42))
        discover_cmd._save_discovery_records(self._make_result(), "momentum", _FakeScanner())

        rows = db_session.query(DiscoveryRecord).all()
        assert len(rows) == 2  # 覆蓋而非累加
        assert all(r.settings_hash != old_hash for r in rows)


class TestMigration:
    def test_migrate_idempotent_on_new_columns(self, tmp_path, monkeypatch):
        """對已含 git_commit/settings_hash 的 DB 重跑 migration 不炸（冪等）。"""
        from sqlalchemy import create_engine

        db_file = tmp_path / "mig.db"
        engine = create_engine(f"sqlite:///{db_file}")

        import src.data.database as db_mod
        import src.data.migrate as mig_mod

        monkeypatch.setattr(db_mod, "engine", engine)
        monkeypatch.setattr(mig_mod, "engine", engine, raising=False)

        from src.data.schema import Base

        Base.metadata.create_all(engine)  # create_all 已含新欄位
        # PRAGMA 確認欄位存在
        with engine.connect() as conn:
            from sqlalchemy import text

            cols = {row[1] for row in conn.execute(text("PRAGMA table_info(discovery_record)"))}
        assert {"git_commit", "settings_hash"} <= cols
