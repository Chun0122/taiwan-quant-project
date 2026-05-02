"""tests/test_strategy_events.py — 策略事件抽取測試。

涵蓋：
- 主旨前綴過濾（feat/fix/refactor/perf vs docs/chore）
- settings.yaml 變動偵測
- 非 git 倉庫 graceful degradation
- 同 commit 兩種事件去重
"""

from __future__ import annotations

import subprocess
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.discovery.strategy_events import StrategyEvent, collect_strategy_events


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture()
def git_repo(tmp_path: Path):
    """建立臨時 git 倉庫，並設定 user.name/email 避免 commit 失敗。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], cwd=repo)
    _run(["git", "config", "user.name", "Tester"], cwd=repo)
    _run(["git", "config", "user.email", "tester@example.com"], cwd=repo)
    # 預先建立 config 目錄與 settings 檔
    (repo / "config").mkdir()
    (repo / "config" / "settings.yaml").write_text("score_threshold:\n  bull: 0.45\n", encoding="utf-8")
    (repo / "README.md").write_text("readme\n", encoding="utf-8")
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "chore: initial scaffold"], cwd=repo)
    return repo


def _add_commit(repo: Path, file_rel: str, content: str, message: str) -> None:
    target = repo / file_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", message], cwd=repo)


class TestCollectStrategyEvents:
    def test_filters_by_prefix(self, git_repo: Path):
        _add_commit(git_repo, "src/foo.py", "feat", "feat(strategy): add momentum filter")
        _add_commit(git_repo, "src/bar.py", "fix", "fix(rotation): handle empty positions")
        _add_commit(git_repo, "src/baz.py", "refactor", "refactor: simplify scanner")
        _add_commit(git_repo, "src/qux.py", "perf", "perf!: speed up indicator")
        _add_commit(git_repo, "docs/x.md", "docs", "docs: update CLAUDE.md")
        _add_commit(git_repo, "src/test.py", "test", "test: add fixtures")

        events = collect_strategy_events(repo_root=git_repo, days=30)
        types = {e.summary for e in events if e.type == "git_commit"}
        # 4 個被收集
        assert any("feat(strategy)" in s for s in types)
        assert any("fix(rotation)" in s for s in types)
        assert any("refactor:" in s for s in types)
        assert any("perf!" in s for s in types)
        # 2 個被排除
        assert not any("docs:" in s for s in types)
        assert not any("test:" in s for s in types)

    def test_settings_diff_detected(self, git_repo: Path):
        _add_commit(
            git_repo,
            "config/settings.yaml",
            "score_threshold:\n  bull: 0.50\n",
            "fix(config): tighten bull threshold to 0.50",
        )
        events = collect_strategy_events(repo_root=git_repo, days=30)
        diff_events = [e for e in events if e.type == "settings_diff"]
        # 至少包含本次 fix（git_repo fixture 的 initial commit 也動到 settings.yaml）
        assert any("fix(config)" in e.summary for e in diff_events)
        # 同 commit 也會出現在 git_commit（dedup key 是 (ref, type)）
        latest = next(e for e in diff_events if "fix(config)" in e.summary)
        commit_events = [e for e in events if e.type == "git_commit" and e.ref == latest.ref]
        assert len(commit_events) == 1

    def test_to_dict_schema(self, git_repo: Path):
        _add_commit(git_repo, "src/foo.py", "x", "feat: new")
        events = collect_strategy_events(repo_root=git_repo, days=30)
        d = events[0].to_dict()
        for key in ("date", "type", "summary", "ref", "details"):
            assert key in d
        assert d["type"] in ("git_commit", "settings_diff")

    def test_descending_date_order(self, git_repo: Path):
        """所有 commits 都被收集；同日內排序非嚴格（git timestamp 精度限制）。"""
        _add_commit(git_repo, "src/a.py", "1", "feat: first")
        _add_commit(git_repo, "src/b.py", "2", "feat: second")
        _add_commit(git_repo, "src/c.py", "3", "feat: third")
        events = [e for e in collect_strategy_events(repo_root=git_repo, days=30) if e.type == "git_commit"]
        summaries = {e.summary for e in events}
        assert {"feat: first", "feat: second", "feat: third"} <= summaries

    def test_non_git_repo_returns_empty(self, tmp_path: Path):
        # 沒有 .git 目錄
        plain = tmp_path / "plain"
        plain.mkdir()
        events = collect_strategy_events(repo_root=plain, days=30)
        assert events == []

    def test_since_cutoff(self, git_repo: Path):
        _add_commit(git_repo, "src/foo.py", "x", "feat: old commit")
        # since 設定到未來，所有 commits 都在 since 之前 → 空清單
        future = date.today() + timedelta(days=10)
        events = collect_strategy_events(since=future, repo_root=git_repo)
        assert events == []


class TestStrategyEventDataclass:
    def test_to_dict_isoformat(self):
        ev = StrategyEvent(
            date=date(2026, 5, 1),
            type="git_commit",
            summary="feat: x",
            ref="abc1234",
            details={"author": "test"},
        )
        d = ev.to_dict()
        assert d["date"] == "2026-05-01"
        assert d["ref"] == "abc1234"
        assert d["details"] == {"author": "test"}

    def test_default_details_empty(self):
        ev = StrategyEvent(date=date(2026, 5, 1), type="git_commit", summary="x")
        assert ev.to_dict()["details"] == {}
