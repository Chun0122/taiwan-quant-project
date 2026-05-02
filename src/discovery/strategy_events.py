"""策略調整事件抽取 — 供 dashboard 與 iOS App 「事件流」頁使用。

最簡版（v1）只做兩件事：
  1. git_commit：抓最近 N 天的 commits，主旨開頭為 feat/fix/refactor 範圍才算
  2. settings_diff：偵測這些 commits 中 config/settings.yaml 的變動

之後可擴充 ic_auto_adjust / kill_switch（v2）。
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# 主旨開頭過濾（避免 commit message lint / docs / chore 雜訊）
_INCLUDED_PREFIX = re.compile(r"^(feat|fix|refactor|perf)(\(|:|\!)", re.IGNORECASE)

_SETTINGS_FILE = "config/settings.yaml"


@dataclass
class StrategyEvent:
    """單一策略調整事件。"""

    date: date
    type: str  # "git_commit" | "settings_diff"
    summary: str
    ref: str | None = None  # commit hash 短碼
    details: dict = field(default_factory=dict)
    # 內部用：同日多 commit 排序鍵（committer timestamp ISO 字串），不序列化到 JSON
    _sort_ts: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "type": self.type,
            "summary": self.summary,
            "ref": self.ref,
            "details": dict(self.details),
        }


def _run_git(args: list[str], cwd: Path) -> str:
    """執行 git 指令並回傳 stdout（失敗時 raise CalledProcessError）。"""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    return proc.stdout


def _collect_git_commits(repo_root: Path, since: date) -> list[StrategyEvent]:
    """抓 since 至今 commits 中主旨匹配 feat|fix|refactor|perf 的記錄。"""
    events: list[StrategyEvent] = []
    since_str = since.isoformat()
    # hash<TAB>committer-iso-strict<TAB>subject
    fmt = "%h%x09%cI%x09%s"
    try:
        out = _run_git(
            [
                "log",
                f"--since={since_str}",
                f"--pretty=format:{fmt}",
                "--no-merges",
            ],
            cwd=repo_root,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("git log 失敗，無法收集 commit 事件: %s", exc)
        return events

    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        sha, ts_iso, subject = parts
        if not _INCLUDED_PREFIX.match(subject):
            continue
        try:
            commit_date = date.fromisoformat(ts_iso[:10])
        except ValueError:
            continue
        events.append(
            StrategyEvent(
                date=commit_date,
                type="git_commit",
                summary=subject.strip(),
                ref=sha.strip(),
                details={},
                _sort_ts=ts_iso,
            )
        )
    return events


def _collect_settings_diffs(repo_root: Path, since: date) -> list[StrategyEvent]:
    """抓 since 至今 config/settings.yaml 有變動的 commits 摘要。

    僅標記「該 commit 動到 settings.yaml」，不解析具體欄位 diff（v1 簡化）。
    """
    events: list[StrategyEvent] = []
    settings_path = repo_root / _SETTINGS_FILE
    if not settings_path.exists():
        return events

    since_str = since.isoformat()
    fmt = "%h%x09%cI%x09%s"
    try:
        out = _run_git(
            [
                "log",
                f"--since={since_str}",
                f"--pretty=format:{fmt}",
                "--no-merges",
                "--",
                _SETTINGS_FILE,
            ],
            cwd=repo_root,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("git log -- %s 失敗: %s", _SETTINGS_FILE, exc)
        return events

    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        sha, ts_iso, subject = parts
        try:
            commit_date = date.fromisoformat(ts_iso[:10])
        except ValueError:
            continue
        events.append(
            StrategyEvent(
                date=commit_date,
                type="settings_diff",
                summary=f"config/settings.yaml 變動：{subject.strip()}",
                ref=sha.strip(),
                details={"file": _SETTINGS_FILE},
                _sort_ts=ts_iso,
            )
        )
    return events


def collect_strategy_events(
    since: date | None = None,
    repo_root: Path | None = None,
    days: int = 30,
) -> list[StrategyEvent]:
    """收集策略調整事件，依 date 降冪 + ref 排序去重。

    Args:
        since:     起始日期（含），預設為今天往前 `days` 天。
        repo_root: git 倉庫根目錄；預設由 src 模組路徑推導。
        days:     `since` 為 None 時的預設回溯天數。

    Returns:
        list[StrategyEvent]：排序後的事件清單。失敗時回傳空清單（不拋例外）。
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    if since is None:
        since = (datetime.now().date()) - timedelta(days=days)

    if not (repo_root / ".git").exists():
        logger.info("非 git 倉庫（%s），跳過策略事件收集", repo_root)
        return []

    events: list[StrategyEvent] = []
    events.extend(_collect_git_commits(repo_root, since))
    events.extend(_collect_settings_diffs(repo_root, since))

    # 同一 commit 同時出現在兩個來源時，settings_diff 較具體，保留它
    seen: dict[tuple[str | None, str], StrategyEvent] = {}
    for ev in events:
        key = (ev.ref, ev.type)
        seen[key] = ev
    deduped = list(seen.values())
    # 主排序：committer 時間戳（同日內保序）；次排序：ref（決定性 fallback）
    deduped.sort(key=lambda e: (e._sort_ts, e.ref or ""), reverse=True)
    return deduped
