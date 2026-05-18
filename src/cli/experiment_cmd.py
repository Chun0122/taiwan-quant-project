"""Experiment Registry CLI（P2 任務 10）— A/B 試驗歷史軌跡。

實驗註冊表記錄每次手動指定的 (git_commit, settings_hash, metrics) 三元組，
供：
  1. 重現：找 settings_hash X 對應的 settings JSON
  2. 對比：experiment compare exp_A exp_B → metric diff
  3. 追溯：experiment list 看歷史變動軌跡

與 baseline_metrics.json 互補：
  - baseline：當下單一 frozen reference（更新即覆寫）
  - experiment：多版本歷史軌跡（DB append-only 表）

用法：
  python main.py experiment record --description "test new chip weight"
  python main.py experiment list [--limit N]
  python main.py experiment show exp_20260518_abc123
  python main.py experiment compare exp_A exp_B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import secrets
import subprocess
from dataclasses import asdict
from datetime import date, datetime
from typing import Any

from sqlalchemy import select

from src.cli.helpers import safe_print as print
from src.config import Settings, settings
from src.data.database import get_session
from src.data.schema import ExperimentLog

logger = logging.getLogger(__name__)

# 哪些 settings 區塊納入 hash + snapshot（研究相關，排除 API token 等 infra 設定）
RESEARCH_SETTINGS_SECTIONS: tuple[str, ...] = ("quant", "fetcher")


# ---------------------------------------------------------------------------
# 純函數
# ---------------------------------------------------------------------------


def sanitize_settings(s: Settings) -> dict[str, Any]:
    """從 Settings 抽研究相關區塊；不含 API token / webhook URL。

    保留：quant / fetcher.watchlist + fetcher.default_start_date
    移除：finmind / anthropic.api_key / discord.webhook_url / database / logging
    """
    out: dict[str, Any] = {}
    if "quant" in RESEARCH_SETTINGS_SECTIONS:
        out["quant"] = s.quant.model_dump()
    if "fetcher" in RESEARCH_SETTINGS_SECTIONS:
        out["fetcher"] = {
            "default_start_date": s.fetcher.default_start_date,
            "watchlist": list(s.fetcher.watchlist),
        }
    return out


def compute_settings_hash(sanitized: dict[str, Any]) -> str:
    """sha256 前 16 hex chars 作為 settings 指紋（idempotent，跨機器一致）。"""
    canonical = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def generate_experiment_id(today: date | None = None) -> str:
    """格式：exp_YYYYMMDD_<6 hex>，例 exp_20260518_a3f8c1。"""
    today = today or date.today()
    suffix = secrets.token_hex(3)  # 6 hex chars
    return f"exp_{today.strftime('%Y%m%d')}_{suffix}"


def try_git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def collect_experiment_payload(
    description: str | None,
    *,
    cfg: Settings | None = None,
) -> dict[str, Any]:
    """純函數：產生 record 所需 payload dict（無 DB 副作用）。

    metrics 透過 src.cli.baseline_cmd.collect_current_metrics 取得；
    若該函式呼叫失敗回傳空 dict（仍允許 record 留 settings 軌跡）。
    """
    cfg = cfg or settings

    sanitized = sanitize_settings(cfg)
    settings_hash = compute_settings_hash(sanitized)

    metrics: dict[str, Any] = {}
    try:
        from src.cli.baseline_cmd import collect_current_metrics

        raw_metrics = collect_current_metrics()
        for name, m in raw_metrics.items():
            metrics[name] = asdict(m)
    except Exception as exc:
        logger.warning("collect_current_metrics 失敗，metrics 留空: %s", exc)

    return {
        "experiment_id": generate_experiment_id(),
        "git_commit": try_git_head(),
        "settings_hash": settings_hash,
        "settings_snapshot_json": json.dumps(sanitized, ensure_ascii=False, indent=2, default=str),
        "metrics_json": json.dumps(metrics, ensure_ascii=False, default=str),
        "description": description,
    }


def diff_metrics(metrics_a: dict, metrics_b: dict) -> list[dict]:
    """比較兩組 metrics 中各 portfolio 的逐 metric 差異。

    回傳 list of {portfolio, metric, a, b, delta}。差異為 b - a。
    """
    out: list[dict] = []
    all_portfolios = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
    for p in all_portfolios:
        ma = metrics_a.get(p) or {}
        mb = metrics_b.get(p) or {}
        keys = set(ma.keys()) | set(mb.keys())
        for k in sorted(keys):
            if k in ("portfolio_name", "as_of", "snapshot_count"):
                continue
            a_val = ma.get(k)
            b_val = mb.get(k)
            delta = None
            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                delta = b_val - a_val
            out.append({"portfolio": p, "metric": k, "a": a_val, "b": b_val, "delta": delta})
    return out


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------


def cmd_experiment(args: argparse.Namespace) -> int:
    """experiment subcommand 統一入口。"""
    from src.cli.helpers import init_db

    init_db()
    action = getattr(args, "exp_action", None)

    if action == "record":
        return _cmd_record(args)
    elif action == "list":
        return _cmd_list(args)
    elif action == "show":
        return _cmd_show(args)
    elif action == "compare":
        return _cmd_compare(args)
    else:
        print("使用方式: python main.py experiment {record|list|show|compare}")
        return 2


def _cmd_record(args: argparse.Namespace) -> int:
    payload = collect_experiment_payload(description=getattr(args, "description", None))

    with get_session() as session:
        # 處理 (極不可能) 的 experiment_id collision
        existing = session.execute(
            select(ExperimentLog).where(ExperimentLog.experiment_id == payload["experiment_id"])
        ).scalar_one_or_none()
        if existing is not None:
            print(f"[experiment] experiment_id 衝突: {payload['experiment_id']}（罕見），請重試")
            return 1

        session.add(ExperimentLog(**payload))
        session.commit()

    print(f"[experiment] 已記錄: {payload['experiment_id']}")
    print(f"  git_commit:    {payload['git_commit'] or '(未取得)'}")
    print(f"  settings_hash: {payload['settings_hash']}")
    metrics_obj = json.loads(payload["metrics_json"])
    print(f"  metrics:       {len(metrics_obj)} portfolio 已凍結")
    if payload["description"]:
        print(f"  description:   {payload['description']}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    limit = int(getattr(args, "limit", 20) or 20)
    with get_session() as session:
        rows = (
            session.execute(select(ExperimentLog).order_by(ExperimentLog.recorded_at.desc()).limit(limit))
            .scalars()
            .all()
        )

    if not rows:
        print("（尚無 experiment 紀錄）執行 `python main.py experiment record` 開始追蹤")
        return 0

    print(f"\n{'experiment_id':<26s} {'recorded_at':<20s} {'git':<10s} {'settings_hash':<18s} description")
    print("─" * 100)
    for r in rows:
        ts = r.recorded_at.strftime("%Y-%m-%d %H:%M:%S") if isinstance(r.recorded_at, datetime) else str(r.recorded_at)
        desc = (r.description or "")[:40]
        print(f"{r.experiment_id:<26s} {ts:<20s} {(r.git_commit or '-'):<10s} {r.settings_hash:<18s} {desc}")
    print(f"\n  共 {len(rows)} 筆\n")
    return 0


def _load_one(experiment_id: str) -> ExperimentLog | None:
    with get_session() as session:
        row = session.execute(
            select(ExperimentLog).where(ExperimentLog.experiment_id == experiment_id)
        ).scalar_one_or_none()
        if row is not None:
            # detach 後仍可讀取屬性
            session.expunge(row)
        return row


def _cmd_show(args: argparse.Namespace) -> int:
    eid = args.experiment_id
    row = _load_one(eid)
    if row is None:
        print(f"找不到 experiment: {eid}")
        return 2

    print(f"\n{'═' * 64}")
    print(f"  Experiment: {row.experiment_id}")
    print(f"{'═' * 64}")
    print(f"  recorded_at:    {row.recorded_at}")
    print(f"  git_commit:     {row.git_commit or '(未取得)'}")
    print(f"  settings_hash:  {row.settings_hash}")
    if row.description:
        print(f"  description:    {row.description}")

    print("\n── settings_snapshot ──")
    print(row.settings_snapshot_json)

    print("\n── metrics ──")
    metrics = json.loads(row.metrics_json or "{}")
    if not metrics:
        print("  (無 metrics)")
    else:
        for name, m in metrics.items():
            print(f"  {name}:")
            for k in ("sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "alpha_cum_pct", "snapshot_count"):
                if k in m:
                    print(f"    {k:<22s} {m[k]}")
    print()
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    a = _load_one(args.id_a)
    b = _load_one(args.id_b)
    if a is None:
        print(f"找不到 experiment A: {args.id_a}")
        return 2
    if b is None:
        print(f"找不到 experiment B: {args.id_b}")
        return 2

    print(f"\n{'═' * 64}")
    print(f"  A: {a.experiment_id}  ({a.recorded_at})  git={a.git_commit or '-'}")
    print(f"  B: {b.experiment_id}  ({b.recorded_at})  git={b.git_commit or '-'}")
    print(f"{'═' * 64}")

    if a.settings_hash == b.settings_hash:
        print("  settings_hash 相同 — 設定無變動，差異可能來自時序（資料更新）。")
    else:
        print(f"  settings_hash:  {a.settings_hash}  →  {b.settings_hash}  (CHANGED)")

    metrics_a = json.loads(a.metrics_json or "{}")
    metrics_b = json.loads(b.metrics_json or "{}")
    diffs = diff_metrics(metrics_a, metrics_b)

    if not diffs:
        print("  無 metrics 可比對")
        return 0

    print(f"\n  {'portfolio':<16s} {'metric':<22s} {'A':>14s} {'B':>14s} {'Δ (B-A)':>14s}")
    print("  " + "─" * 86)
    for d in diffs:
        a_str = f"{d['a']:.4f}" if isinstance(d["a"], (int, float)) else str(d["a"] or "-")
        b_str = f"{d['b']:.4f}" if isinstance(d["b"], (int, float)) else str(d["b"] or "-")
        if d["delta"] is None:
            delta_str = "-"
        else:
            sign = "+" if d["delta"] >= 0 else ""
            delta_str = f"{sign}{d['delta']:.4f}"
        print(f"  {d['portfolio']:<16s} {d['metric']:<22s} {a_str:>14s} {b_str:>14s} {delta_str:>14s}")
    print()
    return 0
