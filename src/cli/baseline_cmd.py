"""Baseline regression guard — 對比當前 portfolio 指標與已凍結 baseline 差異。

對應 2026-05-16 P0 任務 3：commit 1803126 加了硬剔除規則但無自動劣化偵測，
下次再改 scanner 邏輯會延後幾週才被 audit 發現。本模組提供：

1. `data/baseline_metrics.json` 凍結 4 active rotation 的 {sharpe, mdd, win_rate, alpha_cum_pct}
2. CLI `validate-baseline --tolerance X` 比對當前 vs baseline；regression 退出碼 ≠ 0
3. CLI `update-baseline` 重寫 baseline（凍結新基準）
4. morning-routine Step 17 守門：失敗不阻擋，但寫 Discord 警告

指標來源：與 `src/cli/export_dashboard_cmd.py:_build_portfolio_review` 共用語意
（用 RotationDailySnapshot 計算 Sharpe / MDD / win_rate，alpha_cum_pct 直接取最新一筆）。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.cli.helpers import safe_print as print

logger = logging.getLogger(__name__)

BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "baseline_metrics.json"

# 各指標的「regression delta 閾值」— 與 tolerance 倍率相乘決定是否觸發警告。
# 設計：
#   - sharpe_ratio / win_rate_pct / alpha_cum_pct：current 比 baseline 「低」超過 delta → regression
#   - max_drawdown_pct：current 比 baseline 「高」超過 delta → regression
DEFAULT_TOLERANCE_DELTAS: dict[str, float] = {
    "sharpe_ratio": 0.20,  # 年化 Sharpe 退化 > 0.20 警告
    "max_drawdown_pct": 2.0,  # MDD 增加 > 2pp 警告
    "win_rate_pct": 5.0,  # 勝率退化 > 5pp 警告
    "alpha_cum_pct": 0.03,  # 累積 alpha 退化 > 3pp 警告
}

# 樣本門檻（與 export_dashboard_cmd 一致）
_MIN_SAMPLES_FOR_SHARPE = 10
_MIN_SAMPLES_FOR_MDD = 3


@dataclass
class PortfolioMetrics:
    """單一 portfolio 的 baseline-trackable 指標。任一指標可為 None（資料不足）。"""

    portfolio_name: str
    as_of: str  # 'YYYY-MM-DD'
    snapshot_count: int
    sharpe_ratio: float | None
    max_drawdown_pct: float | None
    win_rate_pct: float | None
    alpha_cum_pct: float | None


@dataclass
class RegressionFinding:
    """單一 metric 的 regression 結果。"""

    portfolio_name: str
    metric: str
    baseline_value: float | None
    current_value: float | None
    delta: float | None  # current - baseline（None = 無法比對）
    threshold: float  # tolerance × DEFAULT_TOLERANCE_DELTAS[metric]
    is_regression: bool
    reason: str  # 人話說明


# ---------------------------------------------------------------------------
# Pure 計算（無 DB / IO 副作用）
# ---------------------------------------------------------------------------


def compute_metrics_from_snapshots(
    snapshots: list[dict],
    portfolio_name: str,
    as_of: str,
) -> PortfolioMetrics:
    """從 snapshot dict list（已 asc by date）計算 baseline-trackable 指標。

    Snapshot dict 須含：total_capital, daily_return_pct, alpha_cum_pct（皆可能為 None）

    與 `export_dashboard_cmd._build_portfolio_review` 共用語意：
      - Sharpe 年化 = mean(daily_returns) / std(daily_returns) × √252，N≥10 才算
      - MDD = (peak - cap) / peak × 100，N≥3 才算
      - win_rate = positive_days / total_days × 100，N≥3 才算
      - alpha_cum_pct = 最新一筆 snapshot 的 alpha_cum_pct
    """
    snapshot_count = len(snapshots)

    sharpe: float | None = None
    mdd: float | None = None
    win_rate: float | None = None
    alpha: float | None = None

    if snapshot_count == 0:
        return PortfolioMetrics(
            portfolio_name=portfolio_name,
            as_of=as_of,
            snapshot_count=0,
            sharpe_ratio=None,
            max_drawdown_pct=None,
            win_rate_pct=None,
            alpha_cum_pct=None,
        )

    # Sharpe
    daily_returns = [s["daily_return_pct"] for s in snapshots if s.get("daily_return_pct") is not None]
    if len(daily_returns) >= _MIN_SAMPLES_FOR_SHARPE:
        mean = statistics.fmean(daily_returns)
        try:
            stdev = statistics.stdev(daily_returns)
        except statistics.StatisticsError:
            stdev = 0.0
        if stdev > 1e-8:
            sharpe = round(mean / stdev * math.sqrt(252), 4)

    # MDD + win_rate（N≥3）
    if snapshot_count >= _MIN_SAMPLES_FOR_MDD:
        peak = snapshots[0]["total_capital"]
        max_dd = 0.0
        for s in snapshots:
            cap = s["total_capital"]
            if cap > peak:
                peak = cap
            if peak > 0:
                dd = (peak - cap) / peak
                if dd > max_dd:
                    max_dd = dd
        mdd = round(max_dd * 100, 2)

        if daily_returns:
            wins = sum(1 for r in daily_returns if r > 0)
            win_rate = round(wins / len(daily_returns) * 100, 2)

    # alpha — 最新一筆（asc 排序 → 最後一筆）
    last = snapshots[-1]
    if last.get("alpha_cum_pct") is not None:
        alpha = round(float(last["alpha_cum_pct"]), 6)

    return PortfolioMetrics(
        portfolio_name=portfolio_name,
        as_of=as_of,
        snapshot_count=snapshot_count,
        sharpe_ratio=sharpe,
        max_drawdown_pct=mdd,
        win_rate_pct=win_rate,
        alpha_cum_pct=alpha,
    )


def compare_metrics(
    baseline: PortfolioMetrics,
    current: PortfolioMetrics,
    tolerance: float = 1.0,
    *,
    deltas: dict[str, float] | None = None,
) -> list[RegressionFinding]:
    """逐 metric 比對，回傳所有 regression（含未 regression 的也列出方便 audit）。

    tolerance=1.0 用預設 deltas；0.5 = 嚴格半量；2.0 = 寬鬆雙倍。
    任一值為 None → 跳過該 metric 且不視為 regression（不足以判斷）。
    """
    deltas = deltas or DEFAULT_TOLERANCE_DELTAS
    findings: list[RegressionFinding] = []

    # higher-is-better metrics
    for metric in ("sharpe_ratio", "win_rate_pct", "alpha_cum_pct"):
        bv = getattr(baseline, metric)
        cv = getattr(current, metric)
        threshold = deltas[metric] * tolerance
        if bv is None or cv is None:
            findings.append(
                RegressionFinding(
                    portfolio_name=current.portfolio_name,
                    metric=metric,
                    baseline_value=bv,
                    current_value=cv,
                    delta=None,
                    threshold=threshold,
                    is_regression=False,
                    reason="資料不足（任一值為 None）",
                )
            )
            continue
        delta = cv - bv
        is_reg = delta < -threshold
        findings.append(
            RegressionFinding(
                portfolio_name=current.portfolio_name,
                metric=metric,
                baseline_value=bv,
                current_value=cv,
                delta=round(delta, 6),
                threshold=threshold,
                is_regression=is_reg,
                reason=f"current({cv:+.4f}) - baseline({bv:+.4f}) = {delta:+.4f}，閾值={-threshold:+.4f}",
            )
        )

    # lower-is-better metrics
    for metric in ("max_drawdown_pct",):
        bv = getattr(baseline, metric)
        cv = getattr(current, metric)
        threshold = deltas[metric] * tolerance
        if bv is None or cv is None:
            findings.append(
                RegressionFinding(
                    portfolio_name=current.portfolio_name,
                    metric=metric,
                    baseline_value=bv,
                    current_value=cv,
                    delta=None,
                    threshold=threshold,
                    is_regression=False,
                    reason="資料不足（任一值為 None）",
                )
            )
            continue
        delta = cv - bv
        is_reg = delta > threshold
        findings.append(
            RegressionFinding(
                portfolio_name=current.portfolio_name,
                metric=metric,
                baseline_value=bv,
                current_value=cv,
                delta=round(delta, 6),
                threshold=threshold,
                is_regression=is_reg,
                reason=f"current({cv:.2f}%) - baseline({bv:.2f}%) = {delta:+.2f}pp，閾值=+{threshold:.2f}pp",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# 檔案 I/O
# ---------------------------------------------------------------------------


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, PortfolioMetrics]:
    """讀 baseline JSON 檔。檔不存在或 portfolios 為空時回 {}。"""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, PortfolioMetrics] = {}
    for name, d in (raw.get("portfolios") or {}).items():
        out[name] = PortfolioMetrics(
            portfolio_name=name,
            as_of=d.get("as_of", ""),
            snapshot_count=d.get("snapshot_count", 0),
            sharpe_ratio=d.get("sharpe_ratio"),
            max_drawdown_pct=d.get("max_drawdown_pct"),
            win_rate_pct=d.get("win_rate_pct"),
            alpha_cum_pct=d.get("alpha_cum_pct"),
        )
    return out


def save_baseline(
    metrics: dict[str, PortfolioMetrics],
    path: Path = BASELINE_PATH,
    *,
    git_commit: str | None = None,
) -> None:
    """寫 baseline JSON（含 metadata.created_at / git_commit）。"""
    payload = {
        "metadata": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": git_commit or _try_git_head(),
            "tolerance_deltas": DEFAULT_TOLERANCE_DELTAS,
        },
        "portfolios": {name: asdict(m) for name, m in metrics.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _try_git_head() -> str | None:
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


# ---------------------------------------------------------------------------
# DB 撈當前指標
# ---------------------------------------------------------------------------


def collect_current_metrics(
    portfolio_names: list[str] | None = None,
    as_of: str | None = None,
    *,
    lookback_days: int = 90,
) -> dict[str, PortfolioMetrics]:
    """逐 portfolio 撈 snapshot → 計算 metrics。

    portfolio_names=None → 撈所有 active rotation；as_of=None → 用今日。
    """
    from datetime import date

    from src.portfolio.manager import RotationManager

    if portfolio_names is None:
        portfolios = RotationManager.list_portfolios()
        portfolio_names = [p["name"] for p in portfolios if p["status"] == "active"]

    if as_of is None:
        as_of = date.today().isoformat()

    out: dict[str, PortfolioMetrics] = {}
    for name in portfolio_names:
        mgr = RotationManager(name)
        snaps = mgr.get_recent_snapshots(n_days=lookback_days)
        out[name] = compute_metrics_from_snapshots(snaps, portfolio_name=name, as_of=as_of)
    return out


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------


def cmd_validate_baseline(args: argparse.Namespace) -> int:
    """比對當前指標與 baseline；regression 退出碼 1，全綠 0。

    Returns
    -------
    int
        exit code（0=通過 / 1=有 regression / 2=baseline 缺失）
    """
    from src.cli.helpers import init_db

    init_db()
    tolerance = float(getattr(args, "tolerance", 1.0))
    lookback_days = int(getattr(args, "lookback_days", 90))
    quiet = bool(getattr(args, "quiet", False))

    baseline = load_baseline()
    if not baseline:
        print(f"[baseline] 找不到 baseline 檔（預期路徑：{BASELINE_PATH}）")
        print("  請先執行：python main.py update-baseline --confirm")
        return 2

    current = collect_current_metrics(
        portfolio_names=list(baseline.keys()),
        lookback_days=lookback_days,
    )

    all_findings: list[RegressionFinding] = []
    for name, bm in baseline.items():
        cm = current.get(name)
        if cm is None:
            print(f"[{name}] 找不到當前 portfolio（可能已 paused/delete）— skip")
            continue
        findings = compare_metrics(bm, cm, tolerance=tolerance)
        all_findings.extend(findings)

    regressions = [f for f in all_findings if f.is_regression]

    if not quiet:
        _print_baseline_report(baseline, current, all_findings, tolerance=tolerance)

    return 1 if regressions else 0


def cmd_update_baseline(args: argparse.Namespace) -> int:
    """以當前指標重寫 baseline 檔。需 --confirm 旗標防誤操作。"""
    from src.cli.helpers import init_db

    init_db()
    if not getattr(args, "confirm", False):
        print("[baseline] 拒絕執行：請加 --confirm 旗標確認要覆寫 baseline")
        print(f"  目標路徑：{BASELINE_PATH}")
        return 2

    portfolio_names = getattr(args, "portfolios", None)
    lookback_days = int(getattr(args, "lookback_days", 90))

    metrics = collect_current_metrics(portfolio_names=portfolio_names, lookback_days=lookback_days)
    if not metrics:
        print("[baseline] 找不到任何 active portfolio，未寫入")
        return 2

    save_baseline(metrics)
    print(f"[baseline] 已寫入 {len(metrics)} portfolio 至 {BASELINE_PATH}")
    for name, m in metrics.items():
        print(
            f"  {name}: sharpe={m.sharpe_ratio} mdd={m.max_drawdown_pct} "
            f"win={m.win_rate_pct} alpha_cum={m.alpha_cum_pct}"
        )
    return 0


# ---------------------------------------------------------------------------
# 顯示輔助
# ---------------------------------------------------------------------------


def _print_baseline_report(
    baseline: dict[str, PortfolioMetrics],
    current: dict[str, PortfolioMetrics],
    findings: list[RegressionFinding],
    tolerance: float,
) -> None:
    regressions = [f for f in findings if f.is_regression]
    print(f"\n{'═' * 64}")
    print(f"  Baseline Regression Check  (tolerance={tolerance})")
    print(f"{'═' * 64}")

    for name, bm in baseline.items():
        cm = current.get(name)
        if cm is None:
            continue
        print(
            f"\n[{name}]  baseline as_of={bm.as_of} → current as_of={cm.as_of}"
            f"  (snapshots: {bm.snapshot_count} → {cm.snapshot_count})"
        )
        for f in findings:
            if f.portfolio_name != name:
                continue
            tag = "🔴" if f.is_regression else "  "
            print(f"  {tag} {f.metric:<20s} {f.reason}")

    print(f"\n{'─' * 64}")
    if regressions:
        print(f"  ⚠ 共偵測到 {len(regressions)} 項 regression（exit code = 1）")
    else:
        print("  ✅ 全部 metric 通過 baseline 檢查")
    print(f"{'═' * 64}\n")


def format_regressions_for_discord(regressions: list[RegressionFinding]) -> str:
    """將 regression list 格式化為 Discord summary 區塊（≤ ~300 字元）。"""
    if not regressions:
        return ""
    lines = [f"\n🔴 **Baseline Regression** ({len(regressions)} 項)"]
    # 每個 portfolio 取 max 3 項顯示，避免爆字元
    grouped: dict[str, list[RegressionFinding]] = {}
    for r in regressions:
        grouped.setdefault(r.portfolio_name, []).append(r)
    for name, items in grouped.items():
        lines.append(f"  {name}:")
        for r in items[:3]:
            lines.append(f"    {r.metric} {r.reason}")
    return "\n".join(lines) + "\n"
