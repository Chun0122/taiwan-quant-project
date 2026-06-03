"""rotation-audit CLI — 可重複的修復前後 / 期間對比審計報告。

把 logs/audit_20260529/REPORT.md 的手工 SQL/Python 分析做成單一指令，
兌現「6/15 / 6/30 重審」承諾，重審成本趨近零。

用法：
  python main.py rotation-audit --period-a 2026-04-29:2026-05-08 \
                                --period-b 2026-05-09:2026-05-29
  python main.py rotation-audit --period-b 2026-05-09:2026-05-29   # 只看單期
  python main.py rotation-audit --period-b 2026-05-09:2026-05-29 --out logs/audit_20260615/REPORT.md

純讀 DB（rotation_position / rotation_daily_snapshot / discovery_record）+ 寫 markdown。
不修改任何 src/，不 commit。
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from sqlalchemy import select

from src.cli.helpers import init_db
from src.cli.helpers import safe_print as print
from src.data.database import get_session
from src.data.schema import DailyPrice, DiscoveryRecord, RotationDailySnapshot, RotationPortfolio, RotationPosition
from src.portfolio.audit import (
    compute_alpha_delta,
    compute_jaccard_stability,
    compute_trade_stats,
)

logger = logging.getLogger(__name__)

_BENCHMARK = "0050"


def _parse_period(arg: str | None) -> tuple[date, date] | None:
    """'YYYY-MM-DD:YYYY-MM-DD' → (start, end)；None 或格式錯誤回 None。"""
    if not arg:
        return None
    try:
        start_s, end_s = arg.split(":")
        return date.fromisoformat(start_s.strip()), date.fromisoformat(end_s.strip())
    except (ValueError, AttributeError):
        return None


def _load_closed_trades(session, portfolio_id: int, start: date, end: date) -> list[dict]:
    rows = session.execute(
        select(
            RotationPosition.return_pct,
            RotationPosition.pnl,
            RotationPosition.exit_reason,
        ).where(
            RotationPosition.portfolio_id == portfolio_id,
            RotationPosition.status == "closed",
            RotationPosition.entry_date >= start,
            RotationPosition.entry_date <= end,
        )
    ).all()
    return [{"return_pct": r[0], "pnl": r[1], "exit_reason": r[2]} for r in rows]


def _load_snapshot_on_or_after(session, name: str, target: date) -> dict | None:
    row = session.execute(
        select(
            RotationDailySnapshot.total_capital,
            RotationDailySnapshot.alpha_cum_pct,
            RotationDailySnapshot.benchmark_cum_return_pct,
            RotationDailySnapshot.snapshot_date,
        )
        .where(
            RotationDailySnapshot.portfolio_name == name,
            RotationDailySnapshot.snapshot_date >= target,
        )
        .order_by(RotationDailySnapshot.snapshot_date.asc())
        .limit(1)
    ).first()
    if row is None:
        return None
    return {
        "total_capital": row[0],
        "alpha_cum_pct": row[1],
        "benchmark_cum_return_pct": row[2],
        "snapshot_date": row[3],
    }


def _load_snapshot_on_or_before(session, name: str, target: date) -> dict | None:
    row = session.execute(
        select(
            RotationDailySnapshot.total_capital,
            RotationDailySnapshot.alpha_cum_pct,
            RotationDailySnapshot.benchmark_cum_return_pct,
            RotationDailySnapshot.snapshot_date,
        )
        .where(
            RotationDailySnapshot.portfolio_name == name,
            RotationDailySnapshot.snapshot_date <= target,
        )
        .order_by(RotationDailySnapshot.snapshot_date.desc())
        .limit(1)
    ).first()
    if row is None:
        return None
    return {
        "total_capital": row[0],
        "alpha_cum_pct": row[1],
        "benchmark_cum_return_pct": row[2],
        "snapshot_date": row[3],
    }


def _load_momentum_topn_sets(session, start: date, end: date, top_n: int, mode: str) -> list[tuple[str, set[str]]]:
    rows = session.execute(
        select(DiscoveryRecord.scan_date, DiscoveryRecord.stock_id)
        .where(
            DiscoveryRecord.mode == mode,
            DiscoveryRecord.scan_date >= start,
            DiscoveryRecord.scan_date <= end,
            DiscoveryRecord.rank <= top_n,
        )
        .order_by(DiscoveryRecord.scan_date, DiscoveryRecord.rank)
    ).all()
    by_date: dict[str, set[str]] = {}
    for scan_date, sid in rows:
        by_date.setdefault(scan_date.isoformat(), set()).add(sid)
    return sorted(by_date.items())


def _benchmark_close(session, target: date, strictly_before: bool = False) -> float | None:
    """0050 在 target 當日（含）或之前最近一筆收盤。

    strictly_before=True 取 date < target（重現 snapshot 寫入當下「當日收盤尚未入庫」
    的早上 lag 語意）；False 取 date <= target（含當日）。
    """
    cond = DailyPrice.date < target if strictly_before else DailyPrice.date <= target
    row = session.execute(
        select(DailyPrice.close)
        .where(DailyPrice.stock_id == _BENCHMARK, cond)
        .order_by(DailyPrice.date.desc())
        .limit(1)
    ).first()
    return float(row[0]) if row and row[0] else None


def _aligned_bm_divergence(session, snap_start: dict | None, snap_end: dict | None, snap_delta_pp: float | None):
    """snapshot 凍結 bm 增量 vs daily_price 重算的最小偏差（pp），已做 lag 端點對齊。

    snapshot 在當日收盤入庫前寫入，凍結的是「前一交易日」0050 收盤；audit 時當日收盤
    已存在，直接用 on-or-before 會比錯一天。故對 start/end 各取 {含當日, 嚴格前一日}
    兩個候選，挑出與 snapshot 凍結增量最接近的組合 —— 容忍 0 或 1 個交易日的 lag，
    只有「對齊後仍偏離」才代表 daily_price 參照日真被竄改。

    回傳最小偏差（pp，四捨五入 2 位）；資料不足回傳 None。
    """
    if snap_start is None or snap_end is None or snap_delta_pp is None:
        return None
    d_start = snap_start.get("snapshot_date")
    d_end = snap_end.get("snapshot_date")
    if d_start is None or d_end is None:
        return None
    starts = [c for c in (_benchmark_close(session, d_start, False), _benchmark_close(session, d_start, True)) if c]
    ends = [c for c in (_benchmark_close(session, d_end, False), _benchmark_close(session, d_end, True)) if c]
    if not starts or not ends:
        return None
    # ratio 正規化（÷ start close）與 snapshot 的 base 正規化差異 < 0.1pp，遠小於 2pp 門檻
    best = min(abs((ce / cs - 1) * 100 - snap_delta_pp) for cs in starts for ce in ends if cs > 0)
    return round(best, 2)


def _trade_stats_row(name: str, label: str, stats) -> str:
    def f(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "—"

    return (
        f"| {name} | {label} | {stats.n_closed} | {f(stats.win_pct, '%')} | "
        f"{f(stats.avg_return_pct, '%')} | {stats.total_pnl:,.0f} | "
        f"{stats.stop_loss_count} | {f(stats.stop_loss_pct, '%')} |"
    )


def cmd_rotation_audit(args: argparse.Namespace) -> int:
    """rotation-audit handler。回傳 exit code（0=成功 / 2=參數錯誤）。"""
    init_db()

    period_a = _parse_period(getattr(args, "period_a", None))
    period_b = _parse_period(getattr(args, "period_b", None))
    top_n = int(getattr(args, "top", 5))
    jaccard_mode = getattr(args, "jaccard_mode", "momentum")

    if period_b is None:
        print("錯誤：--period-b 為必填，格式 YYYY-MM-DD:YYYY-MM-DD")
        return 2

    lines: list[str] = []
    lines.append("# Rotation Audit Report")
    lines.append("")
    lines.append(f"- **產生時間**：{date.today().isoformat()}")
    if period_a:
        lines.append(f"- **A 期（對照）**：{period_a[0]} ~ {period_a[1]}")
    lines.append(f"- **B 期（主要）**：{period_b[0]} ~ {period_b[1]}")
    lines.append("- **資料來源**：`data/stock.db`（rotation_position / rotation_daily_snapshot / discovery_record）")
    lines.append("")

    with get_session() as session:
        portfolios = session.execute(
            select(RotationPortfolio.id, RotationPortfolio.name, RotationPortfolio.status).order_by(
                RotationPortfolio.id
            )
        ).all()
        if not portfolios:
            lines.append("⚠ 無 rotation_portfolio 紀錄，無法產生報告。")
            _write_or_print(args, lines)
            return 0

        # ── 1. closed trade 對比表 ──
        lines.append("## 1. Closed Trade 統計（依 entry_date 篩選）")
        lines.append("")
        lines.append("| portfolio | 期間 | N | win% | avg ret% | total pnl | SL | SL% |")
        lines.append("|-----------|------|--:|-----:|---------:|----------:|---:|----:|")
        total_b_closed = 0
        for pid, name, _status in portfolios:
            if period_a:
                a_trades = _load_closed_trades(session, pid, period_a[0], period_a[1])
                if a_trades:
                    lines.append(_trade_stats_row(name, "A", compute_trade_stats(a_trades)))
            b_trades = _load_closed_trades(session, pid, period_b[0], period_b[1])
            total_b_closed += len(b_trades)
            if b_trades or not period_a:
                lines.append(_trade_stats_row(name, "B", compute_trade_stats(b_trades)))
        lines.append("")
        if total_b_closed < 10:
            lines.append(
                f"> ⚠ **樣本不足警告**：B 期全 portfolio 合計僅 {total_b_closed} 筆 closed trade（< 10），"
                "win_rate / avg_return 易受單筆變異支配，不宜定量結論。"
            )
            lines.append("")

        # ── 2. Benchmark alpha 分解 ──
        lines.append("## 2. Benchmark Alpha 分解（B 期，snapshot-based）")
        lines.append("")
        lines.append(
            "| portfolio | cap 期初 | cap 期末 | port Δ% | alpha 期初 | alpha 期末 | **alpha 增量 (pp)** | bm 增量 (pp) |"
        )
        lines.append(
            "|-----------|---------:|---------:|--------:|-----------:|-----------:|-------------------:|------------:|"
        )
        cross_rows: list[tuple[str, dict | None, dict | None, float | None]] = []
        for _pid, name, status in portfolios:
            if status != "active":
                continue
            s_start = _load_snapshot_on_or_after(session, name, period_b[0])
            s_end = _load_snapshot_on_or_before(session, name, period_b[1])
            ad = compute_alpha_delta(name, s_start, s_end)
            cross_rows.append((name, s_start, s_end, ad.benchmark_delta_pp))

            def fmt(v, suffix=""):
                return f"{v}{suffix}" if v is not None else "—"

            flag = ""
            if ad.alpha_delta_pp is not None:
                flag = " 🟢" if ad.alpha_delta_pp > 0 else " 🔴"
            lines.append(
                f"| {name} | {fmt(round(ad.cap_start) if ad.cap_start else None)} | "
                f"{fmt(round(ad.cap_end) if ad.cap_end else None)} | {fmt(ad.portfolio_return_pct, '%')} | "
                f"{fmt(ad.alpha_start_pct, '%')} | {fmt(ad.alpha_end_pct, '%')} | "
                f"{fmt(ad.alpha_delta_pp, 'pp')}{flag} | {fmt(ad.benchmark_delta_pp, 'pp')} |"
            )
        lines.append("")
        lines.append("> alpha 增量 = alpha(期末) − alpha(期初)；恆等式 `port Δ% ≈ alpha 增量 + bm 增量`。")
        lines.append("> bm 增量 = snapshot 內凍結之 0050 累積報酬變化（與 alpha 同源，內部一致）。")
        lines.append("")

        # ── 2b. Benchmark 一致性 cross-check（lag 端點對齊，逐 portfolio）──
        lines.append("**Benchmark cross-check（0050，lag 端點對齊）**")
        lines.append("")
        divergences: list[tuple[str, float]] = []
        for name, s_start, s_end, snap_d in cross_rows:
            div = _aligned_bm_divergence(session, s_start, s_end, snap_d)
            if div is None:
                continue
            divergences.append((name, div))
            lines.append(f"- {name}：snapshot bm 增量 {snap_d}pp，對齊後 raw 重算最小偏差 {div}pp")
        if not divergences:
            lines.append("- N/A（無足夠 snapshot / benchmark 資料對齊）")
        else:
            worst_name, worst = max(divergences, key=lambda x: x[1])
            lines.append("")
            if worst > 2.0:
                lines.append(
                    f"- ⚠ **{worst_name} 對齊後仍偏 {worst:.1f}pp（> 2pp）**：已排除一交易日 lag，"
                    "snapshot 凍結之 0050 收盤與當前 daily_price 在參照日仍不符 → "
                    "可能 daily_price 被事後竄改 / 補抓延遲，建議人工檢查該 portfolio 期末參照日收盤。"
                )
            else:
                lines.append(
                    f"- ✅ 各 portfolio 對齊後偏差 ≤ 2pp（最大 {worst}pp）：raw 與 snapshot 一致，"
                    "原始差異為一交易日 snapshot lag（by-design），非資料異常。"
                )
        lines.append("")

        # ── 3. 訊號穩定性 ──
        lines.append(f"## 3. 訊號穩定性（{jaccard_mode} top-{top_n} 相鄰日 Jaccard）")
        lines.append("")
        daily_sets = _load_momentum_topn_sets(session, period_b[0], period_b[1], top_n, jaccard_mode)
        jac = compute_jaccard_stability(daily_sets)
        if not jac.pairs:
            lines.append(f"⚠ {jaccard_mode} 模式 B 期掃描日 < 2，無法計算 Jaccard。")
        else:
            lines.append(f"- 掃描日數：{len(daily_sets)}　|　日對數：{len(jac.pairs)}")
            lines.append(
                f"- **平均 Jaccard {jac.mean_jaccard}** / 中位數 {jac.median_jaccard} "
                f"/ max {jac.max_jaccard} / min {jac.min_jaccard}"
            )
            lines.append("")
            lines.append("| day1 | day2 | overlap | jaccard |")
            lines.append("|------|------|--------:|--------:|")
            for p in jac.pairs:
                lines.append(f"| {p['day1']} | {p['day2']} | {p['overlap']}/{p['union']} | {p['jaccard']} |")
        lines.append("")

    _write_or_print(args, lines)
    return 0


def _write_or_print(args: argparse.Namespace, lines: list[str]) -> None:
    out = getattr(args, "out", None)
    content = "\n".join(lines) + "\n"
    if out:
        from pathlib import Path

        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print(f"報告已寫入：{out}（{len(lines)} 行）")
    else:
        # 直接 stdout（UTF-8 安全）
        print(content)
