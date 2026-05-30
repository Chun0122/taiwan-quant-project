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
    return {"total_capital": row[0], "alpha_cum_pct": row[1], "benchmark_cum_return_pct": row[2]}


def _load_snapshot_on_or_before(session, name: str, target: date) -> dict | None:
    row = session.execute(
        select(
            RotationDailySnapshot.total_capital,
            RotationDailySnapshot.alpha_cum_pct,
            RotationDailySnapshot.benchmark_cum_return_pct,
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
    return {"total_capital": row[0], "alpha_cum_pct": row[1], "benchmark_cum_return_pct": row[2]}


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


def _load_benchmark_return(session, start: date, end: date) -> float | None:
    """0050 期間報酬（%）：(close_end - close_start) / close_start × 100。

    用 on-or-after(start) 與 on-or-before(end) 對齊非交易日。
    """
    start_row = session.execute(
        select(DailyPrice.close)
        .where(DailyPrice.stock_id == _BENCHMARK, DailyPrice.date >= start)
        .order_by(DailyPrice.date.asc())
        .limit(1)
    ).first()
    end_row = session.execute(
        select(DailyPrice.close)
        .where(DailyPrice.stock_id == _BENCHMARK, DailyPrice.date <= end)
        .order_by(DailyPrice.date.desc())
        .limit(1)
    ).first()
    if start_row is None or end_row is None or not start_row[0]:
        return None
    return round((end_row[0] - start_row[0]) / start_row[0] * 100, 2)


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
        bm_deltas: list[float] = []
        for _pid, name, status in portfolios:
            if status != "active":
                continue
            s_start = _load_snapshot_on_or_after(session, name, period_b[0])
            s_end = _load_snapshot_on_or_before(session, name, period_b[1])
            ad = compute_alpha_delta(name, s_start, s_end)
            if ad.benchmark_delta_pp is not None:
                bm_deltas.append(ad.benchmark_delta_pp)

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

        # ── 2b. Benchmark 一致性 cross-check（自動抓 0050 資料異常）──
        raw_bm = _load_benchmark_return(session, period_b[0], period_b[1])
        snap_bm = round(sum(bm_deltas) / len(bm_deltas), 2) if bm_deltas else None
        lines.append("**Benchmark cross-check（0050）**")
        lines.append("")
        lines.append(f"- raw 0050 收盤對收盤（期內首末交易日）：{raw_bm if raw_bm is not None else 'N/A'}%")
        lines.append(f"- snapshot 凍結 bm 增量（各 portfolio 平均）：{snap_bm if snap_bm is not None else 'N/A'}pp")
        if raw_bm is not None and snap_bm is not None and abs(raw_bm - snap_bm) > 2.0:
            lines.append(
                f"- ⚠ **兩者差 {abs(raw_bm - snap_bm):.1f}pp（> 2pp）**：可能 0050 期末收盤為可疑跳點 / 補抓延遲，"
                "或 snapshot benchmark 錨點與期初不同。alpha 增量以 snapshot 為準（內部一致）；raw 僅供對照，"
                "建議人工檢查 0050 期末收盤是否異常。"
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
