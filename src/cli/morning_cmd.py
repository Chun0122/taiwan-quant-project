"""CLI 早晨例行子命令 — morning-routine + Discord 摘要 + 策略衰減。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.cli.anomaly_cmd import _compute_anomaly_scan, _compute_macro_stress_check, cmd_anomaly_scan, cmd_revenue_scan
from src.cli.discover_cmd import _cmd_discover_all
from src.cli.helpers import init_db
from src.cli.helpers import safe_print as print
from src.cli.rotation_cmd import _rotation_update_all
from src.cli.sync import (
    cmd_alert_check,
    cmd_compute,
    cmd_sync,
    cmd_sync_broker,
    cmd_sync_features,
    cmd_sync_info,
    cmd_sync_mops,
    cmd_sync_revenue,
    cmd_sync_sbl,
)
from src.cli.watch_cmd import _watch_update_status
from src.constants import (
    DEFAULT_HHI_THRESHOLD,
    DEFAULT_INST_THRESHOLD,
    DEFAULT_SBL_SIGMA,
    DEFAULT_VOL_MULT,
    DISCOVERY_IC_HOLDING_DAYS_MAP,
    DISCOVERY_KEY_FACTOR_MAP,
)

logger = logging.getLogger(__name__)


def _build_morning_discord_summary(
    today_str: str,
    top_n: int,
    freshness: dict | None = None,
    stress_result: dict | None = None,
    ic_status: list[dict] | None = None,
    disabled_modes: list[str] | None = None,
    discover_blocked: bool = False,
    baseline_regressions: list | None = None,
) -> str:
    """建立早晨例行報告的 Discord 訊息摘要。

    查詢今日 DiscoveryRecord、近3日 Announcement、以及 WatchEntry 狀態，
    組合成一則 Discord 推播訊息（≤ 1900 字元）。

    Args:
        today_str: 今日日期字串（YYYY-MM-DD）
        top_n: 顯示筆數
        freshness: _verify_data_freshness() 回傳的新鮮度 dict；為 None 時不顯示 banner
        stress_result: Step 0 已計算的 stress check 結果（M4：單一真相來源，
            避免 Discord 區塊重新計算造成 regime 與 rotation 不一致）
        ic_status: Step 8c IC 健康度檢查結果（M3：透明顯示因子有效性）
        disabled_modes: IC 反向被自動停用的 discover 模式列表（M2）
        discover_blocked: True 時表示 Step 9 因資料過期被阻擋（M1）
    """
    import datetime
    from collections import defaultdict

    from sqlalchemy import and_, func, select

    from src.data.database import get_session
    from src.data.schema import Announcement, DiscoveryRecord, WatchEntry

    lines: list[str] = [f"🌅 **早晨例行報告** ({today_str})", ""]
    # ── 資料新鮮度警示（置頂顯示）──────────────────────────
    if freshness and freshness.get("is_stale"):
        gap = freshness.get("gap_days", 0) or 0
        banner = "⚠️ **資料新鮮度警告**"
        if discover_blocked:
            banner += f" — Discover 已阻擋（資料過期 {gap} 天 > 硬阻擋門檻 7 天）"
        else:
            banner += " — 以下推薦可能使用過期數據"
        lines.append(banner)
        lines.append(f"  {freshness.get('message', '')}")
        lines.append("")
    today = datetime.date.fromisoformat(today_str)

    # ── Step 0: 宏觀壓力預檢警示（M4：僅使用傳入的 stress_result，不重新計算）──────
    if stress_result is not None:
        if stress_result.get("breadth_downgraded"):
            bpct = stress_result.get("breadth_below_ma20_pct", 0.0) or 0.0
            lines.append(f"📊 **市場廣度警示**：{bpct:.0%} 股票跌破 MA20 → regime 降級")
            lines.append("")
        if stress_result.get("crisis_triggered"):
            lines.append("🚨 **CRISIS 崩盤警示已啟動**")
            lines.append(f"  {stress_result.get('summary', '')}")
            lines.append("")
        elif stress_result.get("regime") == "bear":
            lines.append(f"⚠️ **空頭市場** {stress_result.get('summary', '')}")
            lines.append("")
        elif stress_result.get("regime") is None:
            lines.append("⚠️ **Regime 未知** — Step 0 宏觀壓力預檢失敗，Rotation 以保守模式運行")
            lines.append("")
        # P0 止血包 #5：crisis 訊號可用性自檢（缺口每日可見）
        avail_msg = _format_crisis_availability(stress_result)
        if avail_msg:
            lines.append(f"🩺 {avail_msg}")
            lines.append("")

    # ── Baseline Regression 守門（Step 17）────────────────────
    if baseline_regressions:
        from src.cli.baseline_cmd import format_regressions_for_discord

        block = format_regressions_for_discord(baseline_regressions)
        if block:
            lines.append(block.rstrip("\n"))
            lines.append("")

    # ── IC 健康度狀態（M3：顯示被降權/停用的子因子）────────────────
    if ic_status:
        failed = [s for s in ic_status if s.get("level") == "error"]
        inverse = [s for s in ic_status if s.get("level") == "inverse"]
        weak = [s for s in ic_status if s.get("level") in ("weak", "decay")]
        if inverse or failed or disabled_modes:
            lines.append("🧪 **IC 健康度**")
            for s in inverse:
                lines.append(f"  🔴 {s['mode']} {s['factor']} IC={s['ic']:+.4f}（反向→已暫停）")
            for s in weak:
                lines.append(f"  ⚠ {s['mode']} {s['factor']} IC={s['ic']:+.4f}（{s['level']}）")
            for s in failed:
                lines.append(f"  ⚠ {s['mode']} IC 計算失敗: {s.get('error', 'unknown')}")
            lines.append("")

    # ── 1. 多模式選股（今日 DiscoveryRecord，出現 2+ 模式）──────────────
    with get_session() as session:
        disc_rows = session.execute(
            select(
                DiscoveryRecord.stock_id,
                DiscoveryRecord.stock_name,
                DiscoveryRecord.mode,
                DiscoveryRecord.rank,
            ).where(DiscoveryRecord.scan_date == today)
        ).all()

    if disc_rows:
        mode_labels = {"momentum": "動", "swing": "波", "value": "值", "dividend": "息", "growth": "長"}
        stock_modes: dict = defaultdict(list)
        stock_names: dict = {}
        for r in disc_rows:
            stock_modes[r.stock_id].append((r.mode, r.rank))
            stock_names[r.stock_id] = r.stock_name or r.stock_id

        multi = {sid: modes for sid, modes in stock_modes.items() if len(modes) >= 2}
        multi_sorted = sorted(multi.items(), key=lambda x: -len(x[1]))

        if multi_sorted:
            lines.append(f"📊 **多模式選股** (出現 2+ 模式，共 {len(multi)} 支)")
            for sid, modes in multi_sorted[:5]:
                name = str(stock_names.get(sid) or "")[:6]
                mode_str = " ".join(f"{mode_labels.get(m, '?')}#{r}" for m, r in sorted(modes, key=lambda x: x[1]))
                lines.append(f"  {'★' * len(modes)} {sid} {name} ({mode_str})")
            if len(multi) > 5:
                lines.append(f"  …共 {len(multi)} 支")
            lines.append("")
        else:
            lines.append(
                f"📊 **多模式選股**：今日無出現 2+ 模式的股票（共掃描 {len(set(r.stock_id for r in disc_rows))} 支）"
            )
            lines.append("")
    else:
        lines.append("📊 **多模式選股**：今日無掃描記錄（請確認 discover all 是否已執行）")
        lines.append("")

    # ── 2. 重大事件（近3日，非 general）────────────────────────────────
    since = today - datetime.timedelta(days=3)
    alert_rows = []
    try:
        with get_session() as session:
            alert_rows = session.execute(
                select(
                    Announcement.date,
                    Announcement.stock_id,
                    Announcement.event_type,
                    Announcement.subject,
                )
                # 注意：SQL 三值邏輯下 `!= "general"` 對 NULL 回傳 UNKNOWN（視為 False）
                # 若要保留 NULL 事件為「可能重要」需加 IS NULL；此處採嚴格過濾
                .where(
                    and_(
                        Announcement.date >= since,
                        Announcement.event_type.isnot(None),
                        Announcement.event_type != "general",
                    )
                )
                .order_by(Announcement.date.desc())
                .limit(10)
            ).all()
    except Exception:
        # event_type 欄位可能尚未 migrate，跳過重大事件區塊
        logger.debug("Discord 摘要：重大事件區塊失敗", exc_info=True)

    _EVENT_SHORT = {
        "earnings_call": "📣法說",
        "investor_day": "🏢投資日",
        "filing": "📋財報",
        "revenue": "💰營收",
    }
    if alert_rows:
        lines.append(f"📣 **重大事件** (近3日，{len(alert_rows)} 件)")
        for r in alert_rows[:4]:
            label = _EVENT_SHORT.get(r.event_type, r.event_type)
            subj = str(r.subject or "")[:20]
            lines.append(f"  {r.date} {r.stock_id} {label} {subj}")
        if len(alert_rows) > 4:
            lines.append(f"  …共 {len(alert_rows)} 件")
        lines.append("")

    # ── 3. 持倉監控狀態 ──────────────────────────────────────────────
    with get_session() as session:
        watch_counts: dict[str, int] = {}
        for st in ("active", "stopped_loss", "taken_profit", "expired"):
            cnt = session.execute(select(func.count()).select_from(WatchEntry).where(WatchEntry.status == st)).scalar()
            watch_counts[st] = cnt or 0

    total_watch = sum(watch_counts.values())
    if total_watch > 0:
        parts = []
        if watch_counts["active"]:
            parts.append(f"監控中 {watch_counts['active']} 支")
        if watch_counts["stopped_loss"]:
            parts.append(f"⛔止損 {watch_counts['stopped_loss']} 支")
        if watch_counts["taken_profit"]:
            parts.append(f"✅止利 {watch_counts['taken_profit']} 支")
        if watch_counts["expired"]:
            parts.append(f"⏰過期 {watch_counts['expired']} 支")
        lines.append(f"👁 **持倉監控**：{' | '.join(parts)}")
        lines.append("")

    # ── 4. 籌碼異動警報（快速摘要）──────────────────────────────────
    try:
        from src.data.database import get_effective_watchlist

        _anomaly = _compute_anomaly_scan(get_effective_watchlist())
        _total_anomaly = sum(len(df) for df in _anomaly.values())
        if _total_anomaly > 0:
            parts_a = []
            if not _anomaly["volume_spike"].empty:
                sids = ", ".join(_anomaly["volume_spike"]["stock_id"].head(3).tolist())
                parts_a.append(f"📊量增{len(_anomaly['volume_spike'])}({sids})")
            if not _anomaly["inst_buy"].empty:
                sids = ", ".join(_anomaly["inst_buy"]["stock_id"].head(3).tolist())
                parts_a.append(f"🏦外資{len(_anomaly['inst_buy'])}({sids})")
            if not _anomaly["sbl_spike"].empty:
                sids = ", ".join(_anomaly["sbl_spike"]["stock_id"].head(3).tolist())
                parts_a.append(f"🔴借券{len(_anomaly['sbl_spike'])}({sids})")
            if not _anomaly["broker_conc"].empty:
                sids = ", ".join(_anomaly["broker_conc"]["stock_id"].head(3).tolist())
                parts_a.append(f"🎯主力{len(_anomaly['broker_conc'])}({sids})")
            if not _anomaly.get("daytrade_risk", pd.DataFrame()).empty:
                df_dt = _anomaly["daytrade_risk"]
                sids = ", ".join(df_dt["stock_id"].head(3).tolist())
                parts_a.append(f"⚡隔沖{len(df_dt)}({sids})")
            lines.append(f"🚨 **籌碼異動** ({_total_anomaly}筆)  " + "  ".join(parts_a))
            lines.append("")
    except Exception:
        logger.debug("Discord 摘要：籌碼異動區塊失敗", exc_info=True)

    # ── 5. 輪動組合摘要 ──────────────────────────────────────────────
    try:
        from src.portfolio.manager import RotationManager

        rot_portfolios = RotationManager.list_portfolios()
        active_rots = [p for p in rot_portfolios if p["status"] == "active"]
        if active_rots:
            lines.append("📊 **輪動組合**")
            for p in active_rots:
                mgr = RotationManager(p["name"])
                st = mgr.get_status()
                if st:
                    ret_pct = st.get("total_return_pct", 0)
                    n_hold = len(st.get("holdings", []))
                    lines.append(
                        f"  [{p['name']}] {n_hold}/{p['max_positions']} 持倉 | "
                        f"資產 {st['current_capital']:,.0f} ({ret_pct:+.1%})"
                    )
            lines.append("")
    except Exception:
        logger.debug("Discord 摘要：輪動組合區塊失敗", exc_info=True)

    msg = "\n".join(lines)
    # Discord 單訊息上限 2000 字元，保留緩衝；截斷時附省略號避免誤以為資料完整
    return msg[:1897] + "..." if len(msg) > 1900 else msg


def _build_discover_discord_detail(today_str: str, top_n_per_mode: int = 5) -> list[str]:
    """建立各模式 Top N 的 Discord 訊息列表（每個模式一則訊息）。

    從 DiscoveryRecord 查詢今日各模式的推薦結果，
    格式化為 Discord 推播訊息（含分數 + 進出場建議）。

    Returns:
        list[str]: 每個模式一則訊息，可能為空列表。
    """
    import datetime

    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import DiscoveryRecord

    today = datetime.date.fromisoformat(today_str)
    mode_labels = {
        "momentum": "動能掃描",
        "swing": "波段掃描",
        "value": "價值掃描",
        "dividend": "高息掃描",
        "growth": "成長掃描",
    }

    messages: list[str] = []

    with get_session() as session:
        for mode_key, label in mode_labels.items():
            rows = session.execute(
                select(
                    DiscoveryRecord.rank,
                    DiscoveryRecord.stock_id,
                    DiscoveryRecord.stock_name,
                    DiscoveryRecord.close,
                    DiscoveryRecord.composite_score,
                    DiscoveryRecord.technical_score,
                    DiscoveryRecord.chip_score,
                    DiscoveryRecord.fundamental_score,
                    DiscoveryRecord.chip_tier,
                    DiscoveryRecord.entry_price,
                    DiscoveryRecord.stop_loss,
                    DiscoveryRecord.take_profit,
                    DiscoveryRecord.industry_category,
                )
                .where(
                    DiscoveryRecord.scan_date == today,
                    DiscoveryRecord.mode == mode_key,
                )
                .order_by(DiscoveryRecord.rank)
                .limit(top_n_per_mode)
            ).all()

            if not rows:
                continue

            lines: list[str] = [f"**{label}** Top {len(rows)} ({today_str})", ""]
            lines.append("```")
            lines.append(
                f"{'#':>2} {'代號':>6} {'名稱':<6} {'收盤':>7} {'綜合':>5} {'技術':>5} {'籌碼':>5} {'基本':>5} {'層':>3} {'產業':<8}"
            )
            lines.append("-" * 70)

            for r in rows:
                name = str(r.stock_name or "")[:6]
                industry = str(r.industry_category or "")[:8]
                chip_tier = str(r.chip_tier or "N/A")
                close_str = f"{r.close:.1f}" if r.close is not None else "   -"
                comp = f"{r.composite_score:.2f}" if r.composite_score is not None else "  -"
                tech = f"{r.technical_score:.2f}" if r.technical_score is not None else "  -"
                chip = f"{r.chip_score:.2f}" if r.chip_score is not None else "  -"
                fund = f"{r.fundamental_score:.2f}" if r.fundamental_score is not None else "  -"
                lines.append(
                    f"{r.rank:>2} {r.stock_id:>6} {name:<6} {close_str:>7} {comp:>5} {tech:>5} {chip:>5} {fund:>5} {chip_tier:>3} {industry:<8}"
                )
            lines.append("```")

            # 進出場建議（Top 3）
            ee_rows = [r for r in rows[:3] if r.entry_price and r.stop_loss and r.take_profit]
            if ee_rows:
                lines.append("**進出場建議：**")
                for r in ee_rows:
                    sl_pct = (r.stop_loss - r.entry_price) / r.entry_price
                    tp_pct = (r.take_profit - r.entry_price) / r.entry_price
                    lines.append(
                        f"  {r.stock_id} {r.stock_name or ''}: "
                        f"進場 {r.entry_price:.1f} / 止損 {r.stop_loss:.1f}({sl_pct:+.1%}) / 止利 {r.take_profit:.1f}({tp_pct:+.1%})"
                    )

            msg = "\n".join(lines)
            messages.append(msg[:1897] + "..." if len(msg) > 1900 else msg)

    return messages


def _export_dashboard_step(top_n: int = 20, target_date: date | None = None) -> None:
    """morning-routine Step 16：匯出 Dashboard JSON。

    失敗不阻擋例行流程（外層 Step loop 已 try/except）。

    C3 修復（2026-05-09 audit）：接受 target_date 參數並 plumb 至 cmd_export_dashboard，
    避免 morning-routine 跨午夜執行時 cmd_export_dashboard 重新呼叫 date.today() 取到隔日，
    導致 _build_discover 查詢 scan_date 對不到當日 DiscoveryRecord（May 8 discover=0 根因之一）。
    """
    import argparse as _argparse

    from src.cli.export_dashboard_cmd import cmd_export_dashboard

    cmd_export_dashboard(
        _argparse.Namespace(
            date=target_date.isoformat() if target_date else None,
            top=top_n,
            event_days=30,
            out=None,
            regenerate_ai_summary=False,
        )
    )


def _baseline_regression_check(state: dict | None = None) -> None:
    """morning-routine Step 17：對比當前 portfolio 指標與 baseline。

    失敗不阻擋（外層 try/except）；發現 regression 時：
      - 寫入 baseline_state["regressions"] 供 _build_morning_discord_summary 顯示
      - stdout 列出明細
      - 不抛例外（不影響後續步驟，與 Step 15 風格一致）
    """
    from src.cli.baseline_cmd import (
        collect_current_metrics,
        compare_metrics,
        load_baseline,
    )

    baseline = load_baseline()
    if not baseline:
        print("  [baseline] 找不到 baseline_metrics.json — 請先執行 python main.py update-baseline --confirm")
        if state is not None:
            state["missing"] = True
        return

    current = collect_current_metrics(portfolio_names=list(baseline.keys()))
    all_findings = []
    for name, bm in baseline.items():
        cm = current.get(name)
        if cm is None:
            continue
        all_findings.extend(compare_metrics(bm, cm, tolerance=1.0))

    regressions = [f for f in all_findings if f.is_regression]
    if state is not None:
        state["regressions"] = regressions

    if regressions:
        print(f"  ⚠ Baseline Regression: {len(regressions)} 項")
        for r in regressions[:10]:
            print(f"    🔴 [{r.portfolio_name}] {r.metric}: {r.reason}")
    else:
        print("  ✅ Baseline 守門通過（無 regression）")


def _check_strategy_decay(scan_date: date | None = None) -> None:
    """檢查所有 Discover 模式的策略衰減（供 morning-routine Step 15 呼叫）。

    2026-05-15 sprint：除了 print，新增 StrategyDecayLog 落庫供 5/29 audit
    訊號穩定性時序對比。同日重複呼叫採 update path 避免 UniqueConstraint 衝突。
    持久化失敗不擋後續步驟（warning + continue）。
    """
    from src.discovery.performance import check_all_modes_decay, compute_signal_stability

    mode_labels = {
        "momentum": "Momentum 短線動能",
        "swing": "Swing 中期波段",
        "value": "Value 價值修復",
        "dividend": "Dividend 高息存股",
        "growth": "Growth 高成長",
    }

    results = check_all_modes_decay(recent_days=30, holding_days=10)
    has_decay = False
    log_date = scan_date or date.today()
    rows_to_persist: list[dict] = []

    for r in results:
        mode = r.get("mode", "?")
        label = mode_labels.get(mode, mode)

        # 訊號穩定性 long-run 監控（§7.3）：與績效衰減獨立，無論有無 closed trade 都算
        stab = compute_signal_stability(mode, window_days=30, top_n=5, reference_date=log_date)
        jac_mean = stab.mean_jaccard
        jac_pairs = len(stab.pairs)
        jac_str = f"Jaccard={jac_mean:.2f}({jac_pairs}對)" if jac_mean is not None else "Jaccard=N/A"

        if r["recent_count"] == 0:
            print(f"  {label}: 近 30 天無足夠推薦績效資料，跳過 | 訊號穩定 {jac_str}")
            rows_to_persist.append(
                {
                    "mode": mode,
                    "recent_win_rate": None,
                    "recent_avg_return": None,
                    "recent_count": 0,
                    "is_decaying": False,
                    "warning": None,
                    "signal_jaccard_mean": jac_mean,
                    "signal_jaccard_pairs": jac_pairs,
                }
            )
            continue

        wr = r["recent_win_rate"]
        avg = r["recent_avg_return"]
        n = r["recent_count"]
        wr_str = f"{wr:.0%}" if wr is not None else "N/A"
        avg_str = f"{avg:+.2%}" if avg is not None else "N/A"
        # 小樣本警示：n < 20 時統計雜訊過大，勝率/均報酬不穩定
        MIN_SAMPLE = 20
        if n < MIN_SAMPLE:
            status = f"⚠ 樣本不足（n={n}）"
        elif r["is_decaying"]:
            status = "⚠ 衰減"
        else:
            status = "✓ 正常"
        print(f"  {label}: 勝率={wr_str}, 均報酬={avg_str} ({n}筆) → {status} | 訊號穩定 {jac_str}")

        warning_text: str | None = None
        if r["is_decaying"] and n >= MIN_SAMPLE:
            has_decay = True
            warning_text = r.get("warning")
            print(f"    {warning_text}")

        rows_to_persist.append(
            {
                "mode": mode,
                "recent_win_rate": wr,
                "recent_avg_return": avg,
                "recent_count": n,
                "is_decaying": bool(r["is_decaying"]),
                "warning": warning_text,
                "signal_jaccard_mean": jac_mean,
                "signal_jaccard_pairs": jac_pairs,
            }
        )

    if not has_decay:
        print("  所有模式績效正常，無衰減警告。")

    # 持久化（同日重複呼叫採 update path）
    try:
        from sqlalchemy import select

        from src.data.database import get_session
        from src.data.schema import StrategyDecayLog

        with get_session() as session:
            for row in rows_to_persist:
                existing = session.execute(
                    select(StrategyDecayLog).where(
                        StrategyDecayLog.scan_date == log_date,
                        StrategyDecayLog.mode == row["mode"],
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    existing.recent_win_rate = row["recent_win_rate"]
                    existing.recent_avg_return = row["recent_avg_return"]
                    existing.recent_count = row["recent_count"]
                    existing.is_decaying = row["is_decaying"]
                    existing.warning = row["warning"]
                    existing.signal_jaccard_mean = row["signal_jaccard_mean"]
                    existing.signal_jaccard_pairs = row["signal_jaccard_pairs"]
                else:
                    session.add(
                        StrategyDecayLog(
                            scan_date=log_date,
                            mode=row["mode"],
                            recent_win_rate=row["recent_win_rate"],
                            recent_avg_return=row["recent_avg_return"],
                            recent_count=row["recent_count"],
                            is_decaying=row["is_decaying"],
                            warning=row["warning"],
                            signal_jaccard_mean=row["signal_jaccard_mean"],
                            signal_jaccard_pairs=row["signal_jaccard_pairs"],
                        )
                    )
            session.commit()
    except Exception as exc:
        print(f"  ⚠ StrategyDecayLog 持久化失敗：{exc}")


# 各模式關鍵因子 — 共用 SSOT（src/constants.py:DISCOVERY_KEY_FACTOR_MAP）
# 避免本地副本與全域定義失同步（先前 swing 因副本錯設 chip_score 被誤殺）
_KEY_FACTORS: dict[str, str] = DISCOVERY_KEY_FACTOR_MAP
_MODE_LABELS: dict[str, str] = {
    "momentum": "Momentum",
    "swing": "Swing",
    "value": "Value",
    "dividend": "Dividend",
    "growth": "Growth",
}


def _ic_level_from_verdict(verdict) -> str:
    """ICVerdict → 顯示用 level（P0 #16）。

    **不可執法一律不得回 "inverse"**——這是 B11 最小切口的核心：過期窗口、
    窗口數不足、樣本不足時只能告警，不能停用模式。原因碼直接當 level 用，
    使 log 說出真正的理由（舊版全部塞進 "insufficient"，見 §6.4 #18）。
    """
    from src.discovery.ic_governance import REASON_OK

    if verdict.reason != REASON_OK:
        return verdict.reason  # stale_window / insufficient_windows / insufficient_samples / no_windows
    ic = verdict.ic
    if ic is None:
        return "no_windows"
    if ic < -0.05:
        return "inverse"
    if ic < 0.05:
        return "weak"
    if ic < 0.10:
        return "decay"
    return "normal"


def _compute_factor_ic_status(today: date | None = None) -> tuple[list[dict], dict[str, "pd.DataFrame"]]:
    """為每個 discover 模式計算 IC：rolling IC（level 判定）+ scanner 用 IC（項目 E）。

    為每個 mode 跑兩次 IC 計算，**共用同一次 DB 查詢**：
      1. `compute_rolling_ic(window_days=14, step_days=7)` — 用於 level 判定（rolling 窗口）
      2. `compute_factor_ic(lookback_days=30)` — 用於 scanner
         `_apply_ic_weight_adjustment` / `_log_factor_effectiveness`（單次靜態 IC）

    Holding days 由 `DISCOVERY_IC_HOLDING_DAYS_MAP` 決定（mode-aware）：
      - momentum: 5（chip_score 短期敏感）
      - swing: 10（中期波段）
      - value/dividend/growth: 20（fundamental_score 中長期兌現）
    避免 fundamental 因子在 5 天 holding 下被誤判 inverse 而誤殺整個模式。

    cutoff 動態擴大為 35 + max(holding_days)，確保 holding=20 模式仍有足夠 forward 樣本。

    C3 修復（2026-05-09 audit）：接受 today 參數，避免在 morning-routine 跨午夜時與
    cmd_morning_routine 開頭設定的 today_str 不一致（推薦由呼叫端傳入）。

    Args:
        today: 基準日期；None 時 fallback 至 date.today()（向後相容）。

    Returns:
        tuple:
          - status_list: 每模式一筆 dict（mode/mode_key/factor/ic/level/sample_count/holding_days/error）
            level ∈ {"normal", "decay", "weak", "inverse", "insufficient", "error"}
          - ic_df_by_mode: {mode_key: ic_df}，ic_df 為 `compute_factor_ic` 輸出
            （factor / ic / evaluable_count / direction），失敗或樣本不足時為空 DataFrame
    """
    from datetime import date as _date
    from datetime import timedelta

    import pandas as pd
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import DailyPrice, DiscoveryRecord
    from src.discovery.ic_governance import select_enforceable_ic
    from src.discovery.scanner._functions import compute_factor_ic, compute_rolling_ic

    if today is None:
        today = _date.today()

    results: list[dict] = []
    ic_df_by_mode: dict[str, pd.DataFrame] = {}
    # 動態 cutoff：base 35 天 + 最大 holding（如 holding=20 需要 records 在 55 天前才有 forward 報酬）
    max_holding = max(DISCOVERY_IC_HOLDING_DAYS_MAP.values(), default=5)
    cutoff = today - timedelta(days=35 + max_holding)

    for mode, key_factor in _KEY_FACTORS.items():
        holding_days = DISCOVERY_IC_HOLDING_DAYS_MAP.get(mode, 5)
        entry: dict = {
            "mode": _MODE_LABELS.get(mode, mode),
            "mode_key": mode,
            "factor": key_factor,
            "holding_days": holding_days,
        }
        ic_df_by_mode[mode] = pd.DataFrame()  # 預設空，失敗或樣本不足時保持
        try:
            with get_session() as session:
                stmt = select(
                    DiscoveryRecord.scan_date,
                    DiscoveryRecord.stock_id,
                    DiscoveryRecord.close,
                    DiscoveryRecord.technical_score,
                    DiscoveryRecord.chip_score,
                    DiscoveryRecord.fundamental_score,
                    DiscoveryRecord.news_score,
                ).where(
                    DiscoveryRecord.mode == mode,
                    DiscoveryRecord.scan_date >= cutoff,
                )
                rows = session.execute(stmt).all()
                if len(rows) < 20:
                    # §6.4 #18：此處才是真正的「樣本不足」；窗口算不出來的情況
                    # 另有 level="no_windows"，不再共用同一標籤（舊版兩者都印
                    # 「樣本不足（n=260，需 ≥20）」這種自相矛盾的訊息）。
                    entry.update({"ic": None, "level": "insufficient", "sample_count": len(rows)})
                    results.append(entry)
                    continue
                df_records = pd.DataFrame(
                    rows,
                    columns=[
                        "scan_date",
                        "stock_id",
                        "close",
                        "technical_score",
                        "chip_score",
                        "fundamental_score",
                        "news_score",
                    ],
                )
                stock_ids = df_records["stock_id"].unique().tolist()
                price_rows = session.execute(
                    select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                        DailyPrice.stock_id.in_(stock_ids),
                        DailyPrice.date >= cutoff,
                    )
                ).all()
                if not price_rows:
                    entry.update({"ic": None, "level": "no_prices", "sample_count": len(rows)})
                    results.append(entry)
                    continue
                df_prices = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])

            # 路徑 1：rolling IC，用於 level 判定（mode-aware holding）
            rolling_df = compute_rolling_ic(
                df_records, df_prices, holding_days=holding_days, window_days=14, step_days=7
            )

            # 路徑 2：scanner 端用的 static IC（mode-aware holding, lookback_days=30）
            # scanner 端 _apply_ic_weight_adjustment / _compute_ic_decay_adjustment 也已改用
            # DISCOVERY_IC_HOLDING_DAYS_MAP（fallback 路徑）；short-circuit 路徑直接消費此處 ic_df。
            try:
                static_ic = compute_factor_ic(df_records, df_prices, holding_days=holding_days, lookback_days=30)
                if not static_ic.empty:
                    ic_df_by_mode[mode] = static_ic
            except Exception:
                logger.warning("scanner 用 static IC 計算失敗 mode=%s（不影響 level 判定）", mode, exc_info=True)

            # P0 #16：三道執法閘門（時效 / 窗口數 / 樣本）統一由 ic_governance 判定。
            # 舊版直接取 factor_df["ic"].iloc[-1] 執法，導致模式停掃後窗口凍結、
            # 每天重讀同一份過期 IC（實測 34 天前、n=40）→ 模式無法自證恢復。
            verdict = select_enforceable_ic(rolling_df, key_factor, as_of=today, holding_days=holding_days, mode=mode)
            entry.update(
                {
                    "ic": verdict.ic,
                    "sample_count": len(rows),
                    "verdict": verdict,
                    "enforceable": verdict.enforceable,
                    "reason": verdict.reason,
                    "window_end": verdict.window_end,
                    "staleness_days": verdict.staleness_days,
                    "window_count": verdict.window_count,
                    "min_sample_count": verdict.min_sample_count,
                }
            )
            entry["level"] = _ic_level_from_verdict(verdict)
            results.append(entry)
        except Exception as exc:
            # C1 修復：明確記錄失敗而非靜默 continue
            logger.warning("IC 計算失敗 mode=%s factor=%s: %s", mode, key_factor, exc, exc_info=True)
            entry.update({"ic": None, "level": "error", "error": str(exc)})
            results.append(entry)

    return results, ic_df_by_mode


def _check_factor_ic_decay(ic_status: list[dict] | None = None) -> list[dict]:
    """檢查各模式關鍵因子 IC 衰退（供 morning-routine 呼叫）。

    C1 修復：
    - 若未傳 ic_status 則重新計算
    - 列印結果時明確區分「失敗」與「正常」，避免靜默誤導
    - 回傳 ic_status 供 Discord 摘要使用

    Args:
        ic_status: 預先計算好的 IC 狀態（Step 8c 已算過時重用）

    Returns:
        list[dict]: 同 _compute_factor_ic_status 回傳格式（status_list 部分）
    """
    if ic_status is None:
        ic_status, _ = _compute_factor_ic_status()

    has_decay = False
    failed_modes: list[str] = []
    unenforceable: list[str] = []
    for s in ic_status:
        label = s.get("mode", "?")
        factor = s.get("factor", "?")
        level = s.get("level", "?")
        ic = s.get("ic")
        if level == "error":
            failed_modes.append(label)
            print(f"  ⚠ {label} 關鍵因子 {factor} IC 計算失敗：{s.get('error', 'unknown')}")
            continue
        if level == "insufficient":
            n = s.get("sample_count", 0)
            print(f"  {label} 關鍵因子 {factor}：推薦記錄不足（n={n}，需 ≥20），跳過")
            continue
        if level == "no_prices":
            print(f"  {label} 關鍵因子 {factor}：無對應價格資料，跳過")
            continue
        # P0 #16：可觀測但不可執法——印出 IC 供人判讀，並明說為何不執法
        verdict = s.get("verdict")
        if verdict is not None and not verdict.enforceable:
            unenforceable.append(label)
            print(f"  ⚠ {label} 關鍵因子 {verdict.describe()}")
            continue
        if level == "inverse":
            has_decay = True
            print(f"  🔴 {label} 關鍵因子 {factor} IC={ic:+.4f}（反向，因子失效）")
        elif level == "weak":
            has_decay = True
            print(f"  ⚠ {label} 關鍵因子 {factor} IC={ic:+.4f}（弱，|IC| < 0.05）")
        elif level == "decay":
            has_decay = True
            print(f"  ⚠ {label} 關鍵因子 {factor} IC={ic:+.4f}（衰減，< 0.10 門檻）")
        else:
            print(f"  ✓ {label} 關鍵因子 {factor} IC={ic:+.4f}")

    if failed_modes:
        print(f"  ⚠ {len(failed_modes)} 個模式 IC 計算失敗：{', '.join(failed_modes)}（請檢查 log）")
    elif not has_decay and not unenforceable:
        print("  所有模式關鍵因子 IC 正常。")
    if unenforceable:
        print(f"  ℹ {len(unenforceable)} 個模式 IC 未達執法門檻（樣本/窗口/時效），僅告警：{', '.join(unenforceable)}")

    return ic_status


def _inverse_modes_from_ic_status(ic_status: list[dict]) -> list[str]:
    """從 IC 狀態抽出「反向**且可執法**」模式的 mode_key 列表（M2 用）。

    P0 #16：`level == "inverse"` 只在 `ICVerdict.enforceable` 為真時才會產生
    （見 `_ic_level_from_verdict`），故此處等同於「通過時效/窗口/樣本三道閘門
    後仍判定反向」。**這些模式照常掃描並落庫**，僅由 rotation 層阻擋新買入
    （`portfolio/manager._build_decision_context`），使 IC 得以續算、模式具備
    自動恢復路徑——這正是修復自鎖迴路的關鍵。
    """
    return [s["mode_key"] for s in ic_status if s.get("level") == "inverse"]


def resolve_regime_after_sync(previous: str | None) -> str | None:
    """同步完成後重解 regime（morning-routine Step 8e，P0 #15）。

    Step 0 宏觀壓力預檢在資料同步（Step 1~8）**之前**執行，其 regime 只反映
    「昨日收盤為止」的資料；而 Step 9 掃描與 Step 12 輪動都在同步之後，若沿用
    Step 0 的結果，rotation 會永遠比 discover 慢一個交易日。此函數以同步後的
    資料重解一次。

    狀態機自 P0 #15 起具冪等性（同一 `data_date` 不重複推進 hysteresis），
    故此處重解不會產生副作用，也不會與 Step 9 各 scanner 的判定分歧。

    Args:
        previous: Step 0 取得的 regime（可能為 None＝預檢失敗）。

    Returns:
        重解後的 regime；重解失敗或結果為空時回傳 `previous`（沿用 Step 0）。
    """
    try:
        from src.regime.detector import MarketRegimeDetector

        refreshed = MarketRegimeDetector().detect().get("regime")
    except Exception as exc:
        logger.warning("同步後 regime 重解失敗，沿用 Step 0 結果: %s", exc, exc_info=True)
        return previous
    if not refreshed:
        return previous
    if refreshed != previous:
        print(f"  市場狀態（同步後重解）: {(previous or 'unknown').upper()} → {refreshed.upper()}")
    return refreshed


def _sync_full_market() -> None:
    """同步全市場 TWSE/TPEX 日K線（確保 rotation 持倉等非 watchlist 股票有最新價格）。"""
    from src.data.pipeline import sync_market_data

    result = sync_market_data(days=1)
    print(
        f"  全市場日K: {result.get('daily_price', 0)} 筆, 法人: {result.get('institutional', 0)} 筆, 融資融券: {result.get('margin', 0)} 筆"
    )


def _sync_rotation_dividends() -> None:
    """補抓 rotation 持倉 + pending 標的的股利資料（A3：除息入帳的資料前提）。

    dividend 表原僅隨 watchlist 逐股同步；rotation 標的來自全市場 discover，
    除息日入帳需要當日除息事件已在庫。標的數少（active 組合持倉 + 待成交），
    逐股 FinMind 補抓成本可控（~20 檔 × 0.5s，7 天內抓過者跳過）。
    """
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.pipeline import sync_dividends_for_stocks
    from src.data.schema import RotationPendingOrder, RotationPortfolio, RotationPosition

    with get_session() as session:
        actives = session.execute(
            select(RotationPortfolio.id, RotationPortfolio.name).where(RotationPortfolio.status == "active")
        ).all()
        active_ids = [r[0] for r in actives]
        active_names = [r[1] for r in actives]
        if not active_ids:
            print("  無 active rotation 組合，跳過")
            return
        pos_sids = {
            r[0]
            for r in session.execute(
                select(RotationPosition.stock_id).where(
                    RotationPosition.portfolio_id.in_(active_ids), RotationPosition.status == "open"
                )
            )
        }
        pending_sids = {
            r[0]
            for r in session.execute(
                select(RotationPendingOrder.stock_id).where(
                    RotationPendingOrder.portfolio_name.in_(active_names),
                    RotationPendingOrder.status == "pending",
                )
            )
        }
    from src.portfolio.market_data import SNAPSHOT_BENCHMARK_STOCK_ID

    # benchmark 0050 一併補抓（A3-4 total return 還原的資料前提）
    sids = sorted(pos_sids | pending_sids | {SNAPSHOT_BENCHMARK_STOCK_ID})
    n = sync_dividends_for_stocks(sids)
    print(f"  股利補抓：{len(sids)} 檔標的（含 benchmark 0050），寫入 {n} 筆")


def _verify_data_freshness(today_str: str, check_phantom: bool = True) -> dict:
    """驗證關鍵資料表的新鮮度（sync 完成後、discover 前執行）。

    檢查 DailyPrice (TAIEX) 最新日期是否為今日或昨日（考慮假日）。

    臨時休市哨兵（P0 #14）：行事曆標今日為交易日、但同步後全市場當日筆數
    低於 PHANTOM_TRADING_DAY_MIN_ROWS → 判定疑似臨時休市（颱風假等），
    後續 Step 9 discover / Step 12 rotation update 據此凍結（2026-07-10
    假交易日以陳舊 close 決策，是 7/13 重複買單事故鏈的觸發源）。
    check_phantom=False（--skip-sync 時傳入）跳過此哨兵——未同步時當日
    筆數必然為 0，會誤判。

    Returns:
        dict: {
            "is_stale": bool,      # True 表示資料過時
            "latest": date | None, # TAIEX 最新日期
            "gap_days": int,       # 今天與最新日期差距
            "phantom": bool,       # True 表示疑似臨時休市（颱風假等）
            "message": str,        # 顯示訊息
        }
    """
    import datetime

    from sqlalchemy import func, select

    from src.constants import PHANTOM_TRADING_DAY_MIN_ROWS
    from src.data.calendar import is_trading_day
    from src.data.database import get_session
    from src.data.schema import DailyPrice

    today = datetime.date.fromisoformat(today_str)
    max_stale_days = 3  # 允許最大落後天數（考慮假日/長週末）
    result: dict = {"is_stale": False, "latest": None, "gap_days": 0, "phantom": False, "message": ""}

    try:
        with get_session() as session:
            latest = session.execute(select(func.max(DailyPrice.date)).where(DailyPrice.stock_id == "TAIEX")).scalar()
            n_today = (
                session.execute(select(func.count()).select_from(DailyPrice).where(DailyPrice.date == today)).scalar()
                or 0
            )

        # ── 臨時休市哨兵：交易日但全市場無今日資料 ──
        if check_phantom and is_trading_day(today) and n_today < PHANTOM_TRADING_DAY_MIN_ROWS:
            msg = (
                f"今日 {today} 行事曆為交易日，但全市場僅 {n_today} 筆"
                f"（< {PHANTOM_TRADING_DAY_MIN_ROWS}）——疑似臨時休市（颱風假？）。"
                "已凍結 discover 與 rotation update；確認後請將該日加入 "
                "src/data/calendar.py:_UNSCHEDULED_CLOSURES"
            )
            print(f"  🚨 臨時休市哨兵：{msg}")
            result["phantom"] = True
            result["is_stale"] = True
            result["message"] = msg
            return result

        if latest is None:
            msg = "DailyPrice 無 TAIEX 資料"
            print(f"  ⚠️ 資料新鮮度警告：{msg}，Discover 結果可能不準確")
            result["is_stale"] = True
            result["message"] = msg
            return result

        gap = (today - latest).days
        result["latest"] = latest
        result["gap_days"] = gap
        if gap > max_stale_days:
            msg = f"TAIEX 最新資料為 {latest}（落後 {gap} 天），Discover 可能使用過期數據"
            print(f"  ⚠️ 資料新鮮度警告：{msg}")
            result["is_stale"] = True
            result["message"] = msg
        else:
            print(f"  ✓ 資料新鮮度正常：TAIEX 最新 {latest}（{gap} 天前）")
            result["message"] = f"TAIEX 最新 {latest}（{gap} 天前）"
    except Exception as e:
        logger.warning("資料新鮮度檢查失敗: %s", e)
        result["is_stale"] = True
        result["message"] = f"檢查失敗：{e}"
    return result


@contextmanager
def _run_lock(lock_path: Path | None = None):
    """morning-routine 重入保護（P0 止血包 #7）：flock 排他鎖。

    - 選 flock 的理由：process 死亡（crash/kill）時 OS 自動釋放，無 stale lock
      問題，因此不需要 --force 逃生口。
    - 鎖檔固定放本機 data/（flock 在 iCloud/網路磁碟上不可靠，勿移至同步目錄）。
    - 取鎖失敗（另一個 morning-routine 執行中）→ 印訊息後 SystemExit(1)。
    - Windows 無 fcntl → 降級為無鎖並告警（graceful degradation 但不靜默）。
    """
    path = lock_path or Path("data/.morning_routine.lock")
    try:
        import fcntl
    except ImportError:
        logger.warning("此平台不支援 fcntl，morning-routine 重入保護停用")
        yield
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(path, "w", encoding="utf-8")
    try:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"⚠️ 另一個 morning-routine 正在執行中（鎖檔 {path}），本次退出。")
            raise SystemExit(1) from None
        # 寫入 PID + 啟動時間供診斷（誰持有鎖）
        fd.write(f"pid={os.getpid()} started={datetime.now().isoformat()}\n")
        fd.flush()
        yield
    finally:
        fd.close()  # close 即釋放 flock


def _run_db_backup() -> None:
    """Step 18：DB 備份（P0 止血包 #1）——本地 backups/ + 異地副本（iCloud）。"""
    from src.config import settings
    from src.data.database import backup_db

    backup_path = backup_db()
    if backup_path is None:
        print("  ⚠️ 備份未執行（非檔案型 DB 或 DB 檔不存在）")
        return
    print(f"  ✓ 本地備份：{backup_path}")
    offsite_dir = settings.backup.offsite_dir
    if offsite_dir:
        print(f"  ✓ 異地副本目錄：{offsite_dir}（失敗時見 log warning）")
    else:
        print("  - 異地副本停用（backup.offsite_dir 為空）")


def _format_crisis_availability(stress_result: dict | None) -> str | None:
    """組出「crisis 訊號可用 N/7」自檢字串（P0 止血包 #5）；無 availability 資料回 None。

    availability 來自 detect_crisis_signals：區分「訊號未觸發」與「資料缺失無法計算」
    （TW_VIX 已死 → vix_spike 恆 False，此自檢讓缺口每天可見）。
    """
    if not stress_result:
        return None
    avail = stress_result.get("availability") or {}
    if not avail:
        return None
    total = len(avail)
    ok = sum(1 for v in avail.values() if v)
    missing = [k for k, v in avail.items() if not v]
    if missing:
        return f"crisis 訊號可用 {ok}/{total}（不可用：{', '.join(missing)}）"
    return f"crisis 訊號可用 {ok}/{total}"


def cmd_morning_routine(args: argparse.Namespace) -> None:
    """每日早晨例行流程進入點（重入保護 wrapper，主體見 _run_morning_routine）。"""
    with _run_lock():
        _run_morning_routine(args)


def _run_morning_routine(args: argparse.Namespace) -> None:
    """每日早晨例行流程。

    依序執行：
      Step 1  sync-info           同步全市場股票基本資料（產業分類 + 上市/上櫃別）
      Step 2  sync                同步日K線資料（OHLCV，watchlist + TAIEX）
      Step 3  compute             計算技術指標（watchlist）
      Step 4  sync-mops           同步 MOPS 重大訊息公告
      Step 5  sync-revenue        同步全市場月營收（最近 1 個月）
      Step 7  sync-sbl            同步全市場借券賣出（TWSE TWT96U，3日）
      Step 8  sync-broker         同步 watchlist 分點資料（5日，歷史累積）
      Step 8b sync-market         同步全市場 TWSE/TPEX 日K線（確保 rotation 持倉有最新價格）
      Step 8d sync-features       計算全市場 DailyFeature（Feature Store，須在 8b 之後）
      Step 9  discover all        五模式全市場掃描（--skip-sync，不重複同步）
      Step 9b sync-broker         補抓 discover 候選股分點資料（使用今日 DiscoveryRecord）
      Step 10 alert-check         MOPS 重大事件警報（近3日）
      Step 11 watch update-status 批次更新持倉止損/止利/過期狀態
      Step 11b sync-dividends     補抓 rotation 持倉/pending 標的股利（除息入帳前提，A3）
      Step 12 rotation update     輪動組合每日更新（所有 active portfolio）
      Step 13 revenue-scan        高成長掃描（YoY≥10%，Top 5）
      Step 14 anomaly-scan        籌碼異動掃描
      Step 15 strategy-decay      策略衰減監控（30天績效趨勢）
      最終     Discord 推播綜合摘要（需加 --notify）

    Flags:
      --dry-run     只顯示步驟與摘要，不執行任何操作
      --skip-sync   跳過 Step 1–8b（所有資料同步），適合資料已新鮮時使用
      --top N       discover 的 Top N（預設 20）
      --notify      執行完畢後推播 Discord 摘要

    Step 8 說明：
      每日同步 watchlist 分點資料，使 Smart Broker（8F）所需的歷史勝率資料自然累積。
      約 20 個交易日後，watchlist 股票即可觸發 Smart Broker 計算；60~120 天後準確度最高。

    Step 8b 說明：
      TWSE/TPEX 官方資料（6 次 API）同步全市場日K線，確保 rotation 持倉中非 watchlist
      股票也有最新收盤價。此步驟在 discover 之前執行，使 discover 與 rotation 都能
      使用當日資料。
    """
    import datetime

    init_db()

    dry_run: bool = getattr(args, "dry_run", False)
    skip_sync: bool = getattr(args, "skip_sync", False)
    top_n: int = getattr(args, "top", 20)
    notify: bool = getattr(args, "notify", False)

    # P0 止血包 #5：dead-man ping（dry-run 不 ping，避免手動預覽干擾排程監控）
    from src.notification.healthchecks import ping_healthchecks

    if not dry_run:
        ping_healthchecks("start")

    # C3 修復（2026-05-09 audit）：建立單一 today 物件貫穿全流程，避免跨午夜執行時
    # 各 step 各自呼叫 date.today() 拿到不同日期（observed: 23:13 啟動跑到 02:17 隔日）。
    # 此 today 會被傳遞給 Step 8c IC 預檢與 Step 16 export-dashboard，
    # 確保 scan_date / target_date 與此處 today_str 完全對齊。
    today = datetime.date.today()
    today_str = today.strftime("%Y-%m-%d")
    TOTAL = 18

    def _step(n: int | str, title: str) -> None:
        print(f"\n{'═' * 64}")
        print(f"  [Step {n}/{TOTAL}] {title}")
        print(f"{'═' * 64}")

    def _skip(reason: str) -> None:
        print(f"  >> 跳過（{reason}）")

    # ── 交易日檢查：非交易日跳過全部流程 ──
    import datetime as _dt  # noqa: F401  保留以備後續使用

    from src.data.calendar import has_calendar_data, is_trading_day

    if not dry_run and not is_trading_day(today):
        print(f"\n{'═' * 64}")
        print(f"  今日 {today_str} 非交易日（休市），跳過早晨例行流程。")
        if not has_calendar_data(today.year):
            print(f"  （注意：{today.year} 年假日資料尚未建立，僅依週末判斷）")
        print(f"{'═' * 64}\n")
        # 非交易日 short-circuit 屬正常結束（避免假日誤發 dead-man 告警）
        ping_healthchecks("success")
        return

    # ── Step 0: VIX 同步 + 宏觀壓力預檢（Macro Stress Check）──────
    print(f"\n{'═' * 64}")
    print("  [Step 0] 宏觀壓力預檢（Macro Stress Check）")
    print(f"{'═' * 64}")
    stress_result: dict = {}
    # C3 修復：regime_now 預設 None，Step 0 失敗時保持 None，
    # Rotation/Discord 皆可識別「未知 regime」並走保守 fallback，
    # 不再假性返回 "sideways" 造成 Crisis 保護被誤略過。
    regime_now: str | None = None
    if dry_run:
        # dry-run 仍執行 stress check（純讀 DB，無副作用），
        # 讓 Discord 預覽與實際執行顯示一致的 regime（M4）。
        try:
            stress_result = _compute_macro_stress_check()
            regime_now = stress_result.get("regime")
            print(f"  [dry-run] 市場狀態預覽: {(regime_now or 'unknown').upper()}")
            print(f"  [dry-run] {stress_result.get('summary', '')}")
            avail_msg = _format_crisis_availability(stress_result)
            if avail_msg:
                print(f"  [dry-run] {avail_msg}")
        except Exception as exc:
            logger.warning("dry-run stress check 失敗: %s", exc, exc_info=True)
            stress_result = {}
    else:
        # 先同步 VIX（失敗不中斷流程）
        try:
            from src.data.pipeline import sync_taiwan_vix

            vix_count = sync_taiwan_vix()
            if vix_count > 0:
                print(f"  台灣 VIX 同步：{vix_count} 筆")
        except (ConnectionError, OSError, ValueError, KeyError) as e:
            logger.warning("台灣 VIX 同步失敗（不影響流程）: %s", e)

        try:
            from src.data.pipeline import sync_us_vix

            us_vix_count = sync_us_vix()
            if us_vix_count > 0:
                print(f"  美國 VIX 同步：{us_vix_count} 筆")
        except (ConnectionError, OSError, ValueError, KeyError) as e:
            logger.warning("美國 VIX 同步失敗（不影響流程）: %s", e)

        # C2 修復：stress check 外層 try/except，任何例外（含 OperationalError）皆不中斷流程
        try:
            stress_result = _compute_macro_stress_check()
            regime_now = stress_result.get("regime")
        except Exception as exc:
            logger.warning("Step 0 宏觀壓力預檢失敗: %s — Rotation 將走 regime fallback", exc, exc_info=True)
            stress_result = {"regime": None, "summary": f"壓力預檢失敗：{exc}"}
            regime_now = None

        if regime_now:
            print(f"  市場狀態: {regime_now.upper()}")
        else:
            print("  !! 市場狀態: UNKNOWN（壓力預檢失敗或資料不足，Rotation 將使用保守預設）")
        print(f"  {stress_result.get('summary', '')}")
        # P0 止血包 #5：crisis 訊號可用性自檢（TW_VIX 已死 → 每天可見 6/7）
        avail_msg = _format_crisis_availability(stress_result)
        if avail_msg:
            print(f"  🩺 {avail_msg}")
        if stress_result.get("breadth_downgraded"):
            bpct = stress_result.get("breadth_below_ma20_pct", 0.0) or 0.0
            print(f"  📊 市場廣度警示：{bpct:.0%} 股票跌破 MA20 → regime 降級")
        if stress_result.get("crisis_triggered"):
            print()
            print("  !! CRISIS 模式啟動 — Discover 將使用最嚴格的過濾與保守參數 !!")
            print(f"     5日報酬: {stress_result.get('fast_return_5d', 0.0):+.1%}")
            print(f"     連跌天數: {stress_result.get('consec_decline_days', 0)}")
            print(f"     波動率倍數: {stress_result.get('vol_ratio', 0.0):.1f}x")
            sigs = stress_result.get("signals", {})
            if sigs.get("panic_volume"):
                print("     爆量長黑: ✓（成交量 > 20日均量 × 1.5 且下跌）")
            if sigs.get("vix_spike"):
                print(f"     台灣 VIX 飆升: ✓（TW_VIX={stress_result.get('vix_val', 0.0):.1f}）")
            if sigs.get("us_vix_spike"):
                print(f"     美國 VIX 飆升: ✓（US_VIX={stress_result.get('us_vix_val', 0.0):.1f}）")
            if sigs.get("single_day_drop"):
                print("     單日急跌: ✓（TAIEX 單日跌幅 > 2.5%）")
            print()

    # M2 + 項目 E：共用變數，Step 8c 寫入 status / disabled_modes / ic_df_by_mode；Step 9 讀取
    ic_status_state: dict = {"ic_status": [], "disabled_modes": [], "ic_df_by_mode": {}}
    # M1：共用變數，Step 9 判斷資料是否過期阻擋
    discover_blocked_state: dict = {"blocked": False, "reason": ""}
    MAX_STALE_HARD_BLOCK_DAYS = 7  # 超過 7 天資料過期直接阻擋 Step 9（M1）
    # Step 17 baseline regression — Step 17 寫入，Discord summary 讀取
    baseline_state: dict = {"regressions": [], "missing": False}
    # P0 #15：Step 0 的 regime 取自**同步前**的資料（Step 0 在 Step 1~8 之前），
    # 故只代表「昨日收盤為止」的判定。Step 9 掃描與 Step 12 輪動都發生在同步後，
    # 必須以同步後的資料重解一次，否則 rotation 永遠比 discover 慢一個交易日。
    # 狀態機已具冪等性（同一 data_date 不重複推進），此處重解無副作用。
    regime_state: dict = {"regime": regime_now}

    def _refresh_regime_after_sync() -> None:
        """同步完成後重解 regime，供 Step 12 輪動使用（P0 #15）。

        skip_sync / dry_run 由 _steps 的 skip 集合負責跳過（未同步時 Step 0
        的判定已是最新可得資料，無須重解）。
        """
        regime_state["regime"] = resolve_regime_after_sync(regime_state["regime"])

    def _step_8c_ic_precheck() -> None:
        """Step 8c：在 Step 9 discover 前檢查關鍵因子 IC，反向模式自動停用（M2）。

        項目 E：同時計算 scanner `_apply_ic_weight_adjustment` / `_log_factor_effectiveness`
        所需的 static IC，存入 ic_status_state["ic_df_by_mode"]，避免 5 個 scanner
        在並行掃描時各自重複查 DB + 重算 IC。

        C3 修復：傳入 outer scope 的 today，與 cmd_morning_routine 開始時的 today_str 對齊。
        """
        status, ic_df_by_mode = _compute_factor_ic_status(today=today)
        ic_status_state["ic_status"] = status
        ic_status_state["ic_df_by_mode"] = ic_df_by_mode
        _check_factor_ic_decay(ic_status=status)
        disabled = _inverse_modes_from_ic_status(status)
        ic_status_state["disabled_modes"] = disabled
        if disabled:
            labels = [_MODE_LABELS.get(m, m) for m in disabled]
            # P0 #16：不再跳過掃描。停用只作用在 rotation 層（阻擋新買入），
            # 掃描照跑並落庫 → IC 可續算 → 模式具備自證恢復路徑（拆自鎖迴路）。
            print(f"  ⚠ {len(disabled)} 個模式 IC 反向且達執法門檻：{', '.join(labels)}")
            print("     → 仍照常掃描落庫（供 IC 續算）；由 rotation 層阻擋新買入")
        else:
            print("  無模式達到 IC 反向執法門檻。")

    def _step_9_discover() -> None:
        """Step 9：discover all；臨時休市/M1 資料過期硬阻擋。

        P0 #16：**不再因 IC 反向跳過任何模式的掃描**。IC 反向的處置移至
        rotation 層（阻擋新買入），此處五模式恆掃、恆落庫，否則停用的模式
        永遠產不出新 `discovery_record`，IC 無從重算而自鎖。
        """
        # P0 #14：臨時休市哨兵（行事曆交易日但全市場無今日資料）
        if freshness.get("phantom"):
            discover_blocked_state["blocked"] = True
            discover_blocked_state["reason"] = "疑似臨時休市（行事曆交易日但全市場無今日資料）"
            print(f"  !! Discover 已阻擋：{discover_blocked_state['reason']}")
            return

        # M1：資料過期硬阻擋
        gap = freshness.get("gap_days", 0) or 0
        if freshness.get("is_stale") and gap > MAX_STALE_HARD_BLOCK_DAYS:
            discover_blocked_state["blocked"] = True
            discover_blocked_state["reason"] = f"資料過期 {gap} 天（> {MAX_STALE_HARD_BLOCK_DAYS} 天硬阻擋）"
            print(f"  !! Discover 已阻擋：{discover_blocked_state['reason']}")
            print("     請先執行完整 sync 後重跑，或確認非預期的資料斷層。")
            return

        ic_df_by_mode = ic_status_state.get("ic_df_by_mode", {}) or {}
        _cmd_discover_all(
            argparse.Namespace(
                skip_sync=True,
                sync_days=30,
                top=top_n,
                min_price=10.0,
                max_price=2000.0,
                min_volume=500_000,
                max_stocks=None,
                min_appearances=1,
                export=None,
                notify=False,
                use_ic_adjustment=True,
                # P0 #16：恆為空——IC 反向不再跳過掃描（處置移至 rotation 層）。
                # 參數保留是為了 `discover all --disable-mode` 之類的人工介入場景。
                disabled_modes=[],
                precomputed_ic_by_mode=ic_df_by_mode,  # 項目 E
            )
        )

    def _step_12_rotation_update() -> None:
        """Step 12：rotation update --all；臨時休市時凍結（T+1 佇列順延至次一真實交易日）。"""
        if freshness.get("phantom"):
            print("  !! 疑似臨時休市（行事曆交易日但全市場無今日資料）——跳過輪動更新，避免以陳舊 close 決策")
            return
        _rotation_update_all(regime=regime_state["regime"])

    # ── Step 1~7: 依序執行 ──────────────────────────────────────────
    _steps = [
        # ── 新增：基礎資料同步（Step 1~6）──────────────────────────
        (
            1,
            "同步全市場基本資料（sync-info）",
            {"dry_run", "skip_sync"},
            lambda: cmd_sync_info(argparse.Namespace(force=False)),
        ),
        (
            2,
            "同步日K線資料（sync watchlist + TAIEX）",
            {"dry_run", "skip_sync"},
            lambda: cmd_sync(argparse.Namespace(stocks=None, start=None, end=None, taiex=False)),
        ),
        (
            3,
            "計算技術指標（compute）",
            {"dry_run", "skip_sync"},
            lambda: cmd_compute(argparse.Namespace(stocks=None)),
        ),
        (
            4,
            "同步 MOPS 重大訊息（sync-mops）",
            {"dry_run", "skip_sync"},
            lambda: cmd_sync_mops(argparse.Namespace()),
        ),
        (
            5,
            "同步全市場月營收（sync-revenue --months 1）",
            {"dry_run", "skip_sync"},
            lambda: cmd_sync_revenue(argparse.Namespace(months=1)),
        ),
        # ── Step 6 (sync-features) 已搬到 8d，改在全市場 sync (8b) 完成後計算 ──
        # ── 原有步驟（重新編號 7~15）──────────────────────────────
        (
            7,
            "同步借券賣出資料（sync-sbl --days 3）",
            {"dry_run", "skip_sync"},
            lambda: cmd_sync_sbl(argparse.Namespace(days=3)),
        ),
        (
            8,
            "同步分點交易資料（watchlist 歷史累積）",
            {"dry_run", "skip_sync"},
            lambda: cmd_sync_broker(argparse.Namespace(stocks=None, days=5, from_discover=False)),
        ),
        (
            "8b",
            "同步全市場 TWSE/TPEX 日K線（rotation 持倉 + discover 候選）",
            {"dry_run", "skip_sync"},
            _sync_full_market,
        ),
        (
            "8d",
            "計算 DailyFeature（sync-features --days 90，全市場資料就緒後）",
            {"dry_run", "skip_sync"},
            lambda: cmd_sync_features(argparse.Namespace(days=90)),
        ),
        (
            "8e",
            "同步後 regime 重解（Step 0 為同步前判定，P0 #15）",
            {"dry_run", "skip_sync"},
            _refresh_regime_after_sync,
        ),
        (
            "8c",
            "關鍵因子 IC 預檢（反向模式自動停用 discover）",
            {"dry_run"},
            _step_8c_ic_precheck,
        ),
        (
            9,
            f"五模式全市場掃描（discover all --skip-sync --top {top_n}）",
            {"dry_run"},
            _step_9_discover,
        ),
        (
            "9b",
            "補抓 discover 候選股分點資料（使用今日 DiscoveryRecord）",
            {"dry_run", "skip_sync"},
            lambda: cmd_sync_broker(argparse.Namespace(stocks=None, days=5, from_discover=True)),
        ),
        (
            10,
            "掃描近期重大事件（alert-check --days 3）",
            {"dry_run"},
            lambda: cmd_alert_check(argparse.Namespace(days=3, types=None, stocks=None, notify=False)),
        ),
        (11, "更新持倉狀態（watch update-status）", {"dry_run"}, lambda: _watch_update_status()),
        (
            "11b",
            "補抓 rotation 標的股利資料（除息入帳前提，A3）",
            {"dry_run", "skip_sync"},
            _sync_rotation_dividends,
        ),
        (12, "輪動組合更新（rotation update --all）", {"dry_run"}, _step_12_rotation_update),
        (
            13,
            "高成長掃描（revenue-scan --min-yoy 10 --top 5）",
            {"dry_run"},
            lambda: cmd_revenue_scan(
                argparse.Namespace(stocks=None, top=5, min_yoy=10.0, min_margin_improve=0.0, notify=False)
            ),
        ),
        (
            14,
            "籌碼異動掃描（anomaly-scan）",
            {"dry_run"},
            lambda: cmd_anomaly_scan(
                argparse.Namespace(
                    stocks=None,
                    lookback=10,
                    vol_mult=DEFAULT_VOL_MULT,
                    inst_threshold=DEFAULT_INST_THRESHOLD,
                    sbl_sigma=DEFAULT_SBL_SIGMA,
                    hhi_threshold=DEFAULT_HHI_THRESHOLD,
                    notify=False,
                )
            ),
        ),
        (
            15,
            "策略衰減監控（30/60/90 天績效趨勢）",
            {"dry_run"},
            lambda: _check_strategy_decay(scan_date=today),
        ),
        (
            16,
            "匯出 Dashboard JSON（iOS App / 下游消費者）",
            {"dry_run"},
            lambda: _export_dashboard_step(top_n=top_n, target_date=today),
        ),
        (
            17,
            "Baseline Regression 守門（對比已凍結指標）",
            {"dry_run"},
            lambda: _baseline_regression_check(state=baseline_state),
        ),
        # P0 止血包 #1（2026-07-05）：放最後 → 備到當日全部同步/更新後的最新狀態
        (
            18,
            "DB 備份（本地 + 異地副本）",
            {"dry_run"},
            _run_db_backup,
        ),
    ]

    active_flags: set[str] = set()
    if dry_run:
        active_flags.add("dry_run")
    if skip_sync:
        active_flags.add("skip_sync")

    # 步驟執行結果追蹤（原子性：單步失敗不影響後續）
    step_results: list[tuple[int | str, str, str]] = []  # (num, title, status)
    freshness: dict = {"is_stale": False, "message": ""}

    for num, title, skip_on, action in _steps:
        _step(num, title)
        if skip_on & active_flags:
            _skip("dry-run" if dry_run else "--skip-sync")
            step_results.append((num, title, "skipped"))
        else:
            try:
                action()
                step_results.append((num, title, "success"))
            except Exception:
                logger.exception("Step %s 執行失敗", num)
                print(f"  !! Step {num} 失敗，繼續執行後續步驟")
                step_results.append((num, title, "failed"))

        # ── 資料新鮮度檢查：在 sync 位置後、discover 前驗證 ──
        # 即使 --skip-sync 也執行（讓使用者誤用 skip-sync 在過期資料上時，M1 硬阻擋仍生效）；
        # 但臨時休市哨兵在 --skip-sync 時跳過（未同步時當日筆數必然為 0，會誤判）
        if num == "8b" and not dry_run:
            freshness = _verify_data_freshness(today_str, check_phantom=not skip_sync)

    # ── 完成提示（含失敗步驟摘要）──────────────────────────────
    failed_steps = [(n, t) for n, t, s in step_results if s == "failed"]
    # P0 止血包 #5：dead-man ping——部分步驟失敗也 ping /fail（healthchecks 顯示紅色）
    if not dry_run:
        ping_healthchecks("fail" if failed_steps else "success")
    suffix = "（dry-run 模式，未執行任何操作）" if dry_run else ""
    print(f"\n{'═' * 64}")
    if failed_steps:
        print(f"  [完成] 早晨例行流程完成（{len(failed_steps)} 個步驟失敗）{suffix}")
        for n, t in failed_steps:
            print(f"    ✗ Step {n}: {t}")
    else:
        print(f"  [完成] 早晨例行流程完成！{suffix}")
    print(f"{'═' * 64}\n")

    # ── Discord 摘要推播（或 dry-run 預覽）────────────────────────
    if notify or dry_run:
        msg = _build_morning_discord_summary(
            today_str,
            top_n,
            freshness=freshness,
            stress_result=stress_result,
            ic_status=ic_status_state.get("ic_status") or None,
            disabled_modes=ic_status_state.get("disabled_modes") or None,
            discover_blocked=discover_blocked_state.get("blocked", False),
            baseline_regressions=baseline_state.get("regressions") or None,
        )
        # 失敗步驟附加至 Discord 摘要
        if failed_steps:
            fail_lines = [f"\n⚠ **{len(failed_steps)} 個步驟失敗：**"]
            for n, t in failed_steps:
                fail_lines.append(f"  ✗ Step {n}: {t}")
            msg += "\n".join(fail_lines) + "\n"
        # m1 修復副作用：失敗步驟附加後再度截斷，避免 Discord 2000 字元上限
        if len(msg) > 1900:
            msg = msg[:1897] + "..."
        discover_msgs = _build_discover_discord_detail(today_str, top_n_per_mode=min(top_n, 10))
        if dry_run:
            # 使用 UTF-8 輸出，繞過 Windows cp950 對 emoji 的限制
            print("-- Discord Summary Preview (dry-run) --")
            sys.stdout.flush()
            sys.stdout.buffer.write(msg.encode("utf-8"))
            sys.stdout.buffer.write(b"\n--\n")
            for i, dm in enumerate(discover_msgs, 1):
                sys.stdout.buffer.write(f"-- Discover Detail {i}/{len(discover_msgs)} --\n".encode("utf-8"))
                sys.stdout.buffer.write(dm.encode("utf-8"))
                sys.stdout.buffer.write(b"\n--\n")
            sys.stdout.buffer.flush()
        else:
            from src.notification.line_notify import send_message

            ok = send_message(msg)
            print(f"Discord 摘要通知: {'成功' if ok else '失敗（請確認 Webhook 設定）'}")
            # 逐模式發送 Discover 詳細推播
            for dm in discover_msgs:
                ok_d = send_message(dm)
                if not ok_d:
                    print("  (部分 Discover 詳細通知失敗)")
                    break
            if discover_msgs:
                print(f"Discord Discover 詳細通知: 已發送 {len(discover_msgs)} 則")
