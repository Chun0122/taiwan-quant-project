"""CLI 早晨例行子命令 — morning-routine + Discord 摘要 + 策略衰減。"""

from __future__ import annotations

import argparse
import logging
import sys

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
)

logger = logging.getLogger(__name__)


def _build_morning_discord_summary(today_str: str, top_n: int) -> str:
    """建立早晨例行報告的 Discord 訊息摘要。

    查詢今日 DiscoveryRecord、近3日 Announcement、以及 WatchEntry 狀態，
    組合成一則 Discord 推播訊息（≤ 1900 字元）。
    """
    import datetime
    from collections import defaultdict

    from sqlalchemy import and_, func, select

    from src.data.database import get_session
    from src.data.schema import Announcement, DiscoveryRecord, WatchEntry

    lines: list[str] = [f"🌅 **早晨例行報告** ({today_str})", ""]
    today = datetime.date.fromisoformat(today_str)

    # ── Step 0: 宏觀壓力預檢警示（crash/bear 時前置顯示）───────────
    try:
        stress = _compute_macro_stress_check()
        if stress.get("breadth_downgraded"):
            bpct = stress.get("breadth_below_ma20_pct", 0.0) or 0.0
            lines.append(f"📊 **市場廣度警示**：{bpct:.0%} 股票跌破 MA20 → regime 降級")
            lines.append("")
        if stress.get("crisis_triggered"):
            lines.append("🚨 **CRISIS 崩盤警示已啟動**")
            lines.append(f"  {stress.get('summary', '')}")
            lines.append("")
        elif stress.get("regime") == "bear":
            lines.append(f"⚠️ **空頭市場** {stress.get('summary', '')}")
            lines.append("")
    except Exception:
        logging.debug("Discord 摘要：宏觀壓力區塊失敗", exc_info=True)

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
                .where(and_(Announcement.date >= since, Announcement.event_type != "general"))
                .order_by(Announcement.date.desc())
                .limit(10)
            ).all()
    except Exception:
        # event_type 欄位可能尚未 migrate，跳過重大事件區塊
        logging.debug("Discord 摘要：重大事件區塊失敗", exc_info=True)

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
        from src.config import settings as _cfg

        _anomaly = _compute_anomaly_scan(_cfg.fetcher.watchlist)
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
        logging.debug("Discord 摘要：籌碼異動區塊失敗", exc_info=True)

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
        logging.debug("Discord 摘要：輪動組合區塊失敗", exc_info=True)

    msg = "\n".join(lines)
    # Discord 單訊息上限 2000 字元，保留緩衝
    return msg[:1900] if len(msg) > 1900 else msg


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
                comp = f"{r.composite_score:.2f}" if r.composite_score else "  -"
                tech = f"{r.technical_score:.2f}" if r.technical_score else "  -"
                chip = f"{r.chip_score:.2f}" if r.chip_score else "  -"
                fund = f"{r.fundamental_score:.2f}" if r.fundamental_score else "  -"
                lines.append(
                    f"{r.rank:>2} {r.stock_id:>6} {name:<6} {r.close:>7.1f} {comp:>5} {tech:>5} {chip:>5} {fund:>5} {chip_tier:>3} {industry:<8}"
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
            messages.append(msg[:1900] if len(msg) > 1900 else msg)

    return messages


def _check_strategy_decay() -> None:
    """檢查所有 Discover 模式的策略衰減（供 morning-routine Step 9 呼叫）。"""
    from src.discovery.performance import check_all_modes_decay

    mode_labels = {
        "momentum": "Momentum 短線動能",
        "swing": "Swing 中期波段",
        "value": "Value 價值修復",
        "dividend": "Dividend 高息存股",
        "growth": "Growth 高成長",
    }

    results = check_all_modes_decay(recent_days=30, holding_days=10)
    has_decay = False

    for r in results:
        mode = r.get("mode", "?")
        label = mode_labels.get(mode, mode)
        if r["recent_count"] == 0:
            print(f"  {label}: 近 30 天無足夠推薦績效資料，跳過")
            continue

        wr = r["recent_win_rate"]
        avg = r["recent_avg_return"]
        wr_str = f"{wr:.0%}" if wr is not None else "N/A"
        avg_str = f"{avg:+.2%}" if avg is not None else "N/A"
        status = "⚠ 衰減" if r["is_decaying"] else "✓ 正常"
        print(f"  {label}: 勝率={wr_str}, 均報酬={avg_str} ({r['recent_count']}筆) → {status}")

        if r["is_decaying"]:
            has_decay = True
            print(f"    {r['warning']}")

    if not has_decay:
        print("  所有模式績效正常，無衰減警告。")


def _sync_full_market() -> None:
    """同步全市場 TWSE/TPEX 日K線（確保 rotation 持倉等非 watchlist 股票有最新價格）。"""
    from src.data.pipeline import sync_market_data

    result = sync_market_data(days=1)
    print(
        f"  全市場日K: {result.get('daily_price', 0)} 筆, 法人: {result.get('institutional', 0)} 筆, 融資融券: {result.get('margin', 0)} 筆"
    )


def _verify_data_freshness(today_str: str) -> None:
    """驗證關鍵資料表的新鮮度（sync 完成後、discover 前執行）。

    檢查 DailyPrice (TAIEX) 最新日期是否為今日或昨日（考慮假日）。
    資料過舊時發出警告但不中斷流程。
    """
    import datetime

    from sqlalchemy import func, select

    from src.data.database import get_session
    from src.data.schema import DailyPrice

    today = datetime.date.fromisoformat(today_str)
    max_stale_days = 3  # 允許最大落後天數（考慮假日/長週末）

    try:
        with get_session() as session:
            latest = session.execute(select(func.max(DailyPrice.date)).where(DailyPrice.stock_id == "TAIEX")).scalar()

        if latest is None:
            print("  ⚠️ 資料新鮮度警告：DailyPrice 無 TAIEX 資料，Discover 結果可能不準確")
            return

        gap = (today - latest).days
        if gap > max_stale_days:
            print(f"  ⚠️ 資料新鮮度警告：TAIEX 最新資料為 {latest}（落後 {gap} 天），Discover 可能使用過期數據")
        else:
            print(f"  ✓ 資料新鮮度正常：TAIEX 最新 {latest}（{gap} 天前）")
    except Exception as e:
        logging.warning("資料新鮮度檢查失敗: %s", e)


def cmd_morning_routine(args: argparse.Namespace) -> None:
    """每日早晨例行流程。

    依序執行：
      Step 1  sync-info           同步全市場股票基本資料（產業分類 + 上市/上櫃別）
      Step 2  sync                同步日K線資料（OHLCV，watchlist + TAIEX）
      Step 3  compute             計算技術指標（watchlist）
      Step 4  sync-mops           同步 MOPS 重大訊息公告
      Step 5  sync-revenue        同步全市場月營收（最近 1 個月）
      Step 6  sync-features       計算全市場 DailyFeature（Feature Store）
      Step 7  sync-sbl            同步全市場借券賣出（TWSE TWT96U，3日）
      Step 8  sync-broker         同步 watchlist 分點資料（5日）+ 補抓 discover 推薦（累積歷史）
      Step 8b sync-market         同步全市場 TWSE/TPEX 日K線（確保 rotation 持倉有最新價格）
      Step 9  discover all        五模式全市場掃描（--skip-sync，不重複同步）
      Step 10 alert-check         MOPS 重大事件警報（近3日）
      Step 11 watch update-status 批次更新持倉止損/止利/過期狀態
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

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    TOTAL = 16

    def _step(n: int | str, title: str) -> None:
        print(f"\n{'═' * 64}")
        print(f"  [Step {n}/{TOTAL}] {title}")
        print(f"{'═' * 64}")

    def _skip(reason: str) -> None:
        print(f"  >> 跳過（{reason}）")

    # ── 交易日檢查：非交易日跳過全部流程 ──
    import datetime as _dt

    from src.data.calendar import has_calendar_data, is_trading_day

    _today_dt = _dt.date.today()
    if not dry_run and not is_trading_day(_today_dt):
        print(f"\n{'═' * 64}")
        print(f"  今日 {today_str} 非交易日（休市），跳過早晨例行流程。")
        if not has_calendar_data(_today_dt.year):
            print(f"  （注意：{_today_dt.year} 年假日資料尚未建立，僅依週末判斷）")
        print(f"{'═' * 64}\n")
        return

    # ── Step 0: VIX 同步 + 宏觀壓力預檢（Macro Stress Check）──────
    print(f"\n{'═' * 64}")
    print("  [Step 0] 宏觀壓力預檢（Macro Stress Check）")
    print(f"{'═' * 64}")
    stress_result: dict = {}
    regime_now: str = "sideways"  # 預設值（dry-run 或 Step 0 異常時使用）
    if dry_run:
        _skip("dry-run")
    else:
        # 先同步 VIX（失敗不中斷流程）
        try:
            from src.data.pipeline import sync_taiwan_vix

            vix_count = sync_taiwan_vix()
            if vix_count > 0:
                print(f"  台灣 VIX 同步：{vix_count} 筆")
        except (ConnectionError, OSError, ValueError, KeyError) as e:
            logging.warning("台灣 VIX 同步失敗（不影響流程）: %s", e)

        try:
            from src.data.pipeline import sync_us_vix

            us_vix_count = sync_us_vix()
            if us_vix_count > 0:
                print(f"  美國 VIX 同步：{us_vix_count} 筆")
        except (ConnectionError, OSError, ValueError, KeyError) as e:
            logging.warning("美國 VIX 同步失敗（不影響流程）: %s", e)

        stress_result = _compute_macro_stress_check()
        regime_now = stress_result.get("regime", "sideways")
        print(f"  市場狀態: {regime_now.upper()}")
        print(f"  {stress_result.get('summary', '')}")
        if stress_result.get("breadth_downgraded"):
            bpct = stress_result.get("breadth_below_ma20_pct", 0.0) or 0.0
            print(f"  📊 市場廣度警示：{bpct:.0%} 股票跌破 MA20 → regime 降級")
        if stress_result.get("crisis_triggered"):
            print()
            print("  !! CRISIS 模式啟動 — Discover 將使用最嚴格的過濾與保守參數 !!")
            print(f"     5日報酬: {stress_result['fast_return_5d']:+.1%}")
            print(f"     連跌天數: {stress_result['consec_decline_days']}")
            print(f"     波動率倍數: {stress_result['vol_ratio']:.1f}x")
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
        (
            6,
            "計算 DailyFeature（sync-features --days 90）",
            {"dry_run", "skip_sync"},
            lambda: cmd_sync_features(argparse.Namespace(days=90)),
        ),
        # ── 原有步驟（重新編號 7~15）──────────────────────────────
        (
            7,
            "同步借券賣出資料（sync-sbl --days 3）",
            {"dry_run", "skip_sync"},
            lambda: cmd_sync_sbl(argparse.Namespace(days=3)),
        ),
        (
            8,
            "同步分點交易資料（watchlist 歷史累積 + discover 補抓）",
            {"dry_run", "skip_sync"},
            lambda: (
                cmd_sync_broker(argparse.Namespace(stocks=None, days=5, from_discover=False)),
                cmd_sync_broker(argparse.Namespace(stocks=None, days=5, from_discover=True)),
            ),
        ),
        (
            "8b",
            "同步全市場 TWSE/TPEX 日K線（rotation 持倉 + discover 候選）",
            {"dry_run", "skip_sync"},
            _sync_full_market,
        ),
        (
            9,
            f"五模式全市場掃描（discover all --skip-sync --top {top_n}）",
            {"dry_run"},
            lambda: _cmd_discover_all(
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
                )
            ),
        ),
        (
            10,
            "掃描近期重大事件（alert-check --days 3）",
            {"dry_run"},
            lambda: cmd_alert_check(argparse.Namespace(days=3, types=None, stocks=None, notify=False)),
        ),
        (11, "更新持倉狀態（watch update-status）", {"dry_run"}, lambda: _watch_update_status()),
        (12, "輪動組合更新（rotation update --all）", {"dry_run"}, lambda: _rotation_update_all(regime=regime_now)),
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
            lambda: _check_strategy_decay(),
        ),
    ]

    active_flags: set[str] = set()
    if dry_run:
        active_flags.add("dry_run")
    if skip_sync:
        active_flags.add("skip_sync")

    # 步驟執行結果追蹤（原子性：單步失敗不影響後續）
    step_results: list[tuple[int | str, str, str]] = []  # (num, title, status)

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

        # ── 資料新鮮度檢查：在 sync 完成後、discover 前驗證 ──
        if num == 8 and not dry_run:
            _verify_data_freshness(today_str)

    # ── 完成提示（含失敗步驟摘要）──────────────────────────────
    failed_steps = [(n, t) for n, t, s in step_results if s == "failed"]
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
        msg = _build_morning_discord_summary(today_str, top_n)
        # 失敗步驟附加至 Discord 摘要
        if failed_steps:
            fail_lines = [f"\n⚠ **{len(failed_steps)} 個步驟失敗：**"]
            for n, t in failed_steps:
                fail_lines.append(f"  ✗ Step {n}: {t}")
            msg += "\n".join(fail_lines) + "\n"
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
