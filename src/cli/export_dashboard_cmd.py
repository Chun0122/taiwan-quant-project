"""每日 Dashboard JSON 匯出 — iOS 監控 App 與其他下游消費者的單一日報來源。

用法：
    python main.py export-dashboard
    python main.py export-dashboard --date 2026-04-30 --top 30
    python main.py export-dashboard --out /tmp/dashboard

詳細 schema 見 docs/dashboard_schema.md。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from src.cli.helpers import init_db
from src.cli.helpers import safe_print as print
from src.config import settings
from src.data.database import get_session
from src.data.schema import DailyPrice, DiscoveryRecord, WatchEntry

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
DEFAULT_TOP_N = 20
DEFAULT_EVENT_DAYS = 30

_DEFAULT_OUT_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/QuantDashboard"

_MODES = ("momentum", "swing", "value", "dividend", "growth")


@dataclass
class _Bundle:
    """組裝中的 dashboard payload + 錯誤紀錄。"""

    payload: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def record_error(self, section: str, exc: BaseException) -> None:
        msg = f"{section}: {type(exc).__name__}: {exc}"
        logger.warning("export-dashboard 區塊失敗 — %s", msg, exc_info=True)
        self.errors.append(msg)


# ---------------------------------------------------------------------------
# 區塊產出
# ---------------------------------------------------------------------------


def _build_regime() -> dict:
    """以 _compute_macro_stress_check() 為來源，整理為 dashboard regime 區塊。"""
    from src.cli.anomaly_cmd import _compute_macro_stress_check

    stress = _compute_macro_stress_check() or {}
    return {
        "state": stress.get("regime"),
        "crisis_triggered": bool(stress.get("crisis_triggered", False)),
        "breadth_downgraded": bool(stress.get("breadth_downgraded", False)),
        "breadth_below_ma20_pct": stress.get("breadth_below_ma20_pct"),
        "taiex_close": stress.get("taiex_close"),
        "fast_return_5d": stress.get("fast_return_5d"),
        "consec_decline_days": stress.get("consec_decline_days"),
        "vol_ratio": stress.get("vol_ratio"),
        "vix_val": stress.get("vix_val"),
        "us_vix_val": stress.get("us_vix_val"),
        "summary": stress.get("summary", ""),
        "_signals": stress.get("signals", {}),  # 內部欄位，build_signals 會用
    }


def _serialize_discovery_row(row) -> dict:
    return {
        "rank": row.rank,
        "stock_id": row.stock_id,
        "stock_name": row.stock_name,
        "close": row.close,
        "composite_score": row.composite_score,
        "scores": {
            "technical": row.technical_score,
            "chip": row.chip_score,
            "fundamental": row.fundamental_score,
            "news": row.news_score,
        },
        "entry": row.entry_price,
        "stop_loss": row.stop_loss,
        "take_profit": row.take_profit,
        "industry": row.industry_category,
        "regime": row.regime,
        "valid_until": row.valid_until.isoformat() if row.valid_until else None,
        "chip_tier": row.chip_tier,
        "chip_tier_change": row.chip_tier_change,
        "concept_bonus": row.concept_bonus,
        "daytrade_penalty": row.daytrade_penalty,
        "entry_trigger": row.entry_trigger,
    }


def _build_discover(target_date: _dt.date, top_n: int) -> dict[str, list[dict]]:
    """從 DiscoveryRecord 撈當日五模式 Top N。"""
    out: dict[str, list[dict]] = {m: [] for m in _MODES}
    with get_session() as session:
        for mode in _MODES:
            rows = (
                session.execute(
                    select(DiscoveryRecord)
                    .where(
                        DiscoveryRecord.scan_date == target_date,
                        DiscoveryRecord.mode == mode,
                    )
                    .order_by(DiscoveryRecord.rank)
                    .limit(top_n)
                )
                .scalars()
                .all()
            )
            out[mode] = [_serialize_discovery_row(r) for r in rows]
    return out


def _build_rotations() -> list[dict]:
    """取所有 active 輪動組合的快照（依 current_capital 降冪排序，第一筆為 primary）。"""
    from src.portfolio.manager import RotationManager

    portfolios = RotationManager.list_portfolios()
    actives = [p for p in portfolios if p.get("status") == "active"]
    if not actives:
        return []

    actives_sorted = sorted(
        actives,
        key=lambda p: p.get("current_capital", 0.0) or 0.0,
        reverse=True,
    )

    out: list[dict] = []
    for p in actives_sorted:
        mgr = RotationManager(p["name"])
        status = mgr.get_status()
        if status is None:
            continue

        holdings = []
        for h in status.get("holdings", []) or []:
            entry_date = h.get("entry_date")
            holdings.append(
                {
                    "stock_id": h.get("stock_id"),
                    "stock_name": h.get("stock_name"),
                    "entry_date": entry_date.isoformat() if isinstance(entry_date, _dt.date) else entry_date,
                    "entry_price": h.get("entry_price"),
                    "current_price": h.get("current_price"),
                    "shares": h.get("shares"),
                    "market_value": h.get("market_value"),
                    "unrealized_pnl": h.get("unrealized_pnl"),
                    "unrealized_pct": h.get("unrealized_pct"),
                    "entry_rank": h.get("entry_rank"),
                }
            )

        out.append(
            {
                "name": status["name"],
                "mode": status["mode"],
                "max_positions": status["max_positions"],
                "holding_days": status["holding_days"],
                "allow_renewal": status["allow_renewal"],
                "initial_capital": status["initial_capital"],
                "current_capital": status["current_capital"],
                "current_cash": status["current_cash"],
                "total_market_value": status.get("total_market_value", 0.0),
                "total_unrealized_pnl": status.get("total_unrealized_pnl", 0.0),
                "total_return_pct": status.get("total_return_pct", 0.0),
                "status": status["status"],
                "updated_at": status.get("updated_at"),
                "holdings": holdings,
            }
        )

    return out


def _build_rotation() -> dict | None:
    """取主要（current_capital 最大的 active）輪動組合的快照。

    保留為 v1 backward-compat alias；內部委派 `_build_rotations()[0]`。
    """
    rotations = _build_rotations()
    return rotations[0] if rotations else None


def _build_watch_entries() -> list[dict]:
    """撈 status='active' 的 WatchEntry。"""
    out: list[dict] = []
    with get_session() as session:
        rows = (
            session.execute(
                select(WatchEntry).where(WatchEntry.status == "active").order_by(WatchEntry.entry_date.desc())
            )
            .scalars()
            .all()
        )
        for r in rows:
            out.append(
                {
                    "id": r.id,
                    "stock_id": r.stock_id,
                    "stock_name": r.stock_name,
                    "entry_date": r.entry_date.isoformat() if r.entry_date else None,
                    "entry_price": r.entry_price,
                    "stop_loss": r.stop_loss,
                    "take_profit": r.take_profit,
                    "quantity": r.quantity,
                    "source": r.source,
                    "mode": r.mode,
                    "status": r.status,
                    "trailing_stop_enabled": bool(r.trailing_stop_enabled),
                    "highest_price_since_entry": r.highest_price_since_entry,
                    "valid_until": r.valid_until.isoformat() if r.valid_until else None,
                    "entry_trigger": r.entry_trigger,
                    "notes": r.notes,
                }
            )
    return out


def _build_signals(regime_block: dict, ic_status: list[dict] | None) -> list[dict]:
    """彙整宏觀 + IC + 資料新鮮度警示為統一陣列。"""
    out: list[dict] = []
    # ── 宏觀 ────────────────────────────────────────────────
    state = regime_block.get("state")
    if regime_block.get("crisis_triggered"):
        out.append(
            {
                "type": "crisis",
                "severity": "critical",
                "message": regime_block.get("summary") or "CRISIS 崩盤訊號觸發",
                "target": None,
            }
        )
    elif state == "bear":
        out.append(
            {
                "type": "bear_market",
                "severity": "warning",
                "message": regime_block.get("summary") or "空頭市場",
                "target": None,
            }
        )
    if regime_block.get("breadth_downgraded"):
        bpct = regime_block.get("breadth_below_ma20_pct") or 0.0
        out.append(
            {
                "type": "breadth_downgrade",
                "severity": "warning",
                "message": f"市場廣度警示：{bpct:.0%} 股票跌破 MA20 → regime 降級",
                "target": None,
            }
        )

    # ── IC 衰減（傳入時才用）───────────────────────────────
    if ic_status:
        for s in ic_status:
            level = s.get("level")
            mode = s.get("mode_key") or s.get("mode")
            factor = s.get("factor")
            ic = s.get("ic")
            if level == "inverse":
                out.append(
                    {
                        "type": "ic_decay",
                        "severity": "critical",
                        "message": f"{s.get('mode')} 關鍵因子 {factor} IC={ic:+.4f}（反向→已暫停）",
                        "target": f"{mode}.{factor}" if mode and factor else None,
                    }
                )
            elif level in ("weak", "decay"):
                if ic is None:
                    continue
                out.append(
                    {
                        "type": "ic_decay",
                        "severity": "warning",
                        "message": f"{s.get('mode')} 關鍵因子 {factor} IC={ic:+.4f}（{level}）",
                        "target": f"{mode}.{factor}" if mode and factor else None,
                    }
                )
            elif level == "error":
                out.append(
                    {
                        "type": "ic_failure",
                        "severity": "warning",
                        "message": f"{s.get('mode')} IC 計算失敗：{s.get('error', 'unknown')}",
                        "target": f"{mode}.{factor}" if mode and factor else None,
                    }
                )

    return out


def _build_data_freshness_signal(target_date: _dt.date) -> dict | None:
    """產出 data_stale 訊號（若有）。"""
    from sqlalchemy import func

    try:
        with get_session() as session:
            latest = session.execute(select(func.max(DailyPrice.date)).where(DailyPrice.stock_id == "TAIEX")).scalar()
    except Exception:
        return None
    if latest is None:
        return {
            "type": "data_stale",
            "severity": "warning",
            "message": "DailyPrice 無 TAIEX 資料",
            "target": "DailyPrice.TAIEX",
        }
    gap = (target_date - latest).days
    if gap > 7:
        severity = "critical"
    elif gap > 3:
        severity = "warning"
    else:
        return None
    return {
        "type": "data_stale",
        "severity": severity,
        "message": f"TAIEX 最新資料為 {latest}（落後 {gap} 天）",
        "target": "DailyPrice.TAIEX",
    }


def _build_strategy_events(days: int) -> list[dict]:
    """收集近 N 天的策略調整事件。"""
    from src.discovery.strategy_events import collect_strategy_events

    events = collect_strategy_events(days=days)
    return [e.to_dict() for e in events]


def _build_ai_summary(target_date: _dt.date, regenerate: bool) -> str | None:
    """AI 摘要：預設不重新呼叫 API；regenerate=True 時對 momentum 模式重新生成。"""
    if not regenerate:
        return None

    import pandas as pd

    from src.discovery.scanner import DiscoveryResult
    from src.report.ai_report import generate_ai_summary

    with get_session() as session:
        rows = (
            session.execute(
                select(DiscoveryRecord)
                .where(
                    DiscoveryRecord.scan_date == target_date,
                    DiscoveryRecord.mode == "momentum",
                )
                .order_by(DiscoveryRecord.rank)
                .limit(20)
            )
            .scalars()
            .all()
        )
    if not rows:
        return None
    df = pd.DataFrame([_serialize_discovery_row(r) for r in rows])
    # generate_ai_summary 期望欄位名 entry_price/stop_loss/take_profit/composite_score/...
    df = df.rename(columns={"entry": "entry_price", "industry": "industry_category"})
    if "scores" in df.columns:
        scores = pd.json_normalize(df["scores"])
        scores.columns = [f"{c}_score" for c in scores.columns]
        df = pd.concat([df.drop(columns=["scores"]), scores], axis=1)

    result = DiscoveryResult(
        mode="momentum",
        scan_date=target_date,
        total_stocks=int(rows[0].total_stocks or 0),
        after_coarse=int(rows[0].after_coarse or 0),
        rankings=df,
        sector_summary=pd.DataFrame(),
        regime=rows[0].regime or "sideways",
    )
    return generate_ai_summary(result, regime=result.regime, top_stocks=df)


# ---------------------------------------------------------------------------
# Portfolio Review（每日績效摘要）
# ---------------------------------------------------------------------------


# 樣本門檻 — 統計指標的最低樣本數（少於此值強制 null，避免無意義數值）
_MIN_SAMPLES_FOR_SHARPE = 10
_MIN_SAMPLES_FOR_MDD = 3


def _build_portfolio_review(rotation_block: dict | None, lookback_days: int) -> dict | None:
    """以 RotationDailySnapshot 計算 portfolio_review 區塊。

    回傳 keys：today_pnl_pct / wtd_return_pct / mtd_return_pct / total_return_pct
            / sharpe_ratio / max_drawdown_pct / win_rate_pct
            / equity_curve / snapshots_count

    資料不足策略：N<10 → sharpe=null；N<3 → mdd/win_rate 也 null；
    snapshot 表為空時除 total_return_pct（從 rotation_block 取）外其餘 null。

    注意：rotation 改名後新舊 snapshot 名稱不一致會斷鏈，這裡只撈當下 primary portfolio。
    """
    if rotation_block is None:
        return None

    portfolio_name = rotation_block.get("name")
    if not portfolio_name:
        return None

    from src.portfolio.manager import RotationManager

    mgr = RotationManager(portfolio_name)
    snapshots = mgr.get_recent_snapshots(n_days=lookback_days)

    total_return_pct = rotation_block.get("total_return_pct")

    review: dict = {
        "today_pnl_pct": None,
        "wtd_return_pct": None,
        "mtd_return_pct": None,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": None,
        "max_drawdown_pct": None,
        "win_rate_pct": None,
        "equity_curve": [],
        "snapshots_count": len(snapshots),
    }

    if not snapshots:
        return review

    # equity_curve（asc by date，給 v1.2 預留）
    review["equity_curve"] = [
        {"date": s["snapshot_date"].isoformat(), "capital": float(s["total_capital"])} for s in snapshots
    ]

    # today / wtd / mtd
    today_snap = snapshots[-1]
    today_date = today_snap["snapshot_date"]
    if today_snap.get("daily_return_pct") is not None:
        review["today_pnl_pct"] = float(today_snap["daily_return_pct"])

    week_start = today_date - _dt.timedelta(days=today_date.weekday())  # 週一=0
    month_start = today_date.replace(day=1)

    def _pct_from(start_date) -> float | None:
        # 取 snapshot_date >= start_date 的第一筆作為基準點
        for s in snapshots:
            if s["snapshot_date"] >= start_date:
                base = s["total_capital"]
                if base and base > 0:
                    return (today_snap["total_capital"] - base) / base
                return None
        return None

    review["wtd_return_pct"] = _pct_from(week_start)
    review["mtd_return_pct"] = _pct_from(month_start)

    # Sharpe（年化）— N>=10 才算
    daily_returns = [s["daily_return_pct"] for s in snapshots if s.get("daily_return_pct") is not None]
    if len(daily_returns) >= _MIN_SAMPLES_FOR_SHARPE:
        import math
        import statistics

        mean = statistics.fmean(daily_returns)
        try:
            stdev = statistics.stdev(daily_returns)
        except statistics.StatisticsError:
            stdev = 0.0
        if stdev > 1e-8:
            review["sharpe_ratio"] = round(mean / stdev * math.sqrt(252), 4)

    # MDD（百分比）+ Win rate（百分比，按日勝率）— N>=3 才算
    if len(snapshots) >= _MIN_SAMPLES_FOR_MDD:
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
        review["max_drawdown_pct"] = round(max_dd * 100, 2)

        if daily_returns:
            wins = sum(1 for r in daily_returns if r > 0)
            review["win_rate_pct"] = round(wins / len(daily_returns) * 100, 2)

    return review


# ---------------------------------------------------------------------------
# Position Timeseries（持倉/Watch 個股小走勢圖）
# ---------------------------------------------------------------------------


def _build_position_timeseries(
    rotations: list[dict] | None,
    watch_block: list[dict],
    days: int,
    target_date: _dt.date,
) -> dict | None:
    """從 DailyPrice 撈所有 rotations.holdings ∪ watch_entries 的最近 N 個交易日 close。

    結構：
        {
          "trading_days": ["2026-04-15", ...],
          "series": {
            "2330": {"close": [...], "first_idx": 0},
            ...
          }
        }

    `first_idx` 處理上市未滿 N 日 / 中段停牌的股票（只給連續最末段）。
    """
    sids: set[str] = set()
    for rot in rotations or []:
        for h in rot.get("holdings", []) or []:
            sid = h.get("stock_id")
            if sid:
                sids.add(sid)
    for w in watch_block or []:
        sid = w.get("stock_id")
        if sid:
            sids.add(sid)

    if not sids:
        return None

    # 多撈一些緩衝避免假日不夠（1.5 倍）
    earliest = target_date - _dt.timedelta(days=int(days * 1.5) + 14)

    with get_session() as session:
        rows = session.execute(
            select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                DailyPrice.stock_id.in_(list(sids)),
                DailyPrice.date >= earliest,
                DailyPrice.date <= target_date,
            )
        ).all()

    if not rows:
        return None

    # 收集所有出現過的交易日（asc）— 以「實際有資料的日期」為交易日來源
    all_dates = sorted({r[1] for r in rows})
    trading_days = all_dates[-days:]
    if not trading_days:
        return None
    td_index = {d: i for i, d in enumerate(trading_days)}

    by_stock: dict[str, list[tuple[_dt.date, float]]] = {}
    for sid, d, close in rows:
        if d in td_index and close is not None:
            by_stock.setdefault(sid, []).append((d, float(close)))

    series: dict[str, dict] = {}
    for sid in sids:
        pairs = sorted(by_stock.get(sid, []), key=lambda p: p[0])
        if not pairs:
            continue
        # 只取連續最末段（從最新一筆向前找連續）
        # 我們用 trading_days 作為基準逐日比對
        # 找出最早有資料且後續連續的起點：從尾向前掃，遇到「該交易日無資料」即斷
        sid_dates = {p[0] for p in pairs}
        first_idx = len(trading_days)
        for i in range(len(trading_days) - 1, -1, -1):
            if trading_days[i] in sid_dates:
                first_idx = i
            else:
                break
        if first_idx >= len(trading_days):
            continue
        closes = [c for d, c in pairs if d in trading_days[first_idx:]]
        if not closes:
            continue
        series[sid] = {"close": closes, "first_idx": first_idx}

    if not series:
        return None

    return {
        "trading_days": [d.isoformat() for d in trading_days],
        "series": series,
    }


# ---------------------------------------------------------------------------
# 組裝 + 寫檔
# ---------------------------------------------------------------------------


def _build_payload(
    target_date: _dt.date,
    top_n: int,
    event_days: int,
    regenerate_ai: bool,
) -> dict:
    """組裝完整 dashboard payload。子區塊失敗皆 graceful。"""
    bundle = _Bundle()

    # regime
    regime_block: dict = {}
    try:
        regime_block = _build_regime()
    except Exception as exc:
        bundle.record_error("regime", exc)
        regime_block = {"state": None, "summary": ""}

    # discover
    discover_block: dict = {m: [] for m in _MODES}
    try:
        discover_block = _build_discover(target_date, top_n)
    except Exception as exc:
        bundle.record_error("discover", exc)

    # rotations（多組 active 輪動）+ rotation（v1 backward-compat alias = primary）
    rotations_list: list[dict] = []
    rotation_block: dict | None = None
    try:
        rotations_list = _build_rotations()
        rotation_block = rotations_list[0] if rotations_list else None
    except Exception as exc:
        bundle.record_error("rotation", exc)

    # watch_entries
    watch_block: list[dict] = []
    try:
        watch_block = _build_watch_entries()
    except Exception as exc:
        bundle.record_error("watch_entries", exc)

    # IC status — 失敗時視為無資料，不阻擋
    ic_status: list[dict] | None = None
    try:
        from src.cli.morning_cmd import _compute_factor_ic_status

        ic_status, _ = _compute_factor_ic_status()
    except Exception as exc:
        bundle.record_error("ic_status", exc)
        ic_status = None

    # signals
    signals: list[dict] = []
    try:
        signals = _build_signals(regime_block, ic_status)
        stale = _build_data_freshness_signal(target_date)
        if stale:
            signals.append(stale)
    except Exception as exc:
        bundle.record_error("signals", exc)

    # strategy events
    events: list[dict] = []
    try:
        events = _build_strategy_events(event_days)
    except Exception as exc:
        bundle.record_error("strategy_events", exc)

    # AI summary
    ai_summary = None
    try:
        ai_summary = _build_ai_summary(target_date, regenerate_ai)
    except Exception as exc:
        bundle.record_error("ai_summary", exc)

    # portfolio_review（每日績效摘要 — 依賴 RotationDailySnapshot 表）
    portfolio_review = None
    try:
        portfolio_review = _build_portfolio_review(
            rotation_block,
            lookback_days=settings.dashboard.portfolio_review_lookback_days,
        )
    except Exception as exc:
        bundle.record_error("portfolio_review", exc)

    # position_timeseries（持倉/Watch 個股小走勢圖；聚合所有 rotations）
    position_timeseries = None
    try:
        position_timeseries = _build_position_timeseries(
            rotations_list,
            watch_block,
            days=settings.dashboard.position_timeseries_days,
            target_date=target_date,
        )
    except Exception as exc:
        bundle.record_error("position_timeseries", exc)

    # 移除 regime 內部欄位
    regime_clean = {k: v for k, v in regime_block.items() if not k.startswith("_")}

    bundle.payload = {
        "version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "date": target_date.isoformat(),
        "regime": regime_clean,
        "discover": discover_block,
        "rotation": rotation_block,
        "rotations": rotations_list,
        "watch_entries": watch_block,
        "signals": signals,
        "strategy_events": events,
        "ai_summary": ai_summary,
        "portfolio_review": portfolio_review,
        "position_timeseries": position_timeseries,
        "errors": bundle.errors,
    }
    return bundle.payload


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")


def _write_atomic(path: Path, content: str) -> None:
    """先寫 .tmp 再 rename，避免 iCloud 同步看到半截檔。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _write_payload(payload: dict, out_dir: Path, target_date: _dt.date) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_date = _SAFE_NAME.sub("_", target_date.isoformat())
    dated = out_dir / f"{safe_date}.json"
    latest = out_dir / "latest.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    _write_atomic(dated, text)
    _write_atomic(latest, text)
    return dated, latest


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def cmd_export_dashboard(args: argparse.Namespace) -> None:
    """Dashboard JSON 匯出 CLI 入口。"""
    init_db()

    date_str = getattr(args, "date", None)
    target_date = _dt.date.fromisoformat(date_str) if date_str else _dt.date.today()
    top_n: int = getattr(args, "top", DEFAULT_TOP_N)
    event_days: int = getattr(args, "event_days", DEFAULT_EVENT_DAYS)
    regenerate_ai: bool = getattr(args, "regenerate_ai_summary", False)

    out_arg = getattr(args, "out", None)
    out_dir = Path(out_arg).expanduser() if out_arg else _DEFAULT_OUT_DIR

    payload = _build_payload(target_date, top_n, event_days, regenerate_ai)
    dated, latest = _write_payload(payload, out_dir, target_date)

    n_disc = sum(len(v) for v in payload["discover"].values())
    rotations_payload = payload.get("rotations") or []
    n_rotations = len(rotations_payload)
    n_holdings_total = sum(len(r.get("holdings", []) or []) for r in rotations_payload)
    primary_name = rotations_payload[0].get("name") if rotations_payload else None
    n_watch = len(payload["watch_entries"])
    n_signals = len(payload["signals"])
    n_events = len(payload["strategy_events"])
    n_errors = len(payload["errors"])
    n_snapshots = (payload.get("portfolio_review") or {}).get("snapshots_count", 0)
    n_pts_series = len((payload.get("position_timeseries") or {}).get("series", {}) or {})

    print(f"Dashboard JSON 已寫出：{dated}")
    print(f"  latest 同步：{latest}")
    rotations_summary = (
        f"rotations={n_rotations}（primary={primary_name}, holdings_total={n_holdings_total}）"
        if n_rotations
        else "rotations=0"
    )
    print(
        f"  discover={n_disc} | {rotations_summary} | "
        f"watch={n_watch} | signals={n_signals} | events={n_events} | errors={n_errors}"
    )
    print(f"  portfolio_review.snapshots={n_snapshots} | position_timeseries.series={n_pts_series}")
    if n_errors:
        print(f"  ⚠ {n_errors} 個區塊產出失敗（payload.errors[] 內含詳情）")
