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
from src.data.database import get_session
from src.data.schema import DiscoveryRecord, WatchEntry

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
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


def _build_rotation() -> dict | None:
    """取主要（current_capital 最大的 active）輪動組合的快照。"""
    from src.portfolio.manager import RotationManager

    portfolios = RotationManager.list_portfolios()
    actives = [p for p in portfolios if p.get("status") == "active"]
    if not actives:
        return None

    primary = max(actives, key=lambda p: p.get("current_capital", 0.0) or 0.0)
    mgr = RotationManager(primary["name"])
    status = mgr.get_status()
    if status is None:
        return None

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

    return {
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

    from src.data.schema import DailyPrice

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

    # rotation
    rotation_block = None
    try:
        rotation_block = _build_rotation()
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

    # 移除 regime 內部欄位
    regime_clean = {k: v for k, v in regime_block.items() if not k.startswith("_")}

    bundle.payload = {
        "version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "date": target_date.isoformat(),
        "regime": regime_clean,
        "discover": discover_block,
        "rotation": rotation_block,
        "watch_entries": watch_block,
        "signals": signals,
        "strategy_events": events,
        "ai_summary": ai_summary,
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
    n_holdings = len((payload.get("rotation") or {}).get("holdings", []))
    n_watch = len(payload["watch_entries"])
    n_signals = len(payload["signals"])
    n_events = len(payload["strategy_events"])
    n_errors = len(payload["errors"])

    print(f"Dashboard JSON 已寫出：{dated}")
    print(f"  latest 同步：{latest}")
    print(
        f"  discover={n_disc} | rotation_holdings={n_holdings} | "
        f"watch={n_watch} | signals={n_signals} | events={n_events} | errors={n_errors}"
    )
    if n_errors:
        print(f"  ⚠ {n_errors} 個區塊產出失敗（payload.errors[] 內含詳情）")
