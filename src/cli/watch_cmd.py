"""CLI 持倉監控子命令 — watch add/list/close/update-status + 純函數。"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from src.cli.helpers import init_db
from src.cli.helpers import safe_print as print
from src.entry_exit import assess_timing, compute_atr_stops, compute_entry_trigger


def _compute_watch_status(
    entry_price: float,  # noqa: ARG001
    stop_loss: float | None,
    take_profit: float | None,
    valid_until: date | None,
    latest_price: float | None,
    today: date,
) -> str:
    """根據最新價格與到期日計算持倉狀態（純函數，不寫 DB）。

    優先級：止損 > 止利 > 過期 > active

    Args:
        entry_price: 進場價（保留供未來擴充，目前未使用）
        stop_loss:   止損價，None 表示未設定
        take_profit: 目標價，None 表示未設定
        valid_until: 有效期限，None 表示永不過期
        latest_price:最新收盤價，None 表示查無資料
        today:       今日日期

    Returns:
        "stopped_loss" | "taken_profit" | "expired" | "active"
    """
    if latest_price is not None:
        if stop_loss is not None and latest_price <= stop_loss:
            return "stopped_loss"
        if take_profit is not None and latest_price >= take_profit:
            return "taken_profit"
    if valid_until is not None and today > valid_until:
        return "expired"
    return "active"


def _compute_trailing_stop(highest_price: float, atr14: float, multiplier: float) -> float:
    """計算移動止損價（純函數）。

    公式：stop = highest_price - atr14 * multiplier
    僅在外部確認 new_stop > current_stop 時才更新（只升不降）。

    Args:
        highest_price: 進場後追蹤的最高收盤價
        atr14:        最近 14 日平均真實波幅
        multiplier:   ATR 倍數（如 1.5）

    Returns:
        移動止損價，四捨五入至小數點後兩位。
    """
    return round(highest_price - atr14 * multiplier, 2)


# 向後相容別名：assess_timing 已遷移至 src.entry_exit
_assess_timing = assess_timing


def _watch_add(args: argparse.Namespace) -> None:
    """watch add：新增持倉監控記錄。"""
    import datetime

    import pandas as pd
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import DailyPrice, DiscoveryRecord, StockInfo, WatchEntry
    from src.discovery.scanner import _calc_atr14

    stock_id: str = args.stock_id
    today = datetime.date.today()

    with get_session() as session:
        stock_name_row = session.execute(select(StockInfo.stock_name).where(StockInfo.stock_id == stock_id)).scalar()
    stock_name: str = stock_name_row or stock_id

    if args.from_discover:
        mode_src: str = args.from_discover
        with get_session() as session:
            rec = (
                session.execute(
                    select(DiscoveryRecord)
                    .where(DiscoveryRecord.stock_id == stock_id)
                    .where(DiscoveryRecord.mode == mode_src)
                    .order_by(DiscoveryRecord.scan_date.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
        if rec is None:
            print(f"錯誤：找不到 {stock_id} 在 {mode_src} 模式的推薦記錄")
            return
        entry_price_val = float(args.price) if args.price else (rec.entry_price or rec.close)
        stop_loss_val = float(args.stop) if args.stop else rec.stop_loss
        take_profit_val = float(args.target) if args.target else rec.take_profit
        entry_trigger_val = rec.entry_trigger
        valid_until_val = rec.valid_until
        source_val = "discover"
        mode_val: str | None = mode_src

    else:
        with get_session() as session:
            rows = (
                session.execute(
                    select(DailyPrice).where(DailyPrice.stock_id == stock_id).order_by(DailyPrice.date.desc()).limit(30)
                )
                .scalars()
                .all()
            )
        if not rows:
            print(f"錯誤：找不到 {stock_id} 的日K線資料（請先執行 sync）")
            return

        df = (
            pd.DataFrame(
                [
                    {
                        "date": r.date,
                        "open": float(r.open),
                        "high": float(r.high),
                        "low": float(r.low),
                        "close": float(r.close),
                        "volume": float(r.volume),
                    }
                    for r in reversed(rows)
                ]
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

        close = float(df["close"].iloc[-1])
        atr14 = _calc_atr14(df)
        sma20 = float(df["close"].tail(20).mean()) if len(df) >= 20 else 0.0
        atr_pct = atr14 / close if close > 0 else 0.0

        # Regime 偵測（與 suggest 一致）
        try:
            from src.regime.detector import MarketRegimeDetector

            _regime: str = MarketRegimeDetector().detect()["regime"]
        except (KeyError, ValueError, TypeError) as exc:
            logging.debug("Regime 偵測失敗，使用 sideways: %s", exc)
            _regime = "sideways"

        entry_price_val = round(float(args.price), 2) if args.price else round(close, 2)
        _auto_sl, _auto_tp = compute_atr_stops(entry_price_val, atr14, _regime)
        stop_loss_val = round(float(args.stop), 2) if args.stop else _auto_sl
        take_profit_val = round(float(args.target), 2) if args.target else _auto_tp

        entry_trigger_val: str | None = compute_entry_trigger(close, sma20, atr_pct, _regime)

        valid_until_val = (pd.Timestamp(today) + pd.offsets.BDay(5)).date()
        source_val = "manual"
        mode_val = None

    # ── 移動止損參數 ────────────────────────────────────────────────
    trailing_enabled: bool = getattr(args, "trailing", False)
    trailing_mult: float = float(getattr(args, "trailing_multiplier", 1.5) or 1.5)

    entry = WatchEntry(
        stock_id=stock_id,
        stock_name=stock_name,
        entry_date=today,
        entry_price=entry_price_val,
        stop_loss=stop_loss_val,
        take_profit=take_profit_val,
        quantity=int(args.qty) if args.qty else None,
        source=source_val,
        mode=mode_val,
        entry_trigger=entry_trigger_val,
        valid_until=valid_until_val,
        status="active",
        notes=args.notes or None,
        trailing_stop_enabled=trailing_enabled,
        trailing_atr_multiplier=trailing_mult if trailing_enabled else None,
        highest_price_since_entry=entry_price_val if trailing_enabled else None,
    )

    with get_session() as session:
        session.add(entry)
        session.commit()
        entry_id = entry.id

    sl_str = f"{stop_loss_val:.2f}" if stop_loss_val else "—"
    tp_str = f"{take_profit_val:.2f}" if take_profit_val else "—"
    trailing_str = f"  [移動止損 ×{trailing_mult}]" if trailing_enabled else ""
    print(f"\n已加入持倉監控 #{entry_id}：{stock_id} {stock_name}{trailing_str}")
    print(f"  進場價：{entry_price_val:.2f}  止損：{sl_str}  目標：{tp_str}")
    if valid_until_val:
        print(f"  有效至：{valid_until_val}  來源：{source_val}")
    print()


def _watch_list(args: argparse.Namespace) -> None:
    """watch list：列出持倉記錄。"""
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import WatchEntry

    status_filter: str = args.status
    with get_session() as session:
        q = select(WatchEntry).order_by(WatchEntry.entry_date.desc())
        if status_filter != "all":
            q = q.where(WatchEntry.status == status_filter)
        entries = session.execute(q).scalars().all()

    if not entries:
        label = "任何" if status_filter == "all" else status_filter
        print(f"目前沒有 {label} 狀態的持倉記錄。使用 `watch add <stock_id>` 新增。")
        return

    STATUS_ZH = {
        "active": "🟢 持倉中",
        "stopped_loss": "🔴 止損",
        "taken_profit": "🟡 止利",
        "expired": "⚫ 過期",
        "closed": "⚪ 已平倉",
    }

    print(
        f"\n{'ID':>4}  {'代號':<8} {'名稱':<12} {'進場日':<12} {'進場價':>8} {'止損':>8} {'目標':>8}  {'類型':<6}  {'狀態'}"
    )
    print("─" * 83)
    for e in entries:
        sl_s = f"{e.stop_loss:.2f}" if e.stop_loss else "  —"
        tp_s = f"{e.take_profit:.2f}" if e.take_profit else "  —"
        st_s = STATUS_ZH.get(e.status, e.status)
        name_s = (e.stock_name or "")[:10]
        # 移動止損標記（[T] = Trailing）
        type_s = f"[T×{e.trailing_atr_multiplier:.1f}]" if e.trailing_stop_enabled else "靜態"
        print(
            f"{e.id:>4}  {e.stock_id:<8} {name_s:<12} {str(e.entry_date):<12} {e.entry_price:>8.2f} {sl_s:>8} {tp_s:>8}  {type_s:<6}  {st_s}"
        )
    print()


def _watch_close(args: argparse.Namespace) -> None:
    """watch close：平倉持倉記錄。"""
    import datetime

    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import WatchEntry

    entry_id_arg: int = args.entry_id
    close_price_arg: float | None = float(args.price) if args.price else None
    close_date_today = datetime.date.today()

    with get_session() as session:
        entry_obj = session.execute(select(WatchEntry).where(WatchEntry.id == entry_id_arg)).scalars().first()
        if entry_obj is None:
            print(f"錯誤：找不到 ID={entry_id_arg} 的持倉記錄")
            return
        entry_obj.status = "closed"
        entry_obj.close_date = close_date_today
        entry_obj.close_price = close_price_arg
        session.commit()

    pnl_str = ""
    if close_price_arg and entry_obj.entry_price:
        pnl_pct = (close_price_arg - entry_obj.entry_price) / entry_obj.entry_price * 100
        pnl_str = f"  損益：{pnl_pct:+.2f}%"
    print(f"\n已平倉 #{entry_id_arg} {entry_obj.stock_id}（{close_date_today}）{pnl_str}\n")


def _watch_update_status() -> None:
    """watch update-status：批次更新止損/止利/過期狀態（含移動止損）。

    對 trailing_stop_enabled=True 的持倉，先依最新收盤價更新
    highest_price_since_entry 與 stop_loss（只升不降），
    再統一檢查止損/止利/過期狀態。
    """
    import datetime

    import pandas as pd
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import DailyPrice, WatchEntry
    from src.discovery.scanner import _calc_atr14

    today = datetime.date.today()

    with get_session() as session:
        active_entries = session.execute(select(WatchEntry).where(WatchEntry.status == "active")).scalars().all()

    if not active_entries:
        print("目前沒有 active 持倉，無需更新。")
        return

    stock_ids = list({e.stock_id for e in active_entries})

    # ── 1. 取得最新收盤價 ─────────────────────────────────────────────
    latest_prices: dict[str, float] = {}
    with get_session() as session:
        for sid in stock_ids:
            row = session.execute(
                select(DailyPrice.close).where(DailyPrice.stock_id == sid).order_by(DailyPrice.date.desc()).limit(1)
            ).scalar()
            if row is not None:
                latest_prices[sid] = float(row)

    # ── 2. 預先計算有移動止損的股票 ATR14 ────────────────────────────
    trailing_ids = {e.stock_id for e in active_entries if e.trailing_stop_enabled}
    atr14_cache: dict[str, float] = {}
    if trailing_ids:
        with get_session() as session:
            for sid in trailing_ids:
                rows = (
                    session.execute(
                        select(DailyPrice).where(DailyPrice.stock_id == sid).order_by(DailyPrice.date.desc()).limit(30)
                    )
                    .scalars()
                    .all()
                )
                if rows:
                    df = (
                        pd.DataFrame(
                            [
                                {
                                    "date": r.date,
                                    "high": float(r.high),
                                    "low": float(r.low),
                                    "close": float(r.close),
                                }
                                for r in reversed(rows)
                            ]
                        )
                        .sort_values("date")
                        .reset_index(drop=True)
                    )
                    atr14_cache[sid] = _calc_atr14(df)

    # ── 3. 批次更新（移動止損 → 狀態判斷）────────────────────────────
    updated = 0
    trailing_updated = 0
    with get_session() as session:
        for e in active_entries:
            latest_price = latest_prices.get(e.stock_id)
            effective_stop = e.stop_loss  # 預設使用目前止損

            # ── 移動止損更新（只升不降）──────────────────────────
            if e.trailing_stop_enabled and latest_price is not None:
                atr14 = atr14_cache.get(e.stock_id, 0.0)
                if atr14 > 0:
                    mult = e.trailing_atr_multiplier or 1.5
                    curr_highest = e.highest_price_since_entry or e.entry_price
                    new_highest = max(curr_highest, latest_price)
                    new_stop = _compute_trailing_stop(new_highest, atr14, mult)

                    # 只在移動止損往上移動時才更新
                    if new_stop > (effective_stop or 0.0):
                        obj = session.execute(select(WatchEntry).where(WatchEntry.id == e.id)).scalars().first()
                        if obj:
                            obj.highest_price_since_entry = new_highest
                            obj.stop_loss = new_stop
                            effective_stop = new_stop
                            trailing_updated += 1

            # ── 狀態判斷（使用更新後的 effective_stop）───────────
            new_status = _compute_watch_status(
                entry_price=e.entry_price,
                stop_loss=effective_stop,
                take_profit=e.take_profit,
                valid_until=e.valid_until,
                latest_price=latest_price,
                today=today,
            )
            if new_status != "active":
                obj = session.execute(select(WatchEntry).where(WatchEntry.id == e.id)).scalars().first()
                if obj:
                    obj.status = new_status
                    updated += 1
        session.commit()

    trailing_msg = f"移動止損更新 {trailing_updated} 筆，" if trailing_ids else ""
    print(f"\n更新完成：{len(active_entries)} 筆持倉，{trailing_msg}觸發狀態變更 {updated} 筆。\n")


def cmd_watch(args: argparse.Namespace) -> None:
    """持倉監控管理（add / list / close / update-status）。"""
    init_db()
    action: str = args.action
    if action == "add":
        _watch_add(args)
    elif action == "list":
        _watch_list(args)
    elif action == "close":
        _watch_close(args)
    elif action == "update-status":
        _watch_update_status()
    else:
        print(f"未知動作：{action}。可用動作：add / list / close / update-status")
