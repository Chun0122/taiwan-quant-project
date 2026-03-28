"""CLI 單股進出場建議子命令 — suggest。"""

from __future__ import annotations

import argparse
import logging
import sys

from src.cli.helpers import init_db
from src.cli.helpers import safe_print as print
from src.entry_exit import assess_timing, compute_atr_stops, compute_entry_trigger
from src.features.indicators import calc_rsi14_from_series as _calc_rsi14_from_series
from src.notification.line_notify import format_suggest_discord as _format_suggest_discord


def cmd_suggest(args: argparse.Namespace) -> None:
    """單股進出場建議。

    從 DB 讀取最近 60 日日K線，計算 ATR14 / SMA20 / RSI14，
    偵測市場 Regime，輸出進場區間、止損、目標價與時機評估。
    """
    import datetime

    import pandas as pd
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import DailyPrice, StockInfo
    from src.discovery.scanner import _calc_atr14
    from src.regime.detector import MarketRegimeDetector

    stock_id: str = args.stock_id
    init_db()

    # ── 1. 從 DB 載入最近 60 日日K（倒序取 60 筆，再反轉為升序）──────
    with get_session() as session:
        rows = (
            session.execute(
                select(DailyPrice).where(DailyPrice.stock_id == stock_id).order_by(DailyPrice.date.desc()).limit(60)
            )
            .scalars()
            .all()
        )

        info_row = session.execute(select(StockInfo.stock_name).where(StockInfo.stock_id == stock_id)).scalar()

    if not rows:
        print(f"錯誤：找不到股票 {stock_id} 的日K線資料（請先執行 sync）")
        sys.exit(1)

    stock_name: str = info_row or stock_id

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

    # ── 2. 計算 ATR14、SMA20、RSI14 ────────────────────────────────
    atr14 = _calc_atr14(df)
    close = float(df["close"].iloc[-1])
    # 資料不足 20 天時給 0.0，使後續 trigger/timing 判斷落入「均線下方」分支
    sma20 = float(df["close"].tail(20).mean()) if len(df) >= 20 else 0.0
    rsi14 = _calc_rsi14_from_series(df["close"])

    # ── 3. 偵測市場 Regime ─────────────────────────────────────────
    try:
        regime_info = MarketRegimeDetector().detect()
        regime: str = regime_info["regime"]
        taiex_close: float = float(regime_info["taiex_close"])
    except (KeyError, ValueError, TypeError) as exc:
        logging.debug("Regime 偵測失敗，使用 sideways: %s", exc)
        regime = "sideways"
        taiex_close = 0.0

    # ── 4. 計算進出場數字 ──────────────────────────────────────────
    today = datetime.date.today()
    valid_until = (pd.Timestamp(today) + pd.offsets.BDay(5)).date()

    entry_price = round(close, 2)
    atr_pct = atr14 / close if close > 0 else 0.0

    stop_loss, take_profit = compute_atr_stops(close, atr14, regime)

    if stop_loss is not None and take_profit is not None:
        risk_pct = (entry_price - stop_loss) / entry_price * 100
        reward_pct = (take_profit - entry_price) / entry_price * 100
        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0.0
        sl_str = f"{stop_loss:.2f}（-{risk_pct:.1f}%）"
        tp_str = f"{take_profit:.2f}（+{reward_pct:.1f}%）"
        rr_str = f"1 : {rr_ratio:.1f}"
        atr_str = f"{atr14:.2f}（{atr_pct:.1%}）"
    else:
        stop_loss = None
        take_profit = None
        sl_str = "—"
        tp_str = "—"
        rr_str = "—"
        atr_str = "—"

    # ── 5. 計算 entry_trigger（共用純函數，與 scanner.py 一致）────
    trigger = compute_entry_trigger(close, sma20, atr_pct, regime)

    # ── 6. 時機評估 ────────────────────────────────────────────────
    timing = assess_timing(rsi14, close, sma20, atr_pct, regime)

    # ── 7. 輸出 CLI ────────────────────────────────────────────────
    regime_zh = {"bull": "多頭", "bear": "空頭", "sideways": "盤整", "crisis": "崩盤"}.get(regime, regime)
    taiex_str = f"TAIEX {taiex_close:,.0f}" if taiex_close > 0 else ""

    sep60 = "═" * 60
    sep_thin = "─" * 60
    print(f"\n{sep60}")
    print(f"  單股進出場建議  ｜  {stock_id} {stock_name}")
    print(sep60)
    print(f"  分析日期  ：{today}")
    print(f"  最新收盤  ：{close:.2f}")
    print(f"  SMA20    ：{sma20:.2f}")
    print(f"  RSI14    ：{rsi14:.1f}")
    print(f"  ATR14    ：{atr_str}")
    print(sep_thin)
    taiex_part = f"（{taiex_str}）" if taiex_str else ""
    print(f"  市場 Regime ：{regime_zh}{taiex_part}")
    print(sep_thin)
    print(f"  進場參考價  ：{entry_price:.2f}")
    print(f"  止 損 價   ：{sl_str}")
    print(f"  目 標 價   ：{tp_str}")
    print(f"  風 險 報 酬 ：{rr_str}")
    print(sep_thin)
    print(f"  進場觸發  ：{trigger}")
    print(f"  時機評估  ：{timing}")
    print(f"  建議有效至：{valid_until}")
    print(f"{sep60}\n")

    # ── 8. Discord 通知（--notify）────────────────────────────────
    if args.notify:
        from src.notification.line_notify import send_message

        msg = _format_suggest_discord(
            stock_id=stock_id,
            stock_name=stock_name,
            today=today,
            close=close,
            sma20=sma20,
            rsi14=rsi14,
            atr_str=atr_str,
            regime_zh=regime_zh,
            taiex_close=taiex_close,
            entry_price=entry_price,
            sl_str=sl_str,
            tp_str=tp_str,
            rr_str=rr_str,
            trigger=trigger,
            timing=timing,
            valid_until=valid_until,
        )
        ok = send_message(msg)
        print("Discord 通知已發送" if ok else "Discord 通知失敗")
