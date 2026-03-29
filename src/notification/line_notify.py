"""Discord Webhook 通知模組 — 發送訊號通知到 Discord 頻道。

使用方式：
    1. 在 Discord 頻道設定 → 整合 → Webhook → 建立 Webhook
    2. 複製 Webhook URL 填入 config/settings.yaml 的 discord.webhook_url
    3. 使用 send_message() 發送文字訊息

NOTE: 檔名保留為 line_notify.py 以維持向後相容的 import 路徑，
      實際實作已改為 Discord Webhook。
"""

from __future__ import annotations

import logging

import requests

from src.config import settings

logger = logging.getLogger(__name__)


def send_message(text: str) -> bool:
    """發送文字訊息到 Discord Webhook。

    Args:
        text: 訊息內容（Discord 上限 2000 字元）

    Returns:
        True 表示發送成功，False 表示失敗
    """
    webhook_url = settings.discord.webhook_url
    if not webhook_url:
        logger.warning("Discord Webhook URL 未設定，跳過通知")
        return False

    if not settings.discord.enabled:
        logger.info("Discord 通知已停用，跳過通知")
        return False

    payload = {"content": text[:2000]}

    # 可選：設定 Bot 顯示名稱和頭像
    if settings.discord.username:
        payload["username"] = settings.discord.username

    try:
        from src.data.retry import request_with_retry

        resp = request_with_retry(
            "POST",
            webhook_url,
            json=payload,
            timeout=10,
            max_retries=2,
            base_delay=1.0,
        )
        if resp.status_code in (200, 204):
            logger.info("Discord 通知發送成功")
            return True
        logger.error("Discord 通知發送失敗: HTTP %d — %s", resp.status_code, resp.text)
        return False
    except requests.RequestException:
        logger.exception("Discord 通知發送失敗（重試後仍失敗）")
        return False


def format_scan_results(df) -> str:
    """將篩選結果 DataFrame 格式化為 Discord 訊息。

    Args:
        df: 篩選結果 DataFrame，需包含 stock_id, close, factor_score 等欄位

    Returns:
        格式化後的文字訊息（使用 Discord Markdown）
    """
    import pandas as pd

    if df is None or df.empty:
        return "**📊 選股篩選結果：**無符合條件的股票"

    lines = ["**📊 選股篩選結果**"]
    lines.append(f"掃描時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"符合條件：{len(df)} 檔")
    lines.append("```")

    for _, row in df.head(10).iterrows():
        stock_id = row.get("stock_id", "?")
        close = row.get("close", 0)
        score = row.get("factor_score", 0)
        parts = [f"{stock_id} | 收盤 {close:.1f} | 分數 {score:.2f}"]

        details = []
        if "rsi_14" in row and pd.notna(row["rsi_14"]):
            details.append(f"RSI={row['rsi_14']:.1f}")
        if "foreign_net" in row and pd.notna(row["foreign_net"]):
            net = row["foreign_net"]
            details.append(f"外資={'買' if net > 0 else '賣'}{abs(net):,.0f}")
        if "yoy_growth" in row and pd.notna(row["yoy_growth"]):
            details.append(f"YoY={row['yoy_growth']:.1f}%")
        if details:
            parts.append(f"  ({', '.join(details)})")

        lines.append("  ".join(parts))

    lines.append("```")

    if len(df) > 10:
        lines.append(f"...及其他 {len(df) - 10} 檔")

    return "\n".join(lines)


def send_scan_results(df) -> bool:
    """格式化篩選結果並發送到 Discord。

    Args:
        df: 篩選結果 DataFrame

    Returns:
        True 表示發送成功
    """
    text = format_scan_results(df)
    return send_message(text)


def format_suggest_discord(
    stock_id: str,
    stock_name: str,
    today: object,
    close: float,
    sma20: float,
    rsi14: float,
    atr_str: str,
    regime_zh: str,
    taiex_close: float,
    entry_price: float,
    sl_str: str,
    tp_str: str,
    rr_str: str,
    trigger: str,
    timing: str,
    valid_until: object,
) -> str:
    """將 suggest 結果格式化為 Discord 訊息（純函數，≤ 2000 字元）。"""
    sep = "─" * 40
    lines = [
        f"**進出場建議 — {stock_id} {stock_name}**",
        f"分析日期：{today}  ｜  市場：{regime_zh}（TAIEX {taiex_close:,.0f}）",
        "```",
        f"收盤  ：{close:.2f}   SMA20：{sma20:.2f}   RSI：{rsi14:.1f}",
        f"ATR14 ：{atr_str}",
        sep,
        f"進場參考：{entry_price:.2f}",
        f"止  損  ：{sl_str}",
        f"目  標  ：{tp_str}",
        f"風險報酬：{rr_str}",
        sep,
        f"觸發條件：{trigger}",
        f"時機評估：{timing}",
        f"有效至  ：{valid_until}",
        "```",
    ]
    return "\n".join(lines)[:2000]
