"""MOPS 公開資訊觀測站重大訊息抓取器。

抓取每日重大訊息公告，用於 discover 消息面評分。

資料來源：
- MOPS 公開資訊觀測站: https://mops.twse.com.tw
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, timedelta

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

# MOPS 的 SSL 憑證在部分系統會驗證失敗，停用警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

# 請求間隔（秒），避免被封鎖
_REQUEST_DELAY = 3

# --- 情緒分類關鍵字 --- #

_POSITIVE_KEYWORDS = [
    "庫藏股", "買回", "營收創", "合併", "收購", "取得專利", "獲獎",
    "上櫃轉上市", "上市", "策略聯盟", "訂單", "中標", "得標",
    "股利", "配息", "增資", "現金增資",
]

_NEGATIVE_KEYWORDS = [
    "下市", "終止上市", "終止上櫃", "重編", "虧損", "違約", "裁罰",
    "減資彌補", "財務危機", "跳票", "解散", "清算", "撤銷",
    "停工", "停產", "訴訟", "求償", "處分", "警示",
]


def classify_sentiment(subject: str) -> int:
    """依公告主旨關鍵字分類情緒。

    Args:
        subject: 公告主旨文字

    Returns:
        +1（正面）、-1（負面）、0（中性）
    """
    if not subject:
        return 0

    for kw in _NEGATIVE_KEYWORDS:
        if kw in subject:
            return -1

    for kw in _POSITIVE_KEYWORDS:
        if kw in subject:
            return 1

    return 0


def _find_last_trading_day(target: date, max_lookback: int = 7) -> date:
    """從 target 往前找最近的交易日（跳過週末）。"""
    d = target
    for _ in range(max_lookback):
        if d.weekday() < 5:
            return d
        d -= timedelta(days=1)
    return target


def fetch_mops_announcements(target_date: date | None = None) -> pd.DataFrame:
    """抓取 MOPS 某日全市場重大訊息公告。

    使用 MOPS ajax_t05sr01_1 端點，一次查詢取得上市 + 上櫃所有重訊。

    Args:
        target_date: 目標日期，預設為最近交易日

    Returns:
        DataFrame 欄位: date, stock_id, seq, subject, spoke_time, sentiment
    """
    if target_date is None:
        target_date = _find_last_trading_day(date.today())

    # MOPS 使用民國年
    roc_year = target_date.year - 1911
    date_str = f"{roc_year}/{target_date.month:02d}/{target_date.day:02d}"

    url = "https://mops.twse.com.tw/mops/web/ajax_t05sr01_1"

    logger.info("抓取 MOPS 重大訊息: %s", target_date.isoformat())

    # 嘗試上市 (sii) 和上櫃 (otc) 兩種市場
    all_rows: list[dict] = []

    for typek in ("sii", "otc"):
        form_data = {
            "encodeURIComponent": "1",
            "step": "1",
            "firstin": "1",
            "off": "1",
            "TYPEK": typek,
            "year": str(roc_year),
            "month": f"{target_date.month:02d}",
            "day": f"{target_date.day:02d}",
        }

        try:
            resp = requests.post(
                url,
                data=form_data,
                headers=_HEADERS,
                timeout=30,
                verify=False,
            )
            resp.raise_for_status()
            resp.encoding = "utf-8"
        except Exception as e:
            logger.error("MOPS 重訊請求失敗 (%s): %s", typek, e)
            time.sleep(_REQUEST_DELAY)
            continue

        rows = _parse_announcement_html(resp.text, target_date)
        all_rows.extend(rows)
        logger.info("MOPS %s 重訊: %d 筆", typek.upper(), len(rows))

        time.sleep(_REQUEST_DELAY)

    if not all_rows:
        logger.info("MOPS 重訊: 當日無公告 (%s)", target_date.isoformat())
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info("MOPS 重訊合計: %d 筆", len(df))
    return df


def _parse_announcement_html(html: str, target_date: date) -> list[dict]:
    """解析 MOPS 重訊回傳的 HTML 表格。

    Args:
        html: MOPS 回傳的 HTML 內容
        target_date: 查詢日期

    Returns:
        解析後的公告列表
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    # MOPS 重訊頁面以 <table class="hasBorder"> 呈現
    tables = soup.find_all("table", class_="hasBorder")
    if not tables:
        return rows

    for table in tables:
        trs = table.find_all("tr")
        for tr in trs:
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue

            # 欄位順序：公司代號、公司名稱、發言日期、發言時間、主旨
            # 實際欄位可能因頁面版本不同，需要彈性處理
            texts = [td.get_text(strip=True) for td in tds]

            # 找到包含股票代號的列（4 位數字開頭）
            stock_id = texts[0].strip()
            if not re.match(r"^\d{4}", stock_id):
                continue

            # 取得主要欄位
            seq = texts[2].strip() if len(texts) > 2 else "1"  # 序號
            spoke_time = texts[3].strip() if len(texts) > 3 else ""
            subject = texts[4].strip() if len(texts) > 4 else ""

            if not subject:
                # 若第 5 欄無資料，嘗試最後一欄
                subject = texts[-1].strip() if texts[-1].strip() else ""

            if not subject:
                continue

            rows.append(
                {
                    "date": target_date,
                    "stock_id": stock_id,
                    "seq": seq,
                    "subject": subject,
                    "spoke_time": spoke_time,
                    "sentiment": classify_sentiment(subject),
                }
            )

    return rows
