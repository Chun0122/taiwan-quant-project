"""MOPS 公開資訊觀測站抓取器。

功能：
- 每日重大訊息公告（消息面評分）
- 全市場月營收（GrowthScanner 粗篩）

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
    "庫藏股",
    "買回",
    "營收創",
    "合併",
    "收購",
    "取得專利",
    "獲獎",
    "上櫃轉上市",
    "上市",
    "策略聯盟",
    "訂單",
    "中標",
    "得標",
    "股利",
    "配息",
    "盈餘轉增資",
    "處分不動產",
    "處分有價證券",
    "處分資產",
]

_NEGATIVE_KEYWORDS = [
    "下市",
    "終止上市",
    "終止上櫃",
    "重編",
    "虧損",
    "違約",
    "裁罰",
    "減資彌補",
    "財務危機",
    "跳票",
    "解散",
    "清算",
    "撤銷",
    "停工",
    "停產",
    "訴訟",
    "求償",
    "罰鍰",
    "糾正",
    "現金增資",
    "警示",
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

    # 「澄清媒體報導」通常為中性（依法回應媒體傳言），不應被內文關鍵字誤判
    if "澄清" in subject or "說明媒體" in subject:
        return 0

    # 負面優先比對（負面關鍵字多為具體搭配如「終止上市」「現金增資」）
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
    """抓取 MOPS 最新全市場重大訊息公告。

    使用 MOPS 備援站 ajax_t05sr01_1 端點，一次查詢取得上市 + 上櫃所有重訊。

    注意：MOPS 備援站僅返回最新一個交易日的公告，target_date 參數不影響
    實際回傳內容。實際公告日期從 HTML 中的「發言日期」欄位解析。

    Args:
        target_date: 未使用（保留參數以維持向後相容），實際日期從 HTML 解析

    Returns:
        DataFrame 欄位: date, stock_id, seq, subject, spoke_time, sentiment
    """
    if target_date is None:
        target_date = _find_last_trading_day(date.today())

    # MOPS 使用民國年（參數不影響結果，但仍需傳入）
    roc_year = target_date.year - 1911

    # 使用 mopsov 備援站（主站 mops.twse.com.tw 會擋自動化請求）
    url = "https://mopsov.twse.com.tw/mops/web/ajax_t05sr01_1"

    logger.info("抓取 MOPS 最新重大訊息")

    # 嘗試上市 (sii) 和上櫃 (otc) 兩種市場
    all_rows: list[dict] = []

    for typek in ("sii", "otc"):
        form_data = {
            "encodeURIComponent": "1",
            "step": "0",
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

        rows = _parse_announcement_html(resp.text)
        all_rows.extend(rows)
        logger.info("MOPS %s 重訊: %d 筆", typek.upper(), len(rows))

        time.sleep(_REQUEST_DELAY)

    if not all_rows:
        logger.info("MOPS 重訊: 當日無公告")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    actual_date = df["date"].iloc[0] if not df.empty else "N/A"
    logger.info("MOPS 重訊合計: %d 筆（公告日期: %s）", len(df), actual_date)
    return df


def _parse_roc_date(roc_date_str: str) -> date | None:
    """將民國日期字串（如 '115/02/26'）轉為 date 物件。"""
    m = re.match(r"(\d{2,3})/(\d{2})/(\d{2})", roc_date_str)
    if not m:
        return None
    year = int(m.group(1)) + 1911
    month = int(m.group(2))
    day = int(m.group(3))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_announcement_html(html: str) -> list[dict]:
    """解析 MOPS 重訊回傳的 HTML 表格。

    從 HTML 中的「發言日期」欄位解析實際公告日期。

    Args:
        html: MOPS 回傳的 HTML 內容

    Returns:
        解析後的公告列表
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    # MOPS 重訊頁面以 <table class="hasBorder"> 呈現
    tables = soup.find_all("table", class_="hasBorder")
    if not tables:
        return rows

    # 用序號計數器為同一 stock_id 同一天產生唯一 seq
    seq_counter: dict[str, int] = {}

    for table in tables:
        trs = table.find_all("tr")
        for tr in trs:
            tds = tr.find_all("td")
            if len(tds) < 5:
                continue

            # 欄位順序：[0]公司代號 [1]公司簡稱 [2]發言日期 [3]發言時間 [4]主旨 [5](空)
            texts = [td.get_text(strip=True) for td in tds]

            # 找到包含股票代號的列（4~6 位數字開頭）
            stock_id = texts[0].strip()
            if not re.match(r"^\d{4,6}$", stock_id):
                continue

            # 從「發言日期」欄位解析實際日期
            actual_date = _parse_roc_date(texts[2].strip())
            if actual_date is None:
                continue

            spoke_time = texts[3].strip()
            subject = texts[4].strip()

            if not subject:
                continue

            # 同一 stock_id 同一天用遞增序號
            key = f"{stock_id}_{actual_date}"
            seq_counter[key] = seq_counter.get(key, 0) + 1
            seq = str(seq_counter[key])

            rows.append(
                {
                    "date": actual_date,
                    "stock_id": stock_id,
                    "seq": seq,
                    "subject": subject,
                    "spoke_time": spoke_time,
                    "sentiment": classify_sentiment(subject),
                }
            )

    return rows


# ------------------------------------------------------------------ #
#  全市場月營收
# ------------------------------------------------------------------ #

# MOPS 月營收 HTML 表格欄位名稱對應（可能隨年度微調）
_REVENUE_COLUMN_MAP = {
    "公司代號": "stock_id",
    "公司名稱": "stock_name",
    "當月營收": "revenue",
    "上月營收": "prev_month_revenue",
    "去年當月營收": "last_year_revenue",
    "上月比較增減": "mom_growth",
    "去年同月增減": "yoy_growth",
    "當月累計營收": "ytd_revenue",
    "去年累計營收": "last_year_ytd_revenue",
    "前期比較增減": "ytd_yoy_growth",
}


def fetch_mops_monthly_revenue(
    year: int | None = None,
    month: int | None = None,
) -> pd.DataFrame:
    """從 MOPS 公開資訊觀測站抓取全市場月營收（上市+上櫃）。

    使用 mopsov 備援站的靜態 HTML 頁面，每次兩個請求（上市+上櫃）
    即可取得全市場 ~2000+ 支股票的月營收。

    Args:
        year: 西元年份，預設為上月所屬年份
        month: 月份（1~12），預設為上月

    Returns:
        DataFrame 欄位: stock_id, date, revenue, revenue_month, revenue_year,
                        mom_growth, yoy_growth
    """
    # 預設取上個月（月營收通常在次月 10 號後公布）
    if year is None or month is None:
        today = date.today()
        # 上個月
        if today.month == 1:
            default_year = today.year - 1
            default_month = 12
        else:
            default_year = today.year
            default_month = today.month - 1
        year = year or default_year
        month = month or default_month

    roc_year = year - 1911

    all_rows: list[pd.DataFrame] = []

    # 上市 (sii) + 上櫃 (otc)
    for market in ("sii", "otc"):
        url = f"https://mopsov.twse.com.tw/nas/t21/{market}/t21sc03_{roc_year}_{month}_0.html"
        logger.info("[MOPS 月營收] 抓取 %s %d/%d ...", market.upper(), year, month)

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30, verify=False)
            resp.raise_for_status()
        except Exception as e:
            logger.error("[MOPS 月營收] %s 請求失敗: %s", market.upper(), e)
            time.sleep(_REQUEST_DELAY)
            continue

        df = _parse_revenue_html(resp.content, year, month)
        if not df.empty:
            all_rows.append(df)
            logger.info("[MOPS 月營收] %s: %d 支股票", market.upper(), len(df))

        time.sleep(_REQUEST_DELAY)

    if not all_rows:
        logger.warning("[MOPS 月營收] %d/%d 無資料", year, month)
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    logger.info("[MOPS 月營收] 合計: %d 支股票 (%d/%d)", len(result), year, month)
    return result


def _parse_revenue_html(html_content: bytes, year: int, month: int) -> pd.DataFrame:
    """解析 MOPS 月營收 HTML 表格。

    MOPS 頁面編碼為 Big5，使用 pd.read_html 解析表格，
    再根據欄位名稱對應到 MonthlyRevenue ORM 欄位。

    Args:
        html_content: 原始 HTML bytes
        year: 西元年份
        month: 月份

    Returns:
        對齊 MonthlyRevenue ORM 的 DataFrame
    """
    # MOPS HTML 編碼為 Big5
    try:
        html_str = html_content.decode("big5", errors="ignore")
    except Exception:
        html_str = html_content.decode("utf-8", errors="ignore")

    from io import StringIO

    try:
        tables = pd.read_html(StringIO(html_str))
    except Exception:
        logger.warning("[MOPS 月營收] HTML 解析失敗")
        return pd.DataFrame()

    if not tables:
        return pd.DataFrame()

    # MOPS 頁面通常有多個表格（依產業分類），合併所有表格
    all_dfs = []
    for tbl in tables:
        # 跳過太小的表格（表頭或備註）
        if len(tbl) < 2 or len(tbl.columns) < 8:
            continue

        # 處理 MultiIndex 欄位（MOPS HTML 有多層表頭）
        if isinstance(tbl.columns, pd.MultiIndex):
            # 取最底層欄位名稱（如 '當月營收'），去除空白
            tbl.columns = [str(levels[-1]).replace("\u3000", "").strip() for levels in tbl.columns]

        all_dfs.append(tbl)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # 嘗試用標準欄位名稱重新命名
    col_names = [str(c).strip() for c in df.columns]
    rename_map = {}
    for orig, target in _REVENUE_COLUMN_MAP.items():
        for i, cn in enumerate(col_names):
            # 移除空白後比對（MOPS 欄位名稱可能含空格如「公司 代號」）
            if orig.replace(" ", "") in cn.replace(" ", ""):
                rename_map[df.columns[i]] = target
                break

    if "stock_id" not in rename_map.values():
        # 備援：假設第一欄是公司代號
        if len(df.columns) >= 10:
            rename_map = {
                df.columns[0]: "stock_id",
                df.columns[1]: "stock_name",
                df.columns[2]: "revenue",
                df.columns[3]: "prev_month_revenue",
                df.columns[4]: "last_year_revenue",
                df.columns[5]: "mom_growth",
                df.columns[6]: "yoy_growth",
            }

    df = df.rename(columns=rename_map)

    # 只保留有效的股票代號列（4~6 位數字）
    if "stock_id" not in df.columns:
        return pd.DataFrame()

    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    df = df[df["stock_id"].str.match(r"^\d{4,6}$")].copy()

    if df.empty:
        return pd.DataFrame()

    # 清洗數值欄位（移除逗號、轉數值）
    for col in ("revenue", "mom_growth", "yoy_growth"):
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce",
            )

    # 建立日期欄位：以該月最後一天為日期
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    month_end = next_month_first - timedelta(days=1)

    df["date"] = month_end
    df["revenue_month"] = month
    df["revenue_year"] = year

    # 營收單位為千元，轉換為元
    if "revenue" in df.columns:
        df["revenue"] = (df["revenue"] * 1000).astype("Int64")

    # 只保留 MonthlyRevenue ORM 所需欄位
    output_cols = ["stock_id", "date", "revenue", "revenue_month", "revenue_year", "mom_growth", "yoy_growth"]
    for col in output_cols:
        if col not in df.columns:
            df[col] = None

    result = df[output_cols].copy()
    # 過濾掉無營收資料的列
    result = result.dropna(subset=["revenue"])
    result = result[result["revenue"] > 0]

    return result.reset_index(drop=True)
