"""TWSE / TPEX 官方開放資料抓取器。

免費、無需 Token，一次請求取得全市場資料。
用於 discover 全市場掃描，4 次 API 呼叫即可覆蓋上市 + 上櫃所有股票。

資料來源：
- TWSE 台灣證券交易所: https://www.twse.com.tw
- TPEX 櫃買中心: https://www.tpex.org.tw
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import urllib3
import pandas as pd
import requests

# TWSE/TPEX 的 SSL 憑證在部分系統會驗證失敗，停用警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# TWSE/TPEX 建議的 User-Agent
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

# 請求間隔（秒），避免被封鎖
_REQUEST_DELAY = 3


def _parse_number(s: str) -> float | None:
    """將逗號分隔的數字字串轉為 float，無效值回傳 None。"""
    if not s or s in ("--", "----", "", "---", "除權息", "除權", "除息"):
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def _to_roc_date(d: date) -> str:
    """西元日期 → 民國日期字串 (YYY/MM/DD)。"""
    roc_year = d.year - 1911
    return f"{roc_year}/{d.month:02d}/{d.day:02d}"


def _find_last_trading_day(target: date, max_lookback: int = 7) -> date:
    """從 target 往前找最近的交易日（跳過週末）。"""
    d = target
    for _ in range(max_lookback):
        if d.weekday() < 5:  # 週一~週五
            return d
        d -= timedelta(days=1)
    return target


# ------------------------------------------------------------------ #
#  TWSE 上市
# ------------------------------------------------------------------ #

def fetch_twse_daily_prices(target_date: date | None = None) -> pd.DataFrame:
    """抓取 TWSE 上市股票全市場日收盤行情。

    回傳欄位: date, stock_id, open, high, low, close, volume, turnover, spread
    """
    if target_date is None:
        target_date = _find_last_trading_day(date.today())

    date_str = target_date.strftime("%Y%m%d")
    url = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
    params = {"date": date_str, "type": "ALLBUT0999", "response": "json"}

    logger.info("抓取 TWSE 上市日行情: %s", target_date.isoformat())

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("TWSE 日行情請求失敗: %s", e)
        return pd.DataFrame()

    if data.get("stat") != "OK":
        logger.warning("TWSE 日行情無資料（可能為假日）: %s", data.get("stat", ""))
        return pd.DataFrame()

    # 找到包含個股資料的 table（fields 包含「證券代號」）
    stock_table = None
    for table in data.get("tables", []):
        fields = table.get("fields", [])
        if fields and fields[0] in ("證券代號",):
            stock_table = table
            break

    if stock_table is None:
        logger.warning("TWSE 日行情: 找不到個股資料 table")
        return pd.DataFrame()

    rows = []
    for item in stock_table.get("data", []):
        if len(item) < 10:
            continue

        stock_id = item[0].strip()
        open_ = _parse_number(item[5])
        high = _parse_number(item[6])
        low = _parse_number(item[7])
        close = _parse_number(item[8])
        volume = _parse_number(item[2])
        turnover = _parse_number(item[4])

        # 跳過無成交的股票
        if close is None:
            continue

        # 漲跌價差
        direction = item[9].strip() if len(item) > 9 else ""
        spread_val = _parse_number(item[10]) if len(item) > 10 else 0
        if spread_val and direction == "-":
            spread_val = -spread_val

        rows.append({
            "date": target_date,
            "stock_id": stock_id,
            "open": open_ or close,
            "high": high or close,
            "low": low or close,
            "close": close,
            "volume": int(volume) if volume else 0,
            "turnover": int(turnover) if turnover else 0,
            "spread": spread_val or 0.0,
        })

    df = pd.DataFrame(rows)
    logger.info("TWSE 日行情: %d 支股票", len(df))
    time.sleep(_REQUEST_DELAY)
    return df


def fetch_twse_institutional(target_date: date | None = None) -> pd.DataFrame:
    """抓取 TWSE 上市股票全市場三大法人買賣超。

    回傳欄位: date, stock_id, name, buy, sell, net
    （每支股票展開為 3 列：外資、投信、自營商）
    """
    if target_date is None:
        target_date = _find_last_trading_day(date.today())

    date_str = target_date.strftime("%Y%m%d")
    url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    params = {"date": date_str, "selectType": "ALLBUT0999", "response": "json"}

    logger.info("抓取 TWSE 上市三大法人: %s", target_date.isoformat())

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("TWSE 三大法人請求失敗: %s", e)
        return pd.DataFrame()

    if data.get("stat") != "OK":
        logger.warning("TWSE 三大法人無資料: %s", data.get("stat", ""))
        return pd.DataFrame()

    # T86 的 fields/data 在頂層
    rows = []
    for item in data.get("data", []):
        if len(item) < 19:
            continue

        stock_id = item[0].strip()

        # 外資（不含外資自營商）: indices 2,3,4
        foreign_buy = _parse_number(item[2]) or 0
        foreign_sell = _parse_number(item[3]) or 0
        foreign_net = _parse_number(item[4]) or 0

        # 投信: indices 8,9,10
        trust_buy = _parse_number(item[8]) or 0
        trust_sell = _parse_number(item[9]) or 0
        trust_net = _parse_number(item[10]) or 0

        # 自營商合計: 自行買賣(12,13,14) + 避險(15,16,17)
        dealer_buy = (_parse_number(item[12]) or 0) + (_parse_number(item[15]) or 0)
        dealer_sell = (_parse_number(item[13]) or 0) + (_parse_number(item[16]) or 0)
        dealer_net = dealer_buy - dealer_sell

        rows.append({"date": target_date, "stock_id": stock_id, "name": "Foreign_Investor", "buy": int(foreign_buy), "sell": int(foreign_sell), "net": int(foreign_net)})
        rows.append({"date": target_date, "stock_id": stock_id, "name": "Investment_Trust", "buy": int(trust_buy), "sell": int(trust_sell), "net": int(trust_net)})
        rows.append({"date": target_date, "stock_id": stock_id, "name": "Dealer_self", "buy": int(dealer_buy), "sell": int(dealer_sell), "net": int(dealer_net)})

    df = pd.DataFrame(rows)
    logger.info("TWSE 三大法人: %d 支股票", len(df) // 3 if len(df) > 0 else 0)
    time.sleep(_REQUEST_DELAY)
    return df


# ------------------------------------------------------------------ #
#  TPEX 上櫃
# ------------------------------------------------------------------ #

def fetch_tpex_daily_prices(target_date: date | None = None) -> pd.DataFrame:
    """抓取 TPEX 上櫃股票全市場日收盤行情。

    回傳欄位同 TWSE: date, stock_id, open, high, low, close, volume, turnover, spread
    """
    if target_date is None:
        target_date = _find_last_trading_day(date.today())

    roc_date = _to_roc_date(target_date)
    url = "https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php"
    params = {"l": "zh-tw", "d": roc_date, "se": "AL"}

    logger.info("抓取 TPEX 上櫃日行情: %s", target_date.isoformat())

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("TPEX 日行情請求失敗: %s", e)
        return pd.DataFrame()

    if data.get("stat") != "ok" and data.get("stat") != "OK":
        logger.warning("TPEX 日行情無資料: %s", data.get("stat", ""))
        return pd.DataFrame()

    # TPEX 的資料在 tables[0].data 裡
    tables = data.get("tables", [])
    if not tables:
        logger.warning("TPEX 日行情: 無 tables")
        return pd.DataFrame()

    rows = []
    for item in tables[0].get("data", []):
        if len(item) < 9:
            continue

        stock_id = item[0].strip()

        # 判斷是否有名稱欄位（TPEX 有時包含有時不包含）
        # 如果 item[1] 是數字，代表沒有名稱欄位；否則有
        offset = 0
        test_val = item[1].replace(",", "").replace("-", "").replace(".", "")
        if not test_val.replace(" ", "").isdigit():
            offset = 1  # 有名稱欄位，後續欄位往後偏移 1

        close = _parse_number(item[1 + offset])
        change = _parse_number(item[2 + offset])
        open_ = _parse_number(item[3 + offset])
        high = _parse_number(item[4 + offset])
        low = _parse_number(item[5 + offset])
        volume = _parse_number(item[6 + offset])
        turnover = _parse_number(item[7 + offset])

        if close is None:
            continue

        rows.append({
            "date": target_date,
            "stock_id": stock_id,
            "open": open_ or close,
            "high": high or close,
            "low": low or close,
            "close": close,
            "volume": int(volume) if volume else 0,
            "turnover": int(turnover) if turnover else 0,
            "spread": change or 0.0,
        })

    df = pd.DataFrame(rows)
    logger.info("TPEX 日行情: %d 支股票", len(df))
    time.sleep(_REQUEST_DELAY)
    return df


def fetch_tpex_institutional(target_date: date | None = None) -> pd.DataFrame:
    """抓取 TPEX 上櫃股票全市場三大法人買賣超。

    回傳欄位同 TWSE: date, stock_id, name, buy, sell, net
    """
    if target_date is None:
        target_date = _find_last_trading_day(date.today())

    roc_date = _to_roc_date(target_date)
    url = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
    params = {"l": "zh-tw", "d": roc_date, "se": "AL", "t": "D"}

    logger.info("抓取 TPEX 上櫃三大法人: %s", target_date.isoformat())

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("TPEX 三大法人請求失敗: %s", e)
        return pd.DataFrame()

    if data.get("stat") != "ok" and data.get("stat") != "OK":
        logger.warning("TPEX 三大法人無資料: %s", data.get("stat", ""))
        return pd.DataFrame()

    tables = data.get("tables", [])
    if not tables:
        logger.warning("TPEX 三大法人: 無 tables")
        return pd.DataFrame()

    rows = []
    for item in tables[0].get("data", []):
        if len(item) < 22:
            continue

        stock_id = item[0].strip()

        # TPEX 法人欄位配置（含名稱偏移 1）：
        # 外資（不含外資自營商）: 2,3,4
        # 外資自營商: 5,6,7
        # 外資合計: 8,9,10
        # 投信: 11,12,13
        # 自營商（自行買賣）: 14,15,16
        # 自營商（避險）: 17,18,19
        # 自營商合計: 20,21,22
        # 三大法人合計: 23

        foreign_buy = _parse_number(item[2]) or 0
        foreign_sell = _parse_number(item[3]) or 0
        foreign_net = _parse_number(item[4]) or 0

        trust_buy = _parse_number(item[11]) or 0
        trust_sell = _parse_number(item[12]) or 0
        trust_net = _parse_number(item[13]) or 0

        dealer_buy = (_parse_number(item[14]) or 0) + (_parse_number(item[17]) or 0)
        dealer_sell = (_parse_number(item[15]) or 0) + (_parse_number(item[18]) or 0)
        dealer_net = dealer_buy - dealer_sell

        rows.append({"date": target_date, "stock_id": stock_id, "name": "Foreign_Investor", "buy": int(foreign_buy), "sell": int(foreign_sell), "net": int(foreign_net)})
        rows.append({"date": target_date, "stock_id": stock_id, "name": "Investment_Trust", "buy": int(trust_buy), "sell": int(trust_sell), "net": int(trust_net)})
        rows.append({"date": target_date, "stock_id": stock_id, "name": "Dealer_self", "buy": int(dealer_buy), "sell": int(dealer_sell), "net": int(dealer_net)})

    df = pd.DataFrame(rows)
    logger.info("TPEX 三大法人: %d 支股票", len(df) // 3 if len(df) > 0 else 0)
    time.sleep(_REQUEST_DELAY)
    return df


# ------------------------------------------------------------------ #
#  整合：全市場 (TWSE + TPEX)
# ------------------------------------------------------------------ #

def fetch_market_daily_prices(target_date: date | None = None) -> pd.DataFrame:
    """抓取全市場（上市 + 上櫃）日收盤行情。

    合併 TWSE + TPEX 資料，回傳統一欄位的 DataFrame。
    僅需 2 次 API 呼叫。
    """
    df_twse = fetch_twse_daily_prices(target_date)
    df_tpex = fetch_tpex_daily_prices(target_date)

    dfs = [df for df in [df_twse, df_tpex] if not df.empty]
    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    logger.info("全市場日行情合計: %d 支股票", len(result))
    return result


def fetch_market_institutional(target_date: date | None = None) -> pd.DataFrame:
    """抓取全市場（上市 + 上櫃）三大法人買賣超。

    合併 TWSE + TPEX 資料，回傳統一欄位的 DataFrame。
    僅需 2 次 API 呼叫。
    """
    df_twse = fetch_twse_institutional(target_date)
    df_tpex = fetch_tpex_institutional(target_date)

    dfs = [df for df in [df_twse, df_tpex] if not df.empty]
    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    logger.info("全市場三大法人合計: %d 支股票", len(result) // 3 if len(result) > 0 else 0)
    return result
