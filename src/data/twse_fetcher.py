"""TWSE / TPEX 官方開放資料抓取器。

免費、無需 Token，一次請求取得全市場資料。
用於 discover 全市場掃描，4 次 API 呼叫即可覆蓋上市 + 上櫃所有股票。

資料來源：
- TWSE 台灣證券交易所: https://www.twse.com.tw
- TPEX 櫃買中心: https://www.tpex.org.tw
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, timedelta

import pandas as pd
import requests
import urllib3

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
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=8, verify=False)
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

        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "open": open_ or close,
                "high": high or close,
                "low": low or close,
                "close": close,
                "volume": int(volume) if volume else 0,
                "turnover": int(turnover) if turnover else 0,
                "spread": spread_val or 0.0,
            }
        )

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
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=8, verify=False)
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

        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "name": "Foreign_Investor",
                "buy": int(foreign_buy),
                "sell": int(foreign_sell),
                "net": int(foreign_net),
            }
        )
        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "name": "Investment_Trust",
                "buy": int(trust_buy),
                "sell": int(trust_sell),
                "net": int(trust_net),
            }
        )
        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "name": "Dealer_self",
                "buy": int(dealer_buy),
                "sell": int(dealer_sell),
                "net": int(dealer_net),
            }
        )

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
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=8, verify=False)
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

        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "open": open_ or close,
                "high": high or close,
                "low": low or close,
                "close": close,
                "volume": int(volume) if volume else 0,
                "turnover": int(turnover) if turnover else 0,
                "spread": change or 0.0,
            }
        )

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
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=8, verify=False)
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

        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "name": "Foreign_Investor",
                "buy": int(foreign_buy),
                "sell": int(foreign_sell),
                "net": int(foreign_net),
            }
        )
        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "name": "Investment_Trust",
                "buy": int(trust_buy),
                "sell": int(trust_sell),
                "net": int(trust_net),
            }
        )
        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "name": "Dealer_self",
                "buy": int(dealer_buy),
                "sell": int(dealer_sell),
                "net": int(dealer_net),
            }
        )

    df = pd.DataFrame(rows)
    logger.info("TPEX 三大法人: %d 支股票", len(df) // 3 if len(df) > 0 else 0)
    time.sleep(_REQUEST_DELAY)
    return df


# ------------------------------------------------------------------ #
#  融資融券
# ------------------------------------------------------------------ #


def fetch_twse_margin(target_date: date | None = None) -> pd.DataFrame:
    """抓取 TWSE 上市股票全市場融資融券。

    回傳欄位: date, stock_id, margin_buy, margin_sell, margin_balance,
              short_sell, short_buy, short_balance
    """
    if target_date is None:
        target_date = _find_last_trading_day(date.today())

    date_str = target_date.strftime("%Y%m%d")
    url = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
    params = {"date": date_str, "selectType": "STOCK", "response": "json"}

    logger.info("抓取 TWSE 上市融資融券: %s", target_date.isoformat())

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=8, verify=False)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("TWSE 融資融券請求失敗: %s", e)
        return pd.DataFrame()

    if data.get("stat") != "OK":
        logger.warning("TWSE 融資融券無資料: %s", data.get("stat", ""))
        return pd.DataFrame()

    # 找到包含個股資料的 table（fields 包含「股票代號」）
    stock_table = None
    for table in data.get("tables", []):
        fields = table.get("fields", [])
        if fields and "股票代號" in fields[0]:
            stock_table = table
            break

    if stock_table is None:
        logger.warning("TWSE 融資融券: 找不到個股資料 table")
        return pd.DataFrame()

    rows = []
    for item in stock_table.get("data", []):
        if len(item) < 13:
            continue

        stock_id = item[0].strip()
        # 融資: 買進(1), 賣出(2), 現金償還(3), 前日餘額(4), 今日餘額(5)
        margin_buy = _parse_number(item[1]) or 0
        margin_sell = _parse_number(item[2]) or 0
        margin_balance = _parse_number(item[5]) or 0
        # 融券: 賣出(7), 買進(8), 現券償還(9), 前日餘額(10), 當日餘額(11)
        short_sell = _parse_number(item[7]) or 0
        short_buy = _parse_number(item[8]) or 0
        short_balance = _parse_number(item[11]) or 0

        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "margin_buy": int(margin_buy),
                "margin_sell": int(margin_sell),
                "margin_balance": int(margin_balance),
                "short_sell": int(short_sell),
                "short_buy": int(short_buy),
                "short_balance": int(short_balance),
            }
        )

    df = pd.DataFrame(rows)
    logger.info("TWSE 融資融券: %d 支股票", len(df))
    time.sleep(_REQUEST_DELAY)
    return df


def fetch_tpex_margin(target_date: date | None = None) -> pd.DataFrame:
    """抓取 TPEX 上櫃股票全市場融資融券。

    回傳欄位同 TWSE: date, stock_id, margin_buy, margin_sell, margin_balance,
                      short_sell, short_buy, short_balance
    """
    if target_date is None:
        target_date = _find_last_trading_day(date.today())

    roc_date = _to_roc_date(target_date)
    url = "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php"
    params = {"l": "zh-tw", "d": roc_date, "se": "AL"}

    logger.info("抓取 TPEX 上櫃融資融券: %s", target_date.isoformat())

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=8, verify=False)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("TPEX 融資融券請求失敗: %s", e)
        return pd.DataFrame()

    if data.get("stat") != "ok" and data.get("stat") != "OK":
        logger.warning("TPEX 融資融券無資料: %s", data.get("stat", ""))
        return pd.DataFrame()

    tables = data.get("tables", [])
    if not tables:
        logger.warning("TPEX 融資融券: 無 tables")
        return pd.DataFrame()

    rows = []
    for item in tables[0].get("data", []):
        if len(item) < 13:
            continue

        stock_id = item[0].strip()
        # TPEX 融資融券欄位配置：
        # 融資: 買進(2), 賣出(3), 現償(4), 餘額(5 or 6)
        # 融券: 賣出(8), 買進(9), 現償(10), 餘額(11 or 12)
        margin_buy = _parse_number(item[2]) or 0
        margin_sell = _parse_number(item[3]) or 0
        margin_balance = _parse_number(item[5]) or 0
        short_sell = _parse_number(item[8]) or 0
        short_buy = _parse_number(item[9]) or 0
        short_balance = _parse_number(item[11]) or 0

        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "margin_buy": int(margin_buy),
                "margin_sell": int(margin_sell),
                "margin_balance": int(margin_balance),
                "short_sell": int(short_sell),
                "short_buy": int(short_buy),
                "short_balance": int(short_balance),
            }
        )

    df = pd.DataFrame(rows)
    logger.info("TPEX 融資融券: %d 支股票", len(df))
    time.sleep(_REQUEST_DELAY)
    return df


def fetch_market_margin(target_date: date | None = None) -> pd.DataFrame:
    """抓取全市場（上市 + 上櫃）融資融券。

    合併 TWSE + TPEX 資料，回傳統一欄位的 DataFrame。
    """
    df_twse = fetch_twse_margin(target_date)
    df_tpex = fetch_tpex_margin(target_date)

    dfs = [df for df in [df_twse, df_tpex] if not df.empty]
    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    logger.info("全市場融資融券合計: %d 支股票", len(result))
    return result


# ------------------------------------------------------------------ #
#  借券賣出彙總（SBL — Securities Borrowing and Lending）
# ------------------------------------------------------------------ #


def fetch_twse_sbl(target_date: date | None = None) -> pd.DataFrame:
    """抓取 TWSE 全市場借券賣出彙總（日資料，TWT96U）。

    每日一次更新，TWSE 免費開放資料，無需 Token。
    借券餘額高 → 空頭壓力大，可用作 MomentumScanner 負向因子。

    回傳欄位: date, stock_id, sbl_sell_volume, sbl_balance, sbl_prev_balance, sbl_change
        - sbl_sell_volume:  當日借券賣出成交量（股）
        - sbl_balance:      借券餘額（股，當日結算後）
        - sbl_prev_balance: 前日借券餘額（股）
        - sbl_change:       借券餘額日變化（sbl_balance - sbl_prev_balance）
    """
    if target_date is None:
        target_date = _find_last_trading_day(date.today())

    date_str = target_date.strftime("%Y%m%d")
    url = "https://www.twse.com.tw/rwd/zh/sbl/TWT96U"
    params = {"date": date_str, "response": "json"}

    logger.info("抓取 TWSE 借券賣出彙總 (TWT96U): %s", target_date.isoformat())

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=8, verify=False)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("TWSE 借券賣出請求失敗: %s", e)
        return pd.DataFrame()

    if data.get("stat") != "OK":
        logger.warning("TWSE 借券賣出無資料（可能為假日）: %s", data.get("stat", ""))
        return pd.DataFrame()

    # TWT96U 回傳格式：資料在頂層 data 陣列（非 tables）
    # 預期欄位順序（TWSE 可能調整，實作時加防護）：
    # [0] 證券代號  [1] 證券名稱  [2] 當日借券成交量  [3] 次一交易日可回補
    # [4] 借券餘額  [5] 前日借券餘額
    raw_data = data.get("data", [])

    # 若資料在 tables 結構中（API 版本差異）則嘗試取得
    if not raw_data:
        for table in data.get("tables", []):
            candidate = table.get("data", [])
            if candidate:
                raw_data = candidate
                break

    if not raw_data:
        logger.warning("TWSE 借券賣出: 無資料列")
        return pd.DataFrame()

    # 欄位結構防護：記錄 fields 以便追蹤 API 改版
    fields = data.get("fields", [])
    if fields and len(fields) >= 4:
        logger.debug("TWSE TWT96U fields: %s", fields[:6])

    rows = []
    for item in raw_data:
        if len(item) < 5:
            continue

        stock_id = item[0].strip()
        # 只保留 4 碼純數字股票
        if not (stock_id.isdigit() and len(stock_id) == 4):
            continue

        # 欄位對應（依 TWT96U 標準格式）
        # item[1] = 證券名稱（跳過）
        sbl_sell_volume = _parse_number(item[2]) if len(item) > 2 else None  # 當日借券成交量
        sbl_balance = _parse_number(item[4]) if len(item) > 4 else None  # 借券餘額
        sbl_prev_balance = _parse_number(item[5]) if len(item) > 5 else None  # 前日借券餘額

        # 計算日變化
        if sbl_balance is not None and sbl_prev_balance is not None:
            sbl_change = int(sbl_balance) - int(sbl_prev_balance)
        else:
            sbl_change = None

        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "sbl_sell_volume": int(sbl_sell_volume) if sbl_sell_volume is not None else None,
                "sbl_balance": int(sbl_balance) if sbl_balance is not None else None,
                "sbl_prev_balance": int(sbl_prev_balance) if sbl_prev_balance is not None else None,
                "sbl_change": sbl_change,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("TWSE 借券賣出: %d 支股票", len(df))
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


# ------------------------------------------------------------------ #
#  估值（PE/PB/殖利率）
# ------------------------------------------------------------------ #


def fetch_twse_valuation_all(target_date: date | None = None) -> pd.DataFrame:
    """抓取 TWSE 上市股票全市場本益比/殖利率/股價淨值比（BWIBBU_d）。

    單次 HTTP 請求即可取得全市場（上市）所有股票的估值資料。
    免費、無需 Token。

    回傳欄位: date, stock_id, pe_ratio, pb_ratio, dividend_yield
    """
    if target_date is None:
        target_date = _find_last_trading_day(date.today())

    date_str = target_date.strftime("%Y%m%d")
    url = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
    params = {"date": date_str, "selectType": "ALL", "response": "json"}

    logger.info("抓取 TWSE 上市估值 (BWIBBU_d): %s", target_date.isoformat())

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=8, verify=False)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("TWSE 估值請求失敗: %s", e)
        return pd.DataFrame()

    if data.get("stat") != "OK":
        logger.warning("TWSE 估值無資料（可能為假日）: %s", data.get("stat", ""))
        return pd.DataFrame()

    # BWIBBU_d 回傳格式：資料在頂層 data 陣列（非 tables）
    # 2026-03 更新後新格式：
    # fields: ["證券代號", "證券名稱", "收盤價", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"]
    raw_data = data.get("data", [])
    if not raw_data:
        logger.warning("TWSE 估值: 無資料列")
        return pd.DataFrame()

    # 欄位動態偵測：根據 fields 判斷新/舊格式，自動選擇正確 index
    fields = data.get("fields", [])
    # 新格式（含收盤價）：index 2=收盤價, 3=殖利率, 5=本益比, 6=股價淨值比
    # 舊格式：index 2=殖利率, 4=本益比, 5=股價淨值比
    if fields and len(fields) > 2 and "收盤價" in fields[2]:
        idx_dy, idx_pe, idx_pb, min_len = 3, 5, 6, 7
        logger.debug("TWSE BWIBBU_d 使用新格式（含收盤價），欄位數=%d", len(fields))
    else:
        idx_dy, idx_pe, idx_pb, min_len = 2, 4, 5, 6
        if fields:
            expected = {2: "殖利率", 4: "本益比", 5: "股價淨值比"}
            for idx, keyword in expected.items():
                if idx < len(fields) and keyword not in fields[idx]:
                    logger.warning(
                        "TWSE BWIBBU_d 欄位結構可能異動！期待含 '%s' 在 index %d，實際: %s",
                        keyword,
                        idx,
                        fields[idx],
                    )

    rows = []
    for item in raw_data:
        if len(item) < min_len:
            continue

        stock_id = item[0].strip()
        # 只保留 4 碼純數字股票（排除指數、ETF 等非個股代號）
        if not (stock_id.isdigit() and len(stock_id) == 4):
            continue

        dividend_yield = _parse_number(item[idx_dy])  # 殖利率(%)
        pe_ratio = _parse_number(item[idx_pe])  # 本益比
        pb_ratio = _parse_number(item[idx_pb])  # 股價淨值比

        # 至少有一項估值資料才寫入
        if dividend_yield is None and pe_ratio is None and pb_ratio is None:
            continue

        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "dividend_yield": dividend_yield,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("TWSE 估值: %d 支股票", len(df))
    time.sleep(_REQUEST_DELAY)
    return df


def fetch_tpex_valuation_all(target_date: date | None = None) -> pd.DataFrame:
    """抓取 TPEX 上櫃股票全市場本益比/殖利率/股價淨值比。

    回傳欄位同 TWSE: date, stock_id, pe_ratio, pb_ratio, dividend_yield
    """
    if target_date is None:
        target_date = _find_last_trading_day(date.today())

    roc_date = _to_roc_date(target_date)
    url = "https://www.tpex.org.tw/web/stock/aftertrading/peratio_book/pera_result.php"
    params = {"l": "zh-tw", "d": roc_date, "s": "0,asc"}

    logger.info("抓取 TPEX 上櫃估值 (pera): %s", target_date.isoformat())

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=8, verify=False)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("TPEX 估值請求失敗: %s", e)
        return pd.DataFrame()

    if data.get("stat") not in ("ok", "OK"):
        logger.warning("TPEX 估值無資料: %s", data.get("stat", ""))
        return pd.DataFrame()

    # TPEX pera 回傳：tables[0].data
    # fields: ["代號", "名稱", "本益比", "殖利率(%)", "股價淨值比"]
    tables = data.get("tables", [])
    if not tables:
        logger.warning("TPEX 估值: 無 tables")
        return pd.DataFrame()

    rows = []
    for item in tables[0].get("data", []):
        if len(item) < 5:
            continue

        stock_id = item[0].strip()
        if not (stock_id.isdigit() and len(stock_id) == 4):
            continue

        pe_ratio = _parse_number(item[2])  # 本益比
        dividend_yield = _parse_number(item[3])  # 殖利率(%)
        pb_ratio = _parse_number(item[4])  # 股價淨值比

        if dividend_yield is None and pe_ratio is None and pb_ratio is None:
            continue

        rows.append(
            {
                "date": target_date,
                "stock_id": stock_id,
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "dividend_yield": dividend_yield,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("TPEX 估值: %d 支股票", len(df))
    time.sleep(_REQUEST_DELAY)
    return df


def fetch_market_valuation_all(target_date: date | None = None) -> pd.DataFrame:
    """抓取全市場（上市 + 上櫃）本益比/殖利率/股價淨值比。

    合併 TWSE BWIBBU_d + TPEX pera，僅需 2 次 API 呼叫。
    免費、無需 Token。用於 ValueScanner / DividendScanner Stage 0.5 cold-start 補抓。

    回傳欄位: date, stock_id, pe_ratio, pb_ratio, dividend_yield
    """
    df_twse = fetch_twse_valuation_all(target_date)
    df_tpex = fetch_tpex_valuation_all(target_date)

    dfs = [df for df in [df_twse, df_tpex] if not df.empty]
    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    logger.info("全市場估值合計: %d 支股票", len(result))
    return result


def fetch_dj_broker_trades(stock_id: str, start: date, end: date) -> pd.DataFrame:
    """從 DJ 分點端點取得分點買賣彙整資料（免費，支援日期範圍）。

    資料來源：富邦 DJ 分點進出端點（fubon-ebrokerdj.fbs.com.tw），免費無需 Token。
    替代 FinMind TaiwanStockTradingDailyReport（免費帳號已無法取得）。

    回傳欄位: date, stock_id, broker_id, broker_name, buy, sell

    注意事項：
    - 回傳資料為 start~end 期間的彙整（非每日分拆），date 欄位統一為 end 日期
    - buy/sell 單位為張（千股），已乘 1000 換算為股後回傳
    - buy_price / sell_price 不提供（DJ 端點無均價欄位），Smart Broker 因子自動停用
    - broker_id 使用分點 BHID（公司代號），若同公司有多分點則彙整加總
    - 僅回傳最活躍的 ~30 個分點（前 15 淨買超 + 前 15 淨賣超）
    - Big5 編碼，以 content.decode('big5', errors='replace') 解析
    """
    url = "https://fubon-ebrokerdj.fbs.com.tw/z/zc/zco/zco.djhtm"
    date_str_start = f"{start.year}-{start.month}-{start.day}"  # YYYY-M-D（月日不補零）
    date_str_end = f"{end.year}-{end.month}-{end.day}"
    params = {"a": stock_id, "e": date_str_start, "f": date_str_end}

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=15, verify=False)
        resp.raise_for_status()
        html = resp.content.decode("big5", errors="replace")
    except Exception as exc:
        logger.warning("[DJ分點] %s 請求失敗: %s", stock_id, exc)
        return pd.DataFrame()

    # 解析 HTML：每個 broker 條目格式為
    #   <a href="...b=BRANCH&BHID=FIRM_ID">broker_name</a></TD>
    #   <TD class="t3n1">buy_qty</TD>
    #   <TD class="t3n1">sell_qty</TD>
    # 每個 <TR> 包含兩組 broker（左：淨買超，右：淨賣超）
    pattern = re.compile(
        r'<a\s+href="[^"]*[?&]b=[^&"]*&BHID=(\d+)[^"]*">([^<]*)</a></TD>'
        r"\s*<TD[^>]*>([\d,]+)</TD>"  # buy（張）
        r"\s*<TD[^>]*>([\d,]+)</TD>",  # sell（張）
        re.IGNORECASE,
    )

    # 彙整同一 BHID（公司代號）的所有分點（若同公司有主分點和子分點則合計）
    firm_data: dict[str, dict] = {}
    for bhid, bname, buy_str, sell_str in pattern.findall(html):
        buy = int(buy_str.replace(",", ""))
        sell = int(sell_str.replace(",", ""))
        if bhid not in firm_data:
            firm_data[bhid] = {"broker_name": bname.strip(), "buy": 0, "sell": 0}
        firm_data[bhid]["buy"] += buy
        firm_data[bhid]["sell"] += sell

    if not firm_data:
        logger.debug("[DJ分點] %s %s~%s 無資料", stock_id, date_str_start, date_str_end)
        return pd.DataFrame()

    rows = [
        {
            "date": end,  # 彙整期間的截止日期
            "stock_id": stock_id,
            "broker_id": bhid,
            "broker_name": info["broker_name"],
            "buy": info["buy"] * 1000,  # 張 → 股（×1000）
            "sell": info["sell"] * 1000,
            "buy_price": None,  # DJ 端點不提供均價，Smart Broker 自動降為 7F
            "sell_price": None,
        }
        for bhid, info in firm_data.items()
    ]

    logger.info("[DJ分點] %s %s~%s 解析到 %d 個分點", stock_id, date_str_start, date_str_end, len(rows))
    time.sleep(_REQUEST_DELAY)
    return pd.DataFrame(rows)
