"""TWSE 交易日行事曆。

提供假日判斷與交易日推算功能，避免假日執行 morning-routine 產生過期訊號。

資料來源：TWSE 公告（每年更新）。
若當年度尚未列入，fallback 為週末判斷 + DB 交易日紀錄。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# ── TWSE 公告休市日（不含週末）──
# 格式：{year: frozenset[date]}
# 每年 12 月前更新次年資料
_TWSE_HOLIDAYS: dict[int, frozenset[date]] = {
    2025: frozenset(
        [
            date(2025, 1, 1),  # 元旦
            date(2025, 1, 27),  # 除夕彈性
            date(2025, 1, 28),  # 農曆除夕
            date(2025, 1, 29),  # 春節
            date(2025, 1, 30),  # 春節
            date(2025, 1, 31),  # 春節
            date(2025, 2, 28),  # 和平紀念日
            date(2025, 4, 3),  # 兒童節（調整）
            date(2025, 4, 4),  # 清明節
            date(2025, 5, 1),  # 勞動節
            date(2025, 5, 30),  # 端午節（調整）
            date(2025, 6, 2),  # 端午節
            date(2025, 10, 6),  # 中秋節
            date(2025, 10, 10),  # 國慶日
        ]
    ),
    2026: frozenset(
        [
            date(2026, 1, 1),  # 元旦
            date(2026, 2, 16),  # 除夕彈性
            date(2026, 2, 17),  # 農曆除夕
            date(2026, 2, 18),  # 春節
            date(2026, 2, 19),  # 春節
            date(2026, 2, 20),  # 春節
            date(2026, 2, 27),  # 和平紀念日（調整）
            date(2026, 4, 3),  # 兒童節
            date(2026, 4, 6),  # 清明節（調整）
            date(2026, 5, 1),  # 勞動節
            date(2026, 6, 19),  # 端午節
            date(2026, 9, 25),  # 中秋節
            date(2026, 10, 9),  # 國慶日（調整）
        ]
    ),
    # 2027：W3 audit fix（2026-05-09）— 預先排入暫定假日表，待 TWSE 官方公告後校對。
    # 春節依農曆推算（2027 農曆新年=2027-02-06 週六），補休與彈性放假以行政院往例估算。
    # 重要：每年 12 月底 TWSE 公告次年正式交易日後，務必對照 https://www.twse.com.tw 校對。
    2027: frozenset(
        [
            date(2027, 1, 1),  # 元旦
            date(2027, 2, 5),  # 除夕（農曆 12/29，週五）
            date(2027, 2, 8),  # 春節（初二補休 — 2/6 週六、2/7 週日）
            date(2027, 2, 9),  # 春節（初三）
            date(2027, 2, 10),  # 春節（初四）
            date(2027, 2, 11),  # 春節（初五，可能彈性放假，待確認）
            date(2027, 3, 1),  # 和平紀念日（2/28 週日 → 補休）
            date(2027, 4, 5),  # 清明節（兒童節合併，週一）
            # 5/1 勞動節為週六（金融業放假，但證券業視往例多有放假）
            date(2027, 6, 8),  # 端午節（農曆 5/5，2027 對應 6/8 週二）
            date(2027, 9, 15),  # 中秋節（農曆 8/15，2027 對應 9/15 週三）
            date(2027, 10, 11),  # 國慶日補休（10/10 週日 → 10/11 週一補休）
        ]
    ),
}

# W3 audit fix：缺資料的年份只 log 一次（避免 morning-routine 全程刷屏）
_LOGGED_MISSING_YEARS: set[int] = set()


def is_weekend(d: date) -> bool:
    """判斷是否為週末（六=5, 日=6）。"""
    return d.weekday() >= 5


def is_twse_holiday(d: date) -> bool:
    """判斷是否為 TWSE 公告休市日（不含週末）。

    若該年度未列入 _TWSE_HOLIDAYS，回傳 False（僅依週末判斷），
    並 log warning 提示需手動更新（W3 audit fix，每年 once）。
    """
    year_holidays = _TWSE_HOLIDAYS.get(d.year)
    if year_holidays is None:
        if d.year not in _LOGGED_MISSING_YEARS:
            logger.warning(
                "[Calendar] %d 年 TWSE 假日資料尚未建立 — 僅依週末判斷可能誤判元旦/春節為交易日。"
                "請更新 src/data/calendar.py:_TWSE_HOLIDAYS",
                d.year,
            )
            _LOGGED_MISSING_YEARS.add(d.year)
        return False
    return d in year_holidays


def is_trading_day(d: date) -> bool:
    """判斷是否為 TWSE 交易日。

    交易日 = 非週末 AND 非 TWSE 公告休市日。
    """
    if is_weekend(d):
        return False
    return not is_twse_holiday(d)


def next_trading_day(d: date) -> date:
    """回傳 d 之後的下一個交易日（不含 d 本身）。"""
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def prev_trading_day(d: date) -> date:
    """回傳 d 之前的最近交易日（不含 d 本身）。"""
    candidate = d - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def get_trading_days(start: date, end: date) -> list[date]:
    """回傳 [start, end] 區間內的所有交易日。"""
    days: list[date] = []
    d = start
    while d <= end:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def has_calendar_data(year: int) -> bool:
    """檢查指定年份是否有 TWSE 假日資料。"""
    return year in _TWSE_HOLIDAYS
