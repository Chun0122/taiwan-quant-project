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
}


def is_weekend(d: date) -> bool:
    """判斷是否為週末（六=5, 日=6）。"""
    return d.weekday() >= 5


def is_twse_holiday(d: date) -> bool:
    """判斷是否為 TWSE 公告休市日（不含週末）。

    若該年度未列入 _TWSE_HOLIDAYS，回傳 False（僅依週末判斷）。
    """
    year_holidays = _TWSE_HOLIDAYS.get(d.year)
    if year_holidays is None:
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
