"""Rotation 股利會計純函數層（A3，MASTER_PLAN §6.2 #10）。

背景：rotation live/backtest 過去完全忽略股利——除息日價格跳空被計為虧損
（權益假性回撤、可能誤觸停損與 25% 熔斷）。本模組提供：
  - 除息事件載入（依日分組，供每日迴圈查詢）
  - 除權息調整因子（與 Strategy._apply_dividend_adjustment 同式，SSOT）
  - 持倉現金入帳與停損價調整

使用約定（裁決 2026-07-05）：
  - 入帳時點 = ``Dividend.date``（除權息基準日/除息交易日，已抽樣驗證：
    2330 2025-12-17 ex-date 跳空恰等於現金股利 5 元），非 cash_payment_date
    ——消除假性回撤優先；應收股利精緻化列實盤前 follow-up。
  - 第一版：現金股利入帳完整實作；股票股利僅用於停損價因子調整，
    **持倉股數不調**（記 warning，配股股數/畸零股列 follow-up）。
  - 除息處理必須在每日迴圈最前（執行 pending 成交之前）：D 日 open 已是
    除息價，成交前帳上停損價須先調整完畢。

資料前提：dividend 表僅涵蓋已同步個股（watchlist + A3-3 起的 rotation
持倉/候選）。查無資料 = 無法區分「無配息」與「未同步」，caller 應搭配
同步步驟與覆蓋率警告。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select

from src.data.schema import Dividend

logger = logging.getLogger(__name__)

# 因子合法區間（與 Strategy._apply_dividend_adjustment 一致）：
# <=0 表示現金股利 >= 前收盤（資料錯誤）；>=1.5 表示異常放大
_FACTOR_MIN = 0.0
_FACTOR_MAX = 1.5


@dataclass(frozen=True)
class DividendEvent:
    """單一個股單一除息日事件。"""

    stock_id: str
    ex_date: date
    cash_dividend: float
    stock_dividend: float

    @property
    def has_payout(self) -> bool:
        return self.cash_dividend > 0 or self.stock_dividend > 0


def load_dividend_events(
    session,
    stock_ids: list[str],
    start: date,
    end: date,
) -> dict[date, dict[str, DividendEvent]]:
    """載入區間內除息事件，依 {ex_date: {stock_id: event}} 分組（供每日迴圈）。

    空 payout（cash=stock=0）的列直接略過。
    """
    if not stock_ids:
        return {}
    rows = (
        session.execute(
            select(Dividend)
            .where(
                Dividend.stock_id.in_(stock_ids),
                Dividend.date >= start,
                Dividend.date <= end,
            )
            .order_by(Dividend.date)
        )
        .scalars()
        .all()
    )
    out: dict[date, dict[str, DividendEvent]] = {}
    for r in rows:
        ev = DividendEvent(
            stock_id=r.stock_id,
            ex_date=r.date,
            cash_dividend=r.cash_dividend or 0.0,
            stock_dividend=r.stock_dividend or 0.0,
        )
        if not ev.has_payout:
            continue
        out.setdefault(r.date, {})[r.stock_id] = ev
    return out


def dividend_adjustment_factor(
    prev_close: float | None,
    cash_dividend: float,
    stock_dividend: float,
) -> float | None:
    """除權息調整因子（與 Strategy._apply_dividend_adjustment :253-255 同式）。

    factor = (prev_close - cash) / prev_close，配股再 /= (1 + stock/10)。
    prev_close 缺失或因子落在 (0, 1.5) 之外回 None（caller 跳過調整並記 warning）。
    """
    if prev_close is None or prev_close <= 0:
        return None
    factor = (prev_close - cash_dividend) / prev_close
    if stock_dividend > 0:
        factor /= 1 + stock_dividend / 10
    if factor <= _FACTOR_MIN or factor >= _FACTOR_MAX:
        return None
    return factor


def adjust_stop_loss_for_dividend(
    stop_loss: float | None,
    prev_close: float | None,
    cash_dividend: float,
    stock_dividend: float,
) -> float | None:
    """除息日停損價調整：stop × factor（維持與前收盤的相對距離，防跳空誤觸）。

    stop_loss 為 None 或因子無效時原樣回傳。
    """
    if stop_loss is None:
        return None
    factor = dividend_adjustment_factor(prev_close, cash_dividend, stock_dividend)
    if factor is None:
        logger.warning(
            "除息因子無效（prev_close=%s, cash=%.4f, stock=%.4f），停損價 %.2f 不調整",
            prev_close,
            cash_dividend,
            stock_dividend,
            stop_loss,
        )
        return stop_loss
    return stop_loss * factor


def dividend_cash_for_position(shares: int, cash_dividend: float) -> float:
    """持倉除息日現金入帳金額 = 股數 × 每股現金股利。"""
    if shares <= 0 or cash_dividend <= 0:
        return 0.0
    return shares * cash_dividend
