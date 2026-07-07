"""Rotation 市場資料查詢 helper（P2 任務 14 phase 1：從 manager.py 抽出）。

皆為 session-based 讀取函式（交易日曆 / 收盤價 / OHLCV / TAIEX / 0050 benchmark）。
無 RotationManager 依賴，可獨立測試。
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import func, select

from src.data.schema import DailyPrice

# Daily snapshot benchmark：採 0050 ETF 作為 alpha 對標
SNAPSHOT_BENCHMARK_STOCK_ID = "0050"


def _get_trading_calendar(session, start: date, end: date) -> list[date]:
    """從 DailyPrice (TAIEX) 取交易日曆。"""
    stmt = (
        select(DailyPrice.date)
        .where(
            DailyPrice.stock_id == "TAIEX",
            DailyPrice.date >= start,
            DailyPrice.date <= end,
        )
        .order_by(DailyPrice.date)
    )
    dates = [row[0] for row in session.execute(stmt).all()]
    if not dates:
        # fallback: 工作日
        d = start
        while d <= end:
            if d.weekday() < 5:
                dates.append(d)
            d += timedelta(days=1)
    return dates


def _get_prices_on_date(session, stock_ids: list[str], target_date: date) -> dict[str, float]:
    """取得指定日期（或最近交易日）的收盤價。

    先嘗試精確比對 target_date，若某些股票找不到資料，
    則 fallback 取最近 5 個交易日內的最新收盤價。
    """
    if not stock_ids:
        return {}

    # 精確比對
    stmt = select(DailyPrice.stock_id, DailyPrice.close).where(
        DailyPrice.stock_id.in_(stock_ids),
        DailyPrice.date == target_date,
    )
    result = {row[0]: row[1] for row in session.execute(stmt).all()}

    # Fallback：找不到精確日期的股票，取最近 5 天內最新收盤價
    missing = [sid for sid in stock_ids if sid not in result]
    if missing:
        fallback_start = target_date - timedelta(days=5)
        sub = (
            select(
                DailyPrice.stock_id,
                func.max(DailyPrice.date).label("max_date"),
            )
            .where(
                DailyPrice.stock_id.in_(missing),
                DailyPrice.date >= fallback_start,
                DailyPrice.date <= target_date,
            )
            .group_by(DailyPrice.stock_id)
            .subquery()
        )
        stmt2 = select(DailyPrice.stock_id, DailyPrice.close).join(
            sub,
            (DailyPrice.stock_id == sub.c.stock_id) & (DailyPrice.date == sub.c.max_date),
        )
        for row in session.execute(stmt2).all():
            result[row[0]] = row[1]

    return result


def _get_ohlcv_exact_date(session, stock_ids: list[str], target_date: date) -> dict[str, dict]:
    """取得指定日期**精確比對**的 OHLCV（無 fallback）。

    A2 fill_pending 用：成交必須用當日真實報價，停牌股缺席即代表不可成交
    （_get_ohlcv_on_date 的 5 天 fallback 會讓停牌股帶著舊 open 誤成交）。
    """
    if not stock_ids:
        return {}
    stmt = select(
        DailyPrice.stock_id,
        DailyPrice.open,
        DailyPrice.high,
        DailyPrice.low,
        DailyPrice.close,
        DailyPrice.volume,
    ).where(
        DailyPrice.stock_id.in_(stock_ids),
        DailyPrice.date == target_date,
    )
    result: dict[str, dict] = {}
    for row in session.execute(stmt).all():
        result[row[0]] = {
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "volume": row[5] or 0,
        }
    return result


def _get_ohlcv_on_date(session, stock_ids: list[str], target_date: date) -> dict[str, dict]:
    """取得指定日期（或最近交易日）的 OHLCV 完整資料。

    先嘗試精確比對 target_date，若某些股票找不到資料，
    則 fallback 取最近 5 天內的最新資料（fallback 時 volume 設為 0 以避免錯誤流動性估算）。

    Returns
    -------
    dict[str, dict]
        {stock_id: {"open": .., "high": .., "low": .., "close": .., "volume": ..}}
    """
    if not stock_ids:
        return {}

    # 精確比對
    stmt = select(
        DailyPrice.stock_id,
        DailyPrice.open,
        DailyPrice.high,
        DailyPrice.low,
        DailyPrice.close,
        DailyPrice.volume,
    ).where(
        DailyPrice.stock_id.in_(stock_ids),
        DailyPrice.date == target_date,
    )
    result: dict[str, dict] = {}
    for row in session.execute(stmt).all():
        result[row[0]] = {
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "volume": row[5] or 0,
        }

    # Fallback：最近 5 天（volume 歸零，避免用非當日量做流動性約束）
    missing = [sid for sid in stock_ids if sid not in result]
    if missing:
        fallback_start = target_date - timedelta(days=5)
        sub = (
            select(
                DailyPrice.stock_id,
                func.max(DailyPrice.date).label("max_date"),
            )
            .where(
                DailyPrice.stock_id.in_(missing),
                DailyPrice.date >= fallback_start,
                DailyPrice.date <= target_date,
            )
            .group_by(DailyPrice.stock_id)
            .subquery()
        )
        stmt2 = select(
            DailyPrice.stock_id,
            DailyPrice.open,
            DailyPrice.high,
            DailyPrice.low,
            DailyPrice.close,
        ).join(
            sub,
            (DailyPrice.stock_id == sub.c.stock_id) & (DailyPrice.date == sub.c.max_date),
        )
        for row in session.execute(stmt2).all():
            result[row[0]] = {
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": 0,  # fallback 資料不代表今日流動性
            }

    return result


def _get_taiex_prices(session, start: date, end: date) -> dict[date, float]:
    """取得 TAIEX 收盤價序列，用於 benchmark 計算。"""
    stmt = select(DailyPrice.date, DailyPrice.close).where(
        DailyPrice.stock_id == "TAIEX",
        DailyPrice.date >= start,
        DailyPrice.date <= end,
    )
    return {row[0]: row[1] for row in session.execute(stmt).all()}


def _get_benchmark_close_on_or_before(session, target_date: date, lookback_days: int = 7) -> float | None:
    """取得 ≤ target_date 的最近一個 0050 收盤價（fallback 非交易日 / 假日）。

    lookback_days 控制最大回溯範圍（預設 7 天 = 涵蓋一週末 + 連假）。
    若找不到回傳 None，呼叫端將寫入 NULL（audit query 過濾即可）。
    """
    stmt = (
        select(DailyPrice.close)
        .where(
            DailyPrice.stock_id == SNAPSHOT_BENCHMARK_STOCK_ID,
            DailyPrice.date <= target_date,
            DailyPrice.date >= target_date - timedelta(days=lookback_days),
        )
        .order_by(DailyPrice.date.desc())
        .limit(1)
    )
    row = session.execute(stmt).first()
    return float(row[0]) if row else None


def _get_benchmark_dividends_between(session, start_exclusive: date, end_inclusive: date) -> float:
    """0050 每股現金股利在 (start_exclusive, end_inclusive] 窗口的加總（A3-4）。

    供 total return benchmark 加法式還原：cum return =
    (close + Σ現金股利 − base_close) / base_close。0050 為 ETF 無股票股利，
    僅加總 cash_dividend；dividend 表未同步 0050 時回 0.0（退化為
    price return，與 morning Step 11b 的 0050 股利補抓搭配消除）。
    """
    if start_exclusive >= end_inclusive:
        return 0.0
    from src.data.schema import Dividend

    # 以實際除息交易日（fallback 基準日）落窗——與持倉入帳同一時點語意
    ex_col = func.coalesce(Dividend.cash_ex_dividend_date, Dividend.date)
    total = session.execute(
        select(func.coalesce(func.sum(Dividend.cash_dividend), 0.0)).where(
            Dividend.stock_id == SNAPSHOT_BENCHMARK_STOCK_ID,
            ex_col > start_exclusive,
            ex_col <= end_inclusive,
        )
    ).scalar_one()
    return float(total or 0.0)
