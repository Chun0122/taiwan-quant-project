"""視覺化資料查詢模組 — 提供 Streamlit 頁面所需的 DB 查詢函式。"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import func, select

from src.data.database import get_session, init_db
from src.data.schema import (
    BacktestResult, DailyPrice, InstitutionalInvestor,
    MarginTrading, TechnicalIndicator, Trade,
)

init_db()


def get_stock_list() -> list[str]:
    """取得 DB 中有日K線資料的股票清單。"""
    with get_session() as session:
        rows = session.execute(
            select(func.distinct(DailyPrice.stock_id)).order_by(DailyPrice.stock_id)
        ).scalars().all()
    return list(rows)


def load_price_with_indicators(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """載入日K線 + 技術指標寬表。"""
    with get_session() as session:
        prices = session.execute(
            select(DailyPrice)
            .where(DailyPrice.stock_id == stock_id)
            .where(DailyPrice.date >= start)
            .where(DailyPrice.date <= end)
            .order_by(DailyPrice.date)
        ).scalars().all()

    if not prices:
        return pd.DataFrame()

    df = pd.DataFrame([
        {"date": r.date, "open": r.open, "high": r.high,
         "low": r.low, "close": r.close, "volume": r.volume,
         "turnover": r.turnover, "spread": r.spread}
        for r in prices
    ])

    # 載入指標並 pivot
    with get_session() as session:
        indicators = session.execute(
            select(TechnicalIndicator)
            .where(TechnicalIndicator.stock_id == stock_id)
            .where(TechnicalIndicator.date >= start)
            .where(TechnicalIndicator.date <= end)
            .order_by(TechnicalIndicator.date)
        ).scalars().all()

    if indicators:
        df_ind = pd.DataFrame([
            {"date": r.date, "name": r.name, "value": r.value}
            for r in indicators
        ])
        df_wide = df_ind.pivot_table(index="date", columns="name", values="value")
        df = df.merge(df_wide, on="date", how="left")

    df["date"] = pd.to_datetime(df["date"])
    return df


def load_institutional(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """載入三大法人買賣超資料。"""
    with get_session() as session:
        rows = session.execute(
            select(InstitutionalInvestor)
            .where(InstitutionalInvestor.stock_id == stock_id)
            .where(InstitutionalInvestor.date >= start)
            .where(InstitutionalInvestor.date <= end)
            .order_by(InstitutionalInvestor.date)
        ).scalars().all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([
        {"date": r.date, "name": r.name, "buy": r.buy, "sell": r.sell, "net": r.net}
        for r in rows
    ])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_margin(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """載入融資融券資料。"""
    with get_session() as session:
        rows = session.execute(
            select(MarginTrading)
            .where(MarginTrading.stock_id == stock_id)
            .where(MarginTrading.date >= start)
            .where(MarginTrading.date <= end)
            .order_by(MarginTrading.date)
        ).scalars().all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([
        {"date": r.date,
         "margin_balance": r.margin_balance,
         "short_balance": r.short_balance}
        for r in rows
    ])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_backtest_list() -> pd.DataFrame:
    """載入所有回測紀錄。"""
    with get_session() as session:
        rows = session.execute(
            select(BacktestResult).order_by(BacktestResult.id.desc())
        ).scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([
        {"id": r.id, "stock_id": r.stock_id, "strategy_name": r.strategy_name,
         "start_date": r.start_date, "end_date": r.end_date,
         "initial_capital": r.initial_capital, "final_capital": r.final_capital,
         "total_return": r.total_return, "annual_return": r.annual_return,
         "sharpe_ratio": r.sharpe_ratio, "max_drawdown": r.max_drawdown,
         "win_rate": r.win_rate, "total_trades": r.total_trades,
         "created_at": r.created_at}
        for r in rows
    ])


def load_backtest_by_id(backtest_id: int) -> dict | None:
    """載入單筆回測紀錄。"""
    with get_session() as session:
        r = session.execute(
            select(BacktestResult).where(BacktestResult.id == backtest_id)
        ).scalar_one_or_none()

    if not r:
        return None

    return {
        "id": r.id, "stock_id": r.stock_id, "strategy_name": r.strategy_name,
        "start_date": r.start_date, "end_date": r.end_date,
        "initial_capital": r.initial_capital, "final_capital": r.final_capital,
        "total_return": r.total_return, "annual_return": r.annual_return,
        "sharpe_ratio": r.sharpe_ratio, "max_drawdown": r.max_drawdown,
        "win_rate": r.win_rate, "total_trades": r.total_trades,
    }


def load_trades(backtest_id: int) -> pd.DataFrame:
    """載入指定回測的交易明細。"""
    with get_session() as session:
        rows = session.execute(
            select(Trade)
            .where(Trade.backtest_id == backtest_id)
            .order_by(Trade.entry_date)
        ).scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([
        {"entry_date": r.entry_date, "entry_price": r.entry_price,
         "exit_date": r.exit_date, "exit_price": r.exit_price,
         "shares": r.shares, "pnl": r.pnl, "return_pct": r.return_pct}
        for r in rows
    ])
