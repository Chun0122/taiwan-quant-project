"""技術指標計算引擎 — 基於日K線計算常用技術指標。

支援指標：
- SMA(5, 10, 20, 60)
- RSI(14)
- MACD(12, 26, 9) → macd, macd_signal, macd_hist
- Bollinger Bands(20, 2) → bb_upper, bb_middle, bb_lower
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import select
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

from src.data.database import get_session
from src.data.schema import DailyPrice

logger = logging.getLogger(__name__)


def _load_daily_price(stock_id: str) -> pd.DataFrame:
    """從 DB 讀取某股票的日K線，回傳按日期排序的 DataFrame。"""
    with get_session() as session:
        rows = (
            session.execute(select(DailyPrice).where(DailyPrice.stock_id == stock_id).order_by(DailyPrice.date))
            .scalars()
            .all()
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {"date": r.date, "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
            for r in rows
        ]
    )
    return df


def compute_indicators(stock_id: str) -> pd.DataFrame:
    """計算單一股票的所有技術指標，回傳 EAV 長表 DataFrame。

    回傳欄位: stock_id, date, name, value
    """
    df = _load_daily_price(stock_id)
    if df.empty:
        logger.warning("[%s] 無日K線資料，跳過指標計算", stock_id)
        return pd.DataFrame()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    records: list[dict] = []
    dates = df["date"].tolist()

    # --- SMA ---
    for period in (5, 10, 20, 60):
        sma = SMAIndicator(close=close, n=period).sma_indicator()
        name = f"sma_{period}"
        for i, val in enumerate(sma):
            if pd.notna(val):
                records.append({"stock_id": stock_id, "date": dates[i], "name": name, "value": round(val, 4)})

    # --- RSI(14) ---
    rsi = RSIIndicator(close=close, n=14).rsi()
    for i, val in enumerate(rsi):
        if pd.notna(val):
            records.append({"stock_id": stock_id, "date": dates[i], "name": "rsi_14", "value": round(val, 4)})

    # --- MACD(12, 26, 9) ---
    macd_ind = MACD(close=close, n_slow=26, n_fast=12, n_sign=9)
    for series, name in [
        (macd_ind.macd(), "macd"),
        (macd_ind.macd_signal(), "macd_signal"),
        (macd_ind.macd_diff(), "macd_hist"),
    ]:
        for i, val in enumerate(series):
            if pd.notna(val):
                records.append({"stock_id": stock_id, "date": dates[i], "name": name, "value": round(val, 4)})

    # --- Bollinger Bands(20, 2) ---
    bb = BollingerBands(close=close, n=20, ndev=2)
    for series, name in [
        (bb.bollinger_hband(), "bb_upper"),
        (bb.bollinger_mavg(), "bb_middle"),
        (bb.bollinger_lband(), "bb_lower"),
    ]:
        for i, val in enumerate(series):
            if pd.notna(val):
                records.append({"stock_id": stock_id, "date": dates[i], "name": name, "value": round(val, 4)})

    result = pd.DataFrame(records)
    logger.info("[%s] 計算完成: %d 筆指標", stock_id, len(result))
    return result
