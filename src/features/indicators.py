"""技術指標計算引擎 — 基於日K線計算常用技術指標。

支援指標：
- SMA(5, 10, 20, 60)
- RSI(14)
- MACD(12, 26, 9) → macd, macd_signal, macd_hist
- Bollinger Bands(20, 2) → bb_upper, bb_middle, bb_lower

週線聚合：
- aggregate_to_weekly() → 日K聚合為週K + 週線 SMA13 / RSI14 / MACD
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


def compute_indicators_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """從 DataFrame 直接計算技術指標，回傳寬表格式。

    用於除權息還原後的價格序列，繞過 EAV 表直接計算。

    Args:
        df: 含 close/high/low 欄位的 DataFrame（index=日期）

    Returns:
        DataFrame with indicator columns (sma_5, sma_10, sma_20, sma_60,
        rsi_14, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower)
    """
    close = df["close"]
    result = pd.DataFrame(index=df.index)

    # SMA
    for period in (5, 10, 20, 60):
        result[f"sma_{period}"] = SMAIndicator(close=close, n=period).sma_indicator()

    # RSI(14)
    result["rsi_14"] = RSIIndicator(close=close, n=14).rsi()

    # MACD(12, 26, 9)
    macd_ind = MACD(close=close, n_slow=26, n_fast=12, n_sign=9)
    result["macd"] = macd_ind.macd()
    result["macd_signal"] = macd_ind.macd_signal()
    result["macd_hist"] = macd_ind.macd_diff()

    # Bollinger Bands(20, 2)
    bb = BollingerBands(close=close, n=20, ndev=2)
    result["bb_upper"] = bb.bollinger_hband()
    result["bb_middle"] = bb.bollinger_mavg()
    result["bb_lower"] = bb.bollinger_lband()

    return result


def aggregate_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """將日K線 DataFrame 聚合為週K線，並計算週線技術指標。

    Args:
        daily_df: 含 open/high/low/close/volume 欄位的日K DataFrame。
                  index 為 datetime 或 date 型別，或含 'date' 欄位。

    Returns:
        週K DataFrame（DatetimeIndex，以週五為錨點），含原始 OHLCV 欄位及：
          sma_13（13 週均線）、rsi_14（週 RSI14）、
          macd / macd_signal / macd_hist（週線 MACD）。
        資料不足（< 3 週）或輸入為空時回傳空 DataFrame。
    """
    if daily_df.empty:
        return pd.DataFrame()

    df = daily_df.copy()

    # 確保 index 為 DatetimeIndex
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    # 聚合為週K（W-FRI：以週五為錨點，台灣股市週一至週五交易）
    weekly = (
        df.resample("W-FRI")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["close"])
    )

    if len(weekly) < 3:
        return pd.DataFrame()

    close = weekly["close"]

    # 週線 SMA(13 週 ≈ 季線)
    weekly["sma_13"] = SMAIndicator(close=close, n=13).sma_indicator()

    # 週線 RSI(14 週)
    weekly["rsi_14"] = RSIIndicator(close=close, n=14).rsi()

    # 週線 MACD(12, 26, 9)
    macd_ind = MACD(close=close, n_slow=26, n_fast=12, n_sign=9)
    weekly["macd"] = macd_ind.macd()
    weekly["macd_signal"] = macd_ind.macd_signal()
    weekly["macd_hist"] = macd_ind.macd_diff()

    return weekly


def calc_rsi14_from_series(closes: pd.Series) -> float:
    """從收盤價序列計算 RSI14（取最後一個有效值）。

    使用 Wilder's smoothing（alpha=1/14 EWM）。
    長度不足 15 時回傳 50.0（中性值）。

    Args:
        closes: 收盤價序列（pd.Series，已按日期升序排列）

    Returns:
        RSI14 值（0.0 ~ 100.0），資料不足時回傳 50.0
    """
    if len(closes) < 15:
        return 50.0

    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    last_gain = float(avg_gain.iloc[-1])
    last_loss = float(avg_loss.iloc[-1])

    if last_loss == 0.0:
        return 100.0 if last_gain > 0 else 50.0

    rs = last_gain / last_loss
    return round(100.0 - 100.0 / (1.0 + rs), 2)
