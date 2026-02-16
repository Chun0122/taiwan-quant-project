"""策略抽象基類 — 定義策略的共用介面與資料載入邏輯。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import DailyPrice, TechnicalIndicator

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """交易策略抽象基類。

    子類只需實作 generate_signals() 和 name 屬性。
    """

    def __init__(self, stock_id: str, start_date: str, end_date: str) -> None:
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self._data: pd.DataFrame | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名稱，例如 'sma_cross_10x20'。"""
        ...

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """產生交易訊號。

        Args:
            data: 包含 OHLCV + 技術指標的寬表 DataFrame (index=日期)

        Returns:
            pd.Series: 訊號序列，1=買入, -1=賣出, 0=持有
        """
        ...

    def load_data(self) -> pd.DataFrame:
        """從 DB 載入日K線 + 技術指標，合併成寬表。"""
        if self._data is not None:
            return self._data

        # 載入日K線
        with get_session() as session:
            prices = session.execute(
                select(DailyPrice)
                .where(DailyPrice.stock_id == self.stock_id)
                .where(DailyPrice.date >= self.start_date)
                .where(DailyPrice.date <= self.end_date)
                .order_by(DailyPrice.date)
            ).scalars().all()

        if not prices:
            logger.warning("[%s] 無日K線資料", self.stock_id)
            return pd.DataFrame()

        df = pd.DataFrame([
            {"date": r.date, "open": r.open, "high": r.high,
             "low": r.low, "close": r.close, "volume": r.volume}
            for r in prices
        ]).set_index("date")

        # 載入技術指標並 pivot 成寬表
        with get_session() as session:
            indicators = session.execute(
                select(TechnicalIndicator)
                .where(TechnicalIndicator.stock_id == self.stock_id)
                .where(TechnicalIndicator.date >= self.start_date)
                .where(TechnicalIndicator.date <= self.end_date)
                .order_by(TechnicalIndicator.date)
            ).scalars().all()

        if indicators:
            df_ind = pd.DataFrame([
                {"date": r.date, "name": r.name, "value": r.value}
                for r in indicators
            ])
            df_wide = df_ind.pivot_table(index="date", columns="name", values="value")
            df = df.join(df_wide, how="left")

        self._data = df
        logger.info("[%s] 載入 %d 筆交易日資料", self.stock_id, len(df))
        return df
