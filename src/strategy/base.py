"""策略抽象基類 — 定義策略的共用介面與資料載入邏輯。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import DailyPrice, Dividend, TechnicalIndicator

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """交易策略抽象基類。

    子類只需實作 generate_signals() 和 name 屬性。
    """

    def __init__(
        self,
        stock_id: str,
        start_date: str,
        end_date: str,
        adjust_dividend: bool = False,
    ) -> None:
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self.adjust_dividend = adjust_dividend
        self._data: pd.DataFrame | None = None
        self._dividends: pd.DataFrame | None = None

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
        """從 DB 載入日K線 + 技術指標，合併成寬表。

        當 adjust_dividend=True 時，會載入除權息資料並：
        1. 回溯調整 OHLC 價格（保留 raw_* 原始價格）
        2. 從調整後價格重新計算技術指標（繞過 EAV 表）
        """
        if self._data is not None:
            return self._data

        # 載入日K線
        with get_session() as session:
            prices = (
                session.execute(
                    select(DailyPrice)
                    .where(DailyPrice.stock_id == self.stock_id)
                    .where(DailyPrice.date >= self.start_date)
                    .where(DailyPrice.date <= self.end_date)
                    .order_by(DailyPrice.date)
                )
                .scalars()
                .all()
            )

        if not prices:
            logger.warning("[%s] 無日K線資料", self.stock_id)
            return pd.DataFrame()

        df = pd.DataFrame(
            [
                {"date": r.date, "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
                for r in prices
            ]
        ).set_index("date")

        if self.adjust_dividend:
            # Layer 1: 除權息回溯調整 → 內聯計算指標
            dividends = self._load_dividends()
            self._dividends = dividends
            df = self._apply_dividend_adjustment(df, dividends)

            from src.features.indicators import compute_indicators_from_df

            df_indicators = compute_indicators_from_df(df)
            df = df.join(df_indicators, how="left")
        else:
            # 原有邏輯：從 EAV 表載入預計算指標
            with get_session() as session:
                indicators = (
                    session.execute(
                        select(TechnicalIndicator)
                        .where(TechnicalIndicator.stock_id == self.stock_id)
                        .where(TechnicalIndicator.date >= self.start_date)
                        .where(TechnicalIndicator.date <= self.end_date)
                        .order_by(TechnicalIndicator.date)
                    )
                    .scalars()
                    .all()
                )

            if indicators:
                df_ind = pd.DataFrame([{"date": r.date, "name": r.name, "value": r.value} for r in indicators])
                df_wide = df_ind.pivot_table(index="date", columns="name", values="value")
                df = df.join(df_wide, how="left")

        self._data = df
        logger.info("[%s] 載入 %d 筆交易日資料 (adjust_dividend=%s)", self.stock_id, len(df), self.adjust_dividend)
        return df

    # ------------------------------------------------------------------ #
    #  除權息調整
    # ------------------------------------------------------------------ #

    def _load_dividends(self) -> pd.DataFrame:
        """載入除權息資料。"""
        with get_session() as session:
            rows = (
                session.execute(
                    select(Dividend)
                    .where(Dividend.stock_id == self.stock_id)
                    .where(Dividend.date >= self.start_date)
                    .where(Dividend.date <= self.end_date)
                    .order_by(Dividend.date)
                )
                .scalars()
                .all()
            )
        if not rows:
            return pd.DataFrame(columns=["cash_dividend", "stock_dividend"])

        df = pd.DataFrame(
            [
                {
                    "date": r.date,
                    "cash_dividend": r.cash_dividend or 0.0,
                    "stock_dividend": r.stock_dividend or 0.0,
                }
                for r in rows
            ]
        ).set_index("date")
        return df

    def _apply_dividend_adjustment(self, df: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
        """套用除權息回溯價格調整。

        保留 raw_open/raw_high/raw_low/raw_close 為原始價格，
        調整 open/high/low/close 為還原權息後的連續價格。

        調整因子公式（每個除權息日）：
            factor = (prev_close - cash_dividend) / prev_close / (1 + stock_dividend / 10)
        """
        if dividends.empty:
            return df

        # 保留原始價格
        df = df.copy()
        df["raw_open"] = df["open"].copy()
        df["raw_high"] = df["high"].copy()
        df["raw_low"] = df["low"].copy()
        df["raw_close"] = df["close"].copy()

        price_cols = ["open", "high", "low", "close"]
        dates = df.index

        for ex_date in sorted(dividends.index, reverse=True):
            if ex_date not in dates:
                continue

            cash_div = dividends.loc[ex_date, "cash_dividend"]
            stock_div = dividends.loc[ex_date, "stock_dividend"]
            if cash_div == 0 and stock_div == 0:
                continue

            # 前一交易日收盤價
            ex_idx = dates.get_loc(ex_date)
            if ex_idx == 0:
                continue

            prev_close = df.iloc[ex_idx - 1]["raw_close"]
            if prev_close <= 0:
                continue

            # 計算因子
            factor = (prev_close - cash_div) / prev_close
            if stock_div > 0:
                factor /= 1 + stock_div / 10

            if factor <= 0 or factor >= 1.5:
                logger.warning(
                    "[%s] 除權息因子異常 (%.4f) on %s, cash=%.2f stock=%.2f, 跳過",
                    self.stock_id,
                    factor,
                    ex_date,
                    cash_div,
                    stock_div,
                )
                continue

            # 調整 ex_date 之前所有 OHLC
            mask = dates < ex_date
            for col in price_cols:
                df.loc[mask, col] = df.loc[mask, col] * factor

        return df
