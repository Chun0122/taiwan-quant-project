"""多因子組合策略。

綜合技術面、籌碼面、基本面因子，加權計算後產生交易訊號。

每個因子產生 +1（看多）/0（中性）/-1（看空）分數，
加權總分 > buy_threshold → 買入，< sell_threshold → 賣出。
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import (
    DailyPrice,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
    TechnicalIndicator,
)
from src.strategy.base import Strategy

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "rsi": 0.2,
    "macd": 0.2,
    "institutional": 0.3,
    "revenue": 0.3,
}


class MultiFactorStrategy(Strategy):
    """多因子組合策略。"""

    def __init__(
        self,
        stock_id: str,
        start_date: str,
        end_date: str,
        weights: dict[str, float] | None = None,
        buy_threshold: float = 0.3,
        sell_threshold: float = -0.3,
    ) -> None:
        super().__init__(stock_id, start_date, end_date)
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    @property
    def name(self) -> str:
        return "multi_factor"

    def load_data(self) -> pd.DataFrame:
        """載入日K線 + 技術指標 + 法人 + 融資券 + 營收，合併成寬表。"""
        if self._data is not None:
            return self._data

        with get_session() as session:
            # 日K線
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

        with get_session() as session:
            # 技術指標
            indicators = (
                session.execute(
                    select(TechnicalIndicator)
                    .where(TechnicalIndicator.stock_id == self.stock_id)
                    .where(TechnicalIndicator.date >= self.start_date)
                    .where(TechnicalIndicator.date <= self.end_date)
                )
                .scalars()
                .all()
            )

            if indicators:
                df_ind = pd.DataFrame([{"date": r.date, "name": r.name, "value": r.value} for r in indicators])
                df_wide = df_ind.pivot_table(index="date", columns="name", values="value")
                df = df.join(df_wide, how="left")

            # 三大法人（pivot by name → 外資/投信/自營商 net）
            institutions = (
                session.execute(
                    select(InstitutionalInvestor)
                    .where(InstitutionalInvestor.stock_id == self.stock_id)
                    .where(InstitutionalInvestor.date >= self.start_date)
                    .where(InstitutionalInvestor.date <= self.end_date)
                )
                .scalars()
                .all()
            )

            if institutions:
                df_inst = pd.DataFrame([{"date": r.date, "name": r.name, "net": r.net} for r in institutions])
                inst_pivot = df_inst.pivot_table(index="date", columns="name", values="net", aggfunc="sum")
                rename_map = {
                    "Foreign_Investor": "foreign_net",
                    "Investment_Trust": "trust_net",
                    "Dealer_self": "dealer_net",
                    "外資": "foreign_net",
                    "投信": "trust_net",
                    "自營商": "dealer_net",
                    "外資及陸資": "foreign_net",
                }
                inst_pivot = inst_pivot.rename(columns=rename_map)
                known = ["foreign_net", "trust_net", "dealer_net"]
                inst_pivot = inst_pivot[[c for c in known if c in inst_pivot.columns]]
                df = df.join(inst_pivot, how="left")

            # 融資融券
            margins = (
                session.execute(
                    select(MarginTrading)
                    .where(MarginTrading.stock_id == self.stock_id)
                    .where(MarginTrading.date >= self.start_date)
                    .where(MarginTrading.date <= self.end_date)
                )
                .scalars()
                .all()
            )

            if margins:
                df_margin = pd.DataFrame(
                    [
                        {"date": r.date, "margin_balance": r.margin_balance, "short_balance": r.short_balance}
                        for r in margins
                    ]
                ).set_index("date")
                df = df.join(df_margin, how="left")

            # 月營收（forward fill 到每日）
            revenues = (
                session.execute(
                    select(MonthlyRevenue)
                    .where(MonthlyRevenue.stock_id == self.stock_id)
                    .where(MonthlyRevenue.date >= self.start_date)
                    .where(MonthlyRevenue.date <= self.end_date)
                    .order_by(MonthlyRevenue.date)
                )
                .scalars()
                .all()
            )

            if revenues:
                df_rev = pd.DataFrame([{"date": r.date, "yoy_growth": r.yoy_growth} for r in revenues]).set_index(
                    "date"
                )
                df = df.join(df_rev, how="left")
                df["yoy_growth"] = df["yoy_growth"].ffill()

        self._data = df
        logger.info("[%s] 多因子策略載入 %d 筆交易日資料", self.stock_id, len(df))
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """產生多因子加權交易訊號。

        各因子計算 +1/0/-1 分數後，依權重加總：
        - RSI: <30→+1, >70→-1, 其他→0
        - MACD: MACD>Signal→+1, MACD<Signal→-1
        - Institutional: 法人合計買超→+1, 賣超→-1
        - Revenue: YoY>20%→+1, YoY<0%→-1

        加權總分 > buy_threshold → 買入, < sell_threshold → 賣出
        """
        signals = pd.Series(0, index=data.index)
        weighted_score = pd.Series(0.0, index=data.index)

        # RSI 因子
        w_rsi = self.weights.get("rsi", 0)
        if w_rsi > 0 and "rsi_14" in data.columns:
            rsi_score = pd.Series(0, index=data.index)
            rsi_score[data["rsi_14"] < 30] = 1
            rsi_score[data["rsi_14"] > 70] = -1
            weighted_score += rsi_score * w_rsi

        # MACD 因子
        w_macd = self.weights.get("macd", 0)
        if w_macd > 0 and "macd" in data.columns and "macd_signal" in data.columns:
            macd_diff = data["macd"] - data["macd_signal"]
            macd_score = pd.Series(0, index=data.index)
            macd_score[macd_diff > 0] = 1
            macd_score[macd_diff < 0] = -1
            weighted_score += macd_score * w_macd

        # 法人因子
        w_inst = self.weights.get("institutional", 0)
        if w_inst > 0:
            net_cols = [c for c in ["foreign_net", "trust_net", "dealer_net"] if c in data.columns]
            if net_cols:
                total_net = data[net_cols].sum(axis=1)
                inst_score = pd.Series(0, index=data.index)
                inst_score[total_net > 0] = 1
                inst_score[total_net < 0] = -1
                weighted_score += inst_score * w_inst

        # 營收因子
        w_rev = self.weights.get("revenue", 0)
        if w_rev > 0 and "yoy_growth" in data.columns:
            rev_score = pd.Series(0, index=data.index)
            rev_score[data["yoy_growth"] > 20] = 1
            rev_score[data["yoy_growth"] < 0] = -1
            weighted_score += rev_score * w_rev

        # 產生買賣訊號
        signals[weighted_score > self.buy_threshold] = 1
        signals[weighted_score < self.sell_threshold] = -1

        return signals
