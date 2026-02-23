"""RSI 超買超賣策略。

買入: RSI 從下方突破超賣線
賣出: RSI 從上方跌破超買線
"""

from __future__ import annotations

import pandas as pd

from src.strategy.base import Strategy


class RSIThresholdStrategy(Strategy):
    """RSI 超買超賣策略。"""

    def __init__(
        self,
        stock_id: str,
        start_date: str,
        end_date: str,
        oversold: int = 30,
        overbought: int = 70,
        adjust_dividend: bool = False,
    ) -> None:
        super().__init__(stock_id, start_date, end_date, adjust_dividend=adjust_dividend)
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return f"rsi_{self.oversold}_{self.overbought}"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        rsi_col = "rsi_14"

        if rsi_col not in data.columns:
            raise ValueError(f"缺少指標欄位: {rsi_col}，請先執行 python main.py compute")

        rsi = data[rsi_col]
        prev_rsi = rsi.shift(1)

        signals = pd.Series(0, index=data.index)
        # RSI 從下方突破超賣線 → 買入
        signals[(prev_rsi <= self.oversold) & (rsi > self.oversold)] = 1
        # RSI 從上方跌破超買線 → 賣出
        signals[(prev_rsi >= self.overbought) & (rsi < self.overbought)] = -1

        return signals
