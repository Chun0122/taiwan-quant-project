"""MACD 交叉策略。

買入: MACD 線上穿訊號線（黃金交叉）
賣出: MACD 線下穿訊號線（死亡交叉）
"""

from __future__ import annotations

import pandas as pd

from src.strategy.base import Strategy


class MACDCrossStrategy(Strategy):
    """MACD 交叉策略。"""

    def __init__(
        self,
        stock_id: str,
        start_date: str,
        end_date: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> None:
        super().__init__(stock_id, start_date, end_date)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    @property
    def name(self) -> str:
        return f"macd_cross_{self.fast}_{self.slow}_{self.signal_period}"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        macd_col = "macd"
        signal_col = "macd_signal"

        if macd_col not in data.columns or signal_col not in data.columns:
            raise ValueError(f"缺少指標欄位: {macd_col} 或 {signal_col}，請先執行 python main.py compute")

        macd_line = data[macd_col]
        signal_line = data[signal_col]

        prev_diff = macd_line.shift(1) - signal_line.shift(1)
        curr_diff = macd_line - signal_line

        signals = pd.Series(0, index=data.index)

        # MACD 上穿 Signal → 買入
        signals[(prev_diff <= 0) & (curr_diff > 0)] = 1

        # MACD 下穿 Signal → 賣出
        signals[(prev_diff >= 0) & (curr_diff < 0)] = -1

        return signals
