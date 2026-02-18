"""SMA 均線交叉策略。

買入: 快線上穿慢線（黃金交叉）
賣出: 快線下穿慢線（死亡交叉）
"""

from __future__ import annotations

import pandas as pd

from src.strategy.base import Strategy


class SMACrossStrategy(Strategy):
    """SMA 均線交叉策略。"""

    def __init__(
        self,
        stock_id: str,
        start_date: str,
        end_date: str,
        fast: int = 10,
        slow: int = 20,
    ) -> None:
        super().__init__(stock_id, start_date, end_date)
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"sma_cross_{self.fast}x{self.slow}"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast_col = f"sma_{self.fast}"
        slow_col = f"sma_{self.slow}"

        if fast_col not in data.columns or slow_col not in data.columns:
            raise ValueError(f"缺少指標欄位: {fast_col} 或 {slow_col}，請先執行 python main.py compute")

        fast_sma = data[fast_col]
        slow_sma = data[slow_col]

        # 計算交叉: 前一天快<慢 → 今天快>慢 = 黃金交叉 (買入)
        prev_diff = fast_sma.shift(1) - slow_sma.shift(1)
        curr_diff = fast_sma - slow_sma

        signals = pd.Series(0, index=data.index)
        signals[(prev_diff <= 0) & (curr_diff > 0)] = 1  # 黃金交叉 → 買
        signals[(prev_diff >= 0) & (curr_diff < 0)] = -1  # 死亡交叉 → 賣

        return signals
