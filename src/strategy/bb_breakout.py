"""Bollinger Band 突破策略。

買入: 價格從下軌反彈（超賣回升）
賣出: 價格從上軌回落（超買回跌）
"""

from __future__ import annotations

import pandas as pd

from src.strategy.base import Strategy


class BollingerBandBreakoutStrategy(Strategy):
    """布林通道突破策略。"""

    def __init__(
        self,
        stock_id: str,
        start_date: str,
        end_date: str,
        period: int = 20,
        std_dev: int = 2,
        adjust_dividend: bool = False,
    ) -> None:
        super().__init__(stock_id, start_date, end_date, adjust_dividend=adjust_dividend)
        self.period = period
        self.std_dev = std_dev

    @property
    def name(self) -> str:
        return f"bb_breakout_{self.period}_{self.std_dev}"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        upper_col = "bb_upper"
        lower_col = "bb_lower"

        if upper_col not in data.columns or lower_col not in data.columns:
            raise ValueError(f"缺少指標欄位: {upper_col} 或 {lower_col}，請先執行 python main.py compute")

        close = data["close"]
        upper = data[upper_col]
        lower = data[lower_col]
        prev_close = close.shift(1)

        signals = pd.Series(0, index=data.index)

        # 價格從下軌反彈 → 買入
        signals[(prev_close <= lower.shift(1)) & (close > lower)] = 1

        # 價格從上軌回落 → 賣出
        signals[(prev_close >= upper.shift(1)) & (close < upper)] = -1

        return signals
