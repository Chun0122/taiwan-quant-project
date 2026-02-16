"""Buy & Hold 策略 — 基準比較用。

在第一天買入，持有到最後一天賣出。
"""

from __future__ import annotations

import pandas as pd

from src.strategy.base import Strategy


class BuyAndHoldStrategy(Strategy):
    """買入持有策略（基準策略）。"""

    def __init__(
        self,
        stock_id: str,
        start_date: str,
        end_date: str,
    ) -> None:
        super().__init__(stock_id, start_date, end_date)

    @property
    def name(self) -> str:
        return "buy_and_hold"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """第一天買入，最後一天賣出。"""
        signals = pd.Series(0, index=data.index)

        if len(data) > 0:
            signals.iloc[0] = 1    # 第一天買入
            signals.iloc[-1] = -1  # 最後一天賣出

        return signals
