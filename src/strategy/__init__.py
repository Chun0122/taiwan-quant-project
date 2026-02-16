"""策略模組 — 提供策略註冊表供 CLI 查找。"""

from src.strategy.sma_cross import SMACrossStrategy
from src.strategy.rsi_threshold import RSIThresholdStrategy
from src.strategy.bb_breakout import BollingerBandBreakoutStrategy
from src.strategy.macd_cross import MACDCrossStrategy
from src.strategy.buy_hold import BuyAndHoldStrategy
from src.strategy.multi_factor import MultiFactorStrategy

STRATEGY_REGISTRY: dict[str, type] = {
    "sma_cross": SMACrossStrategy,
    "rsi_threshold": RSIThresholdStrategy,
    "bb_breakout": BollingerBandBreakoutStrategy,
    "macd_cross": MACDCrossStrategy,
    "buy_and_hold": BuyAndHoldStrategy,
    "multi_factor": MultiFactorStrategy,
}
