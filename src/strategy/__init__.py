"""策略模組 — 提供策略註冊表供 CLI 查找。"""

from src.strategy.sma_cross import SMACrossStrategy
from src.strategy.rsi_threshold import RSIThresholdStrategy

STRATEGY_REGISTRY: dict[str, type] = {
    "sma_cross": SMACrossStrategy,
    "rsi_threshold": RSIThresholdStrategy,
}
