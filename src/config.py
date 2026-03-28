"""設定管理模組 — 載入 config/settings.yaml 並提供統一存取介面。"""

from pathlib import Path

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class FinMindConfig(BaseModel):
    api_url: str = "https://api.finmindtrade.com/api/v4/data"
    api_token: str = ""


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///data/stock.db"


class FetcherConfig(BaseModel):
    default_start_date: str = "2020-01-01"
    watchlist: list[str] = ["2330"]


class LoggingConfig(BaseModel):
    level: str = "INFO"


class DiscordWebhookConfig(BaseModel):
    webhook_url: str = ""
    username: str = "台股量化系統"
    enabled: bool = True


class AnthropicConfig(BaseModel):
    api_key: str = ""
    model: str = "claude-sonnet-4-6"


class TradingCostConfig(BaseModel):
    """交易成本參數（預設值與 src/constants.py 一致）。"""

    commission_rate: float = 0.001425
    tax_rate: float = 0.003
    slippage_rate: float = 0.0005
    slippage_impact_coeff: float = 0.5
    liquidity_participation_limit: float = 0.05


class AtrMultiplierConfig(BaseModel):
    """ATR 倍數參數（預設值與 src/entry_exit.py REGIME_ATR_PARAMS 一致）。"""

    bull_stop: float = 1.5
    bull_target: float = 3.0
    sideways_stop: float = 2.0
    sideways_target: float = 2.5
    bear_stop: float = 2.5
    bear_target: float = 2.0
    crisis_stop: float = 3.0
    crisis_target: float = 1.5


class ScoreThresholdConfig(BaseModel):
    """各 Regime 的最低評分門檻（預設值與 scanner MIN_SCORE_THRESHOLDS 一致）。"""

    bull: float = 0.45
    sideways: float = 0.50
    bear: float = 0.55
    crisis: float = 0.60


class RiskBudgetConfig(BaseModel):
    """組合風險預算參數（Portfolio Heat + Correlation Budget）。"""

    max_heat: float = 0.12  # 組合最大風險上限（0.12 = 12%）
    per_position_risk_cap: float = 0.03  # 單筆最大風險估算（無停損時）
    correlation_threshold: float = 0.7  # 高相關判定門檻
    correlation_penalty: float = 0.5  # 高相關時部位縮減比例


class QuantConfig(BaseModel):
    """量化參數外部化（D2）— 可在 settings.yaml 的 quant 區塊覆蓋預設值。"""

    trading_cost: TradingCostConfig = TradingCostConfig()
    atr_multiplier: AtrMultiplierConfig = AtrMultiplierConfig()
    score_threshold: ScoreThresholdConfig = ScoreThresholdConfig()
    risk_budget: RiskBudgetConfig = RiskBudgetConfig()


class Settings(BaseModel):
    finmind: FinMindConfig = FinMindConfig()
    database: DatabaseConfig = DatabaseConfig()
    fetcher: FetcherConfig = FetcherConfig()
    logging: LoggingConfig = LoggingConfig()
    discord: DiscordWebhookConfig = DiscordWebhookConfig()
    anthropic: AnthropicConfig = AnthropicConfig()
    quant: QuantConfig = QuantConfig()


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    """從 YAML 檔案載入設定，檔案不存在時使用預設值。"""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return Settings(**raw)
    return Settings()


# 全域設定單例
settings = load_settings()
