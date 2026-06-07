"""設定管理模組 — 載入 config/settings.yaml 並提供統一存取介面。"""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class FinMindConfig(BaseModel):
    api_url: str = "https://api.finmindtrade.com/api/v4/data"
    api_token: str = ""


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///data/stock.db"

    @model_validator(mode="after")
    def _validate_url(self) -> "DatabaseConfig":
        if not self.url:
            raise ValueError("database.url 不可為空")
        return self


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


class RotationCostModeOverride(BaseModel):
    """單一 mode 的成本閘門覆蓋；欄位為 None 時沿用全域 RotationCostConfig 值。

    動機（投研 2026-06-07）：成本閘門對不同策略效果相反 —— Gate B(score_gap)
    是動量 alpha 主來源（抱住贏家），卻會勒死波段換手（swing 全開 35%→9%）。
    per-mode 覆蓋讓動量維持嚴格、波段放寬 score_gap。
    """

    enabled: bool | None = None
    min_hold_days: int | None = None
    score_gap_threshold: float | None = None
    weekly_swap_cap: int | None = None


class RotationCostConfig(BaseModel):
    """Rotation 成本閘門參數（降低高頻換手帶來的成本拖累）。

    三道閘門（由便宜到昂貴排序）：
      A. min_hold_days：holding_days 的安全下限（防止極短線換手）
      B. score_gap_threshold：expired 時需要新候選 score 贏現持倉此差距才換
      C. weekly_swap_cap：每 ISO 週最多幾筆 holding_expired 賣出（不含 stop_loss）

    per_mode：以 mode 名（momentum/swing/value/dividend/growth/all）覆蓋上述全域值，
    未列出的 mode 沿用全域。manager 以 for_mode(portfolio.mode) 取解析後設定。
    """

    enabled: bool = False  # 預設關閉以維持向後相容；明確啟用後才生效
    min_hold_days: int = 3  # 最短持有天數（以交易日計）
    score_gap_threshold: float = 0.05  # 切換所需 composite_score 差距
    weekly_swap_cap: int = 4  # 每週非止損換手上限（stop_loss/crisis 不計）
    per_mode: dict[str, RotationCostModeOverride] = {}  # mode → 覆蓋；空 = 全 mode 用全域（向後相容）

    def for_mode(self, mode: str | None) -> "RotationCostConfig":
        """回傳指定 mode 解析後的閘門設定（per_mode 覆蓋全域）。

        回傳新的 RotationCostConfig（per_mode 留空），manager 以 .enabled / .min_hold_days
        等屬性讀取，與既有程式完全相容；無對應 mode 覆蓋時直接回傳 self。
        """
        ov = self.per_mode.get(mode) if mode else None
        if ov is None:
            return self
        return RotationCostConfig(
            enabled=self.enabled if ov.enabled is None else ov.enabled,
            min_hold_days=self.min_hold_days if ov.min_hold_days is None else ov.min_hold_days,
            score_gap_threshold=(
                self.score_gap_threshold if ov.score_gap_threshold is None else ov.score_gap_threshold
            ),
            weekly_swap_cap=self.weekly_swap_cap if ov.weekly_swap_cap is None else ov.weekly_swap_cap,
        )


class QuantConfig(BaseModel):
    """量化參數外部化（D2）— 可在 settings.yaml 的 quant 區塊覆蓋預設值。"""

    trading_cost: TradingCostConfig = TradingCostConfig()
    atr_multiplier: AtrMultiplierConfig = AtrMultiplierConfig()
    score_threshold: ScoreThresholdConfig = ScoreThresholdConfig()
    risk_budget: RiskBudgetConfig = RiskBudgetConfig()
    rotation_cost: RotationCostConfig = RotationCostConfig()


class DashboardConfig(BaseModel):
    """iOS 監控 App 日報 JSON（export-dashboard）相關設定。"""

    position_timeseries_days: int = 14  # 持倉/Watch 個股小走勢圖窗口天數
    portfolio_review_lookback_days: int = 90  # portfolio_review 撈最近 N 天 snapshot 計算指標


class Settings(BaseModel):
    finmind: FinMindConfig = FinMindConfig()
    database: DatabaseConfig = DatabaseConfig()
    fetcher: FetcherConfig = FetcherConfig()
    logging: LoggingConfig = LoggingConfig()
    discord: DiscordWebhookConfig = DiscordWebhookConfig()
    anthropic: AnthropicConfig = AnthropicConfig()
    quant: QuantConfig = QuantConfig()
    dashboard: DashboardConfig = DashboardConfig()

    @model_validator(mode="after")
    def _validate_critical_settings(self) -> "Settings":
        """啟動時驗證關鍵設定 — 在 Settings 層級執行，避免子模型預設值觸發誤警告。"""
        if not self.finmind.api_token:
            logger.warning("finmind.api_token 未設定 — 僅 TWSE/TPEX 免費端點可用，FinMind 逐股查詢將失敗")
        if self.discord.enabled and not self.discord.webhook_url:
            logger.warning("discord.webhook_url 未設定但 enabled=True — 通知將靜默失敗")
        return self


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    """從 YAML 檔案載入設定，檔案不存在時使用預設值。"""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return Settings(**raw)
    return Settings()


# 全域設定單例
settings = load_settings()
