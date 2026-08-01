"""設定管理模組 — 載入 config/ 下的設定檔並提供統一存取介面。

A5 拆檔（2026-07-05）：設定拆為兩檔——
  - config/quant_params.yaml：量化參數與非機密設定（**進版控**，供決策可重放溯源）
  - config/secrets.yaml：token/webhook 等機密（gitignored）
載入時 secrets 深度覆蓋 quant_params。舊單檔 config/settings.yaml 若仍存在，
則以 legacy 模式照舊載入並提示遷移（過渡保護，一個版本週期後移除）。
"""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Legacy 單檔設定（A5 拆檔前）；存在時優先以舊模式載入
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
# A5 雙檔模式：量化參數（進版控）+ 機密（gitignored）
QUANT_PARAMS_PATH = PROJECT_ROOT / "config" / "quant_params.yaml"
SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.yaml"


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


class ICGovernanceConfig(BaseModel):
    """IC 驅動的自動調整開關（P0 #17，2026-08-01）。

    **兩者預設關閉＝凍結**，直到 B2（全候選池因子落庫）落地後重新評估。
    凍結期間 scanner 仍照常計算並記錄「原本會做的調整」，只是不套用。

    凍結理由（`logs/audit_discover_20260731/REPORT.md` §6）：

    - `auto_weight_adjust`（E2b `_apply_ic_weight_adjustment`）：歸一化會把被打壓
      維度釋出的權重**全數塞給從未被量測的維度**。實測 2026-07-31 value 模式
      `fundamental 0.550→0.313`（IC −0.1393）而 `valuation 0.150→0.338`
      （**IC=N/A**——該維度不落庫 `DiscoveryRecord`，永遠算不出 IC），權重 2.25×。
      淨效果＝IC 治理越積極，決策權越集中到唯一沒被檢驗的維度。
      且依據本身不穩：value `fundamental` IC 四個交易日 `+0.155→+0.105→+0.034→−0.139`，
      擺幅全在 n≈100 的標準誤（≈1/√n≈0.10）之內。
    - `auto_score_transform`（E2c `compute_ic_aware_score_transform`）：翻轉/中性化
      門檻 `|IC| ≤ 0.02`、`min_samples=50`，在 SE≈0.10–0.14 下等同**隨機翻轉維度方向**。
      另 `news_score` 有 63–81% 恰為 `fillna(0.5)` 填補值，對七成是常數的變數算
      秩相關本無意義，卻正在改變其他三個維度的相對權重。

    解除條件：B2 落地（IC 擺脫 top-N 截斷樣本）＋ 未量測維度納入落庫 ＋ 調整幅度
    改以標準誤為基準的顯著性判定。屬 B11 完整版範圍。
    """

    auto_weight_adjust: bool = False  # E2b：IC×影響力 四維度權重自動調整
    auto_score_transform: bool = False  # E2c：IC-aware 分數翻轉 / 中性化 / dampen


class QuantConfig(BaseModel):
    """量化參數外部化（D2）— 可在 settings.yaml 的 quant 區塊覆蓋預設值。"""

    trading_cost: TradingCostConfig = TradingCostConfig()
    atr_multiplier: AtrMultiplierConfig = AtrMultiplierConfig()
    score_threshold: ScoreThresholdConfig = ScoreThresholdConfig()
    risk_budget: RiskBudgetConfig = RiskBudgetConfig()
    rotation_cost: RotationCostConfig = RotationCostConfig()
    ic_governance: ICGovernanceConfig = ICGovernanceConfig()


class DashboardConfig(BaseModel):
    """iOS 監控 App 日報 JSON（export-dashboard）相關設定。"""

    position_timeseries_days: int = 14  # 持倉/Watch 個股小走勢圖窗口天數
    portfolio_review_lookback_days: int = 90  # portfolio_review 撈最近 N 天 snapshot 計算指標


class BackupConfig(BaseModel):
    """DB 備份設定（P0 止血包 #1，2026-07-05）。

    offsite_dir：異地副本目錄（預設 iCloud Drive，macOS 自動同步上雲）；
    設為空字串停用異地副本。retention_days：本地與異地共用的保留天數。
    """

    offsite_dir: str = "~/Library/Mobile Documents/com~apple~CloudDocs/taiwan-quant-backups"
    retention_days: int = 7


class MonitoringConfig(BaseModel):
    """運維監控設定（P0 止血包 #5，2026-07-05）。

    healthchecks_url：healthchecks.io 的完整 ping URL（如
    https://hc-ping.com/<uuid>）。morning-routine 起跑 ping /start、
    成功 ping 本體、有步驟失敗 ping /fail；健檢平台端設定排程 + grace period，
    未收到 ping 即發告警（dead-man switch）。空字串 = 停用。
    """

    healthchecks_url: str = ""


class Settings(BaseModel):
    finmind: FinMindConfig = FinMindConfig()
    database: DatabaseConfig = DatabaseConfig()
    fetcher: FetcherConfig = FetcherConfig()
    logging: LoggingConfig = LoggingConfig()
    discord: DiscordWebhookConfig = DiscordWebhookConfig()
    anthropic: AnthropicConfig = AnthropicConfig()
    quant: QuantConfig = QuantConfig()
    dashboard: DashboardConfig = DashboardConfig()
    backup: BackupConfig = BackupConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    @model_validator(mode="after")
    def _validate_critical_settings(self) -> "Settings":
        """啟動時驗證關鍵設定 — 在 Settings 層級執行，避免子模型預設值觸發誤警告。"""
        if not self.finmind.api_token:
            logger.warning("finmind.api_token 未設定 — 僅 TWSE/TPEX 免費端點可用，FinMind 逐股查詢將失敗")
        if self.discord.enabled and not self.discord.webhook_url:
            logger.warning("discord.webhook_url 未設定但 enabled=True — 通知將靜默失敗")
        return self


def _read_yaml(path: Path) -> dict:
    """讀取 YAML 檔為 dict；檔案不存在或內容為空時回傳空 dict。"""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """遞迴合併兩個 dict：overlay 覆蓋 base，巢狀 dict 逐層合併（不修改原 dict）。"""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(
    path: Path | None = None,
    quant_params_path: Path = QUANT_PARAMS_PATH,
    secrets_path: Path = SECRETS_PATH,
) -> Settings:
    """載入設定。

    三種模式：
      1. 明確指定 ``path``（測試/工具用）：單檔載入，不存在時回預設值。
      2. legacy ``config/settings.yaml`` 存在：照舊單檔載入 + 警告提示遷移。
      3. 雙檔模式：quant_params.yaml 為底、secrets.yaml 深度覆蓋；兩檔皆缺回預設值。
    """
    if path is not None:
        return Settings(**_read_yaml(path))
    if CONFIG_PATH.exists():
        logger.warning(
            "偵測到 legacy config/settings.yaml — 以舊模式載入。"
            "請遷移至 quant_params.yaml + secrets.yaml 後刪除本檔（A5 拆檔）"
        )
        return Settings(**_read_yaml(CONFIG_PATH))
    merged = _deep_merge(_read_yaml(quant_params_path), _read_yaml(secrets_path))
    return Settings(**merged)


# 全域設定單例
settings = load_settings()
