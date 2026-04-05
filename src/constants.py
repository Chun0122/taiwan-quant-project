"""集中管理全系統共用常數，避免 magic number 散落各模組。"""

from __future__ import annotations

# ── 交易成本 ─────────────────────────────────────────────────────────
COMMISSION_RATE = 0.001425  # 手續費 0.1425%
TAX_RATE = 0.003  # 交易稅 0.3%（賣出時）
SLIPPAGE_RATE = 0.0005  # 滑價 0.05%
SLIPPAGE_IMPACT_COEFF = 0.5  # 動態滑價衝擊係數 k（slippage = base + k / sqrt(volume)）
SLIPPAGE_MAX_PCT = 0.01  # 滑價上限 1%（防止低流動性股票滑價爆炸）
SLIPPAGE_SPREAD_WEIGHT = 0.5  # OHLC spread 估算權重（spread proxy = (high-low)/close × weight）
SELL_SLIPPAGE_MULTIPLIER = 1.3  # 賣出滑價放大係數（恐慌賣出時滑價通常高於買入）
LIQUIDITY_PARTICIPATION_LIMIT = 0.05  # 流動性約束：單筆交易量 ≤ 當日成交量 × 此比例

# ── DB / ETL ─────────────────────────────────────────────────────────
UPSERT_BATCH_SIZE = 80  # SQLite 變數上限安全批次大小
API_SLEEP_FINMIND = 0.5  # FinMind API 請求間隔（秒）
API_SLEEP_TWSE = 3.0  # TWSE/TPEX 請求間隔（秒）

# ── 籌碼異動預設門檻 ─────────────────────────────────────────────────
DEFAULT_VOL_MULT = 2.0  # 量能暴增倍數
DEFAULT_INST_THRESHOLD = 3_000_000  # 外資大買超金額門檻
DEFAULT_SBL_SIGMA = 2.0  # 借券激增 σ 門檻
DEFAULT_HHI_THRESHOLD = 0.4  # 主力集中度 HHI 門檻
DEFAULT_DT_THRESHOLD = 0.3  # 隔日沖風險門檻

# ── VIX 危機偵測 ─────────────────────────────────────────────────────
VIX_STOCK_ID: str = "TW_VIX"  # DailyPrice 中的 stock_id（台灣 VIX）
CRISIS_VIX_LEVEL: float = 30.0  # 台灣 VIX 絕對值門檻
CRISIS_VIX_DAILY_CHANGE: float = 0.25  # 台灣 VIX 單日漲幅門檻 (25%)
CRISIS_SINGLE_DAY_DROP: float = -0.025  # TAIEX 單日跌幅門檻 (-2.5%)

# ── 美國 VIX (CBOE ^VIX) ────────────────────────────────────────────
US_VIX_STOCK_ID: str = "US_VIX"  # DailyPrice 中的 stock_id（美國 VIX）
CRISIS_US_VIX_LEVEL: float = 30.0  # 美國 VIX 絕對值門檻
CRISIS_US_VIX_DAILY_CHANGE: float = 0.25  # 美國 VIX 單日漲幅門檻 (25%)

# ── 組合風險預算（Portfolio Heat）─────────────────────────────────────
MAX_PORTFOLIO_HEAT: float = 0.12  # 組合最大風險上限 12%
PER_POSITION_RISK_CAP: float = 0.03  # 單筆最大風險估算上限 3%（無停損時使用）

# ── 相關性預算（Correlation Budget）──────────────────────────────────
CORRELATION_THRESHOLD: float = 0.7  # 高相關判定門檻（bull/sideways）
CORRELATION_PENALTY: float = 0.5  # 高相關時部位縮減比例（bull/sideways）
CORRELATION_THRESHOLD_BEAR: float = 0.6  # bear 市高相關判定門檻
CORRELATION_PENALTY_BEAR: float = 0.4  # bear 市部位縮減比例
CORRELATION_THRESHOLD_CRISIS: float = 0.5  # crisis 高相關判定門檻
CORRELATION_PENALTY_CRISIS: float = 0.3  # crisis 部位縮減比例

# ── 最大回撤熔斷（Max Drawdown Kill Switch）──────────────────────────
MAX_DRAWDOWN_LIQUIDATE_PCT: float = 25.0  # 組合回撤超過此值(%)強制平倉所有部位

# ── Kelly Criterion ─────────────────────────────────────────────────────
KELLY_CONFIDENCE_DENOMINATOR: int = 100  # 信心縮放分母：confidence = min(1, trades / N)
KELLY_MAX_FRACTION: float = 0.20  # Kelly 比例硬上限 20%（防止少量交易過度激進）

# ── 公告衰減常數（News Decay）─────────────────────────────────────────
# 結構性事件（董監改選/庫藏股）衰減慢，一般性事件衰減快
NEWS_DECAY_STRUCTURAL: float = 0.07  # 半衰期 ~10 天（ln2/0.07≈9.9）
NEWS_DECAY_TRANSIENT: float = 0.15  # 半衰期 ~4.6 天（ln2/0.15≈4.6）
NEWS_DECAY_DEFAULT: float = 0.12  # 中性事件預設（半衰期 ~5.8 天）
# 結構性事件類型
NEWS_STRUCTURAL_TYPES: frozenset[str] = frozenset({"governance_change", "buyback"})
# 快速衰減事件類型
NEWS_TRANSIENT_TYPES: frozenset[str] = frozenset({"revenue", "general"})
# 公告載入窗口（天）
NEWS_LOAD_WINDOW_DAYS: int = 15

# ── Regime 預設值 ───────────────────────────────────────────────────────
REGIME_FALLBACK_DEFAULT: str = "sideways"  # Regime 偵測失敗時的安全預設值

# ── 回測 Regime 自適應部位乘數 ─────────────────────────────────────────
# ── 漲跌停模擬 ─────────────────────────────────────────────────────────
LIMIT_PRICE_PCT: float = 0.10  # 台股漲跌停幅度 10%
LIMIT_DETECT_THRESHOLD: float = 0.095  # 偵測門檻（略低於 10% 以涵蓋四捨五入）

REGIME_POSITION_MULTIPLIERS: dict[str, float] = {
    "bull": 1.0,  # 多頭：全額曝險
    "sideways": 0.8,  # 盤整：縮減 20%
    "bear": 0.6,  # 空頭：縮減 40%
    "crisis": 0.3,  # 危機：僅 30% 曝險
}

# ── Universe Filter Regime 自適應調整 ─────────────────────────────────
# turnover_multiplier: 乘在流動性門檻上（<1 放寬、>1 收緊）
# volume_ratio_override: 覆寫 UniverseConfig.volume_ratio_min（None = 跳過量比過濾）
REGIME_UNIVERSE_ADJUSTMENTS: dict[str, dict] = {
    "bull": {"turnover_multiplier": 0.8},  # 放寬流動性，更多中型股進入
    "sideways": {"turnover_multiplier": 1.0},  # 預設不調整
    "bear": {"turnover_multiplier": 1.3, "volume_ratio_override": None},  # 收緊流動性、放寬量比
    "crisis": {"turnover_multiplier": 1.5, "volume_ratio_override": None},  # 嚴格流動性、跳過量比
}
