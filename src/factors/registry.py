"""因子註冊表（metadata-only，phase 1）。

每個因子在 FACTOR_REGISTRY 註冊：
- name：canonical 名稱（discover composite 維度為 4 個 *_score，子因子用領域內名稱）
- category：technical / chip / fundamental / news / valuation / dividend / regime
- factor_type：dimension（4 維 composite）/ sub_factor（組件）/ predicate（screener bool）/ indicator（EAV 持久化）
- description：1-2 行人話
- source_module + source_function：實際定義位置（phase 2 遷移時更新）
- expected_sign："+"（值越大越多頭）/ "-"（值越大越空頭）/ "either"
- used_in_modes：哪些 discover 模式使用（"all" 表示 5 個 mode 共用）
- holding_days_target：因子最佳兌現週期（若已知）
- ic_notes：IC 歷史/穩定度補充（audit 用）

新增因子時的步驟：
  1. 編寫實際計算函式（先放原位置，phase 2 才搬到 src/factors/）
  2. 在此檔的 FACTOR_REGISTRY 註冊
  3. 加 IC 監控（discover/scanner/_functions.py:FACTOR_COLUMNS）
  4. 補 docstring 與 audit notes
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

# 合法 category 集合（與 REGIME_WEIGHTS 對齊）
VALID_CATEGORIES: frozenset[str] = frozenset(
    {"technical", "chip", "fundamental", "news", "valuation", "dividend", "regime"}
)

VALID_FACTOR_TYPES: frozenset[str] = frozenset({"dimension", "sub_factor", "predicate", "indicator"})

VALID_SIGNS: frozenset[str] = frozenset({"+", "-", "either"})


@dataclass(frozen=True)
class FactorSpec:
    """單一因子的中介資料（不含計算邏輯）。

    用於 audit / IC monitor / factor-list CLI，建立 factor name → 定義位置的 SSOT。
    """

    name: str
    category: str
    factor_type: str
    description: str
    source_module: str
    source_function: str | None  # None = 該因子是 module-level 常數或 dataframe column
    expected_sign: str = "+"
    used_in_modes: tuple[str, ...] = field(default_factory=tuple)
    holding_days_target: int | None = None
    ic_notes: str | None = None

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"category 必須為 {sorted(VALID_CATEGORIES)}，got {self.category!r}")
        if self.factor_type not in VALID_FACTOR_TYPES:
            raise ValueError(f"factor_type 必須為 {sorted(VALID_FACTOR_TYPES)}，got {self.factor_type!r}")
        if self.expected_sign not in VALID_SIGNS:
            raise ValueError(f"expected_sign 必須為 {sorted(VALID_SIGNS)}，got {self.expected_sign!r}")

    def resolve(self) -> object | None:
        """依 source_module + source_function 取得實際函式物件（introspection check）。

        Returns None if source_function is None or import fails. 不擲例外（audit 友善）。
        """
        try:
            mod = importlib.import_module(self.source_module)
        except ImportError:
            return None
        if self.source_function is None:
            return mod
        return getattr(mod, self.source_function, None)


# ============================================================================
# FACTOR_REGISTRY — 全系統因子 SSOT
# ============================================================================
#
# 註冊原則：
#   - 4 個 dimension（technical_score / chip_score / fundamental_score / news_score）
#     必須註冊
#   - 主要 sub_factor（whale / sbl / broker / VCP / chip_macd / quality / revenue_accel
#     / peer_ranking / news_catalyst 等）建議註冊
#   - screener/factors.py 的 predicate 全部註冊（watchlist filter 用）
#   - features/indicators.py 的 5 個 indicator 全部註冊（EAV 持久化用）

FACTOR_REGISTRY: dict[str, FactorSpec] = {
    # ── 4 維 composite dimension（discover composite_score 主成分）──────────
    "technical_score": FactorSpec(
        name="technical_score",
        category="technical",
        factor_type="dimension",
        description="技術面複合分數（momentum / breakout / vcp 3 cluster 等權）",
        source_module="src.discovery.scanner._base",
        source_function=None,
        expected_sign="+",
        used_in_modes=("swing", "value", "dividend", "growth"),
        ic_notes=(
            "2026-05-09 audit: technical IC=-0.133 反主導排名；momentum 模式已歸零"
            "（REGIME_WEIGHTS['momentum']['*']['technical']=0）"
        ),
    ),
    "chip_score": FactorSpec(
        name="chip_score",
        category="chip",
        factor_type="dimension",
        description="籌碼面複合分數（whale + sbl + broker + smart_broker + chip_macd）",
        source_module="src.discovery.scanner._base",
        source_function=None,
        expected_sign="+",
        used_in_modes=("momentum", "swing", "value", "dividend", "growth"),
        holding_days_target=5,
        ic_notes="momentum 模式的 KEY_FACTOR（chip 0.55 主導 bull regime composite）",
    ),
    "fundamental_score": FactorSpec(
        name="fundamental_score",
        category="fundamental",
        factor_type="dimension",
        description="基本面複合分數（peer ranking + quality + revenue acceleration）",
        source_module="src.discovery.scanner._base",
        source_function=None,
        expected_sign="+",
        used_in_modes=("swing", "value", "dividend", "growth"),
        holding_days_target=20,
        ic_notes="swing/value/dividend/growth 模式 KEY_FACTOR；中長期兌現週期 (20+ 天)",
    ),
    "news_score": FactorSpec(
        name="news_score",
        category="news",
        factor_type="dimension",
        description="消息面複合分數（catalyst × 0.7 + (1-risk) × 0.3，含 decay weight）",
        source_module="src.discovery.scanner._base",
        source_function=None,
        expected_sign="+",
        used_in_modes=("momentum", "swing", "value", "dividend", "growth"),
        ic_notes="bull regime IC 結構性為負（momentum 已歸零）；crisis regime 權重最高 (0.35)",
    ),
    # ── 技術面 sub_factor ────────────────────────────────────────────────
    "vcp_score": FactorSpec(
        name="vcp_score",
        category="technical",
        factor_type="sub_factor",
        description="Volatility Contraction Pattern — Stan Weinstein 第二階段突破型態",
        source_module="src.discovery.scanner._functions",
        source_function="compute_vcp_score",
        expected_sign="+",
        used_in_modes=("swing", "growth"),
        holding_days_target=20,
    ),
    "momentum_decay": FactorSpec(
        name="momentum_decay",
        category="technical",
        factor_type="sub_factor",
        description="動能衰減偵測（短期動能 vs 中期動能比值，低值=動能轉弱）",
        source_module="src.discovery.scanner._functions",
        source_function="compute_momentum_decay",
        expected_sign="+",
        used_in_modes=("momentum", "swing"),
    ),
    "volume_price_divergence": FactorSpec(
        name="volume_price_divergence",
        category="technical",
        factor_type="sub_factor",
        description="量價背離（價漲量縮 = 多頭弱；價跌量增 = 賣壓重）",
        source_module="src.discovery.scanner._functions",
        source_function="compute_volume_price_divergence",
        expected_sign="-",
        used_in_modes=("momentum", "swing"),
    ),
    # ── 籌碼面 sub_factor ────────────────────────────────────────────────
    "whale_score": FactorSpec(
        name="whale_score",
        category="chip",
        factor_type="sub_factor",
        description="大戶持股集中度（HoldingDistribution 8F 變化）",
        source_module="src.discovery.scanner._functions",
        source_function="compute_whale_score",
        expected_sign="+",
        used_in_modes=("momentum", "swing"),
    ),
    "sbl_score": FactorSpec(
        name="sbl_score",
        category="chip",
        factor_type="sub_factor",
        description="借券賣出壓力（SBL σ 倍數，高 = 空方堆積）",
        source_module="src.discovery.scanner._functions",
        source_function="compute_sbl_score",
        expected_sign="-",
        used_in_modes=("momentum", "swing"),
    ),
    "broker_score": FactorSpec(
        name="broker_score",
        category="chip",
        factor_type="sub_factor",
        description="分點主力買賣超（BrokerTrade 集中度）",
        source_module="src.discovery.scanner._functions",
        source_function="compute_broker_score",
        expected_sign="+",
        used_in_modes=("momentum", "swing"),
    ),
    "smart_broker_score": FactorSpec(
        name="smart_broker_score",
        category="chip",
        factor_type="sub_factor",
        description="Smart Broker 8F（識別具歷史 alpha 的主力分點）",
        source_module="src.discovery.scanner._functions",
        source_function="compute_smart_broker_score",
        expected_sign="+",
        used_in_modes=("momentum", "swing"),
    ),
    "chip_macd": FactorSpec(
        name="chip_macd",
        category="chip",
        factor_type="sub_factor",
        description="籌碼動能 MACD（外資累買 / 自營商累買 cross）",
        source_module="src.discovery.scanner._functions",
        source_function="compute_chip_macd",
        expected_sign="+",
        used_in_modes=("momentum", "swing"),
    ),
    "key_player_cost": FactorSpec(
        name="key_player_cost",
        category="chip",
        factor_type="sub_factor",
        description="關鍵分點成本距離（離分點均價越近 = 主力套牢未鬆動）",
        source_module="src.discovery.scanner._functions",
        source_function="score_key_player_cost",
        expected_sign="+",
        used_in_modes=("momentum",),
    ),
    # ── 基本面 sub_factor ─────────────────────────────────────────────────
    "quality_score": FactorSpec(
        name="quality_score",
        category="fundamental",
        factor_type="sub_factor",
        description="財務品質（ROE / 毛利率 / 負債比 / 現金流穩定度）",
        source_module="src.discovery.scanner._functions",
        source_function="compute_quality_score",
        expected_sign="+",
        used_in_modes=("swing", "value", "dividend", "growth"),
        holding_days_target=20,
    ),
    "revenue_acceleration_score": FactorSpec(
        name="revenue_acceleration_score",
        category="fundamental",
        factor_type="sub_factor",
        description="營收加速度（YoY 月成長率的趨勢）",
        source_module="src.discovery.scanner._functions",
        source_function="compute_revenue_acceleration_score",
        expected_sign="+",
        used_in_modes=("growth", "swing"),
        holding_days_target=30,
    ),
    "peer_fundamental_ranking": FactorSpec(
        name="peer_fundamental_ranking",
        category="fundamental",
        factor_type="sub_factor",
        description="同業基本面 percentile 排名（產業內相對強度）",
        source_module="src.discovery.scanner._functions",
        source_function="compute_peer_fundamental_ranking",
        expected_sign="+",
        used_in_modes=("swing", "value", "dividend", "growth"),
    ),
    # ── 消息面 sub_factor ─────────────────────────────────────────────────
    "news_decay_weight": FactorSpec(
        name="news_decay_weight",
        category="news",
        factor_type="sub_factor",
        description="公告時間衰減權重（structural/transient/default 三種半衰期）",
        source_module="src.discovery.scanner._functions",
        source_function="compute_news_decay_weight",
        expected_sign="+",
        used_in_modes=("momentum", "swing", "value", "dividend", "growth"),
    ),
    # ── 技術指標 EAV（features/indicators.py，持久化於 TechnicalIndicator 表）──
    "sma_indicator": FactorSpec(
        name="sma_indicator",
        category="technical",
        factor_type="indicator",
        description="簡單移動平均（含 SMA_5/10/20/60/120 多週期）",
        source_module="src.features.indicators",
        source_function="compute_indicators",
        expected_sign="either",
        used_in_modes=("momentum", "swing", "value", "dividend", "growth"),
    ),
    "rsi_indicator": FactorSpec(
        name="rsi_indicator",
        category="technical",
        factor_type="indicator",
        description="相對強弱指標 RSI14（>70 超買 / <30 超賣）",
        source_module="src.features.indicators",
        source_function="compute_indicators",
        expected_sign="either",
        used_in_modes=("momentum", "swing"),
    ),
    "macd_indicator": FactorSpec(
        name="macd_indicator",
        category="technical",
        factor_type="indicator",
        description="MACD 動能交叉指標（DIF/DEM/Histogram）",
        source_module="src.features.indicators",
        source_function="compute_indicators",
        expected_sign="either",
        used_in_modes=("momentum", "swing"),
    ),
    "bb_indicator": FactorSpec(
        name="bb_indicator",
        category="technical",
        factor_type="indicator",
        description="布林通道（20 日 SMA ± 2σ）",
        source_module="src.features.indicators",
        source_function="compute_indicators",
        expected_sign="either",
        used_in_modes=("swing",),
    ),
    "adx_indicator": FactorSpec(
        name="adx_indicator",
        category="technical",
        factor_type="indicator",
        description="平均趨向指標 ADX14（>25 趨勢成立）",
        source_module="src.features.indicators",
        source_function="compute_indicators",
        expected_sign="+",
        used_in_modes=("swing", "growth"),
    ),
    # ── screener/factors.py predicate（watchlist filter，回傳 pd.Series[bool]）──
    "rsi_oversold": FactorSpec(
        name="rsi_oversold",
        category="technical",
        factor_type="predicate",
        description="RSI 超賣訊號（< threshold，預設 30）",
        source_module="src.screener.factors",
        source_function="rsi_oversold",
        expected_sign="+",
    ),
    "macd_golden_cross": FactorSpec(
        name="macd_golden_cross",
        category="technical",
        factor_type="predicate",
        description="MACD DIF 上穿 DEM（買進訊號）",
        source_module="src.screener.factors",
        source_function="macd_golden_cross",
        expected_sign="+",
    ),
    "price_above_sma": FactorSpec(
        name="price_above_sma",
        category="technical",
        factor_type="predicate",
        description="收盤價站上指定週期 SMA",
        source_module="src.screener.factors",
        source_function="price_above_sma",
        expected_sign="+",
    ),
    "foreign_net_buy": FactorSpec(
        name="foreign_net_buy",
        category="chip",
        factor_type="predicate",
        description="外資淨買超（> threshold，預設 0）",
        source_module="src.screener.factors",
        source_function="foreign_net_buy",
        expected_sign="+",
    ),
    "institutional_consecutive_buy": FactorSpec(
        name="institutional_consecutive_buy",
        category="chip",
        factor_type="predicate",
        description="三大法人連續 N 日買超",
        source_module="src.screener.factors",
        source_function="institutional_consecutive_buy",
        expected_sign="+",
    ),
    "short_squeeze_ratio": FactorSpec(
        name="short_squeeze_ratio",
        category="chip",
        factor_type="predicate",
        description="軋空比（融券餘額 / 融券限額 > threshold）",
        source_module="src.screener.factors",
        source_function="short_squeeze_ratio",
        expected_sign="+",
    ),
    "revenue_yoy_growth": FactorSpec(
        name="revenue_yoy_growth",
        category="fundamental",
        factor_type="predicate",
        description="月營收 YoY 高成長（> threshold%，預設 20%）",
        source_module="src.screener.factors",
        source_function="revenue_yoy_growth",
        expected_sign="+",
    ),
    "revenue_consecutive_growth": FactorSpec(
        name="revenue_consecutive_growth",
        category="fundamental",
        factor_type="predicate",
        description="月營收連續 N 月 YoY 正成長（預設 3 月）",
        source_module="src.screener.factors",
        source_function="revenue_consecutive_growth",
        expected_sign="+",
    ),
}


# ============================================================================
# 查詢輔助
# ============================================================================


def get_factor(name: str) -> FactorSpec | None:
    """依 canonical 名稱取得 FactorSpec；找不到回 None。"""
    return FACTOR_REGISTRY.get(name)


def list_factors(
    *,
    category: str | None = None,
    factor_type: str | None = None,
    used_in_mode: str | None = None,
) -> list[FactorSpec]:
    """依 filter 列出符合的 FactorSpec（依 name 排序）。

    Parameters
    ----------
    category : 限定 dimension（technical/chip/fundamental/news/...）
    factor_type : 限定類型（dimension/sub_factor/predicate/indicator）
    used_in_mode : 限定 discover 模式（momentum/swing/...）
    """
    out: list[FactorSpec] = list(FACTOR_REGISTRY.values())
    if category is not None:
        out = [f for f in out if f.category == category]
    if factor_type is not None:
        out = [f for f in out if f.factor_type == factor_type]
    if used_in_mode is not None:
        out = [f for f in out if used_in_mode in f.used_in_modes]
    return sorted(out, key=lambda f: (f.category, f.factor_type, f.name))
