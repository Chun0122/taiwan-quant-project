"""集中管理全系統共用常數，避免 magic number 散落各模組。"""

from __future__ import annotations

# ── 交易成本 ─────────────────────────────────────────────────────────
COMMISSION_RATE = 0.001425  # 手續費 0.1425%
TAX_RATE = 0.003  # 交易稅 0.3%（賣出時）
SLIPPAGE_RATE = 0.0005  # 滑價 0.05%

# ── A4 交易現實化：混合單成本模型 ─────────────────────────────────────
# 每筆委託拆成「整張單（一般市場）+ 盤中零股單」兩筆，各自計最低手續費；
# 股數計算不整張化（sizing 不變），僅成本模型反映兩個市場的現實。
LOT_SIZE = 1000  # 台股一張 = 1000 股
MIN_COMMISSION_LOT = 20.0  # 整股單最低手續費（元/筆）
MIN_COMMISSION_ODD = 1.0  # 盤中零股單最低手續費（元/筆，券商電子下單實況）
ODD_LOT_SLIPPAGE_PREMIUM = 0.001  # 零股部分額外滑價 0.1%（盤中零股 spread 較寬、集合競價撮合）
PARTICIPATION_IMPACT_COEFF = 0.01  # 下單量衝擊係數 c：impact = c×√(order_shares/volume)；5% 參與率時 ≈ +0.22%
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

# ── News Catalyst / Risk 事件分類（Phase E）───────────────────────────
# 正向催化：forward-looking 正面訊號（法說、投資人日、買回、正面營收）
NEWS_CATALYST_TYPES: frozenset[str] = frozenset({"earnings_call", "investor_day", "buyback", "revenue"})
# 風險事件：容易與負報酬相關（董監改選/市場派、負面 filing）
NEWS_RISK_TYPES: frozenset[str] = frozenset({"governance_change", "filing"})
# News 合成權重：news_score = CATALYST × catalyst + RISK × (1 - risk)
NEWS_CATALYST_WEIGHT: float = 0.7
NEWS_RISK_WEIGHT: float = 0.3

# ── Regime 預設值 ───────────────────────────────────────────────────────
REGIME_FALLBACK_DEFAULT: str = "sideways"  # Regime 偵測失敗時的安全預設值

# ── 回測 Regime 自適應部位乘數 ─────────────────────────────────────────
# ── 漲跌停模擬 ─────────────────────────────────────────────────────────
LIMIT_PRICE_PCT: float = 0.10  # 台股漲跌停幅度 10%
LIMIT_DETECT_THRESHOLD: float = 0.095  # 偵測門檻（略低於 10% 以涵蓋四捨五入）

# ── 資料品質：日 K close-to-close 跳動哨兵 ──────────────────────────────
# 略高於 ±10% 漲跌停；超過此幅度的單日 close 變動視為「物理上不可能」的可疑跳點，
# 僅 WARN 記錄不過濾（合法成因：除權息跳價 / IPO 首日 / 復牌大漲），供 audit 追查。
PRICE_JUMP_WARN_THRESHOLD: float = 0.11

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

# ── Regime × Mode 封鎖矩陣 ───────────────────────────────────────────
# 在指定 regime 下暫停特定模式的 discover 掃描（歷史績效驗證不佳）。
# 實證依據（2026-02-27 ~ 04-19 T+1 entry 回測）：
#   momentum × sideways：5/10/20 日全線負報酬，勝率 15~30%
#   growth   × sideways：高 beta 特性，歷史先驗相同風險
#   growth   × crisis：高 beta 在危機期放大虧損（保留 momentum，實測 10日 crisis +8.35%）
# 「all」模式跨多模式平均，不封鎖（保留 value/dividend 之類防禦性配置）
REGIME_MODE_BLOCK: dict[str, frozenset[str]] = {
    "sideways": frozenset({"momentum", "growth"}),
    "crisis": frozenset({"growth"}),
}

# ── 各模式的關鍵因子（IC 衰退監控目標 + 警示文案綁定） ─────────────────
# 選擇權重最高的主導維度；若 IC 連續衰退視為該模式信號失效。
# 使用方：
#   1. BaseScanner._KEY_FACTOR_MAP（IC-Decay 動態門檻提升）
#   2. factor-diagnostics 警示文案「{mode} 模式高度依賴 {factor}」
#   3. cli/morning_cmd._step_8c_ic_precheck（反向模式自動停用 discover）
# momentum v5（2026-05-09 audit）：technical 權重歸零後，chip 0.55 為最高維度
#   （參見 src/regime/detector.py:REGIME_WEIGHTS["momentum"]["bull"]）。
#   technical_score IC=-0.13 持續為負，已從評分維度移除，不再作為關鍵因子。
# swing 修正：bull 權重 fundamental 0.40 為最高維度，先前誤設為 chip_score
#   (0.20，第三大)，造成 swing 在 Step 8c 被 chip IC 反向誤殺停用。
DISCOVERY_KEY_FACTOR_MAP: dict[str, str] = {
    "momentum": "chip_score",
    "swing": "fundamental_score",
    "value": "fundamental_score",
    "dividend": "fundamental_score",
    "growth": "fundamental_score",
}

# ── 各模式 IC 預檢的合適 holding_days（與因子本質週期對齊） ────────────
# 2026-05-09 audit 發現：morning-routine Step 8c 一律用 holding=5 計算 rolling IC，
# 對 fundamental_score 主導的模式（value/dividend/growth）天然不適用，
# 造成全部誤判為 inverse、被自動停用。
# 設計原則：holding_days 應接近該模式 KEY_FACTOR 的兌現週期。
#   - chip_score 短期敏感（資金流訊號 1~10 天兌現）→ 5 天
#   - fundamental_score 中長期（YoY 營收/獲利週期 30+ 天）→ 20 天
#   - swing 介於兩者，取中間值 10 天
# 使用方：
#   1. morning_cmd._compute_factor_ic_status（Step 8c 預檢）
#   2. BaseScanner._compute_ic_decay_adjustment / _log_factor_effectiveness /
#      _apply_ic_weight_adjustment（fallback 路徑：未走 morning-routine 直跑 discover 時）
DISCOVERY_IC_HOLDING_DAYS_MAP: dict[str, int] = {
    "momentum": 5,
    "swing": 10,
    "value": 20,
    "dividend": 20,
    "growth": 20,
}

# ── IC 執法門檻（P0 #16 / B11 最小切口，2026-08-01）──────────────────────────
# 背景（logs/audit_discover_20260731/REPORT.md §2）：M2 自動停用曾以
# window_end=2026-06-27、n=40、距今 34 天的**凍結 IC** 每日重複執法，而 momentum
# 停掃後該窗口永遠不再更新 → 模式無法自證恢復（自鎖）。同時 §3 原則 6 要求
# 「樣本 <100 或未跨 3 個獨立掃描週時只能告警不能行動」，但程式無一處執行。
# 以下三個門檻由 src/discovery/ic_governance.py 統一執行（M2 停用 +
# scanner IC 衰退門檻加成共用），任一不滿足即「可觀測但不可執法」。
#
# 窗口時效預算 = holding_days + BUFFER。為何與 holding 掛鉤：rolling 窗口需要
# 記錄日之後 holding_days 的 forward 報酬才可評估，故最新「有資料」的窗口本來
# 就落後約 holding_days；再加 step_days(7) + 假日寬限(7) = BUFFER 14。
# 實測（2026-08-01）：momentum 落後 32 天 > 5+14=19 → 正確判定過期；
# value/growth 落後 18 天 < 20+14=34 → 正常；swing 落後 11 天 < 10+14=24 → 正常。
DISCOVERY_IC_WINDOW_STALE_BUFFER_DAYS: int = 14

# 決策所依據的窗口數；rolling 窗口 step_days=7，故 3 窗 ≈ §3 原則 6 的「跨 3 個掃描週」。
# 決策 IC 取最近 N 窗的平均，而非單一最新窗——實測 swing 最新窗 IC=-0.0933(n=140)
# 但前三窗為 +0.005，3 窗平均 -0.0479 未達 -0.05，單窗判定屬小樣本漂移。
DISCOVERY_IC_MIN_WINDOWS: int = 3

# 決策窗口的最低可評估樣本數（取最近 N 窗的**最小值**，最保守）。
# 對應 §3 原則 6 的「樣本 <100 只能告警」。實測 value/growth 最新窗僅 60 → 不可執法。
DISCOVERY_IC_MIN_SAMPLES: int = 100

# ── 過熱反轉懲罰閘門（2026-05-15 audit：5/7-5/8 三連停損根因修復） ────────
# 證據：6224 (+30%/5d, +45%/10d)、6108 (+21%/5d, +33%/10d)、5864 (+25%/5d, +36%/10d)
#       均為進場後 1-2 日跳空崩跌的追高型；
#       根因為 swing/momentum scanner technical_score 完全是「動能延續」邏輯，
#       無過熱反轉懲罰，在 buying climax 時系統性推薦頂部股票。
# 使用方：BaseScanner._apply_overheating_filter（由 _swing/_momentum _apply_risk_filter 呼叫）
# 設計：硬剔除（超過 EXCLUDE 門檻）+ 軟降分（DAMPEN ~ EXCLUDE 區間 composite_score×factor）
DISCOVERY_OVERHEATING_EXCLUDE_RET5D: float = 0.35
DISCOVERY_OVERHEATING_EXCLUDE_RET10D: float = 0.50
DISCOVERY_OVERHEATING_DAMPEN_RET5D: float = 0.25
DISCOVERY_OVERHEATING_DAMPEN_RET10D: float = 0.35
# 軟降分區間 composite_score 折扣係數
# 推導：swing technical 權重 ≈ 0.30，technical 折半（×0.5）即 composite×(1 - 0.15) = ×0.85
DISCOVERY_OVERHEATING_DAMPEN_FACTOR: float = 0.85

# ── Rotation 'all' 模式 mode 配額（2026-05-15 audit：避免單 mode 集中爆雷） ─
# 證據：all10_5d 5/7-5/8 同時從 swing 模式進 4 檔（6224/6108/5864/3094），
#       整 mode 同時爆雷無分散。改用「每個 primary_mode 最多 N 檔」即可避免。
# primary_mode 定義：該股票在各模式 discovery_record 中 composite_score 最高的 mode
# 使用方：portfolio.manager._resolve_all_mode_rankings
# 2026-06-19 收緊 3→2：all10_5d 連兩次審計（5/29 −4.04pp / 6/15 −5.07pp，後者 N=29
#   樣本紮實）皆 underperform 0050，觸發 5/29 報告 §7.1 既定收緊規則。
ROTATION_ALL_MODE_PER_MODE_MAX: int = 2

# ── Composite（多模式合成）輪動模式 ─────────────────────────────────────
# 「合成模式」= 跨多個 scanner 模式以 avg composite_score 排序、並用 per_mode_max
# 配額避免單一 mode 集中爆雷。'all'（五模式綜合）與 'mom_growth'（動量+成長雙引擎）
# 共用 portfolio.rankings._resolve_composite_rankings。
# - 'all'：保留既有行為（五模式，per_mode_max=ROTATION_ALL_MODE_PER_MODE_MAX）。
# - 'mom_growth'：2026-06-20 alpha 裁決後新增，取代結構性失敗的 'all'。momentum 與
#   growth cross-mode 相關 −0.469（互補對沖），且為訊號層僅有的兩個贏家。per_mode_max=3
#   讓兩模式各最多 3 檔（最多 6 候選）足以填滿 5 部位且不單一集中。
# members 僅限 scanner 會產生 DiscoveryRecord 的單一模式（不可含其他 composite）。
COMPOSITE_MODES: dict[str, dict] = {
    "all": {
        "members": ("momentum", "swing", "value", "dividend", "growth"),
        "per_mode_max": ROTATION_ALL_MODE_PER_MODE_MAX,
    },
    "mom_growth": {
        "members": ("momentum", "growth"),
        "per_mode_max": 3,
    },
}


def is_composite_mode(mode: str | None) -> bool:
    """是否為合成（多模式）輪動模式（'all' / 'mom_growth'）。"""
    return mode in COMPOSITE_MODES


# ── Live T+1 Pending-Order（A2，MASTER_PLAN §6.2 #9）───────────────────────
# 設計：docs/design/live_t1_pending_order.md。D 日收盤後決策寫 pending order，
# D+1 開盤後以 open[D+1] 成交——與 rotation backtest 的 pending_exec 對齊，
# 消除 live「夜間決策、當日收盤成交」的 look-ahead。
# 買單 TTL：決策已過期（rankings 是 D 日的），跨 N 個交易日未成交（停牌等）即
# cancel；風控賣單（stop_loss/crisis 等）不設 TTL，必須出場，順延至有報價為止。
PENDING_BUY_TTL_TRADING_DAYS: int = 2

# ── Rankings stale fallback 時效上限（P0 #13，2026-07-19 事故重審）───────────
# decide 當日無 discover 排名時允許 fallback 至最近掃描日，但逾此交易日數即
# 視為訊號斷糧：回空排名 → 組合凍結新買入、僅走風控賣出/到期路徑。
# 背景：momentum 被 M2 IC 反向停用後，無上限 fallback 使 mom5_10d/mg5_20d
# 以月餘舊排名（6/16）持續交易——scanner 層安全機制被 rotation 層架空。
# 3 = 容忍長週末 + 單日掃描失敗；模式停用超過 3 個交易日即凍結新買入。
RANKINGS_FALLBACK_MAX_TRADING_DAYS: int = 3

# ── 臨時休市哨兵（P0 #14，2026-07-19 事故重審）─────────────────────────────
# 行事曆標為交易日但同步後全市場當日 daily_price 筆數低於此門檻 → 判定疑似
# 臨時休市（颱風假等），凍結 discover 與 rotation update（T+1 佇列自然順延）。
# 正常交易日全市場 ~6,000+ 筆；2026-07-10 颱風假僅 1 筆（US_VIX）。
PHANTOM_TRADING_DAY_MIN_ROWS: int = 100

# ── 歷史回補的「全市場覆蓋」判定門檻（B1①，2026-08-01）────────────────────
# backfill 以此判斷某日是否已回補過。**判定用「普通股（4 碼）檔數」而非總筆數**，
# 這個選擇是被兩次實測逼出來的：
#   1. 只看「該日有無資料」→ 2020~2024 每日皆有 6 檔（watchlist + TAIEX），
#      整整 5 年會被靜默跳過、回補什麼都不做。
#   2. 改看總筆數（門檻 3000）→ 仍有兩種誤判：
#      • 偽陽性：2025-04-07 關稅崩盤日 TAIEX −9.7%、80% 普通股無量跌停，
#        權證當日幾乎無報價 → 總筆數僅 2,922，但普通股 1,894 檔其實完整。
#        以總筆數判定會讓崩盤日被永遠重抓。
#      • 偽陰性：2026-03-03 總筆數 5,795（權證多）但普通股僅 879 檔＝半套，
#        總筆數門檻放它過關。
# 普通股檔數不受權證有無影響：實測完整日恆為 1,799~1,971 檔、假日 0 檔、
# 半套日 879 檔——1500 可乾淨分離。
BACKFILL_MIN_COMMON_STOCKS: int = 1500

# DailyFeature 回補的續跑判定：某日特徵列數 ≥ 當日 DailyPrice 列數 × 此比例才算已補。
# **不可改回「該日有無 DailyFeature 列」**——那是上面 BACKFILL_MIN_COMMON_STOCKS
# 註解所述同一個坑的第二現場，2026-08-09 實測踩到：
#   TPEX 當日同步逾時 → 該日只有上市價量 → 特徵以上市資料算完並寫入 →
#   日期被永久標記為已補 → 事後補齊上櫃價量後，特徵**永遠不會重算**。
# 實測 11 天中招（2022-02-17 起 8 天為歷史回補、2026-05-27/06-22/06-24 為 live），
# 這些日子 daily_price 有 4,400~7,300 列但 daily_feature 只有 1,147~1,362 列。
# 後果不只是重放：缺特徵列的股票會被 `_stage2_liquidity_filter` **整批排除於 universe
# 之外**（`avg5_map` 只由既有列建立，回傳的 universe 又只取自 `avg5_map.index`）。
# ⚠ 初判寫的是「踩破 `_FEATURE_COVERAGE_MIN`(0.3) → 退回 DailyPrice fallback」，**有誤**：
# 那 0.185 是含權證的全表比例，而該門檻的分母是 `stage1_ids`（普通股），實際為
# 973/1750 ≈ 0.556 **高於** 0.3 → 不會 fallback，而是走 DailyFeature 路徑並把
# 沒有特徵列的四成股票靜默丟掉。方向比誤判時所想的更嚴重。
#
# 門檻取 0.95：實測比例分布完全雙峰——正常日 1,591 天全在 0.9989~1.0（僅 volume=0
# 的列不算特徵），中招日 11 天全在 0.18~0.26，中間**沒有任何一天**。
# 誤判方向也是安全的：判成未補只是多花 CPU 重算（upsert 冪等），判成已補才會留下靜默缺口。
FEATURE_BACKFILL_MIN_COVERAGE_RATIO: float = 0.95

# DailyFeature 回補的**欄位暖身**門檻（§6.6 #28，2026-09-05）。
#
# 上面那個比例只管「列數」，而列數滿額**不代表欄位可用**——這是 §6.5 #21d 的教訓
# 在回補層的重演：#21d 只修了重放層的覆蓋度判定（新增 `feature_warm_ratio`），
# 回補層的續跑判定沒跟著改，於是資料本身的洞一直在。
#
# 實測 2024-01-02 ~ 2024-02-20 共 **29 個交易日**的 `ma60` 非空率僅 **0.0022**，
# 列數卻是滿的（1,799~1,822 列）：B1① 第一批回補範圍是 2024-01 起，`daily_feature`
# 也從該日起算故頭幾十天暖身不足；2026-08-08 補完 2020–2023 價量後，續跑判定看
# 列數就判為已補、**永不重算**。
#
# 取 0.5 與 `REPLAY_MIN_FEATURE_WARM_RATIO` 同值（同一個現象的兩層判定不該有兩套
# 數字），實測分布完全雙峰：正常日 0.94~1.0、中招日 0.0022，中間無任何一天。
FEATURE_BACKFILL_MIN_WARM_RATIO: float = 0.5

# 全表最早的 60 個交易日 `ma60` 天生填不滿（沒有更早的資料可暖身），
# 對這些日子套用上面的門檻會讓它們**每次執行都重算**——與本項要修的
# 「永遠重抓」是同一種病，方向相反。故暖身判定只作用於此邊界之後。
FEATURE_WARMUP_EXEMPT_TRADING_DAYS: int = 60

# 回補單一交易日（三個 dataset 全開）的實測耗時，用於 ETA 估算。
# 實測 2026-08-03：458 個交易日耗時 3h45m ≈ 29.5 秒/日。
# 原本以「3 秒 × dataset 數」估算低估近 3 倍——TWSE/TPEX 雖並行，但各自 3 秒
# 節流、且每個 dataset 都要跑一輪，再加解析與 upsert。
SECONDS_PER_BACKFILL_DAY: float = 29.5

# 估值回補（§6.5 #20）：某日 `stock_valuation` 檔數達此值即視為已補而跳過。
# 門檻遠低於 BACKFILL_MIN_COMMON_STOCKS，因為估值的母體本來就小得多——
# TWSE `BWIBBU_d` 只收錄有本益比可算的上市普通股（實測 2024-01-02 為 997 檔、
# 2025-06-05 為 1,041 檔），ETF/權證/虧損股不在內。
# 800 可乾淨分離「全市場已補（≈1,000）」與「僅候選股補抓（實測 43~150）」。
BACKFILL_MIN_VALUATION_STOCKS: int = 800

# 上櫃估值逐股回補的續跑門檻：某檔估值日數 ≥ 其價量日數 × 此比例即視為已補。
# 不用固定日數——上櫃股上市時間不一，新股本來就只有少數交易日。
# 0.8 而非 1.0：FinMind PER 對停牌/無 EPS 的日子會缺列，要求全等會永遠重抓。
VALUATION_COVERAGE_RATIO: float = 0.8

# API 節流間隔（秒）——與 CLAUDE.md §2 的速率規則一致，供 ETA 估算與回補迴圈共用
TWSE_REQUEST_INTERVAL: float = 3.0
FINMIND_REQUEST_INTERVAL: float = 0.5
MOPS_REQUEST_INTERVAL: float = 3.0  # MOPS 靜態頁（mopsov）比照官方端點節流

# 月營收回補/同步的續跑門檻（§6.6 #23/#24，2026-08-15）：某營收月份**由 MOPS 全市場
# 抓回**的相異股票數達此值，即視為該月已補齊。
#
# ⚠ 三個細節都是踩過坑才定的，改動前先讀：
#   1. **必須只數 `source='mops'` 的列**。舊版數的是該月全部列數 ≥500，而候選池逐股
#      補抓（`sync_revenue_for_stocks`）每天寫入約 150 檔、一個月累積上千列——門檻
#      被那些列灌滿後，全市場 MOPS 同步**此後永不執行**。實測 2026-02 的 MOPS 列
#      只有 1 筆（候選池 1,284 筆）、2026-06 為 498 筆，整個月永久凍結在半套。
#      這與 §6.5 #22 的估值閘門是同一種病（計數看錯軸）。
#   2. **1,400 而非 500**：MOPS 成熟月份實測 1,658（2020-01）~1,731（2022-07）檔，
#      而月初剛開始公布時只有數百檔。門檻若低於實際母體，第一次半套抓取就會把該月
#      標記為完成——這正是舊版的失效方式。
#   3. 全市場母體只會成長（上市櫃家數逐年增加），故此值對 2020 年也安全。
BACKFILL_MIN_REVENUE_STOCKS: int = 1400

# 財報回補（§6.6 #25）——FinMind 逐股三表（損益/資產負債/現金流），每檔 3 個請求。
#
# 母體門檻：區間內有 ≥60 個交易日的 4 碼普通股（實測 2020-2026 為 1,994 檔）。
# 不補「幾乎沒交易過」的股票——它們永遠不會進 universe，卻要吃掉 3 個請求。
FINANCIAL_BACKFILL_MIN_TRADING_DAYS: int = 60
FINMIND_REQUESTS_PER_FINANCIAL_STOCK: int = 3

# 續跑判定：某檔在區間內「**欄位非空**的季數 ≥ 應有季數 × 此比例」即視為已補。
#
# ⚠ 判定看的是**欄位**不是列數（§6.5 #21d 的教訓）：三表任一在抓取當下逾時，
# `fetch_financial_summary` 仍會回傳只有損益表的 DataFrame，寫進去就是
# `roe`/`debt_ratio`/`free_cf` 全 NULL 的列——列數檢查完全看不出來，而
# `compute_peer_fundamental_ranking` 會照樣拿這些 NULL 去做同業排名。
# 故 eps（損益）、equity（資產負債）、operating_cf（現金流）三者分別計數，
# 任一未達標就重抓；`_upsert_financial` 同步改為 on_conflict_do_update，
# 否則重抓回來的完整值會被舊的半套列擋在門外（C2 教訓）。
#
# 0.8 而非 1.0：FinMind 對部分公司的早期季度本來就缺（興櫃轉上市前無合併報表），
# 要求全等會讓那些股票每次執行都重抓。
FINANCIAL_COVERAGE_RATIO: float = 0.8

# FinMind 免費版每小時請求上限。**只作為配額查詢失敗時的 fallback**——
# 正常路徑是開跑前打 `fetch_quota_status()` 拿帳號真實上限，據以推導節流間隔
# （3600/limit ＝ 600 時每請求 6 秒）。刻意選「連續慢跑」而非「爆衝後撞 402」：
# 兩者的每小時吞吐相同，但前者不會把錯誤日誌塞滿 402、也不需要等整點。
FINMIND_FREE_HOURLY_LIMIT: int = 600

# FinMind 逐股月營收的回溯天數（§6.6 #23）：**必須 ≥ 13 個月**。
# `fetch_monthly_revenue` 的 YoY 是 `revenue / revenue.shift(12) - 1`，需要 13 筆
# 才算得出最新一筆的年增率。舊值 180 天只取回 6 筆 → `yoy_growth` 恆為 NULL
# （實測 8,663 筆 FinMind 列中 8,353 筆 NULL），而 growth 粗篩是
# `yoy_growth.notna() & > 10`，那些列在粗篩階段就全數蒸發。
REVENUE_FINMIND_LOOKBACK_DAYS: int = 430

# Scanner Stage 0.5「估值覆蓋是否足夠」的判定窗口與門檻（2026-08-05）。
# **必須看近期窗口而非全表**——原本數全表相異 stock_id，一旦歷史累積 ≥500 檔就
# 永遠不再觸發全市場同步，而 live 每日只有候選池補抓（實測 43~150 檔）。
# 後果：value/dividend 的 `_coarse_filter` 以 `groupby.last()` 取最新一筆估值，
# 拿到的是數月前的舊 PE。窗口取 7 日以容忍假日與 TWSE 收盤後才發布的落差。
VALUATION_FRESH_WINDOW_DAYS: int = 7
VALUATION_MIN_FRESH_STOCKS: int = 500

# PIT 重放的資料覆蓋門檻（§6.5 #21b，2026-08-06）。
# 用途：區分「模式判斷不進場」與「輸入資料根本缺席」——兩者的 n_picks 都是 0，
# 但前者是結論、後者是**結果無效**。2026-08-04 的跨模式重放正是栽在這裡：
# value 因 fail-open 而 30 天全數產出、dividend 因 fail-closed 而 30 天只產出 4 天，
# 兩者的 `n_picks` 都無法揭露真因，必須實查資料表才發現三個模式的結果不可採信。
#
# 門檻刻意沿用各自的 SSOT，不另立一套數字：
#   • 價量＝BACKFILL_MIN_COMMON_STOCKS（全市場覆蓋的既有判定）
#   • 估值＝VALUATION_MIN_FRESH_STOCKS（Scanner Stage 0.5 的同一門檻）
#   • 特徵＝FEATURE_BACKFILL_MIN_COVERAGE_RATIO（與回補續跑判定同一個 0.95）
#
# ⚠ 2026-08-09 更正：原值 0.3，理由寫的是「＝UniverseFilter._FEATURE_COVERAGE_MIN，
# 低於此值 Stage 2 已 fallback」——**那個理由是錯的**，兩件事都錯：
#   1. 缺特徵列的股票不是「門檻放寬」而是**整批被排除**。`_stage2_liquidity_filter`
#      的 `avg5_map` 只由 `df_feature` 既有列建立，`passed_absolute` 又只取自
#      `avg5_map.index` → 沒有特徵列的股票根本進不了 universe。
#   2. 因此 0.3 這個「fallback 邊界」根本不是可採信與否的邊界。實測 11 個缺口日
#      的比例是 0.556（973/1750），遠在 0.3 之上卻少了四成候選池，靜默通過。
# 改用 0.95：價量/特徵列比的實測分布與回補判定是**同一個雙峰**（正常日 0.9989~1.0、
# 缺口日 0.18~0.26），故沿用同一個數字，不再自立一套。
REPLAY_MIN_FEATURE_RATIO: float = 0.95
# 月營收母體：live 實測 1,896 檔。300（≈16%）以下時 growth 的 universe 已非全市場，
# 重放結果代表的是那個子集而非模式本身——實測 2020~2024 每年僅 **5 支**。
REPLAY_MIN_REVENUE_STOCKS: int = 300

# 特徵「暖身」門檻（§6.5 #21d，2026-08-09）：`daily_feature` 的列數足夠**不代表**
# 欄位可用。MA60 需 60 個交易日才填滿，回補範圍的頭幾十天欄位全是 NaN，而列數檢查
# 完全看不出來——與 fail-open 同一類的靜默失效。
#
# 具體後果不只是「分數不準」：`_stage2_liquidity_filter` 對 `turnover_ma20` 為 NaN
# 的個股**跳過該股的 ma20 門檻**，暖身期等於那道過濾整段消失。
# （NaN 與「整批缺列」是兩種不同失效——後者是直接排除，由 REPLAY_MIN_FEATURE_RATIO 把守。）
#
# 門檻取 0.5——實測分布是乾淨的雙峰，不需判斷。**母體限 4 碼普通股**：
#   • 暖身失效：非空率 **0.000**（2020-01-02、01-20、02-10）
#   • 穩態：**0.988 ~ 0.998**（2020-03-10 至 2026-06-01 抽樣）
# ⚠ 若不限 4 碼，穩態會掉到 0.646~0.786——權證/ETN 上市時間短、MA 天生填不滿，
#   把它們算進母體會讓門檻無從設定。這也是本檢查與 `feature_stocks` 一致採
#   `length(stock_id) == 4` 的原因。
# 取 ma60 與 turnover_ma20 的**較小值**（ma60 窗口長，是 binding constraint）。
REPLAY_MIN_FEATURE_WARM_RATIO: float = 0.5

# RotationPendingOrder.status 狀態機：pending → filled | cancelled（無其他轉移）
PENDING_STATUS_PENDING: str = "pending"
PENDING_STATUS_FILLED: str = "filled"
PENDING_STATUS_CANCELLED: str = "cancelled"

# RotationPendingOrder.side
PENDING_SIDE_BUY: str = "buy"
PENDING_SIDE_SELL: str = "sell"

# RotationActionLog.action_type 新值：decide() 寫入的「明日預定操作」
# （dashboard 顯示用；D+1 成交後另寫 open/close）。買賣分兩型供 UI 直接區分。
ACTION_TYPE_PENDING_BUY: str = "pending_buy"
ACTION_TYPE_PENDING_SELL: str = "pending_sell"
# decide 階段寫入的 action_type 集合（decide 冪等刪除範圍；open/close 屬 fill 階段）
DECIDE_STAGE_ACTION_TYPES: tuple[str, ...] = (ACTION_TYPE_PENDING_BUY, ACTION_TYPE_PENDING_SELL, "renew", "hold")

# A3 股利會計：持倉除息入帳的 ActionLog 類型（同時作為同日冪等入帳標記，
# 與現金更新同一 transaction —— decide/fill 的冪等刪除範圍皆不含此型）
ACTION_TYPE_DIVIDEND: str = "dividend"
