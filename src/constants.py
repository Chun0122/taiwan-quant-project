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

# Scanner Stage 0.5「估值覆蓋是否足夠」的判定窗口與門檻（2026-08-05）。
# **必須看近期窗口而非全表**——原本數全表相異 stock_id，一旦歷史累積 ≥500 檔就
# 永遠不再觸發全市場同步，而 live 每日只有候選池補抓（實測 43~150 檔）。
# 後果：value/dividend 的 `_coarse_filter` 以 `groupby.last()` 取最新一筆估值，
# 拿到的是數月前的舊 PE。窗口取 7 日以容忍假日與 TWSE 收盤後才發布的落差。
VALUATION_FRESH_WINDOW_DAYS: int = 7
VALUATION_MIN_FRESH_STOCKS: int = 500

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
