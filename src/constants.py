"""集中管理全系統共用常數，避免 magic number 散落各模組。"""

from __future__ import annotations

# ── 交易成本 ─────────────────────────────────────────────────────────
COMMISSION_RATE = 0.001425  # 手續費 0.1425%
TAX_RATE = 0.003  # 交易稅 0.3%（賣出時）
SLIPPAGE_RATE = 0.0005  # 滑價 0.05%

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
