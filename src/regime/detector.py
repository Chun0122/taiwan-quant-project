"""市場狀態（Regime）偵測器。

根據加權指數（TAIEX）判斷當前市場狀態：
- bull（多頭）：指數在長短均線之上 + 短期正報酬
- bear（空頭）：指數在長短均線之下 + 短期負報酬
- sideways（盤整）：混合訊號
- crisis（崩盤）：快速崩跌訊號觸發（覆蓋多數決結果）

三訊號多數決（bull/bear/sideways）：
1. TAIEX close vs SMA60
2. TAIEX close vs SMA120
3. 20 日報酬率 > 3% / < -3%

Crisis 快速訊號（≥2 個觸發即覆蓋為 crisis）：
1. 5 日報酬率 < -5%
2. 連續下跌 ≥ 3 天
3. 近 20 日波動率 > 過去 120 日平均波動率 × 1.8
4. 爆量長黑：TAIEX 成交量 > 20 日均量 × 1.5 且當日下跌
5. 台灣 VIX 飆升（VIX > 30 或單日漲幅 > 25%，FinMind TW_VIX）
6. TAIEX 單日急跌 > 2.5%
7. 美國 VIX 飆升（CBOE ^VIX > 30 或單日漲幅 > 25%，yfinance US_VIX）

市場寬度降級（Breadth Downgrade）：
- 跌破 MA20 股票比例 > 60% → regime 降一級（bull→sideways, sideways→bear）
- 降級在多數決之後、crisis 覆蓋之前執行

非對稱 Hysteresis（狀態轉換遲滯）：
- Bull → Sideways：close < SMA60 × 0.99 → 1 天即降級（快速防護）
- Sideways → Bull：close > SMA60 × 1.01 連續 3 天 → 過濾死貓反彈
- Sideways → Bear：raw_regime == bear 連續 2 天確認
- Bear → Sideways：close > SMA60 連續 3 天確認
- Any → Crisis：立即覆蓋（無遲滯）
- Crisis → Bear：3 日低點遞增 + 波動率回落(< 1.3) 連續 2 天
- Crisis → Sideways：crisis exit + close > SMA60 連續 2 天

狀態持久化：JSON 檔（data/regime_state.json），cold-start 時使用 raw regime。
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from sqlalchemy import func, select

from src.data.database import get_session
from src.data.schema import DailyPrice

logger = logging.getLogger(__name__)

RegimeType = Literal["bull", "bear", "sideways", "crisis"]

# Crisis 快速崩盤偵測門檻
_CRISIS_RETURN_5D: float = -0.05  # 5 日報酬率 < -5%
_CRISIS_CONSEC_DOWN: int = 3  # 連跌 ≥ 3 天
_CRISIS_VOL_RATIO: float = 1.8  # rolling-20d-vol / avg-vol-120d > 1.8
_CRISIS_PANIC_VOL_RATIO: float = 1.5  # TAIEX 成交量 > 20 日均量 × 1.5 且下跌
_CRISIS_VIX_LEVEL: float = 30.0  # 台灣 VIX 絕對值 > 30 觸發
_CRISIS_VIX_DAILY_CHANGE: float = 0.25  # 台灣 VIX 單日漲幅 > 25% 觸發
_CRISIS_SINGLE_DAY_DROP: float = -0.025  # TAIEX 單日跌幅 > 2.5% 觸發
_CRISIS_US_VIX_LEVEL: float = 30.0  # 美國 VIX 絕對值 > 30 觸發
_CRISIS_US_VIX_DAILY_CHANGE: float = 0.25  # 美國 VIX 單日漲幅 > 25% 觸發

# 市場寬度降級門檻
_BREADTH_BELOW_MA20_THRESHOLD: float = 0.60  # >60% 股票跌破 MA20 → regime 降一級

# Regime 對 Discover 各模式的權重調整矩陣
# crisis 設計邏輯：崩盤時技術訊號失真（跳空），降至接近 0；
# 新聞/基本面（品質防禦）提升至最高，只留有真實催化劑的防禦股
REGIME_WEIGHTS: dict[str, dict[str, dict[str, float]]] = {
    "momentum": {
        # v5 修正（2026-05-09 audit）：technical 權重歸零
        # 證據鏈：(1) E2 IC: technical_score = -0.1330（持續為負，rolling -0.05 ~ -0.14）
        #         (2) ablation --with-performance: 移除 technical → 勝率 53.4%→56.6% (+3.2pp)
        #         (3) 維度級消融 ρ=0.3293 主導排名，但方向錯誤
        # 政策：直接刪除 technical 訊號（不再依賴 IC-aware flip 治標）；
        #       權重重新分配到 chip（最高 IC 0.18）/ fundamental（0.15）。
        # v4 註解（保留）：news 於 bull 期歸零（IC 結構性為負，Stage 3.5h 負面閘門承接濾網角色）
        # Bull：chip 主導 + fundamental 輔助
        "bull": {"technical": 0.00, "chip": 0.55, "fundamental": 0.45, "news": 0.00},
        # Sideways：blocked（_blocked_regimes），此權重不會被使用，維護正確性
        "sideways": {"technical": 0.00, "chip": 0.45, "fundamental": 0.40, "news": 0.15},
        # Bear：基本面防禦提升
        "bear": {"technical": 0.00, "chip": 0.42, "fundamental": 0.38, "news": 0.20},
        # Crisis：消息面催化（防禦股事件驅動）+ 籌碼/基本面平衡
        "crisis": {"technical": 0.00, "chip": 0.35, "fundamental": 0.30, "news": 0.35},
    },
    "swing": {
        "bull": {"technical": 0.30, "chip": 0.20, "fundamental": 0.40, "news": 0.10},
        "sideways": {"technical": 0.25, "chip": 0.25, "fundamental": 0.35, "news": 0.15},
        "bear": {"technical": 0.15, "chip": 0.25, "fundamental": 0.45, "news": 0.15},
        # Crisis：基本面防禦力最重要，消息面從 0.25 降至 0.20
        "crisis": {"technical": 0.05, "chip": 0.20, "fundamental": 0.55, "news": 0.20},
    },
    "value": {
        "bull": {"fundamental": 0.40, "valuation": 0.35, "chip": 0.15, "news": 0.10},
        "sideways": {"fundamental": 0.45, "valuation": 0.25, "chip": 0.15, "news": 0.15},
        "bear": {"fundamental": 0.50, "valuation": 0.20, "chip": 0.10, "news": 0.20},
        "crisis": {"fundamental": 0.55, "valuation": 0.15, "chip": 0.10, "news": 0.20},
    },
    "dividend": {
        "bull": {"fundamental": 0.35, "dividend": 0.35, "chip": 0.20, "news": 0.10},
        "sideways": {"fundamental": 0.40, "dividend": 0.30, "chip": 0.15, "news": 0.15},
        "bear": {"fundamental": 0.45, "dividend": 0.25, "chip": 0.10, "news": 0.20},
        "crisis": {"fundamental": 0.55, "dividend": 0.15, "chip": 0.10, "news": 0.20},
    },
    "growth": {
        "bull": {"fundamental": 0.45, "technical": 0.30, "chip": 0.15, "news": 0.10},
        "sideways": {"fundamental": 0.40, "technical": 0.25, "chip": 0.20, "news": 0.15},
        "bear": {"fundamental": 0.50, "technical": 0.15, "chip": 0.15, "news": 0.20},
        # Crisis：基本面為核心防禦，消息面從 0.25 降至 0.20
        "crisis": {"fundamental": 0.60, "technical": 0.05, "chip": 0.15, "news": 0.20},
    },
}


# ── Hysteresis 轉換規則 ─────────────────────────────────────────
# (prev_regime, target_regime) → {"confirmation_days": int, ...}
HYSTERESIS_RULES: dict[tuple[str, str], dict] = {
    ("bull", "sideways"): {
        "confirmation_days": 1,
        "label": "bull→sideways 快速降級",
    },
    ("sideways", "bull"): {
        "confirmation_days": 3,
        "label": "sideways→bull 確認（過濾死貓反彈）",
    },
    ("sideways", "bear"): {
        "confirmation_days": 2,
        "label": "sideways→bear 確認",
    },
    ("bear", "sideways"): {
        "confirmation_days": 3,
        "label": "bear→sideways 復甦確認",
    },
    ("crisis", "bear"): {
        "confirmation_days": 2,
        "label": "crisis→bear 退出",
    },
    ("crisis", "sideways"): {
        "confirmation_days": 2,
        "label": "crisis→sideways 退出",
    },
}

# Hysteresis 專用門檻
_HYSTERESIS_SMA60_BUFFER: float = 0.01  # ±1% buffer
_CRISIS_EXIT_VOL_RATIO: float = 1.3  # 波動率回落門檻


def check_transition_condition(
    prev_regime: str,
    target_regime: str,
    closes: pd.Series,
    volumes: pd.Series | None = None,
    sma_short: int = 60,
    sma_long: int = 120,
) -> bool:
    """檢查 Hysteresis 轉換條件是否在最新一天滿足（純函數）。

    Args:
        prev_regime: 前次確認的 regime
        target_regime: 目標 regime
        closes: TAIEX 收盤價序列（由舊至新）
        volumes: TAIEX 成交量序列（可選）
        sma_short: 短期均線天數（預設 60）
        sma_long: 長期均線天數（預設 120）

    Returns:
        bool: 條件是否在最新一天滿足
    """
    if len(closes) < sma_short:
        return False

    current = float(closes.iloc[-1])
    sma60 = float(closes.iloc[-sma_short:].mean())

    key = (prev_regime, target_regime)

    if key == ("bull", "sideways"):
        # 快速降級：close < SMA60 × 0.99（跌破 1%）
        return current < sma60 * (1 - _HYSTERESIS_SMA60_BUFFER)

    if key == ("sideways", "bull"):
        # 確認升級：close > SMA60 × 1.01（站穩 1% 以上）
        return current > sma60 * (1 + _HYSTERESIS_SMA60_BUFFER)

    if key == ("sideways", "bear"):
        # 確認降級：close < SMA120 或 20d return < -3%
        if len(closes) < sma_long:
            return False
        sma120 = float(closes.iloc[-sma_long:].mean())
        below_sma120 = current < sma120
        ret_20d = (current - closes.iloc[-21]) / closes.iloc[-21] if len(closes) >= 22 else 0.0
        return below_sma120 or ret_20d < -0.03

    if key == ("bear", "sideways"):
        # 復甦確認：close > SMA60
        return current > sma60

    if key in (("crisis", "bear"), ("crisis", "sideways")):
        # crisis 退出：3 日低點遞增 + 波動率回落
        if len(closes) < 4:
            return False
        # 條件 1：近 3 日收盤價遞增（close[-2] > close[-3]，close[-1] > close[-2]）
        lows_rising = closes.iloc[-2] > closes.iloc[-3] and closes.iloc[-1] > closes.iloc[-2]
        # 條件 2：近期波動率回落至基準的 1.3 倍以下
        vol_calming = True
        if len(closes) >= 30:
            daily_returns = closes.pct_change().dropna()
            recent_vol = float(daily_returns.iloc[-20:].std()) if len(daily_returns) >= 20 else 0.0
            baseline_start = max(0, len(daily_returns) - 120)
            baseline_end = max(0, len(daily_returns) - 20)
            if baseline_end > baseline_start:
                baseline_vol = float(daily_returns.iloc[baseline_start:baseline_end].std())
                if baseline_vol > 0:
                    vol_calming = (recent_vol / baseline_vol) < _CRISIS_EXIT_VOL_RATIO

        base_met = lows_rising and vol_calming
        if key == ("crisis", "sideways"):
            # 額外要求：close > SMA60
            return base_met and current > sma60
        return base_met

    # 未定義的轉換（例 bull→bear 直接跳）：依賴 raw_regime 判定即可
    return True


def apply_hysteresis(
    raw_regime: str,
    prev_regime: str | None,
    closes: pd.Series,
    volumes: pd.Series | None = None,
    confirmation_count: int = 0,
    sma_short: int = 60,
    sma_long: int = 120,
) -> tuple[str, int, dict]:
    """對 raw regime 套用 Hysteresis 轉換規則（純函數）。

    取 ``detect_from_series()`` 的 raw regime，與前次確認 regime 比對，
    根據轉換矩陣決定是否需要連續 N 天確認才允許轉換。

    規則：
    - prev_regime=None（cold start）→ 接受 raw_regime
    - raw_regime=crisis → 立即覆蓋（無遲滯）
    - raw == prev → 不變，重置計數器
    - 有定義的轉換 → 檢查條件 + 確認天數
    - 未定義的轉換 → 預設 2 天確認

    Args:
        raw_regime: ``detect_from_series()`` 的原始 regime
        prev_regime: 前次確認的 regime（None = cold start）
        closes: TAIEX 收盤價序列
        volumes: TAIEX 成交量序列
        confirmation_count: 目前已連續滿足條件的天數
        sma_short: 短期均線天數
        sma_long: 長期均線天數

    Returns:
        tuple:
        - final_regime (str): Hysteresis 後的 regime
        - new_confirmation_count (int): 更新後的計數器
        - transition_info (dict): 轉換決策資訊
    """
    _info_base = {
        "raw_regime": raw_regime,
        "prev_regime": prev_regime,
    }

    # Cold start：無前次 regime → 直接接受
    if prev_regime is None:
        return raw_regime, 0, {**_info_base, "transition_blocked": False, "reason": "cold_start"}

    # Any → Crisis：立即覆蓋，無需確認
    if raw_regime == "crisis":
        return "crisis", 0, {**_info_base, "transition_blocked": False, "reason": "crisis_immediate"}

    # 同 regime：不變，重置計數器
    if raw_regime == prev_regime:
        return prev_regime, 0, {**_info_base, "transition_blocked": False, "reason": "no_change"}

    # 查找轉換規則
    key = (prev_regime, raw_regime)
    if key in HYSTERESIS_RULES:
        rule = HYSTERESIS_RULES[key]
        required = rule["confirmation_days"]
        condition_met = check_transition_condition(prev_regime, raw_regime, closes, volumes, sma_short, sma_long)
    else:
        # 未定義的轉換（例 bull→bear 直跳）：預設 2 天確認
        required = 2
        condition_met = True

    if condition_met:
        new_count = confirmation_count + 1
        if new_count >= required:
            return (
                raw_regime,
                0,
                {
                    **_info_base,
                    "transition_blocked": False,
                    "reason": f"confirmed_after_{required}_days",
                    "pending_transition": None,
                    "confirmation_progress": f"{new_count}/{required}",
                },
            )
        else:
            return (
                prev_regime,
                new_count,
                {
                    **_info_base,
                    "transition_blocked": True,
                    "pending_transition": f"{prev_regime}→{raw_regime}",
                    "confirmation_progress": f"{new_count}/{required}",
                    "reason": "pending_confirmation",
                },
            )
    else:
        # 條件不滿足，重置計數器
        return (
            prev_regime,
            0,
            {
                **_info_base,
                "transition_blocked": True,
                "pending_transition": None,
                "reason": "condition_not_met",
            },
        )


# ── Regime 狀態持久化 ──────────────────────────────────────────

_DEFAULT_STATE_PATH = Path("data/regime_state.json")


@dataclass
class RegimeState:
    """Regime 狀態持久化資料。"""

    regime: str
    regime_since: str  # ISO date
    confirmation_count: int
    pending_transition: str | None
    last_updated: str  # ISO date


class RegimeStateMachine:
    """有狀態的 Regime 偵測器，包裝 detect_from_series() + apply_hysteresis()。

    管理 regime 狀態持久化（JSON 檔案）。Cold start 時（無檔案或損壞）
    使用 detect_from_series() 的 raw regime 作為初始狀態。

    同一天重複呼叫時回傳快取結果（不重複累加 confirmation_count）。

    Args:
        state_path: JSON 狀態檔路徑（預設 data/regime_state.json）
        sma_short: SMA 短期天數
        sma_long: SMA 長期天數
        return_window: 報酬率回溯天數
    """

    def __init__(
        self,
        state_path: str | Path | None = None,
        sma_short: int = 60,
        sma_long: int = 120,
        return_window: int = 20,
    ) -> None:
        self.state_path = Path(state_path) if state_path else _DEFAULT_STATE_PATH
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.return_window = return_window
        self._cached_result: dict | None = None

    def update(
        self,
        closes: pd.Series,
        volumes: pd.Series | None = None,
        breadth_below_ma20_pct: float | None = None,
        vix_series: pd.Series | None = None,
        us_vix_series: pd.Series | None = None,
    ) -> dict:
        """執行 detect_from_series() → apply_hysteresis()，更新並持久化狀態。

        Args:
            closes: TAIEX 收盤價序列
            volumes: TAIEX 成交量序列
            breadth_below_ma20_pct: 跌破 MA20 比例
            vix_series: 台灣 VIX 收盤價序列（None 則跳過 VIX 訊號）
            us_vix_series: 美國 VIX 收盤價序列（None 則跳過 US VIX 訊號）

        Returns:
            dict: 與 detect_from_series() 相同 schema + 額外 hysteresis 欄位
        """
        today_str = datetime.date.today().isoformat()

        # 同日重複呼叫 → 回傳快取
        state = self._load_state()
        if state is not None and state.last_updated == today_str and self._cached_result is not None:
            return self._cached_result

        # 取 raw regime
        raw_result = detect_from_series(
            closes,
            volumes=volumes,
            vix_series=vix_series,
            us_vix_series=us_vix_series,
            sma_short=self.sma_short,
            sma_long=self.sma_long,
            return_window=self.return_window,
            breadth_below_ma20_pct=breadth_below_ma20_pct,
        )

        prev_regime = state.regime if state else None
        prev_count = state.confirmation_count if state else 0
        prev_pending = state.pending_transition if state else None

        # 如果前次有 pending transition 但這次 raw_regime 改變了方向 → 重置計數器
        if prev_pending and state:
            expected_target = prev_pending.split("→")[-1] if "→" in prev_pending else None
            if expected_target and expected_target != raw_result["regime"]:
                prev_count = 0

        final_regime, new_count, transition_info = apply_hysteresis(
            raw_regime=raw_result["regime"],
            prev_regime=prev_regime,
            closes=closes,
            volumes=volumes,
            confirmation_count=prev_count,
            sma_short=self.sma_short,
            sma_long=self.sma_long,
        )

        # 更新狀態
        regime_since = today_str
        if state and state.regime == final_regime:
            regime_since = state.regime_since  # 延續原 since

        new_state = RegimeState(
            regime=final_regime,
            regime_since=regime_since,
            confirmation_count=new_count,
            pending_transition=transition_info.get("pending_transition"),
            last_updated=today_str,
        )
        self._save_state(new_state)

        # 建構回傳 dict（superset of detect_from_series）
        result = {**raw_result}
        result["regime"] = final_regime
        result["hysteresis_applied"] = final_regime != raw_result["regime"]
        result["raw_regime"] = raw_result["regime"]
        result["transition_info"] = transition_info

        self._cached_result = result
        return result

    def _load_state(self) -> RegimeState | None:
        """從 JSON 載入狀態。檔案不存在或損壞時回傳 None（cold start）。"""
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                return RegimeState(
                    regime=data["regime"],
                    regime_since=data["regime_since"],
                    confirmation_count=data.get("confirmation_count", 0),
                    pending_transition=data.get("pending_transition"),
                    last_updated=data.get("last_updated", ""),
                )
        except Exception:
            logger.debug("regime_state.json 讀取失敗，cold start")
        return None

    def _save_state(self, state: RegimeState) -> None:
        """將狀態寫入 JSON 檔。"""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(asdict(state), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("regime_state.json 寫入失敗: %s", exc)

    @property
    def current_regime(self) -> str | None:
        """目前確認的 regime，未初始化時回傳 None。"""
        state = self._load_state()
        return state.regime if state else None


def compute_market_breadth_pct(
    closes: pd.Series,
    ma20s: pd.Series,
) -> float:
    """計算跌破 MA20 的股票比例（純函數）。

    Args:
        closes: 各股最新收盤價，index 為 stock_id
        ma20s: 各股 MA20 值，index 為 stock_id（與 closes 對齊）

    Returns:
        float: 跌破 MA20 的比例（0.0~1.0），空輸入回傳 0.0
    """
    if len(closes) == 0 or len(ma20s) == 0:
        return 0.0

    # 對齊兩個 Series，排除 NaN
    aligned = pd.DataFrame({"close": closes, "ma20": ma20s}).dropna()
    if len(aligned) == 0:
        return 0.0

    below = (aligned["close"] < aligned["ma20"]).sum()
    return float(below / len(aligned))


def detect_crisis_signals(
    closes: pd.Series,
    volumes: pd.Series | None = None,
    vix_series: pd.Series | None = None,
    us_vix_series: pd.Series | None = None,
    return_5d_threshold: float = _CRISIS_RETURN_5D,
    consec_down_days: int = _CRISIS_CONSEC_DOWN,
    vol_ratio_threshold: float = _CRISIS_VOL_RATIO,
    panic_vol_ratio: float = _CRISIS_PANIC_VOL_RATIO,
    vix_level_threshold: float = _CRISIS_VIX_LEVEL,
    vix_change_threshold: float = _CRISIS_VIX_DAILY_CHANGE,
    single_day_drop_threshold: float = _CRISIS_SINGLE_DAY_DROP,
    us_vix_level_threshold: float = _CRISIS_US_VIX_LEVEL,
    us_vix_change_threshold: float = _CRISIS_US_VIX_DAILY_CHANGE,
    vol_window: int = 20,
    vol_baseline: int = 120,
) -> dict:
    """從收盤價序列偵測快速崩盤訊號（純函數，供測試用）。

    七個快速訊號，任意 ≥2 個觸發 → crisis：
    1. fast_return_5d：5 日報酬率 < return_5d_threshold（預設 -5%）
    2. consec_decline：最後 consec_down_days 天連續收跌
    3. vol_spike：rolling-20d-std(daily_returns) / avg-rolling-20d-std(120d) > vol_ratio_threshold
    4. panic_volume：爆量長黑（TAIEX 成交量 > 20 日均量 × panic_vol_ratio 且當日下跌）
    5. vix_spike：台灣 VIX ≥ vix_level_threshold 或單日漲幅 ≥ vix_change_threshold
    6. single_day_drop：TAIEX 單日跌幅 > single_day_drop_threshold
    7. us_vix_spike：美國 VIX (CBOE ^VIX) ≥ us_vix_level_threshold 或單日漲幅 ≥ us_vix_change_threshold

    Args:
        closes: TAIEX 收盤價序列（pd.Series，時序由舊至新）
        volumes: TAIEX 成交量序列（pd.Series，與 closes 等長；None 則跳過 panic_volume 訊號）
        vix_series: 台灣 VIX 收盤價序列（pd.Series；None 則跳過 vix_spike 訊號）
        us_vix_series: 美國 VIX (CBOE ^VIX) 收盤價序列（pd.Series；None 則跳過 us_vix_spike 訊號）
        return_5d_threshold: 5 日跌幅門檻（預設 -0.05 即 -5%）
        consec_down_days: 連跌天數門檻（預設 3）
        vol_ratio_threshold: 波動率倍數門檻（預設 1.8）
        panic_vol_ratio: 爆量門檻（成交量 / 20 日均量，預設 1.5）
        vix_level_threshold: 台灣 VIX 絕對值門檻（預設 30.0）
        vix_change_threshold: 台灣 VIX 單日漲幅門檻（預設 0.25 即 25%）
        single_day_drop_threshold: TAIEX 單日跌幅門檻（預設 -0.025 即 -2.5%）
        us_vix_level_threshold: 美國 VIX 絕對值門檻（預設 30.0）
        us_vix_change_threshold: 美國 VIX 單日漲幅門檻（預設 0.25 即 25%）
        vol_window: 近期波動率計算窗口（預設 20 天）
        vol_baseline: 基準波動率回溯天數（預設 120 天）

    Returns:
        dict: {
            "crisis": bool,
            "signals": {
                "fast_return_5d": bool,
                "consec_decline": bool,
                "vol_spike": bool,
                "panic_volume": bool,
                "vix_spike": bool,
                "single_day_drop": bool,
                "us_vix_spike": bool,
            },
            "fast_return_5d_val": float,
            "vol_ratio_val": float,
            "vix_val": float,
            "us_vix_val": float,
        }
    """
    _safe = {
        "crisis": False,
        "signals": {
            "fast_return_5d": False,
            "consec_decline": False,
            "vol_spike": False,
            "panic_volume": False,
            "vix_spike": False,
            "single_day_drop": False,
            "us_vix_spike": False,
        },
        "fast_return_5d_val": 0.0,
        "vol_ratio_val": 0.0,
        "vix_val": 0.0,
        "us_vix_val": 0.0,
    }

    if len(closes) < max(consec_down_days + 1, 10):
        return _safe

    # Signal 1: 5 日報酬率
    sig_return5d = False
    ret5d_val = 0.0
    if len(closes) >= 6:
        ret5d_val = float((closes.iloc[-1] - closes.iloc[-6]) / closes.iloc[-6])
        sig_return5d = ret5d_val < return_5d_threshold

    # Signal 2: 連續下跌天數
    sig_consec = False
    count = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes.iloc[i] < closes.iloc[i - 1]:
            count += 1
        else:
            break
        if count >= consec_down_days:
            sig_consec = True
            break

    # Signal 3: 波動率飆升（rolling-20d std 比率）
    sig_vol = False
    vol_ratio_val = 0.0
    if len(closes) >= vol_window + 10:
        daily_returns = closes.pct_change().dropna()
        rolling_vol = daily_returns.rolling(vol_window).std()
        rolling_vol = rolling_vol.dropna()
        if len(rolling_vol) >= 2:
            recent_vol = float(rolling_vol.iloc[-1])
            # 基準：排除最近 vol_window 天，取更早的滾動波動率均值
            baseline_vols = (
                rolling_vol.iloc[-(vol_baseline):-vol_window]
                if len(rolling_vol) > vol_window
                else rolling_vol.iloc[:-1]
            )
            if len(baseline_vols) >= 5:
                avg_vol = float(baseline_vols.mean())
                if avg_vol > 0:
                    vol_ratio_val = recent_vol / avg_vol
                    sig_vol = vol_ratio_val > vol_ratio_threshold
            elif recent_vol > 0:
                # fallback：樣本不足時直接比較近期與全期均值
                avg_vol_all = float(rolling_vol.mean())
                if avg_vol_all > 0:
                    vol_ratio_val = recent_vol / avg_vol_all
                    sig_vol = vol_ratio_val > vol_ratio_threshold

    # Signal 4: 爆量長黑（TAIEX 成交量 > 20d 均量 × panic_vol_ratio 且當日下跌）
    sig_panic_vol = False
    if volumes is not None and len(volumes) >= vol_window + 1 and len(closes) >= 2:
        latest_vol = float(volumes.iloc[-1])
        vol_ma20 = float(volumes.iloc[-(vol_window + 1) : -1].mean())
        latest_return = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
        if vol_ma20 > 0 and latest_vol > vol_ma20 * panic_vol_ratio and latest_return < 0:
            sig_panic_vol = True

    # Signal 5: VIX 飆升（絕對水準 > 30 或單日漲幅 > 25%）
    sig_vix = False
    vix_val = 0.0
    if vix_series is not None and len(vix_series) >= 2:
        vix_val = float(vix_series.iloc[-1])
        vix_prev = float(vix_series.iloc[-2])
        vix_change = (vix_val - vix_prev) / vix_prev if vix_prev > 0 else 0.0
        sig_vix = vix_val >= vix_level_threshold or vix_change >= vix_change_threshold

    # Signal 6: TAIEX 單日急跌 > 2.5%
    sig_single_drop = False
    if len(closes) >= 2:
        daily_ret = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
        sig_single_drop = float(daily_ret) < single_day_drop_threshold

    # Signal 7: 美國 VIX (CBOE ^VIX) 飆升（絕對水準 > 30 或單日漲幅 > 25%）
    sig_us_vix = False
    us_vix_val = 0.0
    if us_vix_series is not None and len(us_vix_series) >= 2:
        us_vix_val = float(us_vix_series.iloc[-1])
        us_vix_prev = float(us_vix_series.iloc[-2])
        us_vix_change = (us_vix_val - us_vix_prev) / us_vix_prev if us_vix_prev > 0 else 0.0
        sig_us_vix = us_vix_val >= us_vix_level_threshold or us_vix_change >= us_vix_change_threshold

    signals = {
        "fast_return_5d": sig_return5d,
        "consec_decline": sig_consec,
        "vol_spike": sig_vol,
        "panic_volume": sig_panic_vol,
        "vix_spike": sig_vix,
        "single_day_drop": sig_single_drop,
        "us_vix_spike": sig_us_vix,
    }
    crisis = sum(signals.values()) >= 2

    return {
        "crisis": crisis,
        "signals": signals,
        "fast_return_5d_val": ret5d_val,
        "vol_ratio_val": vol_ratio_val,
        "vix_val": vix_val,
        "us_vix_val": us_vix_val,
    }


def detect_from_series(
    closes: pd.Series,
    volumes: pd.Series | None = None,
    vix_series: pd.Series | None = None,
    us_vix_series: pd.Series | None = None,
    sma_short: int = 60,
    sma_long: int = 120,
    return_window: int = 20,
    return_threshold: float = 0.03,
    include_crisis: bool = True,
    breadth_below_ma20_pct: float | None = None,
) -> dict:
    """從收盤價序列偵測市場狀態（純函數，供測試用）。

    Args:
        closes: TAIEX 收盤價序列（需至少 sma_long 筆）
        volumes: TAIEX 成交量序列（與 closes 等長；None 則跳過爆量長黑訊號）
        vix_series: 台灣 VIX 收盤價序列（None 則跳過 VIX 訊號）
        us_vix_series: 美國 VIX 收盤價序列（None 則跳過 US VIX 訊號）
        sma_short: 短期均線天數（預設 60）
        sma_long: 長期均線天數（預設 120）
        return_window: 報酬率回溯天數（預設 20）
        return_threshold: 報酬率閾值（預設 3%）
        include_crisis: 是否啟用 crisis 快速崩盤偵測（預設 True）
        breadth_below_ma20_pct: 跌破 MA20 的股票比例（0.0~1.0）；
            > 0.60 時 regime 降一級（bull→sideways, sideways→bear）；
            None 則跳過市場寬度檢查

    Returns:
        dict: {
            "regime": "bull" | "bear" | "sideways" | "crisis",
            "signals": {"vs_sma_short": ..., "vs_sma_long": ..., "return_20d": ...},
            "taiex_close": float,
            "crisis_triggered": bool,       # crisis 是否觸發
            "crisis_signals": dict,         # 七個快速訊號的 bool 值
            "breadth_downgraded": bool,     # 市場寬度是否觸發降級
            "breadth_below_ma20_pct": float | None,  # 跌破 MA20 比例
        }
    """
    if len(closes) < sma_long:
        return {
            "regime": "sideways",
            "signals": {"vs_sma_short": "unknown", "vs_sma_long": "unknown", "return_20d": "unknown"},
            "taiex_close": closes.iloc[-1] if len(closes) > 0 else 0.0,
            "crisis_triggered": False,
            "crisis_signals": {},
            "breadth_downgraded": False,
            "breadth_below_ma20_pct": breadth_below_ma20_pct,
        }

    current = closes.iloc[-1]
    sma_s = closes.iloc[-sma_short:].mean()
    sma_l = closes.iloc[-sma_long:].mean()

    # 20 日報酬率
    if len(closes) > return_window:
        ret_20d = (current - closes.iloc[-return_window - 1]) / closes.iloc[-return_window - 1]
    else:
        ret_20d = 0.0

    # 三訊號
    signal_sma_short = "bull" if current > sma_s else "bear"
    signal_sma_long = "bull" if current > sma_l else "bear"

    if ret_20d > return_threshold:
        signal_return = "bull"
    elif ret_20d < -return_threshold:
        signal_return = "bear"
    else:
        signal_return = "sideways"

    # 多數決
    votes = [signal_sma_short, signal_sma_long, signal_return]
    bull_count = votes.count("bull")
    bear_count = votes.count("bear")

    if bull_count >= 2:
        regime = "bull"
    elif bear_count >= 2:
        regime = "bear"
    else:
        regime = "sideways"

    # 市場寬度降級（多數決之後、crisis 覆蓋之前）
    breadth_downgraded = False
    if breadth_below_ma20_pct is not None and breadth_below_ma20_pct > _BREADTH_BELOW_MA20_THRESHOLD:
        if regime == "bull":
            regime = "sideways"
            breadth_downgraded = True
        elif regime == "sideways":
            regime = "bear"
            breadth_downgraded = True

    # Crisis 快速訊號覆蓋（優先於多數決結果與寬度降級）
    crisis_info: dict = {}
    crisis_triggered = False
    if include_crisis:
        crisis_info = detect_crisis_signals(
            closes,
            volumes=volumes,
            vix_series=vix_series,
            us_vix_series=us_vix_series,
        )
        if crisis_info["crisis"]:
            regime = "crisis"
            crisis_triggered = True

    return {
        "regime": regime,
        "signals": {
            "vs_sma_short": signal_sma_short,
            "vs_sma_long": signal_sma_long,
            "return_20d": signal_return,
        },
        "taiex_close": float(current),
        "crisis_triggered": crisis_triggered,
        "crisis_signals": crisis_info.get("signals", {}),
        "fast_return_5d_val": crisis_info.get("fast_return_5d_val", 0.0),
        "vol_ratio_val": crisis_info.get("vol_ratio_val", 0.0),
        "vix_val": crisis_info.get("vix_val", 0.0),
        "us_vix_val": crisis_info.get("us_vix_val", 0.0),
        "breadth_downgraded": breadth_downgraded,
        "breadth_below_ma20_pct": breadth_below_ma20_pct,
    }


class MarketRegimeDetector:
    """市場狀態偵測器。

    從 DB 讀取 TAIEX 加權指數收盤價，判斷市場狀態。
    啟用 Hysteresis 時使用 ``RegimeStateMachine`` 進行狀態轉換確認。

    Args:
        sma_short: 短期均線天數（預設 60）
        sma_long: 長期均線天數（預設 120）
        return_window: 報酬率回溯天數（預設 20）
        use_hysteresis: 是否啟用 Hysteresis 狀態轉換遲滯（預設 True）
        state_path: Hysteresis 狀態檔路徑（None 使用預設）
    """

    def __init__(
        self,
        sma_short: int = 60,
        sma_long: int = 120,
        return_window: int = 20,
        use_hysteresis: bool = True,
        state_path: str | Path | None = None,
    ) -> None:
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.return_window = return_window
        self.use_hysteresis = use_hysteresis
        if use_hysteresis:
            self._state_machine = RegimeStateMachine(
                state_path=state_path,
                sma_short=sma_short,
                sma_long=sma_long,
                return_window=return_window,
            )

    def detect(self) -> dict:
        """偵測市場狀態。

        Returns:
            dict: {
                "regime": "bull"|"bear"|"sideways"|"crisis",
                "signals": {...},
                "taiex_close": float,
                "crisis_triggered": bool,
                "crisis_signals": dict,
                "breadth_downgraded": bool,
                "breadth_below_ma20_pct": float | None,
                "hysteresis_applied": bool,     # Hysteresis 是否修改了 regime
                "raw_regime": str | None,       # Hysteresis 前的 raw regime
                "transition_info": dict | None, # Hysteresis 轉換決策資訊
            }
        """
        with get_session() as session:
            rows = session.execute(
                select(DailyPrice.date, DailyPrice.close, DailyPrice.volume)
                .where(DailyPrice.stock_id == "TAIEX")
                .order_by(DailyPrice.date)
            ).all()

        if not rows:
            logger.warning("無 TAIEX 資料，預設為 sideways")
            return {
                "regime": "sideways",
                "signals": {"vs_sma_short": "unknown", "vs_sma_long": "unknown", "return_20d": "unknown"},
                "taiex_close": 0.0,
                "crisis_triggered": False,
                "crisis_signals": {},
                "breadth_downgraded": False,
                "breadth_below_ma20_pct": None,
                "hysteresis_applied": False,
                "raw_regime": None,
                "transition_info": None,
            }

        closes = pd.Series([r[1] for r in rows], index=[r[0] for r in rows])
        volumes_raw = pd.Series([r[2] for r in rows], index=[r[0] for r in rows])
        # TAIEX 成交量可能為 0（FinMind 部分資料源），僅在有非零資料時傳入
        volumes = volumes_raw if (volumes_raw > 0).any() else None

        # 載入 VIX 資料（graceful degradation：無資料時為 None）
        with get_session() as session:
            vix_rows = session.execute(
                select(DailyPrice.date, DailyPrice.close)
                .where(DailyPrice.stock_id == "TW_VIX")
                .order_by(DailyPrice.date)
            ).all()
        vix_series = pd.Series([r[1] for r in vix_rows], index=[r[0] for r in vix_rows]) if vix_rows else None

        # 載入美國 VIX 資料（graceful degradation：無資料時為 None）
        with get_session() as session:
            us_vix_rows = session.execute(
                select(DailyPrice.date, DailyPrice.close)
                .where(DailyPrice.stock_id == "US_VIX")
                .order_by(DailyPrice.date)
            ).all()
        us_vix_series = (
            pd.Series([r[1] for r in us_vix_rows], index=[r[0] for r in us_vix_rows]) if us_vix_rows else None
        )

        # 計算市場寬度：跌破 MA20 的股票比例
        breadth_pct = self._compute_breadth()

        if self.use_hysteresis:
            result = self._state_machine.update(
                closes,
                volumes=volumes,
                breadth_below_ma20_pct=breadth_pct,
                vix_series=vix_series,
                us_vix_series=us_vix_series,
            )
        else:
            result = detect_from_series(
                closes,
                volumes=volumes,
                vix_series=vix_series,
                us_vix_series=us_vix_series,
                sma_short=self.sma_short,
                sma_long=self.sma_long,
                return_window=self.return_window,
                breadth_below_ma20_pct=breadth_pct,
            )
            result["hysteresis_applied"] = False
            result["raw_regime"] = None
            result["transition_info"] = None

        # ── Logging ──
        if result.get("breadth_downgraded"):
            logger.warning(
                "📊 市場廣度警示：%.0f%% 股票跌破 MA20，regime 降級 → %s",
                (result.get("breadth_below_ma20_pct") or 0.0) * 100,
                result["regime"],
            )

        if result.get("hysteresis_applied"):
            ti = result.get("transition_info") or {}
            logger.info(
                "🔄 Hysteresis：raw=%s → final=%s (%s)",
                result.get("raw_regime"),
                result["regime"],
                ti.get("reason", ""),
            )

        if result.get("crisis_triggered"):
            crisis_sigs = result.get("crisis_signals", {})
            logger.warning(
                "⚠ Crisis 訊號觸發！regime 覆蓋為 crisis "
                "(5日=%+.1f%%, 連跌=%s, 波動率倍數=%.2f, 爆量長黑=%s, TW_VIX=%.1f, US_VIX=%.1f, 單日急跌=%s)",
                result.get("fast_return_5d_val", 0.0) * 100,
                crisis_sigs.get("consec_decline", False),
                result.get("vol_ratio_val", 0.0),
                crisis_sigs.get("panic_volume", False),
                result.get("vix_val", 0.0),
                result.get("us_vix_val", 0.0),
                crisis_sigs.get("single_day_drop", False),
            )
        elif not result.get("hysteresis_applied"):
            logger.info(
                "市場狀態: %s (TAIEX=%.0f, SMA%d %s, SMA%d %s, %d日報酬 %s)",
                result["regime"],
                result["taiex_close"],
                self.sma_short,
                result["signals"]["vs_sma_short"],
                self.sma_long,
                result["signals"]["vs_sma_long"],
                self.return_window,
                result["signals"]["return_20d"],
            )
        return result

    @staticmethod
    def _compute_breadth() -> float | None:
        """從 DailyFeature 表計算跌破 MA20 的股票比例。

        Returns:
            float | None: 跌破 MA20 比例（0.0~1.0），無資料時回傳 None
        """
        try:
            from src.data.schema import DailyFeature

            with get_session() as session:
                # 取最新日期
                latest_date = session.execute(select(func.max(DailyFeature.date))).scalar()
                if latest_date is None:
                    return None

                rows = session.execute(
                    select(DailyFeature.stock_id, DailyFeature.close, DailyFeature.ma20)
                    .where(DailyFeature.date == latest_date)
                    .where(DailyFeature.ma20.isnot(None))
                ).all()

            if not rows:
                return None

            closes_s = pd.Series([r[1] for r in rows], index=[r[0] for r in rows])
            ma20s_s = pd.Series([r[2] for r in rows], index=[r[0] for r in rows])
            return compute_market_breadth_pct(closes_s, ma20s_s)

        except Exception as exc:
            logger.debug("市場寬度計算失敗（DailyFeature 可能為空）: %s", exc)
            return None

    @staticmethod
    def get_weights(mode: str, regime: RegimeType) -> dict[str, float]:
        """取得指定模式 + 市場狀態下的權重。

        Args:
            mode: discover 模式名稱 ("momentum", "swing", "value", "dividend", "growth")
            regime: 市場狀態 ("bull", "bear", "sideways")

        Returns:
            dict: 各面向權重，例如 {"technical": 0.45, "chip": 0.45, "fundamental": 0.10}
        """
        if mode in REGIME_WEIGHTS and regime in REGIME_WEIGHTS[mode]:
            return REGIME_WEIGHTS[mode][regime]
        # 預設權重
        return {"technical": 0.30, "chip": 0.40, "fundamental": 0.20, "news": 0.10}
