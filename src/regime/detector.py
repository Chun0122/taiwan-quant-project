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
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import DailyPrice

logger = logging.getLogger(__name__)

RegimeType = Literal["bull", "bear", "sideways", "crisis"]

# Crisis 快速崩盤偵測門檻
_CRISIS_RETURN_5D: float = -0.05  # 5 日報酬率 < -5%
_CRISIS_CONSEC_DOWN: int = 3  # 連跌 ≥ 3 天
_CRISIS_VOL_RATIO: float = 1.8  # rolling-20d-vol / avg-vol-120d > 1.8

# Regime 對 Discover 各模式的權重調整矩陣
# crisis 設計邏輯：崩盤時技術訊號失真（跳空），降至接近 0；
# 新聞/基本面（品質防禦）提升至最高，只留有真實催化劑的防禦股
REGIME_WEIGHTS: dict[str, dict[str, dict[str, float]]] = {
    "momentum": {
        # Bull：技術/籌碼等重 40/40，降低「技術突破 + 外資買超同步」的共線性偏誤
        "bull": {"technical": 0.40, "chip": 0.40, "fundamental": 0.10, "news": 0.10},
        # Sideways：籌碼面拉至 50%，盤整期 Smart Broker 蓄積訊號最有效；壓縮技術面避免追假突破
        "sideways": {"technical": 0.30, "chip": 0.50, "fundamental": 0.10, "news": 0.10},
        # Bear：技術面降至 25%，消息面提升至 20%，確保選出有事件催化劑的錯殺股
        "bear": {"technical": 0.25, "chip": 0.40, "fundamental": 0.15, "news": 0.20},
        # Crisis：技術訊號失真，籌碼為主要防禦指標，消息面催化劑最高優先
        "crisis": {"technical": 0.10, "chip": 0.30, "fundamental": 0.20, "news": 0.40},
    },
    "swing": {
        "bull": {"technical": 0.30, "chip": 0.20, "fundamental": 0.40, "news": 0.10},
        "sideways": {"technical": 0.25, "chip": 0.25, "fundamental": 0.35, "news": 0.15},
        "bear": {"technical": 0.15, "chip": 0.25, "fundamental": 0.45, "news": 0.15},
        "crisis": {"technical": 0.05, "chip": 0.20, "fundamental": 0.50, "news": 0.25},
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
        "crisis": {"fundamental": 0.55, "technical": 0.05, "chip": 0.15, "news": 0.25},
    },
}


def detect_crisis_signals(
    closes: pd.Series,
    return_5d_threshold: float = _CRISIS_RETURN_5D,
    consec_down_days: int = _CRISIS_CONSEC_DOWN,
    vol_ratio_threshold: float = _CRISIS_VOL_RATIO,
    vol_window: int = 20,
    vol_baseline: int = 120,
) -> dict:
    """從收盤價序列偵測快速崩盤訊號（純函數，供測試用）。

    三個快速訊號，任意 ≥2 個觸發 → crisis：
    1. fast_return_5d：5 日報酬率 < return_5d_threshold（預設 -5%）
    2. consec_decline：最後 consec_down_days 天連續收跌
    3. vol_spike：rolling-20d-std(daily_returns) / avg-rolling-20d-std(120d) > vol_ratio_threshold

    Args:
        closes: TAIEX 收盤價序列（pd.Series，時序由舊至新）
        return_5d_threshold: 5 日跌幅門檻（預設 -0.05 即 -5%）
        consec_down_days: 連跌天數門檻（預設 3）
        vol_ratio_threshold: 波動率倍數門檻（預設 1.8）
        vol_window: 近期波動率計算窗口（預設 20 天）
        vol_baseline: 基準波動率回溯天數（預設 120 天）

    Returns:
        dict: {
            "crisis": bool,
            "signals": {
                "fast_return_5d": bool,
                "consec_decline": bool,
                "vol_spike": bool,
            },
            "fast_return_5d_val": float,
            "vol_ratio_val": float,
        }
    """
    _safe = {
        "crisis": False,
        "signals": {"fast_return_5d": False, "consec_decline": False, "vol_spike": False},
        "fast_return_5d_val": 0.0,
        "vol_ratio_val": 0.0,
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

    signals = {
        "fast_return_5d": sig_return5d,
        "consec_decline": sig_consec,
        "vol_spike": sig_vol,
    }
    crisis = sum(signals.values()) >= 2

    return {
        "crisis": crisis,
        "signals": signals,
        "fast_return_5d_val": ret5d_val,
        "vol_ratio_val": vol_ratio_val,
    }


def detect_from_series(
    closes: pd.Series,
    sma_short: int = 60,
    sma_long: int = 120,
    return_window: int = 20,
    return_threshold: float = 0.03,
    include_crisis: bool = True,
) -> dict:
    """從收盤價序列偵測市場狀態（純函數，供測試用）。

    Args:
        closes: TAIEX 收盤價序列（需至少 sma_long 筆）
        sma_short: 短期均線天數（預設 60）
        sma_long: 長期均線天數（預設 120）
        return_window: 報酬率回溯天數（預設 20）
        return_threshold: 報酬率閾值（預設 3%）
        include_crisis: 是否啟用 crisis 快速崩盤偵測（預設 True）

    Returns:
        dict: {
            "regime": "bull" | "bear" | "sideways" | "crisis",
            "signals": {"vs_sma_short": ..., "vs_sma_long": ..., "return_20d": ...},
            "taiex_close": float,
            "crisis_triggered": bool,       # crisis 是否觸發
            "crisis_signals": dict,         # 三個快速訊號的 bool 值
        }
    """
    if len(closes) < sma_long:
        return {
            "regime": "sideways",
            "signals": {"vs_sma_short": "unknown", "vs_sma_long": "unknown", "return_20d": "unknown"},
            "taiex_close": closes.iloc[-1] if len(closes) > 0 else 0.0,
            "crisis_triggered": False,
            "crisis_signals": {},
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

    # Crisis 快速訊號覆蓋（優先於多數決結果）
    crisis_info: dict = {}
    crisis_triggered = False
    if include_crisis:
        crisis_info = detect_crisis_signals(closes)
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
    }


class MarketRegimeDetector:
    """市場狀態偵測器。

    從 DB 讀取 TAIEX 加權指數收盤價，判斷市場狀態。

    Args:
        sma_short: 短期均線天數（預設 60）
        sma_long: 長期均線天數（預設 120）
        return_window: 報酬率回溯天數（預設 20）
    """

    def __init__(self, sma_short: int = 60, sma_long: int = 120, return_window: int = 20) -> None:
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.return_window = return_window

    def detect(self) -> dict:
        """偵測市場狀態。

        Returns:
            dict: {
                "regime": "bull"|"bear"|"sideways"|"crisis",
                "signals": {...},
                "taiex_close": float,
                "crisis_triggered": bool,
                "crisis_signals": dict,
            }
        """
        with get_session() as session:
            rows = session.execute(
                select(DailyPrice.date, DailyPrice.close)
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
            }

        closes = pd.Series([r[1] for r in rows], index=[r[0] for r in rows])
        result = detect_from_series(
            closes,
            sma_short=self.sma_short,
            sma_long=self.sma_long,
            return_window=self.return_window,
        )

        if result.get("crisis_triggered"):
            crisis_sigs = result.get("crisis_signals", {})
            logger.warning(
                "⚠ Crisis 訊號觸發！regime 覆蓋為 crisis (5日=%+.1f%%, 連跌=%s, 波動率倍數=%.2f)",
                result.get("fast_return_5d_val", 0.0) * 100,
                crisis_sigs.get("consec_decline", False),
                result.get("vol_ratio_val", 0.0),
            )
        else:
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
