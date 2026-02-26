"""市場狀態（Regime）偵測器。

根據加權指數（TAIEX）判斷當前市場狀態：
- bull（多頭）：指數在長短均線之上 + 短期正報酬
- bear（空頭）：指數在長短均線之下 + 短期負報酬
- sideways（盤整）：混合訊號

三訊號多數決：
1. TAIEX close vs SMA60
2. TAIEX close vs SMA120
3. 20 日報酬率 > 3% / < -3%
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import DailyPrice

logger = logging.getLogger(__name__)

RegimeType = Literal["bull", "bear", "sideways"]

# Regime 對 Discover 各模式的權重調整矩陣
REGIME_WEIGHTS: dict[str, dict[RegimeType, dict[str, float]]] = {
    "momentum": {
        "bull": {"technical": 0.45, "chip": 0.35, "fundamental": 0.10, "news": 0.10},
        "sideways": {"technical": 0.40, "chip": 0.40, "fundamental": 0.10, "news": 0.10},
        "bear": {"technical": 0.30, "chip": 0.40, "fundamental": 0.15, "news": 0.15},
    },
    "swing": {
        "bull": {"technical": 0.30, "chip": 0.20, "fundamental": 0.40, "news": 0.10},
        "sideways": {"technical": 0.25, "chip": 0.25, "fundamental": 0.35, "news": 0.15},
        "bear": {"technical": 0.15, "chip": 0.25, "fundamental": 0.45, "news": 0.15},
    },
    "value": {
        "bull": {"fundamental": 0.40, "valuation": 0.35, "chip": 0.15, "news": 0.10},
        "sideways": {"fundamental": 0.45, "valuation": 0.25, "chip": 0.15, "news": 0.15},
        "bear": {"fundamental": 0.50, "valuation": 0.20, "chip": 0.10, "news": 0.20},
    },
}


def detect_from_series(
    closes: pd.Series,
    sma_short: int = 60,
    sma_long: int = 120,
    return_window: int = 20,
    return_threshold: float = 0.03,
) -> dict:
    """從收盤價序列偵測市場狀態（純函數，供測試用）。

    Args:
        closes: TAIEX 收盤價序列（需至少 sma_long 筆）
        sma_short: 短期均線天數（預設 60）
        sma_long: 長期均線天數（預設 120）
        return_window: 報酬率回溯天數（預設 20）
        return_threshold: 報酬率閾值（預設 3%）

    Returns:
        dict: {
            "regime": "bull" | "bear" | "sideways",
            "signals": {"vs_sma_short": ..., "vs_sma_long": ..., "return_20d": ...},
            "taiex_close": float,
        }
    """
    if len(closes) < sma_long:
        return {
            "regime": "sideways",
            "signals": {"vs_sma_short": "unknown", "vs_sma_long": "unknown", "return_20d": "unknown"},
            "taiex_close": closes.iloc[-1] if len(closes) > 0 else 0.0,
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

    return {
        "regime": regime,
        "signals": {
            "vs_sma_short": signal_sma_short,
            "vs_sma_long": signal_sma_long,
            "return_20d": signal_return,
        },
        "taiex_close": float(current),
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
            dict: {"regime": "bull"|"bear"|"sideways", "signals": {...}, "taiex_close": float}
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
            }

        closes = pd.Series([r[1] for r in rows], index=[r[0] for r in rows])
        result = detect_from_series(
            closes,
            sma_short=self.sma_short,
            sma_long=self.sma_long,
            return_window=self.return_window,
        )
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
            mode: discover 模式名稱 ("momentum", "swing", "value")
            regime: 市場狀態 ("bull", "bear", "sideways")

        Returns:
            dict: 各面向權重，例如 {"technical": 0.45, "chip": 0.45, "fundamental": 0.10}
        """
        if mode in REGIME_WEIGHTS and regime in REGIME_WEIGHTS[mode]:
            return REGIME_WEIGHTS[mode][regime]
        # 預設權重
        return {"technical": 0.30, "chip": 0.40, "fundamental": 0.20, "news": 0.10}
