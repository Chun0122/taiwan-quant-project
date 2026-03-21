"""進出場建議共用純函數。

提供 Discover / Suggest / Watch 三套系統統一使用的 ATR 止損止利計算、
進場觸發文字、時機評估等純函數，消除重複邏輯並確保 Regime 自適應行為一致。
"""

from __future__ import annotations

# ── Regime 自適應 ATR 倍數常數 ─────────────────────────────────────
# (stop_multiplier, target_multiplier)
REGIME_ATR_PARAMS: dict[str, tuple[float, float]] = {
    "bull": (1.5, 3.5),
    "sideways": (1.5, 3.0),
    "bear": (1.2, 2.5),
    "crisis": (1.0, 1.8),  # 崩盤期維持合理止損距離，避免日內波動頻繁洗出
}


def compute_atr_stops(
    close: float,
    atr14: float,
    regime: str = "sideways",
) -> tuple[float | None, float | None]:
    """依 Regime 計算 ATR-based 止損與目標價（純函數）。

    Args:
        close:  當日收盤價（進場參考價）
        atr14:  14 日平均真實波幅
        regime: 市場狀態 "bull" | "sideways" | "bear" | "crisis"

    Returns:
        (stop_loss, take_profit)，atr14 ≤ 0 時回傳 (None, None)
    """
    if atr14 <= 0:
        return (None, None)

    stop_mult, target_mult = REGIME_ATR_PARAMS.get(regime, REGIME_ATR_PARAMS["sideways"])
    stop_loss = round(close - stop_mult * atr14, 2)
    take_profit = round(close + target_mult * atr14, 2)
    return (stop_loss, take_profit)


def compute_entry_trigger(
    close: float,
    sma20: float,
    atr_pct: float,
    regime: str = "sideways",
) -> str:
    """產生進場觸發文字（純函數）。

    依 SMA20 相對位置 + ATR 波動率標籤 + Crisis 警示產生中文說明。

    Args:
        close:   當日收盤價
        sma20:   SMA20 值（≤ 0 時視為資料不足）
        atr_pct: ATR14 / close（波動率比例）
        regime:  市場狀態

    Returns:
        中文觸發文字，例如 "站上均線，低波動"
    """
    # 均線位置判斷
    if sma20 > 0:
        if close > sma20 * 1.01:
            trigger = "站上均線"
        elif close >= sma20 * 0.99:
            trigger = "貼近均線"
        else:
            trigger = "均線下方，等待確認"
    else:
        trigger = "均線下方，等待確認"

    # 附加波動率說明
    if atr_pct < 0.02:
        trigger += "，低波動"
    elif atr_pct > 0.04:
        trigger += "，高波動謹慎"

    # Crisis 模式提示降低部位
    if regime == "crisis":
        trigger += "｜⚠ 崩盤期建議降低部位規模"

    return trigger


def assess_timing(
    rsi14: float,
    close: float,
    sma20: float,
    atr_pct: float,
    regime: str,
) -> str:
    """評估單股進場時機（純函數）。

    綜合 RSI14 超買/超賣位置、均線相對位置、ATR 波動率、市場 Regime，
    輸出中文時機評估字串。

    Args:
        rsi14:    最新 RSI14 值（0.0 ~ 100.0）
        close:    最新收盤價
        sma20:    SMA20 值
        atr_pct:  ATR14 / close（波動率比例）
        regime:   市場狀態 "bull" | "bear" | "sideways" | "crisis"

    Returns:
        中文時機評估字串
    """
    above_sma = sma20 > 0 and close > sma20 * 1.005

    # 波動率修飾符
    if atr_pct < 0.015:
        vol_tag = "，低波動"
    elif atr_pct > 0.04:
        vol_tag = "，高波動謹慎"
    else:
        vol_tag = ""

    # 決策矩陣
    if rsi14 >= 70:
        timing = "謹慎觀望：RSI 超買，追高風險高"
    elif rsi14 <= 30:
        if regime in ("bull", "sideways"):
            timing = "潛在反彈：RSI 超賣，留意止損"
        else:
            timing = "下跌趨勢中超賣，等待企穩訊號"
    elif regime == "crisis":
        timing = "崩盤期：大幅減碼或暫停進場"
    elif above_sma and regime == "bull":
        if rsi14 >= 55:
            timing = "積極做多：動能強勁 + 趨勢向上"
        else:
            timing = "順勢佈局：趨勢向上，動能待確認"
    elif above_sma and regime == "sideways":
        timing = "區間上軌，注意壓力，設好止損"
    elif regime == "bear":
        timing = "空頭環境，建議觀望或嚴守止損"
    else:
        timing = "等待訊號：尚未站上均線"

    return timing + vol_tag
