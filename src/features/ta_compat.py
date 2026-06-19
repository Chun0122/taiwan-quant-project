"""`ta` 套件版本相容層。

`ta` 0.5.x 使用 `n` / `ndev` / `n_slow` / `n_fast` / `n_sign` 參數，
`ta` >= 0.7 改名為 `window` / `window_dev` / `window_slow` /
`window_fast` / `window_sign`。本模組以工廠函式封裝兩版差異，呼叫端不需
關心安裝的版本（依 `__init__` 簽名自動偵測參數名）。
"""

from __future__ import annotations

import inspect

from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, SMAIndicator
from ta.volatility import BollingerBands


def _has_param(cls, name: str) -> bool:
    return name in inspect.signature(cls.__init__).parameters


def make_sma(close, period: int) -> SMAIndicator:
    """SMA 指標（period → n / window）。"""
    if _has_param(SMAIndicator, "window"):
        return SMAIndicator(close=close, window=period)
    return SMAIndicator(close=close, n=period)


def make_rsi(close, period: int = 14) -> RSIIndicator:
    """RSI 指標（period → n / window）。"""
    if _has_param(RSIIndicator, "window"):
        return RSIIndicator(close=close, window=period)
    return RSIIndicator(close=close, n=period)


def make_macd(close, slow: int = 26, fast: int = 12, sign: int = 9) -> MACD:
    """MACD 指標（n_slow/n_fast/n_sign → window_slow/window_fast/window_sign）。"""
    if _has_param(MACD, "window_slow"):
        return MACD(close=close, window_slow=slow, window_fast=fast, window_sign=sign)
    return MACD(close=close, n_slow=slow, n_fast=fast, n_sign=sign)


def make_bollinger(close, period: int = 20, ndev: int = 2) -> BollingerBands:
    """Bollinger Bands 指標（n/ndev → window/window_dev）。"""
    if _has_param(BollingerBands, "window"):
        return BollingerBands(close=close, window=period, window_dev=ndev)
    return BollingerBands(close=close, n=period, ndev=ndev)


def make_adx(high, low, close, period: int = 14) -> ADXIndicator:
    """ADX 指標（period → n / window）。"""
    if _has_param(ADXIndicator, "window"):
        return ADXIndicator(high=high, low=low, close=close, window=period)
    return ADXIndicator(high=high, low=low, close=close, n=period)
