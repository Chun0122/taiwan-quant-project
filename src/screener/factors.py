"""因子定義 — 每個因子接收 DataFrame 回傳 bool Series（符合=True）。

因子分為三大類：
- 技術面：RSI 超賣、MACD 黃金交叉、價格站上均線
- 籌碼面：外資買超、法人連續買超、券資比
- 基本面：月營收年增率、連續營收成長
"""

from __future__ import annotations

import pandas as pd


# ─── 技術面因子 ──────────────────────────────────────────────

def rsi_oversold(df: pd.DataFrame, threshold: float = 30) -> pd.Series:
    """RSI 低於門檻值（超賣區）。

    Args:
        df: 需包含 rsi_14 欄位
        threshold: RSI 門檻值，預設 30
    """
    if "rsi_14" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["rsi_14"] < threshold


def macd_golden_cross(df: pd.DataFrame) -> pd.Series:
    """MACD 上穿 Signal 線（黃金交叉）。

    需要至少 2 天資料才能判斷交叉。
    """
    if "macd" not in df.columns or "macd_signal" not in df.columns:
        return pd.Series(False, index=df.index)

    diff = df["macd"] - df["macd_signal"]
    prev_diff = diff.shift(1)
    return (prev_diff <= 0) & (diff > 0)


def price_above_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """收盤價站上 SMA 均線。

    Args:
        df: 需包含 close 和 sma_{period} 欄位
        period: SMA 週期，預設 20
    """
    sma_col = f"sma_{period}"
    if "close" not in df.columns or sma_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df["close"] > df[sma_col]


# ─── 籌碼面因子 ──────────────────────────────────────────────

def foreign_net_buy(df: pd.DataFrame, threshold: float = 0) -> pd.Series:
    """外資買超。

    Args:
        df: 需包含 foreign_net 欄位（外資買賣超股數）
        threshold: 買超門檻，預設 0（任何正值即可）
    """
    if "foreign_net" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["foreign_net"] > threshold


def institutional_consecutive_buy(df: pd.DataFrame, days: int = 3) -> pd.Series:
    """三大法人合計連續買超天數達門檻。

    Args:
        df: 需包含 foreign_net, trust_net, dealer_net 欄位
        days: 連續天數門檻，預設 3
    """
    net_cols = ["foreign_net", "trust_net", "dealer_net"]
    available = [c for c in net_cols if c in df.columns]
    if not available:
        return pd.Series(False, index=df.index)

    total_net = df[available].sum(axis=1)
    is_buy = (total_net > 0).astype(int)

    # 計算連續正值天數：遇到 0 就重置
    consecutive = pd.Series(0, index=df.index)
    count = 0
    for i in range(len(is_buy)):
        if is_buy.iloc[i] == 1:
            count += 1
        else:
            count = 0
        consecutive.iloc[i] = count

    return consecutive >= days


def short_squeeze_ratio(df: pd.DataFrame, threshold: float = 0.2) -> pd.Series:
    """券資比超過門檻（軋空潛力）。

    券資比 = 融券餘額 / 融資餘額

    Args:
        df: 需包含 short_balance 和 margin_balance 欄位
        threshold: 券資比門檻，預設 0.2 (20%)
    """
    if "short_balance" not in df.columns or "margin_balance" not in df.columns:
        return pd.Series(False, index=df.index)

    margin = df["margin_balance"].replace(0, float("nan"))
    ratio = df["short_balance"] / margin
    return ratio > threshold


# ─── 基本面因子 ──────────────────────────────────────────────

def revenue_yoy_growth(df: pd.DataFrame, threshold: float = 20) -> pd.Series:
    """月營收年增率超過門檻。

    Args:
        df: 需包含 yoy_growth 欄位
        threshold: YoY 成長率門檻 (%)，預設 20%
    """
    if "yoy_growth" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["yoy_growth"] > threshold


def revenue_consecutive_growth(df: pd.DataFrame, months: int = 3) -> pd.Series:
    """連續月營收成長（MoM > 0）。

    Args:
        df: 需包含 mom_growth 欄位
        months: 連續成長月數門檻，預設 3
    """
    if "mom_growth" not in df.columns:
        return pd.Series(False, index=df.index)

    is_growth = (df["mom_growth"] > 0).astype(int)

    consecutive = pd.Series(0, index=df.index)
    count = 0
    for i in range(len(is_growth)):
        if is_growth.iloc[i] == 1:
            count += 1
        else:
            count = 0
        consecutive.iloc[i] = count

    return consecutive >= months


# ─── 因子清單（供 Screener 引擎使用）─────────────────────────

FACTOR_REGISTRY: dict[str, dict] = {
    "rsi_oversold": {
        "func": rsi_oversold,
        "label": "RSI 超賣 (<30)",
        "category": "技術面",
        "params": {"threshold": 30},
    },
    "macd_golden_cross": {
        "func": macd_golden_cross,
        "label": "MACD 黃金交叉",
        "category": "技術面",
        "params": {},
    },
    "price_above_sma": {
        "func": price_above_sma,
        "label": "股價 > SMA20",
        "category": "技術面",
        "params": {"period": 20},
    },
    "foreign_net_buy": {
        "func": foreign_net_buy,
        "label": "外資買超",
        "category": "籌碼面",
        "params": {"threshold": 0},
    },
    "institutional_consecutive_buy": {
        "func": institutional_consecutive_buy,
        "label": "法人連續買超 3 天",
        "category": "籌碼面",
        "params": {"days": 3},
    },
    "short_squeeze_ratio": {
        "func": short_squeeze_ratio,
        "label": "券資比 > 20%",
        "category": "籌碼面",
        "params": {"threshold": 0.2},
    },
    "revenue_yoy_growth": {
        "func": revenue_yoy_growth,
        "label": "營收 YoY > 20%",
        "category": "基本面",
        "params": {"threshold": 20},
    },
    "revenue_consecutive_growth": {
        "func": revenue_consecutive_growth,
        "label": "連續營收成長 3 月",
        "category": "基本面",
        "params": {"months": 3},
    },
}
