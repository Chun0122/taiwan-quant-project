"""Scanner 測試共用資料建構函數，供 test_scanner.py 等使用。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def make_price_df(n_stocks: int = 20, n_days: int = 20) -> pd.DataFrame:
    """建立模擬市場日 K 資料（n_days 天，預設 20 天，供 ATR14 計算使用）。"""
    rows = []
    base_date = date(2025, 1, 2)
    for i in range(n_stocks):
        sid = f"{1000 + i}"
        base_close = 50 + i * 10
        for d in range(n_days):
            day = base_date + timedelta(days=d)
            close = base_close + d * 0.5 + (i - n_stocks // 2) * 0.1
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": close - 1,
                    "high": close + 2,
                    "low": close - 2,
                    "close": close,
                    "volume": 200_000 + i * 50_000,
                }
            )
    return pd.DataFrame(rows)


def make_inst_df(stock_ids: list[str], target_date: date) -> pd.DataFrame:
    """建立模擬法人買賣超資料。"""
    rows = []
    for sid in stock_ids:
        rows.append(
            {
                "stock_id": sid,
                "date": target_date,
                "name": "Foreign_Investor",
                "net": int(sid) % 3 * 1000 - 500,
            }
        )
        rows.append(
            {
                "stock_id": sid,
                "date": target_date,
                "name": "Investment_Trust",
                "net": int(sid) % 5 * 200 - 200,
            }
        )
    return pd.DataFrame(rows)


def make_momentum_price_df(n_days: int = 25, n_stocks: int = 5) -> pd.DataFrame:
    """建立動能模式所需的多日市場資料。"""
    rows = []
    for i in range(n_stocks):
        sid = f"{2000 + i}"
        for d in range(n_days):
            day = date(2025, 1, 1) + timedelta(days=d)
            base_close = 100 + i * 20
            close = base_close * (1 + 0.002 * i) ** d
            vol = 500_000 + i * 100_000 + d * 10_000 * (i + 1)
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": int(vol),
                }
            )
    return pd.DataFrame(rows)


def make_swing_price_df(n_days: int = 80, n_stocks: int = 5) -> pd.DataFrame:
    """建立波段模式所需的長期市場資料。"""
    rows = []
    for i in range(n_stocks):
        sid = f"{4000 + i}"
        for d in range(n_days):
            day = date(2025, 1, 1) + timedelta(days=d)
            base_close = 80 + i * 30
            close = base_close * (1 + 0.001 * (i + 1)) ** d
            vol = 300_000 + i * 50_000 + d * 5_000
            rows.append(
                {
                    "stock_id": sid,
                    "date": day,
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": int(vol),
                }
            )
    return pd.DataFrame(rows)


def make_entry_exit_price_df(sid: str = "1000", n_days: int = 20) -> pd.DataFrame:
    """建立供進出場計算用的模擬價格資料。"""
    rows = []
    for d in range(n_days):
        day = date(2025, 1, 1) + timedelta(days=d)
        close = 100.0 + d * 0.5
        rows.append(
            {
                "stock_id": sid,
                "date": day,
                "open": close - 1,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": 500_000,
            }
        )
    return pd.DataFrame(rows)
