"""全市場資料共用載入層（項目 B）。

`_cmd_discover_all` 在 scanner 迴圈前呼叫 `load_shared_market_data()` 一次，
把結果注入 5 個 scanner 的 `_load_market_data(shared=...)`，
避免各 scanner 獨立 SELECT DailyPrice/Inst/Margin/MonthlyRevenue 造成 4 次重複 I/O。

設計要點：
  - `SharedMarketData` 為 frozen dataclass，scanner 內一律 `.copy()` 避免污染。
  - 載入範圍取 **最大 lookback**（預設 80 天價量 + 180 天營收），scanner 內再做
    日期 / universe 二次過濾，語意與原 `_load_market_data` 完全一致。
  - 欄位子集與既有 `_load_market_data` 完全對齊（open/high/low/close/volume/turnover 等）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import (
    DailyPrice,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
)


@dataclass(frozen=True)
class SharedMarketData:
    """全市場資料快照（單次 discover 掃描共用）。

    DataFrame 不應被 mutate；scanner 端需要修改時務必先 `.copy()`。

    Attributes:
        df_price: 全市場日 K 線（含 OHLCV + turnover），涵蓋
            `today - (price_lookback_days + 10)` 至今。
        df_inst: 全市場三大法人買賣超（name/net）。
        df_margin: 全市場融資融券（margin_balance / short_balance）。
        df_revenue: 全市場月營收原始 raw rows（stock_id, date, yoy_growth, mom_growth），
            涵蓋 `today - revenue_days`；scanner 內部依 months=1/2/4 做 pivot。
        price_cutoff: df_price 的起始日（含）。scanner `lookback_days + 10` 必須 ≥ 此差值。
        revenue_cutoff: df_revenue 的起始日（含）。
        loaded_at: 載入時戳（debug / 快取判斷用）。
    """

    df_price: pd.DataFrame
    df_inst: pd.DataFrame
    df_margin: pd.DataFrame
    df_revenue: pd.DataFrame
    price_cutoff: date
    revenue_cutoff: date
    loaded_at: datetime


def load_shared_market_data(
    price_lookback_days: int = 80,
    revenue_days: int = 180,
) -> SharedMarketData:
    """一次性載入全市場資料，供 5 個 scanner 共用。

    Args:
        price_lookback_days: 價量資料回溯天數（scanner `lookback_days` 的最大值，預設 80）。
            內部會再加 10 天 buffer 對齊 `_load_market_data` 原行為。
        revenue_days: 月營收回溯天數（預設 180，涵蓋 Momentum `_revenue_months=4` 所需）。

    Returns:
        `SharedMarketData`：全市場四張表的 DataFrame + cutoff 日期 + 載入時戳。

    注意：
        - 不做任何 universe 過濾，scanner 內再以 `.isin(universe_ids)` 二次篩選。
        - 日期欄已轉為 Python `date`（SQLAlchemy `Date` 欄預設）。
    """
    today = date.today()
    price_cutoff = today - timedelta(days=price_lookback_days + 10)
    revenue_cutoff = today - timedelta(days=revenue_days)

    with get_session() as session:
        # --- 日K線（含 turnover 供流動性評分）---
        price_rows = session.execute(
            select(
                DailyPrice.stock_id,
                DailyPrice.date,
                DailyPrice.open,
                DailyPrice.high,
                DailyPrice.low,
                DailyPrice.close,
                DailyPrice.volume,
                DailyPrice.turnover,
            ).where(DailyPrice.date >= price_cutoff)
        ).all()
        df_price = pd.DataFrame(
            price_rows,
            columns=["stock_id", "date", "open", "high", "low", "close", "volume", "turnover"],
        )

        # --- 三大法人 ---
        inst_rows = session.execute(
            select(
                InstitutionalInvestor.stock_id,
                InstitutionalInvestor.date,
                InstitutionalInvestor.name,
                InstitutionalInvestor.net,
            ).where(InstitutionalInvestor.date >= price_cutoff)
        ).all()
        df_inst = pd.DataFrame(inst_rows, columns=["stock_id", "date", "name", "net"])

        # --- 融資融券 ---
        margin_rows = session.execute(
            select(
                MarginTrading.stock_id,
                MarginTrading.date,
                MarginTrading.margin_balance,
                MarginTrading.short_balance,
            ).where(MarginTrading.date >= price_cutoff)
        ).all()
        df_margin = pd.DataFrame(margin_rows, columns=["stock_id", "date", "margin_balance", "short_balance"])

        # --- 月營收（raw rows，scanner 內再 pivot 成 months=1/2/4 形態）---
        revenue_rows = session.execute(
            select(
                MonthlyRevenue.stock_id,
                MonthlyRevenue.date,
                MonthlyRevenue.yoy_growth,
                MonthlyRevenue.mom_growth,
            ).where(MonthlyRevenue.date >= revenue_cutoff)
        ).all()
        df_revenue = pd.DataFrame(revenue_rows, columns=["stock_id", "date", "yoy_growth", "mom_growth"])

    return SharedMarketData(
        df_price=df_price,
        df_inst=df_inst,
        df_margin=df_margin,
        df_revenue=df_revenue,
        price_cutoff=price_cutoff,
        revenue_cutoff=revenue_cutoff,
        loaded_at=datetime.now(UTC),
    )


def slice_revenue_raw(
    df_revenue: pd.DataFrame,
    stock_ids: list[str] | None,
    months: int,
) -> pd.DataFrame:
    """從共用的 df_revenue raw rows 產生 scanner 所需的 pivoted DataFrame。

    行為與 `MarketScanner._load_revenue_data` 完全一致：
      - months <= 1：每支股票取最新一筆，回傳 ``[stock_id, yoy_growth, mom_growth]``
      - months >= 2：每支股票取最近 N 筆，回傳 ``[stock_id, yoy_growth, mom_growth,
        prev_yoy_growth, prev_mom_growth]``
      - months >= 4：額外補 ``yoy_3m_ago``

    Args:
        df_revenue: `SharedMarketData.df_revenue`（raw rows）。
        stock_ids: 限定股票清單；None 表示全市場。
        months: 與 scanner `_revenue_months` 一致（1 / 2 / 4）。

    Returns:
        每支股票一列的 DataFrame（欄位依 months 變動）。
    """
    # 依 months 決定回傳欄位
    base_cols = ["stock_id", "yoy_growth", "mom_growth"]
    if months <= 1:
        empty_cols = base_cols
    elif months < 4:
        empty_cols = base_cols + ["prev_yoy_growth", "prev_mom_growth"]
    else:
        empty_cols = base_cols + ["prev_yoy_growth", "prev_mom_growth", "yoy_3m_ago"]

    if df_revenue.empty:
        return pd.DataFrame(columns=empty_cols)

    df = df_revenue
    if stock_ids:
        df = df[df["stock_id"].isin(stock_ids)]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    if months <= 1:
        # 每支股票取 date 最新一筆
        idx = df.groupby("stock_id")["date"].idxmax()
        latest = df.loc[idx, ["stock_id", "yoy_growth", "mom_growth"]].reset_index(drop=True)
        return latest

    # months >= 2：每支股票取最近 N 筆（date desc），計算 prev_*
    sorted_df = df.sort_values(["stock_id", "date"], ascending=[True, False])
    result_rows = []
    for sid, grp in sorted_df.groupby("stock_id", sort=False):
        grp = grp.head(months)
        latest = grp.iloc[0]
        row = {
            "stock_id": sid,
            "yoy_growth": latest["yoy_growth"],
            "mom_growth": latest["mom_growth"],
            "prev_yoy_growth": grp.iloc[1]["yoy_growth"] if len(grp) >= 2 else None,
            "prev_mom_growth": grp.iloc[1]["mom_growth"] if len(grp) >= 2 else None,
        }
        if months >= 4:
            row["yoy_3m_ago"] = grp.iloc[3]["yoy_growth"] if len(grp) >= 4 else None
        result_rows.append(row)

    return pd.DataFrame(result_rows, columns=empty_cols)
