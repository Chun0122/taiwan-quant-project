"""產業輪動分析引擎 — 計算法人動能 + 價格動能，找出熱門產業。

使用方式：
    analyzer = IndustryRotationAnalyzer()
    sector_df = analyzer.rank_sectors()
    top_stocks = analyzer.top_stocks_from_hot_sectors(sector_df)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select, func

from src.config import settings
from src.data.database import get_session, init_db
from src.data.schema import DailyPrice, InstitutionalInvestor, StockInfo

logger = logging.getLogger(__name__)


class IndustryRotationAnalyzer:
    """產業輪動分析器。"""

    def __init__(
        self,
        watchlist: list[str] | None = None,
        lookback_days: int = 20,
        momentum_days: int = 60,
    ) -> None:
        self.watchlist = watchlist or settings.fetcher.watchlist
        self.lookback_days = lookback_days
        self.momentum_days = momentum_days
        init_db()

    def get_industry_map(self) -> dict[str, str]:
        """從 StockInfo 表取得股票→產業對照。

        Returns:
            {stock_id: industry_category}，不在表中的股票標為「未分類」
        """
        with get_session() as session:
            rows = session.execute(
                select(StockInfo.stock_id, StockInfo.industry_category)
            ).all()

        db_map = {r[0]: r[1] or "未分類" for r in rows}

        result = {}
        for sid in self.watchlist:
            result[sid] = db_map.get(sid, "未分類")

        return result

    def compute_sector_institutional_flow(self) -> pd.DataFrame:
        """計算各產業法人資金流向。

        Returns:
            DataFrame: [industry, total_net, stock_count, avg_net_per_stock]
        """
        industry_map = self.get_industry_map()
        end_date = date.today()
        start_date = end_date - timedelta(days=self.lookback_days * 2)

        with get_session() as session:
            rows = session.execute(
                select(InstitutionalInvestor)
                .where(InstitutionalInvestor.stock_id.in_(self.watchlist))
                .where(InstitutionalInvestor.date >= start_date)
                .order_by(InstitutionalInvestor.date.desc())
            ).scalars().all()

        if not rows:
            return pd.DataFrame(columns=["industry", "total_net", "stock_count", "avg_net_per_stock"])

        df = pd.DataFrame([
            {"stock_id": r.stock_id, "date": r.date, "name": r.name, "net": r.net}
            for r in rows
        ])

        # 只取最近 lookback_days 個交易日
        unique_dates = sorted(df["date"].unique(), reverse=True)
        if len(unique_dates) > self.lookback_days:
            cutoff = unique_dates[self.lookback_days - 1]
            df = df[df["date"] >= cutoff]

        # 加入產業分類
        df["industry"] = df["stock_id"].map(industry_map).fillna("未分類")

        # 按產業 + 股票加總
        stock_net = df.groupby(["industry", "stock_id"])["net"].sum().reset_index()
        sector_flow = stock_net.groupby("industry").agg(
            total_net=("net", "sum"),
            stock_count=("stock_id", "nunique"),
        ).reset_index()
        sector_flow["avg_net_per_stock"] = sector_flow["total_net"] / sector_flow["stock_count"]
        sector_flow = sector_flow.sort_values("total_net", ascending=False).reset_index(drop=True)

        return sector_flow

    def compute_sector_price_momentum(self) -> pd.DataFrame:
        """計算各產業價格動能（期間漲跌幅均值）。

        Returns:
            DataFrame: [industry, avg_return_pct, stock_count]
        """
        industry_map = self.get_industry_map()
        end_date = date.today()
        start_date = end_date - timedelta(days=self.momentum_days * 2)

        with get_session() as session:
            rows = session.execute(
                select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close)
                .where(DailyPrice.stock_id.in_(self.watchlist))
                .where(DailyPrice.date >= start_date)
                .order_by(DailyPrice.date)
            ).all()

        if not rows:
            return pd.DataFrame(columns=["industry", "avg_return_pct", "stock_count"])

        df = pd.DataFrame([{"stock_id": r[0], "date": r[1], "close": r[2]} for r in rows])

        # 計算每支股票的期間漲跌幅
        returns = []
        for sid, grp in df.groupby("stock_id"):
            grp = grp.sort_values("date")
            if len(grp) < 2:
                continue
            # 取最近 momentum_days 個交易日
            grp = grp.tail(self.momentum_days)
            first_close = grp.iloc[0]["close"]
            last_close = grp.iloc[-1]["close"]
            if first_close > 0:
                ret_pct = (last_close / first_close - 1) * 100
                returns.append({
                    "stock_id": sid,
                    "return_pct": ret_pct,
                    "industry": industry_map.get(sid, "未分類"),
                })

        if not returns:
            return pd.DataFrame(columns=["industry", "avg_return_pct", "stock_count"])

        df_ret = pd.DataFrame(returns)
        sector_momentum = df_ret.groupby("industry").agg(
            avg_return_pct=("return_pct", "mean"),
            stock_count=("stock_id", "nunique"),
        ).reset_index()
        sector_momentum = sector_momentum.sort_values("avg_return_pct", ascending=False).reset_index(drop=True)

        return sector_momentum

    def rank_sectors(
        self,
        inst_weight: float = 0.5,
        momentum_weight: float = 0.5,
    ) -> pd.DataFrame:
        """綜合排名各產業（法人動能 + 價格動能）。

        Args:
            inst_weight: 法人分數權重
            momentum_weight: 價格動能權重

        Returns:
            DataFrame: [rank, industry, sector_score, institutional_score,
                        momentum_score, total_net, avg_return_pct, stock_count]
        """
        flow_df = self.compute_sector_institutional_flow()
        mom_df = self.compute_sector_price_momentum()

        if flow_df.empty and mom_df.empty:
            return pd.DataFrame()

        # 合併兩張表
        if flow_df.empty:
            merged = mom_df.copy()
            merged["total_net"] = 0
            merged["avg_net_per_stock"] = 0
        elif mom_df.empty:
            merged = flow_df.copy()
            merged["avg_return_pct"] = 0
        else:
            merged = pd.merge(
                flow_df[["industry", "total_net", "avg_net_per_stock", "stock_count"]],
                mom_df[["industry", "avg_return_pct"]],
                on="industry",
                how="outer",
            )
            merged = merged.fillna(0)

        if merged.empty:
            return pd.DataFrame()

        # Min-Max 標準化
        def _minmax(series: pd.Series) -> pd.Series:
            mn, mx = series.min(), series.max()
            if mx == mn:
                return pd.Series(0.5, index=series.index)
            return (series - mn) / (mx - mn)

        merged["institutional_score"] = _minmax(merged["total_net"])
        merged["momentum_score"] = _minmax(merged["avg_return_pct"])
        merged["sector_score"] = (
            inst_weight * merged["institutional_score"]
            + momentum_weight * merged["momentum_score"]
        )

        merged = merged.sort_values("sector_score", ascending=False).reset_index(drop=True)
        merged["rank"] = range(1, len(merged) + 1)

        cols = [
            "rank", "industry", "sector_score", "institutional_score",
            "momentum_score", "total_net", "avg_return_pct", "stock_count",
        ]
        return merged[[c for c in cols if c in merged.columns]]

    def top_stocks_from_hot_sectors(
        self,
        sector_df: pd.DataFrame,
        top_sectors: int = 3,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """從熱門產業中選出精選個股。

        Args:
            sector_df: rank_sectors() 的結果
            top_sectors: 取前 N 名產業
            top_n: 每產業選前 N 支

        Returns:
            DataFrame: [industry, stock_id, stock_name, close, foreign_net_sum, rank_in_sector]
        """
        if sector_df.empty:
            return pd.DataFrame()

        hot_industries = sector_df.head(top_sectors)["industry"].tolist()
        industry_map = self.get_industry_map()

        # 找出屬於熱門產業的股票
        hot_stocks = [sid for sid, ind in industry_map.items() if ind in hot_industries]
        if not hot_stocks:
            return pd.DataFrame()

        end_date = date.today()
        start_date = end_date - timedelta(days=self.lookback_days * 2)

        with get_session() as session:
            # 取得最新收盤價
            price_rows = session.execute(
                select(DailyPrice.stock_id, DailyPrice.close)
                .where(DailyPrice.stock_id.in_(hot_stocks))
                .where(DailyPrice.date >= start_date)
                .order_by(DailyPrice.date.desc())
            ).all()

            # 取得外資淨買超合計
            inst_rows = session.execute(
                select(
                    InstitutionalInvestor.stock_id,
                    func.sum(InstitutionalInvestor.net).label("foreign_net_sum"),
                )
                .where(InstitutionalInvestor.stock_id.in_(hot_stocks))
                .where(InstitutionalInvestor.date >= start_date)
                .where(InstitutionalInvestor.name.in_(["Foreign_Investor", "外資", "外資及陸資"]))
                .group_by(InstitutionalInvestor.stock_id)
            ).all()

            # 取得股票名稱
            info_rows = session.execute(
                select(StockInfo.stock_id, StockInfo.stock_name)
                .where(StockInfo.stock_id.in_(hot_stocks))
            ).all()

        # 最新收盤價（每支取第一筆 = 最新）
        price_map = {}
        for r in price_rows:
            if r[0] not in price_map:
                price_map[r[0]] = r[1]

        inst_map = {r[0]: r[1] for r in inst_rows}
        name_map = {r[0]: r[1] for r in info_rows}

        records = []
        for sid in hot_stocks:
            records.append({
                "industry": industry_map[sid],
                "stock_id": sid,
                "stock_name": name_map.get(sid, ""),
                "close": price_map.get(sid, 0),
                "foreign_net_sum": inst_map.get(sid, 0),
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # 按產業內外資淨買超排序，每產業取 top_n
        result_parts = []
        for ind in hot_industries:
            sector_stocks = df[df["industry"] == ind].sort_values(
                "foreign_net_sum", ascending=False
            ).head(top_n).copy()
            sector_stocks["rank_in_sector"] = range(1, len(sector_stocks) + 1)
            result_parts.append(sector_stocks)

        if not result_parts:
            return pd.DataFrame()

        return pd.concat(result_parts, ignore_index=True)
