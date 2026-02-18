"""全市場選股掃描器 — 四階段漏斗從 ~2000 支股票篩選出 Top N 推薦。

漏斗架構：
  Stage 1: 從 DB 載入全市場日K + 法人資料
  Stage 2: 粗篩（股價/成交量/法人/動能 → 留 ~150 檔）
  Stage 3: 細評（技術面 35% + 籌碼面 45% + 基本面 20%）
  Stage 4: 排名 + 加上產業標籤 → 輸出 Top N
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import DailyPrice, InstitutionalInvestor, MonthlyRevenue, StockInfo

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """掃描結果資料容器。"""

    rankings: pd.DataFrame
    total_stocks: int
    after_coarse: int
    scan_date: date = field(default_factory=date.today)
    sector_summary: pd.DataFrame | None = None


class MarketScanner:
    """全市場選股掃描器。

    Args:
        min_price: 最低股價門檻
        max_price: 最高股價門檻
        min_volume: 最低成交量（股）
        top_n_candidates: 粗篩後保留數量
        top_n_results: 最終輸出數量
        lookback_days: 回溯天數（用於計算指標）
    """

    def __init__(
        self,
        min_price: float = 10,
        max_price: float = 2000,
        min_volume: int = 500_000,
        top_n_candidates: int = 150,
        top_n_results: int = 30,
        lookback_days: int = 5,
    ) -> None:
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
        self.top_n_candidates = top_n_candidates
        self.top_n_results = top_n_results
        self.lookback_days = lookback_days

    def run(self) -> DiscoveryResult:
        """執行四階段漏斗掃描。"""
        # Stage 1: 載入資料
        df_price, df_inst, df_revenue = self._load_market_data()
        if df_price.empty:
            logger.warning("無市場資料可供掃描")
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=0,
                after_coarse=0,
            )

        total_stocks = df_price["stock_id"].nunique()
        logger.info("Stage 1: 載入 %d 支股票的市場資料", total_stocks)

        # Stage 2: 粗篩
        candidates = self._coarse_filter(df_price, df_inst)
        after_coarse = len(candidates)
        logger.info("Stage 2: 粗篩後剩 %d 支候選股", after_coarse)

        if candidates.empty:
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=total_stocks,
                after_coarse=0,
            )

        # Stage 3: 細評
        scored = self._score_candidates(candidates, df_price, df_inst, df_revenue)
        logger.info("Stage 3: 完成 %d 支候選股評分", len(scored))

        # Stage 4: 排名 + 產業標籤
        rankings = self._rank_and_enrich(scored)
        sector_summary = self._compute_sector_summary(rankings)
        logger.info("Stage 4: 輸出 Top %d", min(self.top_n_results, len(rankings)))

        return DiscoveryResult(
            rankings=rankings.head(self.top_n_results),
            total_stocks=total_stocks,
            after_coarse=after_coarse,
            sector_summary=sector_summary,
        )

    # ------------------------------------------------------------------ #
    #  Stage 1: 載入資料
    # ------------------------------------------------------------------ #

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """從 DB 查詢最近的 daily_price + institutional + monthly_revenue 資料。"""
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)
        revenue_cutoff = date.today() - timedelta(days=90)

        with get_session() as session:
            # 日K線
            rows = session.execute(
                select(
                    DailyPrice.stock_id,
                    DailyPrice.date,
                    DailyPrice.open,
                    DailyPrice.high,
                    DailyPrice.low,
                    DailyPrice.close,
                    DailyPrice.volume,
                ).where(DailyPrice.date >= cutoff)
            ).all()
            df_price = pd.DataFrame(
                rows,
                columns=[
                    "stock_id",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )

            # 三大法人
            rows = session.execute(
                select(
                    InstitutionalInvestor.stock_id,
                    InstitutionalInvestor.date,
                    InstitutionalInvestor.name,
                    InstitutionalInvestor.net,
                ).where(InstitutionalInvestor.date >= cutoff)
            ).all()
            df_inst = pd.DataFrame(
                rows,
                columns=[
                    "stock_id",
                    "date",
                    "name",
                    "net",
                ],
            )

            # 月營收（每支股票取最新一筆）
            from sqlalchemy import func

            subq = (
                select(
                    MonthlyRevenue.stock_id,
                    func.max(MonthlyRevenue.date).label("max_date"),
                )
                .where(MonthlyRevenue.date >= revenue_cutoff)
                .group_by(MonthlyRevenue.stock_id)
                .subquery()
            )
            rows = session.execute(
                select(
                    MonthlyRevenue.stock_id,
                    MonthlyRevenue.yoy_growth,
                    MonthlyRevenue.mom_growth,
                ).join(
                    subq,
                    (MonthlyRevenue.stock_id == subq.c.stock_id) & (MonthlyRevenue.date == subq.c.max_date),
                )
            ).all()
            df_revenue = pd.DataFrame(
                rows,
                columns=["stock_id", "yoy_growth", "mom_growth"],
            )

        return df_price, df_inst, df_revenue

    # ------------------------------------------------------------------ #
    #  Stage 2: 粗篩
    # ------------------------------------------------------------------ #

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """粗篩：股價/量/法人/動能加權 → 取 top N candidates。"""
        # 取每支股票最近一天的收盤價和成交量
        latest_date = df_price["date"].max()
        latest = df_price[df_price["date"] == latest_date].copy()

        if latest.empty:
            return pd.DataFrame()

        # 基本過濾：股價範圍 + 成交量
        mask = (
            (latest["close"] >= self.min_price)
            & (latest["close"] <= self.max_price)
            & (latest["volume"] >= self.min_volume)
        )
        # 排除指數類（如 TAIEX）
        mask = mask & (~latest["stock_id"].str.contains(r"[A-Za-z]", na=False))
        filtered = latest[mask].copy()

        if filtered.empty:
            return pd.DataFrame()

        # 計算粗篩分數
        # 1) 成交量排名分數（量大加分）
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 2) 法人淨買超排名
        if not df_inst.empty:
            inst_latest = df_inst[df_inst["date"] == df_inst["date"].max()]
            inst_net = inst_latest.groupby("stock_id")["net"].sum().reset_index()
            inst_net.columns = ["stock_id", "inst_net"]
            filtered = filtered.merge(inst_net, on="stock_id", how="left")
            filtered["inst_net"] = filtered["inst_net"].fillna(0)
            filtered["inst_rank"] = filtered["inst_net"].rank(pct=True)
        else:
            filtered["inst_net"] = 0
            filtered["inst_rank"] = 0.5

        # 3) 動能：用最近日K的漲跌幅
        # 找前一日收盤
        dates = sorted(df_price["date"].unique())
        if len(dates) >= 2:
            prev_date = dates[-2]
            prev = df_price[df_price["date"] == prev_date][["stock_id", "close"]].copy()
            prev.columns = ["stock_id", "prev_close"]
            filtered = filtered.merge(prev, on="stock_id", how="left")
            filtered["momentum"] = ((filtered["close"] - filtered["prev_close"]) / filtered["prev_close"]).fillna(0)
            filtered["mom_rank"] = filtered["momentum"].rank(pct=True)
        else:
            filtered["momentum"] = 0
            filtered["mom_rank"] = 0.5

        # 粗篩綜合分 = 成交量 30% + 法人 40% + 動能 30%
        filtered["coarse_score"] = (
            filtered["vol_rank"] * 0.30 + filtered["inst_rank"] * 0.40 + filtered["mom_rank"] * 0.30
        )

        # 取 top N
        filtered = filtered.nlargest(self.top_n_candidates, "coarse_score")
        return filtered

    # ------------------------------------------------------------------ #
    #  Stage 3: 細評
    # ------------------------------------------------------------------ #

    def _score_candidates(
        self,
        candidates: pd.DataFrame,
        df_price: pd.DataFrame,
        df_inst: pd.DataFrame,
        df_revenue: pd.DataFrame,
    ) -> pd.DataFrame:
        """對候選股進行三維度評分：技術(35%) + 籌碼(45%) + 基本面(20%)。"""
        stock_ids = candidates["stock_id"].tolist()

        # --- 技術分數 ---
        tech_scores = self._compute_technical_scores(stock_ids, df_price)

        # --- 籌碼分數 ---
        chip_scores = self._compute_chip_scores(stock_ids, df_inst)

        # --- 基本面分數 ---
        fund_scores = self._compute_fundamental_scores(stock_ids, df_revenue)

        candidates = candidates.copy()
        candidates = candidates.merge(tech_scores, on="stock_id", how="left")
        candidates = candidates.merge(chip_scores, on="stock_id", how="left")
        candidates = candidates.merge(fund_scores, on="stock_id", how="left")
        candidates["technical_score"] = candidates["technical_score"].fillna(0.5)
        candidates["chip_score"] = candidates["chip_score"].fillna(0.5)
        candidates["fundamental_score"] = candidates["fundamental_score"].fillna(0.5)

        # 綜合分數
        candidates["composite_score"] = (
            candidates["technical_score"] * 0.35
            + candidates["chip_score"] * 0.45
            + candidates["fundamental_score"] * 0.20
        )

        return candidates

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """從原始 OHLCV 計算技術面分數（SMA、動能、價格位置）。"""
        results = []

        for sid in stock_ids:
            stock_data = df_price[df_price["stock_id"] == sid].sort_values("date")
            if len(stock_data) < 3:
                results.append({"stock_id": sid, "technical_score": 0.5})
                continue

            closes = stock_data["close"].values
            highs = stock_data["high"].values
            lows = stock_data["low"].values

            score = 0.0
            n_factors = 0

            # 1) SMA 趨勢：收盤 > SMA5 → 加分
            if len(closes) >= 5:
                sma5 = closes[-5:].mean()
                score += 1.0 if closes[-1] > sma5 else 0.0
                n_factors += 1

            # 2) 短期動能：最近 3 日漲幅
            if len(closes) >= 4:
                ret_3d = (closes[-1] - closes[-4]) / closes[-4]
                # 歸一化到 0~1
                score += max(0.0, min(1.0, 0.5 + ret_3d * 10))
                n_factors += 1

            # 3) 價格位置：收盤在近期高低區間的位置
            if len(closes) >= 3:
                high_max = highs[-5:].max() if len(highs) >= 5 else highs.max()
                low_min = lows[-5:].min() if len(lows) >= 5 else lows.min()
                price_range = high_max - low_min
                if price_range > 0:
                    position = (closes[-1] - low_min) / price_range
                    score += position
                else:
                    score += 0.5
                n_factors += 1

            # 4) 成交量趨勢：最新量 > 平均量
            volumes = stock_data["volume"].values
            if len(volumes) >= 3:
                avg_vol = volumes[:-1].mean()
                if avg_vol > 0:
                    vol_ratio = min(2.0, volumes[-1] / avg_vol)
                    score += vol_ratio / 2.0
                else:
                    score += 0.5
                n_factors += 1

            tech_score = score / max(n_factors, 1)
            results.append({"stock_id": sid, "technical_score": tech_score})

        return pd.DataFrame(results)

    def _compute_chip_scores(self, stock_ids: list[str], df_inst: pd.DataFrame) -> pd.DataFrame:
        """計算籌碼面分數（三大法人買賣超）。"""
        if df_inst.empty:
            return pd.DataFrame(
                {
                    "stock_id": stock_ids,
                    "chip_score": [0.5] * len(stock_ids),
                }
            )

        results = []
        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]

        for sid in stock_ids:
            stock_inst = inst_filtered[inst_filtered["stock_id"] == sid]
            if stock_inst.empty:
                results.append({"stock_id": sid, "chip_score": 0.5})
                continue

            score = 0.0
            n_factors = 0

            # 外資淨買超
            foreign_data = stock_inst[stock_inst["name"].str.contains("外資", na=False)]
            if not foreign_data.empty:
                foreign_net = foreign_data["net"].sum()
                # 歸一化：正值加分
                score += 1.0 if foreign_net > 0 else 0.0
                n_factors += 1

            # 投信淨買超
            trust_data = stock_inst[stock_inst["name"].str.contains("投信", na=False)]
            if not trust_data.empty:
                trust_net = trust_data["net"].sum()
                score += 1.0 if trust_net > 0 else 0.0
                n_factors += 1

            # 三大法人合計淨買超
            total_net = stock_inst["net"].sum()
            score += 1.0 if total_net > 0 else 0.0
            n_factors += 1

            chip_score = score / max(n_factors, 1)
            results.append({"stock_id": sid, "chip_score": chip_score})

        return pd.DataFrame(results)

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """從月營收資料計算基本面分數（YoY 營收成長 + MoM 加分）。"""
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        if rev.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        yoy = rev["yoy_growth"].fillna(0)
        yoy_score = np.clip(yoy / 50, 0, 1)
        mom_bonus = (rev["mom_growth"].fillna(0) > 0).astype(float) * 0.1
        rev["fundamental_score"] = np.clip(yoy_score + mom_bonus, 0, 1)

        # 包含所有 stock_ids，無資料的用 NaN（外層 fillna 處理）
        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    # ------------------------------------------------------------------ #
    #  Stage 4: 排名 + 產業標籤
    # ------------------------------------------------------------------ #

    def _rank_and_enrich(self, scored: pd.DataFrame) -> pd.DataFrame:
        """排名並加上產業 / 股票名稱。"""
        scored = scored.sort_values("composite_score", ascending=False).reset_index(drop=True)
        scored["rank"] = range(1, len(scored) + 1)

        # 從 DB 取 StockInfo
        stock_ids = scored["stock_id"].tolist()
        with get_session() as session:
            rows = session.execute(
                select(StockInfo.stock_id, StockInfo.stock_name, StockInfo.industry_category).where(
                    StockInfo.stock_id.in_(stock_ids)
                )
            ).all()
            info_df = pd.DataFrame(rows, columns=["stock_id", "stock_name", "industry_category"])

        if not info_df.empty:
            scored = scored.merge(info_df, on="stock_id", how="left")
        else:
            scored["stock_name"] = ""
            scored["industry_category"] = ""

        scored["stock_name"] = scored["stock_name"].fillna("")
        scored["industry_category"] = scored["industry_category"].fillna("")

        # 只保留需要的欄位
        keep_cols = [
            "rank",
            "stock_id",
            "stock_name",
            "close",
            "volume",
            "composite_score",
            "technical_score",
            "chip_score",
            "fundamental_score",
            "industry_category",
            "momentum",
            "inst_net",
        ]
        return scored[[c for c in keep_cols if c in scored.columns]]

    def _compute_sector_summary(self, rankings: pd.DataFrame) -> pd.DataFrame:
        """統計推薦結果的產業分布。"""
        if rankings.empty or "industry_category" not in rankings.columns:
            return pd.DataFrame()

        top_n = rankings.head(self.top_n_results)
        summary = (
            top_n.groupby("industry_category")
            .agg(
                count=("stock_id", "count"),
                avg_score=("composite_score", "mean"),
            )
            .reset_index()
            .sort_values("count", ascending=False)
        )
        summary.columns = ["industry", "count", "avg_score"]
        return summary
