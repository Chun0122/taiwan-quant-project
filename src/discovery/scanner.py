"""全市場選股掃描器 — 四階段漏斗從 ~2000 支股票篩選出 Top N 推薦。

漏斗架構：
  Stage 1: 從 DB 載入全市場日K + 法人資料
  Stage 2: 粗篩（股價/成交量/法人/動能 → 留 ~150 檔）
  Stage 3: 細評（模式專屬因子加權）
  Stage 3.5: 風險過濾（剔除高波動股）
  Stage 4: 排名 + 加上產業標籤 → 輸出 Top N

支援五種模式：
  - MomentumScanner: 短線動能（1~10 天），突破 + 資金流 + 量能擴張
  - SwingScanner: 中期波段（1~3 個月），趨勢 + 基本面 + 法人布局
  - ValueScanner: 價值修復，低估值 + 基本面轉佳
  - DividendScanner: 高息存股，高殖利率 + 配息穩定 + 估值合理
  - GrowthScanner: 高成長，營收高速成長 + 動能啟動
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import (
    Announcement,
    DailyPrice,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
    StockInfo,
    StockValuation,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """掃描結果資料容器。"""

    rankings: pd.DataFrame
    total_stocks: int
    after_coarse: int
    scan_date: date = field(default_factory=date.today)
    sector_summary: pd.DataFrame | None = None
    mode: str = "momentum"


class MarketScanner:
    """全市場選股掃描器（基底類別）。

    子類須覆寫 _coarse_filter() 和 _score_candidates() 以實作模式專屬邏輯。

    Args:
        min_price: 最低股價門檻
        max_price: 最高股價門檻
        min_volume: 最低成交量（股）
        top_n_candidates: 粗篩後保留數量
        top_n_results: 最終輸出數量
        lookback_days: 回溯天數（用於計算指標）
    """

    mode_name: str = "base"

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
        # Stage 0: 偵測市場狀態（Regime）
        try:
            from src.regime.detector import MarketRegimeDetector

            regime_info = MarketRegimeDetector().detect()
            self.regime = regime_info["regime"]
            logger.info("Stage 0: 市場狀態 = %s (TAIEX=%.0f)", self.regime, regime_info["taiex_close"])
        except Exception:
            self.regime = "sideways"
            logger.warning("Stage 0: 市場狀態偵測失敗，預設 sideways")

        # Stage 1: 載入資料
        df_price, df_inst, df_margin, df_revenue = self._load_market_data()
        if df_price.empty:
            logger.warning("無市場資料可供掃描")
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=0,
                after_coarse=0,
                mode=self.mode_name,
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
                mode=self.mode_name,
            )

        # Stage 2.5: 補抓候選股月營收（從 FinMind 逐股取得）
        candidate_ids = candidates["stock_id"].tolist()
        try:
            from src.data.pipeline import sync_revenue_for_stocks

            logger.info("Stage 2.5: 補抓 %d 支候選股月營收...", len(candidate_ids))
            rev_count = sync_revenue_for_stocks(candidate_ids)
            logger.info("Stage 2.5: 補抓完成，新增 %d 筆月營收", rev_count)
            # 重新載入營收資料（補抓後 DB 已更新）
            df_revenue = self._load_revenue_data(candidate_ids)
        except Exception:
            logger.warning("Stage 2.5: 月營收補抓失敗（可能無 FinMind token），使用既有資料")

        # Stage 2.7: 載入候選股近期 MOPS 公告
        df_ann = self._load_announcement_data(candidate_ids)
        if not df_ann.empty:
            logger.info("Stage 2.7: 載入 %d 筆 MOPS 公告", len(df_ann))
        else:
            logger.info("Stage 2.7: 無 MOPS 公告資料（消息面分數預設 0.5）")

        # Stage 3: 細評
        scored = self._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue, df_ann)
        logger.info("Stage 3: 完成 %d 支候選股評分", len(scored))

        # Stage 3.3: 產業加成
        scored = self._apply_sector_bonus(scored)

        # Stage 3.5: 風險過濾
        scored = self._apply_risk_filter(scored, df_price)

        # Stage 4: 排名 + 產業標籤
        rankings = self._rank_and_enrich(scored)
        sector_summary = self._compute_sector_summary(rankings)
        logger.info("Stage 4: 輸出 Top %d", min(self.top_n_results, len(rankings)))

        return DiscoveryResult(
            rankings=rankings.head(self.top_n_results),
            total_stocks=total_stocks,
            after_coarse=after_coarse,
            sector_summary=sector_summary,
            mode=self.mode_name,
        )

    # ------------------------------------------------------------------ #
    #  Stage 1: 載入資料
    # ------------------------------------------------------------------ #

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """從 DB 查詢最近的 daily_price + institutional + margin + monthly_revenue 資料。"""
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

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

            # 融資融券
            rows = session.execute(
                select(
                    MarginTrading.stock_id,
                    MarginTrading.date,
                    MarginTrading.margin_balance,
                    MarginTrading.short_balance,
                ).where(MarginTrading.date >= cutoff)
            ).all()
            df_margin = pd.DataFrame(
                rows,
                columns=[
                    "stock_id",
                    "date",
                    "margin_balance",
                    "short_balance",
                ],
            )

        # 月營收（全部股票）
        df_revenue = self._load_revenue_data()

        return df_price, df_inst, df_margin, df_revenue

    def _load_revenue_data(self, stock_ids: list[str] | None = None, months: int = 1) -> pd.DataFrame:
        """從 DB 查詢月營收資料。

        Args:
            stock_ids: 限定查詢的股票清單，None 表示查全部
            months: 取每支股票最近幾個月的營收（1=最新, 2=含上月）
        """
        from sqlalchemy import func

        revenue_cutoff = date.today() - timedelta(days=180)

        with get_session() as session:
            base_filter = MonthlyRevenue.date >= revenue_cutoff
            if stock_ids:
                base_filter = base_filter & MonthlyRevenue.stock_id.in_(stock_ids)

            if months <= 1:
                # 原有邏輯：每支股票取最新一筆
                subq = (
                    select(
                        MonthlyRevenue.stock_id,
                        func.max(MonthlyRevenue.date).label("max_date"),
                    )
                    .where(base_filter)
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

                return pd.DataFrame(
                    rows,
                    columns=["stock_id", "yoy_growth", "mom_growth"],
                )
            else:
                # months >= 2：取每支股票最近 N 筆，並加上 prev_yoy_growth / prev_mom_growth
                rows = session.execute(
                    select(
                        MonthlyRevenue.stock_id,
                        MonthlyRevenue.date,
                        MonthlyRevenue.yoy_growth,
                        MonthlyRevenue.mom_growth,
                    )
                    .where(base_filter)
                    .order_by(MonthlyRevenue.stock_id, MonthlyRevenue.date.desc())
                ).all()

        df_all = pd.DataFrame(
            rows,
            columns=["stock_id", "date", "yoy_growth", "mom_growth"],
        )
        if df_all.empty:
            return pd.DataFrame(columns=["stock_id", "yoy_growth", "mom_growth", "prev_yoy_growth", "prev_mom_growth"])

        # 每支股票取最近 months 筆
        result_rows = []
        for sid, grp in df_all.groupby("stock_id"):
            grp = grp.sort_values("date", ascending=False).head(months)
            if len(grp) >= 2:
                latest = grp.iloc[0]
                prev = grp.iloc[1]
                result_rows.append(
                    {
                        "stock_id": sid,
                        "yoy_growth": latest["yoy_growth"],
                        "mom_growth": latest["mom_growth"],
                        "prev_yoy_growth": prev["yoy_growth"],
                        "prev_mom_growth": prev["mom_growth"],
                    }
                )
            elif len(grp) == 1:
                latest = grp.iloc[0]
                result_rows.append(
                    {
                        "stock_id": sid,
                        "yoy_growth": latest["yoy_growth"],
                        "mom_growth": latest["mom_growth"],
                        "prev_yoy_growth": None,
                        "prev_mom_growth": None,
                    }
                )

        return pd.DataFrame(result_rows)

    def _load_announcement_data(self, stock_ids: list[str] | None = None, days: int = 10) -> pd.DataFrame:
        """從 DB 查詢近 N 日的 MOPS 重大訊息公告。

        Args:
            stock_ids: 限定查詢的股票清單，None 表示查全部
            days: 回溯天數

        Returns:
            DataFrame(stock_id, date, seq, subject, sentiment)
        """
        cutoff = date.today() - timedelta(days=days)

        with get_session() as session:
            query = select(
                Announcement.stock_id,
                Announcement.date,
                Announcement.seq,
                Announcement.subject,
                Announcement.sentiment,
            ).where(Announcement.date >= cutoff)

            if stock_ids:
                query = query.where(Announcement.stock_id.in_(stock_ids))

            rows = session.execute(query).all()

        return pd.DataFrame(
            rows,
            columns=["stock_id", "date", "seq", "subject", "sentiment"],
        )

    def _compute_news_scores(self, stock_ids: list[str], df_ann: pd.DataFrame) -> pd.DataFrame:
        """計算消息面分數（3 因子：公告密度 + 正面訊號 + 負面懲罰）。

        Args:
            stock_ids: 候選股代號清單
            df_ann: 公告資料 DataFrame

        Returns:
            DataFrame(stock_id, news_score) — 分數 0~1，0.5 為中性預設
        """
        default = pd.DataFrame({"stock_id": stock_ids, "news_score": [0.5] * len(stock_ids)})

        if df_ann.empty:
            return default

        ann = df_ann[df_ann["stock_id"].isin(stock_ids)]
        if ann.empty:
            return default

        df = pd.DataFrame({"stock_id": stock_ids})

        # 1) 公告密度：近期重訊數量排名（30%）
        ann_count = ann.groupby("stock_id").size().reset_index(name="ann_count")
        df = df.merge(ann_count, on="stock_id", how="left")
        df["ann_count"] = df["ann_count"].fillna(0)

        # 2) 正面訊號：sentiment=+1 的公告數量排名（40%）
        positive = ann[ann["sentiment"] == 1].groupby("stock_id").size().reset_index(name="pos_count")
        df = df.merge(positive, on="stock_id", how="left")
        df["pos_count"] = df["pos_count"].fillna(0)

        # 3) 負面懲罰：sentiment=-1 的公告數量（倒數排名，越多越低分）（30%）
        negative = ann[ann["sentiment"] == -1].groupby("stock_id").size().reset_index(name="neg_count")
        df = df.merge(negative, on="stock_id", how="left")
        df["neg_count"] = df["neg_count"].fillna(0)

        # 排名百分位（有公告的才有區分度，無公告者 = 0.5）
        if df["ann_count"].sum() == 0:
            return default

        density_rank = df["ann_count"].rank(pct=True)
        pos_rank = df["pos_count"].rank(pct=True)
        # 負面用倒數排名：neg_count 越大 → 排名越低
        neg_rank = (-df["neg_count"]).rank(pct=True)

        df["news_score"] = density_rank * 0.30 + pos_rank * 0.40 + neg_rank * 0.30

        return df[["stock_id", "news_score"]]

    # ------------------------------------------------------------------ #
    #  產業加成
    # ------------------------------------------------------------------ #

    def _compute_sector_bonus(self, stock_ids: list[str]) -> pd.DataFrame:
        """計算候選股的產業熱度加成分數（±5%）。

        呼叫 IndustryRotationAnalyzer 計算產業排名，將排名百分位
        映射為 sector_bonus（-0.05 ~ +0.05）。失敗時回傳全 0。

        Returns:
            DataFrame(stock_id, sector_bonus)
        """
        default = pd.DataFrame({"stock_id": stock_ids, "sector_bonus": [0.0] * len(stock_ids)})
        try:
            from src.industry.analyzer import IndustryRotationAnalyzer

            analyzer = IndustryRotationAnalyzer(watchlist=stock_ids)
            result = analyzer.compute_sector_scores_for_stocks(stock_ids)
            if result.empty:
                return default
            return result
        except Exception:
            logger.warning("產業加成計算失敗，跳過")
            return default

    def _apply_sector_bonus(self, scored: pd.DataFrame) -> pd.DataFrame:
        """將產業加成套用到 composite_score。

        final_score = composite_score × (1 + sector_bonus)
        """
        if scored.empty:
            return scored

        stock_ids = scored["stock_id"].tolist()
        bonus_df = self._compute_sector_bonus(stock_ids)
        scored = scored.merge(bonus_df, on="stock_id", how="left")
        scored["sector_bonus"] = scored["sector_bonus"].fillna(0.0)
        scored["composite_score"] = scored["composite_score"] * (1 + scored["sector_bonus"])
        logger.info(
            "Stage 3.3: 產業加成已套用（範圍 %.3f ~ %.3f）", scored["sector_bonus"].min(), scored["sector_bonus"].max()
        )
        return scored

    # ------------------------------------------------------------------ #
    #  Stage 2: 粗篩
    # ------------------------------------------------------------------ #

    def _base_filter(self, df_price: pd.DataFrame) -> pd.DataFrame:
        """基礎過濾：股價範圍 + 成交量 + 排除指數/ETF。供子類 _coarse_filter 呼叫。"""
        latest_date = df_price["date"].max()
        latest = df_price[df_price["date"] == latest_date].copy()

        if latest.empty:
            return pd.DataFrame()

        mask = (
            (latest["close"] >= self.min_price)
            & (latest["close"] <= self.max_price)
            & (latest["volume"] >= self.min_volume)
        )
        # 排除指數類（如 TAIEX）
        mask = mask & (~latest["stock_id"].str.contains(r"[A-Za-z]", na=False))
        # 排除 ETF（代號 00 開頭，如 0050、00878、009xx）
        mask = mask & (~latest["stock_id"].str.startswith("00"))
        return latest[mask].copy()

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """粗篩：股價/量/法人/動能加權 → 取 top N candidates。"""
        filtered = self._base_filter(df_price)
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
        df_margin: pd.DataFrame,
        df_revenue: pd.DataFrame,
        df_ann: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """對候選股進行多維度評分（通用流程）。

        子類透過覆寫 _compute_*  方法客製化各維度計算，
        透過 _compute_extra_scores() 新增額外維度，
        透過 _post_score() 做加權後處理。
        """
        stock_ids = candidates["stock_id"].tolist()
        ann = df_ann if df_ann is not None else pd.DataFrame()

        # 各維度分數（子類覆寫 _compute_* 即可客製化）
        score_dfs = [
            self._compute_technical_scores(stock_ids, df_price),
            self._compute_chip_scores(stock_ids, df_inst, df_price, df_margin),
            self._compute_fundamental_scores(stock_ids, df_revenue),
            self._compute_news_scores(stock_ids, ann),
        ]
        # hook：子類可加額外維度（如 ValueScanner 的 valuation_score）
        score_dfs.extend(self._compute_extra_scores(stock_ids))

        candidates = candidates.copy()
        for df in score_dfs:
            candidates = candidates.merge(df, on="stock_id", how="left")

        # 所有 *_score 欄位 fillna(0.5)
        score_cols = [c for c in candidates.columns if c.endswith("_score") and c != "composite_score"]
        for col in score_cols:
            candidates[col] = candidates[col].fillna(0.5)

        # 根據 regime 動態加權（weight key 直接映射 {key}_score 欄位）
        from src.regime.detector import MarketRegimeDetector

        regime = getattr(self, "regime", "sideways")
        w = MarketRegimeDetector.get_weights(self.mode_name, regime)

        composite = pd.Series(0.0, index=candidates.index)
        for key, weight in w.items():
            col = f"{key}_score"
            if col in candidates.columns:
                composite += candidates[col] * weight
        candidates["composite_score"] = composite

        # hook：子類可在加權後做額外處理
        candidates = self._post_score(candidates)

        return candidates

    def _compute_extra_scores(self, stock_ids: list[str]) -> list[pd.DataFrame]:
        """hook：子類可覆寫以新增額外評分維度。回傳 DataFrame 的 list。"""
        return []

    def _post_score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """hook：子類可在加權後做額外處理。"""
        return candidates

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """從原始 OHLCV 計算技術面分數（6 因子：SMA + 動能 + 價格位置 + 量能比 + 波動收斂 + 量價背離）。"""
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

            # 5) 波動度收斂（BB 寬度縮窄）：CV 越低代表盤整越緊，突破潛力越大
            if len(closes) >= 5:
                recent_closes = closes[-5:]
                mean_price = recent_closes.mean()
                if mean_price > 0:
                    cv = recent_closes.std(ddof=0) / mean_price
                    # CV 越小分數越高：用 1 - 歸一化 CV（CV 通常 < 0.1，用 0.1 做上限）
                    score += max(0.0, 1.0 - min(cv / 0.1, 1.0))
                else:
                    score += 0.5
                n_factors += 1

            # 6) 量價背離偵測：價格方向 vs 成交量方向一致性
            if len(closes) >= 3 and len(volumes) >= 3:
                price_chg = closes[-1] - closes[-3]
                vol_chg = float(volumes[-1]) - float(volumes[-3])
                # 價漲量增 → 健康上漲（高分），價漲量縮 → 看空背離（低分）
                # 價跌量增 → 看空（低分），價跌量縮 → 中性
                if price_chg > 0 and vol_chg > 0:
                    score += 1.0  # 價漲量增：最佳
                elif price_chg > 0 and vol_chg <= 0:
                    score += 0.3  # 價漲量縮：背離
                elif price_chg <= 0 and vol_chg <= 0:
                    score += 0.5  # 價跌量縮：中性
                else:
                    score += 0.2  # 價跌量增：最差
                n_factors += 1

            tech_score = score / max(n_factors, 1)
            results.append({"stock_id": sid, "technical_score": tech_score})

        return pd.DataFrame(results)

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """計算籌碼面分數（5 因子：淨買超 × 3 + 連續買超天數 + 買超佔量比）。"""
        if df_inst.empty:
            return pd.DataFrame(
                {
                    "stock_id": stock_ids,
                    "chip_score": [0.5] * len(stock_ids),
                }
            )

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]

        # 計算每支股票的外資、投信、合計淨買超 + 連續買超天數 + 買超佔量比
        rows = []
        for sid in stock_ids:
            stock_inst = inst_filtered[inst_filtered["stock_id"] == sid]
            if stock_inst.empty:
                rows.append(
                    {
                        "stock_id": sid,
                        "foreign_net": 0,
                        "trust_net": 0,
                        "total_net": 0,
                        "consec_buy_days": 0,
                        "buy_vol_ratio": 0.0,
                    }
                )
                continue

            foreign_data = stock_inst[stock_inst["name"].str.contains("外資", na=False)]
            trust_data = stock_inst[stock_inst["name"].str.contains("投信", na=False)]
            total_net = stock_inst["net"].sum()

            # 連續買超天數：從最新日期往回數，三大法人合計淨買超 > 0 的連續天數
            daily_net = stock_inst.groupby("date")["net"].sum().sort_index(ascending=False)
            consec_days = 0
            for net_val in daily_net.values:
                if net_val > 0:
                    consec_days += 1
                else:
                    break

            # 買超佔成交量比例：合計淨買超 / 最新日成交量
            buy_vol_ratio = 0.0
            if df_price is not None and not df_price.empty:
                stock_price = df_price[df_price["stock_id"] == sid]
                if not stock_price.empty:
                    latest_vol = stock_price.loc[stock_price["date"].idxmax(), "volume"]
                    if latest_vol > 0:
                        buy_vol_ratio = total_net / latest_vol

            rows.append(
                {
                    "stock_id": sid,
                    "foreign_net": foreign_data["net"].sum() if not foreign_data.empty else 0,
                    "trust_net": trust_data["net"].sum() if not trust_data.empty else 0,
                    "total_net": total_net,
                    "consec_buy_days": consec_days,
                    "buy_vol_ratio": buy_vol_ratio,
                }
            )

        df = pd.DataFrame(rows)

        # 用排名百分位，分數自然分散在 0~1
        foreign_rank = df["foreign_net"].rank(pct=True)
        trust_rank = df["trust_net"].rank(pct=True)
        total_rank = df["total_net"].rank(pct=True)
        consec_rank = df["consec_buy_days"].rank(pct=True)
        buy_vol_rank = df["buy_vol_ratio"].rank(pct=True)

        # 外資 30% + 投信 20% + 合計 20% + 連續買超 15% + 買超佔量 15%
        df["chip_score"] = (
            foreign_rank * 0.30 + trust_rank * 0.20 + total_rank * 0.20 + consec_rank * 0.15 + buy_vol_rank * 0.15
        )

        return df[["stock_id", "chip_score"]]

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """從月營收資料計算基本面分數（YoY 70% + MoM 30%，排名百分位）。"""
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        if rev.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        # 用排名百分位，讓分數自然分散在 0~1
        yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
        mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)

        # YoY 權重 70% + MoM 權重 30%
        rev["fundamental_score"] = yoy_rank * 0.70 + mom_rank * 0.30

        # 包含所有 stock_ids，無資料的用 NaN（外層 fillna 處理）
        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    # ------------------------------------------------------------------ #
    #  Stage 3.5: 風險過濾（子類覆寫）
    # ------------------------------------------------------------------ #

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """風險過濾（基底類別不做任何過濾，子類覆寫）。"""
        return scored

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
            "news_score",
            "sector_bonus",
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


# ====================================================================== #
#  MomentumScanner — 短線動能模式
# ====================================================================== #


class MomentumScanner(MarketScanner):
    """短線動能掃描器（1~10 天）。

    粗篩：動能 + 流動性
    細評：技術面 45% + 籌碼面 45% + 基本面 10%
    風險過濾：ATR ratio > 80th percentile 剔除
    """

    mode_name = "momentum"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 25)
        super().__init__(**kwargs)

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """動能模式粗篩：基本過濾 + 流動性 + 動能/法人/成交量加權。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 額外流動性過濾：成交量 > 20 日均量 × 0.5
        latest_date = df_price["date"].max()
        vol_mean = df_price.groupby("stock_id")["volume"].apply(lambda s: s.tail(20).mean()).reset_index()
        vol_mean.columns = ["stock_id", "avg_vol_20"]
        filtered = filtered.merge(vol_mean, on="stock_id", how="left")
        filtered["avg_vol_20"] = filtered["avg_vol_20"].fillna(0)
        filtered = filtered[filtered["volume"] > filtered["avg_vol_20"] * 0.5].copy()

        if filtered.empty:
            return pd.DataFrame()

        # 1) 成交量排名
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

        # 3) 短期動能
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

        filtered = filtered.nlargest(self.top_n_candidates, "coarse_score")
        return filtered

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """動能模式技術面 5 因子：5日動能 + 10日動能 + 20日突破 + 量比 + 成交量加速。"""
        results = []

        for sid in stock_ids:
            stock_data = df_price[df_price["stock_id"] == sid].sort_values("date")
            if len(stock_data) < 3:
                results.append({"stock_id": sid, "technical_score": 0.5})
                continue

            closes = stock_data["close"].values
            volumes = stock_data["volume"].values.astype(float)

            score = 0.0
            n_factors = 0

            # 1) 5 日動能
            if len(closes) >= 6:
                ret_5d = (closes[-1] - closes[-6]) / closes[-6]
                score += max(0.0, min(1.0, 0.5 + ret_5d * 5))
                n_factors += 1

            # 2) 10 日動能
            if len(closes) >= 11:
                ret_10d = (closes[-1] - closes[-11]) / closes[-11]
                score += max(0.0, min(1.0, 0.5 + ret_10d * 5))
                n_factors += 1

            # 3) 20 日突破：close / max(close[-20:])
            if len(closes) >= 20:
                max_20 = closes[-20:].max()
                if max_20 > 0:
                    score += closes[-1] / max_20
                else:
                    score += 0.5
                n_factors += 1

            # 4) 量比：volume[-1] / mean(volume[-20:])
            if len(volumes) >= 20:
                avg_vol_20 = volumes[-20:].mean()
                if avg_vol_20 > 0:
                    ratio = min(2.0, volumes[-1] / avg_vol_20)
                    score += ratio / 2.0
                else:
                    score += 0.5
                n_factors += 1

            # 5) 成交量加速：mean(vol[-3:]) / mean(vol[-10:])
            if len(volumes) >= 10:
                avg_vol_3 = volumes[-3:].mean()
                avg_vol_10 = volumes[-10:].mean()
                if avg_vol_10 > 0:
                    ratio = min(2.0, avg_vol_3 / avg_vol_10)
                    score += ratio / 2.0
                else:
                    score += 0.5
                n_factors += 1

            tech_score = score / max(n_factors, 1)
            results.append({"stock_id": sid, "technical_score": tech_score})

        return pd.DataFrame(results)

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """動能模式籌碼面：外資連續買超 + 買超佔量比 + 三大法人合計 + 券資比（有資料時）。"""
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids)})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
        rows = []
        for sid in stock_ids:
            stock_inst = inst_filtered[inst_filtered["stock_id"] == sid]
            if stock_inst.empty:
                rows.append({"stock_id": sid, "consec_foreign_days": 0, "buy_vol_ratio": 0.0, "total_net": 0})
                continue

            # 外資連續買超天數
            foreign_data = stock_inst[stock_inst["name"].str.contains("外資", na=False)]
            consec_foreign = 0
            if not foreign_data.empty:
                daily_foreign = foreign_data.groupby("date")["net"].sum().sort_index(ascending=False)
                for val in daily_foreign.values:
                    if val > 0:
                        consec_foreign += 1
                    else:
                        break

            # 法人買超/成交量比例
            total_net = stock_inst["net"].sum()
            buy_vol_ratio = 0.0
            if df_price is not None and not df_price.empty:
                stock_price = df_price[df_price["stock_id"] == sid]
                if not stock_price.empty:
                    latest_vol = stock_price.loc[stock_price["date"].idxmax(), "volume"]
                    if latest_vol > 0:
                        buy_vol_ratio = total_net / latest_vol

            rows.append(
                {
                    "stock_id": sid,
                    "consec_foreign_days": consec_foreign,
                    "buy_vol_ratio": buy_vol_ratio,
                    "total_net": total_net,
                }
            )

        df = pd.DataFrame(rows)

        consec_rank = df["consec_foreign_days"].rank(pct=True)
        bvr_rank = df["buy_vol_ratio"].rank(pct=True)
        total_rank = df["total_net"].rank(pct=True)

        # 有融資融券資料時加入券資比因子（4 因子加權），否則 3 因子
        has_margin = df_margin is not None and not df_margin.empty
        if has_margin:
            # 券資比 = short_balance / margin_balance（越高代表看空情緒越重）
            margin_latest = df_margin[df_margin["date"] == df_margin["date"].max()]
            margin_data = margin_latest[margin_latest["stock_id"].isin(stock_ids)][
                ["stock_id", "margin_balance", "short_balance"]
            ].copy()
            if not margin_data.empty:
                margin_data["short_margin_ratio"] = margin_data.apply(
                    lambda r: r["short_balance"] / r["margin_balance"] if r["margin_balance"] > 0 else 0.0,
                    axis=1,
                )
                df = df.merge(margin_data[["stock_id", "short_margin_ratio"]], on="stock_id", how="left")
                df["short_margin_ratio"] = df["short_margin_ratio"].fillna(0.0)
                smr_rank = df["short_margin_ratio"].rank(pct=True)
                # 外資連續 30% + 買超佔量比 25% + 三大法人合計 25% + 券資比 20%
                df["chip_score"] = consec_rank * 0.30 + bvr_rank * 0.25 + total_rank * 0.25 + smr_rank * 0.20
            else:
                has_margin = False

        if not has_margin:
            # 外資連續買超 40% + 買超佔量比 30% + 三大法人合計 30%
            df["chip_score"] = consec_rank * 0.40 + bvr_rank * 0.30 + total_rank * 0.30

        return df[["stock_id", "chip_score"]]

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """動能模式基本面：僅做過濾 — YoY > 0 加分(0.7)，≤ 0 中性(0.3)，無資料 fallback 0.5。"""
        result_rows = []
        for sid in stock_ids:
            if df_revenue.empty:
                result_rows.append({"stock_id": sid, "fundamental_score": 0.5})
                continue
            rev = df_revenue[df_revenue["stock_id"] == sid]
            if rev.empty:
                result_rows.append({"stock_id": sid, "fundamental_score": 0.5})
            else:
                yoy = rev.iloc[0].get("yoy_growth", None)
                if yoy is not None and not pd.isna(yoy):
                    result_rows.append({"stock_id": sid, "fundamental_score": 0.7 if yoy > 0 else 0.3})
                else:
                    result_rows.append({"stock_id": sid, "fundamental_score": 0.5})

        return pd.DataFrame(result_rows)

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """動能模式風險過濾：ATR(14)/close > 80th percentile 剔除。"""
        if scored.empty or df_price.empty:
            return scored

        atr_ratios = []
        for sid in scored["stock_id"].tolist():
            stock_data = df_price[df_price["stock_id"] == sid].sort_values("date")
            if len(stock_data) < 14:
                atr_ratios.append({"stock_id": sid, "atr_ratio": 0.0})
                continue

            highs = stock_data["high"].values[-14:]
            lows = stock_data["low"].values[-14:]
            closes = stock_data["close"].values[-15:]  # 需要前一天 close

            trs = []
            for i in range(1, len(highs) + 1):
                if i < len(closes):
                    tr = max(
                        highs[i - 1] - lows[i - 1],
                        abs(highs[i - 1] - closes[i - 1]),
                        abs(lows[i - 1] - closes[i - 1]),
                    )
                else:
                    tr = highs[i - 1] - lows[i - 1]
                trs.append(tr)

            atr = np.mean(trs)
            current_close = stock_data["close"].values[-1]
            ratio = atr / current_close if current_close > 0 else 0.0
            atr_ratios.append({"stock_id": sid, "atr_ratio": ratio})

        df_atr = pd.DataFrame(atr_ratios)
        threshold = df_atr["atr_ratio"].quantile(0.80)
        high_vol_ids = df_atr[df_atr["atr_ratio"] > threshold]["stock_id"].tolist()

        before_count = len(scored)
        scored = scored[~scored["stock_id"].isin(high_vol_ids)].copy()
        removed = before_count - len(scored)
        if removed > 0:
            logger.info("Stage 3.5: ATR 風險過濾剔除 %d 支高波動股", removed)

        return scored


# ====================================================================== #
#  SwingScanner — 中期波段模式
# ====================================================================== #


class SwingScanner(MarketScanner):
    """中期波段掃描器（1~3 個月）。

    粗篩：趨勢（close > SMA60）+ 基本面
    細評：技術面 30% + 籌碼面 30% + 基本面 40%
    風險過濾：年化波動率 > 85th percentile 剔除
    """

    mode_name = "swing"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 80)
        super().__init__(**kwargs)

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：swing 模式載入 2 個月營收資料（算加速度）。"""
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        with get_session() as session:
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
                columns=["stock_id", "date", "open", "high", "low", "close", "volume"],
            )

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
                columns=["stock_id", "date", "name", "net"],
            )

            # 融資融券
            rows = session.execute(
                select(
                    MarginTrading.stock_id,
                    MarginTrading.date,
                    MarginTrading.margin_balance,
                    MarginTrading.short_balance,
                ).where(MarginTrading.date >= cutoff)
            ).all()
            df_margin = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "margin_balance", "short_balance"],
            )

        # 載入 2 個月營收（含上月，用於計算加速度）
        df_revenue = self._load_revenue_data(months=2)

        return df_price, df_inst, df_margin, df_revenue

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """波段模式粗篩：基本過濾 + close > SMA60 + 法人累積/趨勢/量加權。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 額外條件：close > SMA60
        sma60_data = {}
        for sid in filtered["stock_id"].unique():
            stock_data = df_price[df_price["stock_id"] == sid].sort_values("date")
            if len(stock_data) >= 60:
                sma60_data[sid] = stock_data["close"].values[-60:].mean()

        if sma60_data:
            filtered["sma60"] = filtered["stock_id"].map(sma60_data)
            filtered = filtered[filtered["sma60"].notna() & (filtered["close"] > filtered["sma60"])].copy()
        else:
            # 資料不足 60 天時跳過 SMA60 過濾
            pass

        if filtered.empty:
            return pd.DataFrame()

        # 1) 法人 20 日累積買超排名
        if not df_inst.empty:
            dates = sorted(df_inst["date"].unique())
            recent_20_dates = dates[-20:] if len(dates) >= 20 else dates
            inst_recent = df_inst[df_inst["date"].isin(recent_20_dates)]
            inst_cum = inst_recent.groupby("stock_id")["net"].sum().reset_index()
            inst_cum.columns = ["stock_id", "inst_net"]
            filtered = filtered.merge(inst_cum, on="stock_id", how="left")
            filtered["inst_net"] = filtered["inst_net"].fillna(0)
            filtered["inst_rank"] = filtered["inst_net"].rank(pct=True)
        else:
            filtered["inst_net"] = 0
            filtered["inst_rank"] = 0.5

        # 2) 趨勢強度：close / SMA60 的比值
        if "sma60" in filtered.columns:
            filtered["trend_strength"] = filtered["close"] / filtered["sma60"]
            filtered["trend_rank"] = filtered["trend_strength"].rank(pct=True)
        else:
            filtered["trend_rank"] = 0.5

        # 3) 成交量排名
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 動能欄位（用於 _rank_and_enrich 保留）
        dates_all = sorted(df_price["date"].unique())
        if len(dates_all) >= 2:
            prev_date = dates_all[-2]
            prev = df_price[df_price["date"] == prev_date][["stock_id", "close"]].copy()
            prev.columns = ["stock_id", "prev_close"]
            filtered = filtered.merge(prev, on="stock_id", how="left")
            filtered["momentum"] = ((filtered["close"] - filtered["prev_close"]) / filtered["prev_close"]).fillna(0)
        else:
            filtered["momentum"] = 0

        # 粗篩綜合分 = 法人累積 40% + 趨勢強度 30% + 成交量 30%
        filtered["coarse_score"] = (
            filtered["inst_rank"] * 0.40 + filtered["trend_rank"] * 0.30 + filtered["vol_rank"] * 0.30
        )

        filtered = filtered.nlargest(self.top_n_candidates, "coarse_score")
        return filtered

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """波段模式技術面 4 因子：趨勢確認 + 均線排列 + 60日動能 + 量價齊揚。"""
        results = []

        for sid in stock_ids:
            stock_data = df_price[df_price["stock_id"] == sid].sort_values("date")
            if len(stock_data) < 3:
                results.append({"stock_id": sid, "technical_score": 0.5})
                continue

            closes = stock_data["close"].values
            volumes = stock_data["volume"].values.astype(float)

            score = 0.0
            n_factors = 0

            # 1) 趨勢確認：close > SMA60
            if len(closes) >= 60:
                sma60 = closes[-60:].mean()
                score += 1.0 if closes[-1] > sma60 else 0.0
                n_factors += 1

            # 2) 均線排列：SMA20 > SMA60
            if len(closes) >= 60:
                sma20 = closes[-20:].mean()
                sma60 = closes[-60:].mean()
                score += 1.0 if sma20 > sma60 else 0.0
                n_factors += 1

            # 3) 60 日動能
            if len(closes) >= 61:
                ret_60d = (closes[-1] - closes[-61]) / closes[-61]
                score += max(0.0, min(1.0, 0.5 + ret_60d * 2))
                n_factors += 1

            # 4) 量價齊揚：近 20 日 price_chg > 0 且 vol_chg > 0 的天數比例
            if len(closes) >= 21 and len(volumes) >= 21:
                recent_closes = closes[-21:]
                recent_volumes = volumes[-21:]
                count = 0
                for i in range(1, 21):
                    price_up = recent_closes[i] > recent_closes[i - 1]
                    vol_up = recent_volumes[i] > recent_volumes[i - 1]
                    if price_up and vol_up:
                        count += 1
                score += count / 20.0
                n_factors += 1

            tech_score = score / max(n_factors, 1)
            results.append({"stock_id": sid, "technical_score": tech_score})

        return pd.DataFrame(results)

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """波段模式籌碼面 2 因子：投信淨買超 50% + 三大法人 20 日累積買超 50%。"""
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids)})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]

        # 20 日期間
        dates = sorted(df_inst["date"].unique())
        recent_20_dates = dates[-20:] if len(dates) >= 20 else dates

        rows = []
        for sid in stock_ids:
            stock_inst = inst_filtered[inst_filtered["stock_id"] == sid]
            if stock_inst.empty:
                rows.append({"stock_id": sid, "trust_net": 0, "cum_20_net": 0})
                continue

            # 投信淨買超（全期間合計）
            trust_data = stock_inst[stock_inst["name"].str.contains("投信", na=False)]
            trust_net = trust_data["net"].sum() if not trust_data.empty else 0

            # 三大法人 20 日累積買超
            recent_inst = stock_inst[stock_inst["date"].isin(recent_20_dates)]
            cum_20_net = recent_inst["net"].sum()

            rows.append({"stock_id": sid, "trust_net": trust_net, "cum_20_net": cum_20_net})

        df = pd.DataFrame(rows)

        trust_rank = df["trust_net"].rank(pct=True)
        cum_rank = df["cum_20_net"].rank(pct=True)

        # 投信 50% + 累積買超 50%
        df["chip_score"] = trust_rank * 0.50 + cum_rank * 0.50

        return df[["stock_id", "chip_score"]]

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """波段模式基本面 3 因子：YoY 40% + MoM 30% + 營收加速度 30%（排名百分位）。"""
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        if rev.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        # YoY 排名
        yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
        # MoM 排名
        mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)

        # 營收加速度 = current_yoy - prev_yoy
        if "prev_yoy_growth" in rev.columns:
            rev["acceleration"] = rev["yoy_growth"].fillna(0) - rev["prev_yoy_growth"].fillna(0)
            accel_rank = rev["acceleration"].rank(pct=True)
        else:
            accel_rank = pd.Series(0.5, index=rev.index)

        # YoY 40% + MoM 30% + 加速度 30%
        rev["fundamental_score"] = yoy_rank * 0.40 + mom_rank * 0.30 + accel_rank * 0.30

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """波段模式風險過濾：近 60 日年化波動率 > 85th percentile 剔除。"""
        if scored.empty or df_price.empty:
            return scored

        vol_data = []
        for sid in scored["stock_id"].tolist():
            stock_data = df_price[df_price["stock_id"] == sid].sort_values("date")
            if len(stock_data) < 20:
                vol_data.append({"stock_id": sid, "annual_vol": 0.0})
                continue

            closes = stock_data["close"].values[-61:] if len(stock_data) >= 61 else stock_data["close"].values
            returns = np.diff(closes) / closes[:-1]
            annual_vol = np.std(returns, ddof=1) * np.sqrt(252) if len(returns) > 1 else 0.0
            vol_data.append({"stock_id": sid, "annual_vol": annual_vol})

        df_vol = pd.DataFrame(vol_data)
        threshold = df_vol["annual_vol"].quantile(0.85)
        high_vol_ids = df_vol[df_vol["annual_vol"] > threshold]["stock_id"].tolist()

        before_count = len(scored)
        scored = scored[~scored["stock_id"].isin(high_vol_ids)].copy()
        removed = before_count - len(scored)
        if removed > 0:
            logger.info("Stage 3.5: 波動率風險過濾剔除 %d 支高波動股", removed)

        return scored


# ====================================================================== #
#  ValueScanner — 價值修復模式
# ====================================================================== #


class ValueScanner(MarketScanner):
    """價值修復掃描器。

    適合低估值 + 基本面轉佳 + 法人開始布局的「價值修復股」。
    粗篩：PE > 0 且 PE < 30 + 殖利率 > 2%
    細評：基本面 50% + 估值面 30% + 籌碼面 20%
    風險過濾：近 20 日波動率 > 90th percentile 剔除
    """

    mode_name = "value"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 25)
        super().__init__(**kwargs)

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：value 模式額外載入估值資料 + 2 個月營收。"""
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        with get_session() as session:
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
                columns=["stock_id", "date", "open", "high", "low", "close", "volume"],
            )

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
                columns=["stock_id", "date", "name", "net"],
            )

            # 融資融券
            rows = session.execute(
                select(
                    MarginTrading.stock_id,
                    MarginTrading.date,
                    MarginTrading.margin_balance,
                    MarginTrading.short_balance,
                ).where(MarginTrading.date >= cutoff)
            ).all()
            df_margin = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "margin_balance", "short_balance"],
            )

            # 估值資料
            rows = session.execute(
                select(
                    StockValuation.stock_id,
                    StockValuation.date,
                    StockValuation.pe_ratio,
                    StockValuation.pb_ratio,
                    StockValuation.dividend_yield,
                ).where(StockValuation.date >= cutoff)
            ).all()
            self._df_valuation = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "pe_ratio", "pb_ratio", "dividend_yield"],
            )

        # 載入 2 個月營收（含上月，算加速度）
        df_revenue = self._load_revenue_data(months=2)

        return df_price, df_inst, df_margin, df_revenue

    def run(self) -> DiscoveryResult:
        """覆寫 run()：在 Stage 2.5 補抓估值資料。"""
        # Stage 0: Regime 偵測
        try:
            from src.regime.detector import MarketRegimeDetector

            regime_info = MarketRegimeDetector().detect()
            self.regime = regime_info["regime"]
            logger.info("Stage 0: 市場狀態 = %s (TAIEX=%.0f)", self.regime, regime_info["taiex_close"])
        except Exception:
            self.regime = "sideways"
            logger.warning("Stage 0: 市場狀態偵測失敗，預設 sideways")

        # Stage 1
        df_price, df_inst, df_margin, df_revenue = self._load_market_data()
        if df_price.empty:
            logger.warning("無市場資料可供掃描")
            return DiscoveryResult(rankings=pd.DataFrame(), total_stocks=0, after_coarse=0, mode=self.mode_name)

        total_stocks = df_price["stock_id"].nunique()
        logger.info("Stage 1: 載入 %d 支股票的市場資料", total_stocks)

        # Stage 2: 粗篩
        candidates = self._coarse_filter(df_price, df_inst)
        after_coarse = len(candidates)
        logger.info("Stage 2: 粗篩後剩 %d 支候選股", after_coarse)

        if candidates.empty:
            return DiscoveryResult(
                rankings=pd.DataFrame(), total_stocks=total_stocks, after_coarse=0, mode=self.mode_name
            )

        # Stage 2.5: 補抓月營收 + 估值資料
        candidate_ids = candidates["stock_id"].tolist()
        try:
            from src.data.pipeline import sync_revenue_for_stocks, sync_valuation_for_stocks

            logger.info("Stage 2.5: 補抓 %d 支候選股月營收 + 估值...", len(candidate_ids))
            rev_count = sync_revenue_for_stocks(candidate_ids)
            val_count = sync_valuation_for_stocks(candidate_ids)
            logger.info("Stage 2.5: 補抓完成，新增 %d 筆月營收, %d 筆估值", rev_count, val_count)
            df_revenue = self._load_revenue_data(candidate_ids, months=2)
            # 重新載入估值
            self._reload_valuation(candidate_ids)
        except Exception:
            logger.warning("Stage 2.5: 資料補抓失敗（可能無 FinMind token），使用既有資料")

        # Stage 2.7: 載入候選股近期 MOPS 公告
        df_ann = self._load_announcement_data(candidate_ids)
        if not df_ann.empty:
            logger.info("Stage 2.7: 載入 %d 筆 MOPS 公告", len(df_ann))
        else:
            logger.info("Stage 2.7: 無 MOPS 公告資料（消息面分數預設 0.5）")

        # Stage 3: 細評
        scored = self._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue, df_ann)
        logger.info("Stage 3: 完成 %d 支候選股評分", len(scored))

        # Stage 3.3: 產業加成
        scored = self._apply_sector_bonus(scored)

        # Stage 3.5: 風險過濾
        scored = self._apply_risk_filter(scored, df_price)

        # Stage 4
        rankings = self._rank_and_enrich(scored)
        sector_summary = self._compute_sector_summary(rankings)
        logger.info("Stage 4: 輸出 Top %d", min(self.top_n_results, len(rankings)))

        return DiscoveryResult(
            rankings=rankings.head(self.top_n_results),
            total_stocks=total_stocks,
            after_coarse=after_coarse,
            sector_summary=sector_summary,
            mode=self.mode_name,
        )

    def _reload_valuation(self, stock_ids: list[str]) -> None:
        """重新載入估值資料（補抓後 DB 已更新）。"""
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)
        with get_session() as session:
            rows = session.execute(
                select(
                    StockValuation.stock_id,
                    StockValuation.date,
                    StockValuation.pe_ratio,
                    StockValuation.pb_ratio,
                    StockValuation.dividend_yield,
                )
                .where(StockValuation.date >= cutoff)
                .where(StockValuation.stock_id.in_(stock_ids))
            ).all()
            self._df_valuation = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "pe_ratio", "pb_ratio", "dividend_yield"],
            )

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """價值模式粗篩：基本過濾 + PE/殖利率門檻。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 用估值資料過濾：PE > 0 且 PE < 30，殖利率 > 2%
        df_val = getattr(self, "_df_valuation", pd.DataFrame())
        if not df_val.empty:
            # 取最新一筆估值
            val_latest = df_val.sort_values("date").groupby("stock_id").last().reset_index()
            filtered = filtered.merge(
                val_latest[["stock_id", "pe_ratio", "pb_ratio", "dividend_yield"]],
                on="stock_id",
                how="left",
            )
            # 嚴格模式：必須有估值資料，且 PE 或殖利率至少一項合格
            has_val = filtered["pe_ratio"].notna()
            pe_ok = (filtered["pe_ratio"] > 0) & (filtered["pe_ratio"] < 30)
            dy_ok = filtered["dividend_yield"] > 2.0
            filtered = filtered[has_val & (pe_ok | dy_ok)].copy()
        else:
            filtered["pe_ratio"] = None
            filtered["pb_ratio"] = None
            filtered["dividend_yield"] = None

        if filtered.empty:
            return pd.DataFrame()

        # 粗篩分數：成交量排名 + 法人
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

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

        # 動能欄位
        dates = sorted(df_price["date"].unique())
        if len(dates) >= 2:
            prev_date = dates[-2]
            prev = df_price[df_price["date"] == prev_date][["stock_id", "close"]].copy()
            prev.columns = ["stock_id", "prev_close"]
            filtered = filtered.merge(prev, on="stock_id", how="left")
            filtered["momentum"] = ((filtered["close"] - filtered["prev_close"]) / filtered["prev_close"]).fillna(0)
        else:
            filtered["momentum"] = 0

        filtered["coarse_score"] = filtered["vol_rank"] * 0.50 + filtered["inst_rank"] * 0.50
        filtered = filtered.nlargest(self.top_n_candidates, "coarse_score")
        return filtered

    def _compute_extra_scores(self, stock_ids: list[str]) -> list[pd.DataFrame]:
        """價值模式額外維度：估值面分數。"""
        return [self._compute_valuation_scores(stock_ids)]

    def _post_score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """用 technical_score 欄位存估值分數（供 _rank_and_enrich 顯示用）。"""
        if "valuation_score" in candidates.columns:
            candidates["technical_score"] = candidates["valuation_score"]
        return candidates

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """價值模式基本面 3 因子：YoY 40% + MoM 30% + 營收加速度 30%。"""
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        if rev.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
        mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)

        if "prev_yoy_growth" in rev.columns:
            rev["acceleration"] = rev["yoy_growth"].fillna(0) - rev["prev_yoy_growth"].fillna(0)
            accel_rank = rev["acceleration"].rank(pct=True)
        else:
            accel_rank = pd.Series(0.5, index=rev.index)

        rev["fundamental_score"] = yoy_rank * 0.40 + mom_rank * 0.30 + accel_rank * 0.30

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    def _compute_valuation_scores(self, stock_ids: list[str]) -> pd.DataFrame:
        """估值面 3 因子：PE 反向排名 40% + PB 反向排名 30% + 殖利率排名 30%。"""
        df_val = getattr(self, "_df_valuation", pd.DataFrame())
        if df_val.empty:
            return pd.DataFrame({"stock_id": stock_ids, "valuation_score": [0.5] * len(stock_ids)})

        # 取最新一筆
        val = df_val[df_val["stock_id"].isin(stock_ids)].copy()
        if val.empty:
            return pd.DataFrame({"stock_id": stock_ids, "valuation_score": [0.5] * len(stock_ids)})

        val = val.sort_values("date").groupby("stock_id").last().reset_index()

        # PE 反向排名：PE 越低分數越高
        pe_rank = val["pe_ratio"].fillna(val["pe_ratio"].max()).rank(pct=True, ascending=False)
        # PB 反向排名：PB 越低分數越高
        pb_rank = val["pb_ratio"].fillna(val["pb_ratio"].max()).rank(pct=True, ascending=False)
        # 殖利率正向排名：越高分數越高
        dy_rank = val["dividend_yield"].fillna(0).rank(pct=True)

        val["valuation_score"] = pe_rank * 0.40 + pb_rank * 0.30 + dy_rank * 0.30

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(val[["stock_id", "valuation_score"]], on="stock_id", how="left")
        return result

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """價值模式籌碼面 2 因子：投信近期買超 50% + 三大法人累積 50%。"""
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids)})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
        dates = sorted(df_inst["date"].unique())
        recent_20_dates = dates[-20:] if len(dates) >= 20 else dates

        rows = []
        for sid in stock_ids:
            stock_inst = inst_filtered[inst_filtered["stock_id"] == sid]
            if stock_inst.empty:
                rows.append({"stock_id": sid, "trust_net": 0, "cum_net": 0})
                continue

            trust_data = stock_inst[stock_inst["name"].str.contains("投信", na=False)]
            trust_net = trust_data["net"].sum() if not trust_data.empty else 0

            recent_inst = stock_inst[stock_inst["date"].isin(recent_20_dates)]
            cum_net = recent_inst["net"].sum()

            rows.append({"stock_id": sid, "trust_net": trust_net, "cum_net": cum_net})

        df = pd.DataFrame(rows)
        trust_rank = df["trust_net"].rank(pct=True)
        cum_rank = df["cum_net"].rank(pct=True)
        df["chip_score"] = trust_rank * 0.50 + cum_rank * 0.50

        return df[["stock_id", "chip_score"]]

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """價值模式風險過濾：近 20 日波動率 > 90th percentile 剔除。"""
        if scored.empty or df_price.empty:
            return scored

        vol_data = []
        for sid in scored["stock_id"].tolist():
            stock_data = df_price[df_price["stock_id"] == sid].sort_values("date")
            if len(stock_data) < 10:
                vol_data.append({"stock_id": sid, "vol_20d": 0.0})
                continue

            closes = stock_data["close"].values[-21:] if len(stock_data) >= 21 else stock_data["close"].values
            returns = np.diff(closes) / closes[:-1]
            vol_20d = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
            vol_data.append({"stock_id": sid, "vol_20d": vol_20d})

        df_vol = pd.DataFrame(vol_data)
        threshold = df_vol["vol_20d"].quantile(0.90)
        high_vol_ids = df_vol[df_vol["vol_20d"] > threshold]["stock_id"].tolist()

        before_count = len(scored)
        scored = scored[~scored["stock_id"].isin(high_vol_ids)].copy()
        removed = before_count - len(scored)
        if removed > 0:
            logger.info("Stage 3.5: 波動率風險過濾剔除 %d 支高波動股", removed)

        return scored


# ====================================================================== #
#  DividendScanner — 高息存股模式
# ====================================================================== #


class DividendScanner(MarketScanner):
    """高息存股掃描器。

    篩選高殖利率、配息穩定、估值合理的存股標的。
    粗篩：殖利率 > 3% + PE > 0
    細評：基本面 + 殖利率/估值 + 籌碼面 + 消息面（依 Regime 動態加權）
    風險過濾：近 20 日波動率 > 90th percentile 剔除
    """

    mode_name = "dividend"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 25)
        super().__init__(**kwargs)

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：dividend 模式額外載入估值資料 + 2 個月營收。"""
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        with get_session() as session:
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
                columns=["stock_id", "date", "open", "high", "low", "close", "volume"],
            )

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
                columns=["stock_id", "date", "name", "net"],
            )

            rows = session.execute(
                select(
                    MarginTrading.stock_id,
                    MarginTrading.date,
                    MarginTrading.margin_balance,
                    MarginTrading.short_balance,
                ).where(MarginTrading.date >= cutoff)
            ).all()
            df_margin = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "margin_balance", "short_balance"],
            )

            # 估值資料
            rows = session.execute(
                select(
                    StockValuation.stock_id,
                    StockValuation.date,
                    StockValuation.pe_ratio,
                    StockValuation.pb_ratio,
                    StockValuation.dividend_yield,
                ).where(StockValuation.date >= cutoff)
            ).all()
            self._df_valuation = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "pe_ratio", "pb_ratio", "dividend_yield"],
            )

        # 載入 2 個月營收（含上月，算加速度）
        df_revenue = self._load_revenue_data(months=2)

        return df_price, df_inst, df_margin, df_revenue

    def run(self) -> DiscoveryResult:
        """覆寫 run()：在 Stage 2.5 補抓估值資料。"""
        # Stage 0: Regime 偵測
        try:
            from src.regime.detector import MarketRegimeDetector

            regime_info = MarketRegimeDetector().detect()
            self.regime = regime_info["regime"]
            logger.info("Stage 0: 市場狀態 = %s (TAIEX=%.0f)", self.regime, regime_info["taiex_close"])
        except Exception:
            self.regime = "sideways"
            logger.warning("Stage 0: 市場狀態偵測失敗，預設 sideways")

        # Stage 1
        df_price, df_inst, df_margin, df_revenue = self._load_market_data()
        if df_price.empty:
            logger.warning("無市場資料可供掃描")
            return DiscoveryResult(rankings=pd.DataFrame(), total_stocks=0, after_coarse=0, mode=self.mode_name)

        total_stocks = df_price["stock_id"].nunique()
        logger.info("Stage 1: 載入 %d 支股票的市場資料", total_stocks)

        # Stage 2: 粗篩
        candidates = self._coarse_filter(df_price, df_inst)
        after_coarse = len(candidates)
        logger.info("Stage 2: 粗篩後剩 %d 支候選股", after_coarse)

        if candidates.empty:
            return DiscoveryResult(
                rankings=pd.DataFrame(), total_stocks=total_stocks, after_coarse=0, mode=self.mode_name
            )

        # Stage 2.5: 補抓月營收 + 估值資料
        candidate_ids = candidates["stock_id"].tolist()
        try:
            from src.data.pipeline import sync_revenue_for_stocks, sync_valuation_for_stocks

            logger.info("Stage 2.5: 補抓 %d 支候選股月營收 + 估值...", len(candidate_ids))
            rev_count = sync_revenue_for_stocks(candidate_ids)
            val_count = sync_valuation_for_stocks(candidate_ids)
            logger.info("Stage 2.5: 補抓完成，新增 %d 筆月營收, %d 筆估值", rev_count, val_count)
            df_revenue = self._load_revenue_data(candidate_ids, months=2)
            self._reload_valuation(candidate_ids)
        except Exception:
            logger.warning("Stage 2.5: 資料補抓失敗（可能無 FinMind token），使用既有資料")

        # Stage 2.7: 載入候選股近期 MOPS 公告
        df_ann = self._load_announcement_data(candidate_ids)
        if not df_ann.empty:
            logger.info("Stage 2.7: 載入 %d 筆 MOPS 公告", len(df_ann))
        else:
            logger.info("Stage 2.7: 無 MOPS 公告資料（消息面分數預設 0.5）")

        # Stage 3: 細評
        scored = self._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue, df_ann)
        logger.info("Stage 3: 完成 %d 支候選股評分", len(scored))

        # Stage 3.3: 產業加成
        scored = self._apply_sector_bonus(scored)

        # Stage 3.5: 風險過濾
        scored = self._apply_risk_filter(scored, df_price)

        # Stage 4
        rankings = self._rank_and_enrich(scored)
        sector_summary = self._compute_sector_summary(rankings)
        logger.info("Stage 4: 輸出 Top %d", min(self.top_n_results, len(rankings)))

        return DiscoveryResult(
            rankings=rankings.head(self.top_n_results),
            total_stocks=total_stocks,
            after_coarse=after_coarse,
            sector_summary=sector_summary,
            mode=self.mode_name,
        )

    def _reload_valuation(self, stock_ids: list[str]) -> None:
        """重新載入估值資料（補抓後 DB 已更新）。"""
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)
        with get_session() as session:
            rows = session.execute(
                select(
                    StockValuation.stock_id,
                    StockValuation.date,
                    StockValuation.pe_ratio,
                    StockValuation.pb_ratio,
                    StockValuation.dividend_yield,
                )
                .where(StockValuation.date >= cutoff)
                .where(StockValuation.stock_id.in_(stock_ids))
            ).all()
            self._df_valuation = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "pe_ratio", "pb_ratio", "dividend_yield"],
            )

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """高息模式粗篩：基本過濾 + 殖利率 > 3% + PE > 0。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 用估值資料過濾：必須有估值資料、殖利率 > 3%、PE > 0
        df_val = getattr(self, "_df_valuation", pd.DataFrame())
        if not df_val.empty:
            val_latest = df_val.sort_values("date").groupby("stock_id").last().reset_index()
            filtered = filtered.merge(
                val_latest[["stock_id", "pe_ratio", "pb_ratio", "dividend_yield"]],
                on="stock_id",
                how="left",
            )
            has_val = filtered["dividend_yield"].notna()
            dy_ok = filtered["dividend_yield"] > 3.0
            pe_ok = filtered["pe_ratio"] > 0
            filtered = filtered[has_val & dy_ok & pe_ok].copy()
        else:
            return pd.DataFrame()

        if filtered.empty:
            return pd.DataFrame()

        # 粗篩分數：殖利率排名 50% + 成交量排名 30% + 法人排名 20%
        filtered["dy_rank"] = filtered["dividend_yield"].rank(pct=True)
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

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

        # 動能欄位
        dates = sorted(df_price["date"].unique())
        if len(dates) >= 2:
            prev_date = dates[-2]
            prev = df_price[df_price["date"] == prev_date][["stock_id", "close"]].copy()
            prev.columns = ["stock_id", "prev_close"]
            filtered = filtered.merge(prev, on="stock_id", how="left")
            filtered["momentum"] = ((filtered["close"] - filtered["prev_close"]) / filtered["prev_close"]).fillna(0)
        else:
            filtered["momentum"] = 0

        filtered["coarse_score"] = (
            filtered["dy_rank"] * 0.50 + filtered["vol_rank"] * 0.30 + filtered["inst_rank"] * 0.20
        )
        filtered = filtered.nlargest(self.top_n_candidates, "coarse_score")
        return filtered

    def _compute_extra_scores(self, stock_ids: list[str]) -> list[pd.DataFrame]:
        """高息模式額外維度：殖利率/估值面分數。"""
        return [self._compute_dividend_scores(stock_ids)]

    def _post_score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """用 technical_score 欄位存殖利率分數（供 _rank_and_enrich 顯示用）。"""
        if "dividend_score" in candidates.columns:
            candidates["technical_score"] = candidates["dividend_score"]
        return candidates

    def _compute_dividend_scores(self, stock_ids: list[str]) -> pd.DataFrame:
        """殖利率面 3 因子：殖利率排名 50% + PE 反向排名 30% + PB 反向排名 20%。"""
        df_val = getattr(self, "_df_valuation", pd.DataFrame())
        if df_val.empty:
            return pd.DataFrame({"stock_id": stock_ids, "dividend_score": [0.5] * len(stock_ids)})

        val = df_val[df_val["stock_id"].isin(stock_ids)].copy()
        if val.empty:
            return pd.DataFrame({"stock_id": stock_ids, "dividend_score": [0.5] * len(stock_ids)})

        val = val.sort_values("date").groupby("stock_id").last().reset_index()

        # 殖利率正向排名：越高分數越高
        dy_rank = val["dividend_yield"].fillna(0).rank(pct=True)
        # PE 反向排名：PE 越低分數越高
        pe_rank = val["pe_ratio"].fillna(val["pe_ratio"].max()).rank(pct=True, ascending=False)
        # PB 反向排名：PB 越低分數越高
        pb_rank = val["pb_ratio"].fillna(val["pb_ratio"].max()).rank(pct=True, ascending=False)

        val["dividend_score"] = dy_rank * 0.50 + pe_rank * 0.30 + pb_rank * 0.20

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(val[["stock_id", "dividend_score"]], on="stock_id", how="left")
        return result

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """高息模式基本面 3 因子：YoY 40% + MoM 30% + 營收加速度 30%。"""
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        if rev.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
        mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)

        if "prev_yoy_growth" in rev.columns:
            rev["acceleration"] = rev["yoy_growth"].fillna(0) - rev["prev_yoy_growth"].fillna(0)
            accel_rank = rev["acceleration"].rank(pct=True)
        else:
            accel_rank = pd.Series(0.5, index=rev.index)

        rev["fundamental_score"] = yoy_rank * 0.40 + mom_rank * 0.30 + accel_rank * 0.30

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """高息模式籌碼面 2 因子：投信淨買超 50% + 三大法人累積買超 50%。"""
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids)})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
        dates = sorted(df_inst["date"].unique())
        recent_20_dates = dates[-20:] if len(dates) >= 20 else dates

        rows = []
        for sid in stock_ids:
            stock_inst = inst_filtered[inst_filtered["stock_id"] == sid]
            if stock_inst.empty:
                rows.append({"stock_id": sid, "trust_net": 0, "cum_net": 0})
                continue

            trust_data = stock_inst[stock_inst["name"].str.contains("投信", na=False)]
            trust_net = trust_data["net"].sum() if not trust_data.empty else 0

            recent_inst = stock_inst[stock_inst["date"].isin(recent_20_dates)]
            cum_net = recent_inst["net"].sum()

            rows.append({"stock_id": sid, "trust_net": trust_net, "cum_net": cum_net})

        df = pd.DataFrame(rows)
        trust_rank = df["trust_net"].rank(pct=True)
        cum_rank = df["cum_net"].rank(pct=True)
        df["chip_score"] = trust_rank * 0.50 + cum_rank * 0.50

        return df[["stock_id", "chip_score"]]

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """高息模式風險過濾：近 20 日波動率 > 90th percentile 剔除。"""
        if scored.empty or df_price.empty:
            return scored

        vol_data = []
        for sid in scored["stock_id"].tolist():
            stock_data = df_price[df_price["stock_id"] == sid].sort_values("date")
            if len(stock_data) < 10:
                vol_data.append({"stock_id": sid, "vol_20d": 0.0})
                continue

            closes = stock_data["close"].values[-21:] if len(stock_data) >= 21 else stock_data["close"].values
            returns = np.diff(closes) / closes[:-1]
            vol_20d = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
            vol_data.append({"stock_id": sid, "vol_20d": vol_20d})

        df_vol = pd.DataFrame(vol_data)
        threshold = df_vol["vol_20d"].quantile(0.90)
        high_vol_ids = df_vol[df_vol["vol_20d"] > threshold]["stock_id"].tolist()

        before_count = len(scored)
        scored = scored[~scored["stock_id"].isin(high_vol_ids)].copy()
        removed = before_count - len(scored)
        if removed > 0:
            logger.info("Stage 3.5: 波動率風險過濾剔除 %d 支高波動股", removed)

        return scored


# ====================================================================== #
#  GrowthScanner — 高成長模式
# ====================================================================== #


class GrowthScanner(MarketScanner):
    """高成長掃描器。

    篩選營收/EPS 高速成長、動能啟動的成長型股票。
    粗篩：YoY > 10%
    細評：基本面 + 技術面（動能確認）+ 籌碼面 + 消息面（依 Regime 動態加權）
    風險過濾：ATR(14)/close > 80th percentile 剔除
    """

    mode_name = "growth"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 25)
        super().__init__(**kwargs)

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：growth 模式載入 2 個月營收資料（算加速度）。"""
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        with get_session() as session:
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
                columns=["stock_id", "date", "open", "high", "low", "close", "volume"],
            )

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
                columns=["stock_id", "date", "name", "net"],
            )

            rows = session.execute(
                select(
                    MarginTrading.stock_id,
                    MarginTrading.date,
                    MarginTrading.margin_balance,
                    MarginTrading.short_balance,
                ).where(MarginTrading.date >= cutoff)
            ).all()
            df_margin = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "margin_balance", "short_balance"],
            )

        # 載入 2 個月營收（含上月，用於計算加速度）
        df_revenue = self._load_revenue_data(months=2)

        return df_price, df_inst, df_margin, df_revenue

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """高成長模式粗篩：基本過濾 + YoY > 10%。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 需要營收資料做粗篩
        df_revenue = getattr(self, "_coarse_revenue", pd.DataFrame())
        if df_revenue.empty:
            df_revenue = self._load_revenue_data(months=1)
            self._coarse_revenue = df_revenue

        if not df_revenue.empty:
            filtered = filtered.merge(
                df_revenue[["stock_id", "yoy_growth"]],
                on="stock_id",
                how="left",
            )
            # 必須有營收資料且 YoY > 10%
            has_rev = filtered["yoy_growth"].notna()
            yoy_ok = filtered["yoy_growth"] > 10.0
            filtered = filtered[has_rev & yoy_ok].copy()
        else:
            return pd.DataFrame()

        if filtered.empty:
            return pd.DataFrame()

        # 粗篩分數：YoY 排名 40% + 成交量 30% + 法人 30%
        filtered["yoy_rank"] = filtered["yoy_growth"].rank(pct=True)
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

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

        # 動能欄位
        dates = sorted(df_price["date"].unique())
        if len(dates) >= 2:
            prev_date = dates[-2]
            prev = df_price[df_price["date"] == prev_date][["stock_id", "close"]].copy()
            prev.columns = ["stock_id", "prev_close"]
            filtered = filtered.merge(prev, on="stock_id", how="left")
            filtered["momentum"] = ((filtered["close"] - filtered["prev_close"]) / filtered["prev_close"]).fillna(0)
        else:
            filtered["momentum"] = 0

        filtered["coarse_score"] = (
            filtered["yoy_rank"] * 0.40 + filtered["vol_rank"] * 0.30 + filtered["inst_rank"] * 0.30
        )
        filtered = filtered.nlargest(self.top_n_candidates, "coarse_score")
        return filtered

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """高成長模式技術面 5 因子：5日動能 + 10日動能 + 20日突破 + 量比 + 成交量加速。"""
        results = []

        for sid in stock_ids:
            stock_data = df_price[df_price["stock_id"] == sid].sort_values("date")
            if len(stock_data) < 3:
                results.append({"stock_id": sid, "technical_score": 0.5})
                continue

            closes = stock_data["close"].values
            volumes = stock_data["volume"].values.astype(float)

            score = 0.0
            n_factors = 0

            # 1) 5 日動能
            if len(closes) >= 6:
                ret_5d = (closes[-1] - closes[-6]) / closes[-6]
                score += max(0.0, min(1.0, 0.5 + ret_5d * 5))
                n_factors += 1

            # 2) 10 日動能
            if len(closes) >= 11:
                ret_10d = (closes[-1] - closes[-11]) / closes[-11]
                score += max(0.0, min(1.0, 0.5 + ret_10d * 5))
                n_factors += 1

            # 3) 20 日突破：close / max(close[-20:])
            if len(closes) >= 20:
                max_20 = closes[-20:].max()
                if max_20 > 0:
                    score += closes[-1] / max_20
                else:
                    score += 0.5
                n_factors += 1

            # 4) 量比：volume[-1] / mean(volume[-20:])
            if len(volumes) >= 20:
                avg_vol_20 = volumes[-20:].mean()
                if avg_vol_20 > 0:
                    ratio = min(2.0, volumes[-1] / avg_vol_20)
                    score += ratio / 2.0
                else:
                    score += 0.5
                n_factors += 1

            # 5) 成交量加速：mean(vol[-3:]) / mean(vol[-10:])
            if len(volumes) >= 10:
                avg_vol_3 = volumes[-3:].mean()
                avg_vol_10 = volumes[-10:].mean()
                if avg_vol_10 > 0:
                    ratio = min(2.0, avg_vol_3 / avg_vol_10)
                    score += ratio / 2.0
                else:
                    score += 0.5
                n_factors += 1

            tech_score = score / max(n_factors, 1)
            results.append({"stock_id": sid, "technical_score": tech_score})

        return pd.DataFrame(results)

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """高成長模式籌碼面：外資連續買超 + 買超佔量比 + 三大法人合計 + 券資比（有資料時）。"""
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids)})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
        rows = []
        for sid in stock_ids:
            stock_inst = inst_filtered[inst_filtered["stock_id"] == sid]
            if stock_inst.empty:
                rows.append({"stock_id": sid, "consec_foreign_days": 0, "buy_vol_ratio": 0.0, "total_net": 0})
                continue

            foreign_data = stock_inst[stock_inst["name"].str.contains("外資", na=False)]
            consec_foreign = 0
            if not foreign_data.empty:
                daily_foreign = foreign_data.groupby("date")["net"].sum().sort_index(ascending=False)
                for val in daily_foreign.values:
                    if val > 0:
                        consec_foreign += 1
                    else:
                        break

            total_net = stock_inst["net"].sum()
            buy_vol_ratio = 0.0
            if df_price is not None and not df_price.empty:
                stock_price = df_price[df_price["stock_id"] == sid]
                if not stock_price.empty:
                    latest_vol = stock_price.loc[stock_price["date"].idxmax(), "volume"]
                    if latest_vol > 0:
                        buy_vol_ratio = total_net / latest_vol

            rows.append(
                {
                    "stock_id": sid,
                    "consec_foreign_days": consec_foreign,
                    "buy_vol_ratio": buy_vol_ratio,
                    "total_net": total_net,
                }
            )

        df = pd.DataFrame(rows)

        consec_rank = df["consec_foreign_days"].rank(pct=True)
        bvr_rank = df["buy_vol_ratio"].rank(pct=True)
        total_rank = df["total_net"].rank(pct=True)

        has_margin = df_margin is not None and not df_margin.empty
        if has_margin:
            margin_latest = df_margin[df_margin["date"] == df_margin["date"].max()]
            margin_data = margin_latest[margin_latest["stock_id"].isin(stock_ids)][
                ["stock_id", "margin_balance", "short_balance"]
            ].copy()
            if not margin_data.empty:
                margin_data["short_margin_ratio"] = margin_data.apply(
                    lambda r: r["short_balance"] / r["margin_balance"] if r["margin_balance"] > 0 else 0.0,
                    axis=1,
                )
                df = df.merge(margin_data[["stock_id", "short_margin_ratio"]], on="stock_id", how="left")
                df["short_margin_ratio"] = df["short_margin_ratio"].fillna(0.0)
                smr_rank = df["short_margin_ratio"].rank(pct=True)
                df["chip_score"] = consec_rank * 0.30 + bvr_rank * 0.25 + total_rank * 0.25 + smr_rank * 0.20
            else:
                has_margin = False

        if not has_margin:
            df["chip_score"] = consec_rank * 0.40 + bvr_rank * 0.30 + total_rank * 0.30

        return df[["stock_id", "chip_score"]]

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """高成長模式基本面 3 因子：YoY 40% + MoM 30% + 營收加速度 30%。"""
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        if rev.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
        mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)

        if "prev_yoy_growth" in rev.columns:
            rev["acceleration"] = rev["yoy_growth"].fillna(0) - rev["prev_yoy_growth"].fillna(0)
            accel_rank = rev["acceleration"].rank(pct=True)
        else:
            accel_rank = pd.Series(0.5, index=rev.index)

        rev["fundamental_score"] = yoy_rank * 0.40 + mom_rank * 0.30 + accel_rank * 0.30

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """高成長模式風險過濾：ATR(14)/close > 80th percentile 剔除。"""
        if scored.empty or df_price.empty:
            return scored

        atr_ratios = []
        for sid in scored["stock_id"].tolist():
            stock_data = df_price[df_price["stock_id"] == sid].sort_values("date")
            if len(stock_data) < 14:
                atr_ratios.append({"stock_id": sid, "atr_ratio": 0.0})
                continue

            highs = stock_data["high"].values[-14:]
            lows = stock_data["low"].values[-14:]
            closes = stock_data["close"].values[-15:]

            trs = []
            for i in range(1, len(highs) + 1):
                if i < len(closes):
                    tr = max(
                        highs[i - 1] - lows[i - 1],
                        abs(highs[i - 1] - closes[i - 1]),
                        abs(lows[i - 1] - closes[i - 1]),
                    )
                else:
                    tr = highs[i - 1] - lows[i - 1]
                trs.append(tr)

            atr = np.mean(trs)
            current_close = stock_data["close"].values[-1]
            ratio = atr / current_close if current_close > 0 else 0.0
            atr_ratios.append({"stock_id": sid, "atr_ratio": ratio})

        df_atr = pd.DataFrame(atr_ratios)
        threshold = df_atr["atr_ratio"].quantile(0.80)
        high_vol_ids = df_atr[df_atr["atr_ratio"] > threshold]["stock_id"].tolist()

        before_count = len(scored)
        scored = scored[~scored["stock_id"].isin(high_vol_ids)].copy()
        removed = before_count - len(scored)
        if removed > 0:
            logger.info("Stage 3.5: ATR 風險過濾剔除 %d 支高波動股", removed)

        return scored
