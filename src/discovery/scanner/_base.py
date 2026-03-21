"""全市場選股掃描器 — 基底類別 MarketScanner。

提供四階段漏斗掃描框架，子類透過覆寫 hook 方法實作模式專屬邏輯。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import (
    Announcement,
    BrokerTrade,
    DailyPrice,
    FinancialStatement,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
    SecuritiesLending,
    StockInfo,
    StockValuation,
)
from src.discovery.scanner._functions import (
    DiscoveryResult,
    _calc_atr14,
    compute_abnormal_announcement_rate,
    compute_daytrade_penalty,
    compute_news_decay_weight,
    compute_taiex_relative_strength,
)
from src.discovery.universe import UniverseConfig, UniverseFilter
from src.entry_exit import REGIME_ATR_PARAMS, compute_atr_stops, compute_entry_trigger

logger = logging.getLogger(__name__)


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
    _auto_sync_broker: bool = False  # 子類設為 True 以在 Stage 2.5 自動補抓分點資料
    _revenue_months: int = 1  # 子類可設為 4 以啟用「本月 YoY - 3 個月前 YoY」加速度因子
    _COARSE_WEIGHTS: dict[str, float] = {"vol_rank": 0.30, "inst_rank": 0.40, "mom_rank": 0.30}

    def __init__(
        self,
        min_price: float = 10,
        max_price: float = 2000,
        min_volume: int = 500_000,
        top_n_candidates: int = 150,
        top_n_results: int = 30,
        lookback_days: int = 5,
        weekly_confirm: bool = False,
        universe_config: UniverseConfig | None = None,
    ) -> None:
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
        self.top_n_candidates = top_n_candidates
        self.top_n_results = top_n_results
        self.lookback_days = lookback_days
        self.weekly_confirm = weekly_confirm
        # Universe Filter：各子類可在 __init__ 中傳入模式專屬 config
        self._universe_config = universe_config or UniverseConfig()
        self._universe_filter = UniverseFilter(self._universe_config)

    def run(self) -> DiscoveryResult:
        """執行四階段漏斗掃描。"""
        self.scan_date = date.today()

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

        # Stage 2.5: 補抓候選股分點資料（僅 MomentumScanner 啟用）
        # 新出現的候選股（不在上次推薦或 watchlist 中）也能取得分點評分，避免因無資料而降級
        if self._auto_sync_broker:
            try:
                from src.data.pipeline import sync_broker_for_stocks

                logger.info("Stage 2.5: 補抓 %d 支候選股分點資料（DB 已有近期資料者跳過）...", len(candidate_ids))
                broker_count = sync_broker_for_stocks(candidate_ids)
                logger.info("Stage 2.5: 分點補抓完成，新增 %d 筆", broker_count)
            except Exception:
                logger.warning("Stage 2.5: 分點資料補抓失敗（可能無 FinMind token），使用既有資料")

        # Stage 2.7: 載入候選股近期 MOPS 公告（含基準期歷史供異常率計算）
        df_ann, df_ann_history = self._load_announcement_data(candidate_ids)
        if not df_ann.empty:
            logger.info("Stage 2.7: 載入 %d 筆 MOPS 公告", len(df_ann))
        else:
            logger.info("Stage 2.7: 無 MOPS 公告資料（消息面分數預設 0.5）")

        # Stage 3: 細評
        scored = self._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue, df_ann, df_ann_history)
        logger.info("Stage 3: 完成 %d 支候選股評分", len(scored))

        # Stage 3.3: 產業加成
        scored = self._apply_sector_bonus(scored)

        # Stage 3.3a: 產業同儕相對強度加成
        scored = self._apply_sector_relative_strength(scored)

        # Stage 3.3b: 概念熱度加成（±5%，sector+concept ≤ ±8%）
        scored = self._apply_concept_bonus(scored)

        # Stage 3.4: 週線趨勢加成（若 weekly_confirm=True）
        if self.weekly_confirm:
            scored = self._apply_weekly_trend_bonus(scored)

        # Stage 3.5: 風險過濾
        scored = self._apply_risk_filter(scored, df_price)

        # Stage 3.5b: Crisis 模式相對強度過濾（僅 crisis regime 執行）
        scored = self._apply_crisis_filter(scored, df_price)

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

    def _get_universe_ids(self) -> list[str]:
        """執行 UniverseFilter 三層過濾，回傳候選 stock_id 清單。

        供 _load_market_data() 及子類覆寫版本呼叫，以 IN 子句限定 SQL 查詢範圍。
        若 UniverseFilter 失敗（DB 空等原因）回傳空清單，呼叫端的 SQL 不加 IN 子句。
        """
        universe_ids, universe_stats = self._universe_filter.run(mode=self.mode_name)
        logger.info(
            "Stage 0.5 UniverseFilter: SQL=%d → 流動性=%d → 趨勢=%d → 最終候選=%d",
            universe_stats.get("total_after_sql", 0),
            universe_stats.get("total_after_liquidity", 0),
            universe_stats.get("total_after_trend", 0),
            universe_stats.get("final_candidates", 0),
        )
        return universe_ids

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """從 DB 查詢最近的 daily_price + institutional + margin + monthly_revenue 資料。

        Stage 0.5（Universe Filter）：先執行三層 SQL/Pandas 過濾，取得 ~150-1500 支候選 stock_id，
        再以 IN 子句限定 DailyPrice/InstitutionalInvestor/MarginTrading 查詢範圍，
        避免全量載入 ~6000 支股票，節省約 75% I/O。
        """
        # Stage 0.5: Universe Filter — SQL 硬過濾 + 流動性 + 趨勢
        universe_ids = self._get_universe_ids()

        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        with get_session() as session:
            # 日K線（含 turnover，供流動性評分使用）
            price_query = select(
                DailyPrice.stock_id,
                DailyPrice.date,
                DailyPrice.open,
                DailyPrice.high,
                DailyPrice.low,
                DailyPrice.close,
                DailyPrice.volume,
                DailyPrice.turnover,
            ).where(DailyPrice.date >= cutoff)

            if universe_ids:
                price_query = price_query.where(DailyPrice.stock_id.in_(universe_ids))

            rows = session.execute(price_query).all()
            df_price = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "open", "high", "low", "close", "volume", "turnover"],
            )

            # 三大法人
            inst_query = select(
                InstitutionalInvestor.stock_id,
                InstitutionalInvestor.date,
                InstitutionalInvestor.name,
                InstitutionalInvestor.net,
            ).where(InstitutionalInvestor.date >= cutoff)

            if universe_ids:
                inst_query = inst_query.where(InstitutionalInvestor.stock_id.in_(universe_ids))

            rows = session.execute(inst_query).all()
            df_inst = pd.DataFrame(rows, columns=["stock_id", "date", "name", "net"])

            # 融資融券
            margin_query = select(
                MarginTrading.stock_id,
                MarginTrading.date,
                MarginTrading.margin_balance,
                MarginTrading.short_balance,
            ).where(MarginTrading.date >= cutoff)

            if universe_ids:
                margin_query = margin_query.where(MarginTrading.stock_id.in_(universe_ids))

            rows = session.execute(margin_query).all()
            df_margin = pd.DataFrame(rows, columns=["stock_id", "date", "margin_balance", "short_balance"])

        # 月營收（限候選股）
        df_revenue = self._load_revenue_data(
            stock_ids=universe_ids if universe_ids else None, months=self._revenue_months
        )

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
            cols = ["stock_id", "yoy_growth", "mom_growth", "prev_yoy_growth", "prev_mom_growth"]
            if months >= 4:
                cols.append("yoy_3m_ago")
            return pd.DataFrame(columns=cols)

        # 每支股票取最近 months 筆
        result_rows = []
        for sid, grp in df_all.groupby("stock_id"):
            grp = grp.sort_values("date", ascending=False).head(months)
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

        return pd.DataFrame(result_rows)

    def _load_financial_data(self, stock_ids: list[str], quarters: int = 5) -> pd.DataFrame:
        """從 DB 查詢最近 N 季財務資料（EPS / ROE / 毛利率 / 負債比）。

        Args:
            stock_ids: 限定查詢的股票清單
            quarters: 每支股票取最近幾季（預設 5 季，足以計算 YoY + QoQ）

        Returns:
            DataFrame(stock_id, date, year, quarter, eps, roe, gross_margin, debt_ratio)
            每支股票最多 quarters 筆，按 date desc 排列。無資料時回傳空 DataFrame。
        """
        _cols = ["stock_id", "date", "year", "quarter", "eps", "roe", "gross_margin", "debt_ratio"]
        cutoff = date.today() - timedelta(days=quarters * 100)  # ~100 天/季，5 季 ≈ 500 天
        try:
            with get_session() as session:
                rows = session.execute(
                    select(
                        FinancialStatement.stock_id,
                        FinancialStatement.date,
                        FinancialStatement.year,
                        FinancialStatement.quarter,
                        FinancialStatement.eps,
                        FinancialStatement.roe,
                        FinancialStatement.gross_margin,
                        FinancialStatement.debt_ratio,
                    )
                    .where(
                        FinancialStatement.stock_id.in_(stock_ids),
                        FinancialStatement.date >= cutoff,
                    )
                    .order_by(FinancialStatement.stock_id, FinancialStatement.date.desc())
                ).all()
        except Exception:
            return pd.DataFrame(columns=_cols)

        if not rows:
            return pd.DataFrame(columns=_cols)
        return pd.DataFrame(rows, columns=_cols)

    def _load_announcement_data(
        self,
        stock_ids: list[str] | None = None,
        days: int = 10,
        baseline_days: int = 180,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """從 DB 查詢 MOPS 重大訊息公告（近期 + 基準期歷史）。

        Args:
            stock_ids: 限定查詢的股票清單，None 表示查全部
            days: 近期回溯天數（供評分用）
            baseline_days: 基準期天數（供異常公告率計算用，預設 180）

        Returns:
            (recent_df, history_df)
            - recent_df: 最近 days 天，含 stock_id/date/seq/subject/sentiment/event_type
            - history_df: 最近 baseline_days 天，含 stock_id/date（供異常率計算）
        """
        today = date.today()
        recent_cutoff = today - timedelta(days=days)
        baseline_cutoff = today - timedelta(days=baseline_days)

        col_names = ["stock_id", "date", "seq", "subject", "sentiment", "event_type"]

        with get_session() as session:
            # 近期完整資料
            query = select(
                Announcement.stock_id,
                Announcement.date,
                Announcement.seq,
                Announcement.subject,
                Announcement.sentiment,
                Announcement.event_type,
            ).where(Announcement.date >= recent_cutoff)

            if stock_ids:
                query = query.where(Announcement.stock_id.in_(stock_ids))

            recent_rows = session.execute(query).all()

            # 基準期（僅需 stock_id + date）
            hist_query = select(Announcement.stock_id, Announcement.date).where(Announcement.date >= baseline_cutoff)
            if stock_ids:
                hist_query = hist_query.where(Announcement.stock_id.in_(stock_ids))

            history_rows = session.execute(hist_query).all()

        recent_df = pd.DataFrame(recent_rows, columns=col_names)
        history_df = pd.DataFrame(history_rows, columns=["stock_id", "date"])
        return recent_df, history_df

    def _compute_news_scores(
        self,
        stock_ids: list[str],
        df_ann: pd.DataFrame,
        df_ann_history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """計算消息面分數（時間衰減 × 事件類型加權 × 異常公告率，percentile 排名）。

        公式：
            各公告加權值 = exp(-0.2 × days_ago) × type_weight
            net_score = Σ(加權值 for 正面) - Σ(加權值 for 負面)
            abnormal_multiplier：z>2 最高 +50%，z<-1 最低降至 70%，平常 1.0
            net_score_adj = net_score × abnormal_multiplier
            news_score = percentile_rank(net_score_adj)

        Args:
            stock_ids: 候選股代號清單
            df_ann: 近期公告 DataFrame（須含 sentiment, event_type, date 欄位）
            df_ann_history: 基準期公告歷史（stock_id, date），供異常率計算，None 則略過

        Returns:
            DataFrame(stock_id, news_score) — 分數 0~1，0.5 為中性預設
        """
        default = pd.DataFrame({"stock_id": stock_ids, "news_score": [0.5] * len(stock_ids)})

        if df_ann.empty:
            return default

        ann = df_ann[df_ann["stock_id"].isin(stock_ids)].copy()
        if ann.empty:
            return default

        today = date.today()
        ann["days_ago"] = ann["date"].apply(lambda d: max(0, (today - d).days))

        # event_type 欄位相容（舊資料無此欄則預設 general）
        if "event_type" not in ann.columns:
            ann["event_type"] = "general"
        else:
            ann["event_type"] = ann["event_type"].fillna("general")

        ann["decay_weight"] = ann.apply(
            lambda row: compute_news_decay_weight(row["days_ago"], row["event_type"]),
            axis=1,
        )

        pos_df = ann[ann["sentiment"] == 1].groupby("stock_id")["decay_weight"].sum().reset_index(name="pos_weighted")
        neg_df = ann[ann["sentiment"] == -1].groupby("stock_id")["decay_weight"].sum().reset_index(name="neg_weighted")

        df = pd.DataFrame({"stock_id": stock_ids})
        df = df.merge(pos_df, on="stock_id", how="left")
        df = df.merge(neg_df, on="stock_id", how="left")
        df["pos_weighted"] = df["pos_weighted"].fillna(0.0)
        df["neg_weighted"] = df["neg_weighted"].fillna(0.0)
        df["net_score"] = df["pos_weighted"] - df["neg_weighted"]

        # 異常公告率乘數（僅在 history 有效時套用）
        if df_ann_history is not None and not df_ann_history.empty:
            z_series = compute_abnormal_announcement_rate(df_ann_history, stock_ids)

            def _to_multiplier(z: float) -> float:
                if z > 2.0:
                    return min(1.0 + (z - 2.0) * 0.15, 1.5)
                elif z < -1.0:
                    return max(1.0 + z * 0.10, 0.7)
                return 1.0

            mult_map = {sid: _to_multiplier(float(z)) for sid, z in z_series.items()}
            df["mult"] = df["stock_id"].map(mult_map).fillna(1.0)
            df["net_score_adj"] = df["net_score"] * df["mult"]
        else:
            df["net_score_adj"] = df["net_score"]

        # 無任何加權公告 → 回傳預設
        if df["net_score_adj"].abs().sum() == 0:
            return default

        df["news_score"] = df["net_score_adj"].rank(pct=True)

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
    #  週線趨勢加成
    # ------------------------------------------------------------------ #

    def _compute_weekly_trend_bonus(self, stock_ids: list[str]) -> pd.DataFrame:
        """計算候選股的週線趨勢加成分數（±5%）。

        從 DB 讀取近 90 天日K，聚合為週K，依下列兩個週線信號判斷趨勢：
          - SMA13（13 週均線）：收盤 > SMA13 → 多頭信號
          - RSI14（週 RSI14）：RSI > 50 → 多頭，< 50 → 空頭

        兩信號均多頭 → +0.05；兩信號均空頭 → -0.05；其餘 → 0.0。
        資料不足（< 13 週）時信號以 NaN 填補，該方向的信號直接略過。

        Returns:
            DataFrame(stock_id, weekly_bonus)  值域 {-0.05, 0.0, +0.05}
        """
        from src.features.indicators import aggregate_to_weekly

        default = pd.DataFrame({"stock_id": stock_ids, "weekly_bonus": [0.0] * len(stock_ids)})

        try:
            cutoff = date.today() - timedelta(days=90)

            with get_session() as session:
                rows = (
                    session.execute(
                        select(DailyPrice)
                        .where(DailyPrice.stock_id.in_(stock_ids))
                        .where(DailyPrice.date >= cutoff)
                        .order_by(DailyPrice.stock_id, DailyPrice.date)
                    )
                    .scalars()
                    .all()
                )

            if not rows:
                return default

            df_all = pd.DataFrame(
                [
                    {
                        "stock_id": r.stock_id,
                        "date": r.date,
                        "open": r.open,
                        "high": r.high,
                        "low": r.low,
                        "close": r.close,
                        "volume": r.volume,
                    }
                    for r in rows
                ]
            )

            results: list[dict] = []
            for sid in stock_ids:
                stock_df = df_all[df_all["stock_id"] == sid].drop(columns=["stock_id"])
                if stock_df.empty:
                    results.append({"stock_id": sid, "weekly_bonus": 0.0})
                    continue

                weekly = aggregate_to_weekly(stock_df)
                if weekly.empty:
                    results.append({"stock_id": sid, "weekly_bonus": 0.0})
                    continue

                last = weekly.iloc[-1]
                last_close = float(last["close"])
                sma13 = last["sma_13"]
                rsi14 = last["rsi_14"]

                bullish = 0
                bearish = 0

                if pd.notna(sma13):
                    if last_close > float(sma13):
                        bullish += 1
                    else:
                        bearish += 1

                if pd.notna(rsi14):
                    if float(rsi14) > 50:
                        bullish += 1
                    else:
                        bearish += 1

                if bullish == 2:
                    bonus = 0.05
                elif bearish == 2:
                    bonus = -0.05
                else:
                    bonus = 0.0

                results.append({"stock_id": sid, "weekly_bonus": bonus})

            return pd.DataFrame(results) if results else default

        except Exception:
            logger.warning("週線趨勢加成計算失敗，跳過")
            return default

    def _apply_weekly_trend_bonus(self, scored: pd.DataFrame) -> pd.DataFrame:
        """將週線趨勢加成套用到 composite_score。

        final_score = composite_score × (1 + weekly_bonus)
        """
        if scored.empty:
            return scored

        stock_ids = scored["stock_id"].tolist()
        bonus_df = self._compute_weekly_trend_bonus(stock_ids)
        scored = scored.merge(bonus_df, on="stock_id", how="left")
        scored["weekly_bonus"] = scored["weekly_bonus"].fillna(0.0)
        scored["composite_score"] = scored["composite_score"] * (1 + scored["weekly_bonus"])
        logger.info(
            "Stage 3.4: 週線趨勢加成已套用（範圍 %.3f ~ %.3f）",
            scored["weekly_bonus"].min(),
            scored["weekly_bonus"].max(),
        )
        return scored

    # ------------------------------------------------------------------ #
    #  產業同儕相對強度加成
    # ------------------------------------------------------------------ #

    def _compute_sector_relative_strength(self, stock_ids: list[str]) -> pd.DataFrame:
        """計算個股相對同產業中位數的相對強度加成（±3%）。

        從 DB 讀取近 30 天日K（確保包含 20 個交易日），
        取得 StockInfo 產業對照，呼叫純函數計算。失敗時回傳全 0。

        Returns:
            DataFrame(stock_id, relative_strength_bonus)  值域 {-0.03, 0.0, +0.03}
        """
        from src.industry.analyzer import compute_sector_relative_strength

        default = pd.DataFrame({"stock_id": stock_ids, "relative_strength_bonus": [0.0] * len(stock_ids)})

        try:
            cutoff = date.today() - timedelta(days=30)

            with get_session() as session:
                price_rows = session.execute(
                    select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close)
                    .where(DailyPrice.stock_id.in_(stock_ids))
                    .where(DailyPrice.date >= cutoff)
                    .order_by(DailyPrice.stock_id, DailyPrice.date)
                ).all()

            if not price_rows:
                return default

            df_price = pd.DataFrame([{"stock_id": r[0], "date": r[1], "close": r[2]} for r in price_rows])

            with get_session() as session:
                info_rows = session.execute(
                    select(StockInfo.stock_id, StockInfo.industry_category).where(StockInfo.stock_id.in_(stock_ids))
                ).all()
            industry_map = {r[0]: (r[1] or "未分類") for r in info_rows}

            return compute_sector_relative_strength(stock_ids, df_price, industry_map)

        except Exception:
            logger.warning("產業相對強度計算失敗，跳過")
            return default

    def _apply_sector_relative_strength(self, scored: pd.DataFrame) -> pd.DataFrame:
        """將產業同儕相對強度加成套用到 composite_score。

        final_score = composite_score × (1 + relative_strength_bonus)
        """
        if scored.empty:
            return scored

        stock_ids = scored["stock_id"].tolist()
        rs_df = self._compute_sector_relative_strength(stock_ids)
        scored = scored.merge(rs_df, on="stock_id", how="left")
        scored["relative_strength_bonus"] = scored["relative_strength_bonus"].fillna(0.0)
        scored["composite_score"] = scored["composite_score"] * (1 + scored["relative_strength_bonus"])
        logger.info(
            "Stage 3.3a: 產業相對強度加成已套用（範圍 %.3f ~ %.3f）",
            scored["relative_strength_bonus"].min(),
            scored["relative_strength_bonus"].max(),
        )
        return scored

    # ------------------------------------------------------------------ #
    #  Stage 3.3b: 概念熱度加成
    # ------------------------------------------------------------------ #

    def _compute_concept_bonus(self, stock_ids: list[str]) -> pd.DataFrame:
        """呼叫 ConceptRotationAnalyzer，取得每股概念熱度加成（±5%）。"""
        try:
            from src.industry.concept_analyzer import ConceptRotationAnalyzer

            analyzer = ConceptRotationAnalyzer(
                lookback_days=self.lookback_days,
                momentum_days=self.momentum_days,
            )
            return analyzer.compute_concept_scores_for_stocks(stock_ids, bonus_range=0.05)
        except Exception as exc:
            logger.debug("Stage 3.3b: 概念加成計算失敗，降回 0（%s）", exc)
            return pd.DataFrame({"stock_id": stock_ids, "concept_bonus": [0.0] * len(stock_ids)})

    def _apply_concept_bonus(self, scored: pd.DataFrame) -> pd.DataFrame:
        """套用概念熱度加成（Stage 3.3b）。

        Cap 機制：|sector_bonus| + |concept_bonus_raw| ≤ 8%
        避免已獲高產業加成的股票被概念加成雙重推高。
        """
        if scored.empty:
            return scored

        stock_ids = scored["stock_id"].tolist()
        concept_df = self._compute_concept_bonus(stock_ids)
        scored = scored.merge(concept_df, on="stock_id", how="left")
        scored["concept_bonus"] = scored["concept_bonus"].fillna(0.0)

        # Cap：sector_bonus + concept_bonus 絕對值不超過 ±0.08
        sector = scored.get("sector_bonus", None)
        if sector is None:
            sector = pd.Series(0.0, index=scored.index)
        else:
            sector = sector.fillna(0.0)

        raw = scored["concept_bonus"]
        remaining = (0.08 - sector.abs()).clip(lower=0.0)  # 剩餘可用加成空間
        capped = raw.clip(lower=-remaining, upper=remaining)
        scored["concept_bonus"] = capped

        scored["composite_score"] = scored["composite_score"] * (1 + scored["concept_bonus"])
        logger.info(
            "Stage 3.3b: 概念加成已套用（範圍 %.3f ~ %.3f）",
            scored["concept_bonus"].min(),
            scored["concept_bonus"].max(),
        )
        return scored

    # ------------------------------------------------------------------ #
    #  Stage 2: 粗篩
    # ------------------------------------------------------------------ #

    def _base_filter(self, df_price: pd.DataFrame) -> pd.DataFrame:
        """基礎過濾：股價範圍 + 成交量。供子類 _coarse_filter 呼叫。

        ETF/指數/權證排除已由 UniverseFilter Stage 1（SQL 硬過濾）負責，
        此處僅保留模式專屬的股價區間與成交量門檻。
        """
        latest_date = df_price["date"].max()
        latest = df_price[df_price["date"] == latest_date].copy()

        if latest.empty:
            return pd.DataFrame()

        mask = (latest["close"] >= self.min_price) & (latest["volume"] >= self.min_volume)
        if self.max_price is not None:
            mask = mask & (latest["close"] <= self.max_price)
        return latest[mask].copy()

    def _effective_top_n(self, universe_size: int) -> int:
        """依 Universe 大小自適應粗篩候選數。

        Universe 超過閾值時以 15% 比例線性擴展，防止從大量候選直接壓縮至固定 N 個；
        Universe 較小時以 top_n_candidates 為下限保護，確保候選池不過少。

        例：top_n_candidates=150 時：
            universe=1000 → max(150, 150) = 150
            universe=1500 → max(150, 225) = 225
            universe=200  → max(150,  30) = 150（下限保護）
        """
        return max(self.top_n_candidates, int(universe_size * 0.15))

    # ------------------------------------------------------------------ #
    #  粗篩共用 helpers（供子類 _coarse_filter 呼叫）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_momentum_5d(df_price: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
        """計算 5 日報酬率並加入 momentum 欄位至 filtered DataFrame。

        Args:
            df_price: 完整日K資料（含多日期）。
            filtered: 已過濾的 DataFrame（需有 stock_id, close 欄位）。

        Returns:
            filtered 加上 momentum 欄位（及 mom_rank 若原本有的話由呼叫端自行 rank）。
        """
        dates = sorted(df_price["date"].unique())
        ref_date = dates[-5] if len(dates) >= 5 else (dates[0] if len(dates) >= 2 else None)
        if ref_date is not None:
            ref = df_price[df_price["date"] == ref_date][["stock_id", "close"]].rename(columns={"close": "ref_close"})
            filtered = filtered.merge(ref, on="stock_id", how="left")
            filtered["momentum"] = (
                (filtered["close"] - filtered["ref_close"]) / filtered["ref_close"].replace(0, float("nan"))
            ).fillna(0)
        else:
            filtered["momentum"] = 0
        return filtered

    @staticmethod
    def _compute_inst_net_buy(df_inst: pd.DataFrame, filtered: pd.DataFrame, days: int = 5) -> pd.DataFrame:
        """計算法人 N 日累積淨買超並加入 inst_net / inst_rank 欄位。

        Args:
            df_inst: 法人進出資料。
            filtered: 已過濾的 DataFrame（需有 stock_id 欄位）。
            days: 回溯天數（Momentum/Value/Dividend/Growth=5，Swing=20）。

        Returns:
            filtered 加上 inst_net, inst_rank 欄位。
        """
        if not df_inst.empty:
            inst_dates = sorted(df_inst["date"].unique())
            recent_dates = inst_dates[-days:] if len(inst_dates) >= days else inst_dates
            inst_recent = df_inst[df_inst["date"].isin(recent_dates)]
            inst_net = inst_recent.groupby("stock_id")["net"].sum().reset_index()
            inst_net.columns = ["stock_id", "inst_net"]
            filtered = filtered.merge(inst_net, on="stock_id", how="left")
            filtered["inst_net"] = filtered["inst_net"].fillna(0)
            filtered["inst_rank"] = filtered["inst_net"].rank(pct=True)
        else:
            filtered["inst_net"] = 0
            filtered["inst_rank"] = 0.5
        return filtered

    def _finalize_coarse(self, filtered: pd.DataFrame) -> pd.DataFrame:
        """計算粗篩綜合分並取 top N 候選。

        依 _COARSE_WEIGHTS 類別屬性動態計算 coarse_score 並取 top N。

        Args:
            filtered: 已含各 rank 欄位的 DataFrame。

        Returns:
            top N 候選 DataFrame。
        """
        filtered["coarse_score"] = sum(
            filtered[k] * v for k, v in self._COARSE_WEIGHTS.items() if k in filtered.columns
        )
        filtered = filtered.nlargest(self._effective_top_n(len(filtered)), "coarse_score")
        return filtered

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """粗篩：股價/量/法人/動能加權 → 取 top N candidates。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 1) 成交量排名分數（量大加分）
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 2) 法人 5 日累積淨買超排名
        filtered = self._compute_inst_net_buy(df_inst, filtered, days=5)

        # 3) 短期動能（5 日報酬）+ mom_rank
        filtered = self._compute_momentum_5d(df_price, filtered)
        if (filtered["momentum"] != 0).any():
            filtered["mom_rank"] = filtered["momentum"].rank(pct=True)
        else:
            filtered["mom_rank"] = 0.5

        return self._finalize_coarse(filtered)

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
        df_ann_history: pd.DataFrame | None = None,
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
            self._compute_news_scores(stock_ids, ann, df_ann_history=df_ann_history),
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

        # chip_tier 字串欄位 fillna
        if "chip_tier" in candidates.columns:
            candidates["chip_tier"] = candidates["chip_tier"].fillna("N/A")

        # 隔日沖欄位（從 _compute_chip_scores 暫存）
        dt_df = getattr(self, "_daytrade_penalty_df", None)
        if dt_df is not None and not dt_df.empty:
            candidates = candidates.merge(
                dt_df[["stock_id", "daytrade_penalty", "daytrade_tags"]], on="stock_id", how="left"
            )
        if "daytrade_penalty" not in candidates.columns:
            candidates["daytrade_penalty"] = 0.0
        if "daytrade_tags" not in candidates.columns:
            candidates["daytrade_tags"] = ""
        candidates["daytrade_penalty"] = candidates["daytrade_penalty"].fillna(0.0)
        candidates["daytrade_tags"] = candidates["daytrade_tags"].fillna("")

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

        # 進出場建議欄位（依 regime 調整 ATR 倍數）
        regime = getattr(self, "regime", "sideways")
        entry_exit = self._compute_entry_exit_cols(stock_ids, df_price, regime=regime)
        candidates = candidates.merge(entry_exit, on="stock_id", how="left")

        return candidates

    def _compute_extra_scores(self, stock_ids: list[str]) -> list[pd.DataFrame]:
        """hook：子類可覆寫以新增額外評分維度。回傳 DataFrame 的 list。"""
        return []

    def _post_score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """hook：子類可在加權後做額外處理。"""
        return candidates

    # Regime 自適應 ATR 止損/目標倍數（引用共用常數）
    _REGIME_ATR_PARAMS = REGIME_ATR_PARAMS

    def _compute_entry_exit_cols(
        self, stock_ids: list[str], df_price: pd.DataFrame, regime: str = "sideways"
    ) -> pd.DataFrame:
        """計算每支股票的進出場建議欄位。

        欄位：
          entry_price  — 當日收盤價
          stop_loss    — entry_price - stop_mult × ATR14（依 regime 調整）
          take_profit  — entry_price + target_mult × ATR14（依 regime 調整）
          entry_trigger — 依均線位置與波動率產生中文說明
          valid_until  — scan_date + 5 工作日

        Args:
            stock_ids: 要計算的股票代號清單
            df_price: 日K線 DataFrame
            regime: 市場狀態（"bull"/"sideways"/"bear"），決定 ATR 倍數
                    bull      → stop×1.5 / target×3.5
                    sideways  → stop×1.5 / target×3.0（預設）
                    bear      → stop×1.2 / target×2.5

        Returns:
            DataFrame，index reset，欄位含 stock_id 及上述五欄
        """
        scan_date = getattr(self, "scan_date", date.today())
        valid_until = (pd.Timestamp(scan_date) + pd.offsets.BDay(5)).date()

        # 預先 groupby 一次（O(N)），避免迴圈中反覆 boolean filter（O(N²)）
        price_grouped = df_price.sort_values("date").groupby("stock_id", sort=False)

        rows = []
        for sid in stock_ids:
            stock_data = price_grouped.get_group(sid).tail(30) if sid in price_grouped.groups else pd.DataFrame()

            if stock_data.empty:
                rows.append(
                    {
                        "stock_id": sid,
                        "entry_price": None,
                        "stop_loss": None,
                        "take_profit": None,
                        "entry_trigger": "資料不足，僅供參考",
                        "valid_until": valid_until,
                    }
                )
                continue

            close = float(stock_data["close"].values[-1])

            if len(stock_data) < 15:
                rows.append(
                    {
                        "stock_id": sid,
                        "entry_price": round(close, 2),
                        "stop_loss": None,
                        "take_profit": None,
                        "entry_trigger": "資料不足，僅供參考",
                        "valid_until": valid_until,
                    }
                )
                continue

            atr14 = _calc_atr14(stock_data)
            stop_loss, take_profit = compute_atr_stops(close, atr14, regime)

            # SMA20 — 用 tail(20) 的平均收盤價
            sma20 = float(stock_data["close"].tail(20).mean())
            atr_pct = atr14 / close if close > 0 else 0.0
            trigger = compute_entry_trigger(close, sma20, atr_pct, regime)

            rows.append(
                {
                    "stock_id": sid,
                    "entry_price": round(close, 2),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_trigger": trigger,
                    "valid_until": valid_until,
                }
            )

        return pd.DataFrame(rows)

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """從原始 OHLCV 計算技術面分數（6 因子：SMA + 動能 + 價格位置 + 量能比 + 波動收斂 + 量價背離）。"""
        results = []

        # 預先 groupby 一次（O(N)），避免迴圈中反覆 boolean filter（O(N²)）
        grouped = df_price.sort_values("date").groupby("stock_id", sort=False)

        for sid in stock_ids:
            stock_data = grouped.get_group(sid) if sid in grouped.groups else pd.DataFrame()
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

    def _compute_value_style_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """價值/打底風格技術面 4 因子：120日低點距離 + RSI超賣 + 三均線糾結度 + SMA5右側確認。

        設計理念：價值型投資者偏好「左側佈局」——股價接近中期低點、技術超賣但已出現回穩、
        多條均線糾結收斂（代表蓄積能量），而非短線動能追高。
        F4（右側確認）確保股票雖在低檔，但已有買盤點火，避免純左側接刀。
        需 lookback_days >= 130（支援 SMA120 + RSI 預熱）。
        """
        from src.features.indicators import calc_rsi14_from_series

        results = []
        grouped = df_price.sort_values("date").groupby("stock_id", sort=False)

        for sid in stock_ids:
            stock_data = grouped.get_group(sid) if sid in grouped.groups else pd.DataFrame()
            if len(stock_data) < 15:
                results.append({"stock_id": sid, "technical_score": 0.5})
                continue

            closes = stock_data["close"].values
            score = 0.0
            n_factors = 0

            # 1) 距 120 日低點幅度（接近低點 = 價值區間）
            #    dist_ratio=0 → score=1.0（AT the low）；50% 以上偏離 → score=0.0
            if len(closes) >= 120:
                min_120d = closes[-120:].min()
                if min_120d > 0:
                    dist_ratio = (closes[-1] - min_120d) / min_120d
                    score += max(0.0, 1.0 - dist_ratio / 0.50)
                else:
                    score += 0.5
                n_factors += 1

            # 2) RSI(14) 超賣反轉：低 RSI = 超賣區，對價值型更有吸引力
            #    score = max(0, (70 - RSI) / 70)：RSI=0→1.0、RSI=70→0.0、RSI>70→0.0
            rsi_val = calc_rsi14_from_series(pd.Series(closes))
            score += max(0.0, (70.0 - rsi_val) / 70.0)
            n_factors += 1

            # 3) 三均線糾結度（SMA20/SMA60/SMA120 變異係數，越低越收斂）
            #    均線糾結代表股價進入盤整蓄積，潛在突破機率提高
            if len(closes) >= 120:
                sma20 = closes[-20:].mean()
                sma60 = closes[-60:].mean()
                sma120 = closes[-120:].mean()
                ma_arr = np.array([sma20, sma60, sma120])
                ma_mean = ma_arr.mean()
                if ma_mean > 0:
                    # CV > 6% 視為過度發散 → score=0；CV≈0 完美收斂 → score=1
                    cv = ma_arr.std(ddof=0) / ma_mean
                    score += max(0.0, 1.0 - cv / 0.06)
                else:
                    score += 0.5
                n_factors += 1

            # 4) 右側確認：收盤 > SMA5（打底後買盤點火確認，避免純左側接刀）
            #    左側三因子找到低檔蓄積位置，此因子確認已有買盤啟動
            if len(closes) >= 5:
                sma5 = closes[-5:].mean()
                score += 1.0 if closes[-1] > sma5 else 0.0
                n_factors += 1

            tech_score = score / max(n_factors, 1)
            results.append({"stock_id": sid, "technical_score": tech_score})

        return pd.DataFrame(results)

    def _compute_dividend_style_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """高息存股風格技術面 4 因子：SMA60 斜率 + RSI 中性帶 + 三均線糾結度 + 低波動率。

        設計理念：高息型投資者偏好「穩健上漲或橫盤蓄積」——均線方向平穩向上、
        RSI 在中性帶而非極端值、均線緩緩收斂（打底階段）、波動率低（存股安全感）。
        與 Value 的差異：不追求「超賣反轉」，而是找「趨勢健康、估值未過熱、低波動」的存股。
        需 lookback_days >= 130（支援 SMA120）。
        """
        from src.features.indicators import calc_rsi14_from_series

        results = []
        grouped = df_price.sort_values("date").groupby("stock_id", sort=False)

        for sid in stock_ids:
            stock_data = grouped.get_group(sid) if sid in grouped.groups else pd.DataFrame()
            if len(stock_data) < 15:
                results.append({"stock_id": sid, "technical_score": 0.5})
                continue

            closes = stock_data["close"].values
            score = 0.0
            n_factors = 0

            # 1) SMA60 方向斜率（長期趨勢穩健度）
            #    slope = (SMA60_today - SMA60_20d_ago) / SMA60_20d_ago
            #    平穩上升是存股安全感的核心指標；下滑趨勢則降分
            #    P3 除息缺口免疫：若近 45 日內出現單日跌幅 ≥ 4.5%（台股除息代理訊號），
            #    SMA60 受除息缺口扭曲，改以中性分 0.5 替代，避免高息股被誤殺
            if len(closes) >= 80:  # 60 + 20 buffer
                sma60_today = closes[-60:].mean()
                sma60_20d_ago = closes[-80:-20].mean()
                # 除息缺口偵測：近 45 日是否有單日跌幅 ≥ 4.5%
                window_45 = closes[max(0, len(closes) - 46) :]
                has_ex_div_gap = False
                if len(window_45) >= 2:
                    daily_drops = np.diff(window_45) / window_45[:-1]
                    has_ex_div_gap = bool((daily_drops <= -0.045).any())
                if has_ex_div_gap:
                    score += 0.5  # 除息缺口期間均線斜率不可靠，給中性分
                elif sma60_20d_ago > 0:
                    slope = (sma60_today - sma60_20d_ago) / sma60_20d_ago
                    # ±1% per 20 days → score ≈ 1.0/0.0；斜率=0 → 0.5
                    score += max(0.0, min(1.0, 0.5 + slope * 50))
                else:
                    score += 0.5
                n_factors += 1

            # 2) RSI(14) 中性帶：RSI 在 40~60 為最佳（穩定不過熱）
            #    score = 1 - abs(RSI - 50) / 50：RSI=50→1.0，RSI=0或100→0.0
            rsi_val = calc_rsi14_from_series(pd.Series(closes))
            score += max(0.0, 1.0 - abs(rsi_val - 50.0) / 50.0)
            n_factors += 1

            # 3) 三均線糾結度（SMA20/SMA60/SMA120 變異係數，同 Value style）
            if len(closes) >= 120:
                sma20 = closes[-20:].mean()
                sma60 = closes[-60:].mean()
                sma120 = closes[-120:].mean()
                ma_arr = np.array([sma20, sma60, sma120])
                ma_mean = ma_arr.mean()
                if ma_mean > 0:
                    cv = ma_arr.std(ddof=0) / ma_mean
                    score += max(0.0, 1.0 - cv / 0.06)
                else:
                    score += 0.5
                n_factors += 1

            # 4) 歷史波動率（負向因子）：波動率越低，存股安全感越高
            #    HV = 20 日日報酬率標準差；台灣穩健存股日 HV 約 0.5%~1.5%
            #    以 3% 為上限門檻：HV=0→1.0，HV=3%→0.0，HV>3%→0.0
            if len(closes) >= 21:
                daily_rets = np.diff(closes[-21:]) / closes[-21:-1]
                hv = float(np.std(daily_rets, ddof=1))
                score += max(0.0, 1.0 - hv / 0.03)
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

        # 預先 groupby 一次（O(N)），避免迴圈中反覆 boolean filter（O(N²)）
        inst_grouped = inst_filtered.groupby("stock_id", sort=False)
        price_grouped = (
            df_price[df_price["stock_id"].isin(stock_ids)].groupby("stock_id", sort=False)
            if df_price is not None and not df_price.empty
            else None
        )

        # 計算每支股票的外資、投信、合計淨買超 + 連續買超天數 + 買超佔量比
        rows = []
        for sid in stock_ids:
            if sid not in inst_grouped.groups:
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

            stock_inst = inst_grouped.get_group(sid)
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
            if price_grouped is not None and sid in price_grouped.groups:
                stock_price = price_grouped.get_group(sid)
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

    def _apply_crisis_filter(
        self,
        scored: pd.DataFrame,
        df_price: pd.DataFrame,
        underperform_threshold: float = -0.10,
    ) -> pd.DataFrame:
        """Stage 3.5b — crisis 模式雙重過濾（相對強度 + 絕對趨勢）。

        僅在 self.regime == "crisis" 時執行，兩道濾網：
        1. 相對強度：剔除 20 日超額報酬（個股 − TAIEX）低於 underperform_threshold 的弱勢股
        2. 絕對趨勢：剔除收盤價跌破 60 日均線的股票（防止選到隨大盤跳水的標的）

        非 crisis regime 時零開銷直接回傳原 DataFrame。

        Args:
            scored: _score_candidates 後含 stock_id 欄位的 DataFrame
            df_price: 含 "TAIEX" 的日K線資料（stock_id/date/close）
            underperform_threshold: 相對跌幅門檻，預設 -0.10（跑輸 TAIEX 超過 10pp 即剔除）

        Returns:
            過濾後的 DataFrame
        """
        if getattr(self, "regime", "sideways") != "crisis":
            return scored

        if scored.empty or df_price.empty:
            return scored

        # --- 濾網 1：相對強度（vs TAIEX） ---
        excess = compute_taiex_relative_strength(df_price, window=20)
        if not excess.empty:
            weak_ids = set(excess[excess < underperform_threshold].index)
            if weak_ids:
                before = len(scored)
                scored = scored[~scored["stock_id"].isin(weak_ids)].copy()
                removed = before - len(scored)
                if removed > 0:
                    logger.info(
                        "Stage 3.5b Crisis 相對強度過濾：剔除 %d 支跑輸 TAIEX 超過 %.0f%% 的弱勢股",
                        removed,
                        abs(underperform_threshold) * 100,
                    )

        # --- 濾網 2：絕對趨勢（close < MA60 剔除） ---
        if scored.empty:
            return scored

        non_taiex = df_price[df_price["stock_id"] != "TAIEX"]
        if non_taiex.empty:
            return scored

        candidate_ids = set(scored["stock_id"])
        candidate_prices = non_taiex[non_taiex["stock_id"].isin(candidate_ids)]
        if candidate_prices.empty:
            return scored

        below_ma60_ids: set[str] = set()
        for sid, grp in candidate_prices.sort_values("date").groupby("stock_id"):
            if len(grp) < 60:
                continue  # 資料不足 60 天無法計算 MA60，不懲罰
            ma60 = grp["close"].rolling(60, min_periods=60).mean().iloc[-1]
            latest_close = grp["close"].iloc[-1]
            if pd.notna(ma60) and latest_close < ma60:
                below_ma60_ids.add(sid)

        if below_ma60_ids:
            before = len(scored)
            scored = scored[~scored["stock_id"].isin(below_ma60_ids)].copy()
            removed = before - len(scored)
            if removed > 0:
                logger.info(
                    "Stage 3.5b Crisis 絕對趨勢過濾：剔除 %d 支收盤價跌破 MA60 的股票",
                    removed,
                )

        return scored

    # ------------------------------------------------------------------ #
    #  共用風險過濾 helpers（子類呼叫，不須覆寫整個方法）
    # ------------------------------------------------------------------ #

    def _apply_atr_risk_filter(
        self,
        scored: pd.DataFrame,
        df_price: pd.DataFrame,
        percentile: int = 80,
        absolute_cap: float = 0.08,
    ) -> pd.DataFrame:
        """ATR-based 風險過濾：ATR(14)/close > N-th percentile 或 > absolute_cap 的股票剔除。

        雙重門檻取嚴格者：相對 percentile + 絕對值上限（預設 8%），
        避免市場整體高波動時 percentile 門檻過度上移。
        """
        if scored.empty or df_price.empty:
            return scored

        grouped = df_price.sort_values("date").groupby("stock_id", sort=False)
        atr_ratios = []
        for sid in scored["stock_id"].tolist():
            stock_data = grouped.get_group(sid) if sid in grouped.groups else pd.DataFrame()
            atr = _calc_atr14(stock_data)
            current_close = stock_data["close"].values[-1] if not stock_data.empty else 1.0
            ratio = atr / current_close if current_close > 0 else 0.0
            atr_ratios.append({"stock_id": sid, "atr_ratio": ratio})

        df_atr = pd.DataFrame(atr_ratios)
        pct_threshold = df_atr["atr_ratio"].quantile(percentile / 100)
        # 取 percentile 門檻與絕對值 cap 的較嚴格者（較低值）
        threshold = min(pct_threshold, absolute_cap)
        high_vol_ids = df_atr[df_atr["atr_ratio"] > threshold]["stock_id"].tolist()

        before_count = len(scored)
        scored = scored[~scored["stock_id"].isin(high_vol_ids)].copy()
        removed = before_count - len(scored)
        if removed > 0:
            logger.info("Stage 3.5: ATR 風險過濾剔除 %d 支高波動股（門檻 %.2f%%）", removed, threshold * 100)
        return scored

    def _apply_vol_risk_filter(
        self,
        scored: pd.DataFrame,
        df_price: pd.DataFrame,
        percentile: int,
        window: int = 20,
        annualize: bool = False,
        absolute_cap: float | None = None,
    ) -> pd.DataFrame:
        """波動率-based 風險過濾：N 日波動率 > M-th percentile 或 > absolute_cap 的股票剔除。

        Args:
            percentile: 剔除閾值（80 表示剔除波動率超過第 80 百分位數的股票）
            window: 計算波動率的回溯天數（預設 20）
            annualize: 是否年化（乘以 sqrt(252)）
            absolute_cap: 絕對波動率上限（預設 None 自動推算：
                          annualize=True 時 0.80（80%年化），
                          annualize=False 時 0.05（5%日波動率））
        """
        if scored.empty or df_price.empty:
            return scored

        if absolute_cap is None:
            absolute_cap = 0.80 if annualize else 0.05

        grouped = df_price.sort_values("date").groupby("stock_id", sort=False)
        vol_data = []
        for sid in scored["stock_id"].tolist():
            stock_data = grouped.get_group(sid) if sid in grouped.groups else pd.DataFrame()
            if len(stock_data) < 10:
                # 資料不足：設為極大值以確保被剔除（防禦性設計）
                vol_data.append({"stock_id": sid, "vol": np.inf})
                continue

            closes = (
                stock_data["close"].values[-(window + 1) :]
                if len(stock_data) >= window + 1
                else stock_data["close"].values
            )
            returns = np.diff(closes) / closes[:-1]
            vol = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
            if annualize:
                vol = vol * np.sqrt(252)
            vol_data.append({"stock_id": sid, "vol": vol})

        df_vol = pd.DataFrame(vol_data)
        pct_threshold = df_vol[df_vol["vol"] < np.inf]["vol"].quantile(percentile / 100)
        # 取 percentile 門檻與絕對值 cap 的較嚴格者
        threshold = min(pct_threshold, absolute_cap) if pd.notna(pct_threshold) else absolute_cap
        high_vol_ids = df_vol[df_vol["vol"] > threshold]["stock_id"].tolist()

        before_count = len(scored)
        scored = scored[~scored["stock_id"].isin(high_vol_ids)].copy()
        removed = before_count - len(scored)
        if removed > 0:
            logger.info("Stage 3.5: 波動率風險過濾剔除 %d 支高波動股", removed)
        return scored

    def _reload_valuation(self, stock_ids: list[str]) -> None:
        """重新載入估值資料（補抓後 DB 已更新）。供 ValueScanner / DividendScanner 呼叫。"""
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

    def _maybe_sync_valuation(self) -> None:
        """Stage 0.5：估值資料覆蓋不足時，自動從 TWSE/TPEX 補抓全市場估值。
        供 ValueScanner / DividendScanner 的 run() 呼叫。
        """
        try:
            from sqlalchemy import func as sa_func

            with get_session() as session:
                val_count = session.execute(select(sa_func.count(sa_func.distinct(StockValuation.stock_id)))).scalar()
            if not val_count or val_count < 500:
                logger.info(
                    "Stage 0.5: 估值資料僅 %d 支，自動從 TWSE/TPEX 同步全市場估值...",
                    val_count or 0,
                )
                from src.data.pipeline import sync_valuation_all_market

                val_synced = sync_valuation_all_market()
                logger.info("Stage 0.5: 全市場估值同步完成，新增 %d 筆", val_synced)
        except Exception:
            logger.warning("Stage 0.5: 全市場估值自動同步失敗，使用既有資料繼續")

    def _compute_momentum_style_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """動能風格技術面 6 因子（橫截面排名版）。

        5日動能 + 10日動能 + 20日突破 + 量比（20日）+ 成交量加速 + 風險調整後動能。
        所有因子改用橫截面 rank(pct=True)，自動適應牛熊市（Regime Adaptable），
        消除原本 clamp(0.5 + ret×5) 在強市場天花板化的鑑別度喪失問題。
        第 6 因子（風險調整後動能）= Return_10d / Volatility_20d（類 Sharpe），
        偏好「上漲過程平穩」的高品質動能股，過濾暴漲暴跌的妖股。
        資料不足或缺失的因子以中性分 0.5 填補。
        供 MomentumScanner 與 GrowthScanner 共用。
        """
        if not stock_ids:
            return pd.DataFrame(columns=["stock_id", "technical_score"])

        df = df_price.sort_values(["stock_id", "date"])
        g = df.groupby("stock_id", sort=False)

        # ── 批次取各期收盤價 ──────────────────────────────────────────
        # 使用 apply 確保結果以 stock_id 為 index（nth() 返回的是原始整數 index，
        # 導致後續 reindex 全部得到 NaN）
        latest_close = g["close"].last()
        close_1d_ago = g["close"].apply(
            lambda s: float(s.iloc[-2]) if len(s) >= 2 else np.nan
        )  # 前一交易日（漲停偵測用）
        close_5d_ago = g["close"].apply(lambda s: float(s.iloc[-6]) if len(s) >= 6 else np.nan)  # 5 個交易日前
        close_10d_ago = g["close"].apply(lambda s: float(s.iloc[-11]) if len(s) >= 11 else np.nan)  # 10 個交易日前
        # 近 60 日最高收盤（季線突破；20日易受短期雜訊干擾，延長至 60 日延續性更強）
        close_60d_max = g["close"].apply(lambda s: float(s.iloc[-60:].max()) if len(s) >= 60 else np.nan)

        # ── 批次取量能序列 ────────────────────────────────────────────
        latest_vol = g["volume"].last().astype(float)
        vol_20d_mean = g["volume"].apply(lambda s: float(s.astype(float).iloc[-20:].mean()) if len(s) >= 20 else np.nan)
        vol_3d_mean = g["volume"].apply(lambda s: float(s.astype(float).iloc[-3:].mean()) if len(s) >= 3 else np.nan)
        vol_10d_mean = g["volume"].apply(lambda s: float(s.astype(float).iloc[-10:].mean()) if len(s) >= 10 else np.nan)

        # ── 第 6 因子：20 日日報酬率標準差（Volatility_20d）────────────
        # 計算 20 日滾動日報酬率的標準差，供風險調整後動能使用
        vol_20d_std = g["close"].apply(
            lambda s: float(s.pct_change().iloc[-20:].dropna().std()) if len(s) >= 21 else np.nan
        )

        # ── 限縮到候選股集合，計算原始因子值 ─────────────────────────
        idx = pd.Index(stock_ids)
        c0 = latest_close.reindex(idx)
        c5 = close_5d_ago.reindex(idx).replace(0, np.nan)
        c10 = close_10d_ago.reindex(idx).replace(0, np.nan)
        c60m = close_60d_max.reindex(idx).replace(0, np.nan)

        ret_5d = (c0 - c5) / c5
        ret_10d = (c0 - c10) / c10
        breakout_60d = c0 / c60m

        vol_20d = vol_20d_mean.reindex(idx).replace(0, np.nan)
        vol_ratio_raw = latest_vol.reindex(idx) / vol_20d
        vol_accel_raw = vol_3d_mean.reindex(idx) / vol_10d_mean.reindex(idx).replace(0, np.nan)

        # 風險調整後動能：Return_10d / Volatility_20d（類 Sharpe ratio）
        vv = vol_20d_std.reindex(idx).replace(0, np.nan)
        sharpe_proxy = ret_10d / vv

        # ── 橫截面百分位排名（Regime Adaptive）─────────────────────────
        r5 = ret_5d.rank(pct=True)
        r10 = ret_10d.rank(pct=True)
        rb = breakout_60d.rank(pct=True)
        rv = vol_ratio_raw.rank(pct=True)
        ra = vol_accel_raw.rank(pct=True)
        rs = sharpe_proxy.rank(pct=True)  # 風險調整後動能排名

        # ── 漲停板特殊處理（台股 10% 漲跌幅限制）────────────────────────
        # 強勢「鎖漲停」時成交量急縮屬正常現象，不應懲罰量比/量能加速因子。
        # 偵測：當日漲幅 ≥ 9.8%（台股實際漲停幅度因計算略低於 10%）
        c1d = close_1d_ago.reindex(idx).replace(0, np.nan)
        limit_up_mask = ((c0 - c1d) / c1d) >= 0.098
        rv = rv.where(~limit_up_mask, other=1.0)
        ra = ra.where(~limit_up_mask, other=1.0)

        # NaN（資料不足）以中性 0.5 填補，取六因子等權平均
        scores = pd.concat([r5, r10, rb, rv, ra, rs], axis=1)
        scores.columns = ["r5", "r10", "rb", "rv", "ra", "rs"]
        scores = scores.fillna(0.5)

        tech_score = scores.mean(axis=1)
        return pd.DataFrame({"stock_id": idx.tolist(), "technical_score": tech_score.to_numpy()})

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
            "chip_tier",
            "fundamental_score",
            "news_score",
            "sector_bonus",
            "concept_bonus",
            "daytrade_penalty",
            "daytrade_tags",
            "industry_category",
            "momentum",
            "inst_net",
            "entry_price",
            "stop_loss",
            "take_profit",
            "entry_trigger",
            "valid_until",
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

    def _load_broker_data(self, stock_ids: list[str]) -> pd.DataFrame:
        """從 DB 載入最近 7 天的分點交易資料。

        共用方法（MarketScanner 基底），供 MomentumScanner / SwingScanner /
        ValueScanner / GrowthScanner 的 _compute_chip_scores() 呼叫。
        若表不存在或無資料則回傳空 DataFrame，呼叫端自動降級。
        """
        cutoff = date.today() - timedelta(days=7)
        try:
            with get_session() as session:
                rows = session.execute(
                    select(
                        BrokerTrade.stock_id,
                        BrokerTrade.date,
                        BrokerTrade.broker_id,
                        BrokerTrade.broker_name,
                        BrokerTrade.buy,
                        BrokerTrade.sell,
                    ).where(
                        BrokerTrade.stock_id.in_(stock_ids),
                        BrokerTrade.date >= cutoff,
                    )
                ).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=["stock_id", "date", "broker_id", "broker_name", "buy", "sell"])
        except Exception:
            return pd.DataFrame()

    def _load_sbl_data(self, stock_ids: list[str]) -> pd.DataFrame:
        """從 DB 載入最近 5 天的借券賣出彙總資料。

        共用方法（MarketScanner 基底），供 MomentumScanner / SwingScanner 的
        _compute_chip_scores() 呼叫。
        若表不存在或無資料則回傳空 DataFrame，呼叫端自動降級。
        """
        cutoff = date.today() - timedelta(days=5)
        try:
            with get_session() as session:
                rows = session.execute(
                    select(
                        SecuritiesLending.stock_id,
                        SecuritiesLending.date,
                        SecuritiesLending.sbl_balance,
                    ).where(
                        SecuritiesLending.stock_id.in_(stock_ids),
                        SecuritiesLending.date >= cutoff,
                    )
                ).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=["stock_id", "date", "sbl_balance"])
        except Exception:
            return pd.DataFrame()

    def _load_broker_data_extended(
        self,
        stock_ids: list[str],
        days: int = 365,
        min_trading_days: int = 20,
    ) -> pd.DataFrame:
        """從 DB 載入所有可用的分點交易資料（含 buy_price / sell_price）。

        共用方法（MarketScanner 基底），供 MomentumScanner / SwingScanner 的
        _compute_chip_scores() 呼叫（Smart Broker 因子）。

        查詢窗口改為 days=365，充分利用 daily sync 累積的歷史資料。
        min_trading_days：每支股票至少需有 N 個交易日資料，否則排除（避免假信號）。

        均價代理策略（方案 B）：
          - DJ 端點不提供 buy_price / sell_price，欄位存 NULL
          - 本函數自動以 DailyPrice.close 填補 NULL 均價（同日收盤價）
          - win_rate / PF 的意義：衡量分點「是否在漲前買、跌前賣」的擇時能力
        """
        cutoff = date.today() - timedelta(days=days)
        try:
            with get_session() as session:
                rows = session.execute(
                    select(
                        BrokerTrade.stock_id,
                        BrokerTrade.date,
                        BrokerTrade.broker_id,
                        BrokerTrade.broker_name,
                        BrokerTrade.buy,
                        BrokerTrade.sell,
                        BrokerTrade.buy_price,
                        BrokerTrade.sell_price,
                    ).where(
                        BrokerTrade.stock_id.in_(stock_ids),
                        BrokerTrade.date >= cutoff,
                    )
                ).all()
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(
                    rows,
                    columns=[
                        "stock_id",
                        "date",
                        "broker_id",
                        "broker_name",
                        "buy",
                        "sell",
                        "buy_price",
                        "sell_price",
                    ],
                )
                # ── 均價代理：以 DailyPrice.close 填補 NULL buy_price / sell_price ──
                # 在同一 session 內查詢，避免開啟第二個連線
                if df["buy_price"].isna().any() or df["sell_price"].isna().any():
                    try:
                        price_rows = session.execute(
                            select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                                DailyPrice.stock_id.in_(stock_ids),
                                DailyPrice.date >= cutoff,
                            )
                        ).all()
                        if price_rows:
                            price_df = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])
                            df = df.merge(price_df, on=["stock_id", "date"], how="left")
                            df["buy_price"] = df["buy_price"].astype("float64").fillna(df["close"])
                            df["sell_price"] = df["sell_price"].astype("float64").fillna(df["close"])
                            df = df.drop(columns=["close"])
                    except Exception:
                        pass  # 無法載入收盤價時保持原始 NULL，系統降回 7F
            # 過濾掉歷史資料不足的股票，避免以少量資料誤判分點行為
            if min_trading_days > 0 and not df.empty:
                day_counts = df.groupby("stock_id")["date"].nunique()
                valid_stocks = day_counts[day_counts >= min_trading_days].index
                df = df[df["stock_id"].isin(valid_stocks)]
            return df
        except Exception:
            return pd.DataFrame()

    def _apply_daytrade_penalty(
        self,
        broker_rank: pd.Series,
        df_broker: pd.DataFrame,
        stock_ids: list[str],
        df_price: pd.DataFrame | None = None,
        penalty_factor: float = 0.5,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """對 broker_rank 施加隔日沖扣分，同時回傳 penalty_df 供持久化。

        隔日沖分點（黑名單 + 行為偵測）的買超佔比越高，broker_rank 扣分越重。
        最多扣除 penalty_factor（預設 50%），避免單一負面因子完全壓制分點因子。

        Args:
            broker_rank: 分點因子的 percentile rank Series（0~1），index 對齊 stock_ids
            df_broker: BrokerTrade DataFrame（含 broker_name 欄位）
            stock_ids: 候選股代號清單
            df_price: 日K 線資料（可選，用於計算 20 日均量做流動性閾值）
            penalty_factor: 最大扣分比例（預設 0.5，即 penalty=1.0 時 rank 打 5 折）

        Returns:
            (adjusted_broker_rank, penalty_df)
            - adjusted_broker_rank: 扣分後的 rank Series
            - penalty_df: DataFrame [stock_id, daytrade_penalty, daytrade_tags]
        """
        empty_penalty = pd.DataFrame({"stock_id": stock_ids, "daytrade_penalty": 0.0, "daytrade_tags": ""})

        if df_broker.empty or "broker_name" not in df_broker.columns:
            return broker_rank, empty_penalty

        # 計算 20 日均量（用於流動性閾值）
        df_vol = None
        if df_price is not None and not df_price.empty:
            vol_data = df_price[df_price["stock_id"].isin(stock_ids)].copy()
            if not vol_data.empty:
                avg_vol = (
                    vol_data.sort_values("date")
                    .groupby("stock_id")["volume"]
                    .apply(lambda s: s.tail(20).mean())
                    .reset_index()
                )
                avg_vol.columns = ["stock_id", "avg_volume_20d"]
                df_vol = avg_vol

        penalty_df = compute_daytrade_penalty(
            df_broker,
            df_volume=df_vol,
        )

        if penalty_df.empty:
            return broker_rank, empty_penalty

        # 建立 stock_id → penalty 映射
        penalty_map = dict(zip(penalty_df["stock_id"], penalty_df["daytrade_penalty"], strict=False))
        tags_map = dict(zip(penalty_df["stock_id"], penalty_df["top_dt_brokers"], strict=False))

        # 扣分：rank *= (1 - penalty × penalty_factor)
        adjusted = broker_rank.copy()
        dt_penalties = []
        dt_tags = []
        for i, sid in enumerate(stock_ids):
            p = penalty_map.get(sid, 0.0)
            dt_penalties.append(p)
            dt_tags.append(tags_map.get(sid, ""))
            if p > 0:
                adjusted.iloc[i] = adjusted.iloc[i] * (1 - p * penalty_factor)

        result_df = pd.DataFrame({"stock_id": stock_ids, "daytrade_penalty": dt_penalties, "daytrade_tags": dt_tags})
        return adjusted, result_df


# ====================================================================== #
#  MomentumScanner — 短線動能模式
# ====================================================================== #
