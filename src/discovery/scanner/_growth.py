"""GrowthScanner — 高成長掃描模式。

營收高速成長 + 動能啟動，適合追逐成長股。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import (
    DailyPrice,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
)
from src.discovery.scanner._base import MarketScanner
from src.discovery.scanner._functions import (
    DiscoveryResult,
    compute_broker_score,
)
from src.discovery.universe import UniverseConfig

logger = logging.getLogger(__name__)


class GrowthScanner(MarketScanner):
    """高成長掃描器。

    篩選營收/EPS 高速成長、動能啟動的成長型股票。
    粗篩：YoY > 10%
    細評：基本面 + 技術面（動能確認）+ 籌碼面 + 消息面（依 Regime 動態加權）
    風險過濾：ATR(14)/close > 80th percentile 剔除
    """

    mode_name = "growth"
    _COARSE_WEIGHTS: dict[str, float] = {"yoy_rank": 0.40, "vol_rank": 0.30, "inst_rank": 0.30}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 80)  # 共用動能技術面評分，F3 季線突破需 60 交易日（80 曆日）
        kwargs.setdefault(
            "universe_config",
            UniverseConfig(
                min_close=5.0,
                trend_ma=20,
                volume_ratio_min=2.0,
                min_available_days=30,
                trend_filter_mode="trend_or_breakout",
            ),
        )
        super().__init__(**kwargs)

    def run(self, shared=None, precomputed_ic=None) -> DiscoveryResult:
        """覆寫 run()：粗篩前自動同步 MOPS 全市場月營收。

        Args:
            shared: 項目 B — 由 `_cmd_discover_all` 預載入的全市場資料；
                傳入時 `_load_market_data` 以 in-memory 過濾取代 DB 查詢。
            precomputed_ic: 項目 E — Step 8c 預算的 static IC DataFrame，
                供 `_apply_ic_weight_adjustment` / `_log_factor_effectiveness` 短路。
        """
        self._shared = shared
        self._precomputed_ic = precomputed_ic
        # Stage 0: Regime 偵測
        try:
            from src.regime.detector import MarketRegimeDetector

            regime_info = MarketRegimeDetector().detect()
            self.regime = regime_info["regime"]
            logger.info("Stage 0: 市場狀態 = %s (TAIEX=%.0f)", self.regime, regime_info["taiex_close"])
        except Exception:
            self.regime = "sideways"
            logger.warning("Stage 0: 市場狀態偵測失敗，預設 sideways")

        # Stage 0.1: Regime gate（與 MarketScanner.run() 共用邏輯）
        if self._is_regime_blocked():
            logger.warning(
                "Stage 0.1: %s 模式在 %s 市場暫停掃描（歷史績效不佳）",
                self.mode_name,
                self.regime,
            )
            return DiscoveryResult(rankings=pd.DataFrame(), total_stocks=0, after_coarse=0, mode=self.mode_name)

        # Stage 0.5: 檢查月營收覆蓋率，不足時自動從 MOPS 補抓
        try:
            from sqlalchemy import func as sa_func

            with get_session() as session:
                rev_count = session.execute(select(sa_func.count(sa_func.distinct(MonthlyRevenue.stock_id)))).scalar()

            if not rev_count or rev_count < 500:
                logger.info(
                    "Stage 0.5: 月營收僅 %d 支，自動從 MOPS 同步全市場月營收...",
                    rev_count or 0,
                )
                from src.data.pipeline import sync_mops_revenue

                mops_count = sync_mops_revenue(months=1)
                logger.info("Stage 0.5: MOPS 月營收同步完成，新增 %d 筆", mops_count)
        except Exception:
            logger.warning("Stage 0.5: MOPS 月營收自動同步失敗，使用既有資料")

        # Stage 1: 載入資料
        df_price, df_inst, df_margin, df_revenue = self._load_market_data()
        if df_price.empty:
            logger.warning("無市場資料可供掃描")
            return DiscoveryResult(rankings=pd.DataFrame(), total_stocks=0, after_coarse=0, mode=self.mode_name)

        total_stocks = df_price["stock_id"].nunique()
        logger.info("Stage 1: 載入 %d 支股票的市場資料", total_stocks)

        # 預填充 _coarse_revenue，避免 _coarse_filter() 重複查詢 DB（Problem 4 修正）
        # _load_market_data() 已載入 4 個月營收，直接重用，無需再次查詢
        self._coarse_revenue = df_revenue

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
            df_revenue = self._load_revenue_data(candidate_ids, months=4)
        except Exception:
            logger.warning("Stage 2.5: 資料補抓失敗（可能無 FinMind token），使用既有資料")

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

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：growth 模式載入 4 個月營收資料（算加速度）。

        項目 B：若 `self._shared` 已由 `run(shared=...)` 注入，則以 in-memory 過濾
        取代 DB 查詢，month=4 透過 `revenue_months` 參數顯式對齊原行為。
        """
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        shared = getattr(self, "_shared", None)
        if shared is not None:
            # Growth 不做 UniverseFilter（全市場掃）→ universe_ids 傳空清單
            return self._slice_shared_market_data(shared, [], cutoff, revenue_months=4)

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

        # 載入 4 個月營收（含 3 個月前，用於計算加速度）
        df_revenue = self._load_revenue_data(months=4)

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

        # 2) 法人 5 日累積淨買超排名
        filtered = self._compute_inst_net_buy(df_inst, filtered, days=5)

        # 3) 短期動能（5 日報酬，用於 _rank_and_enrich 保留，不參與 _COARSE_WEIGHTS）
        filtered = self._compute_momentum_5d(df_price, filtered)

        return self._finalize_coarse(filtered)

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """高成長模式技術面 5 因子（委派至 base class 共用動能評分實作）。"""
        return self._compute_momentum_style_technical_scores(stock_ids, df_price)

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """高成長模式籌碼面：外資連續買超 + 買超佔量比 + 三大法人合計 + 券資比 + 分點集中度（有資料時）。

        權重組合：
        - 5 因子（券資比 + 分點）: 外資 25% + 量比 22% + 法人 22% + 券資比 16% + 分點 15%
        - 4 因子（含券資比）:      外資 30% + 量比 25% + 法人 25% + 券資比 20%
        - 4 因子（含分點）:        外資 32% + 量比 24% + 法人 24% + 分點 20%
        - 3 因子（基本）:          外資 40% + 量比 30% + 法人 30%

        回傳欄位：stock_id, chip_score, chip_tier（"5F"、"4F" 或 "3F"）
        """
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids), "chip_tier": "N/A"})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
        inst_grouped = inst_filtered.groupby("stock_id", sort=False)
        price_grouped = (
            df_price[df_price["stock_id"].isin(stock_ids)].groupby("stock_id", sort=False)
            if df_price is not None and not df_price.empty
            else None
        )
        rows = []
        for sid in stock_ids:
            if sid not in inst_grouped.groups:
                rows.append({"stock_id": sid, "consec_foreign_days": 0, "buy_vol_ratio": 0.0, "total_net": 0})
                continue
            stock_inst = inst_grouped.get_group(sid)

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
            if price_grouped is not None and sid in price_grouped.groups:
                stock_price = price_grouped.get_group(sid)
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

        # ── 券資比因子 ────────────────────────────────────────────────
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
            else:
                has_margin = False

        # ── 分點集中度因子 ────────────────────────────────────────────
        df_broker_raw = self._load_broker_data(stock_ids)
        broker_df = compute_broker_score(df_broker_raw)
        has_broker = not broker_df.empty
        if has_broker:
            df = df.merge(broker_df, on="stock_id", how="left")
            df["broker_concentration"] = df["broker_concentration"].fillna(0.0)
            df["broker_consecutive_days"] = df["broker_consecutive_days"].fillna(0)
            broker_conc_rank = df["broker_concentration"].rank(pct=True)
            broker_consec_rank = df["broker_consecutive_days"].rank(pct=True)
            broker_rank = broker_conc_rank * 0.60 + broker_consec_rank * 0.40

        # ── 隔日沖扣分 ──────────────────────────────────────────
        if has_broker and not df_broker_raw.empty and "broker_name" in df_broker_raw.columns:
            broker_rank, self._daytrade_penalty_df = self._apply_daytrade_penalty(
                broker_rank, df_broker_raw, stock_ids, df_price
            )
        else:
            self._daytrade_penalty_df = pd.DataFrame(
                {"stock_id": stock_ids, "daytrade_penalty": 0.0, "daytrade_tags": ""}
            )

        if has_margin and has_broker:
            # 5 因子：外資 25% + 量比 22% + 法人 22% + 券資比 16% + 分點 15%
            df["chip_score"] = (
                consec_rank * 0.25 + bvr_rank * 0.22 + total_rank * 0.22 + smr_rank * 0.16 + broker_rank * 0.15
            )
            chip_tier = "5F"
        elif has_margin:
            # 4 因子：外資 30% + 量比 25% + 法人 25% + 券資比 20%
            df["chip_score"] = consec_rank * 0.30 + bvr_rank * 0.25 + total_rank * 0.25 + smr_rank * 0.20
            chip_tier = "4F"
        elif has_broker:
            # 4 因子：外資 32% + 量比 24% + 法人 24% + 分點 20%
            df["chip_score"] = consec_rank * 0.32 + bvr_rank * 0.24 + total_rank * 0.24 + broker_rank * 0.20
            chip_tier = "4F"
        else:
            # 3 因子：外資 40% + 量比 30% + 法人 30%
            df["chip_score"] = consec_rank * 0.40 + bvr_rank * 0.30 + total_rank * 0.30
            chip_tier = "3F"

        df["chip_tier"] = chip_tier
        return df[["stock_id", "chip_score", "chip_tier"]]

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """高成長模式基本面：YoY 40% + 營收加速度 25% + 毛利率加速 20% + EPS 季增率 15%。

        毛利率加速 = 最新季毛利率 - 去年同季毛利率（年對年改善）。
        EPS 季增率 = (EPS 最新季 - EPS 上季) / abs(EPS 上季)。
        財報資料不足時自動降回營收雙因子（YoY 60% + 加速度 40%）。
        """
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        if rev.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
        if "yoy_3m_ago" in rev.columns:
            rev["acceleration"] = rev["yoy_growth"].fillna(0) - rev["yoy_3m_ago"].fillna(0)
            accel_rank = rev["acceleration"].rank(pct=True)
        else:
            accel_rank = pd.Series(0.5, index=rev.index)

        # --- 財報因子 ---
        df_fin = self._load_financial_data(stock_ids, quarters=5)
        if df_fin.empty:
            # 降回純營收雙因子
            rev["fundamental_score"] = yoy_rank * 0.60 + accel_rank * 0.40
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
            return result

        grouped = df_fin.groupby("stock_id", sort=False)
        fin_rows = []
        for sid in stock_ids:
            row: dict = {"stock_id": sid, "gm_accel": None, "eps_qoq": None}
            if sid in grouped.groups:
                grp = grouped.get_group(sid).sort_values("date", ascending=False)
                # 毛利率加速 = 最新季 - 去年同季
                if len(grp) >= 1 and pd.notna(grp.iloc[0]["gross_margin"]):
                    cur_q = int(grp.iloc[0]["quarter"])
                    cur_y = int(grp.iloc[0]["year"])
                    same_q = grp[(grp["quarter"] == cur_q) & (grp["year"] == cur_y - 1)]
                    if not same_q.empty and pd.notna(same_q.iloc[0]["gross_margin"]):
                        row["gm_accel"] = float(grp.iloc[0]["gross_margin"]) - float(same_q.iloc[0]["gross_margin"])
                # EPS 季增率 = (最新 - 上季) / abs(上季)
                if (
                    len(grp) >= 2
                    and pd.notna(grp.iloc[0]["eps"])
                    and pd.notna(grp.iloc[1]["eps"])
                    and abs(float(grp.iloc[1]["eps"])) > 0.01
                ):
                    row["eps_qoq"] = (float(grp.iloc[0]["eps"]) - float(grp.iloc[1]["eps"])) / abs(
                        float(grp.iloc[1]["eps"])
                    )
            fin_rows.append(row)

        df_metrics = pd.DataFrame(fin_rows)
        has_any = df_metrics[["gm_accel", "eps_qoq"]].notna().any(axis=1).any()
        if not has_any:
            rev["fundamental_score"] = yoy_rank * 0.60 + accel_rank * 0.40
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
            return result

        gm_accel_rank = df_metrics["gm_accel"].rank(pct=True).fillna(0.5)
        eps_qoq_rank = df_metrics["eps_qoq"].rank(pct=True).fillna(0.5)

        # 將財報指標合進 rev（以 stock_id 對齊）
        df_metrics = df_metrics.merge(
            rev[["stock_id", "yoy_growth"] + (["acceleration"] if "acceleration" in rev.columns else [])],
            on="stock_id",
            how="left",
        )
        df_metrics["yoy_growth"] = df_metrics["yoy_growth"].fillna(0)
        df_metrics["yoy_rank_val"] = df_metrics["yoy_growth"].rank(pct=True)
        if "acceleration" in df_metrics.columns:
            df_metrics["accel_rank_val"] = df_metrics["acceleration"].rank(pct=True).fillna(0.5)
        else:
            df_metrics["accel_rank_val"] = 0.5

        df_metrics["fundamental_score"] = (
            df_metrics["yoy_rank_val"] * 0.40
            + df_metrics["accel_rank_val"] * 0.25
            + gm_accel_rank * 0.20
            + eps_qoq_rank * 0.15
        )

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(df_metrics[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """高成長模式風險過濾：ATR(14)/close > 80th percentile 剔除。"""
        return self._apply_atr_risk_filter(scored, df_price, percentile=80)
