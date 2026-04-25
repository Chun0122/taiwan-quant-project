"""ValueScanner — 價值修復掃描模式。

低估值 + 基本面轉佳，適合中長期投資。
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
    StockInfo,
    StockValuation,
)
from src.discovery.scanner._base import MarketScanner
from src.discovery.scanner._functions import (
    DiscoveryResult,
    compute_broker_score,
    compute_earnings_quality,
    compute_relative_pe_thresholds,
)
from src.discovery.universe import UniverseConfig

logger = logging.getLogger(__name__)


class ValueScanner(MarketScanner):
    """價值修復掃描器。

    適合低估值 + 基本面轉佳 + 法人開始布局的「價值修復股」。
    粗篩：PE > 0 且 PE < 30 + 殖利率 > 2%
    細評：基本面 50% + 估值面 30% + 籌碼面 20%
    風險過濾：近 20 日波動率 > 90th percentile 剔除
    """

    mode_name = "value"
    _COARSE_WEIGHTS: dict[str, float] = {"vol_rank": 0.50, "inst_rank": 0.50}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 130)  # 支援 SMA120 + 120日低點計算
        kwargs.setdefault(
            "universe_config", UniverseConfig(trend_ma=None, volume_ratio_min=None, min_available_days=60)
        )
        super().__init__(**kwargs)

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """價值模式技術面：採用打底/超賣反轉風格（120日低點距離 + RSI超賣 + 三均線糾結度）。"""
        return self._compute_value_style_technical_scores(stock_ids, df_price)

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：value 模式額外載入估值資料 + 2 個月營收。含 UniverseFilter Stage 0.5。

        項目 B：若 `self._shared` 已由 `run(shared=...)` 注入，則以 in-memory 過濾
        取代 DB 查詢 4 張共用表；估值與 stock_info 仍走 DB 查詢（未在 shared 中）。
        """
        universe_ids = self._get_universe_ids()
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        shared = getattr(self, "_shared", None)
        if shared is not None:
            df_price, df_inst, df_margin, df_revenue = self._slice_shared_market_data(
                shared, universe_ids, cutoff, revenue_months=2
            )
            # 估值 + stock_info 不在 shared 中，仍走 DB
            with get_session() as session:
                val_query = select(
                    StockValuation.stock_id,
                    StockValuation.date,
                    StockValuation.pe_ratio,
                    StockValuation.pb_ratio,
                    StockValuation.dividend_yield,
                ).where(StockValuation.date >= cutoff)
                if universe_ids:
                    val_query = val_query.where(StockValuation.stock_id.in_(universe_ids))
                rows = session.execute(val_query).all()
                self._df_valuation = pd.DataFrame(
                    rows,
                    columns=["stock_id", "date", "pe_ratio", "pb_ratio", "dividend_yield"],
                )

                info_query = select(StockInfo.stock_id, StockInfo.industry_category)
                if universe_ids:
                    info_query = info_query.where(StockInfo.stock_id.in_(universe_ids))
                info_rows = session.execute(info_query).all()
                df_info = pd.DataFrame(info_rows, columns=["stock_id", "industry_category"])
                df_info["industry_category"] = df_info["industry_category"].fillna("未分類")
                self._df_stock_info = df_info
            return df_price, df_inst, df_margin, df_revenue

        with get_session() as session:
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

            # 估值資料
            val_query = select(
                StockValuation.stock_id,
                StockValuation.date,
                StockValuation.pe_ratio,
                StockValuation.pb_ratio,
                StockValuation.dividend_yield,
            ).where(StockValuation.date >= cutoff)
            if universe_ids:
                val_query = val_query.where(StockValuation.stock_id.in_(universe_ids))
            rows = session.execute(val_query).all()
            self._df_valuation = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "pe_ratio", "pb_ratio", "dividend_yield"],
            )

            # 載入產業分類（供 _coarse_filter 相對 PE 計算）
            info_query = select(StockInfo.stock_id, StockInfo.industry_category)
            if universe_ids:
                info_query = info_query.where(StockInfo.stock_id.in_(universe_ids))
            info_rows = session.execute(info_query).all()
            df_info = pd.DataFrame(info_rows, columns=["stock_id", "industry_category"])
            df_info["industry_category"] = df_info["industry_category"].fillna("未分類")
            self._df_stock_info = df_info

        # 載入 2 個月營收（含上月，算加速度）
        df_revenue = self._load_revenue_data(stock_ids=universe_ids if universe_ids else None, months=2)

        return df_price, df_inst, df_margin, df_revenue

    def run(self, shared=None, precomputed_ic=None) -> DiscoveryResult:
        """覆寫 run()：在 Stage 0.5 自動補抓估值、Stage 2.5 補抓候選股估值。

        Args:
            shared: 項目 B — 由 `_cmd_discover_all` 預載入的全市場資料；
                `_load_market_data` 會優先以此過濾產生 4 張共用 DataFrame。
            precomputed_ic: 項目 E — Step 8c 預算的 static IC DataFrame。
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

        # Stage 0.5: 估值資料覆蓋不足時，自動從 TWSE/TPEX 補抓全市場估值
        self._maybe_sync_valuation()

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
            # 相對估值 PE：同產業中位數 × 1.5（樣本不足 3 支時 fallback PE < 50）
            df_info = getattr(self, "_df_stock_info", pd.DataFrame())
            if not df_info.empty:
                info_map = df_info.set_index("stock_id")["industry_category"]
                industry_cat = filtered["stock_id"].map(info_map).fillna("未分類")
            else:
                industry_cat = pd.Series("未分類", index=filtered.index)
            pe_thresholds = compute_relative_pe_thresholds(industry_cat, filtered["pe_ratio"])
            pe_ok = (filtered["pe_ratio"] > 0) & (filtered["pe_ratio"] < pe_thresholds.values)
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

        # 2) 法人 5 日累積淨買超排名
        filtered = self._compute_inst_net_buy(df_inst, filtered, days=5)

        # 3) 短期動能（5 日報酬，用於 _rank_and_enrich 保留，不參與 _COARSE_WEIGHTS）
        filtered = self._compute_momentum_5d(df_price, filtered)

        return self._finalize_coarse(filtered)

    def _compute_extra_scores(self, stock_ids: list[str]) -> list[pd.DataFrame]:
        """價值模式額外維度：估值面分數。"""
        return [self._compute_valuation_scores(stock_ids)]

    def _post_score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """用 technical_score 欄位存估值分數（供 _rank_and_enrich 顯示用）。"""
        if "valuation_score" in candidates.columns:
            candidates["technical_score"] = candidates["valuation_score"]
        return candidates

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """價值模式基本面：營收 40% + ROE 25% + 毛利率 QoQ 20% + EPS YoY 15%。

        財報資料不足時自動降回營收單因子（YoY 70% + MoM 30%）。
        """
        # --- 營收基礎分（與 base class 相同）---
        if not df_revenue.empty:
            rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        else:
            rev = pd.DataFrame()

        if not rev.empty:
            yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
            mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)
            rev["rev_base"] = yoy_rank * 0.70 + mom_rank * 0.30
        else:
            rev = pd.DataFrame({"stock_id": stock_ids, "rev_base": [0.5] * len(stock_ids)})

        # --- 財報因子 ---
        df_fin = self._load_financial_data(stock_ids, quarters=5)
        if df_fin.empty:
            # 降回純營收分
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
            result["fundamental_score"] = result["rev_base"].fillna(0.5)
            return result[["stock_id", "fundamental_score"]]

        # 計算每支股票財報指標
        grouped = df_fin.groupby("stock_id", sort=False)
        fin_rows = []
        for sid in stock_ids:
            row: dict = {"stock_id": sid, "roe_val": None, "gm_qoq": None, "eps_yoy": None}
            if sid in grouped.groups:
                grp = grouped.get_group(sid).sort_values("date", ascending=False)
                # ROE：最新一季
                if len(grp) >= 1 and pd.notna(grp.iloc[0]["roe"]):
                    row["roe_val"] = float(grp.iloc[0]["roe"])
                # 毛利率 QoQ：最新季 - 上一季
                if len(grp) >= 2 and pd.notna(grp.iloc[0]["gross_margin"]) and pd.notna(grp.iloc[1]["gross_margin"]):
                    row["gm_qoq"] = float(grp.iloc[0]["gross_margin"]) - float(grp.iloc[1]["gross_margin"])
                # EPS YoY：最新季 vs 去年同季
                if len(grp) >= 1 and pd.notna(grp.iloc[0]["eps"]):
                    cur_q = int(grp.iloc[0]["quarter"])
                    cur_y = int(grp.iloc[0]["year"])
                    same_q = grp[(grp["quarter"] == cur_q) & (grp["year"] == cur_y - 1)]
                    if not same_q.empty and pd.notna(same_q.iloc[0]["eps"]):
                        prev_eps = float(same_q.iloc[0]["eps"])
                        if abs(prev_eps) > 0.01:
                            row["eps_yoy"] = (float(grp.iloc[0]["eps"]) - prev_eps) / abs(prev_eps)
            fin_rows.append(row)

        df_metrics = pd.DataFrame(fin_rows)
        has_any = df_metrics[["roe_val", "gm_qoq", "eps_yoy"]].notna().any(axis=1).any()
        if not has_any:
            # 財報欄位全 NULL → 降回純營收分
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
            result["fundamental_score"] = result["rev_base"].fillna(0.5)
            return result[["stock_id", "fundamental_score"]]

        # 排名百分位（用 min_count=1 避免全 NaN 時 rank 失敗）
        roe_rank = df_metrics["roe_val"].rank(pct=True).fillna(0.5)
        gm_qoq_rank = df_metrics["gm_qoq"].rank(pct=True).fillna(0.5)
        eps_yoy_rank = df_metrics["eps_yoy"].rank(pct=True).fillna(0.5)

        # ── 盈餘品質因子（C1：現金流品質 + 收入品質 + 負債穩定性）──
        eq_df = compute_earnings_quality(df_fin, stock_ids)
        df_metrics = df_metrics.merge(eq_df, on="stock_id", how="left")
        df_metrics["earnings_quality"] = df_metrics["earnings_quality"].fillna(0.5)
        eq_rank = df_metrics["earnings_quality"].rank(pct=True).fillna(0.5)

        # 合併營收基礎分
        df_metrics = df_metrics.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
        df_metrics["rev_base"] = df_metrics["rev_base"].fillna(0.5)

        # 加權：營收 35% + ROE 20% + 毛利率 QoQ 15% + EPS YoY 15% + 盈餘品質 15%
        df_metrics["fundamental_score"] = (
            df_metrics["rev_base"] * 0.35 + roe_rank * 0.20 + gm_qoq_rank * 0.15 + eps_yoy_rank * 0.15 + eq_rank * 0.15
        )

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(df_metrics[["stock_id", "fundamental_score"]], on="stock_id", how="left")
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

        # PE 反向排名：PE 越低分數越高；缺值者給中間分（0.5），不因缺值受益
        pe_rank = val["pe_ratio"].rank(pct=True, ascending=False, na_option="keep").fillna(0.5)
        # PB 反向排名：PB 越低分數越高；缺值者給中間分
        pb_rank = val["pb_ratio"].rank(pct=True, ascending=False, na_option="keep").fillna(0.5)
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
        """價值模式籌碼面：投信近期買超 + 三大法人累積 + 分點集中度（有資料時）。

        權重組合：
        - 3 因子（含分點）: 投信 40% + 累積 40% + 分點 20%
        - 2 因子（基本）:   投信 50% + 累積 50%

        回傳欄位：stock_id, chip_score, chip_tier（"3F" 或 "2F"）
        """
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids), "chip_tier": "N/A"})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
        dates = sorted(df_inst["date"].unique())
        recent_20_dates = set(dates[-20:] if len(dates) >= 20 else dates)
        inst_grouped = inst_filtered.groupby("stock_id", sort=False)

        rows = []
        for sid in stock_ids:
            if sid not in inst_grouped.groups:
                rows.append({"stock_id": sid, "trust_net": 0, "cum_net": 0})
                continue

            stock_inst = inst_grouped.get_group(sid)
            trust_data = stock_inst[stock_inst["name"].str.contains("投信", na=False)]
            trust_net = trust_data["net"].sum() if not trust_data.empty else 0

            recent_inst = stock_inst[stock_inst["date"].isin(recent_20_dates)]
            cum_net = recent_inst["net"].sum()

            rows.append({"stock_id": sid, "trust_net": trust_net, "cum_net": cum_net})

        df = pd.DataFrame(rows)
        trust_rank = df["trust_net"].rank(pct=True)
        cum_rank = df["cum_net"].rank(pct=True)

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

            # ── 隔日沖扣分 ──────────────────────────────────────
            if "broker_name" in df_broker_raw.columns:
                broker_rank, self._daytrade_penalty_df = self._apply_daytrade_penalty(
                    broker_rank, df_broker_raw, stock_ids, df_price
                )
            else:
                self._daytrade_penalty_df = pd.DataFrame(
                    {"stock_id": stock_ids, "daytrade_penalty": 0.0, "daytrade_tags": ""}
                )

            # 3 因子：投信 40% + 累積 40% + 分點 20%
            df["chip_score"] = trust_rank * 0.40 + cum_rank * 0.40 + broker_rank * 0.20
            chip_tier = "3F"
        else:
            # 2 因子：投信 50% + 累積 50%
            df["chip_score"] = trust_rank * 0.50 + cum_rank * 0.50
            chip_tier = "2F"
            self._daytrade_penalty_df = pd.DataFrame(
                {"stock_id": stock_ids, "daytrade_penalty": 0.0, "daytrade_tags": ""}
            )

        df["chip_tier"] = chip_tier
        return df[["stock_id", "chip_score", "chip_tier"]]

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """價值模式風險過濾：近 20 日波動率 > 90th percentile 剔除。"""
        return self._apply_vol_risk_filter(scored, df_price, percentile=90)
