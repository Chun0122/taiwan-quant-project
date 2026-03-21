"""DividendScanner — 高息存股掃描模式。

高殖利率 + 配息穩定 + 估值合理，適合長期存股。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import (
    DailyPrice,
    FinancialStatement,
    InstitutionalInvestor,
    MarginTrading,
    StockValuation,
)
from src.discovery.scanner._base import MarketScanner
from src.discovery.scanner._functions import (
    DiscoveryResult,
    compute_eps_sustainability,
)
from src.discovery.universe import UniverseConfig

logger = logging.getLogger(__name__)


class DividendScanner(MarketScanner):
    """高息存股掃描器。

    篩選高殖利率、配息穩定、估值合理的存股標的。
    粗篩：殖利率 > 3% + PE > 0
    細評：基本面 + 殖利率/估值 + 籌碼面 + 消息面（依 Regime 動態加權）
    風險過濾：近 20 日波動率 > 90th percentile 剔除
    """

    mode_name = "dividend"
    _COARSE_WEIGHTS: dict[str, float] = {"dy_rank": 0.50, "vol_rank": 0.30, "inst_rank": 0.20}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 130)  # 支援 SMA120 + 均線糾結度計算
        kwargs.setdefault(
            "universe_config", UniverseConfig(trend_ma=None, volume_ratio_min=None, min_available_days=60)
        )
        super().__init__(**kwargs)

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """高息模式技術面：採用穩健存股風格（SMA60斜率 + RSI中性帶 + 三均線糾結度）。"""
        return self._compute_dividend_style_technical_scores(stock_ids, df_price)

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：dividend 模式額外載入估值資料 + 2 個月營收。含 UniverseFilter Stage 0.5。"""
        universe_ids = self._get_universe_ids()
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

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

            # 載入近 4 季 EPS（供 _coarse_filter 配息連續性篩選）
            eps_cutoff = date.today() - timedelta(days=400)  # 4 季 ≈ 400 天
            eps_query = select(
                FinancialStatement.stock_id,
                FinancialStatement.date,
                FinancialStatement.eps,
            ).where(FinancialStatement.date >= eps_cutoff)
            if universe_ids:
                eps_query = eps_query.where(FinancialStatement.stock_id.in_(universe_ids))
            eps_rows = session.execute(eps_query).all()
            self._df_eps_quarterly = pd.DataFrame(eps_rows, columns=["stock_id", "date", "eps"])

        # 載入 2 個月營收（含上月，算加速度）
        df_revenue = self._load_revenue_data(stock_ids=universe_ids if universe_ids else None, months=2)

        return df_price, df_inst, df_margin, df_revenue

    def run(self) -> DiscoveryResult:
        """覆寫 run()：在 Stage 0.5 自動補抓估值、Stage 2.5 補抓候選股估值。"""
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

        # 配息連續性篩選：近 4 季 EPS 皆 > 0（無財報資料者 pass through）
        df_eps = getattr(self, "_df_eps_quarterly", pd.DataFrame())
        eps_fail_ids = compute_eps_sustainability(df_eps, min_quarters=4)
        if eps_fail_ids:
            before_count = len(filtered)
            filtered = filtered[~filtered["stock_id"].isin(eps_fail_ids)].copy()
            removed = before_count - len(filtered)
            if removed > 0:
                logger.info("Stage 2 EPS 連續性: 排除 %d 支近 4 季有負 EPS 股票", removed)

        if filtered.empty:
            return pd.DataFrame()

        # 粗篩分數：殖利率排名 50% + 成交量排名 30% + 法人排名 20%
        filtered["dy_rank"] = filtered["dividend_yield"].rank(pct=True)
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 2) 法人 5 日累積淨買超排名
        filtered = self._compute_inst_net_buy(df_inst, filtered, days=5)

        # 3) 短期動能（5 日報酬，用於 _rank_and_enrich 保留，不參與 _COARSE_WEIGHTS）
        filtered = self._compute_momentum_5d(df_price, filtered)

        return self._finalize_coarse(filtered)

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
        # PE 反向排名：PE 越低分數越高；缺值者給中間分（0.5），不因缺值受益
        pe_rank = val["pe_ratio"].rank(pct=True, ascending=False, na_option="keep").fillna(0.5)
        # PB 反向排名：PB 越低分數越高；缺值者給中間分
        pb_rank = val["pb_ratio"].rank(pct=True, ascending=False, na_option="keep").fillna(0.5)

        val["dividend_score"] = dy_rank * 0.50 + pe_rank * 0.30 + pb_rank * 0.20

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(val[["stock_id", "dividend_score"]], on="stock_id", how="left")
        return result

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """高息模式基本面：營收 40% + EPS 穩定性 35% + 配息率代理 25%。

        EPS 穩定性 = 最近 4 季 EPS 標準差（越低越穩定，倒排）。
        配息率代理 = 4 季中 EPS > 0 的比例（能穩定獲利才能持續配息）。
        財報資料不足時自動降回營收單因子（YoY 70% + MoM 30%）。
        """
        # --- 營收基礎分 ---
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

        # --- 財報因子（需要最近 4 季 EPS）---
        df_fin = self._load_financial_data(stock_ids, quarters=5)
        if df_fin.empty:
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
            result["fundamental_score"] = result["rev_base"].fillna(0.5)
            return result[["stock_id", "fundamental_score"]]

        grouped = df_fin.groupby("stock_id", sort=False)
        fin_rows = []
        for sid in stock_ids:
            row: dict = {"stock_id": sid, "eps_std": None, "positive_eps_ratio": None}
            if sid in grouped.groups:
                grp = grouped.get_group(sid).sort_values("date", ascending=False).head(4)
                eps_vals = grp["eps"].dropna().tolist()
                if len(eps_vals) >= 2:
                    row["eps_std"] = float(pd.Series(eps_vals).std())
                    row["positive_eps_ratio"] = sum(1 for e in eps_vals if e > 0) / len(eps_vals)
                elif len(eps_vals) == 1:
                    row["eps_std"] = 0.0  # 只有 1 季，視為完全穩定
                    row["positive_eps_ratio"] = 1.0 if eps_vals[0] > 0 else 0.0
            fin_rows.append(row)

        df_metrics = pd.DataFrame(fin_rows)
        has_any = df_metrics[["eps_std", "positive_eps_ratio"]].notna().any(axis=1).any()
        if not has_any:
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
            result["fundamental_score"] = result["rev_base"].fillna(0.5)
            return result[["stock_id", "fundamental_score"]]

        # EPS 穩定性：std 越低越好 → ascending=False（反向排名）
        eps_stability_rank = df_metrics["eps_std"].rank(pct=True, ascending=False).fillna(0.5)
        # 配息率代理：正 EPS 比例越高越好
        payout_proxy_rank = df_metrics["positive_eps_ratio"].rank(pct=True).fillna(0.5)

        df_metrics = df_metrics.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
        df_metrics["rev_base"] = df_metrics["rev_base"].fillna(0.5)

        # 加權：營收 40% + EPS 穩定性 35% + 配息率代理 25%
        df_metrics["fundamental_score"] = (
            df_metrics["rev_base"] * 0.40 + eps_stability_rank * 0.35 + payout_proxy_rank * 0.25
        )

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(df_metrics[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """高息模式籌碼面 2 因子：投信淨買超 50% + 三大法人累積買超 50%。

        回傳欄位：stock_id, chip_score, chip_tier（固定 "2F"）
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
        df["chip_score"] = trust_rank * 0.50 + cum_rank * 0.50
        df["chip_tier"] = "2F"

        return df[["stock_id", "chip_score", "chip_tier"]]

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """高息模式風險過濾：近 20 日波動率 > 75th percentile 剔除。

        高股息策略具防禦性質，應收緊波動度門檻（而非放寬），
        避免「股價暴跌導致殖利率飆高」的價值陷阱。
        """
        return self._apply_vol_risk_filter(scored, df_price, percentile=75)
