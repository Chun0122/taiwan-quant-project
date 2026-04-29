"""全市場選股掃描器 — 基底類別 MarketScanner。

提供四階段漏斗掃描框架，子類透過覆寫 hook 方法實作模式專屬邏輯。
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.constants import DISCOVERY_KEY_FACTOR_MAP
from src.data.database import get_session
from src.data.schema import (
    Announcement,
    BrokerTrade,
    DailyPrice,
    DiscoveryRecord,
    FinancialStatement,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
    SecuritiesLending,
    StockInfo,
    StockValuation,
)
from src.discovery.scanner._functions import (
    IC_DAMPEN_WEIGHT_MULT,
    MIN_SCORE_THRESHOLDS,
    SECTOR_MAX_RATIO,
    SECTOR_MIN_CAP,
    DiscoveryResult,
    ScanAuditTrail,
    _calc_atr14,
    compute_abnormal_announcement_rate,
    compute_adaptive_atr_multiplier,
    compute_chip_macd,
    compute_daytrade_penalty,
    compute_factor_ic,
    compute_ic_aware_score_transform,
    compute_ic_impact_weight_adjustments,
    compute_institutional_acceleration,
    compute_key_player_cost_basis,
    compute_momentum_decay,
    compute_multi_timeframe_alignment,
    compute_news_decay_weight,
    compute_peer_fundamental_ranking,
    compute_taiex_relative_strength,
    compute_volume_price_divergence,
    compute_win_rate_threshold_adjustment,
    detect_chip_tier_changes,
    score_key_player_cost,
)
from src.discovery.scanner._shared_load import SharedMarketData, slice_revenue_raw
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
    _blocked_regimes: set[str] = set()  # 子類可覆寫以阻擋特定市場狀態

    def __init__(
        self,
        min_price: float = 10,
        max_price: float = 2000,
        min_volume: int = 500_000,
        top_n_candidates: int = 150,
        top_n_results: int = 30,
        lookback_days: int = 5,
        weekly_confirm: bool = False,
        use_ic_adjustment: bool = True,
        universe_config: UniverseConfig | None = None,
    ) -> None:
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
        self.top_n_candidates = top_n_candidates
        self.top_n_results = top_n_results
        self.lookback_days = lookback_days
        self.weekly_confirm = weekly_confirm
        self.use_ic_adjustment = use_ic_adjustment
        # 子因子 rank 收集器（供 IC 診斷使用）
        self._sub_factor_ranks: dict[str, pd.DataFrame] = {}
        # 維度 IC（E2b 設置、_score_candidates 消費做 IC-aware 分數翻轉）
        self._dimension_ic_df: pd.DataFrame | None = None
        # IC-aware 分數轉換動作（_score_candidates 設置，run() 寫入 DiscoveryResult）
        self._ic_actions: dict[str, str] = {}
        # Universe Filter：各子類可在 __init__ 中傳入模式專屬 config
        self._universe_config = universe_config or UniverseConfig()
        self._universe_filter = UniverseFilter(self._universe_config)

    def _is_regime_blocked(self) -> bool:
        """Stage 0.1 regime gate 判定 — 合併兩個來源。

        判定規則（OR 合併）：
          1. 子類 _blocked_regimes 集合（類級自訂）
          2. 全域 REGIME_MODE_BLOCK 矩陣（constants.py 集中管理）

        需在 self.regime 已設置後呼叫。子類覆寫 run() 時應主動呼叫此方法。
        """
        from src.constants import REGIME_MODE_BLOCK as _RMB

        if not getattr(self, "regime", None):
            return False
        if self.regime in self._blocked_regimes:
            return True
        return self.mode_name in _RMB.get(self.regime, frozenset())

    def run(
        self,
        shared: SharedMarketData | None = None,
        precomputed_ic: pd.DataFrame | None = None,
    ) -> DiscoveryResult:
        """執行四階段漏斗掃描。

        流程清楚分為三大區塊：
        1. 資料準備（Stage 0~2.7）
        2. 評分 + 軟加成（Stage 3~3.6）— 調整 composite_score，不剔除
        3. 硬風控（Stage 3.5/3.5b/3.5e/3.7/4.1/4.2）— 通過或剔除

        Args:
            shared: 由 `_cmd_discover_all` 預載入的全市場資料（項目 B）；
                傳入時 `_load_market_data` 以 in-memory 過濾取代 DB 查詢，
                省下 4 次重複 I/O。未傳入時維持原行為（各 scanner 自行查 DB）。
            precomputed_ic: 由 morning-routine Step 8c 預算的 static IC DataFrame
                （項目 E）；傳入時 `_apply_ic_weight_adjustment` 與
                `_log_factor_effectiveness` 直接使用，跳過 DB 查詢 + 重算 IC。
                未傳入時維持原 DB 路徑。
        """
        self._shared = shared
        self._precomputed_ic = precomputed_ic
        self.scan_date = date.today()
        audit = ScanAuditTrail()
        self._audit_trail = audit

        # ============================================================ #
        #  區塊 1: 資料準備（Stage 0 ~ 2.7）
        # ============================================================ #

        # Stage 0: 偵測市場狀態（Regime）
        try:
            from src.regime.detector import MarketRegimeDetector

            regime_info = MarketRegimeDetector().detect()
            self.regime = regime_info["regime"]
            logger.info("Stage 0: 市場狀態 = %s (TAIEX=%.0f)", self.regime, regime_info["taiex_close"])
        except Exception:
            self.regime = "sideways"
            logger.warning("Stage 0: 市場狀態偵測失敗，預設 sideways")

        # Stage 0.1: Regime gate — 特定模式在指定 regime 不執行
        if self._is_regime_blocked():
            logger.warning(
                "Stage 0.1: %s 模式在 %s 市場暫停掃描（歷史績效不佳）",
                self.mode_name,
                self.regime,
            )
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=0,
                after_coarse=0,
                mode=self.mode_name,
                audit_trail=audit,
            )

        # Stage 1: 載入資料
        df_price, df_inst, df_margin, df_revenue = self._load_market_data()
        if df_price.empty:
            logger.warning("無市場資料可供掃描")
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=0,
                after_coarse=0,
                mode=self.mode_name,
                audit_trail=audit,
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
                audit_trail=audit,
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

        # ============================================================ #
        #  區塊 2: 評分 + 軟加成（Score Adjustments — 只調分不剔除）
        # ============================================================ #

        # Stage 3: 四維度加權評分
        scored = self._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue, df_ann, df_ann_history)
        logger.info("Stage 3: 完成 %d 支候選股評分", len(scored))

        # --- 軟加成：前次掃描重疊（降低換手率）---
        scored = self._apply_overlap_bonus(scored)
        if "overlap_bonus" in scored.columns:
            audit.record_score_adjustments_from_column("3.2 前次重疊", scored, "overlap_bonus", "前次掃描重疊加成")

        # --- 軟加成：產業 / 概念 / 同業 ---
        scored = self._apply_sector_bonus(scored)
        audit.record_score_adjustments_from_column("3.3 產業輪動", scored, "sector_bonus", "產業輪動加成")

        scored = self._apply_sector_relative_strength(scored)
        if "relative_strength_bonus" in scored.columns:
            audit.record_score_adjustments_from_column(
                "3.3a 同業強度", scored, "relative_strength_bonus", "同業相對強度"
            )

        scored = self._apply_concept_bonus(scored)
        if "concept_bonus" in scored.columns:
            audit.record_score_adjustments_from_column("3.3b 概念熱度", scored, "concept_bonus", "概念股輪動加成")

        scored = self._apply_peer_fundamental_ranking(scored)
        if "peer_rank_bonus" in scored.columns:
            audit.record_score_adjustments_from_column("3.3c 同業基本面", scored, "peer_rank_bonus", "同業基本面排名")

        # --- 軟加成：週線趨勢 ---
        if self.weekly_confirm:
            scored = self._apply_weekly_trend_bonus(scored)
            if "weekly_bonus" in scored.columns:
                audit.record_score_adjustments_from_column("3.4 週線確認", scored, "weekly_bonus", "週線多時框確認")

        # --- 軟加成：動量衰減 ---
        scored = self._apply_momentum_decay(scored, df_price)
        if "momentum_decay" in scored.columns:
            audit.record_score_adjustments_from_column("3.5c 動量衰減", scored, "momentum_decay", "RSI背離/MACD柱縮")

        # --- 軟加成：籌碼加速度 ---
        scored = self._apply_institutional_acceleration(scored, df_inst)
        if "inst_accel_bonus" in scored.columns:
            audit.record_score_adjustments_from_column("3.5d 籌碼加速", scored, "inst_accel_bonus", "法人買超加速")

        # --- 軟加成：籌碼 MACD ---
        scored = self._apply_chip_macd(scored, df_inst)
        if "chip_macd_adj" in scored.columns:
            audit.record_score_adjustments_from_column("3.5f 籌碼MACD", scored, "chip_macd_adj", "法人淨買超MACD")

        # --- 軟加成：主力成本 ---
        scored = self._apply_key_player_cost(scored, df_price)
        if "kp_adj" in scored.columns:
            audit.record_score_adjustments_from_column("3.5g 主力成本", scored, "kp_adj", "現價vs主力成本")

        # --- 軟加成：消息面負面閘門（過濾壞消息股）---
        scored = self._apply_negative_news_gate(scored)
        if "neg_news_gate" in scored.columns:
            audit.record_score_adjustments_from_column("3.5h 負面消息閘門", scored, "neg_news_gate", "高負面消息股降分")

        # --- 軟加成：量價背離 ---
        scored = self._apply_volume_price_divergence(scored, df_price)
        if "vp_divergence" in scored.columns:
            audit.record_score_adjustments_from_column("3.6 量價背離", scored, "vp_divergence", "量價背離調整")

        # ============================================================ #
        #  區塊 3: 硬風控（Hard Filters — 通過或剔除）
        # ============================================================ #

        # 硬風控 1: 風險過濾（ATR/波動率百分位）
        ids_before = set(scored["stock_id"])
        scored = self._apply_risk_filter(scored, df_price)
        audit.record_hard_filter("3.5 風險過濾", ids_before, set(scored["stock_id"]), "ATR/波動率超過百分位門檻")

        # 硬風控 2: Crisis 相對強度 + 絕對趨勢
        ids_before = set(scored["stock_id"])
        scored = self._apply_crisis_filter(scored, df_price)
        audit.record_hard_filter("3.5b Crisis過濾", ids_before, set(scored["stock_id"]), "跑輸TAIEX或跌破MA60")

        # 硬風控 3: 多時框強制排除（momentum/growth 日多週空）
        ids_before = set(scored["stock_id"])
        scored = self._apply_multi_timeframe_alignment(scored)
        audit.record_hard_filter("3.5e 多時框共振", ids_before, set(scored["stock_id"]), "日線多頭+週線空頭矛盾")
        if "mtf_alignment" in scored.columns:
            # 未被排除的仍有調分（±4%），記錄到軟加成
            audit.record_score_adjustments_from_column("3.5e 多時框共振", scored, "mtf_alignment", "日週一致性調分")

        # 硬風控 4: 動態評分門檻（Regime + 勝率回饋）
        ids_before = set(scored["stock_id"])
        scored = self._apply_score_threshold(scored)
        audit.record_hard_filter("3.7 分數門檻", ids_before, set(scored["stock_id"]), "composite_score低於Regime門檻")

        # E2: 因子有效性日誌（不影響評分，僅記錄 IC 供參考）
        self._log_factor_effectiveness()

        # ============================================================ #
        #  區塊 4: 排名 + 結構化輸出
        # ============================================================ #

        # Stage 4: 排名 + 產業標籤
        rankings = self._rank_and_enrich(scored)

        # 硬風控 5: 同產業分散化（區分「因產業上限剔除」與「Top-N*2 pool 外截斷」）
        ids_before = set(rankings["stock_id"])
        rankings, sector_capped_ids, pool_ids = self._apply_sector_diversification(rankings)
        ids_after = set(rankings["stock_id"])
        # (a) 僅記錄真正因產業上限而被剔除的股票
        audit.record_hard_filter(
            "4.1 產業分散",
            sector_capped_ids | ids_after,
            ids_after,
            "同產業超過25%上限",
        )
        # (b) 另記錄 Top-N*2 pool 外、排名過低未納入考量的截斷（避免與產業分散混淆）
        pool_truncated = ids_before - pool_ids
        if pool_truncated:
            audit.record_hard_filter(
                "4.1b Top-N*2 截斷",
                ids_before,
                pool_ids,
                "排名超出分散化考量範圍（top_n×2）",
            )

        # 硬風控 6: 回撤降頻
        effective_top_n = self._compute_drawdown_adjusted_top_n(df_price)
        sector_summary = self._compute_sector_summary(rankings)
        logger.info("Stage 4: 輸出 Top %d（原始 Top %d）", min(effective_top_n, len(rankings)), self.top_n_results)

        # Stage 4.3: 籌碼層級降級稽核（比對前次掃描 chip_tier，記錄升降級）
        rankings = self._audit_chip_tier_changes(rankings)

        # 記錄被 top_n 截斷的股票
        final_rankings = rankings.head(effective_top_n)
        if len(rankings) > effective_top_n:
            truncated = set(rankings["stock_id"]) - set(final_rankings["stock_id"])
            audit.record_hard_filter(
                "4.2 Drawdown縮表", set(rankings["stock_id"]), set(final_rankings["stock_id"]), "TAIEX回撤降頻截斷"
            )

        # 輸出審計摘要到日誌
        summary = audit.summary()
        if summary["total_hard_filters"] > 0 or summary["total_score_adjustments"] > 0:
            logger.info(
                "審計摘要: 硬風控剔除 %d 支, 軟加成影響 %d 筆",
                summary["total_hard_filters"],
                summary["total_score_adjustments"],
            )

        # 收集子因子 rank（供因子診斷使用）
        sub_factors = self.get_sub_factor_df()

        return DiscoveryResult(
            rankings=final_rankings,
            total_stocks=total_stocks,
            after_coarse=after_coarse,
            sector_summary=sector_summary,
            mode=self.mode_name,
            audit_trail=audit,
            sub_factor_df=sub_factors if not sub_factors.empty else None,
            ic_actions=dict(self._ic_actions),
        )

    # ------------------------------------------------------------------ #
    #  Stage 1: 載入資料
    # ------------------------------------------------------------------ #

    def _get_universe_ids(self) -> list[str]:
        """執行 UniverseFilter 三層過濾，回傳候選 stock_id 清單。

        供 _load_market_data() 及子類覆寫版本呼叫，以 IN 子句限定 SQL 查詢範圍。
        若 UniverseFilter 失敗（DB 空等原因）回傳空清單，呼叫端的 SQL 不加 IN 子句。
        """
        # 將 Regime 傳入 UniverseFilter，使流動性/趨勢門檻自適應市場狀態
        self._universe_config.regime = getattr(self, "regime", None)
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

        項目 B 增強：若 `self._shared` 已由 `run(shared=...)` 注入，則以 in-memory 過濾
        取代 DB 查詢，避免 5 個 scanner 重複相同的全市場讀取。
        """
        # Stage 0.5: Universe Filter — SQL 硬過濾 + 流動性 + 趨勢
        universe_ids = self._get_universe_ids()

        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        # 項目 B：共用資料注入路徑（避免重複 DB 讀取）
        shared = getattr(self, "_shared", None)
        if shared is not None:
            return self._slice_shared_market_data(shared, universe_ids, cutoff)

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

    def _slice_shared_market_data(
        self,
        shared: SharedMarketData,
        universe_ids: list[str],
        cutoff: date,
        *,
        revenue_months: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """項目 B：由 SharedMarketData 以 in-memory 過濾產生 scanner 所需的 4 張 DataFrame。

        語意與原 `_load_market_data` DB 路徑一致：
          - 以 `cutoff = today - (lookback_days + 10)` 過濾日期
          - 若有 `universe_ids` 則以 `.isin()` 過濾股票
          - revenue 依 `revenue_months`（未傳入時用 `self._revenue_months`）pivot

        共用資料不可 mutate；回傳前 `.copy()` 確保 scanner 可安全改欄位。

        Args:
            revenue_months: Swing 等覆寫 `_load_market_data` 的子類可在此顯式傳遞
                `months=2` 對齊原行為，不依賴 `_revenue_months` 類別屬性。
        """
        # 防呆：shared 的 price_cutoff 需早於或等於本 scanner 所需 cutoff
        if shared.price_cutoff > cutoff:
            logger.warning(
                "Shared price_cutoff=%s 晚於 scanner cutoff=%s，資料覆蓋不足；退回 DB 路徑",
                shared.price_cutoff,
                cutoff,
            )
            self._shared = None  # 避免下方遞迴
            try:
                return self._load_market_data()
            finally:
                self._shared = shared

        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            """按 date + universe 過濾共用 DF，並回傳 copy。"""
            if df.empty:
                return df.copy()
            mask = df["date"] >= cutoff
            if universe_ids:
                mask &= df["stock_id"].isin(universe_ids)
            return df.loc[mask].copy()

        df_price = _filter(shared.df_price)
        df_inst = _filter(shared.df_inst)
        df_margin = _filter(shared.df_margin)

        months = revenue_months if revenue_months is not None else self._revenue_months
        df_revenue = slice_revenue_raw(
            shared.df_revenue,
            stock_ids=universe_ids if universe_ids else None,
            months=months,
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
        """從 DB 查詢最近 N 季財務資料。

        Args:
            stock_ids: 限定查詢的股票清單
            quarters: 每支股票取最近幾季（預設 5 季，足以計算 YoY + QoQ）

        Returns:
            DataFrame(stock_id, date, year, quarter, eps, roe, gross_margin, debt_ratio,
                      revenue, net_income, operating_cf)
            每支股票最多 quarters 筆，按 date desc 排列。無資料時回傳空 DataFrame。
        """
        _cols = [
            "stock_id",
            "date",
            "year",
            "quarter",
            "eps",
            "roe",
            "gross_margin",
            "debt_ratio",
            "revenue",
            "net_income",
            "operating_cf",
            "free_cf",
        ]
        cutoff = date.today() - timedelta(days=quarters * 100)
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
                        FinancialStatement.revenue,
                        FinancialStatement.net_income,
                        FinancialStatement.operating_cf,
                        FinancialStatement.free_cf,
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
        days: int | None = None,
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
        from src.constants import NEWS_LOAD_WINDOW_DAYS

        if days is None:
            days = NEWS_LOAD_WINDOW_DAYS
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
        """計算消息面分數 — 拆分 catalyst / risk 雙通道（Phase E v2）。

        設計目標：原 net_score 混雜正反訊號造成 IC 結構性為負；
        改為分別衡量「正向催化」與「風險事件」，兩者獨立 rank 後合成。

        分類規則：
            catalyst：event_type ∈ NEWS_CATALYST_TYPES 且 sentiment != -1
                     （法說會/投資人日/買回/正面營收）
            risk：event_type ∈ NEWS_RISK_TYPES 或 sentiment == -1
                 （董監改選/filing 或任何負面事件）
            其他：無貢獻（避免「正面一般公告」噪音汙染）

        公式：
            weight = exp(-decay × days_ago) × type_weight × abnormal_multiplier
            catalyst_raw[sid] = Σ weight for catalyst events
            risk_raw[sid] = Σ weight for risk events
            catalyst_rank = percentile_rank（僅在有催化訊號的股票間排序，無訊號者 0.5）
            risk_rank = 同上
            news_score = NEWS_CATALYST_WEIGHT × catalyst_rank
                       + NEWS_RISK_WEIGHT × (1 - risk_rank)

        Returns:
            DataFrame(stock_id, news_score, news_catalyst_score, news_risk_score)
            — 三欄皆 0~1，0.5 為中性預設
        """
        from src.constants import (
            NEWS_CATALYST_TYPES,
            NEWS_CATALYST_WEIGHT,
            NEWS_RISK_TYPES,
            NEWS_RISK_WEIGHT,
        )

        default = pd.DataFrame(
            {
                "stock_id": stock_ids,
                "news_score": [0.5] * len(stock_ids),
                "news_catalyst_score": [0.5] * len(stock_ids),
                "news_risk_score": [0.5] * len(stock_ids),
            }
        )

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

        # 催化 / 風險分類
        is_risk = (ann["event_type"].isin(NEWS_RISK_TYPES)) | (ann["sentiment"] == -1)
        is_catalyst = (ann["event_type"].isin(NEWS_CATALYST_TYPES)) & (ann["sentiment"] != -1) & (~is_risk)

        catalyst_df = ann[is_catalyst].groupby("stock_id")["decay_weight"].sum().reset_index(name="catalyst_raw")
        risk_df = ann[is_risk].groupby("stock_id")["decay_weight"].sum().reset_index(name="risk_raw")

        df = pd.DataFrame({"stock_id": stock_ids})
        df = df.merge(catalyst_df, on="stock_id", how="left")
        df = df.merge(risk_df, on="stock_id", how="left")
        df["catalyst_raw"] = df["catalyst_raw"].fillna(0.0)
        df["risk_raw"] = df["risk_raw"].fillna(0.0)

        # 異常公告率乘數（僅在 history 有效時套用，同時放大兩邊）
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
            df["catalyst_raw"] *= df["mult"]
            df["risk_raw"] *= df["mult"]

        # 無訊號股票視為中性 0.5，只在有訊號的股票間做 rank
        df["news_catalyst_score"] = 0.5
        mask_c = df["catalyst_raw"] > 0
        if mask_c.sum() >= 2:
            df.loc[mask_c, "news_catalyst_score"] = df.loc[mask_c, "catalyst_raw"].rank(pct=True)
        elif mask_c.sum() == 1:
            # 單一催化訊號 → 給予正向 0.7（有優於無）
            df.loc[mask_c, "news_catalyst_score"] = 0.7

        df["news_risk_score"] = 0.5
        mask_r = df["risk_raw"] > 0
        if mask_r.sum() >= 2:
            df.loc[mask_r, "news_risk_score"] = df.loc[mask_r, "risk_raw"].rank(pct=True)
        elif mask_r.sum() == 1:
            df.loc[mask_r, "news_risk_score"] = 0.7

        # 合成 news_score：高 catalyst + 低 risk → 高分
        df["news_score"] = NEWS_CATALYST_WEIGHT * df["news_catalyst_score"] + NEWS_RISK_WEIGHT * (
            1.0 - df["news_risk_score"]
        )

        return df[["stock_id", "news_score", "news_catalyst_score", "news_risk_score"]]

    # ------------------------------------------------------------------ #
    #  Stage 3.2：前次掃描重疊加成（降低換手率）
    # ------------------------------------------------------------------ #

    def _apply_overlap_bonus(self, scored: pd.DataFrame, bonus: float = 0.03) -> pd.DataFrame:
        """Stage 3.2 — 與前次同模式推薦重疊的股票 composite_score ×(1+bonus)。

        目的：降低 scan-to-scan 換手率，鼓勵推薦持續性（邊際加成不主導排名）。
        做法：查詢 DB 前一次同模式 DiscoveryRecord.stock_id 集合，與本次候選取交集 → 加成。
        """
        if scored.empty:
            scored["overlap_bonus"] = 0.0
            return scored

        try:
            from sqlalchemy import func

            with get_session() as session:
                prev_date = session.execute(
                    select(func.max(DiscoveryRecord.scan_date)).where(
                        DiscoveryRecord.mode == self.mode_name,
                        DiscoveryRecord.scan_date < self.scan_date,
                    )
                ).scalar()
                if prev_date is None:
                    scored["overlap_bonus"] = 0.0
                    return scored

                prev_ids = set(
                    session.execute(
                        select(DiscoveryRecord.stock_id).where(
                            DiscoveryRecord.mode == self.mode_name,
                            DiscoveryRecord.scan_date == prev_date,
                        )
                    )
                    .scalars()
                    .all()
                )

            if not prev_ids:
                scored["overlap_bonus"] = 0.0
                return scored

            scored["overlap_bonus"] = scored["stock_id"].apply(lambda s: bonus if s in prev_ids else 0.0)
            scored["composite_score"] = scored["composite_score"] * (1 + scored["overlap_bonus"])
            n_overlap = (scored["overlap_bonus"] > 0).sum()
            logger.info(
                "Stage 3.2: 前次掃描重疊加成 — %d/%d 支重疊股 +%.1f%%",
                n_overlap,
                len(scored),
                bonus * 100,
            )
            return scored

        except Exception:
            logger.debug("Stage 3.2: 重疊加成失敗，跳過")
            scored["overlap_bonus"] = 0.0
            return scored

    # ------------------------------------------------------------------ #
    #  Stage 3.5h：消息面負面閘門（過濾壞消息股）
    # ------------------------------------------------------------------ #

    def _apply_negative_news_gate(
        self,
        scored: pd.DataFrame,
        threshold: float = 0.15,
        penalty: float = 0.08,
        percentile: float = 0.15,
        use_percentile: bool = True,
        min_sample: int = 20,
        abs_cutoff_safety: float = 0.30,
    ) -> pd.DataFrame:
        """Stage 3.5h — news_score 偏低的股票 composite_score ×(1-penalty)。

        理由：消融測試顯示 news 的價值來自「過濾壞消息股」，但作為連續排名 IC≈0。
        解法：保留連續評分，額外加硬性負面閘門，僅懲罰 bottom percentile 高負面消息股。

        v2 改為百分位門檻（commit ab53fb8 後 news_score 分布漂移，絕對門檻 0.15 命中率掉到 1）：
          - 預設 use_percentile=True：cutoff = news_score 分布的 percentile 分位數
          - 安全上限 abs_cutoff_safety：避免 bull regime 整體分布偏高時誤殺中性股
          - 樣本 < min_sample → fallback 絕對門檻（小樣本下百分位失真）
          - use_percentile=False：完全退化為 v1 絕對門檻行為

        Args:
            threshold: 絕對門檻 fallback（v1 行為）
            penalty: 懲罰幅度（composite_score × (1-penalty)）
            percentile: 百分位門檻（0~1，預設 0.15 = bottom 15%）
            use_percentile: True=百分位模式，False=絕對門檻
            min_sample: 啟用百分位的最低樣本數
            abs_cutoff_safety: 百分位 cutoff 上限（防 bull regime 誤殺）
        """
        if scored.empty or "news_score" not in scored.columns:
            scored["neg_news_gate"] = 0.0
            return scored

        news = scored["news_score"].dropna()
        if use_percentile and len(news) >= min_sample:
            cutoff = float(news.quantile(percentile))
            cutoff = min(cutoff, abs_cutoff_safety)
            gate_mode = f"p{int(percentile * 100)}={cutoff:.3f}"
        else:
            cutoff = threshold
            gate_mode = f"abs={cutoff:.3f}"

        mask = scored["news_score"] < cutoff
        scored["neg_news_gate"] = 0.0
        scored.loc[mask, "neg_news_gate"] = -penalty
        scored.loc[mask, "composite_score"] = scored.loc[mask, "composite_score"] * (1 - penalty)

        n_blocked = int(mask.sum())
        if n_blocked > 0:
            logger.info(
                "Stage 3.5h: 負面消息閘門 — %d 支股票（門檻 %s）降分 -%.1f%%",
                n_blocked,
                gate_mode,
                penalty * 100,
            )
        return scored

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
    #  Stage 3.3c: 同業基本面排名（C3）
    # ------------------------------------------------------------------ #

    def _apply_peer_fundamental_ranking(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Stage 3.3c — 同產業 ROE/毛利率/營收排名 → composite_score ±3%。

        產業龍頭（前 25%）加分，落後者（後 25%）減分。
        同業不足 4 家時不加減分。
        """
        if scored.empty or "industry_category" not in scored.columns:
            return scored

        stock_ids = scored["stock_id"].tolist()

        # 建立 industry_map
        industry_map: dict[str, str] = {}
        if "industry_category" in scored.columns:
            for _, row in scored[["stock_id", "industry_category"]].dropna().iterrows():
                industry_map[row["stock_id"]] = row["industry_category"]

        if not industry_map:
            return scored

        # 載入最新一季財報
        df_fin = self._load_financial_data(stock_ids, quarters=1)
        if df_fin.empty:
            return scored

        peer_df = compute_peer_fundamental_ranking(df_fin, stock_ids, industry_map, bonus=0.03)
        scored = scored.merge(peer_df, on="stock_id", how="left")
        scored["peer_rank_bonus"] = scored["peer_rank_bonus"].fillna(0.0)
        # peer_rank_bonus 受統一加成上限約束（與 sector/concept 合計 ≤ ±8%）
        sector = scored.get("sector_bonus", pd.Series(0.0, index=scored.index)).fillna(0.0)
        concept = scored.get("concept_bonus", pd.Series(0.0, index=scored.index)).fillna(0.0)
        used = sector + concept  # 已使用的加成空間
        remaining = (0.08 - used.abs()).clip(lower=0.0)
        scored["peer_rank_bonus"] = scored["peer_rank_bonus"].clip(lower=-remaining, upper=remaining)
        scored["composite_score"] = scored["composite_score"] * (1 + scored["peer_rank_bonus"])

        n_leader = (scored["peer_rank_bonus"] > 0).sum()
        n_laggard = (scored["peer_rank_bonus"] < 0).sum()
        if n_leader > 0 or n_laggard > 0:
            logger.info(
                "Stage 3.3c: 同業基本面排名 — %d 支龍頭加分 / %d 支落後減分",
                n_leader,
                n_laggard,
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

        # E2b: Factor IC 動態權重調整（資料充足時自動啟用）
        # ≥20 筆歷史推薦即自動校準；use_ic_adjustment=False 可關閉（測試用）
        if self.use_ic_adjustment:
            w = self._apply_ic_weight_adjustment(w, scored_candidates=candidates)

        # E2c: IC 感知分數翻轉（問題 3 修正）— 對 IC 反向的維度分數做 1 - score 翻轉
        # 與 _apply_ic_weight_adjustment 共用 self._dimension_ic_df，不另打 DB
        # IC_DAMPEN=1（環境變數）啟用降權模式：弱 IC 因子保留分數但 weight×0.25，
        # 取代既有的「歸 0.5 中性化」策略。詳見 IC_DAMPEN_WEIGHT_MULT 註解。
        dampen_mode = os.getenv("IC_DAMPEN", "0") == "1"
        weight_mults: dict[str, float] = {f"{k}_score": 1.0 for k in w}
        if self.use_ic_adjustment and self._dimension_ic_df is not None and not self._dimension_ic_df.empty:
            score_cols_for_ic = [f"{k}_score" for k in w]
            df_for_flip = candidates[["stock_id", *[c for c in score_cols_for_ic if c in candidates.columns]]]
            df_flipped, actions = compute_ic_aware_score_transform(
                df_for_flip, self._dimension_ic_df, dampen_mode=dampen_mode
            )
            for col, action in actions.items():
                if action in ("flipped", "neutralized") and col in df_flipped.columns:
                    candidates[col] = df_flipped[col].values
                    logger.info("E2c IC-aware 分數轉換: %s — %s → %s", self.mode_name, col, action)
                elif action == "dampen":
                    weight_mults[col] = IC_DAMPEN_WEIGHT_MULT
                    logger.info(
                        "E2c IC-aware 分數轉換: %s — %s → dampen (weight×%.2f)",
                        self.mode_name,
                        col,
                        IC_DAMPEN_WEIGHT_MULT,
                    )
            # 持久化 actions 供 CLI 表格標記欄位狀態（N/F/D）
            self._ic_actions = dict(actions)

        # composite 加權：套 weight_mult，再歸一化回原始總和，避免 dampen 後量級下移
        # 觸發 Stage 3.7 動態門檻誤殺
        composite = pd.Series(0.0, index=candidates.index)
        original_total_weight = sum(w.values())
        effective_total_weight = 0.0
        for key, weight in w.items():
            col = f"{key}_score"
            if col in candidates.columns:
                eff_w = weight * weight_mults.get(col, 1.0)
                composite += candidates[col] * eff_w
                effective_total_weight += eff_w
        if effective_total_weight > 0 and effective_total_weight != original_total_weight:
            composite = composite * (original_total_weight / effective_total_weight)
        candidates["composite_score"] = composite

        # hook：子類可在加權後做額外處理
        candidates = self._post_score(candidates)

        # 進出場建議欄位（依 regime 調整 ATR 倍數）
        regime = getattr(self, "regime", "sideways")
        entry_exit = self._compute_entry_exit_cols(stock_ids, df_price, regime=regime)
        candidates = candidates.merge(entry_exit, on="stock_id", how="left")

        return candidates

    def get_sub_factor_df(self) -> pd.DataFrame:
        """合併所有子因子 rank 為單一 DataFrame（供因子診斷使用）。

        回傳 DataFrame 含 stock_id + 所有子因子欄位（如 tech_ret5d, chip_consec 等）。
        若無子因子資料（未執行 run() 或子類未實作），回傳空 DataFrame。
        """
        if not self._sub_factor_ranks:
            return pd.DataFrame()
        result = None
        for _key, df in self._sub_factor_ranks.items():
            if result is None:
                result = df.copy()
            else:
                result = result.merge(df, on="stock_id", how="outer")
        return result if result is not None else pd.DataFrame()

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

            # D1: 動態停損 — 根據個股 MDD 調整 ATR 倍數
            base_stop_mult, base_target_mult = REGIME_ATR_PARAMS.get(regime, REGIME_ATR_PARAMS["sideways"])
            mdd_df = compute_adaptive_atr_multiplier(stock_data, [sid], base_stop_mult=base_stop_mult, mdd_window=20)
            if not mdd_df.empty:
                adj_mult = float(mdd_df.iloc[0]["adjusted_stop_mult"])
                # 按比例調整 target_mult
                ratio = adj_mult / base_stop_mult if base_stop_mult > 0 else 1.0
                adj_target = base_target_mult * ratio
                if atr14 > 0:
                    stop_loss = round(close - adj_mult * atr14, 2)
                    take_profit = round(close + adj_target * atr14, 2)
                else:
                    stop_loss, take_profit = None, None
            else:
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

    # ------------------------------------------------------------------ #
    #  Stage 3.5c: 動量衰減偵測
    # ------------------------------------------------------------------ #

    def _apply_momentum_decay(
        self,
        scored: pd.DataFrame,
        df_price: pd.DataFrame,
    ) -> pd.DataFrame:
        """Stage 3.5c — RSI 頂背離 + MACD 柱縮短 → composite_score 降分。

        僅對 momentum / growth 模式生效（短線動能型），
        swing / value / dividend 等中長期模式不受影響。
        """
        if scored.empty:
            return scored
        if self.mode_name not in ("momentum", "growth"):
            return scored

        stock_ids = scored["stock_id"].tolist()
        decay_df = compute_momentum_decay(df_price, stock_ids)
        scored = scored.merge(decay_df, on="stock_id", how="left")
        scored["momentum_decay"] = scored["momentum_decay"].fillna(0.0)
        scored["composite_score"] = scored["composite_score"] * (1 + scored["momentum_decay"])

        n_affected = (scored["momentum_decay"] < 0).sum()
        if n_affected > 0:
            logger.info(
                "Stage 3.5c: 動量衰減偵測影響 %d 支（輕度 -3%% / 嚴重 -6%%）",
                n_affected,
            )
        return scored

    # ------------------------------------------------------------------ #
    #  Stage 3.5d: 籌碼加速度加成
    # ------------------------------------------------------------------ #

    def _apply_institutional_acceleration(
        self,
        scored: pd.DataFrame,
        df_inst: pd.DataFrame,
    ) -> pd.DataFrame:
        """Stage 3.5d — 法人買超加速 → composite_score 加分。

        法人近 3 日平均淨買超 > 0 且高於前 7 日平均 → 加分。
        適用所有模式（法人加速買超是通用正面訊號）。
        """
        if scored.empty or df_inst.empty:
            return scored

        stock_ids = scored["stock_id"].tolist()
        accel_df = compute_institutional_acceleration(df_inst, stock_ids)
        scored = scored.merge(accel_df, on="stock_id", how="left")
        scored["inst_accel_bonus"] = scored["inst_accel_bonus"].fillna(0.0)
        scored["composite_score"] = scored["composite_score"] * (1 + scored["inst_accel_bonus"])

        n_boosted = (scored["inst_accel_bonus"] > 0).sum()
        if n_boosted > 0:
            logger.info(
                "Stage 3.5d: 籌碼加速度加成 %d 支（+2%%~+4%%）",
                n_boosted,
            )
        return scored

    # ------------------------------------------------------------------ #
    #  Stage 3.5e: 多時框一致性
    # ------------------------------------------------------------------ #

    # 多時框強制共振：Momentum 模式日週矛盾（短多長空）直接排除
    _MTF_FORCE_EXCLUDE_MODES: set[str] = {"momentum", "growth"}

    def _apply_multi_timeframe_alignment(
        self,
        scored: pd.DataFrame,
    ) -> pd.DataFrame:
        """Stage 3.5e — 日線 + 週線多時框共振過濾。

        需要 weekly_confirm=True 時才有週線資料。
        日線趨勢取自 technical_score (>0.55 = 多頭)，週線趨勢取自 weekly_bonus。

        **強制共振（A2 升級）**：
        - momentum / growth 模式：日線多頭 + 週線空頭（短多長空）→ **直接排除**（追高風險最大）
        - 其他模式：矛盾時僅降分（-2%~-4%），不排除
        - 日週一致時加分 +4%
        若 weekly_bonus 不存在（未啟用週線確認），此 Stage 跳過。
        """
        if scored.empty:
            return scored
        if "weekly_bonus" not in scored.columns:
            return scored

        stock_ids = scored["stock_id"].tolist()

        daily_trend: dict[str, bool | None] = {}
        for _, row in scored.iterrows():
            sid = row["stock_id"]
            tech = row.get("technical_score")
            if pd.notna(tech):
                daily_trend[sid] = float(tech) > 0.55
            else:
                daily_trend[sid] = None

        weekly_map: dict[str, float] = dict(zip(scored["stock_id"], scored["weekly_bonus"]))

        mtf_df = compute_multi_timeframe_alignment(daily_trend, weekly_map)
        scored = scored.merge(mtf_df, on="stock_id", how="left")
        scored["mtf_alignment"] = scored["mtf_alignment"].fillna(0.0)

        # 強制共振排除：momentum/growth 模式中「日線多頭 + 週線空頭」直接剔除
        n_excluded = 0
        if self.mode_name in self._MTF_FORCE_EXCLUDE_MODES:
            # mtf_alignment == -0.03 代表「短多長空」矛盾（最危險的追高情境）
            exclude_mask = scored["mtf_alignment"] == -0.03
            n_excluded = int(exclude_mask.sum())
            if n_excluded > 0:
                scored = scored[~exclude_mask].copy()
                logger.info(
                    "Stage 3.5e: 多時框強制共振排除 %d 支（%s 模式日多週空矛盾）",
                    n_excluded,
                    self.mode_name,
                )

        # 非排除的候選股：加減分
        scored["composite_score"] = scored["composite_score"] * (1 + scored["mtf_alignment"])

        n_aligned = (scored["mtf_alignment"] > 0).sum()
        n_conflict = (scored["mtf_alignment"] < 0).sum()
        if n_aligned > 0 or n_conflict > 0 or n_excluded > 0:
            logger.info(
                "Stage 3.5e: 多時框一致性 — %d 支加分，%d 支降分，%d 支排除",
                n_aligned,
                n_conflict,
                n_excluded,
            )
        return scored

    # ------------------------------------------------------------------ #
    #  Stage 3.5f: 籌碼面 MACD
    # ------------------------------------------------------------------ #

    def _apply_chip_macd(
        self,
        scored: pd.DataFrame,
        df_inst: pd.DataFrame,
    ) -> pd.DataFrame:
        """Stage 3.5f — 法人淨買超 MACD 交叉信號 → composite_score 調整。

        短期(5日) vs 長期(20日) EMA 交叉：
        - 強勢吸籌（柱狀圖正值遞增）→ +3%
        - 出貨信號（MACD 和柱狀圖皆負）→ -3%
        """
        if scored.empty or df_inst.empty:
            return scored

        stock_ids = scored["stock_id"].tolist()
        macd_df = compute_chip_macd(df_inst, stock_ids, fast_span=5, slow_span=20, signal_span=5)
        scored = scored.merge(macd_df, on="stock_id", how="left")
        scored["chip_macd_score"] = scored["chip_macd_score"].fillna(0.5)

        # 將 chip_macd_score (0~1) 轉換為 composite_score 調整值
        # 0.5 = 中性（無調整），1.0 = +3%，0.1 = -3%
        scored["chip_macd_adj"] = (scored["chip_macd_score"] - 0.5) * 0.06
        scored["composite_score"] = scored["composite_score"] * (1 + scored["chip_macd_adj"])

        n_boost = (scored["chip_macd_adj"] > 0.005).sum()
        n_penalty = (scored["chip_macd_adj"] < -0.005).sum()
        if n_boost > 0 or n_penalty > 0:
            logger.info(
                "Stage 3.5f: 籌碼面 MACD — %d 支加分（吸籌加速）/ %d 支降分（出貨信號）",
                n_boost,
                n_penalty,
            )
        return scored

    # ------------------------------------------------------------------ #
    #  Stage 3.5g: 主力成本分析（B2）
    # ------------------------------------------------------------------ #

    def _apply_key_player_cost(
        self,
        scored: pd.DataFrame,
        df_price: pd.DataFrame,
    ) -> pd.DataFrame:
        """Stage 3.5g — 主力估計成本 vs 現價 → composite_score 調整。

        現價 < 主力成本（被套）→ +2%（護盤動力）
        現價 > 主力成本 ×1.10（已獲利）→ -2%（出貨風險）
        """
        if scored.empty:
            return scored

        stock_ids = scored["stock_id"].tolist()

        # 載入分點資料（使用 extended 取得更長歷史）
        try:
            df_broker = self._load_broker_data_extended(stock_ids, days=60, min_trading_days=5)
        except Exception:
            df_broker = self._load_broker_data(stock_ids)

        if df_broker.empty:
            return scored

        cost_df = compute_key_player_cost_basis(df_broker, stock_ids, top_n_brokers=3, lookback_days=60)
        if cost_df.empty or cost_df["key_player_cost"].isna().all():
            return scored

        # 取最新收盤價
        price_map: dict[str, float] = {}
        if not df_price.empty:
            latest = df_price.sort_values("date").groupby("stock_id").last()
            price_map = latest["close"].to_dict()

        cost_scored = score_key_player_cost(cost_df, price_map)
        scored = scored.merge(cost_scored[["stock_id", "key_player_score"]], on="stock_id", how="left")
        scored["key_player_score"] = scored["key_player_score"].fillna(0.5)

        # 分數調整：0.8 → +2%，0.2 → -2%，0.5 → 0%
        scored["kp_adj"] = (scored["key_player_score"] - 0.5) * 0.067
        scored["composite_score"] = scored["composite_score"] * (1 + scored["kp_adj"])

        n_protect = (scored["kp_adj"] > 0.005).sum()
        n_risk = (scored["kp_adj"] < -0.005).sum()
        if n_protect > 0 or n_risk > 0:
            logger.info(
                "Stage 3.5g: 主力成本分析 — %d 支護盤加分 / %d 支出貨風險降分",
                n_protect,
                n_risk,
            )
        return scored

    # ------------------------------------------------------------------ #
    #  Stage 3.6: 量價背離偵測
    # ------------------------------------------------------------------ #

    def _apply_volume_price_divergence(
        self,
        scored: pd.DataFrame,
        df_price: pd.DataFrame,
    ) -> pd.DataFrame:
        """Stage 3.6 — 量價背離調整 composite_score。

        價漲量縮 → 降分（假突破風險）；量價齊揚 → 加分。
        調整方式：composite_score *= (1 + vp_divergence)，
        vp_divergence ∈ [-0.05, +0.02]。
        """
        if scored.empty or df_price.empty:
            return scored

        stock_ids = scored["stock_id"].tolist()
        vp_df = compute_volume_price_divergence(df_price, stock_ids, window=5)
        scored = scored.merge(vp_df, on="stock_id", how="left")
        scored["vp_divergence"] = scored["vp_divergence"].fillna(0.0)
        scored["composite_score"] = scored["composite_score"] * (1 + scored["vp_divergence"])

        n_penalty = (scored["vp_divergence"] < 0).sum()
        n_bonus = (scored["vp_divergence"] > 0).sum()
        logger.info(
            "Stage 3.6: 量價背離調整 — %d 支降分 / %d 支加分",
            n_penalty,
            n_bonus,
        )
        return scored

    # ------------------------------------------------------------------ #
    #  Stage 3.7: 動態評分閾值
    # ------------------------------------------------------------------ #

    def _apply_score_threshold(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Stage 3.7 — 依 Regime 剔除 composite_score 低於門檻的候選股。

        門檻：bull=0.45, sideways=0.50, bear=0.55, crisis=0.60。
        Regime 越差門檻越高，寧缺勿濫，確保推薦品質。

        E1 勝率回饋：若近 30 天勝率 < 40% → 門檻額外 +0.05。
        """
        if scored.empty:
            return scored

        regime = getattr(self, "regime", "sideways")
        threshold = MIN_SCORE_THRESHOLDS.get(regime, 0.50)

        # E1: 勝率回饋循環 — 查詢近期推薦勝率，低勝率 → 提高門檻
        wr_adj = self._compute_win_rate_adjustment()
        if wr_adj > 0:
            threshold += wr_adj
            logger.info(
                "Stage 3.7 E1: 勝率回饋 — %s 模式近期勝率偏低，門檻 +%.2f → %.2f",
                self.mode_name,
                wr_adj,
                threshold,
            )

        # IC 衰退回饋：關鍵因子持續失效 → 提高門檻
        ic_decay_adj = self._compute_ic_decay_adjustment()
        if ic_decay_adj > 0:
            threshold += ic_decay_adj
            logger.info(
                "Stage 3.7 IC-Decay: %s 模式關鍵因子 IC 持續衰退，門檻 +%.2f → %.2f",
                self.mode_name,
                ic_decay_adj,
                threshold,
            )

        before = len(scored)
        scored = scored[scored["composite_score"] >= threshold].copy()
        removed = before - len(scored)
        if removed > 0:
            logger.info(
                "Stage 3.7: 動態評分閾值 (regime=%s, threshold=%.2f) 剔除 %d 支低分股",
                regime,
                threshold,
                removed,
            )
        return scored

    def _compute_win_rate_adjustment(self) -> float:
        """E1 — 從 DB 讀取近 30 天推薦記錄，計算勝率門檻調整值。

        若 DB 無足夠資料（<5 筆推薦）則回傳 0（不調整）。
        """
        try:
            from src.discovery.scanner._functions import WIN_RATE_FEEDBACK_CONFIG

            cfg = WIN_RATE_FEEDBACK_CONFIG
            lookback = cfg["lookback_days"]
            holding = cfg["holding_days"]
            cutoff = date.today() - timedelta(days=lookback + holding + 5)

            with get_session() as session:
                # 載入推薦記錄
                stmt = select(
                    DiscoveryRecord.scan_date,
                    DiscoveryRecord.stock_id,
                    DiscoveryRecord.close,
                ).where(
                    DiscoveryRecord.mode == self.mode_name,
                    DiscoveryRecord.scan_date >= cutoff,
                )
                rows = session.execute(stmt).all()
                if not rows:
                    return 0.0
                df_records = pd.DataFrame(rows, columns=["scan_date", "stock_id", "close"])

                # 載入價格
                stock_ids = df_records["stock_id"].unique().tolist()
                price_stmt = select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                    DailyPrice.stock_id.in_(stock_ids),
                    DailyPrice.date >= cutoff,
                )
                price_rows = session.execute(price_stmt).all()
                if not price_rows:
                    return 0.0
                df_prices = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])

            return compute_win_rate_threshold_adjustment(
                df_records,
                df_prices,
                self.mode_name,
                holding_days=holding,
                lookback_days=lookback,
            )
        except Exception:
            logger.debug("E1: 勝率回饋計算失敗，跳過")
            return 0.0

    # 關鍵因子 mapping 集中於 src/constants.py（DISCOVERY_KEY_FACTOR_MAP）
    # 此處保留類屬性別名供子類覆寫；變更請改 constants.py 單一真相來源
    _KEY_FACTOR_MAP = DISCOVERY_KEY_FACTOR_MAP

    def _compute_ic_decay_adjustment(self) -> float:
        """IC 衰退門檻調整：關鍵因子 IC 連續 2 窗口 < 0.05 時回傳 +0.05。"""
        key_factor = self._KEY_FACTOR_MAP.get(self.mode_name)
        if not key_factor:
            return 0.0
        try:
            from src.discovery.scanner._functions import compute_rolling_ic

            cutoff = date.today() - timedelta(days=35)

            with get_session() as session:
                stmt = select(
                    DiscoveryRecord.scan_date,
                    DiscoveryRecord.stock_id,
                    DiscoveryRecord.close,
                    DiscoveryRecord.technical_score,
                    DiscoveryRecord.chip_score,
                    DiscoveryRecord.fundamental_score,
                    DiscoveryRecord.news_score,
                ).where(
                    DiscoveryRecord.mode == self.mode_name,
                    DiscoveryRecord.scan_date >= cutoff,
                )
                rows = session.execute(stmt).all()
                if len(rows) < 20:
                    return 0.0
                df_records = pd.DataFrame(
                    rows,
                    columns=[
                        "scan_date",
                        "stock_id",
                        "close",
                        "technical_score",
                        "chip_score",
                        "fundamental_score",
                        "news_score",
                    ],
                )

                stock_ids = df_records["stock_id"].unique().tolist()
                price_stmt = select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                    DailyPrice.stock_id.in_(stock_ids),
                    DailyPrice.date >= cutoff,
                )
                price_rows = session.execute(price_stmt).all()
                if not price_rows:
                    return 0.0
                df_prices = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])

            rolling_df = compute_rolling_ic(df_records, df_prices, holding_days=5, window_days=14, step_days=7)
            if rolling_df.empty:
                return 0.0

            # 取關鍵因子的 IC 時間序列
            factor_df = rolling_df[rolling_df["factor"] == key_factor].sort_values("window_end")
            if len(factor_df) < 2:
                return 0.0

            # 檢查最近 2 個窗口是否都 < 0.05
            last_two = factor_df["ic"].tail(2).tolist()
            if all(ic < 0.05 for ic in last_two):
                logger.warning(
                    "IC-Decay: %s 模式關鍵因子 %s 連續 2 窗口 IC < 0.05（%s），啟動門檻提升",
                    self.mode_name,
                    key_factor,
                    [f"{v:+.4f}" for v in last_two],
                )
                return 0.05
            return 0.0
        except Exception:
            logger.debug("IC-Decay: 計算失敗，跳過")
            return 0.0

    # ------------------------------------------------------------------ #
    #  E2: 因子有效性監控（日誌記錄 IC，供人工調參參考）
    # ------------------------------------------------------------------ #

    def _log_factor_effectiveness(self) -> None:
        """E2 — 計算四維度因子 IC（Spearman Rank Correlation）並記錄日誌。

        不影響當前評分流程，僅提供因子有效性資訊供後續人工調參。
        若 DB 無足夠資料（<10 筆推薦）則靜默跳過。

        項目 E：若 `self._precomputed_ic` 已由 `run(precomputed_ic=...)` 注入，直接使用，
        跳過 DB 查詢 + `compute_factor_ic` 呼叫。
        """
        # 項目 E 短路：使用 morning-routine Step 8c 預算的 IC
        precomputed = getattr(self, "_precomputed_ic", None)
        if precomputed is not None and not precomputed.empty:
            for _, row in precomputed.iterrows():
                logger.info(
                    "E2 因子IC: %s — %s IC=%.4f (%s, n=%d) [precomputed]",
                    self.mode_name,
                    row["factor"],
                    row["ic"],
                    row["direction"],
                    row["evaluable_count"],
                )
            return

        try:
            cutoff = date.today() - timedelta(days=35)

            with get_session() as session:
                stmt = select(
                    DiscoveryRecord.scan_date,
                    DiscoveryRecord.stock_id,
                    DiscoveryRecord.close,
                    DiscoveryRecord.technical_score,
                    DiscoveryRecord.chip_score,
                    DiscoveryRecord.fundamental_score,
                    DiscoveryRecord.news_score,
                ).where(
                    DiscoveryRecord.mode == self.mode_name,
                    DiscoveryRecord.scan_date >= cutoff,
                )
                rows = session.execute(stmt).all()
                if not rows:
                    return
                df_records = pd.DataFrame(
                    rows,
                    columns=[
                        "scan_date",
                        "stock_id",
                        "close",
                        "technical_score",
                        "chip_score",
                        "fundamental_score",
                        "news_score",
                    ],
                )

                stock_ids = df_records["stock_id"].unique().tolist()
                price_stmt = select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                    DailyPrice.stock_id.in_(stock_ids),
                    DailyPrice.date >= cutoff,
                )
                price_rows = session.execute(price_stmt).all()
                if not price_rows:
                    return
                df_prices = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])

            ic_df = compute_factor_ic(df_records, df_prices, holding_days=5, lookback_days=30)
            if ic_df.empty:
                return

            for _, row in ic_df.iterrows():
                logger.info(
                    "E2 因子IC: %s — %s IC=%.4f (%s, n=%d)",
                    self.mode_name,
                    row["factor"],
                    row["ic"],
                    row["direction"],
                    row["evaluable_count"],
                )
        except Exception:
            logger.debug("E2: 因子有效性計算失敗，跳過")

    # ------------------------------------------------------------------ #
    #  E2b: Factor IC 動態權重調整
    # ------------------------------------------------------------------ #

    def _apply_ic_weight_adjustment(
        self,
        base_weights: dict[str, float],
        scored_candidates: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """E2b — 根據歷史 IC + 今日影響力自動微調四維度權重（Phase C）。

        雙指標軟調整：
          - IC（離散）：方向正確性檢驗，weak/inverse 大幅衰減
          - 影響力（軟）：1 - ρ 鑑別力檢驗，圍繞 mean 做 ±20% 軟調整

        需 >= 20 筆可評估推薦紀錄（~4 天 × top 5）才啟動 IC 端調整。
        影響力需今日候選 ≥ 10 檔 + dimension 樣本 ≥ 3 才啟用。
        調整後權重歸一化至原始總和，確保不改變評分量級。

        Args:
            base_weights: 原始 regime 權重
            scored_candidates: 今日候選（需含 {key}_score 欄位），供 ablation 計算影響力

        Returns:
            調整後權重字典

        項目 E：若 `self._precomputed_ic` 非空，跳過 DB 查詢與 `compute_factor_ic` 呼叫，
        直接使用預算結果（與 morning-routine Step 8c 一致的 holding_days=5/lookback_days=30）。
        """
        try:
            # 項目 E 短路：使用預算 IC，跳過 DB 查詢與重算
            precomputed = getattr(self, "_precomputed_ic", None)
            if precomputed is not None and not precomputed.empty:
                ic_df = precomputed
                self._dimension_ic_df = ic_df
            else:
                cutoff = date.today() - timedelta(days=35)

                with get_session() as session:
                    stmt = select(
                        DiscoveryRecord.scan_date,
                        DiscoveryRecord.stock_id,
                        DiscoveryRecord.close,
                        DiscoveryRecord.technical_score,
                        DiscoveryRecord.chip_score,
                        DiscoveryRecord.fundamental_score,
                        DiscoveryRecord.news_score,
                    ).where(
                        DiscoveryRecord.mode == self.mode_name,
                        DiscoveryRecord.scan_date >= cutoff,
                    )
                    rows = session.execute(stmt).all()
                    if len(rows) < 20:
                        logger.info("E2b: 歷史推薦不足 20 筆（%d），跳過 IC 權重調整", len(rows))
                        return dict(base_weights)

                    df_records = pd.DataFrame(
                        rows,
                        columns=[
                            "scan_date",
                            "stock_id",
                            "close",
                            "technical_score",
                            "chip_score",
                            "fundamental_score",
                            "news_score",
                        ],
                    )

                    stock_ids = df_records["stock_id"].unique().tolist()
                    price_stmt = select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                        DailyPrice.stock_id.in_(stock_ids),
                        DailyPrice.date >= cutoff,
                    )
                    price_rows = session.execute(price_stmt).all()
                    if not price_rows:
                        return dict(base_weights)
                    df_prices = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])

                ic_df = compute_factor_ic(df_records, df_prices, holding_days=5, lookback_days=30)
                # 暫存供 _score_candidates 做 IC 感知分數翻轉使用（問題 3：news_score 反向修正）
                self._dimension_ic_df = ic_df

            # 計算今日影響力（若候選資料充足）
            impact_df = self._compute_dimension_impact(base_weights, scored_candidates)

            # 若兩者皆無資料，直接返回
            if ic_df.empty and impact_df.empty:
                return dict(base_weights)

            # 將 base_weights key (如 "technical") 轉為 score 欄位名 (如 "technical_score")
            score_weights = {f"{k}_score": v for k, v in base_weights.items()}

            # 影響力 factor 名稱亦轉為 {key}_score 對齊 ic_df
            if not impact_df.empty:
                impact_df = impact_df.copy()
                impact_df["factor"] = impact_df["factor"].astype(str) + "_score"

            adjusted_score = compute_ic_impact_weight_adjustments(ic_df, impact_df, score_weights)
            # 轉回原始 key
            adjusted = {k.replace("_score", ""): v for k, v in adjusted_score.items()}

            for key in base_weights:
                if abs(base_weights[key] - adjusted.get(key, base_weights[key])) > 1e-6:
                    logger.info(
                        "E2b IC×影響力權重調整: %s — %s %.3f → %.3f",
                        self.mode_name,
                        key,
                        base_weights[key],
                        adjusted[key],
                    )
            return adjusted

        except Exception:
            logger.debug("E2b: IC 權重調整失敗，使用原始權重")
            return dict(base_weights)

    def _compute_dimension_impact(
        self,
        base_weights: dict[str, float],
        scored_candidates: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """計算今日候選之維度影響力（1 - ρ）。

        邏輯：用 base_weights 對 scored_candidates 執行 dimension ablation，
        取得每個維度移除後的 Spearman ρ。ρ 越低 → 影響力越高。

        最低候選數 10 檔，不足則返回空 DataFrame。
        """
        if scored_candidates is None or scored_candidates.empty or len(scored_candidates) < 10:
            return pd.DataFrame()
        try:
            from src.discovery.ablation import run_dimension_ablation

            results = run_dimension_ablation(scored_candidates, base_weights, top_n=min(20, len(scored_candidates)))
            if not results:
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    {
                        "factor": r.removed_dimension,
                        "rank_correlation": r.rank_correlation,
                    }
                    for r in results
                ]
            )
        except Exception:
            logger.debug("E2b: 影響力計算失敗，回退純 IC 調整")
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Stage 4.3: 籌碼層級降級稽核
    # ------------------------------------------------------------------ #

    def _audit_chip_tier_changes(self, rankings: pd.DataFrame) -> pd.DataFrame:
        """Stage 4.3 — 比對前次掃描 chip_tier，記錄升降級至日誌與 DataFrame。

        從 DB 讀取前一次同模式 DiscoveryRecord 的 chip_tier，
        呼叫 detect_chip_tier_changes() 純函數比對。
        結果寫入 rankings["chip_tier_change"] 欄位（如 "8F→7F"），供儲存至 DB。
        """
        if "chip_tier" not in rankings.columns or rankings.empty:
            rankings["chip_tier_change"] = None
            return rankings

        try:
            from sqlalchemy import func

            with get_session() as session:
                # 找前一次掃描日期
                prev_date_row = session.execute(
                    select(func.max(DiscoveryRecord.scan_date)).where(
                        DiscoveryRecord.mode == self.mode_name,
                        DiscoveryRecord.scan_date < self.scan_date,
                    )
                ).scalar()
                if prev_date_row is None:
                    rankings["chip_tier_change"] = None
                    return rankings

                # 載入前次記錄的 chip_tier
                prev_rows = session.execute(
                    select(DiscoveryRecord.stock_id, DiscoveryRecord.chip_tier).where(
                        DiscoveryRecord.mode == self.mode_name,
                        DiscoveryRecord.scan_date == prev_date_row,
                    )
                ).all()
                if not prev_rows:
                    rankings["chip_tier_change"] = None
                    return rankings

            df_prev = pd.DataFrame(prev_rows, columns=["stock_id", "chip_tier"])
            changes = detect_chip_tier_changes(rankings, df_prev)

            # 寫入 chip_tier_change 欄位
            if changes.empty:
                rankings["chip_tier_change"] = None
                return rankings

            change_map = {}
            for _, row in changes.iterrows():
                label = f"{row['prev_tier']}→{row['curr_tier']}"
                change_map[row["stock_id"]] = label
                if row["direction"] == "downgrade":
                    logger.warning("籌碼層級降級: %s %s", row["stock_id"], label)
                else:
                    logger.info("籌碼層級升級: %s %s", row["stock_id"], label)

            rankings["chip_tier_change"] = rankings["stock_id"].map(change_map)
            return rankings

        except Exception:
            logger.debug("Stage 4.3: 籌碼層級稽核失敗，跳過")
            rankings["chip_tier_change"] = None
            return rankings

    # ------------------------------------------------------------------ #
    #  Stage 4.1: 同產業分散化
    # ------------------------------------------------------------------ #

    def _apply_sector_diversification(self, rankings: pd.DataFrame) -> tuple[pd.DataFrame, set[str], set[str]]:
        """Stage 4.1 — 限制同產業推薦數量，降低集中風險。

        同產業最多佔推薦總數 25%（至少 3 檔），超出部分依 composite_score
        從低到高剔除。被剔除的位置由下一順位（不同產業）遞補。

        Returns:
            (result, sector_capped_ids, pool_ids)
              result             — 分散化後的 DataFrame
              sector_capped_ids  — 於 pool 內被產業上限剔除的 stock_id
              pool_ids           — 本次被納入考慮的 pool（最多 top_n*2 筆）
        """
        if rankings.empty or "industry_category" not in rankings.columns:
            return rankings, set(), set(rankings["stock_id"]) if not rankings.empty else set()

        top_n = self.top_n_results
        sector_cap = max(SECTOR_MIN_CAP, int(top_n * SECTOR_MAX_RATIO))

        # 先取比 top_n 更多的候選（讓遞補有空間）
        pool = rankings.head(top_n * 2) if len(rankings) > top_n else rankings.copy()
        pool_ids = set(pool["stock_id"])

        kept: list[int] = []  # 保留的 row index
        kept_ids: set[str] = set()
        sector_capped_ids: set[str] = set()
        sector_counts: dict[str, int] = {}

        for idx, row in pool.iterrows():
            sector = row.get("industry_category", "") or "未分類"
            count = sector_counts.get(sector, 0)
            if count < sector_cap:
                kept.append(idx)
                kept_ids.add(row["stock_id"])
                sector_counts[sector] = count + 1
                if len(kept) >= top_n:
                    break
            else:
                # 僅當 pool 內仍有機會填滿 top_n 時，視為真正被產業上限剔除
                sector_capped_ids.add(row["stock_id"])

        result = pool.loc[kept].copy()
        result["rank"] = range(1, len(result) + 1)

        capped = [s for s, c in sector_counts.items() if c >= sector_cap]
        if capped:
            logger.info(
                "Stage 4.1: 同產業分散化 — 產業上限 %d，觸及上限：%s",
                sector_cap,
                ", ".join(capped),
            )

        return result, sector_capped_ids, pool_ids

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
        """動能風格技術面 4 因子（橫截面排名版）。

        4 因子分 3 Cluster 等權：
        - Cluster A（報酬動能）：mean(ret5d, ret10d)
        - Cluster B（量能擴張）：mean(vol_ratio, vol_accel)
        - Cluster C（突破強度）：close / max(close, 20d)（20 日高點接近度）

        v2 變更（冗餘因子精簡）：
        - 移除 sharpe_proxy（與 ret_10d r=0.91，計算重疊）
        - 移除 breakout_60d（需 60 日資料，多數股票不足永遠 NaN）
        - 新增 high20_proximity（close / 20日最高，度量突破而非報酬率，降低 Cluster 間冗餘）

        所有因子用橫截面 rank(pct=True)，自動適應牛熊市（Regime Adaptable）。
        資料不足或缺失的因子以中性分 0.5 填補。
        供 MomentumScanner 與 GrowthScanner 共用。
        """
        if not stock_ids:
            return pd.DataFrame(columns=["stock_id", "technical_score"])

        df = df_price.sort_values(["stock_id", "date"])
        g = df.groupby("stock_id", sort=False)

        # ── 批次取各期收盤價 ──────────────────────────────────────────
        latest_close = g["close"].last()
        close_1d_ago = g["close"].apply(
            lambda s: float(s.iloc[-2]) if len(s) >= 2 else np.nan
        )  # 前一交易日（漲停偵測用）
        close_5d_ago = g["close"].apply(lambda s: float(s.iloc[-6]) if len(s) >= 6 else np.nan)
        close_10d_ago = g["close"].apply(lambda s: float(s.iloc[-11]) if len(s) >= 11 else np.nan)
        # 近 20 日最高收盤（突破強度：距近期高點的距離）
        close_20d_max = g["close"].apply(lambda s: float(s.iloc[-20:].max()) if len(s) >= 20 else np.nan)

        # ── 批次取量能序列 ────────────────────────────────────────────
        latest_vol = g["volume"].last().astype(float)
        vol_20d_mean = g["volume"].apply(lambda s: float(s.astype(float).iloc[-20:].mean()) if len(s) >= 20 else np.nan)
        vol_3d_mean = g["volume"].apply(lambda s: float(s.astype(float).iloc[-3:].mean()) if len(s) >= 3 else np.nan)
        vol_10d_mean = g["volume"].apply(lambda s: float(s.astype(float).iloc[-10:].mean()) if len(s) >= 10 else np.nan)

        # ── 限縮到候選股集合，計算原始因子值 ─────────────────────────
        idx = pd.Index(stock_ids)
        c0 = latest_close.reindex(idx)
        c5 = close_5d_ago.reindex(idx).replace(0, np.nan)
        c10 = close_10d_ago.reindex(idx).replace(0, np.nan)
        c20m = close_20d_max.reindex(idx).replace(0, np.nan)

        ret_5d = (c0 - c5) / c5
        ret_10d = (c0 - c10) / c10
        high20_proximity = c0 / c20m  # 20 日高點接近度（=1 時創新高，<1 拉回整理）

        vol_20d = vol_20d_mean.reindex(idx).replace(0, np.nan)
        vol_ratio_raw = latest_vol.reindex(idx) / vol_20d
        vol_accel_raw = vol_3d_mean.reindex(idx) / vol_10d_mean.reindex(idx).replace(0, np.nan)

        # ── 橫截面百分位排名（Regime Adaptive）─────────────────────────
        r5 = ret_5d.rank(pct=True)
        r10 = ret_10d.rank(pct=True)
        rb = high20_proximity.rank(pct=True)
        rv = vol_ratio_raw.rank(pct=True)
        ra = vol_accel_raw.rank(pct=True)

        # ── 漲停板特殊處理（台股 10% 漲跌幅限制）────────────────────────
        # 強勢「鎖漲停」時成交量急縮屬正常現象，不應懲罰量比/量能加速因子。
        c1d = close_1d_ago.reindex(idx).replace(0, np.nan)
        limit_up_mask = ((c0 - c1d) / c1d) >= 0.098
        rv = rv.where(~limit_up_mask, other=1.0)
        ra = ra.where(~limit_up_mask, other=1.0)

        # NaN（資料不足）以中性 0.5 填補
        scores = pd.concat([r5, r10, rb, rv, ra], axis=1)
        scores.columns = ["r5", "r10", "rb", "rv", "ra"]
        scores = scores.fillna(0.5)

        # Cluster 等權：4 因子分 3 Cluster，消除冗餘加權
        # Cluster A（報酬動能）：ret5d, ret10d
        # Cluster B（量能擴張）：vol_ratio, vol_accel
        # Cluster C（突破強度）：high20_proximity
        cluster_a = scores[["r5", "r10"]].mean(axis=1)
        cluster_b = scores[["rv", "ra"]].mean(axis=1)
        cluster_c = scores["rb"]

        # 零方差 Cluster 自動排除：若某 Cluster 完全無鑑別力，排除後等權剩餘
        _eps = 1e-9
        clusters = [("A", cluster_a), ("B", cluster_b), ("C", cluster_c)]
        active = [(name, s) for name, s in clusters if s.std() >= _eps]
        if active:
            tech_score = sum(s for _, s in active) / len(active)
        else:
            tech_score = pd.Series(0.5, index=scores.index)

        # 保存子因子 rank 供 IC 診斷使用
        sub_df = scores.copy()
        sub_df["stock_id"] = idx.tolist()
        sub_df = sub_df.rename(
            columns={
                "r5": "tech_ret5d",
                "r10": "tech_ret10d",
                "rb": "tech_high20_proximity",
                "rv": "tech_vol_ratio",
                "ra": "tech_vol_accel",
            }
        )
        self._sub_factor_ranks["technical"] = sub_df

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
            "vp_divergence",
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

    # ------------------------------------------------------------------ #
    #  Stage 4.2: 回撤降頻
    # ------------------------------------------------------------------ #

    # 防禦型模式：嚴重回撤時仍允許正常推薦
    _DEFENSIVE_MODES: set[str] = {"value", "dividend"}

    def _compute_drawdown_adjusted_top_n(self, df_price: pd.DataFrame) -> int:
        """Stage 4.2 — 根據 TAIEX 20 日回撤幅度調整推薦數量。

        規則（D3）：
        - TAIEX 20 日回撤 > -10% → 正常推薦（top_n_results）
        - TAIEX 20 日回撤 -10%~-15% → 推薦數量砍半
        - TAIEX 20 日回撤 < -15% → 僅防禦型模式（value/dividend）正常推薦，
          其他模式（momentum/swing/growth）推薦數砍至 1/3

        Returns:
            調整後的 top_n
        """
        original = self.top_n_results

        # 計算 TAIEX 20 日回撤
        try:
            taiex = df_price[df_price["stock_id"] == "TAIEX"]
            if taiex.empty:
                return original

            taiex_sorted = taiex.sort_values("date")
            if len(taiex_sorted) < 20:
                return original

            recent_close = float(taiex_sorted["close"].iloc[-1])
            high_20d = float(taiex_sorted["close"].iloc[-20:].max())
            drawdown = (recent_close - high_20d) / high_20d if high_20d > 0 else 0.0

        except Exception:
            return original

        if drawdown > -0.10:
            return original  # 正常市場

        is_defensive = self.mode_name in self._DEFENSIVE_MODES

        if drawdown > -0.15:
            # 中度回撤：砍半（防禦型不受影響）
            adjusted = original if is_defensive else max(3, original // 2)
            logger.info(
                "Stage 4.2: TAIEX 20 日回撤 %.1f%%，%s 推薦數 %d → %d",
                drawdown * 100,
                self.mode_name,
                original,
                adjusted,
            )
            return adjusted
        else:
            # 嚴重回撤：非防禦型砍至 1/3
            adjusted = original if is_defensive else max(3, original // 3)
            logger.info(
                "Stage 4.2: TAIEX 20 日回撤 %.1f%%（嚴重），%s 推薦數 %d → %d",
                drawdown * 100,
                self.mode_name,
                original,
                adjusted,
            )
            return adjusted

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

    @staticmethod
    def _apply_chip_quality_modifiers(
        chip_score: pd.Series,
        stock_ids: list[str],
        slope_df: pd.DataFrame | None = None,
        hhi_trend_df: pd.DataFrame | None = None,
        slope_bonus: float = 0.03,
        hhi_bonus: float = 0.03,
        hhi_high_threshold: float = 0.25,
    ) -> pd.Series:
        """對 chip_score 施加籌碼品質修正（斜率 + HHI 趨勢）。

        - 法人斜率正 → chip_score +bonus；斜率負 → -bonus
        - HHI 趨勢上升 且 HHI 水位高 → +bonus（吸籌）；趨勢下降 → -bonus（出貨）

        修正以 clip(0, 1) 確保 chip_score 不越界。

        Args:
            chip_score: 原始 chip_score Series（0~1），index 對齊 stock_ids
            stock_ids: 候選股代號清單
            slope_df: compute_inst_net_buy_slope() 回傳，[stock_id, inst_slope]
            hhi_trend_df: compute_hhi_trend() 回傳，[stock_id, hhi_trend, hhi_short_avg]
            slope_bonus: 斜率修正幅度（預設 ±3%）
            hhi_bonus: HHI 趨勢修正幅度（預設 ±3%）
            hhi_high_threshold: HHI 絕對值需 ≥ 此門檻才施加趨勢修正（預設 0.25）

        Returns:
            修正後的 chip_score Series
        """
        adjusted = chip_score.copy().astype(float)

        # ── 法人斜率修正 ──────────────────────────────────────────
        if slope_df is not None and not slope_df.empty:
            slope_map = dict(zip(slope_df["stock_id"], slope_df["inst_slope"], strict=False))
            for i, sid in enumerate(stock_ids):
                slope = slope_map.get(sid, 0.0)
                if slope > 0:
                    adjusted.iloc[i] += slope_bonus
                elif slope < 0:
                    adjusted.iloc[i] -= slope_bonus

        # ── HHI 趨勢修正 ─────────────────────────────────────────
        if hhi_trend_df is not None and not hhi_trend_df.empty:
            trend_map = dict(zip(hhi_trend_df["stock_id"], hhi_trend_df["hhi_trend"], strict=False))
            hhi_map = dict(zip(hhi_trend_df["stock_id"], hhi_trend_df["hhi_short_avg"], strict=False))
            for i, sid in enumerate(stock_ids):
                trend = trend_map.get(sid, 0.0)
                hhi_val = hhi_map.get(sid, 0.0)
                # 只有 HHI 水位高時，趨勢才有意義（低 HHI = 散戶市，趨勢無意義）
                if hhi_val >= hhi_high_threshold:
                    if trend > 0:
                        adjusted.iloc[i] += hhi_bonus  # 集中化 = 吸籌
                    elif trend < 0:
                        adjusted.iloc[i] -= hhi_bonus  # 分散化 = 出貨

        return adjusted.clip(0.0, 1.0)

    def _apply_daytrade_penalty(
        self,
        broker_rank: pd.Series,
        df_broker: pd.DataFrame,
        stock_ids: list[str],
        df_price: pd.DataFrame | None = None,
        penalty_factor: float = 0.5,
        persistence_scores: pd.DataFrame | None = None,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """對 broker_rank 施加隔日沖扣分，同時回傳 penalty_df 供持久化。

        隔日沖分點（黑名單 + 行為偵測）的買超佔比越高，broker_rank 扣分越重。
        最多扣除 penalty_factor（預設 50%），避免單一負面因子完全壓制分點因子。

        若提供 persistence_scores（法人連續性），則以 ``(1 - persistence)`` 調節
        penalty — 法人連續買超越強的股票，隔沖扣分越小（隔沖只是來抬轎）。

        Args:
            broker_rank: 分點因子的 percentile rank Series（0~1），index 對齊 stock_ids
            df_broker: BrokerTrade DataFrame（含 broker_name 欄位）
            stock_ids: 候選股代號清單
            df_price: 日K 線資料（可選，用於計算 20 日均量做流動性閾值）
            penalty_factor: 最大扣分比例（預設 0.5，即 penalty=1.0 時 rank 打 5 折）
            persistence_scores: DataFrame [stock_id, inst_persistence]（可選，法人連續性 0~1）

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

        # 建立 stock_id → penalty / tags 映射
        penalty_map = dict(zip(penalty_df["stock_id"], penalty_df["daytrade_penalty"], strict=False))
        tags_map = dict(zip(penalty_df["stock_id"], penalty_df["top_dt_brokers"], strict=False))

        # 建立 stock_id → persistence 映射（用於調節 penalty）
        persist_map: dict[str, float] = {}
        if persistence_scores is not None and not persistence_scores.empty:
            persist_map = dict(
                zip(persistence_scores["stock_id"], persistence_scores["inst_persistence"], strict=False)
            )

        # 扣分：adjusted_penalty = penalty × (1 - persistence)
        #       rank *= (1 - adjusted_penalty × penalty_factor)
        adjusted = broker_rank.copy()
        dt_penalties = []
        dt_tags = []
        for i, sid in enumerate(stock_ids):
            p = penalty_map.get(sid, 0.0)
            dt_penalties.append(p)
            dt_tags.append(tags_map.get(sid, ""))
            if p > 0:
                persistence = persist_map.get(sid, 0.0)
                effective_penalty = p * (1.0 - persistence)
                adjusted.iloc[i] = adjusted.iloc[i] * (1 - effective_penalty * penalty_factor)

        result_df = pd.DataFrame({"stock_id": stock_ids, "daytrade_penalty": dt_penalties, "daytrade_tags": dt_tags})
        return adjusted, result_df


# ====================================================================== #
#  MomentumScanner — 短線動能模式
# ====================================================================== #
