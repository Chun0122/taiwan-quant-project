"""Universe Filtering Module — 三層漏斗：全市場粗篩。

漏斗架構：
  Stage 1: SQL 硬過濾（StockInfo + DailyPrice 子查詢）
           → 排除 ETF/權證/特別股/新股/低價股
           → 約 1200~1500 支
  Stage 2: 流動性過濾（Pandas，基於 turnover 成交金額）
           → 5 日均成交金額 > 3000 萬且最低 > 1000 萬
           → 約 500~700 支
  Stage 3: 趨勢動能過濾（Pandas，close > MA60 + 量能放大）
           → 約 100~200 支
  Candidate Memory: union 前一日 DiscoveryRecord，降低每日換股率

整合進 MarketScanner._load_market_data()，將原本的全量載入改為
「SQL 先篩 → 只讀候選股 DailyPrice」，節省約 75% I/O。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import func, select

from src.data.database import get_session
from src.data.schema import DailyFeature, DailyPrice, DiscoveryRecord, StockInfo

logger = logging.getLogger(__name__)


@dataclass
class UniverseConfig:
    """Universe Filtering 設定，供各 Scanner 模式個別覆寫。

    Attributes:
        min_close:            Stage 1 最低收盤價過濾（預設 10 元）
        min_available_days:   Stage 1 近 365 天最少交易日數（新股保護）
        listing_types:        Stage 1 允許的掛牌類型
        security_type:        Stage 1 允許的有價證券類型（None = 不過濾）
        avg_turnover_5d_min:  Stage 2 5 日均成交金額下限（元）
        min_turnover_5d_min:  Stage 2 5 日最低成交金額下限（元，防假流動性）
        turnover_ma20_min:    Stage 2 20 日均成交金額下限（元，雙窗口確認，防短暫放量誘多）
        trend_ma:             Stage 3 趨勢過濾均線週期（None = 跳過趨勢過濾）
        volume_ratio_min:     Stage 3 量比門檻（volume / volume_ma20，None = 跳過）
        candidate_memory_days: Candidate Memory 回溯天數（0 = 停用）
    """

    min_close: float = 10.0
    min_available_days: int = 20  # ATR14+SMA20 需至少 20 天；DB 初期歷史較短時不誤殺正常股票
    listing_types: tuple[str, ...] = ("twse", "tpex")
    security_type: str | None = "stock"
    avg_turnover_5d_min: float = 30_000_000.0
    min_turnover_5d_min: float = 10_000_000.0
    turnover_ma20_min: float = 20_000_000.0  # Stage 2 20 日均成交金額下限（防短暫放量誘多）
    trend_ma: int | None = 60
    volume_ratio_min: float | None = 1.5
    candidate_memory_days: int = 1

    # 模式覆寫建議（由 Scanner.__init__ 傳入）
    # MomentumScanner: 預設
    # SwingScanner:    volume_ratio_min=1.2
    # ValueScanner:    trend_ma=None, volume_ratio_min=None
    # DividendScanner: trend_ma=None, volume_ratio_min=None
    # GrowthScanner:   trend_ma=20, volume_ratio_min=2.0
    extra: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────
#  純函數（可獨立測試）
# ────────────────────────────────────────────────────────────────


def filter_liquidity(df_5d: pd.DataFrame, config: UniverseConfig) -> list[str]:
    """Stage 2 純函數：依成交金額過濾，回傳通過的 stock_id 清單。

    雙窗口確認邏輯：
    - 基本條件：5 日均成交金額 >= avg_turnover_5d_min，且 5 日最低 >= min_turnover_5d_min
    - 若 df_5d 含 turnover_ma20 欄位（DailyFeature 或 fallback 計算），額外要求
      20 日均成交金額 >= turnover_ma20_min，防止「近 5 日突然爆量、中期流動性不足」
      的誘多陷阱股通過過濾

    Args:
        df_5d: 含 [stock_id, turnover] 欄位的 DataFrame；
               可選含 turnover_ma20（每股最新一日的 20 日均成交金額）
        config: UniverseConfig

    Returns:
        通過流動性門檻的 stock_id 清單
    """
    if df_5d.empty or "turnover" not in df_5d.columns:
        return []

    grp = df_5d.groupby("stock_id")["turnover"]
    avg5 = grp.mean()
    min5 = grp.min()
    mask = (avg5 >= config.avg_turnover_5d_min) & (min5 >= config.min_turnover_5d_min)

    # 雙窗口確認：若提供 20 日均成交金額，額外驗證中期流動性穩定
    # NaN = 資料不足（新股或 DailyFeature 尚未重算），跳過該股的 ma20 門檻
    if "turnover_ma20" in df_5d.columns:
        ma20 = df_5d.groupby("stock_id")["turnover_ma20"].last()
        ma20 = ma20.reindex(avg5.index)
        ma20_ok = ma20.isna() | (ma20 >= config.turnover_ma20_min)
        mask = mask & ma20_ok

    return list(avg5[mask].index)


def filter_trend(df_hist: pd.DataFrame, config: UniverseConfig) -> list[str]:
    """Stage 3 純函數：依趨勢條件過濾，回傳通過的 stock_id 清單。

    過濾條件：
    1. close > MA{trend_ma}（預設 MA60，而非 MA20，可捕捉剛突破的股票）
    2. volume_ratio = volume / volume_ma20 >= volume_ratio_min

    異常排除：近 5 日若有 3 天以上漲跌幅 >= 9.5%（連續漲跌停），排除。

    Args:
        df_hist: 含 [stock_id, date, close, volume, ma60, volume_ma20] 的 DataFrame
                 （若 trend_ma=20 則需 ma20 欄；欄位來自 DailyFeature 或即時計算）
        config: UniverseConfig

    Returns:
        通過趨勢動能門檻的 stock_id 清單
    """
    if config.trend_ma is None:
        # Value/Dividend Scanner：跳過趨勢過濾
        return list(df_hist["stock_id"].unique()) if not df_hist.empty else []

    if df_hist.empty:
        return []

    ma_col = f"ma{config.trend_ma}"
    if ma_col not in df_hist.columns:
        logger.warning("filter_trend: 缺少欄位 %s，跳過趨勢過濾", ma_col)
        return list(df_hist["stock_id"].unique())

    # 取最新一日
    latest_date = df_hist["date"].max()
    latest = df_hist[df_hist["date"] == latest_date].copy()

    # 1. close > MA{N}
    close_ok = latest[ma_col].notna() & (latest["close"] >= latest[ma_col])

    # 2. 量比過濾
    if config.volume_ratio_min is not None and "volume_ma20" in df_hist.columns:
        vol_ma20 = latest["volume_ma20"].replace(0, float("nan"))
        vol_ratio = latest["volume"] / vol_ma20
        vol_ok = vol_ratio >= config.volume_ratio_min
    else:
        vol_ok = pd.Series(True, index=latest.index)

    # 3. 異常排除：近 7 天內出現 3 次以上漲跌停（累計，非必連續）
    recent_5 = df_hist[df_hist["date"] >= latest_date - timedelta(days=7)]
    pct_chg = recent_5.sort_values("date").groupby("stock_id")["close"].pct_change().abs()
    limit_days = pct_chg >= 0.095
    consecutive_limit = limit_days.groupby(level=0).sum() if not limit_days.empty else pd.Series(dtype=float)
    anomaly_ids = set(consecutive_limit[consecutive_limit >= 3].index) if not consecutive_limit.empty else set()
    not_anomaly = ~latest["stock_id"].isin(anomaly_ids)

    pass_mask = close_ok & vol_ok & not_anomaly
    return list(latest.loc[pass_mask, "stock_id"])


# ────────────────────────────────────────────────────────────────
#  UniverseFilter 主類別
# ────────────────────────────────────────────────────────────────


class UniverseFilter:
    """三層漏斗全市場宇宙過濾器。

    使用方式（供 MarketScanner 呼叫）：
        uf = UniverseFilter(config)
        candidate_ids, stats = uf.run(mode="momentum")
        # candidate_ids: ~150-200 支 stock_id
        # stats: 各階段剩餘數量統計
    """

    def __init__(self, config: UniverseConfig | None = None) -> None:
        self.config = config or UniverseConfig()

    def run(self, mode: str = "momentum") -> tuple[list[str], dict]:
        """執行三階段宇宙過濾，回傳候選 stock_id 清單與統計摘要。

        Args:
            mode: Discover 模式名稱，用於 Candidate Memory 查詢

        Returns:
            (candidate_ids, stats)
            stats keys: total_after_sql, total_after_liquidity,
                        total_after_trend, from_memory, final_candidates
        """
        stats: dict[str, int] = {}

        # Stage 1: SQL 硬過濾
        stage1_ids = self._stage1_sql_filter()
        stats["total_after_sql"] = len(stage1_ids)
        logger.info("UniverseFilter Stage 1: SQL 過濾後 %d 支", len(stage1_ids))

        if not stage1_ids:
            return [], {
                **stats,
                "total_after_liquidity": 0,
                "total_after_trend": 0,
                "from_memory": 0,
                "final_candidates": 0,
            }

        # Stage 2: 流動性過濾
        stage2_ids = self._stage2_liquidity_filter(stage1_ids)
        stats["total_after_liquidity"] = len(stage2_ids)
        logger.info("UniverseFilter Stage 2: 流動性過濾後 %d 支", len(stage2_ids))

        if not stage2_ids:
            return [], {**stats, "total_after_trend": 0, "from_memory": 0, "final_candidates": 0}

        # Stage 3: 趨勢動能過濾
        stage3_ids = self._stage3_trend_filter(stage2_ids)
        stats["total_after_trend"] = len(stage3_ids)
        logger.info("UniverseFilter Stage 3: 趨勢過濾後 %d 支", len(stage3_ids))

        # 安全防護：Stage 3 完全過濾時退回 Stage 2 結果（MA60 需 ~60 天，DB 初期資料不足時保護）
        if not stage3_ids and stage2_ids:
            logger.warning(
                "UniverseFilter Stage 3: 趨勢資料不足（MA%s 需 ~%d 天），退回 Stage 2 的 %d 支",
                self.config.trend_ma,
                self.config.trend_ma or 60,
                len(stage2_ids),
            )
            stage3_ids = stage2_ids

        # Candidate Memory: union 前一日推薦（降低每日換股率）
        memory_ids = set()
        if self.config.candidate_memory_days > 0:
            memory_ids = self._load_candidate_memory(mode, stage1_ids)
            extra = memory_ids - set(stage3_ids)
            if extra:
                stage3_ids = stage3_ids + list(extra)
                logger.info("UniverseFilter Memory: 加入 %d 支昨日候選", len(extra))

        stats["from_memory"] = len(memory_ids & (set(stage3_ids) - set(stage3_ids[: stats["total_after_trend"]])))
        stats["final_candidates"] = len(stage3_ids)

        return stage3_ids, stats

    # ------------------------------------------------------------------ #
    #  Stage 1: SQL 硬過濾
    # ------------------------------------------------------------------ #

    def _stage1_sql_filter(self) -> list[str]:
        """Stage 1: SQL 層硬性過濾，回傳通過的 stock_id 清單。

        條件：
        1. StockInfo.listing_type IN ('上市', '上櫃')
        2. StockInfo.security_type = 'stock'（OR NULL 向後相容）
        3. 近 365 天 DailyPrice 可用交易日數 >= min_available_days
        4. 最新收盤價 > min_close
        """
        cutoff_365 = date.today() - timedelta(days=365)

        try:
            with get_session() as session:
                # 子查詢 1：近 365 天可用交易日數
                days_subq = (
                    select(
                        DailyPrice.stock_id,
                        func.count(func.distinct(DailyPrice.date)).label("avail_days"),
                    )
                    .where(DailyPrice.date >= cutoff_365)
                    .group_by(DailyPrice.stock_id)
                    .subquery()
                )

                # 子查詢 2：每股最新日期
                latest_date_subq = (
                    select(
                        DailyPrice.stock_id,
                        func.max(DailyPrice.date).label("latest_date"),
                    )
                    .group_by(DailyPrice.stock_id)
                    .subquery()
                )

                # 子查詢 3：最新收盤價
                close_subq = (
                    select(
                        DailyPrice.stock_id,
                        DailyPrice.close.label("latest_close"),
                    )
                    .join(
                        latest_date_subq,
                        (DailyPrice.stock_id == latest_date_subq.c.stock_id)
                        & (DailyPrice.date == latest_date_subq.c.latest_date),
                    )
                    .subquery()
                )

                # 主查詢：JOIN StockInfo + 兩個子查詢
                stmt = (
                    select(StockInfo.stock_id)
                    .join(days_subq, StockInfo.stock_id == days_subq.c.stock_id)
                    .join(close_subq, StockInfo.stock_id == close_subq.c.stock_id)
                    .where(StockInfo.listing_type.in_(self.config.listing_types))
                    .where(days_subq.c.avail_days >= self.config.min_available_days)
                    .where(close_subq.c.latest_close > self.config.min_close)
                )

                # security_type 過濾（NULL fallback 保持向後相容）
                if self.config.security_type:
                    stmt = stmt.where(
                        (StockInfo.security_type == self.config.security_type) | StockInfo.security_type.is_(None)
                    )

                rows = session.execute(stmt).all()
                return [r[0] for r in rows]

        except Exception:
            logger.exception("UniverseFilter Stage 1 SQL 失敗，回傳空清單")
            return []

    # ------------------------------------------------------------------ #
    #  Stage 2: 流動性過濾
    # ------------------------------------------------------------------ #

    def _stage2_liquidity_filter(self, stage1_ids: list[str]) -> list[str]:
        """Stage 2: 讀取近 5 日 turnover，過濾流動性不足的股票。

        優先從 DailyFeature（turnover_ma5）讀取；
        若 DailyFeature 無資料，fallback 從 DailyPrice 計算。
        """
        cutoff_5d = date.today() - timedelta(days=10)  # 多抓幾天以應對假日

        # 嘗試從 DailyFeature 讀取（有最新特徵快取時）
        df_feature = self._load_feature_turnover(stage1_ids)
        if not df_feature.empty and "turnover_ma5" in df_feature.columns:
            latest = df_feature.dropna(subset=["turnover_ma5"])
            # 用 turnover_ma5 作為 5 日均值代理
            avg5_map = latest.set_index("stock_id")["turnover_ma5"]
            # DailyFeature 中無 min_turnover_5d，使用 0.33×avg5 作為保守估計
            conservative_min = avg5_map * 0.33
            mask = (avg5_map >= self.config.avg_turnover_5d_min) & (conservative_min >= self.config.min_turnover_5d_min)
            # 雙窗口確認：若有 turnover_ma20，額外驗證中期流動性穩定（防短暫放量誘多）
            # NaN = DailyFeature 欄位剛新增尚未重算 → 跳過該股的 ma20 門檻
            if "turnover_ma20" in df_feature.columns:
                ma20_map = latest.set_index("stock_id")["turnover_ma20"]
                ma20_map = ma20_map.reindex(avg5_map.index)
                ma20_ok = ma20_map.isna() | (ma20_map >= self.config.turnover_ma20_min)
                mask = mask & ma20_ok
            return list(avg5_map[mask].index)

        # Fallback: 從 DailyPrice 計算（多抓 30 天以計算 turnover_ma20）
        cutoff_30d = date.today() - timedelta(days=30)
        try:
            with get_session() as session:
                rows = session.execute(
                    select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.turnover)
                    .where(DailyPrice.date >= cutoff_30d)
                    .where(DailyPrice.stock_id.in_(stage1_ids))
                ).all()
        except Exception:
            logger.exception("UniverseFilter Stage 2 讀取 DailyPrice 失敗")
            return stage1_ids  # 失敗時不過濾

        if not rows:
            logger.warning("UniverseFilter Stage 2: 無 turnover 資料，跳過流動性過濾")
            return stage1_ids

        df_full = pd.DataFrame(rows, columns=["stock_id", "date", "turnover"])
        df_full = df_full.sort_values(["stock_id", "date"])

        # 計算每股 turnover_ma20（20 日滾動均值）
        df_full["turnover_ma20"] = df_full.groupby("stock_id")["turnover"].transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )
        # 取最新一日的 turnover_ma20 值（每股一筆）
        latest_ma20 = df_full.groupby("stock_id")["turnover_ma20"].last().reset_index()

        # 只取最近 5 日做 avg5/min5 計算，並附帶 ma20 欄位
        df_5d = df_full.groupby("stock_id").tail(5).copy()
        df_5d = df_5d.drop(columns=["turnover_ma20"], errors="ignore")
        df_5d = df_5d.merge(latest_ma20, on="stock_id", how="left")

        return filter_liquidity(df_5d, self.config)

    def _load_feature_turnover(self, stock_ids: list[str]) -> pd.DataFrame:
        """從 DailyFeature 讀取最新一日的 turnover_ma5 與 turnover_ma20。"""
        try:
            with get_session() as session:
                latest_date_subq = (
                    select(func.max(DailyFeature.date)).where(DailyFeature.stock_id.in_(stock_ids)).scalar_subquery()
                )
                rows = session.execute(
                    select(DailyFeature.stock_id, DailyFeature.turnover_ma5, DailyFeature.turnover_ma20)
                    .where(DailyFeature.stock_id.in_(stock_ids))
                    .where(DailyFeature.date == latest_date_subq)
                ).all()
                return pd.DataFrame(rows, columns=["stock_id", "turnover_ma5", "turnover_ma20"])
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Stage 3: 趨勢動能過濾
    # ------------------------------------------------------------------ #

    def _stage3_trend_filter(self, stage2_ids: list[str]) -> list[str]:
        """Stage 3: 趨勢動能過濾。

        優先從 DailyFeature 讀取 ma60/volume_ma20；
        若無，fallback 從 DailyPrice 即時計算。
        """
        if self.config.trend_ma is None:
            return stage2_ids  # Value/Dividend Scanner 跳過

        # 嘗試從 DailyFeature 讀取
        df_feat = self._load_feature_trend(stage2_ids)
        if not df_feat.empty:
            return filter_trend(df_feat, self.config)

        # Fallback: 從 DailyPrice 即時計算
        ma_period = self.config.trend_ma
        lookback = max(ma_period + 10, 30)
        cutoff = date.today() - timedelta(days=lookback)

        try:
            with get_session() as session:
                rows = session.execute(
                    select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close, DailyPrice.volume)
                    .where(DailyPrice.date >= cutoff)
                    .where(DailyPrice.stock_id.in_(stage2_ids))
                ).all()
        except Exception:
            logger.exception("UniverseFilter Stage 3 讀取 DailyPrice 失敗")
            return stage2_ids

        if not rows:
            return stage2_ids

        df = pd.DataFrame(rows, columns=["stock_id", "date", "close", "volume"])
        df = df.sort_values(["stock_id", "date"])

        # 向量化計算 MA 與量比
        ma_col = f"ma{ma_period}"
        df[ma_col] = df.groupby("stock_id")["close"].transform(
            lambda s: s.rolling(ma_period, min_periods=max(ma_period // 2, 5)).mean()
        )
        df["volume_ma20"] = df.groupby("stock_id")["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())

        return filter_trend(df, self.config)

    def _load_feature_trend(self, stock_ids: list[str]) -> pd.DataFrame:
        """從 DailyFeature 讀取最新一日的趨勢相關欄位。"""
        try:
            with get_session() as session:
                latest_date_subq = (
                    select(func.max(DailyFeature.date)).where(DailyFeature.stock_id.in_(stock_ids)).scalar_subquery()
                )
                rows = session.execute(
                    select(
                        DailyFeature.stock_id,
                        DailyFeature.date,
                        DailyFeature.close,
                        DailyFeature.volume,
                        DailyFeature.ma20,
                        DailyFeature.ma60,
                        DailyFeature.volume_ma20,
                    )
                    .where(DailyFeature.stock_id.in_(stock_ids))
                    .where(DailyFeature.date == latest_date_subq)
                ).all()
                return pd.DataFrame(
                    rows,
                    columns=["stock_id", "date", "close", "volume", "ma20", "ma60", "volume_ma20"],
                )
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Candidate Memory
    # ------------------------------------------------------------------ #

    def _load_candidate_memory(self, mode: str, stage1_ids: list[str]) -> set[str]:
        """讀取前 N 天同模式的 DiscoveryRecord，回傳驗證過的 stock_id 集合。

        只回傳仍通過 Stage 1 的 stock_ids（確保 close > 10 等基本條件）。
        """
        if self.config.candidate_memory_days <= 0:
            return set()

        cutoff = date.today() - timedelta(days=self.config.candidate_memory_days)
        try:
            with get_session() as session:
                rows = session.execute(
                    select(DiscoveryRecord.stock_id)
                    .where(DiscoveryRecord.scan_date >= cutoff)
                    .where(DiscoveryRecord.mode == mode)
                ).all()
            memory_ids = {r[0] for r in rows}
        except Exception:
            return set()

        # 只保留仍通過 Stage 1 的 IDs
        valid = memory_ids & set(stage1_ids)
        if valid:
            logger.debug("UniverseFilter Memory: 找到 %d 支昨日候選（有效 %d 支）", len(memory_ids), len(valid))
        return valid
