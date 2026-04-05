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

from src.constants import REGIME_UNIVERSE_ADJUSTMENTS
from src.data.database import get_session
from src.data.schema import DailyFeature, DailyPrice, DiscoveryRecord, StockInfo

# Sentinel 值：區分「未傳入」與「明確傳入 None」
_SENTINEL = object()

# Candidate Memory 門檻衰減乘數（days_ago → 流動性門檻乘數）
# Day 1: ×0.8（寬鬆，容易留下）→ Day 2: ×0.9 → Day 3: ×1.0（原始門檻）
MEMORY_DECAY: dict[int, float] = {1: 0.8, 2: 0.9, 3: 1.0}

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
    candidate_memory_days: int = 3  # 記憶回溯天數（0 = 停用），越老門檻越嚴
    regime: str | None = None  # bull/bear/sideways/crisis；None = 不調整（由 Scanner 傳入）
    trend_filter_mode: str = "trend_only"  # "trend_only" | "breakout_only" | "trend_or_breakout"

    # 模式覆寫建議（由 Scanner.__init__ 傳入）
    # MomentumScanner: min_close=5.0, min_available_days=30, volume_ratio_min=None
    # SwingScanner:    volume_ratio_min=1.2
    # ValueScanner:    trend_ma=None, volume_ratio_min=None
    # DividendScanner: trend_ma=None, volume_ratio_min=None
    # GrowthScanner:   min_close=5.0, trend_ma=20, volume_ratio_min=2.0
    extra: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────
#  純函數（可獨立測試）
# ────────────────────────────────────────────────────────────────


def filter_liquidity(df_5d: pd.DataFrame, config: UniverseConfig, turnover_multiplier: float = 1.0) -> list[str]:
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
        turnover_multiplier: Regime 門檻乘數（<1 放寬、>1 收緊，預設 1.0 不調整）

    Returns:
        通過流動性門檻的 stock_id 清單
    """
    if df_5d.empty or "turnover" not in df_5d.columns:
        return []

    # 套用 Regime 乘數到門檻值
    eff_avg_min = config.avg_turnover_5d_min * turnover_multiplier
    eff_min_min = config.min_turnover_5d_min * turnover_multiplier
    eff_ma20_min = config.turnover_ma20_min * turnover_multiplier

    grp = df_5d.groupby("stock_id")["turnover"]
    avg5 = grp.mean()
    min5 = grp.min()
    mask = (avg5 >= eff_avg_min) & (min5 >= eff_min_min)

    # 雙窗口確認：若提供 20 日均成交金額，額外驗證中期流動性穩定
    # NaN = 資料不足（新股或 DailyFeature 尚未重算），跳過該股的 ma20 門檻
    if "turnover_ma20" in df_5d.columns:
        ma20 = df_5d.groupby("stock_id")["turnover_ma20"].last()
        ma20 = ma20.reindex(avg5.index)
        ma20_ok = ma20.isna() | (ma20 >= eff_ma20_min)
        mask = mask & ma20_ok

    passed_absolute = list(avg5[mask].index)

    # 相對流動性救援：未通過絕對門檻但 turnover_ratio_5d_20d > 2.0 且 avg5 > 門檻 50%
    # 偵測「突然被市場關注」的股票（平常成交低但近期急升）
    if "turnover_ratio_5d_20d" in df_5d.columns:
        ratio = df_5d.groupby("stock_id")["turnover_ratio_5d_20d"].last()
        failed_ids = avg5.index.difference(passed_absolute)
        if len(failed_ids) > 0:
            half_threshold = eff_avg_min * 0.5
            rescue_ratio = ratio.reindex(failed_ids)
            rescue_avg = avg5.reindex(failed_ids)
            rescue_mask = (rescue_ratio >= 2.0) & (rescue_avg >= half_threshold)
            rescued = list(rescue_mask[rescue_mask].index)
            if rescued:
                logger.info("流動性救援：%d 支股票因相對流動性 > 2x 加入", len(rescued))
                return passed_absolute + rescued

    return passed_absolute


def filter_trend(df_hist: pd.DataFrame, config: UniverseConfig, volume_ratio_override: object = _SENTINEL) -> list[str]:
    """Stage 3 純函數：依趨勢條件過濾，回傳通過的 stock_id 清單。

    過濾條件：
    1. close > MA{trend_ma}（預設 MA60，而非 MA20，可捕捉剛突破的股票）
    2. volume_ratio = volume / volume_ma20 >= volume_ratio_min

    異常排除：近 5 日若有 3 天以上漲跌幅 >= 9.5%（連續漲跌停），排除。

    Args:
        df_hist: 含 [stock_id, date, close, volume, ma60, volume_ma20] 的 DataFrame
                 （若 trend_ma=20 則需 ma20 欄；欄位來自 DailyFeature 或即時計算）
        config: UniverseConfig
        volume_ratio_override: Regime 覆寫量比門檻。
            _SENTINEL = 使用 config.volume_ratio_min（預設）；
            None = 跳過量比過濾；
            float = 使用該值作為門檻。

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

    # 決定有效的量比門檻（Regime 覆寫 > config 預設）
    eff_volume_ratio_min = config.volume_ratio_min if volume_ratio_override is _SENTINEL else volume_ratio_override

    # 取最新一日
    latest_date = df_hist["date"].max()
    latest = df_hist[df_hist["date"] == latest_date].copy()

    # 1. close > MA{N}
    close_ok = latest[ma_col].notna() & (latest["close"] >= latest[ma_col])

    # 2. 量比過濾
    if eff_volume_ratio_min is not None and "volume_ma20" in df_hist.columns:
        vol_ma20 = latest["volume_ma20"].replace(0, float("nan"))
        vol_ratio = latest["volume"] / vol_ma20
        vol_ok = vol_ratio >= eff_volume_ratio_min
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


def filter_trend_breakout(df_hist: pd.DataFrame, config: UniverseConfig) -> list[str]:
    """Type B 突破型過濾：捕捉剛從底部轉強的標的（即使尚未站穩 MA60）。

    條件（全部須滿足）：
    1. close >= ma20（短期趨勢向上）
    2. momentum_20d > 0（正向動能，排除死貓反彈）
    3. close / high_20d >= 0.9（接近 20 日高點，確認是真突破）
    4. volume / volume_ma20 >= 1.5（量能擴張確認）
    5. 異常排除（同 filter_trend：7 日內 3+ 次漲跌停）

    Args:
        df_hist: 含 [stock_id, date, close, volume, ma20, momentum_20d,
                 high_20d, volume_ma20] 的 DataFrame
        config: UniverseConfig

    Returns:
        通過突破型過濾的 stock_id 清單
    """
    if df_hist.empty:
        return []

    # 必要欄位檢查
    required = {"ma20", "volume_ma20"}
    if not required.issubset(df_hist.columns):
        logger.warning("filter_trend_breakout: 缺少欄位 %s，跳過突破過濾", required - set(df_hist.columns))
        return []

    # 取最新一日
    latest_date = df_hist["date"].max()
    latest = df_hist[df_hist["date"] == latest_date].copy()

    # 1. close >= ma20
    close_above_ma20 = latest["ma20"].notna() & (latest["close"] >= latest["ma20"])

    # 2. momentum_20d > 0
    if "momentum_20d" in latest.columns:
        momentum_ok = latest["momentum_20d"].fillna(0) > 0
    else:
        momentum_ok = pd.Series(True, index=latest.index)

    # 3. close / high_20d >= 0.9（接近近期高點）
    if "high_20d" in latest.columns:
        high_20d = latest["high_20d"].replace(0, float("nan"))
        near_high = (latest["close"] / high_20d) >= 0.9
        near_high = near_high.fillna(True)  # 無資料時不阻擋
    else:
        near_high = pd.Series(True, index=latest.index)

    # 4. volume / volume_ma20 >= 1.5（量能擴張）
    vol_ma20 = latest["volume_ma20"].replace(0, float("nan"))
    vol_ratio = latest["volume"] / vol_ma20
    vol_ok = vol_ratio >= 1.5

    # 5. 異常排除：近 7 天內出現 3 次以上漲跌停
    recent_7 = df_hist[df_hist["date"] >= latest_date - timedelta(days=7)]
    pct_chg = recent_7.sort_values("date").groupby("stock_id")["close"].pct_change().abs()
    limit_days = pct_chg >= 0.095
    consecutive_limit = limit_days.groupby(level=0).sum() if not limit_days.empty else pd.Series(dtype=float)
    anomaly_ids = set(consecutive_limit[consecutive_limit >= 3].index) if not consecutive_limit.empty else set()
    not_anomaly = ~latest["stock_id"].isin(anomaly_ids)

    pass_mask = close_above_ma20 & momentum_ok & near_high & vol_ok & not_anomaly
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

        # Regime 自適應：查表取得門檻調整參數
        adjustments = REGIME_UNIVERSE_ADJUSTMENTS.get(self.config.regime or "", {})
        self._turnover_multiplier = adjustments.get("turnover_multiplier", 1.0)
        self._volume_ratio_override = adjustments.get("volume_ratio_override", _SENTINEL)
        if self.config.regime:
            logger.info(
                "UniverseFilter Regime=%s: turnover_mult=%.1f, vol_ratio_override=%s",
                self.config.regime,
                self._turnover_multiplier,
                self._volume_ratio_override,
            )

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

        # Stage 2: 流動性過濾（套用 Regime 乘數）
        stage2_ids = self._stage2_liquidity_filter(stage1_ids)
        stats["total_after_liquidity"] = len(stage2_ids)
        logger.info("UniverseFilter Stage 2: 流動性過濾後 %d 支", len(stage2_ids))

        if not stage2_ids:
            return [], {**stats, "total_after_trend": 0, "from_memory": 0, "final_candidates": 0}

        # Stage 3: 趨勢動能過濾（套用 Regime 量比覆寫）
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

        # Candidate Memory: 漸進衰減記憶（降低每日換股率）
        # Day 1: 門檻 ×0.8（寬鬆）→ Day 2: ×0.9 → Day 3: ×1.0（原始門檻）→ Day 4+: 淘汰
        memory_added = 0
        if self.config.candidate_memory_days > 0:
            memory_map = self._load_candidate_memory(mode, stage1_ids)
            stage3_set = set(stage3_ids)
            stage2_set = set(stage2_ids)
            turnover_cache = getattr(self, "_turnover_cache", {})
            t_mult = getattr(self, "_turnover_multiplier", 1.0)
            base_threshold = self.config.avg_turnover_5d_min * t_mult

            for sid, days_ago in memory_map.items():
                if sid in stage3_set:
                    continue  # 今天已入選，不需記憶加持
                # 漸進衰減：越老的記憶門檻越嚴
                decay = MEMORY_DECAY.get(days_ago, None)
                if decay is None:
                    continue  # 超出記憶範圍
                # 檢查是否通過放寬後的流動性門檻
                sid_turnover = turnover_cache.get(sid, 0)
                relaxed_threshold = base_threshold * decay
                if sid_turnover >= relaxed_threshold and sid in stage2_set:
                    stage3_ids.append(sid)
                    memory_added += 1
                elif days_ago <= 1 and sid not in stage2_set:
                    # Day 1 特例：即使不在 Stage 2（無 turnover 快取），只要通過 Stage 1 也加入
                    stage3_ids.append(sid)
                    memory_added += 1

            if memory_added:
                logger.info("UniverseFilter Memory: 加入 %d 支記憶候選（衰減門檻）", memory_added)

        stats["from_memory"] = memory_added
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

        # Regime 門檻乘數
        t_mult = getattr(self, "_turnover_multiplier", 1.0)

        # 嘗試從 DailyFeature 讀取（有最新特徵快取時）
        df_feature = self._load_feature_turnover(stage1_ids)
        if not df_feature.empty and "turnover_ma5" in df_feature.columns:
            latest = df_feature.dropna(subset=["turnover_ma5"])
            # 用 turnover_ma5 作為 5 日均值代理
            avg5_map = latest.set_index("stock_id")["turnover_ma5"]
            # 快取每股 avg turnover 供 Candidate Memory 衰減門檻使用
            self._turnover_cache = avg5_map.to_dict()
            # DailyFeature 中無 min_turnover_5d，使用 0.33×avg5 作為保守估計
            conservative_min = avg5_map * 0.33
            eff_avg_min = self.config.avg_turnover_5d_min * t_mult
            eff_min_min = self.config.min_turnover_5d_min * t_mult
            mask = (avg5_map >= eff_avg_min) & (conservative_min >= eff_min_min)
            # 雙窗口確認：若有 turnover_ma20，額外驗證中期流動性穩定（防短暫放量誘多）
            # NaN = DailyFeature 欄位剛新增尚未重算 → 跳過該股的 ma20 門檻
            if "turnover_ma20" in df_feature.columns:
                ma20_map = latest.set_index("stock_id")["turnover_ma20"]
                ma20_map = ma20_map.reindex(avg5_map.index)
                eff_ma20_min = self.config.turnover_ma20_min * t_mult
                ma20_ok = ma20_map.isna() | (ma20_map >= eff_ma20_min)
                mask = mask & ma20_ok
            passed_absolute = list(avg5_map[mask].index)

            # 相對流動性救援（DailyFeature 快速路徑）
            if "turnover_ratio_5d_20d" in latest.columns:
                ratio_map = latest.set_index("stock_id")["turnover_ratio_5d_20d"]
                ratio_map = ratio_map.reindex(avg5_map.index)
                failed_ids = avg5_map.index.difference(passed_absolute)
                if len(failed_ids) > 0:
                    half_threshold = eff_avg_min * 0.5
                    rescue_ratio = ratio_map.reindex(failed_ids)
                    rescue_avg = avg5_map.reindex(failed_ids)
                    rescue_mask = (rescue_ratio >= 2.0) & (rescue_avg >= half_threshold)
                    rescued = list(rescue_mask[rescue_mask].index)
                    if rescued:
                        logger.info("流動性救援：%d 支股票因相對流動性 > 2x 加入", len(rescued))
                        return passed_absolute + rescued

            return passed_absolute

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

        # 快取每股 avg turnover 供 Candidate Memory 衰減門檻使用
        self._turnover_cache = df_5d.groupby("stock_id")["turnover"].mean().to_dict()

        return filter_liquidity(df_5d, self.config, turnover_multiplier=t_mult)

    def _load_feature_turnover(self, stock_ids: list[str]) -> pd.DataFrame:
        """從 DailyFeature 讀取最新一日的 turnover_ma5、turnover_ma20、turnover_ratio_5d_20d。"""
        try:
            with get_session() as session:
                latest_date_subq = (
                    select(func.max(DailyFeature.date)).where(DailyFeature.stock_id.in_(stock_ids)).scalar_subquery()
                )
                rows = session.execute(
                    select(
                        DailyFeature.stock_id,
                        DailyFeature.turnover_ma5,
                        DailyFeature.turnover_ma20,
                        DailyFeature.turnover_ratio_5d_20d,
                    )
                    .where(DailyFeature.stock_id.in_(stock_ids))
                    .where(DailyFeature.date == latest_date_subq)
                ).all()
                return pd.DataFrame(
                    rows, columns=["stock_id", "turnover_ma5", "turnover_ma20", "turnover_ratio_5d_20d"]
                )
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Stage 3: 趨勢動能過濾
    # ------------------------------------------------------------------ #

    def _stage3_trend_filter(self, stage2_ids: list[str]) -> list[str]:
        """Stage 3: 趨勢動能過濾。

        依 trend_filter_mode 分派：
        - "trend_only"（預設）：原始 Type A 趨勢過濾（close >= MA60）
        - "breakout_only"：Type B 突破型過濾
        - "trend_or_breakout"：Type A ∪ Type B

        優先從 DailyFeature 讀取；若無，fallback 從 DailyPrice 即時計算。
        """
        if self.config.trend_ma is None:
            return stage2_ids  # Value/Dividend Scanner 跳過

        vol_override = getattr(self, "_volume_ratio_override", _SENTINEL)
        mode = self.config.trend_filter_mode

        # 嘗試從 DailyFeature 讀取
        df_feat = self._load_feature_trend(stage2_ids)
        if not df_feat.empty:
            return self._dispatch_trend_filter(df_feat, vol_override, mode)

        # Fallback: 從 DailyPrice 即時計算
        ma_period = self.config.trend_ma
        lookback = max(ma_period + 10, 30)
        cutoff = date.today() - timedelta(days=lookback)

        try:
            with get_session() as session:
                rows = session.execute(
                    select(
                        DailyPrice.stock_id,
                        DailyPrice.date,
                        DailyPrice.high,
                        DailyPrice.close,
                        DailyPrice.volume,
                    )
                    .where(DailyPrice.date >= cutoff)
                    .where(DailyPrice.stock_id.in_(stage2_ids))
                ).all()
        except Exception:
            logger.exception("UniverseFilter Stage 3 讀取 DailyPrice 失敗")
            return stage2_ids

        if not rows:
            return stage2_ids

        df = pd.DataFrame(rows, columns=["stock_id", "date", "high", "close", "volume"])
        df = df.sort_values(["stock_id", "date"])

        # 向量化計算 MA 與量比
        ma_col = f"ma{ma_period}"
        df[ma_col] = df.groupby("stock_id")["close"].transform(
            lambda s: s.rolling(ma_period, min_periods=max(ma_period // 2, 5)).mean()
        )
        df["ma20"] = df.groupby("stock_id")["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
        df["volume_ma20"] = df.groupby("stock_id")["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
        df["momentum_20d"] = df.groupby("stock_id")["close"].transform(lambda s: s.pct_change(20) * 100)
        df["high_20d"] = df.groupby("stock_id")["high"].transform(lambda s: s.rolling(20, min_periods=10).max())

        return self._dispatch_trend_filter(df, vol_override, mode)

    def _dispatch_trend_filter(self, df: pd.DataFrame, vol_override: object, mode: str) -> list[str]:
        """依 trend_filter_mode 分派到 Type A / Type B / 聯集。"""
        if mode == "trend_or_breakout":
            trend_ids = filter_trend(df, self.config, volume_ratio_override=vol_override)
            breakout_ids = filter_trend_breakout(df, self.config)
            combined = list(set(trend_ids) | set(breakout_ids))
            if breakout_only := set(breakout_ids) - set(trend_ids):
                logger.info("Stage 3 突破型加入 %d 支（趨勢型 %d 支）", len(breakout_only), len(trend_ids))
            return combined
        elif mode == "breakout_only":
            return filter_trend_breakout(df, self.config)
        else:  # "trend_only"
            return filter_trend(df, self.config, volume_ratio_override=vol_override)

    def _load_feature_trend(self, stock_ids: list[str]) -> pd.DataFrame:
        """從 DailyFeature 讀取最新一日的趨勢相關欄位（含突破型所需欄位）。"""
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
                        DailyFeature.momentum_20d,
                        DailyFeature.high_20d,
                    )
                    .where(DailyFeature.stock_id.in_(stock_ids))
                    .where(DailyFeature.date == latest_date_subq)
                ).all()
                return pd.DataFrame(
                    rows,
                    columns=[
                        "stock_id",
                        "date",
                        "close",
                        "volume",
                        "ma20",
                        "ma60",
                        "volume_ma20",
                        "momentum_20d",
                        "high_20d",
                    ],
                )
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Candidate Memory
    # ------------------------------------------------------------------ #

    def _load_candidate_memory(self, mode: str, stage1_ids: list[str]) -> dict[str, int]:
        """讀取前 N 天同模式的 DiscoveryRecord，回傳 {stock_id: days_ago} 字典。

        只回傳仍通過 Stage 1 的 stock_ids（確保 close > 10 等基本條件）。
        多日出現的股票取最近的 days_ago（最小值）。
        """
        if self.config.candidate_memory_days <= 0:
            return {}

        cutoff = date.today() - timedelta(days=self.config.candidate_memory_days)
        today = date.today()
        try:
            with get_session() as session:
                rows = session.execute(
                    select(DiscoveryRecord.stock_id, DiscoveryRecord.scan_date)
                    .where(DiscoveryRecord.scan_date >= cutoff)
                    .where(DiscoveryRecord.mode == mode)
                ).all()
        except Exception:
            return {}

        # 計算 days_ago，多日出現取最近的
        stage1_set = set(stage1_ids)
        memory_map: dict[str, int] = {}
        for stock_id, scan_date in rows:
            if stock_id not in stage1_set:
                continue
            days_ago = (today - scan_date).days
            if days_ago <= 0:
                continue  # 今天的記錄不算記憶
            if stock_id not in memory_map or days_ago < memory_map[stock_id]:
                memory_map[stock_id] = days_ago

        if memory_map:
            logger.debug(
                "UniverseFilter Memory: 找到 %d 支候選（有效 %d 支）",
                len(rows),
                len(memory_map),
            )
        return memory_map
