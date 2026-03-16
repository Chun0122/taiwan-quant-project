"""產業輪動分析引擎 — 計算法人動能 + 價格動能，找出熱門產業。

使用方式：
    analyzer = IndustryRotationAnalyzer()
    sector_df = analyzer.rank_sectors()
    top_stocks = analyzer.top_stocks_from_hot_sectors(sector_df)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import func, select

from src.data.database import get_effective_watchlist, get_session, init_db
from src.data.schema import DailyPrice, InstitutionalInvestor, StockInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 法人分類常數
# ---------------------------------------------------------------------------
# InstitutionalInvestor.name 欄位可能出現的字串值（中英文並存）
_FOREIGN_NAMES: frozenset[str] = frozenset({"外資", "外資及陸資", "Foreign_Investor"})
_TRUST_NAMES: frozenset[str] = frozenset({"投信", "Investment_Trust"})

# ---------------------------------------------------------------------------
# 模組層級 Rank Cache（日期 TTL，同日重複呼叫直接命中）
# ---------------------------------------------------------------------------
_RANK_CACHE: dict[tuple, pd.DataFrame] = {}
_RANK_CACHE_DATE: date | None = None


def _rank_cache_key(
    watchlist: list[str],
    lookback_days: int,
    momentum_days: int,
    trust_weight: float,
    foreign_weight: float,
) -> tuple:
    """生成 rank_sectors 快取 key。"""
    return (
        frozenset(watchlist),
        lookback_days,
        momentum_days,
        round(trust_weight, 3),
        round(foreign_weight, 3),
    )


def _get_cached_rank(key: tuple) -> pd.DataFrame | None:
    """取得快取結果；日期不同時自動清除舊快取。"""
    global _RANK_CACHE, _RANK_CACHE_DATE  # noqa: PLW0603
    today = date.today()
    if _RANK_CACHE_DATE != today:
        _RANK_CACHE.clear()
        _RANK_CACHE_DATE = today
    return _RANK_CACHE.get(key)


def _set_cached_rank(key: tuple, df: pd.DataFrame) -> None:
    """寫入快取。"""
    _RANK_CACHE[key] = df


# ---------------------------------------------------------------------------
# 模組層級純函數
# ---------------------------------------------------------------------------


def compute_sector_relative_strength(
    stock_ids: list[str],
    df_price: pd.DataFrame,
    industry_map: dict[str, str],
    lookback_days: int = 20,
    threshold: float = 0.08,
    bonus: float = 0.03,
    use_dynamic_threshold: bool = True,
) -> pd.DataFrame:
    """計算個股相對同產業中位數的相對強度加成（±3%）。

    計算邏輯：
      1. 計算每支股票近 lookback_days 個交易日的報酬率
      2. 計算每個產業的中位數報酬率
      3. 動態門檻（use_dynamic_threshold=True，預設）：
         threshold_i = max(1.5 × σ_i, 0.05)，依各產業波動度自適應
         靜態門檻（use_dynamic_threshold=False）：使用 threshold 參數
      4. 若個股報酬率高於中位數超過門檻 → +bonus
         若個股報酬率低於中位數超過門檻 → -bonus

    Args:
        stock_ids: 候選股代號清單
        df_price: DataFrame（需含 stock_id, date, close 欄位）
        industry_map: {stock_id: industry_category} 對照表
        lookback_days: 計算報酬率的回看天數（預設 20 日）
        threshold: 靜態門檻，use_dynamic_threshold=False 時使用（預設 0.08 = ±8 個百分點）
        bonus: 加成幅度（預設 0.03 = ±3%）
        use_dynamic_threshold: True 時改用 1.5×σ 動態門檻（預設 True）

    Returns:
        DataFrame(stock_id, relative_strength_bonus)  值域 {-bonus, 0.0, +bonus}
    """
    default = pd.DataFrame({"stock_id": stock_ids, "relative_strength_bonus": [0.0] * len(stock_ids)})

    if df_price.empty or not stock_ids:
        return default

    # --- 向量化計算每支股票的近期報酬率 ---
    df = df_price[df_price["stock_id"].isin(stock_ids)].copy()
    df = df.sort_values(["stock_id", "date"])
    # 標記倒數第 N 筆（0 = 最新）
    df["_ridx"] = df.groupby("stock_id").cumcount(ascending=False)
    df_w = df[df["_ridx"] < lookback_days]
    counts = df_w.groupby("stock_id")["close"].count()
    first_close = df_w.groupby("stock_id")["close"].first()
    last_close = df_w.groupby("stock_id")["close"].last()
    valid_mask = (counts >= 2) & (first_close > 0)
    valid_ids = valid_mask[valid_mask].index.tolist()

    if not valid_ids:
        return default

    ret_series = (last_close[valid_ids] - first_close[valid_ids]) / first_close[valid_ids]
    returns: dict[str, float] = ret_series.to_dict()

    # 計算每個產業的中位數報酬率（與 σ，供動態門檻使用）
    sector_rets: dict[str, list[float]] = {}
    for sid, ret in returns.items():
        industry = industry_map.get(sid, "未分類")
        sector_rets.setdefault(industry, []).append(ret)

    sector_median: dict[str, float] = {ind: float(pd.Series(rets).median()) for ind, rets in sector_rets.items()}

    # 動態門檻：1.5 × σ，最小 0.05，樣本不足時 fallback 靜態門檻
    if use_dynamic_threshold:
        sector_threshold: dict[str, float] = {}
        for ind, rets in sector_rets.items():
            if len(rets) >= 3:
                sigma = float(pd.Series(rets).std())
                sector_threshold[ind] = max(1.5 * sigma, 0.05)
            else:
                sector_threshold[ind] = threshold
    else:
        sector_threshold = {}  # 空字典 → 全部使用靜態 threshold

    # 映射每支股票的加成
    records = []
    for sid in stock_ids:
        industry = industry_map.get(sid, "未分類")
        median = sector_median.get(industry)
        ret = returns.get(sid)
        thr = sector_threshold.get(industry, threshold)

        if ret is None or median is None:
            bonus_val = 0.0
        else:
            diff = ret - median
            if diff > thr:
                bonus_val = bonus
            elif diff < -thr:
                bonus_val = -bonus
            else:
                bonus_val = 0.0

        records.append({"stock_id": sid, "relative_strength_bonus": bonus_val})

    return pd.DataFrame(records)


def compute_flow_acceleration_from_df(
    df: pd.DataFrame,
    recent_days: int = 5,
    base_days: int = 15,
    ema_span: int = 0,
) -> pd.Series:
    """計算各產業法人資金流加速度（純函數，不依賴 DB）。

    acceleration = (近 recent_days 日平均淨買超) - (前 base_days 日平均淨買超)

    當加速度轉正代表資金「剛轉入」該產業，具前瞻性；
    當加速度轉負代表資金「剛流出」該產業。
    使用混合法人分數時（30% level + 70% acceleration），可降低追高風險。

    Args:
        df: DataFrame，需含欄位 stock_id、date、net、industry。
            date 可為 datetime.date 或 pd.Timestamp。
        recent_days: 近期窗口交易日數（預設 5）
        base_days: 前期窗口交易日數（預設 15）
        ema_span: EMA 平滑窗口（0 = 不平滑，建議值 3）。
            > 0 時先對每支股票的 net 序列做 EMA(span) 平滑，再計算加速度，
            可降低單日法人大買賣的噪聲干擾，代價是 ~1 日訊號延遲。

    Returns:
        Series: index=industry, value=acceleration（浮點數）。
        若資料不足 (recent_days + base_days) 個交易日，回傳全零 Series。
    """
    if df.empty:
        return pd.Series(dtype=float)

    all_industries = df["industry"].unique()

    # EMA 平滑（P3）：對每支股票的 net 序列做指數移動平均後再算加速度
    if ema_span > 0:
        df = df.sort_values(["stock_id", "date"]).copy()
        df["net"] = df.groupby("stock_id")["net"].transform(lambda s: s.ewm(span=ema_span, adjust=False).mean())

    # 取得全部交易日（降序）
    unique_dates = sorted(df["date"].unique(), reverse=True)
    total_needed = recent_days + base_days

    if len(unique_dates) < total_needed:
        # 資料不足 → 回傳全零，避免 NaN 污染下游
        return pd.Series(0.0, index=all_industries)

    # 近期窗口：最近 recent_days 個交易日
    recent_cutoff = unique_dates[recent_days - 1]
    df_recent = df[df["date"] >= recent_cutoff]

    # 前期窗口：倒數第 (recent_days+1) 到 (recent_days+base_days) 個交易日
    base_newest = unique_dates[recent_days]
    base_oldest = unique_dates[total_needed - 1]
    df_base = df[(df["date"] >= base_oldest) & (df["date"] <= base_newest)]

    def _sector_avg(sub_df: pd.DataFrame, n_days: int) -> pd.Series:
        return sub_df.groupby("industry")["net"].sum() / n_days

    recent_avg = _sector_avg(df_recent, recent_days)
    base_avg = _sector_avg(df_base, base_days)

    # 對齊產業索引後相減
    recent_filled = recent_avg.reindex(all_industries, fill_value=0.0)
    base_filled = base_avg.reindex(all_industries, fill_value=0.0)
    return recent_filled - base_filled


class IndustryRotationAnalyzer:
    """產業輪動分析器。"""

    def __init__(
        self,
        watchlist: list[str] | None = None,
        lookback_days: int = 20,
        momentum_days: int = 60,
    ) -> None:
        self.watchlist = watchlist or get_effective_watchlist()
        self.lookback_days = lookback_days
        self.momentum_days = momentum_days
        init_db()

    def get_industry_map(self) -> dict[str, str]:
        """從 StockInfo 表取得股票→產業對照。

        Returns:
            {stock_id: industry_category}，不在表中的股票標為「未分類」
        """
        with get_session() as session:
            rows = session.execute(select(StockInfo.stock_id, StockInfo.industry_category)).all()

        db_map = {r[0]: r[1] or "未分類" for r in rows}

        result = {}
        for sid in self.watchlist:
            result[sid] = db_map.get(sid, "未分類")

        return result

    def compute_sector_flow_acceleration(
        self,
        recent_days: int = 5,
        base_days: int = 15,
        ema_span: int = 0,
    ) -> pd.Series:
        """計算各產業法人資金流加速度（二階導數）。

        查詢 DB 並呼叫模組層級純函數 `compute_flow_acceleration_from_df()`。

        Args:
            recent_days: 近期窗口交易日數（預設 5）
            base_days: 前期窗口交易日數（預設 15）
            ema_span: EMA 平滑窗口，傳入 compute_flow_acceleration_from_df（預設 0，不平滑）

        Returns:
            Series: index=industry_category, value=acceleration
            若資料不足回傳全零 Series。
        """
        industry_map = self.get_industry_map()
        end_date = date.today()
        # 查詢足夠的歷史：(recent + base) × 2 個日曆天 + 10 天 buffer
        start_date = end_date - timedelta(days=(recent_days + base_days) * 2 + 10)

        with get_session() as session:
            rows = (
                session.execute(
                    select(InstitutionalInvestor)
                    .where(InstitutionalInvestor.stock_id.in_(self.watchlist))
                    .where(InstitutionalInvestor.date >= start_date)
                    .order_by(InstitutionalInvestor.date.desc())
                )
                .scalars()
                .all()
            )

        if not rows:
            return pd.Series(dtype=float)

        df = pd.DataFrame([{"stock_id": r.stock_id, "date": r.date, "net": r.net} for r in rows])
        df["industry"] = df["stock_id"].map(industry_map).fillna("未分類")

        return compute_flow_acceleration_from_df(df, recent_days=recent_days, base_days=base_days, ema_span=ema_span)

    def compute_sector_institutional_flow(
        self,
        trust_weight: float = 0.7,
        foreign_weight: float = 0.3,
    ) -> pd.DataFrame:
        """計算各產業法人資金流向（投信加權）。

        將 InstitutionalInvestor 按法人類型分類後計算加權淨買超：
        - 投信（trust_weight）：波段操作主力，輪動訊號較強
        - 外資（foreign_weight）：偏長期，輪動敏感度較低
        - 其他（剩餘權重）：自營商等

        Args:
            trust_weight: 投信在加權中的比重（預設 0.7）
            foreign_weight: 外資在加權中的比重（預設 0.3）

        Returns:
            DataFrame: [industry, total_net, stock_count, avg_net_per_stock]
            total_net 為加權後的產業淨買超代理值（可直接用於排名）
        """
        industry_map = self.get_industry_map()
        end_date = date.today()
        start_date = end_date - timedelta(days=self.lookback_days * 2)

        with get_session() as session:
            rows = (
                session.execute(
                    select(InstitutionalInvestor)
                    .where(InstitutionalInvestor.stock_id.in_(self.watchlist))
                    .where(InstitutionalInvestor.date >= start_date)
                    .order_by(InstitutionalInvestor.date.desc())
                )
                .scalars()
                .all()
            )

        if not rows:
            return pd.DataFrame(columns=["industry", "total_net", "stock_count", "avg_net_per_stock"])

        df = pd.DataFrame([{"stock_id": r.stock_id, "date": r.date, "name": r.name, "net": r.net} for r in rows])

        # 只取最近 lookback_days 個交易日
        unique_dates = sorted(df["date"].unique(), reverse=True)
        if len(unique_dates) > self.lookback_days:
            cutoff = unique_dates[self.lookback_days - 1]
            df = df[df["date"] >= cutoff]

        # 加入產業分類
        df["industry"] = df["stock_id"].map(industry_map).fillna("未分類")

        # 計算法人類型加權係數（P1b）
        other_weight = max(1.0 - trust_weight - foreign_weight, 0.0)

        def _inst_weight(name: str) -> float:
            if name in _TRUST_NAMES:
                return trust_weight
            if name in _FOREIGN_NAMES:
                return foreign_weight
            return other_weight

        df["inst_weight"] = df["name"].apply(_inst_weight)
        df["weighted_net"] = df["net"] * df["inst_weight"]

        # 按產業 + 股票加總（使用加權 net）
        stock_net = df.groupby(["industry", "stock_id"])["weighted_net"].sum().reset_index()
        stock_net.columns = ["industry", "stock_id", "net"]
        sector_flow = (
            stock_net.groupby("industry")
            .agg(
                total_net=("net", "sum"),
                stock_count=("stock_id", "nunique"),
            )
            .reset_index()
        )
        sector_flow["avg_net_per_stock"] = sector_flow["total_net"] / sector_flow["stock_count"]
        sector_flow = sector_flow.sort_values("total_net", ascending=False).reset_index(drop=True)

        return sector_flow

    def compute_sector_price_momentum(self) -> pd.DataFrame:
        """計算各產業價格動能（期間漲跌幅均值）。

        使用向量化運算取代逐股迴圈，效能顯著提升（P2b）。

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
        df = df.sort_values(["stock_id", "date"])

        # --- 向量化：取每支股票最近 momentum_days 筆 ---
        df["_ridx"] = df.groupby("stock_id").cumcount(ascending=False)
        df_w = df[df["_ridx"] < self.momentum_days]

        counts = df_w.groupby("stock_id")["close"].count()
        first_close = df_w.groupby("stock_id")["close"].first()
        last_close = df_w.groupby("stock_id")["close"].last()

        # 有效股票：≥2 筆 且 期初收盤 > 0
        valid = (counts >= 2) & (first_close > 0)
        valid_stocks = valid[valid].index

        if valid_stocks.empty:
            return pd.DataFrame(columns=["industry", "avg_return_pct", "stock_count"])

        ret_pct = ((last_close[valid_stocks] - first_close[valid_stocks]) / first_close[valid_stocks]) * 100
        ret_pct.name = "return_pct"
        df_ret = ret_pct.reset_index()
        df_ret.columns = ["stock_id", "return_pct"]
        df_ret["industry"] = df_ret["stock_id"].map(lambda s: industry_map.get(s, "未分類"))

        sector_momentum = (
            df_ret.groupby("industry")
            .agg(
                avg_return_pct=("return_pct", "mean"),
                stock_count=("stock_id", "nunique"),
            )
            .reset_index()
        )
        sector_momentum = sector_momentum.sort_values("avg_return_pct", ascending=False).reset_index(drop=True)

        return sector_momentum

    def rank_sectors(
        self,
        inst_weight: float = 0.5,
        momentum_weight: float = 0.5,
        use_flow_acceleration: bool = True,
        accel_weight: float = 0.7,
        trust_weight: float = 0.7,
        foreign_weight: float = 0.3,
        ema_span: int = 0,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """綜合排名各產業（法人動能 + 價格動能）。

        改進點：
        - P1a：使用 Percentile Rank（rank(pct=True)）取代 Min-Max 標準化，
          避免離群值（如半導體法人大量買超）壓縮其他產業分數
        - P1b：法人分數依投信/外資分類加權（trust_weight / foreign_weight），
          投信是台股輪動主力，預設給予更高權重
        - P2a：加入 module-level 日期 TTL 快取，同日重複呼叫直接命中

        Args:
            inst_weight: 法人分數權重（相對於動能分數）
            momentum_weight: 價格動能權重（相對於法人分數）
            use_flow_acceleration: 是否啟用「資金流加速度」訊號（預設 True）。
                啟用時法人分數 = (1 - accel_weight) × level + accel_weight × acceleration，
                可捕捉資金「剛轉入」初升段，降低追高風險。
            accel_weight: 加速度信號在法人子分數中的權重（預設 0.7）；
                level 信號權重 = 1 - accel_weight。
            trust_weight: 投信在法人加權中的比重（預設 0.7）
            foreign_weight: 外資在法人加權中的比重（預設 0.3）
            ema_span: 加速度 EMA 平滑窗口（0 = 不平滑，預設 0）
            use_cache: 是否使用 module-level 快取（預設 True）

        Returns:
            DataFrame: [rank, industry, sector_score, institutional_score,
                        momentum_score, total_net, avg_return_pct, stock_count]
        """
        # P2a：快取命中直接返回
        if use_cache:
            cache_key = _rank_cache_key(
                self.watchlist, self.lookback_days, self.momentum_days, trust_weight, foreign_weight
            )
            cached = _get_cached_rank(cache_key)
            if cached is not None:
                logger.debug("rank_sectors() 命中快取（watchlist=%d 支）", len(self.watchlist))
                return cached

        flow_df = self.compute_sector_institutional_flow(trust_weight=trust_weight, foreign_weight=foreign_weight)
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

        # --- P1a：Percentile Rank 取代 Min-Max ---
        # rank(pct=True) 將每個產業映射到 [0, 1] 的百分位，
        # 對離群值（如半導體法人買超遠大於其他產業）具有天然 robust 性。
        def _pct_rank(series: pd.Series) -> pd.Series:
            if series.nunique() == 1:
                return pd.Series(0.5, index=series.index)
            return series.rank(pct=True)

        # 法人子分數：level 或 level + acceleration 混合
        level_score = _pct_rank(merged["total_net"])
        if use_flow_acceleration:
            accel_series = self.compute_sector_flow_acceleration(ema_span=ema_span)
            if not accel_series.empty:
                # 對齊 merged 的 industry 索引
                accel_aligned = accel_series.reindex(merged["industry"].values, fill_value=0.0)
                accel_aligned.index = merged.index
                accel_score = _pct_rank(accel_aligned)
                inst_sub_score = (1.0 - accel_weight) * level_score + accel_weight * accel_score
            else:
                inst_sub_score = level_score
        else:
            inst_sub_score = level_score

        merged["institutional_score"] = inst_sub_score
        merged["momentum_score"] = _pct_rank(merged["avg_return_pct"])
        merged["sector_score"] = (
            inst_weight * merged["institutional_score"] + momentum_weight * merged["momentum_score"]
        )

        merged = merged.sort_values("sector_score", ascending=False).reset_index(drop=True)
        merged["rank"] = range(1, len(merged) + 1)

        cols = [
            "rank",
            "industry",
            "sector_score",
            "institutional_score",
            "momentum_score",
            "total_net",
            "avg_return_pct",
            "stock_count",
        ]
        result = merged[[c for c in cols if c in merged.columns]]

        # P2a：寫入快取
        if use_cache:
            _set_cached_rank(cache_key, result)

        return result

    def compute_sector_scores_for_stocks(
        self,
        stock_ids: list[str],
        bonus_range: float = 0.05,
    ) -> pd.DataFrame:
        """計算每支股票所屬產業的熱度加成分數。

        用 rank_sectors() 取得產業排名，將排名百分位線性映射到
        [-bonus_range, +bonus_range]。排名第 1 → +bonus_range，
        排名最末 → -bonus_range。

        Args:
            stock_ids: 要計算的股票代號清單
            bonus_range: 加成幅度上限（預設 0.05 = ±5%）

        Returns:
            DataFrame(stock_id, sector_bonus)
        """
        default = pd.DataFrame({"stock_id": stock_ids, "sector_bonus": [0.0] * len(stock_ids)})

        if not stock_ids:
            return default

        # 用傳入的 stock_ids 計算產業熱度
        analyzer = IndustryRotationAnalyzer(
            watchlist=stock_ids,
            lookback_days=self.lookback_days,
            momentum_days=self.momentum_days,
        )
        sector_df = analyzer.rank_sectors()

        if sector_df.empty:
            return default

        # 從 StockInfo 取得 stock_id → industry_category 對照
        with get_session() as session:
            rows = session.execute(
                select(StockInfo.stock_id, StockInfo.industry_category).where(StockInfo.stock_id.in_(stock_ids))
            ).all()
        industry_map = {r[0]: (r[1] or "未分類") for r in rows}

        # 將產業排名轉為 bonus：rank 1 → +bonus_range, 最末 → -bonus_range
        n_sectors = len(sector_df)
        sector_bonus_map = {}
        for _, row in sector_df.iterrows():
            rank = row["rank"]
            if n_sectors == 1:
                bonus = 0.0
            else:
                # 線性映射：rank=1 → +bonus_range, rank=n → -bonus_range
                bonus = bonus_range * (1 - 2 * (rank - 1) / (n_sectors - 1))
            sector_bonus_map[row["industry"]] = bonus

        # 映射到每支股票
        records = []
        for sid in stock_ids:
            industry = industry_map.get(sid, "未分類")
            bonus = sector_bonus_map.get(industry, 0.0)
            records.append({"stock_id": sid, "sector_bonus": bonus})

        return pd.DataFrame(records)

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
                select(StockInfo.stock_id, StockInfo.stock_name).where(StockInfo.stock_id.in_(hot_stocks))
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
            records.append(
                {
                    "industry": industry_map[sid],
                    "stock_id": sid,
                    "stock_name": name_map.get(sid, ""),
                    "close": price_map.get(sid, 0),
                    "foreign_net_sum": inst_map.get(sid, 0),
                }
            )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # 按產業內外資淨買超排序，每產業取 top_n
        result_parts = []
        for ind in hot_industries:
            sector_stocks = df[df["industry"] == ind].sort_values("foreign_net_sum", ascending=False).head(top_n).copy()
            sector_stocks["rank_in_sector"] = range(1, len(sector_stocks) + 1)
            result_parts.append(sector_stocks)

        if not result_parts:
            return pd.DataFrame()

        return pd.concat(result_parts, ignore_index=True)
