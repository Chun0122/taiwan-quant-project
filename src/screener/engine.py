"""多因子篩選引擎 — 掃描 watchlist 內所有股票，計算因子分數並排名。

使用方式：
    screener = MultiFactorScreener(watchlist=["2330", "2317"])
    results = screener.scan()
    print(results)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select, func

from src.config import settings
from src.data.database import get_session
from src.data.schema import (
    DailyPrice,
    TechnicalIndicator,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
)
from src.screener.factors import FACTOR_REGISTRY

logger = logging.getLogger(__name__)


class MultiFactorScreener:
    """多因子選股篩選器。"""

    def __init__(
        self,
        watchlist: list[str] | None = None,
        lookback_days: int = 5,
    ) -> None:
        """初始化篩選器。

        Args:
            watchlist: 掃描的股票清單，None 時使用設定檔 watchlist
            lookback_days: 回溯天數，用於計算需要時間序列的因子
        """
        self.watchlist = watchlist or settings.fetcher.watchlist
        self.lookback_days = lookback_days

    def _load_snapshot(self, stock_id: str) -> pd.DataFrame:
        """載入單一股票最近 N 天的全維度資料快照。

        合併流程：
        1. DailyPrice → OHLCV
        2. TechnicalIndicator pivot → sma/rsi/macd/bb
        3. InstitutionalInvestor pivot by name → 外資/投信/自營商 net
        4. MarginTrading → margin_balance, short_balance
        5. MonthlyRevenue 最近一筆 → yoy_growth, mom_growth

        Returns:
            完整寬表 DataFrame，index=date
        """
        end_date = date.today()
        # 多抓一些天數以確保有足夠交易日資料
        start_date = end_date - timedelta(days=self.lookback_days * 3)

        with get_session() as session:
            # 1. 日K線
            prices = session.execute(
                select(DailyPrice)
                .where(DailyPrice.stock_id == stock_id)
                .where(DailyPrice.date >= start_date)
                .order_by(DailyPrice.date.desc())
                .limit(self.lookback_days)
            ).scalars().all()

            if not prices:
                return pd.DataFrame()

            df = pd.DataFrame([
                {
                    "date": r.date,
                    "close": r.close,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "volume": r.volume,
                }
                for r in prices
            ]).set_index("date").sort_index()

            date_range_start = df.index.min()
            date_range_end = df.index.max()

            # 2. 技術指標 pivot
            indicators = session.execute(
                select(TechnicalIndicator)
                .where(TechnicalIndicator.stock_id == stock_id)
                .where(TechnicalIndicator.date >= date_range_start)
                .where(TechnicalIndicator.date <= date_range_end)
            ).scalars().all()

            if indicators:
                df_ind = pd.DataFrame([
                    {"date": r.date, "name": r.name, "value": r.value}
                    for r in indicators
                ])
                df_wide = df_ind.pivot_table(index="date", columns="name", values="value")
                df = df.join(df_wide, how="left")

            # 3. 三大法人 pivot by name
            institutions = session.execute(
                select(InstitutionalInvestor)
                .where(InstitutionalInvestor.stock_id == stock_id)
                .where(InstitutionalInvestor.date >= date_range_start)
                .where(InstitutionalInvestor.date <= date_range_end)
            ).scalars().all()

            if institutions:
                df_inst = pd.DataFrame([
                    {"date": r.date, "name": r.name, "net": r.net}
                    for r in institutions
                ])
                inst_pivot = df_inst.pivot_table(
                    index="date", columns="name", values="net", aggfunc="sum"
                )
                # 重新命名為統一欄位名
                rename_map = {
                    "Foreign_Investor": "foreign_net",
                    "Investment_Trust": "trust_net",
                    "Dealer_self": "dealer_net",
                    # 中文名稱（FinMind 可能回傳中文）
                    "外資": "foreign_net",
                    "投信": "trust_net",
                    "自營商": "dealer_net",
                    # 可能的其他外資名稱
                    "外資及陸資": "foreign_net",
                }
                inst_pivot = inst_pivot.rename(columns=rename_map)
                # 只保留已知欄位
                known_cols = ["foreign_net", "trust_net", "dealer_net"]
                inst_pivot = inst_pivot[[c for c in known_cols if c in inst_pivot.columns]]
                df = df.join(inst_pivot, how="left")

            # 4. 融資融券
            margins = session.execute(
                select(MarginTrading)
                .where(MarginTrading.stock_id == stock_id)
                .where(MarginTrading.date >= date_range_start)
                .where(MarginTrading.date <= date_range_end)
            ).scalars().all()

            if margins:
                df_margin = pd.DataFrame([
                    {
                        "date": r.date,
                        "margin_balance": r.margin_balance,
                        "short_balance": r.short_balance,
                    }
                    for r in margins
                ]).set_index("date")
                df = df.join(df_margin, how="left")

            # 5. 月營收（取最近一筆）
            latest_rev = session.execute(
                select(MonthlyRevenue)
                .where(MonthlyRevenue.stock_id == stock_id)
                .order_by(MonthlyRevenue.date.desc())
                .limit(1)
            ).scalars().first()

            if latest_rev:
                df["yoy_growth"] = latest_rev.yoy_growth
                df["mom_growth"] = latest_rev.mom_growth
                df["revenue"] = latest_rev.revenue

        return df

    def scan(
        self,
        factors: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """掃描所有股票，計算因子分數並排名。

        Args:
            factors: 要使用的因子名稱清單，None 表示全部啟用
            weights: 因子權重，None 表示均等權重

        Returns:
            DataFrame，columns: stock_id, close, volume, rsi_14, macd,
            sma_20, foreign_net, trust_net, dealer_net, margin_balance,
            short_balance, yoy_growth, factor_score
        """
        active_factors = factors or list(FACTOR_REGISTRY.keys())
        if weights is None:
            weights = {f: 1.0 / len(active_factors) for f in active_factors}

        results = []

        for stock_id in self.watchlist:
            try:
                snapshot = self._load_snapshot(stock_id)
                if snapshot.empty:
                    logger.warning("[%s] 無資料，跳過", stock_id)
                    continue

                # 以最新一筆為主要資料
                latest = snapshot.iloc[-1].to_dict()
                latest["stock_id"] = stock_id

                # 計算各因子分數
                score = 0.0
                factor_details = {}
                for fname in active_factors:
                    if fname not in FACTOR_REGISTRY:
                        continue
                    finfo = FACTOR_REGISTRY[fname]
                    func = finfo["func"]
                    params = finfo["params"]
                    result = func(snapshot, **params)

                    # 取最新一天的結果
                    hit = bool(result.iloc[-1]) if len(result) > 0 else False
                    factor_details[fname] = hit
                    if hit:
                        score += weights.get(fname, 0)

                latest["factor_score"] = round(score, 4)

                # 加入各因子命中狀態
                for fname, hit in factor_details.items():
                    latest[f"f_{fname}"] = hit

                results.append(latest)

            except Exception:
                logger.exception("[%s] 篩選失敗", stock_id)

        if not results:
            return pd.DataFrame()

        df_result = pd.DataFrame(results)

        # 整理欄位順序
        priority_cols = [
            "stock_id", "close", "volume", "rsi_14", "macd", "sma_20",
            "foreign_net", "trust_net", "dealer_net",
            "margin_balance", "short_balance", "yoy_growth", "factor_score",
        ]
        ordered = [c for c in priority_cols if c in df_result.columns]
        remaining = [c for c in df_result.columns if c not in ordered]
        df_result = df_result[ordered + remaining]

        # 依因子分數排序
        df_result = df_result.sort_values("factor_score", ascending=False).reset_index(drop=True)
        return df_result

    def scan_with_conditions(
        self,
        conditions: list[str],
        require_all: bool = True,
    ) -> pd.DataFrame:
        """依指定條件篩選（AND 或 OR 模式）。

        Args:
            conditions: 因子名稱清單
            require_all: True=所有條件都要符合(AND), False=任一符合(OR)

        Returns:
            篩選後的 DataFrame
        """
        df = self.scan(factors=conditions)
        if df.empty:
            return df

        # 根據因子命中欄位篩選
        factor_cols = [f"f_{c}" for c in conditions if f"f_{c}" in df.columns]
        if not factor_cols:
            return df

        if require_all:
            mask = df[factor_cols].all(axis=1)
        else:
            mask = df[factor_cols].any(axis=1)

        return df[mask].reset_index(drop=True)
