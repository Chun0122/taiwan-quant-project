"""策略回測排名引擎 — 批次回測 watchlist × strategies，找出最佳配對。

使用方式：
    engine = StrategyRankEngine()
    df = engine.run()
    engine.print_summary(df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import select, func

from src.config import settings
from src.data.database import get_session, init_db
from src.data.schema import DailyPrice
from src.strategy import STRATEGY_REGISTRY
from src.backtest.engine import BacktestEngine, RiskConfig

logger = logging.getLogger(__name__)

FAST_STRATEGIES = [
    "sma_cross", "rsi_threshold", "bb_breakout",
    "macd_cross", "buy_and_hold", "multi_factor",
]
ML_STRATEGIES = ["ml_random_forest", "ml_xgboost", "ml_logistic"]

# 各策略需要的最少資料天數
STRATEGY_MIN_DATA = {
    "sma_cross": 60,
    "rsi_threshold": 30,
    "bb_breakout": 30,
    "macd_cross": 40,
    "buy_and_hold": 10,
    "multi_factor": 60,
    "ml_random_forest": 120,
    "ml_xgboost": 120,
    "ml_logistic": 120,
}


@dataclass
class RankResult:
    """單次回測排名結果。"""

    stock_id: str
    strategy_name: str
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float | None = None
    max_drawdown: float = 0.0
    win_rate: float | None = None
    total_trades: int = 0
    sortino_ratio: float | None = None
    profit_factor: float | None = None
    error: str | None = None


class StrategyRankEngine:
    """策略回測排名引擎。"""

    def __init__(
        self,
        watchlist: list[str] | None = None,
        strategy_names: list[str] | None = None,
        metric: str = "sharpe",
        start_date: str | None = None,
        end_date: str | None = None,
        min_trades: int = 3,
    ) -> None:
        self.watchlist = watchlist or settings.fetcher.watchlist
        self.strategy_names = strategy_names or FAST_STRATEGIES
        self.metric = metric
        self.start_date = start_date or settings.fetcher.default_start_date
        self.end_date = end_date or date.today().isoformat()
        self.min_trades = min_trades
        init_db()

    def run(self) -> pd.DataFrame:
        """執行 watchlist × strategies 批次回測並排名。

        Returns:
            DataFrame: [rank, stock_id, strategy_name, total_return, annual_return,
                        sharpe_ratio, max_drawdown, win_rate, total_trades,
                        sortino_ratio, profit_factor]
        """
        results: list[RankResult] = []
        total_combos = len(self.watchlist) * len(self.strategy_names)
        done = 0

        for stock_id in self.watchlist:
            for strategy_name in self.strategy_names:
                done += 1
                logger.info(
                    "[%d/%d] 回測 %s × %s",
                    done, total_combos, stock_id, strategy_name,
                )

                # 先檢查資料量
                ok, reason = self._check_data_available(stock_id, strategy_name)
                if not ok:
                    logger.info("  跳過: %s", reason)
                    results.append(RankResult(
                        stock_id=stock_id,
                        strategy_name=strategy_name,
                        error=reason,
                    ))
                    continue

                result = self._run_single(stock_id, strategy_name)
                results.append(result)

        return self._rank_results(results)

    def _run_single(self, stock_id: str, strategy_name: str) -> RankResult:
        """執行單次回測。"""
        try:
            if strategy_name not in STRATEGY_REGISTRY:
                return RankResult(
                    stock_id=stock_id,
                    strategy_name=strategy_name,
                    error=f"未知策略: {strategy_name}",
                )

            strategy_cls = STRATEGY_REGISTRY[strategy_name]
            strategy = strategy_cls(
                stock_id=stock_id,
                start_date=self.start_date,
                end_date=self.end_date,
            )

            engine = BacktestEngine(strategy, risk_config=RiskConfig())
            bt_result = engine.run()

            return RankResult(
                stock_id=stock_id,
                strategy_name=strategy_name,
                total_return=bt_result.total_return,
                annual_return=bt_result.annual_return,
                sharpe_ratio=bt_result.sharpe_ratio,
                max_drawdown=bt_result.max_drawdown,
                win_rate=bt_result.win_rate,
                total_trades=bt_result.total_trades,
                sortino_ratio=bt_result.sortino_ratio,
                profit_factor=bt_result.profit_factor,
            )

        except Exception as e:
            logger.debug("[%s × %s] 回測失敗: %s", stock_id, strategy_name, e)
            return RankResult(
                stock_id=stock_id,
                strategy_name=strategy_name,
                error=str(e),
            )

    def _check_data_available(self, stock_id: str, strategy_name: str) -> tuple[bool, str]:
        """快速檢查資料天數是否足夠。"""
        min_days = STRATEGY_MIN_DATA.get(strategy_name, 30)

        with get_session() as session:
            count = session.execute(
                select(func.count())
                .select_from(DailyPrice)
                .where(DailyPrice.stock_id == stock_id)
                .where(DailyPrice.date >= self.start_date)
                .where(DailyPrice.date <= self.end_date)
            ).scalar()

        if count < min_days:
            return False, f"資料不足 ({count}/{min_days} 天)"
        return True, ""

    def _rank_results(self, results: list[RankResult]) -> pd.DataFrame:
        """過濾、排序結果。"""
        records = []
        for r in results:
            if r.error:
                continue
            if r.total_trades < self.min_trades:
                continue
            records.append({
                "stock_id": r.stock_id,
                "strategy_name": r.strategy_name,
                "total_return": r.total_return,
                "annual_return": r.annual_return,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "total_trades": r.total_trades,
                "sortino_ratio": r.sortino_ratio,
                "profit_factor": r.profit_factor,
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # 按指定 metric 排序
        metric_col = self.metric
        if metric_col == "sharpe":
            metric_col = "sharpe_ratio"

        if metric_col in df.columns:
            df = df.sort_values(metric_col, ascending=False, na_position="last")
        else:
            df = df.sort_values("total_return", ascending=False)

        df = df.reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        cols = [
            "rank", "stock_id", "strategy_name", "total_return", "annual_return",
            "sharpe_ratio", "max_drawdown", "win_rate", "total_trades",
            "sortino_ratio", "profit_factor",
        ]
        return df[[c for c in cols if c in df.columns]]

    def print_summary(self, df: pd.DataFrame, top_n: int = 20) -> None:
        """CLI 格式化輸出。"""
        if df.empty:
            print("無符合條件的回測結果")
            return

        display = df.head(top_n)
        print(f"\n{'=' * 80}")
        print(f"策略回測排名 (top {min(top_n, len(df))}/{len(df)})  排序: {self.metric}")
        print(f"{'=' * 80}")
        print(f"{'Rank':>4}  {'Stock':>6}  {'Strategy':<16}  {'Return':>8}  "
              f"{'Annual':>8}  {'Sharpe':>7}  {'MDD':>7}  {'WinR':>6}  {'Trades':>6}")
        print(f"{'─' * 80}")

        for _, row in display.iterrows():
            sharpe = f"{row['sharpe_ratio']:.2f}" if pd.notna(row.get('sharpe_ratio')) else "N/A"
            win_r = f"{row['win_rate']:.1f}" if pd.notna(row.get('win_rate')) else "N/A"
            print(
                f"{int(row['rank']):>4}  {row['stock_id']:>6}  {row['strategy_name']:<16}  "
                f"{row['total_return']:>7.2f}%  {row['annual_return']:>7.2f}%  "
                f"{sharpe:>7}  {row['max_drawdown']:>6.2f}%  {win_r:>6}  {int(row['total_trades']):>6}"
            )
