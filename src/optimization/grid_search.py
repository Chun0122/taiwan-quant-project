"""Grid Search 參數優化器 — 窮舉參數組合並排名。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.strategy import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


# 各策略預設參數網格
DEFAULT_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "sma_cross": {
        "fast": [5, 10, 15, 20, 25, 30],
        "slow": [20, 30, 40, 50, 60],
    },
    "rsi_threshold": {
        "oversold": [20, 25, 30, 35, 40],
        "overbought": [60, 65, 70, 75, 80],
    },
    "bb_breakout": {
        "period": [10, 15, 20, 25, 30],
        "std_dev": [1, 2, 3],
    },
    "macd_cross": {
        "fast": [8, 12, 16],
        "slow": [20, 26, 32],
        "signal": [7, 9, 11],
    },
}


@dataclass
class OptimizationResult:
    """單次優化結果。"""

    params: dict
    total_return: float
    annual_return: float
    sharpe_ratio: float | None
    max_drawdown: float
    win_rate: float | None
    total_trades: int


class GridSearchOptimizer:
    """Grid Search 參數優化器。

    用法::

        optimizer = GridSearchOptimizer(
            strategy_name="sma_cross",
            stock_id="2330",
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
        results = optimizer.run()
        optimizer.print_top_n(results, n=10)
    """

    def __init__(
        self,
        strategy_name: str,
        stock_id: str,
        start_date: str,
        end_date: str,
        param_grid: dict[str, list] | None = None,
        config: BacktestConfig | None = None,
    ) -> None:
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(f"未知策略: {strategy_name}")

        self.strategy_cls = STRATEGY_REGISTRY[strategy_name]
        self.strategy_name = strategy_name
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self.param_grid = param_grid or DEFAULT_PARAM_GRIDS.get(strategy_name, {})
        self.config = config or BacktestConfig()

        if not self.param_grid:
            raise ValueError(f"策略 {strategy_name} 無可用參數網格")

    def run(self) -> list[OptimizationResult]:
        """執行網格搜尋，回傳所有結果（依 Sharpe Ratio 降序排列）。"""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(product(*param_values))

        logger.info(
            "開始網格搜尋: %s | %s | %d 組參數",
            self.strategy_name, self.stock_id, len(combinations),
        )

        results: list[OptimizationResult] = []

        for i, combo in enumerate(combinations, 1):
            params = dict(zip(param_names, combo))

            # 過濾無效組合（例如 sma_cross 的 fast >= slow）
            if self.strategy_name == "sma_cross" and params.get("fast", 0) >= params.get("slow", 999):
                continue

            try:
                strategy = self.strategy_cls(
                    stock_id=self.stock_id,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    **params,
                )
                engine = BacktestEngine(strategy, self.config)
                bt = engine.run()

                result = OptimizationResult(
                    params=params,
                    total_return=bt.total_return,
                    annual_return=bt.annual_return,
                    sharpe_ratio=bt.sharpe_ratio,
                    max_drawdown=bt.max_drawdown,
                    win_rate=bt.win_rate,
                    total_trades=bt.total_trades,
                )
                results.append(result)

                logger.info(
                    "[%d/%d] %s | 報酬=%.2f%% | Sharpe=%s | MDD=%.2f%%",
                    i, len(combinations), params,
                    result.total_return,
                    f"{result.sharpe_ratio:.4f}" if result.sharpe_ratio else "N/A",
                    result.max_drawdown,
                )
            except Exception as e:
                logger.warning("[%d/%d] %s | 回測失敗: %s", i, len(combinations), params, e)

        # 排序：Sharpe Ratio 降序
        results.sort(key=lambda r: r.sharpe_ratio if r.sharpe_ratio is not None else -999, reverse=True)

        logger.info("網格搜尋完成，共 %d 組有效結果", len(results))
        return results

    @staticmethod
    def print_top_n(results: list[OptimizationResult], n: int = 10) -> None:
        """列印前 N 名結果。"""
        print("\n" + "=" * 70)
        print(f"Top {min(n, len(results))} 參數組合（依 Sharpe Ratio 排序）")
        print("=" * 70)

        for i, r in enumerate(results[:n], 1):
            print(f"\n#{i}")
            print(f"  參數:         {r.params}")
            print(f"  總報酬率:     {r.total_return:>10.2f}%")
            print(f"  年化報酬率:   {r.annual_return:>10.2f}%")
            sharpe_str = f"{r.sharpe_ratio:.4f}" if r.sharpe_ratio is not None else "N/A"
            print(f"  Sharpe Ratio: {sharpe_str:>10}")
            print(f"  最大回撤:     {r.max_drawdown:>10.2f}%")
            win_str = f"{r.win_rate:.2f}%" if r.win_rate is not None else "N/A"
            print(f"  勝率:         {win_str:>10}")
            print(f"  交易次數:     {r.total_trades:>10}")

    @staticmethod
    def export_to_csv(results: list[OptimizationResult], filepath: str) -> None:
        """匯出結果到 CSV。"""
        records = []
        for r in results:
            row = {**r.params}
            row["total_return"] = r.total_return
            row["annual_return"] = r.annual_return
            row["sharpe_ratio"] = r.sharpe_ratio
            row["max_drawdown"] = r.max_drawdown
            row["win_rate"] = r.win_rate
            row["total_trades"] = r.total_trades
            records.append(row)

        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"結果已匯出至: {filepath}")
