"""Walk-Forward 驗證引擎 — 滾動窗口訓練/測試，避免 ML 策略過擬合。

原理：
  將歷史資料分成多個滾動窗口，每次用 train_window 訓練、test_window 測試，
  依序往前推移，最終將所有 test 期間的交易結果合併計算績效。

  |--- train_1 ---|-- test_1 --|
       |--- train_2 ---|-- test_2 --|
            |--- train_3 ---|-- test_3 --|

Usage::

    wf = WalkForwardEngine(
        strategy_cls=MLStrategy,
        stock_id="2330",
        start_date="2020-01-01",
        end_date="2024-12-31",
        train_window=252,   # 1 年訓練
        test_window=63,     # 1 季測試
    )
    result = wf.run()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, RiskConfig, TradeRecord
from src.backtest.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """單個 fold 的結果。"""

    fold_idx: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    total_return: float
    trades: int
    sharpe_ratio: float | None = None


@dataclass
class WalkForwardResult:
    """Walk-Forward 驗證整體結果。"""

    stock_id: str
    strategy_name: str
    start_date: date
    end_date: date
    train_window: int
    test_window: int
    step_size: int
    total_folds: int

    # 合併所有 test 區間的績效
    total_return: float
    annual_return: float
    sharpe_ratio: float | None
    max_drawdown: float
    win_rate: float | None
    total_trades: int
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    var_95: float | None = None
    cvar_95: float | None = None
    profit_factor: float | None = None

    # 各 fold 詳細
    folds: list[WalkForwardFold] = field(default_factory=list)
    all_trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    # 策略參數
    strategy_params: dict = field(default_factory=dict)


class WalkForwardEngine:
    """Walk-Forward 驗證引擎。"""

    def __init__(
        self,
        strategy_cls: type,
        stock_id: str,
        start_date: str,
        end_date: str,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int | None = None,
        config: BacktestConfig | None = None,
        risk_config: RiskConfig | None = None,
        strategy_params: dict | None = None,
    ) -> None:
        """
        Args:
            strategy_cls: 策略類別
            stock_id: 股票代號
            start_date: 整體起始日期
            end_date: 整體結束日期
            train_window: 訓練窗口天數（交易日）
            test_window: 測試窗口天數（交易日）
            step_size: 每次前進步長（預設=test_window，非重疊）
            config: 回測設定
            risk_config: 風險管理設定
            strategy_params: 策略額外參數
        """
        self.strategy_cls = strategy_cls
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size or test_window
        self.config = config or BacktestConfig()
        self.risk_config = risk_config or RiskConfig()
        self.strategy_params = strategy_params or {}

    def run(self) -> WalkForwardResult:
        """執行 Walk-Forward 驗證。"""
        # 先載入完整資料以取得日期索引
        strategy = self.strategy_cls(
            stock_id=self.stock_id,
            start_date=self.start_date,
            end_date=self.end_date,
            **self.strategy_params,
        )
        full_data = strategy.load_data()

        if full_data.empty:
            raise ValueError(f"[{self.stock_id}] 無交易資料")

        dates = full_data.index.tolist()
        total_days = len(dates)
        min_required = self.train_window + self.test_window

        if total_days < min_required:
            raise ValueError(
                f"資料不足: 需要至少 {min_required} 個交易日 "
                f"(train={self.train_window} + test={self.test_window})，"
                f"目前只有 {total_days} 個交易日"
            )

        # 建立 fold 窗口
        folds: list[WalkForwardFold] = []
        all_trades: list[TradeRecord] = []
        combined_equity: list[float] = []
        capital = self.config.initial_capital

        fold_idx = 0
        start_idx = 0

        while start_idx + self.train_window + self.test_window <= total_days:
            train_start_idx = start_idx
            train_end_idx = start_idx + self.train_window - 1
            test_start_idx = start_idx + self.train_window
            test_end_idx = min(test_start_idx + self.test_window - 1, total_days - 1)

            train_start = dates[train_start_idx]
            train_end = dates[train_end_idx]
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]

            logger.info(
                "[Fold %d] train: %s ~ %s | test: %s ~ %s",
                fold_idx,
                train_start,
                train_end,
                test_start,
                test_end,
            )

            # 建立策略（用整個 train+test 期間的資料）
            fold_strategy = self.strategy_cls(
                stock_id=self.stock_id,
                start_date=str(train_start),
                end_date=str(test_end),
                **self.strategy_params,
            )

            try:
                # 載入資料
                fold_data = fold_strategy.load_data()
                if fold_data.empty:
                    logger.warning("[Fold %d] 無資料，跳過", fold_idx)
                    start_idx += self.step_size
                    fold_idx += 1
                    continue

                # 產生訊號（策略會用 train 部分訓練，test 部分預測）
                signals = fold_strategy.generate_signals(fold_data)

                # 只取 test 期間做回測
                test_data = fold_data.loc[test_start:test_end]
                test_signals = signals.loc[test_start:test_end]

                if test_data.empty:
                    start_idx += self.step_size
                    fold_idx += 1
                    continue

                # 模擬 test 期間交易
                fold_result = self._simulate_fold(test_data, test_signals, capital)

                capital = fold_result["final_capital"]
                folds.append(
                    WalkForwardFold(
                        fold_idx=fold_idx,
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        total_return=fold_result["total_return"],
                        trades=len(fold_result["trades"]),
                        sharpe_ratio=fold_result.get("sharpe_ratio"),
                    )
                )
                all_trades.extend(fold_result["trades"])
                combined_equity.extend(fold_result["equity_curve"])

            except Exception:
                logger.exception("[Fold %d] 回測失敗", fold_idx)

            start_idx += self.step_size
            fold_idx += 1

        if not folds:
            raise ValueError("Walk-Forward 驗證失敗：無有效 fold")

        # 計算合併績效
        metrics = compute_metrics(
            combined_equity,
            all_trades,
            dates[0],
            dates[-1],
            self.config.initial_capital,
        )

        strategy_name = self.strategy_cls.__name__
        # 嘗試取得 name 屬性
        try:
            s = self.strategy_cls(
                stock_id=self.stock_id,
                start_date=self.start_date,
                end_date=self.end_date,
                **self.strategy_params,
            )
            strategy_name = s.name
        except Exception:
            pass

        return WalkForwardResult(
            stock_id=self.stock_id,
            strategy_name=strategy_name,
            start_date=dates[0],
            end_date=dates[-1],
            train_window=self.train_window,
            test_window=self.test_window,
            step_size=self.step_size,
            total_folds=len(folds),
            total_return=metrics["total_return"],
            annual_return=metrics["annual_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            total_trades=len(all_trades),
            sortino_ratio=metrics["sortino_ratio"],
            calmar_ratio=metrics["calmar_ratio"],
            var_95=metrics["var_95"],
            cvar_95=metrics["cvar_95"],
            profit_factor=metrics["profit_factor"],
            folds=folds,
            all_trades=all_trades,
            equity_curve=combined_equity,
            strategy_params=self.strategy_params,
        )

    def _simulate_fold(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float,
    ) -> dict:
        """模擬單個 fold 的 test 期間交易。"""
        capital = initial_capital
        position = 0
        entry_price = 0.0
        entry_date = None
        trades: list[TradeRecord] = []
        equity_curve: list[float] = []
        has_raw = "raw_close" in data.columns

        for dt in data.index:
            close = data.loc[dt, "close"]
            raw_close = data.loc[dt, "raw_close"] if has_raw else close
            signal = signals.get(dt, 0)

            if signal == 1 and position == 0:
                buy_price = raw_close * (1 + self.config.slippage)
                commission = capital * self.config.commission_rate
                available = capital - commission
                shares = int(available // buy_price)
                if shares > 0:
                    cost = shares * buy_price + shares * buy_price * self.config.commission_rate
                    capital -= cost
                    position = shares
                    entry_price = buy_price
                    entry_date = dt

            elif signal == -1 and position > 0:
                sell_price = raw_close * (1 - self.config.slippage)
                revenue = position * sell_price
                commission = revenue * self.config.commission_rate
                tax = revenue * self.config.tax_rate
                capital += revenue - commission - tax

                pnl = (revenue - commission - tax) - position * entry_price
                ret_pct = (sell_price / entry_price - 1) * 100

                trades.append(
                    TradeRecord(
                        entry_date=entry_date,
                        entry_price=round(entry_price, 2),
                        exit_date=dt,
                        exit_price=round(sell_price, 2),
                        shares=position,
                        pnl=round(pnl, 2),
                        return_pct=round(ret_pct, 2),
                        exit_reason="signal",
                    )
                )
                position = 0
                entry_price = 0.0
                entry_date = None

            equity_curve.append(capital + position * raw_close)

        # 強制平倉
        if position > 0:
            last_close = data.iloc[-1]["raw_close"] if has_raw else data.iloc[-1]["close"]
            sell_price = last_close * (1 - self.config.slippage)
            revenue = position * sell_price
            commission = revenue * self.config.commission_rate
            tax = revenue * self.config.tax_rate
            capital += revenue - commission - tax

            pnl = (revenue - commission - tax) - position * entry_price
            ret_pct = (sell_price / entry_price - 1) * 100

            trades.append(
                TradeRecord(
                    entry_date=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_date=data.index[-1],
                    exit_price=round(sell_price, 2),
                    shares=position,
                    pnl=round(pnl, 2),
                    return_pct=round(ret_pct, 2),
                    exit_reason="force_close",
                )
            )
            position = 0
            equity_curve[-1] = capital

        fold_return = (capital / initial_capital - 1) * 100

        # 計算 fold Sharpe
        sharpe = None
        if len(equity_curve) > 1:
            eq = np.array(equity_curve)
            daily_rets = np.diff(eq) / eq[:-1]
            if np.std(daily_rets) > 0:
                sharpe = round(np.mean(daily_rets) / np.std(daily_rets) * math.sqrt(252), 4)

        return {
            "final_capital": capital,
            "total_return": round(fold_return, 2),
            "trades": trades,
            "equity_curve": equity_curve,
            "sharpe_ratio": sharpe,
        }
