"""回測引擎 — 模擬歷史交易並計算績效指標。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.strategy.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回測參數（符合台股實際費用）。"""

    initial_capital: float = 1_000_000      # 初始資金
    commission_rate: float = 0.001425       # 手續費 0.1425%
    tax_rate: float = 0.003                 # 交易稅 0.3%（賣出時）
    slippage: float = 0.0005                # 滑價 0.05%


@dataclass
class TradeRecord:
    """單筆交易記錄。"""

    entry_date: date
    entry_price: float
    exit_date: date | None = None
    exit_price: float | None = None
    shares: int = 0
    pnl: float = 0.0
    return_pct: float = 0.0


@dataclass
class BacktestResultData:
    """回測結果。"""

    stock_id: str
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_return: float          # %
    annual_return: float         # %
    sharpe_ratio: float | None
    max_drawdown: float          # %
    win_rate: float | None       # %
    total_trades: int
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class BacktestEngine:
    """回測引擎。

    使用全倉進出策略：收到買入訊號時以全部資金買入，收到賣出訊號時全部賣出。
    """

    def __init__(self, strategy: Strategy, config: BacktestConfig | None = None) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()

    def run(self) -> BacktestResultData:
        """執行回測，回傳績效結果。"""
        data = self.strategy.load_data()
        if data.empty:
            raise ValueError(f"[{self.strategy.stock_id}] 無交易資料")

        signals = self.strategy.generate_signals(data)

        capital = self.config.initial_capital
        position = 0        # 持有股數
        entry_price = 0.0
        entry_date = None
        trades: list[TradeRecord] = []
        equity_curve: list[float] = []

        for dt in data.index:
            close = data.loc[dt, "close"]
            signal = signals.get(dt, 0)

            # --- 買入 ---
            if signal == 1 and position == 0:
                buy_price = close * (1 + self.config.slippage)
                commission = capital * self.config.commission_rate
                available = capital - commission
                # 台股以 1000 股為一張，這裡以股為單位簡化
                shares = int(available // buy_price)
                if shares > 0:
                    cost = shares * buy_price + shares * buy_price * self.config.commission_rate
                    capital -= cost
                    position = shares
                    entry_price = buy_price
                    entry_date = dt

            # --- 賣出 ---
            elif signal == -1 and position > 0:
                sell_price = close * (1 - self.config.slippage)
                revenue = position * sell_price
                commission = revenue * self.config.commission_rate
                tax = revenue * self.config.tax_rate
                net_revenue = revenue - commission - tax
                capital += net_revenue

                pnl = net_revenue - position * entry_price
                ret_pct = (sell_price / entry_price - 1) * 100

                trades.append(TradeRecord(
                    entry_date=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_date=dt,
                    exit_price=round(sell_price, 2),
                    shares=position,
                    pnl=round(pnl, 2),
                    return_pct=round(ret_pct, 2),
                ))
                position = 0
                entry_price = 0.0
                entry_date = None

            # 記錄每日權益
            mark_to_market = capital + position * close
            equity_curve.append(mark_to_market)

        # 若回測結束時仍持有部位，以最後收盤價平倉
        if position > 0:
            last_close = data.iloc[-1]["close"]
            sell_price = last_close * (1 - self.config.slippage)
            revenue = position * sell_price
            commission = revenue * self.config.commission_rate
            tax = revenue * self.config.tax_rate
            net_revenue = revenue - commission - tax
            capital += net_revenue

            pnl = net_revenue - position * entry_price
            ret_pct = (sell_price / entry_price - 1) * 100

            trades.append(TradeRecord(
                entry_date=entry_date,
                entry_price=round(entry_price, 2),
                exit_date=data.index[-1],
                exit_price=round(sell_price, 2),
                shares=position,
                pnl=round(pnl, 2),
                return_pct=round(ret_pct, 2),
            ))
            position = 0
            equity_curve[-1] = capital

        final_capital = capital
        metrics = self._compute_metrics(
            equity_curve, trades, data.index[0], data.index[-1]
        )

        return BacktestResultData(
            stock_id=self.strategy.stock_id,
            strategy_name=self.strategy.name,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.config.initial_capital,
            final_capital=round(final_capital, 2),
            total_return=metrics["total_return"],
            annual_return=metrics["annual_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            total_trades=len(trades),
            trades=trades,
            equity_curve=equity_curve,
        )

    def _compute_metrics(
        self,
        equity_curve: list[float],
        trades: list[TradeRecord],
        start: date,
        end: date,
    ) -> dict:
        """計算績效指標。"""
        initial = self.config.initial_capital
        final = equity_curve[-1] if equity_curve else initial

        # 總報酬率
        total_return = (final / initial - 1) * 100

        # 年化報酬率
        days = (end - start).days
        years = days / 365.25 if days > 0 else 1
        if final > 0 and initial > 0 and years > 0:
            annual_return = ((final / initial) ** (1 / years) - 1) * 100
        else:
            annual_return = 0.0

        # Sharpe Ratio (以每日報酬率計算，假設無風險利率 = 0)
        sharpe_ratio = None
        if len(equity_curve) > 1:
            eq = np.array(equity_curve)
            daily_returns = np.diff(eq) / eq[:-1]
            if np.std(daily_returns) > 0:
                sharpe_ratio = round(
                    np.mean(daily_returns) / np.std(daily_returns) * math.sqrt(252), 4
                )

        # 最大回撤
        max_drawdown = 0.0
        if equity_curve:
            peak = equity_curve[0]
            for val in equity_curve:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak * 100
                if dd > max_drawdown:
                    max_drawdown = dd

        # 勝率
        win_rate = None
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            win_rate = round(wins / len(trades) * 100, 2)

        return {
            "total_return": round(total_return, 2),
            "annual_return": round(annual_return, 2),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": round(max_drawdown, 2),
            "win_rate": win_rate,
        }
