"""投資組合回測引擎 — 多股票同時回測，共用資金池。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.backtest.engine import (
    BacktestConfig,
    RiskConfig,
)
from src.strategy.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """投資組合配置。"""

    allocation_method: str = "equal_weight"  # "equal_weight" | "custom" | "risk_parity" | "mean_variance"
    weights: dict[str, float] | None = None  # stock_id → 權重（custom 時使用）
    max_position_pct: float = 0.5  # 單股最大持倉比例


@dataclass
class PortfolioTradeRecord:
    """投資組合交易記錄（含 stock_id）。"""

    stock_id: str
    entry_date: date
    entry_price: float
    exit_date: date | None = None
    exit_price: float | None = None
    shares: int = 0
    pnl: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = "signal"


@dataclass
class PortfolioResultData:
    """投資組合回測結果。"""

    strategy_name: str
    stock_ids: list[str]
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
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
    allocation_method: str = "equal_weight"
    trades: list[PortfolioTradeRecord] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    per_stock_returns: dict[str, float] = field(default_factory=dict)


class PortfolioBacktestEngine:
    """投資組合回測引擎。

    對多支股票使用相同策略，共用資金池，各股獨立持倉。
    """

    def __init__(
        self,
        strategies: list[Strategy],
        config: BacktestConfig | None = None,
        risk_config: RiskConfig | None = None,
        portfolio_config: PortfolioConfig | None = None,
    ) -> None:
        self.strategies = strategies
        self.config = config or BacktestConfig()
        self.risk_config = risk_config or RiskConfig()
        self.portfolio_config = portfolio_config or PortfolioConfig()

    def run(self) -> PortfolioResultData:
        """執行投資組合回測。"""
        # 1. 載入每支股票的資料與訊號
        stock_data: dict[str, pd.DataFrame] = {}
        stock_signals: dict[str, dict] = {}
        strategy_name = ""

        for strategy in self.strategies:
            sid = strategy.stock_id
            strategy_name = strategy.name

            data = strategy.load_data()
            if data.empty:
                logger.warning("[%s] 無交易資料，跳過", sid)
                continue

            stock_data[sid] = data
            stock_signals[sid] = strategy.generate_signals(data)

        if not stock_data:
            raise ValueError("所有股票均無交易資料")

        stock_ids = list(stock_data.keys())

        # 2. 建立統一日期軸
        all_dates = sorted(set().union(*(df.index for df in stock_data.values())))

        # 3. 計算配置權重
        weights = self._compute_weights(stock_ids, stock_data)

        # 4. 逐日模擬
        capital = self.config.initial_capital
        positions: dict[str, int] = {sid: 0 for sid in stock_ids}
        entry_prices: dict[str, float] = {sid: 0.0 for sid in stock_ids}
        entry_dates: dict[str, date | None] = {sid: None for sid in stock_ids}
        peak_since_entry: dict[str, float] = {sid: 0.0 for sid in stock_ids}
        trades: list[PortfolioTradeRecord] = []
        equity_curve: list[float] = []
        per_stock_pnl: dict[str, float] = {sid: 0.0 for sid in stock_ids}

        # 除權息資料（每支股票的）
        stock_dividends: dict[str, pd.DataFrame | None] = {}
        for strategy in self.strategies:
            sid = strategy.stock_id
            if sid in stock_data:
                stock_dividends[sid] = getattr(strategy, "_dividends", None)

        for dt in all_dates:
            # 每股處理
            for sid in stock_ids:
                data = stock_data[sid]
                if dt not in data.index:
                    continue

                has_raw = "raw_close" in data.columns
                close = data.loc[dt, "close"]
                high = data.loc[dt, "high"]
                low = data.loc[dt, "low"]
                raw_close = data.loc[dt, "raw_close"] if has_raw else close
                raw_high = data.loc[dt, "raw_high"] if has_raw else high
                raw_low = data.loc[dt, "raw_low"] if has_raw else low
                signal = stock_signals[sid].get(dt, 0)

                # --- 除權息處理 ---
                divs = stock_dividends.get(sid)
                if divs is not None and not divs.empty and positions[sid] > 0:
                    if dt in divs.index:
                        cash_div = divs.loc[dt, "cash_dividend"]
                        stock_div = divs.loc[dt, "stock_dividend"]
                        if cash_div > 0:
                            capital += cash_div * positions[sid]
                        if stock_div > 0:
                            new_shares = int(positions[sid] * stock_div / 10)
                            if new_shares > 0:
                                positions[sid] += new_shares

                # --- 風險出場檢查 ---
                risk_exit = False
                exit_reason = ""

                if positions[sid] > 0:
                    ep = entry_prices[sid]

                    if raw_high > peak_since_entry[sid]:
                        peak_since_entry[sid] = raw_high

                    # 停損
                    if self.risk_config.stop_loss_pct is not None:
                        stop_price = ep * (1 - self.risk_config.stop_loss_pct / 100)
                        if raw_low <= stop_price:
                            risk_exit = True
                            exit_reason = "stop_loss"
                            raw_close = min(raw_close, stop_price)

                    # 停利
                    if not risk_exit and self.risk_config.take_profit_pct is not None:
                        tp_price = ep * (1 + self.risk_config.take_profit_pct / 100)
                        if raw_high >= tp_price:
                            risk_exit = True
                            exit_reason = "take_profit"
                            raw_close = max(raw_close, tp_price)

                    # 移動停損
                    if not risk_exit and self.risk_config.trailing_stop_pct is not None:
                        trail_price = peak_since_entry[sid] * (1 - self.risk_config.trailing_stop_pct / 100)
                        if raw_low <= trail_price:
                            risk_exit = True
                            exit_reason = "trailing_stop"
                            raw_close = min(raw_close, trail_price)

                # 執行風險出場
                if risk_exit and positions[sid] > 0:
                    sell_price = raw_close * (1 - self.config.slippage)
                    revenue = positions[sid] * sell_price
                    commission = revenue * self.config.commission_rate
                    tax = revenue * self.config.tax_rate
                    net_revenue = revenue - commission - tax
                    capital += net_revenue

                    pnl = net_revenue - positions[sid] * entry_prices[sid]
                    ret_pct = (sell_price / entry_prices[sid] - 1) * 100

                    trades.append(
                        PortfolioTradeRecord(
                            stock_id=sid,
                            entry_date=entry_dates[sid],
                            entry_price=round(entry_prices[sid], 2),
                            exit_date=dt,
                            exit_price=round(sell_price, 2),
                            shares=positions[sid],
                            pnl=round(pnl, 2),
                            return_pct=round(ret_pct, 2),
                            exit_reason=exit_reason,
                        )
                    )
                    per_stock_pnl[sid] += pnl
                    positions[sid] = 0
                    entry_prices[sid] = 0.0
                    entry_dates[sid] = None
                    peak_since_entry[sid] = 0.0

                elif not risk_exit:
                    # 買入
                    if signal == 1 and positions[sid] == 0:
                        buy_price = raw_close * (1 + self.config.slippage)
                        # 買入金額 = capital × weight，受 max_position_pct 限制
                        alloc_capital = capital * min(weights[sid], self.portfolio_config.max_position_pct)
                        commission = alloc_capital * self.config.commission_rate
                        available = alloc_capital - commission
                        shares = int(available // buy_price)

                        if shares > 0:
                            cost = shares * buy_price + shares * buy_price * self.config.commission_rate
                            capital -= cost
                            positions[sid] = shares
                            entry_prices[sid] = buy_price
                            entry_dates[sid] = dt
                            peak_since_entry[sid] = raw_high

                    # 賣出
                    elif signal == -1 and positions[sid] > 0:
                        sell_price = raw_close * (1 - self.config.slippage)
                        revenue = positions[sid] * sell_price
                        commission = revenue * self.config.commission_rate
                        tax = revenue * self.config.tax_rate
                        net_revenue = revenue - commission - tax
                        capital += net_revenue

                        pnl = net_revenue - positions[sid] * entry_prices[sid]
                        ret_pct = (sell_price / entry_prices[sid] - 1) * 100

                        trades.append(
                            PortfolioTradeRecord(
                                stock_id=sid,
                                entry_date=entry_dates[sid],
                                entry_price=round(entry_prices[sid], 2),
                                exit_date=dt,
                                exit_price=round(sell_price, 2),
                                shares=positions[sid],
                                pnl=round(pnl, 2),
                                return_pct=round(ret_pct, 2),
                                exit_reason="signal",
                            )
                        )
                        per_stock_pnl[sid] += pnl
                        positions[sid] = 0
                        entry_prices[sid] = 0.0
                        entry_dates[sid] = None
                        peak_since_entry[sid] = 0.0

            # 計算當日總權益
            mtm = capital
            for sid in stock_ids:
                if positions[sid] > 0 and dt in stock_data[sid].index:
                    d = stock_data[sid]
                    rc = d.loc[dt, "raw_close"] if "raw_close" in d.columns else d.loc[dt, "close"]
                    mtm += positions[sid] * rc
            equity_curve.append(mtm)

        # 5. 結束時強制平倉所有持倉
        for sid in stock_ids:
            if positions[sid] > 0:
                data = stock_data[sid]
                last_close = data.iloc[-1]["raw_close"] if "raw_close" in data.columns else data.iloc[-1]["close"]
                sell_price = last_close * (1 - self.config.slippage)
                revenue = positions[sid] * sell_price
                commission = revenue * self.config.commission_rate
                tax = revenue * self.config.tax_rate
                net_revenue = revenue - commission - tax
                capital += net_revenue

                pnl = net_revenue - positions[sid] * entry_prices[sid]
                ret_pct = (sell_price / entry_prices[sid] - 1) * 100

                trades.append(
                    PortfolioTradeRecord(
                        stock_id=sid,
                        entry_date=entry_dates[sid],
                        entry_price=round(entry_prices[sid], 2),
                        exit_date=data.index[-1],
                        exit_price=round(sell_price, 2),
                        shares=positions[sid],
                        pnl=round(pnl, 2),
                        return_pct=round(ret_pct, 2),
                        exit_reason="force_close",
                    )
                )
                per_stock_pnl[sid] += pnl
                positions[sid] = 0

        if equity_curve:
            equity_curve[-1] = capital

        # 6. 計算組合層級指標
        metrics = self._compute_metrics(equity_curve, trades, all_dates[0], all_dates[-1])

        # 個股報酬分解
        per_stock_returns = {}
        for sid in stock_ids:
            per_stock_returns[sid] = round(per_stock_pnl[sid] / self.config.initial_capital * 100, 2)

        return PortfolioResultData(
            strategy_name=strategy_name,
            stock_ids=stock_ids,
            start_date=all_dates[0],
            end_date=all_dates[-1],
            initial_capital=self.config.initial_capital,
            final_capital=round(capital, 2),
            total_return=metrics["total_return"],
            annual_return=metrics["annual_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            total_trades=len(trades),
            sortino_ratio=metrics["sortino_ratio"],
            calmar_ratio=metrics["calmar_ratio"],
            var_95=metrics["var_95"],
            cvar_95=metrics["cvar_95"],
            profit_factor=metrics["profit_factor"],
            allocation_method=self.portfolio_config.allocation_method,
            trades=trades,
            equity_curve=equity_curve,
            per_stock_returns=per_stock_returns,
        )

    def _compute_weights(
        self, stock_ids: list[str], stock_data: dict[str, pd.DataFrame] | None = None
    ) -> dict[str, float]:
        """計算各股票的資金配置權重。"""
        method = self.portfolio_config.allocation_method

        if method == "custom" and self.portfolio_config.weights:
            weights = {}
            for sid in stock_ids:
                weights[sid] = self.portfolio_config.weights.get(sid, 0.0)
            # 正規化
            total = sum(weights.values())
            if total > 0:
                weights = {sid: w / total for sid, w in weights.items()}
            return weights

        if method in ("risk_parity", "mean_variance") and stock_data:
            returns = self._build_returns_df(stock_ids, stock_data)
            if returns is not None and len(returns) >= 30:
                from src.backtest.allocator import mean_variance_weights, risk_parity_weights

                if method == "risk_parity":
                    return risk_parity_weights(returns)
                else:
                    return mean_variance_weights(returns)
            else:
                logger.warning("報酬率資料不足（< 30 天），%s fallback 到 equal_weight", method)

        # equal_weight（預設或 fallback）
        n = len(stock_ids)
        return {sid: 1.0 / n for sid in stock_ids}

    @staticmethod
    def _build_returns_df(stock_ids: list[str], stock_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        """從各股票的 OHLCV DataFrame 建立日報酬率矩陣。"""
        series_map = {}
        for sid in stock_ids:
            df = stock_data[sid]
            col = "raw_close" if "raw_close" in df.columns else "close"
            series_map[sid] = df[col]

        prices = pd.DataFrame(series_map)
        prices = prices.dropna()
        if len(prices) < 2:
            return None
        return prices.pct_change().dropna()

    def _compute_metrics(
        self,
        equity_curve: list[float],
        trades: list[PortfolioTradeRecord],
        start: date,
        end: date,
    ) -> dict:
        """計算組合層級績效指標。"""
        initial = self.config.initial_capital
        final = equity_curve[-1] if equity_curve else initial

        total_return = (final / initial - 1) * 100

        days = (end - start).days
        years = days / 365.25 if days > 0 else 1
        if final > 0 and initial > 0 and years > 0:
            annual_return = ((final / initial) ** (1 / years) - 1) * 100
        else:
            annual_return = 0.0

        sharpe_ratio = None
        sortino_ratio = None
        calmar_ratio = None
        var_95 = None
        cvar_95 = None

        max_drawdown = 0.0
        if equity_curve:
            peak = equity_curve[0]
            for val in equity_curve:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak * 100
                if dd > max_drawdown:
                    max_drawdown = dd

        if len(equity_curve) > 1:
            eq = np.array(equity_curve)
            daily_returns = np.diff(eq) / eq[:-1]

            if np.std(daily_returns) > 0:
                sharpe_ratio = round(np.mean(daily_returns) / np.std(daily_returns) * math.sqrt(252), 4)

            neg_returns = daily_returns[daily_returns < 0]
            if len(neg_returns) > 0 and np.std(neg_returns) > 0:
                sortino_ratio = round(np.mean(daily_returns) / np.std(neg_returns) * math.sqrt(252), 4)

            if max_drawdown > 0:
                calmar_ratio = round(annual_return / max_drawdown, 4)

            var_95 = round(float(np.percentile(daily_returns, 5)) * 100, 4)

            var_threshold = np.percentile(daily_returns, 5)
            tail_returns = daily_returns[daily_returns <= var_threshold]
            if len(tail_returns) > 0:
                cvar_95 = round(float(np.mean(tail_returns)) * 100, 4)

        win_rate = None
        profit_factor = None
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            win_rate = round(wins / len(trades) * 100, 2)

            gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
            if gross_loss > 0:
                profit_factor = round(gross_profit / gross_loss, 4)

        return {
            "total_return": round(total_return, 2),
            "annual_return": round(annual_return, 2),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": round(max_drawdown, 2),
            "win_rate": win_rate,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "profit_factor": profit_factor,
        }
