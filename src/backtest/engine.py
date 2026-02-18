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

    initial_capital: float = 1_000_000  # 初始資金
    commission_rate: float = 0.001425  # 手續費 0.1425%
    tax_rate: float = 0.003  # 交易稅 0.3%（賣出時）
    slippage: float = 0.0005  # 滑價 0.05%


@dataclass
class RiskConfig:
    """風險管理參數。"""

    stop_loss_pct: float | None = None  # 停損 %（例 5.0 = -5% 出場）
    take_profit_pct: float | None = None  # 停利 %（例 15.0 = +15% 出場）
    trailing_stop_pct: float | None = None  # 移動停損 %（從高點回落）
    position_sizing: str = "all_in"  # "all_in"|"fixed_fraction"|"kelly"|"atr"
    fixed_fraction: float = 1.0  # 固定比例（0.0~1.0）
    kelly_fraction: float = 0.5  # Kelly 乘數（預設 half-Kelly）
    atr_risk_pct: float = 1.0  # ATR sizing: 每筆風險占資金 %
    atr_period: int = 14
    atr_multiplier: float = 2.0


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
    exit_reason: str = "signal"  # signal / stop_loss / take_profit / trailing_stop / force_close


@dataclass
class BacktestResultData:
    """回測結果。"""

    stock_id: str
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_return: float  # %
    annual_return: float  # %
    sharpe_ratio: float | None
    max_drawdown: float  # %
    win_rate: float | None  # %
    total_trades: int
    benchmark_return: float | None = None  # 同期 buy & hold 報酬率 (%)
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    var_95: float | None = None  # Value at Risk (95%)
    cvar_95: float | None = None  # Conditional VaR (95%)
    profit_factor: float | None = None
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class BacktestEngine:
    """回測引擎。

    預設使用全倉進出策略（向後相容）。
    可透過 RiskConfig 啟用停損/停利/移動停損與部位大小計算。
    """

    def __init__(
        self,
        strategy: Strategy,
        config: BacktestConfig | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.risk_config = risk_config or RiskConfig()

    def run(self) -> BacktestResultData:
        """執行回測，回傳績效結果。"""
        data = self.strategy.load_data()
        if data.empty:
            raise ValueError(f"[{self.strategy.stock_id}] 無交易資料")

        signals = self.strategy.generate_signals(data)

        capital = self.config.initial_capital
        position = 0  # 持有股數
        entry_price = 0.0
        entry_date = None
        peak_since_entry = 0.0  # 移動停損用：進場後最高價
        trades: list[TradeRecord] = []
        equity_curve: list[float] = []

        for dt in data.index:
            close = data.loc[dt, "close"]
            high = data.loc[dt, "high"]
            low = data.loc[dt, "low"]
            signal = signals.get(dt, 0)

            # --- 風險出場檢查（持倉中才執行） ---
            risk_exit = False
            exit_reason = ""

            if position > 0:
                # 更新持倉最高價
                if high > peak_since_entry:
                    peak_since_entry = high

                # 1. 停損檢查
                if self.risk_config.stop_loss_pct is not None:
                    stop_price = entry_price * (1 - self.risk_config.stop_loss_pct / 100)
                    if low <= stop_price:
                        risk_exit = True
                        exit_reason = "stop_loss"
                        close = min(close, stop_price)  # 以停損價出場

                # 2. 停利檢查
                if not risk_exit and self.risk_config.take_profit_pct is not None:
                    tp_price = entry_price * (1 + self.risk_config.take_profit_pct / 100)
                    if high >= tp_price:
                        risk_exit = True
                        exit_reason = "take_profit"
                        close = max(close, tp_price)  # 以停利價出場

                # 3. 移動停損檢查
                if not risk_exit and self.risk_config.trailing_stop_pct is not None:
                    trail_price = peak_since_entry * (1 - self.risk_config.trailing_stop_pct / 100)
                    if low <= trail_price:
                        risk_exit = True
                        exit_reason = "trailing_stop"
                        close = min(close, trail_price)

            # --- 執行風險出場 ---
            if risk_exit and position > 0:
                sell_price = close * (1 - self.config.slippage)
                revenue = position * sell_price
                commission = revenue * self.config.commission_rate
                tax = revenue * self.config.tax_rate
                net_revenue = revenue - commission - tax
                capital += net_revenue

                pnl = net_revenue - position * entry_price
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
                        exit_reason=exit_reason,
                    )
                )
                position = 0
                entry_price = 0.0
                entry_date = None
                peak_since_entry = 0.0

            # --- 正常訊號處理（風險出場未觸發時） ---
            elif not risk_exit:
                # --- 買入 ---
                if signal == 1 and position == 0:
                    buy_price = data.loc[dt, "close"] * (1 + self.config.slippage)
                    shares = self._calculate_shares(capital, buy_price, data, dt, trades)
                    if shares > 0:
                        cost = shares * buy_price + shares * buy_price * self.config.commission_rate
                        capital -= cost
                        position = shares
                        entry_price = buy_price
                        entry_date = dt
                        peak_since_entry = high

                # --- 賣出 ---
                elif signal == -1 and position > 0:
                    sell_price = data.loc[dt, "close"] * (1 - self.config.slippage)
                    revenue = position * sell_price
                    commission = revenue * self.config.commission_rate
                    tax = revenue * self.config.tax_rate
                    net_revenue = revenue - commission - tax
                    capital += net_revenue

                    pnl = net_revenue - position * entry_price
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
                    peak_since_entry = 0.0

            # 記錄每日權益
            mark_to_market = capital + position * data.loc[dt, "close"]
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

        final_capital = capital
        metrics = self._compute_metrics(equity_curve, trades, data.index[0], data.index[-1])

        # 計算同期 buy & hold 基準報酬
        benchmark_return = self._compute_benchmark(data)

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
            benchmark_return=benchmark_return,
            sortino_ratio=metrics["sortino_ratio"],
            calmar_ratio=metrics["calmar_ratio"],
            var_95=metrics["var_95"],
            cvar_95=metrics["cvar_95"],
            profit_factor=metrics["profit_factor"],
            trades=trades,
            equity_curve=equity_curve,
        )

    # ------------------------------------------------------------------ #
    #  部位大小計算
    # ------------------------------------------------------------------ #

    def _calculate_shares(
        self,
        capital: float,
        buy_price: float,
        data: pd.DataFrame,
        dt,
        trades: list[TradeRecord],
    ) -> int:
        """根據 RiskConfig.position_sizing 計算應買股數。"""
        mode = self.risk_config.position_sizing

        if mode == "fixed_fraction":
            available = capital * self.risk_config.fixed_fraction
            available -= available * self.config.commission_rate
            return int(available // buy_price)

        if mode == "kelly":
            kelly_f = self._compute_kelly_fraction(trades)
            available = capital * kelly_f
            available -= available * self.config.commission_rate
            return int(available // buy_price)

        if mode == "atr":
            atr = self._compute_atr(data, dt)
            if atr is not None and atr > 0:
                risk_amount = capital * (self.risk_config.atr_risk_pct / 100)
                risk_per_share = atr * self.risk_config.atr_multiplier
                max_shares = int(risk_amount / risk_per_share)
                # 也不能超過全部資金
                available = capital - capital * self.config.commission_rate
                max_affordable = int(available // buy_price)
                return min(max_shares, max_affordable)
            # ATR 資料不足，fallback 到 fixed_fraction
            available = capital * self.risk_config.fixed_fraction
            available -= available * self.config.commission_rate
            return int(available // buy_price)

        # all_in（預設，向後相容）
        commission = capital * self.config.commission_rate
        available = capital - commission
        return int(available // buy_price)

    def _compute_kelly_fraction(self, trades: list[TradeRecord]) -> float:
        """從歷史交易計算 Kelly Criterion 比例。

        f = W - (1-W) / (avgWin/avgLoss)
        交易不足 5 筆時 fallback 10%。
        """
        if len(trades) < 5:
            return 0.1

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]

        if not wins or not losses:
            return 0.1

        win_rate = len(wins) / len(trades)
        avg_win = sum(t.pnl for t in wins) / len(wins)
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses))

        if avg_loss == 0:
            return 0.1

        kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        kelly = max(0.01, min(kelly, 1.0))  # 限制在 1%~100%
        return kelly * self.risk_config.kelly_fraction

    def _compute_atr(self, data: pd.DataFrame, dt) -> float | None:
        """計算 ATR(N)，資料不足時回傳 None。"""
        idx = data.index.get_loc(dt)
        period = self.risk_config.atr_period

        if idx < period:
            return None

        window = data.iloc[idx - period : idx]
        high = window["high"].values
        low = window["low"].values
        prev_close = data.iloc[idx - period - 1 : idx - 1]["close"].values

        if len(prev_close) < period:
            return None

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - prev_close),
                np.abs(low - prev_close),
            ),
        )
        return float(np.mean(tr))

    # ------------------------------------------------------------------ #
    #  基準與績效計算
    # ------------------------------------------------------------------ #

    def _compute_benchmark(self, data: pd.DataFrame) -> float | None:
        """計算同期 buy & hold 報酬率（不含交易成本，純價差）。"""
        if data.empty or len(data) < 2:
            return None
        first_close = data.iloc[0]["close"]
        last_close = data.iloc[-1]["close"]
        return round((last_close / first_close - 1) * 100, 2)

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
                sharpe_ratio = round(np.mean(daily_returns) / np.std(daily_returns) * math.sqrt(252), 4)

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

        # --- 進階指標 ---
        sortino_ratio = None
        calmar_ratio = None
        var_95 = None
        cvar_95 = None
        profit_factor = None

        if len(equity_curve) > 1:
            eq = np.array(equity_curve)
            daily_returns = np.diff(eq) / eq[:-1]

            # Sortino Ratio: mean / downside_std × √252
            neg_returns = daily_returns[daily_returns < 0]
            if len(neg_returns) > 0 and np.std(neg_returns) > 0:
                sortino_ratio = round(np.mean(daily_returns) / np.std(neg_returns) * math.sqrt(252), 4)

            # Calmar Ratio: annual_return / max_drawdown
            if max_drawdown > 0:
                calmar_ratio = round(annual_return / max_drawdown, 4)

            # VaR (95%): 5th percentile of daily returns
            var_95 = round(float(np.percentile(daily_returns, 5)) * 100, 4)

            # CVaR (95%): mean of returns <= VaR
            var_threshold = np.percentile(daily_returns, 5)
            tail_returns = daily_returns[daily_returns <= var_threshold]
            if len(tail_returns) > 0:
                cvar_95 = round(float(np.mean(tail_returns)) * 100, 4)

        # Profit Factor: gross_profit / gross_loss
        if trades:
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
