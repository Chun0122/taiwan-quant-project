"""回測引擎 — 模擬歷史交易並計算績效指標。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_metrics
from src.constants import (
    COMMISSION_RATE,
    SLIPPAGE_IMPACT_COEFF,
    SLIPPAGE_RATE,
    TAX_RATE,
)
from src.strategy.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回測參數（符合台股實際費用）。"""

    initial_capital: float = 1_000_000  # 初始資金
    commission_rate: float = COMMISSION_RATE  # 手續費 0.1425%
    tax_rate: float = TAX_RATE  # 交易稅 0.3%（賣出時）
    slippage: float = SLIPPAGE_RATE  # 滑價 0.05%
    dynamic_slippage: bool = False  # 啟用動態滑價（根據成交量調整）
    slippage_impact_coeff: float = SLIPPAGE_IMPACT_COEFF  # 動態滑價衝擊係數 k
    liquidity_limit: float | None = None  # 流動性約束（單筆 ≤ 當日量 × 此比例，None=不限制）


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
    atr_multiplier_stop: float | None = None  # ATR-based 止損乘數（止損 = 進場價 − N×ATR14）
    atr_multiplier_profit: float | None = None  # ATR-based 止利乘數（止利 = 進場價 + N×ATR14）


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
    stop_price: float | None = None  # 進場時計算並固定的止損價
    target_price: float | None = None  # 進場時計算並固定的目標價


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
        current_stop_price: float | None = None  # 進場時固定的止損價
        current_target_price: float | None = None  # 進場時固定的目標價
        trades: list[TradeRecord] = []
        equity_curve: list[float] = []

        # 除權息資料（若有）
        has_raw = "raw_close" in data.columns
        dividends = getattr(self.strategy, "_dividends", None)

        for dt in data.index:
            close = data.loc[dt, "close"]
            high = data.loc[dt, "high"]
            low = data.loc[dt, "low"]

            # 若有除權息調整，交易使用原始價格
            raw_close = data.loc[dt, "raw_close"] if has_raw else close
            raw_high = data.loc[dt, "raw_high"] if has_raw else high
            raw_low = data.loc[dt, "raw_low"] if has_raw else low
            signal = signals.get(dt, 0)

            # --- 除權息處理（持倉中才處理） ---
            if dividends is not None and not dividends.empty and position > 0:
                if dt in dividends.index:
                    cash_div = dividends.loc[dt, "cash_dividend"]
                    stock_div = dividends.loc[dt, "stock_dividend"]
                    if cash_div > 0:
                        capital += cash_div * position
                    if stock_div > 0:
                        new_shares = int(position * stock_div / 10)
                        if new_shares > 0:
                            position += new_shares

            # --- 風險出場檢查（持倉中才執行） ---
            risk_exit = False
            exit_reason = ""

            if position > 0:
                # 更新持倉最高價
                if raw_high > peak_since_entry:
                    peak_since_entry = raw_high

                # 1. 停損檢查（使用進場時已計算並固定的止損價）
                if current_stop_price is not None and raw_low <= current_stop_price:
                    risk_exit = True
                    exit_reason = "stop_loss"
                    raw_close = min(raw_close, current_stop_price)

                # 2. 停利檢查（使用進場時已計算並固定的目標價）
                if not risk_exit and current_target_price is not None and raw_high >= current_target_price:
                    risk_exit = True
                    exit_reason = "take_profit"
                    raw_close = max(raw_close, current_target_price)

                # 3. 移動停損檢查
                if not risk_exit and self.risk_config.trailing_stop_pct is not None:
                    trail_price = peak_since_entry * (1 - self.risk_config.trailing_stop_pct / 100)
                    if raw_low <= trail_price:
                        risk_exit = True
                        exit_reason = "trailing_stop"
                        raw_close = min(raw_close, trail_price)

            # --- 執行風險出場 ---
            if risk_exit and position > 0:
                daily_volume = float(data.loc[dt, "volume"]) if "volume" in data.columns else 0.0
                sell_price = raw_close * (1 - self._get_slippage(daily_volume))
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
                        stop_price=round(current_stop_price, 2) if current_stop_price is not None else None,
                        target_price=round(current_target_price, 2) if current_target_price is not None else None,
                    )
                )
                position = 0
                entry_price = 0.0
                entry_date = None
                peak_since_entry = 0.0
                current_stop_price = None
                current_target_price = None

            # --- 正常訊號處理（風險出場未觸發時） ---
            elif not risk_exit:
                # --- 買入 ---
                if signal == 1 and position == 0:
                    daily_volume = float(data.loc[dt, "volume"]) if "volume" in data.columns else 0.0
                    slip = self._get_slippage(daily_volume)
                    buy_price = raw_close * (1 + slip)
                    shares = self._calculate_shares(capital, buy_price, data, dt, trades)
                    shares = self._apply_liquidity_limit(shares, daily_volume)
                    if shares > 0:
                        cost = shares * buy_price + shares * buy_price * self.config.commission_rate
                        capital -= cost
                        position = shares
                        entry_price = buy_price
                        entry_date = dt
                        peak_since_entry = raw_high

                        # 計算並固定止損價（ATR-based 優先，否則百分比，否則 None）
                        # use_raw=has_raw：除權息模式下以原始價格計算 ATR，確保與 entry_price 同尺度
                        atr_val = self._compute_atr(data, dt, use_raw=has_raw)
                        if self.risk_config.atr_multiplier_stop is not None and atr_val is not None:
                            current_stop_price = entry_price - self.risk_config.atr_multiplier_stop * atr_val
                        elif self.risk_config.stop_loss_pct is not None:
                            current_stop_price = entry_price * (1 - self.risk_config.stop_loss_pct / 100)
                        else:
                            current_stop_price = None

                        # 計算並固定目標價（ATR-based 優先，否則百分比，否則 None）
                        if self.risk_config.atr_multiplier_profit is not None and atr_val is not None:
                            current_target_price = entry_price + self.risk_config.atr_multiplier_profit * atr_val
                        elif self.risk_config.take_profit_pct is not None:
                            current_target_price = entry_price * (1 + self.risk_config.take_profit_pct / 100)
                        else:
                            current_target_price = None

                # --- 賣出 ---
                elif signal == -1 and position > 0:
                    daily_volume = float(data.loc[dt, "volume"]) if "volume" in data.columns else 0.0
                    sell_price = raw_close * (1 - self._get_slippage(daily_volume))
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
                            stop_price=round(current_stop_price, 2) if current_stop_price is not None else None,
                            target_price=round(current_target_price, 2) if current_target_price is not None else None,
                        )
                    )
                    position = 0
                    entry_price = 0.0
                    entry_date = None
                    peak_since_entry = 0.0
                    current_stop_price = None
                    current_target_price = None

            # 記錄每日權益
            mark_to_market = capital + position * raw_close
            equity_curve.append(mark_to_market)

        # 若回測結束時仍持有部位，以最後收盤價平倉
        if position > 0:
            last_close = data.iloc[-1]["raw_close"] if has_raw else data.iloc[-1]["close"]
            last_volume = float(data.iloc[-1]["volume"]) if "volume" in data.columns else 0.0
            sell_price = last_close * (1 - self._get_slippage(last_volume))
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
                    stop_price=round(current_stop_price, 2) if current_stop_price is not None else None,
                    target_price=round(current_target_price, 2) if current_target_price is not None else None,
                )
            )
            position = 0
            equity_curve[-1] = capital

        final_capital = capital
        metrics = compute_metrics(equity_curve, trades, data.index[0], data.index[-1], self.config.initial_capital)

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
    #  動態滑價 & 流動性
    # ------------------------------------------------------------------ #

    def _get_slippage(self, volume: float) -> float:
        """計算滑價比率。

        動態模式：slippage = base + k / sqrt(volume)
        - 大量股（如 2330）：衝擊趨近 base（0.05%）
        - 小量股（日均萬股級）：衝擊可達 0.3%~0.5%
        """
        if not self.config.dynamic_slippage or volume <= 0:
            return self.config.slippage
        base = self.config.slippage
        k = self.config.slippage_impact_coeff
        return base + k / np.sqrt(volume)

    def _apply_liquidity_limit(self, shares: int, daily_volume: float) -> int:
        """流動性約束：限制單筆交易量不超過當日成交量的指定比例。"""
        if self.config.liquidity_limit is None or daily_volume <= 0:
            return shares
        max_shares = int(daily_volume * self.config.liquidity_limit)
        if max_shares < 1:
            return 0
        return min(shares, max_shares)

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

    def _compute_atr(self, data: pd.DataFrame, dt, use_raw: bool = False) -> float | None:
        """計算 ATR(N)，資料不足時回傳 None。

        True Range 定義：TR = max(H-L, |H-PrevClose|, |L-PrevClose|)
        其中 PrevClose 為每根 K 線的前一根收盤價，而非偏移視窗。

        Args:
            use_raw: True 時優先使用 raw_high/raw_low/raw_close（除權息還原模式下
                     確保 ATR 與 entry_price 在相同的原始價格尺度計算）。
        """
        idx = data.index.get_loc(dt)
        period = self.risk_config.atr_period

        if idx < period:
            return None

        # 依 use_raw 決定使用哪組價格欄位，fallback 至調整後欄位
        high_col = "raw_high" if use_raw and "raw_high" in data.columns else "high"
        low_col = "raw_low" if use_raw and "raw_low" in data.columns else "low"
        close_col = "raw_close" if use_raw and "raw_close" in data.columns else "close"

        # 取 period 根 K 線的 high/low，以及對應的 period 個 prev_close
        # window[i] 的 prev_close 是 window[i-1] 的 close
        # 所以需要 data[idx-period-1 : idx-1] 作為 prev_close（長度 = period）
        window = data.iloc[idx - period : idx]
        high = window[high_col].values
        low = window[low_col].values
        # prev_close[i] 對應 window[i] 的前一根收盤價
        prev_close = data[close_col].iloc[idx - period - 1 : idx - 1].values

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
    #  基準計算
    # ------------------------------------------------------------------ #

    def _compute_benchmark(self, data: pd.DataFrame) -> float | None:
        """計算同期 buy & hold 報酬率（含股利，不含交易成本）。"""
        if data.empty or len(data) < 2:
            return None

        close_col = "raw_close" if "raw_close" in data.columns else "close"
        first_close = data.iloc[0][close_col]
        last_close = data.iloc[-1][close_col]
        if first_close <= 0:
            return None

        dividends = getattr(self.strategy, "_dividends", None)
        if dividends is not None and not dividends.empty:
            # 模擬持有 1 股，累計現金股利 + 股票股利增股
            shares = 1.0
            total_cash_div = 0.0
            for ex_date in dividends.index:
                if ex_date < data.index[0] or ex_date > data.index[-1]:
                    continue
                cash_div = dividends.loc[ex_date, "cash_dividend"]
                stock_div = dividends.loc[ex_date, "stock_dividend"]
                total_cash_div += cash_div * shares
                shares *= 1 + stock_div / 10

            total_return = (last_close * shares + total_cash_div) / first_close - 1
            return round(total_return * 100, 2)

        return round((last_close / first_close - 1) * 100, 2)
