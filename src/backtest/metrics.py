"""回測績效指標計算 — 共用純函數，供 engine / portfolio / walk_forward 統一呼叫。

提供三組功能：
1. compute_metrics() — 回測績效指標（Sharpe / Sortino / MDD 等）
2. compute_trade_stats() — 交易明細統計（持倉天數 / 出場原因分佈 / 勝敗分析）
3. trades_to_dataframe() — 交易記錄轉 DataFrame（含 holding_days 計算欄位，可直接匯出 CSV）
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
#  交易記錄 Protocol（僅要求 pnl 屬性，相容 TradeRecord / PortfolioTradeRecord）
# ------------------------------------------------------------------


class _HasPnl(Protocol):
    @property
    def pnl(self) -> float: ...


# ------------------------------------------------------------------
#  共用績效指標計算
# ------------------------------------------------------------------


def compute_metrics(
    equity_curve: list[float],
    trades: list[_HasPnl],
    start: date,
    end: date,
    initial_capital: float,
) -> dict:
    """計算回測績效指標（純函數）。

    包含：total_return, annual_return, sharpe_ratio, sortino_ratio,
          calmar_ratio, max_drawdown, win_rate, var_95, cvar_95, profit_factor。

    Parameters
    ----------
    equity_curve : list[float]
        每日權益序列。
    trades : list
        交易記錄列表，每筆需有 .pnl 屬性。
    start, end : date
        回測起迄日。
    initial_capital : float
        初始資金。
    """
    initial = initial_capital
    final = equity_curve[-1] if equity_curve else initial

    # --- 總報酬率 ---
    total_return = (final / initial - 1) * 100

    # --- 年化報酬率 ---
    days = (end - start).days
    years = days / 365.25 if days > 0 else 1
    if final > 0 and initial > 0 and years > 0:
        annual_return = ((final / initial) ** (1 / years) - 1) * 100
    else:
        annual_return = 0.0

    # --- 每日報酬率（含零值防護，防止爆倉時 division by zero） ---
    daily_returns: np.ndarray | None = None
    if len(equity_curve) > 1:
        eq = np.array(equity_curve, dtype=np.float64)
        prev_eq = eq[:-1]
        safe_prev = np.where(prev_eq == 0, np.nan, prev_eq)
        raw_daily = np.diff(eq) / safe_prev
        daily_returns = raw_daily[np.isfinite(raw_daily)]

    # --- Sharpe Ratio (rf=0, annualized) ---
    sharpe_ratio = None
    if daily_returns is not None and len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe_ratio = round(
            float(np.mean(daily_returns)) / float(np.std(daily_returns)) * math.sqrt(252),
            4,
        )

    # --- 最大回撤 ---
    max_drawdown = 0.0
    if equity_curve:
        peak = equity_curve[0]
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_drawdown:
                max_drawdown = dd

    # --- 勝率 ---
    win_rate = None
    if trades:
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = round(wins / len(trades) * 100, 2)

    # --- Sortino Ratio ---
    sortino_ratio = None
    if daily_returns is not None and len(daily_returns) > 1:
        neg_returns = daily_returns[daily_returns < 0]
        if len(neg_returns) > 0 and np.std(neg_returns) > 0:
            sortino_ratio = round(
                float(np.mean(daily_returns)) / float(np.std(neg_returns)) * math.sqrt(252),
                4,
            )

    # --- Calmar Ratio ---
    calmar_ratio = None
    if max_drawdown > 0:
        calmar_ratio = round(annual_return / max_drawdown, 4)

    # --- VaR (95%) ---
    var_95 = None
    if daily_returns is not None and len(daily_returns) > 1:
        var_95 = round(float(np.percentile(daily_returns, 5)) * 100, 4)

    # --- CVaR (95%) ---
    cvar_95 = None
    if daily_returns is not None and len(daily_returns) > 1:
        var_threshold = np.percentile(daily_returns, 5)
        tail_returns = daily_returns[daily_returns <= var_threshold]
        if len(tail_returns) > 0:
            cvar_95 = round(float(np.mean(tail_returns)) * 100, 4)

    # --- Profit Factor ---
    profit_factor = None
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


# ------------------------------------------------------------------
#  交易明細統計
# ------------------------------------------------------------------


def _holding_days(entry_date: date | None, exit_date: date | None) -> int | None:
    """計算持倉天數（日曆天）。任一端為 None 時回傳 None。"""
    if entry_date is None or exit_date is None:
        return None
    return (exit_date - entry_date).days


def compute_trade_stats(trades: list) -> dict:
    """計算交易明細統計（純函數）。

    Parameters
    ----------
    trades : list
        交易記錄列表，每筆需有 entry_date, exit_date, pnl, return_pct, exit_reason 屬性。

    Returns
    -------
    dict
        holding_days_avg : 平均持倉天數
        holding_days_median : 中位數持倉天數
        holding_days_min : 最短持倉天數
        holding_days_max : 最長持倉天數
        avg_win_pnl : 獲利交易平均損益
        avg_loss_pnl : 虧損交易平均損益
        avg_win_return : 獲利交易平均報酬率 (%)
        avg_loss_return : 虧損交易平均報酬率 (%)
        max_consecutive_wins : 最大連勝次數
        max_consecutive_losses : 最大連敗次數
        exit_reason_counts : 出場原因分佈 dict（reason → count）
    """
    if not trades:
        return {
            "holding_days_avg": None,
            "holding_days_median": None,
            "holding_days_min": None,
            "holding_days_max": None,
            "avg_win_pnl": None,
            "avg_loss_pnl": None,
            "avg_win_return": None,
            "avg_loss_return": None,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "exit_reason_counts": {},
        }

    # --- 持倉天數 ---
    hdays = [_holding_days(getattr(t, "entry_date", None), getattr(t, "exit_date", None)) for t in trades]
    valid_hdays = [d for d in hdays if d is not None]

    if valid_hdays:
        holding_days_avg = round(sum(valid_hdays) / len(valid_hdays), 1)
        sorted_hd = sorted(valid_hdays)
        n = len(sorted_hd)
        holding_days_median = (
            float(sorted_hd[n // 2]) if n % 2 == 1 else round((sorted_hd[n // 2 - 1] + sorted_hd[n // 2]) / 2, 1)
        )
        holding_days_min = min(valid_hdays)
        holding_days_max = max(valid_hdays)
    else:
        holding_days_avg = None
        holding_days_median = None
        holding_days_min = None
        holding_days_max = None

    # --- 勝敗分析 ---
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]

    avg_win_pnl = round(sum(t.pnl for t in wins) / len(wins), 2) if wins else None
    avg_loss_pnl = round(sum(t.pnl for t in losses) / len(losses), 2) if losses else None
    avg_win_return = round(sum(t.return_pct for t in wins) / len(wins), 2) if wins else None
    avg_loss_return = round(sum(t.return_pct for t in losses) / len(losses), 2) if losses else None

    # --- 最大連勝/連敗 ---
    max_con_win = 0
    max_con_loss = 0
    cur_win = 0
    cur_loss = 0
    for t in trades:
        if t.pnl > 0:
            cur_win += 1
            cur_loss = 0
            if cur_win > max_con_win:
                max_con_win = cur_win
        elif t.pnl < 0:
            cur_loss += 1
            cur_win = 0
            if cur_loss > max_con_loss:
                max_con_loss = cur_loss
        else:
            cur_win = 0
            cur_loss = 0

    # --- 出場原因分佈 ---
    reasons = [getattr(t, "exit_reason", "unknown") for t in trades]
    exit_reason_counts = dict(Counter(reasons))

    return {
        "holding_days_avg": holding_days_avg,
        "holding_days_median": holding_days_median,
        "holding_days_min": holding_days_min,
        "holding_days_max": holding_days_max,
        "avg_win_pnl": avg_win_pnl,
        "avg_loss_pnl": avg_loss_pnl,
        "avg_win_return": avg_win_return,
        "avg_loss_return": avg_loss_return,
        "max_consecutive_wins": max_con_win,
        "max_consecutive_losses": max_con_loss,
        "exit_reason_counts": exit_reason_counts,
    }


# ------------------------------------------------------------------
#  交易記錄轉 DataFrame / CSV 匯出
# ------------------------------------------------------------------


def trades_to_dataframe(trades: list, stock_id: str | None = None) -> pd.DataFrame:
    """將交易記錄列表轉為 DataFrame（含 holding_days 計算欄位）。

    支援 TradeRecord（單股回測）和 PortfolioTradeRecord（組合回測）。

    Parameters
    ----------
    trades : list
        交易記錄列表。
    stock_id : str | None
        單股回測時的 stock_id（TradeRecord 沒有 stock_id 屬性時補入）。
    """
    if not trades:
        return pd.DataFrame()

    rows = []
    for t in trades:
        row: dict = {}
        # PortfolioTradeRecord 有 stock_id；TradeRecord 沒有，需外部傳入
        sid = getattr(t, "stock_id", None) or stock_id
        if sid:
            row["stock_id"] = sid
        row["entry_date"] = getattr(t, "entry_date", None)
        row["exit_date"] = getattr(t, "exit_date", None)
        row["holding_days"] = _holding_days(row["entry_date"], row["exit_date"])
        row["entry_price"] = getattr(t, "entry_price", None)
        row["exit_price"] = getattr(t, "exit_price", None)
        row["shares"] = getattr(t, "shares", None)
        row["pnl"] = getattr(t, "pnl", None)
        row["return_pct"] = getattr(t, "return_pct", None)
        row["exit_reason"] = getattr(t, "exit_reason", None)
        # 可選欄位（僅 TradeRecord 有）
        stop = getattr(t, "stop_price", None)
        target = getattr(t, "target_price", None)
        if stop is not None:
            row["stop_price"] = stop
        if target is not None:
            row["target_price"] = target
        rows.append(row)

    return pd.DataFrame(rows)


def export_trades(
    trades: list,
    filepath: str,
    stock_id: str | None = None,
) -> str:
    """匯出交易明細至 CSV 檔案。

    Parameters
    ----------
    trades : list
        交易記錄列表。
    filepath : str
        輸出 CSV 檔案路徑。
    stock_id : str | None
        單股回測時的 stock_id。

    Returns
    -------
    str
        實際寫入的檔案路徑。
    """
    df = trades_to_dataframe(trades, stock_id=stock_id)
    if df.empty:
        raise ValueError("無交易記錄可匯出")
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    return filepath
