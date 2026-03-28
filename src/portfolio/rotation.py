"""輪動組合核心純函數 — 部位控制系統的 rotation 邏輯。

所有函數為純函數（無 DB / IO 副作用），方便單元測試。
由 RotationManager（manager.py）呼叫並負責 DB 讀寫。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.constants import COMMISSION_RATE, SLIPPAGE_RATE, TAX_RATE

# ---------------------------------------------------------------------------
# 資料結構
# ---------------------------------------------------------------------------


@dataclass
class RotationActions:
    """compute_rotation_actions() 的回傳結果。"""

    to_sell: list[dict] = field(default_factory=list)
    to_buy: list[dict] = field(default_factory=list)
    to_hold: list[dict] = field(default_factory=list)
    renewed: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 交易日工具
# ---------------------------------------------------------------------------


def get_trading_dates_from_prices(
    price_dates: list[date],
    start: date,
    end: date,
) -> list[date]:
    """從已排序的 DailyPrice 日期清單中擷取 [start, end] 區間的交易日。

    Parameters
    ----------
    price_dates : list[date]
        已排序的交易日清單（通常為 TAIEX 的所有日期）。
    start, end : date
        查詢起迄日。

    Returns
    -------
    list[date]
        區間內的交易日（含首尾），已排序。
    """
    return [d for d in price_dates if start <= d <= end]


def compute_planned_exit_date(
    entry_date: date,
    holding_days: int,
    trading_calendar: list[date],
) -> date:
    """計算預計到期日（以交易日計算 holding_days 天後）。

    Parameters
    ----------
    entry_date : date
        進場日。
    holding_days : int
        持有天數。
    trading_calendar : list[date]
        已排序的交易日清單。

    Returns
    -------
    date
        預計到期日。若 trading_calendar 不足則用簡單工作日推算。
    """
    # 找到 entry_date 在 calendar 中的位置（含當日）
    future = [d for d in trading_calendar if d >= entry_date]
    if len(future) > holding_days:
        return future[holding_days]
    # fallback: 簡單推算（假設每 7 天有 5 個交易日）
    extra_days = holding_days - len(future) + 1
    calendar_days = math.ceil(extra_days * 7 / 5)
    last_known = future[-1] if future else entry_date
    return last_known + timedelta(days=calendar_days)


def count_trading_days_held(
    entry_date: date,
    today: date,
    trading_calendar: list[date],
) -> int:
    """計算已持有的交易日數（不含 entry_date 當天，含 today）。

    例：entry=Mon, today=Wed → Mon(0), Tue(1), Wed(2) → 回傳 2。
    """
    return len([d for d in trading_calendar if entry_date < d <= today])


# ---------------------------------------------------------------------------
# 損益計算
# ---------------------------------------------------------------------------


def compute_position_pnl(
    entry_price: float,
    exit_price: float,
    shares: int,
) -> tuple[float, float]:
    """計算單筆部位的已實現損益（含交易成本）。

    買入成本 = entry_price × shares × (1 + 手續費 + 滑價)
    賣出收入 = exit_price × shares × (1 - 手續費 - 交易稅 - 滑價)

    Returns
    -------
    (pnl, return_pct) : tuple[float, float]
        pnl = 賣出收入 - 買入成本（新台幣）。
        return_pct = pnl / 買入成本（小數，如 0.05 表示 5%）。
    """
    buy_cost = entry_price * shares * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
    sell_proceeds = exit_price * shares * (1 - COMMISSION_RATE - TAX_RATE - SLIPPAGE_RATE)
    pnl = sell_proceeds - buy_cost
    return_pct = pnl / buy_cost if buy_cost > 0 else 0.0
    return round(pnl, 2), round(return_pct, 6)


def compute_shares(capital: float, price: float) -> int:
    """計算可買入的股數（台股以 1000 股為一張，但此處以股為單位）。

    扣除買入手續費與滑價後計算。回傳整數股數。
    """
    if price <= 0:
        return 0
    effective_price = price * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
    return int(capital / effective_price)


# ---------------------------------------------------------------------------
# 核心 Rotation 演算法
# ---------------------------------------------------------------------------


def compute_rotation_actions(
    current_positions: list[dict],
    new_rankings: list[dict],
    max_positions: int,
    holding_days: int,
    allow_renewal: bool,
    today: date,
    trading_calendar: list[date],
    current_cash: float,
    stop_losses: dict[str, float] | None = None,
    today_prices: dict[str, float] | None = None,
    sector_map: dict[str, str] | None = None,
    max_sector_pct: float = 0.30,
    drawdown_pct: float | None = None,
    drawdown_half_threshold: float = 10.0,
    drawdown_stop_threshold: float = 15.0,
) -> RotationActions:
    """根據目前持倉與今日 discover 排名，計算買賣動作。

    Parameters
    ----------
    current_positions : list[dict]
        目前 open 持倉，每筆至少含：
        {stock_id, entry_date, entry_price, shares, allocated_capital, entry_rank}
    new_rankings : list[dict]
        今日 discover Top-N 排名（已按排名排序），每筆至少含：
        {stock_id, rank, score, close, stock_name, stop_loss(optional)}
    max_positions : int
        最大持股數。
    holding_days : int
        固定持有天數。
    allow_renewal : bool
        到期時是否允許續持。
    today : date
        今日日期。
    trading_calendar : list[date]
        已排序的交易日清單。
    current_cash : float
        目前可用現金。
    stop_losses : dict[str, float] | None
        各股票的止損價 {stock_id: stop_loss_price}。
    today_prices : dict[str, float] | None
        今日各股收盤價 {stock_id: close}。
    sector_map : dict[str, str] | None
        股票→產業映射 {stock_id: sector}，用於產業集中度限制。
    max_sector_pct : float
        同產業持股比例上限（預設 0.30 = 30%）。
    drawdown_pct : float | None
        目前組合回撤百分比（正值，如 12.0 = -12%），用於 Drawdown Guard。
    drawdown_half_threshold : float
        回撤達此閾值（%）時新開倉減半（預設 10.0）。
    drawdown_stop_threshold : float
        回撤達此閾值（%）時停止新開倉（預設 15.0）。

    Returns
    -------
    RotationActions
    """
    stop_losses = stop_losses or {}
    today_prices = today_prices or {}

    # 建立 ranking 集合（stock_id → ranking dict）
    ranked_ids = {r["stock_id"] for r in new_rankings}
    ranking_map = {r["stock_id"]: r for r in new_rankings}

    actions = RotationActions()
    remaining_open: list[dict] = []
    sold_today: set[str] = set()

    for pos in current_positions:
        sid = pos["stock_id"]
        days_held = count_trading_days_held(pos["entry_date"], today, trading_calendar)
        current_price = today_prices.get(sid)

        # ── 止損檢查（優先於持有期判斷）──
        sl = stop_losses.get(sid)
        if sl is not None and current_price is not None and current_price <= sl:
            actions.to_sell.append(
                {
                    "stock_id": sid,
                    "reason": "stop_loss",
                    "exit_price": current_price,
                    "days_held": days_held,
                    **pos,
                }
            )
            sold_today.add(sid)
            continue

        # ── 持有期判斷 ──
        expired = days_held >= holding_days

        if expired:
            if allow_renewal and sid in ranked_ids:
                # 續持：仍在 Top-N，延長持有期
                new_exit = compute_planned_exit_date(today, holding_days, trading_calendar)
                actions.renewed.append(
                    {
                        "stock_id": sid,
                        "new_planned_exit_date": new_exit,
                        "days_held": days_held,
                        "rank": ranking_map[sid]["rank"],
                        **pos,
                    }
                )
                remaining_open.append(pos)
            else:
                # 到期賣出
                exit_price = current_price if current_price is not None else pos["entry_price"]
                actions.to_sell.append(
                    {
                        "stock_id": sid,
                        "reason": "holding_expired",
                        "exit_price": exit_price,
                        "days_held": days_held,
                        **pos,
                    }
                )
                sold_today.add(sid)
        else:
            # 未到期：保持持倉
            rank = ranking_map[sid]["rank"] if sid in ranked_ids else None
            actions.to_hold.append(
                {
                    "stock_id": sid,
                    "days_held": days_held,
                    "rank": rank,
                    **pos,
                }
            )
            remaining_open.append(pos)

    # ── Drawdown Guard：回撤嚴重時限制新開倉 ──
    drawdown_scale = 1.0  # 新開倉資金倍數
    if drawdown_pct is not None:
        if drawdown_pct >= drawdown_stop_threshold:
            drawdown_scale = 0.0  # 停止新開倉
        elif drawdown_pct >= drawdown_half_threshold:
            drawdown_scale = 0.5  # 新開倉減半

    # ── 計算空位並填補 ──
    free_slots = max_positions - len(remaining_open)
    if free_slots > 0 and new_rankings and drawdown_scale > 0:
        held_ids = {p["stock_id"] for p in remaining_open}
        available_cash = current_cash
        # 加回賣出持倉的回收資金（粗略估算，manager 會精確計算）
        for sell_action in actions.to_sell:
            exit_p = sell_action.get("exit_price", sell_action.get("entry_price", 0))
            shares = sell_action.get("shares", 0)
            available_cash += exit_p * shares * (1 - COMMISSION_RATE - TAX_RATE - SLIPPAGE_RATE)

        per_position_capital = available_cash / max_positions if max_positions > 0 else 0
        per_position_capital *= drawdown_scale  # Drawdown Guard 縮減

        # 建構已持有的產業分佈（用於集中度檢查）
        sector_counts: dict[str, int] = {}
        if sector_map:
            for p in remaining_open:
                sec = sector_map.get(p["stock_id"], "")
                if sec:
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1

        max_sector_count = max(1, int(max_positions * max_sector_pct))

        for r in new_rankings:
            if free_slots <= 0:
                break
            sid = r["stock_id"]
            if sid in held_ids or sid in sold_today:
                continue
            price = r.get("close", 0)
            if price <= 0:
                continue

            # 產業集中度檢查
            if sector_map:
                candidate_sector = sector_map.get(sid, "")
                if candidate_sector and sector_counts.get(candidate_sector, 0) >= max_sector_count:
                    continue  # 同產業已達上限，跳過

            shares = compute_shares(per_position_capital, price)
            if shares <= 0:
                continue
            actions.to_buy.append(
                {
                    "stock_id": sid,
                    "stock_name": r.get("stock_name", ""),
                    "rank": r["rank"],
                    "score": r.get("score"),
                    "entry_price": price,
                    "shares": shares,
                    "allocated_capital": per_position_capital,
                }
            )
            held_ids.add(sid)
            if sector_map:
                sec = sector_map.get(sid, "")
                if sec:
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
            free_slots -= 1
    elif drawdown_scale == 0.0:
        # Drawdown Guard: 停止新開倉，但仍保持其餘邏輯
        pass

    return actions


# ---------------------------------------------------------------------------
# B2: 持倉相關性監控
# ---------------------------------------------------------------------------


def compute_correlation_matrix(
    price_data: dict[str, pd.Series],
    window: int = 60,
) -> pd.DataFrame:
    """計算持倉間的 rolling correlation matrix。

    Parameters
    ----------
    price_data : dict[str, pd.Series]
        {stock_id: 收盤價 Series（日期索引）}。
    window : int
        滾動窗口天數（預設 60）。

    Returns
    -------
    pd.DataFrame
        correlation matrix（stock_id × stock_id）。
    """
    if len(price_data) < 2:
        return pd.DataFrame()

    # 建構收盤價 DataFrame，取各股最近 window 天的交集
    prices_df = pd.DataFrame(price_data)
    returns_df = prices_df.pct_change().dropna()

    if len(returns_df) < window:
        # 資料不足，用可用資料計算
        return returns_df.corr()

    # 取最近 window 天
    recent = returns_df.tail(window)
    return recent.corr()


def find_high_correlation_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    """從 correlation matrix 中找出高相關配對。

    Returns
    -------
    list[tuple[str, str, float]]
        [(stock_a, stock_b, correlation), ...] 按相關性降序排列。
    """
    if corr_matrix.empty:
        return []

    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val) and abs(val) >= threshold:
                pairs.append((cols[i], cols[j], round(float(val), 4)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# B3: 波動率反比部位大小
# ---------------------------------------------------------------------------


def compute_vol_inverse_weights(
    volatilities: dict[str, float],
) -> dict[str, float]:
    """根據 realized volatility 計算反比權重。

    波動率大的股票分配較少資金，波動率小的分配較多。

    Parameters
    ----------
    volatilities : dict[str, float]
        {stock_id: 20日 realized volatility（年化 std）}。
        值為 0 或負數的會被排除。

    Returns
    -------
    dict[str, float]
        {stock_id: weight}，權重合為 1.0。
    """
    valid = {sid: vol for sid, vol in volatilities.items() if vol > 0}
    if not valid:
        return {}

    # 反比：weight_i ∝ 1 / vol_i
    inv = {sid: 1.0 / vol for sid, vol in valid.items()}
    total = sum(inv.values())
    return {sid: round(w / total, 6) for sid, w in inv.items()}


# ---------------------------------------------------------------------------
# B4 support: 計算組合回撤
# ---------------------------------------------------------------------------


def compute_portfolio_drawdown(
    equity_history: list[float],
) -> float:
    """計算當前回撤百分比（正值）。

    Parameters
    ----------
    equity_history : list[float]
        淨值序列（最新值在最後）。

    Returns
    -------
    float
        當前回撤百分比（0.0~100.0），0.0 = 在高點。
    """
    if not equity_history or len(equity_history) < 2:
        return 0.0
    peak = max(equity_history)
    current = equity_history[-1]
    if peak <= 0:
        return 0.0
    dd = (peak - current) / peak * 100
    return round(max(dd, 0.0), 2)
