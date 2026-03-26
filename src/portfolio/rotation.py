"""輪動組合核心純函數 — 部位控制系統的 rotation 邏輯。

所有函數為純函數（無 DB / IO 副作用），方便單元測試。
由 RotationManager（manager.py）呼叫並負責 DB 讀寫。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

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

    # ── 計算空位並填補 ──
    free_slots = max_positions - len(remaining_open)
    if free_slots > 0 and new_rankings:
        held_ids = {p["stock_id"] for p in remaining_open}
        available_cash = current_cash
        # 加回賣出持倉的回收資金（粗略估算，manager 會精確計算）
        for sell_action in actions.to_sell:
            exit_p = sell_action.get("exit_price", sell_action.get("entry_price", 0))
            shares = sell_action.get("shares", 0)
            available_cash += exit_p * shares * (1 - COMMISSION_RATE - TAX_RATE - SLIPPAGE_RATE)

        per_position_capital = available_cash / max_positions if max_positions > 0 else 0

        for r in new_rankings:
            if free_slots <= 0:
                break
            sid = r["stock_id"]
            if sid in held_ids or sid in sold_today:
                continue
            price = r.get("close", 0)
            if price <= 0:
                continue
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
            free_slots -= 1

    return actions
