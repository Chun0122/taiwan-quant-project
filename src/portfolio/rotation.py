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

from src.constants import (
    COMMISSION_RATE,
    CORRELATION_PENALTY,
    CORRELATION_THRESHOLD,
    LIMIT_DETECT_THRESHOLD,
    LIQUIDITY_PARTICIPATION_LIMIT,
    MAX_DRAWDOWN_LIQUIDATE_PCT,
    MAX_PORTFOLIO_HEAT,
    PER_POSITION_RISK_CAP,
    SELL_SLIPPAGE_MULTIPLIER,
    SLIPPAGE_IMPACT_COEFF,
    SLIPPAGE_MAX_PCT,
    SLIPPAGE_RATE,
    SLIPPAGE_SPREAD_WEIGHT,
    TAX_RATE,
)

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
    # 本次呼叫產生的非止損換手次數（供 backtest 累積週預算使用）
    holding_expired_sells: int = 0


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
    buy_slippage: float = SLIPPAGE_RATE,
    sell_slippage: float = SLIPPAGE_RATE,
) -> tuple[float, float]:
    """計算單筆部位的已實現損益（含交易成本）。

    買入成本 = entry_price × shares × (1 + 手續費 + buy_slippage)
    賣出收入 = exit_price × shares × (1 - 手續費 - 交易稅 - sell_slippage)

    Parameters
    ----------
    buy_slippage : float
        買入滑價比率（預設 SLIPPAGE_RATE，向後相容）。
    sell_slippage : float
        賣出滑價比率（預設 SLIPPAGE_RATE，向後相容）。

    Returns
    -------
    (pnl, return_pct) : tuple[float, float]
        pnl = 賣出收入 - 買入成本（新台幣）。
        return_pct = pnl / 買入成本（小數，如 0.05 表示 5%）。
    """
    buy_cost = entry_price * shares * (1 + COMMISSION_RATE + buy_slippage)
    sell_proceeds = exit_price * shares * (1 - COMMISSION_RATE - TAX_RATE - sell_slippage)
    pnl = sell_proceeds - buy_cost
    return_pct = pnl / buy_cost if buy_cost > 0 else 0.0
    return round(pnl, 2), round(return_pct, 6)


def compute_shares(
    capital: float,
    price: float,
    slippage: float | None = None,
    daily_volume: float | None = None,
    participation_limit: float = LIQUIDITY_PARTICIPATION_LIMIT,
) -> int:
    """計算可買入的股數（台股以 1000 股為一張，但此處以股為單位）。

    扣除買入手續費與滑價後計算。回傳整數股數。
    可選流動性約束：計算後以 daily_volume × participation_limit 為上限。

    Parameters
    ----------
    capital : float
        可用資金。
    price : float
        買入價格。
    slippage : float | None
        滑價比率。None 時使用預設 SLIPPAGE_RATE。
    daily_volume : float | None
        當日成交量（股），用於流動性約束。None 時不約束。
    participation_limit : float
        流動性約束比例（預設 5%）。
    """
    if price <= 0:
        return 0
    slip = slippage if slippage is not None else SLIPPAGE_RATE
    effective_price = price * (1 + COMMISSION_RATE + slip)
    shares = int(capital / effective_price)
    if daily_volume is not None:
        shares = apply_liquidity_limit(shares, daily_volume, participation_limit)
    return shares


# ---------------------------------------------------------------------------
# 動態滑價 / 流動性約束 / 漲跌停偵測
# ---------------------------------------------------------------------------


def compute_dynamic_slippage(
    volume: float,
    high: float,
    low: float,
    close: float,
    side: str = "buy",
    base_slippage: float = SLIPPAGE_RATE,
    impact_coeff: float = SLIPPAGE_IMPACT_COEFF,
    spread_weight: float = SLIPPAGE_SPREAD_WEIGHT,
    max_pct: float = SLIPPAGE_MAX_PCT,
    sell_multiplier: float = SELL_SLIPPAGE_MULTIPLIER,
) -> float:
    """計算動態滑價比率（三因子模型，從 BacktestEngine._get_slippage 提取）。

    三因子：
      1. 基底滑價 base（0.05%）
      2. 成交量衝擊 k / sqrt(volume) — 低量股懲罰
      3. OHLC spread 估算 (high-low)/close × weight — 隱含 bid-ask spread

    Parameters
    ----------
    volume : float
        當日成交量（股）。volume <= 0 時 fallback 到 base_slippage。
    high, low, close : float
        當日 OHLC 價格。
    side : str
        "buy" 或 "sell"。賣出乘以 sell_multiplier。
    base_slippage : float
        基底滑價率。
    impact_coeff : float
        成交量衝擊係數 k。
    spread_weight : float
        OHLC spread 估算權重。
    max_pct : float
        滑價上限。
    sell_multiplier : float
        賣出滑價放大係數。

    Returns
    -------
    float
        滑價比率（正數），買入時 price × (1 + slip)，賣出時 price × (1 - slip)。
    """
    if volume <= 0:
        base_slip = base_slippage
    else:
        impact = base_slippage + impact_coeff / math.sqrt(volume)
        # OHLC spread proxy
        if close > 0 and high > low:
            spread_proxy = (high - low) / close * spread_weight
            impact = max(impact, spread_proxy)
        base_slip = min(impact, max_pct)

    if side == "sell":
        return min(base_slip * sell_multiplier, max_pct)
    return base_slip


def apply_liquidity_limit(
    shares: int,
    daily_volume: float,
    participation_limit: float = LIQUIDITY_PARTICIPATION_LIMIT,
) -> int:
    """流動性約束：限制單筆交易量不超過當日成交量的指定比例。

    Parameters
    ----------
    shares : int
        原始計算股數。
    daily_volume : float
        當日成交量（股）。<= 0 時不約束（passthrough）。
    participation_limit : float
        上限比例（預設 5%）。

    Returns
    -------
    int
        約束後的股數。
    """
    if daily_volume <= 0:
        return shares
    max_shares = int(daily_volume * participation_limit)
    if max_shares < 1:
        return 0
    return min(shares, max_shares)


def detect_limit_price(
    open_price: float,
    prev_close: float,
    threshold: float = LIMIT_DETECT_THRESHOLD,
) -> tuple[bool, bool]:
    """偵測漲跌停（從 BacktestEngine 提取）。

    Parameters
    ----------
    open_price : float
        當日開盤價。
    prev_close : float
        前日收盤價。
    threshold : float
        偵測門檻（預設 0.095，略低於 10% 以涵蓋四捨五入）。

    Returns
    -------
    (is_limit_up, is_limit_down) : tuple[bool, bool]
    """
    if prev_close <= 0:
        return False, False
    change = (open_price - prev_close) / prev_close
    return change >= threshold, change <= -threshold


@dataclass
class TradeCostBreakdown:
    """交易成本分解（記帳用途，不從 PnL 中重複扣除）。"""

    commission: float = 0.0
    tax: float = 0.0
    slippage_cost: float = 0.0
    total: float = 0.0


def compute_trade_costs(
    price: float,
    shares: int,
    slippage: float,
    side: str = "buy",
) -> TradeCostBreakdown:
    """計算單筆交易的成本分解。

    Parameters
    ----------
    price : float
        成交價格。
    shares : int
        股數。
    slippage : float
        滑價比率。
    side : str
        "buy" 或 "sell"。賣出時含交易稅。

    Returns
    -------
    TradeCostBreakdown
        含 commission, tax, slippage_cost, total。
    """
    notional = price * shares
    commission = notional * COMMISSION_RATE
    tax = notional * TAX_RATE if side == "sell" else 0.0
    slippage_cost = notional * slippage
    return TradeCostBreakdown(
        commission=round(commission, 2),
        tax=round(tax, 2),
        slippage_cost=round(slippage_cost, 2),
        total=round(commission + tax + slippage_cost, 2),
    )


# ---------------------------------------------------------------------------
# 組合風險預算（Portfolio Heat）
# ---------------------------------------------------------------------------


def compute_portfolio_heat(
    current_positions: list[dict],
    stop_losses: dict[str, float] | None,
    today_prices: dict[str, float] | None,
    total_capital: float,
    per_position_risk_cap: float = PER_POSITION_RISK_CAP,
) -> float:
    """計算當前組合風險百分比（Portfolio Heat）。

    heat = Σ max(0, (current_price - stop_loss) × shares) / total_capital

    若某筆持倉無 stop_loss，以 allocated_capital × per_position_risk_cap 估算。

    Parameters
    ----------
    current_positions : list[dict]
        目前 open 持倉，每筆含 stock_id, shares, allocated_capital, entry_price。
    stop_losses : dict[str, float] | None
        各股票的止損價 {stock_id: stop_loss_price}。
    today_prices : dict[str, float] | None
        今日各股收盤價 {stock_id: close}。
    total_capital : float
        組合總資本（current_capital）。
    per_position_risk_cap : float
        無停損時的單筆風險估算比例（預設 3%）。

    Returns
    -------
    float
        組合風險百分比（0.0~1.0），如 0.08 表示 8%。
    """
    if total_capital <= 0 or not current_positions:
        return 0.0

    stop_losses = stop_losses or {}
    today_prices = today_prices or {}
    total_risk = 0.0

    for pos in current_positions:
        sid = pos["stock_id"]
        shares = pos.get("shares", 0)
        price = today_prices.get(sid, pos.get("entry_price", 0))
        sl = stop_losses.get(sid)

        if sl is not None and price > 0 and shares > 0:
            # 有停損價：風險 = (現價 - 停損價) × 股數
            risk = max(0.0, (price - sl) * shares)
        else:
            # 無停損價：以 allocated_capital × risk_cap 估算
            alloc = pos.get("allocated_capital", 0)
            risk = alloc * per_position_risk_cap

        total_risk += risk

    return total_risk / total_capital


def compute_single_trade_risk(
    entry_price: float,
    stop_loss: float | None,
    shares: int,
    total_capital: float,
    per_position_risk_cap: float = PER_POSITION_RISK_CAP,
    allocated_capital: float = 0.0,
) -> float:
    """計算單筆新交易的風險百分比。"""
    if total_capital <= 0 or shares <= 0:
        return 0.0

    if stop_loss is not None and entry_price > 0:
        risk = max(0.0, (entry_price - stop_loss) * shares)
    else:
        risk = allocated_capital * per_position_risk_cap

    return risk / total_capital


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
    # Portfolio Heat 風險預算
    max_heat: float = MAX_PORTFOLIO_HEAT,
    per_position_risk_cap: float = PER_POSITION_RISK_CAP,
    total_capital: float | None = None,
    # Correlation Budget 相關性預算
    corr_matrix: pd.DataFrame | None = None,
    corr_threshold: float = CORRELATION_THRESHOLD,
    corr_penalty: float = CORRELATION_PENALTY,
    # 波動率反比權重
    vol_weights: dict[str, float] | None = None,
    # Regime 硬阻擋
    regime: str | None = None,
    crisis_block_new: bool = True,
    crisis_force_close: bool = False,
    # 成本閘門（問題 1 修正：降低高頻換手拖累）
    min_hold_days: int = 0,
    score_gap_threshold: float = 0.0,
    weekly_swap_cap: int = 0,
    weekly_swaps_used: int = 0,
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
        已棄用，保留以維持向後相容。連續化後僅使用 drawdown_stop_threshold。
    drawdown_stop_threshold : float
        回撤達此閾值（%）時 drawdown_scale 降至 0（預設 15.0）。
        scale = max(0, 1 - drawdown_pct / drawdown_stop_threshold)。
    max_heat : float
        組合最大風險上限（預設 0.12 = 12%），超過時拒絕新開倉。
    per_position_risk_cap : float
        單筆風險估算上限（預設 0.03 = 3%），無停損時使用。
    total_capital : float | None
        組合總資本，用於 Portfolio Heat 計算。None 時停用 Heat 檢查。
    corr_matrix : pd.DataFrame | None
        持倉相關性矩陣，用於 Correlation Budget。None 時停用。
    corr_threshold : float
        高相關判定門檻（預設 0.7）。
    corr_penalty : float
        高相關時部位縮減比例（預設 0.5 = 減半）。
    vol_weights : dict[str, float] | None
        波動率反比權重 {stock_id: weight}，合計 1.0。
        None 時等權分配。由 compute_vol_inverse_weights() 計算。
    regime : str | None
        目前市場狀態（bull/sideways/bear/crisis），用於 Crisis 硬阻擋。
    crisis_block_new : bool
        crisis 時是否阻擋所有新開倉（預設 True）。
    crisis_force_close : bool
        crisis 時是否強制平倉所有既有持倉（預設 False）。
        啟用時 regime='crisis' 會觸發全持倉賣出。

    Returns
    -------
    RotationActions
    """
    stop_losses = stop_losses or {}
    today_prices = today_prices or {}

    # 建立 ranking 集合（stock_id → ranking dict）
    ranked_ids = {r["stock_id"] for r in new_rankings}
    ranking_map = {r["stock_id"]: r for r in new_rankings}

    # ── 成本閘門 A：holding_days 安全下限（min_hold_days） ──
    # 當 min_hold_days > holding_days 時拉高到 min_hold_days，避免極短線換手
    effective_holding_days = max(holding_days, min_hold_days) if min_hold_days > 0 else holding_days

    # ── 成本閘門 C：計算本週剩餘可換手預算 ──
    # 僅計 holding_expired 類型（stop_loss/crisis_exit 不計入；安全優先）
    weekly_swap_budget_remaining: int | None = None
    if weekly_swap_cap > 0:
        weekly_swap_budget_remaining = max(0, weekly_swap_cap - weekly_swaps_used)

    actions = RotationActions()
    remaining_open: list[dict] = []
    sold_today: set[str] = set()
    weekly_swaps_this_call = 0  # 本次呼叫已產生的 holding_expired 賣出計數

    # 成本閘門 B：找出「若賣出此位置將被填補」的最佳新候選分數（供比較用）
    # 取 ranked_ids 中未持有者的 top score（即潛在替補者）
    held_ids_initial = {p["stock_id"] for p in current_positions}
    best_new_score: float | None = None
    if score_gap_threshold > 0.0:
        for r in new_rankings:
            if r["stock_id"] in held_ids_initial:
                continue
            s = r.get("score")
            if s is None:
                continue
            if best_new_score is None or s > best_new_score:
                best_new_score = s

    for pos in current_positions:
        sid = pos["stock_id"]
        days_held = count_trading_days_held(pos["entry_date"], today, trading_calendar)
        current_price = today_prices.get(sid)

        # ── 止損檢查（優先於持有期判斷，閘門豁免） ──
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

        # ── 持有期判斷（套用閘門 A：effective_holding_days）──
        expired = days_held >= effective_holding_days

        if expired:
            # ── 成本閘門 B：若新最佳候選分數與現持分差距不足，阻擋賣出 ──
            entry_score = pos.get("entry_score") or pos.get("composite_score")
            gate_b_block = (
                score_gap_threshold > 0.0
                and entry_score is not None
                and best_new_score is not None
                and (best_new_score - entry_score) < score_gap_threshold
            )

            # ── 成本閘門 C：本週換手預算已用盡 → 阻擋賣出 ──
            gate_c_block = weekly_swap_budget_remaining is not None and weekly_swap_budget_remaining <= 0

            # 閘門阻擋時：treat as hold（延長持有期、不觸發賣出也不新開倉）
            if gate_b_block or gate_c_block:
                reason = "gate_b_score_gap" if gate_b_block else "gate_c_weekly_cap"
                rank = ranking_map[sid]["rank"] if sid in ranked_ids else None
                actions.to_hold.append(
                    {
                        "stock_id": sid,
                        "days_held": days_held,
                        "rank": rank,
                        "gated_by": reason,
                        **pos,
                    }
                )
                remaining_open.append(pos)
                continue

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
                # 到期賣出（holding_expired 計入週換手預算）
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
                weekly_swaps_this_call += 1
                if weekly_swap_budget_remaining is not None:
                    weekly_swap_budget_remaining = max(0, weekly_swap_budget_remaining - 1)
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

    # ── Crisis 強制平倉：crisis 時清空所有既有持倉 ──
    if regime == "crisis" and crisis_force_close and remaining_open:
        for pos in remaining_open:
            price = (today_prices or {}).get(pos["stock_id"], pos["entry_price"])
            actions.to_sell.append(
                {
                    "stock_id": pos["stock_id"],
                    "reason": "crisis_exit",
                    "exit_price": price,
                    "days_held": count_trading_days_held(pos["entry_date"], today, trading_calendar),
                    **pos,
                }
            )
        remaining_open = []
        actions.to_hold = []  # 清除第一輪已加入的 to_hold

    # ── Crisis 硬阻擋：crisis 時直接停止新開倉 ──
    if regime == "crisis" and crisis_block_new:
        drawdown_scale = 0.0
    else:
        # ── Drawdown Guard（連續化）：回撤越深，新開倉縮減越多 ──
        # drawdown_scale = max(0, 1 - dd / threshold)，線性遞減
        drawdown_scale = 1.0
        if drawdown_pct is not None and drawdown_stop_threshold > 0:
            drawdown_scale = max(0.0, 1.0 - drawdown_pct / drawdown_stop_threshold)

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

        # ── Portfolio Heat：計算當前組合風險 ──
        current_heat = 0.0
        heat_enabled = total_capital is not None and total_capital > 0
        if heat_enabled:
            current_heat = compute_portfolio_heat(
                remaining_open, stop_losses, today_prices, total_capital, per_position_risk_cap
            )

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

            # ── 波動率反比權重：按 vol_weights 調整每股分配資金 ──
            adj_capital = per_position_capital
            if vol_weights and sid in vol_weights:
                # vol_weight × max_positions 使總資金不變（weight 合計 1.0）
                adj_capital = per_position_capital * vol_weights[sid] * max_positions

            # ── Correlation Budget：高相關持倉縮減部位（Regime 自適應）──
            # 危機/熊市時收緊相關性門檻，避免分散失效
            effective_corr_threshold = corr_threshold
            effective_corr_penalty = corr_penalty
            if regime == "crisis":
                from src.constants import CORRELATION_PENALTY_CRISIS, CORRELATION_THRESHOLD_CRISIS

                effective_corr_threshold = min(corr_threshold, CORRELATION_THRESHOLD_CRISIS)
                effective_corr_penalty = min(corr_penalty, CORRELATION_PENALTY_CRISIS)
            elif regime == "bear":
                from src.constants import CORRELATION_PENALTY_BEAR, CORRELATION_THRESHOLD_BEAR

                effective_corr_threshold = min(corr_threshold, CORRELATION_THRESHOLD_BEAR)
                effective_corr_penalty = min(corr_penalty, CORRELATION_PENALTY_BEAR)

            if corr_matrix is not None and not corr_matrix.empty:
                for existing_pos in remaining_open:
                    esid = existing_pos["stock_id"]
                    if esid in corr_matrix.columns and sid in corr_matrix.columns and esid != sid:
                        corr_val = corr_matrix.loc[esid, sid]
                        if not np.isnan(corr_val) and abs(corr_val) >= effective_corr_threshold:
                            adj_capital *= effective_corr_penalty
                            break  # 一次 penalty 即可

            # ── Portfolio Heat：檢查新交易是否超過風險上限 ──
            # 止損價優先從 ranking dict 取（discover 原始值），fallback 到 stop_losses 參數
            candidate_sl = r.get("stop_loss") or stop_losses.get(sid)
            tentative_shares = compute_shares(adj_capital, price)
            if heat_enabled and tentative_shares > 0:
                new_risk = compute_single_trade_risk(
                    price,
                    candidate_sl,
                    tentative_shares,
                    total_capital,
                    per_position_risk_cap,
                    adj_capital,
                )
                if current_heat + new_risk > max_heat:
                    # 嘗試縮小部位使 heat 剛好到上限
                    remaining_budget = max_heat - current_heat
                    if remaining_budget <= 0:
                        continue  # 風險預算已用完
                    if candidate_sl is not None and price > candidate_sl:
                        max_affordable_shares = int(remaining_budget * total_capital / (price - candidate_sl))
                    else:
                        max_affordable_shares = (
                            int(remaining_budget / per_position_risk_cap * adj_capital / price) if price > 0 else 0
                        )
                    tentative_shares = min(tentative_shares, max_affordable_shares)
                    if tentative_shares <= 0:
                        continue
                    # 重算 adj_capital
                    adj_capital = tentative_shares * price * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    new_risk = compute_single_trade_risk(
                        price,
                        candidate_sl,
                        tentative_shares,
                        total_capital,
                        per_position_risk_cap,
                        adj_capital,
                    )

            shares = tentative_shares if heat_enabled else compute_shares(adj_capital, price)
            if shares <= 0:
                continue

            # 更新累積 heat
            if heat_enabled:
                current_heat += compute_single_trade_risk(
                    price,
                    candidate_sl,
                    shares,
                    total_capital,
                    per_position_risk_cap,
                    adj_capital,
                )

            actions.to_buy.append(
                {
                    "stock_id": sid,
                    "stock_name": r.get("stock_name", ""),
                    "rank": r["rank"],
                    "score": r.get("score"),
                    "entry_price": price,
                    "shares": shares,
                    "allocated_capital": adj_capital,
                    "stop_loss": candidate_sl,
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

    actions.holding_expired_sells = weekly_swaps_this_call
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
    returns_df = prices_df.pct_change(fill_method=None).dropna()

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
# B2b: 組合層級 Ex-Ante VaR
# ---------------------------------------------------------------------------

# 95% 信賴水準對應的 z 值（避免 scipy import）
_Z_95 = 1.6449


def compute_covariance_matrix(
    price_data: dict[str, pd.Series],
    window: int = 60,
    min_periods: int = 20,
) -> pd.DataFrame:
    """計算持倉間的共變異數矩陣（從日報酬率）。

    與 compute_correlation_matrix() 平行，用於 VaR 計算。

    Parameters
    ----------
    price_data : dict[str, pd.Series]
        {stock_id: 收盤價 Series（日期索引）}。
    window : int
        滾動窗口天數（預設 60）。
    min_periods : int
        最小有效樣本數（預設 20），低於此值的估計不穩定。

    Returns
    -------
    pd.DataFrame
        covariance matrix（stock_id × stock_id）。空組合回傳空 DataFrame。
    """
    if len(price_data) < 1:
        return pd.DataFrame()

    prices_df = pd.DataFrame(price_data)
    returns_df = prices_df.pct_change(fill_method=None).dropna()

    if len(returns_df) < min_periods:
        return pd.DataFrame()

    recent = returns_df.tail(window) if len(returns_df) >= window else returns_df
    return recent.cov(min_periods=min_periods)


def compute_portfolio_var(
    position_weights: dict[str, float],
    covariance_matrix: pd.DataFrame,
    total_capital: float,
    confidence_z: float = _Z_95,
    horizon_days: int = 1,
) -> dict[str, float]:
    """計算組合層級的參數化 VaR（Ex-Ante）。

    VaR = z × σ_p × capital × √horizon
    σ_p = √(w^T × Σ × w)

    Parameters
    ----------
    position_weights : dict[str, float]
        {stock_id: 投資比例}，僅 invested 部分（不含現金）。
        權重合計可 < 1.0（部分現金），VaR 只算 invested 部分。
    covariance_matrix : pd.DataFrame
        日報酬率共變異數矩陣。
    total_capital : float
        組合總資本。
    confidence_z : float
        信賴水準 z 值（預設 1.6449 = 95%）。
    horizon_days : int
        預測天數（預設 1 天）。

    Returns
    -------
    dict[str, float]
        {"var_amount": 金額, "var_pct": 百分比, "component_var": {sid: 金額}}
    """
    empty_result: dict[str, float] = {"var_amount": 0.0, "var_pct": 0.0, "component_var": {}}

    if not position_weights or covariance_matrix.empty or total_capital <= 0:
        return empty_result

    # 取交集：只用共變異數矩陣中有的股票
    common_ids = [sid for sid in position_weights if sid in covariance_matrix.columns]
    if not common_ids:
        return empty_result

    # 建構權重向量
    w = np.array([position_weights[sid] for sid in common_ids], dtype=np.float64)
    cov = covariance_matrix.loc[common_ids, common_ids].values.astype(np.float64)

    # 正則化：避免奇異矩陣（高相關 ETF 等邊界情況）
    n = len(common_ids)
    cov = cov + np.eye(n) * 1e-8

    # 組合標準差
    portfolio_var_raw = float(w @ cov @ w)
    if portfolio_var_raw <= 0:
        return empty_result
    portfolio_std = math.sqrt(portfolio_var_raw)

    # VaR 金額
    sqrt_horizon = math.sqrt(horizon_days) if horizon_days > 1 else 1.0
    var_amount = confidence_z * portfolio_std * total_capital * sqrt_horizon
    var_pct = confidence_z * portfolio_std * sqrt_horizon * 100  # 百分比

    # Component VaR：各持倉對總 VaR 的貢獻
    sigma_w = cov @ w  # Σ × w
    component_var: dict[str, float] = {}
    for i, sid in enumerate(common_ids):
        if portfolio_std > 0:
            comp = w[i] * sigma_w[i] / portfolio_std * confidence_z * total_capital * sqrt_horizon
        else:
            comp = 0.0
        component_var[sid] = round(comp, 2)

    return {
        "var_amount": round(var_amount, 2),
        "var_pct": round(var_pct, 4),
        "component_var": component_var,
    }


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
        淨值序��（最新值在最後）。

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


def check_drawdown_kill_switch(
    equity_history: list[float],
    threshold_pct: float = MAX_DRAWDOWN_LIQUIDATE_PCT,
) -> bool:
    """檢查是否應觸發最大回撤熔斷（強制平倉所有部位）。

    Returns
    -------
    bool
        True = 回撤超過閾值，應立即平倉所有持倉。
    """
    dd = compute_portfolio_drawdown(equity_history)
    return dd >= threshold_pct
