"""輪動成交模擬核心（live 與 backtest 共用的純函數）。

設計動機（audit P2 / 2026-06-20 路線 B1）：
`RotationManager.update()`（live，寫 ORM）與 `RotationManager.backtest()`（in-memory dict）
各自維護一份買賣執行迴圈，其中「金額算式」——已實現損益、交易成本、賣出淨回收現金、
買入總現金支出——在兩處重複，是兩條路徑數字 drift 的來源。本模組把這段「錢的算式」
抽成單一純函數 + 標準成交結果（fill），讓兩條路徑共用同一份金額計算。

A4 交易現實化（2026-07-07）：成本改走 `trade_cost_amounts`（混合單模型——整張單 +
盤中零股單各計最低手續費 20/1 元、零股部分加滑價 premium）。最低手續費是固定成本，
金額不再能用「notional × (1±費率)」比例式表達，故 buy_cost / proceeds 一律以
「notional ± 未捨入成本合計」導出。新守恆不變量：
  buy_cost == notional + Σ trade_cost_amounts(side="buy")
  proceeds == notional − Σ trade_cost_amounts(side="sell")
live 與 backtest 皆呼叫本模組 → parity 由建構保證。

**刻意不納入本核心**（兩條路徑合理且刻意的差異，非 bug，保留在各 caller）：
- 股數定價（live 用 compute_rotation_actions 已算好的 shares + apply_liquidity_limit 下調；
  backtest 用 compute_shares 從 allocated_capital 重算，含 participation_limit）。
- 滑價來源（dynamic_slippage 旗標 / compute_dynamic_slippage vs 固定 SLIPPAGE_RATE）。
- 漲跌停模擬（limit_price_check）、survivorship 回填、現金不足 reshrink 迴圈。

純函數：無 DB、無 IO、無全域狀態。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.portfolio.rotation import (
    TradeCostBreakdown,
    compute_position_pnl,
    compute_trade_costs,
    trade_cost_amounts,
)


@dataclass(frozen=True)
class SellFill:
    """單筆賣出成交結果（金額面，與狀態無關）。"""

    pnl: float
    return_pct: float
    costs: TradeCostBreakdown
    proceeds: float  # 賣出淨回收現金 = 成交金額 − 成本合計

    @property
    def cash_delta(self) -> float:
        """對現金的影響（賣出為正）。"""
        return self.proceeds


@dataclass(frozen=True)
class BuyFill:
    """單筆買入成交結果（金額面，與狀態無關）。"""

    costs: TradeCostBreakdown
    buy_cost: float  # 買入總現金支出 = 成交金額 + 成本合計

    @property
    def cash_delta(self) -> float:
        """對現金的影響（買入為負）。"""
        return -self.buy_cost


def simulate_sell(
    entry_price: float,
    exit_price: float,
    shares: int,
    *,
    buy_slippage: float,
    sell_slippage: float,
) -> SellFill:
    """模擬一筆賣出的金額結果。

    pnl 走 compute_position_pnl（含買賣雙邊成本），淨回收 proceeds = 成交金額 −
    未捨入賣端成本合計（trade_cost_amounts，A4 混合單模型）。costs 為 round 後
    檢視，僅供成本累計 / instrumentation。
    """
    pnl, return_pct = compute_position_pnl(
        entry_price,
        exit_price,
        shares,
        buy_slippage=buy_slippage,
        sell_slippage=sell_slippage,
    )
    costs = compute_trade_costs(exit_price, shares, sell_slippage, side="sell")
    c, t, s = trade_cost_amounts(exit_price, shares, sell_slippage, side="sell")
    proceeds = exit_price * shares - (c + t + s)
    return SellFill(pnl=pnl, return_pct=return_pct, costs=costs, proceeds=proceeds)


def simulate_buy(price: float, shares: int, slippage: float) -> BuyFill:
    """模擬一筆買入的金額結果。

    總支出 buy_cost = 成交金額 + 未捨入買端成本合計（trade_cost_amounts，A4 混合單
    模型：最低手續費 + 零股 premium）。costs 為 round 後檢視，僅供成本累計 /
    instrumentation。股數定價 / 流動性 / 現金不足 reshrink 由 caller 處理後再傳入。
    """
    costs = compute_trade_costs(price, shares, slippage, side="buy")
    c, t, s = trade_cost_amounts(price, shares, slippage, side="buy")
    buy_cost = price * shares + c + t + s
    return BuyFill(costs=costs, buy_cost=buy_cost)
