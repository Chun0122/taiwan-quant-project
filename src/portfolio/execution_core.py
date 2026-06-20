"""輪動成交模擬核心（live 與 backtest 共用的純函數）。

設計動機（audit P2 / 2026-06-20 路線 B1）：
`RotationManager.update()`（live，寫 ORM）與 `RotationManager.backtest()`（in-memory dict）
各自維護一份買賣執行迴圈，其中「金額算式」——已實現損益、交易成本、賣出淨回收現金、
買入總現金支出——在兩處重複，且以不同寫法表達（live 用 `notional - costs.total`，
backtest 用 `notional * (1 - 費率)`），是兩條路徑數字 drift 的來源。

本模組把這段「錢的算式」抽成單一純函數 + 標準成交結果（fill），讓兩條路徑共用同一份
金額計算；各自只負責把 fill 翻譯成自己的狀態（live 寫 RotationPosition row、backtest
更新 positions dict / all_trades）。

**刻意不納入本核心**（兩條路徑合理且刻意的差異，非 bug，保留在各 caller）：
- 股數定價（live 用 compute_rotation_actions 已算好的 shares + apply_liquidity_limit 下調；
  backtest 用 compute_shares 從 allocated_capital 重算，含 participation_limit）。
- 滑價來源（dynamic_slippage 旗標 / compute_dynamic_slippage vs 固定 SLIPPAGE_RATE）。
- 漲跌停模擬（limit_price_check）、survivorship 回填、現金不足 reshrink 迴圈。

純函數：無 DB、無 IO、無全域狀態。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.constants import COMMISSION_RATE, TAX_RATE
from src.portfolio.rotation import (
    TradeCostBreakdown,
    compute_position_pnl,
    compute_trade_costs,
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

    與 live `_execute_sell` / backtest `_execute_action_set` 賣出分支金額算式一致：
    pnl 走 compute_position_pnl（含買賣雙邊滑價），成本走 compute_trade_costs（side=sell，含交易稅），
    淨回收 proceeds = 成交金額 − 成本合計。
    """
    pnl, return_pct = compute_position_pnl(
        entry_price,
        exit_price,
        shares,
        buy_slippage=buy_slippage,
        sell_slippage=sell_slippage,
    )
    costs = compute_trade_costs(exit_price, shares, sell_slippage, side="sell")
    # 淨回收用「未四捨五入」名目計算（與重構前 backtest 的 notional×(1−費率) 寫法等價，
    # 保證回測數字守恆）；costs 本身為 round 後值，僅供成本累計 / instrumentation。
    proceeds = exit_price * shares * (1 - COMMISSION_RATE - TAX_RATE - sell_slippage)
    return SellFill(pnl=pnl, return_pct=return_pct, costs=costs, proceeds=proceeds)


def simulate_buy(price: float, shares: int, slippage: float) -> BuyFill:
    """模擬一筆買入的金額結果。

    與 live `_execute_buy` / backtest `_execute_action_set` 買入分支金額算式一致：
    成本走 compute_trade_costs（side=buy，無交易稅），總支出 buy_cost = 成交金額 + 成本合計。
    股數定價 / 流動性 / 現金不足 reshrink 由 caller 處理後再傳入本函數。
    """
    costs = compute_trade_costs(price, shares, slippage, side="buy")
    # 總支出用「未四捨五入」名目計算（與重構前 backtest 的 notional×(1+費率) 寫法等價，
    # 保證回測數字守恆）；costs 本身為 round 後值，僅供成本累計 / instrumentation。
    buy_cost = price * shares * (1 + COMMISSION_RATE + slippage)
    return BuyFill(costs=costs, buy_cost=buy_cost)
