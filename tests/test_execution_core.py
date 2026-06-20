"""src/portfolio/execution_core.py 純函數測試（路線 B1）。

驗證 simulate_sell / simulate_buy 的金額算式，並以「數字守恆」測試證明本核心與
重構前 live / backtest 兩條路徑使用的兩種寫法等價：
- 賣出淨回收：notional − costs.total  ==  notional × (1 − 手續費 − 交易稅 − 滑價)
- 買入總支出：notional + costs.total  ==  notional × (1 + 手續費 + 滑價)
"""

import pytest

from src.constants import COMMISSION_RATE, SLIPPAGE_RATE, TAX_RATE
from src.portfolio.execution_core import BuyFill, SellFill, simulate_buy, simulate_sell


class TestSimulateSell:
    def test_basic_amounts(self):
        fill = simulate_sell(100.0, 110.0, 1000, buy_slippage=0.0005, sell_slippage=0.0005)
        notional = 110.0 * 1000
        assert isinstance(fill, SellFill)
        assert fill.costs.commission == pytest.approx(notional * COMMISSION_RATE)
        assert fill.costs.tax == pytest.approx(notional * TAX_RATE)  # 賣出含交易稅
        assert fill.costs.slippage_cost == pytest.approx(notional * 0.0005)
        assert fill.costs.total == pytest.approx(notional * (COMMISSION_RATE + TAX_RATE + 0.0005))
        assert fill.proceeds == pytest.approx(notional - fill.costs.total)
        assert fill.cash_delta == fill.proceeds  # 賣出對現金為正

    def test_proceeds_matches_legacy_backtest_form(self):
        """數字守恆：與重構前 backtest 的 notional×(1−費率) 寫法等價。"""
        for entry, exit_p, shares, slip in [
            (50.0, 55.0, 2000, 0.0005),
            (123.45, 98.7, 1300, 0.0012),
            (10.0, 10.0, 999, 0.0),
        ]:
            fill = simulate_sell(entry, exit_p, shares, buy_slippage=0.0005, sell_slippage=slip)
            legacy = exit_p * shares * (1 - COMMISSION_RATE - TAX_RATE - slip)
            assert fill.proceeds == pytest.approx(legacy, rel=1e-12)

    def test_loss_is_negative(self):
        fill = simulate_sell(100.0, 80.0, 1000, buy_slippage=0.0005, sell_slippage=0.0005)
        assert fill.pnl < 0
        assert fill.return_pct < 0

    def test_zero_shares(self):
        fill = simulate_sell(100.0, 110.0, 0, buy_slippage=0.0005, sell_slippage=0.0005)
        assert fill.proceeds == 0.0
        assert fill.costs.total == 0.0
        assert fill.cash_delta == 0.0


class TestSimulateBuy:
    def test_basic_amounts(self):
        fill = simulate_buy(100.0, 1000, 0.0005)
        notional = 100.0 * 1000
        assert isinstance(fill, BuyFill)
        assert fill.costs.commission == pytest.approx(notional * COMMISSION_RATE)
        assert fill.costs.tax == 0.0  # 買入不課交易稅
        assert fill.costs.slippage_cost == pytest.approx(notional * 0.0005)
        assert fill.buy_cost == pytest.approx(notional + fill.costs.total)
        assert fill.cash_delta == pytest.approx(-fill.buy_cost)  # 買入對現金為負

    def test_buy_cost_matches_legacy_backtest_form(self):
        """數字守恆：與重構前 backtest 的 notional×(1+費率) 寫法等價。"""
        for price, shares, slip in [
            (50.0, 2000, 0.0005),
            (123.45, 1300, 0.0012),
            (10.0, 999, 0.0),
        ]:
            fill = simulate_buy(price, shares, slip)
            legacy = price * shares * (1 + COMMISSION_RATE + slip)
            assert fill.buy_cost == pytest.approx(legacy, rel=1e-12)

    def test_default_slippage(self):
        fill = simulate_buy(100.0, 1000, SLIPPAGE_RATE)
        assert fill.buy_cost > 100.0 * 1000  # 成本必大於名目

    def test_zero_shares(self):
        fill = simulate_buy(100.0, 0, 0.0005)
        assert fill.buy_cost == 0.0
        assert fill.costs.total == 0.0
        assert fill.cash_delta == 0.0
