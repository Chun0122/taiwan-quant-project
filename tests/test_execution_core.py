"""src/portfolio/execution_core.py 純函數測試（路線 B1 + A4 交易現實化）。

驗證 simulate_sell / simulate_buy 的金額算式，數字守恆不變量（A4 後）：
- 賣出淨回收：proceeds == notional − Σ trade_cost_amounts(side="sell")
- 買入總支出：buy_cost == notional + Σ trade_cost_amounts(side="buy")
整張倍數且大額（min fee 不綁定、無零股 premium）時退化為舊比例式，仍保留等價測試。
"""

import pytest

from src.constants import COMMISSION_RATE, SLIPPAGE_RATE, TAX_RATE
from src.portfolio.execution_core import BuyFill, SellFill, simulate_buy, simulate_sell
from src.portfolio.rotation import trade_cost_amounts


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

    def test_proceeds_matches_legacy_form_for_whole_lots(self):
        """整張倍數且大額時（min fee 不綁定、無零股）退化為舊比例式。"""
        for entry, exit_p, shares, slip in [
            (50.0, 55.0, 2000, 0.0005),
            (123.45, 98.7, 3000, 0.0012),
        ]:
            fill = simulate_sell(entry, exit_p, shares, buy_slippage=0.0005, sell_slippage=slip)
            legacy = exit_p * shares * (1 - COMMISSION_RATE - TAX_RATE - slip)
            assert fill.proceeds == pytest.approx(legacy, rel=1e-12)

    def test_proceeds_conservation_invariant(self):
        """A4 守恆不變量：proceeds == notional − Σ trade_cost_amounts（含零股單）。"""
        for entry, exit_p, shares, slip in [
            (50.0, 55.0, 2000, 0.0005),
            (123.45, 98.7, 1300, 0.0012),  # 300 股零股尾數
            (10.0, 10.0, 999, 0.0),  # 全零股
        ]:
            fill = simulate_sell(entry, exit_p, shares, buy_slippage=0.0005, sell_slippage=slip)
            c, t, s = trade_cost_amounts(exit_p, shares, slip, side="sell")
            assert fill.proceeds == pytest.approx(exit_p * shares - (c + t + s), rel=1e-12)

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

    def test_buy_cost_matches_legacy_form_for_whole_lots(self):
        """整張倍數且大額時（min fee 不綁定、無零股）退化為舊比例式。"""
        for price, shares, slip in [
            (50.0, 2000, 0.0005),
            (123.45, 3000, 0.0012),
        ]:
            fill = simulate_buy(price, shares, slip)
            legacy = price * shares * (1 + COMMISSION_RATE + slip)
            assert fill.buy_cost == pytest.approx(legacy, rel=1e-12)

    def test_buy_cost_conservation_invariant(self):
        """A4 守恆不變量：buy_cost == notional + Σ trade_cost_amounts（含零股單）。"""
        for price, shares, slip in [
            (50.0, 2000, 0.0005),
            (123.45, 1300, 0.0012),  # 300 股零股尾數
            (10.0, 999, 0.0),  # 全零股
        ]:
            fill = simulate_buy(price, shares, slip)
            c, t, s = trade_cost_amounts(price, shares, slip, side="buy")
            assert t == 0.0  # 買入無交易稅
            assert fill.buy_cost == pytest.approx(price * shares + c + s, rel=1e-12)

    def test_default_slippage(self):
        fill = simulate_buy(100.0, 1000, SLIPPAGE_RATE)
        assert fill.buy_cost > 100.0 * 1000  # 成本必大於名目

    def test_zero_shares(self):
        fill = simulate_buy(100.0, 0, 0.0005)
        assert fill.buy_cost == 0.0
        assert fill.costs.total == 0.0
        assert fill.cash_delta == 0.0
