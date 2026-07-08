"""A4 交易現實化測試 — 混合單成本模型 + 最低手續費 + participation impact。

涵蓋（全部純函數，零 mock）：
1. split_lot_odd 整張/零股拆分邊界
2. trade_cost_amounts 最低手續費綁定（整股 20 / 零股 1）與零股滑價 premium
3. compute_dynamic_slippage participation impact（單調性 / 向後相容 / cap / sell 交互）
4. 守恆不變量：simulate_buy/sell 與 compute_position_pnl 對帳、compute_shares 可負擔性
5. BacktestEngine 旗標：關 = 舊比例式 bit-for-bit；開 = min fee / premium / participation 生效
"""

from __future__ import annotations

import math

import pytest

from src.constants import (
    COMMISSION_RATE,
    LOT_SIZE,
    MIN_COMMISSION_LOT,
    MIN_COMMISSION_ODD,
    ODD_LOT_SLIPPAGE_PREMIUM,
    PARTICIPATION_IMPACT_COEFF,
    SELL_SLIPPAGE_MULTIPLIER,
    SLIPPAGE_MAX_PCT,
    TAX_RATE,
)
from src.portfolio.execution_core import simulate_buy, simulate_sell
from src.portfolio.rotation import (
    compute_dynamic_slippage,
    compute_position_pnl,
    compute_shares,
    compute_trade_costs,
    split_lot_odd,
    trade_cost_amounts,
)

# ---------------------------------------------------------------------------
# 1. split_lot_odd
# ---------------------------------------------------------------------------


class TestSplitLotOdd:
    @pytest.mark.parametrize(
        ("shares", "expected"),
        [
            (0, (0, 0)),
            (-5, (0, 0)),
            (1, (0, 1)),
            (999, (0, 999)),
            (1000, (1000, 0)),
            (1001, (1000, 1)),
            (5000, (5000, 0)),
            (5432, (5000, 432)),
        ],
    )
    def test_boundaries(self, shares, expected):
        assert split_lot_odd(shares) == expected

    def test_custom_lot_size(self):
        assert split_lot_odd(250, lot_size=100) == (200, 50)

    def test_parts_sum_to_total(self):
        for shares in [1, 999, 1000, 1234, 10001]:
            lot, odd = split_lot_odd(shares)
            assert lot + odd == shares
            assert lot % LOT_SIZE == 0
            assert 0 <= odd < LOT_SIZE


# ---------------------------------------------------------------------------
# 2. trade_cost_amounts：最低手續費 + 零股 premium
# ---------------------------------------------------------------------------


class TestTradeCostAmounts:
    def test_large_lot_order_is_proportional(self):
        """大額整張單：比例手續費 > 最低費 → 與舊模型一致，premium=0。"""
        c, t, s = trade_cost_amounts(100.0, 2000, 0.0005, side="buy")
        assert c == pytest.approx(100.0 * 2000 * COMMISSION_RATE)
        assert t == 0.0
        assert s == pytest.approx(100.0 * 2000 * 0.0005)

    def test_small_lot_order_hits_min_fee(self):
        """小額整張單（notional < 20/0.001425 ≈ 14,035）→ 手續費 20 元。"""
        c, _, _ = trade_cost_amounts(10.0, 1000, 0.0, side="buy")  # notional 10,000
        assert c == MIN_COMMISSION_LOT

    def test_small_odd_order_hits_min_fee(self):
        """小額零股單（notional < 1/0.001425 ≈ 702）→ 手續費 1 元。"""
        c, _, _ = trade_cost_amounts(10.0, 50, 0.0, side="buy")  # notional 500
        assert c == MIN_COMMISSION_ODD

    def test_mixed_order_two_min_fees(self):
        """混合單且兩邊皆小額：min fee 各計一次（20 + 1）。"""
        # 1050 股 × 10 元：整張 10,000（<14,035 → 20）+ 零股 500（<702 → 1）
        c, _, _ = trade_cost_amounts(10.0, 1050, 0.0, side="buy")
        assert c == MIN_COMMISSION_LOT + MIN_COMMISSION_ODD

    def test_odd_premium_only_on_odd_notional(self):
        """零股 premium 只算零股部分的 notional。"""
        price, slip = 100.0, 0.0005
        _, _, s_lot_only = trade_cost_amounts(price, 2000, slip)
        assert s_lot_only == pytest.approx(price * 2000 * slip)  # 無零股 → 無 premium
        _, _, s_mixed = trade_cost_amounts(price, 2300, slip)
        assert s_mixed == pytest.approx(price * 2300 * slip + price * 300 * ODD_LOT_SLIPPAGE_PREMIUM)

    def test_sell_tax_on_full_notional(self):
        """交易稅整張/零股同率，對全額 notional 計。"""
        _, t, _ = trade_cost_amounts(100.0, 1300, 0.0005, side="sell")
        assert t == pytest.approx(100.0 * 1300 * TAX_RATE)

    def test_zero_and_negative_inputs(self):
        assert trade_cost_amounts(100.0, 0, 0.0005) == (0.0, 0.0, 0.0)
        assert trade_cost_amounts(0.0, 100, 0.0005) == (0.0, 0.0, 0.0)

    def test_rate_overrides(self):
        """BacktestEngine 路徑：commission_rate/tax_rate 覆蓋生效。"""
        c, t, _ = trade_cost_amounts(100.0, 2000, 0.0, side="sell", commission_rate=0.001, tax_rate=0.002)
        assert c == pytest.approx(100.0 * 2000 * 0.001)
        assert t == pytest.approx(100.0 * 2000 * 0.002)

    def test_compute_trade_costs_is_rounded_view(self):
        """compute_trade_costs == trade_cost_amounts 的 round 檢視。"""
        c, t, s = trade_cost_amounts(123.45, 1300, 0.0012, side="sell")
        breakdown = compute_trade_costs(123.45, 1300, 0.0012, side="sell")
        assert breakdown.commission == round(c, 2)
        assert breakdown.tax == round(t, 2)
        assert breakdown.slippage_cost == round(s, 2)
        assert breakdown.total == round(c + t + s, 2)


# ---------------------------------------------------------------------------
# 3. participation impact
# ---------------------------------------------------------------------------


class TestParticipationImpact:
    VOL = 1_000_000

    def _slip(self, order_shares=None):
        # high=low=close → spread proxy 不觸發，隔離 participation 項
        return compute_dynamic_slippage(self.VOL, 100.0, 100.0, 100.0, order_shares=order_shares)

    def test_none_order_shares_backward_compatible(self):
        """未傳 order_shares → 與舊三因子模型完全相同。"""
        assert self._slip(None) == self._slip()  # 預設值
        base = 0.0005 + 0.5 / math.sqrt(self.VOL)
        assert self._slip(None) == pytest.approx(min(base, SLIPPAGE_MAX_PCT))

    def test_monotonic_increasing_in_order_size(self):
        slips = [self._slip(n) for n in [1_000, 10_000, 50_000, 200_000]]
        assert all(b > a for a, b in zip(slips, slips[1:], strict=False))

    def test_exact_formula(self):
        expected = 0.0005 + 0.5 / math.sqrt(self.VOL) + PARTICIPATION_IMPACT_COEFF * math.sqrt(50_000 / self.VOL)
        assert self._slip(50_000) == pytest.approx(min(expected, SLIPPAGE_MAX_PCT))

    def test_participation_capped_at_full_volume(self):
        """order > volume 時 participation 比率封頂 1.0。"""
        assert self._slip(self.VOL * 5) == self._slip(self.VOL)

    def test_capped_at_max_pct(self):
        slip = compute_dynamic_slippage(100, 100.0, 100.0, 100.0, order_shares=100)  # 低量股
        assert slip <= SLIPPAGE_MAX_PCT

    def test_sell_multiplier_applies_after_participation(self):
        buy = self._slip(50_000)
        sell = compute_dynamic_slippage(self.VOL, 100.0, 100.0, 100.0, side="sell", order_shares=50_000)
        assert sell == pytest.approx(min(buy * SELL_SLIPPAGE_MULTIPLIER, SLIPPAGE_MAX_PCT))

    def test_zero_volume_skips_participation(self):
        assert compute_dynamic_slippage(0, 100.0, 100.0, 100.0, order_shares=10_000) == 0.0005


# ---------------------------------------------------------------------------
# 4. 守恆不變量
# ---------------------------------------------------------------------------


class TestConservation:
    @pytest.mark.parametrize(
        ("entry", "exit_p", "shares"),
        [(100.0, 110.0, 1000), (50.0, 45.0, 1300), (600.0, 615.0, 95), (10.0, 10.0, 5)],
    )
    def test_pnl_equals_proceeds_minus_buy_cost(self, entry, exit_p, shares):
        """pnl == proceeds − buy_cost（三個函數同一成本源）。"""
        buy = simulate_buy(entry, shares, 0.0005)
        sell = simulate_sell(entry, exit_p, shares, buy_slippage=0.0005, sell_slippage=0.0005)
        pnl, _ = compute_position_pnl(entry, exit_p, shares)
        assert pnl == pytest.approx(sell.proceeds - buy.buy_cost, abs=0.01)

    @pytest.mark.parametrize(
        ("capital", "price"),
        [(200_000, 100.0), (100_000, 99.9), (10_000, 1.0), (50_000, 885.0), (1_000_000, 12.8), (25.0, 1.0)],
    )
    def test_compute_shares_always_affordable(self, capital, price):
        """任意 capital/price 下，compute_shares 的結果以真實成本必買得起。"""
        shares = compute_shares(capital, price)
        if shares > 0:
            fill = simulate_buy(price, shares, 0.0005)
            assert fill.buy_cost <= capital + 1e-9

    def test_min_fee_makes_dust_position_expensive(self):
        """5 股 × 18.3 元的灰塵部位：零股 min fee 1 元 ≈ 成本 1.1%（量化現實）。"""
        c, _, s = trade_cost_amounts(18.3, 5, 0.0005, side="buy")
        assert c == MIN_COMMISSION_ODD  # 91.5 × 0.001425 = 0.13 → 綁定 1 元
        assert (c + s) / (18.3 * 5) > 0.01


# ---------------------------------------------------------------------------
# 5. BacktestEngine 旗標
# ---------------------------------------------------------------------------


class TestBacktestEngineFlags:
    def _engine(self, **cfg_kwargs):
        from src.backtest.engine import BacktestConfig, BacktestEngine

        class _Dummy:
            pass

        eng = BacktestEngine.__new__(BacktestEngine)
        eng.config = BacktestConfig(**cfg_kwargs)
        return eng

    def test_flags_default_off(self):
        from src.backtest.engine import BacktestConfig

        cfg = BacktestConfig()
        assert cfg.min_commission is False
        assert cfg.participation_impact is False

    def test_trade_costs_flag_off_is_proportional(self):
        """旗標關：純比例式（與 2026-07 前 bit-for-bit）。"""
        eng = self._engine()
        c, t, odd = eng._trade_costs(10.0, 999, side="sell")  # 小額全零股也不套 min fee
        assert c == pytest.approx(10.0 * 999 * eng.config.commission_rate)
        assert t == pytest.approx(10.0 * 999 * eng.config.tax_rate)
        assert odd == 0.0

    def test_trade_costs_flag_on_min_fee_and_premium(self):
        eng = self._engine(min_commission=True)
        c, t, odd = eng._trade_costs(10.0, 999, side="sell")  # notional 9,990 全零股
        assert c == pytest.approx(10.0 * 999 * eng.config.commission_rate)  # 14.24 > 1 → 比例
        c2, _, odd2 = eng._trade_costs(10.0, 1000, side="buy")  # 整張 10,000 → min 20
        assert c2 == MIN_COMMISSION_LOT
        assert odd2 == 0.0
        assert odd == pytest.approx(10.0 * 999 * ODD_LOT_SLIPPAGE_PREMIUM)

    def test_trade_costs_flag_on_uses_config_rates(self):
        eng = self._engine(min_commission=True)
        eng.config.commission_rate = 0.005  # 覆蓋費率
        c, _, _ = eng._trade_costs(100.0, 2000, side="buy")
        assert c == pytest.approx(100.0 * 2000 * 0.005)

    def test_slippage_participation_flag_off_unchanged(self):
        eng = self._engine(dynamic_slippage=True)
        s_no = eng._get_slippage(1_000_000, 100.0, 100.0, 100.0)
        s_with = eng._get_slippage(1_000_000, 100.0, 100.0, 100.0, order_shares=50_000)
        assert s_no == s_with  # 旗標關 → order_shares 無作用

    def test_slippage_participation_flag_on(self):
        eng = self._engine(dynamic_slippage=True, participation_impact=True)
        s_no = eng._get_slippage(1_000_000, 100.0, 100.0, 100.0)
        s_with = eng._get_slippage(1_000_000, 100.0, 100.0, 100.0, order_shares=50_000)
        assert s_with == pytest.approx(
            min(s_no + PARTICIPATION_IMPACT_COEFF * math.sqrt(50_000 / 1_000_000), eng.config.slippage_max)
        )

    def test_slippage_participation_works_with_static_slippage(self):
        """participation 與 dynamic_slippage 獨立：靜態滑價下也生效。"""
        eng = self._engine(participation_impact=True)
        s = eng._get_slippage(1_000_000, order_shares=50_000)
        assert s == pytest.approx(eng.config.slippage + PARTICIPATION_IMPACT_COEFF * math.sqrt(0.05))
