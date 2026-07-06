"""src/portfolio/dividends.py 純函數測試（A3-1）。

含守恆 property tests：
  - factor 恆落 (0, 1.5) 或 None
  - 停損調整後與 prev_close 的相對比例不變（cash-only 情境）
  - 入帳現金 = shares × cash_dividend
  - 無股利恆等（no-op）
"""

from __future__ import annotations

from datetime import date

import pytest

from src.data.schema import Dividend
from src.portfolio.dividends import (
    DividendEvent,
    adjust_stop_loss_for_dividend,
    dividend_adjustment_factor,
    dividend_cash_for_position,
    load_dividend_events,
)


class TestDividendAdjustmentFactor:
    def test_cash_only_matches_strategy_formula(self):
        # 2330 實例：prev_close=1435, cash=5 → factor = 1430/1435
        f = dividend_adjustment_factor(1435.0, 5.0, 0.0)
        assert f == pytest.approx(1430.0 / 1435.0)

    def test_stock_dividend_divides(self):
        f = dividend_adjustment_factor(100.0, 0.0, 1.0)  # 配股 1 元 → /1.1
        assert f == pytest.approx(1 / 1.1)

    def test_combined(self):
        f = dividend_adjustment_factor(100.0, 2.0, 0.5)
        assert f == pytest.approx((98.0 / 100.0) / 1.05)

    @pytest.mark.parametrize(
        "prev_close,cash,stock",
        [(None, 5.0, 0.0), (0.0, 5.0, 0.0), (-10.0, 5.0, 0.0), (5.0, 5.0, 0.0), (5.0, 10.0, 0.0)],
    )
    def test_invalid_inputs_return_none(self, prev_close, cash, stock):
        assert dividend_adjustment_factor(prev_close, cash, stock) is None

    def test_factor_bounds_property(self):
        """factor 非 None 時恆落 (0, 1.5)。"""
        cases = [(p, c, s) for p in (10.0, 100.0, 1500.0) for c in (0.0, 0.5, 5.0, 50.0) for s in (0.0, 0.5, 2.0)]
        for prev_close, cash, stock in cases:
            f = dividend_adjustment_factor(prev_close, cash, stock)
            if f is not None:
                assert 0.0 < f < 1.5


class TestAdjustStopLoss:
    def test_relative_distance_preserved(self):
        """cash-only：調整後 stop/prev_close_ex（除息參考價）比例 = 原 stop/prev_close。"""
        prev_close, cash, stop = 100.0, 4.0, 92.0
        adjusted = adjust_stop_loss_for_dividend(stop, prev_close, cash, 0.0)
        ex_ref = prev_close - cash  # 除息參考價
        assert adjusted / ex_ref == pytest.approx(stop / prev_close)

    def test_none_stop_passthrough(self):
        assert adjust_stop_loss_for_dividend(None, 100.0, 5.0, 0.0) is None

    def test_invalid_factor_returns_unchanged(self):
        assert adjust_stop_loss_for_dividend(92.0, None, 5.0, 0.0) == 92.0
        assert adjust_stop_loss_for_dividend(92.0, 100.0, 150.0, 0.0) == 92.0

    def test_no_dividend_identity(self):
        # cash=stock=0 → factor=1 → 恆等
        assert adjust_stop_loss_for_dividend(92.0, 100.0, 0.0, 0.0) == pytest.approx(92.0)

    def test_gap_no_longer_triggers_stop(self):
        """除息跳空情境：原 stop 95、prev_close 100、cash 6 → 除息開在 ~94。
        不調整會誤觸（94 < 95）；調整後 stop = 95×0.94 = 89.3 → 不誤觸。"""
        stop = adjust_stop_loss_for_dividend(95.0, 100.0, 6.0, 0.0)
        ex_open = 94.0
        assert stop < ex_open


class TestDividendCash:
    def test_amount(self):
        assert dividend_cash_for_position(1000, 5.0) == 5000.0

    def test_zero_cases(self):
        assert dividend_cash_for_position(0, 5.0) == 0.0
        assert dividend_cash_for_position(1000, 0.0) == 0.0
        assert dividend_cash_for_position(-100, 5.0) == 0.0


class TestLoadDividendEvents:
    def test_groups_by_ex_date_and_skips_empty(self, db_session):
        db_session.add_all(
            [
                Dividend(stock_id="2330", date=date(2026, 6, 18), year="115Q1", cash_dividend=5.0, stock_dividend=0.0),
                Dividend(stock_id="2317", date=date(2026, 6, 18), year="114", cash_dividend=5.4, stock_dividend=0.0),
                Dividend(stock_id="1101", date=date(2026, 7, 2), year="114", cash_dividend=1.0, stock_dividend=0.5),
                Dividend(stock_id="9999", date=date(2026, 6, 20), year="114", cash_dividend=0.0, stock_dividend=0.0),
            ]
        )
        db_session.flush()

        events = load_dividend_events(db_session, ["2330", "2317", "1101", "9999"], date(2026, 6, 1), date(2026, 7, 31))

        assert set(events.keys()) == {date(2026, 6, 18), date(2026, 7, 2)}
        assert set(events[date(2026, 6, 18)].keys()) == {"2330", "2317"}
        ev = events[date(2026, 7, 2)]["1101"]
        assert ev == DividendEvent(stock_id="1101", ex_date=date(2026, 7, 2), cash_dividend=1.0, stock_dividend=0.5)
        # 空 payout 略過
        assert not any("9999" in d for d in events.values())

    def test_filters_by_range_and_ids(self, db_session):
        db_session.add(
            Dividend(stock_id="2330", date=date(2025, 12, 17), year="114Q2", cash_dividend=5.0, stock_dividend=0.0)
        )
        db_session.flush()
        assert load_dividend_events(db_session, ["2330"], date(2026, 1, 1), date(2026, 12, 31)) == {}
        assert load_dividend_events(db_session, ["2317"], date(2025, 1, 1), date(2026, 12, 31)) == {}
        assert load_dividend_events(db_session, [], date(2025, 1, 1), date(2026, 12, 31)) == {}
