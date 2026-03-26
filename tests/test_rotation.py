"""tests/test_rotation.py — 輪動組合部位控制系統測試。

涵蓋：
- rotation.py 純函數測試（Rotation 邏輯 + PnL + 交易日）
- manager.py DB 整合測試（RotationManager CRUD + 回測）
"""

from __future__ import annotations

from datetime import date

from src.constants import COMMISSION_RATE, SLIPPAGE_RATE, TAX_RATE
from src.data.schema import RotationPortfolio, RotationPosition  # noqa: F401 — 確保 ORM 註冊
from src.portfolio.rotation import (
    compute_planned_exit_date,
    compute_position_pnl,
    compute_rotation_actions,
    compute_shares,
    count_trading_days_held,
    get_trading_dates_from_prices,
)

# ---------------------------------------------------------------------------
# 測試用 fixtures
# ---------------------------------------------------------------------------

# 2025-01-06 (Mon) ~ 2025-01-17 (Fri) 共 10 個交易日
TRADING_CAL = [
    date(2025, 1, 6),
    date(2025, 1, 7),
    date(2025, 1, 8),
    date(2025, 1, 9),
    date(2025, 1, 10),
    date(2025, 1, 13),
    date(2025, 1, 14),
    date(2025, 1, 15),
    date(2025, 1, 16),
    date(2025, 1, 17),
]


def _make_rankings(stocks: list[tuple[str, float]], start_rank: int = 1) -> list[dict]:
    """建立排名清單。stocks = [(stock_id, close), ...]"""
    return [
        {
            "stock_id": sid,
            "stock_name": f"Stock {sid}",
            "rank": start_rank + i,
            "score": 0.9 - i * 0.05,
            "close": close,
            "stop_loss": close * 0.9,
        }
        for i, (sid, close) in enumerate(stocks)
    ]


def _make_position(
    stock_id: str,
    entry_date: date,
    entry_price: float = 100.0,
    shares: int = 1000,
    allocated_capital: float = 100000.0,
    entry_rank: int = 1,
) -> dict:
    return {
        "stock_id": stock_id,
        "entry_date": entry_date,
        "entry_price": entry_price,
        "shares": shares,
        "allocated_capital": allocated_capital,
        "entry_rank": entry_rank,
    }


# ===========================================================================
# 交易日工具函數
# ===========================================================================


class TestGetTradingDates:
    def test_filter_range(self):
        result = get_trading_dates_from_prices(TRADING_CAL, date(2025, 1, 8), date(2025, 1, 14))
        assert result == [date(2025, 1, 8), date(2025, 1, 9), date(2025, 1, 10), date(2025, 1, 13), date(2025, 1, 14)]

    def test_empty_range(self):
        result = get_trading_dates_from_prices(TRADING_CAL, date(2025, 2, 1), date(2025, 2, 5))
        assert result == []


class TestCountTradingDaysHeld:
    def test_same_day(self):
        assert count_trading_days_held(date(2025, 1, 6), date(2025, 1, 6), TRADING_CAL) == 0

    def test_next_day(self):
        assert count_trading_days_held(date(2025, 1, 6), date(2025, 1, 7), TRADING_CAL) == 1

    def test_across_weekend(self):
        # entry=Fri 1/10, today=Mon 1/13 → 1 trading day
        assert count_trading_days_held(date(2025, 1, 10), date(2025, 1, 13), TRADING_CAL) == 1

    def test_three_days(self):
        # entry=1/6, today=1/9 → 3 trading days (7,8,9)
        assert count_trading_days_held(date(2025, 1, 6), date(2025, 1, 9), TRADING_CAL) == 3


class TestComputePlannedExitDate:
    def test_basic(self):
        result = compute_planned_exit_date(date(2025, 1, 6), 3, TRADING_CAL)
        assert result == date(2025, 1, 9)  # 6 + 3 trading days = 9

    def test_across_weekend(self):
        result = compute_planned_exit_date(date(2025, 1, 8), 3, TRADING_CAL)
        assert result == date(2025, 1, 13)  # 8,9,10,13 → 13

    def test_fallback_when_calendar_short(self):
        short_cal = [date(2025, 1, 6), date(2025, 1, 7)]
        result = compute_planned_exit_date(date(2025, 1, 6), 10, short_cal)
        assert result > date(2025, 1, 7)


# ===========================================================================
# 損益計算
# ===========================================================================


class TestComputePositionPnl:
    def test_profitable_trade(self):
        pnl, return_pct = compute_position_pnl(100.0, 110.0, 1000)
        buy_cost = 100 * 1000 * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
        sell_proc = 110 * 1000 * (1 - COMMISSION_RATE - TAX_RATE - SLIPPAGE_RATE)
        expected_pnl = sell_proc - buy_cost
        assert abs(pnl - round(expected_pnl, 2)) < 0.01
        assert return_pct > 0

    def test_losing_trade(self):
        pnl, return_pct = compute_position_pnl(100.0, 90.0, 1000)
        assert pnl < 0
        assert return_pct < 0

    def test_zero_shares(self):
        pnl, return_pct = compute_position_pnl(100.0, 110.0, 0)
        assert pnl == 0
        assert return_pct == 0


class TestComputeShares:
    def test_basic(self):
        shares = compute_shares(100000, 100.0)
        effective_price = 100 * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
        expected = int(100000 / effective_price)
        assert shares == expected

    def test_zero_price(self):
        assert compute_shares(100000, 0) == 0

    def test_insufficient_capital(self):
        assert compute_shares(50, 100.0) == 0


# ===========================================================================
# 核心 Rotation 邏輯
# ===========================================================================


class TestColdStart:
    """冷啟動：無持倉時從排名買入 Top-N。"""

    def test_buy_top_n(self):
        rankings = _make_rankings([("2330", 600), ("2317", 150), ("2454", 80), ("3008", 200), ("6669", 300)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=True,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=1_000_000,
        )
        assert len(actions.to_buy) == 3
        assert [b["stock_id"] for b in actions.to_buy] == ["2330", "2317", "2454"]
        assert len(actions.to_sell) == 0
        assert len(actions.to_hold) == 0

    def test_fewer_candidates_than_slots(self):
        rankings = _make_rankings([("2330", 600), ("2317", 150)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=5,
            holding_days=3,
            allow_renewal=True,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=1_000_000,
        )
        assert len(actions.to_buy) == 2

    def test_no_rankings(self):
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=[],
            max_positions=5,
            holding_days=3,
            allow_renewal=True,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=1_000_000,
        )
        assert len(actions.to_buy) == 0


class TestHoldingPeriod:
    """持有期間邏輯。"""

    def test_not_expired_keep(self):
        """未到期不賣，即使排名下降。"""
        pos = _make_position("2330", entry_date=date(2025, 1, 6))
        # 2330 不在排名中了
        rankings = _make_rankings([("2317", 150), ("2454", 80)])
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=5,
            allow_renewal=True,
            today=date(2025, 1, 8),  # 2 days held < 5
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
        )
        assert len(actions.to_hold) == 1
        assert actions.to_hold[0]["stock_id"] == "2330"
        assert len(actions.to_sell) == 0

    def test_expired_not_in_topn_sell(self):
        """到期 + 排名外 → 賣出。"""
        pos = _make_position("2330", entry_date=date(2025, 1, 6))
        rankings = _make_rankings([("2317", 150), ("2454", 80)])
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=True,
            today=date(2025, 1, 9),  # 3 days held >= 3
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
            today_prices={"2330": 105.0},
        )
        assert len(actions.to_sell) == 1
        assert actions.to_sell[0]["stock_id"] == "2330"
        assert actions.to_sell[0]["reason"] == "holding_expired"

    def test_expired_in_topn_renew(self):
        """到期 + 仍在 Top-N + allow_renewal → 續持。"""
        pos = _make_position("2330", entry_date=date(2025, 1, 6), entry_price=600)
        rankings = _make_rankings([("2330", 600), ("2317", 150)])
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=True,
            today=date(2025, 1, 9),
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
        )
        assert len(actions.renewed) == 1
        assert actions.renewed[0]["stock_id"] == "2330"
        assert len(actions.to_sell) == 0

    def test_expired_no_renewal_sell(self):
        """到期 + allow_renewal=False → 一律賣出。"""
        pos = _make_position("2330", entry_date=date(2025, 1, 6), entry_price=600)
        rankings = _make_rankings([("2330", 600), ("2317", 150)])
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 9),
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
            today_prices={"2330": 600.0},
        )
        assert len(actions.to_sell) == 1
        assert actions.to_sell[0]["reason"] == "holding_expired"
        assert len(actions.renewed) == 0


class TestStopLoss:
    """止損邏輯。"""

    def test_stop_loss_overrides_holding(self):
        """止損觸發 → 不論持有天數立即賣出。"""
        pos = _make_position("2330", entry_date=date(2025, 1, 6), entry_price=100)
        rankings = _make_rankings([("2330", 100)])  # stop_loss = 90
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=10,  # 遠未到期
            allow_renewal=True,
            today=date(2025, 1, 7),  # 只持有 1 天
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
            stop_losses={"2330": 90.0},
            today_prices={"2330": 85.0},  # 跌破止損
        )
        assert len(actions.to_sell) == 1
        assert actions.to_sell[0]["reason"] == "stop_loss"
        assert actions.to_sell[0]["exit_price"] == 85.0

    def test_no_stop_loss_when_above(self):
        """價格在止損之上 → 不觸發。"""
        pos = _make_position("2330", entry_date=date(2025, 1, 6), entry_price=100)
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=_make_rankings([("2330", 100)]),
            max_positions=3,
            holding_days=10,
            allow_renewal=True,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
            stop_losses={"2330": 90.0},
            today_prices={"2330": 95.0},  # 仍在止損之上
        )
        assert len(actions.to_sell) == 0
        assert len(actions.to_hold) == 1


class TestRotationReplacement:
    """換股邏輯。"""

    def test_sell_expired_buy_new(self):
        """到期賣出後，空位被新排名填補。"""
        positions = [
            _make_position("2330", entry_date=date(2025, 1, 6), entry_price=600),
            _make_position("2317", entry_date=date(2025, 1, 6), entry_price=150),
        ]
        rankings = _make_rankings([("2454", 80), ("3008", 200), ("6669", 300)])
        actions = compute_rotation_actions(
            current_positions=positions,
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=True,
            today=date(2025, 1, 9),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            today_prices={"2330": 610.0, "2317": 145.0},
        )
        assert len(actions.to_sell) == 2
        assert len(actions.to_buy) == 3  # 3 slots all free after selling 2

    def test_sold_stock_not_rebuy_same_day(self):
        """今日賣出的股票不在同日重買。"""
        pos = _make_position("2330", entry_date=date(2025, 1, 6), entry_price=600)
        # 2330 到期賣出，但 2330 仍在排名中 (allow_renewal=False)
        rankings = _make_rankings([("2330", 600), ("2317", 150)])
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 9),
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
            today_prices={"2330": 600.0},
        )
        # 2330 被賣出
        assert any(s["stock_id"] == "2330" for s in actions.to_sell)
        # 2330 不應同日重買
        buy_ids = [b["stock_id"] for b in actions.to_buy]
        assert "2330" not in buy_ids
        # 2317 應被買入
        assert "2317" in buy_ids


class TestRenewedPositionExitDate:
    """續持後 planned_exit_date 正確更新。"""

    def test_new_exit_date(self):
        pos = _make_position("2330", entry_date=date(2025, 1, 6), entry_price=600)
        rankings = _make_rankings([("2330", 600)])
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=True,
            today=date(2025, 1, 9),  # 到期日
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
        )
        assert len(actions.renewed) == 1
        new_exit = actions.renewed[0]["new_planned_exit_date"]
        # 從 1/9 起算 3 個交易日: 10, 13, 14 → 1/14
        assert new_exit == date(2025, 1, 14)


class TestCapitalTracking:
    """資金配置邏輯。"""

    def test_equal_weight_allocation(self):
        """每支分配 1/N 資金。"""
        rankings = _make_rankings([("2330", 100), ("2317", 100), ("2454", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=True,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
        )
        assert len(actions.to_buy) == 3
        # 每支分配 300000/3 = 100000
        for buy in actions.to_buy:
            assert abs(buy["allocated_capital"] - 100_000) < 1

    def test_zero_price_skipped(self):
        rankings = [{"stock_id": "0000", "rank": 1, "score": 0.9, "close": 0, "stock_name": "Zero"}]
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=True,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
        )
        assert len(actions.to_buy) == 0


# ===========================================================================
# DB 整合測試
# ===========================================================================


class TestResolveRankings:
    """排名解析（需 DB）。"""

    def test_single_mode(self, db_session):
        from src.data.schema import DiscoveryRecord
        from src.portfolio.manager import resolve_rankings

        db_session.add(
            DiscoveryRecord(
                scan_date=date(2025, 1, 6),
                mode="momentum",
                rank=1,
                stock_id="2330",
                stock_name="台積電",
                close=600.0,
                composite_score=0.85,
                stop_loss=570.0,
            )
        )
        db_session.add(
            DiscoveryRecord(
                scan_date=date(2025, 1, 6),
                mode="momentum",
                rank=2,
                stock_id="2317",
                stock_name="鴻海",
                close=150.0,
                composite_score=0.75,
                stop_loss=140.0,
            )
        )
        db_session.flush()

        result = resolve_rankings("momentum", date(2025, 1, 6), db_session)
        assert len(result) == 2
        assert result[0]["stock_id"] == "2330"
        assert result[0]["rank"] == 1
        assert result[1]["stock_id"] == "2317"

    def test_all_mode_avg_score(self, db_session):
        from src.data.schema import DiscoveryRecord
        from src.portfolio.manager import resolve_rankings

        # 2330 出現在 momentum(0.9) + swing(0.8) → avg = 0.85
        # 2317 只在 momentum(0.7) → avg = 0.7
        db_session.add(
            DiscoveryRecord(
                scan_date=date(2025, 1, 6),
                mode="momentum",
                rank=1,
                stock_id="2330",
                stock_name="台積電",
                close=600.0,
                composite_score=0.9,
            )
        )
        db_session.add(
            DiscoveryRecord(
                scan_date=date(2025, 1, 6),
                mode="swing",
                rank=1,
                stock_id="2330",
                stock_name="台積電",
                close=600.0,
                composite_score=0.8,
            )
        )
        db_session.add(
            DiscoveryRecord(
                scan_date=date(2025, 1, 6),
                mode="momentum",
                rank=2,
                stock_id="2317",
                stock_name="鴻海",
                close=150.0,
                composite_score=0.7,
            )
        )
        db_session.flush()

        result = resolve_rankings("all", date(2025, 1, 6), db_session)
        assert len(result) == 2
        assert result[0]["stock_id"] == "2330"
        assert abs(result[0]["score"] - 0.85) < 0.01
        assert result[1]["stock_id"] == "2317"
        assert result[0]["rank"] == 1
        assert result[1]["rank"] == 2


class TestRotationPortfolioCRUD:
    """RotationPortfolio ORM CRUD。"""

    def test_create_and_read(self, db_session):

        p = RotationPortfolio(
            name="test_mom5_3d",
            mode="momentum",
            max_positions=5,
            holding_days=3,
            allow_renewal=True,
            initial_capital=1_000_000,
            current_capital=1_000_000,
            current_cash=1_000_000,
            status="active",
        )
        db_session.add(p)
        db_session.flush()

        from sqlalchemy import select

        loaded = db_session.execute(
            select(RotationPortfolio).where(RotationPortfolio.name == "test_mom5_3d")
        ).scalar_one()
        assert loaded.mode == "momentum"
        assert loaded.max_positions == 5
        assert loaded.holding_days == 3
        assert loaded.allow_renewal is True

    def test_position_create_and_read(self, db_session):

        p = RotationPortfolio(
            name="test_pos",
            mode="swing",
            max_positions=3,
            holding_days=5,
            allow_renewal=False,
            initial_capital=500_000,
            current_capital=500_000,
            current_cash=500_000,
            status="active",
        )
        db_session.add(p)
        db_session.flush()

        pos = RotationPosition(
            portfolio_id=p.id,
            stock_id="2330",
            stock_name="台積電",
            entry_date=date(2025, 1, 6),
            entry_price=600.0,
            entry_rank=1,
            entry_score=0.85,
            holding_days_count=0,
            planned_exit_date=date(2025, 1, 9),
            shares=800,
            allocated_capital=166_666,
            status="open",
        )
        db_session.add(pos)
        db_session.flush()

        from sqlalchemy import select

        loaded = db_session.execute(select(RotationPosition).where(RotationPosition.stock_id == "2330")).scalar_one()
        assert loaded.portfolio_id == p.id
        assert loaded.entry_price == 600.0
        assert loaded.status == "open"
