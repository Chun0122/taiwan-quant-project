"""tests/test_rotation.py — 輪動組合部位控制系統測試。

涵蓋：
- rotation.py 純函數測試（Rotation 邏輯 + PnL + 交易日）
- manager.py DB 整合測試（RotationManager CRUD + 回測）
- save_rotation_backtest() DB 寫入測試
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.constants import COMMISSION_RATE, REGIME_FALLBACK_DEFAULT, SLIPPAGE_RATE, TAX_RATE
from src.data.schema import (  # noqa: F401 — 確保 ORM 註冊
    RotationBacktestSummary,
    RotationBacktestTrade,
    RotationPortfolio,
    RotationPosition,
)
from src.portfolio.rotation import (
    apply_liquidity_limit,
    compute_correlation_matrix,
    compute_dynamic_slippage,
    compute_planned_exit_date,
    compute_portfolio_drawdown,
    compute_portfolio_heat,
    compute_position_pnl,
    compute_rotation_actions,
    compute_shares,
    compute_single_trade_risk,
    compute_trade_costs,
    compute_vol_inverse_weights,
    count_trading_days_held,
    detect_limit_price,
    find_high_correlation_pairs,
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


# ===========================================================================
# B1: 產業集中度限制
# ===========================================================================


class TestSectorConcentration:
    """測試 compute_rotation_actions() 的 sector_map + max_sector_pct 產業集中度限制。"""

    def test_no_sector_map_no_limit(self):
        """未提供 sector_map 時不限制。"""
        rankings = _make_rankings([("A", 100), ("B", 100), ("C", 100), ("D", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=4,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=400_000,
        )
        assert len(actions.to_buy) == 4

    def test_sector_limit_blocks_excess(self):
        """同產業 4 支候選，max_positions=4，max_sector_pct=0.30 → 同產業最多 1 支。"""
        # 4 支全為半導體
        rankings = _make_rankings([("A", 100), ("B", 100), ("C", 100), ("D", 100)])
        sector_map = {"A": "半導體", "B": "半導體", "C": "半導體", "D": "半導體"}
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=4,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=400_000,
            sector_map=sector_map,
            max_sector_pct=0.30,
        )
        # max_sector_count = int(4 * 0.30) = 1，所以同產業只能買 1 支
        semi_bought = [b for b in actions.to_buy if sector_map.get(b["stock_id"]) == "半導體"]
        assert len(semi_bought) == 1

    def test_mixed_sectors_respects_limit(self):
        """混合產業，每個產業 2 支，max 5 positions，30% → 每產業最多 1 支。"""
        rankings = _make_rankings(
            [
                ("A1", 100),
                ("A2", 100),
                ("B1", 100),
                ("B2", 100),
                ("C1", 100),
            ]
        )
        sector_map = {"A1": "半導體", "A2": "半導體", "B1": "金融", "B2": "金融", "C1": "電子"}
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=5,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
            sector_map=sector_map,
            max_sector_pct=0.30,
        )
        # 統計每個產業的買入數
        sector_buys: dict[str, int] = {}
        for b in actions.to_buy:
            sec = sector_map.get(b["stock_id"], "")
            sector_buys[sec] = sector_buys.get(sec, 0) + 1
        for sec, count in sector_buys.items():
            assert count <= 1  # max(1, int(5 * 0.30)) = 1

    def test_existing_positions_counted(self):
        """已持有 1 支半導體，新買入時同產業計數從 1 開始。"""
        pos = _make_position("A1", date(2025, 1, 6))
        rankings = _make_rankings([("A1", 100), ("A2", 100), ("B1", 100)])
        sector_map = {"A1": "半導體", "A2": "半導體", "B1": "金融"}
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=5,
            allow_renewal=False,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            sector_map=sector_map,
            max_sector_pct=0.50,  # max(1, int(3*0.5))=1
        )
        # A1 已持有（to_hold），A2 同產業被跳過，B1 可買
        semi_bought = [b for b in actions.to_buy if sector_map.get(b["stock_id"]) == "半導體"]
        assert len(semi_bought) == 0
        fin_bought = [b for b in actions.to_buy if sector_map.get(b["stock_id"]) == "金融"]
        assert len(fin_bought) == 1


# ===========================================================================
# B2: 持倉相關性監控
# ===========================================================================


class TestCorrelationMonitor:
    """測試 compute_correlation_matrix() 和 find_high_correlation_pairs()。"""

    def test_empty_returns_empty(self):
        """不到 2 支股票 → 空 DataFrame。"""
        result = compute_correlation_matrix({"A": pd.Series([1, 2, 3])})
        assert result.empty

    def test_perfect_correlation(self):
        """完全正相關的兩支股票 → correlation ≈ 1.0。"""
        import numpy as np

        prices_a = pd.Series(np.arange(100, 200, dtype=float))
        prices_b = pd.Series(np.arange(50, 150, dtype=float))
        corr = compute_correlation_matrix({"A": prices_a, "B": prices_b}, window=60)
        assert not corr.empty
        assert corr.loc["A", "B"] > 0.99

    def test_find_high_pairs(self):
        """高相關配對偵測。"""
        import numpy as np

        np.random.seed(42)
        base = np.cumsum(np.random.randn(100))
        prices = {
            "A": pd.Series(100 + base),
            "B": pd.Series(50 + base * 1.1),  # 高度相關
            "C": pd.Series(80 + np.cumsum(np.random.randn(100))),  # 獨立
        }
        corr = compute_correlation_matrix(prices, window=60)
        pairs = find_high_correlation_pairs(corr, threshold=0.7)
        # A-B 應該高度相關
        ab_pairs = [(a, b) for a, b, _ in pairs if {a, b} == {"A", "B"}]
        assert len(ab_pairs) == 1

    def test_threshold_filters_correctly(self):
        """低門檻能抓到更多配對，高門檻過濾嚴格。"""
        import numpy as np

        np.random.seed(42)
        base = np.cumsum(np.random.randn(100))
        prices = {
            "A": pd.Series(100 + base),
            "B": pd.Series(50 + base + np.random.randn(100) * 2),
            "C": pd.Series(80 + np.cumsum(np.random.randn(100))),
        }
        corr = compute_correlation_matrix(prices, window=60)
        pairs_low = find_high_correlation_pairs(corr, threshold=0.3)
        pairs_high = find_high_correlation_pairs(corr, threshold=0.9)
        assert len(pairs_low) >= len(pairs_high)


# ===========================================================================
# B3: 波動率反比部位大小
# ===========================================================================


class TestVolInverseWeights:
    """測試 compute_vol_inverse_weights()。"""

    def test_equal_vol_equal_weight(self):
        """相同波動率 → 等權重。"""
        weights = compute_vol_inverse_weights({"A": 0.2, "B": 0.2, "C": 0.2})
        assert len(weights) == 3
        for w in weights.values():
            assert abs(w - 1 / 3) < 0.001

    def test_higher_vol_lower_weight(self):
        """波動率高的分配較少。"""
        weights = compute_vol_inverse_weights({"LOW": 0.1, "HIGH": 0.5})
        assert weights["LOW"] > weights["HIGH"]

    def test_weights_sum_to_one(self):
        """權重合為 1.0。"""
        weights = compute_vol_inverse_weights({"A": 0.15, "B": 0.25, "C": 0.35})
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_zero_vol_excluded(self):
        """波動率為 0 的被排除。"""
        weights = compute_vol_inverse_weights({"A": 0.2, "B": 0.0, "C": 0.3})
        assert "B" not in weights
        assert len(weights) == 2

    def test_empty_input(self):
        """空輸入 → 空結果。"""
        assert compute_vol_inverse_weights({}) == {}


class TestVolWeightsIntegration:
    """測試 vol_weights 整合進 compute_rotation_actions()。"""

    def test_vol_weights_none_equal_allocation(self):
        """vol_weights=None → 等權分配（向後相容）。"""
        rankings = _make_rankings([("A", 100), ("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            vol_weights=None,
        )
        assert len(actions.to_buy) == 2
        # 等權：兩者 allocated_capital 相近
        cap_a = actions.to_buy[0]["allocated_capital"]
        cap_b = actions.to_buy[1]["allocated_capital"]
        assert abs(cap_a - cap_b) < cap_a * 0.1  # 差異 < 10%

    def test_high_vol_gets_less_capital(self):
        """高波動股票分配較少資金。"""
        # LOW: weight=0.75, HIGH: weight=0.25（vol inverse of 0.1 vs 0.3）
        vol_weights = compute_vol_inverse_weights({"A": 0.1, "B": 0.3})
        rankings = _make_rankings([("A", 100), ("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            vol_weights=vol_weights,
        )
        assert len(actions.to_buy) == 2
        cap_a = actions.to_buy[0]["allocated_capital"]  # A = low vol → more
        cap_b = actions.to_buy[1]["allocated_capital"]  # B = high vol → less
        assert cap_a > cap_b

    def test_vol_weights_stock_not_in_dict(self):
        """候選不在 vol_weights 中 → 使用等權 per_position_capital。"""
        vol_weights = {"A": 1.0}  # 只有 A，B 不在
        rankings = _make_rankings([("A", 100), ("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            vol_weights=vol_weights,
        )
        assert len(actions.to_buy) == 2

    def test_equal_vol_similar_allocation(self):
        """等波動率 → 等權分配。"""
        vol_weights = compute_vol_inverse_weights({"A": 0.2, "B": 0.2})
        rankings = _make_rankings([("A", 100), ("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            vol_weights=vol_weights,
        )
        assert len(actions.to_buy) == 2
        cap_a = actions.to_buy[0]["allocated_capital"]
        cap_b = actions.to_buy[1]["allocated_capital"]
        assert abs(cap_a - cap_b) < cap_a * 0.1


# ===========================================================================
# B4: Drawdown Guard
# ===========================================================================


class TestDrawdownGuard:
    """測試 compute_rotation_actions() 的 drawdown_pct 參數。"""

    def test_no_drawdown_normal_buying(self):
        """drawdown_pct=None → 正常買入。"""
        rankings = _make_rankings([("A", 100), ("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
        )
        assert len(actions.to_buy) == 2

    def test_moderate_drawdown_reduces_capital(self):
        """drawdown 12% → scale = max(0, 1-12/15) = 0.2，持倉量更小。"""
        rankings = _make_rankings([("A", 100)])
        actions_normal = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            drawdown_pct=None,
        )
        actions_dd = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            drawdown_pct=12.0,
        )
        if actions_normal.to_buy and actions_dd.to_buy:
            assert actions_dd.to_buy[0]["shares"] < actions_normal.to_buy[0]["shares"]

    def test_severe_drawdown_stops_buying(self):
        """drawdown 16% >= threshold 15% → scale = 0 → 停止新開倉。"""
        rankings = _make_rankings([("A", 100), ("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            drawdown_pct=16.0,
        )
        assert len(actions.to_buy) == 0

    def test_drawdown_below_threshold_allows_buying(self):
        """drawdown 5% → scale = 1-5/15 ≈ 0.667 → 可買入。"""
        rankings = _make_rankings([("A", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            drawdown_pct=5.0,
        )
        assert len(actions.to_buy) == 1

    def test_drawdown_exactly_at_threshold_stops(self):
        """drawdown 15% = threshold → scale = 0 → 不買。"""
        rankings = _make_rankings([("A", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            drawdown_pct=15.0,
        )
        assert len(actions.to_buy) == 0

    def test_drawdown_half_is_half_scale(self):
        """drawdown 7.5% → scale = 1 - 7.5/15 = 0.5（連續化驗證）。"""
        rankings = _make_rankings([("A", 100)])
        actions_normal = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            drawdown_pct=None,
        )
        actions_half = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            drawdown_pct=7.5,
        )
        if actions_normal.to_buy and actions_half.to_buy:
            # 7.5% dd → scale 0.5，持倉量約為正常的一半
            normal_shares = actions_normal.to_buy[0]["shares"]
            half_shares = actions_half.to_buy[0]["shares"]
            assert half_shares <= normal_shares * 0.6  # 允許取整誤差

    def test_drawdown_20_above_threshold_still_zero(self):
        """drawdown 20% > threshold → scale = 0（clamp）。"""
        rankings = _make_rankings([("A", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=2,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=200_000,
            drawdown_pct=20.0,
        )
        assert len(actions.to_buy) == 0


class TestPortfolioDrawdown:
    """測試 compute_portfolio_drawdown()。"""

    def test_at_peak(self):
        """淨值在高點 → 回撤 0。"""
        assert compute_portfolio_drawdown([100, 110, 120]) == 0.0

    def test_simple_drawdown(self):
        """peak=120, current=100 → 16.67%。"""
        dd = compute_portfolio_drawdown([100, 120, 100])
        assert abs(dd - 16.67) < 0.01

    def test_empty_history(self):
        assert compute_portfolio_drawdown([]) == 0.0

    def test_single_point(self):
        assert compute_portfolio_drawdown([100]) == 0.0


# ===========================================================================
# Portfolio Heat（組合風險預算）
# ===========================================================================


class TestPortfolioHeat:
    """測試 compute_portfolio_heat() 純函數。"""

    def test_no_positions_zero_heat(self):
        """無持倉 → heat = 0。"""
        assert compute_portfolio_heat([], {}, {}, 1_000_000) == 0.0

    def test_with_stop_loss(self):
        """有停損 → heat = (price - sl) × shares / capital。"""
        positions = [_make_position("A", date(2025, 1, 6), entry_price=100, shares=1000)]
        heat = compute_portfolio_heat(
            positions,
            stop_losses={"A": 95.0},
            today_prices={"A": 100.0},
            total_capital=1_000_000,
        )
        # risk = (100 - 95) × 1000 = 5000；heat = 5000 / 1_000_000 = 0.005
        assert abs(heat - 0.005) < 0.001

    def test_without_stop_loss_uses_cap(self):
        """無停損 → 以 allocated_capital × risk_cap 估算。"""
        positions = [_make_position("A", date(2025, 1, 6), entry_price=100, shares=1000, allocated_capital=100_000)]
        heat = compute_portfolio_heat(
            positions,
            stop_losses={},
            today_prices={"A": 100.0},
            total_capital=1_000_000,
            per_position_risk_cap=0.03,
        )
        # risk = 100_000 × 0.03 = 3000；heat = 3000 / 1_000_000 = 0.003
        assert abs(heat - 0.003) < 0.001

    def test_multiple_positions(self):
        """多筆持倉累加風險。"""
        positions = [
            _make_position("A", date(2025, 1, 6), entry_price=100, shares=1000),
            _make_position("B", date(2025, 1, 6), entry_price=200, shares=500),
        ]
        heat = compute_portfolio_heat(
            positions,
            stop_losses={"A": 95.0, "B": 190.0},
            today_prices={"A": 100.0, "B": 200.0},
            total_capital=1_000_000,
        )
        # A: (100-95)×1000=5000, B: (200-190)×500=5000 → total 10000 / 1M = 0.01
        assert abs(heat - 0.01) < 0.001

    def test_zero_capital(self):
        """total_capital=0 → heat=0 防護。"""
        positions = [_make_position("A", date(2025, 1, 6))]
        assert compute_portfolio_heat(positions, {"A": 90.0}, {"A": 100.0}, 0) == 0.0

    def test_price_below_stop_no_negative_risk(self):
        """現價已低於停損 → 風險為 0（max(0, ...)）。"""
        positions = [_make_position("A", date(2025, 1, 6), entry_price=100, shares=1000)]
        heat = compute_portfolio_heat(
            positions,
            stop_losses={"A": 95.0},
            today_prices={"A": 90.0},  # 已跌破停損
            total_capital=1_000_000,
        )
        assert heat == 0.0


class TestSingleTradeRisk:
    """測試 compute_single_trade_risk() 純函數。"""

    def test_with_stop_loss(self):
        risk = compute_single_trade_risk(100.0, 95.0, 1000, 1_000_000)
        assert abs(risk - 0.005) < 0.001

    def test_without_stop_loss(self):
        risk = compute_single_trade_risk(100.0, None, 1000, 1_000_000, allocated_capital=100_000)
        assert abs(risk - 0.003) < 0.001  # 100_000 × 0.03 / 1_000_000

    def test_zero_capital(self):
        assert compute_single_trade_risk(100.0, 95.0, 1000, 0) == 0.0


class TestPortfolioHeatIntegration:
    """測試 Portfolio Heat 整合進 compute_rotation_actions()。"""

    def test_heat_blocks_when_exceeded(self):
        """heat 已超過上限 → 拒絕新開倉。"""
        # 已持有一筆高風險持倉，heat 接近上限
        pos = _make_position("A", date(2025, 1, 6), entry_price=100, shares=10000, allocated_capital=1_000_000)
        rankings = _make_rankings([("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=5,
            holding_days=10,
            allow_renewal=False,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
            stop_losses={"A": 85.0},  # (100-85)×10000=150000, heat=150000/1M=0.15 > 0.12
            today_prices={"A": 100.0, "B": 100.0},
            total_capital=1_000_000,
            max_heat=0.12,
        )
        # heat 已 0.15 > 0.12，新倉應被拒
        assert len(actions.to_buy) == 0

    def test_heat_allows_when_budget_remaining(self):
        """heat 低於上限 → 正常開倉。"""
        pos = _make_position("A", date(2025, 1, 6), entry_price=100, shares=1000, allocated_capital=100_000)
        rankings = _make_rankings([("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=5,
            holding_days=10,
            allow_renewal=False,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
            stop_losses={"A": 95.0},  # (100-95)×1000=5000, heat=0.005 << 0.12
            today_prices={"A": 100.0, "B": 100.0},
            total_capital=1_000_000,
            max_heat=0.12,
        )
        assert len(actions.to_buy) == 1

    def test_heat_disabled_when_no_total_capital(self):
        """total_capital=None → heat 檢查停用，正常開倉。"""
        rankings = _make_rankings([("A", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            total_capital=None,  # 停用
        )
        assert len(actions.to_buy) == 1


# ===========================================================================
# Correlation Budget（相關性決策化）
# ===========================================================================


class TestCorrelationBudget:
    """測試 Correlation Budget 整合進 compute_rotation_actions()。"""

    def test_no_corr_matrix_no_effect(self):
        """未提供 corr_matrix → 不影響開倉。"""
        rankings = _make_rankings([("A", 100), ("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            corr_matrix=None,
        )
        assert len(actions.to_buy) == 2

    def test_low_correlation_no_penalty(self):
        """候選與持倉低相關 → 正常開倉金額。"""
        pos = _make_position("A", date(2025, 1, 6), entry_price=100)
        rankings = _make_rankings([("B", 100)])
        # 低相關矩陣
        corr = pd.DataFrame({"A": {"A": 1.0, "B": 0.3}, "B": {"A": 0.3, "B": 1.0}})
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=10,
            allow_renewal=False,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            corr_matrix=corr,
            corr_threshold=0.7,
        )
        assert len(actions.to_buy) == 1
        # 低相關不 penalty → 正常金額
        assert actions.to_buy[0]["allocated_capital"] > 50_000

    def test_high_correlation_reduces_position(self):
        """候選與持倉高相關 → 部位減半。"""
        pos = _make_position("A", date(2025, 1, 6), entry_price=100)
        rankings = _make_rankings([("B", 100)])
        # 高相關矩陣
        corr = pd.DataFrame({"A": {"A": 1.0, "B": 0.9}, "B": {"A": 0.9, "B": 1.0}})
        actions_no_corr = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=10,
            allow_renewal=False,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            corr_matrix=None,
        )
        actions_with_corr = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=10,
            allow_renewal=False,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            corr_matrix=corr,
            corr_threshold=0.7,
            corr_penalty=0.5,
        )
        assert len(actions_with_corr.to_buy) == 1
        # 高相關 penalty → 部位應更小
        if actions_no_corr.to_buy:
            assert actions_with_corr.to_buy[0]["shares"] < actions_no_corr.to_buy[0]["shares"]

    def test_only_one_penalty_applied(self):
        """候選與多檔持倉都高相關 → 只觸發一次 penalty（break）。"""
        positions = [
            _make_position("A", date(2025, 1, 6), entry_price=100),
            _make_position("B", date(2025, 1, 6), entry_price=100),
        ]
        rankings = _make_rankings([("C", 100)])
        # C 與 A、B 都高相關
        corr = pd.DataFrame(
            {
                "A": {"A": 1.0, "B": 0.5, "C": 0.9},
                "B": {"A": 0.5, "B": 1.0, "C": 0.85},
                "C": {"A": 0.9, "B": 0.85, "C": 1.0},
            }
        )
        actions = compute_rotation_actions(
            current_positions=positions,
            new_rankings=rankings,
            max_positions=5,
            holding_days=10,
            allow_renewal=False,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            corr_matrix=corr,
            corr_threshold=0.7,
            corr_penalty=0.5,
        )
        assert len(actions.to_buy) == 1
        # penalty 只 ×0.5 一次，不是 ×0.5×0.5

    def test_empty_corr_matrix_no_effect(self):
        """空矩陣 → 不影響。"""
        rankings = _make_rankings([("A", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            corr_matrix=pd.DataFrame(),
        )
        assert len(actions.to_buy) == 1


class TestCorrelationBudgetRegimeAdaptive:
    """測試相關性預算 Regime 自適應（crisis/bear 收緊門檻）。"""

    def _run_with_regime(self, regime, corr_val=0.55):
        """建立 corr=corr_val 的持倉+候選，以指定 regime 執行。"""
        pos = _make_position("A", date(2025, 1, 6), entry_price=100)
        rankings = _make_rankings([("B", 100)])
        corr = pd.DataFrame(
            {"A": {"A": 1.0, "B": corr_val}, "B": {"A": corr_val, "B": 1.0}},
        )
        return compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=10,
            allow_renewal=False,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            corr_matrix=corr,
            regime=regime,
            crisis_block_new=False,  # 隔離測試相關性，不受 crisis 硬阻擋影響
        )

    def test_bull_uses_default_threshold(self):
        """bull + corr=0.55 < 0.7 → 不 penalty。"""
        actions = self._run_with_regime("bull", 0.55)
        assert len(actions.to_buy) == 1
        assert actions.to_buy[0]["allocated_capital"] > 50_000

    def test_crisis_lowers_threshold(self):
        """crisis + corr=0.55 >= 0.5 → penalty 觸發，部位更小。"""
        actions_bull = self._run_with_regime("bull", 0.55)
        actions_crisis = self._run_with_regime("crisis", 0.55)
        assert len(actions_crisis.to_buy) == 1
        # crisis penalty = 0.3，bull 無 penalty
        assert actions_crisis.to_buy[0]["allocated_capital"] < actions_bull.to_buy[0]["allocated_capital"]

    def test_bear_lowers_threshold(self):
        """bear + corr=0.65 >= 0.6 → penalty 觸發。"""
        actions_bull = self._run_with_regime("bull", 0.65)
        actions_bear = self._run_with_regime("bear", 0.65)
        # bull: 0.65 < 0.7 → 無 penalty；bear: 0.65 >= 0.6 → penalty
        assert len(actions_bear.to_buy) == 1
        assert actions_bear.to_buy[0]["allocated_capital"] < actions_bull.to_buy[0]["allocated_capital"]


# ===========================================================================
# Crisis 硬阻擋
# ===========================================================================


class TestCrisisBlock:
    """測試 regime='crisis' 硬阻擋新開倉。"""

    def test_regime_none_no_block(self):
        """regime=None → 不影響。"""
        rankings = _make_rankings([("A", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            regime=None,
        )
        assert len(actions.to_buy) == 1

    def test_regime_bull_no_block(self):
        """regime='bull' → 不影響。"""
        rankings = _make_rankings([("A", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            regime="bull",
        )
        assert len(actions.to_buy) == 1

    def test_crisis_blocks_new_buys(self):
        """regime='crisis' + crisis_block_new=True → to_buy 為空。"""
        rankings = _make_rankings([("A", 100), ("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            regime="crisis",
            crisis_block_new=True,
        )
        assert len(actions.to_buy) == 0

    def test_crisis_block_disabled(self):
        """regime='crisis' + crisis_block_new=False → 正常開倉。"""
        rankings = _make_rankings([("A", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            regime="crisis",
            crisis_block_new=False,
        )
        assert len(actions.to_buy) == 1


class TestCrisisForceClose:
    """測試 crisis_force_close 強制平倉既有持倉。"""

    def test_crisis_force_close_sells_all(self):
        """crisis + force_close=True → 既有持倉全部賣出。"""
        pos_a = _make_position("A", date(2025, 1, 6), entry_price=100)
        pos_b = _make_position("B", date(2025, 1, 6), entry_price=200)
        rankings = _make_rankings([("C", 100)])
        actions = compute_rotation_actions(
            current_positions=[pos_a, pos_b],
            new_rankings=rankings,
            max_positions=3,
            holding_days=10,
            allow_renewal=False,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=100_000,
            regime="crisis",
            crisis_block_new=True,
            crisis_force_close=True,
        )
        assert len(actions.to_sell) == 2
        assert all(s["reason"] == "crisis_exit" for s in actions.to_sell)
        assert len(actions.to_hold) == 0
        assert len(actions.to_buy) == 0  # crisis_block_new 也阻擋

    def test_crisis_force_close_false_holds_positions(self):
        """crisis + force_close=False → 持倉不受影響（僅擋新開倉）。"""
        pos = _make_position("A", date(2025, 1, 6), entry_price=100)
        rankings = _make_rankings([("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=10,
            allow_renewal=False,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=100_000,
            regime="crisis",
            crisis_block_new=True,
            crisis_force_close=False,
        )
        assert len(actions.to_sell) == 0
        assert len(actions.to_hold) == 1

    def test_non_crisis_force_close_ignored(self):
        """非 crisis + force_close=True → 不強制平倉。"""
        pos = _make_position("A", date(2025, 1, 6), entry_price=100)
        rankings = _make_rankings([("B", 100)])
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=10,
            allow_renewal=False,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=100_000,
            regime="bull",
            crisis_force_close=True,
        )
        assert len(actions.to_sell) == 0
        assert len(actions.to_hold) == 1

    def test_crisis_force_close_no_positions(self):
        """crisis + force_close=True 但無持倉 → 無事發生。"""
        rankings = _make_rankings([("A", 100)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            regime="crisis",
            crisis_block_new=True,
            crisis_force_close=True,
        )
        assert len(actions.to_sell) == 0
        assert len(actions.to_buy) == 0


# ===========================================================================
# save_rotation_backtest — DB 寫入
# ===========================================================================


def _make_backtest_result(trades=None):
    """建立模擬 RotationBacktestResult。"""
    from src.portfolio.manager import RotationBacktestResult

    if trades is None:
        trades = [
            {
                "stock_id": "2330",
                "entry_date": date(2025, 1, 6),
                "entry_price": 600.0,
                "exit_date": date(2025, 1, 10),
                "exit_price": 620.0,
                "shares": 1000,
                "pnl": 18000.0,
                "return_pct": 3.0,
                "exit_reason": "holding_expired",
                "entry_rank": 1,
                "entry_score": 0.85,
            },
            {
                "stock_id": "2317",
                "entry_date": date(2025, 1, 7),
                "entry_price": 100.0,
                "exit_date": date(2025, 1, 10),
                "exit_price": 95.0,
                "shares": 5000,
                "pnl": -26500.0,
                "return_pct": -5.3,
                "exit_reason": "stop_loss",
                "entry_rank": 2,
                "entry_score": 0.78,
            },
        ]
    return RotationBacktestResult(
        equity_curve=[{"date": date(2025, 1, 6), "equity": 1_000_000}],
        trades=trades,
        metrics={
            "total_return": 0.05,
            "annual_return": 0.12,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
            "win_rate": 0.5,
            "total_trades": 2,
            "avg_return_per_trade": -0.015,
            "avg_win": 0.03,
            "avg_loss": -0.053,
            "final_capital": 1_050_000,
            "trading_days": 5,
        },
        config={
            "portfolio_name": "test_bt",
            "mode": "momentum",
            "max_positions": 5,
            "holding_days": 3,
            "capital": 1_000_000,
            "allow_renewal": True,
            "start_date": date(2025, 1, 6),
            "end_date": date(2025, 1, 10),
        },
    )


class TestSaveRotationBacktest:
    """save_rotation_backtest() DB 寫入測試。

    合併為單一測試避免 SingletonThreadPool session 跨測試干擾
    （save 內部的 ``with get_session()`` 會關閉 fixture session）。
    """

    def test_save_summary_trades_and_empty(self, db_session):
        """寫入摘要 + 交易明細 + 空交易，驗證欄位正確。"""
        from sqlalchemy import select

        from src.data.pipeline import save_rotation_backtest

        # ── Case 1: 含 2 筆交易 ──
        result = _make_backtest_result()
        bt_id = save_rotation_backtest(result)
        assert isinstance(bt_id, int)
        assert bt_id > 0

        # 驗證摘要
        saved = db_session.execute(
            select(RotationBacktestSummary).where(RotationBacktestSummary.id == bt_id)
        ).scalar_one()
        assert saved.portfolio_name == "test_bt"
        assert saved.mode == "momentum"
        assert saved.max_positions == 5
        assert saved.total_trades == 2
        assert saved.total_return == 0.05

        # 驗證交易明細
        trades = (
            db_session.execute(
                select(RotationBacktestTrade)
                .where(RotationBacktestTrade.backtest_id == bt_id)
                .order_by(RotationBacktestTrade.entry_date)
            )
            .scalars()
            .all()
        )
        assert len(trades) == 2
        assert trades[0].stock_id == "2330"
        assert trades[0].entry_rank == 1
        assert trades[0].entry_score == 0.85
        assert trades[0].exit_reason == "holding_expired"
        assert trades[1].stock_id == "2317"
        assert trades[1].entry_rank == 2
        assert trades[1].pnl == -26500.0

        # ── Case 2: 空交易 ──
        result_empty = _make_backtest_result(trades=[])
        bt_id2 = save_rotation_backtest(result_empty)
        assert bt_id2 > bt_id

        empty_trades = (
            db_session.execute(select(RotationBacktestTrade).where(RotationBacktestTrade.backtest_id == bt_id2))
            .scalars()
            .all()
        )
        assert len(empty_trades) == 0


# ===========================================================================
# 止損價持久化 — to_buy 攜帶 stop_loss + 持倉止損不隨 rankings 浮動
# ===========================================================================


class TestStopLossPersistence:
    """止損價應從持倉記錄取，不隨 discover rankings 每日浮動。"""

    def test_buy_action_includes_stop_loss(self):
        """新開倉的 to_buy dict 攜帶 rankings 中的 stop_loss。"""
        rankings = _make_rankings([("2330", 100)])  # stop_loss = 100 * 0.9 = 90
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=5,
            allow_renewal=True,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
        )
        assert len(actions.to_buy) == 1
        assert actions.to_buy[0]["stop_loss"] == 90.0

    def test_position_stop_loss_used_over_rankings(self):
        """持倉有自己的 stop_loss 時，不受 rankings 的新止損價影響。

        模擬：進場價 100，進場止損 90。股價跌到 92，
        新 rankings 重算止損為 92*0.9=82.8 → 不應採用。
        用持倉的 stop_loss=90 判斷，92 > 90 → 不觸發。
        """
        pos = _make_position("2330", entry_date=date(2025, 1, 6), entry_price=100)
        pos["stop_loss"] = 90.0  # 進場時鎖定

        # Rankings 重算後的 stop_loss 為 82.8（不應被使用）
        rankings = _make_rankings([("2330", 92)])  # stop_loss = 92*0.9 = 82.8

        # 以持倉的 stop_loss=90 檢查
        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=10,
            allow_renewal=True,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
            stop_losses={"2330": 90.0},  # 來自持倉記錄
            today_prices={"2330": 92.0},
        )
        assert len(actions.to_sell) == 0
        assert len(actions.to_hold) == 1

    def test_position_dropped_from_rankings_still_has_stop_loss(self):
        """掉出推薦的持倉仍保有止損保護。

        持倉 stop_loss=90，股價跌到 85 → 應觸發止損。
        即使 rankings 裡已無此股票。
        """
        pos = _make_position("2330", entry_date=date(2025, 1, 6), entry_price=100)
        pos["stop_loss"] = 90.0

        # Rankings 不含 2330（已掉出推薦）
        rankings = _make_rankings([("2454", 80)])

        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=10,
            allow_renewal=True,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
            stop_losses={"2330": 90.0},  # 來自持倉記錄
            today_prices={"2330": 85.0},  # 跌破止損
        )
        assert len(actions.to_sell) == 1
        assert actions.to_sell[0]["reason"] == "stop_loss"
        assert actions.to_sell[0]["exit_price"] == 85.0

    def test_position_dropped_from_rankings_no_stop_loss_no_crash(self):
        """掉出推薦且無止損記錄的持倉（舊資料相容）不會 crash。"""
        pos = _make_position("2330", entry_date=date(2025, 1, 6), entry_price=100)
        # 不設 stop_loss（模擬舊版資料）

        rankings = _make_rankings([("2454", 80)])

        actions = compute_rotation_actions(
            current_positions=[pos],
            new_rankings=rankings,
            max_positions=3,
            holding_days=10,
            allow_renewal=True,
            today=date(2025, 1, 7),
            trading_calendar=TRADING_CAL,
            current_cash=500_000,
            stop_losses={},  # 無止損資訊
            today_prices={"2330": 85.0},
        )
        # 無止損 → 不觸發，繼續持有
        assert len(actions.to_sell) == 0
        assert len(actions.to_hold) == 1


# ===========================================================================
# Regime Fallback 安全預設值
# ===========================================================================


class TestRegimeFallbackDefault:
    """REGIME_FALLBACK_DEFAULT 常數正確性 + 'sideways' 不阻擋交易。"""

    def test_fallback_constant_is_sideways(self):
        """安全預設值為 sideways。"""
        assert REGIME_FALLBACK_DEFAULT == "sideways"

    def test_sideways_regime_allows_new_buys(self):
        """regime='sideways'（fallback 值）不阻擋新開倉。"""
        rankings = _make_rankings([("A", 100), ("B", 90)])
        actions = compute_rotation_actions(
            current_positions=[],
            new_rankings=rankings,
            max_positions=3,
            holding_days=3,
            allow_renewal=False,
            today=date(2025, 1, 6),
            trading_calendar=TRADING_CAL,
            current_cash=300_000,
            regime=REGIME_FALLBACK_DEFAULT,
        )
        assert len(actions.to_buy) == 2


# ===========================================================================
# P0: 動態滑價
# ===========================================================================


class TestDynamicSlippage:
    """測試 compute_dynamic_slippage() 三因子模型。"""

    def test_sell_higher_than_buy(self):
        """賣出滑價 >= 買入滑價（恐慌拋售加成）。"""
        buy = compute_dynamic_slippage(1_000_000, 105, 95, 100, side="buy")
        sell = compute_dynamic_slippage(1_000_000, 105, 95, 100, side="sell")
        assert sell >= buy
        assert sell <= 0.01  # SLIPPAGE_MAX_PCT

    def test_low_volume_higher_slippage(self):
        """低成交量 → 較高滑價（high=low 消除 spread proxy，純測 volume impact）。"""
        low = compute_dynamic_slippage(10_000, 100, 100, 100, max_pct=0.10)
        high = compute_dynamic_slippage(10_000_000, 100, 100, 100, max_pct=0.10)
        assert low > high

    def test_zero_volume_fallback(self):
        """volume=0 → fallback 到 base_slippage。"""
        slip = compute_dynamic_slippage(0, 105, 95, 100)
        assert slip == SLIPPAGE_RATE

    def test_negative_volume_fallback(self):
        """volume<0 → fallback 到 base_slippage。"""
        slip = compute_dynamic_slippage(-100, 105, 95, 100)
        assert slip == SLIPPAGE_RATE

    def test_max_cap(self):
        """極低量時滑價受 max_pct 限制。"""
        slip = compute_dynamic_slippage(1, 200, 50, 100, max_pct=0.01)
        assert slip <= 0.01

    def test_wide_spread_increases_slippage(self):
        """大價差（high - low 大）→ spread proxy 提高滑價（提高 max_pct 避免觸頂）。"""
        narrow = compute_dynamic_slippage(1_000_000, 101, 99, 100, max_pct=0.15)
        wide = compute_dynamic_slippage(1_000_000, 110, 90, 100, max_pct=0.15)
        assert wide > narrow


# ===========================================================================
# P0: 流動性約束
# ===========================================================================


class TestLiquidityLimit:
    """測試 apply_liquidity_limit()。"""

    def test_caps_shares(self):
        """超過 participation limit 時被裁切。"""
        assert apply_liquidity_limit(100_000, 500_000, 0.05) == 25_000

    def test_no_volume_passthrough(self):
        """volume=0 → 不約束，原樣回傳。"""
        assert apply_liquidity_limit(1000, 0, 0.05) == 1000

    def test_negative_volume_passthrough(self):
        """volume<0 → 不約束。"""
        assert apply_liquidity_limit(1000, -100, 0.05) == 1000

    def test_within_limit_no_change(self):
        """未超 limit 時不變。"""
        assert apply_liquidity_limit(100, 500_000, 0.05) == 100


# ===========================================================================
# P0: 漲跌停偵測
# ===========================================================================


class TestLimitPrice:
    """測試 detect_limit_price()。"""

    def test_limit_up(self):
        """開盤漲停。"""
        up, down = detect_limit_price(110.0, 100.0)
        assert up is True and down is False

    def test_limit_down(self):
        """開盤跌停。"""
        up, down = detect_limit_price(90.0, 100.0)
        assert up is False and down is True

    def test_normal_open(self):
        """正常開盤。"""
        up, down = detect_limit_price(101.0, 100.0)
        assert up is False and down is False

    def test_zero_prev_close(self):
        """prev_close=0 → 安全回傳 (False, False)。"""
        up, down = detect_limit_price(100.0, 0.0)
        assert up is False and down is False


# ===========================================================================
# P0: 交易成本分解
# ===========================================================================


class TestTradeCosts:
    """測試 compute_trade_costs()。"""

    def test_buy_no_tax(self):
        """買入不含交易稅。"""
        costs = compute_trade_costs(100.0, 1000, 0.001, "buy")
        assert costs.tax == 0.0
        assert costs.commission > 0
        assert costs.slippage_cost > 0
        assert costs.total == costs.commission + costs.slippage_cost

    def test_sell_has_tax(self):
        """賣出含交易稅。"""
        costs = compute_trade_costs(100.0, 1000, 0.001, "sell")
        assert costs.tax > 0
        assert costs.total == costs.commission + costs.tax + costs.slippage_cost

    def test_sell_tax_matches_rate(self):
        """賣出交易稅 = notional × TAX_RATE。"""
        costs = compute_trade_costs(100.0, 1000, 0.001, "sell")
        expected_tax = 100.0 * 1000 * TAX_RATE
        assert abs(costs.tax - round(expected_tax, 2)) < 0.01


# ===========================================================================
# P0: compute_position_pnl 新參數向後相容
# ===========================================================================


class TestPositionPnlSlippageParams:
    """測試 compute_position_pnl() 新增 buy_slippage / sell_slippage 參數。"""

    def test_default_same_as_before(self):
        """不帶新參數時行為不變。"""
        pnl, ret = compute_position_pnl(100, 110, 1000)
        # 驗算
        buy_cost = 100 * 1000 * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
        sell_proceeds = 110 * 1000 * (1 - COMMISSION_RATE - TAX_RATE - SLIPPAGE_RATE)
        expected_pnl = sell_proceeds - buy_cost
        assert abs(pnl - round(expected_pnl, 2)) < 0.01

    def test_higher_slippage_lowers_pnl(self):
        """較高滑價 → 較低 PnL。"""
        pnl_low, _ = compute_position_pnl(100, 110, 1000, buy_slippage=0.001, sell_slippage=0.001)
        pnl_high, _ = compute_position_pnl(100, 110, 1000, buy_slippage=0.005, sell_slippage=0.005)
        assert pnl_low > pnl_high


# ===========================================================================
# P0: compute_shares 新參數向後相容
# ===========================================================================


class TestComputeSharesEnhanced:
    """測試 compute_shares() 新增流動性約束參數。"""

    def test_default_same_as_before(self):
        """不帶新參數時行為不變。"""
        shares = compute_shares(100_000, 100)
        expected = int(100_000 / (100 * (1 + COMMISSION_RATE + SLIPPAGE_RATE)))
        assert shares == expected

    def test_with_slippage_override(self):
        """自訂滑價。"""
        shares = compute_shares(100_000, 100, slippage=0.01)
        expected = int(100_000 / (100 * (1 + COMMISSION_RATE + 0.01)))
        assert shares == expected

    def test_with_liquidity_cap(self):
        """流動性約束裁切。"""
        shares = compute_shares(10_000_000, 100, daily_volume=50_000, participation_limit=0.05)
        assert shares <= 2500  # 50000 * 0.05

    def test_no_volume_no_cap(self):
        """daily_volume=None 不約束。"""
        shares_uncapped = compute_shares(100_000, 100)
        shares_none = compute_shares(100_000, 100, daily_volume=None)
        assert shares_uncapped == shares_none


# ===========================================================================
# P0: TAIEX Benchmark DB 查詢
# ===========================================================================


class TestTaiexBenchmark:
    """測試 _get_taiex_prices() DB 查詢。"""

    def test_taiex_benchmark_with_costs(self, db_session):
        """TAIEX benchmark 含手續費 + 交易稅。"""
        from src.data.schema import DailyPrice as DP
        from src.portfolio.manager import _get_taiex_prices

        # 插入 TAIEX 資料
        for i, d in enumerate(TRADING_CAL[:5]):
            db_session.add(
                DP(
                    stock_id="TAIEX",
                    date=d,
                    open=20000 + i * 100,
                    high=20050 + i * 100,
                    low=19950 + i * 100,
                    close=20000 + i * 100,
                    volume=100_000_000,
                    turnover=0,
                )
            )
        db_session.commit()

        prices = _get_taiex_prices(db_session, TRADING_CAL[0], TRADING_CAL[4])
        assert len(prices) == 5
        assert TRADING_CAL[0] in prices
        assert prices[TRADING_CAL[0]] == 20000

    def test_empty_range(self, db_session):
        """無資料期間回傳空 dict。"""
        from src.portfolio.manager import _get_taiex_prices

        prices = _get_taiex_prices(db_session, date(2020, 1, 1), date(2020, 1, 31))
        assert prices == {}


# ===========================================================================
# P0: _get_ohlcv_on_date DB 查詢
# ===========================================================================


class TestGetOhlcvOnDate:
    """測試 _get_ohlcv_on_date()。"""

    def test_exact_date(self, db_session):
        """精確日期有資料。"""
        from src.data.schema import DailyPrice as DP
        from src.portfolio.manager import _get_ohlcv_on_date

        db_session.add(
            DP(
                stock_id="2330",
                date=TRADING_CAL[0],
                open=600,
                high=610,
                low=595,
                close=605,
                volume=30_000_000,
                turnover=0,
            )
        )
        db_session.commit()

        result = _get_ohlcv_on_date(db_session, ["2330"], TRADING_CAL[0])
        assert "2330" in result
        assert result["2330"]["close"] == 605
        assert result["2330"]["volume"] == 30_000_000

    def test_fallback_volume_zero(self, db_session):
        """Fallback 日期的 volume 歸零（安全措施）。"""
        from src.data.schema import DailyPrice as DP
        from src.portfolio.manager import _get_ohlcv_on_date

        # 只有前一天有資料
        db_session.add(
            DP(
                stock_id="2330",
                date=TRADING_CAL[0],
                open=600,
                high=610,
                low=595,
                close=605,
                volume=30_000_000,
                turnover=0,
            )
        )
        db_session.commit()

        # 查詢隔天（fallback）
        result = _get_ohlcv_on_date(db_session, ["2330"], TRADING_CAL[1])
        assert "2330" in result
        assert result["2330"]["close"] == 605
        assert result["2330"]["volume"] == 0  # fallback 資料 volume 歸零

    def test_empty_ids(self, db_session):
        """空 stock_ids → 空 dict。"""
        from src.portfolio.manager import _get_ohlcv_on_date

        assert _get_ohlcv_on_date(db_session, [], TRADING_CAL[0]) == {}
