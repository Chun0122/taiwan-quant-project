"""2026-05-15 sprint P3 — StrategyDecayLog 落庫測試。

對應 morning-routine Step 15 從純 stdout 升級為 DB 持久化，
供 5/29 audit 訊號穩定性時序對比。
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest
from sqlalchemy import select

from src.data.schema import StrategyDecayLog

# ====================================================================== #
# P3-A: schema 欄位/索引
# ====================================================================== #


class TestSchemaShape:
    def test_log_can_be_created(self, db_session):
        log = StrategyDecayLog(
            scan_date=date(2026, 5, 15),
            mode="momentum",
            recent_win_rate=0.55,
            recent_avg_return=0.025,
            recent_count=120,
            is_decaying=False,
            warning=None,
        )
        db_session.add(log)
        db_session.flush()

        loaded = db_session.execute(select(StrategyDecayLog)).scalar_one()
        assert loaded.scan_date == date(2026, 5, 15)
        assert loaded.mode == "momentum"
        assert loaded.recent_win_rate == pytest.approx(0.55)
        assert loaded.recent_count == 120
        assert loaded.is_decaying is False

    def test_unique_constraint_scan_date_mode(self, db_session):
        """同日同 mode 不能重複插入。"""
        db_session.add(
            StrategyDecayLog(
                scan_date=date(2026, 5, 15),
                mode="momentum",
                recent_count=10,
                is_decaying=False,
            )
        )
        db_session.flush()

        db_session.add(
            StrategyDecayLog(
                scan_date=date(2026, 5, 15),
                mode="momentum",
                recent_count=20,
                is_decaying=False,
            )
        )
        with pytest.raises(Exception):  # IntegrityError 或 OperationalError
            db_session.flush()

    def test_nullable_fields_accept_none(self, db_session):
        """無樣本資料時 recent_win_rate / recent_avg_return / warning 可為 None。"""
        log = StrategyDecayLog(
            scan_date=date(2026, 5, 15),
            mode="value",
            recent_count=0,
            is_decaying=False,
        )
        db_session.add(log)
        db_session.flush()

        loaded = db_session.execute(select(StrategyDecayLog)).scalar_one()
        assert loaded.recent_win_rate is None
        assert loaded.recent_avg_return is None
        assert loaded.warning is None


# ====================================================================== #
# P3-B: _check_strategy_decay() 持久化
# ====================================================================== #


class TestCheckStrategyDecayPersistence:
    """測試 morning_cmd._check_strategy_decay 寫入 StrategyDecayLog。"""

    def _fake_check_results(self) -> list[dict]:
        return [
            {
                "mode": "momentum",
                "recent_win_rate": 0.58,
                "recent_avg_return": 0.035,
                "recent_count": 120,
                "is_decaying": False,
                "warning": None,
            },
            {
                "mode": "swing",
                "recent_win_rate": 0.30,
                "recent_avg_return": -0.025,
                "recent_count": 50,
                "is_decaying": True,
                "warning": "勝率 30% 跌破 50%",
            },
            {
                "mode": "value",
                "recent_win_rate": None,
                "recent_avg_return": None,
                "recent_count": 0,
                "is_decaying": False,
                "warning": None,
            },
        ]

    def test_persists_one_row_per_mode(self, db_session, capsys):
        from src.cli.morning_cmd import _check_strategy_decay

        with patch("src.discovery.performance.check_all_modes_decay", return_value=self._fake_check_results()):
            _check_strategy_decay(scan_date=date(2026, 5, 15))

        rows = db_session.execute(select(StrategyDecayLog).order_by(StrategyDecayLog.mode)).scalars().all()
        assert len(rows) == 3
        modes = {r.mode for r in rows}
        assert modes == {"momentum", "swing", "value"}

        # 衰減模式 is_decaying=True + warning 不為空
        swing_row = next(r for r in rows if r.mode == "swing")
        assert swing_row.is_decaying is True
        assert swing_row.warning is not None
        assert swing_row.recent_win_rate == pytest.approx(0.30)

        # 零樣本 mode 全 None
        value_row = next(r for r in rows if r.mode == "value")
        assert value_row.recent_count == 0
        assert value_row.recent_win_rate is None

    def test_idempotent_same_day_updates(self, db_session):
        """同 scan_date 重複呼叫採 update path，不違反 UniqueConstraint。"""
        from src.cli.morning_cmd import _check_strategy_decay

        with patch("src.discovery.performance.check_all_modes_decay", return_value=self._fake_check_results()):
            _check_strategy_decay(scan_date=date(2026, 5, 15))

        # 第二次：mom 勝率變了
        updated = [
            {
                "mode": "momentum",
                "recent_win_rate": 0.62,  # 改了
                "recent_avg_return": 0.040,
                "recent_count": 125,
                "is_decaying": False,
                "warning": None,
            },
            {
                "mode": "swing",
                "recent_win_rate": 0.30,
                "recent_avg_return": -0.025,
                "recent_count": 50,
                "is_decaying": True,
                "warning": "勝率 30% 跌破 50%",
            },
            {
                "mode": "value",
                "recent_win_rate": None,
                "recent_avg_return": None,
                "recent_count": 0,
                "is_decaying": False,
                "warning": None,
            },
        ]
        with patch("src.discovery.performance.check_all_modes_decay", return_value=updated):
            _check_strategy_decay(scan_date=date(2026, 5, 15))

        rows = db_session.execute(select(StrategyDecayLog)).scalars().all()
        assert len(rows) == 3  # 仍 3 筆，沒新增
        mom = next(r for r in rows if r.mode == "momentum")
        assert mom.recent_win_rate == pytest.approx(0.62)  # 已 update
        assert mom.recent_count == 125

    def test_different_days_create_new_rows(self, db_session):
        """不同 scan_date 各自一筆，形成時序。"""
        from src.cli.morning_cmd import _check_strategy_decay

        with patch("src.discovery.performance.check_all_modes_decay", return_value=self._fake_check_results()):
            _check_strategy_decay(scan_date=date(2026, 5, 15))
            _check_strategy_decay(scan_date=date(2026, 5, 16))
            _check_strategy_decay(scan_date=date(2026, 5, 17))

        rows = db_session.execute(select(StrategyDecayLog)).scalars().all()
        assert len(rows) == 9  # 3 mode × 3 days

    def test_persistence_failure_does_not_block(self, db_session, capsys):
        """持久化失敗（DB 錯）只 warning，不拋例外。"""
        from src.cli.morning_cmd import _check_strategy_decay

        with patch("src.discovery.performance.check_all_modes_decay", return_value=self._fake_check_results()):
            # 模擬 commit 拋例外
            with patch("src.data.database.get_session", side_effect=RuntimeError("db locked")):
                _check_strategy_decay(scan_date=date(2026, 5, 15))  # 不拋

        captured = capsys.readouterr()
        assert "持久化失敗" in captured.out or "持久化失敗" in captured.err

    def test_scan_date_none_defaults_to_today(self, db_session):
        """scan_date=None 用 date.today()。"""
        from src.cli.morning_cmd import _check_strategy_decay

        with patch("src.discovery.performance.check_all_modes_decay", return_value=self._fake_check_results()):
            _check_strategy_decay(scan_date=None)

        rows = db_session.execute(select(StrategyDecayLog)).scalars().all()
        assert len(rows) == 3
        assert all(r.scan_date == date.today() for r in rows)
