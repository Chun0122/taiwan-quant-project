"""tests/test_pending_order_schema.py — RotationPendingOrder schema（A2-1）。

涵蓋：ORM round-trip、唯一約束（防重複決策）、init_db 自動建表、status 預設值。
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError

from src.data.schema import Base, RotationPendingOrder


def _make_order(**overrides) -> RotationPendingOrder:
    defaults = dict(
        portfolio_name="mom5_10d",
        decision_date=date(2026, 7, 3),
        side="buy",
        stock_id="2330",
        stock_name="台積電",
        shares=1000,
        ref_price=1000.0,
        entry_rank=1,
        entry_score=0.9,
        stop_loss=950.0,
    )
    defaults.update(overrides)
    return RotationPendingOrder(**defaults)


class TestRotationPendingOrderSchema:
    def test_round_trip(self, db_session):
        db_session.add(_make_order())
        db_session.flush()

        row = db_session.query(RotationPendingOrder).one()
        assert row.status == "pending"  # 預設值
        assert row.exec_date is None  # 成交後才回填
        assert row.side == "buy"
        assert row.shares == 1000
        assert row.ref_price == 1000.0
        assert "pending" in repr(row)

    def test_unique_constraint_blocks_duplicate_decision(self, db_session):
        """同 portfolio 同決策日同 side 同標的重複寫入應被唯一約束擋下。"""
        db_session.add(_make_order())
        db_session.flush()
        db_session.add(_make_order(shares=2000))
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_same_stock_different_side_allowed(self, db_session):
        """同標的同日 buy 與 sell 各一筆合法（約束含 side）。"""
        db_session.add(_make_order(side="buy"))
        db_session.add(_make_order(side="sell", reason="stop_loss"))
        db_session.flush()
        assert db_session.query(RotationPendingOrder).count() == 2

    def test_sell_order_fields(self, db_session):
        db_session.add(
            _make_order(side="sell", reason="holding_expired", entry_rank=None, entry_score=None, stop_loss=None)
        )
        db_session.flush()
        row = db_session.query(RotationPendingOrder).one()
        assert row.reason == "holding_expired"
        assert row.entry_score is None

    def test_create_all_builds_table(self, tmp_path):
        """新表由 create_all（init_db）自動建立，不需 MIGRATIONS。"""
        engine = create_engine(f"sqlite:///{tmp_path / 'new.db'}")
        Base.metadata.create_all(engine)
        insp = inspect(engine)
        assert "rotation_pending_order" in insp.get_table_names()
        cols = {c["name"] for c in insp.get_columns("rotation_pending_order")}
        assert {
            "portfolio_name",
            "decision_date",
            "exec_date",
            "side",
            "stock_id",
            "shares",
            "ref_price",
            "reason",
            "entry_rank",
            "entry_score",
            "score_breakdown_json",
            "stop_loss",
            "status",
        } <= cols
        uqs = {u["name"] for u in insp.get_unique_constraints("rotation_pending_order")}
        assert "uq_pending_order" in uqs
