"""P2/P3 功能測試。

測試項目：
- classify_event_type: 事件類型分類（法說會/財報/月營收/一般）
- _compute_revenue_scan: 純函數 YoY + 毛利率改善掃描
- Announcement event_type ORM 寫入與查詢
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.mops_fetcher import classify_event_type

# 確保 ORM 類別在 in_memory_engine.create_all() 前已被載入並註冊至 Base.metadata
from src.data.schema import Announcement, FinancialStatement, MonthlyRevenue  # noqa: F401

# ------------------------------------------------------------------ #
#  classify_event_type 事件類型分類（純函數）
# ------------------------------------------------------------------ #


class TestClassifyEventType:
    """事件類型關鍵字分類測試。"""

    # --- earnings_call ---

    @pytest.mark.parametrize(
        "subject",
        [
            "本公司訂於明日舉辦法說會",
            "2330 法人說明會邀請函",
            "本公司 Q3 業績說明會通知",
            "分析師說明會召開公告",
        ],
    )
    def test_earnings_call_keywords(self, subject: str) -> None:
        """包含法說會關鍵字 → earnings_call。"""
        assert classify_event_type(subject) == "earnings_call"

    # --- investor_day ---

    @pytest.mark.parametrize(
        "subject",
        [
            "本公司舉辦投資人日活動",
            "Investor Day 2025 公告",
            "investor day活動通知",
            "股東說明會召開通知",
            "本公司股東會議事錄",
        ],
    )
    def test_investor_day_keywords(self, subject: str) -> None:
        """包含投資人日關鍵字 → investor_day。"""
        assert classify_event_type(subject) == "investor_day"

    # --- filing ---

    @pytest.mark.parametrize(
        "subject",
        [
            "本公司 2024 年報已公告",
            "Q3 季報財務資料公告",
            "半年報揭露通知",
            "本公司財務報告已完成申報",
            "本公司財報公告",
            "本公司 EPS 揭露",
            "本公司每股盈餘 5.2 元",
            "本公司 Q1財報 公告",
            "本公司 Q4財報 完整揭露",
        ],
    )
    def test_filing_keywords(self, subject: str) -> None:
        """包含財報關鍵字 → filing。"""
        assert classify_event_type(subject) == "filing"

    # --- revenue ---

    @pytest.mark.parametrize(
        "subject",
        [
            "本公司 11 月月營收公告",
            "本公司合併營收揭露",
            "本公司自結營收",
            "11 月份營收報告",
        ],
    )
    def test_revenue_keywords(self, subject: str) -> None:
        """包含月營收關鍵字 → revenue。"""
        assert classify_event_type(subject) == "revenue"

    # --- general ---

    @pytest.mark.parametrize(
        "subject",
        [
            "本公司董事會決議庫藏股",
            "本公司與某公司簽訂合作備忘錄",
            "本公司子公司設立通知",
            "",
        ],
    )
    def test_general_fallback(self, subject: str) -> None:
        """不含特定事件關鍵字 → general。"""
        assert classify_event_type(subject) == "general"

    # --- 優先序 ---

    def test_earnings_call_beats_filing(self) -> None:
        """法說會優先於財報（earnings_call > filing）。"""
        subject = "本公司法說會 EPS 5.2 元"
        assert classify_event_type(subject) == "earnings_call"

    def test_investor_day_beats_revenue(self) -> None:
        """投資人日優先於月營收（investor_day > revenue）。"""
        subject = "股東說明會 — 本月月營收報告"
        assert classify_event_type(subject) == "investor_day"

    def test_filing_beats_revenue(self) -> None:
        """財報優先於月營收（filing > revenue）。"""
        subject = "Q3 季報 — 合併營收揭露"
        assert classify_event_type(subject) == "filing"


# ------------------------------------------------------------------ #
#  Announcement event_type ORM 寫入與查詢
# ------------------------------------------------------------------ #


class TestAnnouncementEventType:
    """Announcement 表 event_type 欄位 CRUD 測試。"""

    def _make_ann(self, stock_id, seq, subject, event_type="general"):
        return Announcement(
            stock_id=stock_id,
            date=date(2025, 11, 1),
            seq=seq,
            subject=subject,
            sentiment=0,
            event_type=event_type,
        )

    def test_write_and_read_event_type(self, db_session) -> None:
        """寫入 event_type 後可正確讀回。"""
        from sqlalchemy import select

        ann = self._make_ann("2330", "001", "法說會召開通知", "earnings_call")
        db_session.add(ann)
        db_session.flush()

        row = db_session.execute(select(Announcement).where(Announcement.stock_id == "2330")).scalars().first()
        assert row is not None
        assert row.event_type == "earnings_call"

    def test_default_event_type_is_general(self, db_session) -> None:
        """未指定 event_type 時，預設值為 general。"""
        from sqlalchemy import select

        ann = Announcement(
            stock_id="2317",
            date=date(2025, 11, 2),
            seq="002",
            subject="一般公告",
            sentiment=0,
        )
        db_session.add(ann)
        db_session.flush()

        row = db_session.execute(select(Announcement).where(Announcement.stock_id == "2317")).scalars().first()
        assert row is not None
        assert row.event_type == "general"

    def test_filter_by_event_type(self, db_session) -> None:
        """依 event_type 篩選查詢正確。"""
        from sqlalchemy import select

        records = [
            self._make_ann("2330", "001", "法說會", "earnings_call"),
            self._make_ann("2330", "002", "月營收公告", "revenue"),
            self._make_ann("2330", "003", "一般公告", "general"),
        ]
        db_session.add_all(records)
        db_session.flush()

        rows = (
            db_session.execute(select(Announcement).where(Announcement.event_type == "earnings_call")).scalars().all()
        )
        assert len(rows) == 1
        assert rows[0].subject == "法說會"

    def test_exclude_general_in_alert(self, db_session) -> None:
        """event_type != general 篩選只回傳非一般性公告。"""
        from sqlalchemy import select

        records = [
            self._make_ann("6505", "001", "法說會", "earnings_call"),
            self._make_ann("6505", "002", "財報", "filing"),
            self._make_ann("6505", "003", "一般事項", "general"),
        ]
        db_session.add_all(records)
        db_session.flush()

        rows = db_session.execute(select(Announcement).where(Announcement.event_type != "general")).scalars().all()
        assert len(rows) == 2
        event_types = {r.event_type for r in rows}
        assert "general" not in event_types


# ------------------------------------------------------------------ #
#  _compute_revenue_scan 純函數（in-memory SQLite）
# ------------------------------------------------------------------ #


def _insert_monthly_revenue(session, stock_id, rev_date, revenue, yoy, mom):
    """輔助：插入月營收記錄。"""
    session.add(
        MonthlyRevenue(
            stock_id=stock_id,
            date=rev_date,
            revenue=revenue,
            revenue_month=rev_date.month,
            revenue_year=rev_date.year,
            yoy_growth=yoy,
            mom_growth=mom,
        )
    )


def _insert_financial(session, stock_id, fin_date, gross_margin, year=2024, quarter=3):
    """輔助：插入財報記錄。"""
    session.add(
        FinancialStatement(
            stock_id=stock_id,
            date=fin_date,
            year=year,
            quarter=quarter,
            gross_margin=gross_margin,
        )
    )


class TestComputeRevenueScan:
    """_compute_revenue_scan 純函數測試。"""

    def _run(self, db_session, watchlist, min_yoy=10.0, min_margin=0.0):
        """monkeypatch get_session 並執行 _compute_revenue_scan。"""
        from unittest.mock import patch

        import src.data.database as db_mod
        from main import _compute_revenue_scan

        # db_session 已被 conftest monkeypatched；但 _compute_revenue_scan 使用
        # with get_session() as session: 形式。我們讓 get_session() 回傳一個
        # context manager，其 __enter__ 回傳 db_session。
        class _FakeCtx:
            def __enter__(self):
                return db_session

            def __exit__(self, *a):
                pass

        with patch.object(db_mod, "get_session", return_value=_FakeCtx()):
            return _compute_revenue_scan(watchlist, min_yoy, min_margin)

    # --- 基本案例 ---

    def test_empty_when_no_revenue_data(self, db_session) -> None:
        """DB 無資料時回傳空 DataFrame。"""
        df = self._run(db_session, ["9999"], min_yoy=10.0)
        assert df.empty

    def test_filters_by_min_yoy(self, db_session) -> None:
        """yoy_growth < min_yoy 的股票被過濾掉。"""
        _insert_monthly_revenue(db_session, "A001", date(2025, 10, 1), 1_000_000, yoy=5.0, mom=2.0)
        _insert_monthly_revenue(db_session, "A002", date(2025, 10, 1), 2_000_000, yoy=20.0, mom=3.0)
        db_session.flush()

        df = self._run(db_session, ["A001", "A002"], min_yoy=10.0)
        assert "A001" not in df["stock_id"].values
        assert "A002" in df["stock_id"].values

    def test_returns_expected_columns(self, db_session) -> None:
        """回傳 DataFrame 包含所有必要欄位。"""
        _insert_monthly_revenue(db_session, "B001", date(2025, 10, 1), 1_000_000, yoy=15.0, mom=5.0)
        db_session.flush()

        df = self._run(db_session, ["B001"], min_yoy=10.0)
        expected_cols = {"stock_id", "yoy_growth", "mom_growth", "gross_margin", "margin_change", "revenue_rank"}
        assert expected_cols.issubset(set(df.columns))

    def test_sorted_by_revenue_rank_desc(self, db_session) -> None:
        """結果依 revenue_rank 降序排列（高成長排前面）。"""
        _insert_monthly_revenue(db_session, "C001", date(2025, 10, 1), 1_000_000, yoy=12.0, mom=1.0)
        _insert_monthly_revenue(db_session, "C002", date(2025, 10, 1), 2_000_000, yoy=30.0, mom=5.0)
        _insert_monthly_revenue(db_session, "C003", date(2025, 10, 1), 1_500_000, yoy=20.0, mom=3.0)
        db_session.flush()

        df = self._run(db_session, ["C001", "C002", "C003"], min_yoy=10.0)
        # revenue_rank 必須單調遞減（或相等）
        ranks = df["revenue_rank"].tolist()
        assert ranks == sorted(ranks, reverse=True)

    def test_uses_latest_revenue_per_stock(self, db_session) -> None:
        """同一股票多筆月營收，只使用最新一筆。"""
        _insert_monthly_revenue(db_session, "D001", date(2025, 9, 1), 900_000, yoy=8.0, mom=1.0)
        _insert_monthly_revenue(db_session, "D001", date(2025, 10, 1), 1_000_000, yoy=25.0, mom=5.0)
        db_session.flush()

        df = self._run(db_session, ["D001"], min_yoy=10.0)
        # 以最新的 10 月資料為準（yoy=25 ≥ 10）
        assert not df.empty
        assert df.iloc[0]["yoy_growth"] == pytest.approx(25.0)

    def test_margin_change_computed_from_two_quarters(self, db_session) -> None:
        """毛利率 QoQ 計算：最新季 - 前一季。"""
        _insert_monthly_revenue(db_session, "E001", date(2025, 10, 1), 1_000_000, yoy=20.0, mom=3.0)
        _insert_financial(db_session, "E001", date(2025, 9, 30), gross_margin=40.0, year=2025, quarter=3)
        _insert_financial(db_session, "E001", date(2025, 6, 30), gross_margin=35.0, year=2025, quarter=2)
        db_session.flush()

        df = self._run(db_session, ["E001"], min_yoy=10.0)
        assert not df.empty
        # margin_change = 40.0 - 35.0 = 5.0
        assert df.iloc[0]["margin_change"] == pytest.approx(5.0)

    def test_margin_change_is_none_with_single_quarter(self, db_session) -> None:
        """只有一季財報時，margin_change 為 None/NaN。"""
        _insert_monthly_revenue(db_session, "F001", date(2025, 10, 1), 1_000_000, yoy=20.0, mom=3.0)
        _insert_financial(db_session, "F001", date(2025, 9, 30), gross_margin=38.0, year=2025, quarter=3)
        db_session.flush()

        df = self._run(db_session, ["F001"], min_yoy=10.0)
        assert not df.empty
        assert pd.isna(df.iloc[0]["margin_change"])

    def test_filters_by_min_margin_improve(self, db_session) -> None:
        """min_margin_improve > 0 時，毛利率未改善的股票被過濾掉。"""
        _insert_monthly_revenue(db_session, "G001", date(2025, 10, 1), 1_000_000, yoy=20.0, mom=3.0)
        _insert_monthly_revenue(db_session, "G002", date(2025, 10, 1), 1_500_000, yoy=25.0, mom=4.0)
        _insert_financial(db_session, "G001", date(2025, 9, 30), gross_margin=30.0, year=2025, quarter=3)
        _insert_financial(
            db_session, "G001", date(2025, 6, 30), gross_margin=35.0, year=2025, quarter=2
        )  # margin_change = -5
        _insert_financial(db_session, "G002", date(2025, 9, 30), gross_margin=42.0, year=2025, quarter=3)
        _insert_financial(
            db_session, "G002", date(2025, 6, 30), gross_margin=38.0, year=2025, quarter=2
        )  # margin_change = +4
        db_session.flush()

        df = self._run(db_session, ["G001", "G002"], min_yoy=10.0, min_margin=0.0)
        # min_margin=0.0：G001 margin_change=-5 < 0 被過濾
        assert "G001" not in df["stock_id"].values
        assert "G002" in df["stock_id"].values

    def test_revenue_rank_weight(self, db_session) -> None:
        """revenue_rank = yoy_rank * 0.70 + margin_rank * 0.30（相對排名，區間 [0,1]）。"""
        for sid, yoy, gm_new, gm_old in [
            ("H001", 15.0, 40.0, 35.0),
            ("H002", 25.0, 50.0, 40.0),
        ]:
            _insert_monthly_revenue(db_session, sid, date(2025, 10, 1), 1_000_000, yoy=yoy, mom=2.0)
            _insert_financial(db_session, sid, date(2025, 9, 30), gross_margin=gm_new, year=2025, quarter=3)
            _insert_financial(db_session, sid, date(2025, 6, 30), gross_margin=gm_old, year=2025, quarter=2)
        db_session.flush()

        df = self._run(db_session, ["H001", "H002"], min_yoy=10.0)
        assert len(df) == 2
        # 全部 revenue_rank 應在 0~1 之間
        assert df["revenue_rank"].between(0, 1).all()

    def test_no_financial_data_margin_is_nan(self, db_session) -> None:
        """無財報資料時，gross_margin / margin_change 為 NaN，但仍可顯示（不因此被過濾）。"""
        _insert_monthly_revenue(db_session, "I001", date(2025, 10, 1), 1_000_000, yoy=15.0, mom=2.0)
        db_session.flush()

        df = self._run(db_session, ["I001"], min_yoy=10.0, min_margin=0.0)
        # min_margin=0.0，NaN margin_change 視為通過（mask_margin uses isna() OR >=）
        assert not df.empty
        assert pd.isna(df.iloc[0]["gross_margin"])
