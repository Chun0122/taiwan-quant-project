"""_upsert_dividend 的 year NaN 回填測試（R1 重審發現的股利斷流 bug）。

FinMind 新資料 year 欄一律 NaN，舊版 dropna(subset=["year"]) 會把全部新股利
列丟棄 → sync_dividends_for_stocks 寫入恆 0、A3 除息入帳斷流。
修復：year NaN 以 date 年份回填，不丟行。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.data.schema import Dividend


class _CtxSession:
    def __init__(self, s):
        self._s = s

    def __enter__(self):
        return self._s

    def __exit__(self, *a):
        pass


def _finmind_like_df() -> pd.DataFrame:
    """模擬 FinMind fetch_dividend 回傳（year 全 NaN、日期欄 datetime64 的新資料）。"""
    df = pd.DataFrame(
        {
            "date": ["2026-06-26", "2026-07-06"],
            "stock_id": ["2603", "2882"],
            "year": [np.nan, np.nan],
            "cash_dividend": [16.0, 3.5],
            "stock_dividend": [0.0, 0.0],
            "cash_payment_date": ["2026-07-17", "2026-08-11"],
            "announcement_date": ["2026-06-01", "2026-05-20"],
            "cash_ex_dividend_date": ["2026-06-17", "2026-06-30"],
            "stock_ex_dividend_date": [pd.NaT, pd.NaT],
        }
    )
    for col in ("date", "cash_payment_date", "announcement_date", "cash_ex_dividend_date", "stock_ex_dividend_date"):
        df[col] = pd.to_datetime(df[col])
    return df


class TestUpsertDividendYearBackfill:
    def test_nan_year_rows_are_written_not_dropped(self, db_session, monkeypatch):
        from src.data import pipeline

        monkeypatch.setattr(pipeline, "get_session", lambda: _CtxSession(db_session))
        n = pipeline._upsert_dividend(_finmind_like_df())
        assert n == 2  # 舊版此處為 0（整批被 dropna 丟棄）

        rows = db_session.execute(select(Dividend).order_by(Dividend.stock_id)).scalars().all()
        assert [r.stock_id for r in rows] == ["2603", "2882"]
        # year 以 date 年份回填（ORM year=Integer，SQLAlchemy 將 '2026' coerce 為 int）
        assert all(r.year == 2026 for r in rows)
        assert rows[0].cash_dividend == 16.0

    def test_existing_year_preserved(self, db_session, monkeypatch):
        from src.data import pipeline

        monkeypatch.setattr(pipeline, "get_session", lambda: _CtxSession(db_session))
        df = _finmind_like_df()
        df.loc[0, "year"] = "2025"  # 舊資料有 year 值 → 不覆蓋
        pipeline._upsert_dividend(df)
        row = db_session.execute(select(Dividend).where(Dividend.stock_id == "2603")).scalar_one()
        assert row.year == 2025

    def test_empty_df_returns_zero(self, db_session, monkeypatch):
        from src.data import pipeline

        monkeypatch.setattr(pipeline, "get_session", lambda: _CtxSession(db_session))
        assert pipeline._upsert_dividend(pd.DataFrame()) == 0
