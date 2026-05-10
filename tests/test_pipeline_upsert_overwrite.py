"""C2 修復回歸測試 — _upsert_batch update_cols 正確覆蓋舊值。

對應 audit 2026-05-09 P0-C2：
- 原 _upsert_batch 硬編碼 on_conflict_do_nothing →
  TechnicalIndicator / DailyFeature 同日重算的舊值永不覆蓋
- 修復：新增 update_cols 參數，TechnicalIndicator/DailyFeature 顯式啟用 do_update

測試範圍：
1. _upsert_batch(update_cols=None) → 預設 do_nothing 行為（向後相容）
2. _upsert_batch(update_cols=[...]) → do_update 覆蓋指定欄位
3. _upsert_indicators 的 value 欄位重寫驗證（除權息回溯場景）
4. DailyFeature 的特徵欄位重寫驗證（同日盤中→盤後重算場景）
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from sqlalchemy import select

from src.data.pipeline import _upsert_batch, _upsert_indicators
from src.data.schema import (
    Announcement,
    DailyFeature,
    TechnicalIndicator,
)

# ─────────────────────────────────────────────────────────────────
#  C2-A: _upsert_batch update_cols=None 維持 do_nothing 行為
# ─────────────────────────────────────────────────────────────────


class TestUpsertBatchBackwardCompat:
    def test_no_update_cols_preserves_first_value(self, db_session):
        """update_cols=None：衝突時舊值保留（不可變歷史紀錄場景）。"""
        df1 = pd.DataFrame(
            [
                {
                    "stock_id": "2330",
                    "date": date(2026, 5, 8),
                    "seq": 1,
                    "subject": "原始公告",
                    "sentiment": 0,
                    "event_type": "general",
                }
            ]
        )
        _upsert_batch(Announcement, df1, ["stock_id", "date", "seq"])

        # 第二次寫入相同 conflict_keys 但內容不同
        df2 = df1.copy()
        df2["subject"] = "修改後"
        _upsert_batch(Announcement, df2, ["stock_id", "date", "seq"])

        rows = db_session.execute(select(Announcement)).scalars().all()
        assert len(rows) == 1
        assert rows[0].subject == "原始公告"  # 舊值保留


# ─────────────────────────────────────────────────────────────────
#  C2-B: _upsert_batch update_cols=[...] 覆蓋指定欄位
# ─────────────────────────────────────────────────────────────────


class TestUpsertBatchDoUpdate:
    def test_update_cols_overwrites_value(self, db_session):
        """update_cols=['value']：衝突時 value 欄位被新值覆蓋。"""
        df1 = pd.DataFrame(
            [
                {
                    "stock_id": "2330",
                    "date": date(2026, 5, 8),
                    "name": "sma_20",
                    "value": 580.0,
                }
            ]
        )
        _upsert_batch(TechnicalIndicator, df1, ["stock_id", "date", "name"], update_cols=["value"])

        # 重算（如除權息調整）後寫入新值
        df2 = df1.copy()
        df2["value"] = 555.5
        _upsert_batch(TechnicalIndicator, df2, ["stock_id", "date", "name"], update_cols=["value"])

        rows = db_session.execute(select(TechnicalIndicator)).scalars().all()
        assert len(rows) == 1
        assert rows[0].value == 555.5  # 新值覆蓋

    def test_update_cols_subset_only_overwrites_listed(self, db_session):
        """update_cols 只列出部分欄位時，未列出的欄位保留舊值。"""
        # 用 Announcement 測試：update_cols=["sentiment"] 只覆蓋 sentiment，subject 保留
        df1 = pd.DataFrame(
            [
                {
                    "stock_id": "2330",
                    "date": date(2026, 5, 8),
                    "seq": 1,
                    "subject": "原始 subject",
                    "sentiment": 0,
                    "event_type": "general",
                }
            ]
        )
        _upsert_batch(Announcement, df1, ["stock_id", "date", "seq"])

        df2 = df1.copy()
        df2["subject"] = "新 subject（不應覆蓋）"
        df2["sentiment"] = 1
        _upsert_batch(Announcement, df2, ["stock_id", "date", "seq"], update_cols=["sentiment"])

        row = db_session.execute(select(Announcement)).scalars().first()
        assert row.subject == "原始 subject"  # 未列出，保留
        assert row.sentiment == 1  # 列出，覆蓋


# ─────────────────────────────────────────────────────────────────
#  C2-C: _upsert_indicators 重算正確覆蓋（除權息回溯場景）
# ─────────────────────────────────────────────────────────────────


class TestUpsertIndicatorsOverwrite:
    def test_dividend_adjustment_overwrites_old_sma(self, db_session):
        """除權息回溯後重算 SMA → 舊未調整值應被覆蓋（C2 修復重點場景）。"""
        # 第一次：未調整除權息的 sma_20
        raw_df = pd.DataFrame(
            [
                {"stock_id": "2330", "date": date(2026, 5, 8), "name": "sma_20", "value": 600.0},
                {"stock_id": "2330", "date": date(2026, 5, 8), "name": "sma_60", "value": 580.0},
            ]
        )
        _upsert_indicators(raw_df)

        # 模擬除權息回溯：價格 ×0.95（配息 5%）→ 重算後 sma 略低
        adjusted_df = pd.DataFrame(
            [
                {"stock_id": "2330", "date": date(2026, 5, 8), "name": "sma_20", "value": 570.0},
                {"stock_id": "2330", "date": date(2026, 5, 8), "name": "sma_60", "value": 551.0},
            ]
        )
        _upsert_indicators(adjusted_df)

        rows = db_session.execute(select(TechnicalIndicator).order_by(TechnicalIndicator.name)).scalars().all()
        assert len(rows) == 2  # 沒有重複，仍然是 2 筆
        sma_20 = next(r for r in rows if r.name == "sma_20")
        sma_60 = next(r for r in rows if r.name == "sma_60")
        assert sma_20.value == 570.0  # 已覆蓋
        assert sma_60.value == 551.0  # 已覆蓋


# ─────────────────────────────────────────────────────────────────
#  C2-D: DailyFeature 同日盤中→盤後重算正確覆蓋
# ─────────────────────────────────────────────────────────────────


class TestDailyFeatureOverwrite:
    def test_intraday_to_eod_recompute_overwrites_features(self, db_session):
        """同日盤中算過特徵後，盤後再算一次應該覆蓋（避免 stale momentum/MA）。"""
        # 模擬 _upsert_batch(DailyFeature, df_out, ["stock_id", "date"], update_cols=...)
        # 的呼叫
        feature_cols = [
            "close",
            "volume",
            "turnover",
            "ma20",
            "ma60",
            "volume_ma20",
            "turnover_ma5",
            "turnover_ma20",
            "momentum_20d",
            "volatility_20d",
            "turnover_ratio_5d_20d",
            "high_20d",
            "computed_at",
        ]

        # 盤中暫定特徵
        intraday_df = pd.DataFrame(
            [
                {
                    "stock_id": "2330",
                    "date": date(2026, 5, 8),
                    "close": 580.0,
                    "volume": 10_000_000,
                    "turnover": 5_800_000_000,
                    "ma20": 575.0,
                    "ma60": 560.0,
                    "volume_ma20": 9_000_000,
                    "turnover_ma5": 5_500_000_000,
                    "turnover_ma20": 5_200_000_000,
                    "momentum_20d": 2.5,
                    "volatility_20d": 22.0,
                    "turnover_ratio_5d_20d": 1.06,
                    "high_20d": 590.0,
                    "computed_at": datetime(2026, 5, 8, 13, 30),
                }
            ]
        )
        _upsert_batch(DailyFeature, intraday_df, ["stock_id", "date"], update_cols=feature_cols)

        # 盤後正式特徵（值不同）
        eod_df = intraday_df.copy()
        eod_df["close"] = 578.5
        eod_df["volume"] = 12_500_000
        eod_df["momentum_20d"] = 2.2
        eod_df["computed_at"] = datetime(2026, 5, 8, 23, 30)
        _upsert_batch(DailyFeature, eod_df, ["stock_id", "date"], update_cols=feature_cols)

        rows = db_session.execute(select(DailyFeature)).scalars().all()
        assert len(rows) == 1
        assert rows[0].close == 578.5  # 盤後覆蓋
        assert rows[0].volume == 12_500_000
        assert rows[0].momentum_20d == 2.2
