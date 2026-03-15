"""測試 src/industry/analyzer.py — compute_flow_acceleration_from_df 純函數。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.industry.analyzer import compute_flow_acceleration_from_df


# ---------------------------------------------------------------------------
# 輔助函數
# ---------------------------------------------------------------------------

def _make_inst_df(rows: list[dict]) -> pd.DataFrame:
    """建立測試用法人資料 DataFrame（stock_id, date, net, industry）。"""
    return pd.DataFrame(rows)


def _days_ago(n: int) -> date:
    return date.today() - timedelta(days=n)


# ---------------------------------------------------------------------------
# TestComputeFlowAccelerationFromDf
# ---------------------------------------------------------------------------

class TestComputeFlowAccelerationFromDf:
    """compute_flow_acceleration_from_df() 純函數測試。"""

    def test_acceleration_positive_when_flow_increasing(self):
        """近期資金流入 > 前期 → 加速度為正值。"""
        # 建立 20 個交易日資料（recent_days=5, base_days=15）
        rows = []
        # 近 5 日：每日淨買超 +2000（高流入）
        for i in range(5):
            rows.append({"stock_id": "2330", "date": _days_ago(i), "net": 2000, "industry": "半導體"})
        # 前 15 日：每日淨買超 +500（低流入）
        for i in range(5, 20):
            rows.append({"stock_id": "2330", "date": _days_ago(i), "net": 500, "industry": "半導體"})

        df = _make_inst_df(rows)
        result = compute_flow_acceleration_from_df(df, recent_days=5, base_days=15)

        assert "半導體" in result.index
        assert result["半導體"] > 0

    def test_acceleration_negative_when_flow_decreasing(self):
        """近期資金流入 < 前期 → 加速度為負值。"""
        rows = []
        # 近 5 日：每日 +200（低流入）
        for i in range(5):
            rows.append({"stock_id": "2330", "date": _days_ago(i), "net": 200, "industry": "半導體"})
        # 前 15 日：每日 +3000（高流入）
        for i in range(5, 20):
            rows.append({"stock_id": "2330", "date": _days_ago(i), "net": 3000, "industry": "半導體"})

        df = _make_inst_df(rows)
        result = compute_flow_acceleration_from_df(df, recent_days=5, base_days=15)

        assert result["半導體"] < 0

    def test_acceleration_near_zero_when_flat(self):
        """資金流平穩（每日相同）→ 加速度 ≈ 0。"""
        rows = []
        for i in range(20):
            rows.append({"stock_id": "2330", "date": _days_ago(i), "net": 1000, "industry": "半導體"})

        df = _make_inst_df(rows)
        result = compute_flow_acceleration_from_df(df, recent_days=5, base_days=15)

        assert abs(result["半導體"]) < 1e-6

    def test_fallback_when_insufficient_data(self):
        """資料不足 (recent_days + base_days) 個交易日 → 回傳全零。"""
        # 只有 10 個交易日，但需要 5+15=20
        rows = []
        for i in range(10):
            rows.append({"stock_id": "2330", "date": _days_ago(i), "net": 999, "industry": "半導體"})

        df = _make_inst_df(rows)
        result = compute_flow_acceleration_from_df(df, recent_days=5, base_days=15)

        # 應回傳全零，而非 NaN 或報錯
        assert result["半導體"] == 0.0

    def test_empty_df_returns_empty_series(self):
        """空 DataFrame → 回傳空 Series。"""
        df = pd.DataFrame(columns=["stock_id", "date", "net", "industry"])
        result = compute_flow_acceleration_from_df(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_multiple_industries_independent(self):
        """不同產業的加速度互不影響。"""
        rows = []
        # 半導體：近期高流入
        for i in range(5):
            rows.append({"stock_id": "2330", "date": _days_ago(i), "net": 5000, "industry": "半導體"})
        for i in range(5, 20):
            rows.append({"stock_id": "2330", "date": _days_ago(i), "net": 100, "industry": "半導體"})
        # 金融：近期低流入（甚至流出）
        for i in range(5):
            rows.append({"stock_id": "2882", "date": _days_ago(i), "net": -200, "industry": "金融"})
        for i in range(5, 20):
            rows.append({"stock_id": "2882", "date": _days_ago(i), "net": 2000, "industry": "金融"})

        df = _make_inst_df(rows)
        result = compute_flow_acceleration_from_df(df, recent_days=5, base_days=15)

        assert result["半導體"] > 0
        assert result["金融"] < 0
        # 兩者方向相反
        assert result["半導體"] > result["金融"]
