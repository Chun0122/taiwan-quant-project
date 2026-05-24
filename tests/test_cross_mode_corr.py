"""P2 任務 13 — 跨模式 score 相關性研究測試。

涵蓋：
  P13-A compute_cross_mode_correlation 純函數（perfect corr / 反向 / 對稱 / min_pairs 門檻）
  P13-B compute_mode_overlap 純函數
  P13-C format_cross_mode_report 輸出
  P13-D load_mode_scores DB 整合
  P13-E CLI handler exit code
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd
import pytest

from src.data.schema import DiscoveryRecord
from src.discovery.cross_mode_corr import (
    MODES,
    compute_cross_mode_correlation,
    compute_mode_overlap,
    format_cross_mode_report,
    load_mode_scores,
)


def _long_df(rows: list[tuple]) -> pd.DataFrame:
    """rows: list of (scan_date, mode, stock_id, composite_score)."""
    return pd.DataFrame([{"scan_date": d, "mode": m, "stock_id": s, "composite_score": sc} for (d, m, s, sc) in rows])


# ====================================================================== #
# P13-A: compute_cross_mode_correlation
# ====================================================================== #


class TestComputeCrossModeCorrelation:
    def test_empty_df_returns_empty(self):
        assert compute_cross_mode_correlation(pd.DataFrame()).empty

    def test_single_mode_returns_empty(self):
        df = _long_df([(date(2026, 5, 1), "momentum", f"s{i}", i * 0.1) for i in range(10)])
        assert compute_cross_mode_correlation(df).empty

    def test_perfect_positive_correlation(self):
        """兩 mode score 完全同序 → corr ≈ +1。"""
        rows = []
        d = date(2026, 5, 1)
        for i in range(10):
            rows.append((d, "momentum", f"s{i}", float(i)))
            rows.append((d, "swing", f"s{i}", float(i) * 2))  # 同序
        m = compute_cross_mode_correlation(_long_df(rows), min_pairs=5)
        assert m.loc["momentum", "swing"] == pytest.approx(1.0, abs=1e-6)
        assert m.loc["momentum", "momentum"] == 1.0

    def test_perfect_negative_correlation(self):
        rows = []
        d = date(2026, 5, 1)
        for i in range(10):
            rows.append((d, "momentum", f"s{i}", float(i)))
            rows.append((d, "value", f"s{i}", float(-i)))  # 反序
        m = compute_cross_mode_correlation(_long_df(rows), min_pairs=5)
        assert m.loc["momentum", "value"] == pytest.approx(-1.0, abs=1e-6)

    def test_matrix_is_symmetric(self):
        rows = []
        d = date(2026, 5, 1)
        for i in range(10):
            rows.append((d, "momentum", f"s{i}", float(i)))
            rows.append((d, "swing", f"s{i}", float(i % 3)))
        m = compute_cross_mode_correlation(_long_df(rows), min_pairs=5)
        assert m.loc["momentum", "swing"] == m.loc["swing", "momentum"]

    def test_min_pairs_threshold_excludes_low_sample(self):
        """共同股票 < min_pairs → 該對 corr 為 NaN。"""
        rows = []
        d = date(2026, 5, 1)
        # momentum 有 10 檔，swing 只有 2 檔重疊
        for i in range(10):
            rows.append((d, "momentum", f"s{i}", float(i)))
        rows.append((d, "swing", "s0", 1.0))
        rows.append((d, "swing", "s1", 2.0))
        m = compute_cross_mode_correlation(_long_df(rows), min_pairs=5)
        assert pd.isna(m.loc["momentum", "swing"])

    def test_averages_across_dates(self):
        """多個 scan_date 的 corr 取平均。"""
        rows = []
        # day 1: 完全正相關
        d1 = date(2026, 5, 1)
        for i in range(10):
            rows.append((d1, "momentum", f"s{i}", float(i)))
            rows.append((d1, "swing", f"s{i}", float(i)))
        # day 2: 完全負相關
        d2 = date(2026, 5, 2)
        for i in range(10):
            rows.append((d2, "momentum", f"s{i}", float(i)))
            rows.append((d2, "swing", f"s{i}", float(-i)))
        m = compute_cross_mode_correlation(_long_df(rows), min_pairs=5)
        # 平均 (1 + -1) / 2 = 0
        assert m.loc["momentum", "swing"] == pytest.approx(0.0, abs=1e-6)

    def test_constant_scores_skipped(self):
        """某 mode 當日 score 全相同（std=0）→ 不計入。"""
        rows = []
        d = date(2026, 5, 1)
        for i in range(10):
            rows.append((d, "momentum", f"s{i}", float(i)))
            rows.append((d, "swing", f"s{i}", 5.0))  # 全常數
        m = compute_cross_mode_correlation(_long_df(rows), min_pairs=5)
        assert pd.isna(m.loc["momentum", "swing"])


# ====================================================================== #
# P13-B: compute_mode_overlap
# ====================================================================== #


class TestComputeModeOverlap:
    def test_empty_returns_empty(self):
        assert compute_mode_overlap(pd.DataFrame()).empty

    def test_shared_count(self):
        d = date(2026, 5, 1)
        rows = [
            (d, "momentum", "A", 1.0),
            (d, "momentum", "B", 1.0),
            (d, "momentum", "C", 1.0),
            (d, "swing", "B", 1.0),
            (d, "swing", "C", 1.0),
            (d, "swing", "D", 1.0),
        ]
        m = compute_mode_overlap(_long_df(rows))
        # momentum ∩ swing = {B, C} = 2
        assert m.loc["momentum", "swing"] == pytest.approx(2.0)
        # 對角線 = 該 mode 推薦數
        assert m.loc["momentum", "momentum"] == pytest.approx(3.0)
        assert m.loc["swing", "swing"] == pytest.approx(3.0)

    def test_overlap_averaged_over_dates(self):
        rows = [
            # day1: momentum∩swing = 1
            (date(2026, 5, 1), "momentum", "A", 1.0),
            (date(2026, 5, 1), "swing", "A", 1.0),
            (date(2026, 5, 1), "swing", "X", 1.0),
            # day2: momentum∩swing = 3
            (date(2026, 5, 2), "momentum", "A", 1.0),
            (date(2026, 5, 2), "momentum", "B", 1.0),
            (date(2026, 5, 2), "momentum", "C", 1.0),
            (date(2026, 5, 2), "swing", "A", 1.0),
            (date(2026, 5, 2), "swing", "B", 1.0),
            (date(2026, 5, 2), "swing", "C", 1.0),
        ]
        m = compute_mode_overlap(_long_df(rows))
        # 平均 (1 + 3) / 2 = 2.0
        assert m.loc["momentum", "swing"] == pytest.approx(2.0)


# ====================================================================== #
# P13-C: format_cross_mode_report
# ====================================================================== #


class TestFormatReport:
    def test_empty_matrix_message(self):
        out = format_cross_mode_report(pd.DataFrame(), pd.DataFrame(), lookback_days=60, n_scan_dates=0)
        assert "資料不足" in out

    def test_high_corr_flagged(self):
        modes = ["momentum", "swing"]
        corr = pd.DataFrame([[1.0, 0.85], [0.85, 1.0]], index=modes, columns=modes)
        overlap = pd.DataFrame([[10.0, 5.0], [5.0, 8.0]], index=modes, columns=modes)
        out = format_cross_mode_report(corr, overlap, lookback_days=60, n_scan_dates=30)
        assert "高度冗餘" in out
        assert "momentum" in out

    def test_complementary_flagged(self):
        modes = ["momentum", "value"]
        corr = pd.DataFrame([[1.0, -0.4], [-0.4, 1.0]], index=modes, columns=modes)
        overlap = pd.DataFrame([[10.0, 1.0], [1.0, 8.0]], index=modes, columns=modes)
        out = format_cross_mode_report(corr, overlap, lookback_days=60, n_scan_dates=30)
        assert "互補對沖" in out

    def test_no_flag_message(self):
        modes = ["momentum", "swing"]
        corr = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=modes, columns=modes)
        overlap = pd.DataFrame([[10.0, 2.0], [2.0, 8.0]], index=modes, columns=modes)
        out = format_cross_mode_report(corr, overlap, lookback_days=60, n_scan_dates=30)
        assert "無顯著高相關" in out


# ====================================================================== #
# P13-D: load_mode_scores DB 整合
# ====================================================================== #


class TestLoadModeScores:
    def test_loads_within_lookback(self, db_session, monkeypatch):
        from src.discovery import cross_mode_corr as cmc_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(cmc_mod, "get_session", lambda: _Ctx(db_session))

        today = date(2026, 5, 18)
        # 一筆在窗口內、一筆在窗口外
        db_session.add(
            DiscoveryRecord(
                scan_date=today,
                mode="momentum",
                rank=1,
                stock_id="2330",
                stock_name="t",
                close=100,
                composite_score=0.8,
            )
        )
        db_session.add(
            DiscoveryRecord(
                scan_date=date(2026, 1, 1),  # 遠超 60 天
                mode="momentum",
                rank=1,
                stock_id="2317",
                stock_name="t",
                close=100,
                composite_score=0.7,
            )
        )
        db_session.commit()

        df = load_mode_scores(lookback_days=60, end_date=today)
        assert len(df) == 1
        assert df.iloc[0]["stock_id"] == "2330"


# ====================================================================== #
# P13-E: CLI handler
# ====================================================================== #


class TestCmdCrossModeCorr:
    def test_empty_db_returns_zero(self, db_session, monkeypatch, capsys):
        import src.cli.helpers as helpers
        from src.discovery import cross_mode_corr as cmc_mod

        class _Ctx:
            def __init__(self, s):
                self._s = s

            def __enter__(self):
                return self._s

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(cmc_mod, "get_session", lambda: _Ctx(db_session))
        monkeypatch.setattr(helpers, "init_db", lambda: None)

        from src.cli.discover_cmd import cmd_cross_mode_corr

        exit_code = cmd_cross_mode_corr(argparse.Namespace(lookback_days=60, min_pairs=5, export=None))
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "無 DiscoveryRecord" in out

    def test_modes_constant_has_5(self):
        assert MODES == ("momentum", "swing", "value", "dividend", "growth")
