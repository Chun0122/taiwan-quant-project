"""2026-05-15 audit — P0 修復測試。

對應 logs/audit_20260515/ all10_5d 5/7-5/8 三連停損追查：
1. scanner 過熱反轉懲罰閘門（_apply_overheating_filter）
2. rotation 'all' 模式 mode 配額（_resolve_all_mode_rankings per_mode_max）
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.discovery.scanner import MomentumScanner, SwingScanner

# ====================================================================== #
#  P0-1: 過熱反轉懲罰閘門
# ====================================================================== #


def _make_price_df_with_returns(
    stock_ret: dict[str, float] | dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """建立含指定 5 日 / 10 日報酬的 df_price（11 天 close 序列）。

    傳值可為：
      - float（ret_5d）：自動設 ret_10d = ret_5d（前 5 日持平）
      - tuple (ret_5d, ret_10d)：兩段獨立設定
        前 5 日（i=0..5）從 close_10d_ago 線性漲到 close_5d_ago；
        後 5 日（i=5..10）從 close_5d_ago 線性漲到 close_end（=100）。
    """
    rows = []
    base_date = date(2025, 1, 1)
    for sid, ret in stock_ret.items():
        ret5, ret10 = (ret, ret) if isinstance(ret, (int, float)) else ret
        close_end = 100.0
        close_5d_ago = close_end / (1.0 + ret5)
        close_10d_ago = close_end / (1.0 + ret10)
        for i in range(11):
            if i <= 5:
                t = i / 5.0
                close = close_10d_ago + (close_5d_ago - close_10d_ago) * t
            else:
                t = (i - 5) / 5.0
                close = close_5d_ago + (close_end - close_5d_ago) * t
            rows.append(
                {
                    "stock_id": sid,
                    "date": base_date + timedelta(days=i),
                    "close": close,
                }
            )
    return pd.DataFrame(rows)


def _make_scored(stock_ids: list[str], composite: float = 0.8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": stock_ids,
            "composite_score": [composite for _ in stock_ids],
            "technical_score": [0.9 for _ in stock_ids],
        }
    )


class TestOverheatingFilter:
    """MarketScanner._apply_overheating_filter — 過熱反轉懲罰。"""

    def test_exclude_above_ret5d_threshold(self):
        """ret_5d > 35% 硬剔除。"""
        scanner = MomentumScanner()
        df_price = _make_price_df_with_returns({"HOT": (0.40, 0.45), "OK": (0.05, 0.10)})
        scored = _make_scored(["HOT", "OK"])
        result = scanner._apply_overheating_filter(scored, df_price)
        assert "HOT" not in result["stock_id"].values
        assert "OK" in result["stock_id"].values

    def test_exclude_above_ret10d_threshold(self):
        """ret_10d > 50% 硬剔除（即使 ret_5d 沒到門檻）。"""
        scanner = MomentumScanner()
        # ret_5d=0.20（dampen 區下方）但 ret_10d=0.55 > 0.50 → 硬剔除
        df_price = _make_price_df_with_returns({"HOT": (0.20, 0.55), "OK": (0.05, 0.10)})
        scored = _make_scored(["HOT", "OK"])
        result = scanner._apply_overheating_filter(scored, df_price)
        assert "HOT" not in result["stock_id"].values
        assert "OK" in result["stock_id"].values

    def test_dampen_in_warning_zone(self):
        """25% < ret_5d ≤ 35% 且 ret_10d 未超 50%：composite_score 折扣（default 0.85）。"""
        scanner = MomentumScanner()
        df_price = _make_price_df_with_returns({"WARN": (0.28, 0.30), "OK": (0.05, 0.10)})
        scored = _make_scored(["WARN", "OK"], composite=1.0)
        result = scanner._apply_overheating_filter(scored, df_price)
        warn_score = result.loc[result["stock_id"] == "WARN", "composite_score"].iloc[0]
        ok_score = result.loc[result["stock_id"] == "OK", "composite_score"].iloc[0]
        assert warn_score == pytest.approx(0.85, abs=1e-6)
        assert ok_score == pytest.approx(1.0, abs=1e-6)

    def test_safe_zone_unchanged(self):
        """ret_5d ≤ 25% 且 ret_10d ≤ 35%：不剔除也不降分。"""
        scanner = MomentumScanner()
        df_price = _make_price_df_with_returns({"OK1": (0.10, 0.20), "OK2": (0.05, 0.10)})
        scored = _make_scored(["OK1", "OK2"], composite=0.8)
        result = scanner._apply_overheating_filter(scored, df_price)
        assert len(result) == 2
        assert result["composite_score"].tolist() == pytest.approx([0.8, 0.8], abs=1e-6)

    def test_empty_scored_safe_fallback(self):
        """空 scored 不拋例外。"""
        scanner = MomentumScanner()
        df_price = _make_price_df_with_returns({"A": 0.10})
        result = scanner._apply_overheating_filter(pd.DataFrame(), df_price)
        assert result.empty

    def test_empty_price_safe_fallback(self):
        """空 df_price 不拋例外，原樣回傳。"""
        scanner = MomentumScanner()
        scored = _make_scored(["A"])
        result = scanner._apply_overheating_filter(scored, pd.DataFrame())
        assert len(result) == 1

    def test_custom_thresholds_override(self):
        """允許 caller 覆寫門檻。"""
        scanner = MomentumScanner()
        df_price = _make_price_df_with_returns({"HOT": (0.20, 0.25)})
        scored = _make_scored(["HOT"])
        # 把 exclude 拉到 15% → ret_5d=20% > 15% → 剔除
        result = scanner._apply_overheating_filter(scored, df_price, exclude_ret5d=0.15, exclude_ret10d=0.50)
        assert "HOT" not in result["stock_id"].values

    def test_swing_scanner_inherits_filter(self):
        """SwingScanner 透過 _apply_risk_filter 鏈式呼叫過熱閘門。"""
        scanner = SwingScanner()
        # 為了避開 vol_risk_filter（需要 60+ 天資料）的影響，直接呼叫 helper
        df_price = _make_price_df_with_returns({"HOT": (0.40, 0.45), "OK": (0.05, 0.10)})
        scored = _make_scored(["HOT", "OK"])
        result = scanner._apply_overheating_filter(scored, df_price)
        assert "HOT" not in result["stock_id"].values

    def test_dampen_factor_above_one_no_effect(self):
        """dampen_composite_factor=1.0 等效不降分（邊界）。"""
        scanner = MomentumScanner()
        df_price = _make_price_df_with_returns({"WARN": (0.28, 0.30)})
        scored = _make_scored(["WARN"], composite=0.9)
        result = scanner._apply_overheating_filter(scored, df_price, dampen_composite_factor=1.0)
        assert result.loc[result["stock_id"] == "WARN", "composite_score"].iloc[0] == pytest.approx(0.9, abs=1e-6)

    def test_real_case_5864_dampen(self):
        """2026-05-08 致和證實際資料：ret_5d=24.95%, ret_10d=36.6% → 觸發 dampen。"""
        scanner = MomentumScanner()
        df_price = _make_price_df_with_returns({"5864": (0.2495, 0.366)})
        scored = _make_scored(["5864"], composite=0.764)
        result = scanner._apply_overheating_filter(scored, df_price)
        # ret_5d=24.95% < 25% 門檻、ret_10d=36.6% > 35% dampen 門檻 → 進入 dampen 區
        # ret_10d=36.6% < 50% exclude 門檻 → 不硬剔除
        assert "5864" in result["stock_id"].values
        new_score = result.loc[result["stock_id"] == "5864", "composite_score"].iloc[0]
        assert new_score == pytest.approx(0.764 * 0.85, abs=1e-6)

    def test_real_case_6224_dampen(self):
        """2026-05-07 聚鼎：ret_5d=30%, ret_10d=45% → 兩者均在 dampen 區，未過 exclude。"""
        scanner = MomentumScanner()
        df_price = _make_price_df_with_returns({"6224": (0.30, 0.45)})
        scored = _make_scored(["6224"], composite=0.786)
        result = scanner._apply_overheating_filter(scored, df_price)
        # ret_5d=30% > 25% dampen、ret_10d=45% > 35% dampen，但都 < exclude（35%/50%）
        # → 軟降分而不是硬剔除
        assert "6224" in result["stock_id"].values
        new_score = result.loc[result["stock_id"] == "6224", "composite_score"].iloc[0]
        assert new_score == pytest.approx(0.786 * 0.85, abs=1e-6)

    def test_real_case_6108_borderline_no_dampen(self):
        """2026-05-08 競國：ret_5d=21%, ret_10d=33.7% → 邊界內，未觸發 dampen。"""
        scanner = MomentumScanner()
        df_price = _make_price_df_with_returns({"6108": (0.21, 0.337)})
        scored = _make_scored(["6108"], composite=0.77)
        result = scanner._apply_overheating_filter(scored, df_price)
        # ret_5d=21% < 25%，ret_10d=33.7% < 35% → 安全區，不降分（接近邊界）
        assert "6108" in result["stock_id"].values
        new_score = result.loc[result["stock_id"] == "6108", "composite_score"].iloc[0]
        assert new_score == pytest.approx(0.77, abs=1e-6)


# ====================================================================== #
#  P0-2: rotation 'all' 模式 mode 配額
# ====================================================================== #


class TestAllModePerModeQuota:
    """portfolio.manager._resolve_all_mode_rankings — per_mode_max 配額。"""

    def _seed(self, db_session, scan_date: date, rows: list[tuple[str, str, float]]):
        """寫入 DiscoveryRecord：[(stock_id, mode, composite_score), ...]。"""
        from src.data.schema import DiscoveryRecord

        for i, (sid, mode, score) in enumerate(rows):
            db_session.add(
                DiscoveryRecord(
                    scan_date=scan_date,
                    mode=mode,
                    rank=i + 1,
                    stock_id=sid,
                    stock_name=f"name_{sid}",
                    close=100.0,
                    composite_score=score,
                )
            )
        db_session.flush()

    def test_quota_caps_single_mode_dominance(self, db_session):
        """單 mode 主導前 N 名時，per_mode_max=3 限制最多 3 檔來自該 mode。

        對應 5/7-5/8 案例：swing 模式同時推 6 檔，但 mode 配額擋下後 3 檔。
        """
        from src.portfolio.manager import resolve_rankings

        # 6 檔 swing + 2 檔 momentum，swing 全部分數較高
        self._seed(
            db_session,
            date(2025, 5, 7),
            [
                ("S1", "swing", 0.90),
                ("S2", "swing", 0.88),
                ("S3", "swing", 0.86),
                ("S4", "swing", 0.84),
                ("S5", "swing", 0.82),
                ("S6", "swing", 0.80),
                ("M1", "momentum", 0.70),
                ("M2", "momentum", 0.68),
            ],
        )
        result = resolve_rankings("all", date(2025, 5, 7), db_session, top_n=10, per_mode_max=3)
        # 應該回 5 檔：3 swing + 2 momentum
        assert len(result) == 5
        swing_count = sum(1 for r in result if r["primary_mode"] == "swing")
        momentum_count = sum(1 for r in result if r["primary_mode"] == "momentum")
        assert swing_count == 3
        assert momentum_count == 2
        # 排名 1~3 為 swing，4~5 為 momentum
        assert result[0]["primary_mode"] == "swing"
        assert result[3]["primary_mode"] == "momentum"

    def test_quota_default_from_constants(self, db_session):
        """per_mode_max=None 取 constants.ROTATION_ALL_MODE_PER_MODE_MAX 預設值（3）。"""
        from src.constants import ROTATION_ALL_MODE_PER_MODE_MAX
        from src.portfolio.manager import resolve_rankings

        assert ROTATION_ALL_MODE_PER_MODE_MAX == 3
        # 5 檔 swing，per_mode_max=None → 預設 3
        self._seed(
            db_session,
            date(2025, 5, 7),
            [
                ("S1", "swing", 0.90),
                ("S2", "swing", 0.88),
                ("S3", "swing", 0.86),
                ("S4", "swing", 0.84),
                ("S5", "swing", 0.82),
            ],
        )
        result = resolve_rankings("all", date(2025, 5, 7), db_session, top_n=10)
        assert len(result) == 3

    def test_quota_zero_disables(self, db_session):
        """per_mode_max=0 等效於不限制。"""
        from src.portfolio.manager import resolve_rankings

        self._seed(
            db_session,
            date(2025, 5, 7),
            [
                ("S1", "swing", 0.90),
                ("S2", "swing", 0.88),
                ("S3", "swing", 0.86),
                ("S4", "swing", 0.84),
            ],
        )
        result = resolve_rankings("all", date(2025, 5, 7), db_session, top_n=10, per_mode_max=0)
        assert len(result) == 4

    def test_primary_mode_picks_highest_score(self, db_session):
        """股票出現在多個 mode 時，primary_mode 取最高分。"""
        from src.portfolio.manager import resolve_rankings

        # X 在 swing(0.90) + momentum(0.85) + value(0.40) → primary_mode=swing
        self._seed(
            db_session,
            date(2025, 5, 7),
            [
                ("X", "swing", 0.90),
                ("X", "momentum", 0.85),
                ("X", "value", 0.40),
                ("Y", "value", 0.50),
            ],
        )
        result = resolve_rankings("all", date(2025, 5, 7), db_session, top_n=10, per_mode_max=3)
        x_row = next(r for r in result if r["stock_id"] == "X")
        assert x_row["primary_mode"] == "swing"

    def test_quota_respects_top_n(self, db_session):
        """top_n 與 per_mode_max 同時生效，取較小限制。"""
        from src.portfolio.manager import resolve_rankings

        # 5 mode × 2 檔 = 10 檔，per_mode_max=3, top_n=4 → 取 4 檔
        rows = []
        score = 0.99
        for m in ("momentum", "swing", "value", "dividend", "growth"):
            for i in range(2):
                rows.append((f"{m}_{i}", m, score))
                score -= 0.01
        self._seed(db_session, date(2025, 5, 7), rows)
        result = resolve_rankings("all", date(2025, 5, 7), db_session, top_n=4, per_mode_max=3)
        assert len(result) == 4

    def test_non_all_mode_unaffected(self, db_session):
        """單 mode 查詢不受 per_mode_max 影響。"""
        from src.portfolio.manager import resolve_rankings

        self._seed(
            db_session,
            date(2025, 5, 7),
            [
                ("S1", "swing", 0.90),
                ("S2", "swing", 0.88),
                ("S3", "swing", 0.86),
                ("S4", "swing", 0.84),
                ("S5", "swing", 0.82),
            ],
        )
        # 直查 swing mode：應回 5 筆（per_mode_max 只對 'all' 生效）
        result = resolve_rankings("swing", date(2025, 5, 7), db_session, top_n=10)
        assert len(result) == 5
