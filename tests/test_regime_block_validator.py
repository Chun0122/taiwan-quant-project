"""REGIME_MODE_BLOCK 矩陣自動驗證器測試（P3）。

測試分三層：
  1. _judge 純函數：給定 stats + is_blocked → 判定 recommendation
  2. _compute_stats_from_dfs 純函數：DataFrame → HoldingStats
  3. validate_regime_blocks 整合：當前矩陣應通過驗證（regression test）
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.discovery.regime_block_validator import (
    BlockValidation,
    HoldingStats,
    _compute_stats_from_dfs,
    _judge,
    format_validation_report,
    validate_regime_blocks,
)

# ──────────────────────────────────────────────────────────
#  _judge 純函數判定
# ──────────────────────────────────────────────────────────


class TestJudge:
    def test_insufficient_sample_no_recommendation(self) -> None:
        """樣本 < min_samples → insufficient_data，不下判斷（避免雜訊觸發）。"""
        stats = HoldingStats(avg_return=-0.10, win_rate=0.10, sample_count=10)
        assert _judge(is_blocked=False, stats=stats, min_samples=30) == "insufficient_data"

    def test_blocked_but_profitable_recommends_lift(self) -> None:
        """已封鎖 + 高報酬 + 高勝率 → 建議解除（封鎖過嚴）。"""
        stats = HoldingStats(avg_return=0.025, win_rate=0.58, sample_count=50)
        assert _judge(is_blocked=True, stats=stats) == "lift_block"

    def test_blocked_marginal_profit_keeps(self) -> None:
        """已封鎖 + 報酬僅略正（未過 +2% 門檻）→ 維持封鎖（保守）。"""
        stats = HoldingStats(avg_return=0.015, win_rate=0.58, sample_count=50)
        assert _judge(is_blocked=True, stats=stats) == "keep"

    def test_blocked_high_return_low_win_keeps(self) -> None:
        """已封鎖 + 高報酬但勝率低（雙條件 AND）→ 維持封鎖（少數大勝壓倒多數小敗，不可靠）。"""
        stats = HoldingStats(avg_return=0.05, win_rate=0.30, sample_count=50)
        assert _judge(is_blocked=True, stats=stats) == "keep"

    def test_unblocked_disastrous_recommends_add(self) -> None:
        """未封鎖 + 嚴重虧損 + 低勝率 → 建議新增封鎖。"""
        stats = HoldingStats(avg_return=-0.035, win_rate=0.30, sample_count=40)
        assert _judge(is_blocked=False, stats=stats) == "add_block"

    def test_unblocked_marginal_loss_keeps(self) -> None:
        """未封鎖 + 報酬僅略負（未跌破 -3% 門檻）→ 維持未封鎖（保守）。"""
        stats = HoldingStats(avg_return=-0.02, win_rate=0.40, sample_count=40)
        assert _judge(is_blocked=False, stats=stats) == "keep"

    def test_unblocked_disastrous_return_high_win_keeps(self) -> None:
        """未封鎖 + 報酬差但勝率正常（雙條件 AND）→ 維持（單一極端虧損案例污染均值）。"""
        stats = HoldingStats(avg_return=-0.05, win_rate=0.50, sample_count=40)
        assert _judge(is_blocked=False, stats=stats) == "keep"

    def test_threshold_boundary_inclusive(self) -> None:
        """門檻邊界：恰好等於門檻不觸發（嚴格 > 與 <）。"""
        # 已封鎖 + 報酬 = 0.02（恰好）→ keep（需 > 0.02）
        stats_lift = HoldingStats(avg_return=0.02, win_rate=0.60, sample_count=50)
        assert _judge(is_blocked=True, stats=stats_lift) == "keep"
        # 未封鎖 + 報酬 = -0.03（恰好）→ keep（需 < -0.03）
        stats_add = HoldingStats(avg_return=-0.03, win_rate=0.30, sample_count=40)
        assert _judge(is_blocked=False, stats=stats_add) == "keep"


# ──────────────────────────────────────────────────────────
#  _compute_stats_from_dfs 純函數計算
# ──────────────────────────────────────────────────────────


class TestComputeStatsFromDfs:
    def _make_records(self, n: int, scan_d: date) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "scan_date": [scan_d] * n,
                "stock_id": [f"{i + 1000:04d}" for i in range(n)],
                "close": [100.0] * n,
                "regime": ["bull"] * n,
            }
        )

    def _make_prices(self, stock_ids: list[str], scan_d: date, returns: list[float]) -> pd.DataFrame:
        """為每支股票造價格序列：scan_d 當天 close=100，後 5 日終值決定報酬。"""
        rows = []
        for sid, ret in zip(stock_ids, returns, strict=True):
            for i in range(6):
                rows.append(
                    {
                        "stock_id": sid,
                        "date": scan_d + timedelta(days=i),
                        "close": 100.0 if i == 0 else 100.0 * (1 + ret) if i == 5 else 100.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_empty_records_returns_zero_stats(self) -> None:
        result = _compute_stats_from_dfs(pd.DataFrame(), pd.DataFrame(), holding_days=5)
        assert result.sample_count == 0
        assert result.avg_return == 0.0

    def test_basic_win_rate_and_return(self) -> None:
        """3 賺 2 虧 → 勝率 0.6、平均報酬 = mean。"""
        df_rec = self._make_records(5, date(2026, 3, 1))
        df_price = self._make_prices(
            df_rec["stock_id"].tolist(),
            date(2026, 3, 1),
            [0.05, 0.10, 0.03, -0.02, -0.08],
        )
        stats = _compute_stats_from_dfs(df_rec, df_price, holding_days=5)
        assert stats.sample_count == 5
        assert stats.win_rate == pytest.approx(0.6)
        assert stats.avg_return == pytest.approx((0.05 + 0.10 + 0.03 - 0.02 - 0.08) / 5)

    def test_skip_when_insufficient_future_days(self) -> None:
        """價格序列不足 holding_days → 該筆推薦不計入。"""
        df_rec = self._make_records(2, date(2026, 3, 1))
        # 第 1 支只有 3 天價格資料（< 5）
        rows = []
        for i in range(4):  # scan_d 當天 + 後 3 天 = 4 天 → 不足
            rows.append({"stock_id": "1000", "date": date(2026, 3, 1) + timedelta(days=i), "close": 100.0})
        # 第 2 支完整 5 天
        for i in range(6):
            rows.append({"stock_id": "1001", "date": date(2026, 3, 1) + timedelta(days=i), "close": 105.0})
        df_price = pd.DataFrame(rows)
        stats = _compute_stats_from_dfs(df_rec, df_price, holding_days=5)
        assert stats.sample_count == 1  # 只有 1001 計入

    def test_skip_zero_or_negative_entry_price(self) -> None:
        df_rec = self._make_records(2, date(2026, 3, 1))
        df_rec.loc[0, "close"] = 0.0  # 異常價
        df_price = self._make_prices(df_rec["stock_id"].tolist(), date(2026, 3, 1), [0.05, 0.05])
        stats = _compute_stats_from_dfs(df_rec, df_price, holding_days=5)
        assert stats.sample_count == 1  # 1000 被跳過


# ──────────────────────────────────────────────────────────
#  validate_regime_blocks 整合測試
# ──────────────────────────────────────────────────────────


class TestValidateRegimeBlocks:
    """整合測試：使用 in-memory DB + 真實 DiscoveryRecord 驗證 validator 流程。"""

    def test_empty_db_all_insufficient_data(self, db_session) -> None:
        """DB 無資料 → 所有格子標 insufficient_data，無 actionable。"""
        validations = validate_regime_blocks(holding_days=5, lookback_days=90, min_samples=30)
        assert all(v.recommendation == "insufficient_data" for v in validations)
        assert len({v.regime for v in validations}) == 4  # 4 regimes
        assert len({v.mode for v in validations}) == 5  # 5 modes
        assert len(validations) == 20  # 4 × 5

    def test_actionable_extracted_from_results(self) -> None:
        """format_validation_report 應正確分類 actionable / keep / insufficient。"""
        validations = [
            BlockValidation(
                regime="bull",
                mode="momentum",
                is_blocked=False,
                stats=HoldingStats(0.05, 0.60, 100),
                recommendation="keep",
            ),
            BlockValidation(
                regime="sideways",
                mode="momentum",
                is_blocked=False,
                stats=HoldingStats(-0.05, 0.20, 50),
                recommendation="add_block",
            ),
            BlockValidation(
                regime="crisis",
                mode="dividend",
                is_blocked=False,
                stats=HoldingStats(0.0, 0.0, 5),
                recommendation="insufficient_data",
            ),
        ]
        report = format_validation_report(validations)
        assert "建議調整: 1" in report
        assert "維持現狀: 1" in report
        assert "樣本不足: 1" in report
        assert "新增封鎖" in report  # actionable section
        assert "sideways" in report
        assert "momentum" in report

    def test_format_report_no_actionable_section_when_clean(self) -> None:
        """全部 keep 時不顯示「建議調整」段（避免 noise）。"""
        validations = [
            BlockValidation(
                regime="bull",
                mode="momentum",
                is_blocked=False,
                stats=HoldingStats(0.05, 0.60, 100),
                recommendation="keep",
            ),
        ]
        report = format_validation_report(validations)
        assert "建議調整（請人工 review" not in report
        assert "維持現狀" in report

    def test_current_block_matrix_has_no_pending_actionable(self, db_session) -> None:
        """Regression test：當前空 DB 下 validator 不會誤觸發。

        實際資料下的驗證請於 morning-routine 月度執行。
        """
        validations = validate_regime_blocks(holding_days=5, lookback_days=90, min_samples=30)
        actionable = [v for v in validations if v.recommendation in ("lift_block", "add_block")]
        assert actionable == [], f"空 DB 不該產生 actionable：{actionable}"
