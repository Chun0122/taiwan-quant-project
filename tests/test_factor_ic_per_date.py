"""C4 修復回歸測試 — compute_factor_ic 用 per-date cross-sectional IC。

對應 audit 2026-05-09 P0-C4：
- 原實作對全樣本 pool 做 spearman corr，當 factor 跨日均值漂移時（如 momentum 在
  bull/bear 不同 mean），time-series 訊號會混進 IC，造成 momentum IC 高估 0.05~0.10
- 修復：per-date 平均（IC = mean_t(spearman(factor_t, ret_{t→t+h}))），
  per-date 稀疏時 fallback pooled（保留向後相容）

關鍵測試：構造「跨日均值漂移、日內隨機」場景，pooled IC 高估、per-date IC ≈ 0。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.discovery.scanner._functions import compute_factor_ic

# ─────────────────────────────────────────────────────────────────
#  C4-A: per-date IC 為主路徑（每日 ≥ 3 檔、有效日期 ≥ 3 天）
# ─────────────────────────────────────────────────────────────────


class TestComputeFactorIcPerDate:
    @staticmethod
    def _build(records: list[dict], days_to_exit: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
        """records: [{scan_date, stock_id, close, scores..., forward_return}, ...]
        產生 df_records + df_prices（含 forward_return 對應的 exit close）。
        """
        df_rec = pd.DataFrame([{k: v for k, v in r.items() if k != "forward_return"} for r in records])
        prices: list[dict] = []
        for r in records:
            sid = r["stock_id"]
            scan_d = r["scan_date"]
            entry = float(r["close"])
            exit_close = entry * (1 + r["forward_return"])
            for d in range(1, days_to_exit + 2):
                prices.append({"stock_id": sid, "date": scan_d + timedelta(days=d), "close": exit_close})
        return df_rec, pd.DataFrame(prices)

    def test_per_date_ic_filters_cross_day_drift(self):
        """跨日均值漂移情境：pooled IC 估高、per-date IC 接近 0（C4 核心驗證）。

        構造：
          Day 1：technical_score 平均低（0.3），ret 平均低（0.5%）
          Day 2：technical_score 平均中（0.5），ret 平均中（2.5%）
          Day 3：technical_score 平均高（0.7），ret 平均高（4.5%）
          每日內 4 檔，technical_score 與 ret **無相關**（隨機）

        標準 IC（per-date）= mean(per-date Spearman) ≈ 0
        Pooled corr 包含 time-series 漂移 → 偏高（~0.8+）
        """
        ref = date.today()
        rng = np.random.default_rng(42)

        records = []
        # 每日 4 檔，scores 在當日 mean 附近隨機，returns 在當日 mean 附近隨機
        # （兩組隨機獨立 → 日內 IC 接近 0）
        for day_idx, (mean_score, mean_ret) in enumerate([(0.3, 0.005), (0.5, 0.025), (0.7, 0.045)]):
            scan_d = ref - timedelta(days=20 + day_idx)
            for j in range(4):
                # 用「不同隨機種子」的 score 與 ret，確保日內無 corr
                score = mean_score + rng.uniform(-0.1, 0.1)
                ret = mean_ret + rng.uniform(-0.005, 0.005)
                records.append(
                    {
                        "scan_date": scan_d,
                        "stock_id": f"S{day_idx}{j}",
                        "close": 100.0,
                        "technical_score": score,
                        "chip_score": 0.5,
                        "fundamental_score": 0.5,
                        "news_score": 0.5,
                        "forward_return": ret,
                    }
                )

        df_rec, df_price = self._build(records)
        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)

        tech_row = result[result["factor"] == "technical_score"]
        assert len(tech_row) == 1
        ic = tech_row.iloc[0]["ic"]

        # 修復後：per-date IC 應接近 0（每日內無預測力）
        # 修復前：pooled IC 會 > 0.7（含跨日漂移）
        assert abs(ic) < 0.5, f"per-date IC should suppress drift, got {ic}"

    def test_per_date_ic_preserves_real_predictive_signal(self):
        """日內真實 IC 訊號應該被保留：日內 factor 與 ret 正相關 → IC > 0。"""
        ref = date.today()
        records = []
        # 3 天 × 5 檔，每日內 factor 與 ret 完全相關
        for day_idx in range(3):
            scan_d = ref - timedelta(days=20 + day_idx)
            for j in range(5):
                score = 0.2 + j * 0.1  # 0.2 ~ 0.6
                ret = 0.01 * j  # 0 ~ 0.04（每日 rank 與 score rank 相同）
                records.append(
                    {
                        "scan_date": scan_d,
                        "stock_id": f"S{day_idx}{j}",
                        "close": 100.0,
                        "technical_score": score,
                        "chip_score": 0.5,
                        "fundamental_score": 0.5,
                        "news_score": 0.5,
                        "forward_return": ret,
                    }
                )

        df_rec, df_price = self._build(records)
        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)

        tech_row = result[result["factor"] == "technical_score"]
        assert len(tech_row) == 1
        ic = tech_row.iloc[0]["ic"]
        # 完美 rank 相關 → spearman = 1.0（mean of three 1.0s）
        assert ic > 0.95

    def test_evaluable_count_reflects_used_samples(self):
        """evaluable_count 反映 per-date 路徑使用的樣本總數（過濾 std=0 的日期）。"""
        ref = date.today()
        records = []
        for day_idx in range(3):
            scan_d = ref - timedelta(days=20 + day_idx)
            for j in range(4):
                records.append(
                    {
                        "scan_date": scan_d,
                        "stock_id": f"S{day_idx}{j}",
                        "close": 100.0,
                        "technical_score": 0.3 + j * 0.1,
                        "chip_score": 0.5,
                        "fundamental_score": 0.5,
                        "news_score": 0.5,
                        "forward_return": 0.01 * j,
                    }
                )

        df_rec, df_price = self._build(records)
        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)
        tech = result[result["factor"] == "technical_score"].iloc[0]
        # 3 dates × 4 stocks = 12 樣本，每日都有 std > 0 → 全部納入
        assert tech["evaluable_count"] == 12


# ─────────────────────────────────────────────────────────────────
#  C4-B: per-date 稀疏時 fallback pooled（向後相容）
# ─────────────────────────────────────────────────────────────────


class TestComputeFactorIcFallbackPooled:
    def test_single_stock_per_date_falls_back_to_pooled(self):
        """每日只有 1 檔（< _PER_DATE_MIN_STOCKS=3）→ fallback pooled corr。

        既有合成測試常用此 pattern；確保不破壞既有測試。
        """
        from datetime import date as _date

        ref = _date.today()
        records = []
        # 12 dates × 1 stock per date，跨日 score 與 ret 線性相關
        for i in range(12):
            scan_d = ref - timedelta(days=20 - i)
            score = 0.3 + i * 0.05
            ret = 0.005 * i
            records.append(
                {
                    "scan_date": scan_d,
                    "stock_id": f"X{i:02d}",
                    "close": 100.0,
                    "technical_score": score,
                    "chip_score": 0.5,
                    "fundamental_score": 0.5,
                    "news_score": 0.5,
                    "forward_return": ret,
                }
            )

        df_rec = pd.DataFrame([{k: v for k, v in r.items() if k != "forward_return"} for r in records])
        prices = []
        for r in records:
            entry = 100.0
            exit_close = entry * (1 + r["forward_return"])
            for d in range(1, 7):
                prices.append(
                    {"stock_id": r["stock_id"], "date": r["scan_date"] + timedelta(days=d), "close": exit_close}
                )
        df_price = pd.DataFrame(prices)

        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)
        tech_row = result[result["factor"] == "technical_score"]
        assert len(tech_row) == 1
        # Fallback pooled：score 與 ret 完美 rank 相關 → IC ≈ 1
        assert tech_row.iloc[0]["ic"] > 0.95
        assert tech_row.iloc[0]["evaluable_count"] == 12

    def test_two_dates_falls_back_to_pooled(self):
        """有效日期 < 3 天（_PER_DATE_MIN_DATES=3）→ fallback。"""
        from datetime import date as _date

        ref = _date.today()
        records = []
        for day_idx in range(2):  # 只 2 天
            scan_d = ref - timedelta(days=20 + day_idx)
            for j in range(5):
                records.append(
                    {
                        "scan_date": scan_d,
                        "stock_id": f"S{day_idx}{j}",
                        "close": 100.0,
                        "technical_score": 0.3 + j * 0.1,
                        "chip_score": 0.5,
                        "fundamental_score": 0.5,
                        "news_score": 0.5,
                        "forward_return": 0.01 * j,
                    }
                )

        df_rec = pd.DataFrame([{k: v for k, v in r.items() if k != "forward_return"} for r in records])
        prices = []
        for r in records:
            entry = 100.0
            exit_close = entry * (1 + r["forward_return"])
            for d in range(1, 7):
                prices.append(
                    {"stock_id": r["stock_id"], "date": r["scan_date"] + timedelta(days=d), "close": exit_close}
                )
        df_price = pd.DataFrame(prices)

        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)
        # 兩天都 std>0 → 但只 2 天 < 3 → fallback pooled
        # pooled 模式下完美單調 → IC ≈ 1
        tech = result[result["factor"] == "technical_score"]
        if not tech.empty:
            assert tech.iloc[0]["ic"] > 0.9
            # fallback 路徑 evaluable_count = len(valid) = 10
            assert tech.iloc[0]["evaluable_count"] == 10


# ─────────────────────────────────────────────────────────────────
#  C4-C: 邊界條件
# ─────────────────────────────────────────────────────────────────


class TestComputeFactorIcEdgeCases:
    def test_zero_variance_per_date_skipped_then_fallback(self):
        """每日內 std=0 全跳過 → 0 個 per-date IC → fallback；fallback 也 std=0 → factor 完全跳過。"""
        from datetime import date as _date

        ref = _date.today()
        records = []
        # 全部 score 與 ret 都相同 → std=0
        for day_idx in range(3):
            scan_d = ref - timedelta(days=20 + day_idx)
            for j in range(4):
                records.append(
                    {
                        "scan_date": scan_d,
                        "stock_id": f"S{day_idx}{j}",
                        "close": 100.0,
                        "technical_score": 0.5,
                        "chip_score": 0.5,
                        "fundamental_score": 0.5,
                        "news_score": 0.5,
                        "forward_return": 0.02,
                    }
                )

        df_rec = pd.DataFrame([{k: v for k, v in r.items() if k != "forward_return"} for r in records])
        prices = []
        for r in records:
            for d in range(1, 7):
                prices.append({"stock_id": r["stock_id"], "date": r["scan_date"] + timedelta(days=d), "close": 102.0})
        df_price = pd.DataFrame(prices)

        result = compute_factor_ic(df_rec, df_price, holding_days=5, lookback_days=30)
        # 所有因子 std=0 → 全部不應出現在結果中
        assert result.empty or "technical_score" not in result["factor"].values
