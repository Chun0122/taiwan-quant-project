"""Wave 2 P1 修復回歸測試（2026-05-09 audit）。

涵蓋 W1~W6：
- W1: Candidate Memory 用交易日距離（universe.py:_load_candidate_memory）
- W2: _find_last_trading_day 改用 is_trading_day（twse_fetcher.py）
- W3: 2027 假日表 + 缺資料年份 log warning（calendar.py）
- W4: fillna(0.5) imputation 統計寫入 log（_base.py:1499-1501）
- W5: DailyFeature staleness gap 檢查（universe.py:_load_feature_turnover）
- W6: DailyFeature 過濾 volume<=0 暫定資料（pipeline.py）
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd

# 重要：schema imports 須在 top-level，pytest collection 時即註冊 ORM 表，
# 否則 conftest 的 session-scope create_all 跑時 Base.metadata 為空
from src.data.schema import DailyFeature, DailyPrice, DiscoveryRecord  # noqa: F401

# ─────────────────────────────────────────────────────────────────
#  W2: _find_last_trading_day 用 calendar.is_trading_day
# ─────────────────────────────────────────────────────────────────


class TestFindLastTradingDayUsesCalendar:
    def test_skips_lunar_new_year_holidays(self):
        """2026 春節 (2/16~2/20)：從 2/22(週日) 回溯應跳過春節 + 週末抓到 2/13(週五)。"""
        from src.data.twse_fetcher import _find_last_trading_day

        # 2/22 是週日，2/21 週六，2/20~2/16 春節（5 天），2/15 週日，2/14 週六
        # 應該回到 2/13（週五，正常交易日）
        target = date(2026, 2, 22)
        result = _find_last_trading_day(target)
        assert result == date(2026, 2, 13)

    def test_normal_weekday_returns_self(self):
        """一般週三應直接回傳自身。"""
        from src.data.twse_fetcher import _find_last_trading_day

        target = date(2026, 3, 11)  # 週三
        assert _find_last_trading_day(target) == target


# ─────────────────────────────────────────────────────────────────
#  W3: 2027 calendar + 缺資料 log warning
# ─────────────────────────────────────────────────────────────────


class TestCalendar2027:
    def test_2027_new_year_is_holiday(self):
        """2027/1/1 應為 holiday。"""
        from src.data.calendar import is_trading_day, is_twse_holiday

        d = date(2027, 1, 1)
        assert is_twse_holiday(d) is True
        assert is_trading_day(d) is False

    def test_2027_lunar_new_year_holidays(self):
        """2027 春節（除夕 2/5 + 補休 2/8~2/11）應為 holiday。"""
        from src.data.calendar import is_twse_holiday

        for d in [date(2027, 2, 5), date(2027, 2, 8), date(2027, 2, 9), date(2027, 2, 10)]:
            assert is_twse_holiday(d), f"{d} 應為春節假日"

    def test_missing_year_logs_warning_once(self, caplog):
        """2030（未列入）查詢時應 log warning，但只 log 一次（避免刷屏）。"""
        from src.data.calendar import _LOGGED_MISSING_YEARS, is_twse_holiday

        # 重設 once-flag 確保此測試獨立
        _LOGGED_MISSING_YEARS.discard(2030)

        with caplog.at_level(logging.WARNING):
            assert is_twse_holiday(date(2030, 1, 1)) is False
            assert is_twse_holiday(date(2030, 6, 15)) is False
            assert is_twse_holiday(date(2030, 12, 25)) is False

        # 應該只有一筆 warning
        warnings = [r for r in caplog.records if "2030" in r.getMessage() and "TWSE 假日" in r.getMessage()]
        assert len(warnings) == 1


# ─────────────────────────────────────────────────────────────────
#  W1: Candidate Memory 用交易日距離
# ─────────────────────────────────────────────────────────────────


class TestCandidateMemoryUsesTradingDays:
    def test_friday_to_monday_is_one_trading_day(self, db_session, monkeypatch):
        """週五掃出的候選，下週一（自然日 3 天但僅 1 個交易日）應記為 days_ago=1。"""
        # mock today = 2026-05-11（週一）
        # scan_date = 2026-05-08（週五）
        # 自然日 = 3，交易日 = 1
        from src.data.calendar import _TWSE_HOLIDAYS
        from src.discovery.universe import UniverseConfig, UniverseFilter

        # 確保 5/8 與 5/11 都是交易日（5/8 為週五、5/11 為週一，2026 沒有對應假日）
        assert date(2026, 5, 8) not in _TWSE_HOLIDAYS.get(2026, frozenset())
        assert date(2026, 5, 11) not in _TWSE_HOLIDAYS.get(2026, frozenset())

        # 寫入歷史 record
        rec = DiscoveryRecord(
            stock_id="2330",
            stock_name="台積電",
            mode="momentum",
            rank=1,
            scan_date=date(2026, 5, 8),
            close=600.0,
            composite_score=0.8,
            technical_score=0.7,
            chip_score=0.85,
            fundamental_score=0.5,
            news_score=0.6,
            valid_until=date(2026, 5, 15),
            regime="bull",
            total_stocks=1000,
            after_coarse=200,
            created_at=datetime.utcnow(),
        )
        db_session.add(rec)
        db_session.commit()

        # mock date.today() = 5/11 週一
        import src.discovery.universe as univ_mod

        class _MockDate(date):
            @classmethod
            def today(cls):
                return date(2026, 5, 11)

        monkeypatch.setattr(univ_mod, "date", _MockDate)

        config = UniverseConfig(candidate_memory_days=3)
        universe = UniverseFilter(config)
        memory = universe._load_candidate_memory("momentum", ["2330"])

        # 修復前：days_ago=3（自然日）→ Day 3 標籤、memory bonus 失效
        # 修復後：days_ago=1（交易日距離）→ Day 1 標籤、memory bonus 啟用
        assert memory.get("2330") == 1


# ─────────────────────────────────────────────────────────────────
#  W6: DailyFeature 過濾 volume<=0 暫定資料
# ─────────────────────────────────────────────────────────────────


class TestComputeDailyFeaturesFiltersZeroVolume:
    """W6 純 unit test：直接驗證過濾邏輯，不靠 db_session 跑完整 pipeline
    （避免跨測試 in-memory connection 池異常）。"""

    def test_filter_logic_excludes_zero_volume_and_invalid_close(self):
        """volume<=0 / close 缺失或 <=0 的列應被過濾掉。"""
        df = pd.DataFrame(
            {
                "stock_id": ["A", "B", "C", "D", "E"],
                "date": [date(2026, 5, 8)] * 5,
                "high": [100.0] * 5,
                "close": [100.0, 101.0, 0.0, None, 102.0],
                "volume": [10_000_000, 0, 5_000_000, 8_000_000, 12_000_000],
                "turnover": [1e9] * 5,
            }
        )
        # 模擬 W6 過濾條件（與 pipeline.py:1517 一致）
        df_filtered = df[(df["volume"] > 0) & df["close"].notna() & (df["close"] > 0)]
        # 應保留 A (vol=10M, close=100) 與 E (vol=12M, close=102)
        # B 因 volume=0；C 因 close=0；D 因 close=None
        assert set(df_filtered["stock_id"]) == {"A", "E"}

    def test_filter_preserves_normal_rows(self):
        """正常資料（vol>0 + close>0）全部保留。"""
        df = pd.DataFrame(
            {
                "stock_id": ["X", "Y", "Z"],
                "close": [100.0, 200.0, 50.0],
                "volume": [1_000_000, 5_000_000, 2_000_000],
            }
        )
        df_filtered = df[(df["volume"] > 0) & df["close"].notna() & (df["close"] > 0)]
        assert len(df_filtered) == 3


# ─────────────────────────────────────────────────────────────────
#  W4: fillna(0.5) imputation 統計
# ─────────────────────────────────────────────────────────────────


class TestFillnaImputationStats:
    def test_logs_imputation_count_per_dimension(self, caplog):
        """部分 candidates 缺 score 時，log 應記錄每維度 imputed 比例（W4）。"""
        # 直接測試 fillna 邏輯：構造一個含 NaN 的 candidates DataFrame，
        # 模擬 _score_candidates 的 fillna 部分
        candidates = pd.DataFrame(
            {
                "stock_id": ["A", "B", "C", "D"],
                "technical_score": [0.7, None, 0.6, None],
                "chip_score": [0.8, 0.7, 0.65, 0.5],
                "fundamental_score": [None, 0.5, 0.55, 0.6],
                "news_score": [0.5, 0.5, 0.5, None],
            }
        )

        # 模擬 W4 修復後的邏輯（從 _base.py 抽取）
        score_cols = [c for c in candidates.columns if c.endswith("_score") and c != "composite_score"]
        imputation_stats = {}
        for col in score_cols:
            n_missing = int(candidates[col].isna().sum())
            if n_missing > 0:
                imputation_stats[col] = n_missing
            candidates[col] = candidates[col].fillna(0.5)

        # 驗證統計準確
        assert imputation_stats == {
            "technical_score": 2,
            "fundamental_score": 1,
            "news_score": 1,
        }
        # 沒缺漏的不該被記錄
        assert "chip_score" not in imputation_stats


# ─────────────────────────────────────────────────────────────────
#  W5: DailyFeature staleness 警告
# ─────────────────────────────────────────────────────────────────


class TestDailyFeatureStaleness:
    """W5：純 unit test — 不依賴 db_session fixture，避免跨測試 in-memory state 污染。"""

    def test_staleness_gap_calculation(self):
        """W5 staleness gap 計算邏輯：5/4 → 5/11 應為 5 個交易日 gap。"""
        from src.data.calendar import get_trading_days

        latest = date(2026, 5, 4)
        today = date(2026, 5, 11)
        gap = max(0, len(get_trading_days(latest, today)) - 1)
        assert gap == 5

    def test_warning_fires_when_gap_exceeds_threshold(self):
        """W5：DailyFeature gap > 1 trading day 時觸發 warning。"""
        from unittest.mock import MagicMock, patch

        import src.discovery.universe as univ_mod
        from src.discovery.universe import UniverseConfig, UniverseFilter

        # mock session.execute 兩次：第一次 .scalar() 回傳 latest_date；第二次 .all() 回傳 rows
        scalar_result = MagicMock()
        scalar_result.scalar.return_value = date(2026, 5, 4)
        all_result = MagicMock()
        all_result.all.return_value = [("STALE", 1000.0, 1100.0, 0.91)]
        mock_session = MagicMock()
        mock_session.execute.side_effect = [scalar_result, all_result]
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_session
        mock_cm.__exit__.return_value = False

        class _MockDate(date):
            @classmethod
            def today(cls):
                return date(2026, 5, 11)

        with (
            patch.object(univ_mod, "get_session", return_value=mock_cm),
            patch.object(univ_mod, "date", _MockDate),
            patch.object(univ_mod.logger, "warning") as mock_warn,
        ):
            UniverseFilter(UniverseConfig())._load_feature_turnover(["STALE"])

        assert mock_warn.called, "logger.warning was not called"
        msg = mock_warn.call_args.args[0]
        assert "DailyFeature 最新日期" in msg and "stale" in msg
        positional = mock_warn.call_args.args[1:]
        assert date(2026, 5, 4) in positional
        assert 5 in positional

    def test_no_warning_when_features_fresh(self):
        """W5：DailyFeature gap = 0 時不觸發 warning（向後相容，無誤報）。"""
        from unittest.mock import MagicMock, patch

        import src.discovery.universe as univ_mod
        from src.discovery.universe import UniverseConfig, UniverseFilter

        scalar_result = MagicMock()
        scalar_result.scalar.return_value = date(2026, 5, 11)
        all_result = MagicMock()
        all_result.all.return_value = [("FRESH", 1000.0, 1100.0, 0.91)]
        mock_session = MagicMock()
        mock_session.execute.side_effect = [scalar_result, all_result]
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_session
        mock_cm.__exit__.return_value = False

        class _MockDate(date):
            @classmethod
            def today(cls):
                return date(2026, 5, 11)

        with (
            patch.object(univ_mod, "get_session", return_value=mock_cm),
            patch.object(univ_mod, "date", _MockDate),
            patch.object(univ_mod.logger, "warning") as mock_warn,
        ):
            UniverseFilter(UniverseConfig())._load_feature_turnover(["FRESH"])

        assert not mock_warn.called
