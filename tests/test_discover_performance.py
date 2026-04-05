"""測試 Discover 推薦績效回測 — DiscoveryPerformance + 策略衰減監控。"""

from datetime import date

import pandas as pd

from src.data.schema import DailyPrice, DiscoveryRecord
from src.discovery.performance import DiscoveryPerformance, compute_strategy_decay


def _insert_discovery(session, scan_date, mode, rank, stock_id, close, score=0.8, name=None, regime=None):
    """輔助：插入一筆 DiscoveryRecord。"""
    session.add(
        DiscoveryRecord(
            scan_date=scan_date,
            mode=mode,
            rank=rank,
            stock_id=stock_id,
            stock_name=name or stock_id,
            close=close,
            composite_score=score,
            regime=regime,
        )
    )
    session.flush()


def _insert_prices(session, stock_id, start_date, prices):
    """輔助：插入多日 DailyPrice。prices 為收盤價列表。"""
    dates = pd.bdate_range(start_date, periods=len(prices))
    for dt, close in zip(dates, prices):
        session.add(
            DailyPrice(
                stock_id=stock_id,
                date=dt.date(),
                open=close,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=1_000_000,
                turnover=100_000_000,
                spread=0.5,
            )
        )
    session.flush()


class TestBasicReturn:
    """基本報酬率計算。"""

    def test_single_recommendation(self, db_session):
        """插入 1 筆推薦 + 後續 20 天 DailyPrice，驗證 5/10/20 天報酬率。"""
        scan_date = date(2025, 6, 2)
        _insert_discovery(db_session, scan_date, "momentum", 1, "2330", 100.0)

        # 推薦日後 20 個交易日的價格（從 6/3 開始）
        prices = [
            101,
            102,
            103,
            104,
            105,  # day 1-5
            106,
            107,
            108,
            109,
            110,  # day 6-10
            111,
            112,
            113,
            114,
            115,  # day 11-15
            116,
            117,
            118,
            119,
            120,
        ]  # day 16-20
        _insert_prices(db_session, "2330", "2025-06-03", prices)

        perf = DiscoveryPerformance(mode="momentum", holding_days=[5, 10, 20])
        result = perf.evaluate()

        detail = result["detail"]
        assert len(detail) == 1

        row = detail.iloc[0]
        # 5天報酬: (105 - 100) / 100 = 5%
        assert abs(row["return_5d"] - 0.05) < 1e-10
        # 10天報酬: (110 - 100) / 100 = 10%
        assert abs(row["return_10d"] - 0.10) < 1e-10
        # 20天報酬: (120 - 100) / 100 = 20%
        assert abs(row["return_20d"] - 0.20) < 1e-10

        # summary 驗證
        summary = result["summary"]
        assert len(summary) == 3
        for _, s_row in summary.iterrows():
            assert s_row["evaluable"] == 1
            assert s_row["win_rate"] == 1.0  # 全部正報酬


class TestInsufficientData:
    """資料不足處理。"""

    def test_partial_data(self, db_session):
        """推薦後只有 3 個交易日，5/10/20 天報酬率應為 NaN。"""
        scan_date = date(2025, 7, 1)
        _insert_discovery(db_session, scan_date, "momentum", 1, "2317", 80.0)

        # 只有 3 天價格
        _insert_prices(db_session, "2317", "2025-07-02", [81, 82, 83])

        perf = DiscoveryPerformance(mode="momentum", holding_days=[5, 10, 20])
        result = perf.evaluate()

        detail = result["detail"]
        assert len(detail) == 1

        row = detail.iloc[0]
        assert row["return_5d"] is None
        assert row["return_10d"] is None
        assert row["return_20d"] is None

        # summary 可評估數應為 0
        summary = result["summary"]
        for _, s_row in summary.iterrows():
            assert s_row["evaluable"] == 0


class TestMultipleRecommendations:
    """多筆推薦聚合。"""

    def test_win_rate_and_average(self, db_session):
        """3 筆推薦（2 賺 1 賠），驗證勝率和平均報酬。"""
        scan_date = date(2025, 8, 1)
        _insert_discovery(db_session, scan_date, "swing", 1, "A001", 100.0)
        _insert_discovery(db_session, scan_date, "swing", 2, "A002", 200.0)
        _insert_discovery(db_session, scan_date, "swing", 3, "A003", 50.0)

        # A001: 100 -> 110 (+10%)
        _insert_prices(db_session, "A001", "2025-08-04", [102, 104, 106, 108, 110])
        # A002: 200 -> 220 (+10%)
        _insert_prices(db_session, "A002", "2025-08-04", [204, 208, 212, 216, 220])
        # A003: 50 -> 45 (-10%)
        _insert_prices(db_session, "A003", "2025-08-04", [49, 48, 47, 46, 45])

        perf = DiscoveryPerformance(mode="swing", holding_days=[5])
        result = perf.evaluate()

        summary = result["summary"]
        assert len(summary) == 1

        row = summary.iloc[0]
        assert row["evaluable"] == 3
        # 勝率 = 2/3
        assert abs(row["win_rate"] - 2 / 3) < 1e-10
        # 平均報酬 = (0.1 + 0.1 + (-0.1)) / 3
        expected_avg = (0.1 + 0.1 + (-0.1)) / 3
        assert abs(row["avg_return"] - expected_avg) < 1e-10
        # 最大獲利 = 10%
        assert abs(row["max_gain"] - 0.1) < 1e-10
        # 最大虧損 = -10%
        assert abs(row["max_loss"] - (-0.1)) < 1e-10


class TestTopNFilter:
    """top_n 篩選。"""

    def test_top_n_only_includes_top_ranks(self, db_session):
        """推薦 rank 1~10，top_n=3 只計算前 3 名。"""
        scan_date = date(2025, 9, 1)
        for i in range(1, 11):
            _insert_discovery(
                db_session,
                scan_date,
                "momentum",
                i,
                f"B{i:03d}",
                100.0 + i,
            )

        # 為所有股票插入 5 天價格
        for i in range(1, 11):
            _insert_prices(
                db_session,
                f"B{i:03d}",
                "2025-09-02",
                [101 + i, 102 + i, 103 + i, 104 + i, 105 + i],
            )

        perf = DiscoveryPerformance(mode="momentum", holding_days=[5], top_n=3)
        result = perf.evaluate()

        detail = result["detail"]
        assert len(detail) == 3
        # 確認只有 rank 1, 2, 3
        assert set(detail["rank"]) == {1, 2, 3}


class TestDateRangeFilter:
    """日期範圍篩選。"""

    def test_start_end_filter(self, db_session):
        """3 個掃描日，start/end 篩選出正確子集。"""
        dates = [date(2025, 6, 1), date(2025, 7, 1), date(2025, 8, 1)]
        for d in dates:
            _insert_discovery(db_session, d, "value", 1, "C001", 100.0)

        # 插入後續價格（只需 5 天）
        for d in dates:
            next_day = pd.bdate_range(d, periods=2)[-1]  # 下一個交易日
            _insert_prices(
                db_session,
                "C001",
                str(next_day.date()),
                [101, 102, 103, 104, 105],
            )

        # 只取 6 月和 7 月的推薦
        perf = DiscoveryPerformance(
            mode="value",
            holding_days=[5],
            start_date="2025-06-01",
            end_date="2025-07-31",
        )
        result = perf.evaluate()

        detail = result["detail"]
        assert len(detail) == 2
        scan_dates = set(detail["scan_date"])
        assert date(2025, 6, 1) in scan_dates
        assert date(2025, 7, 1) in scan_dates
        assert date(2025, 8, 1) not in scan_dates


class TestSwingModeHorizons:
    """swing 模式預設持有期 [20, 40, 60] 測試。"""

    def test_swing_default_holding_days(self):
        """swing 模式未指定 holding_days 時，應使用 [20, 40, 60]。"""
        perf = DiscoveryPerformance(mode="swing")
        assert perf.holding_days == [20, 40, 60]

    def test_momentum_default_holding_days(self):
        """momentum 模式未指定 holding_days 時，應使用預設 [5, 10, 20]。"""
        perf = DiscoveryPerformance(mode="momentum")
        assert perf.holding_days == [5, 10, 20]

    def test_explicit_holding_days_override(self):
        """明確傳入 holding_days 時，應覆蓋模式預設值。"""
        perf = DiscoveryPerformance(mode="swing", holding_days=[5, 10])
        assert perf.holding_days == [5, 10]

    def test_swing_summary_columns_20_40_60(self, db_session):
        """swing 模式績效回測摘要的 holding_days 欄位應包含 20, 40, 60。"""
        # 插入 1 筆推薦
        db_session.add(
            DiscoveryRecord(
                scan_date=date(2025, 1, 2),
                mode="swing",
                rank=1,
                stock_id="D001",
                stock_name="D001",
                close=100.0,
                composite_score=0.8,
            )
        )
        db_session.flush()
        # 插入 60 天後的價格
        dates = pd.bdate_range("2025-01-03", periods=65)
        for i, dt in enumerate(dates):
            db_session.add(
                DailyPrice(
                    stock_id="D001",
                    date=dt.date(),
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.0 + i,
                    volume=500_000,
                    turnover=50_000_000,
                    spread=0.5,
                )
            )
        db_session.flush()

        perf = DiscoveryPerformance(mode="swing")
        result = perf.evaluate()
        if not result["summary"].empty:
            holding_days_in_summary = set(result["summary"]["holding_days"].tolist())
            assert 20 in holding_days_in_summary
            assert 40 in holding_days_in_summary
            assert 60 in holding_days_in_summary


# ─── 策略衰減監控（純函數測試，不需 DB） ─────────────────────


class TestComputeStrategyDecay:
    """測試 compute_strategy_decay() 純函數。"""

    def _make_detail(self, scan_dates, returns_10d):
        """建構 detail DataFrame。"""
        return pd.DataFrame(
            {
                "scan_date": scan_dates,
                "stock_id": [f"S{i:03d}" for i in range(len(scan_dates))],
                "return_10d": returns_10d,
            }
        )

    def test_empty_df_returns_no_decay(self):
        """空 DataFrame → is_decaying=False, 所有指標 None。"""
        result = compute_strategy_decay(pd.DataFrame(), holding_days=10)
        assert result["is_decaying"] is False
        assert result["recent_count"] == 0
        assert result["recent_win_rate"] is None

    def test_missing_column_returns_empty(self):
        """缺少 return_10d 欄位 → 安全回傳。"""
        df = pd.DataFrame({"scan_date": [date(2025, 3, 1)], "return_5d": [0.05]})
        result = compute_strategy_decay(df, holding_days=10)
        assert result["recent_count"] == 0

    def test_all_positive_no_decay(self):
        """全正報酬 → 不衰減。"""
        ref = date(2025, 3, 28)
        # 所有資料都在近 30 天內
        dates = [date(2025, 3, d) for d in range(1, 11)]
        returns = [0.05, 0.03, 0.02, 0.04, 0.06, 0.01, 0.03, 0.02, 0.05, 0.04]
        df = self._make_detail(dates, returns)
        result = compute_strategy_decay(df, recent_days=30, holding_days=10, reference_date=ref)
        assert result["is_decaying"] is False
        assert result["recent_win_rate"] == 1.0
        assert result["recent_count"] == 10

    def test_low_win_rate_triggers_decay(self):
        """近期勝率 < 40% → 觸發衰減警告。"""
        ref = date(2025, 3, 28)
        dates = [date(2025, 3, d) for d in range(1, 11)]
        # 3 wins, 7 losses → 30% win rate
        returns = [0.05, -0.03, -0.02, -0.04, 0.02, -0.01, -0.03, -0.02, 0.01, -0.04]
        df = self._make_detail(dates, returns)
        result = compute_strategy_decay(df, recent_days=30, holding_days=10, reference_date=ref)
        assert result["is_decaying"] is True
        assert result["warning"] is not None
        assert "勝率" in result["warning"]

    def test_negative_avg_return_triggers_decay(self):
        """近期平均報酬 < 0 → 觸發衰減。"""
        ref = date(2025, 3, 28)
        dates = [date(2025, 3, d) for d in range(1, 6)]
        returns = [0.02, -0.10, 0.01, -0.08, 0.03]  # avg < 0
        df = self._make_detail(dates, returns)
        result = compute_strategy_decay(df, recent_days=30, holding_days=10, reference_date=ref)
        assert result["is_decaying"] is True
        assert "均報酬" in result["warning"]

    def test_recent_vs_historical_split(self):
        """驗證近期/歷史分割正確。"""
        ref = date(2025, 3, 28)
        # 歷史資料（60 天前）
        hist_dates = [date(2025, 1, d) for d in range(10, 20)]
        hist_returns = [0.08] * 10  # 歷史很好
        # 近期資料
        recent_dates = [date(2025, 3, d) for d in range(1, 11)]
        recent_returns = [0.01] * 10  # 近期較弱但還正

        df = self._make_detail(hist_dates + recent_dates, hist_returns + recent_returns)
        result = compute_strategy_decay(df, recent_days=30, holding_days=10, reference_date=ref)

        assert result["recent_count"] == 10
        assert result["historical_count"] == 10
        assert result["recent_avg_return"] < result["historical_avg_return"]
        # 近期雖弱但勝率 100% 且均報酬 > 0 → 不衰減
        assert result["is_decaying"] is False

    def test_win_rate_decay_calculation(self):
        """驗證 win_rate_decay 計算（pct points）。"""
        ref = date(2025, 3, 28)
        hist_dates = [date(2025, 1, d) for d in range(10, 20)]
        hist_returns = [0.05, 0.03, -0.01, 0.04, 0.02, 0.06, -0.02, 0.03, 0.01, 0.04]
        # 歷史勝率 = 8/10 = 80%

        recent_dates = [date(2025, 3, d) for d in range(1, 11)]
        recent_returns = [0.02, -0.03, -0.01, 0.01, -0.02, 0.03, -0.04, -0.01, 0.02, -0.03]
        # 近期勝率 = 4/10 = 40%

        df = self._make_detail(hist_dates + recent_dates, hist_returns + recent_returns)
        result = compute_strategy_decay(df, recent_days=30, holding_days=10, reference_date=ref)

        # win_rate_decay = (40% - 80%) * 100 = -40 pct points
        assert result["win_rate_decay"] == -40.0


# ─── Regime 分組績效 ──────────────────────────────────────


class TestByRegime:
    """Regime 分組績效統計。"""

    def test_regime_grouping(self, db_session):
        """不同 regime 的推薦應分別統計。"""
        scan_date = date(2025, 6, 2)
        _insert_discovery(db_session, scan_date, "momentum", 1, "R001", 100.0, regime="bull")
        _insert_discovery(db_session, scan_date, "momentum", 2, "R002", 100.0, regime="bull")
        _insert_discovery(db_session, scan_date, "momentum", 3, "R003", 100.0, regime="bear")

        # R001: +10%, R002: +5%, R003: -5%
        _insert_prices(db_session, "R001", "2025-06-03", [102, 104, 106, 108, 110])
        _insert_prices(db_session, "R002", "2025-06-03", [101, 102, 103, 104, 105])
        _insert_prices(db_session, "R003", "2025-06-03", [99, 98, 97, 96, 95])

        perf = DiscoveryPerformance(mode="momentum", holding_days=[5])
        result = perf.evaluate()

        by_regime = result["by_regime"]
        assert not by_regime.empty

        bull = by_regime[by_regime["regime"] == "bull"]
        bear = by_regime[by_regime["regime"] == "bear"]
        assert len(bull) == 1
        assert len(bear) == 1
        assert bull.iloc[0]["count"] == 2
        assert bull.iloc[0]["win_rate"] == 1.0
        assert bear.iloc[0]["count"] == 1
        assert bear.iloc[0]["win_rate"] == 0.0

    def test_no_regime_returns_empty(self, db_session):
        """所有 regime 為 None 時 by_regime 為空。"""
        scan_date = date(2025, 6, 2)
        _insert_discovery(db_session, scan_date, "momentum", 1, "NR01", 100.0, regime=None)
        _insert_prices(db_session, "NR01", "2025-06-03", [101, 102, 103, 104, 105])

        perf = DiscoveryPerformance(mode="momentum", holding_days=[5])
        result = perf.evaluate()
        assert result["by_regime"].empty


# ─── 換手率 ──────────────────────────────────────────────


class TestTurnover:
    """推薦清單換手率計算。"""

    def test_turnover_basic(self, db_session):
        """兩次掃描有部分重疊，驗證換手率。"""
        # 第一次掃描：A, B, C
        d1 = date(2025, 6, 2)
        for i, sid in enumerate(["T001", "T002", "T003"], 1):
            _insert_discovery(db_session, d1, "momentum", i, sid, 100.0)
            _insert_prices(db_session, sid, "2025-06-03", [101, 102, 103, 104, 105])

        # 第二次掃描：B, C, D（A 退出，D 新增）
        d2 = date(2025, 6, 9)
        for i, sid in enumerate(["T002", "T003", "T004"], 1):
            _insert_discovery(db_session, d2, "momentum", i, sid, 100.0)
        _insert_prices(db_session, "T004", "2025-06-10", [101, 102, 103, 104, 105])

        perf = DiscoveryPerformance(mode="momentum", holding_days=[5])
        result = perf.evaluate()

        turnover = result["turnover"]
        assert len(turnover) == 1
        row = turnover.iloc[0]
        assert row["overlap"] == 2  # B, C
        assert row["new_entries"] == 1  # D
        assert row["exits"] == 1  # A
        # turnover = 1 - 2/3 = 0.3333
        assert abs(row["turnover"] - (1 - 2 / 3)) < 0.01

    def test_single_scan_no_turnover(self, db_session):
        """只有一次掃描，turnover 為空。"""
        d1 = date(2025, 6, 2)
        _insert_discovery(db_session, d1, "momentum", 1, "TN01", 100.0)
        _insert_prices(db_session, "TN01", "2025-06-03", [101, 102, 103, 104, 105])

        perf = DiscoveryPerformance(mode="momentum", holding_days=[5])
        result = perf.evaluate()
        assert result["turnover"].empty

    def test_complete_turnover(self, db_session):
        """完全不重疊 → turnover = 1.0。"""
        d1 = date(2025, 6, 2)
        _insert_discovery(db_session, d1, "momentum", 1, "CT01", 100.0)
        _insert_prices(db_session, "CT01", "2025-06-03", [101, 102, 103, 104, 105])

        d2 = date(2025, 6, 9)
        _insert_discovery(db_session, d2, "momentum", 1, "CT02", 100.0)
        _insert_prices(db_session, "CT02", "2025-06-10", [101, 102, 103, 104, 105])

        perf = DiscoveryPerformance(mode="momentum", holding_days=[5])
        result = perf.evaluate()
        assert result["turnover"].iloc[0]["turnover"] == 1.0


# ─── MFE/MAE 整合 ────────────────────────────────────────


class TestMfeMae:
    """MFE/MAE 整合測試。"""

    def test_mfe_mae_computed(self, db_session):
        """驗證 MFE/MAE 在 evaluate() 中被計算。"""
        scan_date = date(2025, 6, 2)
        _insert_discovery(db_session, scan_date, "momentum", 1, "MM01", 100.0)

        # 持續上漲：high 比 close 高 1，low 比 close 低 1
        _insert_prices(db_session, "MM01", "2025-06-03", [102, 104, 106, 108, 110])

        perf = DiscoveryPerformance(mode="momentum", holding_days=[5])
        result = perf.evaluate()

        mfe_mae = result["mfe_mae"]
        assert not mfe_mae.empty
        row = mfe_mae.iloc[0]
        # MFE: (high=111 - 100) / 100 = 0.11
        assert row["mfe"] > 0
        # MAE: (low=101 - 100) / 100 = 0.01
        assert row["mae"] < row["mfe"]
        assert row["mfe_mae_ratio"] > 1.0

    def test_mfe_mae_empty_when_no_data(self, db_session):
        """無足夠價格資料時 MFE/MAE 為空。"""
        scan_date = date(2025, 6, 2)
        _insert_discovery(db_session, scan_date, "momentum", 1, "ME01", 100.0)
        # 不插入任何價格

        perf = DiscoveryPerformance(mode="momentum", holding_days=[5])
        result = perf.evaluate()
        # 沒有價格 → evaluate 整體為空
        assert result["mfe_mae"].empty


# ─── evaluate 回傳格式完整性 ──────────────────────────────


class TestEvaluateReturnKeys:
    """驗證 evaluate() 回傳包含所有新增 key。"""

    def test_empty_result_has_all_keys(self, db_session):
        """無資料時仍應回傳所有 key。"""
        perf = DiscoveryPerformance(mode="momentum")
        result = perf.evaluate()
        for key in ["summary", "by_scan", "detail", "by_regime", "turnover", "mfe_mae"]:
            assert key in result
            assert isinstance(result[key], pd.DataFrame)

    def test_detail_has_regime_column(self, db_session):
        """detail DataFrame 應包含 regime 欄位。"""
        scan_date = date(2025, 6, 2)
        _insert_discovery(db_session, scan_date, "momentum", 1, "RG01", 100.0, regime="bull")
        _insert_prices(db_session, "RG01", "2025-06-03", [101, 102, 103, 104, 105])

        perf = DiscoveryPerformance(mode="momentum", holding_days=[5])
        result = perf.evaluate()
        assert "regime" in result["detail"].columns
        assert result["detail"].iloc[0]["regime"] == "bull"
