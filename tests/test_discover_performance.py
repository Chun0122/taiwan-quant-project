"""測試 Discover 推薦績效回測 — DiscoveryPerformance。"""

from datetime import date

import pandas as pd
import pytest

from src.data.schema import DailyPrice, DiscoveryRecord
from src.discovery.performance import DiscoveryPerformance


def _insert_discovery(session, scan_date, mode, rank, stock_id, close, score=0.8, name=None):
    """輔助：插入一筆 DiscoveryRecord。"""
    session.add(DiscoveryRecord(
        scan_date=scan_date,
        mode=mode,
        rank=rank,
        stock_id=stock_id,
        stock_name=name or stock_id,
        close=close,
        composite_score=score,
    ))
    session.flush()


def _insert_prices(session, stock_id, start_date, prices):
    """輔助：插入多日 DailyPrice。prices 為收盤價列表。"""
    dates = pd.bdate_range(start_date, periods=len(prices))
    for dt, close in zip(dates, prices):
        session.add(DailyPrice(
            stock_id=stock_id,
            date=dt.date(),
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=1_000_000,
            turnover=100_000_000,
            spread=0.5,
        ))
    session.flush()


class TestBasicReturn:
    """基本報酬率計算。"""

    def test_single_recommendation(self, db_session):
        """插入 1 筆推薦 + 後續 20 天 DailyPrice，驗證 5/10/20 天報酬率。"""
        scan_date = date(2025, 6, 2)
        _insert_discovery(db_session, scan_date, "momentum", 1, "2330", 100.0)

        # 推薦日後 20 個交易日的價格（從 6/3 開始）
        prices = [101, 102, 103, 104, 105,  # day 1-5
                  106, 107, 108, 109, 110,  # day 6-10
                  111, 112, 113, 114, 115,  # day 11-15
                  116, 117, 118, 119, 120]  # day 16-20
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
                db_session, scan_date, "momentum", i,
                f"B{i:03d}", 100.0 + i,
            )

        # 為所有股票插入 5 天價格
        for i in range(1, 11):
            _insert_prices(
                db_session, f"B{i:03d}", "2025-09-02",
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
                db_session, "C001", str(next_day.date()),
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
