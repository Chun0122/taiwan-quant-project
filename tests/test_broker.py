"""分點交易資料（Broker Trade）整合測試。

測試項目：
- fetch_broker_trades:           FinMind TaiwanStockTradingDailyReport 欄位映射 + 數值清洗（mock HTTP）
- BrokerTrade ORM:               寫入 + 唯一鍵衝突（in-memory SQLite）
- compute_broker_score:          HHI 集中度計算 + 連續天數（純函數）
- MomentumScanner 7-factor:      7-factor 啟用、降級、集中度/連續天影響（_compute_chip_scores 單元）
- Stage 2.5 自動補抓:            MomentumScanner._auto_sync_broker=True 觸發 sync_broker_for_stocks()；
                                 SwingScanner._auto_sync_broker=False 不觸發
- sync_broker_for_stocks:        pipeline 包裝函數（days=7 透傳）
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

# 模組層級 ORM import：確保 Base.metadata 在 in_memory_engine.create_all() 前已包含全表
from src.data.schema import BrokerTrade  # noqa: F401
from src.discovery.scanner import MomentumScanner, SwingScanner, compute_broker_score

# ------------------------------------------------------------------ #
#  TestFetchBrokerTrades — mock HTTP 測試
# ------------------------------------------------------------------ #


class TestFetchBrokerTrades:
    """fetch_broker_trades() 欄位映射與邊界情況測試。"""

    def _mock_session(self, monkeypatch, json_data: dict):
        """設定 FinMindFetcher._session.get mock 回傳指定 JSON。"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = json_data
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        monkeypatch.setattr("src.data.fetcher.requests.Session", lambda: mock_session)
        monkeypatch.setattr("src.data.fetcher.time.sleep", lambda x: None)

    def test_column_mapping_and_stock_id_injection(self, monkeypatch):
        """欄位映射正確：securities_trader_id→broker_id，stock_id 由參數注入。"""
        from src.data.fetcher import FinMindFetcher

        self._mock_session(
            monkeypatch,
            {
                "msg": "success",
                "data": [
                    {
                        "date": "2026-03-05",
                        "stock_id": "2330",
                        "securities_trader_id": "1020",
                        "securities_trader": "元大",
                        "buy": "5000",
                        "sell": "2000",
                        "buy_price": "960.5",
                        "sell_price": "958.0",
                    }
                ],
            },
        )
        fetcher = FinMindFetcher(api_token="test_token")
        df = fetcher.fetch_broker_trades("2330", "2026-03-01", "2026-03-05")

        assert not df.empty
        assert "broker_id" in df.columns
        assert "broker_name" in df.columns
        assert "stock_id" in df.columns
        assert df.iloc[0]["broker_id"] == "1020"
        assert df.iloc[0]["broker_name"] == "元大"
        assert df.iloc[0]["stock_id"] == "2330"
        assert df.iloc[0]["buy"] == 5000
        assert df.iloc[0]["sell"] == 2000

    def test_empty_data_returns_empty_df(self, monkeypatch):
        """FinMind 回傳空 data 時，回傳空 DataFrame。"""
        from src.data.fetcher import FinMindFetcher

        self._mock_session(monkeypatch, {"msg": "success", "data": []})
        fetcher = FinMindFetcher(api_token="test_token")
        df = fetcher.fetch_broker_trades("2330", "2026-03-01", "2026-03-05")
        assert df.empty

    def test_finmind_error_raises(self, monkeypatch):
        """FinMind API 回傳 msg != 'success' 時應拋出 RuntimeError。"""
        from src.data.fetcher import FinMindFetcher

        self._mock_session(monkeypatch, {"msg": "error", "status": 400})
        fetcher = FinMindFetcher(api_token="test_token")
        with pytest.raises(RuntimeError):
            fetcher.fetch_broker_trades("2330", "2026-03-01", "2026-03-05")

    def test_buy_sell_numeric_cleansing(self, monkeypatch):
        """buy/sell 欄位：字串數字、NaN、無效值皆轉為整數（無效→0）。"""
        from src.data.fetcher import FinMindFetcher

        self._mock_session(
            monkeypatch,
            {
                "msg": "success",
                "data": [
                    {
                        "date": "2026-03-05",
                        "stock_id": "2317",
                        "securities_trader_id": "9999",
                        "securities_trader": "測試券商",
                        "buy": "abc",  # 無效→0
                        "sell": None,  # None→0
                        "buy_price": None,
                        "sell_price": None,
                    }
                ],
            },
        )
        fetcher = FinMindFetcher(api_token="test_token")
        df = fetcher.fetch_broker_trades("2317", "2026-03-01", "2026-03-05")

        assert not df.empty
        assert df.iloc[0]["buy"] == 0
        assert df.iloc[0]["sell"] == 0

    def test_date_column_converted_to_date_type(self, monkeypatch):
        """date 欄位應被轉換為 Python date 型別。"""
        from src.data.fetcher import FinMindFetcher

        self._mock_session(
            monkeypatch,
            {
                "msg": "success",
                "data": [
                    {
                        "date": "2026-03-05",
                        "stock_id": "6505",
                        "securities_trader_id": "5555",
                        "securities_trader": "台新",
                        "buy": "1000",
                        "sell": "500",
                    }
                ],
            },
        )
        fetcher = FinMindFetcher(api_token="test_token")
        df = fetcher.fetch_broker_trades("6505", "2026-03-01", "2026-03-05")
        assert not df.empty
        assert isinstance(df.iloc[0]["date"], date)


# ------------------------------------------------------------------ #
#  TestBrokerTradeORM — in-memory SQLite 測試
# ------------------------------------------------------------------ #


class TestBrokerTradeORM:
    """BrokerTrade ORM CRUD 測試（in-memory SQLite）。"""

    def test_insert_and_query(self, db_session):
        """寫入一筆分點資料後可查詢回來。"""
        dt = date(2026, 3, 5)
        entry = BrokerTrade(
            stock_id="2330",
            date=dt,
            broker_id="1020",
            broker_name="元大",
            buy=5000,
            sell=2000,
            buy_price=960.5,
            sell_price=958.0,
        )
        db_session.add(entry)
        db_session.flush()

        result = db_session.query(BrokerTrade).filter_by(stock_id="2330", broker_id="1020").first()
        assert result is not None
        assert result.broker_name == "元大"
        assert result.buy == 5000
        assert result.sell == 2000

    def test_unique_constraint_on_conflict_do_nothing(self, db_session):
        """相同 stock_id + date + broker_id 衝突時 on_conflict_do_nothing 略過，不報錯。"""
        dt = date(2026, 3, 5)
        record1 = {
            "stock_id": "2317",
            "date": dt,
            "broker_id": "2222",
            "broker_name": "凱基",
            "buy": 1000,
            "sell": 500,
        }
        record2 = {
            "stock_id": "2317",
            "date": dt,
            "broker_id": "2222",
            "broker_name": "凱基改版",  # 不同 broker_name，但唯一鍵相同
            "buy": 9999,
            "sell": 8888,
        }

        stmt1 = sqlite_upsert(BrokerTrade).values([record1])
        stmt1 = stmt1.on_conflict_do_nothing(index_elements=["stock_id", "date", "broker_id"])
        db_session.execute(stmt1)
        db_session.flush()

        stmt2 = sqlite_upsert(BrokerTrade).values([record2])
        stmt2 = stmt2.on_conflict_do_nothing(index_elements=["stock_id", "date", "broker_id"])
        db_session.execute(stmt2)
        db_session.flush()

        rows = db_session.query(BrokerTrade).filter_by(stock_id="2317", broker_id="2222").all()
        assert len(rows) == 1
        # 第一筆值保留
        assert rows[0].buy == 1000

    def test_buy_sell_zero_is_valid(self, db_session):
        """buy=0, sell=0 為合法資料（某日無交易的分點）。"""
        dt = date(2026, 3, 5)
        entry = BrokerTrade(
            stock_id="6505",
            date=dt,
            broker_id="3333",
            broker_name="國泰",
            buy=0,
            sell=0,
        )
        db_session.add(entry)
        db_session.flush()

        result = db_session.query(BrokerTrade).filter_by(stock_id="6505").first()
        assert result is not None
        assert result.buy == 0
        assert result.sell == 0


# ------------------------------------------------------------------ #
#  TestComputeBrokerScore — 純函數測試
# ------------------------------------------------------------------ #


class TestComputeBrokerScore:
    """compute_broker_score() 純函數測試。"""

    def test_empty_df_returns_correct_columns(self):
        """空 DF 回傳空 DataFrame，欄位 [stock_id, broker_concentration, broker_consecutive_days]。"""
        result = compute_broker_score(pd.DataFrame())
        assert result.empty
        assert set(result.columns) == {"stock_id", "broker_concentration", "broker_consecutive_days"}

    def test_missing_required_columns_returns_empty(self):
        """缺少必要欄位時回傳空 DataFrame。"""
        df = pd.DataFrame([{"stock_id": "2330", "buy": 1000}])
        result = compute_broker_score(df)
        assert result.empty
        assert set(result.columns) == {"stock_id", "broker_concentration", "broker_consecutive_days"}

    def test_hhi_calculation_two_brokers(self):
        """2 個分點各買 3000 / 1000 → HHI = (0.75)^2 + (0.25)^2 = 0.625。"""
        dt = date(2026, 3, 5)
        df = pd.DataFrame(
            [
                {"date": dt, "stock_id": "2330", "broker_id": "A001", "buy": 3000, "sell": 0},
                {"date": dt, "stock_id": "2330", "broker_id": "A002", "buy": 1000, "sell": 0},
            ]
        )
        result = compute_broker_score(df)
        assert len(result) == 1
        hhi = result.iloc[0]["broker_concentration"]
        # A001 佔比 3/4=0.75，A002 佔比 1/4=0.25
        # HHI = 0.75^2 + 0.25^2 = 0.5625 + 0.0625 = 0.625
        assert abs(hhi - 0.625) < 1e-6

    def test_hhi_perfect_concentration(self):
        """只有 1 個分點淨買超 → HHI = 1.0（完全集中）。"""
        dt = date(2026, 3, 5)
        df = pd.DataFrame(
            [
                {"date": dt, "stock_id": "2330", "broker_id": "A001", "buy": 5000, "sell": 0},
                {"date": dt, "stock_id": "2330", "broker_id": "A002", "buy": 0, "sell": 3000},  # 淨賣
            ]
        )
        result = compute_broker_score(df)
        assert len(result) == 1
        assert abs(result.iloc[0]["broker_concentration"] - 1.0) < 1e-6

    def test_consecutive_days_calculation(self):
        """最強主力分點連續 3 日淨買超，consecutive_days = 3。"""
        today = date(2026, 3, 7)
        days_data = []
        for i in range(3):
            d = today - timedelta(days=i)
            days_data.append({"date": d, "stock_id": "2330", "broker_id": "A001", "buy": 2000, "sell": 100})
        # 第 4 天該主力賣超
        days_data.append(
            {"date": today - timedelta(days=3), "stock_id": "2330", "broker_id": "A001", "buy": 100, "sell": 5000}
        )
        df = pd.DataFrame(days_data)
        result = compute_broker_score(df)
        assert len(result) == 1
        assert result.iloc[0]["broker_consecutive_days"] == 3

    def test_multiple_stocks_independent(self):
        """多支股票各自獨立計算，不互相干擾。"""
        dt = date(2026, 3, 5)
        df = pd.DataFrame(
            [
                {"date": dt, "stock_id": "2330", "broker_id": "A001", "buy": 4000, "sell": 0},
                {"date": dt, "stock_id": "2330", "broker_id": "A002", "buy": 0, "sell": 1000},
                {"date": dt, "stock_id": "2317", "broker_id": "B001", "buy": 1000, "sell": 0},
                {"date": dt, "stock_id": "2317", "broker_id": "B002", "buy": 1000, "sell": 0},
            ]
        )
        result = compute_broker_score(df)
        assert len(result) == 2

        r2330 = result[result["stock_id"] == "2330"].iloc[0]
        r2317 = result[result["stock_id"] == "2317"].iloc[0]

        # 2330: 只有 A001 淨買 → HHI = 1.0
        assert abs(r2330["broker_concentration"] - 1.0) < 1e-6
        # 2317: B001 和 B002 各半 → HHI = 0.5
        assert abs(r2317["broker_concentration"] - 0.5) < 1e-6

    def test_all_selling_returns_zero_concentration(self):
        """所有分點均賣超時，無淨買超→ HHI = 0。"""
        dt = date(2026, 3, 5)
        df = pd.DataFrame(
            [
                {"date": dt, "stock_id": "1234", "broker_id": "X001", "buy": 100, "sell": 5000},
                {"date": dt, "stock_id": "1234", "broker_id": "X002", "buy": 200, "sell": 3000},
            ]
        )
        result = compute_broker_score(df)
        assert len(result) == 1
        assert result.iloc[0]["broker_concentration"] == 0.0


# ------------------------------------------------------------------ #
#  TestMomentumScannerBrokerFactor — 7-factor 路徑 + 降級測試
# ------------------------------------------------------------------ #


@pytest.fixture()
def momentum_scanner():
    return MomentumScanner(
        min_price=10,
        max_price=2000,
        min_volume=100_000,
        top_n_candidates=10,
    )


def _make_broker_df(stock_ids: list[str], concentrations: list[float]) -> pd.DataFrame:
    """建立測試用 BrokerTrade raw DataFrame（最新一日，各股指定集中度）。

    concentration 以全部買量由單一分點買入的比例模擬。
    """
    dt = date.today()
    rows = []
    for sid, conc in zip(stock_ids, concentrations):
        # 一個主力買入 conc 比例，其餘 (1-conc) 為其他分點
        main_buy = int(conc * 10000)
        rest_buy = int((1.0 - conc) * 10000)
        rows.append({"date": dt, "stock_id": sid, "broker_id": "MAIN", "buy": main_buy, "sell": 0})
        if rest_buy > 0:
            rows.append({"date": dt, "stock_id": sid, "broker_id": "REST", "buy": rest_buy, "sell": 0})
    return pd.DataFrame(rows)


def _make_inst_df(stock_ids: list[str]) -> pd.DataFrame:
    """建立測試用法人資料（各股相同）。"""
    dt = date.today() - timedelta(days=1)
    rows = [{"stock_id": sid, "date": dt, "name": "外資買賣超", "net": 100} for sid in stock_ids]
    return pd.DataFrame(rows)


def _make_margin_df(stock_ids: list[str]) -> pd.DataFrame:
    """建立測試用融資融券資料（各股相同）。"""
    dt = date.today() - timedelta(days=1)
    rows = [{"stock_id": sid, "date": dt, "margin_balance": 1000, "short_balance": 100} for sid in stock_ids]
    return pd.DataFrame(rows)


def _make_sbl_df(stock_ids: list[str]) -> pd.DataFrame:
    """建立測試用借券 DataFrame（各股相同借券餘額）。"""
    dt = date.today()
    rows = [{"date": dt, "stock_id": sid, "sbl_balance": 5000, "sbl_change": 100} for sid in stock_ids]
    return pd.DataFrame(rows)


def _make_holding_df(stock_ids: list[str]) -> pd.DataFrame:
    """建立測試用大戶持股 DataFrame。"""
    dt = date.today()
    rows = [{"date": dt, "stock_id": sid, "level": "400,001-600,000 Shares", "percent": 15.0} for sid in stock_ids]
    return pd.DataFrame(rows)


class TestMomentumScannerBrokerFactor:
    """MomentumScanner._compute_chip_scores() 分點因子整合測試。"""

    def test_seven_factor_path_activated(self, momentum_scanner, monkeypatch):
        """有分點+借券+融資融券+大戶時，啟用 7-factor，chip_score 正常計算。"""
        sids = ["1000", "1001", "1002"]
        df_inst = _make_inst_df(sids)
        df_margin = _make_margin_df(sids)

        monkeypatch.setattr(momentum_scanner, "_load_broker_data", lambda _: _make_broker_df(sids, [0.8, 0.5, 0.3]))
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: _make_sbl_df(sids))
        monkeypatch.setattr(momentum_scanner, "_load_holding_data", lambda _: _make_holding_df(sids))

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, df_margin)
        assert len(result) == 3
        assert (result["chip_score"] >= 0).all()
        assert (result["chip_score"] <= 1.0).all()

    def test_no_broker_falls_back_to_six_factor(self, momentum_scanner, monkeypatch):
        """無分點資料時，降級至 6-factor（借券+融資券+大戶路徑）。"""
        sids = ["1000", "1001"]
        df_inst = _make_inst_df(sids)
        df_margin = _make_margin_df(sids)

        # broker 回傳空 → 不走 7-factor
        monkeypatch.setattr(momentum_scanner, "_load_broker_data", lambda _: pd.DataFrame())
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: _make_sbl_df(sids))
        monkeypatch.setattr(momentum_scanner, "_load_holding_data", lambda _: _make_holding_df(sids))

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, df_margin)
        assert len(result) == 2
        assert (result["chip_score"] >= 0).all()
        assert (result["chip_score"] <= 1.0).all()

    def test_high_concentration_gets_higher_score(self, momentum_scanner, monkeypatch):
        """主力集中度高的股票，chip_score 應較集中度低的股票高（其他因子相同）。"""
        sids = ["high_conc", "low_conc"]

        df_inst = _make_inst_df(sids)
        # 法人買超量相同
        df_margin = _make_margin_df(sids)

        # high_conc 集中度 0.95（接近 1），low_conc 集中度 0.3
        monkeypatch.setattr(
            momentum_scanner,
            "_load_broker_data",
            lambda _: _make_broker_df(sids, [0.95, 0.30]),
        )
        # 其他因子相同（空 DF 或相同值）
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: pd.DataFrame())
        monkeypatch.setattr(momentum_scanner, "_load_holding_data", lambda _: pd.DataFrame())

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, None)
        scores = result.set_index("stock_id")["chip_score"]
        # 集中度高 → broker_rank 高 → chip_score 高
        assert scores["high_conc"] > scores["low_conc"]

    def test_more_consecutive_days_gets_higher_score(self, momentum_scanner, monkeypatch):
        """連續進場天數多的股票，chip_score 應較天數少的股票高（其他因子相同）。"""
        sids = ["many_days", "few_days"]
        df_inst = _make_inst_df(sids)

        today = date.today()
        # many_days：主力連續 5 天淨買
        # few_days：主力連續 1 天淨買
        broker_rows = []
        for i in range(5):
            broker_rows.append(
                {
                    "date": today - timedelta(days=i),
                    "stock_id": "many_days",
                    "broker_id": "M001",
                    "buy": 1000,
                    "sell": 0,
                }
            )
        broker_rows.append({"date": today, "stock_id": "few_days", "broker_id": "F001", "buy": 1000, "sell": 0})
        # few_days 前 4 天賣超
        for i in range(1, 5):
            broker_rows.append(
                {"date": today - timedelta(days=i), "stock_id": "few_days", "broker_id": "F001", "buy": 0, "sell": 500}
            )
        df_broker = pd.DataFrame(broker_rows)

        monkeypatch.setattr(momentum_scanner, "_load_broker_data", lambda _: df_broker)
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: pd.DataFrame())
        monkeypatch.setattr(momentum_scanner, "_load_holding_data", lambda _: pd.DataFrame())

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, None)
        scores = result.set_index("stock_id")["chip_score"]
        # 連續天數多 → broker_rank 高 → chip_score 高
        assert scores["many_days"] > scores["few_days"]

    def test_no_all_data_falls_to_three_factor(self, momentum_scanner, monkeypatch):
        """無分點/借券/融資融券/大戶資料時，使用 3-factor。"""
        sids = ["1000"]
        df_inst = _make_inst_df(sids)

        monkeypatch.setattr(momentum_scanner, "_load_broker_data", lambda _: pd.DataFrame())
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: pd.DataFrame())
        monkeypatch.setattr(momentum_scanner, "_load_holding_data", lambda _: pd.DataFrame())

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, None)
        assert len(result) == 1
        assert 0.0 <= result.iloc[0]["chip_score"] <= 1.0


# ------------------------------------------------------------------ #
#  TestSyncBrokerForStocks — pipeline 包裝函數測試
# ------------------------------------------------------------------ #


class TestSyncBrokerForStocks:
    """sync_broker_for_stocks() 包裝函數測試。"""

    def test_delegates_to_sync_broker_trades_with_days_7(self, monkeypatch):
        """sync_broker_for_stocks() 應以 days=7 呼叫 sync_broker_trades()。"""
        from src.data.pipeline import sync_broker_for_stocks

        calls = []

        def fake_sync(stock_ids, days):
            calls.append({"stock_ids": stock_ids, "days": days})
            return 42

        monkeypatch.setattr("src.data.pipeline.sync_broker_trades", fake_sync)

        result = sync_broker_for_stocks(["2330", "2317"])

        assert len(calls) == 1
        assert calls[0]["stock_ids"] == ["2330", "2317"]
        assert calls[0]["days"] == 7
        assert result == 42


# ------------------------------------------------------------------ #
#  TestStage25AutoFetch — Stage 2.5 自動補抓觸發測試
# ------------------------------------------------------------------ #


class TestStage25AutoFetch:
    """Stage 2.5 分點自動補抓觸發機制測試。"""

    def test_momentum_scanner_has_auto_sync_broker_true(self):
        """MomentumScanner._auto_sync_broker 應為 True。"""
        assert MomentumScanner._auto_sync_broker is True

    def test_swing_scanner_has_auto_sync_broker_false(self):
        """SwingScanner._auto_sync_broker 應為 False（不需分點因子）。"""
        assert SwingScanner._auto_sync_broker is False

    def test_stage25_calls_sync_broker_for_stocks_in_momentum(self, monkeypatch):
        """MomentumScanner.run() Stage 2.5 應呼叫 sync_broker_for_stocks()，且傳入候選股 ID。"""
        from src.discovery.scanner import MomentumScanner

        scanner = MomentumScanner(top_n_candidates=5, top_n_results=3)
        broker_sync_calls = []

        # 模擬完整 run() 所需的 DB 資料
        dummy_price = pd.DataFrame(
            {
                "stock_id": ["1000", "1001", "1002"],
                "date": [date.today()] * 3,
                "open": [100.0] * 3,
                "high": [105.0] * 3,
                "low": [98.0] * 3,
                "close": [103.0] * 3,
                "volume": [5_000_000] * 3,
                "turnover": [500_000_000] * 3,
                "spread": [0.0] * 3,
            }
        )
        dummy_inst = pd.DataFrame(
            [
                {"stock_id": sid, "date": date.today(), "name": "外資買賣超", "net": 100}
                for sid in ["1000", "1001", "1002"]
            ]
        )
        dummy_margin = pd.DataFrame(
            [
                {"stock_id": sid, "date": date.today(), "margin_balance": 1000, "short_balance": 100}
                for sid in ["1000", "1001", "1002"]
            ]
        )

        monkeypatch.setattr(
            scanner, "_load_market_data", lambda: (dummy_price, dummy_inst, dummy_margin, pd.DataFrame())
        )
        monkeypatch.setattr(scanner, "_coarse_filter", lambda p, i: dummy_price[["stock_id"]].drop_duplicates())

        # 攔截 sync_broker_for_stocks

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_sync_broker_for_stocks(stock_ids):
            broker_sync_calls.append(list(stock_ids))
            return 0

        monkeypatch.setattr("src.data.pipeline.sync_broker_for_stocks", mock_sync_broker_for_stocks, raising=False)

        # patch sync_revenue_for_stocks 和其他 Stage 方法，避免 DB 依賴
        monkeypatch.setattr("src.data.pipeline.sync_revenue_for_stocks", lambda ids: 0, raising=False)
        monkeypatch.setattr(scanner, "_load_revenue_data", lambda ids, months=2: pd.DataFrame())
        monkeypatch.setattr(scanner, "_load_announcement_data", lambda ids: pd.DataFrame())
        monkeypatch.setattr(
            scanner,
            "_score_candidates",
            lambda *a, **kw: pd.DataFrame(
                {
                    "stock_id": ["1000"],
                    "composite_score": [0.8],
                    "chip_score": [0.7],
                    "technical_score": [0.7],
                    "fundamental_score": [0.5],
                    "news_score": [0.5],
                    "sector_bonus": [0.0],
                }
            ),
        )
        monkeypatch.setattr(scanner, "_apply_sector_bonus", lambda df: df)
        monkeypatch.setattr(scanner, "_apply_risk_filter", lambda df, p: df)
        monkeypatch.setattr(
            scanner,
            "_rank_and_enrich",
            lambda df: df.assign(
                close=100.0,
                stock_name="測試",
                industry_category="電子",
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                entry_trigger=None,
                valid_until=None,
            ),
        )
        monkeypatch.setattr(scanner, "_compute_sector_summary", lambda df: pd.DataFrame())

        # 模擬 Regime 偵測（避免 DB 依賴）
        import src.regime.detector as regime_module

        monkeypatch.setattr(
            regime_module.MarketRegimeDetector, "detect", lambda self: {"regime": "bull", "taiex_close": 20000.0}
        )

        scanner.run()

        # 驗證 sync_broker_for_stocks 被呼叫，且傳入候選股 ID
        assert len(broker_sync_calls) == 1
        assert set(broker_sync_calls[0]) == {"1000", "1001", "1002"}
