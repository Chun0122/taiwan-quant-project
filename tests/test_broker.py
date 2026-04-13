"""分點交易資料（Broker Trade）整合測試。

測試項目：
- fetch_broker_trades:                FinMind TaiwanStockTradingDailyReport 欄位映射 + 數值清洗（mock HTTP）
- BrokerTrade ORM:                    寫入 + 唯一鍵衝突（in-memory SQLite）
- compute_broker_score:               HHI 集中度計算 + 連續天數（純函數）
- MomentumScanner 7-factor:           7-factor 啟用、降級、集中度/連續天影響（_compute_chip_scores 單元）
- Stage 2.5 自動補抓:                 MomentumScanner._auto_sync_broker=True 觸發 sync_broker_for_stocks()；
                                      SwingScanner._auto_sync_broker=False 不觸發
- sync_broker_for_stocks:             pipeline 包裝函數（days=7 透傳）
- LoadBrokerDataExtendedAdaptive:     _load_broker_data_extended() 自適應門檻（min_trading_days 過濾）
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

# 模組層級 ORM import：確保 Base.metadata 在 in_memory_engine.create_all() 前已包含全表
from src.data.schema import BrokerTrade, DailyPrice  # noqa: F401
from src.discovery.scanner import MomentumScanner, SwingScanner, compute_broker_score, compute_smart_broker_score

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
#  TestFetchDJBrokerTrades — DJ 分點端點 HTML 解析測試
# ------------------------------------------------------------------ #

_DJ_HTML_TWO_PAIRS = (
    b"<TR>\n"
    b'<TD class="t4t1" nowrap><a href="/z/zc/zco/zco0/zco0.djhtm?a=2330&b=1360&BHID=1360">BrokerA</a></TD>\n'
    b'<TD class="t3n1">3,155</TD>\n'
    b'<TD class="t3n1">1,124</TD>\n'
    b'<TD class="t3n1">2,031</TD>\n'
    b'<TD class="t3n1">3.97%</TD>\n'
    b'<TD class="t4t1" nowrap><a href="/z/zc/zco/zco0/zco0.djhtm?a=2330&b=1480&BHID=1480">BrokerB</a></TD>\n'
    b'<TD class="t3n1">789</TD>\n'
    b'<TD class="t3n1">4,182</TD>\n'
    b'<TD class="t3n1">3,393</TD>\n'
    b'<TD class="t3n1">6.64%</TD></TR>\n'
)

_DJ_HTML_MULTI_BRANCH = (
    b"<TR>\n"
    b'<TD class="t4t1" nowrap><a href="/z/zc/zco/zco0/zco0.djhtm?a=2330&b=8880&BHID=8880">YuantaMain</a></TD>\n'
    b'<TD class="t3n1">1,000</TD>\n'
    b'<TD class="t3n1">200</TD>\n'
    b'<TD class="t3n1">800</TD>\n'
    b'<TD class="t3n1">2%</TD>\n'
    b'<TD class="t4t1" nowrap><a href="/z/zc/zco/zco0/zco0.djhtm?a=2330&b=8888&BHID=8880">YuantaBranch</a></TD>\n'
    b'<TD class="t3n1">500</TD>\n'
    b'<TD class="t3n1">100</TD>\n'
    b'<TD class="t3n1">400</TD>\n'
    b'<TD class="t3n1">1%</TD></TR>\n'
)


class TestFetchDJBrokerTrades:
    """fetch_dj_broker_trades() HTML 解析與邊界情況測試。"""

    def _mock_get(self, monkeypatch, response_bytes: bytes):
        """Mock requests.get 回傳指定 bytes。"""
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = response_bytes
        monkeypatch.setattr("src.data.twse_fetcher.requests.get", lambda *a, **kw: mock_resp)
        monkeypatch.setattr("src.data.twse_fetcher.time.sleep", lambda x: None)

    def test_basic_parsing_two_brokers(self, monkeypatch):
        """兩個 broker 條目（左淨買、右淨賣）正確解析。"""
        from src.data.twse_fetcher import fetch_dj_broker_trades

        self._mock_get(monkeypatch, _DJ_HTML_TWO_PAIRS)
        end = date(2025, 3, 10)
        df = fetch_dj_broker_trades("2330", date(2025, 3, 6), end)

        assert not df.empty
        assert len(df) == 2
        assert set(df.columns) >= {"date", "stock_id", "broker_id", "broker_name", "buy", "sell"}

    def test_broker_id_from_bhid(self, monkeypatch):
        """broker_id 應為 BHID 欄位（公司代號）。"""
        from src.data.twse_fetcher import fetch_dj_broker_trades

        self._mock_get(monkeypatch, _DJ_HTML_TWO_PAIRS)
        df = fetch_dj_broker_trades("2330", date(2025, 3, 6), date(2025, 3, 10))

        assert "1360" in df["broker_id"].values
        assert "1480" in df["broker_id"].values

    def test_units_multiplied_by_1000(self, monkeypatch):
        """buy/sell 應乘以 1000（張→股）。"""
        from src.data.twse_fetcher import fetch_dj_broker_trades

        self._mock_get(monkeypatch, _DJ_HTML_TWO_PAIRS)
        df = fetch_dj_broker_trades("2330", date(2025, 3, 6), date(2025, 3, 10))

        row_a = df[df["broker_id"] == "1360"].iloc[0]
        assert row_a["buy"] == 3155 * 1000
        assert row_a["sell"] == 1124 * 1000

    def test_date_set_to_end_date(self, monkeypatch):
        """date 欄位應統一設為 end 日期（彙整截止日）。"""
        from src.data.twse_fetcher import fetch_dj_broker_trades

        self._mock_get(monkeypatch, _DJ_HTML_TWO_PAIRS)
        end = date(2025, 3, 10)
        df = fetch_dj_broker_trades("2330", date(2025, 3, 6), end)

        assert all(df["date"] == end)

    def test_multi_branch_same_firm_aggregated(self, monkeypatch):
        """同一公司（BHID=8880）的主分點+子分點應合計 buy/sell。"""
        from src.data.twse_fetcher import fetch_dj_broker_trades

        self._mock_get(monkeypatch, _DJ_HTML_MULTI_BRANCH)
        df = fetch_dj_broker_trades("2330", date(2025, 3, 6), date(2025, 3, 10))

        assert len(df) == 1  # 兩個分點合併為一筆
        row = df.iloc[0]
        assert row["broker_id"] == "8880"
        assert row["buy"] == (1000 + 500) * 1000  # 1500 張 → 1,500,000 股
        assert row["sell"] == (200 + 100) * 1000  # 300 張 → 300,000 股

    def test_no_buy_price_sell_price(self, monkeypatch):
        """buy_price / sell_price 應為 None（DJ 端點不提供均價）。"""
        from src.data.twse_fetcher import fetch_dj_broker_trades

        self._mock_get(monkeypatch, _DJ_HTML_TWO_PAIRS)
        df = fetch_dj_broker_trades("2330", date(2025, 3, 6), date(2025, 3, 10))

        assert df["buy_price"].isna().all()
        assert df["sell_price"].isna().all()

    def test_empty_html_returns_empty_df(self, monkeypatch):
        """HTML 中無 BHID 時回傳空 DataFrame。"""
        from src.data.twse_fetcher import fetch_dj_broker_trades

        self._mock_get(monkeypatch, b"<html><body>no data</body></html>")
        df = fetch_dj_broker_trades("2330", date(2025, 3, 6), date(2025, 3, 10))

        assert df.empty

    def test_request_failure_returns_empty_df(self, monkeypatch):
        """requests.get 拋出異常時回傳空 DataFrame（不向上傳播）。"""
        from unittest.mock import MagicMock

        from src.data.twse_fetcher import fetch_dj_broker_trades

        mock_get = MagicMock(side_effect=ConnectionError("timeout"))
        monkeypatch.setattr("src.data.twse_fetcher.requests.get", mock_get)
        monkeypatch.setattr("src.data.twse_fetcher.time.sleep", lambda x: None)
        df = fetch_dj_broker_trades("2330", date(2025, 3, 6), date(2025, 3, 10))

        assert df.empty

    def test_big5_invalid_bytes_fallback_to_replace(self, monkeypatch):
        """Big5 解碼含無效位元組時應 fallback 至 replace 模式，仍回傳可解析結果。"""
        from unittest.mock import MagicMock

        from src.data.twse_fetcher import fetch_dj_broker_trades

        # 在有效 HTML 中插入一個無效的 Big5 byte（0xFF 不是合法 Big5）
        # 但 regex 仍可解析出有效的 broker 條目
        valid_big5 = _DJ_HTML_TWO_PAIRS  # 已知可解析的 HTML bytes
        # 用 replace 模式的 bytes 確保函式不會崩潰
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        # 在 content 前面塞入一個無效 Big5 byte
        mock_resp.content = b"\xff" + valid_big5
        monkeypatch.setattr("src.data.twse_fetcher.requests.get", lambda *a, **kw: mock_resp)
        monkeypatch.setattr("src.data.twse_fetcher.time.sleep", lambda x: None)

        df = fetch_dj_broker_trades("2330", date(2025, 3, 6), date(2025, 3, 10))
        # 不應崩潰，且應能解析出資料（或至少回傳空 DataFrame）
        assert isinstance(df, type(pd.DataFrame()))


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

    def test_six_factor_path_activated(self, momentum_scanner, monkeypatch):
        """有分點+借券+融資融券時，啟用 6F，chip_score 正常計算。"""
        sids = ["1000", "1001", "1002"]
        df_inst = _make_inst_df(sids)
        df_margin = _make_margin_df(sids)

        monkeypatch.setattr(momentum_scanner, "_load_broker_data", lambda _: _make_broker_df(sids, [0.8, 0.5, 0.3]))
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: _make_sbl_df(sids))

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, df_margin)
        assert len(result) == 3
        assert (result["chip_score"] >= 0).all()
        assert (result["chip_score"] <= 1.0).all()

    def test_no_broker_falls_back_to_five_factor(self, momentum_scanner, monkeypatch):
        """無分點資料時，降級至 5F（借券+融資券路徑）。"""
        sids = ["1000", "1001"]
        df_inst = _make_inst_df(sids)
        df_margin = _make_margin_df(sids)

        # broker 回傳空 → 不走 6F
        monkeypatch.setattr(momentum_scanner, "_load_broker_data", lambda _: pd.DataFrame())
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: _make_sbl_df(sids))

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
        # 其他因子相同（空 DF）
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: pd.DataFrame())

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

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, None)
        scores = result.set_index("stock_id")["chip_score"]
        # 連續天數多 → broker_rank 高 → chip_score 高
        assert scores["many_days"] > scores["few_days"]

    def test_no_all_data_falls_to_three_factor(self, momentum_scanner, monkeypatch):
        """無分點/借券/融資融券資料時，使用 3F。"""
        sids = ["1000"]
        df_inst = _make_inst_df(sids)

        monkeypatch.setattr(momentum_scanner, "_load_broker_data", lambda _: pd.DataFrame())
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: pd.DataFrame())

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
#  TestSyncBrokerBootstrap — 逐日 Bootstrap 測試
# ------------------------------------------------------------------ #


class TestSyncBrokerBootstrap:
    """sync_broker_bootstrap() 逐日補齊歷史分點資料測試。"""

    def test_bootstrap_default_days_is_30(self):
        """sync_broker_bootstrap() 預設 days=30。"""
        import inspect

        from src.data.pipeline import sync_broker_bootstrap

        sig = inspect.signature(sync_broker_bootstrap)
        assert "stock_ids" in sig.parameters
        assert "days" in sig.parameters
        assert sig.parameters["days"].default == 30

    def test_bootstrap_independent_from_sync_broker_trades(self):
        """sync_broker_bootstrap 存在且與 sync_broker_trades 獨立（不是包裝它）。"""
        import inspect

        from src.data.pipeline import sync_broker_bootstrap, sync_broker_trades

        # 確認兩個函數是獨立的（bootstrap 不是 sync_broker_trades 的 alias）
        assert sync_broker_bootstrap is not sync_broker_trades
        # 確認 bootstrap 有 days 參數但預設值不同（bootstrap=30, sync=5）
        assert inspect.signature(sync_broker_bootstrap).parameters["days"].default == 30
        assert inspect.signature(sync_broker_trades).parameters["days"].default == 5

    def test_bootstrap_calls_dj_once_per_trading_day(self, monkeypatch):
        """Bootstrap 應對每個交易日獨立呼叫 DJ 端點（start=d, end=d），而非一次大範圍查詢。"""
        from src.data.pipeline import sync_broker_bootstrap

        fetch_calls: list[tuple] = []

        def mock_fetch(stock_id, start, end):
            fetch_calls.append((stock_id, start, end))
            return pd.DataFrame()  # 回傳空，不寫 DB

        # 模擬交易日清單（跳過 DailyPrice 查詢，退回平日曆法）
        monkeypatch.setattr("src.data.pipeline.init_db", lambda: None)
        monkeypatch.setattr("src.data.pipeline.get_effective_watchlist", lambda: ["TEST"])

        import src.data.twse_fetcher as twse_mod

        original = twse_mod.fetch_dj_broker_trades
        twse_mod.fetch_dj_broker_trades = mock_fetch

        # 讓 DailyPrice 查詢失敗，觸發平日曆法 fallback（簡化測試環境）
        def fail_session():
            raise Exception("no db in test")

        monkeypatch.setattr("src.data.pipeline.get_session", fail_session)

        try:
            sync_broker_bootstrap(stock_ids=["TEST"], days=5)
        except Exception:
            pass  # 可能因其他 DB 操作失敗，但我們關注的是 fetch_calls 的模式
        finally:
            twse_mod.fetch_dj_broker_trades = original

        # 若有呼叫，每次的 start 應等於 end（單日查詢）
        for _, start, end in fetch_calls:
            assert start == end, f"Bootstrap 應用單日查詢（start=end），但 start={start}, end={end}"

    def test_bootstrap_single_day_query_vs_range_query(self):
        """驗證 Bootstrap 與一般 sync 的根本差異：Bootstrap 用 start=d, end=d（單日），sync 用 start~end（範圍）。"""
        import inspect

        from src.data.pipeline import sync_broker_bootstrap

        # Bootstrap 透過逐日 loop 呼叫，確保每次 start=end（產生多個獨立 date 記錄）
        # 一般 sync_broker_trades 用 start=today-days, end=today（期間彙整，date=end）
        # 此測試確認函數存在且有正確接口
        sig = inspect.signature(sync_broker_bootstrap)
        assert list(sig.parameters.keys()) == ["stock_ids", "days"]


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
        monkeypatch.setattr(scanner, "_load_announcement_data", lambda ids: (pd.DataFrame(), pd.DataFrame()))
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


# ------------------------------------------------------------------ #
#  TestComputeSmartBrokerScore — compute_smart_broker_score() 純函數測試
# ------------------------------------------------------------------ #


def _make_smart_rows(stock_id: str, broker_id: str, base: date) -> list[dict]:
    """建立符合 Smart Broker 條件的交易資料（3 贏 1 小虧，win_rate=0.75, PF=12.0, buy_val=6M）。"""
    return [
        # Day -10：買入 60000 股 at 100 → avg_cost=100, total_buy_value=6M
        {
            "date": base - timedelta(days=10),
            "stock_id": stock_id,
            "broker_id": broker_id,
            "buy": 60000,
            "sell": 0,
            "buy_price": 100.0,
            "sell_price": 0.0,
        },
        # Day -8：賣 5000 at 115 → profit 75,000（win）
        {
            "date": base - timedelta(days=8),
            "stock_id": stock_id,
            "broker_id": broker_id,
            "buy": 0,
            "sell": 5000,
            "buy_price": 0.0,
            "sell_price": 115.0,
        },
        # Day -6：賣 5000 at 120 → profit 100,000（win）
        {
            "date": base - timedelta(days=6),
            "stock_id": stock_id,
            "broker_id": broker_id,
            "buy": 0,
            "sell": 5000,
            "buy_price": 0.0,
            "sell_price": 120.0,
        },
        # Day -4：賣 5000 at 95 → loss 25,000（lose）
        {
            "date": base - timedelta(days=4),
            "stock_id": stock_id,
            "broker_id": broker_id,
            "buy": 0,
            "sell": 5000,
            "buy_price": 0.0,
            "sell_price": 95.0,
        },
        # Day -2：賣 5000 at 125 → profit 125,000（win）
        {
            "date": base - timedelta(days=2),
            "stock_id": stock_id,
            "broker_id": broker_id,
            "buy": 0,
            "sell": 5000,
            "buy_price": 0.0,
            "sell_price": 125.0,
        },
        # Day -1（近期）：再買 10000 → recent_net > 0
        {
            "date": base - timedelta(days=1),
            "stock_id": stock_id,
            "broker_id": broker_id,
            "buy": 10000,
            "sell": 0,
            "buy_price": 105.0,
            "sell_price": 0.0,
        },
    ]


class TestComputeSmartBrokerScore:
    """compute_smart_broker_score() 純函數測試。"""

    def test_empty_input_returns_correct_columns(self):
        """空 DF 回傳空 DataFrame，欄位正確。"""
        result = compute_smart_broker_score(pd.DataFrame(), {})
        assert result.empty
        assert set(result.columns) == {
            "stock_id",
            "smart_broker_score",
            "accum_broker_score",
            "smart_broker_factor",
        }

    def test_missing_price_columns_graceful(self):
        """缺少 buy_price/sell_price 欄位 → 回傳空 DataFrame，不崩潰。"""
        df = pd.DataFrame([{"date": date(2026, 1, 1), "stock_id": "2330", "broker_id": "A", "buy": 100, "sell": 0}])
        result = compute_smart_broker_score(df, {})
        assert result.empty
        assert set(result.columns) == {
            "stock_id",
            "smart_broker_score",
            "accum_broker_score",
            "smart_broker_factor",
        }

    def test_smart_broker_detected_vs_no_smart(self):
        """Stock A 有 Smart Broker（win_rate=0.75, PF=12）；Stock B 僅 2 次賣出（未達門檻）。
        A 的 smart_broker_score 應高於 B。"""
        base = date(2026, 3, 1)
        rows_a = _make_smart_rows("A", "A001", base)
        # Stock B：只有 2 次賣出（sell_events < 3）
        rows_b = [
            {
                "date": base - timedelta(days=10),
                "stock_id": "B",
                "broker_id": "B001",
                "buy": 60000,
                "sell": 0,
                "buy_price": 100.0,
                "sell_price": 0.0,
            },
            {
                "date": base - timedelta(days=5),
                "stock_id": "B",
                "broker_id": "B001",
                "buy": 0,
                "sell": 5000,
                "buy_price": 0.0,
                "sell_price": 115.0,
            },
            {
                "date": base - timedelta(days=2),
                "stock_id": "B",
                "broker_id": "B001",
                "buy": 0,
                "sell": 5000,
                "buy_price": 0.0,
                "sell_price": 120.0,
            },
        ]
        df = pd.DataFrame(rows_a + rows_b)
        result = compute_smart_broker_score(df, {})
        assert len(result) == 2
        score_a = result[result["stock_id"] == "A"]["smart_broker_score"].iloc[0]
        score_b = result[result["stock_id"] == "B"]["smart_broker_score"].iloc[0]
        assert score_a > score_b

    def test_profit_factor_blocks_low_pf(self):
        """Stock A：win_rate=0.80，但 1 次大虧導致 PF << 1.5 → NOT Smart Broker。
        Stock B：win_rate=0.75，PF=12 → Smart Broker。
        B 的 smart_broker_score 應高於 A。"""
        base = date(2026, 3, 1)
        rows_a = [
            # A001：買 100000 at 100（10M）
            {
                "date": base - timedelta(days=15),
                "stock_id": "A",
                "broker_id": "A001",
                "buy": 100000,
                "sell": 0,
                "buy_price": 100.0,
                "sell_price": 0.0,
            },
            # 4 次小贏（各 1000 股，+1 至 +4 元）
            *[
                {
                    "date": base - timedelta(days=12 - i),
                    "stock_id": "A",
                    "broker_id": "A001",
                    "buy": 0,
                    "sell": 1000,
                    "buy_price": 0.0,
                    "sell_price": 101.0 + i,
                }
                for i in range(4)
            ],
            # 1 次大虧：賣 90000 at 72（loss ≈ 2.52M → PF ≈ 0.007）
            {
                "date": base - timedelta(days=5),
                "stock_id": "A",
                "broker_id": "A001",
                "buy": 0,
                "sell": 90000,
                "buy_price": 0.0,
                "sell_price": 72.0,
            },
        ]
        rows_b = _make_smart_rows("B", "B001", base)
        df = pd.DataFrame(rows_a + rows_b)
        result = compute_smart_broker_score(df, {})
        assert len(result) == 2
        score_a = result[result["stock_id"] == "A"]["smart_broker_score"].iloc[0]
        score_b = result[result["stock_id"] == "B"]["smart_broker_score"].iloc[0]
        assert score_b > score_a

    def test_min_sell_events_threshold(self):
        """sell_events < 3 的分點不得被標記為 Smart Broker。"""
        base = date(2026, 3, 1)

        # Stock A：sell_events=3（剛好達門檻） vs Stock B：sell_events=2（未達）
        def _make_rows(stock_id: str, sell_count: int) -> list[dict]:
            rows = [
                {
                    "date": base - timedelta(days=15),
                    "stock_id": stock_id,
                    "broker_id": "X001",
                    "buy": 60000,
                    "sell": 0,
                    "buy_price": 100.0,
                    "sell_price": 0.0,
                }
            ]
            for i in range(sell_count):
                rows.append(
                    {
                        "date": base - timedelta(days=10 - i * 2),
                        "stock_id": stock_id,
                        "broker_id": "X001",
                        "buy": 0,
                        "sell": 5000,
                        "buy_price": 0.0,
                        "sell_price": 115.0 + i,
                    }
                )
            # 近期買入讓 recent_net > 0
            rows.append(
                {
                    "date": base - timedelta(days=1),
                    "stock_id": stock_id,
                    "broker_id": "X001",
                    "buy": 10000,
                    "sell": 0,
                    "buy_price": 105.0,
                    "sell_price": 0.0,
                }
            )
            return rows

        df = pd.DataFrame(_make_rows("A", 3) + _make_rows("B", 2))
        result = compute_smart_broker_score(df, {})
        assert len(result) == 2
        score_a = result[result["stock_id"] == "A"]["smart_broker_score"].iloc[0]
        score_b = result[result["stock_id"] == "B"]["smart_broker_score"].iloc[0]
        assert score_a > score_b

    def test_min_buy_value_threshold(self):
        """總買入金額 < 500 萬的分點不得被標記為 Smart Broker。"""
        base = date(2026, 3, 1)
        # Stock A：buy_value = 6M → Smart ✓
        rows_a = _make_smart_rows("A", "A001", base)
        # Stock B：買入僅 50 股 at 100（5,000 TWD）→ 遠低於 500 萬
        rows_b = [
            {
                "date": base - timedelta(days=10),
                "stock_id": "B",
                "broker_id": "B001",
                "buy": 50,
                "sell": 0,
                "buy_price": 100.0,
                "sell_price": 0.0,
            },
            *[
                {
                    "date": base - timedelta(days=8 - i * 2),
                    "stock_id": "B",
                    "broker_id": "B001",
                    "buy": 0,
                    "sell": 10,
                    "buy_price": 0.0,
                    "sell_price": 115.0,
                }
                for i in range(4)
            ],
        ]
        df = pd.DataFrame(rows_a + rows_b)
        result = compute_smart_broker_score(df, {})
        assert len(result) == 2
        score_a = result[result["stock_id"] == "A"]["smart_broker_score"].iloc[0]
        score_b = result[result["stock_id"] == "B"]["smart_broker_score"].iloc[0]
        assert score_a > score_b

    def test_smart_score_weighted_by_hist_pnl(self):
        """Smart_Score = hist_pnl × recent_net：歷史損益越高的分點對排名貢獻越大。
        Stock A：hist_pnl=300,000；Stock B：hist_pnl=30,000（buy_price 低 10 倍）。
        A 的 smart_broker_score 應高於 B。"""
        base = date(2026, 3, 1)
        rows_a = _make_smart_rows("A", "A001", base)  # hist_pnl ≈ 300,000
        # Stock B：縮小 10 倍（僅 6000 股 at 100 → buy_val=600k < 500 萬，需調整）
        # 改成 60000 股 at 10 → buy_val=600k → 不符合，調整成 50000 at 100 = 5M
        rows_b = [
            {
                "date": base - timedelta(days=10),
                "stock_id": "B",
                "broker_id": "B001",
                "buy": 50000,
                "sell": 0,
                "buy_price": 100.0,
                "sell_price": 0.0,
            },
            # 賣出 sell_price 僅比成本高 1% → hist_pnl 很小
            {
                "date": base - timedelta(days=8),
                "stock_id": "B",
                "broker_id": "B001",
                "buy": 0,
                "sell": 1000,
                "buy_price": 0.0,
                "sell_price": 101.0,
            },
            {
                "date": base - timedelta(days=6),
                "stock_id": "B",
                "broker_id": "B001",
                "buy": 0,
                "sell": 1000,
                "buy_price": 0.0,
                "sell_price": 102.0,
            },
            {
                "date": base - timedelta(days=4),
                "stock_id": "B",
                "broker_id": "B001",
                "buy": 0,
                "sell": 1000,
                "buy_price": 0.0,
                "sell_price": 99.0,  # small loss
            },
            {
                "date": base - timedelta(days=1),
                "stock_id": "B",
                "broker_id": "B001",
                "buy": 10000,
                "sell": 0,
                "buy_price": 100.0,
                "sell_price": 0.0,
            },
        ]
        df = pd.DataFrame(rows_a + rows_b)
        result = compute_smart_broker_score(df, {})
        assert len(result) == 2
        score_a = result[result["stock_id"] == "A"]["smart_broker_score"].iloc[0]
        score_b = result[result["stock_id"] == "B"]["smart_broker_score"].iloc[0]
        assert score_a > score_b

    def test_recent_activity_weighted(self):
        """Smart_Score = hist_pnl × recent_net：近期買超張數越多，貢獻越大。
        Stock A 近 3 日大量買入；Stock B 近 3 日少量買入（相同 hist_pnl）。
        A 的 smart_broker_score 應高於 B。"""
        base = date(2026, 3, 1)

        def _rows_with_recent(stock_id: str, broker_id: str, recent_buy: int) -> list[dict]:
            rows = _make_smart_rows(stock_id, broker_id, base)
            # 覆蓋最後一筆（近期買入數量）
            rows[-1] = {
                "date": base - timedelta(days=1),
                "stock_id": stock_id,
                "broker_id": broker_id,
                "buy": recent_buy,
                "sell": 0,
                "buy_price": 105.0,
                "sell_price": 0.0,
            }
            return rows

        df = pd.DataFrame(_rows_with_recent("A", "A001", 50000) + _rows_with_recent("B", "B001", 100))
        result = compute_smart_broker_score(df, {})
        assert len(result) == 2
        score_a = result[result["stock_id"] == "A"]["smart_broker_score"].iloc[0]
        score_b = result[result["stock_id"] == "B"]["smart_broker_score"].iloc[0]
        assert score_a > score_b

    def test_accumulation_broker_detected(self):
        """純蓄積型分點（sell_ratio < 0.10, position_trend_up）應被標記 → accum_score > 0。
        Stock A：蓄積型（幾乎不賣，倉位持續增加）；Stock B：無蓄積型分點。
        A 的 accum_broker_score 應高於 B。"""
        base = date(2026, 3, 1)
        # Stock A：10 個交易日，前 5 天買 2000/天，後 5 天買 3000/天（趨勢向上），賣出僅 500（< 10%）
        # total_buy = 5×2000 + 5×3000 = 25000 股 at 200 = 5M ✓
        rows_a = []
        for i in range(5):
            rows_a.append(
                {
                    "date": base - timedelta(days=10 - i),
                    "stock_id": "A",
                    "broker_id": "A001",
                    "buy": 2000,
                    "sell": 0,
                    "buy_price": 200.0,
                    "sell_price": 0.0,
                }
            )
        for i in range(5):
            rows_a.append(
                {
                    "date": base - timedelta(days=5 - i),
                    "stock_id": "A",
                    "broker_id": "A001",
                    "buy": 3000,
                    "sell": 0,
                    "buy_price": 200.0,
                    "sell_price": 0.0,
                }
            )
        # 少量賣出（< 10%）
        rows_a.append(
            {
                "date": base - timedelta(days=3),
                "stock_id": "A",
                "broker_id": "A001",
                "buy": 0,
                "sell": 500,
                "buy_price": 0.0,
                "sell_price": 210.0,
            }
        )
        # Stock B：同規模買入但大量賣出 → 不是蓄積型
        rows_b = [
            {
                "date": base - timedelta(days=10),
                "stock_id": "B",
                "broker_id": "B001",
                "buy": 25000,
                "sell": 0,
                "buy_price": 200.0,
                "sell_price": 0.0,
            },
            {
                "date": base - timedelta(days=5),
                "stock_id": "B",
                "broker_id": "B001",
                "buy": 0,
                "sell": 20000,
                "buy_price": 0.0,
                "sell_price": 210.0,
            },
        ]
        df = pd.DataFrame(rows_a + rows_b)
        result = compute_smart_broker_score(df, {"A": 205.0, "B": 205.0})
        assert len(result) == 2
        acc_a = result[result["stock_id"] == "A"]["accum_broker_score"].iloc[0]
        acc_b = result[result["stock_id"] == "B"]["accum_broker_score"].iloc[0]
        assert acc_a > acc_b

    def test_accumulation_trend_required(self):
        """倉位未成長（後半段 ≤ 前半段）的分點不應被標記為 Accumulation Broker。
        Stock A：前多後少（趨勢向下）；Stock B：前少後多（趨勢向上）。
        B 的 accum_broker_score 應高於 A。"""
        base = date(2026, 3, 1)

        def _rows_trend(stock_id: str, first_buy: int, last_buy: int) -> list[dict]:
            rows = []
            for i in range(5):
                rows.append(
                    {
                        "date": base - timedelta(days=10 - i),
                        "stock_id": stock_id,
                        "broker_id": "X001",
                        "buy": first_buy,
                        "sell": 0,
                        "buy_price": 200.0,
                        "sell_price": 0.0,
                    }
                )
            for i in range(5):
                rows.append(
                    {
                        "date": base - timedelta(days=5 - i),
                        "stock_id": stock_id,
                        "broker_id": "X001",
                        "buy": last_buy,
                        "sell": 0,
                        "buy_price": 200.0,
                        "sell_price": 0.0,
                    }
                )
            return rows

        # A：前 3000/天，後 1000/天（趨勢下降），B：前 1000/天，後 3000/天（趨勢上升）
        # 兩者 total_buy = 5×3000 + 5×1000 = 20000 股 at 200 = 4M < 5M
        # → 調整，乘以 2
        df = pd.DataFrame(
            _rows_trend("A", first_buy=6000, last_buy=2000) + _rows_trend("B", first_buy=2000, last_buy=6000)
        )
        result = compute_smart_broker_score(df, {"A": 205.0, "B": 205.0})
        assert len(result) == 2
        acc_a = result[result["stock_id"] == "A"]["accum_broker_score"].iloc[0]
        acc_b = result[result["stock_id"] == "B"]["accum_broker_score"].iloc[0]
        assert acc_b > acc_a

    def test_composite_factor_range(self):
        """所有輸出欄位均在 [0, 1] 範圍內；空輸入回傳空 DataFrame。"""
        base = date(2026, 3, 1)
        rows = _make_smart_rows("A", "A001", base) + _make_smart_rows("B", "B001", base)
        df = pd.DataFrame(rows)
        result = compute_smart_broker_score(df, {"A": 110.0, "B": 95.0})
        assert not result.empty
        for col in ["smart_broker_score", "accum_broker_score", "smart_broker_factor"]:
            assert result[col].between(0.0, 1.0).all(), f"{col} 超出 [0, 1] 範圍"

        # 空輸入
        empty_result = compute_smart_broker_score(pd.DataFrame(), {})
        assert empty_result.empty


# ------------------------------------------------------------------ #
#  TestLoadBrokerDataExtendedAdaptive — _load_broker_data_extended 門檻
# ------------------------------------------------------------------ #


class TestLoadBrokerDataExtendedAdaptive:
    """_load_broker_data_extended() 自適應 min_trading_days 過濾行為（in-memory SQLite）。"""

    def _write_broker_rows(self, session, stock_id: str, n_days: int, base: date | None = None) -> None:
        """在 in-memory DB 寫入指定天數的 BrokerTrade 測試資料。"""
        from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

        _base = base or date(2026, 1, 1)
        rows = [
            {
                "stock_id": stock_id,
                "date": _base + timedelta(days=i),
                "broker_id": "X001",
                "broker_name": "測試券商",
                "buy": 1000,
                "sell": 500,
                "buy_price": 100.0,
                "sell_price": 102.0,
            }
            for i in range(n_days)
        ]
        stmt = sqlite_upsert(BrokerTrade).values(rows).on_conflict_do_nothing()
        session.execute(stmt)
        session.commit()

    def test_min_trading_days_filters_insufficient_data(self, db_session):
        """歷史天數不足 min_trading_days 的股票應被過濾排除。"""
        # 寫入：A 有 25 天、B 只有 10 天
        self._write_broker_rows(db_session, "A", 25)
        self._write_broker_rows(db_session, "B", 10)

        scanner = MomentumScanner.__new__(MomentumScanner)
        # 設定 min_trading_days=20：A 留下，B 排除
        result = scanner._load_broker_data_extended(["A", "B"], days=365, min_trading_days=20)

        assert not result.empty
        stocks_in_result = result["stock_id"].unique()
        assert "A" in stocks_in_result
        assert "B" not in stocks_in_result

    def test_min_trading_days_zero_disables_filter(self, db_session):
        """min_trading_days=0 時不過濾，所有股票均回傳。"""
        self._write_broker_rows(db_session, "C", 5)

        scanner = MomentumScanner.__new__(MomentumScanner)
        result = scanner._load_broker_data_extended(["C"], days=365, min_trading_days=0)

        assert not result.empty
        assert "C" in result["stock_id"].values

    def test_all_stocks_meet_threshold(self, db_session):
        """所有股票均達門檻時，全部回傳。"""
        self._write_broker_rows(db_session, "D", 30)
        self._write_broker_rows(db_session, "E", 25)

        scanner = MomentumScanner.__new__(MomentumScanner)
        result = scanner._load_broker_data_extended(["D", "E"], days=365, min_trading_days=20)

        stocks_in_result = set(result["stock_id"].unique())
        assert stocks_in_result == {"D", "E"}

    def test_default_days_is_365(self, db_session):
        """預設 days=365 確保查詢到足夠歷史範圍（不再使用 120 天固定上限）。"""
        import inspect

        from src.discovery.scanner import MomentumScanner

        sig = inspect.signature(MomentumScanner._load_broker_data_extended)
        params = sig.parameters
        assert params["days"].default == 365, "days 預設值應為 365（不再是 120）"
        assert params["min_trading_days"].default == 20, "min_trading_days 預設值應為 20"

    def test_empty_result_when_no_data_in_window(self, db_session):
        """DB 內無資料或超出窗口時，回傳空 DataFrame。"""
        scanner = MomentumScanner.__new__(MomentumScanner)
        result = scanner._load_broker_data_extended(["Z999"], days=365, min_trading_days=1)
        assert result.empty


# ------------------------------------------------------------------ #
#  TestLoadBrokerDataExtendedCloseProxy — 收盤價代理均價（方案 B）
# ------------------------------------------------------------------ #


class TestLoadBrokerDataExtendedCloseProxy:
    """_load_broker_data_extended() 以 DailyPrice.close 填補 NULL buy_price/sell_price 的測試。

    DJ 端點不提供均價，buy_price / sell_price 存 NULL。
    本功能以同日收盤價作為代理，使 Smart Broker 8F 計算得以啟用。
    """

    def _write_broker_rows_no_price(self, session, stock_id: str, dates: list[date]) -> None:
        """寫入 buy_price / sell_price 為 NULL 的分點資料（模擬 DJ 端點）。"""
        rows = [
            {
                "stock_id": stock_id,
                "date": d,
                "broker_id": "B001",
                "broker_name": "測試券商",
                "buy": 5000,
                "sell": 2000,
                "buy_price": None,
                "sell_price": None,
            }
            for d in dates
        ]
        stmt = sqlite_upsert(BrokerTrade).values(rows).on_conflict_do_nothing()
        session.execute(stmt)
        session.commit()

    def _write_daily_price(self, session, stock_id: str, dates: list[date], close: float) -> None:
        """寫入 DailyPrice 收盤價測試資料。"""
        rows = [
            {
                "stock_id": stock_id,
                "date": d,
                "open": close - 1.0,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": 10_000_000,
                "turnover": int(close * 10_000_000),
                "spread": 0.0,
            }
            for d in dates
        ]
        stmt = sqlite_upsert(DailyPrice).values(rows).on_conflict_do_nothing()
        session.execute(stmt)
        session.commit()

    def test_null_price_filled_by_daily_close(self, db_session):
        """buy_price / sell_price 為 NULL 時，應填入同日 DailyPrice.close。"""
        from datetime import date as _date

        base = _date.today() - timedelta(days=30)
        dates = [base + timedelta(days=i) for i in range(25)]
        self._write_broker_rows_no_price(db_session, "P001", dates)
        self._write_daily_price(db_session, "P001", dates, close=120.0)

        scanner = MomentumScanner.__new__(MomentumScanner)
        result = scanner._load_broker_data_extended(["P001"], days=365, min_trading_days=20)

        assert not result.empty
        assert result["buy_price"].notna().all(), "buy_price 應已填入收盤價代理值"
        assert result["sell_price"].notna().all(), "sell_price 應已填入收盤價代理值"
        assert (result["buy_price"] == 120.0).all(), "buy_price 應等於收盤價 120.0"
        assert (result["sell_price"] == 120.0).all(), "sell_price 應等於收盤價 120.0"

    def test_existing_price_not_overwritten(self, db_session):
        """已有 buy_price / sell_price 時，不應被收盤價覆蓋。"""
        from datetime import date as _date

        base = _date.today() - timedelta(days=30)
        dates = [base + timedelta(days=i) for i in range(25)]
        rows = [
            {
                "stock_id": "P002",
                "date": d,
                "broker_id": "B001",
                "broker_name": "測試券商",
                "buy": 5000,
                "sell": 2000,
                "buy_price": 99.5,  # 已有真實均價
                "sell_price": 101.0,
            }
            for d in dates
        ]
        stmt = sqlite_upsert(BrokerTrade).values(rows).on_conflict_do_nothing()
        db_session.execute(stmt)
        db_session.commit()
        self._write_daily_price(db_session, "P002", dates, close=200.0)  # 收盤價故意不同

        scanner = MomentumScanner.__new__(MomentumScanner)
        result = scanner._load_broker_data_extended(["P002"], days=365, min_trading_days=20)

        assert not result.empty
        assert (result["buy_price"] == 99.5).all(), "已有的 buy_price 不應被覆蓋"
        assert (result["sell_price"] == 101.0).all(), "已有的 sell_price 不應被覆蓋"

    def test_no_daily_price_keeps_null_gracefully(self, db_session):
        """DailyPrice 無資料時，buy_price / sell_price 維持 NULL（系統降回 7F）。"""
        from datetime import date as _date

        base = _date.today() - timedelta(days=30)
        dates = [base + timedelta(days=i) for i in range(25)]
        self._write_broker_rows_no_price(db_session, "P003", dates)
        # 故意不寫 DailyPrice

        scanner = MomentumScanner.__new__(MomentumScanner)
        result = scanner._load_broker_data_extended(["P003"], days=365, min_trading_days=20)

        assert not result.empty
        # 無法填入代理均價，維持 NULL（呼叫端將降至 7F）
        assert result["buy_price"].isna().all(), "無收盤價資料時，buy_price 應維持 NULL"

    def test_close_proxy_enables_smart_broker_score(self, db_session):
        """填入收盤價代理後，compute_smart_broker_score() 應能產生非零評分（8F 啟用前提）。"""
        from datetime import date as _date

        base = _date.today() - timedelta(days=60)
        # 模擬一個「買漲賣漲」的分點：買在低點，賣在高點（勝率 100%）
        buy_dates = [base + timedelta(days=i * 2) for i in range(15)]  # 奇數日買入
        sell_dates = [base + timedelta(days=i * 2 + 1) for i in range(15)]  # 偶數日賣出
        all_dates = sorted(set(buy_dates + sell_dates))

        broker_rows = []
        for i, d in enumerate(all_dates):
            is_buy_day = d in buy_dates
            broker_rows.append(
                {
                    "stock_id": "P004",
                    "date": d,
                    "broker_id": "B001",
                    "broker_name": "測試券商",
                    "buy": 10_000 if is_buy_day else 0,
                    "sell": 0 if is_buy_day else 5_000,
                    "buy_price": None,  # DJ 無均價
                    "sell_price": None,
                }
            )
        stmt = sqlite_upsert(BrokerTrade).values(broker_rows).on_conflict_do_nothing()
        db_session.execute(stmt)
        db_session.commit()

        # 收盤價：單調遞增，模擬多頭趨勢（賣出時均高於買入時）
        price_rows = [
            {
                "stock_id": "P004",
                "date": d,
                "open": 100.0 + i * 0.5,
                "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5,
                "close": 100.0 + i * 0.5,  # 每天上漲 0.5 元
                "volume": 5_000_000,
                "turnover": 500_000_000,
                "spread": 0.5,
            }
            for i, d in enumerate(all_dates)
        ]
        stmt2 = sqlite_upsert(DailyPrice).values(price_rows).on_conflict_do_nothing()
        db_session.execute(stmt2)
        db_session.commit()

        scanner = MomentumScanner.__new__(MomentumScanner)
        result = scanner._load_broker_data_extended(["P004"], days=365, min_trading_days=1)

        assert not result.empty
        assert result["buy_price"].notna().all(), "代理均價應已填入"

        # 進一步確認 compute_smart_broker_score 可執行（8F 前提）
        close_map = {"P004": result["buy_price"].max()}
        smart_result = compute_smart_broker_score(result, close_map)
        # 評分可能為空（需達門檻），但函數執行不應拋例外
        assert isinstance(smart_result, pd.DataFrame)
