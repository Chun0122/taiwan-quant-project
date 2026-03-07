"""借券賣出（Securities Borrowing and Lending）整合測試。

測試項目：
- fetch_twse_sbl: TWSE TWT96U 欄位映射 + sbl_change 計算（mock HTTP）
- SecuritiesLending ORM: 寫入 + 唯一鍵衝突（in-memory SQLite）
- compute_sbl_score: 最新日篩選 + 欄位完整（純函數）
- MomentumScanner 6-factor: 6-factor 啟用、降級、sbl_rank 逆向（_compute_chip_scores 單元）
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

# 模組層級 ORM import：確保 Base.metadata 在 in_memory_engine.create_all() 前已包含全表
from src.data.schema import SecuritiesLending  # noqa: F401
from src.discovery.scanner import MomentumScanner, compute_sbl_score

# ------------------------------------------------------------------ #
#  TestFetchTwseSbl — mock HTTP 測試
# ------------------------------------------------------------------ #


class TestFetchTwseSbl:
    """fetch_twse_sbl() 欄位映射與邊界情況測試。"""

    def _mock_resp(self, monkeypatch, json_data: dict):
        """設定 requests.get mock 回傳指定 JSON。"""
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = json_data
        monkeypatch.setattr("src.data.twse_fetcher.requests.get", lambda *a, **kw: mock)
        monkeypatch.setattr("src.data.twse_fetcher.time.sleep", lambda x: None)

    def test_parses_valid_response(self, monkeypatch):
        """正常回傳格式：欄位映射正確，sbl_change = sbl_balance - sbl_prev_balance。"""
        from src.data.twse_fetcher import fetch_twse_sbl

        self._mock_resp(
            monkeypatch,
            {
                "stat": "OK",
                "fields": ["證券代號", "證券名稱", "當日借券成交量", "次一交易日可回補", "借券餘額", "前日借券餘額"],
                "data": [
                    ["2330", "台積電", "1,000", "500", "50,000", "48,000"],
                    ["2317", "鴻海", "200", "100", "10,000", "9,500"],
                ],
            },
        )
        df = fetch_twse_sbl(date(2026, 3, 5))
        assert len(df) == 2
        assert set(df.columns) >= {
            "date",
            "stock_id",
            "sbl_sell_volume",
            "sbl_balance",
            "sbl_prev_balance",
            "sbl_change",
        }

        row = df[df["stock_id"] == "2330"].iloc[0]
        assert row["sbl_sell_volume"] == 1000
        assert row["sbl_balance"] == 50000
        assert row["sbl_prev_balance"] == 48000
        assert row["sbl_change"] == 2000  # 50000 - 48000

    def test_stat_not_ok_returns_empty(self, monkeypatch):
        """stat != 'OK' 時（假日或無資料）回傳空 DataFrame。"""
        from src.data.twse_fetcher import fetch_twse_sbl

        self._mock_resp(monkeypatch, {"stat": "很抱歉，找不到符合條件的資料！"})
        df = fetch_twse_sbl(date(2026, 3, 7))
        assert df.empty

    def test_sbl_change_calculation(self, monkeypatch):
        """sbl_change = sbl_balance - sbl_prev_balance 計算正確（含負值情況）。"""
        from src.data.twse_fetcher import fetch_twse_sbl

        self._mock_resp(
            monkeypatch,
            {
                "stat": "OK",
                "data": [
                    ["1234", "測試股", "0", "0", "3,000", "5,000"],  # 餘額下降 → change < 0
                ],
            },
        )
        df = fetch_twse_sbl(date(2026, 3, 5))
        assert len(df) == 1
        assert df.iloc[0]["sbl_change"] == -2000  # 3000 - 5000

    def test_filters_non_4digit_stock_ids(self, monkeypatch):
        """非 4 碼純數字代號（如指數）應被過濾。"""
        from src.data.twse_fetcher import fetch_twse_sbl

        self._mock_resp(
            monkeypatch,
            {
                "stat": "OK",
                "data": [
                    ["2330", "台積電", "1,000", "500", "50,000", "48,000"],
                    ["IX0001", "指數", "0", "0", "0", "0"],
                    ["00878", "ETF", "100", "50", "1,000", "900"],
                ],
            },
        )
        df = fetch_twse_sbl(date(2026, 3, 5))
        # 只有 2330 符合 4 碼純數字
        assert len(df) == 1
        assert df.iloc[0]["stock_id"] == "2330"

    def test_request_failure_returns_empty(self, monkeypatch):
        """HTTP 請求失敗應回傳空 DataFrame，不拋例外。"""
        from src.data.twse_fetcher import fetch_twse_sbl

        def _raise(*a, **kw):
            raise ConnectionError("無法連線至 TWSE")

        monkeypatch.setattr("src.data.twse_fetcher.requests.get", _raise)
        monkeypatch.setattr("src.data.twse_fetcher.time.sleep", lambda x: None)
        df = fetch_twse_sbl(date(2026, 3, 5))
        assert df.empty


# ------------------------------------------------------------------ #
#  TestSecuritiesLendingORM — in-memory SQLite 測試
# ------------------------------------------------------------------ #


class TestSecuritiesLendingORM:
    """SecuritiesLending ORM CRUD 測試（in-memory SQLite）。"""

    def test_insert_and_query(self, db_session):
        """寫入一筆資料後可查詢回來。"""
        dt = date(2026, 3, 5)
        entry = SecuritiesLending(
            stock_id="2330",
            date=dt,
            sbl_sell_volume=1000,
            sbl_balance=50000,
            sbl_prev_balance=48000,
            sbl_change=2000,
        )
        db_session.add(entry)
        db_session.flush()

        result = db_session.query(SecuritiesLending).filter_by(stock_id="2330").first()
        assert result is not None
        assert result.sbl_balance == 50000
        assert result.sbl_change == 2000

    def test_unique_constraint_on_conflict_do_nothing(self, db_session):
        """相同 stock_id + date 衝突時 on_conflict_do_nothing 略過，不報錯。"""
        dt = date(2026, 3, 5)
        record1 = {
            "stock_id": "2317",
            "date": dt,
            "sbl_sell_volume": 200,
            "sbl_balance": 10000,
            "sbl_prev_balance": 9500,
            "sbl_change": 500,
        }
        record2 = {
            "stock_id": "2317",
            "date": dt,
            "sbl_sell_volume": 999,
            "sbl_balance": 99999,
            "sbl_prev_balance": 88888,
            "sbl_change": 11111,
        }

        stmt1 = sqlite_upsert(SecuritiesLending).values([record1])
        stmt1 = stmt1.on_conflict_do_nothing(index_elements=["stock_id", "date"])
        db_session.execute(stmt1)
        db_session.flush()

        stmt2 = sqlite_upsert(SecuritiesLending).values([record2])
        stmt2 = stmt2.on_conflict_do_nothing(index_elements=["stock_id", "date"])
        db_session.execute(stmt2)
        db_session.flush()

        rows = db_session.query(SecuritiesLending).filter_by(stock_id="2317").all()
        assert len(rows) == 1
        # 第一筆值保留
        assert rows[0].sbl_balance == 10000

    def test_nullable_fields_allow_none(self, db_session):
        """nullable 欄位可為 None（部分欄位無資料的情況）。"""
        dt = date(2026, 3, 5)
        entry = SecuritiesLending(
            stock_id="6505",
            date=dt,
            sbl_sell_volume=None,
            sbl_balance=None,
            sbl_prev_balance=None,
            sbl_change=None,
        )
        db_session.add(entry)
        db_session.flush()

        result = db_session.query(SecuritiesLending).filter_by(stock_id="6505").first()
        assert result is not None
        assert result.sbl_balance is None
        assert result.sbl_change is None


# ------------------------------------------------------------------ #
#  TestComputeSblScore — 純函數測試
# ------------------------------------------------------------------ #


class TestComputeSblScore:
    """compute_sbl_score() 純函數測試。"""

    def test_empty_df_returns_empty_with_correct_columns(self):
        """空 DF 回傳空 DataFrame，欄位 [stock_id, sbl_balance, sbl_change]。"""
        result = compute_sbl_score(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == ["stock_id", "sbl_balance", "sbl_change"]

    def test_only_latest_date_returned(self):
        """有多日資料時，只取最新一日的資料。"""
        dt_old = date(2026, 3, 4)
        dt_new = date(2026, 3, 5)
        df = pd.DataFrame(
            [
                {"date": dt_old, "stock_id": "2330", "sbl_balance": 10000, "sbl_change": 500},
                {"date": dt_new, "stock_id": "2330", "sbl_balance": 12000, "sbl_change": 2000},
            ]
        )
        result = compute_sbl_score(df)
        assert len(result) == 1
        assert result.iloc[0]["sbl_balance"] == 12000

    def test_correct_columns_returned(self):
        """回傳欄位必須包含 [stock_id, sbl_balance, sbl_change]。"""
        dt = date(2026, 3, 5)
        df = pd.DataFrame(
            [
                {"date": dt, "stock_id": "2330", "sbl_balance": 50000, "sbl_change": 2000},
                {"date": dt, "stock_id": "2317", "sbl_balance": 10000, "sbl_change": -500},
            ]
        )
        result = compute_sbl_score(df)
        assert "stock_id" in result.columns
        assert "sbl_balance" in result.columns
        assert "sbl_change" in result.columns

    def test_multiple_stocks_latest_date(self):
        """多支股票，只取最新日，各股資料正確。"""
        dt_prev = date(2026, 3, 4)
        dt_now = date(2026, 3, 5)
        df = pd.DataFrame(
            [
                {"date": dt_prev, "stock_id": "2330", "sbl_balance": 40000, "sbl_change": 1000},
                {"date": dt_now, "stock_id": "2330", "sbl_balance": 50000, "sbl_change": 10000},
                {"date": dt_prev, "stock_id": "2317", "sbl_balance": 8000, "sbl_change": -200},
                {"date": dt_now, "stock_id": "2317", "sbl_balance": 9000, "sbl_change": 1000},
            ]
        )
        result = compute_sbl_score(df)
        assert len(result) == 2
        bal_2330 = result[result["stock_id"] == "2330"]["sbl_balance"].iloc[0]
        bal_2317 = result[result["stock_id"] == "2317"]["sbl_balance"].iloc[0]
        assert bal_2330 == 50000
        assert bal_2317 == 9000

    def test_missing_required_columns_returns_empty(self):
        """缺少必要欄位時回傳空 DataFrame。"""
        df = pd.DataFrame([{"stock_id": "2330", "some_column": 100}])
        result = compute_sbl_score(df)
        assert result.empty


# ------------------------------------------------------------------ #
#  TestMomentumScannerSblFactor — 6-factor 路徑 + 降級測試
# ------------------------------------------------------------------ #


@pytest.fixture()
def momentum_scanner():
    return MomentumScanner(
        min_price=10,
        max_price=2000,
        min_volume=100_000,
        top_n_candidates=10,
    )


def _make_sbl_df(stock_ids: list[str], balances: list[int]) -> pd.DataFrame:
    """建立測試用 SBL DataFrame（最新一日，各股不同借券餘額）。"""
    dt = date.today()
    rows = [
        {"date": dt, "stock_id": sid, "sbl_balance": bal, "sbl_change": bal // 10}
        for sid, bal in zip(stock_ids, balances)
    ]
    return pd.DataFrame(rows)


def _make_holding_df_for_scanner(stock_ids: list[str]) -> pd.DataFrame:
    """建立測試用大戶持股 DataFrame。"""
    dt = date.today()
    rows = [{"date": dt, "stock_id": sid, "level": "400,001-600,000 Shares", "percent": 15.0} for sid in stock_ids]
    return pd.DataFrame(rows)


class TestMomentumScannerSblFactor:
    """MomentumScanner._compute_chip_scores() SBL 因子整合測試。"""

    def _make_inst_df(self, stock_ids: list[str]) -> pd.DataFrame:
        """建立法人資料（各股相同，避免干擾 SBL 測試）。"""
        dt = date.today() - timedelta(days=1)
        rows = [{"stock_id": sid, "date": dt, "name": "外資買賣超", "net": 100} for sid in stock_ids]
        return pd.DataFrame(rows)

    def _make_margin_df(self, stock_ids: list[str]) -> pd.DataFrame:
        """建立融資融券資料（各股相同）。"""
        dt = date.today() - timedelta(days=1)
        rows = [{"stock_id": sid, "date": dt, "margin_balance": 1000, "short_balance": 100} for sid in stock_ids]
        return pd.DataFrame(rows)

    def test_six_factor_path_activated(self, momentum_scanner, monkeypatch):
        """有 SBL + 融資融券 + 大戶持股時，啟用 6-factor，chip_score 正常計算。"""
        sids = ["1000", "1001", "1002"]
        df_inst = self._make_inst_df(sids)
        df_margin = self._make_margin_df(sids)

        # 模擬 _load_sbl_data 回傳 SBL 資料
        monkeypatch.setattr(
            momentum_scanner,
            "_load_sbl_data",
            lambda stock_ids: _make_sbl_df(sids, [50000, 10000, 30000]),
        )
        # 模擬 _load_holding_data 回傳大戶持股資料
        monkeypatch.setattr(
            momentum_scanner,
            "_load_holding_data",
            lambda stock_ids: _make_holding_df_for_scanner(sids),
        )

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, df_margin)
        assert len(result) == 3
        assert (result["chip_score"] >= 0).all()
        assert (result["chip_score"] <= 1.0).all()

    def test_no_sbl_falls_back_to_five_factor(self, momentum_scanner, monkeypatch):
        """無 SBL 資料時，降級至 5-factor（含券資比 + 大戶持股）。"""
        sids = ["1000", "1001"]
        df_inst = self._make_inst_df(sids)
        df_margin = self._make_margin_df(sids)

        # SBL 回傳空 DF → 降級
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: pd.DataFrame())
        # 大戶持股有資料
        monkeypatch.setattr(
            momentum_scanner,
            "_load_holding_data",
            lambda stock_ids: _make_holding_df_for_scanner(sids),
        )

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, df_margin)
        assert len(result) == 2
        assert (result["chip_score"] >= 0).all()
        assert (result["chip_score"] <= 1.0).all()

    def test_sbl_rank_inverted_high_balance_gets_lower_score(self, momentum_scanner, monkeypatch):
        """借券餘額高 → 空頭壓力大 → chip_score 較低（逆向因子驗證）。"""
        sids = ["low_sbl", "high_sbl"]  # low_sbl 借券少，high_sbl 借券多

        df_inst = pd.DataFrame(
            [
                # 兩股法人資料完全相同（排除法人干擾）
                {"stock_id": "low_sbl", "date": date.today() - timedelta(days=1), "name": "外資買賣超", "net": 100},
                {"stock_id": "high_sbl", "date": date.today() - timedelta(days=1), "name": "外資買賣超", "net": 100},
            ]
        )
        # 融資融券也相同
        df_margin = pd.DataFrame(
            [
                {
                    "stock_id": "low_sbl",
                    "date": date.today() - timedelta(days=1),
                    "margin_balance": 1000,
                    "short_balance": 100,
                },
                {
                    "stock_id": "high_sbl",
                    "date": date.today() - timedelta(days=1),
                    "margin_balance": 1000,
                    "short_balance": 100,
                },
            ]
        )

        # low_sbl 借券餘額 1000（低），high_sbl 借券餘額 100000（高）
        dt_today = date.today()
        sbl_data = pd.DataFrame(
            [
                {"date": dt_today, "stock_id": "low_sbl", "sbl_balance": 1000, "sbl_change": 100},
                {"date": dt_today, "stock_id": "high_sbl", "sbl_balance": 100000, "sbl_change": 5000},
            ]
        )
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: sbl_data)
        # 大戶持股也相同
        monkeypatch.setattr(
            momentum_scanner,
            "_load_holding_data",
            lambda _: pd.DataFrame(
                [
                    {"date": dt_today, "stock_id": "low_sbl", "level": "400,001-600,000 Shares", "percent": 15.0},
                    {"date": dt_today, "stock_id": "high_sbl", "level": "400,001-600,000 Shares", "percent": 15.0},
                ]
            ),
        )

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, df_margin)
        scores = result.set_index("stock_id")["chip_score"]
        # low_sbl 空頭壓力小 → 逆向評分高 → chip_score 應高於 high_sbl
        assert scores["low_sbl"] > scores["high_sbl"]

    def test_sbl_only_uses_four_factor(self, momentum_scanner, monkeypatch):
        """僅有 SBL（無融資融券、無大戶持股）→ 4-factor 路徑。"""
        sids = ["1000", "1001"]
        df_inst = self._make_inst_df(sids)

        dt = date.today()
        sbl_data = pd.DataFrame(
            [
                {"date": dt, "stock_id": "1000", "sbl_balance": 1000, "sbl_change": 100},
                {"date": dt, "stock_id": "1001", "sbl_balance": 50000, "sbl_change": 2000},
            ]
        )
        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: sbl_data)
        monkeypatch.setattr(momentum_scanner, "_load_holding_data", lambda _: pd.DataFrame())

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, None)
        assert len(result) == 2
        assert (result["chip_score"] >= 0).all()
        assert (result["chip_score"] <= 1.0).all()

    def test_no_data_falls_to_three_factor(self, momentum_scanner, monkeypatch):
        """無 SBL、無融資融券、無大戶持股時，使用 3-factor。"""
        sids = ["1000"]
        df_inst = self._make_inst_df(sids)

        monkeypatch.setattr(momentum_scanner, "_load_sbl_data", lambda _: pd.DataFrame())
        monkeypatch.setattr(momentum_scanner, "_load_holding_data", lambda _: pd.DataFrame())

        result = momentum_scanner._compute_chip_scores(sids, df_inst, None, None)
        assert len(result) == 1
        assert 0.0 <= result.iloc[0]["chip_score"] <= 1.0
