"""測試 src/data/io.py — 資料匯出/匯入模組。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.io import (
    CONFLICT_KEYS,
    TABLE_REGISTRY,
    export_table,
    import_table,
    list_tables,
    validate_import,
)
from src.data.schema import DailyPrice, StockInfo

# ── 純函數測試 ─────────────────────────────────────────────────


class TestTableRegistry:
    """TABLE_REGISTRY / CONFLICT_KEYS 完整性。"""

    def test_registry_has_all_tables(self):
        """確認 15 張表全部註冊。"""
        expected = {
            "daily_price",
            "institutional_investor",
            "margin_trading",
            "monthly_revenue",
            "stock_valuation",
            "dividend",
            "technical_indicator",
            "announcement",
            "financial_statement",
            "backtest_result",
            "trade",
            "portfolio_backtest_result",
            "stock_info",
            "portfolio_trade",
            "discovery_record",
        }
        assert set(TABLE_REGISTRY.keys()) == expected

    def test_conflict_keys_match_registry(self):
        """CONFLICT_KEYS 與 TABLE_REGISTRY 一一對應。"""
        assert set(CONFLICT_KEYS.keys()) == set(TABLE_REGISTRY.keys())

    def test_conflict_keys_not_empty(self):
        """每張表的衝突鍵不為空。"""
        for table_name, keys in CONFLICT_KEYS.items():
            assert len(keys) > 0, f"{table_name} 的衝突鍵為空"


class TestValidateImport:
    """validate_import 純函數測試。"""

    def test_valid_daily_price(self):
        """合法 DailyPrice DataFrame 無錯誤。"""
        df = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "date": ["2025-01-02"],
                "open": [600.0],
                "high": [610.0],
                "low": [595.0],
                "close": [605.0],
                "volume": [30000000],
                "turnover": [18000000000],
            }
        )
        errors = validate_import("daily_price", df)
        assert errors == []

    def test_missing_required_columns(self):
        """缺少必要欄位應報錯。"""
        df = pd.DataFrame({"stock_id": ["2330"], "date": ["2025-01-02"]})
        errors = validate_import("daily_price", df)
        assert len(errors) > 0
        assert "缺少必要欄位" in errors[0]

    def test_extra_columns_no_error(self):
        """多餘欄位不報錯。"""
        df = pd.DataFrame(
            {
                "stock_id": ["2330"],
                "date": ["2025-01-02"],
                "open": [600.0],
                "high": [610.0],
                "low": [595.0],
                "close": [605.0],
                "volume": [30000000],
                "turnover": [18000000000],
                "extra_col": ["ignored"],
            }
        )
        errors = validate_import("daily_price", df)
        assert errors == []

    def test_unknown_table(self):
        """不存在的表名應報錯。"""
        df = pd.DataFrame({"a": [1]})
        errors = validate_import("nonexistent_table", df)
        assert len(errors) > 0
        assert "不支援" in errors[0]

    def test_stock_info_required_columns(self):
        """StockInfo 只需 stock_id。"""
        df = pd.DataFrame({"stock_id": ["2330"]})
        errors = validate_import("stock_info", df)
        assert errors == []


# ── DB 整合測試 ────────────────────────────────────────────────
# 注意：db_session fixture 的 transaction rollback 在 SUT 內部 commit 後
# 無法完全隔離，因此每個測試使用獨特 stock_id/date 避免衝突。


class TestExportTable:
    """export_table 匯出測試（需 db_session）。"""

    def test_export_csv(self, db_session, tmp_path):
        """匯出 DailyPrice 為 CSV。"""
        db_session.add(
            DailyPrice(
                stock_id="9901",
                date=date(2025, 1, 2),
                open=600.0,
                high=610.0,
                low=595.0,
                close=605.0,
                volume=30000000,
                turnover=18000000000,
                spread=5.0,
            )
        )
        db_session.commit()

        out = tmp_path / "daily_price.csv"
        count = export_table("daily_price", str(out), fmt="csv", stocks=["9901"])
        assert count == 1

        df = pd.read_csv(out, encoding="utf-8-sig", dtype={"stock_id": str})
        assert "stock_id" in df.columns
        assert "id" not in df.columns  # 不含自增主鍵
        assert df.iloc[0]["stock_id"] == "9901"
        assert df.iloc[0]["close"] == 605.0

    def test_export_with_stock_filter(self, db_session, tmp_path):
        """--stocks 篩選功能。"""
        db_session.add_all(
            [
                DailyPrice(
                    stock_id="9902",
                    date=date(2025, 2, 3),
                    open=600.0,
                    high=610.0,
                    low=595.0,
                    close=605.0,
                    volume=30000000,
                    turnover=18000000000,
                ),
                DailyPrice(
                    stock_id="9903",
                    date=date(2025, 2, 3),
                    open=100.0,
                    high=105.0,
                    low=98.0,
                    close=102.0,
                    volume=5000000,
                    turnover=500000000,
                ),
            ]
        )
        db_session.commit()

        out = tmp_path / "filtered.csv"
        count = export_table("daily_price", str(out), stocks=["9902"])
        assert count == 1

        df = pd.read_csv(out, encoding="utf-8-sig", dtype={"stock_id": str})
        assert len(df) == 1
        assert df.iloc[0]["stock_id"] == "9902"

    def test_export_with_date_filter(self, db_session, tmp_path):
        """--start / --end 日期篩選。"""
        db_session.add_all(
            [
                DailyPrice(
                    stock_id="9904",
                    date=date(2025, 1, 10),
                    open=600.0,
                    high=610.0,
                    low=595.0,
                    close=605.0,
                    volume=30000000,
                    turnover=18000000000,
                ),
                DailyPrice(
                    stock_id="9904",
                    date=date(2025, 6, 15),
                    open=650.0,
                    high=660.0,
                    low=645.0,
                    close=655.0,
                    volume=25000000,
                    turnover=16000000000,
                ),
            ]
        )
        db_session.commit()

        out = tmp_path / "date_filtered.csv"
        count = export_table("daily_price", str(out), stocks=["9904"], start_date="2025-03-01")
        assert count == 1

    def test_export_empty_result(self, db_session, tmp_path):
        """篩選無結果回傳 0。"""
        out = tmp_path / "empty.csv"
        count = export_table("daily_price", str(out), stocks=["0000"])
        assert count == 0

    def test_export_invalid_table(self):
        """不存在的表名 raise ValueError。"""
        with pytest.raises(ValueError, match="不支援"):
            export_table("nonexistent", "out.csv")

    def test_export_invalid_format(self):
        """不支援的格式 raise ValueError。"""
        with pytest.raises(ValueError, match="不支援"):
            export_table("daily_price", "out.json", fmt="json")


class TestImportTable:
    """import_table 匯入測試（需 db_session）。"""

    def test_import_csv(self, db_session, tmp_path):
        """從 CSV 匯入 DailyPrice。"""
        csv_path = tmp_path / "import_test.csv"
        df = pd.DataFrame(
            {
                "stock_id": ["9910"],
                "date": ["2025-04-01"],
                "open": [600.0],
                "high": [610.0],
                "low": [595.0],
                "close": [605.0],
                "volume": [30000000],
                "turnover": [18000000000],
            }
        )
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        count = import_table("daily_price", str(csv_path))
        assert count == 1

        row = db_session.query(DailyPrice).filter_by(stock_id="9910").first()
        assert row is not None
        assert row.close == 605.0

    def test_import_dry_run(self, db_session, tmp_path):
        """dry_run 不實際寫入。"""
        csv_path = tmp_path / "dry_run.csv"
        df = pd.DataFrame(
            {
                "stock_id": ["9911"],
                "date": ["2025-05-01"],
                "open": [600.0],
                "high": [610.0],
                "low": [595.0],
                "close": [605.0],
                "volume": [30000000],
                "turnover": [18000000000],
            }
        )
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        count = import_table("daily_price", str(csv_path), dry_run=True)
        assert count == 1

        row = db_session.query(DailyPrice).filter_by(stock_id="9911").first()
        assert row is None

    def test_import_duplicate_skip(self, db_session, tmp_path):
        """重複資料靜默略過。"""
        # 先插入一筆
        db_session.add(
            DailyPrice(
                stock_id="9912",
                date=date(2025, 6, 1),
                open=600.0,
                high=610.0,
                low=595.0,
                close=605.0,
                volume=30000000,
                turnover=18000000000,
            )
        )
        db_session.commit()

        # 匯入相同資料
        csv_path = tmp_path / "dup.csv"
        df = pd.DataFrame(
            {
                "stock_id": ["9912"],
                "date": ["2025-06-01"],
                "open": [600.0],
                "high": [610.0],
                "low": [595.0],
                "close": [605.0],
                "volume": [30000000],
                "turnover": [18000000000],
            }
        )
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        count = import_table("daily_price", str(csv_path))
        assert count == 1  # 嘗試了 1 筆（衝突略過）

        rows = db_session.query(DailyPrice).filter_by(stock_id="9912").all()
        assert len(rows) == 1

    def test_import_validation_failure(self, tmp_path):
        """缺少必要欄位 raise ValueError。"""
        csv_path = tmp_path / "bad.csv"
        df = pd.DataFrame({"stock_id": ["2330"], "date": ["2025-01-02"]})
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        with pytest.raises(ValueError, match="驗證失敗"):
            import_table("daily_price", str(csv_path))

    def test_import_file_not_found(self):
        """檔案不存在 raise FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            import_table("daily_price", "nonexistent.csv")

    def test_import_unsupported_format(self, tmp_path):
        """不支援的格式 raise ValueError。"""
        bad_file = tmp_path / "data.json"
        bad_file.write_text("{}")
        with pytest.raises(ValueError, match="不支援"):
            import_table("daily_price", str(bad_file))


class TestListTables:
    """list_tables 測試。"""

    def test_list_tables(self, db_session):
        """列出所有表及筆數。"""
        db_session.add(StockInfo(stock_id="9920", stock_name="測試公司", industry_category="測試業"))
        db_session.commit()

        result = list_tables()
        assert len(result) == len(TABLE_REGISTRY)
        table_names = {r["table"] for r in result}
        assert "stock_info" in table_names

        stock_info_entry = next(r for r in result if r["table"] == "stock_info")
        assert stock_info_entry["count"] >= 1


class TestRoundTrip:
    """匯出 → 匯入 round-trip 測試。"""

    def test_csv_round_trip(self, db_session, tmp_path):
        """DailyPrice CSV 匯出再匯入（使用 stock 篩選確保隔離）。"""
        db_session.add(
            DailyPrice(
                stock_id="9930",
                date=date(2025, 3, 10),
                open=100.0,
                high=105.0,
                low=98.0,
                close=103.0,
                volume=5000000,
                turnover=500000000,
                spread=3.0,
            )
        )
        db_session.commit()

        # 匯出（指定 stock 避免拿到其他測試的資料）
        out = tmp_path / "round_trip.csv"
        export_count = export_table("daily_price", str(out), fmt="csv", stocks=["9930"])
        assert export_count == 1

        # 清空該筆
        db_session.query(DailyPrice).filter_by(stock_id="9930").delete()
        db_session.commit()
        assert db_session.query(DailyPrice).filter_by(stock_id="9930").count() == 0

        # 匯入
        import_count = import_table("daily_price", str(out))
        assert import_count == 1

        row = db_session.query(DailyPrice).filter_by(stock_id="9930").first()
        assert row is not None
        assert row.close == 103.0
