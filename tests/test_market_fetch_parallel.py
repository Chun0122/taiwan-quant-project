"""測試全市場 fetch 合併函式的 TWSE+TPEX 並行化（項目 D）。

驗證重點：
  - TWSE 與 TPEX 兩側 fetch 同時啟動（不同 host，啟動時戳相近）
  - wall-clock ≈ 單一 fetch 時間（並行），非兩倍（序列）
  - 單邊例外降級為空 DataFrame，另一邊仍正常回傳
  - 兩邊結果正確合併為單一 DataFrame
"""

from __future__ import annotations

import threading
import time
from datetime import date

import pandas as pd

from src.data import twse_fetcher

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _slow_fn(source: str, delay: float, timestamps: dict):
    """建立模擬 fetcher：sleep `delay` 秒，記錄啟動時戳。"""

    def _inner(target_date):
        timestamps[source] = time.time()
        time.sleep(delay)
        return pd.DataFrame([{"stock_id": source, "date": target_date, "close": 100.0}])

    return _inner


# ------------------------------------------------------------------ #
#  TestMarketFetchParallel
# ------------------------------------------------------------------ #


class TestMarketFetchParallel:
    """`fetch_market_*` 系列應以 ThreadPoolExecutor(max_workers=2) 並行 TWSE/TPEX。"""

    def test_daily_prices_twse_tpex_started_concurrently(self, monkeypatch):
        """TWSE 與 TPEX 的啟動時戳差應 < 50ms（表示確實併發啟動）。"""
        timestamps: dict = {}
        monkeypatch.setattr(twse_fetcher, "fetch_twse_daily_prices", _slow_fn("twse", 0.1, timestamps))
        monkeypatch.setattr(twse_fetcher, "fetch_tpex_daily_prices", _slow_fn("tpex", 0.1, timestamps))

        twse_fetcher.fetch_market_daily_prices(date(2026, 1, 15))
        assert "twse" in timestamps and "tpex" in timestamps
        gap = abs(timestamps["twse"] - timestamps["tpex"])
        assert gap < 0.05, f"啟動時戳差 {gap:.3f}s，應 < 0.05s（確實併發）"

    def test_daily_prices_wallclock_is_parallel_not_sequential(self, monkeypatch):
        """wall-clock 應 ≈ 單一 fetch 時間（並行），不應 ≈ 兩倍（序列）。"""
        timestamps: dict = {}
        monkeypatch.setattr(twse_fetcher, "fetch_twse_daily_prices", _slow_fn("twse", 0.1, timestamps))
        monkeypatch.setattr(twse_fetcher, "fetch_tpex_daily_prices", _slow_fn("tpex", 0.1, timestamps))

        t0 = time.time()
        twse_fetcher.fetch_market_daily_prices(date(2026, 1, 15))
        elapsed = time.time() - t0
        # 並行應 ≤ 0.15（單一 fetch 0.1s + overhead），序列則 ≥ 0.2
        assert elapsed < 0.18, f"wall-clock={elapsed:.3f}s，應 < 0.18s 代表並行"

    def test_institutional_also_parallel(self, monkeypatch):
        """fetch_market_institutional 同樣應並行 TWSE+TPEX。"""
        timestamps: dict = {}
        monkeypatch.setattr(twse_fetcher, "fetch_twse_institutional", _slow_fn("twse", 0.1, timestamps))
        monkeypatch.setattr(twse_fetcher, "fetch_tpex_institutional", _slow_fn("tpex", 0.1, timestamps))

        t0 = time.time()
        twse_fetcher.fetch_market_institutional(date(2026, 1, 15))
        elapsed = time.time() - t0
        assert elapsed < 0.18
        assert abs(timestamps["twse"] - timestamps["tpex"]) < 0.05

    def test_margin_also_parallel(self, monkeypatch):
        """fetch_market_margin 同樣應並行 TWSE+TPEX。"""
        timestamps: dict = {}
        monkeypatch.setattr(twse_fetcher, "fetch_twse_margin", _slow_fn("twse", 0.1, timestamps))
        monkeypatch.setattr(twse_fetcher, "fetch_tpex_margin", _slow_fn("tpex", 0.1, timestamps))

        t0 = time.time()
        twse_fetcher.fetch_market_margin(date(2026, 1, 15))
        elapsed = time.time() - t0
        assert elapsed < 0.18
        assert abs(timestamps["twse"] - timestamps["tpex"]) < 0.05

    def test_twse_exception_does_not_break_tpex_side(self, monkeypatch):
        """TWSE raise 時應降級為空 DataFrame，TPEX 結果仍完整回傳。"""

        def raising(target_date):
            raise ConnectionError("TWSE down")

        def ok(target_date):
            return pd.DataFrame([{"stock_id": "6488", "date": target_date, "close": 100.0}])

        monkeypatch.setattr(twse_fetcher, "fetch_twse_daily_prices", raising)
        monkeypatch.setattr(twse_fetcher, "fetch_tpex_daily_prices", ok)

        df = twse_fetcher.fetch_market_daily_prices(date(2026, 1, 15))
        assert not df.empty
        assert df["stock_id"].tolist() == ["6488"]

    def test_tpex_exception_does_not_break_twse_side(self, monkeypatch):
        """TPEX raise 時應降級為空 DataFrame，TWSE 結果仍完整回傳。"""

        def raising(target_date):
            raise ConnectionError("TPEX down")

        def ok(target_date):
            return pd.DataFrame([{"stock_id": "2330", "date": target_date, "close": 600.0}])

        monkeypatch.setattr(twse_fetcher, "fetch_twse_daily_prices", ok)
        monkeypatch.setattr(twse_fetcher, "fetch_tpex_daily_prices", raising)

        df = twse_fetcher.fetch_market_daily_prices(date(2026, 1, 15))
        assert not df.empty
        assert df["stock_id"].tolist() == ["2330"]

    def test_both_sides_merged_into_single_dataframe(self, monkeypatch):
        """兩邊皆有資料時應合併為單一 DataFrame，rows 數為兩邊之和。"""

        def twse(target_date):
            return pd.DataFrame(
                {
                    "stock_id": ["2330", "2317"],
                    "date": [target_date] * 2,
                    "close": [600.0, 120.0],
                }
            )

        def tpex(target_date):
            return pd.DataFrame(
                {
                    "stock_id": ["6488"],
                    "date": [target_date],
                    "close": [100.0],
                }
            )

        monkeypatch.setattr(twse_fetcher, "fetch_twse_daily_prices", twse)
        monkeypatch.setattr(twse_fetcher, "fetch_tpex_daily_prices", tpex)

        df = twse_fetcher.fetch_market_daily_prices(date(2026, 1, 15))
        assert len(df) == 3
        assert set(df["stock_id"]) == {"2330", "2317", "6488"}

    def test_both_sides_empty_returns_empty(self, monkeypatch):
        """兩邊皆空時回傳空 DataFrame，不 raise。"""
        monkeypatch.setattr(twse_fetcher, "fetch_twse_daily_prices", lambda d: pd.DataFrame())
        monkeypatch.setattr(twse_fetcher, "fetch_tpex_daily_prices", lambda d: pd.DataFrame())
        df = twse_fetcher.fetch_market_daily_prices(date(2026, 1, 15))
        assert df.empty

    def test_only_two_workers_used(self, monkeypatch):
        """同時 in-flight worker ≤ 2（不應超過，否則會同 host 內部衝突）。"""
        lock = threading.Lock()
        inflight = [0]
        peak = [0]

        def slow(name: str):
            def _inner(target_date):
                with lock:
                    inflight[0] += 1
                    peak[0] = max(peak[0], inflight[0])
                time.sleep(0.05)
                with lock:
                    inflight[0] -= 1
                return pd.DataFrame([{"stock_id": name, "date": target_date}])

            return _inner

        monkeypatch.setattr(twse_fetcher, "fetch_twse_daily_prices", slow("twse"))
        monkeypatch.setattr(twse_fetcher, "fetch_tpex_daily_prices", slow("tpex"))

        twse_fetcher.fetch_market_daily_prices(date(2026, 1, 15))
        assert peak[0] == 2, f"peak inflight 應為 2，實際 {peak[0]}"
