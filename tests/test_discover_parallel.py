"""測試 discover all 5 scanner 並行（項目 A）。

驗證重點：
  - 5 個 scanner 確實併發啟動（時戳相近）
  - 單一 scanner 例外不中止其他 scanner
  - 預熱（_prewarm_stage_25）確實在主 thread 跑、跑在 scanner 啟動前
  - CONCURRENCY_DISABLE=1 退化為序列模式（max_workers=1）
  - DiscoveryRecord 寫入路徑（_save_discovery_records）由 worker 觸發
  - 結果渲染依原 mode 順序（不受 thread 完成順序影響）
"""

from __future__ import annotations

import argparse
import threading
import time
from datetime import date, timedelta

import pandas as pd

from src.cli import discover_cmd as dc
from src.discovery.scanner._functions import DiscoveryResult
from src.discovery.scanner._shared_load import SharedMarketData

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _make_args(disabled_modes: list[str] | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        skip_sync=True,
        sync_days=80,
        top=10,
        min_price=10.0,
        max_price=2000.0,
        min_volume=100_000,
        max_stocks=None,
        min_appearances=1,
        export=None,
        notify=False,
        use_ic_adjustment=False,
        weekly_confirm=False,
        disabled_modes=disabled_modes or [],
    )


def _empty_shared() -> SharedMarketData:
    """建立空白 SharedMarketData 供測試（避免實際 DB 查詢）。"""
    return SharedMarketData(
        df_price=pd.DataFrame(columns=["stock_id", "date", "open", "high", "low", "close", "volume", "turnover"]),
        df_inst=pd.DataFrame(columns=["stock_id", "date", "name", "net"]),
        df_margin=pd.DataFrame(columns=["stock_id", "date", "margin_balance", "short_balance"]),
        df_revenue=pd.DataFrame(columns=["stock_id", "date", "yoy_growth", "mom_growth"]),
        price_cutoff=date.today() - timedelta(days=90),
        revenue_cutoff=date.today() - timedelta(days=180),
        loaded_at=pd.Timestamp.utcnow().to_pydatetime(),
    )


def _common_monkeypatch(monkeypatch):
    """共用：patch 掉 DB 初始化、市場同步、shared 載入、預熱。"""
    monkeypatch.setattr(dc, "init_db", lambda: None)
    monkeypatch.setattr(dc, "ensure_sync_market_data", lambda *a, **kw: None)
    monkeypatch.setattr(
        "src.discovery.scanner._shared_load.load_shared_market_data",
        lambda **kw: _empty_shared(),
    )


# ------------------------------------------------------------------ #
#  TestDiscoverParallel
# ------------------------------------------------------------------ #


class TestDiscoverParallel:
    def test_five_scanners_started_concurrently(self, monkeypatch):
        """5 個 scanner 應於 max_workers=5 下併發啟動，啟動時戳兩兩 < 100ms。"""
        _common_monkeypatch(monkeypatch)
        monkeypatch.setattr(dc, "_prewarm_stage_25", lambda *a, **kw: None)

        timestamps: dict[str, float] = {}
        lock = threading.Lock()

        def fake_run(self, shared=None, precomputed_ic=None):
            with lock:
                timestamps[self.mode_name] = time.time()
            time.sleep(0.1)
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=0,
                after_coarse=0,
                mode=self.mode_name,
            )

        # base 與 growth/value/dividend 都有自己的 run；統一替換
        monkeypatch.setattr("src.discovery.scanner._base.MarketScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._growth.GrowthScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._value.ValueScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._dividend.DividendScanner.run", fake_run)

        t0 = time.time()
        dc._cmd_discover_all(_make_args())
        elapsed = time.time() - t0

        assert len(timestamps) == 5, f"應啟動 5 個 scanner，實際 {len(timestamps)}"
        gap = max(timestamps.values()) - min(timestamps.values())
        assert gap < 0.1, f"5 scanner 啟動時戳差 {gap:.3f}s，應 < 0.1s"
        # 並行：總時 ≈ 0.1（單一 scanner sleep）+ overhead；序列會 ≈ 0.5
        assert elapsed < 0.4, f"wall-clock={elapsed:.3f}s，應 < 0.4s 代表確實並行"

    def test_one_scanner_failure_does_not_abort_others(self, monkeypatch):
        """單一 scanner raise 後，其他 4 個 mode 仍應有完整 result。"""
        _common_monkeypatch(monkeypatch)
        monkeypatch.setattr(dc, "_prewarm_stage_25", lambda *a, **kw: None)

        completed_modes: list[str] = []
        lock = threading.Lock()

        def fake_run(self, shared=None, precomputed_ic=None):
            if self.mode_name == "swing":
                raise RuntimeError("swing simulation crash")
            with lock:
                completed_modes.append(self.mode_name)
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=10,
                after_coarse=5,
                mode=self.mode_name,
            )

        monkeypatch.setattr("src.discovery.scanner._base.MarketScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._growth.GrowthScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._value.ValueScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._dividend.DividendScanner.run", fake_run)

        # 應不 raise
        dc._cmd_discover_all(_make_args())
        assert "swing" not in completed_modes
        assert {"momentum", "value", "dividend", "growth"}.issubset(set(completed_modes))

    def test_prewarm_runs_in_main_thread_before_scanners(self, monkeypatch):
        """預熱必須在主 thread 跑，且早於任何 scanner.run()。"""
        _common_monkeypatch(monkeypatch)
        events: list[tuple[str, str, float]] = []  # (event, thread_name, ts)
        lock = threading.Lock()

        def fake_prewarm(shared, top_n=200):
            with lock:
                events.append(("prewarm", threading.current_thread().name, time.time()))
            time.sleep(0.02)

        monkeypatch.setattr(dc, "_prewarm_stage_25", fake_prewarm)

        def fake_run(self, shared=None, precomputed_ic=None):
            with lock:
                events.append(("scanner", threading.current_thread().name, time.time()))
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=0,
                after_coarse=0,
                mode=self.mode_name,
            )

        monkeypatch.setattr("src.discovery.scanner._base.MarketScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._growth.GrowthScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._value.ValueScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._dividend.DividendScanner.run", fake_run)

        dc._cmd_discover_all(_make_args())

        # 第一個 event 必須是 prewarm
        assert events[0][0] == "prewarm"
        # prewarm thread 應為主 thread（不在 ThreadPoolExecutor 開的 worker thread 中）
        assert "discover" not in events[0][1].lower(), f"預熱應在主 thread 跑，實際在 {events[0][1]}"
        # 所有 scanner event 的時戳都晚於 prewarm
        prewarm_ts = events[0][2]
        for evt, _name, ts in events[1:]:
            if evt == "scanner":
                assert ts >= prewarm_ts

    def test_concurrency_disable_falls_back_to_sequential(self, monkeypatch):
        """CONCURRENCY_DISABLE=1 應讓 _resolve_max_workers 回傳 1，scanner 退化為序列。"""
        monkeypatch.setenv("CONCURRENCY_DISABLE", "1")
        assert dc._resolve_max_workers(default=5) == 1

        monkeypatch.delenv("CONCURRENCY_DISABLE", raising=False)
        assert dc._resolve_max_workers(default=5) == 5

    def test_save_discovery_records_called_per_non_empty_result(self, monkeypatch):
        """有 rankings 的 mode 應觸發 _save_discovery_records；空 rankings 不觸發。"""
        _common_monkeypatch(monkeypatch)
        monkeypatch.setattr(dc, "_prewarm_stage_25", lambda *a, **kw: None)

        save_calls: list[str] = []

        def fake_save(result, mode, scanner):
            save_calls.append(mode)

        monkeypatch.setattr(dc, "_save_discovery_records", fake_save)

        def fake_run(self, shared=None, precomputed_ic=None):
            # 三個 mode 有結果、兩個沒有
            if self.mode_name in ("momentum", "value", "growth"):
                return DiscoveryResult(
                    rankings=pd.DataFrame([{"stock_id": "2330", "rank": 1}]),
                    total_stocks=10,
                    after_coarse=5,
                    mode=self.mode_name,
                )
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=10,
                after_coarse=0,
                mode=self.mode_name,
            )

        monkeypatch.setattr("src.discovery.scanner._base.MarketScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._growth.GrowthScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._value.ValueScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._dividend.DividendScanner.run", fake_run)

        dc._cmd_discover_all(_make_args())

        assert sorted(save_calls) == ["growth", "momentum", "value"]

    def test_disabled_modes_are_skipped(self, monkeypatch):
        """disabled_modes 中的 mode 不應 submit 到 ThreadPool。"""
        _common_monkeypatch(monkeypatch)
        monkeypatch.setattr(dc, "_prewarm_stage_25", lambda *a, **kw: None)

        called: list[str] = []
        lock = threading.Lock()

        def fake_run(self, shared=None, precomputed_ic=None):
            with lock:
                called.append(self.mode_name)
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=0,
                after_coarse=0,
                mode=self.mode_name,
            )

        monkeypatch.setattr("src.discovery.scanner._base.MarketScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._growth.GrowthScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._value.ValueScanner.run", fake_run)
        monkeypatch.setattr("src.discovery.scanner._dividend.DividendScanner.run", fake_run)

        dc._cmd_discover_all(_make_args(disabled_modes=["swing", "growth"]))

        assert "swing" not in called
        assert "growth" not in called
        assert {"momentum", "value", "dividend"}.issubset(set(called))


# ------------------------------------------------------------------ #
#  TestPrewarmStage25
# ------------------------------------------------------------------ #


class TestPrewarmStage25:
    def test_picks_top_by_5d_turnover(self, monkeypatch):
        """預熱應依 5 日 turnover 加總排序，取 top_n。"""
        # 構造：A 5 日 turnover=500（最大）、B=200、C=100、D=10
        rows = []
        today = date.today()
        for sid, t in [("A", 100), ("B", 40), ("C", 20), ("D", 2)]:
            for d in range(5):
                rows.append(
                    {
                        "stock_id": sid,
                        "date": today - timedelta(days=d),
                        "open": 0,
                        "high": 0,
                        "low": 0,
                        "close": 0,
                        "volume": 0,
                        "turnover": t,
                    }
                )
        shared = SharedMarketData(
            df_price=pd.DataFrame(rows),
            df_inst=pd.DataFrame(),
            df_margin=pd.DataFrame(),
            df_revenue=pd.DataFrame(),
            price_cutoff=today - timedelta(days=90),
            revenue_cutoff=today - timedelta(days=180),
            loaded_at=pd.Timestamp.utcnow().to_pydatetime(),
        )

        captured_ids: list[list[str]] = []
        monkeypatch.setattr(
            "src.data.pipeline.sync_revenue_for_stocks",
            lambda ids: captured_ids.append(list(ids)) or 0,
        )
        monkeypatch.setattr(
            "src.data.pipeline.sync_broker_for_stocks",
            lambda ids: captured_ids.append(list(ids)) or 0,
        )

        dc._prewarm_stage_25(shared, top_n=2)

        # 兩次 sync 都收到同樣的 top 2: [A, B]（依 turnover 5d 加總排序）
        assert captured_ids == [["A", "B"], ["A", "B"]]

    def test_empty_price_data_skips_silently(self, monkeypatch):
        """df_price 為空時直接 return，不應呼叫 sync_*。"""
        called: list[str] = []
        monkeypatch.setattr(
            "src.data.pipeline.sync_revenue_for_stocks",
            lambda ids: called.append("rev") or 0,
        )
        monkeypatch.setattr(
            "src.data.pipeline.sync_broker_for_stocks",
            lambda ids: called.append("broker") or 0,
        )

        empty = SharedMarketData(
            df_price=pd.DataFrame(columns=["stock_id", "date", "open", "high", "low", "close", "volume", "turnover"]),
            df_inst=pd.DataFrame(),
            df_margin=pd.DataFrame(),
            df_revenue=pd.DataFrame(),
            price_cutoff=date.today() - timedelta(days=90),
            revenue_cutoff=date.today() - timedelta(days=180),
            loaded_at=pd.Timestamp.utcnow().to_pydatetime(),
        )
        dc._prewarm_stage_25(empty)
        assert called == []

    def test_sync_failure_does_not_raise(self, monkeypatch):
        """sync_revenue/broker 任一 raise 應被吞掉，不影響後續 scanner。"""
        rows = [
            {
                "stock_id": "A",
                "date": date.today(),
                "open": 0,
                "high": 0,
                "low": 0,
                "close": 0,
                "volume": 0,
                "turnover": 100,
            }
        ]
        shared = SharedMarketData(
            df_price=pd.DataFrame(rows),
            df_inst=pd.DataFrame(),
            df_margin=pd.DataFrame(),
            df_revenue=pd.DataFrame(),
            price_cutoff=date.today() - timedelta(days=90),
            revenue_cutoff=date.today() - timedelta(days=180),
            loaded_at=pd.Timestamp.utcnow().to_pydatetime(),
        )

        def boom(ids):
            raise ConnectionError("FinMind down")

        monkeypatch.setattr("src.data.pipeline.sync_revenue_for_stocks", boom)
        monkeypatch.setattr("src.data.pipeline.sync_broker_for_stocks", boom)

        # 不應 raise
        dc._prewarm_stage_25(shared, top_n=1)


# ------------------------------------------------------------------ #
#  TestRunScannerWorker — 直接測 worker 函式（單元層）
# ------------------------------------------------------------------ #


class TestRunScannerWorker:
    def test_returns_error_on_exception(self, monkeypatch):
        """scanner.run() 例外 → 回傳 (mode, None, summary, error)。"""
        from src.discovery.scanner import MomentumScanner

        def raising(self, shared=None, precomputed_ic=None):
            raise ValueError("test")

        monkeypatch.setattr("src.discovery.scanner._base.MarketScanner.run", raising)

        mode, result, summary, err = dc._run_scanner_worker(
            "momentum", "動能", MomentumScanner, _make_args(), _empty_shared()
        )
        assert mode == "momentum"
        assert result is None
        assert "失敗" in summary
        assert isinstance(err, ValueError)

    def test_save_discovery_records_failure_does_not_raise(self, monkeypatch):
        """_save_discovery_records 失敗應被吃掉，回傳仍視為 success。"""
        from src.discovery.scanner import MomentumScanner

        monkeypatch.setattr(
            "src.discovery.scanner._base.MarketScanner.run",
            lambda self, shared=None, precomputed_ic=None: DiscoveryResult(
                rankings=pd.DataFrame([{"stock_id": "2330", "rank": 1}]),
                total_stocks=10,
                after_coarse=5,
                mode=self.mode_name,
            ),
        )

        def boom(*a, **kw):
            raise RuntimeError("DB write failed")

        monkeypatch.setattr(dc, "_save_discovery_records", boom)
        mode, result, summary, err = dc._run_scanner_worker(
            "momentum", "動能", MomentumScanner, _make_args(), _empty_shared()
        )
        assert err is None  # save 失敗不算 worker 失敗
        assert result is not None
        assert "Top" in summary
