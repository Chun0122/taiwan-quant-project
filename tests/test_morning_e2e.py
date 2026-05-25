"""P2 任務 11 — morning-routine 端到端整合測試。

涵蓋：
  P11-A 非交易日 short-circuit
  P11-B dry_run=True 跑完所有 18 step（read-only stress check + 其他 step 列為 skipped）
  P11-C dry_run=True Discord 摘要可成功 build（不擲例外）
  P11-D skip_sync=True 模式（無 watchlist sync 但 logic step 全執行）
  P11-E 失敗單步不阻擋後續（atomicity）

設計：
  - 用 fresh_db fixture（每次 get_session 開新 Session），避免單一 session 多次
    開關被測試 fixture rollback 鎖住
  - 用 monkeypatch 攔截 Discord webhook / VIX yfinance / Anthropic AI 等外部 IO
  - 種子資料以 TAIEX 60 日 + 0050 + watchlist 5 檔 30 日為最小可運作集合
"""

from __future__ import annotations

import argparse
import datetime as _dt
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.cli.morning_cmd import cmd_morning_routine
from src.data.database import Base
from src.data.schema import (
    DailyFeature,
    DailyPrice,
    RotationPortfolio,
    StockInfo,
    UniverseStatLog,
)

# ====================================================================== #
# fresh_db fixture（與 test_export_dashboard.py 同構，避免循環 import）
# ====================================================================== #


@pytest.fixture()
def fresh_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "test_morning.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def _session_factory():
        sess = Session()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    import sys

    import src.data.database as db_mod

    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", Session)
    monkeypatch.setattr(db_mod, "get_session", _session_factory)

    # 動態 patch 所有已載入且持有 get_session 參考的 src.* 模組，確保 hermetic
    # （morning-routine 觸發 ~15 個模組：scanner 子模組 + cli + discovery，
    #  硬編清單易漏，會 leak 寫入真實 data/stock.db 污染其他測試）。
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("src.") and hasattr(mod, "get_session"):
            monkeypatch.setattr(mod, "get_session", _session_factory, raising=False)

    yield _session_factory


# ====================================================================== #
# 種子資料 helpers
# ====================================================================== #


def _seed_taiex(session, today: _dt.date, *, n_days: int = 60) -> None:
    """TAIEX 行事曆 + 簡易上漲序列（regime detection 用）。"""
    base_close = 23000.0
    for i in range(n_days):
        d = today - _dt.timedelta(days=n_days - 1 - i)
        close = base_close + i * 50
        session.add(
            DailyPrice(
                stock_id="TAIEX",
                date=d,
                open=close - 10,
                high=close + 20,
                low=close - 30,
                close=close,
                volume=10000000,
                turnover=0.0,
            )
        )
    session.commit()


def _seed_0050(session, today: _dt.date, *, n_days: int = 30) -> None:
    """0050 benchmark prices（alpha 計算用）。"""
    base = 150.0
    for i in range(n_days):
        d = today - _dt.timedelta(days=n_days - 1 - i)
        close = base + i * 0.1
        session.add(
            DailyPrice(
                stock_id="0050",
                date=d,
                open=close,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=100000,
                turnover=15_000_000.0,
            )
        )
    session.commit()


def _seed_watchlist(session, today: _dt.date, *, n_days: int = 30) -> None:
    """5 檔 watchlist + StockInfo（discover universe 最小集合）。"""
    sids = ["2330", "2317", "2454", "3008", "6669"]
    for sid in sids:
        session.add(
            StockInfo(
                stock_id=sid,
                stock_name=f"name_{sid}",
                industry_category="半導體",
                listing_type="上市",
                security_type="stock",
            )
        )
    session.commit()

    base_prices = {"2330": 1000, "2317": 200, "2454": 1500, "3008": 4000, "6669": 800}
    for sid in sids:
        base = base_prices[sid]
        for i in range(n_days):
            d = today - _dt.timedelta(days=n_days - 1 - i)
            close = base * (1 + i * 0.001)
            session.add(
                DailyPrice(
                    stock_id=sid,
                    date=d,
                    open=close - 1,
                    high=close + 5,
                    low=close - 5,
                    close=close,
                    volume=10_000_000,
                    turnover=close * 10_000_000,
                )
            )
            # DailyFeature 用於 universe Stage 2 流動性過濾
            session.add(
                DailyFeature(
                    stock_id=sid,
                    date=d,
                    close=close,
                    volume=10_000_000,
                    turnover=int(close * 10_000_000),
                    ma20=close * 0.99,
                    ma60=close * 0.98,
                    volume_ma20=10_000_000,
                    turnover_ma5=close * 10_000_000,
                    turnover_ma20=close * 10_000_000,
                    momentum_20d=5.0,
                    volatility_20d=20.0,
                    high_20d=close * 1.05,
                )
            )
    session.commit()


def _seed_rotation_portfolio(session) -> RotationPortfolio:
    """1 active rotation portfolio for Step 12 + Step 17."""
    p = RotationPortfolio(
        name="e2e_test",
        mode="momentum",
        max_positions=3,
        holding_days=5,
        allow_renewal=True,
        initial_capital=1_000_000.0,
        current_capital=1_000_000.0,
        current_cash=1_000_000.0,
        status="active",
    )
    session.add(p)
    session.commit()
    return p


def _patch_external_io(monkeypatch):
    """攔截 Discord / yfinance / Anthropic / TWSE-fetcher 等外部 IO。"""
    # Discord webhook
    try:
        monkeypatch.setattr("src.notification.line_notify.send_message", lambda *a, **kw: True)
    except (AttributeError, ImportError):
        pass

    # VIX 同步 (yfinance)
    try:
        monkeypatch.setattr("src.data.pipeline.sync_taiwan_vix", lambda *a, **kw: 0)
        monkeypatch.setattr("src.data.pipeline.sync_us_vix", lambda *a, **kw: 0)
    except AttributeError:
        pass

    # Anthropic AI summary — 通常 dashboard regenerate_ai_summary=False，但保險起見 mock
    try:
        monkeypatch.setattr("src.report.ai_report.generate_ai_summary", lambda *a, **kw: None)
    except (AttributeError, ImportError):
        pass


# ====================================================================== #
# P11-A: 非交易日 short-circuit
# ====================================================================== #


class TestNonTradingDayShortCircuit:
    def test_non_trading_day_returns_early(self, fresh_db, monkeypatch, capsys):
        # patch is_trading_day → False
        monkeypatch.setattr("src.data.calendar.is_trading_day", lambda d: False)
        monkeypatch.setattr("src.data.calendar.has_calendar_data", lambda y: True)
        _patch_external_io(monkeypatch)

        cmd_morning_routine(argparse.Namespace(dry_run=False, skip_sync=True, top=10, notify=False))

        out = capsys.readouterr().out
        assert "非交易日" in out
        # Step 0 / Step 9 等不應出現
        assert "[Step 9/" not in out


# ====================================================================== #
# P11-B: dry_run 跑完 18 step（其他 step 被列為 skipped）
# ====================================================================== #


class TestDryRunFullPipeline:
    def test_dry_run_lists_all_18_steps(self, fresh_db, monkeypatch, capsys):
        """dry_run=True → 18 step 標題全部出現在 stdout。"""
        today = _dt.date.today()
        with fresh_db() as session:
            _seed_taiex(session, today, n_days=60)

        _patch_external_io(monkeypatch)
        # patch stress check 避免依賴更深資料
        monkeypatch.setattr(
            "src.cli.anomaly_cmd._compute_macro_stress_check",
            lambda: {"regime": "bull", "summary": "(test)", "signals": {}},
        )
        # discover 在 dry_run 仍會被呼叫（{"dry_run"} 中），skip 即可；無需 patch
        # 但 Step 8c 內部對 IC 預檢依賴；dry-run skip_on={"dry_run"} → 被略過
        # Step 0 stress check 在 dry_run 仍會跑（read-only）

        cmd_morning_routine(argparse.Namespace(dry_run=True, skip_sync=False, top=10, notify=False))

        out = capsys.readouterr().out
        # 18 個 step number/title 都應出現（含 Step 0 + 整數 1-17 + 字串 "8b"/"8c"/"8d"/"9b"）
        expected_step_labels = [
            "Step 0",
            "Step 1/",
            "Step 2/",
            "Step 3/",
            "Step 4/",
            "Step 5/",
            "Step 7/",
            "Step 8/",
            "Step 8b/",
            "Step 8d/",
            "Step 8c/",
            "Step 9/",
            "Step 9b/",
            "Step 10/",
            "Step 11/",
            "Step 12/",
            "Step 13/",
            "Step 14/",
            "Step 15/",
            "Step 16/",
            "Step 17/",
        ]
        missing = [lbl for lbl in expected_step_labels if lbl not in out]
        assert not missing, f"missing step labels: {missing}"
        # 完成 banner
        assert "早晨例行流程完成" in out

    def test_dry_run_summary_can_be_built(self, fresh_db, monkeypatch, capsys):
        """dry_run + notify=True / dry_run=True 路徑會 build Discord summary preview。"""
        today = _dt.date.today()
        with fresh_db() as session:
            _seed_taiex(session, today, n_days=60)

        _patch_external_io(monkeypatch)
        monkeypatch.setattr(
            "src.cli.anomaly_cmd._compute_macro_stress_check",
            lambda: {"regime": "bull", "summary": "(test)", "signals": {}},
        )

        cmd_morning_routine(argparse.Namespace(dry_run=True, skip_sync=False, top=5, notify=False))

        out = capsys.readouterr().out
        # dry_run 自動觸發 Discord Summary Preview（無需 --notify）
        assert "Discord Summary Preview" in out


# ====================================================================== #
# P11-C: skip_sync 模式 — 跑實際邏輯 step（discover / rotation / dashboard / baseline）
# ====================================================================== #


class TestSkipSyncE2E:
    def test_skip_sync_executes_logic_steps_without_crash(self, fresh_db, monkeypatch, capsys, tmp_path):
        """skip_sync=True + 種子資料 → 11 個 logic step 應全部不擲例外。"""
        today = _dt.date.today()
        with fresh_db() as session:
            _seed_taiex(session, today, n_days=60)
            _seed_0050(session, today, n_days=30)
            _seed_watchlist(session, today, n_days=30)
            _seed_rotation_portfolio(session)

        _patch_external_io(monkeypatch)
        # discord 訊息發送（雖然 notify=False 但 _baseline_regression_check 內 log 仍 OK）
        # patch is_trading_day → True 確保不 short-circuit
        monkeypatch.setattr("src.data.calendar.is_trading_day", lambda d: True)
        # dashboard 寫入路徑導到 tmp
        monkeypatch.setattr("src.cli.export_dashboard_cmd._DEFAULT_OUT_DIR", tmp_path / "dashboard_out")

        # baseline_metrics.json：若無檔，Step 17 應 graceful（不擲例外）
        # 不需要種 baseline；Step 17 內部處理 missing 情境

        # 執行（不擲例外即視為成功）
        cmd_morning_routine(argparse.Namespace(dry_run=False, skip_sync=True, top=5, notify=False))

        out = capsys.readouterr().out
        # 完成 banner（即使有部分步驟 failed，也應印 banner）
        assert "早晨例行流程完成" in out

        # Step 9 (discover) skip_on={"dry_run"}，skip_sync 不阻擋 → 應有印出 Step 9 標題
        assert "[Step 9/" in out
        # Step 12 (rotation update) 同理
        assert "[Step 12/" in out
        # Step 17 baseline 守門
        assert "[Step 17/" in out

    def test_skip_sync_writes_universe_stat_log(self, fresh_db, monkeypatch, tmp_path):
        """skip_sync 模式跑完，UniverseStatLog 應有至少 1 筆寫入（Step 9 discover）。"""
        today = _dt.date.today()
        with fresh_db() as session_fac:
            pass
        with fresh_db() as session:
            _seed_taiex(session, today, n_days=60)
            _seed_0050(session, today, n_days=30)
            _seed_watchlist(session, today, n_days=30)
            _seed_rotation_portfolio(session)

        _patch_external_io(monkeypatch)
        monkeypatch.setattr("src.data.calendar.is_trading_day", lambda d: True)
        monkeypatch.setattr("src.cli.export_dashboard_cmd._DEFAULT_OUT_DIR", tmp_path / "dashboard_out")

        cmd_morning_routine(argparse.Namespace(dry_run=False, skip_sync=True, top=5, notify=False))

        # 驗證 UniverseStatLog 有寫入（至少 1 筆，來自 5 個 mode 之一）
        with fresh_db() as session:
            rows = session.execute(select(UniverseStatLog)).scalars().all()
            assert len(rows) >= 1, "Step 9 discover 應觸發 UniverseFilter → UniverseStatLog 寫入"


# ====================================================================== #
# P11-D: Atomicity — 單步失敗不阻擋後續 step
# ====================================================================== #


class TestStepAtomicity:
    def test_failing_step_does_not_block_subsequent(self, fresh_db, monkeypatch, capsys, tmp_path):
        """強制 cmd_discover 拋例外 → Step 9 失敗但 Step 12/16/17 仍執行。"""
        today = _dt.date.today()
        with fresh_db() as session:
            _seed_taiex(session, today, n_days=60)
            _seed_0050(session, today, n_days=30)
            _seed_watchlist(session, today, n_days=30)
            _seed_rotation_portfolio(session)

        _patch_external_io(monkeypatch)
        monkeypatch.setattr("src.data.calendar.is_trading_day", lambda d: True)
        monkeypatch.setattr("src.cli.export_dashboard_cmd._DEFAULT_OUT_DIR", tmp_path / "dashboard_out")

        # 讓 Step 9 _step_9_discover 失敗
        from src.cli import morning_cmd as mc

        original_step_9 = getattr(mc, "_step_9_discover", None)

        def _bad_step_9():
            raise RuntimeError("simulated step 9 failure")

        # _step_9_discover 是 nested closure 在 cmd_morning_routine 內，無法直接 monkeypatch；
        # 改用 _step_9 呼叫到的 _cmd_discover_all 為攻擊面
        try:
            monkeypatch.setattr(mc, "_cmd_discover_all", lambda *a, **kw: _bad_step_9())
        except AttributeError:
            # 若 helper 改名直接 skip 此測試（程式碼結構變動需重寫）
            pytest.skip("_cmd_discover_all helper missing — adapt this test if refactored")

        cmd_morning_routine(argparse.Namespace(dry_run=False, skip_sync=True, top=5, notify=False))

        out = capsys.readouterr().out
        # Step 9 失敗 + Step 17 仍跑完
        assert "Step 9 失敗" in out or "失敗" in out
        assert "[Step 17/" in out


# ====================================================================== #
# P11-E: TOTAL 常數驗證
# ====================================================================== #


class TestStepInventory:
    def test_total_is_18(self):
        """morning_cmd 應宣告 TOTAL = 18，與 _steps 內的條目對應。"""
        import inspect

        from src.cli.morning_cmd import cmd_morning_routine

        src = inspect.getsource(cmd_morning_routine)
        assert "TOTAL = 18" in src, "TOTAL 應為 18（若 step 數變動請更新此測試）"
