"""tests/test_export_dashboard.py — Dashboard JSON 匯出測試。

涵蓋：
- 各區塊 builder（discover / rotation / watch_entries / signals / data_freshness）
- _build_payload 完整組裝（patch 重型依賴：stress check / IC status / strategy events）
- _write_payload 原子寫檔 + latest.json 同步
- cmd_export_dashboard end-to-end

注意：
- 沿用 test_rotation.py 的合併測試模式，避免 fixture session 與 production code
  內 `with get_session()` 多次開關造成連線關閉。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.cli import export_dashboard_cmd as ed
from src.data.database import Base
from src.data.schema import (
    DailyPrice,
    DiscoveryRecord,
    RotationDailySnapshot,
    RotationPortfolio,
    RotationPosition,
    WatchEntry,
)

# ---------------------------------------------------------------------------
# 共用 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_db(tmp_path: Path, monkeypatch):
    """每次 get_session() 都產生新的 Session（共享同一檔案 SQLite engine）。

    與 conftest 的 `db_session`（單一共享 session）不同 — 此 fixture 適合需要
    多次 `with get_session()` 的整合測試（例如 _build_payload 內部會連續呼叫
    多個 builder，每個都自己開 with）。
    """
    db_path = tmp_path / "test_dashboard.db"
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

    import src.data.database as db_mod

    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", Session)
    monkeypatch.setattr(db_mod, "get_session", _session_factory)

    # 同步 patch 所有已 import 過 get_session 的模組（lazy import 在函式內，
    # 但 export_dashboard_cmd 與 morning_cmd 等模組頂層已 import，需覆蓋）
    for mod_name in (
        "src.cli.export_dashboard_cmd",
        "src.portfolio.manager",
        "src.cli.anomaly_cmd",
    ):
        try:
            mod = __import__(mod_name, fromlist=["get_session"])
            if hasattr(mod, "get_session"):
                monkeypatch.setattr(mod, "get_session", _session_factory)
        except ImportError:
            pass

    yield _session_factory


@pytest.fixture()
def stub_heavy_deps(monkeypatch):
    """Patch 重型依賴：stress check / IC status / strategy events，避免實際呼叫。"""
    stress_payload = {
        "regime": "bull",
        "crisis_triggered": False,
        "breadth_downgraded": False,
        "breadth_below_ma20_pct": 0.62,
        "taiex_close": 23105.5,
        "fast_return_5d": -0.012,
        "consec_decline_days": 0,
        "vol_ratio": 1.05,
        "vix_val": 15.2,
        "us_vix_val": 18.4,
        "summary": "市場狀態=bull TAIEX=23105，5日=-1.2%，MA20寬度=62%",
        "signals": {},
    }
    ic_status = [
        {
            "mode": "Momentum",
            "mode_key": "momentum",
            "factor": "mom5_3d",
            "ic": 0.12,
            "level": "normal",
            "sample_count": 50,
        },
        {
            "mode": "Swing",
            "mode_key": "swing",
            "factor": "mom20_60",
            "ic": -0.08,
            "level": "inverse",
            "sample_count": 50,
        },
    ]
    monkeypatch.setattr(
        "src.cli.anomaly_cmd._compute_macro_stress_check",
        lambda: dict(stress_payload),
    )
    monkeypatch.setattr(
        "src.cli.morning_cmd._compute_factor_ic_status",
        lambda: (list(ic_status), {}),
    )
    monkeypatch.setattr(
        "src.discovery.strategy_events.collect_strategy_events",
        lambda days=30: [],
    )
    return {"stress": stress_payload, "ic_status": ic_status}


def _seed_dataset(db_session, today: _dt.date) -> None:
    """注入：TAIEX + 個股 DailyPrice、5 筆 momentum DiscoveryRecord、
    1 筆 active + 1 筆 closed WatchEntry、2 個 RotationPortfolio + 1 筆 open position。

    把所有 DB 寫入塞到單一函式呼叫，避免測試方法內多次 commit/with。
    """
    db_session.add_all(
        [
            DailyPrice(
                stock_id="TAIEX",
                date=today,
                open=23000,
                high=23200,
                low=22950,
                close=23105,
                volume=0,
                turnover=0.0,
            ),
            DailyPrice(
                stock_id="2330",
                date=today,
                open=1080,
                high=1090,
                low=1075,
                close=1085,
                volume=10_000_000,
                turnover=10_850_000_000.0,
            ),
        ]
    )

    for i in range(5):
        db_session.add(
            DiscoveryRecord(
                scan_date=today,
                mode="momentum",
                rank=i + 1,
                stock_id=f"23{30 + i:02d}",
                stock_name=f"測試股{i}",
                close=100 + i,
                composite_score=0.8 - i * 0.02,
                technical_score=0.7,
                chip_score=0.6,
                fundamental_score=0.5,
                news_score=0.4,
                regime="bull",
                entry_price=101 + i,
                stop_loss=95 + i,
                take_profit=120 + i,
                industry_category="半導體",
                valid_until=today + _dt.timedelta(days=5),
                chip_tier="8F",
            )
        )

    db_session.add_all(
        [
            WatchEntry(
                stock_id="2330",
                stock_name="台積電",
                entry_date=today - _dt.timedelta(days=3),
                entry_price=1050.0,
                stop_loss=1020.0,
                take_profit=1120.0,
                quantity=1000,
                source="discover",
                mode="momentum",
                status="active",
                valid_until=today + _dt.timedelta(days=5),
            ),
            WatchEntry(
                stock_id="2317",
                entry_date=today - _dt.timedelta(days=10),
                entry_price=200.0,
                status="taken_profit",
            ),
        ]
    )

    primary = RotationPortfolio(
        name="default",
        mode="momentum",
        max_positions=5,
        holding_days=10,
        allow_renewal=True,
        initial_capital=1_000_000,
        current_capital=1_050_000,
        current_cash=200_000,
        status="active",
    )
    secondary = RotationPortfolio(
        name="small",
        mode="swing",
        max_positions=3,
        holding_days=20,
        allow_renewal=True,
        initial_capital=500_000,
        current_capital=500_000,
        current_cash=500_000,
        status="active",
    )
    db_session.add_all([primary, secondary])
    db_session.flush()

    db_session.add(
        RotationPosition(
            portfolio_id=primary.id,
            stock_id="2330",
            stock_name="台積電",
            entry_date=today - _dt.timedelta(days=5),
            entry_price=1050.0,
            entry_rank=1,
            entry_score=0.85,
            holding_days_count=5,
            planned_exit_date=today + _dt.timedelta(days=5),
            shares=1000,
            allocated_capital=1_050_000,
            stop_loss=1020.0,
            status="open",
        )
    )
    db_session.flush()
    db_session.commit()


# ---------------------------------------------------------------------------
# Signals — 純函數測試（無 DB 依賴）
# ---------------------------------------------------------------------------


class TestBuildSignals:
    def test_crisis_yields_critical(self):
        regime = {"state": "crisis", "crisis_triggered": True, "summary": "崩盤訊號"}
        signals = ed._build_signals(regime, ic_status=None)
        assert any(s["type"] == "crisis" and s["severity"] == "critical" for s in signals)

    def test_bear_yields_warning(self):
        regime = {"state": "bear", "summary": "空頭"}
        signals = ed._build_signals(regime, ic_status=None)
        assert any(s["type"] == "bear_market" and s["severity"] == "warning" for s in signals)

    def test_breadth_downgrade(self):
        regime = {"state": "sideways", "breadth_downgraded": True, "breadth_below_ma20_pct": 0.42}
        signals = ed._build_signals(regime, ic_status=None)
        assert any(s["type"] == "breadth_downgrade" for s in signals)

    def test_ic_inverse_yields_critical(self):
        regime = {"state": "bull"}
        ic_status = [
            {"mode": "Swing", "mode_key": "swing", "factor": "mom20", "ic": -0.1, "level": "inverse"},
        ]
        signals = ed._build_signals(regime, ic_status=ic_status)
        assert any(s["type"] == "ic_decay" and s["severity"] == "critical" for s in signals)

    def test_ic_weak_yields_warning(self):
        regime = {"state": "bull"}
        ic_status = [
            {"mode": "Momentum", "mode_key": "momentum", "factor": "mom5", "ic": 0.02, "level": "weak"},
        ]
        signals = ed._build_signals(regime, ic_status=ic_status)
        assert any(s["type"] == "ic_decay" and s["severity"] == "warning" for s in signals)

    def test_no_signals_when_normal(self):
        regime = {"state": "bull"}
        ic_status = [
            {"mode": "Momentum", "mode_key": "momentum", "factor": "mom5", "ic": 0.15, "level": "normal"},
        ]
        signals = ed._build_signals(regime, ic_status=ic_status)
        assert signals == []


# ---------------------------------------------------------------------------
# 寫檔 — 純檔案 IO，無 DB
# ---------------------------------------------------------------------------


class TestWritePayload:
    def test_creates_dated_and_latest(self, tmp_path: Path):
        payload = {"version": 1, "date": "2026-05-01"}
        target = _dt.date(2026, 5, 1)
        dated, latest = ed._write_payload(payload, tmp_path, target)
        assert dated.exists() and latest.exists()
        assert dated.name == "2026-05-01.json"
        assert latest.name == "latest.json"
        # 內容相同
        assert dated.read_text(encoding="utf-8") == latest.read_text(encoding="utf-8")
        assert json.loads(dated.read_text(encoding="utf-8"))["version"] == 1

    def test_overwrite_existing(self, tmp_path: Path):
        target = _dt.date(2026, 5, 1)
        ed._write_payload({"version": 1, "date": "old"}, tmp_path, target)
        ed._write_payload({"version": 1, "date": "new"}, tmp_path, target)
        latest = tmp_path / "latest.json"
        assert json.loads(latest.read_text(encoding="utf-8"))["date"] == "new"

    def test_no_tmp_leftover(self, tmp_path: Path):
        target = _dt.date(2026, 5, 1)
        ed._write_payload({"version": 1, "date": "x"}, tmp_path, target)
        assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# 區塊 builder — 各自單一測試（避免 fixture session 與 with get_session() 多次衝突）
# ---------------------------------------------------------------------------


class TestBuildDiscover:
    """Discover 區塊 — 一次測完所有斷言（multi-with 風險）。"""

    def test_full(self, db_session):
        today = _dt.date(2026, 5, 1)
        _seed_dataset(db_session, today)

        out = ed._build_discover(today, top_n=10)
        assert set(out.keys()) == set(ed._MODES)
        assert len(out["momentum"]) == 5
        assert [r["rank"] for r in out["momentum"]] == [1, 2, 3, 4, 5]
        first = out["momentum"][0]
        assert first["scores"] == {"technical": 0.7, "chip": 0.6, "fundamental": 0.5, "news": 0.4}
        assert first["valid_until"] == (today + _dt.timedelta(days=5)).isoformat()
        # 其他模式空 list
        for m in ("swing", "value", "dividend", "growth"):
            assert out[m] == []

    def test_top_n_limit(self, db_session):
        today = _dt.date(2026, 5, 1)
        _seed_dataset(db_session, today)
        out = ed._build_discover(today, top_n=3)
        assert len(out["momentum"]) == 3


class TestBuildWatchEntries:
    def test_only_active(self, db_session):
        today = _dt.date(2026, 5, 1)
        _seed_dataset(db_session, today)

        entries = ed._build_watch_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["stock_id"] == "2330"
        assert e["status"] == "active"
        assert isinstance(e["entry_date"], str)
        assert e["trailing_stop_enabled"] is False


class TestBuildRotation:
    def test_picks_active_with_largest_capital(self, db_session):
        today = _dt.date(2026, 5, 1)
        _seed_dataset(db_session, today)

        rot = ed._build_rotation()
        assert rot is not None
        assert rot["name"] == "default"  # 1.05M > 0.5M
        assert len(rot["holdings"]) == 1
        h = rot["holdings"][0]
        assert h["stock_id"] == "2330"
        assert h["current_price"] == 1085  # latest DailyPrice
        assert isinstance(h["entry_date"], str)


class TestBuildDataFreshnessSignal:
    def test_levels(self, db_session):
        today = _dt.date(2026, 5, 1)
        _seed_dataset(db_session, today)

        # 同日 → 無訊號
        assert ed._build_data_freshness_signal(today) is None
        # 落後 5 天 → warning
        sig = ed._build_data_freshness_signal(today + _dt.timedelta(days=5))
        assert sig is not None
        assert sig["severity"] == "warning"
        # 落後 30 天 → critical
        sig2 = ed._build_data_freshness_signal(today + _dt.timedelta(days=30))
        assert sig2["severity"] == "critical"


# ---------------------------------------------------------------------------
# Payload + CLI 整合 — 合併單一測試，避免多次 with get_session()
# ---------------------------------------------------------------------------


class TestPayloadAndCli:
    """End-to-end 合併測試 — 一次性驗證 payload + 寫檔 + CLI。

    使用 fresh_db fixture（每次 get_session 都產生新 Session），
    避免 _build_payload 內 5+ 個 builder 的多次 `with get_session()` 衝突。
    """

    def test_e2e(self, fresh_db, stub_heavy_deps, tmp_path: Path):
        today = _dt.date(2026, 5, 1)
        # 用 fresh_db 的 Session factory 寫入種子資料
        with fresh_db() as session:
            _seed_dataset(session, today)

        args = argparse.Namespace(
            date=today.isoformat(),
            top=10,
            event_days=30,
            out=str(tmp_path / "out"),
            regenerate_ai_summary=False,
        )
        ed.cmd_export_dashboard(args)

        out_dir = tmp_path / "out"
        latest = out_dir / "latest.json"
        dated = out_dir / f"{today.isoformat()}.json"
        assert latest.exists() and dated.exists()
        data = json.loads(latest.read_text(encoding="utf-8"))

        # 必填頂層欄位
        for key in (
            "version",
            "generated_at",
            "date",
            "regime",
            "discover",
            "rotation",
            "watch_entries",
            "signals",
            "strategy_events",
            "ai_summary",
            "errors",
        ):
            assert key in data, f"missing top-level key: {key}"

        assert data["version"] == ed.SCHEMA_VERSION
        assert data["date"] == today.isoformat()
        assert data["regime"]["state"] == "bull"
        # regime 不外洩 _signals
        assert all(not k.startswith("_") for k in data["regime"])

        # discover 五模式都是 list
        for m in ed._MODES:
            assert isinstance(data["discover"][m], list)
        assert len(data["discover"]["momentum"]) == 5

        # rotation 主組合
        assert data["rotation"]["name"] == "default"
        assert len(data["rotation"]["holdings"]) == 1

        # watch_entries 只取 active
        assert len(data["watch_entries"]) == 1
        assert data["watch_entries"][0]["status"] == "active"

        # signals 應含 IC inverse（來自 stub_heavy_deps）
        assert any(s["type"] == "ic_decay" for s in data["signals"])

        # AI summary 預設不重新呼叫
        assert data["ai_summary"] is None

        # 字串 JSON 序列化中文不亂碼
        assert "台積電" in latest.read_text(encoding="utf-8")

        # v1.1 新區塊存在於 top-level
        assert "portfolio_review" in data
        assert "position_timeseries" in data

        # 無錯誤
        assert data["errors"] == []


# ---------------------------------------------------------------------------
# v1.1 新區塊：portfolio_review
# ---------------------------------------------------------------------------


def _seed_snapshot_history(
    db_session,
    portfolio_name: str,
    end_date: _dt.date,
    n_days: int,
    initial: float = 1_000_000.0,
    daily_drift: float = 0.001,
) -> None:
    """為指定 portfolio 注入連續 n_days 個 daily snapshot（end_date 為最後一天）。

    capital 走勢採線性微幅成長：capital_i = initial * (1 + i * daily_drift)
    其中第一筆 daily_return_pct=None；後續為 (capital_i - capital_{i-1}) / capital_{i-1}.
    """
    prev_cap: float | None = None
    for i in range(n_days):
        d = end_date - _dt.timedelta(days=(n_days - 1 - i))
        cap = initial * (1 + i * daily_drift)
        daily_ret = None if prev_cap is None or prev_cap <= 0 else (cap - prev_cap) / prev_cap
        db_session.add(
            RotationDailySnapshot(
                portfolio_name=portfolio_name,
                snapshot_date=d,
                total_capital=cap,
                total_market_value=cap * 0.7,
                total_cash=cap * 0.3,
                unrealized_pnl=cap - initial,
                daily_return_pct=daily_ret,
                n_holdings=3,
                regime_state="bull",
            )
        )
        prev_cap = cap
    db_session.commit()


class TestBuildPortfolioReview:
    def test_no_rotation_returns_none(self):
        assert ed._build_portfolio_review(None, lookback_days=30) is None

    def test_empty_snapshots_returns_skeleton(self, fresh_db):
        today = _dt.date(2026, 5, 1)
        with fresh_db() as session:
            _seed_dataset(session, today)

        rot = ed._build_rotation()
        review = ed._build_portfolio_review(rot, lookback_days=30)
        assert review is not None
        assert review["snapshots_count"] == 0
        assert review["sharpe_ratio"] is None
        assert review["max_drawdown_pct"] is None
        assert review["wtd_return_pct"] is None
        assert review["mtd_return_pct"] is None
        # total_return_pct 從 rotation_block 取，仍應有值
        assert review["total_return_pct"] is not None

    def test_full_metrics_with_30d(self, fresh_db):
        today = _dt.date(2026, 5, 1)
        with fresh_db() as session:
            _seed_dataset(session, today)
            _seed_snapshot_history(session, "default", today, n_days=30)

        rot = ed._build_rotation()
        review = ed._build_portfolio_review(rot, lookback_days=90)
        assert review["snapshots_count"] == 30
        # 30 天 >> 10，sharpe 必有值
        assert review["sharpe_ratio"] is not None
        # 平滑上漲 → MDD 接近 0
        assert review["max_drawdown_pct"] == 0.0
        # equity_curve asc by date
        assert len(review["equity_curve"]) == 30
        assert review["equity_curve"][0]["date"] < review["equity_curve"][-1]["date"]
        # today_pnl_pct 等於最後一筆 daily_return_pct
        assert review["today_pnl_pct"] is not None
        # win_rate 在線性上漲下接近 100%（首筆 None 排除）
        assert review["win_rate_pct"] >= 95.0

    def test_min_samples_threshold(self, fresh_db):
        """N=5 時 sharpe 強制 null，但 mdd/win_rate 仍計算（N>=3）。"""
        today = _dt.date(2026, 5, 1)
        with fresh_db() as session:
            _seed_dataset(session, today)
            _seed_snapshot_history(session, "default", today, n_days=5)

        rot = ed._build_rotation()
        review = ed._build_portfolio_review(rot, lookback_days=90)
        assert review["snapshots_count"] == 5
        assert review["sharpe_ratio"] is None  # N<10
        assert review["max_drawdown_pct"] is not None  # N>=3
        assert review["win_rate_pct"] is not None


# ---------------------------------------------------------------------------
# v1.1 新區塊：position_timeseries
# ---------------------------------------------------------------------------


def _seed_price_history(db_session, stock_id: str, end_date: _dt.date, days: int) -> None:
    """注入 stock_id 自 end_date 倒推 days 個交易日（跳過週末）的 DailyPrice。

    遇到既有資料（同 stock_id+date）會 skip，避免與 _seed_dataset 重複觸發 UniqueConstraint。
    """
    existing = {r[0] for r in db_session.execute(select(DailyPrice.date).where(DailyPrice.stock_id == stock_id)).all()}
    d = end_date
    inserted = 0
    while inserted < days:
        if d.weekday() < 5:  # 週一至週五
            if d not in existing:
                db_session.add(
                    DailyPrice(
                        stock_id=stock_id,
                        date=d,
                        open=100,
                        high=102,
                        low=99,
                        close=100 + inserted * 0.5,
                        volume=1_000_000,
                        turnover=1.0,
                    )
                )
            inserted += 1
        d -= _dt.timedelta(days=1)
    db_session.commit()


class TestBuildPositionTimeseries:
    def test_no_holdings_returns_none(self, fresh_db):
        with fresh_db() as session:
            _seed_dataset(session, _dt.date(2026, 5, 1))
        ts = ed._build_position_timeseries(
            rotation_block=None, watch_block=[], days=14, target_date=_dt.date(2026, 5, 1)
        )
        assert ts is None

    def test_full_history_first_idx_zero(self, fresh_db):
        today = _dt.date(2026, 5, 1)
        with fresh_db() as session:
            _seed_dataset(session, today)
            _seed_price_history(session, "2330", today, days=20)

        rot = ed._build_rotation()
        watch = ed._build_watch_entries()
        ts = ed._build_position_timeseries(rot, watch, days=14, target_date=today)
        assert ts is not None
        assert len(ts["trading_days"]) == 14
        assert "2330" in ts["series"]
        assert ts["series"]["2330"]["first_idx"] == 0
        assert len(ts["series"]["2330"]["close"]) == 14

    def test_short_history_first_idx_offset(self, fresh_db):
        today = _dt.date(2026, 5, 1)
        with fresh_db() as session:
            _seed_dataset(session, today)
            # 只塞 5 個交易日
            _seed_price_history(session, "2330", today, days=5)
            # 額外塞另一檔有完整 14+ 天的股票，讓 trading_days 長到 14
            _seed_price_history(session, "9999", today, days=20)
            # 把 9999 加入 rotation 持倉以加入 sids
            primary = session.execute(select(RotationPortfolio).where(RotationPortfolio.name == "default")).scalar_one()
            session.add(
                RotationPosition(
                    portfolio_id=primary.id,
                    stock_id="9999",
                    stock_name="影武者",
                    entry_date=today - _dt.timedelta(days=2),
                    entry_price=100.0,
                    entry_rank=2,
                    holding_days_count=2,
                    planned_exit_date=today + _dt.timedelta(days=8),
                    shares=100,
                    allocated_capital=10_000,
                    status="open",
                )
            )
            session.commit()

        rot = ed._build_rotation()
        watch = ed._build_watch_entries()
        ts = ed._build_position_timeseries(rot, watch, days=14, target_date=today)
        assert ts is not None
        # 9999 完整 14 天
        assert ts["series"]["9999"]["first_idx"] == 0
        assert len(ts["series"]["9999"]["close"]) == 14
        # 2330 只有 5 天 → first_idx = 14 - 5 = 9
        assert ts["series"]["2330"]["first_idx"] == 14 - 5
        assert len(ts["series"]["2330"]["close"]) == 5

    def test_excludes_weekends(self, fresh_db):
        today = _dt.date(2026, 5, 1)  # Friday
        with fresh_db() as session:
            _seed_dataset(session, today)
            _seed_price_history(session, "2330", today, days=14)

        rot = ed._build_rotation()
        watch = ed._build_watch_entries()
        ts = ed._build_position_timeseries(rot, watch, days=14, target_date=today)
        # trading_days 應全為平日
        for d_str in ts["trading_days"]:
            d = _dt.date.fromisoformat(d_str)
            assert d.weekday() < 5
