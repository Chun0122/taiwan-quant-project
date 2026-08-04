"""B1 — Point-in-Time 重放測試（2026-08-01）。

B1 的價值全押在一件事上：**PIT 重放看到的必須恰好是「當時看得到的」**。
一旦漏進未來資料，歷史驗證就會系統性高估策略績效（look-ahead 的方向永遠偏
樂觀），整個 B1 反而比沒有更危險——因為它會給出看似可信的假答案。

兩個最容易漏的縫隙，本檔各有專門測試：

  1. **基本面的公布時滯**：`MonthlyRevenue.date` / `FinancialStatement.date`
     是**期間**不是公布日。2026-03-05 重放時，2 月營收符合 `date <= as_of`，
     但依法要 3/10 才公布。→ `TestVisibilityRules` / `TestNoFutureLeak`
  2. **外部 API**：重放時若還去打 FinMind/MOPS，抓回來的是「今天」的資料，
     既污染歷史情境也讓結果不可複現。→ `TestOfflineMode`

另有靜態守門測試禁止引擎層新增裸 `date.today()`（MASTER_PLAN §3 原則 4）。
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.pit import (
    financial_visible_cutoff,
    is_pit_replay,
    latest_visible_revenue_month,
    quarter_publish_deadline,
    revenue_visible_cutoff,
)
from src.discovery.scanner import MomentumScanner, ValueScanner

# ====================================================================== #
# A. 可見性規則（法定申報期限）
# ====================================================================== #


class TestVisibilityRules:
    @pytest.mark.parametrize(
        "as_of,expected",
        [
            ("2026-03-09", (2026, 1)),  # 未到 3/10 → 只看得到 1 月營收
            ("2026-03-10", (2026, 2)),  # 到期限 → 2 月營收可見
            ("2026-03-31", (2026, 2)),
            ("2026-01-05", (2025, 11)),  # 跨年回捲
            ("2026-01-10", (2025, 12)),
        ],
    )
    def test_latest_visible_revenue_month(self, as_of, expected):
        assert latest_visible_revenue_month(date.fromisoformat(as_of)) == expected

    def test_revenue_cutoff_is_month_end(self):
        assert revenue_visible_cutoff(date(2026, 3, 10)) == date(2026, 2, 28)
        assert revenue_visible_cutoff(date(2026, 1, 15)) == date(2025, 12, 31)

    @pytest.mark.parametrize(
        "year,quarter,expected",
        [
            (2026, 1, "2026-05-15"),  # 季後 45 日
            (2026, 2, "2026-08-14"),
            (2026, 3, "2026-11-14"),
            (2025, 4, "2026-03-31"),  # 年報：次年 3/31
        ],
    )
    def test_quarter_publish_deadline(self, year, quarter, expected):
        assert quarter_publish_deadline(year, quarter) == date.fromisoformat(expected)

    def test_financial_cutoff_moves_on_deadline(self):
        """5/14 仍只看得到去年 Q4；5/15 起才看得到今年 Q1。"""
        assert financial_visible_cutoff(date(2026, 5, 14)) == date(2025, 12, 31)
        assert financial_visible_cutoff(date(2026, 5, 15)) == date(2026, 3, 31)

    def test_is_pit_replay(self):
        today = date(2026, 8, 1)
        assert is_pit_replay(date(2026, 1, 1), today) is True
        assert is_pit_replay(today, today) is False
        assert is_pit_replay(None) is False


# ====================================================================== #
# B. as_of 注入
# ====================================================================== #


class TestAsOfInjection:
    def test_as_of_sets_scan_date(self, monkeypatch):
        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        _stub_run_deps(s, monkeypatch)
        s.run(as_of=date(2026, 3, 2))
        assert s.scan_date == date(2026, 3, 2)

    def test_default_is_today(self, monkeypatch):
        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        _stub_run_deps(s, monkeypatch)
        s.run()
        assert s.scan_date == date.today()
        assert s._is_offline() is False, "今日掃描不得進入 offline"

    def test_historical_as_of_enables_offline(self, monkeypatch):
        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        _stub_run_deps(s, monkeypatch)
        s.run(as_of=date(2026, 1, 5))
        assert s._is_offline() is True

    def test_as_of_helper_falls_back_before_run(self):
        """run() 尚未執行時 helper 仍可用（單元測試直接呼叫子函數的情境）。"""
        assert MomentumScanner(min_volume=1)._as_of() == date.today()


# ====================================================================== #
# C. 未來資料不得外洩（核心）
# ====================================================================== #


class TestNoFutureLeak:
    def _seed_revenue(self, session, months: list[tuple[int, int]]):
        from src.data.schema import MonthlyRevenue

        for y, m in months:
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            month_end = date(ny, nm, 1) - timedelta(days=1)
            session.add(
                MonthlyRevenue(
                    stock_id="1101",
                    date=month_end,
                    revenue=1000.0,
                    revenue_year=y,
                    revenue_month=m,
                    yoy_growth=float(m),
                    mom_growth=1.0,
                )
            )
        session.flush()

    def test_revenue_respects_publication_lag(self, db_session, monkeypatch):
        """2026-03-05 重放：2 月營收雖已在 DB，但依法 3/10 才公布 → 不得看到。"""
        import src.data.database as db_mod

        class _Ctx:
            def __enter__(self):
                return db_session

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx())
        self._seed_revenue(db_session, [(2025, 12), (2026, 1), (2026, 2)])

        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        s.scan_date = date(2026, 3, 5)
        df = s._load_revenue_data(["1101"], months=1)

        assert len(df) == 1
        assert df.iloc[0]["yoy_growth"] == pytest.approx(1.0), "應取 1 月營收（yoy=1），非 2 月（yoy=2）"

    def test_revenue_visible_after_deadline(self, db_session, monkeypatch):
        import src.data.database as db_mod

        class _Ctx:
            def __enter__(self):
                return db_session

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx())
        self._seed_revenue(db_session, [(2026, 1), (2026, 2)])

        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        s.scan_date = date(2026, 3, 10)  # 到公布期限
        df = s._load_revenue_data(["1101"], months=1)
        assert df.iloc[0]["yoy_growth"] == pytest.approx(2.0), "3/10 起應看得到 2 月營收"

    def test_shared_path_applies_same_pit_rule(self):
        """shared in-memory 路徑與 DB 路徑必須套用相同上界。

        歷史上 live/backtest 兩路徑漂移已造成 3 次 P0（MASTER_PLAN §7 B7），
        PIT 上界是新的漂移風險點，故明確鎖住。
        """
        from src.discovery.scanner._shared_load import SharedMarketData

        days = [date(2026, 2, 26), date(2026, 3, 4), date(2026, 3, 20)]  # 最後一筆在 as_of 之後
        price = pd.DataFrame(
            [
                {
                    "stock_id": "1101",
                    "date": d,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0,
                    "volume": 1000,
                    "turnover": 10000,
                }
                for d in days
            ]
        )
        shared = SharedMarketData(
            df_price=price,
            df_inst=pd.DataFrame(columns=["stock_id", "date", "name", "net"]),
            df_margin=pd.DataFrame(columns=["stock_id", "date", "margin_balance", "short_balance"]),
            df_revenue=pd.DataFrame(columns=["stock_id", "date", "yoy_growth", "mom_growth"]),
            price_cutoff=date(2026, 1, 1),
            revenue_cutoff=date(2025, 9, 1),
            loaded_at=pd.Timestamp.utcnow().to_pydatetime(),
        )

        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        s.scan_date = date(2026, 3, 5)
        df_price, *_ = s._slice_shared_market_data(shared, [], date(2026, 1, 1))

        assert df_price["date"].max() <= date(2026, 3, 5), "shared 路徑不得回傳 as_of 之後的價量"
        assert len(df_price) == 2


# ====================================================================== #
# D. offline mode
# ====================================================================== #


class TestOfflineMode:
    def test_pit_replay_skips_external_sync(self, monkeypatch):
        """PIT 重放不得呼叫任何外部補抓——否則抓回「今天」的資料污染歷史情境。"""
        called: list[str] = []
        import src.data.pipeline as pipeline_mod

        for fn in ("sync_revenue_for_stocks", "sync_valuation_for_stocks", "sync_broker_for_stocks"):
            if hasattr(pipeline_mod, fn):
                monkeypatch.setattr(pipeline_mod, fn, lambda ids, _n=fn, **kw: called.append(_n) or 0)

        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        _stub_run_deps(s, monkeypatch)
        s.run(as_of=date(2026, 1, 5))
        assert called == [], f"PIT 重放期間仍呼叫了外部 API：{called}"

    def test_today_scan_still_syncs(self, monkeypatch):
        """今日掃描維持原行為（不因 B1 而變成 offline）。"""
        called: list[str] = []
        import src.data.pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod, "sync_revenue_for_stocks", lambda ids, **kw: called.append("revenue") or 0)
        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        _stub_run_deps(s, monkeypatch)
        s.run()
        assert "revenue" in called

    def test_maybe_sync_valuation_guarded(self, monkeypatch):
        """縱深防禦：直接呼叫 _maybe_sync_valuation 也不得在 offline 下打 API。"""
        s = ValueScanner(min_volume=1, use_ic_adjustment=False)
        s.scan_date = date(2026, 1, 5)
        s._offline = True

        def _boom(*a, **kw):
            raise AssertionError("offline 下不得查 DB/API")

        monkeypatch.setattr("src.data.database.get_session", _boom)
        s._maybe_sync_valuation()  # 應直接返回


# ====================================================================== #
# E. 靜態守門：引擎層禁止裸 date.today()
# ====================================================================== #


class TestNoBareTodayInEngine:
    def test_scanner_engine_uses_as_of_only(self):
        """scanner 引擎層不得出現裸 `date.today()`（MASTER_PLAN §3 原則 4）。

        允許的例外只有兩處：`run()` 的 `as_of or date.today()` 注入點，
        以及 `_as_of()` 自身的 fallback。其餘一律應改用 `self._as_of()`，
        否則 PIT 重放會從那個縫隙混入真實今日。
        """
        # 允許的兩種形態：
        #   1. `self.scan_date = as_of or date.today()` —— run() 的唯一注入點
        #   2. `<var> = <某個參數> or date.today()` —— 純函數接受可注入日期，
        #      呼叫端傳 as_of；這正是 §3 原則 4 要求的形態（fallback 才是今日）
        injectable = re.compile(r"^\s*[\w.]+\s*=\s*\w+\s+or\s+date\.today\(\)")
        as_of_fallback = re.compile(r"^\s*return getattr\(self, \"scan_date\", None\) or date\.today\(\)")
        # regime 模組用 `datetime.date.today()`；同樣只允許「可注入」形態
        injectable_dt = re.compile(r"^\s*[\w.]+\s*=\s*\w+\s+or\s+datetime\.date\.today\(\)")
        inline_dt = re.compile(r"\(as_of or datetime\.date\.today\(\)\)")
        # 逐條列舉的合法例外——**每新增一條都必須寫明為何不是 look-ahead**。
        # 用精確字串而非放寬 regex，確保例外是刻意的、不會順手擴大。
        explicit_allow = {
            # 記錄「何時寫入」的掛鐘時間，非資料日；狀態機的冪等鍵是 data_date
            "today_str = datetime.date.today().isoformat()",
            # PIT 判定本身：必須跟真實今日比較才知道這是不是歷史重放
            "if as_of is not None and as_of < datetime.date.today():",
        }

        offenders: list[str] = []
        # 涵蓋 scanner **與 regime**：regime 驅動權重/門檻/模式封鎖，
        # 2026-08-04 實測 detect() 原本完全沒有時間上界，PIT 重放會用到今天的
        # TAIEX 與市場寬度——這個縫隙正是因為守門只掃 scanner 而漏掉的。
        paths = sorted(Path("src/discovery/scanner").glob("*.py")) + sorted(Path("src/regime").glob("*.py"))
        for path in paths:
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if "date.today()" not in stripped:
                    continue
                if stripped.startswith("#") or "`date.today()`" in stripped:
                    continue  # 註解 / docstring 內提及
                if injectable.match(line) or as_of_fallback.match(line):
                    continue
                if injectable_dt.match(line) or inline_dt.search(line):
                    continue
                if stripped in explicit_allow:
                    continue
                offenders.append(f"{path.name}:{i}: {stripped}")
        assert offenders == [], (
            "引擎層出現**不可注入**的 date.today()，PIT 重放會從此漏未來資料。\n"
            "修法：類別內改用 self._as_of()；純函數加 as_of 參數並由呼叫端傳入。\n" + "\n".join(offenders)
        )


# ====================================================================== #
# helpers
# ====================================================================== #


def _stub_run_deps(scanner, monkeypatch):
    """把 run() 的資料層與 regime 打樁，使測試不碰 DB / 外部 API。"""
    cls = type(scanner)
    days = [date(2026, 1, 2), date(2026, 1, 3)]
    price = pd.DataFrame(
        [
            {
                "stock_id": "1101",
                "date": d,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "volume": 500_000,
            }
            for d in days
        ]
    )
    empty = pd.DataFrame()
    monkeypatch.setattr(cls, "_load_market_data", lambda self: (price, empty, empty, empty))
    monkeypatch.setattr(cls, "_load_revenue_data", lambda self, ids=None, months=1: empty)
    monkeypatch.setattr(cls, "_load_announcement_data", lambda self, ids: (empty, empty))
    monkeypatch.setattr(
        cls,
        "_coarse_filter",
        lambda self, dp, di: pd.DataFrame({"stock_id": ["1101"], "close": [10.0], "volume": [500_000]}),
    )
    monkeypatch.setattr(
        cls,
        "_score_candidates",
        lambda self, c, *a, **kw: pd.DataFrame({"stock_id": ["1101"], "close": [10.0], "composite_score": [0.9]}),
    )
    monkeypatch.setattr(cls, "_rank_and_enrich", lambda self, s: s.assign(rank=[1]))
    monkeypatch.setattr(cls, "_compute_sector_summary", lambda self, r: pd.DataFrame())
    monkeypatch.setattr(cls, "_log_factor_effectiveness", lambda self: None)
    monkeypatch.setattr(cls, "get_sub_factor_df", lambda self: pd.DataFrame())

    import src.regime.detector as det_mod

    class _FakeDetector:
        def detect(self, as_of=None):
            # 需接受 as_of：Stage 0 自 B1 起以 detect(as_of=self.scan_date) 呼叫，
            # 簽名不符會拋例外並被 Stage 0 吞掉 → regime 退回 sideways → 模式被封鎖
            return {"regime": "bull", "taiex_close": 20000.0}

    monkeypatch.setattr(det_mod, "MarketRegimeDetector", lambda *a, **kw: _FakeDetector())


# ====================================================================== #
# F. regime 的 PIT 化（2026-08-04 補）
# ====================================================================== #


class TestRegimePIT:
    """regime 驅動評分權重、分數門檻、ATR 倍數、universe 乘數與 REGIME_MODE_BLOCK。

    PIT 重放若沿用今日 regime，重放結果毫無意義。實測 `detect()` 原本三個查詢
    （TAIEX / TW_VIX / US_VIX）**完全沒有時間上界**，且 `_compute_breadth()`
    直接取 `MAX(DailyFeature.date)`＝今日。這個縫隙之所以漏掉，是因為靜態守門
    當時只掃 `src/discovery/scanner`——現已擴及 `src/regime`。
    """

    def _seed_taiex(self, session, closes: list[float], end: date):
        from src.data.schema import DailyPrice

        for i, c in enumerate(reversed(closes)):
            d = end - timedelta(days=i)
            session.add(DailyPrice(stock_id="TAIEX", date=d, open=c, high=c, low=c, close=c, volume=0, turnover=0))
        session.flush()

    def test_detect_respects_as_of_upper_bound(self, db_session, monkeypatch):
        """as_of 之後的 TAIEX 不得影響判定。"""
        import src.data.database as db_mod
        from src.regime.detector import MarketRegimeDetector

        class _Ctx:
            def __enter__(self):
                return db_session

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx())
        # 前 130 天穩定上升（bull），as_of 之後暴跌（若洩題會變 crisis）
        rising = [15000 + i * 30 for i in range(130)]
        self._seed_taiex(db_session, rising, date(2026, 3, 1))
        crash = [18000, 16000, 14000, 12000]
        self._seed_taiex(db_session, crash, date(2026, 3, 5))

        r = MarketRegimeDetector().detect(as_of=date(2026, 3, 1))
        assert r["regime"] != "crisis", "as_of 之後的崩盤不得洩入歷史判定"
        assert r["taiex_close"] == pytest.approx(rising[-1])

    def test_pit_replay_does_not_persist_state(self, db_session, monkeypatch):
        """歷史重放為 read-only——不得推進 live 狀態機。"""
        import src.data.database as db_mod
        from src.data.schema import RegimeStateLog
        from src.regime.detector import MarketRegimeDetector

        class _Ctx:
            def __enter__(self):
                return db_session

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx())
        self._seed_taiex(db_session, [15000 + i * 30 for i in range(130)], date(2026, 3, 1))

        before = db_session.query(RegimeStateLog).count()
        r = MarketRegimeDetector().detect(as_of=date(2026, 3, 1))
        after = db_session.query(RegimeStateLog).count()

        assert after == before, "PIT 重放不得寫入 RegimeStateLog"
        assert r["state_advanced"] is False
        assert r["transition_info"]["reason"] == "pit_replay_readonly"

    def test_breadth_respects_as_of(self, db_session, monkeypatch):
        """市場寬度取 <= as_of 的最新 DailyFeature，非今日。"""
        import src.data.database as db_mod
        from src.data.schema import DailyFeature
        from src.regime.detector import MarketRegimeDetector

        class _Ctx:
            def __enter__(self):
                return db_session

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx())
        for d, below in ((date(2026, 3, 1), False), (date(2026, 6, 1), True)):
            for i in range(10):
                close, ma20 = (90.0, 100.0) if below else (110.0, 100.0)
                db_session.add(
                    DailyFeature(stock_id=f"{1000 + i}", date=d, close=close, volume=1, turnover=1, ma20=ma20)
                )
        db_session.flush()

        assert MarketRegimeDetector._compute_breadth(date(2026, 3, 1)) == pytest.approx(0.0)
        assert MarketRegimeDetector._compute_breadth(date(2026, 6, 1)) == pytest.approx(1.0)
