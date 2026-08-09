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


# ====================================================================== #
# F. 資料覆蓋度（§6.5 #21b）——區分「模式不進場」與「輸入資料缺席」
# ====================================================================== #


class TestDataCoverage:
    """`n_picks == 0` 有兩種意義，混為一談會讓無效結果被當成結論。

    2026-08-04 的跨模式重放就是實例：dividend「30 天只選得出 4 天」被記錄為
    模式產能，真因是 `stock_valuation` 在 2026-01-26 前完全沒有資料。
    """

    def _patch_session(self, db_session, monkeypatch):
        """pit_replay 以 `from ... import get_session` 綁定，須 patch 該模組自身的名稱。"""
        import src.discovery.pit_replay as pr_mod

        class _Ctx:
            def __enter__(self):
                return db_session

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(pr_mod, "get_session", lambda: _Ctx())

    def _seed_market(self, session, as_of, *, n_stocks=1600, with_feature=True):
        from src.data.schema import DailyFeature, DailyPrice

        for i in range(n_stocks):
            sid = f"{1000 + i}"
            session.add(
                DailyPrice(
                    stock_id=sid, date=as_of, open=10.0, high=10.0, low=10.0, close=10.0, volume=1000, turnover=10000
                )
            )
            if with_feature:
                # ma60/turnover_ma20 必填——§6.5 #21d 起特徵的判定同時看列數與欄位暖身，
                # 留空會讓本類每個測試都因暖身失效而 missing，測不到各自要驗的那一軸
                session.add(
                    DailyFeature(
                        stock_id=sid,
                        date=as_of,
                        close=10.0,
                        volume=1000,
                        turnover=10000,
                        ma60=10.0,
                        turnover_ma20=10000.0,
                    )
                )
        session.flush()

    def test_value_without_valuation_is_no_data(self, db_session, monkeypatch):
        """價量齊備但估值表為空 → value 的結果不可採信，而非「模式不進場」。"""
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2026, 3, 5)
        self._seed_market(db_session, as_of)

        cov = assess_data_coverage("value", as_of)
        assert cov.missing == ("stock_valuation",)
        assert cov.sufficient is False
        assert "stock_valuation" in cov.describe()

    def test_momentum_unaffected_by_valuation_gap(self, db_session, monkeypatch):
        """同一天、同一份資料，momentum 不依賴估值 → 必須判為就緒。

        這是本機制的關鍵性質：可採信與否是**per-mode** 的，一律看全部表會把
        純價量模式的有效結果誤殺。
        """
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2026, 3, 5)
        self._seed_market(db_session, as_of)

        assert assess_data_coverage("momentum", as_of).sufficient is True
        assert assess_data_coverage("swing", as_of).sufficient is True

    def test_future_valuation_rows_do_not_count(self, db_session, monkeypatch):
        """as_of 之後的估值列不得計入覆蓋率——否則「當時還沒補」的日子會被誤判為就緒。"""
        from src.data.schema import StockValuation
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2026, 3, 5)
        self._seed_market(db_session, as_of)
        for i in range(800):  # 全部落在 as_of 之後
            db_session.add(StockValuation(stock_id=f"{1000 + i}", date=as_of + timedelta(days=1), pe_ratio=10.0))
        db_session.flush()

        cov = assess_data_coverage("value", as_of)
        assert cov.valuation_stocks == 0
        assert cov.missing == ("stock_valuation",)

    def test_stale_valuation_does_not_count(self, db_session, monkeypatch):
        """窗口外的舊估值不算數——與 Stage 0.5 閘門同一判準（§6.5 #22）。"""
        from src.data.schema import StockValuation
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2026, 3, 5)
        self._seed_market(db_session, as_of)
        for i in range(800):
            db_session.add(StockValuation(stock_id=f"{1000 + i}", date=as_of - timedelta(days=60), pe_ratio=10.0))
        db_session.flush()

        assert assess_data_coverage("value", as_of).sufficient is False

    def test_fresh_valuation_is_sufficient(self, db_session, monkeypatch):
        """對照組：窗口內足量估值 → value 判為就緒（證明上面兩題不是恆 False）。"""
        from src.data.schema import StockValuation
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2026, 3, 5)
        self._seed_market(db_session, as_of)
        for i in range(800):
            db_session.add(StockValuation(stock_id=f"{1000 + i}", date=as_of - timedelta(days=1), pe_ratio=10.0))
        db_session.flush()

        cov = assess_data_coverage("value", as_of)
        assert cov.valuation_stocks == 800
        assert cov.sufficient is True

    def test_revenue_coverage_respects_publication_lag(self, db_session, monkeypatch):
        """覆蓋率本身也要套公布時滯——否則會把「當時看不到的營收」算成已就緒。"""
        from src.data.schema import MonthlyRevenue
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2026, 3, 5)  # 未到 3/10，2 月營收依法尚未公布
        after = date(2026, 3, 10)
        self._seed_market(db_session, as_of)
        self._seed_market(db_session, after)  # 兩天價量都齊備，使差異只來自營收可見性
        for i in range(500):  # 全部是 2 月營收
            db_session.add(
                MonthlyRevenue(
                    stock_id=f"{1000 + i}",
                    date=date(2026, 2, 28),
                    revenue=1000.0,
                    revenue_year=2026,
                    revenue_month=2,
                    yoy_growth=20.0,
                    mom_growth=1.0,
                )
            )
        db_session.flush()

        assert assess_data_coverage("growth", as_of).missing == ("monthly_revenue",)
        # 3/10 起同一份資料變為可見
        assert assess_data_coverage("growth", after).sufficient is True

    def test_thin_market_day_flags_price_gap(self, db_session, monkeypatch):
        """半套日（普通股遠少於全市場）→ 連 momentum 都不可採信。"""
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2026, 3, 5)
        self._seed_market(db_session, as_of, n_stocks=800)

        assert assess_data_coverage("momentum", as_of).missing == ("daily_price",)

    def test_missing_features_flags_gap(self, db_session, monkeypatch):
        """DailyFeature 未回補 → universe Stage 2 已 fallback，結果不同質。"""
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2026, 3, 5)
        self._seed_market(db_session, as_of, with_feature=False)

        assert assess_data_coverage("momentum", as_of).missing == ("daily_feature",)

    @pytest.mark.parametrize(
        "sufficient,n_picks,expected",
        [
            (True, 3, "ok"),
            (True, 0, "no_picks"),
            (False, 0, "no_data"),
            (False, 3, "no_data"),  # 資料缺席時即使有選股也不可採信（退化漏斗的產物）
        ],
    )
    def test_verdict_matrix(self, sufficient, n_picks, expected):
        from src.discovery.pit_replay import DataCoverage, ReplayResult

        cov = DataCoverage(
            as_of=date(2026, 3, 5),
            mode="value",
            price_stocks=1600,
            feature_stocks=1600,
            valuation_stocks=0 if not sufficient else 800,
            revenue_stocks=0,
            required=("stock_valuation",),
            missing=() if sufficient else ("stock_valuation",),
        )
        res = ReplayResult(
            as_of=date(2026, 3, 5),
            mode="value",
            regime="bull",
            total_stocks=1600,
            after_coarse=150,
            picks=pd.DataFrame({"stock_id": [f"{i}" for i in range(n_picks)]}),
            coverage=cov,
        )
        assert res.verdict == expected

    def test_every_scanner_mode_declares_requirements(self):
        """契約測試：新增模式若忘了登記依賴，會預設「恆就緒」而靜默產出無效結果。"""
        from src.discovery.pit_replay import MODE_REQUIRED_TABLES

        source = Path("src/discovery/pit_replay.py").read_text(encoding="utf-8")
        modes = set(re.findall(r'"(\w+)": \w+Scanner,', source))
        assert modes, "未能從 scanner_map 解析出模式清單——此測試需同步更新"
        assert modes <= set(MODE_REQUIRED_TABLES), f"下列模式未登記資料依賴：{modes - set(MODE_REQUIRED_TABLES)}"


class TestFeatureWarmup:
    """§6.5 #21d：列數足夠**不代表**欄位可用。

    `daily_feature` 的 MA60 需 60 個交易日才填滿，回補範圍頭幾十天欄位全是 NaN，
    而列數檢查完全看不出來——與 fail-open 同一類的靜默失效。實測 2020-01-02 的
    ma60/turnover_ma20 非空率皆為 **0.000**，卻有 5,086 列特徵。

    後果不只是分數不準：`universe.py:125` 對 `turnover_ma20` 為 NaN 的個股**跳過
    Stage 2 流動性門檻**，暖身期等於流動性過濾整段消失。
    """

    def _patch_session(self, db_session, monkeypatch):
        import src.discovery.pit_replay as pr_mod

        class _Ctx:
            def __enter__(self):
                return db_session

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(pr_mod, "get_session", lambda: _Ctx())

    def _seed(self, session, as_of, *, n=1600, warm=True):
        from src.data.schema import DailyFeature, DailyPrice

        for i in range(n):
            sid = f"{1000 + i}"
            session.add(
                DailyPrice(
                    stock_id=sid, date=as_of, open=10.0, high=10.0, low=10.0, close=10.0, volume=1000, turnover=10000
                )
            )
            session.add(
                DailyFeature(
                    stock_id=sid,
                    date=as_of,
                    close=10.0,
                    volume=1000,
                    turnover=10000,
                    ma60=10.0 if warm else None,
                    turnover_ma20=10000.0 if warm else None,
                )
            )
        session.flush()

    def test_rows_present_but_columns_null_is_no_data(self, db_session, monkeypatch):
        """列數滿額但 MA 欄位全 NaN → 必須判為不可採信。這是本項的核心回歸。"""
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2020, 1, 2)
        self._seed(db_session, as_of, warm=False)

        cov = assess_data_coverage("momentum", as_of)
        assert cov.feature_stocks == 1600, "列數本身是足夠的——正是舊版看不出問題的原因"
        assert cov.feature_warm_ratio == 0.0
        assert cov.missing == ("daily_feature",)
        assert "未暖身" in cov.describe()

    def test_warm_columns_are_sufficient(self, db_session, monkeypatch):
        """對照組：欄位填滿時判為就緒，證明上一題不是恆 False。"""
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2020, 7, 7)
        self._seed(db_session, as_of, warm=True)

        cov = assess_data_coverage("momentum", as_of)
        assert cov.feature_warm_ratio == 1.0
        assert cov.sufficient is True

    def test_steady_state_null_rate_passes(self, db_session, monkeypatch):
        """遠低於實測穩態、但仍在門檻之上的覆蓋率不得被誤殺。

        4 碼普通股的實測穩態為 **0.988~0.998**（不限 4 碼才會掉到 0.646~0.786，
        因權證上市時間短）。本測試刻意取 0.675 這個遠低於穩態的值，確認門檻 0.5
        的判定邊界是「暖身失效（0.0）vs 其他」而非貼著穩態——否則穩態稍有波動
        就會整批誤殺。
        """
        from src.data.schema import DailyFeature, DailyPrice
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2023, 6, 1)
        for i in range(1600):
            sid = f"{1000 + i}"
            db_session.add(
                DailyPrice(
                    stock_id=sid, date=as_of, open=10.0, high=10.0, low=10.0, close=10.0, volume=1000, turnover=10000
                )
            )
            db_session.add(
                DailyFeature(
                    stock_id=sid,
                    date=as_of,
                    close=10.0,
                    volume=1000,
                    turnover=10000,
                    ma60=10.0 if i < 1080 else None,  # 67.5% 非空，落在實測穩態帶
                    turnover_ma20=10000.0 if i < 1450 else None,  # 90.6%
                )
            )
        db_session.flush()

        cov = assess_data_coverage("momentum", as_of)
        assert cov.feature_warm_ratio == pytest.approx(0.675)
        assert cov.sufficient is True, "穩態的 NaN 率不得被當成暖身失效"

    def test_binding_constraint_is_the_lower_column(self, db_session, monkeypatch):
        """取兩欄的**較小值**——只要有一道閘門的輸入不可用，結果就不可採信。"""
        from src.data.schema import DailyFeature, DailyPrice
        from src.discovery.pit_replay import assess_data_coverage

        self._patch_session(db_session, monkeypatch)
        as_of = date(2020, 1, 20)
        for i in range(1600):
            sid = f"{1000 + i}"
            db_session.add(
                DailyPrice(
                    stock_id=sid, date=as_of, open=10.0, high=10.0, low=10.0, close=10.0, volume=1000, turnover=10000
                )
            )
            db_session.add(
                DailyFeature(
                    stock_id=sid,
                    date=as_of,
                    close=10.0,
                    volume=1000,
                    turnover=10000,
                    ma60=None,  # 20 個交易日時 MA60 尚未填滿
                    turnover_ma20=10000.0,  # 但 20 日窗口的欄位已可用（實測 0.82）
                )
            )
        db_session.flush()

        cov = assess_data_coverage("momentum", as_of)
        assert cov.feature_warm_ratio == 0.0, "ma60 為 0 即整體為 0，不得被 turnover_ma20 稀釋"
        assert cov.sufficient is False
