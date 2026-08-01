"""N2 — 五 scanner 漏斗階段序列 parity 測試（2026-08-01）。

背景（`logs/audit_discover_20260731/REPORT.md` §5）：`value`/`growth`/`dividend`
各自覆寫 `run()`（複製貼上版），跳過分數門檻 / 產業分散 / 回撤縮表 / 負面消息閘門
/ audit_trail / sub_factor 等階段，五個 scanner 實際上不是同一條 pipeline，
造成跨模式不可比（「value/dividend 天天穩定 20 筆」是閘門覆蓋率差異的產物）。

N2 重構將 `run()` 收斂為單一實作 + 宣告式 `StageConfig`。本檔是重構的**安全網**：

  以 stage 呼叫序列為 parity 基準——stage 方法本身未被修改，因此只要每個 mode
  呼叫的階段「有哪些、什麼順序」不變，選股行為即等價。基準序列於重構**之前**
  擷取自當時的 `run()` 實作（見各 `_EXPECTED_*` 常數），重構後必須逐一相符。

用法：`pytest tests/test_scanner_pipeline_parity.py -v`
若要重新擷取基準（僅在**刻意**改變階段組成時）：`python -m tests.test_scanner_pipeline_parity`
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.discovery.scanner import (
    DividendScanner,
    GrowthScanner,
    MomentumScanner,
    SwingScanner,
    ValueScanner,
)

# 會被追蹤的階段方法（scanner 上所有「漏斗階段」層級的方法）
_TRACKED_METHODS: tuple[str, ...] = (
    "_maybe_sync_valuation",
    "_reload_valuation",
    "_coarse_filter",
    "_score_candidates",
    "_apply_overlap_bonus",
    "_apply_sector_bonus",
    "_apply_sector_relative_strength",
    "_apply_concept_bonus",
    "_apply_peer_fundamental_ranking",
    "_apply_weekly_trend_bonus",
    "_apply_momentum_decay",
    "_apply_institutional_acceleration",
    "_apply_chip_macd",
    "_apply_key_player_cost",
    "_apply_negative_news_gate",
    "_apply_volume_price_divergence",
    "_apply_risk_filter",
    "_apply_crisis_filter",
    "_apply_multi_timeframe_alignment",
    "_apply_score_threshold",
    "_log_factor_effectiveness",
    "_rank_and_enrich",
    "_apply_sector_diversification",
    "_compute_drawdown_adjusted_top_n",
    "_compute_sector_summary",
    "_audit_chip_tier_changes",
    "get_sub_factor_df",
)

_SCANNERS = {
    "momentum": MomentumScanner,
    "swing": SwingScanner,
    "value": ValueScanner,
    "dividend": DividendScanner,
    "growth": GrowthScanner,
}


def _stub_frames() -> tuple[pd.DataFrame, ...]:
    """最小可跑的四張市場資料表（兩檔股票 × 三日）。"""
    days = [date(2026, 7, 29) + timedelta(days=i) for i in range(3)]
    price = pd.DataFrame(
        [
            {
                "stock_id": sid,
                "date": d,
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1_000_000,
            }
            for sid in ("1101", "1102")
            for d in days
        ]
    )
    inst = pd.DataFrame([{"stock_id": "1101", "date": days[-1], "name": "外資買賣超", "net": 100}])
    margin = pd.DataFrame(columns=["stock_id", "date", "margin_balance", "short_balance"])
    revenue = pd.DataFrame([{"stock_id": "1101", "yoy_growth": 10.0, "mom_growth": 5.0}])
    return price, inst, margin, revenue


def trace_stages(scanner_cls, monkeypatch, regime: str = "crisis") -> list[str]:
    """跑一次 run()，回傳實際被呼叫的階段名稱（依呼叫順序，去重連續重複）。"""
    calls: list[str] = []
    price, inst, margin, revenue = _stub_frames()
    candidates = pd.DataFrame(
        {
            "stock_id": ["1101", "1102"],
            "close": [100.0, 100.0],
            "volume": [1_000_000, 1_000_000],
            "composite_score": [0.8, 0.7],
        }
    )

    # ── 資料層 stub（不碰 DB / 不碰外部 API）──
    monkeypatch.setattr(scanner_cls, "_load_market_data", lambda self: (price, inst, margin, revenue))
    monkeypatch.setattr(scanner_cls, "_load_revenue_data", lambda self, ids=None, months=1: revenue)
    monkeypatch.setattr(scanner_cls, "_load_announcement_data", lambda self, ids: (pd.DataFrame(), pd.DataFrame()))

    import src.data.pipeline as pipeline_mod

    for fn in ("sync_revenue_for_stocks", "sync_valuation_for_stocks", "sync_broker_for_stocks"):
        if hasattr(pipeline_mod, fn):
            monkeypatch.setattr(pipeline_mod, fn, lambda ids, **kw: 0)
    if hasattr(pipeline_mod, "sync_mops_revenue"):
        monkeypatch.setattr(pipeline_mod, "sync_mops_revenue", lambda **kw: 0)

    # regime 固定，避免依賴 DB 狀態
    import src.regime.detector as det_mod

    class _FakeDetector:
        def detect(self):
            return {"regime": regime, "taiex_close": 20000.0}

    monkeypatch.setattr(det_mod, "MarketRegimeDetector", lambda *a, **kw: _FakeDetector())

    # ── 階段方法 spy：記錄名稱後回傳「不改變流程」的值 ──
    def _make_spy(name: str, original):
        def _spy(self, *args, **kwargs):
            calls.append(name)
            if name == "_coarse_filter":
                return candidates.copy()
            if name == "_score_candidates":
                return candidates.copy()
            if name == "_rank_and_enrich":
                out = candidates.copy()
                out["rank"] = range(1, len(out) + 1)
                return out
            if name == "_apply_sector_diversification":
                df = args[0]
                return df, set(), set(df["stock_id"])
            if name == "_compute_drawdown_adjusted_top_n":
                return 20
            if name == "_compute_sector_summary":
                return pd.DataFrame()
            if name == "get_sub_factor_df":
                return pd.DataFrame()
            if name in ("_maybe_sync_valuation", "_reload_valuation", "_log_factor_effectiveness"):
                return None
            # 其餘軟加成 / 硬風控：原樣回傳輸入的 DataFrame
            return args[0] if args and isinstance(args[0], pd.DataFrame) else None

        return _spy

    for name in _TRACKED_METHODS:
        if hasattr(scanner_cls, name):
            monkeypatch.setattr(scanner_cls, name, _make_spy(name, getattr(scanner_cls, name)))

    scanner = scanner_cls(min_volume=1, use_ic_adjustment=False)
    scanner.run()
    return calls


# ══════════════════════════════════════════════════════════════════════ #
# 重構前擷取的基準序列（ground truth）——重構後必須完全相同
# 擷取時間：2026-08-01，N2 重構前的 HEAD
#
# ── 基準異動紀錄（每次移動都必須寫明原因）────────────────────────────
# 2026-08-01：value / dividend / growth 新增 `_compute_drawdown_adjusted_top_n`
#   ——N2 閘門政策第一項「開啟 4.2 回撤縮表」的**刻意**行為變更，非重構漂移。
#   已逐一比對確認 delta 僅此一個階段、且無任何階段被移除。
#   value/dividend 屬 `_DEFENSIVE_MODES` 故實際輸出不變；growth 於 TAIEX
#   20 日回撤 <-10% 時 top_n 砍半、<-15% 砍至 1/3。
# ══════════════════════════════════════════════════════════════════════ #

# 涵蓋三個 regime：bull（全模式正常路徑）、sideways（momentum/growth 被 REGIME_MODE_BLOCK
# 封鎖）、crisis（growth 封鎖 + crisis filter 生效）——同時鎖住 Stage 0.1 gating 行為。
_REGIMES: tuple[str, ...] = ("bull", "sideways", "crisis")

_EXPECTED: dict[tuple[str, str], list[str]] = {
    ("momentum", "bull"): [
        "_coarse_filter",
        "_score_candidates",
        "_apply_overlap_bonus",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_peer_fundamental_ranking",
        "_apply_momentum_decay",
        "_apply_institutional_acceleration",
        "_apply_chip_macd",
        "_apply_key_player_cost",
        "_apply_negative_news_gate",
        "_apply_volume_price_divergence",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_apply_multi_timeframe_alignment",
        "_apply_score_threshold",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_apply_sector_diversification",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "_audit_chip_tier_changes",
        "get_sub_factor_df",
    ],
    ("momentum", "sideways"): [],  # Stage 0.1 REGIME_MODE_BLOCK 封鎖
    ("momentum", "crisis"): [
        "_coarse_filter",
        "_score_candidates",
        "_apply_overlap_bonus",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_peer_fundamental_ranking",
        "_apply_momentum_decay",
        "_apply_institutional_acceleration",
        "_apply_chip_macd",
        "_apply_key_player_cost",
        "_apply_negative_news_gate",
        "_apply_volume_price_divergence",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_apply_multi_timeframe_alignment",
        "_apply_score_threshold",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_apply_sector_diversification",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "_audit_chip_tier_changes",
        "get_sub_factor_df",
    ],
    ("swing", "bull"): [
        "_coarse_filter",
        "_score_candidates",
        "_apply_overlap_bonus",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_peer_fundamental_ranking",
        "_apply_momentum_decay",
        "_apply_institutional_acceleration",
        "_apply_chip_macd",
        "_apply_key_player_cost",
        "_apply_negative_news_gate",
        "_apply_volume_price_divergence",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_apply_multi_timeframe_alignment",
        "_apply_score_threshold",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_apply_sector_diversification",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "_audit_chip_tier_changes",
        "get_sub_factor_df",
    ],
    ("swing", "sideways"): [
        "_coarse_filter",
        "_score_candidates",
        "_apply_overlap_bonus",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_peer_fundamental_ranking",
        "_apply_momentum_decay",
        "_apply_institutional_acceleration",
        "_apply_chip_macd",
        "_apply_key_player_cost",
        "_apply_negative_news_gate",
        "_apply_volume_price_divergence",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_apply_multi_timeframe_alignment",
        "_apply_score_threshold",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_apply_sector_diversification",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "_audit_chip_tier_changes",
        "get_sub_factor_df",
    ],
    ("swing", "crisis"): [
        "_coarse_filter",
        "_score_candidates",
        "_apply_overlap_bonus",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_peer_fundamental_ranking",
        "_apply_momentum_decay",
        "_apply_institutional_acceleration",
        "_apply_chip_macd",
        "_apply_key_player_cost",
        "_apply_negative_news_gate",
        "_apply_volume_price_divergence",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_apply_multi_timeframe_alignment",
        "_apply_score_threshold",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_apply_sector_diversification",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "_audit_chip_tier_changes",
        "get_sub_factor_df",
    ],
    ("value", "bull"): [
        "_maybe_sync_valuation",
        "_coarse_filter",
        "_reload_valuation",
        "_score_candidates",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "get_sub_factor_df",
    ],
    ("value", "sideways"): [
        "_maybe_sync_valuation",
        "_coarse_filter",
        "_reload_valuation",
        "_score_candidates",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "get_sub_factor_df",
    ],
    ("value", "crisis"): [
        "_maybe_sync_valuation",
        "_coarse_filter",
        "_reload_valuation",
        "_score_candidates",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "get_sub_factor_df",
    ],
    ("dividend", "bull"): [
        "_maybe_sync_valuation",
        "_coarse_filter",
        "_reload_valuation",
        "_score_candidates",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "get_sub_factor_df",
    ],
    ("dividend", "sideways"): [
        "_maybe_sync_valuation",
        "_coarse_filter",
        "_reload_valuation",
        "_score_candidates",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "get_sub_factor_df",
    ],
    ("dividend", "crisis"): [
        "_maybe_sync_valuation",
        "_coarse_filter",
        "_reload_valuation",
        "_score_candidates",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "get_sub_factor_df",
    ],
    ("growth", "bull"): [
        "_coarse_filter",
        "_score_candidates",
        "_apply_sector_bonus",
        "_apply_sector_relative_strength",
        "_apply_concept_bonus",
        "_apply_risk_filter",
        "_apply_crisis_filter",
        "_log_factor_effectiveness",
        "_rank_and_enrich",
        "_compute_drawdown_adjusted_top_n",
        "_compute_sector_summary",
        "get_sub_factor_df",
    ],
    ("growth", "sideways"): [],  # Stage 0.1 REGIME_MODE_BLOCK 封鎖
    ("growth", "crisis"): [],  # Stage 0.1 REGIME_MODE_BLOCK 封鎖
}


# 純觀測階段：只讀取 / 只寫 log，不改動 scored 或 rankings，因此**不影響選股**。
# N2 將這些補齊至五模式（重構前 value/dividend/growth 沒有），故 parity 比對時排除。
_OBSERVATION_STAGES: frozenset[str] = frozenset(
    {
        "_log_factor_effectiveness",  # E2 因子 IC 日誌
        "get_sub_factor_df",  # 子因子 rank 收集（IC 診斷）
    }
)


def _selection_only(seq: list[str]) -> list[str]:
    return [s for s in seq if s not in _OBSERVATION_STAGES]


@pytest.mark.parametrize("mode", list(_SCANNERS))
@pytest.mark.parametrize("regime", _REGIMES)
def test_selection_stage_sequence_matches_baseline(mode, regime, monkeypatch):
    """**選股行為 parity**：五模式 × 三 regime 的選股相關階段序列必須與重構前完全相同。

    這是 N2 重構的核心安全網——使用者已裁定「宣告式統一、先保留現行行為」，
    因此重構**不得**改變任何一個模式選出哪些股票。純觀測階段另由下方測試覆蓋。
    """
    traced = trace_stages(_SCANNERS[mode], monkeypatch, regime=regime)
    assert _selection_only(traced) == _selection_only(_EXPECTED[(mode, regime)])


@pytest.mark.parametrize("mode", list(_SCANNERS))
def test_observation_stages_now_cover_all_modes(mode, monkeypatch):
    """N2 增益：五模式一律產生 IC 日誌與子因子資料（重構前僅 momentum/swing 有）。

    純觀測、不影響選股——這是使用者選項 A 明列的「觀測增益」部分。
    以 bull 取樣（momentum/growth 在 sideways/crisis 會被 REGIME_MODE_BLOCK 擋掉）。
    """
    traced = trace_stages(_SCANNERS[mode], monkeypatch, regime="bull")
    assert _OBSERVATION_STAGES.issubset(set(traced)), f"{mode} 缺少觀測階段：{_OBSERVATION_STAGES - set(traced)}"


def test_baseline_covers_every_mode_and_regime():
    """基準表本身的完整性守門——避免漏擷取而讓 parity 名存實亡。"""
    assert set(_EXPECTED) == {(m, r) for m in _SCANNERS for r in _REGIMES}


def test_no_scanner_overrides_run():
    """N2 契約：`run()` 只允許有單一實作，模式差異一律走 `_STAGES` 與 hook。"""
    from src.discovery.scanner._base import MarketScanner

    offenders = [name for name, cls in _SCANNERS.items() if "run" in vars(cls)]
    assert offenders == [], f"以下 scanner 仍覆寫 run()，違反 N2 單一漏斗契約：{offenders}"
    assert "run" in vars(MarketScanner)


if __name__ == "__main__":  # pragma: no cover - 基準擷取工具
    import logging
    import pprint

    from _pytest.monkeypatch import MonkeyPatch

    logging.disable(logging.CRITICAL)  # 只輸出基準字典，方便直接貼回 _EXPECTED

    captured: dict[tuple[str, str], list[str]] = {}
    for _mode, _cls in _SCANNERS.items():
        for _regime in _REGIMES:
            _mp = MonkeyPatch()
            try:
                captured[(_mode, _regime)] = trace_stages(_cls, _mp, regime=_regime)
            finally:
                _mp.undo()
    pprint.pp(captured, width=100, sort_dicts=False)


# ══════════════════════════════════════════════════════════════════════ #
# Stage 4.2 回撤縮表（N2 閘門政策第一項，2026-08-01 開啟）
# ══════════════════════════════════════════════════════════════════════ #


class TestDrawdownAdjustedTopN:
    """4.2 的 TAIEX 取數修復 + 各模式行為。

    修復前 `_compute_drawdown_adjusted_top_n` 從 `df_price` 撈 TAIEX，但 TAIEX
    不在任何 scanner 的 universe 內，而 momentum/swing/value/dividend 的
    `_load_market_data` 都以 `stock_id.in_(universe_ids)` 過濾 → `taiex.empty`
    恆成立 → **這道風控從未實際觸發過**。改為 df_price 缺 TAIEX 時直接查 DB。
    """

    @staticmethod
    def _taiex_df(closes: list[float]) -> pd.DataFrame:
        base = date(2026, 6, 1)
        return pd.DataFrame(
            [{"stock_id": "TAIEX", "date": base + timedelta(days=i), "close": c} for i, c in enumerate(closes)]
        )

    def _seed_taiex(self, session, closes: list[float], end: date):
        from src.data.schema import DailyPrice

        for i, c in enumerate(reversed(closes)):
            d = end - timedelta(days=i)
            session.add(DailyPrice(stock_id="TAIEX", date=d, open=c, high=c, low=c, close=c, volume=0, turnover=0))
        session.flush()

    def test_uses_df_price_taiex_when_present(self):
        """growth 走全市場載入時 df_price 內就有 TAIEX，應直接使用不查 DB。"""
        s = GrowthScanner(min_volume=1, use_ic_adjustment=False)
        s.scan_date = date(2026, 7, 1)
        flat = self._taiex_df([100.0] * 19 + [80.0])  # 回撤 -20%
        assert s._taiex_drawdown_20d(flat) == pytest.approx(-0.20)

    def test_falls_back_to_db_when_universe_filtered_out(self, db_session, monkeypatch):
        """**核心回歸**：df_price 無 TAIEX（universe 過濾掉）時仍須算得出回撤。"""
        import src.data.database as db_mod

        class _Ctx:
            def __enter__(self):
                return db_session

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx())
        end = date(2026, 7, 1)
        self._seed_taiex(db_session, [100.0] * 19 + [85.0], end)

        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        s.scan_date = end
        only_stocks = pd.DataFrame([{"stock_id": "1101", "date": end, "close": 50.0}])
        assert s._taiex_drawdown_20d(only_stocks) == pytest.approx(-0.15)

    def test_insufficient_history_returns_none(self, db_session, monkeypatch):
        import src.data.database as db_mod

        class _Ctx:
            def __enter__(self):
                return db_session

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(db_mod, "get_session", lambda: _Ctx())
        s = MomentumScanner(min_volume=1, use_ic_adjustment=False)
        s.scan_date = date(2026, 7, 1)
        assert s._taiex_drawdown_20d(pd.DataFrame()) is None

    @pytest.mark.parametrize(
        "mode_cls,drawdown,expected",
        [
            (MomentumScanner, -0.05, 20),  # 未觸發
            (MomentumScanner, -0.12, 10),  # 中度 → 砍半
            (MomentumScanner, -0.20, 6),  # 重度 → 1/3
            (GrowthScanner, -0.12, 10),  # growth 非防禦型
            (GrowthScanner, -0.20, 6),
            (ValueScanner, -0.12, 20),  # 防禦型豁免
            (ValueScanner, -0.20, 20),
            (DividendScanner, -0.20, 20),
        ],
    )
    def test_defensive_modes_exempt(self, mode_cls, drawdown, expected, monkeypatch):
        s = mode_cls(min_volume=1, top_n_results=20, use_ic_adjustment=False)
        s.scan_date = date(2026, 7, 1)
        monkeypatch.setattr(type(s), "_taiex_drawdown_20d", lambda self, df: drawdown)
        assert s._compute_drawdown_adjusted_top_n(pd.DataFrame()) == expected

    def test_stage_enabled_for_all_five_modes(self):
        """N2 閘門政策第一項：五模式一律開啟 4.2。"""
        for name, cls in _SCANNERS.items():
            assert cls._STAGES.drawdown_adjusted_top_n is True, f"{name} 未開啟 4.2"
