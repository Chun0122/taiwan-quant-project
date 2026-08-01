"""P0 #17 — E2b/E2c IC 自動調整凍結測試（2026-08-01）。

背景（`logs/audit_discover_20260731/REPORT.md` §6）：

  - **E2b**（`_apply_ic_weight_adjustment`）的歸一化把被打壓維度釋出的權重
    **全數塞給從未被量測的維度**。實測 2026-07-31 value：`fundamental 0.550→0.313`
    （IC −0.1393）而 `valuation 0.150→0.338`（**IC=N/A**，該維度不落庫
    `DiscoveryRecord`，永遠算不出 IC），權重 2.25×。淨效果＝IC 治理越積極，
    決策權越集中到唯一沒被檢驗的維度。
  - **E2c**（`compute_ic_aware_score_transform`）翻轉/中性化門檻 `|IC| ≤ 0.02`、
    `min_samples=50`，在 SE≈0.10–0.14 下等同隨機翻轉維度方向。

修法：兩者改為**只記錄不生效**（`quant.ic_governance` 兩個開關預設 false），
直到 B2 落地。凍結**只在套用點**（`_score_candidates`）生效——底層純函數與
`_apply_ic_weight_adjustment` 行為不變，故既有測試不受影響。

涵蓋：
  A. 預設即凍結（config 契約）
  B. 凍結時 composite 使用原始 regime 權重
  C. 凍結時分數不被翻轉/中性化，且 _ic_actions 留空（CLI 標記不得誤導）
  D. 開啟後行為回復（證明凍結是唯一的抑制來源）
  E. 凍結時仍保留觀測能力（_dimension_ic_df 照常填入）
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.discovery.scanner import MomentumScanner


@pytest.fixture()
def scanner():
    s = MomentumScanner(
        min_price=10,
        max_price=2000,
        min_volume=100_000,
        top_n_candidates=10,
        top_n_results=5,
        use_ic_adjustment=True,
    )
    s.regime = "bull"
    return s


@pytest.fixture()
def stub_scores(monkeypatch):
    """把四維度分數固定住，讓 composite 只反映權重與 E2c 轉換。"""

    def _mk(col, value):
        def _fn(self, stock_ids, *a, **kw):
            return pd.DataFrame({"stock_id": list(stock_ids), col: [value] * len(stock_ids)})

        return _fn

    monkeypatch.setattr(MomentumScanner, "_compute_technical_scores", _mk("technical_score", 0.8))
    monkeypatch.setattr(MomentumScanner, "_compute_chip_scores", _mk("chip_score", 0.8))
    monkeypatch.setattr(MomentumScanner, "_compute_fundamental_scores", _mk("fundamental_score", 0.2))
    monkeypatch.setattr(MomentumScanner, "_compute_news_scores", _mk("news_score", 0.2))
    monkeypatch.setattr(MomentumScanner, "_compute_extra_scores", lambda self, sids: [])
    monkeypatch.setattr(MomentumScanner, "_post_score", lambda self, c: c)
    monkeypatch.setattr(
        MomentumScanner,
        "_compute_entry_exit_cols",
        lambda self, sids, df, regime="bull": pd.DataFrame({"stock_id": list(sids)}),
    )


def _ic_df(**factor_ic: float) -> pd.DataFrame:
    """建構 compute_factor_ic 形狀（樣本數刻意 > min_samples=50 以便觸發轉換）。"""
    return pd.DataFrame(
        [
            {"factor": f, "ic": ic, "evaluable_count": 200, "direction": "inverse" if ic < 0 else "effective"}
            for f, ic in factor_ic.items()
        ]
    )


def _run(scanner, monkeypatch, *, ic_df: pd.DataFrame, adjusted: dict | None = None):
    """跑 _score_candidates，並注入 E2b 的「調整後權重」與維度 IC。"""

    def _fake_adjust(self, base_weights, scored_candidates=None):
        self._dimension_ic_df = ic_df
        return dict(adjusted) if adjusted is not None else dict(base_weights)

    monkeypatch.setattr(MomentumScanner, "_apply_ic_weight_adjustment", _fake_adjust)

    candidates = pd.DataFrame({"stock_id": ["1000"], "close": [100.0], "volume": [500_000]})
    empty = pd.DataFrame()
    return scanner._score_candidates(candidates, empty, empty, empty, empty)


def _set_flags(monkeypatch, *, weight: bool, transform: bool):
    from src.config import settings

    monkeypatch.setattr(settings.quant.ic_governance, "auto_weight_adjust", weight)
    monkeypatch.setattr(settings.quant.ic_governance, "auto_score_transform", transform)


# ====================================================================== #
# A. config 契約
# ====================================================================== #


class TestConfigDefaults:
    def test_both_switches_default_to_frozen(self):
        from src.config import ICGovernanceConfig

        cfg = ICGovernanceConfig()
        assert cfg.auto_weight_adjust is False
        assert cfg.auto_score_transform is False

    def test_live_settings_are_frozen(self):
        """quant_params.yaml 現況應為凍結——解除前請先確認 B2 已落地。"""
        from src.config import settings

        assert settings.quant.ic_governance.auto_weight_adjust is False
        assert settings.quant.ic_governance.auto_score_transform is False


# ====================================================================== #
# B. E2b 凍結：composite 使用原始 regime 權重
# ====================================================================== #


class TestWeightAdjustFrozen:
    def test_frozen_uses_base_weights(self, scanner, stub_scores, monkeypatch):
        """即使 E2b 算出極端權重，凍結時 composite 仍以原始 regime 權重計算。"""
        _set_flags(monkeypatch, weight=False, transform=False)
        # momentum bull 原始權重：technical 0.00 / chip 0.55 / fundamental 0.45 / news 0.00
        # → composite = 0.8*0.55 + 0.2*0.45 = 0.53
        skewed = {"technical": 0.0, "chip": 0.0, "fundamental": 1.0, "news": 0.0}
        out = _run(scanner, monkeypatch, ic_df=_ic_df(), adjusted=skewed)
        assert out.iloc[0]["composite_score"] == pytest.approx(0.8 * 0.55 + 0.2 * 0.45)

    def test_enabled_uses_adjusted_weights(self, scanner, stub_scores, monkeypatch):
        """開啟後同一組輸入應改用調整後權重（證明凍結是唯一抑制來源）。"""
        _set_flags(monkeypatch, weight=True, transform=False)
        skewed = {"technical": 0.0, "chip": 0.0, "fundamental": 1.0, "news": 0.0}
        out = _run(scanner, monkeypatch, ic_df=_ic_df(), adjusted=skewed)
        assert out.iloc[0]["composite_score"] == pytest.approx(0.2)


# ====================================================================== #
# C. E2c 凍結：分數不轉換、_ic_actions 留空
# ====================================================================== #


class TestScoreTransformFrozen:
    def test_frozen_does_not_flip_scores(self, scanner, stub_scores, monkeypatch):
        """IC 反向時原本會 1-score 翻轉；凍結後分數須原封不動。"""
        _set_flags(monkeypatch, weight=False, transform=False)
        out = _run(scanner, monkeypatch, ic_df=_ic_df(chip_score=-0.30))
        assert out.iloc[0]["chip_score"] == pytest.approx(0.8), "凍結期間不得翻轉"

    def test_frozen_does_not_neutralize_scores(self, scanner, stub_scores, monkeypatch):
        """|IC| <= 0.02 原本會歸 0.5（雜訊中性化）；凍結後須保留原值。"""
        _set_flags(monkeypatch, weight=False, transform=False)
        out = _run(scanner, monkeypatch, ic_df=_ic_df(chip_score=0.001))
        assert out.iloc[0]["chip_score"] == pytest.approx(0.8)

    def test_frozen_leaves_ic_actions_empty(self, scanner, stub_scores, monkeypatch):
        """CLI 的 (N)/(F)/(D) 標記代表「已套用」，凍結期間不得標記。"""
        _set_flags(monkeypatch, weight=False, transform=False)
        _run(scanner, monkeypatch, ic_df=_ic_df(chip_score=-0.30))
        assert scanner._ic_actions == {}

    def test_enabled_flips_and_marks(self, scanner, stub_scores, monkeypatch):
        _set_flags(monkeypatch, weight=False, transform=True)
        out = _run(scanner, monkeypatch, ic_df=_ic_df(chip_score=-0.30))
        assert out.iloc[0]["chip_score"] == pytest.approx(0.2), "1 - 0.8"
        assert scanner._ic_actions.get("chip_score") == "flipped"

    def test_enabled_neutralizes_noise_band(self, scanner, stub_scores, monkeypatch):
        _set_flags(monkeypatch, weight=False, transform=True)
        out = _run(scanner, monkeypatch, ic_df=_ic_df(chip_score=0.001))
        assert out.iloc[0]["chip_score"] == pytest.approx(0.5)
        assert scanner._ic_actions.get("chip_score") == "neutralized"


# ====================================================================== #
# D/E. 凍結不得損失觀測能力
# ====================================================================== #


class TestObservabilityPreserved:
    def test_dimension_ic_still_computed_when_frozen(self, scanner, stub_scores, monkeypatch):
        """凍結是「不生效」不是「不計算」——IC 仍須算出供告警與後續研究。"""
        _set_flags(monkeypatch, weight=False, transform=False)
        _run(scanner, monkeypatch, ic_df=_ic_df(chip_score=-0.30))
        assert scanner._dimension_ic_df is not None
        assert not scanner._dimension_ic_df.empty

    def test_frozen_logs_would_be_actions(self, scanner, stub_scores, monkeypatch, caplog):
        _set_flags(monkeypatch, weight=False, transform=False)
        with caplog.at_level("INFO", logger="src.discovery.scanner._base"):
            _run(
                scanner,
                monkeypatch,
                ic_df=_ic_df(chip_score=-0.30),
                adjusted={"technical": 0.0, "chip": 0.0, "fundamental": 1.0, "news": 0.0},
            )
        text = caplog.text
        assert "凍結中，未生效" in text
        assert "would-be flipped" in text, "E2c 應記錄原本會做的轉換"
        assert "E2b 權重調整【凍結中，未生效】" in text, "E2b 應記錄原本會做的調權"

    def test_use_ic_adjustment_false_skips_entirely(self, stub_scores, monkeypatch):
        """use_ic_adjustment=False（測試用總開關）仍應完全跳過，不受凍結旗標影響。"""
        _set_flags(monkeypatch, weight=True, transform=True)
        s = MomentumScanner(min_volume=100_000, use_ic_adjustment=False)
        s.regime = "bull"
        out = _run(s, monkeypatch, ic_df=_ic_df(chip_score=-0.30))
        assert out.iloc[0]["chip_score"] == pytest.approx(0.8)
        assert s._ic_actions == {}
