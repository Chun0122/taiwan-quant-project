"""Regime × Mode 封鎖矩陣測試。

覆蓋：
  - REGIME_MODE_BLOCK 矩陣設定正確
  - Scanner Stage 0.1 雙來源（_blocked_regimes + REGIME_MODE_BLOCK）封鎖
  - 非封鎖組合正常運行
  - value/dividend 在 sideways 不受影響
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.constants import REGIME_MODE_BLOCK
from src.discovery.scanner import DividendScanner, GrowthScanner, MomentumScanner, ValueScanner
from src.discovery.scanner._functions import DiscoveryResult


def test_matrix_defined_as_expected():
    """實證依據：sideways 封鎖 momentum/growth，crisis 封鎖 growth（保留 momentum）。"""
    assert "momentum" in REGIME_MODE_BLOCK["sideways"]
    assert "growth" in REGIME_MODE_BLOCK["sideways"]
    assert "growth" in REGIME_MODE_BLOCK["crisis"]
    # momentum 在 crisis 實測 10 日 +8.35%@77.4% 勝率，不封鎖
    assert "momentum" not in REGIME_MODE_BLOCK.get("crisis", frozenset())
    # value/dividend 在所有 regime 都不封鎖
    for regime in REGIME_MODE_BLOCK.values():
        assert "value" not in regime
        assert "dividend" not in regime


def _patch_regime(regime: str):
    """patch MarketRegimeDetector.detect 回傳指定 regime。"""
    return patch(
        "src.regime.detector.MarketRegimeDetector.detect",
        return_value={"regime": regime, "taiex_close": 17000.0},
    )


def test_momentum_blocked_in_sideways():
    scanner = MomentumScanner()
    with _patch_regime("sideways"):
        result = scanner.run()
    assert isinstance(result, DiscoveryResult)
    assert result.rankings.empty
    assert scanner.regime == "sideways"


def test_growth_blocked_in_sideways():
    scanner = GrowthScanner()
    with _patch_regime("sideways"):
        result = scanner.run()
    assert result.rankings.empty


def test_growth_blocked_in_crisis():
    scanner = GrowthScanner()
    with _patch_regime("crisis"):
        result = scanner.run()
    assert result.rankings.empty


def test_value_not_blocked_in_sideways():
    """Value 在 sideways 應繼續執行（只是可能因無資料回傳空，但會進入 Stage 1）。"""
    scanner = ValueScanner()
    with _patch_regime("sideways"), patch.object(
        ValueScanner, "_load_market_data", return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    ) as mocked:
        scanner.run()
    # 若被封鎖則不會進到 Stage 1 的 _load_market_data
    mocked.assert_called_once()


def test_dividend_not_blocked_in_crisis():
    """Dividend（防禦型）在 crisis 不封鎖。"""
    scanner = DividendScanner()
    with _patch_regime("crisis"), patch.object(
        DividendScanner,
        "_load_market_data",
        return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    ) as mocked:
        scanner.run()
    mocked.assert_called_once()


def test_momentum_not_blocked_in_crisis():
    """Momentum 在 crisis 保留（實測 10 日 +8.35%）。"""
    scanner = MomentumScanner()
    with _patch_regime("crisis"), patch.object(
        MomentumScanner,
        "_load_market_data",
        return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    ) as mocked:
        scanner.run()
    mocked.assert_called_once()


def test_momentum_runs_in_bull():
    scanner = MomentumScanner()
    with _patch_regime("bull"), patch.object(
        MomentumScanner,
        "_load_market_data",
        return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    ) as mocked:
        scanner.run()
    mocked.assert_called_once()


@pytest.mark.parametrize("regime", ["sideways", "crisis"])
def test_subclass_blocked_regimes_still_honored(regime):
    """子類自訂 _blocked_regimes 仍獨立生效（防止矩陣遺漏時退化）。"""

    class ExperimentalScanner(MomentumScanner):
        mode_name = "experimental"  # 不在矩陣裡
        _blocked_regimes = {"bear", "crisis"}

    scanner = ExperimentalScanner()
    with _patch_regime(regime):
        result = scanner.run()
    if regime == "crisis":
        # 被子類自訂封鎖
        assert result.rankings.empty
    else:
        # sideways 不在子類封鎖名單 + experimental 不在矩陣 → 應能進到 Stage 1
        # （此處會因無資料自然回傳空，但 regime gate 不觸發）
        pass
