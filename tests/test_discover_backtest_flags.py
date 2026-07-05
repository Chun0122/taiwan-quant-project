"""P0 止血包 #4 — discover-backtest 預設翻轉（CLI 層）測試。

引擎 DiscoveryPerformance 預設不動（dashboard/decay 依賴 False），
翻轉只發生在 CLI：未明示 → (include_costs=True, entry_next_open=True)，
--naive 取得舊行為，個別 --no-* 可單獨關閉。
"""

from __future__ import annotations

import argparse

import pandas as pd
import pytest

from main import build_parser
from src.cli.discover_cmd import _resolve_backtest_flags
from src.discovery.performance import DiscoveryPerformance, print_performance_report


def _parse(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(["discover-backtest", "--mode", "momentum", *argv])


class TestArgparseLayer:
    def test_no_flags_sentinel_none(self):
        """未給 flag 時為 None sentinel（翻轉在 _resolve_backtest_flags 執行）。"""
        args = _parse([])
        assert args.include_costs is None
        assert args.entry_next_open is None
        assert args.naive is False

    def test_boolean_optional_action_negative_forms(self):
        args = _parse(["--no-include-costs", "--no-entry-next-open"])
        assert args.include_costs is False
        assert args.entry_next_open is False

    def test_positive_forms(self):
        args = _parse(["--include-costs", "--entry-next-open"])
        assert args.include_costs is True
        assert args.entry_next_open is True


class TestResolveBacktestFlags:
    def _ns(self, include_costs=None, entry_next_open=None, naive=False) -> argparse.Namespace:
        return argparse.Namespace(include_costs=include_costs, entry_next_open=entry_next_open, naive=naive)

    def test_default_flips_to_realistic(self):
        """核心：未明示 → 新預設（含成本 + T+1 開盤）。"""
        assert _resolve_backtest_flags(self._ns()) == (True, True)

    def test_naive_restores_old_defaults(self):
        assert _resolve_backtest_flags(self._ns(naive=True)) == (False, False)

    def test_naive_conflicts_with_explicit_flags(self):
        with pytest.raises(SystemExit):
            _resolve_backtest_flags(self._ns(include_costs=True, naive=True))
        with pytest.raises(SystemExit):
            _resolve_backtest_flags(self._ns(entry_next_open=False, naive=True))

    def test_individual_no_cost_keeps_entry_default(self):
        assert _resolve_backtest_flags(self._ns(include_costs=False)) == (False, True)

    def test_individual_no_entry_keeps_cost_default(self):
        assert _resolve_backtest_flags(self._ns(entry_next_open=False)) == (True, False)

    def test_explicit_true_passthrough(self):
        assert _resolve_backtest_flags(self._ns(include_costs=True, entry_next_open=True)) == (True, True)


class TestEngineDefaultsUnchanged:
    def test_engine_defaults_remain_false(self):
        """引擎層預設不動——dashboard data_loader 與 compute_strategy_decay 依賴。"""
        perf = DiscoveryPerformance(mode="momentum")
        assert perf.include_costs is False
        assert perf.entry_at_next_open is False


class TestReportHeaderLabels:
    def _fake_result(self) -> dict:
        detail = pd.DataFrame(
            {
                "scan_date": ["2026-06-01"],
                "stock_id": ["2330"],
                "holding_days": [5],
                "return_pct": [1.0],
            }
        )
        summary = pd.DataFrame(
            {
                "holding_days": [5],
                "evaluable": [0],
            }
        )
        return {"summary": summary, "by_scan": pd.DataFrame(), "detail": detail}

    def test_realistic_labels(self, capsys):
        print_performance_report(self._fake_result(), "momentum", include_costs=True, entry_at_next_open=True)
        out = capsys.readouterr().out
        assert "成本模型：含交易成本" in out
        assert "進場假設：T+1 開盤價" in out

    def test_naive_labels(self, capsys):
        print_performance_report(self._fake_result(), "momentum", include_costs=False, entry_at_next_open=False)
        out = capsys.readouterr().out
        assert "成本模型：naive（無交易成本）" in out
        assert "進場假設：T 日收盤價" in out
        assert "look-ahead" in out
