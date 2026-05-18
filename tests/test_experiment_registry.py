"""P2 任務 10 — Experiment Registry 測試。

涵蓋：
  P10-A sanitize_settings 不洩 token + 含研究區塊
  P10-B compute_settings_hash idempotent + 鍵順序穩定
  P10-C generate_experiment_id 格式 + 不同次呼叫不同
  P10-D diff_metrics 邏輯
  P10-E ExperimentLog schema + unique constraint
  P10-F CLI record / list / show / compare exit code
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import select

from src.cli.experiment_cmd import (
    compute_settings_hash,
    diff_metrics,
    generate_experiment_id,
    sanitize_settings,
)
from src.config import Settings
from src.data.schema import ExperimentLog

# ====================================================================== #
# P10-A: sanitize_settings
# ====================================================================== #


class TestSanitizeSettings:
    def test_excludes_api_tokens_and_webhook(self):
        s = Settings()
        s.finmind.api_token = "SHOULD_NOT_LEAK"
        s.anthropic.api_key = "SHOULD_NOT_LEAK"
        s.discord.webhook_url = "https://SHOULD_NOT_LEAK"
        sanitized = sanitize_settings(s)
        flat = json.dumps(sanitized, ensure_ascii=False)
        assert "SHOULD_NOT_LEAK" not in flat
        assert "api_token" not in flat
        assert "api_key" not in flat
        assert "webhook_url" not in flat

    def test_includes_quant_section(self):
        s = Settings()
        sanitized = sanitize_settings(s)
        assert "quant" in sanitized
        assert "trading_cost" in sanitized["quant"]
        assert "atr_multiplier" in sanitized["quant"]

    def test_includes_fetcher_watchlist(self):
        s = Settings()
        s.fetcher.watchlist = ["2330", "2317"]
        sanitized = sanitize_settings(s)
        assert sanitized["fetcher"]["watchlist"] == ["2330", "2317"]


# ====================================================================== #
# P10-B: compute_settings_hash
# ====================================================================== #


class TestComputeSettingsHash:
    def test_same_input_same_hash(self):
        d = {"quant": {"x": 1, "y": 2}}
        h1 = compute_settings_hash(d)
        h2 = compute_settings_hash(d)
        assert h1 == h2

    def test_key_order_insensitive(self):
        a = {"a": 1, "b": 2}
        b = {"b": 2, "a": 1}
        assert compute_settings_hash(a) == compute_settings_hash(b)

    def test_value_change_changes_hash(self):
        a = {"quant": {"commission": 0.001425}}
        b = {"quant": {"commission": 0.001500}}
        assert compute_settings_hash(a) != compute_settings_hash(b)

    def test_hash_is_16_chars(self):
        h = compute_settings_hash({"x": 1})
        assert len(h) == 16
        # 全為 hex chars
        assert all(c in "0123456789abcdef" for c in h)


# ====================================================================== #
# P10-C: generate_experiment_id
# ====================================================================== #


class TestGenerateExperimentId:
    def test_format(self):
        eid = generate_experiment_id(today=date(2026, 5, 18))
        assert eid.startswith("exp_20260518_")
        suffix = eid.removeprefix("exp_20260518_")
        assert len(suffix) == 6
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_two_calls_yield_distinct(self):
        e1 = generate_experiment_id()
        e2 = generate_experiment_id()
        assert e1 != e2  # 機率 ~ 1/16M，足以視為穩定


# ====================================================================== #
# P10-D: diff_metrics
# ====================================================================== #


class TestDiffMetrics:
    def test_returns_delta_for_numeric_fields(self):
        a = {"alpha_pf": {"sharpe_ratio": 1.0, "max_drawdown_pct": 5.0}}
        b = {"alpha_pf": {"sharpe_ratio": 1.2, "max_drawdown_pct": 7.5}}
        diffs = diff_metrics(a, b)
        sharpe = next(d for d in diffs if d["metric"] == "sharpe_ratio")
        assert sharpe["delta"] == pytest.approx(0.2)
        mdd = next(d for d in diffs if d["metric"] == "max_drawdown_pct")
        assert mdd["delta"] == pytest.approx(2.5)

    def test_skips_meta_fields(self):
        a = {"x": {"portfolio_name": "x", "as_of": "2026-01-01", "snapshot_count": 10}}
        b = {"x": {"portfolio_name": "x", "as_of": "2026-05-18", "snapshot_count": 33}}
        diffs = diff_metrics(a, b)
        names = {d["metric"] for d in diffs}
        assert "portfolio_name" not in names
        assert "as_of" not in names
        assert "snapshot_count" not in names

    def test_handles_missing_portfolio(self):
        a = {"X": {"sharpe_ratio": 1.0}}
        b = {"Y": {"sharpe_ratio": 0.8}}
        diffs = diff_metrics(a, b)
        portfolios = {d["portfolio"] for d in diffs}
        assert portfolios == {"X", "Y"}

    def test_none_values_no_delta(self):
        a = {"X": {"sharpe_ratio": None}}
        b = {"X": {"sharpe_ratio": 0.5}}
        diffs = diff_metrics(a, b)
        sharpe = next(d for d in diffs if d["metric"] == "sharpe_ratio")
        assert sharpe["delta"] is None


# ====================================================================== #
# P10-E: ExperimentLog schema
# ====================================================================== #


class TestExperimentLogSchema:
    def test_can_persist_minimal_row(self, db_session):
        row = ExperimentLog(
            experiment_id="exp_test_aaa",
            git_commit="abc1234",
            settings_hash="1234567890abcdef",
            settings_snapshot_json='{"quant": {}}',
            metrics_json="{}",
            description="smoke",
        )
        db_session.add(row)
        db_session.commit()
        loaded = db_session.execute(select(ExperimentLog)).scalar_one()
        assert loaded.experiment_id == "exp_test_aaa"

    def test_unique_constraint_on_experiment_id(self, db_session):
        db_session.add(
            ExperimentLog(
                experiment_id="exp_dup",
                git_commit=None,
                settings_hash="h1",
                settings_snapshot_json="{}",
                metrics_json="{}",
            )
        )
        db_session.commit()
        db_session.add(
            ExperimentLog(
                experiment_id="exp_dup",
                git_commit=None,
                settings_hash="h2",
                settings_snapshot_json="{}",
                metrics_json="{}",
            )
        )
        with pytest.raises(Exception):
            db_session.commit()
        db_session.rollback()


# ====================================================================== #
# P10-F: CLI handlers
# ====================================================================== #


@pytest.fixture()
def patch_session(db_session, monkeypatch):
    from src.cli import experiment_cmd as exp_mod

    class _Ctx:
        def __init__(self, s):
            self._s = s

        def __enter__(self):
            return self._s

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(exp_mod, "get_session", lambda: _Ctx(db_session))
    # 防止 init_db 跑真實 DB
    import src.cli.helpers as helpers

    monkeypatch.setattr(helpers, "init_db", lambda: None)
    return db_session


class TestCliRecord:
    def test_record_inserts_row_and_returns_zero(self, patch_session, monkeypatch, capsys):
        from src.cli import experiment_cmd as exp_mod

        # Patch collect_experiment_payload 以避免 collect_current_metrics 真實 DB 查詢
        def _fake_payload(description, *, cfg=None):
            return {
                "experiment_id": "exp_test_fixed_001",
                "git_commit": "deadbeef",
                "settings_hash": "abcdef1234567890",
                "settings_snapshot_json": '{"quant": {}}',
                "metrics_json": '{"alpha_pf": {"sharpe_ratio": 1.0}}',
                "description": description,
            }

        monkeypatch.setattr(exp_mod, "collect_experiment_payload", _fake_payload)

        exit_code = exp_mod.cmd_experiment(argparse.Namespace(exp_action="record", description="smoke test"))
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "exp_test_fixed_001" in out
        assert "deadbeef" in out

        row = patch_session.execute(select(ExperimentLog)).scalar_one()
        assert row.description == "smoke test"
        assert row.git_commit == "deadbeef"


class TestCliList:
    def test_list_empty_returns_zero(self, patch_session, capsys):
        from src.cli import experiment_cmd as exp_mod

        exit_code = exp_mod.cmd_experiment(argparse.Namespace(exp_action="list", limit=10))
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "尚無 experiment 紀錄" in out

    def test_list_shows_recorded_in_desc_order(self, patch_session, capsys):
        from src.cli import experiment_cmd as exp_mod

        # 注入 2 筆，第二筆較新
        now = datetime.utcnow()
        patch_session.add(
            ExperimentLog(
                experiment_id="exp_old",
                settings_hash="h1",
                settings_snapshot_json="{}",
                metrics_json="{}",
                recorded_at=now - timedelta(hours=2),
            )
        )
        patch_session.add(
            ExperimentLog(
                experiment_id="exp_new",
                settings_hash="h2",
                settings_snapshot_json="{}",
                metrics_json="{}",
                recorded_at=now,
            )
        )
        patch_session.commit()

        exit_code = exp_mod.cmd_experiment(argparse.Namespace(exp_action="list", limit=10))
        assert exit_code == 0
        out = capsys.readouterr().out
        # exp_new 應出現在 exp_old 之前
        assert out.index("exp_new") < out.index("exp_old")


class TestCliShow:
    def test_show_unknown_returns_two(self, patch_session, capsys):
        from src.cli import experiment_cmd as exp_mod

        exit_code = exp_mod.cmd_experiment(argparse.Namespace(exp_action="show", experiment_id="exp_nope"))
        assert exit_code == 2
        out = capsys.readouterr().out
        assert "找不到 experiment" in out

    def test_show_existing_prints_details(self, patch_session, capsys):
        from src.cli import experiment_cmd as exp_mod

        patch_session.add(
            ExperimentLog(
                experiment_id="exp_show_test",
                git_commit="abc1234",
                settings_hash="hash_show",
                settings_snapshot_json='{"quant": {"commission": 0.001425}}',
                metrics_json='{"pf_a": {"sharpe_ratio": 1.2, "max_drawdown_pct": 5.0}}',
                description="show test",
            )
        )
        patch_session.commit()

        exit_code = exp_mod.cmd_experiment(argparse.Namespace(exp_action="show", experiment_id="exp_show_test"))
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "abc1234" in out
        assert "hash_show" in out
        assert "show test" in out
        assert "0.001425" in out
        assert "pf_a" in out


class TestCliCompare:
    def test_compare_missing_a_returns_two(self, patch_session, capsys):
        from src.cli import experiment_cmd as exp_mod

        # 注入 B 但 A 缺
        patch_session.add(
            ExperimentLog(
                experiment_id="exp_B",
                settings_hash="hB",
                settings_snapshot_json="{}",
                metrics_json="{}",
            )
        )
        patch_session.commit()

        exit_code = exp_mod.cmd_experiment(argparse.Namespace(exp_action="compare", id_a="exp_nope", id_b="exp_B"))
        assert exit_code == 2
        out = capsys.readouterr().out
        assert "找不到 experiment A" in out

    def test_compare_shows_metric_delta(self, patch_session, capsys):
        from src.cli import experiment_cmd as exp_mod

        patch_session.add(
            ExperimentLog(
                experiment_id="exp_A",
                git_commit="aaa1111",
                settings_hash="hA",
                settings_snapshot_json="{}",
                metrics_json='{"pf_a": {"sharpe_ratio": 1.0, "max_drawdown_pct": 5.0}}',
            )
        )
        patch_session.add(
            ExperimentLog(
                experiment_id="exp_B",
                git_commit="bbb2222",
                settings_hash="hB",
                settings_snapshot_json="{}",
                metrics_json='{"pf_a": {"sharpe_ratio": 1.3, "max_drawdown_pct": 4.5}}',
            )
        )
        patch_session.commit()

        exit_code = exp_mod.cmd_experiment(argparse.Namespace(exp_action="compare", id_a="exp_A", id_b="exp_B"))
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "hA" in out and "hB" in out
        assert "CHANGED" in out  # settings_hash 不同
        # delta sharpe = +0.3, mdd = -0.5
        assert "+0.3000" in out
        assert "-0.5000" in out

    def test_compare_same_settings_hash_message(self, patch_session, capsys):
        from src.cli import experiment_cmd as exp_mod

        patch_session.add(
            ExperimentLog(
                experiment_id="exp_X",
                settings_hash="same_hash",
                settings_snapshot_json="{}",
                metrics_json="{}",
            )
        )
        patch_session.add(
            ExperimentLog(
                experiment_id="exp_Y",
                settings_hash="same_hash",
                settings_snapshot_json="{}",
                metrics_json="{}",
            )
        )
        patch_session.commit()

        exp_mod.cmd_experiment(argparse.Namespace(exp_action="compare", id_a="exp_X", id_b="exp_Y"))
        out = capsys.readouterr().out
        assert "settings_hash 相同" in out


class TestCliMissingAction:
    def test_no_subcommand_returns_two(self, patch_session, capsys):
        from src.cli import experiment_cmd as exp_mod

        exit_code = exp_mod.cmd_experiment(argparse.Namespace(exp_action=None))
        assert exit_code == 2
