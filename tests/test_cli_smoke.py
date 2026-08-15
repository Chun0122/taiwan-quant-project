"""CLI smoke tests（止血包 P0 #2 制度化）。

背景：`Announcement.title` 事故——glue code（CLI handler → pipeline 函數）是測試盲區，
`sync-concepts --from-mops` 上線後 100% crash 卻無測試攔截。

本檔提供兩層防護：
1. Parser 層：`build_parser()` 對全部子命令 render help，攔截 argparse 定義錯誤。
2. Handler 層：代表性子命令以 Namespace 直接呼叫，攔截 handler → 底層函數的欄位/簽名錯配。
"""

import argparse
from datetime import date

import pytest

# 注意：所有 src 模組必須 top-level import（collection 時、未 patch 狀態下載入）。
# 若在測試函數內首次 import，模組 top-level 的 `from src.data.database import get_session`
# 會捕捉到 monkeypatch 過的 stale session，污染後續其他測試檔（conftest 已知陷阱）。
import src.data.migrate as migrate_mod  # noqa: F401 - 確保 collection 時載入
import src.data.pipeline as pipeline_mod
from main import build_parser
from src.cli.watchlist_cmd import cmd_sync_concepts
from src.data.pipeline import sync_concept_tags_from_mops
from src.data.schema import Announcement, ConceptMembership

# ====================================================================== #
# Parser 層 smoke
# ====================================================================== #

# 關鍵子命令清單：未來重構 main.py 時防止誤刪（不需窮舉全部 49 個）
_KEY_COMMANDS = {
    "sync",
    "compute",
    "backtest",
    "discover",
    "discover-backtest",
    "morning-routine",
    "rotation",
    "rotation-audit",
    "sync-concepts",
    "watchlist",
    "watch",
    "export-dashboard",
    "validate-baseline",
    "experiment",
}


def _get_subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """取出主 parser 的全部子命令 parser。"""
    for action in parser._subparsers._group_actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("找不到 subparsers")


class TestParserSmoke:
    def test_build_parser_constructs(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_all_subcommands_help_renders(self):
        """全部子命令 format_help() 不炸（攔截重複 dest / 無效 action 等定義錯誤）。"""
        subparsers = _get_subparsers(build_parser())
        assert len(subparsers) >= 39
        for name, sp in subparsers.items():
            help_text = sp.format_help()
            assert help_text, f"子命令 {name} 的 help 為空"

    def test_key_commands_present(self):
        subparsers = _get_subparsers(build_parser())
        missing = _KEY_COMMANDS - set(subparsers)
        assert not missing, f"關鍵子命令遺失：{missing}"

    def test_no_command_parses(self):
        """無參數 parse 不炸（dispatch 端負責 print_help）。"""
        args = build_parser().parse_args([])
        assert args.command is None

    def test_backtest_a4_flags(self):
        """A4 交易現實化旗標：預設開、--no-* 可關。"""
        parser = build_parser()
        args = parser.parse_args(["backtest", "--stock", "2330", "--strategy", "sma_cross"])
        assert args.min_commission is True
        assert args.participation_impact is True
        args = parser.parse_args(
            [
                "backtest",
                "--stock",
                "2330",
                "--strategy",
                "sma_cross",
                "--no-min-commission",
                "--no-participation-impact",
            ]
        )
        assert args.min_commission is False
        assert args.participation_impact is False


# ====================================================================== #
# Handler 層 smoke：sync-concepts --from-mops（title→subject 回歸測試）
# ====================================================================== #


class _Ctx:
    """把裸 session 包成 context manager（模擬 get_session()）。"""

    def __init__(self, s):
        self._s = s

    def __enter__(self):
        return self._s

    def __exit__(self, *a):
        pass


@pytest.fixture()
def _patched_pipeline_session(db_session, monkeypatch):
    """把 pipeline 模組的 get_session/init_db 指向測試 session。"""

    monkeypatch.setattr(pipeline_mod, "get_session", lambda: _Ctx(db_session))
    monkeypatch.setattr(pipeline_mod, "init_db", lambda: None)
    return db_session


class TestSyncConceptTagsFromMops:
    def test_subject_keyword_creates_membership(self, _patched_pipeline_session):
        """回歸測試：修復前查詢 Announcement.title 直接 AttributeError。"""

        session = _patched_pipeline_session
        session.add(
            Announcement(
                stock_id="3017",
                date=date.today(),
                seq="1",
                subject="公告本公司取得 CoWoS 先進封裝設備",
            )
        )
        session.commit()

        added = sync_concept_tags_from_mops(days=90)

        assert added == 1
        member = session.query(ConceptMembership).one()
        assert member.concept_name == "CoWoS封裝"
        assert member.stock_id == "3017"
        assert member.source == "mops"

    def test_empty_subject_skipped(self, _patched_pipeline_session):

        session = _patched_pipeline_session
        session.add(Announcement(stock_id="2330", date=date.today(), seq="1", subject=""))
        session.commit()

        assert sync_concept_tags_from_mops(days=90) == 0
        assert session.query(ConceptMembership).count() == 0

    def test_existing_membership_not_duplicated(self, _patched_pipeline_session):

        session = _patched_pipeline_session
        session.add(
            ConceptMembership(
                concept_name="CoWoS封裝",
                stock_id="3017",
                source="yaml",
                added_date=date.today(),
            )
        )
        session.add(
            Announcement(
                stock_id="3017",
                date=date.today(),
                seq="1",
                subject="CoWoS 擴產進度說明",
            )
        )
        session.commit()

        assert sync_concept_tags_from_mops(days=90) == 0
        assert session.query(ConceptMembership).count() == 1


class TestCmdSyncConceptsSmoke:
    def test_from_mops_handler_smoke(self, _patched_pipeline_session, monkeypatch, capsys):
        """CLI handler → pipeline 全鏈路 smoke（glue code 覆蓋）。"""

        monkeypatch.setattr(migrate_mod, "run_migrations", lambda: None)

        cmd_sync_concepts(argparse.Namespace(from_mops=True, days=90))
        out = capsys.readouterr().out
        assert "MOPS 關鍵字標記完成" in out


# ====================================================================== #
# §6.5 #20：backfill-history --valuation-only 的 parser 與 handler 接線
# ====================================================================== #


class TestValuationBackfillCli:
    def test_valuation_flags_parse(self):
        parser = build_parser()
        args = parser.parse_args(
            ["backfill-history", "--valuation-only", "--start", "2024-01-01", "--valuation-markets", "twse"]
        )
        assert args.valuation_only is True
        assert args.valuation_markets == "twse"

    def test_valuation_only_defaults(self):
        args = build_parser().parse_args(["backfill-history", "--start", "2024-01-01"])
        assert args.valuation_only is False
        assert args.valuation_markets == "twse,tpex"

    def test_handler_routes_to_valuation_backfill(self, monkeypatch, capsys):
        """--valuation-only 必須走估值路徑，且**不得**觸發下市清單同步或價量回補。

        （下市同步會打 FinMind，價量回補會打 TWSE 數小時——誤觸代價很高。）
        """
        import src.cli.misc_cmd as misc

        captured = {}

        def _fake_valuation(start, end, *, markets, dry_run=False, **kw):
            captured["args"] = (start, end, markets, dry_run)
            return {
                "twse_days": 3,
                "twse_rows": 300,
                "tpex_stocks": 2,
                "tpex_rows": 20,
                "skipped_days": 1,
                "skipped_stocks": 5,
            }

        def _boom(*a, **kw):
            raise AssertionError("--valuation-only 不應觸發此路徑")

        monkeypatch.setattr(pipeline_mod, "backfill_valuation_history", _fake_valuation)
        monkeypatch.setattr(pipeline_mod, "backfill_market_history", _boom)
        monkeypatch.setattr(pipeline_mod, "sync_delisting_info", _boom)
        monkeypatch.setattr(pipeline_mod, "backfill_daily_features", _boom)

        args = build_parser().parse_args(
            ["backfill-history", "--valuation-only", "--start", "2024-01-01", "--end", "2024-01-31"]
        )
        misc.cmd_backfill_history(args)

        assert captured["args"] == (date(2024, 1, 1), date(2024, 1, 31), ("twse", "tpex"), False)
        out = capsys.readouterr().out
        assert "估值回補完成" in out
        assert "300" in out and "20" in out


# ====================================================================== #
# §6.6 #24：backfill-history --revenue-only 的 parser 與 handler 接線
# ====================================================================== #


class TestRevenueBackfillCli:
    def test_revenue_flag_parses(self):
        args = build_parser().parse_args(["backfill-history", "--revenue-only", "--start", "2020-01-01"])
        assert args.revenue_only is True

    def test_revenue_only_defaults_false(self):
        assert build_parser().parse_args(["backfill-history", "--start", "2020-01-01"]).revenue_only is False

    def test_handler_routes_to_revenue_backfill(self, monkeypatch, capsys):
        """--revenue-only 必須走營收路徑，且不得觸發下市同步／價量／估值回補。"""
        import src.cli.misc_cmd as misc

        captured = {}

        def _fake_revenue(start, end, *, dry_run=False, **kw):
            captured["args"] = (start, end, dry_run)
            return {"months": 4, "rows": 6800, "skipped_months": 2, "normalized": 12}

        def _boom(*a, **kw):
            raise AssertionError("--revenue-only 不應觸發此路徑")

        monkeypatch.setattr(pipeline_mod, "backfill_revenue_history", _fake_revenue)
        monkeypatch.setattr(pipeline_mod, "backfill_market_history", _boom)
        monkeypatch.setattr(pipeline_mod, "backfill_valuation_history", _boom)
        monkeypatch.setattr(pipeline_mod, "sync_delisting_info", _boom)
        monkeypatch.setattr(pipeline_mod, "backfill_daily_features", _boom)

        args = build_parser().parse_args(
            ["backfill-history", "--revenue-only", "--start", "2020-01-01", "--end", "2020-04-30"]
        )
        misc.cmd_backfill_history(args)

        assert captured["args"] == (date(2020, 1, 1), date(2020, 4, 30), False)
        out = capsys.readouterr().out
        assert "月營收回補完成" in out
        assert "6800" in out
        assert "12" in out  # 正規化筆數有回報
