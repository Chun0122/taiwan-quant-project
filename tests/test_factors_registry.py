"""P1 任務 6 — Factor Library registry 測試。

涵蓋：
  P6-A FactorSpec 驗證（合法值 / 拒絕非法 category, type, sign）
  P6-B FACTOR_REGISTRY 結構性檢查（無重複 name，必要欄位齊全）
  P6-C list_factors 過濾邏輯
  P6-D resolve() introspection — 所有註冊 spec 的 source_module/function 必須真存在
  P6-E get_factor 查詢
  P6-F CLI handler exit code 與輸出
"""

from __future__ import annotations

import argparse

import pytest

from src.cli.factor_cmd import cmd_factor_list
from src.factors import FACTOR_REGISTRY, FactorSpec, get_factor, list_factors

# ====================================================================== #
# P6-A: FactorSpec __post_init__ 驗證
# ====================================================================== #


class TestFactorSpecValidation:
    def test_valid_spec_constructs_ok(self):
        s = FactorSpec(
            name="test_score",
            category="technical",
            factor_type="dimension",
            description="test",
            source_module="src.constants",
            source_function=None,
        )
        assert s.name == "test_score"
        assert s.expected_sign == "+"  # default

    def test_rejects_invalid_category(self):
        with pytest.raises(ValueError, match="category"):
            FactorSpec(
                name="x",
                category="bogus",
                factor_type="dimension",
                description="x",
                source_module="m",
                source_function=None,
            )

    def test_rejects_invalid_factor_type(self):
        with pytest.raises(ValueError, match="factor_type"):
            FactorSpec(
                name="x",
                category="technical",
                factor_type="garbage",
                description="x",
                source_module="m",
                source_function=None,
            )

    def test_rejects_invalid_sign(self):
        with pytest.raises(ValueError, match="expected_sign"):
            FactorSpec(
                name="x",
                category="technical",
                factor_type="dimension",
                description="x",
                source_module="m",
                source_function=None,
                expected_sign="+++",
            )


# ====================================================================== #
# P6-B: FACTOR_REGISTRY 結構性檢查
# ====================================================================== #


class TestFactorRegistryStructure:
    def test_registry_not_empty(self):
        assert len(FACTOR_REGISTRY) > 0

    def test_no_duplicate_names(self):
        """key 與 spec.name 必須一致（防註冊時抓錯）。"""
        for key, spec in FACTOR_REGISTRY.items():
            assert key == spec.name, f"key {key!r} != spec.name {spec.name!r}"

    def test_four_main_dimensions_registered(self):
        """技術/籌碼/基本面/消息四維 composite 必須註冊。"""
        for dim in ("technical_score", "chip_score", "fundamental_score", "news_score"):
            assert dim in FACTOR_REGISTRY, f"missing dimension: {dim}"
            assert FACTOR_REGISTRY[dim].factor_type == "dimension"

    def test_all_specs_have_descriptions(self):
        for name, spec in FACTOR_REGISTRY.items():
            assert spec.description, f"{name} 缺 description"
            assert len(spec.description) > 5, f"{name} description 太短"

    def test_used_in_modes_only_known(self):
        valid_modes = {"momentum", "swing", "value", "dividend", "growth"}
        for name, spec in FACTOR_REGISTRY.items():
            for m in spec.used_in_modes:
                assert m in valid_modes, f"{name} 含未知 mode: {m}"


# ====================================================================== #
# P6-C: list_factors 過濾邏輯
# ====================================================================== #


class TestListFactors:
    def test_no_filter_returns_all(self):
        all_factors = list_factors()
        assert len(all_factors) == len(FACTOR_REGISTRY)

    def test_filter_by_category(self):
        chip_factors = list_factors(category="chip")
        assert len(chip_factors) > 0
        for f in chip_factors:
            assert f.category == "chip"

    def test_filter_by_type(self):
        dims = list_factors(factor_type="dimension")
        assert len(dims) == 4  # technical / chip / fundamental / news
        for f in dims:
            assert f.factor_type == "dimension"

    def test_filter_by_mode(self):
        mom = list_factors(used_in_mode="momentum")
        assert len(mom) > 0
        for f in mom:
            assert "momentum" in f.used_in_modes

    def test_combined_filter(self):
        """category=chip + mode=momentum → 只回符合兩個條件的。"""
        result = list_factors(category="chip", used_in_mode="momentum")
        assert len(result) > 0
        for f in result:
            assert f.category == "chip" and "momentum" in f.used_in_modes

    def test_sort_order_deterministic(self):
        a = list_factors()
        b = list_factors()
        assert [f.name for f in a] == [f.name for f in b]


# ====================================================================== #
# P6-D: resolve() introspection — 註冊的 source 都必須 import 成功
# ====================================================================== #


class TestRegistryResolvability:
    def test_all_specs_resolve(self):
        """每個 FactorSpec.resolve() 必須回 non-None（防 phase 2 重構漂移）。

        若日後函式被移除或改名，這個測試會立刻在 CI 抓到。
        """
        unresolved = []
        for name, spec in FACTOR_REGISTRY.items():
            obj = spec.resolve()
            if obj is None:
                unresolved.append((name, spec.source_module, spec.source_function))
        assert not unresolved, f"以下因子無法 resolve: {unresolved}"

    def test_function_specs_actually_callable(self):
        """source_function 非 None 的 spec，resolve 結果應該是 callable（函式）。"""
        for name, spec in FACTOR_REGISTRY.items():
            if spec.source_function is None:
                continue
            obj = spec.resolve()
            assert callable(obj), f"{name} resolved to non-callable: {obj!r}"


# ====================================================================== #
# P6-E: get_factor
# ====================================================================== #


class TestGetFactor:
    def test_returns_spec(self):
        spec = get_factor("chip_score")
        assert spec is not None
        assert spec.name == "chip_score"

    def test_returns_none_for_unknown(self):
        assert get_factor("nonexistent_factor_12345") is None


# ====================================================================== #
# P6-F: CLI handler 退出碼與輸出
# ====================================================================== #


class TestCmdFactorList:
    def test_default_lists_all_returns_zero(self, capsys):
        exit_code = cmd_factor_list(
            argparse.Namespace(category=None, type=None, mode=None, name=None, check_resolve=False)
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "chip_score" in out
        assert "fundamental_score" in out
        assert f"合計 {len(FACTOR_REGISTRY)} 個因子" in out

    def test_unknown_name_returns_two(self, capsys):
        exit_code = cmd_factor_list(
            argparse.Namespace(category=None, type=None, mode=None, name="nope", check_resolve=False)
        )
        assert exit_code == 2

    def test_known_name_shows_detail_zero(self, capsys):
        exit_code = cmd_factor_list(
            argparse.Namespace(category=None, type=None, mode=None, name="chip_score", check_resolve=False)
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "dimension" in out
        assert "chip_score" in out
        assert "source_module" in out

    def test_filter_no_match_prints_message(self, capsys):
        exit_code = cmd_factor_list(
            argparse.Namespace(category="regime", type=None, mode=None, name=None, check_resolve=False)
        )
        # category=regime 目前無註冊因子；handler 仍應正常結束（exit 0）並印「無符合條件」
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "無符合條件" in out

    def test_check_resolve_passes(self, capsys):
        exit_code = cmd_factor_list(
            argparse.Namespace(category=None, type=None, mode=None, name=None, check_resolve=True)
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "resolve 通過" in out
