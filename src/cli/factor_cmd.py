"""factor-list CLI — 列出 src/factors FACTOR_REGISTRY 中所有因子（P1 任務 6）。

用法：
  python main.py factor-list                          # 列出全部
  python main.py factor-list --category chip          # 限定維度
  python main.py factor-list --type sub_factor        # 限定類型
  python main.py factor-list --mode momentum          # 限定 discover 模式
  python main.py factor-list --name chip_score        # 顯示單一 spec 詳細
  python main.py factor-list --check-resolve          # introspection 驗證 source_module/function 可解析
"""

from __future__ import annotations

import argparse

from src.cli.helpers import safe_print as print
from src.factors import FACTOR_REGISTRY, FactorSpec, get_factor, list_factors

# ANSI 顏色（可被 NO_COLOR 環境變數關閉，sys.stdout.isatty 由 safe_print 自行處理）
_TAG_BY_TYPE = {
    "dimension": "★",
    "sub_factor": "·",
    "predicate": "?",
    "indicator": "#",
}


def _print_table(factors: list[FactorSpec]) -> None:
    """列印彙整表（簡潔，適合大量顯示）。"""
    if not factors:
        print("（無符合條件的因子）")
        return

    # 依 category 分組顯示
    last_cat: str | None = None
    print(f"\n{'類型':2s} {'name':<32s} {'category':<14s} {'expected':4s} {'modes':<28s} description")
    print("─" * 130)
    for f in factors:
        if f.category != last_cat:
            last_cat = f.category
        tag = _TAG_BY_TYPE.get(f.factor_type, " ")
        modes = ",".join(f.used_in_modes) if f.used_in_modes else "-"
        if len(modes) > 28:
            modes = modes[:25] + "..."
        desc = f.description
        if len(desc) > 50:
            desc = desc[:47] + "..."
        print(f"{tag:2s} {f.name:<32s} {f.category:<14s} {f.expected_sign:<4s} {modes:<28s} {desc}")
    print(f"\n  合計 {len(factors)} 個因子")
    print("  圖例：★ dimension / · sub_factor / ? predicate / # indicator\n")


def _print_spec_detail(spec: FactorSpec) -> None:
    """單一 spec 完整資訊。"""
    print(f"\n{'═' * 64}")
    print(f"  [{spec.factor_type}] {spec.name}  ({spec.category})")
    print(f"{'═' * 64}")
    print(f"  描述:           {spec.description}")
    print(f"  source_module:  {spec.source_module}")
    print(f"  source_function:{spec.source_function or '(module-level)'}")
    print(f"  expected_sign:  {spec.expected_sign}")
    if spec.used_in_modes:
        print(f"  used_in_modes:  {', '.join(spec.used_in_modes)}")
    if spec.holding_days_target is not None:
        print(f"  holding_days:   {spec.holding_days_target}")
    if spec.ic_notes:
        print(f"  IC notes:       {spec.ic_notes}")
    # introspection
    resolved = spec.resolve()
    if resolved is None:
        print("  ⚠ resolve():    failed（source_module 或 source_function 找不到）")
    else:
        print(f"  resolve():      ✓ {resolved!r}")
    print()


def cmd_factor_list(args: argparse.Namespace) -> int:
    """factor-list CLI handler。回傳 exit code（0 = 全部 resolve 成功 / 1 = 有 resolve 失敗）。"""
    name = getattr(args, "name", None)
    category = getattr(args, "category", None)
    factor_type = getattr(args, "type", None)
    mode = getattr(args, "mode", None)
    check_resolve = getattr(args, "check_resolve", False)

    if name:
        spec = get_factor(name)
        if spec is None:
            print(f"找不到因子: {name}")
            print(f"可用因子：{', '.join(sorted(FACTOR_REGISTRY.keys()))}")
            return 2
        _print_spec_detail(spec)
        return 0 if (not check_resolve) or spec.resolve() is not None else 1

    factors = list_factors(category=category, factor_type=factor_type, used_in_mode=mode)

    if check_resolve:
        unresolved = [f for f in factors if f.resolve() is None]
        _print_table(factors)
        if unresolved:
            print(f"\n  ⚠ {len(unresolved)} 個因子無法 resolve：")
            for f in unresolved:
                print(f"    {f.name}  →  {f.source_module}.{f.source_function}")
            return 1
        print(f"  ✅ 全部 {len(factors)} 個因子 resolve 通過")
        return 0

    _print_table(factors)
    return 0
