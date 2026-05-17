"""Factor Library — 全系統因子定義 SSOT（單一真相來源）。

P1 任務 6（2026-05-17）：解決「3 處因子定義可能漂移」問題。

設計策略（phased）：
  Phase 1（本任務）：metadata-only registry，不移動既有程式碼。每個因子在 FACTOR_REGISTRY
                    註冊 name / category / source_module / expected_sign / used_in_modes
                    等中介資料，供 audit / IC monitor / CLI 查詢時作 SSOT。
  Phase 2（後續）：分階段將函數本體遷入 src/factors/*.py，舊位置改 import。

當前內容：
  - registry.py：FactorSpec dataclass + FACTOR_REGISTRY dict + 查詢輔助
  - 目前散落位置（待 phase 2 整併）：
      src/features/indicators.py   — TechnicalIndicator EAV 持久化（SMA/RSI/MACD/BB/ADX）
      src/discovery/scanner/_functions.py — 4 維 composite score + 子因子（discover 主力）
      src/screener/factors.py      — watchlist 預測式 filter（回傳 bool Series）

使用方式：
  from src.factors import FACTOR_REGISTRY, get_factor, list_factors

  # IC audit 起手：列出所有 technical 類因子
  factors = list_factors(category="technical")

  # 查單一因子來源
  spec = get_factor("chip_score")
  print(spec.source_module, spec.source_function)
"""

from src.factors.registry import (
    FACTOR_REGISTRY,
    FactorSpec,
    get_factor,
    list_factors,
)

__all__ = ["FACTOR_REGISTRY", "FactorSpec", "get_factor", "list_factors"]
