"""全市場選股掃描器 — 四階段漏斗從 ~2000 支股票篩選出 Top N 推薦。

漏斗架構：
  Stage 1: 從 DB 載入全市場日K + 法人資料
  Stage 2: 粗篩（股價/成交量/法人/動能 → 留 ~150 檔）
  Stage 3: 細評（模式專屬因子加權）
  Stage 3.5: 風險過濾（剔除高波動股）
  Stage 4: 排名 + 加上產業標籤 → 輸出 Top N

支援五種模式：
  - MomentumScanner: 短線動能（1~10 天），突破 + 資金流 + 量能擴張
  - SwingScanner: 中期波段（1~3 個月），趨勢 + 基本面 + 法人布局
  - ValueScanner: 價值修復，低估值 + 基本面轉佳
  - DividendScanner: 高息存股，高殖利率 + 配息穩定 + 估值合理
  - GrowthScanner: 高成長，營收高速成長 + 動能啟動

此套件將原 scanner.py（5300+ 行）拆分為模組：
  _functions.py  — 模組級純函數與常數（compute_whale_score 等 14 個純函數）
  _base.py       — MarketScanner 基底類別
  _momentum.py   — MomentumScanner
  _swing.py      — SwingScanner
  _value.py      — ValueScanner
  _dividend.py   — DividendScanner
  _growth.py     — GrowthScanner

所有公開名稱從此 __init__.py 重新匯出，維持向後相容：
  from src.discovery.scanner import MomentumScanner  # 仍然可用
"""

# --- 純函數 & 常數（_functions.py）---
# --- 基底類別（_base.py）---
from src.discovery.scanner._base import MarketScanner
from src.discovery.scanner._dividend import DividendScanner
from src.discovery.scanner._functions import (
    DiscoveryResult,
    ScanAuditTrail,
    _calc_atr14,
    _extract_level_lower_bound,
    compute_abnormal_announcement_rate,
    compute_adaptive_atr_multiplier,
    compute_broker_score,
    compute_chip_macd,
    compute_daytrade_penalty,
    compute_eps_sustainability,
    compute_factor_correlation_matrix,
    compute_factor_ic,
    compute_hhi_trend,
    compute_ic_weight_adjustments,
    compute_inst_net_buy_slope,
    compute_institutional_persistence,
    compute_key_player_cost_basis,
    compute_mfe_mae,
    compute_news_decay_weight,
    compute_peer_fundamental_ranking,
    compute_relative_pe_thresholds,
    compute_revenue_acceleration_score,
    compute_sbl_score,
    compute_smart_broker_score,
    compute_sub_factor_ic,
    compute_taiex_relative_strength,
    compute_vcp_score,
    compute_whale_score,
    compute_win_rate_threshold_adjustment,
    detect_daytrade_brokers,
    score_key_player_cost,
)
from src.discovery.scanner._growth import GrowthScanner

# --- 子類（各模式 Scanner）---
from src.discovery.scanner._momentum import MomentumScanner
from src.discovery.scanner._swing import SwingScanner
from src.discovery.scanner._value import ValueScanner

__all__ = [
    # 純函數
    "compute_news_decay_weight",
    "compute_abnormal_announcement_rate",
    "compute_taiex_relative_strength",
    "compute_relative_pe_thresholds",
    "compute_eps_sustainability",
    "compute_vcp_score",
    "_calc_atr14",
    "_extract_level_lower_bound",
    "compute_whale_score",
    "compute_sbl_score",
    "compute_broker_score",
    "compute_institutional_persistence",
    "compute_inst_net_buy_slope",
    "compute_hhi_trend",
    "detect_daytrade_brokers",
    "compute_daytrade_penalty",
    "compute_smart_broker_score",
    "compute_chip_macd",
    "compute_win_rate_threshold_adjustment",
    "compute_factor_ic",
    "compute_ic_weight_adjustments",
    "compute_key_player_cost_basis",
    "score_key_player_cost",
    "compute_adaptive_atr_multiplier",
    "compute_revenue_acceleration_score",
    "compute_peer_fundamental_ranking",
    "compute_mfe_mae",
    "compute_sub_factor_ic",
    "compute_factor_correlation_matrix",
    # 資料容器
    "DiscoveryResult",
    "ScanAuditTrail",
    # 類別
    "MarketScanner",
    "MomentumScanner",
    "SwingScanner",
    "ValueScanner",
    "DividendScanner",
    "GrowthScanner",
]
