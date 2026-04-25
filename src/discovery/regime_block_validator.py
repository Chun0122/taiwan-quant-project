"""REGIME_MODE_BLOCK 矩陣自動驗證器。

定期掃過 (regime × mode) 對照歷史推薦績效，給出維持/解除/新增封鎖的建議。
本模組僅提供「建議」，不自動修改 constants.py — 規則變更走 PR/Code Review。

使用情境：
  - morning-routine 每月第一個交易日呼叫一次，發 Discord 通知
  - 手動執行：python -c "from src.discovery.regime_block_validator import validate_regime_blocks; ..."
  - 迴歸測試：test_current_block_matrix_passes_validation_today

判定邏輯（_judge）：
  - 樣本 < min_samples            → "insufficient_data"（不下判斷）
  - 已封鎖 + 報酬>+2% + 勝率>55%  → "lift_block"（封鎖過嚴）
  - 未封鎖 + 報酬<-3% + 勝率<35%  → "add_block"（建議新增封鎖）
  - 其他                          → "keep"（維持現狀）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import pandas as pd
from sqlalchemy import select

from src.constants import REGIME_MODE_BLOCK
from src.data import database as _db_mod
from src.data.schema import DailyPrice, DiscoveryRecord

logger = logging.getLogger(__name__)

ALL_REGIMES = ("bull", "bear", "sideways", "crisis")
ALL_MODES = ("momentum", "swing", "value", "dividend", "growth")

Recommendation = Literal["keep", "lift_block", "add_block", "insufficient_data"]


@dataclass(frozen=True)
class HoldingStats:
    """單一 (regime, mode) 在持有期 N 日的歷史績效摘要。"""

    avg_return: float  # 平均報酬（小數，0.025=2.5%）
    win_rate: float  # 勝率（小數，0.55=55%）
    sample_count: int  # 樣本數（推薦筆數）


@dataclass(frozen=True)
class BlockValidation:
    """單一 (regime, mode) 的封鎖驗證結果。"""

    regime: str
    mode: str
    is_blocked: bool
    stats: HoldingStats
    recommendation: Recommendation

    @property
    def avg_return(self) -> float:
        return self.stats.avg_return

    @property
    def win_rate(self) -> float:
        return self.stats.win_rate

    @property
    def sample_count(self) -> int:
        return self.stats.sample_count


def _judge(
    is_blocked: bool,
    stats: HoldingStats,
    *,
    min_samples: int = 30,
    lift_return_threshold: float = 0.02,
    lift_win_rate_threshold: float = 0.55,
    add_return_threshold: float = -0.03,
    add_win_rate_threshold: float = 0.35,
) -> Recommendation:
    """純函數判定 — 給定統計與封鎖狀態，回傳建議。

    Args:
        is_blocked: 當前是否在 REGIME_MODE_BLOCK 中
        stats: 歷史績效統計
        min_samples: 低於此樣本數不下判斷
        lift_return_threshold: 已封鎖但報酬高於此值 → 建議解除
        lift_win_rate_threshold: 配合 lift_return — 兩條件 AND
        add_return_threshold: 未封鎖但報酬低於此值 → 建議新增封鎖
        add_win_rate_threshold: 配合 add_return — 兩條件 AND
    """
    if stats.sample_count < min_samples:
        return "insufficient_data"
    if is_blocked:
        if stats.avg_return > lift_return_threshold and stats.win_rate > lift_win_rate_threshold:
            return "lift_block"
        return "keep"
    # 未封鎖
    if stats.avg_return < add_return_threshold and stats.win_rate < add_win_rate_threshold:
        return "add_block"
    return "keep"


def _query_holding_stats(
    regime: str,
    mode: str,
    holding_days: int,
    lookback_days: int,
    today: date | None = None,
) -> HoldingStats:
    """查詢 (regime, mode) 在過去 lookback_days 內的 holding_days 持有期績效。

    使用 DiscoveryRecord（含 regime 欄位）+ DailyPrice 計算 forward return。
    僅統計持有期可完整評估的推薦（exit date <= today）。
    """
    if today is None:
        today = date.today()
    cutoff_start = today - timedelta(days=lookback_days)
    # 持有期需完整評估 → scan_date 必須早於 today - holding_days
    cutoff_end = today - timedelta(days=holding_days)

    with _db_mod.get_session() as session:
        rec_rows = session.execute(
            select(
                DiscoveryRecord.scan_date,
                DiscoveryRecord.stock_id,
                DiscoveryRecord.close,
                DiscoveryRecord.regime,
            ).where(
                DiscoveryRecord.mode == mode,
                DiscoveryRecord.regime == regime,
                DiscoveryRecord.scan_date >= cutoff_start,
                DiscoveryRecord.scan_date <= cutoff_end,
            )
        ).all()
        if not rec_rows:
            return HoldingStats(avg_return=0.0, win_rate=0.0, sample_count=0)

        df_rec = pd.DataFrame(rec_rows, columns=["scan_date", "stock_id", "close", "regime"])
        stock_ids = df_rec["stock_id"].unique().tolist()

        price_rows = session.execute(
            select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close)
            .where(
                DailyPrice.stock_id.in_(stock_ids),
                DailyPrice.date >= cutoff_start,
                DailyPrice.date <= today,
            )
            .order_by(DailyPrice.stock_id, DailyPrice.date)
        ).all()
    if not price_rows:
        return HoldingStats(avg_return=0.0, win_rate=0.0, sample_count=0)

    df_price = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])
    return _compute_stats_from_dfs(df_rec, df_price, holding_days)


def _compute_stats_from_dfs(
    df_rec: pd.DataFrame,
    df_price: pd.DataFrame,
    holding_days: int,
) -> HoldingStats:
    """從推薦 DataFrame + 價格 DataFrame 計算 HoldingStats（純函數，便於測試）。"""
    if df_rec.empty or df_price.empty:
        return HoldingStats(avg_return=0.0, win_rate=0.0, sample_count=0)

    returns: list[float] = []
    grouped = df_price.groupby("stock_id")
    for _, rec in df_rec.iterrows():
        sid = rec["stock_id"]
        entry = float(rec["close"])
        if entry <= 0:
            continue
        if sid not in grouped.groups:
            continue
        sub = grouped.get_group(sid)
        future = sub[sub["date"] > rec["scan_date"]].sort_values("date").head(holding_days)
        if len(future) < holding_days:
            continue
        exit_close = float(future.iloc[-1]["close"])
        returns.append((exit_close - entry) / entry)

    if not returns:
        return HoldingStats(avg_return=0.0, win_rate=0.0, sample_count=0)
    s = pd.Series(returns)
    return HoldingStats(
        avg_return=float(s.mean()),
        win_rate=float((s > 0).mean()),
        sample_count=len(returns),
    )


def validate_regime_blocks(
    holding_days: int = 5,
    lookback_days: int = 90,
    min_samples: int = 30,
    today: date | None = None,
) -> list[BlockValidation]:
    """掃過完整 (regime × mode) 矩陣，回傳每格的封鎖驗證結果。

    僅針對「樣本足夠」的格子下判斷，其餘標 insufficient_data 跳過。
    """
    results: list[BlockValidation] = []
    for regime in ALL_REGIMES:
        blocked_set = REGIME_MODE_BLOCK.get(regime, frozenset())
        for mode in ALL_MODES:
            stats = _query_holding_stats(
                regime=regime,
                mode=mode,
                holding_days=holding_days,
                lookback_days=lookback_days,
                today=today,
            )
            is_blocked = mode in blocked_set
            recommendation = _judge(is_blocked, stats, min_samples=min_samples)
            results.append(
                BlockValidation(
                    regime=regime,
                    mode=mode,
                    is_blocked=is_blocked,
                    stats=stats,
                    recommendation=recommendation,
                )
            )
    return results


def format_validation_report(validations: list[BlockValidation]) -> str:
    """格式化驗證結果為 console / Discord 文字輸出。"""
    actionable = [v for v in validations if v.recommendation in ("lift_block", "add_block")]
    insufficient = [v for v in validations if v.recommendation == "insufficient_data"]
    keep = [v for v in validations if v.recommendation == "keep"]

    lines: list[str] = []
    lines.append(f"\n{'=' * 70}")
    lines.append("REGIME_MODE_BLOCK 矩陣驗證報告")
    lines.append(f"{'=' * 70}")
    lines.append(f"  維持現狀: {len(keep)}  |  建議調整: {len(actionable)}  |  樣本不足: {len(insufficient)}")

    if actionable:
        lines.append(f"\n{'─' * 70}")
        lines.append("⚠ 建議調整（請人工 review 後決定是否修改 constants.py）")
        lines.append(f"{'─' * 70}")
        lines.append(f"{'regime':<10} {'mode':<10} {'狀態':<8} {'建議':<12} {'樣本':>5} {'勝率':>6} {'平均報酬':>9}")
        for v in actionable:
            status = "已封鎖" if v.is_blocked else "未封鎖"
            rec_label = "解除封鎖" if v.recommendation == "lift_block" else "新增封鎖"
            lines.append(
                f"{v.regime:<10} {v.mode:<10} {status:<8} {rec_label:<12} "
                f"{v.sample_count:>5} {v.win_rate:>6.1%} {v.avg_return:>+9.2%}"
            )

    if keep:
        lines.append(f"\n{'─' * 70}")
        lines.append("✓ 維持現狀（封鎖矩陣與歷史績效一致）")
        lines.append(f"{'─' * 70}")
        for v in keep:
            status = "已封鎖" if v.is_blocked else "未封鎖"
            lines.append(
                f"  {v.regime:<10} {v.mode:<10} {status:<8} "
                f"n={v.sample_count:<4} 勝率 {v.win_rate:>6.1%} 報酬 {v.avg_return:>+7.2%}"
            )

    return "\n".join(lines)
