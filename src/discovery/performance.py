"""Discover 推薦績效回測 — 評估歷史推薦的實際報酬率。

讀取 DiscoveryRecord 歷史推薦，對照 DailyPrice 計算推薦後 N 天的
實際報酬率，輸出勝率/平均報酬/最大虧損等統計。

包含策略衰減監控（compute_strategy_decay）：比較近期 vs 歷史勝率，
偵測模式績效衰退。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import DailyPrice, DiscoveryRecord

logger = logging.getLogger(__name__)


class DiscoveryPerformance:
    """評估 Discover 推薦績效。

    Parameters
    ----------
    mode : str
        掃描模式（momentum / swing / value）
    holding_days : list[int]
        持有天數清單，預設 [5, 10, 20]
    top_n : int | None
        只計算每次掃描前 N 名的績效（None = 全部）
    start_date : str | None
        掃描日期範圍起始（YYYY-MM-DD）
    end_date : str | None
        掃描日期範圍結束（YYYY-MM-DD）
    """

    MODE_LABELS = {
        "momentum": "Momentum 短線動能",
        "swing": "Swing 中期波段",
        "value": "Value 價值修復",
        "dividend": "Dividend 高息存股",
        "growth": "Growth 高成長",
    }

    # 各模式預設持有天數（未明確指定時使用）
    MODE_HORIZONS: dict[str, list[int]] = {
        "swing": [20, 40, 60],
    }

    def __init__(
        self,
        mode: str,
        holding_days: Optional[list[int]] = None,
        top_n: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.mode = mode
        self.holding_days = holding_days or self.MODE_HORIZONS.get(mode, [5, 10, 20])
        self.top_n = top_n
        self.start_date = date.fromisoformat(start_date) if start_date else None
        self.end_date = date.fromisoformat(end_date) if end_date else None

    def evaluate(self) -> dict:
        """回傳 { "summary": DataFrame, "by_scan": DataFrame, "detail": DataFrame }。

        若無推薦記錄，回傳三個空 DataFrame。
        """
        records_df = self._load_records()
        if records_df.empty:
            empty = pd.DataFrame()
            return {"summary": empty, "by_scan": empty, "detail": empty}

        prices_df = self._load_prices(records_df)
        if prices_df.empty:
            empty = pd.DataFrame()
            return {"summary": empty, "by_scan": empty, "detail": empty}

        detail = self._calc_returns(records_df, prices_df)
        if detail.empty:
            empty = pd.DataFrame()
            return {"summary": empty, "by_scan": empty, "detail": empty}

        summary = self._aggregate_summary(detail)
        by_scan = self._aggregate_by_scan(detail)

        return {"summary": summary, "by_scan": by_scan, "detail": detail}

    # ------------------------------------------------------------------
    # 內部方法
    # ------------------------------------------------------------------

    def _load_records(self) -> pd.DataFrame:
        """從 DB 載入 DiscoveryRecord。"""
        with get_session() as session:
            stmt = select(
                DiscoveryRecord.scan_date,
                DiscoveryRecord.stock_id,
                DiscoveryRecord.stock_name,
                DiscoveryRecord.rank,
                DiscoveryRecord.close,
                DiscoveryRecord.composite_score,
            ).where(DiscoveryRecord.mode == self.mode)

            if self.start_date:
                stmt = stmt.where(DiscoveryRecord.scan_date >= self.start_date)
            if self.end_date:
                stmt = stmt.where(DiscoveryRecord.scan_date <= self.end_date)
            if self.top_n:
                stmt = stmt.where(DiscoveryRecord.rank <= self.top_n)

            stmt = stmt.order_by(DiscoveryRecord.scan_date, DiscoveryRecord.rank)
            rows = session.execute(stmt).all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(
            rows,
            columns=["scan_date", "stock_id", "stock_name", "rank", "close", "composite_score"],
        )

    def _load_prices(self, records_df: pd.DataFrame) -> pd.DataFrame:
        """批次載入所有相關股票在推薦日之後的 DailyPrice。"""
        stock_ids = records_df["stock_id"].unique().tolist()
        min_date = records_df["scan_date"].min()
        max_holding = max(self.holding_days)

        with get_session() as session:
            stmt = (
                select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close)
                .where(
                    DailyPrice.stock_id.in_(stock_ids),
                    DailyPrice.date > min_date,
                )
                .order_by(DailyPrice.stock_id, DailyPrice.date)
            )
            rows = session.execute(stmt).all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows, columns=["stock_id", "date", "close"])

    def _calc_returns(self, records_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """計算每筆推薦在各持有天數的報酬率。"""
        results = []

        for _, rec in records_df.iterrows():
            scan_date = rec["scan_date"]
            stock_id = rec["stock_id"]
            entry_close = rec["close"]

            # 取該股票在推薦日之後的價格序列
            future_prices = (
                prices_df[(prices_df["stock_id"] == stock_id) & (prices_df["date"] > scan_date)]
                .sort_values("date")
                .reset_index(drop=True)
            )

            row = {
                "scan_date": scan_date,
                "stock_id": stock_id,
                "stock_name": rec["stock_name"],
                "rank": rec["rank"],
                "entry_close": entry_close,
                "composite_score": rec["composite_score"],
            }

            for days in self.holding_days:
                col_ret = f"return_{days}d"
                col_exit = f"exit_close_{days}d"
                if len(future_prices) >= days:
                    exit_close = future_prices.iloc[days - 1]["close"]
                    row[col_ret] = (exit_close - entry_close) / entry_close
                    row[col_exit] = exit_close
                else:
                    row[col_ret] = None
                    row[col_exit] = None

            results.append(row)

        return pd.DataFrame(results)

    def _aggregate_summary(self, detail: pd.DataFrame) -> pd.DataFrame:
        """每個持有天數一行，整體統計。"""
        rows = []
        for days in self.holding_days:
            col = f"return_{days}d"
            valid = detail[col].dropna()
            if valid.empty:
                rows.append(
                    {
                        "holding_days": days,
                        "evaluable": 0,
                        "win_rate": None,
                        "avg_return": None,
                        "median_return": None,
                        "max_gain": None,
                        "max_loss": None,
                    }
                )
                continue

            rows.append(
                {
                    "holding_days": days,
                    "evaluable": len(valid),
                    "win_rate": (valid > 0).mean(),
                    "avg_return": valid.mean(),
                    "median_return": valid.median(),
                    "max_gain": valid.max(),
                    "max_loss": valid.min(),
                }
            )

        return pd.DataFrame(rows)

    def _aggregate_by_scan(self, detail: pd.DataFrame) -> pd.DataFrame:
        """每個掃描日期 × 持有天數的統計。"""
        rows = []
        for days in self.holding_days:
            col = f"return_{days}d"
            for scan_date, group in detail.groupby("scan_date"):
                valid = group[col].dropna()
                if valid.empty:
                    continue

                # 最佳/最差個股
                best_idx = valid.idxmax()
                worst_idx = valid.idxmin()
                best_row = detail.loc[best_idx]
                worst_row = detail.loc[worst_idx]

                rows.append(
                    {
                        "holding_days": days,
                        "scan_date": scan_date,
                        "count": len(valid),
                        "win_rate": (valid > 0).mean(),
                        "avg_return": valid.mean(),
                        "best_stock": best_row["stock_id"],
                        "best_return": valid.max(),
                        "worst_stock": worst_row["stock_id"],
                        "worst_return": valid.min(),
                    }
                )

        return pd.DataFrame(rows)


def print_performance_report(result: dict, mode: str, start_date=None, end_date=None) -> None:
    """輸出績效報告到 console。"""
    summary = result["summary"]
    by_scan = result["by_scan"]
    detail = result["detail"]

    if summary.empty:
        print("無推薦記錄或無法計算績效")
        return

    mode_label = DiscoveryPerformance.MODE_LABELS.get(mode, mode)

    # 統計掃描次數與推薦數
    scan_count = detail["scan_date"].nunique()
    total_recs = len(detail)

    # 期間
    date_min = detail["scan_date"].min()
    date_max = detail["scan_date"].max()
    if start_date:
        date_min = start_date
    if end_date:
        date_max = end_date

    print(f"\n{'=' * 80}")
    print(f"Discover 推薦績效回測 [{mode_label}]")
    print(f"掃描期間：{date_min} ~ {date_max}（共 {scan_count} 次掃描，{total_recs} 筆推薦）")
    print(f"{'=' * 80}")

    # 整體摘要
    print(
        f"\n{'持有天數':>8}  {'可評估':>6}  {'勝率':>7}  {'平均報酬':>8}  {'中位數':>8}  {'最大獲利':>8}  {'最大虧損':>8}"
    )
    print(f"{'─' * 70}")
    for _, row in summary.iterrows():
        days = int(row["holding_days"])
        evaluable = int(row["evaluable"])
        if evaluable == 0:
            print(f"  {days:>3}天     {evaluable:>5}      —         —         —         —         —")
            continue
        wr = f"{row['win_rate']:.1%}"
        avg = f"{row['avg_return']:+.2%}"
        med = f"{row['median_return']:+.2%}"
        mg = f"{row['max_gain']:+.2%}"
        ml = f"{row['max_loss']:+.2%}"
        print(f"  {days:>3}天     {evaluable:>5}  {wr:>7}  {avg:>8}  {med:>8}  {mg:>8}  {ml:>8}")

    # 逐次掃描明細（每個持有天數各一段）
    if not by_scan.empty:
        holding_days = sorted(by_scan["holding_days"].unique())
        for days in holding_days:
            subset = by_scan[by_scan["holding_days"] == days].sort_values("scan_date", ascending=False)
            print(f"\n{'─' * 75}")
            print(f"逐次掃描績效（持有 {days} 天）")
            print(f"{'─' * 75}")
            print(f"{'掃描日期':>10}  {'推薦數':>5}  {'勝率':>7}  {'平均報酬':>8}  {'最佳':>16}  {'最差':>16}")

            for _, row in subset.iterrows():
                wr = f"{row['win_rate']:.1%}"
                avg = f"{row['avg_return']:+.2%}"
                best = f"{row['best_stock']} {row['best_return']:+.1%}"
                worst = f"{row['worst_stock']} {row['worst_return']:+.1%}"
                print(f"{row['scan_date']}  {int(row['count']):>5}  {wr:>7}  {avg:>8}  {best:>16}  {worst:>16}")


# ------------------------------------------------------------------
#  策略衰減監控（純函數）
# ------------------------------------------------------------------

# 衰減警告閾值
DECAY_WIN_RATE_THRESHOLD = 0.40  # 近期勝率低於 40% 視為衰減
DECAY_RETURN_THRESHOLD = 0.0  # 近期平均報酬低於 0% 視為衰減


def compute_strategy_decay(
    detail_df: pd.DataFrame,
    recent_days: int = 30,
    holding_days: int = 10,
    reference_date: date | None = None,
) -> dict:
    """比較近期 vs 歷史的 Discover 推薦績效，偵測策略衰減。

    純函數：接收 _calc_returns() 的 detail DataFrame，不碰 DB。

    Parameters
    ----------
    detail_df : pd.DataFrame
        含 scan_date, return_{N}d 欄位的推薦績效明細表。
    recent_days : int
        「近期」的定義（天數），預設 30。
    holding_days : int
        使用哪個持有天數欄位，預設 10。
    reference_date : date | None
        基準日期（預設 today），近期 = reference_date - recent_days ~ reference_date。

    Returns
    -------
    dict
        mode, holding_days, recent_days,
        recent_count, recent_win_rate, recent_avg_return,
        historical_count, historical_win_rate, historical_avg_return,
        win_rate_decay (pct points), return_decay (pct points),
        is_decaying (bool), warning (str | None)
    """
    ref = reference_date or date.today()
    ret_col = f"return_{holding_days}d"

    if ret_col not in detail_df.columns or detail_df.empty:
        return _empty_decay_result(holding_days, recent_days)

    df = detail_df.copy()
    df["scan_date"] = pd.to_datetime(df["scan_date"]).dt.date

    cutoff = ref - timedelta(days=recent_days)
    recent = df[df["scan_date"] >= cutoff][ret_col].dropna()
    historical = df[df["scan_date"] < cutoff][ret_col].dropna()

    if recent.empty:
        return _empty_decay_result(holding_days, recent_days)

    recent_wr = float((recent > 0).mean())
    recent_avg = float(recent.mean())

    # 歷史基線（若無歷史資料，使用全部資料作基線）
    if historical.empty:
        hist_wr = recent_wr
        hist_avg = recent_avg
        hist_count = 0
    else:
        hist_wr = float((historical > 0).mean())
        hist_avg = float(historical.mean())
        hist_count = len(historical)

    wr_decay = (recent_wr - hist_wr) * 100  # pct points
    ret_decay = (recent_avg - hist_avg) * 100  # pct points

    is_decaying = recent_wr < DECAY_WIN_RATE_THRESHOLD or recent_avg < DECAY_RETURN_THRESHOLD

    warning = None
    if is_decaying:
        parts = []
        if recent_wr < DECAY_WIN_RATE_THRESHOLD:
            parts.append(f"勝率 {recent_wr:.0%} < {DECAY_WIN_RATE_THRESHOLD:.0%}")
        if recent_avg < DECAY_RETURN_THRESHOLD:
            parts.append(f"均報酬 {recent_avg:+.2%} < 0")
        warning = f"近 {recent_days} 天績效衰減：{'、'.join(parts)}"

    return {
        "holding_days": holding_days,
        "recent_days": recent_days,
        "recent_count": len(recent),
        "recent_win_rate": round(recent_wr, 4),
        "recent_avg_return": round(recent_avg, 4),
        "historical_count": hist_count,
        "historical_win_rate": round(hist_wr, 4),
        "historical_avg_return": round(hist_avg, 4),
        "win_rate_decay": round(wr_decay, 2),
        "return_decay": round(ret_decay, 2),
        "is_decaying": is_decaying,
        "warning": warning,
    }


def _empty_decay_result(holding_days: int, recent_days: int) -> dict:
    return {
        "holding_days": holding_days,
        "recent_days": recent_days,
        "recent_count": 0,
        "recent_win_rate": None,
        "recent_avg_return": None,
        "historical_count": 0,
        "historical_win_rate": None,
        "historical_avg_return": None,
        "win_rate_decay": None,
        "return_decay": None,
        "is_decaying": False,
        "warning": None,
    }


def check_all_modes_decay(
    modes: list[str] | None = None,
    recent_days: int = 30,
    holding_days: int = 10,
) -> list[dict]:
    """檢查所有 Discover 模式的策略衰減（DB 版本）。

    Returns
    -------
    list[dict]
        每個模式一個 decay 結果 dict（含 mode 欄位）。
    """
    if modes is None:
        modes = ["momentum", "swing", "value", "dividend", "growth"]

    results = []
    for mode in modes:
        perf = DiscoveryPerformance(mode=mode, holding_days=[holding_days])
        eval_result = perf.evaluate()
        detail = eval_result.get("detail", pd.DataFrame())

        decay = compute_strategy_decay(
            detail,
            recent_days=recent_days,
            holding_days=holding_days,
        )
        decay["mode"] = mode
        results.append(decay)

    return results
