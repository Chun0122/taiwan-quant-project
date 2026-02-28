"""資料品質檢查模組。

提供 6 個純函數檢查 + orchestrator + console 報告輸出：
- check_missing_days: 缺漏交易日偵測
- check_zero_volume: 零成交量偵測
- check_limit_streaks: 連續漲跌停偵測
- check_price_anomalies: 價格異常偵測（high<low、close 超出範圍、負價格）
- check_date_range_consistency: 同股不同表日期範圍一致性
- check_data_freshness: 資料新鮮度檢查
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 資料結構
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """單一品質問題。"""

    stock_id: str
    check_name: str
    severity: str  # "error" | "warning"
    description: str
    details: dict | None = None


@dataclass
class TableCoverage:
    """單一資料表的覆蓋範圍資訊。"""

    table_name: str
    stock_count: int
    row_count: int
    min_date: date | None
    max_date: date | None


@dataclass
class ValidationReport:
    """品質檢查報告。"""

    checked_stocks: int
    total_issues: int
    issues: list[ValidationIssue]
    summary: dict[str, int]  # {check_name: count}
    table_coverage: list[TableCoverage] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 純函數檢查
# ---------------------------------------------------------------------------


def check_missing_days(
    stock_id: str,
    dates: list[date],
    gap_threshold: int = 5,
) -> list[ValidationIssue]:
    """檢測交易日缺漏。

    以營業日差距判斷，連續缺漏 >= gap_threshold 個營業日視為異常。
    """
    if len(dates) < 2:
        return []

    sorted_dates = sorted(dates)
    issues: list[ValidationIssue] = []

    for i in range(1, len(sorted_dates)):
        prev = sorted_dates[i - 1]
        curr = sorted_dates[i]
        # 計算營業日差距（排除週末）
        bdays = int(np.busday_count(prev, curr))
        # bdays 包含 prev 到 curr 之間的營業日數，正常應為 1
        gap = bdays - 1  # 缺漏的營業日數
        if gap >= gap_threshold:
            issues.append(
                ValidationIssue(
                    stock_id=stock_id,
                    check_name="missing_days",
                    severity="error",
                    description=f"缺漏 {gap} 個營業日：{prev} ~ {curr}",
                    details={"from": str(prev), "to": str(curr), "gap": gap},
                )
            )

    return issues


def check_zero_volume(
    stock_id: str,
    df: pd.DataFrame,
) -> list[ValidationIssue]:
    """檢測零成交量。

    連續 3+ 天零成交量為 error，單日為 warning。
    df 需含 'volume' 欄位，index 為 date。
    """
    if df.empty or "volume" not in df.columns:
        return []

    issues: list[ValidationIssue] = []
    zero_mask = df["volume"] == 0
    zero_dates = df.index[zero_mask].tolist()

    if not zero_dates:
        return []

    # 找連續區段
    groups: list[list] = []
    current_group: list = [zero_dates[0]]

    for i in range(1, len(zero_dates)):
        prev = zero_dates[i - 1]
        curr = zero_dates[i]
        # 如果兩個日期之間的營業日差距 <= 1，視為連續
        prev_d = prev.date() if hasattr(prev, "date") else prev
        curr_d = curr.date() if hasattr(curr, "date") else curr
        bdays = int(np.busday_count(prev_d, curr_d))
        if bdays <= 1:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]
    groups.append(current_group)

    for group in groups:
        if len(group) >= 3:
            issues.append(
                ValidationIssue(
                    stock_id=stock_id,
                    check_name="zero_volume",
                    severity="error",
                    description=f"連續 {len(group)} 天零成交量：{group[0]} ~ {group[-1]}",
                    details={"start": str(group[0]), "end": str(group[-1]), "days": len(group)},
                )
            )
        else:
            for d in group:
                issues.append(
                    ValidationIssue(
                        stock_id=stock_id,
                        check_name="zero_volume",
                        severity="warning",
                        description=f"零成交量：{d}",
                        details={"date": str(d)},
                    )
                )

    return issues


def check_limit_streaks(
    stock_id: str,
    df: pd.DataFrame,
    limit_pct: float = 9.5,
    streak_threshold: int = 5,
) -> list[ValidationIssue]:
    """檢測連續漲跌停。

    日報酬率絕對值 >= limit_pct% 且同方向連續 >= streak_threshold 天。
    df 需含 'close' 欄位，index 為 date。
    """
    if df.empty or "close" not in df.columns or len(df) < 2:
        return []

    issues: list[ValidationIssue] = []
    returns = df["close"].pct_change() * 100  # 百分比

    # 標記漲停/跌停
    up_limit = returns >= limit_pct
    down_limit = returns <= -limit_pct

    for direction, mask, label in [("up", up_limit, "漲停"), ("down", down_limit, "跌停")]:
        # 找連續 streak
        streak_start = None
        streak_len = 0

        for i in range(len(mask)):
            if mask.iloc[i]:
                if streak_start is None:
                    streak_start = i
                streak_len += 1
            else:
                if streak_len >= streak_threshold and streak_start is not None:
                    start_date = df.index[streak_start]
                    end_date = df.index[streak_start + streak_len - 1]
                    issues.append(
                        ValidationIssue(
                            stock_id=stock_id,
                            check_name="limit_streak",
                            severity="warning",
                            description=f"連續 {streak_len} 天{label}：{start_date} ~ {end_date}",
                            details={
                                "direction": direction,
                                "start": str(start_date),
                                "end": str(end_date),
                                "days": streak_len,
                            },
                        )
                    )
                streak_start = None
                streak_len = 0

        # 處理尾部 streak
        if streak_len >= streak_threshold and streak_start is not None:
            start_date = df.index[streak_start]
            end_date = df.index[streak_start + streak_len - 1]
            issues.append(
                ValidationIssue(
                    stock_id=stock_id,
                    check_name="limit_streak",
                    severity="warning",
                    description=f"連續 {streak_len} 天{label}：{start_date} ~ {end_date}",
                    details={
                        "direction": direction,
                        "start": str(start_date),
                        "end": str(end_date),
                        "days": streak_len,
                    },
                )
            )

    return issues


def check_price_anomalies(
    stock_id: str,
    df: pd.DataFrame,
) -> list[ValidationIssue]:
    """檢測價格異常：high < low、close 超出 [low, high] 範圍、價格 <= 0。

    df 需含 'open', 'high', 'low', 'close' 欄位，index 為 date。
    """
    if df.empty:
        return []

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return []

    issues: list[ValidationIssue] = []

    # high < low
    bad_hl = df[df["high"] < df["low"]]
    for d, row in bad_hl.iterrows():
        issues.append(
            ValidationIssue(
                stock_id=stock_id,
                check_name="price_anomaly",
                severity="error",
                description=f"最高價 < 最低價：{d}（high={row['high']}, low={row['low']}）",
                details={"date": str(d), "type": "high_lt_low"},
            )
        )

    # close 超出 [low, high] 範圍
    bad_close = df[(df["close"] > df["high"]) | (df["close"] < df["low"])]
    # 排除已在 high<low 中報告的
    bad_close = bad_close[~bad_close.index.isin(bad_hl.index)]
    for d, row in bad_close.iterrows():
        issues.append(
            ValidationIssue(
                stock_id=stock_id,
                check_name="price_anomaly",
                severity="error",
                description=f"收盤價超出當日高低範圍：{d}（close={row['close']}, L={row['low']}, H={row['high']}）",
                details={"date": str(d), "type": "close_out_of_range"},
            )
        )

    # 價格 <= 0
    for col in ["open", "high", "low", "close"]:
        bad = df[df[col] <= 0]
        for d, row in bad.iterrows():
            issues.append(
                ValidationIssue(
                    stock_id=stock_id,
                    check_name="price_anomaly",
                    severity="error",
                    description=f"{col} <= 0：{d}（{col}={row[col]}）",
                    details={"date": str(d), "type": "non_positive", "column": col},
                )
            )

    return issues


def check_date_range_consistency(
    table_ranges: dict[str, dict[str, tuple[date | None, date | None]]],
) -> list[ValidationIssue]:
    """檢測同一股票在不同資料表的日期範圍一致性。

    table_ranges 格式：
    {
        "daily_price": {"2330": (date(2020,1,2), date(2025,1,10)), ...},
        "institutional_investor": {"2330": (date(2020,1,2), date(2025,1,8)), ...},
    }

    若同股在不同表的 max_date 差距 > 30 天，視為 warning。
    """
    issues: list[ValidationIssue] = []

    # 收集每支股票在各表的 max_date
    stock_tables: dict[str, dict[str, tuple[date | None, date | None]]] = {}
    for table_name, stocks in table_ranges.items():
        for sid, (min_d, max_d) in stocks.items():
            stock_tables.setdefault(sid, {})[table_name] = (min_d, max_d)

    base_table = "daily_price"

    for sid, tables in stock_tables.items():
        if base_table not in tables:
            continue

        base_min, base_max = tables[base_table]
        if base_max is None:
            continue

        for table_name, (t_min, t_max) in tables.items():
            if table_name == base_table or t_max is None:
                continue
            diff = abs((base_max - t_max).days)
            if diff > 30:
                issues.append(
                    ValidationIssue(
                        stock_id=sid,
                        check_name="date_range_consistency",
                        severity="warning",
                        description=(
                            f"{table_name} 最新日期落後 daily_price {diff} 天"
                            f"（{table_name}: {t_max}, daily_price: {base_max}）"
                        ),
                        details={
                            "table": table_name,
                            "table_max": str(t_max),
                            "base_max": str(base_max),
                            "diff_days": diff,
                        },
                    )
                )

    return issues


def check_data_freshness(
    table_ranges: dict[str, tuple[date | None, date | None]],
    reference_date: date,
    stale_threshold: int = 7,
) -> list[ValidationIssue]:
    """檢測資料新鮮度。

    table_ranges 格式：{"daily_price": (min_date, max_date), ...}
    若最新資料距 reference_date 超過 stale_threshold 個營業日，視為 warning。
    """
    issues: list[ValidationIssue] = []

    for table_name, (min_d, max_d) in table_ranges.items():
        if max_d is None:
            issues.append(
                ValidationIssue(
                    stock_id="*",
                    check_name="data_freshness",
                    severity="warning",
                    description=f"{table_name} 無資料",
                    details={"table": table_name},
                )
            )
            continue

        bdays = int(np.busday_count(max_d, reference_date))
        if bdays > stale_threshold:
            issues.append(
                ValidationIssue(
                    stock_id="*",
                    check_name="data_freshness",
                    severity="warning",
                    description=(
                        f"{table_name} 資料過期：最新日期 {max_d}（距今 {bdays} 個營業日，門檻 {stale_threshold}）"
                    ),
                    details={
                        "table": table_name,
                        "max_date": str(max_d),
                        "business_days_behind": bdays,
                    },
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_validation(
    stock_ids: list[str] | None = None,
    gap_threshold: int = 5,
    streak_threshold: int = 5,
    check_freshness: bool = True,
) -> ValidationReport:
    """從 DB 讀取資料，執行全部品質檢查，彙整報告。"""
    from sqlalchemy import func, select

    from src.data.database import get_session, init_db
    from src.data.schema import (
        DailyPrice,
        InstitutionalInvestor,
        MarginTrading,
        MonthlyRevenue,
        TechnicalIndicator,
    )

    init_db()

    all_issues: list[ValidationIssue] = []
    table_coverages: list[TableCoverage] = []

    tables = [
        (DailyPrice, "daily_price"),
        (InstitutionalInvestor, "institutional_investor"),
        (MarginTrading, "margin_trading"),
        (MonthlyRevenue, "monthly_revenue"),
        (TechnicalIndicator, "technical_indicator"),
    ]

    # ---------- 表層概覽 + 收集各表日期範圍 ----------
    table_overall_ranges: dict[str, tuple[date | None, date | None]] = {}
    table_stock_ranges: dict[str, dict[str, tuple[date | None, date | None]]] = {}

    with get_session() as session:
        for model, table_name in tables:
            total = session.execute(select(func.count()).select_from(model)).scalar() or 0
            stocks_count = session.execute(select(func.count(func.distinct(model.stock_id)))).scalar() or 0
            min_d = session.execute(select(func.min(model.date))).scalar()
            max_d = session.execute(select(func.max(model.date))).scalar()

            table_coverages.append(
                TableCoverage(
                    table_name=table_name,
                    stock_count=stocks_count,
                    row_count=total,
                    min_date=min_d,
                    max_date=max_d,
                )
            )
            table_overall_ranges[table_name] = (min_d, max_d)

            # 收集每股在各表的日期範圍
            per_stock_ranges = session.execute(
                select(model.stock_id, func.min(model.date), func.max(model.date)).group_by(model.stock_id)
            ).all()
            table_stock_ranges[table_name] = {row[0]: (row[1], row[2]) for row in per_stock_ranges}

        # ---------- 決定要檢查的股票 ----------
        if stock_ids is None:
            stock_ids_to_check = list(session.execute(select(func.distinct(DailyPrice.stock_id))).scalars().all())
        else:
            stock_ids_to_check = stock_ids

        # ---------- 逐股檢查 ----------
        for sid in stock_ids_to_check:
            # 讀取日K線
            rows = session.execute(
                select(
                    DailyPrice.date,
                    DailyPrice.open,
                    DailyPrice.high,
                    DailyPrice.low,
                    DailyPrice.close,
                    DailyPrice.volume,
                )
                .where(DailyPrice.stock_id == sid)
                .order_by(DailyPrice.date)
            ).all()

            if not rows:
                continue

            dates = [r[0] for r in rows]
            df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            # 1. 缺漏交易日
            all_issues.extend(check_missing_days(sid, dates, gap_threshold=gap_threshold))

            # 2. 零成交量
            all_issues.extend(check_zero_volume(sid, df))

            # 3. 連續漲跌停
            all_issues.extend(check_limit_streaks(sid, df, streak_threshold=streak_threshold))

            # 4. 價格異常
            all_issues.extend(check_price_anomalies(sid, df))

    # ---------- 跨表檢查 ----------
    # 5. 日期範圍一致性
    all_issues.extend(check_date_range_consistency(table_stock_ranges))

    # 6. 資料新鮮度
    if check_freshness:
        from datetime import date as date_type

        today = date_type.today()
        all_issues.extend(check_data_freshness(table_overall_ranges, today))

    # ---------- 彙整 ----------
    summary: dict[str, int] = {}
    for issue in all_issues:
        summary[issue.check_name] = summary.get(issue.check_name, 0) + 1

    return ValidationReport(
        checked_stocks=len(stock_ids_to_check),
        total_issues=len(all_issues),
        issues=all_issues,
        summary=summary,
        table_coverage=table_coverages,
    )


# ---------------------------------------------------------------------------
# Console 報告輸出
# ---------------------------------------------------------------------------


def print_validation_report(report: ValidationReport) -> None:
    """將 ValidationReport 以友善格式輸出至 console。"""
    print("\n" + "=" * 65)
    print("資料品質檢查報告")
    print("=" * 65)

    # 資料表概覽
    print("\n【資料表概覽】")
    print(f"  {'資料表':<25} {'筆數':>10}  {'股票數':>6}  {'日期範圍'}")
    print(f"  {'─' * 60}")
    for tc in report.table_coverage:
        date_range = f"{tc.min_date} ~ {tc.max_date}" if tc.min_date else "（無資料）"
        print(f"  {tc.table_name:<25} {tc.row_count:>10,}  {tc.stock_count:>6}  {date_range}")

    # 檢查摘要
    print(f"\n【檢查摘要】已檢查 {report.checked_stocks} 支股票，發現 {report.total_issues} 個問題")
    if report.summary:
        check_labels = {
            "missing_days": "缺漏交易日",
            "zero_volume": "零成交量",
            "limit_streak": "連續漲跌停",
            "price_anomaly": "價格異常",
            "date_range_consistency": "日期範圍不一致",
            "data_freshness": "資料過期",
        }
        for check_name, count in sorted(report.summary.items()):
            label = check_labels.get(check_name, check_name)
            print(f"  {label}: {count} 個")

    if not report.issues:
        print("\n  所有檢查通過，無異常發現。")
        return

    # 按 severity 分組顯示
    errors = [i for i in report.issues if i.severity == "error"]
    warnings = [i for i in report.issues if i.severity == "warning"]

    if errors:
        print(f"\n【錯誤】（{len(errors)} 個）")
        for issue in errors:
            print(f"  [ERROR] {issue.stock_id:>6}  {issue.description}")

    if warnings:
        print(f"\n【警告】（{len(warnings)} 個）")
        for issue in warnings:
            print(f"  [WARN]  {issue.stock_id:>6}  {issue.description}")


def export_issues_csv(report: ValidationReport, path: str) -> None:
    """將問題清單匯出為 CSV。"""
    if not report.issues:
        print("無問題可匯出")
        return

    rows = []
    for issue in report.issues:
        rows.append(
            {
                "stock_id": issue.stock_id,
                "check_name": issue.check_name,
                "severity": issue.severity,
                "description": issue.description,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n問題清單已匯出至: {path}")
