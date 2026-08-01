"""Point-in-Time 資料可見性規則（B1，2026-08-01）。

## 問題

`MonthlyRevenue.date` 與 `FinancialStatement.date` 存的是**期間**（營收月份 /
季末日），**不是公布日**——schema 中沒有任何公布日欄位。因此 PIT 重放若只用
`date <= as_of` 過濾，會系統性地看到「當時還沒公布」的基本面資料：

    在 2026-03-05 重放時，`MonthlyRevenue.date = 2026-02-28`（2 月營收）符合
    `<= as_of`，但該筆營收依法要到 **2026-03-10** 才公布。

這是最典型的 look-ahead bias，而且方向一致（永遠偏樂觀），會讓任何歷史驗證
系統性高估策略績效——B1 的整個目的（讓 scanner 改動能被歷史檢驗）就此落空。

## 解法

台灣有**法定申報期限**，因此不需要公布日欄位也能保守建模：只要假設資料在
「法定最後期限」才可見，就不可能樂觀。實際公布通常更早，故本模組偏保守
（可能低估當時可得資訊），這在研究上是安全的方向。

法源：證券交易法 §36。

| 資料 | 申報期限 | 本模組常數 |
|------|---------|-----------|
| 月營收 | 每月 10 日前公布上月 | `REVENUE_PUBLISH_DAY = 10` |
| Q1 / Q2 / Q3 財報 | 各季終了後 45 日 | `QUARTERLY_PUBLISH_LAG_DAYS = 45` |
| 年報（Q4） | 會計年度終了後 3 個月（次年 3/31） | `ANNUAL_PUBLISH_LAG_DAYS = 90` |

⚠ 這些是**保守近似**，非逐筆真實公布日。要做到逐筆精確，需為
`MonthlyRevenue` / `FinancialStatement` 補公布日欄位並回填（列為 B1 後續）。
"""

from __future__ import annotations

from datetime import date, timedelta

# ── 法定申報期限（證交法 §36）────────────────────────────────────────────
REVENUE_PUBLISH_DAY: int = 10  # 每月 10 日前公布「上月」營收
QUARTERLY_PUBLISH_LAG_DAYS: int = 45  # Q1/Q2/Q3 財報：季終了後 45 日內
ANNUAL_PUBLISH_LAG_DAYS: int = 90  # 年報（Q4）：會計年度終了後 3 個月


def latest_visible_revenue_month(as_of: date) -> tuple[int, int]:
    """回傳 `as_of` 當下**已公布**的最新營收月份 `(year, month)`。

    規則：每月 10 日前公布上月營收。故：
      - as_of = 2026-03-05 → 尚未到 3/10 → 最新可見為 **1 月**營收
      - as_of = 2026-03-10 → 已到期限 → 最新可見為 **2 月**營收

    Args:
        as_of: 判定基準日。

    Returns:
        (year, month) tuple。
    """
    y, m = as_of.year, as_of.month
    # 先退一個月＝「上月」；若當月尚未到公布日，再退一個月
    lag = 1 if as_of.day >= REVENUE_PUBLISH_DAY else 2
    for _ in range(lag):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return y, m


def revenue_visible_cutoff(as_of: date) -> date:
    """回傳可用於 `MonthlyRevenue.date <= X` 的上界日期。

    `MonthlyRevenue.date` 為營收月份的代表日，故取「最新可見月份」的月底。
    """
    y, m = latest_visible_revenue_month(as_of)
    # 該月最後一天 = 次月 1 日往前一天
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return date(ny, nm, 1) - timedelta(days=1)


def quarter_publish_deadline(year: int, quarter: int) -> date:
    """回傳某年某季財報的**法定公布期限**。

    Q1/Q2/Q3：季終了後 45 日；Q4（年報）：會計年度終了後 3 個月（次年 3/31）。
    """
    if quarter == 4:
        return date(year, 12, 31) + timedelta(days=ANNUAL_PUBLISH_LAG_DAYS)
    quarter_end = {1: date(year, 3, 31), 2: date(year, 6, 30), 3: date(year, 9, 30)}[quarter]
    return quarter_end + timedelta(days=QUARTERLY_PUBLISH_LAG_DAYS)


def financial_visible_cutoff(as_of: date) -> date:
    """回傳可用於 `FinancialStatement.date <= X` 的上界日期。

    掃描近幾季，取「公布期限 <= as_of」中最新的一季，回傳其季末日。
    無任何一季已公布時（極早期日期）回傳 `as_of` 往前 1 年，等同查無資料。
    """
    candidates: list[date] = []
    for year in (as_of.year, as_of.year - 1, as_of.year - 2):
        for q in (1, 2, 3, 4):
            if quarter_publish_deadline(year, q) <= as_of:
                candidates.append(
                    {1: date(year, 3, 31), 2: date(year, 6, 30), 3: date(year, 9, 30), 4: date(year, 12, 31)}[q]
                )
    if not candidates:
        return as_of - timedelta(days=365)
    return max(candidates)


def is_pit_replay(as_of: date | None, today: date | None = None) -> bool:
    """`as_of` 是否為歷史重放（而非今日掃描）。

    用於決定是否進入 offline mode——歷史重放時**不得**呼叫外部 API，
    否則會把「今天」的資料寫進歷史情境。
    """
    if as_of is None:
        return False
    return as_of < (today or date.today())
