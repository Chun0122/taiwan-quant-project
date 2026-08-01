"""IC 執法治理 — 判定某個模式的關鍵因子 IC 是否「可據以行動」（P0 #16 / B11 最小切口）。

## 為什麼需要這個模組

`logs/audit_discover_20260731/REPORT.md` §2 證實：M2 自動停用的**實際判準是
「rolling IC 算不算得出來」，而不是「因子有沒有效」**。

  - `compute_rolling_ic()` 的窗口錨定在**推薦記錄的日期範圍**（`min_date + window_days`
    起、`max_date + 1` 止），而非今天；
  - `morning_cmd` 取 `factor_df["ic"].iloc[-1]`＝最後一個**有資料**的窗口。

模式一旦被停掃，`max_date` 就凍結，窗口跟著凍結，於是系統每天重讀同一份過期 IC
執法。實測 2026-07-31 停用 momentum 所用的 IC＝−0.1109，來自 `window_end=2026-06-27`、
n=40、**距今 34 天**的同一批六月觀測；7/30 與 7/31 取到的值完全相同。
反過來，當記錄跨度小於 window_days 時窗口迴圈一次都不跑 → `rolling_df` 為空 →
fail-open 放行。這就是「停用→樣本歸零→放行一天→再停用」振盪的真正機制。

同時 `docs/MASTER_PLAN.md` §3 原則 6 早已寫明「自動開關樣本 <100 或未跨 3 個獨立
掃描週時，只能告警不能行動」，但程式中沒有任何一處在執行它（實測跑在 n=40、
單一窗口、34 天前）。

## 本模組的職責

把「這個 IC 可不可以拿來行動」收斂為單一實作，供三個執法點共用：

  1. `cli/morning_cmd._compute_factor_ic_status` — M2 模式停用
  2. `discovery/scanner/_base._compute_ic_decay_adjustment` — 分數門檻 +0.05 加成
  3. `portfolio/manager._build_decision_context` — rotation 層阻擋新買入

三道閘門（任一不滿足 → `enforceable=False`，只告警不行動）：

  - **時效**：最新窗口的 `window_end` 距 `as_of` 不得超過 `holding_days + BUFFER`
  - **窗口數**：至少 `DISCOVERY_IC_MIN_WINDOWS` 個窗口
  - **樣本**：決策窗口中**最小**的 `evaluable_count` ≥ `DISCOVERY_IC_MIN_SAMPLES`

決策 IC 取最近 N 窗的平均（非單一最新窗）。注意：rolling 窗口 window_days=14、
step_days=7 → 相鄰窗口有 50% 重疊，故此平均**不是**獨立樣本的平均，僅作為降低
單窗漂移的穩健化處理；真正的獨立性要求由 B2（全候選池落庫）之後的 IC 重建處理。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.constants import (
    DISCOVERY_IC_HOLDING_DAYS_MAP,
    DISCOVERY_IC_MIN_SAMPLES,
    DISCOVERY_IC_MIN_WINDOWS,
    DISCOVERY_IC_WINDOW_STALE_BUFFER_DAYS,
    DISCOVERY_KEY_FACTOR_MAP,
)

logger = logging.getLogger(__name__)

# IC 低於此值視為反向（沿用既有 M2 門檻，本次不調整）
IC_INVERSE_THRESHOLD: float = -0.05

# enforceable=False 的原因碼
REASON_OK = "ok"
REASON_NO_WINDOWS = "no_windows"
REASON_INSUFFICIENT_WINDOWS = "insufficient_windows"
REASON_STALE_WINDOW = "stale_window"
REASON_INSUFFICIENT_SAMPLES = "insufficient_samples"

_REASON_LABELS: dict[str, str] = {
    REASON_NO_WINDOWS: "無可用窗口（記錄跨度不足或無 forward 報酬）",
    REASON_INSUFFICIENT_WINDOWS: "窗口數不足",
    REASON_STALE_WINDOW: "最新窗口已過期",
    REASON_INSUFFICIENT_SAMPLES: "樣本不足",
}


@dataclass(frozen=True)
class ICVerdict:
    """某模式關鍵因子的 IC 執法裁定。

    Attributes:
        mode: 模式名稱。
        factor: 關鍵因子欄位名（如 chip_score）。
        enforceable: 是否可據以行動（停用模式 / 提高門檻 / 阻擋買入）。
        reason: 不可執法的原因碼；可執法時為 REASON_OK。
        ic: 決策用 IC（最近 N 窗平均）；無窗口時為 None。**即使不可執法也會填**，
            供告警顯示——「觀測得到」與「可據以行動」是兩件事。
        latest_window_ic: 最新單一窗口的 IC（診斷用，舊版即以此執法）。
        window_end: 最新窗口結束日。
        staleness_days: `as_of - window_end` 的天數。
        staleness_budget_days: 本模式允許的最大過期天數（holding_days + BUFFER）。
        window_count: 可用窗口總數。
        min_sample_count: 決策窗口中最小的 evaluable_count。
        holding_days: 本模式的 forward holding days。
    """

    mode: str
    factor: str
    enforceable: bool
    reason: str
    ic: float | None = None
    latest_window_ic: float | None = None
    window_end: date | None = None
    staleness_days: int | None = None
    staleness_budget_days: int | None = None
    window_count: int = 0
    min_sample_count: int | None = None
    holding_days: int | None = None

    @property
    def is_inverse(self) -> bool:
        """是否判定為反向因子（**且**可執法）。

        不可執法時一律回 False——這正是 P0 #16 的核心：過期/小樣本的 IC
        可以告警，但不得停用模式或阻擋買入。
        """
        return self.enforceable and self.ic is not None and self.ic < IC_INVERSE_THRESHOLD

    def describe(self) -> str:
        """人類可讀的一行說明（供 stdout / Discord / log）。"""
        if self.ic is None:
            return f"{self.factor} 無 IC（{_REASON_LABELS.get(self.reason, self.reason)}）"
        head = f"{self.factor} IC={self.ic:+.4f}"
        if self.enforceable:
            return f"{head}（n≥{self.min_sample_count}, {self.window_count} 窗）"
        detail = _REASON_LABELS.get(self.reason, self.reason)
        if self.reason == REASON_STALE_WINDOW:
            detail = f"{detail}（{self.window_end}，距今 {self.staleness_days} 天 > 上限 {self.staleness_budget_days}）"
        elif self.reason == REASON_INSUFFICIENT_SAMPLES:
            detail = f"{detail}（n={self.min_sample_count} < {DISCOVERY_IC_MIN_SAMPLES}）"
        elif self.reason == REASON_INSUFFICIENT_WINDOWS:
            detail = f"{detail}（{self.window_count} < {DISCOVERY_IC_MIN_WINDOWS}）"
        return f"{head} — 僅告警不執法：{detail}"


def staleness_budget_days(holding_days: int) -> int:
    """該 holding 對應的窗口時效預算（天）。

    rolling 窗口需要記錄日之後 holding_days 的 forward 報酬才可評估，故最新
    「有資料」的窗口天生落後約 holding_days；再加 step_days(7) + 假日寬限(7)。
    """
    return holding_days + DISCOVERY_IC_WINDOW_STALE_BUFFER_DAYS


def select_enforceable_ic(
    rolling_df: pd.DataFrame | None,
    factor: str,
    *,
    as_of: date,
    holding_days: int,
    mode: str = "",
    min_windows: int = DISCOVERY_IC_MIN_WINDOWS,
    min_samples: int = DISCOVERY_IC_MIN_SAMPLES,
) -> ICVerdict:
    """從 rolling IC 序列判定是否可據以行動（純函數）。

    Args:
        rolling_df: `compute_rolling_ic()` 輸出（window_end / factor / ic / sample_count）。
        factor: 關鍵因子欄位名。
        as_of: 判定基準日（計算窗口過期天數用）。
        holding_days: 本模式 forward holding days，決定時效預算。
        mode: 模式名（僅用於回傳與日誌）。
        min_windows: 最低窗口數。
        min_samples: 決策窗口的最低 evaluable_count（取最小值比較）。

    Returns:
        ICVerdict。不可執法時 `enforceable=False` 且 `reason` 說明原因，
        但 `ic` / `latest_window_ic` 仍盡量填入供告警。
    """
    budget = staleness_budget_days(holding_days)
    base = {
        "mode": mode,
        "factor": factor,
        "holding_days": holding_days,
        "staleness_budget_days": budget,
    }

    if rolling_df is None or rolling_df.empty or "factor" not in rolling_df.columns:
        return ICVerdict(enforceable=False, reason=REASON_NO_WINDOWS, **base)

    f = rolling_df[rolling_df["factor"] == factor].sort_values("window_end")
    if f.empty:
        return ICVerdict(enforceable=False, reason=REASON_NO_WINDOWS, **base)

    latest = f.iloc[-1]
    window_end = latest["window_end"]
    if isinstance(window_end, pd.Timestamp):
        window_end = window_end.date()
    staleness = (as_of - window_end).days if isinstance(window_end, date) else None
    latest_ic = float(latest["ic"])

    decision = f.tail(min_windows)
    decision_ic = float(decision["ic"].mean())
    min_n = int(decision["sample_count"].min()) if "sample_count" in decision.columns else None

    observed = {
        **base,
        "ic": decision_ic,
        "latest_window_ic": latest_ic,
        "window_end": window_end if isinstance(window_end, date) else None,
        "staleness_days": staleness,
        "window_count": len(f),
        "min_sample_count": min_n,
    }

    # 閘門 1：時效——凍結的舊窗口不得執法（P0 #16 的核心）
    # 先於窗口數檢查：模式停掃後兩者常同時成立，但「窗口已過期 N 天」才是根因，
    # 對讀 log 的人也更可行動（要恢復掃描，不是等窗口變多）。
    if staleness is None or staleness > budget:
        return ICVerdict(enforceable=False, reason=REASON_STALE_WINDOW, **observed)

    # 閘門 2：窗口數（≈ §3 原則 6 的「跨 3 個獨立掃描週」）
    if len(f) < min_windows:
        return ICVerdict(enforceable=False, reason=REASON_INSUFFICIENT_WINDOWS, **observed)

    # 閘門 3：樣本數
    if min_n is None or min_n < min_samples:
        return ICVerdict(enforceable=False, reason=REASON_INSUFFICIENT_SAMPLES, **observed)

    return ICVerdict(enforceable=True, reason=REASON_OK, **observed)


def compute_mode_ic_verdict(session, mode: str, as_of: date) -> ICVerdict:
    """查 DB 計算某模式的 IC 裁定（薄 IO 層，邏輯全在 select_enforceable_ic）。

    與 `morning_cmd._compute_factor_ic_status` 使用相同的取數與 rolling 參數，
    確保 M2 停用、scanner 門檻加成、rotation 阻擋三處判定一致。

    Args:
        session: SQLAlchemy Session。
        mode: 模式名（composite 模式請由 caller 先展開為成員）。
        as_of: 判定基準日。

    Returns:
        ICVerdict；任何例外或無資料時回傳 `enforceable=False`（fail-open）。
    """
    from sqlalchemy import select

    from src.data.schema import DailyPrice, DiscoveryRecord
    from src.discovery.scanner._functions import compute_rolling_ic

    factor = DISCOVERY_KEY_FACTOR_MAP.get(mode)
    holding_days = DISCOVERY_IC_HOLDING_DAYS_MAP.get(mode, 5)
    if not factor:
        return ICVerdict(
            mode=mode,
            factor="",
            enforceable=False,
            reason=REASON_NO_WINDOWS,
            holding_days=holding_days,
            staleness_budget_days=staleness_budget_days(holding_days),
        )

    max_holding = max(DISCOVERY_IC_HOLDING_DAYS_MAP.values(), default=5)
    cutoff = as_of - timedelta(days=35 + max_holding)
    cols = [
        "scan_date",
        "stock_id",
        "close",
        "technical_score",
        "chip_score",
        "fundamental_score",
        "news_score",
    ]
    try:
        rows = session.execute(
            select(
                DiscoveryRecord.scan_date,
                DiscoveryRecord.stock_id,
                DiscoveryRecord.close,
                DiscoveryRecord.technical_score,
                DiscoveryRecord.chip_score,
                DiscoveryRecord.fundamental_score,
                DiscoveryRecord.news_score,
            ).where(DiscoveryRecord.mode == mode, DiscoveryRecord.scan_date >= cutoff)
        ).all()
        df_records = pd.DataFrame(rows, columns=cols)
        if df_records.empty:
            return select_enforceable_ic(None, factor, as_of=as_of, holding_days=holding_days, mode=mode)

        stock_ids = df_records["stock_id"].unique().tolist()
        price_rows = session.execute(
            select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                DailyPrice.stock_id.in_(stock_ids), DailyPrice.date >= cutoff
            )
        ).all()
        df_prices = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])
        rolling = compute_rolling_ic(df_records, df_prices, holding_days=holding_days, window_days=14, step_days=7)
    except Exception:
        logger.warning("IC 裁定計算失敗 mode=%s — 視為不可執法（fail-open）", mode, exc_info=True)
        return ICVerdict(
            mode=mode,
            factor=factor,
            enforceable=False,
            reason=REASON_NO_WINDOWS,
            holding_days=holding_days,
            staleness_budget_days=staleness_budget_days(holding_days),
        )

    return select_enforceable_ic(rolling, factor, as_of=as_of, holding_days=holding_days, mode=mode)


def load_candidate_factor_records(
    session,
    mode: str,
    since: date,
    *,
    selected_only: bool = False,
) -> pd.DataFrame:
    """讀取 `CandidateFactorLog` 為 IC 計算可直接使用的形狀（B2）。

    回傳欄位與 `compute_factor_ic()` / `compute_rolling_ic()` 期望的
    `df_records` 相同（scan_date / stock_id / close / 各維度 score），因此 B11
    要把 IC 改算在全候選池上時，只需把取數來源由 `DiscoveryRecord` 換成本函數。

    為什麼這是必要的（`logs/audit_discover_20260731/REPORT.md` §7.3）：
    `DiscoveryRecord` 只有最終 top-20，在其上算 IC 屬 range restriction——
    被評分的 ~150 檔只留分數最高的 20 檔，因子變異被人為壓縮。

    ⚠ 本表自 2026-08-01 起累積，**無法回補**（歷史候選池從未落庫）。因此
    B11 切換取數來源前，需先累積足夠掃描日。

    Args:
        session: SQLAlchemy Session。
        mode: scanner 模式。
        since: 起始 scan_date（含）。
        selected_only: True 時只取進入 top-N 者——用於與舊 `DiscoveryRecord`
            路徑做對照，量化截斷偏差的大小。

    Returns:
        DataFrame；無資料時為具正確欄位的空表。
    """
    from sqlalchemy import select

    from src.data.schema import CandidateFactorLog

    cols = [
        "scan_date",
        "stock_id",
        "close",
        "technical_score",
        "chip_score",
        "fundamental_score",
        "news_score",
        "valuation_score",
        "dividend_score",
        "composite_score",
        "selected",
    ]
    stmt = select(
        CandidateFactorLog.scan_date,
        CandidateFactorLog.stock_id,
        CandidateFactorLog.close,
        CandidateFactorLog.technical_score,
        CandidateFactorLog.chip_score,
        CandidateFactorLog.fundamental_score,
        CandidateFactorLog.news_score,
        CandidateFactorLog.valuation_score,
        CandidateFactorLog.dividend_score,
        CandidateFactorLog.composite_score,
        CandidateFactorLog.selected,
    ).where(CandidateFactorLog.mode == mode, CandidateFactorLog.scan_date >= since)
    if selected_only:
        stmt = stmt.where(CandidateFactorLog.selected.is_(True))

    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def blocked_modes(session, modes: list[str], as_of: date) -> dict[str, ICVerdict]:
    """回傳 `modes` 中「IC 反向且可執法」者的裁定（供 rotation 阻擋新買入）。

    只回傳實際被擋的模式；未被擋者不入表，caller 以 `mode in result` 判斷即可。
    """
    out: dict[str, ICVerdict] = {}
    for m in modes:
        verdict = compute_mode_ic_verdict(session, m, as_of)
        if verdict.is_inverse:
            out[m] = verdict
    return out
