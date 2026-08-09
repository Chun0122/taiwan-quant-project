"""PIT 歷史重放與前瞻報酬評估（B1④，2026-08-04）。

## 這是 B1 兌現價值的地方

在此之前，改動 scanner 之後唯一的驗證途徑是 forward test 等數週——每一個
「這個因子有沒有用」的問題都要付數週的等待成本，而等到有答案時市場情境
往往已經改變。有了 PIT 重放，改完當天就能問：**「如果這版 scanner 在
2025-04-07 關稅崩盤那天執行，它會選出什麼？之後 5/10/20 天表現如何？」**

## 正確性依賴（缺一不可，全部已在 B1① ②③ 落地）

1. **價量歷史**（B1①）：全市場逐日回補，且含當時在市、如今已下市的股票，
   否則重放只會看到「活下來的贏家」。
2. **DailyFeature 歷史**（B1②）：universe 的流動性/趨勢過濾需要當日特徵。
3. **時間注入 + 公布時滯**（B1③）：`run(as_of=...)` 使所有查詢帶上界，
   基本面另套法定申報期限；`as_of` 為歷史日時自動 offline 禁止外部 API。
4. **regime PIT 化**：`detect(as_of=...)` 且**唯讀不推進狀態機**。

## 成本

單次重放約 90 秒（全市場 universe 過濾 + 四維評分）。範圍重放請用
`every_n_days` 抽樣——逐日重放 2.6 年需約 15 小時，抽樣到每月則約 45 分鐘。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import func, select

from src.constants import (
    BACKFILL_MIN_COMMON_STOCKS,
    REPLAY_MIN_FEATURE_RATIO,
    REPLAY_MIN_FEATURE_WARM_RATIO,
    REPLAY_MIN_REVENUE_STOCKS,
    VALUATION_FRESH_WINDOW_DAYS,
    VALUATION_MIN_FRESH_STOCKS,
)
from src.data.database import get_session
from src.data.pit import revenue_visible_cutoff
from src.data.schema import DailyFeature, DailyPrice, MonthlyRevenue, StockValuation

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)

# 各模式**定義性**（粗篩閘門）依賴的資料表——缺席時該模式必然產出 0 筆，
# 且那個 0 是「量不到」而非「模式判斷不進場」。
#
# 只列粗篩閘門的依賴，不列評分用的表：`financial_statement` 雖然參與品質評分
# （`_base.py` 載入 EPS/ROE），但它缺席只會使該維度分數退化，**不會**把選股歸零，
# 因此不屬於「結果是否可採信」的判定條件。列進來會把可評價的重放誤判為無效。
MODE_REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "momentum": ("daily_price", "daily_feature"),
    "swing": ("daily_price", "daily_feature"),
    "value": ("daily_price", "daily_feature", "stock_valuation"),
    "dividend": ("daily_price", "daily_feature", "stock_valuation"),
    "growth": ("daily_price", "daily_feature", "monthly_revenue"),
}


@dataclass(frozen=True)
class DataCoverage:
    """`as_of` 當日各輸入資料表的覆蓋度（§6.5 #21b）。

    ## 為什麼需要這個

    `n_picks == 0` 有兩種**完全不同**的意義：模式看過全市場後判斷不進場（是結論），
    或模式的定義性輸入根本不存在（結果無效、不可採信）。2026-08-04 的跨模式重放
    無法區分兩者，導致 dividend「30 天只選得出 4 天」被當成模式特性記錄下來，
    實際上那是 `stock_valuation` 在 2026-01-26 前完全沒有資料。

    本結構在重放當下就把判準記錄下來，使結果自帶可採信與否的標記。
    """

    as_of: date
    mode: str
    price_stocks: int
    feature_stocks: int
    valuation_stocks: int
    revenue_stocks: int
    # §6.5 #21d：列數足夠**不代表**欄位可用。MA60 需 60 個交易日才填滿，回補範圍
    # 頭幾十天欄位全是 NaN 而列數檢查完全看不出來。取 ma60 與 turnover_ma20 非空率
    # 的較小值（ma60 天生較低，是 binding constraint）。
    feature_warm_ratio: float = 1.0
    required: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def sufficient(self) -> bool:
        """本模式的定義性資料是否全部就緒——False 時該日重放結果不可採信。"""
        return not self.missing

    def describe(self) -> str:
        """一行摘要，供 CLI 與報告輸出。

        暖身失效與整表缺席分開講——兩者都讓結果不可採信，但補救方式完全不同：
        前者要往前多回補幾十個交易日，後者要補整張表。
        """
        if self.sufficient:
            return "資料就緒"
        if "daily_feature" in self.missing and self.feature_warm_ratio < REPLAY_MIN_FEATURE_WARM_RATIO:
            return f"特徵未暖身（ma60/turnover_ma20 非空率 {self.feature_warm_ratio:.1%}）"
        return "資料缺席：" + ", ".join(self.missing)


def assess_data_coverage(mode: str, as_of: date) -> DataCoverage:
    """量測 `as_of` 當日各資料表的覆蓋度，判定本模式的重放是否可採信。

    **全部查詢都帶 PIT 上界**——覆蓋率本身若看到未來資料，就會把「當時資料還沒補」
    的日子誤判為就緒。估值另取近 `VALUATION_FRESH_WINDOW_DAYS` 日窗口、月營收另套
    法定公布時滯，與 scanner 實際取數的條件一致（否則量到的不是 scanner 看到的）。

    門檻沿用既有 SSOT（見 `src/constants.py` 的 §6.5 #21b 段落），不另立數字。

    Args:
        mode: momentum / swing / value / dividend / growth。
        as_of: 重放基準日。

    Returns:
        DataCoverage；未知模式的 `required` 為空（即恆視為就緒）。
    """
    val_cutoff = as_of - timedelta(days=VALUATION_FRESH_WINDOW_DAYS)
    # 與 `_load_revenue_data` 完全相同的窗口：近 180 日內、且已依法公布者
    rev_lower = as_of - timedelta(days=180)
    rev_upper = revenue_visible_cutoff(as_of)

    with get_session() as session:
        price_stocks = (
            session.execute(
                select(func.count(func.distinct(DailyPrice.stock_id))).where(
                    DailyPrice.date == as_of,
                    func.length(DailyPrice.stock_id) == 4,
                )
            ).scalar()
            or 0
        )
        feature_stocks = (
            session.execute(
                select(func.count(func.distinct(DailyFeature.stock_id))).where(
                    DailyFeature.date == as_of,
                    func.length(DailyFeature.stock_id) == 4,
                )
            ).scalar()
            or 0
        )
        valuation_stocks = (
            session.execute(
                select(func.count(func.distinct(StockValuation.stock_id))).where(
                    StockValuation.date >= val_cutoff,
                    StockValuation.date <= as_of,
                )
            ).scalar()
            or 0
        )
        revenue_stocks = (
            session.execute(
                select(func.count(func.distinct(MonthlyRevenue.stock_id))).where(
                    MonthlyRevenue.date >= rev_lower,
                    MonthlyRevenue.date <= rev_upper,
                )
            ).scalar()
            or 0
        )
        # §6.5 #21d 暖身檢查：量測**欄位**而非列數。這兩欄各自把守一道閘門——
        # `ma60` 是 universe Stage 3 的趨勢過濾，`turnover_ma20` 是 Stage 2 的流動性
        # 門檻，且後者為 NaN 時 `universe.py:125` 會**跳過該股的門檻**（per-stock
        # fail-open），暖身期等於流動性過濾整段消失。
        warm = session.execute(
            select(
                func.count(),
                func.count(DailyFeature.ma60),
                func.count(DailyFeature.turnover_ma20),
            ).where(
                DailyFeature.date == as_of,
                func.length(DailyFeature.stock_id) == 4,
            )
        ).one()
        total_rows = warm[0] or 0
        feature_warm_ratio = min(warm[1] or 0, warm[2] or 0) / total_rows if total_rows else 0.0

    required = MODE_REQUIRED_TABLES.get(mode, ())
    # 特徵覆蓋率以當日價量檔數為分母——絕對值門檻會在早期市場（上市檔數本來就少）誤判
    feature_rows_ok = price_stocks > 0 and feature_stocks >= price_stocks * REPLAY_MIN_FEATURE_RATIO
    feature_warm_ok = feature_warm_ratio >= REPLAY_MIN_FEATURE_WARM_RATIO
    ok_by_table = {
        "daily_price": price_stocks >= BACKFILL_MIN_COMMON_STOCKS,
        # 列數與欄位**都要**過關：列數足夠但欄位全 NaN 是回補範圍起點的常態
        "daily_feature": feature_rows_ok and feature_warm_ok,
        "stock_valuation": valuation_stocks >= VALUATION_MIN_FRESH_STOCKS,
        "monthly_revenue": revenue_stocks >= REPLAY_MIN_REVENUE_STOCKS,
    }
    missing = tuple(t for t in required if not ok_by_table.get(t, True))

    if missing:
        reason = ""
        if "daily_feature" in missing and feature_rows_ok and not feature_warm_ok:
            reason = f"（特徵列數足夠但欄位未暖身：ma60/turnover_ma20 非空率 {feature_warm_ratio:.1%}）"
        logger.warning(
            "[%s] %s 資料覆蓋不足：%s%s — 本日重放結果不可採信（價量 %d／特徵 %d／估值 %d／營收 %d）",
            mode,
            as_of,
            ", ".join(missing),
            reason,
            price_stocks,
            feature_stocks,
            valuation_stocks,
            revenue_stocks,
        )

    return DataCoverage(
        as_of=as_of,
        mode=mode,
        price_stocks=price_stocks,
        feature_stocks=feature_stocks,
        valuation_stocks=valuation_stocks,
        revenue_stocks=revenue_stocks,
        feature_warm_ratio=feature_warm_ratio,
        required=required,
        missing=missing,
    )


@dataclass
class ReplayResult:
    """單一 as_of 的重放結果。"""

    as_of: date
    mode: str
    regime: str
    total_stocks: int
    after_coarse: int
    picks: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: DataCoverage | None = None

    @property
    def n_picks(self) -> int:
        return len(self.picks)

    @property
    def verdict(self) -> str:
        """本日重放的可採信狀態——彙總時**必須**先看這個再看報酬。

        - `no_data`：定義性輸入缺席，`n_picks` 無論是幾都不可採信
        - `no_picks`：資料就緒，模式判斷不進場（這是結論，可計入產能率）
        - `ok`：資料就緒且有選股
        """
        if self.coverage is not None and not self.coverage.sufficient:
            return "no_data"
        return "ok" if self.n_picks else "no_picks"


def replay_scan(mode: str, as_of: date, top_n: int = 20) -> ReplayResult:
    """在 `as_of` 當日重放一次 scanner。

    使用 `MarketScanner.run(as_of=...)`，因此自動享有全部 PIT 保護
    （查詢上界、公布時滯、offline、regime 唯讀）。**不寫入任何 live 資料表**。

    Args:
        mode: momentum / swing / value / dividend / growth。
        as_of: 重放基準日。
        top_n: 取前 N 名。

    Returns:
        ReplayResult；模式被 regime 封鎖或無候選時 `picks` 為空——此時務必看
        `verdict` 區分「模式不進場」與「輸入資料缺席」。
    """
    from src.discovery.scanner import (
        DividendScanner,
        GrowthScanner,
        MomentumScanner,
        SwingScanner,
        ValueScanner,
    )

    scanner_map = {
        "momentum": MomentumScanner,
        "swing": SwingScanner,
        "value": ValueScanner,
        "dividend": DividendScanner,
        "growth": GrowthScanner,
    }
    if mode not in scanner_map:
        raise ValueError(f"未知模式 {mode}；可用：{', '.join(scanner_map)}")

    # 覆蓋度在**跑之前**量測：跑完才量的話，scanner 的 Stage 0.5 補抓可能已改變資料表，
    # 量到的就不是它當時看到的東西（歷史 as_of 雖為 offline，但 today 重放不是）
    coverage = assess_data_coverage(mode, as_of)

    scanner = scanner_map[mode](top_n_results=top_n, use_ic_adjustment=False)
    result = scanner.run(as_of=as_of)

    picks = result.rankings.copy() if result.rankings is not None else pd.DataFrame()
    if not picks.empty:
        keep = [c for c in ("rank", "stock_id", "stock_name", "close", "composite_score") if c in picks.columns]
        picks = picks[keep].head(top_n)

    return ReplayResult(
        as_of=as_of,
        mode=mode,
        regime=getattr(scanner, "regime", "unknown"),
        total_stocks=result.total_stocks,
        after_coarse=result.after_coarse,
        picks=picks,
        coverage=coverage,
    )


def compute_forward_returns(
    picks: pd.DataFrame,
    as_of: date,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """計算選股清單的前瞻報酬（%）。

    以 `as_of` **之後**的交易日收盤價計算——這是重放中**唯一**允許看未來的地方，
    因為它就是「評分」本身，不參與選股決策。

    Args:
        picks: 含 stock_id / close 的選股清單。
        as_of: 進場基準日（以當日 close 為成本，與 DiscoveryRecord 一致）。
        horizons: 前瞻交易日數。

    Returns:
        picks 加上 `fwd_{h}d` 欄位（%）；無後續報價者為 NaN。
    """
    if picks is None or picks.empty:
        return pd.DataFrame()

    out = picks.copy()
    sids = out["stock_id"].astype(str).tolist()
    max_h = max(horizons)
    # 日曆天緩衝：交易日約為日曆日的 0.7 倍，取 2 倍再加 10 天保險
    end = as_of + timedelta(days=max_h * 2 + 10)

    with get_session() as session:
        rows = session.execute(
            select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                DailyPrice.stock_id.in_(sids),
                DailyPrice.date > as_of,
                DailyPrice.date <= end,
                DailyPrice.volume > 0,
            )
        ).all()

    if not rows:
        for h in horizons:
            out[f"fwd_{h}d"] = float("nan")
        return out

    px = pd.DataFrame(rows, columns=["stock_id", "date", "close"]).sort_values(["stock_id", "date"])
    by_sid = {sid: g["close"].tolist() for sid, g in px.groupby("stock_id")}

    for h in horizons:
        vals = []
        for _, r in out.iterrows():
            series = by_sid.get(str(r["stock_id"]), [])
            base = r.get("close")
            if len(series) >= h and base and base > 0:
                vals.append((series[h - 1] - base) / base * 100)
            else:
                vals.append(float("nan"))
        out[f"fwd_{h}d"] = vals
    return out


def sample_replay_dates(start: date, end: date, every_n_days: int) -> list[date]:
    """在 [start, end] 內取樣**有全市場資料**的交易日。

    單次重放約 90 秒，逐日重放不切實際；抽樣讓成本可控且仍能跨 regime 取樣。
    """
    from sqlalchemy import func

    from src.constants import BACKFILL_MIN_COMMON_STOCKS

    with get_session() as session:
        rows = session.execute(
            select(DailyPrice.date)
            .where(DailyPrice.date >= start, DailyPrice.date <= end, func.length(DailyPrice.stock_id) == 4)
            .group_by(DailyPrice.date)
            .having(func.count(DailyPrice.id) >= BACKFILL_MIN_COMMON_STOCKS)
            .order_by(DailyPrice.date)
        ).all()
    trading_days = [r[0] for r in rows]
    if not trading_days:
        return []
    return trading_days[:: max(1, every_n_days)]


def summarize_replays(results: list[pd.DataFrame], horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """彙總多次重放的前瞻報酬為單列摘要（每個 horizon 一列）。

    Returns:
        DataFrame(horizon, n, avg_return, median, win_rate, best, worst)
    """
    if not results:
        return pd.DataFrame(columns=["horizon", "n", "avg_return", "median", "win_rate", "best", "worst"])

    allp = pd.concat(results, ignore_index=True)
    rows = []
    for h in horizons:
        col = f"fwd_{h}d"
        if col not in allp.columns:
            continue
        s = allp[col].dropna()
        rows.append(
            {
                "horizon": f"{h}d",
                "n": len(s),
                "avg_return": s.mean() if len(s) else float("nan"),
                "median": s.median() if len(s) else float("nan"),
                "win_rate": (s > 0).mean() if len(s) else float("nan"),
                "best": s.max() if len(s) else float("nan"),
                "worst": s.min() if len(s) else float("nan"),
            }
        )
    return pd.DataFrame(rows)
