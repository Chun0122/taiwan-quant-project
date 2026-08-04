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
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import DailyPrice

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)


@dataclass
class ReplayResult:
    """單一 as_of 的重放結果。"""

    as_of: date
    mode: str
    regime: str
    total_stocks: int
    after_coarse: int
    picks: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_picks(self) -> int:
        return len(self.picks)


def replay_scan(mode: str, as_of: date, top_n: int = 20) -> ReplayResult:
    """在 `as_of` 當日重放一次 scanner。

    使用 `MarketScanner.run(as_of=...)`，因此自動享有全部 PIT 保護
    （查詢上界、公布時滯、offline、regime 唯讀）。**不寫入任何 live 資料表**。

    Args:
        mode: momentum / swing / value / dividend / growth。
        as_of: 重放基準日。
        top_n: 取前 N 名。

    Returns:
        ReplayResult；模式被 regime 封鎖或無候選時 `picks` 為空。
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
