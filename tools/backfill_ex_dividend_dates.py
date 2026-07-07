"""一次性回填 dividend 表的除息/除權交易日欄位（A3 修正，2026-07-07）。

背景：FinMind TaiwanStockDividend 的 date 是「除權息基準日」，比實際
除息交易日（價格跳空日）晚約 4-6 天。早期同步的列缺
cash_ex_dividend_date / stock_ex_dividend_date（upsert ON CONFLICT DO
NOTHING 不會補舊列），本腳本逐股重抓並 UPDATE 既有列。

用法：
    python tools/backfill_ex_dividend_dates.py            # 只補 NULL 列
    python tools/backfill_ex_dividend_dates.py --dry-run  # 只列出不寫入

速率：FinMind 0.5 秒/次，股數 = dividend 表 distinct stock_id。
冪等：重跑只影響仍為 NULL 的列（以 (stock_id, date) 對齊）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from src.data.database import get_session  # noqa: E402
from src.data.fetcher import FinMindFetcher  # noqa: E402
from src.data.schema import Dividend  # noqa: E402


def backfill(dry_run: bool = False) -> dict[str, int]:
    stats = {"stocks": 0, "rows_updated": 0, "rows_unmatched": 0}
    fetcher = FinMindFetcher()

    with get_session() as session:
        sids = [
            r[0]
            for r in session.execute(
                select(Dividend.stock_id).where(Dividend.cash_ex_dividend_date.is_(None)).distinct()
            )
        ]
        print(f"待回填股票數：{len(sids)}")

        for sid in sids:
            rows = (
                session.execute(
                    select(Dividend).where(Dividend.stock_id == sid, Dividend.cash_ex_dividend_date.is_(None))
                )
                .scalars()
                .all()
            )
            if not rows:
                continue
            start = min(r.date for r in rows).isoformat()
            df = fetcher.fetch_dividend(sid, start)
            if df.empty:
                stats["rows_unmatched"] += len(rows)
                continue
            # (stock_id, date=基準日) 對齊；NaT/NaN 防禦轉 None（SQLite Date 不吃 NaT）
            import pandas as pd

            def _clean(v):
                return None if v is None or pd.isna(v) else v

            by_date = {}
            for _, rec in df.iterrows():
                by_date[rec["date"]] = (
                    _clean(rec.get("cash_ex_dividend_date")),
                    _clean(rec.get("stock_ex_dividend_date")),
                )
            n = 0
            for r in rows:
                pair = by_date.get(r.date)
                if pair is None:
                    stats["rows_unmatched"] += 1
                    continue
                cash_ex, stock_ex = pair
                if not dry_run:
                    r.cash_ex_dividend_date = cash_ex
                    r.stock_ex_dividend_date = stock_ex
                n += 1
            stats["stocks"] += 1
            stats["rows_updated"] += n
            print(f"  {sid}: {n}/{len(rows)} 列回填")
        if not dry_run:
            session.commit()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回填 dividend 除息交易日欄位")
    parser.add_argument("--dry-run", action="store_true", help="只列出不寫入")
    args = parser.parse_args()
    stats = backfill(dry_run=args.dry_run)
    print(f"完成：{stats}")
