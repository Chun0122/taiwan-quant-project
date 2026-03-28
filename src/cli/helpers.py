"""CLI 共用工具函數 — print override / logging / DB 初始化 / 檔案讀取。"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

from src.config import settings

# Windows cp950 終端無法輸出 emoji/特殊 Unicode，全域覆蓋 print 避免 UnicodeEncodeError
_builtin_print = print


def safe_print(*args: object, **kwargs: object) -> None:  # noqa: A001
    """print() wrapper — UnicodeEncodeError 時 fallback 至 UTF-8 buffer。"""
    try:
        _builtin_print(*args, **kwargs)  # type: ignore[arg-type]
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        end = kwargs.get("end", "\n")
        sys.stdout.flush()
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.write(str(end).encode("utf-8"))
        sys.stdout.buffer.flush()


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.logging.level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def init_db() -> None:
    """共用 DB 初始化，避免各 cmd_ 函數重複 lazy import + 呼叫。"""
    from src.data.database import init_db as _init_db

    _init_db()


def ensure_sync_market_data(sync_days: int, args: argparse.Namespace) -> None:
    """共用全市場資料同步流程（cmd_discover / _cmd_discover_all 共用）。

    依序執行：stock_info → TAIEX → daily_price + institutional + margin，並印出筆數。
    若 args.skip_sync 為 True 則直接跳過。
    """
    if args.skip_sync:
        return

    from src.data.pipeline import sync_market_data, sync_stock_info, sync_taiex_index

    safe_print("正在同步股票基本資料...")
    sync_stock_info(force_refresh=False)
    safe_print("正在同步 TAIEX 加權指數（Regime 偵測用）...")
    sync_taiex_index()
    safe_print(f"正在同步全市場資料（{sync_days} 天，TWSE/TPEX 官方資料）...")
    counts = sync_market_data(days=sync_days, max_stocks=args.max_stocks)
    safe_print(
        f"  日K線: {counts['daily_price']:,} 筆 | "
        f"法人: {counts['institutional']:,} 筆 | "
        f"融資融券: {counts['margin']:,} 筆"
    )


def read_stocks_from_file(path: str) -> list[str]:
    """從文字或 CSV 檔案讀取股票代號清單。

    支援兩種格式：
    - 純文字：每行一個代號，`#` 開頭為注解行，空白行跳過
    - CSV：第一欄為代號，第一行若含非數字字元則視為 header 跳過

    Args:
        path: 檔案路徑（.txt 或 .csv）

    Returns:
        去重後保序的股票代號清單
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到股票清單檔案：{path}")

    stocks: list[str] = []
    ext = os.path.splitext(path)[1].lower()

    for encoding in ("utf-8", "big5"):
        try:
            with open(path, encoding=encoding, newline="") as f:
                if ext == ".csv":
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if not row:
                            continue
                        val = row[0].strip()
                        if i == 0 and not val.isdigit():
                            continue  # 跳過 header
                        if val:
                            stocks.append(val)
                else:
                    for line in f:
                        val = line.strip()
                        if not val or val.startswith("#"):
                            continue
                        stocks.append(val)
            break
        except UnicodeDecodeError:
            stocks = []
            continue

    seen: dict[str, None] = {}
    for s in stocks:
        seen[s] = None
    return list(seen.keys())
