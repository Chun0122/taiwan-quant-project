"""A2 parity 報告產生器（一次性交付物，MASTER_PLAN §6.2 #9 驗收項）。

量化「決策日 close 成交（舊 live，look-ahead）vs T+1 open 成交（新制）」的
績效差距：對每個 active rotation 組合，用同一段 DiscoveryRecord 史料跑兩次
backtest（t1_execution=True/False，save_result=False 不污染回測紀錄表），
輸出 markdown 對照表至 docs/reports/。

用法：
    python tools/parity_t1_report.py [--out docs/reports/t1_parity_YYYYMMDD.md]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func, select  # noqa: E402

from src.data.database import get_session  # noqa: E402
from src.data.schema import DiscoveryRecord  # noqa: E402
from src.portfolio.manager import RotationManager  # noqa: E402

# (metrics key, 顯示名, 值格式, 差距格式)
METRIC_ROWS = [
    ("total_return", "總報酬", "{:+.2%}", "{:+.2%}"),
    ("annual_return", "年化報酬", "{:+.2%}", "{:+.2%}"),
    ("max_drawdown", "最大回撤", "{:.2%}", "{:+.2%}"),
    ("sharpe_ratio", "Sharpe", "{:.2f}", "{:+.2f}"),
    ("win_rate", "勝率", "{:.2%}", "{:+.2%}"),
    ("total_trades", "交易數", "{:d}", "{:+d}"),
    ("total_cost", "總成本", "{:,.0f}", "{:+,.0f}"),
]


def _fmt(fmt: str, v) -> str:
    if v is None:
        return "—"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="T+1 parity 報告")
    parser.add_argument(
        "--out",
        default=f"docs/reports/t1_parity_{date.today().strftime('%Y%m%d')}.md",
        help="輸出 markdown 路徑",
    )
    args = parser.parse_args()

    with get_session() as session:
        start = session.execute(select(func.min(DiscoveryRecord.scan_date))).scalar_one()
        end = session.execute(select(func.max(DiscoveryRecord.scan_date))).scalar_one()
    if start is None:
        print("DiscoveryRecord 無資料，無法產生報告")
        return

    portfolios = [p for p in RotationManager.list_portfolios() if p["status"] == "active"]
    print(f"史料範圍：{start} ~ {end}；active 組合 {len(portfolios)} 個")

    lines: list[str] = [
        "# T+1 Parity 報告 — close 成交 vs T+1 open 成交",
        "",
        f"> 產生日：{date.today().isoformat()}（A2 Live T+1 Pending-Order 交付物）",
        f"> 史料：DiscoveryRecord {start} ~ {end}；同一 backtest 引擎、同參數，僅執行時點不同。",
        "> `close`（舊 live 行為）＝決策日收盤成交——**夜間決策時拿不到的價格（look-ahead）**；",
        "> `T+1 open`（新制）＝決策次一交易日開盤成交。差距 = 舊帳本的系統性偏差幅度。",
        "",
    ]

    for p in portfolios:
        name = p["name"]
        print(f"回測 {name}（close 模式）...")
        r_close = RotationManager(name).backtest(start_date=start, end_date=end, t1_execution=False, save_result=False)
        print(f"回測 {name}（T+1 模式）...")
        r_t1 = RotationManager(name).backtest(start_date=start, end_date=end, t1_execution=True, save_result=False)

        lines.append(f"## {name}（mode={p['mode']}, N={p['max_positions']}, hold={p['holding_days']}d）")
        lines.append("")
        lines.append("| 指標 | close 成交（舊） | T+1 open（新） | 差距（新−舊） |")
        lines.append("|------|-----------------:|---------------:|--------------:|")
        for key, label, fmt, diff_fmt in METRIC_ROWS:
            v_close = r_close.metrics.get(key)
            v_t1 = r_t1.metrics.get(key)
            diff = ""
            if isinstance(v_close, (int, float)) and isinstance(v_t1, (int, float)):
                diff = _fmt(diff_fmt, v_t1 - v_close)
            lines.append(f"| {label} | {_fmt(fmt, v_close)} | {_fmt(fmt, v_t1)} | {diff} |")
        lines.append("")

    lines += [
        "## 解讀",
        "",
        "- `T+1 − close` 為負值時，代表舊帳本高估績效（look-ahead 紅利）；正值代表隔日開盤反而有利（動能延續）。",
        "- 本差距自 2026-07-06 起已從 live 消除（decide/fill 兩段式），之後的 snapshot/alpha 不再含此偏差。",
        "- 熔斷（max_drawdown_liquidation）為刻意差異：live 即時、backtest T+1（風控優先於 parity）。",
        "",
    ]

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"報告已輸出：{out_path}")


if __name__ == "__main__":
    main()
