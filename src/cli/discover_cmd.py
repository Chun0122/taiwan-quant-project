"""CLI Discover 子命令 — discover / discover all / discover-backtest。"""

from __future__ import annotations

import argparse

import pandas as pd

from src.cli.helpers import ensure_sync_market_data, init_db
from src.cli.helpers import safe_print as print


def cmd_discover(args: argparse.Namespace) -> None:
    """執行全市場選股掃描（momentum / swing / value 模式）。"""

    from src.discovery.scanner import DividendScanner, GrowthScanner, MomentumScanner, SwingScanner, ValueScanner

    init_db()

    mode = getattr(args, "mode", "momentum") or "momentum"

    if mode == "all":
        _cmd_discover_all(args)
        return

    mode_label = {
        "momentum": "Momentum 短線動能",
        "swing": "Swing 中期波段",
        "value": "Value 價值修復",
        "dividend": "Dividend 高息存股",
        "growth": "Growth 高成長",
    }[mode]

    # 各模式自動擴展同步天數至 Scanner 所需的 lookback_days
    sync_days = args.sync_days
    if mode == "swing" and sync_days < 80:
        sync_days = 80
        print(f"  [Swing 模式] 自動擴展同步天數至 {sync_days} 天（SMA60 / 60日動能需要）")
    elif mode in ("momentum", "value", "dividend", "growth") and sync_days < 25:
        sync_days = 25
        print(f"  [{mode_label}] 自動擴展同步天數至 {sync_days} 天（ATR14 / SMA20 需要）")

    ensure_sync_market_data(sync_days, args)

    # 選擇 scanner
    scanner_map = {
        "momentum": MomentumScanner,
        "swing": SwingScanner,
        "value": ValueScanner,
        "dividend": DividendScanner,
        "growth": GrowthScanner,
    }
    ScannerClass = scanner_map[mode]
    print(f"正在掃描全市場 [{mode_label}]...")
    scanner = ScannerClass(
        min_price=args.min_price,
        max_price=args.max_price,
        min_volume=args.min_volume,
        top_n_results=args.top,
        weekly_confirm=getattr(args, "weekly_confirm", False),
    )
    result = scanner.run()

    if result.rankings.empty:
        print("無符合條件的股票")
        return

    # 顯示結果
    display = result.rankings.head(args.top)
    print(f"\n{'=' * 80}")
    print(
        f"全市場選股掃描 [{mode_label}] — "
        f"掃描 {result.total_stocks} 支 → 粗篩 {result.after_coarse} 支 → Top {len(display)}"
    )
    print(f"{'=' * 80}")
    print(
        f"{'#':>3}  {'代號':>6} {'名稱':<8}  {'收盤':>8}  {'綜合':>6}  "
        f"{'技術':>6}  {'籌碼':>6} {'層':>3}  {'基本':>6}  {'產業加成':>6}  {'產業':<10}"
    )
    print(f"{'─' * 90}")

    for _, row in display.iterrows():
        name = str(row.get("stock_name", ""))[:8]
        industry = str(row.get("industry_category", ""))[:10]
        sector_bonus = row.get("sector_bonus", 0.0)
        if pd.isna(sector_bonus):
            sector_bonus = 0.0
        chip_tier = str(row.get("chip_tier", "")) or "N/A"
        print(
            f"{int(row['rank']):>3}  {row['stock_id']:>6} {name:<8}  "
            f"{row['close']:>8.1f}  {row['composite_score']:>6.3f}  "
            f"{row['technical_score']:>6.3f}  {row['chip_score']:>6.3f} {chip_tier:>3}  "
            f"{row['fundamental_score']:>6.3f}  {sector_bonus:>+6.1%}  {industry:<10}"
        )

    # 產業分布
    if result.sector_summary is not None and not result.sector_summary.empty:
        print(f"\n{'─' * 40}")
        print("產業分布")
        for _, sr in result.sector_summary.head(8).iterrows():
            print(f"  {sr['industry']:<14} {int(sr['count']):>3} 支  (均分 {sr['avg_score']:.3f})")

    # 進出場建議（Top 5）
    ee_cols = ["entry_price", "stop_loss", "take_profit", "entry_trigger", "valid_until"]
    if all(c in result.rankings.columns for c in ee_cols):
        print(f"\n{'─' * 80}")
        print("[ 進出場建議 ]")
        print(f"{'#':>3}  {'代號':>6}  {'進場價':>8}  {'止損':>12}  {'止利':>12}  {'觸發條件':<18}  {'有效至'}")
        print(f"{'─' * 80}")
        for _, row in result.rankings.head(5).iterrows():
            ep = row.get("entry_price")
            sl = row.get("stop_loss")
            tp = row.get("take_profit")
            sl_str = f"{sl:.1f}({(sl - ep) / ep:+.1%})" if pd.notna(sl) and pd.notna(ep) and ep > 0 else "—"
            tp_str = f"{tp:.1f}({(tp - ep) / ep:+.1%})" if pd.notna(tp) and pd.notna(ep) and ep > 0 else "—"
            ep_str = f"{ep:.1f}" if pd.notna(ep) else "—"
            trigger = str(row.get("entry_trigger", "")) or "—"
            valid = str(row.get("valid_until", "")) or "—"
            print(
                f"{int(row['rank']):>3}  {row['stock_id']:>6}  "
                f"{ep_str:>8}  {sl_str:>12}  {tp_str:>12}  {trigger:<18}  {valid}"
            )

    # 儲存推薦記錄到 DB
    _save_discovery_records(result, mode, scanner)

    # 歷史比較
    if args.compare:
        _show_discovery_comparison(mode, result)

    # 匯出 CSV
    if args.export:
        result.rankings.to_csv(args.export, index=False)
        print(f"\n結果已匯出至: {args.export}")

    # AI 選股摘要
    if getattr(args, "ai_summary", False):
        from src.report.ai_report import generate_ai_summary

        regime = getattr(scanner, "regime", "sideways")
        print(f"\n{'─' * 80}")
        print("[ AI 選股摘要 ]")
        summary = generate_ai_summary(result, regime=regime, top_stocks=result.rankings.head(args.top))
        print(summary)
        print(f"{'─' * 80}")

    # Discord 通知
    if args.notify:
        from src.notification.line_notify import send_message
        from src.report.formatter import format_discovery_report

        msgs = format_discovery_report(result, top_n=args.top)
        for msg in msgs:
            send_message(msg)
        print("Discord 通知已發送")


def _build_cross_comparison(results: dict, top_n: int) -> "pd.DataFrame":
    """建立五模式交叉比較表。

    Args:
        results: dict[mode_key → ScanResult]（只含成功的模式）
        top_n: 每個模式取前 N 名納入比較

    Returns:
        DataFrame，欄位：stock_id, stock_name, close, appearances, best_rank,
        avg_score, best_mode, chip_tier,
        momentum_rank, swing_rank, value_rank, dividend_rank, growth_rank,
        industry_category
    """
    mode_cols = ["momentum", "swing", "value", "dividend", "growth"]
    mode_labels = {"momentum": "動能", "swing": "波段", "value": "價值", "dividend": "高息", "growth": "成長"}

    # chip_tier 排序權重（數字越大越高）
    def _tier_weight(tier: str) -> int:
        if not tier or tier == "N/A":
            return 0
        try:
            return int(tier.replace("F", ""))
        except ValueError:
            return 0

    # 收集各模式的 stock → {rank, composite_score, chip_tier} 映射
    mode_data: dict[str, dict] = {}
    stock_meta: dict[str, dict] = {}  # stock_id → {name, close, industry}

    for mode_key, result in results.items():
        if result is None or result.rankings.empty:
            mode_data[mode_key] = {}
            continue
        subset = result.rankings.head(top_n)
        per_mode: dict[str, dict] = {}
        for _, row in subset.iterrows():
            sid = row["stock_id"]
            per_mode[sid] = {
                "rank": int(row["rank"]),
                "composite_score": float(row.get("composite_score", 0.0)),
                "chip_tier": str(row.get("chip_tier", "N/A") or "N/A"),
            }
            if sid not in stock_meta:
                stock_meta[sid] = {
                    "stock_name": str(row.get("stock_name", "")),
                    "close": float(row.get("close", 0.0)),
                    "industry_category": str(row.get("industry_category", "")),
                }
        mode_data[mode_key] = per_mode

    if not stock_meta:
        return pd.DataFrame()

    rows = []
    for sid, meta in stock_meta.items():
        row = {"stock_id": sid, **meta}
        appearances = 0
        best_rank = 9999
        scores: list[float] = []
        best_mode_key = None
        best_score = -1.0
        best_chip_tier = "N/A"

        for col in mode_cols:
            entry = mode_data.get(col, {}).get(sid)
            if entry is not None:
                rank_val = entry["rank"]
                score_val = entry["composite_score"]
                tier_val = entry["chip_tier"]
                row[f"{col}_rank"] = rank_val
                appearances += 1
                if rank_val < best_rank:
                    best_rank = rank_val
                scores.append(score_val)
                if score_val > best_score:
                    best_score = score_val
                    best_mode_key = col
                if _tier_weight(tier_val) > _tier_weight(best_chip_tier):
                    best_chip_tier = tier_val
            else:
                row[f"{col}_rank"] = None

        row["appearances"] = appearances
        row["best_rank"] = best_rank if best_rank < 9999 else None
        row["avg_score"] = round(sum(scores) / len(scores), 4) if scores else None
        row["best_mode"] = mode_labels.get(best_mode_key, best_mode_key) if best_mode_key else None
        row["chip_tier"] = best_chip_tier
        rows.append(row)

    df = pd.DataFrame(rows)
    # 主要排序：avg_score 降序；次要：appearances 降序；再次：best_rank 升序
    df = df.sort_values(
        ["avg_score", "appearances", "best_rank"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return df


def _cmd_discover_all(args: argparse.Namespace) -> None:
    """執行五個 Scanner 並輸出多模式綜合比較表。"""
    import datetime

    from src.discovery.scanner import DividendScanner, GrowthScanner, MomentumScanner, SwingScanner, ValueScanner

    init_db()

    # Swing 模式需要至少 80 天資料
    sync_days = max(args.sync_days, 80)
    ensure_sync_market_data(sync_days, args)

    scanner_classes = {
        "momentum": MomentumScanner,
        "swing": SwingScanner,
        "value": ValueScanner,
        "dividend": DividendScanner,
        "growth": GrowthScanner,
    }
    mode_labels = {
        "momentum": "動能",
        "swing": "波段",
        "value": "價值",
        "dividend": "高息",
        "growth": "成長",
    }

    results: dict = {}
    scan_summaries: list[str] = []

    for mode_key, ScannerClass in scanner_classes.items():
        label = mode_labels[mode_key]
        print(f"正在掃描 [{label}]...", end="", flush=True)
        scanner = ScannerClass(
            min_price=args.min_price,
            max_price=args.max_price,
            min_volume=args.min_volume,
            top_n_results=args.top,
            weekly_confirm=getattr(args, "weekly_confirm", False),
        )
        result = scanner.run()
        results[mode_key] = result

        if result.rankings.empty:
            print(" (無符合條件的股票)")
            scan_summaries.append(
                f"  {label:<4} 掃描 {result.total_stocks:,} 支 → 粗篩 {result.after_coarse} 支 → 0 筆"
            )
        else:
            actual = len(result.rankings.head(args.top))
            print(f" -> Top {actual}")
            scan_summaries.append(
                f"  {label:<4} 掃描 {result.total_stocks:,} 支 → 粗篩 {result.after_coarse} 支 → Top {actual}"
            )
            _save_discovery_records(result, mode_key, scanner)

    # 建立比較表
    df = _build_cross_comparison(results, args.top)

    # 套用 --min-appearances 篩選
    min_app = getattr(args, "min_appearances", 1)
    if min_app > 1 and not df.empty:
        df = df[df["appearances"] >= min_app].reset_index(drop=True)

    today = datetime.date.today().strftime("%Y-%m-%d")
    n_modes = sum(1 for r in results.values() if r is not None and not r.rankings.empty)

    print(f"\n{'═' * 100}")
    print(f"多模式綜合比較 -- {today}  |  掃描 {n_modes} 個模式 x Top {args.top}  |  排序: avg_score desc")
    print(f"{'═' * 100}")

    if df.empty:
        print(f"  （無出現在 {min_app}+ 個模式的股票）")
    else:
        print(
            f"{'出現':>5}  {'代號':>6} {'名稱':<8}  {'收盤':>8}  {'均分':>6}  {'最佳模式':<6} {'層':>3}  "
            f"{'動能':>5} {'波段':>5} {'價值':>5} {'高息':>5} {'成長':>5}  {'產業':<12}"
        )
        print(f"{'─' * 100}")

        def _fmt_rank(val) -> str:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "  —  "
            return f"#{int(val):<4}"

        for _, row in df.iterrows():
            app = int(row["appearances"])
            star = f"★×{app}" if app < 5 else "★×5!"
            name = str(row.get("stock_name", ""))[:8]
            industry = str(row.get("industry_category", ""))[:12]
            close_val = row.get("close", 0.0)
            close_str = f"{close_val:>8.1f}" if pd.notna(close_val) else "       —"
            avg_score = row.get("avg_score")
            avg_str = f"{avg_score:>6.3f}" if pd.notna(avg_score) else "     —"
            best_mode = str(row.get("best_mode") or "—")[:6]
            chip_tier = str(row.get("chip_tier") or "N/A")
            print(
                f"{star:>5}  {row['stock_id']:>6} {name:<8}  {close_str}  {avg_str}  {best_mode:<6} {chip_tier:>3}  "
                f"{_fmt_rank(row.get('momentum_rank')):>5} "
                f"{_fmt_rank(row.get('swing_rank')):>5} "
                f"{_fmt_rank(row.get('value_rank')):>5} "
                f"{_fmt_rank(row.get('dividend_rank')):>5} "
                f"{_fmt_rank(row.get('growth_rank')):>5}  {industry:<12}"
            )

    print(f"{'─' * 100}")
    print("\n各模式掃描摘要：")
    for s in scan_summaries:
        print(s)

    # CSV 匯出
    if args.export and not df.empty:
        df.to_csv(args.export, index=False)
        print(f"\n結果已匯出至: {args.export}")

    # Discord 通知（以 avg_score 排序）
    if args.notify and not df.empty:
        from src.notification.line_notify import send_message

        top10 = df[df["appearances"] >= max(2, min_app)].head(10) if len(df) > 0 else df.head(10)
        lines = [f"**多模式綜合選股** ({today})", f"掃描 {n_modes} 個模式 × Top {args.top}", ""]
        for _, row in top10.iterrows():
            name = str(row.get("stock_name", ""))[:8]
            app = int(row["appearances"])
            avg_score = row.get("avg_score")
            avg_str = f"{avg_score:.3f}" if pd.notna(avg_score) else "—"
            best_mode = str(row.get("best_mode") or "—")
            chip_tier = str(row.get("chip_tier") or "N/A")
            lines.append(f"{'★' * app} {row['stock_id']} {name}  均分{avg_str} [{best_mode}] {chip_tier}")
        send_message("\n".join(lines))
        print("Discord 通知已發送")


def _save_discovery_records(result, mode: str, scanner) -> None:
    """將 discover 推薦結果存入 DB（供歷史追蹤用）。"""
    from src.data.database import get_session
    from src.data.schema import DiscoveryRecord

    if result.rankings.empty:
        return

    regime = getattr(scanner, "regime", "sideways")
    scan_date = result.scan_date

    records = []
    for _, row in result.rankings.iterrows():
        records.append(
            DiscoveryRecord(
                scan_date=scan_date,
                mode=mode,
                rank=int(row["rank"]),
                stock_id=row["stock_id"],
                stock_name=str(row.get("stock_name", "")) or None,
                close=float(row["close"]),
                composite_score=float(row["composite_score"]),
                technical_score=float(row.get("technical_score", 0.5)),
                chip_score=float(row.get("chip_score", 0.5)),
                fundamental_score=float(row.get("fundamental_score", 0.5)),
                news_score=float(row.get("news_score", 0.5)),
                sector_bonus=float(row.get("sector_bonus", 0.0)) if pd.notna(row.get("sector_bonus")) else 0.0,
                industry_category=str(row.get("industry_category", "")) or None,
                regime=regime,
                total_stocks=result.total_stocks,
                after_coarse=result.after_coarse,
                entry_price=float(row.get("entry_price")) if pd.notna(row.get("entry_price")) else None,
                stop_loss=float(row.get("stop_loss")) if pd.notna(row.get("stop_loss")) else None,
                take_profit=float(row.get("take_profit")) if pd.notna(row.get("take_profit")) else None,
                entry_trigger=str(row.get("entry_trigger", "")) or None,
                valid_until=row.get("valid_until") if pd.notna(row.get("valid_until")) else None,
                chip_tier=str(row.get("chip_tier", "")) or None,
                concept_bonus=float(row.get("concept_bonus", 0.0)) if pd.notna(row.get("concept_bonus")) else None,
                daytrade_penalty=float(row.get("daytrade_penalty", 0.0))
                if pd.notna(row.get("daytrade_penalty"))
                else None,
                daytrade_tags=str(row.get("daytrade_tags", "")) or None,
            )
        )

    with get_session() as session:
        # 先刪除同日同模式的舊記錄（重跑時覆蓋）
        session.query(DiscoveryRecord).filter(
            DiscoveryRecord.scan_date == scan_date,
            DiscoveryRecord.mode == mode,
        ).delete()
        session.add_all(records)
        session.commit()

    print(f"\n推薦記錄已存入 DB（{len(records)} 筆，{scan_date} {mode}）")


def _show_discovery_comparison(mode: str, current_result) -> None:
    """顯示本次推薦與上次推薦的差異（新進/退出/排名變化）。"""
    from sqlalchemy import func, select

    from src.data.database import get_session
    from src.data.schema import DiscoveryRecord

    if current_result.rankings.empty:
        return

    scan_date = current_result.scan_date

    with get_session() as session:
        # 找到上次掃描日期（同模式、日期 < 今天）
        prev_date_row = session.execute(
            select(func.max(DiscoveryRecord.scan_date)).where(
                DiscoveryRecord.mode == mode,
                DiscoveryRecord.scan_date < scan_date,
            )
        ).scalar()

        if prev_date_row is None:
            print(f"\n{'─' * 40}")
            print("歷史比較：無前次記錄可供比較")
            return

        prev_date = prev_date_row

        # 載入上次記錄
        prev_rows = session.execute(
            select(
                DiscoveryRecord.stock_id,
                DiscoveryRecord.stock_name,
                DiscoveryRecord.rank,
                DiscoveryRecord.composite_score,
            ).where(
                DiscoveryRecord.scan_date == prev_date,
                DiscoveryRecord.mode == mode,
            )
        ).all()

    if not prev_rows:
        print(f"\n{'─' * 40}")
        print("歷史比較：無前次記錄可供比較")
        return

    prev_df = pd.DataFrame(prev_rows, columns=["stock_id", "stock_name", "rank", "composite_score"])
    prev_ids = set(prev_df["stock_id"])
    prev_rank = dict(zip(prev_df["stock_id"], prev_df["rank"]))

    curr_ids = set(current_result.rankings["stock_id"])
    curr_rank = dict(zip(current_result.rankings["stock_id"], current_result.rankings["rank"]))

    new_entries = curr_ids - prev_ids
    exits = prev_ids - curr_ids
    stayed = curr_ids & prev_ids

    print(f"\n{'─' * 50}")
    print(f"歷史比較（vs {prev_date}）")
    print(f"{'─' * 50}")

    # 新進
    if new_entries:
        print(f"\n  新進（{len(new_entries)} 支）：")
        for sid in sorted(new_entries, key=lambda s: curr_rank[s]):
            row = current_result.rankings[current_result.rankings["stock_id"] == sid].iloc[0]
            name = str(row.get("stock_name", ""))[:6]
            print(f"    + #{curr_rank[sid]:>2}  {sid:>6} {name:<6}  綜合 {row['composite_score']:.3f}")

    # 退出
    if exits:
        print(f"\n  退出（{len(exits)} 支）：")
        for sid in sorted(exits, key=lambda s: prev_rank[s]):
            prev_row = prev_df[prev_df["stock_id"] == sid].iloc[0]
            name = str(prev_row.get("stock_name", ""))[:6]
            print(f"    - 前#{prev_rank[sid]:>2}  {sid:>6} {name:<6}")

    # 排名變化（僅顯示變動 >= 3 名的）
    rank_changes = []
    for sid in stayed:
        delta = prev_rank[sid] - curr_rank[sid]  # 正=上升, 負=下降
        if abs(delta) >= 3:
            rank_changes.append((sid, curr_rank[sid], delta))

    if rank_changes:
        rank_changes.sort(key=lambda x: -x[2])  # 上升最多的排前面
        print("\n  排名變動（變動 >= 3 名）：")
        for sid, curr_r, delta in rank_changes:
            arrow = "↑" if delta > 0 else "↓"
            row = current_result.rankings[current_result.rankings["stock_id"] == sid].iloc[0]
            name = str(row.get("stock_name", ""))[:6]
            print(f"    {arrow} #{curr_r:>2} ({delta:>+3})  {sid:>6} {name:<6}  綜合 {row['composite_score']:.3f}")

    if not new_entries and not exits and not rank_changes:
        print("  推薦清單無顯著變化")


def cmd_discover_backtest(args: argparse.Namespace) -> None:
    """評估 Discover 推薦的歷史績效。"""

    from src.discovery.performance import DiscoveryPerformance, print_performance_report

    init_db()

    holding_days = [int(d) for d in args.days.split(",")] if args.days else None

    perf = DiscoveryPerformance(
        mode=args.mode,
        holding_days=holding_days,
        top_n=args.top,
        start_date=args.start,
        end_date=args.end,
    )

    print(f"正在計算 {args.mode} 推薦績效...")
    result = perf.evaluate()

    print_performance_report(result, args.mode, args.start, args.end)

    # 匯出 CSV
    if args.export and not result["detail"].empty:
        result["detail"].to_csv(args.export, index=False, encoding="utf-8-sig")
        print(f"\n明細已匯出至: {args.export}")
