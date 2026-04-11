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
        use_ic_adjustment=getattr(args, "use_ic_adjustment", False),
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

    # 審計追蹤（--verbose 時輸出完整軌跡）
    if getattr(args, "verbose", False) and result.audit_trail:
        print(f"\n{result.audit_trail.format_verbose()}")

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
            use_ic_adjustment=getattr(args, "use_ic_adjustment", False),
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
                chip_tier_change=str(row.get("chip_tier_change", "")) or None,
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
        include_costs=getattr(args, "include_costs", False),
        entry_at_next_open=getattr(args, "entry_next_open", False),
    )

    print(f"正在計算 {args.mode} 推薦績效...")
    result = perf.evaluate()

    print_performance_report(
        result,
        args.mode,
        args.start,
        args.end,
        include_costs=getattr(args, "include_costs", False),
        entry_at_next_open=getattr(args, "entry_next_open", False),
    )

    # 匯出 CSV
    if args.export and not result["detail"].empty:
        result["detail"].to_csv(args.export, index=False, encoding="utf-8-sig")
        print(f"\n明細已匯出至: {args.export}")


def cmd_factor_diagnostics(args: "argparse.Namespace") -> None:
    """因子診斷 — 子因子 IC + 相關性矩陣。

    1. 執行一次掃描（取得子因子 rank）
    2. 計算四維度 IC + 子因子 IC
    3. 計算子因子間相關性矩陣
    """
    from src.discovery.scanner import (
        DividendScanner,
        GrowthScanner,
        MomentumScanner,
        SwingScanner,
        ValueScanner,
    )
    from src.discovery.scanner._functions import (
        compute_factor_correlation_matrix,
        compute_factor_ic,
        compute_sub_factor_ic,
    )

    init_db()

    mode = args.mode
    mode_label = {
        "momentum": "Momentum",
        "swing": "Swing",
        "value": "Value",
        "dividend": "Dividend",
        "growth": "Growth",
    }[mode]

    # Step 1: 執行掃描取得子因子
    scanner_map = {
        "momentum": MomentumScanner,
        "swing": SwingScanner,
        "value": ValueScanner,
        "dividend": DividendScanner,
        "growth": GrowthScanner,
    }

    if not getattr(args, "skip_sync", False):
        sync_days = 80 if mode == "swing" else 25
        ensure_sync_market_data(sync_days, args)

    print(f"正在執行 {mode_label} 掃描以收集子因子...")
    scanner = scanner_map[mode](top_n_results=50)
    result = scanner.run()

    if result.rankings.empty:
        print("無符合條件的股票，無法進行因子診斷")
        return

    # Step 2: 四維度 IC（使用歷史推薦記錄）
    holding_days = getattr(args, "holding_days", 5)
    lookback_days = getattr(args, "lookback_days", 30)

    from datetime import timedelta

    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import DailyPrice, DiscoveryRecord

    cutoff = result.scan_date - timedelta(days=lookback_days + holding_days + 10)

    with get_session() as session:
        stmt = select(
            DiscoveryRecord.scan_date,
            DiscoveryRecord.stock_id,
            DiscoveryRecord.close,
            DiscoveryRecord.technical_score,
            DiscoveryRecord.chip_score,
            DiscoveryRecord.fundamental_score,
            DiscoveryRecord.news_score,
        ).where(DiscoveryRecord.mode == mode, DiscoveryRecord.scan_date >= cutoff)
        rows = session.execute(stmt).all()
        df_records = pd.DataFrame(
            rows,
            columns=[
                "scan_date",
                "stock_id",
                "close",
                "technical_score",
                "chip_score",
                "fundamental_score",
                "news_score",
            ],
        )

        if df_records.empty:
            print(f"無 {mode} 模式的歷史推薦記錄，無法計算 IC")
            print("請先執行幾次 discover 掃描累積資料")
            return

        stock_ids = df_records["stock_id"].unique().tolist()
        price_stmt = select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
            DailyPrice.stock_id.in_(stock_ids),
            DailyPrice.date >= cutoff,
        )
        price_rows = session.execute(price_stmt).all()
        df_prices = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])

    # 四維度 IC
    ic_df = compute_factor_ic(df_records, df_prices, holding_days=holding_days, lookback_days=lookback_days)

    print(f"\n{'=' * 70}")
    print(f"因子診斷報告 [{mode_label}] — {holding_days} 日持有期, {lookback_days} 日回溯")
    print(f"歷史推薦: {len(df_records)} 筆")
    print(f"{'=' * 70}")

    if not ic_df.empty:
        print(f"\n{'─' * 50}")
        print("四維度 IC（Information Coefficient）")
        print(f"{'因子':<25} {'IC':>8} {'樣本':>6} {'評價':<10}")
        print(f"{'─' * 50}")
        for _, row in ic_df.iterrows():
            label = {"effective": "✓ 有效", "weak": "○ 弱", "inverse": "✗ 反向"}.get(row["direction"], "?")
            print(f"{row['factor']:<25} {row['ic']:>8.4f} {int(row['evaluable_count']):>6} {label:<10}")
    else:
        print("\n四維度 IC: 資料不足（需 ≥10 筆可評估推薦）")

    # Step 3: 子因子相關性矩陣
    sub_df = result.sub_factor_df
    if sub_df is not None and not sub_df.empty:
        corr_matrix = compute_factor_correlation_matrix(sub_df)
        if not corr_matrix.empty:
            print(f"\n{'─' * 50}")
            print("子因子相關性矩陣（Spearman）")
            print(f"{'─' * 50}")

            # 找出高相關對
            high_corr_pairs: list[tuple[str, str, float]] = []
            cols = corr_matrix.columns.tolist()
            for i, c1 in enumerate(cols):
                for c2 in cols[i + 1 :]:
                    corr = corr_matrix.loc[c1, c2]
                    if abs(corr) > 0.6:
                        high_corr_pairs.append((c1, c2, corr))

            if high_corr_pairs:
                print("\n高相關因子對（|r| > 0.6 — 可能冗餘）:")
                high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                for c1, c2, r in high_corr_pairs:
                    print(f"  {c1:<22} × {c2:<22} r={r:+.3f}")
            else:
                print("\n無高相關因子對（|r| > 0.6）— 因子獨立性良好")

            # 簡潔矩陣輸出
            print(f"\n完整矩陣（{len(cols)} 因子）:")
            # 只打印對角線以下
            header = f"{'':>22}" + "".join(f"{c[:8]:>9}" for c in cols)
            print(header)
            for i, c1 in enumerate(cols):
                row_str = f"{c1:<22}"
                for j, c2 in enumerate(cols):
                    if j > i:
                        row_str += f"{'':>9}"
                    else:
                        row_str += f"{corr_matrix.loc[c1, c2]:>+9.3f}"
                print(row_str)

            # 匯出
            if args.export:
                corr_matrix.to_csv(args.export, encoding="utf-8-sig")
                print(f"\n相關性矩陣已匯出至: {args.export}")
    else:
        print("\n子因子 rank: 此模式尚未實作子因子輸出")

    # Step 4: 子因子 IC（需要歷史推薦 + 子因子 rank 合併）
    # 當前掃描的子因子只有今天的 snapshot，若歷史也有 sub_factor_df 可擴展
    # 此處以今天的 sub_factor + 歷史 close → 計算今天候選的單截面 IC
    if sub_df is not None and not sub_df.empty:
        # 合併 scan_date 和 close 到 sub_factor_df
        close_map = dict(zip(result.rankings["stock_id"], result.rankings["close"]))
        sub_with_meta = sub_df.copy()
        sub_with_meta["scan_date"] = result.scan_date
        sub_with_meta["close"] = sub_with_meta["stock_id"].map(close_map)
        sub_with_meta = sub_with_meta.dropna(subset=["close"])

        if not sub_with_meta.empty:
            sub_ic = compute_sub_factor_ic(
                sub_with_meta,
                df_prices,
                holding_days=holding_days,
                lookback_days=lookback_days + 30,
            )
            if not sub_ic.empty:
                print(f"\n{'─' * 50}")
                print("子因子 IC（當前截面 — 需累積更多歷史方具統計意義）")
                print(f"{'因子':<25} {'IC':>8} {'樣本':>6} {'評價':<10}")
                print(f"{'─' * 50}")
                for _, row in sub_ic.iterrows():
                    label = {"effective": "✓ 有效", "weak": "○ 弱", "inverse": "✗ 反向"}.get(row["direction"], "?")
                    print(f"{row['factor']:<25} {row['ic']:>8.4f} {int(row['evaluable_count']):>6} {label:<10}")

    print(f"\n{'=' * 70}")


def cmd_ablation_test(args: "argparse.Namespace") -> None:
    """因子消融測試 — 維度級 + 子因子級消融分析。

    1. 執行一次掃描（取得候選分數 + 子因子 rank）
    2. 維度級：逐一歸零每個維度權重，觀察排名位移
    3. 子因子級：在 technical / chip 維度內逐一移除子因子
    4. 歷史績效消融（若有歷史推薦記錄）
    """
    from src.discovery.ablation import (
        AblationReport,
        compute_ablation_performance,
        format_ablation_report,
        run_dimension_ablation,
        run_sub_factor_ablation,
    )
    from src.discovery.scanner import (
        DividendScanner,
        GrowthScanner,
        MomentumScanner,
        SwingScanner,
        ValueScanner,
    )

    init_db()

    mode = args.mode
    top_n = getattr(args, "top", 20)
    mode_label = {
        "momentum": "Momentum",
        "swing": "Swing",
        "value": "Value",
        "dividend": "Dividend",
        "growth": "Growth",
    }[mode]

    scanner_map = {
        "momentum": MomentumScanner,
        "swing": SwingScanner,
        "value": ValueScanner,
        "dividend": DividendScanner,
        "growth": GrowthScanner,
    }

    if not getattr(args, "skip_sync", False):
        sync_days = 80 if mode == "swing" else 25
        ensure_sync_market_data(sync_days, args)

    print(f"正在執行 {mode_label} 掃描以收集因子分數...")
    scanner = scanner_map[mode](top_n_results=top_n, use_ic_adjustment=False)
    result = scanner.run()

    if result.rankings.empty:
        print("無符合條件的股票，無法進行消融分析")
        return

    # 取得 regime 權重
    from src.regime.detector import MarketRegimeDetector

    regime = getattr(scanner, "regime", "sideways")
    baseline_weights = MarketRegimeDetector.get_weights(mode, regime)

    report = AblationReport(
        mode=mode_label,
        regime=regime,
        baseline_top_n=top_n,
    )

    # ── 維度級消融 ─────────────────────────
    scored_df = result.rankings.copy()
    dim_results = run_dimension_ablation(scored_df, baseline_weights, top_n=top_n)
    report.dimension_results = dim_results

    # ── 子因子級消融 ───────────────────────
    sub_df = result.sub_factor_df
    if sub_df is not None and not sub_df.empty:
        for dim in ["technical", "chip"]:
            sub_results = run_sub_factor_ablation(sub_df, dim)
            report.sub_factor_results.extend(sub_results)

    # ── 輸出報告 ──────────────────────────
    print(format_ablation_report(report))

    # ── 歷史績效消融（若有歷史推薦記錄）────
    if getattr(args, "with_performance", False):
        from datetime import timedelta

        from sqlalchemy import select

        from src.data.database import get_session
        from src.data.schema import DailyPrice, DiscoveryRecord

        holding_days = getattr(args, "holding_days", 5)
        lookback = getattr(args, "lookback_days", 60)
        cutoff = result.scan_date - timedelta(days=lookback + holding_days + 10)

        with get_session() as session:
            stmt = select(
                DiscoveryRecord.scan_date,
                DiscoveryRecord.stock_id,
                DiscoveryRecord.close,
                DiscoveryRecord.rank,
                DiscoveryRecord.technical_score,
                DiscoveryRecord.chip_score,
                DiscoveryRecord.fundamental_score,
                DiscoveryRecord.news_score,
            ).where(DiscoveryRecord.mode == mode, DiscoveryRecord.scan_date >= cutoff)
            rows = session.execute(stmt).all()
            df_records = pd.DataFrame(
                rows,
                columns=[
                    "scan_date",
                    "stock_id",
                    "close",
                    "rank",
                    "technical_score",
                    "chip_score",
                    "fundamental_score",
                    "news_score",
                ],
            )

            if df_records.empty:
                print("\n無歷史推薦記錄，跳過績效消融分析")
            else:
                stock_ids = df_records["stock_id"].unique().tolist()
                price_stmt = select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                    DailyPrice.stock_id.in_(stock_ids),
                    DailyPrice.date >= cutoff,
                )
                price_rows = session.execute(price_stmt).all()
                df_prices = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])

                perf_df = compute_ablation_performance(
                    df_records,
                    df_prices,
                    baseline_weights,
                    holding_days=holding_days,
                    top_n=top_n,
                )

                if not perf_df.empty:
                    print(f"\n{'─' * 70}")
                    print(f"歷史績效消融（{holding_days} 日持有，回溯 {lookback} 天）")
                    print(f"{'─' * 70}")
                    print(f"{'移除維度':<20}  {'勝率':>7}  {'均報酬':>8}  {'勝率Δ':>7}  {'均報酬Δ':>8}")
                    for _, row in perf_df.iterrows():
                        wr = f"{row['win_rate']:.1%}"
                        avg = f"{row['avg_return']:+.2%}"
                        wd = f"{row['win_rate_delta']:+.1%}" if row["win_rate_delta"] != 0 else "  —  "
                        ad = f"{row['avg_return_delta']:+.2%}" if row["avg_return_delta"] != 0 else "   —   "
                        print(f"{row['removed_dimension']:<20}  {wr:>7}  {avg:>8}  {wd:>7}  {ad:>8}")

    # ── 匯出 CSV ──────────────────────────
    if getattr(args, "export", None) and report.dimension_results:
        rows = []
        for r in report.dimension_results:
            rows.append(
                {
                    "removed_dimension": r.removed_dimension,
                    "rank_correlation": r.rank_correlation,
                    "mean_rank_shift": r.mean_rank_shift,
                    "max_rank_shift": r.max_rank_shift,
                    "stocks_dropped": len(r.stocks_dropped),
                    "stocks_added": len(r.stocks_added),
                }
            )
        pd.DataFrame(rows).to_csv(args.export, index=False, encoding="utf-8-sig")
        print(f"\n消融結果已匯出至: {args.export}")
