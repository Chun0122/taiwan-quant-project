"""Discord 訊息格式化 — 每日報告 / 策略排名 / 產業輪動共用。

Discord 訊息上限 2000 字元，長訊息會拆成多條。
"""

from __future__ import annotations

import pandas as pd


def format_daily_report(df: pd.DataFrame, top_n: int = 10) -> list[str]:
    """將每日選股報告格式化為 Discord 訊息。

    Args:
        df: DailyReportEngine.run() 的結果
        top_n: 顯示前 N 名

    Returns:
        訊息列表（可能多條，因 2000 字元上限）
    """
    if df is None or df.empty:
        return ["**📊 每日選股報告：**無資料"]

    messages = []
    header = [
        "**📊 每日選股報告**",
        f"掃描時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"掃描股票：{len(df)} 檔 | 顯示前 {min(top_n, len(df))} 名",
        "",
    ]

    display = df.head(top_n)
    lines = list(header)
    lines.append("```")
    lines.append(f"{'#':>2} {'代號':>6}  {'收盤':>8}  {'綜合':>5}  {'技術':>5}  {'籌碼':>5}  {'基本':>5}  {'ML':>5}")
    lines.append("─" * 55)

    current_msg_lines = list(lines)

    for _, row in display.iterrows():
        line = (
            f"{int(row.get('rank', 0)):>2} {row['stock_id']:>6}  "
            f"{row['close']:>8.1f}  {row['composite_score']:>5.3f}  "
            f"{row['technical_score']:>5.3f}  {row['chip_score']:>5.3f}  "
            f"{row['fundamental_score']:>5.3f}  {row['ml_score']:>5.3f}"
        )

        # 檢查是否超過 2000 字元
        test_msg = "\n".join(current_msg_lines + [line, "```"])
        if len(test_msg) > 1900:
            current_msg_lines.append("```")
            messages.append("\n".join(current_msg_lines))
            current_msg_lines = ["```", line]
        else:
            current_msg_lines.append(line)

    current_msg_lines.append("```")

    # 加入補充資訊
    extras = []
    for _, row in display.iterrows():
        parts = []
        if pd.notna(row.get("rsi")):
            parts.append(f"RSI={row['rsi']:.0f}")
        if pd.notna(row.get("foreign_net")):
            net = row["foreign_net"]
            parts.append(f"外資={'買' if net > 0 else '賣'}{abs(net):,.0f}")
        if pd.notna(row.get("yoy_growth")):
            parts.append(f"YoY={row['yoy_growth']:.1f}%")
        if parts:
            extras.append(f"`{row['stock_id']}` {' | '.join(parts)}")

    if extras:
        extra_text = "\n".join(extras[:5])
        test = "\n".join(current_msg_lines) + "\n" + extra_text
        if len(test) <= 1950:
            current_msg_lines.append(extra_text)
        else:
            messages.append("\n".join(current_msg_lines))
            current_msg_lines = [extra_text]

    messages.append("\n".join(current_msg_lines))
    return messages


def format_strategy_rank(df: pd.DataFrame, metric: str = "sharpe") -> str:
    """將策略排名結果格式化為 Discord 訊息。

    Args:
        df: StrategyRankEngine.run() 的結果
        metric: 排序指標名稱

    Returns:
        格式化訊息（截斷到 2000 字元）
    """
    if df is None or df.empty:
        return "**🏆 策略回測排名：**無資料"

    lines = [
        "**🏆 策略回測排名**",
        f"掃描時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"排序指標：{metric} | 共 {len(df)} 組合",
        "```",
        f"{'#':>2} {'代號':>6}  {'策略':<14}  {'報酬':>8}  {'Sharpe':>7}  {'MDD':>7}  {'勝率':>6}",
        "─" * 60,
    ]

    for _, row in df.head(15).iterrows():
        sharpe = f"{row['sharpe_ratio']:.2f}" if pd.notna(row.get("sharpe_ratio")) else "N/A"
        win_r = f"{row['win_rate']:.1f}%" if pd.notna(row.get("win_rate")) else "N/A"
        line = (
            f"{int(row['rank']):>2} {row['stock_id']:>6}  "
            f"{row['strategy_name']:<14}  {row['total_return']:>7.2f}%  "
            f"{sharpe:>7}  {row['max_drawdown']:>6.2f}%  {win_r:>6}"
        )
        lines.append(line)

    lines.append("```")

    if len(df) > 15:
        lines.append(f"...及其他 {len(df) - 15} 組合")

    msg = "\n".join(lines)
    return msg[:2000]


def format_industry_report(
    sector_df: pd.DataFrame,
    top_stocks_df: pd.DataFrame,
    top_n: int = 5,
) -> list[str]:
    """將產業輪動分析格式化為 Discord 訊息。

    Args:
        sector_df: IndustryRotationAnalyzer.rank_sectors() 的結果
        top_stocks_df: top_stocks_from_hot_sectors() 的結果
        top_n: 顯示前 N 個產業

    Returns:
        訊息列表
    """
    messages = []

    if sector_df is None or sector_df.empty:
        return ["**🏭 產業輪動分析：**無資料"]

    # 產業排名
    lines = [
        "**🏭 產業輪動分析**",
        f"分析時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "**產業綜合排名：**",
        "```",
        f"{'#':>2} {'產業':<12}  {'綜合':>6}  {'法人':>6}  {'動能':>6}  {'淨買超':>12}  {'漲幅':>7}",
        "─" * 60,
    ]

    for _, row in sector_df.head(top_n).iterrows():
        total_net = row.get("total_net", 0)
        net_str = f"{total_net:>12,.0f}" if total_net != 0 else "         N/A"
        avg_ret = row.get("avg_return_pct", 0)
        ret_str = f"{avg_ret:>6.2f}%" if avg_ret != 0 else "    N/A"
        line = (
            f"{int(row['rank']):>2} {str(row['industry']):<12}  "
            f"{row['sector_score']:>6.3f}  "
            f"{row['institutional_score']:>6.3f}  "
            f"{row['momentum_score']:>6.3f}  "
            f"{net_str}  {ret_str}"
        )
        lines.append(line)

    lines.append("```")
    messages.append("\n".join(lines))

    # 精選個股
    if top_stocks_df is not None and not top_stocks_df.empty:
        stock_lines = ["**🔥 熱門產業精選個股：**"]

        for ind in top_stocks_df["industry"].unique():
            sector_stocks = top_stocks_df[top_stocks_df["industry"] == ind]
            stock_lines.append(f"\n**{ind}**")
            stock_lines.append("```")
            for _, sr in sector_stocks.iterrows():
                name = sr.get("stock_name", "")[:6]
                foreign = sr.get("foreign_net_sum", 0)
                stock_lines.append(
                    f"  {sr['stock_id']} {name:<6} "
                    f"收盤={sr['close']:>8.1f}  "
                    f"外資={'買' if foreign > 0 else '賣'}{abs(foreign):>10,.0f}"
                )
            stock_lines.append("```")

        stock_msg = "\n".join(stock_lines)
        if len(stock_msg) > 2000:
            # 拆分
            messages.append(stock_msg[:1950] + "\n```")
        else:
            messages.append(stock_msg)

    return messages


def format_discovery_report(result, top_n: int = 20) -> list[str]:
    """將全市場掃描結果格式化為 Discord 訊息。

    Args:
        result: DiscoveryResult 實例
        top_n: 顯示前 N 名

    Returns:
        訊息列表（可能多條，因 2000 字元上限）
    """
    if result.rankings is None or result.rankings.empty:
        return ["**🔍 全市場選股掃描：**無符合條件的股票"]

    messages = []
    header = [
        "**🔍 全市場選股掃描**",
        f"掃描時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"掃描範圍：{result.total_stocks} 支 → 粗篩 {result.after_coarse} 支 → Top {min(top_n, len(result.rankings))}",
        "",
    ]

    display = result.rankings.head(top_n)
    lines = list(header)
    lines.append("```")
    lines.append(f"{'#':>2} {'代號':>6} {'名稱':<6}  {'收盤':>7}  {'綜合':>5}  {'技術':>5}  {'籌碼':>5}  {'產業':<8}")
    lines.append("─" * 58)

    current_msg_lines = list(lines)

    for _, row in display.iterrows():
        name = str(row.get("stock_name", ""))[:6]
        industry = str(row.get("industry_category", ""))[:8]
        line = (
            f"{int(row['rank']):>2} {row['stock_id']:>6} {name:<6}  "
            f"{row['close']:>7.1f}  {row['composite_score']:>5.3f}  "
            f"{row['technical_score']:>5.3f}  {row['chip_score']:>5.3f}  "
            f"{industry:<8}"
        )

        test_msg = "\n".join(current_msg_lines + [line, "```"])
        if len(test_msg) > 1900:
            current_msg_lines.append("```")
            messages.append("\n".join(current_msg_lines))
            current_msg_lines = ["```", line]
        else:
            current_msg_lines.append(line)

    current_msg_lines.append("```")

    # 產業分布摘要
    if result.sector_summary is not None and not result.sector_summary.empty:
        sector_lines = ["\n**產業分布：**"]
        for _, sr in result.sector_summary.head(5).iterrows():
            sector_lines.append(f"  {sr['industry']}: {sr['count']} 支 (均分 {sr['avg_score']:.3f})")
        sector_text = "\n".join(sector_lines)

        test = "\n".join(current_msg_lines) + sector_text
        if len(test) <= 1950:
            current_msg_lines.append(sector_text)
        else:
            messages.append("\n".join(current_msg_lines))
            current_msg_lines = [sector_text]

    messages.append("\n".join(current_msg_lines))
    return messages
