"""Plotly 圖表元件 — 提供各類股票分析圖表。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------ #
#  個股分析圖表
# ------------------------------------------------------------------ #


def plot_candlestick(
    df: pd.DataFrame,
    selected_indicators: set[str] | None = None,
    df_announcements: pd.DataFrame | None = None,
) -> go.Figure:
    """K線圖 + 可選 SMA/BB/RSI/MACD + 成交量疊加（Row 1 secondary_y）+ MOPS 公告標記。"""
    if selected_indicators is None:
        selected_indicators = {"sma_5", "sma_10", "sma_20", "sma_60", "bb", "rsi", "macd"}

    show_rsi = "rsi" in selected_indicators and "rsi_14" in df.columns
    show_macd = "macd" in selected_indicators and "macd" in df.columns

    # 動態決定列數與行高
    subplot_titles = ["K線 / 均線 / 布林通道"]
    row_heights_base = [0.55]
    if show_rsi:
        subplot_titles.append("RSI (14)")
        row_heights_base.append(0.2)
    if show_macd:
        subplot_titles.append("MACD")
        row_heights_base.append(0.25)

    n_rows = len(subplot_titles)
    total_h = sum(row_heights_base)
    row_heights = [h / total_h for h in row_heights_base]
    # Row 1 有 secondary_y（成交量），其餘無
    specs = [[{"secondary_y": True}]] + [[{}] for _ in range(n_rows - 1)]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        specs=specs,
    )

    rsi_row = 2 if show_rsi else None
    macd_row = (3 if show_rsi else 2) if show_macd else None

    # --- Row 1: K線 ---
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K線",
            increasing_line_color="#EF5350",
            decreasing_line_color="#26A69A",
            increasing_fillcolor="#EF5350",
            decreasing_fillcolor="#26A69A",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # SMA
    sma_colors = {"sma_5": "#FF9800", "sma_10": "#2196F3", "sma_20": "#9C27B0", "sma_60": "#795548"}
    for sma_key, color in sma_colors.items():
        if sma_key in selected_indicators and sma_key in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[sma_key],
                    name=sma_key.upper(),
                    line=dict(width=1, color=color),
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

    # 布林通道
    if "bb" in selected_indicators and "bb_upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["bb_upper"],
                name="BB Upper",
                line=dict(width=1, color="rgba(150,150,150,0.5)", dash="dot"),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["bb_lower"],
                name="BB Lower",
                line=dict(width=1, color="rgba(150,150,150,0.5)", dash="dot"),
                fill="tonexty",
                fillcolor="rgba(150,150,150,0.08)",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    # --- 成交量疊加（Row 1, secondary_y=True）---
    if "volume" in df.columns:
        vol_colors = []
        for i in range(len(df)):
            if i == 0:
                vol_colors.append("#9E9E9E")
            elif df.iloc[i]["close"] >= df.iloc[i - 1]["close"]:
                vol_colors.append("#EF5350")
            else:
                vol_colors.append("#26A69A")
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="成交量",
                marker_color=vol_colors,
                opacity=0.4,
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
        # 讓成交量只佔底部約 20%
        max_vol = df["volume"].dropna().max()
        if pd.notna(max_vol) and max_vol > 0:
            fig.update_yaxes(
                range=[0, max_vol * 5],
                showticklabels=False,
                row=1,
                col=1,
                secondary_y=True,
            )

    # --- MOPS 公告標記（vline）---
    if df_announcements is not None and not df_announcements.empty:
        sentiment_colors = {
            1: "rgba(38,166,154,0.8)",
            -1: "rgba(239,83,80,0.8)",
            0: "rgba(158,158,158,0.6)",
        }
        for ann_date, group in df_announcements.groupby("date"):
            sentiment = int(group["sentiment"].mode().iloc[0])
            color = sentiment_colors.get(sentiment, sentiment_colors[0])
            # 使用 timestamp 毫秒確保 plotly 正確定位（datetime 軸）
            x_ms = int(pd.Timestamp(ann_date).timestamp() * 1000)
            fig.add_vline(
                x=x_ms,
                line_dash="dot",
                line_color=color,
                line_width=1.5,
            )

    # --- RSI ---
    if show_rsi and rsi_row is not None:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["rsi_14"],
                name="RSI",
                line=dict(width=1.5, color="#E91E63"),
            ),
            row=rsi_row,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=rsi_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=rsi_row, col=1)

    # --- MACD ---
    if show_macd and macd_row is not None:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["macd"],
                name="MACD",
                line=dict(width=1.5, color="#2196F3"),
            ),
            row=macd_row,
            col=1,
        )
        if "macd_signal" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["macd_signal"],
                    name="Signal",
                    line=dict(width=1.5, color="#FF9800"),
                ),
                row=macd_row,
                col=1,
            )
        if "macd_hist" in df.columns:
            hist_colors = ["#EF5350" if v >= 0 else "#26A69A" for v in df["macd_hist"].fillna(0)]
            fig.add_trace(
                go.Bar(
                    x=df["date"],
                    y=df["macd_hist"],
                    name="Histogram",
                    marker_color=hist_colors,
                ),
                row=macd_row,
                col=1,
            )

    height = 500 if n_rows == 1 else (700 if n_rows == 2 else 900)
    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    fig.update_yaxes(title_text="價格", row=1, col=1, secondary_y=False)
    if show_rsi and rsi_row is not None:
        fig.update_yaxes(title_text="RSI", row=rsi_row, col=1)
    if show_macd and macd_row is not None:
        fig.update_yaxes(title_text="MACD", row=macd_row, col=1)

    return fig


def plot_institutional(df: pd.DataFrame) -> go.Figure:
    """三大法人買賣超長條圖。"""
    if df.empty:
        return go.Figure().update_layout(title="無法人資料")

    # pivot: 每日各法人淨買賣超
    pivot = df.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0)

    fig = go.Figure()
    colors = {"Foreign_Investor": "#2196F3", "Investment_Trust": "#FF9800", "Dealer_self": "#9C27B0"}
    labels = {"Foreign_Investor": "外資", "Investment_Trust": "投信", "Dealer_self": "自營商"}

    for col in pivot.columns:
        fig.add_trace(
            go.Bar(
                x=pivot.index,
                y=pivot[col],
                name=labels.get(col, col),
                marker_color=colors.get(col, "#9E9E9E"),
            )
        )

    fig.update_layout(
        barmode="group",
        height=350,
        title="三大法人買賣超",
        yaxis_title="股數",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_institutional_cumulative(df: pd.DataFrame) -> go.Figure:
    """三大法人買賣超累積折線圖（cumsum）。"""
    if df.empty:
        return go.Figure().update_layout(title="無法人資料")

    pivot = df.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0)
    pivot = pivot.sort_index()

    colors = {"Foreign_Investor": "#2196F3", "Investment_Trust": "#FF9800", "Dealer_self": "#9C27B0"}
    labels = {"Foreign_Investor": "外資累積", "Investment_Trust": "投信累積", "Dealer_self": "自營商累積"}

    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[col].cumsum(),
                name=labels.get(col, col),
                line=dict(color=colors.get(col, "#9E9E9E"), width=2),
                mode="lines",
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=350,
        title="三大法人累積買賣超",
        yaxis_title="累積股數",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_margin(df: pd.DataFrame) -> go.Figure:
    """融資融券餘額趨勢圖（含券資比）。"""
    if df.empty:
        return go.Figure().update_layout(title="無融資融券資料")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
        subplot_titles=("融資 / 融券餘額", "券資比 (%)"),
        specs=[[{"secondary_y": True}], [{}]],
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["margin_balance"],
            name="融資餘額",
            line=dict(color="#EF5350"),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["short_balance"],
            name="融券餘額",
            line=dict(color="#26A69A"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # 券資比（融券餘額 / 融資餘額）
    ratio = df["short_balance"] / df["margin_balance"].replace(0, pd.NA) * 100
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=ratio,
            name="券資比 (%)",
            line=dict(color="#FF9800", width=1.5),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=450,
        title="融資融券走勢",
        margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="融資餘額", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="融券餘額", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="券資比 %", row=2, col=1)
    return fig


# ------------------------------------------------------------------ #
#  回測圖表
# ------------------------------------------------------------------ #


def plot_equity_curve(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float = 1_000_000,
) -> go.Figure:
    """從交易明細 + 日K線還原權益曲線。"""
    if prices.empty:
        return go.Figure().update_layout(title="無價格資料")

    dates = prices["date"].tolist()
    closes = prices["close"].tolist()

    capital = initial_capital
    position = 0
    entry_price = 0.0
    equity = []

    # 建立交易事件 lookup
    buy_dates = {}
    sell_dates = {}
    if not trades.empty:
        for _, t in trades.iterrows():
            buy_dates[pd.Timestamp(t["entry_date"]).date()] = t["entry_price"]
            if pd.notna(t["exit_date"]):
                sell_dates[pd.Timestamp(t["exit_date"]).date()] = t["exit_price"]

    for i, dt in enumerate(dates):
        dt_date = pd.Timestamp(dt).date() if not isinstance(dt, pd.Timestamp) else dt.date()
        close = closes[i]

        if dt_date in buy_dates and position == 0:
            ep = buy_dates[dt_date]
            shares = int(capital * 0.998 // ep)
            if shares > 0:
                capital -= shares * ep * 1.001425
                position = shares
                entry_price = ep

        elif dt_date in sell_dates and position > 0:
            sp = sell_dates[dt_date]
            revenue = position * sp * (1 - 0.001425 - 0.003)
            capital += revenue
            position = 0

        equity.append(capital + position * close)

    # 計算回撤
    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    drawdown = (peak - eq_arr) / peak * 100

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("權益曲線", "回撤 (%)"),
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity,
            name="權益",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.1)",
        ),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="初始資金",
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=-drawdown,
            name="回撤",
            line=dict(color="#EF5350", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.15)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=500,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=False,
    )
    fig.update_yaxes(title_text="資金", row=1, col=1)
    fig.update_yaxes(title_text="回撤 %", row=2, col=1)

    return fig


# ------------------------------------------------------------------ #
#  投資組合圖表
# ------------------------------------------------------------------ #


def plot_portfolio_equity(equity_curve: list[float], dates: list, initial_capital: float = 1_000_000) -> go.Figure:
    """投資組合權益曲線。"""
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    drawdown = (peak - eq_arr) / peak * 100

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("組合權益曲線", "回撤 (%)"),
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity_curve,
            name="權益",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.1)",
        ),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="初始資金",
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=-drawdown,
            name="回撤",
            line=dict(color="#EF5350", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.15)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=500,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=False,
    )
    fig.update_yaxes(title_text="資金", row=1, col=1)
    fig.update_yaxes(title_text="回撤 %", row=2, col=1)

    return fig


def plot_allocation_pie(stock_ids: list[str], weights: dict[str, float] | None = None) -> go.Figure:
    """投資組合配置圓餅圖。"""
    if weights:
        labels = list(weights.keys())
        values = list(weights.values())
    else:
        # equal weight
        n = len(stock_ids)
        labels = stock_ids
        values = [1.0 / n] * n

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                textinfo="label+percent",
                hole=0.3,
            )
        ]
    )

    fig.update_layout(
        title="資金配置比例",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_per_stock_returns(per_stock_returns: dict[str, float]) -> go.Figure:
    """個股報酬柱狀圖。"""
    stocks = list(per_stock_returns.keys())
    returns = list(per_stock_returns.values())

    colors = ["#26A69A" if r >= 0 else "#EF5350" for r in returns]

    fig = go.Figure(
        data=[
            go.Bar(
                x=stocks,
                y=returns,
                marker_color=colors,
                text=[f"{r:+.2f}%" for r in returns],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="個股報酬貢獻",
        yaxis_title="報酬率 (%)",
        height=350,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


# ------------------------------------------------------------------ #
#  Discover 推薦歷史圖表
# ------------------------------------------------------------------ #


def plot_discovery_calendar_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    colorscale: str = "Blues",
) -> go.Figure:
    """日曆熱圖 — 以「年月 × 月內第幾週」矩陣呈現。

    Parameters
    ----------
    df : DataFrame
        需含 scan_date 欄（date 型別）和 value_col 欄位
    value_col : str
        熱圖顯示的數值欄位名稱
    title : str
        圖表標題
    colorscale : str
        Plotly 色階名稱
    """
    if df.empty:
        return go.Figure().update_layout(title=f"{title}（無資料）")

    df = df.copy()
    df["scan_date"] = pd.to_datetime(df["scan_date"])
    df["year_month"] = df["scan_date"].dt.to_period("M").astype(str)
    df["day"] = df["scan_date"].dt.day
    # 月內第幾週（1~5）
    df["week_of_month"] = (df["day"] - 1) // 7 + 1

    # 聚合：同一年月 + 同一週可能有多個值
    pivot = df.pivot_table(
        index="year_month",
        columns="week_of_month",
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.sort_index()

    # 格式化 hover text
    if "return" in value_col.lower() or "avg" in value_col.lower():
        text = pivot.map(lambda v: f"{v:+.2%}" if pd.notna(v) else "")
    else:
        text = pivot.map(lambda v: f"{v:.0f}" if pd.notna(v) else "")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"第{w}週" for w in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=colorscale,
            text=text.values,
            texttemplate="%{text}",
            hovertemplate="月份: %{y}<br>%{x}<br>值: %{text}<extra></extra>",
            colorbar=dict(title=value_col),
        )
    )
    fig.update_layout(
        title=title,
        height=max(250, len(pivot) * 30 + 100),
        margin=dict(l=100, r=20, t=40, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_discovery_monthly_stats(
    counts_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> go.Figure:
    """月度統計柱狀圖（推薦數柱 + 勝率折線，雙 Y 軸）。"""
    if counts_df.empty:
        return go.Figure().update_layout(title="月度統計（無資料）")

    counts_df = counts_df.copy()
    counts_df["scan_date"] = pd.to_datetime(counts_df["scan_date"])
    counts_df["month"] = counts_df["scan_date"].dt.to_period("M").astype(str)
    monthly_counts = counts_df.groupby("month")["count"].sum().reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=monthly_counts["month"],
            y=monthly_counts["count"],
            name="推薦數",
            marker_color="#42A5F5",
        ),
        secondary_y=False,
    )

    if not returns_df.empty:
        returns_df = returns_df.copy()
        returns_df["scan_date"] = pd.to_datetime(returns_df["scan_date"])
        returns_df["month"] = returns_df["scan_date"].dt.to_period("M").astype(str)
        monthly_returns = (
            returns_df.groupby("month")
            .apply(
                lambda g: (g["avg_return"] > 0).sum() / len(g) if len(g) > 0 else 0,
                include_groups=False,
            )
            .reset_index(name="win_rate")
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_returns["month"],
                y=monthly_returns["win_rate"],
                name="勝率",
                mode="lines+markers",
                line=dict(color="#FF7043", width=2),
                marker=dict(size=6),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="勝率", tickformat=".0%", secondary_y=True)

    fig.update_layout(
        title="月度推薦統計",
        height=350,
        margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="推薦數", secondary_y=False)
    return fig


def plot_discovery_return_boxplot(
    detail_df: pd.DataFrame,
    holding_days: list[int],
) -> go.Figure:
    """報酬率箱型圖（三組 Box 並列，含盈虧平衡線）。"""
    if detail_df.empty:
        return go.Figure().update_layout(title="報酬率分布（無資料）")

    fig = go.Figure()
    colors = {5: "#42A5F5", 10: "#66BB6A", 20: "#FFA726"}

    for days in holding_days:
        col = f"return_{days}d"
        if col not in detail_df.columns:
            continue
        values = detail_df[col].dropna() * 100  # 轉為百分比
        fig.add_trace(
            go.Box(
                y=values,
                name=f"{days}天",
                marker_color=colors.get(days, "#9E9E9E"),
                boxmean=True,
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="盈虧平衡")
    fig.update_layout(
        title="持有期報酬率分布",
        yaxis_title="報酬率 (%)",
        height=400,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def plot_discovery_cumulative_return(
    by_scan_df: pd.DataFrame,
    holding_days: int,
) -> go.Figure:
    """累積報酬率時間序列（折線 + 填充）。"""
    if by_scan_df.empty:
        return go.Figure().update_layout(title="累積報酬率（無資料）")

    df = by_scan_df[by_scan_df["holding_days"] == holding_days].copy()
    if df.empty:
        return go.Figure().update_layout(title=f"累積報酬率（{holding_days}天，無資料）")

    df = df.sort_values("scan_date").reset_index(drop=True)
    df["cumulative"] = (1 + df["avg_return"]).cumprod() - 1

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["scan_date"],
            y=df["cumulative"] * 100,
            mode="lines",
            name=f"累積報酬（{holding_days}天）",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.1)",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"累積報酬率（持有 {holding_days} 天）",
        yaxis_title="累積報酬率 (%)",
        xaxis_title="掃描日期",
        height=350,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def plot_discovery_winrate_scatter(
    by_scan_df: pd.DataFrame,
    holding_days: int,
) -> go.Figure:
    """勝率 vs 報酬率散佈圖（點大小 = 推薦數，顏色 = 正負）。"""
    if by_scan_df.empty:
        return go.Figure().update_layout(title="勝率 vs 報酬率（無資料）")

    df = by_scan_df[by_scan_df["holding_days"] == holding_days].copy()
    if df.empty:
        return go.Figure().update_layout(title=f"勝率 vs 報酬率（{holding_days}天，無資料）")

    colors = ["#26A69A" if r > 0 else "#EF5350" for r in df["avg_return"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["win_rate"] * 100,
            y=df["avg_return"] * 100,
            mode="markers",
            marker=dict(
                size=df["count"] * 2 + 5,
                color=colors,
                line=dict(width=1, color="white"),
                opacity=0.7,
            ),
            text=df["scan_date"].astype(str),
            hovertemplate="日期: %{text}<br>勝率: %{x:.1f}%<br>平均報酬: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"勝率 vs 平均報酬率（持有 {holding_days} 天）",
        xaxis_title="勝率 (%)",
        yaxis_title="平均報酬率 (%)",
        height=400,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def plot_stock_frequency_bar(
    freq_df: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:
    """推薦頻率 Top N 橫向柱狀圖。"""
    if freq_df.empty:
        return go.Figure().update_layout(title="推薦頻率排行（無資料）")

    df = freq_df.head(top_n).sort_values("recommend_count")
    labels = df["stock_id"] + " " + df["stock_name"].fillna("")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["recommend_count"],
            y=labels,
            orientation="h",
            marker_color="#42A5F5",
            text=df["recommend_count"],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"推薦頻率 Top {top_n}",
        xaxis_title="推薦次數",
        height=max(350, len(df) * 25 + 100),
        margin=dict(l=120, r=40, t=40, b=40),
    )
    return fig


# ------------------------------------------------------------------ #
#  市場總覽圖表
# ------------------------------------------------------------------ #


def plot_taiex_regime(df: pd.DataFrame, regime_info: dict) -> go.Figure:
    """TAIEX K 線 + SMA60/SMA120（上圖）+ 20 日報酬率（下圖）。

    Parameters
    ----------
    df : DataFrame
        需含 date, open, high, low, close 欄位
    regime_info : dict
        MarketRegimeDetector.detect() 的回傳結果
    """
    if df.empty:
        return go.Figure().update_layout(title="TAIEX 走勢（無資料）")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("TAIEX 加權指數", "20 日報酬率 (%)"),
    )

    # K 線
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="TAIEX",
            increasing_line_color="#EF5350",
            decreasing_line_color="#26A69A",
            increasing_fillcolor="#EF5350",
            decreasing_fillcolor="#26A69A",
        ),
        row=1,
        col=1,
    )

    # SMA60 / SMA120
    closes = df["close"]
    for window, color, label in [
        (60, "#FF9800", "SMA60"),
        (120, "#9C27B0", "SMA120"),
    ]:
        if len(closes) >= window:
            sma = closes.rolling(window).mean()
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=sma,
                    name=label,
                    line=dict(width=1.5, color=color),
                ),
                row=1,
                col=1,
            )

    # 20 日報酬率
    if len(closes) > 20:
        ret_20d = closes.pct_change(20) * 100
        colors = ["#EF5350" if v >= 0 else "#26A69A" for v in ret_20d.fillna(0)]
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=ret_20d,
                name="20日報酬率",
                marker_color=colors,
            ),
            row=2,
            col=1,
        )
        # ±3% 警戒線
        fig.add_hline(y=3, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=-3, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # Regime 標註
    regime = regime_info.get("regime", "sideways")
    regime_labels = {"bull": "多頭", "bear": "空頭", "sideways": "盤整"}
    regime_colors = {"bull": "#26A69A", "bear": "#EF5350", "sideways": "#FF9800"}

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
        title=dict(
            text=f"市場狀態: {regime_labels.get(regime, regime)}",
            font=dict(color=regime_colors.get(regime, "#333")),
        ),
    )
    fig.update_yaxes(title_text="指數", row=1, col=1)
    fig.update_yaxes(title_text="報酬率 %", row=2, col=1)
    return fig


def plot_market_breadth_area(df: pd.DataFrame) -> go.Figure:
    """漲跌家數堆疊面積圖。"""
    if df.empty:
        return go.Figure().update_layout(title="市場廣度（無資料）")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["rising"],
            name="上漲",
            mode="lines",
            line=dict(width=0.5, color="#EF5350"),
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.4)",
            stackgroup="one",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["flat"],
            name="平盤",
            mode="lines",
            line=dict(width=0.5, color="#9E9E9E"),
            fill="tonexty",
            fillcolor="rgba(158,158,158,0.3)",
            stackgroup="one",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["falling"],
            name="下跌",
            mode="lines",
            line=dict(width=0.5, color="#26A69A"),
            fill="tonexty",
            fillcolor="rgba(38,166,154,0.4)",
            stackgroup="one",
        )
    )

    fig.update_layout(
        title="每日漲跌家數",
        height=400,
        yaxis_title="家數",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_institutional_ranking(df: pd.DataFrame) -> go.Figure:
    """法人買賣超排行（橫向堆疊柱狀圖）。"""
    if df.empty:
        return go.Figure().update_layout(title="法人買賣超排行（無資料）")

    labels = df["stock_id"] + " " + df["stock_name"].fillna("")

    fig = go.Figure()
    for col_name, color, label in [
        ("foreign", "#2196F3", "外資"),
        ("trust", "#FF9800", "投信"),
        ("dealer", "#9C27B0", "自營商"),
    ]:
        if col_name in df.columns:
            fig.add_trace(
                go.Bar(
                    y=labels,
                    x=df[col_name],
                    name=label,
                    orientation="h",
                    marker_color=color,
                )
            )

    fig.update_layout(
        barmode="relative",
        height=max(350, len(df) * 35 + 100),
        title="法人買賣超排行",
        xaxis_title="淨買賣超（股）",
        margin=dict(l=120, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_sector_treemap(sector_df: pd.DataFrame) -> go.Figure:
    """產業熱度 Treemap（面積=淨買超絕對值，顏色=漲幅）。

    Parameters
    ----------
    sector_df : DataFrame
        需含 industry, total_net, avg_return_pct 欄位
        （來自 IndustryRotationAnalyzer.rank_sectors()）
    """
    if sector_df.empty:
        return go.Figure().update_layout(title="產業熱度（無資料）")

    df = sector_df.copy()
    df["abs_net"] = df["total_net"].abs()
    # 過濾掉零值
    df = df[df["abs_net"] > 0].reset_index(drop=True)
    if df.empty:
        return go.Figure().update_layout(title="產業熱度（無資料）")

    fig = go.Figure(
        go.Treemap(
            labels=df["industry"],
            parents=[""] * len(df),
            values=df["abs_net"],
            marker=dict(
                colors=df["avg_return_pct"],
                colorscale="RdYlGn",
                cmid=0,
                colorbar=dict(title="漲幅 %"),
            ),
            texttemplate="<b>%{label}</b><br>淨買超: %{value:,.0f}<br>漲幅: %{color:.2f}%",
            hovertemplate="<b>%{label}</b><br>淨買超: %{value:,.0f}<br>漲幅: %{color:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="產業熱度 Treemap",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
