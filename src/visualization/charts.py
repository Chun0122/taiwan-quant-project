"""Plotly 圖表元件 — 提供各類股票分析圖表。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------ #
#  個股分析圖表
# ------------------------------------------------------------------ #


def plot_candlestick(df: pd.DataFrame) -> go.Figure:
    """K線圖 + SMA + 布林通道 + 成交量 + RSI + MACD 四合一圖。"""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        vertical_spacing=0.03,
        subplot_titles=("K線 / 均線 / 布林通道", "RSI (14)", "MACD", "成交量"),
    )

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
    )

    # SMA
    sma_colors = {"sma_5": "#FF9800", "sma_10": "#2196F3", "sma_20": "#9C27B0", "sma_60": "#795548"}
    for col, color in sma_colors.items():
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[col],
                    name=col.upper(),
                    line=dict(width=1, color=color),
                ),
                row=1,
                col=1,
            )

    # 布林通道
    if "bb_upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["bb_upper"],
                name="BB Upper",
                line=dict(width=1, color="rgba(150,150,150,0.5)", dash="dot"),
            ),
            row=1,
            col=1,
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
        )

    # --- Row 2: RSI ---
    if "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["rsi_14"],
                name="RSI",
                line=dict(width=1.5, color="#E91E63"),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # --- Row 3: MACD ---
    if "macd" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["macd"],
                name="MACD",
                line=dict(width=1.5, color="#2196F3"),
            ),
            row=3,
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
                row=3,
                col=1,
            )
        if "macd_hist" in df.columns:
            colors = ["#EF5350" if v >= 0 else "#26A69A" for v in df["macd_hist"].fillna(0)]
            fig.add_trace(
                go.Bar(
                    x=df["date"],
                    y=df["macd_hist"],
                    name="Histogram",
                    marker_color=colors,
                ),
                row=3,
                col=1,
            )

    # --- Row 4: 成交量 ---
    if "volume" in df.columns and "close" in df.columns:
        colors = []
        for i in range(len(df)):
            if i == 0:
                colors.append("#9E9E9E")
            elif df.iloc[i]["close"] >= df.iloc[i - 1]["close"]:
                colors.append("#EF5350")
            else:
                colors.append("#26A69A")
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="成交量",
                marker_color=colors,
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        height=900,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="量", row=4, col=1)

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


def plot_margin(df: pd.DataFrame) -> go.Figure:
    """融資融券餘額趨勢圖。"""
    if df.empty:
        return go.Figure().update_layout(title="無融資融券資料")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["margin_balance"],
            name="融資餘額",
            line=dict(color="#EF5350"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["short_balance"],
            name="融券餘額",
            line=dict(color="#26A69A"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        height=350,
        title="融資融券餘額",
        margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="融資餘額", secondary_y=False)
    fig.update_yaxes(title_text="融券餘額", secondary_y=True)
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
