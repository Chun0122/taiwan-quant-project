"""個股分析頁面 — K線、技術指標、法人、融資券。"""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from src.visualization.data_loader import (
    get_stock_list, load_institutional, load_margin, load_price_with_indicators,
)
from src.visualization.charts import plot_candlestick, plot_institutional, plot_margin


def render() -> None:
    st.title("📈 個股分析")

    stocks = get_stock_list()
    if not stocks:
        st.warning("資料庫中尚無股票資料，請先執行 `python main.py sync`")
        return

    # --- 側欄控制 ---
    stock_id = st.sidebar.selectbox("股票代號", stocks, index=0)
    col1, col2 = st.sidebar.columns(2)
    start = col1.date_input("起始日", value=date.today() - timedelta(days=365))
    end = col2.date_input("結束日", value=date.today())

    start_str = start.isoformat()
    end_str = end.isoformat()

    # --- 載入資料 ---
    df = load_price_with_indicators(stock_id, start_str, end_str)
    if df.empty:
        st.warning(f"{stock_id} 在選定期間內無資料")
        return

    # --- 最新報價 ---
    latest = df.iloc[-1]
    prev_close = df.iloc[-2]["close"] if len(df) > 1 else latest["close"]
    change = latest["close"] - prev_close
    change_pct = change / prev_close * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("收盤價", f"{latest['close']:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    m2.metric("最高", f"{latest['high']:.2f}")
    m3.metric("最低", f"{latest['low']:.2f}")
    m4.metric("成交量", f"{latest['volume']:,.0f}")

    # --- K線 + 指標圖 ---
    fig = plot_candlestick(df)
    st.plotly_chart(fig, width="stretch")

    # --- 法人 + 融資券 ---
    col_a, col_b = st.columns(2)

    with col_a:
        df_inst = load_institutional(stock_id, start_str, end_str)
        fig_inst = plot_institutional(df_inst)
        st.plotly_chart(fig_inst, width="stretch")

    with col_b:
        df_margin = load_margin(stock_id, start_str, end_str)
        fig_margin = plot_margin(df_margin)
        st.plotly_chart(fig_margin, width="stretch")
