"""個股分析頁面 — K線、技術指標、法人、融資券、MOPS 公告。"""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from src.visualization.charts import (
    plot_candlestick,
    plot_institutional,
    plot_institutional_cumulative,
    plot_margin,
)
from src.visualization.data_loader import (
    get_stock_list,
    load_announcements,
    load_institutional,
    load_margin,
    load_price_with_indicators,
)


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

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 技術指標")
    show_sma5 = st.sidebar.checkbox("SMA 5", value=True)
    show_sma10 = st.sidebar.checkbox("SMA 10", value=True)
    show_sma20 = st.sidebar.checkbox("SMA 20", value=True)
    show_sma60 = st.sidebar.checkbox("SMA 60", value=True)
    show_bb = st.sidebar.checkbox("布林通道", value=True)
    show_rsi = st.sidebar.checkbox("RSI", value=True)
    show_macd = st.sidebar.checkbox("MACD", value=True)

    selected_indicators: set[str] = set()
    if show_sma5:
        selected_indicators.add("sma_5")
    if show_sma10:
        selected_indicators.add("sma_10")
    if show_sma20:
        selected_indicators.add("sma_20")
    if show_sma60:
        selected_indicators.add("sma_60")
    if show_bb:
        selected_indicators.add("bb")
    if show_rsi:
        selected_indicators.add("rsi")
    if show_macd:
        selected_indicators.add("macd")

    start_str = start.isoformat()
    end_str = end.isoformat()

    # --- 載入資料 ---
    df = load_price_with_indicators(stock_id, start_str, end_str)
    if df.empty:
        st.warning(f"{stock_id} 在選定期間內無資料")
        return

    df_ann = load_announcements(stock_id, start_str, end_str)

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

    # --- K線 + 指標圖（含公告標記）---
    ann_df = df_ann if not df_ann.empty else None
    fig = plot_candlestick(df, selected_indicators, ann_df)
    st.plotly_chart(fig, use_container_width=True)

    # --- 法人每日買賣超 | 法人累積買賣超 ---
    df_inst = load_institutional(stock_id, start_str, end_str)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_institutional(df_inst), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_institutional_cumulative(df_inst), use_container_width=True)

    # --- 融資融券走勢（含券資比）---
    df_margin = load_margin(stock_id, start_str, end_str)
    st.plotly_chart(plot_margin(df_margin), use_container_width=True)

    # --- MOPS 公告明細 ---
    if not df_ann.empty:
        with st.expander(f"📢 MOPS 重大訊息公告（{len(df_ann)} 則）", expanded=False):
            display_df = df_ann[["date", "subject", "sentiment", "spoke_time"]].copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df["sentiment"] = display_df["sentiment"].map({1: "正面 ✅", -1: "負面 ❌", 0: "中性 ➖"})
            st.dataframe(display_df, use_container_width=True)
