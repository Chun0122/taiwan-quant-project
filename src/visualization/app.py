"""台股量化投資系統 — Streamlit 儀表板主入口。"""

import streamlit as st

st.set_page_config(
    page_title="台股量化投資系統",
    page_icon="📈",
    layout="wide",
)

# --- 頁面路由 ---
page = st.sidebar.radio(
    "功能選單",
    ["📈 個股分析", "🔄 回測結果", "📊 投資組合", "🔍 選股篩選", "🤖 ML 策略分析"],
)

if page == "📈 個股分析":
    from src.visualization.pages.stock_analysis import render
    render()
elif page == "🔄 回測結果":
    from src.visualization.pages.backtest_review import render
    render()
elif page == "📊 投資組合":
    from src.visualization.pages.portfolio_review import render
    render()
elif page == "🔍 選股篩選":
    from src.visualization.pages.screener_results import render
    render()
elif page == "🤖 ML 策略分析":
    from src.visualization.pages.ml_analysis import render
    render()
