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
    [
        "🏠 市場總覽",
        "📈 個股分析",
        "🔄 回測結果",
        "📊 投資組合",
        "⚖️ 策略比較",
        "🔍 選股篩選",
        "🤖 ML 策略分析",
        "🏭 產業輪動",
        "🔖 概念輪動",
        "📋 推薦歷史",
        "👁️ 持倉監控",
    ],
)

if page == "🏠 市場總覽":
    from src.visualization.pages.market_overview import render

    render()
elif page == "📈 個股分析":
    from src.visualization.pages.stock_analysis import render

    render()
elif page == "🔄 回測結果":
    from src.visualization.pages.backtest_review import render

    render()
elif page == "📊 投資組合":
    from src.visualization.pages.portfolio_review import render

    render()
elif page == "⚖️ 策略比較":
    from src.visualization.pages.strategy_comparison import render

    render()
elif page == "🔍 選股篩選":
    from src.visualization.pages.screener_results import render

    render()
elif page == "🤖 ML 策略分析":
    from src.visualization.pages.ml_analysis import render

    render()
elif page == "🏭 產業輪動":
    from src.visualization.pages.industry_rotation import render

    render()
elif page == "🔖 概念輪動":
    from src.visualization.pages.concept_rotation import render

    render()
elif page == "📋 推薦歷史":
    from src.visualization.pages.discovery_history import render

    render()
elif page == "👁️ 持倉監控":
    from src.visualization.pages.position_monitoring import render

    render()
