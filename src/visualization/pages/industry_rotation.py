"""Streamlit 產業輪動分析頁面。"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render():
    st.header("🏭 產業輪動分析")

    # --- Sidebar 控制 ---
    st.sidebar.markdown("### 產業輪動參數")
    lookback_days = st.sidebar.slider("法人回溯天數", 5, 60, 20)
    momentum_days = st.sidebar.slider("價格動能天數", 20, 120, 60)
    top_sectors = st.sidebar.selectbox("顯示產業數", [3, 5, 8, 10], index=1)
    top_n = st.sidebar.selectbox("每產業精選股數", [3, 5, 10], index=1)

    if st.sidebar.button("🔄 同步 StockInfo"):
        with st.spinner("正在同步股票基本資料..."):
            from src.data.pipeline import sync_stock_info

            count = sync_stock_info(force_refresh=True)
            st.sidebar.success(f"已同步 {count} 筆")

    # --- 分析 ---
    if st.button("開始分析", type="primary"):
        _run_analysis(lookback_days, momentum_days, top_sectors, top_n)
    else:
        st.info("點擊「開始分析」執行產業輪動分析")


def _run_analysis(lookback_days, momentum_days, top_sectors, top_n):
    from src.data.pipeline import sync_stock_info
    from src.industry.analyzer import IndustryRotationAnalyzer

    # 確保 StockInfo 存在
    sync_stock_info(force_refresh=False)

    with st.spinner("正在分析產業輪動..."):
        analyzer = IndustryRotationAnalyzer(
            lookback_days=lookback_days,
            momentum_days=momentum_days,
        )
        sector_df = analyzer.rank_sectors()

    if sector_df.empty:
        st.warning("無法計算產業排名（資料不足，請先同步資料）")
        return

    tab1, tab2 = st.tabs(["📊 產業排名總覽", "🔥 精選個股"])

    with tab1:
        _render_sector_overview(sector_df, top_sectors, analyzer)

    with tab2:
        _render_top_stocks(sector_df, analyzer, top_sectors, top_n)


def _render_sector_overview(sector_df, top_sectors, analyzer):
    """產業排名總覽 Tab。"""
    display = sector_df.head(top_sectors)

    # 排名表格
    st.subheader("產業綜合排名")
    st.dataframe(
        display.style.format(
            {
                "sector_score": "{:.3f}",
                "institutional_score": "{:.3f}",
                "momentum_score": "{:.3f}",
                "total_net": "{:,.0f}",
                "avg_return_pct": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # 泡泡圖
        st.subheader("法人動能 vs 價格動能")
        if len(sector_df) > 1:
            fig = px.scatter(
                sector_df.head(top_sectors),
                x="avg_return_pct",
                y="total_net",
                size="stock_count",
                color="industry",
                hover_name="industry",
                labels={
                    "avg_return_pct": "平均漲幅 (%)",
                    "total_net": "法人淨買超",
                    "stock_count": "股票數",
                },
                size_max=40,
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("產業數不足，無法繪製泡泡圖")

    with col2:
        # 法人淨買超長條圖
        st.subheader("法人淨買超（按產業）")
        chart_data = sector_df.head(top_sectors).sort_values("total_net")
        fig = go.Figure(
            go.Bar(
                x=chart_data["total_net"],
                y=chart_data["industry"],
                orientation="h",
                marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in chart_data["total_net"]],
            )
        )
        fig.update_layout(
            height=400,
            xaxis_title="淨買超金額",
            yaxis_title="產業",
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_top_stocks(sector_df, analyzer, top_sectors, top_n):
    """精選個股 Tab。"""
    with st.spinner("正在篩選精選個股..."):
        top_stocks = analyzer.top_stocks_from_hot_sectors(sector_df, top_sectors=top_sectors, top_n=top_n)

    if top_stocks.empty:
        st.warning("無精選個股資料")
        return

    for ind in top_stocks["industry"].unique():
        sector_stocks = top_stocks[top_stocks["industry"] == ind]
        with st.expander(f"📁 {ind} ({len(sector_stocks)} 檔)", expanded=True):
            display_cols = ["stock_id", "stock_name", "close", "foreign_net_sum", "rank_in_sector"]
            available = [c for c in display_cols if c in sector_stocks.columns]
            st.dataframe(
                sector_stocks[available].style.format(
                    {
                        "close": "{:.1f}",
                        "foreign_net_sum": "{:,.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
