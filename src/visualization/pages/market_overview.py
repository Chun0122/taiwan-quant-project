"""Streamlit 市場總覽首頁。

提供 TAIEX 走勢 + Regime 狀態、市場廣度指標、法人買賣超排名、產業熱度圖。
"""

import streamlit as st


def render():
    st.header("🏠 市場總覽")

    # --- Sidebar 控制 ---
    st.sidebar.markdown("### 市場總覽參數")
    lookback_days = st.sidebar.slider("法人回溯天數", 3, 20, 5, key="mo_lookback")
    momentum_days = st.sidebar.slider("產業動能天數", 20, 120, 60, key="mo_momentum")
    top_n = st.sidebar.selectbox("法人排行數", [5, 10, 20], index=1, key="mo_top_n")

    if st.button("載入市場總覽", type="primary"):
        _render_overview(lookback_days, momentum_days, top_n)
    else:
        st.info("點擊「載入市場總覽」查看最新市場狀態")


def _render_overview(lookback_days: int, momentum_days: int, top_n: int):
    """主渲染函數。"""
    from datetime import datetime

    from src.visualization.charts import (
        plot_institutional_ranking,
        plot_market_breadth_area,
        plot_taiex_regime,
    )
    from src.visualization.data_loader import (
        load_market_breadth,
        load_market_volume_summary,
        load_taiex_history,
        load_top_institutional,
    )

    with st.spinner("載入市場資料..."):
        # 載入所有資料
        taiex_df = load_taiex_history(days=180)
        breadth_df = load_market_breadth(days=60)
        volume_df = load_market_volume_summary(days=60)
        inst_df = load_top_institutional(lookback=lookback_days, top_n=top_n)

        # Regime 偵測
        from src.regime.detector import MarketRegimeDetector

        regime_info = MarketRegimeDetector().detect()

    # --- 資料新鮮度檢查 ---
    if not taiex_df.empty:
        latest_date = taiex_df["date"].max()
        days_stale = (datetime.now() - latest_date).days
        if days_stale > 3:
            st.warning(
                f"TAIEX 資料最後更新: {latest_date.strftime('%Y-%m-%d')}（已 {days_stale} 天未更新），"
                "請執行 `python main.py sync` 同步最新資料。"
            )
    else:
        st.error("無 TAIEX 資料，請先執行 `python main.py sync --taiex` 同步加權指數。")
        return

    # --- 頂部指標卡片 ---
    _render_metrics(taiex_df, regime_info, breadth_df, volume_df)

    # --- 主圖表 Tabs ---
    tab1, tab2 = st.tabs(["📈 TAIEX 走勢 + Regime", "📊 市場廣度時序"])

    with tab1:
        fig = plot_taiex_regime(taiex_df, regime_info)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = plot_market_breadth_area(breadth_df)
        st.plotly_chart(fig, use_container_width=True)

    # --- 法人排行 + 產業 Treemap ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"法人買賣超排行 Top {top_n}（近 {lookback_days} 日）")
        fig = plot_institutional_ranking(inst_df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("產業熱度 Treemap")
        _render_sector_treemap(momentum_days)

    # --- 市場廣度統計（Expander） ---
    with st.expander("▸ 市場廣度詳細統計"):
        _render_breadth_stats()


def _render_metrics(taiex_df, regime_info, breadth_df, volume_df):
    """頂部 4 欄指標卡片。"""
    col1, col2, col3, col4 = st.columns(4)

    # TAIEX 收盤價 + 漲跌幅
    current_close = taiex_df["close"].iloc[-1]
    if len(taiex_df) >= 2:
        prev_close = taiex_df["close"].iloc[-2]
        change_pct = (current_close - prev_close) / prev_close * 100
        col1.metric(
            "TAIEX 加權指數",
            f"{current_close:,.0f}",
            f"{change_pct:+.2f}%",
        )
    else:
        col1.metric("TAIEX 加權指數", f"{current_close:,.0f}")

    # Regime
    regime = regime_info.get("regime", "sideways")
    regime_labels = {"bull": "🟢 多頭", "bear": "🔴 空頭", "sideways": "🟡 盤整"}
    col2.metric("市場狀態", regime_labels.get(regime, regime))

    # 漲跌家數
    if not breadth_df.empty:
        latest = breadth_df.iloc[-1]
        col3.metric(
            "漲跌家數",
            f"{int(latest['rising'])} / {int(latest['falling'])}",
            f"平盤 {int(latest['flat'])}",
        )
    else:
        col3.metric("漲跌家數", "—")

    # 成交量
    if not volume_df.empty:
        latest_turnover = volume_df["total_turnover"].iloc[-1]
        turnover_billion = latest_turnover / 1e8  # 轉為億元
        col4.metric("成交金額", f"{turnover_billion:,.0f} 億")
    else:
        col4.metric("成交金額", "—")


def _render_sector_treemap(momentum_days: int):
    """渲染產業 Treemap。"""
    from src.visualization.charts import plot_sector_treemap

    try:
        from src.data.pipeline import sync_stock_info
        from src.industry.analyzer import IndustryRotationAnalyzer

        sync_stock_info(force_refresh=False)
        analyzer = IndustryRotationAnalyzer(
            lookback_days=20,
            momentum_days=momentum_days,
        )
        sector_df = analyzer.rank_sectors()
        if sector_df.empty:
            st.info("產業資料不足，請先同步資料")
            return
        fig = plot_sector_treemap(sector_df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"產業分析暫不可用: {e}")


def _render_breadth_stats():
    """市場廣度詳細統計表。"""
    from src.visualization.data_loader import load_market_breadth_stats

    stats = load_market_breadth_stats(windows=[1, 5, 20])
    if stats.empty:
        st.info("無市場廣度資料")
        return

    st.dataframe(
        stats.style.format(
            {
                "rising": "{:,}",
                "falling": "{:,}",
                "flat": "{:,}",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "window": "期間",
            "rising": "上漲家數",
            "falling": "下跌家數",
            "flat": "平盤家數",
        },
    )
