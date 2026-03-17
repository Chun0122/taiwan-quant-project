"""Streamlit 概念輪動分析頁面。"""

import plotly.express as px
import streamlit as st


def render():
    st.header("🔖 概念輪動分析")

    # --- Sidebar 控制 ---
    st.sidebar.markdown("### 概念輪動參數")
    lookback_days = st.sidebar.slider("法人/動能回溯天數", 5, 60, 20)
    momentum_days = st.sidebar.slider("價格動能參考天數", 20, 120, 60)
    top_concepts = st.sidebar.selectbox("顯示概念數", [3, 5, 8, 10], index=1)

    if st.sidebar.button("🔄 同步概念定義"):
        with st.spinner("正在從 concepts.yaml 同步概念..."):
            from src.data.pipeline import sync_concepts_from_yaml

            stats = sync_concepts_from_yaml()
            st.sidebar.success(f"新增概念 {stats['groups']} 個，成員 {stats['members']} 筆")

    # --- 分析 ---
    if st.button("開始分析", type="primary"):
        _run_analysis(lookback_days, momentum_days, top_concepts)
    else:
        st.info("點擊「開始分析」執行概念輪動分析（需先執行 sync-concepts 匯入概念定義）")


def _run_analysis(lookback_days: int, momentum_days: int, top_concepts: int):
    from src.industry.concept_analyzer import ConceptRotationAnalyzer

    with st.spinner("正在分析概念輪動..."):
        analyzer = ConceptRotationAnalyzer(
            lookback_days=lookback_days,
            momentum_days=momentum_days,
        )
        ranked = analyzer.rank_concepts()

    if ranked.empty:
        st.warning("無法計算概念排名（DB 無概念定義，請先執行 sync-concepts）")
        return

    tab1, tab2, tab3 = st.tabs(["📊 概念排名", "🗺️ 成員熱力圖", "📦 報酬率箱型圖"])

    with tab1:
        _render_concept_ranking(ranked, top_concepts)

    with tab2:
        _render_concept_heatmap(ranked, analyzer, top_concepts)

    with tab3:
        _render_concept_return_box(analyzer, top_concepts, ranked)


def _render_concept_ranking(ranked, top_concepts: int):
    """概念排名總覽 Tab。"""
    st.subheader("概念熱度排名")

    display = ranked.head(top_concepts).copy()
    display["concept_score"] = display["concept_score"].round(4)
    display["institutional_score"] = display["institutional_score"].round(4)
    display["momentum_score"] = display["momentum_score"].round(4)

    st.dataframe(
        display.rename(
            columns={
                "rank": "排名",
                "concept": "概念名稱",
                "concept_score": "綜合分數",
                "institutional_score": "法人分數",
                "momentum_score": "動能分數",
                "member_count": "成員股數",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # 橫條圖
    fig = px.bar(
        display,
        x="concept_score",
        y="concept",
        orientation="h",
        color="concept_score",
        color_continuous_scale="RdYlGn",
        labels={"concept_score": "綜合分數", "concept": "概念"},
        title=f"Top {min(top_concepts, len(display))} 概念熱度",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=300)
    st.plotly_chart(fig, use_container_width=True)


def _render_concept_heatmap(ranked, analyzer, top_concepts: int):
    """概念成員熱力圖（Treemap）。"""
    st.subheader("概念成員分布（Treemap）")

    concept_stocks = analyzer.get_concept_stocks()
    top_list = ranked.head(top_concepts)["concept"].tolist()

    records = []
    for concept in top_list:
        score_row = ranked[ranked["concept"] == concept]
        c_score = float(score_row["concept_score"].iloc[0]) if not score_row.empty else 0.5
        for stock in concept_stocks.get(concept, []):
            records.append({"concept": concept, "stock_id": stock, "concept_score": c_score})

    if not records:
        st.info("無成員資料")
        return

    import pandas as pd

    df = pd.DataFrame(records)
    fig = px.treemap(
        df,
        path=["concept", "stock_id"],
        values=[1] * len(df),
        color="concept_score",
        color_continuous_scale="RdYlGn",
        title="概念股成員分布（顏色=概念熱度）",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


def _render_concept_return_box(analyzer, top_concepts: int, ranked):
    """各概念近期報酬率箱型圖。"""
    st.subheader("各概念成員近期報酬率分布")

    import pandas as pd

    concept_stocks = analyzer.get_concept_stocks()
    top_list = ranked.head(top_concepts)["concept"].tolist()
    all_stocks = list({s for c in top_list for s in concept_stocks.get(c, [])})

    if not all_stocks:
        st.info("無成員資料")
        return

    df_price = analyzer._load_price_data(all_stocks, analyzer.lookback_days)
    if df_price.empty:
        st.info("無價格資料（請先同步 DailyPrice）")
        return

    df_price = df_price.sort_values(["stock_id", "date"])
    returns_map: dict[str, float] = {}
    for sid, grp in df_price.groupby("stock_id"):
        grp = grp.tail(analyzer.lookback_days + 1)
        if len(grp) < 2:
            continue
        first_c = grp["close"].iloc[0]
        last_c = grp["close"].iloc[-1]
        if first_c > 0:
            returns_map[str(sid)] = (last_c / first_c - 1) * 100

    box_records = []
    for concept in top_list:
        for stock in concept_stocks.get(concept, []):
            if stock in returns_map:
                box_records.append({"concept": concept, "return_pct": returns_map[stock]})

    if not box_records:
        st.info("無法計算報酬率（DailyPrice 資料不足）")
        return

    df_box = pd.DataFrame(box_records)
    fig = px.box(
        df_box,
        x="concept",
        y="return_pct",
        color="concept",
        title=f"各概念近 {analyzer.lookback_days} 日成員報酬率分布（%）",
        labels={"return_pct": "報酬率 (%)", "concept": "概念"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
