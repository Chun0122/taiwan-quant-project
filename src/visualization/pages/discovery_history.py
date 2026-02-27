"""Streamlit 推薦歷史頁面 — 視覺化 DiscoveryRecord 歷史推薦 + 績效追蹤。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

MODE_OPTIONS = {
    "momentum": "Momentum 短線動能",
    "swing": "Swing 中期波段",
    "value": "Value 價值修復",
    "dividend": "Dividend 高息存股",
    "growth": "Growth 高成長",
}

HOLDING_DAYS_OPTIONS = [5, 10, 20]


def render():
    st.header("📋 推薦歷史")

    # --- Sidebar 參數 ---
    st.sidebar.markdown("### 推薦歷史參數")

    mode = st.sidebar.selectbox(
        "掃描模式",
        list(MODE_OPTIONS.keys()),
        format_func=lambda k: MODE_OPTIONS[k],
    )

    date_range = st.sidebar.radio(
        "日期範圍",
        ["近 30 天", "近 90 天", "近一年", "全部", "自訂"],
        index=1,
    )

    today = date.today()
    if date_range == "近 30 天":
        start_date = (today - timedelta(days=30)).isoformat()
        end_date = today.isoformat()
    elif date_range == "近 90 天":
        start_date = (today - timedelta(days=90)).isoformat()
        end_date = today.isoformat()
    elif date_range == "近一年":
        start_date = (today - timedelta(days=365)).isoformat()
        end_date = today.isoformat()
    elif date_range == "自訂":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_dt = st.date_input("起始日", today - timedelta(days=90))
        with col2:
            end_dt = st.date_input("結束日", today)
        start_date = start_dt.isoformat()
        end_date = end_dt.isoformat()
    else:
        start_date = None
        end_date = None

    top_n = st.sidebar.slider("僅分析前 N 名", 5, 50, 20)
    holding_days = st.sidebar.selectbox(
        "持有天數（日曆/散佈圖用）",
        HOLDING_DAYS_OPTIONS,
        index=0,
    )

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📅 推薦日曆", "📊 績效分析", "🏆 個股排行", "📄 歷史明細"])

    with tab1:
        _render_calendar_tab(mode, start_date, end_date, top_n, holding_days)
    with tab2:
        _render_performance_tab(mode, start_date, end_date, top_n, holding_days)
    with tab3:
        _render_stock_ranking_tab(mode, start_date, end_date, top_n, holding_days)
    with tab4:
        _render_detail_tab(mode, start_date, end_date)


# ------------------------------------------------------------------ #
#  Tab 1：推薦日曆
# ------------------------------------------------------------------ #


def _render_calendar_tab(mode, start_date, end_date, top_n, holding_days):
    from src.visualization.charts import (
        plot_discovery_calendar_heatmap,
        plot_discovery_monthly_stats,
    )
    from src.visualization.data_loader import (
        load_discovery_calendar_counts,
        load_discovery_calendar_returns,
    )

    counts_df = load_discovery_calendar_counts(mode, start_date, end_date)

    if counts_df.empty:
        st.info(f"目前沒有 {MODE_OPTIONS[mode]} 的推薦記錄，請先執行 `python main.py discover {mode}`")
        return

    returns_df = load_discovery_calendar_returns(
        mode,
        holding_days=holding_days,
        top_n=top_n,
        start_date=start_date,
        end_date=end_date,
    )

    # 日曆熱圖：左右並排
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_discovery_calendar_heatmap(
            counts_df,
            "count",
            "推薦次數日曆",
            "Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not returns_df.empty:
            fig = plot_discovery_calendar_heatmap(
                returns_df,
                "avg_return",
                f"平均報酬率日曆（{holding_days}天）",
                "RdYlGn",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("無報酬率資料（可能尚未有足夠的後續交易日資料）")

    # 月度統計
    st.subheader("月度統計")
    fig = plot_discovery_monthly_stats(counts_df, returns_df)
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ #
#  Tab 2：績效分析
# ------------------------------------------------------------------ #


def _render_performance_tab(mode, start_date, end_date, top_n, holding_days):
    from src.visualization.charts import (
        plot_discovery_cumulative_return,
        plot_discovery_return_boxplot,
        plot_discovery_winrate_scatter,
    )
    from src.visualization.data_loader import load_discovery_performance

    with st.spinner("正在計算績效..."):
        result = load_discovery_performance(
            mode,
            holding_days=HOLDING_DAYS_OPTIONS,
            top_n=top_n,
            start_date=start_date,
            end_date=end_date,
        )

    summary = result["summary"]
    detail = result["detail"]
    by_scan = result["by_scan"]

    if summary.empty:
        st.info(f"目前沒有 {MODE_OPTIONS[mode]} 的推薦績效資料")
        return

    # 摘要卡片
    st.subheader("持有期績效摘要")
    cols = st.columns(len(HOLDING_DAYS_OPTIONS))
    for i, days in enumerate(HOLDING_DAYS_OPTIONS):
        row = summary[summary["holding_days"] == days]
        if row.empty:
            continue
        row = row.iloc[0]
        with cols[i]:
            st.markdown(f"**持有 {days} 天**")
            evaluable = int(row["evaluable"])
            if evaluable == 0:
                st.write("無可評估資料")
                continue
            c1, c2 = st.columns(2)
            with c1:
                st.metric("可評估數", evaluable)
                st.metric("勝率", f"{row['win_rate']:.1%}" if pd.notna(row["win_rate"]) else "—")
                st.metric("平均報酬", f"{row['avg_return']:+.2%}" if pd.notna(row["avg_return"]) else "—")
            with c2:
                st.metric("中位數", f"{row['median_return']:+.2%}" if pd.notna(row["median_return"]) else "—")
                st.metric("最大獲利", f"{row['max_gain']:+.2%}" if pd.notna(row["max_gain"]) else "—")
                st.metric("最大虧損", f"{row['max_loss']:+.2%}" if pd.notna(row["max_loss"]) else "—")

    st.divider()

    # 報酬率箱型圖
    if not detail.empty:
        st.subheader("報酬率分布")
        fig = plot_discovery_return_boxplot(detail, HOLDING_DAYS_OPTIONS)
        st.plotly_chart(fig, use_container_width=True)

    # 累積報酬率 + 勝率散佈圖
    if not by_scan.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_discovery_cumulative_return(by_scan, holding_days)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plot_discovery_winrate_scatter(by_scan, holding_days)
            st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ #
#  Tab 3：個股排行
# ------------------------------------------------------------------ #


def _render_stock_ranking_tab(mode, start_date, end_date, top_n, holding_days):
    from src.visualization.charts import plot_stock_frequency_bar
    from src.visualization.data_loader import (
        load_discovery_performance,
        load_discovery_stock_frequency,
    )

    freq_df = load_discovery_stock_frequency(mode, top_n=top_n, start_date=start_date, end_date=end_date)

    if freq_df.empty:
        st.info(f"目前沒有 {MODE_OPTIONS[mode]} 的推薦記錄")
        return

    # 頻率柱狀圖
    col1, col2 = st.columns([3, 2])
    with col1:
        fig = plot_stock_frequency_bar(freq_df, top_n)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("推薦頻率排行")
        display = freq_df.copy()
        display["avg_rank"] = display["avg_rank"].round(1)
        display["avg_composite_score"] = display["avg_composite_score"].round(3)
        display.columns = ["代號", "名稱", "推薦次數", "平均排名", "平均分數"]
        st.dataframe(display, use_container_width=True, hide_index=True)

    # 最佳/最差報酬個股
    st.divider()
    st.subheader(f"個股報酬排行（持有 {holding_days} 天）")

    result = load_discovery_performance(
        mode,
        holding_days=[holding_days],
        top_n=top_n,
        start_date=start_date,
        end_date=end_date,
    )
    detail = result["detail"]

    if detail.empty:
        st.info("無績效資料可計算個股排行")
        return

    col_ret = f"return_{holding_days}d"
    if col_ret not in detail.columns:
        st.info("無對應持有天數的報酬資料")
        return

    stock_stats = detail.groupby(["stock_id", "stock_name"])[col_ret].agg(["mean", "count"]).reset_index()
    stock_stats.columns = ["代號", "名稱", "平均報酬", "推薦次數"]
    stock_stats = stock_stats[stock_stats["推薦次數"] >= 1].copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**最佳報酬 Top 10**")
        best = stock_stats.nlargest(10, "平均報酬").copy()
        best["平均報酬"] = best["平均報酬"].apply(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
        st.dataframe(best, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**最差報酬 Top 10**")
        worst = stock_stats.nsmallest(10, "平均報酬").copy()
        worst["平均報酬"] = worst["平均報酬"].apply(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
        st.dataframe(worst, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ #
#  Tab 4：歷史明細
# ------------------------------------------------------------------ #


def _render_detail_tab(mode, start_date, end_date):
    from src.visualization.data_loader import (
        load_discovery_performance,
        load_discovery_records,
    )

    records_df = load_discovery_records(mode, start_date, end_date)

    if records_df.empty:
        st.info(f"目前沒有 {MODE_OPTIONS[mode]} 的推薦記錄")
        return

    # 搜尋框
    search = st.text_input("搜尋股票（代號或名稱）", "")
    if search:
        mask = records_df["stock_id"].str.contains(search, na=False) | records_df["stock_name"].str.contains(
            search, na=False
        )
        records_df = records_df[mask]
        if records_df.empty:
            st.warning(f"找不到符合「{search}」的推薦記錄")
            return

    # 嘗試合併報酬率資料
    perf_result = load_discovery_performance(
        mode,
        holding_days=HOLDING_DAYS_OPTIONS,
        start_date=start_date,
        end_date=end_date,
    )
    detail = perf_result["detail"]

    if not detail.empty:
        # 合併報酬率欄位
        ret_cols = [f"return_{d}d" for d in HOLDING_DAYS_OPTIONS]
        merge_cols = ["scan_date", "stock_id"] + [c for c in ret_cols if c in detail.columns]
        merged = records_df.merge(
            detail[merge_cols],
            on=["scan_date", "stock_id"],
            how="left",
        )
    else:
        merged = records_df

    # 顯示表格
    st.subheader(f"推薦記錄（共 {len(merged)} 筆）")

    display = merged.copy()

    # 格式化報酬率欄位 — 加色標
    def _color_return(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: #26A69A"
        elif val < 0:
            return "color: #EF5350"
        return ""

    format_dict = {"composite_score": "{:.3f}", "close": "{:.1f}"}
    ret_cols_present = [c for c in [f"return_{d}d" for d in HOLDING_DAYS_OPTIONS] if c in display.columns]

    styler = display.style.format(format_dict, na_rep="—")
    for col in ret_cols_present:
        styler = styler.format({col: lambda v: f"{v:+.2%}" if pd.notna(v) else "—"})
        styler = styler.map(_color_return, subset=[col])

    st.dataframe(styler, use_container_width=True, hide_index=True, height=500)

    # CSV 匯出
    csv = display.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 匯出 CSV",
        data=csv,
        file_name=f"discovery_{mode}_{start_date or 'all'}_{end_date or 'all'}.csv",
        mime="text/csv",
    )
