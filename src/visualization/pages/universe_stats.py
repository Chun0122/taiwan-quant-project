"""Universe 統計時序面板（P1 任務 8）。

讀取 UniverseStatLog 顯示三層漏斗每日剩餘股數時序，協助 audit 過熱閘門 /
regime 收緊 對 universe size 的影響。
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.visualization.data_loader import load_universe_stat_log


def render() -> None:
    st.title("🌐 Universe 統計時序")
    st.caption(
        "三層漏斗每日剩餘股數（Stage 1 SQL → Stage 2 流動性 → Stage 3 趨勢 + Memory）。"
        "過熱閘門 / regime 收緊 會明顯改變 universe size，此面板可看出歷史趨勢。"
    )

    # ── Sidebar ──────────────────────────────────────────────────────
    st.sidebar.markdown("### Universe 統計參數")
    lookback_days = st.sidebar.slider("回溯天數", min_value=7, max_value=180, value=60, step=7)
    mode_filter = st.sidebar.selectbox(
        "模式篩選",
        ["全部", "momentum", "swing", "value", "dividend", "growth"],
    )

    df = load_universe_stat_log(
        lookback_days=lookback_days,
        mode=mode_filter if mode_filter != "全部" else None,
    )

    if df.empty:
        st.info(
            "尚無 universe_stat_log 紀錄。\n\n"
            "執行 `python main.py discover` 或 `python main.py morning-routine` 後會自動寫入。"
        )
        return

    # ── 最末日彙整 metric ────────────────────────────────────────────
    latest_date = df["scan_date"].max()
    latest = df[df["scan_date"] == latest_date]
    st.caption(f"最末掃描日：{latest_date}（共 {len(latest)} mode）")

    cols = st.columns(max(len(latest), 1))
    for col, (_, row) in zip(cols, latest.iterrows()):
        regime_tag = f" ({row['regime']})" if row["regime"] else ""
        col.metric(
            f"{row['mode']}{regime_tag}",
            f"{int(row['final_candidates']):,}",
            delta=f"sql:{int(row['total_after_sql'])} → liq:{int(row['total_after_liquidity'])} → trend:{int(row['total_after_trend'])}",
            delta_color="off",
        )

    st.divider()

    # ── Tabs：時序圖 + 漏斗圖 + 原始表 ────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📈 final_candidates 時序", "📊 stage 漏斗", "📋 原始資料"])

    with tab1:
        _render_timeseries(df)

    with tab2:
        _render_funnel(df, latest)

    with tab3:
        st.dataframe(df.sort_values(["scan_date", "mode"], ascending=[False, True]), width="stretch", hide_index=True)


def _render_timeseries(df: pd.DataFrame) -> None:
    """每 mode 一條 final_candidates 時序線。"""
    fig = go.Figure()
    for mode in sorted(df["mode"].unique()):
        sub = df[df["mode"] == mode].sort_values("scan_date")
        fig.add_trace(
            go.Scatter(
                x=sub["scan_date"],
                y=sub["final_candidates"],
                mode="lines+markers",
                name=mode,
            )
        )
    fig.update_layout(
        height=400,
        xaxis_title="掃描日期",
        yaxis_title="Final Candidates",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 顯示變化率（最末日 vs 7 日均值）
    if len(df) >= 7:
        st.caption("⚠ 規則：final_candidates 跌破近 7 日均值 30% 通常意味 regime 收緊或 data freshness 問題")


def _render_funnel(df: pd.DataFrame, latest: pd.DataFrame) -> None:
    """最末日各 mode stage 漏斗（橫向 stacked bar）。"""
    if latest.empty:
        st.info("無最末日資料")
        return

    modes = latest["mode"].tolist()
    stage1 = latest["total_after_sql"].tolist()
    stage2 = latest["total_after_liquidity"].tolist()
    stage3 = latest["total_after_trend"].tolist()
    memory = latest["from_memory"].tolist()
    final = latest["final_candidates"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Stage 1 SQL", y=modes, x=stage1, orientation="h", marker_color="#9CA3AF"))
    fig.add_trace(go.Bar(name="Stage 2 流動性", y=modes, x=stage2, orientation="h", marker_color="#6B7280"))
    fig.add_trace(go.Bar(name="Stage 3 趨勢", y=modes, x=stage3, orientation="h", marker_color="#4B5563"))
    fig.add_trace(go.Bar(name="+Memory", y=modes, x=memory, orientation="h", marker_color="#10B981"))
    fig.add_trace(go.Bar(name="Final", y=modes, x=final, orientation="h", marker_color="#3B82F6"))
    fig.update_layout(
        barmode="group",
        height=80 + 60 * len(modes),
        xaxis_title="股數",
        yaxis_title="模式",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
