"""Streamlit 持倉監控頁面 — 視覺化 WatchEntry 持倉狀態與即時損益。"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

STATUS_ZH = {
    "active": "🟢 持倉中",
    "stopped_loss": "🔴 止損",
    "taken_profit": "🟡 止利",
    "expired": "⚫ 過期",
    "closed": "⚪ 已平倉",
}

STATUS_OPTIONS = {
    "active": "持倉中",
    "all": "全部",
    "stopped_loss": "已止損",
    "taken_profit": "已止利",
    "expired": "已過期",
    "closed": "已平倉",
}


def render() -> None:
    """持倉監控頁主入口。"""
    st.header("👁️ 持倉監控")
    st.caption("追蹤 `watch add` 新增的持倉，自動比對最新收盤價標記止損/止利/過期狀態。")

    # ── Sidebar ──────────────────────────────────────────────────────
    st.sidebar.markdown("### 持倉監控參數")
    status_key = st.sidebar.selectbox(
        "狀態篩選",
        list(STATUS_OPTIONS.keys()),
        format_func=lambda k: STATUS_OPTIONS[k],
    )

    # ── 載入資料 ─────────────────────────────────────────────────────
    from src.visualization.data_loader import load_watch_entries_with_status

    df = load_watch_entries_with_status(status_filter=status_key)

    if df.empty:
        label = STATUS_OPTIONS.get(status_key, status_key)
        st.info(f"目前沒有「{label}」狀態的持倉記錄。\n\n使用 CLI 新增持倉：\n```\npython main.py watch add 2330\n```")
        return

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 持倉總覽", "📈 個股走勢", "⚠️ 預警列表"])

    with tab1:
        _render_overview_tab(df)
    with tab2:
        _render_chart_tab(df)
    with tab3:
        _render_alert_tab(df)


# ──────────────────────────────────────────────────────────────────── #
#  Tab 1：持倉總覽
# ──────────────────────────────────────────────────────────────────── #


def _render_overview_tab(df: pd.DataFrame) -> None:
    """持倉總覽：指標卡片 + 彩色表格 + CSV 匯出。"""

    # 摘要指標卡片
    active_count = (df["computed_status"] == "active").sum()
    alert_count = df["computed_status"].isin(["stopped_loss", "taken_profit", "expired"]).sum()
    pnl_values = df["unrealized_pnl_pct"].dropna()
    avg_pnl = pnl_values.mean() if not pnl_values.empty else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("持倉總數", len(df))
    with col2:
        st.metric("持倉中", active_count)
    with col3:
        st.metric(
            "需關注",
            alert_count,
            delta=f"{alert_count}" if alert_count > 0 else None,
            delta_color="inverse" if alert_count > 0 else "normal",
        )
    with col4:
        if avg_pnl is not None:
            st.metric("平均損益%", f"{avg_pnl:+.2%}")
        else:
            st.metric("平均損益%", "—")

    st.divider()

    # 準備顯示表格
    display = df.copy()
    display["狀態"] = display["computed_status"].map(STATUS_ZH).fillna(display["computed_status"])

    col_map = {
        "id": "ID",
        "stock_id": "代號",
        "stock_name": "名稱",
        "entry_date": "進場日",
        "entry_price": "進場價",
        "current_price": "現價",
        "unrealized_pnl_pct": "損益%",
        "stop_loss": "止損",
        "take_profit": "目標",
        "狀態": "狀態",
        "valid_until": "有效期",
    }
    show_cols = [c for c in col_map if c in display.columns or c == "狀態"]
    display = display[show_cols].rename(columns=col_map)

    def _color_pnl(val):
        if pd.isna(val):
            return ""
        return "color: #26A69A" if val > 0 else "color: #EF5350" if val < 0 else ""

    format_dict = {
        "進場價": "{:.2f}",
        "現價": "{:.2f}",
        "止損": "{:.2f}",
        "目標": "{:.2f}",
    }

    styler = display.style.format(format_dict, na_rep="—")
    if "損益%" in display.columns:
        styler = styler.format({"損益%": lambda v: f"{v:+.2%}" if pd.notna(v) else "—"})
        styler = styler.map(_color_pnl, subset=["損益%"])

    st.subheader(f"持倉明細（{len(df)} 筆）")
    st.dataframe(styler, use_container_width=True, hide_index=True, height=min(400, 60 + len(df) * 38))

    # CSV 匯出
    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 匯出 CSV",
        data=csv,
        file_name="watch_entries.csv",
        mime="text/csv",
    )

    st.caption(
        "💡 使用 `python main.py watch update-status` 可將止損/止利/過期狀態寫入資料庫。"
        "此頁即時計算的「狀態」欄為 computed_status（不修改 DB）。"
    )


# ──────────────────────────────────────────────────────────────────── #
#  Tab 2：個股走勢
# ──────────────────────────────────────────────────────────────────── #


def _render_chart_tab(df: pd.DataFrame) -> None:
    """個股走勢：K 線圖疊加進場/止損/目標水平線。"""
    from src.visualization.data_loader import load_watch_entry_price_history

    # 股票選擇
    stock_options = df[["stock_id", "stock_name"]].drop_duplicates().copy()
    stock_labels = {row["stock_id"]: f"{row['stock_id']} {row['stock_name']}" for _, row in stock_options.iterrows()}

    selected_id = st.selectbox(
        "選擇股票",
        list(stock_labels.keys()),
        format_func=lambda k: stock_labels[k],
    )

    entry_row = df[df["stock_id"] == selected_id].iloc[0]
    entry_date = str(entry_row["entry_date"])
    entry_price = entry_row["entry_price"]
    stop_loss = entry_row.get("stop_loss")
    take_profit = entry_row.get("take_profit")

    days_back = st.slider("顯示天數", 30, 120, 60)

    price_df = load_watch_entry_price_history(selected_id, entry_date, days=days_back)

    if price_df.empty:
        st.info(f"找不到 {selected_id} 在 {entry_date} 之後的日K線資料。")
        return

    # 繪製 K 線
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=price_df["date"],
            open=price_df["open"],
            high=price_df["high"],
            low=price_df["low"],
            close=price_df["close"],
            name="K 線",
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
        )
    )

    x_range = [price_df["date"].iloc[0], price_df["date"].iloc[-1]]

    # 進場價（藍色實線）
    fig.add_shape(
        type="line",
        x0=x_range[0],
        x1=x_range[1],
        y0=entry_price,
        y1=entry_price,
        line={"color": "#1E88E5", "width": 2, "dash": "solid"},
    )
    fig.add_annotation(
        x=x_range[1],
        y=entry_price,
        text=f"進場 {entry_price:.2f}",
        showarrow=False,
        xanchor="left",
        font={"color": "#1E88E5", "size": 11},
    )

    # 止損（紅色虛線）
    if stop_loss and pd.notna(stop_loss):
        fig.add_shape(
            type="line",
            x0=x_range[0],
            x1=x_range[1],
            y0=stop_loss,
            y1=stop_loss,
            line={"color": "#EF5350", "width": 1.5, "dash": "dash"},
        )
        fig.add_annotation(
            x=x_range[1],
            y=stop_loss,
            text=f"止損 {stop_loss:.2f}",
            showarrow=False,
            xanchor="left",
            font={"color": "#EF5350", "size": 11},
        )

    # 目標（綠色虛線）
    if take_profit and pd.notna(take_profit):
        fig.add_shape(
            type="line",
            x0=x_range[0],
            x1=x_range[1],
            y0=take_profit,
            y1=take_profit,
            line={"color": "#26A69A", "width": 1.5, "dash": "dash"},
        )
        fig.add_annotation(
            x=x_range[1],
            y=take_profit,
            text=f"目標 {take_profit:.2f}",
            showarrow=False,
            xanchor="left",
            font={"color": "#26A69A", "size": 11},
        )

    fig.update_layout(
        title=f"{selected_id} {stock_labels[selected_id]} — 持倉走勢",
        xaxis_rangeslider_visible=False,
        height=480,
        margin={"l": 40, "r": 80, "t": 50, "b": 30},
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)

    # 顯示持倉資訊小卡
    cols = st.columns(4)
    cur = entry_row.get("current_price")
    pnl = entry_row.get("unrealized_pnl_pct")
    with cols[0]:
        st.metric("進場價", f"{entry_price:.2f}")
    with cols[1]:
        st.metric("現價", f"{cur:.2f}" if cur else "—", delta=f"{pnl:+.2%}" if pnl is not None else None)
    with cols[2]:
        st.metric("止損", f"{stop_loss:.2f}" if stop_loss and pd.notna(stop_loss) else "—")
    with cols[3]:
        st.metric("目標", f"{take_profit:.2f}" if take_profit and pd.notna(take_profit) else "—")


# ──────────────────────────────────────────────────────────────────── #
#  Tab 3：預警列表
# ──────────────────────────────────────────────────────────────────── #


def _render_alert_tab(df: pd.DataFrame) -> None:
    """預警列表：已觸發 + 接近止損/止利的持倉。"""

    # 已觸發
    triggered = df[df["computed_status"].isin(["stopped_loss", "taken_profit", "expired"])].copy()

    if not triggered.empty:
        st.subheader("🚨 已觸發（建議處理）")
        triggered["狀態"] = triggered["computed_status"].map(STATUS_ZH).fillna(triggered["computed_status"])
        show = triggered[
            ["stock_id", "stock_name", "entry_date", "entry_price", "current_price", "unrealized_pnl_pct", "狀態"]
        ].copy()
        show = show.rename(
            columns={
                "stock_id": "代號",
                "stock_name": "名稱",
                "entry_date": "進場日",
                "entry_price": "進場價",
                "current_price": "現價",
                "unrealized_pnl_pct": "損益%",
            }
        )

        def _color_pnl(val):
            if pd.isna(val):
                return ""
            return "color: #26A69A" if val > 0 else "color: #EF5350"

        styler = show.style.format({"進場價": "{:.2f}", "現價": "{:.2f}"}, na_rep="—")
        if "損益%" in show.columns:
            styler = styler.format({"損益%": lambda v: f"{v:+.2%}" if pd.notna(v) else "—"})
            styler = styler.map(_color_pnl, subset=["損益%"])
        st.dataframe(styler, use_container_width=True, hide_index=True)
        st.caption("執行 `python main.py watch close <ID>` 平倉，或 `watch update-status` 更新 DB 狀態。")
    else:
        st.success("目前沒有觸發止損/止利/過期的持倉。")

    st.divider()

    # 接近止損（距離 < 3%）
    active_df = df[df["computed_status"] == "active"].copy()
    if not active_df.empty and "current_price" in active_df.columns and "stop_loss" in active_df.columns:

        def _near_stop(row):
            if pd.isna(row["current_price"]) or pd.isna(row["stop_loss"]):
                return False
            if row["entry_price"] <= 0:
                return False
            return (row["current_price"] - row["stop_loss"]) / row["entry_price"] < 0.03

        near_stop = active_df[active_df.apply(_near_stop, axis=1)]

        if not near_stop.empty:
            st.subheader("⚠️ 接近止損（距離 < 3%）")
            show2 = near_stop[
                ["stock_id", "stock_name", "entry_price", "current_price", "stop_loss", "unrealized_pnl_pct"]
            ].copy()
            show2["距止損%"] = ((show2["current_price"] - show2["stop_loss"]) / show2["entry_price"] * 100).round(2)
            show2 = show2.rename(
                columns={
                    "stock_id": "代號",
                    "stock_name": "名稱",
                    "entry_price": "進場價",
                    "current_price": "現價",
                    "stop_loss": "止損",
                    "unrealized_pnl_pct": "損益%",
                }
            )
            st.dataframe(
                show2.style.format(
                    {
                        "進場價": "{:.2f}",
                        "現價": "{:.2f}",
                        "止損": "{:.2f}",
                        "損益%": lambda v: f"{v:+.2%}" if pd.notna(v) else "—",
                        "距止損%": "{:.2f}%",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("目前沒有接近止損的持倉（距離均 > 3%）。")
    elif active_df.empty:
        st.info("目前沒有持倉中的記錄。")
