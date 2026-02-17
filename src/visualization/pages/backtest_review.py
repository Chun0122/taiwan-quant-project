"""回測結果頁面 — 績效摘要、權益曲線、交易明細。"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src.visualization.data_loader import (
    load_backtest_list, load_backtest_by_id, load_trades, load_price_with_indicators,
)
from src.visualization.charts import plot_equity_curve


def _fmt(val, suffix="", default="N/A"):
    if val is None:
        return default
    return f"{val}{suffix}"


def render() -> None:
    st.title("🔄 回測結果")

    bt_list = load_backtest_list()
    if bt_list.empty:
        st.warning("尚無回測紀錄，請先執行 `python main.py backtest --stock 2330 --strategy sma_cross`")
        return

    # --- 回測比較表 ---
    st.subheader("回測紀錄總覽")
    display_df = bt_list[[
        "id", "stock_id", "strategy_name", "start_date", "end_date",
        "total_return", "annual_return", "sharpe_ratio", "max_drawdown",
        "win_rate", "total_trades",
    ]].copy()
    display_df.columns = [
        "ID", "股票", "策略", "起始日", "結束日",
        "總報酬%", "年化報酬%", "Sharpe", "MDD%",
        "勝率%", "交易次數",
    ]
    st.dataframe(display_df, width="stretch", hide_index=True)

    # --- 選擇單筆回測 ---
    st.divider()
    bt_options = {
        f"#{r['id']} {r['stock_id']} {r['strategy_name']} ({r['total_return']:+.2f}%)": r["id"]
        for _, r in bt_list.iterrows()
    }
    selected_label = st.sidebar.selectbox("選擇回測紀錄", list(bt_options.keys()))
    selected_id = bt_options[selected_label]

    bt = load_backtest_by_id(selected_id)
    if not bt:
        st.error("無法載入回測紀錄")
        return

    # --- 績效摘要卡片 ---
    st.subheader(f"#{bt['id']} {bt['stock_id']} — {bt['strategy_name']}")
    st.caption(f"{bt['start_date']} ~ {bt['end_date']}")

    # 第一排：核心指標
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("總報酬", f"{bt['total_return']:+.2f}%")
    c2.metric("年化報酬", f"{bt['annual_return']:+.2f}%")
    c3.metric("Sharpe", _fmt(bt['sharpe_ratio']))
    c4.metric("最大回撤", f"{bt['max_drawdown']:.2f}%")
    c5.metric("勝率", _fmt(bt['win_rate'], "%"))

    # 第二排：進階指標
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Sortino", _fmt(bt.get('sortino_ratio')))
    a2.metric("Calmar", _fmt(bt.get('calmar_ratio')))
    a3.metric("VaR(95%)", _fmt(bt.get('var_95'), "%"))
    a4.metric("CVaR(95%)", _fmt(bt.get('cvar_95'), "%"))
    a5.metric("Profit Factor", _fmt(bt.get('profit_factor')))

    # 第三排：資金
    m1, m2, m3 = st.columns(3)
    m1.metric("初始資金", f"{bt['initial_capital']:,.0f}")
    m2.metric("最終資金", f"{bt['final_capital']:,.2f}")
    m3.metric("交易次數", f"{bt['total_trades']}")

    # --- 權益曲線 ---
    trades = load_trades(selected_id)
    prices = load_price_with_indicators(
        bt["stock_id"], str(bt["start_date"]), str(bt["end_date"])
    )

    if not prices.empty:
        fig_eq = plot_equity_curve(trades, prices, bt["initial_capital"])
        st.plotly_chart(fig_eq, width="stretch")

    # --- 交易明細 ---
    if not trades.empty:
        st.subheader("交易明細")
        trade_display = trades.copy()

        # 判斷是否有 exit_reason 欄位
        has_exit_reason = "exit_reason" in trade_display.columns and trade_display["exit_reason"].notna().any()

        if has_exit_reason:
            trade_display = trade_display[[
                "entry_date", "entry_price", "exit_date", "exit_price",
                "shares", "pnl", "return_pct", "exit_reason",
            ]]
            trade_display.columns = [
                "進場日", "進場價", "出場日", "出場價", "股數", "損益", "報酬%", "出場原因",
            ]
        else:
            trade_display = trade_display[[
                "entry_date", "entry_price", "exit_date", "exit_price",
                "shares", "pnl", "return_pct",
            ]]
            trade_display.columns = [
                "進場日", "進場價", "出場日", "出場價", "股數", "損益", "報酬%",
            ]

        # 損益顏色標記
        st.dataframe(
            trade_display.style.map(
                lambda v: "color: #EF5350" if isinstance(v, (int, float)) and v < 0
                else "color: #26A69A" if isinstance(v, (int, float)) and v > 0
                else "",
                subset=["損益", "報酬%"],
            ),
            width="stretch",
            hide_index=True,
        )
