"""投資組合回測頁面 — 組合績效、權益曲線、個股貢獻、交易明細。"""

from __future__ import annotations

import streamlit as st

from src.visualization.charts import (
    plot_allocation_pie,
    plot_per_stock_returns,
)
from src.visualization.data_loader import (
    load_portfolio_by_id,
    load_portfolio_list,
    load_portfolio_trades,
)


def _fmt(val, suffix="", default="N/A"):
    if val is None:
        return default
    return f"{val}{suffix}"


def render() -> None:
    st.title("📊 投資組合回測")

    pf_list = load_portfolio_list()
    if pf_list.empty:
        st.warning("尚無投資組合回測紀錄，請先執行 `python main.py backtest --stocks 2330 2317 --strategy sma_cross`")
        return

    # --- 組合回測列表 ---
    st.subheader("組合回測紀錄總覽")
    display_df = pf_list[
        [
            "id",
            "stock_ids",
            "strategy_name",
            "start_date",
            "end_date",
            "total_return",
            "annual_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "total_trades",
            "allocation_method",
        ]
    ].copy()
    display_df.columns = [
        "ID",
        "股票",
        "策略",
        "起始日",
        "結束日",
        "總報酬%",
        "年化報酬%",
        "Sharpe",
        "MDD%",
        "勝率%",
        "交易次數",
        "配置方式",
    ]
    st.dataframe(display_df, width="stretch", hide_index=True)

    # --- 選擇單筆回測 ---
    st.divider()
    pf_options = {
        f"#{r['id']} [{r['stock_ids']}] {r['strategy_name']} ({r['total_return']:+.2f}%)": r["id"]
        for _, r in pf_list.iterrows()
    }
    selected_label = st.sidebar.selectbox("選擇組合回測", list(pf_options.keys()))
    selected_id = pf_options[selected_label]

    pf = load_portfolio_by_id(selected_id)
    if not pf:
        st.error("無法載入組合回測紀錄")
        return

    stock_ids = pf["stock_ids"].split(",")

    # --- 績效摘要卡片 ---
    st.subheader(f"#{pf['id']} [{pf['stock_ids']}] — {pf['strategy_name']}")
    st.caption(f"{pf['start_date']} ~ {pf['end_date']} | 配置: {pf.get('allocation_method', 'N/A')}")

    # 第一排
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("總報酬", f"{pf['total_return']:+.2f}%")
    c2.metric("年化報酬", f"{pf['annual_return']:+.2f}%")
    c3.metric("Sharpe", _fmt(pf["sharpe_ratio"]))
    c4.metric("最大回撤", f"{pf['max_drawdown']:.2f}%")
    c5.metric("勝率", _fmt(pf["win_rate"], "%"))

    # 第二排
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Sortino", _fmt(pf.get("sortino_ratio")))
    a2.metric("Calmar", _fmt(pf.get("calmar_ratio")))
    a3.metric("VaR(95%)", _fmt(pf.get("var_95"), "%"))
    a4.metric("CVaR(95%)", _fmt(pf.get("cvar_95"), "%"))
    a5.metric("Profit Factor", _fmt(pf.get("profit_factor")))

    # 第三排
    m1, m2, m3 = st.columns(3)
    m1.metric("初始資金", f"{pf['initial_capital']:,.0f}")
    m2.metric("最終資金", f"{pf['final_capital']:,.2f}")
    m3.metric("交易次數", f"{pf['total_trades']}")

    # --- 配置圓餅圖 + 個股報酬 ---
    trades_df = load_portfolio_trades(selected_id)

    col_left, col_right = st.columns(2)

    with col_left:
        fig_pie = plot_allocation_pie(stock_ids)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        # 從交易明細計算個股報酬
        if not trades_df.empty:
            per_stock_pnl = trades_df.groupby("stock_id")["pnl"].sum()
            per_stock_returns = {sid: round(pnl / pf["initial_capital"] * 100, 2) for sid, pnl in per_stock_pnl.items()}
            fig_bar = plot_per_stock_returns(per_stock_returns)
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- 交易明細 ---
    if not trades_df.empty:
        st.subheader("交易明細")
        trade_display = trades_df.copy()

        has_exit_reason = "exit_reason" in trade_display.columns and trade_display["exit_reason"].notna().any()

        if has_exit_reason:
            trade_display = trade_display[
                [
                    "stock_id",
                    "entry_date",
                    "entry_price",
                    "exit_date",
                    "exit_price",
                    "shares",
                    "pnl",
                    "return_pct",
                    "exit_reason",
                ]
            ]
            trade_display.columns = [
                "股票",
                "進場日",
                "進場價",
                "出場日",
                "出場價",
                "股數",
                "損益",
                "報酬%",
                "出場原因",
            ]
        else:
            trade_display = trade_display[
                [
                    "stock_id",
                    "entry_date",
                    "entry_price",
                    "exit_date",
                    "exit_price",
                    "shares",
                    "pnl",
                    "return_pct",
                ]
            ]
            trade_display.columns = [
                "股票",
                "進場日",
                "進場價",
                "出場日",
                "出場價",
                "股數",
                "損益",
                "報酬%",
            ]

        st.dataframe(
            trade_display.style.map(
                lambda v: (
                    "color: #EF5350"
                    if isinstance(v, (int, float)) and v < 0
                    else "color: #26A69A"
                    if isinstance(v, (int, float)) and v > 0
                    else ""
                ),
                subset=["損益", "報酬%"],
            ),
            width="stretch",
            hide_index=True,
        )
