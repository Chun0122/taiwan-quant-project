"""部位控制總覽 — 輪動組合風險監控頁面。

顯示 Portfolio Heat、相關性矩陣、回撤走勢、Regime 狀態、個股風險貢獻。
"""

from __future__ import annotations

import streamlit as st

from src.constants import (
    CORRELATION_THRESHOLD,
    MAX_PORTFOLIO_HEAT,
    PER_POSITION_RISK_CAP,
)
from src.portfolio.rotation import (
    compute_correlation_matrix,
    compute_portfolio_heat,
    compute_single_trade_risk,
    find_high_correlation_pairs,
)
from src.visualization.charts import (
    compute_drawdown_series,
    plot_correlation_heatmap,
    plot_drawdown_area,
    plot_heat_gauge,
)
from src.visualization.data_loader import (
    load_multi_stock_closes,
    load_regime_state,
    load_rotation_portfolio_info,
    load_rotation_portfolio_names,
    load_rotation_positions,
)

_REGIME_DISPLAY = {
    "bull": ("Bull", "green"),
    "sideways": ("Sideways", "orange"),
    "bear": ("Bear", "red"),
    "crisis": ("Crisis", "darkred"),
    "unknown": ("Unknown", "gray"),
}


def render() -> None:
    """部位控制總覽頁主入口。"""
    st.header("部位控制總覽")
    st.caption("輪動組合風險監控：Portfolio Heat、相關性、回撤、市場 Regime")

    # ── Sidebar: 組合選擇 ──
    portfolio_names = load_rotation_portfolio_names()
    if not portfolio_names:
        st.info("尚未建立輪動組合。使用 CLI 建立：`python main.py rotation create ...`")
        return

    selected = st.sidebar.selectbox("選擇輪動組合", portfolio_names)

    # ── 載入資料 ──
    info = load_rotation_portfolio_info(selected)
    if info is None:
        st.warning(f"無法載入組合 {selected}")
        return

    positions = load_rotation_positions(selected)
    regime_state = load_regime_state()

    # 準備 heat 計算資料
    current_positions = [
        {"stock_id": p["stock_id"], "shares": p["shares"], "allocated_capital": p["allocated_capital"]}
        for p in positions
    ]
    stop_losses = {p["stock_id"]: p["stop_loss"] for p in positions if p.get("stop_loss") is not None}
    today_prices = {p["stock_id"]: p["current_price"] for p in positions if p.get("current_price") is not None}
    total_capital = info["current_capital"]

    # 計算 heat
    heat = compute_portfolio_heat(
        current_positions=current_positions,
        stop_losses=stop_losses if stop_losses else None,
        today_prices=today_prices if today_prices else None,
        total_capital=total_capital,
        per_position_risk_cap=PER_POSITION_RISK_CAP,
    )

    # 計算回撤（簡易：initial vs current）
    initial_cap = info["initial_capital"]
    drawdown_pct = max(0, (initial_cap - total_capital) / initial_cap * 100) if initial_cap > 0 else 0.0

    # ── Row 1: 關鍵指標 ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Portfolio Heat")
        st.plotly_chart(
            plot_heat_gauge(heat, MAX_PORTFOLIO_HEAT),
            use_container_width=True,
        )

    with col2:
        st.subheader("當前回撤")
        st.metric(
            label="回撤幅度",
            value=f"-{drawdown_pct:.1f}%",
            delta=None,
        )
        if drawdown_pct >= 15:
            st.error("回撤超過 15% — Drawdown Guard 高度警戒")
        elif drawdown_pct >= 10:
            st.warning("回撤超過 10% — Drawdown Guard 警戒")
        elif drawdown_pct >= 5:
            st.info("回撤超過 5% — Drawdown Guard 啟動")

    with col3:
        st.subheader("市場 Regime")
        if regime_state:
            regime = regime_state["regime"]
            label, color = _REGIME_DISPLAY.get(regime, ("Unknown", "gray"))
            st.markdown(
                f"<h2 style='color:{color};'>{label}</h2>",
                unsafe_allow_html=True,
            )
            if regime_state.get("regime_since"):
                st.caption(f"自 {regime_state['regime_since']} 起")
            if regime == "crisis":
                st.error("Crisis 模式：新倉開立已被阻擋")
        else:
            st.markdown("<h2 style='color:gray;'>Unknown</h2>", unsafe_allow_html=True)
            st.caption("尚無 Regime 狀態（執行 morning-routine 後生效）")

    with col4:
        st.subheader("組合摘要")
        st.metric("總資產", f"${total_capital:,.0f}")
        st.metric("現金", f"${info['current_cash']:,.0f}")
        open_count = len(positions)
        st.metric("持倉", f"{open_count} / {info['max_positions']}")
        st.caption(f"模式: {info['mode']} | 持有: {info['holding_days']}天")

    st.divider()

    # ── Row 2: 相關性 + 回撤走勢 ──
    stock_ids = [p["stock_id"] for p in positions]

    left, right = st.columns([3, 2])

    with left:
        st.subheader("持倉相關性矩陣")
        if len(stock_ids) >= 2:
            price_df = load_multi_stock_closes(stock_ids, days=90)
            if not price_df.empty and len(price_df.columns) >= 2:
                corr = compute_correlation_matrix(
                    {sid: price_df[sid].dropna() for sid in price_df.columns if sid in stock_ids},
                    window=60,
                )
                if not corr.empty:
                    high_pairs = find_high_correlation_pairs(corr, CORRELATION_THRESHOLD)
                    st.plotly_chart(
                        plot_correlation_heatmap(corr, CORRELATION_THRESHOLD),
                        use_container_width=True,
                    )
                    if high_pairs:
                        pair_strs = [f"{a}-{b} ({v:.2f})" for a, b, v in high_pairs]
                        st.warning(f"高相關配對（>{CORRELATION_THRESHOLD}）：{', '.join(pair_strs)}")
                else:
                    st.info("價格資料不足，無法計算相關性")
            else:
                st.info("價格資料不足，無法計算相關性")
        elif len(stock_ids) == 1:
            st.info("僅 1 檔持倉，無需計算相關性")
        else:
            st.info("目前無持倉")

    with right:
        st.subheader("回撤走勢")
        if initial_cap > 0:
            # 簡易權益序列：初始→現在
            equity = [initial_cap, total_capital]
            dd_series = compute_drawdown_series(equity)
            if len(dd_series) > 1:
                st.plotly_chart(
                    plot_drawdown_area(dd_series),
                    use_container_width=True,
                )
            else:
                st.metric("當前回撤", f"-{drawdown_pct:.1f}%")
        else:
            st.info("無資金資訊")

        # Regime 歷史
        if regime_state:
            st.subheader("Regime 狀態詳情")
            details = {
                "當前狀態": regime_state["regime"],
                "起始日期": regime_state.get("regime_since", "N/A"),
                "最後更新": regime_state.get("last_updated", "N/A"),
                "確認天數": regime_state.get("confirmation_count", 0),
                "待轉換": regime_state.get("pending_transition") or "無",
            }
            for k, v in details.items():
                st.text(f"{k}: {v}")

    st.divider()

    # ── Row 3: 個股風險貢獻明細 ──
    st.subheader("個股風險貢獻明細")

    if not positions:
        st.info("目前無持倉")
        return

    import pandas as pd

    rows = []
    for p in positions:
        risk_frac = compute_single_trade_risk(
            entry_price=p.get("current_price") or p["entry_price"],
            stop_loss=p.get("stop_loss"),
            shares=p["shares"],
            total_capital=total_capital,
            per_position_risk_cap=PER_POSITION_RISK_CAP,
            allocated_capital=p["allocated_capital"],
        )
        risk_amount = risk_frac * total_capital
        risk_pct = risk_frac * 100
        rows.append(
            {
                "股票": f"{p['stock_id']} {p['stock_name']}",
                "進場價": p["entry_price"],
                "現價": p.get("current_price"),
                "停損價": p.get("stop_loss"),
                "股數": p["shares"],
                "配置資金": p["allocated_capital"],
                "風險金額": risk_amount,
                "Heat 貢獻": f"{risk_pct:.2f}%",
                "未實現損益": f"{p['unrealized_pct']:.1%}" if p.get("unrealized_pct") is not None else "N/A",
                "到期日": p.get("planned_exit_date"),
                "持有天數": p.get("holding_days_count", 0),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Heat 摘要
    st.caption(
        f"Portfolio Heat = {heat:.1%} / 上限 {MAX_PORTFOLIO_HEAT:.0%}"
        f" | 剩餘預算 = {max(0, MAX_PORTFOLIO_HEAT - heat):.1%}"
    )
