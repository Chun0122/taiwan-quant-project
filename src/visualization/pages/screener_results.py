"""選股篩選結果頁 — Streamlit 頁面。"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src.screener.factors import FACTOR_REGISTRY


def render() -> None:
    """渲染選股篩選頁面。"""
    st.header("多因子選股篩選")

    # ─── 側欄：因子條件設定 ────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("篩選條件")

    # 依類別分群顯示 checkboxes
    categories: dict[str, list[str]] = {}
    for fname, finfo in FACTOR_REGISTRY.items():
        cat = finfo["category"]
        categories.setdefault(cat, []).append(fname)

    selected_factors: list[str] = []
    factor_params: dict[str, dict] = {}

    for cat, fnames in categories.items():
        st.sidebar.markdown(f"**{cat}**")
        for fname in fnames:
            finfo = FACTOR_REGISTRY[fname]
            checked = st.sidebar.checkbox(finfo["label"], value=False, key=f"factor_{fname}")
            if checked:
                selected_factors.append(fname)

                # 可調參數
                params = finfo["params"].copy()
                if "threshold" in params:
                    params["threshold"] = st.sidebar.number_input(
                        f"  門檻值 ({fname})",
                        value=float(params["threshold"]),
                        key=f"param_{fname}_threshold",
                    )
                if "period" in params:
                    params["period"] = st.sidebar.number_input(
                        f"  週期 ({fname})",
                        value=int(params["period"]),
                        step=1,
                        key=f"param_{fname}_period",
                    )
                if "days" in params:
                    params["days"] = st.sidebar.number_input(
                        f"  天數 ({fname})",
                        value=int(params["days"]),
                        step=1,
                        key=f"param_{fname}_days",
                    )
                if "months" in params:
                    params["months"] = st.sidebar.number_input(
                        f"  月數 ({fname})",
                        value=int(params["months"]),
                        step=1,
                        key=f"param_{fname}_months",
                    )
                factor_params[fname] = params

    st.sidebar.markdown("---")
    filter_mode = st.sidebar.radio(
        "篩選模式",
        ["全部符合 (AND)", "任一符合 (OR)"],
        key="filter_mode",
    )
    require_all = filter_mode == "全部符合 (AND)"

    lookback = st.sidebar.slider("回溯天數", min_value=1, max_value=30, value=5, key="lookback")

    # ─── 主區：執行篩選 ──────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        run_scan = st.button("執行篩選", type="primary", use_container_width=True)
    with col2:
        export_csv = st.button("匯出 CSV", use_container_width=True)

    if run_scan or export_csv:
        if not selected_factors:
            st.warning("請在左側勾選至少一個篩選條件")
            return

        from src.screener.engine import MultiFactorScreener

        with st.spinner("正在掃描股票..."):
            screener = MultiFactorScreener(lookback_days=lookback)

            if require_all:
                results = screener.scan_with_conditions(selected_factors, require_all=True)
            else:
                results = screener.scan_with_conditions(selected_factors, require_all=False)

        if results.empty:
            st.info("無符合條件的股票")
            return

        # 顯示摘要
        st.success(f"找到 {len(results)} 檔符合條件的股票")

        # 主要結果表格
        display_cols = [
            "stock_id", "close", "volume", "factor_score",
        ]
        # 加入可用的指標欄位
        optional_cols = [
            "rsi_14", "macd", "sma_20",
            "foreign_net", "trust_net", "dealer_net",
            "margin_balance", "short_balance", "yoy_growth",
        ]
        for col in optional_cols:
            if col in results.columns:
                display_cols.append(col)

        # 加入因子命中欄位
        factor_hit_cols = [f"f_{f}" for f in selected_factors if f"f_{f}" in results.columns]
        display_cols.extend(factor_hit_cols)

        display_df = results[[c for c in display_cols if c in results.columns]].copy()

        # 格式化因子命中欄位名稱
        rename = {}
        for f in selected_factors:
            col_name = f"f_{f}"
            if col_name in display_df.columns:
                rename[col_name] = FACTOR_REGISTRY[f]["label"]
        display_df = display_df.rename(columns=rename)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "stock_id": st.column_config.TextColumn("股票代號", width="small"),
                "close": st.column_config.NumberColumn("收盤價", format="%.2f"),
                "volume": st.column_config.NumberColumn("成交量", format="%d"),
                "factor_score": st.column_config.ProgressColumn(
                    "因子分數", min_value=0, max_value=1, format="%.2f"
                ),
            },
        )

        # 匯出 CSV
        if export_csv:
            csv_data = results.to_csv(index=False)
            st.download_button(
                label="下載 CSV",
                data=csv_data,
                file_name="scan_results.csv",
                mime="text/csv",
            )

    else:
        # 預設畫面
        st.info("請在左側選擇篩選條件後點擊「執行篩選」")

        st.markdown("""
        ### 使用說明

        1. 在左側勾選要使用的**篩選因子**
        2. 調整各因子的門檻值參數
        3. 選擇篩選模式（AND=所有條件都符合 / OR=任一符合）
        4. 點擊「**執行篩選**」開始掃描

        ### 可用因子

        | 類別 | 因子 | 說明 |
        |------|------|------|
        | 技術面 | RSI 超賣 | RSI < 門檻值 |
        | 技術面 | MACD 黃金交叉 | MACD 上穿 Signal 線 |
        | 技術面 | 股價 > SMA | 收盤價站上均線 |
        | 籌碼面 | 外資買超 | 外資淨買入 > 0 |
        | 籌碼面 | 法人連續買超 | 三大法人合計連買 N 天 |
        | 籌碼面 | 券資比 | 融券/融資餘額比 > 門檻 |
        | 基本面 | 營收 YoY | 月營收年增率 > 門檻 |
        | 基本面 | 連續營收成長 | MoM 連續正成長 N 月 |
        """)
