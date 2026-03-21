"""策略比較頁面 — 多策略累積報酬率曲線 + 績效指標比較表。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.constants import COMMISSION_RATE, TAX_RATE
from src.visualization.charts import (
    plot_strategy_comparison_curves,
    plot_strategy_metrics_bar,
)
from src.visualization.data_loader import (
    load_backtest_list,
    load_price_with_indicators,
    load_trades,
)

# ------------------------------------------------------------------ #
#  輔助：從交易明細 + 日K 重建正規化（%）權益曲線
# ------------------------------------------------------------------ #


def _build_equity_pct(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float,
) -> tuple[list, list]:
    """從交易記錄 + 日K 重建累積報酬率（%）序列。

    Returns
    -------
    (dates, equity_pct):
        dates — 日期序列（與 prices 對齊）。
        equity_pct — 各日相對初始資金的累積報酬率（%），0 = 平持。
    """
    if prices.empty:
        return [], []

    dates = prices["date"].tolist()
    closes = prices["close"].tolist()

    capital = initial_capital
    position = 0
    entry_price_val = 0.0
    equity: list[float] = []

    buy_dates: dict = {}
    sell_dates: dict = {}
    if not trades.empty:
        for _, t in trades.iterrows():
            buy_dates[pd.Timestamp(t["entry_date"]).date()] = float(t["entry_price"])
            if pd.notna(t.get("exit_date")):
                sell_dates[pd.Timestamp(t["exit_date"]).date()] = float(t["exit_price"])

    for i, dt in enumerate(dates):
        dt_date = pd.Timestamp(dt).date() if not isinstance(dt, pd.Timestamp) else dt.date()
        close = closes[i]

        if dt_date in buy_dates and position == 0:
            ep = buy_dates[dt_date]
            shares = int(capital * 0.998 // ep)
            if shares > 0:
                capital -= shares * ep * 1.001425
                position = shares
                entry_price_val = ep

        elif dt_date in sell_dates and position > 0:
            sp = sell_dates[dt_date]
            revenue = position * sp * (1 - COMMISSION_RATE - TAX_RATE)
            capital += revenue
            position = 0

        equity.append(capital + position * close)

    equity_pct = [(v / initial_capital - 1.0) * 100.0 for v in equity]
    return dates, equity_pct


# ------------------------------------------------------------------ #
#  頁面主體
# ------------------------------------------------------------------ #


def render() -> None:
    st.title("⚖️ 策略比較")

    bt_list = load_backtest_list()
    if bt_list.empty:
        st.warning("尚無回測紀錄，請先執行 `python main.py backtest --stock <代號> --strategy <策略>`")
        return

    # ── Sidebar 控制 ──────────────────────────────────────────────── #
    all_stocks = sorted(bt_list["stock_id"].unique())
    selected_stock = st.sidebar.selectbox("股票代號", all_stocks)

    # 過濾出該股票的所有策略
    stock_bt = bt_list[bt_list["stock_id"] == selected_stock]
    available_strategies = sorted(stock_bt["strategy_name"].unique())

    if not available_strategies:
        st.info(f"股票 {selected_stock} 尚無回測記錄。")
        return

    selected_strategies = st.sidebar.multiselect(
        "比較策略（最多 5 個）",
        available_strategies,
        default=available_strategies[: min(3, len(available_strategies))],
        max_selections=5,
    )

    if not selected_strategies:
        st.info("請在左側選擇至少一個策略。")
        return

    # 過濾 + 每個策略取最新一筆回測
    filtered = stock_bt[stock_bt["strategy_name"].isin(selected_strategies)].copy()
    if "created_at" in filtered.columns:
        filtered = filtered.sort_values("created_at")
    latest = filtered.groupby("strategy_name").last().reset_index()

    if latest.empty:
        st.info("所選策略無回測資料。")
        return

    # ── Tab 佈局 ─────────────────────────────────────────────────── #
    tab1, tab2, tab3 = st.tabs(["📊 績效指標比較", "📈 累積報酬率曲線", "🔢 進階指標"])

    # ── Tab 1: 績效指標比較表 ──────────────────────────────────────── #
    with tab1:
        st.subheader(f"股票 {selected_stock} — 各策略績效比較")

        display_cols = {
            "strategy_name": "策略",
            "start_date": "起始日",
            "end_date": "結束日",
            "total_return": "總報酬%",
            "annual_return": "年化報酬%",
            "sharpe_ratio": "Sharpe",
            "max_drawdown": "MDD%",
            "win_rate": "勝率%",
            "total_trades": "交易次數",
        }

        table_df = latest[[c for c in display_cols if c in latest.columns]].copy()
        table_df.columns = [display_cols[c] for c in table_df.columns]

        # 高亮最佳值
        def _highlight_best(col: pd.Series) -> list[str]:
            if col.name in ("總報酬%", "年化報酬%", "Sharpe", "勝率%"):
                best = col.max()
                return ["background-color: rgba(38,166,154,0.20)" if v == best else "" for v in col]
            if col.name == "MDD%":
                best = col.min()  # 越小越好
                return ["background-color: rgba(38,166,154,0.20)" if v == best else "" for v in col]
            return [""] * len(col)

        numeric_cols = [c for c in table_df.columns if c not in ("策略", "起始日", "結束日", "交易次數")]
        styled = table_df.style.apply(_highlight_best, subset=numeric_cols)
        styled = styled.format(
            {c: "{:+.2f}" for c in numeric_cols},
            na_rep="N/A",
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # 各指標長條圖（4 個指標 2x2 排列）
        st.divider()
        c1, c2 = st.columns(2)
        metrics_map = [
            ("total_return", "總報酬率 (%)"),
            ("annual_return", "年化報酬率 (%)"),
            ("sharpe_ratio", "Sharpe Ratio"),
            ("max_drawdown", "最大回撤 (%)"),
        ]
        for idx, (col, title) in enumerate(metrics_map):
            target_col = c1 if idx % 2 == 0 else c2
            if col in latest.columns and latest[col].notna().any():
                bar_df = latest[["strategy_name", col]].dropna().copy()
                bar_df.columns = ["策略", col]
                fig = plot_strategy_metrics_bar(bar_df, col, label_col="策略", title=title)
                target_col.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: 累積報酬率曲線 ─────────────────────────────────────── #
    with tab2:
        st.subheader("各策略累積報酬率走勢")
        st.caption("各策略的逐日標準化權益曲線（初始資金 = 0%），方便直接比較不同策略的累積超額報酬。")

        curves: dict[str, tuple[list, list]] = {}
        missing_data: list[str] = []

        for _, row in latest.iterrows():
            strategy = row["strategy_name"]
            bt_id = int(row["id"])
            initial_cap = float(row.get("initial_capital", 1_000_000))

            trades = load_trades(bt_id)
            prices = load_price_with_indicators(
                selected_stock,
                str(row["start_date"]),
                str(row["end_date"]),
            )

            if prices.empty:
                missing_data.append(strategy)
                continue

            dates, pct = _build_equity_pct(trades, prices, initial_cap)
            if dates:
                curves[strategy] = (dates, pct)

        if missing_data:
            st.warning(f"以下策略缺少價格資料，無法繪製曲線：{', '.join(missing_data)}")

        if not curves:
            st.info("無法取得任何策略的價格資料，請確認已同步日K線（`python main.py sync`）。")
        else:
            fig = plot_strategy_comparison_curves(curves)
            st.plotly_chart(fig, use_container_width=True)

            # 期末報酬率摘要
            summary_rows = []
            for strategy, (_, pct) in curves.items():
                if pct:
                    summary_rows.append(
                        {
                            "策略": strategy,
                            "期末累積報酬%": f"{pct[-1]:+.2f}",
                            "最高點%": f"{max(pct):+.2f}",
                            "最低點%": f"{min(pct):+.2f}",
                        }
                    )
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    # ── Tab 3: 進階指標比較 ────────────────────────────────────────── #
    with tab3:
        st.subheader("進階風險指標比較")

        adv_metrics = [
            ("sortino_ratio", "Sortino Ratio"),
            ("calmar_ratio", "Calmar Ratio"),
            ("profit_factor", "Profit Factor"),
            ("win_rate", "勝率 (%)"),
        ]

        available_adv = [(col, label) for col, label in adv_metrics if col in latest.columns]

        if not available_adv:
            st.info("無進階指標資料（請執行含止損/止利的回測以產生更多指標）。")
        else:
            for col, label in available_adv:
                if latest[col].notna().any():
                    bar_df = latest[["strategy_name", col]].dropna().copy()
                    bar_df.columns = ["策略", col]
                    fig = plot_strategy_metrics_bar(bar_df, col, label_col="策略", title=label)
                    st.plotly_chart(fig, use_container_width=True)

        # 完整指標明細表
        st.divider()
        st.caption("完整指標明細")
        all_adv_cols = {
            "strategy_name": "策略",
            "sortino_ratio": "Sortino",
            "calmar_ratio": "Calmar",
            "var_95": "VaR(95%)",
            "cvar_95": "CVaR(95%)",
            "profit_factor": "Profit Factor",
            "benchmark_return": "基準報酬%",
        }
        adv_table = latest[[c for c in all_adv_cols if c in latest.columns]].copy()
        adv_table.columns = [all_adv_cols[c] for c in adv_table.columns]
        adv_numeric = [c for c in adv_table.columns if c != "策略"]
        st.dataframe(
            adv_table.style.format({c: "{:.3f}" for c in adv_numeric}, na_rep="N/A"),
            use_container_width=True,
            hide_index=True,
        )
