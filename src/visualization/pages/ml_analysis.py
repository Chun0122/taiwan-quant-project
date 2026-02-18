"""ML 策略分析頁 — 特徵重要性、模型表現、Walk-Forward 驗證結果。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.visualization.data_loader import get_stock_list


def render() -> None:
    """渲染 ML 策略分析頁面。"""
    st.header("ML 策略分析")

    # ─── 側欄設定 ──────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("ML 設定")

    stocks = get_stock_list()
    if not stocks:
        st.warning("尚無股票資料，請先執行 `python main.py sync`")
        return

    stock_id = st.sidebar.selectbox("股票代號", stocks, key="ml_stock")

    model_type = st.sidebar.selectbox(
        "模型類型",
        ["random_forest", "xgboost", "logistic"],
        format_func=lambda x: {
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "logistic": "Logistic Regression",
        }.get(x, x),
        key="ml_model",
    )

    lookback = st.sidebar.slider("回溯天數", 5, 60, 20, key="ml_lookback")
    forward_days = st.sidebar.slider("預測天數", 1, 20, 5, key="ml_forward")
    train_ratio = st.sidebar.slider("訓練比例", 0.5, 0.9, 0.7, 0.05, key="ml_train_ratio")
    threshold = st.sidebar.slider("訊號門檻", 0.5, 0.8, 0.6, 0.05, key="ml_threshold")

    # ─── 主區 ─────────────────────────────────────────────
    tab1, tab2 = st.tabs(["模型訓練分析", "Walk-Forward 驗證"])

    # ─── Tab 1: 模型訓練分析 ──────────────────────────────
    with tab1:
        if st.button("訓練模型", type="primary", key="ml_train_btn"):
            _run_model_analysis(stock_id, model_type, lookback, forward_days, train_ratio, threshold)

    # ─── Tab 2: Walk-Forward 驗證 ─────────────────────────
    with tab2:
        st.subheader("Walk-Forward 滾動驗證")
        st.markdown("""
        將歷史資料分成多個滾動窗口，每次用 train_window 訓練、test_window 測試，
        避免 ML 策略的過擬合問題。
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            train_window = st.number_input("訓練窗口（交易日）", 60, 504, 252, 21, key="wf_train")
        with col2:
            test_window = st.number_input("測試窗口（交易日）", 21, 252, 63, 21, key="wf_test")
        with col3:
            step_size = st.number_input("步進大小（交易日）", 21, 252, 63, 21, key="wf_step")

        if st.button("執行 Walk-Forward", type="primary", key="wf_run_btn"):
            _run_walk_forward(
                stock_id,
                model_type,
                lookback,
                forward_days,
                train_window,
                test_window,
                step_size,
                threshold,
            )


def _run_model_analysis(
    stock_id: str,
    model_type: str,
    lookback: int,
    forward_days: int,
    train_ratio: float,
    threshold: float,
) -> None:
    """訓練模型並顯示分析結果。"""
    from src.features.ml_features import build_ml_features, get_feature_columns
    from src.strategy.ml_strategy import MLStrategy

    with st.spinner("載入資料並訓練模型..."):
        strategy = MLStrategy(
            stock_id=stock_id,
            start_date="2020-01-01",
            end_date="2030-12-31",
            model_type=model_type,
            lookback=lookback,
            train_ratio=train_ratio,
            threshold=threshold,
            forward_days=forward_days,
        )

        data = strategy.load_data()
        if data.empty:
            st.error("無可用資料")
            return

        # 建構特徵
        df = build_ml_features(data, lookback=lookback, forward_days=forward_days)
        if len(df) < 50:
            st.error(f"特徵資料不足（{len(df)} 筆），需至少 50 筆")
            return

        feature_cols = get_feature_columns(df)
        X = df[feature_cols].values
        y = df["label"].values

        split_idx = int(len(df) * train_ratio)

        # 標準化
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[:split_idx])
        X_test = scaler.transform(X[split_idx:])
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        model = strategy._create_model()
        model.fit(X_train, y_train)

    # 顯示結果
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("訓練準確率", f"{train_acc:.1%}")
    c2.metric("測試準確率", f"{test_acc:.1%}")
    c3.metric("特徵數", len(feature_cols))
    c4.metric("資料筆數", f"{split_idx} / {len(df) - split_idx}")

    overfitting = train_acc - test_acc
    if overfitting > 0.1:
        st.warning(f"過擬合警告：訓練-測試差距 {overfitting:.1%}，建議降低模型複雜度或增加資料")

    # 特徵重要性
    st.subheader("特徵重要性")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = None

    if importances is not None:
        fi_df = (
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=True)
            .tail(15)
        )

        fig = go.Figure(
            go.Bar(
                x=fi_df["importance"],
                y=fi_df["feature"],
                orientation="h",
                marker_color="#2196F3",
            )
        )
        fig.update_layout(
            title="Top 15 特徵重要性",
            xaxis_title="重要性",
            yaxis_title="",
            height=450,
            margin=dict(l=150),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 預測機率分佈
    st.subheader("測試集預測機率分佈")

    proba = model.predict_proba(X_test)
    prob_up = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=prob_up,
            nbinsx=30,
            marker_color="#2196F3",
            opacity=0.7,
            name="預測機率",
        )
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"買入門檻 ({threshold})")
    fig.add_vline(
        x=1 - threshold, line_dash="dash", line_color="green", annotation_text=f"賣出門檻 ({1 - threshold:.2f})"
    )
    fig.update_layout(
        title="上漲機率分佈",
        xaxis_title="P(上漲)",
        yaxis_title="次數",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def _run_walk_forward(
    stock_id: str,
    model_type: str,
    lookback: int,
    forward_days: int,
    train_window: int,
    test_window: int,
    step_size: int,
    threshold: float,
) -> None:
    """執行 Walk-Forward 驗證並顯示結果。"""
    from src.backtest.walk_forward import WalkForwardEngine
    from src.strategy.ml_strategy import MLStrategy

    with st.spinner(f"執行 Walk-Forward 驗證（窗口: {train_window}/{test_window}）..."):
        try:
            engine = WalkForwardEngine(
                strategy_cls=MLStrategy,
                stock_id=stock_id,
                start_date="2020-01-01",
                end_date="2030-12-31",
                train_window=train_window,
                test_window=test_window,
                step_size=step_size,
                strategy_params={
                    "model_type": model_type,
                    "lookback": lookback,
                    "forward_days": forward_days,
                    "threshold": threshold,
                },
            )
            result = engine.run()
        except ValueError as e:
            st.error(str(e))
            return

    # 績效摘要
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("總報酬率", f"{result.total_return:.2f}%")
    c2.metric("年化報酬率", f"{result.annual_return:.2f}%")
    c3.metric("Sharpe", f"{result.sharpe_ratio or 'N/A'}")
    c4.metric("最大回撤", f"{result.max_drawdown:.2f}%")
    c5.metric("交易次數", result.total_trades)

    c1, c2, c3 = st.columns(3)
    c1.metric("勝率", f"{result.win_rate or 'N/A'}%")
    c2.metric("Profit Factor", f"{result.profit_factor or 'N/A'}")
    c3.metric("Fold 數", result.total_folds)

    # 權益曲線
    if result.equity_curve:
        st.subheader("Walk-Forward 權益曲線")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=result.equity_curve,
                mode="lines",
                line=dict(color="#2196F3", width=1.5),
                name="權益",
            )
        )
        fig.add_hline(
            y=result.equity_curve[0],
            line_dash="dash",
            line_color="gray",
            annotation_text="初始資金",
        )
        fig.update_layout(
            xaxis_title="交易日",
            yaxis_title="權益 (NTD)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 各 Fold 報酬表
    if result.folds:
        st.subheader("各 Fold 績效")
        fold_data = []
        for f in result.folds:
            fold_data.append(
                {
                    "Fold": f.fold_idx,
                    "訓練期": f"{f.train_start} ~ {f.train_end}",
                    "測試期": f"{f.test_start} ~ {f.test_end}",
                    "報酬率": f"{f.total_return:.2f}%",
                    "Sharpe": f"{f.sharpe_ratio:.4f}" if f.sharpe_ratio else "N/A",
                    "交易數": f.trades,
                }
            )

        st.dataframe(
            pd.DataFrame(fold_data),
            use_container_width=True,
            hide_index=True,
        )

        # Fold 報酬柱狀圖
        fold_returns = [f.total_return for f in result.folds]
        fold_labels = [f"Fold {f.fold_idx}" for f in result.folds]
        colors = ["#26A69A" if r >= 0 else "#EF5350" for r in fold_returns]

        fig = go.Figure(
            go.Bar(
                x=fold_labels,
                y=fold_returns,
                marker_color=colors,
            )
        )
        fig.update_layout(
            title="各 Fold 報酬率",
            yaxis_title="報酬率 (%)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
