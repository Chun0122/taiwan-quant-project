"""投資組合配置計算模組 — risk_parity / mean_variance 權重計算。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def risk_parity_weights(returns: pd.DataFrame) -> dict[str, float]:
    """風險平價配置 — 使各資產的風險貢獻相等。

    波動大的股票分配較少權重，使每支股票對組合總風險的貢獻一致。

    Parameters
    ----------
    returns : pd.DataFrame
        各股票日報酬率，columns = stock_ids。

    Returns
    -------
    dict[str, float]
        正規化權重字典（和 = 1）。
    """
    from scipy.optimize import minimize

    stock_ids = list(returns.columns)
    n = len(stock_ids)

    if n == 1:
        return {stock_ids[0]: 1.0}

    cov = returns.cov().values

    def objective(w: np.ndarray) -> float:
        """最小化各資產風險貢獻比例的差異（目標為均等 1/n）。"""
        w = np.array(w)
        port_var = w @ cov @ w
        if port_var < 1e-16:
            return 0.0
        # 風險貢獻 = w_i * (Σw)_i
        risk_contrib = w * (cov @ w)
        # 各資產風險貢獻占比
        rc_pct = risk_contrib / port_var
        target = 1.0 / n
        return float(np.sum((rc_pct - target) ** 2))

    w0 = np.ones(n) / n
    bounds = [(0.01, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        weights = result.x / result.x.sum()
    else:
        logger.warning("Risk parity 優化未收斂，使用等權重")
        weights = np.ones(n) / n

    return {sid: float(w) for sid, w in zip(stock_ids, weights)}


def mean_variance_weights(returns: pd.DataFrame) -> dict[str, float]:
    """均值-方差優化 — 最大化 Sharpe ratio（Markowitz）。

    Parameters
    ----------
    returns : pd.DataFrame
        各股票日報酬率，columns = stock_ids。

    Returns
    -------
    dict[str, float]
        正規化權重字典（和 = 1）。
    """
    from scipy.optimize import minimize

    stock_ids = list(returns.columns)
    n = len(stock_ids)

    if n == 1:
        return {stock_ids[0]: 1.0}

    mean_returns = returns.mean().values
    cov = returns.cov().values

    def neg_sharpe(w: np.ndarray) -> float:
        """負 Sharpe ratio（最小化 = 最大化 Sharpe）。"""
        port_return = w @ mean_returns
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 0.0
        return -port_return / port_vol

    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        weights = result.x / result.x.sum()
    else:
        logger.warning("Mean-variance 優化未收斂，fallback 到等權重")
        weights = np.ones(n) / n

    return {sid: float(w) for sid, w in zip(stock_ids, weights)}
