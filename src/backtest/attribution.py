"""策略績效歸因分析 — 計算五因子暴露與期間報酬相關係數。

五因子定義（均在進場日評估）：
  momentum  — 20 日動能（進場前 20 日累積報酬）
  reversal  — 5 日短線反轉（進場前 5 日累積報酬，負值代表超賣反彈機會）
  quality   — 品質代理（RSI14，衡量相對強弱）
  size      — 規模代理（對數 20 日平均成交量）
  liquidity — 流動性（相對量能 = 進場日量 / 20 日均量）
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorAttributionResult:
    """五因子歸因計算結果。"""

    correlations: dict[str, float | None]
    """各因子與交易報酬的 Pearson 相關係數，資料不足時為 None。"""

    factor_exposures: dict[str, list[float]]
    """各因子在每筆交易進場時的暴露值（與 trade_returns 等長）。"""

    trade_returns: list[float]
    """每筆交易的報酬率（%）。"""

    n_trades: int
    """有效交易筆數。"""

    factor_labels: dict[str, str] = field(default_factory=dict)
    """因子英文名 → 中文顯示名稱。"""


class FactorAttribution:
    """策略績效因子歸因分析器。"""

    MIN_TRADES = 3  # 計算相關係數所需的最少交易筆數
    LOOKBACK_MOMENTUM = 20  # 動能回顧期（交易日）
    LOOKBACK_REVERSAL = 5  # 短線反轉回顧期（交易日）
    LOOKBACK_VOL = 20  # 計算平均量所用的回顧期（交易日）

    FACTOR_LABELS: dict[str, str] = {
        "momentum": "動能(20d)",
        "reversal": "反轉(5d)",
        "quality": "品質(RSI)",
        "size": "規模(對數量)",
        "liquidity": "流動性(相對量)",
    }

    def compute(
        self,
        backtest_result: object,
        data: pd.DataFrame,
    ) -> FactorAttributionResult:
        """計算五因子歸因（從 BacktestResultData）。

        Parameters
        ----------
        backtest_result:
            BacktestResultData 物件（含 .trades 屬性，每筆 TradeRecord 有
            entry_date、return_pct 欄位）。
        data:
            Strategy.load_data() 回傳的 DataFrame（DatetimeIndex，含 close、volume，
            以及選用的 rsi_14 欄位）。

        Returns
        -------
        FactorAttributionResult
        """
        trades = getattr(backtest_result, "trades", [])
        trade_dicts = [
            {
                "entry_date": t.entry_date,
                "return_pct": t.return_pct,
            }
            for t in trades
            if t.exit_date is not None  # 排除未平倉
        ]
        return self._compute_from_dicts(trade_dicts, data)

    def compute_from_df(
        self,
        trades_df: pd.DataFrame,
        data: pd.DataFrame,
    ) -> FactorAttributionResult:
        """計算五因子歸因（從 DataFrame，適合 Dashboard 使用）。

        Parameters
        ----------
        trades_df:
            包含 entry_date、return_pct 欄位的 DataFrame（load_trades() 回傳值）。
        data:
            含 close、volume 的日K DataFrame（DatetimeIndex）。
        """
        if trades_df.empty or "entry_date" not in trades_df.columns or "return_pct" not in trades_df.columns:
            return self._empty_result()

        trade_dicts = trades_df[["entry_date", "return_pct"]].dropna().to_dict("records")
        return self._compute_from_dicts(trade_dicts, data)

    # ------------------------------------------------------------------ #
    #  內部計算
    # ------------------------------------------------------------------ #

    def _compute_from_dicts(
        self,
        trade_dicts: list[dict],
        data: pd.DataFrame,
    ) -> FactorAttributionResult:
        """核心計算流程。"""
        if not trade_dicts or data.empty:
            return self._empty_result()

        # 確保 index 為 DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("data.index 非 DatetimeIndex，嘗試轉換")
            try:
                data = data.copy()
                data.index = pd.to_datetime(data.index)
            except Exception:
                return self._empty_result()

        close = data["close"] if "close" in data.columns else None
        volume = data["volume"] if "volume" in data.columns else None
        rsi_col = self._find_rsi_column(data)

        # 計算每筆交易的因子暴露
        momentum_vals: list[float] = []
        reversal_vals: list[float] = []
        quality_vals: list[float] = []
        size_vals: list[float] = []
        liquidity_vals: list[float] = []
        trade_returns: list[float] = []

        for td in trade_dicts:
            entry_date = td["entry_date"]
            return_pct = float(td["return_pct"])

            # 在 index 中定位進場日
            entry_ts = pd.Timestamp(entry_date)
            idx = data.index.searchsorted(entry_ts)

            # 確保進場日存在 & 有足夠歷史
            if idx >= len(data.index):
                continue
            if data.index[idx] != entry_ts:
                # 若進場日不在 index 中（例如休市），取最近前一交易日
                logger.warning("Attribution: 進場日 %s 不在價格資料中，改用前一交易日", entry_date)
                idx = max(0, idx - 1)

            trade_returns.append(return_pct)

            # --- momentum: 20 日動能 ---
            m_val = self._momentum(close, idx, self.LOOKBACK_MOMENTUM)
            momentum_vals.append(m_val if m_val is not None else float("nan"))

            # --- reversal: 5 日短線反轉（負動能 = 超賣反彈機會，越負越高） ---
            # 取負值：近 5 日跌幅越大 → reversal 值越正 → 反彈機會越高
            _r = self._momentum(close, idx, self.LOOKBACK_REVERSAL)
            reversal_vals.append(-_r if _r is not None else float("nan"))

            # --- quality: RSI ---
            q_val = self._rsi_at(data, rsi_col, idx)
            quality_vals.append(q_val if q_val is not None else float("nan"))

            # --- size: 對數 20 日均量 ---
            s_val = self._log_avg_volume(volume, idx, self.LOOKBACK_VOL)
            size_vals.append(s_val if s_val is not None else float("nan"))

            # --- liquidity: 相對量 ---
            l_val = self._relative_volume(volume, idx, self.LOOKBACK_VOL)
            liquidity_vals.append(l_val if l_val is not None else float("nan"))

        n = len(trade_returns)
        if n < self.MIN_TRADES:
            return self._empty_result()

        returns_arr = np.array(trade_returns, dtype=float)

        exposures_raw = {
            "momentum": momentum_vals,
            "reversal": reversal_vals,
            "quality": quality_vals,
            "size": size_vals,
            "liquidity": liquidity_vals,
        }

        # 計算 Pearson 相關係數（僅用兩者均非 NaN 的交易）
        correlations: dict[str, float | None] = {}
        factor_exposures: dict[str, list[float]] = {}

        for fname, vals in exposures_raw.items():
            arr = np.array(vals, dtype=float)
            valid = ~(np.isnan(arr) | np.isnan(returns_arr))
            factor_exposures[fname] = arr.tolist()

            if valid.sum() < self.MIN_TRADES:
                correlations[fname] = None
                continue

            corr = self._pearson_r(arr[valid], returns_arr[valid])
            correlations[fname] = corr

        return FactorAttributionResult(
            correlations=correlations,
            factor_exposures=factor_exposures,
            trade_returns=trade_returns,
            n_trades=n,
            factor_labels=dict(self.FACTOR_LABELS),
        )

    # ------------------------------------------------------------------ #
    #  因子計算輔助函數（純函數，方便單獨測試）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _momentum(
        close: pd.Series | None,
        idx: int,
        lookback: int,
    ) -> float | None:
        """計算進場日前 lookback 交易日的累積報酬率（%）。"""
        if close is None:
            return None
        if idx < lookback:
            return None
        past_close = close.iloc[idx - lookback]
        cur_close = close.iloc[idx]
        if past_close <= 0:
            return None
        return (cur_close / past_close - 1.0) * 100.0

    @staticmethod
    def _rsi_at(
        data: pd.DataFrame,
        rsi_col: str | None,
        idx: int,
    ) -> float | None:
        """從已載入的 RSI 欄位取得進場日的值。"""
        if rsi_col is None:
            return None
        val = data[rsi_col].iloc[idx]
        if pd.isna(val):
            return None
        return float(val)

    @staticmethod
    def _log_avg_volume(
        volume: pd.Series | None,
        idx: int,
        lookback: int,
    ) -> float | None:
        """計算進場日前 lookback 日平均成交量的自然對數。"""
        if volume is None:
            return None
        start = max(0, idx - lookback)
        window = volume.iloc[start:idx]
        if window.empty or window.mean() <= 0:
            return None
        return math.log(window.mean())

    @staticmethod
    def _relative_volume(
        volume: pd.Series | None,
        idx: int,
        lookback: int,
    ) -> float | None:
        """計算進場日成交量相對於前 lookback 日均量的倍數。"""
        if volume is None:
            return None
        start = max(0, idx - lookback)
        window = volume.iloc[start:idx]
        avg = window.mean() if not window.empty else 0
        if avg <= 0:
            return None
        cur_vol = volume.iloc[idx]
        return float(cur_vol) / avg

    @staticmethod
    def _find_rsi_column(data: pd.DataFrame) -> str | None:
        """在 data 中尋找 RSI 欄位（優先 rsi_14，次之含 rsi 的任意欄）。"""
        if "rsi_14" in data.columns:
            return "rsi_14"
        rsi_cols = [c for c in data.columns if "rsi" in c.lower()]
        return rsi_cols[0] if rsi_cols else None

    @staticmethod
    def _pearson_r(x: np.ndarray, y: np.ndarray) -> float | None:
        """計算兩向量的 Pearson 相關係數。標準差為 0 時回傳 None。"""
        if len(x) < 2:
            return None
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x == 0 or std_y == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def _empty_result() -> FactorAttributionResult:
        """交易筆數不足時回傳空結果。"""
        return FactorAttributionResult(
            correlations={k: None for k in FactorAttribution.FACTOR_LABELS},
            factor_exposures={k: [] for k in FactorAttribution.FACTOR_LABELS},
            trade_returns=[],
            n_trades=0,
            factor_labels=dict(FactorAttribution.FACTOR_LABELS),
        )
