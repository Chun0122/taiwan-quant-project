"""每日選股報告引擎 — 四維度（技術/籌碼/基本面/ML）綜合評分排名。

使用方式：
    engine = DailyReportEngine()
    df = engine.run()
    print(df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import settings
from src.data.database import init_db
from src.screener.engine import MultiFactorScreener

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "technical": 0.30,
    "chip": 0.30,
    "fundamental": 0.20,
    "ml": 0.20,
}


class DailyReportEngine:
    """每日選股報告引擎。"""

    def __init__(
        self,
        watchlist: list[str] | None = None,
        weights: dict[str, float] | None = None,
        lookback_days: int = 5,
        ml_enabled: bool = True,
        start_date: str | None = None,
    ) -> None:
        self.watchlist = watchlist or settings.fetcher.watchlist
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.lookback_days = lookback_days
        self.ml_enabled = ml_enabled
        self.start_date = start_date or settings.fetcher.default_start_date
        self._screener = MultiFactorScreener(
            watchlist=self.watchlist, lookback_days=lookback_days
        )
        init_db()

    def run(self) -> pd.DataFrame:
        """對每支股票計算四維度分數，加權合成排名。

        Returns:
            DataFrame: [rank, stock_id, close, technical_score, chip_score,
                        fundamental_score, ml_score, composite_score,
                        rsi, macd, foreign_net, yoy_growth]
        """
        results = []

        for stock_id in self.watchlist:
            try:
                snapshot = self._screener._load_snapshot(stock_id)
                if snapshot.empty:
                    logger.warning("[%s] 無資料，跳過", stock_id)
                    continue

                latest = snapshot.iloc[-1]

                tech = self._compute_technical_score(snapshot)
                chip = self._compute_chip_score(snapshot)
                fund = self._compute_fundamental_score(snapshot)
                ml = self._compute_ml_score(stock_id) if self.ml_enabled else 0.5

                composite = self._compute_composite(tech, chip, fund, ml)

                results.append({
                    "stock_id": stock_id,
                    "close": latest.get("close", 0),
                    "technical_score": round(tech, 3),
                    "chip_score": round(chip, 3),
                    "fundamental_score": round(fund, 3),
                    "ml_score": round(ml, 3),
                    "composite_score": round(composite, 3),
                    "rsi": round(latest.get("rsi_14", 50), 1) if pd.notna(latest.get("rsi_14")) else None,
                    "macd": round(latest.get("macd", 0), 2) if pd.notna(latest.get("macd")) else None,
                    "foreign_net": latest.get("foreign_net", 0) if pd.notna(latest.get("foreign_net")) else None,
                    "yoy_growth": round(latest.get("yoy_growth", 0), 1) if pd.notna(latest.get("yoy_growth")) else None,
                })
            except Exception:
                logger.exception("[%s] 報告計算失敗", stock_id)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        cols = [
            "rank", "stock_id", "close", "technical_score", "chip_score",
            "fundamental_score", "ml_score", "composite_score",
            "rsi", "macd", "foreign_net", "yoy_growth",
        ]
        return df[[c for c in cols if c in df.columns]]

    def _compute_technical_score(self, snapshot: pd.DataFrame) -> float:
        """技術面分數 (0~1)。

        - RSI: 越接近中性（50）越好，偏高偏低扣分
        - MACD: MACD > Signal 加分
        - SMA: 收盤價 > SMA20 加分
        """
        latest = snapshot.iloc[-1]

        # RSI score: rsi=30 → 1.0, rsi=50 → 0.5, rsi=70 → 0.0
        rsi = latest.get("rsi_14", 50) if pd.notna(latest.get("rsi_14")) else 50
        rsi_score = np.clip((70 - rsi) / 40, 0, 1)

        # MACD score: macd > signal → positive
        macd_val = latest.get("macd", 0) if pd.notna(latest.get("macd")) else 0
        signal_val = latest.get("macd_signal", 0) if pd.notna(latest.get("macd_signal")) else 0
        close = latest.get("close", 1)
        if close > 0:
            macd_score = np.clip(0.5 + (macd_val - signal_val) / (close * 0.02), 0, 1)
        else:
            macd_score = 0.5

        # SMA score: close > sma_20 → 1.0
        sma_20 = latest.get("sma_20") if pd.notna(latest.get("sma_20")) else None
        if sma_20 and sma_20 > 0:
            sma_score = 1.0 if close > sma_20 else 0.0
        else:
            sma_score = 0.5

        return float(np.mean([rsi_score, macd_score, sma_score]))

    def _compute_chip_score(self, snapshot: pd.DataFrame) -> float:
        """籌碼面分數 (0~1)。

        - 外資淨買超天數比例
        - 三大法人淨買超天數比例
        - 融資餘額變化（減少加分）
        """
        n = len(snapshot)
        if n == 0:
            return 0.5

        # 外資買超比例
        foreign = snapshot.get("foreign_net")
        if foreign is not None and not foreign.isna().all():
            foreign_ratio = (foreign.dropna() > 0).sum() / max(foreign.dropna().count(), 1)
        else:
            foreign_ratio = 0.5

        # 三大法人合計買超比例
        trust = snapshot.get("trust_net")
        dealer = snapshot.get("dealer_net")

        inst_total = pd.Series(0, index=snapshot.index, dtype=float)
        for col in [foreign, trust, dealer]:
            if col is not None and not col.isna().all():
                inst_total = inst_total + col.fillna(0)

        inst_ratio = (inst_total > 0).sum() / max(len(inst_total), 1)

        # 融資餘額變化（減少 → 加分）
        margin = snapshot.get("margin_balance")
        if margin is not None and not margin.isna().all() and len(margin.dropna()) >= 2:
            margin_clean = margin.dropna()
            margin_change = (margin_clean.iloc[-1] - margin_clean.iloc[0]) / max(margin_clean.iloc[0], 1)
            margin_score = np.clip(0.5 - margin_change * 2, 0, 1)
        else:
            margin_score = 0.5

        return float(0.4 * foreign_ratio + 0.4 * inst_ratio + 0.2 * margin_score)

    def _compute_fundamental_score(self, snapshot: pd.DataFrame) -> float:
        """基本面分數 (0~1)。

        - YoY 營收成長率
        - MoM 成長加分
        """
        latest = snapshot.iloc[-1]

        yoy = latest.get("yoy_growth", 0)
        if pd.isna(yoy):
            yoy = 0
        yoy_score = np.clip(yoy / 50, 0, 1)

        mom = latest.get("mom_growth", 0)
        if pd.isna(mom):
            mom = 0
        mom_bonus = 0.1 if mom > 0 else 0

        return float(np.clip(yoy_score + mom_bonus, 0, 1))

    def _compute_ml_score(self, stock_id: str) -> float:
        """ML 分數 (0~1)。

        使用 RandomForest 策略 predict_proba 作為信心指標。
        失敗時回傳 0.5（中性）。
        """
        try:
            from datetime import date
            from src.strategy.ml_strategy import MLStrategy

            end = date.today().isoformat()
            strategy = MLStrategy(
                stock_id=stock_id,
                start_date=self.start_date,
                end_date=end,
                model_type="random_forest",
            )
            data = strategy.load_data()
            if data.empty or len(data) < 60:
                return 0.5

            signals = strategy.generate_signals(data)

            # 取最後一筆信號作為 ML 分數
            if hasattr(strategy, '_last_proba') and strategy._last_proba is not None:
                return float(np.clip(strategy._last_proba, 0, 1))

            last_signal = signals.iloc[-1] if len(signals) > 0 else 0
            if last_signal == 1:
                return 0.8
            elif last_signal == -1:
                return 0.2
            else:
                return 0.5

        except Exception:
            logger.debug("[%s] ML 分數計算失敗，使用中性值", stock_id)
            return 0.5

    def _compute_composite(
        self, tech: float, chip: float, fund: float, ml: float
    ) -> float:
        """加權合成分數。"""
        return (
            self.weights["technical"] * tech
            + self.weights["chip"] * chip
            + self.weights["fundamental"] * fund
            + self.weights["ml"] * ml
        )
