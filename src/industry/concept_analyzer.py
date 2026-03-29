"""概念股輪動分析模組。

提供概念熱度計算、概念成員管理、及價格相關性候選推薦。

純函數（不依賴 DB）：
  compute_concept_momentum()          ← 概念平均價格動能
  compute_concept_institutional_flow() ← 概念加權法人淨買超
  compute_concept_correlation_candidates() ← 相關性候選推薦（P2）

類別：
  ConceptRotationAnalyzer             ← 概念輪動分析引擎（依賴 DB）
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 法人分類常數（與 analyzer.py 保持一致）
# ---------------------------------------------------------------------------
_FOREIGN_NAMES: frozenset[str] = frozenset(["外資", "外資及陸資", "Foreign_Investor", "外資自營商"])
_TRUST_NAMES: frozenset[str] = frozenset(["投信", "Investment_Trust"])


# ---------------------------------------------------------------------------
# 純函數
# ---------------------------------------------------------------------------


def compute_concept_momentum(
    concept_stocks: dict[str, list[str]],
    df_price: pd.DataFrame,
    lookback_days: int = 20,
) -> pd.Series:
    """計算各概念的平均價格動能（純函數，不依賴 DB）。

    Parameters
    ----------
    concept_stocks:
        {concept_name: [stock_ids]} 概念成員清單。
    df_price:
        日K線 DataFrame，需含 stock_id, date, close 欄位。
    lookback_days:
        計算動能的回溯天數（預設 20 日）。

    Returns
    -------
    pd.Series
        index=concept_name，value=各成員近 lookback_days 日平均報酬率（%）。
        無法計算的概念填 0.0。
    """
    if df_price.empty or not concept_stocks:
        return pd.Series(dtype=float)

    df = df_price.copy()
    df = df.sort_values(["stock_id", "date"])

    # 計算各股近 lookback_days 日報酬率
    returns: dict[str, float] = {}
    for sid, grp in df.groupby("stock_id"):
        grp = grp.tail(lookback_days + 1)
        if len(grp) < 2:
            continue
        first_close = grp["close"].iloc[0]
        last_close = grp["close"].iloc[-1]
        if first_close > 0:
            returns[str(sid)] = (last_close / first_close - 1) * 100

    result: dict[str, float] = {}
    for concept, stocks in concept_stocks.items():
        valid = [returns[s] for s in stocks if s in returns]
        result[concept] = float(np.mean(valid)) if valid else 0.0

    return pd.Series(result, dtype=float)


def compute_concept_institutional_flow(
    concept_stocks: dict[str, list[str]],
    df_inst: pd.DataFrame,
    trust_weight: float = 0.7,
    foreign_weight: float = 0.3,
) -> pd.Series:
    """計算各概念的加權法人淨買超（純函數，不依賴 DB）。

    Parameters
    ----------
    concept_stocks:
        {concept_name: [stock_ids]} 概念成員清單。
    df_inst:
        法人買賣超 DataFrame，需含 stock_id, name, net_buy 欄位。
    trust_weight:
        投信權重（預設 0.7，因台股投信對題材股影響更顯著）。
    foreign_weight:
        外資權重（預設 0.3）。

    Returns
    -------
    pd.Series
        index=concept_name，value=加權平均淨買超（單位：股）。
    """
    if df_inst.empty or not concept_stocks:
        return pd.Series(dtype=float)

    df = df_inst.copy()

    # 計算加權淨買超
    def _weight(name: str) -> float:
        if name in _TRUST_NAMES:
            return trust_weight
        if name in _FOREIGN_NAMES:
            return foreign_weight
        return 0.0

    df["inst_weight"] = df["name"].apply(_weight)
    df["weighted_net"] = df["net_buy"] * df["inst_weight"]

    # 各股加總
    stock_flow = df.groupby("stock_id")["weighted_net"].sum()

    result: dict[str, float] = {}
    for concept, stocks in concept_stocks.items():
        valid = [stock_flow[s] for s in stocks if s in stock_flow.index]
        result[concept] = float(np.mean(valid)) if valid else 0.0

    return pd.Series(result, dtype=float)


def compute_concept_correlation_candidates(
    concept_name: str,
    seed_stocks: list[str],
    candidate_stocks: list[str],
    df_price: pd.DataFrame,
    lookback_days: int = 60,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """找出與種子股平均相關係數 ≥ threshold 的候選股（純函數，P2）。

    Parameters
    ----------
    concept_name:
        概念名稱（僅供回傳 DataFrame 標記用）。
    seed_stocks:
        已知概念成員（種子股），用來計算相關性基準。
    candidate_stocks:
        候選股池（排除已在種子股的股票）。
    df_price:
        日K線 DataFrame，需含 stock_id, date, close 欄位。
    lookback_days:
        相關性計算的回溯天數（預設 60 日）。
    threshold:
        與種子股平均相關係數的最低門檻（預設 0.7）。

    Returns
    -------
    pd.DataFrame
        columns=[stock_id, avg_corr]，依 avg_corr 降序排列，
        不含已在種子股中的股票。
    """
    if df_price.empty or not seed_stocks or not candidate_stocks:
        return pd.DataFrame(columns=["stock_id", "avg_corr"])

    df = df_price.copy()
    df = df.sort_values(["stock_id", "date"])

    # 取最近 lookback_days 日，樞紐成寬表
    cutoff = df["date"].max() - timedelta(days=lookback_days + 10)
    df = df[df["date"] >= cutoff].tail(lookback_days * 10)  # safety margin
    pivot = df.pivot_table(index="date", columns="stock_id", values="close")
    returns = pivot.pct_change().dropna(how="all")

    seed_cols = [s for s in seed_stocks if s in returns.columns]
    if not seed_cols:
        return pd.DataFrame(columns=["stock_id", "avg_corr"])

    # 計算候選股與所有種子股的相關係數均值
    cands = [s for s in candidate_stocks if s in returns.columns and s not in seed_stocks]
    if not cands:
        return pd.DataFrame(columns=["stock_id", "avg_corr"])

    records: list[dict] = []
    seed_matrix = returns[seed_cols]
    for cand in cands:
        corrs = [
            returns[cand].corr(seed_matrix[s])
            for s in seed_cols
            if returns[cand].notna().sum() > 10 and seed_matrix[s].notna().sum() > 10
        ]
        if not corrs:
            continue
        avg = float(np.nanmean(corrs))
        if avg >= threshold:
            records.append({"stock_id": cand, "avg_corr": avg})

    result = pd.DataFrame(records, columns=["stock_id", "avg_corr"])
    return result.sort_values("avg_corr", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# ConceptRotationAnalyzer
# ---------------------------------------------------------------------------


class ConceptRotationAnalyzer:
    """概念股輪動分析引擎。

    從 DB 讀取 ConceptMembership，計算概念熱度排名，
    並輸出每支股票的概念加成分數供 Scanner Stage 3.3b 使用。
    """

    def __init__(
        self,
        lookback_days: int = 20,
        momentum_days: int = 60,
    ) -> None:
        self.lookback_days = lookback_days
        self.momentum_days = momentum_days

    # ------------------------------------------------------------------
    # 資料存取
    # ------------------------------------------------------------------

    def get_concept_stocks(self) -> dict[str, list[str]]:
        """從 DB 查詢 ConceptMembership，回傳 {concept_name: [stock_ids]}。"""
        from src.data.database import get_session
        from src.data.schema import ConceptMembership

        with get_session() as session:
            rows = session.query(
                ConceptMembership.concept_name,
                ConceptMembership.stock_id,
            ).all()

        result: dict[str, list[str]] = {}
        for concept, stock in rows:
            result.setdefault(concept, []).append(stock)
        return result

    def get_stock_concepts(self) -> dict[str, list[str]]:
        """從 DB 查詢 ConceptMembership，回傳 {stock_id: [concept_names]}（多對多）。"""
        from src.data.database import get_session
        from src.data.schema import ConceptMembership

        with get_session() as session:
            rows = session.query(
                ConceptMembership.stock_id,
                ConceptMembership.concept_name,
            ).all()

        result: dict[str, list[str]] = {}
        for stock, concept in rows:
            result.setdefault(stock, []).append(concept)
        return result

    # ------------------------------------------------------------------
    # 核心計算
    # ------------------------------------------------------------------

    def _load_price_data(self, stock_ids: list[str], days: int) -> pd.DataFrame:
        """從 DB 載入日K線資料。"""
        from src.data.database import get_session
        from src.data.schema import DailyPrice

        cutoff = date.today() - timedelta(days=days + 10)
        with get_session() as session:
            rows = (
                session.query(
                    DailyPrice.stock_id,
                    DailyPrice.date,
                    DailyPrice.close,
                )
                .filter(
                    DailyPrice.stock_id.in_(stock_ids),
                    DailyPrice.date >= cutoff,
                )
                .all()
            )

        if not rows:
            return pd.DataFrame(columns=["stock_id", "date", "close"])
        return pd.DataFrame(rows, columns=["stock_id", "date", "close"])

    def _load_inst_data(self, stock_ids: list[str], days: int) -> pd.DataFrame:
        """從 DB 載入法人買賣超資料。"""
        from src.data.database import get_session
        from src.data.schema import InstitutionalInvestor

        cutoff = date.today() - timedelta(days=days + 5)
        with get_session() as session:
            rows = (
                session.query(
                    InstitutionalInvestor.stock_id,
                    InstitutionalInvestor.name,
                    InstitutionalInvestor.net,
                )
                .filter(
                    InstitutionalInvestor.stock_id.in_(stock_ids),
                    InstitutionalInvestor.date >= cutoff,
                )
                .all()
            )

        if not rows:
            return pd.DataFrame(columns=["stock_id", "name", "net_buy"])
        df = pd.DataFrame(rows, columns=["stock_id", "name", "net_buy"])
        return df

    def rank_concepts(
        self,
        inst_weight: float = 0.5,
        momentum_weight: float = 0.5,
        trust_weight: float = 0.7,
        foreign_weight: float = 0.3,
    ) -> pd.DataFrame:
        """綜合排名各概念（法人動能 + 價格動能）。

        使用 Percentile Rank（rank(pct=True)）取代 Min-Max，對極端值更 robust。

        Returns
        -------
        pd.DataFrame
            columns=[rank, concept, concept_score, institutional_score,
                     momentum_score, member_count]
        """
        concept_stocks = self.get_concept_stocks()
        if not concept_stocks:
            return pd.DataFrame(
                columns=["rank", "concept", "concept_score", "institutional_score", "momentum_score", "member_count"]
            )

        all_stocks = list({s for stocks in concept_stocks.values() for s in stocks})

        df_price = self._load_price_data(all_stocks, self.momentum_days)
        df_inst = self._load_inst_data(all_stocks, self.lookback_days)

        # 計算各概念原始分數
        momentum_scores = compute_concept_momentum(concept_stocks, df_price, self.lookback_days)
        inst_scores = compute_concept_institutional_flow(
            concept_stocks, df_inst, trust_weight=trust_weight, foreign_weight=foreign_weight
        )

        # 對齊索引
        concepts = sorted(concept_stocks.keys())
        momentum_s = momentum_scores.reindex(concepts).fillna(0.0)
        inst_s = inst_scores.reindex(concepts).fillna(0.0)

        # Percentile Rank（0~1，越高越好）
        def _pct_rank(s: pd.Series) -> pd.Series:
            if s.nunique() <= 1:
                return pd.Series(0.5, index=s.index)
            return s.rank(pct=True)

        momentum_pct = _pct_rank(momentum_s)
        inst_pct = _pct_rank(inst_s)

        concept_score = inst_weight * inst_pct + momentum_weight * momentum_pct

        df = pd.DataFrame(
            {
                "concept": concepts,
                "concept_score": concept_score.values,
                "institutional_score": inst_pct.values,
                "momentum_score": momentum_pct.values,
                "member_count": [len(concept_stocks[c]) for c in concepts],
            }
        )
        df = df.sort_values("concept_score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    def compute_concept_scores_for_stocks(
        self,
        stock_ids: list[str],
        bonus_range: float = 0.05,
    ) -> pd.DataFrame:
        """計算每支股票的概念熱度加成分數。

        一股多概念時取最高加成（不累加）。

        Parameters
        ----------
        stock_ids:
            需要計算加成的股票清單。
        bonus_range:
            加成範圍，[-bonus_range, +bonus_range]（預設 ±5%）。

        Returns
        -------
        pd.DataFrame
            columns=[stock_id, concept_bonus]，bonus 範圍 [-bonus_range, +bonus_range]。
        """
        ranked = self.rank_concepts()
        if ranked.empty:
            return pd.DataFrame({"stock_id": stock_ids, "concept_bonus": [0.0] * len(stock_ids)})

        # concept_score 轉換為 [-bonus_range, +bonus_range]
        # 中位數對應 0，最高對應 +bonus_range，最低對應 -bonus_range
        score_col = ranked.set_index("concept")["concept_score"]
        min_s, max_s = score_col.min(), score_col.max()
        rng = max_s - min_s if max_s > min_s else 1.0
        concept_bonus_map = ((score_col - min_s) / rng * 2 - 1) * bonus_range  # [-range, +range]

        stock_concepts = self.get_stock_concepts()

        records: list[dict] = []
        for sid in stock_ids:
            concepts = stock_concepts.get(sid, [])
            bonuses = [concept_bonus_map.get(c, 0.0) for c in concepts]
            # 取最高加成（最強概念決定加成方向）
            bonus = float(max(bonuses)) if bonuses else 0.0
            records.append({"stock_id": sid, "concept_bonus": bonus})

        return pd.DataFrame(records)
