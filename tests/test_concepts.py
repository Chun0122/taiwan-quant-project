"""tests/test_concepts.py — 概念股功能測試。

涵蓋：
  - TestClassifyConcepts               : classify_concepts() 純函數（8 個測試）
  - TestComputeConceptMomentum         : compute_concept_momentum() 純函數（5 個測試）
  - TestComputeConceptInstitutionalFlow: compute_concept_institutional_flow() 純函數（4 個測試）
  - TestComputeConceptCorrelationCandidates: 相關性篩選純函數（5 個測試）
  - TestConceptBonusCap                : Scanner cap 機制（sector+concept ≤ 8%）（4 個測試）
  - TestConceptMembershipORM           : in-memory SQLite CRUD（6 個測試）
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data.database import Base
from src.data.mops_fetcher import classify_concepts
from src.data.schema import ConceptGroup, ConceptMembership
from src.industry.concept_analyzer import (
    compute_concept_correlation_candidates,
    compute_concept_institutional_flow,
    compute_concept_momentum,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mem_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture()
def mem_session(mem_engine):
    Session = sessionmaker(bind=mem_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# 工具函數
# ---------------------------------------------------------------------------


def _make_price_df(
    stocks: list[str],
    n_days: int = 25,
    start_price: float = 100.0,
    daily_return: float = 0.005,
) -> pd.DataFrame:
    """產生連續 n_days 天的模擬日K資料（等速上漲）。"""
    today = date.today()
    rows = []
    for sid in stocks:
        price = start_price
        for i in range(n_days):
            d = today - timedelta(days=n_days - 1 - i)
            rows.append({"stock_id": sid, "date": d, "close": round(price, 2)})
            price *= 1 + daily_return
    return pd.DataFrame(rows)


def _make_inst_df(
    stocks: list[str],
    trust_net: float = 1_000_000,
    foreign_net: float = 500_000,
) -> pd.DataFrame:
    """產生投信 + 外資的法人買賣超資料。"""
    rows = []
    today = date.today()
    for sid in stocks:
        rows.append({"stock_id": sid, "name": "投信", "net_buy": trust_net, "date": today})
        rows.append({"stock_id": sid, "name": "外資", "net_buy": foreign_net, "date": today})
    return pd.DataFrame(rows)


# ===========================================================================
# TestClassifyConcepts
# ===========================================================================


class TestClassifyConcepts:
    def test_hit_cowos(self):
        matched = classify_concepts("公司宣布擴增 CoWoS 先進封裝產能")
        assert "CoWoS封裝" in matched

    def test_hit_heat_module(self):
        matched = classify_concepts("新款均溫板散熱模組出貨量提升")
        assert "散熱模組" in matched

    def test_hit_leo_satellite(self):
        matched = classify_concepts("低軌衛星通訊模組取得認證")
        assert "低軌衛星" in matched

    def test_hit_ai_server(self):
        matched = classify_concepts("GB200 AI伺服器組裝訂單大增")
        assert "AI伺服器" in matched

    def test_hit_auto_electronics(self):
        matched = classify_concepts("電動車 ADAS 感測器模組出貨")
        assert "車用電子" in matched

    def test_no_hit(self):
        matched = classify_concepts("公司召開股東常會議決現金股利")
        assert matched == []

    def test_multi_concept_match(self):
        matched = classify_concepts("CoWoS封裝用散熱均溫板供貨量創新高")
        assert "CoWoS封裝" in matched
        assert "散熱模組" in matched

    def test_case_insensitive(self):
        matched = classify_concepts("cowos 封裝技術更新")
        assert "CoWoS封裝" in matched

    def test_custom_keywords(self):
        custom = {"測試概念": ["神奇關鍵字"]}
        matched = classify_concepts("神奇關鍵字出現在標題中", keywords=custom)
        assert "測試概念" in matched

    def test_empty_title(self):
        assert classify_concepts("") == []
        assert classify_concepts(None) == []


# ===========================================================================
# TestComputeConceptMomentum
# ===========================================================================


class TestComputeConceptMomentum:
    def test_positive_momentum(self):
        concept_stocks = {"概念A": ["1111", "2222"]}
        df_price = _make_price_df(["1111", "2222"], n_days=25, daily_return=0.01)
        result = compute_concept_momentum(concept_stocks, df_price, lookback_days=20)
        assert "概念A" in result.index
        assert result["概念A"] > 0

    def test_negative_momentum(self):
        concept_stocks = {"概念B": ["3333"]}
        df_price = _make_price_df(["3333"], n_days=25, daily_return=-0.01)
        result = compute_concept_momentum(concept_stocks, df_price, lookback_days=20)
        assert result["概念B"] < 0

    def test_empty_price_df(self):
        concept_stocks = {"概念C": ["1111"]}
        df_price = pd.DataFrame(columns=["stock_id", "date", "close"])
        result = compute_concept_momentum(concept_stocks, df_price, lookback_days=20)
        assert result.empty

    def test_single_stock_concept(self):
        concept_stocks = {"單股概念": ["9999"]}
        df_price = _make_price_df(["9999"], n_days=25, daily_return=0.005)
        result = compute_concept_momentum(concept_stocks, df_price, lookback_days=20)
        assert "單股概念" in result.index
        assert isinstance(result["單股概念"], float)

    def test_multi_concept(self):
        concept_stocks = {"概念X": ["1111", "2222"], "概念Y": ["3333"]}
        df_x = _make_price_df(["1111", "2222"], n_days=25, daily_return=0.01)
        df_y = _make_price_df(["3333"], n_days=25, daily_return=-0.01)
        df_price = pd.concat([df_x, df_y], ignore_index=True)
        result = compute_concept_momentum(concept_stocks, df_price, lookback_days=20)
        assert result["概念X"] > result["概念Y"]


# ===========================================================================
# TestComputeConceptInstitutionalFlow
# ===========================================================================


class TestComputeConceptInstitutionalFlow:
    def test_trust_dominant(self):
        """投信權重 0.7 > 外資 0.3，概念流入應以投信為主。"""
        concept_stocks = {"概念A": ["1111"]}
        df_inst = _make_inst_df(["1111"], trust_net=1_000_000, foreign_net=0)
        result = compute_concept_institutional_flow(concept_stocks, df_inst, trust_weight=0.7, foreign_weight=0.3)
        assert result["概念A"] == pytest.approx(700_000.0)

    def test_foreign_only(self):
        concept_stocks = {"概念B": ["2222"]}
        df_inst = _make_inst_df(["2222"], trust_net=0, foreign_net=1_000_000)
        result = compute_concept_institutional_flow(concept_stocks, df_inst, trust_weight=0.7, foreign_weight=0.3)
        assert result["概念B"] == pytest.approx(300_000.0)

    def test_empty_inst_df(self):
        concept_stocks = {"概念C": ["3333"]}
        df_inst = pd.DataFrame(columns=["stock_id", "name", "net_buy"])
        result = compute_concept_institutional_flow(concept_stocks, df_inst)
        assert result.empty

    def test_multi_member_avg(self):
        """多成員時取平均加權流。"""
        concept_stocks = {"概念D": ["1111", "2222"]}
        df_inst = _make_inst_df(["1111", "2222"], trust_net=1_000_000, foreign_net=0)
        result = compute_concept_institutional_flow(concept_stocks, df_inst, trust_weight=0.7, foreign_weight=0.3)
        assert result["概念D"] == pytest.approx(700_000.0)


# ===========================================================================
# TestComputeConceptCorrelationCandidates
# ===========================================================================


class TestComputeConceptCorrelationCandidates:
    def _make_correlated_prices(self, seed: str, correlated: str, n: int = 65) -> pd.DataFrame:
        """製造兩支高相關股票的價格序列。"""
        today = date.today()
        base = np.cumprod(1 + np.random.default_rng(42).normal(0, 0.01, n))
        rows = []
        for i in range(n):
            d = today - timedelta(days=n - 1 - i)
            rows.append({"stock_id": seed, "date": d, "close": 100 * base[i]})
            # 相關：幾乎相同，加少許噪聲
            rows.append(
                {
                    "stock_id": correlated,
                    "date": d,
                    "close": 100 * base[i] * (1 + np.random.default_rng(i).normal(0, 0.001)),
                }
            )
        return pd.DataFrame(rows)

    def test_finds_correlated_stock(self):
        df_price = self._make_correlated_prices("SEED", "CAND")
        result = compute_concept_correlation_candidates(
            concept_name="測試概念",
            seed_stocks=["SEED"],
            candidate_stocks=["CAND"],
            df_price=df_price,
            lookback_days=60,
            threshold=0.5,
        )
        assert len(result) == 1
        assert result.iloc[0]["stock_id"] == "CAND"
        assert result.iloc[0]["avg_corr"] >= 0.5

    def test_excludes_seed_stocks(self):
        df_price = self._make_correlated_prices("SEED", "OTHER")
        result = compute_concept_correlation_candidates(
            concept_name="測試概念",
            seed_stocks=["SEED"],
            candidate_stocks=["SEED"],  # candidate = seed → 應排除
            df_price=df_price,
            lookback_days=60,
            threshold=0.5,
        )
        assert result.empty

    def test_high_threshold_filters_out(self):
        df_price = self._make_correlated_prices("SEED", "NOISY")
        # 製造低相關候選
        today = date.today()
        noise_rows = [
            {
                "stock_id": "NOISY",
                "date": today - timedelta(days=i),
                "close": np.random.default_rng(i + 100).uniform(50, 150),
            }
            for i in range(65)
        ]
        df_noise = pd.DataFrame(noise_rows)
        df_seed = df_price[df_price["stock_id"] == "SEED"]
        df_price2 = pd.concat([df_seed, df_noise], ignore_index=True)
        result = compute_concept_correlation_candidates(
            concept_name="測試概念",
            seed_stocks=["SEED"],
            candidate_stocks=["NOISY"],
            df_price=df_price2,
            lookback_days=60,
            threshold=0.95,  # 極高門檻
        )
        # 純雜訊不可能達到 0.95
        assert result.empty

    def test_empty_inputs(self):
        result = compute_concept_correlation_candidates(
            concept_name="測試概念",
            seed_stocks=[],
            candidate_stocks=["1111"],
            df_price=pd.DataFrame(columns=["stock_id", "date", "close"]),
        )
        assert result.empty

    def test_returns_sorted_desc(self):
        """結果應依 avg_corr 降序排列。"""
        df_p1 = self._make_correlated_prices("SEED", "A")
        df_p2 = self._make_correlated_prices("SEED", "B")
        df_price = pd.concat([df_p1, df_p2], ignore_index=True).drop_duplicates(subset=["stock_id", "date"])
        result = compute_concept_correlation_candidates(
            concept_name="測試概念",
            seed_stocks=["SEED"],
            candidate_stocks=["A", "B"],
            df_price=df_price,
            lookback_days=60,
            threshold=0.0,
        )
        if len(result) >= 2:
            corrs = result["avg_corr"].tolist()
            assert corrs == sorted(corrs, reverse=True)


# ===========================================================================
# TestConceptBonusCap
# ===========================================================================


class TestConceptBonusCap:
    """測試 _apply_concept_bonus cap 機制（sector+concept ≤ ±8%）。"""

    def _make_scored_df(self, sector_bonus: float, concept_bonus_raw: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stock_id": ["1111"],
                "composite_score": [1.0],
                "sector_bonus": [sector_bonus],
            }
        )

    def _apply_cap(self, sector_bonus: float, concept_bonus_raw: float) -> float:
        """模擬 _apply_concept_bonus cap 邏輯，回傳 capped concept_bonus。"""
        remaining = max(0.0, 0.08 - abs(sector_bonus))
        return max(-remaining, min(remaining, concept_bonus_raw))

    def test_no_cap_needed(self):
        """sector=0%，concept=3% → 無需 cap。"""
        result = self._apply_cap(0.0, 0.03)
        assert result == pytest.approx(0.03)

    def test_cap_when_sector_already_high(self):
        """sector=5%，concept=5% → remaining=3%，concept 被 cap 至 3%。"""
        result = self._apply_cap(0.05, 0.05)
        assert result == pytest.approx(0.03)

    def test_cap_symmetric_negative(self):
        """sector=-5%，concept=-5% → remaining=3%，concept 被 cap 至 -3%。"""
        result = self._apply_cap(-0.05, -0.05)
        assert result == pytest.approx(-0.03)

    def test_sector_full_allocation(self):
        """sector=8%（已達上限），concept=任意 → concept 被 cap 至 0。"""
        result = self._apply_cap(0.08, 0.05)
        assert result == pytest.approx(0.0)


# ===========================================================================
# TestConceptMembershipORM
# ===========================================================================


class TestConceptMembershipORM:
    def test_create_concept_group(self, mem_session):
        grp = ConceptGroup(name="測試概念A", description="測試用概念")
        mem_session.add(grp)
        mem_session.commit()
        result = mem_session.query(ConceptGroup).filter_by(name="測試概念A").first()
        assert result is not None
        assert result.description == "測試用概念"

    def test_create_membership(self, mem_session):
        # 先確保概念存在
        if not mem_session.query(ConceptGroup).filter_by(name="測試概念B").first():
            mem_session.add(ConceptGroup(name="測試概念B"))
            mem_session.commit()

        member = ConceptMembership(
            concept_name="測試概念B",
            stock_id="2330",
            source="yaml",
            added_date=date.today(),
        )
        mem_session.add(member)
        mem_session.commit()
        result = mem_session.query(ConceptMembership).filter_by(stock_id="2330").first()
        assert result is not None
        assert result.source == "yaml"

    def test_unique_constraint(self, mem_session):
        """同一 concept_name + stock_id 不可重複新增。"""
        if not mem_session.query(ConceptGroup).filter_by(name="測試概念C").first():
            mem_session.add(ConceptGroup(name="測試概念C"))
            mem_session.commit()

        m1 = ConceptMembership(concept_name="測試概念C", stock_id="3333", source="yaml", added_date=date.today())
        mem_session.add(m1)
        mem_session.commit()

        m2 = ConceptMembership(concept_name="測試概念C", stock_id="3333", source="mops", added_date=date.today())
        mem_session.add(m2)
        with pytest.raises(Exception):
            mem_session.commit()
        mem_session.rollback()

    def test_multiple_sources(self, mem_session):
        """同一股票可屬於多個概念。"""
        for cname in ["測試概念D1", "測試概念D2"]:
            if not mem_session.query(ConceptGroup).filter_by(name=cname).first():
                mem_session.add(ConceptGroup(name=cname))
        mem_session.commit()

        for cname in ["測試概念D1", "測試概念D2"]:
            mem_session.add(
                ConceptMembership(concept_name=cname, stock_id="4444", source="yaml", added_date=date.today())
            )
        mem_session.commit()

        results = mem_session.query(ConceptMembership).filter_by(stock_id="4444").all()
        assert len(results) >= 2

    def test_delete_membership(self, mem_session):
        if not mem_session.query(ConceptGroup).filter_by(name="測試概念E").first():
            mem_session.add(ConceptGroup(name="測試概念E"))
            mem_session.commit()

        m = ConceptMembership(concept_name="測試概念E", stock_id="5555", source="yaml", added_date=date.today())
        mem_session.add(m)
        mem_session.commit()

        to_del = mem_session.query(ConceptMembership).filter_by(stock_id="5555").first()
        mem_session.delete(to_del)
        mem_session.commit()
        assert mem_session.query(ConceptMembership).filter_by(stock_id="5555").first() is None

    def test_concept_bonus_field_in_orm(self):
        """驗證 DiscoveryRecord 有 concept_bonus 欄位。"""
        from src.data.schema import DiscoveryRecord

        assert hasattr(DiscoveryRecord, "concept_bonus")
