"""MOPS 公告整合測試。

測試項目：
- classify_sentiment: 關鍵字情緒分類（正面/負面/中性）
- _compute_news_scores: 消息面評分計算
- Announcement ORM: 資料寫入 + 唯一鍵衝突
- Regime 權重矩陣: 加入 news 後總和 = 1.0
"""

from datetime import date

import pandas as pd
import pytest

from src.data.mops_fetcher import classify_sentiment
from src.regime.detector import REGIME_WEIGHTS, MarketRegimeDetector

# ------------------------------------------------------------------ #
#  classify_sentiment 情緒分類
# ------------------------------------------------------------------ #


class TestClassifySentiment:
    """關鍵字情緒分類測試。"""

    @pytest.mark.parametrize(
        "subject",
        [
            "本公司董事會決議執行庫藏股買回",
            "本公司合併營收創歷史新高",
            "本公司與XX公司簽訂合併契約",
            "本公司取得專利權",
            "本公司董事會決議現金增資發行新股",
        ],
    )
    def test_positive_keywords(self, subject):
        """包含正面關鍵字 → +1。"""
        assert classify_sentiment(subject) == 1

    @pytest.mark.parametrize(
        "subject",
        [
            "本公司股票終止上市",
            "本公司財務報表重編",
            "本公司發生重大虧損",
            "本公司違約交割",
            "本公司受金管會裁罰",
            "本公司董事會決議減資彌補虧損",
        ],
    )
    def test_negative_keywords(self, subject):
        """包含負面關鍵字 → -1。"""
        assert classify_sentiment(subject) == -1

    @pytest.mark.parametrize(
        "subject",
        [
            "本公司董事會決議召開股東常會",
            "本公司變更公司章程",
            "代子公司公告董事異動",
            "本公司發言人變更",
        ],
    )
    def test_neutral_keywords(self, subject):
        """一般公告 → 0。"""
        assert classify_sentiment(subject) == 0

    def test_empty_subject(self):
        """空字串 → 0。"""
        assert classify_sentiment("") == 0

    def test_none_subject(self):
        """None → 0。"""
        assert classify_sentiment(None) == 0

    def test_negative_takes_priority(self):
        """同時包含正負面關鍵字時，負面優先。"""
        subject = "本公司因虧損決議減資彌補後辦理現金增資"
        assert classify_sentiment(subject) == -1


# ------------------------------------------------------------------ #
#  _compute_news_scores 消息面評分
# ------------------------------------------------------------------ #


class TestComputeNewsScores:
    """消息面評分計算測試。"""

    def _get_scanner(self):
        """建立基底 MarketScanner 實例（不做 DB 操作）。"""
        from src.discovery.scanner import MarketScanner

        return MarketScanner()

    def test_empty_announcements_returns_neutral(self):
        """無公告資料 → 全部 0.5。"""
        scanner = self._get_scanner()
        stock_ids = ["2330", "2317", "2454"]
        result = scanner._compute_news_scores(stock_ids, pd.DataFrame())
        assert len(result) == 3
        assert all(result["news_score"] == 0.5)

    def test_stocks_not_in_announcements_get_neutral(self):
        """公告中沒有的股票 → 0.5。"""
        scanner = self._get_scanner()
        df_ann = pd.DataFrame(
            {
                "stock_id": ["9999"],
                "date": [date(2025, 6, 1)],
                "seq": ["1"],
                "subject": ["測試公告"],
                "sentiment": [0],
            }
        )
        result = scanner._compute_news_scores(["2330", "2317"], df_ann)
        assert len(result) == 2
        assert all(result["news_score"] == 0.5)

    def test_positive_announcements_boost_score(self):
        """有正面公告的股票分數應高於無公告的股票。"""
        scanner = self._get_scanner()

        df_ann = pd.DataFrame(
            {
                "stock_id": ["2330", "2330", "2330"],
                "date": [date(2025, 6, 1)] * 3,
                "seq": ["1", "2", "3"],
                "subject": ["庫藏股買回", "營收創新高", "合併案通過"],
                "sentiment": [1, 1, 1],
            }
        )
        result = scanner._compute_news_scores(["2330", "2317"], df_ann)

        score_2330 = result[result["stock_id"] == "2330"]["news_score"].values[0]
        score_2317 = result[result["stock_id"] == "2317"]["news_score"].values[0]
        assert score_2330 > score_2317

    def test_negative_announcements_reduce_score(self):
        """有負面公告的股票分數應低於有正面公告的。"""
        scanner = self._get_scanner()

        df_ann = pd.DataFrame(
            {
                "stock_id": ["2330", "2317"],
                "date": [date(2025, 6, 1)] * 2,
                "seq": ["1", "1"],
                "subject": ["庫藏股買回", "公司終止上市"],
                "sentiment": [1, -1],
            }
        )
        result = scanner._compute_news_scores(["2330", "2317"], df_ann)

        score_2330 = result[result["stock_id"] == "2330"]["news_score"].values[0]
        score_2317 = result[result["stock_id"] == "2317"]["news_score"].values[0]
        assert score_2330 > score_2317

    def test_output_columns(self):
        """輸出 DataFrame 應只包含 stock_id 和 news_score。"""
        scanner = self._get_scanner()
        result = scanner._compute_news_scores(["2330"], pd.DataFrame())
        assert list(result.columns) == ["stock_id", "news_score"]


# ------------------------------------------------------------------ #
#  Announcement ORM
# ------------------------------------------------------------------ #


class TestAnnouncementORM:
    """Announcement ORM 測試。"""

    def test_upsert_announcement(self, db_session):
        """寫入公告並驗證唯一鍵衝突處理。"""
        from src.data.schema import Announcement

        ann = Announcement(
            stock_id="2330",
            date=date(2025, 6, 1),
            seq="1",
            subject="本公司董事會決議庫藏股買回",
            spoke_time="15:30",
            sentiment=1,
        )
        db_session.add(ann)
        db_session.flush()

        result = db_session.query(Announcement).filter_by(stock_id="2330").all()
        assert len(result) == 1
        assert result[0].subject == "本公司董事會決議庫藏股買回"
        assert result[0].sentiment == 1

    def test_multiple_announcements_same_stock(self, db_session):
        """同一股票同一天多則公告。"""
        from src.data.schema import Announcement

        for i in range(3):
            ann = Announcement(
                stock_id="2330",
                date=date(2025, 6, 1),
                seq=str(i + 1),
                subject=f"公告 {i + 1}",
                sentiment=0,
            )
            db_session.add(ann)
        db_session.flush()

        result = db_session.query(Announcement).filter_by(stock_id="2330").all()
        assert len(result) == 3


# ------------------------------------------------------------------ #
#  Regime 權重矩陣
# ------------------------------------------------------------------ #


class TestRegimeWeightsWithNews:
    """確認權重矩陣加入 news 後仍正確。"""

    @pytest.mark.parametrize("mode", ["momentum", "swing", "value"])
    def test_all_weights_sum_to_one(self, mode):
        """所有模式各 regime 權重加總 = 1.0。"""
        for regime in ("bull", "sideways", "bear"):
            w = REGIME_WEIGHTS[mode][regime]
            assert sum(w.values()) == pytest.approx(1.0), f"{mode}/{regime}: {w} 總和={sum(w.values())}"

    @pytest.mark.parametrize("mode", ["momentum", "swing", "value"])
    def test_news_key_exists(self, mode):
        """所有模式各 regime 都包含 news 鍵。"""
        for regime in ("bull", "sideways", "bear"):
            w = REGIME_WEIGHTS[mode][regime]
            assert "news" in w, f"{mode}/{regime} 缺少 news 權重"

    def test_default_weights_include_news(self):
        """預設 fallback 權重也包含 news。"""
        w = MarketRegimeDetector.get_weights("unknown_mode", "bull")
        assert "news" in w
        assert sum(w.values()) == pytest.approx(1.0)

    def test_bear_news_weight_higher_than_bull(self):
        """空頭時消息面權重應不低於多頭（重訊更關鍵）。"""
        for mode in ("momentum", "swing", "value"):
            bull_news = REGIME_WEIGHTS[mode]["bull"]["news"]
            bear_news = REGIME_WEIGHTS[mode]["bear"]["news"]
            assert bear_news >= bull_news, f"{mode}: bear news={bear_news} < bull news={bull_news}"
