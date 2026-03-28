"""tests/test_anomaly.py — P5 籌碼異動警報純函數測試。

測試四個純函數：
  detect_volume_spike           量能暴增
  detect_institutional_buy      外資大買超
  detect_sbl_spike              借券賣出激增
  detect_broker_concentration   主力分點集中買進
"""

import datetime

import pandas as pd
import pytest

from src.cli.detection import (
    detect_broker_concentration,
    detect_institutional_buy,
    detect_sbl_spike,
    detect_volume_spike,
)

# ────────────────────────────────────────────────────────────
#  輔助函數：快速建構 DataFrame
# ────────────────────────────────────────────────────────────

_BASE_DATE = datetime.date(2026, 3, 7)


def _make_price_df(stock_id: str, volumes: list[int]) -> pd.DataFrame:
    """建構日K量能 DataFrame，最後一筆為「今日」。"""
    dates = [_BASE_DATE - datetime.timedelta(days=len(volumes) - i - 1) for i in range(len(volumes))]
    return pd.DataFrame({"stock_id": stock_id, "date": dates, "volume": volumes})


def _make_inst_df(stock_id: str, net: int, name: str = "外資買賣超") -> pd.DataFrame:
    return pd.DataFrame({"stock_id": [stock_id], "date": [_BASE_DATE], "name": [name], "net": [net]})


def _make_sbl_df(stock_id: str, changes: list[float]) -> pd.DataFrame:
    """最後一筆為今日 sbl_change。"""
    dates = [_BASE_DATE - datetime.timedelta(days=len(changes) - i - 1) for i in range(len(changes))]
    return pd.DataFrame({"stock_id": stock_id, "date": dates, "sbl_change": changes})


def _make_broker_df(stock_id: str, brokers: list[dict]) -> pd.DataFrame:
    """brokers: list of {broker_id, buy, sell}，均為今日。"""
    rows = [{"stock_id": stock_id, "date": _BASE_DATE, **b} for b in brokers]
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
#  detect_volume_spike
# ────────────────────────────────────────────────────────────


def test_detect_volume_spike_triggered():
    """今日量 = 5,000 > 均量 1,000 × 2.0 → 應觸發。"""
    # 10 天歷史量 + 1 天今日
    vols = [1000] * 10 + [5000]
    df = _make_price_df("2330", vols)
    result = detect_volume_spike(df, lookback=10, threshold=2.0)
    assert len(result) == 1
    assert result.iloc[0]["stock_id"] == "2330"
    assert result.iloc[0]["vol_ratio"] == pytest.approx(5.0)


def test_detect_volume_spike_not_triggered():
    """今日量 = 1,500 < 均量 1,000 × 2.0 → 不觸發。"""
    vols = [1000] * 10 + [1500]
    df = _make_price_df("2317", vols)
    result = detect_volume_spike(df, lookback=10, threshold=2.0)
    assert result.empty


def test_detect_volume_spike_insufficient_data():
    """只有今日，無歷史資料 → 回傳空 DataFrame。"""
    df = _make_price_df("2454", [5000])  # 只有 1 筆
    result = detect_volume_spike(df, lookback=10, threshold=2.0)
    assert result.empty


# ────────────────────────────────────────────────────────────
#  detect_institutional_buy
# ────────────────────────────────────────────────────────────


def test_detect_institutional_buy_above():
    """外資 net = 5,000,000 > threshold 3,000,000 → 應觸發。"""
    df = _make_inst_df("2330", net=5_000_000)
    result = detect_institutional_buy(df, threshold=3_000_000)
    assert len(result) == 1
    assert result.iloc[0]["stock_id"] == "2330"
    assert result.iloc[0]["inst_net"] == 5_000_000


def test_detect_institutional_buy_below():
    """外資 net = 1,000,000 < threshold 3,000,000 → 不觸發。"""
    df = _make_inst_df("2317", net=1_000_000)
    result = detect_institutional_buy(df, threshold=3_000_000)
    assert result.empty


def test_detect_institutional_buy_foreign_name_filter():
    """非外資（投信）不應被計入。"""
    df_mixed = pd.concat(
        [
            _make_inst_df("2330", net=5_000_000, name="外資買賣超"),
            _make_inst_df("2330", net=8_000_000, name="投信買賣超"),
        ]
    ).reset_index(drop=True)
    result = detect_institutional_buy(df_mixed, threshold=3_000_000)
    assert len(result) == 1
    assert result.iloc[0]["inst_net"] == 5_000_000  # 只計外資


# ────────────────────────────────────────────────────────────
#  detect_sbl_spike
# ────────────────────────────────────────────────────────────


def test_detect_sbl_spike_triggered():
    """今日 sbl_change = 5000，歷史均值 100，std 20，z = (5000-100)/20 = 245 >> 2σ → 觸發。"""
    hist = [100.0, 110.0, 90.0, 95.0, 105.0, 100.0, 98.0, 102.0, 97.0, 103.0]
    today_val = 5000.0
    changes = hist + [today_val]
    df = _make_sbl_df("2454", changes)
    result = detect_sbl_spike(df, lookback=10, sigma=2.0)
    assert len(result) == 1
    assert result.iloc[0]["stock_id"] == "2454"
    assert result.iloc[0]["sbl_change"] == 5000


def test_detect_sbl_spike_not_triggered():
    """今日 sbl_change 正常，不超過均值 + 2σ → 不觸發。"""
    changes = [100.0, 110.0, 90.0, 95.0, 105.0, 100.0, 98.0, 102.0, 97.0, 103.0, 108.0]
    df = _make_sbl_df("2330", changes)
    result = detect_sbl_spike(df, lookback=10, sigma=2.0)
    assert result.empty


def test_detect_sbl_spike_insufficient_data():
    """少於 3 筆 → 回傳空 DataFrame。"""
    df = _make_sbl_df("2317", [500.0, 600.0])  # 只有 2 筆
    result = detect_sbl_spike(df, lookback=10, sigma=2.0)
    assert result.empty


# ────────────────────────────────────────────────────────────
#  detect_broker_concentration
# ────────────────────────────────────────────────────────────


def test_detect_broker_concentration_triggered():
    """單一分點買 1000 股，其他分點買 100，HHI = (1000/1100)² + (100/1100)² ≈ 0.830 > 0.4 → 觸發。"""
    brokers = [
        {"broker_id": "A001", "buy": 1000, "sell": 0},
        {"broker_id": "B002", "buy": 100, "sell": 0},
    ]
    df = _make_broker_df("2382", brokers)
    result = detect_broker_concentration(df, hhi_threshold=0.4)
    assert len(result) == 1
    assert result.iloc[0]["stock_id"] == "2382"
    assert result.iloc[0]["broker_hhi"] > 0.4


def test_detect_broker_concentration_net_sell():
    """所有分點淨賣出（net_buy_total ≤ 0）→ 不觸發。"""
    brokers = [
        {"broker_id": "A001", "buy": 100, "sell": 1000},
        {"broker_id": "B002", "buy": 50, "sell": 500},
    ]
    df = _make_broker_df("2330", brokers)
    result = detect_broker_concentration(df, hhi_threshold=0.4)
    assert result.empty


def test_detect_broker_concentration_dispersed():
    """5 個分點各買 200 股（均勻分布）→ HHI = 0.2 < 0.4 → 不觸發。"""
    brokers = [{"broker_id": f"X{i:03d}", "buy": 200, "sell": 0} for i in range(5)]
    df = _make_broker_df("2317", brokers)
    result = detect_broker_concentration(df, hhi_threshold=0.4)
    assert result.empty


# ────────────────────────────────────────────────────────────
#  TestDetectDaytradeRisk — 隔日沖風險偵測
# ────────────────────────────────────────────────────────────


class TestDetectDaytradeRisk:
    """detect_daytrade_risk() 純函數測試。"""

    def _make_dt_broker_df(self, stock_id: str, brokers: list[dict]) -> pd.DataFrame:
        rows = [
            {
                "stock_id": stock_id,
                "date": _BASE_DATE,
                "broker_id": b.get("broker_id", f"B{i:03d}"),
                "broker_name": b.get("broker_name", f"分點{i}"),
                "buy": b.get("buy", 0),
                "sell": b.get("sell", 0),
            }
            for i, b in enumerate(brokers)
        ]
        return pd.DataFrame(rows)

    def test_risk_triggered(self):
        """penalty > threshold → 觸發。"""
        from src.cli.detection import detect_daytrade_risk

        df = self._make_dt_broker_df(
            "2330",
            [
                {"broker_name": "凱基-台北", "buy": 80000, "sell": 0},
                {"broker_name": "一般分點", "buy": 20000, "sell": 0},
            ],
        )
        result = detect_daytrade_risk(df, penalty_threshold=0.3)
        assert len(result) == 1
        assert result.iloc[0]["daytrade_penalty"] >= 0.3

    def test_below_threshold(self):
        """penalty < threshold → 不觸發。"""
        from src.cli.detection import detect_daytrade_risk

        df = self._make_dt_broker_df(
            "2330",
            [
                {"broker_name": "一般分點A", "buy": 90000, "sell": 0},
                {"broker_name": "一般分點B", "buy": 10000, "sell": 0},
            ],
        )
        result = detect_daytrade_risk(df, penalty_threshold=0.3)
        assert result.empty

    def test_top_dt_brokers_returned(self):
        """正確回傳隔日沖分點名稱。"""
        from src.cli.detection import detect_daytrade_risk

        df = self._make_dt_broker_df(
            "2330",
            [
                {"broker_name": "凱基-台北", "buy": 60000, "sell": 0},
                {"broker_name": "美林", "buy": 30000, "sell": 0},
                {"broker_name": "一般分點", "buy": 10000, "sell": 0},
            ],
        )
        result = detect_daytrade_risk(df, penalty_threshold=0.1)
        assert len(result) == 1
        tags = result.iloc[0]["top_dt_brokers"]
        assert "凱基-台北" in tags

    def test_empty_data(self):
        """空資料 → 不觸發。"""
        from src.cli.detection import detect_daytrade_risk

        df = pd.DataFrame()
        result = detect_daytrade_risk(df, penalty_threshold=0.3)
        assert result.empty
