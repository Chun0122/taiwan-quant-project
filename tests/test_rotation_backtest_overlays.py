"""Rotation 回測風控 overlay 對齊 live（P0-1 修復）測試。

背景：RotationManager.backtest() 過去呼叫 compute_rotation_actions 時未傳
total_capital / corr_matrix / vol_weights / regime，導致 Portfolio Heat、
Correlation Budget、波動率反比權重、Crisis 阻擋在回測中全部失效 —— 回測等於
「等權 top-N」而非實際部署的策略。本測試以「歷史 regime=crisis 應阻擋回測新開倉」
作為最直接的回歸守門：舊碼會照買（regime 未傳→不阻擋），新碼應為 0 筆交易。
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.data.schema import DailyPrice, DiscoveryRecord, RotationPortfolio
from src.portfolio.manager import RotationManager

_START = date(2025, 3, 3)  # 週一
_DAYS = 14
_STOCKS = ("2330", "2317", "2454", "3008", "6669")


@pytest.fixture()
def patch_session(monkeypatch):
    """專屬 in-memory engine/session（不共用 conftest db_session）。

    backtest() 內部 save_rotation_backtest() 會 commit 共享 session，會撕掉 conftest
    的 transaction-rollback 隔離造成跨測試污染（見記憶 feedback_test_db_fixture_gotchas）。
    這裡每個測試建立獨立 engine，徹底隔離。
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.data.database import Base

    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    class _Ctx:
        def __enter__(self):
            return session

        def __exit__(self, *a):
            return False

    from src.data import pipeline as pipeline_module
    from src.portfolio import manager as mgr_module

    monkeypatch.setattr(mgr_module, "get_session", lambda: _Ctx())
    monkeypatch.setattr(pipeline_module, "get_session", lambda: _Ctx())
    yield session
    session.close()


def _seed(db_session, *, regime: str) -> RotationPortfolio:
    """TAIEX 行事曆 + 5 檔個股每日 K 線 + 第一日 5 筆 momentum DiscoveryRecord(指定 regime)。"""
    for i in range(_DAYS):
        d = _START + timedelta(days=i)
        db_session.add(
            DailyPrice(stock_id="TAIEX", date=d, open=23000, high=23100, low=22900, close=23050, volume=0, turnover=0.0)
        )
        for sid in _STOCKS:
            db_session.add(
                DailyPrice(
                    stock_id=sid,
                    date=d,
                    open=100,
                    high=102,
                    low=99,
                    close=100 + i * 0.1,  # 緩漲，避免觸發止損
                    volume=10_000_000,
                    turnover=1_000_000_000.0,
                )
            )
    for rank, sid in enumerate(_STOCKS, start=1):
        db_session.add(
            DiscoveryRecord(
                scan_date=_START,
                mode="momentum",
                rank=rank,
                stock_id=sid,
                stock_name=f"name_{sid}",
                close=100,
                composite_score=0.9 - rank * 0.05,
                technical_score=0.8,
                chip_score=0.7,
                fundamental_score=0.5,
                news_score=0.4,
                regime=regime,
                entry_price=100,
                stop_loss=90,
            )
        )
    p = RotationPortfolio(
        name="overlay_test",
        mode="momentum",
        max_positions=3,
        holding_days=10,
        allow_renewal=True,
        initial_capital=1_000_000.0,
        current_capital=1_000_000.0,
        current_cash=1_000_000.0,
        status="active",
    )
    db_session.add(p)
    db_session.commit()
    return p


def test_backtest_passes_historical_regime_crisis_blocks_buys(patch_session):
    """歷史 regime=crisis → 回測不應有任何進場（P0-1：regime 已接線）。"""
    _seed(patch_session, regime="crisis")
    result = RotationManager("overlay_test").backtest(_START, _START + timedelta(days=_DAYS - 1))
    assert result.trades == [], "crisis regime 下回測仍開倉 → regime 未正確傳入 compute_rotation_actions"


def test_backtest_non_crisis_regime_allows_buys(patch_session):
    """對照組：regime=bull → 回測正常進場（確認阻擋來自 regime 而非其他因素）。"""
    _seed(patch_session, regime="bull")
    result = RotationManager("overlay_test").backtest(_START, _START + timedelta(days=_DAYS - 1))
    assert len(result.trades) >= 1, "bull regime 下回測未開倉 → overlay 接線把正常交易也擋掉了"


def _seed_t1(db_session) -> RotationPortfolio:
    """T+1 測試：掃描日 close=100，隔日 open=110（明顯不同），用以區分 T 收盤 vs T+1 開盤進場。"""
    for i in range(_DAYS):
        d = _START + timedelta(days=i)
        db_session.add(
            DailyPrice(stock_id="TAIEX", date=d, open=23000, high=23100, low=22900, close=23050, volume=0, turnover=0.0)
        )
        for sid in _STOCKS:
            if i == 0:
                o, c = 100.0, 100.0  # 掃描日 D0：收盤 100（= DiscoveryRecord.close）
            elif i == 1:
                o, c = 110.0, 111.0  # D1 開盤 110：T+1 進場應成交在此，而非 D0 收盤 100
            else:
                o, c = 111.0, 111.0
            db_session.add(
                DailyPrice(
                    stock_id=sid, date=d, open=o, high=c + 2, low=o - 2, close=c, volume=10_000_000, turnover=1e9
                )
            )
    db_session.add(
        DiscoveryRecord(
            scan_date=_START,
            mode="momentum",
            rank=1,
            stock_id="2330",
            stock_name="台積電",
            close=100,
            composite_score=0.9,
            technical_score=0.8,
            chip_score=0.7,
            fundamental_score=0.5,
            news_score=0.4,
            regime="bull",
            entry_price=100,
            stop_loss=80,
        )
    )
    p = RotationPortfolio(
        name="t1_test",
        mode="momentum",
        max_positions=1,
        holding_days=5,
        allow_renewal=False,
        initial_capital=1_000_000.0,
        current_capital=1_000_000.0,
        current_cash=1_000_000.0,
        status="active",
    )
    db_session.add(p)
    db_session.commit()
    return p


def test_backtest_entry_fills_at_next_day_open_not_scan_close(patch_session):
    """T+1（P0-2）：掃描日 D0 決策 → 隔日 D1 開盤成交。

    entry_price 應 = open[D1]=110、entry_date = D1，而非 close[D0]=100、entry_date=D0。
    舊碼（同日成交 close[D0]）會 entry_price=100、entry_date=D0 → 本測試失敗。
    """
    _seed_t1(patch_session)
    result = RotationManager("t1_test").backtest(_START, _START + timedelta(days=_DAYS - 1))
    assert result.trades, "未產生任何交易"
    entry = result.trades[0]
    assert entry["entry_price"] == 110.0, f"進場價應為 D1 開盤 110（T+1），實得 {entry['entry_price']}"
    assert entry["entry_date"] == _START + timedelta(days=1), f"進場日應為 D1，實得 {entry['entry_date']}"


# ====================================================================== #
# P1-3 survivorship：期末持有但個股下市 → 以最後已知價平倉、計數、警告
# ====================================================================== #


def _seed_delisting(db_session) -> RotationPortfolio:
    """個股 9999 僅 D0~D3 有報價（D3 收 80 後下市），D0 推薦、買進後持有到期末。"""
    for i in range(_DAYS):
        d = _START + timedelta(days=i)
        db_session.add(
            DailyPrice(stock_id="TAIEX", date=d, open=23000, high=23100, low=22900, close=23050, volume=0, turnover=0.0)
        )
    # 9999 只有前 4 天報價：100 → 98 → 90 → 80（之後下市，無資料）
    closes = {0: 100.0, 1: 98.0, 2: 90.0, 3: 80.0}
    for i, c in closes.items():
        o = 100.0 if i <= 1 else c + 1
        db_session.add(
            DailyPrice(
                stock_id="9999",
                date=_START + timedelta(days=i),
                open=o,
                high=c + 2,
                low=c - 2,
                close=c,
                volume=10_000_000,
                turnover=1e9,
            )
        )
    db_session.add(
        DiscoveryRecord(
            scan_date=_START,
            mode="momentum",
            rank=1,
            stock_id="9999",
            stock_name="下市股",
            close=100,
            composite_score=0.9,
            technical_score=0.8,
            chip_score=0.7,
            fundamental_score=0.5,
            news_score=0.4,
            regime="bull",
            entry_price=100,
            stop_loss=None,
        )
    )
    p = RotationPortfolio(
        name="delist_test",
        mode="momentum",
        max_positions=1,
        holding_days=30,  # 30 > 視窗 → 持有到期末
        allow_renewal=True,
        initial_capital=1_000_000.0,
        current_capital=1_000_000.0,
        current_cash=1_000_000.0,
        status="active",
    )
    db_session.add(p)
    db_session.commit()
    return p


def test_backtest_delisted_holding_realizes_loss_and_warns(patch_session):
    """下市股持有到期末 → 以最後已知價(80)平倉實現虧損、計入 stranded、發警告。

    舊行為：today_prices 缺值 fallback entry_price(100) → 凍結無虧損（樂觀偏差）。
    """
    _seed_delisting(patch_session)
    r = RotationManager("delist_test").backtest(_START, _START + timedelta(days=_DAYS - 1))
    assert r.metrics["survivorship_warning"] is True
    assert r.metrics["survivorship_stranded"] >= 1
    t = next((x for x in r.trades if x["stock_id"] == "9999"), None)
    assert t is not None, "9999 未被買進/平倉"
    assert t["exit_reason"] == "delisted_stranded"
    # 以最後已知價 80 平倉（非凍結在 entry 100）→ 實現虧損
    assert t["exit_price"] == pytest.approx(80.0)
    assert t["pnl"] < 0


# ── 停損實驗旋鈕（disable_stop_loss / stop_loss_widen，B1 診斷後）──


def _seed_stop(db_session) -> RotationPortfolio:
    """bull regime，價格在持有期內跌破 stop_loss=90（觸發停損）。

    價路：D0~D1=100、D2=95、D3 起=86（< 90 觸發；> 80 不觸發放寬後停損）。
    """

    def level(i: int) -> float:
        if i <= 1:
            return 100.0
        if i == 2:
            return 95.0
        return 86.0

    for i in range(_DAYS):
        d = _START + timedelta(days=i)
        db_session.add(
            DailyPrice(stock_id="TAIEX", date=d, open=23000, high=23100, low=22900, close=23050, volume=0, turnover=0.0)
        )
        px = level(i)
        for sid in _STOCKS:
            db_session.add(
                DailyPrice(
                    stock_id=sid,
                    date=d,
                    open=px,
                    high=px + 1,
                    low=px - 1,
                    close=px,
                    volume=10_000_000,
                    turnover=1_000_000_000.0,
                )
            )
    for rank, sid in enumerate(_STOCKS, start=1):
        db_session.add(
            DiscoveryRecord(
                scan_date=_START,
                mode="momentum",
                rank=rank,
                stock_id=sid,
                stock_name=f"name_{sid}",
                close=100,
                composite_score=0.9 - rank * 0.05,
                regime="bull",
                entry_price=100,
                stop_loss=90,
            )
        )
    p = RotationPortfolio(
        name="stop_test",
        mode="momentum",
        max_positions=3,
        holding_days=10,
        allow_renewal=True,
        initial_capital=1_000_000.0,
        current_capital=1_000_000.0,
        current_cash=1_000_000.0,
        status="active",
    )
    db_session.add(p)
    db_session.commit()
    return p


def _stop_exits(result) -> int:
    return sum(1 for t in result.trades if t.get("exit_reason") == "stop_loss")


def test_backtest_baseline_triggers_stop_loss(patch_session):
    """baseline（預設旋鈕）：價格跌破 90 → 至少一筆 stop_loss 出場。"""
    _seed_stop(patch_session)
    result = RotationManager("stop_test").backtest(_START, _START + timedelta(days=_DAYS - 1))
    assert _stop_exits(result) >= 1


def test_backtest_disable_stop_loss_removes_stop_exits(patch_session):
    """disable_stop_loss=True → 無任何 stop_loss 出場。"""
    _seed_stop(patch_session)
    result = RotationManager("stop_test").backtest(_START, _START + timedelta(days=_DAYS - 1), disable_stop_loss=True)
    assert _stop_exits(result) == 0


def test_backtest_widen_stop_reduces_stop_exits(patch_session):
    """stop_loss_widen=2.0（停損移到 80）→ 86 不觸發，停損出場數 < baseline。"""
    _seed_stop(patch_session)
    base = RotationManager("stop_test").backtest(_START, _START + timedelta(days=_DAYS - 1))
    widened = RotationManager("stop_test").backtest(_START, _START + timedelta(days=_DAYS - 1), stop_loss_widen=2.0)
    assert _stop_exits(widened) < _stop_exits(base)
