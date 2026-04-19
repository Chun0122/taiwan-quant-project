"""Rotation 三道成本閘門測試（min_hold_days / score_gap / weekly_swap_cap）。

測試覆蓋：
  Gate A：min_hold_days 拉高 holding_days 安全下限
  Gate B：score_gap_threshold 阻擋 swap（現持分數與新最佳接近）
  Gate C：weekly_swap_cap 阻擋本週超額換手
  豁免：stop_loss/crisis_exit 不受閘門限制（安全優先）
  向後相容：所有閘門預設 0 時行為等同未加閘門
"""

from __future__ import annotations

from datetime import date

from src.portfolio.rotation import compute_rotation_actions

# 2025-01-06 (Mon) ~ 2025-01-24 (Fri) 共 15 個交易日
TRADING_CAL = [
    date(2025, 1, 6),
    date(2025, 1, 7),
    date(2025, 1, 8),
    date(2025, 1, 9),
    date(2025, 1, 10),
    date(2025, 1, 13),
    date(2025, 1, 14),
    date(2025, 1, 15),
    date(2025, 1, 16),
    date(2025, 1, 17),
    date(2025, 1, 20),
    date(2025, 1, 21),
    date(2025, 1, 22),
    date(2025, 1, 23),
    date(2025, 1, 24),
]


def _pos(stock_id: str, entry_date: date, entry_score: float | None = None) -> dict:
    d = {
        "stock_id": stock_id,
        "entry_date": entry_date,
        "entry_price": 100.0,
        "shares": 1000,
        "allocated_capital": 100_000.0,
        "entry_rank": 1,
    }
    if entry_score is not None:
        d["entry_score"] = entry_score
    return d


def _rank(stocks: list[tuple[str, float, float]]) -> list[dict]:
    """stocks = [(stock_id, close, score), ...]"""
    return [
        {
            "stock_id": sid,
            "stock_name": f"S{sid}",
            "rank": i + 1,
            "score": score,
            "close": close,
            "stop_loss": close * 0.9,
        }
        for i, (sid, close, score) in enumerate(stocks)
    ]


# =============================================================
# Gate A：min_hold_days
# =============================================================


def test_gate_a_floors_holding_days():
    """holding_days=1 + min_hold_days=3 → 實際持有 2 日時不到期。"""
    # entry=1/6, today=1/8 → days_held=2；min_hold_days=3 → 有效 holding_days=3
    pos = _pos("1111", date(2025, 1, 6))
    # 新排名不含 1111 → 正常情況下到期會賣出
    rankings = _rank([("2222", 50.0, 0.8)])
    actions = compute_rotation_actions(
        current_positions=[pos],
        new_rankings=rankings,
        max_positions=3,
        holding_days=1,  # 會被 min_hold_days=3 拉高
        allow_renewal=True,
        today=date(2025, 1, 8),
        trading_calendar=TRADING_CAL,
        current_cash=100_000.0,
        today_prices={"1111": 105.0, "2222": 50.0},
        min_hold_days=3,
    )
    # 應保持持倉，不到期
    assert not any(s["stock_id"] == "1111" for s in actions.to_sell)
    assert any(h["stock_id"] == "1111" for h in actions.to_hold)


def test_gate_a_does_not_lower_holding_days():
    """min_hold_days=2 + holding_days=5 → 以 5 為準，不下調。"""
    pos = _pos("1111", date(2025, 1, 6))
    # today=1/13 → days_held=5 → holding_days=5 到期
    rankings = _rank([("2222", 50.0, 0.8)])  # 1111 不在
    actions = compute_rotation_actions(
        current_positions=[pos],
        new_rankings=rankings,
        max_positions=3,
        holding_days=5,
        allow_renewal=True,
        today=date(2025, 1, 13),
        trading_calendar=TRADING_CAL,
        current_cash=100_000.0,
        today_prices={"1111": 105.0, "2222": 50.0},
        min_hold_days=2,  # 較小值，不影響
    )
    assert any(s["stock_id"] == "1111" and s["reason"] == "holding_expired" for s in actions.to_sell)


def test_gate_a_stop_loss_still_fires():
    """stop_loss 在 min_hold_days 內仍需觸發（安全優先）。"""
    pos = _pos("1111", date(2025, 1, 6))
    rankings = _rank([("2222", 50.0, 0.8)])
    actions = compute_rotation_actions(
        current_positions=[pos],
        new_rankings=rankings,
        max_positions=3,
        holding_days=1,
        allow_renewal=True,
        today=date(2025, 1, 7),  # 僅持有 1 日
        trading_calendar=TRADING_CAL,
        current_cash=100_000.0,
        stop_losses={"1111": 90.0},
        today_prices={"1111": 85.0, "2222": 50.0},  # 觸發 stop_loss
        min_hold_days=5,  # 即使設很大
    )
    assert any(s["stock_id"] == "1111" and s["reason"] == "stop_loss" for s in actions.to_sell)


# =============================================================
# Gate B：score_gap_threshold
# =============================================================


def test_gate_b_blocks_marginal_swap():
    """現持 score=0.60，新最佳 score=0.62 → gap=0.02 < 0.05 → 阻擋到期賣出。"""
    pos = _pos("1111", date(2025, 1, 6), entry_score=0.60)
    # 1111 不在 ranked_ids，會到期
    # 2222 是未持有的最佳，score=0.62，與 0.60 差距 0.02
    rankings = _rank([("2222", 50.0, 0.62), ("3333", 30.0, 0.55)])
    actions = compute_rotation_actions(
        current_positions=[pos],
        new_rankings=rankings,
        max_positions=3,
        holding_days=5,
        allow_renewal=True,
        today=date(2025, 1, 13),  # days_held=5 到期
        trading_calendar=TRADING_CAL,
        current_cash=100_000.0,
        today_prices={"1111": 105.0, "2222": 50.0, "3333": 30.0},
        score_gap_threshold=0.05,
    )
    # Gate B 阻擋 → 1111 仍在 hold（帶 gated_by）
    held = [h for h in actions.to_hold if h["stock_id"] == "1111"]
    assert len(held) == 1
    assert held[0]["gated_by"] == "gate_b_score_gap"
    assert not any(s["stock_id"] == "1111" for s in actions.to_sell)


def test_gate_b_allows_clear_swap():
    """現持 score=0.60，新最佳 score=0.72 → gap=0.12 > 0.05 → 允許賣出。"""
    pos = _pos("1111", date(2025, 1, 6), entry_score=0.60)
    rankings = _rank([("2222", 50.0, 0.72)])
    actions = compute_rotation_actions(
        current_positions=[pos],
        new_rankings=rankings,
        max_positions=3,
        holding_days=5,
        allow_renewal=True,
        today=date(2025, 1, 13),
        trading_calendar=TRADING_CAL,
        current_cash=100_000.0,
        today_prices={"1111": 105.0, "2222": 50.0},
        score_gap_threshold=0.05,
    )
    assert any(s["stock_id"] == "1111" and s["reason"] == "holding_expired" for s in actions.to_sell)


def test_gate_b_no_entry_score_skipped():
    """持倉無 entry_score 時，Gate B 不阻擋（graceful degradation）。"""
    pos = _pos("1111", date(2025, 1, 6))  # 無 entry_score
    rankings = _rank([("2222", 50.0, 0.62)])
    actions = compute_rotation_actions(
        current_positions=[pos],
        new_rankings=rankings,
        max_positions=3,
        holding_days=5,
        allow_renewal=True,
        today=date(2025, 1, 13),
        trading_calendar=TRADING_CAL,
        current_cash=100_000.0,
        today_prices={"1111": 105.0, "2222": 50.0},
        score_gap_threshold=0.05,
    )
    # 無 entry_score → Gate B 視為不阻擋 → 正常賣出
    assert any(s["stock_id"] == "1111" and s["reason"] == "holding_expired" for s in actions.to_sell)


# =============================================================
# Gate C：weekly_swap_cap
# =============================================================


def test_gate_c_blocks_when_budget_exhausted():
    """weekly_swap_cap=2、本週已用 2 → 下一筆 holding_expired 被阻擋。"""
    pos = _pos("1111", date(2025, 1, 6))
    rankings = _rank([("2222", 50.0, 0.8)])  # 1111 不在
    actions = compute_rotation_actions(
        current_positions=[pos],
        new_rankings=rankings,
        max_positions=3,
        holding_days=5,
        allow_renewal=True,
        today=date(2025, 1, 13),
        trading_calendar=TRADING_CAL,
        current_cash=100_000.0,
        today_prices={"1111": 105.0, "2222": 50.0},
        weekly_swap_cap=2,
        weekly_swaps_used=2,  # 本週已用盡
    )
    held = [h for h in actions.to_hold if h["stock_id"] == "1111"]
    assert len(held) == 1
    assert held[0]["gated_by"] == "gate_c_weekly_cap"


def test_gate_c_permits_within_budget():
    pos = _pos("1111", date(2025, 1, 6))
    rankings = _rank([("2222", 50.0, 0.8)])
    actions = compute_rotation_actions(
        current_positions=[pos],
        new_rankings=rankings,
        max_positions=3,
        holding_days=5,
        allow_renewal=True,
        today=date(2025, 1, 13),
        trading_calendar=TRADING_CAL,
        current_cash=100_000.0,
        today_prices={"1111": 105.0, "2222": 50.0},
        weekly_swap_cap=3,
        weekly_swaps_used=1,  # 還剩 2 個預算
    )
    assert any(s["stock_id"] == "1111" and s["reason"] == "holding_expired" for s in actions.to_sell)
    assert actions.holding_expired_sells == 1


def test_gate_c_returns_expired_counter():
    """RotationActions.holding_expired_sells 計數正確（供 backtest 累積週預算）。"""
    positions = [_pos("1111", date(2025, 1, 6)), _pos("2222", date(2025, 1, 6))]
    rankings = _rank([("3333", 30.0, 0.8)])  # 兩檔皆到期且皆未入榜
    actions = compute_rotation_actions(
        current_positions=positions,
        new_rankings=rankings,
        max_positions=3,
        holding_days=5,
        allow_renewal=True,
        today=date(2025, 1, 13),
        trading_calendar=TRADING_CAL,
        current_cash=100_000.0,
        today_prices={"1111": 105.0, "2222": 105.0, "3333": 30.0},
    )
    assert actions.holding_expired_sells == 2


# =============================================================
# 豁免：stop_loss 與 crisis_exit 不計入、不受閘門限制
# =============================================================


def test_stop_loss_not_counted_in_weekly_cap():
    """stop_loss 不消耗 weekly_swap_cap 預算。"""
    pos = _pos("1111", date(2025, 1, 6))
    rankings = _rank([("2222", 50.0, 0.8)])
    actions = compute_rotation_actions(
        current_positions=[pos],
        new_rankings=rankings,
        max_positions=3,
        holding_days=5,
        allow_renewal=True,
        today=date(2025, 1, 7),
        trading_calendar=TRADING_CAL,
        current_cash=100_000.0,
        stop_losses={"1111": 90.0},
        today_prices={"1111": 85.0, "2222": 50.0},  # 觸 stop_loss
        weekly_swap_cap=1,
        weekly_swaps_used=0,
    )
    # stop_loss 應照常觸發
    sl_sells = [s for s in actions.to_sell if s["reason"] == "stop_loss"]
    assert len(sl_sells) == 1
    # holding_expired_sells 為 0
    assert actions.holding_expired_sells == 0


# =============================================================
# 向後相容：所有閘門預設關閉時行為不變
# =============================================================


def test_backward_compat_defaults_off():
    """所有閘門預設 0 時行為與舊版一致（無閘門阻擋）。"""
    pos = _pos("1111", date(2025, 1, 6), entry_score=0.60)
    rankings = _rank([("2222", 50.0, 0.62)])  # gap 微小但無閘門
    actions = compute_rotation_actions(
        current_positions=[pos],
        new_rankings=rankings,
        max_positions=3,
        holding_days=5,
        allow_renewal=True,
        today=date(2025, 1, 13),
        trading_calendar=TRADING_CAL,
        current_cash=100_000.0,
        today_prices={"1111": 105.0, "2222": 50.0},
        # 所有閘門參數皆預設 0 / 0.0
    )
    assert any(s["stock_id"] == "1111" and s["reason"] == "holding_expired" for s in actions.to_sell)
