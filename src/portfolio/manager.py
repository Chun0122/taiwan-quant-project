"""RotationManager — 輪動組合的 DB 調度與回測引擎。

負責讀寫 RotationPortfolio / RotationPosition ORM，
呼叫 rotation.py 純函數計算買賣動作，並支援歷史回測。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
from sqlalchemy import delete, select

from src.config import settings
from src.constants import (
    ACTION_TYPE_PENDING_BUY,
    ACTION_TYPE_PENDING_SELL,
    COMMISSION_RATE,
    COMPOSITE_MODES,
    DECIDE_STAGE_ACTION_TYPES,
    LIQUIDITY_PARTICIPATION_LIMIT,
    MAX_DRAWDOWN_LIQUIDATE_PCT,
    PENDING_BUY_TTL_TRADING_DAYS,
    PENDING_STATUS_CANCELLED,
    PENDING_STATUS_FILLED,
    PENDING_STATUS_PENDING,
    REGIME_FALLBACK_DEFAULT,
    SLIPPAGE_RATE,
    TAX_RATE,
    is_composite_mode,
)
from src.data.database import get_session
from src.data.schema import (
    DailyPrice,
    DiscoveryRecord,
    RotationActionLog,
    RotationDailySnapshot,
    RotationPendingOrder,
    RotationPortfolio,
    RotationPosition,
)
from src.portfolio.execution_core import simulate_buy, simulate_sell

# P2 任務 14 phase 1：以下函式已抽出至主題模組（rankings/market_data/metrics）。
# 此處 re-import 維持 `from src.portfolio.manager import X` 的向後相容
# （既有 tests / CLI 無需修改 import）。標 noqa 者僅為 re-export，manager 內未直接使用。
from src.portfolio.market_data import (
    SNAPSHOT_BENCHMARK_STOCK_ID,  # noqa: F401  re-export
    _get_benchmark_close_on_or_before,
    _get_ohlcv_exact_date,
    _get_ohlcv_on_date,
    _get_prices_on_date,
    _get_taiex_prices,
    _get_trading_calendar,
)
from src.portfolio.metrics import compute_benchmark_alpha_fields, compute_cost_metrics
from src.portfolio.rankings import (
    _record_to_score_breakdown,
    _resolve_all_mode_rankings,  # noqa: F401  re-export
    resolve_rankings,
)
from src.portfolio.rotation import (
    RotationActions,
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_drawdown_with_snapshots,
    compute_dynamic_slippage,
    compute_planned_exit_date,
    compute_portfolio_drawdown,
    compute_portfolio_var,
    compute_position_pnl,
    compute_rotation_actions,
    compute_shares,
    compute_vol_inverse_weights,
    detect_limit_price,
)

logger = logging.getLogger(__name__)

MODE_LABELS = {
    "momentum": "動能",
    "swing": "波段",
    "value": "價值",
    "dividend": "高息",
    "growth": "成長",
    "all": "綜合",
    "mom_growth": "動量+成長雙引擎",
}


@dataclass
class RotationBacktestResult:
    """回測結果。"""

    equity_curve: list[dict] = field(default_factory=list)  # [{date, equity}]
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    daily_positions: list[dict] = field(default_factory=list)  # 每日持倉快照


@dataclass
class _TradeAdapter:
    """適配器：讓 trade dict 能被 compute_metrics() 使用（需 .pnl 屬性）。"""

    pnl: float
    return_pct: float = 0.0
    entry_date: date | None = None
    exit_date: date | None = None
    exit_reason: str = ""


@dataclass
class _DecisionContext:
    """update()/decide() 共用的決策日組裝結果（A2：兩入口單一實作防 drift）。

    純查詢與計算，無任何 DB 寫入副作用；kill-switch 的平倉執行留在 caller。
    """

    open_positions: list[dict]
    rankings: list[dict]
    trading_cal: list[date]
    today_prices: dict[str, float]
    today_ohlcv: dict[str, dict]
    stop_losses: dict[str, float]
    regime: str
    corr_matrix: pd.DataFrame | None
    price_rows: list | None
    vol_weights: dict[str, float] | None
    dd_pct: float
    gate_min_hold: int
    gate_score_gap: float
    gate_weekly_cap: int
    weekly_swaps_used: int


# ---------------------------------------------------------------------------
# RotationManager
# ---------------------------------------------------------------------------


class RotationManager:
    """輪動組合管理器。"""

    def __init__(self, portfolio_name: str):
        self.portfolio_name = portfolio_name

    # ── 建立 ──

    @staticmethod
    def create_portfolio(
        name: str,
        mode: str,
        max_positions: int,
        holding_days: int,
        capital: float,
        allow_renewal: bool = True,
    ) -> RotationPortfolio:
        """建立新的輪動組合。"""
        with get_session() as session:
            portfolio = RotationPortfolio(
                name=name,
                mode=mode,
                max_positions=max_positions,
                holding_days=holding_days,
                allow_renewal=allow_renewal,
                initial_capital=capital,
                current_capital=capital,
                current_cash=capital,
                status="active",
            )
            session.add(portfolio)
            session.commit()
            session.refresh(portfolio)
            logger.info("建立輪動組合: %s (mode=%s, N=%d, hold=%dd)", name, mode, max_positions, holding_days)
            return portfolio

    # ── 每日更新 ──

    def update(
        self,
        today: date | None = None,
        regime: str | None = None,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> RotationActions | None:
        """每日更新（A2 T+1 兩段式 wrapper）：先成交昨日 pending，再做今日決策。

        2026-07-06 起 live 全面 T+1（MASTER_PLAN §6.2 #9，設計文件
        docs/design/live_t1_pending_order.md）：
          1. fill_pending(today)——以 open[today] 成交昨日（含更早順延）的
             pending orders；
          2. decide(today)——以 close[today] 決策，寫明日 pending orders；
             renew 與熔斷即時執行。
        與 backtest 迴圈「每日開頭先執行上一日 pending」同構，消除 live
        「夜間決策、當日收盤成交」的 look-ahead（該價格實際拿不到）。

        Parameters
        ----------
        today : date | None
            今日日期，預設 date.today()。
        regime : str | None
            目前市場狀態（bull/sideways/bear/crisis）；None 時由
            RegimeStateMachine 持久化狀態讀取。
        dry_run : bool
            True 時 fill 與 decide 皆只計算不寫入（rotation preview 用）。
        force : bool
            True 時繞過 decide 的同日冪等 guard（fill 靠 pending order 的
            status 轉移天然冪等，不受 force 影響）。

        Returns
        -------
        decide() 的 RotationActions（to_buy/to_sell 為「明日預定操作」）；
        組合不存在/非 active/同日已決策時為 None。
        """
        if today is None:
            today = date.today()

        fill_result = self.fill_pending(exec_day=today, dry_run=dry_run)
        if fill_result and (fill_result["filled"] or fill_result["cancelled"] or fill_result["deferred"]):
            logger.info(
                "[%s] 昨日 pending 成交結果：filled=%d, cancelled=%d, deferred=%d",
                self.portfolio_name,
                fill_result["filled"],
                fill_result["cancelled"],
                fill_result["deferred"],
            )
        return self.decide(today=today, regime=regime, dry_run=dry_run, force=force)

    # ── 決策日組裝（update/decide 共用）──

    def _build_decision_context(
        self,
        session,
        portfolio: RotationPortfolio,
        today: date,
        regime: str | None,
    ) -> _DecisionContext:
        """組裝決策日所需全部輸入（無 DB 寫入副作用）。

        A2：update()（legacy 同日成交）與 decide()（T+1 決策）共用單一實作，
        差異只允許存在於「執行時點」；kill-switch 的平倉執行留在 caller。
        """
        # 載入 open positions
        open_positions = self._load_open_positions(session, portfolio.id)

        # 載入今日排名
        rankings = resolve_rankings(portfolio.mode, today, session, top_n=portfolio.max_positions * 3)
        # 若掃描器因 regime 封鎖回傳空結果，不拉取前日排名（避免用不同 regime 的名單買入）
        # regime 變數稍後才解析，這裡先預讀一次（若失敗視為未封鎖，由 fallback 處理）
        _regime_for_gate = regime
        if _regime_for_gate is None:
            try:
                from src.regime.detector import RegimeStateMachine

                _regime_for_gate = RegimeStateMachine().current_regime
            except Exception:
                _regime_for_gate = None
        from src.constants import REGIME_MODE_BLOCK as _RMB

        _mode_blocked_today = bool(
            _regime_for_gate
            and not is_composite_mode(portfolio.mode)
            and portfolio.mode in _RMB.get(_regime_for_gate, frozenset())
        )
        if not rankings and not _mode_blocked_today:
            # 嘗試找最近的 scan_date（未被 regime 封鎖時才 fallback）
            rankings = self._find_latest_rankings(session, portfolio.mode, today)
        elif not rankings and _mode_blocked_today:
            logger.info(
                "Rotation: %s 模式在 %s 被封鎖 — 跳過新買入，僅處理止損/到期/風控",
                portfolio.mode,
                _regime_for_gate,
            )

        # 交易日曆（前 120 天 ~ 未來 30 天）
        from datetime import timedelta

        cal_start = today - timedelta(days=180)
        cal_end = today + timedelta(days=60)
        trading_cal = _get_trading_calendar(session, cal_start, cal_end)

        # 今日收盤價
        all_sids = list({p["stock_id"] for p in open_positions} | {r["stock_id"] for r in rankings})
        today_prices = _get_prices_on_date(session, all_sids, today)
        # P1-4：今日完整 OHLCV，供成交端計算動態滑價 + 流動性限制（與 backtest 對齊）
        today_ohlcv = _get_ohlcv_on_date(session, all_sids, today)

        # 止損價：從持倉記錄取進場時鎖定的止損價（不隨 discover 每日浮動）
        stop_losses = {p["stock_id"]: p["stop_loss"] for p in open_positions if p.get("stop_loss") is not None}
        # 新買入候選的止損價從 rankings 取（進場後會寫入 position）
        for r in rankings:
            if r["stock_id"] not in stop_losses and r.get("stop_loss") is not None:
                stop_losses[r["stock_id"]] = r["stop_loss"]

        # ── Regime fallback：嘗試從持久化狀態讀取 ──
        if regime is None:
            try:
                from src.regime.detector import RegimeStateMachine

                rsm = RegimeStateMachine()
                regime = rsm.current_regime
                if regime:
                    logger.info("從 RegimeStateMachine 讀取 regime=%s", regime)
            except Exception:
                logger.warning(
                    "Regime 偵測失敗，使用安全預設值 %s",
                    REGIME_FALLBACK_DEFAULT,
                )
                regime = REGIME_FALLBACK_DEFAULT
        # 最終防線：若 rsm.current_regime 回傳 None / 空字串
        if not regime:
            logger.warning(
                "Regime 為空值，使用安全預設值 %s",
                REGIME_FALLBACK_DEFAULT,
            )
            regime = REGIME_FALLBACK_DEFAULT

        # ── Correlation Budget：計算持倉+候選相關性矩陣 ──
        corr_matrix = None
        price_rows = None  # 用於後續 VaR 計算
        held_sids = [p["stock_id"] for p in open_positions]
        candidate_sids = [r["stock_id"] for r in rankings[: portfolio.max_positions]]
        corr_sids = list(set(held_sids + candidate_sids))
        if len(corr_sids) >= 2:
            price_start = today - timedelta(days=90)
            stmt_prices = (
                select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close)
                .where(
                    DailyPrice.stock_id.in_(corr_sids),
                    DailyPrice.date >= price_start,
                    DailyPrice.date <= today,
                )
                .order_by(DailyPrice.date)
            )
            price_rows = session.execute(stmt_prices).all()
            if price_rows:
                price_data: dict[str, pd.Series] = {}
                for sid in corr_sids:
                    sid_rows = [(r[1], r[2]) for r in price_rows if r[0] == sid]
                    if sid_rows:
                        dates_list, prices_list = zip(*sid_rows)
                        price_data[sid] = pd.Series(prices_list, index=pd.DatetimeIndex(dates_list))
                if len(price_data) >= 2:
                    corr_matrix = compute_correlation_matrix(price_data, window=60)

        # ── 波動率反比權重：候選股波動率較高者分配較少資金 ──
        vol_weights = None
        if candidate_sids and price_rows:
            vol_dict: dict[str, float] = {}
            for sid in candidate_sids:
                sid_rows = [r[2] for r in price_rows if r[0] == sid]
                if len(sid_rows) >= 20:
                    s = pd.Series(sid_rows)
                    daily_ret = s.pct_change().dropna()
                    if len(daily_ret) >= 10:
                        vol_dict[sid] = float(daily_ret.std() * (252**0.5))
            if vol_dict:
                vol_weights = compute_vol_inverse_weights(vol_dict)
                logger.info("波動率反比權重：%s", vol_weights)

        # ── 回撤（kill-switch 判斷 + Drawdown Guard 共用同一 dd_pct）──
        # C1 修復（2026-05-09）：傳入 open_positions + today_prices，
        # 讓 equity_history 反映當日盤中浮動損益（含 gap-down），
        # 否則用過時 portfolio.current_capital 會在真實回撤時不觸發熔斷。
        equity_history = self._compute_equity_history(
            session,
            portfolio,
            open_positions=open_positions,
            today_prices=today_prices,
        )
        # P0 止血包 #3（2026-07-05）：peak 額外納入每日 snapshot MtM 權益序列，
        # 修復 realized-only 序列低估「浮盈回吐型」回撤、熔斷不觸發的漏洞。
        snapshot_capitals = self._load_snapshot_capitals(session)
        dd_pct = compute_drawdown_with_snapshots(equity_history, snapshot_capitals)

        # ── Rotation 成本閘門（A/B/C）參數（per-mode：閘門對 momentum/swing 效果相反）──
        cost_cfg = settings.quant.rotation_cost.for_mode(portfolio.mode)
        if cost_cfg.enabled:
            iso_year, iso_week, _ = today.isocalendar()
            week_start = date.fromisocalendar(iso_year, iso_week, 1)
            stmt_swaps = select(RotationPosition).where(
                RotationPosition.portfolio_id == portfolio.id,
                RotationPosition.exit_reason == "holding_expired",
                RotationPosition.exit_date >= week_start,
                RotationPosition.exit_date <= today,
            )
            weekly_swaps_used = len(list(session.execute(stmt_swaps).scalars()))
            gate_min_hold = cost_cfg.min_hold_days
            gate_score_gap = cost_cfg.score_gap_threshold
            gate_weekly_cap = cost_cfg.weekly_swap_cap
        else:
            weekly_swaps_used = 0
            gate_min_hold = 0
            gate_score_gap = 0.0
            gate_weekly_cap = 0

        return _DecisionContext(
            open_positions=open_positions,
            rankings=rankings,
            trading_cal=trading_cal,
            today_prices=today_prices,
            today_ohlcv=today_ohlcv,
            stop_losses=stop_losses,
            regime=regime,
            corr_matrix=corr_matrix,
            price_rows=price_rows,
            vol_weights=vol_weights,
            dd_pct=dd_pct,
            gate_min_hold=gate_min_hold,
            gate_score_gap=gate_score_gap,
            gate_weekly_cap=gate_weekly_cap,
            weekly_swaps_used=weekly_swaps_used,
        )

    # ── T+1 兩段式：階段 A 決策（A2）──

    def decide(
        self,
        today: date | None = None,
        regime: str | None = None,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> RotationActions | None:
        """T+1 階段 A：D 日收盤後決策，把買賣意圖寫成 RotationPendingOrder。

        與 update() 的差異：to_buy / to_sell **不成交**，逐筆落
        RotationPendingOrder(status="pending")，由 fill_pending(D+1) 以
        open[D+1] 成交。續持（renew）與 kill-switch 熔斷維持即時：
        - renew 只動 planned_exit_date / trailing stop，不涉及成交；
        - 熔斷為風控優先於 T+1 parity 的**刻意差異**（backtest 熔斷走 T+1
          open，live 即時清倉，parity 測試須顯式排除此情境）。

        當日 snapshot 於此寫入（今日 close MtM；含尚未賣出的到期持倉——
        比 legacy「假裝當日已成交」更誠實）。
        """
        if today is None:
            today = date.today()

        with get_session() as session:
            portfolio = self._load_portfolio(session)
            if portfolio is None:
                logger.warning("找不到組合: %s", self.portfolio_name)
                return None
            if portfolio.status != "active":
                logger.info("組合 %s 已暫停，跳過決策", self.portfolio_name)
                return None

            # 同日冪等 guard（decide 階段）：pending orders ∪ decide 階段 ActionLog ∪ snapshot
            if not dry_run and not force and self._already_decided_today(session, today):
                logger.info(
                    "[%s] 今日（%s）已執行過 decide，跳過（--force 可覆蓋）",
                    self.portfolio_name,
                    today,
                )
                return None

            ctx = self._build_decision_context(session, portfolio, today, regime)

            # ── Max Drawdown Kill Switch：即時強制平倉（不走 T+1）──
            if ctx.dd_pct >= MAX_DRAWDOWN_LIQUIDATE_PCT:
                logger.error(
                    "⚠️ [%s] 回撤熔斷觸發 (%.1f%%)！強制平倉所有持倉",
                    self.portfolio_name,
                    ctx.dd_pct,
                )
                if dry_run:
                    logger.info(
                        "[%s] DRY RUN — 偵測到回撤熔斷將平倉 %d 倉位（未實際寫入）",
                        self.portfolio_name,
                        len(ctx.open_positions),
                    )
                else:
                    cash = portfolio.current_cash
                    for pos in ctx.open_positions:
                        sell_action = {
                            "stock_id": pos["stock_id"],
                            "reason": "max_drawdown_liquidation",
                            "exit_price": ctx.today_prices.get(pos["stock_id"], pos["entry_price"]),
                            "days_held": 0,
                            **pos,
                        }
                        cash = self._execute_sell(session, portfolio.id, sell_action, today, cash)
                    portfolio.current_cash = cash
                    portfolio.current_capital = cash
                    portfolio.status = "liquidated"
                    portfolio.updated_at = datetime.utcnow()
                    # 組合清算 → 尚未成交的 pending 買單全部作廢（賣單標的已強平）
                    self._cancel_pending_orders(session, reason="max_drawdown_liquidation")
                    session.commit()
                    logger.error("[%s] 已強制平倉，組合狀態設為 liquidated", self.portfolio_name)
                liquidation_actions = RotationActions(
                    to_sell=[
                        {
                            **p,
                            "reason": "max_drawdown_liquidation",
                            "exit_price": ctx.today_prices.get(p["stock_id"], p["entry_price"]),
                        }
                        for p in ctx.open_positions
                    ]
                )
                if not dry_run:
                    try:
                        self._write_action_log(
                            session,
                            portfolio_name=self.portfolio_name,
                            action_date=today,
                            actions=liquidation_actions,
                            stage="liquidation",
                        )
                        session.commit()
                    except Exception as exc:
                        logger.warning("[%s] 寫入 action log（熔斷）失敗：%s", self.portfolio_name, exc)
                return liquidation_actions

            actions = compute_rotation_actions(
                current_positions=ctx.open_positions,
                new_rankings=ctx.rankings,
                max_positions=portfolio.max_positions,
                holding_days=portfolio.holding_days,
                allow_renewal=portfolio.allow_renewal,
                today=today,
                trading_calendar=ctx.trading_cal,
                current_cash=portfolio.current_cash,
                stop_losses=ctx.stop_losses,
                today_prices=ctx.today_prices,
                total_capital=portfolio.current_capital,
                corr_matrix=ctx.corr_matrix,
                vol_weights=ctx.vol_weights,
                regime=ctx.regime,
                drawdown_pct=ctx.dd_pct,
                min_hold_days=ctx.gate_min_hold,
                score_gap_threshold=ctx.gate_score_gap,
                weekly_swap_cap=ctx.gate_weekly_cap,
                weekly_swaps_used=ctx.weekly_swaps_used,
            )

            if dry_run:
                logger.info(
                    "[%s] DRY RUN 決策預覽: 明日預定賣出=%d, 續持=%d, 明日預定買入=%d, 保持=%d",
                    self.portfolio_name,
                    len(actions.to_sell),
                    len(actions.renewed),
                    len(actions.to_buy),
                    len(actions.to_hold),
                )
                return actions

            # ── 續持即時執行（只動 planned_exit / trailing stop，不涉及成交）──
            ranking_sl = {r["stock_id"]: r["stop_loss"] for r in ctx.rankings if r.get("stop_loss") is not None}
            for renew in actions.renewed:
                self._execute_renewal(session, portfolio.id, renew, new_stop_loss=ranking_sl.get(renew["stock_id"]))

            # ── 買賣意圖落庫（明日開盤成交）──
            self._write_pending_orders(session, actions, today)

            # ── ActionLog（decide 階段：pending_buy/pending_sell/renew/hold）──
            try:
                self._write_action_log(
                    session,
                    portfolio_name=self.portfolio_name,
                    action_date=today,
                    actions=actions,
                    stage="decide",
                )
            except Exception as exc:
                logger.warning("[%s] 寫入 action log（decide）失敗：%s", self.portfolio_name, exc)

            # ── Daily Snapshot：今日 close MtM（持倉未變動，到期倉明日才出）──
            market_value = sum(
                ctx.today_prices.get(p["stock_id"], p["entry_price"]) * p["shares"] for p in ctx.open_positions
            )
            portfolio.current_capital = portfolio.current_cash + market_value
            portfolio.updated_at = datetime.utcnow()
            try:
                self._write_daily_snapshot(
                    session,
                    portfolio=portfolio,
                    snapshot_date=today,
                    market_value=market_value,
                    n_holdings=len(ctx.open_positions),
                    regime=ctx.regime,
                )
            except Exception as exc:
                logger.warning("[%s] 寫入 daily snapshot 失敗：%s", self.portfolio_name, exc)
            session.commit()

            logger.info(
                "[%s] 決策完成: 明日預定賣出=%d, 續持=%d, 明日預定買入=%d, 保持=%d",
                self.portfolio_name,
                len(actions.to_sell),
                len(actions.renewed),
                len(actions.to_buy),
                len(actions.to_hold),
            )
            return actions

    def _already_decided_today(self, session, today: date) -> bool:
        """decide 階段同日冪等 guard：pending orders ∪ decide 階段 ActionLog ∪ snapshot。

        與 fill 階段分開判斷：fill 靠 pending order 的 status 轉移天然冪等，
        decide 靠本 guard。snapshot 由 decide 寫入，納入聯集與 legacy guard 一致。
        """
        has_pending = (
            session.execute(
                select(RotationPendingOrder.id)
                .where(
                    RotationPendingOrder.portfolio_name == self.portfolio_name,
                    RotationPendingOrder.decision_date == today,
                )
                .limit(1)
            ).scalar_one_or_none()
            is not None
        )
        if has_pending:
            return True
        has_decide_log = (
            session.execute(
                select(RotationActionLog.id)
                .where(
                    RotationActionLog.portfolio_name == self.portfolio_name,
                    RotationActionLog.action_date == today,
                    RotationActionLog.action_type.in_(DECIDE_STAGE_ACTION_TYPES),
                )
                .limit(1)
            ).scalar_one_or_none()
            is not None
        )
        if has_decide_log:
            return True
        has_snapshot = (
            session.execute(
                select(RotationDailySnapshot.id)
                .where(
                    RotationDailySnapshot.portfolio_name == self.portfolio_name,
                    RotationDailySnapshot.snapshot_date == today,
                )
                .limit(1)
            ).scalar_one_or_none()
            is not None
        )
        return has_snapshot

    def _write_pending_orders(self, session, actions: RotationActions, decision_date: date) -> None:
        """把 to_buy / to_sell 逐筆寫成 RotationPendingOrder（冪等：先刪同決策日舊單）。

        buy 存 allocated_capital——fill 日以 open[D+1] 重算股數（與 backtest 同式）；
        sell 存 days_held——成交時寫回 position.holding_days_count。
        """
        session.execute(
            delete(RotationPendingOrder).where(
                RotationPendingOrder.portfolio_name == self.portfolio_name,
                RotationPendingOrder.decision_date == decision_date,
            )
        )

        import json

        rows: list[RotationPendingOrder] = []
        for sell in actions.to_sell:
            rows.append(
                RotationPendingOrder(
                    portfolio_name=self.portfolio_name,
                    decision_date=decision_date,
                    side="sell",
                    stock_id=sell["stock_id"],
                    stock_name=sell.get("stock_name") or "",
                    shares=int(sell.get("shares") or 0),
                    ref_price=float(sell.get("exit_price") or sell.get("entry_price") or 0.0),
                    days_held=sell.get("days_held"),
                    reason=sell.get("reason"),
                    status=PENDING_STATUS_PENDING,
                )
            )
        for buy in actions.to_buy:
            breakdown = buy.get("score_breakdown")
            breakdown_json: str | None = None
            if breakdown:
                try:
                    breakdown_json = json.dumps(breakdown, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    breakdown_json = None
            rows.append(
                RotationPendingOrder(
                    portfolio_name=self.portfolio_name,
                    decision_date=decision_date,
                    side="buy",
                    stock_id=buy["stock_id"],
                    stock_name=buy.get("stock_name") or "",
                    shares=int(buy.get("shares") or 0),
                    ref_price=float(buy.get("entry_price") or 0.0),
                    allocated_capital=buy.get("allocated_capital"),
                    entry_rank=buy.get("rank"),
                    entry_score=buy.get("score"),
                    score_breakdown_json=breakdown_json,
                    stop_loss=buy.get("stop_loss"),
                    status=PENDING_STATUS_PENDING,
                )
            )
        if rows:
            session.add_all(rows)

    def _cancel_pending_orders(self, session, *, reason: str) -> int:
        """把本組合所有 status=pending 的訂單標為 cancelled（清算/停用時呼叫）。"""
        stmt = select(RotationPendingOrder).where(
            RotationPendingOrder.portfolio_name == self.portfolio_name,
            RotationPendingOrder.status == PENDING_STATUS_PENDING,
        )
        orders = list(session.execute(stmt).scalars())
        for order in orders:
            order.status = PENDING_STATUS_CANCELLED
        if orders:
            logger.info("[%s] 取消 %d 筆 pending orders（%s）", self.portfolio_name, len(orders), reason)
        return len(orders)

    # ── T+1 兩段式：階段 B 成交（A2）──

    def fill_pending(
        self,
        exec_day: date | None = None,
        *,
        dry_run: bool = False,
    ) -> dict | None:
        """T+1 階段 B：以 open[exec_day] 成交 decision_date < exec_day 的 pending orders。

        規則（設計文件 §3/§5 定案）：
        - 賣單優先於買單（先回收現金再買，與 backtest _execute_action_set 順序一致）。
        - 賣單無報價（停牌）時 fallback 以 ref_price（決策日 close）成交——與
          backtest 的 fallback 一致，風控賣單絕不凍結。
        - 買單無報價時順延（保持 pending）；跨逾 PENDING_BUY_TTL_TRADING_DAYS
          個交易日未成交即 cancelled（決策已過期）。此為與 backtest 的刻意差異
          （backtest 無視停牌 fallback 買進；live 買不到就是買不到）。
        - 買單股數以 allocated_capital 對 open[exec_day] 重算（compute_shares，
          與 backtest 同式）；資金不足由 _execute_buy 內建 reshrink 縮股。
        - 部分成交不轉次日（流動性縮股後 fill 多少算多少，order 仍標 filled）。
        - 冪等：靠 status 轉移天然冪等（重跑時查無 pending → no-op）。

        Returns:
            {"filled": n, "cancelled": n, "deferred": n} — portfolio 不存在時 None。
            dry_run=True 時只計算不寫入。
        """
        if exec_day is None:
            exec_day = date.today()

        with get_session() as session:
            portfolio = self._load_portfolio(session)
            if portfolio is None:
                logger.warning("找不到組合: %s", self.portfolio_name)
                return None
            if portfolio.status != "active":
                logger.info("組合 %s 非 active，跳過 fill_pending", self.portfolio_name)
                return None

            stmt = (
                select(RotationPendingOrder)
                .where(
                    RotationPendingOrder.portfolio_name == self.portfolio_name,
                    RotationPendingOrder.status == PENDING_STATUS_PENDING,
                    RotationPendingOrder.decision_date < exec_day,
                )
                .order_by(RotationPendingOrder.decision_date)
            )
            orders = list(session.execute(stmt).scalars())
            if not orders:
                return {"filled": 0, "cancelled": 0, "deferred": 0}

            from datetime import timedelta

            sids = [o.stock_id for o in orders]
            # 精確比對當日報價（無 fallback）：停牌股缺席 = 不可成交
            exec_ohlcv = _get_ohlcv_exact_date(session, sids, exec_day)
            trading_cal = _get_trading_calendar(session, exec_day - timedelta(days=30), exec_day + timedelta(days=60))

            filled = cancelled = deferred = 0
            fill_rows: list[dict] = []  # ActionLog 用（實際成交價/股數）
            cash = portfolio.current_cash

            # ── 賣單優先（回收現金）──
            sells = [o for o in orders if o.side == "sell"]
            buys = [o for o in orders if o.side == "buy"]

            for order in sells:
                ohlcv = exec_ohlcv.get(order.stock_id) or {}
                # 停牌 fallback：以決策日 close 成交（與 backtest 一致，風控賣單不凍結）
                exit_price = ohlcv.get("open") or order.ref_price
                sell_slip = (
                    compute_dynamic_slippage(
                        ohlcv.get("volume", 0), ohlcv.get("high", 0), ohlcv.get("low", 0), exit_price, side="sell"
                    )
                    if ohlcv
                    else SLIPPAGE_RATE
                )
                if dry_run:
                    filled += 1
                    continue
                sell_action = {
                    "stock_id": order.stock_id,
                    "reason": order.reason or "holding_expired",
                    "exit_price": exit_price,
                    "days_held": order.days_held or 0,
                }
                cash = self._execute_sell(session, portfolio.id, sell_action, exec_day, cash, slippage=sell_slip)
                order.status = PENDING_STATUS_FILLED
                order.exec_date = exec_day
                filled += 1
                fill_rows.append(
                    {
                        "action_type": "close",
                        "reason": order.reason,
                        "stock_id": order.stock_id,
                        "stock_name": order.stock_name,
                        "shares": order.shares,
                        "price": exit_price,
                        "ref_price": order.ref_price,
                    }
                )

            # ── 買單 ──
            for order in buys:
                ohlcv = exec_ohlcv.get(order.stock_id) or {}
                open_price = ohlcv.get("open")
                if not open_price:
                    # 無報價：TTL 內順延，逾期 cancel（買單決策已過期）
                    elapsed_tds = sum(1 for d in trading_cal if order.decision_date < d <= exec_day)
                    if elapsed_tds > PENDING_BUY_TTL_TRADING_DAYS:
                        if not dry_run:
                            order.status = PENDING_STATUS_CANCELLED
                        cancelled += 1
                        logger.info(
                            "[%s] pending 買單 %s 逾 %d 交易日無報價，取消",
                            self.portfolio_name,
                            order.stock_id,
                            PENDING_BUY_TTL_TRADING_DAYS,
                        )
                    else:
                        deferred += 1
                    continue

                buy_slip = compute_dynamic_slippage(
                    ohlcv.get("volume", 0), ohlcv.get("high", 0), ohlcv.get("low", 0), open_price, side="buy"
                )
                # 股數以決策時分配資金對 open 重算（與 backtest 同式）；
                # 舊單無 allocated_capital 時以 ref_price × 規劃股數近似
                alloc = order.allocated_capital or (order.ref_price * order.shares)
                shares = compute_shares(
                    alloc,
                    open_price,
                    slippage=buy_slip,
                    daily_volume=ohlcv.get("volume"),
                    participation_limit=LIQUIDITY_PARTICIPATION_LIMIT,
                )
                if shares <= 0:
                    if not dry_run:
                        order.status = PENDING_STATUS_CANCELLED
                    cancelled += 1
                    continue
                if dry_run:
                    filled += 1
                    continue

                breakdown = None
                if order.score_breakdown_json:
                    import json

                    try:
                        breakdown = json.loads(order.score_breakdown_json)
                    except (TypeError, ValueError):
                        breakdown = None
                buy_action = {
                    "stock_id": order.stock_id,
                    "stock_name": order.stock_name or "",
                    "entry_price": open_price,
                    "shares": shares,
                    "rank": order.entry_rank or 0,
                    "score": order.entry_score,
                    "allocated_capital": alloc,
                    "stop_loss": order.stop_loss,
                    "score_breakdown": breakdown,
                }
                cash_after = self._execute_buy(
                    session,
                    portfolio.id,
                    buy_action,
                    exec_day,
                    trading_cal,
                    cash,
                    slippage=buy_slip,
                    daily_volume=ohlcv.get("volume"),
                )
                if cash_after == cash:
                    # _execute_buy 內 reshrink 後 shares<=0 → 未成交（現金不足）
                    order.status = PENDING_STATUS_CANCELLED
                    cancelled += 1
                    logger.warning("[%s] pending 買單 %s 現金不足未成交，取消", self.portfolio_name, order.stock_id)
                    continue
                cash = cash_after
                order.status = PENDING_STATUS_FILLED
                order.exec_date = exec_day
                filled += 1
                fill_rows.append(
                    {
                        "action_type": "open",
                        "reason": None,
                        "stock_id": order.stock_id,
                        "stock_name": order.stock_name,
                        "shares": shares,
                        "price": open_price,
                        "entry_rank": order.entry_rank,
                        "ref_price": order.ref_price,
                    }
                )

            if dry_run:
                logger.info(
                    "[%s] DRY RUN fill_pending(%s)：可成交=%d, 取消=%d, 順延=%d",
                    self.portfolio_name,
                    exec_day,
                    filled,
                    cancelled,
                    deferred,
                )
                return {"filled": filled, "cancelled": cancelled, "deferred": deferred}

            # ── 資金與市值收斂 ──
            open_after = self._load_open_positions(session, portfolio.id)
            after_prices = _get_prices_on_date(session, [p["stock_id"] for p in open_after], exec_day)
            market_value = sum(after_prices.get(p["stock_id"], p["entry_price"]) * p["shares"] for p in open_after)
            portfolio.current_cash = cash
            portfolio.current_capital = cash + market_value
            portfolio.updated_at = datetime.utcnow()

            # ── ActionLog（fill 階段：open/close，實際成交價）──
            try:
                self._write_fill_action_log(session, exec_day, fill_rows)
            except Exception as exc:
                logger.warning("[%s] 寫入 action log（fill）失敗：%s", self.portfolio_name, exc)
            session.commit()

            logger.info(
                "[%s] fill_pending(%s) 完成：成交=%d, 取消=%d, 順延=%d | 現金=%.0f, 總資產=%.0f",
                self.portfolio_name,
                exec_day,
                filled,
                cancelled,
                deferred,
                cash,
                portfolio.current_capital,
            )
            return {"filled": filled, "cancelled": cancelled, "deferred": deferred}

    def _write_fill_action_log(self, session, exec_day: date, fill_rows: list[dict]) -> None:
        """fill 階段 ActionLog：寫實際成交的 open/close 列（冪等：只刪同日 open/close）。

        不動同日 decide 階段的 pending_buy/pending_sell/renew/hold 列。
        """
        if not fill_rows:
            return
        session.execute(
            delete(RotationActionLog).where(
                RotationActionLog.portfolio_name == self.portfolio_name,
                RotationActionLog.action_date == exec_day,
                RotationActionLog.action_type.in_(("open", "close")),
            )
        )
        non_risk_sells = [
            r for r in fill_rows if r["action_type"] == "close" and r.get("reason") not in self._RISK_EXIT_REASONS
        ]
        has_buys = any(r["action_type"] == "open" for r in fill_rows)
        is_switch_day = bool(non_risk_sells) and has_buys
        switch_group = f"{self.portfolio_name}:{exec_day.isoformat()}" if is_switch_day else None

        rows: list[RotationActionLog] = []
        for r in fill_rows:
            is_close = r["action_type"] == "close"
            is_risk = is_close and r.get("reason") in self._RISK_EXIT_REASONS
            rows.append(
                RotationActionLog(
                    portfolio_name=self.portfolio_name,
                    action_date=exec_day,
                    action_type=r["action_type"],
                    reason=r.get("reason"),
                    is_risk_exit=is_risk,
                    switch_group=None if is_risk else switch_group,
                    stock_id=r["stock_id"],
                    stock_name=r.get("stock_name"),
                    shares=r.get("shares"),
                    price=r.get("price"),
                    entry_rank=r.get("entry_rank"),
                )
            )
        session.add_all(rows)

    # ── Daily Snapshot ──

    def _write_daily_snapshot(
        self,
        session,
        *,
        portfolio: RotationPortfolio,
        snapshot_date: date,
        market_value: float,
        n_holdings: int,
        regime: str | None,
    ) -> None:
        """寫入單日權益快照（同日重複呼叫採 update 而非新增，避免 UniqueConstraint 衝突）。

        2026-05-15 sprint：附加計算 0050 benchmark 與 alpha：
        - benchmark_return_pct = (today_0050 - prev_snapshot_0050) / prev_snapshot_0050
        - benchmark_cum_return_pct = (today_0050 - base_0050) / base_0050
          base = 該 portfolio 在 rotation_daily_snapshot 中最早一筆的對應 0050 close
        - alpha_cum_pct = portfolio_cum_return - benchmark_cum_return_pct
        0050 缺資料時 3 欄位皆 None。
        """
        unrealized_pnl = market_value + portfolio.current_cash - portfolio.initial_capital

        existing = session.execute(
            select(RotationDailySnapshot).where(
                RotationDailySnapshot.portfolio_name == portfolio.name,
                RotationDailySnapshot.snapshot_date == snapshot_date,
            )
        ).scalar_one_or_none()

        prev = session.execute(
            select(RotationDailySnapshot)
            .where(
                RotationDailySnapshot.portfolio_name == portfolio.name,
                RotationDailySnapshot.snapshot_date < snapshot_date,
            )
            .order_by(RotationDailySnapshot.snapshot_date.desc())
            .limit(1)
        ).scalar_one_or_none()

        # 該 portfolio 最早 snapshot — 作為 cum return 的 base
        # 若 existing 為當前正在寫入的第一筆（None prev），則 base 也是當前 snapshot_date
        first_snapshot = session.execute(
            select(RotationDailySnapshot)
            .where(RotationDailySnapshot.portfolio_name == portfolio.name)
            .order_by(RotationDailySnapshot.snapshot_date.asc())
            .limit(1)
        ).scalar_one_or_none()

        daily_return_pct: float | None
        if prev is not None and prev.total_capital > 0:
            daily_return_pct = (portfolio.current_capital - prev.total_capital) / prev.total_capital
        else:
            daily_return_pct = None

        # ── Benchmark / alpha 計算（純函數）──
        today_bm_close = _get_benchmark_close_on_or_before(session, snapshot_date)
        prev_bm_close = _get_benchmark_close_on_or_before(session, prev.snapshot_date) if prev is not None else None
        base_date = first_snapshot.snapshot_date if first_snapshot is not None else snapshot_date
        base_bm_close = _get_benchmark_close_on_or_before(session, base_date)

        portfolio_cum_return: float | None = None
        if portfolio.initial_capital > 0:
            portfolio_cum_return = (portfolio.current_capital - portfolio.initial_capital) / portfolio.initial_capital

        benchmark_return_pct, benchmark_cum_return_pct, alpha_cum_pct = compute_benchmark_alpha_fields(
            today_bm_close=today_bm_close,
            prev_bm_close=prev_bm_close,
            base_bm_close=base_bm_close,
            portfolio_cum_return=portfolio_cum_return,
        )

        if existing is not None:
            existing.total_capital = portfolio.current_capital
            existing.total_market_value = market_value
            existing.total_cash = portfolio.current_cash
            existing.unrealized_pnl = unrealized_pnl
            existing.daily_return_pct = daily_return_pct
            existing.n_holdings = n_holdings
            existing.regime_state = regime
            existing.benchmark_return_pct = benchmark_return_pct
            existing.benchmark_cum_return_pct = benchmark_cum_return_pct
            existing.alpha_cum_pct = alpha_cum_pct
        else:
            session.add(
                RotationDailySnapshot(
                    portfolio_name=portfolio.name,
                    snapshot_date=snapshot_date,
                    total_capital=portfolio.current_capital,
                    total_market_value=market_value,
                    total_cash=portfolio.current_cash,
                    unrealized_pnl=unrealized_pnl,
                    daily_return_pct=daily_return_pct,
                    n_holdings=n_holdings,
                    regime_state=regime,
                    benchmark_return_pct=benchmark_return_pct,
                    benchmark_cum_return_pct=benchmark_cum_return_pct,
                    alpha_cum_pct=alpha_cum_pct,
                )
            )
        session.commit()

    # ── Action Log ──

    # 風控出場原因（UI 以 ⚠️ 區分；非單純到期/排名換股）
    _RISK_EXIT_REASONS = frozenset({"stop_loss", "crisis_exit", "max_drawdown_liquidation"})

    def _write_action_log(
        self,
        session,
        *,
        portfolio_name: str,
        action_date: date,
        actions: RotationActions,
        stage: str = "legacy",
    ) -> None:
        """把當日 RotationActions 逐筆落庫，供 dashboard `today_actions` 帶出。

        冪等：morning-routine 同日可能重跑，故寫入前先刪除當日同 portfolio 舊紀錄，
        再依 to_buy / to_sell / renewed / to_hold 重建。

        stage（A2 T+1 兩段式）：
        - "legacy"：update() 同日成交路徑——buy/sell 寫 open/close，刪除範圍為
          當日全部紀錄（維持既有行為）。
        - "decide"：decide() 決策路徑——buy/sell 寫 pending_buy/pending_sell
          （明日預定操作），刪除範圍僅限 decide 階段類型，**不可**動到同日
          fill_pending() 已寫入的 open/close（D 日先 fill 昨日單、再 decide）。
        - "liquidation"：decide() 的熔斷即時清倉——sell 寫 close，但刪除範圍
          同 decide 階段（保留同日早上 fill 的成交紀錄）。

        換股標記：同日同時存在「非風控賣出（到期/排名換股）」與「新買入」時，
        將當日所有非風控賣出與買入標上同一 switch_group，供 UI 顯示 🔁。
        gated（gate_b/gate_c 阻擋）的 to_hold 不計為換股對象。
        """
        delete_stmt = delete(RotationActionLog).where(
            RotationActionLog.portfolio_name == portfolio_name,
            RotationActionLog.action_date == action_date,
        )
        if stage in ("decide", "liquidation"):
            delete_stmt = delete_stmt.where(RotationActionLog.action_type.in_(DECIDE_STAGE_ACTION_TYPES))
        session.execute(delete_stmt)

        buy_type = ACTION_TYPE_PENDING_BUY if stage == "decide" else "open"
        sell_type = ACTION_TYPE_PENDING_SELL if stage == "decide" else "close"

        non_risk_sells = [s for s in actions.to_sell if s.get("reason") not in self._RISK_EXIT_REASONS]
        is_switch_day = bool(non_risk_sells) and bool(actions.to_buy)
        switch_group = f"{portfolio_name}:{action_date.isoformat()}" if is_switch_day else None

        rows: list[RotationActionLog] = []

        for b in actions.to_buy:
            rows.append(
                RotationActionLog(
                    portfolio_name=portfolio_name,
                    action_date=action_date,
                    action_type=buy_type,
                    reason=None,
                    is_risk_exit=False,
                    switch_group=switch_group,
                    stock_id=b["stock_id"],
                    stock_name=b.get("stock_name"),
                    shares=b.get("shares"),
                    price=b.get("entry_price"),
                    entry_rank=b.get("rank"),
                )
            )

        for s in actions.to_sell:
            reason = s.get("reason")
            is_risk = reason in self._RISK_EXIT_REASONS
            entry_price = s.get("entry_price")
            exit_price = s.get("exit_price")
            shares = s.get("shares")
            return_pct: float | None = None
            pnl: float | None = None
            # 顯示用報酬：以 exit/entry 概算（不含成本，與 RotationPosition 已實現損益略有差異）
            if entry_price and exit_price is not None:
                return_pct = (exit_price - entry_price) / entry_price * 100.0
                if shares:
                    pnl = (exit_price - entry_price) * shares
            rows.append(
                RotationActionLog(
                    portfolio_name=portfolio_name,
                    action_date=action_date,
                    action_type=sell_type,
                    reason=reason,
                    is_risk_exit=is_risk,
                    switch_group=None if is_risk else switch_group,
                    stock_id=s["stock_id"],
                    stock_name=s.get("stock_name"),
                    shares=shares,
                    price=exit_price,
                    entry_rank=s.get("entry_rank"),
                    pnl=pnl,
                    return_pct=return_pct,
                )
            )

        for r in actions.renewed:
            rows.append(
                RotationActionLog(
                    portfolio_name=portfolio_name,
                    action_date=action_date,
                    action_type="renew",
                    reason=None,
                    is_risk_exit=False,
                    switch_group=None,
                    stock_id=r["stock_id"],
                    stock_name=r.get("stock_name"),
                    shares=r.get("shares"),
                    price=r.get("entry_price"),
                    entry_rank=r.get("rank") or r.get("entry_rank"),
                )
            )

        for h in actions.to_hold:
            rows.append(
                RotationActionLog(
                    portfolio_name=portfolio_name,
                    action_date=action_date,
                    action_type="hold",
                    reason=h.get("gated_by"),  # gate_b_score_gap / gate_c_weekly_cap，一般保持為 None
                    is_risk_exit=False,
                    switch_group=None,
                    stock_id=h["stock_id"],
                    stock_name=h.get("stock_name"),
                    shares=h.get("shares"),
                    price=h.get("entry_price"),
                    entry_rank=h.get("rank") or h.get("entry_rank"),
                )
            )

        if rows:
            session.add_all(rows)

    @staticmethod
    def backfill_snapshot_benchmark_alpha(
        session,
        *,
        portfolio_name: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, int]:
        """重算歷史 snapshot 的 benchmark/alpha 三欄位。

        2026-05-16：commit 7f13f08 加欄位前的 33 筆 snapshot 全為 NULL，需一次性補齊。

        Args:
            portfolio_name: 限制單一 portfolio；None=全部
            overwrite: True 時連已有值的列也重算；False（預設）只補 NULL

        Returns:
            {"updated": N, "skipped_no_initial_capital": K, "skipped_no_benchmark": M}
        """
        # 依 (name, date asc) 排序，逐筆計算（需 base = 該 portfolio 最早 snapshot）
        stmt = select(RotationDailySnapshot)
        if portfolio_name is not None:
            stmt = stmt.where(RotationDailySnapshot.portfolio_name == portfolio_name)
        stmt = stmt.order_by(RotationDailySnapshot.portfolio_name, RotationDailySnapshot.snapshot_date)
        rows = session.execute(stmt).scalars().all()

        # 預先載每個 portfolio 的 initial_capital 與最早 snapshot_date（base）
        portfolio_names = {r.portfolio_name for r in rows}
        portfolios = (
            session.execute(select(RotationPortfolio).where(RotationPortfolio.name.in_(portfolio_names)))
            .scalars()
            .all()
        )
        initial_cap_map = {p.name: p.initial_capital for p in portfolios}

        base_date_map: dict[str, date] = {}
        for r in rows:
            base_date_map.setdefault(r.portfolio_name, r.snapshot_date)

        # 預先取每個 base_date 的 0050 close（reuse helper）
        base_bm_map: dict[str, float | None] = {
            name: _get_benchmark_close_on_or_before(session, d) for name, d in base_date_map.items()
        }

        # 群組 prev close 查詢：對每筆 row 依該 portfolio 上一筆 snapshot_date 找 0050
        prev_date_by_row: dict[int, date | None] = {}
        last_seen: dict[str, date] = {}
        for r in rows:
            prev_date_by_row[r.id] = last_seen.get(r.portfolio_name)
            last_seen[r.portfolio_name] = r.snapshot_date

        stats = {"updated": 0, "skipped_no_initial_capital": 0, "skipped_no_benchmark": 0}

        for r in rows:
            already_filled = r.alpha_cum_pct is not None
            if already_filled and not overwrite:
                continue

            initial_capital = initial_cap_map.get(r.portfolio_name, 0.0) or 0.0
            if initial_capital <= 0:
                stats["skipped_no_initial_capital"] += 1
                continue

            today_bm = _get_benchmark_close_on_or_before(session, r.snapshot_date)
            prev_date = prev_date_by_row.get(r.id)
            prev_bm = _get_benchmark_close_on_or_before(session, prev_date) if prev_date is not None else None
            base_bm = base_bm_map.get(r.portfolio_name)

            if today_bm is None or base_bm is None:
                stats["skipped_no_benchmark"] += 1
                continue

            portfolio_cum_return = (r.total_capital - initial_capital) / initial_capital
            bm_ret, bm_cum, alpha_cum = compute_benchmark_alpha_fields(
                today_bm_close=today_bm,
                prev_bm_close=prev_bm,
                base_bm_close=base_bm,
                portfolio_cum_return=portfolio_cum_return,
            )
            r.benchmark_return_pct = bm_ret
            r.benchmark_cum_return_pct = bm_cum
            r.alpha_cum_pct = alpha_cum
            stats["updated"] += 1

        session.commit()
        return stats

    def get_recent_snapshots(self, n_days: int = 90) -> list[dict]:
        """回傳本組合最近 n_days 個 snapshot（asc by snapshot_date）。

        注意：rotation 改名後與舊 snapshot 名稱不一致時會斷鏈，僅撈當前 portfolio_name。
        """
        with get_session() as session:
            stmt = (
                select(RotationDailySnapshot)
                .where(RotationDailySnapshot.portfolio_name == self.portfolio_name)
                .order_by(RotationDailySnapshot.snapshot_date.desc())
                .limit(n_days)
            )
            rows = session.execute(stmt).scalars().all()
            rows_asc = list(reversed(rows))
            return [
                {
                    "snapshot_date": r.snapshot_date,
                    "total_capital": r.total_capital,
                    "total_market_value": r.total_market_value,
                    "total_cash": r.total_cash,
                    "unrealized_pnl": r.unrealized_pnl,
                    "daily_return_pct": r.daily_return_pct,
                    "n_holdings": r.n_holdings,
                    "regime_state": r.regime_state,
                    "benchmark_return_pct": r.benchmark_return_pct,
                    "benchmark_cum_return_pct": r.benchmark_cum_return_pct,
                    "alpha_cum_pct": r.alpha_cum_pct,
                }
                for r in rows_asc
            ]

    # ── 進場理由 backfill（P1 任務 5）──

    @staticmethod
    def backfill_entry_score_breakdown(
        session,
        *,
        portfolio_name: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, int]:
        """重建歷史 RotationPosition 的 entry_score_breakdown_json。

        以 (entry_date, portfolio.mode, stock_id) 反查 DiscoveryRecord；mode='all' 時
        以 _record_to_score_breakdown 同邏輯挑 primary_mode。找不到 record 則略過。

        Args:
            portfolio_name: 限制單一 portfolio；None=全部
            overwrite: True 連已有值的列也重算；False（預設）只補 NULL

        Returns:
            {"updated": N, "skipped_no_record": K, "skipped_already_filled": M}
        """
        import json

        stmt = select(RotationPosition, RotationPortfolio).join(
            RotationPortfolio, RotationPosition.portfolio_id == RotationPortfolio.id
        )
        if portfolio_name is not None:
            stmt = stmt.where(RotationPortfolio.name == portfolio_name)
        rows = session.execute(stmt).all()

        stats = {"updated": 0, "skipped_no_record": 0, "skipped_already_filled": 0}

        for pos, portfolio in rows:
            if pos.entry_score_breakdown_json is not None and not overwrite:
                stats["skipped_already_filled"] += 1
                continue

            breakdown: dict | None = None
            if is_composite_mode(portfolio.mode):
                members = COMPOSITE_MODES[portfolio.mode]["members"]
                # 多 mode：抓該股當日 members 內所有 record，挑 primary_mode
                recs = (
                    session.execute(
                        select(DiscoveryRecord).where(
                            DiscoveryRecord.scan_date == pos.entry_date,
                            DiscoveryRecord.stock_id == pos.stock_id,
                            DiscoveryRecord.mode.in_(members),
                        )
                    )
                    .scalars()
                    .all()
                )
                if recs:
                    mode_scores = {r.mode: r.composite_score for r in recs}
                    primary_mode = max(mode_scores.items(), key=lambda kv: kv[1])[0]
                    primary_rec = next(r for r in recs if r.mode == primary_mode)
                    breakdown = _record_to_score_breakdown(primary_rec, primary_mode=primary_mode)
                    breakdown["mode_scores"] = mode_scores
                    breakdown["avg_score"] = round(sum(mode_scores.values()) / len(mode_scores), 6)
                    breakdown["mode"] = portfolio.mode
            else:
                rec = session.execute(
                    select(DiscoveryRecord).where(
                        DiscoveryRecord.scan_date == pos.entry_date,
                        DiscoveryRecord.mode == portfolio.mode,
                        DiscoveryRecord.stock_id == pos.stock_id,
                    )
                ).scalar_one_or_none()
                if rec is not None:
                    breakdown = _record_to_score_breakdown(rec)

            if breakdown is None:
                stats["skipped_no_record"] += 1
                continue

            try:
                pos.entry_score_breakdown_json = json.dumps(breakdown, ensure_ascii=False, default=str)
                stats["updated"] += 1
            except (TypeError, ValueError):
                stats["skipped_no_record"] += 1

        session.commit()
        return stats

    # ── 成本歸因（5/29 audit alpha 拖累驗證）──

    def get_cost_attribution(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        *,
        include_open: bool = False,
    ):
        """聚合本組合在 [start, end] 期間 RotationPosition 的成本歸因。

        Parameters
        ----------
        start_date, end_date : 期間過濾。entry_date 落在窗口內視為納入。
            None 表不過濾該端。
        include_open : 是否納入 open position（只計買端成本）。預設 False。

        Returns
        -------
        PositionCostAttribution | None
            找不到 portfolio 時回傳 None。
        """
        from src.portfolio.rotation import compute_positions_cost_attribution

        with get_session() as session:
            portfolio = self._load_portfolio(session)
            if portfolio is None:
                return None

            stmt = select(RotationPosition).where(RotationPosition.portfolio_id == portfolio.id)
            if start_date is not None:
                stmt = stmt.where(RotationPosition.entry_date >= start_date)
            if end_date is not None:
                stmt = stmt.where(RotationPosition.entry_date <= end_date)
            rows = session.execute(stmt).scalars().all()

            positions: list[dict] = []
            for r in rows:
                if r.status != "closed" and not include_open:
                    continue
                positions.append(
                    {
                        "entry_price": r.entry_price,
                        "exit_price": r.exit_price,
                        "shares": r.shares,
                        "status": r.status,
                        "buy_slippage": r.buy_slippage,
                        "sell_slippage": r.sell_slippage,
                    }
                )

            return compute_positions_cost_attribution(
                positions,
                portfolio_name=portfolio.name,
                initial_capital=portfolio.initial_capital,
            )

    # ── 回測 ──

    def backtest(
        self,
        start_date: date,
        end_date: date,
        mode: str | None = None,
        max_positions: int | None = None,
        holding_days: int | None = None,
        capital: float | None = None,
        allow_renewal: bool = True,
        dynamic_slippage: bool = True,
        liquidity_limit: float = LIQUIDITY_PARTICIPATION_LIMIT,
        limit_price_check: bool = False,
        disable_stop_loss: bool = False,
        stop_loss_widen: float = 1.0,
        t1_execution: bool = True,
        save_result: bool = True,
    ) -> RotationBacktestResult:
        """歷史回測：逐日模擬 rotation 策略。

        可直接從已建立的 portfolio 讀取參數，
        或用傳入參數覆蓋（適用於 ad-hoc 回測）。

        Parameters
        ----------
        dynamic_slippage : bool
            啟用三因子動態滑價模型（預設 True）。
        liquidity_limit : float
            流動性約束比例（預設 5%）。
        limit_price_check : bool
            啟用漲跌停模擬（預設 False，向後相容）。
        t1_execution : bool
            **研究旋鈕（A2 parity 報告專用，live 一律 T+1）**：False 時改為
            決策日收盤成交（2026-07-06 前的舊 live 行為，look-ahead），供量化
            「close 成交 vs T+1 open」的 alpha 差距；預設 True 勿動。
        """
        with get_session() as session:
            # 參數解析
            portfolio = self._load_portfolio(session)
            if portfolio is not None:
                mode = mode or portfolio.mode
                max_positions = max_positions or portfolio.max_positions
                holding_days = holding_days or portfolio.holding_days
                capital = capital or portfolio.initial_capital
                allow_renewal = portfolio.allow_renewal
            else:
                if mode is None or max_positions is None or holding_days is None or capital is None:
                    raise ValueError("未指定回測參數且找不到組合設定")

            # 交易日曆
            trading_cal = _get_trading_calendar(session, start_date, end_date)
            if not trading_cal:
                return RotationBacktestResult(config={"error": "無交易日資料"})

            # 收集所有 scan_date 的 DiscoveryRecord
            all_rankings: dict[date, list[dict]] = {}
            if is_composite_mode(mode):
                stmt = (
                    select(DiscoveryRecord.scan_date)
                    .where(
                        DiscoveryRecord.scan_date >= start_date,
                        DiscoveryRecord.scan_date <= end_date,
                    )
                    .distinct()
                )
            else:
                stmt = (
                    select(DiscoveryRecord.scan_date)
                    .where(
                        DiscoveryRecord.scan_date >= start_date,
                        DiscoveryRecord.scan_date <= end_date,
                        DiscoveryRecord.mode == mode,
                    )
                    .distinct()
                )
            scan_dates = {row[0] for row in session.execute(stmt).all()}

            for sd in scan_dates:
                all_rankings[sd] = resolve_rankings(mode, sd, session, top_n=max_positions * 3)

            # 逐日歷史 regime（取自 DiscoveryRecord，與當日 discover 計算一致 → 無 look-ahead）。
            # P0-1 修復：rotation backtest 過去未傳 regime，導致 Crisis 阻擋/相關性收緊在回測中失效。
            all_regime: dict[date, str] = {}
            regime_rows = session.execute(
                select(DiscoveryRecord.scan_date, DiscoveryRecord.regime)
                .where(
                    DiscoveryRecord.scan_date >= start_date,
                    DiscoveryRecord.scan_date <= end_date,
                    DiscoveryRecord.regime.isnot(None),
                )
                .distinct()
            ).all()
            for _sd, _rg in regime_rows:
                all_regime.setdefault(_sd, _rg)

            # TAIEX benchmark 資料
            taiex_prices = _get_taiex_prices(session, trading_cal[0], trading_cal[-1])

            # ── OHLCV 預載入：一次查詢整段時間範圍，避免逐日 DB 查詢 ──
            _bt_stock_ids: set[str] = {"TAIEX"}
            for _sd_rankings in all_rankings.values():
                for _r in _sd_rankings:
                    _bt_stock_ids.add(_r["stock_id"])
            _ohlcv_cache: dict[date, dict[str, dict]] = {}
            if _bt_stock_ids:
                _ohlcv_stmt = select(
                    DailyPrice.stock_id,
                    DailyPrice.date,
                    DailyPrice.open,
                    DailyPrice.high,
                    DailyPrice.low,
                    DailyPrice.close,
                    DailyPrice.volume,
                ).where(
                    DailyPrice.stock_id.in_(list(_bt_stock_ids)),
                    DailyPrice.date >= start_date,
                    DailyPrice.date <= end_date,
                )
                for row in session.execute(_ohlcv_stmt).all():
                    sid, d, o, h, lo, c, v = row
                    _ohlcv_cache.setdefault(d, {})[sid] = {
                        "open": o,
                        "high": h,
                        "low": lo,
                        "close": c,
                        "volume": v or 0,
                    }

            # 逐日模擬
            positions: list[dict] = []  # 模擬持倉
            cash = capital
            equity_curve: list[dict] = []
            all_trades: list[dict] = []
            last_rankings: list[dict] = []
            # 逐日 regime（無 scan 記錄的日子沿用前值；起始用安全預設）
            last_regime: str = REGIME_FALLBACK_DEFAULT

            # 成本歸因累計器
            total_commission = 0.0
            total_tax = 0.0
            total_slippage_cost = 0.0
            turnover_value = 0.0

            # 漲跌停偵測用前日收盤價（亦作為停牌/下市個股的「最後已知價」）
            prev_close_map: dict[str, float] = {}
            # P1-3 survivorship：期末仍持有但個股已停止交易（停牌/下市）的倉位數
            n_stranded = 0

            # VaR 計算用：累計每日收盤價（最近 60 日窗口）
            daily_close_history: dict[str, list[float]] = {}  # {stock_id: [close1, close2, ...]}
            # Correlation Budget / 波動率反比權重用：日期索引滾動窗口（與 live update 對齊；
            # 須帶日期 index 才能跨股對齊，避免不同歷史長度的 series 位置錯位）
            close_window_dated: dict[str, list[tuple[date, float]]] = {}

            # 每日持倉快照
            daily_positions_snapshot: list[dict] = []

            # 成本閘門：每 ISO 週 holding_expired 累計（per-mode 解析，與 live update() 對齊）
            cost_cfg = settings.quant.rotation_cost.for_mode(mode)
            weekly_swap_counter: dict[tuple[int, int], int] = {}

            # ── T+1 執行（audit P0-2）──
            # D 日收盤後用 close[D] 決策 → 暫存 pending_exec → D+1 開盤成交（買賣一律 open[D+1]），
            # 消除「用 close[D] 排名又成交在 close[D]」的 look-ahead。
            pending_exec: tuple[RotationActions, list[dict]] | None = None

            def _execute_action_set(
                acts: RotationActions,
                decision_rankings: list[dict],
                exec_ohlcv: dict[str, dict],
                exec_day: date,
            ) -> None:
                """以 exec_day 開盤價成交 acts（T+1）。就地更新 positions / cash / 成本累計 / all_trades。"""
                nonlocal cash, positions, total_commission, total_tax, total_slippage_cost, turnover_value

                # ── 賣出（到期/換股/止損/危機/回撤 一律 open[exec_day]）──
                sold_ids: set[str] = set()
                for sell in acts.to_sell:
                    sid = sell["stock_id"]
                    ohlcv = exec_ohlcv.get(sid, {})
                    # 成交價 = 執行日開盤；無開盤資料時 fallback 決策日 exit_price / entry_price
                    exit_price = ohlcv.get("open") or sell.get("exit_price") or sell.get("entry_price", 0)
                    shares = sell.get("shares", 0)

                    if dynamic_slippage:
                        sell_slip = compute_dynamic_slippage(
                            ohlcv.get("volume", 0), ohlcv.get("high", 0), ohlcv.get("low", 0), exit_price, side="sell"
                        )
                    else:
                        sell_slip = SLIPPAGE_RATE

                    # 跌停模擬：跌停時以跌停價成交
                    if limit_price_check:
                        prev_c = prev_close_map.get(sid)
                        if prev_c:
                            _, is_down = detect_limit_price(ohlcv.get("open", exit_price), prev_c)
                            if is_down:
                                exit_price = min(exit_price, prev_c * (1 - SLIPPAGE_RATE))

                    buy_slip = sell.get("buy_slippage", SLIPPAGE_RATE)
                    fill = simulate_sell(
                        sell["entry_price"], exit_price, shares, buy_slippage=buy_slip, sell_slippage=sell_slip
                    )
                    pnl, return_pct = fill.pnl, fill.return_pct
                    costs = fill.costs
                    cash += fill.proceeds

                    total_commission += costs.commission
                    total_tax += costs.tax
                    total_slippage_cost += costs.slippage_cost
                    turnover_value += exit_price * shares

                    all_trades.append(
                        {
                            "stock_id": sid,
                            "entry_date": sell["entry_date"],
                            "entry_price": sell["entry_price"],
                            "exit_date": exec_day,
                            "exit_price": exit_price,
                            "shares": shares,
                            "pnl": pnl,
                            "return_pct": return_pct,
                            "exit_reason": sell["reason"],
                            "entry_rank": sell.get("entry_rank"),
                            "entry_score": sell.get("entry_score"),
                            "buy_slippage": buy_slip,
                            "sell_slippage": sell_slip,
                            "commission": costs.commission,
                            "tax": costs.tax,
                            "slippage_cost": costs.slippage_cost,
                        }
                    )
                    sold_ids.add(sid)

                positions = [p for p in positions if p["stock_id"] not in sold_ids]

                # ── 續持（planned_exit 延展 + 止損價只上移）──
                renewed_ids = {r["stock_id"] for r in acts.renewed}
                ranking_sl = {
                    r["stock_id"]: r["stop_loss"] for r in decision_rankings if r.get("stop_loss") is not None
                }
                for pos in positions:
                    if pos["stock_id"] in renewed_ids:
                        renew_info = next(r for r in acts.renewed if r["stock_id"] == pos["stock_id"])
                        pos["planned_exit_date"] = renew_info["new_planned_exit_date"]
                        new_sl = ranking_sl.get(pos["stock_id"])
                        if new_sl is not None:
                            old_sl = pos.get("stop_loss")
                            if old_sl is None or new_sl > old_sl:
                                pos["stop_loss"] = new_sl

                # ── 買入（open[exec_day]）──
                for buy in acts.to_buy:
                    sid = buy["stock_id"]
                    ohlcv = exec_ohlcv.get(sid, {})
                    price = ohlcv.get("open") or buy["entry_price"]
                    alloc = buy["allocated_capital"]

                    # 漲停跳過（開盤即漲停 → 買不到）
                    if limit_price_check:
                        prev_c = prev_close_map.get(sid)
                        if prev_c:
                            is_up, _ = detect_limit_price(ohlcv.get("open", price), prev_c)
                            if is_up:
                                continue

                    if dynamic_slippage:
                        buy_slip = compute_dynamic_slippage(
                            ohlcv.get("volume", 0), ohlcv.get("high", 0), ohlcv.get("low", 0), price, side="buy"
                        )
                    else:
                        buy_slip = SLIPPAGE_RATE

                    shares = compute_shares(
                        alloc,
                        price,
                        slippage=buy_slip,
                        daily_volume=ohlcv.get("volume"),
                        participation_limit=liquidity_limit,
                    )
                    fill = simulate_buy(price, shares, buy_slip)
                    # 資金不足保護：T+1 開盤價可能高於決策日收盤，縮減股數避免現金為負
                    if fill.buy_cost > cash:
                        shares = compute_shares(
                            cash,
                            price,
                            slippage=buy_slip,
                            daily_volume=ohlcv.get("volume"),
                            participation_limit=liquidity_limit,
                        )
                        fill = simulate_buy(price, shares, buy_slip)
                    if shares <= 0:
                        continue

                    costs = fill.costs
                    cash -= fill.buy_cost

                    total_commission += costs.commission
                    total_slippage_cost += costs.slippage_cost
                    turnover_value += price * shares

                    positions.append(
                        {
                            "stock_id": sid,
                            "stock_name": buy.get("stock_name", ""),
                            "entry_date": exec_day,
                            "entry_price": price,
                            "entry_rank": buy["rank"],
                            "entry_score": buy.get("score"),
                            "shares": shares,
                            "allocated_capital": alloc,
                            "planned_exit_date": compute_planned_exit_date(exec_day, holding_days, trading_cal),
                            "stop_loss": buy.get("stop_loss"),
                            "buy_slippage": buy_slip,
                        }
                    )

            for day in trading_cal:
                # 取今日排名（持續沿用最近一次 scan 的結果）
                if day in all_rankings:
                    last_rankings = all_rankings[day]

                # 今日需報價標的：持倉 + 候選 + 待執行（pending）買入標的
                _pend_buy_sids = [b["stock_id"] for b in pending_exec[0].to_buy] if pending_exec else []
                all_sids = list(
                    {p["stock_id"] for p in positions} | {r["stock_id"] for r in last_rankings} | set(_pend_buy_sids)
                )
                if not all_sids:
                    equity_curve.append({"date": day, "equity": cash})
                    continue

                # 取今日 OHLCV（優先從預載入快取取得，cache miss 時 fallback 到 DB 查詢）
                today_ohlcv: dict[str, dict] = {}
                _cache_day = _ohlcv_cache.get(day, {})
                _cache_miss: list[str] = []
                for sid in all_sids:
                    if sid in _cache_day:
                        today_ohlcv[sid] = _cache_day[sid]
                    else:
                        _cache_miss.append(sid)
                if _cache_miss:
                    _fb = _get_ohlcv_on_date(session, _cache_miss, day)
                    today_ohlcv.update(_fb)
                today_prices = {sid: d["close"] for sid, d in today_ohlcv.items()}
                # P1-3 survivorship：持倉個股今日無報價（停牌/下市）→ 以最後已知收盤估值，
                # 避免 today_prices 缺值時 fallback 到 entry_price（凍結價=隱藏跌價=樂觀偏差）。
                # 使停損/到期賣出/MtM 皆以最後已知價計，跌價得以反映。
                for _p in positions:
                    if _p["stock_id"] not in today_prices and _p["stock_id"] in prev_close_map:
                        today_prices[_p["stock_id"]] = prev_close_map[_p["stock_id"]]

                # ── T+1 執行（audit P0-2）：前一決策日算出的 actions 於今日開盤成交 ──
                if pending_exec is not None:
                    _prev_actions, _prev_rankings = pending_exec
                    _execute_action_set(_prev_actions, _prev_rankings, today_ohlcv, day)
                    pending_exec = None

                # 無排名（首次 scan 前）：記錄權益後跳過決策
                if not last_rankings:
                    _eq = cash + sum(today_prices.get(p["stock_id"], p["entry_price"]) * p["shares"] for p in positions)
                    equity_curve.append({"date": day, "equity": _eq})
                    continue

                # 止損價：從持倉記錄取進場時鎖定的止損價
                # 研究旋鈕（B1 診斷：停損砍肥尾贏家）：disable_stop_loss=關閉、
                # stop_loss_widen>1=放寬停損與參考價距離（持倉用 entry、新候選用 close）。
                if disable_stop_loss:
                    stop_losses = {}
                else:
                    stop_losses = {}
                    for p in positions:
                        sl = p.get("stop_loss")
                        if sl is not None:
                            ref = p.get("entry_price")
                            stop_losses[p["stock_id"]] = (
                                sl if stop_loss_widen == 1.0 or ref is None else ref - (ref - sl) * stop_loss_widen
                            )
                    for r in last_rankings:
                        sl = r.get("stop_loss")
                        if r["stock_id"] not in stop_losses and sl is not None:
                            ref = r.get("close")
                            stop_losses[r["stock_id"]] = (
                                sl if stop_loss_widen == 1.0 or ref is None else ref - (ref - sl) * stop_loss_widen
                            )

                # ── 風控 overlay（P0-1 修復：與 live update() 對齊）──
                # Portfolio Heat 分母 = pre-action 權益（cash + 既有持倉 today MtM）
                pre_equity = cash + sum(
                    today_prices.get(p["stock_id"], p["entry_price"]) * p["shares"] for p in positions
                )
                # 逐日 regime（沿用最近 scan 的歷史 regime）
                if day in all_regime:
                    last_regime = all_regime[day]

                held_sids = [p["stock_id"] for p in positions]
                candidate_sids = [r["stock_id"] for r in last_rankings[:max_positions]]

                # 相關性矩陣（持倉 + 候選，60 日窗口；資料不足回 None → 不生效）
                corr_matrix = None
                corr_sids = list(set(held_sids + candidate_sids))
                if len(corr_sids) >= 2:
                    corr_price_data = {
                        sid: pd.Series(
                            [c for _, c in close_window_dated[sid]],
                            index=pd.DatetimeIndex([d for d, _ in close_window_dated[sid]]),
                        )
                        for sid in corr_sids
                        if len(close_window_dated.get(sid, [])) >= 2
                    }
                    if len(corr_price_data) >= 2:
                        corr_matrix = compute_correlation_matrix(corr_price_data, window=60)

                # 波動率反比權重（候選股，需 ≥20 日歷史）
                vol_weights = None
                vol_dict: dict[str, float] = {}
                for sid in candidate_sids:
                    hist = close_window_dated.get(sid, [])
                    if len(hist) >= 20:
                        daily_ret = pd.Series([c for _, c in hist]).pct_change().dropna()
                        if len(daily_ret) >= 10:
                            vol_dict[sid] = float(daily_ret.std() * (252**0.5))
                if vol_dict:
                    vol_weights = compute_vol_inverse_weights(vol_dict)

                # 成本閘門 — 取本 ISO 週 holding_expired 用量
                iso_y, iso_w, _ = day.isocalendar()
                if cost_cfg.enabled:
                    weekly_used_this_week = weekly_swap_counter.get((iso_y, iso_w), 0)
                    gate_min_hold = cost_cfg.min_hold_days
                    gate_score_gap = cost_cfg.score_gap_threshold
                    gate_weekly_cap = cost_cfg.weekly_swap_cap
                else:
                    weekly_used_this_week = 0
                    gate_min_hold = 0
                    gate_score_gap = 0.0
                    gate_weekly_cap = 0

                # P1-2：當前回撤（已實現權益序列 + 今日 pre-action 權益）供 Drawdown Guard 連續減倉
                _dd_pct = compute_portfolio_drawdown([e["equity"] for e in equity_curve] + [pre_equity])

                # 計算 rotation
                actions = compute_rotation_actions(
                    current_positions=positions,
                    new_rankings=last_rankings,
                    max_positions=max_positions,
                    holding_days=holding_days,
                    allow_renewal=allow_renewal,
                    today=day,
                    trading_calendar=trading_cal,
                    current_cash=cash,
                    stop_losses=stop_losses,
                    today_prices=today_prices,
                    # P0-1：風控 overlay 與 live update() 對齊
                    total_capital=pre_equity,
                    corr_matrix=corr_matrix,
                    vol_weights=vol_weights,
                    regime=last_regime,
                    # P1-2：Drawdown Guard 與 live update() 對齊
                    drawdown_pct=_dd_pct,
                    min_hold_days=gate_min_hold,
                    score_gap_threshold=gate_score_gap,
                    weekly_swap_cap=gate_weekly_cap,
                    weekly_swaps_used=weekly_used_this_week,
                )
                if cost_cfg.enabled and actions.holding_expired_sells > 0:
                    weekly_swap_counter[(iso_y, iso_w)] = weekly_used_this_week + actions.holding_expired_sells

                # ── T+1（audit P0-2）：不立即成交，暫存至下一交易日開盤由 _execute_action_set 執行 ──
                if t1_execution:
                    pending_exec = (actions, last_rankings)
                else:
                    # 研究旋鈕（A2 parity 報告）：同日收盤成交——把 close 充當 open
                    # 餵給同一套執行函數，重現舊 live 的 look-ahead 行為
                    _close_as_open = {sid: {**d, "open": d.get("close")} for sid, d in today_ohlcv.items()}
                    _execute_action_set(actions, last_rankings, _close_as_open, day)

                # 計算當日權益
                total_equity = cash + sum(
                    today_prices.get(p["stock_id"], p["entry_price"]) * p["shares"] for p in positions
                )

                # 累計每日收盤價（VaR 用，保留最近 65 日供 60 日窗口）
                for sid, ohlcv_data in today_ohlcv.items():
                    daily_close_history.setdefault(sid, []).append(ohlcv_data["close"])
                    if len(daily_close_history[sid]) > 65:
                        daily_close_history[sid] = daily_close_history[sid][-65:]
                    # 日期索引版本（次日 rotation 的 corr/vol overlay 用，含當日收盤、無 look-ahead）
                    close_window_dated.setdefault(sid, []).append((day, ohlcv_data["close"]))
                    if len(close_window_dated[sid]) > 65:
                        close_window_dated[sid] = close_window_dated[sid][-65:]

                # Ex-Ante VaR（有持倉時計算）
                var_pct = None
                if positions and total_equity > 0:
                    held_sids = [p["stock_id"] for p in positions]
                    price_series = {
                        sid: pd.Series(daily_close_history[sid])
                        for sid in held_sids
                        if sid in daily_close_history and len(daily_close_history[sid]) >= 20
                    }
                    if price_series:
                        cov_mat = compute_covariance_matrix(price_series, window=60, min_periods=20)
                        if not cov_mat.empty:
                            pos_weights = {}
                            for p in positions:
                                mv = today_prices.get(p["stock_id"], p["entry_price"]) * p["shares"]
                                pos_weights[p["stock_id"]] = mv / total_equity
                            var_result = compute_portfolio_var(pos_weights, cov_mat, total_equity)
                            var_pct = var_result["var_pct"]

                equity_curve.append({"date": day, "equity": total_equity, "var_pct": var_pct})

                # 記錄持倉快照
                for pos in positions:
                    _sid = pos["stock_id"]
                    _cur_p = today_prices.get(_sid, pos["entry_price"])
                    _mv = _cur_p * pos["shares"]
                    daily_positions_snapshot.append(
                        {
                            "date": day,
                            "stock_id": _sid,
                            "stock_name": pos.get("stock_name", ""),
                            "shares": pos["shares"],
                            "entry_price": pos["entry_price"],
                            "current_price": _cur_p,
                            "market_value": round(_mv, 2),
                            "unrealized_pct": round((_cur_p - pos["entry_price"]) / pos["entry_price"], 6)
                            if pos["entry_price"] > 0
                            else 0.0,
                            "weight": round(_mv / total_equity, 6) if total_equity > 0 else 0.0,
                        }
                    )

                # 更新 prev_close_map（漲跌停偵測用）
                for sid, ohlcv_data in today_ohlcv.items():
                    prev_close_map[sid] = ohlcv_data["close"]

            # 強制平倉期末持倉
            if positions and trading_cal:
                last_day = trading_cal[-1]
                _last_sids = [p["stock_id"] for p in positions]
                _cache_last = _ohlcv_cache.get(last_day, {})
                last_ohlcv = {sid: _cache_last[sid] for sid in _last_sids if sid in _cache_last}
                _last_miss = [sid for sid in _last_sids if sid not in last_ohlcv]
                if _last_miss:
                    last_ohlcv.update(_get_ohlcv_on_date(session, _last_miss, last_day))
                for pos in positions:
                    sid = pos["stock_id"]
                    ohlcv = last_ohlcv.get(sid, {})
                    # P1-3 survivorship：期末無報價＝停牌/下市仍持有 → 以最後已知價平倉並計數，
                    # 不再 fallback entry_price（凍結價會把下市虧損藏起來，導致回測樂觀）
                    is_stranded = "close" not in ohlcv
                    if is_stranded:
                        n_stranded += 1
                        exit_p = prev_close_map.get(sid, pos["entry_price"])
                    else:
                        exit_p = ohlcv["close"]
                    exit_reason = "delisted_stranded" if is_stranded else "backtest_end"

                    if dynamic_slippage:
                        sell_slip = compute_dynamic_slippage(
                            ohlcv.get("volume", 0),
                            ohlcv.get("high", 0),
                            ohlcv.get("low", 0),
                            exit_p,
                            side="sell",
                        )
                    else:
                        sell_slip = SLIPPAGE_RATE

                    buy_slip = pos.get("buy_slippage", SLIPPAGE_RATE)
                    fill = simulate_sell(
                        pos["entry_price"], exit_p, pos["shares"], buy_slippage=buy_slip, sell_slippage=sell_slip
                    )
                    pnl, return_pct = fill.pnl, fill.return_pct
                    costs = fill.costs
                    cash += fill.proceeds

                    total_commission += costs.commission
                    total_tax += costs.tax
                    total_slippage_cost += costs.slippage_cost
                    turnover_value += exit_p * pos["shares"]

                    all_trades.append(
                        {
                            "stock_id": sid,
                            "entry_date": pos["entry_date"],
                            "entry_price": pos["entry_price"],
                            "exit_date": last_day,
                            "exit_price": exit_p,
                            "shares": pos["shares"],
                            "pnl": pnl,
                            "return_pct": return_pct,
                            "exit_reason": exit_reason,
                            "entry_rank": pos.get("entry_rank"),
                            "entry_score": pos.get("entry_score"),
                            "buy_slippage": buy_slip,
                            "sell_slippage": sell_slip,
                            "commission": costs.commission,
                            "tax": costs.tax,
                            "slippage_cost": costs.slippage_cost,
                        }
                    )

            # ── 計算績效指標（委託 compute_metrics）──
            from src.backtest.metrics import compute_metrics as _compute_full_metrics

            equities = [e["equity"] for e in equity_curve]
            trade_adapters = [_TradeAdapter(pnl=t["pnl"]) for t in all_trades]
            raw_metrics = _compute_full_metrics(equities, trade_adapters, start_date, end_date, capital)

            # 交易統計（保留既有欄位格式）
            trade_returns = [t["return_pct"] for t in all_trades if t.get("return_pct") is not None]
            wins = [r for r in trade_returns if r > 0]
            losses = [r for r in trade_returns if r <= 0]

            # TAIEX benchmark（含交易成本）
            benchmark_return = None
            if taiex_prices and len(trading_cal) >= 2:
                first_t = taiex_prices.get(trading_cal[0])
                last_t = taiex_prices.get(trading_cal[-1])
                if first_t and last_t and first_t > 0:
                    buy_cost_bm = first_t * (1 + COMMISSION_RATE)
                    sell_net_bm = last_t * (1 - COMMISSION_RATE - TAX_RATE)
                    benchmark_return = round((sell_net_bm / buy_cost_bm - 1) * 100, 2)

            # 轉換為 rotation 既有格式（小數制），維持 CLI 向後相容
            metrics = {
                "total_return": round(raw_metrics["total_return"] / 100, 4),
                "annual_return": round(raw_metrics["annual_return"] / 100, 4),
                "max_drawdown": round(raw_metrics["max_drawdown"] / 100, 4),
                "sharpe_ratio": raw_metrics["sharpe_ratio"] or 0.0,
                "sortino_ratio": raw_metrics.get("sortino_ratio"),
                "calmar_ratio": raw_metrics.get("calmar_ratio"),
                "var_95": raw_metrics.get("var_95"),
                "cvar_95": raw_metrics.get("cvar_95"),
                "profit_factor": raw_metrics.get("profit_factor"),
                # 保留既有欄位
                "total_trades": len(all_trades),
                "win_rate": round(raw_metrics["win_rate"] / 100, 4) if raw_metrics.get("win_rate") else 0,
                "avg_return_per_trade": round(sum(trade_returns) / len(trade_returns), 4) if trade_returns else 0,
                "avg_win": round(sum(wins) / len(wins), 4) if wins else 0,
                "avg_loss": round(sum(losses) / len(losses), 4) if losses else 0,
                "final_capital": round(equities[-1], 2) if equities else capital,
                "trading_days": len(equity_curve),
                # TAIEX Benchmark
                "benchmark_return": benchmark_return,
                # P1-3 survivorship：期末仍持有但已停牌/下市的倉位數 + 警告旗標
                "survivorship_stranded": n_stranded,
                "survivorship_warning": n_stranded > 0,
            }
            if n_stranded > 0:
                logger.warning(
                    "[%s] survivorship：%d 個期末持倉個股已停牌/下市，以最後已知價平倉；"
                    "歷史回測若含下市股偏樂觀，請留意",
                    self.portfolio_name,
                    n_stranded,
                )
            # 成本歸因 + 拆解（合併進 metrics）
            metrics.update(
                compute_cost_metrics(
                    commission=total_commission,
                    tax=total_tax,
                    slippage=total_slippage_cost,
                    turnover_value=turnover_value,
                    capital=capital,
                )
            )

            result = RotationBacktestResult(
                equity_curve=equity_curve,
                trades=all_trades,
                metrics=metrics,
                daily_positions=daily_positions_snapshot,
                config={
                    "portfolio_name": self.portfolio_name,
                    "mode": mode,
                    "max_positions": max_positions,
                    "holding_days": holding_days,
                    "capital": capital,
                    "allow_renewal": allow_renewal,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )

            # 寫入 DB（研究重跑可用 save_result=False 跳過，避免污染回測紀錄）
            if save_result:
                from src.data.pipeline import save_rotation_backtest

                save_rotation_backtest(result)

            return result

    # ── 查詢 ──

    def get_status(self) -> dict | None:
        """回傳目前組合狀態 + 持倉明細。"""
        with get_session() as session:
            portfolio = self._load_portfolio(session)
            if portfolio is None:
                return None

            open_positions = self._load_open_positions(session, portfolio.id)
            # 取最新收盤價
            sids = [p["stock_id"] for p in open_positions]
            latest_prices = {}
            if sids:
                # 取最近交易日的價格
                for sid in sids:
                    stmt = (
                        select(DailyPrice.close, DailyPrice.date)
                        .where(DailyPrice.stock_id == sid)
                        .order_by(DailyPrice.date.desc())
                        .limit(1)
                    )
                    row = session.execute(stmt).first()
                    if row:
                        latest_prices[sid] = row[0]

            holdings = []
            total_market_value = 0.0
            total_unrealized_pnl = 0.0
            for pos in open_positions:
                sid = pos["stock_id"]
                current_price = latest_prices.get(sid, pos["entry_price"])
                market_val = current_price * pos["shares"]
                unrealized_pnl, unrealized_pct = compute_position_pnl(pos["entry_price"], current_price, pos["shares"])
                holdings.append(
                    {
                        **pos,
                        "current_price": current_price,
                        "market_value": market_val,
                        "unrealized_pnl": unrealized_pnl,
                        "unrealized_pct": unrealized_pct,
                    }
                )
                total_market_value += market_val
                total_unrealized_pnl += unrealized_pnl

            return {
                "name": portfolio.name,
                "mode": portfolio.mode,
                "max_positions": portfolio.max_positions,
                "holding_days": portfolio.holding_days,
                "allow_renewal": portfolio.allow_renewal,
                "initial_capital": portfolio.initial_capital,
                "current_capital": portfolio.current_capital,
                "current_cash": portfolio.current_cash,
                "status": portfolio.status,
                "holdings": holdings,
                "total_market_value": total_market_value,
                "total_unrealized_pnl": total_unrealized_pnl,
                "total_return_pct": (portfolio.current_capital - portfolio.initial_capital) / portfolio.initial_capital
                if portfolio.initial_capital > 0
                else 0.0,
                "updated_at": str(portfolio.updated_at) if portfolio.updated_at else None,
            }

    def get_history(self, limit: int = 50) -> pd.DataFrame:
        """回傳已平倉交易記錄。"""
        with get_session() as session:
            portfolio = self._load_portfolio(session)
            if portfolio is None:
                return pd.DataFrame()

            stmt = (
                select(RotationPosition)
                .where(
                    RotationPosition.portfolio_id == portfolio.id,
                    RotationPosition.status == "closed",
                )
                .order_by(RotationPosition.exit_date.desc())
                .limit(limit)
            )
            positions = session.execute(stmt).scalars().all()
            if not positions:
                return pd.DataFrame()

            return pd.DataFrame(
                [
                    {
                        "stock_id": p.stock_id,
                        "stock_name": p.stock_name,
                        "entry_date": p.entry_date,
                        "entry_price": p.entry_price,
                        "exit_date": p.exit_date,
                        "exit_price": p.exit_price,
                        "shares": p.shares,
                        "pnl": p.pnl,
                        "return_pct": p.return_pct,
                        "exit_reason": p.exit_reason,
                        "holding_days_count": p.holding_days_count,
                    }
                    for p in positions
                ]
            )

    # ── 管理 ──

    def pause(self) -> bool:
        with get_session() as session:
            portfolio = self._load_portfolio(session)
            if portfolio is None:
                return False
            portfolio.status = "paused"
            session.commit()
            return True

    def resume(self) -> bool:
        with get_session() as session:
            portfolio = self._load_portfolio(session)
            if portfolio is None:
                return False
            portfolio.status = "active"
            session.commit()
            return True

    def delete(self) -> bool:
        with get_session() as session:
            portfolio = self._load_portfolio(session)
            if portfolio is None:
                return False
            # 刪除所有持倉
            stmt = select(RotationPosition).where(RotationPosition.portfolio_id == portfolio.id)
            for pos in session.execute(stmt).scalars().all():
                session.delete(pos)
            session.delete(portfolio)
            session.commit()
            return True

    @staticmethod
    def list_portfolios() -> list[dict]:
        """列出所有輪動組合。"""
        with get_session() as session:
            stmt = select(RotationPortfolio).order_by(RotationPortfolio.created_at)
            portfolios = session.execute(stmt).scalars().all()
            return [
                {
                    "name": p.name,
                    "mode": p.mode,
                    "max_positions": p.max_positions,
                    "holding_days": p.holding_days,
                    "allow_renewal": p.allow_renewal,
                    "initial_capital": p.initial_capital,
                    "current_capital": p.current_capital,
                    "status": p.status,
                }
                for p in portfolios
            ]

    # ── 內部方法 ──

    def _load_portfolio(self, session) -> RotationPortfolio | None:
        stmt = select(RotationPortfolio).where(RotationPortfolio.name == self.portfolio_name)
        return session.execute(stmt).scalar_one_or_none()

    def _compute_equity_history(
        self,
        session,
        portfolio: RotationPortfolio,
        open_positions: list[dict] | None = None,
        today_prices: dict[str, float] | None = None,
    ) -> list[float]:
        """從 DB 載入已平倉 pnl + 計算最新淨值，委派 build_equity_history 組裝序列。

        S4 重構（2026-05-09 audit）：原 in-line 邏輯改委派至 rotation.build_equity_history
        純函數，manager 此處僅負責 IO（DB 查詢 + MtM 計算）。

        C1 修復語意保留：
          - open_positions + today_prices 提供時 → final_equity = cash + Σ(MtM)，
            反映當日浮動損益（避免 gap-down 情境 kill switch 不觸發）
          - 否則 fallback portfolio.current_capital（向後相容）
        """
        from src.portfolio.rotation import build_equity_history

        stmt = (
            select(RotationPosition)
            .where(
                RotationPosition.portfolio_id == portfolio.id,
                RotationPosition.status != "open",
                RotationPosition.exit_date.isnot(None),
            )
            .order_by(RotationPosition.exit_date)
        )
        closed = session.execute(stmt).scalars().all()
        closed_pnls = [pos.pnl or 0.0 for pos in closed]

        # 計算 final_equity（C1 修復：含當日 MtM）
        if open_positions is not None and today_prices is not None:
            mtm = sum(today_prices.get(p["stock_id"], p["entry_price"]) * p["shares"] for p in open_positions)
            final_equity = portfolio.current_cash + mtm
        else:
            # Fallback：滯後 1 輪，僅供無價格資訊的場景
            final_equity = portfolio.current_capital

        return build_equity_history(portfolio.initial_capital, closed_pnls, final_equity)

    def _load_snapshot_capitals(self, session) -> list[float]:
        """載入該組合全部每日 snapshot 的 MtM 權益序列（依日期升序）。

        P0 止血包 #3（2026-07-05）：供 compute_drawdown_with_snapshots 補回
        realized-only equity_history 缺失的浮盈高點（peak 低估 → 熔斷不觸發）。
        """
        stmt = (
            select(RotationDailySnapshot.total_capital)
            .where(RotationDailySnapshot.portfolio_name == self.portfolio_name)
            .order_by(RotationDailySnapshot.snapshot_date)
        )
        return [v for v in session.execute(stmt).scalars().all() if v is not None]

    def _load_open_positions(self, session, portfolio_id: int) -> list[dict]:
        stmt = select(RotationPosition).where(
            RotationPosition.portfolio_id == portfolio_id,
            RotationPosition.status == "open",
        )
        positions = session.execute(stmt).scalars().all()
        return [
            {
                "stock_id": p.stock_id,
                "stock_name": p.stock_name,
                "entry_date": p.entry_date,
                "entry_price": p.entry_price,
                "entry_rank": p.entry_rank,
                # P1-1：載入 entry_score，使成本閘門 B（score_gap_threshold）在 live 生效
                # （過去缺此欄 → Gate B 的 pos.get("entry_score") 永遠 None → 閘門失效）
                "entry_score": p.entry_score,
                "shares": p.shares,
                "allocated_capital": p.allocated_capital,
                "planned_exit_date": p.planned_exit_date,
                "stop_loss": p.stop_loss,
                "entry_score_breakdown_json": p.entry_score_breakdown_json,
            }
            for p in positions
        ]

    def _find_latest_rankings(self, session, mode: str, before_date: date) -> list[dict]:
        """找最近的 scan_date 排名（當日無 discover 結果時使用）。"""
        if is_composite_mode(mode):
            stmt = (
                select(DiscoveryRecord.scan_date)
                .where(DiscoveryRecord.scan_date < before_date)
                .order_by(DiscoveryRecord.scan_date.desc())
                .limit(1)
            )
        else:
            stmt = (
                select(DiscoveryRecord.scan_date)
                .where(DiscoveryRecord.scan_date < before_date, DiscoveryRecord.mode == mode)
                .order_by(DiscoveryRecord.scan_date.desc())
                .limit(1)
            )
        row = session.execute(stmt).first()
        if row:
            return resolve_rankings(mode, row[0], session)
        return []

    def _execute_sell(
        self, session, portfolio_id: int, sell: dict, today: date, cash: float, slippage: float = SLIPPAGE_RATE
    ) -> float:
        """執行賣出：更新 RotationPosition 狀態並回收資金。

        P1-4：slippage 由 caller 以 compute_dynamic_slippage 計算（fallback SLIPPAGE_RATE），
        用於成本/淨收入/已實現 pnl，並寫入 pos.sell_slippage（與 backtest 對齊）。
        """
        stmt = select(RotationPosition).where(
            RotationPosition.portfolio_id == portfolio_id,
            RotationPosition.stock_id == sell["stock_id"],
            RotationPosition.status == "open",
        )
        pos = session.execute(stmt).scalar_one_or_none()
        if pos is None:
            return cash

        exit_price = sell.get("exit_price", pos.entry_price)
        fill = simulate_sell(
            pos.entry_price,
            exit_price,
            pos.shares,
            buy_slippage=pos.buy_slippage if pos.buy_slippage is not None else SLIPPAGE_RATE,
            sell_slippage=slippage,
        )

        pos.exit_date = today
        pos.exit_price = exit_price
        pos.exit_reason = sell["reason"]
        pos.pnl = fill.pnl
        pos.return_pct = fill.return_pct
        pos.status = "closed"
        pos.holding_days_count = sell.get("days_held", 0)
        # 滑價/成本 instrumentation：sell 端記滑價率 + 累加 trade_cost（買端已寫入）
        pos.sell_slippage = slippage
        pos.trade_cost = round((pos.trade_cost or 0.0) + fill.costs.total, 2)

        return cash + fill.proceeds

    def _execute_renewal(self, session, portfolio_id: int, renew: dict, new_stop_loss: float | None = None) -> None:
        """執行續持：更新 planned_exit_date，止損價只能上移（trailing stop）。"""
        stmt = select(RotationPosition).where(
            RotationPosition.portfolio_id == portfolio_id,
            RotationPosition.stock_id == renew["stock_id"],
            RotationPosition.status == "open",
        )
        pos = session.execute(stmt).scalar_one_or_none()
        if pos is not None:
            pos.planned_exit_date = renew["new_planned_exit_date"]
            pos.holding_days_count = renew.get("days_held", pos.holding_days_count)
            # 續持時止損價只能上移（trailing stop），不能下移
            if new_stop_loss is not None:
                if pos.stop_loss is None or new_stop_loss > pos.stop_loss:
                    pos.stop_loss = new_stop_loss

    def _execute_buy(
        self,
        session,
        portfolio_id: int,
        buy: dict,
        today: date,
        trading_cal: list[date],
        cash: float,
        slippage: float = SLIPPAGE_RATE,
        daily_volume: float | None = None,
    ) -> float:
        """執行買入：建立 RotationPosition 並扣減現金。

        P1-4：slippage 由 caller 以 compute_dynamic_slippage 計算；daily_volume 提供時
        以 apply_liquidity_limit 限制單筆量（≤ 當日量 × LIQUIDITY_PARTICIPATION_LIMIT），
        並寫入 pos.buy_slippage（與 backtest 對齊）。
        """
        from src.portfolio.rotation import apply_liquidity_limit

        price = buy["entry_price"]
        shares = buy["shares"]
        # 流動性限制（只下調，不放大 compute_rotation_actions 已算好的 heat/corr 調整後股數）
        if daily_volume:
            shares = apply_liquidity_limit(shares, daily_volume, LIQUIDITY_PARTICIPATION_LIMIT)
        if shares <= 0:
            return cash

        fill = simulate_buy(price, shares, slippage)
        if fill.buy_cost > cash:
            # 資金不足，縮減股數
            shares = compute_shares(cash, price, slippage=slippage, daily_volume=daily_volume)
            if shares <= 0:
                return cash
            fill = simulate_buy(price, shares, slippage)

        from src.portfolio.rotation import compute_planned_exit_date

        # 需要 portfolio 的 holding_days
        portfolio = session.execute(select(RotationPortfolio).where(RotationPortfolio.id == portfolio_id)).scalar_one()
        planned_exit = compute_planned_exit_date(today, portfolio.holding_days, trading_cal)

        # P1 任務 5：序列化進場理由 JSON（debug 為何進這檔，audit 用）
        breakdown_json: str | None = None
        breakdown = buy.get("score_breakdown")
        if breakdown:
            import json

            try:
                breakdown_json = json.dumps(breakdown, ensure_ascii=False, default=str)
            except (TypeError, ValueError) as exc:
                logger.warning("[%s] entry_score_breakdown_json 序列化失敗 (%s)，寫入 NULL", buy["stock_id"], exc)
                breakdown_json = None

        pos = RotationPosition(
            portfolio_id=portfolio_id,
            stock_id=buy["stock_id"],
            stock_name=buy.get("stock_name", ""),
            entry_date=today,
            entry_price=price,
            entry_rank=buy["rank"],
            entry_score=buy.get("score"),
            holding_days_count=0,
            planned_exit_date=planned_exit,
            shares=shares,
            allocated_capital=buy.get("allocated_capital", fill.buy_cost),
            stop_loss=buy.get("stop_loss"),
            status="open",
            # 滑價/成本 instrumentation：buy 端記滑價率 + 初始化 trade_cost（賣出時累加）
            buy_slippage=slippage,
            trade_cost=round(fill.costs.total, 2),
            entry_score_breakdown_json=breakdown_json,
        )
        session.add(pos)
        return cash - fill.buy_cost

    @staticmethod
    def _compute_backtest_metrics(
        equity_curve: list[dict],
        trades: list[dict],
        initial_capital: float,
    ) -> dict:
        """從權益曲線與交易記錄計算績效指標。"""
        if not equity_curve:
            return {}

        equities = [e["equity"] for e in equity_curve]
        final = equities[-1]
        total_return = (final - initial_capital) / initial_capital if initial_capital > 0 else 0

        # 最大回撤
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # 交易統計
        trade_returns = [t["return_pct"] for t in trades if t.get("return_pct") is not None]
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        win_rate = len(wins) / len(trade_returns) if trade_returns else 0
        avg_return = sum(trade_returns) / len(trade_returns) if trade_returns else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        # 年化報酬
        n_days = len(equity_curve)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

        # 日報酬率序列 → Sharpe
        daily_returns = []
        for i in range(1, len(equities)):
            dr = (equities[i] - equities[i - 1]) / equities[i - 1] if equities[i - 1] > 0 else 0
            daily_returns.append(dr)

        import statistics

        sharpe = 0.0
        if daily_returns and len(daily_returns) > 1:
            mean_r = statistics.mean(daily_returns)
            std_r = statistics.stdev(daily_returns)
            if std_r > 0:
                sharpe = (mean_r / std_r) * (252**0.5)

        return {
            "total_return": round(total_return, 4),
            "annual_return": round(annual_return, 4),
            "max_drawdown": round(max_dd, 4),
            "sharpe_ratio": round(sharpe, 4),
            "total_trades": len(trades),
            "win_rate": round(win_rate, 4),
            "avg_return_per_trade": round(avg_return, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "final_capital": round(final, 2),
            "trading_days": n_days,
        }
