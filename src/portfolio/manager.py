"""RotationManager — 輪動組合的 DB 調度與回測引擎。

負責讀寫 RotationPortfolio / RotationPosition ORM，
呼叫 rotation.py 純函數計算買賣動作，並支援歷史回測。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
from sqlalchemy import select

from src.config import settings
from src.constants import (
    COMMISSION_RATE,
    LIQUIDITY_PARTICIPATION_LIMIT,
    REGIME_FALLBACK_DEFAULT,
    SLIPPAGE_RATE,
    TAX_RATE,
)
from src.data.database import get_session
from src.data.schema import (
    DailyPrice,
    DiscoveryRecord,
    RotationPortfolio,
    RotationPosition,
)
from src.portfolio.rotation import (
    RotationActions,
    check_drawdown_kill_switch,
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_dynamic_slippage,
    compute_planned_exit_date,
    compute_portfolio_drawdown,
    compute_portfolio_var,
    compute_position_pnl,
    compute_rotation_actions,
    compute_shares,
    compute_trade_costs,
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


# ---------------------------------------------------------------------------
# 排名解析
# ---------------------------------------------------------------------------


def resolve_rankings(
    mode: str,
    scan_date: date,
    session,
    top_n: int = 50,
) -> list[dict]:
    """從 DiscoveryRecord 解析指定日期的排名。

    Parameters
    ----------
    mode : str
        'momentum'/'swing'/.../'all'。
    scan_date : date
        掃描日期。
    session : SQLAlchemy Session
    top_n : int
        最大取用筆數。

    Returns
    -------
    list[dict]
        按排名排序的清單，每筆含：
        {stock_id, stock_name, rank, score, close, stop_loss}
    """
    if mode == "all":
        return _resolve_all_mode_rankings(scan_date, session, top_n)

    stmt = (
        select(DiscoveryRecord)
        .where(DiscoveryRecord.scan_date == scan_date, DiscoveryRecord.mode == mode)
        .order_by(DiscoveryRecord.rank)
        .limit(top_n)
    )
    records = session.execute(stmt).scalars().all()
    return [
        {
            "stock_id": r.stock_id,
            "stock_name": r.stock_name or "",
            "rank": r.rank,
            "score": r.composite_score,
            "close": r.close,
            "stop_loss": r.stop_loss,
        }
        for r in records
    ]


def _resolve_all_mode_rankings(
    scan_date: date,
    session,
    top_n: int,
) -> list[dict]:
    """解析 'all' 模式排名 — 所有模式取 avg_score 排序。"""
    stmt = select(DiscoveryRecord).where(DiscoveryRecord.scan_date == scan_date)
    records = session.execute(stmt).scalars().all()

    # 按 stock_id 分組，計算 avg_score
    stock_data: dict[str, dict] = {}
    for r in records:
        sid = r.stock_id
        if sid not in stock_data:
            stock_data[sid] = {
                "stock_id": sid,
                "stock_name": r.stock_name or "",
                "close": r.close,
                "stop_loss": r.stop_loss,
                "scores": [],
            }
        stock_data[sid]["scores"].append(r.composite_score)
        # 保留最嚴格的 stop_loss（最高的）
        existing_sl = stock_data[sid]["stop_loss"]
        if r.stop_loss is not None:
            if existing_sl is None or r.stop_loss > existing_sl:
                stock_data[sid]["stop_loss"] = r.stop_loss

    # 計算 avg_score 並排序
    ranked = []
    for sid, data in stock_data.items():
        avg_score = sum(data["scores"]) / len(data["scores"])
        ranked.append(
            {
                "stock_id": sid,
                "stock_name": data["stock_name"],
                "score": avg_score,
                "close": data["close"],
                "stop_loss": data["stop_loss"],
            }
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)

    # 加上排名
    for i, r in enumerate(ranked[:top_n], 1):
        r["rank"] = i

    return ranked[:top_n]


# ---------------------------------------------------------------------------
# 交易日曆
# ---------------------------------------------------------------------------


def _get_trading_calendar(session, start: date, end: date) -> list[date]:
    """從 DailyPrice (TAIEX) 取交易日曆。"""
    stmt = (
        select(DailyPrice.date)
        .where(
            DailyPrice.stock_id == "TAIEX",
            DailyPrice.date >= start,
            DailyPrice.date <= end,
        )
        .order_by(DailyPrice.date)
    )
    dates = [row[0] for row in session.execute(stmt).all()]
    if not dates:
        # fallback: 工作日
        from datetime import timedelta

        d = start
        while d <= end:
            if d.weekday() < 5:
                dates.append(d)
            d += timedelta(days=1)
    return dates


def _get_prices_on_date(session, stock_ids: list[str], target_date: date) -> dict[str, float]:
    """取得指定日期（或最近交易日）的收盤價。

    先嘗試精確比對 target_date，若某些股票找不到資料，
    則 fallback 取最近 5 個交易日內的最新收盤價。
    """
    if not stock_ids:
        return {}

    # 精確比對
    stmt = select(DailyPrice.stock_id, DailyPrice.close).where(
        DailyPrice.stock_id.in_(stock_ids),
        DailyPrice.date == target_date,
    )
    result = {row[0]: row[1] for row in session.execute(stmt).all()}

    # Fallback：找不到精確日期的股票，取最近 5 天內最新收盤價
    missing = [sid for sid in stock_ids if sid not in result]
    if missing:
        from datetime import timedelta

        fallback_start = target_date - timedelta(days=5)
        from sqlalchemy import func

        sub = (
            select(
                DailyPrice.stock_id,
                func.max(DailyPrice.date).label("max_date"),
            )
            .where(
                DailyPrice.stock_id.in_(missing),
                DailyPrice.date >= fallback_start,
                DailyPrice.date <= target_date,
            )
            .group_by(DailyPrice.stock_id)
            .subquery()
        )
        stmt2 = select(DailyPrice.stock_id, DailyPrice.close).join(
            sub,
            (DailyPrice.stock_id == sub.c.stock_id) & (DailyPrice.date == sub.c.max_date),
        )
        for row in session.execute(stmt2).all():
            result[row[0]] = row[1]

    return result


def _get_ohlcv_on_date(session, stock_ids: list[str], target_date: date) -> dict[str, dict]:
    """取得指定日期（或最近交易日）的 OHLCV 完整資料。

    先嘗試精確比對 target_date，若某些股票找不到資料，
    則 fallback 取最近 5 天內的最新資料（fallback 時 volume 設為 0 以避免錯誤流動性估算）。

    Returns
    -------
    dict[str, dict]
        {stock_id: {"open": .., "high": .., "low": .., "close": .., "volume": ..}}
    """
    if not stock_ids:
        return {}

    # 精確比對
    stmt = select(
        DailyPrice.stock_id,
        DailyPrice.open,
        DailyPrice.high,
        DailyPrice.low,
        DailyPrice.close,
        DailyPrice.volume,
    ).where(
        DailyPrice.stock_id.in_(stock_ids),
        DailyPrice.date == target_date,
    )
    result: dict[str, dict] = {}
    for row in session.execute(stmt).all():
        result[row[0]] = {
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "volume": row[5] or 0,
        }

    # Fallback：最近 5 天（volume 歸零，避免用非當日量做流動性約束）
    missing = [sid for sid in stock_ids if sid not in result]
    if missing:
        from datetime import timedelta

        from sqlalchemy import func

        fallback_start = target_date - timedelta(days=5)
        sub = (
            select(
                DailyPrice.stock_id,
                func.max(DailyPrice.date).label("max_date"),
            )
            .where(
                DailyPrice.stock_id.in_(missing),
                DailyPrice.date >= fallback_start,
                DailyPrice.date <= target_date,
            )
            .group_by(DailyPrice.stock_id)
            .subquery()
        )
        stmt2 = select(
            DailyPrice.stock_id,
            DailyPrice.open,
            DailyPrice.high,
            DailyPrice.low,
            DailyPrice.close,
        ).join(
            sub,
            (DailyPrice.stock_id == sub.c.stock_id) & (DailyPrice.date == sub.c.max_date),
        )
        for row in session.execute(stmt2).all():
            result[row[0]] = {
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": 0,  # fallback 資料不代表今日流動性
            }

    return result


def _get_taiex_prices(session, start: date, end: date) -> dict[date, float]:
    """取得 TAIEX 收盤價序列，用於 benchmark 計算。"""
    stmt = select(DailyPrice.date, DailyPrice.close).where(
        DailyPrice.stock_id == "TAIEX",
        DailyPrice.date >= start,
        DailyPrice.date <= end,
    )
    return {row[0]: row[1] for row in session.execute(stmt).all()}


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

    def update(self, today: date | None = None, regime: str | None = None) -> RotationActions | None:
        """每日更新：讀取 discover 排名 → 計算 rotation → 寫入 DB。

        Parameters
        ----------
        today : date | None
            今日日期，預設 date.today()。
        regime : str | None
            目前市場狀態（bull/sideways/bear/crisis），用於 Crisis 硬阻擋。
            None 時嘗試從 RegimeStateMachine JSON 持久化讀取。
        """
        if today is None:
            today = date.today()

        with get_session() as session:
            portfolio = self._load_portfolio(session)
            if portfolio is None:
                logger.warning("找不到組合: %s", self.portfolio_name)
                return None
            if portfolio.status != "active":
                logger.info("組合 %s 已暫停，跳過更新", self.portfolio_name)
                return None

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
                and portfolio.mode != "all"
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

            # 止損價：從持倉記錄取進場時鎖定的止損價（不隨 discover 每日浮動）
            stop_losses = {p["stock_id"]: p["stop_loss"] for p in open_positions if p.get("stop_loss") is not None}
            # 新買入候選的止損價從 rankings 取（進場後會寫入 position）
            for r in rankings:
                if r["stock_id"] not in stop_losses and r.get("stop_loss") is not None:
                    stop_losses[r["stock_id"]] = r["stop_loss"]

            # ── Regime fallback：嘗試從 JSON 持久化讀取 ──
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
                from datetime import timedelta

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

            # ── Max Drawdown Kill Switch：回撤超過閾值強制平倉 ──
            equity_history = self._compute_equity_history(session, portfolio)
            if check_drawdown_kill_switch(equity_history):
                dd_pct = compute_portfolio_drawdown(equity_history)
                logger.error(
                    "⚠️ [%s] 回撤熔斷觸發 (%.1f%%)！強制平倉所有持倉",
                    self.portfolio_name,
                    dd_pct,
                )
                cash = portfolio.current_cash
                for pos in open_positions:
                    sell_action = {
                        "stock_id": pos["stock_id"],
                        "reason": "max_drawdown_liquidation",
                        "exit_price": today_prices.get(pos["stock_id"], pos["entry_price"]),
                        "days_held": 0,
                        **pos,
                    }
                    cash = self._execute_sell(session, portfolio.id, sell_action, today, cash)
                portfolio.current_cash = cash
                portfolio.current_capital = cash
                portfolio.status = "liquidated"
                portfolio.updated_at = datetime.utcnow()
                session.commit()
                logger.error("[%s] 已強制平倉，組合狀態設為 liquidated", self.portfolio_name)
                return RotationActions(to_sell=[{**p, "reason": "max_drawdown_liquidation"} for p in open_positions])

            # ── Rotation 成本閘門（A/B/C）參數 ──
            cost_cfg = settings.quant.rotation_cost
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

            # 計算 rotation
            actions = compute_rotation_actions(
                current_positions=open_positions,
                new_rankings=rankings,
                max_positions=portfolio.max_positions,
                holding_days=portfolio.holding_days,
                allow_renewal=portfolio.allow_renewal,
                today=today,
                trading_calendar=trading_cal,
                current_cash=portfolio.current_cash,
                stop_losses=stop_losses,
                today_prices=today_prices,
                total_capital=portfolio.current_capital,
                corr_matrix=corr_matrix,
                vol_weights=vol_weights,
                regime=regime,
                min_hold_days=gate_min_hold,
                score_gap_threshold=gate_score_gap,
                weekly_swap_cap=gate_weekly_cap,
                weekly_swaps_used=weekly_swaps_used,
            )

            # 執行賣出
            cash = portfolio.current_cash
            for sell in actions.to_sell:
                cash = self._execute_sell(session, portfolio.id, sell, today, cash)

            # 執行續持（從 rankings 取最新止損價，只上移不下移）
            ranking_sl = {r["stock_id"]: r["stop_loss"] for r in rankings if r.get("stop_loss") is not None}
            for renew in actions.renewed:
                new_sl = ranking_sl.get(renew["stock_id"])
                self._execute_renewal(session, portfolio.id, renew, new_stop_loss=new_sl)

            # 執行買入
            for buy in actions.to_buy:
                cash = self._execute_buy(session, portfolio.id, buy, today, trading_cal, cash)

            # 更新組合狀態
            portfolio.current_cash = cash
            # 重新計算 current_capital = cash + 持倉市值
            open_after = self._load_open_positions(session, portfolio.id)
            market_value = sum(today_prices.get(p["stock_id"], p["entry_price"]) * p["shares"] for p in open_after)
            portfolio.current_capital = cash + market_value
            portfolio.updated_at = datetime.utcnow()
            session.commit()

            # Ex-Ante VaR（不阻擋交易，僅記錄日誌）
            if open_after and portfolio.current_capital > 0 and price_rows:
                after_sids = [p["stock_id"] for p in open_after]
                var_price_data: dict[str, pd.Series] = {}
                for sid in after_sids:
                    sid_closes = [r[2] for r in price_rows if r[0] == sid]
                    if len(sid_closes) >= 20:
                        var_price_data[sid] = pd.Series(sid_closes)
                if var_price_data:
                    cov_mat = compute_covariance_matrix(var_price_data, window=60, min_periods=20)
                    if not cov_mat.empty:
                        pos_weights = {}
                        for p in open_after:
                            mv = today_prices.get(p["stock_id"], p["entry_price"]) * p["shares"]
                            pos_weights[p["stock_id"]] = mv / portfolio.current_capital
                        var_result = compute_portfolio_var(pos_weights, cov_mat, portfolio.current_capital)
                        logger.info(
                            "[%s] Ex-Ante VaR(95%%): %.0f (%.2f%%)",
                            self.portfolio_name,
                            var_result["var_amount"],
                            var_result["var_pct"],
                        )

            logger.info(
                "[%s] 更新完成: 賣出=%d, 續持=%d, 買入=%d, 保持=%d | 現金=%.0f, 總資產=%.0f",
                self.portfolio_name,
                len(actions.to_sell),
                len(actions.renewed),
                len(actions.to_buy),
                len(actions.to_hold),
                portfolio.current_cash,
                portfolio.current_capital,
            )
            return actions

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
            if mode == "all":
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

            # 成本歸因累計器
            total_commission = 0.0
            total_tax = 0.0
            total_slippage_cost = 0.0
            turnover_value = 0.0

            # 漲跌停偵測用前日收盤價
            prev_close_map: dict[str, float] = {}

            # VaR 計算用：累計每日收盤價（最近 60 日窗口）
            daily_close_history: dict[str, list[float]] = {}  # {stock_id: [close1, close2, ...]}

            # 每日持倉快照
            daily_positions_snapshot: list[dict] = []

            # 成本閘門：每 ISO 週 holding_expired 累計
            cost_cfg = settings.quant.rotation_cost
            weekly_swap_counter: dict[tuple[int, int], int] = {}

            for day in trading_cal:
                # 取今日排名
                if day in all_rankings:
                    last_rankings = all_rankings[day]

                if not last_rankings:
                    equity_curve.append({"date": day, "equity": cash})
                    continue

                # 取今日 OHLCV（優先從預載入快取取得，cache miss 時 fallback 到 DB 查詢）
                all_sids = list({p["stock_id"] for p in positions} | {r["stock_id"] for r in last_rankings})
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

                # 止損價：從持倉記錄取進場時鎖定的止損價
                stop_losses = {p["stock_id"]: p["stop_loss"] for p in positions if p.get("stop_loss") is not None}
                for r in last_rankings:
                    if r["stock_id"] not in stop_losses and r.get("stop_loss") is not None:
                        stop_losses[r["stock_id"]] = r["stop_loss"]

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
                    min_hold_days=gate_min_hold,
                    score_gap_threshold=gate_score_gap,
                    weekly_swap_cap=gate_weekly_cap,
                    weekly_swaps_used=weekly_used_this_week,
                )
                if cost_cfg.enabled and actions.holding_expired_sells > 0:
                    weekly_swap_counter[(iso_y, iso_w)] = weekly_used_this_week + actions.holding_expired_sells

                # ── 執行賣出 ──
                sold_ids = set()
                for sell in actions.to_sell:
                    sid = sell["stock_id"]
                    exit_price = sell.get("exit_price", sell.get("entry_price", 0))
                    shares = sell.get("shares", 0)
                    ohlcv = today_ohlcv.get(sid, {})

                    # 動態滑價
                    if dynamic_slippage:
                        sell_slip = compute_dynamic_slippage(
                            ohlcv.get("volume", 0),
                            ohlcv.get("high", 0),
                            ohlcv.get("low", 0),
                            exit_price,
                            side="sell",
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
                    pnl, return_pct = compute_position_pnl(
                        sell["entry_price"],
                        exit_price,
                        shares,
                        buy_slippage=buy_slip,
                        sell_slippage=sell_slip,
                    )
                    costs = compute_trade_costs(exit_price, shares, sell_slip, side="sell")
                    sell_proceeds = exit_price * shares * (1 - COMMISSION_RATE - TAX_RATE - sell_slip)
                    cash += sell_proceeds

                    total_commission += costs.commission
                    total_tax += costs.tax
                    total_slippage_cost += costs.slippage_cost
                    turnover_value += exit_price * shares

                    all_trades.append(
                        {
                            "stock_id": sid,
                            "entry_date": sell["entry_date"],
                            "entry_price": sell["entry_price"],
                            "exit_date": day,
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

                # 更新持倉列表
                new_positions = []
                for pos in positions:
                    if pos["stock_id"] not in sold_ids:
                        new_positions.append(pos)
                positions = new_positions

                # 處理續持（止損價只上移不下移）
                renewed_ids = {r["stock_id"] for r in actions.renewed}
                ranking_sl = {r["stock_id"]: r["stop_loss"] for r in last_rankings if r.get("stop_loss") is not None}
                for pos in positions:
                    if pos["stock_id"] in renewed_ids:
                        renew_info = next(r for r in actions.renewed if r["stock_id"] == pos["stock_id"])
                        pos["planned_exit_date"] = renew_info["new_planned_exit_date"]
                        new_sl = ranking_sl.get(pos["stock_id"])
                        if new_sl is not None:
                            old_sl = pos.get("stop_loss")
                            if old_sl is None or new_sl > old_sl:
                                pos["stop_loss"] = new_sl

                # ── 執行買入 ──
                for buy in actions.to_buy:
                    sid = buy["stock_id"]
                    price = buy["entry_price"]
                    alloc = buy["allocated_capital"]
                    ohlcv = today_ohlcv.get(sid, {})

                    # 漲停跳過
                    if limit_price_check:
                        prev_c = prev_close_map.get(sid)
                        if prev_c:
                            is_up, _ = detect_limit_price(ohlcv.get("open", price), prev_c)
                            if is_up:
                                continue

                    # 動態滑價
                    if dynamic_slippage:
                        buy_slip = compute_dynamic_slippage(
                            ohlcv.get("volume", 0),
                            ohlcv.get("high", 0),
                            ohlcv.get("low", 0),
                            price,
                            side="buy",
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
                    if shares <= 0:
                        continue

                    costs = compute_trade_costs(price, shares, buy_slip, side="buy")
                    buy_cost = price * shares * (1 + COMMISSION_RATE + buy_slip)
                    cash -= buy_cost

                    total_commission += costs.commission
                    total_slippage_cost += costs.slippage_cost
                    turnover_value += price * shares

                    planned_exit = compute_planned_exit_date(day, holding_days, trading_cal)
                    positions.append(
                        {
                            "stock_id": sid,
                            "stock_name": buy.get("stock_name", ""),
                            "entry_date": day,
                            "entry_price": price,
                            "entry_rank": buy["rank"],
                            "entry_score": buy.get("score"),
                            "shares": shares,
                            "allocated_capital": alloc,
                            "planned_exit_date": planned_exit,
                            "stop_loss": buy.get("stop_loss"),
                            "buy_slippage": buy_slip,
                        }
                    )

                # 計算當日權益
                total_equity = cash + sum(
                    today_prices.get(p["stock_id"], p["entry_price"]) * p["shares"] for p in positions
                )

                # 累計每日收盤價（VaR 用，保留最近 65 日供 60 日窗口）
                for sid, ohlcv_data in today_ohlcv.items():
                    daily_close_history.setdefault(sid, []).append(ohlcv_data["close"])
                    if len(daily_close_history[sid]) > 65:
                        daily_close_history[sid] = daily_close_history[sid][-65:]

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
                    exit_p = ohlcv.get("close", pos["entry_price"])

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
                    pnl, return_pct = compute_position_pnl(
                        pos["entry_price"],
                        exit_p,
                        pos["shares"],
                        buy_slippage=buy_slip,
                        sell_slippage=sell_slip,
                    )
                    costs = compute_trade_costs(exit_p, pos["shares"], sell_slip, side="sell")
                    sell_proceeds = exit_p * pos["shares"] * (1 - COMMISSION_RATE - TAX_RATE - sell_slip)
                    cash += sell_proceeds

                    total_commission += costs.commission
                    total_tax += costs.tax
                    total_slippage_cost += costs.slippage_cost

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
                            "exit_reason": "backtest_end",
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
                # 成本歸因
                "total_commission": round(total_commission, 2),
                "total_tax": round(total_tax, 2),
                "total_slippage_cost": round(total_slippage_cost, 2),
                "total_cost": round(total_commission + total_tax + total_slippage_cost, 2),
                "cost_drag_pct": round((total_commission + total_tax + total_slippage_cost) / capital * 100, 4)
                if capital > 0
                else 0,
                # TAIEX Benchmark
                "benchmark_return": benchmark_return,
            }

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

            # 寫入 DB
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

    def _compute_equity_history(self, session, portfolio: RotationPortfolio) -> list[float]:
        """從已平倉記錄重建淨值序列（用於回撤計算）。"""
        equity = [portfolio.initial_capital]
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
        running = portfolio.initial_capital
        for pos in closed:
            running += pos.pnl or 0.0
            equity.append(running)
        # 最後加上當前資本
        equity.append(portfolio.current_capital)
        return equity

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
                "shares": p.shares,
                "allocated_capital": p.allocated_capital,
                "planned_exit_date": p.planned_exit_date,
                "stop_loss": p.stop_loss,
            }
            for p in positions
        ]

    def _find_latest_rankings(self, session, mode: str, before_date: date) -> list[dict]:
        """找最近的 scan_date 排名（當日無 discover 結果時使用）。"""
        if mode == "all":
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

    def _execute_sell(self, session, portfolio_id: int, sell: dict, today: date, cash: float) -> float:
        """執行賣出：更新 RotationPosition 狀態並回收資金。"""
        stmt = select(RotationPosition).where(
            RotationPosition.portfolio_id == portfolio_id,
            RotationPosition.stock_id == sell["stock_id"],
            RotationPosition.status == "open",
        )
        pos = session.execute(stmt).scalar_one_or_none()
        if pos is None:
            return cash

        exit_price = sell.get("exit_price", pos.entry_price)
        pnl, return_pct = compute_position_pnl(pos.entry_price, exit_price, pos.shares)
        sell_proceeds = exit_price * pos.shares * (1 - COMMISSION_RATE - TAX_RATE - SLIPPAGE_RATE)

        pos.exit_date = today
        pos.exit_price = exit_price
        pos.exit_reason = sell["reason"]
        pos.pnl = pnl
        pos.return_pct = return_pct
        pos.status = "closed"
        pos.holding_days_count = sell.get("days_held", 0)

        return cash + sell_proceeds

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
        self, session, portfolio_id: int, buy: dict, today: date, trading_cal: list[date], cash: float
    ) -> float:
        """執行買入：建立 RotationPosition 並扣減現金。"""
        price = buy["entry_price"]
        shares = buy["shares"]
        if shares <= 0:
            return cash

        buy_cost = price * shares * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
        if buy_cost > cash:
            # 資金不足，縮減股數
            shares = compute_shares(cash, price)
            if shares <= 0:
                return cash
            buy_cost = price * shares * (1 + COMMISSION_RATE + SLIPPAGE_RATE)

        from src.portfolio.rotation import compute_planned_exit_date

        # 需要 portfolio 的 holding_days
        portfolio = session.execute(select(RotationPortfolio).where(RotationPortfolio.id == portfolio_id)).scalar_one()
        planned_exit = compute_planned_exit_date(today, portfolio.holding_days, trading_cal)

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
            allocated_capital=buy.get("allocated_capital", buy_cost),
            stop_loss=buy.get("stop_loss"),
            status="open",
        )
        session.add(pos)
        return cash - buy_cost

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
