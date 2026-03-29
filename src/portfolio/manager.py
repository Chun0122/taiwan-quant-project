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

from src.constants import COMMISSION_RATE, SLIPPAGE_RATE, TAX_RATE
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
    compute_planned_exit_date,
    compute_portfolio_drawdown,
    compute_position_pnl,
    compute_rotation_actions,
    compute_shares,
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
    """取得指定日期的收盤價。"""
    if not stock_ids:
        return {}
    stmt = select(DailyPrice.stock_id, DailyPrice.close).where(
        DailyPrice.stock_id.in_(stock_ids),
        DailyPrice.date == target_date,
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
            if not rankings:
                # 嘗試找最近的 scan_date
                rankings = self._find_latest_rankings(session, portfolio.mode, today)

            # 交易日曆（前 120 天 ~ 未來 30 天）
            from datetime import timedelta

            cal_start = today - timedelta(days=180)
            cal_end = today + timedelta(days=60)
            trading_cal = _get_trading_calendar(session, cal_start, cal_end)

            # 今日收盤價
            all_sids = list({p["stock_id"] for p in open_positions} | {r["stock_id"] for r in rankings})
            today_prices = _get_prices_on_date(session, all_sids, today)

            # 止損價
            stop_losses = {r["stock_id"]: r["stop_loss"] for r in rankings if r.get("stop_loss") is not None}

            # ── Regime fallback：嘗試從 JSON 持久化讀取 ──
            if regime is None:
                try:
                    from src.regime.detector import RegimeStateMachine

                    rsm = RegimeStateMachine()
                    if rsm._state.get("current_regime"):
                        regime = rsm._state["current_regime"]
                        logger.info("從 RegimeStateMachine 讀取 regime=%s", regime)
                except Exception:
                    pass  # 無持久化資料時 regime 維持 None

            # ── Correlation Budget：計算持倉+候選相關性矩陣 ──
            corr_matrix = None
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
                regime=regime,
            )

            # 執行賣出
            cash = portfolio.current_cash
            for sell in actions.to_sell:
                cash = self._execute_sell(session, portfolio.id, sell, today, cash)

            # 執行續持
            for renew in actions.renewed:
                self._execute_renewal(session, portfolio.id, renew)

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
    ) -> RotationBacktestResult:
        """歷史回測：逐日模擬 rotation 策略。

        可直接從已建立的 portfolio 讀取參數，
        或用傳入參數覆蓋（適用於 ad-hoc 回測）。
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

            # 逐日模擬
            positions: list[dict] = []  # 模擬持倉
            cash = capital
            equity_curve: list[dict] = []
            all_trades: list[dict] = []
            last_rankings: list[dict] = []

            for day in trading_cal:
                # 取今日排名
                if day in all_rankings:
                    last_rankings = all_rankings[day]

                if not last_rankings:
                    equity_curve.append({"date": day, "equity": cash})
                    continue

                # 取今日收盤價
                all_sids = list({p["stock_id"] for p in positions} | {r["stock_id"] for r in last_rankings})
                today_prices = _get_prices_on_date(session, all_sids, day)

                # 止損價
                stop_losses = {r["stock_id"]: r["stop_loss"] for r in last_rankings if r.get("stop_loss") is not None}

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
                )

                # 執行賣出
                sold_ids = set()
                for sell in actions.to_sell:
                    sid = sell["stock_id"]
                    exit_price = sell.get("exit_price", sell.get("entry_price", 0))
                    shares = sell.get("shares", 0)
                    pnl, return_pct = compute_position_pnl(sell["entry_price"], exit_price, shares)
                    sell_proceeds = exit_price * shares * (1 - COMMISSION_RATE - TAX_RATE - SLIPPAGE_RATE)
                    cash += sell_proceeds
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
                        }
                    )
                    sold_ids.add(sid)

                # 更新持倉列表
                new_positions = []
                for pos in positions:
                    if pos["stock_id"] not in sold_ids:
                        new_positions.append(pos)
                positions = new_positions

                # 處理續持
                renewed_ids = {r["stock_id"] for r in actions.renewed}
                for pos in positions:
                    if pos["stock_id"] in renewed_ids:
                        renew_info = next(r for r in actions.renewed if r["stock_id"] == pos["stock_id"])
                        pos["planned_exit_date"] = renew_info["new_planned_exit_date"]

                # 執行買入
                for buy in actions.to_buy:
                    sid = buy["stock_id"]
                    price = buy["entry_price"]
                    alloc = buy["allocated_capital"]
                    shares = compute_shares(alloc, price)
                    if shares <= 0:
                        continue
                    buy_cost = price * shares * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    cash -= buy_cost
                    planned_exit = compute_planned_exit_date(day, holding_days, trading_cal)
                    positions.append(
                        {
                            "stock_id": sid,
                            "stock_name": buy.get("stock_name", ""),
                            "entry_date": day,
                            "entry_price": price,
                            "entry_rank": buy["rank"],
                            "shares": shares,
                            "allocated_capital": alloc,
                            "planned_exit_date": planned_exit,
                        }
                    )

                # 計算當日權益
                market_value = sum(today_prices.get(p["stock_id"], p["entry_price"]) * p["shares"] for p in positions)
                equity_curve.append({"date": day, "equity": cash + market_value})

            # 強制平倉期末持倉
            if positions and trading_cal:
                last_day = trading_cal[-1]
                last_prices = _get_prices_on_date(session, [p["stock_id"] for p in positions], last_day)
                for pos in positions:
                    sid = pos["stock_id"]
                    exit_p = last_prices.get(sid, pos["entry_price"])
                    pnl, return_pct = compute_position_pnl(pos["entry_price"], exit_p, pos["shares"])
                    sell_proceeds = exit_p * pos["shares"] * (1 - COMMISSION_RATE - TAX_RATE - SLIPPAGE_RATE)
                    cash += sell_proceeds
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
                        }
                    )

            # 計算績效指標
            metrics = self._compute_backtest_metrics(equity_curve, all_trades, capital)

            return RotationBacktestResult(
                equity_curve=equity_curve,
                trades=all_trades,
                metrics=metrics,
                config={
                    "mode": mode,
                    "max_positions": max_positions,
                    "holding_days": holding_days,
                    "capital": capital,
                    "allow_renewal": allow_renewal,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                },
            )

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

    def _execute_renewal(self, session, portfolio_id: int, renew: dict) -> None:
        """執行續持：更新 planned_exit_date。"""
        stmt = select(RotationPosition).where(
            RotationPosition.portfolio_id == portfolio_id,
            RotationPosition.stock_id == renew["stock_id"],
            RotationPosition.status == "open",
        )
        pos = session.execute(stmt).scalar_one_or_none()
        if pos is not None:
            pos.planned_exit_date = renew["new_planned_exit_date"]
            pos.holding_days_count = renew.get("days_held", pos.holding_days_count)

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
