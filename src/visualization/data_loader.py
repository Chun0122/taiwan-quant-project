"""視覺化資料查詢模組 — 提供 Streamlit 頁面所需的 DB 查詢函式。"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import case, func, select

from src.data.database import get_session, init_db
from src.data.migrate import run_migrations
from src.data.schema import (
    BacktestResult,
    DailyPrice,
    DiscoveryRecord,
    InstitutionalInvestor,
    MarginTrading,
    PortfolioBacktestResult,
    PortfolioTrade,
    RotationPortfolio,
    RotationPosition,
    StockInfo,
    TechnicalIndicator,
    Trade,
    WatchEntry,
)

init_db()
run_migrations()


def get_stock_list() -> list[str]:
    """取得 DB 中有日K線資料的股票清單。"""
    with get_session() as session:
        rows = session.execute(select(func.distinct(DailyPrice.stock_id)).order_by(DailyPrice.stock_id)).scalars().all()
    return list(rows)


def load_price_with_indicators(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """載入日K線 + 技術指標寬表。"""
    with get_session() as session:
        prices = (
            session.execute(
                select(DailyPrice)
                .where(DailyPrice.stock_id == stock_id)
                .where(DailyPrice.date >= start)
                .where(DailyPrice.date <= end)
                .order_by(DailyPrice.date)
            )
            .scalars()
            .all()
        )

    if not prices:
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "turnover": r.turnover,
                "spread": r.spread,
            }
            for r in prices
        ]
    )

    # 載入指標並 pivot
    with get_session() as session:
        indicators = (
            session.execute(
                select(TechnicalIndicator)
                .where(TechnicalIndicator.stock_id == stock_id)
                .where(TechnicalIndicator.date >= start)
                .where(TechnicalIndicator.date <= end)
                .order_by(TechnicalIndicator.date)
            )
            .scalars()
            .all()
        )

    if indicators:
        df_ind = pd.DataFrame([{"date": r.date, "name": r.name, "value": r.value} for r in indicators])
        df_wide = df_ind.pivot_table(index="date", columns="name", values="value")
        df = df.merge(df_wide, on="date", how="left")

    df["date"] = pd.to_datetime(df["date"])
    return df


def load_institutional(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """載入三大法人買賣超資料。"""
    with get_session() as session:
        rows = (
            session.execute(
                select(InstitutionalInvestor)
                .where(InstitutionalInvestor.stock_id == stock_id)
                .where(InstitutionalInvestor.date >= start)
                .where(InstitutionalInvestor.date <= end)
                .order_by(InstitutionalInvestor.date)
            )
            .scalars()
            .all()
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([{"date": r.date, "name": r.name, "buy": r.buy, "sell": r.sell, "net": r.net} for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_margin(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """載入融資融券資料。"""
    with get_session() as session:
        rows = (
            session.execute(
                select(MarginTrading)
                .where(MarginTrading.stock_id == stock_id)
                .where(MarginTrading.date >= start)
                .where(MarginTrading.date <= end)
                .order_by(MarginTrading.date)
            )
            .scalars()
            .all()
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [{"date": r.date, "margin_balance": r.margin_balance, "short_balance": r.short_balance} for r in rows]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_backtest_list() -> pd.DataFrame:
    """載入所有回測紀錄。"""
    with get_session() as session:
        rows = session.execute(select(BacktestResult).order_by(BacktestResult.id.desc())).scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "id": r.id,
                "stock_id": r.stock_id,
                "strategy_name": r.strategy_name,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "initial_capital": r.initial_capital,
                "final_capital": r.final_capital,
                "total_return": r.total_return,
                "annual_return": r.annual_return,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "total_trades": r.total_trades,
                "sortino_ratio": getattr(r, "sortino_ratio", None),
                "calmar_ratio": getattr(r, "calmar_ratio", None),
                "var_95": getattr(r, "var_95", None),
                "cvar_95": getattr(r, "cvar_95", None),
                "profit_factor": getattr(r, "profit_factor", None),
                "created_at": r.created_at,
            }
            for r in rows
        ]
    )


def load_backtest_by_id(backtest_id: int) -> dict | None:
    """載入單筆回測紀錄。"""
    with get_session() as session:
        r = session.execute(select(BacktestResult).where(BacktestResult.id == backtest_id)).scalar_one_or_none()

    if not r:
        return None

    return {
        "id": r.id,
        "stock_id": r.stock_id,
        "strategy_name": r.strategy_name,
        "start_date": r.start_date,
        "end_date": r.end_date,
        "initial_capital": r.initial_capital,
        "final_capital": r.final_capital,
        "total_return": r.total_return,
        "annual_return": r.annual_return,
        "sharpe_ratio": r.sharpe_ratio,
        "max_drawdown": r.max_drawdown,
        "win_rate": r.win_rate,
        "total_trades": r.total_trades,
        "sortino_ratio": getattr(r, "sortino_ratio", None),
        "calmar_ratio": getattr(r, "calmar_ratio", None),
        "var_95": getattr(r, "var_95", None),
        "cvar_95": getattr(r, "cvar_95", None),
        "profit_factor": getattr(r, "profit_factor", None),
    }


def load_trades(backtest_id: int) -> pd.DataFrame:
    """載入指定回測的交易明細。"""
    with get_session() as session:
        rows = (
            session.execute(select(Trade).where(Trade.backtest_id == backtest_id).order_by(Trade.entry_date))
            .scalars()
            .all()
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "entry_date": r.entry_date,
                "entry_price": r.entry_price,
                "exit_date": r.exit_date,
                "exit_price": r.exit_price,
                "shares": r.shares,
                "pnl": r.pnl,
                "return_pct": r.return_pct,
                "exit_reason": getattr(r, "exit_reason", None),
            }
            for r in rows
        ]
    )


# ------------------------------------------------------------------ #
#  投資組合查詢
# ------------------------------------------------------------------ #


def load_portfolio_list() -> pd.DataFrame:
    """載入所有投資組合回測紀錄。"""
    with get_session() as session:
        rows = (
            session.execute(select(PortfolioBacktestResult).order_by(PortfolioBacktestResult.id.desc())).scalars().all()
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "id": r.id,
                "strategy_name": r.strategy_name,
                "stock_ids": r.stock_ids,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "initial_capital": r.initial_capital,
                "final_capital": r.final_capital,
                "total_return": r.total_return,
                "annual_return": r.annual_return,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "total_trades": r.total_trades,
                "sortino_ratio": getattr(r, "sortino_ratio", None),
                "calmar_ratio": getattr(r, "calmar_ratio", None),
                "var_95": getattr(r, "var_95", None),
                "cvar_95": getattr(r, "cvar_95", None),
                "profit_factor": getattr(r, "profit_factor", None),
                "allocation_method": getattr(r, "allocation_method", None),
                "created_at": r.created_at,
            }
            for r in rows
        ]
    )


def load_portfolio_by_id(portfolio_id: int) -> dict | None:
    """載入單筆投資組合回測紀錄。"""
    with get_session() as session:
        r = session.execute(
            select(PortfolioBacktestResult).where(PortfolioBacktestResult.id == portfolio_id)
        ).scalar_one_or_none()

    if not r:
        return None

    return {
        "id": r.id,
        "strategy_name": r.strategy_name,
        "stock_ids": r.stock_ids,
        "start_date": r.start_date,
        "end_date": r.end_date,
        "initial_capital": r.initial_capital,
        "final_capital": r.final_capital,
        "total_return": r.total_return,
        "annual_return": r.annual_return,
        "sharpe_ratio": r.sharpe_ratio,
        "max_drawdown": r.max_drawdown,
        "win_rate": r.win_rate,
        "total_trades": r.total_trades,
        "sortino_ratio": getattr(r, "sortino_ratio", None),
        "calmar_ratio": getattr(r, "calmar_ratio", None),
        "var_95": getattr(r, "var_95", None),
        "cvar_95": getattr(r, "cvar_95", None),
        "profit_factor": getattr(r, "profit_factor", None),
        "allocation_method": getattr(r, "allocation_method", None),
    }


def load_stock_info_map() -> dict[str, dict]:
    """載入全部股票基本資料。

    Returns:
        {stock_id: {"stock_name": ..., "industry_category": ..., "listing_type": ...}}
    """
    with get_session() as session:
        rows = session.execute(select(StockInfo).order_by(StockInfo.stock_id)).scalars().all()

    return {
        r.stock_id: {
            "stock_name": r.stock_name,
            "industry_category": r.industry_category,
            "listing_type": r.listing_type,
        }
        for r in rows
    }


def load_portfolio_trades(portfolio_id: int) -> pd.DataFrame:
    """載入指定投資組合回測的交易明細。"""
    with get_session() as session:
        rows = (
            session.execute(
                select(PortfolioTrade)
                .where(PortfolioTrade.portfolio_backtest_id == portfolio_id)
                .order_by(PortfolioTrade.entry_date)
            )
            .scalars()
            .all()
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "stock_id": r.stock_id,
                "entry_date": r.entry_date,
                "entry_price": r.entry_price,
                "exit_date": r.exit_date,
                "exit_price": r.exit_price,
                "shares": r.shares,
                "pnl": r.pnl,
                "return_pct": r.return_pct,
                "exit_reason": getattr(r, "exit_reason", None),
            }
            for r in rows
        ]
    )


# ------------------------------------------------------------------ #
#  Discover 推薦歷史
# ------------------------------------------------------------------ #


def _discovery_date_filters(stmt, start_date=None, end_date=None):
    """共用日期篩選。"""
    if start_date:
        stmt = stmt.where(DiscoveryRecord.scan_date >= start_date)
    if end_date:
        stmt = stmt.where(DiscoveryRecord.scan_date <= end_date)
    return stmt


def load_discovery_calendar_counts(mode: str, start_date=None, end_date=None) -> pd.DataFrame:
    """每日推薦次數（日曆熱圖用）。"""
    with get_session() as session:
        stmt = (
            select(
                DiscoveryRecord.scan_date,
                func.count(DiscoveryRecord.id).label("count"),
            )
            .where(DiscoveryRecord.mode == mode)
            .group_by(DiscoveryRecord.scan_date)
            .order_by(DiscoveryRecord.scan_date)
        )
        stmt = _discovery_date_filters(stmt, start_date, end_date)
        rows = session.execute(stmt).all()

    if not rows:
        return pd.DataFrame(columns=["scan_date", "count"])
    return pd.DataFrame(rows, columns=["scan_date", "count"])


def load_discovery_calendar_returns(
    mode: str,
    holding_days: int = 5,
    top_n: int | None = None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """每日平均報酬率（日曆熱圖用，透過 DiscoveryPerformance 計算）。"""
    from src.discovery.performance import DiscoveryPerformance

    perf = DiscoveryPerformance(
        mode=mode,
        holding_days=[holding_days],
        top_n=top_n,
        start_date=start_date,
        end_date=end_date,
    )
    result = perf.evaluate()
    by_scan = result["by_scan"]
    if by_scan.empty:
        return pd.DataFrame(columns=["scan_date", "avg_return", "count"])
    by_scan = by_scan[by_scan["holding_days"] == holding_days].copy()
    return by_scan[["scan_date", "avg_return", "count"]].reset_index(drop=True)


def load_discovery_performance(
    mode: str,
    holding_days: list[int] | None = None,
    top_n: int | None = None,
    start_date=None,
    end_date=None,
) -> dict:
    """呼叫 DiscoveryPerformance.evaluate() 回傳完整結果。"""
    from src.discovery.performance import DiscoveryPerformance

    perf = DiscoveryPerformance(
        mode=mode,
        holding_days=holding_days or [5, 10, 20],
        top_n=top_n,
        start_date=start_date,
        end_date=end_date,
    )
    return perf.evaluate()


def load_discovery_stock_frequency(
    mode: str,
    top_n: int = 20,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """推薦頻率排行（SQL GROUP BY）。"""
    with get_session() as session:
        stmt = (
            select(
                DiscoveryRecord.stock_id,
                DiscoveryRecord.stock_name,
                func.count(DiscoveryRecord.id).label("recommend_count"),
                func.avg(DiscoveryRecord.rank).label("avg_rank"),
                func.avg(DiscoveryRecord.composite_score).label("avg_composite_score"),
            )
            .where(DiscoveryRecord.mode == mode)
            .group_by(DiscoveryRecord.stock_id, DiscoveryRecord.stock_name)
            .order_by(func.count(DiscoveryRecord.id).desc())
            .limit(top_n)
        )
        stmt = _discovery_date_filters(stmt, start_date, end_date)
        rows = session.execute(stmt).all()

    if not rows:
        return pd.DataFrame(columns=["stock_id", "stock_name", "recommend_count", "avg_rank", "avg_composite_score"])
    return pd.DataFrame(
        rows,
        columns=["stock_id", "stock_name", "recommend_count", "avg_rank", "avg_composite_score"],
    )


def load_discovery_records(
    mode: str,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """原始推薦記錄（純 DiscoveryRecord 查詢）。"""
    with get_session() as session:
        stmt = (
            select(DiscoveryRecord)
            .where(DiscoveryRecord.mode == mode)
            .order_by(DiscoveryRecord.scan_date.desc(), DiscoveryRecord.rank)
        )
        stmt = _discovery_date_filters(stmt, start_date, end_date)
        rows = session.execute(stmt).scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "scan_date": r.scan_date,
                "rank": r.rank,
                "stock_id": r.stock_id,
                "stock_name": r.stock_name,
                "close": r.close,
                "composite_score": r.composite_score,
                "technical_score": r.technical_score,
                "chip_score": r.chip_score,
                "fundamental_score": r.fundamental_score,
                "news_score": r.news_score,
                "sector_bonus": r.sector_bonus,
                "industry_category": r.industry_category,
                "regime": r.regime,
            }
            for r in rows
        ]
    )


# ------------------------------------------------------------------ #
#  市場總覽
# ------------------------------------------------------------------ #


def load_taiex_history(days: int = 120) -> pd.DataFrame:
    """載入 TAIEX 加權指數歷史 K 線。"""
    with get_session() as session:
        rows = session.execute(
            select(
                DailyPrice.date,
                DailyPrice.open,
                DailyPrice.high,
                DailyPrice.low,
                DailyPrice.close,
                DailyPrice.volume,
            )
            .where(DailyPrice.stock_id == "TAIEX")
            .order_by(DailyPrice.date.desc())
            .limit(days)
        ).all()

    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_market_breadth(days: int = 60) -> pd.DataFrame:
    """計算每日漲跌家數 + 全市場成交量。

    使用 DailyPrice.spread 欄位判斷漲跌，排除 TAIEX 本身。
    """
    with get_session() as session:
        stmt = (
            select(
                DailyPrice.date,
                func.sum(case((DailyPrice.spread > 0, 1), else_=0)).label("rising"),
                func.sum(case((DailyPrice.spread < 0, 1), else_=0)).label("falling"),
                func.sum(case((DailyPrice.spread == 0, 1), else_=0)).label("flat"),
                func.sum(DailyPrice.volume).label("total_volume"),
            )
            .where(DailyPrice.stock_id != "TAIEX")
            .where(DailyPrice.spread.is_not(None))
            .group_by(DailyPrice.date)
            .order_by(DailyPrice.date.desc())
            .limit(days)
        )
        rows = session.execute(stmt).all()

    if not rows:
        return pd.DataFrame(columns=["date", "rising", "falling", "flat", "total_volume"])

    df = pd.DataFrame(rows, columns=["date", "rising", "falling", "flat", "total_volume"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_market_breadth_stats(windows: list[int] | None = None) -> pd.DataFrame:
    """計算多窗口漲跌家數聚合統計（近 N 日平均）。"""
    if windows is None:
        windows = [1, 5, 20]

    max_window = max(windows)
    breadth = load_market_breadth(days=max_window + 5)
    if breadth.empty:
        return pd.DataFrame(columns=["window", "rising", "falling", "flat"])

    results = []
    for w in windows:
        recent = breadth.tail(w)
        results.append(
            {
                "window": f"近{w}日" if w > 1 else "當日",
                "rising": int(recent["rising"].mean()),
                "falling": int(recent["falling"].mean()),
                "flat": int(recent["flat"].mean()),
            }
        )
    return pd.DataFrame(results)


def load_top_institutional(lookback: int = 5, top_n: int = 10) -> pd.DataFrame:
    """載入近 N 日法人買賣超排行（合併三大法人）。"""
    with get_session() as session:
        # 取最近 lookback 個交易日
        recent_dates = (
            session.execute(
                select(func.distinct(InstitutionalInvestor.date))
                .order_by(InstitutionalInvestor.date.desc())
                .limit(lookback)
            )
            .scalars()
            .all()
        )
        if not recent_dates:
            return pd.DataFrame(columns=["stock_id", "stock_name", "foreign", "trust", "dealer", "total"])

        min_date = min(recent_dates)

        # 按法人類型分組聚合
        stmt = (
            select(
                InstitutionalInvestor.stock_id,
                InstitutionalInvestor.name,
                func.sum(InstitutionalInvestor.net).label("net_sum"),
            )
            .where(InstitutionalInvestor.date >= min_date)
            .group_by(InstitutionalInvestor.stock_id, InstitutionalInvestor.name)
        )
        rows = session.execute(stmt).all()

    if not rows:
        return pd.DataFrame(columns=["stock_id", "stock_name", "foreign", "trust", "dealer", "total"])

    # pivot 成寬表
    df = pd.DataFrame(rows, columns=["stock_id", "name", "net_sum"])
    pivot = df.pivot_table(index="stock_id", columns="name", values="net_sum", aggfunc="sum").fillna(0)

    result = pd.DataFrame({"stock_id": pivot.index})
    result["foreign"] = pivot.get("Foreign_Investor", 0).values if "Foreign_Investor" in pivot.columns else 0
    result["trust"] = pivot.get("Investment_Trust", 0).values if "Investment_Trust" in pivot.columns else 0
    result["dealer"] = pivot.get("Dealer_self", 0).values if "Dealer_self" in pivot.columns else 0
    result["total"] = result["foreign"] + result["trust"] + result["dealer"]

    # JOIN stock_name
    info_map = load_stock_info_map()
    result["stock_name"] = result["stock_id"].map(lambda sid: info_map.get(sid, {}).get("stock_name", ""))

    # 排序：取 total 絕對值最大的 top_n（含買超和賣超排行）
    result = result.reindex(result["total"].abs().sort_values(ascending=False).index)
    result = result.head(top_n).reset_index(drop=True)
    return result


def load_market_volume_summary(days: int = 60) -> pd.DataFrame:
    """載入全市場每日成交量摘要。"""
    with get_session() as session:
        stmt = (
            select(
                DailyPrice.date,
                func.sum(DailyPrice.turnover).label("total_turnover"),
            )
            .where(DailyPrice.stock_id != "TAIEX")
            .group_by(DailyPrice.date)
            .order_by(DailyPrice.date.desc())
            .limit(days)
        )
        rows = session.execute(stmt).all()

    if not rows:
        return pd.DataFrame(columns=["date", "total_turnover"])

    df = pd.DataFrame(rows, columns=["date", "total_turnover"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_announcements(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """載入指定股票的 MOPS 重大訊息公告。"""
    from src.data.schema import Announcement

    with get_session() as session:
        rows = (
            session.execute(
                select(Announcement)
                .where(Announcement.stock_id == stock_id)
                .where(Announcement.date >= start)
                .where(Announcement.date <= end)
                .order_by(Announcement.date, Announcement.seq)
            )
            .scalars()
            .all()
        )

    if not rows:
        return pd.DataFrame(columns=["date", "subject", "sentiment", "spoke_time"])

    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "subject": r.subject,
                "sentiment": r.sentiment,
                "spoke_time": r.spoke_time,
            }
            for r in rows
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


# ──────────────────────────────────────────────────────────────────── #
#  持倉監控
# ──────────────────────────────────────────────────────────────────── #


def load_watch_entries_with_status(status_filter: str | None = "active") -> pd.DataFrame:
    """載入 WatchEntry 持倉記錄，並根據最新收盤價計算即時狀態。

    Args:
        status_filter: "active" 只取持倉中，None/"all" 取全部。

    Returns:
        DataFrame，含欄位：
        id, stock_id, stock_name, entry_date, entry_price,
        stop_loss, take_profit, quantity, source, mode,
        entry_trigger, valid_until, status, close_date, close_price,
        notes, current_price, unrealized_pnl_pct, computed_status
    """
    import datetime

    from main import _compute_watch_status

    with get_session() as session:
        q = select(WatchEntry).order_by(WatchEntry.entry_date.desc())
        if status_filter and status_filter != "all":
            q = q.where(WatchEntry.status == status_filter)
        entries = session.execute(q).scalars().all()

    if not entries:
        return pd.DataFrame()

    # 查各股票最新收盤價
    stock_ids = list({e.stock_id for e in entries})
    latest_prices: dict[str, float] = {}
    with get_session() as session:
        for sid in stock_ids:
            row = session.execute(
                select(DailyPrice.close).where(DailyPrice.stock_id == sid).order_by(DailyPrice.date.desc()).limit(1)
            ).scalar()
            if row is not None:
                latest_prices[sid] = float(row)

    today = datetime.date.today()
    records = []
    for e in entries:
        cur_price = latest_prices.get(e.stock_id)
        if cur_price is not None and e.entry_price:
            pnl_pct = (cur_price - e.entry_price) / e.entry_price
        else:
            pnl_pct = None

        # 對 active 記錄即時計算狀態
        if e.status == "active":
            computed = _compute_watch_status(
                entry_price=e.entry_price,
                stop_loss=e.stop_loss,
                take_profit=e.take_profit,
                valid_until=e.valid_until,
                latest_price=cur_price,
                today=today,
            )
        else:
            computed = e.status  # closed/stopped_loss/taken_profit 已固化

        records.append(
            {
                "id": e.id,
                "stock_id": e.stock_id,
                "stock_name": e.stock_name or e.stock_id,
                "entry_date": e.entry_date,
                "entry_price": e.entry_price,
                "stop_loss": e.stop_loss,
                "take_profit": e.take_profit,
                "quantity": e.quantity,
                "source": e.source,
                "mode": e.mode,
                "entry_trigger": e.entry_trigger,
                "valid_until": e.valid_until,
                "status": e.status,
                "close_date": e.close_date,
                "close_price": e.close_price,
                "notes": e.notes,
                "current_price": cur_price,
                "unrealized_pnl_pct": pnl_pct,
                "computed_status": computed,
                # 移動止損欄位
                "trailing_stop_enabled": getattr(e, "trailing_stop_enabled", False),
                "trailing_atr_multiplier": getattr(e, "trailing_atr_multiplier", None),
                "highest_price_since_entry": getattr(e, "highest_price_since_entry", None),
            }
        )

    return pd.DataFrame(records)


def load_watch_entry_price_history(stock_id: str, entry_date: str, days: int = 60) -> pd.DataFrame:
    """載入個股持倉區間日K線（從 entry_date 往後 days 天）。

    Args:
        stock_id:   股票代號
        entry_date: 進場日（ISO 字串 YYYY-MM-DD）
        days:       往後取幾天（預設 60）

    Returns:
        DataFrame，含 date/open/high/low/close/volume 欄位（升序）。
    """
    import datetime

    try:
        start = datetime.date.fromisoformat(str(entry_date))
    except (ValueError, TypeError):
        start = datetime.date.today()
    end = (start + datetime.timedelta(days=days)).isoformat()

    with get_session() as session:
        rows = (
            session.execute(
                select(DailyPrice)
                .where(DailyPrice.stock_id == stock_id)
                .where(DailyPrice.date >= start.isoformat())
                .where(DailyPrice.date <= end)
                .order_by(DailyPrice.date)
            )
            .scalars()
            .all()
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
    )


# ──────────────────────────────────────────────────────────────────── #
#  輪動組合 — 部位控制總覽
# ──────────────────────────────────────────────────────────────────── #


def load_rotation_portfolio_names() -> list[str]:
    """取得所有輪動組合名稱（按建立時間排序）。"""
    with get_session() as session:
        rows = session.execute(select(RotationPortfolio.name).order_by(RotationPortfolio.created_at)).scalars().all()
    return list(rows)


def load_rotation_portfolio_info(name: str) -> dict | None:
    """載入輪動組合基本資訊。"""
    with get_session() as session:
        r = session.execute(select(RotationPortfolio).where(RotationPortfolio.name == name)).scalar_one_or_none()

    if not r:
        return None

    return {
        "name": r.name,
        "mode": r.mode,
        "max_positions": r.max_positions,
        "holding_days": r.holding_days,
        "initial_capital": r.initial_capital,
        "current_capital": r.current_capital,
        "current_cash": r.current_cash,
        "status": r.status,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
    }


def load_rotation_positions(name: str) -> list[dict]:
    """載入指定組合的開倉持倉（status='open'）。"""
    with get_session() as session:
        portfolio = session.execute(
            select(RotationPortfolio).where(RotationPortfolio.name == name)
        ).scalar_one_or_none()
        if not portfolio:
            return []

        rows = (
            session.execute(
                select(RotationPosition)
                .where(RotationPosition.portfolio_id == portfolio.id)
                .where(RotationPosition.status == "open")
                .order_by(RotationPosition.entry_date)
            )
            .scalars()
            .all()
        )

    # 查最新收盤價
    stock_ids = list({r.stock_id for r in rows})
    latest_prices: dict[str, float] = {}
    if stock_ids:
        with get_session() as session:
            for sid in stock_ids:
                price = session.execute(
                    select(DailyPrice.close).where(DailyPrice.stock_id == sid).order_by(DailyPrice.date.desc()).limit(1)
                ).scalar()
                if price is not None:
                    latest_prices[sid] = float(price)

    # 查 stop_loss（從最近的 DiscoveryRecord）
    stop_losses: dict[str, float] = {}
    if stock_ids:
        with get_session() as session:
            for sid in stock_ids:
                sl = session.execute(
                    select(DiscoveryRecord.stop_loss)
                    .where(DiscoveryRecord.stock_id == sid)
                    .where(DiscoveryRecord.stop_loss.is_not(None))
                    .order_by(DiscoveryRecord.scan_date.desc())
                    .limit(1)
                ).scalar()
                if sl is not None:
                    stop_losses[sid] = float(sl)

    positions = []
    for r in rows:
        cur_price = latest_prices.get(r.stock_id)
        unrealized_pnl = None
        unrealized_pct = None
        if cur_price is not None:
            unrealized_pnl = (cur_price - r.entry_price) * r.shares
            unrealized_pct = (cur_price - r.entry_price) / r.entry_price if r.entry_price else None

        positions.append(
            {
                "stock_id": r.stock_id,
                "stock_name": r.stock_name or r.stock_id,
                "entry_date": r.entry_date,
                "entry_price": r.entry_price,
                "shares": r.shares,
                "allocated_capital": r.allocated_capital,
                "planned_exit_date": r.planned_exit_date,
                "holding_days_count": r.holding_days_count,
                "current_price": cur_price,
                "stop_loss": stop_losses.get(r.stock_id),
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pct": unrealized_pct,
            }
        )
    return positions


def load_multi_stock_closes(stock_ids: list[str], days: int = 90) -> pd.DataFrame:
    """載入多股最近 N 日收盤價（columns=stock_id, index=date）。"""
    if not stock_ids:
        return pd.DataFrame()

    import datetime

    end_date = datetime.date.today().isoformat()
    start_date = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()

    with get_session() as session:
        rows = session.execute(
            select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close)
            .where(DailyPrice.stock_id.in_(stock_ids))
            .where(DailyPrice.date >= start_date)
            .where(DailyPrice.date <= end_date)
            .order_by(DailyPrice.date)
        ).all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["stock_id", "date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="stock_id", values="close")
    return pivot


def load_regime_state() -> dict | None:
    """從 JSON 檔讀取最新 Regime 狀態。"""
    import json
    from pathlib import Path

    state_file = Path("data/regime_state.json")
    if not state_file.exists():
        return None

    try:
        with open(state_file, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "regime": data.get("regime", "unknown"),
            "regime_since": data.get("regime_since"),
            "last_updated": data.get("last_updated"),
            "confirmation_count": data.get("confirmation_count", 0),
            "pending_transition": data.get("pending_transition"),
        }
    except (json.JSONDecodeError, OSError):
        return None
