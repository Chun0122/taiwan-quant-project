"""視覺化資料查詢模組 — 提供 Streamlit 頁面所需的 DB 查詢函式。"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import case, func, select

from src.data.database import get_session, init_db
from src.data.schema import (
    BacktestResult,
    DailyPrice,
    DiscoveryRecord,
    InstitutionalInvestor,
    MarginTrading,
    PortfolioBacktestResult,
    PortfolioTrade,
    StockInfo,
    TechnicalIndicator,
    Trade,
)

init_db()


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
