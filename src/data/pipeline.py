"""ETL Pipeline — 整合 抓取 → 清洗 → 寫入資料庫 流程。"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

from src.config import settings
from src.data.database import get_session, init_db
from src.data.fetcher import FinMindFetcher
from src.data.schema import (
    Announcement,
    BacktestResult,
    DailyPrice,
    Dividend,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
    PortfolioBacktestResult,
    PortfolioTrade,
    StockInfo,
    StockValuation,
    TechnicalIndicator,
    Trade,
)

logger = logging.getLogger(__name__)


def _get_last_date(model, stock_id: str) -> str | None:
    """查詢某股票在指定表中的最後一筆日期，用於增量更新。"""
    with get_session() as session:
        result = session.execute(select(func.max(model.date)).where(model.stock_id == stock_id)).scalar()
        if result:
            return result.isoformat()
    return None


def _upsert_batch(model, df: pd.DataFrame, conflict_keys: list[str], batch_size: int = 80) -> int:
    """將 DataFrame 分批寫入指定表（衝突時略過）。

    SQLite 有 SQL 變數上限，必須分批 INSERT。
    """
    if df.empty:
        return 0

    records = df.to_dict("records")
    with get_session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            stmt = sqlite_upsert(model).values(batch)
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_keys)
            session.execute(stmt)
        session.commit()
    return len(records)


def _upsert_daily_price(df: pd.DataFrame) -> int:
    """將日K線 DataFrame 寫入 daily_price 表（衝突時略過）。"""
    return _upsert_batch(DailyPrice, df, ["stock_id", "date"])


def _upsert_institutional(df: pd.DataFrame) -> int:
    """將三大法人 DataFrame 寫入 institutional_investor 表。"""
    return _upsert_batch(InstitutionalInvestor, df, ["stock_id", "date", "name"])


def _upsert_margin(df: pd.DataFrame) -> int:
    """將融資融券 DataFrame 寫入 margin_trading 表。"""
    return _upsert_batch(MarginTrading, df, ["stock_id", "date"])


def _upsert_monthly_revenue(df: pd.DataFrame) -> int:
    """將月營收 DataFrame 寫入 monthly_revenue 表。"""
    return _upsert_batch(MonthlyRevenue, df, ["stock_id", "date"])


def _upsert_dividend(df: pd.DataFrame) -> int:
    """將股利 DataFrame 寫入 dividend 表。"""
    return _upsert_batch(Dividend, df, ["stock_id", "date"])


def _upsert_valuation(df: pd.DataFrame) -> int:
    """將估值 DataFrame 寫入 stock_valuation 表。"""
    return _upsert_batch(StockValuation, df, ["stock_id", "date"])


def _upsert_announcement(df: pd.DataFrame) -> int:
    """將 MOPS 公告 DataFrame 寫入 announcement 表。"""
    return _upsert_batch(Announcement, df, ["stock_id", "date", "seq"])


def sync_valuation_for_stocks(stock_ids: list[str]) -> int:
    """為指定股票補抓最新估值資料（PE/PB/殖利率）。

    用於 discover value 模式：粗篩後候選股約 150 支，
    在細評前自動從 FinMind 補抓估值資料。

    Args:
        stock_ids: 要補抓的股票代號清單

    Returns:
        新增的估值筆數
    """
    fetcher = FinMindFetcher()
    total = 0
    start = (date.today() - timedelta(days=30)).isoformat()
    end = date.today().isoformat()
    skipped = 0

    for sid in stock_ids:
        last = _get_last_date(StockValuation, sid)
        # 如果 DB 已有 7 天內的資料，跳過
        if last and (date.today() - date.fromisoformat(last)).days < 7:
            skipped += 1
            continue
        try:
            df = fetcher.fetch_per_pbr(sid, last or start, end)
            total += _upsert_valuation(df)
        except Exception:
            logger.warning("[%s] 估值資料補抓失敗，跳過", sid)

    if skipped:
        logger.info("[估值補抓] 跳過 %d 支（DB 已有近期資料）", skipped)
    return total


def sync_revenue_for_stocks(stock_ids: list[str]) -> int:
    """為指定股票補抓最新月營收（跳過 DB 已有近期資料的）。

    用於 discover 全市場掃描：粗篩後候選股約 150 支，
    在細評前自動從 FinMind 補抓月營收，讓基本面分數不再 fallback 到 0.5。

    Args:
        stock_ids: 要補抓營收的股票代號清單

    Returns:
        新增的月營收筆數
    """
    fetcher = FinMindFetcher()
    total = 0
    start = (date.today() - timedelta(days=180)).isoformat()
    end = date.today().isoformat()
    skipped = 0

    for sid in stock_ids:
        last = _get_last_date(MonthlyRevenue, sid)
        # 如果 DB 已有 60 天內的資料，跳過（避免重複抓取）
        if last and (date.today() - date.fromisoformat(last)).days < 60:
            skipped += 1
            continue
        try:
            df = fetcher.fetch_monthly_revenue(sid, last or start, end)
            total += _upsert_monthly_revenue(df)
        except Exception:
            logger.warning("[%s] 月營收補抓失敗，跳過", sid)

    if skipped:
        logger.info("[營收補抓] 跳過 %d 支（DB 已有近期資料）", skipped)
    return total


def sync_mops_announcements(days: int = 7) -> int:
    """同步最近 N 個交易日的 MOPS 重大訊息公告。

    Args:
        days: 回溯天數（預設 7 天）

    Returns:
        新增的公告筆數
    """
    from src.data.mops_fetcher import fetch_mops_announcements

    init_db()
    total = 0
    d = date.today()
    success_count = 0
    max_attempts = days + 10  # 預留假日空間

    logger.info("[MOPS] 同步重大訊息，目標 %d 個交易日", days)

    for _ in range(max_attempts):
        if success_count >= days:
            break
        if d.weekday() >= 5:
            d -= timedelta(days=1)
            continue

        df = fetch_mops_announcements(d)
        if not df.empty:
            count = _upsert_announcement(df)
            total += count
            success_count += 1
            logger.info("[MOPS] %s: %d 筆公告", d.isoformat(), count)
        else:
            success_count += 1  # 假日也算一個交易日嘗試

        d -= timedelta(days=1)

    logger.info("[MOPS] 同步完成 — 共 %d 筆公告", total)
    return total


def sync_stock(
    stock_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    fetcher: FinMindFetcher | None = None,
) -> dict[str, int]:
    """同步單一股票的所有資料（日K + 三大法人 + 融資融券）。

    支援增量更新：若 DB 已有資料，自動從最後一筆日期開始抓取。

    Returns:
        dict: 各資料表新增筆數，例如 {"daily_price": 100, "institutional": 300, "margin": 100}
    """
    if fetcher is None:
        fetcher = FinMindFetcher()

    default_start = start_date or settings.fetcher.default_start_date
    if end_date is None:
        end_date = date.today().isoformat()

    result = {}

    # --- 日K線 ---
    last = _get_last_date(DailyPrice, stock_id)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步日K線: %s ~ %s", stock_id, s, end_date)
    df_price = fetcher.fetch_daily_price(stock_id, s, end_date)
    result["daily_price"] = _upsert_daily_price(df_price)

    # --- 三大法人 ---
    last = _get_last_date(InstitutionalInvestor, stock_id)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步三大法人: %s ~ %s", stock_id, s, end_date)
    df_inst = fetcher.fetch_institutional(stock_id, s, end_date)
    result["institutional"] = _upsert_institutional(df_inst)

    # --- 融資融券 ---
    last = _get_last_date(MarginTrading, stock_id)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步融資融券: %s ~ %s", stock_id, s, end_date)
    df_margin = fetcher.fetch_margin_trading(stock_id, s, end_date)
    result["margin"] = _upsert_margin(df_margin)

    # --- 月營收 ---
    last = _get_last_date(MonthlyRevenue, stock_id)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步月營收: %s ~ %s", stock_id, s, end_date)
    df_rev = fetcher.fetch_monthly_revenue(stock_id, s, end_date)
    result["revenue"] = _upsert_monthly_revenue(df_rev)

    # --- 股利 ---
    last = _get_last_date(Dividend, stock_id)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步股利: %s ~ %s", stock_id, s, end_date)
    df_div = fetcher.fetch_dividend(stock_id, s, end_date)
    result["dividend"] = _upsert_dividend(df_div)

    return result


def sync_watchlist(
    watchlist: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, dict[str, int]]:
    """批次同步關注清單中所有股票的資料。"""
    if watchlist is None:
        watchlist = settings.fetcher.watchlist

    init_db()
    fetcher = FinMindFetcher()

    all_results = {}
    for stock_id in watchlist:
        logger.info("=" * 50)
        logger.info("開始同步: %s", stock_id)
        try:
            all_results[stock_id] = sync_stock(stock_id, start_date, end_date, fetcher)
            logger.info("[%s] 完成 — %s", stock_id, all_results[stock_id])
        except Exception:
            logger.exception("[%s] 同步失敗", stock_id)
            all_results[stock_id] = {"error": True}

    return all_results


def sync_stock_info(force_refresh: bool = False) -> int:
    """同步全市場股票基本資料（產業分類）到 stock_info 表。

    Args:
        force_refresh: True 時強制重新抓取，否則 DB 已有資料就跳過

    Returns:
        新增/更新的筆數
    """
    init_db()

    if not force_refresh:
        with get_session() as session:
            count = session.execute(select(func.count()).select_from(StockInfo)).scalar()
            if count and count > 0:
                logger.info("[StockInfo] DB 已有 %d 筆，跳過同步（使用 force_refresh=True 強制更新）", count)
                return 0

    fetcher = FinMindFetcher()
    df = fetcher.fetch_stock_info()
    if df.empty:
        logger.warning("[StockInfo] 未取得任何資料")
        return 0

    records = df.to_dict("records")
    with get_session() as session:
        for i in range(0, len(records), 80):
            batch = records[i : i + 80]
            stmt = sqlite_upsert(StockInfo).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_id"],
                set_={
                    "stock_name": stmt.excluded.stock_name,
                    "industry_category": stmt.excluded.industry_category,
                    "listing_type": stmt.excluded.listing_type,
                },
            )
            session.execute(stmt)
        session.commit()

    logger.info("[StockInfo] 已同步 %d 筆股票基本資料", len(records))
    return len(records)


def sync_taiex_index(
    start_date: str | None = None,
    end_date: str | None = None,
    fetcher: FinMindFetcher | None = None,
) -> int:
    """同步加權指數資料（用於 benchmark）。"""
    init_db()

    if fetcher is None:
        fetcher = FinMindFetcher()

    default_start = start_date or settings.fetcher.default_start_date
    if end_date is None:
        end_date = date.today().isoformat()

    last = _get_last_date(DailyPrice, "TAIEX")
    s = last if last and last > default_start else default_start

    logger.info("[TAIEX] 同步加權指數: %s ~ %s", s, end_date)
    df = fetcher.fetch_taiex_index(s, end_date)
    count = _upsert_daily_price(df)
    logger.info("[TAIEX] 完成 — %d 筆", count)
    return count


def sync_market_data(
    days: int = 10,
    fetcher: FinMindFetcher | None = None,
    max_stocks: int = 200,
) -> dict[str, int]:
    """同步全市場資料（日K + 三大法人 + 融資融券），用於 discover 掃描。

    資料來源優先順序：
    1. TWSE/TPEX 官方開放資料（免費，6 次 API 取得全市場）
    2. FinMind 批次 API（需付費帳號）
    3. FinMind 逐股抓取（免費帳號備案，較慢）

    Args:
        days: 抓取最近 N 天的資料
        fetcher: 可注入 FinMind fetcher 實例（用於備案策略）
        max_stocks: 備案策略最多抓取的股票數

    Returns:
        dict: {"daily_price": N, "institutional": M, "margin": K}
    """
    from src.data.twse_fetcher import (
        fetch_market_daily_prices,
        fetch_market_institutional,
        fetch_market_margin,
    )

    init_db()
    result = {"daily_price": 0, "institutional": 0, "margin": 0}

    # --- 策略 1：TWSE/TPEX 官方資料（免費、快速） ---
    end = date.today()

    # 從今天往前找，跳過週末，直到抓到 days 個有資料的交易日
    # （假日時 API 回傳空資料，自動往前找）
    d = end
    success_count = 0
    max_attempts = days + 20  # 預留假日空間
    attempts = 0

    logger.info("[全市場] 使用 TWSE/TPEX 官方資料，目標 %d 個交易日", days)

    while success_count < days and attempts < max_attempts:
        attempts += 1
        if d.weekday() >= 5:  # 跳過週末
            d -= timedelta(days=1)
            continue

        logger.info("[全市場] 抓取 %s ...", d.isoformat())

        df_price = fetch_market_daily_prices(d)
        if not df_price.empty:
            success_count += 1
            result["daily_price"] += _upsert_daily_price(df_price)

            df_inst = fetch_market_institutional(d)
            if not df_inst.empty:
                result["institutional"] += _upsert_institutional(df_inst)

            df_margin = fetch_market_margin(d)
            if not df_margin.empty:
                result["margin"] += _upsert_margin(df_margin)
        else:
            logger.info("[全市場] %s 無資料（假日），跳過", d.isoformat())

        d -= timedelta(days=1)

    # --- MOPS 重大訊息同步（附加於全市場同步，失敗不影響其他資料） ---
    try:
        from src.data.mops_fetcher import fetch_mops_announcements

        result["announcements"] = 0
        mops_d = date.today()
        for _ in range(min(days, 5)):  # MOPS 最多抓 5 天，避免太慢
            if mops_d.weekday() >= 5:
                mops_d -= timedelta(days=1)
                continue
            df_ann = fetch_mops_announcements(mops_d)
            if not df_ann.empty:
                result["announcements"] += _upsert_announcement(df_ann)
            mops_d -= timedelta(days=1)
        if result["announcements"]:
            logger.info("[全市場] MOPS 重訊: %d 筆", result["announcements"])
    except Exception:
        logger.warning("[全市場] MOPS 重訊同步失敗，不影響其他資料")

    if success_count > 0:
        logger.info(
            "[全市場] TWSE/TPEX 同步完成 — %d 個交易日, 日K %d 筆, 法人 %d 筆, 融資融券 %d 筆",
            success_count,
            result["daily_price"],
            result["institutional"],
            result["margin"],
        )
        return result

    # --- 策略 2：FinMind 批次 API（付費帳號） ---
    if fetcher is None:
        fetcher = FinMindFetcher()

    start = end - timedelta(days=days)
    start_str = start.isoformat()
    end_str = end.isoformat()

    logger.info("[全市場] TWSE/TPEX 失敗，嘗試 FinMind 批次 API: %s ~ %s", start_str, end_str)
    df_price = fetcher.fetch_all_daily_price(start_str, end_str)

    if not df_price.empty:
        result["daily_price"] = _upsert_daily_price(df_price)
        df_inst = fetcher.fetch_all_institutional(start_str, end_str)
        result["institutional"] = _upsert_institutional(df_inst)
        logger.info(
            "[全市場] FinMind 批次完成 — 日K %d 筆, 法人 %d 筆",
            result["daily_price"],
            result["institutional"],
        )
        return result

    # --- 策略 3：FinMind 逐股抓取（免費帳號備案） ---
    logger.info("[全市場] 所有批次來源不可用，改用 FinMind 逐股抓取（上限 %d 支）", max_stocks)

    with get_session() as session:
        rows = (
            session.execute(select(StockInfo.stock_id).where(StockInfo.listing_type.in_(["twse", "tpex"])))
            .scalars()
            .all()
        )

    if not rows:
        sync_stock_info(force_refresh=True)
        with get_session() as session:
            rows = (
                session.execute(select(StockInfo.stock_id).where(StockInfo.listing_type.in_(["twse", "tpex"])))
                .scalars()
                .all()
            )

    stock_ids = [sid for sid in rows if sid.isdigit() and len(sid) == 4]
    stock_ids = stock_ids[:max_stocks]
    total = len(stock_ids)
    logger.info("[全市場] 逐股抓取 %d 支", total)

    start_str = (end - timedelta(days=days)).isoformat()
    end_str = end.isoformat()

    for i, sid in enumerate(stock_ids, 1):
        try:
            if i % 20 == 0 or i == total:
                logger.info("[全市場] 進度: %d/%d", i, total)
            df_p = fetcher.fetch_daily_price(sid, start_str, end_str)
            result["daily_price"] += _upsert_daily_price(df_p)
            df_i = fetcher.fetch_institutional(sid, start_str, end_str)
            result["institutional"] += _upsert_institutional(df_i)
        except Exception:
            logger.warning("[%s] 抓取失敗，跳過", sid)

    logger.info(
        "[全市場] 逐股抓取完成 — 日K %d 筆, 法人 %d 筆",
        result["daily_price"],
        result["institutional"],
    )
    return result


# ------------------------------------------------------------------ #
#  P1: 技術指標計算
# ------------------------------------------------------------------ #


def _upsert_indicators(df: pd.DataFrame) -> int:
    """將技術指標 DataFrame 寫入 technical_indicator 表。"""
    return _upsert_batch(TechnicalIndicator, df, ["stock_id", "date", "name"])


def sync_indicators(
    watchlist: list[str] | None = None,
) -> dict[str, int]:
    """計算關注清單中所有股票的技術指標並寫入 DB。"""
    from src.features.indicators import compute_indicators

    if watchlist is None:
        watchlist = settings.fetcher.watchlist

    init_db()

    all_results = {}
    for stock_id in watchlist:
        logger.info("=" * 50)
        logger.info("計算指標: %s", stock_id)
        try:
            df = compute_indicators(stock_id)
            count = _upsert_indicators(df)
            all_results[stock_id] = count
            logger.info("[%s] 完成 — %d 筆指標", stock_id, count)
        except Exception:
            logger.exception("[%s] 指標計算失敗", stock_id)
            all_results[stock_id] = -1

    return all_results


# ------------------------------------------------------------------ #
#  P2: 回測結果存入 DB
# ------------------------------------------------------------------ #


def save_backtest_result(result_data) -> int:
    """將回測結果與交易明細寫入 DB，回傳 backtest_result.id。"""
    init_db()

    with get_session() as session:
        # 寫入回測摘要
        bt = BacktestResult(
            stock_id=result_data.stock_id,
            strategy_name=result_data.strategy_name,
            start_date=result_data.start_date,
            end_date=result_data.end_date,
            initial_capital=result_data.initial_capital,
            final_capital=result_data.final_capital,
            total_return=result_data.total_return,
            annual_return=result_data.annual_return,
            sharpe_ratio=result_data.sharpe_ratio,
            max_drawdown=result_data.max_drawdown,
            win_rate=result_data.win_rate,
            total_trades=result_data.total_trades,
            benchmark_return=getattr(result_data, "benchmark_return", None),
            sortino_ratio=getattr(result_data, "sortino_ratio", None),
            calmar_ratio=getattr(result_data, "calmar_ratio", None),
            var_95=getattr(result_data, "var_95", None),
            cvar_95=getattr(result_data, "cvar_95", None),
            profit_factor=getattr(result_data, "profit_factor", None),
        )
        session.add(bt)
        session.flush()  # 取得 id
        bt_id = bt.id

        # 寫入交易明細
        for t in result_data.trades:
            trade = Trade(
                backtest_id=bt_id,
                entry_date=t.entry_date,
                entry_price=t.entry_price,
                exit_date=t.exit_date,
                exit_price=t.exit_price,
                shares=t.shares,
                pnl=t.pnl,
                return_pct=t.return_pct,
                exit_reason=getattr(t, "exit_reason", None),
            )
            session.add(trade)

        session.commit()
        logger.info("回測結果已儲存 (id=%d, %d 筆交易)", bt_id, len(result_data.trades))

    return bt_id


def save_portfolio_result(result_data) -> int:
    """將投資組合回測結果與交易明細寫入 DB，回傳 portfolio_backtest_result.id。"""
    init_db()

    with get_session() as session:
        pbt = PortfolioBacktestResult(
            strategy_name=result_data.strategy_name,
            stock_ids=",".join(result_data.stock_ids),
            start_date=result_data.start_date,
            end_date=result_data.end_date,
            initial_capital=result_data.initial_capital,
            final_capital=result_data.final_capital,
            total_return=result_data.total_return,
            annual_return=result_data.annual_return,
            sharpe_ratio=result_data.sharpe_ratio,
            max_drawdown=result_data.max_drawdown,
            win_rate=result_data.win_rate,
            total_trades=result_data.total_trades,
            sortino_ratio=getattr(result_data, "sortino_ratio", None),
            calmar_ratio=getattr(result_data, "calmar_ratio", None),
            var_95=getattr(result_data, "var_95", None),
            cvar_95=getattr(result_data, "cvar_95", None),
            profit_factor=getattr(result_data, "profit_factor", None),
            allocation_method=getattr(result_data, "allocation_method", None),
        )
        session.add(pbt)
        session.flush()
        pbt_id = pbt.id

        for t in result_data.trades:
            trade = PortfolioTrade(
                portfolio_backtest_id=pbt_id,
                stock_id=t.stock_id,
                entry_date=t.entry_date,
                entry_price=t.entry_price,
                exit_date=t.exit_date,
                exit_price=t.exit_price,
                shares=t.shares,
                pnl=t.pnl,
                return_pct=t.return_pct,
                exit_reason=getattr(t, "exit_reason", None),
            )
            session.add(trade)

        session.commit()
        logger.info("投資組合回測結果已儲存 (id=%d, %d 筆交易)", pbt_id, len(result_data.trades))

    return pbt_id
