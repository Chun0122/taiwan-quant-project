"""ETL Pipeline — 整合 抓取 → 清洗 → 寫入資料庫 流程。"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

from src.config import settings
from src.data.database import get_effective_watchlist, get_session, init_db
from src.data.fetcher import FinMindFetcher
from src.data.schema import (
    Announcement,
    BacktestResult,
    BrokerTrade,
    DailyFeature,
    DailyPrice,
    Dividend,
    FinancialStatement,
    HoldingDistribution,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
    PortfolioBacktestResult,
    PortfolioTrade,
    SecuritiesLending,
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


def _batch_get_last_dates(model, stock_ids: list[str]) -> dict[str, str | None]:
    """一次查詢多支股票在指定表中的最後日期。回傳 {stock_id: 'YYYY-MM-DD' | None}。"""
    if not stock_ids:
        return {}
    with get_session() as session:
        rows = session.execute(
            select(model.stock_id, func.max(model.date)).where(model.stock_id.in_(stock_ids)).group_by(model.stock_id)
        ).all()
    last_map = {r[0]: r[1].isoformat() for r in rows if r[1] is not None}
    return {sid: last_map.get(sid) for sid in stock_ids}


def _upsert_batch(model, df: pd.DataFrame, conflict_keys: list[str], batch_size: int = 80) -> int:
    """將 DataFrame 分批寫入指定表（衝突時略過）。

    SQLite 有 SQL 變數上限，必須分批 INSERT。
    """
    if df.empty:
        return 0

    # 先轉為 dict，再逐欄清理 NaN / NaT → None
    # 注意：df.where(notna, None) 對 datetime64[ns] 欄位無效（None 會被轉回 NaT）
    # 因此改用 to_dict 後逐欄位判斷
    raw_records = df.to_dict("records")
    records = [{k: None if pd.isna(v) else v for k, v in row.items()} for row in raw_records]
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
    if df.empty:
        return 0
    # 移除 year 為 NaN 的行（year 是 nullable=False，不能為空）
    df = df.dropna(subset=["year"])
    if df.empty:
        return 0
    return _upsert_batch(Dividend, df, ["stock_id", "date"])


def _upsert_valuation(df: pd.DataFrame) -> int:
    """將估值 DataFrame 寫入 stock_valuation 表。"""
    return _upsert_batch(StockValuation, df, ["stock_id", "date"])


def _upsert_sbl(df: pd.DataFrame) -> int:
    """將借券賣出 DataFrame 寫入 securities_lending 表。"""
    return _upsert_batch(SecuritiesLending, df, ["stock_id", "date"])


def _upsert_broker_trade(df: pd.DataFrame) -> int:
    """將分點交易 DataFrame 寫入 broker_trade 表。"""
    return _upsert_batch(BrokerTrade, df, ["stock_id", "date", "broker_id"])


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

    last_dates = _batch_get_last_dates(StockValuation, stock_ids)
    for sid in stock_ids:
        last = last_dates.get(sid)
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

    last_dates = _batch_get_last_dates(MonthlyRevenue, stock_ids)
    for sid in stock_ids:
        last = last_dates.get(sid)
        # 如果 DB 已有 30 天內的資料，跳過（月營收每月公布，確保月份更新時能重新補抓）
        if last and (date.today() - date.fromisoformat(last)).days < 30:
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


def _upsert_financial(df: pd.DataFrame) -> int:
    """將財報 DataFrame 寫入 financial_statement 表。"""
    return _upsert_batch(FinancialStatement, df, ["stock_id", "date"])


def sync_financial_statements(
    watchlist: list[str] | None = None,
    quarters: int = 4,
) -> int:
    """同步 watchlist 財報資料（最近 N 季）。

    Args:
        watchlist: 股票代號清單（預設使用 settings.fetcher.watchlist）
        quarters: 抓取最近幾季（預設 4 季 = 1 年）

    Returns:
        新增的財報筆數
    """
    if watchlist is None:
        watchlist = get_effective_watchlist()

    init_db()
    fetcher = FinMindFetcher()

    # 每季約 90 天，多抓一點確保覆蓋
    start = (date.today() - timedelta(days=quarters * 95 + 30)).isoformat()
    end = date.today().isoformat()
    total = 0
    skipped = 0

    last_dates = _batch_get_last_dates(FinancialStatement, watchlist)
    for sid in watchlist:
        last = last_dates.get(sid)
        # 如果 DB 已有 60 天內的資料，跳過
        if last and (date.today() - date.fromisoformat(last)).days < 60:
            skipped += 1
            continue

        try:
            df = fetcher.fetch_financial_summary(sid, last or start, end)
            total += _upsert_financial(df)
        except Exception:
            logger.warning("[%s] 財報同步失敗，跳過", sid)

    if skipped:
        logger.info("[財報同步] 跳過 %d 支（DB 已有近期資料）", skipped)
    logger.info("[財報同步] 完成，共寫入 %d 筆", total)
    return total


def sync_financial_for_stocks(stock_ids: list[str], quarters: int = 4) -> int:
    """為指定股票補抓財報資料（供 discover 或其他模組用）。

    Args:
        stock_ids: 要補抓財報的股票代號清單
        quarters: 抓取最近幾季

    Returns:
        新增的財報筆數
    """
    fetcher = FinMindFetcher()
    total = 0
    start = (date.today() - timedelta(days=quarters * 95 + 30)).isoformat()
    end = date.today().isoformat()
    skipped = 0

    last_dates = _batch_get_last_dates(FinancialStatement, stock_ids)
    for sid in stock_ids:
        last = last_dates.get(sid)
        # 如果 DB 已有 60 天內的資料，跳過
        if last and (date.today() - date.fromisoformat(last)).days < 60:
            skipped += 1
            continue
        try:
            df = fetcher.fetch_financial_summary(sid, last or start, end)
            total += _upsert_financial(df)
        except Exception:
            logger.warning("[%s] 財報補抓失敗，跳過", sid)

    if skipped:
        logger.info("[財報補抓] 跳過 %d 支（DB 已有近期資料）", skipped)
    return total


def _upsert_holding(df: pd.DataFrame) -> int:
    """將持股分級 DataFrame 寫入 holding_distribution 表。"""
    return _upsert_batch(HoldingDistribution, df, ["stock_id", "date", "level"])


def sync_holding_distribution(
    watchlist: list[str] | None = None,  # noqa: ARG001 — 保留相容性，實際存全市場
    weeks: int = 4,  # noqa: ARG001 — 保留參數相容性，TDCC 僅提供最新一週
) -> int:
    """同步全市場大戶持股分級資料（最新一週，TDCC）。

    資料來源：TDCC 集保戶股權分散表（免費開放，一次取全市場 ~2928 支）。
    每週更新一次，若 DB 已有 7 天內的資料（任意股票），自動跳過。

    注意：TDCC 一次抓全市場，存全市場（不限 watchlist），讓 discover 全市場掃描
    也能使用大戶資料（Stage 3 whale 因子，觸發 7F/8F）。

    Args:
        watchlist: 保留參數（實際存全市場）
        weeks:     保留參數（TDCC 僅提供最新一週，歷史靠每週累積）

    Returns:
        新增的持股分級筆數
    """
    from src.data.twse_fetcher import fetch_tdcc_holding_all_market

    init_db()

    # 快速跳過：DB 已有 7 天內任意持股分級資料則跳過（TDCC 全市場一次性同步）
    with get_session() as session:
        recent_count = session.execute(
            select(func.count())
            .select_from(HoldingDistribution)
            .where(HoldingDistribution.date >= (date.today() - timedelta(days=7)))
        ).scalar_one()
    if recent_count > 0:
        logger.info("[持股分級] DB 已有 7 天內資料（%d 筆），跳過同步", recent_count)
        return 0

    # 一次抓全市場最新一週，存全部（~2928 支 × 15 tier ≈ 43,920 筆）
    df_all = fetch_tdcc_holding_all_market()
    if df_all.empty:
        logger.warning("[持股分級] TDCC 回傳空資料")
        return 0

    total = _upsert_holding(df_all)
    logger.info(
        "[持股分級] 完成，共寫入 %d 筆（%d 支股票）",
        total,
        df_all["stock_id"].nunique(),
    )
    return total


def sync_mops_announcements(days: int = 7) -> int:
    """同步 MOPS 最新重大訊息公告。

    MOPS 備援站僅提供最新一個交易日的公告，因此每次呼叫只會抓取
    一天的資料。建議搭配每日排程使用，逐日累積歷史公告。

    Args:
        days: 未使用（保留以維持 CLI 相容），實際只抓取最新一天

    Returns:
        新增的公告筆數
    """
    from src.data.mops_fetcher import fetch_mops_announcements

    init_db()

    logger.info("[MOPS] 同步最新重大訊息")

    df = fetch_mops_announcements()
    if df.empty:
        logger.info("[MOPS] 無公告資料")
        return 0

    total = _upsert_announcement(df)
    actual_date = df["date"].iloc[0] if not df.empty else "N/A"
    logger.info("[MOPS] 同步完成 — %s: %d 筆公告", actual_date, total)
    return total


def sync_mops_revenue(months: int = 1) -> int:
    """從 MOPS 同步全市場月營收（上市+上櫃）。

    使用 MOPS 公開資訊觀測站的靜態 HTML 頁面，
    兩次 HTTP 請求即可取得全市場 ~2000+ 支股票的月營收。

    Args:
        months: 同步最近幾個月的營收（預設 1 = 上月）

    Returns:
        新增的月營收筆數
    """
    from src.data.mops_fetcher import fetch_mops_monthly_revenue

    init_db()

    total = 0
    today = date.today()

    for i in range(months):
        # 計算目標月份（從上月往回推）
        target = today.replace(day=1) - timedelta(days=1)  # 上月底
        for _ in range(i):
            target = target.replace(day=1) - timedelta(days=1)  # 再往前推
        target_year = target.year
        target_month = target.month

        # 檢查 DB 是否已有該月份全市場資料
        with get_session() as session:
            count = session.execute(
                select(func.count())
                .select_from(MonthlyRevenue)
                .where(
                    MonthlyRevenue.revenue_year == target_year,
                    MonthlyRevenue.revenue_month == target_month,
                )
            ).scalar()

        if count and count >= 500:
            logger.info(
                "[MOPS 月營收] %d/%d 已有 %d 筆（跳過）",
                target_year,
                target_month,
                count,
            )
            continue

        df = fetch_mops_monthly_revenue(year=target_year, month=target_month)
        if df.empty:
            continue

        n = _upsert_monthly_revenue(df)
        total += n
        logger.info(
            "[MOPS 月營收] %d/%d 寫入 %d 筆",
            target_year,
            target_month,
            n,
        )

    logger.info("[MOPS 月營收] 同步完成，共寫入 %d 筆", total)
    return total


def sync_valuation_all_market() -> int:
    """從 TWSE/TPEX 同步全市場估值資料（PE/PB/殖利率）。

    使用 TWSE BWIBBU_d + TPEX pera 端點，
    兩次 HTTP 請求即可取得全市場 ~1700+ 支股票的估值資料。
    免費、無需 FinMind token。

    用於 ValueScanner / DividendScanner 的 Stage 0.5 cold-start 補抓。

    Returns:
        新增的估值筆數
    """
    from src.data.twse_fetcher import _find_last_trading_day, fetch_market_valuation_all

    init_db()

    # 找最近一個交易日（避免週末/假日無資料）
    target = _find_last_trading_day(date.today())

    # 若 DB 已有該日期足夠資料，跳過
    with get_session() as session:
        count = session.execute(
            select(func.count()).select_from(StockValuation).where(StockValuation.date == target)
        ).scalar()

    if count and count >= 500:
        logger.info("[全市場估值] %s 已有 %d 筆（跳過）", target.isoformat(), count)
        return 0

    df = fetch_market_valuation_all(target)

    # 非交易日 fallback：往前找最多 7 天，直到取到資料或確認 DB 已有舊資料
    if df.empty:
        logger.warning("[全市場估值] %s 無資料，往前尋找最近有效資料...", target.isoformat())
        for days_back in range(1, 8):
            alt = target - timedelta(days=days_back)
            if alt.weekday() >= 5:
                continue
            with get_session() as session:
                alt_count = session.execute(
                    select(func.count()).select_from(StockValuation).where(StockValuation.date == alt)
                ).scalar()
            if alt_count and alt_count >= 500:
                logger.info("[全市場估值] %s 已有 %d 筆（使用既有資料）", alt.isoformat(), alt_count)
                return 0
            df = fetch_market_valuation_all(alt)
            if not df.empty:
                break

    if df.empty:
        logger.warning("[全市場估值] 無法取得全市場估值資料")
        return 0

    n = _upsert_valuation(df)
    logger.info("[全市場估值] 寫入 %d 筆估值資料", n)
    return n


def sync_sbl_all_market(days: int = 3) -> int:
    """從 TWSE 同步全市場借券賣出彙總（日資料，TWT96U）。

    最近 days 個交易日逐日抓取，若 DB 當日已有 >= 500 筆則跳過。

    Args:
        days: 同步最近幾個交易日（預設 3）

    Returns:
        新增的借券筆數
    """
    from src.data.twse_fetcher import _find_last_trading_day, fetch_twse_sbl

    init_db()
    total = 0
    target = _find_last_trading_day(date.today())

    for i in range(days):
        d = target - timedelta(days=i)
        # 跳過週末
        if d.weekday() >= 5:
            continue

        with get_session() as session:
            count = session.execute(
                select(func.count()).select_from(SecuritiesLending).where(SecuritiesLending.date == d)
            ).scalar()

        if count and count >= 500:
            logger.info("[全市場借券] %s 已有 %d 筆（跳過）", d.isoformat(), count)
            continue

        df = fetch_twse_sbl(d)
        if df.empty:
            logger.warning("[全市場借券] %s 無資料", d.isoformat())
            continue

        n = _upsert_sbl(df)
        logger.info("[全市場借券] %s 寫入 %d 筆", d.isoformat(), n)
        total += n

    return total


def sync_broker_trades(
    stock_ids: list[str] | None = None,
    days: int = 5,
) -> int:
    """同步分點交易資料（DJ 分點端點，免費，支援日期範圍）。

    若 DB 已有 2 天內資料則跳過該股票（避免重複抓取）。
    每次 API 呼叫取得 start~end 期間彙整，date 欄位統一為 end（今日）。
    速率控制由 fetch_dj_broker_trades() 內部處理（3 秒間隔）。

    Args:
        stock_ids: 指定股票代號清單，預設使用 watchlist
        days:      查詢最近幾個交易日的彙整範圍（預設 5）

    Returns:
        新增的分點交易筆數
    """
    from src.data.twse_fetcher import fetch_dj_broker_trades

    if stock_ids is None:
        stock_ids = get_effective_watchlist()

    init_db()
    end_date = date.today()
    start_date = end_date - timedelta(days=days + 3)

    total = 0
    last_dates = _batch_get_last_dates(BrokerTrade, stock_ids)
    for sid in stock_ids:
        latest_str = last_dates.get(sid)
        latest = date.fromisoformat(latest_str) if latest_str else None
        if latest and (date.today() - latest).days < 2:
            logger.info("[分點] %s 已有最新資料（%s），跳過", sid, latest)
            continue

        df = fetch_dj_broker_trades(sid, start_date, end_date)

        if not df.empty:
            n = _upsert_broker_trade(df)
            total += n
            logger.info("[分點] %s 寫入 %d 筆", sid, n)

    return total


def sync_broker_for_stocks(stock_ids: list[str]) -> int:
    """為指定股票補抓最新分點交易資料（跳過 DB 已有近期資料的）。

    用於 discover momentum 模式：粗篩後候選股約 150 支，
    在細評前自動從 FinMind 補抓分點資料，讓籌碼面分點因子能正確評分。
    使用 days=7 覆蓋 _load_broker_data() 所需的 7 天查詢窗口。

    Args:
        stock_ids: 要補抓的股票代號清單

    Returns:
        新增的分點交易筆數
    """
    return sync_broker_trades(stock_ids=stock_ids, days=7)


def sync_broker_bootstrap(
    stock_ids: list[str] | None = None,
    days: int = 30,
) -> int:
    """逐日補齊分點交易歷史（Bootstrap 模式，用於啟用 Smart Broker 8F）。

    DJ 端點每次呼叫只回傳期間彙整（date = end），因此普通的 sync_broker_trades()
    無論 days 多大，每次都只增加 1 個 date 記錄。本函數改為對每個交易日分別呼叫
    DJ 端點（start=d, end=d），使每日產生獨立的 date 記錄，累積後達到
    _load_broker_data_extended() 的 min_trading_days=20 門檻，啟用 8F。

    交易日來源：從 DailyPrice 查詢過去 days 天內的實際有成交日期（自動排除假日）。
    若 DailyPrice 無資料，退回使用平日曆法（跳過週末）。

    Args:
        stock_ids: 指定股票清單，預設使用 watchlist
        days:      補齊最近幾個交易日（預設 30，建議 ≥ 20 以啟用 8F）

    Returns:
        新增的分點交易總筆數

    時間估算（30 支 × 30 天 × 3s = 45 分鐘）：僅適合一次性部署使用。
    """
    from src.data.twse_fetcher import fetch_dj_broker_trades

    if stock_ids is None:
        stock_ids = get_effective_watchlist()

    init_db()
    cutoff = date.today() - timedelta(days=days + 5)

    # 取得過去 days 天的實際交易日（從 DailyPrice 查任意有資料的股票）
    trading_dates: list[date] = []
    try:
        with get_session() as session:
            rows = (
                session.execute(
                    select(DailyPrice.date)
                    .where(DailyPrice.date >= cutoff)
                    .group_by(DailyPrice.date)
                    .order_by(DailyPrice.date.desc())
                    .limit(days)
                )
                .scalars()
                .all()
            )
        trading_dates = list(rows)
    except Exception:
        pass

    # Fallback：若 DailyPrice 無資料，或資料天數不足時，用平日曆法補足
    if len(trading_dates) < days:
        # 從最早已知交易日往前補，或從今日開始（完全無資料時）
        earliest = min(trading_dates) if trading_dates else date.today()
        d = earliest - timedelta(days=1)
        while len(trading_dates) < days:
            if d.weekday() < 5:  # 週一至週五
                trading_dates.append(d)
            d -= timedelta(days=1)

    if not trading_dates:
        logger.warning("[Bootstrap] 無法確定交易日，放棄")
        return 0

    logger.info("[Bootstrap] 對 %d 支股票逐日補齊最近 %d 個交易日...", len(stock_ids), len(trading_dates))

    # 一次查詢所有已存在的 (stock_id, date) 對，避免雙層迴圈內 N×M 次 EXISTS 查詢
    with get_session() as session:
        existing_pairs: set[tuple] = set(
            session.execute(
                select(BrokerTrade.stock_id, BrokerTrade.date).where(BrokerTrade.stock_id.in_(stock_ids))
            ).all()
        )

    total = 0
    for sid in stock_ids:
        sid_total = 0
        for trading_day in trading_dates:  # 從新到舊
            # 已存在此日資料則跳過
            if (sid, trading_day) in existing_pairs:
                continue

            df = fetch_dj_broker_trades(sid, trading_day, trading_day)
            if not df.empty:
                n = _upsert_broker_trade(df)
                sid_total += n

        if sid_total > 0:
            logger.info("[Bootstrap] %s 補齊 %d 筆", sid, sid_total)
        total += sid_total

    logger.info("[Bootstrap] 完成，總計新增 %d 筆", total)
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

    # --- 財報 ---
    last = _get_last_date(FinancialStatement, stock_id)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步財報: %s ~ %s", stock_id, s, end_date)
    try:
        df_fin = fetcher.fetch_financial_summary(stock_id, s, end_date)
        result["financial"] = _upsert_financial(df_fin)
    except Exception:
        logger.warning("[%s] 財報同步失敗，跳過", stock_id)
        result["financial"] = 0

    return result


def sync_watchlist(
    watchlist: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, dict[str, int]]:
    """批次同步關注清單中所有股票的資料。"""
    if watchlist is None:
        watchlist = get_effective_watchlist()

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


def _classify_security_type(stock_id: str, stock_name: str = "") -> str:
    """從股票代號與名稱推斷有價證券類型（純函數）。

    分類規則（優先順序）：
    1. 6 位數字開頭 00：ETF（如 0050、00878）
    2. 名稱含 ETF 字樣：ETF
    3. 6 位數字：權證（warrant）
    4. 名稱含「特」：特別股（preferred）
    5. 其餘（4 位數字等）：普通股（stock）

    Args:
        stock_id: 股票代號
        stock_name: 股票名稱（可選）

    Returns:
        "stock" / "etf" / "warrant" / "preferred"
    """
    import re

    sid = str(stock_id).strip()
    name = str(stock_name or "").upper()

    if re.match(r"^00\d{4}$", sid) or "ETF" in name:
        return "etf"
    if len(sid) == 6 and sid.isdigit():
        return "warrant"
    if "特" in (stock_name or ""):
        return "preferred"
    return "stock"


def sync_stock_info(force_refresh: bool = False) -> int:
    """同步全市場股票基本資料（產業分類 + security_type）到 stock_info 表。

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

    # 自動填入 security_type
    df["security_type"] = df.apply(
        lambda row: _classify_security_type(
            row.get("stock_id", ""),
            row.get("stock_name", ""),
        ),
        axis=1,
    )

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
                    "security_type": stmt.excluded.security_type,
                },
            )
            session.execute(stmt)
        session.commit()

    logger.info("[StockInfo] 已同步 %d 筆股票基本資料（含 security_type）", len(records))
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

    # 增量檢查：若 DB 已有近期資料，縮減 days 至實際缺口，避免重抓已有資料
    try:
        from sqlalchemy import func, select

        with get_session() as session:
            latest_in_db = session.execute(select(func.max(DailyPrice.date))).scalar()
        if latest_in_db is not None:
            days_gap = (end - latest_in_db).days
            if days_gap < days:
                logger.info(
                    "[全市場] DB 最新日期 %s，縮減同步目標 %d→%d 天",
                    latest_in_db,
                    days,
                    max(1, days_gap),
                )
                days = max(1, days_gap)
    except Exception:
        pass  # 查詢失敗不影響同步，使用原始 days

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
    # fetch_mops_announcements 固定回傳今天的公告，只需呼叫一次
    try:
        from src.data.mops_fetcher import fetch_mops_announcements

        result["announcements"] = 0
        df_ann = fetch_mops_announcements(date.today())
        if not df_ann.empty:
            result["announcements"] = _upsert_announcement(df_ann)
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
        watchlist = get_effective_watchlist()

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


# ────────────────────────────────────────────────────────────────
#  Feature Store ETL
# ────────────────────────────────────────────────────────────────


def compute_and_store_daily_features(lookback_days: int = 90) -> int:
    """計算並儲存全市場每日特徵到 DailyFeature 表（Feature Store）。

    從 DailyPrice 讀取最近 lookback_days 天資料，以 Pandas 向量化 rolling
    計算：MA20/MA60、均量、均成交金額、動能、波動率。
    只將「最新一日」的特徵寫入 DB（增量更新），避免全量重寫。

    供 UniverseFilter Stage 2/3 使用，加速全市場過濾流程。
    建議每日收盤後由 sync-features 命令呼叫，或整合進 morning-routine。

    Args:
        lookback_days: 讀取多少天的 DailyPrice（至少需 MA60+緩衝 = 80 天）

    Returns:
        寫入 DailyFeature 的筆數
    """
    init_db()

    # 確保至少有足夠歷史計算 MA60
    lookback_days = max(lookback_days, 80)
    cutoff = date.today() - timedelta(days=lookback_days)

    logger.info("[DailyFeature] 讀取近 %d 天 DailyPrice...", lookback_days)
    with get_session() as session:
        rows = session.execute(
            select(
                DailyPrice.stock_id,
                DailyPrice.date,
                DailyPrice.close,
                DailyPrice.volume,
                DailyPrice.turnover,
            ).where(DailyPrice.date >= cutoff)
        ).all()

    if not rows:
        logger.warning("[DailyFeature] 無 DailyPrice 資料可計算")
        return 0

    df = pd.DataFrame(rows, columns=["stock_id", "date", "close", "volume", "turnover"])
    df = df.sort_values(["stock_id", "date"])

    logger.info("[DailyFeature] 共 %d 筆原始資料，開始向量化計算...", len(df))

    # 向量化 rolling 計算（groupby + transform，無 Python for-loop）
    g_close = df.groupby("stock_id")["close"]
    g_vol = df.groupby("stock_id")["volume"]
    g_turnover = df.groupby("stock_id")["turnover"]

    df["ma20"] = g_close.transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["ma60"] = g_close.transform(lambda s: s.rolling(60, min_periods=30).mean())
    df["volume_ma20"] = g_vol.transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["turnover_ma5"] = g_turnover.transform(lambda s: s.rolling(5, min_periods=3).mean())

    # 20 日報酬率 (%)
    df["momentum_20d"] = g_close.transform(lambda s: s.pct_change(20) * 100)

    # 20 日年化波動率 (%)
    df["volatility_20d"] = g_close.transform(
        lambda s: s.pct_change().rolling(20, min_periods=10).std() * (252**0.5) * 100
    )

    # 只取最新一日（增量更新策略）
    latest_date = df["date"].max()
    df_latest = df[df["date"] == latest_date].copy()
    df_latest["computed_at"] = pd.Timestamp.utcnow()

    keep_cols = [
        "stock_id",
        "date",
        "close",
        "volume",
        "turnover",
        "ma20",
        "ma60",
        "volume_ma20",
        "turnover_ma5",
        "momentum_20d",
        "volatility_20d",
        "computed_at",
    ]
    df_out = df_latest[keep_cols].reset_index(drop=True)

    written = _upsert_batch(DailyFeature, df_out, ["stock_id", "date"])
    logger.info("[DailyFeature] 已寫入 %d 筆（日期 %s）", written, latest_date)
    return written
