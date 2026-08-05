"""ETL Pipeline — 整合 抓取 → 清洗 → 寫入資料庫 流程。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

from src.config import settings
from src.constants import (
    BACKFILL_MIN_COMMON_STOCKS,
    BACKFILL_MIN_VALUATION_STOCKS,
    FINMIND_REQUEST_INTERVAL,
    PRICE_JUMP_WARN_THRESHOLD,
    SECONDS_PER_BACKFILL_DAY,
    TWSE_REQUEST_INTERVAL,
    UPSERT_BATCH_SIZE,
    VALUATION_COVERAGE_RATIO,
)
from src.data.database import get_effective_watchlist, get_session, init_db
from src.data.fetcher import FinMindFetcher
from src.data.schema import (
    Announcement,
    BacktestResult,
    BrokerTrade,
    ConceptGroup,
    ConceptMembership,
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
    RotationBacktestSummary,
    RotationBacktestTrade,
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


def _upsert_batch(
    model,
    df: pd.DataFrame,
    conflict_keys: list[str],
    batch_size: int = UPSERT_BATCH_SIZE,
    update_cols: list[str] | None = None,
) -> int:
    """將 DataFrame 分批寫入指定表（衝突解決可選）。

    SQLite 有 SQL 變數上限，必須分批 INSERT。

    Args:
        model: ORM model class
        df: 要寫入的資料
        conflict_keys: 衝突判定欄位（通常為 unique constraint 的欄位組合）
        batch_size: 每批筆數（預設 UPSERT_BATCH_SIZE=80）
        update_cols: 衝突時要更新的欄位列表。
            - None（預設）：on_conflict_do_nothing，舊值保留（不可變歷史紀錄場景）
            - list[str]：on_conflict_do_update，覆蓋指定欄位
              （重算指標 / DailyFeature 等場景，避免 stale value 殘留）

    C2 修復（2026-05-09 audit）：原始實作硬編碼 do_nothing，導致 TechnicalIndicator /
    DailyFeature 同日重算的舊值永不覆蓋（除權息回溯 ~2% 偏差、universe 雜訊 +5~10%）。
    新增 update_cols 參數讓 caller 顯式選擇覆蓋語意。
    """
    if df.empty:
        return 0

    # 清理 NaN / NaT → None（SQLite 不認得 pandas 的 NaN/NaT）
    # 先將 object 欄位的 NaN 轉 None（df.where 對 datetime64 欄位無效，需特殊處理）
    clean = df.copy()
    for col in clean.columns:
        if clean[col].dtype == "datetime64[ns]":
            clean[col] = clean[col].astype(object).where(clean[col].notna(), None)
    records = clean.where(pd.notna(clean), None).to_dict("records")
    with get_session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            stmt = sqlite_upsert(model).values(batch)
            if update_cols:
                stmt = stmt.on_conflict_do_update(
                    index_elements=conflict_keys,
                    set_={c: stmt.excluded[c] for c in update_cols},
                )
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=conflict_keys)
            session.execute(stmt)
        session.commit()
    return len(records)


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """驗證 OHLCV 資料值域，過濾無效列並記錄。

    檢查項目：close > 0、high >= low、low <= close <= high（OHLC 一致性）、volume >= 0。

    OHLC 一致性（2026-05-30 新增）：close 必須落在 [low, high] 區間內。
    防範上游回傳「close 超出當日高低範圍」的髒值（如 close 誤填成非當日數值），
    這類列在 high>=low 檢查下仍會漏網。注意 close==high / close==low 為合法收盤
    （收在最高/最低價），不會被攔；單日異常跳點需另以日間跳動哨兵處理。
    """
    if df.empty:
        return df

    n_before = len(df)
    mask = pd.Series(True, index=df.index)

    # close 必須為正
    if "close" in df.columns:
        invalid_close = df["close"].isna() | (df["close"] <= 0)
        mask &= ~invalid_close

    # high >= low
    if "high" in df.columns and "low" in df.columns:
        invalid_hl = df["high"].notna() & df["low"].notna() & (df["high"] < df["low"])
        mask &= ~invalid_hl

    # OHLC 一致性：low <= close <= high（缺值轉 NaN，比較結果為 False 故自動略過）
    if "close" in df.columns:
        close_num = pd.to_numeric(df["close"], errors="coerce")
        if "high" in df.columns:
            mask &= ~(close_num > pd.to_numeric(df["high"], errors="coerce"))
        if "low" in df.columns:
            mask &= ~(close_num < pd.to_numeric(df["low"], errors="coerce"))

    # volume >= 0
    if "volume" in df.columns:
        invalid_vol = df["volume"].notna() & (df["volume"] < 0)
        mask &= ~invalid_vol

    filtered = df[mask]
    n_dropped = n_before - len(filtered)
    if n_dropped > 0:
        logger.warning("OHLCV 值域驗證：過濾 %d 筆無效資料（共 %d 筆）", n_dropped, n_before)

    return filtered


def _batch_get_prior_closes(rows: pd.DataFrame, lookback_days: int = 14) -> dict[str, float]:
    """回查每支股票在其 df 內最早日期之前、DB 最近一筆收盤。

    供跳動哨兵在「單日全市場同步」（df 內無前一日）時取得 close-to-close 基準。
    一次查詢 DB（lookback 視窗涵蓋連假），於 Python 端逐股取 < 最早日期的最後一筆。
    """
    stock_ids = rows["stock_id"].unique().tolist()
    if not stock_ids:
        return {}
    earliest = rows.groupby("stock_id")["date"].min().to_dict()
    max_date = max(earliest.values())
    min_lookback = min(earliest.values()) - timedelta(days=lookback_days)
    with get_session() as session:
        db_rows = session.execute(
            select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close)
            .where(
                DailyPrice.stock_id.in_(stock_ids),
                DailyPrice.date < max_date,
                DailyPrice.date >= min_lookback,
            )
            .order_by(DailyPrice.stock_id, DailyPrice.date)
        ).all()
    by_stock: dict[str, list] = {}
    for sid, d, c in db_rows:
        by_stock.setdefault(sid, []).append((d, c))
    result: dict[str, float] = {}
    for sid, pairs in by_stock.items():
        cutoff = earliest[sid]
        prior = [c for d, c in pairs if d < cutoff and c is not None]
        if prior:
            result[sid] = float(prior[-1])  # pairs 已依日期升冪，取最後一筆
    return result


def _detect_price_jumps(df: pd.DataFrame, threshold: float = PRICE_JUMP_WARN_THRESHOLD) -> int:
    """日 K close-to-close 跳動哨兵：WARN 記錄異常跳動但不過濾。回傳可疑列數。

    比較每列 close 與「前一交易日收盤」：先取 df 內同股前一列，df 內無前值者
    （多為單日全市場同步）回查 DB 最近一筆較早收盤。|報酬| > threshold 即記錄。

    僅警示不刪列 —— 合法成因（除權息跳價 / IPO 首日 / 復牌大漲）一併會被點名，
    過濾會誤殺；此哨兵旨在留下 audit 線索（單一序列內 ±10% 內的離群點如 0050
    5/29 不在偵測範圍，需另以跨來源 reconciliation 處理）。
    """
    if df.empty or not {"stock_id", "date", "close"}.issubset(df.columns):
        return 0

    work = df[["stock_id", "date", "close"]].copy()
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.sort_values(["stock_id", "date"])
    work["prev"] = work.groupby("stock_id")["close"].shift(1)

    need_db = work[work["prev"].isna()]
    if not need_db.empty:
        prior = _batch_get_prior_closes(need_db)
        work.loc[need_db.index, "prev"] = need_db["stock_id"].map(prior)

    n_flagged = 0
    for _, r in work.iterrows():
        prev, close = r["prev"], r["close"]
        if prev is None or pd.isna(prev) or prev <= 0 or pd.isna(close):
            continue
        pct = close / prev - 1
        if abs(pct) > threshold:
            n_flagged += 1
            logger.warning(
                "OHLCV 跳動哨兵：%s %s close=%.2f 較前收 %.2f 跳動 %+.1f%%（>門檻 %.0f%%）",
                r["stock_id"],
                r["date"],
                close,
                prev,
                pct * 100,
                threshold * 100,
            )
    return n_flagged


def _upsert_daily_price(df: pd.DataFrame) -> int:
    """將日K線 DataFrame 寫入 daily_price 表（含值域驗證 + 跳動哨兵，衝突時略過）。"""
    df = _validate_ohlcv(df)
    _detect_price_jumps(df)
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
    """將股利 DataFrame 寫入 dividend 表。

    R1 重審發現（2026-07-08）：FinMind 新資料的 year 欄一律 NaN，舊版
    `dropna(subset=["year"])` 會把**全部**新股利列丟棄 → Step 11b 每日執行
    但寫入恆為 0，A3 除息入帳自上線起實際斷流（dividend 表停在 2026-01-28）。
    year 僅為屬性欄（dedup key 是 stock_id+date），改以 date 年份回填。
    """
    if df.empty:
        return 0
    # year 是 nullable=False：NaN 以 date 年份回填（勿丟行——FinMind 新資料 year 恆 NaN）
    df = df.copy()
    df["year"] = df["year"].fillna(df["date"].astype(str).str[:4])
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


def _sync_per_stock(
    *,
    model,
    stock_ids: list[str],
    fetch_fn: Callable[[FinMindFetcher, str, str, str], pd.DataFrame],
    upsert_fn: Callable[[pd.DataFrame], int],
    cache_days: int,
    lookback_days: int,
    label: str,
) -> int:
    """通用逐股同步：cache 檢查 → fetch → upsert。

    Args:
        model:         ORM model（需有 stock_id, date 欄位）
        stock_ids:     要同步的股票代號清單
        fetch_fn:      擷取函數 (fetcher, stock_id, start, end) -> DataFrame
        upsert_fn:     寫入函數 (df) -> int
        cache_days:    DB 資料在此天數內視為新鮮，跳過
        lookback_days: 回溯查詢天數
        label:         日誌標籤（如 "估值補抓"）

    Returns:
        新增筆數
    """
    fetcher = FinMindFetcher()
    total = 0
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    end = date.today().isoformat()
    skipped = 0

    last_dates = _batch_get_last_dates(model, stock_ids)
    for sid in stock_ids:
        last = last_dates.get(sid)
        if last and (date.today() - date.fromisoformat(last)).days < cache_days:
            skipped += 1
            continue
        try:
            df = fetch_fn(fetcher, sid, last or start, end)
            total += upsert_fn(df)
        except Exception:
            logger.warning("[%s] %s失敗，跳過", sid, label, exc_info=True)

    if skipped:
        logger.info("[%s] 跳過 %d 支（DB 已有近期資料）", label, skipped)
    return total


def sync_valuation_for_stocks(stock_ids: list[str]) -> int:
    """為指定股票補抓最新估值資料（PE/PB/殖利率）。"""
    return _sync_per_stock(
        model=StockValuation,
        stock_ids=stock_ids,
        fetch_fn=lambda f, sid, s, e: f.fetch_per_pbr(sid, s, e),
        upsert_fn=_upsert_valuation,
        cache_days=7,
        lookback_days=30,
        label="估值補抓",
    )


def sync_revenue_for_stocks(stock_ids: list[str]) -> int:
    """為指定股票補抓最新月營收。"""
    return _sync_per_stock(
        model=MonthlyRevenue,
        stock_ids=stock_ids,
        fetch_fn=lambda f, sid, s, e: f.fetch_monthly_revenue(sid, s, e),
        upsert_fn=_upsert_monthly_revenue,
        cache_days=30,
        lookback_days=180,
        label="營收補抓",
    )


def sync_dividends_for_stocks(stock_ids: list[str]) -> int:
    """為指定股票補抓股利資料（A3：rotation 持倉/pending 除息入帳的資料前提）。

    dividend 表原僅隨 watchlist 逐股同步；rotation 持倉來自全市場 discover，
    除息入帳需要當日除息事件在庫。morning-routine Step 11b 於 rotation update
    前呼叫（持倉 + pending 標的，量少：~20 檔 × 0.5s）。
    lookback_days=400：涵蓋跨年度除息季 + 增量起點。
    """
    return _sync_per_stock(
        model=Dividend,
        stock_ids=stock_ids,
        fetch_fn=lambda f, sid, s, e: f.fetch_dividend(sid, s, e),
        upsert_fn=_upsert_dividend,
        cache_days=7,
        lookback_days=400,
        label="股利補抓",
    )


def _upsert_financial(df: pd.DataFrame) -> int:
    """將財報 DataFrame 寫入 financial_statement 表。"""
    return _upsert_batch(FinancialStatement, df, ["stock_id", "date"])


def sync_financial_statements(
    watchlist: list[str] | None = None,
    quarters: int = 4,
) -> int:
    """同步 watchlist 財報資料（最近 N 季）。"""
    if watchlist is None:
        watchlist = get_effective_watchlist()
    init_db()
    total = sync_financial_for_stocks(watchlist, quarters)
    logger.info("[財報同步] 完成，共寫入 %d 筆", total)
    return total


def sync_financial_for_stocks(stock_ids: list[str], quarters: int = 4) -> int:
    """為指定股票補抓財報資料。"""
    return _sync_per_stock(
        model=FinancialStatement,
        stock_ids=stock_ids,
        fetch_fn=lambda f, sid, s, e: f.fetch_financial_summary(sid, s, e),
        upsert_fn=_upsert_financial,
        cache_days=60,
        lookback_days=quarters * 95 + 30,
        label="財報補抓",
    )


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

    項目 C：以 `ThreadPoolExecutor(max_workers=3)` 並行化（與 `sync_broker_bootstrap`
    一致），等效請求速率 ≈ 1 req/s，對 DJ 端點仍溫和。cache 篩選與 `_batch_get_last_dates`
    保留於主 thread，僅 HTTP + upsert 進入 worker。

    Args:
        stock_ids: 指定股票代號清單，預設使用 watchlist
        days:      查詢最近幾個交易日的彙整範圍（預設 5）

    Returns:
        新增的分點交易筆數
    """
    from concurrent.futures import ThreadPoolExecutor

    from src.data.twse_fetcher import fetch_dj_broker_trades

    if stock_ids is None:
        stock_ids = get_effective_watchlist()

    init_db()
    end_date = date.today()
    start_date = end_date - timedelta(days=days + 3)

    # 主 thread：批次查 last_date，篩出 pending（跳過 cache < 2 天的股票）
    last_dates = _batch_get_last_dates(BrokerTrade, stock_ids)
    pending: list[str] = []
    skipped: list[tuple[str, date]] = []
    for sid in stock_ids:
        latest_str = last_dates.get(sid)
        latest = date.fromisoformat(latest_str) if latest_str else None
        if latest and (date.today() - latest).days < 2:
            skipped.append((sid, latest))
        else:
            pending.append(sid)

    for sid, latest in skipped:
        logger.info("[分點] %s 已有最新資料（%s），跳過", sid, latest)

    if not pending:
        logger.info("[分點] 全部 %d 支快取新鮮，無需同步", len(stock_ids))
        return 0

    logger.info("[分點] 待抓 %d 支（跳過 %d 支快取新鮮）", len(pending), len(skipped))

    def _fetch_and_upsert(sid: str) -> int:
        """Worker：抓一支股票 → 寫 DB；失敗降級為 0，不中止其他 worker。"""
        try:
            df = fetch_dj_broker_trades(sid, start_date, end_date)
        except Exception:
            logger.warning("[分點] %s 抓取失敗，跳過", sid, exc_info=True)
            return 0
        if df.empty:
            return 0
        try:
            n = _upsert_broker_trade(df)
        except Exception:
            logger.warning("[分點] %s 寫入失敗，跳過", sid, exc_info=True)
            return 0
        if n > 0:
            logger.info("[分點] %s 寫入 %d 筆", sid, n)
        return n

    total = 0
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="broker_sync") as pool:
        # pool.map 會保留 pending 的順序、iteration 時 raise 也會被 _fetch_and_upsert 的 try 吞掉
        for n in pool.map(_fetch_and_upsert, pending):
            total += n

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
        logger.warning("[Bootstrap] 查詢交易日失敗，將使用工作日曆法", exc_info=True)

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

    # 建立待抓取清單（排除已存在的 pair）
    tasks: list[tuple[str, date]] = [
        (sid, td) for sid in stock_ids for td in trading_dates if (sid, td) not in existing_pairs
    ]

    if not tasks:
        logger.info("[Bootstrap] 全部已同步，無需補齊")
        return 0

    logger.info("[Bootstrap] 共 %d 筆待抓取（已排除 %d 筆既有資料）", len(tasks), len(existing_pairs))

    # 並行抓取（max_workers=3 尊重 TWSE 速率限制）
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = 0
    sid_counts: dict[str, int] = {}

    def _fetch_one(sid: str, td: date) -> tuple[str, int]:
        df = fetch_dj_broker_trades(sid, td, td)
        if not df.empty:
            n = _upsert_broker_trade(df)
            return (sid, n)
        return (sid, 0)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_fetch_one, sid, td): (sid, td) for sid, td in tasks}
        for future in as_completed(futures):
            sid, td = futures[future]
            try:
                _, n = future.result()
                if n > 0:
                    sid_counts[sid] = sid_counts.get(sid, 0) + n
                    total += n
            except Exception:
                logger.warning("[Bootstrap] %s %s 抓取失敗", sid, td, exc_info=True)

    for sid, cnt in sid_counts.items():
        logger.info("[Bootstrap] %s 補齊 %d 筆", sid, cnt)

    logger.info("[Bootstrap] 完成，總計新增 %d 筆", total)
    return total


def sync_stock(
    stock_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    fetcher: FinMindFetcher | None = None,
    *,
    last_dates: dict[str, str | None] | None = None,
) -> dict[str, int]:
    """同步單一股票的所有資料（日K + 三大法人 + 融資融券）。

    支援增量更新：若 DB 已有資料，自動從最後一筆日期開始抓取。

    Parameters
    ----------
    last_dates : dict[str, str | None] | None
        各表預先批次查詢的最後日期，key 為表名（daily_price / institutional /
        margin / revenue / dividend / financial），由 sync_watchlist() 傳入以
        減少 DB 查詢次數。若為 None 則退回逐表查詢。

    Returns:
        dict: 各資料表新增筆數，例如 {"daily_price": 100, "institutional": 300, "margin": 100}
    """
    if fetcher is None:
        fetcher = FinMindFetcher()

    default_start = start_date or settings.fetcher.default_start_date
    if end_date is None:
        end_date = date.today().isoformat()

    def _resolve_last(table_key: str, model) -> str | None:
        """從預查結果取得 last_date，若無則退回單次查詢。"""
        if last_dates is not None:
            return last_dates.get(table_key)
        return _get_last_date(model, stock_id)

    result = {}

    # --- 日K線 ---
    last = _resolve_last("daily_price", DailyPrice)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步日K線: %s ~ %s", stock_id, s, end_date)
    df_price = fetcher.fetch_daily_price(stock_id, s, end_date)
    result["daily_price"] = _upsert_daily_price(df_price)

    # --- 三大法人 ---
    last = _resolve_last("institutional", InstitutionalInvestor)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步三大法人: %s ~ %s", stock_id, s, end_date)
    df_inst = fetcher.fetch_institutional(stock_id, s, end_date)
    result["institutional"] = _upsert_institutional(df_inst)

    # --- 融資融券 ---
    last = _resolve_last("margin", MarginTrading)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步融資融券: %s ~ %s", stock_id, s, end_date)
    df_margin = fetcher.fetch_margin_trading(stock_id, s, end_date)
    result["margin"] = _upsert_margin(df_margin)

    # --- 月營收 ---
    last = _resolve_last("revenue", MonthlyRevenue)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步月營收: %s ~ %s", stock_id, s, end_date)
    df_rev = fetcher.fetch_monthly_revenue(stock_id, s, end_date)
    result["revenue"] = _upsert_monthly_revenue(df_rev)

    # --- 股利 ---
    last = _resolve_last("dividend", Dividend)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步股利: %s ~ %s", stock_id, s, end_date)
    df_div = fetcher.fetch_dividend(stock_id, s, end_date)
    result["dividend"] = _upsert_dividend(df_div)

    # --- 財報 ---
    last = _resolve_last("financial", FinancialStatement)
    s = last if last and last > default_start else default_start
    logger.info("[%s] 同步財報: %s ~ %s", stock_id, s, end_date)
    try:
        df_fin = fetcher.fetch_financial_summary(stock_id, s, end_date)
        result["financial"] = _upsert_financial(df_fin)
    except Exception:
        logger.warning("[%s] 財報同步失敗，跳過", stock_id, exc_info=True)
        result["financial"] = 0

    return result


# 批次查詢用的 (表名, ORM Model) 映射
_SYNC_TABLE_MODELS: list[tuple[str, type]] = [
    ("daily_price", DailyPrice),
    ("institutional", InstitutionalInvestor),
    ("margin", MarginTrading),
    ("revenue", MonthlyRevenue),
    ("dividend", Dividend),
    ("financial", FinancialStatement),
]


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

    # 批次預查各表的 last_date（6 次 DB 查詢，而非 N×6 次）
    batch_last: dict[str, dict[str, str | None]] = {}
    for table_key, model in _SYNC_TABLE_MODELS:
        batch_last[table_key] = _batch_get_last_dates(model, watchlist)
    logger.info("已批次查詢 %d 張表的 last_date（%d 支股票）", len(_SYNC_TABLE_MODELS), len(watchlist))

    all_results = {}
    for stock_id in watchlist:
        logger.info("=" * 50)
        logger.info("開始同步: %s", stock_id)
        # 為每支股票組裝預查結果
        per_stock_last = {tbl: batch_last[tbl].get(stock_id) for tbl in batch_last}
        try:
            all_results[stock_id] = sync_stock(stock_id, start_date, end_date, fetcher, last_dates=per_stock_last)
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

    # 台股 ETF 代號皆以 "00" 開頭：0050、00878、00882、00991A、00715L 等
    if re.match(r"^00", sid) or "ETF" in name:
        return "etf"
    if len(sid) == 6 and sid.isdigit():
        return "warrant"
    if "特" in (stock_name or ""):
        return "preferred"
    return "stock"


def backfill_market_history(
    start: date,
    end: date | None = None,
    *,
    datasets: tuple[str, ...] = ("price", "institutional", "margin"),
    dry_run: bool = False,
    progress_every: int = 20,
    stop_flag: "callable | None" = None,
) -> dict[str, int]:
    """以 TWSE/TPEX 每日全市場端點回補歷史資料（B1①）。

    ## 為什麼走 TWSE/TPEX 而非 FinMind 逐股

    每日全市場端點回傳的是**當日實際掛牌交易的所有股票**，因此 2020 年的檔案
    自然含有當時在市、如今已下市的標的——倖存者偏差在資料源頭就被解掉，
    不需要另外逐股補抓下市股（下市清單仍需同步，見 `sync_delisting_info`，
    那是為了知道「何時下市」以供 PIT 判定可交易性）。

    另一個理由是量：逐股回補 3,000+ 檔 × 5 年 × 3 個 dataset 在 FinMind
    0.5s/次 下不可行；每日端點是 1,225 個交易日 × 3s。

    ## 續跑

    **不另外維護進度檔**——直接以 DB 現況為準：某日**普通股（4 碼）檔數**達
    `BACKFILL_MIN_COMMON_STOCKS` 即視為已回補而跳過。中斷後重跑會自動從缺口
    續行，且重跑安全（upsert）。

    ⚠ 判定既不能用「該日是否有資料」也不能用「總筆數」——兩者都被實測打臉，
    理由詳見 `constants.BACKFILL_MIN_COMMON_STOCKS` 的註解。

    Args:
        start: 起始日（含）。
        end: 結束日（含）；None ＝今日。範圍內已達全市場覆蓋的日期會自動跳過，
            因此預設值可安全地涵蓋全部歷史（同時補中間的任何缺口）。
        datasets: 要回補的資料集子集。
        dry_run: 只列出將回補的日期與預估時間，不實際抓取。
        progress_every: 每 N 個交易日輸出一次進度。
        stop_flag: 可選的中止判定函數；回傳 True 時提前結束（供 CLI 接 Ctrl-C）。

    Returns:
        {"trading_days": N, "daily_price": M, "institutional": K, "margin": J, "skipped": S}
    """
    from src.data.twse_fetcher import (
        fetch_market_daily_prices,
        fetch_market_institutional,
        fetch_market_margin,
    )

    init_db()
    result = {"trading_days": 0, "daily_price": 0, "institutional": 0, "margin": 0, "skipped": 0}

    if end is None:
        end = date.today()
    if start > end:
        logger.info("回補範圍為空（start=%s > end=%s）", start, end)
        return result

    # 已達「全市場覆蓋」的日期（見 docstring：不能只看該日是否有資料）
    with get_session() as session:
        covered = {
            r[0]
            for r in session.execute(
                select(DailyPrice.date)
                .where(DailyPrice.date >= start, DailyPrice.date <= end)
                # 只數 4 碼普通股：權證/ETF 在崩盤日會整批無報價，用總筆數會誤判
                .where(func.length(DailyPrice.stock_id) == 4)
                .group_by(DailyPrice.date)
                .having(func.count(DailyPrice.id) >= BACKFILL_MIN_COMMON_STOCKS)
            ).all()
        }

    # 待補的候選日（排除週末與已覆蓋者）；實際是否為交易日由 API 回空判定
    pending = [
        d
        for d in (start + timedelta(days=i) for i in range((end - start).days + 1))
        if d.weekday() < 5 and d not in covered
    ]
    result["skipped"] = (end - start).days + 1 - len(pending)

    # 每個交易日的實際耗時：每個 dataset 都要 TWSE + TPEX（雖並行但各自 3 秒節流）
    # 再加解析/寫入。實測 2026-08-03 全程 458 個交易日 / 3h45m ≈ 29 秒/日；
    # 原估算用 3 秒/dataset 低估近 3 倍，改用實測值。
    n_ds = len([d for d in datasets if d in ("price", "institutional", "margin")])
    est_sec = len(pending) * SECONDS_PER_BACKFILL_DAY * max(1, n_ds) / 3
    logger.info(
        "[回補] %s ~ %s：待補 %d 個平日（已跳過 %d 日），dataset=%s，預估 %.1f 小時",
        start,
        end,
        len(pending),
        result["skipped"],
        ",".join(datasets),
        est_sec / 3600,
    )
    if dry_run:
        return result

    for i, d in enumerate(pending, 1):
        if stop_flag is not None and stop_flag():
            logger.warning("[回補] 收到中止訊號，已完成 %d/%d 日", i - 1, len(pending))
            break

        df_price = fetch_market_daily_prices(d) if "price" in datasets else pd.DataFrame()
        if df_price.empty:
            # 假日或該日無資料——不是錯誤
            continue

        result["trading_days"] += 1
        result["daily_price"] += _upsert_daily_price(df_price)

        if "institutional" in datasets:
            df_inst = fetch_market_institutional(d)
            if not df_inst.empty:
                result["institutional"] += _upsert_institutional(df_inst)
        if "margin" in datasets:
            df_margin = fetch_market_margin(d)
            if not df_margin.empty:
                result["margin"] += _upsert_margin(df_margin)

        if result["trading_days"] % progress_every == 0:
            logger.info(
                "[回補] 進度 %d/%d 平日（%s）— 交易日 %d, 日K %d 筆",
                i,
                len(pending),
                d.isoformat(),
                result["trading_days"],
                result["daily_price"],
            )

    logger.info(
        "[回補] 完成 — 交易日 %d, 日K %d 筆, 法人 %d 筆, 融資融券 %d 筆",
        result["trading_days"],
        result["daily_price"],
        result["institutional"],
        result["margin"],
    )
    return result


def _is_quota_exhausted(exc: Exception) -> bool:
    """判斷例外是否為 FinMind 配額/流量限制（而非個股層級的錯誤）。

    FinMind 配額用盡回 **402 Payment Required**、超速回 **429**。兩者都代表
    「再打下去也沒用」，與「這支股票沒資料」性質完全不同——後者該跳過續跑，
    前者該立刻停手並告訴使用者稍後重跑。
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (402, 429):
        return True
    # 部分路徑只剩字串（例如被包過一層），退而求其次比對訊息
    text = str(exc)
    return "402" in text or "Payment Required" in text or "429" in text


def backfill_valuation_history(
    start: date,
    end: date | None = None,
    *,
    markets: tuple[str, ...] = ("twse", "tpex"),
    dry_run: bool = False,
    progress_every: int = 20,
    stop_flag: "callable | None" = None,
) -> dict[str, int]:
    """回補 `stock_valuation` 歷史（B1① 的基本面缺口，§6.5 #20）。

    ## 為什麼要分兩條路走

    B1① 只補了價量。估值缺口使 value/dividend/growth 的 PIT 重放全部失效
    （粗篩在估值表為空時 fail-open，模式靜默退化成流動性篩選），
    詳 `logs/pit_crossmode_20260804/REPORT.md` §3。

    補法依市場分流，因為**兩邊的官方端點狀況不同**：

    - **上市（twse）**：`BWIBBU_d` 每日全市場端點健在且**有完整歷史**
      （實測 2024-01-02 回傳 997 檔）。一天一次呼叫拿全市場，走這條。
    - **上櫃（tpex）**：`peratio_book/pera_result.php` **已下架**——所有日期
      （含當日）皆 302 導向 `/errors`。TPEX 新版 openapi
      （`tpex_mainboard_peratio_analysis`）只回**當日**、無日期參數。
      故上櫃歷史**無官方來源**，改走 FinMind `TaiwanStockPER` 逐股。

    逐股看似違反「官方優先」的資料源順序，但這裡沒有官方選項；且
    FinMind PER 支援日期區間，一檔股票一次呼叫即涵蓋全期間
    （1,100 檔 × 0.5s ≈ 10 分鐘），比每日端點更省。

    ## 續跑

    與 `backfill_market_history` 同樣**不維護進度檔**，直接以 DB 現況為準：

    - twse：某日估值檔數達 `BACKFILL_MIN_VALUATION_STOCKS` 即視為已補而跳過。
    - tpex：某檔在區間內的估值日數達該檔**價量日數的 8 成**即視為已補。
      不用固定門檻——上櫃股上市時間不一，新股本來就只有少數日子有資料。

    Args:
        start: 起始日（含）。
        end: 結束日（含）；None ＝今日。
        markets: 要回補的市場子集（"twse" / "tpex"）。
        dry_run: 只估算不抓取。
        progress_every: 每 N 個單位輸出一次進度。
        stop_flag: 回傳 True 時提前結束（供 CLI 接 Ctrl-C）。

    Returns:
        {"twse_days": N, "twse_rows": M, "tpex_stocks": K, "tpex_rows": J, "skipped_days": S, "skipped_stocks": T}
    """
    from src.data.twse_fetcher import fetch_twse_valuation_all

    init_db()
    result = {
        "twse_days": 0,
        "twse_rows": 0,
        "tpex_stocks": 0,
        "tpex_rows": 0,
        "skipped_days": 0,
        "skipped_stocks": 0,
        "quota_exhausted": 0,
    }

    if end is None:
        end = date.today()
    if start > end:
        logger.info("估值回補範圍為空（start=%s > end=%s）", start, end)
        return result

    # ---------- 上市：TWSE 每日全市場 ---------- #
    if "twse" in markets:
        with get_session() as session:
            covered = {
                r[0]
                for r in session.execute(
                    select(StockValuation.date)
                    .where(StockValuation.date >= start, StockValuation.date <= end)
                    .group_by(StockValuation.date)
                    .having(func.count(StockValuation.id) >= BACKFILL_MIN_VALUATION_STOCKS)
                ).all()
            }
        pending_days = [
            d
            for d in (start + timedelta(days=i) for i in range((end - start).days + 1))
            if d.weekday() < 5 and d not in covered
        ]
        result["skipped_days"] = (end - start).days + 1 - len(pending_days)
        logger.info(
            "[估值回補/上市] %s ~ %s：待補 %d 個平日（已跳過 %d 日），預估 %.1f 分鐘",
            start,
            end,
            len(pending_days),
            result["skipped_days"],
            len(pending_days) * TWSE_REQUEST_INTERVAL / 60,
        )

        if not dry_run:
            for i, d in enumerate(pending_days, 1):
                if stop_flag is not None and stop_flag():
                    logger.warning("[估值回補/上市] 收到中止訊號，已完成 %d/%d 日", i - 1, len(pending_days))
                    break
                df = fetch_twse_valuation_all(d)
                if df.empty:
                    continue  # 假日或該日無資料——不是錯誤
                result["twse_days"] += 1
                result["twse_rows"] += _upsert_valuation(df)
                if result["twse_days"] % progress_every == 0:
                    logger.info(
                        "[估值回補/上市] 進度 %d/%d 平日（%s）— 交易日 %d, 累計 %d 筆",
                        i,
                        len(pending_days),
                        d.isoformat(),
                        result["twse_days"],
                        result["twse_rows"],
                    )

    # ---------- 上櫃：FinMind 逐股全區間 ---------- #
    if "tpex" in markets:
        with get_session() as session:
            otc_ids = [
                r[0]
                for r in session.execute(
                    select(StockInfo.stock_id)
                    .where(StockInfo.listing_type == "tpex")
                    # 只補普通股：ETF/權證/特別股無 PE 語意
                    .where(StockInfo.security_type == "stock")
                    .order_by(StockInfo.stock_id)
                ).all()
            ]
            # 各檔在區間內「已有的估值日數」與「應有的價量日數」
            val_days = dict(
                session.execute(
                    select(StockValuation.stock_id, func.count(func.distinct(StockValuation.date)))
                    .where(StockValuation.date >= start, StockValuation.date <= end)
                    .group_by(StockValuation.stock_id)
                ).all()
            )
            price_days = dict(
                session.execute(
                    select(DailyPrice.stock_id, func.count(func.distinct(DailyPrice.date)))
                    .where(DailyPrice.date >= start, DailyPrice.date <= end)
                    .where(DailyPrice.stock_id.in_(otc_ids))
                    .group_by(DailyPrice.stock_id)
                ).all()
            )

        pending_ids = [
            sid
            for sid in otc_ids
            # 沒有價量的（未上市/已下市於區間外）不必補；有價量者要求估值覆蓋 8 成
            if price_days.get(sid, 0) > 0 and val_days.get(sid, 0) < price_days[sid] * VALUATION_COVERAGE_RATIO
        ]
        result["skipped_stocks"] = len(otc_ids) - len(pending_ids)
        logger.info(
            "[估值回補/上櫃] 待補 %d 檔（已跳過 %d 檔），FinMind 逐股，預估 %.1f 分鐘",
            len(pending_ids),
            result["skipped_stocks"],
            len(pending_ids) * FINMIND_REQUEST_INTERVAL / 60,
        )

        if not dry_run and pending_ids:
            fetcher = FinMindFetcher()
            for i, sid in enumerate(pending_ids, 1):
                if stop_flag is not None and stop_flag():
                    logger.warning("[估值回補/上櫃] 收到中止訊號，已完成 %d/%d 檔", i - 1, len(pending_ids))
                    break
                try:
                    df = fetcher.fetch_per_pbr(sid, start, end)
                except Exception as exc:
                    # 配額用盡與個股錯誤要分開處理：前者續跑只會空轉。
                    # 實測 2026-08-05 首跑——580 檔後 FinMind 回 402，其後 299 次
                    # 呼叫全部瞬間失敗（間隔 0 秒），純粹浪費且把真正的原因淹沒在
                    # 299 行相同的 WARNING 裡。
                    if _is_quota_exhausted(exc):
                        logger.error(
                            "[估值回補/上櫃] FinMind 配額用盡（%s），已完成 %d/%d 檔——"
                            "配額恢復後重跑本指令即可從缺口續行",
                            type(exc).__name__,
                            i - 1,
                            len(pending_ids),
                        )
                        result["quota_exhausted"] = 1
                        break
                    logger.warning("[估值回補/上櫃] %s 抓取失敗，跳過：%s", sid, exc)
                    continue
                if df.empty:
                    continue
                result["tpex_stocks"] += 1
                result["tpex_rows"] += _upsert_valuation(df)
                if i % progress_every == 0:
                    logger.info(
                        "[估值回補/上櫃] 進度 %d/%d 檔（%s）— 有資料 %d 檔, 累計 %d 筆",
                        i,
                        len(pending_ids),
                        sid,
                        result["tpex_stocks"],
                        result["tpex_rows"],
                    )

    logger.info(
        "[估值回補] 完成 — 上市 %d 日/%d 筆, 上櫃 %d 檔/%d 筆",
        result["twse_days"],
        result["twse_rows"],
        result["tpex_stocks"],
        result["tpex_rows"],
    )
    return result


def clear_false_delistings(min_trading_days_after: int = 3) -> int:
    """清除「下市後仍在交易」的 `delisted_date`（B1① 資料語意防線）。

    FinMind `TaiwanStockDelisting` 收錄的是「**從該板終止**」，其中包含**轉板**
    （如上櫃轉上市＝終止上櫃），並非全部都是真正停止交易。實測 2026-08-03：
    30 檔有價量的下市股中，5236 凌陽創新於「下市日」2026-07-16 之後仍持續正常
    交易（2026-08-03 成交 138,695 股）——它是轉板，不是下市。

    誤判方向是**過度保守**（把還在交易的股票當成不可交易），比倖存者偏差安全，
    但仍會讓 PIT universe 少掉真實可交易標的。

    判定採「下市日之後仍有 >= N 個交易日的報價」，而非單日——避免單筆髒資料
    就把真正的下市翻案。

    Args:
        min_trading_days_after: 下市日後需觀察到幾個交易日才判定為誤記。

    Returns:
        被清除 `delisted_date` 的股票數。
    """
    cleared = 0
    with get_session() as session:
        rows = session.execute(select(StockInfo).where(StockInfo.delisted_date.isnot(None))).scalars().all()
        for info in rows:
            n_after = session.execute(
                select(func.count(func.distinct(DailyPrice.date))).where(
                    DailyPrice.stock_id == info.stock_id,
                    DailyPrice.date > info.delisted_date,
                )
            ).scalar()
            if (n_after or 0) >= min_trading_days_after:
                logger.warning(
                    "疑似轉板而非下市：%s %s 於 %s 後仍有 %d 個交易日報價 — 清除 delisted_date",
                    info.stock_id,
                    info.stock_name or "",
                    info.delisted_date,
                    n_after,
                )
                info.delisted_date = None
                cleared += 1
        if cleared:
            session.commit()
    return cleared


def sync_delisting_info(fetcher: FinMindFetcher | None = None) -> int:
    """同步下市清單至 `stock_info.delisted_date`（B1① 倖存者偏差修正）。

    為什麼需要：`stock_info` 原本只描述「今天還在市」的股票，歷史重放會自動
    排除當時在市、後來下市的標的——而下市股往往正是表現最差的那批，排除它們
    會讓歷史績效系統性偏高。有了 `delisted_date`，PIT 才能判定
    「該股於 as_of 當時是否可交易」。

    行為：
      - 下市股若不在 `stock_info` 中（已從現行清單消失）→ **新增**一列，
        `security_type='stock'` 讓它能通過 universe SQL 過濾
      - 已存在者 → 僅補 `delisted_date`，不動既有產業分類
      - 取得失敗（空 DataFrame）→ 回傳 0 且**不修改任何資料**，避免把全部
        股票誤標為未下市

    Returns:
        寫入/更新的筆數。
    """
    init_db()
    f = fetcher or FinMindFetcher()
    df = f.fetch_delisting_list()
    if df.empty or "stock_id" not in df.columns:
        logger.warning("下市清單為空，跳過（維持 stock_info 現狀）")
        return 0

    updated = 0
    with get_session() as session:
        for _, row in df.iterrows():
            sid = str(row["stock_id"]).strip()
            if not sid:
                continue
            try:
                dl_date = pd.to_datetime(row["date"]).date()
            except Exception:
                continue

            existing = session.execute(select(StockInfo).where(StockInfo.stock_id == sid)).scalar_one_or_none()
            if existing is None:
                session.add(
                    StockInfo(
                        stock_id=sid,
                        stock_name=str(row.get("stock_name") or "") or None,
                        industry_category=None,
                        # 讓 UniverseFilter 的 SQL 過濾不會直接排除它；
                        # 實際可交易性由 delisted_date + 當日有無報價決定
                        security_type="stock",
                        delisted_date=dl_date,
                    )
                )
                updated += 1
            elif existing.delisted_date != dl_date:
                existing.delisted_date = dl_date
                updated += 1
        session.commit()

    cleared = clear_false_delistings()
    logger.info(
        "下市清單同步完成：%d 筆 stock_info 更新（清單共 %d 筆，另清除 %d 筆疑似轉板）",
        updated,
        len(df),
        cleared,
    )
    return updated


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
        for i in range(0, len(records), UPSERT_BATCH_SIZE):
            batch = records[i : i + UPSERT_BATCH_SIZE]
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


def sync_taiwan_vix(
    start_date: str | None = None,
    end_date: str | None = None,
    fetcher: FinMindFetcher | None = None,
) -> int:
    """同步台灣 VIX 波動率指數至 DailyPrice（stock_id='TW_VIX'）。"""
    init_db()

    if fetcher is None:
        fetcher = FinMindFetcher()

    default_start = start_date or settings.fetcher.default_start_date
    if end_date is None:
        end_date = date.today().isoformat()

    last = _get_last_date(DailyPrice, "TW_VIX")
    s = last if last and last > default_start else default_start

    logger.info("[VIX] 同步台灣 VIX: %s ~ %s", s, end_date)
    df = fetcher.fetch_taiwan_vix(s, end_date)
    count = _upsert_daily_price(df)
    logger.info("[VIX] 完成 — %d 筆", count)
    return count


def sync_us_vix(
    start_date: str | None = None,
    end_date: str | None = None,
) -> int:
    """同步美國 VIX (CBOE ^VIX) 至 DailyPrice（stock_id='US_VIX'）。

    使用 yfinance 抓取，與 FinMind 無關。
    """
    init_db()

    default_start = start_date or settings.fetcher.default_start_date
    if end_date is None:
        end_date = date.today().isoformat()

    last = _get_last_date(DailyPrice, "US_VIX")
    s = last if last and last > default_start else default_start

    logger.info("[US_VIX] 同步美國 VIX: %s ~ %s", s, end_date)
    from src.data.fetcher import fetch_us_vix

    df = fetch_us_vix(s, end_date)
    count = _upsert_daily_price(df)
    logger.info("[US_VIX] 完成 — %d 筆", count)
    return count


def sync_market_data(
    days: int = 10,
    fetcher: FinMindFetcher | None = None,
    max_stocks: int | None = 200,
) -> dict[str, int]:
    """同步全市場資料（日K + 三大法人 + 融資融券），用於 discover 掃描。

    資料來源優先順序：
    1. TWSE/TPEX 官方開放資料（免費，6 次 API 取得全市場）
    2. FinMind 批次 API（需付費帳號）
    3. FinMind 逐股抓取（免費帳號備案，較慢）

    Args:
        days: 抓取最近 N 天的資料
        fetcher: 可注入 FinMind fetcher 實例（用於備案策略）
        max_stocks: 備案策略最多抓取的股票數；None 表示不限制（上游 CLI 常傳 None）

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
        logger.warning("[SBL] 查詢 DB 最新日期失敗，使用預設 days=%d", days, exc_info=True)

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
        logger.warning("[全市場] MOPS 重訊同步失敗，不影響其他資料", exc_info=True)

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
    logger.info(
        "[全市場] 所有批次來源不可用，改用 FinMind 逐股抓取（上限 %s 支）",
        max_stocks if max_stocks is not None else "無",
    )

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
            logger.warning("[%s] 抓取失敗，跳過", sid, exc_info=True)

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
    """將技術指標 DataFrame 寫入 technical_indicator 表。

    C2 修復：用 update_cols=["value"]，同日重算指標（如除權息回溯、--adjust-dividend）
    時可正確覆蓋舊值，避免 EAV 寬表載入時拿到 stale value。
    """
    return _upsert_batch(
        TechnicalIndicator,
        df,
        ["stock_id", "date", "name"],
        update_cols=["value"],
    )


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


def save_rotation_backtest(result) -> int:
    """將輪動回測結果與交易明細寫入 DB，回傳 rotation_backtest_summary.id。"""
    init_db()

    config = result.config
    metrics = result.metrics

    with get_session() as session:
        summary = RotationBacktestSummary(
            portfolio_name=config.get("portfolio_name", "__adhoc__"),
            mode=config.get("mode", ""),
            max_positions=config.get("max_positions", 0),
            holding_days=config.get("holding_days", 0),
            allow_renewal=config.get("allow_renewal", True),
            start_date=config.get("start_date"),
            end_date=config.get("end_date"),
            initial_capital=config.get("capital", 0),
            final_capital=metrics.get("final_capital", 0),
            total_return=metrics.get("total_return", 0),
            annual_return=metrics.get("annual_return", 0),
            sharpe_ratio=metrics.get("sharpe_ratio"),
            max_drawdown=metrics.get("max_drawdown", 0),
            win_rate=metrics.get("win_rate"),
            total_trades=metrics.get("total_trades", 0),
            avg_return_per_trade=metrics.get("avg_return_per_trade"),
            avg_win=metrics.get("avg_win"),
            avg_loss=metrics.get("avg_loss"),
            trading_days=metrics.get("trading_days"),
            # P0 擬真度新增指標
            sortino_ratio=metrics.get("sortino_ratio"),
            calmar_ratio=metrics.get("calmar_ratio"),
            var_95=metrics.get("var_95"),
            cvar_95=metrics.get("cvar_95"),
            profit_factor=metrics.get("profit_factor"),
            benchmark_return=metrics.get("benchmark_return"),
            total_cost=metrics.get("total_cost"),
            cost_drag_pct=metrics.get("cost_drag_pct"),
        )
        session.add(summary)
        session.flush()
        summary_id = summary.id

        for t in result.trades:
            trade = RotationBacktestTrade(
                backtest_id=summary_id,
                stock_id=t["stock_id"],
                entry_date=t["entry_date"],
                entry_price=t["entry_price"],
                exit_date=t.get("exit_date"),
                exit_price=t.get("exit_price"),
                shares=t.get("shares", 0),
                pnl=t.get("pnl"),
                return_pct=t.get("return_pct"),
                exit_reason=t.get("exit_reason"),
                entry_rank=t.get("entry_rank"),
                entry_score=t.get("entry_score"),
                buy_slippage=t.get("buy_slippage"),
                sell_slippage=t.get("sell_slippage"),
                trade_cost=t.get("trade_cost")
                or (
                    (t.get("commission", 0) + t.get("tax", 0) + t.get("slippage_cost", 0))
                    if any(t.get(k) for k in ("commission", "tax", "slippage_cost"))
                    else None
                ),
            )
            session.add(trade)

        session.commit()
        logger.info("輪動回測結果已儲存 (id=%d, %d 筆交易)", summary_id, len(result.trades))

    return summary_id


# ────────────────────────────────────────────────────────────────
#  Feature Store ETL
# ────────────────────────────────────────────────────────────────


DAILY_FEATURE_COLUMNS: list[str] = [
    "stock_id",
    "date",
    "close",
    "volume",
    "turnover",
    "ma20",
    "ma60",
    "volume_ma20",
    "turnover_ma5",
    "turnover_ma20",
    "momentum_20d",
    "volatility_20d",
    "turnover_ratio_5d_20d",
    "high_20d",
    "computed_at",
]


def compute_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """在既有 DailyPrice 明細上向量化計算 DailyFeature 各欄（純函數）。

    B1②（2026-08-04）抽出為共用實作：每日增量路徑
    （`compute_and_store_daily_features`）與歷史回補路徑
    （`backfill_daily_features`）必須用**完全相同**的算式，否則歷史特徵與
    今日特徵不同質，PIT 重放出來的 universe 就不是當時真正會得到的那個。

    輸入需已按 (stock_id, date) 排序且過濾掉 volume<=0 / close 無效者。
    所有 rolling 皆為**後視窗**，天然不會用到未來資料。

    Args:
        df: 含 stock_id / date / high / close / volume / turnover 的明細。

    Returns:
        原 df 加上各特徵欄（原地修改後回傳同一物件）。
    """
    g_close = df.groupby("stock_id")["close"]
    g_vol = df.groupby("stock_id")["volume"]
    g_turnover = df.groupby("stock_id")["turnover"]

    df["ma20"] = g_close.transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["ma60"] = g_close.transform(lambda s: s.rolling(60, min_periods=30).mean())
    df["volume_ma20"] = g_vol.transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["turnover_ma5"] = g_turnover.transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["turnover_ma20"] = g_turnover.transform(lambda s: s.rolling(20, min_periods=10).mean())

    # 20 日報酬率 (%)
    df["momentum_20d"] = g_close.transform(lambda s: s.pct_change(20) * 100)

    # 20 日年化波動率 (%)
    df["volatility_20d"] = g_close.transform(
        lambda s: s.pct_change().rolling(20, min_periods=10).std() * (252**0.5) * 100
    )

    # 5日/20日成交金額比（相對流動性：偵測「突然被市場關注」的股票）
    df["turnover_ratio_5d_20d"] = df["turnover_ma5"] / df["turnover_ma20"].replace(0, float("nan"))

    # 20 日最高價（突破型過濾：close / high_20d >= 0.9 確認真突破）
    df["high_20d"] = df.groupby("stock_id")["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    return df


def compute_and_store_daily_features(lookback_days: int = 90, min_stocks_per_day: int = 1000) -> int:
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
                DailyPrice.high,
                DailyPrice.close,
                DailyPrice.volume,
                DailyPrice.turnover,
            ).where(DailyPrice.date >= cutoff)
        ).all()

    if not rows:
        logger.warning("[DailyFeature] 無 DailyPrice 資料可計算")
        return 0

    df = pd.DataFrame(rows, columns=["stock_id", "date", "high", "close", "volume", "turnover"])
    df = df.sort_values(["stock_id", "date"])

    # W6 修復（2026-05-09 audit）：排除盤前/未開盤暫定資料（volume<=0 或 close 缺失），
    # 避免污染 momentum_20d / volatility_20d 的 rolling 視窗
    n_before = len(df)
    df = df[(df["volume"] > 0) & df["close"].notna() & (df["close"] > 0)]
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        logger.info(
            "[DailyFeature] 過濾 %d 筆 volume<=0 / close 無效資料（W6 pre-market guard）",
            n_filtered,
        )

    logger.info("[DailyFeature] 共 %d 筆有效資料，開始向量化計算...", len(df))
    df = compute_feature_columns(df)

    # 只取最新一日（增量更新策略），並加最低覆蓋率守門
    # 防止「watchlist 子集先寫 DailyPrice → sync-features 抓到部分日期當 latest」的污染
    date_counts = df.groupby("date").size().sort_index(ascending=False)
    valid_dates = date_counts[date_counts >= min_stocks_per_day].index

    if len(valid_dates) == 0:
        logger.warning(
            "[DailyFeature] 無任何日期達最低覆蓋（≥%d 支），跳過寫入。請先執行全市場 sync 再 sync-features。",
            min_stocks_per_day,
        )
        return 0

    latest_date = max(valid_dates)
    raw_latest = df["date"].max()
    if latest_date != raw_latest:
        logger.warning(
            "[DailyFeature] %s 僅 %d 支（低於門檻 %d），fallback 至 %s（%d 支）",
            raw_latest,
            int(date_counts.iloc[0]),
            min_stocks_per_day,
            latest_date,
            int(date_counts[latest_date]),
        )
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
        "turnover_ma20",
        "momentum_20d",
        "volatility_20d",
        "turnover_ratio_5d_20d",
        "high_20d",
        "computed_at",
    ]
    df_out = df_latest[keep_cols].reset_index(drop=True)

    # C2 修復：update_cols 涵蓋所有計算欄位，避免重算後舊特徵殘留
    # （e.g., 同日盤中 + 盤後分別跑 sync-features 時）
    feature_update_cols = [c for c in keep_cols if c not in ("stock_id", "date")]
    written = _upsert_batch(
        DailyFeature,
        df_out,
        ["stock_id", "date"],
        update_cols=feature_update_cols,
    )
    logger.info("[DailyFeature] 已寫入 %d 筆（日期 %s）", written, latest_date)
    return written


def backfill_daily_features(
    start: date,
    end: date | None = None,
    *,
    chunk_days: int = 90,
    lookback_buffer_days: int = 130,
    min_stocks_per_day: int = 1000,
    dry_run: bool = False,
) -> dict[str, int]:
    """回補歷史 `DailyFeature`（B1②）— PIT universe 過濾的資料前提。

    `compute_and_store_daily_features()` 是**增量**設計：只寫最新一日，且錨在
    `date.today()`。PIT 重放需要「當時那一天的特徵」，故需本函數逐日補齊歷史。

    ## PIT 正確性

    所有 rolling 皆為後視窗，計算 D 日特徵時只會用到 <= D 的資料，天然無
    look-ahead。為使 chunk 邊界的 MA60 正確，每個 chunk 會**多讀
    `lookback_buffer_days` 天**作為暖身，但只寫入 chunk 區間內的日期。

    ## 續跑

    以 DB 現況為準：`daily_feature` 已有的日期直接跳過，中斷後重跑自動續行。

    Args:
        start: 起始日（含）。
        end: 結束日（含）；None ＝今日。
        chunk_days: 每批處理的日曆天數（控制記憶體；90 天 ≈ 40 萬筆）。
        lookback_buffer_days: 每批往前多讀的暖身天數（須 > MA60 所需交易日）。
        min_stocks_per_day: 該日 DailyPrice 少於此數視為覆蓋不足，不計算特徵。
        dry_run: 只估算待補日數，不計算也不寫入。

    Returns:
        {"dates": N, "rows": M, "skipped_dates": S}
    """
    init_db()
    result = {"dates": 0, "rows": 0, "skipped_dates": 0}
    if end is None:
        end = date.today()
    if start > end:
        logger.info("[DailyFeature 回補] 範圍為空（start=%s > end=%s）", start, end)
        return result

    with get_session() as session:
        # 有足夠 DailyPrice 覆蓋、值得算特徵的日期
        price_dates = {
            r[0]
            for r in session.execute(
                select(DailyPrice.date)
                .where(DailyPrice.date >= start, DailyPrice.date <= end)
                .group_by(DailyPrice.date)
                .having(func.count(DailyPrice.id) >= min_stocks_per_day)
            ).all()
        }
        done_dates = {
            r[0]
            for r in session.execute(
                select(DailyFeature.date).where(DailyFeature.date >= start, DailyFeature.date <= end).distinct()
            ).all()
        }

    pending = sorted(price_dates - done_dates)
    result["skipped_dates"] = len(price_dates & done_dates)
    logger.info(
        "[DailyFeature 回補] %s ~ %s：待補 %d 日（已有 %d 日），chunk=%d 天",
        start,
        end,
        len(pending),
        result["skipped_dates"],
        chunk_days,
    )
    if dry_run or not pending:
        return result

    pending_set = set(pending)
    cursor = pending[0]
    last = pending[-1]
    while cursor <= last:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), last)
        targets = sorted(d for d in pending_set if cursor <= d <= chunk_end)
        if not targets:
            cursor = chunk_end + timedelta(days=1)
            continue

        warm_start = cursor - timedelta(days=lookback_buffer_days)
        with get_session() as session:
            rows = session.execute(
                select(
                    DailyPrice.stock_id,
                    DailyPrice.date,
                    DailyPrice.high,
                    DailyPrice.close,
                    DailyPrice.volume,
                    DailyPrice.turnover,
                ).where(DailyPrice.date >= warm_start, DailyPrice.date <= chunk_end)
            ).all()

        if not rows:
            cursor = chunk_end + timedelta(days=1)
            continue

        df = pd.DataFrame(rows, columns=["stock_id", "date", "high", "close", "volume", "turnover"])
        df = df[(df["volume"] > 0) & df["close"].notna() & (df["close"] > 0)]
        df = df.sort_values(["stock_id", "date"])
        if df.empty:
            cursor = chunk_end + timedelta(days=1)
            continue

        df = compute_feature_columns(df)
        df_out = df[df["date"].isin(targets)].copy()
        df_out["computed_at"] = pd.Timestamp.utcnow()
        df_out = df_out[DAILY_FEATURE_COLUMNS].reset_index(drop=True)

        update_cols = [c for c in DAILY_FEATURE_COLUMNS if c not in ("stock_id", "date")]
        written = _upsert_batch(DailyFeature, df_out, ["stock_id", "date"], update_cols=update_cols)
        result["dates"] += len(targets)
        result["rows"] += written
        logger.info(
            "[DailyFeature 回補] %s ~ %s：%d 日 / %d 筆（累計 %d 日 / %d 筆）",
            targets[0],
            targets[-1],
            len(targets),
            written,
            result["dates"],
            result["rows"],
        )
        cursor = chunk_end + timedelta(days=1)

    logger.info("[DailyFeature 回補] 完成 — %d 日 / %d 筆", result["dates"], result["rows"])
    return result


def sync_concepts_from_yaml(
    concepts_path: str = "config/concepts.yaml",
    purge_yaml: bool = False,
) -> dict[str, int]:
    """將 concepts.yaml 同步至 ConceptGroup + ConceptMembership。

    Parameters
    ----------
    concepts_path:
        概念定義 YAML 路徑（預設 config/concepts.yaml）。
    purge_yaml:
        True 時先刪除 source="yaml" 的舊記錄再重新匯入（概念重組時用）。

    Returns
    -------
    dict
        {"groups": N, "members": M} 新增/更新筆數統計。
    """
    import yaml

    init_db()

    with open(concepts_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    concepts: dict[str, dict] = raw.get("concepts", {})
    if not concepts:
        logger.warning("[sync-concepts] concepts.yaml 無概念定義，跳過。")
        return {"groups": 0, "members": 0}

    today = date.today()
    groups_count = 0
    members_count = 0

    with get_session() as session:
        if purge_yaml:
            deleted = session.query(ConceptMembership).filter(ConceptMembership.source == "yaml").delete()
            session.commit()
            logger.info("[sync-concepts] 已清除舊 yaml 成員記錄 %d 筆", deleted)

        for name, info in concepts.items():
            desc = info.get("description", "")
            stocks: list[str] = [str(s) for s in info.get("stocks", [])]

            # Upsert ConceptGroup
            existing_group = session.query(ConceptGroup).filter(ConceptGroup.name == name).first()
            if existing_group:
                existing_group.description = desc
                existing_group.updated_at = date.today()
            else:
                session.add(ConceptGroup(name=name, description=desc))
                groups_count += 1
            session.commit()

            # Upsert ConceptMembership（on_conflict_do_nothing）
            for stock_id in stocks:
                existing_member = (
                    session.query(ConceptMembership)
                    .filter(
                        ConceptMembership.concept_name == name,
                        ConceptMembership.stock_id == stock_id,
                    )
                    .first()
                )
                if not existing_member:
                    session.add(
                        ConceptMembership(
                            concept_name=name,
                            stock_id=stock_id,
                            source="yaml",
                            added_date=today,
                        )
                    )
                    members_count += 1
            session.commit()

    logger.info("[sync-concepts] 新增概念 %d 個，新增成員 %d 筆", groups_count, members_count)
    return {"groups": groups_count, "members": members_count}


def sync_concept_tags_from_mops(days: int = 90) -> int:
    """掃描近 days 天的 Announcement，以關鍵字比對更新 ConceptMembership（source="mops"）。

    Parameters
    ----------
    days:
        回溯天數（預設 90 天）。

    Returns
    -------
    int
        新增 ConceptMembership 筆數。
    """
    from src.data.mops_fetcher import classify_concepts

    init_db()
    cutoff = date.today() - timedelta(days=days)
    today = date.today()
    added = 0

    with get_session() as session:
        rows = session.query(Announcement.stock_id, Announcement.subject).filter(Announcement.date >= cutoff).all()

        for stock_id, subject in rows:
            if not subject:
                continue
            matched_concepts = classify_concepts(subject)
            for concept_name in matched_concepts:
                existing = (
                    session.query(ConceptMembership)
                    .filter(
                        ConceptMembership.concept_name == concept_name,
                        ConceptMembership.stock_id == stock_id,
                    )
                    .first()
                )
                if not existing:
                    session.add(
                        ConceptMembership(
                            concept_name=concept_name,
                            stock_id=stock_id,
                            source="mops",
                            added_date=today,
                        )
                    )
                    added += 1

        if added:
            session.commit()

    logger.info("[sync-concepts] MOPS 關鍵字標記新增 %d 筆成員", added)
    return added
