"""全市場選股掃描器 — 純函數與常數。

此模組包含所有模組級純函數與常數，供 MarketScanner 及其子類呼叫。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 事件類型加權值（重要度由高到低）
_EVENT_TYPE_WEIGHTS: dict[str, float] = {
    "governance_change": 5.0,  # 董監改選 / 市場派（最強結構性事件）
    "buyback": 4.0,  # 庫藏股決議（真金白銀護盤）
    "earnings_call": 3.0,
    "investor_day": 2.0,
    "filing": 1.5,
    "revenue": 1.2,
    "general": 1.0,
}


def compute_news_decay_weight(days_ago: int, event_type: str) -> float:
    """計算單則公告的時間衰減加權值。

    公式：exp(-0.2 × days_ago) × type_weight

    衰減常數 0.2：7 天後仍保留 ~25% 權重（exp(-1.4) ≈ 0.247），
    適合「董監改選」等後續發酵型事件。

    Args:
        days_ago: 公告距今天數（≥0）
        event_type: 事件類型（governance_change / buyback / earnings_call /
                    investor_day / filing / revenue / general）

    Returns:
        加權值（≥0）
    """
    type_weight = _EVENT_TYPE_WEIGHTS.get(event_type, 1.0)
    return float(np.exp(-0.2 * max(0, days_ago)) * type_weight)


def compute_abnormal_announcement_rate(
    df_ann_history: pd.DataFrame,
    stock_ids: list[str],
    recent_days: int = 10,
    baseline_days: int = 180,
) -> pd.Series:
    """計算各股公告頻率異常 Z-Score（近 recent_days 與過去 baseline_days 基準比較）。

    算法：
        1. 將 baseline_days 拆為 (baseline_days // recent_days - 1) 個非重疊基準窗口
        2. 計算各基準窗口的公告數 → 估算 μ, σ
        3. Z-Score = (最近 recent_days 公告數 - μ) / max(σ, 1)

    Z-Score > 2  → 異常活躍（外部乘數加成，最高 +50%）
    Z-Score < -1 → 異常沉寂（外部乘數降權，最低至 70%）
    -1 ≤ Z ≤ 2  → 正常範圍（乘數 = 1.0）

    Args:
        df_ann_history: 公告歷史 DataFrame，須含 stock_id, date 欄位
        stock_ids: 候選股代號清單
        recent_days: 近期窗口天數（預設 10）
        baseline_days: 基準期天數（預設 180，約 6 個月）

    Returns:
        Series(index=stock_id, dtype=float)；無歷史資料的股票 Z=0.0
    """
    if df_ann_history.empty or "stock_id" not in df_ann_history.columns:
        return pd.Series(0.0, index=pd.Index(stock_ids))

    today = date.today()
    recent_cutoff = today - timedelta(days=recent_days)
    baseline_start = today - timedelta(days=baseline_days)

    df_hist = df_ann_history[["stock_id", "date"]].copy()
    df_hist = df_hist[df_hist["date"] >= baseline_start]

    # 基準窗口數（排除最近窗口本身）
    n_baseline_windows = baseline_days // recent_days - 1
    if n_baseline_windows < 1:
        return pd.Series(0.0, index=pd.Index(stock_ids))

    z_scores: dict[str, float] = {}
    for sid in stock_ids:
        stock_df = df_hist[df_hist["stock_id"] == sid]
        if stock_df.empty:
            z_scores[sid] = 0.0
            continue

        # 最近 recent_days 的公告數
        recent_count = int((stock_df["date"] >= recent_cutoff).sum())

        # 基準期各窗口的公告數
        window_counts: list[int] = []
        for i in range(1, n_baseline_windows + 1):
            wend = today - timedelta(days=recent_days * i)
            wstart = today - timedelta(days=recent_days * (i + 1))
            cnt = int(((stock_df["date"] >= wstart) & (stock_df["date"] < wend)).sum())
            window_counts.append(cnt)

        if not window_counts:
            z_scores[sid] = 0.0
            continue

        mu = float(np.mean(window_counts))
        sigma = float(np.std(window_counts))
        z_scores[sid] = (recent_count - mu) / max(sigma, 1.0)

    return pd.Series(z_scores).reindex(stock_ids).fillna(0.0)


def compute_taiex_relative_strength(
    df_price: pd.DataFrame,
    taiex_stock_id: str = "TAIEX",
    window: int = 20,
) -> pd.Series:
    """計算個股相對 TAIEX 的 N 日超額報酬（純函數，crisis 模式過濾用）。

    Args:
        df_price: DailyPrice DataFrame，需含 stock_id/date/close 欄位，
                  且應包含 taiex_stock_id 的列
        taiex_stock_id: TAIEX 識別代號（預設 "TAIEX"）
        window: 計算報酬率的回溯天數（預設 20）

    Returns:
        pd.Series(index=stock_id, values=excess_return_Nd)
        - 正值：跑贏 TAIEX，負值：跑輸 TAIEX
        - 不含 TAIEX 本身
        - 資料不足的股票填 0.0（不懲罰新股，避免冷啟動誤殺）
        - TAIEX 資料不存在時返回全 0（不過濾，安全 fallback）
    """
    if df_price.empty:
        return pd.Series(dtype=float)

    # 取 TAIEX 近期收盤序列
    taiex_df = df_price[df_price["stock_id"] == taiex_stock_id].sort_values("date")
    if taiex_df.empty or len(taiex_df) < window + 1:
        # TAIEX 資料不足：返回所有個股 0（不過濾）
        stock_ids = df_price[df_price["stock_id"] != taiex_stock_id]["stock_id"].unique()
        return pd.Series(0.0, index=stock_ids)

    taiex_latest = float(taiex_df["close"].iloc[-1])
    taiex_past = float(taiex_df["close"].iloc[-(window + 1)])
    taiex_ret = (taiex_latest - taiex_past) / taiex_past if taiex_past > 0 else 0.0

    # 計算各個股的 N 日報酬率
    non_taiex = df_price[df_price["stock_id"] != taiex_stock_id]
    results: dict[str, float] = {}
    for sid, grp in non_taiex.groupby("stock_id"):
        grp = grp.sort_values("date")
        if len(grp) < window + 1:
            results[sid] = 0.0  # 資料不足：不懲罰
            continue
        latest = float(grp["close"].iloc[-1])
        past = float(grp["close"].iloc[-(window + 1)])
        stock_ret = (latest - past) / past if past > 0 else 0.0
        results[sid] = stock_ret - taiex_ret

    return pd.Series(results)


def compute_relative_pe_thresholds(
    industry_series: pd.Series,
    pe_series: pd.Series,
    multiplier: float = 1.5,
    fallback_pe: float = 50.0,
    min_industry_count: int = 3,
) -> pd.Series:
    """計算各股票相對估值 PE 門檻（模組級純函數，方便測試）。

    依同產業有效 PE（> 0）中位數 × multiplier 計算個股門檻；
    同業樣本不足 min_industry_count 支時，fallback 至絕對門檻 fallback_pe。

    Args:
        industry_series: 產業分類 Series（與 pe_series 同 index）
        pe_series: PE 比率 Series（> 0 才納入產業中位數計算）
        multiplier: 相對中位數倍數，預設 1.5
        fallback_pe: 樣本不足時的絕對 PE 上限，預設 50.0（取代舊有絕對值 30）
        min_industry_count: 最小產業樣本數，預設 3

    Returns:
        pd.Series（同 index），每股的 PE 門檻值
    """
    if industry_series.empty:
        return pd.Series(dtype=float)

    df_tmp = pd.DataFrame(
        {"industry": industry_series.values, "pe": pe_series.values},
        index=industry_series.index,
    )

    # 有效 PE > 0 的樣本用於計算產業中位數
    valid = df_tmp[df_tmp["pe"] > 0]
    if not valid.empty:
        industry_stats = valid.groupby("industry")["pe"].agg(["median", "count"])
        sufficient = industry_stats[industry_stats["count"] >= min_industry_count]
        threshold_map = (sufficient["median"] * multiplier).to_dict()
    else:
        threshold_map = {}

    # 對每股映射門檻（查不到充足同業資料時用 fallback_pe）
    thresholds = df_tmp["industry"].map(threshold_map).fillna(fallback_pe)
    return thresholds


def compute_eps_sustainability(
    df_financial: pd.DataFrame,
    min_quarters: int = 4,
) -> frozenset[str]:
    """近 min_quarters 季 EPS 配息可持續性評估（模組級純函數）。

    掃描每支股票最近 min_quarters 季的 EPS 記錄，找出任一季 EPS ≤ 0 的股票。

    Args:
        df_financial: 財務資料（需含 stock_id, date, eps 欄位）
        min_quarters: 評估最近幾季，預設 4

    Returns:
        frozenset[str]，代表「EPS 不可持續」的 stock_id（應被排除）。
        - df_financial 為空 → 回傳空集合（呼叫方不做任何排除）
        - 某股無 EPS 記錄 → 不加入集合（pass through，避免冷啟動誤殺）
        - 某股近期有任一季 EPS ≤ 0 → 加入集合（排除）
    """
    if df_financial.empty or "eps" not in df_financial.columns:
        return frozenset()

    df = df_financial[df_financial["eps"].notna()][["stock_id", "date", "eps"]].copy()
    if df.empty:
        return frozenset()

    # 每股取最近 min_quarters 季（date 降序，各股取前 min_quarters 筆）
    recent = df.sort_values(["stock_id", "date"], ascending=[True, False]).groupby("stock_id").head(min_quarters)

    # 任一季 EPS ≤ 0 → 不可持續（一次性高配息 / 業外收益等風險）
    return frozenset(recent.loc[recent["eps"] <= 0, "stock_id"].unique())


def compute_vcp_score(stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
    """VCP（波動收斂形態）評分純函數（SwingScanner Stage 3 加成用）。

    以 close-based 代替 high/low，計算近期波動收斂與量縮狀態：
    - 條件一：近 10 日 close 波動幅度 = (max - min) / mean < 8%（價格整理中）
    - 條件二：近 3 日均量 / 近 20 日均量 < 0.8（量縮，主動賣壓減少）

    兩個條件同時滿足 → +3% composite_score 加成（作為加分而非硬門，不影響其他 Scanner）。
    資料不足時回傳 0.0 加成（安全 fallback）。

    Args:
        stock_ids: 候選股代號清單
        df_price: 日K線 DataFrame，需含 stock_id/date/close/volume 欄位

    Returns:
        DataFrame，欄位：[stock_id, vcp_bonus]
    """
    if df_price.empty or not stock_ids:
        return pd.DataFrame({"stock_id": stock_ids, "vcp_bonus": 0.0})

    sorted_dates = sorted(df_price["date"].unique())
    if len(sorted_dates) < 3:
        return pd.DataFrame({"stock_id": stock_ids, "vcp_bonus": 0.0})

    dates_10d = sorted_dates[-10:] if len(sorted_dates) >= 10 else sorted_dates
    dates_3d = sorted_dates[-3:]

    df_10d = df_price[df_price["date"].isin(dates_10d)]
    df_3d = df_price[df_price["date"].isin(dates_3d)]

    # 條件一：10 日 close 波動幅度（max - min）/ mean
    price_stats = df_10d.groupby("stock_id")["close"].agg(["max", "min", "mean"])
    price_range = (price_stats["max"] - price_stats["min"]) / price_stats["mean"].replace(0, float("nan"))
    price_ok = price_range < 0.08

    # 條件二：3 日均量 / 20 日均量
    vol_3d = df_3d.groupby("stock_id")["volume"].mean()
    vol_20d = df_price.groupby("stock_id")["volume"].apply(lambda s: s.tail(20).mean())
    vol_ratio = (vol_3d / vol_20d.replace(0, float("nan"))).fillna(1.0)
    vol_ok = vol_ratio < 0.8

    vcp_ok = price_ok & vol_ok

    results = []
    for sid in stock_ids:
        bonus = 0.03 if (sid in vcp_ok.index and bool(vcp_ok.get(sid, False))) else 0.0
        results.append({"stock_id": sid, "vcp_bonus": bonus})

    return pd.DataFrame(results)


def _calc_atr14(stock_data: pd.DataFrame) -> float:
    """計算個股 ATR14。

    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR14 = 最近 14 日 TR 均值

    stock_data 需含 high/low/close 欄位，已按日期排序，長度至少 2。
    不足時回傳 0.0。
    """
    data = stock_data.tail(15)
    if len(data) < 2:
        return 0.0

    highs = data["high"].values
    lows = data["low"].values
    closes = data["close"].values

    trs = []
    for i in range(1, len(highs)):
        tr = max(
            float(highs[i] - lows[i]),
            abs(float(highs[i]) - float(closes[i - 1])),
            abs(float(lows[i]) - float(closes[i - 1])),
        )
        trs.append(tr)

    if not trs:
        return 0.0

    return float(np.mean(trs[-14:]))


def _extract_level_lower_bound(level: str) -> int:
    """從持股分級字串中提取下限股數。

    範例：
      "400,001-600,000 Shares" → 400001
      "over 1,000,000"         → 1000000
      "1-999 shares"           → 1
    """
    nums = re.findall(r"\d+", level.replace(",", ""))
    return int(nums[0]) if nums else 0


def compute_whale_score(df_holding: pd.DataFrame) -> pd.DataFrame:
    """計算大戶持股集中度分數（純函數，可獨立測試）。

    大戶定義：持股區間下限 >= 400,000 股（約 400 張）。

    Args:
        df_holding: HoldingDistribution 資料，欄位需包含
                    [date, stock_id, level, percent]

    Returns:
        DataFrame 包含欄位 [stock_id, whale_percent, whale_change]
        - whale_percent: 最新週大戶持股比例 (%)
        - whale_change:  與前一週的差值（正值 = 大戶增持）
    """
    if df_holding.empty:
        return pd.DataFrame(columns=["stock_id", "whale_percent", "whale_change"])

    df = df_holding.copy()
    df["_lower"] = df["level"].apply(_extract_level_lower_bound)
    df["is_whale"] = df["_lower"] >= 400_000

    dates = sorted(df["date"].unique())
    if not dates:
        return pd.DataFrame(columns=["stock_id", "whale_percent", "whale_change"])

    latest_date = dates[-1]
    df_latest = df[df["date"] == latest_date]

    whale_latest = (
        df_latest.groupby("stock_id")
        .apply(lambda g: g.loc[g["is_whale"], "percent"].sum(), include_groups=False)
        .reset_index()
    )
    whale_latest.columns = ["stock_id", "whale_percent"]

    # 週環比變化（大戶增持 = 正訊號）
    if len(dates) >= 2:
        prev_date = dates[-2]
        df_prev = df[df["date"] == prev_date]
        whale_prev = (
            df_prev.groupby("stock_id")
            .apply(lambda g: g.loc[g["is_whale"], "percent"].sum(), include_groups=False)
            .reset_index()
        )
        whale_prev.columns = ["stock_id", "whale_prev_percent"]
        whale_latest = whale_latest.merge(whale_prev, on="stock_id", how="left")
        whale_latest["whale_change"] = whale_latest["whale_percent"] - whale_latest["whale_prev_percent"].fillna(0.0)
    else:
        whale_latest["whale_change"] = 0.0

    return whale_latest[["stock_id", "whale_percent", "whale_change"]]


def compute_sbl_score(df_sbl: pd.DataFrame) -> pd.DataFrame:
    """從最新日借券資料提取 sbl_balance（純函數，可獨立測試）。

    可借券賣出股數越低 → 空頭壓力越小 → 後續評分越高（逆向因子）。
    注意：2026 年 TWSE API 改版後，sbl_change 欄位不再提供，評分僅依 sbl_balance。

    Args:
        df_sbl: SecuritiesLending 資料，欄位需含 [date, stock_id, sbl_balance]

    Returns:
        DataFrame [stock_id, sbl_balance]（只取最新一日）
    """
    if df_sbl.empty:
        return pd.DataFrame(columns=["stock_id", "sbl_balance"])

    required = {"date", "stock_id", "sbl_balance"}
    if not required.issubset(df_sbl.columns):
        return pd.DataFrame(columns=["stock_id", "sbl_balance"])

    latest = df_sbl["date"].max()
    df = df_sbl[df_sbl["date"] == latest][["stock_id", "sbl_balance"]].copy()
    return df.reset_index(drop=True)


def compute_broker_score(df_broker: pd.DataFrame) -> pd.DataFrame:
    """計算主力分點集中度（HHI）與最強主力連續進場天數（純函數，可獨立測試）。

    Args:
        df_broker: BrokerTrade 資料，欄位需含 [date, stock_id, broker_id, buy, sell]

    Returns:
        DataFrame [stock_id, broker_concentration, broker_consecutive_days]
        - broker_concentration:    當日淨買超分點的 HHI（0~1，越高=主力越集中）
        - broker_consecutive_days: 最強主力分點連續淨買超天數（近 5 日）
    """
    required = {"date", "stock_id", "broker_id", "buy", "sell"}
    if df_broker.empty or not required.issubset(df_broker.columns):
        return pd.DataFrame(columns=["stock_id", "broker_concentration", "broker_consecutive_days"])

    df = df_broker.copy()
    df["net_buy"] = df["buy"].fillna(0).astype(int) - df["sell"].fillna(0).astype(int)

    results = []
    for stock_id, grp in df.groupby("stock_id"):
        # ── HHI：以最新交易日計算 ──────────────────────────────────────
        latest_date = grp["date"].max()
        day_df = grp[grp["date"] == latest_date]
        net_buyers = day_df[day_df["net_buy"] > 0]
        if net_buyers.empty:
            hhi = 0.0
        else:
            total_net = net_buyers["net_buy"].sum()
            shares = net_buyers["net_buy"] / total_net
            hhi = float((shares**2).sum())

        # ── 連續天數：找近 5 日最活躍（累計淨買最多）的主力分點 ────────
        recent_dates = sorted(grp["date"].unique())[-5:]
        # 各分點在近 5 日的累計淨買
        broker_net = grp[grp["date"].isin(recent_dates)].groupby("broker_id")["net_buy"].sum()
        if broker_net.empty or broker_net.max() <= 0:
            consec = 0
        else:
            top_broker = broker_net.idxmax()
            top_broker_data = (
                grp[grp["broker_id"] == top_broker].groupby("date")["net_buy"].sum().sort_index(ascending=False)
            )
            consec = 0
            for net in top_broker_data.values:
                if net > 0:
                    consec += 1
                else:
                    break

        results.append(
            {
                "stock_id": stock_id,
                "broker_concentration": hhi,
                "broker_consecutive_days": consec,
            }
        )

    if not results:
        return pd.DataFrame(columns=["stock_id", "broker_concentration", "broker_consecutive_days"])

    return pd.DataFrame(results)


# ====================================================================== #
#  隔日沖大戶偵測 — 靜態黑名單 + 動態行為偵測 + 扣分計算
# ====================================================================== #

# 已知隔日沖券商分點名稱（broker_name 匹配，因同一券商所有分點共用同一 BHID）
_KNOWN_DAYTRADE_BROKER_NAMES: frozenset[str] = frozenset(
    {
        "凱基-台北",
        "凱基-虎尾",
        "富邦-台南",
        "元大-土城永寧",
        "美林",
    }
)


def detect_daytrade_brokers(
    df_broker: pd.DataFrame,
    short_hold_window: int = 3,
    sell_buy_ratio_min: float = 0.70,
    min_events: int = 3,
) -> pd.DataFrame:
    """從 BrokerTrade 歷史識別具有隔日沖行為模式的分點（純函數）。

    演算法（向量化）：
    1. 計算每筆 net = buy - sell
    2. 以 groupby(['stock_id', 'broker_id']) + shift(-1/-2/-3) 建立未來 1~3 日 net
    3. 買進事件（net > 0）+ T+1~T+3 內任一天 net < 0 且 |sell| >= net × sell_buy_ratio_min → 配對成功
    4. daytrade_events >= min_events 的分點被標記為隔日沖

    Args:
        df_broker: BrokerTrade DataFrame，須含 [stock_id, date, broker_id, broker_name, buy, sell]
        short_hold_window: 短持有配對窗口天數（預設 3，覆蓋 T+1~T+3）
        sell_buy_ratio_min: 賣出量須達買入量的此比例才算配對（預設 0.70）
        min_events: 配對成功至少 N 次才標記為隔日沖（預設 3）

    Returns:
        DataFrame [stock_id, broker_id, broker_name, daytrade_events, daytrade_ratio, avg_hold_days]
        空結果時回傳空 DataFrame（含正確欄位）
    """
    _EMPTY = pd.DataFrame(
        columns=["stock_id", "broker_id", "broker_name", "daytrade_events", "daytrade_ratio", "avg_hold_days"]
    )
    required = {"stock_id", "date", "broker_id", "buy", "sell"}
    if df_broker.empty or not required.issubset(df_broker.columns):
        return _EMPTY

    df = df_broker.copy()
    df["net"] = df["buy"].fillna(0).astype("int64") - df["sell"].fillna(0).astype("int64")
    df = df.sort_values(["stock_id", "broker_id", "date"])

    # 向量化：在每個 (stock_id, broker_id) 群組內 shift net 取得未來 1~3 天值
    grp = df.groupby(["stock_id", "broker_id"], sort=False)
    shift_cols = {}
    for i in range(1, min(short_hold_window, 3) + 1):
        col = f"net_t{i}"
        df[col] = grp["net"].shift(-i)
        shift_cols[i] = col

    # 買進事件：今日 net > 0
    buy_mask = df["net"] > 0

    # 配對判定：T+1~T+3 內任一天 net < 0 且 |net_t| >= net_today × ratio
    net_today = df["net"]
    threshold = net_today * sell_buy_ratio_min

    match_any = pd.Series(False, index=df.index)
    best_hold = pd.Series(np.nan, index=df.index)

    for i, col in shift_cols.items():
        cond = buy_mask & (df[col] < 0) & (df[col].abs() >= threshold)
        # 對於首次配對成功的，記錄持有天數
        first_match = cond & ~match_any
        best_hold = best_hold.where(~first_match, i)
        match_any = match_any | cond

    df["is_buy_event"] = buy_mask
    df["is_daytrade_match"] = match_any & buy_mask
    df["hold_days"] = best_hold

    # 聚合：每個 (stock_id, broker_id) 的配對統計
    agg = (
        df[df["is_buy_event"]]
        .groupby(["stock_id", "broker_id"], sort=False)
        .agg(
            total_buy_events=("is_buy_event", "sum"),
            daytrade_events=("is_daytrade_match", "sum"),
            avg_hold_days=("hold_days", "mean"),
        )
        .reset_index()
    )

    # 過濾 min_events
    agg = agg[agg["daytrade_events"] >= min_events].copy()
    if agg.empty:
        return _EMPTY

    agg["daytrade_ratio"] = agg["daytrade_events"] / agg["total_buy_events"].clip(lower=1)
    agg["avg_hold_days"] = agg["avg_hold_days"].fillna(1.0)

    # 補上 broker_name（取最後一筆）
    if "broker_name" in df_broker.columns:
        name_map = df_broker.groupby(["stock_id", "broker_id"])["broker_name"].last().reset_index()
        agg = agg.merge(name_map, on=["stock_id", "broker_id"], how="left")
    else:
        agg["broker_name"] = ""

    return agg[
        ["stock_id", "broker_id", "broker_name", "daytrade_events", "daytrade_ratio", "avg_hold_days"]
    ].reset_index(drop=True)


def compute_daytrade_penalty(
    df_broker: pd.DataFrame,
    df_volume: pd.DataFrame | None = None,
    known_broker_names: frozenset[str] | None = None,
    short_hold_window: int = 3,
    min_events: int = 3,
    min_volume_ratio: float = 0.05,
    instant_volume_ratio: float = 0.10,
) -> pd.DataFrame:
    """計算各股的隔日沖風險扣分值（純函數）。

    三層邏輯：
    1. 行為偵測層：呼叫 detect_daytrade_brokers() 找出動態隔日沖分點
    2. 黑名單層：_KNOWN_DAYTRADE_BROKER_NAMES union known_broker_names
    3. 即時風險層：黑名單分點單日買超 ≥ 總成交量 × instant_volume_ratio → 直接高風險

    扣分計算：
    - dt_brokers = 行為偵測 ∪ 黑名單中最新日有淨買超者
    - dt_net_buy = 所有 dt_brokers 淨買超加總（群聚效應）
    - raw_penalty = dt_net_buy / total_net_buy（total_net_buy <= 0 → penalty=0）
    - 流動性閾值：dt_net_buy < avg_volume_20d × min_volume_ratio → penalty 降半
    - daytrade_penalty = min(1.0, raw_penalty)

    Args:
        df_broker: BrokerTrade DataFrame [stock_id, date, broker_id, broker_name, buy, sell]
        df_volume: 各股 20 日均量 DataFrame [stock_id, avg_volume_20d]（可選）
        known_broker_names: 額外的隔日沖分點名稱集合（union 至預設黑名單）
        short_hold_window: detect_daytrade_brokers 的配對窗口
        min_events: detect_daytrade_brokers 的最低配對次數
        min_volume_ratio: 流動性閾值（隔日沖買超 < 均量 × 此值 → penalty 降半）
        instant_volume_ratio: 即時風險閾值（黑名單分點買超 ≥ 總量 × 此值 → 直接觸發）

    Returns:
        DataFrame [stock_id, daytrade_penalty, has_daytrade_risk, top_dt_brokers]
        - daytrade_penalty: 0.0~1.0（0=無風險，1=極高風險）
        - has_daytrade_risk: 布林值
        - top_dt_brokers: 逗號分隔的隔日沖分點名稱
    """
    _EMPTY = pd.DataFrame(columns=["stock_id", "daytrade_penalty", "has_daytrade_risk", "top_dt_brokers"])
    required = {"stock_id", "date", "broker_id", "buy", "sell"}
    if df_broker.empty or not required.issubset(df_broker.columns):
        # 回傳 0 penalty（無資料不扣分）
        if df_broker.empty:
            return _EMPTY
        stock_ids = df_broker["stock_id"].unique()
        return pd.DataFrame(
            {"stock_id": stock_ids, "daytrade_penalty": 0.0, "has_daytrade_risk": False, "top_dt_brokers": ""}
        )

    # 合併黑名單
    all_known = _KNOWN_DAYTRADE_BROKER_NAMES
    if known_broker_names:
        all_known = all_known | known_broker_names

    # ── 1. 行為偵測 ──────────────────────────────────────────
    detected = detect_daytrade_brokers(
        df_broker,
        short_hold_window=short_hold_window,
        min_events=min_events,
    )
    # detected: [stock_id, broker_id, broker_name, daytrade_events, ...]
    detected_pairs: set[tuple[str, str]] = set()
    if not detected.empty:
        detected_pairs = set(zip(detected["stock_id"], detected["broker_id"], strict=False))

    # ── 2. 找出最新交易日各分點資料 ──────────────────────────
    df = df_broker.copy()
    df["net"] = df["buy"].fillna(0).astype("int64") - df["sell"].fillna(0).astype("int64")

    results = []
    for stock_id, stock_grp in df.groupby("stock_id", sort=False):
        latest_date = stock_grp["date"].max()
        day_df = stock_grp[stock_grp["date"] == latest_date].copy()

        # 判定哪些分點是隔日沖
        dt_mask = pd.Series(False, index=day_df.index)

        # 黑名單匹配（broker_name）
        if "broker_name" in day_df.columns:
            for name in all_known:
                dt_mask = dt_mask | (day_df["broker_name"].fillna("") == name)

        # 行為偵測匹配（stock_id, broker_id）
        for idx, row in day_df.iterrows():
            if (stock_id, row["broker_id"]) in detected_pairs:
                dt_mask.at[idx] = True

        # 隔日沖分點的淨買超加總
        dt_day = day_df[dt_mask]
        dt_net_buy = max(0, dt_day["net"].sum()) if not dt_day.empty else 0

        # 全部分點的淨買超加總（僅計算淨買超者）
        buyers = day_df[day_df["net"] > 0]
        total_net_buy = buyers["net"].sum() if not buyers.empty else 0

        # 計算 raw penalty
        if total_net_buy <= 0 or dt_net_buy <= 0:
            penalty = 0.0
        else:
            penalty = min(1.0, dt_net_buy / total_net_buy)

        # ── 3. 即時風險：黑名單分點佔當日總買量 ≥ instant_volume_ratio ──
        total_buy = day_df["buy"].fillna(0).sum()
        if total_buy > 0 and "broker_name" in day_df.columns:
            for name in all_known:
                name_mask = day_df["broker_name"].fillna("") == name
                name_buy = day_df.loc[name_mask, "buy"].fillna(0).sum()
                if name_buy / total_buy >= instant_volume_ratio:
                    penalty = max(penalty, 0.5)  # 至少 0.5 penalty
                    break

        # ── 4. 流動性閾值：小量不扣重 ──────────────────────────
        if penalty > 0 and df_volume is not None and not df_volume.empty:
            vol_row = df_volume[df_volume["stock_id"] == stock_id]
            if not vol_row.empty:
                avg_vol = vol_row["avg_volume_20d"].values[0]
                if avg_vol > 0 and dt_net_buy < avg_vol * min_volume_ratio:
                    penalty *= 0.5  # 降半

        # 收集隔日沖分點名稱
        top_names = []
        if not dt_day.empty and "broker_name" in dt_day.columns:
            dt_sorted = dt_day.sort_values("net", ascending=False)
            top_names = [n for n in dt_sorted["broker_name"].values if n and str(n) != "nan"][:3]

        results.append(
            {
                "stock_id": stock_id,
                "daytrade_penalty": round(penalty, 4),
                "has_daytrade_risk": penalty > 0,
                "top_dt_brokers": ",".join(top_names),
            }
        )

    if not results:
        return _EMPTY

    return pd.DataFrame(results)


def compute_smart_broker_score(
    df_broker: pd.DataFrame,
    current_prices: dict[str, float],
    min_win_rate: float = 0.60,
    min_profit_factor: float = 1.50,
    min_sell_events: int = 3,
    min_buy_value: float = 5_000_000.0,
    recent_days: int = 3,
) -> pd.DataFrame:
    """識別「高勝率+高獲利因子」Smart Broker 與「純蓄積型」Accumulation Broker（純函數）。

    Smart Broker 判定（同時滿足）：
    - win_rate >= min_win_rate：有獲利賣出比例
    - profit_factor >= min_profit_factor：Σ獲利金額 / Σ虧損金額（防多小贏/一大虧陷阱）
    - sell_events >= min_sell_events：至少 N 次賣出（降低運氣成分）
    - total_buy_value >= min_buy_value：排除散單雜訊（TWD）

    Accumulation Broker 判定（同時滿足）：
    - sell_ratio <= 0.10：幾乎不賣（地緣/公司派大戶）
    - net_position > 0：持有淨部位
    - total_buy_value >= min_buy_value：排除散單
    - position_trend_up：後半段倉位 > 前半段（持續蓄積中）

    計算僅針對 Stage 2 粗篩後的候選股（~150 支），不涉及全市場。

    Args:
        df_broker: BrokerTrade 資料，欄位需含
                   [date, stock_id, broker_id, buy, sell, buy_price, sell_price]
        current_prices: {stock_id: close_price} 當日收盤字典
        min_win_rate: Smart Broker 最低勝率門檻（預設 0.60）
        min_profit_factor: Smart Broker 最低獲利因子門檻（預設 1.50）
        min_sell_events: Smart Broker 最低賣出次數（預設 3）
        min_buy_value: 最低買入總金額（TWD，預設 500 萬）
        recent_days: 「近期活躍」的交易日天數（預設 3）

    Returns:
        DataFrame [stock_id, smart_broker_score, accum_broker_score, smart_broker_factor]
        - smart_broker_score:   高勝率+高獲利分點的活躍加權評分 [0, 1]
        - accum_broker_score:   蓄積型分點的底撐評分 [0, 1]
        - smart_broker_factor:  合成因子 = 0.60 × smart + 0.40 × accum [0, 1]
    """
    _EMPTY = pd.DataFrame(columns=["stock_id", "smart_broker_score", "accum_broker_score", "smart_broker_factor"])
    required = {"date", "stock_id", "broker_id", "buy", "sell", "buy_price", "sell_price"}
    if df_broker.empty or not required.issubset(df_broker.columns):
        return _EMPTY

    df = df_broker.copy()
    df["buy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0.0)
    df["sell"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0.0)
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce").fillna(0.0)
    df["sell_price"] = pd.to_numeric(df["sell_price"], errors="coerce").fillna(0.0)
    df["net_buy"] = df["buy"] - df["sell"]

    broker_metrics: list[dict] = []
    for (stock_id, broker_id), grp in df.groupby(["stock_id", "broker_id"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        all_dates = sorted(grp["date"].unique())

        # 使用 numpy 陣列取代 iterrows，避免 pandas 逐列 Python 迴圈開銷
        buy_arr = grp["buy"].to_numpy(dtype=np.float64)
        sell_arr = grp["sell"].to_numpy(dtype=np.float64)
        bp_arr = grp["buy_price"].to_numpy(dtype=np.float64)
        sp_arr = grp["sell_price"].to_numpy(dtype=np.float64)
        net_buy_arr = grp["net_buy"].to_numpy(dtype=np.float64)

        avg_cost = 0.0
        net_position = 0.0
        total_profit = 0.0
        total_loss = 0.0
        wins = 0
        sell_events = 0
        total_buy_value = 0.0

        for i in range(len(buy_arr)):
            buy_sh = buy_arr[i]
            sell_sh = sell_arr[i]
            bp = bp_arr[i]
            sp = sp_arr[i]

            if buy_sh > 0 and bp > 0:
                total_buy_value += buy_sh * bp
                new_pos = net_position + buy_sh
                if new_pos > 0:
                    avg_cost = (avg_cost * net_position + bp * buy_sh) / new_pos
                net_position = new_pos

            if sell_sh > 0 and sp > 0 and avg_cost > 0:
                sell_events += 1
                pnl = (sp - avg_cost) * sell_sh
                if pnl > 0:
                    wins += 1
                    total_profit += pnl
                else:
                    total_loss += abs(pnl)
                net_position = max(0.0, net_position - sell_sh)

        win_rate = wins / sell_events if sell_events > 0 else 0.0
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        elif total_profit > 0:
            profit_factor = 999.0
        else:
            profit_factor = 0.0
        hist_pnl = total_profit - total_loss

        # 近期活躍度（向量化：最後 recent_days 個交易日的淨買超）
        n_dates = len(all_dates)
        recent_start = max(0, n_dates - recent_days)
        recent_date_set = set(all_dates[recent_start:])
        recent_mask = grp["date"].isin(recent_date_set)
        recent_net = float(net_buy_arr[recent_mask.to_numpy()].sum())

        # 倉位趨勢（前後半段比較，向量化）
        mid = n_dates // 2
        first_half_set = set(all_dates[:mid])
        last_half_set = set(all_dates[mid:])
        first_mask = grp["date"].isin(first_half_set).to_numpy()
        last_mask = grp["date"].isin(last_half_set).to_numpy()
        first_net = float(net_buy_arr[first_mask].sum())
        last_net = float(net_buy_arr[last_mask].sum())
        position_trend_up = last_net > first_net

        # 賣出比例（向量化）
        total_buy_sh = float(buy_arr.sum())
        total_sell_sh = float(sell_arr.sum())
        sell_ratio = total_sell_sh / total_buy_sh if total_buy_sh > 0 else 0.0

        broker_metrics.append(
            {
                "stock_id": stock_id,
                "broker_id": broker_id,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "sell_events": sell_events,
                "total_buy_value": total_buy_value,
                "hist_pnl": hist_pnl,
                "recent_net": recent_net,
                "net_position": net_position,
                "avg_cost": avg_cost,
                "sell_ratio": sell_ratio,
                "position_trend_up": position_trend_up,
            }
        )

    if not broker_metrics:
        return _EMPTY

    bm = pd.DataFrame(broker_metrics)

    # ── Smart Broker 篩選 ──────────────────────────────────────────────
    smart_mask = (
        (bm["win_rate"] >= min_win_rate)
        & (bm["profit_factor"] >= min_profit_factor)
        & (bm["sell_events"] >= min_sell_events)
        & (bm["total_buy_value"] >= min_buy_value)
    )
    smart_brokers = bm[smart_mask].copy()

    # ── Accumulation Broker 篩選 ──────────────────────────────────────
    accum_mask = (
        (bm["sell_ratio"] <= 0.10)
        & (bm["net_position"] > 0)
        & (bm["total_buy_value"] >= min_buy_value)
        & bm["position_trend_up"]
    )
    accum_brokers = bm[accum_mask].copy()

    # ── 各股彙總 ──────────────────────────────────────────────────────
    results_smart: list[dict] = []
    for stock_id in bm["stock_id"].unique():
        # Smart Broker 彙總
        sb = smart_brokers[smart_brokers["stock_id"] == stock_id]
        active_sb = sb[sb["recent_net"] > 0] if not sb.empty else sb
        smart_score_raw = float((active_sb["hist_pnl"] * active_sb["recent_net"]).sum()) if not active_sb.empty else 0.0
        avg_win_rate = float(sb["win_rate"].mean()) if not sb.empty else 0.0
        avg_pf = float(sb["profit_factor"].clip(upper=10.0).mean()) if not sb.empty else 0.0

        # Smart Broker 未實現損益
        holding_sb = sb[sb["net_position"] > 0].copy() if not sb.empty else sb
        curr_price = current_prices.get(str(stock_id), 0.0)
        if not holding_sb.empty and curr_price > 0:
            holding_sb["unrealized"] = holding_sb["avg_cost"].apply(lambda c: (curr_price - c) / c if c > 0 else 0.0)
            avg_unrealized = float(holding_sb["unrealized"].mean())
        else:
            avg_unrealized = 0.0

        # Accumulation Broker 彙總
        ab = accum_brokers[accum_brokers["stock_id"] == stock_id]
        accum_count = len(ab)
        if accum_count > 0 and curr_price > 0:
            ab = ab.copy()
            ab["proximity"] = ab["avg_cost"].apply(lambda c: (curr_price - c) / c if c > 0 else 0.0)
            avg_proximity = float(ab["proximity"].mean())
            avg_trend_strength = float(ab["position_trend_up"].mean())
        else:
            avg_proximity = 0.0
            avg_trend_strength = 0.0

        results_smart.append(
            {
                "stock_id": stock_id,
                "smart_score_raw": smart_score_raw,
                "avg_win_rate": avg_win_rate,
                "avg_pf": avg_pf,
                "avg_unrealized": avg_unrealized,
                "accum_count": accum_count,
                "avg_proximity": avg_proximity,
                "avg_trend_strength": avg_trend_strength,
            }
        )

    if not results_smart:
        return _EMPTY

    res = pd.DataFrame(results_smart)

    def _rank(s: pd.Series) -> pd.Series:
        if len(s) <= 1:
            return pd.Series([0.5] * len(s), index=s.index)
        return s.rank(pct=True)

    res["smart_broker_score"] = (
        _rank(res["smart_score_raw"]) * 0.40
        + _rank(res["avg_win_rate"]) * 0.25
        + _rank(res["avg_pf"]) * 0.25
        + _rank(res["avg_unrealized"]) * 0.10
    )

    res["accum_broker_score"] = (
        _rank(res["accum_count"]) * 0.40 + _rank(res["avg_proximity"]) * 0.35 + _rank(res["avg_trend_strength"]) * 0.25
    )

    res["smart_broker_factor"] = res["smart_broker_score"] * 0.60 + res["accum_broker_score"] * 0.40

    return res[["stock_id", "smart_broker_score", "accum_broker_score", "smart_broker_factor"]].reset_index(drop=True)


@dataclass
class DiscoveryResult:
    """掃描結果資料容器。"""

    rankings: pd.DataFrame
    total_stocks: int
    after_coarse: int
    scan_date: date = field(default_factory=date.today)
    sector_summary: pd.DataFrame | None = None
    mode: str = "momentum"
