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

    公式：exp(-decay_constant × days_ago) × type_weight

    衰減常數依事件類型分化：
      - 結構性事件（governance_change/buyback）：0.07（半衰期 ~10 天）
      - 一般性事件（revenue/general）：0.15（半衰期 ~4.6 天）
      - 其他事件：0.12（半衰期 ~5.8 天）

    Args:
        days_ago: 公告距今天數（≥0）
        event_type: 事件類型（governance_change / buyback / earnings_call /
                    investor_day / filing / revenue / general）

    Returns:
        加權值（≥0）
    """
    from src.constants import (
        NEWS_DECAY_DEFAULT,
        NEWS_DECAY_STRUCTURAL,
        NEWS_DECAY_TRANSIENT,
        NEWS_STRUCTURAL_TYPES,
        NEWS_TRANSIENT_TYPES,
    )

    type_weight = _EVENT_TYPE_WEIGHTS.get(event_type, 1.0)

    if event_type in NEWS_STRUCTURAL_TYPES:
        decay = NEWS_DECAY_STRUCTURAL
    elif event_type in NEWS_TRANSIENT_TYPES:
        decay = NEWS_DECAY_TRANSIENT
    else:
        decay = NEWS_DECAY_DEFAULT

    return float(np.exp(-decay * max(0, days_ago)) * type_weight)


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
    # 零均價股票 → NaN → inf，確保被過濾掉（不符合 < 0.08）
    price_stats = df_10d.groupby("stock_id")["close"].agg(["max", "min", "mean"])
    price_range = (price_stats["max"] - price_stats["min"]) / price_stats["mean"].replace(0, float("nan"))
    price_range = price_range.fillna(float("inf"))
    price_ok = price_range < 0.08

    # 條件二：3 日均量 / 20 日均量
    # 缺少成交量資料時 → inf，確保不通過收縮檢查（< 0.8）
    vol_3d = df_3d.groupby("stock_id")["volume"].mean()
    # 向量化計算 20 日均量：rolling(20) 後取各股最後一筆
    _sorted = df_price.sort_values(["stock_id", "date"])
    _sorted["_vol_ma20"] = _sorted.groupby("stock_id")["volume"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    vol_20d = _sorted.groupby("stock_id")["_vol_ma20"].last()
    vol_ratio = (vol_3d / vol_20d.replace(0, float("nan"))).fillna(float("inf"))
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
        - broker_consecutive_days: 最強主力分點連續淨買超天數（近 7 日）
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
            if total_net < 1e-6:
                hhi = 0.0
            else:
                shares = net_buyers["net_buy"] / total_net
                hhi = float((shares**2).sum())

        # ── 連續天數：找近 7 日最活躍（累計淨買最多）的主力分點 ────────
        recent_dates = sorted(grp["date"].unique())[-7:]
        # 各分點在近 7 日的累計淨買
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


def compute_institutional_persistence(
    df_inst: pd.DataFrame,
    stock_ids: list[str],
    window: int = 10,
) -> pd.DataFrame:
    """計算法人連續性因子 — 近 N 日中法人淨買超為正的天數比例（純函數）。

    連續性高（如 8/10 天為正）代表法人持續布局，而非一日行情。
    用於區分「真波段吸籌」與「假主力 / 隔日沖」。

    Args:
        df_inst: InstitutionalInvestor 資料，欄位需含 [stock_id, date, net]
        stock_ids: 候選股代號清單
        window: 觀察窗口天數（預設 10）

    Returns:
        DataFrame [stock_id, inst_persistence, inst_positive_days]
        - inst_persistence:   正淨買超天數佔比（0.0~1.0），資料不足時回傳 0.5（中性）
        - inst_positive_days: 正淨買超天數（整數）
    """
    result_cols = ["stock_id", "inst_persistence", "inst_positive_days"]

    if df_inst.empty or not {"stock_id", "date", "net"}.issubset(df_inst.columns):
        return pd.DataFrame({"stock_id": stock_ids, "inst_persistence": 0.5, "inst_positive_days": 0})

    # 取最近 window 個交易日
    all_dates = sorted(df_inst["date"].unique())
    recent_dates = set(all_dates[-window:] if len(all_dates) >= window else all_dates)
    actual_window = len(recent_dates)

    if actual_window == 0:
        return pd.DataFrame({"stock_id": stock_ids, "inst_persistence": 0.5, "inst_positive_days": 0})

    # 篩選近 N 日資料，按 stock_id + date 彙總法人淨買超
    recent = df_inst[df_inst["date"].isin(recent_dates)]
    daily_net = recent.groupby(["stock_id", "date"])["net"].sum().reset_index()

    # 計算每檔股票的正淨買超天數
    positive_days = daily_net[daily_net["net"] > 0].groupby("stock_id").size().reset_index(name="inst_positive_days")

    # 組裝結果
    result = pd.DataFrame({"stock_id": stock_ids})
    result = result.merge(positive_days, on="stock_id", how="left")
    result["inst_positive_days"] = result["inst_positive_days"].fillna(0).astype(int)

    # 有出現在 df_inst 的股票：用實際天數計算比例；未出現的：中性 0.5
    stocks_in_data = set(df_inst["stock_id"].unique())
    result["inst_persistence"] = result.apply(
        lambda r: r["inst_positive_days"] / actual_window if r["stock_id"] in stocks_in_data else 0.5,
        axis=1,
    )

    return result[result_cols]


def compute_inst_net_buy_slope(
    df_inst: pd.DataFrame,
    stock_ids: list[str],
    window: int = 10,
) -> pd.DataFrame:
    """計算法人淨買超斜率 — 判斷法人買盤是「加速」還是「退潮」（純函數）。

    使用最小平方法計算近 window 日法人日淨買超的線性斜率，並以
    日均量歸一化（slope / mean_abs_net_buy）使跨股可比。

    斜率正 = 法人買盤加速（吸籌初期或加碼）；
    斜率負 = 法人買盤減退（出貨或退場）。

    Args:
        df_inst: InstitutionalInvestor 資料，欄位需含 [stock_id, date, net]
        stock_ids: 候選股代號清單
        window: 觀察窗口天數（預設 10）

    Returns:
        DataFrame [stock_id, inst_slope]
        - inst_slope: 歸一化斜率，正值=加速，負值=減退；
          資料不足（< 3 天）回傳 0.0（中性）
    """
    if df_inst.empty or not {"stock_id", "date", "net"}.issubset(df_inst.columns):
        return pd.DataFrame({"stock_id": stock_ids, "inst_slope": 0.0})

    # 按 stock_id + date 彙總淨買超
    daily_net = df_inst.groupby(["stock_id", "date"])["net"].sum().reset_index()

    # 取最近 window 個交易日
    all_dates = sorted(daily_net["date"].unique())
    recent_dates = set(all_dates[-window:] if len(all_dates) >= window else all_dates)
    recent = daily_net[daily_net["date"].isin(recent_dates)]

    results = []
    for sid in stock_ids:
        stock_data = recent[recent["stock_id"] == sid].sort_values("date")
        n = len(stock_data)
        if n < 3:
            results.append({"stock_id": sid, "inst_slope": 0.0})
            continue

        # 線性回歸斜率：y = net_buy, x = 0,1,2,...,n-1
        y = stock_data["net"].values.astype(float)
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            results.append({"stock_id": sid, "inst_slope": 0.0})
            continue
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom

        # 歸一化：除以日均絕對淨買超（避免大型股斜率天然大）
        mean_abs = np.abs(y).mean()
        norm_slope = slope / mean_abs if mean_abs > 0 else 0.0

        results.append({"stock_id": sid, "inst_slope": float(norm_slope)})

    return pd.DataFrame(results)


def compute_hhi_trend(
    df_broker: pd.DataFrame,
    stock_ids: list[str],
    short_window: int = 3,
    long_window: int = 7,
) -> pd.DataFrame:
    """計算分點集中度（HHI）趨勢 — 判斷主力是「吸籌」還是「出貨」（純函數）。

    逐日計算各股 HHI（淨買超分點的赫芬達指數），比較近期均值 vs 較長期均值：
    - HHI 趨勢上升 = 籌碼越來越集中（主力加碼吸籌） → 加分
    - HHI 趨勢下降 = 籌碼分散（多分點出貨） → 扣分

    建議搭配 HHI 絕對值使用：高 HHI + 趨勢上升 → 強訊號。

    Args:
        df_broker: BrokerTrade 資料，欄位需含 [date, stock_id, broker_id, buy, sell]
        stock_ids: 候選股代號清單
        short_window: 短期均值天數（預設 3）
        long_window: 長期均值天數（預設 7）

    Returns:
        DataFrame [stock_id, hhi_trend, hhi_short_avg]
        - hhi_trend:     short_avg - long_avg（正=集中化，負=分散化）
        - hhi_short_avg: 近期 HHI 均值（0~1）
        資料不足時回傳 0.0（中性）
    """
    required = {"date", "stock_id", "broker_id", "buy", "sell"}
    if df_broker.empty or not required.issubset(df_broker.columns):
        return pd.DataFrame({"stock_id": stock_ids, "hhi_trend": 0.0, "hhi_short_avg": 0.0})

    df = df_broker.copy()
    df["net_buy"] = df["buy"].fillna(0).astype("int64") - df["sell"].fillna(0).astype("int64")

    results = []
    for sid in stock_ids:
        stock_df = df[df["stock_id"] == sid]
        if stock_df.empty:
            results.append({"stock_id": sid, "hhi_trend": 0.0, "hhi_short_avg": 0.0})
            continue

        # 逐日計算 HHI
        dates_sorted = sorted(stock_df["date"].unique())
        daily_hhis: list[float] = []
        for d in dates_sorted:
            day_df = stock_df[stock_df["date"] == d]
            net_buyers = day_df[day_df["net_buy"] > 0]
            if net_buyers.empty:
                daily_hhis.append(0.0)
            else:
                total_net = net_buyers["net_buy"].sum()
                shares = net_buyers["net_buy"] / total_net
                daily_hhis.append(float((shares**2).sum()))

        n = len(daily_hhis)
        if n < 2:
            results.append({"stock_id": sid, "hhi_trend": 0.0, "hhi_short_avg": daily_hhis[0] if n == 1 else 0.0})
            continue

        # 短期與長期均值
        short_vals = daily_hhis[-short_window:] if n >= short_window else daily_hhis
        long_vals = daily_hhis[-long_window:] if n >= long_window else daily_hhis
        hhi_short = sum(short_vals) / len(short_vals)
        hhi_long = sum(long_vals) / len(long_vals)

        results.append(
            {
                "stock_id": sid,
                "hhi_trend": float(hhi_short - hhi_long),
                "hhi_short_avg": float(hhi_short),
            }
        )

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


# ------------------------------------------------------------------ #
#  量價背離偵測（Volume-Price Divergence）
# ------------------------------------------------------------------ #

# Regime 別最低 composite_score 門檻（低於此分數的推薦不輸出）
MIN_SCORE_THRESHOLDS: dict[str, float] = {
    "bull": 0.45,
    "sideways": 0.50,
    "bear": 0.55,
    "crisis": 0.60,
}

# 同產業推薦上限（top_n_results 對應的最大同產業數量比例）
SECTOR_MAX_RATIO: float = 0.25  # 同產業最多佔推薦總數 25%
SECTOR_MIN_CAP: int = 3  # 同產業至少保留 3 檔（小推薦數時避免過度限制）


def compute_volume_price_divergence(
    df_price: pd.DataFrame,
    stock_ids: list[str],
    window: int = 5,
) -> pd.DataFrame:
    """計算量價背離分數（純函數）。

    衡量近 N 日價格方向與成交量方向的一致性：
    - 健康上漲 = 價漲量增（correlation > 0）→ 加分
    - 危險信號 = 價漲量縮（correlation < 0）→ 減分

    計算方式：
        1. 取近 window+1 日 close 與 volume
        2. 計算逐日變化率（pct_change）
        3. 求 Pearson 相關係數
        4. 映射為 penalty/bonus：
           - corr < -0.3  → penalty = -0.05（嚴重背離）
           - corr < 0     → penalty = -0.02（輕度背離）
           - corr >= 0.3  → bonus = +0.02（量價齊揚）
           - 其餘 → 0（中性）

    Args:
        df_price: 含 stock_id / date / close / volume 欄位的 DataFrame
        stock_ids: 要評估的股票清單
        window: 背離計算的回溯天數（預設 5）

    Returns:
        DataFrame(stock_id, vp_divergence)  值域 [-0.05, +0.02]
    """
    if df_price.empty or not stock_ids:
        return pd.DataFrame({"stock_id": stock_ids, "vp_divergence": [0.0] * len(stock_ids)})

    sorted_price = df_price.sort_values(["stock_id", "date"])
    grouped = sorted_price.groupby("stock_id", sort=False)

    results: list[dict] = []
    for sid in stock_ids:
        if sid not in grouped.groups:
            results.append({"stock_id": sid, "vp_divergence": 0.0})
            continue

        grp = grouped.get_group(sid)
        # 需要 window+1 筆資料才能算 window 筆 pct_change
        if len(grp) < window + 2:
            results.append({"stock_id": sid, "vp_divergence": 0.0})
            continue

        recent = grp.tail(window + 1)
        close_chg = recent["close"].pct_change().dropna()
        vol_chg = recent["volume"].astype(float).pct_change().dropna()

        if len(close_chg) < 3 or vol_chg.std() == 0 or close_chg.std() == 0:
            results.append({"stock_id": sid, "vp_divergence": 0.0})
            continue

        corr = float(
            np.corrcoef(
                close_chg.values[-window:],
                vol_chg.values[-window:],
            )[0, 1]
        )

        if np.isnan(corr):
            adj = 0.0
        elif corr < -0.3:
            adj = -0.05  # 嚴重背離
        elif corr < 0:
            adj = -0.02  # 輕度背離
        elif corr >= 0.3:
            adj = 0.02  # 量價齊揚
        else:
            adj = 0.0

        results.append({"stock_id": sid, "vp_divergence": adj})

    return pd.DataFrame(results)


def compute_momentum_decay(
    df_price: pd.DataFrame,
    stock_ids: list[str],
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    price_lookback: int = 5,
) -> pd.DataFrame:
    """偵測動量衰減訊號（RSI 頂背離 + MACD 柱狀縮短）。

    **RSI 頂背離**：近 price_lookback 日價格創新高，但 RSI14 未創新高 → 動能減弱。
    **MACD 柱縮短**：MACD histogram 連續 2 天以上縮短 → 趨勢轉弱。

    兩個訊號各產生一個布林值，最終合成連續衰減因子：
    - 兩訊號同時觸發 → -0.06（嚴重衰減）
    - 單一訊號觸發 → -0.03（輕度衰減）
    - 無訊號 → 0.0

    Args:
        df_price: 含 stock_id / date / close / high 欄位的 DataFrame
        stock_ids: 要評估的股票清單
        rsi_window: RSI 計算期間（預設 14）
        macd_fast: MACD 快線期間（預設 12）
        macd_slow: MACD 慢線期間（預設 26）
        macd_signal: MACD 信號線期間（預設 9）
        price_lookback: 價格/RSI 回溯比較天數（預設 5）

    Returns:
        DataFrame(stock_id, momentum_decay)  值域 [-0.06, 0.0]
    """
    if df_price.empty or not stock_ids:
        return pd.DataFrame({"stock_id": stock_ids, "momentum_decay": [0.0] * len(stock_ids)})

    min_required = macd_slow + macd_signal + 5  # 最少需要的資料筆數
    sorted_price = df_price.sort_values(["stock_id", "date"])
    grouped = sorted_price.groupby("stock_id", sort=False)

    results: list[dict] = []
    for sid in stock_ids:
        if sid not in grouped.groups:
            results.append({"stock_id": sid, "momentum_decay": 0.0})
            continue

        grp = grouped.get_group(sid)
        if len(grp) < min_required:
            results.append({"stock_id": sid, "momentum_decay": 0.0})
            continue

        close = grp["close"].astype(float).values
        high = grp["high"].astype(float).values if "high" in grp.columns else close

        # ── RSI 頂背離偵測 ─────────────────────────────────────
        # 計算 RSI14
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        # EMA-based RSI
        avg_gain = np.full(len(delta), np.nan)
        avg_loss = np.full(len(delta), np.nan)
        if len(delta) >= rsi_window:
            avg_gain[rsi_window - 1] = np.mean(gain[:rsi_window])
            avg_loss[rsi_window - 1] = np.mean(loss[:rsi_window])
            for i in range(rsi_window, len(delta)):
                avg_gain[i] = (avg_gain[i - 1] * (rsi_window - 1) + gain[i]) / rsi_window
                avg_loss[i] = (avg_loss[i - 1] * (rsi_window - 1) + loss[i]) / rsi_window

        rsi = np.full(len(delta), np.nan)
        valid = ~np.isnan(avg_gain) & (avg_loss != 0)
        rsi[valid] = 100 - 100 / (1 + avg_gain[valid] / avg_loss[valid])
        # avg_loss == 0 且 avg_gain 不為 NaN → RSI = 100
        zero_loss = ~np.isnan(avg_gain) & (avg_loss == 0)
        rsi[zero_loss] = 100.0

        rsi_divergence = False
        if len(rsi) >= price_lookback + 1 and not np.all(np.isnan(rsi[-price_lookback:])):
            recent_high = high[-price_lookback:]
            prev_high = high[-(price_lookback * 2) : -price_lookback] if len(high) >= price_lookback * 2 else None
            recent_rsi = rsi[-(price_lookback + 1) :]  # RSI 比 close 少一筆
            prev_rsi = (
                rsi[-(price_lookback * 2 + 1) : -(price_lookback + 1)] if len(rsi) >= price_lookback * 2 + 1 else None
            )

            if prev_high is not None and prev_rsi is not None:
                # 價格新高但 RSI 未新高
                recent_rsi_clean = recent_rsi[~np.isnan(recent_rsi)]
                prev_rsi_clean = prev_rsi[~np.isnan(prev_rsi)]
                if (
                    len(recent_rsi_clean) > 0
                    and len(prev_rsi_clean) > 0
                    and np.max(recent_high) > np.max(prev_high)
                    and np.max(recent_rsi_clean) < np.max(prev_rsi_clean)
                ):
                    rsi_divergence = True

        # ── MACD 柱狀縮短偵測 ──────────────────────────────────
        # 計算 MACD histogram
        ema_fast = pd.Series(close).ewm(span=macd_fast, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=macd_slow, adjust=False).mean().values
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=macd_signal, adjust=False).mean().values
        histogram = macd_line - signal_line

        macd_shrinking = False
        if len(histogram) >= 4:
            # MACD 柱狀體 > 0（多頭區間）且連續 2 天縮短
            recent_hist = histogram[-3:]
            if recent_hist[-1] > 0 and recent_hist[-2] > 0:
                if recent_hist[-1] < recent_hist[-2] and recent_hist[-2] < recent_hist[-3]:
                    macd_shrinking = True

        # ── 合成衰減因子 ───────────────────────────────────────
        signals = int(rsi_divergence) + int(macd_shrinking)
        if signals == 2:
            decay = -0.06
        elif signals == 1:
            decay = -0.03
        else:
            decay = 0.0

        results.append({"stock_id": sid, "momentum_decay": decay})

    return pd.DataFrame(results)


def compute_institutional_acceleration(
    df_inst: pd.DataFrame,
    stock_ids: list[str],
    window: int = 10,
    recent_days: int = 3,
) -> pd.DataFrame:
    """計算法人買超加速度，獎勵連續且加速的籌碼流入。

    比較近 recent_days 日的平均法人淨買超 vs 前 (window - recent_days) 日的平均值。
    加速比 = recent_avg / older_avg - 1。

    加分規則：
    - 近 recent_days 日平均淨買超 > 0 **且** 加速比 > 0.5 → +0.04
    - 近 recent_days 日平均淨買超 > 0 **且** 0 < 加速比 ≤ 0.5 → +0.02
    - 其餘 → 0.0

    Args:
        df_inst: 三大法人 DataFrame（stock_id / date / name / net）
        stock_ids: 要評估的股票清單
        window: 總回溯天數（預設 10）
        recent_days: 近期天數（預設 3）

    Returns:
        DataFrame(stock_id, inst_accel_bonus)  值域 [0.0, +0.04]
    """
    if df_inst.empty or not stock_ids:
        return pd.DataFrame({"stock_id": stock_ids, "inst_accel_bonus": [0.0] * len(stock_ids)})

    # 按股票+日期彙總法人淨買超
    inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
    if inst_filtered.empty:
        return pd.DataFrame({"stock_id": stock_ids, "inst_accel_bonus": [0.0] * len(stock_ids)})

    daily_net = inst_filtered.groupby(["stock_id", "date"])["net"].sum().reset_index()
    daily_net = daily_net.sort_values(["stock_id", "date"])
    grouped = daily_net.groupby("stock_id", sort=False)

    results: list[dict] = []
    for sid in stock_ids:
        if sid not in grouped.groups:
            results.append({"stock_id": sid, "inst_accel_bonus": 0.0})
            continue

        grp = grouped.get_group(sid)
        if len(grp) < window:
            results.append({"stock_id": sid, "inst_accel_bonus": 0.0})
            continue

        recent = grp.tail(window)
        recent_part = recent.tail(recent_days)["net"]
        older_part = recent.head(window - recent_days)["net"]

        recent_avg = float(recent_part.mean())
        older_avg = float(older_part.mean())

        # 近期淨買超必須為正才可能加分
        if recent_avg <= 0:
            results.append({"stock_id": sid, "inst_accel_bonus": 0.0})
            continue

        # 計算加速度
        if older_avg <= 0:
            # 從賣超轉為買超 = 明顯加速
            accel_ratio = 1.0
        else:
            accel_ratio = recent_avg / older_avg - 1.0

        if accel_ratio > 0.5:
            bonus = 0.04
        elif accel_ratio > 0:
            bonus = 0.02
        else:
            bonus = 0.0

        results.append({"stock_id": sid, "inst_accel_bonus": bonus})

    return pd.DataFrame(results)


def compute_multi_timeframe_alignment(
    daily_trend_bullish: dict[str, bool | None],
    weekly_bonus: dict[str, float],
) -> pd.DataFrame:
    """計算日線/週線多時框一致性分數。

    將日線趨勢（是否站上均線 → bullish/bearish）與週線趨勢（weekly_bonus）結合：
    - 日線 bullish + 週線 bullish (+0.05) → +0.04（一致多頭）
    - 日線 bearish + 週線 bearish (-0.05) → -0.04（一致空頭）
    - 日線 bullish + 週線 bearish → -0.03（時框矛盾，短多長空）
    - 日線 bearish + 週線 bullish → -0.02（時框矛盾，短空長多，但較輕）
    - 任一方無資料 → 0.0

    Args:
        daily_trend_bullish: {stock_id: True/False/None} 日線是否多頭
        weekly_bonus: {stock_id: float} 週線加成值（+0.05=多頭, -0.05=空頭, 0=中性）

    Returns:
        DataFrame(stock_id, mtf_alignment)  值域 [-0.04, +0.04]
    """
    stock_ids = list(daily_trend_bullish.keys())
    results: list[dict] = []

    for sid in stock_ids:
        daily = daily_trend_bullish.get(sid)
        weekly = weekly_bonus.get(sid, 0.0)

        # 任一方無資料 → 中性
        if daily is None or weekly == 0.0:
            results.append({"stock_id": sid, "mtf_alignment": 0.0})
            continue

        weekly_bullish = weekly > 0
        weekly_bearish = weekly < 0

        if daily and weekly_bullish:
            alignment = 0.04  # 日週一致多頭
        elif not daily and weekly_bearish:
            alignment = -0.04  # 日週一致空頭
        elif daily and weekly_bearish:
            alignment = -0.03  # 短多長空矛盾（較危險）
        elif not daily and weekly_bullish:
            alignment = -0.02  # 短空長多矛盾（較輕微）
        else:
            alignment = 0.0

        results.append({"stock_id": sid, "mtf_alignment": alignment})

    return pd.DataFrame(results)


def compute_value_weighted_inst_flow(
    df_inst: pd.DataFrame,
    stock_ids: list[str],
    window: int = 10,
    decay: float = 0.85,
) -> pd.DataFrame:
    """計算法人金額加權連續性分數（Value-Weighted Institutional Flow）。

    替代單純的「連續買超天數」，改用「Σ(net_buy × decay^days_ago)」衰減加權：
    - 大額且持續 > 小額且持續 > 大額一次性
    - decay=0.85 → 10 天前保留 20% 權重

    回傳 percentile rank（0~1），供替代或補充 consec_rank。

    Args:
        df_inst: 三大法人 DataFrame（stock_id / date / name / net）
        stock_ids: 要評估的股票清單
        window: 回溯天數（預設 10）
        decay: 衰減係數（預設 0.85）

    Returns:
        DataFrame(stock_id, inst_flow_weighted)  值域為原始加權值（呼叫端 rank）
    """
    if df_inst.empty or not stock_ids:
        return pd.DataFrame({"stock_id": stock_ids, "inst_flow_weighted": [0.0] * len(stock_ids)})

    # 按 stock_id+date 彙總全部法人淨買超
    inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
    if inst_filtered.empty:
        return pd.DataFrame({"stock_id": stock_ids, "inst_flow_weighted": [0.0] * len(stock_ids)})

    daily_net = inst_filtered.groupby(["stock_id", "date"])["net"].sum().reset_index()
    daily_net = daily_net.sort_values(["stock_id", "date"])
    grouped = daily_net.groupby("stock_id", sort=False)

    results: list[dict] = []
    for sid in stock_ids:
        if sid not in grouped.groups:
            results.append({"stock_id": sid, "inst_flow_weighted": 0.0})
            continue

        grp = grouped.get_group(sid).tail(window)
        if grp.empty:
            results.append({"stock_id": sid, "inst_flow_weighted": 0.0})
            continue

        # 從最近一天 (days_ago=0) 到最早 (days_ago=n-1)
        nets = grp["net"].values[::-1]  # 反轉：index 0 = 最近一天
        weighted_sum = 0.0
        for i, net_val in enumerate(nets):
            weighted_sum += float(net_val) * (decay**i)

        results.append({"stock_id": sid, "inst_flow_weighted": weighted_sum})

    return pd.DataFrame(results)


def compute_earnings_quality(
    df_financial: pd.DataFrame,
    stock_ids: list[str],
) -> pd.DataFrame:
    """計算盈餘品質分數（Earnings Quality Score）。

    三個子指標（等權平均），分數越高品質越好：
    1. 現金流品質：operating_cf / net_income > 1.0 → 高品質（1.0），
       0.5~1.0 → 中等（0.6），< 0.5 或負 → 低品質（0.2）
    2. 應收帳款品質：應收帳款增速 < 營收增速 → 正常（0.8），
       反之 → 可能灌水（0.3）。（需兩季資料，無資料 → 0.5）
    3. 負債穩定性：debt_ratio < 50% → 穩健（0.8），50~70% → 中等（0.5），
       > 70% → 高風險（0.2）

    Args:
        df_financial: FinancialStatement 查詢結果
            （stock_id / date / operating_cf / net_income / revenue / debt_ratio / total_assets）
        stock_ids: 要評估的股票清單

    Returns:
        DataFrame(stock_id, earnings_quality)  值域 [0.0, 1.0]
    """
    default = pd.DataFrame({"stock_id": stock_ids, "earnings_quality": [0.5] * len(stock_ids)})

    if df_financial.empty or not stock_ids:
        return default

    grouped = df_financial.sort_values("date", ascending=False).groupby("stock_id", sort=False)

    results: list[dict] = []
    for sid in stock_ids:
        if sid not in grouped.groups:
            results.append({"stock_id": sid, "earnings_quality": 0.5})
            continue

        grp = grouped.get_group(sid)
        sub_scores: list[float] = []

        # 1. 現金流品質：OCF / Net Income
        latest = grp.iloc[0]
        ocf = latest.get("operating_cf")
        ni = latest.get("net_income")
        if pd.notna(ocf) and pd.notna(ni) and ni != 0:
            ocf_ratio = float(ocf) / float(ni)
            if ocf_ratio > 1.0:
                sub_scores.append(1.0)
            elif ocf_ratio > 0.5:
                sub_scores.append(0.6)
            else:
                sub_scores.append(0.2)
        else:
            sub_scores.append(0.5)

        # 2. 應收帳款品質（需兩季比較營收增速 vs 應收帳款增速）
        # 以 total_assets 作為規模代理（FinancialStatement 無 accounts_receivable）
        # 改用 revenue 增速 vs net_income 增速差異（盈餘灌水代理指標）
        if len(grp) >= 2:
            cur_rev = grp.iloc[0].get("revenue")
            prev_rev = grp.iloc[1].get("revenue")
            cur_ni = grp.iloc[0].get("net_income")
            prev_ni = grp.iloc[1].get("net_income")
            if (
                pd.notna(cur_rev)
                and pd.notna(prev_rev)
                and pd.notna(cur_ni)
                and pd.notna(prev_ni)
                and prev_rev != 0
                and prev_ni != 0
            ):
                rev_growth = (float(cur_rev) - float(prev_rev)) / abs(float(prev_rev))
                ni_growth = (float(cur_ni) - float(prev_ni)) / abs(float(prev_ni))
                # 淨利增速遠超營收增速 → 可能灌水
                if ni_growth > rev_growth + 0.2:
                    sub_scores.append(0.3)
                else:
                    sub_scores.append(0.8)
            else:
                sub_scores.append(0.5)
        else:
            sub_scores.append(0.5)

        # 3. 負債穩定性
        debt = latest.get("debt_ratio")
        if pd.notna(debt):
            d = float(debt)
            if d < 50:
                sub_scores.append(0.8)
            elif d < 70:
                sub_scores.append(0.5)
            else:
                sub_scores.append(0.2)
        else:
            sub_scores.append(0.5)

        quality = float(np.mean(sub_scores))
        results.append({"stock_id": sid, "earnings_quality": quality})

    return pd.DataFrame(results)


# 回撤降頻常數：TAIEX 回撤閾值 → 推薦數量調整
DRAWDOWN_FREQUENCY_RULES: dict[str, dict] = {
    "severe": {"threshold": -0.15, "allowed_modes": {"value", "dividend"}},
    "moderate": {"threshold": -0.10, "multiplier": 0.5},
}


# ── P2-B3: 籌碼面 MACD ──────────────────────────────────────────────


def compute_chip_macd(
    df_inst: pd.DataFrame,
    stock_ids: list[str],
    fast_span: int = 5,
    slow_span: int = 20,
    signal_span: int = 5,
) -> pd.DataFrame:
    """計算法人淨買超的 MACD 指標（短期 vs 長期 EMA 交叉）。

    以每日法人淨買超（三大法人合計）作為信號源，計算：
    - fast_ema: 短期（5日）EMA
    - slow_ema: 長期（20日）EMA
    - chip_macd_line: fast_ema - slow_ema（正值 = 吸籌加速）
    - chip_macd_signal: chip_macd_line 的 signal_span EMA
    - chip_macd_hist: chip_macd_line - signal_span（柱狀圖，正值擴大 = 加速吸籌）

    評分規則：
    - chip_macd_hist > 0 且遞增（最近值 > 前一日）→ 強勢吸籌 = 1.0
    - chip_macd_hist > 0                        → 溫和吸籌 = 0.7
    - chip_macd_hist ≤ 0 但 MACD line > 0        → 減速但仍正向 = 0.4
    - chip_macd_hist ≤ 0 且 MACD line ≤ 0        → 出貨信號 = 0.1

    Args:
        df_inst: 三大法人 DataFrame（stock_id / date / name / net）
        stock_ids: 要評估的股票清單
        fast_span: 短期 EMA 窗口（預設 5）
        slow_span: 長期 EMA 窗口（預設 20）
        signal_span: 信號線 EMA 窗口（預設 5）

    Returns:
        DataFrame(stock_id, chip_macd_score)  值域 [0.0, 1.0]
    """
    if df_inst.empty or not stock_ids:
        return pd.DataFrame({"stock_id": stock_ids, "chip_macd_score": [0.5] * len(stock_ids)})

    # 按股票+日期彙總法人淨買超
    inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
    if inst_filtered.empty:
        return pd.DataFrame({"stock_id": stock_ids, "chip_macd_score": [0.5] * len(stock_ids)})

    daily_net = inst_filtered.groupby(["stock_id", "date"])["net"].sum().reset_index()
    daily_net = daily_net.sort_values(["stock_id", "date"])

    results: list[dict] = []
    grouped = daily_net.groupby("stock_id", sort=False)

    for sid in stock_ids:
        if sid not in grouped.groups:
            results.append({"stock_id": sid, "chip_macd_score": 0.5})
            continue

        grp = grouped.get_group(sid).sort_values("date")
        nets = grp["net"].values.astype(float)

        # 至少需 slow_span 天資料才能計算有意義的 EMA
        if len(nets) < slow_span:
            results.append({"stock_id": sid, "chip_macd_score": 0.5})
            continue

        # 計算 EMA
        fast_ema = pd.Series(nets).ewm(span=fast_span, adjust=False).mean()
        slow_ema = pd.Series(nets).ewm(span=slow_span, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
        hist = macd_line - signal_line

        # 取最近值
        cur_hist = float(hist.iloc[-1])
        prev_hist = float(hist.iloc[-2]) if len(hist) >= 2 else 0.0
        cur_macd = float(macd_line.iloc[-1])

        # 避免極小浮點誤差：接近零視為中性
        _eps = 1e-6
        macd_pos = cur_macd > _eps
        macd_neg = cur_macd < -_eps
        hist_pos = cur_hist > _eps
        hist_rising = cur_hist > prev_hist + _eps

        if macd_pos and hist_pos and hist_rising:
            score = 1.0  # 強勢吸籌：MACD 正 + 柱狀圖正且遞增
        elif macd_pos and hist_pos:
            score = 0.7  # 溫和吸籌：MACD 正 + 柱狀圖正但遞減
        elif macd_pos:
            score = 0.5  # 減速吸籌：MACD 仍正但柱狀圖翻負
        elif not macd_pos and not macd_neg:
            score = 0.5  # 中性：MACD 接近零（穩定買入或無趨勢）
        elif macd_neg and hist_pos:
            score = 0.3  # 出貨減速：MACD 負但柱狀圖回升（跌勢趨緩）
        else:
            score = 0.1  # 出貨信號：MACD 負且柱狀圖負或下降

        results.append({"stock_id": sid, "chip_macd_score": score})

    return pd.DataFrame(results)


# ── P2-E1: 勝率回饋循環 ─────────────────────────────────────────────

# 勝率回饋門檻調整：近期勝率低 → 提高 min_composite_score
WIN_RATE_FEEDBACK_CONFIG: dict[str, float] = {
    "low_threshold": 0.40,  # 勝率低於此值 → 提高門檻
    "moderate_threshold": 0.50,  # 勝率低於此值 → 輕微提高
    "low_penalty": 0.05,  # 勝率 < 40% → 門檻 +0.05
    "moderate_penalty": 0.02,  # 勝率 40~50% → 門檻 +0.02
    "holding_days": 5,  # 以 5 天報酬計算勝率
    "lookback_days": 30,  # 回溯 30 天的推薦記錄
}


def compute_win_rate_threshold_adjustment(
    df_records: pd.DataFrame,
    df_prices: pd.DataFrame,
    mode: str,
    holding_days: int = 5,
    lookback_days: int = 30,
    reference_date: date | None = None,
) -> float:
    """根據近期推薦勝率計算門檻調整值。

    讀取指定模式過去 lookback_days 天的 DiscoveryRecord，
    計算 holding_days 天報酬率的勝率，
    勝率低則回傳正值（提高門檻），否則回傳 0。

    Args:
        df_records: 推薦記錄 DataFrame
            （scan_date / stock_id / close / composite_score / mode）
        df_prices: 日K線 DataFrame（stock_id / date / close）
        mode: 掃描模式（momentum / swing / value / dividend / growth）
        holding_days: 評估報酬天數（預設 5）
        lookback_days: 回溯推薦記錄天數（預設 30）
        reference_date: 基準日期（預設 today）

    Returns:
        門檻調整值（≥ 0.0）
    """
    if df_records.empty or df_prices.empty:
        return 0.0

    ref_date = reference_date or date.today()
    cutoff = ref_date - timedelta(days=lookback_days)

    # 過濾指定模式 + 時間範圍
    mask = df_records["scan_date"].apply(lambda d: d >= cutoff if isinstance(d, date) else False)
    recent = df_records[mask]
    if recent.empty:
        return 0.0

    # 計算每筆推薦的 N 天報酬
    returns = []
    for _, rec in recent.iterrows():
        scan_d = rec["scan_date"]
        sid = rec["stock_id"]
        entry_close = float(rec["close"])
        if entry_close <= 0:
            continue

        future = df_prices[(df_prices["stock_id"] == sid) & (df_prices["date"] > scan_d)].sort_values("date")

        if len(future) >= holding_days:
            exit_close = float(future.iloc[holding_days - 1]["close"])
            returns.append((exit_close - entry_close) / entry_close)

    if len(returns) < 5:  # 太少樣本，不做調整
        return 0.0

    win_rate = sum(1 for r in returns if r > 0) / len(returns)

    cfg = WIN_RATE_FEEDBACK_CONFIG
    if win_rate < cfg["low_threshold"]:
        return cfg["low_penalty"]
    elif win_rate < cfg["moderate_threshold"]:
        return cfg["moderate_penalty"]
    else:
        return 0.0


# ── P2-E2: 因子有效性監控 ────────────────────────────────────────────

# 因子名稱映射（DiscoveryRecord 欄位 → 中文標籤）
FACTOR_COLUMNS: list[str] = [
    "technical_score",
    "chip_score",
    "fundamental_score",
    "news_score",
]


_PER_DATE_MIN_STOCKS = 3  # 每日 cross-section 至少 N 檔才計算當日 IC（預設）
_PER_DATE_MIN_DATES = 3  # 至少 N 個有效日期才用 per-date 平均；否則 fallback pooled


def compute_factor_ic(
    df_records: pd.DataFrame,
    df_prices: pd.DataFrame,
    holding_days: int = 5,
    lookback_days: int = 30,
    reference_date: date | None = None,
    min_per_date_count: int | None = None,
) -> pd.DataFrame:
    """計算各評分因子與後續報酬率的 IC（Information Coefficient）。

    IC 定義（標準 cross-sectional）：
        IC = mean_t( spearman_corr(factor_t, ret_{t→t+h}) )
    即每日先做 cross-section corr，再對日期平均。

    C4 修復（2026-05-09 audit）：原本對全樣本（跨日 pool）做 spearman corr，
    當 factor 跨日均值漂移時（如 momentum 在 bull/bear 不同 mean）會把 time-series
    訊號混進 IC，造成 momentum IC 高估 0.05~0.10。改為標準 per-date 平均。

    Backward compat：當 per-date 資料稀疏（每日 < 3 檔 或 有效日期 < 3 天）時，
    fallback 至 pooled corr（保留舊行為，主要支援既有合成測試資料）。

    高 IC（> 0.05）= 因子有效；低 IC（< -0.05）= 因子反向。

    Args:
        df_records: 推薦記錄 DataFrame
            （scan_date / stock_id / close / technical_score /
              chip_score / fundamental_score / news_score）
        df_prices: 日K線 DataFrame（stock_id / date / close）
        holding_days: 報酬天數（預設 5）
        lookback_days: 回溯天數（預設 30）
        reference_date: 基準日期（預設 today）
        min_per_date_count: 每日 cross-section 最少股票數，少於此值該日 IC 不計入
            （預設 None → 套用模組常數 _PER_DATE_MIN_STOCKS=3）。S2 audit fix：
            可由 caller 調整（如測試或低樣本場景），保持向後相容。

    Returns:
        DataFrame(factor, ic, evaluable_count, direction)
        direction: "effective" (IC > 0.05) / "weak" (|IC| ≤ 0.05) / "inverse" (IC < -0.05)
    """
    if df_records.empty or df_prices.empty:
        return pd.DataFrame(columns=["factor", "ic", "evaluable_count", "direction"])

    min_per_date = min_per_date_count if min_per_date_count is not None else _PER_DATE_MIN_STOCKS
    ref_date = reference_date or date.today()
    cutoff = ref_date - timedelta(days=lookback_days)

    mask = df_records["scan_date"].apply(lambda d: d >= cutoff if isinstance(d, date) else False)
    recent = df_records[mask].copy()
    if recent.empty:
        return pd.DataFrame(columns=["factor", "ic", "evaluable_count", "direction"])

    # 計算每筆推薦的 N 天報酬
    ret_map: dict[tuple, float] = {}
    for _, rec in recent.iterrows():
        scan_d = rec["scan_date"]
        sid = rec["stock_id"]
        entry_close = float(rec["close"])
        if entry_close <= 0:
            continue

        future = df_prices[(df_prices["stock_id"] == sid) & (df_prices["date"] > scan_d)].sort_values("date")

        if len(future) >= holding_days:
            exit_close = float(future.iloc[holding_days - 1]["close"])
            ret_map[(scan_d, sid)] = (exit_close - entry_close) / entry_close

    if len(ret_map) < 10:  # 太少樣本，無統計意義
        return pd.DataFrame(columns=["factor", "ic", "evaluable_count", "direction"])

    # 建立含報酬的 DataFrame
    recent = recent.copy()
    recent["forward_return"] = recent.apply(
        lambda r: ret_map.get((r["scan_date"], r["stock_id"]), None),
        axis=1,
    )
    recent = recent.dropna(subset=["forward_return"])

    results: list[dict] = []
    for factor in FACTOR_COLUMNS:
        if factor not in recent.columns:
            continue

        # valid 含 scan_date + factor + forward_return，後續 groupby 用
        valid = recent[["scan_date", factor, "forward_return"]].dropna()
        if len(valid) < 10:
            continue

        # ── C4 主路徑：per-date cross-sectional IC ─────────────
        per_date_ics: list[float] = []
        used_samples = 0
        for _scan_d, grp in valid.groupby("scan_date"):
            if len(grp) < min_per_date:
                continue
            if grp[factor].std() == 0 or grp["forward_return"].std() == 0:
                continue
            ic_d = grp[factor].corr(grp["forward_return"], method="spearman")
            if not np.isnan(ic_d):
                per_date_ics.append(float(ic_d))
                used_samples += len(grp)

        if len(per_date_ics) >= _PER_DATE_MIN_DATES:
            ic = float(np.mean(per_date_ics))
            evaluable = used_samples
        else:
            # ── Fallback：per-date 資料稀疏時用 pooled corr（保留舊語意） ──
            if valid[factor].std() == 0 or valid["forward_return"].std() == 0:
                continue
            ic = float(valid[factor].corr(valid["forward_return"], method="spearman"))
            if np.isnan(ic):
                ic = 0.0
            evaluable = len(valid)

        if ic > 0.05:
            direction = "effective"
        elif ic < -0.05:
            direction = "inverse"
        else:
            direction = "weak"

        results.append(
            {
                "factor": factor,
                "ic": round(ic, 4),
                "evaluable_count": evaluable,
                "direction": direction,
            }
        )

    return pd.DataFrame(results)


def compute_rolling_ic(
    df_records: pd.DataFrame,
    df_prices: pd.DataFrame,
    holding_days: int = 5,
    window_days: int = 14,
    step_days: int = 7,
) -> pd.DataFrame:
    """計算滾動 IC 時間序列，觀察因子有效性的時間趨勢。

    以 step_days 為步進、window_days 為窗口，逐段計算各因子 IC。
    每個窗口使用 compute_factor_ic() 邏輯。

    Args:
        df_records: 歷史推薦記錄（scan_date / stock_id / close / 4 scores）
        df_prices: 日K線（stock_id / date / close）
        holding_days: 報酬天數（預設 5）
        window_days: 滾動窗口天數（預設 14）
        step_days: 步進天數（預設 7）

    Returns:
        DataFrame(window_end, factor, ic, sample_count, direction)
        按 window_end 排序，可直接繪製時間序列
    """
    if df_records.empty or df_prices.empty:
        return pd.DataFrame(columns=["window_end", "factor", "ic", "sample_count", "direction"])

    # 取得所有 scan_date 的時間範圍
    scan_dates = sorted(df_records["scan_date"].apply(lambda d: d if isinstance(d, date) else None).dropna().unique())
    if not scan_dates:
        return pd.DataFrame(columns=["window_end", "factor", "ic", "sample_count", "direction"])

    min_date = scan_dates[0]
    max_date = scan_dates[-1]

    results: list[dict] = []
    # 從 min_date + window_days 開始，每 step_days 計算一個窗口
    cursor = min_date + timedelta(days=window_days)
    while cursor <= max_date + timedelta(days=1):
        ic_df = compute_factor_ic(
            df_records,
            df_prices,
            holding_days=holding_days,
            lookback_days=window_days,
            reference_date=cursor,
        )
        for _, row in ic_df.iterrows():
            results.append(
                {
                    "window_end": cursor,
                    "factor": row["factor"],
                    "ic": row["ic"],
                    "sample_count": int(row["evaluable_count"]),
                    "direction": row["direction"],
                }
            )
        cursor += timedelta(days=step_days)

    return pd.DataFrame(results)


def compute_regime_ic(
    df_records: pd.DataFrame,
    df_prices: pd.DataFrame,
    df_taiex: pd.DataFrame,
    holding_days: int = 5,
) -> pd.DataFrame:
    """按 Regime 分組計算各因子 IC。

    使用 TAIEX 收盤價透過 detect_from_series() 判定每筆推薦所在的 Regime，
    再分組計算 Spearman IC。

    Args:
        df_records: 歷史推薦記錄
        df_prices: 日K線
        df_taiex: TAIEX 日K線（date / close 欄位，按 date 排序）
        holding_days: 報酬天數

    Returns:
        DataFrame(regime, factor, ic, sample_count, direction)
    """
    from src.regime.detector import detect_from_series

    if df_records.empty or df_prices.empty or df_taiex.empty:
        return pd.DataFrame(columns=["regime", "factor", "ic", "sample_count", "direction"])

    # 建立 date → regime 映射
    taiex_sorted = df_taiex.sort_values("date")
    taiex_closes = taiex_sorted["close"].reset_index(drop=True)
    taiex_dates = taiex_sorted["date"].tolist()

    date_regime: dict[date, str] = {}
    for i, d in enumerate(taiex_dates):
        if i < 120:  # 需要至少 sma_long=120 天才能判定
            date_regime[d] = "unknown"
        else:
            partial = taiex_closes.iloc[: i + 1]
            result = detect_from_series(partial, include_crisis=False)
            date_regime[d] = result["regime"]

    # 為每筆推薦附上 regime
    records = df_records.copy()
    records["regime"] = records["scan_date"].apply(
        lambda d: date_regime.get(d, "unknown") if isinstance(d, date) else "unknown"
    )
    records = records[records["regime"] != "unknown"]

    if records.empty:
        return pd.DataFrame(columns=["regime", "factor", "ic", "sample_count", "direction"])

    # 分組計算 IC
    results: list[dict] = []
    for regime, group in records.groupby("regime"):
        if len(group) < 10:
            continue

        # 用整組推薦做 IC（不限 lookback，因為已按 regime 分組）
        ic_df = compute_factor_ic(
            group,
            df_prices,
            holding_days=holding_days,
            lookback_days=9999,  # 不限回溯，因為已按 regime 過濾
        )
        for _, row in ic_df.iterrows():
            results.append(
                {
                    "regime": regime,
                    "factor": row["factor"],
                    "ic": row["ic"],
                    "sample_count": int(row["evaluable_count"]),
                    "direction": row["direction"],
                }
            )

    return pd.DataFrame(results)


def compute_sub_factor_ic(
    sub_factor_df: pd.DataFrame,
    df_prices: pd.DataFrame,
    holding_days: int = 5,
    lookback_days: int = 30,
    reference_date: date | None = None,
) -> pd.DataFrame:
    """計算子因子與後續報酬率的 IC（Spearman Rank Correlation）。

    與 compute_factor_ic 類似，��接受任意欄位名的子因子 DataFrame。
    要求 sub_factor_df 含 scan_date / stock_id / close 欄位，
    其餘所有數值欄位視為子因子。

    Returns:
        DataFrame(factor, ic, evaluable_count, direction)
    """
    if sub_factor_df.empty or df_prices.empty:
        return pd.DataFrame(columns=["factor", "ic", "evaluable_count", "direction"])

    ref_date = reference_date or date.today()
    cutoff = ref_date - timedelta(days=lookback_days)

    mask = sub_factor_df["scan_date"].apply(lambda d: d >= cutoff if isinstance(d, date) else False)
    recent = sub_factor_df[mask].copy()
    if recent.empty:
        return pd.DataFrame(columns=["factor", "ic", "evaluable_count", "direction"])

    # 計算前瞻報酬
    ret_map: dict[tuple, float] = {}
    for _, rec in recent.iterrows():
        scan_d = rec["scan_date"]
        sid = rec["stock_id"]
        entry_close = float(rec["close"])
        if entry_close <= 0:
            continue
        future = df_prices[(df_prices["stock_id"] == sid) & (df_prices["date"] > scan_d)].sort_values("date")
        if len(future) >= holding_days:
            exit_close = float(future.iloc[holding_days - 1]["close"])
            ret_map[(scan_d, sid)] = (exit_close - entry_close) / entry_close

    if len(ret_map) < 10:
        return pd.DataFrame(columns=["factor", "ic", "evaluable_count", "direction"])

    recent["forward_return"] = recent.apply(lambda r: ret_map.get((r["scan_date"], r["stock_id"]), None), axis=1)
    recent = recent.dropna(subset=["forward_return"])

    # 找出所有子因子欄位（排除 metadata 欄位）
    meta_cols = {"scan_date", "stock_id", "close", "forward_return", "mode"}
    factor_cols = [
        c for c in recent.columns if c not in meta_cols and recent[c].dtype in ("float64", "float32", "int64")
    ]

    results: list[dict] = []
    for factor in factor_cols:
        valid = recent[[factor, "forward_return"]].dropna()
        if len(valid) < 10:
            continue
        # 常數輸入（std == 0）→ 相關係數未定義，跳過
        if valid[factor].std() == 0 or valid["forward_return"].std() == 0:
            continue
        ic = float(valid[factor].corr(valid["forward_return"], method="spearman"))
        if np.isnan(ic):
            ic = 0.0
        direction = "effective" if ic > 0.05 else ("inverse" if ic < -0.05 else "weak")
        results.append({"factor": factor, "ic": round(ic, 4), "evaluable_count": len(valid), "direction": direction})

    return pd.DataFrame(results)


def compute_factor_correlation_matrix(
    sub_factor_df: pd.DataFrame,
) -> pd.DataFrame:
    """計算子因子間的 Spearman 相關性矩陣。

    Args:
        sub_factor_df: 含子因子 rank 欄位的 DataFrame（stock_id + 因子欄位）

    Returns:
        相關性矩陣 DataFrame（index/columns 皆為因子名稱）
    """
    meta_cols = {"scan_date", "stock_id", "close", "mode", "forward_return"}
    factor_cols = [
        c
        for c in sub_factor_df.columns
        if c not in meta_cols and sub_factor_df[c].dtype in ("float64", "float32", "int64")
    ]
    # 排除常數欄位（std == 0），避免 ConstantInputWarning
    factor_cols = [c for c in factor_cols if sub_factor_df[c].std() > 0]
    if len(factor_cols) < 2:
        return pd.DataFrame()
    return sub_factor_df[factor_cols].corr(method="spearman")


def compute_ic_weight_adjustments(
    ic_df: pd.DataFrame,
    base_weights: dict[str, float],
    dampen_factor: float = 0.5,
) -> dict[str, float]:
    """根據 IC 結果調整因子權重。

    規則：
    - IC "effective" → 維持原權重
    - IC "weak" → 權重 × dampen_factor（降半）
    - IC "inverse" → 權重 × dampen_factor²（大幅降權）
    - 調整後重新歸一化至原始權重總和

    Args:
        ic_df: compute_factor_ic 的輸出
        base_weights: 原始權重字典（如 {"technical_score": 0.45, ...}）
        dampen_factor: 衰減因子（預設 0.5）

    Returns:
        調整後權重字典（總和等於原始 base_weights 總和）
    """
    if ic_df.empty:
        return dict(base_weights)

    adjusted = dict(base_weights)
    ic_map = dict(zip(ic_df["factor"], ic_df["direction"]))

    for factor, weight in adjusted.items():
        direction = ic_map.get(factor, "effective")
        if direction == "weak":
            adjusted[factor] = weight * dampen_factor
        elif direction == "inverse":
            adjusted[factor] = weight * (dampen_factor**2)
        # "effective" → 維持

    # 歸一化：保持原始權重總和不變
    original_total = sum(base_weights.values())
    adjusted_total = sum(adjusted.values())
    if adjusted_total > 0:
        scale = original_total / adjusted_total
        adjusted = {k: v * scale for k, v in adjusted.items()}

    return adjusted


def compute_ic_impact_weight_adjustments(
    ic_df: pd.DataFrame,
    impact_df: pd.DataFrame,
    base_weights: dict[str, float],
    dampen_factor: float = 0.5,
    impact_alpha: float = 0.2,
    impact_clip: tuple[float, float] = (0.6, 1.4),
    min_impact_samples: int = 3,
) -> dict[str, float]:
    """IC × 影響力雙指標軟調整（Phase C）。

    IC 檢驗「方向正確性」，影響力（1 - ρ）檢驗「鑑別力」。
    兩者獨立作用後歸一化，避免單一訊號主導。

    規則：
      1. IC 離散衰減（sharp）— sharp multiplier: effective=1.0 / weak=0.5 / inverse=0.25
      2. 影響力軟調整（soft）— mult = 1 + alpha × (impact - mean_impact)，clip [0.6, 1.4]
      3. 歸一化至原始 base_weights 總和

    影響力不穩定性防護：
      - impact_df 樣本 < min_impact_samples → 跳過影響力調整
      - clip 範圍防止極端 impact 爆炸變異
      - alpha 預設 0.2（保守），避免 IC × ρ 雙重放大 noise

    例：某維度 IC 方向正確（effective）且 impact 顯著高於平均 → 權重略增
         某維度 IC=inverse（方向錯）+ impact 高 → IC 端已大幅降權，impact 不再放大降權

    Args:
        ic_df: compute_factor_ic 輸出（含 factor, direction 欄位）；factor 名稱需與 base_weights key 對齊
        impact_df: 含 "factor", "rank_correlation" 欄位；factor 名稱需與 base_weights key 對齊
        base_weights: 原始權重字典
        dampen_factor: IC weak/inverse 衰減因子（沿用 compute_ic_weight_adjustments）
        impact_alpha: 影響力調整強度，0.0 = 關閉
        impact_clip: 影響力倍率夾擠範圍
        min_impact_samples: 啟用影響力調整的最低樣本數

    Returns:
        調整後權重字典（總和等於原始 base_weights 總和）
    """
    if not base_weights:
        return dict(base_weights)
    if (ic_df is None or ic_df.empty) and (impact_df is None or impact_df.empty):
        return dict(base_weights)

    adjusted = dict(base_weights)

    # Step 1: IC 方向離散衰減
    if ic_df is not None and not ic_df.empty:
        ic_map = dict(zip(ic_df["factor"], ic_df["direction"]))
        for factor in adjusted:
            direction = ic_map.get(factor, "effective")
            if direction == "weak":
                adjusted[factor] *= dampen_factor
            elif direction == "inverse":
                adjusted[factor] *= dampen_factor**2

    # Step 2: 影響力軟調整（樣本充足才啟用）
    if impact_df is not None and not impact_df.empty and impact_alpha > 0 and len(impact_df) >= min_impact_samples:
        impact_map: dict[str, float] = {}
        for _, row in impact_df.iterrows():
            factor = row.get("factor") if hasattr(row, "get") else None
            rho = row.get("rank_correlation") if hasattr(row, "get") else None
            if factor is None or rho is None:
                continue
            try:
                rho_val = float(rho)
            except (TypeError, ValueError):
                continue
            if np.isnan(rho_val):
                continue
            # impact = 1 - ρ（ρ 越低表示移除後排名變動越大 → 影響力越高）
            impact_map[str(factor)] = 1.0 - rho_val

        if impact_map:
            impacts = np.array(list(impact_map.values()))
            mean_impact = float(np.mean(impacts))
            lo, hi = impact_clip
            for factor in adjusted:
                impact = impact_map.get(factor)
                if impact is None:
                    continue
                mult = 1.0 + impact_alpha * (impact - mean_impact)
                mult = max(lo, min(hi, mult))
                adjusted[factor] *= mult

    # Step 3: 歸一化至原始總和
    original_total = sum(base_weights.values())
    adjusted_total = sum(adjusted.values())
    if adjusted_total > 0:
        scale = original_total / adjusted_total
        adjusted = {k: v * scale for k, v in adjusted.items()}

    return adjusted


# IC dampen 模式下，|IC| < ic_threshold_weak 時的權重衰減倍率
# 設計理由：保留分數排序資訊但壓抑該維度對 composite 的貢獻，避免直接歸 0.5 造成
# 該維度權重全送給常數值（top 名單失去區分度）
IC_DAMPEN_WEIGHT_MULT: float = 0.25


def compute_ic_aware_score_transform(
    candidates: pd.DataFrame,
    ic_df: pd.DataFrame,
    ic_threshold_weak: float = 0.02,
    min_samples: int = 50,
    dampen_mode: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """IC 感知維度分數轉換 — 根據歷史 IC 修正方向性錯誤的因子。

    與 compute_ic_weight_adjustments()（僅衰減權重）不同，本函數直接變換 **分數值**：
      - IC > +ic_threshold_weak  → 原樣保留（正向有效）
      - IC < -ic_threshold_weak  → score → 1 - score（反向訊號翻轉為正向使用）
      - |IC| ≤ ic_threshold_weak（雜訊區間）：
          * dampen_mode=False（預設）→ score → 0.5（中性化，向後相容）
          * dampen_mode=True          → score 保留原值（action="dampen"，由
            caller 在 composite 加權階段以 IC_DAMPEN_WEIGHT_MULT 倍率衰減）
      - evaluable_count < min_samples → 原樣保留（資料不足不翻）

    設計理由：
      原系統對 inverse IC 採權重 × 0.25 的「衰減」處理，但分數方向仍錯誤，
      仍會以較小比例將反向訊號累加進 composite，造成 α 流失。
      直接翻轉能保留訊息內容同時修正方向，IC 絕對值越大翻轉後貢獻越強。

    安全設計：
      - ic_df 為空時原樣回傳（cold-start 保護）
      - 只處理 ic_df["factor"] 中且確實存在於 candidates 欄位的分數
      - 轉換後的分數仍在 [0, 1] 區間（clip 防護）
      - 回傳 mapping 記錄每個 factor 的動作（"kept"/"flipped"/"neutralized"），供稽核與日誌

    Args:
        candidates: 候選 DataFrame，含 *_score 欄位（例 technical_score, news_score）
        ic_df: compute_factor_ic 輸出，需含 factor、ic、evaluable_count 欄位
        ic_threshold_weak: IC 絕對值小於此值視為雜訊（預設 0.02）
        min_samples: 最低有效樣本（低於則不翻轉，預設 50）

    Returns:
        (轉換後 DataFrame, {factor_name: action} mapping)
    """
    if candidates is None or candidates.empty:
        return candidates, {}
    actions: dict[str, str] = {}
    if ic_df is None or ic_df.empty:
        return candidates, actions

    out = candidates.copy()
    for _, row in ic_df.iterrows():
        factor = row.get("factor")
        if not isinstance(factor, str) or factor not in out.columns:
            continue
        try:
            ic_val = float(row.get("ic", 0.0))
        except (TypeError, ValueError):
            continue
        if np.isnan(ic_val):
            continue
        evaluable = int(row.get("evaluable_count", 0) or 0)
        if evaluable < min_samples:
            actions[factor] = "kept_low_samples"
            continue

        if ic_val > ic_threshold_weak:
            actions[factor] = "kept"
            continue
        if ic_val < -ic_threshold_weak:
            out[factor] = (1.0 - out[factor]).clip(lower=0.0, upper=1.0)
            actions[factor] = "flipped"
            continue
        # 雜訊區間：dampen 模式保留分數讓 caller 衰減權重，否則歸 0.5 中性化
        if dampen_mode:
            actions[factor] = "dampen"
        else:
            out[factor] = 0.5
            actions[factor] = "neutralized"

    return out, actions


def compute_sub_factor_weight_adjustments(
    sub_factor_ic_df: pd.DataFrame,
    base_weights: dict[str, float],
    dampen_factor: float = 0.5,
    min_samples: int = 20,
) -> dict[str, float]:
    """根據子因子 IC 結果調整子因子權重（chip 層級）。

    與 compute_ic_weight_adjustments() 同邏輯，額外新增 min_samples 防護：
    evaluable_count < min_samples 的因子視為 "effective"（不調整）。

    Args:
        sub_factor_ic_df: 含 (factor, ic, evaluable_count, direction) 的 DataFrame
        base_weights: 原始子因子權重字典（如 {"consec": 0.16, "bvr": 0.14, ...}）
        dampen_factor: 衰減因子（預設 0.5）
        min_samples: 最低有效樣本數（預設 20）

    Returns:
        調整後權重字典（總和等於原始 base_weights 總和）
    """
    if sub_factor_ic_df.empty:
        return dict(base_weights)

    adjusted = dict(base_weights)

    # 建立 factor → direction 映射，但過濾掉樣本不足的因子
    for _, row in sub_factor_ic_df.iterrows():
        factor = row["factor"]
        if factor not in adjusted:
            continue
        evaluable = row.get("evaluable_count", 0)
        if evaluable < min_samples:
            continue  # 樣本不足 → 維持原權重
        direction = row.get("direction", "effective")
        if direction == "weak":
            adjusted[factor] = adjusted[factor] * dampen_factor
        elif direction == "inverse":
            adjusted[factor] = adjusted[factor] * (dampen_factor**2)
        # "effective" → 維持

    # 歸一化：保持原始權重總和不變
    original_total = sum(base_weights.values())
    adjusted_total = sum(adjusted.values())
    if adjusted_total > 0:
        scale = original_total / adjusted_total
        adjusted = {k: v * scale for k, v in adjusted.items()}

    return adjusted


def exclude_zero_variance_factors(
    rank_map: dict[str, "pd.Series"],  # noqa: F821
    weights: dict[str, float],
    eps: float = 1e-9,
) -> tuple[dict[str, "pd.Series"], dict[str, float]]:  # noqa: F821
    """偵測零方差因子（rank 全部相同），排除並重新分配權重。

    當某子因子 rank 標準差 < eps（所有候選股得到相同排名），
    該因子無鑑別力，自動排除並將其權重按比例分配給剩餘因子。

    Args:
        rank_map: 因子名稱 → rank Series 映射
        weights: 因子名稱 → 權重映射（key 必須為 rank_map 的子集）
        eps: 零方差閾值（預設 1e-9）

    Returns:
        (filtered_rank_map, adjusted_weights) — 排除零方差因子後的映射與權重
    """
    if not rank_map or not weights:
        return rank_map, weights

    zero_var_keys = set()
    for key, series in rank_map.items():
        if key not in weights:
            continue
        if series.std() < eps:
            zero_var_keys.add(key)

    if not zero_var_keys:
        return rank_map, weights

    # 排除零方差因子
    filtered = {k: v for k, v in rank_map.items() if k not in zero_var_keys}
    remaining = {k: v for k, v in weights.items() if k not in zero_var_keys}

    # 重新歸一化權重至原始總和
    original_total = sum(weights.values())
    remaining_total = sum(remaining.values())
    if remaining_total > 0:
        scale = original_total / remaining_total
        remaining = {k: v * scale for k, v in remaining.items()}

    return filtered, remaining


# ── P3-B2: 主力成本分析 ──────────────────────────────────────────────


def compute_key_player_cost_basis(
    df_broker: pd.DataFrame,
    stock_ids: list[str],
    top_n_brokers: int = 3,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """計算 Top-N 主力估計成本與現價的關係。

    從分點歷史資料計算 top_n_brokers 個最大淨買超分點的加權平均成本，
    與現價比較判斷主力是被套（護盤動力）還是獲利豐厚（出貨風險）。

    評分規則：
    - 現價 < 主力成本 × 0.95 → 主力被套，有護盤動力 = 0.8
    - 現價 介於 0.95~1.10 × 主力成本 → 成本附近 = 0.5
    - 現價 > 主力成本 × 1.10 → 主力已獲利，出貨風險 = 0.2
    - 無法計算 → 中性 0.5

    Args:
        df_broker: BrokerTrade DataFrame（stock_id / date / broker_id /
                   buy / sell / buy_price / sell_price）
        stock_ids: 要評估的股票清單
        top_n_brokers: 取淨買超前 N 大主力（預設 3）
        lookback_days: 回溯天數（預設 60）

    Returns:
        DataFrame(stock_id, key_player_cost, key_player_score)
    """
    default = pd.DataFrame(
        {
            "stock_id": stock_ids,
            "key_player_cost": [None] * len(stock_ids),
            "key_player_score": [0.5] * len(stock_ids),
        }
    )

    if df_broker.empty or not stock_ids:
        return default

    cutoff = None
    if "date" in df_broker.columns and not df_broker.empty:
        max_date = df_broker["date"].max()
        if hasattr(max_date, "toordinal"):
            cutoff = max_date - timedelta(days=lookback_days)

    results: list[dict] = []
    grouped = df_broker[df_broker["stock_id"].isin(stock_ids)].groupby("stock_id", sort=False)

    for sid in stock_ids:
        if sid not in grouped.groups:
            results.append({"stock_id": sid, "key_player_cost": None, "key_player_score": 0.5})
            continue

        grp = grouped.get_group(sid)
        if cutoff is not None:
            grp = grp[grp["date"] >= cutoff]
        if grp.empty:
            results.append({"stock_id": sid, "key_player_cost": None, "key_player_score": 0.5})
            continue

        # 彙總每個分點的淨買超量和加權成本
        broker_agg = (
            grp.groupby("broker_id")
            .agg(
                total_buy=("buy", "sum"),
                total_sell=("sell", "sum"),
            )
            .reset_index()
        )
        broker_agg["net_buy"] = broker_agg["total_buy"] - broker_agg["total_sell"]

        # 取淨買超前 N 大的分點
        top_brokers = broker_agg.nlargest(top_n_brokers, "net_buy")
        top_brokers = top_brokers[top_brokers["net_buy"] > 0]

        if top_brokers.empty:
            results.append({"stock_id": sid, "key_player_cost": None, "key_player_score": 0.5})
            continue

        # 計算加權平均成本（使用 buy_price，若 NULL 則用同日收盤價代理）
        top_broker_ids = top_brokers["broker_id"].tolist()
        broker_details = grp[grp["broker_id"].isin(top_broker_ids)].copy()

        # 取有效的買入價
        if "buy_price" in broker_details.columns:
            valid_buys = broker_details[(broker_details["buy"] > 0) & broker_details["buy_price"].notna()]
        else:
            valid_buys = pd.DataFrame()

        if valid_buys.empty:
            results.append({"stock_id": sid, "key_player_cost": None, "key_player_score": 0.5})
            continue

        # 加權平均成本 = Σ(buy × buy_price) / Σ(buy)
        total_buy_value = (valid_buys["buy"] * valid_buys["buy_price"]).sum()
        total_buy_qty = valid_buys["buy"].sum()
        if total_buy_qty <= 0:
            results.append({"stock_id": sid, "key_player_cost": None, "key_player_score": 0.5})
            continue

        avg_cost = float(total_buy_value / total_buy_qty)
        results.append({"stock_id": sid, "key_player_cost": avg_cost, "key_player_score": 0.5})

    # 第二步：用 df_broker 最新日期的收盤價代理（或直接 close）計算 score
    result_df = pd.DataFrame(results)
    # 需要呼叫端傳入 current_price → 在 _base.py 整合時處理
    return result_df


def score_key_player_cost(
    cost_df: pd.DataFrame,
    price_map: dict[str, float],
) -> pd.DataFrame:
    """根據主力成本與現價比較計算分數。

    Args:
        cost_df: compute_key_player_cost_basis 的輸出
        price_map: {stock_id: latest_close_price}

    Returns:
        更新 key_player_score 後的 DataFrame
    """
    if cost_df.empty:
        return cost_df

    result = cost_df.copy()
    scores = []
    for _, row in result.iterrows():
        sid = row["stock_id"]
        cost = row.get("key_player_cost")
        current_price = price_map.get(sid)

        if cost is None or current_price is None or cost <= 0:
            scores.append(0.5)
            continue

        ratio = current_price / cost
        if ratio < 0.95:
            scores.append(0.8)  # 主力被套，護盤動力
        elif ratio > 1.10:
            scores.append(0.2)  # 主力已獲利，出貨風險
        else:
            scores.append(0.5)  # 成本附近

    result["key_player_score"] = scores
    return result


# ── P3-D1: 動態停損 ─────────────────────────────────────────────────


def compute_adaptive_atr_multiplier(
    df_price: pd.DataFrame,
    stock_ids: list[str],
    base_stop_mult: float = 1.5,
    mdd_window: int = 20,
) -> pd.DataFrame:
    """根據個股歷史最大回撤（MDD）調整 ATR 止損倍數。

    高 MDD 股票用更緊的止損（降低倍數），低 MDD 股票可用更寬的止損。

    規則：
    - MDD < 5% → 穩定股，倍數 × 1.2（放寬）
    - MDD 5~10% → 正常，維持 base
    - MDD 10~15% → 偏高，倍數 × 0.85
    - MDD > 15% → 高波動，倍數 × 0.7

    Args:
        df_price: DailyPrice DataFrame（stock_id / date / close / high / low）
        stock_ids: 要評估的股票清單
        base_stop_mult: 基準 ATR 止損倍數（預設 1.5）
        mdd_window: MDD 回溯天數（預設 20）

    Returns:
        DataFrame(stock_id, mdd_pct, adjusted_stop_mult)
    """
    if df_price.empty or not stock_ids:
        return pd.DataFrame(
            {
                "stock_id": stock_ids,
                "mdd_pct": [0.0] * len(stock_ids),
                "adjusted_stop_mult": [base_stop_mult] * len(stock_ids),
            }
        )

    results: list[dict] = []
    filtered = df_price[df_price["stock_id"].isin(stock_ids)]
    grouped = filtered.groupby("stock_id", sort=False)

    for sid in stock_ids:
        if sid not in grouped.groups:
            results.append({"stock_id": sid, "mdd_pct": 0.0, "adjusted_stop_mult": base_stop_mult})
            continue

        grp = grouped.get_group(sid).sort_values("date").tail(mdd_window)
        if len(grp) < 5:
            results.append({"stock_id": sid, "mdd_pct": 0.0, "adjusted_stop_mult": base_stop_mult})
            continue

        closes = grp["close"].values.astype(float)
        # 計算滾動最大回撤
        peak = closes[0]
        max_dd = 0.0
        for c in closes:
            if c > peak:
                peak = c
            dd = (c - peak) / peak if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd

        mdd_pct = abs(max_dd)  # 轉為正值百分比

        if mdd_pct < 0.05:
            mult = base_stop_mult * 1.2  # 穩定股，放寬
        elif mdd_pct < 0.10:
            mult = base_stop_mult  # 正常
        elif mdd_pct < 0.15:
            mult = base_stop_mult * 0.85  # 偏高，收緊
        else:
            mult = base_stop_mult * 0.70  # 高波動，大幅收緊

        results.append({"stock_id": sid, "mdd_pct": round(mdd_pct, 4), "adjusted_stop_mult": round(mult, 3)})

    return pd.DataFrame(results)


# ── P3-C2: 營收加速度推廣 ────────────────────────────────────────────


def compute_revenue_acceleration_score(
    df_revenue: pd.DataFrame,
    stock_ids: list[str],
    consecutive_threshold: int = 3,
) -> pd.DataFrame:
    """計算營收 YoY 加速度分數（推廣至所有模式）。

    指標：
    1. acceleration = YoY_latest - YoY_3m_ago（已存在於營收資料）
    2. consecutive_months: 連續 N 月 YoY 加速的計數（越長越好）
    3. 綜合分數（0~1）：區分「可持續成長」vs「一次性基數效應」

    評分規則：
    - 連續 ≥ 3 月加速且 acceleration > 10pp → 0.9（強勁加速）
    - 連續 ≥ 3 月加速 → 0.75（穩定加速）
    - acceleration > 0 但未連續 → 0.6（單月加速）
    - acceleration ≤ 0 → 0.3（減速）
    - 無資料 → 0.5

    Args:
        df_revenue: MonthlyRevenue DataFrame
            （stock_id / yoy_growth / yoy_3m_ago，最多 4 個月）
        stock_ids: 要評估的股票清單
        consecutive_threshold: 連續加速月數門檻（預設 3）

    Returns:
        DataFrame(stock_id, rev_accel_score, consecutive_accel_months)
    """
    if df_revenue.empty or not stock_ids:
        return pd.DataFrame(
            {
                "stock_id": stock_ids,
                "rev_accel_score": [0.5] * len(stock_ids),
                "consecutive_accel_months": [0] * len(stock_ids),
            }
        )

    results: list[dict] = []
    grouped = df_revenue[df_revenue["stock_id"].isin(stock_ids)]

    # 需要多個月的資料來計算連續性
    if "yoy_growth" not in grouped.columns:
        return pd.DataFrame(
            {
                "stock_id": stock_ids,
                "rev_accel_score": [0.5] * len(stock_ids),
                "consecutive_accel_months": [0] * len(stock_ids),
            }
        )

    rev_grouped = grouped.groupby("stock_id", sort=False)

    for sid in stock_ids:
        if sid not in rev_grouped.groups:
            results.append({"stock_id": sid, "rev_accel_score": 0.5, "consecutive_accel_months": 0})
            continue

        grp = rev_grouped.get_group(sid)
        if "date" in grp.columns:
            grp = grp.sort_values("date", ascending=False)
        else:
            grp = grp.reset_index(drop=True)
        if grp.empty:
            results.append({"stock_id": sid, "rev_accel_score": 0.5, "consecutive_accel_months": 0})
            continue

        # 計算加速度
        latest_yoy = grp.iloc[0].get("yoy_growth")
        yoy_3m = grp.iloc[0].get("yoy_3m_ago") if "yoy_3m_ago" in grp.columns else None

        if pd.isna(latest_yoy):
            results.append({"stock_id": sid, "rev_accel_score": 0.5, "consecutive_accel_months": 0})
            continue

        # 計算連續加速月數（YoY 遞增）
        consec = 0
        yoy_values = grp["yoy_growth"].dropna().values
        if len(yoy_values) >= 2:
            for i in range(len(yoy_values) - 1):
                if yoy_values[i] > yoy_values[i + 1]:
                    consec += 1
                else:
                    break

        accel = (float(latest_yoy) - float(yoy_3m)) if yoy_3m is not None and not pd.isna(yoy_3m) else None

        if accel is not None and accel > 10 and consec >= consecutive_threshold:
            score = 0.9  # 強勁加速：連續多月 + 加速幅度大（>10 百分點）
        elif consec >= consecutive_threshold:
            score = 0.75  # 穩定加速
        elif accel is not None and accel > 0:
            score = 0.6  # 單月加速
        elif accel is not None and accel <= 0:
            score = 0.3  # 減速
        else:
            score = 0.5  # 無法判定

        results.append({"stock_id": sid, "rev_accel_score": score, "consecutive_accel_months": consec})

    return pd.DataFrame(results)


def compute_quality_score(
    df_financial: pd.DataFrame,
    stock_ids: list[str],
    gm_weight: float = 0.6,
    fcf_weight: float = 0.4,
) -> pd.DataFrame:
    """基本面品質分數（Phase F）— 毛利率 YoY 改善 + FCF 正值旗標。

    目標：補足單一營收引擎的盲區，抓「營收成長但毛利萎縮」vs「雙引擎驅動」的區分。

    計算邏輯：
      1. gm_yoy_change：最新季毛利率 - 一年前同季毛利率
         （若前一年同季無資料，退回跨 4 季前的鄰近季度）
      2. fcf_positive：最新季 free_cf > 0 → 1，否則 0（含 NaN → 0）
      3. gm_yoy_rank：跨全池 percentile rank（0~1，無變化資料者為 0.5）
      4. quality_score = gm_weight × gm_yoy_rank + fcf_weight × fcf_positive

    Fallback：
      - 某股無任何財報資料 → quality_score = 0.5（中性）
      - gross_margin 全池缺失 → gm_yoy_rank = 0.5
      - free_cf 欄位缺失 → fcf_positive = 0.5

    Args:
        df_financial: _load_financial_data 輸出，至少含 stock_id, date, year, quarter,
                      gross_margin, free_cf 欄位（free_cf 缺失時退化）
        stock_ids: 候選股清單
        gm_weight: 毛利率 YoY 改善權重（預設 0.6）
        fcf_weight: FCF 正值權重（預設 0.4）

    Returns:
        DataFrame(stock_id, quality_score) — 分數 0~1，0.5 為中性
    """
    default = pd.DataFrame({"stock_id": stock_ids, "quality_score": [0.5] * len(stock_ids)})
    if df_financial is None or df_financial.empty or len(stock_ids) == 0:
        return default

    has_fcf = "free_cf" in df_financial.columns
    has_gm = "gross_margin" in df_financial.columns
    if not has_gm and not has_fcf:
        return default

    df = df_financial[df_financial["stock_id"].isin(stock_ids)].copy()
    if df.empty:
        return default

    # 依 stock_id + date 排序，方便取最新季與前一年同季
    if "date" in df.columns:
        df = df.sort_values(["stock_id", "date"], ascending=[True, False])

    # 逐股計算 gm_yoy_change 與 fcf_positive
    rows = []
    for sid in stock_ids:
        grp = df[df["stock_id"] == sid]
        if grp.empty:
            rows.append({"stock_id": sid, "gm_yoy_change": np.nan, "fcf_positive": np.nan})
            continue

        latest = grp.iloc[0]
        latest_gm = latest.get("gross_margin") if has_gm else None
        latest_fcf = latest.get("free_cf") if has_fcf else None

        gm_yoy_change = np.nan
        if has_gm and latest_gm is not None and not pd.isna(latest_gm):
            # 前一年同季：quarter 相同、year - 1
            latest_year = latest.get("year")
            latest_q = latest.get("quarter")
            if latest_year is not None and latest_q is not None:
                prior = grp[(grp["year"] == latest_year - 1) & (grp["quarter"] == latest_q)]
                if not prior.empty:
                    prior_gm = prior.iloc[0].get("gross_margin")
                    if prior_gm is not None and not pd.isna(prior_gm):
                        gm_yoy_change = float(latest_gm) - float(prior_gm)

        fcf_positive = np.nan
        if has_fcf and latest_fcf is not None and not pd.isna(latest_fcf):
            fcf_positive = 1.0 if float(latest_fcf) > 0 else 0.0

        rows.append(
            {
                "stock_id": sid,
                "gm_yoy_change": gm_yoy_change,
                "fcf_positive": fcf_positive,
            }
        )

    out = pd.DataFrame(rows)

    # gm YoY 跨池 percentile rank
    valid_gm = out["gm_yoy_change"].dropna()
    if len(valid_gm) >= 2:
        out["gm_yoy_rank"] = out["gm_yoy_change"].rank(pct=True)
        out["gm_yoy_rank"] = out["gm_yoy_rank"].fillna(0.5)
    else:
        out["gm_yoy_rank"] = 0.5

    # fcf 缺失者視為 0.5
    out["fcf_flag"] = out["fcf_positive"].fillna(0.5)

    out["quality_score"] = gm_weight * out["gm_yoy_rank"] + fcf_weight * out["fcf_flag"]

    # 完全無資料（gm_yoy_change 與 fcf_positive 皆 NaN）→ 0.5
    no_data_mask = out["gm_yoy_change"].isna() & out["fcf_positive"].isna()
    out.loc[no_data_mask, "quality_score"] = 0.5

    return out[["stock_id", "quality_score"]]


# ── P3-C3: 同業基本面排名 ────────────────────────────────────────────


def compute_peer_fundamental_ranking(
    df_financial: pd.DataFrame,
    stock_ids: list[str],
    industry_map: dict[str, str],
    bonus: float = 0.03,
) -> pd.DataFrame:
    """計算同產業基本面排名加成。

    同產業 ROE/毛利率/營收成長率排名：
    - 產業龍頭（基本面前 25%）→ +bonus
    - 產業落後者（後 25%）→ -bonus
    - 中間 50% → 0

    三指標等權平均。樣本不足（<4 家同業）時不做加減分。

    Args:
        df_financial: FinancialStatement 最新一季
            （stock_id / roe / gross_margin / revenue）
        stock_ids: 候選股清單
        industry_map: {stock_id: industry_category}
        bonus: 加成幅度（預設 ±0.03）

    Returns:
        DataFrame(stock_id, peer_rank_bonus)  值域 {-bonus, 0.0, +bonus}
    """
    default = pd.DataFrame({"stock_id": stock_ids, "peer_rank_bonus": [0.0] * len(stock_ids)})

    if df_financial.empty or not stock_ids:
        return default

    # 只取候選股
    df = df_financial[df_financial["stock_id"].isin(stock_ids)].copy()
    if df.empty:
        return default

    # 確保有產業映射
    df["industry"] = df["stock_id"].map(industry_map)
    df = df.dropna(subset=["industry"])
    if df.empty:
        return default

    metrics = ["roe", "gross_margin", "revenue"]
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        return default

    # 對每個指標計算產業內百分位排名
    rank_cols = []
    for metric in available_metrics:
        col_name = f"_peer_{metric}_pctile"
        df[col_name] = df.groupby("industry")[metric].rank(pct=True)
        rank_cols.append(col_name)

    # 計算同業中的平均排名
    df["_avg_peer_rank"] = df[rank_cols].mean(axis=1)

    # 計算每個產業的樣本數
    industry_counts = df.groupby("industry")["stock_id"].transform("count")

    # 加成規則
    bonuses = []
    for _, row in df.iterrows():
        n_peers = industry_counts.loc[row.name]
        if n_peers < 4:
            bonuses.append(0.0)  # 同業太少，不加減分
        elif row["_avg_peer_rank"] >= 0.75:
            bonuses.append(bonus)  # 龍頭
        elif row["_avg_peer_rank"] <= 0.25:
            bonuses.append(-bonus)  # 落後者
        else:
            bonuses.append(0.0)

    df["peer_rank_bonus"] = bonuses

    result = pd.DataFrame({"stock_id": stock_ids})
    result = result.merge(df[["stock_id", "peer_rank_bonus"]], on="stock_id", how="left")
    result["peer_rank_bonus"] = result["peer_rank_bonus"].fillna(0.0)
    return result


# ── P3-E3: MFE/MAE 分析 ─────────────────────────────────────────────


def compute_mfe_mae(
    df_records: pd.DataFrame,
    df_prices: pd.DataFrame,
    holding_days: int = 20,
) -> pd.DataFrame:
    """計算每筆推薦的最大有利偏移（MFE）和最大不利偏移（MAE）。

    追蹤推薦從進場日到 holding_days 天的完整路徑：
    - MFE (Maximum Favorable Excursion): 路徑中最高報酬率
    - MAE (Maximum Adverse Excursion): 路徑中最大虧損幅度
    - MFE/MAE 比率: > 1 表示有利偏移大於不利偏移（好的推薦）

    Args:
        df_records: 推薦記錄（scan_date / stock_id / close）
        df_prices: 日K線（stock_id / date / close / high / low）
        holding_days: 追蹤天數（預設 20）

    Returns:
        DataFrame(scan_date, stock_id, entry_close, mfe, mae, mfe_mae_ratio, final_return)
    """
    if df_records.empty or df_prices.empty:
        return pd.DataFrame(
            columns=["scan_date", "stock_id", "entry_close", "mfe", "mae", "mfe_mae_ratio", "final_return"]
        )

    results: list[dict] = []
    for _, rec in df_records.iterrows():
        scan_d = rec["scan_date"]
        sid = rec["stock_id"]
        entry_close = float(rec["close"])
        if entry_close <= 0:
            continue

        future = (
            df_prices[(df_prices["stock_id"] == sid) & (df_prices["date"] > scan_d)]
            .sort_values("date")
            .head(holding_days)
        )

        if future.empty:
            continue

        # 使用 high/low 計算日內最大偏移（若有），否則用 close
        if "high" in future.columns and "low" in future.columns:
            max_high = float(future["high"].max())
            min_low = float(future["low"].min())
        else:
            max_high = float(future["close"].max())
            min_low = float(future["close"].min())

        mfe = (max_high - entry_close) / entry_close
        mae = (min_low - entry_close) / entry_close  # 負值

        # MFE/MAE 比率（MAE 取絕對值）
        mfe_mae_ratio = mfe / abs(mae) if abs(mae) > 0.001 else float("inf")

        # 最終報酬
        final_close = float(future.iloc[-1]["close"])
        final_return = (final_close - entry_close) / entry_close

        results.append(
            {
                "scan_date": scan_d,
                "stock_id": sid,
                "entry_close": entry_close,
                "mfe": round(mfe, 4),
                "mae": round(mae, 4),
                "mfe_mae_ratio": round(mfe_mae_ratio, 2),
                "final_return": round(final_return, 4),
            }
        )

    return pd.DataFrame(results)


# ── 籌碼層級序數（數字越大 = 越高層級）──────────────────
_CHIP_TIER_ORDER: dict[str, int] = {
    "N/A": 0,
    "3F": 1,
    "4F": 2,
    "5F": 3,
    "6F": 4,
    "7F": 5,
    "8F": 6,
}


def detect_chip_tier_changes(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> pd.DataFrame:
    """比對當前與前次掃描的 chip_tier，回傳有變化的股票。

    Args:
        current: 含 stock_id, chip_tier 的 DataFrame（本次掃描結果）
        previous: 含 stock_id, chip_tier 的 DataFrame（前次 DiscoveryRecord）

    Returns:
        DataFrame[stock_id, prev_tier, curr_tier, direction]
        direction: "upgrade" / "downgrade"
    """
    if current.empty or previous.empty:
        return pd.DataFrame(columns=["stock_id", "prev_tier", "curr_tier", "direction"])

    merged = current[["stock_id", "chip_tier"]].merge(
        previous[["stock_id", "chip_tier"]],
        on="stock_id",
        suffixes=("_curr", "_prev"),
    )
    if merged.empty:
        return pd.DataFrame(columns=["stock_id", "prev_tier", "curr_tier", "direction"])

    merged["curr_ord"] = merged["chip_tier_curr"].map(_CHIP_TIER_ORDER).fillna(0).astype(int)
    merged["prev_ord"] = merged["chip_tier_prev"].map(_CHIP_TIER_ORDER).fillna(0).astype(int)

    changed = merged[merged["curr_ord"] != merged["prev_ord"]].copy()
    if changed.empty:
        return pd.DataFrame(columns=["stock_id", "prev_tier", "curr_tier", "direction"])

    changed["direction"] = np.where(changed["curr_ord"] < changed["prev_ord"], "downgrade", "upgrade")
    return changed.rename(columns={"chip_tier_prev": "prev_tier", "chip_tier_curr": "curr_tier"})[
        ["stock_id", "prev_tier", "curr_tier", "direction"]
    ].reset_index(drop=True)


class ScanAuditTrail:
    """掃描流程審計追蹤器。

    記錄每支股票在各階段的過濾/調分原因，供 debug 與效能分析使用。
    事件分為兩類：
    - hard_filter: 硬風控（通過或剔除，不可逆）
    - score_adj: 軟加成（composite_score 乘數調整，可累加）
    """

    def __init__(self) -> None:
        self._events: list[dict] = []

    def record_hard_filter(
        self,
        stage: str,
        stock_ids_before: set[str],
        stock_ids_after: set[str],
        reason: str,
    ) -> None:
        """記錄硬風控事件（剔除型）。"""
        removed = stock_ids_before - stock_ids_after
        if not removed:
            return
        for sid in removed:
            self._events.append(
                {
                    "stock_id": sid,
                    "stage": stage,
                    "type": "hard_filter",
                    "reason": reason,
                    "detail": "剔除",
                }
            )

    def record_score_adjustment(
        self,
        stage: str,
        stock_id: str,
        adjustment: float,
        reason: str,
    ) -> None:
        """記錄軟加成事件（調分型）。"""
        if abs(adjustment) < 1e-6:
            return
        self._events.append(
            {
                "stock_id": stock_id,
                "stage": stage,
                "type": "score_adj",
                "reason": reason,
                "detail": f"{adjustment:+.4f}",
            }
        )

    def record_score_adjustments_from_column(
        self,
        stage: str,
        df: pd.DataFrame,
        adj_column: str,
        reason: str,
    ) -> None:
        """從 DataFrame 欄位批次記錄調分事件。"""
        if adj_column not in df.columns:
            return
        for _, row in df.iterrows():
            val = row[adj_column]
            if pd.notna(val) and abs(val) >= 1e-6:
                self._events.append(
                    {
                        "stock_id": row["stock_id"],
                        "stage": stage,
                        "type": "score_adj",
                        "reason": reason,
                        "detail": f"{val:+.4f}",
                    }
                )

    def get_events(self) -> list[dict]:
        """回傳所有事件（原始 list）。"""
        return self._events

    def get_stock_trail(self, stock_id: str) -> list[dict]:
        """查詢特定股票的完整審計軌跡。"""
        return [e for e in self._events if e["stock_id"] == stock_id]

    def summary(self) -> dict:
        """產生摘要統計。"""
        hard = [e for e in self._events if e["type"] == "hard_filter"]
        adj = [e for e in self._events if e["type"] == "score_adj"]
        # 按 stage 分組統計硬風控剔除數
        hard_by_stage: dict[str, int] = {}
        for e in hard:
            hard_by_stage[e["stage"]] = hard_by_stage.get(e["stage"], 0) + 1
        # 按 stage 分組統計調分影響數
        adj_by_stage: dict[str, int] = {}
        for e in adj:
            adj_by_stage[e["stage"]] = adj_by_stage.get(e["stage"], 0) + 1
        return {
            "total_hard_filters": len(hard),
            "total_score_adjustments": len(adj),
            "hard_filters_by_stage": hard_by_stage,
            "score_adjustments_by_stage": adj_by_stage,
            "unique_stocks_filtered": len({e["stock_id"] for e in hard}),
            "unique_stocks_adjusted": len({e["stock_id"] for e in adj}),
        }

    def format_verbose(self, top_n: int = 10) -> str:
        """格式化為人類可讀的 verbose 輸出。"""
        lines: list[str] = []
        s = self.summary()
        lines.append("=" * 60)
        lines.append("掃描審計摘要")
        lines.append("=" * 60)
        lines.append(f"硬風控剔除: {s['total_hard_filters']} 支（{s['unique_stocks_filtered']} 支不重複）")
        lines.append(f"軟加成調分: {s['total_score_adjustments']} 筆（{s['unique_stocks_adjusted']} 支不重複）")

        if s["hard_filters_by_stage"]:
            lines.append("")
            lines.append("硬風控分階段剔除數:")
            for stage, count in sorted(s["hard_filters_by_stage"].items()):
                lines.append(f"  {stage}: {count}")

        if s["score_adjustments_by_stage"]:
            lines.append("")
            lines.append("軟加成分階段影響數:")
            for stage, count in sorted(s["score_adjustments_by_stage"].items()):
                lines.append(f"  {stage}: {count}")

        # 顯示前 N 支被剔除的股票詳情
        hard = [e for e in self._events if e["type"] == "hard_filter"]
        if hard:
            lines.append("")
            lines.append(f"被剔除股票（前 {top_n} 支）:")
            shown: set[str] = set()
            for e in hard:
                if e["stock_id"] in shown:
                    continue
                shown.add(e["stock_id"])
                lines.append(f"  {e['stock_id']} — {e['stage']}: {e['reason']}")
                if len(shown) >= top_n:
                    break

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class DiscoveryResult:
    """掃描結果資料容器。"""

    rankings: pd.DataFrame
    total_stocks: int
    after_coarse: int
    scan_date: date = field(default_factory=date.today)
    sector_summary: pd.DataFrame | None = None
    mode: str = "momentum"
    audit_trail: ScanAuditTrail | None = None
    sub_factor_df: pd.DataFrame | None = None
    # IC-aware 分數轉換動作 mapping（{factor_col: "kept"/"flipped"/"neutralized"}）
    # 供 CLI Top N 表格標記欄位狀態（N=neutralized, F=flipped）
    ic_actions: dict[str, str] = field(default_factory=dict)
