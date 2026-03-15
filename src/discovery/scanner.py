"""全市場選股掃描器 — 四階段漏斗從 ~2000 支股票篩選出 Top N 推薦。

漏斗架構：
  Stage 1: 從 DB 載入全市場日K + 法人資料
  Stage 2: 粗篩（股價/成交量/法人/動能 → 留 ~150 檔）
  Stage 3: 細評（模式專屬因子加權）
  Stage 3.5: 風險過濾（剔除高波動股）
  Stage 4: 排名 + 加上產業標籤 → 輸出 Top N

支援五種模式：
  - MomentumScanner: 短線動能（1~10 天），突破 + 資金流 + 量能擴張
  - SwingScanner: 中期波段（1~3 個月），趨勢 + 基本面 + 法人布局
  - ValueScanner: 價值修復，低估值 + 基本面轉佳
  - DividendScanner: 高息存股，高殖利率 + 配息穩定 + 估值合理
  - GrowthScanner: 高成長，營收高速成長 + 動能啟動
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import (
    Announcement,
    BrokerTrade,
    DailyPrice,
    FinancialStatement,
    HoldingDistribution,
    InstitutionalInvestor,
    MarginTrading,
    MonthlyRevenue,
    SecuritiesLending,
    StockInfo,
    StockValuation,
)
from src.discovery.universe import UniverseConfig, UniverseFilter

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

        avg_cost = 0.0
        net_position = 0.0
        total_profit = 0.0
        total_loss = 0.0
        wins = 0
        sell_events = 0
        total_buy_value = 0.0

        for _, row in grp.iterrows():
            buy_sh = float(row["buy"])
            sell_sh = float(row["sell"])
            bp = float(row["buy_price"])
            sp = float(row["sell_price"])

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

        # 近期活躍度（最後 recent_days 個交易日的淨買超）
        recent_dates = set(all_dates[-recent_days:] if len(all_dates) >= recent_days else all_dates)
        recent_net = float(grp[grp["date"].isin(recent_dates)]["net_buy"].sum())

        # 倉位趨勢（前後半段比較）
        mid = len(all_dates) // 2
        first_half = set(all_dates[:mid])
        last_half = set(all_dates[mid:])
        first_net = float(grp[grp["date"].isin(first_half)]["net_buy"].sum())
        last_net = float(grp[grp["date"].isin(last_half)]["net_buy"].sum())
        position_trend_up = last_net > first_net

        # 賣出比例
        total_buy_sh = float(grp["buy"].sum())
        total_sell_sh = float(grp["sell"].sum())
        sell_ratio = total_sell_sh / total_buy_sh if total_buy_sh > 0 else 1.0

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


class MarketScanner:
    """全市場選股掃描器（基底類別）。

    子類須覆寫 _coarse_filter() 和 _score_candidates() 以實作模式專屬邏輯。

    Args:
        min_price: 最低股價門檻
        max_price: 最高股價門檻
        min_volume: 最低成交量（股）
        top_n_candidates: 粗篩後保留數量
        top_n_results: 最終輸出數量
        lookback_days: 回溯天數（用於計算指標）
    """

    mode_name: str = "base"
    _auto_sync_broker: bool = False  # 子類設為 True 以在 Stage 2.5 自動補抓分點資料
    _revenue_months: int = 1  # 子類可設為 4 以啟用「本月 YoY - 3 個月前 YoY」加速度因子
    _COARSE_WEIGHTS: dict[str, float] = {"vol_rank": 0.30, "inst_rank": 0.40, "mom_rank": 0.30}

    def __init__(
        self,
        min_price: float = 10,
        max_price: float = 2000,
        min_volume: int = 500_000,
        top_n_candidates: int = 150,
        top_n_results: int = 30,
        lookback_days: int = 5,
        weekly_confirm: bool = False,
        universe_config: UniverseConfig | None = None,
    ) -> None:
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
        self.top_n_candidates = top_n_candidates
        self.top_n_results = top_n_results
        self.lookback_days = lookback_days
        self.weekly_confirm = weekly_confirm
        # Universe Filter：各子類可在 __init__ 中傳入模式專屬 config
        self._universe_config = universe_config or UniverseConfig()
        self._universe_filter = UniverseFilter(self._universe_config)

    def run(self) -> DiscoveryResult:
        """執行四階段漏斗掃描。"""
        self.scan_date = date.today()

        # Stage 0: 偵測市場狀態（Regime）
        try:
            from src.regime.detector import MarketRegimeDetector

            regime_info = MarketRegimeDetector().detect()
            self.regime = regime_info["regime"]
            logger.info("Stage 0: 市場狀態 = %s (TAIEX=%.0f)", self.regime, regime_info["taiex_close"])
        except Exception:
            self.regime = "sideways"
            logger.warning("Stage 0: 市場狀態偵測失敗，預設 sideways")

        # Stage 1: 載入資料
        df_price, df_inst, df_margin, df_revenue = self._load_market_data()
        if df_price.empty:
            logger.warning("無市場資料可供掃描")
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=0,
                after_coarse=0,
                mode=self.mode_name,
            )

        total_stocks = df_price["stock_id"].nunique()
        logger.info("Stage 1: 載入 %d 支股票的市場資料", total_stocks)

        # Stage 2: 粗篩
        candidates = self._coarse_filter(df_price, df_inst)
        after_coarse = len(candidates)
        logger.info("Stage 2: 粗篩後剩 %d 支候選股", after_coarse)

        if candidates.empty:
            return DiscoveryResult(
                rankings=pd.DataFrame(),
                total_stocks=total_stocks,
                after_coarse=0,
                mode=self.mode_name,
            )

        # Stage 2.5: 補抓候選股月營收（從 FinMind 逐股取得）
        candidate_ids = candidates["stock_id"].tolist()
        try:
            from src.data.pipeline import sync_revenue_for_stocks

            logger.info("Stage 2.5: 補抓 %d 支候選股月營收...", len(candidate_ids))
            rev_count = sync_revenue_for_stocks(candidate_ids)
            logger.info("Stage 2.5: 補抓完成，新增 %d 筆月營收", rev_count)
            # 重新載入營收資料（補抓後 DB 已更新）
            df_revenue = self._load_revenue_data(candidate_ids)
        except Exception:
            logger.warning("Stage 2.5: 月營收補抓失敗（可能無 FinMind token），使用既有資料")

        # Stage 2.5: 補抓候選股分點資料（僅 MomentumScanner 啟用）
        # 新出現的候選股（不在上次推薦或 watchlist 中）也能取得分點評分，避免因無資料而降級
        if self._auto_sync_broker:
            try:
                from src.data.pipeline import sync_broker_for_stocks

                logger.info("Stage 2.5: 補抓 %d 支候選股分點資料（DB 已有近期資料者跳過）...", len(candidate_ids))
                broker_count = sync_broker_for_stocks(candidate_ids)
                logger.info("Stage 2.5: 分點補抓完成，新增 %d 筆", broker_count)
            except Exception:
                logger.warning("Stage 2.5: 分點資料補抓失敗（可能無 FinMind token），使用既有資料")

        # Stage 2.7: 載入候選股近期 MOPS 公告（含基準期歷史供異常率計算）
        df_ann, df_ann_history = self._load_announcement_data(candidate_ids)
        if not df_ann.empty:
            logger.info("Stage 2.7: 載入 %d 筆 MOPS 公告", len(df_ann))
        else:
            logger.info("Stage 2.7: 無 MOPS 公告資料（消息面分數預設 0.5）")

        # Stage 3: 細評
        scored = self._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue, df_ann, df_ann_history)
        logger.info("Stage 3: 完成 %d 支候選股評分", len(scored))

        # Stage 3.3: 產業加成
        scored = self._apply_sector_bonus(scored)

        # Stage 3.3a: 產業同儕相對強度加成
        scored = self._apply_sector_relative_strength(scored)

        # Stage 3.4: 週線趨勢加成（若 weekly_confirm=True）
        if self.weekly_confirm:
            scored = self._apply_weekly_trend_bonus(scored)

        # Stage 3.5: 風險過濾
        scored = self._apply_risk_filter(scored, df_price)

        # Stage 4: 排名 + 產業標籤
        rankings = self._rank_and_enrich(scored)
        sector_summary = self._compute_sector_summary(rankings)
        logger.info("Stage 4: 輸出 Top %d", min(self.top_n_results, len(rankings)))

        return DiscoveryResult(
            rankings=rankings.head(self.top_n_results),
            total_stocks=total_stocks,
            after_coarse=after_coarse,
            sector_summary=sector_summary,
            mode=self.mode_name,
        )

    # ------------------------------------------------------------------ #
    #  Stage 1: 載入資料
    # ------------------------------------------------------------------ #

    def _get_universe_ids(self) -> list[str]:
        """執行 UniverseFilter 三層過濾，回傳候選 stock_id 清單。

        供 _load_market_data() 及子類覆寫版本呼叫，以 IN 子句限定 SQL 查詢範圍。
        若 UniverseFilter 失敗（DB 空等原因）回傳空清單，呼叫端的 SQL 不加 IN 子句。
        """
        universe_ids, universe_stats = self._universe_filter.run(mode=self.mode_name)
        logger.info(
            "Stage 0.5 UniverseFilter: SQL=%d → 流動性=%d → 趨勢=%d → 最終候選=%d",
            universe_stats.get("total_after_sql", 0),
            universe_stats.get("total_after_liquidity", 0),
            universe_stats.get("total_after_trend", 0),
            universe_stats.get("final_candidates", 0),
        )
        return universe_ids

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """從 DB 查詢最近的 daily_price + institutional + margin + monthly_revenue 資料。

        Stage 0.5（Universe Filter）：先執行三層 SQL/Pandas 過濾，取得 ~150-1500 支候選 stock_id，
        再以 IN 子句限定 DailyPrice/InstitutionalInvestor/MarginTrading 查詢範圍，
        避免全量載入 ~6000 支股票，節省約 75% I/O。
        """
        # Stage 0.5: Universe Filter — SQL 硬過濾 + 流動性 + 趨勢
        universe_ids = self._get_universe_ids()

        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        with get_session() as session:
            # 日K線（含 turnover，供流動性評分使用）
            price_query = select(
                DailyPrice.stock_id,
                DailyPrice.date,
                DailyPrice.open,
                DailyPrice.high,
                DailyPrice.low,
                DailyPrice.close,
                DailyPrice.volume,
                DailyPrice.turnover,
            ).where(DailyPrice.date >= cutoff)

            if universe_ids:
                price_query = price_query.where(DailyPrice.stock_id.in_(universe_ids))

            rows = session.execute(price_query).all()
            df_price = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "open", "high", "low", "close", "volume", "turnover"],
            )

            # 三大法人
            inst_query = select(
                InstitutionalInvestor.stock_id,
                InstitutionalInvestor.date,
                InstitutionalInvestor.name,
                InstitutionalInvestor.net,
            ).where(InstitutionalInvestor.date >= cutoff)

            if universe_ids:
                inst_query = inst_query.where(InstitutionalInvestor.stock_id.in_(universe_ids))

            rows = session.execute(inst_query).all()
            df_inst = pd.DataFrame(rows, columns=["stock_id", "date", "name", "net"])

            # 融資融券
            margin_query = select(
                MarginTrading.stock_id,
                MarginTrading.date,
                MarginTrading.margin_balance,
                MarginTrading.short_balance,
            ).where(MarginTrading.date >= cutoff)

            if universe_ids:
                margin_query = margin_query.where(MarginTrading.stock_id.in_(universe_ids))

            rows = session.execute(margin_query).all()
            df_margin = pd.DataFrame(rows, columns=["stock_id", "date", "margin_balance", "short_balance"])

        # 月營收（限候選股）
        df_revenue = self._load_revenue_data(
            stock_ids=universe_ids if universe_ids else None, months=self._revenue_months
        )

        return df_price, df_inst, df_margin, df_revenue

    def _load_revenue_data(self, stock_ids: list[str] | None = None, months: int = 1) -> pd.DataFrame:
        """從 DB 查詢月營收資料。

        Args:
            stock_ids: 限定查詢的股票清單，None 表示查全部
            months: 取每支股票最近幾個月的營收（1=最新, 2=含上月）
        """
        from sqlalchemy import func

        revenue_cutoff = date.today() - timedelta(days=180)

        with get_session() as session:
            base_filter = MonthlyRevenue.date >= revenue_cutoff
            if stock_ids:
                base_filter = base_filter & MonthlyRevenue.stock_id.in_(stock_ids)

            if months <= 1:
                # 原有邏輯：每支股票取最新一筆
                subq = (
                    select(
                        MonthlyRevenue.stock_id,
                        func.max(MonthlyRevenue.date).label("max_date"),
                    )
                    .where(base_filter)
                    .group_by(MonthlyRevenue.stock_id)
                    .subquery()
                )
                rows = session.execute(
                    select(
                        MonthlyRevenue.stock_id,
                        MonthlyRevenue.yoy_growth,
                        MonthlyRevenue.mom_growth,
                    ).join(
                        subq,
                        (MonthlyRevenue.stock_id == subq.c.stock_id) & (MonthlyRevenue.date == subq.c.max_date),
                    )
                ).all()

                return pd.DataFrame(
                    rows,
                    columns=["stock_id", "yoy_growth", "mom_growth"],
                )
            else:
                # months >= 2：取每支股票最近 N 筆，並加上 prev_yoy_growth / prev_mom_growth
                rows = session.execute(
                    select(
                        MonthlyRevenue.stock_id,
                        MonthlyRevenue.date,
                        MonthlyRevenue.yoy_growth,
                        MonthlyRevenue.mom_growth,
                    )
                    .where(base_filter)
                    .order_by(MonthlyRevenue.stock_id, MonthlyRevenue.date.desc())
                ).all()

        df_all = pd.DataFrame(
            rows,
            columns=["stock_id", "date", "yoy_growth", "mom_growth"],
        )
        if df_all.empty:
            cols = ["stock_id", "yoy_growth", "mom_growth", "prev_yoy_growth", "prev_mom_growth"]
            if months >= 4:
                cols.append("yoy_3m_ago")
            return pd.DataFrame(columns=cols)

        # 每支股票取最近 months 筆
        result_rows = []
        for sid, grp in df_all.groupby("stock_id"):
            grp = grp.sort_values("date", ascending=False).head(months)
            latest = grp.iloc[0]
            row = {
                "stock_id": sid,
                "yoy_growth": latest["yoy_growth"],
                "mom_growth": latest["mom_growth"],
                "prev_yoy_growth": grp.iloc[1]["yoy_growth"] if len(grp) >= 2 else None,
                "prev_mom_growth": grp.iloc[1]["mom_growth"] if len(grp) >= 2 else None,
            }
            if months >= 4:
                row["yoy_3m_ago"] = grp.iloc[3]["yoy_growth"] if len(grp) >= 4 else None
            result_rows.append(row)

        return pd.DataFrame(result_rows)

    def _load_financial_data(self, stock_ids: list[str], quarters: int = 5) -> pd.DataFrame:
        """從 DB 查詢最近 N 季財務資料（EPS / ROE / 毛利率 / 負債比）。

        Args:
            stock_ids: 限定查詢的股票清單
            quarters: 每支股票取最近幾季（預設 5 季，足以計算 YoY + QoQ）

        Returns:
            DataFrame(stock_id, date, year, quarter, eps, roe, gross_margin, debt_ratio)
            每支股票最多 quarters 筆，按 date desc 排列。無資料時回傳空 DataFrame。
        """
        _cols = ["stock_id", "date", "year", "quarter", "eps", "roe", "gross_margin", "debt_ratio"]
        cutoff = date.today() - timedelta(days=quarters * 100)  # ~100 天/季，5 季 ≈ 500 天
        try:
            with get_session() as session:
                rows = session.execute(
                    select(
                        FinancialStatement.stock_id,
                        FinancialStatement.date,
                        FinancialStatement.year,
                        FinancialStatement.quarter,
                        FinancialStatement.eps,
                        FinancialStatement.roe,
                        FinancialStatement.gross_margin,
                        FinancialStatement.debt_ratio,
                    )
                    .where(
                        FinancialStatement.stock_id.in_(stock_ids),
                        FinancialStatement.date >= cutoff,
                    )
                    .order_by(FinancialStatement.stock_id, FinancialStatement.date.desc())
                ).all()
        except Exception:
            return pd.DataFrame(columns=_cols)

        if not rows:
            return pd.DataFrame(columns=_cols)
        return pd.DataFrame(rows, columns=_cols)

    def _load_announcement_data(
        self,
        stock_ids: list[str] | None = None,
        days: int = 10,
        baseline_days: int = 180,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """從 DB 查詢 MOPS 重大訊息公告（近期 + 基準期歷史）。

        Args:
            stock_ids: 限定查詢的股票清單，None 表示查全部
            days: 近期回溯天數（供評分用）
            baseline_days: 基準期天數（供異常公告率計算用，預設 180）

        Returns:
            (recent_df, history_df)
            - recent_df: 最近 days 天，含 stock_id/date/seq/subject/sentiment/event_type
            - history_df: 最近 baseline_days 天，含 stock_id/date（供異常率計算）
        """
        today = date.today()
        recent_cutoff = today - timedelta(days=days)
        baseline_cutoff = today - timedelta(days=baseline_days)

        col_names = ["stock_id", "date", "seq", "subject", "sentiment", "event_type"]

        with get_session() as session:
            # 近期完整資料
            query = select(
                Announcement.stock_id,
                Announcement.date,
                Announcement.seq,
                Announcement.subject,
                Announcement.sentiment,
                Announcement.event_type,
            ).where(Announcement.date >= recent_cutoff)

            if stock_ids:
                query = query.where(Announcement.stock_id.in_(stock_ids))

            recent_rows = session.execute(query).all()

            # 基準期（僅需 stock_id + date）
            hist_query = select(Announcement.stock_id, Announcement.date).where(Announcement.date >= baseline_cutoff)
            if stock_ids:
                hist_query = hist_query.where(Announcement.stock_id.in_(stock_ids))

            history_rows = session.execute(hist_query).all()

        recent_df = pd.DataFrame(recent_rows, columns=col_names)
        history_df = pd.DataFrame(history_rows, columns=["stock_id", "date"])
        return recent_df, history_df

    def _compute_news_scores(
        self,
        stock_ids: list[str],
        df_ann: pd.DataFrame,
        df_ann_history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """計算消息面分數（時間衰減 × 事件類型加權 × 異常公告率，percentile 排名）。

        公式：
            各公告加權值 = exp(-0.2 × days_ago) × type_weight
            net_score = Σ(加權值 for 正面) - Σ(加權值 for 負面)
            abnormal_multiplier：z>2 最高 +50%，z<-1 最低降至 70%，平常 1.0
            net_score_adj = net_score × abnormal_multiplier
            news_score = percentile_rank(net_score_adj)

        Args:
            stock_ids: 候選股代號清單
            df_ann: 近期公告 DataFrame（須含 sentiment, event_type, date 欄位）
            df_ann_history: 基準期公告歷史（stock_id, date），供異常率計算，None 則略過

        Returns:
            DataFrame(stock_id, news_score) — 分數 0~1，0.5 為中性預設
        """
        default = pd.DataFrame({"stock_id": stock_ids, "news_score": [0.5] * len(stock_ids)})

        if df_ann.empty:
            return default

        ann = df_ann[df_ann["stock_id"].isin(stock_ids)].copy()
        if ann.empty:
            return default

        today = date.today()
        ann["days_ago"] = ann["date"].apply(lambda d: max(0, (today - d).days))

        # event_type 欄位相容（舊資料無此欄則預設 general）
        if "event_type" not in ann.columns:
            ann["event_type"] = "general"
        else:
            ann["event_type"] = ann["event_type"].fillna("general")

        ann["decay_weight"] = ann.apply(
            lambda row: compute_news_decay_weight(row["days_ago"], row["event_type"]),
            axis=1,
        )

        pos_df = ann[ann["sentiment"] == 1].groupby("stock_id")["decay_weight"].sum().reset_index(name="pos_weighted")
        neg_df = ann[ann["sentiment"] == -1].groupby("stock_id")["decay_weight"].sum().reset_index(name="neg_weighted")

        df = pd.DataFrame({"stock_id": stock_ids})
        df = df.merge(pos_df, on="stock_id", how="left")
        df = df.merge(neg_df, on="stock_id", how="left")
        df["pos_weighted"] = df["pos_weighted"].fillna(0.0)
        df["neg_weighted"] = df["neg_weighted"].fillna(0.0)
        df["net_score"] = df["pos_weighted"] - df["neg_weighted"]

        # 異常公告率乘數（僅在 history 有效時套用）
        if df_ann_history is not None and not df_ann_history.empty:
            z_series = compute_abnormal_announcement_rate(df_ann_history, stock_ids)

            def _to_multiplier(z: float) -> float:
                if z > 2.0:
                    return min(1.0 + (z - 2.0) * 0.15, 1.5)
                elif z < -1.0:
                    return max(1.0 + z * 0.10, 0.7)
                return 1.0

            mult_map = {sid: _to_multiplier(float(z)) for sid, z in z_series.items()}
            df["mult"] = df["stock_id"].map(mult_map).fillna(1.0)
            df["net_score_adj"] = df["net_score"] * df["mult"]
        else:
            df["net_score_adj"] = df["net_score"]

        # 無任何加權公告 → 回傳預設
        if df["net_score_adj"].abs().sum() == 0:
            return default

        df["news_score"] = df["net_score_adj"].rank(pct=True)

        return df[["stock_id", "news_score"]]

    # ------------------------------------------------------------------ #
    #  產業加成
    # ------------------------------------------------------------------ #

    def _compute_sector_bonus(self, stock_ids: list[str]) -> pd.DataFrame:
        """計算候選股的產業熱度加成分數（±5%）。

        呼叫 IndustryRotationAnalyzer 計算產業排名，將排名百分位
        映射為 sector_bonus（-0.05 ~ +0.05）。失敗時回傳全 0。

        Returns:
            DataFrame(stock_id, sector_bonus)
        """
        default = pd.DataFrame({"stock_id": stock_ids, "sector_bonus": [0.0] * len(stock_ids)})
        try:
            from src.industry.analyzer import IndustryRotationAnalyzer

            analyzer = IndustryRotationAnalyzer(watchlist=stock_ids)
            result = analyzer.compute_sector_scores_for_stocks(stock_ids)
            if result.empty:
                return default
            return result
        except Exception:
            logger.warning("產業加成計算失敗，跳過")
            return default

    def _apply_sector_bonus(self, scored: pd.DataFrame) -> pd.DataFrame:
        """將產業加成套用到 composite_score。

        final_score = composite_score × (1 + sector_bonus)
        """
        if scored.empty:
            return scored

        stock_ids = scored["stock_id"].tolist()
        bonus_df = self._compute_sector_bonus(stock_ids)
        scored = scored.merge(bonus_df, on="stock_id", how="left")
        scored["sector_bonus"] = scored["sector_bonus"].fillna(0.0)
        scored["composite_score"] = scored["composite_score"] * (1 + scored["sector_bonus"])
        logger.info(
            "Stage 3.3: 產業加成已套用（範圍 %.3f ~ %.3f）", scored["sector_bonus"].min(), scored["sector_bonus"].max()
        )
        return scored

    # ------------------------------------------------------------------ #
    #  週線趨勢加成
    # ------------------------------------------------------------------ #

    def _compute_weekly_trend_bonus(self, stock_ids: list[str]) -> pd.DataFrame:
        """計算候選股的週線趨勢加成分數（±5%）。

        從 DB 讀取近 90 天日K，聚合為週K，依下列兩個週線信號判斷趨勢：
          - SMA13（13 週均線）：收盤 > SMA13 → 多頭信號
          - RSI14（週 RSI14）：RSI > 50 → 多頭，< 50 → 空頭

        兩信號均多頭 → +0.05；兩信號均空頭 → -0.05；其餘 → 0.0。
        資料不足（< 13 週）時信號以 NaN 填補，該方向的信號直接略過。

        Returns:
            DataFrame(stock_id, weekly_bonus)  值域 {-0.05, 0.0, +0.05}
        """
        from src.features.indicators import aggregate_to_weekly

        default = pd.DataFrame({"stock_id": stock_ids, "weekly_bonus": [0.0] * len(stock_ids)})

        try:
            cutoff = date.today() - timedelta(days=90)

            with get_session() as session:
                rows = (
                    session.execute(
                        select(DailyPrice)
                        .where(DailyPrice.stock_id.in_(stock_ids))
                        .where(DailyPrice.date >= cutoff)
                        .order_by(DailyPrice.stock_id, DailyPrice.date)
                    )
                    .scalars()
                    .all()
                )

            if not rows:
                return default

            df_all = pd.DataFrame(
                [
                    {
                        "stock_id": r.stock_id,
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

            results: list[dict] = []
            for sid in stock_ids:
                stock_df = df_all[df_all["stock_id"] == sid].drop(columns=["stock_id"])
                if stock_df.empty:
                    results.append({"stock_id": sid, "weekly_bonus": 0.0})
                    continue

                weekly = aggregate_to_weekly(stock_df)
                if weekly.empty:
                    results.append({"stock_id": sid, "weekly_bonus": 0.0})
                    continue

                last = weekly.iloc[-1]
                last_close = float(last["close"])
                sma13 = last["sma_13"]
                rsi14 = last["rsi_14"]

                bullish = 0
                bearish = 0

                if pd.notna(sma13):
                    if last_close > float(sma13):
                        bullish += 1
                    else:
                        bearish += 1

                if pd.notna(rsi14):
                    if float(rsi14) > 50:
                        bullish += 1
                    else:
                        bearish += 1

                if bullish == 2:
                    bonus = 0.05
                elif bearish == 2:
                    bonus = -0.05
                else:
                    bonus = 0.0

                results.append({"stock_id": sid, "weekly_bonus": bonus})

            return pd.DataFrame(results) if results else default

        except Exception:
            logger.warning("週線趨勢加成計算失敗，跳過")
            return default

    def _apply_weekly_trend_bonus(self, scored: pd.DataFrame) -> pd.DataFrame:
        """將週線趨勢加成套用到 composite_score。

        final_score = composite_score × (1 + weekly_bonus)
        """
        if scored.empty:
            return scored

        stock_ids = scored["stock_id"].tolist()
        bonus_df = self._compute_weekly_trend_bonus(stock_ids)
        scored = scored.merge(bonus_df, on="stock_id", how="left")
        scored["weekly_bonus"] = scored["weekly_bonus"].fillna(0.0)
        scored["composite_score"] = scored["composite_score"] * (1 + scored["weekly_bonus"])
        logger.info(
            "Stage 3.4: 週線趨勢加成已套用（範圍 %.3f ~ %.3f）",
            scored["weekly_bonus"].min(),
            scored["weekly_bonus"].max(),
        )
        return scored

    # ------------------------------------------------------------------ #
    #  產業同儕相對強度加成
    # ------------------------------------------------------------------ #

    def _compute_sector_relative_strength(self, stock_ids: list[str]) -> pd.DataFrame:
        """計算個股相對同產業中位數的相對強度加成（±3%）。

        從 DB 讀取近 30 天日K（確保包含 20 個交易日），
        取得 StockInfo 產業對照，呼叫純函數計算。失敗時回傳全 0。

        Returns:
            DataFrame(stock_id, relative_strength_bonus)  值域 {-0.03, 0.0, +0.03}
        """
        from src.industry.analyzer import compute_sector_relative_strength

        default = pd.DataFrame({"stock_id": stock_ids, "relative_strength_bonus": [0.0] * len(stock_ids)})

        try:
            cutoff = date.today() - timedelta(days=30)

            with get_session() as session:
                price_rows = session.execute(
                    select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close)
                    .where(DailyPrice.stock_id.in_(stock_ids))
                    .where(DailyPrice.date >= cutoff)
                    .order_by(DailyPrice.stock_id, DailyPrice.date)
                ).all()

            if not price_rows:
                return default

            df_price = pd.DataFrame([{"stock_id": r[0], "date": r[1], "close": r[2]} for r in price_rows])

            with get_session() as session:
                info_rows = session.execute(
                    select(StockInfo.stock_id, StockInfo.industry_category).where(StockInfo.stock_id.in_(stock_ids))
                ).all()
            industry_map = {r[0]: (r[1] or "未分類") for r in info_rows}

            return compute_sector_relative_strength(stock_ids, df_price, industry_map)

        except Exception:
            logger.warning("產業相對強度計算失敗，跳過")
            return default

    def _apply_sector_relative_strength(self, scored: pd.DataFrame) -> pd.DataFrame:
        """將產業同儕相對強度加成套用到 composite_score。

        final_score = composite_score × (1 + relative_strength_bonus)
        """
        if scored.empty:
            return scored

        stock_ids = scored["stock_id"].tolist()
        rs_df = self._compute_sector_relative_strength(stock_ids)
        scored = scored.merge(rs_df, on="stock_id", how="left")
        scored["relative_strength_bonus"] = scored["relative_strength_bonus"].fillna(0.0)
        scored["composite_score"] = scored["composite_score"] * (1 + scored["relative_strength_bonus"])
        logger.info(
            "Stage 3.3a: 產業相對強度加成已套用（範圍 %.3f ~ %.3f）",
            scored["relative_strength_bonus"].min(),
            scored["relative_strength_bonus"].max(),
        )
        return scored

    # ------------------------------------------------------------------ #
    #  Stage 2: 粗篩
    # ------------------------------------------------------------------ #

    def _base_filter(self, df_price: pd.DataFrame) -> pd.DataFrame:
        """基礎過濾：股價範圍 + 成交量。供子類 _coarse_filter 呼叫。

        ETF/指數/權證排除已由 UniverseFilter Stage 1（SQL 硬過濾）負責，
        此處僅保留模式專屬的股價區間與成交量門檻。
        """
        latest_date = df_price["date"].max()
        latest = df_price[df_price["date"] == latest_date].copy()

        if latest.empty:
            return pd.DataFrame()

        mask = (latest["close"] >= self.min_price) & (latest["volume"] >= self.min_volume)
        if self.max_price is not None:
            mask = mask & (latest["close"] <= self.max_price)
        return latest[mask].copy()

    def _effective_top_n(self, universe_size: int) -> int:
        """依 Universe 大小自適應粗篩候選數。

        Universe 超過閾值時以 15% 比例線性擴展，防止從大量候選直接壓縮至固定 N 個；
        Universe 較小時以 top_n_candidates 為下限保護，確保候選池不過少。

        例：top_n_candidates=150 時：
            universe=1000 → max(150, 150) = 150
            universe=1500 → max(150, 225) = 225
            universe=200  → max(150,  30) = 150（下限保護）
        """
        return max(self.top_n_candidates, int(universe_size * 0.15))

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """粗篩：股價/量/法人/動能加權 → 取 top N candidates。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 計算粗篩分數
        # 1) 成交量排名分數（量大加分）
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 2) 法人 5 日累積淨買超排名（比單日穩定，過濾換手雜訊）
        if not df_inst.empty:
            inst_dates = sorted(df_inst["date"].unique())
            recent_5_dates = inst_dates[-5:]
            inst_5d = df_inst[df_inst["date"].isin(recent_5_dates)]
            inst_net = inst_5d.groupby("stock_id")["net"].sum().reset_index()
            inst_net.columns = ["stock_id", "inst_net"]
            filtered = filtered.merge(inst_net, on="stock_id", how="left")
            filtered["inst_net"] = filtered["inst_net"].fillna(0)
            filtered["inst_rank"] = filtered["inst_net"].rank(pct=True)
        else:
            filtered["inst_net"] = 0
            filtered["inst_rank"] = 0.5

        # 3) 短期動能（5 日報酬，比單日漲跌更穩定）
        dates = sorted(df_price["date"].unique())
        ref_date = dates[-5] if len(dates) >= 5 else (dates[0] if len(dates) >= 2 else None)
        if ref_date is not None:
            ref = df_price[df_price["date"] == ref_date][["stock_id", "close"]].rename(columns={"close": "ref_close"})
            filtered = filtered.merge(ref, on="stock_id", how="left")
            filtered["momentum"] = (
                (filtered["close"] - filtered["ref_close"]) / filtered["ref_close"].replace(0, float("nan"))
            ).fillna(0)
            filtered["mom_rank"] = filtered["momentum"].rank(pct=True)
        else:
            filtered["momentum"] = 0
            filtered["mom_rank"] = 0.5

        # 粗篩綜合分（依 _COARSE_WEIGHTS 類別屬性動態計算）
        filtered["coarse_score"] = sum(
            filtered[k] * v for k, v in self._COARSE_WEIGHTS.items() if k in filtered.columns
        )

        # 取 top N
        filtered = filtered.nlargest(self._effective_top_n(len(filtered)), "coarse_score")
        return filtered

    # ------------------------------------------------------------------ #
    #  Stage 3: 細評
    # ------------------------------------------------------------------ #

    def _score_candidates(
        self,
        candidates: pd.DataFrame,
        df_price: pd.DataFrame,
        df_inst: pd.DataFrame,
        df_margin: pd.DataFrame,
        df_revenue: pd.DataFrame,
        df_ann: pd.DataFrame | None = None,
        df_ann_history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """對候選股進行多維度評分（通用流程）。

        子類透過覆寫 _compute_*  方法客製化各維度計算，
        透過 _compute_extra_scores() 新增額外維度，
        透過 _post_score() 做加權後處理。
        """
        stock_ids = candidates["stock_id"].tolist()
        ann = df_ann if df_ann is not None else pd.DataFrame()

        # 各維度分數（子類覆寫 _compute_* 即可客製化）
        score_dfs = [
            self._compute_technical_scores(stock_ids, df_price),
            self._compute_chip_scores(stock_ids, df_inst, df_price, df_margin),
            self._compute_fundamental_scores(stock_ids, df_revenue),
            self._compute_news_scores(stock_ids, ann, df_ann_history=df_ann_history),
        ]
        # hook：子類可加額外維度（如 ValueScanner 的 valuation_score）
        score_dfs.extend(self._compute_extra_scores(stock_ids))

        candidates = candidates.copy()
        for df in score_dfs:
            candidates = candidates.merge(df, on="stock_id", how="left")

        # 所有 *_score 欄位 fillna(0.5)
        score_cols = [c for c in candidates.columns if c.endswith("_score") and c != "composite_score"]
        for col in score_cols:
            candidates[col] = candidates[col].fillna(0.5)

        # chip_tier 字串欄位 fillna
        if "chip_tier" in candidates.columns:
            candidates["chip_tier"] = candidates["chip_tier"].fillna("N/A")

        # 根據 regime 動態加權（weight key 直接映射 {key}_score 欄位）
        from src.regime.detector import MarketRegimeDetector

        regime = getattr(self, "regime", "sideways")
        w = MarketRegimeDetector.get_weights(self.mode_name, regime)

        composite = pd.Series(0.0, index=candidates.index)
        for key, weight in w.items():
            col = f"{key}_score"
            if col in candidates.columns:
                composite += candidates[col] * weight
        candidates["composite_score"] = composite

        # hook：子類可在加權後做額外處理
        candidates = self._post_score(candidates)

        # 進出場建議欄位（依 regime 調整 ATR 倍數）
        regime = getattr(self, "regime", "sideways")
        entry_exit = self._compute_entry_exit_cols(stock_ids, df_price, regime=regime)
        candidates = candidates.merge(entry_exit, on="stock_id", how="left")

        return candidates

    def _compute_extra_scores(self, stock_ids: list[str]) -> list[pd.DataFrame]:
        """hook：子類可覆寫以新增額外評分維度。回傳 DataFrame 的 list。"""
        return []

    def _post_score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """hook：子類可在加權後做額外處理。"""
        return candidates

    # Regime 自適應 ATR 止損/目標倍數
    _REGIME_ATR_PARAMS: dict[str, tuple[float, float]] = {
        "bull": (1.5, 3.5),
        "sideways": (1.5, 3.0),
        "bear": (1.2, 2.5),
    }

    def _compute_entry_exit_cols(
        self, stock_ids: list[str], df_price: pd.DataFrame, regime: str = "sideways"
    ) -> pd.DataFrame:
        """計算每支股票的進出場建議欄位。

        欄位：
          entry_price  — 當日收盤價
          stop_loss    — entry_price - stop_mult × ATR14（依 regime 調整）
          take_profit  — entry_price + target_mult × ATR14（依 regime 調整）
          entry_trigger — 依均線位置與波動率產生中文說明
          valid_until  — scan_date + 5 工作日

        Args:
            stock_ids: 要計算的股票代號清單
            df_price: 日K線 DataFrame
            regime: 市場狀態（"bull"/"sideways"/"bear"），決定 ATR 倍數
                    bull      → stop×1.5 / target×3.5
                    sideways  → stop×1.5 / target×3.0（預設）
                    bear      → stop×1.2 / target×2.5

        Returns:
            DataFrame，index reset，欄位含 stock_id 及上述五欄
        """
        stop_mult, target_mult = self._REGIME_ATR_PARAMS.get(regime, (1.5, 3.0))

        scan_date = getattr(self, "scan_date", date.today())
        valid_until = (pd.Timestamp(scan_date) + pd.offsets.BDay(5)).date()

        # 預先 groupby 一次（O(N)），避免迴圈中反覆 boolean filter（O(N²)）
        price_grouped = df_price.sort_values("date").groupby("stock_id", sort=False)

        rows = []
        for sid in stock_ids:
            stock_data = price_grouped.get_group(sid).tail(30) if sid in price_grouped.groups else pd.DataFrame()

            if stock_data.empty:
                rows.append(
                    {
                        "stock_id": sid,
                        "entry_price": None,
                        "stop_loss": None,
                        "take_profit": None,
                        "entry_trigger": "資料不足，僅供參考",
                        "valid_until": valid_until,
                    }
                )
                continue

            close = float(stock_data["close"].values[-1])

            if len(stock_data) < 15:
                rows.append(
                    {
                        "stock_id": sid,
                        "entry_price": round(close, 2),
                        "stop_loss": None,
                        "take_profit": None,
                        "entry_trigger": "資料不足，僅供參考",
                        "valid_until": valid_until,
                    }
                )
                continue

            atr14 = _calc_atr14(stock_data)
            stop_loss = round(close - stop_mult * atr14, 2) if atr14 > 0 else None
            take_profit = round(close + target_mult * atr14, 2) if atr14 > 0 else None

            # SMA20 — 用 tail(20) 的平均收盤價
            sma20 = float(stock_data["close"].tail(20).mean())

            # 均線位置判斷
            if sma20 > 0:
                if close > sma20 * 1.01:
                    trigger = "站上均線"
                elif close >= sma20 * 0.99:
                    trigger = "貼近均線"
                else:
                    trigger = "均線下方，等待確認"
            else:
                trigger = "均線下方，等待確認"

            # 附加波動率說明
            atr_pct = atr14 / close if close > 0 else 0.0
            if atr_pct < 0.02:
                trigger += "，低波動"
            elif atr_pct > 0.04:
                trigger += "，高波動謹慎"

            rows.append(
                {
                    "stock_id": sid,
                    "entry_price": round(close, 2),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_trigger": trigger,
                    "valid_until": valid_until,
                }
            )

        return pd.DataFrame(rows)

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """從原始 OHLCV 計算技術面分數（6 因子：SMA + 動能 + 價格位置 + 量能比 + 波動收斂 + 量價背離）。"""
        results = []

        # 預先 groupby 一次（O(N)），避免迴圈中反覆 boolean filter（O(N²)）
        grouped = df_price.sort_values("date").groupby("stock_id", sort=False)

        for sid in stock_ids:
            stock_data = grouped.get_group(sid) if sid in grouped.groups else pd.DataFrame()
            if len(stock_data) < 3:
                results.append({"stock_id": sid, "technical_score": 0.5})
                continue

            closes = stock_data["close"].values
            highs = stock_data["high"].values
            lows = stock_data["low"].values

            score = 0.0
            n_factors = 0

            # 1) SMA 趨勢：收盤 > SMA5 → 加分
            if len(closes) >= 5:
                sma5 = closes[-5:].mean()
                score += 1.0 if closes[-1] > sma5 else 0.0
                n_factors += 1

            # 2) 短期動能：最近 3 日漲幅
            if len(closes) >= 4:
                ret_3d = (closes[-1] - closes[-4]) / closes[-4]
                # 歸一化到 0~1
                score += max(0.0, min(1.0, 0.5 + ret_3d * 10))
                n_factors += 1

            # 3) 價格位置：收盤在近期高低區間的位置
            if len(closes) >= 3:
                high_max = highs[-5:].max() if len(highs) >= 5 else highs.max()
                low_min = lows[-5:].min() if len(lows) >= 5 else lows.min()
                price_range = high_max - low_min
                if price_range > 0:
                    position = (closes[-1] - low_min) / price_range
                    score += position
                else:
                    score += 0.5
                n_factors += 1

            # 4) 成交量趨勢：最新量 > 平均量
            volumes = stock_data["volume"].values
            if len(volumes) >= 3:
                avg_vol = volumes[:-1].mean()
                if avg_vol > 0:
                    vol_ratio = min(2.0, volumes[-1] / avg_vol)
                    score += vol_ratio / 2.0
                else:
                    score += 0.5
                n_factors += 1

            # 5) 波動度收斂（BB 寬度縮窄）：CV 越低代表盤整越緊，突破潛力越大
            if len(closes) >= 5:
                recent_closes = closes[-5:]
                mean_price = recent_closes.mean()
                if mean_price > 0:
                    cv = recent_closes.std(ddof=0) / mean_price
                    # CV 越小分數越高：用 1 - 歸一化 CV（CV 通常 < 0.1，用 0.1 做上限）
                    score += max(0.0, 1.0 - min(cv / 0.1, 1.0))
                else:
                    score += 0.5
                n_factors += 1

            # 6) 量價背離偵測：價格方向 vs 成交量方向一致性
            if len(closes) >= 3 and len(volumes) >= 3:
                price_chg = closes[-1] - closes[-3]
                vol_chg = float(volumes[-1]) - float(volumes[-3])
                # 價漲量增 → 健康上漲（高分），價漲量縮 → 看空背離（低分）
                # 價跌量增 → 看空（低分），價跌量縮 → 中性
                if price_chg > 0 and vol_chg > 0:
                    score += 1.0  # 價漲量增：最佳
                elif price_chg > 0 and vol_chg <= 0:
                    score += 0.3  # 價漲量縮：背離
                elif price_chg <= 0 and vol_chg <= 0:
                    score += 0.5  # 價跌量縮：中性
                else:
                    score += 0.2  # 價跌量增：最差
                n_factors += 1

            tech_score = score / max(n_factors, 1)
            results.append({"stock_id": sid, "technical_score": tech_score})

        return pd.DataFrame(results)

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """計算籌碼面分數（5 因子：淨買超 × 3 + 連續買超天數 + 買超佔量比）。"""
        if df_inst.empty:
            return pd.DataFrame(
                {
                    "stock_id": stock_ids,
                    "chip_score": [0.5] * len(stock_ids),
                }
            )

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]

        # 預先 groupby 一次（O(N)），避免迴圈中反覆 boolean filter（O(N²)）
        inst_grouped = inst_filtered.groupby("stock_id", sort=False)
        price_grouped = (
            df_price[df_price["stock_id"].isin(stock_ids)].groupby("stock_id", sort=False)
            if df_price is not None and not df_price.empty
            else None
        )

        # 計算每支股票的外資、投信、合計淨買超 + 連續買超天數 + 買超佔量比
        rows = []
        for sid in stock_ids:
            if sid not in inst_grouped.groups:
                rows.append(
                    {
                        "stock_id": sid,
                        "foreign_net": 0,
                        "trust_net": 0,
                        "total_net": 0,
                        "consec_buy_days": 0,
                        "buy_vol_ratio": 0.0,
                    }
                )
                continue

            stock_inst = inst_grouped.get_group(sid)
            foreign_data = stock_inst[stock_inst["name"].str.contains("外資", na=False)]
            trust_data = stock_inst[stock_inst["name"].str.contains("投信", na=False)]
            total_net = stock_inst["net"].sum()

            # 連續買超天數：從最新日期往回數，三大法人合計淨買超 > 0 的連續天數
            daily_net = stock_inst.groupby("date")["net"].sum().sort_index(ascending=False)
            consec_days = 0
            for net_val in daily_net.values:
                if net_val > 0:
                    consec_days += 1
                else:
                    break

            # 買超佔成交量比例：合計淨買超 / 最新日成交量
            buy_vol_ratio = 0.0
            if price_grouped is not None and sid in price_grouped.groups:
                stock_price = price_grouped.get_group(sid)
                latest_vol = stock_price.loc[stock_price["date"].idxmax(), "volume"]
                if latest_vol > 0:
                    buy_vol_ratio = total_net / latest_vol

            rows.append(
                {
                    "stock_id": sid,
                    "foreign_net": foreign_data["net"].sum() if not foreign_data.empty else 0,
                    "trust_net": trust_data["net"].sum() if not trust_data.empty else 0,
                    "total_net": total_net,
                    "consec_buy_days": consec_days,
                    "buy_vol_ratio": buy_vol_ratio,
                }
            )

        df = pd.DataFrame(rows)

        # 用排名百分位，分數自然分散在 0~1
        foreign_rank = df["foreign_net"].rank(pct=True)
        trust_rank = df["trust_net"].rank(pct=True)
        total_rank = df["total_net"].rank(pct=True)
        consec_rank = df["consec_buy_days"].rank(pct=True)
        buy_vol_rank = df["buy_vol_ratio"].rank(pct=True)

        # 外資 30% + 投信 20% + 合計 20% + 連續買超 15% + 買超佔量 15%
        df["chip_score"] = (
            foreign_rank * 0.30 + trust_rank * 0.20 + total_rank * 0.20 + consec_rank * 0.15 + buy_vol_rank * 0.15
        )

        return df[["stock_id", "chip_score"]]

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """從月營收資料計算基本面分數（YoY 70% + MoM 30%，排名百分位）。"""
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        if rev.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        # 用排名百分位，讓分數自然分散在 0~1
        yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
        mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)

        # YoY 權重 70% + MoM 權重 30%
        rev["fundamental_score"] = yoy_rank * 0.70 + mom_rank * 0.30

        # 包含所有 stock_ids，無資料的用 NaN（外層 fillna 處理）
        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    # ------------------------------------------------------------------ #
    #  Stage 3.5: 風險過濾（子類覆寫）
    # ------------------------------------------------------------------ #

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """風險過濾（基底類別不做任何過濾，子類覆寫）。"""
        return scored

    # ------------------------------------------------------------------ #
    #  共用風險過濾 helpers（子類呼叫，不須覆寫整個方法）
    # ------------------------------------------------------------------ #

    def _apply_atr_risk_filter(
        self, scored: pd.DataFrame, df_price: pd.DataFrame, percentile: int = 80
    ) -> pd.DataFrame:
        """ATR-based 風險過濾：ATR(14)/close > N-th percentile 的股票剔除。"""
        if scored.empty or df_price.empty:
            return scored

        grouped = df_price.sort_values("date").groupby("stock_id", sort=False)
        atr_ratios = []
        for sid in scored["stock_id"].tolist():
            stock_data = grouped.get_group(sid) if sid in grouped.groups else pd.DataFrame()
            atr = _calc_atr14(stock_data)
            current_close = stock_data["close"].values[-1] if not stock_data.empty else 1.0
            ratio = atr / current_close if current_close > 0 else 0.0
            atr_ratios.append({"stock_id": sid, "atr_ratio": ratio})

        df_atr = pd.DataFrame(atr_ratios)
        threshold = df_atr["atr_ratio"].quantile(percentile / 100)
        high_vol_ids = df_atr[df_atr["atr_ratio"] > threshold]["stock_id"].tolist()

        before_count = len(scored)
        scored = scored[~scored["stock_id"].isin(high_vol_ids)].copy()
        removed = before_count - len(scored)
        if removed > 0:
            logger.info("Stage 3.5: ATR 風險過濾剔除 %d 支高波動股", removed)
        return scored

    def _apply_vol_risk_filter(
        self,
        scored: pd.DataFrame,
        df_price: pd.DataFrame,
        percentile: int,
        window: int = 20,
        annualize: bool = False,
    ) -> pd.DataFrame:
        """波動率-based 風險過濾：N 日波動率 > M-th percentile 的股票剔除。

        Args:
            percentile: 剔除閾值（80 表示剔除波動率超過第 80 百分位數的股票）
            window: 計算波動率的回溯天數（預設 20）
            annualize: 是否年化（乘以 sqrt(252)）
        """
        if scored.empty or df_price.empty:
            return scored

        grouped = df_price.sort_values("date").groupby("stock_id", sort=False)
        vol_data = []
        for sid in scored["stock_id"].tolist():
            stock_data = grouped.get_group(sid) if sid in grouped.groups else pd.DataFrame()
            if len(stock_data) < 10:
                vol_data.append({"stock_id": sid, "vol": 0.0})
                continue

            closes = (
                stock_data["close"].values[-(window + 1) :]
                if len(stock_data) >= window + 1
                else stock_data["close"].values
            )
            returns = np.diff(closes) / closes[:-1]
            vol = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
            if annualize:
                vol = vol * np.sqrt(252)
            vol_data.append({"stock_id": sid, "vol": vol})

        df_vol = pd.DataFrame(vol_data)
        threshold = df_vol["vol"].quantile(percentile / 100)
        high_vol_ids = df_vol[df_vol["vol"] > threshold]["stock_id"].tolist()

        before_count = len(scored)
        scored = scored[~scored["stock_id"].isin(high_vol_ids)].copy()
        removed = before_count - len(scored)
        if removed > 0:
            logger.info("Stage 3.5: 波動率風險過濾剔除 %d 支高波動股", removed)
        return scored

    def _reload_valuation(self, stock_ids: list[str]) -> None:
        """重新載入估值資料（補抓後 DB 已更新）。供 ValueScanner / DividendScanner 呼叫。"""
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)
        with get_session() as session:
            rows = session.execute(
                select(
                    StockValuation.stock_id,
                    StockValuation.date,
                    StockValuation.pe_ratio,
                    StockValuation.pb_ratio,
                    StockValuation.dividend_yield,
                )
                .where(StockValuation.date >= cutoff)
                .where(StockValuation.stock_id.in_(stock_ids))
            ).all()
            self._df_valuation = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "pe_ratio", "pb_ratio", "dividend_yield"],
            )

    def _maybe_sync_valuation(self) -> None:
        """Stage 0.5：估值資料覆蓋不足時，自動從 TWSE/TPEX 補抓全市場估值。
        供 ValueScanner / DividendScanner 的 run() 呼叫。
        """
        try:
            from sqlalchemy import func as sa_func

            with get_session() as session:
                val_count = session.execute(select(sa_func.count(sa_func.distinct(StockValuation.stock_id)))).scalar()
            if not val_count or val_count < 500:
                logger.info(
                    "Stage 0.5: 估值資料僅 %d 支，自動從 TWSE/TPEX 同步全市場估值...",
                    val_count or 0,
                )
                from src.data.pipeline import sync_valuation_all_market

                val_synced = sync_valuation_all_market()
                logger.info("Stage 0.5: 全市場估值同步完成，新增 %d 筆", val_synced)
        except Exception:
            logger.warning("Stage 0.5: 全市場估值自動同步失敗，使用既有資料繼續")

    def _compute_momentum_style_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """動能風格技術面 5 因子：5日動能 + 10日動能 + 20日突破 + 量比 + 成交量加速。
        供 MomentumScanner 與 GrowthScanner 共用。
        """
        results = []
        # 預先 groupby 一次（O(N)），避免迴圈中反覆 boolean filter（O(N²)）
        grouped = df_price.sort_values("date").groupby("stock_id", sort=False)

        for sid in stock_ids:
            if sid not in grouped.groups:
                results.append({"stock_id": sid, "technical_score": 0.5})
                continue
            stock_data = grouped.get_group(sid)
            if len(stock_data) < 3:
                results.append({"stock_id": sid, "technical_score": 0.5})
                continue

            closes = stock_data["close"].values
            volumes = stock_data["volume"].values.astype(float)

            score = 0.0
            n_factors = 0

            # 1) 5 日動能
            if len(closes) >= 6:
                ret_5d = (closes[-1] - closes[-6]) / closes[-6]
                score += max(0.0, min(1.0, 0.5 + ret_5d * 5))
                n_factors += 1

            # 2) 10 日動能
            if len(closes) >= 11:
                ret_10d = (closes[-1] - closes[-11]) / closes[-11]
                score += max(0.0, min(1.0, 0.5 + ret_10d * 5))
                n_factors += 1

            # 3) 20 日突破：close / max(close[-20:])
            if len(closes) >= 20:
                max_20 = closes[-20:].max()
                if max_20 > 0:
                    score += closes[-1] / max_20
                else:
                    score += 0.5
                n_factors += 1

            # 4) 量比：volume[-1] / mean(volume[-20:])
            if len(volumes) >= 20:
                avg_vol_20 = volumes[-20:].mean()
                if avg_vol_20 > 0:
                    ratio = min(2.0, volumes[-1] / avg_vol_20)
                    score += ratio / 2.0
                else:
                    score += 0.5
                n_factors += 1

            # 5) 成交量加速：mean(vol[-3:]) / mean(vol[-10:])
            if len(volumes) >= 10:
                avg_vol_3 = volumes[-3:].mean()
                avg_vol_10 = volumes[-10:].mean()
                if avg_vol_10 > 0:
                    ratio = min(2.0, avg_vol_3 / avg_vol_10)
                    score += ratio / 2.0
                else:
                    score += 0.5
                n_factors += 1

            tech_score = score / max(n_factors, 1)
            results.append({"stock_id": sid, "technical_score": tech_score})

        return pd.DataFrame(results)

    # ------------------------------------------------------------------ #
    #  Stage 4: 排名 + 產業標籤
    # ------------------------------------------------------------------ #

    def _rank_and_enrich(self, scored: pd.DataFrame) -> pd.DataFrame:
        """排名並加上產業 / 股票名稱。"""
        scored = scored.sort_values("composite_score", ascending=False).reset_index(drop=True)
        scored["rank"] = range(1, len(scored) + 1)

        # 從 DB 取 StockInfo
        stock_ids = scored["stock_id"].tolist()
        with get_session() as session:
            rows = session.execute(
                select(StockInfo.stock_id, StockInfo.stock_name, StockInfo.industry_category).where(
                    StockInfo.stock_id.in_(stock_ids)
                )
            ).all()
            info_df = pd.DataFrame(rows, columns=["stock_id", "stock_name", "industry_category"])

        if not info_df.empty:
            scored = scored.merge(info_df, on="stock_id", how="left")
        else:
            scored["stock_name"] = ""
            scored["industry_category"] = ""

        scored["stock_name"] = scored["stock_name"].fillna("")
        scored["industry_category"] = scored["industry_category"].fillna("")

        # 只保留需要的欄位
        keep_cols = [
            "rank",
            "stock_id",
            "stock_name",
            "close",
            "volume",
            "composite_score",
            "technical_score",
            "chip_score",
            "chip_tier",
            "fundamental_score",
            "news_score",
            "sector_bonus",
            "industry_category",
            "momentum",
            "inst_net",
            "entry_price",
            "stop_loss",
            "take_profit",
            "entry_trigger",
            "valid_until",
        ]
        return scored[[c for c in keep_cols if c in scored.columns]]

    def _compute_sector_summary(self, rankings: pd.DataFrame) -> pd.DataFrame:
        """統計推薦結果的產業分布。"""
        if rankings.empty or "industry_category" not in rankings.columns:
            return pd.DataFrame()

        top_n = rankings.head(self.top_n_results)
        summary = (
            top_n.groupby("industry_category")
            .agg(
                count=("stock_id", "count"),
                avg_score=("composite_score", "mean"),
            )
            .reset_index()
            .sort_values("count", ascending=False)
        )
        summary.columns = ["industry", "count", "avg_score"]
        return summary

    def _load_broker_data(self, stock_ids: list[str]) -> pd.DataFrame:
        """從 DB 載入最近 7 天的分點交易資料。

        共用方法（MarketScanner 基底），供 MomentumScanner / SwingScanner /
        ValueScanner / GrowthScanner 的 _compute_chip_scores() 呼叫。
        若表不存在或無資料則回傳空 DataFrame，呼叫端自動降級。
        """
        cutoff = date.today() - timedelta(days=7)
        try:
            with get_session() as session:
                rows = session.execute(
                    select(
                        BrokerTrade.stock_id,
                        BrokerTrade.date,
                        BrokerTrade.broker_id,
                        BrokerTrade.buy,
                        BrokerTrade.sell,
                    ).where(
                        BrokerTrade.stock_id.in_(stock_ids),
                        BrokerTrade.date >= cutoff,
                    )
                ).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=["stock_id", "date", "broker_id", "buy", "sell"])
        except Exception:
            return pd.DataFrame()


# ====================================================================== #
#  MomentumScanner — 短線動能模式
# ====================================================================== #


class MomentumScanner(MarketScanner):
    """短線動能掃描器（1~10 天）。

    粗篩：動能 + 流動性
    細評：技術面 45% + 籌碼面 45% + 基本面 10%
    風險過濾：ATR ratio > 80th percentile 剔除
    """

    mode_name = "momentum"
    _auto_sync_broker = True  # Stage 2.5 自動補抓候選股分點資料
    _revenue_months = 4  # 載入 4 個月營收，啟用「本月 YoY - 3 個月前 YoY」加速度輕微加成
    _COARSE_WEIGHTS: dict[str, float] = {"vol_rank": 0.30, "inst_rank": 0.40, "mom_rank": 0.30}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 25)
        super().__init__(**kwargs)

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """動能模式粗篩：基本過濾 + 流動性 + 動能/法人/成交量加權。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 額外流動性過濾：成交量 > 20 日均量 × 0.5
        latest_date = df_price["date"].max()
        vol_mean = df_price.groupby("stock_id")["volume"].apply(lambda s: s.tail(20).mean()).reset_index()
        vol_mean.columns = ["stock_id", "avg_vol_20"]
        filtered = filtered.merge(vol_mean, on="stock_id", how="left")
        filtered["avg_vol_20"] = filtered["avg_vol_20"].fillna(0)
        filtered = filtered[filtered["volume"] > filtered["avg_vol_20"] * 0.5].copy()

        if filtered.empty:
            return pd.DataFrame()

        # 1) 成交量排名
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 2) 法人 5 日累積淨買超排名（比單日穩定，過濾換手雜訊）
        if not df_inst.empty:
            inst_dates = sorted(df_inst["date"].unique())
            recent_5_dates = inst_dates[-5:]
            inst_5d = df_inst[df_inst["date"].isin(recent_5_dates)]
            inst_net = inst_5d.groupby("stock_id")["net"].sum().reset_index()
            inst_net.columns = ["stock_id", "inst_net"]
            filtered = filtered.merge(inst_net, on="stock_id", how="left")
            filtered["inst_net"] = filtered["inst_net"].fillna(0)
            filtered["inst_rank"] = filtered["inst_net"].rank(pct=True)
        else:
            filtered["inst_net"] = 0
            filtered["inst_rank"] = 0.5

        # 3) 短期動能（5 日報酬，比單日漲跌更穩定）
        dates = sorted(df_price["date"].unique())
        ref_date = dates[-5] if len(dates) >= 5 else (dates[0] if len(dates) >= 2 else None)
        if ref_date is not None:
            ref = df_price[df_price["date"] == ref_date][["stock_id", "close"]].rename(columns={"close": "ref_close"})
            filtered = filtered.merge(ref, on="stock_id", how="left")
            filtered["momentum"] = (
                (filtered["close"] - filtered["ref_close"]) / filtered["ref_close"].replace(0, float("nan"))
            ).fillna(0)
            filtered["mom_rank"] = filtered["momentum"].rank(pct=True)
        else:
            filtered["momentum"] = 0
            filtered["mom_rank"] = 0.5

        # 粗篩綜合分（依 _COARSE_WEIGHTS 類別屬性動態計算）
        filtered["coarse_score"] = sum(
            filtered[k] * v for k, v in self._COARSE_WEIGHTS.items() if k in filtered.columns
        )

        filtered = filtered.nlargest(self._effective_top_n(len(filtered)), "coarse_score")
        return filtered

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """動能模式技術面 5 因子（委派至 base class 共用實作）。"""
        return self._compute_momentum_style_technical_scores(stock_ids, df_price)

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """動能模式籌碼面：外資連續買超 + 買超佔量比 + 三大法人合計 + 券資比 + 大戶持股 + 借券 + 分點（有資料時）。

        權重組合（由資料可用性決定）：
        - 8 因子（含智慧分點）: 外資18%+量比16%+法人16%+券資比10%+大戶12%+借券7%+分點HHI11%+智慧分點10%
        - 7 因子（含分點）: 外資20%+量比18%+法人18%+券資比11%+大戶13%+借券8%+分點12%
        - 6 因子（含借券）: 外資 22% + 量比 20% + 法人 20% + 券資比 13% + 大戶 15% + 借券(逆) 10%
        - 5 因子（含大戶持股，無借券）: 外資 25% + 量比 22% + 法人 22% + 券資比 15% + 大戶 16%
        - 4 因子（含券資比，無大戶/借券）: 外資 30% + 量比 25% + 法人 25% + 券資比 20%
        - 3 因子（基本）: 外資 40% + 量比 30% + 法人 30%

        回傳欄位：stock_id, chip_score, chip_tier（因子層級字串，如 "8F"/"3F"/"N/A"）
        """
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids), "chip_tier": "N/A"})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
        inst_grouped = inst_filtered.groupby("stock_id", sort=False)
        price_grouped = (
            df_price[df_price["stock_id"].isin(stock_ids)].groupby("stock_id", sort=False)
            if df_price is not None and not df_price.empty
            else None
        )
        rows = []
        for sid in stock_ids:
            if sid not in inst_grouped.groups:
                rows.append({"stock_id": sid, "consec_foreign_days": 0, "buy_vol_ratio": 0.0, "total_net": 0})
                continue

            stock_inst = inst_grouped.get_group(sid)
            # 外資連續買超天數
            foreign_data = stock_inst[stock_inst["name"].str.contains("外資", na=False)]
            consec_foreign = 0
            if not foreign_data.empty:
                daily_foreign = foreign_data.groupby("date")["net"].sum().sort_index(ascending=False)
                for val in daily_foreign.values:
                    if val > 0:
                        consec_foreign += 1
                    else:
                        break

            # 法人買超/成交量比例
            total_net = stock_inst["net"].sum()
            buy_vol_ratio = 0.0
            if price_grouped is not None and sid in price_grouped.groups:
                stock_price = price_grouped.get_group(sid)
                latest_vol = stock_price.loc[stock_price["date"].idxmax(), "volume"]
                if latest_vol > 0:
                    buy_vol_ratio = total_net / latest_vol

            rows.append(
                {
                    "stock_id": sid,
                    "consec_foreign_days": consec_foreign,
                    "buy_vol_ratio": buy_vol_ratio,
                    "total_net": total_net,
                }
            )

        df = pd.DataFrame(rows)

        consec_rank = df["consec_foreign_days"].rank(pct=True)
        bvr_rank = df["buy_vol_ratio"].rank(pct=True)
        total_rank = df["total_net"].rank(pct=True)

        # ── 大戶持股因子（從 DB 查詢最近 2 週資料）──────────────────
        df_whale = self._load_holding_data(stock_ids)
        whale_df = compute_whale_score(df_whale)
        has_whale = not whale_df.empty
        if has_whale:
            df = df.merge(whale_df, on="stock_id", how="left")
            df["whale_percent"] = df["whale_percent"].fillna(
                df["whale_percent"].median() if not df["whale_percent"].isna().all() else 0.0
            )
            df["whale_change"] = df["whale_change"].fillna(0.0)
            whale_pct_rank = df["whale_percent"].rank(pct=True)
            whale_chg_rank = df["whale_change"].rank(pct=True)
            # 大戶持股綜合排名：持股比例 60% + 週變化 40%
            whale_rank = whale_pct_rank * 0.60 + whale_chg_rank * 0.40

        # ── 借券賣出因子（空頭壓力逆向評分）────────────────────────
        df_sbl_raw = self._load_sbl_data(stock_ids)
        sbl_df = compute_sbl_score(df_sbl_raw)
        has_sbl = not sbl_df.empty
        if has_sbl:
            df = df.merge(sbl_df[["stock_id", "sbl_balance"]], on="stock_id", how="left")
            df["sbl_balance"] = df["sbl_balance"].fillna(
                df["sbl_balance"].median() if not df["sbl_balance"].isna().all() else 0.0
            )
            # 逆向評分：借券餘額低 → 空頭壓力小 → 評分高
            sbl_rank = 1.0 - df["sbl_balance"].rank(pct=True)

        # ── 分點因子（主力集中度 HHI + 連續進場天數）────────────────
        df_broker_raw = self._load_broker_data(stock_ids)
        broker_df = compute_broker_score(df_broker_raw)
        has_broker = not broker_df.empty
        if has_broker:
            df = df.merge(broker_df, on="stock_id", how="left")
            df["broker_concentration"] = df["broker_concentration"].fillna(0.0)
            df["broker_consecutive_days"] = df["broker_consecutive_days"].fillna(0.0)
            broker_conc_rank = df["broker_concentration"].rank(pct=True)
            broker_consec_rank = df["broker_consecutive_days"].rank(pct=True)
            broker_rank = broker_conc_rank * 0.60 + broker_consec_rank * 0.40

        # ── 智慧分點因子（120 天歷史勝率 + 蓄積型分點，需 buy_price / sell_price）──
        _close_map: dict[str, float] = {}
        if df_price is not None and not df_price.empty:
            _close_map = df_price.sort_values("date").groupby("stock_id")["close"].last().to_dict()
        df_broker_ext = self._load_broker_data_extended(stock_ids)
        has_smart_broker = False
        if not df_broker_ext.empty:
            smart_df = compute_smart_broker_score(df_broker_ext, _close_map)
            has_smart_broker = not smart_df.empty and float(smart_df["smart_broker_factor"].sum()) > 0
            if has_smart_broker:
                df = df.merge(smart_df[["stock_id", "smart_broker_factor"]], on="stock_id", how="left")
                df["smart_broker_factor"] = df["smart_broker_factor"].fillna(0.0)
                smart_broker_rank = df["smart_broker_factor"].rank(pct=True)

        # ── 融資融券因子 ──────────────────────────────────────────────
        has_margin = df_margin is not None and not df_margin.empty
        if has_margin:
            margin_latest = df_margin[df_margin["date"] == df_margin["date"].max()]
            margin_data = margin_latest[margin_latest["stock_id"].isin(stock_ids)][
                ["stock_id", "margin_balance", "short_balance"]
            ].copy()
            if not margin_data.empty:
                margin_data["short_margin_ratio"] = margin_data.apply(
                    lambda r: r["short_balance"] / r["margin_balance"] if r["margin_balance"] > 0 else 0.0,
                    axis=1,
                )
                df = df.merge(margin_data[["stock_id", "short_margin_ratio"]], on="stock_id", how="left")
                df["short_margin_ratio"] = df["short_margin_ratio"].fillna(0.0)
                smr_rank = df["short_margin_ratio"].rank(pct=True)
            else:
                has_margin = False

        # ── 加權組合 ─────────────────────────────────────────────────
        if has_smart_broker and has_broker and has_sbl and has_margin and has_whale:
            # 8 因子：外資18%+量比16%+法人16%+券資比10%+大戶12%+借券7%+分點HHI11%+智慧分點10%
            df["chip_score"] = (
                consec_rank * 0.18
                + bvr_rank * 0.16
                + total_rank * 0.16
                + smr_rank * 0.10
                + whale_rank * 0.12
                + sbl_rank * 0.07
                + broker_rank * 0.11
                + smart_broker_rank * 0.10
            )
            chip_tier = "8F"
        elif has_broker and has_sbl and has_margin and has_whale:
            # 7 因子：外資20%+量比18%+法人18%+券資比11%+大戶13%+借券8%+分點12%
            df["chip_score"] = (
                consec_rank * 0.20
                + bvr_rank * 0.18
                + total_rank * 0.18
                + smr_rank * 0.11
                + whale_rank * 0.13
                + sbl_rank * 0.08
                + broker_rank * 0.12
            )
            chip_tier = "7F"
        elif has_broker and has_sbl and has_margin:
            # 6 因子（有分點、無大戶）：外資22%+量比20%+法人20%+券資比14%+借券12%+分點12%
            df["chip_score"] = (
                consec_rank * 0.22
                + bvr_rank * 0.20
                + total_rank * 0.20
                + smr_rank * 0.14
                + sbl_rank * 0.12
                + broker_rank * 0.12
            )
            chip_tier = "6F"
        elif has_broker and has_sbl:
            # 5 因子（有分點+借券、無大戶/融資券）：外資28%+量比22%+法人22%+借券14%+分點14%
            df["chip_score"] = (
                consec_rank * 0.28 + bvr_rank * 0.22 + total_rank * 0.22 + sbl_rank * 0.14 + broker_rank * 0.14
            )
            chip_tier = "5F"
        elif has_broker:
            # 4 因子（僅分點）：外資32%+量比24%+法人24%+分點20%
            df["chip_score"] = consec_rank * 0.32 + bvr_rank * 0.24 + total_rank * 0.24 + broker_rank * 0.20
            chip_tier = "4F"
        elif has_sbl and has_margin and has_whale:
            # 6 因子：外資 22% + 量比 20% + 法人 20% + 券資比 13% + 大戶 15% + 借券(逆) 10%
            df["chip_score"] = (
                consec_rank * 0.22
                + bvr_rank * 0.20
                + total_rank * 0.20
                + smr_rank * 0.13
                + whale_rank * 0.15
                + sbl_rank * 0.10
            )
            chip_tier = "6F"
        elif has_sbl and has_margin:
            # 5 因子（有借券、無大戶）：外資 25% + 量比 22% + 法人 22% + 券資比 16% + 借券(逆) 15%
            df["chip_score"] = (
                consec_rank * 0.25 + bvr_rank * 0.22 + total_rank * 0.22 + smr_rank * 0.16 + sbl_rank * 0.15
            )
            chip_tier = "5F"
        elif has_sbl and has_whale:
            # 5 因子（有借券、無券資比）：外資 28% + 量比 20% + 法人 20% + 大戶 22% + 借券(逆) 10%
            df["chip_score"] = (
                consec_rank * 0.28 + bvr_rank * 0.20 + total_rank * 0.20 + whale_rank * 0.22 + sbl_rank * 0.10
            )
            chip_tier = "5F"
        elif has_sbl:
            # 4 因子（僅借券）：外資 35% + 量比 25% + 法人 25% + 借券(逆) 15%
            df["chip_score"] = consec_rank * 0.35 + bvr_rank * 0.25 + total_rank * 0.25 + sbl_rank * 0.15
            chip_tier = "4F"
        elif has_margin and has_whale:
            # 5 因子：外資 25% + 量比 22% + 法人 22% + 券資比 15% + 大戶 16%
            df["chip_score"] = (
                consec_rank * 0.25 + bvr_rank * 0.22 + total_rank * 0.22 + smr_rank * 0.15 + whale_rank * 0.16
            )
            chip_tier = "5F"
        elif has_margin:
            # 4 因子：外資 30% + 量比 25% + 法人 25% + 券資比 20%
            df["chip_score"] = consec_rank * 0.30 + bvr_rank * 0.25 + total_rank * 0.25 + smr_rank * 0.20
            chip_tier = "4F"
        elif has_whale:
            # 4 因子：外資 35% + 量比 25% + 法人 25% + 大戶 15%
            df["chip_score"] = consec_rank * 0.35 + bvr_rank * 0.25 + total_rank * 0.25 + whale_rank * 0.15
            chip_tier = "4F"
        else:
            # 3 因子：外資 40% + 量比 30% + 法人 30%
            df["chip_score"] = consec_rank * 0.40 + bvr_rank * 0.30 + total_rank * 0.30
            chip_tier = "3F"

        df["chip_tier"] = chip_tier
        return df[["stock_id", "chip_score", "chip_tier"]]

    def _load_sbl_data(self, stock_ids: list[str]) -> pd.DataFrame:
        """從 DB 載入最近 5 天的借券賣出彙總資料。

        若表不存在或無資料則回傳空 DataFrame，_compute_chip_scores 會自動降級。
        """
        cutoff = date.today() - timedelta(days=5)
        try:
            with get_session() as session:
                rows = session.execute(
                    select(
                        SecuritiesLending.stock_id,
                        SecuritiesLending.date,
                        SecuritiesLending.sbl_balance,
                    ).where(
                        SecuritiesLending.stock_id.in_(stock_ids),
                        SecuritiesLending.date >= cutoff,
                    )
                ).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=["stock_id", "date", "sbl_balance"])
        except Exception:
            return pd.DataFrame()

    def _load_holding_data(self, stock_ids: list[str]) -> pd.DataFrame:
        """從 DB 載入最近 2 週的大戶持股分級資料。

        若表不存在或無資料則回傳空 DataFrame，_compute_chip_scores 會自動降級。
        """
        cutoff = date.today() - timedelta(days=21)
        try:
            with get_session() as session:
                rows = session.execute(
                    select(
                        HoldingDistribution.stock_id,
                        HoldingDistribution.date,
                        HoldingDistribution.level,
                        HoldingDistribution.percent,
                    ).where(
                        HoldingDistribution.stock_id.in_(stock_ids),
                        HoldingDistribution.date >= cutoff,
                    )
                ).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=["stock_id", "date", "level", "percent"])
        except Exception:
            return pd.DataFrame()

    def _load_broker_data_extended(
        self,
        stock_ids: list[str],
        days: int = 365,
        min_trading_days: int = 20,
    ) -> pd.DataFrame:
        """從 DB 載入所有可用的分點交易資料（含 buy_price / sell_price）。

        查詢窗口改為 days=365，充分利用 daily sync 累積的歷史資料。
        min_trading_days：每支股票至少需有 N 個交易日資料，否則排除（避免假信號）。

        計算僅針對 Stage 2 粗篩後的候選股（~150 支），不涉及全市場。
        若表不存在或無資料則回傳空 DataFrame（呼叫端自動降級）。

        資料累積說明：
          - morning-routine Step 2 每日同步 watchlist 分點資料（5日）
          - 連續執行 20 天後即可觸發 Smart Broker 8F 計算
          - 資料越豐富（60~120 天），勝率/PF 估算越準確

        均價代理策略（方案 B）：
          - DJ 端點不提供 buy_price / sell_price，欄位存 NULL
          - 本函數自動以 DailyPrice.close 填補 NULL 均價（同日收盤價）
          - win_rate / PF 的意義：衡量分點「是否在漲前買、跌前賣」的擇時能力
          - 所有分點使用相同日收盤價，失去「執行品質」訊號，但「擇時」信號仍有效
        """
        cutoff = date.today() - timedelta(days=days)
        try:
            with get_session() as session:
                rows = session.execute(
                    select(
                        BrokerTrade.stock_id,
                        BrokerTrade.date,
                        BrokerTrade.broker_id,
                        BrokerTrade.buy,
                        BrokerTrade.sell,
                        BrokerTrade.buy_price,
                        BrokerTrade.sell_price,
                    ).where(
                        BrokerTrade.stock_id.in_(stock_ids),
                        BrokerTrade.date >= cutoff,
                    )
                ).all()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "broker_id", "buy", "sell", "buy_price", "sell_price"],
            )
            # ── 均價代理：以 DailyPrice.close 填補 NULL buy_price / sell_price ──
            # DJ 端點無均價欄位，改用收盤價作為代理，使 Smart Broker 8F 得以啟用。
            if df["buy_price"].isna().any() or df["sell_price"].isna().any():
                try:
                    with get_session() as session:
                        price_rows = session.execute(
                            select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                                DailyPrice.stock_id.in_(stock_ids),
                                DailyPrice.date >= cutoff,
                            )
                        ).all()
                    if price_rows:
                        price_df = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])
                        df = df.merge(price_df, on=["stock_id", "date"], how="left")
                        df["buy_price"] = df["buy_price"].astype("float64").fillna(df["close"])
                        df["sell_price"] = df["sell_price"].astype("float64").fillna(df["close"])
                        df = df.drop(columns=["close"])
                except Exception:
                    pass  # 無法載入收盤價時保持原始 NULL，系統降回 7F
            # 過濾掉歷史資料不足的股票，避免以少量資料誤判分點行為
            if min_trading_days > 0 and not df.empty:
                day_counts = df.groupby("stock_id")["date"].nunique()
                valid_stocks = day_counts[day_counts >= min_trading_days].index
                df = df[df["stock_id"].isin(valid_stocks)]
            return df
        except Exception:
            return pd.DataFrame()

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """動能模式基本面：四階梯短線爆發濾網（此 10% 權重不評長線品質，只捕捉短線催化劑）。

        Tier 1 (0.85)：MoM > 0 且 YoY > 0 且 YoY > yoy_3m_ago（月營收雙增 + YoY 近期創高）
        Tier 2 (0.72)：YoY > 0 且 YoY > yoy_3m_ago（YoY 正且加速）
        Tier 3 (0.55)：YoY > 0（YoY 正但未加速，或無加速度資料）
        Tier 4 (0.30)：YoY <= 0（YoY 衰退）
        無資料 fallback：0.50（中性）
        """
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        # 預先 groupby 一次（O(N)），避免迴圈中反覆 boolean filter（O(N²)）
        rev_grouped = df_revenue.groupby("stock_id", sort=False)
        result_rows = []
        for sid in stock_ids:
            if sid not in rev_grouped.groups:
                result_rows.append({"stock_id": sid, "fundamental_score": 0.5})
                continue

            rev = rev_grouped.get_group(sid)
            yoy = rev.iloc[0].get("yoy_growth", None)
            if yoy is None or pd.isna(yoy):
                result_rows.append({"stock_id": sid, "fundamental_score": 0.5})
                continue

            mom = rev.iloc[0].get("mom_growth", None)
            yoy_3m = rev.iloc[0].get("yoy_3m_ago", None)

            has_mom_pos = mom is not None and not pd.isna(mom) and float(mom) > 0
            has_accel = yoy_3m is not None and not pd.isna(yoy_3m) and float(yoy) > float(yoy_3m)

            if float(yoy) > 0 and has_mom_pos and has_accel:
                score = 0.85  # Tier 1: 月營收雙增 + YoY 近期創高
            elif float(yoy) > 0 and has_accel:
                score = 0.72  # Tier 2: YoY 正且加速（MoM 未必正）
            elif float(yoy) > 0:
                score = 0.55  # Tier 3: YoY 正但未加速
            else:
                score = 0.30  # Tier 4: YoY 衰退

            result_rows.append({"stock_id": sid, "fundamental_score": score})

        return pd.DataFrame(result_rows)

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """動能模式風險過濾：ATR(14)/close > 80th percentile 剔除。"""
        return self._apply_atr_risk_filter(scored, df_price, percentile=80)


# ====================================================================== #
#  SwingScanner — 中期波段模式
# ====================================================================== #


class SwingScanner(MarketScanner):
    """中期波段掃描器（1~3 個月）。

    粗篩：趨勢（close > SMA60）+ 基本面
    細評：技術面 30% + 籌碼面 30% + 基本面 40%
    風險過濾：年化波動率 > 85th percentile 剔除
    """

    mode_name = "swing"
    _COARSE_WEIGHTS: dict[str, float] = {"inst_rank": 0.40, "trend_rank": 0.30, "vol_rank": 0.30}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 80)
        kwargs.setdefault("universe_config", UniverseConfig(volume_ratio_min=1.2))
        super().__init__(**kwargs)

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：swing 模式載入 2 個月營收資料（算加速度）。含 UniverseFilter Stage 0.5。"""
        universe_ids = self._get_universe_ids()
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        with get_session() as session:
            price_query = select(
                DailyPrice.stock_id,
                DailyPrice.date,
                DailyPrice.open,
                DailyPrice.high,
                DailyPrice.low,
                DailyPrice.close,
                DailyPrice.volume,
                DailyPrice.turnover,
            ).where(DailyPrice.date >= cutoff)
            if universe_ids:
                price_query = price_query.where(DailyPrice.stock_id.in_(universe_ids))
            rows = session.execute(price_query).all()
            df_price = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "open", "high", "low", "close", "volume", "turnover"],
            )

            inst_query = select(
                InstitutionalInvestor.stock_id,
                InstitutionalInvestor.date,
                InstitutionalInvestor.name,
                InstitutionalInvestor.net,
            ).where(InstitutionalInvestor.date >= cutoff)
            if universe_ids:
                inst_query = inst_query.where(InstitutionalInvestor.stock_id.in_(universe_ids))
            rows = session.execute(inst_query).all()
            df_inst = pd.DataFrame(rows, columns=["stock_id", "date", "name", "net"])

            margin_query = select(
                MarginTrading.stock_id,
                MarginTrading.date,
                MarginTrading.margin_balance,
                MarginTrading.short_balance,
            ).where(MarginTrading.date >= cutoff)
            if universe_ids:
                margin_query = margin_query.where(MarginTrading.stock_id.in_(universe_ids))
            rows = session.execute(margin_query).all()
            df_margin = pd.DataFrame(rows, columns=["stock_id", "date", "margin_balance", "short_balance"])

        # 載入 2 個月營收（含上月，用於計算加速度）
        df_revenue = self._load_revenue_data(stock_ids=universe_ids if universe_ids else None, months=2)

        return df_price, df_inst, df_margin, df_revenue

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """波段模式粗篩：基本過濾 + close > SMA60 + 法人累積/趨勢/量加權。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 額外條件：close > SMA60（全量向量化，取代逐股 for 迴圈）
        # 對 df_price 全量做 rolling(60)，一次計算所有股票，再取各股最新值
        price_sorted = df_price[["stock_id", "date", "close"]].sort_values(["stock_id", "date"])
        sma60_rolling = price_sorted.groupby("stock_id")["close"].transform(
            lambda s: s.rolling(60, min_periods=60).mean()
        )
        # last() 取每支股票最後一列的 SMA60；NaN = 資料不足 60 天 → dropna 排除
        sma60_latest = price_sorted.assign(_sma60=sma60_rolling).groupby("stock_id")["_sma60"].last().dropna()

        if not sma60_latest.empty:
            filtered["sma60"] = filtered["stock_id"].map(sma60_latest)
            filtered = filtered[filtered["sma60"].notna() & (filtered["close"] > filtered["sma60"])].copy()
        # else: 全市場資料不足 60 天時跳過 SMA60 過濾（通常不會發生）

        if filtered.empty:
            return pd.DataFrame()

        # 1) 法人 20 日累積買超排名
        if not df_inst.empty:
            dates = sorted(df_inst["date"].unique())
            recent_20_dates = dates[-20:] if len(dates) >= 20 else dates
            inst_recent = df_inst[df_inst["date"].isin(recent_20_dates)]
            inst_cum = inst_recent.groupby("stock_id")["net"].sum().reset_index()
            inst_cum.columns = ["stock_id", "inst_net"]
            filtered = filtered.merge(inst_cum, on="stock_id", how="left")
            filtered["inst_net"] = filtered["inst_net"].fillna(0)
            filtered["inst_rank"] = filtered["inst_net"].rank(pct=True)
        else:
            filtered["inst_net"] = 0
            filtered["inst_rank"] = 0.5

        # 2) 趨勢強度：close / SMA60 的比值
        if "sma60" in filtered.columns:
            filtered["trend_strength"] = filtered["close"] / filtered["sma60"]
            filtered["trend_rank"] = filtered["trend_strength"].rank(pct=True)
        else:
            filtered["trend_rank"] = 0.5

        # 3) 成交量排名
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 動能欄位（用於 _rank_and_enrich 保留，5 日報酬比單日穩定）
        dates_all = sorted(df_price["date"].unique())
        ref_date_swing = dates_all[-5] if len(dates_all) >= 5 else (dates_all[0] if len(dates_all) >= 2 else None)
        if ref_date_swing is not None:
            ref = df_price[df_price["date"] == ref_date_swing][["stock_id", "close"]].rename(
                columns={"close": "ref_close"}
            )
            filtered = filtered.merge(ref, on="stock_id", how="left")
            filtered["momentum"] = (
                (filtered["close"] - filtered["ref_close"]) / filtered["ref_close"].replace(0, float("nan"))
            ).fillna(0)
        else:
            filtered["momentum"] = 0

        # 粗篩綜合分（依 _COARSE_WEIGHTS 類別屬性動態計算）
        filtered["coarse_score"] = sum(
            filtered[k] * v for k, v in self._COARSE_WEIGHTS.items() if k in filtered.columns
        )

        filtered = filtered.nlargest(self._effective_top_n(len(filtered)), "coarse_score")
        return filtered

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """波段模式技術面 4 因子：趨勢確認 + 均線排列 + 60日動能 + 量價齊揚。"""
        results = []
        grouped = df_price.sort_values("date").groupby("stock_id", sort=False)

        for sid in stock_ids:
            if sid not in grouped.groups:
                results.append({"stock_id": sid, "technical_score": 0.5})
                continue
            stock_data = grouped.get_group(sid)
            if len(stock_data) < 3:
                results.append({"stock_id": sid, "technical_score": 0.5})
                continue

            closes = stock_data["close"].values
            volumes = stock_data["volume"].values.astype(float)

            score = 0.0
            n_factors = 0

            # 1) 趨勢確認：close > SMA60
            if len(closes) >= 60:
                sma60 = closes[-60:].mean()
                score += 1.0 if closes[-1] > sma60 else 0.0
                n_factors += 1

            # 2) 均線排列：SMA20 > SMA60
            if len(closes) >= 60:
                sma20 = closes[-20:].mean()
                sma60 = closes[-60:].mean()
                score += 1.0 if sma20 > sma60 else 0.0
                n_factors += 1

            # 3) 60 日動能
            if len(closes) >= 61:
                ret_60d = (closes[-1] - closes[-61]) / closes[-61]
                score += max(0.0, min(1.0, 0.5 + ret_60d * 2))
                n_factors += 1

            # 4) 量價齊揚：近 20 日 price_chg > 0 且 vol_chg > 0 的天數比例
            if len(closes) >= 21 and len(volumes) >= 21:
                recent_closes = closes[-21:]
                recent_volumes = volumes[-21:]
                count = 0
                for i in range(1, 21):
                    price_up = recent_closes[i] > recent_closes[i - 1]
                    vol_up = recent_volumes[i] > recent_volumes[i - 1]
                    if price_up and vol_up:
                        count += 1
                score += count / 20.0
                n_factors += 1

            tech_score = score / max(n_factors, 1)
            results.append({"stock_id": sid, "technical_score": tech_score})

        return pd.DataFrame(results)

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """波段模式籌碼面：投信淨買超 + 三大法人 20 日累積買超 + 大戶持股 + 分點集中度（有資料時）。

        權重組合：
        - 4 因子（大戶 + 分點）: 投信 35% + 累積 35% + 大戶 15% + 分點 15%
        - 3 因子（含大戶）:      投信 40% + 累積 40% + 大戶 20%
        - 3 因子（含分點）:      投信 40% + 累積 40% + 分點 20%
        - 2 因子（基本）:        投信 50% + 累積 50%

        回傳欄位：stock_id, chip_score, chip_tier
        """
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids), "chip_tier": "N/A"})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]

        # 20 日期間
        dates = sorted(df_inst["date"].unique())
        recent_20_dates = set(dates[-20:] if len(dates) >= 20 else dates)
        inst_grouped = inst_filtered.groupby("stock_id", sort=False)

        rows = []
        for sid in stock_ids:
            if sid not in inst_grouped.groups:
                rows.append({"stock_id": sid, "trust_net": 0, "cum_20_net": 0})
                continue

            stock_inst = inst_grouped.get_group(sid)
            # 投信淨買超（全期間合計）
            trust_data = stock_inst[stock_inst["name"].str.contains("投信", na=False)]
            trust_net = trust_data["net"].sum() if not trust_data.empty else 0

            # 三大法人 20 日累積買超
            recent_inst = stock_inst[stock_inst["date"].isin(recent_20_dates)]
            cum_20_net = recent_inst["net"].sum()

            rows.append({"stock_id": sid, "trust_net": trust_net, "cum_20_net": cum_20_net})

        df = pd.DataFrame(rows)

        trust_rank = df["trust_net"].rank(pct=True)
        cum_rank = df["cum_20_net"].rank(pct=True)

        # ── 大戶持股因子 ──────────────────────────────────────────────
        df_whale = self._load_holding_data(stock_ids)
        whale_df = compute_whale_score(df_whale)
        has_whale = not whale_df.empty
        if has_whale:
            df = df.merge(whale_df, on="stock_id", how="left")
            df["whale_percent"] = df["whale_percent"].fillna(
                df["whale_percent"].median() if not df["whale_percent"].isna().all() else 0.0
            )
            df["whale_change"] = df["whale_change"].fillna(0.0)
            whale_pct_rank = df["whale_percent"].rank(pct=True)
            whale_chg_rank = df["whale_change"].rank(pct=True)
            whale_rank = whale_pct_rank * 0.60 + whale_chg_rank * 0.40

        # ── 分點集中度因子 ────────────────────────────────────────────
        df_broker_raw = self._load_broker_data(stock_ids)
        broker_df = compute_broker_score(df_broker_raw)
        has_broker = not broker_df.empty
        if has_broker:
            df = df.merge(broker_df, on="stock_id", how="left")
            df["broker_concentration"] = df["broker_concentration"].fillna(0.0)
            df["broker_consecutive_days"] = df["broker_consecutive_days"].fillna(0)
            broker_conc_rank = df["broker_concentration"].rank(pct=True)
            broker_consec_rank = df["broker_consecutive_days"].rank(pct=True)
            broker_rank = broker_conc_rank * 0.60 + broker_consec_rank * 0.40

        if has_whale and has_broker:
            # 4 因子：投信 35% + 累積 35% + 大戶 15% + 分點 15%
            df["chip_score"] = trust_rank * 0.35 + cum_rank * 0.35 + whale_rank * 0.15 + broker_rank * 0.15
            chip_tier = "4F"
        elif has_whale:
            # 3 因子：投信 40% + 累積 40% + 大戶 20%
            df["chip_score"] = trust_rank * 0.40 + cum_rank * 0.40 + whale_rank * 0.20
            chip_tier = "3F"
        elif has_broker:
            # 3 因子：投信 40% + 累積 40% + 分點 20%
            df["chip_score"] = trust_rank * 0.40 + cum_rank * 0.40 + broker_rank * 0.20
            chip_tier = "3F"
        else:
            # 2 因子：投信 50% + 累積 50%
            df["chip_score"] = trust_rank * 0.50 + cum_rank * 0.50
            chip_tier = "2F"

        df["chip_tier"] = chip_tier
        return df[["stock_id", "chip_score", "chip_tier"]]

    def _load_holding_data(self, stock_ids: list[str]) -> pd.DataFrame:
        """從 DB 載入最近 2 週的大戶持股分級資料（波段模式）。"""
        cutoff = date.today() - timedelta(days=21)
        try:
            with get_session() as session:
                rows = session.execute(
                    select(
                        HoldingDistribution.stock_id,
                        HoldingDistribution.date,
                        HoldingDistribution.level,
                        HoldingDistribution.percent,
                    ).where(
                        HoldingDistribution.stock_id.in_(stock_ids),
                        HoldingDistribution.date >= cutoff,
                    )
                ).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=["stock_id", "date", "level", "percent"])
        except Exception:
            return pd.DataFrame()

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """波段模式基本面 3 因子：YoY 40% + MoM 30% + 營收加速度 30%（排名百分位）。"""
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        if rev.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        # YoY 排名
        yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
        # MoM 排名
        mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)

        # 營收加速度 = current_yoy - prev_yoy
        if "prev_yoy_growth" in rev.columns:
            rev["acceleration"] = rev["yoy_growth"].fillna(0) - rev["prev_yoy_growth"].fillna(0)
            accel_rank = rev["acceleration"].rank(pct=True)
        else:
            accel_rank = pd.Series(0.5, index=rev.index)

        # YoY 40% + MoM 30% + 加速度 30%
        rev["fundamental_score"] = yoy_rank * 0.40 + mom_rank * 0.30 + accel_rank * 0.30

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """波段模式風險過濾：近 60 日年化波動率 > 85th percentile 剔除。"""
        return self._apply_vol_risk_filter(scored, df_price, percentile=85, window=60, annualize=True)

    def _score_candidates(
        self,
        candidates: pd.DataFrame,
        df_price: pd.DataFrame,
        df_inst: pd.DataFrame,
        df_margin: pd.DataFrame,
        df_revenue: pd.DataFrame,
        df_ann: pd.DataFrame | None = None,
        df_ann_history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """覆寫：執行通用評分後套用 VCP 波動收斂加成。"""
        # 儲存 df_price 供 _post_score() 的 VCP 計算使用
        self._df_price_for_vcp = df_price
        return super()._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue, df_ann, df_ann_history)

    def _post_score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """波段模式加成：VCP 波動收斂形態符合者 composite_score +3%。"""
        df_price = getattr(self, "_df_price_for_vcp", pd.DataFrame())
        if df_price.empty or candidates.empty:
            return candidates
        candidates = candidates.copy()
        vcp_df = compute_vcp_score(candidates["stock_id"].tolist(), df_price)
        candidates = candidates.merge(vcp_df, on="stock_id", how="left")
        candidates["vcp_bonus"] = candidates["vcp_bonus"].fillna(0.0)
        candidates["composite_score"] = candidates["composite_score"] + candidates["vcp_bonus"]
        return candidates


# ====================================================================== #
#  ValueScanner — 價值修復模式
# ====================================================================== #


class ValueScanner(MarketScanner):
    """價值修復掃描器。

    適合低估值 + 基本面轉佳 + 法人開始布局的「價值修復股」。
    粗篩：PE > 0 且 PE < 30 + 殖利率 > 2%
    細評：基本面 50% + 估值面 30% + 籌碼面 20%
    風險過濾：近 20 日波動率 > 90th percentile 剔除
    """

    mode_name = "value"
    _COARSE_WEIGHTS: dict[str, float] = {"vol_rank": 0.50, "inst_rank": 0.50}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 25)
        kwargs.setdefault("universe_config", UniverseConfig(trend_ma=None, volume_ratio_min=None))
        super().__init__(**kwargs)

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：value 模式額外載入估值資料 + 2 個月營收。含 UniverseFilter Stage 0.5。"""
        universe_ids = self._get_universe_ids()
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        with get_session() as session:
            price_query = select(
                DailyPrice.stock_id,
                DailyPrice.date,
                DailyPrice.open,
                DailyPrice.high,
                DailyPrice.low,
                DailyPrice.close,
                DailyPrice.volume,
                DailyPrice.turnover,
            ).where(DailyPrice.date >= cutoff)
            if universe_ids:
                price_query = price_query.where(DailyPrice.stock_id.in_(universe_ids))
            rows = session.execute(price_query).all()
            df_price = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "open", "high", "low", "close", "volume", "turnover"],
            )

            inst_query = select(
                InstitutionalInvestor.stock_id,
                InstitutionalInvestor.date,
                InstitutionalInvestor.name,
                InstitutionalInvestor.net,
            ).where(InstitutionalInvestor.date >= cutoff)
            if universe_ids:
                inst_query = inst_query.where(InstitutionalInvestor.stock_id.in_(universe_ids))
            rows = session.execute(inst_query).all()
            df_inst = pd.DataFrame(rows, columns=["stock_id", "date", "name", "net"])

            margin_query = select(
                MarginTrading.stock_id,
                MarginTrading.date,
                MarginTrading.margin_balance,
                MarginTrading.short_balance,
            ).where(MarginTrading.date >= cutoff)
            if universe_ids:
                margin_query = margin_query.where(MarginTrading.stock_id.in_(universe_ids))
            rows = session.execute(margin_query).all()
            df_margin = pd.DataFrame(rows, columns=["stock_id", "date", "margin_balance", "short_balance"])

            # 估值資料
            val_query = select(
                StockValuation.stock_id,
                StockValuation.date,
                StockValuation.pe_ratio,
                StockValuation.pb_ratio,
                StockValuation.dividend_yield,
            ).where(StockValuation.date >= cutoff)
            if universe_ids:
                val_query = val_query.where(StockValuation.stock_id.in_(universe_ids))
            rows = session.execute(val_query).all()
            self._df_valuation = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "pe_ratio", "pb_ratio", "dividend_yield"],
            )

            # 載入產業分類（供 _coarse_filter 相對 PE 計算）
            info_query = select(StockInfo.stock_id, StockInfo.industry_category)
            if universe_ids:
                info_query = info_query.where(StockInfo.stock_id.in_(universe_ids))
            info_rows = session.execute(info_query).all()
            df_info = pd.DataFrame(info_rows, columns=["stock_id", "industry_category"])
            df_info["industry_category"] = df_info["industry_category"].fillna("未分類")
            self._df_stock_info = df_info

        # 載入 2 個月營收（含上月，算加速度）
        df_revenue = self._load_revenue_data(stock_ids=universe_ids if universe_ids else None, months=2)

        return df_price, df_inst, df_margin, df_revenue

    def run(self) -> DiscoveryResult:
        """覆寫 run()：在 Stage 0.5 自動補抓估值、Stage 2.5 補抓候選股估值。"""
        # Stage 0: Regime 偵測
        try:
            from src.regime.detector import MarketRegimeDetector

            regime_info = MarketRegimeDetector().detect()
            self.regime = regime_info["regime"]
            logger.info("Stage 0: 市場狀態 = %s (TAIEX=%.0f)", self.regime, regime_info["taiex_close"])
        except Exception:
            self.regime = "sideways"
            logger.warning("Stage 0: 市場狀態偵測失敗，預設 sideways")

        # Stage 0.5: 估值資料覆蓋不足時，自動從 TWSE/TPEX 補抓全市場估值
        self._maybe_sync_valuation()

        # Stage 1
        df_price, df_inst, df_margin, df_revenue = self._load_market_data()
        if df_price.empty:
            logger.warning("無市場資料可供掃描")
            return DiscoveryResult(rankings=pd.DataFrame(), total_stocks=0, after_coarse=0, mode=self.mode_name)

        total_stocks = df_price["stock_id"].nunique()
        logger.info("Stage 1: 載入 %d 支股票的市場資料", total_stocks)

        # Stage 2: 粗篩
        candidates = self._coarse_filter(df_price, df_inst)
        after_coarse = len(candidates)
        logger.info("Stage 2: 粗篩後剩 %d 支候選股", after_coarse)

        if candidates.empty:
            return DiscoveryResult(
                rankings=pd.DataFrame(), total_stocks=total_stocks, after_coarse=0, mode=self.mode_name
            )

        # Stage 2.5: 補抓月營收 + 估值資料
        candidate_ids = candidates["stock_id"].tolist()
        try:
            from src.data.pipeline import sync_revenue_for_stocks, sync_valuation_for_stocks

            logger.info("Stage 2.5: 補抓 %d 支候選股月營收 + 估值...", len(candidate_ids))
            rev_count = sync_revenue_for_stocks(candidate_ids)
            val_count = sync_valuation_for_stocks(candidate_ids)
            logger.info("Stage 2.5: 補抓完成，新增 %d 筆月營收, %d 筆估值", rev_count, val_count)
            df_revenue = self._load_revenue_data(candidate_ids, months=2)
            # 重新載入估值
            self._reload_valuation(candidate_ids)
        except Exception:
            logger.warning("Stage 2.5: 資料補抓失敗（可能無 FinMind token），使用既有資料")

        # Stage 2.7: 載入候選股近期 MOPS 公告（含基準期歷史供異常率計算）
        df_ann, df_ann_history = self._load_announcement_data(candidate_ids)
        if not df_ann.empty:
            logger.info("Stage 2.7: 載入 %d 筆 MOPS 公告", len(df_ann))
        else:
            logger.info("Stage 2.7: 無 MOPS 公告資料（消息面分數預設 0.5）")

        # Stage 3: 細評
        scored = self._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue, df_ann, df_ann_history)
        logger.info("Stage 3: 完成 %d 支候選股評分", len(scored))

        # Stage 3.3: 產業加成
        scored = self._apply_sector_bonus(scored)

        # Stage 3.3a: 產業同儕相對強度加成
        scored = self._apply_sector_relative_strength(scored)

        # Stage 3.4: 週線趨勢加成（若 weekly_confirm=True）
        if self.weekly_confirm:
            scored = self._apply_weekly_trend_bonus(scored)

        # Stage 3.5: 風險過濾
        scored = self._apply_risk_filter(scored, df_price)

        # Stage 4
        rankings = self._rank_and_enrich(scored)
        sector_summary = self._compute_sector_summary(rankings)
        logger.info("Stage 4: 輸出 Top %d", min(self.top_n_results, len(rankings)))

        return DiscoveryResult(
            rankings=rankings.head(self.top_n_results),
            total_stocks=total_stocks,
            after_coarse=after_coarse,
            sector_summary=sector_summary,
            mode=self.mode_name,
        )

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """價值模式粗篩：基本過濾 + PE/殖利率門檻。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 用估值資料過濾：PE > 0 且 PE < 30，殖利率 > 2%
        df_val = getattr(self, "_df_valuation", pd.DataFrame())
        if not df_val.empty:
            # 取最新一筆估值
            val_latest = df_val.sort_values("date").groupby("stock_id").last().reset_index()
            filtered = filtered.merge(
                val_latest[["stock_id", "pe_ratio", "pb_ratio", "dividend_yield"]],
                on="stock_id",
                how="left",
            )
            # 嚴格模式：必須有估值資料，且 PE 或殖利率至少一項合格
            has_val = filtered["pe_ratio"].notna()
            # 相對估值 PE：同產業中位數 × 1.5（樣本不足 3 支時 fallback PE < 50）
            df_info = getattr(self, "_df_stock_info", pd.DataFrame())
            if not df_info.empty:
                info_map = df_info.set_index("stock_id")["industry_category"]
                industry_cat = filtered["stock_id"].map(info_map).fillna("未分類")
            else:
                industry_cat = pd.Series("未分類", index=filtered.index)
            pe_thresholds = compute_relative_pe_thresholds(industry_cat, filtered["pe_ratio"])
            pe_ok = (filtered["pe_ratio"] > 0) & (filtered["pe_ratio"] < pe_thresholds.values)
            dy_ok = filtered["dividend_yield"] > 2.0
            filtered = filtered[has_val & (pe_ok | dy_ok)].copy()
        else:
            filtered["pe_ratio"] = None
            filtered["pb_ratio"] = None
            filtered["dividend_yield"] = None

        if filtered.empty:
            return pd.DataFrame()

        # 粗篩分數：成交量排名 + 法人
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 2) 法人 5 日累積淨買超排名（比單日穩定，過濾換手雜訊）
        if not df_inst.empty:
            inst_dates = sorted(df_inst["date"].unique())
            recent_5_dates = inst_dates[-5:]
            inst_5d = df_inst[df_inst["date"].isin(recent_5_dates)]
            inst_net = inst_5d.groupby("stock_id")["net"].sum().reset_index()
            inst_net.columns = ["stock_id", "inst_net"]
            filtered = filtered.merge(inst_net, on="stock_id", how="left")
            filtered["inst_net"] = filtered["inst_net"].fillna(0)
            filtered["inst_rank"] = filtered["inst_net"].rank(pct=True)
        else:
            filtered["inst_net"] = 0
            filtered["inst_rank"] = 0.5

        # 3) 短期動能（5 日報酬，比單日漲跌更穩定）
        dates = sorted(df_price["date"].unique())
        ref_date = dates[-5] if len(dates) >= 5 else (dates[0] if len(dates) >= 2 else None)
        if ref_date is not None:
            ref = df_price[df_price["date"] == ref_date][["stock_id", "close"]].rename(columns={"close": "ref_close"})
            filtered = filtered.merge(ref, on="stock_id", how="left")
            filtered["momentum"] = (
                (filtered["close"] - filtered["ref_close"]) / filtered["ref_close"].replace(0, float("nan"))
            ).fillna(0)
        else:
            filtered["momentum"] = 0

        filtered["coarse_score"] = sum(
            filtered[k] * v for k, v in self._COARSE_WEIGHTS.items() if k in filtered.columns
        )
        filtered = filtered.nlargest(self._effective_top_n(len(filtered)), "coarse_score")
        return filtered

    def _compute_extra_scores(self, stock_ids: list[str]) -> list[pd.DataFrame]:
        """價值模式額外維度：估值面分數。"""
        return [self._compute_valuation_scores(stock_ids)]

    def _post_score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """用 technical_score 欄位存估值分數（供 _rank_and_enrich 顯示用）。"""
        if "valuation_score" in candidates.columns:
            candidates["technical_score"] = candidates["valuation_score"]
        return candidates

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """價值模式基本面：營收 40% + ROE 25% + 毛利率 QoQ 20% + EPS YoY 15%。

        財報資料不足時自動降回營收單因子（YoY 70% + MoM 30%）。
        """
        # --- 營收基礎分（與 base class 相同）---
        if not df_revenue.empty:
            rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        else:
            rev = pd.DataFrame()

        if not rev.empty:
            yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
            mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)
            rev["rev_base"] = yoy_rank * 0.70 + mom_rank * 0.30
        else:
            rev = pd.DataFrame({"stock_id": stock_ids, "rev_base": [0.5] * len(stock_ids)})

        # --- 財報因子 ---
        df_fin = self._load_financial_data(stock_ids, quarters=5)
        if df_fin.empty:
            # 降回純營收分
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
            result["fundamental_score"] = result["rev_base"].fillna(0.5)
            return result[["stock_id", "fundamental_score"]]

        # 計算每支股票財報指標
        grouped = df_fin.groupby("stock_id", sort=False)
        fin_rows = []
        for sid in stock_ids:
            row: dict = {"stock_id": sid, "roe_val": None, "gm_qoq": None, "eps_yoy": None}
            if sid in grouped.groups:
                grp = grouped.get_group(sid).sort_values("date", ascending=False)
                # ROE：最新一季
                if len(grp) >= 1 and pd.notna(grp.iloc[0]["roe"]):
                    row["roe_val"] = float(grp.iloc[0]["roe"])
                # 毛利率 QoQ：最新季 - 上一季
                if len(grp) >= 2 and pd.notna(grp.iloc[0]["gross_margin"]) and pd.notna(grp.iloc[1]["gross_margin"]):
                    row["gm_qoq"] = float(grp.iloc[0]["gross_margin"]) - float(grp.iloc[1]["gross_margin"])
                # EPS YoY：最新季 vs 去年同季
                if len(grp) >= 1 and pd.notna(grp.iloc[0]["eps"]):
                    cur_q = int(grp.iloc[0]["quarter"])
                    cur_y = int(grp.iloc[0]["year"])
                    same_q = grp[(grp["quarter"] == cur_q) & (grp["year"] == cur_y - 1)]
                    if not same_q.empty and pd.notna(same_q.iloc[0]["eps"]):
                        prev_eps = float(same_q.iloc[0]["eps"])
                        if abs(prev_eps) > 0.01:
                            row["eps_yoy"] = (float(grp.iloc[0]["eps"]) - prev_eps) / abs(prev_eps)
            fin_rows.append(row)

        df_metrics = pd.DataFrame(fin_rows)
        has_any = df_metrics[["roe_val", "gm_qoq", "eps_yoy"]].notna().any(axis=1).any()
        if not has_any:
            # 財報欄位全 NULL → 降回純營收分
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
            result["fundamental_score"] = result["rev_base"].fillna(0.5)
            return result[["stock_id", "fundamental_score"]]

        # 排名百分位（用 min_count=1 避免全 NaN 時 rank 失敗）
        roe_rank = df_metrics["roe_val"].rank(pct=True).fillna(0.5)
        gm_qoq_rank = df_metrics["gm_qoq"].rank(pct=True).fillna(0.5)
        eps_yoy_rank = df_metrics["eps_yoy"].rank(pct=True).fillna(0.5)

        # 合併營收基礎分
        df_metrics = df_metrics.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
        df_metrics["rev_base"] = df_metrics["rev_base"].fillna(0.5)

        # 加權：營收 40% + ROE 25% + 毛利率 QoQ 20% + EPS YoY 15%
        df_metrics["fundamental_score"] = (
            df_metrics["rev_base"] * 0.40 + roe_rank * 0.25 + gm_qoq_rank * 0.20 + eps_yoy_rank * 0.15
        )

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(df_metrics[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    def _compute_valuation_scores(self, stock_ids: list[str]) -> pd.DataFrame:
        """估值面 3 因子：PE 反向排名 40% + PB 反向排名 30% + 殖利率排名 30%。"""
        df_val = getattr(self, "_df_valuation", pd.DataFrame())
        if df_val.empty:
            return pd.DataFrame({"stock_id": stock_ids, "valuation_score": [0.5] * len(stock_ids)})

        # 取最新一筆
        val = df_val[df_val["stock_id"].isin(stock_ids)].copy()
        if val.empty:
            return pd.DataFrame({"stock_id": stock_ids, "valuation_score": [0.5] * len(stock_ids)})

        val = val.sort_values("date").groupby("stock_id").last().reset_index()

        # PE 反向排名：PE 越低分數越高；缺值者給中間分（0.5），不因缺值受益
        pe_rank = val["pe_ratio"].rank(pct=True, ascending=False, na_option="keep").fillna(0.5)
        # PB 反向排名：PB 越低分數越高；缺值者給中間分
        pb_rank = val["pb_ratio"].rank(pct=True, ascending=False, na_option="keep").fillna(0.5)
        # 殖利率正向排名：越高分數越高
        dy_rank = val["dividend_yield"].fillna(0).rank(pct=True)

        val["valuation_score"] = pe_rank * 0.40 + pb_rank * 0.30 + dy_rank * 0.30

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(val[["stock_id", "valuation_score"]], on="stock_id", how="left")
        return result

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """價值模式籌碼面：投信近期買超 + 三大法人累積 + 分點集中度（有資料時）。

        權重組合：
        - 3 因子（含分點）: 投信 40% + 累積 40% + 分點 20%
        - 2 因子（基本）:   投信 50% + 累積 50%

        回傳欄位：stock_id, chip_score, chip_tier（"3F" 或 "2F"）
        """
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids), "chip_tier": "N/A"})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
        dates = sorted(df_inst["date"].unique())
        recent_20_dates = set(dates[-20:] if len(dates) >= 20 else dates)
        inst_grouped = inst_filtered.groupby("stock_id", sort=False)

        rows = []
        for sid in stock_ids:
            if sid not in inst_grouped.groups:
                rows.append({"stock_id": sid, "trust_net": 0, "cum_net": 0})
                continue

            stock_inst = inst_grouped.get_group(sid)
            trust_data = stock_inst[stock_inst["name"].str.contains("投信", na=False)]
            trust_net = trust_data["net"].sum() if not trust_data.empty else 0

            recent_inst = stock_inst[stock_inst["date"].isin(recent_20_dates)]
            cum_net = recent_inst["net"].sum()

            rows.append({"stock_id": sid, "trust_net": trust_net, "cum_net": cum_net})

        df = pd.DataFrame(rows)
        trust_rank = df["trust_net"].rank(pct=True)
        cum_rank = df["cum_net"].rank(pct=True)

        # ── 分點集中度因子 ────────────────────────────────────────────
        df_broker_raw = self._load_broker_data(stock_ids)
        broker_df = compute_broker_score(df_broker_raw)
        has_broker = not broker_df.empty
        if has_broker:
            df = df.merge(broker_df, on="stock_id", how="left")
            df["broker_concentration"] = df["broker_concentration"].fillna(0.0)
            df["broker_consecutive_days"] = df["broker_consecutive_days"].fillna(0)
            broker_conc_rank = df["broker_concentration"].rank(pct=True)
            broker_consec_rank = df["broker_consecutive_days"].rank(pct=True)
            broker_rank = broker_conc_rank * 0.60 + broker_consec_rank * 0.40
            # 3 因子：投信 40% + 累積 40% + 分點 20%
            df["chip_score"] = trust_rank * 0.40 + cum_rank * 0.40 + broker_rank * 0.20
            chip_tier = "3F"
        else:
            # 2 因子：投信 50% + 累積 50%
            df["chip_score"] = trust_rank * 0.50 + cum_rank * 0.50
            chip_tier = "2F"

        df["chip_tier"] = chip_tier
        return df[["stock_id", "chip_score", "chip_tier"]]

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """價值模式風險過濾：近 20 日波動率 > 90th percentile 剔除。"""
        return self._apply_vol_risk_filter(scored, df_price, percentile=90)


# ====================================================================== #
#  DividendScanner — 高息存股模式
# ====================================================================== #


class DividendScanner(MarketScanner):
    """高息存股掃描器。

    篩選高殖利率、配息穩定、估值合理的存股標的。
    粗篩：殖利率 > 3% + PE > 0
    細評：基本面 + 殖利率/估值 + 籌碼面 + 消息面（依 Regime 動態加權）
    風險過濾：近 20 日波動率 > 90th percentile 剔除
    """

    mode_name = "dividend"
    _COARSE_WEIGHTS: dict[str, float] = {"dy_rank": 0.50, "vol_rank": 0.30, "inst_rank": 0.20}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 25)
        kwargs.setdefault("universe_config", UniverseConfig(trend_ma=None, volume_ratio_min=None))
        super().__init__(**kwargs)

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：dividend 模式額外載入估值資料 + 2 個月營收。含 UniverseFilter Stage 0.5。"""
        universe_ids = self._get_universe_ids()
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        with get_session() as session:
            price_query = select(
                DailyPrice.stock_id,
                DailyPrice.date,
                DailyPrice.open,
                DailyPrice.high,
                DailyPrice.low,
                DailyPrice.close,
                DailyPrice.volume,
                DailyPrice.turnover,
            ).where(DailyPrice.date >= cutoff)
            if universe_ids:
                price_query = price_query.where(DailyPrice.stock_id.in_(universe_ids))
            rows = session.execute(price_query).all()
            df_price = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "open", "high", "low", "close", "volume", "turnover"],
            )

            inst_query = select(
                InstitutionalInvestor.stock_id,
                InstitutionalInvestor.date,
                InstitutionalInvestor.name,
                InstitutionalInvestor.net,
            ).where(InstitutionalInvestor.date >= cutoff)
            if universe_ids:
                inst_query = inst_query.where(InstitutionalInvestor.stock_id.in_(universe_ids))
            rows = session.execute(inst_query).all()
            df_inst = pd.DataFrame(rows, columns=["stock_id", "date", "name", "net"])

            margin_query = select(
                MarginTrading.stock_id,
                MarginTrading.date,
                MarginTrading.margin_balance,
                MarginTrading.short_balance,
            ).where(MarginTrading.date >= cutoff)
            if universe_ids:
                margin_query = margin_query.where(MarginTrading.stock_id.in_(universe_ids))
            rows = session.execute(margin_query).all()
            df_margin = pd.DataFrame(rows, columns=["stock_id", "date", "margin_balance", "short_balance"])

            # 估值資料
            val_query = select(
                StockValuation.stock_id,
                StockValuation.date,
                StockValuation.pe_ratio,
                StockValuation.pb_ratio,
                StockValuation.dividend_yield,
            ).where(StockValuation.date >= cutoff)
            if universe_ids:
                val_query = val_query.where(StockValuation.stock_id.in_(universe_ids))
            rows = session.execute(val_query).all()
            self._df_valuation = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "pe_ratio", "pb_ratio", "dividend_yield"],
            )

            # 載入近 4 季 EPS（供 _coarse_filter 配息連續性篩選）
            eps_cutoff = date.today() - timedelta(days=400)  # 4 季 ≈ 400 天
            eps_query = select(
                FinancialStatement.stock_id,
                FinancialStatement.date,
                FinancialStatement.eps,
            ).where(FinancialStatement.date >= eps_cutoff)
            if universe_ids:
                eps_query = eps_query.where(FinancialStatement.stock_id.in_(universe_ids))
            eps_rows = session.execute(eps_query).all()
            self._df_eps_quarterly = pd.DataFrame(eps_rows, columns=["stock_id", "date", "eps"])

        # 載入 2 個月營收（含上月，算加速度）
        df_revenue = self._load_revenue_data(stock_ids=universe_ids if universe_ids else None, months=2)

        return df_price, df_inst, df_margin, df_revenue

    def run(self) -> DiscoveryResult:
        """覆寫 run()：在 Stage 0.5 自動補抓估值、Stage 2.5 補抓候選股估值。"""
        # Stage 0: Regime 偵測
        try:
            from src.regime.detector import MarketRegimeDetector

            regime_info = MarketRegimeDetector().detect()
            self.regime = regime_info["regime"]
            logger.info("Stage 0: 市場狀態 = %s (TAIEX=%.0f)", self.regime, regime_info["taiex_close"])
        except Exception:
            self.regime = "sideways"
            logger.warning("Stage 0: 市場狀態偵測失敗，預設 sideways")

        # Stage 0.5: 估值資料覆蓋不足時，自動從 TWSE/TPEX 補抓全市場估值
        self._maybe_sync_valuation()

        # Stage 1
        df_price, df_inst, df_margin, df_revenue = self._load_market_data()
        if df_price.empty:
            logger.warning("無市場資料可供掃描")
            return DiscoveryResult(rankings=pd.DataFrame(), total_stocks=0, after_coarse=0, mode=self.mode_name)

        total_stocks = df_price["stock_id"].nunique()
        logger.info("Stage 1: 載入 %d 支股票的市場資料", total_stocks)

        # Stage 2: 粗篩
        candidates = self._coarse_filter(df_price, df_inst)
        after_coarse = len(candidates)
        logger.info("Stage 2: 粗篩後剩 %d 支候選股", after_coarse)

        if candidates.empty:
            return DiscoveryResult(
                rankings=pd.DataFrame(), total_stocks=total_stocks, after_coarse=0, mode=self.mode_name
            )

        # Stage 2.5: 補抓月營收 + 估值資料
        candidate_ids = candidates["stock_id"].tolist()
        try:
            from src.data.pipeline import sync_revenue_for_stocks, sync_valuation_for_stocks

            logger.info("Stage 2.5: 補抓 %d 支候選股月營收 + 估值...", len(candidate_ids))
            rev_count = sync_revenue_for_stocks(candidate_ids)
            val_count = sync_valuation_for_stocks(candidate_ids)
            logger.info("Stage 2.5: 補抓完成，新增 %d 筆月營收, %d 筆估值", rev_count, val_count)
            df_revenue = self._load_revenue_data(candidate_ids, months=2)
            self._reload_valuation(candidate_ids)
        except Exception:
            logger.warning("Stage 2.5: 資料補抓失敗（可能無 FinMind token），使用既有資料")

        # Stage 2.7: 載入候選股近期 MOPS 公告（含基準期歷史供異常率計算）
        df_ann, df_ann_history = self._load_announcement_data(candidate_ids)
        if not df_ann.empty:
            logger.info("Stage 2.7: 載入 %d 筆 MOPS 公告", len(df_ann))
        else:
            logger.info("Stage 2.7: 無 MOPS 公告資料（消息面分數預設 0.5）")

        # Stage 3: 細評
        scored = self._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue, df_ann, df_ann_history)
        logger.info("Stage 3: 完成 %d 支候選股評分", len(scored))

        # Stage 3.3: 產業加成
        scored = self._apply_sector_bonus(scored)

        # Stage 3.3a: 產業同儕相對強度加成
        scored = self._apply_sector_relative_strength(scored)

        # Stage 3.4: 週線趨勢加成（若 weekly_confirm=True）
        if self.weekly_confirm:
            scored = self._apply_weekly_trend_bonus(scored)

        # Stage 3.5: 風險過濾
        scored = self._apply_risk_filter(scored, df_price)

        # Stage 4
        rankings = self._rank_and_enrich(scored)
        sector_summary = self._compute_sector_summary(rankings)
        logger.info("Stage 4: 輸出 Top %d", min(self.top_n_results, len(rankings)))

        return DiscoveryResult(
            rankings=rankings.head(self.top_n_results),
            total_stocks=total_stocks,
            after_coarse=after_coarse,
            sector_summary=sector_summary,
            mode=self.mode_name,
        )

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """高息模式粗篩：基本過濾 + 殖利率 > 3% + PE > 0。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 用估值資料過濾：必須有估值資料、殖利率 > 3%、PE > 0
        df_val = getattr(self, "_df_valuation", pd.DataFrame())
        if not df_val.empty:
            val_latest = df_val.sort_values("date").groupby("stock_id").last().reset_index()
            filtered = filtered.merge(
                val_latest[["stock_id", "pe_ratio", "pb_ratio", "dividend_yield"]],
                on="stock_id",
                how="left",
            )
            has_val = filtered["dividend_yield"].notna()
            dy_ok = filtered["dividend_yield"] > 3.0
            pe_ok = filtered["pe_ratio"] > 0
            filtered = filtered[has_val & dy_ok & pe_ok].copy()
        else:
            return pd.DataFrame()

        if filtered.empty:
            return pd.DataFrame()

        # 配息連續性篩選：近 4 季 EPS 皆 > 0（無財報資料者 pass through）
        df_eps = getattr(self, "_df_eps_quarterly", pd.DataFrame())
        eps_fail_ids = compute_eps_sustainability(df_eps, min_quarters=4)
        if eps_fail_ids:
            before_count = len(filtered)
            filtered = filtered[~filtered["stock_id"].isin(eps_fail_ids)].copy()
            removed = before_count - len(filtered)
            if removed > 0:
                logger.info("Stage 2 EPS 連續性: 排除 %d 支近 4 季有負 EPS 股票", removed)

        if filtered.empty:
            return pd.DataFrame()

        # 粗篩分數：殖利率排名 50% + 成交量排名 30% + 法人排名 20%
        filtered["dy_rank"] = filtered["dividend_yield"].rank(pct=True)
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 2) 法人 5 日累積淨買超排名（比單日穩定，過濾換手雜訊）
        if not df_inst.empty:
            inst_dates = sorted(df_inst["date"].unique())
            recent_5_dates = inst_dates[-5:]
            inst_5d = df_inst[df_inst["date"].isin(recent_5_dates)]
            inst_net = inst_5d.groupby("stock_id")["net"].sum().reset_index()
            inst_net.columns = ["stock_id", "inst_net"]
            filtered = filtered.merge(inst_net, on="stock_id", how="left")
            filtered["inst_net"] = filtered["inst_net"].fillna(0)
            filtered["inst_rank"] = filtered["inst_net"].rank(pct=True)
        else:
            filtered["inst_net"] = 0
            filtered["inst_rank"] = 0.5

        # 3) 短期動能（5 日報酬，比單日漲跌更穩定）
        dates = sorted(df_price["date"].unique())
        ref_date = dates[-5] if len(dates) >= 5 else (dates[0] if len(dates) >= 2 else None)
        if ref_date is not None:
            ref = df_price[df_price["date"] == ref_date][["stock_id", "close"]].rename(columns={"close": "ref_close"})
            filtered = filtered.merge(ref, on="stock_id", how="left")
            filtered["momentum"] = (
                (filtered["close"] - filtered["ref_close"]) / filtered["ref_close"].replace(0, float("nan"))
            ).fillna(0)
        else:
            filtered["momentum"] = 0

        filtered["coarse_score"] = sum(
            filtered[k] * v for k, v in self._COARSE_WEIGHTS.items() if k in filtered.columns
        )
        filtered = filtered.nlargest(self._effective_top_n(len(filtered)), "coarse_score")
        return filtered

    def _compute_extra_scores(self, stock_ids: list[str]) -> list[pd.DataFrame]:
        """高息模式額外維度：殖利率/估值面分數。"""
        return [self._compute_dividend_scores(stock_ids)]

    def _post_score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """用 technical_score 欄位存殖利率分數（供 _rank_and_enrich 顯示用）。"""
        if "dividend_score" in candidates.columns:
            candidates["technical_score"] = candidates["dividend_score"]
        return candidates

    def _compute_dividend_scores(self, stock_ids: list[str]) -> pd.DataFrame:
        """殖利率面 3 因子：殖利率排名 50% + PE 反向排名 30% + PB 反向排名 20%。"""
        df_val = getattr(self, "_df_valuation", pd.DataFrame())
        if df_val.empty:
            return pd.DataFrame({"stock_id": stock_ids, "dividend_score": [0.5] * len(stock_ids)})

        val = df_val[df_val["stock_id"].isin(stock_ids)].copy()
        if val.empty:
            return pd.DataFrame({"stock_id": stock_ids, "dividend_score": [0.5] * len(stock_ids)})

        val = val.sort_values("date").groupby("stock_id").last().reset_index()

        # 殖利率正向排名：越高分數越高
        dy_rank = val["dividend_yield"].fillna(0).rank(pct=True)
        # PE 反向排名：PE 越低分數越高；缺值者給中間分（0.5），不因缺值受益
        pe_rank = val["pe_ratio"].rank(pct=True, ascending=False, na_option="keep").fillna(0.5)
        # PB 反向排名：PB 越低分數越高；缺值者給中間分
        pb_rank = val["pb_ratio"].rank(pct=True, ascending=False, na_option="keep").fillna(0.5)

        val["dividend_score"] = dy_rank * 0.50 + pe_rank * 0.30 + pb_rank * 0.20

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(val[["stock_id", "dividend_score"]], on="stock_id", how="left")
        return result

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """高息模式基本面：營收 40% + EPS 穩定性 35% + 配息率代理 25%。

        EPS 穩定性 = 最近 4 季 EPS 標準差（越低越穩定，倒排）。
        配息率代理 = 4 季中 EPS > 0 的比例（能穩定獲利才能持續配息）。
        財報資料不足時自動降回營收單因子（YoY 70% + MoM 30%）。
        """
        # --- 營收基礎分 ---
        if not df_revenue.empty:
            rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        else:
            rev = pd.DataFrame()

        if not rev.empty:
            yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
            mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)
            rev["rev_base"] = yoy_rank * 0.70 + mom_rank * 0.30
        else:
            rev = pd.DataFrame({"stock_id": stock_ids, "rev_base": [0.5] * len(stock_ids)})

        # --- 財報因子（需要最近 4 季 EPS）---
        df_fin = self._load_financial_data(stock_ids, quarters=5)
        if df_fin.empty:
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
            result["fundamental_score"] = result["rev_base"].fillna(0.5)
            return result[["stock_id", "fundamental_score"]]

        grouped = df_fin.groupby("stock_id", sort=False)
        fin_rows = []
        for sid in stock_ids:
            row: dict = {"stock_id": sid, "eps_std": None, "positive_eps_ratio": None}
            if sid in grouped.groups:
                grp = grouped.get_group(sid).sort_values("date", ascending=False).head(4)
                eps_vals = grp["eps"].dropna().tolist()
                if len(eps_vals) >= 2:
                    row["eps_std"] = float(pd.Series(eps_vals).std())
                    row["positive_eps_ratio"] = sum(1 for e in eps_vals if e > 0) / len(eps_vals)
                elif len(eps_vals) == 1:
                    row["eps_std"] = 0.0  # 只有 1 季，視為完全穩定
                    row["positive_eps_ratio"] = 1.0 if eps_vals[0] > 0 else 0.0
            fin_rows.append(row)

        df_metrics = pd.DataFrame(fin_rows)
        has_any = df_metrics[["eps_std", "positive_eps_ratio"]].notna().any(axis=1).any()
        if not has_any:
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
            result["fundamental_score"] = result["rev_base"].fillna(0.5)
            return result[["stock_id", "fundamental_score"]]

        # EPS 穩定性：std 越低越好 → ascending=False（反向排名）
        eps_stability_rank = df_metrics["eps_std"].rank(pct=True, ascending=False).fillna(0.5)
        # 配息率代理：正 EPS 比例越高越好
        payout_proxy_rank = df_metrics["positive_eps_ratio"].rank(pct=True).fillna(0.5)

        df_metrics = df_metrics.merge(rev[["stock_id", "rev_base"]], on="stock_id", how="left")
        df_metrics["rev_base"] = df_metrics["rev_base"].fillna(0.5)

        # 加權：營收 40% + EPS 穩定性 35% + 配息率代理 25%
        df_metrics["fundamental_score"] = (
            df_metrics["rev_base"] * 0.40 + eps_stability_rank * 0.35 + payout_proxy_rank * 0.25
        )

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(df_metrics[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """高息模式籌碼面 2 因子：投信淨買超 50% + 三大法人累積買超 50%。

        回傳欄位：stock_id, chip_score, chip_tier（固定 "2F"）
        """
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids), "chip_tier": "N/A"})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
        dates = sorted(df_inst["date"].unique())
        recent_20_dates = set(dates[-20:] if len(dates) >= 20 else dates)
        inst_grouped = inst_filtered.groupby("stock_id", sort=False)

        rows = []
        for sid in stock_ids:
            if sid not in inst_grouped.groups:
                rows.append({"stock_id": sid, "trust_net": 0, "cum_net": 0})
                continue

            stock_inst = inst_grouped.get_group(sid)
            trust_data = stock_inst[stock_inst["name"].str.contains("投信", na=False)]
            trust_net = trust_data["net"].sum() if not trust_data.empty else 0

            recent_inst = stock_inst[stock_inst["date"].isin(recent_20_dates)]
            cum_net = recent_inst["net"].sum()

            rows.append({"stock_id": sid, "trust_net": trust_net, "cum_net": cum_net})

        df = pd.DataFrame(rows)
        trust_rank = df["trust_net"].rank(pct=True)
        cum_rank = df["cum_net"].rank(pct=True)
        df["chip_score"] = trust_rank * 0.50 + cum_rank * 0.50
        df["chip_tier"] = "2F"

        return df[["stock_id", "chip_score", "chip_tier"]]

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """高息模式風險過濾：近 20 日波動率 > 90th percentile 剔除。"""
        return self._apply_vol_risk_filter(scored, df_price, percentile=90)


# ====================================================================== #
#  GrowthScanner — 高成長模式
# ====================================================================== #


class GrowthScanner(MarketScanner):
    """高成長掃描器。

    篩選營收/EPS 高速成長、動能啟動的成長型股票。
    粗篩：YoY > 10%
    細評：基本面 + 技術面（動能確認）+ 籌碼面 + 消息面（依 Regime 動態加權）
    風險過濾：ATR(14)/close > 80th percentile 剔除
    """

    mode_name = "growth"
    _COARSE_WEIGHTS: dict[str, float] = {"yoy_rank": 0.40, "vol_rank": 0.30, "inst_rank": 0.30}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 25)
        kwargs.setdefault("universe_config", UniverseConfig(trend_ma=20, volume_ratio_min=2.0))
        super().__init__(**kwargs)

    def run(self) -> DiscoveryResult:
        """覆寫 run()：粗篩前自動同步 MOPS 全市場月營收。"""
        # Stage 0: Regime 偵測
        try:
            from src.regime.detector import MarketRegimeDetector

            regime_info = MarketRegimeDetector().detect()
            self.regime = regime_info["regime"]
            logger.info("Stage 0: 市場狀態 = %s (TAIEX=%.0f)", self.regime, regime_info["taiex_close"])
        except Exception:
            self.regime = "sideways"
            logger.warning("Stage 0: 市場狀態偵測失敗，預設 sideways")

        # Stage 0.5: 檢查月營收覆蓋率，不足時自動從 MOPS 補抓
        try:
            from sqlalchemy import func as sa_func

            with get_session() as session:
                rev_count = session.execute(select(sa_func.count(sa_func.distinct(MonthlyRevenue.stock_id)))).scalar()

            if not rev_count or rev_count < 500:
                logger.info(
                    "Stage 0.5: 月營收僅 %d 支，自動從 MOPS 同步全市場月營收...",
                    rev_count or 0,
                )
                from src.data.pipeline import sync_mops_revenue

                mops_count = sync_mops_revenue(months=1)
                logger.info("Stage 0.5: MOPS 月營收同步完成，新增 %d 筆", mops_count)
        except Exception:
            logger.warning("Stage 0.5: MOPS 月營收自動同步失敗，使用既有資料")

        # Stage 1: 載入資料
        df_price, df_inst, df_margin, df_revenue = self._load_market_data()
        if df_price.empty:
            logger.warning("無市場資料可供掃描")
            return DiscoveryResult(rankings=pd.DataFrame(), total_stocks=0, after_coarse=0, mode=self.mode_name)

        total_stocks = df_price["stock_id"].nunique()
        logger.info("Stage 1: 載入 %d 支股票的市場資料", total_stocks)

        # 預填充 _coarse_revenue，避免 _coarse_filter() 重複查詢 DB（Problem 4 修正）
        # _load_market_data() 已載入 4 個月營收，直接重用，無需再次查詢
        self._coarse_revenue = df_revenue

        # Stage 2: 粗篩
        candidates = self._coarse_filter(df_price, df_inst)
        after_coarse = len(candidates)
        logger.info("Stage 2: 粗篩後剩 %d 支候選股", after_coarse)

        if candidates.empty:
            return DiscoveryResult(
                rankings=pd.DataFrame(), total_stocks=total_stocks, after_coarse=0, mode=self.mode_name
            )

        # Stage 2.5: 補抓月營收 + 估值資料
        candidate_ids = candidates["stock_id"].tolist()
        try:
            from src.data.pipeline import sync_revenue_for_stocks, sync_valuation_for_stocks

            logger.info("Stage 2.5: 補抓 %d 支候選股月營收 + 估值...", len(candidate_ids))
            rev_count = sync_revenue_for_stocks(candidate_ids)
            val_count = sync_valuation_for_stocks(candidate_ids)
            logger.info("Stage 2.5: 補抓完成，新增 %d 筆月營收, %d 筆估值", rev_count, val_count)
            df_revenue = self._load_revenue_data(candidate_ids, months=4)
        except Exception:
            logger.warning("Stage 2.5: 資料補抓失敗（可能無 FinMind token），使用既有資料")

        # Stage 2.7: 載入候選股近期 MOPS 公告（含基準期歷史供異常率計算）
        df_ann, df_ann_history = self._load_announcement_data(candidate_ids)
        if not df_ann.empty:
            logger.info("Stage 2.7: 載入 %d 筆 MOPS 公告", len(df_ann))
        else:
            logger.info("Stage 2.7: 無 MOPS 公告資料（消息面分數預設 0.5）")

        # Stage 3: 細評
        scored = self._score_candidates(candidates, df_price, df_inst, df_margin, df_revenue, df_ann, df_ann_history)
        logger.info("Stage 3: 完成 %d 支候選股評分", len(scored))

        # Stage 3.3: 產業加成
        scored = self._apply_sector_bonus(scored)

        # Stage 3.3a: 產業同儕相對強度加成
        scored = self._apply_sector_relative_strength(scored)

        # Stage 3.4: 週線趨勢加成（若 weekly_confirm=True）
        if self.weekly_confirm:
            scored = self._apply_weekly_trend_bonus(scored)

        # Stage 3.5: 風險過濾
        scored = self._apply_risk_filter(scored, df_price)

        # Stage 4
        rankings = self._rank_and_enrich(scored)
        sector_summary = self._compute_sector_summary(rankings)
        logger.info("Stage 4: 輸出 Top %d", min(self.top_n_results, len(rankings)))

        return DiscoveryResult(
            rankings=rankings.head(self.top_n_results),
            total_stocks=total_stocks,
            after_coarse=after_coarse,
            sector_summary=sector_summary,
            mode=self.mode_name,
        )

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：growth 模式載入 2 個月營收資料（算加速度）。"""
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        with get_session() as session:
            rows = session.execute(
                select(
                    DailyPrice.stock_id,
                    DailyPrice.date,
                    DailyPrice.open,
                    DailyPrice.high,
                    DailyPrice.low,
                    DailyPrice.close,
                    DailyPrice.volume,
                ).where(DailyPrice.date >= cutoff)
            ).all()
            df_price = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "open", "high", "low", "close", "volume"],
            )

            rows = session.execute(
                select(
                    InstitutionalInvestor.stock_id,
                    InstitutionalInvestor.date,
                    InstitutionalInvestor.name,
                    InstitutionalInvestor.net,
                ).where(InstitutionalInvestor.date >= cutoff)
            ).all()
            df_inst = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "name", "net"],
            )

            rows = session.execute(
                select(
                    MarginTrading.stock_id,
                    MarginTrading.date,
                    MarginTrading.margin_balance,
                    MarginTrading.short_balance,
                ).where(MarginTrading.date >= cutoff)
            ).all()
            df_margin = pd.DataFrame(
                rows,
                columns=["stock_id", "date", "margin_balance", "short_balance"],
            )

        # 載入 4 個月營收（含 3 個月前，用於計算加速度）
        df_revenue = self._load_revenue_data(months=4)

        return df_price, df_inst, df_margin, df_revenue

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """高成長模式粗篩：基本過濾 + YoY > 10%。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 需要營收資料做粗篩
        df_revenue = getattr(self, "_coarse_revenue", pd.DataFrame())
        if df_revenue.empty:
            df_revenue = self._load_revenue_data(months=1)
            self._coarse_revenue = df_revenue

        if not df_revenue.empty:
            filtered = filtered.merge(
                df_revenue[["stock_id", "yoy_growth"]],
                on="stock_id",
                how="left",
            )
            # 必須有營收資料且 YoY > 10%
            has_rev = filtered["yoy_growth"].notna()
            yoy_ok = filtered["yoy_growth"] > 10.0
            filtered = filtered[has_rev & yoy_ok].copy()
        else:
            return pd.DataFrame()

        if filtered.empty:
            return pd.DataFrame()

        # 粗篩分數：YoY 排名 40% + 成交量 30% + 法人 30%
        filtered["yoy_rank"] = filtered["yoy_growth"].rank(pct=True)
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 2) 法人 5 日累積淨買超排名（比單日穩定，過濾換手雜訊）
        if not df_inst.empty:
            inst_dates = sorted(df_inst["date"].unique())
            recent_5_dates = inst_dates[-5:]
            inst_5d = df_inst[df_inst["date"].isin(recent_5_dates)]
            inst_net = inst_5d.groupby("stock_id")["net"].sum().reset_index()
            inst_net.columns = ["stock_id", "inst_net"]
            filtered = filtered.merge(inst_net, on="stock_id", how="left")
            filtered["inst_net"] = filtered["inst_net"].fillna(0)
            filtered["inst_rank"] = filtered["inst_net"].rank(pct=True)
        else:
            filtered["inst_net"] = 0
            filtered["inst_rank"] = 0.5

        # 3) 短期動能（5 日報酬，比單日漲跌更穩定）
        dates = sorted(df_price["date"].unique())
        ref_date = dates[-5] if len(dates) >= 5 else (dates[0] if len(dates) >= 2 else None)
        if ref_date is not None:
            ref = df_price[df_price["date"] == ref_date][["stock_id", "close"]].rename(columns={"close": "ref_close"})
            filtered = filtered.merge(ref, on="stock_id", how="left")
            filtered["momentum"] = (
                (filtered["close"] - filtered["ref_close"]) / filtered["ref_close"].replace(0, float("nan"))
            ).fillna(0)
        else:
            filtered["momentum"] = 0

        filtered["coarse_score"] = sum(
            filtered[k] * v for k, v in self._COARSE_WEIGHTS.items() if k in filtered.columns
        )
        filtered = filtered.nlargest(self._effective_top_n(len(filtered)), "coarse_score")
        return filtered

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """高成長模式技術面 5 因子（委派至 base class 共用動能評分實作）。"""
        return self._compute_momentum_style_technical_scores(stock_ids, df_price)

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """高成長模式籌碼面：外資連續買超 + 買超佔量比 + 三大法人合計 + 券資比 + 分點集中度（有資料時）。

        權重組合：
        - 5 因子（券資比 + 分點）: 外資 25% + 量比 22% + 法人 22% + 券資比 16% + 分點 15%
        - 4 因子（含券資比）:      外資 30% + 量比 25% + 法人 25% + 券資比 20%
        - 4 因子（含分點）:        外資 32% + 量比 24% + 法人 24% + 分點 20%
        - 3 因子（基本）:          外資 40% + 量比 30% + 法人 30%

        回傳欄位：stock_id, chip_score, chip_tier（"5F"、"4F" 或 "3F"）
        """
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids), "chip_tier": "N/A"})

        inst_filtered = df_inst[df_inst["stock_id"].isin(stock_ids)]
        inst_grouped = inst_filtered.groupby("stock_id", sort=False)
        price_grouped = (
            df_price[df_price["stock_id"].isin(stock_ids)].groupby("stock_id", sort=False)
            if df_price is not None and not df_price.empty
            else None
        )
        rows = []
        for sid in stock_ids:
            if sid not in inst_grouped.groups:
                rows.append({"stock_id": sid, "consec_foreign_days": 0, "buy_vol_ratio": 0.0, "total_net": 0})
                continue
            stock_inst = inst_grouped.get_group(sid)

            foreign_data = stock_inst[stock_inst["name"].str.contains("外資", na=False)]
            consec_foreign = 0
            if not foreign_data.empty:
                daily_foreign = foreign_data.groupby("date")["net"].sum().sort_index(ascending=False)
                for val in daily_foreign.values:
                    if val > 0:
                        consec_foreign += 1
                    else:
                        break

            total_net = stock_inst["net"].sum()
            buy_vol_ratio = 0.0
            if price_grouped is not None and sid in price_grouped.groups:
                stock_price = price_grouped.get_group(sid)
                latest_vol = stock_price.loc[stock_price["date"].idxmax(), "volume"]
                if latest_vol > 0:
                    buy_vol_ratio = total_net / latest_vol

            rows.append(
                {
                    "stock_id": sid,
                    "consec_foreign_days": consec_foreign,
                    "buy_vol_ratio": buy_vol_ratio,
                    "total_net": total_net,
                }
            )

        df = pd.DataFrame(rows)

        consec_rank = df["consec_foreign_days"].rank(pct=True)
        bvr_rank = df["buy_vol_ratio"].rank(pct=True)
        total_rank = df["total_net"].rank(pct=True)

        # ── 券資比因子 ────────────────────────────────────────────────
        has_margin = df_margin is not None and not df_margin.empty
        if has_margin:
            margin_latest = df_margin[df_margin["date"] == df_margin["date"].max()]
            margin_data = margin_latest[margin_latest["stock_id"].isin(stock_ids)][
                ["stock_id", "margin_balance", "short_balance"]
            ].copy()
            if not margin_data.empty:
                margin_data["short_margin_ratio"] = margin_data.apply(
                    lambda r: r["short_balance"] / r["margin_balance"] if r["margin_balance"] > 0 else 0.0,
                    axis=1,
                )
                df = df.merge(margin_data[["stock_id", "short_margin_ratio"]], on="stock_id", how="left")
                df["short_margin_ratio"] = df["short_margin_ratio"].fillna(0.0)
                smr_rank = df["short_margin_ratio"].rank(pct=True)
            else:
                has_margin = False

        # ── 分點集中度因子 ────────────────────────────────────────────
        df_broker_raw = self._load_broker_data(stock_ids)
        broker_df = compute_broker_score(df_broker_raw)
        has_broker = not broker_df.empty
        if has_broker:
            df = df.merge(broker_df, on="stock_id", how="left")
            df["broker_concentration"] = df["broker_concentration"].fillna(0.0)
            df["broker_consecutive_days"] = df["broker_consecutive_days"].fillna(0)
            broker_conc_rank = df["broker_concentration"].rank(pct=True)
            broker_consec_rank = df["broker_consecutive_days"].rank(pct=True)
            broker_rank = broker_conc_rank * 0.60 + broker_consec_rank * 0.40

        if has_margin and has_broker:
            # 5 因子：外資 25% + 量比 22% + 法人 22% + 券資比 16% + 分點 15%
            df["chip_score"] = (
                consec_rank * 0.25 + bvr_rank * 0.22 + total_rank * 0.22 + smr_rank * 0.16 + broker_rank * 0.15
            )
            chip_tier = "5F"
        elif has_margin:
            # 4 因子：外資 30% + 量比 25% + 法人 25% + 券資比 20%
            df["chip_score"] = consec_rank * 0.30 + bvr_rank * 0.25 + total_rank * 0.25 + smr_rank * 0.20
            chip_tier = "4F"
        elif has_broker:
            # 4 因子：外資 32% + 量比 24% + 法人 24% + 分點 20%
            df["chip_score"] = consec_rank * 0.32 + bvr_rank * 0.24 + total_rank * 0.24 + broker_rank * 0.20
            chip_tier = "4F"
        else:
            # 3 因子：外資 40% + 量比 30% + 法人 30%
            df["chip_score"] = consec_rank * 0.40 + bvr_rank * 0.30 + total_rank * 0.30
            chip_tier = "3F"

        df["chip_tier"] = chip_tier
        return df[["stock_id", "chip_score", "chip_tier"]]

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """高成長模式基本面：YoY 40% + 營收加速度 25% + 毛利率加速 20% + EPS 季增率 15%。

        毛利率加速 = 最新季毛利率 - 去年同季毛利率（年對年改善）。
        EPS 季增率 = (EPS 最新季 - EPS 上季) / abs(EPS 上季)。
        財報資料不足時自動降回營收雙因子（YoY 60% + 加速度 40%）。
        """
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        if rev.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
        if "yoy_3m_ago" in rev.columns:
            rev["acceleration"] = rev["yoy_growth"].fillna(0) - rev["yoy_3m_ago"].fillna(0)
            accel_rank = rev["acceleration"].rank(pct=True)
        else:
            accel_rank = pd.Series(0.5, index=rev.index)

        # --- 財報因子 ---
        df_fin = self._load_financial_data(stock_ids, quarters=5)
        if df_fin.empty:
            # 降回純營收雙因子
            rev["fundamental_score"] = yoy_rank * 0.60 + accel_rank * 0.40
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
            return result

        grouped = df_fin.groupby("stock_id", sort=False)
        fin_rows = []
        for sid in stock_ids:
            row: dict = {"stock_id": sid, "gm_accel": None, "eps_qoq": None}
            if sid in grouped.groups:
                grp = grouped.get_group(sid).sort_values("date", ascending=False)
                # 毛利率加速 = 最新季 - 去年同季
                if len(grp) >= 1 and pd.notna(grp.iloc[0]["gross_margin"]):
                    cur_q = int(grp.iloc[0]["quarter"])
                    cur_y = int(grp.iloc[0]["year"])
                    same_q = grp[(grp["quarter"] == cur_q) & (grp["year"] == cur_y - 1)]
                    if not same_q.empty and pd.notna(same_q.iloc[0]["gross_margin"]):
                        row["gm_accel"] = float(grp.iloc[0]["gross_margin"]) - float(same_q.iloc[0]["gross_margin"])
                # EPS 季增率 = (最新 - 上季) / abs(上季)
                if (
                    len(grp) >= 2
                    and pd.notna(grp.iloc[0]["eps"])
                    and pd.notna(grp.iloc[1]["eps"])
                    and abs(float(grp.iloc[1]["eps"])) > 0.01
                ):
                    row["eps_qoq"] = (float(grp.iloc[0]["eps"]) - float(grp.iloc[1]["eps"])) / abs(
                        float(grp.iloc[1]["eps"])
                    )
            fin_rows.append(row)

        df_metrics = pd.DataFrame(fin_rows)
        has_any = df_metrics[["gm_accel", "eps_qoq"]].notna().any(axis=1).any()
        if not has_any:
            rev["fundamental_score"] = yoy_rank * 0.60 + accel_rank * 0.40
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
            return result

        gm_accel_rank = df_metrics["gm_accel"].rank(pct=True).fillna(0.5)
        eps_qoq_rank = df_metrics["eps_qoq"].rank(pct=True).fillna(0.5)

        # 將財報指標合進 rev（以 stock_id 對齊）
        df_metrics = df_metrics.merge(
            rev[["stock_id", "yoy_growth"] + (["acceleration"] if "acceleration" in rev.columns else [])],
            on="stock_id",
            how="left",
        )
        df_metrics["yoy_growth"] = df_metrics["yoy_growth"].fillna(0)
        df_metrics["yoy_rank_val"] = df_metrics["yoy_growth"].rank(pct=True)
        if "acceleration" in df_metrics.columns:
            df_metrics["accel_rank_val"] = df_metrics["acceleration"].rank(pct=True).fillna(0.5)
        else:
            df_metrics["accel_rank_val"] = 0.5

        df_metrics["fundamental_score"] = (
            df_metrics["yoy_rank_val"] * 0.40
            + df_metrics["accel_rank_val"] * 0.25
            + gm_accel_rank * 0.20
            + eps_qoq_rank * 0.15
        )

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(df_metrics[["stock_id", "fundamental_score"]], on="stock_id", how="left")
        return result

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """高成長模式風險過濾：ATR(14)/close > 80th percentile 剔除。"""
        return self._apply_atr_risk_filter(scored, df_price, percentile=80)
