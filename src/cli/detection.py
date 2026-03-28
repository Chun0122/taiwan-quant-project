"""籌碼異動偵測純函數 — 從 main.py 抽出，供 anomaly-scan / morning-routine 使用。

所有函數為純函數（無 DB / IO 副作用），接受 DataFrame 輸入，回傳 DataFrame 結果。
"""

from __future__ import annotations

import pandas as pd


def detect_volume_spike(
    df_price: pd.DataFrame,
    lookback: int = 10,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """量能暴增偵測：今日量 > 近 lookback 天均量 × threshold。

    輸入欄位: stock_id, date, volume（股）
    輸出欄位: stock_id, today_vol, avg_vol, vol_ratio
    需至少 2 天資料（1 天歷史 + 1 天今日）；不足回傳空 DataFrame。
    """
    empty = pd.DataFrame(columns=["stock_id", "today_vol", "avg_vol", "vol_ratio"])
    if df_price.empty:
        return empty

    results = []
    sorted_price = df_price.sort_values(["stock_id", "date"])
    for stock_id, grp in sorted_price.groupby("stock_id", sort=False):
        latest_date = grp["date"].iloc[-1]
        today_row = grp[grp["date"] == latest_date]
        hist = grp[grp["date"] < latest_date].tail(lookback)
        if hist.empty or today_row.empty:
            continue
        today_vol = int(today_row["volume"].iloc[0])
        avg_vol = float(hist["volume"].mean())
        if avg_vol <= 0:
            continue
        ratio = today_vol / avg_vol
        if ratio >= threshold:
            results.append(
                {
                    "stock_id": stock_id,
                    "today_vol": today_vol,
                    "avg_vol": round(avg_vol),
                    "vol_ratio": round(ratio, 2),
                }
            )

    if not results:
        return empty
    return pd.DataFrame(results).sort_values("vol_ratio", ascending=False).reset_index(drop=True)


def detect_institutional_buy(
    df_inst: pd.DataFrame,
    threshold: float = 3_000_000,
) -> pd.DataFrame:
    """外資大買超偵測：最新日外資 net > threshold（股）。

    輸入欄位: stock_id, date, name, net
    name 用 str.contains("外資") 篩選。
    輸出欄位: stock_id, inst_net（股）
    """
    empty = pd.DataFrame(columns=["stock_id", "inst_net"])
    if df_inst.empty:
        return empty

    foreign = df_inst[df_inst["name"].str.contains("外資", na=False)].copy()
    if foreign.empty:
        return empty

    latest_date = foreign["date"].max()
    today_foreign = foreign[foreign["date"] == latest_date]

    summed = today_foreign.groupby("stock_id")["net"].sum().reset_index()
    summed.columns = ["stock_id", "inst_net"]

    result = summed[summed["inst_net"] > threshold]
    return result.sort_values("inst_net", ascending=False).reset_index(drop=True)


def detect_sbl_spike(
    df_sbl: pd.DataFrame,
    lookback: int = 10,
    sigma: float = 2.0,
) -> pd.DataFrame:
    """借券賣出激增偵測：最新日 sbl_change > mean + sigma × std。

    需至少 3 筆歷史資料，只偵測 sbl_change > 0 的情況。
    輸入欄位: stock_id, date, sbl_change
    輸出欄位: stock_id, sbl_change, sbl_mean, sbl_std
    """
    empty = pd.DataFrame(columns=["stock_id", "sbl_change", "sbl_mean", "sbl_std"])
    if df_sbl.empty:
        return empty

    results = []
    for stock_id, grp in df_sbl.groupby("stock_id"):
        grp = grp.sort_values("date")
        if len(grp) < 3:
            continue
        latest_date = grp["date"].max()
        today_row = grp[grp["date"] == latest_date]
        if today_row.empty:
            continue
        today_change = today_row["sbl_change"].iloc[0]
        if pd.isna(today_change):
            continue
        today_change = float(today_change)

        hist_changes = grp[grp["date"] < latest_date]["sbl_change"].dropna()
        if len(hist_changes) < 2:
            continue
        mean_c = float(hist_changes.tail(lookback).mean())
        std_c = float(hist_changes.tail(lookback).std())
        if std_c <= 0:
            continue
        z = (today_change - mean_c) / std_c
        if today_change > 0 and z >= sigma:
            results.append(
                {
                    "stock_id": stock_id,
                    "sbl_change": int(today_change),
                    "sbl_mean": round(mean_c, 1),
                    "sbl_std": round(std_c, 1),
                }
            )

    if not results:
        return empty
    return pd.DataFrame(results).sort_values("sbl_change", ascending=False).reset_index(drop=True)


def detect_broker_concentration(
    df_broker: pd.DataFrame,
    hhi_threshold: float = 0.4,
) -> pd.DataFrame:
    """主力分點集中買進：最新日 HHI(淨買超分點) > hhi_threshold AND 總淨買 > 0。

    輸入欄位: stock_id, date, broker_id, buy, sell
    輸出欄位: stock_id, broker_hhi, net_buy_total（股）
    """
    empty = pd.DataFrame(columns=["stock_id", "broker_hhi", "net_buy_total"])
    if df_broker.empty:
        return empty

    latest_date = df_broker["date"].max()
    today = df_broker[df_broker["date"] == latest_date].copy()
    today["net"] = (today["buy"].fillna(0) - today["sell"].fillna(0)).astype(int)

    results = []
    for stock_id, grp in today.groupby("stock_id"):
        net_buy_total = int(grp["net"].sum())
        if net_buy_total <= 0:
            continue
        buyers = grp[grp["net"] > 0]
        if buyers.empty:
            continue
        total = buyers["net"].sum()
        shares = buyers["net"] / total
        hhi = float((shares**2).sum())
        if hhi >= hhi_threshold:
            results.append(
                {
                    "stock_id": stock_id,
                    "broker_hhi": round(hhi, 3),
                    "net_buy_total": net_buy_total,
                }
            )

    if not results:
        return empty
    return pd.DataFrame(results).sort_values("broker_hhi", ascending=False).reset_index(drop=True)


def detect_daytrade_risk(
    df_broker: pd.DataFrame,
    df_volume: pd.DataFrame | None = None,
    penalty_threshold: float = 0.3,
) -> pd.DataFrame:
    """偵測隔日沖風險超過門檻的股票。

    Args:
        df_broker: BrokerTrade DataFrame [stock_id, date, broker_id, broker_name, buy, sell]
        df_volume: 各股 20 日均量 DataFrame [stock_id, avg_volume_20d]（可選）
        penalty_threshold: 觸發門檻（預設 0.3）

    Returns:
        DataFrame [stock_id, daytrade_penalty, top_dt_brokers]
    """
    from src.discovery.scanner import compute_daytrade_penalty

    empty = pd.DataFrame(columns=["stock_id", "daytrade_penalty", "top_dt_brokers"])
    if df_broker.empty:
        return empty

    result = compute_daytrade_penalty(df_broker, df_volume=df_volume)
    if result.empty:
        return empty

    triggered = result[result["daytrade_penalty"] >= penalty_threshold].copy()
    if triggered.empty:
        return empty

    return (
        triggered[["stock_id", "daytrade_penalty", "top_dt_brokers"]]
        .sort_values("daytrade_penalty", ascending=False)
        .reset_index(drop=True)
    )
