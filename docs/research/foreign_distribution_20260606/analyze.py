"""外資派貨偵測 — 純研究分析（5/29 audit §7.3 item 6）。

假說：外資「派貨」= 持續淨賣超且股價未跌（賣壓中股價硬撐 = 出貨給散戶），
應領先未來負報酬。核心訊號用 institutional_investor（外資淨額，深歷史），
SBL（借券，僅 2026-03+）作近期次要確認。

輸出：分位單調性 + 訊號 vs 對照組 forward 報酬 + t 檢定 + SBL overlay。
純讀 data/stock.db，不寫庫、不改 production。
"""

from __future__ import annotations

import math

import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import DailyPrice, InstitutionalInvestor, SecuritiesLending

SIGNAL_START = "2024-01-01"
SIGNAL_END = "2026-04-30"  # 留 forward runway 到 ~2026-06
MIN_MEDIAN_VOL = 1_000_000  # 流動性門檻（股）：日中位量 ≥ 1000 張
FWD_HORIZONS = [5, 10, 20]


def _load() -> pd.DataFrame:
    with get_session() as s:
        inst = pd.DataFrame(
            s.execute(
                select(InstitutionalInvestor.stock_id, InstitutionalInvestor.date, InstitutionalInvestor.net).where(
                    InstitutionalInvestor.name == "Foreign_Investor",
                    InstitutionalInvestor.date >= SIGNAL_START,
                )
            ).all(),
            columns=["stock_id", "date", "foreign_net"],
        )
        px = pd.DataFrame(
            s.execute(
                select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close, DailyPrice.volume).where(
                    DailyPrice.date >= SIGNAL_START
                )
            ).all(),
            columns=["stock_id", "date", "close", "volume"],
        )
    df = px.merge(inst, on=["stock_id", "date"], how="inner")
    df = df[df["stock_id"].str.fullmatch(r"[1-9]\d{3}")]  # 4 碼普通股，排除 ETF/權證
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["foreign_net"] = pd.to_numeric(df["foreign_net"], errors="coerce")
    return df.dropna(subset=["close", "volume", "foreign_net"])


def _features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("stock_id", group_keys=False)
    df["net5"] = g["foreign_net"].apply(lambda s: s.rolling(5).sum())
    df["vol5"] = g["volume"].apply(lambda s: s.rolling(5).sum())
    df["net_ratio5"] = df["net5"] / df["vol5"].replace(0, math.nan)
    df["pret5"] = g["close"].apply(lambda s: s.pct_change(5))
    # 流動性過濾
    med_vol = df.groupby("stock_id")["volume"].transform("median")
    df = df[med_vol >= MIN_MEDIAN_VOL]
    # forward 報酬：t+1 收盤進場（外資淨額盤後公布，避免 look-ahead），持有 k 日
    g2 = df.groupby("stock_id", group_keys=False)
    entry = g2["close"].apply(lambda s: s.shift(-1))
    for k in FWD_HORIZONS:
        exit_k = g2["close"].apply(lambda s: s.shift(-(1 + k)))
        df[f"fwd{k}"] = exit_k / entry - 1
    return df.dropna(subset=["net_ratio5", "pret5", "fwd5"])


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


def main() -> None:
    df = _features(_load())
    n_obs = len(df)
    n_stocks = df["stock_id"].nunique()
    print("# 外資派貨偵測 — 研究結果\n")
    print(f"- 期間：{SIGNAL_START} ~ {SIGNAL_END}（forward 量到資料末）")
    print(f"- 樣本：{n_obs:,} 個 stock-day，{n_stocks} 檔（日中位量 ≥ {MIN_MEDIAN_VOL:,}）\n")

    # ── 1. net_ratio5 分位單調性（越賣 → forward 越差？）──
    print("## 1. net_ratio5 五分位 vs forward 報酬（單調性）\n")
    df["q"] = pd.qcut(df["net_ratio5"], 5, labels=["Q1賣最多", "Q2", "Q3", "Q4", "Q5買最多"])
    grp = df.groupby("q", observed=True)
    tbl = grp[[f"fwd{k}" for k in FWD_HORIZONS]].mean() * 100
    tbl["n"] = grp.size()
    print(tbl.round(3).to_string())
    print()

    # ── 2. 派貨訊號 vs 對照（賣超 into strength）──
    print("## 2. 派貨訊號（net_ratio5 ≤ -θ 且 pret5 ≥ 0）vs 對照\n")
    print("| θ (net_ratio5) | N | fwd5% | fwd10% | fwd20% | fwd10 勝率 | fwd10 t |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for theta in [0.05, 0.08, 0.10, 0.15]:
        sig = df[(df["net_ratio5"] <= -theta) & (df["pret5"] >= 0)]
        if len(sig) < 20:
            print(f"| {theta:.2f} | {len(sig)} | 樣本不足 |||||")
            continue
        wr = (sig["fwd10"] < 0).mean() * 100  # 派貨 → 預期跌，故看跌的比例
        print(
            f"| {theta:.2f} | {len(sig):,} | {sig['fwd5'].mean() * 100:.2f} | "
            f"{sig['fwd10'].mean() * 100:.2f} | {sig['fwd20'].mean() * 100:.2f} | "
            f"{wr:.1f}% | {_tstat(sig['fwd10']):.2f} |"
        )
    base = df
    print(
        f"| (全體對照) | {len(base):,} | {base['fwd5'].mean() * 100:.2f} | "
        f"{base['fwd10'].mean() * 100:.2f} | {base['fwd20'].mean() * 100:.2f} | "
        f"{(base['fwd10'] < 0).mean() * 100:.1f}% | {_tstat(base['fwd10']):.2f} |"
    )
    print()

    # ── 3. 拆解：純賣超 vs 賣超+背離（背離是否加值）──
    print("## 3. 背離條件是否加值（θ=0.08）\n")
    theta = 0.08
    plain = df[df["net_ratio5"] <= -theta]  # 純賣超
    diverge = df[(df["net_ratio5"] <= -theta) & (df["pret5"] >= 0)]  # 賣超 + 股價未跌
    falling = df[(df["net_ratio5"] <= -theta) & (df["pret5"] < 0)]  # 賣超 + 股價已跌
    for label, sub in [("純賣超", plain), ("賣超+背離(股價未跌)", diverge), ("賣超+股價已跌", falling)]:
        if len(sub) < 20:
            print(f"- {label}: 樣本不足 ({len(sub)})")
            continue
        print(
            f"- {label}: N={len(sub):,}  fwd10={sub['fwd10'].mean() * 100:.2f}%  "
            f"跌比例={(sub['fwd10'] < 0).mean() * 100:.1f}%  t={_tstat(sub['fwd10']):.2f}"
        )
    print()

    # ── 4. SBL overlay（近期 2026-03+，借券餘額上升是否加強）──
    # 注意：dev DB 的 sbl_change 欄 100% NULL，僅 sbl_balance 有值 → 由餘額逐股 diff 自推。
    print("## 4. SBL overlay（2026-03+，借券餘額增 = 空壓；change 由 balance 自推）\n")
    with get_session() as s:
        sbl = pd.DataFrame(
            s.execute(
                select(SecuritiesLending.stock_id, SecuritiesLending.date, SecuritiesLending.sbl_balance).where(
                    SecuritiesLending.date >= "2026-03-11"
                )
            ).all(),
            columns=["stock_id", "date", "sbl_balance"],
        )
    sbl = sbl.sort_values(["stock_id", "date"])
    sbl["sbl_change"] = sbl.groupby("stock_id", group_keys=False)["sbl_balance"].apply(lambda s: s.diff())
    recent = df.merge(sbl[["stock_id", "date", "sbl_change"]], on=["stock_id", "date"], how="inner")
    recent = recent.dropna(subset=["sbl_change"])
    if len(recent) < 50:
        print(f"- SBL 重疊樣本過少（{len(recent)}），近期 forward runway 也短，僅供參考。")
    else:
        sig_sbl = recent[(recent["net_ratio5"] <= -0.08) & (recent["pret5"] >= 0) & (recent["sbl_change"] > 0)]
        sig_nosbl = recent[(recent["net_ratio5"] <= -0.08) & (recent["pret5"] >= 0) & (recent["sbl_change"] <= 0)]
        for label, sub in [("派貨+借券增", sig_sbl), ("派貨+借券未增", sig_nosbl)]:
            if len(sub) < 10:
                print(f"- {label}: 樣本不足 ({len(sub)})")
                continue
            print(
                f"- {label}: N={len(sub)}  fwd5={sub['fwd5'].mean() * 100:.2f}%  fwd10={sub['fwd10'].mean() * 100:.2f}%"
            )
    print()


if __name__ == "__main__":
    main()
