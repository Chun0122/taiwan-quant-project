"""CLI 異動偵測/營收掃描/宏觀壓力 子命令。"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.cli.helpers import init_db
from src.cli.helpers import safe_print as print
from src.constants import (
    DEFAULT_DT_THRESHOLD,
    DEFAULT_HHI_THRESHOLD,
    DEFAULT_INST_THRESHOLD,
    DEFAULT_SBL_SIGMA,
    DEFAULT_VOL_MULT,
)


def _compute_macro_stress_check() -> dict:
    """宏觀壓力預檢：從 DB 讀取 TAIEX 計算快速崩盤訊號（純函數）。

    Returns:
        dict: {
            "regime": str,               # "crisis" | "bear" | "sideways" | "bull"
            "crisis_triggered": bool,
            "taiex_close": float,
            "fast_return_5d": float,     # 5 日報酬率
            "consec_decline_days": int,  # 實際連跌天數
            "vol_ratio": float,          # 波動率倍數
            "signals": dict,             # 三個 bool flags
            "summary": str,              # 中文摘要，用於 Discord / 終端輸出
        }
    """
    import pandas as pd
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import DailyPrice
    from src.regime.detector import MarketRegimeDetector, detect_crisis_signals

    try:
        with get_session() as session:
            rows = session.execute(
                select(DailyPrice.date, DailyPrice.close, DailyPrice.volume)
                .where(DailyPrice.stock_id == "TAIEX")
                .order_by(DailyPrice.date.desc())
                .limit(130)
            ).all()

            # 載入 VIX 資料（graceful degradation）
            vix_rows = session.execute(
                select(DailyPrice.date, DailyPrice.close)
                .where(DailyPrice.stock_id == "TW_VIX")
                .order_by(DailyPrice.date.desc())
                .limit(30)
            ).all()

            # 載入美國 VIX 資料（graceful degradation）
            us_vix_rows = session.execute(
                select(DailyPrice.date, DailyPrice.close)
                .where(DailyPrice.stock_id == "US_VIX")
                .order_by(DailyPrice.date.desc())
                .limit(30)
            ).all()

        if not rows or len(rows) < 10:
            return {
                "regime": "sideways",
                "crisis_triggered": False,
                "taiex_close": 0.0,
                "fast_return_5d": 0.0,
                "consec_decline_days": 0,
                "vol_ratio": 0.0,
                "vix_val": 0.0,
                "us_vix_val": 0.0,
                "signals": {},
                "summary": "TAIEX 資料不足，跳過壓力檢查",
                "breadth_below_ma20_pct": None,
                "breadth_downgraded": False,
            }

        rows_sorted = sorted(rows, key=lambda r: r[0])
        closes = pd.Series([float(r[1]) for r in rows_sorted])
        volumes_raw = pd.Series([float(r[2]) for r in rows_sorted])
        volumes = volumes_raw if (volumes_raw > 0).any() else None

        # VIX 序列（由舊至新）
        vix_series = None
        if vix_rows:
            vix_sorted = sorted(vix_rows, key=lambda r: r[0])
            vix_series = pd.Series([float(r[1]) for r in vix_sorted])

        # 美國 VIX 序列（由舊至新）
        us_vix_series = None
        if us_vix_rows:
            us_vix_sorted = sorted(us_vix_rows, key=lambda r: r[0])
            us_vix_series = pd.Series([float(r[1]) for r in us_vix_sorted])

        crisis_info = detect_crisis_signals(
            closes,
            volumes=volumes,
            vix_series=vix_series,
            us_vix_series=us_vix_series,
        )
        regime_info = MarketRegimeDetector().detect()
        regime = regime_info.get("regime", "sideways")

        # 計算實際連跌天數（顯示用）
        consec = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes.iloc[i] < closes.iloc[i - 1]:
                consec += 1
            else:
                break

        taiex_close = float(closes.iloc[-1])
        ret_5d = float((closes.iloc[-1] - closes.iloc[-6]) / closes.iloc[-6]) if len(closes) >= 6 else 0.0

        breadth_pct = regime_info.get("breadth_below_ma20_pct")
        breadth_downgraded = regime_info.get("breadth_downgraded", False)
        vix_val = crisis_info.get("vix_val", 0.0)

        us_vix_val = crisis_info.get("us_vix_val", 0.0)

        if regime == "crisis":
            panic_tag = "，爆量長黑" if crisis_info.get("signals", {}).get("panic_volume") else ""
            vix_tag = f"，TW_VIX={vix_val:.1f}" if crisis_info.get("signals", {}).get("vix_spike") else ""
            us_vix_tag = f"，US_VIX={us_vix_val:.1f}" if crisis_info.get("signals", {}).get("us_vix_spike") else ""
            drop_tag = "，單日急跌" if crisis_info.get("signals", {}).get("single_day_drop") else ""
            summary = (
                f"⚠ CRISIS 崩盤訊號觸發！"
                f"5日={ret_5d:+.1%}，連跌{consec}天，波動率={crisis_info.get('vol_ratio_val', 0.0):.1f}x"
                f"{panic_tag}{vix_tag}{us_vix_tag}{drop_tag}"
            )
        elif regime == "bear":
            breadth_tag = f"，MA20寬度={breadth_pct:.0%}" if breadth_pct is not None else ""
            summary = f"空頭市場 TAIEX={taiex_close:.0f}，5日={ret_5d:+.1%}{breadth_tag}"
        else:
            breadth_tag = f"，MA20寬度={breadth_pct:.0%}" if breadth_pct is not None else ""
            summary = f"市場狀態={regime} TAIEX={taiex_close:.0f}，5日={ret_5d:+.1%}{breadth_tag}"

        return {
            "regime": regime,
            "crisis_triggered": crisis_info.get("crisis", False),
            "taiex_close": taiex_close,
            "fast_return_5d": ret_5d,
            "consec_decline_days": consec,
            "vol_ratio": crisis_info.get("vol_ratio_val", 0.0),
            "vix_val": vix_val,
            "us_vix_val": us_vix_val,
            "signals": crisis_info.get("signals", {}),
            "summary": summary,
            "breadth_below_ma20_pct": breadth_pct,
            "breadth_downgraded": breadth_downgraded,
        }

    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        logging.warning("宏觀壓力預檢失敗: %s", exc)
        return {
            "regime": "sideways",
            "crisis_triggered": False,
            "taiex_close": 0.0,
            "fast_return_5d": 0.0,
            "consec_decline_days": 0,
            "vol_ratio": 0.0,
            "signals": {},
            "summary": "壓力預檢失敗，跳過",
            "breadth_below_ma20_pct": None,
            "breadth_downgraded": False,
        }


def _compute_revenue_scan(
    watchlist: list[str],
    min_yoy: float,
    min_margin_improve: float,
) -> "pd.DataFrame":
    """掃描 watchlist 中 YoY 高成長 + 毛利率改善的個股（純函數）。

    Args:
        watchlist:           要掃描的股票代號清單
        min_yoy:             最低 YoY 門檻（%，例 10.0）
        min_margin_improve:  毛利率 QoQ 最低改善幅度（百分點，例 0.0 = 正向即可）

    Returns:
        DataFrame 含欄位：stock_id, yoy_growth, mom_growth, gross_margin,
                          margin_change, revenue_rank
    """
    import pandas as pd
    from sqlalchemy import select

    from src.data.database import get_session
    from src.data.schema import FinancialStatement, MonthlyRevenue

    # ── 月營收（取每支股票最新一筆）──────────────────────────────────
    with get_session() as session:
        rev_rows = session.execute(
            select(
                MonthlyRevenue.stock_id,
                MonthlyRevenue.date,
                MonthlyRevenue.revenue,
                MonthlyRevenue.yoy_growth,
                MonthlyRevenue.mom_growth,
            ).where(MonthlyRevenue.stock_id.in_(watchlist))
        ).all()

    if not rev_rows:
        return pd.DataFrame()

    df_rev = pd.DataFrame(rev_rows, columns=["stock_id", "date", "revenue", "yoy_growth", "mom_growth"])
    # 每支股票只取最新一筆
    df_rev = df_rev.sort_values("date").groupby("stock_id", sort=False).last().reset_index()

    # ── 財報毛利率（取最新兩季，計算 QoQ 趨勢）─────────────────────
    with get_session() as session:
        fin_rows = session.execute(
            select(
                FinancialStatement.stock_id,
                FinancialStatement.date,
                FinancialStatement.gross_margin,
            )
            .where(FinancialStatement.stock_id.in_(watchlist))
            .order_by(FinancialStatement.stock_id, FinancialStatement.date.desc())
        ).all()

    df_fin = pd.DataFrame(fin_rows, columns=["stock_id", "date", "gross_margin"])
    df_fin = df_fin.dropna(subset=["gross_margin"])

    # 計算毛利率 QoQ 變化（最新季 - 前一季）
    margin_rows = []
    for sid, grp in df_fin.groupby("stock_id", sort=False):
        grp = grp.sort_values("date", ascending=False)
        latest_gm = grp["gross_margin"].iloc[0] if len(grp) >= 1 else None
        prev_gm = grp["gross_margin"].iloc[1] if len(grp) >= 2 else None
        margin_change = (latest_gm - prev_gm) if (latest_gm is not None and prev_gm is not None) else None
        margin_rows.append({"stock_id": sid, "gross_margin": latest_gm, "margin_change": margin_change})

    df_margin = (
        pd.DataFrame(margin_rows)
        if margin_rows
        else pd.DataFrame(columns=["stock_id", "gross_margin", "margin_change"])
    )

    # ── 合併 + 篩選 ───────────────────────────────────────────────────
    df = df_rev.merge(df_margin, on="stock_id", how="left")
    df["yoy_growth"] = pd.to_numeric(df["yoy_growth"], errors="coerce")
    df["mom_growth"] = pd.to_numeric(df["mom_growth"], errors="coerce")

    # 條件篩選
    mask = df["yoy_growth"] >= min_yoy
    if min_margin_improve is not None:
        mask_margin = df["margin_change"].isna() | (df["margin_change"] >= min_margin_improve)
        mask = mask & mask_margin

    df = df[mask].copy()
    if df.empty:
        return df

    # 排名分數：YoY 70% + 毛利率改善 30%
    df["yoy_rank"] = df["yoy_growth"].rank(pct=True)
    df["margin_rank"] = df["margin_change"].fillna(0).rank(pct=True)
    df["revenue_rank"] = df["yoy_rank"] * 0.70 + df["margin_rank"] * 0.30
    df = df.sort_values("revenue_rank", ascending=False)

    return df[["stock_id", "yoy_growth", "mom_growth", "gross_margin", "margin_change", "revenue_rank"]]


# ═══════════════════════════════════════════════════════════════════════════
# P5 籌碼異動警報 — 四個純函數（零 mock 可測試）
# ═══════════════════════════════════════════════════════════════════════════


# 籌碼異動偵測純函數 — 已移至 src/cli/detection.py（D1 CLI 模組化）
from src.cli.detection import (  # noqa: E402
    detect_broker_concentration,
    detect_daytrade_risk,
    detect_institutional_buy,
    detect_sbl_spike,
    detect_volume_spike,
)


def _compute_anomaly_scan(
    watchlist: list[str],
    lookback: int = 10,
    vol_mult: float = DEFAULT_VOL_MULT,
    inst_threshold: float = DEFAULT_INST_THRESHOLD,
    sbl_sigma: float = DEFAULT_SBL_SIGMA,
    hhi_threshold: float = DEFAULT_HHI_THRESHOLD,
    dt_threshold: float = DEFAULT_DT_THRESHOLD,
) -> "dict[str, pd.DataFrame]":
    """從 DB 讀取五類資料，呼叫五個純函數，回傳異常偵測結果。

    Keys: "volume_spike", "inst_buy", "sbl_spike", "broker_conc", "daytrade_risk"
    各值為 DataFrame；無資料時為空 DataFrame。
    """
    import datetime

    import pandas as pd
    from sqlalchemy import select
    from sqlalchemy.exc import OperationalError, ProgrammingError

    from src.data.database import get_session
    from src.data.schema import BrokerTrade, DailyPrice, InstitutionalInvestor, SecuritiesLending

    cutoff = datetime.date.today() - datetime.timedelta(days=lookback + 5)
    inst_cutoff = datetime.date.today() - datetime.timedelta(days=5)
    broker_cutoff = datetime.date.today() - datetime.timedelta(days=max(20, lookback + 5))

    # 合併為單一 DB session，減少 4 次額外的連線開關開銷
    df_price = pd.DataFrame()
    df_inst = pd.DataFrame()
    df_sbl = pd.DataFrame()
    df_broker = pd.DataFrame()
    try:
        with get_session() as session:
            # A. 量能暴增
            price_rows = session.execute(
                select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.volume).where(
                    DailyPrice.stock_id.in_(watchlist),
                    DailyPrice.date >= cutoff,
                )
            ).all()
            if price_rows:
                df_price = pd.DataFrame(price_rows, columns=["stock_id", "date", "volume"])

            # B. 外資大買超
            inst_rows = session.execute(
                select(
                    InstitutionalInvestor.stock_id,
                    InstitutionalInvestor.date,
                    InstitutionalInvestor.name,
                    InstitutionalInvestor.net,
                ).where(
                    InstitutionalInvestor.stock_id.in_(watchlist),
                    InstitutionalInvestor.date >= inst_cutoff,
                )
            ).all()
            if inst_rows:
                df_inst = pd.DataFrame(inst_rows, columns=["stock_id", "date", "name", "net"])

            # C. 借券賣出激增
            sbl_rows = session.execute(
                select(
                    SecuritiesLending.stock_id,
                    SecuritiesLending.date,
                    SecuritiesLending.sbl_change,
                ).where(
                    SecuritiesLending.stock_id.in_(watchlist),
                    SecuritiesLending.date >= cutoff,
                )
            ).all()
            if sbl_rows:
                df_sbl = pd.DataFrame(sbl_rows, columns=["stock_id", "date", "sbl_change"])

            # D. 主力分點集中買進 + E. 隔日沖偵測
            broker_rows = session.execute(
                select(
                    BrokerTrade.stock_id,
                    BrokerTrade.date,
                    BrokerTrade.broker_id,
                    BrokerTrade.broker_name,
                    BrokerTrade.buy,
                    BrokerTrade.sell,
                ).where(
                    BrokerTrade.stock_id.in_(watchlist),
                    BrokerTrade.date >= broker_cutoff,
                )
            ).all()
            if broker_rows:
                df_broker = pd.DataFrame(
                    broker_rows, columns=["stock_id", "date", "broker_id", "broker_name", "buy", "sell"]
                )
    except (OperationalError, ProgrammingError) as exc:
        logging.warning("anomaly_scan DB 查詢失敗: %s", exc)

    # 計算 20 日均量供隔日沖流動性閾值使用
    df_vol_for_dt = None
    if not df_price.empty:
        avg_vol = (
            df_price.sort_values("date").groupby("stock_id")["volume"].apply(lambda s: s.tail(20).mean()).reset_index()
        )
        avg_vol.columns = ["stock_id", "avg_volume_20d"]
        df_vol_for_dt = avg_vol

    return {
        "volume_spike": detect_volume_spike(df_price, lookback=lookback, threshold=vol_mult),
        "inst_buy": detect_institutional_buy(df_inst, threshold=inst_threshold),
        "sbl_spike": detect_sbl_spike(df_sbl, lookback=lookback, sigma=sbl_sigma),
        "broker_conc": detect_broker_concentration(df_broker, hhi_threshold=hhi_threshold),
        "daytrade_risk": detect_daytrade_risk(df_broker, df_volume=df_vol_for_dt, penalty_threshold=dt_threshold),
    }


def cmd_anomaly_scan(args: argparse.Namespace) -> None:
    """掃描 watchlist 中成交量/籌碼異動的即時警報。

    偵測五類異常：量能暴增、外資大買超、借券賣出激增、主力分點集中買進、隔日沖風險。
    資料直接從 DB 讀取（需先執行 sync / sync-sbl / sync-broker）。
    """
    import datetime

    init_db()

    from src.data.database import get_effective_watchlist

    watchlist = args.stocks if args.stocks else get_effective_watchlist()
    lookback: int = getattr(args, "lookback", 10)
    vol_mult: float = getattr(args, "vol_mult", DEFAULT_VOL_MULT)
    inst_threshold: float = getattr(args, "inst_threshold", DEFAULT_INST_THRESHOLD)
    sbl_sigma: float = getattr(args, "sbl_sigma", DEFAULT_SBL_SIGMA)
    hhi_threshold: float = getattr(args, "hhi_threshold", DEFAULT_HHI_THRESHOLD)
    dt_threshold: float = getattr(args, "dt_threshold", DEFAULT_DT_THRESHOLD)
    notify: bool = getattr(args, "notify", False)

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    print(f"\n掃描 {len(watchlist)} 支股票的籌碼異動（{today_str}）...")

    results = _compute_anomaly_scan(
        watchlist=watchlist,
        lookback=lookback,
        vol_mult=vol_mult,
        inst_threshold=inst_threshold,
        sbl_sigma=sbl_sigma,
        hhi_threshold=hhi_threshold,
        dt_threshold=dt_threshold,
    )

    df_vol = results["volume_spike"]
    df_inst = results["inst_buy"]
    df_sbl = results["sbl_spike"]
    df_broker = results["broker_conc"]
    df_dt = results["daytrade_risk"]

    total = len(df_vol) + len(df_inst) + len(df_sbl) + len(df_broker) + len(df_dt)
    print(f"\n=== 籌碼異動警報（{today_str}，共 {total} 筆）===")

    if not df_vol.empty:
        print(f"\n[量增] 量能暴增（今日量 > {lookback}MA x {vol_mult}x，共 {len(df_vol)} 支）")
        for _, row in df_vol.iterrows():
            today_lot = int(row["today_vol"]) // 1000
            avg_lot = int(row["avg_vol"]) // 1000
            print(f"  {row['stock_id']}  今日量 {today_lot:,} 張  均量 {avg_lot:,} 張  倍率 {row['vol_ratio']:.2f}x")
    else:
        print(f"\n[量增] 量能暴增 -- 無（門檻: > {lookback}MA x {vol_mult}x）")

    if not df_inst.empty:
        thresh_lot = int(inst_threshold) // 1000
        print(f"\n[外資] 外資大買超（淨買超 > {thresh_lot:,} 張，共 {len(df_inst)} 支）")
        for _, row in df_inst.iterrows():
            lots = int(row["inst_net"]) // 1000
            print(f"  {row['stock_id']}  外資淨買 +{lots:,} 張（+{int(row['inst_net']):,} 股）")
    else:
        print(f"\n[外資] 外資大買超 -- 無（門檻: > {int(inst_threshold) // 1000:,} 張）")

    if not df_sbl.empty:
        print(f"\n[借券] 借券賣出激增（sbl_change > mean + {sbl_sigma}x std，共 {len(df_sbl)} 支）")
        for _, row in df_sbl.iterrows():
            chg_lot = int(row["sbl_change"]) // 1000
            print(
                f"  {row['stock_id']}  借券增加 +{chg_lot:,} 張（均值 {row['sbl_mean']:.0f}  std {row['sbl_std']:.0f}）"
            )
    else:
        print(f"\n[借券] 借券賣出激增 -- 無（門檻: > mean + {sbl_sigma}x std）")

    if not df_broker.empty:
        print(f"\n[主力] 主力分點集中買進（HHI > {hhi_threshold:.2f}，共 {len(df_broker)} 支）")
        for _, row in df_broker.iterrows():
            net_lot = int(row["net_buy_total"]) // 1000
            print(f"  {row['stock_id']}  HHI={row['broker_hhi']:.3f}  淨買超 +{net_lot:,} 張")
    else:
        print(f"\n[主力] 主力分點集中買進 -- 無（門檻: HHI > {hhi_threshold:.2f}）")

    if not df_dt.empty:
        print(f"\n[隔沖] 隔日沖風險（penalty > {dt_threshold:.1f}，共 {len(df_dt)} 支）")
        for _, row in df_dt.iterrows():
            tags = row.get("top_dt_brokers", "")
            print(f"  {row['stock_id']}  penalty={row['daytrade_penalty']:.2f}  分點: {tags}")
    else:
        print(f"\n[隔沖] 隔日沖風險 -- 無（門檻: penalty > {dt_threshold:.1f}）")

    if notify:
        from src.notification.line_notify import send_message

        lines = [f"📡 **籌碼異動警報** ({today_str})，共 {total} 筆"]
        if not df_vol.empty:
            lines.append(f"\n📊 量能暴增 ({len(df_vol)} 支)")
            for _, row in df_vol.head(5).iterrows():
                lots = int(row["today_vol"]) // 1000
                lines.append(f"  {row['stock_id']} 今日 {lots:,}張 倍率 {row['vol_ratio']:.1f}x")
        if not df_inst.empty:
            lines.append(f"\n🏦 外資大買超 ({len(df_inst)} 支)")
            for _, row in df_inst.head(5).iterrows():
                lots = int(row["inst_net"]) // 1000
                lines.append(f"  {row['stock_id']} +{lots:,}張")
        if not df_sbl.empty:
            lines.append(f"\n🔴 借券激增 ({len(df_sbl)} 支)")
            for _, row in df_sbl.head(3).iterrows():
                lots = int(row["sbl_change"]) // 1000
                lines.append(f"  {row['stock_id']} +{lots:,}張")
        if not df_broker.empty:
            lines.append(f"\n🎯 主力集中買進 ({len(df_broker)} 支)")
            for _, row in df_broker.head(3).iterrows():
                net_lot = int(row["net_buy_total"]) // 1000
                lines.append(f"  {row['stock_id']} HHI={row['broker_hhi']:.2f} +{net_lot:,}張")
        if total == 0:
            lines.append("（今日無異常訊號）")
        ok = send_message("\n".join(lines[:40]))
        print(f"\nDiscord 通知: {'成功' if ok else '失敗（請確認 Webhook 設定）'}")


def cmd_revenue_scan(args: argparse.Namespace) -> None:
    """掃描 watchlist 中 YoY 高成長 + 毛利率改善的個股。

    資料直接從 DB 讀取（需先執行 sync-revenue + sync-financial）。
    """
    import pandas as pd

    init_db()

    from src.data.database import get_effective_watchlist

    watchlist = args.stocks if args.stocks else get_effective_watchlist()
    top_n = getattr(args, "top", 20)
    min_yoy = getattr(args, "min_yoy", 10.0)
    min_margin = getattr(args, "min_margin_improve", 0.0)
    notify = getattr(args, "notify", False)

    print(f"掃描 {len(watchlist)} 支股票（YoY >= {min_yoy}%，毛利率 QoQ >= {min_margin:.1f} pp）...")

    df = _compute_revenue_scan(watchlist, min_yoy=min_yoy, min_margin_improve=min_margin)

    if df.empty:
        print("無符合條件的個股")
        return

    df_show = df.head(top_n).copy()
    print(f"\n=== 營收成長掃描結果 Top {min(top_n, len(df_show))} ===\n")
    print(f"{'排名':>4}  {'股票':>6}  {'YoY%':>7}  {'MoM%':>7}  {'毛利率%':>8}  {'毛利率 QoQ':>10}")
    print("-" * 55)
    for rank, (_, row) in enumerate(df_show.iterrows(), 1):
        yoy = f"{row['yoy_growth']:+.1f}%" if pd.notna(row["yoy_growth"]) else "  N/A "
        mom = f"{row['mom_growth']:+.1f}%" if pd.notna(row["mom_growth"]) else "  N/A "
        gm = f"{row['gross_margin']:.1f}%" if pd.notna(row["gross_margin"]) else "  N/A "
        mc = f"{row['margin_change']:+.1f}pp" if pd.notna(row["margin_change"]) else "   N/A  "
        print(f"{rank:>4}  {row['stock_id']:>6}  {yoy:>7}  {mom:>7}  {gm:>8}  {mc:>10}")

    print(f"\n共 {len(df)} 支符合條件（{len(watchlist)} 支中）")

    if notify:
        from src.notification.line_notify import send_message

        lines = [f"📈 營收成長掃描（YoY≥{min_yoy:.0f}%，共 {len(df)} 支）"]
        for rank, (_, row) in enumerate(df_show.iterrows(), 1):
            yoy = f"{row['yoy_growth']:+.1f}%" if pd.notna(row["yoy_growth"]) else "N/A"
            mc = f"{row['margin_change']:+.1f}pp" if pd.notna(row["margin_change"]) else "N/A"
            lines.append(f"#{rank} {row['stock_id']} YoY={yoy} 毛利率QoQ={mc}")
        ok = send_message("\n".join(lines[:30]))
        print(f"\nDiscord 通知: {'成功' if ok else '失敗（請確認 Webhook 設定）'}")
