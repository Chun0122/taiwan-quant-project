"""MomentumScanner — 短線動能掃描模式。

突破 + 資金流 + 量能擴張，適合 1~10 天短線操作。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import HoldingDistribution
from src.discovery.scanner._base import MarketScanner
from src.discovery.scanner._functions import (
    compute_broker_score,
    compute_hhi_trend,
    compute_inst_net_buy_slope,
    compute_institutional_persistence,
    compute_revenue_acceleration_score,
    compute_sbl_score,
    compute_smart_broker_score,
    compute_value_weighted_inst_flow,
    compute_whale_score,
)
from src.discovery.universe import UniverseConfig

logger = logging.getLogger(__name__)


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
        kwargs.setdefault("lookback_days", 80)  # F3 季線突破需 60 交易日（80 曆日）
        kwargs.setdefault(
            "universe_config",
            UniverseConfig(
                min_close=5.0, min_available_days=30, volume_ratio_min=None, trend_filter_mode="trend_or_breakout"
            ),
        )
        super().__init__(**kwargs)

    def _coarse_filter(self, df_price: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
        """動能模式粗篩：基本過濾 + 流動性 + 動能/法人/成交量加權。"""
        filtered = self._base_filter(df_price)
        if filtered.empty:
            return pd.DataFrame()

        # 額外流動性過濾：成交量 > 20 日均量 × 0.5
        vol_mean = df_price.groupby("stock_id")["volume"].apply(lambda s: s.tail(20).mean()).reset_index()
        vol_mean.columns = ["stock_id", "avg_vol_20"]
        filtered = filtered.merge(vol_mean, on="stock_id", how="left")
        filtered["avg_vol_20"] = filtered["avg_vol_20"].fillna(0)
        filtered = filtered[filtered["volume"] > filtered["avg_vol_20"] * 0.5].copy()

        if filtered.empty:
            return pd.DataFrame()

        # 1) 成交量排名
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 2) 法人 5 日累積淨買超排名
        filtered = self._compute_inst_net_buy(df_inst, filtered, days=5)

        # 3) 短期動能（5 日報酬）+ mom_rank
        filtered = self._compute_momentum_5d(df_price, filtered)
        if (filtered["momentum"] != 0).any():
            filtered["mom_rank"] = filtered["momentum"].rank(pct=True)
        else:
            filtered["mom_rank"] = 0.5

        return self._finalize_coarse(filtered)

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
        """動能模式籌碼面：外資連續買超 + 買超佔量比 + 三大法人合計 + 法人連續性 + 券資比 + 大戶持股 + 借券 + 分點（有資料時）。

        **法人連續性因子**（persist_rank, 6%）始終啟用，衡量近 10 日法人淨買超為正的天數比例。
        連續性高 = 法人持續布局（真波段），連續性低 = 一日行情（假主力）。
        同時用於調節隔日沖扣分 — 法人連續買超越強，隔沖扣分越輕。

        權重組合（由資料可用性決定，皆含連續性 6%）：
        - 8F: 外資16%+量比14%+法人14%+連續性6%+券資比10%+大戶12%+借券7%+分點11%+智慧分點10%
        - 7F: 外資18%+量比16%+法人16%+連續性6%+券資比11%+大戶13%+借券8%+分點12%
        - 3F: 外資37%+量比27%+法人27%+連續性9%

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

        consec_days_rank = df["consec_foreign_days"].rank(pct=True)
        bvr_rank = df["buy_vol_ratio"].rank(pct=True)
        total_rank = df["total_net"].rank(pct=True)

        # ── 法人金額加權因子（B1：衰減加權取代純天數）──────────────
        flow_df = compute_value_weighted_inst_flow(df_inst, stock_ids, window=10, decay=0.85)
        df = df.merge(flow_df, on="stock_id", how="left")
        df["inst_flow_weighted"] = df["inst_flow_weighted"].fillna(0.0)
        flow_rank = df["inst_flow_weighted"].rank(pct=True)
        # 混合排名：金額加權 60% + 天數 40%（大額持續 > 小額持續 > 大額一次）
        consec_rank = flow_rank * 0.60 + consec_days_rank * 0.40

        # ── 法人連續性因子（近 10 日正淨買超天數比例）──────────────
        persist_df = compute_institutional_persistence(df_inst, stock_ids, window=10)
        df = df.merge(persist_df, on="stock_id", how="left")
        df["inst_persistence"] = df["inst_persistence"].fillna(0.5)
        persist_rank = df["inst_persistence"].rank(pct=True)

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

        # ── 隔日沖扣分（broker_rank 修正）───────────────────────────
        # 使用 _load_broker_data() 已取得的 7 天資料（含 broker_name）偵測隔日沖
        # 若有 extended 資料則用範圍更廣的歷史做行為偵測
        _dt_broker_src = df_broker_ext if not df_broker_ext.empty else df_broker_raw
        if has_broker and not _dt_broker_src.empty and "broker_name" in _dt_broker_src.columns:
            broker_rank, self._daytrade_penalty_df = self._apply_daytrade_penalty(
                broker_rank,
                _dt_broker_src,
                stock_ids,
                df_price,
                persistence_scores=persist_df,
            )
        else:
            self._daytrade_penalty_df = pd.DataFrame(
                {"stock_id": stock_ids, "daytrade_penalty": 0.0, "daytrade_tags": ""}
            )

        # ── 加權組合（含法人連續性 persist_rank）──────────────────────
        if has_smart_broker and has_broker and has_sbl and has_margin and has_whale:
            # 8F+P：外資16%+量比14%+法人14%+連續性6%+券資比10%+大戶12%+借券7%+分點HHI11%+智慧分點10%
            df["chip_score"] = (
                consec_rank * 0.16
                + bvr_rank * 0.14
                + total_rank * 0.14
                + persist_rank * 0.06
                + smr_rank * 0.10
                + whale_rank * 0.12
                + sbl_rank * 0.07
                + broker_rank * 0.11
                + smart_broker_rank * 0.10
            )
            chip_tier = "8F"
        elif has_broker and has_sbl and has_margin and has_whale:
            # 7F+P：外資18%+量比16%+法人16%+連續性6%+券資比11%+大戶13%+借券8%+分點12%
            df["chip_score"] = (
                consec_rank * 0.18
                + bvr_rank * 0.16
                + total_rank * 0.16
                + persist_rank * 0.06
                + smr_rank * 0.11
                + whale_rank * 0.13
                + sbl_rank * 0.08
                + broker_rank * 0.12
            )
            chip_tier = "7F"
        elif has_broker and has_sbl and has_margin:
            # 6F+P：外資20%+量比18%+法人18%+連續性6%+券資比14%+借券12%+分點12%
            df["chip_score"] = (
                consec_rank * 0.20
                + bvr_rank * 0.18
                + total_rank * 0.18
                + persist_rank * 0.06
                + smr_rank * 0.14
                + sbl_rank * 0.12
                + broker_rank * 0.12
            )
            chip_tier = "6F"
        elif has_broker and has_sbl:
            # 5F+P：外資26%+量比20%+法人20%+連續性6%+借券14%+分點14%
            df["chip_score"] = (
                consec_rank * 0.26
                + bvr_rank * 0.20
                + total_rank * 0.20
                + persist_rank * 0.06
                + sbl_rank * 0.14
                + broker_rank * 0.14
            )
            chip_tier = "5F"
        elif has_broker:
            # 4F+P：外資30%+量比22%+法人22%+連續性6%+分點20%
            df["chip_score"] = (
                consec_rank * 0.30 + bvr_rank * 0.22 + total_rank * 0.22 + persist_rank * 0.06 + broker_rank * 0.20
            )
            chip_tier = "4F"
        elif has_sbl and has_margin and has_whale:
            # 6F+P：外資20%+量比18%+法人18%+連續性6%+券資比13%+大戶15%+借券10%
            df["chip_score"] = (
                consec_rank * 0.20
                + bvr_rank * 0.18
                + total_rank * 0.18
                + persist_rank * 0.06
                + smr_rank * 0.13
                + whale_rank * 0.15
                + sbl_rank * 0.10
            )
            chip_tier = "6F"
        elif has_sbl and has_margin:
            # 5F+P：外資23%+量比20%+法人20%+連續性6%+券資比16%+借券15%
            df["chip_score"] = (
                consec_rank * 0.23
                + bvr_rank * 0.20
                + total_rank * 0.20
                + persist_rank * 0.06
                + smr_rank * 0.16
                + sbl_rank * 0.15
            )
            chip_tier = "5F"
        elif has_sbl and has_whale:
            # 5F+P：外資26%+量比18%+法人18%+連續性6%+大戶22%+借券10%
            df["chip_score"] = (
                consec_rank * 0.26
                + bvr_rank * 0.18
                + total_rank * 0.18
                + persist_rank * 0.06
                + whale_rank * 0.22
                + sbl_rank * 0.10
            )
            chip_tier = "5F"
        elif has_sbl:
            # 4F+P：外資33%+量比23%+法人23%+連續性6%+借券15%
            df["chip_score"] = (
                consec_rank * 0.33 + bvr_rank * 0.23 + total_rank * 0.23 + persist_rank * 0.06 + sbl_rank * 0.15
            )
            chip_tier = "4F"
        elif has_margin and has_whale:
            # 5F+P：外資23%+量比20%+法人20%+連續性6%+券資比15%+大戶16%
            df["chip_score"] = (
                consec_rank * 0.23
                + bvr_rank * 0.20
                + total_rank * 0.20
                + persist_rank * 0.06
                + smr_rank * 0.15
                + whale_rank * 0.16
            )
            chip_tier = "5F"
        elif has_margin:
            # 4F+P：外資28%+量比23%+法人23%+連續性6%+券資比20%
            df["chip_score"] = (
                consec_rank * 0.28 + bvr_rank * 0.23 + total_rank * 0.23 + persist_rank * 0.06 + smr_rank * 0.20
            )
            chip_tier = "4F"
        elif has_whale:
            # 4F+P：外資33%+量比23%+法人23%+連續性6%+大戶15%
            df["chip_score"] = (
                consec_rank * 0.33 + bvr_rank * 0.23 + total_rank * 0.23 + persist_rank * 0.06 + whale_rank * 0.15
            )
            chip_tier = "4F"
        else:
            # 3F+P：外資37%+量比27%+法人27%+連續性9%
            df["chip_score"] = consec_rank * 0.37 + bvr_rank * 0.27 + total_rank * 0.27 + persist_rank * 0.09
            chip_tier = "3F"

        # ── 籌碼品質修正（法人斜率 ±3% + HHI 趨勢 ±3%）───────────
        slope_df = compute_inst_net_buy_slope(df_inst, stock_ids, window=10)
        hhi_trend_df = compute_hhi_trend(df_broker_raw, stock_ids) if not df_broker_raw.empty else None
        df["chip_score"] = self._apply_chip_quality_modifiers(
            df["chip_score"],
            stock_ids,
            slope_df=slope_df,
            hhi_trend_df=hhi_trend_df,
        )

        df["chip_tier"] = chip_tier
        return df[["stock_id", "chip_score", "chip_tier"]]

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

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """動能模式基本面：四階梯 + 營收加速度連續性加成。

        基礎分（80%）：四階梯短線爆發濾網
        Tier 1 (0.85)：MoM > 0 且 YoY > 0 且 YoY > yoy_3m_ago（月營收雙增 + YoY 近期創高）
        Tier 2 (0.72)：YoY > 0 且 YoY > yoy_3m_ago（YoY 正且加速）
        Tier 3 (0.55)：YoY > 0（YoY 正但未加速，或無加速度資料）
        Tier 4 (0.30)：YoY <= 0（YoY 衰退）
        無資料 fallback：0.50（中性）

        加速度連續性加成（20%，C2）：連續 N 月 YoY 加速 → 可持續成長 vs 一次性暴增
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

        base_df = pd.DataFrame(result_rows)

        # C2: 營收加速度連續性加成（20% 權重混合）
        accel_df = compute_revenue_acceleration_score(df_revenue, stock_ids, consecutive_threshold=3)
        if not accel_df.empty:
            base_df = base_df.merge(accel_df[["stock_id", "rev_accel_score"]], on="stock_id", how="left")
            base_df["rev_accel_score"] = base_df["rev_accel_score"].fillna(0.5)
            # 混合：基礎 Tier 80% + 加速度連續性 20%
            base_df["fundamental_score"] = base_df["fundamental_score"] * 0.80 + base_df["rev_accel_score"] * 0.20
            base_df = base_df.drop(columns=["rev_accel_score"])

        return base_df

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """動能模式風險過濾：ATR(14)/close > 80th percentile 剔除。"""
        return self._apply_atr_risk_filter(scored, df_price, percentile=80)
