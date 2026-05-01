"""MomentumScanner — 短線動能掃描模式。

突破 + 資金流 + 量能擴張，適合 1~10 天短線操作。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.discovery.scanner._base import MarketScanner
from src.discovery.scanner._functions import (
    compute_broker_score,
    compute_hhi_trend,
    compute_inst_net_buy_slope,
    compute_institutional_persistence,
    compute_quality_score,
    compute_revenue_acceleration_score,
    compute_sbl_score,
    compute_sub_factor_weight_adjustments,
    compute_value_weighted_inst_flow,
    exclude_zero_variance_factors,
)
from src.discovery.universe import UniverseConfig

logger = logging.getLogger(__name__)


def _get_chip_base_weights(
    has_broker: bool,
    has_sbl: bool,
    has_margin: bool,
) -> tuple[dict[str, float], str]:
    """根據資料可用性決定籌碼子因子權重組合。

    將 _compute_chip_scores() 的分支提取為純函數，
    回傳 (weight_dict, chip_tier)。weight_dict 的 key 對應 rank 變數名稱：
    consec / bvr / persist / smr / sbl / broker。

    v3 變更：移除 whale（消融影響低 ρ=0.972，週資料時效不匹配）
             移除 smart_broker（消融影響最低 ρ=0.985），15 tier → 7 tier。

    Returns:
        (weight_dict, chip_tier) — chip_tier 如 "6F"/"5F"/"4F"/"3F" 等
    """
    if has_broker and has_sbl and has_margin:
        return {
            "consec": 0.29,
            "bvr": 0.27,
            "persist": 0.06,
            "smr": 0.14,
            "sbl": 0.12,
            "broker": 0.12,
        }, "6F"
    elif has_broker and has_sbl:
        return {
            "consec": 0.36,
            "bvr": 0.30,
            "persist": 0.06,
            "sbl": 0.14,
            "broker": 0.14,
        }, "5F"
    elif has_broker:
        return {
            "consec": 0.41,
            "bvr": 0.33,
            "persist": 0.06,
            "broker": 0.20,
        }, "4F"
    elif has_sbl and has_margin:
        return {
            "consec": 0.33,
            "bvr": 0.30,
            "persist": 0.06,
            "smr": 0.16,
            "sbl": 0.15,
        }, "5F"
    elif has_sbl:
        return {
            "consec": 0.44,
            "bvr": 0.35,
            "persist": 0.06,
            "sbl": 0.15,
        }, "4F"
    elif has_margin:
        return {
            "consec": 0.39,
            "bvr": 0.35,
            "persist": 0.06,
            "smr": 0.20,
        }, "4F"
    else:
        return {
            "consec": 0.50,
            "bvr": 0.40,
            "persist": 0.10,
        }, "3F"


class MomentumScanner(MarketScanner):
    """短線動能掃描器（1~10 天）。

    粗篩：動能 + 流動性
    細評：技術面 + 籌碼面 + 消息面（三維度，Regime 動態權重）
    風險過濾：ATR ratio > 80th percentile 剔除
    盤整期（sideways）自動暫停掃描。
    """

    mode_name = "momentum"
    _auto_sync_broker = True  # Stage 2.5 自動補抓候選股分點資料
    _revenue_months = 4  # 載入 4 個月營收，啟用「本月 YoY - 3 個月前 YoY」加速度輕微加成
    _blocked_regimes = {"sideways"}  # 盤整期歷史勝率 15%，暫停掃描
    _COARSE_WEIGHTS: dict[str, float] = {"vol_rank": 0.30, "inst_rank": 0.40, "mom_rank": 0.30}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 80)  # F3 季線突破需 60 交易日（80 曆日）
        # 2026-04-30 歸因實驗：min_close 5→30、volume_ratio_min None→1.0
        # 移除 thin 池（avg_turnover<50M 或 close<NT$30）佔 26.6%；clean 子集 +3.1pp 勝率 / +1.4pp 報酬
        kwargs.setdefault(
            "universe_config",
            UniverseConfig(
                min_close=30.0,
                min_available_days=30,
                volume_ratio_min=1.0,
                trend_filter_mode="trend_or_breakout",
            ),
        )
        super().__init__(**kwargs)
        self._chip_sub_factor_ic: pd.DataFrame | None = None

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
        """動能模式技術面 3 因子（冗餘精簡版）。

        審計結論：ret5d×ret10d r=0.74、vol_ratio×vol_accel r=0.78，
        各 Cluster 內保留 1 個代表因子，消除共線性。

        3 因子 / 3 Cluster（各 1/3）：
        - Cluster A（報酬動能）：ret5d
        - Cluster B（量能擴張）：vol_ratio
        - Cluster C（突破強度）：high20_proximity
        """
        if not stock_ids:
            return pd.DataFrame(columns=["stock_id", "technical_score"])

        df = df_price.sort_values(["stock_id", "date"])
        g = df.groupby("stock_id", sort=False)

        # ── 批次取各期收盤價 ──────────────────────────────────────────
        latest_close = g["close"].last()
        close_1d_ago = g["close"].apply(lambda s: float(s.iloc[-2]) if len(s) >= 2 else np.nan)
        close_5d_ago = g["close"].apply(lambda s: float(s.iloc[-6]) if len(s) >= 6 else np.nan)
        close_20d_max = g["close"].apply(lambda s: float(s.iloc[-20:].max()) if len(s) >= 20 else np.nan)

        # ── 批次取量能序列 ────────────────────────────────────────────
        latest_vol = g["volume"].last().astype(float)
        vol_20d_mean = g["volume"].apply(lambda s: float(s.astype(float).iloc[-20:].mean()) if len(s) >= 20 else np.nan)

        # ── 限縮到候選股集合，計算原始因子值 ─────────────────────────
        idx = pd.Index(stock_ids)
        c0 = latest_close.reindex(idx)
        c5 = close_5d_ago.reindex(idx).replace(0, np.nan)
        c20m = close_20d_max.reindex(idx).replace(0, np.nan)

        ret_5d = (c0 - c5) / c5
        high20_proximity = c0 / c20m
        vol_20d = vol_20d_mean.reindex(idx).replace(0, np.nan)
        vol_ratio_raw = latest_vol.reindex(idx) / vol_20d

        # ── 橫截面百分位排名（Regime Adaptive）─────────────────────────
        r5 = ret_5d.rank(pct=True)
        rb = high20_proximity.rank(pct=True)
        rv = vol_ratio_raw.rank(pct=True)

        # ── 漲停板特殊處理（台股 10% 漲跌幅限制）────────────────────────
        c1d = close_1d_ago.reindex(idx).replace(0, np.nan)
        limit_up_mask = ((c0 - c1d) / c1d) >= 0.098
        rv = rv.where(~limit_up_mask, other=1.0)

        # NaN 以中性 0.5 填補
        scores = pd.concat([r5, rb, rv], axis=1)
        scores.columns = ["r5", "rb", "rv"]
        scores = scores.fillna(0.5)

        # 3 Cluster 各 1 因子，等權 1/3
        cluster_a = scores["r5"]  # 報酬動能
        cluster_b = scores["rv"]  # 量能擴張
        cluster_c = scores["rb"]  # 突破強度

        # 零方差 Cluster 自動排除
        _eps = 1e-9
        clusters = [("A", cluster_a), ("B", cluster_b), ("C", cluster_c)]
        active = [(name, s) for name, s in clusters if s.std() >= _eps]
        if active:
            tech_score = sum(s for _, s in active) / len(active)
        else:
            tech_score = pd.Series(0.5, index=scores.index)

        # 保存子因子 rank 供 IC 診斷使用（3 因子版）
        sub_df = scores.copy()
        sub_df["stock_id"] = idx.tolist()
        sub_df = sub_df.rename(
            columns={
                "r5": "tech_ret5d",
                "rb": "tech_high20_proximity",
                "rv": "tech_vol_ratio",
            }
        )
        self._sub_factor_ranks["technical"] = sub_df

        return pd.DataFrame({"stock_id": idx.tolist(), "technical_score": tech_score.to_numpy()})

    def _compute_chip_scores(
        self,
        stock_ids: list[str],
        df_inst: pd.DataFrame,
        df_price: pd.DataFrame | None = None,
        df_margin: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """動能模式籌碼面：外資連續買超 + 買超佔量比 + 法人連續性 + 券資比 + 大戶持股 + 借券 + 分點（有資料時）。

        v2 變更：移除三大法人合計（total_net）因子，與 bvr r=0.86 冗餘。

        **法人連續性因子**（persist_rank, 6~10%）始終啟用，衡量近 10 日法人淨買超為正的天數比例。
        連續性高 = 法人持續布局（真波段），連續性低 = 一日行情（假主力）。
        同時用於調節隔日沖扣分 — 法人連續買超越強，隔沖扣分越輕。

        回傳欄位：stock_id, chip_score, chip_tier（因子層級字串，如 "8F"/"3F"/"N/A"）
        """
        if df_inst.empty:
            return pd.DataFrame({"stock_id": stock_ids, "chip_score": [0.5] * len(stock_ids), "chip_tier": "N/A"})

        # P1b: 從 DB 載入過去 30 天的 chip 子因子 IC（僅在啟用 IC 調整時）
        if self.use_ic_adjustment and self._chip_sub_factor_ic is None:
            self._chip_sub_factor_ic = self._load_chip_sub_factor_ic()

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
        if has_broker and not df_broker_raw.empty and "broker_name" in df_broker_raw.columns:
            broker_rank, self._daytrade_penalty_df = self._apply_daytrade_penalty(
                broker_rank,
                df_broker_raw,
                stock_ids,
                df_price,
                persistence_scores=persist_df,
            )
        else:
            self._daytrade_penalty_df = pd.DataFrame(
                {"stock_id": stock_ids, "daytrade_penalty": 0.0, "daytrade_tags": ""}
            )

        # ── 加權組合（含法人連續性 persist_rank）──────────────────────
        weights, chip_tier = _get_chip_base_weights(
            has_broker=has_broker,
            has_sbl=has_sbl,
            has_margin=has_margin,
        )

        # P1b: 子因子 IC 動態權重調整（use_ic_adjustment=True 且有歷史 IC 資料時啟用）
        if self.use_ic_adjustment and self._chip_sub_factor_ic is not None:
            weights = compute_sub_factor_weight_adjustments(
                self._chip_sub_factor_ic,
                weights,
            )

        # 建立 rank 名稱 → Series 映射
        rank_map = {
            "consec": consec_rank,
            "bvr": bvr_rank,
            "persist": persist_rank,
        }
        if has_margin:
            rank_map["smr"] = smr_rank
        if has_sbl:
            rank_map["sbl"] = sbl_rank
        if has_broker:
            rank_map["broker"] = broker_rank

        # 零方差因子自動排除：rank 全部相同時無鑑別力，重新分配權重
        rank_map, weights = exclude_zero_variance_factors(rank_map, weights)

        df["chip_score"] = sum(rank_map[k] * w for k, w in weights.items()) if rank_map else 0.5

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

        # 保存子因子 rank 供 IC 診斷���用
        chip_sub = pd.DataFrame({"stock_id": df["stock_id"].tolist()})
        chip_sub["chip_consec"] = consec_rank.to_numpy()
        chip_sub["chip_bvr"] = bvr_rank.to_numpy()
        chip_sub["chip_persist"] = persist_rank.to_numpy()
        if has_margin:
            chip_sub["chip_smr"] = smr_rank.to_numpy()
        if has_sbl:
            chip_sub["chip_sbl"] = sbl_rank.to_numpy()
        if has_broker:
            chip_sub["chip_broker"] = broker_rank.to_numpy()
        self._sub_factor_ranks["chip"] = chip_sub

        return df[["stock_id", "chip_score", "chip_tier"]]

    _FUND_TIER_ANCHOR: dict[int, float] = {1: 0.85, 2: 0.72, 3: 0.55, 4: 0.30}
    _FUND_WEIGHT_TIER: float = 0.55
    _FUND_WEIGHT_INTRA: float = 0.15
    _FUND_WEIGHT_ACCEL: float = 0.20
    _FUND_WEIGHT_QUALITY: float = 0.10
    _FUND_DYNAMIC_THRESHOLD_MIN_SAMPLES: int = 10

    def _compute_fundamental_scores(self, stock_ids: list[str], df_revenue: pd.DataFrame) -> pd.DataFrame:
        """動能模式基本面：Tier Anchor + 同 Tier 內 Rank + 加速度 Boost + 品質 Boost（v5）。

        四段式公式（Phase F — 新增 quality 維度，含 FinancialStatement 毛利率/FCF）：
            final = 0.55 × tier_anchor + 0.15 × intra_rank + 0.20 × accel_boost + 0.10 × quality

        Tier 判定（動態 percentile 門檻 + YoY > 0 絕對下限）：
            Tier 1 (anchor=0.85): YoY > p80 且 YoY > 0 且 加速度 > 0 且 MoM > 0
            Tier 2 (anchor=0.72): YoY > p60 且 YoY > 0 且 加速度 > 0
            Tier 3 (anchor=0.55): YoY > 0
            Tier 4 (anchor=0.30): YoY ≤ 0
            Fallback (0.50):      無資料 / 候選股不足以計算 percentile

        當有效樣本 < 10 時退回 2574682 的絕對條件版本（避免單股測試誤判）。

        intra_rank 為同 tier 內 yoy_growth 的 percentile rank（0~1），不跨 tier。
        accel_boost 來自 compute_revenue_acceleration_score（4 個月連續性加速）。
        quality 來自 compute_quality_score（毛利率 YoY 改善 0.6 + FCF 正值 0.4），
        補足單引擎盲區 — 抓「營收成長但毛利萎縮」vs「雙引擎驅動」的區分。
        """
        default = pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})
        if df_revenue.empty or len(stock_ids) == 0:
            return default

        rev_grouped = df_revenue.groupby("stock_id", sort=False)
        raw_rows = []
        for sid in stock_ids:
            if sid not in rev_grouped.groups:
                raw_rows.append({"stock_id": sid, "yoy": np.nan, "mom": np.nan, "yoy_3m": np.nan})
                continue

            grp = rev_grouped.get_group(sid)
            if "date" in grp.columns:
                grp = grp.sort_values("date", ascending=False)

            yoy_raw = grp.iloc[0].get("yoy_growth", None)
            mom_raw = grp.iloc[0].get("mom_growth", None)
            yoy3_raw = grp.iloc[0].get("yoy_3m_ago", None)
            raw_rows.append(
                {
                    "stock_id": sid,
                    "yoy": float(yoy_raw) if yoy_raw is not None and not pd.isna(yoy_raw) else np.nan,
                    "mom": float(mom_raw) if mom_raw is not None and not pd.isna(mom_raw) else np.nan,
                    "yoy_3m": float(yoy3_raw) if yoy3_raw is not None and not pd.isna(yoy3_raw) else np.nan,
                }
            )

        df = pd.DataFrame(raw_rows)
        valid_yoy = df["yoy"].dropna()

        use_dynamic = len(valid_yoy) >= self._FUND_DYNAMIC_THRESHOLD_MIN_SAMPLES
        if use_dynamic:
            yoy_p80 = float(valid_yoy.quantile(0.80))
            yoy_p60 = float(valid_yoy.quantile(0.60))
        else:
            yoy_p80 = yoy_p60 = np.nan

        def _assign_tier(row: pd.Series) -> int:
            yoy = row["yoy"]
            if pd.isna(yoy):
                return 0  # fallback
            has_mom_pos = not pd.isna(row["mom"]) and row["mom"] > 0
            has_accel = not pd.isna(row["yoy_3m"]) and yoy > row["yoy_3m"]

            if use_dynamic:
                if yoy > yoy_p80 and yoy > 0 and has_accel and has_mom_pos:
                    return 1
                if yoy > yoy_p60 and yoy > 0 and has_accel:
                    return 2
                if yoy > 0:
                    return 3
                return 4
            # 小樣本：退回絕對條件
            if yoy > 0 and has_mom_pos and has_accel:
                return 1
            if yoy > 0 and has_accel:
                return 2
            if yoy > 0:
                return 3
            return 4

        df["tier"] = df.apply(_assign_tier, axis=1).astype(int)
        df["tier_anchor"] = df["tier"].map(self._FUND_TIER_ANCHOR).fillna(0.5)

        # 同 tier 內 percentile rank（yoy_growth），≤1 支則中性 0.5
        df["intra_rank"] = 0.5
        for t in (1, 2, 3, 4):
            mask = df["tier"] == t
            if mask.sum() >= 2:
                df.loc[mask, "intra_rank"] = df.loc[mask, "yoy"].rank(pct=True)

        accel_df = compute_revenue_acceleration_score(df_revenue, stock_ids, consecutive_threshold=3)
        if not accel_df.empty:
            df = df.merge(accel_df[["stock_id", "rev_accel_score"]], on="stock_id", how="left")
            df["rev_accel_score"] = df["rev_accel_score"].fillna(0.5)
        else:
            df["rev_accel_score"] = 0.5

        # Phase F — 品質維度（毛利率 YoY + FCF 正值）
        df_fin = self._load_financial_data(stock_ids, quarters=5)
        quality_df = compute_quality_score(df_fin, stock_ids)
        df = df.merge(quality_df, on="stock_id", how="left")
        df["quality_score"] = df["quality_score"].fillna(0.5)

        df["fundamental_score"] = (
            self._FUND_WEIGHT_TIER * df["tier_anchor"]
            + self._FUND_WEIGHT_INTRA * df["intra_rank"]
            + self._FUND_WEIGHT_ACCEL * df["rev_accel_score"]
            + self._FUND_WEIGHT_QUALITY * df["quality_score"]
        )
        # fallback (tier=0)：無營收資料，維持中性 0.5
        df.loc[df["tier"] == 0, "fundamental_score"] = 0.5

        return df[["stock_id", "fundamental_score"]]

    def _load_chip_sub_factor_ic(self) -> pd.DataFrame | None:
        """從 DB 載入歷史推薦紀錄，計算 chip 子因子 IC。

        使用 DiscoveryRecord（scan_date/stock_id/close）+ 同期 DailyPrice
        重新計算 chip_score 維度的 IC，然後按子因子已知的貢獻比例映射。
        若歷史不足 20 筆，回傳 None（graceful degradation → 使用硬編碼權重）。
        """
        try:
            from src.data.schema import DailyPrice, DiscoveryRecord

            cutoff = date.today() - timedelta(days=35)
            with get_session() as session:
                stmt = select(
                    DiscoveryRecord.scan_date,
                    DiscoveryRecord.stock_id,
                    DiscoveryRecord.close,
                    DiscoveryRecord.chip_score,
                    DiscoveryRecord.chip_tier,
                ).where(
                    DiscoveryRecord.mode == self.mode_name,
                    DiscoveryRecord.scan_date >= cutoff,
                    DiscoveryRecord.chip_score.isnot(None),
                )
                rows = session.execute(stmt).all()
                if len(rows) < 20:
                    logger.info("P1b: chip IC 歷史推薦不足 20 筆（%d），跳過子因子 IC 調整", len(rows))
                    return None

                df_records = pd.DataFrame(
                    rows,
                    columns=["scan_date", "stock_id", "close", "chip_score", "chip_tier"],
                )

                stock_ids = df_records["stock_id"].unique().tolist()
                price_stmt = select(DailyPrice.stock_id, DailyPrice.date, DailyPrice.close).where(
                    DailyPrice.stock_id.in_(stock_ids),
                    DailyPrice.date >= cutoff,
                )
                price_rows = session.execute(price_stmt).all()
                if not price_rows:
                    return None
                df_prices = pd.DataFrame(price_rows, columns=["stock_id", "date", "close"])

            # 計算 chip_score 整體 IC
            from src.discovery.scanner._functions import compute_factor_ic

            ic_input = df_records[["scan_date", "stock_id", "close"]].copy()
            ic_input["chip_score"] = df_records["chip_score"]
            ic_df = compute_factor_ic(ic_input, df_prices, holding_days=5, lookback_days=30)
            if ic_df.empty:
                return None

            chip_row = ic_df[ic_df["factor"] == "chip_score"]
            if chip_row.empty:
                return None

            chip_direction = chip_row.iloc[0]["direction"]
            chip_count = int(chip_row.iloc[0]["evaluable_count"])

            if chip_count < 20:
                return None

            # 整體 chip_score 有效 → 維持原權重；弱/反向 → 標記所有子因子
            # 使用整體方向作為各子因子的代理方向
            chip_ic = float(chip_row.iloc[0]["ic"])
            logger.info("P1b: chip_score IC=%.4f (%s, n=%d)", chip_ic, chip_direction, chip_count)

            if chip_direction == "effective":
                # chip 整體有效 → 不需調整子因子
                return None

            # chip 整體弱/反向 → 標記��有子因子為同一方向，讓權重歸一化自動調整
            all_keys = ["consec", "bvr", "persist", "smr", "whale", "sbl", "broker", "smart_broker"]
            results = []
            for key in all_keys:
                results.append(
                    {
                        "factor": key,
                        "ic": chip_ic,
                        "evaluable_count": chip_count,
                        "direction": chip_direction,
                    }
                )
            return pd.DataFrame(results)

        except Exception:
            logger.debug("P1b: chip 子因子 IC 載入失敗，使用硬編碼權重", exc_info=True)
            return None

    def _apply_risk_filter(self, scored: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        """動能模式風險過濾：ATR(14)/close > 80th percentile 剔除。"""
        return self._apply_atr_risk_filter(scored, df_price, percentile=80)
