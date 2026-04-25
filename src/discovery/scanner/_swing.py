"""SwingScanner — 中期波段掃描模式。

趨勢 + 基本面 + 法人布局，適合 1~3 個月波段操作。
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.data.database import get_session
from src.data.schema import (
    DailyPrice,
    HoldingDistribution,
    InstitutionalInvestor,
    MarginTrading,
)
from src.discovery.scanner._base import MarketScanner
from src.discovery.scanner._functions import (
    compute_broker_score,
    compute_hhi_trend,
    compute_inst_net_buy_slope,
    compute_institutional_persistence,
    compute_sbl_score,
    compute_smart_broker_score,
    compute_vcp_score,
    compute_whale_score,
)
from src.discovery.universe import UniverseConfig

logger = logging.getLogger(__name__)


class SwingScanner(MarketScanner):
    """中期波段掃描器（1~3 個月）。

    粗篩：趨勢（close > SMA60）+ 法人 20 日累積 + 量能
    細評：技術面 30% + 籌碼面 30% + 基本面 40%（依 Regime 動態調整）
    基本面：YoY 30% + MoM 20% + 加速度 20% + ROE QoQ 15% + 毛利率趨勢 15%；財報不足時降回純營收三因子
    籌碼面：最高 6F（投信 + 累積 + 大戶 + 分點 + 借券逆向 + 智慧分點），依資料可用性自適應降級
    風險過濾：年化波動率 > 85th percentile 剔除
    """

    mode_name = "swing"
    _COARSE_WEIGHTS: dict[str, float] = {"inst_rank": 0.40, "trend_rank": 0.30, "vol_rank": 0.30}

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lookback_days", 80)
        kwargs.setdefault("universe_config", UniverseConfig(volume_ratio_min=1.2, min_available_days=60))
        super().__init__(**kwargs)

    def _load_market_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """覆寫：swing 模式載入 2 個月營收資料（算加速度）。含 UniverseFilter Stage 0.5。

        項目 B：若 `self._shared` 已由 `run(shared=...)` 注入，則以 in-memory 過濾
        取代 DB 查詢，month=2 透過 `revenue_months` 參數顯式對齊原行為。
        """
        universe_ids = self._get_universe_ids()
        cutoff = date.today() - timedelta(days=self.lookback_days + 10)

        shared = getattr(self, "_shared", None)
        if shared is not None:
            return self._slice_shared_market_data(shared, universe_ids, cutoff, revenue_months=2)

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
        price_sorted = df_price[["stock_id", "date", "close"]].sort_values(["stock_id", "date"])
        sma60_rolling = price_sorted.groupby("stock_id")["close"].transform(
            lambda s: s.rolling(60, min_periods=60).mean()
        )
        sma60_latest = price_sorted.assign(_sma60=sma60_rolling).groupby("stock_id")["_sma60"].last().dropna()

        if not sma60_latest.empty:
            filtered["sma60"] = filtered["stock_id"].map(sma60_latest)
            filtered = filtered[filtered["sma60"].notna() & (filtered["close"] > filtered["sma60"])].copy()

        if filtered.empty:
            return pd.DataFrame()

        # 1) 法人 20 日累積買超排名
        filtered = self._compute_inst_net_buy(df_inst, filtered, days=20)

        # 2) 趨勢強度：close / SMA60 的比值
        if "sma60" in filtered.columns:
            filtered["trend_strength"] = filtered["close"] / filtered["sma60"]
            filtered["trend_rank"] = filtered["trend_strength"].rank(pct=True)
        else:
            filtered["trend_rank"] = 0.5

        # 3) 成交量排名
        filtered["vol_rank"] = filtered["volume"].rank(pct=True)

        # 動能欄位（用於 _rank_and_enrich 保留，不參與 _COARSE_WEIGHTS）
        filtered = self._compute_momentum_5d(df_price, filtered)

        return self._finalize_coarse(filtered)

    def _compute_technical_scores(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """波段模式技術面 5 因子：趨勢確認 + 均線排列 + 60日動能 + 量價齊揚 + ADX趨勢強度。"""
        from ta.trend import ADXIndicator as _ADXIndicator

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
            highs = stock_data["high"].values
            lows = stock_data["low"].values

            score = 0.0
            n_factors = 0

            # 1) SMA60 方向斜率（消除 binary cliff：close > SMA60 的 0/1 → 連續分數）
            #    衡量「SMA60 是否持續上揚」；slope = (SMA60_today - SMA60_20d_ago) / SMA60_20d_ago
            #    SMA60 為慢線，採 20 日比較窗口（5 日窗口數值極小且充滿雜訊）
            #    上揚 +1%/20日 → ~1.0，持平 → 0.5，下滑 -1%/20日 → ~0.0
            if len(closes) >= 80:
                sma60_today = closes[-60:].mean()
                sma60_20d_ago = closes[-80:-20].mean()
                if sma60_20d_ago > 0:
                    slope = (sma60_today - sma60_20d_ago) / sma60_20d_ago
                    score += max(0.0, min(1.0, 0.5 + slope * 50))
                else:
                    score += 0.5
                n_factors += 1

            # 2) SMA20/SMA60 乖離率（消除 binary cliff：SMA20 > SMA60 的 0/1 → 連續分數）
            #    spread = (SMA20 - SMA60) / SMA60：正值=多頭排列，越大越強
            #    乖離 +2.5% → 1.0，0% → 0.5，-2.5% → 0.0
            if len(closes) >= 60:
                sma20 = closes[-20:].mean()
                sma60 = closes[-60:].mean()
                if sma60 > 0:
                    spread = (sma20 - sma60) / sma60
                    score += max(0.0, min(1.0, 0.5 + spread * 20))
                else:
                    score += 0.5
                n_factors += 1

            # 3) 60 日動能
            if len(closes) >= 61:
                ret_60d = (closes[-1] - closes[-61]) / closes[-61]
                score += max(0.0, min(1.0, 0.5 + ret_60d * 2))
                n_factors += 1

            # 4) 量價結構：漲日均量 / 跌日均量（VPT 概念）
            #    健康多頭特徵：漲時放量、跌時縮量（洗盤）；比率越高越有利
            #    ratio=2 → ~1.0，ratio=1（均衡）→ ~0.5，ratio=0.5 → ~0.25
            #    「全漲日」罕見但給滿分，「全跌/平日」給 0 分
            if len(closes) >= 21 and len(volumes) >= 21:
                recent_closes = closes[-21:]
                recent_volumes = volumes[-21:]
                up_vols: list[float] = []
                down_vols: list[float] = []
                for i in range(1, 21):
                    if recent_closes[i] > recent_closes[i - 1]:
                        up_vols.append(float(recent_volumes[i]))
                    else:
                        down_vols.append(float(recent_volumes[i]))
                if up_vols and down_vols:
                    ratio = float(np.mean(up_vols)) / max(float(np.mean(down_vols)), 1.0)
                    # ratio=2 → 1.0，ratio=1 → 0.5，ratio→0 → 0.0
                    score += max(0.0, min(1.0, ratio / 2.0))
                elif up_vols:  # 近 20 日全為漲日
                    score += 1.0
                else:  # 近 20 日全為跌/平日
                    score += 0.0
                n_factors += 1

            # 5) ADX(14) + DMI 方向過濾
            #    ADX 無方向性，空頭強跌時 ADX 同樣飆高；必須搭配 +DI / -DI 確認多頭方向
            #    +DI > -DI（多頭趨勢）：score = (ADX - 15) / 30，ADX=15→0，ADX=45→1
            #    +DI ≤ -DI（空頭趨勢）：此因子給 0 分，避免把主跌段評為高分
            if len(closes) >= 28:
                window = min(len(closes), 60)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    adx_indicator = _ADXIndicator(
                        high=pd.Series(highs[-window:]),
                        low=pd.Series(lows[-window:]),
                        close=pd.Series(closes[-window:]),
                        n=14,
                    )
                    adx_s = adx_indicator.adx().dropna()
                    di_pos_s = adx_indicator.adx_pos().dropna()
                    di_neg_s = adx_indicator.adx_neg().dropna()
                if not adx_s.empty and not di_pos_s.empty and not di_neg_s.empty:
                    adx_val = float(adx_s.iloc[-1])
                    di_pos = float(di_pos_s.iloc[-1])
                    di_neg = float(di_neg_s.iloc[-1])
                    if di_pos > di_neg:  # 多頭趨勢才給分
                        score += min(1.0, max(0.0, (adx_val - 15) / 30))
                    # else: 空頭趨勢（+DI ≤ -DI），ADX 加成為 0
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
        """波段模式籌碼面：投信 + 累積 + 法人連續性 + 大戶 + 分點 + 借券（逆向）+ Smart Broker（有資料時）。

        **法人連續性因子**（persist_rank, 8%）始終啟用，衡量近 20 日法人淨買超為正的天數比例。
        波段模式用 20 日窗口（對應 1~3 個月操作週期），連續性高 = 法人穩定布局。
        同時用於調節隔日沖扣分 — 法人連續買超越強，隔沖扣分越輕。

        權重組合（由資料可用性決定，皆含連續性 8%）：
        - 6F+P: 投信18%+累積18%+連續性8%+大戶12%+分點12%+借券10%+智慧分點22%
        - 5F+P: 投信24%+累積24%+連續性8%+大戶16%+分點16%+借券12%
        - 2F+P: 投信42%+累積42%+連續性16%

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

        # ── 法人連續性因子（近 20 日正淨買超天數比例，波段模式用較長窗口）──
        persist_df = compute_institutional_persistence(df_inst, stock_ids, window=20)
        df = df.merge(persist_df, on="stock_id", how="left")
        df["inst_persistence"] = df["inst_persistence"].fillna(0.5)
        persist_rank = df["inst_persistence"].rank(pct=True)

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

        # ── 智慧分點因子（365 天歷史勝率 + 蓄積型分點）────────────────
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

        # ── 隔日沖扣分（broker_rank 修正，含法人連續性調節）──────────
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
        if has_smart_broker and has_broker and has_sbl and has_whale:
            # 6F+P：投信18%+累積18%+連續性8%+大戶12%+分點12%+借券10%+智慧分點22%
            df["chip_score"] = (
                trust_rank * 0.18
                + cum_rank * 0.18
                + persist_rank * 0.08
                + whale_rank * 0.12
                + broker_rank * 0.12
                + sbl_rank * 0.10
                + smart_broker_rank * 0.22
            )
            chip_tier = "6F"
        elif has_whale and has_broker and has_sbl:
            # 5F+P：投信24%+累積24%+連續性8%+大戶16%+分點16%+借券12%
            df["chip_score"] = (
                trust_rank * 0.24
                + cum_rank * 0.24
                + persist_rank * 0.08
                + whale_rank * 0.16
                + broker_rank * 0.16
                + sbl_rank * 0.12
            )
            chip_tier = "5F"
        elif has_whale and has_sbl:
            # 4F+P：投信29%+累積29%+連續性8%+大戶19%+借券15%
            df["chip_score"] = (
                trust_rank * 0.29 + cum_rank * 0.29 + persist_rank * 0.08 + whale_rank * 0.19 + sbl_rank * 0.15
            )
            chip_tier = "4F"
        elif has_broker and has_sbl:
            # 4F+P：投信29%+累積29%+連續性8%+分點19%+借券15%
            df["chip_score"] = (
                trust_rank * 0.29 + cum_rank * 0.29 + persist_rank * 0.08 + broker_rank * 0.19 + sbl_rank * 0.15
            )
            chip_tier = "4F"
        elif has_sbl:
            # 3F+P：投信36%+累積36%+連續性8%+借券20%
            df["chip_score"] = trust_rank * 0.36 + cum_rank * 0.36 + persist_rank * 0.08 + sbl_rank * 0.20
            chip_tier = "3F"
        elif has_whale and has_broker:
            # 4F+P：投信31%+累積31%+連續性8%+大戶15%+分點15%
            df["chip_score"] = (
                trust_rank * 0.31 + cum_rank * 0.31 + persist_rank * 0.08 + whale_rank * 0.15 + broker_rank * 0.15
            )
            chip_tier = "4F"
        elif has_whale:
            # 3F+P：投信36%+累積36%+連續性8%+大戶20%
            df["chip_score"] = trust_rank * 0.36 + cum_rank * 0.36 + persist_rank * 0.08 + whale_rank * 0.20
            chip_tier = "3F"
        elif has_broker:
            # 3F+P：投信36%+累積36%+連續性8%+分點20%
            df["chip_score"] = trust_rank * 0.36 + cum_rank * 0.36 + persist_rank * 0.08 + broker_rank * 0.20
            chip_tier = "3F"
        else:
            # 2F+P：投信42%+累積42%+連續性16%
            df["chip_score"] = trust_rank * 0.42 + cum_rank * 0.42 + persist_rank * 0.16
            chip_tier = "2F"

        # ── 籌碼品質修正（法人斜率 ±3% + HHI 趨勢 ±3%）───────────
        slope_df = compute_inst_net_buy_slope(df_inst, stock_ids, window=20)
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
        """波段模式基本面：YoY 30% + MoM 20% + 加速度 20% + ROE QoQ 15% + 毛利率趨勢 15%。

        ROE QoQ = roe[t] - roe[t-1]（季度環比，資本使用效率是否提升）。
        毛利率趨勢 = gross_margin[t] - gross_margin[t-2]（兩季對比，判斷是否正在轉好）。
        財報資料不足時自動降回純營收三因子（YoY 40% + MoM 30% + 加速度 30%）。
        """
        if df_revenue.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        rev = df_revenue[df_revenue["stock_id"].isin(stock_ids)].copy()
        if rev.empty:
            return pd.DataFrame({"stock_id": stock_ids, "fundamental_score": [0.5] * len(stock_ids)})

        # 營收加速度 = current_yoy - prev_yoy
        if "prev_yoy_growth" in rev.columns:
            rev["acceleration"] = rev["yoy_growth"].fillna(0) - rev["prev_yoy_growth"].fillna(0)

        # --- 財報因子 ---
        df_fin = self._load_financial_data(stock_ids, quarters=5)
        if df_fin.empty:
            # 降回純營收三因子
            yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
            mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)
            accel_rank = (
                rev["acceleration"].rank(pct=True) if "acceleration" in rev.columns else pd.Series(0.5, index=rev.index)
            )
            rev["fundamental_score"] = yoy_rank * 0.40 + mom_rank * 0.30 + accel_rank * 0.30
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
            return result

        grouped = df_fin.groupby("stock_id", sort=False)
        fin_rows = []
        for sid in stock_ids:
            row: dict = {"stock_id": sid, "roe_qoq": None, "gm_trend": None}
            if sid in grouped.groups:
                grp = grouped.get_group(sid).sort_values("date", ascending=False)
                # ROE QoQ = roe[t] - roe[t-1]（最新季 - 上季）
                if len(grp) >= 2 and pd.notna(grp.iloc[0]["roe"]) and pd.notna(grp.iloc[1]["roe"]):
                    row["roe_qoq"] = float(grp.iloc[0]["roe"]) - float(grp.iloc[1]["roe"])
                # 毛利率趨勢 = gross_margin[t] - gross_margin[t-2]（兩季對比）
                if len(grp) >= 3 and pd.notna(grp.iloc[0]["gross_margin"]) and pd.notna(grp.iloc[2]["gross_margin"]):
                    row["gm_trend"] = float(grp.iloc[0]["gross_margin"]) - float(grp.iloc[2]["gross_margin"])
            fin_rows.append(row)

        df_metrics = pd.DataFrame(fin_rows)
        has_any = df_metrics[["roe_qoq", "gm_trend"]].notna().any(axis=1).any()
        if not has_any:
            yoy_rank = rev["yoy_growth"].fillna(0).rank(pct=True)
            mom_rank = rev["mom_growth"].fillna(0).rank(pct=True)
            accel_rank = (
                rev["acceleration"].rank(pct=True) if "acceleration" in rev.columns else pd.Series(0.5, index=rev.index)
            )
            rev["fundamental_score"] = yoy_rank * 0.40 + mom_rank * 0.30 + accel_rank * 0.30
            result = pd.DataFrame({"stock_id": stock_ids})
            result = result.merge(rev[["stock_id", "fundamental_score"]], on="stock_id", how="left")
            return result

        roe_qoq_rank = df_metrics["roe_qoq"].rank(pct=True).fillna(0.5)
        gm_trend_rank = df_metrics["gm_trend"].rank(pct=True).fillna(0.5)

        # 合入營收指標（以 stock_id 對齊）
        merge_cols = ["stock_id", "yoy_growth", "mom_growth"]
        if "acceleration" in rev.columns:
            merge_cols.append("acceleration")
        df_metrics = df_metrics.merge(rev[merge_cols], on="stock_id", how="left")
        df_metrics["yoy_growth"] = pd.to_numeric(df_metrics["yoy_growth"], errors="coerce").fillna(0)
        df_metrics["mom_growth"] = pd.to_numeric(df_metrics["mom_growth"], errors="coerce").fillna(0)
        df_metrics["yoy_rank_val"] = df_metrics["yoy_growth"].rank(pct=True)
        df_metrics["mom_rank_val"] = df_metrics["mom_growth"].rank(pct=True)
        if "acceleration" in df_metrics.columns:
            df_metrics["accel_rank_val"] = df_metrics["acceleration"].rank(pct=True).fillna(0.5)
        else:
            df_metrics["accel_rank_val"] = 0.5

        # YoY 30% + MoM 20% + 加速度 20% + ROE QoQ 15% + 毛利率趨勢 15%
        df_metrics["fundamental_score"] = (
            df_metrics["yoy_rank_val"] * 0.30
            + df_metrics["mom_rank_val"] * 0.20
            + df_metrics["accel_rank_val"] * 0.20
            + roe_qoq_rank * 0.15
            + gm_trend_rank * 0.15
        )

        result = pd.DataFrame({"stock_id": stock_ids})
        result = result.merge(df_metrics[["stock_id", "fundamental_score"]], on="stock_id", how="left")
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

    def _compute_breakout_bonus(self, stock_ids: list[str], df_price: pd.DataFrame) -> pd.DataFrame:
        """技術形態突破加成（互斥取最高）：

        1. 季線突破（+4%）：前 5 日收盤 < SMA60，今日 close > SMA60，且成交量 > MA20量 × 1.5
        2. 箱型突破（+3%）：近 20 日高低差 < ATR20 × 3（窄幅整理），今日突破 20 日最高點
        """
        rows = []
        grouped = df_price.sort_values("date").groupby("stock_id", sort=False)

        for sid in stock_ids:
            bonus = 0.0
            if sid not in grouped.groups:
                rows.append({"stock_id": sid, "breakout_bonus": bonus})
                continue

            stock_data = grouped.get_group(sid)
            if len(stock_data) < 21:
                rows.append({"stock_id": sid, "breakout_bonus": bonus})
                continue

            closes = stock_data["close"].values
            highs = stock_data["high"].values
            lows = stock_data["low"].values
            volumes = stock_data["volume"].values.astype(float)

            # ── 季線突破 +4% ──────────────────────────────────────────
            if len(closes) >= 66:
                sma60 = closes[-60:].mean()
                prev5_below = all(closes[-(i + 2)] < sma60 for i in range(1, 6))
                vol_ma20 = volumes[-21:-1].mean()
                if prev5_below and closes[-1] > sma60 and volumes[-1] > vol_ma20 * 1.5:
                    bonus = max(bonus, 0.04)

            # ── 箱型突破 +3% ──────────────────────────────────────────
            if len(closes) >= 20:
                recent_high = highs[-21:-1].max()
                recent_low = lows[-21:-1].min()
                # ATR20（TR = max(H-L, |H-prevC|, |L-prevC|)）
                tr_list = []
                for i in range(-20, 0):
                    h = highs[i]
                    lo = lows[i]
                    pc = closes[i - 1]
                    tr_list.append(max(h - lo, abs(h - pc), abs(lo - pc)))
                atr20 = sum(tr_list) / len(tr_list)
                # 窄幅整理（range < ATR20×3）且今日突破前 20 日最高點
                if (recent_high - recent_low) < atr20 * 3 and closes[-1] > recent_high:
                    bonus = max(bonus, 0.03)

            rows.append({"stock_id": sid, "breakout_bonus": bonus})

        return pd.DataFrame(rows)

    def _post_score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """波段模式加成：VCP / 季線突破 / 箱型突破三種形態，互斥取最高加成。"""
        df_price = getattr(self, "_df_price_for_vcp", pd.DataFrame())
        if df_price.empty or candidates.empty:
            return candidates
        candidates = candidates.copy()

        # VCP 加成（+3%）
        vcp_df = compute_vcp_score(candidates["stock_id"].tolist(), df_price)
        candidates = candidates.merge(vcp_df, on="stock_id", how="left")
        candidates["vcp_bonus"] = candidates["vcp_bonus"].fillna(0.0)

        # 技術突破加成（+3% or +4%）
        breakout_df = self._compute_breakout_bonus(candidates["stock_id"].tolist(), df_price)
        candidates = candidates.merge(breakout_df, on="stock_id", how="left")
        candidates["breakout_bonus"] = candidates["breakout_bonus"].fillna(0.0)

        # 互斥取最高
        candidates["composite_score"] = candidates["composite_score"] + candidates[["vcp_bonus", "breakout_bonus"]].max(
            axis=1
        )
        return candidates
