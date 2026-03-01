"""資料抓取模組 — 透過 FinMind API 取得台股歷史資料。

支援的資料集：
- TaiwanStockPrice:                            日K線
- TaiwanStockInstitutionalInvestorsBuySell:    三大法人買賣超
- TaiwanStockMarginPurchaseShortSale:          融資融券
- TaiwanStockMonthRevenue:                     月營收
- TaiwanStockDividend:                         股利
- TaiwanStockTotalReturnIndex:                 加權指數
- TaiwanStockFinancialStatements:              綜合損益表
- TaiwanStockBalanceSheet:                     資產負債表
- TaiwanStockCashFlowsStatement:               現金流量表
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd
import requests

from src.config import settings

logger = logging.getLogger(__name__)


class DataFetcher(ABC):
    """資料抓取抽象基類 — 日後可擴充 Fugle、TWSE 等來源。"""

    @abstractmethod
    def fetch_daily_price(self, stock_id: str, start: str, end: str) -> pd.DataFrame: ...

    @abstractmethod
    def fetch_institutional(self, stock_id: str, start: str, end: str) -> pd.DataFrame: ...

    @abstractmethod
    def fetch_margin_trading(self, stock_id: str, start: str, end: str) -> pd.DataFrame: ...

    @abstractmethod
    def fetch_monthly_revenue(self, stock_id: str, start: str, end: str) -> pd.DataFrame: ...

    @abstractmethod
    def fetch_dividend(self, stock_id: str, start: str, end: str) -> pd.DataFrame: ...


class FinMindFetcher(DataFetcher):
    """FinMind API 資料抓取實作。

    API 文件：https://finmindtrade.com/analysis/#/data/api
    """

    def __init__(self, api_token: str | None = None) -> None:
        self.api_url = settings.finmind.api_url
        self.api_token = api_token or settings.finmind.api_token
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    #  內部共用方法
    # ------------------------------------------------------------------ #

    def _request(self, dataset: str, stock_id: str, start: str, end: str) -> pd.DataFrame:
        """統一 API 請求邏輯，含錯誤處理與速率控制。"""
        params = {
            "dataset": dataset,
            "data_id": stock_id,
            "start_date": start,
            "end_date": end,
        }
        if self.api_token:
            params["token"] = self.api_token

        logger.info("抓取 %s | %s | %s ~ %s", dataset, stock_id, start, end)

        resp = self._session.get(self.api_url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("msg") != "success":
            raise RuntimeError(f"FinMind API 錯誤: {payload.get('msg')} (status={payload.get('status')})")

        df = pd.DataFrame(payload.get("data", []))
        if df.empty:
            logger.warning("無資料: %s %s %s~%s", dataset, stock_id, start, end)

        # 免費版速率限制：避免過於頻繁
        time.sleep(0.5)
        return df

    def _request_by_date(self, dataset: str, start: str, end: str) -> pd.DataFrame:
        """按日期查全市場（不指定 data_id），回傳所有股票的資料。

        注意：此功能需要 FinMind 付費帳號。免費帳號會回傳 400 錯誤，
        此時回傳空 DataFrame，由上層改用逐股抓取的備案策略。
        """
        params = {
            "dataset": dataset,
            "start_date": start,
            "end_date": end,
        }
        if self.api_token:
            params["token"] = self.api_token

        logger.info("抓取全市場 %s | %s ~ %s", dataset, start, end)

        resp = self._session.get(self.api_url, params=params, timeout=60)

        if resp.status_code == 400:
            payload = resp.json()
            logger.warning("全市場批次查詢不可用（需付費帳號）: %s", payload.get("msg", ""))
            return pd.DataFrame()

        resp.raise_for_status()
        payload = resp.json()

        if payload.get("msg") != "success":
            raise RuntimeError(f"FinMind API 錯誤: {payload.get('msg')} (status={payload.get('status')})")

        df = pd.DataFrame(payload.get("data", []))
        if df.empty:
            logger.warning("無資料: 全市場 %s %s~%s", dataset, start, end)

        time.sleep(1)
        return df

    # ------------------------------------------------------------------ #
    #  公開介面
    # ------------------------------------------------------------------ #

    def fetch_daily_price(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取日K線資料。

        回傳欄位: date, stock_id, open, high, low, close,
                  Trading_Volume(成交股數), Trading_money(成交金額), spread
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request("TaiwanStockPrice", stock_id, start, end)
        if df.empty:
            return df

        df = df.rename(
            columns={
                "Trading_Volume": "volume",
                "Trading_money": "turnover",
                "max": "high",
                "min": "low",
            }
        )
        # 只保留需要的欄位
        keep = ["date", "stock_id", "open", "high", "low", "close", "volume", "turnover", "spread"]
        df = df[[c for c in keep if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def fetch_institutional(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取三大法人買賣超資料。

        回傳欄位: date, stock_id, name, buy, sell, net
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start, end)
        if df.empty:
            return df

        df = df.rename(columns={"name": "name", "buy": "buy", "sell": "sell"})
        # 計算淨買賣超
        if "net" not in df.columns:
            df["net"] = df["buy"] - df["sell"]

        keep = ["date", "stock_id", "name", "buy", "sell", "net"]
        df = df[[c for c in keep if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def fetch_margin_trading(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取融資融券資料。

        回傳欄位: date, stock_id, margin_buy, margin_sell, margin_balance,
                  short_sell, short_buy, short_balance
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request("TaiwanStockMarginPurchaseShortSale", stock_id, start, end)
        if df.empty:
            return df

        rename_map = {
            "MarginPurchaseBuy": "margin_buy",
            "MarginPurchaseSell": "margin_sell",
            "MarginPurchaseTodayBalance": "margin_balance",
            "ShortSaleSell": "short_sell",
            "ShortSaleBuy": "short_buy",
            "ShortSaleTodayBalance": "short_balance",
        }
        df = df.rename(columns=rename_map)

        keep = [
            "date",
            "stock_id",
            "margin_buy",
            "margin_sell",
            "margin_balance",
            "short_sell",
            "short_buy",
            "short_balance",
        ]
        df = df[[c for c in keep if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def fetch_monthly_revenue(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取月營收資料。

        回傳欄位: date, stock_id, revenue, revenue_month, revenue_year,
                  mom_growth, yoy_growth
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request("TaiwanStockMonthRevenue", stock_id, start, end)
        if df.empty:
            return df

        keep = ["date", "stock_id", "revenue", "revenue_month", "revenue_year"]
        df = df[[c for c in keep if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # 計算月增率 (MoM) 和年增率 (YoY)
        df = df.sort_values("date").reset_index(drop=True)
        df["mom_growth"] = df["revenue"].pct_change() * 100

        # 年增率：與 12 個月前比較
        if len(df) > 12:
            df["yoy_growth"] = (df["revenue"] / df["revenue"].shift(12) - 1) * 100
        else:
            df["yoy_growth"] = None

        return df

    def fetch_per_pbr(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取本益比/股價淨值比/殖利率（FinMind TaiwanStockPER）。

        回傳欄位: date, stock_id, pe_ratio, pb_ratio, dividend_yield
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request("TaiwanStockPER", stock_id, start, end)
        if df.empty:
            return df

        rename_map = {
            "PER": "pe_ratio",
            "PBR": "pb_ratio",
            "DividendYield": "dividend_yield",
        }
        df = df.rename(columns=rename_map)

        keep = ["date", "stock_id", "pe_ratio", "pb_ratio", "dividend_yield"]
        df = df[[c for c in keep if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def fetch_dividend(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取股利資料。

        回傳欄位: date, stock_id, year, cash_dividend, stock_dividend,
                  cash_payment_date, announcement_date
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request("TaiwanStockDividend", stock_id, start, end)
        if df.empty:
            return df

        rename_map = {
            "CashEarningsDistribution": "cash_dividend",
            "StockEarningsDistribution": "stock_dividend",
            "CashDividendPaymentDate": "cash_payment_date",
            "AnnouncementDate": "announcement_date",
        }
        df = df.rename(columns=rename_map)

        keep = [
            "date",
            "stock_id",
            "year",
            "cash_dividend",
            "stock_dividend",
            "cash_payment_date",
            "announcement_date",
        ]
        df = df[[c for c in keep if c in df.columns]]

        df["date"] = pd.to_datetime(df["date"]).dt.date
        for col in ("cash_payment_date", "announcement_date"):
            if col in df.columns:
                # 先轉換為 datetime，然後轉換為 date，最後將 NaT 替換為 None
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
                # 將 NaT（pandas datetime NaT）替換為 None
                df.loc[pd.isna(df[col]), col] = None
        df["cash_dividend"] = df.get("cash_dividend", pd.Series(dtype=float)).fillna(0.0)
        df["stock_dividend"] = df.get("stock_dividend", pd.Series(dtype=float)).fillna(0.0)
        # 確保 year 欄位為字符串，並移除 NaN/None 值
        if "year" in df.columns:
            df["year"] = df["year"].astype(str)
            # 將 'nan' 字符串或空字符串替換為 None（保持 nullable=False 的資料品質）
            df.loc[df["year"].isin(["nan", ""]), "year"] = None
        return df

    def fetch_all_daily_price(self, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取全市場日K線（不指定 stock_id，按日期查詢）。

        回傳欄位同 fetch_daily_price: date, stock_id, open, high, low, close, volume, turnover, spread
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request_by_date("TaiwanStockPrice", start, end)
        if df.empty:
            return df

        df = df.rename(
            columns={
                "Trading_Volume": "volume",
                "Trading_money": "turnover",
                "max": "high",
                "min": "low",
            }
        )
        keep = ["date", "stock_id", "open", "high", "low", "close", "volume", "turnover", "spread"]
        df = df[[c for c in keep if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def fetch_all_institutional(self, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取全市場三大法人買賣超（不指定 stock_id，按日期查詢）。

        回傳欄位同 fetch_institutional: date, stock_id, name, buy, sell, net
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request_by_date("TaiwanStockInstitutionalInvestorsBuySell", start, end)
        if df.empty:
            return df

        df = df.rename(columns={"name": "name", "buy": "buy", "sell": "sell"})
        if "net" not in df.columns:
            df["net"] = df["buy"] - df["sell"]

        keep = ["date", "stock_id", "name", "buy", "sell", "net"]
        df = df[[c for c in keep if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def fetch_stock_info(self) -> pd.DataFrame:
        """抓取全市場股票基本資料（產業分類）。

        使用 TaiwanStockInfo dataset，不需 stock_id/日期，一次回傳全市場。

        回傳欄位: stock_id, stock_name, industry_category, listing_type
        """
        params = {"dataset": "TaiwanStockInfo"}
        if self.api_token:
            params["token"] = self.api_token

        logger.info("抓取 TaiwanStockInfo（全市場股票基本資料）")

        resp = self._session.get(self.api_url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("msg") != "success":
            raise RuntimeError(f"FinMind API 錯誤: {payload.get('msg')} (status={payload.get('status')})")

        df = pd.DataFrame(payload.get("data", []))
        if df.empty:
            logger.warning("無資料: TaiwanStockInfo")
            return df

        time.sleep(0.5)

        rename_map = {
            "stock_id": "stock_id",
            "stock_name": "stock_name",
            "industry_category": "industry_category",
            "type": "listing_type",
        }
        df = df.rename(columns=rename_map)

        keep = ["stock_id", "stock_name", "industry_category", "listing_type"]
        df = df[[c for c in keep if c in df.columns]]

        return df

    def fetch_taiex_index(self, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取加權指數日資料（用於 benchmark）。

        使用 TAIEX 作為 stock_id，存入 DailyPrice 表複用基礎設施。
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request("TaiwanStockTotalReturnIndex", "TAIEX", start, end)
        if df.empty:
            return df

        # 將指數值作為 close，其他欄位補齊
        if "price" in df.columns:
            df = df.rename(columns={"price": "close"})

        df["stock_id"] = "TAIEX"
        for col in ("open", "high", "low"):
            if col not in df.columns:
                df[col] = df["close"]
        for col in ("volume", "turnover"):
            if col not in df.columns:
                df[col] = 0
        if "spread" not in df.columns:
            df["spread"] = 0.0

        keep = ["date", "stock_id", "open", "high", "low", "close", "volume", "turnover", "spread"]
        df = df[[c for c in keep if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    # ------------------------------------------------------------------ #
    #  財報三表（EAV → pivot 寬表）
    # ------------------------------------------------------------------ #

    # FinMind 財報 EAV 中需要的 type 值對應
    _INCOME_TYPES = {
        "Revenue": "revenue",
        "GrossProfit": "gross_profit",
        "OperatingIncome": "operating_income",
        "IncomeAfterTaxes": "net_income",
        "EPS": "eps",
    }

    _BALANCE_TYPES = {
        "TotalAssets": "total_assets",
        "TotalLiabilities": "total_liabilities",
        "Equity": "equity",
    }

    _CASHFLOW_TYPES = {
        "CashFlowsFromOperatingActivities": "operating_cf",
        "CashProvidedByInvestingActivities": "investing_cf",
        "CashFlowsFromFinancingActivities": "financing_cf",
    }

    def _pivot_eav(self, df: pd.DataFrame, type_map: dict[str, str]) -> pd.DataFrame:
        """將 FinMind EAV 格式 pivot 為寬表。

        Args:
            df: FinMind API 回傳的 EAV DataFrame（date, stock_id, type, value）
            type_map: {FinMind type 名稱: 目標欄位名}

        Returns:
            pivot 後的 DataFrame，index 為 (date, stock_id)
        """
        if df.empty:
            return pd.DataFrame()

        # 只保留需要的 type
        df = df[df["type"].isin(type_map.keys())].copy()
        if df.empty:
            return pd.DataFrame()

        # 重新命名 type 值
        df["type"] = df["type"].map(type_map)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # pivot：每個 date+stock_id 一列，type 值變成欄位
        pivoted = df.pivot_table(index=["date", "stock_id"], columns="type", values="value", aggfunc="first")
        pivoted = pivoted.reset_index()
        pivoted.columns.name = None
        pivoted["date"] = pd.to_datetime(pivoted["date"]).dt.date

        return pivoted

    def fetch_financial_statements(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取綜合損益表（Revenue, GrossProfit, OperatingIncome, IncomeAfterTaxes, EPS）。

        回傳欄位: date, stock_id, revenue, gross_profit, operating_income, net_income, eps
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request("TaiwanStockFinancialStatements", stock_id, start, end)
        return self._pivot_eav(df, self._INCOME_TYPES)

    def fetch_balance_sheet(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取資產負債表（TotalAssets, TotalLiabilities, Equity）。

        回傳欄位: date, stock_id, total_assets, total_liabilities, equity
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request("TaiwanStockBalanceSheet", stock_id, start, end)
        return self._pivot_eav(df, self._BALANCE_TYPES)

    def fetch_cash_flows(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        """抓取現金流量表（Operating, Investing, Financing）。

        回傳欄位: date, stock_id, operating_cf, investing_cf, financing_cf
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request("TaiwanStockCashFlowsStatement", stock_id, start, end)
        return self._pivot_eav(df, self._CASHFLOW_TYPES)

    def fetch_financial_summary(self, stock_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        """合併三表並計算衍生比率，回傳完整財報寬表。

        回傳欄位: date, stock_id, year, quarter,
                  revenue ~ eps, total_assets ~ equity, operating_cf ~ free_cf,
                  gross_margin ~ debt_ratio
        """
        if end is None:
            end = date.today().isoformat()

        df_income = self.fetch_financial_statements(stock_id, start, end)
        df_balance = self.fetch_balance_sheet(stock_id, start, end)
        df_cashflow = self.fetch_cash_flows(stock_id, start, end)

        # 合併三表
        if df_income.empty:
            return pd.DataFrame()

        merged = df_income.copy()
        if not df_balance.empty:
            merged = merged.merge(df_balance, on=["date", "stock_id"], how="left")
        if not df_cashflow.empty:
            merged = merged.merge(df_cashflow, on=["date", "stock_id"], how="left")

        # 推算 year / quarter
        date_series = pd.to_datetime(merged["date"])
        merged["year"] = date_series.dt.year
        merged["quarter"] = date_series.dt.quarter
        merged["date"] = date_series.dt.date

        # 計算衍生比率
        merged = compute_financial_ratios(merged)

        return merged


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """計算財報衍生比率（純函數，可獨立測試）。

    輸入 DataFrame 需包含：revenue, gross_profit, operating_income, net_income,
    total_assets, total_liabilities, equity, operating_cf, investing_cf。

    新增欄位：free_cf, gross_margin, operating_margin, net_margin, roe, roa, debt_ratio
    """
    df = df.copy()

    # 自由現金流
    if "operating_cf" in df.columns and "investing_cf" in df.columns:
        df["free_cf"] = df["operating_cf"].fillna(0) + df["investing_cf"].fillna(0)

    # 毛利率 (%)
    if "gross_profit" in df.columns and "revenue" in df.columns:
        df["gross_margin"] = df.apply(
            lambda r: r["gross_profit"] / r["revenue"] * 100 if r.get("revenue") else None, axis=1
        )

    # 營益率 (%)
    if "operating_income" in df.columns and "revenue" in df.columns:
        df["operating_margin"] = df.apply(
            lambda r: r["operating_income"] / r["revenue"] * 100 if r.get("revenue") else None, axis=1
        )

    # 淨利率 (%)
    if "net_income" in df.columns and "revenue" in df.columns:
        df["net_margin"] = df.apply(
            lambda r: r["net_income"] / r["revenue"] * 100 if r.get("revenue") else None, axis=1
        )

    # ROE (%)
    if "net_income" in df.columns and "equity" in df.columns:
        df["roe"] = df.apply(lambda r: r["net_income"] / r["equity"] * 100 if r.get("equity") else None, axis=1)

    # ROA (%)
    if "net_income" in df.columns and "total_assets" in df.columns:
        df["roa"] = df.apply(
            lambda r: r["net_income"] / r["total_assets"] * 100 if r.get("total_assets") else None, axis=1
        )

    # 負債比 (%)
    if "total_liabilities" in df.columns and "total_assets" in df.columns:
        df["debt_ratio"] = df.apply(
            lambda r: r["total_liabilities"] / r["total_assets"] * 100 if r.get("total_assets") else None, axis=1
        )

    return df
