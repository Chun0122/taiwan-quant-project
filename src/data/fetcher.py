"""資料抓取模組 — 透過 FinMind API 取得台股歷史資料。

支援的資料集：
- TaiwanStockPrice:                            日K線
- TaiwanStockInstitutionalInvestorsBuySell:    三大法人買賣超
- TaiwanStockMarginPurchaseShortSale:          融資融券
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
    def fetch_daily_price(
        self, stock_id: str, start: str, end: str
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def fetch_institutional(
        self, stock_id: str, start: str, end: str
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def fetch_margin_trading(
        self, stock_id: str, start: str, end: str
    ) -> pd.DataFrame:
        ...


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
            raise RuntimeError(
                f"FinMind API 錯誤: {payload.get('msg')} (status={payload.get('status')})"
            )

        df = pd.DataFrame(payload.get("data", []))
        if df.empty:
            logger.warning("無資料: %s %s %s~%s", dataset, stock_id, start, end)

        # 免費版速率限制：避免過於頻繁
        time.sleep(0.5)
        return df

    # ------------------------------------------------------------------ #
    #  公開介面
    # ------------------------------------------------------------------ #

    def fetch_daily_price(
        self, stock_id: str, start: str, end: str | None = None
    ) -> pd.DataFrame:
        """抓取日K線資料。

        回傳欄位: date, stock_id, open, high, low, close,
                  Trading_Volume(成交股數), Trading_money(成交金額), spread
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request("TaiwanStockPrice", stock_id, start, end)
        if df.empty:
            return df

        df = df.rename(columns={
            "Trading_Volume": "volume",
            "Trading_money": "turnover",
            "max": "high",
            "min": "low",
        })
        # 只保留需要的欄位
        keep = ["date", "stock_id", "open", "high", "low", "close", "volume", "turnover", "spread"]
        df = df[[c for c in keep if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def fetch_institutional(
        self, stock_id: str, start: str, end: str | None = None
    ) -> pd.DataFrame:
        """抓取三大法人買賣超資料。

        回傳欄位: date, stock_id, name, buy, sell, net
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request(
            "TaiwanStockInstitutionalInvestorsBuySell", stock_id, start, end
        )
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

    def fetch_margin_trading(
        self, stock_id: str, start: str, end: str | None = None
    ) -> pd.DataFrame:
        """抓取融資融券資料。

        回傳欄位: date, stock_id, margin_buy, margin_sell, margin_balance,
                  short_sell, short_buy, short_balance
        """
        if end is None:
            end = date.today().isoformat()

        df = self._request(
            "TaiwanStockMarginPurchaseShortSale", stock_id, start, end
        )
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
            "date", "stock_id",
            "margin_buy", "margin_sell", "margin_balance",
            "short_sell", "short_buy", "short_balance",
        ]
        df = df[[c for c in keep if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
