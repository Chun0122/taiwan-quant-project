"""設定管理模組 — 載入 config/settings.yaml 並提供統一存取介面。"""

from pathlib import Path

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class FinMindConfig(BaseModel):
    api_url: str = "https://api.finmindtrade.com/api/v4/data"
    api_token: str = ""


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///data/stock.db"


class FetcherConfig(BaseModel):
    default_start_date: str = "2020-01-01"
    watchlist: list[str] = ["2330"]


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Settings(BaseModel):
    finmind: FinMindConfig = FinMindConfig()
    database: DatabaseConfig = DatabaseConfig()
    fetcher: FetcherConfig = FetcherConfig()
    logging: LoggingConfig = LoggingConfig()


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    """從 YAML 檔案載入設定，檔案不存在時使用預設值。"""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return Settings(**raw)
    return Settings()


# 全域設定單例
settings = load_settings()
