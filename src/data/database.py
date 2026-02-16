"""資料庫引擎與 Session 管理。"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings, PROJECT_ROOT


class Base(DeclarativeBase):
    pass


def _resolve_db_url(url: str) -> str:
    """將相對路徑的 SQLite URL 轉為絕對路徑，並確保目錄存在。"""
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        relative = url.replace("sqlite:///", "")
        absolute = PROJECT_ROOT / relative
        absolute.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{absolute}"
    return url


engine = create_engine(_resolve_db_url(settings.database.url), echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """建立所有資料表（若不存在）。"""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """取得一個新的 DB Session。"""
    return SessionLocal()
