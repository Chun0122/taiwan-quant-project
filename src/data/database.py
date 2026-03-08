"""資料庫引擎與 Session 管理。"""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import PROJECT_ROOT, settings


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


def get_db_watchlist() -> list[str]:
    """從 DB 取得 watchlist 股票 ID 清單；DB 為空或查詢失敗時回傳空 list。"""
    from src.data.schema import Watchlist  # local import 避免循環依賴

    try:
        with get_session() as session:
            rows = session.execute(select(Watchlist.stock_id).order_by(Watchlist.added_date, Watchlist.stock_id)).all()
            return [r[0] for r in rows]
    except Exception:
        return []


def get_effective_watchlist() -> list[str]:
    """取得有效 watchlist：DB 優先，settings.yaml fallback。

    若 DB watchlist 表非空，回傳 DB 中的股票清單；
    否則 fallback 至 settings.fetcher.watchlist（YAML 設定值）。
    """
    db_wl = get_db_watchlist()
    return db_wl if db_wl else list(settings.fetcher.watchlist)
