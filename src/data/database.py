"""資料庫引擎與 Session 管理。"""

import logging
import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)


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


def _resolve_db_path(url: str) -> Path | None:
    """��� SQLite URL 擷取實際檔案路徑。"""
    if url.startswith("sqlite:///"):
        path_str = url.replace("sqlite:///", "")
        return Path(path_str) if path_str and path_str != ":memory:" else None
    return None


engine = create_engine(
    _resolve_db_url(settings.database.url),
    echo=False,
    connect_args={"timeout": 30},  # busy_timeout 30 秒，防止並發鎖死
)


# 啟用 WAL mode + busy_timeout（每次新連線時設定）
@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """建立所有資料表（若不存在）。

    完整性檢查已移至 backup_db()（低頻執行），避免每次啟動耗時數秒。
    """
    Base.metadata.create_all(engine)


def backup_db() -> Path | None:
    """備份 SQLite 資料庫檔案（複製為 .bak 加時間戳）。

    備份前執行 PRAGMA integrity_check（每次備份時檢查，而非每次啟動）。

    Returns
    -------
    Path | None
        備份檔案路徑，或 None（非檔案型 DB 時）。
    """
    db_path = _resolve_db_path(_resolve_db_url(settings.database.url))
    if db_path is None or not db_path.exists():
        return None

    # 備份前完整性檢查
    try:
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA integrity_check")).scalar()
            if result != "ok":
                logger.error("SQLite integrity check 失敗: %s — 備份仍執行但請檢查 DB", result)
            else:
                logger.debug("SQLite integrity check: OK")
    except Exception:
        logger.warning("SQLite integrity check 無法執行", exc_info=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{db_path.stem}_{timestamp}.bak"

    shutil.copy2(db_path, backup_path)
    logger.info("SQLite 備份完成: %s", backup_path)

    # 清理超過 7 天的舊備份
    import time as _time

    cutoff = _time.time() - 7 * 86400
    for old_bak in backup_dir.glob("*.bak"):
        if old_bak.stat().st_mtime < cutoff:
            old_bak.unlink()
            logger.debug("刪除過期備份: %s", old_bak)

    return backup_path


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
