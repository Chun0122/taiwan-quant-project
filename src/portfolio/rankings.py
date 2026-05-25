"""Rotation 排名解析（P2 任務 14 phase 1：從 manager.py 抽出）。

從 DiscoveryRecord 解析各模式排名 + 進場理由 breakdown。
無 RotationManager 依賴，可獨立測試。
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import select

from src.data.schema import DiscoveryRecord


def resolve_rankings(
    mode: str,
    scan_date: date,
    session,
    top_n: int = 50,
    per_mode_max: int | None = None,
) -> list[dict]:
    """從 DiscoveryRecord 解析指定日期的排名。

    Parameters
    ----------
    mode : str
        'momentum'/'swing'/.../'all'。
    scan_date : date
        掃描日期。
    session : SQLAlchemy Session
    top_n : int
        最大取用筆數。
    per_mode_max : int | None
        僅對 mode='all' 生效；每個 primary_mode 最多 N 檔（避免單 mode 集中爆雷）。
        None = 取 constants.ROTATION_ALL_MODE_PER_MODE_MAX 預設；0 或負值 = 不限制。

    Returns
    -------
    list[dict]
        按排名排序的清單，每筆含：
        {stock_id, stock_name, rank, score, close, stop_loss}
        （mode='all' 時額外含 primary_mode）
    """
    if mode == "all":
        return _resolve_all_mode_rankings(scan_date, session, top_n, per_mode_max=per_mode_max)

    stmt = (
        select(DiscoveryRecord)
        .where(DiscoveryRecord.scan_date == scan_date, DiscoveryRecord.mode == mode)
        .order_by(DiscoveryRecord.rank)
        .limit(top_n)
    )
    records = session.execute(stmt).scalars().all()
    return [
        {
            "stock_id": r.stock_id,
            "stock_name": r.stock_name or "",
            "rank": r.rank,
            "score": r.composite_score,
            "close": r.close,
            "stop_loss": r.stop_loss,
            "score_breakdown": _record_to_score_breakdown(r),
        }
        for r in records
    ]


def _record_to_score_breakdown(r: DiscoveryRecord, *, primary_mode: str | None = None) -> dict:
    """單筆 DiscoveryRecord → 進場理由 audit dict（JSON-serializable）。

    供 P1 任務 5 凍結進場 rationale；即使日後 scanner 規則改動，仍可回溯當時選股理由。
    """
    out: dict = {
        "scan_date": r.scan_date.isoformat() if r.scan_date else None,
        "mode": r.mode,
        "rank": r.rank,
        "composite_score": r.composite_score,
        "regime": r.regime,
        "scores": {
            "chip": r.chip_score,
            "technical": r.technical_score,
            "fundamental": r.fundamental_score,
            "news": r.news_score,
        },
        "chip_tier": r.chip_tier,
        "chip_tier_change": r.chip_tier_change,
        "concept_bonus": r.concept_bonus,
        "daytrade_penalty": r.daytrade_penalty,
        "discovery_record_id": r.id,
    }
    if primary_mode is not None:
        out["primary_mode"] = primary_mode
    return out


def _resolve_all_mode_rankings(
    scan_date: date,
    session,
    top_n: int,
    per_mode_max: int | None = None,
) -> list[dict]:
    """解析 'all' 模式排名 — 所有模式取 avg_score 排序，可選 primary_mode 配額。

    Parameters
    ----------
    per_mode_max : int | None
        每個 primary_mode 最多 N 檔。primary_mode 定義：該股票在各模式
        discovery_record 中 composite_score 最高的 mode。
        None = 取 constants.ROTATION_ALL_MODE_PER_MODE_MAX 預設；0 或負值 = 不限制。

        2026-05-15 audit：all10_5d 5/7-5/8 從 swing 模式同時進 4 檔導致集中爆雷，
        加入此配額避免單一 mode 因子失效時整組合受重傷。
    """
    from src.constants import ROTATION_ALL_MODE_PER_MODE_MAX

    if per_mode_max is None:
        per_mode_max = ROTATION_ALL_MODE_PER_MODE_MAX

    stmt = select(DiscoveryRecord).where(DiscoveryRecord.scan_date == scan_date)
    records = session.execute(stmt).scalars().all()

    # 按 stock_id 分組，計算 avg_score + 紀錄各 mode 分數 + 保留每 mode 最佳 record（給 breakdown）
    stock_data: dict[str, dict] = {}
    stock_records: dict[str, dict[str, DiscoveryRecord]] = {}  # sid → {mode → record (最高分)}
    for r in records:
        sid = r.stock_id
        prev_rec = stock_records.setdefault(sid, {}).get(r.mode)
        if prev_rec is None or r.composite_score > prev_rec.composite_score:
            stock_records[sid][r.mode] = r
        if sid not in stock_data:
            stock_data[sid] = {
                "stock_id": sid,
                "stock_name": r.stock_name or "",
                "close": r.close,
                "stop_loss": r.stop_loss,
                "scores": [],
                "mode_scores": {},
            }
        stock_data[sid]["scores"].append(r.composite_score)
        # 同一 mode 若多筆紀錄，取最高分（防禦性，正常 unique(scan_date, mode, stock_id)）
        prev = stock_data[sid]["mode_scores"].get(r.mode, float("-inf"))
        if r.composite_score > prev:
            stock_data[sid]["mode_scores"][r.mode] = r.composite_score
        # 保留最嚴格的 stop_loss（最高的）
        existing_sl = stock_data[sid]["stop_loss"]
        if r.stop_loss is not None:
            if existing_sl is None or r.stop_loss > existing_sl:
                stock_data[sid]["stop_loss"] = r.stop_loss

    # 計算 avg_score 並排序
    ranked = []
    for sid, data in stock_data.items():
        avg_score = sum(data["scores"]) / len(data["scores"])
        primary_mode = max(data["mode_scores"].items(), key=lambda kv: kv[1])[0] if data["mode_scores"] else "unknown"
        # 取 primary_mode 對應的 record 作為 breakdown 主體（保留各 mode 分數於 mode_scores）
        primary_record = stock_records.get(sid, {}).get(primary_mode)
        breakdown: dict | None = None
        if primary_record is not None:
            breakdown = _record_to_score_breakdown(primary_record, primary_mode=primary_mode)
            breakdown["mode_scores"] = dict(data["mode_scores"])  # 揭露各模式 composite_score
            breakdown["avg_score"] = round(avg_score, 6)
            breakdown["mode"] = "all"  # 來源 portfolio 是 all
        ranked.append(
            {
                "stock_id": sid,
                "stock_name": data["stock_name"],
                "score": avg_score,
                "close": data["close"],
                "stop_loss": data["stop_loss"],
                "primary_mode": primary_mode,
                "score_breakdown": breakdown,
            }
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)

    # mode 配額（per_mode_max > 0 啟用）
    if per_mode_max is not None and per_mode_max > 0:
        mode_count: dict[str, int] = {}
        filtered: list[dict] = []
        for r in ranked:
            m = r.get("primary_mode", "unknown")
            if mode_count.get(m, 0) < per_mode_max:
                filtered.append(r)
                mode_count[m] = mode_count.get(m, 0) + 1
            if len(filtered) >= top_n:
                break
        ranked = filtered

    # 加上排名
    for i, r in enumerate(ranked[:top_n], 1):
        r["rank"] = i

    return ranked[:top_n]
