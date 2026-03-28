"""CLI Watchlist/概念股 子命令 — watchlist / sync-concepts / concepts / concept-expand。"""

from __future__ import annotations

import argparse

from src.cli.helpers import init_db
from src.cli.helpers import safe_print as print


def cmd_watchlist(args: argparse.Namespace) -> None:
    """觀察清單管理（add / remove / list / import）。

    DB-based watchlist 取代 settings.yaml watchlist：
    - add：新增股票至 DB watchlist
    - remove：從 DB watchlist 移除股票
    - list：列出 DB watchlist 清單
    - import：從 settings.yaml 一次性匯入所有股票
    """
    import datetime

    from sqlalchemy import select

    from src.config import settings
    from src.data.database import get_effective_watchlist, get_session, init_db
    from src.data.schema import Watchlist

    init_db()
    action: str | None = getattr(args, "wl_action", None)

    if action == "list" or action is None:
        with get_session() as session:
            rows = session.execute(select(Watchlist).order_by(Watchlist.added_date, Watchlist.stock_id)).scalars().all()
        if not rows:
            # DB 為空時顯示 YAML fallback
            yaml_wl = list(settings.fetcher.watchlist)
            print(f"DB watchlist 為空，目前使用 settings.yaml 清單（{len(yaml_wl)} 支）：")
            for sid in yaml_wl:
                print(f"  {sid}")
            print("\n使用 'watchlist import' 將 YAML 清單匯入 DB，或用 'watchlist add <stock_id>' 逐筆新增。")
            return
        print(f"{'股票ID':<8} {'股票名稱':<14} {'加入日期':<12} 備註")
        print("-" * 55)
        for row in rows:
            print(f"{row.stock_id:<8} {row.stock_name or '':<14} {str(row.added_date):<12} {row.note or ''}")
        print(f"\n共 {len(rows)} 支")

    elif action == "add":
        stock_id: str = args.stock_id
        with get_session() as session:
            existing = session.execute(select(Watchlist).where(Watchlist.stock_id == stock_id)).scalar_one_or_none()
            if existing:
                print(f"⚠️  {stock_id} 已在觀察清單中（加入日期：{existing.added_date}）")
                return
            session.add(
                Watchlist(
                    stock_id=stock_id,
                    stock_name=getattr(args, "name", None),
                    added_date=datetime.date.today(),
                    note=getattr(args, "note", None),
                )
            )
            session.commit()
        print(f"[OK] 已新增 {stock_id} 至觀察清單")
        print(f"   目前有效 watchlist：{len(get_effective_watchlist())} 支")

    elif action == "remove":
        stock_id = args.stock_id
        with get_session() as session:
            existing = session.execute(select(Watchlist).where(Watchlist.stock_id == stock_id)).scalar_one_or_none()
            if not existing:
                print(f"⚠️  {stock_id} 不在觀察清單中")
                return
            session.delete(existing)
            session.commit()
        print(f"[OK] 已從觀察清單移除 {stock_id}")
        print(f"   目前有效 watchlist：{len(get_effective_watchlist())} 支")

    elif action == "import":
        yaml_watchlist = list(settings.fetcher.watchlist)
        added = 0
        skipped = 0
        with get_session() as session:
            for sid in yaml_watchlist:
                existing = session.execute(select(Watchlist).where(Watchlist.stock_id == sid)).scalar_one_or_none()
                if existing:
                    skipped += 1
                else:
                    session.add(
                        Watchlist(
                            stock_id=sid,
                            added_date=datetime.date.today(),
                        )
                    )
                    added += 1
            session.commit()
        print(f"[OK] 從 settings.yaml 匯入完成：新增 {added} 支，已存在 {skipped} 支跳過")
        print(f"   DB watchlist 共 {added + skipped} 支")

    else:
        print(f"未知動作：{action}。可用動作：add / remove / list / import")


def cmd_sync_concepts(args: argparse.Namespace) -> None:
    """將 concepts.yaml 同步至 DB（ConceptGroup + ConceptMembership）。

    --purge      先清除 source=yaml 的舊記錄再重新匯入
    --from-mops  掃描近期 MOPS 公告以關鍵字自動標記概念成員
    --days N     MOPS 掃描回溯天數（預設 90，僅 --from-mops 有效）
    """
    from src.data.migrate import run_migrations
    from src.data.pipeline import sync_concept_tags_from_mops, sync_concepts_from_yaml

    run_migrations()

    if getattr(args, "from_mops", False):
        days = getattr(args, "days", 90)
        added = sync_concept_tags_from_mops(days=days)
        print(f"MOPS 關鍵字標記完成：新增 {added} 筆概念成員")
    else:
        purge = getattr(args, "purge", False)
        stats = sync_concepts_from_yaml(purge_yaml=purge)
        print(f"概念同步完成：新增概念組 {stats['groups']} 個，新增成員 {stats['members']} 筆")


def cmd_concepts(args: argparse.Namespace) -> None:
    """概念股管理（list / add / remove）。"""
    from datetime import date as _date

    from src.data.database import get_session
    from src.data.schema import ConceptGroup, ConceptMembership

    init_db()
    action: str | None = getattr(args, "concept_action", None)

    if action == "list" or action is None:
        concept_name: str | None = getattr(args, "concept_name", None)
        with get_session() as session:
            if concept_name:
                rows = (
                    session.query(ConceptMembership)
                    .filter(ConceptMembership.concept_name == concept_name)
                    .order_by(ConceptMembership.source, ConceptMembership.stock_id)
                    .all()
                )
                if not rows:
                    print(f"概念「{concept_name}」無成員記錄（或不存在）")
                    return
                print(f"概念「{concept_name}」成員清單（共 {len(rows)} 支）：")
                print(f"{'股票ID':<10} {'來源':<12} {'加入日期'}")
                print("-" * 38)
                for r in rows:
                    print(f"{r.stock_id:<10} {r.source:<12} {r.added_date}")
            else:
                groups = session.query(ConceptGroup).order_by(ConceptGroup.name).all()
                if not groups:
                    print("DB 無概念定義，請先執行 sync-concepts")
                    return
                print(f"{'概念名稱':<16} {'說明':<32} 成員數")
                print("-" * 60)
                for g in groups:
                    cnt = session.query(ConceptMembership).filter(ConceptMembership.concept_name == g.name).count()
                    print(f"{g.name:<16} {(g.description or '')[:30]:<32} {cnt}")

    elif action == "add":
        concept_name = args.concept_name
        stock_id: str = args.stock_id
        with get_session() as session:
            existing = (
                session.query(ConceptMembership)
                .filter(
                    ConceptMembership.concept_name == concept_name,
                    ConceptMembership.stock_id == stock_id,
                )
                .first()
            )
            if existing:
                print(f"⚠️  {stock_id} 已在概念「{concept_name}」中（source={existing.source}）")
                return
            # 確保 ConceptGroup 存在
            grp = session.query(ConceptGroup).filter(ConceptGroup.name == concept_name).first()
            if not grp:
                session.add(ConceptGroup(name=concept_name))
                session.commit()
            session.add(
                ConceptMembership(
                    concept_name=concept_name,
                    stock_id=stock_id,
                    source="manual",
                    added_date=_date.today(),
                )
            )
            session.commit()
        print(f"[OK] 已將 {stock_id} 新增至概念 [{concept_name}]（source=manual）")

    elif action == "remove":
        concept_name = args.concept_name
        stock_id = args.stock_id
        with get_session() as session:
            existing = (
                session.query(ConceptMembership)
                .filter(
                    ConceptMembership.concept_name == concept_name,
                    ConceptMembership.stock_id == stock_id,
                )
                .first()
            )
            if not existing:
                print(f"⚠️  {stock_id} 不在概念「{concept_name}」中")
                return
            session.delete(existing)
            session.commit()
        print(f"[OK] 已從概念 [{concept_name}] 移除 {stock_id}")

    else:
        print(f"未知動作：{action}。可用動作：list / add / remove")


def cmd_concept_expand(args: argparse.Namespace) -> None:
    """以價格相關性找出候選股，可選擇自動加入 DB（--auto）。"""
    from src.data.database import get_session
    from src.data.schema import ConceptMembership, DailyPrice
    from src.industry.concept_analyzer import (
        ConceptRotationAnalyzer,
        compute_concept_correlation_candidates,
    )

    init_db()
    concept_name: str = args.concept_name
    threshold: float = getattr(args, "threshold", 0.7)
    auto: bool = getattr(args, "auto", False)
    lookback: int = getattr(args, "lookback", 60)

    analyzer = ConceptRotationAnalyzer()
    concept_stocks = analyzer.get_concept_stocks()
    seed_stocks = concept_stocks.get(concept_name, [])
    if not seed_stocks:
        print(f"⚠️  概念「{concept_name}」無成員或不存在，請先執行 sync-concepts")
        return

    # 取所有有日K資料的股票為候選池
    from datetime import date as _date
    from datetime import timedelta

    cutoff = _date.today() - timedelta(days=lookback + 10)
    with get_session() as session:
        rows = session.query(DailyPrice.stock_id).filter(DailyPrice.date >= cutoff).distinct().all()
    all_stocks = [r[0] for r in rows]

    df_price = analyzer._load_price_data(all_stocks, lookback)
    result = compute_concept_correlation_candidates(
        concept_name=concept_name,
        seed_stocks=seed_stocks,
        candidate_stocks=all_stocks,
        df_price=df_price,
        lookback_days=lookback,
        threshold=threshold,
    )

    if result.empty:
        print(f"未找到與 [{concept_name}] 相關係數 >= {threshold} 的候選股")
        return

    print(f"概念「{concept_name}」相關性候選（門檻 {threshold}，共 {len(result)} 支）：")
    print(f"{'股票ID':<10} {'平均相關係數'}")
    print("-" * 25)
    for _, row in result.iterrows():
        print(f"{row['stock_id']:<10} {row['avg_corr']:.4f}")

    if auto:
        from datetime import date as _date2

        added = 0
        with get_session() as session:
            for _, row in result.iterrows():
                sid = row["stock_id"]
                existing = (
                    session.query(ConceptMembership)
                    .filter(
                        ConceptMembership.concept_name == concept_name,
                        ConceptMembership.stock_id == sid,
                    )
                    .first()
                )
                if not existing:
                    session.add(
                        ConceptMembership(
                            concept_name=concept_name,
                            stock_id=sid,
                            source="correlation",
                            added_date=_date2.today(),
                        )
                    )
                    added += 1
            session.commit()
        print(f"\n[OK] 已自動新增 {added} 支候選股至概念 [{concept_name}]（source=correlation）")
    else:
        print("\n提示：加上 --auto 可自動將候選股寫入 DB")
