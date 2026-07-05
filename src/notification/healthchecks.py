"""healthchecks.io dead-man 告警 ping（P0 止血包 #5，2026-07-05）。

morning-routine 起跑 ping start、完成 ping success、有步驟失敗 ping fail。
healthchecks.io 端設定排程 + grace period：未在期限內收到 ping（筆電沒開、
launchd 壞掉、routine 卡死）即發告警信——覆蓋「靜默失敗」情境。

設計原則：告警機制本身絕不影響主流程——URL 未設定直接略過，
任何網路例外吞掉只記 warning。
"""

from __future__ import annotations

import logging

from src.config import settings
from src.data.retry import request_with_retry

logger = logging.getLogger(__name__)

# ping 種類 → URL 尾綴（healthchecks.io 慣例）
_KIND_SUFFIX = {
    "start": "/start",
    "success": "",
    "fail": "/fail",
}


def ping_healthchecks(kind: str = "success") -> bool:
    """對 healthchecks.io 發送 ping。

    Parameters
    ----------
    kind : str
        "start"（開始執行）/ "success"（成功完成）/ "fail"（有步驟失敗）。

    Returns
    -------
    bool
        True = ping 送達；False = 未設定 URL 或發送失敗（僅記 log，不 raise）。
    """
    url = settings.monitoring.healthchecks_url
    if not url:
        return False
    if kind not in _KIND_SUFFIX:
        logger.warning("未知的 healthchecks ping 種類：%s", kind)
        return False

    target = url.rstrip("/") + _KIND_SUFFIX[kind]
    try:
        resp = request_with_retry("GET", target, timeout=10, max_retries=2)
        ok = resp is not None and resp.status_code == 200
        if ok:
            logger.debug("healthchecks ping (%s) 成功", kind)
        else:
            logger.warning("healthchecks ping (%s) 回應異常：%s", kind, getattr(resp, "status_code", None))
        return ok
    except Exception:
        # 告警機制絕不影響主流程
        logger.warning("healthchecks ping (%s) 失敗", kind, exc_info=True)
        return False
