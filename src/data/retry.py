"""HTTP 請求重試工具 — 為所有 fetcher 提供 exponential backoff 重試邏輯。"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

# 預設可重試的 HTTP 狀態碼
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    timeout: int | float = 30,
    retryable_status_codes: set[int] | None = None,
    session: requests.Session | None = None,
    **kwargs,
) -> requests.Response:
    """帶 exponential backoff 的 HTTP 請求。

    Parameters
    ----------
    method : str
        HTTP 方法（"GET" 或 "POST"）。
    url : str
        請求 URL。
    max_retries : int
        最大重試次數（不含首次請求，預設 3）。
    base_delay : float
        首次重試的等待秒數（預設 2.0），後續翻倍。
    timeout : int | float
        單次請求超時秒數。
    retryable_status_codes : set[int] | None
        可重試的 HTTP 狀態碼集合，None 時使用預設值。
    session : requests.Session | None
        可選的 Session 物件，None 時使用 requests 全域函數。
    **kwargs
        傳遞給 requests 的其他參數（params, json, headers, verify 等）。

    Returns
    -------
    requests.Response
        成功的回應物件。

    Raises
    ------
    requests.RequestException
        所有重試用盡後仍失敗時拋出最後一次的例外。
    """
    retryable = retryable_status_codes or _RETRYABLE_STATUS_CODES
    requester = session or requests
    last_exc: Exception | None = None

    for attempt in range(1 + max_retries):
        try:
            method_func = getattr(requester, method.lower())
            resp = method_func(url, timeout=timeout, **kwargs)

            # 非可重試狀態碼 → 直接回傳（含 4xx 非 429）
            if resp.status_code not in retryable:
                return resp

            # 可重試狀態碼 → 記錄並重試
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "HTTP %d from %s — 第 %d/%d 次重試（等待 %.1fs）",
                    resp.status_code,
                    url[:80],
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
            else:
                # 最後一次也失敗 → raise
                resp.raise_for_status()

        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "請求失敗 (%s) %s — 第 %d/%d 次重試（等待 %.1fs）",
                    type(exc).__name__,
                    url[:80],
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
            else:
                raise

    # 理論上不會到達此處
    if last_exc:
        raise last_exc
    raise RuntimeError("request_with_retry 內部錯誤")  # pragma: no cover
