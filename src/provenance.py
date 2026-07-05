"""決策可重放 provenance 純函數（A5，MASTER_PLAN §6.2 #8）。

每筆訊號/實驗攜帶 (git_commit, settings_hash) 版本戳，供任一歷史決策回溯
「系統當天跑的是哪版程式、哪組參數」。原實作位於 src/cli/experiment_cmd.py
（P2 任務 10），A5 抽至此供 DiscoveryRecord 蓋章與 ExperimentLog 共用——
兩處 hash 同語意，可互相對照。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any

from src.config import Settings

# 哪些 settings 區塊納入 hash + snapshot（研究相關，排除 API token 等 infra 設定）
RESEARCH_SETTINGS_SECTIONS: tuple[str, ...] = ("quant", "fetcher")


def sanitize_settings(s: Settings) -> dict[str, Any]:
    """從 Settings 抽研究相關區塊；不含 API token / webhook URL。

    保留：quant / fetcher.watchlist + fetcher.default_start_date
    移除：finmind / anthropic.api_key / discord.webhook_url / database / logging
    """
    out: dict[str, Any] = {}
    if "quant" in RESEARCH_SETTINGS_SECTIONS:
        out["quant"] = s.quant.model_dump()
    if "fetcher" in RESEARCH_SETTINGS_SECTIONS:
        out["fetcher"] = {
            "default_start_date": s.fetcher.default_start_date,
            "watchlist": list(s.fetcher.watchlist),
        }
    return out


def compute_settings_hash(sanitized: dict[str, Any]) -> str:
    """sha256 前 16 hex chars 作為 settings 指紋（idempotent，跨機器一致）。"""
    canonical = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def try_git_head() -> str | None:
    """取目前 git HEAD 短 hash；非 git 環境或逾時回 None（不 raise）。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def current_provenance(cfg: Settings) -> tuple[str | None, str]:
    """回傳 (git_commit, settings_hash) 版本戳——DiscoveryRecord 蓋章入口。"""
    return try_git_head(), compute_settings_hash(sanitize_settings(cfg))
