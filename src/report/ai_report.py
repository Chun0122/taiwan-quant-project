"""AI 選股報告模組 — 呼叫 Claude API 生成繁體中文量化分析摘要。

使用方式：
    from src.report.ai_report import generate_ai_summary
    summary = generate_ai_summary(discover_result, regime="bull", top_stocks=df)
    print(summary)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.config import settings

if TYPE_CHECKING:
    from src.discovery.scanner import DiscoveryResult

logger = logging.getLogger(__name__)

_MODE_LABEL = {
    "momentum": "短線動能",
    "swing": "中期波段",
    "value": "價值修復",
    "dividend": "高息存股",
    "growth": "高成長",
}

_REGIME_LABEL = {
    "bull": "多頭",
    "bear": "空頭",
    "sideways": "盤整",
}


def _build_prompt(
    result: DiscoveryResult,
    regime: str,
    top_stocks: pd.DataFrame,
) -> str:
    """建立傳給 Claude 的結構化提示詞。"""
    mode_label = _MODE_LABEL.get(result.mode, result.mode)
    regime_label = _REGIME_LABEL.get(regime, regime)
    scan_date = str(result.scan_date)

    # 整理前五名股票資訊
    top5 = top_stocks.head(5)
    stocks_lines: list[str] = []
    for _, row in top5.iterrows():
        name = str(row.get("stock_name", ""))
        sid = str(row["stock_id"])
        close = float(row.get("close", 0))
        composite = float(row.get("composite_score", 0))
        technical = float(row.get("technical_score", 0))
        chip = float(row.get("chip_score", 0))
        fundamental = float(row.get("fundamental_score", 0))
        industry = str(row.get("industry_category", ""))
        chip_tier = str(row.get("chip_tier", "N/A"))
        entry = row.get("entry_price")
        stop = row.get("stop_loss")
        target = row.get("take_profit")

        entry_str = f"進場:{entry:.1f}" if pd.notna(entry) else ""
        stop_str = f"止損:{stop:.1f}" if pd.notna(stop) else ""
        target_str = f"目標:{target:.1f}" if pd.notna(target) else ""
        price_info = "  ".join(filter(None, [entry_str, stop_str, target_str]))

        stocks_lines.append(
            f"- {sid} {name}（{industry}）: 收盤={close:.1f}  "
            f"綜合={composite:.3f}  技術={technical:.3f}  "
            f"籌碼={chip:.3f}({chip_tier})  基本面={fundamental:.3f}" + (f"  {price_info}" if price_info else "")
        )

    stocks_text = "\n".join(stocks_lines)

    # 產業分布
    sector_text = ""
    if result.sector_summary is not None and not result.sector_summary.empty:
        top_sectors = result.sector_summary.head(5)
        sector_lines = [
            f"- {row['industry']}：{int(row['count'])} 支（均分 {row['avg_score']:.3f}）"
            for _, row in top_sectors.iterrows()
        ]
        sector_text = "\n產業分布（前五）：\n" + "\n".join(sector_lines)

    prompt = f"""你是一位台灣股市量化投資分析師，請根據以下量化數據，生成一份約 300 字的繁體中文選股摘要報告。

【掃描資訊】
- 掃描日期：{scan_date}
- 掃描模式：{mode_label}
- 市場狀態（Regime）：{regime_label}
- 全市場掃描股票數：{result.total_stocks} 支
- 粗篩通過：{result.after_coarse} 支
- 最終推薦：{len(top_stocks)} 支

【前五名推薦股票】
{stocks_text}
{sector_text}

【報告要求】
1. 首段（2-3 句）：說明當前市場狀態（{regime_label}市場）對本次 {mode_label} 掃描的影響
2. 中段（3-4 句）：重點介紹前三名股票的量化亮點（技術面、籌碼面、基本面各取重點）
3. 末段（2-3 句）：風險提示，包含本模式適合的持有週期與需注意的市場風險

請直接輸出報告內文，不要加標題、不要加分點符號，以流暢的段落呈現。"""

    return prompt


def generate_ai_summary(
    result: DiscoveryResult,
    regime: str = "sideways",
    top_stocks: pd.DataFrame | None = None,
) -> str:
    """呼叫 Claude API 生成量化選股摘要。

    Args:
        result:     DiscoveryResult 掃描結果
        regime:     市場狀態 "bull" | "bear" | "sideways"
        top_stocks: 要分析的股票 DataFrame（預設使用 result.rankings）

    Returns:
        約 300 字繁體中文摘要字串；若 API key 未設定或呼叫失敗，回傳錯誤訊息。
    """
    anthropic_cfg = getattr(settings, "anthropic", None)
    api_key = getattr(anthropic_cfg, "api_key", "") if anthropic_cfg else ""
    model = getattr(anthropic_cfg, "model", "claude-sonnet-4-6") if anthropic_cfg else "claude-sonnet-4-6"

    if not api_key:
        return "（AI 摘要未啟用：請在 config/settings.yaml 設定 anthropic.api_key）"

    try:
        import anthropic
    except ImportError:
        return "（AI 摘要未啟用：請執行 pip install anthropic）"

    if top_stocks is None:
        top_stocks = result.rankings

    prompt = _build_prompt(result, regime, top_stocks)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as exc:
        logger.warning("Claude API 呼叫失敗：%s", exc)
        return f"（AI 摘要生成失敗：{exc}）"
