# Daily Dashboard JSON Schema

`python main.py export-dashboard` 與 `morning-routine` Step 16 產出的單一日報檔。供 iOS 監控 App、Watch 小工具、外部 BI 等下游消費者使用。

- **產出路徑**：預設 `~/Library/Mobile Documents/com~apple~CloudDocs/QuantDashboard/`，可透過 `settings.yaml` 的 `dashboard.out_dir` 或 CLI `--out` 覆寫。
- **檔案**：`<out_dir>/<YYYY-MM-DD>.json` 與 `<out_dir>/latest.json`（兩者內容相同；後者為下游 App 固定路徑入口）。
- **產生失敗哲學**：任一區塊失敗皆 graceful degradation — 該欄位填 `null` 並寫入 `errors[]`，其餘區塊照常輸出。

---

## Top-level

| 欄位 | 型別 | 必填 | 說明 |
|-----|-----|------|-----|
| `version` | int | ✓ | Schema 版本，現為 `4`；breaking change 時遞增。 |
| `generated_at` | string (ISO 8601) | ✓ | 產出時間（含時區）。 |
| `date` | string (YYYY-MM-DD) | ✓ | 報告對應日期。 |
| `regime` | object | ✓ | 市場狀態。 |
| `discover` | object | ✓ | 五模式推薦（每模式可為空陣列）。 |
| `rotation` | object \| null | ✓ | **v1 backward-compat 別名** = `rotations[0]`（即 primary）；無 active 時為 `null`。 |
| `rotations` | array | ✓ | 全部 active 輪動組合（依 `current_capital` 降冪；v2 新增）。 |
| `watch_entries` | array | ✓ | 持倉監控（status=active）。 |
| `signals` | array | ✓ | 異常 / 警告訊號（可為空）。 |
| `strategy_events` | array | ✓ | 策略調整事件（可為空）。 |
| `ai_summary` | string \| null | ✓ | AI 摘要文字；未啟用 / 無快取時為 `null`。 |
| `portfolio_review` | object \| null | ✓ | **primary 組合**每日績效摘要；無 active rotation 時為 `null`（v1.1 新增）。 |
| `position_timeseries` | object \| null | ✓ | **聚合所有 rotations 的持倉 ∪ Watch** 最近 N 個交易日 close（v1.1 新增；v2 改為跨組合聚合）；無持倉時為 `null`。 |
| `alpha_chart` | object \| null | ✓ | **跨組合 alpha vs 0050 時序**（v3 新增）；所有 active rotation 全部 snapshot 之 alpha_cum_pct/benchmark_cum_return_pct 長表；無資料時為 `null`。 |
| `errors` | array of string | ✓ | 子區塊產出失敗訊息（空陣列代表全部成功）。 |

> **v4 schema 變更**（2026-06-06）：
> - `SCHEMA_VERSION` 從 `3` 升至 `4`。
> - 每個 `RotationBlock` 新增 `today_actions`：當日該組合的操作明細（回答「今天各 Rotation 做了什麼」），來源為新表 `rotation_action_log`（`RotationManager.update()` 落庫；`rotation preview` dry_run 不寫）。
> - 結構為 `[{action_type, reason, is_risk_exit, switch_group, stock_id, stock_name, shares, price, entry_rank, pnl, return_pct}, ...]`；`action_type ∈ {open, close, renew, hold}`；無異動時為 `[]`。舊版下游請設為 optional。詳見 `rotations[]` 區塊說明。

> **v3 schema 變更**（2026-05-17）：
> - `SCHEMA_VERSION` 從 `2` 升至 `3`。
> - 新增 `alpha_chart`：跨組合 alpha vs 0050 長表時序，5/29 audit「alpha 拖累歸因」串接點。
> - 結構為 `{lookback_days, series: [{date, name, alpha_cum_pct, benchmark_cum_return_pct, portfolio_cum_return_pct, total_capital}, ...]}`；downstream 可直接 pivot by `name` 畫多線圖。

> **v2 schema 變更**（2026-05）：
> - `SCHEMA_VERSION` 從 `1` 升至 `2`。
> - 新增 `rotations: [RotationBlock]`，承載**全部** active 輪動組合（先前 `rotation` 只挑 capital 最大者，導致多組同時跑時無法在下游全部呈現）。
> - 保留 `rotation` 欄位作為 v1 backward-compat alias，內容等同 `rotations[0]`。新下游請改用 `rotations`。
> - `position_timeseries` 的 `series` keys 來源從「primary 持倉 ∪ watch」擴大為「所有 rotations 持倉 ∪ watch」。
>
> v1.1 為向下相容增量（不升版號）：`portfolio_review` / `position_timeseries` 為新增欄位，舊版下游若以嚴格 Codable 解析，新欄位請設為 optional。

---

## `regime`（來自 `_compute_macro_stress_check()`）

```json
{
  "state": "bull",
  "crisis_triggered": false,
  "breadth_downgraded": false,
  "breadth_below_ma20_pct": 0.62,
  "taiex_close": 23105.5,
  "fast_return_5d": -0.012,
  "consec_decline_days": 0,
  "vol_ratio": 1.05,
  "vix_val": 15.2,
  "us_vix_val": 18.4,
  "summary": "市場狀態=bull TAIEX=23105，5日=-1.2%，MA20寬度=62%"
}
```

- `state`：`"bull" | "bear" | "sideways" | "crisis" | null`（壓力預檢失敗時 `null`）
- 其他欄位若計算失敗則為 `null` 或 `0.0`（沿用 `_compute_macro_stress_check()` 既有 fallback）

---

## `discover`（來自 `DiscoveryRecord` 表，當日資料）

```json
{
  "momentum": [ {DiscoveryItem}, ... ],
  "swing": [],
  "value": [],
  "dividend": [],
  "growth": []
}
```

每個模式為陣列，依 `rank` 升冪排序。預設取 Top 20，可用 `--top` 覆寫。

### `DiscoveryItem`

| 欄位 | 型別 | 來源 |
|-----|-----|-----|
| `rank` | int | DiscoveryRecord.rank |
| `stock_id` | string | DiscoveryRecord.stock_id |
| `stock_name` | string \| null | DiscoveryRecord.stock_name |
| `close` | float | DiscoveryRecord.close |
| `composite_score` | float | DiscoveryRecord.composite_score |
| `scores` | object | `{technical, chip, fundamental, news}`，各為 float \| null |
| `entry` | float \| null | DiscoveryRecord.entry_price |
| `stop_loss` | float \| null | DiscoveryRecord.stop_loss |
| `take_profit` | float \| null | DiscoveryRecord.take_profit |
| `industry` | string \| null | DiscoveryRecord.industry_category |
| `regime` | string \| null | DiscoveryRecord.regime |
| `valid_until` | string (YYYY-MM-DD) \| null | DiscoveryRecord.valid_until |
| `chip_tier` | string \| null | `"3F" .. "8F"` 或 `null` |
| `chip_tier_change` | string \| null | 例 `"7F→8F"` |
| `concept_bonus` | float \| null | DiscoveryRecord.concept_bonus |
| `daytrade_penalty` | float \| null | DiscoveryRecord.daytrade_penalty |
| `entry_trigger` | string \| null | 進場觸發條件描述 |

---

## `rotation` / `rotations`（來自 `RotationManager.list_portfolios()` + `get_status()`）

`rotations` 為全部 active 組合的陣列（v2 新增），依 `current_capital` 降冪排序，`rotations[0]` 即 primary。
`rotation` 為 v1 backward-compat alias = `rotations[0]`；無 active 組合時 `rotation=null` 且 `rotations=[]`。

```json
{
  "rotation": {RotationBlock},
  "rotations": [
    {RotationBlock},  // primary（current_capital 最大）
    {RotationBlock},  // 次要
    ...
  ]
}
```

每個 `RotationBlock`：

```json
{
  "name": "default",
  "mode": "momentum",
  "max_positions": 5,
  "holding_days": 10,
  "allow_renewal": true,
  "initial_capital": 1000000,
  "current_capital": 1050000,
  "current_cash": 200000,
  "total_market_value": 850000,
  "total_unrealized_pnl": 45000,
  "total_return_pct": 0.05,
  "status": "active",
  "updated_at": "2026-05-01T18:30:00",
  "holdings": [ {RotationHolding}, ... ],
  "today_actions": [ {RotationAction}, ... ]
}
```

### `RotationAction`（v4 新增 — 今日操作）

當日該組合的操作明細，回答「今天這個 Rotation 做了什麼」。來源為 `rotation_action_log`
（`RotationManager.update()` 落庫，同日重跑冪等覆寫；`rotation preview` dry_run 不寫）。無異動時為 `[]`。

| 欄位 | 型別 | 說明 |
|-----|-----|-----|
| `action_type` | string | `open`（買入）/ `close`（賣出）/ `renew`（續持）/ `hold`（保持不動） |
| `reason` | string \| null | 賣出原因：`holding_expired` / `stop_loss` / `crisis_exit` / `max_drawdown_liquidation`；open/renew/一般 hold 為 `null`（gated hold 可能為 `gate_b_score_gap` / `gate_c_weekly_cap`） |
| `is_risk_exit` | bool | 風控出場（`stop_loss` / `crisis_exit` / `max_drawdown_liquidation`），UI 以 ⚠️ 區分 |
| `switch_group` | string \| null | 同日非風控賣出＋買入配對群組 id（「換股」🔁）；無配對為 `null` |
| `stock_id` | string | |
| `stock_name` | string \| null | |
| `shares` | int \| null | |
| `price` | float \| null | 買入為 entry_price、賣出為 exit_price |
| `entry_rank` | int \| null | open 為當日排名；close/renew/hold 為進場時排名 |
| `pnl` | float \| null | 僅 close：以 exit/entry 概算（不含成本，與 RotationPosition 已實現損益略有差異） |
| `return_pct` | float \| null | 僅 close：概算報酬率（%） |

> **引擎現實**：此 rotation 引擎交易整筆部位，無部分加碼/減碼，故 `action_type` 不含 add/reduce。
> 「換股」非獨立 action_type，而是以 `switch_group` 標記同日的 close + open 組合。

### `RotationHolding`

| 欄位 | 型別 | 說明 |
|-----|-----|-----|
| `stock_id` | string | |
| `stock_name` | string \| null | |
| `entry_date` | string (YYYY-MM-DD) | |
| `entry_price` | float | |
| `current_price` | float | 從 DailyPrice 最新收盤取得；若無則等於 entry_price |
| `shares` | int | |
| `market_value` | float | current_price × shares |
| `unrealized_pnl` | float | |
| `unrealized_pct` | float | |
| `entry_rank` | int \| null | 進場時的排名 |
| `entry_breakdown` | object \| null | 進場時凍結的選股 rationale（v3.1 新增 P1 任務 5）；歷史 position 與 schema 變更前可能為 `null`。詳見下方。 |

#### `entry_breakdown`（凍結進場理由）

```json
{
  "scan_date": "2026-05-15",
  "mode": "momentum",
  "rank": 1,
  "composite_score": 0.85,
  "regime": "bull",
  "scores": {
    "chip": 0.72,
    "technical": 0.0,
    "fundamental": 0.91,
    "news": 0.55
  },
  "chip_tier": "7F",
  "chip_tier_change": null,
  "concept_bonus": 0.05,
  "daytrade_penalty": null,
  "discovery_record_id": 12345,
  "primary_mode": "swing",     // 僅 portfolio.mode='all' 時出現
  "mode_scores": {              // 僅 portfolio.mode='all' 時出現
    "momentum": 0.6,
    "swing": 0.9
  },
  "avg_score": 0.75             // 僅 portfolio.mode='all' 時出現
}
```

- 寫入時機：`_execute_buy()` 時序列化當下 DiscoveryRecord 內容到 `RotationPosition.entry_score_breakdown_json`。
- 用途：debug「為何進這檔」；日後 scanner 規則改動仍可回溯當時 rationale，是 audit drill-down 主要入口。
- 補洞：`RotationManager.backfill_entry_score_breakdown(session, portfolio_name=None, overwrite=False)` 可回填歷史 position（依 entry_date + portfolio.mode + stock_id 反查 DiscoveryRecord）。

---

## `watch_entries`（來自 `WatchEntry` 表，status="active"）

```json
[
  {
    "id": 12,
    "stock_id": "2330",
    "stock_name": "台積電",
    "entry_date": "2026-04-20",
    "entry_price": 1050.0,
    "stop_loss": 1020.0,
    "take_profit": 1120.0,
    "quantity": 1000,
    "source": "discover",
    "mode": "momentum",
    "status": "active",
    "trailing_stop_enabled": false,
    "highest_price_since_entry": null,
    "valid_until": "2026-05-08",
    "notes": null
  }
]
```

---

## `signals`（合併多來源警示）

```json
[
  {
    "type": "ic_decay",
    "severity": "warning",
    "message": "Momentum 關鍵因子 mom5_3d IC=-0.08（反向→已暫停）",
    "target": "momentum.mom5_3d"
  },
  {
    "type": "crisis",
    "severity": "critical",
    "message": "CRISIS 崩盤訊號觸發：5日=-6.2%，連跌5天，VIX=32",
    "target": null
  }
]
```

| `type` | `severity` | 來源 |
|--------|-----------|-----|
| `crisis` | `critical` | `_compute_macro_stress_check()` 的 `crisis_triggered=True` |
| `bear_market` | `warning` | regime=`bear` |
| `breadth_downgrade` | `warning` | `breadth_downgraded=True` |
| `ic_decay` | `warning`（weak/decay）/ `critical`（inverse）| `_compute_factor_ic_status()` |
| `ic_failure` | `warning` | IC 計算 `level="error"` |
| `data_stale` | `warning`（>3 天）/ `critical`（>7 天）| `_verify_data_freshness()` |
| `kill_switch` | `critical` | 待 v2 實作 |

`severity` enum：`info | warning | critical`。

---

## `strategy_events`（來自 `src/discovery/strategy_events.py`）

```json
[
  {
    "date": "2026-04-30",
    "type": "git_commit",
    "summary": "收緊 universe filter（min_close 5→30, volume_ratio_min None→1.0）",
    "ref": "5aba2ac",
    "details": {"author": "Chun0122", "files_changed": 3}
  },
  {
    "date": "2026-04-25",
    "type": "settings_diff",
    "summary": "config/settings.yaml: quant.score_threshold.bull 0.45 → 0.50",
    "ref": "abc1234",
    "details": {"field": "quant.score_threshold.bull", "before": 0.45, "after": 0.50}
  }
]
```

| `type` | 說明 | 來源 |
|--------|-----|-----|
| `git_commit` | 近 30 天 commits（過濾 `feat/fix/refactor` 範圍） | `git log` |
| `settings_diff` | `config/settings.yaml` 在最近 commit 的 diff | `git diff` |
| `ic_auto_adjust` | 預留，v2 |
| `kill_switch` | 預留，v2 |

預設取最近 30 天，依 `date` 降冪排序。

---

## `ai_summary`

morning-routine 既有的 AI 摘要（若 settings.yaml 有 `anthropic.api_key`）。`export-dashboard` 預設**不**重新呼叫 API（避免每次 export 燒 token）；要重呼叫請加 `--regenerate-ai-summary`。

---

## `portfolio_review`（v1.1，來自 `RotationDailySnapshot`）

```json
{
  "today_pnl_pct": 0.0035,
  "wtd_return_pct": 0.012,
  "mtd_return_pct": 0.024,
  "total_return_pct": 0.05,
  "sharpe_ratio": 1.42,
  "max_drawdown_pct": 4.5,
  "win_rate_pct": 56.2,
  "equity_curve": [
    {"date": "2026-04-01", "capital": 1000000.0},
    {"date": "2026-04-02", "capital": 1004500.0}
  ],
  "snapshots_count": 30
}
```

| 欄位 | 型別 | 說明 |
|-----|-----|-----|
| `today_pnl_pct` | float \| null | 與前一筆 snapshot 比較的當日報酬；首日 / 無前日為 `null` |
| `wtd_return_pct` | float \| null | 自本週週一以來累積報酬；無資料 `null` |
| `mtd_return_pct` | float \| null | 自本月 1 號以來累積報酬；無資料 `null` |
| `total_return_pct` | float \| null | 來自 `rotation.total_return_pct`，自初始資金以來累積 |
| `sharpe_ratio` | float \| null | 年化（×√252）；snapshot 數 <10 強制 `null` |
| `max_drawdown_pct` | float \| null | 0~100；snapshot 數 <3 強制 `null` |
| `win_rate_pct` | float \| null | daily_return_pct>0 的日數比例；snapshot 數 <3 強制 `null` |
| `equity_curve` | array | `[{date, capital}]` asc by date；可繪製淨值曲線 |
| `snapshots_count` | int | 撈到的 snapshot 筆數（受 `dashboard.portfolio_review_lookback_days` 限制，預設 90） |

**斷鏈限制**：rotation 改名後新舊 snapshot 的 `portfolio_name` 不一致時，舊資料不會被列入。

---

## `position_timeseries`（v1.1）

```json
{
  "trading_days": ["2026-04-15", "2026-04-16", ..., "2026-05-02"],
  "series": {
    "2330": {"close": [580.0, 585.0, ...], "first_idx": 0},
    "2317": {"close": [62.5, 63.0, 64.0], "first_idx": 11}
  }
}
```

- 來源：`DailyPrice` 表，`stock_id` 為 `rotation.holdings ∪ watch_entries` 去重集合。
- `trading_days`：自然從 `DailyPrice` 取得，已排除週末/假日，最多 `dashboard.position_timeseries_days`（預設 14）筆。
- `series[sid].close`：與 `trading_days[first_idx:]` 對齊；長度通常為 `len(trading_days) - first_idx`。
- `first_idx`：上市未滿 N 日 / 中段停牌的股票只給「連續最末段」起點；正常股 `first_idx=0`。
- 無持倉時整個欄位為 `null`。

---

## `alpha_chart`（v3，跨組合 alpha vs 0050 時序）

```json
{
  "lookback_days": 90,
  "series": [
    {
      "date": "2026-05-15",
      "name": "all10_5d",
      "alpha_cum_pct": 0.1101,
      "benchmark_cum_return_pct": -0.0037,
      "portfolio_cum_return_pct": 0.1064,
      "total_capital": 1106467.12
    },
    {
      "date": "2026-05-15",
      "name": "swing5_3d",
      "alpha_cum_pct": -0.0168,
      "benchmark_cum_return_pct": -0.0037,
      "portfolio_cum_return_pct": -0.0205,
      "total_capital": 979527.7
    }
  ]
}
```

- 來源：所有 active rotation 各自 `RotationDailySnapshot` 最近 `dashboard.portfolio_review_lookback_days`（預設 90）筆。
- Long-format：每 `(date, name)` 一筆，downstream pivot by `name` 即可畫多線圖（Streamlit `pages/portfolio_review.py` 已串）。
- `alpha_cum_pct`：自 portfolio 第一筆 snapshot 起，組合累積報酬 − 0050 累積報酬。
- `benchmark_cum_return_pct`：自同一 base_date 起 0050 累積報酬。
- `portfolio_cum_return_pct`：= `alpha_cum_pct + benchmark_cum_return_pct`，便利下游直接畫淨值曲線。
- 缺資料策略：任一 portfolio 缺 alpha 則該列被略過；全部缺則整個欄位為 `null`。
- 對應 CLI：`python main.py rotation cost-attribution --name X`（成本歸因）與本欄位（alpha 時序）互補，audit 可拆「alpha 拖累 = 成本 + 選股 alpha」。

---

## Universe 統計時序（Streamlit 面板，**非** JSON 區塊）

P1 任務 8：`UniverseStatLog` 表記每次 UniverseFilter.run() 各階段剩餘股數。Streamlit
`pages/universe_stats.py` 顯示時序圖 + 漏斗圖。資料來源：`load_universe_stat_log(lookback_days, mode)`。

欄位：`scan_date / mode / regime / total_after_sql / total_after_liquidity /
total_after_trend / from_memory / final_candidates / turnover_multiplier`。

unique constraint：`(scan_date, mode)`；同日多次 scan 取最後一次（upsert）。

---

## 驗證範例

```bash
# 產生今日報告
python main.py export-dashboard

# 指定日期 + 自訂路徑
python main.py export-dashboard --date 2026-04-30 --out /tmp/dashboard

# JSON 結構檢查
jq '.version, .regime.state, (.discover.momentum | length), (.signals | length)' \
  ~/Library/Mobile\ Documents/com~apple~CloudDocs/QuantDashboard/latest.json

# v1.1 新區塊驗證
jq '.portfolio_review.snapshots_count, .portfolio_review.sharpe_ratio,
    (.position_timeseries.trading_days | length),
    (.position_timeseries.series | keys | length)' \
  ~/Library/Mobile\ Documents/com~apple~CloudDocs/QuantDashboard/latest.json
```
