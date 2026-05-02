# Daily Dashboard JSON Schema

`python main.py export-dashboard` 與 `morning-routine` Step 16 產出的單一日報檔。供 iOS 監控 App、Watch 小工具、外部 BI 等下游消費者使用。

- **產出路徑**：預設 `~/Library/Mobile Documents/com~apple~CloudDocs/QuantDashboard/`，可透過 `settings.yaml` 的 `dashboard.out_dir` 或 CLI `--out` 覆寫。
- **檔案**：`<out_dir>/<YYYY-MM-DD>.json` 與 `<out_dir>/latest.json`（兩者內容相同；後者為下游 App 固定路徑入口）。
- **產生失敗哲學**：任一區塊失敗皆 graceful degradation — 該欄位填 `null` 並寫入 `errors[]`，其餘區塊照常輸出。

---

## Top-level

| 欄位 | 型別 | 必填 | 說明 |
|-----|-----|------|-----|
| `version` | int | ✓ | Schema 版本，現為 `1`；breaking change 時遞增。 |
| `generated_at` | string (ISO 8601) | ✓ | 產出時間（含時區）。 |
| `date` | string (YYYY-MM-DD) | ✓ | 報告對應日期。 |
| `regime` | object | ✓ | 市場狀態。 |
| `discover` | object | ✓ | 五模式推薦（每模式可為空陣列）。 |
| `rotation` | object \| null | ✓ | 主要輪動組合狀態；無 active 組合時為 `null`。 |
| `watch_entries` | array | ✓ | 持倉監控（status=active）。 |
| `signals` | array | ✓ | 異常 / 警告訊號（可為空）。 |
| `strategy_events` | array | ✓ | 策略調整事件（可為空）。 |
| `ai_summary` | string \| null | ✓ | AI 摘要文字；未啟用 / 無快取時為 `null`。 |
| `errors` | array of string | ✓ | 子區塊產出失敗訊息（空陣列代表全部成功）。 |

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

## `rotation`（來自 `RotationManager.list_portfolios()` + `get_status()`）

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
  "holdings": [ {RotationHolding}, ... ]
}
```

若有多個 active 組合，輸出 **`current_capital` 最大** 的一個（單一主要組合）。未來如需多組合，可改為 `rotations: [...]`（v2）。

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

## 驗證範例

```bash
# 產生今日報告
python main.py export-dashboard

# 指定日期 + 自訂路徑
python main.py export-dashboard --date 2026-04-30 --out /tmp/dashboard

# JSON 結構檢查
jq '.version, .regime.state, (.discover.momentum | length), (.signals | length)' \
  ~/Library/Mobile\ Documents/com~apple~CloudDocs/QuantDashboard/latest.json
```
