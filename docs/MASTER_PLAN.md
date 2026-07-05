# 專案主計畫（Master Plan）— Single Source of Truth

> 版本：v1.0（2026-07-05）
> 來源：完整 repository 探索 + 深度工程審計 + CTO Roadmap 三輪分析的整併定稿。
> 定位：本文件是策略與工程決策的**單一真相來源**。與 `CLAUDE.md`（開發規則）、`docs/project_history.md`（歷史）互補，不重複。
> 維護規則：任何 P0/P1 事項完成或推翻，須更新本文件對應條目並註記日期。

---

## 1. Executive Summary

**現況**：單人操作、CLI 驅動、SQLite 持久化的台股量化系統（paper trading）。工程紀律（純函數化、1,784+ 測試、audit 文化、自我監控迴路）達 L4，但整體成熟度 **L2.5**，瓶頸不在功能數量（已過剩），在三根斷柱：

1. **帳本不可信**：live 以「夜間決策、當日收盤成交」模擬（該價格拿不到）；rotation 全面忽略股利（除息跳空計為虧損）；Drawdown Kill Switch 的 peak 來自 realized-only 序列（回撤低估）。三者疊加 → 現有 alpha/baseline/歷史裁決全部帶系統性偏差。
2. **研究迴路是斷的**：無 point-in-time 能力。回測只能重播已落庫的 DiscoveryRecord（2026-02-27 起、單一多頭 regime、歷代版本混雜）。改 scanner 後無法歷史驗證，只能 forward test 等數週。
3. **執行層為零**：無 broker 抽象，實盤路徑 L0。

**核心裁決（凍結令）**：在 T+1（A2）與股利會計（A3）完成前，**凍結所有基於現有數字的策略裁決與資金決策**。過去的裁決（dividend「純因子弱」、all10_5d 結構性失敗、停損研究）須在 A2+A3 後重審（研究議程 R1）。

**未來 24 個月的三條主軸**：Truth（帳本可信，Phase A）→ Velocity（研究速度，PIT，Phase A末–B）→ Autonomy（無人化運營 + 實盤，Phase B–C）。驗收標準一句話：*「敢用自己的錢，跑自己看得懂的帳。」*

---

## 2. Current Architecture（現況架構速覽）

### 2.1 分層與資料流

```
外部源(FinMind/TWSE/TPEX/MOPS/TDCC/DJ分點/yfinance)
  → data/pipeline ETL（_upsert_batch 80、OHLCV 閘門、跳動哨兵）
  → SQLite data/stock.db（28 ORM 表、WAL）
  → regime/detector（bull/bear/sideways/crisis；Hysteresis 狀態機，DB 持久化）
  → discovery/（UniverseFilter 三層漏斗 → 5 Scanner 四維評分 → DiscoveryRecord）
  → portfolio/（RotationManager live update + backtest；rotation.py 純函數風控；
                execution_core 金額算式 SSOT）
  → 輸出（Discord Webhook / Streamlit 13 頁 / export-dashboard JSON v4 → iOS QuantMonitor）
```

### 2.2 每日執行（morning-routine 18 步，launchd 排程）

Step 0 宏觀壓力預檢（VIX+crisis）→ 1–8d 資料同步 + DailyFeature → 8c IC 預檢（反向模式自動停用）→ 9 discover all（5 scanner 並行）→ 10–11 alert/watch → 12 rotation update --all → 13–15 掃描與衰減監控 → 16 export-dashboard → 17 baseline 守門 → Discord 摘要。

### 2.3 Live 狀態（2026-07 實查）

- 13 個 rotation 組合，4 active：`mom5_10d`、`swing5_3d`、`mom3_20d`、`mg5_20d`（mom_growth 雙引擎）；9 paused。
- DailyPrice 97.4 萬筆 / 20,393 檔 / 至 2026-07-03；個股全市場歷史僅自 **2025-01-02**。
- DiscoveryRecord 5,685 筆，自 **2026-02-27**（單一多頭 regime）。
- 0050 benchmark 僅自 2025-11-19；**TW_VIX 零筆**（crisis 訊號實際 6/7 可用）。

### 2.4 關鍵設計模式（沿用，勿破壞）

- 純函數 + IO 分離（rotation/audit/metrics/execution_core/entry_exit 零 DB）。
- 常數 SSOT（`src/constants.py`）、Factor SSOT（`src/factors/registry.py`）。
- EAV 指標表 + 寬表 pivot；Watchlist DB 優先 YAML fallback。
- Graceful degradation 貫穿（資料源三級 fallback、dashboard 區塊失敗寫 errors[]）。
- Regime 四狀態驅動：評分權重、門檻、ATR 倍數、Universe 門檻、部位縮放、模式封鎖。

---

## 3. Core Design Principles（核心設計原則）

既有原則（保留）＋ 本輪定案新增（★）：

1. **純函數優先，IO 收斂於 manager/CLI 層**；新計算邏輯必附測試。
2. **SSOT**：常數進 constants.py、因子進 registry、金額算式進 execution_core——不允許第二份實作。
3. ★ **模擬 = 可執行的現實**：任何成交模型必須是真實市場拿得到的（T+1 開盤、整張、含股利、含最低手續費）。「拿不到的價格」視為 bug。
4. ★ **時間注入**：`date.today()` 只准出現在 CLI 入口；所有引擎函數接受 `as_of` 參數（PIT 的前提）。
5. ★ **對齊靠結構不靠自律**：live 與 backtest 的差異只允許存在於「價格來源」與「執行時點」，overlay 組裝必須單一實作（RotationContext 方向）。
6. ★ **自動化決策需要統計門檻**：任何自動開關（IC 停用、權重翻轉）樣本 <100 或未跨 3 個獨立掃描週時，只能告警不能行動。
7. ★ **可重放**：每筆訊號/持倉攜帶 `git_commit` + `settings_hash`；量化參數進版控，secrets 分離。
8. **Graceful degradation 但不靜默**：fallback 可以，必須落下可觀測的痕跡（signal/metrics），連續 fallback 須升級告警。
9. ★ **避免無謂複雜度**（明確不做清單見 §11）。

---

## 4. Coding Standards（編碼規範）

沿用 `CLAUDE.md` 全部規則（繁中註解/commit、UTF-8、_upsert_batch 80、API 速率、ruff check/format），本輪新增：

1. 新增/修改 CLI 子命令必附 smoke test（argparse → 函數呼叫不炸即可）——`Announcement.title` 事故的教訓：glue code 是測試盲區。
2. 交易/帳務邏輯必附「守恆 property test」（任意交易序列後 cash + 持倉成本 + 已實現 pnl 守恆）。
3. 禁止在引擎層呼叫 `date.today()` / 即時 API（offline-safe）；違者 code review 打回。
4. 魔法數新增一律進 constants.py 或 quant 參數檔，inline 數字需附「為什麼是這個值」註解。
5. 依賴以 lockfile 鎖定（pip-compile）；升級依賴為獨立 PR。
6. 涉及 live/backtest 兩路徑的修改，必須同 PR 更新 parity 測試。

---

## 5. System Constraints（系統約束——規劃時的既定事實）

| 約束 | 內容 | 到期/解除條件 |
|------|------|--------------|
| 團隊 | 單人 + AI 協作，每週約 10–15 有效工時 | 常態 |
| 執行環境 | 目前 macOS 筆電 + launchd（Phase B 遷 always-on） | B5 完成 |
| 持久層 | SQLite 單機單寫者；**不遷 PostgreSQL**（除非組合 >10 或引入分 K） | Phase C 再議 |
| 資料深度 | 全市場個股 2025-01 起；DiscoveryRecord 2026-02-27 起；0050 2025-11 起 | B1 歷史回補後解除 |
| 資料頻率 | 日頻。schema 以 date 為鍵，**不支援日內**（勿硬塞） | Future Research |
| 訊號歷史 | 現有 DiscoveryRecord 由混雜版本產生，跨版本回測結果不可比 | A5 之後新記錄可比 |
| 外部介面 | TWSE/TPEX `verify=False` 刻意；MOPS 備援站、DJ Big5 regex——脆弱面，改版風險常在 | 常態，靠對帳+告警圍堵 |
| API 速率 | FinMind 0.5s、TWSE/TPEX 3s——morning-routine 時長的硬下限 | 常態 |
| 交易階段 | Paper only；**實盤須過 §9 六條 Gate** | Gate 通過 |
| 裁決凍結 | A2+A3 完成前，凍結基於現有數字的策略裁決 | A2+A3 完成 |
| 已死資料欄 | TW_VIX（FinMind 移除）、SBL sbl_change 三欄恆 NULL、DJ 分點無均價（close 代理） | 各自替代方案落地 |

---

## 6. High Priority TODO（P0 — 立即到 3 個月）

> 對應 Roadmap Phase A（M1–M3）。順序即建議執行順序。

### 6.1 止血包（A1，合計 ~1 週）

| # | 項目 | 內容 / 驗收 |
|---|------|------------|
| 1 | **DB 自動備份** | `backup_db()` 現為零呼叫者。morning-routine 加尾步 + 異地副本；做一次還原演練。全 repo ROI 最高單項 |
| 2 | **確認 bug：`Announcement.title`** | `pipeline.py:1802` 欄位應為 `subject`，`sync-concepts --from-mops` 100% crash。修復 + CLI smoke test 制度化 |
| 3 | **Kill Switch peak 修復** | `_compute_equity_history` 的 peak 改用 `RotationDailySnapshot.total_capital` 序列（現用 realized-only 累積，浮盈回吐型崩跌不觸發熔斷） |
| 4 | **discover-backtest 預設翻轉** | 預設 `entry_next_open=True, include_costs=True`（現預設 same-close entry = look-ahead），舊行為改 `--naive` |
| 5 | **Dead-man 告警** | healthchecks.io ping + Step 0 印出「crisis 訊號 N/7 可用」自檢（TW_VIX 已死須可見） |
| 6 | **依賴鎖定** | ✅ 2026-07-05 完成：requirements.in（來源）+ freeze 鎖定 requirements.txt（131 套件全 `==`，`--no-deps` 安裝）。pip-compile 無解：FinMind 1.x 釘 ta~=0.5 / 2.x 釘 lxml<5（Py3.14 無 wheel），上游修 pin 後再改回。全新 venv 全測試綠驗收；升級 SOP 見 USAGE.md §1 |
| 7 | **重入保護** | morning-routine 檔案鎖；`rotation update` 同日冪等檢查（查 ActionLog） |

### 6.2 帳本可信三部曲

| # | 項目 | 工期 | 內容 / 驗收 |
|---|------|:---:|------------|
| 8 | **A5 決策可重放** | 3 天 | `DiscoveryRecord` 加 `git_commit`/`settings_hash` 欄位；`settings.yaml` 拆為 `secrets.yaml`（gitignored）+ `quant_params.yaml`（**進版控**）。副作用：strategy_events 的 settings diff 功能復活（現因 settings.yaml 不在 git 永遠回空） |
| 9 | **A2 Live T+1 Pending-Order** | 2 週 | 依 `docs/design/live_t1_pending_order.md` 實作 `RotationPendingOrder` + decide/fill 兩段式。交付 parity 報告：量化「close 成交 vs T+1 open」的 alpha 差距 |
| 10 | **A3 股利會計** | 1.5 週 | rotation live+backtest：持倉除息日現金入帳 + 停損價除息調整；benchmark 0050 還原或標注。**時效**：正值除息季。完成後啟動 R1 歷史裁決重審 |
| 11 | **A4 交易現實化** | 1 週 | 股數整張化（1000 股）+ 零股策略決定、最低手續費 20 元、滑價加入 participation-based impact 項（現模型與下單量無關）。完成後 baseline 重錨 |

---

## 7. Medium Priority TODO（P1 — 3 到 9 個月）

> 對應 Phase A 末–B（M4–M10）。

| # | 項目 | 工期 | 內容 / 價值 |
|---|------|:---:|------------|
| 1 | **B1 Point-in-Time 研究環境** | 4–6 週 | 全 roadmap 最大單一效益。①全市場歷史回補至 2020（含下市股）；②DailyFeature 全歷史化；③scanner 注入 `as_of` + offline mode；④PIT 回測 CLI。解鎖：跨 regime 驗證（R2）、scanner 改動當日見真章 |
| 2 | **B2 全候選池因子落庫** | 1 週 | 掃描時將粗篩後全部候選（非只 top-N）因子值落新表。IC 體系擺脫截斷樣本偏差 |
| 3 | **B11 IC 治理改革** | 1 週 | 樣本門檻 ≥100 且跨 ≥3 掃描週；「自動停用模式」降級為告警+人工確認；以 B2 資料重建 IC。拆除噪音驅動的自動開關（歷史多次誤殺 swing/value/dividend 的根因） |
| 4 | **B5 Always-on 運行環境** | 1 週 | Mac mini 或 VPS 常駐 + 容器化部署腳本；launchd → systemd/cron。消除筆電 SPOF |
| 5 | **B10 告警分級** | 3 天 | critical/warning/info 三管道，critical 需 ack；absence 告警全覆蓋 |
| 6 | **B3 Broker 抽象層 + Shioaji** | 4–6 週 | `Broker` interface（place/cancel/query/positions/fills）+ PaperBroker 重寫 + 永豐 Shioaji sandbox。實盤前提；PaperBroker 順帶統一 fill 模擬 |
| 7 | **B4 對帳引擎** | 2 週 | 現金流水帳（複式 ledger）取代「反覆加減的 float」；每日 internal-consistency 對帳，差異即告警+凍結標的。實盤 Gate 硬條件 |
| 8 | **B6 風險引擎可視化** | 2 週 | Heat/Corr/VaR/因子曝險每日落庫 + 風險頁（現 VaR 只進 log）；VaR 升級為軟限制 |
| 9 | **B7 RotationContext 統一組裝** | 2 週 | live/backtest 的 overlay 組裝（corr/vol/gates/regime/drawdown）單一實作，只差價格源與時點。結構性終結 parity drift（歷史已 3 次 P0） |
| 10 | **B12 停損現實化** | 1.5 週 | 盤中 `low ≤ sl` 觸發、`min(open, sl)` 成交（BacktestEngine 已有邏輯可移植）；停利（rotation 現完全未實作，DiscoveryRecord.take_profit 零引用）先 A/B 回測再決定 |
| 11 | **B9 雙源對帳 + 隔離區** | 2–3 週 | TAIEX/0050/持倉股 TWSE↔FinMind 日對帳；髒值 quarantine 不進決策（0050 事件的制度化解） |
| 12 | **0050 歷史回補 + 第二 benchmark** | 1 天 | alpha 去 size 混雜（現 0050 僅 143 筆） |
| 13 | **C2 AI 研究週報** | 2 週 | 績效歸因/因子健康度/異常清單自動產出（Claude API 已整合）；LLM 只摘要不裁決 |

---

## 8. Low Priority TODO（P2/P3 — 有空檔或條件觸發時做）

| # | 項目 | 觸發條件 / 備註 |
|---|------|----------------|
| 1 | B8 回測引擎合併（walk_forward/Portfolio 委託單一 fill simulator） | B3 PaperBroker 完成後；先做最小版：walk_forward fold 委託 BacktestEngine |
| 2 | C3 Allocator 接入 rotation（多組合資金配置） | 組合數增長或實盤加碼時；先 shadow 模式 |
| 3 | N2 Scanner 宣告式重構（消 5×600 行重複） | 新模式需求出現時做，順帶合併 `_load_revenue_data`/`slice_revenue_raw` 兩份同邏輯 |
| 4 | N1 多 benchmark / 風格歸因擴展 | R2 之後 |
| 5 | N3 DuckDB 研究分析層（唯讀掛 stock.db） | 研究 query 變慢時 |
| 6 | N4 QuantMonitor critical 推播（APNs） | B10 之後 |
| 7 | N5 TWSE 行事曆自動抓取 | 每年 12 月前；2027 表現為暫定推算 |
| 8 | N6 交易日誌富化（進出場快照 markdown） | 隨手做 |
| 9 | 逐股 Python 迴圈向量化（`compute_smart_broker_score` 為首） | Universe 放寬到 >500 候選前必做 |
| 10 | conftest `get_session` monkeypatch 模式改造 | 再次發生 dev DB 污染時升級為 P1 |

---

## 9. Long-term Roadmap（長期路線圖）

### Phase A — Foundation（M1–M6）：刪除謊言
止血包 → T+1 → 股利 → 整張化 → 可重放 → PIT + 全池因子 → always-on。
**驗收**：新主機無人值守 10 交易日零靜默失敗；任一歷史日可 PIT 重建 universe；帳本守恆測試進 CI。

### Phase B — Professional（M7–M12）：買回時間 + 通往實盤
Broker 層 + 對帳 → 風險引擎 → RotationContext → 停損現實化 → 實盤 Gate 審查 → **M12 小額實盤試點**（單組合 ≤20 萬 TWD）。

**實盤 Gate（六條，缺一不可）**：
1. T+1 paper 運行 ≥3 個月且 parity 誤差有解釋；
2. 股利會計 + 整張化上線 ≥2 個月零帳差；
3. 對帳引擎連續 20 交易日零未解差異；
4. 跨 regime PIT 回測（R2）完成且明文接受最壞歷史回撤；
5. Broker 層單日下單金額硬上限保險絲 + kill switch sandbox 演練；
6. 起始資金 ≤ 風險資本 20%、單一組合。

### Phase C — Institutional Grade（M13–M24）：人在迴路之上
實盤校準（滑價實測 vs 模型）→ 資金階梯（每季檢視）→ 風險引擎獲 pre-trade 否決權 → B8/B9/C3 → 事件溯源級重放。**Institutional grade 指紀律，不是拓撲**：依然一個 repo、一台主機、SQLite。

### 研究議程（平台解鎖後，按序）
R1 歷史裁決重審（A2+A3 後）→ R2 跨 regime 穩健性 2020–2026（B1 後）→ R3 全池 IC 重建 + REGIME_WEIGHTS 重估（B2 後）→ R4 成本敏感度曲面 → R5 滑價模型實測校準（實盤 3 個月後）→ R6 多組合配置 → R7 Crisis 引擎歷史回放驗證（2020/03、2022、2024/08）。

---

## 10. Known Technical Debt（已知技術債登記簿）

### 確認的 bug / 死碼（已驗證）
| 項目 | 位置 | 狀態 |
|------|------|------|
| `Announcement.title` 欄位不存在 | `pipeline.py:1802` | **P0 #2 修復中** |
| `backup_db()` 零呼叫者 | `database.py` | **P0 #1 修復中** |
| `_compute_backtest_metrics` 死碼 | `manager.py` 尾端 | 待 sweep |
| `fetch_taiwan_vix` 永遠回空（dataset 已亡） | `fetcher.py` | 待替代（期交所 VIX）或移除 |
| `_collect_settings_diffs` 永遠回空（settings.yaml 不在 git） | `strategy_events.py` | A5 修復後自動復活 |
| SBL `sbl_change` 等三欄恆 NULL（API 改版） | schema + twse_fetcher | 記錄在案，因子已降級 |
| `DailyReportEngine._compute_ml_score` 引用不存在的 `_last_proba` | `report/engine.py` | 永遠走 fallback，低優先 |

### 結構債
- **四套交易模擬器**（BacktestEngine / PortfolioBacktestEngine / walk_forward fold / rotation backtest），成本行為已漂移（walk_forward 無停損無動態滑價）→ B8。
- **live/backtest overlay 組裝兩份**（manager.update vs backtest 各 400 行）→ B7。
- `compute_rotation_actions` 25 參數 400 行；`manager.py` 仍 2,156 行 → 隨 B7 拆。
- 5 scanner 近重複 ~2,800 行；`_load_revenue_data` 與 `slice_revenue_raw` 同邏輯兩份 → N2。
- 魔法數散落（加成 ±3%/±5%、cap 8%、dampen 0.85、news p15 等 inline 預設值）→ 隨 A5 參數檔收斂。
- `settings` import-time 全域單例 → 多環境需求出現時再還。
- 行事曆手工維護（2027 為暫定推算，每年 12 月須校對）→ N5。
- 逐股 Python 迴圈效能地雷（150 候選安全，1,500 不安全）→ Low #9。
- conftest session-scope engine + get_session monkeypatch 的污染陷阱（曾污染 dev DB）→ Low #10。
- Fallback 疊加路徑（資料延遲 + 模式停用 + 前日排名 + 5 日前價格）無測試覆蓋 → 隨 B7 的 parity 套件補。

### 統計/方法債
- IC 建立在 top-N 截斷樣本、n≈20–30，驅動自動停用/翻轉 → B2+B11。
- 全部閾值/權重為 in-sample 調參，跨 regime 未驗證 → R2/R3。
- 停損只看收盤、無停利、live 成交=決策價 → A2+B12。
- Crisis 邏輯從未被歷史崩跌段檢驗 → R7。

---

## 11. Future Architecture Vision（目標架構）

### 24 個月目標形態

```
┌────────────┐   ┌─────────────────┐   ┌──────────────────┐
│ Data Layer │──▶│ Signal Engine    │──▶│ Portfolio Engine  │
│ PIT + 對帳 │   │ scanner(as_of)   │   │ RotationContext   │
│ quarantine │   │ offline-safe     │   │ （單一組裝實作）  │
└────────────┘   └─────────────────┘   └────────┬─────────┘
                                          PendingOrder(DB)
      ┌──────────────┐                 ┌────────▼─────────┐
      │ Risk Engine  │◀── 每日落庫 ────│ Execution Layer   │
      │ pre-trade 否決│                │ Broker interface  │
      │ (Phase C)    │                 │ Paper ⇄ Shioaji   │
      └──────────────┘                 └────────┬─────────┘
                                       ┌────────▼─────────┐
                                       │ Reconciliation    │
                                       │ 複式 ledger 對帳  │
                                       └──────────────────┘
```

- **Modular monolith 到底**：分離的是模組邊界與合約，不是部署單元。單 process、單 repo、SQLite（+DuckDB 分析層）。
- 每筆決策可完整重放：任一歷史日重建「系統當天看到什麼、為何這樣做」。
- Paper 與 live 並行：同一 Signal Engine，兩個 Portfolio+Broker 實例。

### 明確不做（Anti-goals，防 scope creep）

| 不做 | 理由 | 重新評估條件 |
|------|------|-------------|
| 微服務 / K8s / 訊息佇列 | 單人日頻，monolith 到 L4 都夠 | 永不（此規模下） |
| SQLite → PostgreSQL | 單機單寫者無瓶頸 | 組合 >10 或引入分 K |
| Tick/分 K 與日內策略 | 日頻 alpha 未驗證完，不開第二戰場 | 實盤滑價實測顯著劣於模型時，僅限執行優化 |
| 自建 ML 平台 / GPU | sklearn 路線夠用 | 無 |
| 多資產（期貨/選擇權/美股） | 聚焦 | Phase C 後；台指期對沖列 Future Research |
| Prometheus/Grafana 級監控 | metrics 落 SQLite + Streamlit 即可 | 多機部署時 |

---

*本文件由三輪分析（2026-07-04 探索/審計/Roadmap）整併定稿。衝突解決原則：審計發現的嚴重度 + Roadmap 的執行順序 = 本文件的優先級；重複建議已合併至單一條目。*
