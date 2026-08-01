# 專案主計畫（Master Plan）— Single Source of Truth

> 版本：v1.2（2026-08-01；v1.1 為 2026-07-31、v1.0 為 2026-07-05）
> 來源：完整 repository 探索 + 深度工程審計 + CTO Roadmap 三輪分析的整併定稿；v1.1 併入 discover 選股邏輯全鏈路審計（`logs/audit_discover_20260731/REPORT.md`）。
> 定位：本文件是策略與工程決策的**單一真相來源**。與 `CLAUDE.md`（開發規則）、`docs/project_history.md`（歷史）互補，不重複。
> 維護規則：任何 P0/P1 事項完成或推翻，須更新本文件對應條目並註記日期。

---

## 1. Executive Summary

**現況**：單人操作、CLI 驅動、SQLite 持久化的台股量化系統（paper trading）。工程紀律（純函數化、1,784+ 測試、audit 文化、自我監控迴路）達 L4，但整體成熟度 **L2.5**，瓶頸不在功能數量（已過剩），在四根斷柱：

1. **帳本不可信**：live 以「夜間決策、當日收盤成交」模擬（該價格拿不到）；rotation 全面忽略股利（除息跳空計為虧損）；Drawdown Kill Switch 的 peak 來自 realized-only 序列（回撤低估）。三者疊加 → 現有 alpha/baseline/歷史裁決全部帶系統性偏差。→ **Phase A 三部曲已完成（A2/A3/A4/A5）**
2. **研究迴路是斷的**：無 point-in-time 能力。回測只能重播已落庫的 DiscoveryRecord（2026-02-27 起、單一多頭 regime、歷代版本混雜）。改 scanner 後無法歷史驗證，只能 forward test 等數週。
3. **執行層為零**：無 broker 抽象，實盤路徑 L0。
4. ★ **訊號引擎曾全面停產**（2026-07-31 discover 審計，`logs/audit_discover_20260731/REPORT.md`）：7/17–7/31 共 11 個交易日中 momentum 掃出 1 天、swing 1 天、growth 0 天；僅存的兩個 active 組合皆 momentum 模式且**持倉 0 筆、100% 現金**。**停產原因與因子好壞無關**，是治理機制自身的三重缺陷：①~~M2 以 34 天前 n=40 的凍結 IC 執法~~ ✅2026-08-01 已修（§6.4 #16，五模式恢復掃描）；②~~regime 同日被偵測 4–15 次~~ ✅2026-08-01 已修（§6.4 #15）；③**五個 scanner 非同一條 pipeline，三個模式繞過分數門檻使跨模式全不可比——仍未修**（N2，§7 #3b）。另 §6.4 #17 已於 2026-08-01 凍結 E2b/E2c（噪音驅動的自動調權與翻轉不再生效）。**現況**：訊號產能已恢復、噪音開關已止血，但**模式級比較仍不可信**；下一關鍵為 N2（統一 `run()`）與 B2（全候選池落庫）

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

- 13 個 rotation 組合，3 active：`mom5_10d`、`mom3_20d`、`mg5_20d`（mom_growth 雙引擎，**警戒**）；10 paused（swing5_3d 2026-07-19 終裁暫停）。
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
| 裁決凍結 | ✅已解除（2026-07-07 A2+A3 完成）；**R1 歷史裁決重審 ✅2026-07-08 完成**（`logs/r1_20260708/REPORT.md`）：6/20 全部裁決無翻案；副產物=修復 FinMind year=NaN 股利斷流 bug；**7/19 終裁**（`logs/audit_mg5_20d_20260718/REPORT.md`）：swing5_3d 🔴暫停（live B期 1/13 勝、alpha −5.13pp）；mg5_20d 首審=維持但降級警戒（MtM alpha −7.6%，惟窗口被事故+訊號斷糧+7/17 崩盤三重污染，N=2），二審 gate 在 P0 訊號斷糧修復；**7/30 二審＝No-Verdict 不可評估**（`logs/audit_mg5_20d_20260730/REPORT.md`）：gate 形式條件已成立但修復後 8 交易日新進場 **0 筆**，forward OOS 樣本為空；原訂 8/8 判點取消、改掛 B11 條件（詳 §7 #3）；mg5_20d 已 pause。**連帶盤點**：僅存 active 組合 mom5_10d／mom3_20d 皆 momentum 模式、7/31 起 100% 現金，且其歷史績效受 renewal 污染（§10 結構債）——兩者正式重審待污染量化後另辦。**⚠ 2026-07-31 discover 審計新增凍結範圍**（`logs/audit_discover_20260731/REPORT.md`）：value/growth/dividend 覆寫 `run()` 跳過分數門檻/產業分散/回撤縮表，五個 scanner 非同一 pipeline → **任何模式級比較（哪個模式較好、跨模式 IC、cross-mode-corr、mode 選擇）在 N2 統一 `run()` + B2 落地前一律不成案**，含現行「暫停 swing、保留 value/dividend」之既成判斷 | 部分重新凍結（模式級比較） |
| 已死資料欄 | TW_VIX（FinMind 移除）、SBL sbl_change 三欄恆 NULL、DJ 分點無均價（close 代理） | 各自替代方案落地 |

---

## 6. High Priority TODO（P0 — 立即到 3 個月）

> 對應 Roadmap Phase A（M1–M3）。順序即建議執行順序。

### 6.1 止血包（A1，合計 ~1 週）

| # | 項目 | 內容 / 驗收 |
|---|------|------------|
| 1 | **DB 自動備份** | ✅ 2026-07-05 完成：morning-routine Step 18 接上 `backup_db()`；異地副本至 iCloud Drive（`backup.offsite_dir` 可覆蓋，失敗僅 warning）；保留天數 config 化。還原演練 2026-07-05 完成（integrity ok + 功能驗證），SOP 見 usage.md §8 |
| 2 | **確認 bug：`Announcement.title`** | ✅ 2026-07-05 完成：`title`→`subject` 修復；`main.py` 抽出 `build_parser()`；新增 `tests/test_cli_smoke.py`（全 49 子命令 parse + handler 鏈路回歸測試） |
| 3 | **Kill Switch peak 修復** | ✅ 2026-07-05 完成：新增 `compute_drawdown_with_snapshots` 純函數（peak = max(equity_history ∪ snapshot 序列)，容忍缺日）；熔斷與 Drawdown Guard 共用同一 dd_pct。backtest 路徑經查無此 bug（equity_curve 逐日含 MtM） |
| 4 | **discover-backtest 預設翻轉** | ✅ 2026-07-05 完成（僅 CLI 層）：未明示 → (True, True)；`--naive` 取得舊行為（與明示 flag 並用即報錯）；`--no-*` 可單獨關閉；報告 header 印「成本模型/進場假設」。引擎預設不動（dashboard/decay 依賴 False） |
| 5 | **Dead-man 告警** | ✅ 2026-07-05 完成：`monitoring.healthchecks_url` config + start/success/fail ping（例外全吞不影響主流程；非交易日視為 success；dry-run 不 ping）；`detect_crisis_signals` 新增 `availability`，Step 0 與 Discord 摘要每日印「crisis 訊號可用 N/7」。**待辦**：使用者自行到 healthchecks.io 建 check 填 URL |
| 6 | **依賴鎖定** | ✅ 2026-07-05 完成：requirements.in（來源）+ freeze 鎖定 requirements.txt（131 套件全 `==`，`--no-deps` 安裝）。pip-compile 無解：FinMind 1.x 釘 ta~=0.5 / 2.x 釘 lxml<5（Py3.14 無 wheel），上游修 pin 後再改回。全新 venv 全測試綠驗收；升級 SOP 見 USAGE.md §1 |
| 7 | **重入保護** | ✅ 2026-07-05 完成：morning-routine flock 檔案鎖（`data/.morning_routine.lock`，process 死亡自動釋放）；`update()` 入口同日冪等 guard（ActionLog ∪ Snapshot 雙重判斷，`--force` 覆蓋，dry_run 不受限） |

### 6.2 帳本可信三部曲

| # | 項目 | 工期 | 內容 / 驗收 |
|---|------|:---:|------------|
| 8 | **A5 決策可重放** ✅2026-07-06 | 3 天 | `DiscoveryRecord` 加 `git_commit`/`settings_hash` 欄位；`settings.yaml` 拆為 `secrets.yaml`（gitignored）+ `quant_params.yaml`（**進版控**）。副作用：strategy_events 的 settings diff 功能復活（現因 settings.yaml 不在 git 永遠回空） |
| 9 | **A2 Live T+1 Pending-Order** ✅2026-07-06 | 2 週 | 依 `docs/design/live_t1_pending_order.md` 實作 `RotationPendingOrder` + decide/fill 兩段式。交付 parity 報告：量化「close 成交 vs T+1 open」的 alpha 差距 |
| 10 | **A3 股利會計** ✅2026-07-07 | 1.5 週 | rotation live+backtest：持倉除息日現金入帳 + 停損價除息調整；benchmark 0050 還原或標注。**時效**：正值除息季。完成後啟動 R1 歷史裁決重審 |
| 11 | **A4 交易現實化** ✅2026-07-08 | 1 週 | 零股策略裁決=**混合單**（sizing 不整張化；成本模型拆整張單+盤中零股單，各計最低手續費 20/1 元、零股 notional 加 0.1% 滑價 premium）；滑價加 participation impact（c×√(下單量/當日量)，c=0.01）。成本 SSOT=`rotation.trade_cost_amounts`，rotation live+backtest 恆開、BacktestEngine 旗標化（引擎預設關、`backtest` CLI 預設開）。baseline 重錨 ✅2026-07-08（4 portfolio 重新凍結，validate-baseline 全綠） |

### 6.3 二次止血包（2026-07-19 事故重審新增；詳 `logs/audit_mg5_20d_20260718/REPORT.md` §3）

| # | 項目 | 內容 / 驗收 |
|---|------|------------|
| 12 | **pending 重複買單 UNIQUE 炸鍋** | ✅ 2026-07-19 已修（分支 `fix-pending-order-duplicate-crash` 待 PR）：7/10 假交易日→3617 順延→decide 重複開單→7/13 雙成交 UNIQUE 違反→Step 12 連炸 5 天（7/13–17 swing5_3d/mg5_20d/mom3_20d 未更新、停損凍結、snapshot 永久缺口）。三層防護：decide 在途去重 / fill TTL 提前+同股防護 / update --all per-portfolio 隔離。+5 回歸測試 |
| 13 | **rankings stale fallback 架空 M2 停用** | ✅ 2026-07-19 已修（分支 `fix-rankings-stale-fallback`）：fallback 設時效上限 `RANKINGS_FALLBACK_MAX_TRADING_DAYS=3`——逾期回空排名 → 組合凍結新買入、僅走風控賣出/到期路徑；composite 分支只在含 member mode 記錄的掃描日中找。live 驗證：mom5_10d preview 正確凍結（momentum stale 21 交易日）。+5 測試。後續與 B11（IC 治理：自動停用降級為告警）連動——**2026-07-30 二審實證本修復行為正確**（mg5_20d 於 growth 掃描逾期後正確凍結新買入），但也暴露 B11 未修時組合會被無限期凍結（詳 §7 #3） |
| 14 | **行事曆假交易日哨兵** | ✅ 2026-07-19 已修（分支 `fix-calendar-phantom-trading-day`）：①`calendar.py` 新增 `_UNSCHEDULED_CLOSURES`（含 2026-07-10）+ `is_unscheduled_closure()`，`is_trading_day` 排除；②`_verify_data_freshness` 加臨時休市哨兵（`PHANTOM_TRADING_DAY_MIN_ROWS=100`）：行事曆交易日但同步後全市場當日 <100 筆 → 凍結 Step 9 discover + Step 12 rotation update（T+1 佇列自然順延），Discord banner 告警；--skip-sync 跳過哨兵（未同步必為 0 筆會誤判，e2e 測試路徑）。+9 測試 |

### 6.4 三次止血包（2026-07-31 discover 全鏈路審計新增；詳 `logs/audit_discover_20260731/REPORT.md`）

> 背景：7/17–7/31 共 11 個交易日，momentum 掃出 1 天、swing 1 天、growth 0 天；僅存 active 組合 mom5_10d／mom3_20d 持倉 0 筆。審計結論＝**訊號停產的原因與因子好壞無關，是治理機制自身的統計缺陷與結構不一致**。

| # | 項目 | 內容 / 驗收 |
|---|------|------------|
| 15 | **regime 同日多次偵測（P0，新）** | ✅ **2026-08-01 已修**（分支 `fix-regime-same-day-redetect`）。**病因**：`MarketRegimeDetector` 的同日快取是 `self._cached_result`（**per-instance**，`detector.py:409`），而每個 scanner 的 Stage 0 各建新實例 → 快取**從未命中** → 每次呼叫重跑 `apply_hysteresis()` 並 append 一筆 `RegimeStateLog`。實測每日 **4–15 次**呼叫；2026-07-24 一次 morning-routine 內出現 `sideways→sideways→bear→sideways→bear→bear`，`universe_stat_log` 顯示同批掃描 value=sideways／dividend=bear／swing=bear（時間戳相隔 <1 秒）。**三重後果**：①同日各模式在不同 regime 下評分（門檻 0.50 vs 0.55、權重、universe 乘數、`REGIME_MODE_BLOCK` 全部分歧）；②模式當天跑不跑取決於排在第幾個呼叫；③**hysteresis 確認計數按「呼叫次數」而非「天數」消耗**，遲滯機制（防 regime 抖動的唯一保護）被架空。<br>**修法（採無狀態冪等鍵，非「注入單一實例」）**：新增 `RegimeState.data_date` / `regime_state_log.data_date` = 本次判定依據的**最新 TAIEX 資料日**，作為狀態機冪等鍵；同一 `data_date` 無論被呼叫幾次、由幾個實例／行程呼叫，都回傳同一結論且不推進 hysteresis、不寫 log。**不採用日曆日為鍵**——morning-routine Step 0 在資料同步（Step 1~8）**之前**執行，以日曆日為鍵會把 regime 永久凍結在前一交易日資料上。回傳 dict 新增 `state_advanced` 供觀測（§3 原則 8）。<br>**附帶修復**：Step 0 的 regime 為同步前判定，Step 12 輪動卻在同步後執行 → 新增 **Step 8e**「同步後 regime 重解」（`resolve_regime_after_sync()`，dry_run/skip_sync 自動跳過），消除 discover 與 rotation 差一個交易日的落差。<br>**驗收**：`python main.py migrate` 已執行（新增 1 欄）；live 實測 7 次全新實例呼叫 → 新增 **1** 筆 log、regime 全部一致、`state_advanced` 僅第 1 次為 True。新增 `tests/test_regime_idempotence.py` **21 測試**（含核心回歸：同日 3 次呼叫不得湊滿 sideways→bull 的 3 天確認）；停用冪等閘門可複現 6 個失敗。全測試 2,630 passed |
| 16 | **M2 凍結 IC 執法（P0，B11 前置最小切口）** | ✅ **2026-08-01 已修**（分支 `fix-ic-governance-min-cut`）。病因詳見 §7 #3 與 §10「統計/方法債」。**修法**：新增 `src/discovery/ic_governance.py` 作為「IC 可不可以拿來行動」的**單一實作**，三個執法點（M2 停用 / scanner 門檻 +0.05 加成 / rotation 阻擋買入）共用同一套三道閘門：<br>　①**時效**：最新 `window_end` 距 `as_of` 不得超過 `holding_days + 14`（與 holding 掛鉤——rolling 窗口需 forward 報酬，天生落後約 holding_days；momentum 19 天、value/dividend/growth 34 天）<br>　②**窗口數** ≥3（step_days=7，≈§3 原則 6 的「跨 3 個掃描週」）<br>　③**樣本**：決策窗口中**最小** `evaluable_count` ≥100（§3 原則 6）<br>決策 IC 改取**最近 3 窗平均**而非單一最新窗（實測 swing 最新窗 −0.0933/n=140 但前三窗 +0.005，3 窗平均 −0.0479 未達門檻＝單窗判定屬小樣本漂移）。<br>**②停用語意改變**：`_step_9_discover` 恆傳 `disabled_modes=[]`——**五模式恆掃恆落庫**，IC 反向改由 `manager._build_decision_context` 經 `resolve_rankings(exclude_modes=...)` 在 rotation 層阻擋新買入（賣出/停損/到期/風控路徑不受影響）。composite 模式只濾掉被擋成員，全數被擋才回空。rotation 端**即時重算**而非讀取 Step 8c 結果（`rotation update` 可單獨執行，且天然無「治理資料本身過期」問題）；backtest 路徑**刻意不套用**（今日裁定套到歷史掃描日＝look-ahead，留待 B1 PIT）。<br>**驗收（live 實測 2026-08-01）**：momentum「最新窗口已過期（2026-06-30，距今 32 天 > 上限 19）」→ 僅告警不執法；value/growth「樣本不足（n=60 < 100）」；dividend「窗口數不足（1 < 3）」；swing 可執法但 3 窗平均 −0.0479 未達 −0.05 → 放行。**五個模式全數恢復掃描，自鎖迴路解除**。新增 `tests/test_ic_governance.py` 24 測試（停用閘門可複現 8 個失敗）；全測試 2,654 passed |
| 17 | **凍結 IC 驅動的自動調權與翻轉（P1，暫時性）** | ✅ **2026-08-01 已修**（分支 `freeze-ic-auto-adjust`）。E2b `_apply_ic_weight_adjustment` 與 E2c `compute_ic_aware_score_transform` 改為**只記錄不生效**。理由見 §7 #3(d)（權重被搬給未量測維度 2.25× + ±0.02 翻轉門檻深埋雜訊帶 + news_score 七成是填補值）。<br>**開關**：`config/quant_params.yaml` → `quant.ic_governance.auto_weight_adjust` / `auto_score_transform`，**兩者預設 false＝凍結**（`src/config.py:ICGovernanceConfig`，解除條件詳列於該處 docstring）。<br>**設計**：凍結只作用在**唯一套用點** `_score_candidates`——底層純函數與 `_apply_ic_weight_adjustment` 行為不變（既有 21 個相關測試零修改即通過），且**仍照常計算**（`_dimension_ic_df` 照填），只是不套用。log 標【凍結中，未生效】並印出 would-be 動作；`discover all` 開頭印一行凍結橫幅（§3 原則 8：降級但不靜默）。`_ic_actions` 凍結時刻意留空——CLI 的 (N)/(F)/(D) 代表「已套用」，標上去會誤導。<br>**解除條件**：B2 落地（IC 擺脫 top-N 截斷樣本）＋ 未量測維度（`valuation`/`dividend`）納入落庫 ＋ 調整幅度改以標準誤為基準的顯著性判定 → 屬 B11 完整版。<br>**驗收**：新增 `tests/test_ic_freeze.py` 12 測試（含「開啟後行為回復」對照組，證明凍結是唯一抑制來源）；停用凍結可複現 5 個失敗。全測試 2,666 passed |
| 18 | **`ScoreThresholdConfig` 接線 + `insufficient` 分支拆分（P2）** | ①`src/config.py:84-90` 定義了 bull/sideways/bear/crisis 門檻但**零消費者**，scanner 讀死的 `_functions.py:1196 MIN_SCORE_THRESHOLDS` → 使用者改 `quant_params.yaml` **靜默無效**（違反 §3 原則 2 SSOT）—— **尚未修**；②`insufficient` 分支語意混淆 ✅**2026-08-01 隨 #16 一併修**：`level` 現分為 `insufficient`（推薦記錄 <20，訊息改為「推薦記錄不足」）/ `no_prices` / `no_windows` / `stale_window` / `insufficient_windows` / `insufficient_samples`，各自說出真正原因，不再印「樣本不足（n=260，需 ≥20）」這種自相矛盾訊息。dividend 之所以「從未受 M2 管轄」的真因（holding=20 使窗口數常為 1）現由 `窗口數不足（1 < 3）` 明確顯示 |

---

## 7. Medium Priority TODO（P1 — 3 到 9 個月）

> 對應 Phase A 末–B（M4–M10）。

| # | 項目 | 工期 | 內容 / 價值 |
|---|------|:---:|------------|
| 1 | **B1 Point-in-Time 研究環境** | 4–6 週 | 全 roadmap 最大單一效益。①全市場歷史回補至 2020（含下市股）；②DailyFeature 全歷史化；③scanner 注入 `as_of` + offline mode；④PIT 回測 CLI。解鎖：跨 regime 驗證（R2）、scanner 改動當日見真章 |
| 2 | **B2 全候選池因子落庫** 🔗**與 B11 綁為同一批交付**（2026-07-31） | 1 週 | 掃描時將粗篩後全部候選（非只 top-N）因子值落新表。IC 體系擺脫截斷樣本偏差。**綁定理由**：實測漏斗 `1,857 →(流動性) 659 →(粗篩) 150 →(落庫) 20`，**所有 IC 都只算在最終 top-20 上**（range restriction），是 B11／E2b／E2c 三處噪音開關的**共同上游**；B11 單獨落地仍是在截斷樣本上調參。附帶需求：`valuation`／`dividend` 維度目前不落庫 → 恆 `IC=N/A` → 見 B11 更新之「未量測維度吸收歸一化殘差」 |
| 3 | **B11 IC 治理改革** 🔺**2026-07-30 升為 P1 最優先**；🔺**2026-07-31 擴大範圍** | 1 週 | 樣本門檻 ≥100 且跨 ≥3 掃描週；「自動停用模式」降級為告警+人工確認；以 B2 資料重建 IC。拆除噪音驅動的自動開關（歷史多次誤殺 swing/value/dividend 的根因）。**升順位理由（mg5_20d 二審）**：現行 M2 停用的是**掃描**而非**下單**（`morning_cmd.py:1201` → Step 9 skip），停用後不再產生 `discovery_record` → IC 無從重算 → **模式無法自證恢復，形成自鎖**。實測：momentum 6/16 停用後中斷 29 個交易日（至 7/29 才有 7 筆部分掃描）、growth 7/16 起停用至今。後果＝雙引擎組合 mg5_20d 訊號歸零、forward 驗證無樣本可取。**修法方向：停用時照常掃描並落庫，僅在 rotation 層阻擋新買入**，使 IC 可續算、模式具自動恢復路徑。此項未落地前，mom_growth／momentum 系組合的任何 forward 裁決都不成案。**影響範圍＝全系統**（2026-07-30 盤點，報告 §6）：mg5_20d pause 後僅存的 active 組合 mom5_10d／mom3_20d 皆為 momentum 模式，7/31 起雙雙 100% 現金、momentum 7/29 排名 8/4 逾期；momentum 過去 30 個交易日僅掃描 1 天（7/29 放行、7/30 又停用），M2 呈「停用→樣本歸零→判 insufficient→放行一天→再停用」振盪。後果＝**crisis 解除後全系統無可用訊號重新進場**。<br>**🔺2026-07-31 discover 審計修正與擴大**（`logs/audit_discover_20260731/REPORT.md`）：<br>**(a) 自鎖機制比原描述嚴重**——不只是「IC 無從重算」。`compute_rolling_ic`（`_functions.py:2067`）的窗口錨定在**推薦記錄的日期範圍**而非今天，`morning_cmd.py:724` 又取 `factor_df["ic"].iloc[-1]`；模式停掃後 `max_date` 凍結 → 窗口凍結 → **系統每天重讀同一份過期 IC 執法**。實測：7/31 停用 momentum 所用的 IC＝**−0.1109，來自 `window_end=2026-06-27`、n=40、距今 34 天**，且 7/30 與 7/31 取到完全相同的值（＝同一凍結值重讀兩次，非兩次獨立判斷）。重現腳本 `logs/audit_discover_20260731/repro_rolling_ic.py`。<br>**(b) 「振盪」真因不是 IC 回升**——7/29 放行是因當下記錄僅到 6/16，`min_date+14=6/19 > max_date+1=6/17` → while 迴圈零次 → `rolling_df` 空 → fail-open。**故此開關的實際判準是「rolling IC 算不算得出來」，與因子有效性無關。**<br>**(c) 與 §3 原則 6 的落差已量化**：原則要求 n≥100 且跨 ≥3 掃描週，實際跑在 **n=40、單一窗口、34 天前**——原則已寫入文件但程式無一處執行。<br>**(d) 範圍擴大至 E2b/E2c 兩個同源開關**：①`_apply_ic_weight_adjustment` 的歸一化把被打壓維度釋出的權重**全數塞給從未量測的維度**（2026-07-31 value：`fundamental 0.550→0.313` IC −0.1393，而 `valuation 0.150→0.338` **IC=N/A**，2.25×）——淨效果＝IC 治理越積極，決策權越集中到唯一沒被檢驗的維度；且依據本身不穩（value fundamental IC 四個交易日 `+0.155→+0.105→+0.034→−0.139`，擺幅全在 n≈100 的 SE≈0.10 雜訊帶內）。②`compute_ic_aware_score_transform(ic_threshold_weak=0.02, min_samples=50)` 在 SE≈0.10–0.14 下等同**隨機翻轉維度方向**。③`news_score` 有 **63–81% 恰為填補值 0.5**（`fillna(0.5)`）卻佔權重 0.20 且每日參與 IC 與調權——對七成是常數的變數算秩相關無意義。**修法須含**：覆蓋率不足的維度不得參與自動調權；未落庫維度不得被動吸收權重；翻轉門檻改為以 SE 為基準的顯著性判定。**2026-08-01 進度**：E2b/E2c 已整體凍結（§6.4 #17，只記錄不生效），屬止血非根治——上述三項仍是**解凍的前提條件** |
| 3b | **N2 Scanner 統一 `run()`** 🔺**2026-07-31 由 P2 升 P1** | 1 週 | 原列 §8 #3「新模式需求出現時做」，**定性低估**：這不是重複程式碼的整潔問題，是**量測有效性問題**。`_value.py:168`／`_growth.py:57`／`_dividend.py:173` 各自覆寫 `run()`（複製貼上版），相對 `BaseScanner.run()` 跳過：**3.7 動態分數門檻／4.1 產業分散化／4.2 回撤縮表**／3.2 前次重疊／3.3c 同業基本面／3.5c 動量衰減／3.5d-g 籌碼系／3.5h 負面消息閘門／3.5e 多時框／3.6 量價背離／4.3 籌碼降級稽核／`ScanAuditTrail`／`sub_factor_df`／`ic_actions`。**直接證據**：dividend 2026-07-31 落庫最低分 **0.536**，當日 regime=crisis 門檻應為 0.60。**後果**：「value/dividend 天天穩定 20 筆、momentum/swing 常態 0 筆」是**閘門覆蓋率差異的產物，不是模式強弱**——此假象正在污染跨模式 IC 比較、`cross-mode-corr`、`mom_growth` 雙引擎的模式選擇、以及「暫停 swing、保留 value/dividend」之裁決。**驗收**：五 scanner 走同一條漏斗，模式差異只留在 `_coarse_filter`／`_compute_*_scores`／`_compute_extra_scores`；順帶合併 `_load_revenue_data`／`slice_revenue_raw` 兩份同邏輯。詳 `logs/audit_discover_20260731/REPORT.md` §5 |
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
| 3 | ~~N2 Scanner 宣告式重構（消 5×600 行重複）~~ | 🔺**2026-07-31 升 P1，移至 §7 #3b**（跨模式不可比，非整潔問題） |
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
R1 歷史裁決重審 ✅2026-07-08（無翻案；報告 `logs/r1_20260708/REPORT.md`）→ R1b 7/19 終裁 ✅（swing5_3d 暫停；mg5_20d 首審降級警戒，二審 gate=P0 #13 修復＋事故污染排除，非固定日期）→ **R1c mg5_20d 二審 7/30＝No-Verdict**（`logs/audit_mg5_20d_20260730/REPORT.md`；gate 已過但新進場 0 筆、樣本空集合；生涯 closed N=7 勝率 0%／−180,919／capital −18.06%，建議 pause；**二審重掛條件＝B11 落地且 mom_growth 恢復連續掃描滿 10 個交易日**，不設固定日期）→ R2 跨 regime 穩健性 2020–2026（B1 後）→ R3 全池 IC 重建 + REGIME_WEIGHTS 重估（B2 後）→ R4 成本敏感度曲面 → R5 滑價模型實測校準（實盤 3 個月後）→ R6 多組合配置 → R7 Crisis 引擎歷史回放驗證（2020/03、2022、2024/08）。

**⚠ 2026-07-31 研究議程前置條件更新**：discover 審計（`logs/audit_discover_20260731/REPORT.md` §5）證實五個 scanner 非同一條 pipeline，**任何模式級比較在 N2（§7 #3b）+ B2 落地前皆不成案**。受影響：R3（全池 IC 重建，本就依賴 B2）、以及所有既成的模式級裁決——含「暫停 swing、保留 value/dividend」。另 `data/baseline_metrics.json` 的模式級門檻在 N2 修復後需重新評估（與 §10 renewal 訊號污染的重錨需求**合併處理**，勿分兩次重錨）。

---

## 10. Known Technical Debt（已知技術債登記簿）

### 確認的 bug / 死碼（已驗證）
| 項目 | 位置 | 狀態 |
|------|------|------|
| `Announcement.title` 欄位不存在 | `pipeline.py:1802` | ✅ 2026-07-05 已修復（`subject` + smoke test） |
| `backup_db()` 零呼叫者 | `database.py` | ✅ 2026-07-05 已接上 morning-routine Step 18 |
| `_compute_backtest_metrics` 死碼 | `manager.py` 尾端 | 待 sweep |
| `fetch_taiwan_vix` 永遠回空（dataset 已亡） | `fetcher.py` | 待替代（期交所 VIX）或移除 |
| `_collect_settings_diffs` 永遠回空（settings.yaml 不在 git） | `strategy_events.py` | ✅2026-07-06 A5 拆檔後已復活（改追 quant_params.yaml） |
| SBL `sbl_change` 等三欄恆 NULL（API 改版） | schema + twse_fetcher | 記錄在案，因子已降級 |
| `DailyReportEngine._compute_ml_score` 引用不存在的 `_last_proba` | `report/engine.py` | 永遠走 fallback，低優先 |
| `ScoreThresholdConfig` 零消費者（scanner 讀死常數，`quant_params.yaml` 門檻靜默無效） | `config.py:84-90` vs `_functions.py:1196` | 2026-07-31 發現 → §6.4 #18 |
| `MarketRegimeDetector` 同日快取 per-instance，從未命中（每日 4–15 次重跑 hysteresis） | `detector.py:409` | ✅ 2026-08-01 已修（§6.4 #15；改以 `data_date` 為冪等鍵 + Step 8e 同步後重解） |
| `level="insufficient"` 混用四種語意，log 印「樣本不足（n=260，需 ≥20）」 | `morning_cmd.py:669/693/714/720` | 2026-07-31 發現 → §6.4 #18 |

### 結構債
- **四套交易模擬器**（BacktestEngine / PortfolioBacktestEngine / walk_forward fold / rotation backtest），成本行為已漂移（walk_forward 無停損無動態滑價）→ B8。
- **live/backtest overlay 組裝兩份**（manager.update vs backtest 各 400 行）→ B7。
- `compute_rotation_actions` 25 參數 400 行；`manager.py` 仍 2,156 行 → 隨 B7 拆。
- **5 scanner 非同一條 pipeline**（2026-07-31 重新定性）：`value`/`growth`/`dividend` 覆寫 `run()` 並跳過分數門檻/產業分散/回撤縮表/負面消息閘門/audit_trail/sub_factor 落庫，momentum/swing 走 base `run()` 吃滿 6 道硬風控。**這不是 ~2,800 行重複的整潔問題，是跨模式不可比**——所有模式級比較（IC、cross-mode-corr、mode 選擇、模式暫停裁決）皆建立在此不對等基礎上 → **N2 升 P1**（§7 #3b）。`_load_revenue_data` 與 `slice_revenue_raw` 同邏輯兩份一併處理。
- 魔法數散落（加成 ±3%/±5%、cap 8%、dampen 0.85、news p15 等 inline 預設值）→ 隨 A5 參數檔收斂。
- `settings` import-time 全域單例 → 多環境需求出現時再還。
- 行事曆手工維護（2027 為暫定推算，每年 12 月須校對）→ N5。
- 逐股 Python 迴圈效能地雷（150 候選安全，1,500 不安全）→ Low #9。
- conftest session-scope engine + get_session monkeypatch 的污染陷阱（曾污染 dev DB）→ Low #10。
- Fallback 疊加路徑（資料延遲 + 模式停用 + 前日排名 + 5 日前價格）無測試覆蓋 → 隨 B7 的 parity 套件補。
- **renewal 訊號污染（2026-07-30 發現）**：續持依賴當前排名（`rotation.py:841` `if allow_renewal and sid in ranked_ids`），P0 #13 修復前的 stale fallback 使部位靠過期排名**無限續持**——mom5_10d 1303（hold=10，實持 6/16–7/21、+62.78%）、mom3_20d 2890/5871（hold=20，實持 6/09–7/21）等多筆遠超名目天數，並於 7/21 修復後首次 update 集體到期。含意：①mom5_10d／mom3_20d 的 +12%／+11% 生涯績效部分係 bug 行為所得，與 post-fix 行為**不同質**；②`data/baseline_metrics.json`（2026-07-08 A4 重錨）凍結區間涵蓋污染期，其 sharpe/win/alpha 門檻對 post-fix 行為未必適用 → **B11 落地後應重評是否再次重錨**；③兩組合正式重審須待污染量化後辦理。

### 統計/方法債
- IC 建立在 top-N 截斷樣本、n≈20–30，驅動自動停用/翻轉 → B2+B11。**2026-07-31 量化**：漏斗 `1,857 →(流動性) 659 →(粗篩) 150 →(落庫) 20`，IC 只算在 top-20 上（range restriction）。
- ~~**M2 自動停用自鎖迴路**~~ ✅**2026-08-01 已解除**（§6.4 #16）。歷史記錄：停用作用在掃描層 → 無新 `discovery_record` → IC 無從重算 → 模式無法自證恢復；momentum 中斷 29 個交易日、growth 中斷至今。真正機制是 `compute_rolling_ic` 窗口錨在記錄日期範圍、`iloc[-1]` 取最後一個有資料的窗口 → 停掃後**每天重讀同一份凍結 IC 執法**（7/31 用的是 `window_end=2026-06-27`、n=40、34 天前，7/30 與 7/31 完全相同）；「放行」則是窗口生不出來時的 fail-open。**開關的實際判準是「rolling IC 算不算得出來」，與因子有效性無關。** 現由 `src/discovery/ic_governance.py` 的時效閘門阻擋過期執法，且停用改為只作用在 rotation 層（掃描照常）。
- **§3 原則 6 的程式化執行**：✅**M2 與 scanner 門檻加成已於 2026-08-01 落地**（n≥100、窗口 ≥3、時效上限，見 §6.4 #16）；E2b/E2c ✅**已於 2026-08-01 凍結**（§6.4 #17，只記錄不生效）——注意這是**繞過**而非解決：兩者的樣本門檻仍未實作，解凍前必須先補上以標準誤為基準的顯著性判定（B11 完整版）。**仍在運作且無樣本門檻者**：`_compute_win_rate_adjustment` 的勝率回饋 +0.05（見下條）。
- **IC 調權把權重搬給未量測維度**（2026-07-31）：`valuation`／`dividend` 維度不落庫 → 恆 `IC=N/A` → `_apply_ic_weight_adjustment` 的歸一化使其被動吸收被打壓維度釋出的權重（value 7/31：0.150→0.338，2.25×）。淨效果＝IC 治理越積極，決策權越集中到唯一沒被檢驗的維度。**2026-08-01 已凍結止血**（§6.4 #17，不生效但仍記錄）；**根因未解**——解凍前須先把 `valuation`/`dividend` 維度納入 `DiscoveryRecord` 落庫（隨 B2），否則一解凍就復發。
- **分數鑑別力不足**（2026-07-31）：`news_score` 63–81% 恰為 `fillna(0.5)` 填補值卻佔權重 0.20 且每日參與 IC 與調權；momentum 的 `fundamental_score` 73.9%／`technical_score` 45.8% 為填補值。單日 top-20 內 composite 全距僅 0.10–0.15，與 `signal_jaccard_mean` 0.17–0.24（growth/swing）相互印證 → B2 後重估。
- **門檻疊加正回饋**（2026-07-31）：`_apply_score_threshold` = regime 基礎 + 勝率回饋 +0.05 + IC 衰退 +0.05，三個加項全由「近期表現差」驅動，構成「表現差→門檻升→樣本少→統計更不穩→更易判定為差」閉環；`strategy_decay_log` 顯示五個模式全部 `is_decaying=1`，此加壓為常態。實測 7/28 swing 門檻 0.70 → 剔除 88 支 → 落庫 0 筆。**部分緩解**：IC 衰退加成已於 2026-08-01 納入三道閘門（§6.4 #16），過期/小樣本 IC 不再推高門檻；**勝率回饋 +0.05 仍無樣本門檻**，閉環未完全拆除 → 留待 B11 完整版。
- 全部閾值/權重為 in-sample 調參，跨 regime 未驗證 → R2/R3。
- 停損只看收盤、無停利 → B12（live 成交=決策價已由 A2 T+1 解除，2026-07-06）。
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

*v1.1（2026-07-31）：併入 discover 選股邏輯全鏈路審計。異動摘要——§1 斷柱由三根增為四根（訊號引擎停產）；§5 裁決凍結對「模式級比較」部分重新凍結；新增 §6.4 三次止血包（#15 regime 同日多次偵測 P0、#16 M2 凍結 IC、#17 凍結 E2b/E2c、#18 死設定與 log 語意）；§7 B2 與 B11 綁為同一批交付、B11 範圍擴大至 E2b/E2c、N2 由 P2 升 P1 為 §7 #3b；§9 研究議程新增前置條件；§10 新增 3 筆確認 bug、5 筆統計/方法債，並重新定性 scanner 重複為「跨模式不可比」。*

*v1.2（2026-08-01）：止血包三之 #15、#16、#17 落地。異動摘要——§6.4 #15（regime 同日多次偵測）、#16（M2 凍結 IC 執法）、#17（凍結 E2b/E2c）標記完成並補上修法與 live 驗收；#18 的 `insufficient` 分支拆分隨 #16 一併完成（`ScoreThresholdConfig` 接線仍待辦）；§1 斷柱 4 的三項成因兩項劃除，現況改為「訊號產能已恢復、噪音開關已止血，模式級比較仍不可信」；§10 M2 自鎖迴路標記解除，並在「IC 調權搬權重」「§3 原則 6」兩條明示**凍結屬止血非根治、解凍前提條件為何**。**新增模組** `src/discovery/ic_governance.py`＝IC 可執法性的單一實作；**新增設定** `quant.ic_governance`（E2b/E2c 開關，預設凍結）。*
