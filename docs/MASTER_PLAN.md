# 專案主計畫（Master Plan）— Single Source of Truth

> 版本：v1.6（2026-08-09；v1.5 為 2026-08-08、v1.4 為 2026-08-08、v1.3 為 2026-08-04、v1.2 為 2026-08-01、v1.1 為 2026-07-31、v1.0 為 2026-07-05）
> 來源：完整 repository 探索 + 深度工程審計 + CTO Roadmap 三輪分析的整併定稿；v1.1 併入 discover 選股邏輯全鏈路審計（`logs/audit_discover_20260731/REPORT.md`）；v1.3 併入跨模式 PIT 歷史重放審計（`logs/pit_crossmode_20260804/REPORT.md`）；v1.4 併入 momentum 全期 PIT 重放（`logs/pit_momentum_20260807/REPORT.md`）並**改寫 §6.5 #21 裁決**；v1.5 併入 swing 全期重放（`logs/pit_swing_20260808/REPORT.md`）——**純價量兩模式的「顯著為負」裁決全部撤銷**；v1.6 併入 value/dividend 全期重放與 §6.5 #21d 修復；v1.7 更正特徵缺列的影響機制（非 fallback 而是**整批排除於 universe 之外**）並複查 #21e——316 個組合僅 1 天受影響，重跑後結論不變。
> 定位：本文件是策略與工程決策的**單一真相來源**。與 `CLAUDE.md`（開發規則）、`docs/project_history.md`（歷史）互補，不重複。
> 維護規則：任何 P0/P1 事項完成或推翻，須更新本文件對應條目並註記日期。

---

## 1. Executive Summary

**現況**：單人操作、CLI 驅動、SQLite 持久化的台股量化系統（paper trading）。工程紀律（純函數化、1,784+ 測試、audit 文化、自我監控迴路）達 L4，但整體成熟度 **L2.5**，瓶頸不在功能數量（已過剩），在四根斷柱：

1. **帳本不可信**：live 以「夜間決策、當日收盤成交」模擬（該價格拿不到）；rotation 全面忽略股利（除息跳空計為虧損）；Drawdown Kill Switch 的 peak 來自 realized-only 序列（回撤低估）。三者疊加 → 現有 alpha/baseline/歷史裁決全部帶系統性偏差。→ **Phase A 三部曲已完成（A2/A3/A4/A5）**
2. **研究迴路是斷的**：無 point-in-time 能力。回測只能重播已落庫的 DiscoveryRecord（2026-02-27 起、單一多頭 regime、歷代版本混雜）。改 scanner 後無法歷史驗證，只能 forward test 等數週。
3. **執行層為零**：無 broker 抽象，實盤路徑 L0。
4. ★ **訊號引擎曾全面停產**（2026-07-31 discover 審計，`logs/audit_discover_20260731/REPORT.md`）：7/17–7/31 共 11 個交易日中 momentum 掃出 1 天、swing 1 天、growth 0 天；僅存的兩個 active 組合皆 momentum 模式且**持倉 0 筆、100% 現金**。**停產原因與因子好壞無關**，是治理機制自身的三重缺陷：①~~M2 以 34 天前 n=40 的凍結 IC 執法~~ ✅2026-08-01 已修（§6.4 #16，五模式恢復掃描）；②~~regime 同日被偵測 4–15 次~~ ✅2026-08-01 已修（§6.4 #15）；③五個 scanner 非同一條 pipeline → ✅2026-08-01 **結構已統一**（N2，§7 #3b），但**閘門政策刻意保留現行行為**，跨模式此刻**仍不可比**。另 §6.4 #17 已凍結 E2b/E2c（噪音驅動的自動調權與翻轉不再生效）。**現況**：訊號產能已恢復、噪音開關已止血、漏斗已收斂為單一實作且差異可審查；**剩下的是政策決定**——逐項開啟閘門（實測影響見 §7 #3b）＋ B2 全候選池落庫，兩者完成後模式級比較才成案

5. ★ **訊號本身尚未證明有正超額**（2026-08-04 跨模式 PIT 重放，`logs/pit_crossmode_20260804/REPORT.md`）：首次以**單一程式版本**重放 30 個歷史基準日（2024-01 ~ 2026-06）。以「基準日籃子平均」為觀測單位、扣除同期 TAIEX 後，**五個模式的 20 日超額報酬全數無法與零區分**，其中 momentum **−2.66%（t=−2.29）**、swing **−2.58%（t=−2.12）顯著為負**，且**未計交易成本**（A4 模型約 0.7–0.9%/來回）。原始報酬看似「價值系完勝、momentum 歸零」——扣掉 beta 後全部塌回零附近，「momentum 2026 復甦 +5.02%」的超額僅 **+0.64%**。<br>**但三個模式的結果無效**：`stock_valuation` 表僅有 2026-01-26 起的資料（B1① 只回補了 OHLCV，未回補基本面）。三者的失效機制不同——**value 是 fail-open**（`_value.py:_coarse_filter` 的 `else` 分支跳過整段 PE/殖利率閘門），2024–2025 跑的是退化後的流動性＋法人買超篩選；**dividend 與 growth 則是 fail-closed**，dividend「30 天只有 4 天選得出股票」正是因此（那 4 天恰好全在 `stock_valuation` 有資料之後），growth 則受限於 `monthly_revenue` 2024 年僅 5 支股票。**故唯一可採信者是 momentum/swing（純價量），而結論是負面的。**<br>**2026-08-05 進度**：`stock_valuation` 已回補（§6.5 #20，每日 1,580~1,656 檔）、fail-open 已修（#19）。實測同一基準日 value 的選股 **9/20 換人**，證實原結果確實在量測別的東西。`monthly_revenue`／`financial_statement` 仍未補 → **growth 的歷史重放仍不可用**。<br>**🔄 2026-08-08 momentum 部分已推翻**（`logs/pit_momentum_20260807/REPORT.md`，詳 §6.5 #21）：2020–2023 回補完成後重放 79 個基準日，在評分輸入同質的 2020–2024（39 日、bear 14／bull 22／crisis 3）momentum 20 日超額 **−0.16%（t=−0.24）＝與零無法區分**，非原記載的「顯著為負」。上述 −2.66% 是 2024–2026 單一窗口的結果，本次重跑該窗口得 −2.90%(t=−3.54) 與之一致——**計算沒錯，錯在把一個窗口當成模式的性質**。**修正後的表述**：momentum 毛超額為零，扣 A4 成本後為負（10 日超額 −0.35% vs 成本 0.7–0.9%/來回）。**swing 已於 2026-08-08 同步重跑並同樣推翻**（同質期 55 個基準日、20 日超額 **+0.04%（t=0.06）**，詳 §6.5 #21c）。**✅ 2026-08-09 四模式全部重跑完畢**（§6.5 #21e）：2020–2024 同質期的 20 日超額 momentum −0.16%／swing +0.04%／value −0.13%／dividend −0.38%，**四者皆與零無法區分**。本節標題的「尚未證明有正超額」由此升級為實證——不再是 30 個基準日、輸入殘缺的推測，而是 79 個基準日、四模式、跨四種 regime 的量測結果；同時**所有「顯著為負」的裁決全部撤銷**，正確表述是「毛超額為零、扣成本後為負」。growth 因 `monthly_revenue` 缺口仍不可評價。<br>**⚠ 新增的跨期不可比來源**：`announcement` 全表僅 2026 年有資料（45,613 筆），而 `news_score` 佔評分權重 **0.20**——2026 以前該維度恆為 `fillna(0.5)` 常數。同理 `broker_trade`／`securities_lending` 只有 2025–2026、`monthly_revenue` 只有 2025 起才成規模。**任何橫跨 2025 的期間比較都同時混著市場變化與評分輸入變化**。

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
| 資料深度 | **全市場個股 2024-01 起**（B1① 回補完成，624 交易日）；DailyFeature **2024-01 起**（B1② 歷史化完成，339 萬筆）；DiscoveryRecord 2026-02-27 起；0050 2025-11 起。⚠ 原記載「2025-01 起」**有誤**——實測 2024 全年至 2025-10 每日僅 5–7 檔（watchlist+TAIEX），真正全市場覆蓋自 2025-12 才開始，現已回補至 2024-01 | 2020~2023 仍待回補（約 9 小時）|
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

### 6.5 四次止血包（2026-08-04 跨模式 PIT 重放審計新增；詳 `logs/pit_crossmode_20260804/REPORT.md`）

> 背景：首次以單一程式版本重放 30 個歷史基準日。發現 **①訊號扣除大盤後無正超額、momentum/swing 顯著為負**；**②三個模式的重放結果因基本面資料缺口 + 粗篩 fail-open 而無效**。

| # | 項目 | 內容 / 驗收 |
|---|------|------------|
| 22 | **Stage 0.5 覆蓋率閘門看錯軸（P1）** | ✅ **2026-08-05 已修**。`_maybe_sync_valuation` 原本數的是 `stock_valuation` **全表的相異 stock_id**（`select(count(distinct(stock_id)))` 無日期條件），一旦歷史上曾累積 ≥500 檔就**永遠不再觸發**全市場同步。而 live 每日真正寫入的只有 `sync_valuation_for_stocks` 補的候選池——**實測 2026-07-31 全表 1,505 檔（閘門關閉）但當日僅 43 檔有估值**（近三日 43／75／134）。後果：value/dividend 的 `_coarse_filter` 以 `groupby("stock_id").last()` 取「最新一筆」估值，實際拿到的是**數月前的舊 PE** 在評分——這不是 look-ahead（用的是過去資料）但是嚴重失真，且與 §6.5 #19 的 fail-open 疊加，共同構成「估值表看似有 1,505 檔、實則每日只有數十檔新鮮值」的假象。<br>**修法**：改看近 `VALUATION_FRESH_WINDOW_DAYS`（7）日窗口內的相異股票數，門檻 `VALUATION_MIN_FRESH_STOCKS`（500）；窗口上界為 `self._as_of()`，故 PIT 重放時未來資料不得計入覆蓋率。**驗收**：`tests/test_backfill.py::TestValuationFreshnessGate` 3 測試（陳舊大量不算數／新鮮足量跳過／未來列不算數）；還原為全表計數可複現 2 個失敗 |
| 19 | **粗篩 fail-open（P1）** | ✅ **2026-08-05 已修**。`_value.py:_coarse_filter` 在 `_df_valuation` 為空時走 `else` 分支，把 `pe_ratio`/`pb_ratio`/`dividend_yield` 設為 None 後**整段放行**——定義性閘門不是收緊而是消失，模式靜默變成「基本過濾 + 成交量排名 + 法人淨買超 + 5 日動能」＝流動性篩選，且 log 無任何警示（違反 §3 原則 8）。<br>**⚠ 原記載「`_dividend.py`／`_growth.py` 同構」有誤**（2026-08-05 實查更正）：只有 value 是 fail-open，另兩者的 `else` 分支本來就是 `return pd.DataFrame()`。**三個模式的失效機制不同**——value＝fail-open（跑成流動性篩選、產能 100% 是假象）；dividend＝fail-closed（**這才是 4/30 產能的真因**）；growth＝fail-closed 但 `monthly_revenue` 2024 年僅 5 支股票故 universe 極小。§6.5 的「三模式重放結果無效」裁決不變。<br>**修法**：新增 `MarketScanner._require_coarse_data(df, table=, gate=)` 作為共用守門（缺席時記 WARN 並回 False），三個資料依賴型模式統一走它——value 由 fail-open 改為 fail-closed，dividend/growth 行為不變但語意與 log 統一。<br>**live 影響為零**：回補後估值表每日 1,580~1,656 檔，`else` 分支在 live 不會走到；此修法只作用於歷史重放與新環境部署。<br>**回補後實測（2024-06-05，value）**：選股 20 檔中 **9 檔換人（重疊率 55%）**，20 日報酬 +6.05%→+4.78%。**漏斗數字兩次都是 `769→150→20`**（`top_n_candidates` 恆截在 150）——即**筆數相同、內容不同**，代表此問題無法從漏斗統計察覺，只能實查資料表。<br>**驗收**：`tests/test_scanner.py::TestCoarseFilterFailClosed` 5 測試（三模式 fail-closed + 有資料時行為不變的對照組 + 原始碼契約測試防止新模式再寫出 fail-open）；還原 fail-open 可複現 2 個失敗。<br>**後續**：`ReplayResult` 的資料覆蓋欄位 → §6.5 #21b（✅2026-08-06 已完成） |
| 20 | **`stock_valuation` 歷史回補（P1，B1① 的缺口）** | ✅ **2026-08-05 已完成**（`backfill_valuation_history` + `backfill-history --valuation-only`）。<br>**分兩條路走，因為兩邊官方端點狀況不同**：<br>　• **上市**：TWSE `BWIBBU_d` 每日全市場端點健在且**有完整歷史**（實測 2024-01-02 回傳 997 檔）。<br>　• **上櫃**：`peratio_book/pera_result.php` **已下架**——所有日期（含當日）皆 302 導向 `/errors`；TPEX 新版 openapi（`tpex_mainboard_peratio_analysis`）只回**當日**、無日期參數。故上櫃歷史**無官方來源**，改走 FinMind `TaiwanStockPER` 逐股（支援日期區間，一檔一次呼叫涵蓋全期間 ≈ 0.5s）。此處逐股看似違反「官方優先」的資料源順序，但這裡沒有官方選項。<br>**續跑判定**（沿用 B1① 的「以 DB 現況為準、不存進度檔」）：上市看當日估值檔數 ≥ `BACKFILL_MIN_VALUATION_STOCKS`（**800**，因估值母體僅約 1,000 遠小於價量的 1,900）；上櫃看該檔估值日數 ≥ 其價量日數 × `VALUATION_COVERAGE_RATIO`（0.8，不用固定日數——上櫃股上市時間不一）。<br>**驗收**：`tests/test_backfill.py` 新增 12 測試（含「僅有候選股估值的日期必須視為未補」「ETF 不應消耗 FinMind 額度」「單檔失敗不中斷整輪」）+ `tests/test_cli_smoke.py` 3 測試（含「`--valuation-only` 不得誤觸下市同步/價量回補」）。<br>**⚠ 仍未補**：`monthly_revenue`（2024 年僅 5 支股票）與 `financial_statement`（全表 365 筆）→ growth 模式的歷史重放仍不可用 |
| 21 | **momentum 系組合重審（P1）** | 🔄 **2026-08-08 裁決改寫**（詳 `logs/pit_momentum_20260807/REPORT.md`）。<br>**原記載已推翻**：2026-08-04 的「momentum 20 日超額 −2.66%（t=−2.29）**顯著為負**」建立在 22 個基準日、且全在 2024–2026 的單一窗口上。2020–2023 回補完成後以**同一程式版本**重放 79 個基準日（52 個有選股），在**評分輸入同質**的 2020–2024 期間（39 個基準日、跨 bear 14／bull 22／crisis 3），20 日超額為 **−0.16%（t=−0.24，p=0.809，勝率 49%）——與零無法區分**；5d +0.06%(t=0.20)、10d −0.35%(t=−0.76) 同樣不顯著。同一窗口（2024–2026）本次重跑得 −2.90%(t=−3.54)，與前次一致——**前次的計算沒有錯，錯的是把一個窗口的結果當成模式的性質**。<br>**新裁決**：momentum **毛超額為零**，而非為負。但配上 A4 成本（0.7–0.9%/來回）與 live 的 5–10 日持有期，以同質期 10 日超額 −0.35% 計，扣成本後每週期約 **−1.05%~−1.25%**、一年約 25 個週期。**問題不是 momentum 有害，是它沒有 edge 可以支付交易成本**——行動方向與原裁決相同，但修法方向不同：**降換手／提高進場門檻，而非停用模式**。<br>**⚠ 2025–2026 的負值（2025 −3.68% t=−4.10；2026 −3.05%）真實但不可歸因**：`announcement`（**權重 0.20**，2026 前全表 0 筆＝`news_score` 恆為 `fillna(0.5)` 常數）、`monthly_revenue`（2025 起 402→1,902 檔）、`broker_trade`、`securities_lending` 恰在 2025–2026 才變成真實值，**期間效應與輸入變更完全共線**。已排除 regime 組成解釋——同一個 bull regime 在同質期 +0.07%、在此期 −3.16%。分離實驗見報告 §7 行動 2（停用 news/broker/SBL 維度重跑那 13 日，約 42 分鐘）。<br>**附帶推翻**：「momentum 在 bear 應納入 `REGIME_MODE_BLOCK`」假設**正面否定**——同質期 bear 14 個基準日超額 **+0.39%、勝率 64%**。<br>**不成案（明確記錄）**：crisis 4 天超額 −5.64%(t=−2.70、勝率 0%) 方向一致但 **n=4**，遠低於 §3 原則 6 門檻，與 2026-08-04 被正確擱置的 bear 假設是同一個陷阱——**不採取行動**，登記為待驗證假設。<br>**與既有裁決的關係**：mg5_20d 降級（R1c）不受影響；**swing 已於 2026-08-08 同步重跑，裁決一併推翻**（見下列 21c）。<br>**⚠ 仍不是「立即停用」的授權**：重放假設等權買進持有到期、**無停損無輪動**，而 live 有停損與輪動，兩者不等價 |
| 21c | **swing 裁決同步改寫（P1）** | 🔄 **2026-08-08**（詳 `logs/pit_swing_20260808/REPORT.md`）。以與 momentum **逐項相同**的設定重放 79 個基準日（swing 不受 `REGIME_MODE_BLOCK` 約束故全數實掃，350 分鐘；`no_data` 0 個）。<br>**原記載已推翻**：2026-08-04 的「swing 20 日超額 −2.58%（t=−2.12）顯著為負」同樣建立在 2024–2026 單一窗口。同質期 2020–2024（**55 個基準日**，bear 13／bull 22／sideways 19／crisis 1）20 日超額為 **+0.04%（t=0.06，p=0.953，勝率 45%）——與零無法區分**；5d +0.32%(t=1.18)、10d −0.43%(t=−0.92) 亦不顯著。**兩個模式的舊裁決都來自同一個窗口，也都在擴大樣本後塌回零。**<br>**⚠ R1b（swing5_3d 暫停）維持，但理由必須更換**：其真正依據是 2026-05-16 的 **live 成本歸因**（`cost 1.68%／turnover 4.90×`，−1.68% alpha 完全由成本解釋），屬實盤證據不受重放影響。swing5_3d 持有期 **3 天**、換手高於 momentum 系，零毛超額配上 A4 成本必然為負——**結論不變、理由要換**。§6.5 #21 原寫的「swing −2.58% 獨立支持 R1b」已失效。<br>**與 momentum 的關鍵差異**：swing **沒有** momentum 的 2025 劣化（swing 2025 +0.89% t=0.36 vs momentum −3.68% t=−4.10）。若「2025 起接上的 news／broker／SBL 維度有害」為真，兩模式應同時劣化——這**削弱**（未否定，因 `REGIME_WEIGHTS` per-mode 權重不同）該讀法，使 #21 的歸因實驗優先度下降。<br>**不成案**：sideways 是 swing 最差 regime（−1.17%、勝率 37%、n=19），而 momentum 正是在 sideways 被封鎖——但 **t=−1.19 未達顯著**，不足以支持把 swing 加入 `REGIME_MODE_BLOCK`，登記為待驗證假設 |
| 21d | **指標暖身缺口（P2，新）** | ⚠ **2026-08-08 發現，分析層已擋、重放層未修**。`daily_feature` 自 2020-01-02 起算，MA60 需 60 個交易日才填滿：實測 2020-01-02 的 ma20/ma60 **100% NULL**、2020-02-10 的 ma60 仍 **100% NULL**（穩態約 23%）。**§6.5 #21b 的覆蓋度閘門抓不到**——它數「當日有幾列特徵」，數不到**欄位本身是否為 NaN**，故這些日子被標成 `ok` 卻是退化輸入，與 fail-open 同一類的靜默失效。<br>**momentum 恰好未受污染**（那幾天全是 sideways 被 `REGIME_MODE_BLOCK` 擋掉、未進籃子，籃子最早 2020-07-07）——**這是運氣不是設計**；swing 不吃該封鎖就踩到了。<br>**✅ 2026-08-09 已修**（併入 §6.5 #21b 的 `DataCoverage`）：新增 `feature_warm_ratio` 欄位——取 `ma60` 與 `turnover_ma20` 非空率的**較小值**（ma60 窗口長，是 binding constraint），門檻 `REPLAY_MIN_FEATURE_WARM_RATIO`=0.5；`daily_feature` 的判定改為**列數與欄位都要過關**。<br>**母體限 4 碼普通股**——這是關鍵細節：不限 4 碼時穩態會從 **0.988~0.998 掉到 0.646~0.786**（權證/ETN 上市時間短、MA 天生填不滿），門檻將無從設定。實測分離乾淨：暖身失效 0.000（2020-01-02／01-20／02-10）vs 穩態 0.988~0.998。<br>**順帶修正一項過度保守**：先前分析層用「起算後第 61 個交易日」推得邊界 2020-04-09，但實測 2020-03-10 的暖身率已達 **0.988**——**比率量測比日期啟發式準確**，日期法會白白丟掉可用的基準日。<br>**驗收**：`tests/test_pit.py::TestFeatureWarmup` 4 測試（列數滿額但欄位全 NaN 必判 `no_data`／欄位填滿的對照組／遠低於穩態但在門檻上不得誤殺／取較小值不得被高覆蓋欄位稀釋）；既有 `TestDataCoverage` 的種子同步補上 MA 欄位。全測試 2,830 passed |
| 21e | **四模式全期重放對照（P1）** | ✅ **2026-08-09**（`logs/pit_swing_20260808/`，value 27.5 分／dividend 22 分）。四個模式、同一程式版本、同一組 79 個基準日、同一套統計口徑。**2020–2024 同質期 20 日超額全數與零無法區分**：momentum −0.16%(t=−0.24，n=39)／swing +0.04%(t=0.06，n=55)／value −0.13%(t=−0.26，n=58)／dividend −0.38%(t=−0.73，n=58)。**沒有任何一個模式展示出可辨識的 edge**——這比「某個模式為負」更根本，也把 §1 第 5 點的「訊號尚未證明有正超額」從 30 個基準日、輸入殘缺的推測，升級為 79 個基準日、四模式、跨四種 regime 的實證。<br>**2025–2026 全部轉負**：momentum −3.39%(t=−2.86，p=0.014)／dividend −2.78%(t=−2.20，p=0.042) 顯著；value −1.65%(t=−1.54)／swing −0.30%(t=−0.16) 不顯著。勝率 28~50%。<br>**部分可歸因於大盤形態**：2026 的 TAIEX 20 日平均報酬達 **+6.77%、中位 +10.85%**（噴出行情），等權 20 檔籃子跑輸市值加權指數屬預期。**但 2025 不是**——大盤 +2.79% 與 2024 的 +2.51% 相當，四模式中三個仍為負。**2025 的劣化無法用大盤速度解釋，仍與新輸入上線共線**（見 §1 第 5 點的跨期不可比清單）。<br>**⚠ 解讀限制（N2）**：value/dividend 的單日掃描僅需約 20 秒，momentum/swing 需約 5 分鐘——差距來自 `StageConfig` 多數旗標為 False，跳過 3.7 分數門檻／4.1 產業分散／3.5 系籌碼閘門等六道。**四模式的數字並非在同一條漏斗上產生**，跨模式強弱比較仍受 §7 #3b 的閘門政策未決所限。<br>**✅ 2026-08-09 特徵缺列複查——本表數字維持**：`backfill_daily_features` 的續跑判定 bug（§10 登記簿）使 11 個交易日的上櫃股沒有特徵列，而本次重放正是在修復前跑的。以 CSV 當初記錄的 `feature_stocks`／`price_stocks` 兩欄回溯全部 **316 個（模式 × 基準日）組合，僅 4 個覆蓋率 < 0.95**——全部來自同一天 **2026-06-22**（1,089/1,971 ＝ 0.553，四模式皆中），其餘 75 個基準日乾淨。修復資料後**固定原格點只重跑該日**（見下方「為何不重跑 run.py」），結果：**2020–2024 同質期四模式數字一個都沒動**（該日不在此窗口）；2025–2026 位移 ≤0.09pp——momentum −3.39%(t=−2.86)→**−3.30%(t=−2.68)**、swing −0.30%→−0.29%、**value／dividend 完全相同**（那 882 檔缺特徵的上櫃股原本就進不了它們的漏斗，`_STAGES` 跳過 DailyFeature 相關階段——同一個 N2 不對等，這次剛好使它們免疫）。momentum 的顯著性維持（p 仍 <0.05）。**本表所有裁決不變。**<br>**⚠ 為何不重跑 `run.py` 做這個對照**：那 8 個被補齊的 2022~2023 交易日從「不合格」（973~987 檔 < `BACKFILL_MIN_COMMON_STOCKS`）變成合格，`sample_replay_dates` 的 every-20 格點因此整個推移——實測新舊 79 個日期**只有 26 個相同**，2022-02-24 起分歧。重跑會得到**另一組樣本**，差異將混著資料修復與抽樣雜訊，無法回答「修好資料後結論有沒有變」。正確作法是固定原格點、只重算受影響的基準日 |
| 21b | **重放結果須自帶可採信標記（P2）** | ✅ **2026-08-06 已完成**。`n_picks == 0` 有兩種**完全不同**的意義——模式看過全市場後判斷不進場（是結論），或定義性輸入根本缺席（結果無效）。舊版 `ReplayResult` 只記 `n_picks`，兩者無從分辨，這正是 2026-08-04 審計把「dividend 30 天只選得出 4 天」記成模式產能的原因（真因是 `stock_valuation` 在 2026-01-26 前無資料）。<br>**修法**：新增 `assess_data_coverage(mode, as_of)` → `DataCoverage`，以及 `ReplayResult.coverage` / `verdict`（`ok` / `no_picks` / `no_data`）。依賴宣告在 `MODE_REQUIRED_TABLES`——**只列粗篩閘門的依賴**：`financial_statement` 雖參與品質評分，但缺席只使該維度分數退化、不會把選股歸零，列進來會把可評價的重放誤殺。門檻**沿用各自的 SSOT 不另立數字**（`BACKFILL_MIN_COMMON_STOCKS` / `VALUATION_MIN_FRESH_STOCKS` / `REPLAY_MIN_FEATURE_RATIO`=universe Stage 2 的 0.3 / `REPLAY_MIN_REVENUE_STOCKS`=300）；估值取近 7 日窗口、月營收套法定公布時滯，與 scanner 實際取數條件一致（否則量到的不是 scanner 看到的）。**覆蓋度在 scanner 跑之前量測**——跑完才量的話 Stage 0.5 補抓可能已改變資料表。CLI 將 `no_data` 日**排除於彙總**並單獨計數，產能率改以可採信日為母體。<br>**實跑即抓到真例**：growth 2024-06-03 從 15,237 檔 universe「產出 2 檔」——舊版會把這 2 檔的報酬平均進彙總，實際只來自當時僅有的 5 支有營收股票。<br>**2026-08-09 擴充（併入 #21d）**：`daily_feature` 的判定改為**列數與欄位都要過關**——新增 `feature_warm_ratio`（`ma60`／`turnover_ma20` 非空率取較小值，限 4 碼普通股，門檻 `REPLAY_MIN_FEATURE_WARM_RATIO`=0.5）。<br>**驗收**：`tests/test_pit.py::TestDataCoverage` 13 測試 + `TestFeatureWarmup` 4 測試（per-mode 判定／未來列不算數／陳舊列不算數／營收公布時滯／半套日／特徵缺席／欄位未暖身／verdict 四象限矩陣／新模式未登記依賴的契約測試）。真實 DB 驗證：2024-06-05 與 2025-04-07 的 momentum/value/dividend 判為就緒（估值已回補 1,578~1,611 檔）、growth 判為 `no_data`（營收僅 5 檔），與已知缺口完全吻合。全測試 2,830 passed |

---

### 6.6 基本面資料層補齊（B1① 收尾；2026-08-15 規劃，探勘實測見各項）

> 背景：§6.5 收尾後 `daily_price`／`daily_feature`／`stock_valuation` 已零缺口，**只剩 `monthly_revenue` 與 `financial_statement`**。補齊後 growth 首次可評價、§6.5 #21e 的四模式對照才能補成五模式。
>
> **規劃期探勘推翻了三項既有認知**：①月營收**不必動用 FinMind 配額**（MOPS 歷史頁面 2020 起健在且現有 parser 直接可解析）；②`monthly_revenue` 存在**兩套日期語意**且已產生 2,488 組重複；③「2025 起營收成規模」是**假象**（2025-01~07 僅 5 檔、2025-08~12 僅 ~400 檔且 yoy 全 NULL）。
>
> **執行順序**：#23 → #24 →（#27 可並行）→ #25 → #26。**#26 必須排在 #25 之後**——補營收與補財報都會改變五個模式的分數，先跑重放等於 20~30 小時算力白花。
>
> **⚠ live 副作用**：#24/#25 動的是同一顆 `data/stock.db`，故 live 選股會跟著變（dividend 的 EPS 連續性閘門首次生效、momentum/swing/value 首次吃到 peer 加成、fundamental 維度不再是常數 0.5）。回補前後各留一次 `rotation preview` 作 A/B；預期 morning-routine Step 17 `validate-baseline` 告警，確認係資料補齊而非策略退化後 `update-baseline` 重錨並登記 `strategy_events`。

| # | 項目 | 內容 / 驗收 |
|---|------|------------|
| 23 | **月營收資料語意先修（P1，#24 的前置）** | ✅ **2026-08-15 已修**（分支 `feature/revenue-semantics-fix`）。四個缺陷疊在同一張表上，不先修就回補會把重複面積放大 60 倍。<br>**修法**：canonical 日期＝營收月份月底（SSOT `pit.month_end`），`fetch_monthly_revenue` 輸出正規化；MoM/YoY 改以「年 ×12＋月」序位對齊查表（缺月即 NaN，不再用位置 `shift(12)` 張冠李戴）；`_upsert_monthly_revenue(df, source=)` 依來源決定衝突語意（mops 覆寫、finmind 保守不覆寫）；`_sync_per_stock` 新增 `incremental=False` 供月營收恆取完整窗口（**只改 lookback 不夠**——增量起點會讓窗口塌回一兩個月，YoY 照樣 NULL）；`monthly_revenue.source` 欄位 + `normalize_revenue_date_semantics()` 一次性遷移。<br>**live DB 實跑**：標記來源 13,916 筆、改期 6,175 筆、**合併同月重複 2,488 筆**（與規劃期實測完全一致）。<br>**驗收**：`tests/test_backfill.py::TestRevenueDateNormalization`(4) / `TestMopsRevenueGate`(4) / `TestGrowthStage05RevenueGate`(3)、`tests/test_fetcher.py::TestMonthlyRevenueDateSemantics`(4)、`tests/test_cli_smoke.py::TestRevenueBackfillCli`(3)。全測試 **2,852 passed**。<br>**四個缺陷原貌**：<br>**① 兩套日期語意**：FinMind 寫「次月 1 日」（`2024-02-01` ＝ 1 月營收），MOPS 寫「當月月底」（`2026-01-31` ＝ 1 月營收），而 unique key 是 `(stock_id, date)` **擋不住**——實測 **2,488 組 `(stock_id, year, month)` 重複**。後果不只是多幾列：`pivot_revenue_rows` 取 `grp.head(months)`，故 growth 要的 **4 個月窗口實際只拿到 2 個月**，`prev_yoy_growth` 還可能是同一個月的另一份來源。canonical 取**月底**（`data/pit.py:revenue_visible_cutoff` 即照月底建模）。<br>**② FinMind 逐股列 96% 的 `yoy_growth` 是 NULL**：`sync_revenue_for_stocks` 的 `lookback_days=180` 只取回 6 列，而 `fetch_monthly_revenue` 要 `len(df) > 12` 才算得出 YoY（實測 8,663 筆 day-01 列中 8,353 筆 NULL）。growth 粗篩是 `yoy_growth.notna() & > 10`，**這些列在粗篩就全數蒸發**。修法：`lookback_days` 180 → 430。<br>**③ `sync_mops_revenue` 續跑閘門看錯軸（與 #22 同型）**：門檻「該月列數 ≥ 500」**混算了候選池逐股補抓的列**，月初跑一次拿到部分公司 + 候選池累積即越過 500 → 該月**此後永不重抓**。實測 2026-02 的 MOPS 列只有 **1 筆**（候選池 1,284 筆）、2026-06 為 498 筆。修法：改數 `source='mops'` 的相異股票數 ≥ `BACKFILL_MIN_REVENUE_STOCKS`(1,400)（判定與補抓統一收斂到 `_sync_mops_revenue_month`）。門檻取 1,400 因成熟月份實測 1,658~1,761 檔、月初半套時只有數百檔，落在中間可乾淨分離；不另設「月齡」條件——morning-routine Step 5 每日呼叫，未達標就每天重抓到達標為止，自癒不需要額外旋鈕。<br>**④ growth 的 Stage 0.5 閘門同病（實作時發現）**：`_growth.py:_prepare_before_load` 數的是 `monthly_revenue` **全表**相異 stock_id（無日期條件）< 500 才補抓——全表早已 1,900 檔，這道自癒**從未觸發過**。改為看「當下依法已公布的那個月份」由 MOPS 抓回幾檔，並委派給 `_sync_mops_revenue_month`（門檻 SSOT 單一）。注意目標月份要用 `latest_visible_revenue_month(as_of)` 而非「日曆上的上個月」——月初尚未到 10 日時兩者差一個月，用錯會每天重抓一個還沒公布的月份。PIT 重放不受影響（`_prepare_before_load` 只在非 offline 時呼叫）。 |
| 24 | **`monthly_revenue` 2020-01 ~ 今回補（P1）** | ✅ **2026-08-15 已完成**（8 分鐘、158 個免費請求，零配額消耗）。**實跑結果**：78 個月 / 136,474 筆，全表 139,774 筆 / 1,919 檔 / 2020-01 ~ 2026-07 **79 個月全部達標**（每月 1,687~1,918 檔）、同月重複 **0**、非月底日期 **0**、`yoy_growth` 非空率 **40% → 99.3%**。<br>**續跑判定當場證明有效**：2022-06 的上櫃頁面回 502，該月只寫進 927 檔 → 未達 1,400 故**未被標記完成**，重跑該月即補足 1,720 筆。若沿用舊的「≥500 就跳過」，這個月會永久停在半套且無人察覺。<br>**growth 首次可評價**：`assess_data_coverage("growth", d)` 對 2020~2025 六個抽樣日全部 `sufficient=True`（revenue_stocks 1,666~1,841，原本恆為 5）；實跑 `pit-replay growth --date 2022-06-15` 得 10,488 → 粗篩 150 → 產出 10 檔、verdict 可採信，且 `fundamental_score` IC=0.221（**不再是常數 0.5**）。<br>**原規劃**：**走 MOPS 不走 FinMind**——實測 `mopsov.twse.com.tw/nas/t21/{sii,otc}/t21sc03_{roc}_{m}_0.html` 歷史頁面健在（2020/1 回 HTTP 200、既有 `_parse_revenue_html` 直接解析出 **1,658 檔**且 yoy 僅 4 筆 NULL；2022/7 為 1,731 檔）。79 個月 × 2 市場 ＝ **158 個免費請求、約 8 分鐘**，取代原規劃的「FinMind 逐股 2,000 檔吃配額」，且符合 §2 資料來源優先序①。<br>**修法**：新增 `pipeline.backfill_revenue_history(start_ym, end_ym)`，逐月呼叫既有 `fetch_mops_monthly_revenue`，續跑判定沿用 #23 的閘門；CLI `backfill-history --revenue-only --start 2020-01`。<br>**驗收**：2020-01 ~ 今每月 ≥1,400 檔；`assess_data_coverage("growth", d)` 對 2020–2024 抽樣日回 `ok`（門檻 `REPLAY_MIN_REVENUE_STOCKS`=300）。<br>**已知限制**（寫進 docstring）：MOPS 頁面是**現在**的版本，公司事後更正的營收無法還原當時值 |
| 25 | **`financial_statement` 2020 ~ 今回補（P1）** | 🔨 **2026-08-15 實作完成，待執行**（`backfill_financial_history` + `backfill-history --financial-only [--wait-on-quota]` + `finmind-quota`；分支 `feature/revenue-semantics-fix`）。**dry-run 實測**：待補 **1,979 檔**、已跳過 15 檔、預估 **9.9 小時**。執行指令：<br>`caffeinate -i python main.py backfill-history --financial-only --start 2020-01-01 --wait-on-quota`<br>**實作期的三個設計決定**：<br>　• **續跑判定看欄位不看列數**（§6.5 #21d 同型）：三表任一逾時時 `fetch_financial_summary` 仍回傳只有損益表的 DataFrame，寫進去就是 `equity`/`operating_cf` 全 NULL 的半套列，列數檢查完全看不出來。故 `eps`/`equity`/`operating_cf` 分別計數。連帶**必須**把 `_upsert_financial` 改為 `on_conflict_do_update`——否則重抓回來的完整值被半套列擋住，判定永遠自癒不了（C2 教訓）。<br>　• **應有季數受法定申報期限約束**：不扣掉「尚未到申報期限」的最近一季，該季會被當成缺漏使每檔每次都重抓（用 `pit.quarter_publish_deadline`）；同時受該股價量區間約束，新上市/已下市股不被要求補滿全期。<br>　• **連續慢跑而非爆衝**：節流間隔由 `fetch_quota_status()` 的真實上限推導（3600/600＝6 秒/請求）。與「0.5 秒衝到撞 402 再等整點」每小時吞吐相同，但不會把日誌塞滿 402。<br>**驗收**：`tests/test_backfill.py::TestFinancialExpectedQuarters`(3)／`TestFinancialBackfillResume`(4)／`TestFinancialBackfillUniverse`(3)／`TestFinancialBackfillQuota`(4)／`TestSecondsUntilNextHour`(1)、`tests/test_fetcher.py::TestQuotaStatus`(4)、`tests/test_cli_smoke.py`(6)。**還原判定為列數可複現 2 個失敗、還原 upsert 為 do_nothing 可複現 1 個**（已實測）。全測試 **2,880 passed**。<br>**原規劃**：無官方全市場歷史端點，走 FinMind 逐股三表（損益／資產負債／現金流）。**配額實測（2026-08-15）**：level=1 Free、`api_request_limit` **600/hr**；免費版**不限歷史深度**（實測 2330 單次呼叫取回 2020-03-31 ~ 2026-06-30 共 26 季）。<br>**母體**：4 碼普通股且區間內 ≥60 交易日 ＝ **1,994 檔** × 3 dataset ＝ **5,982 次 ≈ 10 小時**（已裁示全補，不砍尾端——`peer_fundamental_ranking` 是「同業相對排名」，母體半套等於同業比較半套，又一個靜默退化）。<br>**修法**：`pipeline.backfill_financial_history()`，節流走 **6.0 s/req 連續**（＝600/hr）而非 0.5s 爆衝撞 402；沿用 `_is_quota_exhausted`，加 `--wait-on-quota` 等到整點續跑；續跑判定＝該檔已有季數 ／ 該檔有價量的季數 ≥ 0.8。**三表全成功才寫入**——只補到損益表會讓 `roe`／`debt_ratio`／`free_cf` 靜默變 NULL（fail-open 同型）。開跑前先印配額（新增 `FinMindFetcher.fetch_quota_status()` + `python main.py finmind-quota`）。<br>**⚠ 必須在使用者自己的 Terminal + `caffeinate -i` 跑**——長跑綁互動 session 會被砍（2020–2023 價量回補跑三趟的教訓） |
| 26 | **五模式全期重放（P1，#21e → 五模式）** | 🔨 **2026-08-15 腳本就緒，待 #25 完成後執行**（`logs/pit_fivemode_20260815/`：`run.py`／`analyze.py`／`compare.py`／`_common.py`／`README.md`，runbook 見該目錄 README）。<br>**`run.py` 自帶前置條件檢查**：營收達標月份 <70 或財報（含資產負債表）<1,000 檔即中止——沒補就跑只是把 #21e 重跑一次，25 小時白花。實測目前擋在「財報僅 19 檔」。<br>**格點釘死**在 `logs/pit_swing_20260808/results_momentum.csv` 的 79 個 as_of（`load_grid()` 數量不符直接中止），三個腳本共用 `_common.py` 的籃子建構與統計口徑，避免上一版 `bench_returns` 複製兩份的問題。額外記錄 `revenue_stocks`／`feature_warm_ratio` 供事後查核每日可採信依據。`compare.py` 直接輸出與 #21e 舊值的位移對照。<br>**已用合成資料跑通三個腳本**（含 no_data／regime 封鎖／首次可評價三條分支）。<br>**排在 #25 之後**。**四模式必須跟著重跑，不能只補 growth 一列**——實查出三條污染路徑：①`_compute_fundamental_scores` 在 `_base.py`**五模式共用**，營收空表時恆回 `0.5` 常數，故 #21e 的四模式等於少一個維度；②`peer_fundamental_ranking` ±3% 對 momentum/swing/value 為 True，但財報全表僅 15 檔，**這道加成從沒作用過**；③`_dividend.py` 的 EPS 連續性閘門註明「無財報資料者 pass through」，全表 15 檔＝**閘門是死的**，補完才首次生效。<br>**格點固定沿用** `logs/pit_swing_20260808/results_*.csv` 的 79 個 as_of，**不重呼叫 `sample_replay_dates`**（交易日數變動會推移抽樣格點，新舊只剩 26 個相同）。<br>**成本**：5 模式 × 79 日 ≈ 20~30 小時，分模式夜跑（run.py 已支援續跑 + 模式參數化）。<br>**產出**：`logs/pit_fivemode_<date>/` — 五模式 20 日超額 + t 值 + regime 分層、**growth 首次裁決**、#21e 全面改寫。<br>**同質期定義同步更新**：補完後 fundamental 維度全期都有資料，2020–2024 與 2025+ 的差異只剩 `announcement`／`broker_trade`／`securities_lending` |
| 27 | **估值回補重打假日 + 早退路徑無節流（P2，可獨立先做）** | 📋 **待實作**。`backfill_valuation_history` 上市段的 `pending_days` 只濾 `d.weekday() < 5`，而續跑判定是「當日估值檔數 ≥ `BACKFILL_MIN_VALUATION_STOCKS`(800)」——假日永遠達不到，故**每次執行都重打同一批假日**且永遠列為待補（實測 2020–2023 有 69 天，資料其實零缺口；ETA 因此失真 5 倍）。<br>**⚠ 直覺修法無效**：`_TWSE_HOLIDAYS` 只有 2025/2026/2027，其他年份 `is_twse_holiday` 回 False → `is_trading_day` 退化成「只判週末」，2020–2024 照打不誤。**改用 DB 判定交易日**（該日 4 碼普通股 `daily_price` ≥ `BACKFILL_MIN_COMMON_STOCKS`），與 `backfill_market_history` 同源。<br>**第二半**：`fetch_twse_valuation_all` 的 `time.sleep(_REQUEST_DELAY)` 在成功路徑（`return df` 之前），**4 條提前返回路徑**（請求失敗／`stat != OK`／無資料列）完全不節流 → 69 個假日以約 1.7 req/s 連發，違反 §2 的 TWSE 3 秒/次。修法：移入 `try/finally`。<br>**驗收**：假日不進 `pending_days`（monkeypatch DB 交易日）／早退路徑仍節流（monkeypatch `time.sleep` 計數） |

---

## 7. Medium Priority TODO（P1 — 3 到 9 個月）

> 對應 Phase A 末–B（M4–M10）。

| # | 項目 | 工期 | 內容 / 價值 |
|---|------|:---:|------------|
| 1 | **B1 Point-in-Time 研究環境** 🟡**2026-08-01 完成 ③，①②④ 未開始** | 4–6 週（已用 ~1 天） | 全 roadmap 最大單一效益。①全市場歷史回補至 2020（含下市股）；②DailyFeature 全歷史化；③scanner 注入 `as_of` + offline mode；④PIT 回測 CLI。解鎖：跨 regime 驗證（R2）、scanner 改動當日見真章。<br>**✅ ③ 已完成**（分支 `b1-pit-as-of-injection`）：<br>　• `MarketScanner.run(as_of=...)` 為唯一注入點；引擎層一律改用 `self._as_of()`（12 處 getattr 模式 + 8 處子類裸 `date.today()` 收斂）<br>　• **所有查詢加時間上界**（24 處），日頻表用 `<= as_of`<br>　• **新增 `src/data/pit.py`——公布時滯建模**。這是本項最關鍵的發現：`MonthlyRevenue.date` / `FinancialStatement.date` 存的是**期間**不是公布日，schema 無任何公布日欄位。若只用 `date <= as_of` 過濾，2026-03-05 重放會看到依法 3/10 才公布的 2 月營收——look-ahead 且方向恆偏樂觀。改以證交法 §36 法定期限保守建模（月營收次月 10 日、Q1–Q3 季後 45 日、年報次年 3/31）<br>　• **offline mode**：`as_of` 為歷史日時自動停用 Stage 0.5／2.5 所有外部補抓（否則抓回「今天」的資料污染歷史情境並使重放不可複現），`_maybe_sync_valuation` 另有縱深防禦<br>　• shared in-memory 路徑套用**與 DB 路徑相同**的上界——重構過程中 `test_shared_market_load` 實際攔下一次兩路徑漂移<br>**驗收**：`tests/test_pit.py` 23 測試，含靜態守門測試（引擎層禁止不可注入的 `date.today()`，違者 CI 紅）；停用時滯/上界/offline 可複現 4 個失敗。全測試 2,727 passed<br>**⚠ ① 原本只完成一半（2026-08-04 跨模式重放查明）**：價量已補（`daily_price` 156 萬筆 / `daily_feature` 339 萬筆，2024-01 起全市場含下市股），**但基本面四表完全沒補**——`stock_valuation` 僅 **2026-01-26 起**、`monthly_revenue` 2024 年只有 **5 支**股票、`dividend` 全表 367 筆、`financial_statement` 全表 365 筆。後果：value/dividend/growth 的 PIT 重放**不可用**。**✅ `stock_valuation` 已於 2026-08-05 補齊**（§6.5 #20，上市 TWSE 每日端點 + 上櫃 FinMind 逐股）；**`monthly_revenue` / `financial_statement` 仍未補 → growth 的歷史重放仍不可用**（📋 2026-08-15 已規劃為 **§6.6 #23~#27**：月營收走 MOPS 免費全市場、財報走 FinMind 逐股 ~10 小時，補齊後 #21e 補成五模式）；~~②DailyFeature 全歷史化~~ ✅**2026-08-04 完成**（518 日 / 339 萬筆 / 9 分 16 秒，純 CPU）；~~④PIT 回測 CLI~~ ✅**2026-08-04 完成**（`pit-replay` 子命令 + `discovery/pit_replay.py`）——**過程中補掉一個 B1③ 漏掉的洞**：`MarketRegimeDetector.detect()` 原本三個查詢（TAIEX/TW_VIX/US_VIX）完全無時間上界、`_compute_breadth()` 直接取今日 DailyFeature，且會寫入 `RegimeStateLog`。regime 驅動權重/門檻/模式封鎖，用今日 regime 重放毫無意義。現已 PIT 化並改為**唯讀**（`state_advanced=False`、`reason=pit_replay_readonly`）；`_get_universe_ids` 亦在 offline 時不寫 `universe_stat_log`。此洞之所以漏掉，是因為靜態守門當時只掃 `src/discovery/scanner`——現已擴及 `src/regime`。**故目前 `as_of` 只能重放到 DiscoveryRecord/DailyPrice 既有覆蓋範圍（個股 2025-01 起）**，跨 regime 驗證（R2）仍未解鎖。<br>**後續可選精確化**：為 `MonthlyRevenue`/`FinancialStatement` 補真實公布日欄位並回填，取代法定期限的保守近似 |
| 2 | **B2 全候選池因子落庫** ✅**2026-08-01 落庫層完成**（分支 `b2-candidate-factor-log`） | 1 週 | **已完成**：新表 `CandidateFactorLog`（`candidate_factor_log`），擷取點＝**軟加成後、硬風控前**，保有被硬閘門剔除者；`selected` 標記是否進入最終 top-N，`pool_rank` 記全池名次；A5 provenance（git_commit/settings_hash）獨立落庫，不 join `DiscoveryRecord`（模式落庫 0 筆時無列可 join）。**rankings 為空時仍落庫**——這正是 M2 自鎖的資料面成因。live 首跑（value）樣本量 **5 → 150（30×）**。`load_candidate_factor_records()` 提供 IC 就緒的取數介面，`selected_only=True` 可重現舊截斷樣本以量化偏差。<br>**附帶需求已解決，但真因與原記載不同**：原寫「`valuation`／`dividend` 維度不落庫」，實測為 **`_post_score()` 在 composite 之後把 `technical_score` 覆寫為估值/殖利率分數（顯示別名）**——導致 IC 管線把估值維度標記成 technical、被加權的 `valuation` 鍵找不到 IC（恆 N/A）、且**真實技術分在落庫前被丟棄**。實證：2026-08-01 value 2615 的 `discovery_record.technical_score`=0.7643 恰等於新表 `valuation_score`，而真實技術分為 0.5966。修法＝覆寫前保留 `technical_score_raw`，新表以真實語意落六維度（詳報告附錄 D）。<br>**⚠ 尚未完成（屬 B11）**：IC 體系**尚未**改用本表取數。原因是本表**無法回補**（歷史候選池從未落庫），資料自 2026-08-01 起累積，需先滿足 §3 原則 6 的「跨 ≥3 掃描週」才可切換。<br>**驗收**：新增 `tests/test_candidate_pool.py` 16 測試；停用「取 raw technical」與「無入選仍落庫」兩項可複現 2 個失敗。全測試 2,704 passed |
| 3 | **B11 IC 治理改革** 🔺**2026-07-30 升為 P1 最優先**；🔺**2026-07-31 擴大範圍** | 1 週 | 樣本門檻 ≥100 且跨 ≥3 掃描週；「自動停用模式」降級為告警+人工確認；以 B2 資料重建 IC。拆除噪音驅動的自動開關（歷史多次誤殺 swing/value/dividend 的根因）。**升順位理由（mg5_20d 二審）**：現行 M2 停用的是**掃描**而非**下單**（`morning_cmd.py:1201` → Step 9 skip），停用後不再產生 `discovery_record` → IC 無從重算 → **模式無法自證恢復，形成自鎖**。實測：momentum 6/16 停用後中斷 29 個交易日（至 7/29 才有 7 筆部分掃描）、growth 7/16 起停用至今。後果＝雙引擎組合 mg5_20d 訊號歸零、forward 驗證無樣本可取。**修法方向：停用時照常掃描並落庫，僅在 rotation 層阻擋新買入**，使 IC 可續算、模式具自動恢復路徑。此項未落地前，mom_growth／momentum 系組合的任何 forward 裁決都不成案。**影響範圍＝全系統**（2026-07-30 盤點，報告 §6）：mg5_20d pause 後僅存的 active 組合 mom5_10d／mom3_20d 皆為 momentum 模式，7/31 起雙雙 100% 現金、momentum 7/29 排名 8/4 逾期；momentum 過去 30 個交易日僅掃描 1 天（7/29 放行、7/30 又停用），M2 呈「停用→樣本歸零→判 insufficient→放行一天→再停用」振盪。後果＝**crisis 解除後全系統無可用訊號重新進場**。<br>**🔺2026-07-31 discover 審計修正與擴大**（`logs/audit_discover_20260731/REPORT.md`）：<br>**(a) 自鎖機制比原描述嚴重**——不只是「IC 無從重算」。`compute_rolling_ic`（`_functions.py:2067`）的窗口錨定在**推薦記錄的日期範圍**而非今天，`morning_cmd.py:724` 又取 `factor_df["ic"].iloc[-1]`；模式停掃後 `max_date` 凍結 → 窗口凍結 → **系統每天重讀同一份過期 IC 執法**。實測：7/31 停用 momentum 所用的 IC＝**−0.1109，來自 `window_end=2026-06-27`、n=40、距今 34 天**，且 7/30 與 7/31 取到完全相同的值（＝同一凍結值重讀兩次，非兩次獨立判斷）。重現腳本 `logs/audit_discover_20260731/repro_rolling_ic.py`。<br>**(b) 「振盪」真因不是 IC 回升**——7/29 放行是因當下記錄僅到 6/16，`min_date+14=6/19 > max_date+1=6/17` → while 迴圈零次 → `rolling_df` 空 → fail-open。**故此開關的實際判準是「rolling IC 算不算得出來」，與因子有效性無關。**<br>**(c) 與 §3 原則 6 的落差已量化**：原則要求 n≥100 且跨 ≥3 掃描週，實際跑在 **n=40、單一窗口、34 天前**——原則已寫入文件但程式無一處執行。<br>**(d) 範圍擴大至 E2b/E2c 兩個同源開關**：①`_apply_ic_weight_adjustment` 的歸一化把被打壓維度釋出的權重**全數塞給從未量測的維度**（2026-07-31 value：`fundamental 0.550→0.313` IC −0.1393，而 `valuation 0.150→0.338` **IC=N/A**，2.25×）——淨效果＝IC 治理越積極，決策權越集中到唯一沒被檢驗的維度；且依據本身不穩（value fundamental IC 四個交易日 `+0.155→+0.105→+0.034→−0.139`，擺幅全在 n≈100 的 SE≈0.10 雜訊帶內）。②`compute_ic_aware_score_transform(ic_threshold_weak=0.02, min_samples=50)` 在 SE≈0.10–0.14 下等同**隨機翻轉維度方向**。③`news_score` 有 **63–81% 恰為填補值 0.5**（`fillna(0.5)`）卻佔權重 0.20 且每日參與 IC 與調權——對七成是常數的變數算秩相關無意義。**修法須含**：覆蓋率不足的維度不得參與自動調權；未落庫維度不得被動吸收權重；翻轉門檻改為以 SE 為基準的顯著性判定。**2026-08-01 進度**：E2b/E2c 已整體凍結（§6.4 #17，只記錄不生效），屬止血非根治——上述三項仍是**解凍的前提條件** |
| 3b | **N2 Scanner 統一 `run()`** 🔺**2026-07-31 由 P2 升 P1**；✅**2026-08-01 結構部分完成** | 1 週 | ✅ 分支 `n2-unify-scanner-run`。**已完成**：`run()` 收斂為 `MarketScanner` 單一實作 + 宣告式 `StageConfig`（22 個階段旗標）；三個覆寫刪除，模式差異改以 `_STAGES` 宣告 + 4 個 hook（`_prepare_before_load` / `_after_market_data_loaded` / `_sync_candidate_valuation` / `_reload_candidate_valuation`）表達；`_load_revenue_data` 與 `slice_revenue_raw` 的推導邏輯抽出 `pivot_revenue_rows()` 單一實作（原為逐字相同的兩份）。觀測階段（audit_trail / sub_factor / factor IC 日誌）補齊至五模式。淨 −81 行，三個 scanner 各減約 100 行。<br>**⚠ 刻意未完成：閘門政策**。使用者裁定「宣告式統一、**先保留現行行為**」——`StageConfig` 的多數 False 是**現況存檔而非設計主張**，跨模式此刻**仍不可比**。開啟閘門的實測一階影響（2026-07 起 886 筆）：**3.7 分數門檻 → 保留 72.8%（但 crisis 日 dividend 20→1、value 20→3，風險最高）**、4.1 產業分散 → ~~94.1%~~ **此數字已於 2026-08-01 更正：筆數零損失、名單換約 15%**（見下）、4.2 回撤縮表 → 100%（value/dividend 本就在 `_DEFENSIVE_MODES` 豁免）。**建議順序**：先開 4.2 → 4.1（低風險），3.7 須先解決 crisis 門檻與防禦型模式的關係再議。軟加成類需實跑 scanner 才能量化。<br>**✅ 4.2 已於 2026-08-01 開啟**（分支 `enable-stage-42-drawdown`）。**過程中發現 4.2 原本是死的**：`_compute_drawdown_adjusted_top_n` 從 `df_price` 撈 TAIEX，但 TAIEX 不在任何 scanner 的 universe，而 momentum/swing/value/dividend 的 `_load_market_data` 都以 `stock_id.in_(universe_ids)` 過濾 → `taiex.empty` 恆成立 → **這道風控自 UniverseFilter 上線以來從未觸發過**（growth 因全市場載入而有 TAIEX，但它沒開這階段）。已改為 df_price 缺 TAIEX 時直接查 DB（含 PIT 上界）。**實測影響**：2026-08-01 TAIEX 20 日回撤 −7.40% 未達 −10% 門檻，五模式 top_n 均不變；回撤惡化至 −12% 時 momentum/swing/growth 由 20→10、−20% 時 →6，value/dividend 因 `_DEFENSIVE_MODES` 恆不受影響。2026-05 以來 43 個交易日中有 5 天會觸發。<br>**🔍 4.1 有效性查核（2026-08-01，使用者裁定暫不開啟）**：與 4.2 相反，**4.1 確實在運作**。自然實驗證據——開啟 4.1 的 momentum/swing 在 37 個掃描日中單一產業最大 5 檔、**0 次超過上限**；未開啟的 value/dividend/growth 經常突破（value 最高 11 檔、41 天中 33 次超限）。`industry_category` 覆蓋率 100%，無「全部歸未分類」的塌陷風險。<br>**原 94.1% 估算有誤**：未計入遞補。4.1 的 pool 深度為 `top_n × 2`，被產業上限擠掉的位置會由下一順位補上。以 B2 落庫的 value 完整候選池（150 檔）實測：輸出 **20 → 20 筆（零損失）**，航運業 7→5、金融保險 6→5，被擠掉 3 檔／遞補 3 檔，**名單重疊率 85%**。正確描述是「筆數不變、集中度下降、15% 名單換人」。<br>**未開啟的理由**（使用者裁定）：4.1 是刻意以分數換分散度——被換掉的標的 composite 高於遞補者。若 value 模式的 edge 來自產業集中（如押對航運循環），4.1 會削弱它。此為策略立場問題，資料無法裁決。<br>**驗收**：`tests/test_scanner_pipeline_parity.py` 22 測試——五模式 × 三 regime 的**選股階段序列**與重構前基準逐一相同（基準於重構前擷取），另加「無 scanner 覆寫 `run()`」契約測試。全測試 2,688 passed。<br>原始定性（保留）：這不是重複程式碼的整潔問題，是**量測有效性問題**。`_value.py:168`／`_growth.py:57`／`_dividend.py:173` 各自覆寫 `run()`（複製貼上版），相對 `BaseScanner.run()` 跳過：**3.7 動態分數門檻／4.1 產業分散化／4.2 回撤縮表**／3.2 前次重疊／3.3c 同業基本面／3.5c 動量衰減／3.5d-g 籌碼系／3.5h 負面消息閘門／3.5e 多時框／3.6 量價背離／4.3 籌碼降級稽核／`ScanAuditTrail`／`sub_factor_df`／`ic_actions`。**直接證據**：dividend 2026-07-31 落庫最低分 **0.536**，當日 regime=crisis 門檻應為 0.60。**後果**：「value/dividend 天天穩定 20 筆、momentum/swing 常態 0 筆」是**閘門覆蓋率差異的產物，不是模式強弱**——此假象正在污染跨模式 IC 比較、`cross-mode-corr`、`mom_growth` 雙引擎的模式選擇、以及「暫停 swing、保留 value/dividend」之裁決。**驗收**：五 scanner 走同一條漏斗，模式差異只留在 `_coarse_filter`／`_compute_*_scores`／`_compute_extra_scores`；順帶合併 `_load_revenue_data`／`slice_revenue_raw` 兩份同邏輯。詳 `logs/audit_discover_20260731/REPORT.md` §5 |
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

**⚠ 2026-08-04 研究議程前置條件再更新**（跨模式 PIT 重放，`logs/pit_crossmode_20260804/REPORT.md`）：**R2 尚未解鎖**。B1④ 工具已可用、價量已補齊，但 ①`stock_valuation` 缺 2026-01-26 以前的全部資料（§6.5 #20）→ 三個模式不可評價；②現有 30 個基準日的 regime 分布為 bull 17／sideways 8／crisis 3／**bear 僅 2**——**2024–2026 沒有真正的空頭市場**，跨 regime 穩健性無從驗證。R2 與 R7 都需 **2020–2023 回補**（含 2020/03 COVID 崩跌與 2022 全年空頭，~9 小時）。<br>**同時記錄一項不成案的假設**：初步分析顯示 momentum 在 bear 的 20 日報酬 −9.83%／勝率 26%／n=34／t=−4.01，看似足以支持「bear 納入 `REGIME_MODE_BLOCK`」——但那 34 筆只來自 **2 個基準日**（2025-03-07、2025-05-07，同一段關稅衝擊回檔），有效樣本 ≈2，t 值是偽複製膨脹，違反 §3 原則 6 與 §6.4 #16 的窗口數 ≥3 閘門。**已裁決不採取行動**，留待 2020–2023 回補後檢驗。<br>**✅ 2026-08-08 更新**：2020–2023 全市場回補已完成（價量 966 交易日／DailyFeature 974 日 5,048,286 筆／估值 974 日、每年 1,535~1,587 檔），**R2 前置條件解除**。79 個基準日的 regime 分布改善為 bull 32／sideways 27／**bear 16**／crisis 4。上述 bear 假設經 momentum 全期重放**正面否定**（同質期 bear 14 個基準日超額 +0.39%、勝率 64%，詳 §6.5 #21）。<br>**⚠ R2 仍有一項未解的前置**：`announcement`（`news_score` 權重 0.20）全表僅 2026 年有資料、`broker_trade`／`securities_lending` 僅 2025–2026、`monthly_revenue` 僅 2025 起成規模——**跨 2025 的期間比較同時混著市場變化與評分輸入變化**。跨 regime 驗證在 2020–2024 的同質區間內成立，橫跨 2025 的結論則不成案。

**⚠ 2026-07-31 研究議程前置條件更新**：discover 審計（`logs/audit_discover_20260731/REPORT.md` §5）證實五個 scanner 非同一條 pipeline，**任何模式級比較在 N2（§7 #3b）+ B2 落地前皆不成案**。受影響：R3（全池 IC 重建，本就依賴 B2）、以及所有既成的模式級裁決——含「暫停 swing、保留 value/dividend」。另 `data/baseline_metrics.json` 的模式級門檻在 N2 修復後需重新評估（與 §10 renewal 訊號污染的重錨需求**合併處理**，勿分兩次重錨）。

---

## 10. Known Technical Debt（已知技術債登記簿）

### 確認的 bug / 死碼（已驗證）
| 項目 | 位置 | 狀態 |
|------|------|------|
| `Announcement.title` 欄位不存在 | `pipeline.py:1802` | ✅ 2026-07-05 已修復（`subject` + smoke test） |
| `backup_db()` 零呼叫者 | `database.py` | ✅ 2026-07-05 已接上 morning-routine Step 18 |
| ★ **`monthly_revenue` 兩套日期語意**：FinMind 寫「次月 1 日」、MOPS 寫「當月月底」，unique key `(stock_id, date)` 擋不住 → 實測 **2,488 組 `(stock_id, year, month)` 重複**；`pivot_revenue_rows` 取 `head(months)` 故 growth 的 4 個月窗口**實際只拿到 2 個月**，`prev_yoy_growth` 可能是同月的另一份來源 | `fetcher.py:fetch_monthly_revenue` vs `mops_fetcher.py:_parse_revenue_html` | 📋 §6.6 #23（**live 現在就在發生**） |
| ★ **`sync_mops_revenue` 續跑閘門看錯軸**（與 §6.5 #22 同型）：門檻「該月列數 ≥ 500」混算候選池逐股補抓的列，月初跑一次 + 候選池累積即越過門檻 → 該月**此後永不重抓**。實測 2026-02 的 MOPS 列僅 **1 筆**、2026-06 僅 498 筆 | `pipeline.py:sync_mops_revenue` | 📋 §6.6 #23（**live 每月都在發生**，growth 因此長期吃半套營收） |
| **FinMind 逐股營收 96% `yoy_growth` 為 NULL**：`lookback_days=180` 只取 6 列，而 YoY 需 `len(df) > 12`；growth 粗篩 `yoy_growth.notna() & > 10` 使這些列在粗篩即蒸發 | `pipeline.py:sync_revenue_for_stocks` | 📋 §6.6 #23（改 430 天） |
| **估值回補每次重打假日 + 早退路徑無節流**：`pending_days` 只濾週末（2020–2023 有 69 天永遠列為待補）；`time.sleep` 只在成功路徑，4 條提前返回路徑以 ~1.7 req/s 連發，違反 TWSE 3 秒/次 | `pipeline.py:backfill_valuation_history` / `twse_fetcher.py:fetch_twse_valuation_all` | 📋 §6.6 #27（**不可用 `calendar.is_trading_day` 修**——假日表只有 2025~2027，其餘年份退化成只判週末） |
| `_compute_backtest_metrics` 死碼 | `manager.py` 尾端 | 待 sweep |
| `fetch_taiwan_vix` 永遠回空（dataset 已亡） | `fetcher.py` | 待替代（期交所 VIX）或移除 |
| `_collect_settings_diffs` 永遠回空（settings.yaml 不在 git） | `strategy_events.py` | ✅2026-07-06 A5 拆檔後已復活（改追 quant_params.yaml） |
| SBL `sbl_change` 等三欄恆 NULL（API 改版） | schema + twse_fetcher | 記錄在案，因子已降級 |
| `DailyReportEngine._compute_ml_score` 引用不存在的 `_last_proba` | `report/engine.py` | 永遠走 fallback，低優先 |
| Stage 4.2 回撤縮表因 TAIEX 不在 universe 而恆不觸發 | `_base._compute_drawdown_adjusted_top_n` | ✅ 2026-08-01 已修（改查 DB + PIT 上界），並同步對五模式開啟 |
| `ScoreThresholdConfig` 零消費者（scanner 讀死常數，`quant_params.yaml` 門檻靜默無效） | `config.py:84-90` vs `_functions.py:1196` | 2026-07-31 發現 → §6.4 #18 |
| `MarketRegimeDetector` 同日快取 per-instance，從未命中（每日 4–15 次重跑 hysteresis） | `detector.py:409` | ✅ 2026-08-01 已修（§6.4 #15；改以 `data_date` 為冪等鍵 + Step 8e 同步後重解） |
| `level="insufficient"` 混用四種語意，log 印「樣本不足（n=260，需 ≥20）」 | `morning_cmd.py:669/693/714/720` | 2026-07-31 發現 → §6.4 #18 |
| ★ **粗篩對缺失資料 fail-open**：估值表為空時**跳過定義性閘門而非收斂**，模式靜默退化成流動性篩選且無 log 警示 | `_value.py:_coarse_filter` 的 `else` 分支（**僅此一處**——`_dividend.py`／`_growth.py` 原本即 fail-closed，初判「同構」有誤） | ✅ 2026-08-05 已修（§6.5 #19，改走共用守門 `_require_coarse_data`）。歷史教訓：若非實查資料表，2026-08-04 的審計會得出「value 是最強模式」的錯誤結論——**漏斗統計察覺不到**，兩次重放的 `769→150→20` 完全相同，但選股 9/20 換人 |

| ★ **DailyFeature 回補的續跑判定看「該日有無特徵列」** — 與 B1① 價量回補當初的缺陷同型（`ee128d0` 已把價量改為看普通股檔數，特徵沒跟著改） | `pipeline.py:backfill_daily_features` 的 `done_dates` | ✅ 2026-08-09 已修（改為看「特徵列數 ≥ 價量列數 × `FEATURE_BACKFILL_MIN_COVERAGE_RATIO`(0.95)」）。**病灶**：TPEX 同步逾時 → 該日只有上市價量 → 特徵以半套資料算完寫入 → 日期被永久標記已補 → 事後補齊上櫃價量後**永不重算**。實測 **11 天中招**（2022~2023 歷史回補 8 天 + **live 3 天**：2026-05-27／06-22／06-24），`daily_price` 4,400~7,300 列但 `daily_feature` 僅 1,147~1,362 列。**後果不限於重放**：缺特徵列的股票被 `_stage2_liquidity_filter` **整批排除於 universe 之外**——`avg5_map` 只由 `df_feature` 既有列建立、回傳的 universe 又只取自 `avg5_map.index`，故那 11 天的候選池少了約四成（上櫃股全數消失），且無任何 log。<br>**⚠ 2026-08-09 更正**：初判寫的是「踩破 `_FEATURE_COVERAGE_MIN`(0.3) → 退回 DailyPrice fallback」，**有誤**——0.185 是含權證的全表比例，而該門檻的分母是 `stage1_ids`（普通股），實際 973/1750 ≈ **0.556 高於 0.3**，不會 fallback。真正的機制是走 DailyFeature 路徑並靜默丟掉沒有特徵列的股票，方向比誤判時所想的更嚴重。連帶更正 `REPLAY_MIN_FEATURE_RATIO`（§6.5 #21b）：原值 0.3 的理由正是這個錯誤認知，已改為 0.95 與回補判定同源。<br>門檻取 0.95 因實測分布完全雙峰（正常日 1,591 天全在 0.9989~1.0、中招日 11 天全在 0.18~0.26，中間無任何一天），且誤判方向安全（判成未補只多花 CPU，upsert 冪等）。<br>**對已完成研究的影響已複查**：§6.5 #21e 的四模式全期重放正是在修復前跑的，316 個（模式 × 基準日）組合中僅 2026-06-22 一天受影響，重跑後結論不變（詳 #21e） |

### 結構債
- **四套交易模擬器**（BacktestEngine / PortfolioBacktestEngine / walk_forward fold / rotation backtest），成本行為已漂移（walk_forward 無停損無動態滑價）→ B8。
- **live/backtest overlay 組裝兩份**（manager.update vs backtest 各 400 行）→ B7。
- `compute_rotation_actions` 25 參數 400 行；`manager.py` 仍 2,156 行 → 隨 B7 拆。
- **5 scanner 非同一條 pipeline**（2026-07-31 定性；✅2026-08-01 **結構已統一**，閘門政策仍待決）：`value`/`growth`/`dividend` 覆寫 `run()` 並跳過分數門檻/產業分散/回撤縮表/負面消息閘門/audit_trail/sub_factor 落庫，momentum/swing 走 base `run()` 吃滿 6 道硬風控。**這不是 ~2,800 行重複的整潔問題，是跨模式不可比**——所有模式級比較（IC、cross-mode-corr、mode 選擇、模式暫停裁決）皆建立在此不對等基礎上 → **N2 升 P1**（§7 #3b）。**2026-08-01 進度**：`run()` 已收斂為單一實作 + 宣告式 `StageConfig`，`_load_revenue_data`／`slice_revenue_raw` 的推導已抽出 `pivot_revenue_rows()` 共用；**但差異本身尚未消除**——只是從「藏在 3 份複製貼上的流程碼」變成「一張可審查的旗標表」。跨模式可比性要等閘門逐項開啟（實測影響見 §7 #3b），在那之前模式級比較仍不成案。
- 魔法數散落（加成 ±3%/±5%、cap 8%、dampen 0.85、news p15 等 inline 預設值）→ 隨 A5 參數檔收斂。
- `settings` import-time 全域單例 → 多環境需求出現時再還。
- 行事曆手工維護（2027 為暫定推算，每年 12 月須校對）→ N5。
- 逐股 Python 迴圈效能地雷（150 候選安全，1,500 不安全）→ Low #9。
- conftest session-scope engine + get_session monkeypatch 的污染陷阱（曾污染 dev DB）→ Low #10。
- Fallback 疊加路徑（資料延遲 + 模式停用 + 前日排名 + 5 日前價格）無測試覆蓋 → 隨 B7 的 parity 套件補。
- **renewal 訊號污染（2026-07-30 發現）**：續持依賴當前排名（`rotation.py:841` `if allow_renewal and sid in ranked_ids`），P0 #13 修復前的 stale fallback 使部位靠過期排名**無限續持**——mom5_10d 1303（hold=10，實持 6/16–7/21、+62.78%）、mom3_20d 2890/5871（hold=20，實持 6/09–7/21）等多筆遠超名目天數，並於 7/21 修復後首次 update 集體到期。含意：①mom5_10d／mom3_20d 的 +12%／+11% 生涯績效部分係 bug 行為所得，與 post-fix 行為**不同質**；②`data/baseline_metrics.json`（2026-07-08 A4 重錨）凍結區間涵蓋污染期，其 sharpe/win/alpha 門檻對 post-fix 行為未必適用 → **B11 落地後應重評是否再次重錨**；③兩組合正式重審須待污染量化後辦理。

### 統計/方法債
- IC 建立在 top-N 截斷樣本、n≈20–30，驅動自動停用/翻轉 → B2+B11。**2026-07-31 量化**：漏斗 `1,857 →(流動性) 659 →(粗篩) 150 →(落庫) 20`，IC 只算在 top-20 上（range restriction）。**2026-08-01 進度**：B2 落庫層已上線（`candidate_factor_log`，樣本量 30×），但**IC 仍在舊的截斷樣本上計算**——新表無法回補、須累積 ≥3 掃描週後才能由 B11 切換取數來源。在那之前所有 IC 數字仍帶截斷偏差。
- ~~**M2 自動停用自鎖迴路**~~ ✅**2026-08-01 已解除**（§6.4 #16）。歷史記錄：停用作用在掃描層 → 無新 `discovery_record` → IC 無從重算 → 模式無法自證恢復；momentum 中斷 29 個交易日、growth 中斷至今。真正機制是 `compute_rolling_ic` 窗口錨在記錄日期範圍、`iloc[-1]` 取最後一個有資料的窗口 → 停掃後**每天重讀同一份凍結 IC 執法**（7/31 用的是 `window_end=2026-06-27`、n=40、34 天前，7/30 與 7/31 完全相同）；「放行」則是窗口生不出來時的 fail-open。**開關的實際判準是「rolling IC 算不算得出來」，與因子有效性無關。** 現由 `src/discovery/ic_governance.py` 的時效閘門阻擋過期執法，且停用改為只作用在 rotation 層（掃描照常）。
- **§3 原則 6 的程式化執行**：✅**M2 與 scanner 門檻加成已於 2026-08-01 落地**（n≥100、窗口 ≥3、時效上限，見 §6.4 #16）；E2b/E2c ✅**已於 2026-08-01 凍結**（§6.4 #17，只記錄不生效）——注意這是**繞過**而非解決：兩者的樣本門檻仍未實作，解凍前必須先補上以標準誤為基準的顯著性判定（B11 完整版）。**仍在運作且無樣本門檻者**：`_compute_win_rate_adjustment` 的勝率回饋 +0.05（見下條）。
- **IC 調權把權重搬給未量測維度**（2026-07-31）：`valuation`／`dividend` 維度不落庫 → 恆 `IC=N/A` → `_apply_ic_weight_adjustment` 的歸一化使其被動吸收被打壓維度釋出的權重（value 7/31：0.150→0.338，2.25×）。淨效果＝IC 治理越積極，決策權越集中到唯一沒被檢驗的維度。**2026-08-01 已凍結止血**（§6.4 #17，不生效但仍記錄）。**真因已於 B2 查明並修正**：非「維度不落庫」，而是 `_post_score()` 用顯示別名覆寫 `technical_score`，使 IC 與權重鍵對不上（詳 §7 #2 與報告附錄 D）；`candidate_factor_log` 已以真實語意落六維度。**但解凍仍須等 B11** ——IC 取數來源尚未切換到新表。
- **分數鑑別力不足**（2026-07-31）：`news_score` 63–81% 恰為 `fillna(0.5)` 填補值卻佔權重 0.20 且每日參與 IC 與調權；momentum 的 `fundamental_score` 73.9%／`technical_score` 45.8% 為填補值。單日 top-20 內 composite 全距僅 0.10–0.15，與 `signal_jaccard_mean` 0.17–0.24（growth/swing）相互印證 → B2 後重估。
- **門檻疊加正回饋**（2026-07-31）：`_apply_score_threshold` = regime 基礎 + 勝率回饋 +0.05 + IC 衰退 +0.05，三個加項全由「近期表現差」驅動，構成「表現差→門檻升→樣本少→統計更不穩→更易判定為差」閉環；`strategy_decay_log` 顯示五個模式全部 `is_decaying=1`，此加壓為常態。實測 7/28 swing 門檻 0.70 → 剔除 88 支 → 落庫 0 筆。**部分緩解**：IC 衰退加成已於 2026-08-01 納入三道閘門（§6.4 #16），過期/小樣本 IC 不再推高門檻；**勝率回饋 +0.05 仍無樣本門檻**，閉環未完全拆除 → 留待 B11 完整版。
- ★ **交叉相關造成的樣本膨脹**（2026-08-04）：同一掃描日選出的 N 檔股票**不是 N 個獨立觀測**——共享當天市場衝擊。以「每筆選股」為單位算顯著性會嚴重高估：實測名目 n 膨脹 **16.7×（momentum）／18.7×（value）／20.0×（dividend）**，value 的 t 值由 +6.06 掉到 +2.92、momentum 由 −0.01 變 +0.31。**凡涉及「一籃子選股的前瞻報酬」的統計，觀測單位一律取「基準日的籃子平均」**（本專案的掃描間隔 ≈20 交易日，持有窗口幾乎不重疊，可視為獨立）。此偏差同樣影響 IC 的樣本數認定——現行 `evaluable_count` 計的是選股筆數而非獨立掃描日數，§3 原則 6 的「n≥100」實質寬鬆於字面。
- ★ **報酬未扣大盤即比較**（2026-08-04）：跨模式重放的原始報酬（value +3.40%／growth +6.86%／momentum −0.00%）扣除同期 TAIEX 後全部塌回零附近（−0.37%／+0.85%／−2.66%）。**先前所有「哪個模式比較好」的印象主要是 beta 排序**。任何模式級績效敘述須明示是否已扣基準。
- 全部閾值/權重為 in-sample 調參，跨 regime 未驗證 → R2/R3。**2026-08-04 補充**：現有可重放樣本的 regime 分布為 bull 17／sideways 8／crisis 3／bear 2，**空頭樣本實質為零**。
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

*v1.2（2026-08-01）：止血包三之 #15、#16、#17 落地，另完成 N2（scanner 統一 run()）與 B2（全候選池落庫）。異動摘要——§6.4 #15（regime 同日多次偵測）、#16（M2 凍結 IC 執法）、#17（凍結 E2b/E2c）標記完成並補上修法與 live 驗收；#18 的 `insufficient` 分支拆分隨 #16 一併完成（`ScoreThresholdConfig` 接線仍待辦）；§1 斷柱 4 的三項成因兩項劃除，現況改為「訊號產能已恢復、噪音開關已止血，模式級比較仍不可信」；§10 M2 自鎖迴路標記解除，並在「IC 調權搬權重」「§3 原則 6」兩條明示**凍結屬止血非根治、解凍前提條件為何**。**新增模組** `src/discovery/ic_governance.py`＝IC 可執法性的單一實作；**新增設定** `quant.ic_governance`（E2b/E2c 開關，預設凍結）；**新增資料表** `candidate_factor_log`（B2 全候選池，樣本量 30×）。**歸因更正**：§7 #2 與 §10 對「valuation/dividend 維度不落庫」的說明，實測真因為 `_post_score()` 以顯示別名覆寫 `technical_score`（報告附錄 D）。*
