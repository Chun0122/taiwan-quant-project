# CLAUDE.md

本檔案為 Claude Code 在此專案中的**決策指南**。僅包含影響行為與判斷的規則，詳細參考資料見 `docs/` 目錄。

---

## 1. Claude 行為準則（最高優先）

- 所有 UI 文字、註解、commit message 使用**繁體中文**
- 所有 Python 原始碼 UTF-8，開啟檔案務必 `encoding='utf-8'`

---

## 2. 開發與操作規則（不可違反）

| 項目 | 規則 |
|------|------|
| **DB 寫入** | `_upsert_batch()`，batch_size=80（SQLite 變數上限） |
| **API 速率** | FinMind 0.5 秒/次；TWSE/TPEX 3 秒/次 |
| **日期格式** | FinMind `YYYY-MM-DD`；TWSE `YYYYMMDD`；TPEX 民國曆 `YYY/MM/DD`（年 = 西元 - 1911） |
| **回測成本** | 手續費 0.1425%（A4 混合單：整張/零股單各計最低 20/1 元）、交易稅 0.3%（賣出）、滑價 0.05%（零股部分 +0.1% premium）；成本 SSOT = `rotation.trade_cost_amounts` |
| **Session** | `with get_session() as session:`；批次寫入 `sqlite_upsert().on_conflict_do_nothing()` |
| **常數** | 全系統共用常數集中於 `src/constants.py`，勿在各模組硬編碼 |
| **設定** | `config/quant_params.yaml`（量化參數，**進版控**）+ `config/secrets.yaml`（機密，gitignored）→ `src/config.py` deep-merge 載入（A5 拆檔；legacy `settings.yaml` 存在時照舊+警告）；機密勿寫入 quant_params |
| **資料來源優先序** | ①TWSE/TPEX 官方（免費，全市場）→ ②FinMind 批次（付費）→ ③FinMind 逐股（免費備援） |

### 提交前必執行

```bash
ruff check .    # Lint 檢查
ruff format .   # 格式化
```

---

## 3. 專案架構（精簡版）

### 概述

台股量化投資系統。CLI 驅動流水線：資料擷取（FinMind API + TWSE/TPEX + MOPS）→ SQLite → 技術指標 → 策略訊號 → 回測 → 報告/通知。

### 資料流程

```
FinMind API / TWSE+TPEX / MOPS ──→ Pipeline (ETL) ──→ SQLite DB
                                                    │
Strategy.load_data() ← 寬表（OHLCV + 指標合併）
         │
    generate_signals() → BacktestEngine.run() → BacktestResult → DB
                                                    │
                              DailyReportEngine / StrategyRankEngine → Discord 通知
```

### 模組地圖

**資料層**

| 模組 | 職責 |
|------|------|
| `data/fetcher.py` | FinMind API（逐股/批次/財報 EAV pivot）、US VIX（yfinance） |
| `data/twse_fetcher.py` | TWSE/TPEX 全市場免費資料、SBL（TWT96U）、DJ 分點（Big5 HTML）、TDCC |
| `data/pipeline.py` | ETL 調度 + DB 寫入、OHLCV 品質閘門（值域 + OHLC 一致性 `low≤close≤high`）、close-to-close 跳動哨兵（`_detect_price_jumps` WARN，門檻 `PRICE_JUMP_WARN_THRESHOLD`=11%）、DailyFeature 計算、Broker Bootstrap、`sync_dividends_for_stocks`（rotation 標的股利補抓，morning Step 11b） |
| `data/mops_fetcher.py` | MOPS 重大訊息 + 月營收、事件分類（7 類）、情緒分類 |
| `data/schema.py` | 30 張 ORM 表（`RotationActionLog`：每日輪動操作明細，dashboard `today_actions` 來源；`RotationPendingOrder`：live T+1 待成交意圖，A2；`CandidateFactorLog`：B2 全候選池因子快照，軟加成後/硬風控前擷取，解除 IC 截斷樣本偏差） |
| `data/validator.py` | 7 個品質檢查純函數 |
| `data/calendar.py` | TWSE 交易日行事曆（2025-2026）+ 臨時休市日 `_UNSCHEDULED_CLOSURES`（颱風假等；morning-routine 哨兵偵測「行事曆交易日但全市場無資料」後手動登錄） |
| `data/io.py` | CSV/Parquet 匯出匯入（欄位驗證 + upsert） |
| `data/pipeline.py` 之 `compute_feature_columns` | **DailyFeature 算式 SSOT**（B1②）：每日增量與歷史回補共用同一實作，兩邊漂移會使歷史特徵與今日特徵不同質、PIT 重放的 universe 失真 |
| `data/pipeline.py` 之 `backfill_daily_features` | B1② DailyFeature 歷史化：分批計算並**多讀 130 天暖身**確保 chunk 邊界 MA60 正確；rolling 皆後視窗故天然無 look-ahead |
| `data/pipeline.py` 之 `backfill_market_history` | B1① 歷史回補：以 TWSE/TPEX 每日全市場端點逐交易日補齊；**續跑判定看「當日普通股（4 碼）檔數 ≥ `BACKFILL_MIN_COMMON_STOCKS`」**——既不能看「有無資料」（2020~2024 每日皆有 6 檔 watchlist，會靜默跳過 5 年）也不能看總筆數（崩盤日權證無報價會偽陽性、權證多的半套日會偽陰性） |
| `data/pipeline.py` 之 `backfill_valuation_history` | §6.5 #20 估值回補：**上市走 TWSE `BWIBBU_d` 每日端點、上櫃走 FinMind 逐股**——TPEX 估值端點（`peratio_book/pera_result.php`）已下架（所有日期含當日皆 302 導向 `/errors`），新版 openapi 只回當日無歷史，故上櫃無官方來源。續跑判定：上市看當日檔數 ≥ `BACKFILL_MIN_VALUATION_STOCKS`（800，母體僅約 1,000 遠小於價量）、上櫃看該檔估值日數 ≥ 價量日數 × `VALUATION_COVERAGE_RATIO`（0.8） |
| `data/pit.py` | **PIT 資料可見性 SSOT**（B1）：月營收/季報無公布日欄位，以證交法 §36 法定期限建模（`revenue_visible_cutoff` / `financial_visible_cutoff` / `is_pit_replay`） |
| `data/retry.py` | `request_with_retry()` exponential backoff（429/5xx） |
| `data/migrate.py` | DB schema 遷移工具 |

**策略/回測層**

| 模組 | 職責 |
|------|------|
| `strategy/base.py` | 抽象 `Strategy`：`load_data()` / `generate_signals()` / 除權息調整 |
| `strategy/ml_strategy.py` | ML 策略（RF/XGBoost/Logistic）+ CV + Optuna + SHAP |
| `backtest/engine.py` | 交易模擬、T+1 訊號延遲、三因子動態滑價、流動性約束 |
| `backtest/metrics.py` | 10 指標 + Monte Carlo（Bootstrap 1000 次） |
| `backtest/attribution.py` | 五因子歸因（momentum/reversal/quality/size/liquidity） |
| `backtest/allocator.py` | risk_parity / mean_variance 配置 |
| `backtest/portfolio.py` | 多股票組合回測（4 種配置模式） |
| `backtest/walk_forward.py` | Walk-Forward 滾動驗證 |

**選股/Universe 層**

| 模組 | 職責 |
|------|------|
| `discovery/scanner/` | 五模式選股（Momentum/Swing/Value/Dividend/Growth）、四維度評分、Regime 動態權重；**單一 `run()`**（N2）＋ 宣告式 `StageConfig` |
| `discovery/universe.py` | Universe 三層漏斗（SQL→流動性→趨勢）+ Candidate Memory |
| `discovery/performance.py` | 推薦績效回測、策略衰減警告、訊號穩定性監控（`compute_signal_stability`：top-N 相鄰掃描日 Jaccard，落 `StrategyDecayLog.signal_jaccard_mean/pairs`） |
| `discovery/ablation.py` | 因子消融測試（維度級 + 子因子級 + 績效消融） |
| `discovery/cross_mode_corr.py` | 跨模式 score 相關性研究（per-date Spearman + 重疊統計，`cross-mode-corr` CLI） |
| `discovery/pit_replay.py` | **B1④ PIT 歷史重放**：`replay_scan(as_of)` / `compute_forward_returns` / `sample_replay_dates`；`pit-replay` CLI。**唯讀**——不寫任何 live 表。單次重放約 90 秒，範圍重放須抽樣 |
| `discovery/ic_governance.py` | **IC 可執法性 SSOT**（P0 #16）：`select_enforceable_ic()` 三道閘門（窗口時效 `holding+14` 天／窗口數 ≥3／最小樣本 ≥100）+ `ICVerdict`；決策 IC = 最近 3 窗平均。M2 停用、scanner 門檻加成、rotation 阻擋買入三處共用 |
| `discovery/strategy_events.py` | 策略調整事件抽取（git log + quant_params.yaml diff，供 dashboard 事件流） |
| `discovery/universe.py:log_universe_stats` | UniverseFilter 每次 scan 後落庫 `UniverseStatLog`（P1 任務 8，audit 時序對比用） |
| `regime/detector.py` | 市場狀態（bull/bear/sideways/crisis）、Hysteresis 狀態機（**冪等**：以 `data_date`＝最新 TAIEX 資料日為鍵，同一份資料重複呼叫不推進狀態、不寫 log） |
| `industry/analyzer.py` | 產業輪動、同業相對強度（±3%） |
| `industry/concept_analyzer.py` | 概念股輪動、Percentile Rank（±5% 加成） |
| `screener/` | 多因子篩選引擎（8 因子，watchlist 內掃描） |

**進出場/組合層**

| 模組 | 職責 |
|------|------|
| `entry_exit.py` | 共用純函數：ATR 止損止利、進場觸發、時機評估（Discover/Suggest/Watch 三系統共用） |
| `portfolio/rotation.py` | 輪動核心：換股 + 風控（Drawdown Guard/Portfolio Heat/Correlation/VaR） |
| `portfolio/manager.py` | RotationManager：每日更新（A2 T+1 兩段式：`decide`/`fill_pending`/`update` wrapper + `_build_decision_context` 共用組裝）/ Kill Switch / 歷史回測（`backtest()` 含研究旋鈕 `disable_stop_loss`/`stop_loss_widen`/`t1_execution`/`save_result`，僅回測用、live 不受影響）。A3 股利會計：live `fill_pending` 開頭與 backtest 迴圈頂端同構呼叫除息處理（現金入帳 + 停損調整，ActionLog `dividend` 型為冪等標記） |
| `portfolio/dividends.py` | 股利會計純函數（A3）：`load_dividend_events` / `dividend_adjustment_factor`（與 Strategy Layer 1 同式）/ `adjust_stop_loss_for_dividend` / `dividend_cash_for_position`；入帳時點=`Dividend.date`（除息日）；配股第一版僅調停損不調股數 |
| `portfolio/execution_core.py` | 成交模擬核心純函數（`simulate_buy`/`simulate_sell` + `BuyFill`/`SellFill`）：live 與 backtest 共用同一份金額算式（pnl/成本/淨回收/總支出），消除兩路徑 drift；A4 起金額由 `rotation.trade_cost_amounts`（混合單成本 SSOT）導出；股數定價/滑價/流動性/漲跌停留各 caller |
| `portfolio/rankings.py` | 排名解析（resolve_rankings / _resolve_composite_rankings / 進場理由 breakdown），manager.py 抽出。**Composite mode**（`constants.COMPOSITE_MODES` + `is_composite_mode`）：'all'（五模式）與 'mom_growth'（動量+成長雙引擎，2026-06-20 取代結構性失敗的 'all'）共用 avg-score + per_mode_max 配額 resolver |
| `portfolio/market_data.py` | 市場資料查詢（交易日曆 / 收盤價 / OHLCV / TAIEX / 0050 benchmark + `_get_benchmark_dividends_between` 股利窗口加總），manager.py 抽出 |
| `portfolio/metrics.py` | 純計算指標（compute_cost_metrics / compute_benchmark_alpha_fields——A3-4 起支援 0050 total return 加法還原，`div_since_prev/base` 參數），manager.py 抽出 |
| `portfolio/audit.py` | rotation-audit 純函數（trade stats / alpha delta / Jaccard 穩定性），`rotation-audit` CLI 用 |

**CLI/報告/視覺化層**

| 模組 | 職責 |
|------|------|
| `main.py` | CLI 調度器（argparse，39 子命令 + dispatch table） |
| `cli/*.py` | 各子命令實作（sync/discover/backtest/watch/rotation/anomaly/morning/export-dashboard 等） |
| `report/` | 每日報告 + Discord 格式化（2000 字元限制）+ AI 摘要（`claude-sonnet-4-6`） |
| `notification/line_notify.py` | Discord Webhook（檔名歷史遺留） |
| `visualization/` | Streamlit 儀表板（12 分頁）+ Plotly 圖表 |
| `scheduler/` | 排程（前景 / Windows Task Scheduler / macOS LaunchAgent） |
| `features/indicators.py` | SMA/RSI/MACD/BB/ADX EAV + 週線聚合 |
| `features/ta_compat.py` | `ta` 套件版本相容層（0.5.x `n=` vs ≥0.7 `window=` 自動偵測，`make_sma/rsi/macd/bollinger/adx` 工廠） |
| `features/ml_features.py` | ML 特徵矩陣 + SHAP 篩選 |
| `factors/registry.py` | **Factor SSOT**：所有因子（dimension/sub_factor/predicate/indicator）的 metadata 註冊表；`python main.py factor-list` 查詢；新增因子須同步註冊 |
| `config.py` | Pydantic 設定模型（`QuantConfig`：TradingCost/AtrMultiplier/ScoreThreshold/RiskBudget/RotationCost）；`RotationCostConfig.for_mode(mode)` per-mode 成本閘門覆蓋（閘門對 momentum/swing 效果相反，manager 依 portfolio.mode 解析） |

---

## 4. 核心設計模式（新增/修改時必須遵循）

| 模式 | 規則 |
|------|------|
| **策略註冊** | `STRATEGY_REGISTRY`（`src/strategy/__init__.py`）；9 策略；新策略繼承 `Strategy`，實作 `generate_signals(data) → Series[1/-1/0]` |
| **EAV 指標** | `TechnicalIndicator`（stock_id, date, name, value），`load_data()` pivot 為寬表 |
| **除權息** | Layer 1 回溯調整 OHLC + 重算指標（保留 `raw_*`）；Layer 2 原始價格交易 + 股利入帳；預設關閉，`--adjust-dividend` 啟用 |
| **Watchlist** | `get_effective_watchlist()`：DB 優先，`quant_params.yaml` fallback，全模組統一呼叫 |
| **粗篩不得 fail-open** | 定義性資料（估值/營收）缺席時，`_coarse_filter` 必須**收斂**而非放行。歷史教訓（2026-08-04）：`_value.py` 的 `else` 分支在估值表為空時把 PE/殖利率閘門整段跳過，模式靜默退化成流動性篩選且無 log 警示——2024~2025 的 PIT 重放因此全部失效，且方向偏樂觀（value 看似「產能率 100%、五模式之冠」）。Stage 0.5 的覆蓋率閘門亦須看**近 `VALUATION_FRESH_WINDOW_DAYS` 日窗口**而非全表相異股票數（全表計數一旦累積夠就永不觸發，live 只剩候選池補抓的 43~150 檔） |
| **Universe 漏斗** | Stage 1 SQL 硬過濾 → Stage 2 流動性（DailyFeature 優先/覆蓋率≥30% 時使用，否則 fallback DailyPrice + 相對流動性救援）→ Stage 3 趨勢（Value/Dividend 跳過）→ Candidate Memory（3 天漸進衰減）；Regime 自適應門檻（`REGIME_UNIVERSE_ADJUSTMENTS`） |
| **Regime 四狀態** | bull/bear/sideways/crisis；三訊號多數決 + 市場寬度降級 + Crisis 快速覆蓋；影響：選股權重、評分閾值（bull=0.45/crisis=0.60）、ATR 倍數、Universe 門檻、部位大小 |
| **Regime 冪等（P0 #15）** | `MarketRegimeDetector().detect()` 對**同一 TAIEX 資料日**恆等：呼叫端可自由 `MarketRegimeDetector()` 新建實例（現況 10+ 處），跨實例/跨行程都拿到同一 regime，hysteresis 每個資料日只推進一次。**勿**改回以 `date.today()` 為鍵——morning-routine Step 0 在同步前執行，會把 regime 凍結在前一交易日。回傳 `state_advanced` 標示本次是否推進 |
| **PIT 時間注入（B1）** | `MarketScanner.run(as_of=...)` 與 `MarketRegimeDetector.detect(as_of=...)` 是兩個注入點（regime 驅動權重/門檻/模式封鎖，漏了它重放就不成立）；歷史 as_of 時 regime 走**唯讀**路徑不推進狀態機；引擎層一律用 `self._as_of()`，**禁止裸 `date.today()`**（`tests/test_pit.py` 靜態守門，純函數則須加 `as_of` 參數）。查詢一律加時間上界；基本面另套公布時滯（`data/pit.py`）——`MonthlyRevenue.date` 是營收月份不是公布日，直接用 `<= as_of` 會漏未來。`as_of` 為歷史日時自動 offline，禁止一切外部補抓。shared 與 DB 兩路徑須套**相同**上界 |
| **IC 執法治理（P0 #16）** | 任何「因 IC 而自動行動」一律先過 `ic_governance.select_enforceable_ic()`：不可執法時只告警。**IC 反向不再跳過掃描**——五模式恆掃恆落庫（否則模式產不出 `discovery_record`，IC 無從重算而自鎖），停用只作用在 rotation 層（`resolve_rankings(exclude_modes=...)` 阻擋新買入，賣出/停損/到期不受影響）。backtest 路徑**刻意不套用**（今日裁定套到歷史日＝look-ahead）。新增 IC 驅動開關時務必沿用此模組，勿再自行 `iloc[-1]` |
| **E2b/E2c 凍結中（P0 #17）** | IC 自動調權（E2b）與分數翻轉/中性化（E2c）**預設不生效**，開關在 `quant.ic_governance`（兩者 false）。凍結只在唯一套用點 `_score_candidates`——底層純函數與 `_apply_ic_weight_adjustment` 行為不變，**仍照常計算並記錄** would-be 動作（log 標【凍結中，未生效】）。`_ic_actions` 凍結時留空，避免 CLI 的 (N)/(F)/(D) 誤示為已套用。**解凍前提**：B2 落地 + `valuation`/`dividend` 維度納入落庫 + 改用標準誤顯著性判定，詳 `config.py:ICGovernanceConfig` |
| **Scanner 評分** | 四維度（技術+籌碼+基本面+消息面）；技術面 3 Cluster 等權 v2（報酬動能/量能/突破，各 1/3）；零方差因子自動排除（`exclude_zero_variance_factors`）；子因子 IC 自動權重調整；Rolling IC + Per-Regime IC 監控 |
| **單一漏斗（N2）** | **`run()` 只有 `MarketScanner` 一份實作，子類禁止覆寫**（`tests/test_scanner_pipeline_parity.py` 有契約測試守門）。模式差異只能透過：①`_STAGES = StageConfig(...)` 宣告不跑哪些階段；②4 個 hook（`_prepare_before_load` / `_after_market_data_loaded` / `_sync_candidate_valuation` / `_reload_candidate_valuation`）；③`_coarse_filter` / `_compute_*_scores` / `_compute_extra_scores`。<br>⚠ value/dividend/growth 的 `_STAGES` 多數 False 是**現況存檔非設計主張**（源自舊複製貼上），改動任一旗標都會改變選股——先看 MASTER_PLAN §7 #3b 的實測影響表。**已開啟**：4.2 回撤縮表（2026-08-01，五模式一致）。改旗標時務必同步更新 `tests/test_scanner_pipeline_parity.py` 的 `_EXPECTED` 基準**並在該處寫明原因** |
| **輪動風控** | Drawdown Kill Switch（≥25% 清倉）、Portfolio Heat、Correlation Budget（60 日 rolling）、Crisis 硬阻擋、Ex-Ante VaR（Component VaR 分解） |
| **T+1 延遲** | BacktestEngine + Walk-Forward + Discover + **Rotation 回測與 live** 一致執行訊號延遲，消除 look-ahead bias。Rotation backtest：D 日 close 決策 → 暫存 pending_exec → D+1 開盤成交。**Live（A2，2026-07-06）**：`update()` = `fill_pending(today)`（先以 open 成交昨日 `RotationPendingOrder`）→ `decide(today)`（close 決策寫明日 pending）；renew 與熔斷即時（熔斷為與 backtest 的刻意差異）。買單 TTL 2 交易日（逾期不論有無報價一律取消）、同股僅允許一張在途買單（decide 去重 + fill 端 UNIQUE 防護）、風控賣單停牌以 ref_price 成交不凍結；`update --all` per-portfolio 隔離 |
| **動態滑價** | 三因子模型 + A4 participation impact（`compute_dynamic_slippage`，傳 `order_shares` 時加 c×√(下單量/當日量)）；流動性約束（`apply_liquidity_limit`）；漲跌停偵測（`detect_limit_price`） |
| **A4 交易現實化** | 混合單成本模型（2026-07-08）：委託拆整張單+盤中零股單，各計最低手續費 20/1 元、零股 notional 加 0.1% 滑價 premium；**股數計算不整張化**（sizing 不變，僅成本真實化）。成本 SSOT=`rotation.trade_cost_amounts`（未捨入），rotation live+backtest 恆開；BacktestEngine 走 `BacktestConfig.min_commission`/`participation_impact` 旗標（引擎預設關、`backtest` CLI 預設開，`--no-*` 可關） |

---

## 5. 開發流程與文件聯動

### 文件聯動規則

修改 `src/` 或 `main.py` 後：
- **`CLAUDE.md`**：架構變更、新指令、新測試、模組職責異動時更新
- **`usage.md`**：CLI 參數變動、工作流程調整、新功能上線時更新
- 僅規劃/詢問不涉及寫入時免更新

### CLI 設計原則

- 入口：`python main.py <子命令>`（51 子命令；parser 建構在 `main.py build_parser()`，dispatch 在 `main()`）
- 每日例行：`morning-routine`（Step 0~18 + 子步驟 8b/8c/8d/8e/9b/11b，含全市場同步 + discover + 風控 + 通知）
  - Step 8e「同步後 regime 重解」：Step 0 宏觀預檢在同步**之前**執行，其 regime 只到前一交易日；Step 12 輪動須用同步後的判定（`resolve_regime_after_sync()`，dry_run/skip_sync 跳過）
- 新增子命令須更新 `main.py` dispatch table + `docs/cli_commands.md`，**並附 CLI smoke test**（`tests/test_cli_smoke.py`；glue code 是測試盲區，`Announcement.title` 事故教訓）
- 完整指令參考見 [`docs/cli_commands.md`](docs/cli_commands.md)

---

## 6. 測試規範

- **策略**：純函數優先（零 mock）；DB 整合用 in-memory SQLite + transaction rollback；HTTP mock `requests.Session.get` + `time.sleep`
- **要求**：新增計算邏輯**必須**補測試
- **執行**：`pytest -v`（2808 測試 / 107 檔）
- **Fixtures**：`tests/conftest.py`（`in_memory_engine`/`db_session`/`sample_ohlcv`）；共用建構函數 `tests/scanner_helpers.py`
- 詳細測試檔對照表見 [`docs/testing_guide.md`](docs/testing_guide.md)

---

## 7. 外部文件索引

| 文件 | 內容 |
|------|------|
| [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) | **專案主計畫 SSOT**：成熟度評估、設計原則、P0/P1/P2 TODO、長期路線圖、技術債登記簿（規劃/審計/裁決前必讀） |
| [`docs/cli_commands.md`](docs/cli_commands.md) | 38 個子命令完整用法與範例 |
| [`docs/testing_guide.md`](docs/testing_guide.md) | 45 個測試檔對照表、Fixtures、覆蓋率指引 |
| [`docs/project_history.md`](docs/project_history.md) | 85 項已完成任務歷史（Phase 1~2） |
| `usage.md` | 使用者導向操作手冊 |
| `config/quant_params.yaml` | 量化參數與非機密設定（進版控，A5） |
| `config/secrets.yaml` | 機密設定（`.gitignore` 已排除） |

---

## 8. 已確認事項（規劃時勿重複提出）

- `config/secrets.yaml`（及 legacy `settings.yaml`）已在 `.gitignore`，token 從未進入 Git；A5 拆檔後量化參數（`quant_params.yaml`）進版控
- TWSE/TPEX `verify=False`：刻意設計（Windows 憑證問題）
- `src/notification/line_notify.py`：歷史遺留檔名，實為 Discord Webhook，不需重命名
- `datetime.utcnow()` DeprecationWarning：SQLAlchemy schema default，低優先級不影響功能
- FinMind token 為逐股資料必需；TWSE/TPEX 免 token
- `export-dashboard`（`SCHEMA_VERSION=4`）的 JSON 由獨立 repo **QuantMonitor**（`~/Projects/QuantMonitor`，SwiftUI iOS App）消費；改 export schema 會影響該 App，欄位異動須維持向後相容（新欄位設 optional）

---

## 9. 專案狀態

- Phase 1 + Phase 2 **全部完成**（85 項，測試 231→1761）
- Phase 3 實盤上線**暫緩**
- 無 Pending Tasks
