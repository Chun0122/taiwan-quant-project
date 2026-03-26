# CLAUDE.md

本檔案為 Claude Code (claude.ai/code) 在此專案中的操作指南。

## 專案概述

台股量化投資系統。CLI 驅動的流水線：資料擷取（FinMind API + TWSE/TPEX 官方資料）→ SQLite 儲存 → 技術指標計算 → 策略訊號 → 回測 → 報告/通知。

所有中文註解與 UI 文字皆為預期行為 — 本專案以繁體中文為主，目標市場為台灣證券交易所。

## 常用指令

```bash
# 安裝依賴
pip install -r requirements.txt

# CLI 入口 — 所有操作透過 main.py 執行
python main.py [command] [options]

# 常見工作流程：
python main.py sync                          # 從 FinMind 同步觀察清單資料（含 TAIEX）
python main.py compute                       # 計算技術指標
python main.py backtest --stock 2330 --strategy sma_cross
python main.py backtest --stock 2330 --strategy sma_cross --attribution  # 含五因子歸因分析
python main.py backtest --stock 2330 --strategy sma_cross --export-trades trades.csv  # 匯出交易明細
python main.py discover --top 20             # 全市場掃描（預設 momentum 模式）
python main.py discover momentum --top 20   # 短線動能掃描
python main.py discover swing --top 20      # 中期波段掃描
python main.py discover value --top 20      # 價值修復掃描
python main.py discover dividend --top 20   # 高息存股掃描
python main.py discover growth --top 20     # 高成長掃描
python main.py discover --skip-sync --top 10 # 使用已快取的 DB 資料
python main.py discover --compare            # 顯示與上次推薦的差異比較
python main.py discover all --skip-sync --top 20           # 五模式綜合比較
python main.py discover all --skip-sync --min-appearances 2 # 只顯示出現 2+ 模式的股票
python main.py discover all --skip-sync --export compare.csv # 匯出交叉比較表
python main.py discover momentum --weekly-confirm           # 啟用週線多時框確認（週線多頭 +5%，週線空頭 -5%）
python main.py discover-backtest --mode momentum  # 推薦績效回測（預設 5,10,20 天）
python main.py sync-mops                     # 同步 MOPS 重大訊息（預設 7 天）
python main.py sync-mops --days 30           # 同步最近 30 天
python main.py sync-revenue                  # 同步全市場月營收（上月，從 MOPS）
python main.py sync-revenue --months 3       # 同步最近 3 個月
python main.py dashboard                     # Streamlit 儀表板（localhost:8501）
python main.py validate                      # 資料品質檢查（全部股票）
python main.py validate --stocks 2330 2317   # 指定股票品質檢查
python main.py validate --export issues.csv  # 匯出問題清單
python main.py sync-financial                # 同步 watchlist 財報（預設最近 4 季）
python main.py sync-financial --stocks 2330  # 指定股票
python main.py sync-financial --quarters 8   # 最近 8 季
python main.py sync-holding                  # 同步大戶持股分級（週資料，TDCC 全市場 ~2928 支，存全市場讓 discover 7F 可用）
python main.py sync-holding --weeks 8        # 保留參數（TDCC 僅提供最新一週，歷史靠每週累積）
python main.py sync-broker                   # 同步 watchlist 分點交易資料（最近 5 日）
python main.py sync-broker --stocks 2330 2317  # 指定股票
python main.py sync-broker --days 10         # 最近 10 個交易日
python main.py sync-broker --from-discover   # 補抓最近 discover 推薦的分點資料（非 watchlist 股票）
python main.py sync-broker --watchlist-bootstrap  # 一次性補齊 watchlist 最大歷史分點資料（預設 120 天，首次部署用）
python main.py sync-broker --watchlist-bootstrap --days 60  # 自訂天數
python main.py sync-broker --from-file stocks.txt --watchlist-bootstrap  # 從文字檔讀取股票清單並 bootstrap（120 天）
python main.py sync-broker --from-file stocks.csv --watchlist-bootstrap --days 60  # CSV 格式 + 自訂天數
python main.py sync-info                     # 同步全市場股票基本資料（產業分類 + 上市/上櫃別）
python main.py sync-info --force             # 強制重新同步（即使 DB 已有資料）
python main.py sync-features                 # 計算全市場 DailyFeature（Feature Store，供 UniverseFilter 使用，預設 90 天）
python main.py sync-features --days 60       # 自訂回溯天數
python main.py sync-vix                       # 同步 VIX 波動率指數（台灣 TW_VIX via FinMind（已停用，graceful degradation）+ 美國 US_VIX via yfinance ^VIX）
python main.py sync-sbl                      # 同步全市場借券賣出資料（TWSE TWT96U，預設最近 3 天）
python main.py sync-sbl --days 5             # 同步最近 5 個交易日
python main.py alert-check                   # 掃描近期 MOPS 重大事件（法說會/財報/月營收，預設 7 天）
python main.py alert-check --days 14 --types earnings_call filing  # 指定天數與事件類型
python main.py alert-check --stocks 2330 2317 --notify  # 指定股票 + Discord 通知
python main.py revenue-scan                  # 掃描 watchlist 中 YoY ≥ 10% + 毛利率改善的個股
python main.py revenue-scan --min-yoy 20 --min-margin-improve 1.0  # 自訂篩選門檻
python main.py revenue-scan --top 10 --notify  # 顯示前 10 名並推播 Discord
python main.py export --list                 # 列出所有可匯出的資料表及筆數
python main.py export daily_price -o data/export/daily_price.csv  # 匯出日K線
python main.py export daily_price --stocks 2330 --start 2024-01-01  # 篩選匯出
python main.py export daily_price --format parquet -o data/export/dp.parquet  # Parquet 格式
python main.py import-data daily_price data/export/daily_price.csv  # 匯入 CSV
python main.py import-data daily_price data.csv --dry-run  # 僅驗證不寫入
python main.py suggest 2330                  # 單股進出場建議（ATR14+SMA20+RSI14+Regime）
python main.py suggest 2330 --notify         # 含 Discord 通知
python main.py watch add 2330                # 新增持倉監控（ATR14-based 止損止利）
python main.py watch add 2330 --price 580 --stop 555 --target 635 --qty 1000  # 手動指定
python main.py watch add 2330 --from-discover momentum  # 從 discover 記錄匯入
python main.py watch add 2330 --trailing     # 啟用移動止損（ATR×1.5，預設倍數）
python main.py watch add 2330 --trailing --trailing-multiplier 2.0  # 自訂 ATR 倍數
python main.py watch list                    # 列出持倉中的記錄
python main.py watch list --status all       # 列出全部（含已平倉/止損/止利/過期）
python main.py watch close 1 --price 595     # 平倉 ID=1 的持倉
python main.py watch update-status           # 批次更新止損/止利/過期狀態（含移動止損自動上移）
python main.py anomaly-scan                  # 掃描 watchlist 籌碼異動（量能暴增/外資大買超/借券激增/主力集中/隔日沖）
python main.py anomaly-scan --stocks 2330 2317  # 指定股票
python main.py anomaly-scan --vol-mult 3.0 --inst-threshold 5000000  # 自訂門檻
python main.py anomaly-scan --dt-threshold 0.3 # 隔日沖風險門檻（預設 0.2）
python main.py anomaly-scan --notify         # 掃描並推播 Discord
python main.py morning-routine --notify      # 每日早晨例行流程（sync-sbl → sync-broker watchlist + discover → discover all → alert-check → watch update-status → revenue-scan → anomaly-scan → Discord 摘要）
python main.py morning-routine --skip-sync --notify  # 跳過借券/分點同步（資料已新鮮時使用）
python main.py morning-routine --dry-run     # 預覽步驟與摘要（不實際執行）
python main.py watchlist list                # 列出 DB watchlist 清單（DB 空時顯示 YAML fallback）
python main.py watchlist add 2330            # 新增股票至 DB watchlist
python main.py watchlist add 2330 --name 台積電 --note 核心持倉  # 含名稱與備註
python main.py watchlist remove 2330        # 從 DB watchlist 移除股票
python main.py watchlist import              # 從 settings.yaml 一次性匯入所有股票至 DB
python main.py sync-concepts                 # 從 config/concepts.yaml 同步概念定義至 DB
python main.py sync-concepts --purge         # 先清除舊 yaml 記錄再重新匯入（概念重組時使用）
python main.py sync-concepts --from-mops     # 掃描近 90 天 MOPS 公告，以關鍵字自動標記概念成員
python main.py sync-concepts --from-mops --days 30  # 指定掃描天數
python main.py concepts list                 # 列出所有概念及成員股數
python main.py concepts list CoWoS封裝       # 列出特定概念的成員清單
python main.py concepts add CoWoS封裝 2330   # 手動新增成員（source=manual）
python main.py concepts remove CoWoS封裝 2330 # 移除成員
python main.py concept-expand CoWoS封裝 --threshold 0.7  # 以價格相關性找出候選股
python main.py concept-expand CoWoS封裝 --threshold 0.7 --auto  # 自動加入 DB（source=correlation）
python main.py rotation create --name mom5_3d --mode momentum --max-positions 5 --holding-days 3 --capital 1000000  # 建立輪動組合
python main.py rotation create --name all10_5d --mode all --max-positions 10 --holding-days 5 --capital 2000000 --no-renewal  # 綜合模式，停用續持
python main.py rotation update --name mom5_3d   # 每日更新指定組合
python main.py rotation update --all            # 更新所有 active 組合
python main.py rotation status --name mom5_3d   # 查看組合狀態與持倉
python main.py rotation status --all            # 列出所有組合概覽
python main.py rotation history --name mom5_3d --limit 30  # 已平倉交易記錄
python main.py rotation backtest --name mom5_3d --start 2025-01-01 --end 2025-12-31  # 歷史回測
python main.py rotation backtest --mode momentum --max-positions 5 --holding-days 3 --start 2025-01-01 --end 2025-12-31  # Ad-hoc 回測
python main.py rotation list                    # 列出所有輪動組合
python main.py rotation pause --name mom5_3d    # 暫停每日更新
python main.py rotation resume --name mom5_3d   # 恢復每日更新
python main.py rotation delete --name mom5_3d   # 刪除組合及持倉
```

### 測試

使用 pytest 測試框架，1377 個測試（38 個測試檔）覆蓋核心模組：

```bash
# 執行全部測試
pytest -v

# 執行單一測試檔
pytest tests/test_factors.py -v

# 帶覆蓋率報告
pytest --cov=src --cov-report=term-missing
```

測試檔案結構：

| 測試檔                          | 測試對象                                         | 類型                   |
| ------------------------------- | ------------------------------------------------ | ---------------------- |
| `tests/test_factors.py`         | `src/screener/factors.py` 8 個篩選因子           | 純函數                 |
| `tests/test_ml_features.py`     | `src/features/ml_features.py` 特徵工程           | 純函數                 |
| `tests/test_backtest_engine.py` | `src/backtest/engine.py` 回測計算（含 TestAtrBasedStop ATR-based 止損止利）+ `src/backtest/metrics.py` 共用績效指標（Sortino/Calmar/VaR/CVaR/爆倉防護）+ 交易統計（`TestComputeTradeStats` 7 個 + `TestTradesToDataframe` 4 個 + `TestExportTrades` 2 個） | 純函數 + mock Strategy |
| `tests/test_twse_helpers.py`    | `src/data/twse_fetcher.py` 工具函數              | 純函數                 |
| `tests/test_scanner.py`         | `src/discovery/scanner.py` 基底+Momentum+Swing+Value+Dividend+Growth 六類掃描 + 產業加成 + `TestComputeTaixRelativeStrength`（6 個）+ `TestApplyCrisisFilter`（4 個）+ `TestCrisisFilterMA60`（4 個）+ `TestRiskFilterEnhancements`（4 個）+ `TestCrisisEntryTriggerText`（3 個）+ `TestDetectDaytradeBrokers`（6 個）+ `TestComputeDaytradePenalty`（7 個）+ `TestComputeInstitutionalPersistence`（8 個）+ `TestDaytradePersistenceDampening`（4 個）+ `TestComputeInstNetBuySlope`（7 個）+ `TestComputeHhiTrend`（6 個）+ `TestApplyChipQualityModifiers`（6 個）+ `TestComputeVolumePriceDivergence`（6 個）+ `TestApplyScoreThreshold`（7 個）+ `TestApplySectorDiversification`（6 個）+ `TestApplyVolumePriceDivergence`（2 個）+ `TestComputeMomentumDecay`（8 個）+ `TestComputeInstitutionalAcceleration`（9 個）+ `TestComputeMultiTimeframeAlignment`（8 個）+ `TestMultiTimeframeForceExclude`（3 個）+ `TestComputeValueWeightedInstFlow`（6 個）+ `TestComputeEarningsQuality`（7 個）+ `TestDrawdownAdjustedTopN`（7 個）+ `TestComputeChipMacd`（7 個）+ `TestWinRateThresholdAdjustment`（6 個）+ `TestComputeFactorIc`（5 個）+ `TestComputeIcWeightAdjustments`（5 個）+ `TestKeyPlayerCostBasis`（6 個）+ `TestAdaptiveAtrMultiplier`（6 個）+ `TestRevenueAccelerationScore`（6 個）+ `TestPeerFundamentalRanking`（5 個）+ `TestComputeMfeMae`（6 個）| 純函數                 |
| `tests/test_mops.py`            | `mops_fetcher.py` 情緒分類 + 月營收解析 + `scanner.py` 消息面評分 + Announcement ORM + 權重矩陣 | 純函數 + in-memory SQLite |
| `tests/test_regime.py`          | `src/regime/detector.py` 市場狀態偵測 + 權重矩陣 + `TestDetectCrisisSignals`（23 個，含台灣 VIX spike + 單日急跌 + 美國 VIX spike 7 個）+ `TestRegimeWeightsCrisis`（5 個）+ `TestComputeMarketBreadthPct`（5 個）+ `TestBreadthDowngrade`（5 個）+ `TestPanicVolumeSignal`（6 個）+ `TestCheckTransitionCondition`（6 個）+ `TestApplyHysteresis`（13 個）+ `TestRegimeStateMachine`（6 個）| 純函數                 |
| `tests/test_fetcher.py`         | `src/data/fetcher.py` API 封裝                   | mock HTTP              |
| `tests/test_config.py`          | `src/config.py` 設定載入                         | tmp_path               |
| `tests/test_dividend_adjustment.py` | 除權息還原（價格調整 + 指標重算 + 回測股利入帳） | 純函數 + mock Strategy |
| `tests/test_discover_performance.py` | `src/discovery/performance.py` 推薦績效回測    | in-memory SQLite       |
| `tests/test_db_integration.py`  | ORM + upsert + pipeline + DiscoveryRecord + Watchlist ORM CRUD + `get_effective_watchlist()` DB優先/YAML fallback | in-memory SQLite       |
| `tests/test_indicators.py`     | `src/features/indicators.py` compute_indicators_from_df()（含 adx_14）+ aggregate_to_weekly() | 純函數           |
| `tests/test_strategies.py`     | 6 個策略 generate_signals() + 除權息調整          | 純函數                 |
| `tests/test_formatter.py`      | `src/report/formatter.py` 4 個格式化函數          | 純函數                 |
| `tests/test_notification.py`   | `src/notification/line_notify.py` 通知模組        | 純函數 + mock HTTP     |
| `tests/test_report_engine.py`  | `src/report/engine.py` 4 個 _compute_* 評分函數   | 純函數                 |
| `tests/test_portfolio.py`      | `src/backtest/portfolio.py` 組合回測 + P6 進階指標（Sortino/Calmar/VaR/CVaR） | 純函數 + mock Strategy |
| `tests/test_walk_forward.py`   | `src/backtest/walk_forward.py` Walk-Forward 驗證  | 純函數 + mock Strategy |
| `tests/test_pipeline.py`       | `src/data/pipeline.py` ETL 函數                   | in-memory SQLite       |
| `tests/test_allocator.py`      | `src/backtest/allocator.py` risk_parity + mean_variance 權重計算 + fallback | 純函數 + scipy         |
| `tests/test_validator.py`      | `src/data/validator.py` 6 個品質檢查純函數         | 純函數                 |
| `tests/test_financial.py`     | `src/data/fetcher.py` 財報 EAV pivot + 衍生比率 + pipeline upsert | 純函數 + mock API + in-memory SQLite |
| `tests/test_market_overview.py` | `data_loader` 市場總覽查詢 + `charts` 4 個圖表函數 | in-memory SQLite + 純函數 |
| `tests/test_io.py`             | `src/data/io.py` 匯出/匯入 + 驗證 + round-trip     | 純函數 + in-memory SQLite |
| `tests/test_entry_exit.py`     | `src/entry_exit.py` `compute_atr_stops`（7 個）+ `compute_entry_trigger`（7 個）+ `assess_timing` crisis（3 個）+ `REGIME_ATR_PARAMS` 一致性（2 個） | 純函數 |
| `tests/test_suggest.py`        | `src/features/indicators.py` `calc_rsi14_from_series` + `src/entry_exit.py` `assess_timing`（含 crisis 測試）+ `src/notification/line_notify.py` `format_suggest_discord` | 純函數 |
| `tests/test_watch.py`          | `main.py` `_compute_watch_status` / `_compute_trailing_stop` 純函數 + `WatchEntry` ORM CRUD（含 trailing stop 欄位） | 純函數 + in-memory SQLite |
| `tests/test_holding.py`        | `_extract_level_lower_bound` + `compute_whale_score` + `HoldingDistribution` ORM + `fetch_holding_distribution`（FinMind mock，舊接口相容） | 純函數 + in-memory SQLite + mock HTTP |
| `tests/test_alert.py`          | `classify_event_type` 事件分類 + `Announcement` event_type ORM + `_compute_revenue_scan` 純函數（YoY + 毛利率掃描） | 純函數 + in-memory SQLite |
| `tests/test_sbl.py`            | `fetch_twse_sbl` 欄位映射 + `SecuritiesLending` ORM + `compute_sbl_score` 純函數 + `MomentumScanner` 6-factor 啟用/降級/逆向排名 | 純函數 + in-memory SQLite + mock HTTP |
| `tests/test_broker.py`         | `fetch_dj_broker_trades` HTML 解析（Big5/BHID/多分點彙整/單位換算）+ `BrokerTrade` ORM + `compute_broker_score` HHI/連續天 + `MomentumScanner` 7-factor 啟用/降級/集中度影響 + `TestLoadBrokerDataExtendedCloseProxy` 收盤價代理均價（NULL 填補/不覆蓋/無資料降回 7F）+ `TestSyncBrokerBootstrap` 逐日查詢簽名/獨立性/預設值/單日 vs 期間 | 純函數 + in-memory SQLite + mock HTTP |
| `tests/test_anomaly.py`        | `detect_volume_spike`/`detect_institutional_buy`/`detect_sbl_spike`/`detect_broker_concentration`/`detect_daytrade_risk` 五個純函數（量能暴增/外資大買超/借券激增/主力集中/隔日沖風險）| 純函數 |
| `tests/test_attribution.py`    | `FactorAttribution.compute()` / `compute_from_df()` 五因子歸因（momentum/reversal/quality/size/liquidity）純函數測試 | 純函數 |
| `tests/test_universe.py`       | `filter_liquidity`/`filter_trend` 純函數 + `UniverseFilter._stage1_sql_filter()`（ETF 排除/天數不足/低價/掛牌類型/NULL fallback）+ Stage 2 DailyPrice fallback + Candidate Memory 模式隔離 + `compute_and_store_daily_features()` ETL（ma20/turnover_ma5/upsert 冪等） | 純函數 + in-memory SQLite |
| `tests/test_concepts.py`       | `classify_concepts()` 關鍵字比對（10 個）+ `compute_concept_momentum()` 純函數（5 個）+ `compute_concept_institutional_flow()` 純函數（4 個）+ `compute_concept_correlation_candidates()` 純函數（5 個）+ `TestConceptBonusCap` cap 機制（4 個）+ `ConceptGroup/ConceptMembership` ORM CRUD（6 個） | 純函數 + in-memory SQLite |
| `tests/test_rotation.py`      | `compute_rotation_actions()` 核心輪動邏輯（冷啟動/持有期/到期續持/止損/換股/資金配置，共 19 個）+ `compute_position_pnl()`/`compute_shares()` 損益計算（6 個）+ 交易日工具函數（6 個）+ `resolve_rankings()` 排名解析（2 個）+ `RotationPortfolio`/`RotationPosition` ORM CRUD（2 個） | 純函數 + in-memory SQLite |

共用 fixtures 在 `tests/conftest.py`：`in_memory_engine`（session scope）、`db_session`（function scope，transaction rollback 隔離）、`sample_ohlcv`。測試用資料建構函數（`make_price_df` 等 5 個）集中於 `tests/scanner_helpers.py`（因 `tests/` 有 `__init__.py`，conftest 不可直接 import）。

新增或修改模組後，應確保現有測試通過（`pytest -v`），並為新的純函數/計算邏輯補充測試。

## 架構

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

### 核心設計模式

**策略註冊機制** (`src/strategy/__init__.py`)：所有策略註冊於 `STRATEGY_REGISTRY` 字典，CLI 依名稱解析策略。目前 9 個策略：`sma_cross`、`rsi_threshold`、`bb_breakout`、`macd_cross`、`buy_and_hold`、`multi_factor`、`ml_random_forest`、`ml_xgboost`、`ml_logistic`。新增策略方式：繼承 `Strategy`，實作 `generate_signals(data) → Series[1/-1/0]`，加入註冊表。

**EAV 指標儲存** (`src/data/schema.py:TechnicalIndicator`)：技術指標採用 Entity-Attribute-Value 模式（stock_id, date, name, value），於 `Strategy.load_data()` 時樞紐轉換為寬表。

**除權息還原回測** (`--adjust-dividend`)：兩層架構 — Layer 1 在 `Strategy.load_data()` 回溯調整 OHLC 價格並從調整後價格重算技術指標（避免除權息日產生假訊號），保留 `raw_*` 原始價格；Layer 2 在 `BacktestEngine.run()` 使用原始價格交易，並在除權息日將現金股利入帳、股票股利增加持股。預設關閉（`adjust_dividend=False`），透過 CLI `--adjust-dividend` 旗標啟用。

**SQLAlchemy Session** (`src/data/database.py`)：一律使用 `with get_session() as session:` 上下文管理器。批次寫入使用 `sqlite_upsert().on_conflict_do_nothing()`。DB 操作前需呼叫 `init_db()`。新增 `get_db_watchlist()` 從 DB 查 Watchlist 表、`get_effective_watchlist()` 實作「DB 優先，settings.yaml fallback」邏輯，供所有需要 watchlist 的模組統一呼叫。

**三層資料來源策略**（`src/data/pipeline.py:sync_market_data`）：

1. TWSE/TPEX 官方開放資料（免費，6 次 API 呼叫取得全市場 ~6000 支股票，含融資融券）
2. FinMind 批次 API（付費帳號）
3. FinMind 逐股擷取（免費備援，速度較慢）

### 模組職責

| 模組                                | 角色                                                                                                              |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `src/data/fetcher.py`               | FinMind API 封裝（逐股 + 批次 + 財報三表 EAV pivot + `fetch_taiwan_vix()` 台灣 VIX（FinMind 已移除 `TaiwanOptionMarketVIX` dataset，graceful degradation 回傳空 DataFrame））；`fetch_us_vix()` 模組級純函數（yfinance ^VIX，stock_id="US_VIX"，獨立於 FinMind） |
| `src/data/twse_fetcher.py`          | TWSE/TPEX 官方資料（全市場、免費）；`fetch_twse_sbl()` 借券賣出彙總（TWT96U）；`fetch_dj_broker_trades()` 分點進出彙整（DJ 端點，替代 FinMind，Big5 HTML 解析，BHID 彙整，date=end，buy/sell 已換算為股）；**`fetch_tdcc_holding_all_market()`** TDCC 集保戶股權分散表（全市場，免費，替代 FinMind TaiwanStockHoldingSharesPer，CSV，tier 1-15 → level 字串） |
| `src/data/pipeline.py`              | ETL 調度、寫入 DB；`_sync_per_stock()` 通用逐股同步輔助（cache 檢查→fetch→upsert，供估值/營收/財報共用）；`_classify_security_type()` 純函數（stock_id 規則推斷 security_type）；`compute_and_store_daily_features(lookback_days=90)` 計算全市場 DailyFeature（ma20/ma60/volume_ma20/turnover_ma5/momentum_20d/volatility_20d）並以 upsert 寫入 Feature Store |
| `src/data/mops_fetcher.py`          | MOPS 公開資訊觀測站（重大訊息 + 全市場月營收，免費）；`classify_event_type()` 純函數（governance_change / buyback / earnings_call / investor_day / filing / revenue / general，優先序由高到低）；`classify_sentiment()` 使用 Regex 上下文比對（`_NEGATIVE_CONTEXT_PATTERNS` / `_POSITIVE_CONTEXT_PATTERNS`）優先於單詞比對，正確處理「處分利益/持股」、「澄清衰退/報導」等模糊語境 |
| `src/data/schema.py`                | 25 張 SQLAlchemy ORM 資料表（含 Announcement、DiscoveryRecord、FinancialStatement、HoldingDistribution、SecuritiesLending、BrokerTrade、WatchEntry、Watchlist、DailyFeature、ConceptGroup、ConceptMembership、RotationPortfolio、RotationPosition）；`StockInfo` 含 `security_type` 欄位；`DiscoveryRecord` 含 `concept_bonus`/`chip_tier`/`daytrade_penalty`/`daytrade_tags` 欄位 |
| `src/data/validator.py`             | 資料品質檢查（6 個純函數檢查 + orchestrator + console 報告）                                                       |
| `src/data/io.py`                    | 通用資料匯出/匯入（CSV/Parquet，含欄位驗證 + upsert）                                                            |
| `src/data/migrate.py`               | DB schema 遷移工具                                                                                                |
| `src/constants.py`                  | 全系統共用常數集中管理：交易成本（COMMISSION_RATE/TAX_RATE/SLIPPAGE_RATE）、DB/ETL（UPSERT_BATCH_SIZE/API_SLEEP_*）、籌碼異動門檻（DEFAULT_VOL_MULT/DEFAULT_INST_THRESHOLD/DEFAULT_SBL_SIGMA/DEFAULT_HHI_THRESHOLD/DEFAULT_DT_THRESHOLD）、VIX 危機偵測（VIX_STOCK_ID/CRISIS_VIX_LEVEL/CRISIS_VIX_DAILY_CHANGE/CRISIS_SINGLE_DAY_DROP）、美國 VIX（US_VIX_STOCK_ID/CRISIS_US_VIX_LEVEL/CRISIS_US_VIX_DAILY_CHANGE） |
| `src/entry_exit.py`                 | 進出場建議共用純函數：`REGIME_ATR_PARAMS` 常數（bull/sideways/bear/crisis ATR 倍數）、`compute_atr_stops()` ATR 止損止利、`compute_entry_trigger()` 進場觸發文字、`assess_timing()` 時機評估（RSI+SMA+Regime 決策矩陣）；Discover/Suggest/Watch 三系統共用 |
| `src/config.py`                     | Pydantic 設定模型 + `load_settings()`                                                                             |
| `src/features/indicators.py`        | SMA/RSI/MACD/BB/ADX(14) → EAV 格式；`compute_indicators_from_df()` 寬表純函數（含 adx_14）；`calc_rsi14_from_series()` 純函數；`aggregate_to_weekly()` 純函數（日K聚合週K + 週線 SMA13/RSI14/MACD） |
| `src/features/ml_features.py`       | ML 特徵矩陣（動能、波動度、量比）                                                                                 |
| `src/strategy/base.py`              | 抽象 `Strategy`：`load_data()` + `generate_signals()` + 除權息調整（`_apply_dividend_adjustment`）                |
| `src/strategy/__init__.py`          | `STRATEGY_REGISTRY`（9 個策略）                                                                                   |
| `src/strategy/ml_strategy.py`       | ML 策略（Random Forest / XGBoost / Logistic）                                                                     |
| `src/backtest/attribution.py`       | 五因子歸因分析（`FactorAttribution` 類別）：momentum/reversal/quality/size/liquidity 因子暴露 × Pearson 相關係數；`compute(BacktestResultData, data)` + `compute_from_df(trades_df, data)` 雙接口 |
| `src/backtest/metrics.py`           | 回測績效指標計算共用純函數：`compute_metrics()`（10 個指標+爆倉防護）、`compute_trade_stats()`（持倉天數/連勝連敗/出場原因分佈/勝敗損益分析）、`trades_to_dataframe()`（含 holding_days 計算欄位）、`export_trades()`（CSV 匯出），供 engine/portfolio/walk_forward 統一呼叫 |
| `src/backtest/engine.py`            | 交易模擬、風險管理、部位控管、除權息股利入帳                                                                      |
| `src/backtest/allocator.py`         | 投資組合配置計算（risk_parity / mean_variance），scipy 優化純函數                                                 |
| `src/backtest/portfolio.py`         | 多股票組合回測，支援 equal_weight / custom / risk_parity / mean_variance 四種配置                                 |
| `src/backtest/walk_forward.py`      | Walk-Forward 滾動窗口驗證（防過擬合）                                                                             |
| `src/optimization/grid_search.py`   | Grid Search 參數優化器                                                                                            |
| `src/screener/factors.py`           | 8 個篩選因子（技術面/籌碼面/基本面）                                                                              |
| `src/screener/engine.py`            | 多因子篩選引擎（watchlist 內掃描）                                                                                |
| `src/discovery/scanner/`            | 全市場選股 package（`_base.py` 基類 + `_momentum.py`/`_swing.py`/`_value.py`/`_dividend.py`/`_growth.py` 五模式 + `_functions.py` 共用純函數）。四階段漏斗（含風險過濾），四維度評分（技術+籌碼+基本面+消息面）+ 產業/概念/週線加成，權重依 Regime 動態調整（含 crisis 模式）。MomentumScanner 籌碼面最高 8-factor（含智慧分點 Smart Broker）；隔日沖偵測+扣分（含法人連續性調節）；籌碼品質修正（斜率+HHI趨勢）；Stage 3.6 量價背離偵測（`compute_volume_price_divergence()` 純函數，價漲量縮 -5%/量價齊揚 +2%）；Stage 3.7 動態評分閾值（`MIN_SCORE_THRESHOLDS` Regime 別最低分，bull=0.45/crisis=0.60）；Stage 3.5c 動量衰減偵測（`compute_momentum_decay()` RSI 頂背離+MACD 柱縮短，momentum/growth -3%~-6%）；Stage 3.5d 籌碼加速度（`compute_institutional_acceleration()` 法人買超加速 +2%~+4%）；Stage 3.5e 多時框強制共振（`compute_multi_timeframe_alignment()` 日週方向一致±4%，momentum/growth 日多週空矛盾直接排除）；Stage 4.1 同產業分散化（`SECTOR_MAX_RATIO=25%`）；Stage 4.2 回撤降頻（TAIEX 20日回撤>-10% 砍半/>-15% 砍1/3，value/dividend 不受影響）；籌碼面法人金額加權（`compute_value_weighted_inst_flow()` 衰減加權替代純天數）；盈餘品質分數（`compute_earnings_quality()` 現金流+收入品質+負債，Value/Dividend 15%權重）；Stage 3.5f 籌碼面 MACD（`compute_chip_macd()` 法人淨買超短期5日 vs 長期20日 EMA 交叉，吸籌加速 +3%/出貨信號 -3%）；E1 勝率回饋循環（`compute_win_rate_threshold_adjustment()` 追蹤模式過去30天勝率，勝率<40% → 門檻+0.05，<50% → +0.02）；E2 因子有效性監控（`compute_factor_ic()` Spearman IC 計算 + `compute_ic_weight_adjustments()` IC 驅動權重調整，日誌記錄因子有效性）；Stage 3.5g 主力成本分析（`compute_key_player_cost_basis()` Top-3 主力加權成本 vs 現價 → `score_key_player_cost()` 護盤/風險評分 ±3%）；D1 動態停損（`compute_adaptive_atr_multiplier()` 個股 MDD 調整 ATR 倍數，穩定股放寬×1.2/高波動收緊×0.7）；C2 營收加速推廣（`compute_revenue_acceleration_score()` 連續 N 月 YoY 加速計數，MomentumScanner 基本面 80% base + 20% 加速度）；C3 同業基本面排名（`compute_peer_fundamental_ranking()` 同產業 ROE/毛利率/營收 percentile，前25% +3%/後25% -3%）；E3 MFE/MAE 分析（`compute_mfe_mae()` 追蹤推薦後最大有利/不利偏移） |
| `src/discovery/universe.py`         | Universe Filtering 三層漏斗（`UniverseConfig` dataclass + `UniverseFilter`）：Stage 1 SQL 硬過濾（StockInfo security_type/listing_type/可用天數/收盤價）→ Stage 2 流動性（5 日均成交金額，DailyFeature 優先/DailyPrice fallback）→ Stage 3 趨勢動能（close > MA60 + 量比，Value/Dividend Scanner 跳過）→ Candidate Memory（union 前一日推薦降低換股率）；`filter_liquidity()` / `filter_trend()` 可獨立測試的純函數 |
| `src/discovery/performance.py`      | Discover 推薦績效回測（讀取歷史推薦 vs DailyPrice，計算 N 日報酬率、勝率、三層聚合統計）                           |
| `src/regime/detector.py`            | 市場狀態偵測（bull/bear/sideways/**crisis**），三訊號多數決（TAIEX vs SMA60/SMA120 + 20日報酬率）+ **市場寬度降級**（`compute_market_breadth_pct()` 純函數，>60% 股票跌破 MA20 → regime 降一級，資料來源 DailyFeature）+ **Crisis 快速訊號覆蓋**（7 訊號 ≥2 觸發：5日跌>5% / 連跌≥3天 / 波動率飆升1.8x / **爆量長黑**（成交量>20日均量×1.5且下跌）/ **台灣 VIX 飆升**（VIX>30 或單日漲幅>25%，TW_VIX from DailyPrice）/ **單日急跌**（TAIEX 單日跌>2.5%）/ **美國 VIX 飆升**（CBOE ^VIX>30 或單日漲幅>25%，US_VIX from yfinance），crisis 優先於寬度降級）；**Hysteresis 狀態機**（`HYSTERESIS_RULES` 轉換矩陣、`check_transition_condition()` 純函數、`apply_hysteresis()` 純函數、`RegimeStateMachine` 類別 JSON 持久化）：sideways→bull 需 3 天確認 +1%/3d、bull→sideways 快速降級 1 天、crisis 退出需低點遞增+波動率回落；`detect_crisis_signals()` 純函數；輸出五模式四維度權重矩陣（含 crisis 保守模式：news 25~40%，tech 5~10%） |
| `src/industry/analyzer.py`          | 產業輪動分析（法人動能 + 價格動能），提供 `compute_sector_scores_for_stocks()` 供 scanner 產業加成用；`compute_sector_relative_strength()` 模組級純函數（個股 20 日報酬率 vs 同產業中位數，超越 +20pp → +3%，落後 -20pp → -3%）               |
| `src/industry/concept_analyzer.py`  | 概念股輪動分析引擎；`compute_concept_momentum()` / `compute_concept_institutional_flow()` / `compute_concept_correlation_candidates()` 純函數；`ConceptRotationAnalyzer` 類別（`rank_concepts()` Percentile Rank + `compute_concept_scores_for_stocks()` ±5% 加成供 scanner Stage 3.3b 使用） |
| `src/portfolio/rotation.py`         | 輪動組合核心純函數：`compute_rotation_actions()`（到期/續持/止損/換股邏輯）、`compute_position_pnl()`（含交易成本）、`compute_shares()`、交易日工具函數 |
| `src/portfolio/manager.py`          | `RotationManager` 類別：每日更新（讀 DiscoveryRecord → rotation → DB）、歷史回測（逐日模擬 + 績效指標）、狀態查詢、交易歷史；`resolve_rankings()` 排名解析（單模式 / all 綜合 avg_score） |
| `src/report/engine.py`              | 每日選股報告（四維度綜合評分）                                                                                    |
| `src/report/formatter.py`           | Discord 訊息格式化（2000 字元限制）                                                                               |
| `src/report/ai_report.py`           | AI 選股摘要（`generate_ai_summary()`，呼叫 Claude API `claude-sonnet-4-6`，生成約 300 字繁中摘要；`discover --ai-summary` 旗標觸發） |
| `src/strategy_rank/engine.py`       | 策略排名引擎（批次回測 watchlist × strategies）                                                                   |
| `src/notification/line_notify.py`   | Discord Webhook 通知（檔名為歷史遺留）+ `format_suggest_discord()` 純函數（從 main.py 遷移）                       |
| `src/scheduler/simple_scheduler.py` | 前景排程（schedule 函式庫），`daily_sync_job()` delegate 給 `cmd_morning_routine()`（Step 0~8 + Discord），`weekly_holding_job()` 每週四同步 TDCC |
| `src/scheduler/windows_task.py`     | Windows 工作排程器 .bat + XML 產生器，每日 .bat 執行 `morning-routine --notify`，每週四 .bat 執行 `sync-holding`    |
| `src/visualization/app.py`          | Streamlit 儀表板入口                                                                                              |
| `src/visualization/charts.py`       | Plotly 圖表元件                                                                                                   |
| `src/visualization/data_loader.py`  | 儀表板資料載入                                                                                                    |
| `src/visualization/pages/`          | 儀表板 12 分頁（market_overview, stock_analysis, backtest_review, portfolio_review, strategy_comparison, screener_results, ml_analysis, industry_rotation, concept_rotation, discovery_history, position_monitoring） |
| `main.py`                           | CLI 調度器（argparse 子命令，36 個頂層子命令）；`_compute_revenue_scan()` 純函數；`_compute_trailing_stop()` 純函數；`detect_volume_spike`/`detect_institutional_buy`/`detect_sbl_spike`/`detect_broker_concentration`/`detect_daytrade_risk` 五個籌碼異動純函數；`_compute_anomaly_scan()` 聚合函數（含隔日沖偵測）；`cmd_anomaly_scan()` 籌碼異動警報；`cmd_morning_routine()` 早晨例行流程（含 Step 0 VIX 同步 + Step 6 rotation update + Step 8 anomaly-scan）；`cmd_rotation()` 輪動組合管理（create/update/status/history/backtest/list/pause/resume/delete）；`cmd_sync_vix()` VIX 同步；`cmd_watchlist()` DB-based watchlist 管理（add/remove/list/import）；`cmd_sync_info()` 全市場股票基本資料同步（StockInfo，`--force` 強制更新）；`cmd_sync_features()` 計算全市場 DailyFeature（`--days` 回溯天數） |

### 設定

`config/settings.yaml` 透過 `src/config.py` 中的 Pydantic 模型載入。全域存取方式：`settings.finmind.api_token`、`settings.fetcher.watchlist` 等。

**注意**：FinMind API token 為逐股資料所必需。TWSE/TPEX 端點無需 token。因部分系統憑證問題，TWSE/TPEX 的 SSL 驗證已停用（`verify=False`）。

## 開發慣例

- **檔案編碼**：所有原始碼為 UTF-8 含中文內容。在 Windows 上開啟檔案時務必指定 `encoding='utf-8'`。
- **DB 寫入**：使用 `_upsert_batch()`，batch_size=80 以符合 SQLite 變數上限。
- **API 速率控制**：FinMind 每次請求間隔 0.5 秒，TWSE/TPEX 間隔 3 秒。
- **日期格式**：FinMind 使用 ISO 格式（`YYYY-MM-DD`）；TWSE 使用 `YYYYMMDD`；TPEX 使用民國曆（`YYY/MM/DD`，年 = 西元年 - 1911）。
- **回測成本**：手續費 0.1425%、交易稅 0.3%（賣出時）、滑價 0.05%。
- **測試慣例**：純函數優先測試（零 mock）。DB 整合測試使用 in-memory SQLite + transaction rollback 隔離。HTTP 測試 mock `requests.Session.get` + `time.sleep`。新增計算邏輯時應補充對應測試。
- **代碼品質與格式 (Ruff)**：
  - 在提交任何變更或 Push 到 GitHub 之前，**必須執行 Ruff 檢查**。
  - 執行 `ruff check .` 確保無 Lint 錯誤。
  - 執行 `ruff format .` 確保格式統一。
- **文件聯動更新 (重要)**：
  - 每當修改原始碼（如 `src/` 或 `main.py`）後，**必須自動同步更新**以下兩份檔案：
    1. `CLAUDE.md`：若涉及架構變更、新增指令、新增測試或模組職責異動，須立即修正。
    2. `usage.md`：若涉及用戶端指令 (CLI) 參數變動、工作流程調整或新功能上線，須同步更新使用手冊。
  - **排除對象**：僅進行「規劃 (Planning)」或「詢問 (Ask)」而不涉及實際檔案寫入時，無需更新。

## Completed Tasks（已完成，共 58 項）

所有 58 項開發任務已完成（✅），測試數從 231 → 1323。以下為各階段摘要：

| 階段 | Task # | 重點 |
|------|--------|------|
| **基礎建設** | 1~10 | 五模式 Scanner（Momentum/Swing/Value/Dividend/Growth）、Dashboard 10 頁、測試覆蓋 360+、validate/export/import CLI、discover all 綜合比較 |
| **進出場 + 持倉** | 11~14 | Discover 進出場建議（ATR14+SMA20）、suggest 單股建議、ATR-based 自動止損止利、watch 持倉監控 + Dashboard |
| **資料整合** | 15~20 | 估值 Cold-Start（TWSE/TPEX）、大戶持股（TDCC）、MOPS 事件分類、revenue-scan、借券賣出（SBL）、分點進出（DJ 端點） |
| **自動化 + 監控** | 21~24 | morning-routine 7 步驟、移動止損（Trailing Stop）、anomaly-scan 五類籌碼異動、DB-based watchlist |
| **評分升級** | 25~41 | 週線多時框、進階回測指標、sync-info、五因子歸因、chip_tier 透明度、AI 選股摘要（Claude API）、策略比較頁、Smart Broker 8F、DJ 端點遷移、均價代理、Bootstrap、營收加速、Regime 自適應、消息面升級、財報因子、產業同儕強度、discover all avg_score |
| **Universe + 品質** | 42~49 | Universe Filtering 三層漏斗 + Feature Store、粗篩訊號/效能優化、相對估值（同業 PE）、EPS 連續性、消息面四項優化（異常率/事件加乘/Regex/衰減）、Swing 六項強化（財報/SBL/Smart Broker/ADX/突破/回測期間） |
| **概念 + Crisis** | 50~53 | 概念股系統（ORM/YAML/CLI/Dashboard/Scanner Stage 3.3b）、Crisis Regime 第四狀態（偵測/權重/過濾/morning-routine 警示）、隔日沖偵測+扣分、風險過濾強化（波動率 cap/MA60 濾網） |
| **統一重構 + 強化** | 54~58 | 進出場共用模組（entry_exit.py）、市場寬度降級+爆量長黑、Hysteresis 狀態機（JSON 持久化）、法人連續性因子+隔日沖交互、籌碼斜率+HHI 趨勢 |

## 已確認事項（規劃時勿重複提出）

以下項目已處理或為刻意設計，進行程式碼審查或規劃時應跳過：

- **`config/settings.yaml` 機密管理**：已在 `.gitignore` 中排除，Git 僅追蹤 `settings.yaml.example`。Token 從未進入版本控制。
- **TWSE/TPEX SSL `verify=False`**：已知的刻意行為，因部分 Windows 環境缺少 TWSE/TPEX 的根憑證鏈，停用驗證為目前的可接受方案。
- **`src/notification/line_notify.py` 檔名**：歷史遺留，實際實作為 Discord Webhook。不需重新命名，import 路徑已穩定。
- **`datetime.utcnow()` DeprecationWarning**：來自 SQLAlchemy schema 的 `default=datetime.utcnow`，為低優先級項目，不影響功能。
