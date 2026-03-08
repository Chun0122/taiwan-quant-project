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
python main.py sync-holding                  # 同步大戶持股分級（週資料，預設最近 4 週）
python main.py sync-holding --stocks 2330 2317  # 指定股票
python main.py sync-holding --weeks 8        # 最近 8 週
python main.py sync-broker                   # 同步 watchlist 分點交易資料（最近 5 日）
python main.py sync-broker --stocks 2330 2317  # 指定股票
python main.py sync-broker --days 10         # 最近 10 個交易日
python main.py sync-broker --from-discover   # 補抓最近 discover 推薦的分點資料（非 watchlist 股票）
python main.py sync-broker --watchlist-bootstrap  # 一次性補齊 watchlist 最大歷史分點資料（首次部署用）
python main.py sync-broker --watchlist-bootstrap --days 60  # 指定天數（預設 120，受 FinMind 帳號限制）
python main.py sync-info                     # 同步全市場股票基本資料（產業分類 + 上市/上櫃別）
python main.py sync-info --force             # 強制重新同步（即使 DB 已有資料）
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
python main.py anomaly-scan                  # 掃描 watchlist 籌碼異動（量能暴增/外資大買超/借券激增/主力集中）
python main.py anomaly-scan --stocks 2330 2317  # 指定股票
python main.py anomaly-scan --vol-mult 3.0 --inst-threshold 5000000  # 自訂門檻
python main.py anomaly-scan --notify         # 掃描並推播 Discord
python main.py morning-routine --notify      # 每日早晨例行流程（sync-sbl → sync-broker watchlist + discover → discover all → alert-check → watch update-status → revenue-scan → anomaly-scan → Discord 摘要）
python main.py morning-routine --skip-sync --notify  # 跳過借券/分點同步（資料已新鮮時使用）
python main.py morning-routine --dry-run     # 預覽步驟與摘要（不實際執行）
python main.py watchlist list                # 列出 DB watchlist 清單（DB 空時顯示 YAML fallback）
python main.py watchlist add 2330            # 新增股票至 DB watchlist
python main.py watchlist add 2330 --name 台積電 --note 核心持倉  # 含名稱與備註
python main.py watchlist remove 2330        # 從 DB watchlist 移除股票
python main.py watchlist import              # 從 settings.yaml 一次性匯入所有股票至 DB
```

### 測試

使用 pytest 測試框架，756 個測試覆蓋核心模組：

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
| `tests/test_backtest_engine.py` | `src/backtest/engine.py` 回測計算（含 TestAtrBasedStop ATR-based 止損止利）+ P6 進階指標（Sortino/Calmar/VaR/CVaR） | 純函數 + mock Strategy |
| `tests/test_twse_helpers.py`    | `src/data/twse_fetcher.py` 工具函數              | 純函數                 |
| `tests/test_scanner.py`         | `src/discovery/scanner.py` 基底+Momentum+Swing+Value+Dividend+Growth 六類掃描 + 產業加成 | 純函數                 |
| `tests/test_mops.py`            | `mops_fetcher.py` 情緒分類 + 月營收解析 + `scanner.py` 消息面評分 + Announcement ORM + 權重矩陣 | 純函數 + in-memory SQLite |
| `tests/test_regime.py`          | `src/regime/detector.py` 市場狀態偵測 + 權重矩陣 | 純函數                 |
| `tests/test_fetcher.py`         | `src/data/fetcher.py` API 封裝                   | mock HTTP              |
| `tests/test_config.py`          | `src/config.py` 設定載入                         | tmp_path               |
| `tests/test_dividend_adjustment.py` | 除權息還原（價格調整 + 指標重算 + 回測股利入帳） | 純函數 + mock Strategy |
| `tests/test_discover_performance.py` | `src/discovery/performance.py` 推薦績效回測    | in-memory SQLite       |
| `tests/test_db_integration.py`  | ORM + upsert + pipeline + DiscoveryRecord + Watchlist ORM CRUD + `get_effective_watchlist()` DB優先/YAML fallback | in-memory SQLite       |
| `tests/test_indicators.py`     | `src/features/indicators.py` compute_indicators_from_df() + aggregate_to_weekly() | 純函數           |
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
| `tests/test_suggest.py`        | `src/features/indicators.py` `calc_rsi14_from_series` + `main.py` `_assess_timing` + `src/notification/line_notify.py` `format_suggest_discord` | 純函數 |
| `tests/test_watch.py`          | `main.py` `_compute_watch_status` / `_compute_trailing_stop` 純函數 + `WatchEntry` ORM CRUD（含 trailing stop 欄位） | 純函數 + in-memory SQLite |
| `tests/test_holding.py`        | `_extract_level_lower_bound` + `compute_whale_score` + `HoldingDistribution` ORM + `fetch_holding_distribution` | 純函數 + in-memory SQLite + mock HTTP |
| `tests/test_alert.py`          | `classify_event_type` 事件分類 + `Announcement` event_type ORM + `_compute_revenue_scan` 純函數（YoY + 毛利率掃描） | 純函數 + in-memory SQLite |
| `tests/test_sbl.py`            | `fetch_twse_sbl` 欄位映射 + `SecuritiesLending` ORM + `compute_sbl_score` 純函數 + `MomentumScanner` 6-factor 啟用/降級/逆向排名 | 純函數 + in-memory SQLite + mock HTTP |
| `tests/test_broker.py`         | `fetch_broker_trades` 欄位映射 + `BrokerTrade` ORM + `compute_broker_score` HHI/連續天 + `MomentumScanner` 7-factor 啟用/降級/集中度影響 | 純函數 + in-memory SQLite + mock HTTP |
| `tests/test_anomaly.py`        | `detect_volume_spike`/`detect_institutional_buy`/`detect_sbl_spike`/`detect_broker_concentration` 四個純函數（量能暴增/外資大買超/借券激增/主力集中）| 純函數 |
| `tests/test_attribution.py`    | `FactorAttribution.compute()` / `compute_from_df()` 五因子歸因（momentum/reversal/quality/size/liquidity）純函數測試 | 純函數 |

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
| `src/data/fetcher.py`               | FinMind API 封裝（逐股 + 批次 + 財報三表 EAV pivot）                                                              |
| `src/data/twse_fetcher.py`          | TWSE/TPEX 官方資料（全市場、免費）；`fetch_twse_sbl()` 借券賣出彙總（TWT96U）                                     |
| `src/data/pipeline.py`              | ETL 調度、寫入 DB                                                                                                 |
| `src/data/mops_fetcher.py`          | MOPS 公開資訊觀測站（重大訊息 + 全市場月營收，免費）；`classify_event_type()` 純函數（earnings_call / investor_day / filing / revenue / general） |
| `src/data/schema.py`                | 19 張 SQLAlchemy ORM 資料表（含 Announcement、DiscoveryRecord、FinancialStatement、HoldingDistribution、SecuritiesLending、BrokerTrade、WatchEntry、Watchlist）  |
| `src/data/validator.py`             | 資料品質檢查（6 個純函數檢查 + orchestrator + console 報告）                                                       |
| `src/data/io.py`                    | 通用資料匯出/匯入（CSV/Parquet，含欄位驗證 + upsert）                                                            |
| `src/data/migrate.py`               | DB schema 遷移工具                                                                                                |
| `src/config.py`                     | Pydantic 設定模型 + `load_settings()`                                                                             |
| `src/features/indicators.py`        | SMA/RSI/MACD/BB → EAV 格式 + `compute_indicators_from_df()` 純函數（除權息還原用）+ `calc_rsi14_from_series()` 純函數（從 main.py 遷移）+ `aggregate_to_weekly()` 純函數（日K聚合週K + 週線 SMA13/RSI14/MACD） |
| `src/features/ml_features.py`       | ML 特徵矩陣（動能、波動度、量比）                                                                                 |
| `src/strategy/base.py`              | 抽象 `Strategy`：`load_data()` + `generate_signals()` + 除權息調整（`_apply_dividend_adjustment`）                |
| `src/strategy/__init__.py`          | `STRATEGY_REGISTRY`（9 個策略）                                                                                   |
| `src/strategy/ml_strategy.py`       | ML 策略（Random Forest / XGBoost / Logistic）                                                                     |
| `src/backtest/attribution.py`       | 五因子歸因分析（`FactorAttribution` 類別）：momentum/reversal/quality/size/liquidity 因子暴露 × Pearson 相關係數；`compute(BacktestResultData, data)` + `compute_from_df(trades_df, data)` 雙接口 |
| `src/backtest/engine.py`            | 交易模擬、風險管理、部位控管、除權息股利入帳                                                                      |
| `src/backtest/allocator.py`         | 投資組合配置計算（risk_parity / mean_variance），scipy 優化純函數                                                 |
| `src/backtest/portfolio.py`         | 多股票組合回測，支援 equal_weight / custom / risk_parity / mean_variance 四種配置                                 |
| `src/backtest/walk_forward.py`      | Walk-Forward 滾動窗口驗證（防過擬合）                                                                             |
| `src/optimization/grid_search.py`   | Grid Search 參數優化器                                                                                            |
| `src/screener/factors.py`           | 8 個篩選因子（技術面/籌碼面/基本面）                                                                              |
| `src/screener/engine.py`            | 多因子篩選引擎（watchlist 內掃描）                                                                                |
| `src/discovery/scanner.py`          | 全市場四階段漏斗（含風險過濾），支援 Momentum / Swing / Value / Dividend / Growth 五模式，四維度評分（技術+籌碼+基本面+消息面）+ 產業熱度加成（±5%）+ 週線趨勢加成（±5%，`--weekly-confirm` 啟用），權重依 Regime 動態調整。MomentumScanner 籌碼面支援最高 8-factor（含智慧分點因子）。`compute_broker_score()` 計算主力集中度（HHI）+ 連續進場天數（7 天窗口）。`compute_smart_broker_score()` 識別「高勝率+高獲利因子」Smart Broker（win_rate≥0.60、PF≥1.50、sell_events≥3、buy_val≥500萬）與「純蓄積型地緣分點」Accumulation Broker（sell_ratio≤0.10、倉位趨勢向上），合成為第 8 因子（僅對 Stage 2 候選股計算）。`_load_broker_data_extended(days=365, min_trading_days=20)` 載入全部可用 DB 歷史（最多 365 天），需有 ≥20 個交易日才啟用 Smart Broker（自適應，morning-routine 每日累積）。Value/Dividend 模式粗篩為嚴格模式：必須有估值資料。Growth 模式粗篩需 YoY > 10%。`_compute_weekly_trend_bonus()` / `_apply_weekly_trend_bonus()` 週線趨勢信號加成（Stage 3.4） |
| `src/discovery/performance.py`      | Discover 推薦績效回測（讀取歷史推薦 vs DailyPrice，計算 N 日報酬率、勝率、三層聚合統計）                           |
| `src/regime/detector.py`            | 市場狀態偵測（bull/bear/sideways），三訊號多數決（TAIEX vs SMA60/SMA120 + 20日報酬率），輸出五模式四維度權重矩陣（技術+籌碼+基本面+消息面） |
| `src/industry/analyzer.py`          | 產業輪動分析（法人動能 + 價格動能），提供 `compute_sector_scores_for_stocks()` 供 scanner 產業加成用               |
| `src/report/engine.py`              | 每日選股報告（四維度綜合評分）                                                                                    |
| `src/report/formatter.py`           | Discord 訊息格式化（2000 字元限制）                                                                               |
| `src/strategy_rank/engine.py`       | 策略排名引擎（批次回測 watchlist × strategies）                                                                   |
| `src/notification/line_notify.py`   | Discord Webhook 通知（檔名為歷史遺留）+ `format_suggest_discord()` 純函數（從 main.py 遷移）                       |
| `src/scheduler/simple_scheduler.py` | 前景排程（schedule 函式庫）                                                                                       |
| `src/scheduler/windows_task.py`     | Windows 工作排程器整合                                                                                            |
| `src/visualization/app.py`          | Streamlit 儀表板入口                                                                                              |
| `src/visualization/charts.py`       | Plotly 圖表元件                                                                                                   |
| `src/visualization/data_loader.py`  | 儀表板資料載入                                                                                                    |
| `src/visualization/pages/`          | 儀表板分頁（market_overview, stock_analysis, backtest_review, portfolio_review, strategy_comparison, screener_results, ml_analysis, industry_rotation, discovery_history, position_monitoring） |
| `main.py`                           | CLI 調度器（argparse 子命令，32 個子命令）；`_compute_revenue_scan()` 純函數；`_compute_trailing_stop()` 純函數；`detect_volume_spike`/`detect_institutional_buy`/`detect_sbl_spike`/`detect_broker_concentration` 四個籌碼異動純函數；`_compute_anomaly_scan()` 聚合函數；`cmd_anomaly_scan()` 籌碼異動警報；`cmd_morning_routine()` 早晨例行流程（含 Step 7 anomaly-scan）；`cmd_watchlist()` DB-based watchlist 管理（add/remove/list/import）；`cmd_sync_info()` 全市場股票基本資料同步（StockInfo，`--force` 強制更新） |

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

## Pending Tasks

依優先順序排列，完成後將狀態改為 ✅：

| # | 狀態 | 項目 | 說明 |
|---|------|------|------|
| 1 | ✅ | **新增 Scanner 模式（高息股 / 高成長）** | DividendScanner（殖利率>3%、PE>0）+ GrowthScanner（YoY>10%、動能確認），已完成並通過 231 測試 |
| 2 | ✅ | **Dashboard 新增 Discover 推薦歷史頁** | 視覺化 DiscoveryRecord 歷史推薦 + 績效追蹤（日曆熱圖、報酬率箱型圖、個股排行、明細 CSV 匯出），已完成 |
| 3 | ✅ | **補齊測試覆蓋** | 新增 8 個測試檔共 129 個測試（231→360），覆蓋 indicators、strategies、formatter、notification、report engine、portfolio、walk_forward、pipeline |
| 4 | ✅ | **CLI `validate` 命令（資料品質檢查）** | 6 個純函數檢查（缺漏交易日、零成交量、連續漲跌停、價格異常、日期範圍一致性、資料新鮮度），支援 CSV 匯出 |
| 5 | ✅ | **投資組合配置模式擴充** | 新增 risk_parity（風險平價）、mean_variance（均值-方差優化），allocator.py 純函數模組 + scipy 優化 |
| 6 | ✅ | **財報資料同步** | 新增季報/年報資料（EPS、ROE、毛利率、負債比、現金流），FinancialStatement ORM 表 + fetcher EAV pivot + pipeline sync + CLI sync-financial |
| 7 | ✅ | **Dashboard 市場總覽首頁** | TAIEX 走勢 + Regime 狀態、市場廣度指標、法人買賣超排名、產業熱度 Treemap，已完成 |
| 8 | ✅ | **CLI `export`/`import-data` 通用命令** | export：匯出任意資料表為 CSV/Parquet（含 --stocks/--start/--end 篩選）；import-data：從 CSV/Parquet 匯入（含欄位驗證 + --dry-run） |
| 9 | ✅ | **個股分析頁面增強** | 成交量疊加 K 線（secondary_y）、Sidebar 指標 checkbox（SMA/BB/RSI/MACD）、法人累積買賣超折線、融資融券+券資比雙列圖、MOPS 公告 vline 標記 + expander 明細表，已完成並通過 491 測試 |
| 10 | ✅ | **多模式綜合比較（discover all）** | `discover all` 一次執行五個 Scanner，輸出交叉比較表（出現越多模式 = 高信心度），支援 `--min-appearances` 篩選、CSV 匯出、Discord 通知，只修改 main.py（+`_build_cross_comparison` 純函數 + `_cmd_discover_all`），通過 491 測試 |
| 11 | ✅ | **Discover 進出場建議（Task A+D）** | `DiscoveryResult.rankings` 新增 entry_price/stop_loss/take_profit/entry_trigger/valid_until 五欄（基於 ATR14 + SMA20）；CLI 顯示 Top 5 進出場建議；Discord 通知附加進出場區塊；`DiscoveryRecord` ORM 新增對應欄位（含 migration）；507 測試通過 |
| 12 | ✅ | **`suggest` 單股進出場命令** | 新增 `python main.py suggest <stock_id>` 命令，從 DB 讀取 60 日日K，計算 ATR14/SMA20/RSI14 + Regime 偵測，輸出進場區間/止損/目標價/時機評估，可選 `--notify`；541 測試通過 |
| 13 | ✅ | **回測引擎 ATR-based 自動止損止利** | RiskConfig 新增 `atr_multiplier_stop/profit`，Engine 進場時計算並固定止損/目標價，TradeRecord 記錄 stop_price/target_price，ATR-based 優先於百分比，Trade ORM 同步新增欄位，545 測試通過 |
| 14 | ✅ | **持倉監控 Dashboard 頁面** | WatchEntry ORM 表（15 欄）+ CLI `watch add/list/close/update-status` + Dashboard「👁️ 持倉監控」頁（3 Tab：總覽/個股走勢/預警列表），`_compute_watch_status` 純函數自動標記止損/止利/過期狀態，555 測試通過 |
| 15 | ✅ | **估值 Cold-Start 修正（Value/Dividend Scanner）** | 新增 `fetch_twse_valuation_all / fetch_tpex_valuation_all / fetch_market_valuation_all`（twse_fetcher.py，TWSE BWIBBU_d + TPEX pera，免費）+ `sync_valuation_all_market()`（pipeline.py）+ Stage 0.5 機制插入 ValueScanner.run() 和 DividendScanner.run()（閾值 500 支），568 測試通過 |
| 16 | ✅ | **大戶持股分級（HoldingDistribution）** | 新增 `HoldingDistribution` ORM（schema.py）+ `fetch_holding_distribution()`（fetcher.py，FinMind TaiwanStockHoldingSharesPer）+ `sync_holding_distribution()`（pipeline.py）+ CLI `sync-holding`；`compute_whale_score()` 純函數計算大戶集中度（下限 ≥ 400,000 股）+ 週環比；整合至 MomentumScanner（5 因子 chip_score）+ SwingScanner（3 因子），588 測試通過 |
| 17 | ✅ | **MOPS 事件類型分類 + alert-check 命令（P2）** | `Announcement` ORM 新增 `event_type` 欄位（DB migration）；`mops_fetcher.py` 新增 `classify_event_type()` 純函數（earnings_call / investor_day / filing / revenue / general）+ 關鍵字群組；`_parse_announcement_html()` 自動分類；CLI `alert-check`（--days / --types / --stocks / --notify）；631 測試通過 |
| 18 | ✅ | **revenue-scan 高成長掃描命令（P3）** | `_compute_revenue_scan()` 純函數：讀取 MonthlyRevenue（最新月 YoY）+ FinancialStatement（毛利率 QoQ），篩選 YoY ≥ min_yoy 且毛利率改善，revenue_rank = YoY 70% + 毛利率 30%；CLI `revenue-scan`（--stocks / --top / --min-yoy / --min-margin-improve / --notify）；631 測試通過 |
| 19 | ✅ | **借券賣出整合（P4，TWSE TWT96U）** | 新增 `SecuritiesLending` ORM（schema.py，17 張表）；`fetch_twse_sbl()` 全市場日資料（twse_fetcher.py，免費）；`sync_sbl_all_market(days=3)`（pipeline.py）；`compute_sbl_score()` 純函數 + `_load_sbl_data()` + MomentumScanner 升級為最高 6-factor（外資22%+量比20%+法人20%+券資比13%+大戶15%+借券逆向10%）；CLI `sync-sbl`；649 測試通過 |
| 20 | ✅ | **分點進出整合（P5，FinMind TaiwanStockTradingDailyReport）** | 新增 `BrokerTrade` ORM（schema.py，18 張表）；`fetch_broker_trades()`（fetcher.py，免費逐股）；`sync_broker_trades(days=5)`（pipeline.py，跳過已有 2 日內資料）；`compute_broker_score()` 純函數計算主力集中度 HHI + 連續進場天數；`_load_broker_data()` + MomentumScanner 升級為最高 7-factor（外資20%+量比18%+法人18%+券資比11%+大戶13%+借券8%+分點12%）；CLI `sync-broker`（含 `--from-discover`）；669 測試通過 |
| 21 | ✅ | **每日早晨例行流程（morning-routine）** | `cmd_morning_routine()` 依序執行 6 步驟（sync-sbl → sync-broker --from-discover → discover all --skip-sync → alert-check --days 3 → watch update-status → revenue-scan --top 5）；`_build_morning_discord_summary()` 純函數建立 Discord 摘要（多模式選股 + 重大事件 + 持倉狀態）；CLI `morning-routine`（--dry-run / --skip-sync / --top / --notify）；673 測試通過 |
| 22 | ✅ | **動態止損追蹤（P3，Watch 升級）** | `WatchEntry` ORM 新增 `trailing_stop_enabled`/`trailing_atr_multiplier`/`highest_price_since_entry` 三欄（含 migration）；`_compute_trailing_stop()` 純函數（stop = highest - ATR14 × mult，只升不降）；`watch update-status` 每次執行時自動更新移動止損；`watch add --trailing --trailing-multiplier` CLI 旗標；Dashboard 持倉監控頁顯示移動止損類型/顏色/追蹤最高價；681 測試通過 |
| 23 | ✅ | **成交量/籌碼異動即時警報（P5）** | `detect_volume_spike`/`detect_institutional_buy`/`detect_sbl_spike`/`detect_broker_concentration` 四個純函數；`_compute_anomaly_scan()` 從 DailyPrice/InstitutionalInvestor/SecuritiesLending/BrokerTrade 四表讀取並偵測；`cmd_anomaly_scan()` + CLI `anomaly-scan`（--stocks/--lookback/--vol-mult/--inst-threshold/--sbl-sigma/--hhi-threshold/--notify）；整合進 `morning-routine` Step 7 + Discord 摘要；`tests/test_anomaly.py` 12 個純函數測試；693 測試通過 |
| 24 | ✅ | **動態 Watchlist 管理（P1）** | `Watchlist` ORM（schema.py，第 19 張表）；`get_db_watchlist()` + `get_effective_watchlist()`（DB 優先，YAML fallback）；CLI `watchlist add/remove/list/import`；所有呼叫 `settings.fetcher.watchlist` 的模組（pipeline/screener/report/strategy_rank/industry/scheduler/main.py）改為 `get_effective_watchlist()`；`tests/test_db_integration.py` 新增 7 個測試；700 測試通過 |
| 25 | ✅ | **週線多時框確認（P4）** | `aggregate_to_weekly()` 純函數（indicators.py）：日K聚合週K + 週線 SMA13/RSI14/MACD；`_compute_weekly_trend_bonus()` / `_apply_weekly_trend_bonus()` 方法（scanner.py，Stage 3.4）：SMA13 + RSI14 週線信號 → ±5% 加成；`--weekly-confirm` CLI 旗標（discover 子命令）；`tests/test_indicators.py` 新增 10 個 `TestAggregateToWeekly` 純函數測試；710 測試通過 |
| 26 | ✅ | **P6 進階回測指標測試補齊** | `test_backtest_engine.py` + `test_portfolio.py` 各新增 8 個 TestComputeMetrics 測試（Sortino/Calmar/VaR/CVaR 四指標：正常計算、None 邊界、下跌驗證、CVaR ≤ VaR 關係）；726 測試通過 |
| 27 | ✅ | **P8 sync-info 獨立 CLI 命令** | `cmd_sync_info()` + `sync-info` subparser（`--force` 旗標強制更新）；呼叫 `sync_stock_info(force_refresh=...)` 同步 StockInfo 表（產業分類 + 上市/上櫃別）；726 測試通過 |
| 28 | ✅ | **策略績效歸因分析（Factor Attribution）** | 新建 `src/backtest/attribution.py`（`FactorAttribution` 類別）；`compute(backtest_result, data)` 計算 momentum/reversal/quality/size/liquidity 五因子暴露與期間報酬相關係數；`backtest` 子命令新增 `--attribution` 旗標；Dashboard `backtest_review.py` 新增因子貢獻長條圖（Plotly）；740 測試通過 |
| 29 | ⬜ | **Claude API 整合 — AI 選股報告** | 新建 `src/report/ai_report.py`（`generate_ai_summary(discover_result, regime, top_stocks)` 函數）；呼叫 Claude API（`claude-sonnet-4-6`），傳入結構化量化數據，生成約 300 字繁體中文摘要（市場狀態 + 推薦邏輯 + 風險提示）；`discover` 子命令新增 `--ai-summary` 旗標；`requirements.txt` 新增 `anthropic`；`config/settings.yaml.example` 新增 `anthropic.api_key` |
| 30 | ✅ | **Dashboard 策略比較頁** | 新建 `src/visualization/pages/strategy_comparison.py`（第 10 個 Dashboard 頁面）；左側多選框（1~5 個策略）+ 股票選擇；3 Tab（績效指標比較 + 累積報酬率多線折線圖 + 進階指標長條圖）；`app.py` 新增「⚖️ 策略比較」sidebar 入口；`charts.py` 新增 `plot_strategy_comparison_curves()` + `plot_strategy_metrics_bar()`；740 測試通過 |
| 31 | ✅ | **Smart Broker 關鍵分點追蹤因子** | `compute_smart_broker_score()` 純函數（scanner.py）：BrokerTrade 計算 Smart Broker（win_rate≥0.60、PF≥1.50、sell_events≥3、buy_val≥500萬、Smart_Score=Σ(hist_pnl×recent_net)）+ Accumulation Broker（sell_ratio≤0.10、倉位趨勢向上），合成 `smart_broker_factor=0.60×smart+0.40×accum`；`_load_broker_data_extended(days=365, min_trading_days=20)` 自適應載入全部可用歷史（需 ≥20 交易日才啟用）；`_compute_chip_scores()` 新增 8-Factor Tier（外資18%+量比16%+法人16%+券資比10%+大戶12%+借券7%+分點HHI11%+智慧分點10%）；僅對 Stage 2 ~150 候選股計算；`tests/test_broker.py` 新增 11 個純函數測試；751 測試通過（Task 31 完成時） |
| 32 | ✅ | **Smart Broker 自適應歷史累積** | `_load_broker_data_extended()` 查詢窗口改 365 天（充分利用 daily sync 累積），新增 `min_trading_days=20` 過濾（≥20 交易日才啟用 8F）；morning-routine Step 2 拆為兩次：2a `sync-broker`（watchlist 全部，確保每日累積）+ 2b `sync-broker --from-discover`（僅非 watchlist 新發現股）；`sync-broker --watchlist-bootstrap` 新旗標供首次部署時一次性補齊最大歷史；更新 `--from-discover` 排除已在 watchlist 的股票避免重複呼叫；`TestLoadBrokerDataExtendedAdaptive` 5 個新測試；756 測試通過 |

## 已確認事項（規劃時勿重複提出）

以下項目已處理或為刻意設計，進行程式碼審查或規劃時應跳過：

- **`config/settings.yaml` 機密管理**：已在 `.gitignore` 中排除，Git 僅追蹤 `settings.yaml.example`。Token 從未進入版本控制。
- **TWSE/TPEX SSL `verify=False`**：已知的刻意行為，因部分 Windows 環境缺少 TWSE/TPEX 的根憑證鏈，停用驗證為目前的可接受方案。
- **`src/notification/line_notify.py` 檔名**：歷史遺留，實際實作為 Discord Webhook。不需重新命名，import 路徑已穩定。
- **`datetime.utcnow()` DeprecationWarning**：來自 SQLAlchemy schema 的 `default=datetime.utcnow`，為低優先級項目，不影響功能。
