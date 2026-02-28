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
python main.py discover --top 20             # 全市場掃描（預設 momentum 模式）
python main.py discover momentum --top 20   # 短線動能掃描
python main.py discover swing --top 20      # 中期波段掃描
python main.py discover value --top 20      # 價值修復掃描
python main.py discover dividend --top 20   # 高息存股掃描
python main.py discover growth --top 20     # 高成長掃描
python main.py discover --skip-sync --top 10 # 使用已快取的 DB 資料
python main.py discover --compare            # 顯示與上次推薦的差異比較
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
```

### 測試

使用 pytest 測試框架，~458 個測試覆蓋核心模組：

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
| `tests/test_backtest_engine.py` | `src/backtest/engine.py` 回測計算                | 純函數 + mock Strategy |
| `tests/test_twse_helpers.py`    | `src/data/twse_fetcher.py` 工具函數              | 純函數                 |
| `tests/test_scanner.py`         | `src/discovery/scanner.py` 基底+Momentum+Swing+Value+Dividend+Growth 六類掃描 + 產業加成 | 純函數                 |
| `tests/test_mops.py`            | `mops_fetcher.py` 情緒分類 + 月營收解析 + `scanner.py` 消息面評分 + Announcement ORM + 權重矩陣 | 純函數 + in-memory SQLite |
| `tests/test_regime.py`          | `src/regime/detector.py` 市場狀態偵測 + 權重矩陣 | 純函數                 |
| `tests/test_fetcher.py`         | `src/data/fetcher.py` API 封裝                   | mock HTTP              |
| `tests/test_config.py`          | `src/config.py` 設定載入                         | tmp_path               |
| `tests/test_dividend_adjustment.py` | 除權息還原（價格調整 + 指標重算 + 回測股利入帳） | 純函數 + mock Strategy |
| `tests/test_discover_performance.py` | `src/discovery/performance.py` 推薦績效回測    | in-memory SQLite       |
| `tests/test_db_integration.py`  | ORM + upsert + pipeline + DiscoveryRecord        | in-memory SQLite       |
| `tests/test_indicators.py`     | `src/features/indicators.py` compute_indicators_from_df() | 純函數           |
| `tests/test_strategies.py`     | 6 個策略 generate_signals() + 除權息調整          | 純函數                 |
| `tests/test_formatter.py`      | `src/report/formatter.py` 4 個格式化函數          | 純函數                 |
| `tests/test_notification.py`   | `src/notification/line_notify.py` 通知模組        | 純函數 + mock HTTP     |
| `tests/test_report_engine.py`  | `src/report/engine.py` 4 個 _compute_* 評分函數   | 純函數                 |
| `tests/test_portfolio.py`      | `src/backtest/portfolio.py` 組合回測              | 純函數 + mock Strategy |
| `tests/test_walk_forward.py`   | `src/backtest/walk_forward.py` Walk-Forward 驗證  | 純函數 + mock Strategy |
| `tests/test_pipeline.py`       | `src/data/pipeline.py` ETL 函數                   | in-memory SQLite       |
| `tests/test_allocator.py`      | `src/backtest/allocator.py` risk_parity + mean_variance 權重計算 + fallback | 純函數 + scipy         |
| `tests/test_validator.py`      | `src/data/validator.py` 6 個品質檢查純函數         | 純函數                 |
| `tests/test_financial.py`     | `src/data/fetcher.py` 財報 EAV pivot + 衍生比率 + pipeline upsert | 純函數 + mock API + in-memory SQLite |
| `tests/test_market_overview.py` | `data_loader` 市場總覽查詢 + `charts` 4 個圖表函數 | in-memory SQLite + 純函數 |

共用 fixtures 在 `tests/conftest.py`：`in_memory_engine`（session scope）、`db_session`（function scope，transaction rollback 隔離）、`sample_ohlcv`。

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

**SQLAlchemy Session** (`src/data/database.py`)：一律使用 `with get_session() as session:` 上下文管理器。批次寫入使用 `sqlite_upsert().on_conflict_do_nothing()`。DB 操作前需呼叫 `init_db()`。

**三層資料來源策略**（`src/data/pipeline.py:sync_market_data`）：

1. TWSE/TPEX 官方開放資料（免費，6 次 API 呼叫取得全市場 ~6000 支股票，含融資融券）
2. FinMind 批次 API（付費帳號）
3. FinMind 逐股擷取（免費備援，速度較慢）

### 模組職責

| 模組                                | 角色                                                                                                              |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `src/data/fetcher.py`               | FinMind API 封裝（逐股 + 批次 + 財報三表 EAV pivot）                                                              |
| `src/data/twse_fetcher.py`          | TWSE/TPEX 官方資料（全市場、免費）                                                                                |
| `src/data/pipeline.py`              | ETL 調度、寫入 DB                                                                                                 |
| `src/data/mops_fetcher.py`          | MOPS 公開資訊觀測站（重大訊息 + 全市場月營收，免費）                                                              |
| `src/data/schema.py`                | 14 張 SQLAlchemy ORM 資料表（含 Announcement、DiscoveryRecord、FinancialStatement）                               |
| `src/data/validator.py`             | 資料品質檢查（6 個純函數檢查 + orchestrator + console 報告）                                                       |
| `src/data/migrate.py`               | DB schema 遷移工具                                                                                                |
| `src/config.py`                     | Pydantic 設定模型 + `load_settings()`                                                                             |
| `src/features/indicators.py`        | SMA/RSI/MACD/BB → EAV 格式 + `compute_indicators_from_df()` 純函數（除權息還原用）                                |
| `src/features/ml_features.py`       | ML 特徵矩陣（動能、波動度、量比）                                                                                 |
| `src/strategy/base.py`              | 抽象 `Strategy`：`load_data()` + `generate_signals()` + 除權息調整（`_apply_dividend_adjustment`）                |
| `src/strategy/__init__.py`          | `STRATEGY_REGISTRY`（9 個策略）                                                                                   |
| `src/strategy/ml_strategy.py`       | ML 策略（Random Forest / XGBoost / Logistic）                                                                     |
| `src/backtest/engine.py`            | 交易模擬、風險管理、部位控管、除權息股利入帳                                                                      |
| `src/backtest/allocator.py`         | 投資組合配置計算（risk_parity / mean_variance），scipy 優化純函數                                                 |
| `src/backtest/portfolio.py`         | 多股票組合回測，支援 equal_weight / custom / risk_parity / mean_variance 四種配置                                 |
| `src/backtest/walk_forward.py`      | Walk-Forward 滾動窗口驗證（防過擬合）                                                                             |
| `src/optimization/grid_search.py`   | Grid Search 參數優化器                                                                                            |
| `src/screener/factors.py`           | 8 個篩選因子（技術面/籌碼面/基本面）                                                                              |
| `src/screener/engine.py`            | 多因子篩選引擎（watchlist 內掃描）                                                                                |
| `src/discovery/scanner.py`          | 全市場四階段漏斗（含風險過濾），支援 Momentum / Swing / Value / Dividend / Growth 五模式，四維度評分（技術+籌碼+基本面+消息面）+ 產業熱度加成（±5%），權重依 Regime 動態調整。Value/Dividend 模式粗篩為嚴格模式：必須有估值資料。Growth 模式粗篩需 YoY > 10% |
| `src/discovery/performance.py`      | Discover 推薦績效回測（讀取歷史推薦 vs DailyPrice，計算 N 日報酬率、勝率、三層聚合統計）                           |
| `src/regime/detector.py`            | 市場狀態偵測（bull/bear/sideways），三訊號多數決（TAIEX vs SMA60/SMA120 + 20日報酬率），輸出五模式四維度權重矩陣（技術+籌碼+基本面+消息面） |
| `src/industry/analyzer.py`          | 產業輪動分析（法人動能 + 價格動能），提供 `compute_sector_scores_for_stocks()` 供 scanner 產業加成用               |
| `src/report/engine.py`              | 每日選股報告（四維度綜合評分）                                                                                    |
| `src/report/formatter.py`           | Discord 訊息格式化（2000 字元限制）                                                                               |
| `src/strategy_rank/engine.py`       | 策略排名引擎（批次回測 watchlist × strategies）                                                                   |
| `src/notification/line_notify.py`   | Discord Webhook 通知（檔名為歷史遺留）                                                                            |
| `src/scheduler/simple_scheduler.py` | 前景排程（schedule 函式庫）                                                                                       |
| `src/scheduler/windows_task.py`     | Windows 工作排程器整合                                                                                            |
| `src/visualization/app.py`          | Streamlit 儀表板入口                                                                                              |
| `src/visualization/charts.py`       | Plotly 圖表元件                                                                                                   |
| `src/visualization/data_loader.py`  | 儀表板資料載入                                                                                                    |
| `src/visualization/pages/`          | 儀表板分頁（market_overview, stock_analysis, backtest_review, portfolio_review, screener_results, ml_analysis, industry_rotation, discovery_history） |
| `main.py`                           | CLI 調度器（argparse 子命令）                                                                                     |

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
| 8 | ⬜ | **CLI `export`/`import` 通用命令** | export：匯出任意資料表為 CSV/Parquet；import：從 CSV 批次匯入自訂資料（含驗證） |
| 9 | ⬜ | **個股分析頁面增強** | 成交量柱狀圖疊加 K 線、技術指標可勾選疊加、法人買賣超累積圖、融資融券走勢、MOPS 公告時間軸 |

## 已確認事項（規劃時勿重複提出）

以下項目已處理或為刻意設計，進行程式碼審查或規劃時應跳過：

- **`config/settings.yaml` 機密管理**：已在 `.gitignore` 中排除，Git 僅追蹤 `settings.yaml.example`。Token 從未進入版本控制。
- **TWSE/TPEX SSL `verify=False`**：已知的刻意行為，因部分 Windows 環境缺少 TWSE/TPEX 的根憑證鏈，停用驗證為目前的可接受方案。
- **`src/notification/line_notify.py` 檔名**：歷史遺留，實際實作為 Discord Webhook。不需重新命名，import 路徑已穩定。
- **`datetime.utcnow()` DeprecationWarning**：來自 SQLAlchemy schema 的 `default=datetime.utcnow`，為低優先級項目，不影響功能。
