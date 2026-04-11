# CLAUDE.md

本檔案為 Claude Code (claude.ai/code) 在此專案中的操作指南。

## 專案概述

台股量化投資系統。CLI 驅動流水線：資料擷取（FinMind API + TWSE/TPEX）→ SQLite → 技術指標 → 策略訊號 → 回測 → 報告/通知。

所有 UI 文字、註解、commit message 使用**繁體中文**。

---

## 開發規則

| 項目 | 規則 |
|------|------|
| **編碼** | 所有 Python 原始碼 UTF-8，開啟檔案務必 `encoding='utf-8'` |
| **DB 寫入** | `_upsert_batch()`，batch_size=80（SQLite 變數上限） |
| **API 速率** | FinMind 0.5 秒/次；TWSE/TPEX 3 秒/次 |
| **日期格式** | FinMind `YYYY-MM-DD`；TWSE `YYYYMMDD`；TPEX 民國曆 `YYY/MM/DD`（年 = 西元 - 1911） |
| **回測成本** | 手續費 0.1425%、交易稅 0.3%（賣出）、滑價 0.05% |
| **測試** | 純函數優先（零 mock）；DB 整合用 in-memory SQLite + transaction rollback；HTTP mock `requests.Session.get` + `time.sleep`；新增計算邏輯須補測試 |

### 提交前必執行

```bash
ruff check .    # Lint 檢查
ruff format .   # 格式化
```

### 文件聯動（程式碼修改後必更新）

修改 `src/` 或 `main.py` 後：
- **`CLAUDE.md`**：架構變更、新指令、新測試、模組職責異動時更新
- **`usage.md`**：CLI 參數變動、工作流程調整、新功能上線時更新
- 僅規劃/詢問不涉及寫入時免更新

---

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

| 模式 | 說明 |
|------|------|
| **策略註冊** | `STRATEGY_REGISTRY`（`src/strategy/__init__.py`）；9 策略：`sma_cross`/`rsi_threshold`/`bb_breakout`/`macd_cross`/`buy_and_hold`/`multi_factor`/`ml_random_forest`/`ml_xgboost`/`ml_logistic`；新策略繼承 `Strategy`，實作 `generate_signals(data) → Series[1/-1/0]` |
| **EAV 指標儲存** | `TechnicalIndicator`（stock_id, date, name, value），`load_data()` 時 pivot 為寬表 |
| **除權息回測** | Layer 1：`load_data()` 回溯調整 OHLC + 重算指標，保留 `raw_*`；Layer 2：`BacktestEngine` 原始價格交易 + 股利入帳；預設關閉，`--adjust-dividend` 啟用 |
| **SQLAlchemy Session** | `with get_session() as session:`；批次寫入 `sqlite_upsert().on_conflict_do_nothing()`；`init_db()` 含 WAL + `busy_timeout=30000` |
| **三層資料來源** | ①TWSE/TPEX 官方（免費，全市場）→ ②FinMind 批次（付費）→ ③FinMind 逐股（免費備援） |
| **Watchlist** | `get_effective_watchlist()`：DB 優先，`settings.yaml` fallback，全模組統一呼叫 |
| **常數集中** | `src/constants.py`：交易成本/速率/籌碼門檻/VIX 危機/風險預算/Kelly/相關性/新聞衰減/Regime/漲跌停/`REGIME_UNIVERSE_ADJUSTMENTS`（Universe 流動性×Regime 乘數）等全系統共用常數 |

### 模組職責

**資料層**

| 模組 | 職責 |
|------|------|
| `src/data/fetcher.py` | FinMind API（逐股/批次/財報 EAV pivot）；`fetch_us_vix()`（yfinance ^VIX）；TW_VIX graceful degradation |
| `src/data/twse_fetcher.py` | TWSE/TPEX 全市場免費資料；`fetch_twse_sbl()`（TWT96U）；`fetch_dj_broker_trades()`（DJ 端點，Big5 HTML）；`fetch_tdcc_holding_all_market()`（TDCC CSV） |
| `src/data/pipeline.py` | ETL 調度 + DB 寫入；OHLCV 品質閘門；`_classify_security_type()`；`compute_and_store_daily_features()`；`sync_broker_bootstrap()`（ThreadPoolExecutor）；`save_rotation_backtest()` |
| `src/data/mops_fetcher.py` | MOPS 重大訊息 + 月營收；`classify_event_type()`（7 類，優先序）；Regex 上下文情緒分類 |
| `src/data/schema.py` | 27 張 ORM 表（含 Announcement/DiscoveryRecord/WatchEntry/Watchlist/DailyFeature/ConceptGroup/ConceptMembership/RotationPortfolio/RotationPosition/RotationBacktest*） |
| `src/data/validator.py` | 7 個純函數品質檢查 + orchestrator（含 `check_per_stock_freshness()`） |
| `src/data/calendar.py` | TWSE 交易日行事曆（2025-2026）；`is_trading_day()` / `next_trading_day()` / `prev_trading_day()` / `get_trading_days()` |
| `src/data/io.py` | CSV/Parquet 匯出匯入（欄位驗證 + upsert） |
| `src/data/retry.py` | `request_with_retry()` exponential backoff（429/500/502/503/504） |
| `src/data/migrate.py` | DB schema 遷移工具 |

**策略/回測層**

| 模組 | 職責 |
|------|------|
| `src/strategy/base.py` | 抽象 `Strategy`：`load_data()` / `generate_signals()` / 除權息調整 / 存活者偏差偵測 |
| `src/strategy/ml_strategy.py` | ML 策略（RF/XGBoost/Logistic）+ TimeSeriesSplit 5-fold CV + Optuna 調優 + SHAP 特徵重要性 |
| `src/backtest/engine.py` | 交易模擬；T+1 訊號延遲（消除 look-ahead bias）；三因子動態滑價；流動性約束；跳空成交修正（`min(raw_open, stop_price)`）；存活者偏差偵測 |
| `src/backtest/metrics.py` | `compute_metrics()`（10 指標+爆倉防護）；`compute_trade_stats()`；`monte_carlo_equity()`（Bootstrap 1000 次，P5/P50/P95）；數值穩定性常數 |
| `src/backtest/attribution.py` | 五因子歸因（momentum/reversal/quality/size/liquidity × Pearson）；雙接口 `compute()` / `compute_from_df()` |
| `src/backtest/allocator.py` | risk_parity / mean_variance 配置（scipy） |
| `src/backtest/portfolio.py` | 多股票組合回測（equal_weight / custom / risk_parity / mean_variance） |
| `src/backtest/walk_forward.py` | Walk-Forward 滾動驗證（防過擬合）；T+1 訊號延遲（與 BacktestEngine 一致） |

**選股/Universe 層**

| 模組 | 職責 |
|------|------|
| `src/discovery/scanner/` | 五模式（Momentum/Swing/Value/Dividend/Growth）；四階段漏斗；四維度評分（技術+籌碼+基本面+消息面）+ 產業/概念/週線加成；Regime 動態權重（含 crisis 保守模式）；**技術面 Cluster 等權**（3 群：報酬動能 mean(ret5d,ret10d,sharpe_proxy) / 量能 mean(vol_ratio,vol_accel) / 突破 breakout60d，各 1/3）；**Momentum 權重 IC 校準**（bull: tech=0.40/chip=0.30/fund=0.10/news=0.20）；MomentumScanner 最高 8-factor Smart Broker；隔日沖偵測+扣分；多時框強制共振；量價背離；動態評分閾值（bull=0.45/crisis=0.60）；動量衰減；籌碼加速度；主力成本分析；勝率回饋循環（E1）；因子 IC 監控（E2）；**子因子 IC 自動權重調整**（`compute_sub_factor_weight_adjustments` + `_get_chip_base_weights` 純函數，chip 層級 IC-driven 權重微調）；MFE/MAE 分析（E3） |
| `src/discovery/universe.py` | Universe 三層漏斗：Stage 1 SQL 硬過濾 → Stage 2 流動性（DailyFeature 優先/覆蓋率≥30% 時使用，否則 fallback DailyPrice + 相對流動性救援 turnover_ratio > 2x）→ Stage 3 趨勢（同樣覆蓋率檢查；trend_only/breakout_only/trend_or_breakout 三模式；Value/Dividend 跳過）→ Candidate Memory（3 天漸進衰減）；Regime 自適應門檻（`REGIME_UNIVERSE_ADJUSTMENTS`）；`_FEATURE_COVERAGE_MIN=0.3` |
| `src/discovery/performance.py` | 推薦績效回測（N 日報酬率/勝率/Regime 分組/換手率/MFE-MAE）；`compute_strategy_decay()`（勝率<40% 或均報酬<0 觸發警告）；`--include-costs`（交易成本扣減）；`--entry-next-open`（T+1 開盤價進場）；向量化 `_calc_returns()` |
| `src/discovery/ablation.py` | 因子消融測試：維度級（歸零重分配 Spearman ρ）+ 子因子級 + 歷史績效消融；CLI `ablation-test` |
| `src/regime/detector.py` | 市場狀態（bull/bear/sideways/crisis）；三訊號多數決；市場寬度降級（>60% 跌破 MA20）；Crisis 快速覆蓋（7 訊號 ≥2 觸發）；Hysteresis 狀態機（JSON 持久化） |
| `src/industry/analyzer.py` | 產業輪動（法人+價格動能）；`compute_sector_relative_strength()`（個股 vs 同業中位數，±3%） |
| `src/industry/concept_analyzer.py` | 概念股輪動；`ConceptRotationAnalyzer` Percentile Rank；±5% 加成（scanner Stage 3.3b） |
| `src/screener/factors.py` | 8 個篩選因子（技術/籌碼/基本面） |
| `src/screener/engine.py` | 多因子篩選引擎（watchlist 內掃描） |

**進出場/組合層**

| 模組 | 職責 |
|------|------|
| `src/entry_exit.py` | 共用純函數：`REGIME_ATR_PARAMS`（bull/sideways/bear/crisis ATR 倍數）；`compute_atr_stops()`（ATR≤0 fallback 百分比）；`compute_entry_trigger()`；`assess_timing()`（RSI+SMA+Regime 決策矩陣）；Discover/Suggest/Watch 三系統共用 |
| `src/portfolio/rotation.py` | 輪動核心：`compute_rotation_actions()`（到期/續持/止損/換股 + 產業集中度 + Drawdown Guard + Portfolio Heat + Correlation Budget + Crisis 硬阻擋）；`check_drawdown_kill_switch()`（回撤≥25% 清倉）；波動率反比權重；60 日 rolling 相關性/共變異數矩陣；`compute_dynamic_slippage()`（三因子動態滑價）；`apply_liquidity_limit()`（流動性約束）；`detect_limit_price()`（漲跌停偵測）；`TradeCostBreakdown` / `compute_trade_costs()`（成本歸因）；`compute_portfolio_var()`（Ex-Ante VaR + Component VaR） |
| `src/portfolio/manager.py` | `RotationManager`：每日更新 / Kill Switch / 歷史回測（OHLCV 預載入快取+動態滑價+流動性約束+漲跌停模擬+TAIEX Benchmark+成本歸因+委託 `compute_metrics()`+每日持倉快照 `daily_positions`）/ `resolve_rankings()`（單模式 / all avg_score）/ `_get_ohlcv_on_date()` / `_get_taiex_prices()` |

**特徵/CLI/報告層**

| 模組 | 職責 |
|------|------|
| `src/features/indicators.py` | SMA/RSI/MACD/BB/ADX(14) EAV；`compute_indicators_from_df()`；`aggregate_to_weekly()`（日K→週K+SMA13/RSI14/MACD） |
| `src/features/ml_features.py` | ML 特徵矩陣（動能/波動/量比/交互特徵/Lag 5-10-20/sector_ranks）+ SHAP 特徵篩選 |
| `src/config.py` | Pydantic 設定模型；`QuantConfig`（TradingCost/AtrMultiplier/ScoreThreshold/RiskBudget 四子模型）；啟動驗證 |
| `src/cli/detection.py` | 5 個籌碼異動偵測純函數（volume_spike/institutional_buy/sbl_spike/broker_concentration/daytrade_risk） |
| `main.py` | CLI 調度器（argparse，38 子命令 + dispatch table） |
| `src/cli/helpers.py` | `safe_print()`/`setup_logging()`/`init_db()`/`ensure_sync_market_data()`/`read_stocks_from_file()` |
| `src/cli/sync.py` | sync/compute/sync-mops/sync-revenue/sync-financial/sync-info/sync-features/sync-holding/sync-vix/sync-sbl/sync-broker/alert-check |
| `src/cli/discover_cmd.py` | discover/discover-backtest/factor-diagnostics/ablation-test |
| `src/cli/backtest_cmd.py` | backtest/walk-forward（attribution/trade stats） |
| `src/cli/watch_cmd.py` | watch add/list/close/update-status；`_compute_watch_status()`/`_compute_trailing_stop()` |
| `src/cli/morning_cmd.py` | morning-routine（Step 0~15+8b）；全市場同步；TAIEX 資料新鮮度驗證 |
| `src/cli/rotation_cmd.py` | rotation create/update/status/history/backtest/list/pause/resume/delete |
| `src/cli/anomaly_cmd.py` | anomaly-scan/revenue-scan |
| `src/cli/watchlist_cmd.py` | watchlist/sync-concepts/concepts/concept-expand |
| `src/cli/suggest_cmd.py` | suggest（單股進出場建議） |
| `src/cli/misc_cmd.py` | dashboard/optimize/schedule/status/scan/notify/report/strategy-rank/industry/migrate/validate/export/import-data |
| `src/report/engine.py` | 每日選股報告（四維度綜合評分） |
| `src/report/formatter.py` | Discord 格式化（2000 字元限制） |
| `src/report/ai_report.py` | AI 選股摘要（`claude-sonnet-4-6`，`--ai-summary` 觸發） |
| `src/notification/line_notify.py` | Discord Webhook（檔名歷史遺留）+ `format_suggest_discord()` |
| `src/visualization/app.py` | Streamlit 儀表板（12 分頁） |
| `src/visualization/charts.py` | Plotly 圖表 + 純計算函數（equity curve/drawdown/heatmap/heat gauge/correlation） |
| `src/visualization/data_loader.py` | 儀表板資料載入 + 輪動組合查詢 |
| `src/visualization/pages/` | 12 分頁（market_overview/stock_analysis/backtest_review/portfolio_review/strategy_comparison/screener_results/ml_analysis/industry_rotation/concept_rotation/discovery_history/position_monitoring/risk_control） |
| `src/scheduler/simple_scheduler.py` | 前景排程；`daily_sync_job()` → `cmd_morning_routine()`；`weekly_holding_job()` 每週四 TDCC |
| `src/scheduler/windows_task.py` | Windows Task Scheduler .bat + XML 產生器 |
| `src/scheduler/launchd_task.py` | macOS LaunchAgent .sh + .plist 產生器 |

**設定**：`config/settings.yaml` → `src/config.py` Pydantic 載入。存取：`settings.finmind.api_token`、`settings.fetcher.watchlist`。FinMind token 為逐股資料必需；TWSE/TPEX 免 token，SSL `verify=False`（刻意設計）。

---

## 常用指令

```bash
pip install -r requirements.txt

# ── 資料同步 ─────────────────────────────────────────────
python main.py sync                              # watchlist OHLCV（含 TAIEX）
python main.py compute                           # 計算技術指標
python main.py sync-info                         # 全市場基本資料
python main.py sync-info --force
python main.py sync-features                     # DailyFeature（預設 90 天）
python main.py sync-features --days 60
python main.py sync-vix                          # US_VIX（yfinance ^VIX）
python main.py sync-mops                         # MOPS 重大訊息（預設 7 天）
python main.py sync-mops --days 30
python main.py sync-revenue                      # 全市場月營收
python main.py sync-revenue --months 3
python main.py sync-financial                    # watchlist 財報（4 季）
python main.py sync-financial --stocks 2330 --quarters 8
python main.py sync-holding                      # TDCC 大戶持股（全市場）
python main.py sync-sbl                          # 借券賣出（3 天）
python main.py sync-sbl --days 5
python main.py sync-broker                       # 分點資料（watchlist，5 日）
python main.py sync-broker --stocks 2330 2317 --days 10
python main.py sync-broker --from-discover
python main.py sync-broker --watchlist-bootstrap             # 首次部署（120 天）
python main.py sync-broker --watchlist-bootstrap --days 60
python main.py sync-broker --from-file stocks.txt --watchlist-bootstrap

# ── 選股掃描 ─────────────────────────────────────────────
python main.py discover momentum --top 20
python main.py discover swing --top 20
python main.py discover value --top 20
python main.py discover dividend --top 20
python main.py discover growth --top 20
python main.py discover --top 20                 # 預設 momentum
python main.py discover --skip-sync --top 10
python main.py discover --compare
python main.py discover all --skip-sync --top 20
python main.py discover all --skip-sync --min-appearances 2
python main.py discover all --skip-sync --export compare.csv
python main.py discover momentum --weekly-confirm  # 週線多時框確認
python main.py discover momentum --use-ic-adjustment  # Factor IC 動態權重調整
python main.py discover-backtest --mode momentum   # 推薦績效回測
python main.py discover-backtest --mode momentum --include-costs  # 含交易成本
python main.py discover-backtest --mode momentum --entry-next-open  # T+1 開盤進場
python main.py factor-diagnostics --mode momentum  # 因子 IC + 相關性矩陣
python main.py ablation-test --mode momentum       # 因子消融測試
python main.py ablation-test --mode momentum --with-performance  # 含歷史績效消融
python main.py ablation-test --mode momentum --skip-sync --export ablation.csv

# ── 回測 ─────────────────────────────────────────────────
python main.py backtest --stock 2330 --strategy sma_cross
python main.py backtest --stock 2330 --strategy sma_cross --attribution
python main.py backtest --stock 2330 --strategy sma_cross --export-trades trades.csv

# ── 輪動組合 ─────────────────────────────────────────────
python main.py rotation create --name mom5_3d --mode momentum --max-positions 5 --holding-days 3 --capital 1000000
python main.py rotation create --name all10_5d --mode all --max-positions 10 --holding-days 5 --capital 2000000 --no-renewal
python main.py rotation update --name mom5_3d
python main.py rotation update --all
python main.py rotation status --name mom5_3d
python main.py rotation status --all
python main.py rotation history --name mom5_3d --limit 30
python main.py rotation backtest --name mom5_3d --start 2025-01-01 --end 2025-12-31
python main.py rotation backtest --mode momentum --max-positions 5 --holding-days 3 --start 2025-01-01 --end 2025-12-31
python main.py rotation backtest --name mom5_3d --start 2025-01-01 --end 2025-12-31 --export-positions positions.csv
python main.py rotation list
python main.py rotation pause --name mom5_3d
python main.py rotation resume --name mom5_3d
python main.py rotation delete --name mom5_3d

# ── 持倉監控 ─────────────────────────────────────────────
python main.py suggest 2330
python main.py suggest 2330 --notify
python main.py watch add 2330
python main.py watch add 2330 --price 580 --stop 555 --target 635 --qty 1000
python main.py watch add 2330 --from-discover momentum
python main.py watch add 2330 --trailing
python main.py watch add 2330 --trailing --trailing-multiplier 2.0
python main.py watch list
python main.py watch list --status all
python main.py watch close 1 --price 595
python main.py watch update-status

# ── 警報與掃描 ────────────────────────────────────────────
python main.py alert-check
python main.py alert-check --days 14 --types earnings_call filing
python main.py alert-check --stocks 2330 2317 --notify
python main.py revenue-scan
python main.py revenue-scan --min-yoy 20 --min-margin-improve 1.0
python main.py revenue-scan --top 10 --notify
python main.py anomaly-scan
python main.py anomaly-scan --stocks 2330 2317
python main.py anomaly-scan --vol-mult 3.0 --inst-threshold 5000000 --dt-threshold 0.3
python main.py anomaly-scan --notify

# ── 每日例行 ─────────────────────────────────────────────
python main.py morning-routine --notify          # 完整流程（Step 0~15+8b）
python main.py morning-routine --skip-sync --notify  # 跳過 Step 1~8b
python main.py morning-routine --dry-run

# ── Watchlist / 概念股 ────────────────────────────────────
python main.py watchlist list
python main.py watchlist add 2330
python main.py watchlist add 2330 --name 台積電 --note 核心持倉
python main.py watchlist remove 2330
python main.py watchlist import
python main.py sync-concepts
python main.py sync-concepts --purge
python main.py sync-concepts --from-mops --days 30
python main.py concepts list
python main.py concepts list CoWoS封裝
python main.py concepts add CoWoS封裝 2330
python main.py concepts remove CoWoS封裝 2330
python main.py concept-expand CoWoS封裝 --threshold 0.7
python main.py concept-expand CoWoS封裝 --threshold 0.7 --auto

# ── 資料品質 / 匯出匯入 ───────────────────────────────────
python main.py validate
python main.py validate --stocks 2330 2317
python main.py validate --export issues.csv
python main.py export --list
python main.py export daily_price -o data/export/daily_price.csv
python main.py export daily_price --stocks 2330 --start 2024-01-01
python main.py export daily_price --format parquet -o data/export/dp.parquet
python main.py import-data daily_price data/export/daily_price.csv
python main.py import-data daily_price data.csv --dry-run

# ── 排程 ──────────────────────────────────────────────────
python main.py schedule                          # auto 偵測平台（Windows→Task Scheduler / macOS→LaunchAgent）
python main.py schedule --mode simple            # 前景阻塞式排程（跨平台）
python main.py schedule --mode windows           # 產生 .bat + Task Scheduler XML
python main.py schedule --mode macos             # 產生 .sh + LaunchAgent .plist

# ── 儀表板 ────────────────────────────────────────────────
python main.py dashboard              # Streamlit localhost:8501
```

### 測試

```bash
pytest -v
pytest tests/test_factors.py -v
pytest --cov=src --cov-report=term-missing
```

1750 個測試，45 個測試檔。Fixtures 在 `tests/conftest.py`（`in_memory_engine`/`db_session`/`sample_ohlcv`）；共用建構函數在 `tests/scanner_helpers.py`。

| 測試檔 | 涵蓋模組 | 類型 |
|--------|----------|------|
| `test_factors.py` | `screener/factors.py` | 純函數 |
| `test_ml_features.py` | `features/ml_features.py` | 純函數 |
| `test_ml_strategy.py` | `strategy/ml_strategy.py` | 純函數 |
| `test_backtest_engine.py` | `backtest/engine.py` + `metrics.py` | 純函數+mock |
| `test_twse_helpers.py` | `data/twse_fetcher.py` | 純函數 |
| `test_scanner.py` | `discovery/scanner/` 全模組 | 純函數 |
| `test_mops.py` | `mops_fetcher.py` + scanner 消息面 | 純函數+SQLite |
| `test_regime.py` | `regime/detector.py` | 純函數 |
| `test_fetcher.py` | `data/fetcher.py` | mock HTTP |
| `test_config.py` | `config.py` | tmp_path |
| `test_charts.py` | `visualization/charts.py` | 純函數 |
| `test_dividend_adjustment.py` | 除權息還原 | 純函數+mock |
| `test_discover_performance.py` | `discovery/performance.py` | SQLite+純函數 |
| `test_db_integration.py` | ORM/upsert/pipeline/Watchlist | SQLite |
| `test_indicators.py` | `features/indicators.py` | 純函數 |
| `test_strategies.py` | 6 個策略 generate_signals() | 純函數 |
| `test_formatter.py` | `report/formatter.py` | 純函數 |
| `test_notification.py` | `notification/line_notify.py` | 純函數+mock |
| `test_report_engine.py` | `report/engine.py` | 純函數 |
| `test_portfolio.py` | `backtest/portfolio.py` | 純函數+mock |
| `test_walk_forward.py` | `backtest/walk_forward.py` | 純函數+mock |
| `test_pipeline.py` | `data/pipeline.py` | SQLite |
| `test_allocator.py` | `backtest/allocator.py` | 純函數+scipy |
| `test_validator.py` | `data/validator.py` | 純函數 |
| `test_financial.py` | `data/fetcher.py` 財報 EAV | mock+SQLite |
| `test_market_overview.py` | `data_loader` + charts | SQLite+純函數 |
| `test_io.py` | `data/io.py` | SQLite+純函數 |
| `test_entry_exit.py` | `entry_exit.py` | 純函數 |
| `test_suggest.py` | indicators + entry_exit + notify | 純函數 |
| `test_watch.py` | `cli/watch_cmd.py` + WatchEntry ORM | 純函數+SQLite |
| `test_holding.py` | HoldingDistribution + `compute_whale_score` | SQLite+mock |
| `test_alert.py` | `classify_event_type` + `_compute_revenue_scan` | 純函數+SQLite |
| `test_sbl.py` | SBL ORM + `compute_sbl_score` + MomentumScanner | SQLite+mock |
| `test_broker.py` | BrokerTrade ORM + Smart Broker + Bootstrap | SQLite+mock |
| `test_anomaly.py` | `cli/detection.py` 5 個偵測函數 | 純函數 |
| `test_attribution.py` | `FactorAttribution` 五因子 | 純函數 |
| `test_universe.py` | `discovery/universe.py` + DailyFeature ETL | 純函數+SQLite |
| `test_concepts.py` | `classify_concepts()` + ConceptGroup ORM | 純函數+SQLite |
| `test_rotation.py` | `portfolio/rotation.py` 全模組 | 純函數+SQLite |
| `test_calendar.py` | `data/calendar.py` TWSE 交易日行事曆 | 純函數 |
| `test_morning_atomicity.py` | `cli/morning_cmd.py` 原子性 | mock |
| `test_ablation.py` | `discovery/ablation.py` 因子消融 | 純函數 |
| `test_scheduler.py` | `scheduler/` launchd + windows + auto 偵測 | 純函數+mock |

新增或修改模組後，執行 `pytest -v` 確保全部通過，並為新計算邏輯補充測試。

---

## Pending Tasks（未完成）

（Phase 1 + Phase 2 全部完成，Phase 3 實盤上線暫緩）

## 已確認事項（規劃時勿重複提出）

- `config/settings.yaml` 已在 `.gitignore`，token 從未進入 Git
- TWSE/TPEX `verify=False`：刻意設計（Windows 憑證問題）
- `src/notification/line_notify.py`：歷史遺留檔名，實為 Discord Webhook，不需重命名
- `datetime.utcnow()` DeprecationWarning：SQLAlchemy schema default，低優先級不影響功能

## Completed Tasks（已完成，共 83 項）

測試數從 231 → 1750。各階段摘要：

| 階段 | Task # | 重點 |
|------|--------|------|
| **基礎建設** | 1~10 | 五模式 Scanner、Dashboard、validate/export/import CLI |
| **進出場 + 持倉** | 11~14 | ATR14+SMA20 建議、suggest、ATR 止損止利、watch 監控 |
| **資料整合** | 15~20 | TWSE/TPEX Cold-Start、TDCC 大戶、MOPS 事件、SBL、DJ 分點 |
| **自動化 + 監控** | 21~24 | morning-routine、Trailing Stop、anomaly-scan、DB watchlist |
| **評分升級** | 25~41 | 週線多時框、進階回測指標、五因子歸因、Smart Broker、Bootstrap、AI 摘要、Regime 自適應 |
| **Universe + 品質** | 42~49 | Universe 三層漏斗 + Feature Store、同業 PE、EPS 連續性、消息面優化、Swing 強化 |
| **概念 + Crisis** | 50~53 | 概念股系統、Crisis 第四狀態、隔日沖偵測、風險過濾強化 |
| **統一重構** | 54~58 | entry_exit.py 共用、市場寬度降級、Hysteresis 狀態機、法人連續性 + HHI 趨勢 |
| **Phase 1 基盤強化** | 59~67 | Regime 預設值、Kelly 收縮、Drawdown 連續化、相關性危機自適應、分數上限修正、早晨原子性、每股新鮮度、波動率權重、危機強制平倉 |
| **Phase 2 擬真度** | 68~75 | 假日行事曆、公告衰減分化、籌碼層級稽核、滑價不對稱、Factor IC 動態權重、Regime 部位大小、漲跌停模擬、部分止利 |
| **Universe 強化** | 76 | Regime 自適應門檻、min_close 軟化（Momentum/Growth=5）、相對流動性救援（turnover_ratio>2x）、突破型過濾器（Type B）、Candidate Memory 3 天漸進衰減 |
| **因子權重優化** | 77 | 技術面 Cluster 等權（3 群 mean 降維）、Momentum Regime 權重 IC 校準（chip 降權/news 升權）、morning-routine 啟用 IC 動態調整 |
| **跨平台排程** | 78 | macOS LaunchAgent 排程（`launchd_task.py` .sh+.plist）、`--mode auto` 平台自動偵測、Windows 既有流程不變（+11 測試） |
| **Rotation 回測擬真度** | 79 | 三因子動態滑價（`compute_dynamic_slippage`）、流動性約束（`apply_liquidity_limit`）、漲跌停模擬（`detect_limit_price`）、成本歸因（`TradeCostBreakdown`）、TAIEX Benchmark+Alpha、委託 `compute_metrics()`（Sortino/Calmar/VaR/CVaR/PF）、Schema 遷移 11 欄位（+28 測試） |
| **Ex-Ante VaR** | 80 | `compute_covariance_matrix()`（共變異數矩陣+ridge正則化）、`compute_portfolio_var()`（參數化VaR+Component VaR分解）、backtest 每日 VaR 記錄、update() VaR 日誌（+11 測試） |
| **子因子 IC 自動化** | 81 | `compute_sub_factor_weight_adjustments()`（子因子 IC 權重調整+min_samples 防護）、`_get_chip_base_weights()`（15 分支→純函數）、`_compute_chip_scores()` 重構整合 IC 調整、`_load_chip_sub_factor_ic()` DB 歷史 IC 載入+graceful degradation（+11 測試） |
| **Discover+Rotation 效能強化** | 82 | Rotation 回測 OHLCV 預載入快取（消除 N×M 逐日查詢）、Discover 回測交易成本扣減（`--include-costs`）、T+1 開盤價進場（`--entry-next-open`）、`_calc_returns()` 向量化（iterrows→merge+join）、Rotation 回測每日持倉快照（`--export-positions` CSV）（+11 測試） |
| **Universe 覆蓋率 fallback** | 83 | `_stage2_liquidity_filter` / `_stage3_trend_filter` DailyFeature 覆蓋率檢查（`_FEATURE_COVERAGE_MIN=0.3`），覆蓋不足時 fallback DailyPrice 全市場計算；修復 DailyFeature 僅覆蓋 watchlist 時 Universe 從 1930→6 支的瓶頸（修復後 Stage 2=745/Stage 3=469）（+1 測試） |
