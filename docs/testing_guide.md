# 測試指南

## 執行方式

```bash
pytest -v                                    # 全部測試
pytest tests/test_factors.py -v              # 單一檔案
pytest --cov=src --cov-report=term-missing   # 覆蓋率報告
```

## 測試統計

- **1761 個測試**，45 個測試檔
- Fixtures：`tests/conftest.py`（`in_memory_engine` / `db_session` / `sample_ohlcv`）
- 共用建構函數：`tests/scanner_helpers.py`

## 測試策略

| 類型 | 適用場景 | 做法 |
|------|----------|------|
| **純函數** | 計算邏輯、評分、分類 | 直接呼叫，斷言輸入輸出 |
| **DB 整合** | ORM、upsert、pipeline | in-memory SQLite + transaction rollback |
| **HTTP mock** | API 呼叫、外部資料 | mock `requests.Session.get` + `time.sleep` |

## 測試檔對照表

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

## 新增測試注意事項

- 新增計算邏輯**必須**補測試
- 優先寫純函數測試（輸入→輸出，零外部依賴）
- DB 測試使用 `conftest.py` 的 `db_session` fixture（in-memory SQLite，自動 rollback）
- Scanner 相關測試使用 `scanner_helpers.py` 的共用建構函數
- 修改模組後執行 `pytest -v` 確保全部通過
