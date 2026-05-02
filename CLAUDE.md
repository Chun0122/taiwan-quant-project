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
| **回測成本** | 手續費 0.1425%、交易稅 0.3%（賣出）、滑價 0.05% |
| **Session** | `with get_session() as session:`；批次寫入 `sqlite_upsert().on_conflict_do_nothing()` |
| **常數** | 全系統共用常數集中於 `src/constants.py`，勿在各模組硬編碼 |
| **設定** | `config/settings.yaml` → `src/config.py` Pydantic 載入（4 子模型 + 啟動驗證） |
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
| `data/pipeline.py` | ETL 調度 + DB 寫入、OHLCV 品質閘門、DailyFeature 計算、Broker Bootstrap |
| `data/mops_fetcher.py` | MOPS 重大訊息 + 月營收、事件分類（7 類）、情緒分類 |
| `data/schema.py` | 27 張 ORM 表 |
| `data/validator.py` | 7 個品質檢查純函數 |
| `data/calendar.py` | TWSE 交易日行事曆（2025-2026） |
| `data/io.py` | CSV/Parquet 匯出匯入（欄位驗證 + upsert） |
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
| `discovery/scanner/` | 五模式選股（Momentum/Swing/Value/Dividend/Growth）、四維度評分、Regime 動態權重 |
| `discovery/universe.py` | Universe 三層漏斗（SQL→流動性→趨勢）+ Candidate Memory |
| `discovery/performance.py` | 推薦績效回測、策略衰減警告 |
| `discovery/ablation.py` | 因子消融測試（維度級 + 子因子級 + 績效消融） |
| `discovery/strategy_events.py` | 策略調整事件抽取（git log + settings.yaml diff，供 dashboard 事件流） |
| `regime/detector.py` | 市場狀態（bull/bear/sideways/crisis）、Hysteresis 狀態機 |
| `industry/analyzer.py` | 產業輪動、同業相對強度（±3%） |
| `industry/concept_analyzer.py` | 概念股輪動、Percentile Rank（±5% 加成） |
| `screener/` | 多因子篩選引擎（8 因子，watchlist 內掃描） |

**進出場/組合層**

| 模組 | 職責 |
|------|------|
| `entry_exit.py` | 共用純函數：ATR 止損止利、進場觸發、時機評估（Discover/Suggest/Watch 三系統共用） |
| `portfolio/rotation.py` | 輪動核心：換股 + 風控（Drawdown Guard/Portfolio Heat/Correlation/VaR） |
| `portfolio/manager.py` | RotationManager：每日更新 / Kill Switch / 歷史回測 |

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
| `features/ml_features.py` | ML 特徵矩陣 + SHAP 篩選 |
| `config.py` | Pydantic 設定模型（`QuantConfig`：TradingCost/AtrMultiplier/ScoreThreshold/RiskBudget） |

---

## 4. 核心設計模式（新增/修改時必須遵循）

| 模式 | 規則 |
|------|------|
| **策略註冊** | `STRATEGY_REGISTRY`（`src/strategy/__init__.py`）；9 策略；新策略繼承 `Strategy`，實作 `generate_signals(data) → Series[1/-1/0]` |
| **EAV 指標** | `TechnicalIndicator`（stock_id, date, name, value），`load_data()` pivot 為寬表 |
| **除權息** | Layer 1 回溯調整 OHLC + 重算指標（保留 `raw_*`）；Layer 2 原始價格交易 + 股利入帳；預設關閉，`--adjust-dividend` 啟用 |
| **Watchlist** | `get_effective_watchlist()`：DB 優先，`settings.yaml` fallback，全模組統一呼叫 |
| **Universe 漏斗** | Stage 1 SQL 硬過濾 → Stage 2 流動性（DailyFeature 優先/覆蓋率≥30% 時使用，否則 fallback DailyPrice + 相對流動性救援）→ Stage 3 趨勢（Value/Dividend 跳過）→ Candidate Memory（3 天漸進衰減）；Regime 自適應門檻（`REGIME_UNIVERSE_ADJUSTMENTS`） |
| **Regime 四狀態** | bull/bear/sideways/crisis；三訊號多數決 + 市場寬度降級 + Crisis 快速覆蓋；影響：選股權重、評分閾值（bull=0.45/crisis=0.60）、ATR 倍數、Universe 門檻、部位大小 |
| **Scanner 評分** | 四維度（技術+籌碼+基本面+消息面）；技術面 3 Cluster 等權 v2（報酬動能/量能/突破，各 1/3）；零方差因子自動排除（`exclude_zero_variance_factors`）；子因子 IC 自動權重調整；Rolling IC + Per-Regime IC 監控 |
| **輪動風控** | Drawdown Kill Switch（≥25% 清倉）、Portfolio Heat、Correlation Budget（60 日 rolling）、Crisis 硬阻擋、Ex-Ante VaR（Component VaR 分解） |
| **T+1 延遲** | BacktestEngine + Walk-Forward + Discover 回測一致執行訊號延遲，消除 look-ahead bias |
| **動態滑價** | 三因子模型（`compute_dynamic_slippage`）；流動性約束（`apply_liquidity_limit`）；漲跌停偵測（`detect_limit_price`） |

---

## 5. 開發流程與文件聯動

### 文件聯動規則

修改 `src/` 或 `main.py` 後：
- **`CLAUDE.md`**：架構變更、新指令、新測試、模組職責異動時更新
- **`usage.md`**：CLI 參數變動、工作流程調整、新功能上線時更新
- 僅規劃/詢問不涉及寫入時免更新

### CLI 設計原則

- 入口：`python main.py <子命令>`（39 子命令，dispatch table 在 `main.py`）
- 每日例行：`morning-routine`（Step 0~15+8b，含全市場同步 + discover + 風控 + 通知）
- 新增子命令須更新 `main.py` dispatch table + `docs/cli_commands.md`
- 完整指令參考見 [`docs/cli_commands.md`](docs/cli_commands.md)

---

## 6. 測試規範

- **策略**：純函數優先（零 mock）；DB 整合用 in-memory SQLite + transaction rollback；HTTP mock `requests.Session.get` + `time.sleep`
- **要求**：新增計算邏輯**必須**補測試
- **執行**：`pytest -v`（1784 測試 / 47 檔）
- **Fixtures**：`tests/conftest.py`（`in_memory_engine`/`db_session`/`sample_ohlcv`）；共用建構函數 `tests/scanner_helpers.py`
- 詳細測試檔對照表見 [`docs/testing_guide.md`](docs/testing_guide.md)

---

## 7. 外部文件索引

| 文件 | 內容 |
|------|------|
| [`docs/cli_commands.md`](docs/cli_commands.md) | 38 個子命令完整用法與範例 |
| [`docs/testing_guide.md`](docs/testing_guide.md) | 45 個測試檔對照表、Fixtures、覆蓋率指引 |
| [`docs/project_history.md`](docs/project_history.md) | 85 項已完成任務歷史（Phase 1~2） |
| `usage.md` | 使用者導向操作手冊 |
| `config/settings.yaml` | 執行期設定（`.gitignore` 已排除） |

---

## 8. 已確認事項（規劃時勿重複提出）

- `config/settings.yaml` 已在 `.gitignore`，token 從未進入 Git
- TWSE/TPEX `verify=False`：刻意設計（Windows 憑證問題）
- `src/notification/line_notify.py`：歷史遺留檔名，實為 Discord Webhook，不需重命名
- `datetime.utcnow()` DeprecationWarning：SQLAlchemy schema default，低優先級不影響功能
- FinMind token 為逐股資料必需；TWSE/TPEX 免 token

---

## 9. 專案狀態

- Phase 1 + Phase 2 **全部完成**（85 項，測試 231→1761）
- Phase 3 實盤上線**暫緩**
- 無 Pending Tasks
