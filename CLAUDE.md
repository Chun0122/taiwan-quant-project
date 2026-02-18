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
python main.py sync                          # 從 FinMind 同步觀察清單資料
python main.py compute                       # 計算技術指標
python main.py backtest --stock 2330 --strategy sma_cross
python main.py discover --top 20             # 全市場掃描（TWSE/TPEX）
python main.py discover --skip-sync --top 10 # 使用已快取的 DB 資料
python main.py dashboard                     # Streamlit 儀表板（localhost:8501）
```

目前無正式測試套件。變更後可用語法檢查與匯入測試驗證：
```bash
python -c "import ast; ast.parse(open('path/to/file.py', encoding='utf-8').read())"
python -c "from src.module import ClassName; print('OK')"
```

## 架構

### 資料流程
```
FinMind API / TWSE+TPEX ──→ Pipeline (ETL) ──→ SQLite DB
                                                    │
Strategy.load_data() ← 寬表（OHLCV + 指標合併）
         │
    generate_signals() → BacktestEngine.run() → BacktestResult → DB
```

### 核心設計模式

**策略註冊機制** (`src/strategy/__init__.py`)：所有策略註冊於 `STRATEGY_REGISTRY` 字典，CLI 依名稱解析策略。新增策略方式：繼承 `Strategy`，實作 `generate_signals(data) → Series[1/-1/0]`，加入註冊表。

**EAV 指標儲存** (`src/data/schema.py:TechnicalIndicator`)：技術指標採用 Entity-Attribute-Value 模式（stock_id, date, name, value），於 `Strategy.load_data()` 時樞紐轉換為寬表。

**SQLAlchemy Session** (`src/data/database.py`)：一律使用 `with get_session() as session:` 上下文管理器。批次寫入使用 `sqlite_upsert().on_conflict_do_nothing()`。DB 操作前需呼叫 `init_db()`。

**三層資料來源策略**（`src/data/pipeline.py:sync_market_data`）：
1. TWSE/TPEX 官方開放資料（免費，4 次 API 呼叫取得全市場 ~6000 支股票）
2. FinMind 批次 API（付費帳號）
3. FinMind 逐股擷取（免費備援，速度較慢）

### 模組職責

| 模組 | 角色 |
|------|------|
| `src/data/fetcher.py` | FinMind API 封裝（逐股 + 批次） |
| `src/data/twse_fetcher.py` | TWSE/TPEX 官方資料（全市場、免費） |
| `src/data/pipeline.py` | ETL 調度、寫入 DB |
| `src/data/schema.py` | 9 張 SQLAlchemy ORM 資料表 |
| `src/features/indicators.py` | SMA/RSI/MACD/BB → EAV 格式 |
| `src/features/ml_features.py` | ML 特徵矩陣（動能、波動度、量比） |
| `src/strategy/base.py` | 抽象 `Strategy`：`load_data()` + `generate_signals()` |
| `src/backtest/engine.py` | 交易模擬、風險管理、部位控管 |
| `src/backtest/portfolio.py` | 多股票組合回測 |
| `src/discovery/scanner.py` | 四階段漏斗：~6000 → 粗篩 150 → 評分 → Top N |
| `src/report/formatter.py` | Discord 訊息格式化（2000 字元限制） |
| `main.py` | CLI 調度器（argparse 子命令） |

### 設定

`config/settings.yaml` 透過 `src/config.py` 中的 Pydantic 模型載入。全域存取方式：`settings.finmind.api_token`、`settings.fetcher.watchlist` 等。

**注意**：FinMind API token 為逐股資料所必需。TWSE/TPEX 端點無需 token。因部分系統憑證問題，TWSE/TPEX 的 SSL 驗證已停用（`verify=False`）。

## 開發慣例

- **檔案編碼**：所有原始碼為 UTF-8 含中文內容。在 Windows 上開啟檔案時務必指定 `encoding='utf-8'`。
- **DB 寫入**：使用 `_upsert_batch()`，batch_size=80 以符合 SQLite 變數上限。
- **API 速率控制**：FinMind 每次請求間隔 0.5 秒，TWSE/TPEX 間隔 3 秒。
- **日期格式**：FinMind 使用 ISO 格式（`YYYY-MM-DD`）；TWSE 使用 `YYYYMMDD`；TPEX 使用民國曆（`YYY/MM/DD`，年 = 西元年 - 1911）。
- **回測成本**：手續費 0.1425%、交易稅 0.3%（賣出時）、滑價 0.05%。
