# 台股量化投資系統 — 操作手冊

## 1. 環境安裝

### 前置需求

- Python 3.10+
- pip

### 安裝步驟

```bash
# 1. 安裝 Python 套件
pip install -r requirements.txt
```

### FinMind API Token

系統使用 FinMind 作為資料來源，需要免費申請 API Token：

1. 前往 [FinMind](https://finmindtrade.com/) 註冊帳號
2. 登入後至「API Token」頁面取得 Token
3. 將 Token 填入 `config/settings.yaml`：

```yaml
finmind:
  api_token: "你的 token"
```

---

## 2. 專案結構

```
taiwan-quant-project/
├── main.py                  # CLI 主程式入口
├── requirements.txt         # Python 套件清單
├── config/
│   └── settings.yaml        # 系統設定檔
├── data/
│   ├── stock.db             # SQLite 資料庫
│   ├── raw/                 # 原始資料（保留）
│   └── processed/           # 處理後資料（保留）
├── src/
│   ├── config.py            # 設定管理模組
│   ├── data/
│   │   ├── database.py      # 資料庫引擎與 Session 管理
│   │   ├── schema.py        # ORM 資料表定義（含投資組合表）
│   │   ├── fetcher.py       # FinMind API 資料抓取
│   │   ├── pipeline.py      # ETL Pipeline（抓取→清洗→寫入）
│   │   └── migrate.py       # SQLite Schema 遷移
│   ├── features/
│   │   ├── indicators.py   # 技術指標計算引擎（SMA/RSI/MACD/BB）
│   │   └── ml_features.py  # ML 特徵工程
│   ├── strategy/
│   │   ├── base.py          # 策略抽象基類
│   │   ├── sma_cross.py     # SMA 均線交叉策略
│   │   ├── rsi_threshold.py # RSI 超買超賣策略
│   │   ├── bb_breakout.py   # 布林通道突破策略
│   │   ├── macd_cross.py    # MACD 交叉策略
│   │   ├── buy_hold.py      # 買入持有基準策略
│   │   ├── multi_factor.py  # 多因子組合策略
│   │   └── ml_strategy.py   # ML 機器學習策略（RF/XGBoost/Logistic）
│   ├── screener/
│   │   ├── factors.py       # 因子定義（8 個因子）
│   │   └── engine.py        # 多因子篩選引擎
│   ├── notification/
│   │   └── line_notify.py   # Discord Webhook 通知
│   ├── backtest/
│   │   ├── engine.py        # 回測引擎 + 風險管理 + 績效計算
│   │   ├── portfolio.py     # 投資組合回測引擎
│   │   └── walk_forward.py  # Walk-Forward 滾動驗證引擎
│   ├── optimization/
│   │   └── grid_search.py   # Grid Search 參數優化
│   ├── scheduler/
│   │   ├── simple_scheduler.py  # 前景排程器
│   │   └── windows_task.py      # Windows Task Scheduler 產生器
│   └── visualization/
│       ├── app.py            # Streamlit 儀表板入口
│       ├── data_loader.py    # 資料查詢模組
│       ├── charts.py         # Plotly 圖表元件
│       └── pages/            # 頁面模組
│           ├── stock_analysis.py    # 個股分析頁
│           ├── backtest_review.py   # 回測結果頁
│           ├── screener_results.py  # 選股篩選頁
│           ├── portfolio_review.py  # 投資組合頁
│           └── ml_analysis.py       # ML 策略分析頁
├── notebooks/               # Jupyter 分析筆記本
└── tests/                   # 測試
```

---

## 3. 設定檔說明

設定檔位於 `config/settings.yaml`：

```yaml
# FinMind API 設定
finmind:
  api_url: "https://api.finmindtrade.com/api/v4/data"
  api_token: "你的 token"

# 資料庫設定（預設使用 SQLite）
database:
  url: "sqlite:///data/stock.db"

# 資料抓取參數
fetcher:
  default_start_date: "2020-01-01"   # 歷史資料起始日
  watchlist:                          # 預設關注股票
    - "2330"   # 台積電
    - "2317"   # 鴻海
    - "2454"   # 聯發科
    - "2308"   # 台達電
    - "2412"   # 中華電

# 日誌等級（DEBUG / INFO / WARNING / ERROR）
logging:
  level: "INFO"
```

若要新增關注股票，在 `watchlist` 下加入股票代號即可。

---

## 4. CLI 指令

所有操作透過 `main.py` 執行：

### 4.1 同步資料 (`sync`)

從 FinMind API 抓取資料並寫入本地資料庫。支援增量更新 — 已有資料會自動從最後一筆日期繼續抓取。

```bash
# 同步 watchlist 中所有股票（最常用）
python main.py sync

# 只同步指定股票
python main.py sync --stocks 2330 2317

# 指定日期範圍
python main.py sync --start 2023-01-01 --end 2024-12-31

# 組合使用
python main.py sync --stocks 2330 --start 2024-01-01

# 同時同步加權指數（用於 benchmark）
python main.py sync --taiex
```

每檔股票會同步五種資料：
| 資料類型 | FinMind Dataset | DB 資料表 |
|----------|----------------|-----------|
| 日K線（OHLCV） | TaiwanStockPrice | `daily_price` |
| 三大法人買賣超 | TaiwanStockInstitutionalInvestorsBuySell | `institutional_investor` |
| 融資融券 | TaiwanStockMarginPurchaseShortSale | `margin_trading` |
| 月營收 | TaiwanStockMonthRevenue | `monthly_revenue` |
| 股利 | TaiwanStockDividend | `dividend` |

加上 `--taiex` 可同步加權指數（存入 daily_price，stock_id=TAIEX）。

### 4.2 計算技術指標 (`compute`)

基於已入庫的日K線資料，計算技術指標並寫入 `technical_indicator` 表。

```bash
# 計算 watchlist 中所有股票的指標
python main.py compute

# 只計算指定股票
python main.py compute --stocks 2330 2317
```

目前支援的指標：
| 指標 | name 欄位值 | 說明 |
|------|-------------|------|
| SMA | sma_5, sma_10, sma_20, sma_60 | 簡單移動平均線 |
| RSI | rsi_14 | 相對強弱指標 (14日) |
| MACD | macd, macd_signal, macd_hist | 指數平滑異同移動平均線 (12,26,9) |
| Bollinger Bands | bb_upper, bb_middle, bb_lower | 布林通道 (20日, 2倍標準差) |

### 4.3 執行回測 (`backtest`)

基於日K線與技術指標，模擬歷史交易並計算績效。

```bash
# SMA 均線交叉策略 (預設 10日 x 20日)
python main.py backtest --stock 2330 --strategy sma_cross

# RSI 超買超賣策略 (預設 30/70)
python main.py backtest --stock 2330 --strategy rsi_threshold

# 布林通道突破策略
python main.py backtest --stock 2330 --strategy bb_breakout

# MACD 交叉策略
python main.py backtest --stock 2330 --strategy macd_cross

# 買入持有基準策略
python main.py backtest --stock 2330 --strategy buy_and_hold

# 指定回測期間
python main.py backtest --stock 2330 --strategy sma_cross --start 2023-01-01 --end 2025-12-31
```

可用策略：
| 策略名稱 | 說明 | 買入條件 | 賣出條件 |
|----------|------|----------|----------|
| `sma_cross` | SMA 均線交叉 | 快線上穿慢線（黃金交叉） | 快線下穿慢線（死亡交叉） |
| `rsi_threshold` | RSI 超買超賣 | RSI 突破超賣線(30) | RSI 跌破超買線(70) |
| `bb_breakout` | 布林通道突破 | 價格從下軌反彈 | 價格從上軌回落 |
| `macd_cross` | MACD 交叉 | MACD 上穿 Signal | MACD 下穿 Signal |
| `buy_and_hold` | 買入持有 | 第一天買入 | 最後一天賣出 |
| `multi_factor` | 多因子組合 | 加權分數 > 0.3 | 加權分數 < -0.3 |
| `ml_random_forest` | ML Random Forest | P(上漲) > threshold | P(上漲) < 1-threshold |
| `ml_xgboost` | ML XGBoost | P(上漲) > threshold | P(上漲) < 1-threshold |
| `ml_logistic` | ML Logistic Regression | P(上漲) > threshold | P(上漲) < 1-threshold |

多因子策略綜合 RSI、MACD、法人動向、營收成長四大因子，加權計算後產生訊號。

ML 策略使用機器學習模型預測未來 N 天漲跌，自動建構 15+ 技術特徵（動量、均線比率、波動率、成交量比率、布林位置等），依訓練比例分割歷史資料進行訓練與測試。XGBoost 未安裝時自動 fallback 到 Random Forest。

```bash
# ML 策略回測
python main.py backtest --stock 2330 --strategy ml_random_forest
python main.py backtest --stock 2330 --strategy ml_xgboost
python main.py backtest --stock 2330 --strategy ml_logistic
```

回測結果會自動顯示同期 Buy & Hold 基準報酬率與超額報酬。

交易成本設定（符合台股實際費率）：
- 手續費: 0.1425%（買賣各收一次）
- 交易稅: 0.3%（僅賣出時收取）
- 滑價: 0.05%
- 初始資金: 1,000,000 元

#### 風險管理參數

回測支援停損、停利、移動停損與部位大小計算：

```bash
# 停損停利
python main.py backtest --stock 2330 --strategy sma_cross --stop-loss 5 --take-profit 15

# 移動停損（從最高點回落 8% 出場）
python main.py backtest --stock 2330 --strategy sma_cross --trailing-stop 8

# 固定比例部位（每次只投入 30% 資金）
python main.py backtest --stock 2330 --strategy rsi_threshold --sizing fixed_fraction --fraction 0.3

# Kelly Criterion 部位計算（根據歷史勝率自動調整）
python main.py backtest --stock 2330 --strategy sma_cross --sizing kelly

# ATR 部位計算（根據波動度控制風險）
python main.py backtest --stock 2330 --strategy sma_cross --sizing atr
```

| 參數 | 說明 |
|------|------|
| `--stop-loss N` | 停損百分比，例 5.0 = 虧損 5% 時出場 |
| `--take-profit N` | 停利百分比，例 15.0 = 獲利 15% 時出場 |
| `--trailing-stop N` | 移動停損百分比，從持倉最高點回落 N% 時出場 |
| `--sizing MODE` | 部位大小計算：`all_in`（預設）/ `fixed_fraction` / `kelly` / `atr` |
| `--fraction N` | `fixed_fraction` 模式的資金比例（0.0~1.0） |

部位計算模式說明：
| 模式 | 說明 |
|------|------|
| `all_in` | 全部資金買入（預設，向後相容） |
| `fixed_fraction` | 以 `fraction` 比例的資金買入 |
| `kelly` | Kelly Criterion：根據歷史勝率與賠率計算最佳部位（預設 half-Kelly） |
| `atr` | ATR 部位：根據 ATR(14) 波動度計算股數上限，控制單筆風險 |

#### 投資組合回測

多股票同時回測，共用資金池：

```bash
# 等權配置
python main.py backtest --stocks 2330 2317 2454 --strategy sma_cross

# 加停損
python main.py backtest --stocks 2330 2317 --strategy multi_factor --stop-loss 5

# 等權 + 固定比例部位
python main.py backtest --stocks 2330 2317 2454 --strategy rsi_threshold --stop-loss 5 --take-profit 15
```

| 參數 | 說明 |
|------|------|
| `--stocks SID1 SID2 ...` | 多支股票代號（與 `--stock` 互斥） |
| `--allocation METHOD` | 配置方式：`equal_weight`（預設）/ `custom` |

#### 績效指標

除了基本指標（總報酬、年化報酬、Sharpe、最大回撤、勝率），P6 新增五項進階指標：

| 指標 | 說明 |
|------|------|
| Sortino Ratio | 僅考慮下行風險的風險調整報酬（越高越好） |
| Calmar Ratio | 年化報酬 / 最大回撤（越高越好） |
| VaR (95%) | 95% 信心水準下的每日最大損失 |
| CVaR (95%) | 超過 VaR 的平均損失（尾端風險） |
| Profit Factor | 總獲利 / 總虧損（> 1 表示獲利） |

### 4.4 啟動視覺化儀表板 (`dashboard`)

```bash
python main.py dashboard
```

瀏覽器會自動開啟 `http://localhost:8501`，包含五個頁面：

- **個股分析**: K線圖 + SMA/BB/RSI/MACD 疊加 + 成交量 + 法人買賣超 + 融資融券
- **回測結果**: 績效摘要卡片（含進階指標）+ 權益曲線/回撤圖 + 交易明細（含出場原因）+ 回測比較表
- **投資組合**: 組合回測績效 + 個股配置圓餅圖 + 個股報酬柱狀圖 + 交易明細
- **選股篩選**: 多因子條件篩選 + 因子分數排名 + CSV 匯出
- **ML 策略分析**: 模型訓練（準確率、特徵重要性、預測機率分佈）+ Walk-Forward 滾動驗證

### 4.5 多因子選股篩選 (`scan`)

掃描 watchlist 中所有股票，計算多因子分數並排名。

```bash
# 使用全部因子掃描（預設）
python main.py scan

# 匯出結果到 CSV
python main.py scan --export scan_results.csv

# 掃描後發送 Discord 通知
python main.py scan --notify

# 指定篩選條件（只看 RSI 超賣 + 外資買超）
python main.py scan --conditions rsi_oversold foreign_net_buy

# 指定股票範圍
python main.py scan --stocks 2330 2317 2454

# 調整回溯天數
python main.py scan --lookback 10
```

可用因子：
| 因子名稱 | 類別 | 說明 |
|----------|------|------|
| `rsi_oversold` | 技術面 | RSI < 30（超賣） |
| `macd_golden_cross` | 技術面 | MACD 上穿 Signal 線 |
| `price_above_sma` | 技術面 | 收盤價 > SMA20 |
| `foreign_net_buy` | 籌碼面 | 外資買超 |
| `institutional_consecutive_buy` | 籌碼面 | 法人連續買超 3 天 |
| `short_squeeze_ratio` | 籌碼面 | 券資比 > 20% |
| `revenue_yoy_growth` | 基本面 | 月營收 YoY > 20% |
| `revenue_consecutive_growth` | 基本面 | 連續營收月增 3 月 |

### 4.6 Discord Webhook 通知 (`notify`)

發送訊息到 Discord 頻道。需先在 `config/settings.yaml` 設定 Discord Webhook URL。

```bash
# 發送測試訊息
python main.py notify --message "測試通知"
```

**設定步驟：**

1. 開啟 Discord，進入要接收通知的頻道
2. 頻道設定 → 整合 → Webhook → 建立 Webhook
3. 複製 Webhook URL
4. 在 `config/settings.yaml` 新增：

```yaml
discord:
  webhook_url: "https://discord.com/api/webhooks/你的webhook"
  username: "台股量化系統"
  enabled: true
```

### 4.7 參數優化 (`optimize`)


使用 Grid Search 窮舉參數組合，找出最佳策略參數。

```bash
# 優化 SMA 交叉策略的快慢線天數
python main.py optimize --stock 2330 --strategy sma_cross

# 顯示前 5 名
python main.py optimize --stock 2330 --strategy sma_cross --top-n 5

# 匯出結果到 CSV
python main.py optimize --stock 2330 --strategy rsi_threshold --export results.csv

# 指定回測期間
python main.py optimize --stock 2330 --strategy macd_cross --start 2023-01-01
```

支援的策略與預設參數網格：
| 策略 | 參數 | 搜尋範圍 |
|------|------|----------|
| `sma_cross` | fast, slow | fast=[5,10,15,20,25,30], slow=[20,30,40,50,60] |
| `rsi_threshold` | oversold, overbought | oversold=[20,25,30,35,40], overbought=[60,65,70,75,80] |
| `bb_breakout` | period, std_dev | period=[10,15,20,25,30], std_dev=[1,2,3] |
| `macd_cross` | fast, slow, signal | fast=[8,12,16], slow=[20,26,32], signal=[7,9,11] |
| `ml_random_forest` | lookback, forward_days, threshold, train_ratio | lookback=[10,20,30], forward_days=[3,5,10], threshold=[0.55,0.6,0.65], train_ratio=[0.6,0.7,0.8] |
| `ml_xgboost` | lookback, forward_days, threshold, train_ratio | 同上 |
| `ml_logistic` | lookback, forward_days, threshold, train_ratio | 同上 |

### 4.8 自動排程 (`schedule`)

設定每日自動同步資料、計算指標、執行篩選並發送 Discord 通知。

```bash
# 產生 Windows Task Scheduler 腳本（建議）
python main.py schedule --mode windows

# 前景執行排程器（測試用，每日 23:00 自動同步）
python main.py schedule --mode simple
```

Windows 模式會在 `scripts/` 目錄產生 `daily_sync.bat` 和 `task_schedule.xml`，
按照輸出的說明匯入 Windows 工作排程器即可。

### 4.9 Walk-Forward 滾動驗證 (`walk-forward`)

對 ML 策略進行滾動窗口訓練/測試驗證，避免過擬合。將歷史資料分成多個滾動窗口，每次用 train_window 訓練、test_window 測試，依序往前推移。

```
|--- train_1 ---|-- test_1 --|
     |--- train_2 ---|-- test_2 --|
          |--- train_3 ---|-- test_3 --|
```

```bash
# 基本用法（預設 252 日訓練 / 63 日測試 / 63 日步進）
python main.py walk-forward --stock 2330 --strategy ml_random_forest

# 自訂窗口大小
python main.py walk-forward --stock 2330 --strategy ml_xgboost --train-window 504 --test-window 126 --step-size 63

# 調整 ML 參數
python main.py walk-forward --stock 2330 --strategy ml_random_forest --lookback 30 --forward-days 10 --threshold 0.55

# 搭配風險管理
python main.py walk-forward --stock 2330 --strategy ml_xgboost --stop-loss 5 --take-profit 15
```

| 參數 | 說明 |
|------|------|
| `--stock SID` | 股票代號 |
| `--strategy NAME` | 策略名稱（可用任何已註冊策略，特別適合 ML 策略） |
| `--train-window N` | 訓練窗口天數（預設 252，約 1 年） |
| `--test-window N` | 測試窗口天數（預設 63，約 1 季） |
| `--step-size N` | 每次前進天數（預設 63） |
| `--lookback N` | ML 回溯天數（覆蓋策略預設值） |
| `--forward-days N` | ML 預測天數（覆蓋策略預設值） |
| `--threshold N` | ML 訊號門檻（覆蓋策略預設值） |
| `--train-ratio N` | ML 訓練比例（覆蓋策略預設值） |

Walk-Forward 也支援所有風險管理參數（`--stop-loss`、`--take-profit`、`--trailing-stop`、`--sizing`、`--fraction`）。

### 4.10 資料庫遷移 (`migrate`)

若升級 P6 後使用既有資料庫，需執行遷移以新增欄位與表：

```bash
python main.py migrate
```

此命令會：
- 為 `backtest_result` 表新增 `sortino_ratio`、`calmar_ratio`、`var_95`、`cvar_95`、`profit_factor` 欄位
- 為 `trade` 表新增 `exit_reason` 欄位
- 建立 `portfolio_backtest_result` 和 `portfolio_trade` 新表

已存在的欄位會自動跳過，可重複執行。

### 4.11 查看資料庫概況 (`status`)

```bash
python main.py status
```

輸出範例：

```
[日K線] 6,224 筆 | 5 檔股票 | 2020-01-02 ~ 2026-02-11
[三大法人] 31,111 筆 | 5 檔股票 | 2020-01-02 ~ 2026-02-11
[融資融券] 6,224 筆 | 5 檔股票 | 2020-01-02 ~ 2026-02-11
```

---

## 5. 資料庫 Schema

資料庫使用 SQLite，檔案位於 `data/stock.db`。八張核心表：

### daily_price（日K線）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 交易日 |
| open | Float | 開盤價 |
| high | Float | 最高價 |
| low | Float | 最低價 |
| close | Float | 收盤價 |
| volume | BigInteger | 成交股數 |
| turnover | BigInteger | 成交金額 |
| spread | Float | 漲跌價差 |

唯一鍵：`(stock_id, date)`

### institutional_investor（三大法人）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 交易日 |
| name | String | 法人名稱（外資/投信/自營商） |
| buy | BigInteger | 買進股數 |
| sell | BigInteger | 賣出股數 |
| net | BigInteger | 買賣超股數 |

唯一鍵：`(stock_id, date, name)`

### margin_trading（融資融券）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 交易日 |
| margin_buy | BigInteger | 融資買進 |
| margin_sell | BigInteger | 融資賣出 |
| margin_balance | BigInteger | 融資餘額 |
| short_sell | BigInteger | 融券賣出 |
| short_buy | BigInteger | 融券買進 |
| short_balance | BigInteger | 融券餘額 |

唯一鍵：`(stock_id, date)`

### monthly_revenue（月營收）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 資料日期 |
| revenue | BigInteger | 營收金額 |
| revenue_month | Integer | 月份 |
| revenue_year | Integer | 年度 |
| mom_growth | Float | 月增率 (%) |
| yoy_growth | Float | 年增率 (%) |

唯一鍵：`(stock_id, date)`

### dividend（股利）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 除權息基準日 |
| year | String | 股利所屬年度 |
| cash_dividend | Float | 現金股利 |
| stock_dividend | Float | 股票股利 |
| cash_payment_date | Date | 現金發放日 |
| announcement_date | Date | 公告日 |

唯一鍵：`(stock_id, date)`

### technical_indicator（技術指標 — EAV 長表）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 交易日 |
| name | String | 指標名稱（如 sma_20, rsi_14, macd） |
| value | Float | 指標值 |

唯一鍵：`(stock_id, date, name)`

### backtest_result（回測結果）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| strategy_name | String | 策略名稱 |
| start_date | Date | 回測起始日 |
| end_date | Date | 回測結束日 |
| initial_capital | Float | 初始資金 |
| final_capital | Float | 最終資金 |
| total_return | Float | 總報酬率 (%) |
| annual_return | Float | 年化報酬率 (%) |
| sharpe_ratio | Float | Sharpe Ratio |
| max_drawdown | Float | 最大回撤 (%) |
| win_rate | Float | 勝率 (%) |
| total_trades | Integer | 交易次數 |
| benchmark_return | Float | 基準報酬率 (%) |
| sortino_ratio | Float | Sortino Ratio（P6 新增） |
| calmar_ratio | Float | Calmar Ratio（P6 新增） |
| var_95 | Float | VaR 95%（P6 新增） |
| cvar_95 | Float | CVaR 95%（P6 新增） |
| profit_factor | Float | Profit Factor（P6 新增） |
| created_at | DateTime | 建立時間 |

### trade（交易明細）

| 欄位 | 型別 | 說明 |
|------|------|------|
| backtest_id | Integer | 關聯的回測結果 ID (FK) |
| entry_date | Date | 進場日期 |
| entry_price | Float | 進場價格 |
| exit_date | Date | 出場日期 |
| exit_price | Float | 出場價格 |
| shares | Integer | 股數 |
| pnl | Float | 損益金額 |
| return_pct | Float | 報酬率 (%) |
| exit_reason | String | 出場原因: signal/stop_loss/take_profit/trailing_stop/force_close（P6 新增） |

### portfolio_backtest_result（投資組合回測結果，P6 新增）

| 欄位 | 型別 | 說明 |
|------|------|------|
| strategy_name | String | 策略名稱 |
| stock_ids | Text | 股票代號（逗號分隔） |
| start_date | Date | 回測起始日 |
| end_date | Date | 回測結束日 |
| initial_capital | Float | 初始資金 |
| final_capital | Float | 最終資金 |
| total_return | Float | 總報酬率 (%) |
| annual_return | Float | 年化報酬率 (%) |
| sharpe_ratio | Float | Sharpe Ratio |
| max_drawdown | Float | 最大回撤 (%) |
| win_rate | Float | 勝率 (%) |
| total_trades | Integer | 交易次數 |
| sortino_ratio | Float | Sortino Ratio |
| calmar_ratio | Float | Calmar Ratio |
| var_95 | Float | VaR 95% |
| cvar_95 | Float | CVaR 95% |
| profit_factor | Float | Profit Factor |
| allocation_method | String | 配置方式（equal_weight/custom） |
| created_at | DateTime | 建立時間 |

### portfolio_trade（投資組合交易明細，P6 新增）

| 欄位 | 型別 | 說明 |
|------|------|------|
| portfolio_backtest_id | Integer | 關聯的組合回測 ID (FK) |
| stock_id | String | 股票代號 |
| entry_date | Date | 進場日期 |
| entry_price | Float | 進場價格 |
| exit_date | Date | 出場日期 |
| exit_price | Float | 出場價格 |
| shares | Integer | 股數 |
| pnl | Float | 損益金額 |
| return_pct | Float | 報酬率 (%) |
| exit_reason | String | 出場原因 |

---

## 6. 在 Python 中直接查詢資料

除了 CLI，也可以在 Python 腳本或 Jupyter Notebook 中直接操作：

```python
from src.data.database import get_session, init_db
from src.data.schema import DailyPrice, InstitutionalInvestor, MarginTrading
from sqlalchemy import select

init_db()

# 查詢台積電近期收盤價
with get_session() as session:
    rows = session.execute(
        select(DailyPrice)
        .where(DailyPrice.stock_id == "2330")
        .order_by(DailyPrice.date.desc())
        .limit(5)
    ).scalars().all()

    for r in rows:
        print(f"{r.date}  close={r.close}  volume={r.volume}")
```

```python
# 用 pandas 讀取整張表
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///data/stock.db")
df = pd.read_sql("SELECT * FROM daily_price WHERE stock_id='2330'", engine)
print(df.tail())
```

---

## 7. 常見問題

### Q: sync 出現 `too many SQL variables` 錯誤？

已修復。Pipeline 現在會自動分批寫入（每批 80 筆），不會超出 SQLite 變數上限。

### Q: 如何新增要追蹤的股票？

編輯 `config/settings.yaml`，在 `fetcher.watchlist` 下新增股票代號，然後執行 `python main.py sync`。

### Q: 資料多久更新一次？

FinMind 資料約在每個交易日收盤後更新。建議每天盤後執行一次 `python main.py sync`，系統會自動增量抓取新資料。

### Q: API 有速率限制嗎？

FinMind 免費版有請求頻率限制。系統已內建每次請求後等待 0.5 秒的節流機制。

### Q: 資料庫檔案在哪裡？

`data/stock.db`，是一個 SQLite 檔案，可用任何 SQLite 工具（如 DB Browser for SQLite）直接瀏覽。
