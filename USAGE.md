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
│   │   ├── schema.py        # ORM 資料表定義（14 張表，含 FinancialStatement）
│   │   ├── fetcher.py       # FinMind API 資料抓取
│   │   ├── twse_fetcher.py  # TWSE/TPEX 官方開放資料抓取（全市場）
│   │   ├── mops_fetcher.py  # MOPS 公開資訊觀測站重大訊息抓取
│   │   ├── pipeline.py      # ETL Pipeline（抓取→清洗→寫入）
│   │   ├── validator.py     # 資料品質檢查（6 項檢查 + 報告）
│   │   ├── io.py            # 通用匯出/匯入（CSV/Parquet）
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
│   ├── discovery/
│   │   └── scanner.py       # 全市場選股掃描器（四階段漏斗）
│   ├── report/
│   │   ├── engine.py        # 每日選股報告引擎（四維度評分）
│   │   └── formatter.py     # Discord 訊息格式化
│   ├── strategy_rank/
│   │   └── engine.py        # 策略回測排名引擎
│   ├── industry/
│   │   └── analyzer.py      # 產業輪動分析引擎
│   ├── notification/
│   │   └── line_notify.py   # Discord Webhook 通知
│   ├── backtest/
│   │   ├── engine.py        # 回測引擎 + 風險管理 + 績效計算
│   │   ├── portfolio.py     # 投資組合回測引擎
│   │   └── walk_forward.py  # Walk-Forward 滾動驗證引擎
│   ├── optimization/
│   │   └── grid_search.py   # Grid Search 參數優化
│   ├── scheduler/
│   │   ├── simple_scheduler.py  # 前景排程器（含每日報告）
│   │   └── windows_task.py      # Windows Task Scheduler 產生器
│   └── visualization/
│       ├── app.py            # Streamlit 儀表板入口
│       ├── data_loader.py    # 資料查詢模組
│       ├── charts.py         # Plotly 圖表元件
│       └── pages/            # 頁面模組
│           ├── stock_analysis.py      # 個股分析頁
│           ├── backtest_review.py     # 回測結果頁
│           ├── screener_results.py    # 選股篩選頁
│           ├── portfolio_review.py    # 投資組合頁
│           ├── ml_analysis.py         # ML 策略分析頁
│           ├── market_overview.py     # 市場總覽首頁
│           ├── industry_rotation.py   # 產業輪動分析頁
│           └── discovery_history.py   # 推薦歷史頁
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

# 同時同步加權指數（用於 benchmark，現已預設啟用）
python main.py sync --taiex
```

> **注意**：`sync` 命令現在預設會自動同步 TAIEX 加權指數，無需手動加 `--taiex`。

每檔股票會同步六種資料：
| 資料類型 | FinMind Dataset | DB 資料表 |
|----------|----------------|-----------|
| 日K線（OHLCV） | TaiwanStockPrice | `daily_price` |
| 三大法人買賣超 | TaiwanStockInstitutionalInvestorsBuySell | `institutional_investor` |
| 融資融券 | TaiwanStockMarginPurchaseShortSale | `margin_trading` |
| 月營收 | TaiwanStockMonthRevenue | `monthly_revenue` |
| 股利 | TaiwanStockDividend | `dividend` |
| 財報（季報） | FinancialStatements + BalanceSheet + CashFlows | `financial_statement` |

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

#### 除權息還原回測

啟用 `--adjust-dividend` 旗標可讓回測考慮除權息影響：

```bash
# 除權息還原回測
python main.py backtest --stock 2330 --strategy sma_cross --adjust-dividend

# 投資組合 + 除權息還原
python main.py backtest --stocks 2330 2317 2454 --strategy multi_factor --adjust-dividend
```

啟用後系統會自動執行兩件事：
1. **價格還原**：回溯調整除權息前的 OHLC 價格，重新計算技術指標，避免除權息日產生假訊號（如 SMA 假跌破、RSI 假超賣）
2. **股利入帳**：持倉期間遇到除權息日，現金股利自動加入資金，股票股利自動增加持股數量

> **注意**：需先透過 `python main.py sync` 同步股利資料（`dividend` 表）。預設關閉以保持向後相容性。

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
| `--adjust-dividend` | 啟用除權息還原（回溯調整價格 + 股利入帳） |

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

# 風險平價配置（波動大的股票權重低）
python main.py backtest --stocks 2330 2317 2454 --strategy sma_cross --allocation risk_parity

# 均值-方差最佳化（最大化 Sharpe ratio）
python main.py backtest --stocks 2330 2317 2454 --strategy sma_cross --allocation mean_variance

# 加停損
python main.py backtest --stocks 2330 2317 --strategy multi_factor --stop-loss 5

# 等權 + 固定比例部位
python main.py backtest --stocks 2330 2317 2454 --strategy rsi_threshold --stop-loss 5 --take-profit 15
```

配置方式說明：

| 模式 | 說明 |
|------|------|
| `equal_weight` | 等權配置，每支股票分配相同資金（預設） |
| `custom` | 自訂權重（程式化呼叫時使用） |
| `risk_parity` | 風險平價，使各資產的風險貢獻相等。波動大的股票分配較少權重（需 scipy） |
| `mean_variance` | Markowitz 均值-方差最佳化，最大化 Sharpe ratio。高報酬低風險的股票獲得較高權重（需 scipy） |

> **注意**：`risk_parity` 和 `mean_variance` 需要至少 30 天的歷史資料計算報酬率。資料不足時會自動 fallback 到 `equal_weight`。

| 參數 | 說明 |
|------|------|
| `--stocks SID1 SID2 ...` | 多支股票代號（與 `--stock` 互斥） |
| `--allocation METHOD` | 配置方式：`equal_weight`（預設）/ `custom` / `risk_parity` / `mean_variance` |

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

瀏覽器會自動開啟 `http://localhost:8501`，包含八個頁面：

- **市場總覽**: TAIEX 走勢 K 線 + SMA60/120 + Regime 狀態、市場廣度（漲跌家數）、法人買賣超排行、產業熱度 Treemap
- **個股分析**: K線圖 + SMA/BB/RSI/MACD 疊加 + 成交量 + 法人買賣超 + 融資融券
- **回測結果**: 績效摘要卡片（含進階指標）+ 權益曲線/回撤圖 + 交易明細（含出場原因）+ 回測比較表
- **投資組合**: 組合回測績效 + 個股配置圓餅圖 + 個股報酬柱狀圖 + 交易明細
- **選股篩選**: 多因子條件篩選 + 因子分數排名 + CSV 匯出
- **ML 策略分析**: 模型訓練（準確率、特徵重要性、預測機率分佈）+ Walk-Forward 滾動驗證
- **產業輪動**: 產業綜合排名 + 泡泡圖（法人 vs 動能）+ 法人淨買超長條圖 + 精選個股
- **推薦歷史**: Discover 推薦歷史視覺化（日曆熱圖、績效分析、個股排行、歷史明細 CSV 匯出）

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

# 除權息還原
python main.py walk-forward --stock 2330 --strategy ml_random_forest --adjust-dividend
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
| `--adjust-dividend` | 啟用除權息還原（回溯調整價格 + 股利入帳） |

Walk-Forward 也支援所有風險管理參數（`--stop-loss`、`--take-profit`、`--trailing-stop`、`--sizing`、`--fraction`）。

### 4.10 每日選股報告 (`report`)

四維度（技術面/籌碼面/基本面/ML）綜合評分，自動排名最值得關注的股票。

```bash
# 預設前 10 名
python main.py report

# 顯示前 20 名
python main.py report --top 20

# 跳過 ML 評分（速度更快）
python main.py report --no-ml

# 指定股票範圍
python main.py report --stocks 2330 2317 2454

# 發送 Discord 通知
python main.py report --notify

# 匯出 CSV
python main.py report --export daily_report.csv
```

| 參數 | 說明 |
|------|------|
| `--top N` | 顯示前 N 名（預設 10） |
| `--no-ml` | 跳過 ML 評分，加快速度 |
| `--stocks SID ...` | 指定股票代號 |
| `--export PATH` | 匯出 CSV |
| `--notify` | 發送 Discord 通知 |

四維度評分說明：

| 維度 | 權重 | 評分依據 |
|------|------|----------|
| 技術面 | 30% | RSI 位置、MACD 多空、SMA20 多空 |
| 籌碼面 | 30% | 外資買超比例、法人合計買超比例、融資變化 |
| 基本面 | 20% | 營收 YoY 成長率、MoM 成長加分 |
| ML 預測 | 20% | Random Forest predict_proba 信心指標 |

### 4.11 策略回測排名 (`strategy-rank`)

批次回測 watchlist × 策略的所有組合，找出最佳股票-策略配對。

```bash
# 預設 watchlist × 6 個快速策略
python main.py strategy-rank

# 按 Sharpe Ratio 排名（預設）
python main.py strategy-rank --metric sharpe

# 按勝率排名
python main.py strategy-rank --metric win_rate

# 指定策略
python main.py strategy-rank --strategies sma_cross rsi_threshold macd_cross

# 指定股票
python main.py strategy-rank --stocks 2330 2317 2454

# 指定回測期間
python main.py strategy-rank --start 2023-01-01

# 設定最少交易次數門檻
python main.py strategy-rank --min-trades 5

# 匯出 + Discord 通知
python main.py strategy-rank --export rank.csv --notify
```

| 參數 | 說明 |
|------|------|
| `--metric NAME` | 排名指標：`sharpe`（預設）/ `total_return` / `win_rate` / `annual_return` |
| `--strategies NAME ...` | 指定策略名稱（預設 6 個快速策略） |
| `--stocks SID ...` | 指定股票代號 |
| `--start DATE` | 回測起始日期 |
| `--end DATE` | 回測結束日期 |
| `--min-trades N` | 最少交易次數（預設 3） |
| `--export PATH` | 匯出 CSV |
| `--notify` | 發送 Discord 通知 |

### 4.12 產業輪動分析 (`industry`)

分析各產業的法人資金動能與價格動能，找出當前熱門產業及精選個股。

```bash
# 預設分析
python main.py industry

# 強制重新抓取產業分類資料
python main.py industry --refresh

# 顯示前 5 名產業
python main.py industry --top-sectors 5

# 每產業選前 10 支股票
python main.py industry --top 10

# 調整法人回溯天數
python main.py industry --lookback 30

# 調整價格動能回溯天數
python main.py industry --momentum 90

# 發送 Discord 通知
python main.py industry --notify
```

| 參數 | 說明 |
|------|------|
| `--refresh` | 強制重新抓取 StockInfo 產業分類 |
| `--top-sectors N` | 顯示前 N 名產業（預設 5） |
| `--top N` | 每產業精選 N 支股票（預設 5） |
| `--lookback N` | 法人資金流回溯天數（預設 20） |
| `--momentum N` | 價格動能回溯天數（預設 60） |
| `--stocks SID ...` | 指定股票範圍 |
| `--notify` | 發送 Discord 通知 |

產業排名綜合兩個維度（各占 50%）：
- **法人動能**：各產業內股票的三大法人淨買超合計
- **價格動能**：各產業內股票的平均漲跌幅

### 4.13 全市場選股掃描 (`discover`)

從全台灣 ~6000 支股票（上市 + 上櫃）中自動篩選出值得關注的標的。支援五種掃描模式：

- **momentum**（預設）：短線動能股（1~10 天），抓突破 + 資金流 + 量能擴張
- **swing**：中期波段股（1~3 個月），抓趨勢 + 基本面 + 法人布局
- **value**：價值修復股，低估值 + 基本面轉佳 + 法人布局
- **dividend**：高息存股，高殖利率 + 配息穩定 + 估值合理
- **growth**：高成長股，營收高速成長 + 動能啟動

**漏斗架構：**
```
全市場 ~6000 支 → 粗篩 ~150 支 → 細評排名 → 風險過濾 → Top N 推薦
```

**資料來源優先順序：**
1. TWSE/TPEX 官方開放資料（免費，6 次 API 呼叫取得全市場，含融資融券）
2. FinMind 批次 API（需付費帳號）
3. FinMind 逐股抓取（免費帳號備案，較慢）

**市場狀態（Regime）自動偵測：** 系統自動根據加權指數（TAIEX）判斷市場狀態（bull/bear/sideways），動態調整各模式的四維度因子權重（技術面 + 籌碼面 + 基本面 + 消息面）。多頭加重技術面，空頭加重基本面與消息面。

**消息面評分（MOPS 重大訊息）：** 系統會自動從公開資訊觀測站抓取近期重大訊息，以關鍵字規則分類情緒（正面/中性/負面），並計算消息面分數。三因子加權：公告密度 30% + 正面訊號 40% + 負面懲罰 30%。

```bash
# 預設 momentum 模式（同步資料 + 篩選 Top 20）
python main.py discover

# 明確指定 momentum 模式
python main.py discover momentum --top 30

# 中期波段模式（自動擴展 sync-days 至 80 天）
python main.py discover swing --top 20

# 價值修復模式（低 PE + 高殖利率 + 基本面轉佳）
python main.py discover value --top 20

# 高息存股模式（殖利率 > 3% + PE > 0 + 配息穩定）
python main.py discover dividend --top 20

# 高成長模式（營收 YoY > 10% + 動能啟動）
python main.py discover growth --top 20

# 調整股價範圍
python main.py discover momentum --min-price 50 --max-price 500

# 跳過資料同步（使用 DB 已有資料）
python main.py discover --skip-sync

# 匯出 CSV + 通知
python main.py discover swing --top 30 --export picks.csv --notify

# 與上次推薦比較（顯示新進/退出/排名變化）
python main.py discover --compare
python main.py discover momentum --skip-sync --compare
```

每次執行 `discover` 時，推薦結果會自動存入 DB（`discovery_record` 表），供歷史追蹤使用。同日同模式重跑會覆蓋先前記錄。

| 參數 | 說明 |
|------|------|
| `mode` | 掃描模式：`momentum`（短線動能）、`swing`（中期波段）、`value`（價值修復）、`dividend`（高息存股）、`growth`（高成長），預設 momentum |
| `--top N` | 顯示前 N 名（預設 20） |
| `--min-price N` | 最低股價門檻（預設 10） |
| `--max-price N` | 最高股價門檻（預設 2000） |
| `--min-volume N` | 最低成交量/股（預設 500000） |
| `--sync-days N` | 同步最近幾個交易日（預設 3，swing 模式自動擴展至 80） |
| `--skip-sync` | 跳過全市場資料同步，直接用 DB 既有資料 |
| `--export PATH` | 匯出 CSV |
| `--notify` | 發送 Discord 通知 |
| `--compare` | 顯示與上次推薦的差異（新進/退出/排名變動 >= 3 名） |

**Momentum 模式（sideways 基準權重，bull/bear 自動微調）：**

| 維度 | 權重 | 因子 |
|------|------|------|
| 技術面 | 45% | 5日動能、10日動能、20日突破、量比、成交量加速 |
| 籌碼面 | 45% | 外資連續買超天數 + 法人買超/成交量 + 合計買超 + 券資比（有資料時） |
| 基本面 | 10% | 營收 YoY > 0 過濾（加分/不加分） |
| 風險過濾 | — | ATR(14)/close > 80th percentile 剔除 |

**Swing 模式：**

| 維度 | 權重 | 因子 |
|------|------|------|
| 技術面 | 30% | 趨勢確認(close>SMA60)、均線排列(SMA20>SMA60)、60日動能、量價齊揚 |
| 籌碼面 | 30% | 投信淨買超(50%)、法人20日累積買超(50%) |
| 基本面 | 40% | 營收YoY(40%)、MoM(30%)、營收加速度(30%) |
| 風險過濾 | — | 年化波動率 > 85th percentile 剔除 |

**Value 模式：**

| 維度 | 權重 | 因子 |
|------|------|------|
| 基本面 | 50% | 營收YoY(40%)、MoM(30%)、營收加速度(30%) |
| 估值面 | 30% | PE反向排名(40%)、PB反向排名(30%)、殖利率排名(30%) |
| 籌碼面 | 20% | 投信近期買超(50%)、法人累積買超(50%) |
| 粗篩門檻 | — | PE > 0 且 < 30、殖利率 > 2% |
| 風險過濾 | — | 近20日波動率 > 90th percentile 剔除 |

**Dividend 模式：**

| 維度 | 權重 | 因子 |
|------|------|------|
| 基本面 | 35~45% | 營收YoY(40%)、MoM(30%)、營收加速度(30%) |
| 殖利率面 | 25~35% | 殖利率排名(50%)、PE反向排名(30%)、PB反向排名(20%) |
| 籌碼面 | 10~20% | 投信近期買超(50%)、法人累積買超(50%) |
| 粗篩門檻 | — | 殖利率 > 3%、PE > 0 |
| 風險過濾 | — | 近20日波動率 > 90th percentile 剔除 |

**Growth 模式：**

| 維度 | 權重 | 因子 |
|------|------|------|
| 基本面 | 40~50% | 營收YoY(40%)、MoM(30%)、營收加速度(30%) |
| 技術面 | 15~30% | 5日動能、10日動能、20日突破、量比、成交量加速 |
| 籌碼面 | 15~20% | 外資連續買超天數 + 買超佔量比 + 合計買超 + 券資比（有資料時） |
| 粗篩門檻 | — | 營收 YoY > 10% |
| 風險過濾 | — | ATR(14)/close > 80th percentile 剔除 |

**Regime 權重調整矩陣（系統自動偵測）：**

| 模式 | 面向 | 多頭 | 盤整 | 空頭 |
|------|------|------|------|------|
| Momentum | Tech / Chip / Fund / News | 45/35/10/10 | 40/40/10/10 | 30/40/15/15 |
| Swing | Tech / Chip / Fund / News | 30/20/40/10 | 25/25/35/15 | 15/25/45/15 |
| Value | Fund / Val / Chip / News | 40/35/15/10 | 45/25/15/15 | 50/20/10/20 |
| Dividend | Fund / Div / Chip / News | 35/35/20/10 | 40/30/15/15 | 45/25/10/20 |
| Growth | Fund / Tech / Chip / News | 45/30/15/10 | 40/25/20/15 | 50/15/15/20 |

**篩選流程說明：**

| 階段 | 動作 | 說明 |
|------|------|------|
| Stage 0 | 市場狀態偵測 | 根據 TAIEX 判斷 bull/bear/sideways，動態調整權重 |
| Stage 1 | 資料載入 | 從 DB 讀取全市場日K + 三大法人 + 融資融券 |
| Stage 2 | 粗篩 | 模式專屬條件篩選，取前 150 名 |
| Stage 0.5 | MOPS 月營收同步 | （僅 growth 模式）檢查月營收覆蓋率，不足 500 支時自動從 MOPS 同步全市場月營收 |
| Stage 2.5 | 營收/估值補抓 | 從 FinMind 逐股補抓候選股月營收（value 模式另補抓 PE/PB/殖利率） |
| Stage 2.7 | 公告載入 | 從 DB 載入候選股近期 MOPS 重大訊息 |
| Stage 3 | 細評 | 四維度因子（技術+籌碼+基本面+消息面）+ Regime 動態權重評分 |
| Stage 3.3 | 產業加成 | 用 IndustryRotationAnalyzer 計算產業排名，熱門產業 +5%、冷門產業 -5% 線性加成到 composite_score |
| Stage 3.5 | 風險過濾 | 剔除高波動股 |
| Stage 4 | 排名輸出 | 加上產業標籤與股票名稱，統計產業分布 |

> **注意**：Stage 2.5 需要 FinMind API Token 才能補抓月營收與估值資料。若無 Token，基本面/估值分數會 fallback 到 0.5（中性值），不影響其他維度評分。

### 4.14 Discover 推薦績效回測 (`discover-backtest`)

評估歷史 `discover` 推薦的實際表現：讀取 DiscoveryRecord 歷史記錄，對照 DailyPrice 計算推薦後 N 天的實際報酬率，輸出勝率/平均報酬/最大虧損等統計。

```bash
# 評估 momentum 推薦績效（預設持有 5,10,20 天）
python main.py discover-backtest --mode momentum

# 自訂持有天數
python main.py discover-backtest --mode swing --days 5,10,20,60

# 只看每次掃描前 10 名的績效
python main.py discover-backtest --mode value --top 10

# 指定掃描日期範圍
python main.py discover-backtest --mode momentum --start 2025-06-01 --end 2025-12-31

# 匯出明細 CSV
python main.py discover-backtest --mode momentum --export result.csv
```

**參數說明：**

| 參數 | 說明 |
|------|------|
| `--mode` | 必填，掃描模式：`momentum` / `swing` / `value` / `dividend` / `growth` |
| `--days` | 持有天數，逗號分隔（預設 `5,10,20`） |
| `--top` | 只計算每次掃描前 N 名的績效（預設全部） |
| `--start` | 掃描日期範圍起始（YYYY-MM-DD） |
| `--end` | 掃描日期範圍結束（YYYY-MM-DD） |
| `--export` | 匯出明細 CSV 路徑 |

**輸出三層聚合：**

1. **整體摘要**：每個持有天數的勝率、平均報酬、中位數、最大獲利/虧損
2. **逐次掃描**：每個掃描日期的平均報酬、勝率、最佳/最差個股
3. **個股明細**：每筆推薦的報酬率（供 `--export` 匯出）

> **前提**：須先有足夠的 `discover` 歷史記錄（每次執行 `discover` 會自動存入 DB），以及推薦日之後的 DailyPrice 資料。

### 4.15 同步 MOPS 重大訊息 (`sync-mops`)

從公開資訊觀測站（MOPS）備援站抓取上市/上櫃公司最新重大訊息公告。資料用於 discover 的消息面評分。

```bash
# 同步最新一天的重大訊息
python main.py sync-mops
```

系統會自動對公告主旨進行關鍵字情緒分類（+1 正面 / 0 中性 / -1 負面），同步完成後顯示情緒分布統計。

> **注意**：MOPS 備援站僅提供最新一個交易日的公告，無法查詢歷史資料。建議搭配每日排程使用，逐日累積公告歷史。`discover` 命令的全市場同步也會自動附帶 MOPS 同步。

### 4.16 同步全市場月營收 (`sync-revenue`)

從 MOPS 公開資訊觀測站抓取全市場（上市+上櫃）月營收。每次僅需 2 個 HTTP 請求即可取得 ~2000+ 支股票的月營收資料（含 YoY、MoM 成長率）。

```bash
# 同步上月全市場月營收
python main.py sync-revenue

# 同步最近 3 個月
python main.py sync-revenue --months 3
```

**參數說明：**

| 參數 | 說明 |
|------|------|
| `--months` | 同步最近幾個月的營收（預設 1 = 上月） |

> **自動同步**：`discover growth` 執行時若偵測到月營收覆蓋不足（< 500 支股票），會自動觸發 `sync_mops_revenue()` 補抓，無需手動執行此命令。

### 4.17 同步財報資料 (`sync-financial`)

從 FinMind 抓取季報財務資料（綜合損益表 + 資產負債表 + 現金流量表），合併三表並計算衍生比率後存入 `financial_statement` 表。

```bash
# 同步 watchlist 財報（預設最近 4 季）
python main.py sync-financial

# 指定股票
python main.py sync-financial --stocks 2330 2317

# 同步最近 8 季（2 年）
python main.py sync-financial --quarters 8
```

**參數說明：**

| 參數 | 說明 |
|------|------|
| `--stocks SID ...` | 指定股票代號（預設使用 watchlist） |
| `--quarters N` | 同步最近幾季（預設 4） |

**同步內容：**

| 來源 Dataset | 欄位 |
|-------------|------|
| TaiwanStockFinancialStatements | 營收、毛利、營業利益、稅後淨利、EPS |
| TaiwanStockBalanceSheet | 總資產、總負債、股東權益 |
| TaiwanStockCashFlowsStatement | 營業現金流、投資現金流、融資現金流 |

**自動計算衍生比率：** 毛利率、營益率、淨利率、ROE、ROA、負債比、自由現金流。

> **注意**：`sync` 命令也會自動同步財報，無需額外執行此命令。此命令適合初次設定或補抓歷史財報時使用。

### 4.18 資料品質檢查 (`validate`)

檢測 DB 中的資料品質問題：缺漏交易日、零成交量、連續漲跌停、價格異常（high < low、負價格）、資料表日期範圍不一致、資料新鮮度。

```bash
# 檢查全部股票（預設）
python main.py validate

# 只檢查指定股票
python main.py validate --stocks 2330 2317

# 調整缺漏門檻（預設 5 個營業日）
python main.py validate --gap-threshold 3

# 調整連續漲跌停門檻（預設 5 天）
python main.py validate --streak-threshold 3

# 跳過資料新鮮度檢查
python main.py validate --no-freshness

# 匯出問題清單到 CSV
python main.py validate --export issues.csv
```

**參數說明：**

| 參數 | 說明 |
|------|------|
| `--stocks SID ...` | 指定股票代號（預設檢查全部） |
| `--gap-threshold N` | 缺漏營業日門檻（預設 5，>= 此值報 error） |
| `--streak-threshold N` | 連續漲跌停天數門檻（預設 5，>= 此值報 warning） |
| `--no-freshness` | 跳過資料新鮮度檢查 |
| `--export PATH` | 匯出問題清單 CSV 路徑 |

**六項檢查：**

| 檢查項 | 嚴重度 | 說明 |
|--------|--------|------|
| 缺漏交易日 | error | 連續缺漏 >= 門檻個營業日 |
| 零成交量 | error/warning | 連續 3+ 天為 error，單日為 warning |
| 連續漲跌停 | warning | 日報酬率 >= 9.5% 且同方向連續 >= 門檻天 |
| 價格異常 | error | high < low、close 超出 [low, high]、價格 <= 0 |
| 日期範圍不一致 | warning | 同股不同表的最新日期差距 > 30 天 |
| 資料過期 | warning | 最新資料距今 > 7 個營業日 |

### 4.19 資料庫遷移 (`migrate`)

若升級 P6 後使用既有資料庫，需執行遷移以新增欄位與表：

```bash
python main.py migrate
```

此命令會：
- 為 `backtest_result` 表新增 `sortino_ratio`、`calmar_ratio`、`var_95`、`cvar_95`、`profit_factor` 欄位
- 為 `trade` 表新增 `exit_reason` 欄位
- 建立 `portfolio_backtest_result` 和 `portfolio_trade` 新表

已存在的欄位會自動跳過，可重複執行。

### 4.20 查看資料庫概況 (`status`)

```bash
python main.py status
```

輸出範例：

```
[日K線] 6,224 筆 | 5 檔股票 | 2020-01-02 ~ 2026-02-11
[三大法人] 31,111 筆 | 5 檔股票 | 2020-01-02 ~ 2026-02-11
[融資融券] 6,224 筆 | 5 檔股票 | 2020-01-02 ~ 2026-02-11
```

### 4.21 匯出資料表 (`export`)

將任意 ORM 資料表匯出為 CSV 或 Parquet 檔案。支援股票代號篩選與日期範圍篩選。

```bash
# 列出所有可匯出的資料表及筆數
python main.py export --list

# 匯出日K線（CSV，預設輸出到 data/export/daily_price.csv）
python main.py export daily_price

# 指定輸出路徑
python main.py export daily_price -o data/export/daily_price.csv

# 篩選股票
python main.py export daily_price --stocks 2330 2317

# 篩選日期範圍
python main.py export daily_price --start 2024-01-01 --end 2024-12-31

# Parquet 格式（需安裝 pyarrow）
python main.py export daily_price --format parquet -o data/export/daily_price.parquet

# 組合使用
python main.py export institutional_investor --stocks 2330 --start 2025-01-01 -o ii.csv
```

**參數說明：**

| 參數 | 說明 |
|------|------|
| `table` | 資料表名稱（見 `--list` 輸出） |
| `-o / --output` | 輸出檔案路徑（預設: `data/export/<table>.<format>`） |
| `--format` | 輸出格式：`csv`（預設）或 `parquet` |
| `--stocks SID ...` | 篩選股票代號（僅限有 stock_id 欄位的表） |
| `--start DATE` | 起始日期 YYYY-MM-DD（僅限有 date 欄位的表） |
| `--end DATE` | 結束日期 YYYY-MM-DD |
| `--list` | 列出所有支援的資料表名稱及筆數 |

> **注意**：匯出時自動排除 `id` 自增主鍵欄位。CSV 編碼為 UTF-8 with BOM（`utf-8-sig`），可在 Excel 中正確顯示中文。

### 4.22 匯入資料 (`import-data`)

從 CSV 或 Parquet 檔案匯入資料到指定表。自動驗證欄位格式，重複資料靜默略過（upsert）。

```bash
# 匯入 CSV 到 daily_price 表
python main.py import-data daily_price data/export/daily_price.csv

# 匯入 Parquet
python main.py import-data daily_price data/export/daily_price.parquet

# 僅驗證資料格式，不實際寫入
python main.py import-data daily_price data.csv --dry-run
```

**參數說明：**

| 參數 | 說明 |
|------|------|
| `table` | 目標資料表名稱 |
| `source` | 來源檔案路徑（`.csv` 或 `.parquet`） |
| `--dry-run` | 僅驗證資料格式（檢查必要欄位、日期格式），不實際寫入 DB |

**匯入行為：**
- 自動偵測檔案格式（依副檔名 `.csv` / `.parquet`）
- 驗證必要欄位是否存在（缺少則報錯）
- 多餘欄位自動忽略
- 重複資料（依唯一鍵衝突）靜默略過，不覆寫
- 批次寫入（每批 80 筆），符合 SQLite 變數上限

> **典型用途**：備份還原、跨環境資料搬遷、從 CSV 匯入外部資料。先用 `export` 匯出，再用 `import-data` 匯入。

---

## 5. 資料庫 Schema

資料庫使用 SQLite，檔案位於 `data/stock.db`。十四張核心表：

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
| allocation_method | String | 配置方式（equal_weight/custom/risk_parity/mean_variance） |
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

### announcement（MOPS 重大訊息公告）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 公告日期 |
| seq | String | 當日序號 |
| subject | String | 公告主旨 |
| spoke_time | String | 發言時間 |
| sentiment | Integer | 情緒分類（+1 正面 / 0 中性 / -1 負面） |

唯一鍵：`(stock_id, date, seq)`

### financial_statement（季報財務資料）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 季度結束日（如 2024-03-31） |
| year | Integer | 年度 |
| quarter | Integer | 季度（1~4） |
| revenue | BigInteger | 營收 |
| gross_profit | BigInteger | 毛利 |
| operating_income | BigInteger | 營業利益 |
| net_income | BigInteger | 稅後淨利 |
| eps | Float | 每股盈餘 |
| total_assets | BigInteger | 總資產 |
| total_liabilities | BigInteger | 總負債 |
| equity | BigInteger | 股東權益 |
| operating_cf | BigInteger | 營業現金流 |
| investing_cf | BigInteger | 投資現金流 |
| financing_cf | BigInteger | 融資現金流 |
| free_cf | BigInteger | 自由現金流（營業+投資） |
| gross_margin | Float | 毛利率 (%) |
| operating_margin | Float | 營益率 (%) |
| net_margin | Float | 淨利率 (%) |
| roe | Float | 股東權益報酬率 (%) |
| roa | Float | 資產報酬率 (%) |
| debt_ratio | Float | 負債比 (%) |

唯一鍵：`(stock_id, date)`

### stock_info（股票基本資料，P8 新增）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| stock_name | String | 股票名稱 |
| industry_category | String | 產業分類 |
| listing_type | String | 上市/上櫃 |
| updated_at | DateTime | 更新時間 |

唯一鍵：`(stock_id)`

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
