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
│   ├── settings.yaml        # 系統設定檔
│   └── concepts.yaml        # 概念股定義（CoWoS封裝/散熱模組/低軌衛星/AI伺服器/車用電子）
├── data/
│   ├── stock.db             # SQLite 資料庫
│   ├── raw/                 # 原始資料（保留）
│   └── processed/           # 處理後資料（保留）
├── src/
│   ├── config.py            # 設定管理模組
│   ├── constants.py         # 全系統共用常數（交易成本/DB/ETL/籌碼門檻）
│   ├── entry_exit.py        # 進出場共用純函數（ATR 止損止利/進場觸發/時機評估，Discover/Suggest/Watch 三系統共用）
│   ├── data/
│   │   ├── database.py      # 資料庫引擎與 Session 管理
│   │   ├── schema.py        # ORM 資料表定義（23 張表，含 DiscoveryRecord、WatchEntry、StockValuation、HoldingDistribution、SecuritiesLending、BrokerTrade、Watchlist、DailyFeature、ConceptGroup、ConceptMembership）
│   │   ├── fetcher.py       # FinMind API 資料抓取（含財報 EAV pivot）
│   │   ├── twse_fetcher.py  # TWSE/TPEX 官方開放資料（全市場）；TDCC 集保戶股權分散；分點交易 DJ 端點；借券賣出 TWT96U
│   │   ├── mops_fetcher.py  # MOPS 公開資訊觀測站（重大訊息 + 全市場月營收）；情緒分類；事件類型分類
│   │   ├── pipeline.py      # ETL Pipeline（抓取→清洗→寫入）；Feature Store 計算
│   │   ├── validator.py     # 資料品質檢查（6 項檢查 + 報告）
│   │   ├── io.py            # 通用匯出/匯入（CSV/Parquet）
│   │   └── migrate.py       # SQLite Schema 遷移
│   ├── features/
│   │   ├── indicators.py   # 技術指標計算（SMA/RSI/MACD/BB/ADX）；週K聚合 aggregate_to_weekly()
│   │   └── ml_features.py  # ML 特徵工程
│   ├── strategy/
│   │   ├── base.py          # 策略抽象基類（含除權息調整）
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
│   │   ├── scanner/         # 全市場選股 package（_base.py 基類 + 五模式 + _functions.py 共用純函數）
│   │   ├── universe.py      # Universe Filtering 三層漏斗（UniverseConfig + UniverseFilter）
│   │   └── performance.py   # Discover 推薦績效回測
│   ├── report/
│   │   ├── engine.py        # 每日選股報告引擎（四維度評分）
│   │   ├── formatter.py     # Discord 訊息格式化
│   │   └── ai_report.py     # AI 選股摘要（Claude API claude-sonnet-4-6）
│   ├── regime/
│   │   └── detector.py      # 市場狀態偵測（bull/bear/sideways/crisis 四狀態，三訊號多數決 + 市場寬度降級 + Crisis 覆蓋 + Hysteresis 狀態機）
│   ├── strategy_rank/
│   │   └── engine.py        # 策略回測排名引擎
│   ├── industry/
│   │   ├── analyzer.py      # 產業輪動分析；產業同儕相對強度計算
│   │   └── concept_analyzer.py  # 概念股輪動分析引擎（動能/法人流/相關性候選）
│   ├── notification/
│   │   └── line_notify.py   # Discord Webhook 通知
│   ├── backtest/
│   │   ├── engine.py        # 回測引擎 + 風險管理 + 績效計算（含 ATR-based 止損止利）
│   │   ├── portfolio.py     # 投資組合回測引擎（4 種配置模式）
│   │   ├── walk_forward.py  # Walk-Forward 滾動驗證引擎
│   │   └── attribution.py   # 五因子歸因分析（FactorAttribution）
│   ├── optimization/
│   │   └── grid_search.py   # Grid Search 參數優化
│   ├── scheduler/
│   │   ├── simple_scheduler.py  # 前景排程器（含每日報告）
│   │   └── windows_task.py      # Windows Task Scheduler 產生器
│   └── visualization/
│       ├── app.py            # Streamlit 儀表板入口
│       ├── data_loader.py    # 資料查詢模組
│       ├── charts.py         # Plotly 圖表元件
│       └── pages/            # 頁面模組（12 頁）
│           ├── market_overview.py     # 市場總覽首頁
│           ├── stock_analysis.py      # 個股分析頁
│           ├── backtest_review.py     # 回測結果頁
│           ├── portfolio_review.py    # 投資組合頁
│           ├── strategy_comparison.py # 策略比較頁
│           ├── screener_results.py    # 選股篩選頁
│           ├── ml_analysis.py         # ML 策略分析頁
│           ├── industry_rotation.py   # 產業輪動分析頁
│           ├── concept_rotation.py    # 概念輪動分析頁
│           ├── discovery_history.py   # 推薦歷史頁
│           └── position_monitoring.py # 持倉監控頁
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

# Anthropic API（AI 選股摘要功能，可選）
anthropic:
  api_key: "你的 anthropic api key"    # 需要 discover --ai-summary 功能時填入

# 量化參數（可選，未設定則使用程式預設值）
# quant:
#   trading_cost:
#     commission_rate: 0.001425      # 手續費
#     tax_rate: 0.003                # 交易稅
#     slippage_rate: 0.0005          # 滑價
#   atr_multiplier:
#     bull_stop: 1.5                 # 多頭止損 ATR 倍數
#     crisis_stop: 3.0               # 危機止損 ATR 倍數
#   score_threshold:
#     bull: 0.45                     # 各 Regime 最低評分門檻
#     crisis: 0.60
```

若要新增關注股票，在 `watchlist` 下加入股票代號，或使用 DB-based watchlist 管理（推薦）：

```bash
# 一次性從 YAML 匯入至 DB
python main.py watchlist import

# 之後使用 CLI 動態管理
python main.py watchlist add 2330 --name 台積電
python main.py watchlist remove 2330
python main.py watchlist list
```

> **DB 優先**：若 DB watchlist 非空，所有指令（sync、discover、anomaly-scan 等）優先使用 DB 清單；DB 為空時自動 fallback 至 `settings.yaml` 中的 watchlist。

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
| ADX | adx_14 | 平均方向性指標 (14日，SwingScanner 趨勢強度因子) |

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

# 五因子歸因分析（momentum/reversal/quality/size/liquidity 因子暴露 × 期間報酬相關係數）
python main.py backtest --stock 2330 --strategy sma_cross --attribution

# 匯出交易明細 CSV（含持倉天數、出場原因、止損/目標價）
python main.py backtest --stock 2330 --strategy sma_cross --export-trades trades.csv
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

ML 策略使用機器學習模型預測未來 N 天漲跌，自動建構 25+ 技術特徵（動量、均線比率、波動率、成交量比率、布林位置、交互特徵、Lag 特徵等），依訓練比例分割歷史資料進行訓練與測試。XGBoost 未安裝時自動 fallback 到 Random Forest。

```bash
# ML 策略回測
python main.py backtest --stock 2330 --strategy ml_random_forest
python main.py backtest --stock 2330 --strategy ml_xgboost
python main.py backtest --stock 2330 --strategy ml_logistic

# Phase C 增強功能
python main.py backtest --stock 2330 --strategy ml_random_forest --shap              # SHAP 特徵重要性 Top-10
python main.py backtest --stock 2330 --strategy ml_random_forest --optuna            # Optuna 超參數調優
python main.py backtest --stock 2330 --strategy ml_random_forest --shap --feature-selection  # SHAP 篩選 + 重新訓練
python main.py backtest --stock 2330 --strategy ml_random_forest --shap --optuna     # 組合使用
```

**Phase C ML 增強參數：**

| 參數 | 說明 |
|------|------|
| `--shap` | 使用 SHAP TreeExplainer 輸出特徵重要性排名 Top-10 |
| `--optuna` | 使用 Optuna 進行超參數調優（30 trials，TimeSeriesSplit 3-fold） |
| `--feature-selection` | 基於 SHAP 篩選特徵（移除 bottom 20%），需搭配 `--shap` |

所有 ML 策略均自動執行 TimeSeriesSplit 5-fold 交叉驗證，並輸出平均準確率 ± 標準差。

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
- 滑價: 0.05%（固定模式）；可啟用動態滑價（三因子：成交量衝擊 + OHLC spread + 上限 cap 1%）
- 訊號延遲: T+1（預設，消除 look-ahead bias；可設 signal_delay=0 向後相容）
- 初始資金: 1,000,000 元

#### 動態滑價 & 流動性約束（Phase A 回測可信度強化）

```python
# BacktestConfig 參數（程式碼層級，非 CLI 旗標）：
# dynamic_slippage=True  → 三因子動態滑價：
#   1. 成交量衝擊 base + k / √(volume)
#   2. OHLC spread proxy (high-low)/close × weight
#   3. 上限 cap（slippage_max=1%，防止極端值）
# liquidity_limit=0.05   → 單筆交易量 ≤ 當日成交量 × 5%
# signal_delay=1          → T+1 訊號延遲（預設，消除 look-ahead bias）
```

| 成交量級 | 固定滑價 | 動態滑價（k=0.5） |
|----------|----------|-------------------|
| 3000 萬股（TSMC 級） | 0.05% | ~0.06% |
| 100 萬股（中型股） | 0.05% | ~0.10% |
| 5 萬股（小型股） | 0.05% | ~0.27% |
| 1 萬股（冷門股） | 0.05% | ~0.55%（cap at 1%） |

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
| `--attribution` | 回測後輸出五因子歸因分析（momentum/reversal/quality/size/liquidity 因子暴露與報酬 Pearson 相關係數）；Dashboard 回測結果頁亦會顯示因子貢獻長條圖 |
| `--export-trades PATH` | 匯出交易明細至 CSV（含持倉天數、出場原因、止損/目標價） |

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

瀏覽器會自動開啟 `http://localhost:8501`，包含**十二個頁面**：

- **市場總覽**: TAIEX 走勢 K 線 + SMA60/120 + Regime 狀態、市場廣度（漲跌家數）、法人買賣超排行、產業熱度 Treemap
- **個股分析**: K線圖 + SMA/BB/RSI/MACD 疊加 + 成交量 + 法人買賣超 + 融資融券 + MOPS 公告 vline 標記
- **回測結果**: 績效摘要卡片（含進階指標）+ 權益曲線/回撤圖 + 交易明細（含出場原因）+ 回測比較表 + 五因子歸因分析圖
- **投資組合**: 組合回測績效 + 個股配置圓餅圖 + 個股報酬柱狀圖 + 交易明細
- **策略比較**: 多策略同場比較（1~5 個策略 × 1 支股票）— 績效指標表 + 累積報酬多線折線圖 + 進階指標長條圖
- **選股篩選**: 多因子條件篩選 + 因子分數排名 + CSV 匯出
- **ML 策略分析**: 模型訓練（準確率、特徵重要性、預測機率分佈）+ Walk-Forward 滾動驗證
- **產業輪動**: 產業綜合排名 + 泡泡圖（法人 vs 動能）+ 法人淨買超長條圖 + 精選個股
- **概念輪動**: 概念股排名（動能+法人流 Percentile Rank）+ Treemap + 箱型圖
- **推薦歷史**: Discover 推薦歷史視覺化（日曆熱圖、績效分析、個股排行、歷史明細 CSV 匯出）
- **持倉監控**: 持倉總覽（狀態/損益/距目標%）+ 個股 K 線走勢（含進場/止損/目標水平線）+ 預警列表（觸發止損/止利/過期）

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

從全台灣 ~6000 支股票（上市 + 上櫃）中自動篩選出值得關注的標的。支援六種掃描模式：

- **momentum**（預設）：短線動能股（1~10 天），抓突破 + 資金流 + 量能擴張
- **swing**：中期波段股（1~3 個月），抓趨勢 + 基本面 + 法人布局
- **value**：價值修復股，低估值 + 基本面轉佳 + 法人布局
- **dividend**：高息存股，高殖利率 + 配息穩定 + 估值合理
- **growth**：高成長股，營收高速成長 + 動能啟動
- **all**：五模式綜合比較，一次執行全部 Scanner，輸出交叉出現次數排行（出現越多模式 = 高信心度）

**漏斗架構：**
```
全市場 ~6000 支 → 粗篩 ~150 支 → 細評排名 → 風險過濾 → 量價背離調整 → 動態評分閾值 → 排名 → 同產業分散化 → Top N 推薦
```

**品質控制機制（P0 三層防護）：**
1. **量價背離偵測**（Stage 3.6）：計算近 5 日價格與成交量的 Pearson 相關係數，價漲量縮（corr < -0.3）降分 -5%、價漲量增（corr ≥ 0.3）加分 +2%，過濾假突破
2. **動態評分閾值**（Stage 3.7）：依市場狀態設定最低分門檻（bull=0.45 / sideways=0.50 / bear=0.55 / crisis=0.60），Regime 越差門檻越高，寧缺勿濫
3. **同產業分散化**（Stage 4.1）：同產業推薦不超過總推薦數 25%（至少 3 檔），超出部分由其他產業遞補，降低集中風險

**訊號品質強化機制（P1 三層）：**
1. **動量衰減偵測**（Stage 3.5c）：RSI 頂背離（價格新高但 RSI 未新高）+ MACD 柱狀連縮（histogram 連續 2 天縮短），兩訊號同時觸發 -6%、單一觸發 -3%，避免追在動能頂端（僅 momentum/growth 模式）
2. **籌碼加速度加成**（Stage 3.5d）：比較近 3 日 vs 前 7 日法人平均淨買超，加速買入 +4%、溫和加速 +2%，獎勵有品質的籌碼流入
3. **多時框強制共振**（Stage 3.5e）：日線 × 週線方向一致 +4%；**momentum/growth 模式中日多週空矛盾者直接排除**（追高風險最大），其他模式僅降分 -2%~-4%

**籌碼 & 基本面精準度強化（P1）：**
1. **法人金額加權**（B1）：MomentumScanner 外資連續性因子從純天數升級為 `Σ(net × 0.85^days_ago)` 衰減加權，混合比：金額加權 60% + 天數 40%。大額且持續 > 小額且持續 > 大額一次性
2. **盈餘品質分數**（C1，Value/Dividend 模式）：三子指標等權 — 現金流品質（OCF/NI）、收入品質（淨利 vs 營收增速差異）、負債穩定性（debt_ratio），權重 15%。過濾價值陷阱
3. **回撤降頻**（D3）：TAIEX 20 日回撤 > -10% 正常推薦；-10%~-15% → momentum/swing/growth 砍半；< -15% → 砍至 1/3。value/dividend 防禦型模式不受影響

**資料來源優先順序：**
1. TWSE/TPEX 官方開放資料（免費，6 次 API 呼叫取得全市場，含融資融券）
2. FinMind 批次 API（需付費帳號）
3. FinMind 逐股抓取（免費帳號備案，較慢）

**市場狀態（Regime）自動偵測：** 系統自動根據加權指數（TAIEX）判斷市場狀態（bull/bear/sideways/crisis 四狀態），動態調整各模式的四維度因子權重（技術面 + 籌碼面 + 基本面 + 消息面）。多頭加重技術面，空頭加重基本面與消息面，崩盤（crisis）大幅加重消息面。偵測機制：三訊號多數決 + 市場寬度降級（>60% 股票跌破 MA20 降一級）+ Crisis 快速覆蓋（4 訊號 ≥2 觸發：5 日跌>5%、連跌≥3 天、波動率 1.8x、爆量長黑）+ Hysteresis 狀態機（sideways→bull 需 3 天確認，bull→sideways 快速降級 1 天，crisis 退出需低點遞增+波動率回落）。

**消息面評分（MOPS 重大訊息）：** 系統自動從公開資訊觀測站抓取近期重大訊息，採四項機制計算消息面分數：
1. **事件類型加權**：governance_change=5.0（董監改選/市場派）、buyback=4.0（庫藏股）、earnings_call=3.0、investor_day=2.0、filing=1.5、revenue=1.2、general=1.0
2. **時間衰減**：`exp(-0.2 × days_ago) × type_weight`，decay 常數 0.2（7 天後保留 ~25%）
3. **異常公告率**：Z-Score vs 180 天基準期，Z > 2 乘數最高 +50%，Z < -1 降至 70%
4. **Regex 上下文過濾**：優先以情境 pattern 判斷情緒（正確區分「處分利益」vs「處分持股」、「澄清衰退」vs「澄清報導」等）

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

# ── all 模式：五模式綜合比較 ──
# 一次跑完五個 Scanner，輸出交叉排行（出現越多模式 = 高信心度）
python main.py discover all --skip-sync --top 20

# 只顯示出現在 2 個以上模式的股票
python main.py discover all --skip-sync --min-appearances 2

# 匯出交叉比較表 CSV
python main.py discover all --skip-sync --export compare.csv

# 啟用週線多時框確認（週線多頭 +5%，週線空頭 -5%）
python main.py discover momentum --weekly-confirm
python main.py discover all --skip-sync --weekly-confirm --min-appearances 2

# 啟用 AI 選股摘要（需設定 anthropic.api_key）
python main.py discover momentum --skip-sync --ai-summary
python main.py discover swing --top 20 --ai-summary --notify
```

每次執行 `discover` 時，推薦結果會自動存入 DB（`discovery_record` 表），供歷史追蹤使用。同日同模式重跑會覆蓋先前記錄。`all` 模式會將五個模式分別存入 DB。

**`all` 模式輸出欄位說明：**

| 欄位 | 說明 |
|------|------|
| `appearances` | 出現模式數（越多 = 越多模式確認） |
| `avg_score` | 各模式 composite_score 加權平均（主要排序依據） |
| `best_mode` | 最高分模式中文名稱（如「短線動能」） |
| `chip_tier` | 籌碼評分最高 Tier（如「8F」、「7F」、「N/A"） |

| 參數 | 說明 |
|------|------|
| `mode` | 掃描模式：`momentum`（短線動能）、`swing`（中期波段）、`value`（價值修復）、`dividend`（高息存股）、`growth`（高成長）、`all`（五模式綜合比較），預設 momentum |
| `--top N` | 顯示前 N 名（預設 20） |
| `--min-price N` | 最低股價門檻（預設 10） |
| `--max-price N` | 最高股價門檻（預設 2000） |
| `--min-volume N` | 最低成交量/股（預設 500000） |
| `--sync-days N` | 同步最近幾個交易日（預設 3，swing 模式自動擴展至 80，all 模式自動取 max(sync_days, 80)） |
| `--skip-sync` | 跳過全市場資料同步，直接用 DB 既有資料 |
| `--export PATH` | 匯出 CSV |
| `--notify` | 發送 Discord 通知 |
| `--compare` | 顯示與上次推薦的差異（新進/退出/排名變動 >= 3 名，單模式有效） |
| `--min-appearances N` | [all 模式] 只顯示出現在 N 個以上模式的股票（預設 1 = 全部顯示） |
| `--weekly-confirm` | 啟用週線多時框確認：從 DB 讀取近 90 天日K 聚合週K，SMA13 + RSI14 週線信號同為多頭 → composite_score ×1.05（+5%），同為空頭 → ×0.95（-5%），預設關閉 |
| `--ai-summary` | 掃描完成後呼叫 Claude API（`claude-sonnet-4-6`）生成約 300 字繁體中文摘要（市場狀態分析 + 前三名亮點 + 風險提示），需在 `config/settings.yaml` 設定 `anthropic.api_key` |

**Regime 自適應進出場 ATR 參數（discover 輸出的 stop_loss / take_profit）：**

| 市場狀態 | 止損倍數（ATR） | 目標倍數（ATR） | 說明 |
|---------|--------------|--------------|------|
| Bull（多頭） | 1.5× | 3.5× | 擴大獲利空間 |
| Sideways（盤整，預設） | 1.5× | 3.0× | 標準風險報酬 1:2 |
| Bear（空頭） | 1.2× | 2.5× | 縮緊止損，降低虧損 |
| Crisis（崩盤） | 1.0× | 1.8× | 合理止損距離，避免日內波動洗出；建議降低部位規模 |

**Momentum 模式（sideways 基準權重，bull/bear 自動微調）：**

| 維度 | 權重 | 因子 |
|------|------|------|
| 技術面 | 45% | 5日動能、10日動能、20日突破、量比、成交量加速 |
| 籌碼面 | 45% | 最高 8 因子，依資料可用性自動升降級（詳見下方「籌碼面因子分級」） |
| 基本面 | 10% | 營收 YoY > 0 過濾（加分/不加分） |
| 風險過濾 | — | ATR(14)/close > 80th percentile 或 > 8% 剔除（取嚴格者） |

**籌碼面因子分級（MomentumScanner，依資料可用性自動選擇最高 Tier）：**

| Tier | 啟用條件 | 外資 | 量比 | 法人 | 券資比 | 大戶 | 借券 | 分點HHI | 智慧分點 |
|------|---------|:----:|:----:|:----:|:------:|:----:|:----:|:-------:|:-------:|
| **8F** | 分點歷史 ≥ 20 交易日（buy_price/sell_price）+ 全部 7F 條件 | 18% | 16% | 16% | 10% | 12% | 7% | 11% | **10%** |
| 7F | 分點HHI + 借券 + 融資券 + 大戶 | 20% | 18% | 18% | 11% | 13% | 8% | 12% | — |
| 6F | 分點HHI + 借券 + 融資券（無大戶） | 22% | 20% | 20% | 14% | — | 12% | 12% | — |
| 5F | 分點HHI + 借券（無融資券） | 28% | 22% | 22% | — | — | 14% | 14% | — |
| 4F | 僅分點HHI | 32% | 24% | 24% | — | — | — | 20% | — |
| 3F | 基本（無分點資料） | 40% | 30% | 30% | — | — | — | — | — |

**智慧分點因子（8F 第 8 因子）說明：**

| 子因子 | 權重 | 說明 |
|--------|------|------|
| Smart Broker Score | 60% | 歷史精準分點（win_rate ≥ 0.60、Profit Factor ≥ 1.50、sell_events ≥ 3、買入總額 ≥ 500 萬）的近期加碼評分。Smart_Score = Σ(歷史損益 × 近3日淨買超)，賺錢贏家近期加碼貢獻更大 |
| Accumulation Broker Score | 40% | 純蓄積型地緣分點（sell_ratio ≤ 10%、淨持倉 > 0、後60天倉位 > 前60天），成本線具底撐效果 |

> **資料前提（自適應累積設計）**：8F 採用「自然累積」策略，`_load_broker_data_extended()` 查詢最近 365 天 DB 歷史，並要求每股 **≥ 20 個交易日**的資料才啟用 Smart Broker（不足自動降回 7F）。
>
> - **首次部署（一次性）**：執行 `python main.py sync-broker --watchlist-bootstrap` 補齊 watchlist 所有股票的最近 120 個交易日分點歷史（半年），Bootstrap 後即可觸發 8F。若有額外股票清單，可搭配 `--from-file stocks.txt` 一次性補齊任意股票（支援純文字或 CSV 格式，`--days` 可自訂天數）。
> - **日常運作**：`morning-routine` Step 8 每日同步 watchlist（5 日），歷史資料自然累積，120 天後準確度最高。
> - **新股加入 watchlist**：累積約 1 個月後自動升級至 8F，無需手動操作。

**Swing 模式：**

| 維度 | 權重 | 因子 |
|------|------|------|
| 技術面 | 30% | 趨勢確認(close>SMA60)、均線排列(SMA20>SMA60)、60日動能、量價齊揚 |
| 籌碼面 | 30% | 投信淨買超(50%)、法人20日累積買超(50%) |
| 基本面 | 40% | 營收YoY(40%)、MoM(30%)、營收加速度(30%) |
| 風險過濾 | — | 年化波動率 > 85th percentile 或 > 80% 剔除（取嚴格者） |

**Value 模式：**

| 維度 | 權重 | 因子 |
|------|------|------|
| 基本面 | 50% | 營收YoY(40%)、ROE(25%)、毛利率QoQ(20%)、EPS YoY(15%)；財報資料不足時自動降回純營收分 |
| 估值面 | 30% | PE反向排名(40%)、PB反向排名(30%)、殖利率排名(30%) |
| 籌碼面 | 20% | 投信近期買超(50%)、法人累積買超(50%) |
| 粗篩門檻 | — | PE > 0 且 < **同業 PE 中位數 × 1.5**（同業 < 3 支時 fallback PE < 50）、殖利率 > 2% |
| 風險過濾 | — | 近20日波動率 > 90th percentile 或 > 5%日波動率 剔除（取嚴格者） |

**Dividend 模式：**

| 維度 | 權重 | 因子 |
|------|------|------|
| 基本面 | 35~45% | 營收YoY(40%)、EPS 穩定性（近4季無虧損，倒排）(35%)、配息率代理（正EPS比例）(25%)；財報不足時降回純營收分 |
| 殖利率面 | 25~35% | 殖利率排名(50%)、PE反向排名(30%)、PB反向排名(20%) |
| 籌碼面 | 10~20% | 投信近期買超(50%)、法人累積買超(50%) |
| 粗篩門檻 | — | 殖利率 > 3%、PE > 0、**EPS 連續性**（近4季任一季 EPS ≤ 0 者排除；無財報資料者 pass through 不誤殺） |
| 風險過濾 | — | 近20日波動率 > **75th percentile** 或 > 5%日波動率 剔除（防禦性策略收緊波動門檻，避免價值陷阱） |

**Growth 模式：**

| 維度 | 權重 | 因子 |
|------|------|------|
| 基本面 | 40~50% | 營收YoY(40%)、加速度（本月YoY − 3個月前YoY）(25%)、毛利率YoY加速(20%)、EPS季增率(15%)；財報不足時降回純YoY+加速度 |
| 技術面 | 15~30% | 5日動能、10日動能、20日突破、量比、成交量加速 |
| 籌碼面 | 15~20% | 外資連續買超天數 + 買超佔量比 + 合計買超 + 券資比（有資料時） |
| 粗篩門檻 | — | 營收 YoY > 10% |
| 風險過濾 | — | ATR(14)/close > 80th percentile 或 > 8% 剔除（取嚴格者） |

**Regime 權重調整矩陣（系統自動偵測）：**

| 模式 | 面向 | 多頭 | 盤整 | 空頭 | 崩盤 (crisis) |
|------|------|------|------|------|---------------|
| Momentum | Tech / Chip / Fund / News | 45/35/10/10 | 40/40/10/10 | 30/40/15/15 | 10/30/20/40 |
| Swing | Tech / Chip / Fund / News | 30/20/40/10 | 25/25/35/15 | 15/25/45/15 | 5/20/50/25 |
| Value | Fund / Val / Chip / News | 40/35/15/10 | 45/25/15/15 | 50/20/10/20 | 55/15/10/20 |
| Dividend | Fund / Div / Chip / News | 35/35/20/10 | 40/30/15/15 | 45/25/10/20 | 55/15/10/20 |
| Growth | Fund / Tech / Chip / News | 45/30/15/10 | 40/25/20/15 | 50/15/15/20 | 55/5/15/25 |

**篩選流程說明：**

| 階段 | 動作 | 說明 |
|------|------|------|
| Stage 0 | 市場狀態偵測 | 根據 TAIEX 判斷 bull/bear/sideways，動態調整權重 |
| Stage 0.5 | 資料冷啟動補抓 | **growth**：月營收覆蓋 < 500 支時自動從 MOPS 同步；**value/dividend**：估值資料（PE/PB/殖利率）覆蓋 < 500 支時自動從 TWSE/TPEX 同步 |
| Stage 1 | 資料載入 | 從 DB 讀取全市場日K + 三大法人 + 融資融券 |
| Stage 2 | 粗篩 | 模式專屬條件篩選，取前 ~150 名 |
| Stage 2.5 | 補抓候選股資料 | **MomentumScanner**：自動從 FinMind 補抓月營收 + 最近 5 天分點交易資料（`sync_broker_for_stocks`）。其他模式僅補抓月營收/估值 |
| Stage 2.7 | 公告載入 | 從 DB 載入候選股近期 MOPS 重大訊息 |
| Stage 3 | 細評 | 四維度因子（技術+籌碼+基本面+消息面）+ Regime 動態權重評分 |
| ↳ 籌碼面（Momentum） | 智慧分點計算 | 從 DB 載入候選股最近 365 天分點歷史（`buy_price`/`sell_price`），需 ≥ 20 個交易日才識別 Smart Broker（高勝率+高獲利因子）與 Accumulation Broker（蓄積型地緣分點），有資料時升級至 8-Factor |
| Stage 3.3 | 產業加成 | 用 IndustryRotationAnalyzer 計算產業排名，熱門產業 +5%、冷門產業 -5% 線性加成到 composite_score |
| Stage 3.3a | 產業同儕相對強度 | 個股近 20 日報酬率 vs 同產業中位數：超越 +20pp → composite_score ×1.03，落後 -20pp → ×0.97 |
| Stage 3.4 | 週線趨勢加成（可選） | `--weekly-confirm` 啟用時：從 DB 讀取近 90 天日K → 聚合週K → SMA13 + RSI14 週線信號，同為多頭 → composite_score ×1.05，同為空頭 → ×0.95 |
| Stage 3.5 | 風險過濾 | 剔除高波動股 |
| Stage 3.5b | **Crisis 雙重過濾** | 僅 crisis regime 執行：(1) 相對強度 — 剔除 20 日超額報酬低於 TAIEX -10% 的弱勢股；(2) 絕對趨勢 — 剔除收盤價跌破 MA60 的股票（防止選到隨大盤跳水的標的） |
| Stage 4 | 排名輸出 | 加上產業標籤與股票名稱，統計產業分布 |

> **注意**：Stage 2.5 補抓月營收需要 FinMind API Token；若無 Token，基本面分數 fallback 到 0.5（中性值），不影響其他維度評分。智慧分點（8F）採自適應累積設計，需 ≥ 20 個交易日分點歷史資料（由 `morning-routine` 每日自動累積），資料不足時自動降回 7F。

> **Crisis 模式**：當 TAIEX 觸發快速崩盤訊號（5 日跌幅 > 5%、連跌 ≥3 天、波動率飆升 1.8x、爆量長黑，4 訊號 ≥2 觸發），系統自動切換 crisis regime：評分權重向消息面/基本面傾斜（技術訊號在崩盤時失真），ATR 止損收窄至 1.0×（避免日內波動洗出），只保留相對強勢股（剔除跑輸 TAIEX >10% 及跌破 MA60 者）。`morning-routine` Step 0 會於掃描前顯示預警並推播 Discord。

### 4.14 Discover 推薦績效回測 (`discover-backtest`)

評估歷史 `discover` 推薦的實際表現：讀取 DiscoveryRecord 歷史記錄，對照 DailyPrice 計算推薦後 N 天的實際報酬率，輸出勝率/平均報酬/最大虧損等統計。

```bash
# 評估 momentum 推薦績效（預設持有 5,10,20 天）
python main.py discover-backtest --mode momentum

# swing 模式預設持有 20,40,60 天（波段持有期）
python main.py discover-backtest --mode swing

# 自訂持有天數（覆蓋模式預設值）
python main.py discover-backtest --mode swing --days 5,10,20,60

# 只看每次掃描前 10 名的績效
python main.py discover-backtest --mode value --top 10

# 指定掃描日期範圍
python main.py discover-backtest --mode momentum --start 2025-06-01 --end 2025-12-31

# 匯出明細 CSV
python main.py discover-backtest --mode momentum --export result.csv

# 含交易成本（手續費+稅+滑價）
python main.py discover-backtest --mode momentum --include-costs

# T+1 開盤價進場（消除 look-ahead bias）
python main.py discover-backtest --mode momentum --entry-next-open

# 同時啟用成本 + T+1 開盤進場
python main.py discover-backtest --mode momentum --include-costs --entry-next-open
```

**參數說明：**

| 參數 | 說明 |
|------|------|
| `--mode` | 必填，掃描模式：`momentum` / `swing` / `value` / `dividend` / `growth` |
| `--days` | 持有天數，逗號分隔（swing 預設 `20,40,60`，其他模式預設 `5,10,20`） |
| `--top` | 只計算每次掃描前 N 名的績效（預設全部） |
| `--start` | 掃描日期範圍起始（YYYY-MM-DD） |
| `--end` | 掃描日期範圍結束（YYYY-MM-DD） |
| `--export` | 匯出明細 CSV 路徑 |
| `--include-costs` | 績效計算納入交易成本（手續費 0.1425% + 交易稅 0.3% + 滑價 0.05%） |
| `--entry-next-open` | 以 T+1 開盤價作為進場價（預設使用推薦日收盤價） |

**輸出三層聚合：**

1. **整體摘要**：每個持有天數的勝率、平均報酬、中位數、最大獲利/虧損
2. **逐次掃描**：每個掃描日期的平均報酬、勝率、最佳/最差個股
3. **個股明細**：每筆推薦的報酬率（供 `--export` 匯出）

> **前提**：須先有足夠的 `discover` 歷史記錄（每次執行 `discover` 會自動存入 DB），以及推薦日之後的 DailyPrice 資料。

### 4.15 同步 MOPS 重大訊息 (`sync-mops`)

從公開資訊觀測站（MOPS）備援站抓取上市/上櫃公司最新重大訊息公告。資料用於 discover 的消息面評分。

```bash
# 同步最近 7 天（預設）
python main.py sync-mops

# 同步最近 30 天
python main.py sync-mops --days 30
```

**參數說明：**

| 參數 | 說明 |
|------|------|
| `--days N` | 同步最近 N 天的公告（預設 7） |

系統會自動對公告主旨進行情緒分類（+1 正面 / 0 中性 / -1 負面）及事件類型分類（governance_change / buyback / earnings_call / investor_day / filing / revenue / general），分類使用 Regex 上下文比對，正確處理「處分利益」vs「處分持股」等模糊語境。同步完成後顯示情緒與事件類型分布統計。

> **建議**：搭配每日排程使用，逐日累積公告歷史。`discover` 命令的全市場同步也會自動附帶 MOPS 同步。

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
- 為 `trade` 表新增 `exit_reason`、`stop_price`、`target_price` 欄位
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

### 4.23 單股進出場建議 (`suggest`)

針對單一股票，從 DB 讀取最近 60 日日K線，計算 ATR14／SMA20／RSI14 並偵測市場 Regime，輸出詳細的進出場建議。

```bash
python main.py suggest <stock_id>          # 輸出進場區間/止損/目標價/時機評估
python main.py suggest 2330                # 台積電進出場建議
python main.py suggest 2317 --notify       # 附加 Discord 通知
```

**參數說明：**

| 參數 | 說明 |
|------|------|
| `stock_id` | 股票代號（必填，例：2330） |
| `--notify` | 發送 Discord 通知 |

**計算邏輯：**

- **ATR14**：14 日 Average True Range（True Range 定義：max(H-L, |H-prev_C|, |L-prev_C|)）
- **進場價**：最新收盤價
- **止損價**：進場價 - stop_mult × ATR14（**依 Regime 自適應調整**）
- **目標價**：進場價 + target_mult × ATR14（**依 Regime 自適應調整**）
- **Regime 自適應 ATR 倍數**：bull=(1.5×, 3.5×)、sideways=(1.5×, 3.0×)、bear=(1.2×, 2.5×)、crisis=(1.0×, 1.8×)
- **SMA20**：最近 20 日收盤均線，判斷均線位置
- **RSI14**：Wilder EWM 平滑，判斷超買（≥70）/ 超賣（≤30）
- **市場 Regime**：讀取 TAIEX（加權指數）→ 多頭 / 空頭 / 盤整 / 崩盤

> **Regime 自適應**：suggest 與 discover 使用相同的 ATR 倍數常數（`src/entry_exit.py:REGIME_ATR_PARAMS`），確保止損/目標價在不同市場狀態下一致。crisis 模式止損更窄（1.0×）、目標更保守（1.8×），避免崩盤期過度曝險。

**輸出範例：**

```
════════════════════════════════════════════════════════════
  單股進出場建議  ｜  2330 台積電
════════════════════════════════════════════════════════════
  分析日期  ：2026-03-01
  最新收盤  ：850.00
  SMA20    ：832.50
  RSI14    ：62.3
  ATR14    ：18.45（2.17%）
────────────────────────────────────────────────────────────
  市場 Regime ：多頭（TAIEX 22,350）
────────────────────────────────────────────────────────────
  進場參考價  ：850.00
  止 損 價   ：822.33（-3.3%）
  目 標 價   ：905.35（+6.5%）
  風 險 報 酬 ：1 : 2.0
────────────────────────────────────────────────────────────
  進場觸發  ：站上均線
  時機評估  ：積極做多：動能強勁 + 趨勢向上
  建議有效至：2026-03-06
════════════════════════════════════════════════════════════
```

> **注意**：`suggest` 讀取 DB 現有資料，不執行網路同步。若資料過舊請先執行 `python main.py sync`。

---

### 4.24 持倉監控 (`watch`)

追蹤進場後的持倉狀態，自動比對最新收盤價標記止損/止利/過期。

```bash
# 新增持倉（自動計算 ATR14-based 止損止利，依 Regime 自適應調整）
python main.py watch add 2330

# 手動指定進場價/止損/目標/股數
python main.py watch add 2330 --price 580 --stop 555 --target 635 --qty 1000

# 從最新 discover 記錄匯入
python main.py watch add 2330 --from-discover momentum

# 啟用移動止損（ATR×1.5，預設倍數）
python main.py watch add 2330 --trailing

# 啟用移動止損並自訂 ATR 倍數
python main.py watch add 2330 --trailing --trailing-multiplier 2.0

# 列出持倉中記錄（[T×1.5] 標記代表移動止損類型）
python main.py watch list

# 列出全部記錄（含已平倉/止損/止利/過期）
python main.py watch list --status all

# 平倉（標記 closed，記錄平倉價）
python main.py watch close 1 --price 595

# 批次更新狀態（比對最新收盤價；移動止損自動向上追蹤最高價）
python main.py watch update-status
```

**watch add 參數：**

| 參數 | 說明 |
|------|------|
| `stock_id` | 股票代號（必填） |
| `--price P` | 進場價（預設使用最新收盤）|
| `--stop S` | 止損價（預設依 Regime 自適應：entry - stop_mult×ATR14）|
| `--target T` | 目標價（預設依 Regime 自適應：entry + target_mult×ATR14）|
| `--qty Q` | 股數（選填）|
| `--from-discover MODE` | 從最新 discover 記錄匯入（MODE: momentum/swing/value/dividend/growth）|
| `--notes TEXT` | 備註（選填）|
| `--trailing` | 啟用移動止損（Trailing Stop），每次 update-status 自動上移 |
| `--trailing-multiplier M` | ATR 倍數（預設 1.5）；止損 = 歷史最高價 - ATR14 × M |

**狀態說明：**

| 狀態 | 觸發條件 |
|------|----------|
| `active` | 持倉中（未觸發任何條件）|
| `stopped_loss` | 最新收盤 ≤ stop_loss |
| `taken_profit` | 最新收盤 ≥ take_profit |
| `expired` | 今日 > valid_until |
| `closed` | 手動執行 `watch close` 平倉 |

**移動止損（Trailing Stop）說明：**

- 新增持倉時加上 `--trailing` 旗標即啟用移動止損模式
- 每次執行 `watch update-status` 時，系統會：
  1. 從 DB 讀取最新日 K 資料，計算 ATR14
  2. 若最新收盤價 > `highest_price_since_entry`，更新歷史最高價
  3. 計算新止損 = `highest_price_since_entry` - ATR14 × `trailing_atr_multiplier`
  4. 只有「新止損 > 當前止損」時才更新（保證止損只升不降）
- Dashboard「個股走勢」Tab 以橙色虛線顯示移動止損，並顯示「追蹤最高」指標卡

> **Dashboard 整合**：啟動儀表板後，側邊欄選擇「👁️ 持倉監控」可視覺化持倉狀態，含 K 線圖（進場/止損/目標水平線）及預警列表。移動止損以橙色顯示，靜態止損以紅色顯示。

---

### 4.25 籌碼異動警報 (`anomaly-scan`)

從 DB 現有資料（DailyPrice、InstitutionalInvestor、SecuritiesLending、BrokerTrade）偵測五類量化異常訊號，無需額外 API 呼叫。

```bash
# 掃描 watchlist 中所有股票
python main.py anomaly-scan

# 指定股票清單
python main.py anomaly-scan --stocks 2330 2317 2454

# 自訂門檻
python main.py anomaly-scan --vol-mult 3.0 --inst-threshold 5000000 --sbl-sigma 3.0 --hhi-threshold 0.5

# 隔日沖風險門檻（預設 0.2）
python main.py anomaly-scan --dt-threshold 0.3

# 掃描並推播 Discord
python main.py anomaly-scan --notify
```

**五類異常訊號：**

| 訊號 | 資料來源 | 邏輯 | 預設門檻 |
|------|---------|------|---------|
| 📊 量能暴增 | `DailyPrice.volume` | 今日量 > 近 N 天均量 × 倍數 | lookback=10, vol-mult=2.0 |
| 🏦 外資大買超 | `InstitutionalInvestor` (外資) | 最新日外資淨買超 > 門檻 | inst-threshold=3,000,000 股（≈3,000張）|
| 🔴 借券賣出激增 | `SecuritiesLending.sbl_change` | 最新日 sbl_change > mean + σ×std | sbl-sigma=2.0 |
| 🎯 主力分點集中買進 | `BrokerTrade` | 最新日 HHI(淨買超分點) > 門檻 AND 淨買 > 0 | hhi-threshold=0.4 |
| ⚡ 隔日沖風險 | `BrokerTrade` | 三層偵測（行為配對+黑名單+即時大量），penalty > 門檻 | dt-threshold=0.2 |

**參數說明：**

| 參數 | 預設 | 說明 |
|------|------|------|
| `--stocks SID ...` | watchlist | 掃描股票清單 |
| `--lookback N` | 10 | 計算均量/均值的天數 |
| `--vol-mult F` | 2.0 | 量能倍數門檻 |
| `--inst-threshold N` | 3000000 | 外資淨買超股數門檻（股）|
| `--sbl-sigma F` | 2.0 | 借券激增標準差倍數 |
| `--hhi-threshold F` | 0.4 | 主力集中度 HHI 門檻（0~1）|
| `--dt-threshold F` | 0.2 | 隔日沖風險 penalty 門檻（0~1）|
| `--notify` | False | 掃描完成後推播 Discord |

> **資料準備**：需先執行 `python main.py sync`（DailyPrice/InstitutionalInvestor）、`python main.py sync-sbl`（借券）、`python main.py sync-broker`（分點）。`morning-routine` 已自動包含所有資料同步（Step 1~8）及 anomaly-scan（Step 14）。

---

### 4.26 DB-based 觀察清單管理 (`watchlist`)

將觀察清單從 `settings.yaml` 遷移至資料庫，支援動態新增/移除，無需手動編輯 YAML 檔案。

**DB 優先邏輯**：若 DB watchlist 非空，所有命令（sync、discover、anomaly-scan、revenue-scan 等）優先使用 DB 清單；DB 為空時自動 fallback 至 `settings.yaml` 的 watchlist。

```bash
# 列出目前有效 watchlist（DB 非空顯示 DB，DB 空顯示 YAML）
python main.py watchlist list

# 新增股票（可選名稱與備註）
python main.py watchlist add 2330
python main.py watchlist add 2330 --name 台積電 --note 核心持倉

# 移除股票
python main.py watchlist remove 2330

# 從 settings.yaml 一次性匯入（首次使用時執行）
python main.py watchlist import
```

**參數說明（watchlist add）：**

| 參數 | 說明 |
|------|------|
| `stock_id` | 股票代號（必填，例：2330）|
| `--name TEXT` | 股票名稱（可選，例：台積電）|
| `--note TEXT` | 備註（可選，例：核心持倉）|

> **遷移建議**：首次使用執行 `python main.py watchlist import` 將 YAML 清單匯入 DB，之後即可完全透過 CLI 管理，無需再修改 `settings.yaml`。

---

### sync-concepts / concepts / concept-expand — 概念股管理

概念股（CoWoS封裝、散熱模組、低軌衛星等）以「疊加方式」在現有產業加成之上新增 ±5% 的概念熱度加成，sector + concept 合計上限 ±8%。

#### 初次設定

```bash
# 1. 執行 migrate 確保 DB 有新表
python main.py migrate

# 2. 從 config/concepts.yaml 匯入概念定義
python main.py sync-concepts

# 3. 確認概念清單
python main.py concepts list
```

#### concepts.yaml 管理

```yaml
# config/concepts.yaml — 概念定義（已納入 git，非敏感資料）
concepts:
  CoWoS封裝:
    description: "台積電 CoWoS 先進封裝供應鏈"
    stocks: ["2330", "3034", "5347"]
```

修改 YAML 後重新執行 `sync-concepts`；若要重組概念（清除舊記錄），加 `--purge`。

#### 每日自動標記（MOPS 關鍵字）

```bash
# 掃描近 90 天 MOPS 公告，命中關鍵字時自動新增成員（source=mops）
python main.py sync-concepts --from-mops

# 自訂回溯天數
python main.py sync-concepts --from-mops --days 30
```

#### 手動管理成員

```bash
# 列出所有概念
python main.py concepts list

# 列出特定概念成員
python main.py concepts list CoWoS封裝

# 手動新增成員（source=manual）
python main.py concepts add CoWoS封裝 2330

# 移除成員
python main.py concepts remove CoWoS封裝 2330
```

#### 相關性候選推薦（P2）

```bash
# 找出與 CoWoS封裝 種子股相關係數 ≥ 0.7 的候選股
python main.py concept-expand CoWoS封裝 --threshold 0.7

# 自動加入 DB（source=correlation），無需確認
python main.py concept-expand CoWoS封裝 --threshold 0.7 --auto
```

**參數說明（concept-expand）：**

| 參數 | 說明 |
|------|------|
| `concept_name` | 概念名稱（必填，例：CoWoS封裝）|
| `--threshold` | 相關係數門檻（預設 0.7）|
| `--lookback` | 相關性計算回溯天數（預設 60）|
| `--auto` | 自動加入 DB，不需手動確認 |

---

### rotation — 輪動組合部位控制

自動化的部位控制系統，根據 discover 推薦排名管理模擬投資組合。支援等權配置、固定持有天數、排名淘汰、到期續持，以及歷史回測。

**建立組合：**

```bash
# 單模式：momentum Top-5，持有 3 天，100 萬資金
python main.py rotation create --name mom5_3d --mode momentum --max-positions 5 --holding-days 3 --capital 1000000

# 綜合模式：discover all 的 avg_score 排名，Top-10，持有 5 天
python main.py rotation create --name all10_5d --mode all --max-positions 10 --holding-days 5 --capital 2000000

# 停用續持（到期一律賣出再重買）
python main.py rotation create --name all10_5d --mode all --max-positions 10 --holding-days 5 --capital 2000000 --no-renewal
```

**每日更新：**

```bash
python main.py rotation update --name mom5_3d   # 指定組合
python main.py rotation update --all            # 所有 active 組合
```

**查看狀態 / 歷史：**

```bash
python main.py rotation status --name mom5_3d   # 持倉明細 + 未實現損益
python main.py rotation status --all            # 所有組合概覽
python main.py rotation history --name mom5_3d --limit 30  # 已平倉交易
python main.py rotation list                    # 列出所有組合
```

**歷史回測：**

```bash
# 從已建立組合讀取參數
python main.py rotation backtest --name mom5_3d --start 2025-01-01 --end 2025-12-31

# Ad-hoc 回測（不需先建立組合）
python main.py rotation backtest --mode momentum --max-positions 5 --holding-days 3 --capital 1000000 --start 2025-01-01 --end 2025-12-31

# 匯出每日持倉快照 CSV（含日期/股票/股數/進場價/現價/市值/未實現損益/權重）
python main.py rotation backtest --name mom5_3d --start 2025-01-01 --end 2025-12-31 --export-positions positions.csv
```

回測結果（績效摘要 + 逐筆交易含 entry_rank/entry_score）會自動寫入 DB（`rotation_backtest_summary` + `rotation_backtest_trade`），供後續比較與分析。

**管理：**

```bash
python main.py rotation pause --name mom5_3d    # 暫停每日更新
python main.py rotation resume --name mom5_3d   # 恢復
python main.py rotation delete --name mom5_3d   # 刪除組合及所有持倉
```

**參數說明（create）：**

| 參數 | 說明 |
|------|------|
| `--name` | 組合名稱（唯一，如 mom5_3d） |
| `--mode` | discover 模式：momentum / swing / value / dividend / growth / all |
| `--max-positions` | 最大持股數 N（等權配置，每支 1/N 資金） |
| `--holding-days` | 固定持有天數（到期自動處理） |
| `--capital` | 初始資金（TWD） |
| `--no-renewal` | 停用續持（預設啟用：到期時仍在 Top-N 則免賣續持） |

**Rotation 邏輯：**

1. 每日讀取 discover 排名（單模式按 rank，all 模式按 avg_score）
2. 未到期持倉：不動（不因排名下降提前賣出）
3. 到期持倉：若仍在 Top-N 且啟用續持 → 延長持有期；否則賣出
4. 止損：今日收盤 ≤ DiscoveryRecord.stop_loss → 立即賣出（不受持有期限制）
5. 空位填補：從今日排名由高到低選入（排除已持有 + 今日剛賣出的股票）
6. 已整合 morning-routine Step 12，每日自動更新

**Phase B 組合層級風控：**

| 功能 | 說明 | 參數 |
|------|------|------|
| **產業集中度限制** | 同產業持股不超過上限比例，超限時跳過排名較後的候選 | `sector_map` + `max_sector_pct`（預設 30%） |
| **Drawdown Guard** | 回撤 >10% 新開倉減半，>15% 暫停新開倉，回撤恢復自動解除 | `drawdown_pct` + 閾值 |
| **持倉相關性監控** | 60 日 rolling correlation matrix，偵測高相關配對（>0.7） | `compute_correlation_matrix()` + `find_high_correlation_pairs()` |
| **波動率反比權重** | 波動率大的股票分配較少資金，波動率小的分配較多 | `compute_vol_inverse_weights()` |
| **組合回撤計算** | 從淨值序列計算當前回撤百分比（0~100%） | `compute_portfolio_drawdown()` |

---

### morning-routine — 每日早晨例行流程

一鍵執行十七個步驟（Step 0~15+8b），適合搭配 Windows 工作排程器在每日收盤後自動執行。涵蓋完整資料同步 → 選股掃描 → 監控警報流程。

```bash
# 完整流程 + Discord 摘要推播
python main.py morning-routine --notify

# 跳過所有資料同步 Step 1~8b（資料已是最新時使用，加快執行）
python main.py morning-routine --skip-sync --notify

# 預覽各步驟與 Discord 摘要內容（不實際執行）
python main.py morning-routine --dry-run

# discover 顯示 Top 30（預設 20）
python main.py morning-routine --top 30 --notify
```

**十七個執行步驟：**

| 步驟 | 動作 | 說明 |
|------|------|------|
| Step 0 | VIX 同步 + Macro Stress Check | 同步台灣 VIX + 美國 VIX → 偵測 TAIEX crisis 訊號（7 訊號：5日跌>5%/連跌/波動率/爆量長黑/台灣VIX飆升/單日急跌/美國VIX飆升），≥2 觸發時顯示 CRISIS 警示 banner + Discord 預警 |
| Step 1 | `sync-info` | 同步全市場股票基本資料（產業分類 + 上市/上櫃別，DB 已有則跳過） |
| Step 2 | `sync`（OHLCV） | 同步 watchlist + TAIEX 日K線資料（FinMind 逐股） |
| Step 3 | `compute` | 計算 watchlist 技術指標 |
| Step 4 | `sync-mops` | 同步 MOPS 重大訊息公告 |
| Step 5 | `sync-revenue --months 1` | 同步全市場月營收（最近 1 個月） |
| Step 6 | `sync-features --days 90` | 計算全市場 DailyFeature（Feature Store，供 Universe Filtering 使用） |
| Step 7 | `sync-sbl --days 3` | 同步全市場借券賣出資料（TWSE TWT96U） |
| Step 8 | `sync-broker`（watchlist + discover） | 同步 watchlist 分點資料（5日）+ 補抓 discover 推薦的非 watchlist 股票 |
| Step 8b | `sync_market_data`（TWSE/TPEX 全市場） | 同步全市場日K線+法人+融資融券（6 次 API），確保 rotation 持倉等非 watchlist 股票有最新價格 |
| Step 9 | `discover all --skip-sync --top N` | 五模式全市場掃描（不重複同步市場資料） |
| Step 10 | `alert-check --days 3` | MOPS 近3日重大事件警報 |
| Step 11 | `watch update-status` | 批次更新持倉止損/止利/過期狀態 |
| Step 12 | `rotation update --all` | 更新所有 active 輪動組合（讀取 discover 排名，執行換股） |
| Step 13 | `revenue-scan --min-yoy 10 --top 5` | 高成長個股掃描 |
| Step 14 | `anomaly-scan` | 籌碼異動掃描（量能/外資/借券/主力/隔日沖） |
| Step 15 | 策略衰減監控 | 比較五模式近 30 天 vs 歷史勝率/均報酬，衰減時顯示警告（勝率<40% 或均報酬<0） |

**參數說明：**

| 參數 | 說明 |
|------|------|
| `--dry-run` | 只顯示步驟與 Discord 摘要預覽，不實際執行任何操作 |
| `--skip-sync` | 跳過 Step 1–8b（所有資料同步），適合資料已新鮮時加速執行 |
| `--top N` | discover all 的 Top N（預設 20） |
| `--notify` | 執行完畢後推播 Discord 摘要（多模式選股 + 重大事件 + 持倉狀態） |

**Discord 摘要格式：**
- 📊 多模式選股（出現 2+ 模式的股票，含各模式排名）
- 📣 重大事件（近3日非一般性公告）
- 👁 持倉監控（各狀態數量統計）
- 🚨 籌碼異動警報（量能暴增/外資大買超/借券激增/主力集中，各列前3支）

---

### 4.27 同步股票基本資料（`sync-info`）

同步全市場股票基本資料（StockInfo 表）到 DB，包含股票名稱、產業分類、上市/上櫃別。

```bash
# 同步股票基本資料（DB 若已有資料則自動跳過）
python main.py sync-info

# 強制重新同步（覆蓋 DB 現有資料）
python main.py sync-info --force
```

**參數說明：**

| 參數 | 預設 | 說明 |
|------|------|------|
| `--force` | False | 強制重新抓取，即使 DB 已有資料 |

**使用時機：**
- 初次部署後，執行 `sync-info --force` 一次性建立完整基礎資料
- 定期更新（每月一次），確保新上市/更名股票資訊正確
- `industry` / `discover` 命令需仰賴 StockInfo 的產業分類資料

> **注意**：`sync` 命令在同步主流程前也會自動調用 `sync_stock_info(force_refresh=False)`，若 DB 已有資料則不重複拉取。`sync-info` 提供獨立的控制入口，適合需要手動強制更新時使用。

---

### 4.28 計算 Feature Store（`sync-features`）

計算全市場每日特徵（`DailyFeature` 表），供 `UniverseFilter` 三層漏斗的 Stage 2（流動性）與 Stage 3（趨勢動能）快取使用，避免每次 `discover` 時重複計算滾動均值。

```bash
# 計算全市場 DailyFeature（預設回溯 90 天）
python main.py sync-features

# 自訂回溯天數
python main.py sync-features --days 60
```

**計算欄位：**

| 欄位 | 說明 |
|------|------|
| `ma20` / `ma60` | 20 / 60 日收盤均線（趨勢判斷） |
| `volume_ma20` | 20 日均量（量比計算基準） |
| `turnover_ma5` | 5 日均成交金額（流動性過濾基準） |
| `momentum_20d` | 20 日報酬率（%） |
| `volatility_20d` | 20 日年化波動率（%） |

**使用時機：**
- 初次部署後，執行 `sync-info --force && sync-features` 建立完整基礎資料
- 已整合至 `morning-routine` Step 6（每日自動執行），也可獨立夜間排程確保特徵最新
- DailyFeature 表為空時，`UniverseFilter` 會自動從 `DailyPrice` fallback 計算（冷啟動相容）

> **冷啟動順序**：`sync-info --force` → `sync-features` → `discover`

### 4.29 同步 VIX 波動率指數（`sync-vix`）

同步兩個 VIX 來源至 `DailyPrice` 表，供 `detect_crisis_signals()` 使用：

1. **台灣 VIX**（`stock_id="TW_VIX"`）— FinMind `TaiwanOptionMarketVIX`（目前 FinMind 已移除此 dataset，graceful degradation 回傳 0 筆）
2. **美國 VIX**（`stock_id="US_VIX"`）— yfinance `^VIX`（CBOE VIX，穩定可用）

```bash
python main.py sync-vix
```

- 台灣 VIX > 30 或單日漲幅 > 25% 觸發 `vix_spike` crisis 訊號（Signal 5）
- 美國 VIX > 30 或單日漲幅 > 25% 觸發 `us_vix_spike` crisis 訊號（Signal 7）
- `morning-routine` Step 0 自動同步兩者（失敗不中斷流程）
- 無 VIX 資料時自動跳過對應訊號（graceful degradation）

---

## 5. 資料庫 Schema

資料庫使用 SQLite，檔案位於 `data/stock.db`。共 **23 張核心表**：

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
| stop_price | Float | 進場時計算並固定的止損價（ATR-based 或百分比，無設定時為 NULL） |
| target_price | Float | 進場時計算並固定的目標價（ATR-based 或百分比，無設定時為 NULL） |

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
| event_type | String | 事件類型（governance_change/buyback/earnings_call/investor_day/filing/revenue/general） |

唯一鍵：`(stock_id, date, seq)`

> **事件類型（event_type）枚舉**：`governance_change`（董監改選/市場派）、`buyback`（庫藏股決議）、`earnings_call`（法說會）、`investor_day`（投資人說明會）、`filing`（財報申報）、`revenue`（月營收）、`general`（其他一般公告）

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

### stock_info（股票基本資料）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| stock_name | String | 股票名稱 |
| industry_category | String | 產業分類 |
| listing_type | String | 上市/上櫃 |
| security_type | String | 證券類型（stock/etf/warrant/preferred/None） |
| updated_at | DateTime | 更新時間 |

唯一鍵：`(stock_id)`

### discovery_record（Discover 推薦歷史）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| stock_name | String | 股票名稱 |
| scan_date | Date | 掃描日期 |
| mode | String | 掃描模式（momentum/swing/value/dividend/growth） |
| rank | Integer | 本次掃描排名 |
| composite_score | Float | 綜合評分 |
| entry_price | Float | 建議進場價（ATR-based） |
| stop_loss | Float | 建議止損價（entry - 1.5×ATR14） |
| take_profit | Float | 建議目標價（entry + 3.0×ATR14） |
| entry_trigger | String | 進場觸發說明（站上均線等） |
| valid_until | Date | 建議有效日期（scan_date + 5 工作日） |
| chip_tier | String | 籌碼評分 Tier（"8F"~"2F" 或 "N/A"） |
| concept_bonus | Float | 概念股加成分數（±5%，sector+concept ≤ ±8% cap） |
| daytrade_penalty | Float | 隔日沖扣分（0~1） |
| daytrade_tags | String | 隔日沖標記（行為/黑名單/即時風險分點名稱） |

唯一鍵：`(stock_id, scan_date, mode)`

### watch_entry（持倉監控）

| 欄位 | 型別 | 說明 |
|------|------|------|
| id | Integer | 自增主鍵 |
| stock_id | String | 股票代號 |
| stock_name | String | 股票名稱 |
| entry_price | Float | 進場價格 |
| stop_loss | Float | 止損價格（移動止損時隨最高價自動上移） |
| take_profit | Float | 目標價格 |
| qty | Integer | 持倉股數（可選） |
| status | String | 狀態：active/stopped_loss/taken_profit/expired/closed |
| added_date | Date | 加入日期 |
| valid_until | Date | 有效截止日 |
| close_price | Float | 平倉價格（手動平倉時記錄） |
| close_date | Date | 平倉日期 |
| source_mode | String | 來源模式（discover 模式或 manual） |
| notes | String | 備註 |
| created_at | DateTime | 建立時間 |
| trailing_stop_enabled | Boolean | 是否啟用移動止損（`--trailing` 旗標設定） |
| trailing_atr_multiplier | Float | 移動止損 ATR 倍數（預設 1.5） |
| highest_price_since_entry | Float | 進場後歷史最高收盤價（移動止損追蹤用） |

### stock_valuation（估值資料 — TWSE/TPEX 官方）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 資料日期 |
| pe_ratio | Float | 本益比（P/E） |
| pb_ratio | Float | 股價淨值比（P/B） |
| dividend_yield | Float | 殖利率 (%) |

唯一鍵：`(stock_id, date)`

### holding_distribution（大戶持股分級 — TDCC 集保戶股權分散表）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 週資料日期（TDCC 每週發布） |
| level | String | 持股級別（如 "400,000 ~ 800,000 股" 等） |
| holder_count | Integer | 持股人數 |
| share_count | BigInteger | 持股股數 |
| share_pct | Float | 占總股本比例 (%) |

唯一鍵：`(stock_id, date, level)`

### securities_lending（借券賣出 — TWSE TWT96U）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 交易日 |
| sbl_sell | BigInteger | 當日借券賣出股數 |
| sbl_return | BigInteger | 當日借券還券股數 |
| sbl_balance | BigInteger | 借券餘額股數 |
| sbl_change | BigInteger | 借券餘額變化（正=增加借券壓力） |

唯一鍵：`(stock_id, date)`

### broker_trade（分點進出 — DJ 分點端點）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 交易日（DJ 端點為期間 end 日期） |
| broker_id | String | 分點代號（BHID） |
| broker_name | String | 分點名稱 |
| buy | BigInteger | 買進股數 |
| sell | BigInteger | 賣出股數 |
| net | BigInteger | 淨買超股數（buy - sell） |
| buy_price | Float | 均買價（DJ 端點目前為 NULL，以 DailyPrice 收盤代理） |
| sell_price | Float | 均賣價（DJ 端點目前為 NULL，以 DailyPrice 收盤代理） |

唯一鍵：`(stock_id, date, broker_id)`

### watchlist（DB-based 觀察清單）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| stock_name | String | 股票名稱（選填） |
| note | String | 備註（選填） |
| added_at | DateTime | 加入時間 |

唯一鍵：`(stock_id)`

### daily_feature（Feature Store — Universe Filtering 快取）

| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | String | 股票代號 |
| date | Date | 交易日 |
| ma20 | Float | 20 日收盤均線 |
| ma60 | Float | 60 日收盤均線 |
| volume_ma20 | Float | 20 日均量 |
| turnover_ma5 | Float | 5 日均成交金額（流動性指標） |
| momentum_20d | Float | 20 日報酬率 (%) |
| volatility_20d | Float | 20 日年化波動率 (%) |

唯一鍵：`(stock_id, date)`

### concept_group（概念股群組定義）

| 欄位 | 型別 | 說明 |
|------|------|------|
| id | Integer | 自增主鍵 |
| name | String | 概念名稱（如 CoWoS封裝、散熱模組） |
| description | String | 概念描述 |
| source | String | 來源（yaml / manual） |
| created_at | DateTime | 建立時間 |

唯一鍵：`(name)`

### concept_membership（概念股成員）

| 欄位 | 型別 | 說明 |
|------|------|------|
| id | Integer | 自增主鍵 |
| concept_id | Integer | 關聯的概念群組 ID (FK) |
| stock_id | String | 股票代號 |
| source | String | 來源（yaml / manual / mops / correlation） |
| added_at | DateTime | 加入時間 |

唯一鍵：`(concept_id, stock_id)`

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

推薦使用 DB watchlist 管理（`python main.py watchlist add 2330 --name 台積電`），或編輯 `config/settings.yaml` 的 `fetcher.watchlist`。DB 非空時優先使用 DB 清單。首次可執行 `python main.py watchlist import` 從 YAML 批次匯入。

### Q: 資料多久更新一次？

FinMind 資料約在每個交易日收盤後更新。建議每天盤後執行一次 `python main.py sync`，系統會自動增量抓取新資料。

### Q: API 有速率限制嗎？

FinMind 免費版有請求頻率限制。系統已內建每次請求後等待 0.5 秒的節流機制。

### Q: 資料庫檔案在哪裡？

`data/stock.db`，是一個 SQLite 檔案，可用任何 SQLite 工具（如 DB Browser for SQLite）直接瀏覽。
