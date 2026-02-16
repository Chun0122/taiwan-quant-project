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
│   │   ├── schema.py        # ORM 資料表定義
│   │   ├── fetcher.py       # FinMind API 資料抓取
│   │   └── pipeline.py      # ETL Pipeline（抓取→清洗→寫入）
│   ├── features/
│   │   └── indicators.py   # 技術指標計算引擎（SMA/RSI/MACD/BB）
│   ├── strategy/
│   │   ├── base.py          # 策略抽象基類
│   │   ├── sma_cross.py     # SMA 均線交叉策略
│   │   ├── rsi_threshold.py # RSI 超買超賣策略
│   │   ├── bb_breakout.py   # 布林通道突破策略
│   │   ├── macd_cross.py    # MACD 交叉策略
│   │   └── buy_hold.py      # 買入持有基準策略
│   ├── backtest/
│   │   └── engine.py        # 回測引擎 + 績效計算
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
│           └── backtest_review.py   # 回測結果頁
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

回測結果會自動顯示同期 Buy & Hold 基準報酬率與超額報酬。

交易成本設定（符合台股實際費率）：
- 手續費: 0.1425%（買賣各收一次）
- 交易稅: 0.3%（僅賣出時收取）
- 滑價: 0.05%
- 初始資金: 1,000,000 元

### 4.4 啟動視覺化儀表板 (`dashboard`)

```bash
python main.py dashboard
```

瀏覽器會自動開啟 `http://localhost:8501`，包含兩個頁面：

- **個股分析**: K線圖 + SMA/BB/RSI/MACD 疊加 + 成交量 + 法人買賣超 + 融資融券
- **回測結果**: 績效摘要卡片 + 權益曲線/回撤圖 + 交易明細 + 回測比較表

### 4.5 參數優化 (`optimize`)

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

### 4.6 自動排程 (`schedule`)

設定每日自動同步資料與計算指標。

```bash
# 產生 Windows Task Scheduler 腳本（建議）
python main.py schedule --mode windows

# 前景執行排程器（測試用，每日 23:00 自動同步）
python main.py schedule --mode simple
```

Windows 模式會在 `scripts/` 目錄產生 `daily_sync.bat` 和 `task_schedule.xml`，
按照輸出的說明匯入 Windows 工作排程器即可。

### 4.7 查看資料庫概況 (`status`)

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
