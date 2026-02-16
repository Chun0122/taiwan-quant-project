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
│   ├── features/            # 技術指標（P1 待開發）
│   ├── strategy/            # 交易策略（P2 待開發）
│   ├── backtest/            # 回測引擎（P2 待開發）
│   └── visualization/       # 視覺化（P3 待開發）
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
```

每檔股票會同步三種資料：
| 資料類型 | FinMind Dataset | DB 資料表 |
|----------|----------------|-----------|
| 日K線（OHLCV） | TaiwanStockPrice | `daily_price` |
| 三大法人買賣超 | TaiwanStockInstitutionalInvestorsBuySell | `institutional_investor` |
| 融資融券 | TaiwanStockMarginPurchaseShortSale | `margin_trading` |

### 4.2 查看資料庫概況 (`status`)

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

資料庫使用 SQLite，檔案位於 `data/stock.db`。三張核心表：

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
