# CLI 指令完整參考

入口：`python main.py <子命令>`（38 子命令，dispatch table 在 `main.py`）

---

## 安裝

```bash
# requirements.txt 為凍結鎖定檔；--no-deps 必要（FinMind 上游 pin 過緊，詳見 USAGE.md §1）
pip install --no-deps -r requirements.txt
```

---

## 資料同步

```bash
# OHLCV（watchlist + TAIEX）
python main.py sync

# 技術指標計算
python main.py compute

# 全市場基本資料
python main.py sync-info
python main.py sync-info --force

# DailyFeature（預設 90 天）
python main.py sync-features
python main.py sync-features --days 60

# US VIX（yfinance ^VIX）
python main.py sync-vix

# MOPS 重大訊息（預設 7 天）
python main.py sync-mops
python main.py sync-mops --days 30

# 全市場月營收
python main.py sync-revenue
python main.py sync-revenue --months 3

# Watchlist 財報（預設 4 季）
python main.py sync-financial
python main.py sync-financial --stocks 2330 --quarters 8

# TDCC 大戶持股（全市場）
python main.py sync-holding

# 借券賣出（預設 3 天）
python main.py sync-sbl
python main.py sync-sbl --days 5

# 分點資料（watchlist，預設 5 日）
python main.py sync-broker
python main.py sync-broker --stocks 2330 2317 --days 10
python main.py sync-broker --from-discover
python main.py sync-broker --watchlist-bootstrap              # 首次部署（120 天）
python main.py sync-broker --watchlist-bootstrap --days 60
python main.py sync-broker --from-file stocks.txt --watchlist-bootstrap
```

---

## 選股掃描

```bash
# 五模式（預設 momentum）
python main.py discover momentum --top 20
python main.py discover swing --top 20
python main.py discover value --top 20
python main.py discover dividend --top 20
python main.py discover growth --top 20
python main.py discover --top 20                  # 預設 momentum
python main.py discover --skip-sync --top 10

# 跨模式比較
python main.py discover --compare
python main.py discover all --skip-sync --top 20
python main.py discover all --skip-sync --min-appearances 2
python main.py discover all --skip-sync --export compare.csv

# 進階選項
python main.py discover momentum --weekly-confirm     # 週線多時框確認
python main.py discover momentum --use-ic-adjustment  # Factor IC 動態權重調整

# 推薦績效回測（2026-07-05 預設翻轉：含成本 + T+1 開盤進場）
python main.py discover-backtest --mode momentum                     # 新預設：含成本 + T+1 開盤
python main.py discover-backtest --mode momentum --naive             # 舊假設（無成本 + T 日收盤，僅供對照）
python main.py discover-backtest --mode momentum --no-include-costs  # 單獨關閉成本
python main.py discover-backtest --mode momentum --no-entry-next-open  # 單獨關閉 T+1 進場

# OOS Hold-Out 紀律（P1 任務 7：預設保留最近 90 天為 holdout，回測跨界印警告）
python main.py discover-backtest --mode momentum --start 2026-01-01 --end 2026-02-15  # ✅ 純 in-sample
python main.py discover-backtest --mode momentum --start 2026-04-01 --end 2026-05-15  # 🟢 純 forward (OOS)
python main.py discover-backtest --mode momentum --start 2026-01-01 --end 2026-05-15  # ⚠️ partial 跨界
python main.py discover-backtest --mode momentum --holdout-start 2026-03-01           # 自訂 holdout 起點
python main.py discover-backtest --mode momentum --start 2026-04-01 --end 2026-05-15 --ignore-holdout  # 明示放行

# 跨模式 score 相關性研究（P2 任務 13：5 mode 間 score 是否冗餘/互補）
python main.py cross-mode-corr                          # 預設 lookback 60 天
python main.py cross-mode-corr --lookback-days 90 --min-pairs 3
python main.py cross-mode-corr --export cross_mode.csv  # 匯出相關性矩陣
# 解讀：corr ≥0.7 = 模式冗餘（all quota 浪費）；corr ≤-0.3 = 互補對沖（分散佳）

# 因子診斷
python main.py factor-diagnostics --mode momentum  # IC + 相關性矩陣 + Rolling IC + Per-Regime IC

# 因子消融測試
python main.py ablation-test --mode momentum
python main.py ablation-test --mode momentum --with-performance    # 含歷史績效消融
python main.py ablation-test --mode momentum --skip-sync --export ablation.csv
```

---

## 回測

```bash
python main.py backtest --stock 2330 --strategy sma_cross
python main.py backtest --stock 2330 --strategy sma_cross --attribution
python main.py backtest --stock 2330 --strategy sma_cross --export-trades trades.csv

# A4 交易現實化旗標（單股回測預設開啟；報告 header 印成本假設）
python main.py backtest --stock 2330 --strategy sma_cross --no-min-commission      # 關閉最低手續費（整張 20 / 零股 1 元 + 零股滑價 premium）
python main.py backtest --stock 2330 --strategy sma_cross --no-participation-impact  # 關閉下單量滑價衝擊項
```

---

## 輪動組合

```bash
# 建立
python main.py rotation create --name mom5_3d --mode momentum --max-positions 5 --holding-days 3 --capital 1000000
python main.py rotation create --name all10_5d --mode all --max-positions 10 --holding-days 5 --capital 2000000 --no-renewal
# 合成模式：mom_growth = 動量+成長雙引擎（momentum+growth，per_mode_max=3）
python main.py rotation create --name mg5_20d --mode mom_growth --max-positions 5 --holding-days 20 --capital 1000000

# 更新（A2 T+1 兩段式，2026-07-06）：先以今日開盤成交昨日 pending orders，
# 再以今日收盤決策、寫明日 pending（RotationPendingOrder）。續持與熔斷即時。
# 同日冪等：當日已 decide 過（pending/ActionLog/Snapshot 任一存在）自動跳過。
python main.py rotation update --name mom5_3d
python main.py rotation update --all
python main.py rotation update --name mom5_3d --force   # 繞過 decide 冪等保護（fill 天然冪等）

# Pre-Trade 預覽（dry_run，不寫 DB）：顯示待成交佇列 + 明日預定買賣
python main.py rotation preview --name swing5_3d                # 預覽單一組合
python main.py rotation preview --all                            # 預覽所有 active 組合
python main.py rotation preview --name swing5_3d --date 2026-05-20  # 指定決策日

# 查詢
python main.py rotation status --name mom5_3d
python main.py rotation status --all
python main.py rotation history --name mom5_3d --limit 30
python main.py rotation list

# 回測
python main.py rotation backtest --name mom5_3d --start 2025-01-01 --end 2025-12-31
python main.py rotation backtest --mode momentum --max-positions 5 --holding-days 3 --start 2025-01-01 --end 2025-12-31
python main.py rotation backtest --name mom5_3d --start 2025-01-01 --end 2025-12-31 --export-positions positions.csv

# 實盤成本歸因（手續費/交易稅/滑價 + 累計周轉 + bps per turnover，對應 5/29 audit alpha 拖累驗證）
python main.py rotation cost-attribution --name all10_5d --start 2026-04-01 --end 2026-05-15
python main.py rotation cost-attribution --name mom5_10d --include-open    # 納入未平倉（僅計買端）
python main.py rotation cost-attribution --name swing5_3d --export swing_cost.csv
# 注意：buy/sell_slippage 為 NULL 的歷史 position 會以 SLIPPAGE_RATE 預設值估算，
# 輸出會顯示「估算滑價(buy/sell): N/M」透明化計數。

# 管理
python main.py rotation pause --name mom5_3d
python main.py rotation resume --name mom5_3d
python main.py rotation delete --name mom5_3d
```

### 期間對比審計（rotation-audit）

可重複的修復前後 / 期間對比審計，兌現 audit 報告的「定期重審」承諾（取代手工 SQL）。

```bash
# 修復前 A 期 vs 修復後 B 期
python main.py rotation-audit --period-a 2026-04-29:2026-05-08 --period-b 2026-05-09:2026-05-29
# 只看單期 + 寫入檔案
python main.py rotation-audit --period-b 2026-06-01:2026-06-15 --out logs/audit_20260615/REPORT.md
python main.py rotation-audit --period-b 2026-06-01:2026-06-15 --jaccard-mode swing --top 10
```

輸出三大區塊：
- **Closed Trade A/B 對比**（N / win% / avg ret% / total pnl / stop-loss%；< 10 筆自動標樣本不足）
- **Benchmark Alpha 分解**（snapshot-based；含 `port Δ ≈ alpha 增量 + bm 增量` 恆等式 + 0050 raw-price cross-check，自動標 > 2pp 的資料異常）
- **訊號穩定性**（discover top-N 相鄰日 Jaccard）

---

## 持倉監控

```bash
# 單股建議
python main.py suggest 2330
python main.py suggest 2330 --notify

# Watch 管理
python main.py watch add 2330
python main.py watch add 2330 --price 580 --stop 555 --target 635 --qty 1000
python main.py watch add 2330 --from-discover momentum
python main.py watch add 2330 --trailing
python main.py watch add 2330 --trailing --trailing-multiplier 2.0
python main.py watch list
python main.py watch list --status all
python main.py watch close 1 --price 595
python main.py watch update-status
```

---

## 警報與掃描

```bash
# MOPS 事件警報
python main.py alert-check
python main.py alert-check --days 14 --types earnings_call filing
python main.py alert-check --stocks 2330 2317 --notify

# 營收高成長掃描
python main.py revenue-scan
python main.py revenue-scan --min-yoy 20 --min-margin-improve 1.0
python main.py revenue-scan --top 10 --notify

# 籌碼異動掃描
python main.py anomaly-scan
python main.py anomaly-scan --stocks 2330 2317
python main.py anomaly-scan --vol-mult 3.0 --inst-threshold 5000000 --dt-threshold 0.3
python main.py anomaly-scan --notify
```

---

## 每日例行

```bash
python main.py morning-routine --notify              # 完整流程（Step 0~16+8b）
python main.py morning-routine --skip-sync --notify  # 跳過 Step 1~8b
python main.py morning-routine --dry-run
```

Step 16 會自動把當日狀態寫成 `daily_dashboard.json`，供 iOS 監控 App 與其他下游消費者使用。

---

## Dashboard JSON 匯出

每日狀態統一輸出檔，schema 與欄位對照表見 [`docs/dashboard_schema.md`](dashboard_schema.md)。

```bash
python main.py export-dashboard                                   # 今日，寫到 iCloud Drive 預設路徑
python main.py export-dashboard --date 2026-04-30 --top 30        # 補產歷史日期
python main.py export-dashboard --out /tmp/dashboard              # 自訂輸出目錄
python main.py export-dashboard --regenerate-ai-summary           # 重呼 Claude API 產 AI 摘要（會燒 token）
```

預設輸出兩個檔：`<out_dir>/<YYYY-MM-DD>.json` + `<out_dir>/latest.json`（後者為 App 固定入口）。

---

## Watchlist / 概念股

```bash
# Watchlist 管理
python main.py watchlist list
python main.py watchlist add 2330
python main.py watchlist add 2330 --name 台積電 --note 核心持倉
python main.py watchlist remove 2330
python main.py watchlist import

# 概念股同步
python main.py sync-concepts
python main.py sync-concepts --purge
python main.py sync-concepts --from-mops --days 30

# 概念股管理
python main.py concepts list
python main.py concepts list CoWoS封裝
python main.py concepts add CoWoS封裝 2330
python main.py concepts remove CoWoS封裝 2330
python main.py concept-expand CoWoS封裝 --threshold 0.7
python main.py concept-expand CoWoS封裝 --threshold 0.7 --auto
```

---

## Factor Library（P1 任務 6：因子 SSOT）

```bash
# 列出全部因子
python main.py factor-list

# 限定維度（technical / chip / fundamental / news / valuation / dividend / regime）
python main.py factor-list --category chip

# 限定類型（dimension / sub_factor / predicate / indicator）
python main.py factor-list --type dimension
python main.py factor-list --type predicate   # screener/factors.py watchlist filter
python main.py factor-list --type indicator   # features/indicators.py EAV 持久化

# 限定 discover 模式
python main.py factor-list --mode momentum

# 顯示單一因子完整 spec
python main.py factor-list --name chip_score

# Introspection 守門：驗證所有 source_module/function 可解析（CI 用）
python main.py factor-list --check-resolve   # 失敗 exit code 1
```

註冊位置：`src/factors/registry.py:FACTOR_REGISTRY`。新增因子請同步註冊 metadata（name / category / source / expected_sign / used_in_modes / IC notes）。

---

## 資料品質 / 匯出匯入

```bash
# 資料驗證
python main.py validate
python main.py validate --stocks 2330 2317
python main.py validate --export issues.csv

# Experiment Registry（P2 任務 10：A/B 試驗歷史軌跡，與 baseline_metrics.json 互補）
python main.py experiment record --description "test new chip weight 0.6"   # 凍結 settings + metrics
python main.py experiment list                                                # 最近 20 筆
python main.py experiment list --limit 50
python main.py experiment show exp_20260518_a3f8c1                            # 完整 settings + metrics
python main.py experiment compare exp_A exp_B                                 # 逐 metric 差異 + settings_hash 變動標記
# 注意：settings_snapshot 只記 quant + fetcher 區塊，不含 API token / webhook URL

# Baseline Regression 守門（5/29 audit 策略劣化偵測，morning-routine Step 17 自動執行）
python main.py update-baseline --confirm                  # 凍結當前 active portfolio 指標為新 baseline
python main.py validate-baseline                          # 對比當前 vs baseline；regression 退出碼 1
python main.py validate-baseline --tolerance 0.5          # 嚴格半量閾值
python main.py validate-baseline --tolerance 2.0 --quiet  # 寬鬆 + 只用 exit code（CI 用）
# baseline_metrics.json 結構：每個 portfolio 凍結 sharpe / max_drawdown_pct / win_rate_pct / alpha_cum_pct
# 預設 deltas：sharpe -0.20 / mdd +2pp / win_rate -5pp / alpha -3pp 觸發 regression

# 匯出
python main.py export --list
python main.py export daily_price -o data/export/daily_price.csv
python main.py export daily_price --stocks 2330 --start 2024-01-01
python main.py export daily_price --format parquet -o data/export/dp.parquet

# 匯入
python main.py import-data daily_price data/export/daily_price.csv
python main.py import-data daily_price data.csv --dry-run
```

---

## PIT 研究環境（B1）

### 歷史回補（`backfill-history`）

長時間作業，**可隨時 Ctrl-C 中止，重跑自動從缺口續行**——進度以 DB 現況判定，不另存進度檔。

```bash
# 價量回補（TWSE/TPEX 每日全市場端點；含當時在市、如今已下市的標的）
python main.py backfill-history --start 2024-01-01
python main.py backfill-history --start 2024-01-01 --end 2024-12-31
python main.py backfill-history --start 2024-01-01 --dry-run          # 只估算待補日數與時間
python main.py backfill-history --start 2024-01-01 --datasets price   # 只補日K
python main.py backfill-history --start 2024-01-01 --with-features    # 補完接著算 DailyFeature

# DailyFeature 歷史化（B1②，純 CPU 不打 API；需 DailyPrice 已就緒）
python main.py backfill-history --start 2024-01-01 --features-only

# 估值回補（§6.5 #20；上市走 TWSE 每日端點、上櫃走 FinMind 逐股）
python main.py backfill-history --valuation-only --start 2024-01-01
python main.py backfill-history --valuation-only --start 2024-01-01 --valuation-markets twse
python main.py backfill-history --valuation-only --start 2024-01-01 --dry-run
```

**估值為何要分兩條路**：TPEX 的估值端點（`peratio_book/pera_result.php`）**已下架**，所有日期含當日皆 302 導向 `/errors`；新版 openapi 只回當日、無日期參數。故上櫃歷史無官方來源，改走 FinMind `TaiwanStockPER` 逐股（支援日期區間，一檔一次呼叫涵蓋全期間）。上市則走 TWSE `BWIBBU_d`，健在且有完整歷史。

**續跑判定**（不看「有無資料」——那會靜默跳過整段歷史）：

| 資料 | 判定 |
|------|------|
| 價量 | 當日**普通股（4 碼）**檔數 ≥ `BACKFILL_MIN_COMMON_STOCKS`（1500） |
| 估值/上市 | 當日估值檔數 ≥ `BACKFILL_MIN_VALUATION_STOCKS`（800） |
| 估值/上櫃 | 該檔估值日數 ≥ 其價量日數 × `VALUATION_COVERAGE_RATIO`（0.8） |

### PIT 歷史重放（`pit-replay`）

在歷史日重跑 scanner 並評估前瞻報酬。**唯讀**——不寫 `DiscoveryRecord` / `CandidateFactorLog` / `universe_stat_log`，regime 亦不推進狀態機。

```bash
python main.py pit-replay --as-of 2025-04-08 --mode momentum
python main.py pit-replay --as-of 2025-04-08 --mode value --top-n 20
python main.py pit-replay --start 2024-01-01 --end 2024-12-31 --mode momentum --every-n-days 20
```

單次重放約 90 秒，**範圍重放務必抽樣**（`--every-n-days`）。前瞻報酬是唯一允許看 `as_of` 之後資料之處（評分而非決策輸入）。

⚠ 重放結果的有效性受資料覆蓋限制：估值/營收未回補的期間，value/dividend/growth 的粗篩會 **fail-open**（閘門消失而非收緊），模式靜默退化為流動性篩選。解讀前先確認對應期間的基本面資料存在。

---

## 排程

```bash
python main.py schedule                   # auto 偵測平台（Windows→Task Scheduler / macOS→LaunchAgent）
python main.py schedule --mode simple     # 前景阻塞式排程（跨平台）
python main.py schedule --mode windows    # 產生 .bat + Task Scheduler XML
python main.py schedule --mode macos      # 產生 .sh + LaunchAgent .plist
```

---

## 儀表板

```bash
python main.py dashboard                  # Streamlit localhost:8501
```
