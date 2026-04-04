"""台股量化投資系統 — 主程式入口。

Usage:
    # 同步預設關注清單
    python main.py sync

    # 同步指定股票
    python main.py sync --stocks 2330 2317

    # 指定日期範圍
    python main.py sync --start 2023-01-01 --end 2024-12-31

    # 計算技術指標
    python main.py compute

    # 執行回測
    python main.py backtest --stock 2330 --strategy sma_cross

    # 加停損停利
    python main.py backtest --stock 2330 --strategy sma_cross --stop-loss 5 --take-profit 15

    # 固定比例部位
    python main.py backtest --stock 2330 --strategy rsi_threshold --sizing fixed_fraction --fraction 0.3

    # 投資組合回測
    python main.py backtest --stocks 2330 2317 2454 --strategy sma_cross --stop-loss 5

    # 多因子策略回測
    python main.py backtest --stock 2330 --strategy multi_factor

    # 啟動視覺化儀表板
    python main.py dashboard

    # 參數優化
    python main.py optimize --stock 2330 --strategy sma_cross

    # 設定排程
    python main.py schedule --mode windows

    # 查詢已入庫的資料概況
    python main.py status

    # 多因子選股篩選
    python main.py scan
    python main.py scan --export scan_results.csv
    python main.py scan --notify

    # Discord 通知
    python main.py notify --message "測試訊息"

    # Walk-Forward 驗證（ML 策略防過擬合）
    python main.py walk-forward --stock 2330 --strategy ml_random_forest
    python main.py walk-forward --stock 2330 --strategy ml_xgboost --train-window 504 --test-window 126

    # 每日選股報告
    python main.py report --top 10
    python main.py report --no-ml --notify
    python main.py report --export daily_report.csv

    # 策略回測排名
    python main.py strategy-rank --metric sharpe
    python main.py strategy-rank --strategies sma_cross rsi_threshold --stocks 2330 2317

    # 產業輪動分析
    python main.py industry --refresh --top-sectors 5
    python main.py industry --notify

    # 全市場選股掃描
    python main.py discover
    python main.py discover --top 30 --min-price 50
    python main.py discover --skip-sync --top 10
    python main.py discover --export picks.csv --notify
    python main.py discover all --skip-sync --top 20
    python main.py discover all --skip-sync --min-appearances 2
    python main.py discover all --skip-sync --export compare.csv

    # Discover 推薦績效回測
    python main.py discover-backtest --mode momentum
    python main.py discover-backtest --mode swing --days 5,10,20,60
    python main.py discover-backtest --mode value --top 10
    python main.py discover-backtest --mode momentum --start 2025-06-01 --end 2025-12-31
    python main.py discover-backtest --mode momentum --export result.csv

    # 同步財報資料
    python main.py sync-financial                    # 同步 watchlist 財報（預設最近 4 季）
    python main.py sync-financial --stocks 2330 2317 # 指定股票
    python main.py sync-financial --quarters 8       # 最近 8 季

    # DB 遷移
    python main.py migrate

    # 資料品質檢查
    python main.py validate
    python main.py validate --stocks 2330 2317
    python main.py validate --gap-threshold 3 --streak-threshold 3
    python main.py validate --no-freshness
    python main.py validate --export issues.csv

    # 每日早晨例行流程（一鍵執行）
    python main.py morning-routine --notify         # 完整流程 + Discord 摘要
    python main.py morning-routine --skip-sync --notify  # 跳過借券/分點同步（資料已新鮮時）
    python main.py morning-routine --dry-run        # 預覽步驟與摘要（不實際執行）
    python main.py morning-routine --top 30 --notify     # discover Top 30
"""

from __future__ import annotations

import argparse
import sys

from src.cli.anomaly_cmd import cmd_anomaly_scan, cmd_revenue_scan
from src.cli.backtest_cmd import cmd_backtest, cmd_walk_forward
from src.cli.discover_cmd import cmd_discover, cmd_discover_backtest
from src.cli.helpers import setup_logging
from src.cli.misc_cmd import (
    cmd_dashboard,
    cmd_export,
    cmd_import_data,
    cmd_industry,
    cmd_migrate,
    cmd_notify,
    cmd_optimize,
    cmd_report,
    cmd_scan,
    cmd_schedule,
    cmd_status,
    cmd_strategy_rank,
    cmd_validate,
)
from src.cli.morning_cmd import cmd_morning_routine
from src.cli.rotation_cmd import cmd_rotation
from src.cli.suggest_cmd import cmd_suggest
from src.cli.sync import (
    cmd_alert_check,
    cmd_compute,
    cmd_sync,
    cmd_sync_broker,
    cmd_sync_features,
    cmd_sync_financial,
    cmd_sync_holding,
    cmd_sync_info,
    cmd_sync_mops,
    cmd_sync_revenue,
    cmd_sync_sbl,
    cmd_sync_vix,
)
from src.cli.watch_cmd import cmd_watch
from src.cli.watchlist_cmd import cmd_concept_expand, cmd_concepts, cmd_sync_concepts, cmd_watchlist
from src.constants import (
    DEFAULT_DT_THRESHOLD,
    DEFAULT_HHI_THRESHOLD,
    DEFAULT_INST_THRESHOLD,
    DEFAULT_SBL_SIGMA,
    DEFAULT_VOL_MULT,
)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="台股量化投資系統")
    subparsers = parser.add_subparsers(dest="command")

    # sync 子命令
    sp_sync = subparsers.add_parser("sync", help="同步股票資料")
    sp_sync.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_sync.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_sync.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")
    sp_sync.add_argument("--taiex", action="store_true", help="同步加權指數（現預設啟用）")

    # compute 子命令
    sp_compute = subparsers.add_parser("compute", help="計算技術指標")
    sp_compute.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")

    # backtest 子命令
    sp_bt = subparsers.add_parser("backtest", help="執行回測")
    sp_bt.add_argument("--stock", default=None, help="股票代號（單股回測）")
    sp_bt.add_argument("--stocks", nargs="+", default=None, help="多支股票代號（投資組合回測）")
    sp_bt.add_argument("--strategy", required=True, help="策略名稱 (sma_cross, rsi_threshold, ...)")
    sp_bt.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_bt.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")
    # 風險管理參數
    sp_bt.add_argument("--stop-loss", type=float, default=None, help="停損百分比 (例: 5.0 = -5%%)")
    sp_bt.add_argument("--take-profit", type=float, default=None, help="停利百分比 (例: 15.0 = +15%%)")
    sp_bt.add_argument("--trailing-stop", type=float, default=None, help="移動停損百分比 (例: 8.0)")
    sp_bt.add_argument(
        "--sizing", default="all_in", choices=["all_in", "fixed_fraction", "kelly", "atr"], help="部位大小計算方式"
    )
    sp_bt.add_argument("--fraction", type=float, default=1.0, help="fixed_fraction 比例 (0.0~1.0)")
    sp_bt.add_argument(
        "--allocation",
        default="equal_weight",
        choices=["equal_weight", "custom", "risk_parity", "mean_variance"],
        help="投資組合配置方式 (equal_weight/custom/risk_parity/mean_variance)",
    )
    sp_bt.add_argument(
        "--adjust-dividend", action="store_true", default=False, help="啟用除權息還原（回溯調整價格 + 股利入帳）"
    )
    sp_bt.add_argument("--attribution", action="store_true", default=False, help="回測結束後計算五因子歸因分析")
    sp_bt.add_argument("--export-trades", default=None, help="匯出交易明細 CSV 路徑")
    sp_bt.add_argument("--shap", action="store_true", default=False, help="ML 策略回測時輸出 SHAP 特徵重要性 Top-10")
    sp_bt.add_argument("--optuna", action="store_true", default=False, help="ML 策略啟用 Optuna 超參數調優")
    sp_bt.add_argument(
        "--feature-selection", action="store_true", default=False, help="ML 策略啟用 SHAP 特徵篩選（需搭配 --shap）"
    )

    # dashboard 子命令
    subparsers.add_parser("dashboard", help="啟動視覺化儀表板")

    # optimize 子命令
    sp_opt = subparsers.add_parser("optimize", help="參數優化（Grid Search）")
    sp_opt.add_argument("--stock", required=True, help="股票代號")
    sp_opt.add_argument("--strategy", required=True, help="策略名稱")
    sp_opt.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_opt.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")
    sp_opt.add_argument("--top-n", type=int, default=10, help="顯示前 N 名結果 (預設 10)")
    sp_opt.add_argument("--export", default=None, help="匯出 CSV 路徑")

    # schedule 子命令
    sp_sched = subparsers.add_parser("schedule", help="設定自動排程")
    sp_sched.add_argument(
        "--mode",
        choices=["simple", "windows"],
        default="windows",
        help="排程模式: simple=前景執行, windows=產生 Task Scheduler 腳本",
    )

    # scan 子命令
    sp_scan = subparsers.add_parser("scan", help="多因子選股篩選")
    sp_scan.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_scan.add_argument("--conditions", nargs="+", help="篩選條件（因子名稱）")
    sp_scan.add_argument("--lookback", type=int, default=5, help="回溯天數 (預設 5)")
    sp_scan.add_argument("--export", default=None, help="匯出 CSV 路徑")
    sp_scan.add_argument("--notify", action="store_true", help="將結果發送 Discord 通知")

    # notify 子命令
    sp_notify = subparsers.add_parser("notify", help="發送 Discord Webhook 訊息")
    sp_notify.add_argument("--message", required=True, help="訊息內容")

    # walk-forward 子命令
    sp_wf = subparsers.add_parser("walk-forward", help="Walk-Forward 滾動驗證")
    sp_wf.add_argument("--stock", required=True, help="股票代號")
    sp_wf.add_argument("--strategy", required=True, help="策略名稱 (ml_random_forest, ml_xgboost, ...)")
    sp_wf.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_wf.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")
    sp_wf.add_argument("--train-window", type=int, default=252, help="訓練窗口天數 (預設 252)")
    sp_wf.add_argument("--test-window", type=int, default=63, help="測試窗口天數 (預設 63)")
    sp_wf.add_argument("--step-size", type=int, default=63, help="步進大小 (預設 63)")
    sp_wf.add_argument("--lookback", type=int, default=None, help="ML 回溯天數")
    sp_wf.add_argument("--forward-days", type=int, default=None, help="ML 預測天數")
    sp_wf.add_argument("--threshold", type=float, default=None, help="ML 訊號門檻")
    sp_wf.add_argument("--train-ratio", type=float, default=None, help="ML 訓練比例")
    # 風險管理參數（複用）
    sp_wf.add_argument("--stop-loss", type=float, default=None, help="停損百分比")
    sp_wf.add_argument("--take-profit", type=float, default=None, help="停利百分比")
    sp_wf.add_argument("--trailing-stop", type=float, default=None, help="移動停損百分比")
    sp_wf.add_argument(
        "--sizing", default="all_in", choices=["all_in", "fixed_fraction", "kelly", "atr"], help="部位大小計算方式"
    )
    sp_wf.add_argument("--fraction", type=float, default=1.0, help="fixed_fraction 比例")
    sp_wf.add_argument(
        "--adjust-dividend", action="store_true", default=False, help="啟用除權息還原（回溯調整價格 + 股利入帳）"
    )
    sp_wf.add_argument("--export-trades", default=None, help="匯出交易明細 CSV 路徑")

    # report 子命令
    sp_report = subparsers.add_parser("report", help="每日選股報告")
    sp_report.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_report.add_argument("--top", type=int, default=10, help="顯示前 N 名 (預設 10)")
    sp_report.add_argument("--no-ml", action="store_true", help="跳過 ML 評分（較快）")
    sp_report.add_argument("--export", default=None, help="匯出 CSV 路徑")
    sp_report.add_argument("--notify", action="store_true", help="發送 Discord 通知")

    # strategy-rank 子命令
    sp_sr = subparsers.add_parser("strategy-rank", help="策略回測排名")
    sp_sr.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_sr.add_argument("--strategies", nargs="+", help="策略名稱（預設 6 個快速策略）")
    sp_sr.add_argument("--metric", default="sharpe", help="排名指標 (sharpe/total_return/win_rate/annual_return)")
    sp_sr.add_argument("--start", default=None, help="回測起始日期 (YYYY-MM-DD)")
    sp_sr.add_argument("--end", default=None, help="回測結束日期 (YYYY-MM-DD)")
    sp_sr.add_argument("--min-trades", type=int, default=3, help="最少交易次數 (預設 3)")
    sp_sr.add_argument("--export", default=None, help="匯出 CSV 路徑")
    sp_sr.add_argument("--notify", action="store_true", help="發送 Discord 通知")

    # industry 子命令
    sp_ind = subparsers.add_parser("industry", help="產業輪動分析")
    sp_ind.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_ind.add_argument("--refresh", action="store_true", help="強制重新抓取 StockInfo")
    sp_ind.add_argument("--top-sectors", type=int, default=5, help="顯示前 N 名產業 (預設 5)")
    sp_ind.add_argument("--top", type=int, default=5, help="每產業顯示前 N 支股票 (預設 5)")
    sp_ind.add_argument("--lookback", type=int, default=20, help="法人流量回溯天數 (預設 20)")
    sp_ind.add_argument("--momentum", type=int, default=60, help="價格動能回溯天數 (預設 60)")
    sp_ind.add_argument("--notify", action="store_true", help="發送 Discord 通知")

    # discover 子命令
    sp_disc = subparsers.add_parser("discover", help="全市場選股掃描 (momentum/swing/value/dividend/growth)")
    sp_disc.add_argument(
        "mode",
        nargs="?",
        default="momentum",
        choices=["momentum", "swing", "value", "dividend", "growth", "all"],
        help="掃描模式：momentum=短線動能, swing=中期波段, value=價值修復, dividend=高息存股, growth=高成長, all=五模式綜合比較 (預設 momentum)",
    )
    sp_disc.add_argument("--top", type=int, default=20, help="顯示前 N 名 (預設 20)")
    sp_disc.add_argument("--min-price", type=float, default=10, help="最低股價 (預設 10)")
    sp_disc.add_argument("--max-price", type=float, default=2000, help="最高股價 (預設 2000)")
    sp_disc.add_argument("--min-volume", type=int, default=500_000, help="最低成交量 (預設 500000)")
    sp_disc.add_argument("--sync-days", type=int, default=3, help="同步最近幾個交易日 (預設 3)")
    sp_disc.add_argument("--max-stocks", type=int, default=200, help="備案逐股抓取上限 (預設 200)")
    sp_disc.add_argument("--skip-sync", action="store_true", help="跳過全市場資料同步")
    sp_disc.add_argument("--export", default=None, help="匯出 CSV 路徑")
    sp_disc.add_argument("--notify", action="store_true", help="發送 Discord 通知")
    sp_disc.add_argument("--compare", action="store_true", help="顯示與上次推薦的差異比較")
    sp_disc.add_argument(
        "--min-appearances",
        type=int,
        default=1,
        help="[all 模式] 只顯示出現在 N 個以上模式的股票（預設 1 = 全部顯示）",
    )
    sp_disc.add_argument(
        "--weekly-confirm",
        action="store_true",
        default=False,
        help="啟用週線多時框確認（週線多頭 +5%%，週線空頭 -5%%，預設關閉）",
    )
    sp_disc.add_argument(
        "--use-ic-adjustment",
        action="store_true",
        default=False,
        help="啟用 Factor IC 動態權重調整（需 ≥20 筆歷史推薦，預設關閉）",
    )
    sp_disc.add_argument(
        "--ai-summary",
        action="store_true",
        default=False,
        help="呼叫 Claude API 生成 AI 選股摘要（需在 settings.yaml 設定 anthropic.api_key）",
    )

    # discover-backtest 子命令
    sp_db = subparsers.add_parser("discover-backtest", help="評估 Discover 推薦的歷史績效")
    sp_db.add_argument(
        "--mode", required=True, choices=["momentum", "swing", "value", "dividend", "growth"], help="掃描模式"
    )
    sp_db.add_argument(
        "--days",
        default=None,
        help="持有天數（逗號分隔；swing 預設 20,40,60，其他模式預設 5,10,20）",
    )
    sp_db.add_argument("--top", type=int, default=None, help="只計算每次掃描前 N 名的績效")
    sp_db.add_argument("--start", default=None, help="掃描日期範圍起始 (YYYY-MM-DD)")
    sp_db.add_argument("--end", default=None, help="掃描日期範圍結束 (YYYY-MM-DD)")
    sp_db.add_argument("--export", default=None, help="匯出明細 CSV 路徑")

    # sync-mops 子命令
    sp_mops = subparsers.add_parser("sync-mops", help="同步 MOPS 最新重大訊息公告")

    # sync-revenue 子命令
    sp_rev = subparsers.add_parser("sync-revenue", help="從 MOPS 同步全市場月營收（上市+上櫃）")
    sp_rev.add_argument("--months", type=int, default=1, help="同步最近幾個月（預設 1）")

    # sync-financial 子命令
    sp_fin = subparsers.add_parser("sync-financial", help="同步財報資料（季報損益表+資產負債表+現金流量表）")
    sp_fin.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_fin.add_argument("--quarters", type=int, default=4, help="同步最近幾季（預設 4）")

    # sync-info 子命令
    sp_info = subparsers.add_parser("sync-info", help="同步全市場股票基本資料（產業分類 + 上市/上櫃別）")
    sp_info.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="強制重新同步（即使 DB 已有資料，預設跳過）",
    )

    # sync-features 子命令
    sp_feat = subparsers.add_parser(
        "sync-features", help="計算全市場 DailyFeature（Feature Store，供 UniverseFilter 使用）"
    )
    sp_feat.add_argument("--days", type=int, default=90, help="回溯計算天數（預設 90）")

    # sync-holding 子命令
    sp_hold = subparsers.add_parser("sync-holding", help="同步大戶持股分級資料（週資料）")
    sp_hold.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_hold.add_argument("--weeks", type=int, default=4, help="同步最近幾週（預設 4）")

    # sync-vix 子命令
    subparsers.add_parser("sync-vix", help="同步 VIX 波動率指數（台灣 TW_VIX + 美國 US_VIX via yfinance）")

    # sync-sbl 子命令
    sp_sbl = subparsers.add_parser("sync-sbl", help="同步全市場借券賣出資料（TWSE TWT96U）")
    sp_sbl.add_argument("--days", type=int, default=3, help="同步最近幾個交易日（預設 3）")

    # sync-broker 子命令
    sp_broker = subparsers.add_parser("sync-broker", help="同步分點交易資料（FinMind TaiwanStockTradingDailyReport）")
    sp_broker.add_argument("--stocks", nargs="+", help="指定股票代號（預設使用 watchlist）")
    sp_broker.add_argument("--days", type=int, default=5, help="同步最近幾個交易日（預設 5）")
    sp_broker.add_argument("--from-discover", action="store_true", help="補抓最近一次 discover 推薦結果的分點資料")
    sp_broker.add_argument(
        "--watchlist-bootstrap",
        action="store_true",
        help="一次性逐日補齊 watchlist 所有股票分點歷史（預設 120 個交易日），啟用 Smart Broker 8F",
    )
    sp_broker.add_argument(
        "--from-file",
        metavar="PATH",
        help="從文字/CSV 檔案讀取股票代號清單（每行一筆，或 CSV 第一欄；可與 --watchlist-bootstrap 合用）",
    )

    # alert-check 子命令
    sp_alert = subparsers.add_parser("alert-check", help="掃描近期 MOPS 重大事件警報（法說會/財報/月營收）")
    sp_alert.add_argument("--days", type=int, default=7, help="查詢最近幾天（預設 7）")
    sp_alert.add_argument(
        "--types",
        nargs="+",
        choices=["earnings_call", "investor_day", "filing", "revenue"],
        help="篩選事件類型（預設全部）",
    )
    sp_alert.add_argument("--stocks", nargs="+", help="指定股票代號（預設全部）")
    sp_alert.add_argument("--notify", action="store_true", help="推播 Discord")

    # revenue-scan 子命令
    sp_rscan = subparsers.add_parser("revenue-scan", help="掃描 watchlist 營收高成長 + 毛利率改善個股")
    sp_rscan.add_argument("--stocks", nargs="+", help="股票代號（預設使用 watchlist）")
    sp_rscan.add_argument("--top", type=int, default=20, help="顯示前 N 支（預設 20）")
    sp_rscan.add_argument("--min-yoy", type=float, default=10.0, help="最低 YoY 門檻 %%（預設 10.0）")
    sp_rscan.add_argument("--min-margin-improve", type=float, default=0.0, help="毛利率 QoQ 最低改善 pp（預設 0.0）")
    sp_rscan.add_argument("--notify", action="store_true", help="推播 Discord")

    # validate 子命令
    sp_val = subparsers.add_parser("validate", help="資料品質檢查（缺漏、異常值、新鮮度）")
    sp_val.add_argument("--stocks", nargs="+", help="指定股票代號（預設檢查全部）")
    sp_val.add_argument("--gap-threshold", type=int, default=5, help="缺漏營業日門檻（預設 5）")
    sp_val.add_argument("--streak-threshold", type=int, default=5, help="連續漲跌停天數門檻（預設 5）")
    sp_val.add_argument("--no-freshness", action="store_true", help="跳過資料新鮮度檢查")
    sp_val.add_argument("--export", default=None, help="匯出問題清單 CSV 路徑")

    # export 子命令
    sp_exp = subparsers.add_parser("export", help="匯出資料表為 CSV/Parquet")
    sp_exp.add_argument("table", nargs="?", default=None, help="資料表名稱")
    sp_exp.add_argument("-o", "--output", default=None, help="輸出檔案路徑")
    sp_exp.add_argument("--format", default="csv", choices=["csv", "parquet"], help="輸出格式 (預設 csv)")
    sp_exp.add_argument("--stocks", nargs="+", help="篩選股票代號")
    sp_exp.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    sp_exp.add_argument("--end", default=None, help="結束日期 (YYYY-MM-DD)")
    sp_exp.add_argument("--list", action="store_true", help="列出所有可匯出的資料表及筆數")

    # import-data 子命令
    sp_imp = subparsers.add_parser("import-data", help="從 CSV/Parquet 匯入資料")
    sp_imp.add_argument("table", help="目標資料表名稱")
    sp_imp.add_argument("source", help="來源檔案路徑 (.csv 或 .parquet)")
    sp_imp.add_argument("--dry-run", action="store_true", help="僅驗證資料格式，不實際寫入")

    # suggest 子命令
    sp_suggest = subparsers.add_parser("suggest", help="單股進出場建議（ATR14 + SMA20 + RSI14 + Regime）")
    sp_suggest.add_argument("stock_id", help="股票代號（例：2330）")
    sp_suggest.add_argument("--notify", action="store_true", help="發送 Discord 通知")

    # watchlist 子命令（DB-based 觀察清單管理）
    sp_wlcmd = subparsers.add_parser("watchlist", help="DB-based 觀察清單管理（新增/移除/列出/從YAML匯入）")
    wlcmd_sub = sp_wlcmd.add_subparsers(dest="wl_action")

    # watchlist list
    wlcmd_sub.add_parser("list", help="列出 DB watchlist 清單")

    # watchlist add
    sp_wla = wlcmd_sub.add_parser("add", help="新增股票至 DB watchlist")
    sp_wla.add_argument("stock_id", help="股票代號（例：2330）")
    sp_wla.add_argument("--name", help="股票名稱（例：台積電）")
    sp_wla.add_argument("--note", help="備註")

    # watchlist remove
    sp_wlr = wlcmd_sub.add_parser("remove", help="從 DB watchlist 移除股票")
    sp_wlr.add_argument("stock_id", help="股票代號")

    # watchlist import（從 settings.yaml 一次性匯入）
    wlcmd_sub.add_parser("import", help="從 settings.yaml watchlist 一次性匯入所有股票至 DB")

    # watch 子命令
    sp_watch = subparsers.add_parser("watch", help="持倉監控管理（新增/列出/平倉/更新狀態）")
    watch_sub = sp_watch.add_subparsers(dest="action")

    # watch add
    sp_wa = watch_sub.add_parser("add", help="新增持倉監控")
    sp_wa.add_argument("stock_id", help="股票代號（例：2330）")
    sp_wa.add_argument("--price", type=float, default=None, help="進場價（預設使用最新收盤）")
    sp_wa.add_argument("--stop", type=float, default=None, help="止損價（預設 entry - 1.5×ATR14）")
    sp_wa.add_argument("--target", type=float, default=None, help="目標價（預設 entry + 3.0×ATR14）")
    sp_wa.add_argument("--qty", type=int, default=None, help="股數")
    sp_wa.add_argument(
        "--from-discover",
        metavar="MODE",
        default=None,
        help="從最新 discover 推薦記錄匯入（MODE: momentum/swing/value/dividend/growth）",
    )
    sp_wa.add_argument("--trailing", action="store_true", help="啟用移動止損（隨最高價自動上移止損位置）")
    sp_wa.add_argument(
        "--trailing-multiplier",
        type=float,
        default=1.5,
        metavar="MULT",
        help="移動止損 ATR 倍數（預設 1.5，即止損 = 最高價 - 1.5×ATR14）",
    )
    sp_wa.add_argument("--notes", default=None, help="備註")

    # watch list
    sp_wl = watch_sub.add_parser("list", help="列出持倉")
    sp_wl.add_argument(
        "--status",
        default="active",
        choices=["active", "stopped_loss", "taken_profit", "expired", "closed", "all"],
        help="篩選狀態（預設 active）",
    )

    # watch close
    sp_wc = watch_sub.add_parser("close", help="平倉（標記 closed）")
    sp_wc.add_argument("entry_id", type=int, help="持倉 ID（由 watch list 查詢）")
    sp_wc.add_argument("--price", type=float, default=None, help="平倉價格")

    # watch update-status
    watch_sub.add_parser("update-status", help="批次更新持倉狀態（比對最新收盤價自動標記止損/止利/過期）")

    # anomaly-scan 子命令
    sp_anomaly = subparsers.add_parser(
        "anomaly-scan", help="掃描 watchlist 成交量/籌碼異動警報（量能暴增/外資大買超/借券激增/主力集中）"
    )
    sp_anomaly.add_argument("--stocks", nargs="+", help="指定股票代號（預設使用 watchlist）")
    sp_anomaly.add_argument("--lookback", type=int, default=10, help="計算均量/均值的天數（預設 10）")
    sp_anomaly.add_argument(
        "--vol-mult", type=float, default=DEFAULT_VOL_MULT, dest="vol_mult", help="量能倍數門檻（預設 2.0）"
    )
    sp_anomaly.add_argument(
        "--inst-threshold",
        type=float,
        default=DEFAULT_INST_THRESHOLD,
        dest="inst_threshold",
        help="外資淨買超股數門檻（預設 3,000,000 股 = 3,000 張）",
    )
    sp_anomaly.add_argument(
        "--sbl-sigma", type=float, default=DEFAULT_SBL_SIGMA, dest="sbl_sigma", help="借券激增標準差倍數（預設 2.0σ）"
    )
    sp_anomaly.add_argument(
        "--hhi-threshold",
        type=float,
        default=DEFAULT_HHI_THRESHOLD,
        dest="hhi_threshold",
        help="主力分點集中度 HHI 門檻（預設 0.4）",
    )
    sp_anomaly.add_argument(
        "--dt-threshold",
        type=float,
        default=DEFAULT_DT_THRESHOLD,
        dest="dt_threshold",
        help="隔日沖風險 penalty 門檻（預設 0.3）",
    )
    sp_anomaly.add_argument("--notify", action="store_true", help="推播 Discord 通知")

    # morning-routine 子命令
    sp_mr = subparsers.add_parser(
        "morning-routine",
        help="每日早晨例行流程（sync-sbl → sync-broker → discover all → alert-check → watch update-status → revenue-scan → anomaly-scan → Discord 摘要）",
    )
    sp_mr.add_argument(
        "--dry-run",
        action="store_true",
        help="只顯示各步驟與摘要預覽，不實際執行",
    )
    sp_mr.add_argument(
        "--skip-sync",
        action="store_true",
        help="跳過 Step 1–2（借券/分點同步），適合資料已是最新時使用",
    )
    sp_mr.add_argument(
        "--top",
        type=int,
        default=20,
        help="discover all 的 Top N（預設 20）",
    )
    sp_mr.add_argument(
        "--notify",
        action="store_true",
        help="流程完成後推播 Discord 摘要",
    )

    # sync-concepts 子命令
    sp_sc = subparsers.add_parser("sync-concepts", help="從 concepts.yaml 同步概念定義至 DB，或以 MOPS 關鍵字自動標記")
    sp_sc.add_argument("--purge", action="store_true", help="先清除 source=yaml 的舊記錄再重新匯入（概念重組時使用）")
    sp_sc.add_argument(
        "--from-mops", action="store_true", dest="from_mops", help="掃描近期 MOPS 公告以關鍵字自動標記概念成員"
    )
    sp_sc.add_argument("--days", type=int, default=90, help="MOPS 掃描回溯天數（搭配 --from-mops，預設 90）")

    # concepts 子命令
    sp_cpt = subparsers.add_parser("concepts", help="概念股管理（列出 / 新增成員 / 移除成員）")
    cpt_sub = sp_cpt.add_subparsers(dest="concept_action")

    # concepts list [concept_name]
    sp_cpt_list = cpt_sub.add_parser("list", help="列出所有概念及成員數，或列出特定概念的成員清單")
    sp_cpt_list.add_argument("concept_name", nargs="?", default=None, help="概念名稱（省略則列出全部概念）")

    # concepts add <concept_name> <stock_id>
    sp_cpt_add = cpt_sub.add_parser("add", help="手動新增成員至概念（source=manual）")
    sp_cpt_add.add_argument("concept_name", help="概念名稱（例：CoWoS封裝）")
    sp_cpt_add.add_argument("stock_id", help="股票代號（例：2330）")

    # concepts remove <concept_name> <stock_id>
    sp_cpt_rem = cpt_sub.add_parser("remove", help="從概念移除成員")
    sp_cpt_rem.add_argument("concept_name", help="概念名稱")
    sp_cpt_rem.add_argument("stock_id", help="股票代號")

    # concept-expand 子命令（P2：相關性候選推薦）
    sp_cexp = subparsers.add_parser("concept-expand", help="以價格相關性找出概念候選股（P2）")
    sp_cexp.add_argument("concept_name", help="概念名稱（例：CoWoS封裝）")
    sp_cexp.add_argument("--threshold", type=float, default=0.7, help="平均相關係數門檻（預設 0.7）")
    sp_cexp.add_argument("--lookback", type=int, default=60, help="相關性計算回溯天數（預設 60）")
    sp_cexp.add_argument("--auto", action="store_true", help="自動將候選股寫入 DB（source=correlation），無需確認")

    # status 子命令
    subparsers.add_parser("status", help="顯示資料庫概況")

    # migrate 子命令
    subparsers.add_parser("migrate", help="執行資料庫 schema 遷移")

    # rotation 子命令（輪動組合部位控制）
    sp_rot = subparsers.add_parser("rotation", help="輪動組合部位控制（建立/更新/狀態/回測/管理）")
    rot_sub = sp_rot.add_subparsers(dest="action")

    # rotation create
    sp_rc = rot_sub.add_parser("create", help="建立輪動組合")
    sp_rc.add_argument("--name", required=True, help="組合名稱（唯一，如 mom5_3d）")
    sp_rc.add_argument(
        "--mode",
        required=True,
        choices=["momentum", "swing", "value", "dividend", "growth", "all"],
        help="discover 模式（all = 綜合排名）",
    )
    sp_rc.add_argument("--max-positions", type=int, required=True, help="最大持股數 N")
    sp_rc.add_argument("--holding-days", type=int, required=True, help="固定持有天數")
    sp_rc.add_argument("--capital", type=float, required=True, help="初始資金（TWD）")
    sp_rc.add_argument("--no-renewal", action="store_true", help="停用續持（到期一律賣出）")

    # rotation update
    sp_ru = rot_sub.add_parser("update", help="每日更新（讀取 discover 排名，執行換股）")
    sp_ru.add_argument("--name", default=None, help="指定組合名稱")
    sp_ru.add_argument("--all", action="store_true", help="更新所有 active 組合")

    # rotation status
    sp_rs = rot_sub.add_parser("status", help="查看組合狀態與持倉")
    sp_rs.add_argument("--name", default=None, help="指定組合名稱")
    sp_rs.add_argument("--all", action="store_true", help="列出所有組合概覽")

    # rotation history
    sp_rh = rot_sub.add_parser("history", help="已平倉交易記錄")
    sp_rh.add_argument("--name", required=True, help="組合名稱")
    sp_rh.add_argument("--limit", type=int, default=30, help="顯示筆數（預設 30）")

    # rotation backtest
    sp_rb = rot_sub.add_parser("backtest", help="歷史回測（使用過去 DiscoveryRecord）")
    sp_rb.add_argument("--name", default=None, help="從已建立組合讀取參數")
    sp_rb.add_argument("--mode", default=None, help="覆蓋模式（ad-hoc 回測用）")
    sp_rb.add_argument("--max-positions", type=int, default=None, help="覆蓋持股數")
    sp_rb.add_argument("--holding-days", type=int, default=None, help="覆蓋持有天數")
    sp_rb.add_argument("--capital", type=float, default=None, help="覆蓋初始資金")
    sp_rb.add_argument("--start", required=True, help="回測起始日 (YYYY-MM-DD)")
    sp_rb.add_argument("--end", required=True, help="回測結束日 (YYYY-MM-DD)")

    # rotation list
    rot_sub.add_parser("list", help="列出所有輪動組合")

    # rotation pause / resume / delete
    sp_rp = rot_sub.add_parser("pause", help="暫停組合每日更新")
    sp_rp.add_argument("--name", required=True, help="組合名稱")
    sp_rr = rot_sub.add_parser("resume", help="恢復組合每日更新")
    sp_rr.add_argument("--name", required=True, help="組合名稱")
    sp_rd = rot_sub.add_parser("delete", help="刪除組合及所有持倉")
    sp_rd.add_argument("--name", required=True, help="組合名稱")

    args = parser.parse_args()

    if args.command == "sync":
        cmd_sync(args)
    elif args.command == "compute":
        cmd_compute(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "dashboard":
        cmd_dashboard()
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "schedule":
        cmd_schedule(args)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "notify":
        cmd_notify(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "walk-forward":
        cmd_walk_forward(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "strategy-rank":
        cmd_strategy_rank(args)
    elif args.command == "industry":
        cmd_industry(args)
    elif args.command == "discover":
        cmd_discover(args)
    elif args.command == "discover-backtest":
        cmd_discover_backtest(args)
    elif args.command == "sync-mops":
        cmd_sync_mops(args)
    elif args.command == "sync-revenue":
        cmd_sync_revenue(args)
    elif args.command == "sync-financial":
        cmd_sync_financial(args)
    elif args.command == "sync-info":
        cmd_sync_info(args)
    elif args.command == "sync-features":
        cmd_sync_features(args)
    elif args.command == "sync-holding":
        cmd_sync_holding(args)
    elif args.command == "sync-vix":
        cmd_sync_vix(args)
    elif args.command == "sync-sbl":
        cmd_sync_sbl(args)
    elif args.command == "sync-broker":
        cmd_sync_broker(args)
    elif args.command == "alert-check":
        cmd_alert_check(args)
    elif args.command == "revenue-scan":
        cmd_revenue_scan(args)
    elif args.command == "migrate":
        cmd_migrate(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "import-data":
        cmd_import_data(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "suggest":
        cmd_suggest(args)
    elif args.command == "anomaly-scan":
        cmd_anomaly_scan(args)
    elif args.command == "morning-routine":
        cmd_morning_routine(args)
    elif args.command == "sync-concepts":
        cmd_sync_concepts(args)
    elif args.command == "concepts":
        cmd_concepts(args)
    elif args.command == "concept-expand":
        cmd_concept_expand(args)
    elif args.command == "watchlist":
        cmd_watchlist(args)
    elif args.command == "watch":
        if not args.action:
            sp_watch.print_help()
        else:
            cmd_watch(args)
    elif args.command == "rotation":
        if not getattr(args, "action", None):
            sp_rot.print_help()
        else:
            cmd_rotation(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
