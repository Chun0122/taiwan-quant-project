# 專案完成歷史

Phase 1 + Phase 2 全部完成，共 85 項任務。測試數從 231 → 1761。

Phase 3 實盤上線**暫緩**。

---

## 各階段摘要

| 階段 | Task # | 重點 |
|------|--------|------|
| **基礎建設** | 1~10 | 五模式 Scanner、Dashboard、validate/export/import CLI |
| **進出場 + 持倉** | 11~14 | ATR14+SMA20 建議、suggest、ATR 止損止利、watch 監控 |
| **資料整合** | 15~20 | TWSE/TPEX Cold-Start、TDCC 大戶、MOPS 事件、SBL、DJ 分點 |
| **自動化 + 監控** | 21~24 | morning-routine、Trailing Stop、anomaly-scan、DB watchlist |
| **評分升級** | 25~41 | 週線多時框、進階回測指標、五因子歸因、Smart Broker、Bootstrap、AI 摘要、Regime 自適應 |
| **Universe + 品質** | 42~49 | Universe 三層漏斗 + Feature Store、同業 PE、EPS 連續性、消息面優化、Swing 強化 |
| **概念 + Crisis** | 50~53 | 概念股系統、Crisis 第四狀態、隔日沖偵測、風險過濾強化 |
| **統一重構** | 54~58 | entry_exit.py 共用、市場寬度降級、Hysteresis 狀態機、法人連續性 + HHI 趨勢 |
| **Phase 1 基盤強化** | 59~67 | Regime 預設值、Kelly 收縮、Drawdown 連續化、相關性危機自適應、分數上限修正、早晨原子性、每股新鮮度、波動率權重、危機強制平倉 |
| **Phase 2 擬真度** | 68~75 | 假日行事曆、公告衰減分化、籌碼層級稽核、滑價不對稱、Factor IC 動態權重、Regime 部位大小、漲跌停模擬、部分止利 |

---

## 後期重點任務（Task 76~85）

| Task | 名稱 | 重點 |
|------|------|------|
| 76 | Universe 強化 | Regime 自適應門檻、min_close 軟化、相對流動性救援、突破型過濾器、Candidate Memory 3 天漸進衰減 |
| 77 | 因子權重優化 | 技術面 Cluster 等權（3 群 mean 降維）、Momentum Regime 權重 IC 校準、morning-routine 啟用 IC 動態調整 |
| 78 | 跨平台排程 | macOS LaunchAgent 排程、`--mode auto` 平台自動偵測（+11 測試） |
| 79 | Rotation 回測擬真度 | 三因子動態滑價、流動性約束、漲跌停模擬、成本歸因、TAIEX Benchmark、Schema 遷移 11 欄位（+28 測試） |
| 80 | Ex-Ante VaR | 共變異數矩陣 + ridge 正則化、參數化 VaR + Component VaR 分解（+11 測試） |
| 81 | 子因子 IC 自動化 | 子因子 IC 權重調整 + min_samples 防護、`_get_chip_base_weights` 純函數化（+11 測試） |
| 82 | Discover+Rotation 效能強化 | OHLCV 預載入快取、交易成本扣減、T+1 開盤進場、向量化、持倉快照（+11 測試） |
| 83 | Universe 覆蓋率 fallback | DailyFeature 覆蓋率檢查、不足時 fallback DailyPrice（+1 測試） |
| 84 | 因子結構優化 | 技術面 Cluster v2、籌碼面冗餘移除、零方差因子自動排除（+6 測試） |
| 85 | IC 分析機制強化 | Rolling IC、Per-Regime IC、消融績效修正（select_ratio + selection_overlap）（+5 測試） |
