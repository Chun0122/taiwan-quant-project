# 設計文件：Live 全面 T+1（Pending-Order 機制）

> 狀態：**設計（DESIGN ONLY，尚未實作）** — 路線 B2，2026-06-20
> 前置依賴：路線 B1 已完成（`src/portfolio/execution_core.py` 的 `simulate_buy`/`simulate_sell`）。
> 延後實作主因：目前系統 paper-only，paper 階段 live 同日成交與 T+1 的 parity 差異影響有限；
> 且本案會動到每日例行流程（morning-routine + 排程），成本高於眼前效益。實盤上線（Route C）前再落地。

---

## 1. 問題

`RotationManager.backtest()` 已是 **T+1**：D 日 close 決策 → 暫存 `pending_exec` → D+1 開盤成交（`manager.py` `_execute_action_set`，買賣一律 `open[D+1]`）。

但 `RotationManager.update()`（live）仍 **同日成交**：在 `today` 迴圈內直接呼叫 `_execute_sell`（`manager.py:~331/443`）/ `_execute_buy`（`~465`），以 `today` 的 close/OHLCV 成交。決策用 `close[today]` 排名 → 同日就以 `today` 價格模擬成交。

**後果**：回測（D+1 open）與實盤（D close）執行時點不一致 → 回測無法忠實預測實盤；且 live 用了決策當日收盤價，實務上夜間決策時根本買不到當日收盤價（look-ahead 性質）。要讓「回測 = 實盤」在執行時點上成立，live 必須改為 D 決策、D+1 開盤成交。

**Schema 現況**：無任何 pending / intent 概念。`RotationPosition.status` 僅 `open`/`closed`；`RotationActionLog` 是事後紀錄。backtest 的 `pending_exec` 只是 in-memory tuple，從不落庫。跨夜的 live T+1 必須把「待成交意圖」持久化 → **net-new schema**。

---

## 2. Schema 草案：`RotationPendingOrder`

新增 ORM model 至 `src/data/schema.py`。新表由 `init_db()` 的 `create_all` 自動建立，**不需**列入 `src/data/migrate.py` 的 `MIGRATIONS`（該清單僅供既有表 `ALTER TABLE ADD COLUMN`；新表自動建立，見 `migrate.py:run_migrations` 先呼叫 `init_db()`）。

```python
class RotationPendingOrder(Base):
    """待成交意圖（live T+1）：D 日決策、D+1 開盤成交之間的持久化暫存。"""
    __tablename__ = "rotation_pending_order"

    id = Column(Integer, primary_key=True)
    portfolio_name = Column(String, nullable=False, index=True)
    decision_date = Column(Date, nullable=False)   # D：產生此意圖的決策日
    exec_date = Column(Date, nullable=True)        # D+1：實際成交日（成交後回填）
    side = Column(String, nullable=False)          # "buy" | "sell"
    stock_id = Column(String, nullable=False)
    stock_name = Column(String, default="")
    shares = Column(Integer, nullable=False)       # 決策時規劃股數（買=目標，賣=持倉股數）
    ref_price = Column(Float, nullable=False)      # 決策日 close（僅供 audit / 資金不足保護比較）
    reason = Column(String, default="")            # sell 的 exit_reason（stop_loss/holding_expired/...）
    entry_rank = Column(Integer, nullable=True)    # buy 用
    entry_score = Column(Float, nullable=True)     # buy 用（Gate B / audit）
    score_breakdown_json = Column(Text, nullable=True)  # buy 用（entry_score_breakdown，沿用 P1 任務 5）
    stop_loss = Column(Float, nullable=True)       # buy 用
    status = Column(String, nullable=False, default="pending")  # pending | filled | cancelled
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("portfolio_name", "decision_date", "side", "stock_id",
                         name="uq_pending_order"),
        Index("ix_pending_status", "portfolio_name", "status"),
    )
```

設計要點：
- `status` 三態：`pending`（已決策待成交）→ `filled`（D+1 成交）/ `cancelled`（D+1 無法成交且逾期）。
- 一個 portfolio 同一 `decision_date` 同一 `stock_id` 同一 `side` 唯一（防重複決策）。
- `ref_price` 不是成交價，只供 audit 與「D+1 open 高於 ref_price 的資金不足保護」比較（backtest 已有此邏輯）。

---

## 3. 兩段式流程

把目前 `update()` 一次到位的「決策 + 成交」拆成兩個觸點：

### 階段 A — 決策（D 日收盤後 / 夜間）
新方法 `RotationManager.decide(today)`（或 `update(stage="decide")`）：
1. 載入 open positions + rankings（同現行 `update()` 前半段）。
2. 跑 drawdown kill switch、`compute_rotation_actions(...)`（風控 overlay 全傳，與現行一致）。
3. **不成交**。把 `actions.to_buy` / `to_sell`（含 stop_loss 觸發、到期、危機、回撤）逐筆寫成 `RotationPendingOrder(status="pending", decision_date=today)`。
4. `RotationActionLog` 以新 `action_type="pending"` 記錄（dashboard 可顯示「明日預定操作」）。
5. **續持（renew）** 不需 pending：直接更新 `RotationPosition.planned_exit_date` / trailing stop（不涉及成交）。

### 階段 B — 成交（D+1 開盤後）
新方法 `RotationManager.fill_pending(exec_day)`：
1. 查 `status="pending"` 且 `decision_date < exec_day` 的 orders。
2. 取 `exec_day` 的 OHLCV（open + volume，供動態滑價 / 流動性）。
3. 逐筆呼叫 **B1 的 `simulate_sell` / `simulate_buy`**，以 `open[exec_day]` 成交：
   - 賣出：`simulate_sell(...)` → 更新對應 `RotationPosition`（status=closed、exit_*、pnl）+ 回收現金。
   - 買入：先 `apply_liquidity_limit` + 資金不足 reshrink（**沿用 backtest `manager.py:~1299` 的「open 高於決策 close → 縮股」保護**）→ `simulate_buy(...)` → 建 `RotationPosition`。
4. order 標 `status="filled"`、回填 `exec_date`。
5. 重算 `portfolio.current_cash / current_capital`、寫 `RotationDailySnapshot`、`RotationActionLog`（action_type=open/close）。

> **B1 紅利**：階段 B 的成交金額算式直接重用 `simulate_buy`/`simulate_sell`，與 backtest 同一份核心 → 執行時點與金額算式雙重對齊，drift 風險降到最低。

---

## 4. morning-routine / 排程影響

現行 morning-routine **Step 12 `rotation update`** 一次做完決策 + 成交。改為 T+1 後需拆成兩觸點：

| 觸點 | 時機 | 動作 |
|------|------|------|
| 決策 | D 日收盤後（現行夜間 morning-routine 即可） | `decide(D)` 寫 pending orders |
| 成交 | D+1 開盤後（**新增晨間觸點**） | `fill_pending(D+1)` 以 open 成交 |

- 需新增一個「開盤後」排程（launchd 約 09:05 台股開盤後），或把 `fill_pending` 併入「隔日」morning-routine 開頭（先成交昨日 pending，再做今日決策）——後者較省排程，且與 backtest 迴圈「每日開頭先執行上一日 pending」結構一致（`manager.py:~1369`）。**建議採後者**：morning-routine Step 12 改為「先 `fill_pending(today)` → 再 `decide(today)`」。
- baseline regression（Step 17）、export-dashboard（Step 16）落在 `fill_pending` 之後不變。

---

## 5. 未決點（實作時定案）

1. **D+1 open 缺報價（停牌/暫停交易）**：pending order 如何處理？選項：(a) 順延至下一交易日（保持 pending）；(b) 直接 cancel。建議買單順延有 TTL、賣單（尤其 stop_loss）以最後已知價成交避免凍結（對齊 backtest survivorship 守門）。
2. **pending order TTL**：跨日未成交保留幾日？建議買單 ≤2 交易日後 cancel（決策已過期），風控賣單不設 TTL（必須出場）。
3. **dry_run preview**：現行 `update(dry_run=True)`（rotation preview CLI）如何顯示？建議 `decide(dry_run=True)` 列出將寫入的 pending orders 但不落庫。
4. **危機/kill-switch 的即時性**：drawdown kill switch（≥25% 清倉）目前 live 同日強平。T+1 化後，熔斷是否也延到 open[D+1]？風控上「越快越好」與「T+1 一致性」衝突。建議**熔斷維持即時**（風控優先於 parity），僅一般換股走 T+1。需在 backtest 對應（backtest 目前熔斷也走 T+1 open，會有 live/backtest 對此項的刻意差異，需註記）。
5. **部分成交**：流動性限制下單筆買不滿目標股數，剩餘是否轉下一日？建議不轉（當日 fill 多少算多少，與 backtest 一致）。

---

## 6. 測試計畫（實作時）

- `RotationPendingOrder` schema + migration round-trip。
- `decide()`：寫出正確 pending orders、不動 cash/position、dry_run 不落庫。
- `fill_pending()`：以 open 成交、資金不足縮股、停牌順延/取消、status 轉移、重用 simulate_* 金額正確。
- E2E：`decide(D)` → `fill_pending(D+1)` 兩段式與既有同日 `update()` 在「無跳空」情境下結果一致；有跳空時 entry 落在 open[D+1]。
- 與 backtest 對照：同一段 discovery 史料，live 兩段式與 backtest T+1 路徑成交時點一致。

---

## 7. 不在本案範圍

- 券商下單 API / 真實訂單狀態機 / 部位對帳（Route C，凍結）。`RotationPendingOrder` 是**模擬**的待成交意圖，不是真實掛單。
- backtest 不需改（已 T+1）。
