"""SQLAlchemy ORM 資料表定義。

十四張核心表：
- DailyPrice:              日K線（OHLCV + 還原收盤價）
- InstitutionalInvestor:   三大法人買賣超
- MarginTrading:           融資融券
- MonthlyRevenue:          月營收
- StockValuation:          估值資料（PE/PB/殖利率）
- Dividend:                股利資料
- TechnicalIndicator:      技術指標（EAV 長表）
- Announcement:            MOPS 重大訊息公告
- FinancialStatement:      季報財務資料（損益表+資產負債表+現金流量表）
- BacktestResult:          回測結果摘要
- Trade:                   交易明細
- StockInfo:               股票基本資料（產業分類）
- PortfolioBacktestResult: 投資組合回測結果
- DiscoveryRecord:         Discover 推薦記錄（歷史追蹤）
"""

from datetime import date, datetime

from sqlalchemy import BigInteger, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.data.database import Base


class DailyPrice(Base):
    """個股日 K 線資料。"""

    __tablename__ = "daily_price"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_daily_price"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 成交股數
    turnover: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 成交金額
    spread: Mapped[float] = mapped_column(Float, nullable=True)  # 漲跌價差

    def __repr__(self) -> str:
        return f"<DailyPrice {self.stock_id} {self.date} close={self.close}>"


class InstitutionalInvestor(Base):
    """三大法人買賣超資料。"""

    __tablename__ = "institutional_investor"
    __table_args__ = (UniqueConstraint("stock_id", "date", "name", name="uq_institutional"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(20), nullable=False)  # 外資/投信/自營商
    buy: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 買進股數
    sell: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 賣出股數
    net: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 買賣超股數

    def __repr__(self) -> str:
        return f"<Institutional {self.stock_id} {self.date} {self.name} net={self.net}>"


class MarginTrading(Base):
    """融資融券資料。"""

    __tablename__ = "margin_trading"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_margin_trading"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    margin_buy: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 融資買進
    margin_sell: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 融資賣出
    margin_balance: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 融資餘額
    short_sell: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 融券賣出
    short_buy: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 融券買進
    short_balance: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 融券餘額

    def __repr__(self) -> str:
        return f"<Margin {self.stock_id} {self.date} 融資餘額={self.margin_balance}>"


class MonthlyRevenue(Base):
    """月營收資料。"""

    __tablename__ = "monthly_revenue"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_monthly_revenue"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    revenue: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 營收金額
    revenue_month: Mapped[int] = mapped_column(Integer, nullable=False)  # 月份
    revenue_year: Mapped[int] = mapped_column(Integer, nullable=False)  # 年度
    mom_growth: Mapped[float | None] = mapped_column(Float, nullable=True)  # 月增率 (%)
    yoy_growth: Mapped[float | None] = mapped_column(Float, nullable=True)  # 年增率 (%)

    def __repr__(self) -> str:
        return f"<MonthlyRevenue {self.stock_id} {self.date} revenue={self.revenue:,}>"


class StockValuation(Base):
    """估值資料（本益比/股淨比/殖利率）。"""

    __tablename__ = "stock_valuation"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_stock_valuation"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    pe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)  # 本益比
    pb_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)  # 股價淨值比
    dividend_yield: Mapped[float | None] = mapped_column(Float, nullable=True)  # 殖利率 (%)

    def __repr__(self) -> str:
        return f"<Valuation {self.stock_id} {self.date} PE={self.pe_ratio} PB={self.pb_ratio}>"


class Dividend(Base):
    """股利資料。"""

    __tablename__ = "dividend"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_dividend"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)  # 除權息基準日
    year: Mapped[str] = mapped_column(String(10), nullable=False)  # 股利所屬年度
    cash_dividend: Mapped[float | None] = mapped_column(Float, nullable=True)  # 現金股利
    stock_dividend: Mapped[float | None] = mapped_column(Float, nullable=True)  # 股票股利
    cash_payment_date: Mapped[date | None] = mapped_column(Date, nullable=True)  # 現金發放日
    announcement_date: Mapped[date | None] = mapped_column(Date, nullable=True)  # 公告日

    def __repr__(self) -> str:
        return f"<Dividend {self.stock_id} {self.date} cash={self.cash_dividend}>"


class TechnicalIndicator(Base):
    """技術指標（EAV 長表）。

    每一列代表某股票某日某指標的值，例如：
    (2330, 2024-01-02, "sma_20", 580.5)
    """

    __tablename__ = "technical_indicator"
    __table_args__ = (UniqueConstraint("stock_id", "date", "name", name="uq_technical_indicator"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(30), nullable=False, index=True)  # e.g. sma_5, rsi_14
    value: Mapped[float] = mapped_column(Float, nullable=False)

    def __repr__(self) -> str:
        return f"<Indicator {self.stock_id} {self.date} {self.name}={self.value}>"


class Announcement(Base):
    """MOPS 重大訊息公告。"""

    __tablename__ = "announcement"
    __table_args__ = (UniqueConstraint("stock_id", "date", "seq", name="uq_announcement"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    seq: Mapped[str] = mapped_column(String(10), nullable=False)  # 當日序號
    subject: Mapped[str] = mapped_column(String(500), nullable=False)  # 公告主旨
    spoke_time: Mapped[str | None] = mapped_column(String(10), nullable=True)  # 發言時間
    sentiment: Mapped[int] = mapped_column(Integer, default=0)  # +1 正面 / 0 中性 / -1 負面

    def __repr__(self) -> str:
        return f"<Announcement {self.stock_id} {self.date} seq={self.seq}>"


class FinancialStatement(Base):
    """季報財務資料（損益表 + 資產負債表 + 現金流量表）。

    來源：FinMind API（TaiwanStockFinancialStatements / BalanceSheet / CashFlowsStatement）。
    EAV 格式 pivot 後存入此寬表，每支股票每季一筆。
    """

    __tablename__ = "financial_statement"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_financial_statement"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)  # 季度結束日
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    quarter: Mapped[int] = mapped_column(Integer, nullable=False)  # 1~4

    # 綜合損益表
    revenue: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    gross_profit: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    operating_income: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    net_income: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    eps: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 資產負債表
    total_assets: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    total_liabilities: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    equity: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # 現金流量表
    operating_cf: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    investing_cf: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    financing_cf: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    free_cf: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # = operating_cf + investing_cf

    # 衍生比率（寫入時計算）
    gross_margin: Mapped[float | None] = mapped_column(Float, nullable=True)  # 毛利率 %
    operating_margin: Mapped[float | None] = mapped_column(Float, nullable=True)  # 營益率 %
    net_margin: Mapped[float | None] = mapped_column(Float, nullable=True)  # 淨利率 %
    roe: Mapped[float | None] = mapped_column(Float, nullable=True)  # ROE %
    roa: Mapped[float | None] = mapped_column(Float, nullable=True)  # ROA %
    debt_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)  # 負債比 %

    def __repr__(self) -> str:
        return f"<FinancialStatement {self.stock_id} {self.date} Q{self.quarter} EPS={self.eps}>"


class BacktestResult(Base):
    """回測結果摘要。"""

    __tablename__ = "backtest_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    initial_capital: Mapped[float] = mapped_column(Float, nullable=False)
    final_capital: Mapped[float] = mapped_column(Float, nullable=False)
    total_return: Mapped[float] = mapped_column(Float, nullable=False)  # 總報酬率 (%)
    annual_return: Mapped[float] = mapped_column(Float, nullable=False)  # 年化報酬率 (%)
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=True)  # Sharpe Ratio
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)  # 最大回撤 (%)
    win_rate: Mapped[float] = mapped_column(Float, nullable=True)  # 勝率 (%)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    benchmark_return: Mapped[float | None] = mapped_column(Float, nullable=True)  # 基準報酬率 (%)
    sortino_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    calmar_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    var_95: Mapped[float | None] = mapped_column(Float, nullable=True)  # VaR (95%)
    cvar_95: Mapped[float | None] = mapped_column(Float, nullable=True)  # CVaR (95%)
    profit_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Backtest {self.stock_id} {self.strategy_name} return={self.total_return:.2f}%>"


class Trade(Base):
    """交易明細。"""

    __tablename__ = "trade"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_id: Mapped[int] = mapped_column(Integer, ForeignKey("backtest_result.id"), nullable=False, index=True)
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_date: Mapped[date] = mapped_column(Date, nullable=True)
    exit_price: Mapped[float] = mapped_column(Float, nullable=True)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)  # 股數
    pnl: Mapped[float] = mapped_column(Float, nullable=True)  # 損益金額
    return_pct: Mapped[float] = mapped_column(Float, nullable=True)  # 報酬率 (%)
    exit_reason: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # signal/stop_loss/take_profit/trailing_stop/force_close

    def __repr__(self) -> str:
        return f"<Trade {self.entry_date}~{self.exit_date} pnl={self.pnl}>"


class PortfolioBacktestResult(Base):
    """投資組合回測結果摘要。"""

    __tablename__ = "portfolio_backtest_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    stock_ids: Mapped[str] = mapped_column(Text, nullable=False)  # 逗號分隔
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    initial_capital: Mapped[float] = mapped_column(Float, nullable=False)
    final_capital: Mapped[float] = mapped_column(Float, nullable=False)
    total_return: Mapped[float] = mapped_column(Float, nullable=False)
    annual_return: Mapped[float] = mapped_column(Float, nullable=False)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    sortino_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    calmar_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    var_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    cvar_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    allocation_method: Mapped[str | None] = mapped_column(String(30), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<PortfolioBacktest {self.stock_ids} {self.strategy_name} return={self.total_return:.2f}%>"


class StockInfo(Base):
    """股票基本資料（產業分類）。"""

    __tablename__ = "stock_info"
    __table_args__ = (UniqueConstraint("stock_id", name="uq_stock_info"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    stock_name: Mapped[str] = mapped_column(String(50), nullable=True)
    industry_category: Mapped[str] = mapped_column(String(50), nullable=True, index=True)
    listing_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<StockInfo {self.stock_id} {self.stock_name} [{self.industry_category}]>"


class PortfolioTrade(Base):
    """投資組合交易明細。"""

    __tablename__ = "portfolio_trade"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_backtest_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("portfolio_backtest_result.id"), nullable=False, index=True
    )
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(20), nullable=True)

    def __repr__(self) -> str:
        return f"<PortfolioTrade {self.stock_id} {self.entry_date}~{self.exit_date} pnl={self.pnl}>"


class DiscoveryRecord(Base):
    """Discover 推薦記錄 — 追蹤每次掃描的推薦結果。

    每次 discover 執行後，將 Top N 結果逐筆存入此表，
    供歷史比較（新進/退出/排名變化）與推薦績效回測使用。
    """

    __tablename__ = "discovery_record"
    __table_args__ = (UniqueConstraint("scan_date", "mode", "stock_id", name="uq_discovery_record"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # momentum/swing/value
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    stock_name: Mapped[str] = mapped_column(String(50), nullable=True)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    technical_score: Mapped[float] = mapped_column(Float, nullable=True)
    chip_score: Mapped[float] = mapped_column(Float, nullable=True)
    fundamental_score: Mapped[float] = mapped_column(Float, nullable=True)
    news_score: Mapped[float] = mapped_column(Float, nullable=True)
    sector_bonus: Mapped[float] = mapped_column(Float, nullable=True)
    industry_category: Mapped[str] = mapped_column(String(50), nullable=True)
    regime: Mapped[str] = mapped_column(String(20), nullable=True)  # bull/bear/sideways
    total_stocks: Mapped[int] = mapped_column(Integer, nullable=True)  # 掃描總股數
    after_coarse: Mapped[int] = mapped_column(Integer, nullable=True)  # 粗篩後股數
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<DiscoveryRecord {self.scan_date} {self.mode} #{self.rank} {self.stock_id}>"
