"""SQLAlchemy ORM 資料表定義。

六張核心表：
- DailyPrice:              日K線（OHLCV + 還原收盤價）
- InstitutionalInvestor:   三大法人買賣超
- MarginTrading:           融資融券
- TechnicalIndicator:      技術指標（EAV 長表）
- BacktestResult:          回測結果摘要
- Trade:                   交易明細
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, BigInteger, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.data.database import Base


class DailyPrice(Base):
    """個股日 K 線資料。"""

    __tablename__ = "daily_price"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_daily_price"),
    )

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
    __table_args__ = (
        UniqueConstraint("stock_id", "date", "name", name="uq_institutional"),
    )

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
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_margin_trading"),
    )

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


class TechnicalIndicator(Base):
    """技術指標（EAV 長表）。

    每一列代表某股票某日某指標的值，例如：
    (2330, 2024-01-02, "sma_20", 580.5)
    """

    __tablename__ = "technical_indicator"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", "name", name="uq_technical_indicator"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(30), nullable=False, index=True)  # e.g. sma_5, rsi_14
    value: Mapped[float] = mapped_column(Float, nullable=False)

    def __repr__(self) -> str:
        return f"<Indicator {self.stock_id} {self.date} {self.name}={self.value}>"


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
    total_return: Mapped[float] = mapped_column(Float, nullable=False)      # 總報酬率 (%)
    annual_return: Mapped[float] = mapped_column(Float, nullable=False)     # 年化報酬率 (%)
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=True)       # Sharpe Ratio
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)      # 最大回撤 (%)
    win_rate: Mapped[float] = mapped_column(Float, nullable=True)           # 勝率 (%)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False)
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
    shares: Mapped[int] = mapped_column(Integer, nullable=False)            # 股數
    pnl: Mapped[float] = mapped_column(Float, nullable=True)               # 損益金額
    return_pct: Mapped[float] = mapped_column(Float, nullable=True)        # 報酬率 (%)

    def __repr__(self) -> str:
        return f"<Trade {self.entry_date}~{self.exit_date} pnl={self.pnl}>"
