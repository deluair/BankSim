"""
Foreign Exchange Models.

This module defines SQLAlchemy models for foreign exchange operations:
- Exchange rates
- Foreign reserves
- FX interventions
- Balance of payments data
"""

from sqlalchemy import Column, Integer, Float, String, Date, DateTime, ForeignKey, Text, Boolean, Enum
from sqlalchemy.orm import relationship
import enum
from datetime import date, datetime

from ..db.session import Base  # Corrected Base import


class ExchangeRate(Base):
    """
    Exchange rate data between the local currency and foreign currencies.
    
    Tracks daily exchange rates for major currencies.
    """
    __tablename__ = "exchange_rates"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    currency_code = Column(String(3), nullable=False, index=True)  # ISO currency code (e.g., USD, EUR)
    
    # Exchange rates (local currency per unit of foreign currency)
    reference_rate = Column(Float, nullable=False)  # Official central bank rate
    buying_rate = Column(Float)  # Central bank buying rate
    selling_rate = Column(Float)  # Central bank selling rate
    
    # Market rates
    interbank_rate = Column(Float)  # Rate in interbank market
    weighted_average_rate = Column(Float)  # Weighted average of all transactions
    
    # Additional metrics
    daily_volatility = Column(Float)  # Standard deviation of intraday rates
    daily_volume = Column(Float)  # Trading volume in millions of USD
    
    # Metadata
    is_official = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EffectiveExchangeRate(Base):
    """
    Effective exchange rate indices (nominal and real).
    
    These indices measure the value of the local currency against a basket of currencies.
    """
    __tablename__ = "effective_exchange_rates"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True, unique=True)
    
    # Nominal effective exchange rate (index value)
    neer = Column(Float, nullable=False)
    neer_change_mom = Column(Float)  # Month-over-month percentage change
    neer_change_yoy = Column(Float)  # Year-over-year percentage change
    
    # Real effective exchange rate (index value)
    reer = Column(Float, nullable=False)
    reer_change_mom = Column(Float)  # Month-over-month percentage change
    reer_change_yoy = Column(Float)  # Year-over-year percentage change
    
    # Base period information
    base_year = Column(Integer, nullable=False)  # e.g., 2015 = 100
    basket_currencies = Column(String)  # Comma-separated list of currencies in the basket
    
    # Methodology details
    weighting_method = Column(String)  # e.g., "Trade weights", "GDP weights"
    price_index_used = Column(String)  # For REER calculation, e.g., "CPI", "PPI"
    
    # Metadata
    is_provisional = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ReserveAssetType(enum.Enum):
    """Types of foreign reserve assets."""
    CURRENCY_DEPOSITS = "currency_deposits"
    SECURITIES = "securities"
    SDR = "sdr"  # Special Drawing Rights
    IMF_POSITION = "imf_position"  # Reserve position in the IMF
    GOLD = "gold"
    OTHER = "other"


class ForeignReserve(Base):
    """
    Foreign reserve holdings of the central bank.
    
    Tracks the composition and value of international reserves.
    """
    __tablename__ = "foreign_reserves"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True, unique=True)
    
    # Total reserves value
    total_reserves_usd = Column(Float, nullable=False)  # In millions of USD
    total_reserves_months_imports = Column(Float)  # Reserves in months of imports
    
    # Composition by asset type (in millions of USD)
    currency_deposits = Column(Float)
    securities = Column(Float)
    sdr_holdings = Column(Float)  # Special Drawing Rights
    imf_reserve_position = Column(Float)
    gold_value = Column(Float)
    other_reserve_assets = Column(Float)
    
    # Composition by currency (in millions of USD)
    usd_holdings = Column(Float)
    eur_holdings = Column(Float)
    gbp_holdings = Column(Float)
    jpy_holdings = Column(Float)
    cny_holdings = Column(Float)
    other_currency_holdings = Column(Float)
    
    # Gold holdings
    gold_tonnes = Column(Float)  # Physical gold in tonnes
    gold_price_used = Column(Float)  # USD per troy ounce
    
    # Additional metrics
    reserves_to_short_term_debt = Column(Float)  # Ratio of reserves to short-term external debt
    reserves_to_broad_money = Column(Float)  # Ratio of reserves to broad money (M2)
    
    # Metadata
    is_provisional = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class InterventionType(enum.Enum):
    """Types of FX market interventions."""
    SPOT_PURCHASE = "spot_purchase"
    SPOT_SALE = "spot_sale"
    FORWARD_PURCHASE = "forward_purchase"
    FORWARD_SALE = "forward_sale"
    SWAP = "swap"
    VERBAL = "verbal"  # Verbal intervention without actual transactions


class FXIntervention(Base):
    """
    Foreign exchange market interventions by the central bank.
    
    Records instances when the central bank buys or sells foreign currency
    to influence the exchange rate.
    """
    __tablename__ = "fx_interventions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    intervention_type = Column(Enum(InterventionType), nullable=False, index=True)
    
    # Transaction details
    currency_pair = Column(String, nullable=False)  # e.g., "USD/BDT"
    amount = Column(Float, nullable=False)  # In millions of foreign currency
    exchange_rate = Column(Float)  # Rate at which intervention was conducted
    
    # Market context
    pre_intervention_rate = Column(Float)  # Market rate before intervention
    post_intervention_rate = Column(Float)  # Market rate after intervention
    daily_volatility_pre = Column(Float)  # Market volatility before intervention
    daily_volatility_post = Column(Float)  # Market volatility after intervention
    
    # Intervention effectiveness
    rate_impact_percent = Column(Float)  # Percentage change in exchange rate
    target_achieved = Column(Boolean)  # Whether the intervention achieved its objective
    
    # Policy context
    rationale = Column(Text)  # Reason for intervention
    public_announcement = Column(Boolean, default=False)  # Whether the intervention was announced
    announcement_text = Column(Text)  # Text of any public announcement
    
    # Metadata
    is_sterilized = Column(Boolean)  # Whether the intervention was sterilized
    sterilization_method = Column(String)  # How the intervention was sterilized
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BalanceOfPaymentsComponent(enum.Enum):
    """Major components of the balance of payments."""
    CURRENT_ACCOUNT = "current_account"
    CAPITAL_ACCOUNT = "capital_account"
    FINANCIAL_ACCOUNT = "financial_account"
    ERRORS_OMISSIONS = "errors_omissions"
    OVERALL_BALANCE = "overall_balance"


class BalanceOfPayments(Base):
    """
    Balance of payments data for the country.
    
    The BoP records all economic transactions between residents and non-residents.
    """
    __tablename__ = "balance_of_payments"

    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, nullable=False, index=True)
    quarter = Column(Integer, nullable=False, index=True)  # 1-4
    
    # Main components (in millions of USD)
    current_account_balance = Column(Float, nullable=False)
    capital_account_balance = Column(Float, nullable=False)
    financial_account_balance = Column(Float, nullable=False)
    errors_omissions = Column(Float)
    overall_balance = Column(Float, nullable=False)
    
    # Current account components
    goods_exports = Column(Float)
    goods_imports = Column(Float)
    trade_balance = Column(Float)
    services_exports = Column(Float)
    services_imports = Column(Float)
    services_balance = Column(Float)
    primary_income_balance = Column(Float)
    secondary_income_balance = Column(Float)
    remittances_inflow = Column(Float)  # Important for Bangladesh
    
    # Financial account components
    direct_investment_net = Column(Float)
    portfolio_investment_net = Column(Float)
    other_investment_net = Column(Float)
    reserve_assets_change = Column(Float)
    
    # Metadata
    is_provisional = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)
