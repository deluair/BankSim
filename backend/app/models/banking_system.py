"""
Banking System Models.

This module defines SQLAlchemy models for the banking system:
- Banks and financial institutions
- Balance sheets and financial statements
- Capital adequacy and liquidity metrics
- CAMELS ratings and supervision data
"""

from sqlalchemy import Column, Integer, Float, String, Date, DateTime, ForeignKey, Text, Boolean, Enum
from sqlalchemy.orm import relationship
import enum
from datetime import date, datetime

from ..db.session import Base  # Corrected Base import


class BankType(enum.Enum):
    """Types of banks and financial institutions."""
    COMMERCIAL = "commercial"
    SPECIALIZED = "specialized"
    ISLAMIC = "islamic"
    FOREIGN = "foreign"
    DEVELOPMENT = "development"
    COOPERATIVE = "cooperative"
    MICROFINANCE = "microfinance"
    STATE_OWNED = "state_owned"


class Bank(Base):
    """
    Banks and financial institutions supervised by the central bank.
    """
    __tablename__ = "banks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    short_name = Column(String, nullable=False, index=True)
    registration_number = Column(String, nullable=False, unique=True)
    
    # Classification
    bank_type = Column(Enum(BankType), nullable=False, index=True)
    is_active = Column(Boolean, default=True)
    is_systemically_important = Column(Boolean, default=False)
    
    # Establishment info
    established_date = Column(Date, nullable=False)
    license_issue_date = Column(Date, nullable=False)
    
    # Size and scale
    total_branches = Column(Integer)
    total_atms = Column(Integer)
    total_employees = Column(Integer)
    
    # Contact and location
    headquarters_address = Column(String)
    website = Column(String)
    swift_code = Column(String)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    balances = relationship("BankBalance", back_populates="bank")
    loan_portfolios = relationship("LoanPortfolio", back_populates="bank")
    camels_ratings = relationship("CAMELSRating", back_populates="bank")
    standing_facility_operations = relationship("StandingFacilityOperation", back_populates="bank")


class BankBalance(Base):
    """
    Bank balance sheet data reported to the central bank.
    
    Includes major asset and liability categories, updated periodically.
    """
    __tablename__ = "bank_balances"

    id = Column(Integer, primary_key=True, index=True)
    bank_id = Column(Integer, ForeignKey("banks.id"), nullable=False, index=True)
    report_date = Column(Date, nullable=False, index=True)
    
    # Assets (in local currency, e.g., millions of BDT)
    total_assets = Column(Float, nullable=False)
    cash_and_equivalents = Column(Float, nullable=False)  # Cash and balances with central bank
    interbank_assets = Column(Float)  # Claims on other banks
    government_securities = Column(Float)  # Treasury bills, bonds, etc.
    loans_and_advances = Column(Float, nullable=False)  # Total loans to customers
    investments = Column(Float)  # Other investments
    fixed_assets = Column(Float)  # Property, plant, equipment
    other_assets = Column(Float)
    
    # Liabilities
    total_liabilities = Column(Float, nullable=False)
    deposits = Column(Float, nullable=False)  # Total customer deposits
    demand_deposits = Column(Float)  # Current accounts
    savings_deposits = Column(Float)  # Savings accounts
    term_deposits = Column(Float)  # Fixed deposits
    interbank_liabilities = Column(Float)  # Borrowings from other banks
    central_bank_borrowings = Column(Float)  # Borrowings from central bank
    other_borrowings = Column(Float)  # Other debt
    other_liabilities = Column(Float)
    
    # Equity
    total_equity = Column(Float, nullable=False)
    paid_up_capital = Column(Float)
    reserves = Column(Float)
    retained_earnings = Column(Float)
    
    # Metadata
    is_audited = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    bank = relationship("Bank", back_populates="balances")


class LoanClassification(enum.Enum):
    """Loan classification categories for asset quality assessment."""
    STANDARD = "standard"
    SPECIAL_MENTION = "special_mention"
    SUBSTANDARD = "substandard"
    DOUBTFUL = "doubtful"
    LOSS = "loss"


class LoanPortfolio(Base):
    """
    Bank loan portfolio data reported to the central bank.
    
    Includes loan classifications, sectoral breakdown, and non-performing loans.
    """
    __tablename__ = "loan_portfolios"

    id = Column(Integer, primary_key=True, index=True)
    bank_id = Column(Integer, ForeignKey("banks.id"), nullable=False, index=True)
    report_date = Column(Date, nullable=False, index=True)
    
    # Total loans (in local currency, e.g., millions of BDT)
    total_loans = Column(Float, nullable=False)
    
    # Loan classification
    standard_loans = Column(Float)
    special_mention_loans = Column(Float)
    substandard_loans = Column(Float)
    doubtful_loans = Column(Float)
    loss_loans = Column(Float)
    
    # Non-performing loans
    npl_amount = Column(Float)  # Substandard + Doubtful + Loss
    npl_ratio = Column(Float)  # NPL / Total Loans (percentage)
    
    # Provisions
    required_provisions = Column(Float)
    actual_provisions = Column(Float)
    provision_coverage_ratio = Column(Float)  # Actual Provisions / NPL (percentage)
    
    # Sectoral breakdown
    agriculture_loans = Column(Float)
    industry_loans = Column(Float)
    service_loans = Column(Float)
    trade_loans = Column(Float)
    consumer_loans = Column(Float)
    real_estate_loans = Column(Float)
    other_loans = Column(Float)
    
    # Bangladesh-specific sectors
    rmg_loans = Column(Float)  # Ready-made garments
    textile_loans = Column(Float)
    sme_loans = Column(Float)  # Small and medium enterprises
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    bank = relationship("Bank", back_populates="loan_portfolios")


class CAMELSComponent(enum.Enum):
    """Components of the CAMELS rating system."""
    CAPITAL = "capital"
    ASSET = "asset"
    MANAGEMENT = "management"
    EARNINGS = "earnings"
    LIQUIDITY = "liquidity"
    SENSITIVITY = "sensitivity"


class CAMELSRating(Base):
    """
    CAMELS ratings assigned to banks by the central bank.
    
    CAMELS is a supervisory rating system for evaluating bank health.
    """
    __tablename__ = "camels_ratings"

    id = Column(Integer, primary_key=True, index=True)
    bank_id = Column(Integer, ForeignKey("banks.id"), nullable=False, index=True)
    rating_date = Column(Date, nullable=False, index=True)
    
    # Individual component ratings (1-5 scale, 1 is best)
    capital_rating = Column(Integer, nullable=False)
    asset_rating = Column(Integer, nullable=False)
    management_rating = Column(Integer, nullable=False)
    earnings_rating = Column(Integer, nullable=False)
    liquidity_rating = Column(Integer, nullable=False)
    sensitivity_rating = Column(Integer, nullable=False)
    
    # Composite rating (1-5 scale, 1 is best)
    composite_rating = Column(Integer, nullable=False)
    
    # Examination details
    examination_start_date = Column(Date)
    examination_end_date = Column(Date)
    is_onsite = Column(Boolean, default=True)
    examination_team = Column(String)
    
    # Key metrics used in assessment
    capital_adequacy_ratio = Column(Float)
    tier1_capital_ratio = Column(Float)
    npl_ratio = Column(Float)
    provision_coverage_ratio = Column(Float)
    return_on_assets = Column(Float)
    return_on_equity = Column(Float)
    cost_income_ratio = Column(Float)
    liquidity_coverage_ratio = Column(Float)
    net_stable_funding_ratio = Column(Float)
    
    # Supervisory actions
    supervisory_concerns = Column(Text)
    required_actions = Column(Text)
    compliance_deadline = Column(Date)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    bank = relationship("Bank", back_populates="camels_ratings")


class BankingSystemAggregate(Base):
    """
    Aggregate data for the entire banking system.
    
    Provides a system-wide view of key banking indicators.
    """
    __tablename__ = "banking_system_aggregates"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True, unique=True)
    
    # System totals (in local currency, e.g., billions of BDT)
    total_assets = Column(Float, nullable=False)
    total_loans = Column(Float, nullable=False)
    total_deposits = Column(Float, nullable=False)
    total_capital = Column(Float, nullable=False)
    
    # System-wide metrics
    average_car = Column(Float)  # Capital Adequacy Ratio
    system_npl_ratio = Column(Float)  # Non-Performing Loan Ratio
    system_provision_coverage = Column(Float)
    system_return_on_assets = Column(Float)
    system_return_on_equity = Column(Float)
    system_liquidity_ratio = Column(Float)
    
    # Interbank market
    interbank_volume = Column(Float)  # Total interbank lending volume
    interbank_rate = Column(Float)  # Average interbank lending rate
    
    # Concentration metrics
    hhi_assets = Column(Float)  # Herfindahl-Hirschman Index for assets
    hhi_deposits = Column(Float)  # Herfindahl-Hirschman Index for deposits
    top5_banks_asset_share = Column(Float)  # Percentage of assets held by top 5 banks
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
