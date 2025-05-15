"""
Monetary Policy Models.

This module defines SQLAlchemy models for monetary policy operations:
- Policy rates (repo, reverse repo, etc.)
- Open market operations (OMOs)
- Reserve requirements
- Liquidity facilities
- Standing facilities
"""

from sqlalchemy import Column, Integer, Float, String, Date, DateTime, ForeignKey, Text, Boolean, Enum
from sqlalchemy.orm import relationship
import enum
from datetime import date, datetime

from ..db.session import Base  # Corrected Base import


class PolicyRateType(enum.Enum):
    """Types of policy rates used by central banks."""
    REPO = "repo"  # Main policy rate / repurchase rate
    REVERSE_REPO = "reverse_repo"
    STANDING_DEPOSIT = "standing_deposit"
    STANDING_LENDING = "standing_lending"
    BANK_RATE = "bank_rate"
    DISCOUNT_RATE = "discount_rate"
    INTEREST_ON_RESERVES = "interest_on_reserves"


class PolicyRate(Base):
    """
    Policy interest rates set by the central bank.
    
    Tracks changes in policy rates over time, which are the primary 
    monetary policy tool for most central banks.
    """
    __tablename__ = "policy_rates"

    id = Column(Integer, primary_key=True, index=True)
    effective_date = Column(Date, nullable=False, index=True)
    rate_type = Column(Enum(PolicyRateType), nullable=False, index=True)
    
    # Rate value in percentage points
    rate_value = Column(Float, nullable=False)
    previous_value = Column(Float)
    change = Column(Float)  # Current - previous
    
    # Decision context
    decision_date = Column(Date, nullable=False)
    announcement_date = Column(Date, nullable=False)
    rationale = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    # related_statements = relationship("PolicyStatement", back_populates="rate_change")


class RepoOperationType(enum.Enum):
    """Types of repo operations."""
    MAIN_REFINANCING = "main_refinancing"
    OVERNIGHT = "overnight"
    ONE_WEEK = "one_week"
    TWO_WEEK = "two_week"
    ONE_MONTH = "one_month"
    THREE_MONTH = "three_month"
    SPECIAL = "special"


class RepoOperation(Base):
    """
    Repurchase agreement operations conducted by the central bank.
    
    Repos are a key tool for managing short-term liquidity in the banking system.
    """
    __tablename__ = "repo_operations"

    id = Column(Integer, primary_key=True, index=True)
    operation_date = Column(Date, nullable=False, index=True)
    settlement_date = Column(Date, nullable=False)
    maturity_date = Column(Date, nullable=False)
    
    operation_type = Column(Enum(RepoOperationType), nullable=False, index=True)
    is_reverse = Column(Boolean, default=False)  # True for reverse repo
    
    # Operation details
    announced_amount = Column(Float)  # In local currency (e.g., millions of BDT)
    bid_amount = Column(Float)  # Total bids received
    allotted_amount = Column(Float, nullable=False)  # Actual amount provided
    
    # Rate information
    min_rate = Column(Float)  # Minimum bid rate
    max_rate = Column(Float)  # Maximum bid rate
    marginal_rate = Column(Float, nullable=False)  # Rate at which bids were cut off
    weighted_avg_rate = Column(Float, nullable=False)  # Weighted average of accepted bids
    
    # Operation characteristics
    is_fixed_rate = Column(Boolean, default=False)  # Fixed rate vs variable rate tender
    fixed_rate = Column(Float)  # Only for fixed rate operations
    
    # Counterparties
    number_of_bidders = Column(Integer)
    number_of_accepted_bidders = Column(Integer)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)


class OMOType(enum.Enum):
    """Types of open market operations."""
    OUTRIGHT_PURCHASE = "outright_purchase"
    OUTRIGHT_SALE = "outright_sale"
    SECURITY_LENDING = "security_lending"


class SecurityType(enum.Enum):
    """Types of securities used in open market operations."""
    TREASURY_BILL = "treasury_bill"
    TREASURY_BOND = "treasury_bond"
    CENTRAL_BANK_BILL = "central_bank_bill"
    GOVERNMENT_BOND = "government_bond"
    CORPORATE_BOND = "corporate_bond"
    OTHER = "other"


class OpenMarketOperation(Base):
    """
    Open market operations conducted by the central bank.
    
    OMOs involve the buying and selling of securities to influence 
    money supply and interest rates.
    """
    __tablename__ = "open_market_operations"

    id = Column(Integer, primary_key=True, index=True)
    operation_date = Column(Date, nullable=False, index=True)
    settlement_date = Column(Date, nullable=False)
    
    operation_type = Column(Enum(OMOType), nullable=False, index=True)
    security_type = Column(Enum(SecurityType), nullable=False)
    
    # Operation details
    announced_amount = Column(Float)  # In local currency (e.g., millions of BDT)
    executed_amount = Column(Float, nullable=False)  # Actual amount executed
    
    # Security details
    maturity_range_min = Column(Integer)  # In days
    maturity_range_max = Column(Integer)  # In days
    average_yield = Column(Float)  # Percentage
    
    # Market impact
    pre_operation_yield = Column(Float)  # Market yield before operation
    post_operation_yield = Column(Float)  # Market yield after operation
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)


class ReserveRequirementType(enum.Enum):
    """Types of reserve requirements."""
    CASH_RESERVE_RATIO = "crr"  # Cash reserve ratio
    STATUTORY_LIQUIDITY_RATIO = "slr"  # Statutory liquidity ratio
    MINIMUM_RESERVE_REQUIREMENT = "mrr"  # Minimum reserve requirement


class ReserveRequirement(Base):
    """
    Reserve requirements imposed by the central bank on commercial banks.
    
    These requirements specify the minimum reserves banks must hold against their deposits.
    """
    __tablename__ = "reserve_requirements"

    id = Column(Integer, primary_key=True, index=True)
    effective_date = Column(Date, nullable=False, index=True)
    requirement_type = Column(Enum(ReserveRequirementType), nullable=False, index=True)
    
    # Requirement details
    requirement_ratio = Column(Float, nullable=False)  # Percentage
    previous_ratio = Column(Float)
    change = Column(Float)  # Current - previous
    
    # Implementation details
    announcement_date = Column(Date, nullable=False)
    compliance_start_date = Column(Date, nullable=False)  # When banks must start complying
    phase_in_period = Column(Integer)  # In days, if gradual implementation
    
    # Applicability
    applies_to_all_banks = Column(Boolean, default=True)
    bank_type_exceptions = Column(String)  # e.g., "Islamic banks, Development banks"
    
    # Calculation basis
    calculation_base = Column(String)  # e.g., "Total deposits", "Demand deposits only"
    averaging_period = Column(Integer)  # In days, for averaging compliance
    
    # Metadata
    rationale = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)


class StandingFacilityOperation(Base):
    """
    Standing facility operations used by banks to deposit or borrow from the central bank.
    
    These facilities provide an interest rate corridor and help manage short-term liquidity.
    """
    __tablename__ = "standing_facility_operations"

    id = Column(Integer, primary_key=True, index=True)
    operation_date = Column(Date, nullable=False, index=True)
    is_lending = Column(Boolean, nullable=False)  # True for lending, False for deposit
    
    # Operation details
    amount = Column(Float, nullable=False)  # In local currency
    rate = Column(Float, nullable=False)  # Interest rate applied
    tenor = Column(Integer, nullable=False)  # In days
    
    # Bank details (could be anonymized for confidentiality)
    bank_id = Column(Integer, ForeignKey("banks.id"), nullable=True)
    bank = relationship("Bank", back_populates="standing_facility_operations")
    
    # Collateral for lending operations
    collateral_type = Column(String)
    collateral_value = Column(Float)
    haircut_applied = Column(Float)  # Percentage haircut on collateral
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
