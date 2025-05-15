"""
Economic Indicators Models.

This module defines SQLAlchemy models for storing macroeconomic indicators
that are crucial for central bank policy decisions:
- GDP and its components
- Consumer Price Index (CPI) and inflation
- Unemployment rates
- Other key economic indicators
"""

from sqlalchemy import Column, Integer, Float, String, Date, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import date as date_type

from ..db.session import Base  # Corrected Base import


class GDP(Base):
    """
    Gross Domestic Product data, stored quarterly.
    
    Includes both aggregate GDP and its components (consumption, investment, etc.)
    """
    __tablename__ = "gdp"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True, unique=True)
    year = Column(Integer, nullable=False, index=True)
    quarter = Column(Integer, nullable=False, index=True)  # 1-4
    
    # Values in local currency (e.g., billions of BDT)
    nominal_gdp = Column(Float, nullable=False)
    real_gdp = Column(Float, nullable=False)
    
    # GDP components
    consumption = Column(Float)
    government_spending = Column(Float)
    investment = Column(Float)
    net_exports = Column(Float)
    
    # Growth rates (percentage)
    nominal_growth_yoy = Column(Float)  # Year-over-year
    real_growth_yoy = Column(Float)
    real_growth_qoq = Column(Float)  # Quarter-over-quarter, annualized
    
    # Metadata
    is_provisional = Column(Boolean, default=False)
    last_updated = Column(Date, default=date_type.today)
    notes = Column(Text)
    
    # Relationships
    # gdp_forecasts = relationship("GDPForecast", back_populates="actual_gdp")


class CPI(Base):
    """
    Consumer Price Index data, stored monthly.
    
    Includes both headline and core inflation, as well as major CPI components.
    """
    __tablename__ = "cpi"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True, unique=True)
    year = Column(Integer, nullable=False, index=True)
    month = Column(Integer, nullable=False, index=True)  # 1-12
    
    # Index values (base year = 100)
    headline_index = Column(Float, nullable=False)
    core_index = Column(Float)  # Excluding food and energy
    food_index = Column(Float)
    energy_index = Column(Float)
    housing_index = Column(Float)
    healthcare_index = Column(Float)
    transport_index = Column(Float)
    education_index = Column(Float)
    
    # Inflation rates (percentage)
    headline_inflation_yoy = Column(Float)  # Year-over-year
    core_inflation_yoy = Column(Float)
    headline_inflation_mom = Column(Float)  # Month-over-month
    core_inflation_mom = Column(Float)
    
    # Metadata
    is_provisional = Column(Boolean, default=False)
    last_updated = Column(Date, default=date_type.today)
    notes = Column(Text)
    
    # Relationships
    # inflation_forecasts = relationship("InflationForecast", back_populates="actual_inflation")


class UnemploymentRate(Base):
    """
    Unemployment rate data, stored monthly.
    
    Includes both aggregate unemployment and breakdowns by demographic groups.
    """
    __tablename__ = "unemployment_rate"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True, unique=True)
    year = Column(Integer, nullable=False, index=True)
    month = Column(Integer, nullable=False, index=True)  # 1-12
    
    # Rates as percentages
    overall_rate = Column(Float, nullable=False)
    urban_rate = Column(Float)
    rural_rate = Column(Float)
    
    # Demographic breakdowns
    male_rate = Column(Float)
    female_rate = Column(Float)
    youth_rate = Column(Float)  # e.g., ages 15-24
    
    # Additional metrics
    labor_force_participation = Column(Float)  # Percentage
    underemployment_rate = Column(Float)  # Percentage
    
    # Metadata
    is_provisional = Column(Boolean, default=False)
    last_updated = Column(Date, default=date_type.today)
    notes = Column(Text)


class TradeBalance(Base):
    """
    Trade balance data, stored monthly.
    
    Includes imports, exports, and trade balance figures.
    """
    __tablename__ = "trade_balance"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True, unique=True)
    year = Column(Integer, nullable=False, index=True)
    month = Column(Integer, nullable=False, index=True)  # 1-12
    
    # Values in local currency (e.g., billions of BDT)
    exports = Column(Float, nullable=False)
    imports = Column(Float, nullable=False)
    trade_balance = Column(Float, nullable=False)  # exports - imports
    
    # Key sector breakdowns for Bangladesh
    rmg_exports = Column(Float)  # Ready-made garments
    textile_exports = Column(Float)
    agricultural_exports = Column(Float)
    
    oil_imports = Column(Float)
    food_imports = Column(Float)
    capital_goods_imports = Column(Float)
    
    # Year-over-year growth rates
    exports_growth_yoy = Column(Float)
    imports_growth_yoy = Column(Float)
    
    # Metadata
    is_provisional = Column(Boolean, default=False)
    last_updated = Column(Date, default=date_type.today)
    notes = Column(Text)
