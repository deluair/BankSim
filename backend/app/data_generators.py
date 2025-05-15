"""
Data Generator Module for BankSim.

This module provides functions to generate realistic sample data
for the BankSim database, enabling testing and demonstrations.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session
import random
import logging

from . import models
from .models.economic_indicators import GDP, CPI, UnemploymentRate, TradeBalance
from .models.monetary_policy import (
    PolicyRate, PolicyRateType, RepoOperation, RepoOperationType,
    OpenMarketOperation, OMOType, SecurityType, ReserveRequirement, ReserveRequirementType
)
from .models.banking_system import (
    Bank, BankType, BankBalance, LoanPortfolio, LoanClassification, 
    CAMELSRating, CAMELSComponent, BankingSystemAggregate
)
from .models.foreign_exchange import (
    ExchangeRate, EffectiveExchangeRate, ForeignReserve, ReserveAssetType,
    FXIntervention, InterventionType, BalanceOfPayments, BalanceOfPaymentsComponent
)

logger = logging.getLogger(__name__)


def generate_all_data(db: Session, start_date: date, end_date: date) -> None:
    """
    Generate all sample data for the BankSim database.
    
    Args:
        db: Database session
        start_date: Start date for historical data
        end_date: End date for historical data
    """
    logger.info(f"Generating sample data from {start_date} to {end_date}")
    
    # Generate data in a logical order (dependencies first)
    generate_banks(db)
    
    generate_gdp_data(db, start_date, end_date)
    generate_cpi_data(db, start_date, end_date)
    generate_unemployment_data(db, start_date, end_date)
    generate_trade_data(db, start_date, end_date)
    
    generate_policy_rates(db, start_date, end_date)
    generate_repo_operations(db, start_date, end_date)
    generate_open_market_operations(db, start_date, end_date)
    generate_reserve_requirements(db, start_date, end_date)
    
    generate_bank_balances(db, start_date, end_date)
    generate_loan_portfolios(db, start_date, end_date)
    generate_camels_ratings(db, start_date, end_date)
    generate_banking_system_aggregates(db, start_date, end_date)
    
    generate_exchange_rates(db, start_date, end_date)
    generate_effective_exchange_rates(db, start_date, end_date)
    generate_foreign_reserves(db, start_date, end_date)
    generate_fx_interventions(db, start_date, end_date)
    generate_balance_of_payments(db, start_date, end_date)
    
    db.commit()
    logger.info("Sample data generation completed")


def generate_banks(db: Session, num_banks: int = 40) -> List[Bank]:
    """Generate sample bank data."""
    logger.info(f"Generating {num_banks} banks")
    
    bank_types = [
        (BankType.COMMERCIAL, 15),
        (BankType.SPECIALIZED, 5),
        (BankType.ISLAMIC, 8),
        (BankType.FOREIGN, 7),
        (BankType.DEVELOPMENT, 3),
        (BankType.STATE_OWNED, 2)
    ]
    
    # Sample bank names
    commercial_names = [
        "City Bank", "Metro Bank", "United Bank", "Trust Bank", "Provincial Bank",
        "National Bank", "Community Bank", "Citizen's Bank", "Capital Bank", "Heritage Bank",
        "Mercantile Bank", "Commerce Bank", "Standard Bank", "Prime Bank", "Eastern Bank"
    ]
    
    islamic_names = [
        "Al-Amanah Islamic Bank", "Al-Baraka Islamic Bank", "Takaful Bank", "Shariah Bank",
        "Al-Falah Islamic Bank", "Amanah Trust Bank", "Barakah Finance", "Tawhid Bank",
        "Islam Trust Bank", "Mudarabah Bank"
    ]
    
    specialized_names = [
        "Agricultural Development Bank", "Housing Finance Bank", "SME Development Bank",
        "Industrial Credit Bank", "Export Finance Bank", "Microfinance Development Bank",
        "Rural Development Bank", "Infrastructure Finance Bank"
    ]
    
    foreign_names = [
        "HSBC", "Citibank", "Standard Chartered", "Deutsche Bank", "Bank of Tokyo",
        "ANZ Banking Group", "Bank of America", "BNP Paribas", "DBS Bank", "UBS"
    ]
    
    development_names = [
        "National Development Bank", "Rural Investment Bank", "Infrastructure Development Bank",
        "Industrial Development Corp", "Export Development Bank"
    ]
    
    state_owned_names = [
        "State Bank of Bangladesh", "People's Bank", "Government Savings Bank",
        "National Savings Bank", "Central Investment Bank"
    ]
    
    # Create banks
    banks = []
    bank_count = 0
    
    for bank_type, count in bank_types:
        # Choose appropriate name list based on bank type
        if bank_type == BankType.COMMERCIAL:
            names = commercial_names
        elif bank_type == BankType.ISLAMIC:
            names = islamic_names
        elif bank_type == BankType.SPECIALIZED:
            names = specialized_names
        elif bank_type == BankType.FOREIGN:
            names = foreign_names
        elif bank_type == BankType.DEVELOPMENT:
            names = development_names
        elif bank_type == BankType.STATE_OWNED:
            names = state_owned_names
        else:
            names = commercial_names
        
        # Create the specified number of banks of this type
        for i in range(min(count, len(names))):
            name = names[i]
            if not name:  # Check if name from list is empty
                logger.warning(f"Bank name from list is empty for type {bank_type}, index {i} in name list. Skipping this bank entry.")
                continue # Skip this bank entry

            short_name = ''.join([w[0] for w in name.split() if w[0].isupper()])
            if not short_name:
                short_name = name[:3].upper()
            
            if not short_name: # If still empty (e.g. name was problematic like <3 non-alpha chars)
                logger.warning(f"Generated short_name is empty for name '{name}'. Using 'DFLT_SN'.")
                short_name = "DFLT_SN" # Default short name to ensure it's not empty
            
            # Generate random establishment date
            est_year = random.randint(1972, 2015)
            est_month = random.randint(1, 12)
            est_day = random.randint(1, 28)
            established_date = date(est_year, est_month, est_day)
            
            # License usually issued a bit after establishment
            license_date = established_date + timedelta(days=random.randint(30, 180))
            
            # Other properties
            total_branches = random.randint(5, 200)
            total_atms = total_branches * random.randint(1, 3)
            total_employees = total_branches * random.randint(8, 25)
            
            # Systemically important if large
            is_systemically_important = (total_branches > 100)
            
            bank = Bank(
                name=name,
                short_name=short_name,
                registration_number=f"REG-{bank_count}-{random.randint(10000, 99999)}-{est_year}",
                bank_type=bank_type,
                is_active=True,
                is_systemically_important=is_systemically_important,
                established_date=established_date,
                license_issue_date=license_date,
                total_branches=total_branches,
                total_atms=total_atms,
                total_employees=total_employees,
                headquarters_address=f"{random.randint(1, 100)} Main Street, Dhaka",
                website=f"https://www.{name.lower().replace(' ', '').replace('.', '')}.com.bd",
                swift_code=f"{short_name.replace(' ', '')[:4].upper()}BDDH{random.randint(10,99)}"
            )
            
            db.add(bank)
            banks.append(bank)
            bank_count += 1
            
            if bank_count >= num_banks:
                break
        
        if bank_count >= num_banks:
            break
    
    db.flush()  # Ensure banks have IDs assigned
    logger.info(f"Generated {len(banks)} banks")
    return banks


def generate_gdp_data(db: Session, start_date: date, end_date: date) -> None:
    """Generate quarterly GDP data."""
    logger.info("Generating GDP data")
    
    # Parameters
    base_gdp = 12000.0  # Billions BDT
    trend_growth_rate = 0.06  # 6% annual growth trend
    seasonal_factors = [0.95, 1.02, 1.05, 0.98]  # Quarterly seasonality
    
    # Generate quarterly dates
    current_date = date(start_date.year, ((start_date.month-1)//3)*3+1, 1)  # Start at quarter boundary
    
    quarter_count = 0
    while current_date <= end_date:
        year = current_date.year
        quarter = (current_date.month - 1) // 3 + 1
        
        # Calculate trend growth
        time_years = quarter_count / 4  # Convert quarters to years
        trend_factor = (1 + trend_growth_rate) ** time_years
        
        # Add cyclical component (5-year business cycle)
        cycle_phase = (quarter_count % 20) / 20  # 0 to 1 over 5 years (20 quarters)
        cycle_factor = 1 + 0.04 * np.sin(2 * np.pi * cycle_phase)  # +/- 4% cycle
        
        # Add random component
        random_factor = 1 + np.random.normal(0, 0.01)  # +/- 1% random noise
        
        # Calculate GDP with seasonal adjustment
        seasonal_factor = seasonal_factors[quarter - 1]
        nominal_gdp = base_gdp * trend_factor * cycle_factor * seasonal_factor * random_factor
        
        # Real GDP (adjusted for inflation, assuming 5% annual inflation)
        deflator = (1.05) ** time_years
        real_gdp = nominal_gdp / deflator
        
        # Calculate growth rates
        if quarter_count >= 4:  # After first year, calculate YoY growth
            yoy_nominal_growth = (nominal_gdp / (base_gdp * (1 + trend_growth_rate) ** (time_years - 1) * 
                                             seasonal_factor * cycle_factor * random_factor) - 1) * 100
            yoy_real_growth = (real_gdp / (base_gdp * (1 + trend_growth_rate) ** (time_years - 1) / 
                                        (1.05) ** (time_years - 1) * seasonal_factor * cycle_factor * random_factor) - 1) * 100
        else:
            yoy_nominal_growth = trend_growth_rate * 100
            yoy_real_growth = (trend_growth_rate - 0.05) * 100  # Nominal growth minus inflation
        
        # Calculate QoQ growth (annualized)
        if quarter_count >= 1:
            last_quarter_gdp = base_gdp * (1 + trend_growth_rate) ** ((quarter_count - 1) / 4) * \
                              seasonal_factors[(quarter - 2) % 4 + 1] * \
                              (1 + 0.04 * np.sin(2 * np.pi * ((quarter_count - 1) % 20) / 20)) * \
                              (1 + np.random.normal(0, 0.01))
            qoq_growth = ((real_gdp / last_quarter_gdp) ** 4 - 1) * 100  # Annualized
        else:
            qoq_growth = (trend_growth_rate - 0.05) * 100
        
        # GDP components
        consumption = nominal_gdp * random.uniform(0.55, 0.65)
        government = nominal_gdp * random.uniform(0.15, 0.20)
        investment = nominal_gdp * random.uniform(0.15, 0.25)
        net_exports = nominal_gdp - consumption - government - investment
        
        # Create GDP record
        gdp_record = GDP(
            date=current_date,
            year=year,
            quarter=quarter,
            nominal_gdp=round(nominal_gdp, 2),
            real_gdp=round(real_gdp, 2),
            consumption=round(consumption, 2),
            government_spending=round(government, 2),
            investment=round(investment, 2),
            net_exports=round(net_exports, 2),
            nominal_growth_yoy=round(yoy_nominal_growth, 2),
            real_growth_yoy=round(yoy_real_growth, 2),
            real_growth_qoq=round(qoq_growth, 2),
            is_provisional=(current_date > date.today() - timedelta(days=180)),
            last_updated=date.today(),
            notes="Generated sample data"
        )
        
        db.add(gdp_record)
        
        # Move to next quarter
        quarter_count += 1
        current_date = date(year + (quarter) // 4, ((quarter) % 4) * 3 + 1, 1)
    
    db.flush()
    logger.info(f"Generated {quarter_count} quarters of GDP data")


def generate_cpi_data(db: Session, start_date: date, end_date: date) -> None:
    """Generate monthly CPI data."""
    logger.info("Generating CPI data")
    
    # Parameters
    base_index = 100.0
    trend_inflation = 0.05  # 5% annual trend inflation
    food_weight = 0.35
    energy_weight = 0.15
    housing_weight = 0.20
    healthcare_weight = 0.10
    transport_weight = 0.10
    education_weight = 0.05
    other_weight = 0.05
    
    # Generate monthly dates
    current_date = date(start_date.year, start_date.month, 1)  # Start at month boundary
    
    month_count = 0
    prev_headline_index = None
    prev_core_index = None
    
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        
        # Calculate trend inflation (compounding monthly)
        time_years = month_count / 12  # Convert months to years
        monthly_inflation = (1 + trend_inflation) ** (1/12) - 1
        trend_factor = (1 + monthly_inflation) ** month_count
        
        # Add cyclical component (3-year inflation cycle)
        cycle_phase = (month_count % 36) / 36  # 0 to 1 over 3 years
        cycle_factor = 1 + 0.02 * np.sin(2 * np.pi * cycle_phase)  # +/- 2% cycle
        
        # Seasonal factors (food prices higher in certain months)
        month_seasonal = [1.01, 1.00, 0.99, 0.98, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.02]
        seasonal_factor = month_seasonal[month - 1]
        
        # Random component
        random_factor = 1 + np.random.normal(0, 0.003)  # +/- 0.3% monthly noise
        
        # Calculate CPI components with different volatility
        food_volatility = np.random.normal(0, 0.01)  # More volatile
        energy_volatility = np.random.normal(0, 0.015)  # Most volatile
        housing_volatility = np.random.normal(0, 0.003)  # Less volatile
        healthcare_volatility = np.random.normal(0, 0.002)  # Stable
        transport_volatility = np.random.normal(0, 0.007)  # Moderately volatile
        education_volatility = np.random.normal(0, 0.001)  # Very stable
        other_volatility = np.random.normal(0, 0.005)  # Moderate
        
        # Calculate component indices
        food_index = base_index * trend_factor * (seasonal_factor * 1.05) * (1 + food_volatility)
        energy_index = base_index * trend_factor * (1 + energy_volatility)
        housing_index = base_index * trend_factor * (1 + housing_volatility)
        healthcare_index = base_index * trend_factor * (1 + healthcare_volatility)
        transport_index = base_index * trend_factor * (1 + transport_volatility)
        education_index = base_index * trend_factor * (1 + education_volatility)
        other_index = base_index * trend_factor * (1 + other_volatility)
        
        # Calculate headline and core inflation
        headline_index = (food_weight * food_index + 
                         energy_weight * energy_index + 
                         housing_weight * housing_index + 
                         healthcare_weight * healthcare_index + 
                         transport_weight * transport_index + 
                         education_weight * education_index +
                         other_weight * other_index)
        
        # Core inflation excludes food and energy
        core_weight_sum = housing_weight + healthcare_weight + transport_weight + education_weight + other_weight
        core_index = ((housing_weight * housing_index + 
                      healthcare_weight * healthcare_index + 
                      transport_weight * transport_index + 
                      education_weight * education_index +
                      other_weight * other_index) / core_weight_sum)
        
        # Calculate inflation rates
        if prev_headline_index is not None:
            headline_inflation_mom = (headline_index / prev_headline_index - 1) * 100
            core_inflation_mom = (core_index / prev_core_index - 1) * 100
        else:
            headline_inflation_mom = monthly_inflation * 100
            core_inflation_mom = monthly_inflation * 100 * 0.8  # Core inflation usually less volatile
        
        if month_count >= 12:  # After first year, calculate YoY inflation
            yoy_headline_index = base_index * (1 + monthly_inflation) ** (month_count - 12) * cycle_factor * seasonal_factor * random_factor
            yoy_core_index = base_index * (1 + monthly_inflation) ** (month_count - 12) * cycle_factor * random_factor
            headline_inflation_yoy = (headline_index / yoy_headline_index - 1) * 100
            core_inflation_yoy = (core_index / yoy_core_index - 1) * 100
        else:
            headline_inflation_yoy = trend_inflation * 100
            core_inflation_yoy = trend_inflation * 100 * 0.8
        
        # Create CPI record
        cpi_record = CPI(
            date=current_date,
            year=year,
            month=month,
            headline_index=round(headline_index, 2),
            core_index=round(core_index, 2),
            food_index=round(food_index, 2),
            energy_index=round(energy_index, 2),
            housing_index=round(housing_index, 2),
            healthcare_index=round(healthcare_index, 2),
            transport_index=round(transport_index, 2),
            education_index=round(education_index, 2),
            headline_inflation_yoy=round(headline_inflation_yoy, 2),
            core_inflation_yoy=round(core_inflation_yoy, 2),
            headline_inflation_mom=round(headline_inflation_mom, 2),
            core_inflation_mom=round(core_inflation_mom, 2),
            is_provisional=(current_date > date.today() - timedelta(days=60)),
            last_updated=date.today(),
            notes="Generated sample data"
        )
        
        db.add(cpi_record)
        
        # Save for next iteration
        prev_headline_index = headline_index
        prev_core_index = core_index
        
        # Move to next month
        month_count += 1
        if current_date.month == 12:
            current_date = date(current_date.year + 1, 1, 1)
        else:
            current_date = date(current_date.year, current_date.month + 1, 1)
    
    db.flush()
    logger.info(f"Generated {month_count} months of CPI data")


def generate_policy_rates(db: Session, start_date: date, end_date: date) -> None:
    """Generate policy rate data."""
    logger.info("Generating policy rate data")
    
    # Parameters for different policy rates
    rate_params = {
        PolicyRateType.REPO: {
            "initial_rate": 5.0,
            "min_rate": 1.0,
            "max_rate": 12.0,
            "std_dev": 0.25,  # Standard deviation of changes
            "mean_months_between_changes": 2
        },
        PolicyRateType.REVERSE_REPO: {
            "initial_rate": 3.0,
            "min_rate": 0.5,
            "max_rate": 10.0,
            "std_dev": 0.25,
            "mean_months_between_changes": 2
        },
        PolicyRateType.STANDING_DEPOSIT: {
            "initial_rate": 2.0,
            "min_rate": 0.0,
            "max_rate": 9.0,
            "std_dev": 0.25,
            "mean_months_between_changes": 2
        },
        PolicyRateType.STANDING_LENDING: {
            "initial_rate": 8.0,
            "min_rate": 2.0,
            "max_rate": 15.0,
            "std_dev": 0.25,
            "mean_months_between_changes": 2
        },
    }
    
    # State variables
    current_rates = {rate_type: params["initial_rate"] for rate_type, params in rate_params.items()}
    last_change_date = {rate_type: start_date - timedelta(days=30) for rate_type in rate_params}
    
    # Monetary policy cycle: 0 = neutral, 1 = tightening, -1 = easing
    policy_cycle = 0
    cycle_duration_months = random.randint(18, 30)
    months_in_current_cycle = 0
    
    # Process months
    current_date = date(start_date.year, start_date.month, 1)
    month_count = 0
    
    while current_date <= end_date:
        month_count += 1
        
        # Update policy cycle state
        months_in_current_cycle += 1
        if months_in_current_cycle >= cycle_duration_months:
            # Change policy direction
            months_in_current_cycle = 0
            cycle_duration_months = random.randint(18, 30)
            
            if policy_cycle == 0:
                # From neutral to either tightening or easing
                policy_cycle = 1 if random.random() > 0.5 else -1
            elif policy_cycle == 1:
                # From tightening to neutral or easing
                policy_cycle = 0 if random.random() > 0.7 else -1
            else:  # policy_cycle == -1
                # From easing to neutral or tightening
                policy_cycle = 0 if random.random() > 0.7 else 1
        
        # Check for rate changes for each policy rate
        for rate_type, params in rate_params.items():
            months_since_change = (current_date.year - last_change_date[rate_type].year) * 12 + (current_date.month - last_change_date[rate_type].month)
            
            # Probability of change increases with time since last change
            change_probability = 1 - np.exp(-months_since_change / params["mean_months_between_changes"])
            change_probability *= (1 + abs(policy_cycle) * 0.5)  # Higher probability during active cycles
            
            if random.random() < change_probability:
                # Determine direction and magnitude of change
                direction = policy_cycle if policy_cycle != 0 else (1 if random.random() > 0.5 else -1)
                magnitude = abs(np.random.normal(0, params["std_dev"]))
                magnitude = round(magnitude * 4) / 4  # Round to nearest 0.25
                magnitude = max(0.25, min(1.0, magnitude))  # Constrain to reasonable range
                
                # Apply change
                previous_rate = current_rates[rate_type]
                new_rate = previous_rate + direction * magnitude
                new_rate = max(params["min_rate"], min(params["max_rate"], new_rate))
                new_rate = round(new_rate * 4) / 4  # Round to nearest 0.25
                
                if new_rate != previous_rate:
                    current_rates[rate_type] = new_rate
                    last_change_date[rate_type] = current_date
                    
                    # Decision is usually made a couple of days before announcement
                    decision_date = current_date - timedelta(days=random.randint(2, 5))
                    
                    # Create policy rate record
                    policy_rate = PolicyRate(
                        effective_date=current_date,
                        rate_type=rate_type,
                        rate_value=new_rate,
                        previous_value=previous_rate,
                        change=new_rate - previous_rate,
                        decision_date=decision_date,
                        announcement_date=decision_date,
                        rationale=f"{'Increase' if new_rate > previous_rate else 'Decrease'} in response to {'inflationary pressures' if new_rate > previous_rate else 'economic slowdown'}"
                    )
                    db.add(policy_rate)
        
        # Move to next month
        if current_date.month == 12:
            current_date = date(current_date.year + 1, 1, 1)
        else:
            current_date = date(current_date.year, current_date.month + 1, 1)
    
    db.flush()
    logger.info("Generated policy rate data")


# Additional data generation functions for other models would go here...
