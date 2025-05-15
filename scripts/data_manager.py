#!/usr/bin/env python3
"""
BankSim Data Manager

This script provides utilities for managing economic datasets used in the BankSim platform:
1. Importing data from various sources (CSV, Excel, APIs)
2. Validating data integrity and structure
3. Converting between different formats
4. Loading sample data for development and testing

Usage:
    python data_manager.py import --source <source_path> --type <data_type>
    python data_manager.py validate --source <source_path>
    python data_manager.py generate-sample --output <output_path>
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import random
import numpy as np


def import_data(source_path, data_type):
    """
    Import data from the specified source into the BankSim database.
    
    Args:
        source_path: Path to the data source file
        data_type: Type of data (e.g., 'gdp', 'cpi', 'interest_rates')
    """
    print(f"Importing {data_type} data from {source_path}")
    
    # Placeholder for actual implementation
    print("NOTE: This is a placeholder. Actual implementation will depend on database structure.")
    
    # Example implementation would:
    # 1. Read data from source file (CSV, Excel, etc.)
    # 2. Validate data structure
    # 3. Transform data if needed
    # 4. Connect to database
    # 5. Insert or update data
    
    print(f"Successfully imported {data_type} data")


def validate_data(source_path):
    """
    Validate data structure and integrity from the specified source.
    
    Args:
        source_path: Path to the data source file
    """
    print(f"Validating data from {source_path}")
    
    # Placeholder for actual implementation
    print("NOTE: This is a placeholder. Actual implementation will perform real validation.")
    
    # Example validation would:
    # 1. Check file format
    # 2. Verify column names and data types
    # 3. Check for missing values
    # 4. Validate date ranges
    # 5. Check for outliers or anomalies
    
    print("Data validation completed")


def generate_sample_data(output_path):
    """
    Generate sample economic data for development and testing.
    
    Args:
        output_path: Directory to save generated data files
    """
    print(f"Generating sample economic data to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate sample GDP data
    generate_gdp_data(output_path)
    
    # Generate sample CPI data
    generate_cpi_data(output_path)
    
    # Generate sample interest rate data
    generate_interest_rate_data(output_path)
    
    print(f"Sample data generation completed. Files saved to {output_path}")


def generate_gdp_data(output_path):
    """Generate sample quarterly GDP data with components."""
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2025, 1, 1)
    
    # Generate quarterly dates
    dates = []
    current_date = start_date
    while current_date < end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date = current_date + timedelta(days=91)
    
    # Base GDP value (in billions)
    base_gdp = 100.0
    
    # Generate GDP with growth trend and seasonal variation
    gdp_data = []
    for i, date in enumerate(dates):
        year = i // 4  # Approximate year count
        quarter = i % 4
        
        # Growth trend: ~5% annual growth with some randomness
        trend_factor = (1.05 ** year) * (1 + random.uniform(-0.02, 0.02))
        
        # Seasonal variation
        seasonal_factor = 1.0 + 0.03 * (quarter == 2) - 0.02 * (quarter == 0)
        
        # Final GDP value
        gdp_value = base_gdp * trend_factor * seasonal_factor
        
        # GDP components (simplified)
        consumption = gdp_value * random.uniform(0.55, 0.65)
        investment = gdp_value * random.uniform(0.15, 0.25)
        government = gdp_value * random.uniform(0.15, 0.20)
        net_exports = gdp_value - consumption - investment - government
        
        gdp_data.append({
            "date": date,
            "gdp": round(gdp_value, 2),
            "consumption": round(consumption, 2),
            "investment": round(investment, 2),
            "government": round(government, 2),
            "net_exports": round(net_exports, 2)
        })
    
    # Save to file
    output_file = os.path.join(output_path, "gdp_quarterly.json")
    with open(output_file, "w") as f:
        json.dump(gdp_data, f, indent=2)
    
    print(f"Generated GDP data: {output_file}")


def generate_cpi_data(output_path):
    """Generate sample monthly CPI data with components."""
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2025, 1, 1)
    
    # Generate monthly dates
    dates = []
    current_date = start_date
    while current_date < end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date = current_date + timedelta(days=30)
    
    # Base CPI value
    base_cpi = 100.0
    
    # Generate CPI with inflation trend and some volatility
    cpi_data = []
    
    # Components and their weights
    components = {
        "food": 0.25,
        "housing": 0.35,
        "transportation": 0.15,
        "healthcare": 0.10,
        "education": 0.05,
        "other": 0.10
    }
    
    monthly_inflation = 0.003  # ~3.6% annual inflation
    
    for i, date in enumerate(dates):
        # Inflation trend with some randomness
        trend_factor = (1 + monthly_inflation) ** i * (1 + random.uniform(-0.001, 0.001))
        
        # Final CPI value
        cpi_value = base_cpi * trend_factor
        
        # Component-specific values with their own variations
        component_values = {}
        for component, weight in components.items():
            # Each component has slightly different inflation dynamics
            component_trend = trend_factor * (1 + random.uniform(-0.01, 0.01))
            component_values[component] = round(base_cpi * component_trend, 2)
        
        cpi_data.append({
            "date": date,
            "cpi": round(cpi_value, 2),
            **component_values
        })
    
    # Save to file
    output_file = os.path.join(output_path, "cpi_monthly.json")
    with open(output_file, "w") as f:
        json.dump(cpi_data, f, indent=2)
    
    print(f"Generated CPI data: {output_file}")


def generate_interest_rate_data(output_path):
    """Generate sample daily interest rate data."""
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2025, 1, 1)
    
    # Generate daily dates (business days only, simplified)
    dates = []
    current_date = start_date
    while current_date < end_date:
        # Skip weekends (simplified)
        if current_date.weekday() < 5:
            dates.append(current_date.strftime("%Y-%m-%d"))
        current_date = current_date + timedelta(days=1)
    
    # Initial policy rate
    policy_rate = 5.0
    
    # Interest rate data with policy changes and market reactions
    interest_data = []
    
    # Policy change points (approximately every 6 months)
    policy_changes = {}
    for year in range(2000, 2025):
        for month in (1, 7):
            change_date = datetime(year, month, 15)
            # Skip if beyond our range
            if change_date >= end_date:
                continue
            # Random policy change between -0.5 and +0.5
            policy_changes[change_date.strftime("%Y-%m-%d")] = random.uniform(-0.5, 0.5)
    
    # Generate daily rates
    for date in dates:
        # Apply policy changes
        if date in policy_changes:
            policy_rate += policy_changes[date]
            policy_rate = max(0.25, min(10.0, policy_rate))  # Keep between 0.25% and 10%
        
        # Market rates derived from policy rate with spreads
        overnight_rate = policy_rate + random.uniform(-0.1, 0.1)
        tbill_3m = policy_rate + random.uniform(0.0, 0.2)
        tbill_1y = policy_rate + random.uniform(0.1, 0.4)
        bond_5y = policy_rate + random.uniform(0.3, 1.0)
        bond_10y = policy_rate + random.uniform(0.5, 1.5)
        
        interest_data.append({
            "date": date,
            "policy_rate": round(policy_rate, 2),
            "overnight_rate": round(overnight_rate, 2),
            "tbill_3m": round(tbill_3m, 2),
            "tbill_1y": round(tbill_1y, 2),
            "bond_5y": round(bond_5y, 2),
            "bond_10y": round(bond_10y, 2)
        })
    
    # Save to file
    output_file = os.path.join(output_path, "interest_rates_daily.json")
    with open(output_file, "w") as f:
        json.dump(interest_data, f, indent=2)
    
    print(f"Generated interest rate data: {output_file}")


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(description="BankSim Data Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import data from source")
    import_parser.add_argument("--source", required=True, help="Path to data source")
    import_parser.add_argument("--type", required=True, help="Type of data to import")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data structure")
    validate_parser.add_argument("--source", required=True, help="Path to data source")
    
    # Generate sample data command
    sample_parser = subparsers.add_parser("generate-sample", help="Generate sample data")
    sample_parser.add_argument("--output", required=True, help="Output directory for sample data")
    
    args = parser.parse_args()
    
    if args.command == "import":
        import_data(args.source, args.type)
    elif args.command == "validate":
        validate_data(args.source)
    elif args.command == "generate-sample":
        generate_sample_data(args.output)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
