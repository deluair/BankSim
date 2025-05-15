"""
Foreign Exchange Simulation Module.

This module implements simulation logic for foreign exchange operations,
including exchange rate dynamics, foreign reserves management, and intervention.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import date, datetime, timedelta
import logging
from sqlalchemy.orm import Session

from .config import ForeignExchangeConfig
from .. import models

logger = logging.getLogger(__name__)


class ForeignExchangeSimulator:
    """
    Simulator for foreign exchange operations.
    
    This class models central bank foreign exchange operations including:
    - Exchange rate dynamics and forecasting
    - Foreign reserves management
    - FX market interventions
    - Balance of payments effects
    """
    
    def __init__(self, config: ForeignExchangeConfig, db: Optional[Session] = None):
        """
        Initialize the foreign exchange simulator.
        
        Args:
            config: Configuration parameters for the simulation
            db: Database session for loading/saving data (optional)
        """
        self.config = config
        self.db = db
        
        # Initialize simulation state
        self.current_date = config.start_date
        self.end_date = config.end_date
        
        # Set up time series storage
        self.time_points = self._generate_time_points()
        self.state = self._initialize_state()
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def _generate_time_points(self) -> List[date]:
        """Generate a list of time points for the simulation based on the config."""
        time_points = []
        current = self.config.start_date
        
        if self.config.time_step == "daily":
            delta = timedelta(days=1)
        elif self.config.time_step == "weekly":
            delta = timedelta(days=7)
        elif self.config.time_step == "monthly":
            delta = timedelta(days=30)
        elif self.config.time_step == "quarterly":
            delta = timedelta(days=90)
        else:
            raise ValueError(f"Unknown time step: {self.config.time_step}")
        
        while current <= self.config.end_date:
            time_points.append(current)
            current += delta
        
        return time_points
    
    def _initialize_state(self) -> Dict:
        """Initialize the simulation state."""
        # Initialize the main structure of the state dictionary
        state = {
            "currencies": list(self.config.currencies), # Ensure it's a list, not a Pydantic type
            "exchange_rates": {
                currency: pd.Series(index=self.time_points, dtype=float)
                for currency in self.config.currencies
            },
            "effective_exchange_rate": pd.Series(index=self.time_points, dtype=float),
            "real_effective_exchange_rate": pd.Series(index=self.time_points, dtype=float),
            "foreign_reserves": {
                currency: pd.Series(index=self.time_points, dtype=float)
                for currency in self.config.currencies
            },
            "total_reserves_usd": pd.Series(index=self.time_points, dtype=float),
            "reserves_to_imports": pd.Series(index=self.time_points, dtype=float),
            "reserves_to_short_term_debt": pd.Series(index=self.time_points, dtype=float),
            "reserves_to_m2": pd.Series(index=self.time_points, dtype=float),
            "current_account": pd.Series(index=self.time_points, dtype=float),
            "financial_account": pd.Series(index=self.time_points, dtype=float),
            "capital_account": pd.Series(index=self.time_points, dtype=float),
            "errors_omissions": pd.Series(index=self.time_points, dtype=float),
            "interventions": pd.DataFrame(index=self.time_points, 
                                         columns=["amount_usd", "type", "currency", "result"]),
            "exchange_market_pressure": pd.Series(index=self.time_points, dtype=float),
            "domestic_inflation": pd.Series(index=self.time_points, dtype=float),
            "domestic_interest_rate": pd.Series(index=self.time_points, dtype=float),
            "domestic_gdp_growth": pd.Series(index=self.time_points, dtype=float),
            "global_risk_aversion": pd.Series(index=self.time_points, dtype=float),
            "commodity_price_index": pd.Series(index=self.time_points, dtype=float),
            "trading_partner_growth": pd.Series(index=self.time_points, dtype=float),
            "us_interest_rate": pd.Series(index=self.time_points, dtype=float),
        }
        
        # Load initial data from database or initialize synthetically into the local 'state' dict
        if self.db is not None:
            self._load_initial_data_from_db(state) # Pass local state
        else:
            self._initialize_synthetic_data(state) # Pass local state
        
        return state
    
    def _load_initial_data_from_db(self, state: Dict):
        """Load initial FX data from the database into the provided state dictionary."""
        if self.db is None:
            return
        
        # Get the latest exchange rates
        for currency in state["currencies"]: # Use passed state
            latest_rate = (
                self.db.query(models.ExchangeRate)
                .filter(models.ExchangeRate.currency_code == currency)
                .order_by(models.ExchangeRate.date.desc())
                .first()
            )
            
            if latest_rate:
                state["exchange_rates"][currency].iloc[0] = latest_rate.rate # Use passed state
        
        # Get the latest foreign reserves
        for currency in state["currencies"]: # Use passed state
            latest_reserves = (
                self.db.query(models.ForeignReserves)
                .filter(models.ForeignReserves.currency_code == currency)
                .order_by(models.ForeignReserves.date.desc())
                .first()
            )
            
            if latest_reserves:
                state["foreign_reserves"][currency].iloc[0] = latest_reserves.amount # Use passed state
        
        # Get the latest balance of payments data
        latest_bop = (
            self.db.query(models.BalanceOfPayments)
            .order_by(models.BalanceOfPayments.date.desc())
            .first()
        )
        
        if latest_bop:
            state["current_account"].iloc[0] = latest_bop.current_account # Use passed state
            state["financial_account"].iloc[0] = latest_bop.financial_account # Use passed state
            state["capital_account"].iloc[0] = latest_bop.capital_account # Use passed state
            state["errors_omissions"].iloc[0] = latest_bop.errors_omissions # Use passed state
        
        # Get the latest economic indicators for FX model
        latest_indicators = {
            "domestic_inflation": (
                self.db.query(models.CPIData)
                .order_by(models.CPIData.date.desc())
                .first()
            ),
            "domestic_interest_rate": (
                self.db.query(models.PolicyRate)
                .order_by(models.PolicyRate.effective_date.desc())
                .first()
            ),
            "domestic_gdp_growth": (
                self.db.query(models.GDPData)
                .order_by(models.GDPData.date.desc())
                .first()
            )
        }
        
        if latest_indicators["domestic_inflation"]:
            state["domestic_inflation"].iloc[0] = latest_indicators["domestic_inflation"].yoy_change # Use passed state
        
        if latest_indicators["domestic_interest_rate"]:
            state["domestic_interest_rate"].iloc[0] = latest_indicators["domestic_interest_rate"].rate_value # Use passed state
        
        if latest_indicators["domestic_gdp_growth"]:
            state["domestic_gdp_growth"].iloc[0] = latest_indicators["domestic_gdp_growth"].yoy_growth # Use passed state
    
    def _initialize_synthetic_data(self, state: Dict):
        """Initialize provided state dictionary with synthetic data for simulation."""
        # Set initial exchange rates
        for currency in state["currencies"]: # Use passed state
            if currency == "USD":
                initial_rate = 1.0  # Base currency
            elif currency == "EUR":
                initial_rate = 0.9  # EUR/USD
            elif currency == "GBP":
                initial_rate = 0.8  # GBP/USD
            elif currency == "JPY":
                initial_rate = 110.0  # JPY/USD
            elif currency == "CNY":
                initial_rate = 6.5  # CNY/USD
            else:
                # For other currencies, use random values
                initial_rate = np.random.uniform(10, 100)  # Generic value
            
            state["exchange_rates"][currency].iloc[0] = initial_rate # Use passed state
        
        # Calculate initial effective exchange rate (simple equally-weighted basket)
        state["effective_exchange_rate"].iloc[0] = 100.0  # Index value # Use passed state
        state["real_effective_exchange_rate"].iloc[0] = 100.0  # Index value # Use passed state
        
        # Set initial foreign reserves
        total_reserves_usd = 0
        for currency in state["currencies"]: # Use passed state
            if currency == "USD":
                # Most reserves in USD
                initial_reserves = np.random.uniform(20000, 30000)  # in millions
            elif currency in ["EUR", "JPY"]:
                # Substantial reserves in major currencies
                initial_reserves = np.random.uniform(5000, 15000)
            else:
                # Smaller amounts in other currencies
                initial_reserves = np.random.uniform(500, 5000)
            
            state["foreign_reserves"][currency].iloc[0] = initial_reserves # Use passed state
            if currency == "USD":
                total_reserves_usd += initial_reserves
            elif state["exchange_rates"][currency].iloc[0] != 0: # Check for division by zero for non-USD
                total_reserves_usd += initial_reserves / state["exchange_rates"][currency].iloc[0]
        
        state["total_reserves_usd"].iloc[0] = total_reserves_usd # Use passed state
        
        # Initialize reserves adequacy metrics (placeholders)
        state["reserves_to_imports"].iloc[0] = 6.0  # Months of import cover # Use passed state
        state["reserves_to_short_term_debt"].iloc[0] = 2.0  # Ratio # Use passed state
        state["reserves_to_m2"].iloc[0] = 0.2  # Ratio # Use passed state
        
        # Initialize balance of payments components (placeholders for initial period)
        state["current_account"].iloc[0] = np.random.uniform(-500, 200)  # Millions USD # Use passed state
        state["financial_account"].iloc[0] = np.random.uniform(-300, 600)  # Millions USD # Use passed state
        state["capital_account"].iloc[0] = np.random.uniform(10, 50)  # Millions USD # Use passed state
        state["errors_omissions"].iloc[0] = np.random.uniform(-50, 50)  # Millions USD # Use passed state
        
        # Initialize FX interventions (empty for initial period)
        # state["interventions"].iloc[0] will be NaN by default for DataFrame
        
        # Initialize Market Pressure Index
        state["exchange_market_pressure"].iloc[0] = 0.0 # Use passed state
        
        # Initialize domestic economic variables (placeholders)
        state["domestic_inflation"].iloc[0] = np.random.uniform(2.0, 8.0)  # Percent # Use passed state
        state["domestic_interest_rate"].iloc[0] = np.random.uniform(4.0, 10.0)  # Percent # Use passed state
        state["domestic_gdp_growth"].iloc[0] = np.random.uniform(1.0, 5.0)  # Percent # Use passed state
        
        # Initialize external variables (placeholders)
        state["global_risk_aversion"].iloc[0] = np.random.uniform(0.2, 0.8)  # Index # Use passed state
        state["commodity_price_index"].iloc[0] = np.random.uniform(80, 120)  # Index # Use passed state
        state["trading_partner_growth"].iloc[0] = np.random.uniform(1.0, 3.0)  # Percent # Use passed state
        state["us_interest_rate"].iloc[0] = np.random.uniform(0.5, 3.0) # Percent # Use passed state
    
    def run_simulation(self) -> Dict:
        """
        Run the full foreign exchange simulation.
        
        Returns:
            Results of the simulation
        """
        # Process each time point
        for i, current_date in enumerate(self.time_points):
            self.current_date = current_date
            
            # Skip the first point which is initialized already
            if i == 0:
                continue
            
            # Update external variables
            self._update_external_variables(i)
            
            # Update domestic variables
            self._update_domestic_variables(i)
            
            # Update balance of payments
            self._update_balance_of_payments(i)
            
            # Calculate market pressure
            self._calculate_exchange_market_pressure(i)
            
            # Determine if intervention is needed
            intervention = self._determine_intervention(i)
            
            # Update exchange rates
            self._update_exchange_rates(i, intervention)
            
            # Update foreign reserves
            self._update_foreign_reserves(i, intervention)
            
            # Calculate effective exchange rate indices
            self._calculate_effective_exchange_rates(i)
            
            # Update reserves adequacy metrics
            self._update_reserves_adequacy_metrics(i)
        
        return self._prepare_results()
    
    def _update_external_variables(self, index):
        """Update external variables that affect the FX market."""
        # Use AR(1) processes with shocks for external variables
        variables = {
            "global_risk_aversion": {
                "mean_reversion": 0.7,  # High persistence
                "mean": 0,  # Long-run mean
                "volatility": 0.2,  # Volatility of shocks
            },
            "commodity_price_index": {
                "growth": 0.001,  # Slight growth trend
                "volatility": 2.0,  # Volatility in index points
            },
            "trading_partner_growth": {
                "mean_reversion": 0.8,
                "mean": 2.5,  # Long-run mean growth rate
                "volatility": 0.3,
            },
            "us_interest_rate": {
                "mean_reversion": 0.9,  # Very persistent
                "mean": 2.0,  # Long-run mean rate
                "volatility": 0.1,
            }
        }
        
        for var_name, params in variables.items():
            prev_value = self.state[var_name].iloc[index-1]
            
            if "growth" in params:
                # Random walk with drift
                growth = params["growth"]
                volatility = params["volatility"]
                
                # New value = old value * (1 + growth + random shock)
                shock = np.random.normal(0, volatility)
                if var_name == "commodity_price_index":
                    # Absolute shock for index
                    new_value = prev_value * (1 + growth) + shock
                else:
                    # Percentage shock
                    new_value = prev_value * (1 + growth + shock/100)
            else:
                # Mean reverting process
                mean = params["mean"]
                mean_reversion = params["mean_reversion"]
                volatility = params["volatility"]
                
                # New value = (1-α) * old value + α * mean + shock
                shock = np.random.normal(0, volatility)
                new_value = mean_reversion * prev_value + (1 - mean_reversion) * mean + shock
            
            self.state[var_name].iloc[index] = new_value
    
    def _update_domestic_variables(self, index):
        """Update domestic variables based on policy and economic developments."""
        # For this simulation, we'll use simple autoregressive processes
        # In a full model, these would be linked to the monetary policy simulator
        
        variables = {
            "domestic_inflation": {
                "mean_reversion": 0.7,
                "mean": 2.0,  # Target inflation
                "volatility": 0.2,
            },
            "domestic_interest_rate": {
                "mean_reversion": 0.85,
                "mean": 3.0,  # Neutral rate
                "volatility": 0.1,
            },
            "domestic_gdp_growth": {
                "mean_reversion": 0.6,
                "mean": 2.5,  # Potential growth
                "volatility": 0.3,
            }
        }
        
        for var_name, params in variables.items():
            prev_value = self.state[var_name].iloc[index-1]
            
            # Mean reverting process
            mean = params["mean"]
            mean_reversion = params["mean_reversion"]
            volatility = params["volatility"]
            
            # New value = (1-α) * old value + α * mean + shock
            shock = np.random.normal(0, volatility)
            new_value = mean_reversion * prev_value + (1 - mean_reversion) * mean + shock
            
            self.state[var_name].iloc[index] = new_value
    
    def _update_balance_of_payments(self, index):
        """Update balance of payments components based on economic conditions."""
        # In a quarterly model, only update every ~90 days
        is_quarterly_update = (index % (90 // max(1, (self.time_points[1] - self.time_points[0]).days))) == 0
        
        if not is_quarterly_update and index > 1:
            # Copy previous values for non-quarterly updates
            self.state["current_account"].iloc[index] = self.state["current_account"].iloc[index-1]
            self.state["financial_account"].iloc[index] = self.state["financial_account"].iloc[index-1]
            self.state["capital_account"].iloc[index] = self.state["capital_account"].iloc[index-1]
            self.state["errors_omissions"].iloc[index] = self.state["errors_omissions"].iloc[index-1]
            return
        
        # Current account influenced by:
        # - Exchange rate (competitiveness)
        # - Domestic growth (imports)
        # - Trading partner growth (exports)
        # - Commodity prices (for commodity exporters/importers)
        
        prev_current = self.state["current_account"].iloc[index-1]
        
        # Real exchange rate effect (lagged)
        reer_effect = -0.3 * (self.state["real_effective_exchange_rate"].iloc[max(0, index-2)] - 100) / 10
        
        # Domestic growth effect (higher growth -> more imports -> lower CA)
        domestic_growth_effect = -0.5 * (self.state["domestic_gdp_growth"].iloc[index] - 2.5)
        
        # Partner growth effect (higher partner growth -> more exports -> higher CA)
        partner_growth_effect = 0.7 * (self.state["trading_partner_growth"].iloc[index] - 2.5)
        
        # Commodity price effect (depends on if net importer or exporter)
        is_commodity_exporter = self.config.is_commodity_exporter
        commodity_price_change = (
            self.state["commodity_price_index"].iloc[index] / 
            self.state["commodity_price_index"].iloc[index-1] - 1
        ) * 100
        
        commodity_effect = 0.2 * commodity_price_change if is_commodity_exporter else -0.1 * commodity_price_change
        
        # Random shock
        shock = np.random.normal(0, 100)
        
        # Calculate new current account
        persistence = 0.7  # Autocorrelation
        new_current = (
            persistence * prev_current + 
            (1 - persistence) * (reer_effect + domestic_growth_effect + partner_growth_effect + commodity_effect) * 500 + 
            shock
        )
        
        self.state["current_account"].iloc[index] = new_current
        
        # Financial account influenced by:
        # - Interest rate differentials
        # - Growth differentials
        # - Global risk sentiment
        
        prev_financial = self.state["financial_account"].iloc[index-1]
        
        # Interest rate differential effect
        interest_diff = self.state["domestic_interest_rate"].iloc[index] - self.state["us_interest_rate"].iloc[index]
        interest_effect = 100 * interest_diff
        
        # Growth differential effect
        growth_diff = self.state["domestic_gdp_growth"].iloc[index] - self.state["trading_partner_growth"].iloc[index]
        growth_effect = 75 * growth_diff
        
        # Risk sentiment effect (higher risk aversion -> capital outflows)
        risk_effect = -200 * self.state["global_risk_aversion"].iloc[index]
        
        # Random shock
        shock = np.random.normal(0, 150)
        
        # Calculate new financial account
        persistence = 0.6
        new_financial = (
            persistence * prev_financial + 
            (1 - persistence) * (interest_effect + growth_effect + risk_effect) + 
            shock
        )
        
        self.state["financial_account"].iloc[index] = new_financial
        
        # Capital account (mostly stable with small variations)
        prev_capital = self.state["capital_account"].iloc[index-1]
        new_capital = prev_capital * 0.9 + np.random.normal(20, 15)
        self.state["capital_account"].iloc[index] = new_capital
        
        # Errors and omissions (random)
        self.state["errors_omissions"].iloc[index] = np.random.normal(0, 50)
    
    def _calculate_exchange_market_pressure(self, index):
        """Calculate exchange market pressure index."""
        # Exchange Market Pressure (EMP) combines:
        # - Exchange rate changes
        # - Reserves changes
        # - Interest rate changes
        
        # Get exchange rate change (using USD as reference)
        if "USD" in self.state["exchange_rates"]:
            fx_change = (
                self.state["exchange_rates"]["USD"].iloc[index-1] / 
                self.state["exchange_rates"]["USD"].iloc[max(0, index-2)] - 1
            ) * 100 if index > 1 else 0
        else:
            # Use effective exchange rate if USD not available
            fx_change = (
                self.state["effective_exchange_rate"].iloc[index-1] / 
                self.state["effective_exchange_rate"].iloc[max(0, index-2)] - 1
            ) * 100 if index > 1 else 0
        
        # Get reserves change
        reserves_change = (
            self.state["total_reserves_usd"].iloc[index-1] / 
            self.state["total_reserves_usd"].iloc[max(0, index-2)] - 1
        ) * 100 if index > 1 else 0
        
        # Get interest rate change
        interest_change = (
            self.state["domestic_interest_rate"].iloc[index] - 
            self.state["domestic_interest_rate"].iloc[index-1]
        )
        
        # Calculate EMP (positive means depreciation pressure)
        # Standardize components by their standard deviations
        fx_std = max(0.5, np.std(self.state["effective_exchange_rate"].iloc[:index]))
        reserves_std = max(0.5, np.std(self.state["total_reserves_usd"].iloc[:index]))
        interest_std = max(0.1, np.std(self.state["domestic_interest_rate"].iloc[:index]))
        
        emp = (
            fx_change / fx_std -
            reserves_change / reserves_std +
            interest_change / interest_std
        )
        
        # Store the EMP index
        self.state["exchange_market_pressure"].iloc[index] = emp
    
    def _determine_intervention(self, index) -> Optional[Dict]:
        """Determine if FX intervention is needed based on market pressure and policy objectives."""
        # Get current exchange market pressure
        emp = self.state["exchange_market_pressure"].iloc[index]
        
        # Check if intervention thresholds are breached
        depreciation_threshold = self.config.intervention_thresholds["depreciation"]
        appreciation_threshold = self.config.intervention_thresholds["appreciation"]
        
        # No intervention if pressure is within thresholds
        if emp < depreciation_threshold and emp > appreciation_threshold:
            return None
        
        # Determine intervention type and size based on pressure
        if emp >= depreciation_threshold:  # Depreciation pressure
            intervention_type = "sell_foreign_currency"
            # Scale intervention size based on pressure intensity
            severity = min(1.0, (emp - depreciation_threshold) / depreciation_threshold)
            max_size = self.config.max_intervention_size["sell"]
            size = max_size * severity
            
            # Limit intervention by available reserves
            available_reserves = self.state["total_reserves_usd"].iloc[index-1]
            min_reserves = self.config.minimum_reserves_threshold * available_reserves
            max_allowed = available_reserves - min_reserves
            size = min(size, max_allowed)
            
            # Determine currency to sell (usually USD)
            currency = "USD"
            
        else:  # Appreciation pressure
            intervention_type = "buy_foreign_currency"
            # Scale intervention size based on pressure intensity
            severity = min(1.0, (appreciation_threshold - emp) / abs(appreciation_threshold))
            max_size = self.config.max_intervention_size["buy"]
            size = max_size * severity
            
            # Determine currency to buy (usually USD)
            currency = "USD"
        
        # Don't intervene if size is too small to be effective
        if size < self.config.min_intervention_size:
            return None
        
        # Return intervention details
        return {
            "type": intervention_type,
            "amount_usd": size,
            "currency": currency,
            "date": self.current_date
        }
    
    def _update_exchange_rates(self, index, intervention: Optional[Dict]):
        """Update exchange rates based on fundamentals, market pressure and intervention."""
        # Exchange rate model is based on:
        # 1. Interest rate differentials (uncovered interest parity)
        # 2. Inflation differentials (purchasing power parity)
        # 3. Growth differentials
        # 4. Risk premiums and sentiment
        # 5. Intervention effects
        
        for currency in self.state["currencies"]:
            prev_rate = self.state["exchange_rates"][currency].iloc[index-1]
            
            # Calculate fundamental drivers
            
            # 1. Interest rate differential (higher domestic rates -> currency appreciation)
            if currency == "USD": # Assuming domestic currency is NOT USD, and rates are vs USD
                interest_diff = self.state["domestic_interest_rate"].iloc[index] - self.state["us_interest_rate"].iloc[index]
            else:
                # For non-USD foreign currencies, the model assumes their rate vs USD is what's stored.
                # If we are modeling LocalCurrency/ForeignCurrency, then the interest_diff logic needs care.
                # For simplicity, let's assume the stored rates are ForeignCurrency/USD.
                # And domestic_interest_rate is for LocalCurrency.
                # The comparison should be domestic vs the specific foreign currency's benchmark rate.
                # Using US interest rate as a proxy for all foreign benchmark rates if not USD.
                # This part might need refinement based on how `self.state["exchange_rates"][currency]` is defined.
                # For now, sticking to the original logic's apparent intention:
                interest_diff = self.state["domestic_interest_rate"].iloc[index] - self.state["us_interest_rate"].iloc[index]
            
            interest_effect = -0.5 * interest_diff / 100  # Effect size parameter
            
            # 2. Inflation differential (higher domestic inflation -> currency depreciation)
            # Assumes foreign inflation for the specific currency pair is proxied by config
            inflation_diff = self.state["domestic_inflation"].iloc[index] - self.config.assumed_foreign_inflation_pct
            inflation_effect = 0.3 * inflation_diff / 100  # Effect size parameter
            
            # 3. Growth differential (higher domestic growth -> currency appreciation, but can increase imports)
            growth_diff = self.state["domestic_gdp_growth"].iloc[index] - self.state["trading_partner_growth"].iloc[index]
            growth_effect = -0.2 * growth_diff / 100  # Effect size parameter
            
            # 4. Risk premium (higher global risk aversion -> emerging market currency depreciation)
            risk_premium = 0.3 * self.state["global_risk_aversion"].iloc[index] / 100
            
            # 5. Intervention effect
            intervention_effect = 0
            if intervention is not None and self.config.fx_market_size > 0:
                if intervention["currency"] == currency: # Intervention is in this specific currency
                    # Assuming intervention amount is in USD, and exchange rate is ForeignCur/USD
                    # If selling foreign currency (e.g. USD), you are buying domestic, appreciating domestic.
                    # If prev_rate is FC/USD, selling USD means prev_rate should decrease.
                    if intervention["type"] == "sell_foreign_currency":
                        impact_pct = -0.5 * (intervention["amount_usd"] / self.config.fx_market_size) # Proportional impact
                        intervention_effect = max(-0.02, min(0, impact_pct))  # Cap effect, ensure negative or zero
                    else:  # buy_foreign_currency (e.g. USD)
                        # Buying USD means selling domestic, depreciating domestic.
                        # prev_rate (FC/USD) should increase.
                        impact_pct = 0.5 * (intervention["amount_usd"] / self.config.fx_market_size) # Proportional impact
                        intervention_effect = min(0.02, max(0, impact_pct))  # Cap effect, ensure positive or zero
            
            # 6. Random market noise
            volatility = self.config.fx_volatility.get(currency, 1.0) # Default volatility if currency not in config
            noise = np.random.normal(0, volatility / 100)
            
            # Combine all effects (multiplicative model)
            # New rate = Old rate * (1 + sum of effects)
            rate_change = (interest_effect + inflation_effect + growth_effect + risk_premium + intervention_effect + noise)
            new_rate = prev_rate * (1 + rate_change)
            
            # Store new exchange rate
            self.state["exchange_rates"][currency].iloc[index] = new_rate
    
    def _update_foreign_reserves(self, index, intervention: Optional[Dict]):
        """Update foreign reserves based on interventions and valuation changes."""
        # Copy previous reserves as starting point
        for currency in self.state["currencies"]:
            self.state["foreign_reserves"][currency].iloc[index] = self.state["foreign_reserves"][currency].iloc[index-1]
        
        # Apply intervention changes if any
        if intervention is not None:
            currency = intervention["currency"]
            amount = intervention["amount_usd"]
            
            if intervention["type"] == "sell_foreign_currency":
                # Reduce reserves when selling
                current_reserves = self.state["foreign_reserves"][currency].iloc[index]
                self.state["foreign_reserves"][currency].iloc[index] = max(0, current_reserves - amount)
            else:  # buy_foreign_currency
                # Increase reserves when buying
                current_reserves = self.state["foreign_reserves"][currency].iloc[index]
                self.state["foreign_reserves"][currency].iloc[index] = current_reserves + amount
        
        # Record intervention details
        if intervention is not None:
            self.state["interventions"].loc[self.current_date, "amount_usd"] = intervention["amount_usd"]
            self.state["interventions"].loc[self.current_date, "type"] = intervention["type"]
            self.state["interventions"].loc[self.current_date, "currency"] = intervention["currency"]
            
            # Determine if intervention was successful based on exchange rate movement
            if intervention["type"] == "sell_foreign_currency":
                # Success if currency appreciated or depreciated less than it would have
                success = self.state["exchange_market_pressure"].iloc[index] < self.state["exchange_market_pressure"].iloc[index-1]
            else:  # buy_foreign_currency
                # Success if currency depreciated or appreciated less than it would have
                success = self.state["exchange_market_pressure"].iloc[index] > self.state["exchange_market_pressure"].iloc[index-1]
            
            self.state["interventions"].loc[self.current_date, "result"] = "success" if success else "limited_impact"
        
        # Update total reserves in USD equivalent
        total_usd = 0
        for currency in self.state["currencies"]:
            amount = self.state["foreign_reserves"][currency].iloc[index]
            rate = self.state["exchange_rates"][currency].iloc[index]
            # Convert to USD
            usd_value = amount / rate if rate > 0 else 0
            total_usd += usd_value
        
        self.state["total_reserves_usd"].iloc[index] = total_usd
    
    def _calculate_effective_exchange_rates(self, index):
        """Calculate nominal and real effective exchange rate indices."""
        # Effective exchange rates are weighted averages of bilateral rates
        # Here we use a simplified approach with equal weights
        
        # 1. Nominal effective exchange rate (NEER)
        rates = []
        weights = []
        
        # Assuming self.config.target_currency (e.g. 'USD') is the 'local' context for these rates
        # And other currencies in self.state["currencies"] are foreign against this target_currency.
        # If self.state["exchange_rates"][currency] is ForeignCur/TargetCur:
        # An increase in rate means TargetCur depreciates against ForeignCur.
        # For NEER, an increase usually means appreciation of local currency.
        # So, if rates are ForeignCur/Local, then NEER index = (initial_rate / current_rate) * 100 for appreciation.
        
        # Let's clarify: assume self.state["exchange_rates"][currency] stores units of DOMESTIC currency per unit of FOREIGN currency.
        # Example: if domestic is BDT and foreign is USD, rate is BDT/USD. Increase means BDT depreciation.
        # NEER: increase means appreciation. So need to invert if using BDT/USD.
        # The original code's (initial_rate / current_rate) implies rates are Foreign/Domestic or it defines appreciation inversely.
        # Let's assume rates are ForeignCur/DomesticCur for NEER calc to match (initial_rate / current_rate)
        # OR, if rates are DomesticCur/ForeignCur, then NEER index is (current_rate / initial_rate) * 100 for depreciation index, invert for appreciation index.
        
        # Sticking to the original (initial_rate / current_rate) * 100 implies an increase in NEER is appreciation.
        # This means self.state["exchange_rates"][currency] is treated as ForeignCur/DomesticCur for NEER calculation.

        num_currencies_for_neer = 0
        for currency_code in self.state["currencies"]:
            if currency_code != self.config.target_currency: # Exclude the base currency itself from NEER basket components
                num_currencies_for_neer +=1

        if num_currencies_for_neer == 0 and len(self.state["currencies"]) == 1 and self.config.target_currency in self.state["currencies"]:
            # Only one currency listed, and it's the target_currency. NEER is trivially 100.
            self.state["effective_exchange_rate"].iloc[index] = 100.0
            # REER calculation
            # Domestic inflation index (normalized by foreign inflation)
            # If domestic inflation > foreign, domestic_inflation_idx > 1, REER > NEER (real appreciation)
            domestic_inflation_idx = (1 + self.state["domestic_inflation"].iloc[index] / 100) / \
                                   (1 + self.config.assumed_foreign_inflation_pct / 100)
            reer = 100.0 * domestic_inflation_idx
            self.state["real_effective_exchange_rate"].iloc[index] = reer
            return
        elif num_currencies_for_neer == 0:
             self.state["effective_exchange_rate"].iloc[index] = 100.0 # Default or warning
             self.state["real_effective_exchange_rate"].iloc[index] = 100.0
             logger.warning("NEER calculation skipped: No foreign currencies in basket or basket misconfigured.")
             return

        current_neer_sum = 0.0
        equal_weight = 1.0 / num_currencies_for_neer
        
        for currency_code in self.state["currencies"]:
            if currency_code == self.config.target_currency: # Skip the base currency (e.g. USD if rates are vs USD)
                continue
                
            current_rate = self.state["exchange_rates"][currency_code].iloc[index]
            initial_rate = self.state["exchange_rates"][currency_code].iloc[0]
            
            if initial_rate == 0: # Avoid division by zero if initial rate is 0
                rate_index_component = 100.0 # or handle as error/warning
            else:
                # Assuming rate is ForeignCur/DomesticCur: (initial / current) * 100 -> appreciation = index > 100
                # If rate is DomesticCur/ForeignCur: (current / initial) * 100 -> appreciation = index < 100. So use (initial / current) for consistency.
                rate_index_component = (initial_rate / current_rate) * 100 
            
            current_neer_sum += rate_index_component * equal_weight
            
        self.state["effective_exchange_rate"].iloc[index] = current_neer_sum
        
        # 2. Real effective exchange rate (REER) - adjusts for inflation differentials
        # REER = NEER * (Pd / Pf)
        # Pd = domestic price level index, Pf = foreign price level index
        # Using inflation rates: (1 + domestic_inflation) / (1 + foreign_inflation)
        
        domestic_inflation_factor = 1 + (self.state["domestic_inflation"].iloc[index] / 100.0)
        foreign_inflation_factor = 1 + (self.config.assumed_foreign_inflation_pct / 100.0)
        
        if foreign_inflation_factor == 0: # Avoid division by zero
            price_ratio = 1.0
        else:
            price_ratio = domestic_inflation_factor / foreign_inflation_factor
            
        reer = current_neer_sum * price_ratio
        
        self.state["real_effective_exchange_rate"].iloc[index] = reer
    
    def _update_reserves_adequacy_metrics(self, index):
        """Update metrics that measure the adequacy of foreign reserves."""
        # Get current total reserves
        reserves = self.state["total_reserves_usd"].iloc[index]
        
        # 1. Reserves to imports (in months)
        # For simplicity, use a synthetic imports trend
        monthly_imports = 1000 + (self.state["domestic_gdp_growth"].iloc[index] * 20)  # Millions USD
        reserves_to_imports = reserves / monthly_imports if monthly_imports > 0 else 0
        
        self.state["reserves_to_imports"].iloc[index] = reserves_to_imports
        
        # 2. Reserves to short-term debt
        # For simplicity, use a synthetic short-term debt trend
        short_term_debt = 10000 + (self.state["global_risk_aversion"].iloc[index] * 1000)  # Millions USD
        reserves_to_debt = reserves / short_term_debt if short_term_debt > 0 else 0
        
        self.state["reserves_to_short_term_debt"].iloc[index] = reserves_to_debt
        
        # 3. Reserves to M2 (broad money)
        # For simplicity, use synthetic M2 based on GDP growth and inflation
        m2 = 50000 * (1 + (self.state["domestic_gdp_growth"].iloc[index] / 100) + 
                      (self.state["domestic_inflation"].iloc[index] / 100))  # Millions domestic currency
        reserves_to_m2 = reserves / m2 if m2 > 0 else 0
        
        self.state["reserves_to_m2"].iloc[index] = reserves_to_m2
    
    def _prepare_results(self) -> Dict:
        """Prepare the simulation results for output."""
        return {
            "simulation_period": {
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "time_step": self.config.time_step
            },
            "exchange_rates": {
                "time_series": [
                    {
                        "date": date.isoformat(),
                        **{currency: float(self.state["exchange_rates"][currency][date])
                           for currency in self.state["currencies"]}
                    } for date in self.time_points
                ],
                "current": {
                    currency: float(self.state["exchange_rates"][currency].iloc[-1])
                    for currency in self.state["currencies"]
                },
                "percent_change": {
                    currency: float((self.state["exchange_rates"][currency].iloc[-1] / 
                                   self.state["exchange_rates"][currency].iloc[0] - 1) * 100)
                    for currency in self.state["currencies"]
                }
            },
            "effective_exchange_rates": {
                "time_series": [
                    {
                        "date": date.isoformat(),
                        "neer": float(self.state["effective_exchange_rate"][date]),
                        "reer": float(self.state["real_effective_exchange_rate"][date])
                    } for date in self.time_points
                ],
                "current": {
                    "neer": float(self.state["effective_exchange_rate"].iloc[-1]),
                    "reer": float(self.state["real_effective_exchange_rate"].iloc[-1])
                },
                "percent_change": {
                    "neer": float((self.state["effective_exchange_rate"].iloc[-1] / 
                                 self.state["effective_exchange_rate"].iloc[0] - 1) * 100),
                    "reer": float((self.state["real_effective_exchange_rate"].iloc[-1] / 
                                 self.state["real_effective_exchange_rate"].iloc[0] - 1) * 100)
                }
            },
            "foreign_reserves": {
                "time_series": [
                    {
                        "date": date.isoformat(),
                        "total_usd": float(self.state["total_reserves_usd"][date]),
                        "composition": {
                            currency: float(self.state["foreign_reserves"][currency][date])
                            for currency in self.state["currencies"]
                        }
                    } for date in self.time_points
                ],
                "current": {
                    "total_usd": float(self.state["total_reserves_usd"].iloc[-1]),
                    "composition": {
                        currency: float(self.state["foreign_reserves"][currency].iloc[-1])
                        for currency in self.state["currencies"]
                    }
                },
                "adequacy": {
                    "reserves_to_imports": float(self.state["reserves_to_imports"].iloc[-1]),
                    "reserves_to_short_term_debt": float(self.state["reserves_to_short_term_debt"].iloc[-1]),
                    "reserves_to_m2": float(self.state["reserves_to_m2"].iloc[-1])
                }
            },
            "balance_of_payments": {
                "time_series": [
                    {
                        "date": date.isoformat(),
                        "current_account": float(self.state["current_account"][date]),
                        "financial_account": float(self.state["financial_account"][date]),
                        "capital_account": float(self.state["capital_account"][date]),
                        "errors_omissions": float(self.state["errors_omissions"][date])
                    } for date in self.time_points
                ],
                "current": {
                    "current_account": float(self.state["current_account"].iloc[-1]),
                    "financial_account": float(self.state["financial_account"].iloc[-1]),
                    "capital_account": float(self.state["capital_account"].iloc[-1]),
                    "errors_omissions": float(self.state["errors_omissions"].iloc[-1])
                }
            },
            "interventions": [
                {
                    "date": index.isoformat(),
                    "amount_usd": float(row["amount_usd"]),
                    "type": row["type"],
                    "currency": row["currency"],
                    "result": row["result"]
                }
                for index, row in self.state["interventions"].dropna().iterrows()
            ],
            "market_conditions": {
                "time_series": [
                    {
                        "date": date.isoformat(),
                        "exchange_market_pressure": float(self.state["exchange_market_pressure"][date])
                    } for date in self.time_points
                ],
                "current": {
                    "exchange_market_pressure": float(self.state["exchange_market_pressure"].iloc[-1])
                }
            },
            "external_factors": {
                "global_risk_aversion": float(self.state["global_risk_aversion"].iloc[-1]),
                "commodity_price_index": float(self.state["commodity_price_index"].iloc[-1]),
                "trading_partner_growth": float(self.state["trading_partner_growth"].iloc[-1]),
                "us_interest_rate": float(self.state["us_interest_rate"].iloc[-1])
            }
        }
