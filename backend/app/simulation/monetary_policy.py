"""
Monetary Policy Simulation Module.

This module implements the core simulation engine for monetary policy operations,
modeling the transmission mechanisms from policy rates to the broader economy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import date, datetime, timedelta
import logging
from sqlalchemy.orm import Session

from .config import MonetaryPolicyConfig
from .. import models

logger = logging.getLogger(__name__)


class MonetaryPolicySimulator:
    """
    Simulator for monetary policy operations and their economic effects.
    
    This class models the transmission of monetary policy changes through
    various channels to the broader economy, including effects on output,
    inflation, exchange rates, and the banking system.
    """
    
    def __init__(self, config: MonetaryPolicyConfig, db: Optional[Session] = None):
        """
        Initialize the monetary policy simulator.
        
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
            # Approximate a month as 30 days
            delta = timedelta(days=30)
        elif self.config.time_step == "quarterly":
            # Approximate a quarter as 90 days
            delta = timedelta(days=90)
        else:
            raise ValueError(f"Unknown time step: {self.config.time_step}")
        
        while current <= self.config.end_date:
            time_points.append(current)
            current += delta
        
        return time_points
    
    def _initialize_state(self) -> Dict[str, pd.Series]:
        """Initialize the state variables for the simulation."""
        # Create empty series for each state variable
        state = {}
        for var in [
            # Policy rates
            "policy_rate", "interbank_rate", "lending_rate", "deposit_rate",
            
            # Economic indicators
            "inflation", "core_inflation", "output", "output_gap", "potential_output",
            
            # Exchange rate
            "exchange_rate", "effective_exchange_rate",
            
            # Banking system
            "credit_growth", "deposit_growth", "excess_reserves", "liquidity_ratio",
            
            # Expectations
            "inflation_expectations", "output_expectations",
            
            # Shocks
            "supply_shock", "demand_shock", "external_shock"
        ]:
            state[var] = pd.Series(index=self.time_points, dtype=float)
        
        # Load initial conditions from database if available
        if self.db is not None:
            self._load_initial_conditions()
        else:
            # Set default initial values
            state["policy_rate"].iloc[0] = 5.0  # 5% policy rate
            state["interbank_rate"].iloc[0] = 5.2  # Slight spread over policy rate
            state["lending_rate"].iloc[0] = 9.0  # Higher lending rate
            state["deposit_rate"].iloc[0] = 3.0  # Lower deposit rate
            
            state["inflation"].iloc[0] = 5.5  # Current inflation
            state["core_inflation"].iloc[0] = 5.0  # Core inflation without volatile items
            state["output"].iloc[0] = 100.0  # Index value
            state["potential_output"].iloc[0] = 100.0  # At potential initially
            state["output_gap"].iloc[0] = 0.0  # No gap initially
            
            state["exchange_rate"].iloc[0] = 100.0  # Index value
            state["effective_exchange_rate"].iloc[0] = 100.0  # Index value
            
            state["credit_growth"].iloc[0] = 12.0  # Annual growth rate
            state["deposit_growth"].iloc[0] = 10.0  # Annual growth rate
            state["excess_reserves"].iloc[0] = 2.0  # As percentage of required reserves
            state["liquidity_ratio"].iloc[0] = self.config.banking_system_liquidity
            
            state["inflation_expectations"].iloc[0] = 5.0  # Expected inflation
            state["output_expectations"].iloc[0] = 2.0  # Expected growth rate
            
            # No shocks initially
            state["supply_shock"].iloc[0] = 0.0
            state["demand_shock"].iloc[0] = 0.0
            state["external_shock"].iloc[0] = 0.0
        
        return state
    
    def _load_initial_conditions(self):
        """Load initial conditions from the database."""
        if self.db is None:
            return
        
        # Example: Load the most recent policy rate
        latest_policy_rate = (
            self.db.query(models.PolicyRate)
            .filter(models.PolicyRate.rate_type == models.PolicyRateType.REPO)
            .order_by(models.PolicyRate.effective_date.desc())
            .first()
        )
        
        if latest_policy_rate:
            self.state["policy_rate"].iloc[0] = latest_policy_rate.rate_value
        
        # Example: Load the most recent inflation data
        latest_cpi = (
            self.db.query(models.CPI)
            .order_by(models.CPI.date.desc())
            .first()
        )
        
        if latest_cpi:
            self.state["inflation"].iloc[0] = latest_cpi.headline_inflation_yoy
            self.state["core_inflation"].iloc[0] = latest_cpi.core_inflation_yoy
        
        # Add more data loading as needed
    
    def _get_step_adjustment_factor(self) -> float:
        """Calculates the factor to convert annual rates/magnitudes to per-step rates/magnitudes.
        E.g., if time_step is 'quarterly', this returns 4.0. So an annual rate should be divided by this factor."""
        if self.config.time_step == "daily":
            # Assuming 365 days for simplicity, could be more precise for financial calculations
            return 365.0
        elif self.config.time_step == "weekly":
            return 52.0
        elif self.config.time_step == "monthly":
            return 12.0
        elif self.config.time_step == "quarterly":
            return 4.0
        else: # Default to annual if time_step is not recognized or is 'annual'
            logger.warning(f"Unknown or annual time_step '{self.config.time_step}', defaulting adjustment factor to 1.0 for rates.")
            return 1.0
    
    def run_simulation(self) -> Dict[str, pd.Series]:
        """
        Run the full simulation from start date to end date.
        
        Returns:
            Dictionary of time series data for all state variables
        """
        # Loop through all time points after the initial one
        for i in range(1, len(self.time_points)):
            self.current_date = self.time_points[i]
            previous_date = self.time_points[i-1]
            
            # Get the current data point for each state variable
            current_state = {var: self.state[var].iloc[i-1] for var in self.state}
            
            # Calculate new policy rate using Taylor rule
            new_policy_rate = self.calculate_policy_rate(current_state)
            
            # Generate shocks for this time period
            shocks = self.generate_shocks()
            
            # Update state variables for this time period
            new_state = self.update_state(current_state, new_policy_rate, shocks)
            
            # Store the new values in the state dictionary
            for var, value in new_state.items():
                self.state[var].iloc[i] = value
        
        return self.state
    
    def calculate_policy_rate(self, current_state: Dict[str, float]) -> float:
        """
        Calculate the new policy rate using a Taylor rule.
        
        Args:
            current_state: Current values of all state variables. Assumes rates like inflation are annualized.
            
        Returns:
            New policy rate value (annualized).
        """
        # Get parameters from config
        inflation_target = self.config.inflation_target
        inflation_weight = self.config.inflation_weight
        output_gap_weight = self.config.output_gap_weight
        smoothing = self.config.interest_rate_smoothing
        
        # Get current values
        current_inflation = current_state["inflation"] # Assumed to be an annualized rate
        current_output_gap = current_state["output_gap"] # As a percentage deviation
        current_policy_rate = current_state["policy_rate"] # Annualized rate
        
        # Taylor rule: r_t = ρ * r_{t-1} + (1-ρ) * [r* + π_t + α(π_t - π*) + β(y_t)]
        # where r* is the neutral real rate, π_t is current inflation, π* is target inflation, y_t is output gap.
        # All rates (r*, π_t, π*) are annualized percentages.
        
        neutral_real_rate = self.config.neutral_real_rate_pct 
        inflation_gap = current_inflation - inflation_target
        
        # Target rate based on Taylor rule components (annualized)
        taylor_component = neutral_real_rate + current_inflation + \
                           inflation_weight * inflation_gap + \
                           output_gap_weight * current_output_gap
        
        # Apply interest rate smoothing
        new_policy_rate = smoothing * current_policy_rate + (1 - smoothing) * taylor_component
        
        # Ensure non-negative rate (zero lower bound)
        return max(0.0, new_policy_rate)
    
    def generate_shocks(self) -> Dict[str, float]:
        """Generate random shocks for the current time period."""
        if not self.config.enable_shocks:
            return {"supply_shock": 0.0, "demand_shock": 0.0, "external_shock": 0.0}
        
        # Generate normally distributed shocks
        supply_shock = np.random.normal(0, self.config.supply_shock_magnitude)
        demand_shock = np.random.normal(0, self.config.demand_shock_magnitude)
        external_shock = np.random.normal(0, self.config.external_shock_magnitude)
        
        return {
            "supply_shock": supply_shock,
            "demand_shock": demand_shock,
            "external_shock": external_shock
        }
    
    def update_state(self, 
                     current_state: Dict[str, float], 
                     new_policy_rate: float, # This is an annualized rate from calculate_policy_rate
                     shocks: Dict[str, float]) -> Dict[str, float]:
        """
        Update all state variables based on the new policy rate and shocks.
        
        Args:
            current_state: Current values of all state variables. Assumes rates like inflation, growth are annualized.
            new_policy_rate: Newly calculated policy rate (annualized).
            shocks: Random shocks for the current period (effects are per-step unless specified otherwise by config interpretation).
            
        Returns:
            Updated values for all state variables. Rates remain annualized where applicable (e.g. inflation, policy_rate).
        """
        new_state = current_state.copy()
        # Factor to convert annual percentage growth rates/effects to per-step percentage growth rates/effects
        step_adjustment_factor = self._get_step_adjustment_factor()
        
        # Update policy rate and related interest rates (all are annualized rates)
        new_state["policy_rate"] = new_policy_rate
        # Spreads are in basis points, convert to percentage points
        new_state["interbank_rate"] = new_policy_rate + (self.config.interbank_rate_spread_bps / 100.0) + np.random.normal(0, 0.05) # Small noise
        new_state["lending_rate"] = new_policy_rate + (self.config.lending_rate_policy_spread_bps / 100.0) + np.random.normal(0, 0.1) # Small noise
        new_state["deposit_rate"] = max(0, new_policy_rate + (self.config.deposit_rate_policy_margin_bps / 100.0) + np.random.normal(0, 0.1)) # Small noise
        
        # --- Banking system variables ---    
        # Policy rate change in this step (impact from change in annualized policy rate)
        
        policy_rate_change_annual_points = new_policy_rate - current_state["policy_rate"] 
        
        # Credit growth: current_state["credit_growth"] is an annualized rate.
        # Sensitivity (config) is pp change in annual credit growth for 1pp change in annual policy rate.
        credit_growth_impact_annual_points = self.config.credit_growth_interest_rate_sensitivity * policy_rate_change_annual_points
        # Shocks are typically per-step. Assume demand_shock directly affects the annualized rate's change for this step.
        new_state["credit_growth"] = current_state["credit_growth"] + credit_growth_impact_annual_points + shocks["demand_shock"] 
        
        # Deposit growth: current_state["deposit_growth"] is an annualized rate.
        deposit_rate_change_annual_points = new_state["deposit_rate"] - current_state["deposit_rate"]
        deposit_growth_from_rate_annual_points = self.config.deposit_growth_deposit_rate_sensitivity * deposit_rate_change_annual_points
        deposit_growth_from_gap_annual_points = self.config.deposit_growth_output_gap_sensitivity * current_state["output_gap"] # output_gap is % deviation
        # Change in annualized deposit growth. Base deposit growth is annual. Allow reversion to base.
        change_in_annual_deposit_growth = deposit_growth_from_rate_annual_points + deposit_growth_from_gap_annual_points + (shocks["demand_shock"] * 0.5) \
                                          - (current_state["deposit_growth"] - self.config.base_deposit_growth_annual_pct) * 0.1 # Mean reversion factor
        new_state["deposit_growth"] = current_state["deposit_growth"] + change_in_annual_deposit_growth
        
        # Liquidity ratio and excess reserves are levels (percentages).
        # Sensitivity (config) is change in level for 1pp change in annual policy rate.
        new_state["liquidity_ratio"] = current_state["liquidity_ratio"] + self.config.liquidity_ratio_interest_rate_sensitivity * policy_rate_change_annual_points + shocks["demand_shock"] * 0.2
        new_state["excess_reserves"] = max(0, current_state["excess_reserves"] + self.config.excess_reserves_interest_rate_sensitivity * policy_rate_change_annual_points + shocks["demand_shock"] * 0.3)
        
        # --- Output and Potential Output ---    
        # potential_output_growth_annual_pct is annual. Convert to per-step rate for updating the level.
        potential_output_growth_step_pct = self.config.potential_output_growth_annual_pct / step_adjustment_factor
        new_state["potential_output"] = current_state["potential_output"] * (1 + potential_output_growth_step_pct / 100.0)
        
        # Output growth: output_growth_base_annual_pct is annual. Convert to per-step.
        base_output_growth_step_pct = self.config.output_growth_base_annual_pct / step_adjustment_factor
        # Sensitivity (config) is impact on *annual* output growth for 1pp change in *annual* policy rate. 
        # So, impact on *step* growth is (sensitivity / N) * change_in_annual_policy_rate, where N is steps_per_year.
        output_growth_from_rate_step_pct = (self.config.output_growth_interest_rate_sensitivity / step_adjustment_factor) * policy_rate_change_annual_points
        # Output gap feedback (config) is impact on *annual* output growth. Convert to step impact.
        output_growth_from_gap_feedback_step_pct = (self.config.output_growth_output_gap_feedback / step_adjustment_factor) * current_state["output_gap"]
        
        # Shocks (demand_shock) are assumed to be per-step effects on growth rate.
        output_growth_step_pct_total = base_output_growth_step_pct + output_growth_from_rate_step_pct + output_growth_from_gap_feedback_step_pct + shocks["demand_shock"]
        new_state["output"] = current_state["output"] * (1 + output_growth_step_pct_total / 100.0)
        new_state["output_gap"] = 100 * (new_state["output"] / new_state["potential_output"] - 1)
        
        # --- Inflation --- (all rates like inflation, expectations are annualized percentages)
        # Change in *annualized* inflation rate due to various factors.
        # Shocks (supply_shock) assumed to directly impact annualized inflation for the current step's change calculation.
        inflation_change_annual_points = (
            self.config.inflation_expectation_feedback_coeff * (current_state["inflation_expectations"] - current_state["inflation"]) +\
            self.config.inflation_output_gap_sensitivity * current_state["output_gap"] +\
            self.config.inflation_interest_rate_sensitivity * policy_rate_change_annual_points +\
            shocks["supply_shock"] 
        )
        new_state["inflation"] = current_state["inflation"] + inflation_change_annual_points
        
        # Core inflation (annualized) also changes based on the change in total annualized inflation.
        core_inflation_change_annual_points = self.config.core_inflation_smoothing_factor * inflation_change_annual_points
        new_state["core_inflation"] = current_state["core_inflation"] + core_inflation_change_annual_points
        
        # --- Exchange Rate --- (exchange_rate is DomesticCurrencyPerForeignCurrency, increase = depreciation)
        # interest_rate_effect_on_exchange_rate (config): % change in ER for 1pp change in ANNUAL IR.
        # Positive policy_rate_change_annual_points (domestic rate up) -> appreciation -> ER decreases.
        exchange_rate_pct_change_due_to_policy = self.config.interest_rate_effect_on_exchange_rate * policy_rate_change_annual_points
        # shocks["external_shock"] is a direct % change for the current step (e.g., positive means depreciation pressure)
        total_exchange_rate_pct_change = exchange_rate_pct_change_due_to_policy - shocks["external_shock"] # Subtract if shock means depreciation
        
        new_state["exchange_rate"] = current_state["exchange_rate"] * (1 - total_exchange_rate_pct_change / 100.0) 
        new_state["effective_exchange_rate"] = current_state["effective_exchange_rate"] * (1 - (total_exchange_rate_pct_change * self.config.effective_exchange_rate_sensitivity_factor) / 100.0)
        
        # --- Expectations (all are annualized rates) --- 
        if self.config.expectations_formation == "adaptive":
            # new_state["inflation"] is the new annualized inflation
            new_state["inflation_expectations"] = (
                self.config.adaptive_expectations_weight_current * new_state["inflation"] +\
                (1 - self.config.adaptive_expectations_weight_current) * current_state["inflation_expectations"]
            )
            # output_expectations are for annual growth rate. Use the total per-step growth calculated, annualized.
            implied_annual_output_growth_from_step = output_growth_step_pct_total * step_adjustment_factor
            new_state["output_expectations"] = (
                self.config.adaptive_expectations_weight_current * implied_annual_output_growth_from_step + 
                (1 - self.config.adaptive_expectations_weight_current) * current_state["output_expectations"]
            )
        else: # Simplified "rational" - model-consistent would be more complex
            new_state["inflation_expectations"] = new_state["inflation"] + np.random.normal(0, self.config.rational_expectations_noise_std_dev_inflation)
            implied_annual_output_growth_from_step = output_growth_step_pct_total * step_adjustment_factor 
            new_state["output_expectations"] = implied_annual_output_growth_from_step + np.random.normal(0, self.config.rational_expectations_noise_std_dev_output)
        
        # Store shock values for the current period (these are the original shock values, not their compounded effects)
        new_state["supply_shock"] = shocks["supply_shock"]
        new_state["demand_shock"] = shocks["demand_shock"]
        new_state["external_shock"] = shocks["external_shock"]
        
        return new_state
    
    def get_results(self) -> Dict[str, pd.DataFrame]:
        """Get organized results from the simulation."""
        # Group results into categories
        policy_rates = pd.DataFrame({
            "Policy Rate": self.state["policy_rate"],
            "Interbank Rate": self.state["interbank_rate"],
            "Lending Rate": self.state["lending_rate"],
            "Deposit Rate": self.state["deposit_rate"]
        })
        
        economic_indicators = pd.DataFrame({
            "Inflation": self.state["inflation"],
            "Core Inflation": self.state["core_inflation"],
            "Output": self.state["output"],
            "Output Gap (%)": self.state["output_gap"],
            "Potential Output": self.state["potential_output"]
        })
        
        banking_system = pd.DataFrame({
            "Credit Growth (%)": self.state["credit_growth"],
            "Deposit Growth (%)": self.state["deposit_growth"],
            "Excess Reserves (%)": self.state["excess_reserves"],
            "Liquidity Ratio (%)": self.state["liquidity_ratio"]
        })
        
        exchange_rates = pd.DataFrame({
            "Exchange Rate Index": self.state["exchange_rate"],
            "Effective Exchange Rate Index": self.state["effective_exchange_rate"]
        })
        
        expectations = pd.DataFrame({
            "Inflation Expectations": self.state["inflation_expectations"],
            "Output Growth Expectations": self.state["output_expectations"]
        })
        
        shocks = pd.DataFrame({
            "Supply Shock": self.state["supply_shock"],
            "Demand Shock": self.state["demand_shock"],
            "External Shock": self.state["external_shock"]
        })
        
        return {
            "policy_rates": policy_rates,
            "economic_indicators": economic_indicators,
            "banking_system": banking_system,
            "exchange_rates": exchange_rates,
            "expectations": expectations,
            "shocks": shocks
        }
    
    def save_results_to_db(self) -> None:
        """Save simulation results to the database."""
        if self.db is None:
            logger.warning("No database session provided, cannot save results")
            return
        
        # Implementation would depend on database schema
        # This would create records for each time point in the appropriate tables
        logger.info("Saving simulation results to database")
        
        # Example: Save policy rate decisions
        for i, date_point in enumerate(self.time_points):
            # Only save policy rate changes
            if i > 0 and abs(self.state["policy_rate"].iloc[i] - self.state["policy_rate"].iloc[i-1]) > 0.001:
                policy_rate = models.PolicyRate(
                    effective_date=date_point,
                    rate_type=models.PolicyRateType.REPO,
                    rate_value=float(self.state["policy_rate"].iloc[i]),
                    previous_value=float(self.state["policy_rate"].iloc[i-1]),
                    change=float(self.state["policy_rate"].iloc[i] - self.state["policy_rate"].iloc[i-1]),
                    decision_date=date_point - timedelta(days=1),
                    announcement_date=date_point - timedelta(days=1),
                    rationale="Simulation generated policy rate change"
                )
                self.db.add(policy_rate)
        
        try:
            self.db.commit()
            logger.info("Successfully saved simulation results")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving simulation results: {e}")
    
    def run_policy_scenario(self, scenario_name: str, policy_changes: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Run a specific policy scenario with predetermined policy rate changes.
        
        Args:
            scenario_name: Name of the scenario
            policy_changes: List of dictionaries with date and new_rate keys
            
        Returns:
            Dictionary of resulting time series
        """
        logger.info(f"Running policy scenario: {scenario_name}")
        
        # Create a mapping of dates to policy rates
        policy_change_map = {datetime.strptime(change["date"], "%Y-%m-%d").date(): change["new_rate"] 
                             for change in policy_changes}
        
        # Reset simulation
        self.state = self._initialize_state()
        
        # Run simulation with predetermined policy changes
        for i in range(1, len(self.time_points)):
            self.current_date = self.time_points[i]
            previous_date = self.time_points[i-1]
            
            # Get the current data point for each state variable
            current_state = {var: self.state[var].iloc[i-1] for var in self.state}
            
            # Check if there's a policy change for this date
            if self.current_date in policy_change_map:
                new_policy_rate = policy_change_map[self.current_date]
                logger.info(f"Applying policy change on {self.current_date}: {new_policy_rate}%")
            else:
                # Calculate policy rate using Taylor rule
                new_policy_rate = self.calculate_policy_rate(current_state)
            
            # Generate shocks
            shocks = self.generate_shocks()
            
            # Update state
            new_state = self.update_state(current_state, new_policy_rate, shocks)
            
            # Store the new values
            for var, value in new_state.items():
                self.state[var].iloc[i] = value
        
        return self.get_results()


def run_simulation_from_config(config_json: str, db: Optional[Session] = None) -> Dict[str, pd.DataFrame]:
    """
    Run a monetary policy simulation from a JSON configuration.
    
    Args:
        config_json: JSON string with simulation configuration
        db: Database session (optional)
        
    Returns:
        Dictionary of resulting time series
    """
    # Parse configuration
    config = MonetaryPolicyConfig.from_json(config_json)
    
    # Create and run simulator
    simulator = MonetaryPolicySimulator(config, db)
    simulator.run_simulation()
    
    # Return results
    return simulator.get_results()
