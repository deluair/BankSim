"""
Simulation Configuration Module.

This module defines configuration parameters and settings for the various 
simulation engines in the BankSim platform.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
import json
from datetime import date, datetime, timedelta


class SimulationConfig(BaseModel):
    """Base configuration class for all simulations."""
    
    name: str
    description: str = ""
    start_date: date
    end_date: date
    time_step: str = "monthly"  # daily, weekly, monthly, quarterly
    random_seed: Optional[int] = None
    
    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert the configuration to a JSON string."""
        return json.dumps(self.model_dump(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SimulationConfig':
        """Create a configuration from a JSON string."""
        data = json.loads(json_str)
        # Convert date strings to date objects
        if 'start_date' in data and isinstance(data['start_date'], str):
            data['start_date'] = datetime.fromisoformat(data['start_date']).date()
        if 'end_date' in data and isinstance(data['end_date'], str):
            data['end_date'] = datetime.fromisoformat(data['end_date']).date()
        return cls(**data)


class MonetaryPolicyConfig(SimulationConfig):
    """Configuration for monetary policy simulations."""
    
    # Policy rule (e.g., Taylor rule)
    inflation_target: float = Field(2.0, description="Target inflation rate (%)")
    inflation_weight: float = Field(0.5, description="Weight on inflation gap in Taylor rule")
    output_gap_weight: float = Field(0.5, description="Weight on output gap in Taylor rule")
    interest_rate_smoothing: float = Field(0.7, description="Coefficient for interest rate smoothing (0 to 1)")
    neutral_real_rate_pct: float = Field(2.0, description="Neutral real interest rate for Taylor rule (%)")
    
    # Transmission lags (in simulation time steps)
    policy_rate_lag_inflation: int = Field(2, description="Lag for policy rate effect on inflation (e.g., 2 quarters)")
    policy_rate_lag_output: int = Field(1, description="Lag for policy rate effect on output (e.g., 1 quarter)")
    
    # Shock parameters
    enable_shocks: bool = Field(True, description="Enable random shocks in the simulation")
    supply_shock_magnitude: float = Field(0.5, description="Standard deviation of supply shocks")
    demand_shock_magnitude: float = Field(0.5, description="Standard deviation of demand shocks")
    external_shock_magnitude: float = Field(0.5, description="Standard deviation of external shocks (e.g., to exchange rate)")
    
    # Banking system parameters
    banking_system_liquidity: float = Field(20.0, description="Initial overall liquidity ratio in the banking system (%)")
    reserve_requirement_ratio: float = Field(5.0, description="Required reserve ratio (% of deposits)")
    interbank_rate_spread_bps: float = Field(20.0, description="Spread of interbank rate over policy rate in basis points (e.g., 20 for 0.2%)")
    lending_rate_policy_spread_bps: float = Field(400.0, description="Average spread of commercial lending rates over the policy rate in basis points (e.g., 400 for 4.0%)")
    deposit_rate_policy_margin_bps: float = Field(-200.0, description="Average margin of commercial deposit rates relative to the policy rate in basis points (e.g., -200 for policy rate - 2.0%)")
    credit_growth_interest_rate_sensitivity: float = Field(-1.5, description="Percentage point change in credit growth for a 1pp change in policy rate")
    deposit_growth_deposit_rate_sensitivity: float = Field(0.5, description="Sensitivity of deposit growth to changes in deposit rates")
    deposit_growth_output_gap_sensitivity: float = Field(0.3, description="Sensitivity of deposit growth to the output gap")
    base_deposit_growth_annual_pct: float = Field(6.0, description="Baseline annual deposit growth rate (%)")
    liquidity_ratio_interest_rate_sensitivity: float = Field(-0.8, description="Sensitivity of liquidity ratio to policy rate changes")
    excess_reserves_interest_rate_sensitivity: float = Field(-0.5, description="Sensitivity of excess reserves to policy rate changes")

    # Economic dynamics
    potential_output_growth_annual_pct: float = Field(2.0, description="Annual growth rate of potential output (%)")
    output_growth_base_annual_pct: float = Field(2.0, description="Baseline annual real output growth, absent shocks or deviations (%)")
    output_growth_interest_rate_sensitivity: float = Field(-0.8, description="Impact on output growth for a 1pp change in policy rate")
    output_growth_output_gap_feedback: float = Field(0.7, description="Feedback coefficient from output gap to output growth")
    inflation_expectation_feedback_coeff: float = Field(0.3, description="Weight of expected inflation in determining current inflation")
    inflation_output_gap_sensitivity: float = Field(0.2, description="Phillips curve slope: sensitivity of inflation to the output gap")
    inflation_interest_rate_sensitivity: float = Field(-0.1, description="Direct impact on inflation for a 1pp change in policy rate")
    core_inflation_smoothing_factor: float = Field(0.7, description="Smoothing factor for changes in core inflation (0 to 1)")

    # Exchange rate parameters
    interest_rate_effect_on_exchange_rate: float = Field(0.5, description="Effect of interest rate differential on exchange rate (% change in ER per 1pp change in IR)")
    effective_exchange_rate_sensitivity_factor: float = Field(0.8, description="Factor scaling the impact on effective exchange rate relative to bilateral ER change (0 to 1)")

    # Expectations formation
    expectations_formation: str = Field("adaptive", description="Method for forming expectations (adaptive or rational)")
    adaptive_expectations_weight_current: float = Field(0.8, description="Weight on current period's actual value for adaptive expectations (0 to 1)")
    rational_expectations_noise_std_dev_inflation: float = Field(0.2, description="Standard deviation for noise in simplified rational inflation expectations (%)")
    rational_expectations_noise_std_dev_output: float = Field(0.3, description="Standard deviation for noise in simplified rational output growth expectations (%)")

    # Bangladesh specific parameters (examples)
    remittance_sensitivity_to_exchange_rate: float = Field(0.2, description="Elasticity of remittances to exchange rate changes")


class BankingSupervisionConfig(SimulationConfig):
    """Configuration for banking supervision simulations."""
    
    # Capital requirements
    minimum_capital_adequacy_ratio: float = Field(10.0, description="Minimum capital adequacy ratio (%)")
    minimum_tier1_ratio: float = Field(6.0, description="Minimum Tier 1 capital ratio (%)")
    conservation_buffer: float = Field(2.5, description="Capital conservation buffer (%)")
    countercyclical_buffer: float = Field(0.0, description="Countercyclical capital buffer (%)")
    
    # Liquidity requirements
    minimum_liquidity_coverage_ratio: float = Field(100.0, description="Minimum liquidity coverage ratio (%)")
    minimum_net_stable_funding_ratio: float = Field(100.0, description="Minimum net stable funding ratio (%)")
    
    # Inspection parameters
    risk_based_inspection_threshold: float = Field(3.0, description="CAMELS rating threshold for increased supervision")
    inspection_frequency_months: int = Field(12, description="Months between regular inspections")
    high_risk_inspection_frequency_months: int = Field(6, description="Months between high-risk bank inspections")


class ForeignExchangeConfig(SimulationConfig):
    """Configuration for foreign exchange simulations."""
    
    currencies: List[str] = Field(
        default_factory=lambda: ["USD", "EUR", "GBP", "JPY", "CNY", "INR", "BDT"],
        description="List of relevant currency codes for the simulation (e.g., for cross-rates, reserves)"
    )
    # Exchange rate regime
    exchange_rate_regime: str = Field("managed_float", description="Exchange rate regime (fixed, managed_float, free_float)")
    target_currency: str = Field("USD", description="Primary target currency for exchange rate management")
    is_commodity_exporter: bool = Field(False, description="Whether the country is a net commodity exporter")
    assumed_foreign_inflation_pct: float = Field(2.0, description="Assumed average inflation rate of trading partners for REER calculation, in percent.")
    
    # Market characteristics
    fx_market_size: float = Field(500.0, description="Estimated size/turnover of the relevant FX market per simulation step (e.g., daily, in millions USD), used to scale intervention impact.")
    fx_volatility: Dict[str, float] = Field(
        default_factory=lambda: {"USD": 0.5, "EUR": 0.6, "GBP": 0.7, "JPY": 0.8, "CNY": 0.4, "OTHER": 1.0},
        description="Typical volatility (e.g., standard deviation as percentage) for each currency pair against the domestic currency, used for random noise in exchange rate model."
    )
    
    # Intervention parameters
    max_daily_intervention_pct: float = Field(0.5, description="Maximum daily intervention (% of reserves)")
    intervention_threshold_volatility: float = Field(1.0, description="Volatility threshold for intervention (std dev of price changes)")
    intervention_threshold_trend: float = Field(2.0, description="Trend deviation threshold for intervention (% deviation from target/moving average)")
    
    intervention_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"depreciation": 1.5, "appreciation": -1.0},
        description="Exchange Market Pressure (EMP) thresholds for triggering intervention. Positive for depreciation pressure, negative for appreciation."
    )
    max_intervention_size: Dict[str, float] = Field(
        default_factory=lambda: {"sell": 100.0, "buy": 100.0},
        description="Maximum size of a single intervention operation in millions USD (e.g., for selling or buying foreign currency)."
    )
    min_intervention_size: float = Field(10.0, description="Minimum size for an intervention to be considered, in millions USD.")
    
    # Reserve management
    minimum_reserve_months_imports: float = Field(3.0, description="Minimum reserves in months of imports coverage - an adequacy metric.")
    optimal_reserve_months_imports: float = Field(6.0, description="Optimal reserves in months of imports coverage - an adequacy metric.")
    minimum_reserves_threshold_pct: float = Field(
        0.20, 
        description="Minimum percentage of total reserves that should ideally not be used for interventions (e.g., 0.20 means 20% floor). Used in _determine_intervention to cap selling size."
    )
    reserve_currency_composition: Dict[str, float] = Field(
        default_factory=lambda: {"USD": 50.0, "EUR": 20.0, "GBP": 10.0, "JPY": 10.0, "CNY": 5.0, "OTHER": 5.0},
        description="Target reserve currency composition (%)"
    )


class FinancialStabilityConfig(SimulationConfig):
    """Configuration for financial stability simulations."""
    
    systemic_risk_threshold: float = Field(0.75, description="Threshold for systemic risk indicator to trigger alerts or actions.")
    
    # Bank financial evolution sensitivities
    asset_growth_gdp_sensitivity: float = Field(0.5, description="Sensitivity of bank asset growth to GDP growth (e.g., 0.5 means assets grow at 50% of GDP growth rate adjustment).")
    base_asset_growth_annual_pct: float = Field(2.0, description="Baseline annual asset growth for banks, in percent, before GDP adjustment.")
    
    npl_unemployment_sensitivity: float = Field(0.2, description="Increase in NPL ratio (pp) for a 1pp increase in unemployment rate.")
    npl_gdp_sensitivity: float = Field(0.1, description="Decrease in NPL ratio (pp) for a 1pp increase in GDP growth rate.")
    base_npl_change_annual_pct: float = Field(0.05, description="Baseline annual change in NPL ratio (pp), e.g., slight improvement or degradation.")

    car_npl_sensitivity: float = Field(-0.5, description="Change in CAR (pp) for a 1pp change in NPL ratio.")
    car_profitability_sensitivity: float = Field(0.1, description="Change in CAR (pp) due to 1pp of ROA (proxy for retained earnings impact).")
    
    liquidity_asset_growth_sensitivity: float = Field(-0.1, description="Change in liquidity ratio (pp) for a 1pp growth in assets (e.g., growth consumes some liquidity).")
    liquidity_npl_sensitivity: float = Field(-0.2, description="Change in liquidity ratio (pp) for a 1pp increase in NPL ratio (e.g., higher NPLs might strain liquidity perception/buffers).")

    base_roa_annual_pct: float = Field(1.0, description="Baseline annual Return on Assets for banks (%), used for capital generation.")
    provision_coverage_new_npls: float = Field(0.3, description="Fraction of new NPLs (increase in NPL amount) to be provisioned, affecting capital.")
    
    # Stress Test Parameters
    stress_test_enabled: bool = Field(False, description="Enable stress testing module.")
    stress_test_start_delay_steps: int = Field(12, description="Number of time steps after simulation start before stress scenario begins (e.g., 12 for 1 year if monthly steps).")
    stress_test_duration_steps: int = Field(4, description="Duration of the stress scenario in time steps (e.g., 4 for 1 quarter if monthly, or 1 if quarterly steps).")
    stress_gdp_shock_annual_pct_change: float = Field(-3.0, description="Additional annualized percentage point change to GDP growth during each stress step (e.g., -3.0 for an extra 3pp drop).")
    stress_unemployment_shock_abs_change: float = Field(2.0, description="Additional absolute percentage point change to unemployment rate during each stress step (e.g., +2.0 for an extra 2pp rise).")
    stress_asset_price_shock_pct_change: float = Field(-0.10, description="Percentage change to apply to the general asset price index at the start of each stress step (e.g., -0.10 for a 10% drop).")

    # Regulatory Minimums (relevant for stress impact)
    regulatory_min_car_pct: float = Field(8.0, description="Regulatory minimum Capital Adequacy Ratio (%), e.g., Basel III minimum.")

    # Bank Failure and Contagion Parameters
    car_failure_threshold_pct: float = Field(4.0, description="Capital Adequacy Ratio threshold below which a bank is considered to have failed during the simulation.")
    enable_contagion_effects: bool = Field(False, description="Enable contagion effects from bank failures through the interbank network.")
    interbank_asset_recovery_rate_after_failure: float = Field(0.2, description="Recovery rate (0.0 to 1.0) on interbank exposures to a bank that has failed. E.g., 0.2 means 20% recovery, 80% loss.")

    # Initial macro variable values (can be overridden by specific simulation setup)
    # These are already in _initialize_state, but could be config if desired for scenario definition
    # initial_gdp_growth_pct: float = Field(3.0, description="Initial annual GDP growth rate in percent.")
    # initial_inflation_pct: float = Field(2.5, description="Initial annual inflation rate in percent.")
    # initial_unemployment_pct: float = Field(5.0, description="Initial unemployment rate in percent.")


def load_default_config(simulation_type: str) -> SimulationConfig:
    """
    Load the default configuration for a given simulation type.
    
    Args:
        simulation_type: Type of simulation (monetary_policy, banking_supervision, foreign_exchange, financial_stability)
        
    Returns:
        A configuration object for the requested simulation type
    """
    today = date.today()
    start_date = today - timedelta(days=365)  # One year ago
    end_date = today + timedelta(days=730)    # Two years from now
    
    if simulation_type == "monetary_policy":
        return MonetaryPolicyConfig(
            name="Default Monetary Policy Simulation",
            description="Standard monetary policy simulation with Bangladesh parameters",
            start_date=start_date,
            end_date=end_date
        )
    elif simulation_type == "banking_supervision":
        return BankingSupervisionConfig(
            name="Default Banking Supervision Simulation",
            description="Standard banking supervision simulation with Bangladesh parameters",
            start_date=start_date,
            end_date=end_date
        )
    elif simulation_type == "foreign_exchange":
        return ForeignExchangeConfig(
            name="Default Foreign Exchange Simulation",
            description="Standard foreign exchange simulation with Bangladesh parameters",
            start_date=start_date,
            end_date=end_date
        )
    elif simulation_type == "financial_stability":
        return FinancialStabilityConfig(
            name="Default Financial Stability Simulation",
            description="Standard financial stability simulation with placeholder parameters",
            start_date=start_date,
            end_date=end_date,
            # Default values for new sensitivity params
            asset_growth_gdp_sensitivity=0.5,
            base_asset_growth_annual_pct=2.0,
            npl_unemployment_sensitivity=0.2,
            npl_gdp_sensitivity=0.1,
            base_npl_change_annual_pct=0.05,
            car_npl_sensitivity=-0.5,
            car_profitability_sensitivity=0.1,
            liquidity_asset_growth_sensitivity=-0.1,
            liquidity_npl_sensitivity=-0.2,
            base_roa_annual_pct=1.0,
            provision_coverage_new_npls=0.3,
            # Default values for stress test params
            stress_test_enabled=False,
            stress_test_start_delay_steps=12, # e.g., 1 year for monthly steps
            stress_test_duration_steps=4,    # e.g., 1 quarter for monthly steps
            stress_gdp_shock_annual_pct_change=-3.0,
            stress_unemployment_shock_abs_change=2.0,
            stress_asset_price_shock_pct_change=-0.10,
            regulatory_min_car_pct=8.0, # Default for regulatory minimum CAR
            # Default values for failure & contagion
            car_failure_threshold_pct=4.0,
            enable_contagion_effects=False,
            interbank_asset_recovery_rate_after_failure=0.2
        )
    else:
        raise ValueError(f"Unknown simulation type: {simulation_type}")
