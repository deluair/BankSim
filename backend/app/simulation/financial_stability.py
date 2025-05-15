"""
Financial Stability Simulation Module.

This module implements simulation logic for financial stability analysis,
including systemic risk assessment, stress testing, and macroprudential policy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import date, datetime, timedelta
import logging
import networkx as nx
from sqlalchemy.orm import Session

from .config import FinancialStabilityConfig
from .. import models

logger = logging.getLogger(__name__)


class FinancialStabilitySimulator:
    """
    Simulator for financial stability assessment.
    
    This class models system-wide financial stability including:
    - Systemic risk indicators
    - Interconnections between financial institutions
    - Stress testing and scenario analysis
    - Macroprudential policy tools
    """
    
    def __init__(self, config: FinancialStabilityConfig, db: Optional[Session] = None):
        """
        Initialize the financial stability simulator.
        
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

        # If synthetic data is used, the interbank network needs to be built 
        # and scores recalculated now that self.state is fully initialized.
        if self.db is None:
            logger.debug("Database not provided, creating synthetic interbank network.")
            self._create_synthetic_interbank_network() # Builds network using self.state['banks'] and populates self.state['interbank_network']
            
            logger.debug("Recalculating SIFI and Vulnerability scores with network data for synthetic banks.")
            for bank_id, bank_data in self.state["banks"].items():
                bank_data["systemic_importance_score"] = self._calculate_synthetic_sifi_score(
                    bank_data, 
                    self.state["banks"], 
                    self.state["interbank_network"]
                )
                bank_data["vulnerability_score"] = self._calculate_synthetic_vulnerability_score(
                    bank_data, 
                    self.state["interbank_network"]
                )
    
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
        state = {
            "banks": {},
            "interbank_network": self._create_interbank_network(),
            "systemic_risk_indicators": {
                "sri": pd.Series(index=self.time_points, dtype=float),  # Systemic Risk Index
                "vulnerability": pd.Series(index=self.time_points, dtype=float),  # System vulnerability
                "concentration": pd.Series(index=self.time_points, dtype=float),  # Market concentration
                "interconnectedness": pd.Series(index=self.time_points, dtype=float),  # Network interconnectedness
                "credit_growth": pd.Series(index=self.time_points, dtype=float),  # Excessive credit growth
                "asset_price_bubble": pd.Series(index=self.time_points, dtype=float),  # Asset price bubble indicator
                "liquidity_risk": pd.Series(index=self.time_points, dtype=float),  # System-wide liquidity risk
            },
            "macro_variables": {
                "gdp_growth": pd.Series(index=self.time_points, dtype=float),
                "inflation": pd.Series(index=self.time_points, dtype=float),
                "unemployment": pd.Series(index=self.time_points, dtype=float),
                "asset_prices": pd.Series(index=self.time_points, dtype=float),
                "credit_to_gdp": pd.Series(index=self.time_points, dtype=float),
                "housing_price_index": pd.Series(index=self.time_points, dtype=float),
                "stock_market_index": pd.Series(index=self.time_points, dtype=float),
            },
            "macroprudential_tools": {
                "countercyclical_buffer": pd.Series(index=self.time_points, dtype=float),
                "lTV_ratio_cap": pd.Series(index=self.time_points, dtype=float),
                "dSTI_ratio_cap": pd.Series(index=self.time_points, dtype=float),
                "capital_surcharge_sifi": pd.Series(index=self.time_points, dtype=float),
                "liquidity_requirements": pd.Series(index=self.time_points, dtype=float),
            },
            "stress_test_results": { # Initialize stress test results structure
                "active": False,
                "start_step_index": -1,
                "end_step_index": -1,
                "scenario_applied": {},
                "impact_summary": { # Placeholder for key outcomes
                    "peak_sri_during_stress": np.nan,
                    "banks_failing_stress_count": 0,
                    "total_capital_shortfall_stress": 0.0
                },
                "macro_var_trajectory_during_stress": { # To store how macro vars actually evolved
                    "gdp_growth": [],
                    "unemployment": [],
                    "asset_prices": []
                }
            },
            "failed_banks_log": [] # Log for (bank_id, failure_date)
        }
        
        if self.db is not None:
            self._load_banks_from_db()
        else:
            # Create synthetic banks and assign to local state dict
            state["banks"] = self._create_synthetic_banks()
        
        # After banks are loaded or created in the local 'state' dictionary,
        # calculate SIFI and vulnerability scores.
        # If db is None (synthetic banks), state["interbank_network"] is still empty at this point.
        # These scores will be recalculated in __init__ after the network is built for the synthetic case.
        if self.db is None: 
            logger.debug("Calculating initial SIFI/Vulnerability for synthetic banks (pre-network).")
            for bank_id, bank_data in state["banks"].items():
                bank_data["systemic_importance_score"] = self._calculate_synthetic_sifi_score(bank_data, state["banks"], state["interbank_network"]) # Pass the current (empty) graph
                bank_data["vulnerability_score"] = self._calculate_synthetic_vulnerability_score(bank_data, state["interbank_network"]) # Pass the current (empty) graph
        elif self.db is not None: # For DB loaded banks, assume scores are calculated in _load_banks_from_db
             # If _load_banks_from_db doesn't calculate them or network, that's a separate issue.
             # For now, focusing on synthetic path.
             pass
        
        # Initialize macro variables with reasonable starting values
        state["macro_variables"]["gdp_growth"].iloc[0] = 3.0  # Annual growth rate
        state["macro_variables"]["inflation"].iloc[0] = 2.5
        state["macro_variables"]["unemployment"].iloc[0] = 5.0
        state["macro_variables"]["asset_prices"].iloc[0] = 100.0  # Index value
        state["macro_variables"]["credit_to_gdp"].iloc[0] = 80.0  # In percent
        state["macro_variables"]["housing_price_index"].iloc[0] = 100.0
        state["macro_variables"]["stock_market_index"].iloc[0] = 1000.0
        
        # Initialize macroprudential tools with reasonable starting values
        state["macroprudential_tools"]["countercyclical_buffer"].iloc[0] = 0.0  # In percent
        state["macroprudential_tools"]["lTV_ratio_cap"].iloc[0] = 85.0  # Loan-to-Value cap
        state["macroprudential_tools"]["dSTI_ratio_cap"].iloc[0] = 40.0  # Debt Service to Income cap
        state["macroprudential_tools"]["capital_surcharge_sifi"].iloc[0] = 1.0  # Additional capital for SIFIs
        state["macroprudential_tools"]["liquidity_requirements"].iloc[0] = 100.0  # Liquidity Coverage Ratio
        
        return state
    
    def _create_interbank_network(self) -> nx.DiGraph:
        """Create the interbank network for modeling institutional interconnections."""
        # Initialize an empty directed graph
        G = nx.DiGraph()
        
        # We'll populate this with bank data later
        # For now, create a placeholder with basic properties
        G.graph["total_exposure"] = 0.0
        G.graph["network_density"] = 0.0
        G.graph["largest_exposure"] = 0.0
        
        return G
    
    def _load_banks_from_db(self):
        """Load bank data from the database."""
        if self.db is None:
            return
        
        banks = self.db.query(models.Bank).filter(models.Bank.is_active == True).all()
        
        for bank in banks:
            # Get the latest balance sheet data
            latest_balance = (
                self.db.query(models.BankBalance)
                .filter(models.BankBalance.bank_id == bank.id)
                .order_by(models.BankBalance.report_date.desc())
                .first()
            )
            
            # Get the latest interbank exposures
            interbank_exposures = (
                self.db.query(models.InterbankExposure)
                .filter(models.InterbankExposure.creditor_bank_id == bank.id)
                .filter(models.InterbankExposure.report_date == 
                        self.db.query(func.max(models.InterbankExposure.report_date))
                        .filter(models.InterbankExposure.creditor_bank_id == bank.id)
                        .scalar_subquery())
                .all()
            )
            
            # Store bank data in state
            self.state["banks"][bank.id] = {
                "bank": bank,
                "balance": latest_balance,
                "systemic_importance_score": self._calculate_sifi_score(bank, latest_balance),
                "vulnerability_score": self._calculate_vulnerability_score(bank, latest_balance),
            }
            
            # Add bank to network
            self._add_bank_to_network(bank.id, interbank_exposures)
    
    def _create_synthetic_banks(self) -> Dict[int, Dict[str, Any]]:
        """Create synthetic banks for simulation when no database is available."""
        # This would replicate the bank creation logic from banking_supervision.py
        # For now, we'll create a simple representation
        
        banks_data = {}

        # Create different bank sizes
        bank_sizes = [
            ("Large", 5),
            ("Medium", 15),
            ("Small", 30)
        ]
        
        bank_id = 1
        for size, count in bank_sizes:
            for i in range(count):
                # Generate basic bank properties
                if size == "Large":
                    assets = np.random.uniform(50000, 200000)  # in millions
                    is_systemic = True
                elif size == "Medium":
                    assets = np.random.uniform(5000, 50000)
                    is_systemic = np.random.random() < 0.2  # 20% chance
                else:  # Small
                    assets = np.random.uniform(500, 5000)
                    is_systemic = False
                
                # Generate financial health indicators (random but size-correlated)
                health_prob = [0.2, 0.4, 0.3, 0.08, 0.02]  # Probabilities for Strong to Weak
                if size == "Small":
                    # Small banks more likely to be weaker
                    health_prob = [0.1, 0.3, 0.4, 0.15, 0.05]
                
                health = np.random.choice(
                    ["Strong", "Satisfactory", "Fair", "Marginal", "Weak"], 
                    p=health_prob
                )
                
                # Map health to metrics
                if health == "Strong":
                    capital_ratio = np.random.uniform(15, 20)
                    npl_ratio = np.random.uniform(0.5, 2)
                    liquidity_ratio = np.random.uniform(120, 150)
                elif health == "Satisfactory":
                    capital_ratio = np.random.uniform(12, 15)
                    npl_ratio = np.random.uniform(2, 4)
                    liquidity_ratio = np.random.uniform(100, 120)
                elif health == "Fair":
                    capital_ratio = np.random.uniform(10, 12)
                    npl_ratio = np.random.uniform(4, 7)
                    liquidity_ratio = np.random.uniform(90, 100)
                elif health == "Marginal":
                    capital_ratio = np.random.uniform(8, 10)
                    npl_ratio = np.random.uniform(7, 12)
                    liquidity_ratio = np.random.uniform(80, 90)
                else:  # Weak
                    capital_ratio = np.random.uniform(4, 8)
                    npl_ratio = np.random.uniform(12, 20)
                    liquidity_ratio = np.random.uniform(60, 80)
                
                # Store bank in state
                banks_data[bank_id] = {
                    "bank": {
                        "id": bank_id,
                        "name": f"{size} Bank {i+1}",
                        "total_assets": assets,
                        "is_systemically_important": is_systemic,
                        "size_category": size
                    },
                    "financial_metrics": {
                        "capital_adequacy_ratio": capital_ratio,
                        "npl_ratio": npl_ratio,
                        "liquidity_ratio": liquidity_ratio,
                        "rwa_ratio": np.random.uniform(50, 80)  # Risk-weighted assets to total assets
                    },
                    "systemic_importance_score": 0,  # Will be calculated later
                    "vulnerability_score": 0,  # Will be calculated later
                    "status": "active" # Initial status
                }
                
                bank_id += 1
        
        # After creating all banks, create the interbank network based on them
        # self._create_synthetic_interbank_network(banks_data) # Pass banks_data if needed by this method
        # For now, assuming _initialize_state handles network creation separately after banks are in state.

        return banks_data
    
    def _create_synthetic_interbank_network(self):
        """Create a synthetic interbank network for simulation."""
        G = self.state["interbank_network"]
        
        # Add all banks as nodes
        for bank_id, bank_data in self.state["banks"].items():
            G.add_node(bank_id, 
                       assets=bank_data["bank"]["total_assets"],
                       is_systemic=bank_data["bank"]["is_systemically_important"],
                       size_category=bank_data["bank"]["size_category"])
        
        # Now create edges (exposures between banks)
        # Larger banks are more likely to lend to many other banks
        # Smaller banks mainly borrow from larger banks
        
        total_system_assets = sum(data["bank"]["total_assets"] for data in self.state["banks"].values())
        
        for creditor_id, creditor_data in self.state["banks"].items():
            # Determine how many banks this one connects to based on size
            if creditor_data["bank"]["size_category"] == "Large":
                # Large banks lend to many others
                connection_ratio = np.random.uniform(0.5, 0.8)
            elif creditor_data["bank"]["size_category"] == "Medium":
                # Medium banks lend to some others
                connection_ratio = np.random.uniform(0.2, 0.5)
            else:  # Small
                # Small banks lend to few others
                connection_ratio = np.random.uniform(0.05, 0.2)
            
            potential_debtors = [bid for bid in self.state["banks"].keys() if bid != creditor_id]
            num_connections = max(1, int(len(potential_debtors) * connection_ratio))
            
            # Weighted selection - more likely to connect to larger banks
            weights = [self.state["banks"][bid]["bank"]["total_assets"] for bid in potential_debtors]
            total_weight = sum(weights)
            probs = [w/total_weight for w in weights]
            
            debtor_ids = np.random.choice(
                potential_debtors, 
                size=min(num_connections, len(potential_debtors)), 
                replace=False, 
                p=probs
            )
            
            for debtor_id in debtor_ids:
                # Calculate exposure size based on bank sizes
                creditor_assets = creditor_data["bank"]["total_assets"]
                debtor_assets = self.state["banks"][debtor_id]["bank"]["total_assets"]
                
                # Exposure as percentage of creditor's assets (larger for larger debtors)
                exposure_pct = np.random.uniform(0.01, 0.05) * (debtor_assets / total_system_assets * 10)
                exposure_amount = creditor_assets * exposure_pct
                
                # Add edge to the network
                G.add_edge(creditor_id, debtor_id, weight=exposure_amount)
        
        # Update network statistics
        num_possible_edges = len(G.nodes) * (len(G.nodes) - 1)
        G.graph["network_density"] = G.number_of_edges() / num_possible_edges if num_possible_edges > 0 else 0
        G.graph["total_exposure"] = sum(data["weight"] for _, _, data in G.edges(data=True))
        G.graph["largest_exposure"] = max([data["weight"] for _, _, data in G.edges(data=True)], default=0)
    
    def _calculate_synthetic_sifi_score(self, bank_data, all_banks_data, interbank_network_graph):
        """Calculate a systemic importance score for synthetic bank data."""
        # Systemically Important Financial Institution score
        # Based on size, interconnectedness, complexity, and substitutability
        
        score = 0
        
        # Size component (0-40 points)
        assets = bank_data["bank"]["total_assets"]
        total_system_assets = sum(data["bank"]["total_assets"] for data in all_banks_data.values())
        size_score = 40 * (assets / total_system_assets if total_system_assets > 0 else 0)
        
        # Interconnectedness component (0-30 points)
        G = interbank_network_graph
        bank_id = bank_data["bank"]["id"]
        
        if bank_id in G:
            # Outgoing edges (bank's lending to others)
            outgoing_exposures = sum(G[bank_id][v]["weight"] for v in G.successors(bank_id)) if G.successors(bank_id) else 0
            # Incoming edges (others' lending to this bank)
            incoming_exposures = sum(G[u][bank_id]["weight"] for u in G.predecessors(bank_id)) if G.predecessors(bank_id) else 0
            
            total_system_exposures = G.graph["total_exposure"]
            if total_system_exposures > 0:
                interconnectedness_score = 30 * ((outgoing_exposures + incoming_exposures) / total_system_exposures)
            else:
                interconnectedness_score = 0
        else:
            interconnectedness_score = 0
        
        # Complexity/Substitutability approximation (0-30 points)
        # In a real system, this would be based on detailed business line data
        # For our simulation, we'll use size category as a proxy
        if bank_data["bank"]["size_category"] == "Large":
            complexity_score = np.random.uniform(20, 30)
        elif bank_data["bank"]["size_category"] == "Medium":
            complexity_score = np.random.uniform(5, 20)
        else:  # Small
            complexity_score = np.random.uniform(0, 5)
        
        score = size_score + interconnectedness_score + complexity_score
        return min(100, score)  # Cap at 100

    def _calculate_synthetic_vulnerability_score(self, bank_data, interbank_network_graph):
        """Calculate a vulnerability score for synthetic bank data."""
        # How vulnerable the bank is to shocks
        
        score = 0
        
        # Financial health component (0-50 points)
        metrics = bank_data.get("financial_metrics", {})
        
        # Capital adequacy (higher is better -> lower vulnerability)
        car = metrics.get("capital_adequacy_ratio", 10)
        car_score = max(0, 25 * (1 - (car / 20)))  # 20% CAR -> 0 points, 0% CAR -> 25 points
        
        # NPL ratio (higher is worse -> higher vulnerability)
        npl = metrics.get("npl_ratio", 5)
        npl_score = min(15, 15 * (npl / 15))  # 15% NPL -> 15 points, 0% NPL -> 0 points
        
        # Liquidity (higher is better -> lower vulnerability)
        liquidity = metrics.get("liquidity_ratio", 100)
        liquidity_score = max(0, 10 * (1 - (liquidity / 150)))  # 150% LCR -> 0 points, 0% LCR -> 10 points
        
        financial_score = car_score + npl_score + liquidity_score
        
        # Interconnectedness vulnerability (0-50 points)
        G = interbank_network_graph
        bank_id = bank_data["bank"]["id"]
        
        if bank_id in G:
            # Concentration of funding sources
            incoming = list(G[u][bank_id]["weight"] for u in G.predecessors(bank_id)) if G.predecessors(bank_id) else []
            
            if incoming:
                max_incoming = max(incoming, default=0)
                funding_concentration = max_incoming / sum(incoming) if sum(incoming) > 0 else 0
                # Higher concentration is worse
                concentration_score = 25 * funding_concentration
            else:
                concentration_score = 0  # No incoming funding in the interbank market
            
            # Dependency on interbank funding
            total_assets = bank_data["bank"]["total_assets"]
            interbank_funding_ratio = sum(incoming) / total_assets if total_assets > 0 else 0
            
            # Higher dependency is worse up to a point
            dependency_score = 25 * min(1, interbank_funding_ratio * 5)  # Cap at 20% of assets
        else:
            concentration_score = 0
            dependency_score = 0 # Added to ensure it's defined
        
        score = financial_score + concentration_score + dependency_score # Modified to sum all parts
        return min(100, max(0, score)) # Ensure score is between 0-100

    def _calculate_intermediate_system_metrics(self, banks_copy, total_assets_overall): # Placeholder for the orphaned block
        # This method now encapsulates the previously orphaned block.
        # Parameters 'banks_copy' and 'total_assets_overall' are placeholders
        # based on the variables used in the block. Their actual origin
        # needs to be determined from the intended logic of run_simulation.

        failing_banks = [] # Placeholder
        capital_shortfall = 0 # Placeholder
        system_car = 0
        system_npl = 0
        
        # The following lines were part of the orphaned block.
        # Their correct context within the simulation loop needs to be verified.
        # For example, 'bank_assets' and 'metrics' would need to be defined in the loop.
        # This is a temporary fix to make the file syntactically correct.
        
        # Assuming a loop over banks_copy might have existed here:
        # for bank_data_in_loop in banks_copy:
        #     bank_assets = bank_data_in_loop.get("assets", 0) # Example
        #     metrics = bank_data_in_loop.get("financial_metrics", {}) # Example
        #     car = metrics.get("capital_adequacy_ratio", 10) # Example

        #     system_car += car * bank_assets
        #     system_npl += metrics.get("npl_ratio", 5) * bank_assets
    
        if total_assets_overall > 0: # Renamed total_assets to total_assets_overall to avoid conflict
            system_car /= total_assets_overall
            system_npl /= total_assets_overall
        
        return {
            "failing_banks": failing_banks,
            "capital_shortfall": capital_shortfall,
            "system_car": system_car,
            "system_npl": system_npl,
            "percent_failing": len(failing_banks) / len(banks_copy) * 100 if banks_copy else 0,
            "assets_failing": sum(bank.get('assets', 0) for bank in failing_banks) / total_assets_overall * 100 if total_assets_overall > 0 else 0
        }

    def _evolve_state_for_timestep(self, current_date: date, previous_date: Optional[date]):
        """
        Evolves macro variables and policy tools for the current time step.
        Applies stress shocks if stress testing is active and within the shock period.
        """
        if previous_date is None: # For the first time step, values are already initialized
            return

        # Determine current step index
        current_step_index = -1
        for idx, dt in enumerate(self.time_points):
            if dt == current_date:
                current_step_index = idx
                break
        
        # Check if stress test is active for this step
        apply_stress_shock_this_step = False
        if self.config.stress_test_enabled:
            # Ensure stress test configuration is valid before using it
            start_stress_idx = self.config.stress_test_start_delay_steps
            end_stress_idx = start_stress_idx + self.config.stress_test_duration_steps -1 # inclusive index

            if current_step_index >= start_stress_idx and current_step_index <= end_stress_idx:
                apply_stress_shock_this_step = True
                if not self.state["stress_test_results"]["active"]:
                    logger.info(f"Stress test starting at step {current_step_index} ({current_date}).")
                    self.state["stress_test_results"]["active"] = True
                    self.state["stress_test_results"]["start_step_index"] = start_stress_idx
                    self.state["stress_test_results"]["end_step_index"] = end_stress_idx
                    self.state["stress_test_results"]["scenario_applied"] = {
                        "gdp_shock_annual_pct_change": self.config.stress_gdp_shock_annual_pct_change,
                        "unemployment_shock_abs_change": self.config.stress_unemployment_shock_abs_change,
                        "asset_price_shock_pct_change": self.config.stress_asset_price_shock_pct_change,
                        "duration_steps": self.config.stress_test_duration_steps
                    }
            
            if current_step_index > end_stress_idx and self.state["stress_test_results"]["active"]:
                # Mark stress test as completed if it was active
                logger.info(f"Stress test period ended at step {end_stress_idx}.")
                # We could set active to False here, or keep it True to indicate it ran.
                # Let's keep active=True to know it was triggered.

        # Evolve Macro Variables
        # Calculate baseline evolution first
        base_macro_values_this_step = {}
        for key, series in self.state["macro_variables"].items():
            # Simple evolution: previous value + small random change (baseline)
            change = np.random.normal(0, 0.1) 
            evolved_value = series.loc[previous_date] * (1 + change / 100) # Small percentage change
            
            if pd.isna(evolved_value):
                evolved_value = series.loc[previous_date] if not pd.isna(series.loc[previous_date]) else self.config.get(f"initial_{key}", np.nan)
            base_macro_values_this_step[key] = evolved_value

        # Apply stress shocks if active for this step, adjusting the baseline values
        stressed_macro_values_this_step = base_macro_values_this_step.copy()
        if apply_stress_shock_this_step:
            # GDP Growth Shock: Additive to the baseline annualized rate for this step
            if "gdp_growth" in stressed_macro_values_this_step:
                base_gdp = stressed_macro_values_this_step["gdp_growth"]
                stressed_gdp = base_gdp + self.config.stress_gdp_shock_annual_pct_change
                stressed_macro_values_this_step["gdp_growth"] = stressed_gdp
                logger.debug(f"Stress GDP: Base={base_gdp:.2f}, ShockFactor={self.config.stress_gdp_shock_annual_pct_change}, Stressed={stressed_gdp:.2f}")

            # Unemployment Shock: Additive to the baseline rate for this step
            if "unemployment" in stressed_macro_values_this_step:
                base_unemp = stressed_macro_values_this_step["unemployment"]
                stressed_unemp = base_unemp + self.config.stress_unemployment_shock_abs_change
                stressed_macro_values_this_step["unemployment"] = stressed_unemp
                logger.debug(f"Stress Unemp: Base={base_unemp:.2f}, ShockFactor={self.config.stress_unemployment_shock_abs_change}, Stressed={stressed_unemp:.2f}")

            # Asset Price Shock: Multiplicative on the baseline index value for this step (remains as is)
            if "asset_prices" in stressed_macro_values_this_step:
                base_asset_price = stressed_macro_values_this_step["asset_prices"]
                stressed_asset_price = base_asset_price * (1 + self.config.stress_asset_price_shock_pct_change)
                stressed_macro_values_this_step["asset_prices"] = stressed_asset_price
                logger.debug(f"Stress AssetPrices: Base={base_asset_price:.2f}, ShockFactor={self.config.stress_asset_price_shock_pct_change}, Stressed={stressed_asset_price:.2f}")

        # Finalize new values and store
        for key, series in self.state["macro_variables"].items():
            new_value = stressed_macro_values_this_step.get(key, base_macro_values_this_step.get(key)) # Fallback if key somehow missed

            # Ensure non-negativity for certain variables if applicable
            if key in ["unemployment", "inflation", "asset_prices", "credit_to_gdp", "housing_price_index", "stock_market_index"]:
                new_value = max(0, new_value if pd.notna(new_value) else 0)
            series.loc[current_date] = new_value

            # Store trajectory during stress if active
            if apply_stress_shock_this_step and key in self.state["stress_test_results"]["macro_var_trajectory_during_stress"]:
                self.state["stress_test_results"]["macro_var_trajectory_during_stress"][key].append(
                    {"date": current_date.isoformat(), "value": new_value}
                )
        
        # Evolve Policy Tools (Placeholder: typically these don't change randomly)
        for key, series in self.state["macroprudential_tools"].items():
            # For now, keep policy tools constant or apply minimal change
            # In a real scenario, these would change based on policy rules or scenarios
            series.loc[current_date] = series.loc[previous_date] # Keep constant

    def _calculate_systemic_risk_indicators_for_timestep(self, current_date: date) -> Dict[str, float]:
        """
        Calculates systemic risk indicators for the current time step based on current bank states and network.
        """
        all_banks_data = list(self.state["banks"].values()) # Get current data for all banks
        num_banks = len(all_banks_data)
        sri_components = {}

        if not all_banks_data:
            # No banks, return NaNs or defaults
            sri_components = {
                "vulnerability": np.nan,
                "concentration": np.nan,
                "interconnectedness": np.nan,
                "credit_growth": self.state["macro_variables"]["credit_to_gdp"].get(current_date, np.nan),
                "asset_price_bubble": self.state["macro_variables"]["asset_prices"].get(current_date, np.nan) / 1000 if pd.notna(self.state["macro_variables"]["asset_prices"].get(current_date)) else np.nan,
                "liquidity_risk": np.nan,
            }
        else:
            # 1. Vulnerability: Average bank vulnerability score
            total_vulnerability_score = sum(b.get("vulnerability_score", np.nan) for b in all_banks_data if pd.notna(b.get("vulnerability_score")))
            valid_vulnerability_scores = sum(1 for b in all_banks_data if pd.notna(b.get("vulnerability_score")))
            sri_components["vulnerability"] = (total_vulnerability_score / valid_vulnerability_scores) / 100.0 if valid_vulnerability_scores > 0 else np.nan # Normalize to 0-1 scale

            # 2. Concentration: HHI based on current total assets
            total_system_assets = sum(b["bank"]["total_assets"] for b in all_banks_data if pd.notna(b["bank"]["total_assets"]))
            if total_system_assets > 0:
                asset_shares_sq = [((b["bank"]["total_assets"] / total_system_assets) ** 2) for b in all_banks_data if pd.notna(b["bank"]["total_assets"])]
                hhi = sum(asset_shares_sq)
                sri_components["concentration"] = hhi # HHI is typically 0 to 1 (or 0 to 10000 if shares are %)
            else:
                sri_components["concentration"] = np.nan

            # 3. Interconnectedness: Use network density (from initial graph for now)
            # This would be more dynamic if the network itself evolved during simulation
            interbank_graph = self.state["interbank_network"]
            sri_components["interconnectedness"] = interbank_graph.graph.get("network_density", np.nan)

            # 4. Credit Growth (from macro variables, as before)
            sri_components["credit_growth"] = self.state["macro_variables"]["credit_to_gdp"].get(current_date, np.nan) / 100.0 if pd.notna(self.state["macro_variables"]["credit_to_gdp"].get(current_date, np.nan)) else np.nan

            # 5. Asset Price Bubble (from macro variables, as before)
            asset_prices_val = self.state["macro_variables"]["asset_prices"].get(current_date, np.nan)
            sri_components["asset_price_bubble"] = asset_prices_val / 1000.0 if pd.notna(asset_prices_val) else np.nan # Example scaling

            # 6. Liquidity Risk: Average bank liquidity ratio
            total_liquidity_ratio = sum(b["financial_metrics"].get("liquidity_ratio", np.nan) for b in all_banks_data if pd.notna(b["financial_metrics"].get("liquidity_ratio")))
            valid_liquidity_ratios = sum(1 for b in all_banks_data if pd.notna(b["financial_metrics"].get("liquidity_ratio")))
            avg_liquidity_ratio = (total_liquidity_ratio / valid_liquidity_ratios) if valid_liquidity_ratios > 0 else np.nan
            # Convert to a risk score (e.g. 1 - (avg_liquidity_ratio / desired_max_liquidity_ratio_for_no_risk) )
            # For simplicity, let's use a proxy: if avg liquidity is 150%, risk is low (0). If 50%, risk is high (1).
            if pd.notna(avg_liquidity_ratio):
                sri_components["liquidity_risk"] = max(0, min(1, (150.0 - avg_liquidity_ratio) / (150.0 - 50.0))) 
            else:
                sri_components["liquidity_risk"] = np.nan
        
        # Calculate overall SRI (e.g., weighted average or more complex model)
        # Handle potential NaN values in components
        valid_components = [v for v in sri_components.values() if pd.notna(v)]
        overall_sri = np.mean(valid_components) if valid_components else np.nan

        # Update the state
        self.state["systemic_risk_indicators"]["sri"].loc[current_date] = overall_sri
        for key, value in sri_components.items():
            self.state["systemic_risk_indicators"][key].loc[current_date] = value
        
        return {"sri": overall_sri, **sri_components}

    def run_simulation(self) -> Dict:
        """
        Run the financial stability simulation over the configured period.
        """
        logger.info(f"Starting Financial Stability simulation from {self.config.start_date} to {self.config.end_date}")

        # Initial values for systemic risk indicators based on initial state
        # (The _initialize_state already sets up the first data point for macro and policy)
        # We need to calculate SRI for the first time point.
        initial_sri_metrics = self._calculate_systemic_risk_indicators_for_timestep(self.time_points[0])
        for key, value in initial_sri_metrics.items():
            self.state["systemic_risk_indicators"][key].loc[self.time_points[0]] = value

        previous_date = None
        for i, current_date in enumerate(self.time_points):
            if i == 0: # First time step, already initialized or calculated above
                previous_date = current_date
                continue

            logger.debug(f"Simulating step for date: {current_date}")
            
            # 1. Evolve system state (macro vars, policy tools)
            self._evolve_state_for_timestep(current_date, previous_date)
            
            # 2. Update bank states based on new macro environment / policies
            self._update_bank_financials(current_date, previous_date)
            
            # 2.5 Process bank failures and contagion effects
            self._process_bank_failures_and_contagion(current_date)
            
            # 3. (Future) Update interbank network if dynamic
            # self._update_interbank_network(current_date)
            
            # 4. Calculate systemic risk indicators for the current state
            self._calculate_systemic_risk_indicators_for_timestep(current_date)
            
            # 5. (Future) Apply stress tests if applicable for this timestep
            # self._apply_stress_tests_for_timestep(current_date)

            previous_date = current_date

        logger.info("Financial Stability simulation finished.")
        return self._prepare_results()

    def _update_bank_financials(self, current_date: date, previous_date: date):
        """
        Update financial metrics for all banks based on macro-economic changes and bank-specific factors.
        Recalculates SIFI and Vulnerability scores after updating financials.
        Skips updates for already failed banks.
        """
        if previous_date is None:
            logger.warning("_update_bank_financials called with no previous_date, skipping update for this step.")
            return

        time_step_days = (current_date - previous_date).days
        if time_step_days == 0: # Should not happen with proper time_points
            annualization_factor = 1.0
        else:
            annualization_factor = 365.0 / time_step_days

        # Get macro variables for current and previous periods
        gdp_growth_current = self.state["macro_variables"]["gdp_growth"].get(current_date, self.config.base_asset_growth_annual_pct)
        unemployment_current = self.state["macro_variables"]["unemployment"].get(current_date, 5.0) # Default if NaN
        unemployment_previous = self.state["macro_variables"]["unemployment"].get(previous_date, unemployment_current) # Use current if prev is NaN
        
        if pd.isna(gdp_growth_current):
             gdp_growth_current = self.config.base_asset_growth_annual_pct # Use base if current is NaN
        if pd.isna(unemployment_current):
            unemployment_current = unemployment_previous # If current is still NaN, use previous known

        for bank_id, bank_data in self.state["banks"].items():
            # Skip updates for already failed banks
            if bank_data.get("status") == "failed":
                continue 

            metrics = bank_data["financial_metrics"]
            old_total_assets = bank_data["bank"]["total_assets"]
            old_car = metrics.get("capital_adequacy_ratio", 10.0)
            old_npl_ratio = metrics.get("npl_ratio", 5.0)
            old_liquidity_ratio = metrics.get("liquidity_ratio", 100.0)
            rwa_to_assets_ratio = metrics.get("rwa_ratio", 70.0) / 100 # Default 70% RWA/Assets

            old_rwa = old_total_assets * rwa_to_assets_ratio
            old_capital_amount = (old_car / 100) * old_rwa
            old_npl_amount = (old_npl_ratio / 100) * old_total_assets
            
            # --- 1. Update Total Assets ---
            base_growth_rate_step = (self.config.base_asset_growth_annual_pct / 100) / annualization_factor
            gdp_adj_factor = (gdp_growth_current - self.config.base_asset_growth_annual_pct) / 100 # anpualized diff
            gdp_adj_growth_rate_step = (gdp_adj_factor * self.config.asset_growth_gdp_sensitivity) / annualization_factor
            
            total_asset_growth_rate_step = base_growth_rate_step + gdp_adj_growth_rate_step
            current_total_assets = old_total_assets * (1 + total_asset_growth_rate_step)
            current_total_assets = max(1.0, current_total_assets) # Ensure assets don't go to zero or negative
            bank_data["bank"]["total_assets"] = current_total_assets
            asset_change_abs = current_total_assets - old_total_assets
            asset_growth_pct_step = (asset_change_abs / old_total_assets) * 100 if old_total_assets > 0 else 0

            # --- 2. Update NPL Ratio & Amount ---
            unemployment_diff_step = (unemployment_current - unemployment_previous) # Already per step if macro vars are per step
            
            gdp_npl_effect_pp_step = (gdp_adj_factor * self.config.npl_gdp_sensitivity) / annualization_factor # Sensitivity is pp change for 1pp GDP change
            unemployment_npl_effect_pp_step = (unemployment_diff_step * self.config.npl_unemployment_sensitivity) / annualization_factor # Sensitivity is pp change for 1pp Unemp change
            base_npl_ratio_change_pp_step = (self.config.base_npl_change_annual_pct / 100) / annualization_factor

            # Change in NPL ratio (additive effects on percentage points)
            npl_ratio_change_pp = base_npl_ratio_change_pp_step - gdp_npl_effect_pp_step + unemployment_npl_effect_pp_step
            new_npl_ratio = old_npl_ratio + npl_ratio_change_pp * 100 # Convert pp change to ratio change
            new_npl_ratio = max(0.01, min(new_npl_ratio, 75.0)) # Bound NPL ratio (e.g. 0.01% to 75%)
            metrics["npl_ratio"] = new_npl_ratio
            current_npl_amount = (new_npl_ratio / 100) * current_total_assets
            increase_in_npl_amount = max(0, current_npl_amount - old_npl_amount)

            # --- 3. Update Capital Adequacy Ratio (CAR) ---
            # Profitability effect on capital
            profit_for_step = (self.config.base_roa_annual_pct / 100) * old_total_assets / annualization_factor
            
            # Provisioning effect on capital
            provision_for_step = increase_in_npl_amount * self.config.provision_coverage_new_npls
            
            change_in_capital = profit_for_step - provision_for_step
            current_capital_amount = old_capital_amount + change_in_capital
            
            # New RWA (assuming RWA/Assets ratio is constant for now)
            current_rwa = current_total_assets * rwa_to_assets_ratio
            current_rwa = max(1.0, current_rwa) # Ensure RWA is not zero

            new_car = (current_capital_amount / current_rwa) * 100 if current_rwa > 0 else 0
            
            # Apply direct sensitivities from config as further adjustments (simplified)
            new_car += (new_npl_ratio - old_npl_ratio) * self.config.car_npl_sensitivity
            # Assuming ROA effect is on top of the calculated profit retention
            # This might double count, let's use car_profitability_sensitivity for direct impact on CAR from ROA deviation from a norm (norm assumed zero here)
            # effective_roa_for_step = (profit_for_step / old_total_assets) * annualization_factor * 100 if old_total_assets > 0 else 0
            # new_car += (effective_roa_for_step - self.config.base_roa_annual_pct) * self.config.car_profitability_sensitivity
            # Simpler: just use base ROA impact
            new_car += (self.config.base_roa_annual_pct * self.config.car_profitability_sensitivity) 

            new_car = max(0.0, min(new_car, 50.0)) # Bound CAR (e.g. 0% to 50%)
            metrics["capital_adequacy_ratio"] = new_car
            
            # --- 4. Update Liquidity Ratio ---
            asset_growth_liq_effect_pp = asset_growth_pct_step * self.config.liquidity_asset_growth_sensitivity
            npl_liq_effect_pp = (new_npl_ratio - old_npl_ratio) * self.config.liquidity_npl_sensitivity
            
            new_liquidity_ratio = old_liquidity_ratio + asset_growth_liq_effect_pp + npl_liq_effect_pp
            new_liquidity_ratio = max(10.0, min(new_liquidity_ratio, 300.0)) # Bound Liquidity Ratio (e.g. 10% to 300%)
            metrics["liquidity_ratio"] = new_liquidity_ratio

            # --- 5. Recalculate SIFI and Vulnerability Scores ---
            bank_data["systemic_importance_score"] = self._calculate_synthetic_sifi_score(
                bank_data, 
                self.state["banks"], 
                self.state["interbank_network"]
            )
            bank_data["vulnerability_score"] = self._calculate_synthetic_vulnerability_score(
                bank_data, 
                self.state["interbank_network"]
            )
            # Log updated bank data for one bank for one step for debugging
            # if bank_id == 1 and current_date == self.time_points[1]: # First bank, second timestep
            # logger.debug(f"Bank {bank_id} updated for {current_date}: Assets={bank_data['bank']['total_assets']:.2f}, NPL%={npl_metrics['npl_ratio']:.2f}, SIFI={bank_data['systemic_importance_score']:.2f}, Vuln={bank_data['vulnerability_score']:.2f}")

    def _process_bank_failures_and_contagion(self, current_date: date):
        """
        Processes bank failures based on CAR thresholds and applies contagion effects through interbank exposures.
        For simplicity, this version does one round of contagion per failing bank in a timestep.
        A more complex model might loop until no further contagion occurs in the step.
        """
        newly_failed_this_step = []

        # Phase 1: Identify new failures based on CAR
        for bank_id, bank_data in self.state["banks"].items():
            if bank_data.get("status") == "active": # Only consider active banks
                current_car = bank_data.get("financial_metrics", {}).get("capital_adequacy_ratio", np.nan)
                if pd.notna(current_car) and current_car < self.config.car_failure_threshold_pct:
                    bank_data["status"] = "failed"
                    self.state["failed_banks_log"].append({"bank_id": bank_id, "date": current_date.isoformat(), "reason": f"CAR fell to {current_car:.2f}%"})
                    newly_failed_this_step.append(bank_id)
                    logger.info(f"Bank {bank_id} failed on {current_date} due to CAR dropping to {current_car:.2f}%. Threshold: {self.config.car_failure_threshold_pct}%")
                    
                    # Optional: Set terminal values for failed bank, e.g., NPL to 100%
                    # bank_data["financial_metrics"]["npl_ratio"] = 100.0
                    # bank_data["financial_metrics"]["liquidity_ratio"] = 0.0 
                    # bank_data["bank"]["total_assets"] *= 0.5 # Example asset writedown

        # Phase 2: Process contagion if enabled and there are new failures
        if self.config.enable_contagion_effects and newly_failed_this_step:
            logger.debug(f"Processing contagion for {len(newly_failed_this_step)} newly failed banks on {current_date}.")
            interbank_graph = self.state["interbank_network"]
            loss_rate = 1.0 - self.config.interbank_asset_recovery_rate_after_failure

            for failed_bank_id in newly_failed_this_step:
                if not interbank_graph.has_node(failed_bank_id):
                    continue

                # Iterate over creditors of the failed bank
                for creditor_id in list(interbank_graph.predecessors(failed_bank_id)): # list() to avoid issues if graph changes
                    creditor_bank_data = self.state["banks"].get(creditor_id)
                    if not creditor_bank_data or creditor_bank_data.get("status") == "failed":
                        continue # Skip if creditor not found or already failed

                    exposure_edge = interbank_graph.get_edge_data(creditor_id, failed_bank_id)
                    if exposure_edge and "weight" in exposure_edge:
                        exposure_amount = exposure_edge["weight"]
                        loss_amount = exposure_amount * loss_rate
                        
                        logger.debug(f"Bank {creditor_id} faces loss of {loss_amount:.2f} from failed bank {failed_bank_id} (Exposure: {exposure_amount:.2f})")

                        # Apply loss to creditor's capital
                        metrics = creditor_bank_data["financial_metrics"]
                        old_car_creditor = metrics.get("capital_adequacy_ratio", np.nan)
                        total_assets_creditor = creditor_bank_data["bank"]["total_assets"]
                        rwa_ratio_creditor = metrics.get("rwa_ratio", 70.0) / 100.0
                        current_rwa_creditor = total_assets_creditor * rwa_ratio_creditor
                        
                        if pd.notna(old_car_creditor) and current_rwa_creditor > 0:
                            old_capital_amount_creditor = (old_car_creditor / 100.0) * current_rwa_creditor
                            new_capital_amount_creditor = old_capital_amount_creditor - loss_amount
                            
                            # Prevent capital from going excessively negative for CAR calculation stability
                            # new_capital_amount_creditor = max(new_capital_amount_creditor, -current_rwa_creditor) # Cap min capital at -100% RWA for CAR calc

                            new_car_creditor = (new_capital_amount_creditor / current_rwa_creditor) * 100.0 if current_rwa_creditor > 0 else 0.0
                            metrics["capital_adequacy_ratio"] = new_car_creditor
                            logger.info(f"Bank {creditor_id} CAR changed from {old_car_creditor:.2f}% to {new_car_creditor:.2f}% due to contagion loss from bank {failed_bank_id}.")
                            
                            # Recalculate vulnerability for the affected creditor bank
                            creditor_bank_data["vulnerability_score"] = self._calculate_synthetic_vulnerability_score(
                                creditor_bank_data,
                                interbank_graph
                            )
                            # SIFI score might also change if assets are significantly impacted or bank fails
                            # For now, focusing on CAR and vulnerability.

                            # Check if this creditor bank now fails due to contagion
                            if new_car_creditor < self.config.car_failure_threshold_pct and creditor_bank_data.get("status") == "active":
                                creditor_bank_data["status"] = "failed"
                                self.state["failed_banks_log"].append({"bank_id": creditor_id, "date": current_date.isoformat(), "reason": f"Contagion: CAR fell to {new_car_creditor:.2f}% after loss from bank {failed_bank_id}"})
                                # newly_failed_this_step.append(creditor_id) # Add to list for recursive contagion in same step (potential loop issue if not careful)
                                logger.info(f"Contagion: Bank {creditor_id} failed on {current_date} due to CAR dropping to {new_car_creditor:.2f}% after loss from bank {failed_bank_id}.")

    def _prepare_results(self) -> Dict:
        """Prepare the final results dictionary from the simulation state."""
        
        # Prepare time series data
        systemic_risk_ts = []
        for t_date in self.time_points:
            entry = {"date": t_date.isoformat()}
            components = {}
            is_any_sri_component_valid = False
            for key, series in self.state["systemic_risk_indicators"].items():
                value = series.get(t_date, np.nan) # Use .get for safety
                if key == "sri":
                    entry["sri"] = value
                else:
                    components[key] = value
                if pd.notna(value):
                    is_any_sri_component_valid = True
            entry["components"] = components
            # If all components and sri are NaN, maybe skip or represent appropriately.
            # For now, we add it as is, as the report handles NaN.
            systemic_risk_ts.append(entry)

        macro_variables_ts = []
        for t_date in self.time_points:
            entry = {"date": t_date.isoformat()}
            is_any_macro_valid = False
            for key, series in self.state["macro_variables"].items():
                value = series.get(t_date, np.nan)
                entry[key] = value
                if pd.notna(value):
                    is_any_macro_valid = True
            macro_variables_ts.append(entry)

        policy_tools_ts = []
        for t_date in self.time_points:
            entry = {"date": t_date.isoformat()}
            is_any_policy_valid = False
            for key, series in self.state["macroprudential_tools"].items():
                value = series.get(t_date, np.nan)
                entry[key] = value
                if pd.notna(value):
                    is_any_policy_valid = True
            policy_tools_ts.append(entry)
            
        # Current SRI (last valid point or overall average)
        # Find last non-NaN SRI or average
        sri_series = self.state["systemic_risk_indicators"]["sri"]
        last_valid_sri = sri_series[sri_series.notna()].iloc[-1] if sri_series.notna().any() else np.nan
        sri_trend = (sri_series.diff().mean()) if sri_series.notna().sum() > 1 else np.nan

        # Current Policy tools (values from the last time point)
        current_policy = {}
        if self.time_points:
            last_date = self.time_points[-1]
            for key, series in self.state["macroprudential_tools"].items():
                current_policy[key] = series.get(last_date, np.nan)
        else: # Should not happen if time_points are generated
            for key in self.state["macroprudential_tools"].keys():
                current_policy[key] = np.nan

        results = {
            "simulation_period": {
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "time_step": self.config.time_step,
            },
            "systemic_risk": {
                "time_series": systemic_risk_ts,
                "current": last_valid_sri,
                "trend": sri_trend, # Example: average change
            },
            "macro_variables": {
                "time_series": macro_variables_ts,
                # Could add current/summary stats for macro vars if needed
            },
            "policy_tools": {
                "time_series": policy_tools_ts,
                "current": current_policy,
            },
            "banks": [bank_data for bank_id, bank_data in self.state.get("banks", {}).items()], # Still using initial synthetic/loaded banks
            "interbank_network": { # Still using initial network
                "density": self.state["interbank_network"].graph.get("network_density", 0.0),
                "total_exposure": self.state["interbank_network"].graph.get("total_exposure", 0.0),
                "largest_exposure": self.state["interbank_network"].graph.get("largest_exposure", 0.0),
                "node_count": self.state["interbank_network"].number_of_nodes(),
                "edge_count": self.state["interbank_network"].number_of_edges(),
            },
            "stress_tests": self.state.get("stress_test_results", {}),
            "failed_banks_log": self.state.get("failed_banks_log", []) # Add failed banks log
        }

        # If stress test was active, calculate summary metrics
        stress_results = results["stress_tests"] # Get a reference to the dict
        if stress_results.get("active", False):
            start_idx = stress_results.get("start_step_index", -1)
            end_idx = stress_results.get("end_step_index", -1)
            
            if start_idx != -1 and end_idx != -1 and start_idx < len(self.time_points) and end_idx < len(self.time_points):
                stress_period_dates = self.time_points[start_idx : end_idx + 1]
                sri_during_stress = self.state["systemic_risk_indicators"]["sri"].loc[stress_period_dates]
                
                peak_sri = sri_during_stress.max() if not sri_during_stress.empty else np.nan
                stress_results["impact_summary"]["peak_sri_during_stress"] = peak_sri
                
                # Calculate bank failures and capital shortfall based on end-of-simulation state
                # if stress test was active. This is an approximation of stress impact.
                num_failing_banks = 0
                total_shortfall = 0.0
                regulatory_min_car = self.config.regulatory_min_car_pct

                for bank_id, bank_data in self.state["banks"].items(): # self.state["banks"] has end-of-simulation states
                    current_car = bank_data.get("financial_metrics", {}).get("capital_adequacy_ratio", np.nan)
                    if pd.notna(current_car) and current_car < regulatory_min_car:
                        num_failing_banks += 1
                        total_assets = bank_data.get("bank", {}).get("total_assets", 0)
                        rwa_ratio = bank_data.get("financial_metrics", {}).get("rwa_ratio", 70.0) # Default if not found
                        rwa = total_assets * (rwa_ratio / 100.0)
                        if rwa > 0:
                            shortfall_for_bank = rwa * (regulatory_min_car - current_car) / 100.0
                            total_shortfall += max(0, shortfall_for_bank) # Ensure shortfall is not negative if CAR > min_CAR due to float issues

                stress_results["impact_summary"]["banks_failing_stress_count"] = num_failing_banks
                stress_results["impact_summary"]["total_capital_shortfall_stress"] = total_shortfall
                
                logger.info(f"Stress test impact summary: Peak SRI={peak_sri:.2f}, Failing Banks (end of sim)={num_failing_banks}, Capital Shortfall (end of sim)={total_shortfall:.2f}")
            else:
                logger.warning("Stress test was marked active, but start/end indices are invalid for summary calculation.")

        return results