"""
Banking Supervision Simulation Module.

This module implements simulation logic for banking supervision activities,
including risk-based examinations, CAMELS ratings, and regulatory actions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import date, datetime, timedelta
import logging
from sqlalchemy.orm import Session

from .config import BankingSupervisionConfig
from .. import models
from ..models.banking_system import CAMELSComponent, Bank, BankType

logger = logging.getLogger(__name__)


class BankingSupervisionSimulator:
    """
    Simulator for banking supervision operations.
    
    This class models banking supervision activities including:
    - Risk-based examination scheduling
    - CAMELS ratings assessment
    - Regulatory actions based on bank conditions
    """
    
    def __init__(self, config: BankingSupervisionConfig, db: Optional[Session] = None):
        """
        Initialize the banking supervision simulator.
        
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
        state = {
            "banks": {},
            "examinations": {},
            "regulatory_actions": {},
            "system_health": pd.Series(index=self.time_points, dtype=float),
            "examination_schedule": {},
        }
        
        # Load banks from database if available
        if self.db is not None:
            self._load_banks_from_db()
        else:
            # Create synthetic banks for simulation
            state["banks"] = self._create_synthetic_banks()
        
        return state
    
    def _load_banks_from_db(self):
        """Load bank data from the database."""
        if self.db is None:
            return
        
        banks = self.db.query(models.Bank).filter(models.Bank.is_active == True).all()
        
        for bank in banks:
            # Get the latest CAMELS rating for each bank
            latest_rating = (
                self.db.query(models.CAMELSRating)
                .filter(models.CAMELSRating.bank_id == bank.id)
                .order_by(models.CAMELSRating.rating_date.desc())
                .first()
            )
            
            # Get the latest balance sheet data
            latest_balance = (
                self.db.query(models.BankBalance)
                .filter(models.BankBalance.bank_id == bank.id)
                .order_by(models.BankBalance.report_date.desc())
                .first()
            )
            
            # Get the latest loan portfolio data
            latest_portfolio = (
                self.db.query(models.LoanPortfolio)
                .filter(models.LoanPortfolio.bank_id == bank.id)
                .order_by(models.LoanPortfolio.report_date.desc())
                .first()
            )
            
            # Store bank data in state
            self.state["banks"][bank.id] = {
                "bank": bank,
                "camels_rating": latest_rating,
                "balance": latest_balance,
                "loan_portfolio": latest_portfolio,
                "last_examination_date": latest_rating.rating_date if latest_rating else None,
                "risk_profile": self._calculate_risk_profile(bank, latest_rating, latest_balance, latest_portfolio),
            }
    
    def _create_synthetic_banks(self) -> Dict[int, Dict[str, Any]]:
        """Create synthetic banks for simulation when no database is available."""
        banks_data: Dict[int, Dict[str, Any]] = {}

        # Create different types of banks with varying characteristics
        bank_types = [
            (BankType.COMMERCIAL, 15),
            (BankType.SPECIALIZED, 5),
            (BankType.ISLAMIC, 8),
            (BankType.FOREIGN, 7),
            (BankType.DEVELOPMENT, 3),
            (BankType.STATE_OWNED, 2)
        ]
        
        bank_id = 1
        for bank_type, count in bank_types:
            for i in range(count):
                # Generate basic bank properties
                name = f"{bank_type.value.title()} Bank {i+1}"
                size_category = np.random.choice(["Small", "Medium", "Large"], p=[0.5, 0.3, 0.2])
                
                # Size-based parameters
                if size_category == "Small":
                    assets = np.random.uniform(1000, 5000)  # in millions
                    branches = np.random.randint(5, 30)
                elif size_category == "Medium":
                    assets = np.random.uniform(5000, 20000)
                    branches = np.random.randint(30, 100)
                else:  # Large
                    assets = np.random.uniform(20000, 100000)
                    branches = np.random.randint(100, 300)
                
                # Generate financial health indicators
                health = np.random.choice(["Strong", "Satisfactory", "Fair", "Marginal", "Weak"], p=[0.1, 0.3, 0.4, 0.15, 0.05])
                
                # Map health to CAMELS composite rating
                if health == "Strong":
                    composite_rating = 1
                elif health == "Satisfactory":
                    composite_rating = 2
                elif health == "Fair":
                    composite_rating = 3
                elif health == "Marginal":
                    composite_rating = 4
                else:  # Weak
                    composite_rating = 5
                
                # Generate component ratings with some correlation to composite
                component_ratings = {}
                for component in ["capital", "asset", "management", "earnings", "liquidity", "sensitivity"]:
                    # Base rating on composite with some random variation
                    base = composite_rating
                    variation = np.random.randint(-1, 2)  # -1, 0, or 1
                    component_ratings[component] = max(1, min(5, base + variation))
                
                # Financial metrics based on health
                car = 16 - (composite_rating * 1.5) + np.random.uniform(-1, 1)  # CAR decreases with worse ratings
                npl_ratio = (composite_rating * 2) + np.random.uniform(-1, 1)  # NPL increases with worse ratings
                roa = 2.5 - (composite_rating * 0.5) + np.random.uniform(-0.2, 0.2)  # ROA decreases with worse ratings
                liquidity_ratio = 30 - (composite_rating * 3) + np.random.uniform(-2, 2)
                
                # Store synthetic bank in state
                banks_data[bank_id] = {
                    "bank": {
                        "id": bank_id,
                        "name": name,
                        "bank_type": bank_type,
                        "total_assets": assets,
                        "total_branches": branches,
                        "is_systemically_important": (size_category == "Large")
                    },
                    "camels_rating": {
                        "composite_rating": composite_rating,
                        "capital_rating": component_ratings["capital"],
                        "asset_rating": component_ratings["asset"],
                        "management_rating": component_ratings["management"],
                        "earnings_rating": component_ratings["earnings"],
                        "liquidity_rating": component_ratings["liquidity"],
                        "sensitivity_rating": component_ratings["sensitivity"],
                        "rating_date": self.current_date - timedelta(days=np.random.randint(30, 365))
                    },
                    "financial_metrics": {
                        "capital_adequacy_ratio": car,
                        "npl_ratio": npl_ratio,
                        "return_on_assets": roa,
                        "liquidity_ratio": liquidity_ratio
                    },
                    "last_examination_date": self.current_date - timedelta(days=np.random.randint(30, 365)),
                    "risk_profile": self._calculate_synthetic_risk_profile(composite_rating, size_category)
                }
                
                bank_id += 1
        return banks_data
    
    def _calculate_risk_profile(self, bank, rating, balance, portfolio):
        """Calculate a bank's risk profile based on its characteristics and financial data."""
        risk_score = 0
        
        # Higher CAMELS rating (worse condition) increases risk
        if rating:
            risk_score += rating.composite_rating * 5
        
        # Systemic importance increases risk
        if bank.is_systemically_important:
            risk_score += 10
        
        # Asset size affects risk
        if balance:
            # Large assets increase risk due to systemic impact
            if balance.total_assets > 50000:  # in millions
                risk_score += 10
            elif balance.total_assets > 10000:
                risk_score += 5
        
        # NPL ratio affects risk
        if portfolio and portfolio.npl_ratio:
            if portfolio.npl_ratio > 10:
                risk_score += 15
            elif portfolio.npl_ratio > 5:
                risk_score += 10
            elif portfolio.npl_ratio > 2:
                risk_score += 5
        
        return min(100, max(0, risk_score))
    
    def _calculate_synthetic_risk_profile(self, composite_rating, size_category):
        """Calculate a synthetic risk profile for simulation."""
        risk_score = composite_rating * 10  # Base on CAMELS
        
        # Adjust for bank size
        if size_category == "Large":
            risk_score += 10
        elif size_category == "Medium":
            risk_score += 5
        
        # Add some randomness
        risk_score += np.random.randint(-5, 6)
        
        return min(100, max(0, risk_score))
    
    def _get_rating_from_metric(self, value: Optional[float], thresholds: List[float], higher_is_better: bool = True, component_name: str = "Metric") -> int:
        """
        Maps a metric value to a 1-5 rating based on thresholds.
        Rating 1 is best, 5 is worst.

        Args:
            value: The metric value to rate. If None, returns 5 (worst rating).
            thresholds: A list of 4 thresholds. Order depends on higher_is_better.
                        If higher_is_better=True: [T1, T2, T3, T4] where T1<T2<T3<T4.
                            value < T1 -> 5; T1 <= value < T2 -> 4; ...; value >= T4 -> 1.
                        If higher_is_better=False: [T1, T2, T3, T4] where T1<T2<T3<T4.
                            value > T4 -> 5; T3 < value <= T4 -> 4; ...; value <= T1 -> 1.
            higher_is_better: True if higher metric values are better.
            component_name: Name of the component for logging.

        Returns:
            An integer rating from 1 to 5.
        """
        if value is None:
            logger.warning(f"Bank ID Not Specified: Missing value for CAMELS component '{component_name}', defaulting to rating 5.")
            return 5
        if len(thresholds) != 4:
            logger.error(f"Bank ID Not Specified: Thresholds list for '{component_name}' must contain exactly 4 values. Received: {thresholds}. Defaulting to rating 5.")
            return 5
        
        # Ensure thresholds are sorted for consistent logic based on how they are defined (lower bounds for good, upper bounds for bad)
        # For higher_is_better: thresholds are [T_for_rating4, T_for_rating3, T_for_rating2, T_for_rating1]
        # For lower_is_better: thresholds are [T_for_rating1, T_for_rating2, T_for_rating3, T_for_rating4]
        # The code below expects sorted_thresholds to be ascending for both cases.
        sorted_thresholds = sorted(thresholds) 

        if higher_is_better:
            if value >= sorted_thresholds[3]: return 1
            elif value >= sorted_thresholds[2]: return 2
            elif value >= sorted_thresholds[1]: return 3
            elif value >= sorted_thresholds[0]: return 4
            else: return 5
        else: # Lower is better
            if value <= sorted_thresholds[0]: return 1
            elif value <= sorted_thresholds[1]: return 2
            elif value <= sorted_thresholds[2]: return 3
            elif value <= sorted_thresholds[3]: return 4
            else: return 5

    def _calculate_detailed_camels(self, bank_id: Any, financial_metrics: Dict[str, Any],
                                   existing_camels_ratings: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Assess CAMELS rating for a given bank using financial data and configuration thresholds.
        
        Args:
            bank_id: Identifier for the bank (for logging).
            financial_metrics: Dict of financial values from bank_data['financial_metrics'], 
                               e.g., {'capital_adequacy_ratio': 12.0, 'npl_ratio': 2.5, ...}.
            existing_camels_ratings: Optional dict of previous raw component ratings (1-5 scale) from 
                                     bank_data['camels_rating'], e.g. {'capital_rating': 2, 'management_rating': 3, ...}.
                                     Used for M and S components if available, and potentially for persistence.
        Returns:
            Dict with CAMELS ratings: {'capital_rating': 1, ..., 'composite_rating': 1.0 (float), 'rating_date': date}
        """
        ratings_output = {} 

        if existing_camels_ratings is None:
            existing_camels_ratings = {}

        # C: Capital Adequacy (Higher is better)
        car_value = financial_metrics.get('capital_adequacy_ratio')
        car_base_min = self.config.minimum_capital_adequacy_ratio + self.config.conservation_buffer
        car_thresholds = sorted([
            getattr(self.config, 'car_rating_threshold_marginal', car_base_min - 2.0), 
            getattr(self.config, 'car_rating_threshold_fair', car_base_min),       
            getattr(self.config, 'car_rating_threshold_good', car_base_min + 1.5), 
            getattr(self.config, 'car_rating_threshold_excellent', car_base_min + 3.0) 
        ])
        ratings_output['capital_rating'] = self._get_rating_from_metric(car_value, car_thresholds, True, f'Bank {bank_id} Capital Adequacy')

        # A: Asset Quality (NPL ratio - Lower is better)
        npl_value = financial_metrics.get('npl_ratio')
        npl_thresholds = sorted([
            getattr(self.config, 'npl_rating_threshold_excellent', 2.0),
            getattr(self.config, 'npl_rating_threshold_good', 5.0),
            getattr(self.config, 'npl_rating_threshold_fair', 8.0),
            getattr(self.config, 'npl_rating_threshold_marginal', 12.0)
        ])
        ratings_output['asset_rating'] = self._get_rating_from_metric(npl_value, npl_thresholds, False, f'Bank {bank_id} Asset Quality')

        # M: Management Quality (Qualitative/from existing)
        ratings_output['management_rating'] = existing_camels_ratings.get('management_rating', getattr(self.config, 'default_management_rating', 3))

        # E: Earnings (ROA - Higher is better)
        roa_value = financial_metrics.get('return_on_assets')
        roa_thresholds = sorted([
            getattr(self.config, 'roa_rating_threshold_marginal', 0.25),
            getattr(self.config, 'roa_rating_threshold_fair', 0.5),
            getattr(self.config, 'roa_rating_threshold_good', 1.0),
            getattr(self.config, 'roa_rating_threshold_excellent', 1.5)
        ])
        ratings_output['earnings_rating'] = self._get_rating_from_metric(roa_value, roa_thresholds, True, f'Bank {bank_id} Earnings')

        # L: Liquidity (LCR - Higher is better)
        lcr_value = financial_metrics.get('liquidity_coverage_ratio') # Assuming LCR is in financial_metrics
        if lcr_value is None:
             lcr_value = financial_metrics.get('liquidity_ratio') # Fallback to generic liquidity_ratio
        lcr_base_min = self.config.minimum_liquidity_coverage_ratio
        lcr_thresholds = sorted([
            getattr(self.config, 'lcr_rating_threshold_marginal', lcr_base_min - 20.0),
            getattr(self.config, 'lcr_rating_threshold_fair', lcr_base_min),
            getattr(self.config, 'lcr_rating_threshold_good', lcr_base_min + 20.0),
            getattr(self.config, 'lcr_rating_threshold_excellent', lcr_base_min + 50.0)
        ])
        ratings_output['liquidity_rating'] = self._get_rating_from_metric(lcr_value, lcr_thresholds, True, f'Bank {bank_id} Liquidity')

        # S: Sensitivity to Market Risk (Qualitative/from existing)
        ratings_output['sensitivity_rating'] = existing_camels_ratings.get('sensitivity_rating', getattr(self.config, 'default_sensitivity_rating', 3))

        # Composite Rating (as a float score)
        component_values_for_composite = [
            ratings_output['capital_rating'], ratings_output['asset_rating'],
            ratings_output['management_rating'], ratings_output['earnings_rating'],
            ratings_output['liquidity_rating'], ratings_output['sensitivity_rating']
        ]
        
        valid_component_ratings = [r for r in component_values_for_composite if isinstance(r, (int, float))]

        if not valid_component_ratings:
            logger.error(f"Bank {bank_id}: No valid CAMELS components for composite rating. Defaulting to 5.0.")
            composite_score_numeric = 5.0
        else:
            weights = getattr(self.config, 'camels_component_weights', 
                              {'C': 0.20, 'A': 0.25, 'M': 0.15, 'E': 0.15, 'L': 0.15, 'S': 0.10})
            
            current_weighted_sum = 0
            current_sum_of_weights = 0
            component_to_weight_key = {
                'capital_rating': 'C', 'asset_rating': 'A', 'management_rating': 'M',
                'earnings_rating': 'E', 'liquidity_rating': 'L', 'sensitivity_rating': 'S'
            }

            for component_key_output, rating_value in ratings_output.items():
                weight_key = component_to_weight_key.get(component_key_output)
                if weight_key and weight_key in weights and isinstance(rating_value, (int, float)):
                    current_weighted_sum += rating_value * weights[weight_key]
                    current_sum_of_weights += weights[weight_key]
            
            if current_sum_of_weights > 0:
                composite_score_numeric = current_weighted_sum / current_sum_of_weights
            else: 
                logger.warning(f"Bank {bank_id}: Sum of weights for CAMELS is zero or weights misconfigured. Using simple average for composite score.")
                composite_score_numeric = np.mean(valid_component_ratings) if valid_component_ratings else 5.0
        
        ratings_output['composite_rating'] = round(composite_score_numeric, 2) 
        ratings_output['rating_date'] = self.current_date
        
        return ratings_output

    def run_simulation(self) -> Dict:
        """
        Run the full banking supervision simulation.
        
        Returns:
            Results of the simulation
        """
        # Process each time point
        for i, current_date in enumerate(self.time_points):
            self.current_date = current_date
            
            # Schedule examinations for this time point
            scheduled_exams = self._schedule_examinations()
            
            # Conduct the scheduled examinations
            examination_results = self._conduct_examinations(scheduled_exams)
            
            # Determine regulatory actions based on examination results
            regulatory_actions = self._determine_regulatory_actions(examination_results)
            
            # Update bank conditions
            self._update_bank_conditions()
            
            # Calculate system health metric
            self.state["system_health"].iloc[i] = self._calculate_system_health()
            
        return self._prepare_results()
    
    def _schedule_examinations(self) -> List[int]:
        """Schedule bank examinations based on risk profiles and time since last exam."""
        scheduled_exams = []
        
        for bank_id, bank_data in self.state["banks"].items():
            # Skip if this bank already has a scheduled examination
            if bank_id in self.state["examination_schedule"] and self.state["examination_schedule"][bank_id] >= self.current_date:
                continue
            
            last_exam_date = bank_data.get("last_examination_date")
            risk_profile = bank_data.get("risk_profile", 50)
            is_systemic = bank_data.get("bank", {}).get("is_systemically_important", False)
            
            # Determine examination frequency based on risk profile
            if risk_profile > self.config.risk_based_inspection_threshold * 15:  # High risk
                months_between_exams = self.config.high_risk_inspection_frequency_months
            else:  # Normal risk
                months_between_exams = self.config.inspection_frequency_months
            
            # Systemically important banks get examined more frequently
            if is_systemic:
                months_between_exams = max(6, months_between_exams - 3)
            
            # Check if it's time for an examination
            if last_exam_date is None or (self.current_date - last_exam_date).days >= months_between_exams * 30:
                scheduled_exams.append(bank_id)
                self.state["examination_schedule"][bank_id] = self.current_date
        
        return scheduled_exams
    
    def _conduct_examinations(self, scheduled_exams: List[int]) -> Dict[int, Dict]:
        """Conduct examinations for scheduled banks using detailed CAMELS assessment."""
        examination_results = {}

        for bank_id in scheduled_exams:
            bank_data = self.state["banks"].get(bank_id)
            if not bank_data:
                logger.warning(f"Bank data not found for bank_id {bank_id} during examination. Skipping.")
                continue

            financial_metrics = bank_data.get("financial_metrics", {})
            previous_camels_ratings = bank_data.get("camels_rating", {}) # Contains component ratings like 'capital_rating': 2
            previous_composite_grade = previous_camels_ratings.get("composite_grade", 
                                                                  previous_camels_ratings.get("composite_rating", 3)) # Fallback for old structure
            if not isinstance(previous_composite_grade, (int, float)):
                 previous_composite_grade = 3 # Default if unparseable

            # Calculate new CAMELS ratings using the detailed method
            new_camels_ratings = self._calculate_detailed_camels(
                bank_id=bank_id,
                financial_metrics=financial_metrics,
                existing_camels_ratings=previous_camels_ratings 
            )

            new_composite_grade = new_camels_ratings.get('composite_grade', 3) # This is the 1-5 grade
            
            # Record examination results
            # The _generate_examination_findings method expects keys like 'capital_rating', 'asset_rating', etc.
            # which _calculate_detailed_camels provides directly in its output dict.
            examination_results[bank_id] = {
                "bank_id": bank_id,
                "bank_name": bank_data.get("bank", {}).get("name", f"Bank {bank_id}"),
                "examination_date": self.current_date,
                "previous_composite_grade": int(round(previous_composite_grade)), # ensure it's an int for consistency
                "new_composite_grade": new_composite_grade, # This is already a 1-5 int rating
                "new_composite_score": new_camels_ratings.get('composite_rating'), # This is the float score
                "component_ratings": new_camels_ratings, # Pass the whole dict from _calculate_detailed_camels
                "findings": self._generate_examination_findings(bank_data, new_camels_ratings)
            }

            # Update bank data with the new comprehensive CAMELS ratings
            bank_data["camels_rating"] = new_camels_ratings 
            bank_data["last_examination_date"] = self.current_date
            
            # Update risk profile based on the new composite grade (1-5)
            bank_data["risk_profile"] = self._calculate_synthetic_risk_profile(
                composite_rating=new_composite_grade, # _calculate_synthetic_risk_profile expects a 1-5 rating
                size_category="Large" if bank_data.get("bank", {}).get("is_systemically_important", False) else (
                    "Medium" if bank_data.get("bank", {}).get("total_assets", 0) > 5000 else "Small"
                )
            )

            # Store examination in state
            if bank_id not in self.state["examinations"]:
                self.state["examinations"][bank_id] = []
            self.state["examinations"][bank_id].append(examination_results[bank_id])

        return examination_results
    
    def _generate_examination_findings(self, bank_data, ratings):
        """Generate examination findings based on CAMELS ratings."""
        findings = []
        
        # Generate findings based on poor ratings
        if ratings["capital_rating"] >= 4:
            findings.append("Inadequate capital levels for risk profile")
        
        if ratings["asset_rating"] >= 4:
            findings.append("High level of non-performing assets")
        
        if ratings["management_rating"] >= 4:
            findings.append("Weak management oversight and risk controls")
        
        if ratings["earnings_rating"] >= 4:
            findings.append("Insufficient earnings to support operations and maintain capital")
        
        if ratings["liquidity_rating"] >= 4:
            findings.append("Inadequate liquidity management and funding sources")
        
        if ratings["sensitivity_rating"] >= 4:
            findings.append("High sensitivity to market risk")
        
        # Add some random findings
        potential_findings = [
            "Documentation deficiencies in loan files",
            "Weaknesses in IT security controls",
            "Inadequate compliance with regulatory reporting requirements",
            "Concentrations in commercial real estate lending",
            "Insufficient loan loss reserves",
            "Weaknesses in internal audit function",
            "Inadequate business continuity planning",
            "Deficiencies in anti-money laundering program",
            "Non-compliance with consumer protection regulations",
            "Interest rate risk measurement deficiencies"
        ]
        
        # Add more random findings based on overall rating
        # Ensure composite_rating is treated as an integer for this calculation
        composite_rating_int = int(round(ratings.get("composite_rating", 3))) # Default to 3 if missing, round and cast to int
        num_random = max(0, min(5, composite_rating_int - 1))
        
        if num_random > 0:
            # Ensure num_random is explicitly an integer for np.random.choice size parameter
            random_findings = np.random.choice(potential_findings, size=int(num_random), replace=False)
            findings.extend(random_findings)
        
        return findings
        
    def _determine_regulatory_actions(self, examination_results: Dict[int, Dict]) -> Dict[int, List[str]]:
        """Determine regulatory actions based on examination results."""
        regulatory_actions = {}
        
        for bank_id, result in examination_results.items():
            bank_data = self.state["banks"][bank_id]
            new_composite = result["new_composite_grade"]
            actions = []
            
            # Determine actions based on CAMELS rating
            if new_composite == 3:
                actions.append("Memorandum of Understanding")
                actions.append("More frequent reporting requirements")
            elif new_composite == 4:
                actions.append("Written Agreement")
                actions.append("Restriction on certain activities")
                actions.append("Capital improvement plan required")
            elif new_composite == 5:
                actions.append("Cease and Desist Order")
                actions.append("Capital restoration plan required")
                actions.append("Management changes required")
                
                # Consider resolution for severely troubled banks
                if bank_data.get("financial_metrics", {}).get("capital_adequacy_ratio", 10) < 4:
                    actions.append("Resolution planning initiated")
            
            # Additional actions based on specific component ratings
            component_ratings = result["component_ratings"]
            
            if component_ratings.get("capital_rating", 1) >= 4:
                actions.append("Capital restoration plan required")
                
            if component_ratings.get("asset_rating", 1) >= 4:
                actions.append("Asset quality improvement plan required")
                actions.append("Increased loan loss reserves required")
                
            if component_ratings.get("management_rating", 1) >= 4:
                actions.append("Management improvement plan required")
                if component_ratings.get("management_rating", 1) == 5:
                    actions.append("Management changes recommended")
                    
            if component_ratings.get("liquidity_rating", 1) >= 4:
                actions.append("Liquidity management improvement required")
                actions.append("Contingency funding plan required")
            
            # Record actions if any were determined
            if actions:
                regulatory_actions[bank_id] = list(set(actions))  # Remove duplicates
                
                # Store in state
                if bank_id not in self.state["regulatory_actions"]:
                    self.state["regulatory_actions"][bank_id] = []
                
                self.state["regulatory_actions"][bank_id].append({
                    "date": self.current_date,
                    "actions": regulatory_actions[bank_id],
                    "examination_date": result["examination_date"],
                    "composite_rating": new_composite
                })
        
        return regulatory_actions
    
    def _update_bank_conditions(self):
        """Update bank conditions over time based on regulatory actions and market conditions."""
        for bank_id, bank_data in self.state["banks"].items():
            # Check if there are recent regulatory actions
            recent_actions = []
            if bank_id in self.state["regulatory_actions"]:
                for action_record in self.state["regulatory_actions"][bank_id]:
                    if (self.current_date - action_record["date"]).days <= 180:  # Actions in last 6 months
                        recent_actions.extend(action_record["actions"])
            
            # If there are regulatory actions, bank may improve (with some probability)
            if recent_actions:
                # More severe actions have higher probability of improvement
                severe_actions = sum(1 for action in recent_actions if "Cease and Desist" in action 
                                    or "restoration plan" in action or "changes required" in action)
                
                improvement_prob = min(0.7, 0.3 + (severe_actions * 0.1))  # Cap at 70%
                
                # Check if improvement occurs
                if np.random.random() < improvement_prob:
                    # Improve financial metrics
                    if "financial_metrics" in bank_data:
                        metrics = bank_data["financial_metrics"]
                        if "capital_adequacy_ratio" in metrics:
                            metrics["capital_adequacy_ratio"] += np.random.uniform(0.2, 1.0)
                        if "npl_ratio" in metrics:
                            metrics["npl_ratio"] = max(0.5, metrics["npl_ratio"] - np.random.uniform(0.3, 1.2))
                        if "return_on_assets" in metrics:
                            metrics["return_on_assets"] += np.random.uniform(0.05, 0.2)
            
            # Natural evolution of bank conditions (with some mean reversion)
            if "financial_metrics" in bank_data:
                metrics = bank_data["financial_metrics"]
                
                # Capital adequacy ratio mean reverts to regulatory minimum plus buffer
                if "capital_adequacy_ratio" in metrics:
                    target_car = 10.5  # Regulatory minimum plus buffer
                    mean_reversion = 0.1
                    random_shock = np.random.normal(0, 0.2)
                    metrics["capital_adequacy_ratio"] += (target_car - metrics["capital_adequacy_ratio"]) * mean_reversion + random_shock
                
                # NPL ratio has upward pressure with mean reversion
                if "npl_ratio" in metrics:
                    target_npl = 3.0  # Industry average
                    mean_reversion = 0.05
                    # Slight upward bias in random shocks
                    random_shock = np.random.normal(0.02, 0.15)
                    metrics["npl_ratio"] = max(0, metrics["npl_ratio"] + 
                                             (target_npl - metrics["npl_ratio"]) * mean_reversion + random_shock)
                
                # ROA mean reverts to industry average
                if "return_on_assets" in metrics:
                    target_roa = 1.0  # Industry average
                    mean_reversion = 0.1
                    random_shock = np.random.normal(0, 0.05)
                    metrics["return_on_assets"] += (target_roa - metrics["return_on_assets"]) * mean_reversion + random_shock
    
    def _calculate_system_health(self) -> float:
        """Calculate an overall banking system health index."""
        total_weight = 0
        weighted_health = 0
        
        for bank_id, bank_data in self.state["banks"].items():
            # Get bank's assets for weighting
            assets = bank_data.get("bank", {}).get("total_assets", 0)
            if assets == 0:
                continue
                
            # Calculate bank health based on CAMELS and financial metrics
            bank_health = 0
            
            # CAMELS ratings data
            camels_ratings_dict = bank_data.get("camels_rating", {})

            # Overall CAMELS score (using composite_grade, 1-5 scale)
            # Rating 1 is best, 5 is worst. Convert to 0-100 score (100 best).
            composite_grade = camels_ratings_dict.get("composite_grade", 3) # Default to 3 (Fair) if not found
            camels_score = 100 - ((composite_grade - 1) * 20)

            # Financial metrics scores derived from individual CAMELS component ratings (1-5 scale)
            # Default to rating 3 (60 score) if a specific component rating is missing.
            
            # Capital Adequacy Score (from capital_rating)
            capital_rating = camels_ratings_dict.get('capital_rating', 3)
            car_score = 100 - ((capital_rating - 1) * 20)

            # Asset Quality Score (from asset_rating, reflecting NPLs etc.)
            asset_rating = camels_ratings_dict.get('asset_rating', 3)
            npl_score = 100 - ((asset_rating - 1) * 20) # Higher asset_rating (worse) means lower npl_score

            # Earnings Score (from earnings_rating, reflecting ROA etc.)
            earnings_rating = camels_ratings_dict.get('earnings_rating', 3)
            roa_score = 100 - ((earnings_rating - 1) * 20)
            
            # Combine scores with weights
            # Weights: Composite CAMELS: 40%, Capital: 30%, Asset Quality: 20%, Earnings: 10%
            bank_health = (camels_score * 0.4) + (car_score * 0.3) + (npl_score * 0.2) + (roa_score * 0.1)
            
            # Add to system health with weight based on bank size
            weighted_health += bank_health * assets
            total_weight += assets
        
        return weighted_health / total_weight if total_weight > 0 else 50  # Default to middle value
    
    def _prepare_results(self) -> Dict:
        """Prepare the simulation results for output."""
        return {
            "simulation_period": {
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "time_step": self.config.time_step
            },
            "system_health": {
                "time_series": [
                    {
                        "date": date.isoformat(),
                        "health_index": float(self.state["system_health"][date])
                    } for date in self.time_points
                ],
                "current": float(self.state["system_health"].iloc[-1]),
                "trend": float(self.state["system_health"].iloc[-1] - self.state["system_health"].iloc[0])
            },
            "banks": [
                {
                    "id": bank_id,
                    "name": bank_data.get("bank", {}).get("name", f"Bank {bank_id}"),
                    "type": str(bank_data.get("bank", {}).get("bank_type", "")),
                    "total_assets": bank_data.get("bank", {}).get("total_assets", 0),
                    "is_systemic": bank_data.get("bank", {}).get("is_systemically_important", False),
                    "current_camels": bank_data.get("camels_rating", {}).get("composite_rating", 3),
                    "latest_exam_date": bank_data.get("last_examination_date", None),
                    "current_metrics": bank_data.get("financial_metrics", {}),
                    "risk_profile": bank_data.get("risk_profile", 50)
                } for bank_id, bank_data in self.state["banks"].items()
            ],
            "examinations": [
                examination 
                for bank_id, exams in self.state["examinations"].items() 
                for examination in exams
            ],
            "regulatory_actions": [
                {
                    "bank_id": bank_id,
                    "bank_name": self.state["banks"][bank_id].get("bank", {}).get("name", f"Bank {bank_id}"),
                    "date": action["date"].isoformat(),
                    "actions": action["actions"],
                    "composite_rating": action["composite_rating"]
                }
                for bank_id, actions in self.state["regulatory_actions"].items()
                for action in actions
            ]
        }
