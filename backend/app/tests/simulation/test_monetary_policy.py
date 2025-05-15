import unittest
import pandas as pd
from datetime import datetime
from backend.app.simulation.monetary_policy import MonetaryPolicySimulator
from backend.app.simulation.config import MonetaryPolicyConfig, DEFAULT_MONETARY_POLICY_CONFIG

class TestMonetaryPolicySimulator(unittest.TestCase):

    def setUp(self):
        """Set up a default MonetaryPolicyConfig and Simulator for each test."""
        self.config_dict = DEFAULT_MONETARY_POLICY_CONFIG.copy()
        # Ensure default time_step for consistent testing if not overridden
        if 'time_step' not in self.config_dict:
            self.config_dict['time_step'] = 'quarterly' # or 'annual', 'monthly'

        self.config = MonetaryPolicyConfig(**self.config_dict)
        
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 12, 31) # 1 year, 4 quarters if quarterly
        
        # Provide a minimal initial_state for the simulator to initialize
        self.initial_state_values = {
            "policy_rate": 2.5,
            "inflation": 2.0,
            "core_inflation": 2.0,
            "output_gap": 0.0,
            "exchange_rate": 100.0,
            "effective_exchange_rate": 100.0,
            "credit_growth": 5.0,
            "deposit_growth": 5.0,
            "interbank_rate": 2.7,
            "lending_rate": 6.5,
            "deposit_rate": 1.0,
            "liquidity_ratio": 20.0,
            "excess_reserves": 5.0,
            "output": 1000.0,
            "potential_output": 1000.0,
            "inflation_expectations": 2.0,
            "output_expectations": 2.0, # Annual growth expectation
            "supply_shock": 0.0,
            "demand_shock": 0.0,
            "external_shock": 0.0,
        }
        self.simulator = MonetaryPolicySimulator(
            config=self.config, 
            start_date=self.start_date, 
            end_date=self.end_date,
            initial_state_values=self.initial_state_values,
            db_session=None # No database for these unit tests
        )

    def test_simulator_initialization(self):
        """Test if the simulator initializes correctly."""
        self.assertIsNotNone(self.simulator)
        self.assertEqual(self.simulator.config, self.config)
        self.assertIn("policy_rate", self.simulator.state)
        self.assertEqual(self.simulator.state["policy_rate"].iloc[0], self.initial_state_values["policy_rate"])
        self.assertEqual(len(self.simulator.state["policy_rate"]), self.simulator.num_periods + 1)

    def test_get_step_adjustment_factor(self):
        """Test the _get_step_adjustment_factor method."""
        test_cases = {
            "annual": 1.0,
            "quarterly": 4.0,
            "monthly": 12.0,
            "weekly": 52.0,
            "daily": 365.0,
            "unknown_step": 1.0 # Default case
        }
        for step, expected_factor in test_cases.items():
            with self.subTest(time_step=step):
                self.simulator.config.time_step = step
                self.assertEqual(self.simulator._get_step_adjustment_factor(), expected_factor)
        # Reset to original for other tests
        self.simulator.config.time_step = self.config_dict.get('time_step', 'quarterly')

    def test_calculate_policy_rate_no_change(self):
        """Test policy rate calculation when inflation is at target and output gap is zero."""
        current_state = {
            "inflation": self.config.inflation_target,
            "output_gap": 0.0,
            "policy_rate": self.config.neutral_real_rate_pct + self.config.inflation_target
        }
        # With smoothing = 1, rate should be current_policy_rate
        # With smoothing = 0, rate should be neutral_real_rate + inflation_target
        # Taylor component = neutral_real_rate + current_inflation + infl_weight * (infl - target) + gap_weight * gap
        #                  = neutral_real_rate + target_inflation + 0 + 0
        expected_taylor_component = self.config.neutral_real_rate_pct + self.config.inflation_target
        
        # Test with full smoothing (rate should not change from current)
        self.simulator.config.interest_rate_smoothing = 1.0
        calculated_rate_full_smoothing = self.simulator.calculate_policy_rate(current_state)
        self.assertAlmostEqual(calculated_rate_full_smoothing, current_state["policy_rate"], places=5)

        # Test with no smoothing (rate should go to Taylor component)
        self.simulator.config.interest_rate_smoothing = 0.0
        calculated_rate_no_smoothing = self.simulator.calculate_policy_rate(current_state)
        self.assertAlmostEqual(calculated_rate_no_smoothing, expected_taylor_component, places=5)

        # Test with partial smoothing
        self.simulator.config.interest_rate_smoothing = 0.5
        expected_partial_smoothing = 0.5 * current_state["policy_rate"] + 0.5 * expected_taylor_component
        calculated_rate_partial_smoothing = self.simulator.calculate_policy_rate(current_state)
        self.assertAlmostEqual(calculated_rate_partial_smoothing, expected_partial_smoothing, places=5)
        
    def test_calculate_policy_rate_inflation_above_target(self):
        """Test policy rate calculation when inflation is above target."""
        current_policy = self.config.neutral_real_rate_pct + self.config.inflation_target
        current_state = {
            "inflation": self.config.inflation_target + 1.0, # 1pp above target
            "output_gap": 0.0,
            "policy_rate": current_policy 
        }
        self.simulator.config.interest_rate_smoothing = 0.0 # No smoothing for direct check
        
        # Expected Taylor component: neutral_real + current_inflation + inflation_weight * (inflation_gap) + output_gap_weight * output_gap
        expected_rate = (self.config.neutral_real_rate_pct + 
                         current_state["inflation"] + 
                         self.config.inflation_weight * 1.0 + 
                         self.config.output_gap_weight * 0.0)
        
        calculated_rate = self.simulator.calculate_policy_rate(current_state)
        self.assertAlmostEqual(calculated_rate, expected_rate, places=5)

if __name__ == '__main__':
    unittest.main()
