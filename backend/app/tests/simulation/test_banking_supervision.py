import pytest
from datetime import date, timedelta

from app.simulation.banking_supervision import BankingSupervisionSimulator
from app.simulation.config import BankingSupervisionConfig, BankConfig  # Assuming BankConfig might be needed for a fuller config

# A fixture to provide a basic BankingSupervisionConfig instance
@pytest.fixture
def basic_config():
    """Provides a basic BankingSupervisionConfig for testing."""
    return BankingSupervisionConfig(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        time_step="month",
        random_seed=42,
        # CAMELS thresholds - using the ones from our conceptual test scenario
        minimum_capital_adequacy_ratio = 8.0,
        conservation_buffer = 2.5, 
        car_rating_threshold_marginal = 8.5, 
        car_rating_threshold_fair = 10.5,       
        car_rating_threshold_good = 12.0, 
        car_rating_threshold_excellent = 13.5,
        npl_rating_threshold_excellent = 2.0,
        npl_rating_threshold_good = 5.0,
        npl_rating_threshold_fair = 8.0,
        npl_rating_threshold_marginal = 12.0,
        roa_rating_threshold_marginal = 0.25,
        roa_rating_threshold_fair = 0.5,
        roa_rating_threshold_good = 1.0,
        roa_rating_threshold_excellent = 1.5,
        minimum_liquidity_coverage_ratio = 100.0,
        lcr_rating_threshold_marginal = 80.0,
        lcr_rating_threshold_fair = 100.0,
        lcr_rating_threshold_good = 120.0,
        lcr_rating_threshold_excellent = 150.0,
        default_management_rating = 3,
        default_sensitivity_rating = 3,
        camels_component_weights = {'C': 0.20, 'A': 0.25, 'M': 0.15, 'E': 0.15, 'L': 0.15, 'S': 0.10},
        composite_rating_grade_thresholds = [1.5, 2.5, 3.5, 4.5]
        # Add other necessary minimal config parameters if simulator instantiation fails
    )

# A fixture to provide a BankingSupervisionSimulator instance with the basic_config
@pytest.fixture
def simulator(basic_config):
    """Provides a BankingSupervisionSimulator instance for testing."""
    # We might need to mock or provide minimal bank data if the __init__ expects it
    # For now, assuming __init__ can run with just the config for testing these specific methods
    return BankingSupervisionSimulator(config=basic_config)

# --- Tests for _get_rating_from_metric --- 

@pytest.mark.parametrize(
    "value, thresholds, higher_is_better, component_name, expected_rating",
    [
        # Higher is better (e.g., CAR: [8.5, 10.5, 12.0, 13.5])
        (7.0, [8.5, 10.5, 12.0, 13.5], True, "CAR Test Low", 5),
        (8.5, [8.5, 10.5, 12.0, 13.5], True, "CAR Test Threshold 4", 4),
        (10.0, [8.5, 10.5, 12.0, 13.5], True, "CAR Test Mid 4", 4),
        (10.5, [8.5, 10.5, 12.0, 13.5], True, "CAR Test Threshold 3", 3),
        (11.5, [8.5, 10.5, 12.0, 13.5], True, "CAR Test Mid 3", 3),
        (12.0, [8.5, 10.5, 12.0, 13.5], True, "CAR Test Threshold 2", 2),
        (13.0, [8.5, 10.5, 12.0, 13.5], True, "CAR Test Mid 2", 2),
        (13.5, [8.5, 10.5, 12.0, 13.5], True, "CAR Test Threshold 1", 1),
        (15.0, [8.5, 10.5, 12.0, 13.5], True, "CAR Test High", 1),
        # Lower is better (e.g., NPL: [2.0, 5.0, 8.0, 12.0])
        (1.0, [2.0, 5.0, 8.0, 12.0], False, "NPL Test Low", 1),
        (2.0, [2.0, 5.0, 8.0, 12.0], False, "NPL Test Threshold 1", 1),
        (3.0, [2.0, 5.0, 8.0, 12.0], False, "NPL Test Mid 2", 2),
        (5.0, [2.0, 5.0, 8.0, 12.0], False, "NPL Test Threshold 2", 2),
        (6.0, [2.0, 5.0, 8.0, 12.0], False, "NPL Test Mid 3", 3),
        (8.0, [2.0, 5.0, 8.0, 12.0], False, "NPL Test Threshold 3", 3),
        (10.0, [2.0, 5.0, 8.0, 12.0], False, "NPL Test Mid 4", 4),
        (12.0, [2.0, 5.0, 8.0, 12.0], False, "NPL Test Threshold 4", 4),
        (15.0, [2.0, 5.0, 8.0, 12.0], False, "NPL Test High", 5),
        # Edge case: None value
        (None, [2.0, 5.0, 8.0, 12.0], True, "None Value Test", 5),
    ]
)
def test_get_rating_from_metric_scenarios(simulator, value, thresholds, higher_is_better, component_name, expected_rating):
    """Test _get_rating_from_metric with various valid scenarios."""
    assert simulator._get_rating_from_metric(value, thresholds, higher_is_better, component_name) == expected_rating

@pytest.mark.parametrize(
    "value, thresholds, higher_is_better, component_name, expected_rating",
    [
        # Invalid thresholds
        (10.0, [1, 2, 3], True, "Short Thresholds", 5), # Expecting 5 due to error log and default
        (10.0, [1, 2, 3, 4, 5], True, "Long Thresholds", 5),
    ]
)
def test_get_rating_from_metric_invalid_thresholds(simulator, value, thresholds, higher_is_better, component_name, expected_rating, caplog):
    """Test _get_rating_from_metric with invalid threshold configurations."""
    # caplog is a pytest fixture to capture log output
    rating = simulator._get_rating_from_metric(value, thresholds, higher_is_better, component_name)
    assert rating == expected_rating
    assert "Thresholds list for" in caplog.text
    assert "must contain exactly 4 values" in caplog.text

# --- Tests for _calculate_detailed_camels ---

def test_calculate_detailed_camels_conceptual_scenario(simulator):
    """Test _calculate_detailed_camels using the detailed conceptual scenario.
    This scenario was manually traced in previous steps.
    """
    bank_id = "TestBank001"
    financial_metrics = {
        'capital_adequacy_ratio': 11.0,
        'npl_ratio': 6.0,
        'return_on_assets': 0.7,
        'liquidity_coverage_ratio': 90.0
    }
    # Simulate previous ratings for components not derived from current metrics
    previous_camels_ratings = {
        'management_rating': 2, # Assumed pre-existing or expert-judged
        'sensitivity_rating': 4  # Assumed pre-existing or expert-judged
    }
    # Set a fixed current date for the simulator instance for this test if not already set
    # This is important as 'rating_date' in the output depends on it.
    # If the fixture 'simulator' already has a fixed date via its config, this might not be needed.
    # For safety, let's assume we can set it if the simulator allows, or ensure fixture does.
    # simulator.current_date = date(2024, 1, 15) # Assuming current_date can be set or is fixed by fixture config

    expected_camels = {
        'capital_rating': 3,
        'asset_rating': 3,
        'management_rating': 2,
        'earnings_rating': 3,
        'liquidity_rating': 4,
        'sensitivity_rating': 4,
        'composite_rating': 3.10,
        'composite_grade': 3,
        'rating_date': simulator.current_date # This should match simulator's current date
    }

    actual_camels = simulator._calculate_detailed_camels(bank_id, financial_metrics, previous_camels_ratings)

    # Compare float value with tolerance
    assert actual_camels['composite_rating'] == pytest.approx(expected_camels['composite_rating'])
    
    # Compare the rest of the dictionary, excluding the float composite_rating for a moment
    actual_camels_no_float = {k: v for k, v in actual_camels.items() if k != 'composite_rating'}
    expected_camels_no_float = {k: v for k, v in expected_camels.items() if k != 'composite_rating'}
    assert actual_camels_no_float == expected_camels_no_float

def test_calculate_detailed_camels_all_excellent(simulator):
    """Test _calculate_detailed_camels with all excellent financial metrics."""
    bank_id = "ExcellentBank"
    financial_metrics = {
        'capital_adequacy_ratio': 15.0, # Rating 1
        'npl_ratio': 1.0,               # Rating 1
        'return_on_assets': 2.0,        # Rating 1
        'liquidity_coverage_ratio': 160.0 # Rating 1
    }
    previous_camels_ratings = {
        'management_rating': 1, 
        'sensitivity_rating': 1
    }
    
    expected_camels = {
        'capital_rating': 1,
        'asset_rating': 1,
        'management_rating': 1,
        'earnings_rating': 1,
        'liquidity_rating': 1,
        'sensitivity_rating': 1,
        'composite_rating': 1.0,
        'composite_grade': 1,
        'rating_date': simulator.current_date
    }

    actual_camels = simulator._calculate_detailed_camels(bank_id, financial_metrics, previous_camels_ratings)
    assert actual_camels['composite_rating'] == pytest.approx(expected_camels['composite_rating'])
    actual_camels_no_float = {k: v for k, v in actual_camels.items() if k != 'composite_rating'}
    expected_camels_no_float = {k: v for k, v in expected_camels.items() if k != 'composite_rating'}
    assert actual_camels_no_float == expected_camels_no_float

def test_calculate_detailed_camels_all_poor(simulator):
    """Test _calculate_detailed_camels with all poor financial metrics."""
    bank_id = "PoorBank"
    financial_metrics = {
        'capital_adequacy_ratio': 7.0,  # Rating 5
        'npl_ratio': 15.0,              # Rating 5
        'return_on_assets': 0.1,        # Rating 5
        'liquidity_coverage_ratio': 70.0 # Rating 5
    }
    previous_camels_ratings = {
        'management_rating': 5, 
        'sensitivity_rating': 5
    }
    
    expected_camels = {
        'capital_rating': 5,
        'asset_rating': 5,
        'management_rating': 5,
        'earnings_rating': 5,
        'liquidity_rating': 5,
        'sensitivity_rating': 5,
        'composite_rating': 5.0,
        'composite_grade': 5,
        'rating_date': simulator.current_date
    }

    actual_camels = simulator._calculate_detailed_camels(bank_id, financial_metrics, previous_camels_ratings)
    assert actual_camels['composite_rating'] == pytest.approx(expected_camels['composite_rating'])
    actual_camels_no_float = {k: v for k, v in actual_camels.items() if k != 'composite_rating'}
    expected_camels_no_float = {k: v for k, v in expected_camels.items() if k != 'composite_rating'}
    assert actual_camels_no_float == expected_camels_no_float

def test_calculate_detailed_camels_missing_lcr_fallback_liquidity_ratio(simulator):
    """Test LCR missing, fallback to liquidity_ratio (assumed good)."""
    bank_id = "FallbackBank1"
    financial_metrics = {
        'capital_adequacy_ratio': 11.0, 
        'npl_ratio': 6.0,
        'return_on_assets': 0.7,
        'liquidity_ratio': 130.0 # Should give rating 2 for Liquidity
    }
    previous_camels_ratings = {'management_rating': 3, 'sensitivity_rating': 3}
    # C=3, A=3, M=3, E=3, L=2, S=3
    # Score = (3*0.20)+(3*0.25)+(3*0.15)+(3*0.15)+(2*0.15)+(3*0.10) 
    #       = 0.60 + 0.75 + 0.45 + 0.45 + 0.30 + 0.30 = 2.85
    # Grade for 2.85 is 3
    expected_camels = {
        'capital_rating': 3,
        'asset_rating': 3,
        'management_rating': 3,
        'earnings_rating': 3,
        'liquidity_rating': 2, # Based on liquidity_ratio and LCR thresholds
        'sensitivity_rating': 3,
        'composite_rating': 2.85,
        'composite_grade': 3,
        'rating_date': simulator.current_date
    }
    actual_camels = simulator._calculate_detailed_camels(bank_id, financial_metrics, previous_camels_ratings)
    assert actual_camels['composite_rating'] == pytest.approx(expected_camels['composite_rating'])
    del actual_camels['composite_rating']
    del expected_camels['composite_rating']
    assert actual_camels == expected_camels

def test_calculate_detailed_camels_missing_all_liquidity_ratios(simulator):
    """Test LCR and liquidity_ratio missing, liquidity defaults to 5."""
    bank_id = "NoLiquidityBank"
    financial_metrics = {
        'capital_adequacy_ratio': 11.0, 
        'npl_ratio': 6.0,
        'return_on_assets': 0.7
        # No liquidity_coverage_ratio or liquidity_ratio
    }
    previous_camels_ratings = {'management_rating': 3, 'sensitivity_rating': 3}
    # C=3, A=3, M=3, E=3, L=5, S=3
    # Score = (3*0.20)+(3*0.25)+(3*0.15)+(3*0.15)+(5*0.15)+(3*0.10) 
    #       = 0.60 + 0.75 + 0.45 + 0.45 + 0.75 + 0.30 = 3.30
    # Grade for 3.30 is 3
    expected_camels = {
        'capital_rating': 3,
        'asset_rating': 3,
        'management_rating': 3,
        'earnings_rating': 3,
        'liquidity_rating': 5, # Default worst rating for missing liquidity info
        'sensitivity_rating': 3,
        'composite_rating': 3.30,
        'composite_grade': 3,
        'rating_date': simulator.current_date
    }
    actual_camels = simulator._calculate_detailed_camels(bank_id, financial_metrics, previous_camels_ratings)
    assert actual_camels['composite_rating'] == pytest.approx(expected_camels['composite_rating'])
    del actual_camels['composite_rating']
    del expected_camels['composite_rating']
    assert actual_camels == expected_camels

def test_calculate_detailed_camels_missing_m_s_ratings(simulator):
    """Test M and S missing from previous_camels_ratings, fallback to config defaults."""
    bank_id = "DefaultMSBank"
    financial_metrics = {
        'capital_adequacy_ratio': 11.0, # Rating 3
        'npl_ratio': 6.0,               # Rating 3
        'return_on_assets': 0.7,        # Rating 3
        'liquidity_coverage_ratio': 90.0 # Rating 4
    }
    previous_camels_ratings = {} # M and S are missing
    # Config default M=3, S=3
    # C=3, A=3, M=3, E=3, L=4, S=3
    # Score = (3*0.20)+(3*0.25)+(3*0.15)+(3*0.15)+(4*0.15)+(3*0.10) 
    #       = 0.60 + 0.75 + 0.45 + 0.45 + 0.60 + 0.30 = 3.15
    # Grade for 3.15 is 3
    expected_camels = {
        'capital_rating': 3,
        'asset_rating': 3,
        'management_rating': 3, # Default from config
        'earnings_rating': 3,
        'liquidity_rating': 4,
        'sensitivity_rating': 3, # Default from config
        'composite_rating': 3.15,
        'composite_grade': 3,
        'rating_date': simulator.current_date
    }
    actual_camels = simulator._calculate_detailed_camels(bank_id, financial_metrics, previous_camels_ratings)
    assert actual_camels['composite_rating'] == pytest.approx(expected_camels['composite_rating'])
    del actual_camels['composite_rating']
    del expected_camels['composite_rating']
    assert actual_camels == expected_camels

def test_calculate_detailed_camels_empty_inputs(simulator):
    """Test with empty financial_metrics and previous_camels_ratings."""
    bank_id = "EmptyInputBank"
    financial_metrics = {}
    previous_camels_ratings = {}
    # C=5 (no metric), A=5 (no metric), M=3 (config default), E=5 (no metric), L=5 (no metric), S=3 (config default)
    # Score = (5*0.20)+(5*0.25)+(3*0.15)+(5*0.15)+(5*0.15)+(3*0.10) 
    #       = 1.00 + 1.25 + 0.45 + 0.75 + 0.75 + 0.30 = 4.50
    # Grade for 4.50 is 4 (<=4.5)
    expected_camels = {
        'capital_rating': 5,
        'asset_rating': 5,
        'management_rating': 3, # Default from config
        'earnings_rating': 5,
        'liquidity_rating': 5,
        'sensitivity_rating': 3, # Default from config
        'composite_rating': 4.50,
        'composite_grade': 4, 
        'rating_date': simulator.current_date
    }
    actual_camels = simulator._calculate_detailed_camels(bank_id, financial_metrics, previous_camels_ratings)
    assert actual_camels['composite_rating'] == pytest.approx(expected_camels['composite_rating'])
    del actual_camels['composite_rating']
    del expected_camels['composite_rating']
    assert actual_camels == expected_camels

# --- Integration Tests (to be added later) --- 
