import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import date, datetime

# Add the backend directory to sys.path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from app.simulation.config import load_default_config, FinancialStabilityConfig
    from app.simulation.financial_stability import FinancialStabilitySimulator
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure that the script is in the workspace root, the backend structure is correct (backend/app/...),")
    print("and all necessary __init__.py files are present in the packages.")
    sys.exit(1)

def json_default_handler(obj):
    # Custom JSON handler for types not serializable by default.
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        df_copy = obj.copy()
        try:
            if isinstance(df_copy.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                df_copy.index = df_copy.index.strftime('%Y-%m-%d')
        except AttributeError:
            pass
        return df_copy.to_dict(orient='split')
    if isinstance(obj, pd.Series):
        series_copy = obj.copy()
        try:
            if isinstance(series_copy.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                series_copy.index = series_copy.index.strftime('%Y-%m-%d')
        except AttributeError:
            pass
        return series_copy.to_dict()
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)

def main():
    print("Loading default financial stability configuration...")
    try:
        fs_config = load_default_config(simulation_type="financial_stability")
        if not isinstance(fs_config, FinancialStabilityConfig):
            print(f"Error: Expected FinancialStabilityConfig, but got {type(fs_config)}")
            sys.exit(1)
            
        # --- Enable and configure stress test for this run --- 
        fs_config.stress_test_enabled = True
        # Keep default stress timings and shock magnitudes for now, or adjust as needed:
        # fs_config.stress_test_start_delay_steps = 6 # Example: Start after 6 months if monthly
        # fs_config.stress_test_duration_steps = 6    # Example: Lasts for 6 months
        # fs_config.stress_gdp_shock_annual_pct_change = -5.0 # More severe GDP shock
        # fs_config.stress_unemployment_shock_abs_change = 3.0 # Higher unemployment jump
        # fs_config.stress_asset_price_shock_pct_change = -0.20 # Steeper asset price fall
        # --------

        # --- Enable Contagion Effects ---
        fs_config.enable_contagion_effects = True
        # fs_config.car_failure_threshold_pct = 3.0 # Optionally adjust threshold
        # fs_config.interbank_asset_recovery_rate_after_failure = 0.1 # Optionally adjust recovery rate
        # --------

    except ValueError as e:
        print(f"Error loading default config: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("Default configuration loaded successfully.")
    # print(fs_config.to_json(indent=2)) # Uncomment to see the full config
    
    print("\\nRunning Financial Stability simulation with default config (no database)...")
    try:
        # Instantiate the simulator
        simulator = FinancialStabilitySimulator(config=fs_config, db=None)
        
        # Run the simulation
        results = simulator.run_simulation()
        print("\\nSimulation completed successfully.")
        
        output_filename = "fs_results.json"
        print(f"Saving Financial Stability simulation results to {output_filename}...")
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2, default=json_default_handler)
        print(f"Results saved to {output_filename}")
        
    except Exception as e:
        print(f"\\nAn error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 