import sys
import os
import json
import pandas as pd
import numpy as np # Added for numpy types
from datetime import date, datetime # Added for datetime types

# Add the backend directory to sys.path
# This assumes the script is run from the workspace root
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from app.simulation.config import load_default_config
    from app.simulation.monetary_policy import run_simulation_from_config
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
        # Convert DataFrame index to string if it's DatetimeIndex or PeriodIndex
        df_copy = obj.copy() # Operate on a copy to avoid modifying original DataFrame in results dict
        try:
            if isinstance(df_copy.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                df_copy.index = df_copy.index.strftime('%Y-%m-%d')
        except AttributeError:
            pass # Index might not have strftime or already be suitable
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
    # Fallback for other unhandled types
    # print(f"Warning: Type {type(obj)} not handled by json_default_handler, converting to string.")
    return str(obj)

def main():
    print("Loading default monetary policy configuration...")
    try:
        # Explicitly pass simulation_type as a keyword argument
        mp_config = load_default_config(simulation_type="monetary_policy")
    except ValueError as e:
        print(f"Error loading default config: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("Default configuration loaded successfully.")
    
    config_json_str = mp_config.to_json()
    
    print("\nRunning Monetary Policy simulation with default config (no database)...")
    try:
        results = run_simulation_from_config(config_json_str, db=None)
        print("\nSimulation completed successfully.")
        
        output_filename = "mp_results.json"
        print(f"Saving Monetary Policy simulation results to {output_filename}...")
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2, default=json_default_handler)
        print(f"Results saved to {output_filename}")
            
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 