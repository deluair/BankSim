import json
import pandas as pd
from pathlib import Path
import numpy as np # For checking np.nan
from typing import Dict, Any

def robust_json_load(file_path):
    # Loads JSON data, returning None if file not found or JSON is invalid.
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from - {file_path}")
        return None

def format_value(value):
    if pd.isna(value) or value is None: # Handles None, np.nan, pd.NA
        return "<span style='color: #999;'>N/A</span>"
    if isinstance(value, float):
        return f"{value:.2f}" # Format floats to 2 decimal places
    return str(value)

def dict_to_html_table(data_dict, table_id, title, is_fs_module=False):
    # Converts a dictionary (possibly with nested pandas DataFrames/Series) to an HTML table string.
    
    special_notes = {
        "Financial_Stability_Banks": "Bank details are currently placeholders in this simulation.",
        "Financial_Stability_Stress_Tests": "Stress test results are currently placeholders in this simulation.",
        "Financial_Stability_Systemic_Risk": "Systemic risk indicators are not yet fully implemented in this simulation; values may be N/A."
    }
    
    html = f"<h2>{title}</h2>\\n"
    
    # Check for special notes for FS module sections
    fs_note_key = f"{table_id.split('_')[0].title()}_{title.replace(' ', '_')}" # e.g., Financial_Stability_Systemic_Risk
    if is_fs_module and fs_note_key in special_notes:
        html += f"<p><em style='color: #e07a5f;'>Note: {special_notes[fs_note_key]}</em></p>\\n"

    if not isinstance(data_dict, dict):
        if isinstance(data_dict, pd.Series):
            # Handle pandas Series directly
            html += "<table class='table table-sm table-bordered'><thead><tr><th>Index</th><th>Value</th></tr></thead><tbody>"
            for idx, val in data_dict.items():
                html += f"<tr><td>{idx}</td><td>{format_value(val)}</td></tr>"
            html += "</tbody></table>"
        elif isinstance(data_dict, np.ndarray): # Handle NumPy arrays when data_dict itself is an array
            if data_dict.size == 0:
                html += "<p><em style='color: #999;'>Empty array.</em></p>"
            else:
                html += "<strong>Array Data:</strong><ul>"
                for item in data_dict.flatten(): # Flatten in case of multi-dimensional arrays
                    html += f"<li>{format_value(item)}</li>"
                html += "</ul>"
        elif isinstance(data_dict, list): # Handle all lists (empty or not)
            if not data_dict: # Empty list
                html += "<p><em style='color: #999;'>No data available for this section (empty list).</em></p>"
            else: # Non-empty list
                html += "<strong>List Data:</strong><ul>"
                for item in data_dict:
                    html += f"<li>{format_value(item)}</li>" # format_value on each item
                html += "</ul>"
        elif data_dict is None: # Explicit check for None
             html += "<p><em style='color: #999;'>No data available for this section (None).</em></p>"
        else:
            # This is a fallback for other non-dict, non-Series, non-ndarray, non-list inputs.
            # It assumes format_value can handle it (i.e., it's scalar or string).
            html += f"<p>{format_value(data_dict)}</p>" 
        return html
    
    if not data_dict: # Empty dictionary
        html += "<p><em style='color: #999;'>No data available for this section.</em></p>"
        return html

    if 'data' in data_dict and 'index' in data_dict and 'columns' in data_dict:
        try:
            df = pd.DataFrame(data_dict['data'], index=data_dict['index'], columns=data_dict['columns'])
            if df.empty:
                html += "<p><em style='color: #999;'>No data available in table.</em></p>"
            else:
                formatted_df = df.map(format_value)
                html += formatted_df.to_html(table_id=table_id, classes=["table", "table-striped", "table-hover", "table-sm"], border=0, escape=False)
            return html
        except Exception:
            pass 

    html += f"<table id='{table_id}' class='table table-bordered table-sm'>\\n"
    html += "<thead class='thead-light'><tr><th>Key</th><th>Value</th></tr></thead>\\n<tbody>"
    for key, value in data_dict.items():
        html += f"<tr><td><strong>{key}</strong></td><td>"
        if isinstance(value, dict):
            nested_table_id = f"{table_id}_{key.replace(' ', '_')}"
            # Pass is_fs_module flag down for nested FS tables
            html += dict_to_html_table(value, nested_table_id, key, is_fs_module=(is_fs_module and table_id.startswith("financial_stability")))
        elif isinstance(value, list):
            if not value: # Empty list
                 html += "<em style='color: #999;'>No items in list.</em>"
            elif all(isinstance(item, dict) for item in value): # List of dicts
                try:
                    df = pd.DataFrame(value)
                    if df.empty:
                        html += "<em style='color: #999;'>No data available in table.</em>"
                    else:
                        # Ensure all elements are passed to format_value individually
                        formatted_df = df.map(format_value)
                        html += formatted_df.to_html(table_id=f"{table_id}_{key}", classes=["table", "table-striped", "table-hover", "table-sm"], border=0, index=False, escape=False)
                except Exception as e: # Catch potential errors during DataFrame conversion/mapping
                    # Fallback rendering for list of dicts if DataFrame processing fails
                    # print(f"Error processing list of dicts for key '{key}': {e}") # Optional: for debugging
                    html += "<ul>"
                    for item in value:
                        html += "<li>"
                        if isinstance(item, dict): # Should be true based on outer check
                             html += "<table class='table table-sm table-borderless' style='margin-bottom: 0;'>"
                             for k_i, v_i in item.items():
                                 html += f"<tr><td style='padding:0.2rem;'><strong>{k_i}:</strong></td><td style='padding:0.2rem;'>{format_value(v_i)}</td></tr>"
                             html += "</table>"
                        else: # Fallback if item is not a dict (unexpected)
                             html += format_value(item)
                        html += "</li>"
                    html += "</ul>"
            else: # Simple list of non-dict items
                html += "<ul>"
                for item in value:
                    html += f"<li>{format_value(item)}</li>"
                html += "</ul>"
        elif isinstance(value, pd.Series): # Handle pandas Series
            if value.empty:
                html += "<em style='color: #999;'>Empty series.</em>"
            else:
                html += "<table class='table table-sm table-bordered' style='width: auto; min-width: 300px;'><thead><tr><th>Index</th><th>Value</th></tr></thead><tbody>"
                for idx, val_item in value.items():
                    html += f"<tr><td>{idx}</td><td>{format_value(val_item)}</td></tr>"
                html += "</tbody></table>"
        elif isinstance(value, np.ndarray): # Handle NumPy arrays
            if value.size == 0:
                html += "<em style='color: #999;'>Empty array.</em>"
            else:
                html += "<ul>"
                for item in value.flatten(): # Flatten in case of multi-dimensional arrays
                    html += f"<li>{format_value(item)}</li>"
                html += "</ul>"
        else: # Fallback for other types, hopefully scalar
            html += format_value(value)
        html += "</td></tr>\\n"
    html += "</tbody></table>\\n"
    return html

def format_stress_test_results_to_html(data_dict: Dict, table_id: str) -> str:
    """Formats the stress test results into an HTML section."""
    if not data_dict or not data_dict.get("active", False):
        return "<p><em>Stress test was not enabled or no results available.</em></p>"

    html = "<div class='stress-test-results'><h5>Stress Test Details</h5>"
    
    html += "<h6>Scenario Applied:</h6>"
    if "scenario_applied" in data_dict and isinstance(data_dict["scenario_applied"], dict):
        html += dict_to_html_table(data_dict["scenario_applied"], f"{table_id}_scenario", "Scenario Parameters")
    else:
        html += "<p><em>Scenario parameters not available.</em></p>"
        
    html += "<h6 class='mt-3'>Impact Summary:</h6>"
    if "impact_summary" in data_dict and isinstance(data_dict["impact_summary"], dict):
        html += dict_to_html_table(data_dict["impact_summary"], f"{table_id}_impact", "Impact Metrics")
    else:
        html += "<p><em>Impact summary not available.</em></p>"

    html += "<h6 class='mt-3'>Macro Variable Trajectory During Stress:</h6>"
    if "macro_var_trajectory_during_stress" in data_dict and isinstance(data_dict["macro_var_trajectory_during_stress"], dict):
        trajectory_data = data_dict["macro_var_trajectory_during_stress"]
        if not trajectory_data:
            html += "<p><em>No macro variable trajectory data recorded.</em></p>"
        else:
            for var_name, trajectory_list in trajectory_data.items():
                if isinstance(trajectory_list, list) and trajectory_list:
                    try:
                        df = pd.DataFrame(trajectory_list)
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        if pd.api.types.is_numeric_dtype(df['value']):
                             df['value'] = df['value'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
                        
                        html += f"<p><strong>{var_name.replace('_', ' ').title()}:</strong></p>"
                        html += df.to_html(classes=["table", "table-sm", "table-bordered"], index=False, na_rep="N/A", border=0)
                    except Exception as e:
                        html += f"<p><em>Error formatting trajectory for {var_name}: {e}</em></p>"
                        html += "<pre>" + str(trajectory_list) + "</pre>"
                elif not trajectory_list:
                     html += f"<p><em>No trajectory data for {var_name}.</em></p>"
    else:
        html += "<p><em>Macro variable trajectory data not available.</em></p>"
        
    html += "</div>"
    return html

def create_html_report(data_sources, output_filename="simulation_report.html"):
    # Generates an HTML report from a dictionary of data sources.
    # Each key in data_sources is the title for a section.
    # Each value is the loaded JSON data for that section.
    html_parts = []
    html_parts.append("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BankSim Simulation Report</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }
            .container { background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; margin-bottom: 30px; }
            h2 { color: #555; margin-top: 30px; border-bottom: 2px solid #eee; padding-bottom: 10px;}
            table { width: 100%; margin-bottom: 20px; }
            th, td { text-align: left; padding: 8px; vertical-align: top;}
            .table-sm th, .table-sm td { padding: 0.4rem; }
            .table thead th { background-color: #e9ecef; }
            .footer { text-align: center; margin-top: 40px; font-size: 0.9em; color: #777; }
            .nav-tabs .nav-link.active { background-color: #007bff; color: white; }
            .tab-content { border: 1px solid #dee2e6; border-top: none; padding: 15px; margin-bottom: 20px; border-radius: 0 0 0.25rem 0.25rem;}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BankSim Simulation Run Report</h1>
            
            <ul class="nav nav-tabs" id="simulationTabs" role="tablist">
    """) # End of first static block

    tab_content_html_parts = ["<div class='tab-content' id='simulationTabsContent'>"]
    is_first_tab = True

    for idx, (title, data) in enumerate(data_sources.items()):
        tab_id = title.replace(" ", "_").lower()
        active_class_nav = "active" if is_first_tab else ""
        active_class_pane = "show active" if is_first_tab else ""

        # Append the tab navigation item
        html_parts.append(f"""
                <li class="nav-item" role="presentation">
                    <a class="nav-link {active_class_nav}" id="{tab_id}-tab" data-toggle="tab" href="#{tab_id}" role="tab" aria-controls="{tab_id}" aria-selected="{str(is_first_tab).lower()}">{title}</a>
                </li>
        """)
        
        tab_pane_inner_html = f"<h3>{title} Simulation Results</h3>"
        if data is None:
            tab_pane_inner_html += "<p><em style='color: #999;'>No data loaded for this simulation. Check for warnings during JSON loading.</em></p>"
        elif isinstance(data, dict):
            for section_key, section_data in data.items():
                current_table_id = f"{tab_id}_{section_key.replace(' ', '_').lower()}"
                tab_pane_inner_html += dict_to_html_table(section_data, current_table_id, section_key.replace('_', ' ').title(), is_fs_module=(title == "Financial Stability"))
        else:
            tab_pane_inner_html += f"<p>{str(data)}</p>"
        
        # Append to the list that will form the content of the tabs div
        tab_content_html_parts.append(f"""
            <div class="tab-pane fade {active_class_pane}" id="{tab_id}" role="tabpanel" aria-labelledby="{tab_id}-tab">
                {tab_pane_inner_html}
            </div>
        """)
        is_first_tab = False

    html_parts.append("""
            </ul> 
    """) # End of ul tag for tabs
    html_parts.append("".join(tab_content_html_parts))
    html_parts.append("</div>") # Close tab-content div itself

    html_parts.append("""
            <div class="footer">
                Report generated on: <span id="datetime"></span>
            </div>
        </div>

        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        <script>
            document.getElementById('datetime').textContent = new Date().toLocaleString();
        </script>
    </body>
    </html>
    """) # End of final static block
    
    final_html_content = "\n".join(html_parts)

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(final_html_content)
    print(f"HTML report generated: {output_filename}")

def main():
    simulation_files = {
        "Monetary Policy": "mp_results.json",
        "Banking Supervision": "bs_results.json",
        "Foreign Exchange": "fx_results.json",
        "Financial Stability": "fs_results.json"
    }
    
    all_simulation_data = {}
    for sim_name, file_name in simulation_files.items():
        data = robust_json_load(file_name)
        all_simulation_data[sim_name] = data

    create_html_report(all_simulation_data)

if __name__ == "__main__":
    main() 