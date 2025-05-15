# BankSim: Banking and Economic Simulation Platform

BankSim is a Python-based simulation platform designed to model and analyze various scenarios within the banking sector and broader economy. It allows users to explore the dynamics of monetary policy, banking supervision, foreign exchange markets, and financial stability, including stress testing and contagion effects.

## Core Objectives

*   Provide a flexible framework for simulating complex financial and economic interactions.
*   Enable analysis of policy impacts and risk propagation.
*   Support research and educational purposes in finance and economics.

## Modules Implemented

The platform is built around several key simulation modules:

1.  **Monetary Policy (`monetary_policy.py`)**:
    *   Simulates the impact of monetary policy decisions (e.g., interest rate changes) on macroeconomic variables like inflation and output.
    *   Includes configurable parameters for policy rules (e.g., Taylor rule), transmission lags, and economic shock modeling.

2.  **Banking Supervision (`banking_supervision.py`)**:
    *   Focuses on the regulatory aspect of banking, including capital adequacy (CAMELS ratings, CAR).
    *   Allows for modeling of bank examinations, risk assessment, and potential regulatory actions based on bank health.

3.  **Foreign Exchange (`foreign_exchange.py`)**:
    *   Models foreign exchange market dynamics, including exchange rate determination under different regimes (e.g., managed float).
    *   Simulates central bank interventions in the FX market and their impact on reserves and exchange rates.

4.  **Financial Stability (`financial_stability.py`)**:
    *   Provides a system-wide view of financial stability, incorporating interbank networks and systemic risk indicators.
    *   **Key Features within Financial Stability**:
        *   **Stress Testing**: Allows users to define and apply stress scenarios (e.g., GDP shocks, unemployment shocks, asset price shocks) to the banking system and observe their impact on bank financials and systemic risk.
        *   **Bank Failures**: Models bank failures based on configurable capital adequacy thresholds (e.g., CAR falling below a critical point).
        *   **Contagion Effects**: Implements an initial model for contagion, where the failure of one bank can lead to losses for its creditors in the interbank market, potentially triggering further failures.

## Key Platform Features

*   **Configurable Simulations**: Each module uses a detailed configuration (`config.py`) allowing users to tailor parameters, dates, and scenarios.
*   **Synthetic Data Generation**: For all modules, the system can generate synthetic bank data and economic conditions if no database is connected, enabling standalone runs.
*   **Modular Design**: Simulations are separated into distinct Python scripts and modules, promoting clarity and extensibility.
*   **JSON Output**: Simulation results are saved in structured JSON format (`mp_results.json`, `bs_results.json`, `fx_results.json`, `fs_results.json`), facilitating further analysis.
*   **HTML Report Generation**: A `generate_report.py` script compiles the JSON outputs into a user-friendly HTML report (`simulation_report.html`) with tabbed sections for each simulation module.

## Directory Structure

The project is organized as follows:

```
BankSim/
├── backend/                  # Core simulation logic and API (FastAPI based)
│   ├── app/
│   │   ├── auth/             # Authentication components (if full API is used)
│   │   ├── db/               # Database interaction (models, crud, session)
│   │   ├── models/           # SQLAlchemy models for database entities
│   │   ├── routers/          # API routers
│   │   ├── simulation/       # Core simulation engines and configuration
│   │   │   ├── config.py     # Simulation configuration classes
│   │   │   ├── monetary_policy.py
│   │   │   ├── banking_supervision.py
│   │   │   ├── foreign_exchange.py
│   │   │   └── financial_stability.py
│   │   ├── tests/            # Unit and integration tests
│   │   └── main.py           # FastAPI application entry point
│   ├── requirements.txt      # Python dependencies for the backend
│   └── banksim.db            # SQLite database file (if used)
├── data/                     # Placeholder for raw data files (if any)
├── docs/                     # Project documentation
├── frontend/                 # Placeholder/submodule for a potential UI
├── scripts/                  # Utility scripts (e.g., data management, setup)
│── .gitignore                # Specifies intentionally untracked files
│── README.md                 # This file
│── run_mp_simulation.py      # Script to run Monetary Policy simulation
│── run_bs_simulation.py      # Script to run Banking Supervision simulation
│── run_fx_simulation.py      # Script to run Foreign Exchange simulation
│── run_fs_simulation.py      # Script to run Financial Stability simulation
│── generate_report.py        # Script to generate the HTML summary report
│── mp_results.json           # Example output for Monetary Policy
│── bs_results.json           # Example output for Banking Supervision
│── fx_results.json           # Example output for Foreign Exchange
│── fs_results.json           # Example output for Financial Stability
└── simulation_report.html    # Generated HTML report
```

## Setup and Installation

1.  **Prerequisites**:
    *   Python 3.8 or higher.
    *   `pip` for package installation.
    *   Git for cloning the repository.

2.  **Clone the Repository** (if you haven't already):
    ```bash
    git clone <repository_url>
    cd BankSim
    ```

3.  **Install Dependencies**:
    The primary dependencies for running the simulations are listed in `backend/requirements.txt`.
    ```bash
    pip install -r backend/requirements.txt
    ```
    This will install packages such as `pandas`, `numpy`, `networkx`, `pydantic`, and `fastapi` related libraries.

## How to Run Simulations

Simulations are run using dedicated Python scripts located in the root of the project directory.

1.  **Run Individual Simulations**:
    Execute the desired simulation script from the project root directory:
    *   Monetary Policy: `python run_mp_simulation.py`
    *   Banking Supervision: `python run_bs_simulation.py`
    *   Foreign Exchange: `python run_fx_simulation.py`
    *   Financial Stability: `python run_fs_simulation.py`

    Each script will:
    *   Load a default configuration for its respective module.
    *   Instantiate and run the simulator.
    *   Save the results to a corresponding JSON file (e.g., `fs_results.json`).

2.  **Customize Simulation Parameters**:
    *   **Directly in run scripts**: You can modify parameters within the `run_*.py` scripts before the simulator is instantiated. For example, in `run_fs_simulation.py`, you can enable stress tests or contagion effects:
        ```python
        # In run_fs_simulation.py
        fs_config = load_default_config(simulation_type="financial_stability")
        fs_config.stress_test_enabled = True
        fs_config.stress_gdp_shock_annual_pct_change = -5.0 # Example customization
        fs_config.enable_contagion_effects = True
        fs_config.car_failure_threshold_pct = 3.5 # Example customization
        ```
    *   **In `config.py`**: For more persistent changes to default behaviors, you can modify the default values within `backend/app/simulation/config.py`.

3.  **Generate HTML Report**:
    After running one or more simulations and generating their respective `*_results.json` files, you can create a consolidated HTML report:
    ```bash
    python generate_report.py
    ```
    This will produce `simulation_report.html` in the project root, with different tabs for each simulation module for which results were found.

## Backend (FastAPI)

The `backend` directory contains a FastAPI application. While the `run_*.py` scripts execute simulations directly, the FastAPI setup provides a C:/Users/mhossen/OneDrive - University of Tennessee/AI/BankSim for exposing simulation capabilities via an API, potentially interacting with a database (models defined in `backend/app/models/`), and user authentication. For the current command-line driven simulation runs, the API and database components are not directly utilized but form part of the broader platform structure.

## Current Status & Limitations

*   BankSim is an actively developed platform. While core functionalities for the listed modules are implemented, some aspects are simplified.
*   The contagion model in Financial Stability is an initial version; more sophisticated propagation mechanisms and feedback loops can be added.
*   Bank behavior in response to failure or extreme stress is currently simplified (e.g., failed banks cease normal operations but detailed resolution or asset fire sales are not modeled).
*   The system primarily uses synthetically generated data for broad scenario testing. Full integration with real-world datasets via the database layer is possible but requires data loading and mapping.
*   The `frontend` directory is a placeholder, suggesting a user interface could be a future addition.

## Contributing

(Placeholder for contribution guidelines if the project becomes open to external contributions.)

## License

(Placeholder for license information - e.g., MIT, Apache 2.0, etc.)
