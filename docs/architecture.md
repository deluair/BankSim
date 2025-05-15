# BankSim Architecture Overview

## System Components

BankSim is structured around several key components, each handling a specific aspect of the central banking simulation:

### Core Simulation Engine

#### Monetary Policy Operations Module
- Repo Rate Mechanism
- Open Market Operations
- Reserve Requirement Adjustments
- Liquidity Management
- Standing Facilities
- Policy Rate Transmission

#### Banking Supervision Module
- CAMELS Rating System
- Risk-Based Supervision
- Stress Testing Framework
- Off-Site Surveillance
- Regulatory Reporting
- Enforcement Actions

#### Financial Stability Module
- Systemic Risk Indicators
- Macro-Prudential Tools
- Financial Cycle Measurement
- Crisis Simulation
- Cross-Border Contagion
- Housing Market Monitor

#### Foreign Exchange & Reserves Module
- Intervention Simulation
- Reserve Management
- Exchange Rate Models
- Balance of Payments
- FX Market Microstructure
- SDR Allocation

### Technical Architecture

- **API Layer**: FastAPI backend exposing REST endpoints
- **Computational Core**: Python-based simulation engine using NumPy, SciPy, etc.
- **Frontend**: React/TypeScript application with D3.js visualizations
- **Data Storage**: PostgreSQL database with TimescaleDB for time series data

## Directory Structure

```
BankSim/
├── backend/              # Python/FastAPI backend
│   ├── app/              # Application code
│   ├── tests/            # Unit and integration tests
│   └── requirements.txt  # Python dependencies
├── frontend/             # React/TypeScript frontend
│   ├── src/              # Source code
│   ├── public/           # Static assets
│   └── package.json      # Node.js dependencies
├── docs/                 # Documentation
├── data/                 # Sample datasets and data management scripts
└── scripts/              # Utility scripts
```

## Data Flow

1. User interacts with the frontend interface
2. Frontend makes API calls to the backend
3. Backend processes requests through the simulation engine
4. Results are stored in the database
5. Data is retrieved and visualized in the frontend

## Technology Stack

- **Backend**: Python 3.11+, FastAPI, SQLAlchemy, NumPy, SciPy, pandas, PyTorch
- **Frontend**: React 18+, TypeScript, Redux, D3.js
- **Database**: PostgreSQL, TimescaleDB
- **Infrastructure**: Docker, Kubernetes
