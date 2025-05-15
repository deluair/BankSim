"""
Simulation API router.

This module provides API endpoints for running various types of simulations
in the BankSim platform, including monetary policy simulations.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from datetime import date, datetime, timedelta
from pydantic import BaseModel, Field
import json

from ..database import get_db
from ..simulation.config import MonetaryPolicyConfig, BankingSupervisionConfig, ForeignExchangeConfig
from ..simulation.monetary_policy import MonetaryPolicySimulator
from .. import data_generators

router = APIRouter(
    prefix="/api/simulation",
    tags=["Simulation"],
    responses={404: {"description": "Not found"}},
)


class SimulationRequest(BaseModel):
    """Base model for simulation requests."""
    name: str = Field(..., description="Name of the simulation")
    description: Optional[str] = Field(None, description="Description of the simulation")
    start_date: date = Field(..., description="Start date for the simulation")
    end_date: date = Field(..., description="End date for the simulation")
    time_step: str = Field("monthly", description="Time step for the simulation (daily, weekly, monthly, quarterly)")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class MonetaryPolicySimulationRequest(SimulationRequest):
    """Request model for monetary policy simulations."""
    inflation_target: float = Field(4.5, description="Central bank's inflation target in percentage")
    inflation_weight: float = Field(1.5, description="Weight on inflation gap in Taylor rule")
    output_gap_weight: float = Field(0.5, description="Weight on output gap in Taylor rule")
    interest_rate_smoothing: float = Field(0.7, description="Degree of interest rate smoothing (0-1)")
    enable_shocks: bool = Field(True, description="Whether to enable random economic shocks")


class PolicyScenarioRequest(BaseModel):
    """Request model for policy scenario simulations."""
    scenario_name: str = Field(..., description="Name of the policy scenario")
    config: MonetaryPolicySimulationRequest = Field(..., description="Base configuration for the simulation")
    policy_changes: List[Dict[str, Any]] = Field(..., description="List of policy rate changes with dates")


class GenerateDataRequest(BaseModel):
    """Request model for sample data generation."""
    start_date: date = Field(..., description="Start date for historical data")
    end_date: date = Field(..., description="End date for historical data")
    include_banks: bool = Field(True, description="Whether to generate bank data")
    include_economic: bool = Field(True, description="Whether to generate economic indicators")
    include_monetary: bool = Field(True, description="Whether to generate monetary policy data")
    include_forex: bool = Field(True, description="Whether to generate foreign exchange data")


@router.post("/monetary-policy")
async def run_monetary_policy_simulation(
    request: MonetaryPolicySimulationRequest,
    db: Session = Depends(get_db)
):
    """
    Run a monetary policy simulation with the specified parameters.
    
    This endpoint runs a simulation of monetary policy and its effects
    on the economy using the provided configuration parameters.
    """
    try:
        # Convert request to configuration
        config = MonetaryPolicyConfig(
            name=request.name,
            description=request.description or "",
            start_date=request.start_date,
            end_date=request.end_date,
            time_step=request.time_step,
            random_seed=request.random_seed,
            inflation_target=request.inflation_target,
            inflation_weight=request.inflation_weight,
            output_gap_weight=request.output_gap_weight,
            interest_rate_smoothing=request.interest_rate_smoothing,
            enable_shocks=request.enable_shocks
        )
        
        # Create and run simulator
        simulator = MonetaryPolicySimulator(config, db)
        simulator.run_simulation()
        
        # Get results organized by category
        results = simulator.get_results()
        
        # Convert pandas DataFrames to dict for JSON serialization
        formatted_results = {}
        for category, df in results.items():
            formatted_results[category] = df.to_dict(orient='records')
        
        return {
            "simulation_name": request.name,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "results": formatted_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@router.post("/policy-scenario")
async def run_policy_scenario(
    request: PolicyScenarioRequest,
    db: Session = Depends(get_db)
):
    """
    Run a specific monetary policy scenario with predetermined policy changes.
    
    This endpoint allows testing the effects of specific policy rate decisions
    over time, rather than using the Taylor rule for automatic decisions.
    """
    try:
        # Convert request to configuration
        config = MonetaryPolicyConfig(
            name=request.config.name,
            description=request.config.description or "",
            start_date=request.config.start_date,
            end_date=request.config.end_date,
            time_step=request.config.time_step,
            random_seed=request.config.random_seed,
            inflation_target=request.config.inflation_target,
            inflation_weight=request.config.inflation_weight,
            output_gap_weight=request.config.output_gap_weight,
            interest_rate_smoothing=request.config.interest_rate_smoothing,
            enable_shocks=request.config.enable_shocks
        )
        
        # Create simulator
        simulator = MonetaryPolicySimulator(config, db)
        
        # Run the policy scenario
        results = simulator.run_policy_scenario(request.scenario_name, request.policy_changes)
        
        # Convert pandas DataFrames to dict for JSON serialization
        formatted_results = {}
        for category, df in results.items():
            formatted_results[category] = df.to_dict(orient='records')
        
        return {
            "scenario_name": request.scenario_name,
            "simulation_name": request.config.name,
            "start_date": request.config.start_date.isoformat(),
            "end_date": request.config.end_date.isoformat(),
            "policy_changes": request.policy_changes,
            "results": formatted_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenario simulation error: {str(e)}")


@router.post("/generate-data")
async def generate_sample_data(
    request: GenerateDataRequest,
    db: Session = Depends(get_db)
):
    """
    Generate sample data for the BankSim database.
    
    This endpoint populates the database with realistic sample data
    for testing and demonstration purposes.
    """
    try:
        # Generate all data
        data_generators.generate_all_data(db, request.start_date, request.end_date)
        
        return {
            "status": "success",
            "message": f"Generated sample data from {request.start_date.isoformat()} to {request.end_date.isoformat()}",
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation error: {str(e)}")


@router.get("/configs/default/monetary-policy")
async def get_default_monetary_policy_config():
    """Get the default configuration for monetary policy simulations."""
    today = date.today()
    start_date = today - timedelta(days=365)  # One year ago
    end_date = today + timedelta(days=730)    # Two years from now
    
    config = MonetaryPolicyConfig(
        name="Default Monetary Policy Simulation",
        description="Standard monetary policy simulation with Bangladesh parameters",
        start_date=start_date,
        end_date=end_date
    )
    
    return json.loads(config.to_json())
