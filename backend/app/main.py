from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date, timedelta
import os

from . import models
from .models import user # Ensure User and Role models are loaded for table creation
from .db.session import engine, get_db, Base, init_db # Updated import path, add init_db
from .routers import simulation
from .auth.router import router as auth_router_instance # Import the auth APIRouter instance

# Tables will be created via startup event

app = FastAPI(
    title="BankSim API",
    description="API for the Central Banking Operations Simulation Platform",
    version="0.1.0"
)

@app.on_event("startup")
async def on_startup():
    init_db()

# Include routers
app.include_router(simulation.router)
app.include_router(auth_router_instance) # Add the auth router

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to BankSim API",
        "description": "Central Banking Operations Simulation Platform",
        "version": "0.1.0",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "ok", "timestamp": date.today().isoformat()}

# Economic Indicators Endpoints

@app.get("/api/economic/gdp", tags=["Economic Indicators"])
async def get_gdp_data(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve GDP data within the specified date range.
    """
    query = db.query(models.GDP)
    
    if start_date:
        query = query.filter(models.GDP.date >= start_date)
    if end_date:
        query = query.filter(models.GDP.date <= end_date)
    
    return query.order_by(models.GDP.date).all()

@app.get("/api/economic/inflation", tags=["Economic Indicators"])
async def get_inflation_data(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve inflation data (CPI) within the specified date range.
    """
    query = db.query(models.CPI)
    
    if start_date:
        query = query.filter(models.CPI.date >= start_date)
    if end_date:
        query = query.filter(models.CPI.date <= end_date)
    
    return query.order_by(models.CPI.date).all()

# Monetary Policy Endpoints

@app.get("/api/monetary/policy_rates", tags=["Monetary Policy"])
async def get_policy_rates(
    rate_type: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve policy interest rates set by the central bank.
    """
    query = db.query(models.PolicyRate)
    
    if rate_type:
        query = query.filter(models.PolicyRate.rate_type == rate_type)
    if start_date:
        query = query.filter(models.PolicyRate.effective_date >= start_date)
    if end_date:
        query = query.filter(models.PolicyRate.effective_date <= end_date)
    
    return query.order_by(models.PolicyRate.effective_date.desc()).all()

@app.get("/api/monetary/repo_operations", tags=["Monetary Policy"])
async def get_repo_operations(
    operation_type: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve repo operations conducted by the central bank.
    """
    query = db.query(models.RepoOperation)
    
    if operation_type:
        query = query.filter(models.RepoOperation.operation_type == operation_type)
    if start_date:
        query = query.filter(models.RepoOperation.operation_date >= start_date)
    if end_date:
        query = query.filter(models.RepoOperation.operation_date <= end_date)
    
    return query.order_by(models.RepoOperation.operation_date.desc()).all()

# Banking System Endpoints

@app.get("/api/banking/banks", tags=["Banking System"])
async def get_banks(
    bank_type: Optional[str] = None,
    is_active: bool = True,
    db: Session = Depends(get_db)
):
    """
    Retrieve list of banks in the system.
    """
    query = db.query(models.Bank).filter(models.Bank.is_active == is_active)
    
    if bank_type:
        query = query.filter(models.Bank.bank_type == bank_type)
    
    return query.order_by(models.Bank.name).all()

@app.get("/api/banking/system_aggregates", tags=["Banking System"])
async def get_banking_system_aggregates(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve aggregate data for the entire banking system.
    """
    query = db.query(models.BankingSystemAggregate)
    
    if start_date:
        query = query.filter(models.BankingSystemAggregate.date >= start_date)
    if end_date:
        query = query.filter(models.BankingSystemAggregate.date <= end_date)
    
    return query.order_by(models.BankingSystemAggregate.date.desc()).all()

# Foreign Exchange Endpoints

@app.get("/api/forex/exchange_rates", tags=["Foreign Exchange"])
async def get_exchange_rates(
    currency_code: str = "USD",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve exchange rate data for the specified currency.
    """
    query = db.query(models.ExchangeRate).filter(models.ExchangeRate.currency_code == currency_code)
    
    if start_date:
        query = query.filter(models.ExchangeRate.date >= start_date)
    if end_date:
        query = query.filter(models.ExchangeRate.date <= end_date)
    
    return query.order_by(models.ExchangeRate.date.desc()).all()

@app.get("/api/forex/reserves", tags=["Foreign Exchange"])
async def get_foreign_reserves(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve foreign reserve holdings of the central bank.
    """
    query = db.query(models.ForeignReserve)
    
    if start_date:
        query = query.filter(models.ForeignReserve.date >= start_date)
    if end_date:
        query = query.filter(models.ForeignReserve.date <= end_date)
    
    return query.order_by(models.ForeignReserve.date.desc()).all()
