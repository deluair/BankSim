"""
CRUD operations for BankSim database.

This module contains functions for Create, Read, Update, Delete operations
for all database models used in the BankSim application.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import date, datetime
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from ..models.economic_indicators import GDPData, CPIData, UnemploymentData, TradeBalanceData
from ..models.monetary_policy import PolicyRate, PolicyRateType, RepoOperation, OpenMarketOperation
from ..models.banking_system import Bank, BankType, BankBalance, LoanPortfolio, CAMELSRating
from ..models.foreign_exchange import ExchangeRate, ForeignReserves, FXIntervention, BalanceOfPayments


# ---------- Economic Indicators CRUD ----------

def create_gdp_data(db: Session, data: Dict[str, Any]) -> GDPData:
    """Create a new GDP data entry."""
    db_item = GDPData(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_gdp_data(db: Session, skip: int = 0, limit: int = 100) -> List[GDPData]:
    """Get list of GDP data entries."""
    return db.query(GDPData).order_by(GDPData.date.desc()).offset(skip).limit(limit).all()

def get_gdp_data_by_date_range(db: Session, start_date: date, end_date: date) -> List[GDPData]:
    """Get GDP data within a date range."""
    return db.query(GDPData).filter(
        GDPData.date >= start_date,
        GDPData.date <= end_date
    ).order_by(GDPData.date).all()

def create_cpi_data(db: Session, data: Dict[str, Any]) -> CPIData:
    """Create a new CPI data entry."""
    db_item = CPIData(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_cpi_data(db: Session, skip: int = 0, limit: int = 100) -> List[CPIData]:
    """Get list of CPI data entries."""
    return db.query(CPIData).order_by(CPIData.date.desc()).offset(skip).limit(limit).all()

def get_cpi_data_by_date_range(db: Session, start_date: date, end_date: date) -> List[CPIData]:
    """Get CPI data within a date range."""
    return db.query(CPIData).filter(
        CPIData.date >= start_date,
        CPIData.date <= end_date
    ).order_by(CPIData.date).all()

def create_unemployment_data(db: Session, data: Dict[str, Any]) -> UnemploymentData:
    """Create a new unemployment data entry."""
    db_item = UnemploymentData(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_unemployment_data(db: Session, skip: int = 0, limit: int = 100) -> List[UnemploymentData]:
    """Get list of unemployment data entries."""
    return db.query(UnemploymentData).order_by(UnemploymentData.date.desc()).offset(skip).limit(limit).all()

def create_trade_balance_data(db: Session, data: Dict[str, Any]) -> TradeBalanceData:
    """Create a new trade balance data entry."""
    db_item = TradeBalanceData(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_trade_balance_data(db: Session, skip: int = 0, limit: int = 100) -> List[TradeBalanceData]:
    """Get list of trade balance data entries."""
    return db.query(TradeBalanceData).order_by(TradeBalanceData.date.desc()).offset(skip).limit(limit).all()


# ---------- Monetary Policy CRUD ----------

def create_policy_rate(db: Session, data: Dict[str, Any]) -> PolicyRate:
    """Create a new policy rate entry."""
    db_item = PolicyRate(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_policy_rates(db: Session, skip: int = 0, limit: int = 100) -> List[PolicyRate]:
    """Get list of policy rate entries."""
    return db.query(PolicyRate).order_by(PolicyRate.effective_date.desc()).offset(skip).limit(limit).all()

def get_policy_rates_by_type(db: Session, rate_type: PolicyRateType) -> List[PolicyRate]:
    """Get policy rates of a specific type."""
    return db.query(PolicyRate).filter(
        PolicyRate.rate_type == rate_type
    ).order_by(PolicyRate.effective_date.desc()).all()

def get_latest_policy_rate(db: Session, rate_type: PolicyRateType) -> Optional[PolicyRate]:
    """Get the latest policy rate of a specific type."""
    return db.query(PolicyRate).filter(
        PolicyRate.rate_type == rate_type
    ).order_by(PolicyRate.effective_date.desc()).first()

def create_repo_operation(db: Session, data: Dict[str, Any]) -> RepoOperation:
    """Create a new repo operation entry."""
    db_item = RepoOperation(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_repo_operations(db: Session, skip: int = 0, limit: int = 100) -> List[RepoOperation]:
    """Get list of repo operation entries."""
    return db.query(RepoOperation).order_by(RepoOperation.operation_date.desc()).offset(skip).limit(limit).all()

def create_open_market_operation(db: Session, data: Dict[str, Any]) -> OpenMarketOperation:
    """Create a new open market operation entry."""
    db_item = OpenMarketOperation(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_open_market_operations(db: Session, skip: int = 0, limit: int = 100) -> List[OpenMarketOperation]:
    """Get list of open market operation entries."""
    return db.query(OpenMarketOperation).order_by(OpenMarketOperation.operation_date.desc()).offset(skip).limit(limit).all()


# ---------- Banking System CRUD ----------

def create_bank(db: Session, data: Dict[str, Any]) -> Bank:
    """Create a new bank entry."""
    db_item = Bank(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_banks(db: Session, skip: int = 0, limit: int = 100) -> List[Bank]:
    """Get list of banks."""
    return db.query(Bank).filter(
        Bank.is_active == True
    ).offset(skip).limit(limit).all()

def get_bank_by_id(db: Session, bank_id: int) -> Optional[Bank]:
    """Get a bank by ID."""
    return db.query(Bank).filter(Bank.id == bank_id).first()

def get_banks_by_type(db: Session, bank_type: BankType) -> List[Bank]:
    """Get banks of a specific type."""
    return db.query(Bank).filter(
        Bank.bank_type == bank_type,
        Bank.is_active == True
    ).all()

def create_bank_balance(db: Session, data: Dict[str, Any]) -> BankBalance:
    """Create a new bank balance entry."""
    db_item = BankBalance(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_bank_balances(db: Session, bank_id: int, skip: int = 0, limit: int = 100) -> List[BankBalance]:
    """Get list of bank balance entries for a specific bank."""
    return db.query(BankBalance).filter(
        BankBalance.bank_id == bank_id
    ).order_by(BankBalance.report_date.desc()).offset(skip).limit(limit).all()

def get_latest_bank_balance(db: Session, bank_id: int) -> Optional[BankBalance]:
    """Get latest bank balance for a specific bank."""
    return db.query(BankBalance).filter(
        BankBalance.bank_id == bank_id
    ).order_by(BankBalance.report_date.desc()).first()

def create_loan_portfolio(db: Session, data: Dict[str, Any]) -> LoanPortfolio:
    """Create a new loan portfolio entry."""
    db_item = LoanPortfolio(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_loan_portfolios(db: Session, bank_id: int, skip: int = 0, limit: int = 100) -> List[LoanPortfolio]:
    """Get list of loan portfolio entries for a specific bank."""
    return db.query(LoanPortfolio).filter(
        LoanPortfolio.bank_id == bank_id
    ).order_by(LoanPortfolio.report_date.desc()).offset(skip).limit(limit).all()

def create_camels_rating(db: Session, data: Dict[str, Any]) -> CAMELSRating:
    """Create a new CAMELS rating entry."""
    db_item = CAMELSRating(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_camels_ratings(db: Session, bank_id: int, skip: int = 0, limit: int = 100) -> List[CAMELSRating]:
    """Get list of CAMELS rating entries for a specific bank."""
    return db.query(CAMELSRating).filter(
        CAMELSRating.bank_id == bank_id
    ).order_by(CAMELSRating.rating_date.desc()).offset(skip).limit(limit).all()

def get_latest_camels_rating(db: Session, bank_id: int) -> Optional[CAMELSRating]:
    """Get latest CAMELS rating for a specific bank."""
    return db.query(CAMELSRating).filter(
        CAMELSRating.bank_id == bank_id
    ).order_by(CAMELSRating.rating_date.desc()).first()


# ---------- Foreign Exchange CRUD ----------

def create_exchange_rate(db: Session, data: Dict[str, Any]) -> ExchangeRate:
    """Create a new exchange rate entry."""
    db_item = ExchangeRate(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_exchange_rates(db: Session, currency_code: str, skip: int = 0, limit: int = 100) -> List[ExchangeRate]:
    """Get list of exchange rate entries for a specific currency."""
    return db.query(ExchangeRate).filter(
        ExchangeRate.currency_code == currency_code
    ).order_by(ExchangeRate.date.desc()).offset(skip).limit(limit).all()

def get_exchange_rates_by_date_range(db: Session, currency_code: str, start_date: date, end_date: date) -> List[ExchangeRate]:
    """Get exchange rates within a date range for a specific currency."""
    return db.query(ExchangeRate).filter(
        ExchangeRate.currency_code == currency_code,
        ExchangeRate.date >= start_date,
        ExchangeRate.date <= end_date
    ).order_by(ExchangeRate.date).all()

def create_foreign_reserves(db: Session, data: Dict[str, Any]) -> ForeignReserves:
    """Create a new foreign reserves entry."""
    db_item = ForeignReserves(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_foreign_reserves(db: Session, currency_code: str, skip: int = 0, limit: int = 100) -> List[ForeignReserves]:
    """Get list of foreign reserves entries for a specific currency."""
    return db.query(ForeignReserves).filter(
        ForeignReserves.currency_code == currency_code
    ).order_by(ForeignReserves.date.desc()).offset(skip).limit(limit).all()

def create_fx_intervention(db: Session, data: Dict[str, Any]) -> FXIntervention:
    """Create a new FX intervention entry."""
    db_item = FXIntervention(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_fx_interventions(db: Session, skip: int = 0, limit: int = 100) -> List[FXIntervention]:
    """Get list of FX intervention entries."""
    return db.query(FXIntervention).order_by(FXIntervention.intervention_date.desc()).offset(skip).limit(limit).all()

def create_balance_of_payments(db: Session, data: Dict[str, Any]) -> BalanceOfPayments:
    """Create a new balance of payments entry."""
    db_item = BalanceOfPayments(**data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_balance_of_payments(db: Session, skip: int = 0, limit: int = 100) -> List[BalanceOfPayments]:
    """Get list of balance of payments entries."""
    return db.query(BalanceOfPayments).order_by(BalanceOfPayments.date.desc()).offset(skip).limit(limit).all()

def get_balance_of_payments_by_date_range(db: Session, start_date: date, end_date: date) -> List[BalanceOfPayments]:
    """Get balance of payments within a date range."""
    return db.query(BalanceOfPayments).filter(
        BalanceOfPayments.date >= start_date,
        BalanceOfPayments.date <= end_date
    ).order_by(BalanceOfPayments.date).all()


# ---------- Simulation Results CRUD ----------

def save_simulation_results(db: Session, simulator_type: str, config: Dict[str, Any], results: Dict[str, Any]) -> None:
    """
    Save simulation results to the database.
    
    This generic function converts simulation results to the appropriate database models
    and saves them to the database based on the simulator type.
    """
    if simulator_type == "monetary_policy":
        _save_monetary_policy_results(db, config, results)
    elif simulator_type == "banking_supervision":
        _save_banking_supervision_results(db, config, results)
    elif simulator_type == "financial_stability":
        _save_financial_stability_results(db, config, results)
    elif simulator_type == "foreign_exchange":
        _save_foreign_exchange_results(db, config, results)
    else:
        raise ValueError(f"Unknown simulator type: {simulator_type}")

def _save_monetary_policy_results(db: Session, config: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Save monetary policy simulation results."""
    # Extract data from results
    if "policy_rates" in results:
        for rate_data in results["policy_rates"]:
            # Check if rate already exists for the date and type
            existing_rate = db.query(PolicyRate).filter(
                PolicyRate.effective_date == date.fromisoformat(rate_data["date"]),
                PolicyRate.rate_type == rate_data["type"]
            ).first()
            
            if not existing_rate:
                # Create new policy rate entry
                create_policy_rate(db, {
                    "effective_date": date.fromisoformat(rate_data["date"]),
                    "rate_type": rate_data["type"],
                    "rate_value": rate_data["value"],
                    "is_simulated": True,
                    "simulation_id": config.get("simulation_id", None)
                })
    
    # Save other monetary policy results as needed

def _save_banking_supervision_results(db: Session, config: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Save banking supervision simulation results."""
    # Extract data from results
    if "examinations" in results:
        for exam_data in results["examinations"]:
            # Save CAMELS ratings from examinations
            create_camels_rating(db, {
                "bank_id": exam_data["bank_id"],
                "rating_date": date.fromisoformat(exam_data["examination_date"]),
                "composite_rating": exam_data["new_composite"],
                "capital_rating": exam_data["component_ratings"]["capital_rating"],
                "asset_rating": exam_data["component_ratings"]["asset_rating"],
                "management_rating": exam_data["component_ratings"]["management_rating"],
                "earnings_rating": exam_data["component_ratings"]["earnings_rating"],
                "liquidity_rating": exam_data["component_ratings"]["liquidity_rating"],
                "sensitivity_rating": exam_data["component_ratings"]["sensitivity_rating"],
                "is_simulated": True,
                "simulation_id": config.get("simulation_id", None)
            })
    
    # Save other banking supervision results as needed

def _save_financial_stability_results(db: Session, config: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Save financial stability simulation results."""
    # Implementation for saving financial stability results
    pass

def _save_foreign_exchange_results(db: Session, config: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Save foreign exchange simulation results."""
    # Extract data from results
    if "exchange_rates" in results and "time_series" in results["exchange_rates"]:
        for rate_data in results["exchange_rates"]["time_series"]:
            rate_date = date.fromisoformat(rate_data["date"])
            
            # Save exchange rates for each currency
            for currency, rate_value in rate_data.items():
                if currency != "date":  # Skip the date field
                    # Check if rate already exists for the date and currency
                    existing_rate = db.query(ExchangeRate).filter(
                        ExchangeRate.date == rate_date,
                        ExchangeRate.currency_code == currency
                    ).first()
                    
                    if not existing_rate:
                        # Create new exchange rate entry
                        create_exchange_rate(db, {
                            "date": rate_date,
                            "currency_code": currency,
                            "rate": rate_value,
                            "is_simulated": True,
                            "simulation_id": config.get("simulation_id", None)
                        })
    
    # Save interventions
    if "interventions" in results:
        for intervention_data in results["interventions"]:
            create_fx_intervention(db, {
                "intervention_date": date.fromisoformat(intervention_data["date"]),
                "currency_code": intervention_data["currency"],
                "amount": intervention_data["amount_usd"],
                "intervention_type": intervention_data["type"],
                "result": intervention_data["result"],
                "is_simulated": True,
                "simulation_id": config.get("simulation_id", None)
            })
    
    # Save other foreign exchange results as needed
