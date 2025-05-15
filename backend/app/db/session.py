"""
Database session management module.

This module provides functions for managing SQLAlchemy database sessions
and connection pooling for the BankSim application.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import os
import logging  # Added
import sqlite3  # Added
from typing import Generator

logger = logging.getLogger(__name__)  # Added logger

# Get database URL from environment or use default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./banksim.db")

# Create engine with appropriate settings
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )
else:
    # For PostgreSQL, MySQL, etc.
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,  # Recycle connections after 1 hour
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a thread-local scoped session for multi-threaded use
ScopedSession = scoped_session(SessionLocal)

# Base class for all SQLAlchemy model classes
Base = declarative_base()


def get_db() -> Generator:
    """
    Get a database session from the local session factory.
    
    Yields:
        SQLAlchemy session that will be automatically closed when the caller is done
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_scoped_db() -> Generator:
    """
    Get a thread-local scoped database session.
    
    Yields:
        SQLAlchemy scoped session that will be automatically closed when the caller is done
    """
    db = ScopedSession()
    try:
        yield db
    finally:
        ScopedSession.remove()


def init_db() -> None:
    """
    Initialize the database by creating all tables.
    
    This should be called during application startup.
    """
    logger.info("Attempting to initialize database...")
    
    # Ensure all models are imported so Base.metadata registers them.
    # This should be implicitly handled by imports in app.main and app.models.__init__,
    # but an explicit import here can be a safeguard or for clarity.
    try:
        logger.info("Importing model modules to ensure registration with Base...")
        from .. import models # This should trigger app.models.__init__
        # For more explicit control, you might import specific model modules if needed:
        # from ..models import user, economic_indicators, monetary_policy, banking_system, foreign_exchange
        logger.info(f"Models imported. Tables known to Base.metadata BEFORE create_all: {list(Base.metadata.tables.keys())}")
    except ImportError as e:
        logger.error(f"Error importing models in init_db: {e}", exc_info=True)
        # Depending on severity, you might want to raise this or handle

    if not Base.metadata.tables:
        logger.warning("Base.metadata.tables is empty before create_all. No tables will be created unless models are registered.")

    try:
        logger.info(f"Creating tables for database: {engine.url}")
        Base.metadata.create_all(bind=engine)
        logger.info("Base.metadata.create_all() executed.")
    except Exception as e:
        logger.error(f"Error during Base.metadata.create_all(): {e}", exc_info=True)
        raise # Re-raise the exception to ensure startup fails clearly if table creation fails

    # Verify tables in the database file directly after creation for SQLite
    if DATABASE_URL.startswith("sqlite"):
        # Correctly extract the database file path relative to the backend directory
        db_path_parts = DATABASE_URL.split("sqlite:///./")
        if len(db_path_parts) > 1 and db_path_parts[1]:
            db_file_name = db_path_parts[1]
            # Assuming the server runs from the 'backend' directory, db_file_name is relative to it.
            # For absolute path, one might need to join with a base directory if CWD is different.
            db_path_to_check = db_file_name 
            logger.info(f"Verifying tables directly in SQLite file: {db_path_to_check}")
            try:
                conn = sqlite3.connect(db_path_to_check)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables_in_db = cursor.fetchall()
                logger.info(f"Tables found in '{db_path_to_check}' via sqlite3: {tables_in_db}")
                conn.close()
            except sqlite3.Error as e_sqlite:
                logger.error(f"SQLite error verifying tables in '{db_path_to_check}': {e_sqlite}", exc_info=True)
            except Exception as e_verify:
                logger.error(f"Generic error verifying tables in '{db_path_to_check}': {e_verify}", exc_info=True)
        else:
            logger.warning(f"Could not determine SQLite file path from DATABASE_URL: {DATABASE_URL}")
    else:
        logger.info("Database is not SQLite, skipping direct table verification step in init_db.")
    
    logger.info("Database initialization process finished.")
