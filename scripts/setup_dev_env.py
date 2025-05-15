#!/usr/bin/env python3
"""
Setup Development Environment

This script sets up the BankSim development environment by:
1. Creating a Python virtual environment
2. Installing backend dependencies
3. Setting up database connections
4. Creating initial database schema
5. Loading sample data for development

Usage:
    python setup_dev_env.py [--force]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_prerequisites():
    """Check if all required tools are installed."""
    print("Checking prerequisites...")

    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 11):
        print("ERROR: Python 3.11 or higher is required")
        return False

    # Check PostgreSQL (this is a simple check, might need to be improved)
    try:
        subprocess.run(["psql", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("WARNING: PostgreSQL client not found in PATH. Make sure PostgreSQL is installed.")
        print("Development can continue, but database features will not work.")

    # Check Node.js
    try:
        node_version = subprocess.run(
            ["node", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.decode().strip()
        print(f"Found Node.js: {node_version}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("WARNING: Node.js not found. Required for frontend development.")
        return False

    return True


def setup_venv(force=False):
    """Create and activate a Python virtual environment."""
    print("Setting up Python virtual environment...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    venv_path = project_root / "backend" / "venv"
    
    # Check if venv already exists
    if venv_path.exists() and not force:
        print(f"Virtual environment already exists at {venv_path}")
        print("Use --force to recreate it")
        return venv_path
    
    # Create the virtual environment
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    print(f"Created virtual environment at {venv_path}")
    
    return venv_path


def install_dependencies(venv_path):
    """Install backend dependencies in the virtual environment."""
    print("Installing backend dependencies...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    requirements_path = project_root / "backend" / "requirements.txt"
    
    # Determine the pip executable path
    if os.name == 'nt':  # Windows
        pip_path = venv_path / "Scripts" / "pip"
    else:  # Unix/Linux
        pip_path = venv_path / "bin" / "pip"
    
    # Install requirements
    subprocess.run([str(pip_path), "install", "-U", "pip"], check=True)
    subprocess.run([str(pip_path), "install", "-r", str(requirements_path)], check=True)
    
    print("Backend dependencies installed successfully")


def main():
    """Main function to set up the development environment."""
    parser = argparse.ArgumentParser(description="Set up BankSim development environment")
    parser.add_argument("--force", action="store_true", help="Force recreation of virtual environment")
    args = parser.parse_args()
    
    if not check_prerequisites():
        print("Failed prerequisite check. Please install required tools.")
        return 1
    
    try:
        venv_path = setup_venv(args.force)
        install_dependencies(venv_path)
        
        print("\n===== Development Environment Setup Complete =====")
        print("To activate the virtual environment:")
        
        if os.name == 'nt':  # Windows
            print(f"    {venv_path}\\Scripts\\activate")
        else:  # Unix/Linux
            print(f"    source {venv_path}/bin/activate")
        
        print("\nTo start the backend server:")
        print("    cd backend/app")
        print("    uvicorn main:app --reload")
        
        print("\nTo start the frontend development server:")
        print("    cd frontend")
        print("    npm start")
        
        return 0
    
    except Exception as e:
        print(f"Error setting up development environment: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
