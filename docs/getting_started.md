# Getting Started with BankSim

This guide will help you set up the BankSim development environment and run the application for local development.

## Prerequisites

- **Python**: 3.11 or higher
- **Node.js**: 16.x or higher
- **PostgreSQL**: 14.x or higher
- **Docker**: (Optional) For containerized development

## Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the development server:
   ```bash
   cd app
   uvicorn main:app --reload
   ```

5. Access the API documentation at `http://localhost:8000/docs`

## Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Access the application at `http://localhost:3000`

## Development Workflow

### Backend Development

1. API endpoints are defined in the FastAPI application in `backend/app/`.
2. Core simulation modules will be developed as separate Python packages.
3. Database models are defined using SQLAlchemy.
4. Unit tests are written using pytest.

### Frontend Development

1. React components are organized in a feature-based structure in `frontend/src/components/`.
2. State management is handled with Redux.
3. API calls are made using axios or fetch.
4. Visualizations are created using D3.js.

## Project Structure

For a complete overview of the project architecture, refer to the [Architecture Document](architecture.md).

## Simulation Scenarios

The BankSim platform includes various simulation scenarios for central banking operations:

1. Monetary Policy Implementation
2. Banking Crisis Resolution
3. Foreign Reserve Management
4. Bangladesh-Specific Scenarios

Each scenario can be configured and run through the web interface once fully implemented.

## Troubleshooting

If you encounter any issues during setup or development:

1. Check the API logs for backend errors
2. Verify database connection settings
3. Ensure all dependencies are installed correctly
4. Refer to the documentation or open an issue in the repository
