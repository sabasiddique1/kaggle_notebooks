@echo off
REM Quick start script for EmergencyCareNavigator (Windows)

echo 🚑 Starting EmergencyCareNavigator...
echo.

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies if needed
if not exist "venv\.installed" (
    echo Installing dependencies...
    pip install -r requirements.txt
    type nul > venv\.installed
)

REM Check for Gemini API key
if "%GEMINI_API_KEY%"=="" (
    echo ⚠️  GEMINI_API_KEY not set. Running in MockLLM mode.
    echo    (Set GEMINI_API_KEY environment variable for LLM-powered explanations)
    echo.
)

REM Start server
echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop
echo.
uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000






