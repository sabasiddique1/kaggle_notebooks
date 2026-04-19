#!/bin/bash
# Quick start script for EmergencyCareNavigator

echo "🚑 Starting EmergencyCareNavigator..."
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Check for Gemini API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY not set. Running in MockLLM mode."
    echo "   (Set GEMINI_API_KEY environment variable for LLM-powered explanations)"
    echo ""
fi

# Start server
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""
uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000






