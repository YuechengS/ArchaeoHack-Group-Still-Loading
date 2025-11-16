#!/bin/bash

echo "==================================="
echo "Egyptian Hieroglyphs Learning App"
echo "==================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Navigate to backend directory
cd backend

# Install dependencies if needed
echo "Installing backend dependencies..."
pip install -r requirements.txt

echo ""
echo "Starting Flask backend server..."
echo "Server will run on http://localhost:5000"
echo ""
echo "To use the app:"
echo "1. Keep this terminal running (Flask backend)"
echo "2. Open frontend/index.html in a web browser"
echo "3. Draw hieroglyphs on the canvas!"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Flask app
python3 app.py
