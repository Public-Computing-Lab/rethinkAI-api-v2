#!/bin/bash
# Daily Ingestion Cron Job Script
# Runs the complete daily ingestion pipeline at 2 AM
# This script should be called from cron

set -e  # Exit on error

# Configuration
PROJECT_DIR="$HOME/source/Spark-F25"
VENV_DIR="$PROJECT_DIR/venv"
INGESTION_DIR="$PROJECT_DIR/main_chat/data_ingestion"
LOG_DIR="$PROJECT_DIR/logs"
SCRIPT_LOG="$LOG_DIR/daily_ingestion.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "Checking paths..."
echo "Project dir exists: $([ -d "$PROJECT_DIR" ] && echo YES || echo NO)"
echo "Venv dir exists: $([ -d "$VENV_DIR" ] && echo YES || echo NO)"
echo "Ingestion dir exists: $([ -d "$INGESTION_DIR" ] && echo YES || echo NO)"

cd "$INGESTION_DIR" || exit 1
echo "Changed to: $(pwd)"

source "$VENV_DIR/bin/activate"
echo "Python location: $(which python3)"

# Run the main daily ingestion script
{
    echo "==========================================" 
    echo "Daily Ingestion Started: $(date)" 
    echo "==========================================" 
    
    python3 main_daily_ingestion.py  2>&1
    
    EXIT_CODE=$?
    
    echo "==========================================" 
    echo "Daily Ingestion Finished: $(date) (Exit Code: $EXIT_CODE)" 
    echo "==========================================" 
    echo "" 
} | tee -a "$SCRIPT_LOG"

# Exit with the same code as the Python script
exit $EXIT_CODE

