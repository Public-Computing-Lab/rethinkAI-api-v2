#!/bin/bash
# Daily Ingestion Cron Job Script
# Runs the complete daily ingestion pipeline at 2 AM
# This script should be called from cron

set -e  # Exit on error

# Configuration
PROJECT_DIR="PATH_TO_PROJECT"
VENV_DIR="$PROJECT_DIR/venv"
INGESTION_DIR="$PROJECT_DIR/main_chat/data_ingestion"
LOG_DIR="$PROJECT_DIR/logs"
SCRIPT_LOG="$LOG_DIR/daily_ingestion.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "Checking environment..."
for dir in "$PROJECT_DIR" "$VENV_DIR" "$INGESTION_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Directory does not exist: $dir"
        exit 1
    fi    
done

cd "$INGESTION_DIR" || exit 1
#echo "Changed to: $(pwd)"
echo "  Project paths OK."
source "$VENV_DIR/bin/activate"

if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Virtual environment failed to activate"
    exit 1
fi
#echo "Python location: $(which python3)"
echo "  Venv activated, using $(python3 --version)"

# Run the main daily ingestion script
{
    START_TIME=$(date +%s)
    echo "========================================" 
    echo " Daily Ingestion Started:" 
    echo " $(date)" 
    echo "========================================" 
    
    python3 main_daily_ingestion.py -p -l quiet  2>&1
    
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    DURATION_SEC=$((DURATION % 60))
    
    echo " Completed: $(date)"
    echo " Duration: ${DURATION_MIN}m ${DURATION_SEC}s"    
    echo "" 
    
} | tee -a "$SCRIPT_LOG"

# Exit with the same code as the Python script
exit $EXIT_CODE