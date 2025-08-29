#!/bin/bash

# Activate virtual environment if available
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -d "rag-env" ]]; then
        source rag-env/bin/activate
    elif [[ -d "venv" ]]; then
        source venv/bin/activate
    else
        echo "Warning: No virtual environment found (rag-env or venv)"
        echo "Continuing with system Python..."
    fi
fi

# Run ingestion with optional path argument
if [[ $# -gt 0 ]]; then
    python ingest_folder.py "$1"
else
    python ingest_folder.py
fi