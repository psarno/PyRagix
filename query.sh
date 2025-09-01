#!/bin/bash

# Activate virtual environment if available
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -d "venv" ]]; then
        source venv/bin/activate
    else
        echo "Warning: No virtual environment found (venv)"
        echo "Continuing with system Python..."
    fi
fi

python query_rag.py