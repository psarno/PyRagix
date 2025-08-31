#!/bin/bash
echo "======================================"
echo "  RAG Web Interface Startup Script"
echo "======================================"
echo

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    if [[ -f "rag-env/bin/activate" ]]; then
        source rag-env/bin/activate
    elif [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    else
        echo "Warning: No virtual environment found (rag-env or venv)"
        echo "Continuing with system Python..."
    fi
else
    echo "Virtual environment already active: $VIRTUAL_ENV"
fi

echo
echo "Checking Python dependencies..."
python -c "import fastapi, uvicorn" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "Installing missing dependencies..."
    pip install fastapi uvicorn[standard]
fi

echo
echo "Setting up GPU environment for Ollama..."
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_COMPUTE_CAPABILITY=7.5
export OLLAMA_NUM_GPU=1

echo "Starting RAG Web Server..."
echo "Web interface: http://localhost:8000/web/"
echo "API docs: http://localhost:8000/docs"
echo

python web_server.py