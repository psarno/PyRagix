@echo off
echo ======================================
echo  RAG Web Interface Startup Script
echo ======================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    if exist "rag-env" (
        call rag-env\Scripts\activate.bat
    ) else if exist "venv" (
        call venv\Scripts\activate.bat
    ) else (
        echo Warning: No virtual environment found (rag-env or venv)
        echo Continuing with system Python...
    )
) else (
    echo Virtual environment already active: %VIRTUAL_ENV%
)

echo.
echo Checking Python dependencies...
python -c "import fastapi, uvicorn" 2>nul
if %errorlevel% neq 0 (
    echo Installing missing dependencies...
    pip install fastapi uvicorn[standard]
)

echo.
echo Setting up GPU environment for Ollama...
set CUDA_VISIBLE_DEVICES=0
set OLLAMA_GPU_COMPUTE_CAPABILITY=7.5
set OLLAMA_NUM_GPU=1

echo Starting RAG Web Server...
echo Web interface: http://localhost:8000/web/
echo API docs: http://localhost:8000/docs
echo.

python web_server.py

pause