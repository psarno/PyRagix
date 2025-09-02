@echo off
chcp 65001>nul
echo ======================================
echo 🚀 RAG Web Interface Startup Script
echo ======================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo 🔄 Activating virtual environment...
    if exist "venv\Scripts\activate.bat" (
        call venv\Scripts\activate.bat
    ) else (
        echo ⚠️ Warning: No virtual environment found (venv)
        echo 🐍 Continuing with system Python...
    )
) else (
    echo ✅ Virtual environment already active: %VIRTUAL_ENV%
)

echo 📦 Checking Python dependencies...
python -c "import fastapi, uvicorn" 2>nul
if %errorlevel% neq 0 (
    echo ⬇️ Installing missing dependencies...
    pip install fastapi uvicorn[standard]
)

echo 🎮 Setting up GPU environment for Ollama...
set CUDA_VISIBLE_DEVICES=0
set OLLAMA_GPU_COMPUTE_CAPABILITY=7.5
set OLLAMA_NUM_GPU=1

echo 🔧 Compiling TypeScript...

cd web

if exist "script.ts" (
    call tsc
    if %errorlevel% neq 0 (
        echo ⚠️ Warning: TypeScript compilation failed, but continuing...
        echo 💡 Check your TypeScript installation with: npm install -g typescript
    ) else (
        echo ✅ TypeScript compiled successfully!
    )
) else (
    echo 📄 No TypeScript files found to compile
)

cd ..

echo 🌐 Starting RAG Web Server...
echo 🖥️ Web interface: http://localhost:8000/web/
echo 📚 API docs: http://localhost:8000/docs
echo.

python web_server.py

pause