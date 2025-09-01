@echo off
REM Activate virtual environment if available
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found (venv^)
    echo Continuing with system Python...
)

python query_rag.py

pause