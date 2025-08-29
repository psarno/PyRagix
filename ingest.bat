@echo off
REM Activate virtual environment if available
if exist "rag-env\Scripts\activate.bat" (
    call rag-env\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found (rag-env or venv^)
    echo Continuing with system Python...
)

REM Run ingestion with optional path argument
if "%~1"=="" (
    python ingest_folder.py
) else (
    python ingest_folder.py "%~1"
)

pause