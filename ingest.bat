@echo off
REM Activate virtual environment if available
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found (venv^)
    echo Continuing with system Python...
)

REM Run ingestion with all arguments (path, flags, etc.)
python ingest_folder.py %*

pause