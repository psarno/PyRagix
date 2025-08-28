@echo off
echo Starting TypeScript watch mode...
echo Make sure you have TypeScript installed globally: npm install -g typescript
echo.

REM Check if tsc is available
where tsc >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: TypeScript compiler not found!
    echo Please install TypeScript globally: npm install -g typescript
    pause
    exit /b 1
)

echo TypeScript compiler found. Starting watch mode...
tsc --watch script.ts

pause