@echo off
REM Quick Status Check for Legal AI Services

echo.
echo =====================================
echo    Legal AI Services Status Check
echo =====================================
echo.

REM Check PostgreSQL
netstat -an | findstr :5432 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] PostgreSQL     - Port 5432
) else (
    echo [--] PostgreSQL     - Not Running
)

REM Check Ollama
curl -s http://localhost:11434/api/version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Ollama         - Port 11434
) else (
    echo [--] Ollama         - Not Running
)

REM Check Neo4j
netstat -an | findstr :7474 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Neo4j          - Port 7474
) else (
    echo [--] Neo4j          - Not Running
)

REM Check Redis
netstat -an | findstr :6379 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Redis          - Port 6379
) else (
    echo [--] Redis          - Not Running
)

REM Check MinIO
netstat -an | findstr :9000 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] MinIO          - Port 9000
) else (
    echo [--] MinIO          - Not Running
)

REM Check Frontend
curl -s http://localhost:5173 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Frontend       - Port 5173
) else (
    echo [--] Frontend       - Not Running
)

echo.
echo =====================================
echo.
echo URLs:
echo   Frontend:  http://localhost:5173
echo   Neo4j:     http://localhost:7474
echo   MinIO:     http://localhost:9001
echo   Admin:     admin-dashboard.html
echo.
pause