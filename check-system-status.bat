@echo off
echo Checking Native Windows Legal AI System Status...
echo ═══════════════════════════════════════════════════════════════
echo.

echo 🔍 WINDOWS SERVICES STATUS:
echo ───────────────────────────────────────────────────────────────
echo PostgreSQL Service:
sc query legal-ai-postgres | findstr STATE
if %errorlevel% neq 0 echo   ❌ Service not found

echo.
echo MinIO Service:
sc query MinIO | findstr STATE
if %errorlevel% neq 0 echo   ❌ Service not found

echo.
echo Qdrant Service:
sc query Qdrant | findstr STATE  
if %errorlevel% neq 0 echo   ❌ Service not found

echo.
echo Redis Service:
sc query Redis | findstr STATE
if %errorlevel% neq 0 echo   ❌ Service not found

echo.
echo 🌐 PORT STATUS:
echo ───────────────────────────────────────────────────────────────
echo PostgreSQL (5432):
netstat -an | findstr :5432 | findstr LISTENING
if %errorlevel% neq 0 echo   ❌ Port 5432 not listening

echo.
echo MinIO API (9000):
netstat -an | findstr :9000 | findstr LISTENING
if %errorlevel% neq 0 echo   ❌ Port 9000 not listening

echo.
echo MinIO Console (9001):
netstat -an | findstr :9001 | findstr LISTENING
if %errorlevel% neq 0 echo   ❌ Port 9001 not listening

echo.
echo Qdrant (6333):
netstat -an | findstr :6333 | findstr LISTENING
if %errorlevel% neq 0 echo   ❌ Port 6333 not listening

echo.
echo Redis (6379):
netstat -an | findstr :6379 | findstr LISTENING
if %errorlevel% neq 0 echo   ❌ Port 6379 not listening

echo.
echo Ollama (11434):
netstat -an | findstr :11434 | findstr LISTENING
if %errorlevel% neq 0 echo   ❌ Port 11434 not listening

echo.
echo Go Backend (8084):
netstat -an | findstr :8084 | findstr LISTENING
if %errorlevel% neq 0 echo   ❌ Port 8084 not listening

echo.
echo SvelteKit Frontend (5173):
netstat -an | findstr :5173 | findstr LISTENING
if %errorlevel% neq 0 echo   ❌ Port 5173 not listening

echo.
echo 🔄 PROCESS STATUS:
echo ───────────────────────────────────────────────────────────────
echo Ollama Process:
tasklist /FI "IMAGENAME eq ollama.exe" | findstr ollama.exe
if %errorlevel% neq 0 echo   ❌ Ollama not running

echo.
echo Node.js Processes:
tasklist /FI "IMAGENAME eq node.exe" | findstr node.exe
if %errorlevel% neq 0 echo   ❌ No Node.js processes running

echo.
echo Go Processes:
tasklist /FI "IMAGENAME eq main.exe" | findstr main.exe
tasklist /FI "IMAGENAME eq go.exe" | findstr go.exe
if %errorlevel% neq 0 echo   ❌ No Go processes running

echo.
echo 🗂️ DATA DIRECTORY STATUS:
echo ───────────────────────────────────────────────────────────────
if exist "data\postgres" (
    echo   ✅ PostgreSQL data directory exists
    for /f %%i in ('dir /s /b "data\postgres\*.*" 2^>nul ^| find /c /v ""') do echo      Files: %%i
) else (
    echo   ❌ PostgreSQL data directory missing
)

if exist "data\minio" (
    echo   ✅ MinIO data directory exists
    for /f %%i in ('dir /s /b "data\minio\*.*" 2^>nul ^| find /c /v ""') do echo      Files: %%i
) else (
    echo   ❌ MinIO data directory missing
)

if exist "data\qdrant" (
    echo   ✅ Qdrant data directory exists
    for /f %%i in ('dir /s /b "data\qdrant\*.*" 2^>nul ^| find /c /v ""') do echo      Files: %%i
) else (
    echo   ❌ Qdrant data directory missing
)

if exist "data\redis" (
    echo   ✅ Redis data directory exists
) else (
    echo   ❌ Redis data directory missing
)

echo.
echo 🌍 CONNECTIVITY TEST:
echo ───────────────────────────────────────────────────────────────
echo Testing PostgreSQL connection...
where psql >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ psql command available
) else (
    echo   ⚠️ psql not in PATH
)

echo.
echo Testing MinIO client...
if exist "C:\minio\mc.exe" (
    echo   ✅ MinIO client available
) else (
    echo   ⚠️ MinIO client not found
)

echo.
echo Testing Ollama...
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ Ollama command available
    ollama list 2>nul | findstr "nomic-embed-text"
    if %errorlevel% equ 0 (
        echo   ✅ Embedding model ready
    ) else (
        echo   ⚠️ Embedding model not downloaded
    )
) else (
    echo   ⚠️ Ollama not in PATH
)

echo.
echo 📊 SYSTEM RESOURCES:
echo ───────────────────────────────────────────────────────────────
echo Memory Usage:
wmic computersystem get TotalPhysicalMemory /value | findstr "="
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value | findstr "="

echo.
echo Disk Space (C:):
for /f "tokens=3" %%i in ('dir C:\ ^| findstr "free"') do echo   Available: %%i

echo.
echo ═══════════════════════════════════════════════════════════════
echo 🎯 QUICK ACTIONS:
echo   • Start system:    start-native-system.bat
echo   • Stop system:     stop-native-system.bat
echo   • Fix errors:      node scripts\fix-svelte5-errors.mjs
echo   • Setup services:  scripts\native-windows\setup-native-windows.ps1
echo.
echo 🌐 WEB INTERFACES:
echo   • Frontend:        http://localhost:5173
echo   • MinIO Console:   http://localhost:9001
echo   • Qdrant API:      http://localhost:6333
echo   • Backend Health:  http://localhost:8084/api/health
echo.
pause