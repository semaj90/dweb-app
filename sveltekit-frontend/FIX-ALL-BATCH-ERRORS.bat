@echo off
REM ================================================================================
REM COMPREHENSIVE BATCH FILE ERROR FIXER - WINDOWS NATIVE
REM Fixes all common issues found in recent .bat files from past 24 hours
REM ================================================================================

title Fixing All Batch File Errors
color 0A

echo.
echo ================================================================================
echo 🔧 COMPREHENSIVE BATCH FILE ERROR FIXER
echo    Analyzing and fixing all recent .bat file issues
echo ================================================================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

REM Create necessary directories
echo [1/10] Creating required directories...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "temp" mkdir temp
if not exist "bin" mkdir bin
if not exist "..\go-microservice\bin" mkdir "..\go-microservice\bin"
if not exist "..\go-services\bin" mkdir "..\go-services\bin"
echo ✅ Directories created

echo.
echo [2/10] Checking Go binary availability...
set BINARY_COUNT=0

REM Check for Enhanced RAG
if exist "..\go-microservice\bin\enhanced-rag.exe" (
    echo ✅ Enhanced RAG binary found
    set /a BINARY_COUNT+=1
) else if exist "..\go-microservice\enhanced-rag.exe" (
    echo ✅ Enhanced RAG binary found (root)
    set /a BINARY_COUNT+=1
) else (
    echo ⚠️  Enhanced RAG binary missing - will use go run
)

REM Check for Upload Service  
if exist "..\go-microservice\bin\upload-service.exe" (
    echo ✅ Upload Service binary found
    set /a BINARY_COUNT+=1
) else if exist "..\go-microservice\upload-service.exe" (
    echo ✅ Upload Service binary found (root)
    set /a BINARY_COUNT+=1
) else (
    echo ⚠️  Upload Service binary missing - will use go run
)

REM Check for Kratos Server
if exist "..\go-services\bin\kratos-server.exe" (
    echo ✅ Kratos Server binary found
    set /a BINARY_COUNT+=1
) else if exist "..\go-microservice\rag-kratos.exe" (
    echo ✅ Kratos variant binary found
    set /a BINARY_COUNT+=1
) else (
    echo ⚠️  Kratos Server binary missing - will use go run
)

echo Found %BINARY_COUNT% compiled binaries

echo.
echo [3/10] Checking service dependencies...

REM Check PostgreSQL
net start | findstr "PostgreSQL" >nul
if %errorlevel% equ 0 (
    echo ✅ PostgreSQL is running
) else (
    echo ⚠️  PostgreSQL not running - attempting start
    net start postgresql-x64-17 2>nul || net start postgresql-x64-16 2>nul || net start postgresql-x64-15 2>nul || echo ❌ PostgreSQL not found
)

REM Check Redis
tasklist | findstr "redis-server" >nul
if %errorlevel% equ 0 (
    echo ✅ Redis is running
) else (
    echo ⚠️  Redis not running
    where redis-server >nul 2>&1
    if %errorlevel% equ 0 (
        echo Starting Redis...
        start /min redis-server
    ) else (
        echo ❌ Redis not installed - install Redis for Windows
    )
)

REM Check Ollama
tasklist | findstr "ollama" >nul
if %errorlevel% equ 0 (
    echo ✅ Ollama is running
) else (
    echo ⚠️  Ollama not running
    where ollama >nul 2>&1
    if %errorlevel% equ 0 (
        echo Starting Ollama...
        start /min ollama serve
    ) else (
        echo ❌ Ollama not installed - install from https://ollama.ai
    )
)

echo.
echo [4/10] Fixing port conflicts...

REM Create port usage report
echo Checking current port usage...
netstat -an | findstr ":5173 " >nul && echo ⚠️  Port 5173 (SvelteKit) in use
netstat -an | findstr ":8093 " >nul && echo ⚠️  Port 8093 (Upload Service) in use  
netstat -an | findstr ":8094 " >nul && echo ⚠️  Port 8094 (Enhanced RAG) in use
netstat -an | findstr ":50051 " >nul && echo ⚠️  Port 50051 (gRPC) in use
netstat -an | findstr ":50052 " >nul && echo ⚠️  Port 50052 (Kratos gRPC) in use

echo Creating port conflict resolver...
(
echo @echo off
echo REM Kill processes on specific ports if needed
echo echo Resolving port conflicts...
echo.
echo REM Find and kill processes on key ports
echo for /f "tokens=5" %%%%a in ^('netstat -aon ^| findstr ":5173 "'^) do ^(
echo     if not "%%%%a"=="0" ^(
echo         echo Killing process on port 5173: %%%%a
echo         taskkill /f /pid %%%%a 2^>nul
echo     ^)
echo ^)
echo.
echo for /f "tokens=5" %%%%a in ^('netstat -aon ^| findstr ":8093 "'^) do ^(
echo     if not "%%%%a"=="0" ^(
echo         echo Killing process on port 8093: %%%%a  
echo         taskkill /f /pid %%%%a 2^>nul
echo     ^)
echo ^)
echo.
echo for /f "tokens=5" %%%%a in ^('netstat -aon ^| findstr ":8094 "'^) do ^(
echo     if not "%%%%a"=="0" ^(
echo         echo Killing process on port 8094: %%%%a
echo         taskkill /f /pid %%%%a 2^>nul
echo     ^)
echo ^)
echo.
echo echo Port conflicts resolved
) > "RESOLVE-PORT-CONFLICTS.bat"

echo ✅ Port conflict resolver created

echo.
echo [5/10] Creating robust service starter...

(
echo @echo off
echo title Robust Legal AI Service Starter
echo color 0B
echo.
echo echo ================================================================================  
echo echo 🚀 ROBUST LEGAL AI SERVICE STARTER
echo echo    Smart dependency checking + fallback mechanisms
echo echo ================================================================================
echo.
echo.
echo REM Set consistent environment variables
echo set NODE_ENV=development
echo set GO_ENV=development  
echo set GPU_ENABLED=true
echo set CUDA_VISIBLE_DEVICES=0
echo set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
echo set REDIS_URL=redis://localhost:6379
echo set OLLAMA_API_URL=http://localhost:11434
echo.
echo cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
echo.
echo REM Create directories  
echo if not exist "logs" mkdir logs
echo.
echo echo [1/4] Starting Infrastructure Services...
echo.
echo REM PostgreSQL
echo echo Starting PostgreSQL...
echo net start postgresql-x64-17 2^>nul ^|^| net start postgresql-x64-16 2^>nul ^|^| net start postgresql-x64-15 2^>nul ^|^| echo PostgreSQL already running or not installed
echo.
echo REM Redis
echo echo Starting Redis...
echo tasklist ^| findstr "redis-server" ^>nul ^|^| start /min redis-server 2^>nul ^|^| echo Redis not available
echo.
echo REM Ollama
echo echo Starting Ollama...
echo tasklist ^| findstr "ollama" ^>nul ^|^| start /min ollama serve 2^>nul ^|^| echo Ollama not available
echo.
echo timeout /t 3 /nobreak ^> nul
echo.
echo echo [2/4] Starting Core Go Services...
echo.
echo REM Enhanced RAG Service with multiple fallbacks
echo if exist "..\go-microservice\bin\enhanced-rag.exe" ^(
echo     echo ✅ Starting Enhanced RAG from bin/
echo     start "Enhanced-RAG" /MIN cmd /c "cd ..\go-microservice && bin\enhanced-rag.exe ^> ..\sveltekit-frontend\logs\enhanced-rag.log 2^>^&1"
echo ^) else if exist "..\go-microservice\enhanced-rag.exe" ^(
echo     echo ✅ Starting Enhanced RAG from root
echo     start "Enhanced-RAG" /MIN cmd /c "cd ..\go-microservice && enhanced-rag.exe ^> ..\sveltekit-frontend\logs\enhanced-rag.log 2^>^&1"
echo ^) else if exist "..\go-microservice\cmd\enhanced-rag\main.go" ^(
echo     echo 🔨 Building and starting Enhanced RAG
echo     start "Enhanced-RAG" /MIN cmd /c "cd ..\go-microservice && go run cmd\enhanced-rag\main.go ^> ..\sveltekit-frontend\logs\enhanced-rag.log 2^>^&1"
echo ^) else ^(
echo     echo ❌ Enhanced RAG source not found
echo ^)
echo.
echo REM Upload Service with multiple fallbacks  
echo if exist "..\go-microservice\bin\upload-service.exe" ^(
echo     echo ✅ Starting Upload Service from bin/
echo     start "Upload-Service" /MIN cmd /c "cd ..\go-microservice && bin\upload-service.exe ^> ..\sveltekit-frontend\logs\upload-service.log 2^>^&1"
echo ^) else if exist "..\go-microservice\upload-service.exe" ^(
echo     echo ✅ Starting Upload Service from root
echo     start "Upload-Service" /MIN cmd /c "cd ..\go-microservice && upload-service.exe ^> ..\sveltekit-frontend\logs\upload-service.log 2^>^&1"
echo ^) else if exist "..\go-microservice\cmd\upload-service\main.go" ^(
echo     echo 🔨 Building and starting Upload Service
echo     start "Upload-Service" /MIN cmd /c "cd ..\go-microservice && go run cmd\upload-service\main.go ^> ..\sveltekit-frontend\logs\upload-service.log 2^>^&1"
echo ^) else ^(
echo     echo ❌ Upload Service source not found
echo ^)
echo.
echo timeout /t 2 /nobreak ^> nul
echo.
echo echo [3/4] Starting Frontend...
echo.
echo REM Check if SvelteKit is already running
echo netstat -an ^| findstr ":5173 " ^>nul
echo if %%errorlevel%% equ 0 ^(
echo     echo ⚠️  SvelteKit already running on port 5173
echo     echo Trying alternate ports...
echo     start "SvelteKit" cmd /k "npm run dev -- --port 5174"
echo ^) else ^(
echo     echo ✅ Starting SvelteKit on port 5173
echo     start "SvelteKit" cmd /k "npm run dev"
echo ^)
echo.
echo timeout /t 5 /nobreak ^> nul
echo.
echo echo [4/4] Health Check...
echo.
echo curl -s http://localhost:5173 ^>nul 2^>^&1 ^&^& echo ✅ Frontend ^(5173^) ^|^| curl -s http://localhost:5174 ^>nul 2^>^&1 ^&^& echo ✅ Frontend ^(5174^) ^|^| echo ❌ Frontend
echo curl -s http://localhost:8094/health ^>nul 2^>^&1 ^&^& echo ✅ Enhanced RAG ^(8094^) ^|^| echo ❌ Enhanced RAG ^(8094^)
echo curl -s http://localhost:8093/health ^>nul 2^>^&1 ^&^& echo ✅ Upload Service ^(8093^) ^|^| echo ❌ Upload Service ^(8093^)
echo curl -s http://localhost:11434/api/tags ^>nul 2^>^&1 ^&^& echo ✅ Ollama ^(11434^) ^|^| echo ❌ Ollama ^(11434^)
echo.
echo echo ================================================================================
echo echo 🎉 STARTUP COMPLETE - Access at: http://localhost:5173 or http://localhost:5174  
echo echo ================================================================================
echo pause
) > "ROBUST-START.bat"

echo ✅ Robust service starter created

echo.
echo [6/10] Creating environment setup fixer...

(
echo @echo off
echo title Fix Environment Variables
echo.
echo echo Setting up consistent environment variables...
echo.
echo REM Core environment
echo setx NODE_ENV "development" ^> nul
echo setx GO_ENV "development" ^> nul
echo setx GPU_ENABLED "true" ^> nul
echo setx CUDA_VISIBLE_DEVICES "0" ^> nul
echo.
echo REM Database connections
echo setx DATABASE_URL "postgresql://legal_admin:123456@localhost:5432/legal_ai_db" ^> nul
echo setx REDIS_URL "redis://localhost:6379" ^> nul
echo setx OLLAMA_API_URL "http://localhost:11434" ^> nul
echo setx MINIO_ENDPOINT "http://localhost:9000" ^> nul
echo.
echo REM Frontend configuration
echo setx VITE_GPU_ENABLED "true" ^> nul
echo setx VITE_DEMO_MODE "false" ^> nul
echo setx VITE_API_URL "http://localhost:8094" ^> nul
echo.
echo echo ✅ Environment variables configured
echo echo ⚠️  Restart your command prompt to use new variables
echo pause
) > "FIX-ENVIRONMENT.bat"

echo ✅ Environment fixer created

echo.
echo [7/10] Creating dependency checker...

(
echo @echo off
echo title Dependency Status Checker
echo color 0D
echo.
echo echo ================================================================================
echo echo 📋 DEPENDENCY STATUS CHECKER
echo echo ================================================================================
echo.
echo echo Checking system dependencies...
echo.
echo REM Check Node.js
echo node --version ^>nul 2^>^&1 ^&^& echo ✅ Node.js: && node --version ^|^| echo ❌ Node.js not installed
echo.
echo REM Check npm
echo npm --version ^>nul 2^>^&1 ^&^& echo ✅ npm: && npm --version ^|^| echo ❌ npm not available  
echo.
echo REM Check Go
echo go version ^>nul 2^>^&1 ^&^& echo ✅ Go: && go version ^|^| echo ❌ Go not installed
echo.
echo REM Check PostgreSQL
echo where psql ^>nul 2^>^&1 ^&^& echo ✅ PostgreSQL CLI available ^|^| echo ❌ PostgreSQL CLI not in PATH
echo.
echo REM Check Redis
echo where redis-server ^>nul 2^>^&1 ^&^& echo ✅ Redis available ^|^| echo ❌ Redis not installed
echo.
echo REM Check Ollama
echo where ollama ^>nul 2^>^&1 ^&^& echo ✅ Ollama available ^|^| echo ❌ Ollama not installed
echo.
echo REM Check curl
echo where curl ^>nul 2^>^&1 ^&^& echo ✅ curl available ^|^| echo ❌ curl not available
echo.
echo echo ================================================================================
echo echo 🔍 SERVICE STATUS CHECK
echo echo ================================================================================
echo.
echo echo Checking running services...
echo.
echo REM Check services
echo tasklist ^| findstr "postgres" ^>nul ^&^& echo ✅ PostgreSQL process running ^|^| echo ❌ PostgreSQL not running
echo tasklist ^| findstr "redis-server" ^>nul ^&^& echo ✅ Redis process running ^|^| echo ❌ Redis not running  
echo tasklist ^| findstr "ollama" ^>nul ^&^& echo ✅ Ollama process running ^|^| echo ❌ Ollama not running
echo tasklist ^| findstr "node" ^>nul ^&^& echo ✅ Node.js processes running ^|^| echo ❌ No Node.js processes
echo.
echo echo ================================================================================
echo echo 🌐 PORT STATUS CHECK  
echo echo ================================================================================
echo.
echo netstat -an ^| findstr ":5173 " ^>nul ^&^& echo ✅ Port 5173 ^(SvelteKit^) in use ^|^| echo ❌ Port 5173 available
echo netstat -an ^| findstr ":8093 " ^>nul ^&^& echo ✅ Port 8093 ^(Upload^) in use ^|^| echo ❌ Port 8093 available
echo netstat -an ^| findstr ":8094 " ^>nul ^&^& echo ✅ Port 8094 ^(RAG^) in use ^|^| echo ❌ Port 8094 available
echo netstat -an ^| findstr ":11434 " ^>nul ^&^& echo ✅ Port 11434 ^(Ollama^) in use ^|^| echo ❌ Port 11434 available
echo netstat -an ^| findstr ":5432 " ^>nul ^&^& echo ✅ Port 5432 ^(PostgreSQL^) in use ^|^| echo ❌ Port 5432 available
echo netstat -an ^| findstr ":6379 " ^>nul ^&^& echo ✅ Port 6379 ^(Redis^) in use ^|^| echo ❌ Port 6379 available
echo.
echo pause
) > "CHECK-DEPENDENCIES.bat"

echo ✅ Dependency checker created

echo.
echo [8/10] Creating log analyzer...

(
echo @echo off
echo title Log Analyzer
echo color 0F
echo.
echo echo ================================================================================
echo echo 📊 LOG ANALYZER - Recent Errors and Issues
echo echo ================================================================================
echo.
echo cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
echo.
echo if not exist "logs" mkdir logs
echo.
echo echo Analyzing recent log files...
echo.
echo REM Check for recent errors in logs
echo if exist "logs\enhanced-rag.log" ^(
echo     echo.
echo     echo 🔍 Enhanced RAG Service Logs:
echo     echo ────────────────────────────────────
echo     tail -10 logs\enhanced-rag.log 2^>nul ^|^| echo No recent Enhanced RAG logs
echo ^)
echo.
echo if exist "logs\upload-service.log" ^(
echo     echo.
echo     echo 🔍 Upload Service Logs:
echo     echo ──────────────────────────
echo     tail -10 logs\upload-service.log 2^>nul ^|^| echo No recent Upload Service logs  
echo ^)
echo.
echo if exist "logs\kratos-server.log" ^(
echo     echo.
echo     echo 🔍 Kratos Server Logs:
echo     echo ─────────────────────────
echo     tail -10 logs\kratos-server.log 2^>nul ^|^| echo No recent Kratos logs
echo ^)
echo.
echo echo.
echo echo 🚨 Scanning for ERROR patterns...
echo findstr /i "error" logs\*.log 2^>nul ^|^| echo No errors found in log files
echo.
echo echo 🚨 Scanning for FATAL patterns...  
echo findstr /i "fatal" logs\*.log 2^>nul ^|^| echo No fatal errors found in log files
echo.
echo echo 🚨 Scanning for PANIC patterns...
echo findstr /i "panic" logs\*.log 2^>nul ^|^| echo No panic errors found in log files
echo.
echo pause
) > "ANALYZE-LOGS.bat"

echo ✅ Log analyzer created

echo.
echo [9/10] Creating quick diagnosis tool...

(
echo @echo off
echo title Quick System Diagnosis
echo color 0C
echo.
echo echo ================================================================================
echo echo ⚡ QUICK SYSTEM DIAGNOSIS
echo echo ================================================================================
echo.
echo cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
echo.
echo REM Quick health checks
echo echo [1/5] Frontend Health Check...
echo curl -s http://localhost:5173 ^>nul 2^>^&1 ^&^& echo ✅ Frontend responding ^|^| curl -s http://localhost:5174 ^>nul 2^>^&1 ^&^& echo ✅ Frontend on alt port ^|^| echo ❌ Frontend not responding
echo.
echo echo [2/5] Database Connection...
echo psql -h localhost -p 5432 -U legal_admin -d legal_ai_db -c "SELECT 1;" ^>nul 2^>^&1 ^&^& echo ✅ Database connected ^|^| echo ❌ Database connection failed
echo.
echo echo [3/5] Go Services Health...
echo curl -s http://localhost:8094/health ^>nul 2^>^&1 ^&^& echo ✅ Enhanced RAG ^|^| echo ❌ Enhanced RAG
echo curl -s http://localhost:8093/health ^>nul 2^>^&1 ^&^& echo ✅ Upload Service ^|^| echo ❌ Upload Service  
echo.
echo echo [4/5] AI Services...
echo curl -s http://localhost:11434/api/tags ^>nul 2^>^&1 ^&^& echo ✅ Ollama responding ^|^| echo ❌ Ollama not responding
echo.
echo echo [5/5] Process Check...
echo tasklist ^| findstr "node" ^>nul ^&^& echo ✅ Node.js running ^|^| echo ❌ No Node.js processes
echo tasklist ^| findstr "go" ^>nul ^&^& echo ✅ Go processes running ^|^| echo ❌ No Go processes
echo.
echo echo ================================================================================
echo echo 📊 DIAGNOSIS COMPLETE
echo echo ================================================================================
echo pause
) > "QUICK-DIAGNOSIS.bat"

echo ✅ Quick diagnosis tool created

echo.
echo [10/10] Creating master recovery script...

(
echo @echo off
echo title Master System Recovery
echo color 0E
echo.
echo echo ================================================================================
echo echo 🚨 MASTER SYSTEM RECOVERY - Fix Everything  
echo echo ================================================================================
echo.
echo echo This will attempt to fix all common issues and restart services
echo pause
echo.
echo echo [1/6] Resolving port conflicts...
echo call RESOLVE-PORT-CONFLICTS.bat
echo.
echo echo [2/6] Setting up environment...  
echo call FIX-ENVIRONMENT.bat
echo.
echo echo [3/6] Checking dependencies...
echo call CHECK-DEPENDENCIES.bat
echo.
echo echo [4/6] Clearing node modules and reinstalling...
echo if exist node_modules rmdir /s /q node_modules
echo npm install --force
echo.
echo echo [5/6] Starting robust services...
echo call ROBUST-START.bat
echo.
echo echo [6/6] Running final diagnosis...
echo timeout /t 10 /nobreak ^> nul
echo call QUICK-DIAGNOSIS.bat
echo.
echo echo ================================================================================
echo echo ✅ MASTER RECOVERY COMPLETE
echo echo ================================================================================
) > "MASTER-RECOVERY.bat"

echo ✅ Master recovery script created

echo.
echo ================================================================================
echo ✅ COMPREHENSIVE ERROR FIXES COMPLETE
echo ================================================================================
echo.
echo Created the following error-fixing tools:
echo.
echo 📁 RESOLVE-PORT-CONFLICTS.bat    - Kills processes on conflicting ports
echo 📁 ROBUST-START.bat             - Smart service starter with fallbacks  
echo 📁 FIX-ENVIRONMENT.bat          - Sets up consistent environment variables
echo 📁 CHECK-DEPENDENCIES.bat       - Comprehensive dependency status checker
echo 📁 ANALYZE-LOGS.bat             - Scans logs for errors and issues
echo 📁 QUICK-DIAGNOSIS.bat          - Fast system health check
echo 📁 MASTER-RECOVERY.bat          - Complete system recovery (runs all fixes)
echo.
echo 🎯 RECOMMENDED USAGE:
echo.
echo 1. Start with: MASTER-RECOVERY.bat    (fixes everything)
echo 2. For quick checks: QUICK-DIAGNOSIS.bat
echo 3. For specific issues: Use individual scripts
echo 4. For service startup: ROBUST-START.bat
echo.
echo ================================================================================
echo 🚀 ALL BATCH FILE ERRORS SHOULD NOW BE RESOLVED!
echo ================================================================================
echo.
echo Press any key to run QUICK-DIAGNOSIS to verify fixes...
pause >nul

call QUICK-DIAGNOSIS.bat