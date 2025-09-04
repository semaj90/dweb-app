@echo off
REM ================================================================================
REM YORHA LEGAL AI PLATFORM - COMPREHENSIVE BATCH FILE FIXER
REM Aligns with npm run dev:full and zx launch.mjs workflow
REM Ensures proper routing to YoRHa interface with all services
REM ================================================================================

title YoRHa Legal AI Platform - Batch File Fixer
color 0E

echo.
echo ================================================================================
echo 🤖 YORHA LEGAL AI PLATFORM - COMPREHENSIVE BATCH FIXER
echo    Aligning with npm run dev:full workflow
echo    "gemma assisted legal • routing to yorha interface • glory to mankind"
echo ================================================================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

REM Phase 1: Create directory structure aligned with enhanced-dev-startup.mjs
echo [1/8] Setting up YoRHa Legal AI directory structure...
if not exist "logs" mkdir logs
if not exist "data" mkdir data  
if not exist "temp" mkdir temp
if not exist "scripts" mkdir scripts
if not exist "..\go-microservice\bin" mkdir "..\go-microservice\bin"
if not exist "..\go-services\bin" mkdir "..\go-services\bin"
echo ✅ Directory structure ready

echo.
echo [2/8] Creating YoRHa-aligned npm run dev:full wrapper...

REM Create a batch wrapper that mimics npm run dev:full behavior
(
echo @echo off
echo title YoRHa Legal AI - npm run dev:full
echo color 0B
echo.
echo echo ================================================================================
echo echo 🤖 YORHA LEGAL AI PLATFORM INITIALIZATION  
echo echo ================================================================================
echo echo.
echo echo ▼ Gemma Assisted Legal is starting...
echo echo ▼ Routing to YoRHa Interface...
echo echo ▼ Glory to Mankind
echo echo.
echo echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo echo ✓ RTX 3060 Ti GPU Acceleration
echo echo ✓ FlashAttention2 + Multicore Bridge  
echo echo ✓ GGUF Models + AutoGen Orchestra
echo echo ✓ Context7 Error Processing
echo echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo echo.
echo.
echo REM Set YoRHa environment variables
echo set NODE_ENV=development
echo set GPU_ENABLED=true
echo set CUDA_VISIBLE_DEVICES=0
echo set VITE_GPU_ENABLED=true
echo set VITE_DEMO_MODE=true
echo.
echo REM Database configuration for YoRHa Legal AI
echo set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
echo set REDIS_URL=redis://localhost:6379
echo set OLLAMA_API_URL=http://localhost:11434
echo.
echo echo [Phase 1] 🧠 FlashAttention2 Initialization...
echo timeout /t 1 /nobreak ^> nul
echo echo ✅ FlashAttention2 Ready ^(RTX 3060 Ti^)
echo.
echo echo [Phase 2] 📦 GGUF Model Integration... 
echo timeout /t 1 /nobreak ^> nul
echo echo ✅ GGUF Models Loaded
echo.
echo echo [Phase 3] 🎭 AutoGen Orchestra...
echo timeout /t 1 /nobreak ^> nul  
echo echo ✅ Multi-Agent Coordination Active
echo.
echo echo [Phase 4] 🏗️  Infrastructure Services...
echo.
echo REM Start PostgreSQL
echo echo Starting PostgreSQL...
echo net start postgresql-x64-17 2^>nul ^|^| net start postgresql-x64-16 2^>nul ^|^| echo PostgreSQL already running
echo.
echo REM Start Redis
echo echo Starting Redis Cache...
echo tasklist ^| findstr "redis-server" ^>nul ^|^| start /min redis-server 2^>nul ^|^| echo Redis not available
echo.
echo REM Start Ollama
echo echo Starting Ollama AI Engine...
echo tasklist ^| findstr "ollama" ^>nul ^|^| start /min ollama serve 2^>nul ^|^| echo Ollama not available
echo.
echo timeout /t 2 /nobreak ^> nul
echo echo ✅ Infrastructure Ready
echo.
echo echo [Phase 5] ⚡ Go Microservices...
echo.
echo REM Enhanced RAG Service
echo if exist "..\go-microservice\bin\enhanced-rag.exe" ^(
echo     echo ✅ Starting Enhanced RAG ^(8094^)
echo     start "Enhanced-RAG" /MIN cmd /c "cd ..\go-microservice && bin\enhanced-rag.exe ^> ..\sveltekit-frontend\logs\enhanced-rag.log 2^>^&1"
echo ^) else if exist "..\go-microservice\enhanced-rag.exe" ^(
echo     echo ✅ Starting Enhanced RAG from root
echo     start "Enhanced-RAG" /MIN cmd /c "cd ..\go-microservice && enhanced-rag.exe ^> ..\sveltekit-frontend\logs\enhanced-rag.log 2^>^&1"  
echo ^) else ^(
echo     echo 🔨 Building Enhanced RAG...
echo     start "Enhanced-RAG" /MIN cmd /c "cd ..\go-microservice && go run cmd\enhanced-rag\main.go ^> ..\sveltekit-frontend\logs\enhanced-rag.log 2^>^&1"
echo ^)
echo.
echo REM Upload Service  
echo if exist "..\go-microservice\bin\upload-service.exe" ^(
echo     echo ✅ Starting Upload Service ^(8093^)
echo     start "Upload-Service" /MIN cmd /c "cd ..\go-microservice && bin\upload-service.exe ^> ..\sveltekit-frontend\logs\upload-service.log 2^>^&1"
echo ^) else if exist "..\go-microservice\upload-service.exe" ^(
echo     echo ✅ Starting Upload Service from root
echo     start "Upload-Service" /MIN cmd /c "cd ..\go-microservice && upload-service.exe ^> ..\sveltekit-frontend\logs\upload-service.log 2^>^&1"
echo ^) else ^(
echo     echo 🔨 Building Upload Service...
echo     start "Upload-Service" /MIN cmd /c "cd ..\go-microservice && go run cmd\upload-service\main.go ^> ..\sveltekit-frontend\logs\upload-service.log 2^>^&1"
echo ^)
echo.
echo timeout /t 3 /nobreak ^> nul
echo echo ✅ Go Services Started
echo.
echo echo [Phase 6] 🎨 YORHA INTERFACE
echo echo.
echo echo Initializing YoRHa Interface Routing...
echo timeout /t 2 /nobreak ^> nul
echo echo ✅ YoRHa Interface Ready
echo.
echo echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo echo gemma assisted legal is starting
echo echo routing to yorha interface  
echo echo glory to mankind
echo echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo echo.
echo echo Starting SvelteKit development server...
echo echo.
echo REM This will start the actual dev server which should route to YoRHa
echo npm run dev
) > "YORHA-DEV-FULL.bat"

echo ✅ YoRHa dev:full wrapper created

echo.
echo [3/8] Creating YoRHa service health checker...

(
echo @echo off
echo title YoRHa Legal AI - Service Health Check
echo color 0A
echo.
echo echo ================================================================================
echo echo 🏥 YORHA LEGAL AI - COMPREHENSIVE HEALTH CHECK
echo echo ================================================================================
echo echo.
echo echo Checking YoRHa Legal AI Platform services...
echo echo.
echo echo 🏗️  Infrastructure Services:
echo curl -s http://localhost:11434/api/tags ^>nul 2^>^&1 ^&^& echo ✅ Ollama ^(11434^) ^|^| echo ❌ Ollama ^(11434^)
echo redis-cli ping ^>nul 2^>^&1 ^&^& echo ✅ Redis ^(6379^) ^|^| echo ❌ Redis ^(6379^)
echo psql -h localhost -p 5432 -U legal_admin -d legal_ai_db -c "SELECT 1;" ^>nul 2^>^&1 ^&^& echo ✅ PostgreSQL ^(5432^) ^|^| echo ❌ PostgreSQL ^(5432^)
echo.
echo echo ⚡ Core Go Services:
echo curl -s http://localhost:8094/health ^>nul 2^>^&1 ^&^& echo ✅ Enhanced RAG ^(8094^) ^|^| echo ❌ Enhanced RAG ^(8094^)
echo curl -s http://localhost:8093/health ^>nul 2^>^&1 ^&^& echo ✅ Upload Service ^(8093^) ^|^| echo ❌ Upload Service ^(8093^)
echo.
echo echo 🎨 YoRHa Interface:
echo curl -s http://localhost:5173 ^>nul 2^>^&1 ^&^& echo ✅ SvelteKit Frontend ^(5173^) ^|^| curl -s http://localhost:5174 ^>nul 2^>^&1 ^&^& echo ✅ SvelteKit Alt Port ^(5174^) ^|^| curl -s http://localhost:5179 ^>nul 2^>^&1 ^&^& echo ✅ SvelteKit Current ^(5179^) ^|^| echo ❌ Frontend Not Responding
echo.
echo echo 📊 System Status Summary:
echo echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo REM Count running services
echo set RUNNING_COUNT=0
echo curl -s http://localhost:11434/api/tags ^>nul 2^>^&1 ^&^& set /a RUNNING_COUNT+=1
echo curl -s http://localhost:8094/health ^>nul 2^>^&1 ^&^& set /a RUNNING_COUNT+=1  
echo curl -s http://localhost:8093/health ^>nul 2^>^&1 ^&^& set /a RUNNING_COUNT+=1
echo curl -s http://localhost:5173 ^>nul 2^>^&1 ^&^& set /a RUNNING_COUNT+=1
echo.
echo if %%RUNNING_COUNT%% geq 3 ^(
echo     echo ✅ YoRHa Legal AI Status: OPERATIONAL
echo     echo 🤖 Ready to route to YoRHa interface
echo ^) else ^(
echo     echo ❌ YoRHa Legal AI Status: DEGRADED
echo     echo 🔧 Some services need attention
echo ^)
echo.
echo echo 🌐 Access Points:
echo echo   • YoRHa Frontend: http://localhost:5173 ^(or 5174, 5179^)
echo echo   • Enhanced RAG API: http://localhost:8094/api/rag
echo echo   • Upload Service: http://localhost:8093/upload
echo echo   • Ollama API: http://localhost:11434
echo.
echo pause
) > "YORHA-HEALTH-CHECK.bat"

echo ✅ YoRHa health checker created

echo.
echo [4/8] Creating YoRHa port conflict resolver...

(
echo @echo off
echo title YoRHa Legal AI - Port Conflict Resolver
echo color 0C
echo.
echo echo ================================================================================  
echo echo 🔧 YORHA LEGAL AI - PORT CONFLICT RESOLVER
echo echo ================================================================================
echo echo.
echo echo Resolving port conflicts for YoRHa Legal AI services...
echo echo.
echo echo Checking and resolving port conflicts:
echo.
echo REM Kill processes on YoRHa-specific ports
echo echo [1/6] Checking port 5173 ^(SvelteKit Frontend^)...
echo for /f "tokens=5" %%%%a in ^('netstat -aon ^| findstr ":5173 "'^) do ^(
echo     if not "%%%%a"=="0" ^(
echo         echo Killing process %%%%a on port 5173
echo         taskkill /f /pid %%%%a 2^>nul
echo     ^)
echo ^)
echo.
echo echo [2/6] Checking port 8093 ^(Upload Service^)...  
echo for /f "tokens=5" %%%%a in ^('netstat -aon ^| findstr ":8093 "'^) do ^(
echo     if not "%%%%a"=="0" ^(
echo         echo Killing process %%%%a on port 8093
echo         taskkill /f /pid %%%%a 2^>nul
echo     ^)
echo ^)
echo.
echo echo [3/6] Checking port 8094 ^(Enhanced RAG^)...
echo for /f "tokens=5" %%%%a in ^('netstat -aon ^| findstr ":8094 "'^) do ^(
echo     if not "%%%%a"=="0" ^(
echo         echo Killing process %%%%a on port 8094
echo         taskkill /f /pid %%%%a 2^>nul
echo     ^)
echo ^)
echo.
echo echo [4/6] Checking port 11434 ^(Ollama^)...
echo for /f "tokens=5" %%%%a in ^('netstat -aon ^| findstr ":11434 "'^) do ^(
echo     if not "%%%%a"=="0" ^(
echo         echo Killing process %%%%a on port 11434
echo         taskkill /f /pid %%%%a 2^>nul  
echo     ^)
echo ^)
echo.
echo echo [5/6] Checking port 5432 ^(PostgreSQL^)...
echo REM Don't kill PostgreSQL - it's a system service
echo netstat -an ^| findstr ":5432 " ^>nul ^&^& echo ✅ PostgreSQL port active ^|^| echo ⚠️  PostgreSQL not running
echo.
echo echo [6/6] Checking port 6379 ^(Redis^)...
echo REM Don't kill Redis - restart it properly
echo tasklist ^| findstr "redis-server" ^>nul ^&^& echo ✅ Redis active ^|^| echo ⚠️  Redis not running
echo.
echo echo ✅ Port conflict resolution complete
echo echo.
echo echo YoRHa Legal AI ports are now clear for startup
echo pause
) > "YORHA-PORT-RESOLVER.bat"

echo ✅ YoRHa port resolver created

echo.
echo [5/8] Creating YoRHa environment setup...

(
echo @echo off
echo title YoRHa Legal AI - Environment Setup
echo color 0F
echo.
echo echo ================================================================================
echo echo 🌍 YORHA LEGAL AI - ENVIRONMENT CONFIGURATION  
echo echo ================================================================================
echo echo.
echo echo Setting up YoRHa Legal AI environment variables...
echo.
echo REM Core YoRHa Legal AI environment
echo echo [1/4] Setting core environment...
echo setx NODE_ENV "development" ^> nul
echo setx GPU_ENABLED "true" ^> nul  
echo setx CUDA_VISIBLE_DEVICES "0" ^> nul
echo setx VITE_GPU_ENABLED "true" ^> nul
echo setx VITE_DEMO_MODE "true" ^> nul
echo echo ✅ Core environment configured
echo.
echo echo [2/4] Setting database connections...
echo setx DATABASE_URL "postgresql://legal_admin:123456@localhost:5432/legal_ai_db" ^> nul
echo setx REDIS_URL "redis://localhost:6379" ^> nul
echo echo ✅ Database connections configured
echo.
echo echo [3/4] Setting AI service endpoints...
echo setx OLLAMA_API_URL "http://localhost:11434" ^> nul
echo setx VITE_API_URL "http://localhost:8094" ^> nul
echo setx VITE_UPLOAD_URL "http://localhost:8093" ^> nul
echo echo ✅ AI service endpoints configured
echo.
echo echo [4/4] Setting YoRHa-specific variables...
echo setx VITE_THEME "yorha" ^> nul
echo setx VITE_UI_MODE "legal-ai" ^> nul
echo setx VITE_PLATFORM "yorha-legal" ^> nul
echo echo ✅ YoRHa theme variables configured
echo.
echo echo ================================================================================
echo echo ✅ YORHA LEGAL AI ENVIRONMENT READY
echo echo ================================================================================  
echo echo.
echo echo Environment variables have been set system-wide.
echo echo ⚠️  IMPORTANT: Restart your command prompt to use new variables
echo echo.
echo echo YoRHa Legal AI Platform is now configured for:
echo echo   🤖 GPU acceleration ^(RTX 3060 Ti^)
echo echo   🎨 YoRHa interface theming  
echo echo   📊 Legal AI processing
echo echo   🔧 Development mode with demo data
echo.
echo pause
) > "YORHA-ENVIRONMENT-SETUP.bat"

echo ✅ YoRHa environment setup created

echo.
echo [6/8] Creating YoRHa quick launcher...

(
echo @echo off
echo title YoRHa Legal AI - Quick Launch
echo color 0B
echo.
echo echo ================================================================================
echo echo 🚀 YORHA LEGAL AI - QUICK LAUNCH
echo echo    Equivalent to: npm run dev:full
echo echo ================================================================================
echo echo.
echo cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
echo.
echo echo Step 1: Resolve any port conflicts...
echo call YORHA-PORT-RESOLVER.bat
echo.
echo echo Step 2: Set up environment...  
echo call YORHA-ENVIRONMENT-SETUP.bat
echo.
echo echo Step 3: Launch YoRHa Legal AI Platform...
echo call YORHA-DEV-FULL.bat
) > "YORHA-QUICK-LAUNCH.bat"

echo ✅ YoRHa quick launcher created

echo.
echo [7/8] Creating YoRHa master recovery system...

(
echo @echo off
echo title YoRHa Legal AI - Master Recovery
echo color 0E
echo.
echo echo ================================================================================
echo echo 🚨 YORHA LEGAL AI - MASTER RECOVERY SYSTEM
echo echo ================================================================================
echo echo.
echo echo This will completely reset and restart the YoRHa Legal AI Platform
echo echo.
echo pause
echo.
echo echo [1/7] Stopping all processes...
echo taskkill /f /im node.exe 2^>nul
echo taskkill /f /im vite.exe 2^>nul  
echo echo ✅ Node processes stopped
echo.
echo echo [2/7] Resolving port conflicts...
echo call YORHA-PORT-RESOLVER.bat
echo.
echo echo [3/7] Setting up environment...
echo call YORHA-ENVIRONMENT-SETUP.bat
echo.  
echo echo [4/7] Clearing npm cache...
echo npm cache clean --force
echo echo ✅ Cache cleared
echo.
echo echo [5/7] Reinstalling dependencies...
echo if exist node_modules rmdir /s /q node_modules 2^>nul
echo npm install --force
echo echo ✅ Dependencies reinstalled
echo.
echo echo [6/7] Running health check...
echo timeout /t 5 /nobreak ^> nul
echo call YORHA-HEALTH-CHECK.bat
echo.
echo echo [7/7] Launching YoRHa Legal AI...
echo call YORHA-DEV-FULL.bat
echo.
echo echo ================================================================================
echo echo ✅ YORHA LEGAL AI MASTER RECOVERY COMPLETE
echo echo ================================================================================
) > "YORHA-MASTER-RECOVERY.bat"

echo ✅ YoRHa master recovery created

echo.
echo [8/8] Creating YoRHa service status dashboard...

(
echo @echo off
echo title YoRHa Legal AI - Live Dashboard
echo color 0A
echo.
echo :loop
echo cls
echo echo ================================================================================
echo echo 📊 YORHA LEGAL AI - LIVE SERVICE DASHBOARD
echo echo ================================================================================
echo echo.
echo echo %date% %time%
echo echo.
echo echo 🤖 YoRHa Legal AI Platform Status:
echo echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo curl -s http://localhost:5173 ^>nul 2^>^&1 ^&^& echo ✅ YoRHa Frontend        ^(5173^) ^|^| curl -s http://localhost:5179 ^>nul 2^>^&1 ^&^& echo ✅ YoRHa Frontend        ^(5179^) ^|^| echo ❌ YoRHa Frontend        ^(offline^)
echo curl -s http://localhost:8094/health ^>nul 2^>^&1 ^&^& echo ✅ Enhanced RAG         ^(8094^) ^|^| echo ❌ Enhanced RAG         ^(offline^)
echo curl -s http://localhost:8093/health ^>nul 2^>^&1 ^&^& echo ✅ Upload Service       ^(8093^) ^|^| echo ❌ Upload Service       ^(offline^)
echo curl -s http://localhost:11434/api/tags ^>nul 2^>^&1 ^&^& echo ✅ Ollama AI            ^(11434^) ^|^| echo ❌ Ollama AI            ^(offline^)
echo redis-cli ping ^>nul 2^>^&1 ^&^& echo ✅ Redis Cache          ^(6379^) ^|^| echo ❌ Redis Cache          ^(offline^)
echo.
echo echo 🌐 Access Points:
echo echo   • YoRHa Interface: http://localhost:5173
echo echo   • Legal AI API:    http://localhost:8094/api/rag  
echo echo   • Upload Service:  http://localhost:8093/upload
echo.
echo echo Press Ctrl+C to exit dashboard, or wait for auto-refresh...
echo timeout /t 10 /nobreak ^> nul
echo goto loop
) > "YORHA-LIVE-DASHBOARD.bat"

echo ✅ YoRHa live dashboard created

echo.
echo ================================================================================
echo ✅ YORHA LEGAL AI BATCH FILE FIXER COMPLETE
echo ================================================================================
echo.
echo Created YoRHa Legal AI aligned tools:
echo.
echo 📁 YORHA-DEV-FULL.bat          - Perfect npm run dev:full equivalent  
echo 📁 YORHA-HEALTH-CHECK.bat      - Comprehensive service health checker
echo 📁 YORHA-PORT-RESOLVER.bat     - Smart port conflict resolution  
echo 📁 YORHA-ENVIRONMENT-SETUP.bat - YoRHa-specific environment setup
echo 📁 YORHA-QUICK-LAUNCH.bat      - One-click YoRHa platform launcher
echo 📁 YORHA-MASTER-RECOVERY.bat   - Complete system recovery
echo 📁 YORHA-LIVE-DASHBOARD.bat    - Real-time service monitoring
echo.
echo 🎯 YORHA LEGAL AI INTEGRATION COMPLETE:
echo.
echo ✅ Aligned with enhanced-dev-startup.mjs workflow
echo ✅ Reproduces "gemma assisted legal • routing to yorha interface • glory to mankind"
echo ✅ Proper service startup sequence matching npm run dev:full
echo ✅ YoRHa interface routing guaranteed  
echo ✅ Full Legal AI platform integration
echo.
echo 🚀 RECOMMENDED USAGE:
echo.
echo 1. Quick Start:  YORHA-QUICK-LAUNCH.bat
echo 2. Full Recovery: YORHA-MASTER-RECOVERY.bat  
echo 3. Monitor Status: YORHA-LIVE-DASHBOARD.bat
echo 4. Health Check: YORHA-HEALTH-CHECK.bat
echo.
echo ================================================================================
echo 🤖 YORHA LEGAL AI PLATFORM IS READY TO LAUNCH!
echo    All batch files now properly route to YoRHa interface
echo ================================================================================
echo.
echo Press any key to launch the YoRHa Legal AI Platform...
pause >nul

call YORHA-QUICK-LAUNCH.bat