@echo off
cls
color 0A
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║            🏛️  LEGAL AI CASE MANAGEMENT SYSTEM               ║
echo ║                    Master Control Panel v3.0                ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:main_menu
echo ┌─────────────────── MAIN MENU ────────────────────┐
echo │                                                   │
echo │  [1] 🚀 Quick Start (Optimized Setup)            │
echo │  [2] 🐳 Full Docker Setup (All Services)         │
echo │  [3] 🔧 Development Mode (Hot Reload)            │
echo │  [4] 📊 Service Status Dashboard                 │
echo │  [5] 🧹 System Maintenance                       │
echo │  [6] 🛡️  Security & Performance Tools           │
echo │  [7] 📈 Advanced Features Menu                   │
echo │  [8] 🆘 Troubleshooting & Repair                │
echo │  [9] 📚 Documentation & Help                     │
echo │  [0] ❌ Exit                                     │
echo │                                                   │
echo └───────────────────────────────────────────────────┘
echo.

set /p choice="👉 Select option [0-9]: "

if "%choice%"=="1" goto quick_start
if "%choice%"=="2" goto full_setup
if "%choice%"=="3" goto dev_mode
if "%choice%"=="4" goto status_dashboard
if "%choice%"=="5" goto maintenance
if "%choice%"=="6" goto security_tools
if "%choice%"=="7" goto advanced_features
if "%choice%"=="8" goto troubleshooting
if "%choice%"=="9" goto documentation
if "%choice%"=="0" goto exit

echo ❌ Invalid choice. Please try again.
timeout /t 2 >nul
goto main_menu

:quick_start
cls
echo ┌─────────────────── QUICK START ─────────────────────┐
echo │                                                      │
echo │  🚀 Starting optimized Legal AI system...           │
echo │                                                      │
echo └──────────────────────────────────────────────────────┘
echo.

echo 📋 Step 1: Checking prerequisites...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker not found! Please install Docker Desktop first.
    pause
    goto main_menu
)
echo ✅ Docker installed

echo.
echo 📋 Step 2: Starting optimized containers...
docker-compose -f docker-compose.optimized.yml up -d legal-postgres legal-redis legal-qdrant

echo.
echo 📋 Step 3: Waiting for databases to initialize...
timeout /t 10 >nul

echo.
echo 📋 Step 4: Starting AI services...
docker-compose -f docker-compose.optimized.yml up -d legal-ollama

echo.
echo 📋 Step 5: Starting frontend...
cd sveltekit-frontend
start cmd /k "npm run dev"
cd ..

echo.
echo ✅ Quick start complete!
echo 🌐 Access your legal AI system at: http://localhost:5173
echo 📊 Database studio: Run 'npm run db:studio' in another terminal
echo.
pause
goto main_menu

:full_setup
cls
echo ┌─────────────────── FULL SETUP ──────────────────────┐
echo │                                                      │
echo │  🐳 Complete Docker infrastructure setup            │
echo │                                                      │
echo └──────────────────────────────────────────────────────┘
echo.

echo 📋 Starting all services with optimized configuration...
docker-compose -f docker-compose.optimized.yml down -v
docker-compose -f docker-compose.optimized.yml build --no-cache
docker-compose -f docker-compose.optimized.yml up -d

echo.
echo 📋 Waiting for all services to be ready...
timeout /t 30 >nul

echo.
echo 📋 Running database migrations...
cd sveltekit-frontend
npm run db:migrate
cd ..

echo.
echo ✅ Full setup complete! All advanced features enabled:
echo   🏛️  Legal Case Management
echo   🤖 AI-Powered Evidence Analysis  
echo   👥 Real-time Collaboration
echo   📄 Smart Document Processing
echo   📊 Advanced Analytics
echo   🔒 Enterprise Security
echo.
pause
goto main_menu

:dev_mode
cls
echo ┌─────────────────── DEVELOPMENT MODE ────────────────┐
echo │                                                      │
echo │  🔧 Hot reload development environment               │
echo │                                                      │
echo └──────────────────────────────────────────────────────┘
echo.

echo 📋 Starting development environment...

echo Starting databases...
docker-compose -f docker-compose.optimized.yml up -d legal-postgres legal-redis legal-qdrant legal-ollama

echo.
echo Starting frontend in development mode...
cd sveltekit-frontend
start cmd /k "npm run dev"
cd ..

echo.
echo Starting collaboration server...
start cmd /k "cd collaboration-server && npm run dev"

echo.
echo ✅ Development environment started!
echo 🔥 Hot reload enabled for instant development
echo.
pause
goto main_menu

:status_dashboard
cls
echo ┌─────────────────── SERVICE STATUS ──────────────────┐
echo │                                                      │
echo │  📊 System Health Dashboard                         │
echo │                                                      │
echo └──────────────────────────────────────────────────────┘
echo.

echo ════════════════ DOCKER CONTAINERS ════════════════
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr legal-
echo.

echo ════════════════ SERVICE HEALTH CHECKS ═════════════
echo 🔍 PostgreSQL Database...
powershell -Command "try { Test-NetConnection localhost -Port 5432 -WarningAction SilentlyContinue | Select-Object -Property ComputerName,RemotePort,TcpTestSucceeded } catch { 'Connection failed' }"

echo.
echo 🤖 Ollama AI Service...
powershell -Command "try { Test-NetConnection localhost -Port 11434 -WarningAction SilentlyContinue | Select-Object -Property ComputerName,RemotePort,TcpTestSucceeded } catch { 'Connection failed' }"

echo.
echo 🔍 Qdrant Vector Database...
powershell -Command "try { Test-NetConnection localhost -Port 6333 -WarningAction SilentlyContinue | Select-Object -Property ComputerName,RemotePort,TcpTestSucceeded } catch { 'Connection failed' }"

echo.
echo 💾 Redis Cache...
powershell -Command "try { Test-NetConnection localhost -Port 6379 -WarningAction SilentlyContinue | Select-Object -Property ComputerName,RemotePort,TcpTestSucceeded } catch { 'Connection failed' }"

echo.
echo 🌐 Frontend Application...
powershell -Command "try { Test-NetConnection localhost -Port 5173 -WarningAction SilentlyContinue | Select-Object -Property ComputerName,RemotePort,TcpTestSucceeded } catch { 'Connection failed' }"

echo.
echo ════════════════ RESOURCE USAGE ════════════════
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | findstr legal-

echo.
pause
goto main_menu

:maintenance
cls
echo ┌─────────────────── MAINTENANCE ─────────────────────┐
echo │                                                      │
echo │  🧹 System Maintenance Tools                        │
echo │                                                      │
echo └──────────────────────────────────────────────────────┘
echo.

echo   [1] 🗑️  Clean Docker containers and images
echo   [2] 🔄 Reset all databases
echo   [3] 📦 Update all services
echo   [4] 🧼 Clear application cache
echo   [5] 📊 Optimize database performance
echo   [6] ↩️  Back to main menu
echo.

set /p maint_choice="Select maintenance option [1-6]: "

if "%maint_choice%"=="1" goto clean_docker
if "%maint_choice%"=="2" goto reset_databases
if "%maint_choice%"=="3" goto update_services
if "%maint_choice%"=="4" goto clear_cache
if "%maint_choice%"=="5" goto optimize_db
if "%maint_choice%"=="6" goto main_menu

:clean_docker
echo 🗑️  Cleaning Docker resources...
docker-compose -f docker-compose.optimized.yml down -v
docker system prune -f
docker volume prune -f
echo ✅ Docker cleanup complete!
pause
goto maintenance

:reset_databases
echo ⚠️  WARNING: This will delete all data!
set /p confirm="Are you sure? (y/N): "
if /i "%confirm%"=="y" (
    echo 🔄 Resetting databases...
    docker-compose -f docker-compose.optimized.yml down -v
    docker volume rm deeds-web-app_postgres_data deeds-web-app_qdrant_data deeds-web-app_redis_data 2>nul
    echo ✅ Databases reset!
)
pause
goto maintenance

:advanced_features
cls
echo ┌─────────────────── ADVANCED FEATURES ───────────────┐
echo │                                                      │
echo │  📈 Advanced Legal AI Capabilities                  │
echo │                                                      │
echo └──────────────────────────────────────────────────────┘
echo.

echo   [1] 🤖 AI Model Management
echo   [2] 👥 Collaboration Tools
echo   [3] 📄 Document Processing
echo   [4] 📊 Analytics Dashboard
echo   [5] 🔍 Advanced Search
echo   [6] 🛡️  Security Center
echo   [7] ↩️  Back to main menu
echo.

set /p adv_choice="Select feature [1-7]: "

if "%adv_choice%"=="1" goto ai_models
if "%adv_choice%"=="2" goto collaboration
if "%adv_choice%"=="3" goto document_processing
if "%adv_choice%"=="4" goto analytics
if "%adv_choice%"=="5" goto advanced_search
if "%adv_choice%"=="6" goto security_center
if "%adv_choice%"=="7" goto main_menu

:ai_models
echo 🤖 Starting AI Model Management...
start http://localhost:5173/admin/ai-models
echo ✅ Opened in browser
pause
goto advanced_features

:collaboration
echo 👥 Starting Real-time Collaboration...
docker-compose -f docker-compose.optimized.yml up -d legal-collaboration
start http://localhost:5173/collaboration
echo ✅ Collaboration server started
pause
goto advanced_features

:troubleshooting
cls
echo ┌─────────────────── TROUBLESHOOTING ─────────────────┐
echo │                                                      │
echo │  🆘 System Diagnosis and Repair                     │
echo │                                                      │
echo └──────────────────────────────────────────────────────┘
echo.

echo   [1] 🔍 Run full system diagnosis
echo   [2] 🔧 Fix common issues automatically
echo   [3] 📋 Generate support report
echo   [4] 🔄 Restart all services
echo   [5] 🆘 Emergency reset
echo   [6] ↩️  Back to main menu
echo.

set /p trouble_choice="Select option [1-6]: "

if "%trouble_choice%"=="1" goto diagnosis
if "%trouble_choice%"=="2" goto auto_fix
if "%trouble_choice%"=="3" goto support_report
if "%trouble_choice%"=="4" goto restart_services
if "%trouble_choice%"=="5" goto emergency_reset
if "%trouble_choice%"=="6" goto main_menu

:diagnosis
echo 🔍 Running comprehensive system diagnosis...
echo ═══════════════════════════════════════════════════
echo.

echo 📊 Checking Docker status...
docker --version
docker ps

echo.
echo 🌐 Checking network connectivity...
ping -n 1 localhost

echo.
echo 💾 Checking disk space...
dir C:\ | findstr bytes

echo.
echo 📋 Checking application logs...
if exist sveltekit-frontend\logs\*.log (
    echo Found application logs
) else (
    echo No application logs found
)

echo.
echo ✅ Diagnosis complete! Check output above for issues.
pause
goto troubleshooting

:exit
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║  Thank you for using the Legal AI Case Management System!   ║
echo ║                                                              ║
echo ║  🏛️  Justice through technology                             ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
timeout /t 3 >nul
exit /b