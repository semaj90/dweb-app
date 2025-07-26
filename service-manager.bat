@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo   Legal AI Assistant - Service Manager
echo =====================================================
echo.

:: Color codes
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

:menu
echo %BLUE%📋 Service Management Menu:%NC%
echo.
echo 1. Start All Services (Optimized)
echo 2. Stop All Services
echo 3. Restart Services
echo 4. View Service Status
echo 5. View Logs
echo 6. Health Check
echo 7. Monitor Resources
echo 8. AI Model Management
echo 9. Database Management
echo 0. Exit
echo.
set /p choice="%BLUE%Select option (0-9): %NC%"

if "%choice%"=="1" goto :start_services
if "%choice%"=="2" goto :stop_services
if "%choice%"=="3" goto :restart_services
if "%choice%"=="4" goto :service_status
if "%choice%"=="5" goto :view_logs
if "%choice%"=="6" goto :health_check
if "%choice%"=="7" goto :monitor_resources
if "%choice%"=="8" goto :model_management
if "%choice%"=="9" goto :database_management
if "%choice%"=="0" goto :exit
goto :menu

:start_services
echo.
echo %BLUE%🚀 Starting all services with optimized configuration...%NC%
docker-compose -f docker-compose-optimized.yml up -d
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to start services%NC%
) else (
    echo %GREEN%✅ Services started successfully%NC%
    timeout /t 3 >nul
    goto :service_status
)
goto :menu

:stop_services
echo.
echo %BLUE%🛑 Stopping all services...%NC%
docker-compose -f docker-compose-optimized.yml down
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to stop services%NC%
) else (
    echo %GREEN%✅ Services stopped successfully%NC%
)
goto :menu

:restart_services
echo.
echo %BLUE%🔄 Restarting services...%NC%
docker-compose -f docker-compose-optimized.yml restart
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to restart services%NC%
) else (
    echo %GREEN%✅ Services restarted successfully%NC%
    timeout /t 3 >nul
    goto :service_status
)
goto :menu

:service_status
echo.
echo %BLUE%📊 Current Service Status:%NC%
echo.
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr -E "(deeds-|NAMES)"
echo.

:: Check service health
echo %BLUE%🏥 Health Status:%NC%
docker exec deeds-postgres pg_isready -U legal_admin -d prosecutor_db >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ PostgreSQL: Unhealthy%NC%
) else (
    echo %GREEN%✅ PostgreSQL: Healthy%NC%
)

curl -f http://localhost:11434/api/version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Ollama: Unhealthy%NC%
) else (
    echo %GREEN%✅ Ollama: Healthy%NC%
)

curl -f http://localhost:6333/health >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Qdrant: Unhealthy%NC%
) else (
    echo %GREEN%✅ Qdrant: Healthy%NC%
)

docker exec deeds-redis redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Redis: Unhealthy%NC%
) else (
    echo %GREEN%✅ Redis: Healthy%NC%
)

goto :menu

:view_logs
echo.
echo %BLUE%📋 Service Logs Menu:%NC%
echo.
echo 1. All Services
echo 2. Ollama Only
echo 3. PostgreSQL Only
echo 4. Qdrant Only
echo 5. Redis Only
echo 6. Back to Main Menu
echo.
set /p log_choice="%BLUE%Select log option (1-6): %NC%"

if "%log_choice%"=="1" (
    echo %BLUE%📜 Showing all service logs (Ctrl+C to exit):%NC%
    docker-compose -f docker-compose-optimized.yml logs -f
)
if "%log_choice%"=="2" (
    echo %BLUE%📜 Showing Ollama logs (Ctrl+C to exit):%NC%
    docker-compose -f docker-compose-optimized.yml logs -f ollama
)
if "%log_choice%"=="3" (
    echo %BLUE%📜 Showing PostgreSQL logs (Ctrl+C to exit):%NC%
    docker-compose -f docker-compose-optimized.yml logs -f postgres
)
if "%log_choice%"=="4" (
    echo %BLUE%📜 Showing Qdrant logs (Ctrl+C to exit):%NC%
    docker-compose -f docker-compose-optimized.yml logs -f qdrant
)
if "%log_choice%"=="5" (
    echo %BLUE%📜 Showing Redis logs (Ctrl+C to exit):%NC%
    docker-compose -f docker-compose-optimized.yml logs -f redis
)
if "%log_choice%"=="6" goto :menu
goto :menu

:health_check
echo.
echo %BLUE%🏥 Running comprehensive health check...%NC%
call check-setup.bat
goto :menu

:monitor_resources
echo.
echo %BLUE%📊 Resource Monitoring (Ctrl+C to exit):%NC%
echo.
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"
goto :menu

:model_management
echo.
echo %BLUE%🤖 AI Model Management:%NC%
echo.
echo 1. List Available Models
echo 2. Pull Gemma2 9B Model
echo 3. Pull Custom Model
echo 4. Remove Model
echo 5. Test Model
echo 6. Back to Main Menu
echo.
set /p model_choice="%BLUE%Select model option (1-6): %NC%"

if "%model_choice%"=="1" (
    echo %BLUE%📋 Available Models:%NC%
    docker exec deeds-ollama-gpu ollama list
    pause
)
if "%model_choice%"=="2" (
    echo %BLUE%📥 Pulling Gemma3 Legal model...%NC%
    echo %BLUE%This is the specialized legal AI model for document analysis%NC%
    docker exec deeds-ollama-gpu ollama pull gemma3-legal
    if %errorlevel% neq 0 (
        echo %YELLOW%⚠️  Gemma3-legal not available, pulling Gemma2 9B as fallback...%NC%
        docker exec deeds-ollama-gpu ollama pull gemma2:9b
        if %errorlevel% neq 0 (
            echo %RED%❌ Failed to pull any model%NC%
        ) else (
            echo %GREEN%✅ Gemma2 9B model pulled successfully (fallback)%NC%
        )
    ) else (
        echo %GREEN%✅ Gemma3 Legal model pulled successfully%NC%
    )
    pause
)
if "%model_choice%"=="3" (
    set /p model_name="%BLUE%Enter model name (e.g., llama2:7b): %NC%"
    echo %BLUE%📥 Pulling !model_name!...%NC%
    docker exec deeds-ollama-gpu ollama pull !model_name!
    if %errorlevel% neq 0 (
        echo %RED%❌ Failed to pull model%NC%
    ) else (
        echo %GREEN%✅ Model pulled successfully%NC%
    )
    pause
)
if "%model_choice%"=="4" (
    echo %BLUE%📋 Available Models:%NC%
    docker exec deeds-ollama-gpu ollama list
    echo.
    set /p remove_model="%BLUE%Enter model name to remove: %NC%"
    docker exec deeds-ollama-gpu ollama rm !remove_model!
    if %errorlevel% neq 0 (
        echo %RED%❌ Failed to remove model%NC%
    ) else (
        echo %GREEN%✅ Model removed successfully%NC%
    )
    pause
)
if "%model_choice%"=="5" (
    echo %BLUE%🧪 Testing Legal AI model...%NC%
    echo %BLUE%Testing with legal document analysis prompt...%NC%
    curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\":\"gemma3-legal\",\"prompt\":\"Analyze the following legal text for potential compliance issues: This contract contains standard liability clauses.\",\"stream\":false}"
    if %errorlevel% neq 0 (
        echo %YELLOW%⚠️  Testing with fallback model...%NC%
        curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\":\"gemma2:9b\",\"prompt\":\"Analyze the following legal text for potential compliance issues: This contract contains standard liability clauses.\",\"stream\":false}"
    )
    echo.
    pause
)
if "%model_choice%"=="6" goto :menu
goto :model_management

:database_management
echo.
echo %BLUE%🗄️  Database Management:%NC%
echo.
echo 1. Check Database Status
echo 2. View Database Tables
echo 3. Run SQL Query
echo 4. Reset Database
echo 5. Backup Database
echo 6. Import Sample Data
echo 7. Back to Main Menu
echo.
set /p db_choice="%BLUE%Select database option (1-7): %NC%"

if "%db_choice%"=="1" (
    echo %BLUE%📊 Database Status:%NC%
    docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "\l"
    echo.
    docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "\dt"
    pause
)
if "%db_choice%"=="2" (
    echo %BLUE%📋 Database Tables:%NC%
    docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "\dt+"
    pause
)
if "%db_choice%"=="3" (
    echo %BLUE%💻 SQL Query Interface:%NC%
    echo Enter your SQL query (press Enter twice to execute):
    set /p sql_query="%BLUE%SQL> %NC%"
    docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "!sql_query!"
    pause
)
if "%db_choice%"=="4" (
    echo %YELLOW%⚠️  This will delete all data! Are you sure? (y/N)%NC%
    set /p confirm=""
    if /i "!confirm!"=="y" (
        echo %BLUE%🔄 Resetting database...%NC%
        docker exec deeds-postgres psql -U legal_admin -d postgres -c "DROP DATABASE IF EXISTS prosecutor_db;"
        docker exec deeds-postgres psql -U legal_admin -d postgres -c "CREATE DATABASE prosecutor_db;"
        docker exec -i deeds-postgres psql -U legal_admin -d prosecutor_db < database\migrations\001_init.sql
        echo %GREEN%✅ Database reset completed%NC%
    ) else (
        echo %YELLOW%Operation cancelled%NC%
    )
    pause
)
if "%db_choice%"=="5" (
    echo %BLUE%💾 Creating database backup...%NC%
    docker exec deeds-postgres pg_dump -U legal_admin prosecutor_db > "data\backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%.sql"
    echo %GREEN%✅ Backup created in data directory%NC%
    pause
)
if "%db_choice%"=="6" (
    echo %BLUE%📥 Importing sample data...%NC%
    docker exec -i deeds-postgres psql -U legal_admin -d prosecutor_db < database\migrations\001_init.sql
    echo %GREEN%✅ Sample data imported%NC%
    pause
)
if "%db_choice%"=="7" goto :menu
goto :database_management

:exit
echo.
echo %BLUE%👋 Thank you for using Legal AI Assistant Service Manager!%NC%
echo.
echo %YELLOW%💡 Quick Commands:%NC%
echo %YELLOW%• Start: docker-compose -f docker-compose-optimized.yml up -d%NC%
echo %YELLOW%• Stop:  docker-compose -f docker-compose-optimized.yml down%NC%
echo %YELLOW%• Logs:  docker-compose -f docker-compose-optimized.yml logs -f%NC%
echo %YELLOW%• NPM:   npm run dev (if Node.js is installed)%NC%
echo.
pause
exit /b 0
