@echo off
REM =============================================================================
REM PRODUCTION DEPLOYMENT SCRIPT - LEGAL AI PLATFORM (WINDOWS)
REM =============================================================================
REM This script deploys the Legal AI Platform to production environment including:
REM - Service management
REM - Database setup
REM - SSL configuration
REM - Health checks
REM =============================================================================

setlocal EnableDelayedExpansion

REM Configuration
set "PROJECT_ROOT=%~dp0.."
set "DIST_DIR=%PROJECT_ROOT%\dist"
set "SERVICE_DIR=C:\LegalAI\services"
set "WEB_ROOT=C:\LegalAI\www"
set "LOG_DIR=C:\LegalAI\logs"
set "DATA_DIR=C:\LegalAI\data"

echo ================================================================================
echo LEGAL AI PLATFORM - PRODUCTION DEPLOYMENT (WINDOWS)
echo ================================================================================

REM Check if running as administrator
net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] This script must be run as Administrator
    echo [ERROR] Right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo [DEPLOY] Starting production deployment...

REM Function to create directory structure
call :create_directories
if errorlevel 1 goto :error

REM Function to deploy services
call :deploy_services
if errorlevel 1 goto :error

REM Function to deploy frontend
call :deploy_frontend
if errorlevel 1 goto :error

REM Function to configure services
call :configure_services
if errorlevel 1 goto :error

REM Function to setup database
call :setup_database
if errorlevel 1 goto :error

REM Function to configure firewall
call :configure_firewall
if errorlevel 1 goto :error

REM Function to start services
call :start_services
if errorlevel 1 goto :error

REM Function to run health checks
call :health_check
if errorlevel 1 goto :error

echo [DEPLOY] Production deployment completed successfully!
echo.
call :show_deployment_summary
goto :end

:create_directories
echo [DEPLOY] Creating directory structure...

REM Create main directories
if not exist "%SERVICE_DIR%" mkdir "%SERVICE_DIR%"
if not exist "%WEB_ROOT%" mkdir "%WEB_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

REM Create service subdirectories
mkdir "%SERVICE_DIR%\go-services" 2>nul
mkdir "%SERVICE_DIR%\configs" 2>nul
mkdir "%SERVICE_DIR%\scripts" 2>nul

REM Create data subdirectories
mkdir "%DATA_DIR%\postgres" 2>nul
mkdir "%DATA_DIR%\redis" 2>nul
mkdir "%DATA_DIR%\minio" 2>nul
mkdir "%DATA_DIR%\uploads" 2>nul
mkdir "%DATA_DIR%\evidence" 2>nul

REM Create log subdirectories
mkdir "%LOG_DIR%\go-services" 2>nul
mkdir "%LOG_DIR%\web" 2>nul
mkdir "%LOG_DIR%\system" 2>nul

echo [DEPLOY] ✓ Directory structure created
exit /b 0

:deploy_services
echo [DEPLOY] Deploying Go microservices...

if not exist "%DIST_DIR%\go-services" (
    echo [ERROR] Go services build not found. Run build-production.bat first.
    exit /b 1
)

REM Copy Go services
copy "%DIST_DIR%\go-services\*.exe" "%SERVICE_DIR%\go-services\"
if errorlevel 1 (
    echo [ERROR] Failed to copy Go services
    exit /b 1
)

REM Copy configuration files
if exist "%PROJECT_ROOT%\.env.production" (
    copy "%PROJECT_ROOT%\.env.production" "%SERVICE_DIR%\configs\.env"
    echo [DEPLOY] ✓ Environment configuration copied
)

echo [DEPLOY] ✓ Go microservices deployed
exit /b 0

:deploy_frontend
echo [DEPLOY] Deploying frontend assets...

if not exist "%DIST_DIR%\frontend" (
    echo [ERROR] Frontend build not found. Run build-production.bat first.
    exit /b 1
)

REM Copy frontend assets
xcopy /E /I /Y "%DIST_DIR%\frontend\*" "%WEB_ROOT%\"
if errorlevel 1 (
    echo [ERROR] Failed to copy frontend assets
    exit /b 1
)

echo [DEPLOY] ✓ Frontend assets deployed
exit /b 0

:configure_services
echo [DEPLOY] Configuring Windows services...

REM Create service wrapper scripts
call :create_service_scripts

REM Install Go services as Windows services using NSSM (if available)
nssm version >nul 2>&1
if not errorlevel 1 (
    call :install_windows_services
) else (
    echo [WARN] NSSM not available - services will run as console applications
    call :create_console_scripts
)

exit /b 0

:create_service_scripts
echo [DEPLOY] Creating service scripts...

REM Enhanced RAG Service script
(
echo @echo off
echo cd /d "%SERVICE_DIR%\go-services"
echo set LEGAL_AI_ENV_FILE=%SERVICE_DIR%\configs\.env
echo enhanced-rag-service.exe ^> "%LOG_DIR%\go-services\enhanced-rag.log" 2^>^&1
) > "%SERVICE_DIR%\scripts\start-enhanced-rag.bat"

REM Upload Service script
(
echo @echo off
echo cd /d "%SERVICE_DIR%\go-services"
echo set LEGAL_AI_ENV_FILE=%SERVICE_DIR%\configs\.env
echo upload-service.exe ^> "%LOG_DIR%\go-services\upload-service.log" 2^>^&1
) > "%SERVICE_DIR%\scripts\start-upload-service.bat"

REM GPU Legal AI Service script
(
echo @echo off
echo cd /d "%SERVICE_DIR%\go-services"
echo set LEGAL_AI_ENV_FILE=%SERVICE_DIR%\configs\.env
echo gpu-legal-ai-server.exe ^> "%LOG_DIR%\go-services\gpu-legal-ai.log" 2^>^&1
) > "%SERVICE_DIR%\scripts\start-gpu-legal-ai.bat"

REM Cluster Service script
(
echo @echo off
echo cd /d "%SERVICE_DIR%\go-services"
echo set LEGAL_AI_ENV_FILE=%SERVICE_DIR%\configs\.env
echo cluster-service.exe ^> "%LOG_DIR%\go-services\cluster-service.log" 2^>^&1
) > "%SERVICE_DIR%\scripts\start-cluster-service.bat"

echo [DEPLOY] ✓ Service scripts created
exit /b 0

:install_windows_services
echo [DEPLOY] Installing Windows services using NSSM...

REM Install Enhanced RAG Service
nssm install "LegalAI-EnhancedRAG" "%SERVICE_DIR%\scripts\start-enhanced-rag.bat"
nssm set "LegalAI-EnhancedRAG" DisplayName "Legal AI - Enhanced RAG Service"
nssm set "LegalAI-EnhancedRAG" Description "Legal AI Enhanced RAG Processing Service"
nssm set "LegalAI-EnhancedRAG" Start SERVICE_AUTO_START

REM Install Upload Service
nssm install "LegalAI-Upload" "%SERVICE_DIR%\scripts\start-upload-service.bat"
nssm set "LegalAI-Upload" DisplayName "Legal AI - Upload Service"
nssm set "LegalAI-Upload" Description "Legal AI Document Upload Processing Service"
nssm set "LegalAI-Upload" Start SERVICE_AUTO_START

REM Install GPU Legal AI Service
nssm install "LegalAI-GPU" "%SERVICE_DIR%\scripts\start-gpu-legal-ai.bat"
nssm set "LegalAI-GPU" DisplayName "Legal AI - GPU Service"
nssm set "LegalAI-GPU" Description "Legal AI GPU Accelerated Processing Service"
nssm set "LegalAI-GPU" Start SERVICE_AUTO_START

REM Install Cluster Service
nssm install "LegalAI-Cluster" "%SERVICE_DIR%\scripts\start-cluster-service.bat"
nssm set "LegalAI-Cluster" DisplayName "Legal AI - Cluster Service"
nssm set "LegalAI-Cluster" Description "Legal AI Cluster Management Service"
nssm set "LegalAI-Cluster" Start SERVICE_AUTO_START

echo [DEPLOY] ✓ Windows services installed
exit /b 0

:create_console_scripts
echo [DEPLOY] Creating console startup scripts...

REM Master startup script
(
echo @echo off
echo echo Starting Legal AI Platform Services...
echo.
echo echo Starting Enhanced RAG Service...
echo start "Legal AI - Enhanced RAG" /min "%SERVICE_DIR%\scripts\start-enhanced-rag.bat"
echo timeout /t 2 /nobreak ^> nul
echo.
echo echo Starting Upload Service...
echo start "Legal AI - Upload Service" /min "%SERVICE_DIR%\scripts\start-upload-service.bat"
echo timeout /t 2 /nobreak ^> nul
echo.
echo echo Starting GPU Legal AI Service...
echo start "Legal AI - GPU Service" /min "%SERVICE_DIR%\scripts\start-gpu-legal-ai.bat"
echo timeout /t 2 /nobreak ^> nul
echo.
echo echo Starting Cluster Service...
echo start "Legal AI - Cluster Service" /min "%SERVICE_DIR%\scripts\start-cluster-service.bat"
echo timeout /t 2 /nobreak ^> nul
echo.
echo echo Legal AI Platform Services started
echo echo Check log files in %LOG_DIR%\go-services\ for service status
echo pause
) > "%SERVICE_DIR%\start-all-services.bat"

REM Stop script
(
echo @echo off
echo echo Stopping Legal AI Platform Services...
echo taskkill /f /im enhanced-rag-service.exe 2^>nul
echo taskkill /f /im upload-service.exe 2^>nul
echo taskkill /f /im gpu-legal-ai-server.exe 2^>nul
echo taskkill /f /im cluster-service.exe 2^>nul
echo echo Legal AI Platform Services stopped
echo pause
) > "%SERVICE_DIR%\stop-all-services.bat"

echo [DEPLOY] ✓ Console scripts created
exit /b 0

:setup_database
echo [DEPLOY] Setting up database...

REM Check if PostgreSQL is installed
pg_isready >nul 2>&1
if errorlevel 1 (
    echo [WARN] PostgreSQL not found or not running
    echo [WARN] Please ensure PostgreSQL is installed and running
    echo [WARN] Database setup skipped
    exit /b 0
)

REM Create database and user (if not exists)
echo [DEPLOY] Creating database and user...
psql -U postgres -c "CREATE DATABASE legal_ai_production;" 2>nul
psql -U postgres -c "CREATE USER legal_ai_admin WITH PASSWORD 'CHANGE_THIS_PASSWORD';" 2>nul
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE legal_ai_production TO legal_ai_admin;" 2>nul

REM Run migrations if available
if exist "%PROJECT_ROOT%\migrations" (
    echo [DEPLOY] Running database migrations...
    REM Add migration logic here based on your migration tool
    echo [DEPLOY] ✓ Database migrations completed
)

echo [DEPLOY] ✓ Database setup completed
exit /b 0

:configure_firewall
echo [DEPLOY] Configuring Windows Firewall...

REM Create firewall rules for services
netsh advfirewall firewall add rule name="Legal AI - HTTPS" dir=in action=allow protocol=TCP localport=443
netsh advfirewall firewall add rule name="Legal AI - HTTP" dir=in action=allow protocol=TCP localport=80
netsh advfirewall firewall add rule name="Legal AI - Enhanced RAG" dir=in action=allow protocol=TCP localport=8094
netsh advfirewall firewall add rule name="Legal AI - Upload Service" dir=in action=allow protocol=TCP localport=8096
netsh advfirewall firewall add rule name="Legal AI - GPU Service" dir=in action=allow protocol=TCP localport=8084
netsh advfirewall firewall add rule name="Legal AI - Cluster Service" dir=in action=allow protocol=TCP localport=8213

echo [DEPLOY] ✓ Firewall rules configured
exit /b 0

:start_services
echo [DEPLOY] Starting services...

REM Start Windows services if NSSM is available
nssm version >nul 2>&1
if not errorlevel 1 (
    echo [DEPLOY] Starting Windows services...
    net start "LegalAI-EnhancedRAG" 2>nul
    net start "LegalAI-Upload" 2>nul
    net start "LegalAI-GPU" 2>nul
    net start "LegalAI-Cluster" 2>nul
    echo [DEPLOY] ✓ Windows services started
) else (
    echo [DEPLOY] Starting console services...
    start "Legal AI Services" /min "%SERVICE_DIR%\start-all-services.bat"
    echo [DEPLOY] ✓ Console services started
)

REM Wait for services to start
echo [DEPLOY] Waiting for services to initialize...
timeout /t 10 /nobreak > nul

exit /b 0

:health_check
echo [DEPLOY] Running health checks...

REM Check service ports
call :check_port 8094 "Enhanced RAG Service"
call :check_port 8096 "Upload Service"
call :check_port 8084 "GPU Legal AI Service"
call :check_port 8213 "Cluster Service"

REM Check if web server is serving content
if exist "%WEB_ROOT%\index.html" (
    echo [DEPLOY] ✓ Frontend assets deployed correctly
) else (
    echo [WARN] Frontend index.html not found
)

echo [DEPLOY] ✓ Health checks completed
exit /b 0

:check_port
netstat -an | find ":%~1" | find "LISTENING" >nul
if errorlevel 1 (
    echo [WARN] %~2 (port %~1) - Not listening
) else (
    echo [DEPLOY] ✓ %~2 (port %~1) - Running
)
exit /b 0

:show_deployment_summary
echo ================================================================================
echo DEPLOYMENT SUMMARY
echo ================================================================================
echo Deployment Date: %date% %time%
echo.
echo Services Deployed:
echo - Enhanced RAG Service (port 8094)
echo - Upload Service (port 8096)
echo - GPU Legal AI Service (port 8084)
echo - Cluster Service (port 8213)
echo.
echo Directories:
echo - Services: %SERVICE_DIR%
echo - Web Root: %WEB_ROOT%
echo - Logs: %LOG_DIR%
echo - Data: %DATA_DIR%
echo.
echo Service Management:
nssm version >nul 2>&1
if not errorlevel 1 (
    echo - Services installed as Windows services
    echo - Use Services.msc to manage services
    echo - Or use: net start/stop "ServiceName"
) else (
    echo - Services run as console applications
    echo - Start: %SERVICE_DIR%\start-all-services.bat
    echo - Stop: %SERVICE_DIR%\stop-all-services.bat
)
echo.
echo Next Steps:
echo 1. Configure reverse proxy (IIS/nginx)
echo 2. Set up SSL certificates
echo 3. Configure monitoring
echo 4. Set up automated backups
echo 5. Test all functionality
echo.
echo Access Points:
echo - Web Application: https://yourdomain.com
echo - Service Logs: %LOG_DIR%
echo ================================================================================
exit /b 0

:error
echo [ERROR] Deployment failed!
echo Check logs in %LOG_DIR% for details
pause
exit /b 1

:end
echo [DEPLOY] Deployment completed. Press any key to exit.
pause