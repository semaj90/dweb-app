@echo off
:: Integrated GPU Legal AI System Startup - Production Ready

setlocal EnableDelayedExpansion

:: Environment validation
call :validate_environment || exit /b 1

:: Database migration
call :run_migrations || exit /b 1

:: Start core services
call :start_core_services || exit /b 1

:: Start GPU services
call :start_gpu_services || exit /b 1

:: Start monitoring
call :start_monitoring || exit /b 1

:: Health verification
call :verify_system_health || exit /b 1

echo ✅ Integrated system online
goto :eof

:validate_environment
nvcc --version >nul || (echo CUDA missing & exit /b 1)
redis-windows\redis-cli.exe ping >nul || net start Redis
psql -U legal_admin -d legal_ai_db -c "SELECT 1" >nul || (echo PostgreSQL offline & exit /b 1)
exit /b 0

:run_migrations
echo Applying database migrations...
psql -U legal_admin -d legal_ai_db -f database\gpu-schema-migration.sql
if %ERRORLEVEL% NEQ 0 (echo Migration failed & exit /b 1)
exit /b 0

:start_core_services
echo Starting core services...
start /B redis-windows\redis-server.exe redis-windows\redis.conf
timeout /t 2 /nobreak >nul
pm2 start ecosystem.config.js --env production
exit /b 0

:start_gpu_services
echo Starting GPU services...
cd go-microservice

:: Build if needed
if not exist legal-processor-enhanced.exe (
    call BUILD-GPU-SIMD-FIXED.bat || exit /b 1
)

:: Start main GPU processor
start /B legal-processor-enhanced.exe

:: Start auto-indexer
start /B go run auto-indexer-service.go

cd ..
exit /b 0

:start_monitoring
echo Starting monitoring...
start /B node monitor-dashboard.mjs
start /B python analytics-engine\performance-monitor.py
exit /b 0

:verify_system_health
timeout /t 5 /nobreak >nul
curl -f http://localhost:8080/health | findstr gpu_enabled >nul || (echo GPU service failed & exit /b 1)
curl -f http://localhost:5173 >nul || echo Frontend may still be starting
exit /b 0

endlocal