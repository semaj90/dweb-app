@echo off
echo Stopping Legal AI Phase 3-4 System - Enhanced Features...
echo.

echo [1/2] Stopping Phase 3-4 Docker services...
docker-compose -f docker-compose-phase34-DEFINITIVE.yml down

if %ERRORLEVEL% neq 0 (
    echo [!] WARNING: Some containers may not have stopped cleanly.
    echo This is usually not a problem.
)

echo [2/2] Cleaning up resources...
echo   - Stopping PostgreSQL
echo   - Stopping Redis cache
echo   - Stopping RabbitMQ
echo   - Stopping Neo4j
echo   - Stopping Qdrant
echo   - Stopping Ollama
echo   - Stopping TTS service

echo.
echo ========================================
echo   PHASE 3-4 SYSTEM STOPPED SUCCESSFULLY
echo ========================================
echo.
echo All enhanced features have been safely shut down:
echo   ❌ XState AI Summary Components (stopped)
echo   ❌ Evidence Report Analysis (stopped)
echo   ❌ Case Synthesis Workflow (stopped)
echo   ❌ Voice synthesis services (stopped)
echo   ❌ All databases and services (stopped)
echo.
echo To restart the system, run: START-PHASE34-ENHANCED.bat
echo.
pause