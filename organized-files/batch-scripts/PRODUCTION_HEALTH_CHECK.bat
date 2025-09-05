@echo off
setlocal EnableDelayedExpansion

title Legal AI Platform - Production Health Check

echo.
echo ================================================================================
echo 🏥 LEGAL AI PLATFORM - COMPREHENSIVE HEALTH CHECK
echo ================================================================================
echo Verifying all services and end-to-end connectivity...
echo.

set HEALTH_SCORE=0
set MAX_SCORE=15
set FAILED_SERVICES=

REM ================================================================================
REM DATABASE SERVICES HEALTH CHECK
REM ================================================================================

echo 📊 DATABASE SERVICES HEALTH CHECK
echo ================================================================================

REM PostgreSQL Health Check
echo Testing PostgreSQL connection...
timeout 3 >nul
psql -h localhost -U postgres -d legal_ai_db -c "SELECT 1 as health_check;" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ PostgreSQL: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ PostgreSQL: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! PostgreSQL
)

REM Redis Health Check
echo Testing Redis connection...
redis-cli ping >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Redis: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ Redis: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! Redis
)

REM Neo4j Health Check
echo Testing Neo4j connection...
curl -s http://localhost:7474/db/system/tx/commit >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Neo4j: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ Neo4j: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! Neo4j
)

echo.

REM ================================================================================
REM MESSAGE BROKER SERVICES HEALTH CHECK
REM ================================================================================

echo 📨 MESSAGE BROKER SERVICES HEALTH CHECK
echo ================================================================================

REM RabbitMQ Health Check
echo Testing RabbitMQ connection...
curl -s http://guest:guest@localhost:15672/api/overview >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ RabbitMQ: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ RabbitMQ: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! RabbitMQ
)

echo.

REM ================================================================================
REM STORAGE SERVICES HEALTH CHECK
REM ================================================================================

echo 🪣 STORAGE SERVICES HEALTH CHECK
echo ================================================================================

REM MinIO Health Check
echo Testing MinIO connection...
curl -s http://localhost:9000/minio/health/live >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ MinIO: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ MinIO: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! MinIO
)

REM Qdrant Health Check
echo Testing Qdrant connection...
curl -s http://localhost:6333/health >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Qdrant: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ Qdrant: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! Qdrant
)

echo.

REM ================================================================================
REM AI SERVICES HEALTH CHECK
REM ================================================================================

echo 🧠 AI SERVICES HEALTH CHECK
echo ================================================================================

REM Ollama Primary Health Check
echo Testing Ollama Primary (11434)...
curl -s http://localhost:11434/api/version >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Ollama Primary: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ Ollama Primary: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! Ollama-Primary
)

REM Ollama Secondary Health Check
echo Testing Ollama Secondary (11435)...
curl -s http://localhost:11435/api/version >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Ollama Secondary: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ Ollama Secondary: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! Ollama-Secondary
)

REM Ollama Embeddings Health Check
echo Testing Ollama Embeddings (11436)...
curl -s http://localhost:11436/api/version >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Ollama Embeddings: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ Ollama Embeddings: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! Ollama-Embeddings
)

echo.

REM ================================================================================
REM GO MICROSERVICES HEALTH CHECK
REM ================================================================================

echo ⚙️ GO MICROSERVICES HEALTH CHECK
echo ================================================================================

REM Enhanced RAG Service Health Check
echo Testing Enhanced RAG Service (8094)...
curl -s http://localhost:8094/health >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Enhanced RAG Service: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ Enhanced RAG Service: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! Enhanced-RAG
)

REM Upload Service Health Check
echo Testing Upload Service (8093)...
curl -s http://localhost:8093/health >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Upload Service: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ Upload Service: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! Upload-Service
)

REM Vector Redis Service Health Check
echo Testing Vector Redis Service (8095)...
curl -s http://localhost:8095/health >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Vector Redis Service: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ Vector Redis Service: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! Vector-Redis
)

echo.

REM ================================================================================
REM FRONTEND SERVICE HEALTH CHECK
REM ================================================================================

echo 🎨 FRONTEND SERVICE HEALTH CHECK
echo ================================================================================

REM SvelteKit Frontend Health Check
echo Testing SvelteKit Frontend (5173)...
curl -s http://localhost:5173 >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ SvelteKit Frontend: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ❌ SvelteKit Frontend: UNHEALTHY
    set FAILED_SERVICES=!FAILED_SERVICES! SvelteKit
)

echo.

REM ================================================================================
REM MONITORING SERVICES HEALTH CHECK
REM ================================================================================

echo 📊 MONITORING SERVICES HEALTH CHECK
echo ================================================================================

REM Elasticsearch Health Check
echo Testing Elasticsearch (9200)...
curl -s http://localhost:9200/_cluster/health >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Elasticsearch: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ⚠️ Elasticsearch: OPTIONAL (Not Required)
)

REM Kibana Health Check
echo Testing Kibana (5601)...
curl -s http://localhost:5601/api/status >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Kibana: HEALTHY
    set /a HEALTH_SCORE+=1
) else (
    echo ⚠️ Kibana: OPTIONAL (Not Required)
)

echo.

REM ================================================================================
REM END-TO-END WORKFLOW TEST
REM ================================================================================

echo 🔄 END-TO-END WORKFLOW TEST
echo ================================================================================

echo Testing complete workflow chain...

REM Test 1: Frontend to API Gateway
echo 1. Testing Frontend → API Gateway...
curl -s http://localhost:5173/api/health >nul 2>&1
if !errorlevel! equ 0 (
    echo   ✅ Frontend API Gateway: Working
) else (
    echo   ❌ Frontend API Gateway: Failed
)

REM Test 2: API to Go Microservices
echo 2. Testing API → Go Microservices...
curl -s -X POST http://localhost:8094/api/test -H "Content-Type: application/json" -d "{\"test\":\"health\"}" >nul 2>&1
if !errorlevel! equ 0 (
    echo   ✅ API → Microservices: Working
) else (
    echo   ❌ API → Microservices: Failed
)

REM Test 3: Database Integration
echo 3. Testing Database Integration...
psql -h localhost -U postgres -d legal_ai_db -c "SELECT version();" >nul 2>&1
if !errorlevel! equ 0 (
    echo   ✅ Database Integration: Working
) else (
    echo   ❌ Database Integration: Failed
)

echo.

REM ================================================================================
REM HEALTH SCORE CALCULATION
REM ================================================================================

echo ================================================================================
echo 📊 SYSTEM HEALTH REPORT
echo ================================================================================

set /a HEALTH_PERCENTAGE=(!HEALTH_SCORE! * 100) / !MAX_SCORE!

echo Health Score: !HEALTH_SCORE!/!MAX_SCORE! (!HEALTH_PERCENTAGE!%%)
echo.

if !HEALTH_PERCENTAGE! geq 90 (
    echo 🟢 SYSTEM STATUS: EXCELLENT
    echo All critical services are operational.
) else if !HEALTH_PERCENTAGE! geq 75 (
    echo 🟡 SYSTEM STATUS: GOOD
    echo Most services are operational with minor issues.
) else if !HEALTH_PERCENTAGE! geq 50 (
    echo 🟠 SYSTEM STATUS: WARNING
    echo Several services need attention.
) else (
    echo 🔴 SYSTEM STATUS: CRITICAL
    echo Major service failures detected.
)

echo.

if defined FAILED_SERVICES (
    echo ❌ FAILED SERVICES:
    for %%s in (!FAILED_SERVICES!) do (
        echo   • %%s
    )
    echo.
    echo 🔧 RECOMMENDED ACTIONS:
    echo   1. Check service logs for error details
    echo   2. Verify service configurations
    echo   3. Restart failed services individually
    echo   4. Check network connectivity and ports
    echo   5. Verify database connections and permissions
    echo.
)

echo ================================================================================
echo 🌐 SERVICE ACCESS URLS
echo ================================================================================
echo Frontend:          http://localhost:5173
echo Enhanced RAG:       http://localhost:8094/health
echo Upload Service:     http://localhost:8093/health
echo Ollama API:         http://localhost:11434/api/version
echo MinIO Console:      http://localhost:9001
echo Neo4j Browser:      http://localhost:7474
echo Qdrant Dashboard:   http://localhost:6333/dashboard
echo ================================================================================

if !HEALTH_PERCENTAGE! geq 75 (
    echo.
    echo ✅ SYSTEM READY FOR PRODUCTION USE
    echo.
    pause
    start http://localhost:5173
) else (
    echo.
    echo ⚠️ SYSTEM REQUIRES MAINTENANCE BEFORE PRODUCTION USE
    echo.
    pause
)

endlocal