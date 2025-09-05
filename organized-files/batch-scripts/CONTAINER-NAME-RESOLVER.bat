@echo off
title Container Name Resolver
color 0A

echo ========================================  
echo CONTAINER NAME RESOLVER
echo ========================================
echo.

echo Detecting active containers...
echo.

:: Find PostgreSQL container
for /f "tokens=1" %%i in ('docker ps --format "{{.Names}}" | findstr postgres') do (
    set "POSTGRES_CONTAINER=%%i"
    goto :postgres_found
)
:postgres_found
echo PostgreSQL: %POSTGRES_CONTAINER%

:: Find Redis container  
for /f "tokens=1" %%i in ('docker ps --format "{{.Names}}" | findstr redis') do (
    set "REDIS_CONTAINER=%%i"
    goto :redis_found
)
:redis_found
echo Redis: %REDIS_CONTAINER%

:: Find Qdrant container
for /f "tokens=1" %%i in ('docker ps --format "{{.Names}}" | findstr qdrant') do (
    set "QDRANT_CONTAINER=%%i"
    goto :qdrant_found
)
:qdrant_found
echo Qdrant: %QDRANT_CONTAINER%

:: Find Ollama container
for /f "tokens=1" %%i in ('docker ps --format "{{.Names}}" | findstr ollama') do (
    set "OLLAMA_CONTAINER=%%i"
    goto :ollama_found
)
:ollama_found
echo Ollama: %OLLAMA_CONTAINER%

echo.
echo Testing services with detected names...

echo PostgreSQL:
docker exec %POSTGRES_CONTAINER% pg_isready -U legal_admin && echo ✅ Ready || echo ❌ Down

echo Redis:
docker exec %REDIS_CONTAINER% redis-cli ping && echo ✅ Ready || echo ❌ Down

echo Qdrant:
curl -s http://localhost:6333/health && echo ✅ Ready || echo ❌ Down

echo Ollama:
curl -s http://localhost:11434/api/version && echo ✅ Ready || echo ❌ Down

pause
