@echo off
echo 🔍 System Health Check
echo ====================

echo Checking Docker containers...
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo Checking database connection...
set PGPASSWORD=LegalAI2024!
psql -h localhost -p 5432 -U legal_admin -d legal_ai_db -c "SELECT 'Database OK' as status;" 2>nul || echo ❌ Database connection failed

echo.
echo Checking Redis...
redis-cli ping 2>nul || echo ❌ Redis connection failed

echo.
echo Checking Qdrant...
curl -s http://localhost:6333/health | findstr "ok" >nul || echo ❌ Qdrant connection failed

echo.
echo Checking Ollama...
curl -s http://localhost:11434/api/version | findstr "version" >nul || echo ❌ Ollama connection failed

echo.
echo System check complete.
pause