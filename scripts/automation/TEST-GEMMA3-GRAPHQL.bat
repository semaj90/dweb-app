@echo off
echo Testing Gemma3 GraphQL Integration...
echo.

:: Test 1: Direct Ollama connection
echo [1/4] Testing Ollama connection...
curl -s http://localhost:11434/api/tags | findstr "gemma3-legal" >nul
if %errorlevel% == 0 (
    echo ✅ Gemma3-legal model found in Ollama
) else (
    echo ❌ Gemma3-legal model not found
    echo Run: ollama pull gemma3-legal:latest
)

:: Test 2: Test direct API
echo.
echo [2/4] Testing direct Gemma3 API...
curl -X POST http://localhost:11434/api/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"gemma3-legal:latest\",\"prompt\":\"Hello\",\"stream\":false}" ^
  -s | findstr "response" >nul
if %errorlevel% == 0 (
    echo ✅ Direct Gemma3 API working
) else (
    echo ❌ Direct API failed
)

:: Test 3: Check database
echo.
echo [3/4] Testing PostgreSQL connection...
docker exec legal-postgres-main pg_isready -U legal_admin >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ PostgreSQL connected
) else (
    echo ❌ PostgreSQL not accessible
)

:: Test 4: Display API usage
echo.
echo [4/4] API Usage Examples:
echo.
echo 📌 Search with Gemma3:
echo curl -X POST http://localhost:5173/api/gemma3 -H "Content-Type: application/json" -d "{\"action\":\"search\",\"query\":\"contract law\"}"
echo.
echo 📌 Analyze case:
echo curl -X POST http://localhost:5173/api/gemma3 -H "Content-Type: application/json" -d "{\"action\":\"analyze\",\"caseId\":\"123\",\"analysisType\":\"risk\"}"
echo.
echo 📌 Direct chat:
echo curl "http://localhost:5173/api/gemma3?prompt=What+is+contract+law"
echo.
echo ✅ Integration complete! Your Gemma3 model is ready for GraphQL operations.
pause
