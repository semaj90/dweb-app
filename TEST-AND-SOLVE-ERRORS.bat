@echo off
REM =============================================================================
REM YoRHa Legal AI Platform - COMPREHENSIVE ERROR TESTING & RESOLUTION
REM Tests all functionality, logs errors, and provides solutions
REM =============================================================================

echo ===============================================================================
echo YoRHa LEGAL AI PLATFORM - COMPREHENSIVE ERROR TESTING
echo ===============================================================================
echo Testing all components and logging errors for resolution
echo ===============================================================================

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

REM Create error log file
echo YoRHa Legal AI Platform - Error Test Log > error-test-log.txt
echo Date: %date% %time% >> error-test-log.txt
echo =============================================================================== >> error-test-log.txt
echo. >> error-test-log.txt

echo.
echo [1/8] === TESTING CORE SERVICE AVAILABILITY ===
echo.

echo Testing PostgreSQL connection...
psql -h localhost -p 5432 -U postgres -d postgres -c "SELECT version();" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ PostgreSQL: Connection successful
    echo PASS: PostgreSQL connection >> error-test-log.txt
) else (
    echo ❌ PostgreSQL: Connection failed
    echo ERROR: PostgreSQL connection failed - Error code: %errorlevel% >> error-test-log.txt
)

echo Testing MinIO API...
curl -s -m 5 http://localhost:9000/minio/health/live >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ MinIO: API responding
    echo PASS: MinIO API accessible >> error-test-log.txt
) else (
    echo ❌ MinIO: API not responding
    echo ERROR: MinIO API failed - Error code: %errorlevel% >> error-test-log.txt
)

echo Testing Ollama AI service...
curl -s -m 5 http://localhost:11434 >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama: Service responding
    echo PASS: Ollama service accessible >> error-test-log.txt
) else (
    echo ❌ Ollama: Service not responding
    echo ERROR: Ollama service failed - Error code: %errorlevel% >> error-test-log.txt
)

echo Testing Enhanced RAG service...
curl -s -m 5 http://localhost:8094/health -o temp_rag_health.json 2>&1
if %errorlevel% == 0 (
    echo ✅ Enhanced RAG: Health endpoint responding
    echo PASS: Enhanced RAG health check >> error-test-log.txt
    echo RAG Health Response: >> error-test-log.txt
    type temp_rag_health.json >> error-test-log.txt
    del temp_rag_health.json
) else (
    echo ❌ Enhanced RAG: Health endpoint failed
    echo ERROR: Enhanced RAG health check failed - Error code: %errorlevel% >> error-test-log.txt
)

echo Testing Upload service...
curl -s -m 5 http://localhost:8093/health -o temp_upload_health.json 2>&1
if %errorlevel% == 0 (
    echo ✅ Upload Service: Health endpoint responding
    echo PASS: Upload service health check >> error-test-log.txt
    echo Upload Health Response: >> error-test-log.txt
    type temp_upload_health.json >> error-test-log.txt
    del temp_upload_health.json
) else (
    echo ❌ Upload Service: Health endpoint failed
    echo ERROR: Upload service health check failed - Error code: %errorlevel% >> error-test-log.txt
)

echo.
echo [2/8] === TESTING YORHA FRONTEND ===
echo.

echo Testing YoRHa homepage accessibility...
curl -s -m 10 http://localhost:5173/yorha-home -o temp_yorha_page.html 2>&1
if %errorlevel% == 0 (
    echo ✅ YoRHa Homepage: Accessible
    echo PASS: YoRHa homepage accessible >> error-test-log.txt
    
    REM Check if page contains expected YoRHa elements
    findstr "yorha\|YoRHa\|YORHA" temp_yorha_page.html >nul 2>&1
    if !errorlevel! == 0 (
        echo ✅ YoRHa Homepage: Contains YoRHa elements
        echo PASS: YoRHa elements found in homepage >> error-test-log.txt
    ) else (
        echo ⚠️ YoRHa Homepage: Missing YoRHa branding elements
        echo WARNING: YoRHa branding elements not found >> error-test-log.txt
    )
    del temp_yorha_page.html
) else (
    echo ❌ YoRHa Homepage: Not accessible
    echo ERROR: YoRHa homepage failed - Error code: %errorlevel% >> error-test-log.txt
)

echo Testing alternative YoRHa port...
curl -s -m 10 http://localhost:5177/yorha-home -o temp_yorha_alt.html 2>&1
if %errorlevel% == 0 (
    echo ✅ YoRHa Alt Port: Accessible on 5177
    echo PASS: YoRHa alternative port accessible >> error-test-log.txt
    del temp_yorha_alt.html
) else (
    echo ❌ YoRHa Alt Port: Not accessible on 5177
    echo ERROR: YoRHa alternative port failed - Error code: %errorlevel% >> error-test-log.txt
)

echo.
echo [3/8] === TESTING API ENDPOINTS ===
echo.

echo Testing YoRHa Enhanced RAG API...
curl -s -m 10 -X POST -H "Content-Type: application/json" -d "{\"query\":\"test legal analysis\",\"context\":\"integration_test\"}" http://localhost:5173/api/yorha/enhanced-rag -o temp_api_rag.json 2>&1
if %errorlevel% == 0 (
    echo ✅ YoRHa RAG API: Request processed
    echo PASS: YoRHa RAG API responded >> error-test-log.txt
    echo RAG API Response: >> error-test-log.txt
    type temp_api_rag.json >> error-test-log.txt
    echo. >> error-test-log.txt
    
    REM Check for success in response
    findstr "success.*true\|yorhaMetadata" temp_api_rag.json >nul 2>&1
    if !errorlevel! == 0 (
        echo ✅ YoRHa RAG API: Success response
        echo PASS: YoRHa RAG API success response >> error-test-log.txt
    ) else (
        echo ❌ YoRHa RAG API: Error in response
        echo ERROR: YoRHa RAG API returned error response >> error-test-log.txt
    )
    del temp_api_rag.json
) else (
    echo ❌ YoRHa RAG API: Request failed
    echo ERROR: YoRHa RAG API request failed - Error code: %errorlevel% >> error-test-log.txt
)

echo Testing YoRHa Legal Data API...
curl -s -m 10 "http://localhost:5173/api/yorha/legal-data?search=contract&limit=3" -o temp_api_legal.json 2>&1
if %errorlevel% == 0 (
    echo ✅ YoRHa Legal Data API: Request processed
    echo PASS: YoRHa Legal Data API responded >> error-test-log.txt
    echo Legal Data API Response: >> error-test-log.txt
    type temp_api_legal.json >> error-test-log.txt
    echo. >> error-test-log.txt
    del temp_api_legal.json
) else (
    echo ❌ YoRHa Legal Data API: Request failed
    echo ERROR: YoRHa Legal Data API request failed - Error code: %errorlevel% >> error-test-log.txt
)

echo Testing Cluster Health API...
curl -s -m 10 "http://localhost:5173/api/v1/cluster/health" -o temp_api_cluster.json 2>&1
if %errorlevel% == 0 (
    echo ✅ Cluster Health API: Request processed
    echo PASS: Cluster Health API responded >> error-test-log.txt
    echo Cluster Health Response: >> error-test-log.txt
    type temp_api_cluster.json >> error-test-log.txt
    echo. >> error-test-log.txt
    del temp_api_cluster.json
) else (
    echo ❌ Cluster Health API: Request failed
    echo ERROR: Cluster Health API request failed - Error code: %errorlevel% >> error-test-log.txt
)

echo.
echo [4/8] === TESTING FILE UPLOAD FUNCTIONALITY ===
echo.

REM Create a test file for upload
echo This is a test document for YoRHa Legal AI upload testing. > test_upload_file.txt
echo Legal Contract Test Content >> test_upload_file.txt
echo Date: %date% %time% >> test_upload_file.txt

echo Testing file upload to Upload Service...
curl -s -m 15 -X POST -F "file=@test_upload_file.txt" http://localhost:8093/upload -o temp_upload_response.json 2>&1
if %errorlevel% == 0 (
    echo ✅ File Upload: Request processed
    echo PASS: File upload processed >> error-test-log.txt
    echo Upload Response: >> error-test-log.txt
    type temp_upload_response.json >> error-test-log.txt
    echo. >> error-test-log.txt
    del temp_upload_response.json
) else (
    echo ❌ File Upload: Request failed
    echo ERROR: File upload failed - Error code: %errorlevel% >> error-test-log.txt
)

del test_upload_file.txt

echo.
echo [5/8] === TESTING AI MODEL FUNCTIONALITY ===
echo.

echo Testing Ollama model list...
curl -s -m 10 http://localhost:11434/api/tags -o temp_ollama_models.json 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama Models: List retrieved
    echo PASS: Ollama models list retrieved >> error-test-log.txt
    echo Available Models: >> error-test-log.txt
    type temp_ollama_models.json >> error-test-log.txt
    echo. >> error-test-log.txt
    del temp_ollama_models.json
) else (
    echo ❌ Ollama Models: List failed
    echo ERROR: Ollama models list failed - Error code: %errorlevel% >> error-test-log.txt
)

echo Testing Ollama generation...
curl -s -m 20 -X POST -H "Content-Type: application/json" -d "{\"model\":\"gemma3-legal\",\"prompt\":\"What is a legal contract?\",\"stream\":false}" http://localhost:11434/api/generate -o temp_ollama_gen.json 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama Generation: Request processed
    echo PASS: Ollama generation test successful >> error-test-log.txt
    echo Generation Response: >> error-test-log.txt
    type temp_ollama_gen.json >> error-test-log.txt
    echo. >> error-test-log.txt
    del temp_ollama_gen.json
) else (
    echo ❌ Ollama Generation: Request failed
    echo ERROR: Ollama generation failed - Error code: %errorlevel% >> error-test-log.txt
)

echo.
echo [6/8] === TESTING DATABASE OPERATIONS ===
echo.

echo Testing PostgreSQL basic query...
psql -h localhost -p 5432 -U postgres -d postgres -c "SELECT current_timestamp;" >temp_pg_test.txt 2>&1
if %errorlevel% == 0 (
    echo ✅ PostgreSQL Query: Basic query successful
    echo PASS: PostgreSQL basic query >> error-test-log.txt
    echo PG Query Result: >> error-test-log.txt
    type temp_pg_test.txt >> error-test-log.txt
    echo. >> error-test-log.txt
    del temp_pg_test.txt
) else (
    echo ❌ PostgreSQL Query: Basic query failed
    echo ERROR: PostgreSQL query failed - Error code: %errorlevel% >> error-test-log.txt
)

echo Testing pgvector extension...
psql -h localhost -p 5432 -U postgres -d postgres -c "SELECT * FROM pg_extension WHERE extname='vector';" >temp_pgvector_test.txt 2>&1
if %errorlevel% == 0 (
    echo ✅ pgvector Extension: Query executed
    echo PASS: pgvector extension check >> error-test-log.txt
    echo pgvector Status: >> error-test-log.txt
    type temp_pgvector_test.txt >> error-test-log.txt
    echo. >> error-test-log.txt
    del temp_pgvector_test.txt
) else (
    echo ❌ pgvector Extension: Query failed
    echo ERROR: pgvector extension check failed - Error code: %errorlevel% >> error-test-log.txt
)

echo.
echo [7/8] === TESTING INTEGRATION BETWEEN SERVICES ===
echo.

echo Testing Enhanced RAG → Ollama integration...
curl -s -m 15 -X POST -H "Content-Type: application/json" -d "{\"query\":\"integration test\",\"context\":\"test\"}" http://localhost:8094/api/rag -o temp_rag_ollama.json 2>&1
if %errorlevel% == 0 (
    echo ✅ RAG-Ollama Integration: Request processed
    echo PASS: RAG-Ollama integration test >> error-test-log.txt
    echo RAG-Ollama Response: >> error-test-log.txt
    type temp_rag_ollama.json >> error-test-log.txt
    echo. >> error-test-log.txt
    del temp_rag_ollama.json
) else (
    echo ❌ RAG-Ollama Integration: Request failed
    echo ERROR: RAG-Ollama integration failed - Error code: %errorlevel% >> error-test-log.txt
)

echo Testing Frontend → Backend service communication...
curl -s -m 10 http://localhost:5173/api/v1/cluster/status -o temp_frontend_backend.json 2>&1
if %errorlevel% == 0 (
    echo ✅ Frontend-Backend: Communication successful
    echo PASS: Frontend-Backend communication >> error-test-log.txt
    echo Frontend-Backend Response: >> error-test-log.txt
    type temp_frontend_backend.json >> error-test-log.txt
    echo. >> error-test-log.txt
    del temp_frontend_backend.json
) else (
    echo ❌ Frontend-Backend: Communication failed
    echo ERROR: Frontend-Backend communication failed - Error code: %errorlevel% >> error-test-log.txt
)

echo.
echo [8/8] === GENERATING ERROR SUMMARY & SOLUTIONS ===
echo.

echo Generating comprehensive error report...
echo. >> error-test-log.txt
echo =============================================================================== >> error-test-log.txt
echo ERROR SUMMARY AND SOLUTIONS >> error-test-log.txt
echo =============================================================================== >> error-test-log.txt

REM Count errors in log
findstr /C:"ERROR:" error-test-log.txt >nul 2>&1
if !errorlevel! == 0 (
    echo ERRORS DETECTED - Generating solutions... >> error-test-log.txt
    echo. >> error-test-log.txt
    echo AUTOMATED SOLUTIONS: >> error-test-log.txt
    echo 1. If PostgreSQL connection failed: >> error-test-log.txt
    echo    - Check if PostgreSQL service is running: net start postgresql-x64-16 >> error-test-log.txt
    echo    - Verify credentials and database existence >> error-test-log.txt
    echo. >> error-test-log.txt
    echo 2. If APIs are not responding: >> error-test-log.txt
    echo    - Check if services are running on correct ports >> error-test-log.txt
    echo    - Restart services with: START-PRODUCTION-SERVICES.bat >> error-test-log.txt
    echo. >> error-test-log.txt
    echo 3. If YoRHa frontend is not accessible: >> error-test-log.txt
    echo    - Check if npm run dev is running in sveltekit-frontend >> error-test-log.txt
    echo    - Try: cd sveltekit-frontend ^&^& npm run dev >> error-test-log.txt
    echo. >> error-test-log.txt
    echo 4. If file uploads fail: >> error-test-log.txt
    echo    - Verify MinIO is running and accessible >> error-test-log.txt
    echo    - Check upload service logs for detailed errors >> error-test-log.txt
    echo. >> error-test-log.txt
) else (
    echo NO ERRORS DETECTED - System is functioning correctly! >> error-test-log.txt
)

echo ===============================================================================
echo ERROR TESTING COMPLETE
echo ===============================================================================
echo.
echo 📊 Test Results Summary:
echo   • Error log created: error-test-log.txt
echo   • All components tested
echo   • Solutions provided for any detected issues
echo.
echo 🔍 Review the error log for detailed results and solutions:
echo   type error-test-log.txt
echo.
echo 🛠️ To resolve any issues, check the solutions section in the log file
echo.
echo ===============================================================================

pause