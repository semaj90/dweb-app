@echo off
echo ================================
echo GPU AI Load Testing Suite
echo RTX 3060 Ti Performance Test
echo ================================
echo.

REM Check if service is running
curl -s http://localhost:8084/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Legal AI Service not running. Please start it first.
    echo     Run: START-GPU-LEGAL-AI-8084.bat
    pause
    exit /b 1
)

echo Select test type:
echo [1] Quick Test (1 document)
echo [2] Concurrent Test (3 documents - GPU limit)
echo [3] Batch Test (5 documents)
echo [4] Stress Test (10 documents, sequential)
echo [5] Streaming Test
echo [6] Cache Performance Test
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto quick_test
if "%choice%"=="2" goto concurrent_test
if "%choice%"=="3" goto batch_test
if "%choice%"=="4" goto stress_test
if "%choice%"=="5" goto stream_test
if "%choice%"=="6" goto cache_test
goto invalid_choice

:quick_test
echo.
echo [*] Running Quick Test...
echo.
curl -X POST http://localhost:8084/api/ai/summarize ^
  -H "Content-Type: application/json" ^
  -d "{\"content\":\"This Purchase Agreement is entered into as of January 1, 2025, between ABC Corporation, a Delaware corporation (Buyer), and XYZ Holdings, LLC (Seller). The Seller agrees to sell and the Buyer agrees to purchase certain assets as described herein for a total purchase price of $5,000,000 USD. The closing shall occur within 30 days of execution. Seller warrants clear title to all assets. Each party shall indemnify the other against losses arising from breach of this agreement.\",\"document_type\":\"contract\",\"options\":{\"style\":\"executive\",\"max_length\":500}}"
echo.
echo.
echo Test complete!
goto end

:concurrent_test
echo.
echo [*] Running Concurrent Test (3 simultaneous requests)...
echo.

REM Start 3 concurrent requests (GPU concurrency limit)
start /B curl -X POST http://localhost:8084/api/ai/summarize -H "Content-Type: application/json" -d "{\"content\":\"Legal document 1: Contract for services...\",\"document_type\":\"contract\"}" > test1.txt 2>&1
start /B curl -X POST http://localhost:8084/api/ai/summarize -H "Content-Type: application/json" -d "{\"content\":\"Legal document 2: Employment agreement...\",\"document_type\":\"employment\"}" > test2.txt 2>&1
start /B curl -X POST http://localhost:8084/api/ai/summarize -H "Content-Type: application/json" -d "{\"content\":\"Legal document 3: Non-disclosure agreement...\",\"document_type\":\"nda\"}" > test3.txt 2>&1

echo Waiting for concurrent requests to complete...
timeout /t 5 /nobreak >nul

echo.
echo Results:
type test1.txt 2>nul
type test2.txt 2>nul
type test3.txt 2>nul
del test1.txt test2.txt test3.txt 2>nul
goto end

:batch_test
echo.
echo [*] Running Batch Test (5 documents)...
echo.

echo Creating batch request...
echo [ > batch.json
echo   {\"content\":\"Contract 1: Purchase agreement for $1M acquisition...\",\"document_type\":\"contract\"}, >> batch.json
echo   {\"content\":\"Contract 2: Service level agreement with 99.9% uptime...\",\"document_type\":\"sla\"}, >> batch.json
echo   {\"content\":\"Contract 3: Intellectual property licensing agreement...\",\"document_type\":\"ip_license\"}, >> batch.json
echo   {\"content\":\"Contract 4: Joint venture agreement between parties...\",\"document_type\":\"joint_venture\"}, >> batch.json
echo   {\"content\":\"Contract 5: Merger and acquisition agreement...\",\"document_type\":\"merger\"} >> batch.json
echo ] >> batch.json

curl -X POST http://localhost:8084/api/ai/summarize/batch ^
  -H "Content-Type: application/json" ^
  -d @batch.json

del batch.json 2>nul
goto end

:stress_test
echo.
echo [*] Running Stress Test (10 sequential requests)...
echo.
echo This will test GPU memory management and cooling...
echo.

for /L %%i in (1,1,10) do (
    echo Request %%i/10...
    curl -s -X POST http://localhost:8084/api/ai/summarize ^
      -H "Content-Type: application/json" ^
      -d "{\"content\":\"Test document %%i: This is a legal document requiring summarization. It contains multiple clauses, terms, and conditions that need to be analyzed and summarized by the AI model.\",\"document_type\":\"test_%%i\"}" ^
      -o nul
    echo Done.
)

echo.
echo Stress test complete!
goto end

:stream_test
echo.
echo [*] Running Streaming Test...
echo.
echo Streaming responses from GPU-accelerated AI...
echo.

curl -N -X POST http://localhost:8084/api/ai/summarize/stream ^
  -H "Content-Type: application/json" ^
  -H "Accept: text/event-stream" ^
  -d "{\"content\":\"Comprehensive legal document requiring detailed analysis and streaming response. This agreement covers multiple aspects including intellectual property rights, confidentiality clauses, payment terms, and dispute resolution mechanisms.\",\"document_type\":\"comprehensive\"}"

goto end

:cache_test
echo.
echo [*] Running Cache Performance Test...
echo.

echo Creating test document...
set "test_content=This is a test legal document for cache performance testing. It includes standard contractual terms and conditions."

echo First request (will be cached)...
curl -w "\nTime: %%{time_total}s\n" -X POST http://localhost:8084/api/ai/summarize ^
  -H "Content-Type: application/json" ^
  -d "{\"content\":\"%test_content%\",\"document_type\":\"cache_test\"}" ^
  -o first_response.json

echo.
echo Second request (should hit cache)...
curl -w "\nTime: %%{time_total}s\n" -X POST http://localhost:8084/api/ai/summarize ^
  -H "Content-Type: application/json" ^
  -d "{\"content\":\"%test_content%\",\"document_type\":\"cache_test\"}" ^
  -o second_response.json

echo.
echo Comparing responses...
fc first_response.json second_response.json >nul 2>&1
if %errorlevel%==0 (
    echo [✓] Cache working correctly - responses match
) else (
    echo [!] Responses differ - cache may not be working
)

del first_response.json second_response.json 2>nul
goto end

:invalid_choice
echo Invalid choice. Please run the script again.
goto end

:end
echo.
echo ================================
echo Test Complete
echo ================================
echo.
echo View metrics at: http://localhost:8084/api/metrics
echo View health at: http://localhost:8084/api/health
echo.
pause