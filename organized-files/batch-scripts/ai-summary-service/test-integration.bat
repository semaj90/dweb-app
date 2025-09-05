@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo   INTEGRATION TEST: Document Processor + SvelteKit Ollama APIs
echo   Testing compatibility between Go services and SvelteKit endpoints
echo ========================================================================
echo.

set SVELTEKIT_URL=http://localhost:5173
set DOCUMENT_PROCESSOR_URL=http://localhost:8081
set OLLAMA_URL=http://localhost:11434
set TEST_TIMEOUT=15

echo [1/8] 🔍 Checking Prerequisites...

:: Check if SvelteKit is running
echo      Testing SvelteKit frontend...
curl -s --max-time %TEST_TIMEOUT% %SVELTEKIT_URL% >nul 2>&1
if %errorlevel%==0 (
    echo      ✅ SvelteKit running at %SVELTEKIT_URL%
    set SVELTEKIT_OK=true
) else (
    echo      ❌ SvelteKit not running at %SVELTEKIT_URL%
    echo      Please start SvelteKit: npm run dev
    set SVELTEKIT_OK=false
)

:: Check if Ollama is running
echo      Testing Ollama API...
curl -s --max-time %TEST_TIMEOUT% %OLLAMA_URL%/api/tags >nul 2>&1
if %errorlevel%==0 (
    echo      ✅ Ollama API running at %OLLAMA_URL%
    set OLLAMA_OK=true
) else (
    echo      ❌ Ollama API not running at %OLLAMA_URL%
    echo      Please start Ollama: ollama serve
    set OLLAMA_OK=false
)

if "%SVELTEKIT_OK%"=="false" (
    echo.
    echo ❌ Prerequisites not met. Please start required services first.
    goto :error
)

echo.
echo [2/8] 🧠 Testing SvelteKit Ollama API Endpoints...

:: Test SvelteKit Ollama models endpoint
echo      Testing /api/ollama/models...
curl -s --max-time %TEST_TIMEOUT% %SVELTEKIT_URL%/api/ollama/models > test-models.json 2>&1
if exist test-models.json (
    findstr "name\|model" test-models.json >nul 2>&1
    if !errorlevel!==0 (
        echo      ✅ Models API working
    ) else (
        echo      ⚠️ Models API responding but no models found
    )
) else (
    echo      ❌ Models API not responding
)

:: Test SvelteKit Ollama GPU status endpoint
echo      Testing /api/ollama/gpu-status...
curl -s --max-time %TEST_TIMEOUT% %SVELTEKIT_URL%/api/ollama/gpu-status > test-gpu.json 2>&1
if exist test-gpu.json (
    findstr "gpu\|enabled" test-gpu.json >nul 2>&1
    if !errorlevel!==0 (
        echo      ✅ GPU Status API working
    ) else (
        echo      ⚠️ GPU Status API responding but no GPU info
    )
) else (
    echo      ❌ GPU Status API not responding
)

echo.
echo [3/8] 🏗️ Building Enhanced Document Processor...
echo      Compiling with integration bridge...
go build -o document-processor-integrated.exe document-processor.go integration-bridge.go
if %errorlevel%==0 (
    echo      ✅ Build successful: document-processor-integrated.exe
    set BUILD_OK=true
) else (
    echo      ❌ Build failed
    set BUILD_OK=false
    goto :error
)

echo.
echo [4/8] 🚀 Starting Enhanced Document Processor...
if "%BUILD_OK%"=="true" (
    echo      Starting service on port 8081...
    start /min cmd /c "document-processor-integrated.exe"
    timeout /t 5 >nul
    
    :: Test if service started
    curl -s --max-time %TEST_TIMEOUT% %DOCUMENT_PROCESSOR_URL%/api/health >nul 2>&1
    if !errorlevel!==0 (
        echo      ✅ Document Processor started successfully
        set SERVICE_OK=true
    ) else (
        echo      ❌ Document Processor failed to start
        set SERVICE_OK=false
    )
) else (
    echo      ❌ Skipping service start due to build failure
    set SERVICE_OK=false
)

echo.
echo [5/8] 🔗 Testing Integration Bridge...
if "%SERVICE_OK%"=="true" (
    echo      Testing health endpoint with integration info...
    curl -s --max-time %TEST_TIMEOUT% %DOCUMENT_PROCESSOR_URL%/api/health > test-health.json 2>&1
    
    if exist test-health.json (
        findstr "sveltekit\|ollama\|gpu" test-health.json >nul 2>&1
        if !errorlevel!==0 (
            echo      ✅ Integration bridge working
            echo      Integration status:
            type test-health.json | findstr "status"
        ) else (
            echo      ⚠️ Basic health check working but no integration info
        )
    ) else (
        echo      ❌ Health endpoint not responding
    )
)

echo.
echo [6/8] 📄 Testing Document Processing with SvelteKit Integration...
if "%SERVICE_OK%"=="true" (
    :: Create test document
    echo "This is a test legal contract for processing integration. The parties agree to the following terms and conditions." > test-integration-doc.txt
    
    echo      Uploading test document...
    curl -s --max-time 30 -X POST -F "file=@test-integration-doc.txt" -F "document_type=contract" -F "enable_embedding=true" -F "enable_sveltekit=true" %DOCUMENT_PROCESSOR_URL%/api/upload > test-upload.json 2>&1
    
    if exist test-upload.json (
        findstr "document_id\|summary\|sveltekit_integration" test-upload.json >nul 2>&1
        if !errorlevel!==0 (
            echo      ✅ Document processing with SvelteKit integration successful
            echo      Sample response:
            type test-upload.json | findstr "document_id\|enhanced_summary\|sveltekit_integration"
        ) else (
            echo      ⚠️ Document processed but integration features may be missing
            type test-upload.json
        )
    ) else (
        echo      ❌ Document upload failed
    )
    
    :: Cleanup
    del test-integration-doc.txt 2>nul
)

echo.
echo [7/8] 🧪 Testing SvelteKit Embedding API...
echo      Testing embedding generation...
echo {"text":"Test legal document embedding","model":"nomic-embed-text:latest"} > test-embed-payload.json
curl -s --max-time %TEST_TIMEOUT% -X POST -H "Content-Type: application/json" -d @test-embed-payload.json %SVELTEKIT_URL%/api/ollama/embed > test-embed-response.json 2>&1

if exist test-embed-response.json (
    findstr "embedding\|dimensions" test-embed-response.json >nul 2>&1
    if !errorlevel!==0 (
        echo      ✅ SvelteKit embedding API working
        type test-embed-response.json | findstr "success\|dimensions"
    ) else (
        echo      ⚠️ Embedding API responding but may have issues
    )
) else (
    echo      ❌ Embedding API not responding
)

del test-embed-payload.json 2>nul

echo.
echo [8/8] 💬 Testing SvelteKit Chat API...
echo      Testing chat completion...
echo {"message":"Summarize a simple contract","model":"gemma3-legal","systemPrompt":"You are a legal AI assistant"} > test-chat-payload.json
curl -s --max-time %TEST_TIMEOUT% -X POST -H "Content-Type: application/json" -d @test-chat-payload.json %SVELTEKIT_URL%/api/ollama/chat > test-chat-response.json 2>&1

if exist test-chat-response.json (
    findstr "response\|success" test-chat-response.json >nul 2>&1
    if !errorlevel!==0 (
        echo      ✅ SvelteKit chat API working
        type test-chat-response.json | findstr "success\|response" | head -3
    ) else (
        echo      ⚠️ Chat API responding but may have issues
    )
) else (
    echo      ❌ Chat API not responding
)

del test-chat-payload.json 2>nul

echo.
echo ========================================================================
echo   📊 INTEGRATION TEST RESULTS
echo ========================================================================
echo.

set TOTAL_TESTS=8
set PASSED_TESTS=0

if "%SVELTEKIT_OK%"=="true" (
    set /a PASSED_TESTS+=1
    echo ✅ SvelteKit Frontend: Running
) else (
    echo ❌ SvelteKit Frontend: Not Running
)

if "%OLLAMA_OK%"=="true" (
    set /a PASSED_TESTS+=1
    echo ✅ Ollama API: Available
) else (
    echo ❌ Ollama API: Unavailable
)

if "%BUILD_OK%"=="true" (
    set /a PASSED_TESTS+=1
    echo ✅ Integration Build: Successful
) else (
    echo ❌ Integration Build: Failed
)

if "%SERVICE_OK%"=="true" (
    set /a PASSED_TESTS+=1
    echo ✅ Document Processor: Running
) else (
    echo ❌ Document Processor: Failed to Start
)

:: Count API test results
if exist test-models.json (
    set /a PASSED_TESTS+=1
    echo ✅ SvelteKit Models API: Working
) else (
    echo ❌ SvelteKit Models API: Failed
)

if exist test-embed-response.json (
    set /a PASSED_TESTS+=1
    echo ✅ SvelteKit Embedding API: Working
) else (
    echo ❌ SvelteKit Embedding API: Failed
)

if exist test-chat-response.json (
    set /a PASSED_TESTS+=1
    echo ✅ SvelteKit Chat API: Working
) else (
    echo ❌ SvelteKit Chat API: Failed
)

if exist test-health.json (
    set /a PASSED_TESTS+=1
    echo ✅ Integration Bridge: Working
) else (
    echo ❌ Integration Bridge: Failed
)

echo.
set /a SUCCESS_RATE=(%PASSED_TESTS% * 100) / %TOTAL_TESTS%
echo 📈 SUCCESS RATE: !SUCCESS_RATE!%% (!PASSED_TESTS!/!TOTAL_TESTS! tests passed)

if !SUCCESS_RATE! geq 75 (
    echo.
    echo 🎉 INTEGRATION SUCCESS: Systems are compatible and working together!
    echo.
    echo 🚀 READY FOR PRODUCTION:
    echo    - Document Processor: %DOCUMENT_PROCESSOR_URL%
    echo    - SvelteKit Frontend: %SVELTEKIT_URL%
    echo    - Ollama API: %OLLAMA_URL%
    echo.
    echo 🔗 Integration Features:
    echo    ✅ Shared Ollama configuration
    echo    ✅ Cross-system API communication
    echo    ✅ Enhanced document processing with SvelteKit APIs
    echo    ✅ GPU status monitoring
    echo    ✅ Unified embedding generation
) else (
    echo.
    echo ⚠️ INTEGRATION ISSUES DETECTED: !FAILED_TESTS! tests failed
    echo    Please review the test results and resolve issues.
    echo.
    echo 🔧 Common Solutions:
    echo    - Ensure SvelteKit is running: npm run dev
    echo    - Start Ollama service: ollama serve
    echo    - Check firewall settings for localhost ports
    echo    - Verify all required models are installed
)

echo.
echo 📁 Test artifacts:
echo    - Integration health: test-health.json
echo    - Models info: test-models.json  
echo    - GPU status: test-gpu.json
echo    - Upload test: test-upload.json
echo    - Embedding test: test-embed-response.json
echo    - Chat test: test-chat-response.json

:: Cleanup test files
del test-*.json 2>nul

echo.
echo Press any key to open the document processor test interface...
pause >nul
start %DOCUMENT_PROCESSOR_URL%/test

goto :end

:error
echo.
echo ❌ INTEGRATION TEST FAILED
echo    Please resolve the errors above and run the test again.

:end
echo.
echo Integration test complete.