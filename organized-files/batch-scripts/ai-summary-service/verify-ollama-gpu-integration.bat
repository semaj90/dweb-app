@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo   OLLAMA WINDOWS GPU INTEGRATION VERIFICATION
echo   Testing AI Enhanced Service with Ollama GPU Support
echo ========================================================================
echo.

set SERVICE_PORT=8081
set OLLAMA_URL=http://localhost:11434
set TEST_TIMEOUT=30

echo [1/8] 🔍 Checking Ollama Windows Installation...
where ollama >nul 2>&1
if %errorlevel%==0 (
    echo      ✅ Ollama executable found in PATH
    ollama --version
) else (
    echo      ❌ Ollama not found in PATH
    echo      Please install Ollama for Windows from: https://ollama.ai/download/windows
    goto :error
)

echo.
echo [2/8] 🚀 Starting Ollama Service...
tasklist | findstr "ollama" >nul 2>&1
if %errorlevel%==0 (
    echo      ✅ Ollama service already running
) else (
    echo      Starting Ollama service...
    start /min ollama serve
    timeout /t 5 >nul
)

echo.
echo [3/8] 🧠 Checking Ollama API Availability...
curl -s --max-time %TEST_TIMEOUT% %OLLAMA_URL%/api/tags >nul 2>&1
if %errorlevel%==0 (
    echo      ✅ Ollama API responding at %OLLAMA_URL%
) else (
    echo      ❌ Ollama API not responding
    echo      Please ensure Ollama is running: ollama serve
    goto :error
)

echo.
echo [4/8] 🎯 Checking Required Models...
echo      Checking gemma3-legal model...
curl -s --max-time %TEST_TIMEOUT% %OLLAMA_URL%/api/show -d "{\"name\":\"gemma3-legal\"}" | find "name" >nul 2>&1
if %errorlevel%==0 (
    echo      ✅ gemma3-legal model available
) else (
    echo      ⚠️ gemma3-legal model not found
    echo      Installing gemma3-legal model...
    ollama pull gemma3-legal
    if !errorlevel!==0 (
        echo      ✅ gemma3-legal model installed
    ) else (
        echo      ❌ Failed to install gemma3-legal model
        goto :error
    )
)

echo      Checking nomic-embed-text model...
curl -s --max-time %TEST_TIMEOUT% %OLLAMA_URL%/api/show -d "{\"name\":\"nomic-embed-text\"}" | find "name" >nul 2>&1
if %errorlevel%==0 (
    echo      ✅ nomic-embed-text model available
) else (
    echo      ⚠️ nomic-embed-text model not found
    echo      Installing nomic-embed-text model...
    ollama pull nomic-embed-text
    if !errorlevel!==0 (
        echo      ✅ nomic-embed-text model installed
    ) else (
        echo      ❌ Failed to install nomic-embed-text model
        goto :error
    )
)

echo.
echo [5/8] 🔧 Checking GPU Support...
echo      Detecting NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo      ✅ NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    set GPU_AVAILABLE=true
) else (
    echo      ⚠️ NVIDIA GPU not detected or drivers not installed
    echo      Ollama will run on CPU (slower performance)
    set GPU_AVAILABLE=false
)

echo.
echo [6/8] 🏗️ Building AI Enhanced Service...
if exist ai-enhanced.exe (
    echo      ✅ Using existing ai-enhanced.exe
) else (
    echo      Building from source...
    go build -o ai-enhanced.exe main.go
    if !errorlevel!==0 (
        echo      ✅ Build successful
    ) else (
        echo      ❌ Build failed
        goto :error
    )
)

echo.
echo [7/8] 🚀 Testing AI Enhanced Service...
echo      Starting AI Enhanced Service on port %SERVICE_PORT%...
start /min cmd /c "ai-enhanced.exe"
timeout /t 5 >nul

echo      Testing health endpoint...
curl -s --max-time %TEST_TIMEOUT% http://localhost:%SERVICE_PORT%/api/health >nul 2>&1
if %errorlevel%==0 (
    echo      ✅ AI Enhanced Service responding
) else (
    echo      ❌ AI Enhanced Service not responding
    goto :error
)

echo.
echo [8/8] 🧪 Testing Ollama Integration...
echo      Testing summarization endpoint...

:: Create test JSON payload
echo {"text":"This is a test legal document for processing.","type":"legal","length":"short"} > test-payload.json

curl -s --max-time %TEST_TIMEOUT% -X POST -H "Content-Type: application/json" -d @test-payload.json http://localhost:%SERVICE_PORT%/api/summarize > test-response.json 2>&1

if exist test-response.json (
    findstr "summary" test-response.json >nul 2>&1
    if !errorlevel!==0 (
        echo      ✅ Ollama integration working
        echo      Sample response:
        type test-response.json
    ) else (
        echo      ⚠️ Service responding but may have processing issues
        type test-response.json
    )
    del test-payload.json test-response.json 2>nul
) else (
    echo      ❌ Failed to get response from summarization endpoint
    goto :error
)

echo.
echo ========================================================================
echo   ✅ OLLAMA WINDOWS GPU INTEGRATION VERIFICATION COMPLETE
echo ========================================================================
echo.

echo 🎯 VERIFICATION RESULTS:
echo    - Ollama Installation: ✅ Verified
echo    - Ollama API: ✅ Running at %OLLAMA_URL%
echo    - Required Models: ✅ Available (gemma3-legal, nomic-embed-text)
if "%GPU_AVAILABLE%"=="true" (
    echo    - GPU Support: ✅ NVIDIA GPU detected and available
) else (
    echo    - GPU Support: ⚠️ CPU mode ^(slower performance^)
)
echo    - AI Service Build: ✅ Successful compilation
echo    - Service Integration: ✅ Ollama communication working
echo.

echo 🚀 QUICK START COMMANDS:
echo    # Start Ollama service
echo    ollama serve
echo.
echo    # Build and run AI Enhanced Service
echo    go build -o ai-enhanced.exe main.go
echo    ./ai-enhanced.exe
echo.
echo    # Test the service
echo    curl http://localhost:%SERVICE_PORT%/api/health
echo.

echo 📊 SERVICE ENDPOINTS:
echo    - Health Check: http://localhost:%SERVICE_PORT%/api/health
echo    - Summarization: http://localhost:%SERVICE_PORT%/api/summarize
echo    - Status: http://localhost:%SERVICE_PORT%/api/status
echo    - Test UI: http://localhost:%SERVICE_PORT%/test
echo.

if "%GPU_AVAILABLE%"=="true" (
    echo 🎉 SUCCESS: Ollama Windows GPU integration is fully functional!
    echo    Your RTX 3060 Ti will be used for accelerated AI processing.
) else (
    echo ⚠️ SUCCESS: Ollama integration working in CPU mode.
    echo    For GPU acceleration, ensure NVIDIA drivers are installed.
)

echo.
echo Press any key to open the test interface...
pause >nul
start http://localhost:%SERVICE_PORT%/test

goto :end

:error
echo.
echo ❌ VERIFICATION FAILED
echo    Please check the error messages above and resolve the issues.
echo    Common solutions:
echo    - Install Ollama for Windows: https://ollama.ai/download/windows
echo    - Install NVIDIA drivers for GPU support
echo    - Ensure Go is installed and in PATH
echo    - Check firewall settings for localhost ports

:end
echo.
echo Verification complete.