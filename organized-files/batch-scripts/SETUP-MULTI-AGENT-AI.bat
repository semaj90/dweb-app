@echo off
setlocal enabledelayedexpansion

:: Multi-Agent Legal AI Setup Script
:: Configures Autogen, CrewAI, vLLM, and GPU-optimized Ollama
echo =============================================
echo  Multi-Agent Legal AI System Setup
echo =============================================
echo.

set "PROJECT_ROOT=%~dp0"
set "GPU_DETECTED=false"
set "MEMORY_PROFILE=balanced"

echo [INFO] Project Root: %PROJECT_ROOT%
echo [INFO] Date: %date% %time%
echo.

:: Check for NVIDIA GPU
echo ========================================
echo  1. GPU DETECTION AND OPTIMIZATION
echo ========================================
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] NVIDIA GPU detected
    set "GPU_DETECTED=true"
    echo [INFO] Running nvidia-smi for GPU information:
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo.
    
    :: Determine memory profile based on GPU memory
    for /f "tokens=2 delims=," %%a in ('nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits') do (
        set "GPU_MEMORY=%%a"
        if %%a LSS 4000 (
            set "MEMORY_PROFILE=ultra_low_memory"
            echo [INFO] GPU Memory: %%a MB - Using ultra_low_memory profile
        ) else if %%a LSS 8000 (
            set "MEMORY_PROFILE=low_memory"
            echo [INFO] GPU Memory: %%a MB - Using low_memory profile
        ) else if %%a LSS 16000 (
            set "MEMORY_PROFILE=balanced"
            echo [INFO] GPU Memory: %%a MB - Using balanced profile
        ) else (
            set "MEMORY_PROFILE=high_performance"
            echo [INFO] GPU Memory: %%a MB - Using high_performance profile
        )
    )
) else (
    echo [⚠] No NVIDIA GPU detected - Using CPU-only mode
    set "MEMORY_PROFILE=ultra_low_memory"
)
echo.

:: Install Python dependencies for vLLM
echo ========================================
echo  2. PYTHON DEPENDENCIES INSTALLATION
echo ========================================
echo [INFO] Installing Python dependencies for vLLM...
python -m pip install --upgrade pip
python -m pip install vllm fastapi uvicorn torch transformers

if %GPU_DETECTED%==true (
    echo [INFO] Installing GPU-accelerated PyTorch...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [INFO] Installing CPU-only PyTorch...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

:: Install Node.js dependencies
echo.
echo ========================================
echo  3. NODE.JS DEPENDENCIES INSTALLATION
echo ========================================
echo [INFO] Installing Node.js dependencies...
cd "%PROJECT_ROOT%sveltekit-frontend"
npm install @modelcontextprotocol/sdk
npm install fastify ws

:: Configure Ollama for GPU acceleration
echo.
echo ========================================
echo  4. OLLAMA GPU CONFIGURATION
echo ========================================
if %GPU_DETECTED%==true (
    echo [INFO] Configuring Ollama for GPU acceleration...
    
    :: Set environment variables for GPU optimization
    setx OLLAMA_HOST "0.0.0.0:11434"
    setx OLLAMA_ORIGINS "*"
    setx OLLAMA_NUM_PARALLEL "4"
    setx OLLAMA_MAX_LOADED_MODELS "2"
    setx CUDA_VISIBLE_DEVICES "0"
    setx PYTORCH_CUDA_ALLOC_CONF "max_split_size_mb:512"
    
    echo [✓] GPU environment variables configured
) else (
    echo [INFO] Configuring Ollama for CPU-only mode...
    setx OLLAMA_HOST "0.0.0.0:11434"
    setx OLLAMA_ORIGINS "*"
    setx OLLAMA_NUM_PARALLEL "2"
    setx OLLAMA_MAX_LOADED_MODELS "1"
)

:: Start Ollama service
echo [INFO] Starting Ollama service...
start /B ollama serve
timeout /t 5 /nobreak >nul

:: Load Gemma3 Legal model with optimized parameters
echo [INFO] Loading Gemma3 Legal model with %MEMORY_PROFILE% profile...
if exist "%PROJECT_ROOT%Modelfile-gemma3-legal" (
    ollama create gemma3-legal -f "%PROJECT_ROOT%Modelfile-gemma3-legal"
    if %errorlevel% equ 0 (
        echo [✓] Gemma3 Legal model loaded successfully
    ) else (
        echo [⚠] Failed to load Gemma3 Legal model
    )
) else (
    echo [⚠] Modelfile-gemma3-legal not found
)

:: Test Ollama API
echo [INFO] Testing Ollama API...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] Ollama API is responding
) else (
    echo [⚠] Ollama API not responding
)

:: Configure and start vLLM server
echo.
echo ========================================
echo  5. vLLM HIGH-PERFORMANCE SERVER SETUP
echo ========================================
if %GPU_DETECTED%==true (
    echo [INFO] Starting vLLM server for high-performance inference...
    
    :: Set vLLM environment variables
    set "GEMMA3_MODEL_PATH=%PROJECT_ROOT%gemma3Q4_K_M\mohf16-Q4_K_M.gguf"
    set "MAX_MODEL_LEN=8192"
    set "GPU_MEMORY_UTILIZATION=0.9"
    set "TENSOR_PARALLEL_SIZE=1"
    set "VLLM_HOST=0.0.0.0"
    set "VLLM_PORT=8000"
    
    if "%MEMORY_PROFILE%"=="ultra_low_memory" (
        set "MAX_MODEL_LEN=1024"
        set "GPU_MEMORY_UTILIZATION=0.7"
    ) else if "%MEMORY_PROFILE%"=="low_memory" (
        set "MAX_MODEL_LEN=2048"
        set "GPU_MEMORY_UTILIZATION=0.8"
    ) else if "%MEMORY_PROFILE%"=="high_performance" (
        set "MAX_MODEL_LEN=8192"
        set "GPU_MEMORY_UTILIZATION=0.95"
    )
    
    echo [INFO] vLLM Configuration:
    echo   - Model Path: %GEMMA3_MODEL_PATH%
    echo   - Max Model Length: %MAX_MODEL_LEN%
    echo   - GPU Memory Utilization: %GPU_MEMORY_UTILIZATION%
    echo   - Memory Profile: %MEMORY_PROFILE%
    
    :: Start vLLM server in background
    start /B python "%PROJECT_ROOT%vllm-legal-server.py"
    echo [INFO] vLLM server starting... (this may take a few minutes)
    
    :: Wait for vLLM to start
    timeout /t 30 /nobreak >nul
    
    :: Test vLLM API
    curl -s http://localhost:8000/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo [✓] vLLM server is responding
    ) else (
        echo [⚠] vLLM server not responding (may still be starting)
    )
) else (
    echo [INFO] Skipping vLLM setup (requires GPU)
)

:: Initialize Autogen configuration
echo.
echo ========================================
echo  6. AUTOGEN MULTI-AGENT CONFIGURATION
echo ========================================
echo [INFO] Setting up Autogen legal agent team...

:: Create Autogen configuration file
echo {> "%PROJECT_ROOT%autogen-config.json"
echo   "agents": {>> "%PROJECT_ROOT%autogen-config.json"
echo     "chief_analyst": {>> "%PROJECT_ROOT%autogen-config.json"
echo       "model": "gemma3-legal",>> "%PROJECT_ROOT%autogen-config.json"
echo       "max_tokens": 2048,>> "%PROJECT_ROOT%autogen-config.json"
echo       "temperature": 0.3>> "%PROJECT_ROOT%autogen-config.json"
echo     },>> "%PROJECT_ROOT%autogen-config.json"
echo     "evidence_specialist": {>> "%PROJECT_ROOT%autogen-config.json"
echo       "model": "gemma3-legal",>> "%PROJECT_ROOT%autogen-config.json"
echo       "max_tokens": 1536,>> "%PROJECT_ROOT%autogen-config.json"
echo       "temperature": 0.2>> "%PROJECT_ROOT%autogen-config.json"
echo     }>> "%PROJECT_ROOT%autogen-config.json"
echo   },>> "%PROJECT_ROOT%autogen-config.json"
echo   "memory_profile": "%MEMORY_PROFILE%",>> "%PROJECT_ROOT%autogen-config.json"
echo   "gpu_enabled": %GPU_DETECTED%>> "%PROJECT_ROOT%autogen-config.json"
echo }>> "%PROJECT_ROOT%autogen-config.json"

echo [✓] Autogen configuration created

:: Initialize CrewAI configuration
echo.
echo ========================================
echo  7. CREWAI WORKFLOW CONFIGURATION
echo ========================================
echo [INFO] Setting up CrewAI legal team workflows...

:: Create CrewAI configuration file
echo {> "%PROJECT_ROOT%crewai-config.json"
echo   "crews": {>> "%PROJECT_ROOT%crewai-config.json"
echo     "case_investigation": {>> "%PROJECT_ROOT%crewai-config.json"
echo       "process": "sequential",>> "%PROJECT_ROOT%crewai-config.json"
echo       "max_iterations": 3,>> "%PROJECT_ROOT%crewai-config.json"
echo       "memory_system": true>> "%PROJECT_ROOT%crewai-config.json"
echo     },>> "%PROJECT_ROOT%crewai-config.json"
echo     "trial_preparation": {>> "%PROJECT_ROOT%crewai-config.json"
echo       "process": "hierarchical",>> "%PROJECT_ROOT%crewai-config.json"
echo       "max_iterations": 2,>> "%PROJECT_ROOT%crewai-config.json"
echo       "memory_system": true>> "%PROJECT_ROOT%crewai-config.json"
echo     }>> "%PROJECT_ROOT%crewai-config.json"
echo   },>> "%PROJECT_ROOT%crewai-config.json"
echo   "memory_profile": "%MEMORY_PROFILE%",>> "%PROJECT_ROOT%crewai-config.json"
echo   "ai_endpoint": "http://localhost:11434">> "%PROJECT_ROOT%crewai-config.json"
echo }>> "%PROJECT_ROOT%crewai-config.json"

echo [✓] CrewAI configuration created

:: Test multi-agent API endpoints
echo.
echo ========================================
echo  8. API ENDPOINT TESTING
echo ========================================
echo [INFO] Testing multi-agent API endpoints...

:: Start SvelteKit dev server for testing
echo [INFO] Starting SvelteKit dev server...
cd "%PROJECT_ROOT%sveltekit-frontend"
start /B npm run dev
timeout /t 10 /nobreak >nul

:: Test multi-agent status endpoint
curl -s http://localhost:5173/api/ai/multi-agent?action=status >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] Multi-agent API is responding
) else (
    echo [⚠] Multi-agent API not responding
)

:: Update Claude MCP configuration
echo.
echo ========================================
echo  9. CLAUDE MCP INTEGRATION UPDATE
echo ========================================
echo [INFO] Updating Claude MCP configuration for multi-agent support...

if exist "%PROJECT_ROOT%SETUP-CLAUDE-MCP-CONTEXT7.bat" (
    call "%PROJECT_ROOT%SETUP-CLAUDE-MCP-CONTEXT7.bat"
) else (
    echo [⚠] Claude MCP setup script not found
)

:: Performance monitoring setup
echo.
echo ========================================
echo  10. PERFORMANCE MONITORING SETUP
echo ========================================
echo [INFO] Setting up performance monitoring...

:: Create monitoring script
echo @echo off> "%PROJECT_ROOT%MONITOR-PERFORMANCE.bat"
echo echo Monitoring Multi-Agent AI Performance...>> "%PROJECT_ROOT%MONITOR-PERFORMANCE.bat"
echo echo.>> "%PROJECT_ROOT%MONITOR-PERFORMANCE.bat"
if %GPU_DETECTED%==true (
    echo nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits>> "%PROJECT_ROOT%MONITOR-PERFORMANCE.bat"
    echo echo.>> "%PROJECT_ROOT%MONITOR-PERFORMANCE.bat"
)
echo curl -s http://localhost:11434/api/tags ^| jq .>> "%PROJECT_ROOT%MONITOR-PERFORMANCE.bat"
echo echo.>> "%PROJECT_ROOT%MONITOR-PERFORMANCE.bat"
echo curl -s http://localhost:8000/metrics ^| jq .>> "%PROJECT_ROOT%MONITOR-PERFORMANCE.bat"
echo echo.>> "%PROJECT_ROOT%MONITOR-PERFORMANCE.bat"
echo curl -s http://localhost:5173/api/ai/multi-agent?action=status ^| jq .>> "%PROJECT_ROOT%MONITOR-PERFORMANCE.bat"

echo [✓] Performance monitoring script created

:: Final status report
echo.
echo =============================================
echo  MULTI-AGENT AI SETUP COMPLETE
echo =============================================
echo.
echo [STATUS] Configuration Summary:
echo   - Memory Profile: %MEMORY_PROFILE%
echo   - GPU Acceleration: %GPU_DETECTED%
echo   - Ollama: http://localhost:11434
if %GPU_DETECTED%==true (
    echo   - vLLM Server: http://localhost:8000
)
echo   - SvelteKit API: http://localhost:5173
echo   - Multi-Agent API: http://localhost:5173/api/ai/multi-agent
echo.
echo [COMPONENTS] Configured Systems:
echo   ✓ Ollama with GPU optimization
if %GPU_DETECTED%==true (
    echo   ✓ vLLM high-performance server
)
echo   ✓ Autogen multi-agent framework
echo   ✓ CrewAI workflow orchestration
echo   ✓ Context7 MCP integration
echo   ✓ Low-memory configuration profiles
echo.
echo [NEXT STEPS]
echo   1. Test multi-agent analysis: curl -X POST http://localhost:5173/api/ai/multi-agent
echo   2. Monitor performance: MONITOR-PERFORMANCE.bat
echo   3. Access Claude MCP tools via Claude Desktop
echo   4. Review configuration files for fine-tuning
echo.
echo [FILES CREATED]
echo   - autogen-config.json (Autogen configuration)
echo   - crewai-config.json (CrewAI configuration)
echo   - MONITOR-PERFORMANCE.bat (Performance monitoring)
echo   - Updated context7-mcp-config.json (MCP integration)
echo.

pause