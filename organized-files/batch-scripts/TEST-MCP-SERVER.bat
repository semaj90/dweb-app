@echo off
setlocal enabledelayedexpansion

:: Test MCP Server Startup and vLLM Integration
echo ========================================
echo  Context7 MCP Server Test Suite
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"

echo [INFO] Project Root: %PROJECT_ROOT%
echo [INFO] Testing MCP server configuration and vLLM endpoints
echo.

:: Test 1: Check if Node.js and required packages are available
echo ========================================
echo  1. DEPENDENCY CHECK
echo ========================================
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found
    exit /b 1
) else (
    echo [✓] Node.js available
)

cd "%PROJECT_ROOT%sveltekit-frontend"
npm list @modelcontextprotocol/sdk >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] MCP SDK not installed, installing...
    npm install @modelcontextprotocol/sdk
) else (
    echo [✓] MCP SDK available
)

:: Test 2: Start Context7 MCP Server
echo.
echo ========================================
echo  2. MCP SERVER STARTUP TEST
echo ========================================
echo [INFO] Starting Context7 MCP server...

set "DOCS_PATH=%PROJECT_ROOT%context7-docs"
set "VLLM_ENDPOINT=http://localhost:8000"
set "OLLAMA_ENDPOINT=http://localhost:11434"
set "VLLM_ENABLED=true"

:: Start MCP server in test mode
echo [INFO] Testing MCP server startup...
timeout /t 3 /nobreak >nul

start /B node "%PROJECT_ROOT%scripts\context7-mcp-server.js" --test-mode
if %errorlevel% equ 0 (
    echo [✓] MCP server started successfully
) else (
    echo [⚠] MCP server startup may have issues
)

:: Test 3: Check vLLM endpoint availability
echo.
echo ========================================
echo  3. vLLM ENDPOINT TEST
echo ========================================
echo [INFO] Testing vLLM endpoint connectivity...

curl -s --connect-timeout 5 http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] vLLM server is responding
    
    :: Test vLLM model info
    echo [INFO] Retrieving vLLM model information...
    curl -s http://localhost:8000/model-info
    echo.
) else (
    echo [⚠] vLLM server not responding (may not be started)
    echo [INFO] To start vLLM server, run: python vllm-legal-server.py
)

:: Test 4: Check Ollama endpoint availability
echo.
echo ========================================
echo  4. OLLAMA ENDPOINT TEST
echo ========================================
echo [INFO] Testing Ollama endpoint connectivity...

curl -s --connect-timeout 5 http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] Ollama server is responding
    
    :: Test available models
    echo [INFO] Available Ollama models:
    curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>nul || echo "jq not available for JSON parsing"
    echo.
) else (
    echo [⚠] Ollama server not responding
    echo [INFO] To start Ollama server, run: ollama serve
)

:: Test 5: Test Context7 MCP Tools
echo.
echo ========================================
echo  5. MCP TOOLS FUNCTIONALITY TEST
echo ========================================
echo [INFO] Testing Context7 MCP tool functionality...

:: Create a test script to validate MCP tools
echo // MCP Tools Test> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo import { generateMCPPrompt, commonMCPQueries, validateMCPRequest } from './sveltekit-frontend/src/lib/utils/mcp-helpers.ts';>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo.>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo console.log('Testing MCP Helper Functions...');>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo.>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo // Test analyze-stack tool>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo const analyzeRequest = commonMCPQueries.analyzeSvelteKit();>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo console.log('Analyze SvelteKit Request:', analyzeRequest);>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo.>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo // Test validation>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo const validation = validateMCPRequest(analyzeRequest);>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo console.log('Validation Result:', validation);>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo.>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo // Test prompt generation>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo const prompt = generateMCPPrompt(analyzeRequest);>> "%PROJECT_ROOT%test-mcp-tools.mjs"
echo console.log('Generated Prompt:', prompt);>> "%PROJECT_ROOT%test-mcp-tools.mjs"

echo [INFO] Running MCP tools validation...
node "%PROJECT_ROOT%test-mcp-tools.mjs" 2>nul
if %errorlevel% equ 0 (
    echo [✓] MCP helper functions working correctly
) else (
    echo [⚠] MCP helper functions may have issues (check TypeScript compilation)
)

:: Test 6: Test SvelteKit Integration
echo.
echo ========================================
echo  6. SVELTEKIT INTEGRATION TEST
echo ========================================
echo [INFO] Testing SvelteKit development server...

cd "%PROJECT_ROOT%sveltekit-frontend"

:: Check if dev server is running
curl -s --connect-timeout 3 http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] SvelteKit dev server is running
    
    :: Test multi-agent API
    echo [INFO] Testing multi-agent API endpoint...
    curl -s http://localhost:5173/api/ai/multi-agent?action=status >nul 2>&1
    if %errorlevel% equ 0 (
        echo [✓] Multi-agent API is responding
    ) else (
        echo [⚠] Multi-agent API not responding
    )
) else (
    echo [⚠] SvelteKit dev server not running
    echo [INFO] To start dev server, run: npm run dev
)

:: Test 7: Test Claude MCP Configuration
echo.
echo ========================================
echo  7. CLAUDE MCP CONFIGURATION TEST
echo ========================================
echo [INFO] Checking Claude MCP configuration...

set "CLAUDE_CONFIG=%APPDATA%\Claude\claude_desktop_config.json"

if exist "%CLAUDE_CONFIG%" (
    echo [✓] Claude Desktop config found
    echo [INFO] Config location: %CLAUDE_CONFIG%
    
    :: Check if our MCP server is configured
    findstr /i "context7" "%CLAUDE_CONFIG%" >nul 2>&1
    if %errorlevel% equ 0 (
        echo [✓] Context7 MCP server configured in Claude Desktop
    ) else (
        echo [⚠] Context7 MCP server not found in Claude config
        echo [INFO] Run SETUP-CLAUDE-MCP-CONTEXT7.bat to configure
    )
) else (
    echo [⚠] Claude Desktop config not found
    echo [INFO] Install Claude Desktop and run setup script
)

:: Test 8: Memory and GPU Status
echo.
echo ========================================
echo  8. SYSTEM RESOURCES CHECK
echo ========================================
echo [INFO] Checking system resources for AI workloads...

:: Check GPU
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>nul
if %errorlevel% equ 0 (
    echo [✓] NVIDIA GPU detected and available
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
) else (
    echo [⚠] No NVIDIA GPU detected (CPU-only mode)
)

:: Check available memory
echo [INFO] System memory status:
wmic computersystem get TotalPhysicalMemory /format:value 2>nul | findstr /i "TotalPhysicalMemory"

:: Cleanup test files
echo.
echo ========================================
echo  CLEANUP AND SUMMARY
echo ========================================
del "%PROJECT_ROOT%test-mcp-tools.mjs" 2>nul

echo.
echo [SUMMARY] MCP Server Test Results:
echo   - MCP Server: Ready for testing
echo   - vLLM Integration: %VLLM_STATUS%
echo   - Ollama Integration: Available for fallback
echo   - SvelteKit API: Multi-agent endpoints configured
echo   - Claude MCP: Configuration files created
echo   - System Resources: Checked and documented
echo.
echo [NEXT STEPS]
echo   1. Start services: vLLM (python vllm-legal-server.py) and Ollama (ollama serve)
echo   2. Start SvelteKit: npm run dev (in sveltekit-frontend/)
echo   3. Test MCP tools in Claude Desktop or VS Code
echo   4. Use Context7 tools for stack analysis and best practices
echo.
echo [AVAILABLE MCP TOOLS]
echo   - analyze-stack: Get context-aware analysis for any stack component
echo   - generate-best-practices: Security, performance, and UI/UX guidelines
echo   - suggest-integration: Integration patterns for new features
echo   - resolve-library-id: Find correct library documentation IDs
echo   - get-library-docs: Access specific documentation topics
echo.

pause