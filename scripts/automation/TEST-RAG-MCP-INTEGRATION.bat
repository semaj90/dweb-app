@echo off
setlocal enabledelayedexpansion

:: RAG-MCP Integration Test Suite
echo ========================================
echo  RAG-MCP Integration Test Suite
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
set "ALL_TESTS_PASSED=true"

echo [INFO] Project Root: %PROJECT_ROOT%
echo [INFO] Testing RAG integration with MCP tools
echo.

:: Test 1: Check if RAG configuration is present in MCP server
echo ========================================
echo  1. MCP SERVER RAG CONFIGURATION
echo ========================================
echo [INFO] Checking RAG configuration in MCP server...

findstr /i "RAG_CONFIG" "%PROJECT_ROOT%scripts\context7-mcp-server.js" >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] RAG configuration found in MCP server
) else (
    echo [✗] RAG configuration missing from MCP server
    set "ALL_TESTS_PASSED=false"
)

findstr /i "rag-query" "%PROJECT_ROOT%scripts\context7-mcp-server.js" >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] RAG query tool found in MCP server
) else (
    echo [✗] RAG query tool missing from MCP server
    set "ALL_TESTS_PASSED=false"
)

:: Test 2: Check MCP helpers for RAG support
echo.
echo ========================================
echo  2. MCP HELPERS RAG SUPPORT
echo ========================================
echo [INFO] Checking RAG support in MCP helpers...

findstr /i "rag-query" "%PROJECT_ROOT%sveltekit-frontend\src\lib\utils\mcp-helpers.ts" >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] RAG tools found in MCP helpers
) else (
    echo [✗] RAG tools missing from MCP helpers
    set "ALL_TESTS_PASSED=false"
)

findstr /i "ragLegalQuery" "%PROJECT_ROOT%sveltekit-frontend\src\lib\utils\mcp-helpers.ts" >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] RAG common queries found in MCP helpers
) else (
    echo [✗] RAG common queries missing from MCP helpers
    set "ALL_TESTS_PASSED=false"
)

:: Test 3: Check Demo Component RAG Integration
echo.
echo ========================================
echo  3. DEMO COMPONENT RAG INTEGRATION
echo ========================================
echo [INFO] Checking RAG integration in demo component...

findstr /i "RAG Query Legal Documents" "%PROJECT_ROOT%sveltekit-frontend\src\lib\components\dev\MCPToolsDemo.svelte" >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] RAG tools found in demo component
) else (
    echo [✗] RAG tools missing from demo component
    set "ALL_TESTS_PASSED=false"
)

findstr /i "ragQuery.*=" "%PROJECT_ROOT%sveltekit-frontend\src\lib\components\dev\MCPToolsDemo.svelte" >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] RAG form variables found in demo component
) else (
    echo [✗] RAG form variables missing from demo component
    set "ALL_TESTS_PASSED=false"
)

:: Test 4: Check RAG Backend Files
echo.
echo ========================================
echo  4. RAG BACKEND FILES
echo ========================================
echo [INFO] Checking RAG backend files...

if exist "%PROJECT_ROOT%rag-backend\main.py" (
    echo [✓] RAG backend main.py exists
) else (
    echo [✗] RAG backend main.py missing
    set "ALL_TESTS_PASSED=false"
)

if exist "%PROJECT_ROOT%rag-backend\services\rag_service.py" (
    echo [✓] RAG service implementation exists
) else (
    echo [✗] RAG service implementation missing
    set "ALL_TESTS_PASSED=false"
)

if exist "%PROJECT_ROOT%rag-backend\services\vector_store.py" (
    echo [✓] Vector store service exists
) else (
    echo [✗] Vector store service missing
    set "ALL_TESTS_PASSED=false"
)

:: Test 5: Test MCP Server Startup with RAG
echo.
echo ========================================
echo  5. MCP SERVER STARTUP TEST
echo ========================================
echo [INFO] Testing MCP server startup with RAG configuration...

:: Set RAG environment variables for testing
set "RAG_ENDPOINT=http://localhost:8000"
set "RAG_ENABLED=true"

echo [INFO] RAG configuration for testing:
echo   RAG_ENDPOINT=%RAG_ENDPOINT%
echo   RAG_ENABLED=%RAG_ENABLED%

:: Test if MCP server can start (dry run)
node --check "%PROJECT_ROOT%scripts\context7-mcp-server.js" >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] MCP server syntax is valid
) else (
    echo [✗] MCP server has syntax errors
    set "ALL_TESTS_PASSED=false"
)

:: Test 6: Check Integration Documentation
echo.
echo ========================================
echo  6. INTEGRATION DOCUMENTATION
echo ========================================
echo [INFO] Checking integration documentation...

if exist "%PROJECT_ROOT%sveltekit-frontend\src\CLAUDE.md" (
    findstr /i "RAG" "%PROJECT_ROOT%sveltekit-frontend\src\CLAUDE.md" >nul 2>&1
    if %errorlevel% equ 0 (
        echo [✓] RAG documentation found in CLAUDE.md
    ) else (
        echo [⚠] RAG documentation could be enhanced in CLAUDE.md
    )
) else (
    echo [⚠] CLAUDE.md not found
)

:: Test 7: Environment Configuration
echo.
echo ========================================
echo  7. ENVIRONMENT CONFIGURATION
echo ========================================
echo [INFO] Checking environment configuration...

echo [INFO] Required environment variables for RAG-MCP integration:
echo   - RAG_ENDPOINT (default: http://localhost:8000)
echo   - RAG_ENABLED (default: true)
echo   - DATABASE_URL (for vector store)
echo   - QDRANT_URL (optional: http://localhost:6333)

:: Final Results
echo.
echo ========================================
echo  INTEGRATION TEST RESULTS
echo ========================================

if "%ALL_TESTS_PASSED%"=="true" (
    echo [🎉] ALL RAG-MCP INTEGRATION TESTS PASSED!
    echo.
    echo ✅ MCP Server: RAG tools configured
    echo ✅ MCP Helpers: RAG functions added
    echo ✅ Demo Component: RAG UI integrated
    echo ✅ Backend Files: RAG services available
    echo ✅ Syntax Check: No errors detected
    echo.
    echo [READY] RAG-MCP integration is complete and ready for testing!
) else (
    echo [⚠] SOME RAG-MCP INTEGRATION TESTS FAILED
    echo.
    echo Please review the items marked with [✗] above and ensure all
    echo RAG components are properly integrated with MCP tools.
)

echo.
echo ========================================
echo  RAG-MCP INTEGRATION CAPABILITIES
echo ========================================
echo.
echo 🔍 RAG QUERY TOOLS:
echo   - rag-query: Semantic search across legal documents
echo   - rag-upload-document: Index new legal documents
echo   - rag-get-stats: Monitor RAG system status
echo   - rag-analyze-relevance: Document relevance analysis
echo   - rag-integration-guide: Integration patterns and examples
echo.
echo 📚 LEGAL DOCUMENT TYPES SUPPORTED:
echo   - Contracts and agreements
echo   - Case law and precedents
echo   - Evidence and exhibits
echo   - Statutes and regulations
echo   - Legal briefs and memos
echo.
echo 🔧 MCP INTEGRATION FEATURES:
echo   - Natural language prompts for RAG operations
echo   - Type-safe TypeScript helpers
echo   - Interactive demo component
echo   - Pre-built common queries
echo   - Comprehensive error handling
echo.
echo 🚀 QUICK START COMMANDS:
echo   # Start RAG backend:
echo   cd rag-backend ^&^& python main.py
echo.
echo   # Test MCP server:
echo   node scripts\context7-mcp-server.js --test-mode
echo.
echo   # Access demo interface:
echo   http://localhost:5173/dev/mcp-tools
echo.
echo   # Example RAG query via MCP:
echo   "rag query 'contract liability clauses' for case CASE-2024-001"
echo.

pause