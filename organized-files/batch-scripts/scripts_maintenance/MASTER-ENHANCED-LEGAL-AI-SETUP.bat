@echo off
setlocal enabledelayedexpansion

title Enhanced Legal AI - Phase 3 Setup

echo ========================================
echo ENHANCED LEGAL AI - PHASE 3 SETUP
echo YoRHa Aesthetic + Vector Search + AI
echo ========================================
echo.

:: Set colors
color 0A

:: Get current directory
set "PROJECT_ROOT=%~dp0"
set "FRONTEND_PATH=%PROJECT_ROOT%sveltekit-frontend"

echo Project Root: %PROJECT_ROOT%
echo Frontend Path: %FRONTEND_PATH%
echo.

:: Check if frontend directory exists
if not exist "%FRONTEND_PATH%" (
    echo ERROR: Frontend directory not found!
    echo Expected: %FRONTEND_PATH%
    echo.
    echo Please ensure you're running this from the correct directory.
    pause
    exit /b 1
)

:: Main menu
:MAIN_MENU
cls
echo ========================================
echo ENHANCED LEGAL AI - PHASE 3 SETUP
echo ========================================
echo.
echo Choose setup option:
echo.
echo 1. Quick Start (All services)
echo 2. Development Setup Only
echo 3. Docker Services Only  
echo 4. Ollama + AI Models Only
echo 5. Database Setup Only
echo 6. Health Check
echo 7. Start Development Server
echo 8. View Documentation
echo 9. Exit
echo.
set /p "choice=Enter your choice (1-9): "

if "%choice%"=="1" goto QUICK_START
if "%choice%"=="2" goto DEV_SETUP
if "%choice%"=="3" goto DOCKER_ONLY
if "%choice%"=="4" goto OLLAMA_ONLY
if "%choice%"=="5" goto DATABASE_ONLY
if "%choice%"=="6" goto HEALTH_CHECK
if "%choice%"=="7" goto START_DEV
if "%choice%"=="8" goto DOCUMENTATION
if "%choice%"=="9" goto EXIT

echo Invalid choice. Please try again.
timeout /t 2 > nul
goto MAIN_MENU

:QUICK_START
echo.
echo === QUICK START - ALL SERVICES ===
echo.
echo This will set up everything:
echo - Docker services (PostgreSQL, Redis, Qdrant)
echo - Ollama with AI models
echo - Database schema and vector collections
echo - Development environment
echo.
pause

echo.
echo Starting comprehensive setup...
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%setup-enhanced-legal-ai-v3.ps1" -StartDocker -SetupDatabase -StartOllama -LoadModels -StartDev

goto END

:DEV_SETUP
echo.
echo === DEVELOPMENT SETUP ===
echo.
echo Setting up development environment only...
cd /d "%FRONTEND_PATH%"

echo Installing dependencies...
call npm install

echo Setting up TypeScript configuration...
if exist "tsconfig.json" (
    echo TypeScript config found.
) else (
    echo Creating TypeScript configuration...
    echo {} > tsconfig.json
)

echo Checking for environment file...
if not exist ".env" (
    echo Creating development environment file...
    echo DATABASE_URL="postgresql://postgres:password123@localhost:5432/prosecutor_db" > .env
    echo REDIS_HOST=localhost >> .env
    echo REDIS_PORT=6379 >> .env
    echo QDRANT_URL=http://localhost:6333 >> .env
    echo OLLAMA_URL=http://localhost:11434 >> .env
    echo NODE_ENV=development >> .env
)

echo.
echo Development setup complete!
echo.
echo To start development server: npm run dev
echo To start with AI services: npm run dev:with-llm
echo.
pause
goto MAIN_MENU

:DOCKER_ONLY
echo.
echo === DOCKER SERVICES SETUP ===
echo.
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%setup-enhanced-legal-ai-v3.ps1" -StartDocker
echo.
pause
goto MAIN_MENU

:OLLAMA_ONLY
echo.
echo === OLLAMA + AI MODELS SETUP ===
echo.
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%setup-enhanced-legal-ai-v3.ps1" -StartOllama -LoadModels
echo.
pause
goto MAIN_MENU

:DATABASE_ONLY
echo.
echo === DATABASE SETUP ===
echo.
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%setup-enhanced-legal-ai-v3.ps1" -SetupDatabase
echo.
pause
goto MAIN_MENU

:HEALTH_CHECK
echo.
echo === SYSTEM HEALTH CHECK ===
echo.
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%setup-enhanced-legal-ai-v3.ps1"
echo.
pause
goto MAIN_MENU

:START_DEV
echo.
echo === STARTING DEVELOPMENT SERVER ===
echo.
cd /d "%FRONTEND_PATH%"

echo Checking dependencies...
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)

echo Starting development server...
echo.
echo Opening http://localhost:5173
echo Press Ctrl+C to stop the server
echo.
start http://localhost:5173
call npm run dev

goto END

:DOCUMENTATION
echo.
echo === ENHANCED LEGAL AI DOCUMENTATION ===
echo.
echo ========================================
echo FEATURES OVERVIEW
echo ========================================
echo.
echo 1. YoRHa Terminal Aesthetic:
echo    - Pure black/white/gray color scheme
echo    - Monospace typography throughout
echo    - Clean geometric styling
echo    - Terminal-inspired interface elements
echo.
echo 2. Multi-Step Forms (XState v5):
echo    - Case entry forms
echo    - Evidence cataloging
echo    - Criminal profile creation
echo    - Real-time validation with Zod
echo.
echo 3. Vector Search Integration:
echo    - PostgreSQL with pgvector extension
echo    - Qdrant vector database
echo    - Redis caching layer
echo    - Semantic search capabilities
echo.
echo 4. AI Integration:
echo    - Ollama with Gemma 3 model
echo    - Legal-enhanced model with custom prompts
echo    - Streaming chat interface
echo    - Case analysis and evidence review
echo.
echo 5. Interactive Evidence Board:
echo    - Drag-and-drop evidence items
echo    - Connection visualization
echo    - Evidence type classification
echo    - Real-time collaboration
echo.
echo ========================================
echo TECHNICAL STACK
echo ========================================
echo.
echo Frontend:
echo - SvelteKit 2.0 with TypeScript
echo - UnoCSS for styling
echo - XState v5 for state management
echo - Superforms for form handling
echo.
echo Backend:
echo - PostgreSQL with pgvector
echo - Drizzle ORM
echo - Redis for caching
echo - Qdrant for vector search
echo.
echo AI/ML:
echo - Ollama local AI runtime
echo - Gemma 3 7B model
echo - Nomic Embed for embeddings
echo - Custom legal prompts
echo.
echo ========================================
echo API ENDPOINTS
echo ========================================
echo.
echo /api/chat             - AI chat interface
echo /api/vector/search    - Vector similarity search  
echo /api/embeddings       - Generate embeddings
echo.
echo ========================================
echo DEVELOPMENT COMMANDS
echo ========================================
echo.
echo npm run dev           - Start development server
echo npm run dev:with-llm  - Start with AI services
echo npm run db:studio     - Open database studio
echo npm run type-check    - TypeScript validation
echo npm run test          - Run test suite
echo.
echo ========================================
echo TROUBLESHOOTING
echo ========================================
echo.
echo Common Issues:
echo.
echo 1. Docker services not starting:
echo    - Ensure Docker Desktop is running
echo    - Check port availability (5432, 6379, 6333)
echo    - Run: docker-compose down; docker-compose up -d
echo.
echo 2. Ollama not responding:
echo    - Verify Ollama is installed
echo    - Check if service is running on port 11434
echo    - Restart: ollama serve
echo.
echo 3. Database connection errors:
echo    - Verify PostgreSQL is running
echo    - Check .env file configuration
echo    - Run: npm run db:push
echo.
echo 4. Vector search issues:
echo    - Ensure Qdrant is accessible
echo    - Check collection initialization
echo    - Verify embedding model is available
echo.
echo 5. TypeScript errors:
echo    - Run: npm run type-check
echo    - Update dependencies: npm update
echo    - Clear cache: npm run clean
echo.
echo ========================================
echo.
pause
goto MAIN_MENU

:EXIT
echo.
echo Thank you for using Enhanced Legal AI!
echo.
echo If you need help, check the documentation or run health checks.
echo.
timeout /t 3 > nul
exit /b 0

:END
echo.
echo Setup process completed.
echo.
pause
goto MAIN_MENU

endlocal