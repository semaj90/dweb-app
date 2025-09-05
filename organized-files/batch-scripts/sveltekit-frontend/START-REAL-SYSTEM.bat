@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo   LEGAL AI SYSTEM - PRODUCTION STARTUP
echo ========================================
echo.

REM Set working directory
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

REM Color codes for output
set GREEN=[32m
set RED=[31m
set YELLOW=[33m
set BLUE=[34m
set NC=[0m

echo %BLUE%Checking system requirements...%NC%
echo.

REM Check if Node.js is available
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Node.js not found. Please install Node.js first.%NC%
    pause
    exit /b 1
)
echo %GREEN%✅ Node.js found%NC%

REM Check if npm is available
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ npm not found.%NC%
    pause
    exit /b 1
)
echo %GREEN%✅ npm found%NC%

REM Check if PostgreSQL is running
netstat -an | findstr ":5432" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  PostgreSQL not detected on port 5432%NC%
    echo Starting PostgreSQL...
    net start postgresql-x64-15 >nul 2>&1
    timeout /t 3 >nul
)

REM Check PostgreSQL again
netstat -an | findstr ":5432" >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ PostgreSQL running%NC%
) else (
    echo %RED%❌ PostgreSQL not running. Please start PostgreSQL manually.%NC%
)

REM Check if Redis is running
netstat -an | findstr ":6379" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Redis not detected on port 6379%NC%
    echo Starting Redis...
    start /min redis-server >nul 2>&1
    timeout /t 2 >nul
)

REM Check Redis again
netstat -an | findstr ":6379" >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Redis running%NC%
) else (
    echo %YELLOW%⚠️  Redis not running. Some features may be limited.%NC%
)

REM Check if Ollama is running
netstat -an | findstr ":11434" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Ollama not detected on port 11434%NC%
    echo Starting Ollama...
    start /min ollama serve >nul 2>&1
    timeout /t 3 >nul
)

REM Check Ollama again
netstat -an | findstr ":11434" >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Ollama running%NC%
) else (
    echo %YELLOW%⚠️  Ollama not running. Embeddings will use fallback.%NC%
)

echo.
echo %BLUE%Setting up database...%NC%

REM Install dependencies if needed
if not exist "node_modules" (
    echo Installing dependencies...
    npm install --silent
)

REM Setup database
echo Running database setup...
node scripts/setup-database.mjs
if %errorlevel% neq 0 (
    echo %RED%❌ Database setup failed%NC%
    pause
    exit /b 1
)

echo %GREEN%✅ Database setup completed%NC%
echo.

REM Check if embedding model exists
echo %BLUE%Checking Ollama models...%NC%
ollama list | findstr "nomic-embed-text" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  nomic-embed-text model not found%NC%
    echo Pulling embedding model...
    ollama pull nomic-embed-text
    if %errorlevel% neq 0 (
        echo %YELLOW%⚠️  Failed to pull model. Fallback embeddings will be used.%NC%
    ) else (
        echo %GREEN%✅ Embedding model ready%NC%
    )
) else (
    echo %GREEN%✅ nomic-embed-text model found%NC%
)

echo.
echo %BLUE%Starting Legal AI System...%NC%
echo.

REM Copy environment configuration
if exist ".env.production" (
    copy ".env.production" ".env" >nul 2>&1
    echo %GREEN%✅ Environment configuration loaded%NC%
)

echo Starting development server...
echo.
echo %GREEN%🚀 Legal AI System is starting!%NC%
echo.
echo %BLUE%Available endpoints:%NC%
echo   Frontend: http://localhost:5173/ai-upload-demo
echo   OCR API: http://localhost:5173/api/ocr/langextract
echo   Embeddings: http://localhost:5173/api/embeddings/generate
echo   Search: http://localhost:5173/api/documents/search
echo.
echo %BLUE%System Features:%NC%
echo   ✅ Real OCR with Tesseract.js
echo   ✅ Real embeddings with Ollama
echo   ✅ PostgreSQL + pgvector storage
echo   ✅ Redis caching
echo   ✅ Legal document analysis
echo   ✅ Semantic search
echo.
echo %YELLOW%Press Ctrl+C to stop the system%NC%
echo.

REM Start the development server
npm run dev

echo.
echo %BLUE%System stopped.%NC%
pause
