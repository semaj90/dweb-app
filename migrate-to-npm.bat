@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo   Legal AI Assistant - Migration to NPM Scripts
echo =====================================================
echo.
echo This script helps transition from .bat files to modern
echo cross-platform npm script workflows with Gemma3 Legal AI.
echo.

:: Color codes
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

:: Check Node.js availability
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Node.js not found%NC%
    echo.
    echo %YELLOW%To use npm scripts, please install Node.js first:%NC%
    echo %YELLOW%1. Download from https://nodejs.org/%NC%
    echo %YELLOW%2. Install LTS version%NC%
    echo %YELLOW%3. Restart command prompt%NC%
    echo %YELLOW%4. Run this script again%NC%
    echo.
    echo %BLUE%For now, you can continue using .bat files:%NC%
    echo %YELLOW%• quick-start.bat - Quick development start%NC%
    echo %YELLOW%• setup-complete-with-ollama.bat - Full setup%NC%
    echo %YELLOW%• service-manager.bat - Service management%NC%
    echo %YELLOW%• check-setup.bat - Health diagnostics%NC%
    pause
    exit /b 1
)

npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ npm not found%NC%
    pause
    exit /b 1
)

echo %GREEN%✅ Node.js and npm are available%NC%
echo.

:: Install npm dependencies
echo %BLUE%📦 Installing npm dependencies...%NC%
if not exist "node_modules" (
    npm install
    if %errorlevel% neq 0 (
        echo %RED%❌ Failed to install dependencies%NC%
        echo %YELLOW%Try running: npm install --legacy-peer-deps%NC%
        pause
        exit /b 1
    )
    echo %GREEN%✅ Dependencies installed%NC%
) else (
    echo %GREEN%✅ Dependencies already installed%NC%
)

:: Show migration guide
echo.
echo %BLUE%🔄 Migration Guide - OLD .bat files → NEW npm scripts:%NC%
echo.
echo %YELLOW%├── setup-complete-with-ollama.bat  →  npm run setup%NC%
echo %YELLOW%├── quick-start.bat                 →  npm run dev%NC%
echo %YELLOW%├── check-setup.bat                 →  npm run health%NC%
echo %YELLOW%├── service-manager.bat             →  Multiple npm commands%NC%
echo %YELLOW%└── AI model: Gemma3-Legal specialized for legal document analysis%NC%
echo.

echo %BLUE%📋 Available npm commands:%NC%
echo.
echo %GREEN%Development Commands:%NC%
echo %YELLOW%  npm run setup      - Complete environment setup (replaces setup-complete-with-ollama.bat)%NC%
echo %YELLOW%  npm run dev        - Start development environment (replaces quick-start.bat)%NC%
echo %YELLOW%  npm run build      - Build production application%NC%
echo %YELLOW%  npm run preview    - Preview production build%NC%
echo.
echo %GREEN%Docker & AI Commands:%NC%
echo %YELLOW%  npm run docker:up  - Start all Docker services%NC%
echo %YELLOW%  npm run docker:down- Stop all Docker services%NC%
echo %YELLOW%  npm run ai:start   - Start AI services only (Postgres, Ollama, Qdrant)%NC%
echo %YELLOW%  npm run ai:stop    - Stop AI services%NC%
echo %YELLOW%  npm run ai:test    - Test Gemma3 Legal AI model%NC%
echo.
echo %GREEN%Monitoring & Maintenance:%NC%
echo %YELLOW%  npm run health     - System health check (replaces check-setup.bat)%NC%
echo %YELLOW%  npm run monitor    - Resource monitoring%NC%
echo %YELLOW%  npm run docker:logs- View service logs%NC%
echo %YELLOW%  npm run docker:status - Show container status%NC%
echo.
echo %GREEN%Database Commands:%NC%
echo %YELLOW%  npm run db:migrate - Apply database migrations%NC%
echo %YELLOW%  npm run db:studio  - Open database studio%NC%
echo.

:: Test npm functionality
echo %BLUE%🧪 Testing npm script functionality...%NC%
echo.

echo %BLUE%Testing health check...%NC%
npm run health >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Health check script needs setup completion%NC%
) else (
    echo %GREEN%✅ Health check script working%NC%
)

echo %BLUE%Testing Docker commands...%NC%
npm run docker:status >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Docker commands need Docker Desktop running%NC%
) else (
    echo %GREEN%✅ Docker commands working%NC%
)

:: Show recommended workflow
echo.
echo %BLUE%🚀 Recommended Development Workflow:%NC%
echo.
echo %YELLOW%1. Initial Setup (one time):%NC%
echo %YELLOW%   npm run setup%NC%
echo.
echo %YELLOW%2. Daily Development:%NC%
echo %YELLOW%   npm run dev%NC%
echo.
echo %YELLOW%3. Health Monitoring:%NC%
echo %YELLOW%   npm run health%NC%
echo %YELLOW%   npm run monitor%NC%
echo.
echo %YELLOW%4. AI Model Testing:%NC%
echo %YELLOW%   npm run ai:test%NC%
echo.

:: Show Gemma3 Legal model information
echo %BLUE%🤖 Gemma3 Legal AI Model Features:%NC%
echo.
echo %GREEN%✓ Specialized for legal document analysis%NC%
echo %GREEN%✓ Compliance checking and risk assessment%NC%
echo %GREEN%✓ Legal entity extraction (cases, citations, statutes)%NC%
echo %GREEN%✓ Contract clause analysis%NC%
echo %GREEN%✓ Audit trail generation for legal compliance%NC%
echo %GREEN%✓ Optimized for 6GB memory allocation%NC%
echo.
echo %YELLOW%Model fallback: If gemma3-legal is not available, system uses gemma2:9b%NC%
echo.

:: Migration completion
echo %BLUE%✨ Migration Benefits:%NC%
echo.
echo %GREEN%✓ Cross-platform compatibility (Windows, macOS, Linux)%NC%
echo %GREEN%✓ Simplified command structure%NC%
echo %GREEN%✓ Better error handling and logging%NC%
echo %GREEN%✓ Integrated with modern development tools%NC%
echo %GREEN%✓ Consistent with industry standards%NC%
echo %GREEN%✓ Easier CI/CD integration%NC%
echo.

echo %BLUE%💡 Pro Tips:%NC%
echo %YELLOW%• Use 'npm run' to see all available commands%NC%
echo %YELLOW%• Add '--silent' flag to reduce npm output%NC%
echo %YELLOW%• Use 'npm run dev' for parallel frontend + backend development%NC%
echo %YELLOW%• Check 'npm run health' before starting development%NC%
echo.

echo %BLUE%📝 Next Steps:%NC%
echo %YELLOW%1. Try: npm run setup (if not already done)%NC%
echo %YELLOW%2. Then: npm run dev%NC%
echo %YELLOW%3. Test: npm run ai:test%NC%
echo %YELLOW%4. Monitor: npm run health%NC%
echo.

echo %GREEN%🎉 You're ready to use modern npm scripts for legal AI development!%NC%
echo %BLUE%The .bat files remain available for Windows-specific tasks if needed.%NC%
echo.
pause
