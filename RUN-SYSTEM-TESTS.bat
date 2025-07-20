@echo off
cls
echo ==========================================
echo     System Test Suite
echo ==========================================
echo.
echo Running comprehensive system tests...
echo This will verify:
echo   - Docker services
echo   - Database connections
echo   - Ollama LLM integration
echo   - TypeScript compilation
echo   - All dependencies
echo.

node test-system-complete.mjs

echo.
echo ==========================================
echo Test suite completed!
echo Check system-test-report.json for details
echo ==========================================
echo.
pause
