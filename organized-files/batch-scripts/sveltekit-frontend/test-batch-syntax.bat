@echo off
echo Testing batch file syntax...
echo.

REM Test if node_modules check works
if not exist "node_modules" (
    echo node_modules not found - this is expected behavior
) else (
    echo node_modules found
)

REM Test curl availability
curl --version >nul 2>&1
if errorlevel 1 (
    echo curl not available - tests will need manual confirmation
) else (
    echo curl is available
)

REM Test node availability
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found
    exit /b 1
) else (
    echo Node.js is available
)

echo.
echo Batch file syntax test completed successfully!
echo.