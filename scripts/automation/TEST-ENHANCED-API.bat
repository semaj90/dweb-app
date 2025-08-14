@echo off
echo ================================================
echo Enhanced REST API Test Suite
echo ================================================
echo.

echo Step 1: Setting up test data...
node setup-test-data.mjs
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Test data setup failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Starting SvelteKit dev server (if not running)...
echo 📌 Make sure your SvelteKit dev server is running on http://localhost:5173
echo    Run: cd sveltekit-frontend && npm run dev
echo.
pause

echo Step 3: Running simple API tests...
node test-api-simple.mjs
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️ Some simple tests failed, but continuing...
)

echo.
echo Step 4: Running comprehensive API tests...
node test-enhanced-rest-api.mjs
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️ Some comprehensive tests failed
)

echo.
echo ================================================
echo Test Results Summary
echo ================================================
echo.

if exist enhanced-rest-api-test-report.json (
    echo 📊 Detailed test report saved to: enhanced-rest-api-test-report.json
    echo.
    echo 🔍 Quick Summary:
    findstr "passed\|failed\|successRate" enhanced-rest-api-test-report.json
) else (
    echo ⚠️ No detailed test report generated
)

echo.
echo ================================================
echo Next Steps
echo ================================================
echo 1. Review test results above
echo 2. Check enhanced-rest-api-test-report.json for details
echo 3. Fix any failing endpoints if needed
echo 4. Run clustering operations through the API
echo.
pause