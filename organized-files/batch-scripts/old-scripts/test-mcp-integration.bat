@echo off
echo ===========================================
echo  MCP SERVER INTEGRATION TEST SUITE
echo ===========================================
echo.

echo [1/4] Running npm run check for TypeScript errors...
cd sveltekit-frontend
call npm run check
if %errorlevel% neq 0 (
    echo ERROR: TypeScript check failed!
    echo Check the output above for specific errors.
    echo.
) else (
    echo SUCCESS: TypeScript check passed!
    echo.
)

echo [2/4] Running database migration check...
cd ..
call npm run db:check
if %errorlevel% neq 0 (
    echo WARNING: Database check may have issues
    echo.
) else (
    echo SUCCESS: Database is accessible!
    echo.
)

echo [3/4] Starting development server test...
echo Starting dev server in the background for 10 seconds...
cd sveltekit-frontend
start /b npm run dev
timeout /t 10 /nobreak >nul
echo Dev server test complete (check http://localhost:5173)
echo.

echo [4/4] Running database push/migration...
cd ..
call npm run db:migrate
if %errorlevel% neq 0 (
    echo ERROR: Database migration failed!
    echo.
) else (
    echo SUCCESS: Database migration completed!
    echo.
)

echo ===========================================
echo  TEST SUITE COMPLETE
echo ===========================================
echo Check the output above for any errors.
echo If all tests passed, your MCP server setup is ready!
echo.
pause
