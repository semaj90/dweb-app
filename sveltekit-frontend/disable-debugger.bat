@echo off
echo Disabling Node.js debugger environment...
set NODE_OPTIONS=
set NODE_INSPECT=
set NODE_DEBUG=
set DEBUG=

echo ✅ Environment variables cleared
echo ✅ Debugger disabled for current session
echo.
echo Now you can run:
echo   npm run check:ultra-fast
echo   npm run check:full
echo.
echo Environment is ready!