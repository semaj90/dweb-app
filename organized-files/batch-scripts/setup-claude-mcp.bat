@echo off
echo Setting up Claude Desktop MCP Context7 Configuration...
echo.

REM Create Claude config directory if it doesn't exist
if not exist "%APPDATA%\Claude" (
    echo Creating Claude config directory...
    mkdir "%APPDATA%\Claude"
)

REM Copy the configuration file
echo Copying Claude Desktop configuration...
copy "claude_desktop_config.json" "%APPDATA%\Claude\claude_desktop_config.json"

if %errorlevel% equ 0 (
    echo ✅ Configuration copied successfully!
    echo.
    echo Configuration location: %APPDATA%\Claude\claude_desktop_config.json
) else (
    echo ❌ Failed to copy configuration
    exit /b 1
)

echo.
echo Testing Node.js environment...
node --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Node.js is available
    for /f "tokens=*" %%i in ('node --version') do echo    Version: %%i
) else (
    echo ❌ Node.js not found in PATH
    echo Please ensure Node.js is installed and in your PATH
)

echo.
echo Testing npm...
npm --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ npm is available
    for /f "tokens=*" %%i in ('npm --version') do echo    Version: %%i
) else (
    echo ❌ npm not found in PATH
)

echo.
echo Testing npx...
npx --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ npx is available
    for /f "tokens=*" %%i in ('npx --version') do echo    Version: %%i
) else (
    echo ❌ npx not found in PATH
)

echo.
echo Testing Context7 server...
echo This may take a moment as it downloads the package...
npx --yes @modelcontextprotocol/server-context7 --help >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Context7 server is accessible
) else (
    echo ❌ Context7 server test failed
    echo This might be due to network issues or Node.js environment problems
)

echo.
echo Setup complete! Please:
echo 1. Restart Claude Desktop completely
echo 2. Check if Context7 appears in your available tools
echo 3. Test with a question about your project structure
echo.
pause
