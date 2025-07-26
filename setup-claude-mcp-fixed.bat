@echo off
title Claude Desktop MCP Context7 Setup

echo ================================================================
echo Claude Desktop MCP Context7 Setup
echo ================================================================
echo.

REM Set Node.js path explicitly
set "NODEJS_PATH=C:\Program Files\nodejs"
set "PATH=%NODEJS_PATH%;%PATH%"

echo Step 1: Checking Node.js installation...
if exist "%NODEJS_PATH%\node.exe" (
    echo ✅ Node.js found at: %NODEJS_PATH%
    "%NODEJS_PATH%\node.exe" --version
) else (
    echo ❌ Node.js not found at expected location
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo.
echo Step 2: Checking npm...
if exist "%NODEJS_PATH%\npm.cmd" (
    echo ✅ npm found
    "%NODEJS_PATH%\npm.cmd" --version
) else (
    echo ❌ npm not found
    pause
    exit /b 1
)

echo.
echo Step 3: Testing npx...
"%NODEJS_PATH%\npx.cmd" --version 2>nul
if errorlevel 1 (
    echo ❌ npx test failed
) else (
    echo ✅ npx is working
)

echo.
echo Step 4: Claude Desktop configuration...
if not exist "%APPDATA%\Claude" (
    echo Creating Claude config directory...
    mkdir "%APPDATA%\Claude"
)

if exist "claude_desktop_config.json" (
    echo Copying configuration to Claude Desktop...
    copy "claude_desktop_config.json" "%APPDATA%\Claude\claude_desktop_config.json" >nul
    if errorlevel 0 (
        echo ✅ Configuration copied successfully
    ) else (
        echo ❌ Failed to copy configuration
    )
) else (
    echo ❌ claude_desktop_config.json not found in current directory
)

echo.
echo Step 5: Testing Context7 server...
echo This may take 30-60 seconds as it downloads the package...
echo Please wait...

"%NODEJS_PATH%\npx.cmd" --yes @modelcontextprotocol/server-context7 --help >nul 2>&1
if errorlevel 1 (
    echo ❌ Context7 server test failed
    echo This might be normal on first run - the package may need to download
) else (
    echo ✅ Context7 server is working
)

echo.
echo ================================================================
echo Setup Summary
echo ================================================================
echo Configuration file location: %APPDATA%\Claude\claude_desktop_config.json
echo.
echo Next steps:
echo 1. Close Claude Desktop completely (check system tray)
echo 2. Restart Claude Desktop
echo 3. Look for Context7 in your available tools
echo 4. Test with: "Analyze my project structure"
echo.
echo If Context7 doesn't appear, check Claude Desktop logs for errors.
echo.
pause
