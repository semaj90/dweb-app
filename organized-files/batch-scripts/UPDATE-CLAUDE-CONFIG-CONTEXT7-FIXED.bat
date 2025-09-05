@echo off
setlocal enabledelayedexpansion
title Update Claude Config - Fixed Version
color 0A

echo ========================================
echo CLAUDE DESKTOP CONFIG - FIXED VERSION
echo ========================================
echo.

echo Stopping Claude Desktop if running...
taskkill /f /im Claude.exe > nul 2>&1
timeout /t 2 > nul

set "CLAUDE_CONFIG=%APPDATA%\Claude\claude_desktop_config.json"

echo Backing up existing configuration...
if exist "%CLAUDE_CONFIG%" (
    copy "%CLAUDE_CONFIG%" "%CLAUDE_CONFIG%.backup" > nul
    echo ✅ Backup created
) else (
    mkdir "%APPDATA%\Claude" > nul 2>&1
)

echo Creating simplified Claude configuration...
> "%CLAUDE_CONFIG%" (
echo {
echo   "mcpServers": {
echo     "filesystem": {
echo       "command": "npx",
echo       "args": ["--yes", "@modelcontextprotocol/server-filesystem", "C:/Users/james/Desktop/deeds-web/deeds-web-app", "--write-access"]
echo     }
echo   }
echo }
)

echo ✅ Claude Desktop configuration updated!
echo.
echo 📁 Config location: %CLAUDE_CONFIG%
echo.
echo ⚠️  IMPORTANT: Restart Claude Desktop for changes to take effect
echo.
pause