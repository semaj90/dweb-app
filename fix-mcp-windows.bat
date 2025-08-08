@echo off
echo ========================================
echo MCP Server Windows Configuration Fixer
echo ========================================
echo.

echo This script will fix the MCP server configurations for Windows
echo by adding the required 'cmd /c' wrapper to npx commands.
echo.

echo The following servers need fixing:
echo   - puppeteer
echo   - postgres  
echo   - filesystem
echo   - context7
echo.

echo IMPORTANT: Your .claude.json file is very large (12.6MB).
echo Manual editing is recommended for safety.
echo.

echo Please manually edit: %USERPROFILE%\.claude.json
echo.
echo For each server (puppeteer, postgres, filesystem, context7), change:
echo   "command": "npx ..."
echo To:
echo   "command": "cmd /c npx ..."
echo.

echo Example:
echo   BEFORE: "command": "npx @modelcontextprotocol/server-puppeteer"
echo   AFTER:  "command": "cmd /c npx @modelcontextprotocol/server-puppeteer"
echo.

echo Press any key to open the configuration file in Notepad...
pause > nul

start notepad "%USERPROFILE%\.claude.json"

echo.
echo After making the changes:
echo   1. Save the file
echo   2. Restart VS Code
echo   3. Run /doctor command again to verify
echo.
pause