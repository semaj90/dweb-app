@echo off
echo Fixing MCP write permissions...

:: Kill Claude Desktop
taskkill /f /im Claude.exe 2>nul

:: Update config with write permissions
set CONFIG=%APPDATA%\Claude\claude_desktop_config.json

(
echo {
echo   "mcpServers": {
echo     "filesystem": {
echo       "command": "npx",
echo       "args": [
echo         "--yes",
echo         "@modelcontextprotocol/server-filesystem", 
echo         "C:/Users/james/Desktop/deeds-web",
echo         "--write-access"
echo       ]
echo     }
echo   }
echo }
) > "%CONFIG%"

echo ✅ Config updated with write access
echo Restart Claude Desktop

pause