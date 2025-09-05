@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo    UPDATE CLAUDE DESKTOP CONFIG
echo    Enable MCP Filesystem Integration
echo =====================================================

echo Locating Claude Desktop config...
set CLAUDE_CONFIG=%APPDATA%\Claude\claude_desktop_config.json

if not exist "%CLAUDE_CONFIG%" (
    echo Creating Claude config directory...
    mkdir "%APPDATA%\Claude" 2>nul
)

echo Backing up existing config...
if exist "%CLAUDE_CONFIG%" (
    copy "%CLAUDE_CONFIG%" "%CLAUDE_CONFIG%.backup" >nul 2>&1
    echo ✓ Backup created: %CLAUDE_CONFIG%.backup
)

echo Creating updated Claude config with MCP filesystem...
> "%CLAUDE_CONFIG%" (
echo {
echo   "mcpServers": {
echo     "filesystem": {
echo       "name": "filesystem",
echo       "command": "npx",
echo       "args": ["--yes", "@modelcontextprotocol/server-filesystem", "C:/Users/james/Desktop/deeds-web/deeds-web-app", "C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend", "--silent"],
echo       "enabled": true
echo     },
echo     "serena": {
echo       "name": "serena", 
echo       "command": "uvx",
echo       "args": ["serena.main", "serve"],
echo       "enabled": true
echo     },
echo     "frontend": {
echo       "name": "frontend",
echo       "command": "pnpm",
echo       "args": ["--prefix", "C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend", "dev"],
echo       "enabled": true
echo     }
echo   },
echo   "context7": {
echo     "env": "development",
echo     "debug": true,
echo     "defaultPort": 3000,
echo     "logLevel": "verbose"
echo   }
echo }
)

echo ✅ Claude Desktop config updated!
echo.
echo Config location: %CLAUDE_CONFIG%
echo.
echo ⚠️  IMPORTANT: Restart Claude Desktop for changes to take effect
echo.
echo MCP Servers enabled:
echo - filesystem: Access to deeds-web project files
echo - serena: Code intelligence and generation
echo - frontend: Development server management
echo.

pause