@echo off
setlocal enabledelayedexpansion

:: Context7 Legal AI - Claude MCP Integration Setup
echo ========================================
echo  Context7 Legal AI - Claude MCP Setup
echo ========================================
echo.

:: Check if Claude Desktop is installed
where claude >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Claude CLI not found. Please install Claude Desktop first.
    echo Download from: https://claude.ai/download
    pause
    exit /b 1
)

:: Set project paths
set "PROJECT_ROOT=%~dp0"
set "MCP_CONFIG=%PROJECT_ROOT%context7-mcp-config.json"
set "CLAUDE_CONFIG=%APPDATA%\Claude\claude_desktop_config.json"

echo [INFO] Project Root: %PROJECT_ROOT%
echo [INFO] MCP Config: %MCP_CONFIG%
echo [INFO] Claude Config: %CLAUDE_CONFIG%
echo.

:: Check if MCP config exists
if not exist "%MCP_CONFIG%" (
    echo [ERROR] Context7 MCP config not found: %MCP_CONFIG%
    pause
    exit /b 1
)

:: Backup existing Claude config
if exist "%CLAUDE_CONFIG%" (
    echo [INFO] Backing up existing Claude Desktop config...
    copy "%CLAUDE_CONFIG%" "%CLAUDE_CONFIG%.backup.%date:~-4,4%%date:~-10,2%%date:~-7,2%"
)

:: Install required MCP packages
echo [INFO] Installing MCP server dependencies...
npm install -g @modelcontextprotocol/sdk
npm install -g @context7/mcp-server

:: Create Claude Desktop config directory if it doesn't exist
if not exist "%APPDATA%\Claude" mkdir "%APPDATA%\Claude"

:: Copy MCP configuration to Claude Desktop
echo [INFO] Installing Context7 MCP configuration...
copy "%MCP_CONFIG%" "%CLAUDE_CONFIG%"

:: Test MCP server
echo [INFO] Testing Context7 MCP server...
node "%PROJECT_ROOT%scripts\context7-mcp-server.js" --test

if %errorlevel% equ 0 (
    echo [SUCCESS] Context7 MCP server test passed!
) else (
    echo [WARNING] MCP server test failed, but configuration installed.
)

echo.
echo ========================================
echo  Context7 MCP Integration Complete
echo ========================================
echo.
echo Next steps:
echo 1. Restart Claude Desktop
echo 2. Open a new conversation
echo 3. Test with: "List legal documents" or "Get project status"
echo.
echo Available MCP Tools:
echo - get_legal_docs: Access legal documentation
echo - get_project_status: Check integration status
echo - get_legal_schema: View database schema
echo - get_api_endpoints: List API endpoints
echo - get_gemma3_config: View AI model config
echo.
echo For manual setup, use:
echo claude mcp add context7 -- npx -y @context7/mcp-server
echo.
pause