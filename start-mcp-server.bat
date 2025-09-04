@echo off
echo 🚀 Starting Context7 MCP Server...
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\mcp-servers"

set PROJECT_ROOT=C:\Users\james\Desktop\deeds-web\deeds-web-app
set OLLAMA_ENDPOINT=http://localhost:11434
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set NODE_ENV=development

echo Environment configured:
echo - Project Root: %PROJECT_ROOT%
echo - Ollama: %OLLAMA_ENDPOINT%
echo - Database: %DATABASE_URL%
echo.

node mcp-context7-wrapper.js

pause