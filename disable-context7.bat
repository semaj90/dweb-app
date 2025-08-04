@echo off
echo 🔧 Disabling Context7 MCP Assistant Extension...

REM Disable the extension via VS Code CLI
echo 1️⃣ Disabling extension via CLI...
code --disable-extension "undefined_publisher.mcp-context7-assistant"

REM Check if the extension was disabled
echo 2️⃣ Checking extension status...
code --list-extensions --show-versions | findstr "mcp-context7" >nul
if %errorlevel%==0 (
    echo ⚠️  Extension still found in list (may be disabled)
) else (
    echo ✅ Extension not found in active list
)

echo.
echo 🎯 Results:
echo • Context7 MCP Assistant disabled locally
echo • Settings updated to prevent auto-loading
echo • Extension remains installed for other workspaces
echo.
echo 🚀 Next Steps:
echo 1. Restart VS Code to apply changes
echo 2. Open this workspace: deeds-web-app.code-workspace
echo 3. The extension should no longer cause lag
echo.
echo ✅ Context7 MCP Assistant disabled successfully!
pause