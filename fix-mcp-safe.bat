@echo off
echo Applying safe MCP Windows fixes...

cd /d "%USERPROFILE%"

:: Create backup
copy ".claude.json" ".claude.json.backup_safe_%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%" >nul

:: Use PowerShell for safe regex replacement
powershell -Command "$content = Get-Content '.claude.json' -Raw; $content = $content -replace '(\"command\":\s*\")npx(@modelcontextprotocol/server-postgres)', '${1}cmd /c npx${2}'; $content = $content -replace '(\"command\":\s*\")npx(@modelcontextprotocol/server-context7)', '${1}cmd /c npx${2}'; Set-Content '.claude.json' $content -Encoding UTF8"

echo.
echo MCP servers fixed:
echo - postgres: Added 'cmd /c' wrapper
echo - context7: Added 'cmd /c' wrapper
echo.
echo Please restart VS Code and run /doctor to verify
pause