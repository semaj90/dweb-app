@echo off
echo Reading entire Claude folder...

set CLAUDE_DIR=%APPDATA%\Claude
set OUTPUT_FILE=C:\Users\james\Desktop\deeds-web\claude-folder-audit.txt

echo ===== CLAUDE FOLDER AUDIT ===== > "%OUTPUT_FILE%"
echo Timestamp: %date% %time% >> "%OUTPUT_FILE%"
echo. >> "%OUTPUT_FILE%"

echo Directory listing: >> "%OUTPUT_FILE%"
dir "%CLAUDE_DIR%" /s /a >> "%OUTPUT_FILE%" 2>&1

echo. >> "%OUTPUT_FILE%"
echo ===== CONFIG FILE ===== >> "%OUTPUT_FILE%"
if exist "%CLAUDE_DIR%\claude_desktop_config.json" (
    type "%CLAUDE_DIR%\claude_desktop_config.json" >> "%OUTPUT_FILE%" 2>&1
) else (
    echo No config file found >> "%OUTPUT_FILE%"
)

echo. >> "%OUTPUT_FILE%"
echo ===== LOGS FOLDER ===== >> "%OUTPUT_FILE%"
if exist "%CLAUDE_DIR%\logs" (
    dir "%CLAUDE_DIR%\logs" >> "%OUTPUT_FILE%" 2>&1
) else (
    echo No logs folder >> "%OUTPUT_FILE%"
)

echo Audit saved to: claude-folder-audit.txt
type "%OUTPUT_FILE%"

pause