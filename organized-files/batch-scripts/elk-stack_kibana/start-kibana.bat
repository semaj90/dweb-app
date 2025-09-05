@echo off
echo Starting Kibana for Legal AI system...

REM Check if Kibana is installed
where kibana >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Kibana not found in PATH
    echo Please install Kibana and add it to your PATH
    echo Download from: https://www.elastic.co/downloads/kibana
    pause
    exit /b 1
)

echo Starting Kibana with legal AI configuration...
kibana --config "%~dp0kibana.yml"

echo Kibana started. Access at http://localhost:5601
pause