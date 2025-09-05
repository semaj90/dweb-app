@echo off
echo Starting Logstash for Legal AI system...

REM Check if Logstash is installed
where logstash >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Logstash not found in PATH
    echo Please install Logstash and add it to your PATH
    echo Download from: https://www.elastic.co/downloads/logstash
    pause
    exit /b 1
)

echo Starting Logstash with legal AI pipeline...
logstash -f "%~dp0legal-ai-pipeline.conf" --config.reload.automatic

echo Logstash started with legal AI configuration
pause