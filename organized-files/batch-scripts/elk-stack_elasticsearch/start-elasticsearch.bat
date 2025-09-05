@echo off
echo Starting Elasticsearch for Legal AI system...

REM Check if Elasticsearch is installed
where elasticsearch >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Elasticsearch not found in PATH
    echo Please install Elasticsearch and add it to your PATH
    echo Download from: https://www.elastic.co/downloads/elasticsearch
    pause
    exit /b 1
)

REM Set environment variables
set ES_JAVA_OPTS=-Xms1g -Xmx1g
set ES_PATH_CONF=%~dp0

echo Starting Elasticsearch with legal AI configuration...
elasticsearch -Epath.conf="%~dp0"

echo Elasticsearch started. Access at http://localhost:9200
pause