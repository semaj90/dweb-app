@echo off
echo 🧹 Removing Docker References for Windows Native Development
echo ==============================================================
echo.

echo 📁 Removing Docker configuration files...

REM Remove Docker Compose files
if exist docker-compose*.yml (
    del /q docker-compose*.yml
    echo ✅ Removed docker-compose files
)

REM Remove Docker environment files
if exist .env.docker (
    del /q .env.docker
    echo ✅ Removed .env.docker
)

REM Remove Docker configuration directory
if exist docker-config (
    rmdir /s /q docker-config
    echo ✅ Removed docker-config directory
)

REM Remove Docker files directory
if exist docker-files (
    rmdir /s /q docker-files
    echo ✅ Removed docker-files directory
)

REM Remove Docker scripts
del /q docker-cli-manager.ps1 2>nul
del /q docker-wsl-manager.sh 2>nul
del /q setup-docker-wsl2.sh 2>nul
del /q validate-wsl-docker.sh 2>nul
del /q start-postgres-docker.* 2>nul
del /q nvidia-docker.list 2>nul
del /q dockererrors*.txt 2>nul
echo ✅ Removed Docker scripts

REM Remove Docker todos and documentation
del /q tododocker*.md 2>nul
del /q sveltekit-frontend\tododocker*.md 2>nul
echo ✅ Removed Docker documentation

REM Remove Docker optimization files
del /q src\lib\optimization\docker-*.ts 2>nul
del /q sveltekit-frontend\src\lib\optimization\docker-*.ts 2>nul
echo ✅ Removed Docker optimization files

REM Remove Dockerfiles
for /r . %%f in (Dockerfile*) do (
    if exist "%%f" (
        del /q "%%f"
    )
)
echo ✅ Removed Dockerfiles

REM Remove Docker-specific directories
if exist pgvector (
    rmdir /s /q pgvector
    echo ✅ Removed pgvector Docker directory
)

if exist rag-backend (
    rmdir /s /q rag-backend
    echo ✅ Removed rag-backend Docker service
)

if exist services (
    rmdir /s /q services
    echo ✅ Removed Docker services directory
)

if exist sveltekit-frontend\deployment\monitoring (
    rmdir /s /q sveltekit-frontend\deployment\monitoring
    echo ✅ Removed Docker monitoring
)

if exist sveltekit-frontend\docker-compose.yml (
    del /q sveltekit-frontend\docker-compose.yml
    echo ✅ Removed SvelteKit Docker compose
)

if exist sveltekit-frontend\tts-service (
    rmdir /s /q sveltekit-frontend\tts-service
    echo ✅ Removed TTS Docker service
)

REM Clean up maintenance scripts
if exist scripts\maintenance (
    del /q scripts\maintenance\*docker*.ps1 2>nul
    echo ✅ Cleaned Docker maintenance scripts
)

echo.
echo 🚀 Updating VS Code settings for Windows native development...

REM Update VS Code settings to remove Docker references
powershell -Command "(Get-Content .vscode\settings.json) -replace '.*docker.*', '' | Set-Content .vscode\settings.json"

echo.
echo ✅ Docker cleanup completed!
echo.
echo 📋 Summary of actions:
echo - Removed all docker-compose files
echo - Removed Docker configuration directories
echo - Removed Docker scripts and batch files  
echo - Removed Docker optimization files
echo - Removed Docker services and containers
echo - Updated VS Code settings for native Windows development
echo.
echo 🎯 Your project is now configured for Windows native development!
echo Use Ollama, PostgreSQL, and services running directly on Windows.
echo.
pause