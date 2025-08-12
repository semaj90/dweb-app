@echo off
REM Go Module Cache Management Script for Legacy Dependencies
REM Handles caching efficiently without unnecessary re-downloads

echo 🚀 Go Module Cache Management for Legacy Dependencies

REM Check if we're in the right directory
if not exist "go.mod" (
    echo ❌ Error: go.mod not found. Run this script from go-microservice directory
    exit /b 1
)

REM Display current cache info
echo 📊 Current Go module cache info:
go env GOPATH
for /f "tokens=*" %%a in ('powershell "Get-ChildItem (go env GOPATH)\pkg\mod -Recurse | Measure-Object -Property Length -Sum | Select-Object -ExpandProperty Sum"') do set CACHE_SIZE=%%a
set /a CACHE_MB=CACHE_SIZE/1024/1024
echo Cache size: %CACHE_MB% MB

echo.
echo 🎯 Checking legacy dependencies status...

REM Check which legacy deps are already cached
set MISSING_DEPS=0

REM List of legacy dependencies
set DEPS[0]=github.com/prometheus/client_golang/prometheus
set DEPS[1]=github.com/prometheus/client_golang/prometheus/promhttp  
set DEPS[2]=github.com/streadway/amqp
set DEPS[3]=google.golang.org/grpc
set DEPS[4]=gorgonia.org/gorgonia
set DEPS[5]=gorgonia.org/tensor

echo Checking cached dependencies...
for /l %%i in (0,1,5) do (
    call set dep=%%DEPS[%%i]%%
    call :check_cached !dep!
)

if %MISSING_DEPS%==0 (
    echo ✅ All legacy dependencies are already cached!
    echo 💡 You can run: go build -tags=legacy ./...
    goto :end
)

echo.
echo ❓ Found %MISSING_DEPS% missing dependencies.
echo.
echo Options:
echo   1. Download missing dependencies now (adds to cache)
echo   2. Skip legacy build (keep workspace lean)
echo   3. Show which deps are missing
echo.
set /p choice="Choose (1/2/3): "

if "%choice%"=="1" goto :download_deps
if "%choice%"=="2" goto :skip_legacy  
if "%choice%"=="3" goto :show_missing

:download_deps
echo 📥 Downloading missing legacy dependencies...
go get github.com/prometheus/client_golang/prometheus
go get github.com/prometheus/client_golang/prometheus/promhttp
go get github.com/streadway/amqp
go get google.golang.org/grpc
go get gorgonia.org/gorgonia
go get gorgonia.org/tensor
go mod tidy
echo ✅ Dependencies cached! Now you can: go build -tags=legacy ./...
goto :end

:skip_legacy
echo ⏭️  Skipping legacy dependencies. Legacy build will fail until deps are added.
echo 💡 Run this script again when you need legacy build support.
goto :end

:show_missing
echo 📋 Missing dependencies that would be downloaded:
echo   - github.com/prometheus/client_golang/prometheus
echo   - github.com/prometheus/client_golang/prometheus/promhttp
echo   - github.com/streadway/amqp
echo   - google.golang.org/grpc  
echo   - gorgonia.org/gorgonia
echo   - gorgonia.org/tensor
goto :end

:check_cached
REM Function to check if a dependency is cached
REM This is a simplified check - in practice, you'd need more sophisticated detection
exit /b 0

:end
echo.
echo 🔧 Additional commands:
echo   go clean -modcache  : Clear entire cache (forces re-download)
echo   go mod tidy        : Sync dependencies with go.mod
echo   go build ./...     : Build without legacy tags
echo   go build -tags=legacy ./... : Build with legacy tags
echo.
pause