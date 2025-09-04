@echo off
REM Smart Go Build Script - Uses existing binaries before rebuilding
REM Optimized for Legal AI Production Environment

echo ======================================
echo Smart Go Build System - Legal AI
echo ======================================

set SCRIPT_DIR=%~dp0
set BIN_DIR=%SCRIPT_DIR%bin\
set GO_SERVICES_BIN=%SCRIPT_DIR%..\go-services\bin\

REM Create bin directory if it doesn't exist
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

echo.
echo [PHASE 1] Checking existing binaries...

REM Priority 1: Enhanced RAG Service (Port 8094)
echo.
echo Checking Enhanced RAG Service...
if exist "%BIN_DIR%enhanced-rag.exe" (
    echo ✅ Found: %BIN_DIR%enhanced-rag.exe
    set ENHANCED_RAG_READY=1
) else if exist "%GO_SERVICES_BIN%enhanced-rag.exe" (
    echo ✅ Found: %GO_SERVICES_BIN%enhanced-rag.exe  
    echo    Copying to main bin directory...
    copy "%GO_SERVICES_BIN%enhanced-rag.exe" "%BIN_DIR%enhanced-rag.exe"
    set ENHANCED_RAG_READY=1
) else (
    echo ❌ enhanced-rag.exe not found - will build
    set ENHANCED_RAG_READY=0
)

REM Priority 2: Upload Service (Port 8093)  
echo.
echo Checking Upload Service...
if exist "%BIN_DIR%upload-service.exe" (
    echo ✅ Found: %BIN_DIR%upload-service.exe
    set UPLOAD_SERVICE_READY=1
) else if exist "%SCRIPT_DIR%cmd\upload-service\upload-service.exe" (
    echo ✅ Found: %SCRIPT_DIR%cmd\upload-service\upload-service.exe
    echo    Copying to bin directory...
    copy "%SCRIPT_DIR%cmd\upload-service\upload-service.exe" "%BIN_DIR%upload-service.exe"
    set UPLOAD_SERVICE_READY=1
) else (
    echo ❌ upload-service.exe not found - will build
    set UPLOAD_SERVICE_READY=0
)

REM Priority 3: Kratos Server (gRPC Port 50051)
echo.
echo Checking Kratos Server...
if exist "%GO_SERVICES_BIN%kratos-server.exe" (
    echo ✅ Found: %GO_SERVICES_BIN%kratos-server.exe
    set KRATOS_READY=1
) else if exist "%BIN_DIR%kratos-server.exe" (
    echo ✅ Found: %BIN_DIR%kratos-server.exe  
    set KRATOS_READY=1
) else (
    echo ❌ kratos-server.exe not found - will build
    set KRATOS_READY=0
)

REM Priority 4: Load Balancer
echo.
echo Checking Load Balancer...
if exist "%BIN_DIR%load-balancer.exe" (
    echo ✅ Found: %BIN_DIR%load-balancer.exe
    set LOAD_BALANCER_READY=1
) else (
    echo ❌ load-balancer.exe not found - will build  
    set LOAD_BALANCER_READY=0
)

REM Summary of existing binaries
echo.
echo [PHASE 1 SUMMARY] Binary Status:
if %ENHANCED_RAG_READY%==1 echo ✅ Enhanced RAG Service - Ready
if %ENHANCED_RAG_READY%==0 echo 🔨 Enhanced RAG Service - Needs Build
if %UPLOAD_SERVICE_READY%==1 echo ✅ Upload Service - Ready  
if %UPLOAD_SERVICE_READY%==0 echo 🔨 Upload Service - Needs Build
if %KRATOS_READY%==1 echo ✅ Kratos Server - Ready
if %KRATOS_READY%==0 echo 🔨 Kratos Server - Needs Build  
if %LOAD_BALANCER_READY%==1 echo ✅ Load Balancer - Ready
if %LOAD_BALANCER_READY%==0 echo 🔨 Load Balancer - Needs Build

echo.
echo [PHASE 2] Smart Building (only what's needed)...

REM Build Enhanced RAG if needed
if %ENHANCED_RAG_READY%==0 (
    echo.
    echo Building Enhanced RAG Service...
    cd /d "%SCRIPT_DIR%"
    go build -o "%BIN_DIR%enhanced-rag.exe" "./cmd/enhanced-rag"
    if errorlevel 1 (
        echo ❌ Enhanced RAG build failed
        pause
        exit /b 1
    )
    echo ✅ Enhanced RAG built successfully
)

REM Build Upload Service if needed  
if %UPLOAD_SERVICE_READY%==0 (
    echo.
    echo Building Upload Service...
    cd /d "%SCRIPT_DIR%"
    go build -o "%BIN_DIR%upload-service.exe" "./cmd/upload-service"
    if errorlevel 1 (
        echo ❌ Upload Service build failed
        pause
        exit /b 1  
    )
    echo ✅ Upload Service built successfully
)

REM Build Kratos Server if needed
if %KRATOS_READY%==0 (
    echo.
    echo Building Kratos Server...
    cd /d "%SCRIPT_DIR%..\go-services"
    go build -o "bin\kratos-server.exe" "./cmd/kratos-server"
    if errorlevel 1 (
        echo ❌ Kratos Server build failed
        pause
        exit /b 1
    )
    echo ✅ Kratos Server built successfully
)

REM Build Load Balancer if needed
if %LOAD_BALANCER_READY%==0 (
    echo.
    echo Building Load Balancer...
    cd /d "%SCRIPT_DIR%"
    go build -o "%BIN_DIR%load-balancer.exe" "./cmd/load-balancer"
    if errorlevel 1 (
        echo ❌ Load Balancer build failed
        pause
        exit /b 1
    )
    echo ✅ Load Balancer built successfully  
)

echo.
echo [PHASE 3] Final Binary Verification...
echo.
echo Available binaries:
dir /b "%BIN_DIR%*.exe" 2>nul
echo.
if exist "%GO_SERVICES_BIN%" (
    echo Available in go-services/bin:
    dir /b "%GO_SERVICES_BIN%*.exe" 2>nul
)

echo.
echo ======================================
echo ✅ Smart Build Complete!
echo ======================================
echo.
echo Key binaries ready:
if exist "%BIN_DIR%enhanced-rag.exe" echo   → Enhanced RAG: %BIN_DIR%enhanced-rag.exe
if exist "%BIN_DIR%upload-service.exe" echo   → Upload Service: %BIN_DIR%upload-service.exe  
if exist "%GO_SERVICES_BIN%kratos-server.exe" echo   → Kratos Server: %GO_SERVICES_BIN%kratos-server.exe
if exist "%BIN_DIR%load-balancer.exe" echo   → Load Balancer: %BIN_DIR%load-balancer.exe

echo.
echo Usage:
echo   Enhanced RAG:    %BIN_DIR%enhanced-rag.exe
echo   Upload Service:  %BIN_DIR%upload-service.exe  
echo   Kratos Server:   %GO_SERVICES_BIN%kratos-server.exe
echo   Load Balancer:   %BIN_DIR%load-balancer.exe

pause