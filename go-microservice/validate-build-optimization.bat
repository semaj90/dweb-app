@echo off
REM Build Validation Script - Legal AI Production Environment
REM Validates build optimization strategy and binary health

echo ========================================
echo Build Optimization Validation
echo ========================================

set SCRIPT_DIR=%~dp0
set BIN_DIR=%SCRIPT_DIR%bin\
set GO_SERVICES_BIN=%SCRIPT_DIR%..\go-services\bin\
set VALIDATION_LOG=%SCRIPT_DIR%build-validation-report.txt

REM Create validation log
echo Build Optimization Validation Report > "%VALIDATION_LOG%"
echo Generated: %date% %time% >> "%VALIDATION_LOG%"
echo ======================================== >> "%VALIDATION_LOG%"

echo.
echo [PHASE 1] Dependency Analysis...
echo.

REM Check Go version and environment
echo Checking Go environment...
go version
if errorlevel 1 (
    echo ❌ Go not installed or not in PATH
    echo ERROR: Go not available >> "%VALIDATION_LOG%"
    pause
    exit /b 1
)

echo ✅ Go environment ready
echo Go environment: OK >> "%VALIDATION_LOG%"

REM Check module dependencies
echo.
echo Analyzing go.mod dependencies...
cd /d "%SCRIPT_DIR%"

REM Count dependencies
for /f "tokens=*" %%i in ('go list -m all ^| find /c /v ""') do set DEPENDENCY_COUNT=%%i
echo Found %DEPENDENCY_COUNT% total dependencies (including main module)
echo Dependencies: %DEPENDENCY_COUNT% >> "%VALIDATION_LOG%"

REM Check for key dependencies
echo.
echo Verifying key dependencies...
set KEY_DEPS_OK=0

echo   Checking gin-gonic/gin...
go list -m github.com/gin-gonic/gin > nul 2>&1
if errorlevel 1 (
    echo     ❌ gin-gonic/gin missing
    echo ERROR: gin-gonic/gin missing >> "%VALIDATION_LOG%"
) else (
    echo     ✅ gin-gonic/gin available
    set /a KEY_DEPS_OK+=1
)

echo   Checking gorilla/websocket...
go list -m github.com/gorilla/websocket > nul 2>&1
if errorlevel 1 (
    echo     ❌ gorilla/websocket missing
    echo ERROR: gorilla/websocket missing >> "%VALIDATION_LOG%"
) else (
    echo     ✅ gorilla/websocket available
    set /a KEY_DEPS_OK+=1
)

echo   Checking nats-io/nats.go...
go list -m github.com/nats-io/nats.go > nul 2>&1
if errorlevel 1 (
    echo     ❌ nats-io/nats.go missing
    echo ERROR: nats-io/nats.go missing >> "%VALIDATION_LOG%"
) else (
    echo     ✅ nats-io/nats.go available
    set /a KEY_DEPS_OK+=1
)

echo   Checking pgvector/pgvector-go...
go list -m github.com/pgvector/pgvector-go > nul 2>&1
if errorlevel 1 (
    echo     ❌ pgvector/pgvector-go missing
    echo ERROR: pgvector/pgvector-go missing >> "%VALIDATION_LOG%"
) else (
    echo     ✅ pgvector/pgvector-go available
    set /a KEY_DEPS_OK+=1
)

echo.
echo Key dependencies check: %KEY_DEPS_OK%/4 available
echo Key dependencies: %KEY_DEPS_OK%/4 >> "%VALIDATION_LOG%"

echo.
echo [PHASE 2] Binary Inventory...
echo.

REM Binary analysis
echo Scanning existing binaries...
echo. >> "%VALIDATION_LOG%"
echo BINARY INVENTORY >> "%VALIDATION_LOG%"
echo ================ >> "%VALIDATION_LOG%"

set TOTAL_BINARIES=0

if exist "%BIN_DIR%" (
    echo.
    echo Main bin directory (%BIN_DIR%):
    echo Main bin/ directory: >> "%VALIDATION_LOG%"
    for %%f in ("%BIN_DIR%*.exe") do (
        echo   ✅ %%~nxf
        echo   - %%~nxf >> "%VALIDATION_LOG%"
        set /a TOTAL_BINARIES+=1
    )
) else (
    echo ❌ Main bin directory not found
    echo ERROR: Main bin/ directory missing >> "%VALIDATION_LOG%"
)

if exist "%GO_SERVICES_BIN%" (
    echo.
    echo Go services bin directory (%GO_SERVICES_BIN%):
    echo go-services bin/ directory: >> "%VALIDATION_LOG%"
    for %%f in ("%GO_SERVICES_BIN%*.exe") do (
        echo   ✅ %%~nxf
        echo   - %%~nxf >> "%VALIDATION_LOG%"
        set /a TOTAL_BINARIES+=1
    )
) else (
    echo ❌ Go services bin directory not found
    echo ERROR: go-services bin/ directory missing >> "%VALIDATION_LOG%"
)

echo.
echo Total existing binaries: %TOTAL_BINARIES%
echo Total binaries: %TOTAL_BINARIES% >> "%VALIDATION_LOG%"

echo.
echo [PHASE 3] Priority Service Analysis...
echo.

REM Check priority services
echo Priority Service Status:
echo. >> "%VALIDATION_LOG%"
echo PRIORITY SERVICES >> "%VALIDATION_LOG%"
echo ================= >> "%VALIDATION_LOG%"

set PRIORITY_SERVICES_READY=0

echo   P1: Enhanced RAG Service (Port 8094)
if exist "%BIN_DIR%enhanced-rag.exe" (
    echo     ✅ Binary available: %BIN_DIR%enhanced-rag.exe
    echo     ✅ Enhanced RAG Service - Ready >> "%VALIDATION_LOG%"
    set /a PRIORITY_SERVICES_READY+=1
) else if exist "%GO_SERVICES_BIN%enhanced-rag.exe" (
    echo     ✅ Binary available: %GO_SERVICES_BIN%enhanced-rag.exe
    echo     ✅ Enhanced RAG Service - Ready (go-services) >> "%VALIDATION_LOG%"
    set /a PRIORITY_SERVICES_READY+=1
) else (
    echo     ❌ Enhanced RAG binary missing - needs build
    echo     ❌ Enhanced RAG Service - Missing >> "%VALIDATION_LOG%"
)

echo   P2: Upload Service (Port 8093)
if exist "%BIN_DIR%upload-service.exe" (
    echo     ✅ Binary available: %BIN_DIR%upload-service.exe
    echo     ✅ Upload Service - Ready >> "%VALIDATION_LOG%"
    set /a PRIORITY_SERVICES_READY+=1
) else (
    echo     ❌ Upload Service binary missing - needs build
    echo     ❌ Upload Service - Missing >> "%VALIDATION_LOG%"
)

echo   P3: Kratos Server (gRPC Port 50051)
if exist "%GO_SERVICES_BIN%kratos-server.exe" (
    echo     ✅ Binary available: %GO_SERVICES_BIN%kratos-server.exe
    echo     ✅ Kratos Server - Ready >> "%VALIDATION_LOG%"
    set /a PRIORITY_SERVICES_READY+=1
) else (
    echo     ❌ Kratos Server binary missing - needs build
    echo     ❌ Kratos Server - Missing >> "%VALIDATION_LOG%"
)

echo   P4: Load Balancer (Port 8222)
if exist "%BIN_DIR%load-balancer.exe" (
    echo     ✅ Binary available: %BIN_DIR%load-balancer.exe
    echo     ✅ Load Balancer - Ready >> "%VALIDATION_LOG%"
    set /a PRIORITY_SERVICES_READY+=1
) else (
    echo     ❌ Load Balancer binary missing - needs build
    echo     ❌ Load Balancer - Missing >> "%VALIDATION_LOG%"
)

echo.
echo Priority services ready: %PRIORITY_SERVICES_READY%/4
echo Priority services ready: %PRIORITY_SERVICES_READY%/4 >> "%VALIDATION_LOG%"

echo.
echo [PHASE 4] Build Capability Test...
echo.

REM Test build capability
echo Testing build environment...

REM Test simple build
echo   Testing Go build capability...
go build -o "%TEMP%\test-build.exe" -x -ldflags="-s -w" "%SCRIPT_DIR%main.go" > nul 2>&1
if exist "%TEMP%\test-build.exe" (
    echo   ✅ Go build capability verified
    echo   ✅ Build capability - OK >> "%VALIDATION_LOG%"
    del "%TEMP%\test-build.exe" > nul 2>&1
) else (
    echo   ❌ Go build capability failed
    echo   ERROR: Build capability - Failed >> "%VALIDATION_LOG%"
)

REM Check CGO capability (for GPU builds)
echo   Testing CGO capability...
set CGO_ENABLED=1
go env CGO_ENABLED | find "1" > nul
if errorlevel 1 (
    echo   ⚠️  CGO not enabled - GPU builds may not work
    echo   WARNING: CGO not enabled >> "%VALIDATION_LOG%"
) else (
    echo   ✅ CGO capability available
    echo   ✅ CGO capability - OK >> "%VALIDATION_LOG%"
)

echo.
echo [PHASE 5] Optimization Recommendations...
echo.

echo Build optimization recommendations:
echo. >> "%VALIDATION_LOG%"
echo OPTIMIZATION RECOMMENDATIONS >> "%VALIDATION_LOG%"
echo ============================== >> "%VALIDATION_LOG%"

if %TOTAL_BINARIES% gtr 15 (
    echo ✅ Excellent binary reuse potential - %TOTAL_BINARIES% binaries available
    echo ✅ Binary reuse: Excellent (%TOTAL_BINARIES% available) >> "%VALIDATION_LOG%"
) else if %TOTAL_BINARIES% gtr 10 (
    echo ✅ Good binary reuse potential - %TOTAL_BINARIES% binaries available  
    echo ✅ Binary reuse: Good (%TOTAL_BINARIES% available) >> "%VALIDATION_LOG%"
) else if %TOTAL_BINARIES% gtr 5 (
    echo ⚠️  Moderate binary reuse potential - %TOTAL_BINARIES% binaries available
    echo ⚠️ Binary reuse: Moderate (%TOTAL_BINARIES% available) >> "%VALIDATION_LOG%"
) else (
    echo ❌ Limited binary reuse potential - only %TOTAL_BINARIES% binaries available
    echo ❌ Binary reuse: Limited (%TOTAL_BINARIES% available) >> "%VALIDATION_LOG%"
)

if %KEY_DEPS_OK% equ 4 (
    echo ✅ All key dependencies available - optimal for build speed
    echo ✅ Dependencies: All key deps available >> "%VALIDATION_LOG%"
) else (
    echo ⚠️  Some key dependencies missing - may need go mod tidy
    echo ⚠️ Dependencies: Some missing, run go mod tidy >> "%VALIDATION_LOG%"
)

if %PRIORITY_SERVICES_READY% gtr 2 (
    echo ✅ Most priority services ready - minimal build time expected
    echo ✅ Priority services: Most ready >> "%VALIDATION_LOG%"
) else (
    echo ⚠️  Several priority services need building - expect longer build time  
    echo ⚠️ Priority services: Several need building >> "%VALIDATION_LOG%"
)

echo.
echo [SUMMARY] Build Optimization Assessment
echo ========================================
echo. >> "%VALIDATION_LOG%"
echo SUMMARY >> "%VALIDATION_LOG%"
echo ======= >> "%VALIDATION_LOG%"

REM Calculate overall score
set /a SCORE=0
if %KEY_DEPS_OK% equ 4 set /a SCORE+=25
if %TOTAL_BINARIES% gtr 15 set /a SCORE+=25
if %PRIORITY_SERVICES_READY% gtr 2 set /a SCORE+=25
if exist "%BIN_DIR%enhanced-rag.exe" set /a SCORE+=25

echo Overall Optimization Score: %SCORE%/100
echo Overall score: %SCORE%/100 >> "%VALIDATION_LOG%"

if %SCORE% geq 75 (
    echo 🎉 EXCELLENT - Build optimization highly effective
    echo Status: EXCELLENT - Highly optimized >> "%VALIDATION_LOG%"
) else if %SCORE% geq 50 (
    echo ✅ GOOD - Build optimization will provide significant benefits
    echo Status: GOOD - Well optimized >> "%VALIDATION_LOG%"  
) else if %SCORE% geq 25 (
    echo ⚠️  FAIR - Some optimization benefits, consider running smart-build first
    echo Status: FAIR - Partially optimized >> "%VALIDATION_LOG%"
) else (
    echo ❌ POOR - Limited optimization benefits, full builds may be needed
    echo Status: POOR - Needs optimization >> "%VALIDATION_LOG%"
)

echo.
echo Next steps:
if %SCORE% lss 75 (
    echo   1. Run smart-build-optimized.bat to build missing services
    echo   Recommendation: Run smart-build-optimized.bat >> "%VALIDATION_LOG%"
)
if %KEY_DEPS_OK% lss 4 (
    echo   2. Run 'go mod tidy' to resolve missing dependencies
    echo   Recommendation: Run go mod tidy >> "%VALIDATION_LOG%"  
)
if %PRIORITY_SERVICES_READY% lss 3 (
    echo   3. Build priority services for optimal performance
    echo   Recommendation: Build priority services >> "%VALIDATION_LOG%"
)

echo.
echo Validation report saved to: %VALIDATION_LOG%
echo.
echo ========================================
echo ✅ Validation Complete
echo ========================================

pause