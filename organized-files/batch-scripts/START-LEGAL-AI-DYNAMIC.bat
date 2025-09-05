@echo off
setlocal enabledelayedexpansion

:: Enhanced Legal AI System with Dynamic Port Discovery
:: Implements Vite-style port discovery and Context7 best practices

echo.
echo ========================================
echo 🚀 LEGAL AI PLATFORM - DYNAMIC STARTUP
echo ========================================
echo.

:: Set color for better visibility
color 0A

:: Kill any existing processes to free ports
echo 🔫 Freeing conflicting ports...
taskkill /F /IM enhanced-rag.exe >nul 2>&1
taskkill /F /IM upload-service.exe >nul 2>&1
taskkill /F /IM quic-gateway.exe >nul 2>&1
taskkill /F /IM kratos-server.exe >nul 2>&1

:: Wait for processes to terminate
timeout /t 3 /nobreak >nul

:: Function to find available port
:find_available_port
set service_name=%1
set preferred_port=%2
set found_port=%preferred_port%

for /l %%i in (0,1,20) do (
    set /a test_port=!preferred_port! + %%i
    netstat -an | findstr ":!test_port! " >nul
    if errorlevel 1 (
        set found_port=!test_port!
        goto port_found
    )
)

echo ❌ Could not find available port for %service_name% starting from %preferred_port%
exit /b 1

:port_found
if !found_port! neq %preferred_port% (
    echo 🔄 %service_name%: Port %preferred_port% occupied, using !found_port!
) else (
    echo ✅ %service_name%: Using preferred port !found_port!
)
exit /b 0

:: Dynamic port discovery for each service
echo 📡 Discovering available ports...

:: Enhanced RAG Service
call :find_available_port "Enhanced RAG" 8094
set RAG_PORT=!found_port!

:: Upload Service  
call :find_available_port "Upload Service" 8093
set UPLOAD_PORT=!found_port!

:: QUIC Gateway
call :find_available_port "QUIC Gateway" 8447
set QUIC_PORT=!found_port!

:: Kratos gRPC Server
call :find_available_port "Kratos Server" 50051
set GRPC_PORT=!found_port!

echo.
echo 🎯 Port Allocation Results:
echo   Enhanced RAG: %RAG_PORT%
echo   Upload Service: %UPLOAD_PORT% 
echo   QUIC Gateway: %QUIC_PORT%
echo   Kratos gRPC: %GRPC_PORT%
echo.

:: Save port configuration for Vite proxy
echo { > .vscode\dynamic-ports.json
echo   "timestamp": "%date% %time%", >> .vscode\dynamic-ports.json
echo   "ports": { >> .vscode\dynamic-ports.json
echo     "enhanced-rag": %RAG_PORT%, >> .vscode\dynamic-ports.json
echo     "upload-service": %UPLOAD_PORT%, >> .vscode\dynamic-ports.json
echo     "quic-gateway": %QUIC_PORT%, >> .vscode\dynamic-ports.json
echo     "kratos-server": %GRPC_PORT% >> .vscode\dynamic-ports.json
echo   }, >> .vscode\dynamic-ports.json
echo   "generator": "START-LEGAL-AI-DYNAMIC.bat" >> .vscode\dynamic-ports.json
echo } >> .vscode\dynamic-ports.json

echo 💾 Port configuration saved to .vscode\dynamic-ports.json

:: Start services with dynamic ports
echo.
echo 🚀 Starting Legal AI services with dynamic ports...

:: Start Enhanced RAG Service
echo 📊 Starting Enhanced RAG Service on port %RAG_PORT%...
set RAG_HTTP_PORT=%RAG_PORT%
set OLLAMA_BASE_URL=http://localhost:11434
start /b "Enhanced RAG" go-microservice\bin\enhanced-rag.exe
timeout /t 2 /nobreak >nul

:: Start Upload Service
echo 📁 Starting Upload Service on port %UPLOAD_PORT%...
set UPLOAD_PORT=%UPLOAD_PORT%
set MINIO_ENDPOINT=localhost:9000
start /b "Upload Service" go-microservice\bin\upload-service.exe
timeout /t 2 /nobreak >nul

:: Start QUIC Gateway (if available)
if exist go-microservice\bin\quic-gateway.exe (
    echo ⚡ Starting QUIC Gateway on port %QUIC_PORT%...
    set QUIC_HTTP_PORT=%QUIC_PORT%
    start /b "QUIC Gateway" go-microservice\bin\quic-gateway.exe
    timeout /t 2 /nobreak >nul
)

:: Start Kratos gRPC Server (if available)
if exist go-services\cmd\kratos-server\kratos-server.exe (
    echo 🔧 Starting Kratos gRPC Server on port %GRPC_PORT%...
    set GRPC_PORT=%GRPC_PORT%
    start /b "Kratos Server" go-services\cmd\kratos-server\kratos-server.exe
    timeout /t 2 /nobreak >nul
)

:: Update Vite configuration with new ports
echo 📝 Updating Vite proxy configuration...

:: Create updated vite proxy config snippet
echo // Auto-generated proxy configuration > .vscode\vite-proxy-dynamic.js
echo // Generated: %date% %time% >> .vscode\vite-proxy-dynamic.js
echo. >> .vscode\vite-proxy-dynamic.js
echo const dynamicProxyConfig = { >> .vscode\vite-proxy-dynamic.js
echo   '/api/go/enhanced-rag': { >> .vscode\vite-proxy-dynamic.js
echo     target: 'http://localhost:%RAG_PORT%', >> .vscode\vite-proxy-dynamic.js
echo     changeOrigin: true, >> .vscode\vite-proxy-dynamic.js
echo     rewrite: (path) =^> path.replace(/^\/api\/go\/enhanced-rag/, '') >> .vscode\vite-proxy-dynamic.js
echo   }, >> .vscode\vite-proxy-dynamic.js
echo   '/api/go/upload': { >> .vscode\vite-proxy-dynamic.js
echo     target: 'http://localhost:%UPLOAD_PORT%', >> .vscode\vite-proxy-dynamic.js
echo     changeOrigin: true, >> .vscode\vite-proxy-dynamic.js
echo     rewrite: (path) =^> path.replace(/^\/api\/go\/upload/, '') >> .vscode\vite-proxy-dynamic.js
echo   }, >> .vscode\vite-proxy-dynamic.js
echo   '/api/quic': { >> .vscode\vite-proxy-dynamic.js
echo     target: 'http://localhost:%QUIC_PORT%', >> .vscode\vite-proxy-dynamic.js
echo     changeOrigin: true, >> .vscode\vite-proxy-dynamic.js
echo     rewrite: (path) =^> path.replace(/^\/api\/quic/, '') >> .vscode\vite-proxy-dynamic.js
echo   }, >> .vscode\vite-proxy-dynamic.js
echo   '/api/grpc': { >> .vscode\vite-proxy-dynamic.js
echo     target: 'http://localhost:%GRPC_PORT%', >> .vscode\vite-proxy-dynamic.js
echo     changeOrigin: true, >> .vscode\vite-proxy-dynamic.js
echo     rewrite: (path) =^> path.replace(/^\/api\/grpc/, '') >> .vscode\vite-proxy-dynamic.js
echo   } >> .vscode\vite-proxy-dynamic.js
echo }; >> .vscode\vite-proxy-dynamic.js
echo. >> .vscode\vite-proxy-dynamic.js
echo export default dynamicProxyConfig; >> .vscode\vite-proxy-dynamic.js

:: Verify services are running
echo.
echo 🔍 Verifying service health...
timeout /t 5 /nobreak >nul

:: Check Enhanced RAG
curl -s -o nul --connect-timeout 3 http://localhost:%RAG_PORT%/health
if errorlevel 1 (
    echo ⚠️  Enhanced RAG may not be ready yet
) else (
    echo ✅ Enhanced RAG is healthy on port %RAG_PORT%
)

:: Check Upload Service
curl -s -o nul --connect-timeout 3 http://localhost:%UPLOAD_PORT%/health
if errorlevel 1 (
    echo ⚠️  Upload Service may not be ready yet
) else (
    echo ✅ Upload Service is healthy on port %UPLOAD_PORT%
)

echo.
echo ========================================
echo 🎉 LEGAL AI PLATFORM IS STARTING
echo ========================================
echo.
echo 📊 Service Dashboard:
echo   Enhanced RAG: http://localhost:%RAG_PORT%
echo   Upload Service: http://localhost:%UPLOAD_PORT%
echo   QUIC Gateway: http://localhost:%QUIC_PORT%
echo   Kratos gRPC: localhost:%GRPC_PORT%
echo.
echo 🔗 Next Steps:
echo   1. Start SvelteKit: cd sveltekit-frontend ^&^& npm run dev
echo   2. Access frontend: http://localhost:5173
echo   3. Monitor logs: Check running processes for errors
echo.
echo 💡 Port configuration saved for Vite integration
echo 📁 Configuration: .vscode\dynamic-ports.json
echo 📁 Proxy config: .vscode\vite-proxy-dynamic.js
echo.

:: Keep window open for monitoring
echo Press any key to close or Ctrl+C to stop services...
pause >nul