@echo off
:: RUN-ENHANCED-RAG-V2-COMPLETE.bat
:: Quick launcher for Enhanced RAG V2 with CRUD and Async

echo ========================================
echo   ENHANCED RAG V2 - COMPLETE INTEGRATION
echo   With CRUD, Async, and PostgreSQL
echo ========================================
echo.

:: Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo This script requires Administrator privileges.
    echo Right-click and select "Run as Administrator"
    pause
    exit /b 1
)

:: Run the PowerShell script with all features
echo Starting Enhanced RAG V2 with all features...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "RUN-ENHANCED-RAG-V2-COMPLETE.ps1" -Action all

echo.
echo ========================================
echo   ENHANCED RAG V2 IS NOW RUNNING!
echo ========================================
echo.
echo Access Points:
echo   API: http://localhost:8097
echo   WebSocket: ws://localhost:8097/ws
echo   
echo API Endpoints:
echo   - POST   /api/intents           (Create user intent)
echo   - GET    /api/intents/{id}      (Get intent by ID)
echo   - PUT    /api/intents/{id}      (Update intent)
echo   - DELETE /api/intents/{id}      (Delete intent)
echo   - GET    /api/intents           (List intents)
echo   
echo   - POST   /api/recommendations   (Create recommendation)
echo   - GET    /api/recommendations/{id} (Get recommendation)
echo   - POST   /api/recommendations/generate (Generate AI recommendations)
echo   
echo   - POST   /api/todos             (Create todo)
echo   - GET    /api/todos/{id}        (Get todo)
echo   - POST   /api/todos/solve       (Auto-solve todos with AI)
echo   - GET    /api/todos/pending     (List pending todos)
echo   
echo   - POST   /api/sessions          (Update user session)
echo   - GET    /api/sessions/{user}   (Get user session)
echo   
echo   - POST   /api/analytics/event   (Track analytics event)
echo   - GET    /health                (Health check)
echo.
echo Press any key to keep services running...
pause >nul
