@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo    🎮 LAUNCHING YORHA LEGAL AI DEMO SYSTEM 🎮
echo    Complete with your custom Gemma3 model integration
echo ============================================================
echo.

set "GREEN=[92m"
set "BLUE=[94m"
set "YELLOW=[93m"
set "NC=[0m"

echo %BLUE%🚀 STEP 1: Starting Backend Services%NC%
echo.

echo %YELLOW%Starting Docker containers...%NC%
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"
docker-compose up -d

echo.
echo %YELLOW%Waiting for services to initialize...%NC%
timeout /t 15 /nobreak > nul

echo.
echo %BLUE%🚀 STEP 2: Starting Phase 3 Agentic Backend%NC%
echo.

echo %YELLOW%Installing Python dependencies...%NC%
cd backend
pip install -r requirements.txt

echo %YELLOW%Starting Phase 3 RAG API Server on port 9000...%NC%
start "Phase 3 Agentic Backend" python phase3_agentic_rag.py

echo.
echo %BLUE%🚀 STEP 3: Launching Demo Page%NC%
echo.

echo %YELLOW%Opening YoRHa Legal AI Demo in browser...%NC%
cd ..\frontend
start "YoRHa Legal AI Demo" "demo.html"

echo.
echo %GREEN%✨ YORHA LEGAL AI DEMO SYSTEM LAUNCHED! ✨%NC%
echo.
echo %BLUE%📋 Available Services:%NC%
echo   • Demo Page: file:///frontend/demo.html
echo   • Gemma3 Model: http://localhost:11434/api/generate
echo   • Enhanced API: http://localhost:8000/v1/chat/completions  
echo   • Phase 3 RAG API: http://localhost:9000/docs
echo   • Qdrant Dashboard: http://localhost:6333
echo   • PostgreSQL: localhost:5432
echo.
echo %BLUE%🎯 Demo Features:%NC%
echo   ✅ YoRHa-styled chat assistant
echo   ✅ Draggable AI widget
echo   ✅ Real-time chat with Gemma3 model
echo   ✅ Sample legal queries
echo   ✅ API status monitoring
echo   ✅ NieR: Automata inspired design
echo.
echo %BLUE%🎮 How to Use:%NC%
echo   1. Click the "Legal AI" button in bottom-right corner
echo   2. Try the sample query buttons
echo   3. Chat with your custom Gemma3 legal assistant
echo   4. Drag the chat window around the screen
echo   5. Explore the advanced RAG capabilities
echo.
echo %GREEN%🎊 Your Unsloth-trained Gemma3 model is ready for legal AI!%NC%
echo.
pause
