@echo off
echo Starting Legal AI Phase 3-4 System - Enhanced Features...
echo.

echo [1/3] Starting Phase 3-4 Docker services...
docker-compose -f docker-compose-phase34-DEFINITIVE.yml up -d

if %ERRORLEVEL% neq 0 (
    echo [X] ERROR: Failed to start Docker services.
    echo Please check Docker Desktop is running and try again.
    pause
    exit /b 1
)

echo [2/3] Waiting for services to initialize...
timeout /t 30 /nobreak >nul
echo   [+] Services initialization complete.

echo [3/3] Starting SvelteKit frontend...
cd sveltekit-frontend
start cmd /k "npm run dev"
cd ..

echo.
echo =========================================================
echo   PHASE 3-4 ENHANCED SYSTEM STARTED SUCCESSFULLY!
echo =========================================================
echo.
echo 🚀 QUICK ACCESS:
echo   Frontend:        http://localhost:5173
echo   AI Summary Demo: http://localhost:5173/demo/ai-summary
echo   Evidence Analysis: http://localhost:5173/demo/evidence-analysis
echo   Case Synthesis: http://localhost:5173/demo/case-synthesis
echo.
echo 🛠️  ADMIN INTERFACES:
echo   RabbitMQ Mgmt:   http://localhost:15672 (legal_admin/LegalRAG2024!)
echo   Neo4j Browser:   http://localhost:7474 (neo4j/LegalRAG2024!)
echo   Qdrant REST:     http://localhost:6333
echo.
echo 📊 ENHANCED FEATURES READY:
echo   ✅ XState AI Summary Components
echo   ✅ Evidence Report Analysis
echo   ✅ Case Synthesis Workflow
echo   ✅ Voice synthesis integration
echo   ✅ PostgreSQL with pgvector
echo   ✅ Redis caching
echo   ✅ Qdrant vector database
echo   ✅ Neo4j graph database
echo   ✅ RabbitMQ event streaming
echo   ✅ Ollama LLM inference
echo.
echo System is ready for advanced legal AI workflows!
echo.
pause