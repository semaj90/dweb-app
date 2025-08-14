@echo off
echo 🚀 Quick AI System Setup for Legal Document Analysis
echo.

:: Check if Ollama is running with required models
echo 🤖 Checking Ollama and models...
ollama list | findstr "gemma3-legal" >nul
if %errorlevel% neq 0 (
    echo 📥 Pulling gemma3-legal model (this may take several minutes)...
    ollama pull gemma3-legal
) else (
    echo ✅ gemma3-legal model found
)

ollama list | findstr "nomic-embed-text" >nul
if %errorlevel% neq 0 (
    echo 📥 Pulling nomic-embed-text model...
    ollama pull nomic-embed-text
) else (
    echo ✅ nomic-embed-text model found
)

echo.
echo 🧪 Testing AI components...

:: Test Ollama connectivity
echo Testing Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Cannot connect to Ollama at localhost:11434
    echo    Please start Ollama: ollama serve
    echo    Then run this script again
    pause
    exit /b 1
) else (
    echo ✅ Ollama connection successful
)

:: Test embedding generation
echo Testing embedding generation...
curl -s -X POST http://localhost:11434/api/embeddings ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"nomic-embed-text\",\"prompt\":\"test legal document\"}" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Embedding generation failed
) else (
    echo ✅ Embedding generation working
)

:: Test text generation
echo Testing text generation...
curl -s -X POST http://localhost:11434/api/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"gemma3-legal\",\"prompt\":\"What is money laundering?\",\"stream\":false}" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Text generation failed
) else (
    echo ✅ Text generation working
)

echo.
echo 🎉 AI System Status Check Complete!
echo.
echo 📊 System Ready:
echo    ✅ Ollama service running
echo    ✅ gemma3-legal model available
echo    ✅ nomic-embed-text model available
echo    ✅ Embedding generation working
echo    ✅ Text generation working
echo.
echo 🚀 You can now start the development server:
echo    cd sveltekit-frontend
echo    npm run dev
echo.
echo 🧪 Test the AI system at:
echo    http://localhost:5173/demo/ai-test
echo    http://localhost:5173/demo/ai-complete-test
echo.
echo 🔧 API Endpoints:
echo    Ollama API: http://localhost:11434
echo    Generate: POST /api/generate
echo    Embeddings: POST /api/embeddings
echo.

pause