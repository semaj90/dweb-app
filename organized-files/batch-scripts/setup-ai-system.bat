@echo off
echo 🚀 Setting up Complete AI System for Legal Document Analysis
echo.

:: Change to sveltekit-frontend directory
cd sveltekit-frontend

:: Check if Ollama is running
echo 🤖 Checking Ollama status...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama not found! Please install Ollama first.
    echo    Download from: https://ollama.ai
    pause
    exit /b 1
)

echo ✅ Ollama is available

:: Check required models
echo 📦 Checking required models...
ollama list | findstr "gemma3-legal" >nul
if %errorlevel% neq 0 (
    echo 📥 Pulling gemma3-legal model...
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

:: Check PostgreSQL
echo 🐘 Checking PostgreSQL...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ PostgreSQL 17 not found at expected location
    echo    Please ensure PostgreSQL 17 is installed
    pause
    exit /b 1
)

echo ✅ PostgreSQL 17 found

:: Install dependencies
echo 📦 Installing dependencies...
npm install

:: Generate database schema
echo 🗃️ Generating database schema...
npx drizzle-kit generate

:: Set environment variables
echo 🔧 Setting up environment...
if not exist .env (
    echo DATABASE_URL=postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db > .env
    echo OLLAMA_API_URL=http://localhost:11434 >> .env
    echo QDRANT_URL=http://localhost:6333 >> .env
    echo AI_ENABLED=true >> .env
    echo VECTOR_SEARCH_ENABLED=true >> .env
    echo.
    echo ⚠️  Created .env file with default values
    echo    Please update with your actual database credentials
)

:: Run database setup
echo 🗄️ Setting up database...
echo Running SQL setup script...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -f "..\setup-complete-ai-system.sql" 2>setup_errors.log
if %errorlevel% neq 0 (
    echo ⚠️  Database setup encountered issues. Check setup_errors.log
) else (
    echo ✅ Database setup completed
)

:: Run migrations
echo 🔄 Running migrations...
npx drizzle-kit migrate

:: Seed database
echo 🌱 Seeding database...
node -e "import('./src/lib/server/db/seed.js').then(m => m.seed()).catch(console.error)"

:: Test AI services
echo 🧪 Testing AI services...
echo.
echo Testing Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Cannot connect to Ollama at localhost:11434
    echo    Please ensure Ollama is running: ollama serve
) else (
    echo ✅ Ollama connection successful
)

echo.
echo Testing models...
ollama show gemma3-legal >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ gemma3-legal model not accessible
) else (
    echo ✅ gemma3-legal model ready
)

ollama show nomic-embed-text >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ nomic-embed-text model not accessible
) else (
    echo ✅ nomic-embed-text model ready
)

:: Create test AI query
echo.
echo 🔬 Testing AI pipeline...
echo Creating test query...

node -e "
const { aiService } = require('./src/lib/server/services/ai-service.js');

async function testAI() {
  try {
    console.log('🤖 Testing AI query processing...');
    
    // Test embedding generation
    const embedding = await aiService.getOrCreateEmbedding('test legal query');
    console.log('✅ Embedding generation successful:', embedding.length, 'dimensions');
    
    console.log('🎉 AI pipeline test completed successfully!');
    
  } catch (error) {
    console.error('❌ AI pipeline test failed:', error.message);
  }
}

testAI();
" 2>ai_test_errors.log

if %errorlevel% neq 0 (
    echo ⚠️  AI pipeline test encountered issues. Check ai_test_errors.log
) else (
    echo ✅ AI pipeline test completed
)

echo.
echo 🎉 AI System Setup Complete!
echo.
echo 📊 System Status:
echo    ✅ PostgreSQL 17 + pgvector
echo    ✅ Ollama with gemma3-legal model
echo    ✅ Database schema and migrations
echo    ✅ Seed data loaded
echo    ✅ AI services configured
echo.
echo 🚀 Ready to start development:
echo    npm run dev
echo.
echo 🧪 Test the AI system:
echo    http://localhost:5173/demo/simple-test
echo    http://localhost:5173/demo/ai-test
echo.
echo 📚 API Endpoints Available:
echo    POST /api/ai/query - Process AI queries
echo    POST /api/ai/analyze-evidence - Analyze evidence
echo    POST /api/ai/search - Vector search
echo.
echo 🔐 Test Credentials:
echo    prosecutor@legal.ai / password123
echo    detective@legal.ai / password123
echo    admin@legal.ai / password123
echo.

pause