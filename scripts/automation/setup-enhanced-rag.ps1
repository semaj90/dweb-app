# Enhanced RAG System Setup Script
# Sets up PostgreSQL, Redis, Ollama, and Enhanced RAG Backend

Write-Host "🚀 Setting up Enhanced RAG System..." -ForegroundColor Cyan

# Create necessary directories
$directories = @(
    "rag-backend/logs",
    "rag-backend/uploads",
    "rag-backend/init-db"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -Path $dir -ItemType Directory -Force
        Write-Host "✅ Created directory: $dir" -ForegroundColor Green
    }
}

# Create PostgreSQL initialization script
$pgInitScript = @"
-- Enhanced RAG Database Initialization
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enhanced schema
CREATE SCHEMA IF NOT EXISTS rag;

-- Set search path
SET search_path TO rag, public;

-- Create tables (if not exists)
CREATE TABLE IF NOT EXISTS rag_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    content TEXT,
    document_type VARCHAR(50) DEFAULT 'general',
    file_path VARCHAR(500),
    file_size INTEGER,
    mime_type VARCHAR(100),
    case_id VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES rag_documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rag_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    query_type VARCHAR(20) DEFAULT 'hybrid',
    results_count INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rag_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    input_data JSONB,
    result_data JSONB,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_rag_documents_type ON rag_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_rag_documents_case ON rag_documents(case_id);
CREATE INDEX IF NOT EXISTS idx_rag_documents_created ON rag_documents(created_at);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_document ON rag_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_rag_queries_created ON rag_queries(created_at);
CREATE INDEX IF NOT EXISTS idx_rag_jobs_status ON rag_jobs(status);
CREATE INDEX IF NOT EXISTS idx_rag_jobs_type ON rag_jobs(job_type);

-- Create vector similarity search indexes
CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding ON rag_documents 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding ON rag_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA rag TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA rag TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA rag TO postgres;

ANALYZE;

-- Log setup completion
INSERT INTO rag_jobs (job_type, status, result_data) VALUES 
('database_setup', 'completed', '{"message": "Enhanced RAG database initialized successfully"}');

SELECT 'Enhanced RAG Database Setup Complete!' as status;
"@

$pgInitScript | Out-File -FilePath "rag-backend/init-db/01-enhanced-rag-setup.sql" -Encoding UTF8
Write-Host "✅ Created PostgreSQL initialization script" -ForegroundColor Green

# Create .env file for Enhanced RAG Backend
$envContent = @"
# Enhanced RAG Backend Configuration
NODE_ENV=development
PORT=8000

# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/deeds_web_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=deeds_web_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379

# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:9b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=30000

# CORS Configuration
CORS_ORIGIN=http://localhost:5173
CORS_CREDENTIALS=true

# Logging Configuration
LOG_LEVEL=info
LOG_FILE=logs/rag-backend.log

# Performance Configuration
MAX_CONCURRENT_JOBS=5
MAX_UPLOAD_SIZE=50MB
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Security Configuration
JWT_SECRET=your-jwt-secret-key-here
API_KEY_REQUIRED=false

# Feature Flags
ENABLE_METRICS=true
ENABLE_CACHING=true
ENABLE_QUEUE=true
ENABLE_WEBSOCKET=true

# Storage Configuration
UPLOAD_DIR=uploads
TEMP_DIR=temp
MAX_FILE_SIZE=52428800

# Vector Search Configuration
VECTOR_DIMENSIONS=1536
SIMILARITY_THRESHOLD=0.7
MAX_SEARCH_RESULTS=20

# Agent Configuration
ENABLE_MULTI_AGENT=true
AGENT_TIMEOUT=60000
MAX_AGENT_RETRIES=3
"@

$envContent | Out-File -FilePath "rag-backend/.env" -Encoding UTF8
Write-Host "✅ Created .env configuration file" -ForegroundColor Green

# Check Docker installation
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker not found. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if Docker Compose file exists
if (!(Test-Path "rag-backend/docker-compose.yml")) {
    Write-Host "❌ docker-compose.yml not found in rag-backend directory" -ForegroundColor Red
    exit 1
}

# Start Enhanced RAG services
Write-Host "🐳 Starting Enhanced RAG services with Docker Compose..." -ForegroundColor Cyan
Push-Location "rag-backend"

try {
    # Pull latest images
    docker-compose pull
    
    # Start services in background
    docker-compose up -d
    
    Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    
    # Check service health
    $services = @("postgres", "redis", "ollama")
    foreach ($service in $services) {
        $status = docker-compose ps $service --format "table {{.Service}}\t{{.Status}}"
        Write-Host "📊 $service status: $status" -ForegroundColor Cyan
    }
    
    # Wait for PostgreSQL to be ready
    Write-Host "⏳ Waiting for PostgreSQL to initialize..." -ForegroundColor Yellow
    $timeout = 60
    $counter = 0
    do {
        Start-Sleep -Seconds 2
        $counter += 2
        try {
            $pgReady = docker-compose exec -T postgres pg_isready -U postgres -d deeds_web_db
            if ($pgReady -match "accepting connections") {
                Write-Host "✅ PostgreSQL is ready!" -ForegroundColor Green
                break
            }
        } catch {
            # Continue waiting
        }
        
        if ($counter -ge $timeout) {
            Write-Host "⚠️ PostgreSQL startup timeout, but continuing..." -ForegroundColor Yellow
            break
        }
    } while ($true)
    
    # Check if Ollama models need to be pulled
    Write-Host "🤖 Checking Ollama models..." -ForegroundColor Cyan
    try {
        docker-compose exec -T ollama ollama list
        
        # Pull required models if they don't exist
        Write-Host "📥 Pulling required Ollama models..." -ForegroundColor Cyan
        docker-compose exec -T ollama ollama pull gemma2:9b
        docker-compose exec -T ollama ollama pull nomic-embed-text
        Write-Host "✅ Ollama models ready!" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Could not connect to Ollama, but service may still be starting..." -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ Error starting services: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

# Install Enhanced RAG Backend dependencies
if (Test-Path "rag-backend/package.json") {
    Write-Host "📦 Installing Enhanced RAG Backend dependencies..." -ForegroundColor Cyan
    Push-Location "rag-backend"
    try {
        npm install
        Write-Host "✅ Enhanced RAG Backend dependencies installed!" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Could not install dependencies. Please run 'npm install' manually in rag-backend directory." -ForegroundColor Yellow
    } finally {
        Pop-Location
    }
}

# Test Enhanced RAG Backend connectivity
Write-Host "🔍 Testing Enhanced RAG Backend connectivity..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 10
    if ($healthResponse.status -eq "healthy") {
        Write-Host "✅ Enhanced RAG Backend is healthy!" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Enhanced RAG Backend responded but status: $($healthResponse.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Could not connect to Enhanced RAG Backend yet. It may still be starting..." -ForegroundColor Yellow
    Write-Host "   Try running: docker-compose -f rag-backend/docker-compose.yml up -d" -ForegroundColor Gray
}

# VS Code Extension setup
Write-Host "🔧 Setting up VS Code Extension..." -ForegroundColor Cyan
$extensionPath = ".vscode/extensions/mcp-context7-assistant"

if (Test-Path $extensionPath) {
    Push-Location $extensionPath
    try {
        if (Test-Path "package.json") {
            npm install
            npm run compile
            Write-Host "✅ VS Code Extension compiled successfully!" -ForegroundColor Green
        }
    } catch {
        Write-Host "⚠️ Could not compile VS Code Extension. Please check the logs." -ForegroundColor Yellow
    } finally {
        Pop-Location
    }
} else {
    Write-Host "⚠️ VS Code Extension not found at $extensionPath" -ForegroundColor Yellow
}

# Create startup script
$startupScript = @"
@echo off
echo Starting Enhanced RAG System...

cd rag-backend
echo Starting services...
docker-compose up -d

echo Waiting for services...
timeout /t 10

echo Starting Enhanced RAG Backend...
start cmd /k "npm run start"

echo Enhanced RAG System started!
echo.
echo Services:
echo - PostgreSQL: http://localhost:5432
echo - Redis: http://localhost:6379  
echo - Ollama: http://localhost:11434
echo - Enhanced RAG Backend: http://localhost:8000
echo - SvelteKit Frontend: http://localhost:5173
echo.
echo Press any key to continue...
pause
"@

$startupScript | Out-File -FilePath "start-enhanced-rag.bat" -Encoding ASCII
Write-Host "✅ Created startup script: start-enhanced-rag.bat" -ForegroundColor Green

# Final summary
Write-Host ""
Write-Host "🎉 Enhanced RAG System Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Start the Enhanced RAG Backend:" -ForegroundColor White
Write-Host "   cd rag-backend && npm run start" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start SvelteKit Frontend:" -ForegroundColor White
Write-Host "   cd sveltekit-frontend && npm run dev" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Test the integration:" -ForegroundColor White
Write-Host "   Open VS Code and use Command Palette: 'RAG: Search Documents'" -ForegroundColor Gray
Write-Host ""
Write-Host "🌐 Service URLs:" -ForegroundColor Cyan
Write-Host "- Enhanced RAG Backend: http://localhost:8000" -ForegroundColor White
Write-Host "- SvelteKit Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "- PostgreSQL: localhost:5432" -ForegroundColor White
Write-Host "- Redis: localhost:6379" -ForegroundColor White
Write-Host "- Ollama: http://localhost:11434" -ForegroundColor White
Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "- API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "- Health Check: http://localhost:8000/health" -ForegroundColor White
Write-Host "- RAG Stats: http://localhost:8000/api/v1/rag/stats" -ForegroundColor White
Write-Host ""

# Create test script
$testScript = @"
# Test Enhanced RAG System
Write-Host "🧪 Testing Enhanced RAG System..." -ForegroundColor Cyan

# Test health endpoints
`$endpoints = @(
    "http://localhost:8000/health",
    "http://localhost:8000/api/v1/rag/stats", 
    "http://localhost:5173/api/rag?action=status"
)

foreach (`$endpoint in `$endpoints) {
    try {
        `$response = Invoke-RestMethod -Uri `$endpoint -Method GET -TimeoutSec 5
        Write-Host "✅ `$endpoint - OK" -ForegroundColor Green
    } catch {
        Write-Host "❌ `$endpoint - Failed: `$(`$_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "✅ Enhanced RAG System test complete!" -ForegroundColor Green
"@

$testScript | Out-File -FilePath "test-enhanced-rag.ps1" -Encoding UTF8
Write-Host "✅ Created test script: test-enhanced-rag.ps1" -ForegroundColor Green

Write-Host ""
Write-Host "🚀 Setup completed successfully!" -ForegroundColor Green
Write-Host "Run './start-enhanced-rag.bat' to start all services" -ForegroundColor Cyan