# Start Enhanced RAG System
Write-Host '🚀 Starting Enhanced RAG System...' -ForegroundColor Green
Write-Host '=================================' -ForegroundColor Green

# Check if Docker services are running
Write-Host ""
Write-Host "Checking Docker services..." -ForegroundColor Cyan
$runningContainers = docker ps --format "table {{.Names}}" | Select-String "legal-ai"

if ($runningContainers.Count -lt 5) {
    Write-Host "Starting Docker services..." -ForegroundColor Yellow
    docker-compose up -d
    
    # Wait for services to be ready
    Write-Host "Waiting for services to be healthy..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
} else {
    Write-Host "✓ Docker services are already running" -ForegroundColor Green
}

# Check Ollama
Write-Host ""
Write-Host "Checking Ollama..." -ForegroundColor Cyan
$ollamaRunning = Get-Process "ollama" -ErrorAction SilentlyContinue

if (!$ollamaRunning) {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
} else {
    Write-Host "✓ Ollama is already running" -ForegroundColor Green
}

# Check for models
Write-Host ""
Write-Host "Checking AI models..." -ForegroundColor Cyan
$models = ollama list 2>$null

if ($models -notmatch "gemma3legal") {
    Write-Host "⚠ gemma3legal model not found. Trying alternatives..." -ForegroundColor Yellow
    
    # Try to pull gemma3legal
    ollama pull gemma3legal:latest 2>$null
    
    if ($LASTEXITCODE -ne 0) {
        # Fallback to gemma:2b
        Write-Host "Falling back to gemma:2b model..." -ForegroundColor Yellow
        ollama pull gemma:2b
    }
}

if ($models -notmatch "nomic-embed-text") {
    Write-Host "Pulling nomic-embed-text model..." -ForegroundColor Yellow
    ollama pull nomic-embed-text
}

Write-Host "✓ AI models ready" -ForegroundColor Green

# Start the application
Write-Host ""
Write-Host "Starting SvelteKit application..." -ForegroundColor Cyan

# Check if node_modules exists
if (!(Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
}

# Set environment variables
$env:NODE_ENV = "development"
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/legal_ai_db?sslmode=disable"
$env:REDIS_URL = "redis://localhost:6379"
$env:NEO4J_URI = "neo4j://localhost:7687"
$env:MINIO_ENDPOINT = "localhost:9000"
$env:OLLAMA_URL = "http://localhost:11434"

# Display service URLs
Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "✅ System is running!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Cyan
Write-Host "  • Application: http://localhost:3000" -ForegroundColor White
Write-Host "  • MinIO Console: http://localhost:9001" -ForegroundColor White
Write-Host "  • RabbitMQ Management: http://localhost:15672" -ForegroundColor White
Write-Host "  • Neo4j Browser: http://localhost:7474" -ForegroundColor White
Write-Host ""
Write-Host "Default Credentials:" -ForegroundColor Cyan
Write-Host "  • MinIO: minioadmin / minioadmin123" -ForegroundColor White
Write-Host "  • RabbitMQ: guest / guest" -ForegroundColor White
Write-Host "  • Neo4j: neo4j / password" -ForegroundColor White
Write-Host "  • PostgreSQL: postgres / postgres" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

# Start the dev server
npm run dev
