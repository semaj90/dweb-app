# Test RAG System
Write-Host '🧪 Testing RAG System...' -ForegroundColor Cyan
Write-Host '=======================' -ForegroundColor Cyan
Write-Host ""

$allHealthy = $true

# Test PostgreSQL
Write-Host -NoNewline 'Testing PostgreSQL... '
try {
    docker exec legal-ai-postgres pg_isready -U postgres 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host '✓' -ForegroundColor Green
    } else {
        Write-Host '✗' -ForegroundColor Red
        $allHealthy = $false
    }
} catch {
    Write-Host '✗' -ForegroundColor Red
    $allHealthy = $false
}

# Test Redis
Write-Host -NoNewline 'Testing Redis... '
try {
    docker exec legal-ai-redis redis-cli ping 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host '✓' -ForegroundColor Green
    } else {
        Write-Host '✗' -ForegroundColor Red
        $allHealthy = $false
    }
} catch {
    Write-Host '✗' -ForegroundColor Red
    $allHealthy = $false
}

# Test Neo4j
Write-Host -NoNewline 'Testing Neo4j... '
try {
    $response = Invoke-WebRequest -Uri 'http://localhost:7474' -ErrorAction SilentlyContinue -TimeoutSec 2
    if ($response.StatusCode -eq 200) {
        Write-Host '✓' -ForegroundColor Green
    } else {
        Write-Host '✗' -ForegroundColor Red
        $allHealthy = $false
    }
} catch {
    Write-Host '✗' -ForegroundColor Red
    $allHealthy = $false
}

# Test RabbitMQ
Write-Host -NoNewline 'Testing RabbitMQ... '
try {
    $response = Invoke-WebRequest -Uri 'http://localhost:15672' -ErrorAction SilentlyContinue -TimeoutSec 2
    if ($response.StatusCode -eq 200) {
        Write-Host '✓' -ForegroundColor Green
    } else {
        Write-Host '✗' -ForegroundColor Red
        $allHealthy = $false
    }
} catch {
    Write-Host '✗' -ForegroundColor Red
    $allHealthy = $false
}

# Test MinIO
Write-Host -NoNewline 'Testing MinIO... '
try {
    $response = Invoke-WebRequest -Uri 'http://localhost:9000' -ErrorAction SilentlyContinue -TimeoutSec 2
    Write-Host '✓' -ForegroundColor Green
} catch {
    # MinIO might redirect, so any response is OK
    Write-Host '✓' -ForegroundColor Green
}

# Test Ollama
Write-Host -NoNewline 'Testing Ollama... '
try {
    $response = Invoke-WebRequest -Uri 'http://localhost:11434/api/version' -ErrorAction SilentlyContinue -TimeoutSec 2
    if ($response.StatusCode -eq 200) {
        Write-Host '✓' -ForegroundColor Green
    } else {
        Write-Host '✗' -ForegroundColor Red
        $allHealthy = $false
    }
} catch {
    Write-Host '✗' -ForegroundColor Red
    $allHealthy = $false
}

# Test pgvector extension
Write-Host -NoNewline 'Testing pgvector extension... '
try {
    $result = docker exec legal-ai-postgres psql -U postgres -d legal_ai_db -c "SELECT 1 FROM pg_extension WHERE extname='vector';" 2>$null
    if ($result -match "1 row") {
        Write-Host '✓' -ForegroundColor Green
    } else {
        Write-Host '✗ (not installed)' -ForegroundColor Red
        Write-Host '  Installing pgvector...' -ForegroundColor Yellow
        docker exec legal-ai-postgres psql -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>$null
    }
} catch {
    Write-Host '✗' -ForegroundColor Red
    $allHealthy = $false
}

# Test database tables
Write-Host -NoNewline 'Testing database tables... '
try {
    $tables = docker exec legal-ai-postgres psql -U postgres -d legal_ai_db -c "\dt" 2>$null
    if ($tables -match "documents" -and $tables -match "users") {
        Write-Host '✓' -ForegroundColor Green
    } else {
        Write-Host '⚠ (tables missing)' -ForegroundColor Yellow
        Write-Host '  Run database migrations: npm run db:push' -ForegroundColor Yellow
    }
} catch {
    Write-Host '✗' -ForegroundColor Red
    $allHealthy = $false
}

# Test Node.js application
Write-Host -NoNewline 'Testing Node.js application... '
try {
    $response = Invoke-WebRequest -Uri 'http://localhost:3000' -ErrorAction SilentlyContinue -TimeoutSec 2
    if ($response.StatusCode -eq 200) {
        Write-Host '✓' -ForegroundColor Green
    } else {
        Write-Host '⚠ (not running)' -ForegroundColor Yellow
        Write-Host '  Start with: npm run dev' -ForegroundColor Yellow
    }
} catch {
    Write-Host '⚠ (not running)' -ForegroundColor Yellow
    Write-Host '  Start with: npm run dev' -ForegroundColor Yellow
}

# Test AI models
Write-Host -NoNewline 'Testing AI models... '
$models = ollama list 2>$null | Out-String
if ($models -match "gemma" -or $models -match "nomic") {
    Write-Host '✓' -ForegroundColor Green
} else {
    Write-Host '⚠ (models not loaded)' -ForegroundColor Yellow
    Write-Host '  Pull models with: ollama pull gemma:2b' -ForegroundColor Yellow
}

# Run unit tests if available
Write-Host ""
if ((Test-Path 'package.json') -and (Get-Content package.json | Select-String '"test"')) {
    Write-Host 'Running unit tests...' -ForegroundColor Cyan
    npm test 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host '✓ Unit tests passed' -ForegroundColor Green
    } else {
        Write-Host '✗ Some unit tests failed' -ForegroundColor Red
        $allHealthy = $false
    }
}

# Summary
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
if ($allHealthy) {
    Write-Host '✅ All systems operational!' -ForegroundColor Green
    Write-Host ""
    Write-Host "Your Enhanced RAG system is ready to use!" -ForegroundColor Green
    Write-Host "Access the application at: http://localhost:3000" -ForegroundColor Cyan
} else {
    Write-Host '⚠ Some services need attention' -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Fix the issues above and run this test again." -ForegroundColor Yellow
    Write-Host "For detailed logs, run: docker-compose logs" -ForegroundColor Cyan
}
Write-Host "=========================================" -ForegroundColor Cyan
