# Quick PostgreSQL Setup for Enhanced RAG System
param([switch]$UseDocker = $true)

Write-Host "Setting up PostgreSQL for Enhanced RAG System..." -ForegroundColor Cyan

if ($UseDocker) {
    Write-Host "Using Docker PostgreSQL setup..." -ForegroundColor Yellow
    
    # Stop and remove existing container
    docker stop postgres 2>$null | Out-Null
    docker rm postgres 2>$null | Out-Null
    
    # Start PostgreSQL with pgvector
    Write-Host "Starting PostgreSQL container..." -ForegroundColor Green
    docker run -d --name postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=legal_ai_db -p 5432:5432 pgvector/pgvector:pg15
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PostgreSQL container started successfully!" -ForegroundColor Green
        Write-Host "Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
        
        # Wait for container to be ready
        $attempts = 0
        do {
            Start-Sleep -Seconds 3
            $attempts++
            $ready = docker exec postgres pg_isready -U postgres 2>$null
            Write-Host "Attempt $attempts - checking readiness..." -ForegroundColor Gray
        } while (-not ($ready -match "accepting connections") -and $attempts -lt 20)
        
        if ($ready -match "accepting connections") {
            Write-Host "PostgreSQL is ready!" -ForegroundColor Green
            
            # Enable pgvector extension
            Write-Host "Enabling pgvector extension..." -ForegroundColor Cyan
            docker exec postgres psql -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
            docker exec postgres psql -U postgres -d legal_ai_db -c "SELECT 1 as setup_complete;"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Database setup completed successfully!" -ForegroundColor Green
                Write-Host "Connection string: postgresql://postgres:password@localhost:5432/legal_ai_db" -ForegroundColor White
            } else {
                Write-Host "Warning: Extension setup may have failed" -ForegroundColor Yellow
            }
        } else {
            Write-Host "PostgreSQL failed to start properly" -ForegroundColor Red
        }
    } else {
        Write-Host "Failed to start PostgreSQL container" -ForegroundColor Red
    }
} else {
    Write-Host "Native PostgreSQL setup not implemented in quick setup" -ForegroundColor Yellow
    Write-Host "Please ensure PostgreSQL is running locally with pgvector extension" -ForegroundColor White
}

Write-Host "Setup completed!" -ForegroundColor Green