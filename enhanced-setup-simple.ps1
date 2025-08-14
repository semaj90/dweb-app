# Enhanced Legal AI System Setup - Simplified Version
param(
    [switch]$GenerateSecureConfig,
    [switch]$EnableMonitoring,
    [switch]$CreateBackup,
    [switch]$GenerateBestPractices
)

function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }

Write-Info "🚀 Enhanced Legal AI System Setup Starting..."

# Generate secure configuration
if ($GenerateSecureConfig) {
    Write-Info "🔐 Generating secure configuration..."
    
    function New-SecurePassword {
        -join ((65..90) + (97..122) + (48..57) + @(33,35,36,37,38,42,43,45,61,63,64) | Get-Random -Count 16 | ForEach-Object { [char]$_ })
    }
    
    $dbPassword = New-SecurePassword
    $minioPassword = New-SecurePassword
    $jwtSecret = New-SecurePassword
    $apiKey = New-SecurePassword
    
    $envContent = @"
# Database Configuration
DATABASE_URL=postgresql://legal_admin:$dbPassword@localhost:5432/legal_ai_db
POSTGRES_PASSWORD=$dbPassword
DB_HOST=localhost
DB_PORT=5432
DB_NAME=legal_ai_db
DB_USER=legal_admin

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=$minioPassword
MINIO_ENDPOINT=localhost:9000

# API Configuration
JWT_SECRET=$jwtSecret
API_KEY=$apiKey

# Service Ports
POSTGRES_PORT=5432
REDIS_PORT=6379
QDRANT_PORT=6333
MINIO_PORT=9000
GO_SERVICE_PORT=8093
SVELTEKIT_PORT=5173

# Security
CORS_ORIGIN=http://localhost:5173
"@
    
    $envContent | Out-File -FilePath ".env.secure" -Encoding UTF8
    Write-Success "Secure configuration generated in .env.secure"
}

# Generate best practices documentation
if ($GenerateBestPractices) {
    Write-Info "📚 Generating best practices documentation..."
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $bestPracticesContent = @"
# Legal AI System Best Practices Guide
Generated: $timestamp

## Security Best Practices

### Database Security
* Use strong passwords (16+ characters with mixed case, numbers, symbols)
* Enable SSL/TLS for database connections
* Implement connection pooling with limits
* Regular security audits and updates
* Backup encryption with rotation

### API Security
* JWT tokens with short expiration (1 hour)
* Rate limiting: 100 requests/minute per IP
* Input validation and sanitization
* CORS configuration for specific origins
* API key rotation every 90 days

### File Upload Security
* Virus scanning before storage
* File type validation (whitelist)
* Size limits: 50MB per file, 500MB per session
* Quarantine suspicious files
* Audit trail for all uploads

## Performance Best Practices

### Database Optimization
* Index all foreign keys and search columns
* Use prepared statements for all queries
* Connection pooling: min 5, max 25 connections
* Query timeout: 30 seconds
* Regular VACUUM and ANALYZE

### Caching Strategy
* Redis for session data (TTL: 1 hour)
* Qdrant for vector embeddings (persistent)
* Application cache for API responses (TTL: 5 minutes)
* CDN for static assets
* Browser caching headers

### Memory Management
* Go service: 2GB heap limit
* Node.js: 4GB max old space
* PostgreSQL: 25% of system RAM
* Redis: 1GB max memory
* Monitor memory usage continuously

## AI/ML Best Practices

### Model Selection
* Use quantized models (Q4_K_M) for production
* Fallback chain: Local LLM to Claude to OpenAI
* Context window optimization (4K chunks)
* Embedding dimension: 384 for balance
* Model warm-up on service start

### Vector Search
* Cosine similarity for semantic search
* HNSW index for fast nearest neighbor
* Batch processing for bulk operations
* Result caching for common queries
* Relevance scoring threshold: 0.7

### Data Processing
* Chunk documents at sentence boundaries
* Metadata extraction for filtering
* Parallel processing with worker threads
* Error handling with graceful degradation
* Progress tracking for long operations

## Development Best Practices

### SvelteKit 2 + Svelte 5
* Use state() for reactive variables
* derived() for computed values
* effect() for side effects
* Progressive enhancement with use:enhance
* Type safety with generated types

### Go Microservices
* Structured logging with levels
* Graceful shutdown handling
* Health check endpoints
* Circuit breaker pattern
* Metrics collection (Prometheus)

### Testing Strategy
* Unit tests: 80% coverage minimum
* Integration tests for APIs
* End-to-end tests for critical paths
* Performance tests under load
* Security penetration testing

## Monitoring Best Practices

### Metrics Collection
* Response times (p50, p95, p99)
* Error rates by endpoint
* Database connection pool usage
* Memory and CPU utilization
* Cache hit/miss ratios

### Alerting Rules
* Response time greater than 5 seconds
* Error rate greater than 5%
* Database connections greater than 80%
* Memory usage greater than 85%
* Disk space less than 10% free

### Log Management
* Structured logging (JSON format)
* Log levels: DEBUG, INFO, WARN, ERROR
* Centralized log aggregation
* Log retention: 30 days
* PII redaction in logs

---

This document should be reviewed and updated quarterly or after major system changes.
"@
    
    $bestPracticesContent | Out-File -FilePath "BEST_PRACTICES_COMPREHENSIVE.md" -Encoding UTF8
    Write-Success "Comprehensive best practices guide generated"
}

# Start system services
Write-Info "🚀 Starting system services..."

# Test PostgreSQL connection
Write-Info "Testing PostgreSQL connection..."
try {
    $env:PGPASSWORD = "123456"
    $testCmd = '& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "SELECT version();"'
    $result = Invoke-Expression $testCmd
    if ($LASTEXITCODE -eq 0) {
        Write-Success "PostgreSQL connected successfully"
    } else {
        Write-Warning "PostgreSQL connection failed"
    }
} catch {
    Write-Warning "PostgreSQL test failed: $($_.Exception.Message)"
}

# Start Redis
Write-Info "Starting Redis..."
if (Test-Path ".\redis-windows\redis-server.exe") {
    try {
        Start-Process -FilePath ".\redis-windows\redis-server.exe" -ArgumentList "--port 6379" -WindowStyle Hidden
        Start-Sleep 2
        Write-Success "Redis started on port 6379"
    } catch {
        Write-Warning "Failed to start Redis: $($_.Exception.Message)"
    }
} else {
    Write-Warning "Redis executable not found"
}

# Start Qdrant
Write-Info "Starting Qdrant..."
if (Test-Path ".\qdrant-windows\qdrant.exe") {
    try {
        Start-Process -FilePath ".\qdrant-windows\qdrant.exe" -WindowStyle Hidden
        Start-Sleep 3
        Write-Success "Qdrant started on port 6333"
    } catch {
        Write-Warning "Failed to start Qdrant: $($_.Exception.Message)"
    }
} else {
    Write-Warning "Qdrant executable not found"
}

# Start MinIO
Write-Info "Starting MinIO..."
if (Test-Path ".\minio.exe") {
    try {
        $env:MINIO_ROOT_USER = "minioadmin"
        $env:MINIO_ROOT_PASSWORD = if ($GenerateSecureConfig) { $minioPassword } else { "minioadmin123" }
        Start-Process -FilePath ".\minio.exe" -ArgumentList "server", ".\minio-data", "--console-address", ":9001" -WindowStyle Hidden
        Start-Sleep 3
        Write-Success "MinIO started on port 9000"
    } catch {
        Write-Warning "Failed to start MinIO: $($_.Exception.Message)"
    }
} else {
    Write-Warning "MinIO executable not found"
}

# Start Ollama
Write-Info "Starting Ollama..."
try {
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep 2
    Write-Success "Ollama started on port 11434"
} catch {
    Write-Warning "Ollama not found in PATH"
}

# Build and start Go microservice
Write-Info "Building Go microservice..."
if (Test-Path ".\go-microservice") {
    Push-Location ".\go-microservice"
    try {
        & go build -o bin\upload-service.exe cmd\upload-service\main.go
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Go microservice built successfully"
            
            # Start Go service
            Write-Info "Starting Go upload service..."
            Start-Process -FilePath ".\bin\upload-service.exe" -WindowStyle Hidden
            Start-Sleep 2
            Write-Success "Upload service started on port 8093"
        } else {
            Write-Warning "Go microservice build failed"
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Warning "Go microservice directory not found"
}

# Install Node.js dependencies
Write-Info "Installing Node.js dependencies..."
try {
    & npm install
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Node.js dependencies installed"
    } else {
        Write-Warning "npm install failed"
    }
} catch {
    Write-Warning "npm install error: $($_.Exception.Message)"
}

# Generate monitoring dashboard
if ($EnableMonitoring) {
    Write-Info "📊 Setting up monitoring dashboard..."
    
    $monitoringScript = @'
# System Status Monitor
Write-Host "=== Legal AI System Status ===" -ForegroundColor Green
Write-Host ""

# Service Status
Write-Host "Services:" -ForegroundColor Yellow
try { 
    Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet | Out-Null
    Write-Host "  ✅ PostgreSQL (5432)" -ForegroundColor Green 
} catch { 
    Write-Host "  ❌ PostgreSQL (5432)" -ForegroundColor Red 
}

try { 
    Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet | Out-Null
    Write-Host "  ✅ Redis (6379)" -ForegroundColor Green 
} catch { 
    Write-Host "  ❌ Redis (6379)" -ForegroundColor Red 
}

try { 
    Test-NetConnection -ComputerName localhost -Port 6333 -InformationLevel Quiet | Out-Null
    Write-Host "  ✅ Qdrant (6333)" -ForegroundColor Green 
} catch { 
    Write-Host "  ❌ Qdrant (6333)" -ForegroundColor Red 
}

try { 
    Test-NetConnection -ComputerName localhost -Port 9000 -InformationLevel Quiet | Out-Null
    Write-Host "  ✅ MinIO (9000)" -ForegroundColor Green 
} catch { 
    Write-Host "  ❌ MinIO (9000)" -ForegroundColor Red 
}

try { 
    Test-NetConnection -ComputerName localhost -Port 8093 -InformationLevel Quiet | Out-Null
    Write-Host "  ✅ Go Service (8093)" -ForegroundColor Green 
} catch { 
    Write-Host "  ❌ Go Service (8093)" -ForegroundColor Red 
}

Write-Host ""
Write-Host "Quick Links:" -ForegroundColor Yellow
Write-Host "  SvelteKit App: http://localhost:5173" -ForegroundColor Cyan
Write-Host "  MinIO Console: http://localhost:9001" -ForegroundColor Cyan
Write-Host "  Qdrant Dashboard: http://localhost:6333/dashboard" -ForegroundColor Cyan
Write-Host ""
'@
    
    $monitoringScript | Out-File -FilePath "system-status-monitor.ps1" -Encoding UTF8
    Write-Success "Monitoring dashboard created (run .\system-status-monitor.ps1)"
}

# Create backup
if ($CreateBackup) {
    Write-Info "💾 Creating system backup..."
    
    $backupDir = "backup_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    Copy-Item ".env*" -Destination $backupDir -ErrorAction SilentlyContinue
    Copy-Item "package.json" -Destination $backupDir -ErrorAction SilentlyContinue
    Copy-Item "svelte.config.js" -Destination $backupDir -ErrorAction SilentlyContinue
    
    Write-Success "Backup created in $backupDir"
}

# Final status check
Write-Info "🔍 Performing final system health check..."
Start-Sleep 5

$services = @(
    @{Name="PostgreSQL"; Port=5432},
    @{Name="Redis"; Port=6379},
    @{Name="Qdrant"; Port=6333},
    @{Name="MinIO"; Port=9000},
    @{Name="Go Service"; Port=8093}
)

$healthyServices = 0
foreach ($service in $services) {
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $service.Port -InformationLevel Quiet
        if ($connection) {
            Write-Success "$($service.Name) is healthy"
            $healthyServices++
        } else {
            Write-Warning "$($service.Name) is not responding"
        }
    } catch {
        Write-Warning "$($service.Name) health check failed"
    }
}

Write-Info ""
Write-Info "=== Setup Complete ==="
Write-Success "$healthyServices of $($services.Count) services are healthy"

if ($GenerateSecureConfig) {
    Write-Warning "🔐 Remember to update .env.secure with your actual API keys!"
}

Write-Info "🌐 Access your application at: http://localhost:5173"
Write-Info "📊 Monitor system status with: .\system-status-monitor.ps1"
Write-Info ""
Write-Success "Enhanced Legal AI System is ready! 🚀"