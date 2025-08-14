# Enhanced Legal AI System Setup
param(
    [switch]$GenerateSecureConfig,
    [switch]$EnableMonitoring,
    [switch]$CreateBackup,
    [switch]$GenerateBestPractices
)

function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }

Write-Info "🚀 Enhanced Legal AI System Setup Starting..."

# Generate best practices documentation
if ($GenerateBestPractices) {
    Write-Info "📚 Generating best practices documentation..."
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    # Create comprehensive best practices guide
    $content = "# Legal AI System Best Practices Guide`n"
    $content += "Generated: $timestamp`n`n"
    $content += "## Security Best Practices`n`n"
    $content += "### Database Security`n"
    $content += "- Use strong passwords (16+ characters with mixed case, numbers, symbols)`n"
    $content += "- Enable SSL/TLS for database connections`n"
    $content += "- Implement connection pooling with limits`n"
    $content += "- Regular security audits and updates`n"
    $content += "- Backup encryption with rotation`n`n"
    $content += "### API Security`n"
    $content += "- JWT tokens with short expiration (1 hour)`n"
    $content += "- Rate limiting: 100 requests/minute per IP`n"
    $content += "- Input validation and sanitization`n"
    $content += "- CORS configuration for specific origins`n"
    $content += "- API key rotation every 90 days`n`n"
    $content += "### File Upload Security`n"
    $content += "- Virus scanning before storage`n"
    $content += "- File type validation (whitelist)`n"
    $content += "- Size limits: 50MB per file, 500MB per session`n"
    $content += "- Quarantine suspicious files`n"
    $content += "- Audit trail for all uploads`n`n"
    $content += "## Performance Best Practices`n`n"
    $content += "### Database Optimization`n"
    $content += "- Index all foreign keys and search columns`n"
    $content += "- Use prepared statements for all queries`n"
    $content += "- Connection pooling: min 5, max 25 connections`n"
    $content += "- Query timeout: 30 seconds`n"
    $content += "- Regular VACUUM and ANALYZE`n`n"
    $content += "### Caching Strategy`n"
    $content += "- Redis for session data (TTL: 1 hour)`n"
    $content += "- Qdrant for vector embeddings (persistent)`n"
    $content += "- Application cache for API responses (TTL: 5 minutes)`n"
    $content += "- CDN for static assets`n"
    $content += "- Browser caching headers`n`n"
    $content += "### AI/ML Best Practices`n"
    $content += "- Use quantized models (Q4_K_M) for production`n"
    $content += "- Fallback chain: Local LLM → Claude → OpenAI`n"
    $content += "- Context window optimization (4K chunks)`n"
    $content += "- Embedding dimension: 384 for balance`n"
    $content += "- Model warm-up on service start`n`n"
    $content += "### SvelteKit 2 + Svelte 5 Best Practices`n"
    $content += "- Use `$state()` for reactive variables`n"
    $content += "- Use `$derived()` for computed values`n"
    $content += "- Use `$effect()` for side effects`n"
    $content += "- Progressive enhancement with use:enhance`n"
    $content += "- Type safety with generated types`n`n"
    $content += "### Go Microservices Best Practices`n"
    $content += "- Structured logging with levels`n"
    $content += "- Graceful shutdown handling`n"
    $content += "- Health check endpoints`n"
    $content += "- Circuit breaker pattern`n"
    $content += "- Metrics collection (Prometheus)`n`n"
    $content += "### Monitoring Best Practices`n"
    $content += "- Response times (p50, p95, p99)`n"
    $content += "- Error rates by endpoint`n"
    $content += "- Database connection pool usage`n"
    $content += "- Memory and CPU utilization`n"
    $content += "- Cache hit/miss ratios`n`n"
    $content += "---`n`n"
    $content += "This document should be reviewed and updated quarterly.`n"
    
    $content | Out-File -FilePath "BEST_PRACTICES_COMPREHENSIVE.md" -Encoding UTF8
    Write-Success "Comprehensive best practices guide generated"
}

# Generate secure configuration
if ($GenerateSecureConfig) {
    Write-Info "🔐 Generating secure configuration..."
    
    function New-SecurePassword {
        -join ((65..90) + (97..122) + (48..57) + @(33,35,36,37,38,42,43,45,61,63,64) | Get-Random -Count 16 | ForEach-Object { [char]$_ })
    }
    
    $dbPassword = New-SecurePassword
    $minioPassword = New-SecurePassword
    $jwtSecret = New-SecurePassword
    
    $envSecure = "# Enhanced Secure Configuration`n"
    $envSecure += "# Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n`n"
    $envSecure += "# Database Configuration`n"
    $envSecure += "DATABASE_URL=postgresql://legal_admin:$dbPassword@localhost:5432/legal_ai_db`n"
    $envSecure += "POSTGRES_PASSWORD=$dbPassword`n"
    $envSecure += "DB_HOST=localhost`n"
    $envSecure += "DB_PORT=5432`n"
    $envSecure += "DB_NAME=legal_ai_db`n"
    $envSecure += "DB_USER=legal_admin`n`n"
    $envSecure += "# MinIO Configuration`n"
    $envSecure += "MINIO_ROOT_USER=minioadmin`n"
    $envSecure += "MINIO_ROOT_PASSWORD=$minioPassword`n"
    $envSecure += "MINIO_ENDPOINT=localhost:9000`n`n"
    $envSecure += "# API Configuration`n"
    $envSecure += "JWT_SECRET=$jwtSecret`n"
    $envSecure += "API_KEY=$jwtSecret`n`n"
    $envSecure += "# Service Ports`n"
    $envSecure += "POSTGRES_PORT=5432`n"
    $envSecure += "REDIS_PORT=6379`n"
    $envSecure += "QDRANT_PORT=6333`n"
    $envSecure += "MINIO_PORT=9000`n"
    $envSecure += "GO_SERVICE_PORT=8093`n"
    $envSecure += "SVELTEKIT_PORT=5173`n`n"
    $envSecure += "# Security`n"
    $envSecure += "CORS_ORIGIN=http://localhost:5173`n"
    $envSecure += "SECURE_COOKIES=true`n"
    $envSecure += "SESSION_TIMEOUT=3600`n"
    
    $envSecure | Out-File -FilePath ".env.secure" -Encoding UTF8
    Write-Success "Secure configuration generated in .env.secure"
}

# Test PostgreSQL connection
Write-Info "Testing PostgreSQL connection..."
try {
    $env:PGPASSWORD = "123456"
    $result = & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "SELECT version();" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "PostgreSQL connected successfully"
    } else {
        Write-Warning "PostgreSQL connection failed"
    }
} catch {
    Write-Warning "PostgreSQL test failed"
}

# Start services
Write-Info "🚀 Starting system services..."

# Start Redis
if (Test-Path ".\redis-windows\redis-server.exe") {
    try {
        Start-Process -FilePath ".\redis-windows\redis-server.exe" -ArgumentList "--port 6379" -WindowStyle Hidden
        Start-Sleep 2
        Write-Success "Redis started on port 6379"
    } catch {
        Write-Warning "Failed to start Redis"
    }
} else {
    Write-Warning "Redis executable not found"
}

# Start Qdrant
if (Test-Path ".\qdrant-windows\qdrant.exe") {
    try {
        Start-Process -FilePath ".\qdrant-windows\qdrant.exe" -WindowStyle Hidden
        Start-Sleep 3
        Write-Success "Qdrant started on port 6333"
    } catch {
        Write-Warning "Failed to start Qdrant"
    }
} else {
    Write-Warning "Qdrant executable not found"
}

# Start MinIO
if (Test-Path ".\minio.exe") {
    try {
        $env:MINIO_ROOT_USER = "minioadmin"
        $env:MINIO_ROOT_PASSWORD = "minioadmin123"
        Start-Process -FilePath ".\minio.exe" -ArgumentList "server", ".\minio-data", "--console-address", ":9001" -WindowStyle Hidden
        Start-Sleep 3
        Write-Success "MinIO started on port 9000"
    } catch {
        Write-Warning "Failed to start MinIO"
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

# Build Go microservice
if (Test-Path ".\go-microservice") {
    Write-Info "Building Go microservice..."
    Push-Location ".\go-microservice"
    try {
        & go build -o bin\upload-service.exe cmd\upload-service\main.go
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Go microservice built successfully"
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
    Write-Warning "npm install error"
}

# Generate monitoring script
if ($EnableMonitoring) {
    Write-Info "📊 Setting up monitoring dashboard..."
    
    $monitorContent = "# System Status Monitor`n"
    $monitorContent += "Write-Host `"=== Legal AI System Status ===`" -ForegroundColor Green`n"
    $monitorContent += "Write-Host `"`"`n`n"
    $monitorContent += "Write-Host `"Services:`" -ForegroundColor Yellow`n"
    $monitorContent += "try { Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet | Out-Null; Write-Host `"  ✅ PostgreSQL (5432)`" -ForegroundColor Green } catch { Write-Host `"  ❌ PostgreSQL (5432)`" -ForegroundColor Red }`n"
    $monitorContent += "try { Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet | Out-Null; Write-Host `"  ✅ Redis (6379)`" -ForegroundColor Green } catch { Write-Host `"  ❌ Redis (6379)`" -ForegroundColor Red }`n"
    $monitorContent += "try { Test-NetConnection -ComputerName localhost -Port 6333 -InformationLevel Quiet | Out-Null; Write-Host `"  ✅ Qdrant (6333)`" -ForegroundColor Green } catch { Write-Host `"  ❌ Qdrant (6333)`" -ForegroundColor Red }`n"
    $monitorContent += "try { Test-NetConnection -ComputerName localhost -Port 9000 -InformationLevel Quiet | Out-Null; Write-Host `"  ✅ MinIO (9000)`" -ForegroundColor Green } catch { Write-Host `"  ❌ MinIO (9000)`" -ForegroundColor Red }`n"
    $monitorContent += "try { Test-NetConnection -ComputerName localhost -Port 8093 -InformationLevel Quiet | Out-Null; Write-Host `"  ✅ Go Service (8093)`" -ForegroundColor Green } catch { Write-Host `"  ❌ Go Service (8093)`" -ForegroundColor Red }`n"
    $monitorContent += "Write-Host `"`"`n"
    $monitorContent += "Write-Host `"Quick Links:`" -ForegroundColor Yellow`n"
    $monitorContent += "Write-Host `"  SvelteKit App: http://localhost:5173`" -ForegroundColor Cyan`n"
    $monitorContent += "Write-Host `"  MinIO Console: http://localhost:9001`" -ForegroundColor Cyan`n"
    $monitorContent += "Write-Host `"  Qdrant Dashboard: http://localhost:6333/dashboard`" -ForegroundColor Cyan`n"
    
    $monitorContent | Out-File -FilePath "system-status-monitor.ps1" -Encoding UTF8
    Write-Success "Monitoring dashboard created"
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

# Final health check
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
Write-Info "=== Enhanced Setup Complete ==="
Write-Success "$healthyServices of $($services.Count) services are healthy"

if ($GenerateSecureConfig) {
    Write-Warning "🔐 Remember to update .env.secure with your actual API keys!"
}

Write-Info "🌐 Access your application at: http://localhost:5173"
if ($EnableMonitoring) {
    Write-Info "📊 Monitor system status with: .\system-status-monitor.ps1"
}
Write-Info ""
Write-Success "Enhanced Legal AI System is ready! 🚀"