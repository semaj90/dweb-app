# Native Windows Services Setup Guide
## Complete Production-Ready Installation for Legal AI Platform

This guide provides step-by-step instructions for installing and configuring all native Windows services required for the Legal AI platform.

---

## 🎯 **Overview**

Your Legal AI platform requires these services:
- **PostgreSQL 17** with pgvector extension
- **Redis** for caching and session management  
- **MinIO** for object storage
- **Neo4j** for graph database operations
- **Ollama** for local LLM inference
- **NATS** for real-time messaging
- **RabbitMQ** for message queuing (optional)

---

## 🔧 **Prerequisites**

- Windows 10/11 (64-bit)
- Administrator privileges
- 16GB+ RAM recommended
- 50GB+ free disk space
- PowerShell 5.1 or higher

---

## 📦 **Service Installation**

### 1. PostgreSQL 17 + pgvector

**Download & Install:**
```powershell
# Download PostgreSQL 17 installer
$pgUrl = "https://get.enterprisedb.com/postgresql/postgresql-17.0-1-windows-x64.exe"
Invoke-WebRequest -Uri $pgUrl -OutFile "$env:TEMP\postgresql-17.0-1-windows-x64.exe"

# Run installer (interactive)
Start-Process -FilePath "$env:TEMP\postgresql-17.0-1-windows-x64.exe" -Wait
```

**Installation Steps:**
1. Run the installer as Administrator
2. Set password for `postgres` user: `123456`
3. Port: `5432` (default)
4. Locale: `Default locale`
5. Enable Stack Builder for extensions

**Install pgvector Extension:**
```powershell
# After PostgreSQL installation, run Stack Builder
# Or download pgvector manually:
$pgVectorUrl = "https://github.com/pgvector/pgvector/releases/download/v0.5.1/pgvector-v0.5.1-pg17-windows-x64.zip"
Invoke-WebRequest -Uri $pgVectorUrl -OutFile "$env:TEMP\pgvector.zip"
Expand-Archive -Path "$env:TEMP\pgvector.zip" -DestinationPath "$env:TEMP\pgvector"

# Copy to PostgreSQL directory
Copy-Item "$env:TEMP\pgvector\*" -Destination "C:\Program Files\PostgreSQL\17\lib\" -Force
```

**Configure Database:**
```powershell
# Create database and enable extensions
$env:PGPASSWORD = "123456"
& "C:\Program Files\PostgreSQL\17\bin\createdb.exe" -U postgres legal_ai_db

# Connect and enable extensions
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

### 2. Redis (using Memurai)

**Download & Install Memurai:**
```powershell
# Download Memurai (Redis-compatible for Windows)
$memuriUrl = "https://distrib.memurai.com/installer/memurai-setup-2.0.7.exe"
Invoke-WebRequest -Uri $memuriUrl -OutFile "$env:TEMP\memurai-setup.exe"

# Install Memurai
Start-Process -FilePath "$env:TEMP\memurai-setup.exe" -ArgumentList "/S" -Wait

# Start service
Start-Service -Name "Memurai"
Set-Service -Name "Memurai" -StartupType Automatic
```

**Verify Installation:**
```powershell
# Test Redis connection
& "C:\Program Files\Memurai\memurai-cli.exe" ping
# Should return: PONG
```

### 3. MinIO Object Storage

**Download & Setup:**
```powershell
# Create MinIO directory
New-Item -ItemType Directory -Path "C:\minio" -Force
New-Item -ItemType Directory -Path "C:\minio\data" -Force

# Download MinIO server
$minioUrl = "https://dl.min.io/server/minio/release/windows-amd64/minio.exe"
Invoke-WebRequest -Uri $minioUrl -OutFile "C:\minio\minio.exe"

# Create startup script
@"
@echo off
set MINIO_ROOT_USER=minioadmin
set MINIO_ROOT_PASSWORD=minioadmin123
C:\minio\minio.exe server C:\minio\data --console-address ":9001"
"@ | Out-File -FilePath "C:\minio\start-minio.bat" -Encoding ASCII

# Create Windows service
$serviceName = "MinIO"
$exePath = "C:\minio\start-minio.bat"
$displayName = "MinIO Object Storage"
$description = "MinIO S3-compatible object storage server"

$params = @{
    Name = $serviceName
    BinaryPathName = $exePath
    DisplayName = $displayName
    Description = $description
    StartupType = "Automatic"
}

New-Service @params
Start-Service -Name $serviceName
```

**Configure MinIO:**
```powershell
# Install MinIO client
$mcUrl = "https://dl.min.io/client/mc/release/windows-amd64/mc.exe"
Invoke-WebRequest -Uri $mcUrl -OutFile "C:\minio\mc.exe"

# Wait for service to start
Start-Sleep -Seconds 10

# Configure client
& "C:\minio\mc.exe" alias set local http://localhost:9000 minioadmin minioadmin123

# Create bucket for legal documents
& "C:\minio\mc.exe" mb local/legal-documents
& "C:\minio\mc.exe" mb local/evidence-files
```

### 4. Neo4j Graph Database

**Download & Install:**
```powershell
# Download Neo4j Community Edition
$neo4jUrl = "https://neo4j.com/artifact.php?name=neo4j-community-5.15.0-windows.zip"
Invoke-WebRequest -Uri $neo4jUrl -OutFile "$env:TEMP\neo4j-community.zip"

# Extract to Program Files
Expand-Archive -Path "$env:TEMP\neo4j-community.zip" -DestinationPath "C:\neo4j"

# Install as Windows service
& "C:\neo4j\neo4j-community-5.15.0\bin\neo4j.bat" install-service

# Configure initial password
$env:NEO4J_AUTH = "neo4j/password123"
& "C:\neo4j\neo4j-community-5.15.0\bin\neo4j.bat" start

# Wait for startup
Start-Sleep -Seconds 30

# Set initial password
& "C:\neo4j\neo4j-community-5.15.0\bin\cypher-shell.exe" -u neo4j -p neo4j "ALTER USER neo4j SET PASSWORD 'password123'"
```

### 5. Ollama for Local LLM

**Download & Install:**
```powershell
# Download Ollama for Windows
$ollamaUrl = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip"
Invoke-WebRequest -Uri $ollamaUrl -OutFile "$env:TEMP\ollama.zip"

# Extract and install
Expand-Archive -Path "$env:TEMP\ollama.zip" -DestinationPath "C:\ollama"

# Add to PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
if ($currentPath -notlike "*C:\ollama*") {
    [Environment]::SetEnvironmentVariable("PATH", "$currentPath;C:\ollama", "Machine")
}

# Create Windows service for Ollama
$serviceName = "Ollama"
$exePath = "C:\ollama\ollama.exe"
$arguments = "serve"

# Create service wrapper script
@"
@echo off
cd /d C:\ollama
ollama.exe serve
"@ | Out-File -FilePath "C:\ollama\ollama-service.bat" -Encoding ASCII

# Install service
& sc.exe create Ollama binPath= "C:\ollama\ollama-service.bat" start= auto
Start-Service -Name "Ollama"
```

**Download Required Models:**
```powershell
# Wait for Ollama service to start
Start-Sleep -Seconds 15

# Download models for legal AI
& "C:\ollama\ollama.exe" pull gemma2:9b
& "C:\ollama\ollama.exe" pull nomic-embed-text
& "C:\ollama\ollama.exe" pull llama3.1:8b

# Create legal-optimized model (optional)
@"
FROM gemma2:9b
PARAMETER temperature 0.1
PARAMETER top_k 40
PARAMETER top_p 0.9
SYSTEM You are a legal AI assistant specializing in contract analysis, case law research, and legal document processing.
"@ | Out-File -FilePath "C:\ollama\Modelfile.legal" -Encoding ASCII

& "C:\ollama\ollama.exe" create gemma3-legal -f "C:\ollama\Modelfile.legal"
```

### 6. NATS Messaging Server

**Download & Install:**
```powershell
# Download NATS Server
$natsUrl = "https://github.com/nats-io/nats-server/releases/latest/download/nats-server-v2.10.4-windows-amd64.zip"
Invoke-WebRequest -Uri $natsUrl -OutFile "$env:TEMP\nats-server.zip"

# Extract
Expand-Archive -Path "$env:TEMP\nats-server.zip" -DestinationPath "C:\nats"

# Create configuration file
@"
# NATS Server Configuration for Legal AI Platform
port: 4222
http_port: 8222

# WebSocket support for browser clients
websocket {
  port: 4223
  no_tls: true
}

# JetStream for persistent messaging
jetstream {
  store_dir: "C:\nats\jetstream"
  max_mem_store: 1GB
  max_file_store: 10GB
}

# Logging
log_file: "C:\nats\nats-server.log"
debug: false
trace: false
logtime: true

# Authentication (basic)
authorization {
  users = [
    {
      user: "legal_ai_client"
      password: "legal_ai_2024"
      permissions: {
        publish: ["legal.>", "system.>"]
        subscribe: ["legal.>", "system.>", "_INBOX.>"]
      }
    }
  ]
}

# Clustering (for future scaling)
cluster {
  name: "legal-ai-cluster"
  port: 6222
}
"@ | Out-File -FilePath "C:\nats\nats-server.conf" -Encoding ASCII

# Create startup script
@"
@echo off
cd /d C:\nats
nats-server.exe -c nats-server.conf
"@ | Out-File -FilePath "C:\nats\start-nats.bat" -Encoding ASCII

# Install as Windows service
& sc.exe create NATS binPath= "C:\nats\start-nats.bat" start= auto DisplayName= "NATS Messaging Server"
Start-Service -Name "NATS"
```

---

## 🔄 **Service Management Scripts**

Create automated service management scripts:

### Start All Services Script
```powershell
# save as: start-all-services.ps1
Write-Host "🚀 Starting Legal AI Platform Services..."

$services = @(
    "postgresql-x64-17",
    "Memurai", 
    "MinIO",
    "Neo4j",
    "Ollama",
    "NATS"
)

foreach ($service in $services) {
    try {
        Write-Host "Starting $service..."
        Start-Service -Name $service -ErrorAction Stop
        Write-Host "✅ $service started" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to start $service`: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "🎯 Service startup complete!"
```

### Health Check Script
```powershell
# save as: check-services-health.ps1
Write-Host "🔍 Legal AI Platform Health Check"
Write-Host "=" * 50

# Service status
$services = @(
    @{Name="PostgreSQL"; Service="postgresql-x64-17"; Port=5432},
    @{Name="Redis"; Service="Memurai"; Port=6379},
    @{Name="MinIO"; Service="MinIO"; Port=9000},
    @{Name="Neo4j"; Service="Neo4j"; Port=7474},
    @{Name="Ollama"; Service="Ollama"; Port=11434},
    @{Name="NATS"; Service="NATS"; Port=4222}
)

foreach ($svc in $services) {
    $status = Get-Service -Name $svc.Service -ErrorAction SilentlyContinue
    
    if ($status -and $status.Status -eq "Running") {
        # Test port connectivity
        $connection = Test-NetConnection -ComputerName localhost -Port $svc.Port -WarningAction SilentlyContinue
        
        if ($connection.TcpTestSucceeded) {
            Write-Host "✅ $($svc.Name): Running & Accessible" -ForegroundColor Green
        } else {
            Write-Host "⚠️ $($svc.Name): Running but port $($svc.Port) not accessible" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ $($svc.Name): Not running" -ForegroundColor Red
    }
}

# Test database connectivity
try {
    $env:PGPASSWORD = "123456"
    $dbTest = & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d legal_ai_db -c "SELECT 1" 2>&1
    if ($dbTest -like "*1*") {
        Write-Host "✅ Database: Connected & Accessible" -ForegroundColor Green
    } else {
        Write-Host "❌ Database: Connection failed" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Database: Connection test failed" -ForegroundColor Red
}

Write-Host "`n🎯 Health check complete!"
```

### Environment Setup Script
```powershell
# save as: setup-environment.ps1
Write-Host "🔧 Setting up Legal AI Platform Environment Variables..."

# Set environment variables for the application
[Environment]::SetEnvironmentVariable("DATABASE_URL", "postgresql://postgres:123456@localhost:5432/legal_ai_db", "Machine")
[Environment]::SetEnvironmentVariable("REDIS_URL", "redis://localhost:6379", "Machine")
[Environment]::SetEnvironmentVariable("MINIO_ENDPOINT", "localhost:9000", "Machine")
[Environment]::SetEnvironmentVariable("MINIO_ACCESS_KEY", "minioadmin", "Machine")
[Environment]::SetEnvironmentVariable("MINIO_SECRET_KEY", "minioadmin123", "Machine")
[Environment]::SetEnvironmentVariable("NEO4J_URL", "bolt://localhost:7687", "Machine")
[Environment]::SetEnvironmentVariable("NEO4J_USERNAME", "neo4j", "Machine")
[Environment]::SetEnvironmentVariable("NEO4J_PASSWORD", "password123", "Machine")
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "localhost:11434", "Machine")
[Environment]::SetEnvironmentVariable("NATS_URL", "nats://legal_ai_client:legal_ai_2024@localhost:4222", "Machine")

Write-Host "✅ Environment variables configured!" -ForegroundColor Green
Write-Host "⚠️ Please restart your terminal/IDE to load new environment variables" -ForegroundColor Yellow
```

---

## 🧪 **Testing the Setup**

### Automated Test Script
```powershell
# save as: test-platform-setup.ps1
Write-Host "🧪 Testing Legal AI Platform Setup..."

# Test each service
$tests = @(
    @{
        Name = "PostgreSQL + pgvector"
        Test = {
            $env:PGPASSWORD = "123456"
            $result = & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d legal_ai_db -c "SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector as similarity;" 2>&1
            return $result -like "*similarity*"
        }
    },
    @{
        Name = "Redis/Memurai"
        Test = {
            $result = & "C:\Program Files\Memurai\memurai-cli.exe" ping 2>&1
            return $result -eq "PONG"
        }
    },
    @{
        Name = "MinIO"
        Test = {
            $result = Invoke-WebRequest -Uri "http://localhost:9000/minio/health/live" -Method GET 2>&1
            return $result.StatusCode -eq 200
        }
    },
    @{
        Name = "Ollama Models"
        Test = {
            $result = & "C:\ollama\ollama.exe" list 2>&1
            return $result -like "*gemma*" -or $result -like "*llama*"
        }
    },
    @{
        Name = "NATS"
        Test = {
            $result = Invoke-WebRequest -Uri "http://localhost:8222/varz" -Method GET 2>&1
            return $result.StatusCode -eq 200
        }
    }
)

$passed = 0
$total = $tests.Count

foreach ($test in $tests) {
    try {
        $result = & $test.Test
        if ($result) {
            Write-Host "✅ $($test.Name): PASS" -ForegroundColor Green
            $passed++
        } else {
            Write-Host "❌ $($test.Name): FAIL" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "❌ $($test.Name): ERROR - $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`n🎯 Test Results: $passed/$total tests passed"

if ($passed -eq $total) {
    Write-Host "🎉 All services configured correctly! Platform ready for development." -ForegroundColor Green
} else {
    Write-Host "⚠️ Some services need attention. Review the failed tests above." -ForegroundColor Yellow
}
```

---

## 🚀 **Integration with SvelteKit App**

### Update your .env file:
```env
# Database
DATABASE_URL="postgresql://postgres:123456@localhost:5432/legal_ai_db"
POSTGRES_HOST="localhost"
POSTGRES_PORT="5432"
POSTGRES_DB="legal_ai_db"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="123456"

# Redis
REDIS_URL="redis://localhost:6379"

# MinIO Object Storage  
MINIO_ENDPOINT="localhost"
MINIO_PORT="9000"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin123"
MINIO_BUCKET="legal-documents"

# Neo4j Graph Database
NEO4J_URL="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password123"

# Ollama LLM
OLLAMA_HOST="http://localhost:11434"
OLLAMA_MODELS="gemma2:9b,nomic-embed-text,gemma3-legal"

# NATS Messaging
NATS_URL="nats://legal_ai_client:legal_ai_2024@localhost:4222"
NATS_WEBSOCKET_URL="ws://localhost:4223"
```

### Run the complete test suite:
```bash
# From your SvelteKit frontend directory
npm run test:e2e
node scripts/production-test-suite.mjs

# Or run the PowerShell health check
powershell -ExecutionPolicy Bypass -File check-services-health.ps1
```

---

## 🔧 **Troubleshooting**

### Common Issues:

1. **PostgreSQL Connection Refused**
   ```powershell
   # Check if service is running
   Get-Service postgresql-x64-17
   
   # Check port binding
   netstat -an | findstr :5432
   
   # Restart service
   Restart-Service postgresql-x64-17
   ```

2. **pgvector Extension Not Found**
   ```sql
   -- Connect to database and check extensions
   \dx
   
   -- If not present, install manually
   CREATE EXTENSION vector;
   ```

3. **Ollama Models Not Loading**
   ```powershell
   # Check available models
   C:\ollama\ollama.exe list
   
   # Re-download if missing
   C:\ollama\ollama.exe pull gemma2:9b
   ```

4. **Port Conflicts**
   ```powershell
   # Find process using port
   netstat -ano | findstr :5432
   
   # Kill process if needed
   taskkill /PID [PID_NUMBER] /F
   ```

---

## 🎉 **Success Verification**

When everything is working correctly:

1. ✅ All services show "Running" status
2. ✅ All ports are accessible (5432, 6379, 9000, 7474, 11434, 4222)
3. ✅ Database accepts connections and vector queries work
4. ✅ Ollama responds with model list
5. ✅ MinIO web interface accessible at http://localhost:9001
6. ✅ Neo4j browser accessible at http://localhost:7474
7. ✅ Your SvelteKit app loads without database connection errors

**Your Legal AI platform is now ready for production-grade development with full native Windows service integration!**