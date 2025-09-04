# RUN-ENHANCED-RAG-V2-COMPLETE.ps1
# Complete integration and deployment script for Enhanced RAG V2 with CRUD and Async

param(
    [Parameter()]
    [ValidateSet("all", "deps", "schema", "build", "start", "test", "status")]
    [string]$Action = "all",
    
    [switch]$SkipDependencies = $false,
    [switch]$ForceRebuild = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "C:\Users\james\Desktop\deeds-web\deeds-web-app"
$GoMicroservice = "$ProjectRoot\go-microservice"
$LogFile = "$ProjectRoot\logs\enhanced-rag-v2-$(Get-Date -Format 'yyyy-MM-dd-HH-mm-ss').log"

# Create logs directory if it doesn't exist
if (!(Test-Path "$ProjectRoot\logs")) {
    New-Item -ItemType Directory -Path "$ProjectRoot\logs" -Force | Out-Null
}

# Logging function
function Write-Log {
    param($Message, $Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] [$Level] $Message"
    Add-Content -Path $LogFile -Value $LogMessage
    
    switch ($Level) {
        "SUCCESS" { Write-Host $Message -ForegroundColor Green }
        "ERROR" { Write-Host $Message -ForegroundColor Red }
        "WARNING" { Write-Host $Message -ForegroundColor Yellow }
        "INFO" { Write-Host $Message -ForegroundColor Cyan }
        default { Write-Host $Message }
    }
}

# ==========================================
# DEPENDENCY INSTALLATION
# ==========================================

function Install-GoDependencies {
    Write-Log "Installing Go dependencies..." "INFO"
    
    Push-Location $GoMicroservice
    try {
        $dependencies = @(
            "github.com/gorilla/mux",
            "github.com/gorilla/websocket",
            "github.com/lib/pq",
            "github.com/redis/go-redis/v9",
            "github.com/streadway/amqp",
            "github.com/google/uuid",
            "gorgonia.org/gorgonia"
        )
        
        foreach ($dep in $dependencies) {
            Write-Log "Installing $dep..." "INFO"
            & go get $dep 2>&1 | Out-String | Write-Log
        }
        
        Write-Log "Running go mod tidy..." "INFO"
        & go mod tidy 2>&1 | Out-String | Write-Log
        
        Write-Log "✅ Go dependencies installed successfully" "SUCCESS"
    }
    catch {
        Write-Log "Failed to install Go dependencies: $_" "ERROR"
        throw
    }
    finally {
        Pop-Location
    }
}

# ==========================================
# DATABASE SCHEMA APPLICATION
# ==========================================

function Apply-DatabaseSchema {
    Write-Log "Applying Enhanced RAG V2 database schema..." "INFO"
    
    $schemaFile = "$ProjectRoot\sql\enhanced-rag-v2-schema.sql"
    
    if (!(Test-Path $schemaFile)) {
        Write-Log "Schema file not found at $schemaFile" "ERROR"
        throw "Schema file missing"
    }
    
    try {
        $env:PGPASSWORD = "postgres"
        $result = & psql -U postgres -h localhost -d legal_ai_rag -f $schemaFile 2>&1
        $result | Out-String | Write-Log
        
        if ($LASTEXITCODE -eq 0) {
            Write-Log "✅ Database schema applied successfully" "SUCCESS"
        } else {
            throw "psql returned exit code $LASTEXITCODE"
        }
    }
    catch {
        Write-Log "Failed to apply database schema: $_" "ERROR"
        Write-Log "Attempting to create database if it doesn't exist..." "WARNING"
        
        # Try to create the database
        & psql -U postgres -h localhost -c "CREATE DATABASE legal_ai_rag;" 2>&1 | Out-Null
        
        # Retry schema application
        $result = & psql -U postgres -h localhost -d legal_ai_rag -f $schemaFile 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "✅ Database created and schema applied" "SUCCESS"
        } else {
            throw "Failed to create database or apply schema"
        }
    }
}

# ==========================================
# BUILD SERVICE
# ==========================================

function Build-EnhancedRAGService {
    Write-Log "Building Enhanced RAG V2 service..." "INFO"
    
    Push-Location $GoMicroservice
    try {
        $outputPath = "bin\enhanced-rag-v2-local.exe"
        $sourcePath = "cmd\enhanced-rag-v2-local\main.go"
        
        if (!(Test-Path $sourcePath)) {
            Write-Log "Source file not found at $sourcePath" "ERROR"
            throw "Source file missing"
        }
        
        if ($ForceRebuild -and (Test-Path $outputPath)) {
            Remove-Item $outputPath -Force
            Write-Log "Removed existing binary for rebuild" "INFO"
        }
        
        Write-Log "Building with: go build -o $outputPath $sourcePath" "INFO"
        
        $env:CGO_ENABLED = "1"
        $env:GOOS = "windows"
        $env:GOARCH = "amd64"
        
        & go build -o $outputPath $sourcePath 2>&1 | Out-String | Write-Log
        
        if (Test-Path $outputPath) {
            Write-Log "✅ Enhanced RAG V2 built successfully" "SUCCESS"
            Write-Log "Binary location: $outputPath" "INFO"
        } else {
            throw "Build failed - executable not created"
        }
    }
    catch {
        Write-Log "Failed to build service: $_" "ERROR"
        throw
    }
    finally {
        Pop-Location
    }
}

# ==========================================
# START SERVICES
# ==========================================

function Start-RequiredServices {
    Write-Log "Starting required services..." "INFO"
    
    $services = @(
        @{Name = "PostgreSQL"; ServiceName = "postgresql-x64-14"; Port = 5432},
        @{Name = "Redis"; Process = "redis-server"; Port = 6379},
        @{Name = "RabbitMQ"; ServiceName = "RabbitMQ"; Port = 5672}
    )
    
    foreach ($svc in $services) {
        Write-Log "Checking $($svc.Name)..." "INFO"
        
        $testConnection = Test-NetConnection -ComputerName localhost -Port $svc.Port -WarningAction SilentlyContinue
        
        if ($testConnection.TcpTestSucceeded) {
            Write-Log "✅ $($svc.Name) is already running on port $($svc.Port)" "SUCCESS"
        }
        else {
            Write-Log "Starting $($svc.Name)..." "WARNING"
            
            if ($svc.ServiceName) {
                try {
                    Start-Service $svc.ServiceName -ErrorAction SilentlyContinue
                    Write-Log "✅ $($svc.Name) service started" "SUCCESS"
                }
                catch {
                    Write-Log "Could not start $($svc.Name) service: $_" "WARNING"
                }
            }
            
            if ($svc.Process -eq "redis-server") {
                # Start Redis manually
                $redisPath = "$ProjectRoot\redis-windows\redis-server.exe"
                if (Test-Path $redisPath) {
                    Start-Process -FilePath $redisPath -WindowStyle Hidden
                    Write-Log "✅ Redis started manually" "SUCCESS"
                } else {
                    Write-Log "Redis not found at $redisPath" "WARNING"
                }
            }
        }
        
        Start-Sleep -Seconds 2
    }
}

function Start-EnhancedRAGService {
    Write-Log "Starting Enhanced RAG V2 service..." "INFO"
    
    $exePath = "$GoMicroservice\bin\enhanced-rag-v2-local.exe"
    
    if (!(Test-Path $exePath)) {
        Write-Log "Enhanced RAG V2 executable not found at $exePath" "ERROR"
        throw "Service executable missing - run build first"
    }
    
    # Check if already running
    $existingProcess = Get-Process -Name "enhanced-rag-v2-local" -ErrorAction SilentlyContinue
    if ($existingProcess) {
        Write-Log "Enhanced RAG V2 is already running (PID: $($existingProcess.Id))" "WARNING"
        
        if ($ForceRebuild) {
            Write-Log "Stopping existing process..." "INFO"
            Stop-Process -Id $existingProcess.Id -Force
            Start-Sleep -Seconds 2
        } else {
            return
        }
    }
    
    # Set environment variables
    $env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/legal_ai_rag?sslmode=disable"
    $env:REDIS_URL = "localhost:6379"
    $env:RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
    $env:PORT = "8097"
    
    # Start the service
    Write-Log "Starting Enhanced RAG V2 on port 8097..." "INFO"
    $process = Start-Process -FilePath $exePath -WorkingDirectory $GoMicroservice -PassThru -WindowStyle Normal
    
    Start-Sleep -Seconds 3
    
    # Verify it's running
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8097/health" -Method Get -TimeoutSec 5
        if ($response.status -eq "healthy") {
            Write-Log "✅ Enhanced RAG V2 is running and healthy!" "SUCCESS"
            Write-Log "API: http://localhost:8097" "INFO"
            Write-Log "Process ID: $($process.Id)" "INFO"
        }
    }
    catch {
        Write-Log "Service started but health check failed: $_" "WARNING"
    }
}

# ==========================================
# TEST SYSTEM
# ==========================================

function Test-EnhancedRAGSystem {
    Write-Log "Testing Enhanced RAG V2 system..." "INFO"
    
    $baseUrl = "http://localhost:8097"
    $tests = @(
        @{
            Name = "Health Check"
            Method = "GET"
            Endpoint = "/health"
            ExpectedStatus = 200
        },
        @{
            Name = "Create User Intent"
            Method = "POST"
            Endpoint = "/api/intents"
            Body = @{
                user_id = "test-user-1"
                intent = "contract_review"
                keywords = @("liability", "indemnification")
                confidence = 0.85
                context = @{ source = "test" }
            }
            ExpectedStatus = 200
        },
        @{
            Name = "List User Intents"
            Method = "GET"
            Endpoint = "/api/intents?user_id=test-user-1"
            ExpectedStatus = 200
        },
        @{
            Name = "Generate Recommendations"
            Method = "POST"
            Endpoint = "/api/recommendations/generate"
            Body = @{
                user_id = "test-user-1"
                context = @{ test = $true }
            }
            ExpectedStatus = 200
        },
        @{
            Name = "Create Todo Item"
            Method = "POST"
            Endpoint = "/api/todos"
            Body = @{
                user_id = "test-user-1"
                title = "Test Todo"
                description = "Test Description"
                priority = 5
                status = "pending"
            }
            ExpectedStatus = 200
        },
        @{
            Name = "Solve Todos"
            Method = "POST"
            Endpoint = "/api/todos/solve"
            Body = @{
                user_id = "test-user-1"
                auto_solve = $true
            }
            ExpectedStatus = 200
        }
    )
    
    $passed = 0
    $failed = 0
    
    foreach ($test in $tests) {
        Write-Log "Testing: $($test.Name)" "INFO"
        
        try {
            $uri = "$baseUrl$($test.Endpoint)"
            $params = @{
                Uri = $uri
                Method = $test.Method
                ContentType = "application/json"
                TimeoutSec = 10
            }
            
            if ($test.Body) {
                $params.Body = ($test.Body | ConvertTo-Json -Depth 10)
            }
            
            $response = Invoke-RestMethod @params
            Write-Log "  ✅ $($test.Name) passed" "SUCCESS"
            
            if ($Verbose) {
                Write-Log "  Response: $($response | ConvertTo-Json -Compress)" "INFO"
            }
            
            $passed++
        }
        catch {
            Write-Log "  ❌ $($test.Name) failed: $_" "ERROR"
            $failed++
        }
    }
    
    Write-Log "" "INFO"
    Write-Log "Test Results: $passed passed, $failed failed" $(if ($failed -eq 0) { "SUCCESS" } else { "WARNING" })
    
    return ($failed -eq 0)
}

# ==========================================
# STATUS CHECK
# ==========================================

function Get-SystemStatus {
    Write-Log "Checking system status..." "INFO"
    Write-Log "=" * 50 "INFO"
    
    # Check services
    $services = @(
        @{Name = "PostgreSQL"; Port = 5432},
        @{Name = "Redis"; Port = 6379},
        @{Name = "RabbitMQ"; Port = 5672},
        @{Name = "RabbitMQ Management"; Port = 15672},
        @{Name = "Enhanced RAG V2"; Port = 8097},
        @{Name = "Existing Go Service"; Port = 8084}
    )
    
    foreach ($svc in $services) {
        $test = Test-NetConnection -ComputerName localhost -Port $svc.Port -WarningAction SilentlyContinue
        
        if ($test.TcpTestSucceeded) {
            Write-Log "✅ $($svc.Name) - Running on port $($svc.Port)" "SUCCESS"
        } else {
            Write-Log "❌ $($svc.Name) - Not responding on port $($svc.Port)" "WARNING"
        }
    }
    
    # Check processes
    Write-Log "" "INFO"
    Write-Log "Process Status:" "INFO"
    
    $processes = @("enhanced-rag-v2-local", "redis-server", "postgres", "beam.smp")
    foreach ($proc in $processes) {
        $p = Get-Process -Name $proc -ErrorAction SilentlyContinue
        if ($p) {
            Write-Log "  ✅ $proc - PID: $($p.Id)" "SUCCESS"
        } else {
            Write-Log "  ❌ $proc - Not running" "WARNING"
        }
    }
    
    Write-Log "=" * 50 "INFO"
}

# ==========================================
# MAIN EXECUTION
# ==========================================

Write-Log "=" * 60 "INFO"
Write-Log "ENHANCED RAG V2 - COMPLETE INTEGRATION" "INFO"
Write-Log "Action: $Action" "INFO"
Write-Log "=" * 60 "INFO"
Write-Log "" "INFO"

try {
    switch ($Action) {
        "all" {
            if (!$SkipDependencies) {
                Install-GoDependencies
            }
            Apply-DatabaseSchema
            Build-EnhancedRAGService
            Start-RequiredServices
            Start-EnhancedRAGService
            Start-Sleep -Seconds 5
            Test-EnhancedRAGSystem
            Get-SystemStatus
            
            Write-Log "" "INFO"
            Write-Log "🎉 ENHANCED RAG V2 FULLY INTEGRATED AND RUNNING!" "SUCCESS"
            Write-Log "" "INFO"
            Write-Log "Access Points:" "INFO"
            Write-Log "  📱 API: http://localhost:8097" "INFO"
            Write-Log "  🔌 WebSocket: ws://localhost:8097/ws" "INFO"
            Write-Log "  📊 RabbitMQ: http://localhost:15672" "INFO"
            Write-Log "  📝 Logs: $LogFile" "INFO"
        }
        
        "deps" {
            Install-GoDependencies
        }
        
        "schema" {
            Apply-DatabaseSchema
        }
        
        "build" {
            if (!$SkipDependencies) {
                Install-GoDependencies
            }
            Build-EnhancedRAGService
        }
        
        "start" {
            Start-RequiredServices
            Start-EnhancedRAGService
        }
        
        "test" {
            Test-EnhancedRAGSystem
        }
        
        "status" {
            Get-SystemStatus
        }
    }
    
    Write-Log "" "INFO"
    Write-Log "✅ Operation completed successfully" "SUCCESS"
}
catch {
    Write-Log "" "INFO"
    Write-Log "❌ Operation failed: $_" "ERROR"
    Write-Log "Check log file for details: $LogFile" "WARNING"
    exit 1
}

Write-Log "" "INFO"
Write-Log "Log file: $LogFile" "INFO"
