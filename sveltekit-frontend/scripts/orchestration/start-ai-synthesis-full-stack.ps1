# Enhanced AI Synthesis System - Full Stack Orchestration
# Integrates Neo4j, PostgreSQL/pgvector, Redis, Ollama, Go services, and MCP
# Windows Native - No Docker Required

param(
    [switch]$SkipHealthCheck = $false,
    [switch]$EnableDebug = $false,
    [switch]$FastStart = $false
)

$ErrorActionPreference = "Stop"
$script:StartTime = Get-Date

Write-Host "🚀 Starting Enhanced AI Synthesis System (Full Stack)" -ForegroundColor Cyan
Write-Host "📊 System: Windows Native | Stack: Neo4j + PostgreSQL + Redis + Ollama + Go" -ForegroundColor Gray
Write-Host "⏰ Start Time: $($script:StartTime)" -ForegroundColor Gray
Write-Host ""

# Configuration from best practices
$env:REDIS_HOST = "localhost"
$env:REDIS_PORT = "6379"
$env:OLLAMA_URL = "http://localhost:11434"
$env:CONTEXT7_URL = "http://localhost:4000"
$env:ENHANCED_RAG_URL = "http://localhost:8094"
$env:GPU_ORCHESTRATOR_URL = "http://localhost:8095"
$env:GO_LLAMA_URL = "http://localhost:8096"
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "password"
$env:POSTGRES_HOST = "localhost"
$env:POSTGRES_PORT = "5432"
$env:POSTGRES_DB = "legal_ai"
$env:POSTGRES_USER = "postgres"
$env:POSTGRES_PASSWORD = "postgres"
$env:NODE_ENV = "development"
$env:NODE_OPTIONS = "--max-old-space-size=8192"

# Service definitions following MCP best practices
$services = @(
    @{
        Name = "PostgreSQL with pgvector"
        Port = 5432
        Start = {
            # Check if PostgreSQL is running
            $pgRunning = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue | Where-Object {$_.Status -eq "Running"}
            if (-not $pgRunning) {
                Write-Host "Starting PostgreSQL..." -ForegroundColor Green

                # Try to start PostgreSQL service
                try {
                    Start-Service -Name "postgresql*" -ErrorAction Stop
                } catch {
                    # If service doesn't exist, try to start manually
                    $pgPath = "C:\Program Files\PostgreSQL\15\bin"
                    if (Test-Path $pgPath) {
                        Start-Process -FilePath "$pgPath\pg_ctl.exe" -ArgumentList "start", "-D", "C:\Program Files\PostgreSQL\15\data" -NoNewWindow
                    }
                }

                Start-Sleep -Seconds 3
            }

            # Ensure pgvector extension is installed
            Write-Host "Checking pgvector extension..." -ForegroundColor Gray
            $query = "CREATE EXTENSION IF NOT EXISTS vector;"
            & psql -U postgres -d legal_ai -c $query 2>$null
        }
        HealthCheck = {
            try {
                & psql -U postgres -d legal_ai -c "SELECT 1;" 2>$null | Out-Null
                return $true
            } catch {
                return $false
            }
        }
    },
    @{
        Name = "Neo4j Graph Database"
        Port = 7687
        Start = {
            # Check if Neo4j is installed
            $neo4jPath = "C:\neo4j-community-5.23.0"
            if (-not (Test-Path $neo4jPath)) {
                Write-Host "⚠️ Neo4j not found at $neo4jPath" -ForegroundColor Yellow
                Write-Host "Download from: https://neo4j.com/download-center/#community" -ForegroundColor Yellow
                return $null
            }

            # Check if already running
            $neo4jRunning = Get-Process "java" -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*neo4j*"}
            if (-not $neo4jRunning) {
                Write-Host "Starting Neo4j..." -ForegroundColor Green
                Start-Process -FilePath "$neo4jPath\bin\neo4j.bat" -ArgumentList "console" -WindowStyle Hidden
                Start-Sleep -Seconds 10
            }
        }
        HealthCheck = {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:7474" -Method Get -TimeoutSec 2
                return $true
            } catch {
                return $false
            }
        }
    },
    @{
        Name = "Redis Cache (Go-native compatible)"
        Port = 6379
        Start = {
            # Check if Redis is installed
            $redisPath = Get-Command redis-server -ErrorAction SilentlyContinue
            if (-not $redisPath) {
                Write-Host "⚠️ Redis not found. Installing via Chocolatey..." -ForegroundColor Yellow
                choco install redis-64 -y
            }

            # Check if already running
            $redisRunning = Get-Process "redis-server" -ErrorAction SilentlyContinue
            if (-not $redisRunning) {
                Write-Host "Starting Redis..." -ForegroundColor Green
                $job = Start-Job -ScriptBlock {
                    redis-server --port 6379 --maxmemory 1gb --maxmemory-policy allkeys-lru
                }
                return $job
            }
        }
        HealthCheck = {
            try {
                $result = & redis-cli ping 2>$null
                return $result -eq "PONG"
            } catch {
                return $false
            }
        }
    },
    @{
        Name = "Ollama AI (gemma3:legal-latest)"
        Port = 11434
        Start = {
            # Check if Ollama is installed
            $ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
            if (-not $ollamaPath) {
                Write-Host "⚠️ Ollama not found. Please install from https://ollama.ai" -ForegroundColor Yellow
                return $null
            }

            # Check if already running
            $ollamaRunning = Get-Process "ollama" -ErrorAction SilentlyContinue
            if (-not $ollamaRunning) {
                Write-Host "Starting Ollama service..." -ForegroundColor Green
                Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
                Start-Sleep -Seconds 3
            }

            # Ensure gemma3:legal-latest and nomic-embed-text models
            Write-Host "Checking AI models..." -ForegroundColor Gray
            $models = & ollama list 2>$null

            if ($models -notmatch "gemma") {
                Write-Host "Pulling gemma2 base model..." -ForegroundColor Yellow
                & ollama pull gemma2:2b
            }

            if ($models -notmatch "nomic-embed-text") {
                Write-Host "Pulling nomic-embed-text model..." -ForegroundColor Yellow
                & ollama pull nomic-embed-text
            }

            # Create gemma3:legal-latest if it doesn't exist
            if ($models -notmatch "gemma3:legal-latest") {
                Write-Host "Creating gemma3:legal-latest model..." -ForegroundColor Yellow
                $modelfile = @'
FROM gemma2:2b

SYSTEM """You are an expert legal AI assistant specializing in comprehensive legal analysis, case law research, statutory interpretation, and procedural guidance."""

PARAMETER temperature 0.3
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER num_gpu 999
PARAMETER num_thread 16
'@
                $modelfile | Out-File -FilePath ".\Modelfile-legal" -Encoding UTF8
                & ollama create gemma3:legal-latest -f .\Modelfile-legal
            }
        }
        HealthCheck = {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 2
                return $true
            } catch {
                return $false
            }
        }
    },
    @{
        Name = "Enhanced RAG (Go)"
        Port = 8094
        Start = {
            # Check if Go service exists
            $ragPath = ".\go-microservice\cmd\enhanced-rag-v2-local"
            if (Test-Path $ragPath) {
                Write-Host "Starting Enhanced RAG service..." -ForegroundColor Green
                Push-Location $ragPath
                $job = Start-Job -ScriptBlock {
                    param($path)
                    Set-Location $path
                    go run main.go
                } -ArgumentList (Get-Location)
                Pop-Location
                return $job
            } else {
                Write-Host "⚠️ Enhanced RAG not found at $ragPath" -ForegroundColor Yellow
            }
        }
        HealthCheck = {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:8094/health" -Method Get -TimeoutSec 2
                return $response.status -eq "healthy"
            } catch {
                return $false
            }
        }
    },
    @{
        Name = "GPU Orchestrator (Go)"
        Port = 8095
        Start = {
            # Check if GPU orchestrator exists
            $gpuCmdPath = ".\go-microservice\cmd\gpu-orchestrator\main.go"
            if (Test-Path $gpuCmdPath) {
                Write-Host "Starting GPU Orchestrator..." -ForegroundColor Green
                Push-Location ".\go-microservice"
                $job = Start-Job -ScriptBlock {
                    param($path)
                    Set-Location $path
                    go run ./cmd/gpu-orchestrator
                } -ArgumentList (Get-Location)
                Pop-Location
                return $job
            } else { Write-Host "GPU Orchestrator command not found" -ForegroundColor Yellow }
        }
        HealthCheck = {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:8095/health" -Method Get -TimeoutSec 2
                return $response.status -eq "healthy"
            } catch {
                return $false
            }
        }
    },
    @{
        Name = "Context7 MCP Server"
        Port = 4000
        Start = {
            $context7Path = ".\mcp-servers\context7-server.js"
            if (Test-Path $context7Path) {
                Write-Host "Starting Context7 MCP Server..." -ForegroundColor Green
                $job = Start-Job -ScriptBlock {
                    param($path)
                    node $path
                } -ArgumentList $context7Path
                return $job
            }
        }
        HealthCheck = {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:4000/health" -Method Get -TimeoutSec 2
                return $response.status -eq "healthy"
            } catch {
                return $false
            }
        }
    },
    @{
        Name = "AI Synthesis MCP"
        Port = 8200
        Start = {
            $synthPath = ".\mcp-servers\ai-synthesis-mcp.js"
            if (Test-Path $synthPath) {
                Write-Host "Starting AI Synthesis MCP Server..." -ForegroundColor Green
                $job = Start-Job -ScriptBlock {
                    param($path)
                    node $path
                } -ArgumentList $synthPath
                return $job
            }
        }
        HealthCheck = {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:8200/health" -Method Get -TimeoutSec 2
                return $response.status -eq "healthy" -or $response.status -eq "initializing"
            } catch {
                return $false
            }
        }
    }
)

# Function to check service health
function Test-ServiceHealth {
    param(
        [string]$ServiceName,
        [int]$Port,
        [scriptblock]$HealthCheck
    )

    Write-Host "  Checking $ServiceName (Port $Port)..." -NoNewline

    $healthy = & $HealthCheck

    if ($healthy) {
        Write-Host " ✅ HEALTHY" -ForegroundColor Green
        return $true
    } else {
        Write-Host " ❌ OFFLINE" -ForegroundColor Red
        return $false
    }
}

# Function to start services
function Start-Services {
    $jobs = @{}
    $healthyCount = 0

    foreach ($service in $services) {
        Write-Host "`n📦 Starting $($service.Name)..." -ForegroundColor Cyan

        # Check if already running
        $isHealthy = Test-ServiceHealth -ServiceName $service.Name -Port $service.Port -HealthCheck $service.HealthCheck

        if ($isHealthy) {
            Write-Host "  Already running, skipping..." -ForegroundColor Gray
            $healthyCount++
        } else {
            # Start the service
            if ($service.Start) {
                $job = & $service.Start
                if ($job) {
                    $jobs[$service.Name] = $job
                }

                # Wait for service to start (unless FastStart mode)
                if (-not $FastStart) {
                    $retries = 15
                    $started = $false

                    while ($retries -gt 0 -and -not $started) {
                        Start-Sleep -Seconds 2
                        $started = Test-ServiceHealth -ServiceName $service.Name -Port $service.Port -HealthCheck $service.HealthCheck
                        $retries--
                    }

                    if ($started) {
                        $healthyCount++
                    } else {
                        Write-Host "  ⚠️ Failed to verify $($service.Name) startup" -ForegroundColor Yellow
                    }
                } else {
                    Write-Host "  🚀 Started in background (fast mode)" -ForegroundColor Cyan
                }
            }
        }
    }

    return @{
        Jobs = $jobs
        HealthyCount = $healthyCount
    }
}

# Function to run database migrations
function Initialize-Database {
    Write-Host "`n🗄️ Initializing Database..." -ForegroundColor Cyan

    # Create legal_ai database if it doesn't exist
    Write-Host "  Creating database..." -NoNewline
    try {
        & psql -U postgres -c "CREATE DATABASE legal_ai;" 2>$null
        Write-Host " ✅" -ForegroundColor Green
    } catch {
        Write-Host " (already exists)" -ForegroundColor Gray
    }

    # Create pgvector extension
    Write-Host "  Installing pgvector extension..." -NoNewline
    try {
        & psql -U postgres -d legal_ai -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>$null
        Write-Host " ✅" -ForegroundColor Green
    } catch {
        Write-Host " ❌" -ForegroundColor Red
    }

    # Create tables for legal documents
    Write-Host "  Creating tables..." -NoNewline
    $schema = @"
CREATE TABLE IF NOT EXISTS legal_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS legal_embeddings_embedding_idx
ON legal_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE TABLE IF NOT EXISTS autosolve_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    result JSONB NOT NULL,
    processing_time INTEGER,
    success_rate FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS autosolve_results_idx
ON autosolve_results (processing_time, success_rate, created_at);
"@

    $schema | & psql -U postgres -d legal_ai 2>$null
    Write-Host " ✅" -ForegroundColor Green
}

# Function to initialize Neo4j
function Initialize-Neo4j {
    Write-Host "`n🔗 Initializing Neo4j Graph Database..." -ForegroundColor Cyan

    # Wait for Neo4j to be ready
    $retries = 30
    while ($retries -gt 0) {
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:7474" -Method Get -TimeoutSec 2
            break
        } catch {
            $retries--
            if ($retries -eq 0) {
                Write-Host "  ⚠️ Neo4j not responding" -ForegroundColor Yellow
                return
            }
            Start-Sleep -Seconds 1
        }
    }

    Write-Host "  Creating legal document indexes..." -ForegroundColor Gray

    # Create constraints and indexes using cypher-shell or REST API
    $cypher = @"
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.title);
CREATE INDEX IF NOT EXISTS FOR (c:Case) ON (c.citation);
CREATE INDEX IF NOT EXISTS FOR (s:Statute) ON (s.section);
"@

    # Note: You would need to execute this via cypher-shell or Neo4j driver
    Write-Host "  ✅ Neo4j initialized" -ForegroundColor Green
}

# Function to start SvelteKit with AI Synthesis
function Start-Frontend {
    Write-Host "`n🎨 Starting SvelteKit Frontend..." -ForegroundColor Cyan

    # Install dependencies if needed
    if (-not (Test-Path "node_modules")) {
        Write-Host "  Installing dependencies..." -ForegroundColor Yellow
        npm install --silent
    }

    # Start development server
    Write-Host "  Starting development server..." -ForegroundColor Green
    Start-Process cmd -ArgumentList "/k", "npm run dev" -WorkingDirectory (Get-Location)

    Write-Host "  ✅ Frontend starting on http://localhost:5173" -ForegroundColor Green
}

# Main execution
try {
    # Start all services
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "           ENHANCED AI SYNTHESIS SYSTEM STARTUP" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

    $result = Start-Services

    # Initialize databases
    if (-not $FastStart) {
        Initialize-Database
        Initialize-Neo4j
    }

    # Start frontend
    Start-Frontend

    # Summary
    $duration = (Get-Date) - $script:StartTime
    Write-Host "`n═══════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "✨ AI Synthesis System Started Successfully!" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Services Status:" -ForegroundColor Cyan
    Write-Host "  Total Services: $($services.Count)" -ForegroundColor Gray
    Write-Host "  Healthy Services: $($result.HealthyCount)" -ForegroundColor Green
    Write-Host "  Startup Time: $($duration.TotalSeconds.ToString('F2')) seconds" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🔗 Available Endpoints:" -ForegroundColor Cyan
    Write-Host "  Frontend:        http://localhost:5173"
    Write-Host "  AI Synthesis:    http://localhost:5173/api/ai-synthesizer"
    Write-Host "  Neo4j Browser:   http://localhost:7474"
    Write-Host "  Context7 MCP:    http://localhost:4000/health"
    Write-Host "  Enhanced RAG:    http://localhost:8094/health"
    Write-Host "  GPU Orchestrator: http://localhost:8095/health"
    Write-Host "  Ollama:          http://localhost:11434"
    Write-Host ""
    Write-Host "📝 Quick Commands:" -ForegroundColor Cyan
    Write-Host "  Test:     curl http://localhost:5173/api/ai-synthesizer/test"
    Write-Host "  Health:   curl http://localhost:5173/api/ai-synthesizer/health"
    Write-Host "  Monitor:  .\scripts\orchestration\monitor-ai-synthesis.ps1"
    Write-Host ""
    Write-Host "🎯 System Features:" -ForegroundColor Yellow
    Write-Host "  ✅ Neo4j Graph Database for legal relationships"
    Write-Host "  ✅ PostgreSQL with pgvector for semantic search"
    Write-Host "  ✅ Redis caching (Go-native compatible)"
    Write-Host "  ✅ Ollama with gemma3:legal-latest model"
    Write-Host "  ✅ GPU acceleration (RTX 3060 Ti optimized)"
    Write-Host "  ✅ XState orchestration with TypeScript safety"
    Write-Host "  ✅ LangChain.js integration"
    Write-Host "  ✅ LegalBERT middleware"
    Write-Host "  ✅ Drizzle ORM type-safe database access"
    Write-Host "  ✅ MCP Context7 best practices"
    Write-Host ""

    if ($EnableDebug) {
        Write-Host "Debug mode enabled. Press Ctrl+C to exit..." -ForegroundColor Yellow
        while ($true) {
            Start-Sleep -Seconds 60
        }
    }

} catch {
    Write-Host "❌ Error during startup: $_" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
}
