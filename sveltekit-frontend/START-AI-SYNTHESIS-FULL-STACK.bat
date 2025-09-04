# filepath: c:\Users\james\Desktop\deeds-web\deeds-web-app\start-legal-ai-system.ps1
<#
    Enhanced Legal AI System Orchestrator (PowerShell Native)
    Components:
        - PostgreSQL (pgvector), Neo4j, Redis
        - Go Microservices (Enhanced RAG, GPU Orchestrator, Go-LLaMA)
        - MCP Servers (context7-multicore, ai-synthesis if present)
        - SvelteKit Frontend
    Usage:
        ./start-legal-ai-system.ps1 [-Dev] [-SkipFrontend] [-SkipGo] [-SkipDB] [-Verbose]
#>

param(
    [switch]$Dev,
    [switch]$SkipFrontend,
    [switch]$SkipGo,
    [switch]$SkipDB,
    [switch]$NoOllama,
    [switch]$SkipNeo4j,
    [switch]$SkipRedis
)

$ErrorActionPreference = 'Stop'
$host.UI.RawUI.WindowTitle = "Legal AI System Orchestrator"

# -------------------- Configuration --------------------
$Config = [ordered]@{
    Ports = @{
        Frontend     = 5173
        Context7MCP  = 4000
        MultiCoreLB  = 40000
        RAG          = 8094
        GPU          = 8095
        GoLLaMA      = 8096
        AISynthesis  = 8200
        Redis        = 6379
        Neo4jHTTP    = 7474
        Neo4jBolt    = 7687
        Ollama       = 11434
    }
    Paths = @{
        RepoRoot     = (Resolve-Path ".").Path
        GoRoot       = "go-microservice"
        MCPDir       = "mcp-servers"
        Context7Multi = "mcp\context7-multicore.js"
        BatchScript  = "sveltekit-frontend\START-AI-SYNTHESIS-FULL-STACK.bat"
    }
    Env = @{
        NODE_ENV                = $(if ($Dev) { "development" } else { "production" })
        NODE_OPTIONS            = "--max-old-space-size=8192"
        PGPASSWORD              = "123456"
        POSTGRES_DB             = "legal_ai"
        POSTGRES_USER           = "postgres"
        POSTGRES_PASSWORD       = "123456"
        POSTGRES_HOST           = "localhost"
        POSTGRES_PORT           = "5432"
        NEO4J_URI               = "bolt://localhost:7687"
        NEO4J_USER              = "neo4j"
        NEO4J_PASSWORD          = "password"
        OLLAMA_URL              = "http://localhost:11434"
        CONTEXT7_URL            = "http://localhost:4000"
        MCP_MULTICORE           = "true"
        MAX_WORKERS             = "8"
    }
}

# -------------------- Utility Functions --------------------
function Write-Section($t) { Write-Host "`n==== $t ====" -ForegroundColor Cyan }
function Write-Info($m){ Write-Host "[INFO] $m" -ForegroundColor DarkGray }
function Write-Ok($m){ Write-Host "[OK]   $m" -ForegroundColor Green }
function Write-Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Write-Err($m){ Write-Host "[ERR]  $m" -ForegroundColor Red }

function Test-Cmd($name){
    $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

function Test-PortOpen($port, $host='127.0.0.1'){
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $iar = $client.BeginConnect($host, [int]$port, $null, $null)
        $success = $iar.AsyncWaitHandle.WaitOne(400)
        if($success -and $client.Connected){ $client.Close(); return $true }
        $client.Close()
        return $false
    } catch { return $false }
}

function Wait-Port($port, $timeoutSec=30){
    $start = Get-Date
    while(-not (Test-PortOpen $port)){
        if((Get-Date) - $start -gt [TimeSpan]::FromSeconds($timeoutSec)){
            Write-Warn "Port $port not ready after $timeoutSec sec"
            return $false
        }
        Start-Sleep -Milliseconds 600
    }
    Write-Ok "Port $port ready"
    return $true
}

function Start-ProcSimple($name, $file, $args, $wd=$Config.Paths.RepoRoot){
    if(-not (Test-Path $wd)){ Write-Warn "$name: working dir missing $wd"; return }
    Write-Info "Starting $name ..."
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $file
    $psi.Arguments = $args
    $psi.WorkingDirectory = $wd
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    Write-Ok "$name pid=$($p.Id)"
}

function Ensure-Env {
    foreach($k in $Config.Env.Keys){
        $val = $Config.Env[$k]
        if(-not [string]::IsNullOrWhiteSpace($val)){
            $Env:$k = $val
        }
    }
    Write-Ok "Environment variables applied"
}

# -------------------- Prerequisites --------------------
Write-Section "Prerequisite Checks"

if(-not (Test-Cmd node)){ Write-Err "Node.js not found"; exit 1 } else { Write-Ok "Node.js $(node -v)" }
if(-not (Test-Cmd npm)){ Write-Warn "npm missing?" }
if(-not (Test-Cmd go)){ if(-not $SkipGo){ Write-Warn "Go not found - Go services skipped"; $SkipGo = $true } }
if(-not (Test-Cmd redis-server)){ if(-not $SkipRedis){ Write-Warn "Redis not found - will attempt continue" } }
if(-not (Test-Cmd psql)){ if(-not $SkipDB){ Write-Warn "psql not found - DB setup skipped"; $SkipDB = $true } }
if(-not (Test-Cmd ollama)){ if(-not $NoOllama){ Write-Warn "Ollama not installed - LLM features limited"; $NoOllama = $true } }

Ensure-Env

# -------------------- PostgreSQL --------------------
if(-not $SkipDB){
    Write-Section "PostgreSQL + pgvector"
    try {
        psql -U $Config.Env.POSTGRES_USER -h $Config.Env.POSTGRES_HOST -p $Config.Env.POSTGRES_PORT -c "SELECT 1" 2>$null | Out-Null
        Write-Ok "PostgreSQL reachable"
        psql -U $Config.Env.POSTGRES_USER -h $Config.Env.POSTGRES_HOST -p $Config.Env.POSTGRES_PORT -tc "SELECT 1 FROM pg_database WHERE datname='legal_ai'" | Select-String 1 -Quiet
        if(-not $?) {
            Write-Info "Creating database legal_ai"
            psql -U $Config.Env.POSTGRES_USER -h $Config.Env.POSTGRES_HOST -p $Config.Env.POSTGRES_PORT -c "CREATE DATABASE legal_ai;"
        }
        psql -U $Config.Env.POSTGRES_USER -h $Config.Env.POSTGRES_HOST -d legal_ai -c "CREATE EXTENSION IF NOT EXISTS vector;" | Out-Null
        Write-Ok "pgvector ready"
    } catch { Write-Warn "PostgreSQL setup failed: $_" }
}

# -------------------- Neo4j --------------------
if(-not $SkipNeo4j){
    Write-Section "Neo4j"
    $neoBat = "C:\neo4j-community-5.23.0\bin\neo4j.bat"
    if(Test-Path $neoBat){
        if(-not (Test-PortOpen $Config.Ports.Neo4jHTTP)){
            Start-ProcSimple "Neo4j" $neoBat "console" (Split-Path $neoBat -Parent)
            Wait-Port $Config.Ports.Neo4jHTTP 40 | Out-Null
        } else { Write-Ok "Neo4j already running" }
    } else {
        Write-Warn "Neo4j not found (expected $neoBat)"
    }
}

# -------------------- Redis --------------------
if(-not $SkipRedis){
    Write-Section "Redis"
    Write-Info "[Redis] Checking Redis availability on port $($Config.Ports.Redis)..."

    $redisUp = Test-PortOpen $Config.Ports.Redis
    if($redisUp){
        Write-Ok "[Redis] Already running"
    } else {
        if(Test-Cmd redis-server){
            # Choose a lightweight config (in-memory only unless a redis.conf exists)
            $redisArgs = @(
                "--port", $Config.Ports.Redis
                "--maxmemory", "1024mb"
                "--maxmemory-policy", "allkeys-lru"
            )

            # Optional persistence if redis.conf exists locally
            $redisConf = Join-Path $Config.Paths.RepoRoot "redis.conf"
            if(Test-Path $redisConf){
                Write-Info "[Redis] Using redis.conf at $redisConf"
                $redisArgs = @($redisConf)
            } else {
                Write-Info "[Redis] Starting with ephemeral in‑memory settings"
            }

            Write-Info "[Redis] Starting service..."
            Start-ProcSimple "Redis" "redis-server" ($redisArgs -join " ")
            if(Wait-Port $Config.Ports.Redis 25){
                try {
                    # Quick PING (needs redis-cli)
                    if(Test-Cmd redis-cli){
                        $pong = redis-cli -p $Config.Ports.Redis PING 2>$null
                        if($pong -match "PONG"){ Write-Ok "[Redis] Health check PONG" } else { Write-Warn "[Redis] Health check failed (no PONG)" }
                    } else {
                        Write-Warn "[Redis] redis-cli not found; skipped PING"
                    }
                } catch { Write-Warn "[Redis] Post-start check failed: $_" }
            } else {
                Write-Err "[Redis] Failed to open port $($Config.Ports.Redis)"
            }
        } else {
            Write-Warn "[Redis] Cannot start (redis-server binary missing). Use -SkipRedis to suppress."
        }
    }
}

# -------------------- Ollama + Models --------------------
if(-not $NoOllama){
    Write-Section "Ollama"
    if(-not (Test-PortOpen $Config.Ports.Ollama)){
        if(Test-Cmd ollama){
            Start-ProcSimple "Ollama" "ollama" "serve"
            Wait-Port $Config.Ports.Ollama 25 | Out-Null
        }
    } else { Write-Ok "Ollama running" }

    Write-Info "Ensuring models (gemma2:2b, nomic-embed-text, gemma3:legal-latest)"
    $models = (ollama list 2>$null)
    if(-not ($models -match "gemma2:2b")) { Write-Info "Pull gemma2:2b"; ollama pull gemma2:2b | Out-Null }
    if(-not ($models -match "nomic-embed-text")) { Write-Info "Pull nomic-embed-text"; ollama pull nomic-embed-text | Out-Null }
    if(-not ($models -match "gemma3:legal-latest")){
        Write-Info "Create gemma3:legal-latest"
@'
FROM gemma3:legal-latest
SYSTEM """
You are an expert legal AI assistant.
"""
PARAMETER temperature 0.3
PARAMETER num_thread 16
'@ | Set-Content Modelfile-legal
        ollama create gemma3:legal-latest -f Modelfile-legal | Out-Null
        Remove-Item Modelfile-legal -Force
    }
    Write-Ok "Ollama models ready"
}

# -------------------- Go Microservices --------------------
if(-not $SkipGo){
    Write-Section "Go Microservices"
    $goDir = Join-Path $Config.Paths.RepoRoot $Config.Paths.GoRoot
    if(Test-Path $goDir){
        if(Test-Path "$goDir\cmd\enhanced-rag-v2-local\main.go"){
            Start-ProcSimple "Enhanced RAG" "go" "run main.go" "$goDir\cmd\enhanced-rag-v2-local"
        } else { Write-Warn "Enhanced RAG main.go missing" }
        if(Test-Path "$goDir\cmd\gpu-orchestrator\main.go"){
            Start-ProcSimple "GPU Orchestrator" "go" "run ./cmd/gpu-orchestrator" $goDir
        } else { Write-Warn "GPU orchestrator command missing" }
        if(Test-Path "$goDir\go-llama-chat-service.go"){
            Start-ProcSimple "Go-LLaMA" "go" "run go-llama-chat-service.go" $goDir
        } else { Write-Warn "Go-LLaMA service missing" }
    } else {
        Write-Warn "Go microservice directory not found: $goDir"
    }
}

# -------------------- MCP Servers --------------------
Write-Section "MCP Servers"

# context7-mcp-server-multicore.js (original multi-core MCP)
$context7MultiCoreJs = "context7-mcp-server-multicore.js"
if(Test-Path $context7MultiCoreJs){
    Start-ProcSimple "Context7 MCP MultiCore" "node" $context7MultiCoreJs
    Wait-Port 4100 20 | Out-Null
} else {
    Write-Warn "Missing $context7MultiCoreJs"
}

# Enhanced multi-core orchestrated MCP (mcp\context7-multicore.js)
if(Test-Path $Config.Paths.Context7Multi){
    Start-ProcSimple "Context7 Orchestrator" "node" $Config.Paths.Context7Multi
    Wait-Port $Config.Ports.MultiCoreLB 25 | Out-Null
} else {
    Write-Warn "Enhanced multi-core MCP not found at $($Config.Paths.Context7Multi)"
}

# ai-synthesis-mcp (optional)
# Discover AI Synthesis MCP entrypoint (supports multiple layouts)
$aiSynth = $null
$aiSynthCandidates = @(
    "mcp-servers\ai-synthesis-mcp.js",
    "mcp-servers\ai-synthesis-mcp.mjs",
    "mcp-servers\ai-synthesis-mcp.ts",
    "mcp\ai-synthesis-mcp.js",
    "ai-synthesis-mcp\dist\index.js",
    "ai-synthesis\dist\server.js"
)

foreach($c in $aiSynthCandidates){
    if(Test-Path $c){
        $aiSynth = $c
        break
    }
}

if($aiSynth){
    Write-Ok "AI Synthesis MCP found at $aiSynth"
    # Export port (script later uses Config.Ports.AISynthesis)
    $Env:AI_SYNTHESIS_PORT = $Config.Ports.AISynthesis
} else {
    Write-Warn "AI Synthesis MCP not found. Searched: $($aiSynthCandidates -join ', ')"
}
if(Test-Path $aiSynth){
    Start-ProcSimple "AI Synthesis MCP" "node" $aiSynth
    Wait-Port $Config.Ports.AISynthesis 25 | Out-Null
}

# -------------------- Frontend & Install --------------------
if(-not $SkipFrontend){
    Write-Section "Dependencies"
    if(-not (Test-Path "node_modules")){
        Write-Info "Installing npm dependencies..."
        npm install --no-audit --no-fund | Out-Null
        Write-Ok "Dependencies installed"
    } else { Write-Ok "Dependencies present" }

    Write-Section "TypeScript Check / AutoSolve"
    try {
        if(Test-Path "package.json"){
            npm run check:ultra-fast 2>$null | Out-Null
            if($LASTEXITCODE -ne 0 -and (Test-Path "scripts\autosolve-ai-synthesis.mjs")){
                Write-Info "Running AutoSolve..."
                node scripts\autosolve-ai-synthesis.mjs | Out-Null
                Write-Ok "AutoSolve done"
            } else { Write-Ok "Type check passed" }
        }
    } catch { Write-Warn "Check step failed: $_" }

    Write-Section "SvelteKit Dev Server"
    Start-ProcSimple "SvelteKit" "npm" "run dev" "."
    Wait-Port $Config.Ports.Frontend 40 | Out-Null
}

# -------------------- Status Summary --------------------
Write-Section "Service Status"
$summary = @(
    @{Name="Frontend"; Port=$Config.Ports.Frontend}
    @{Name="Context7 MCP"; Port=4100}
    @{Name="Context7 MultiCore LB"; Port=$Config.Ports.MultiCoreLB}
    @{Name="Enhanced RAG"; Port=$Config.Ports.RAG}
    @{Name="GPU Orchestrator"; Port=$Config.Ports.GPU}
    @{Name="Go-LLaMA"; Port=$Config.Ports.GoLLaMA}
    @{Name="AI Synthesis MCP"; Port=$Config.Ports.AISynthesis}
    @{Name="Redis"; Port=$Config.Ports.Redis}
    @{Name="Neo4j Browser"; Port=$Config.Ports.Neo4jHTTP}
    @{Name="Neo4j Bolt"; Port=$Config.Ports.Neo4jBolt}
    @{Name="Ollama"; Port=$Config.Ports.Ollama}
)

foreach($s in $summary){
    $up = Test-PortOpen $s.Port
    if($up){ Write-Ok ("{0,-20} port {1}" -f $s.Name, $s.Port) } else { Write-Warn ("{0,-20} port {1} (down)" -f $s.Name, $s.Port) }
}

Write-Section "Quick Test Commands"
Write-Host "curl http://localhost:$($Config.Ports.Frontend)/api/ai-synthesizer/health"
Write-Host "curl http://localhost:4100/health"
Write-Host "curl http://localhost:$($Config.Ports.MultiCoreLB)/status"

Write-Section "Monitoring"
if(Test-Path "scripts\orchestration\monitor-ai-synthesis.ps1"){
    Write-Info "Launching monitoring dashboard (separate window)"
    Start-Process powershell -ArgumentList "-NoExit","-File","scripts\orchestration\monitor-ai-synthesis.ps1" | Out-Null
}

Write-Section "Runtime"
Write-Info "Press Ctrl+C to terminate this orchestrator (child processes continue unless stopped separately)."
Write-Info "Tail logs with: Get-Process | where { $_.Path -match 'node|go|redis|ollama' }"

# Keep session alive
while($true){ Start-Sleep -Seconds 60 }
