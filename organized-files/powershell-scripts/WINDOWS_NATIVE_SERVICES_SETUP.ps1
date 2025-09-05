#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Windows Native Services Setup for Legal AI Platform
    Creates local Windows services with admin UIs for all required components

.DESCRIPTION
    This script installs and configures:
    - PostgreSQL 17 with pgvector as Windows Service + pgAdmin
    - Redis (Memurai) as Windows Service + Redis Commander
    - RabbitMQ as Windows Service + Management UI
    - Neo4j as Windows Service + Browser UI
    - MinIO as Windows Service + Console UI
    - Ollama as Windows Service
    - Custom Node.js workers as Windows Services
    - Complete admin dashboard setup

.PARAMETER InstallAll
    Install all services and dependencies

.PARAMETER SetupAdminUI
    Setup admin interfaces for all services

.EXAMPLE
    .\WINDOWS_NATIVE_SERVICES_SETUP.ps1 -InstallAll -SetupAdminUI
#>

param(
    [switch]$InstallAll,
    [switch]$SetupAdminUI,
    [switch]$CompileWorkers,
    [string]$ServicePrefix = "LegalAI"
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================================" -ForegroundColor Yellow
Write-Host "🚀 WINDOWS NATIVE SERVICES SETUP - LEGAL AI PLATFORM" -ForegroundColor Yellow  
Write-Host "=================================================================================" -ForegroundColor Yellow
Write-Host ""

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Error "This script must be run as Administrator. Please run PowerShell as Administrator and try again."
    exit 1
}

# Create installation directory
$InstallPath = "C:\LegalAI\Services"
$DataPath = "C:\LegalAI\Data"
$LogPath = "C:\LegalAI\Logs"

New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
New-Item -ItemType Directory -Path $DataPath -Force | Out-Null
New-Item -ItemType Directory -Path $LogPath -Force | Out-Null

Write-Host "📁 Created directory structure at C:\LegalAI\" -ForegroundColor Green

# Install Chocolatey if not present
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Installing Chocolatey package manager..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    refreshenv
}

# Install NSSM (Non-Sucking Service Manager)
Write-Host "🔧 Installing NSSM for service management..." -ForegroundColor Yellow
choco install nssm -y

if ($InstallAll) {
    Write-Host ""
    Write-Host "=================================================================================" -ForegroundColor Cyan
    Write-Host "🗃️ INSTALLING DATABASE SERVICES" -ForegroundColor Cyan
    Write-Host "=================================================================================" -ForegroundColor Cyan

    # 1. PostgreSQL with pgvector
    Write-Host "🐘 Installing PostgreSQL 17 with pgvector..." -ForegroundColor Yellow
    
    # Download PostgreSQL installer
    $PGInstaller = "$env:TEMP\postgresql-17-x64.exe"
    if (!(Test-Path $PGInstaller)) {
        Write-Host "   Downloading PostgreSQL installer..." -ForegroundColor Gray
        Invoke-WebRequest -Uri "https://get.enterprisedb.com/postgresql/postgresql-17.0-1-windows-x64.exe" -OutFile $PGInstaller
    }
    
    # Silent install PostgreSQL
    Write-Host "   Installing PostgreSQL (this may take a few minutes)..." -ForegroundColor Gray
    Start-Process -FilePath $PGInstaller -ArgumentList @(
        "--mode", "unattended",
        "--superpassword", "123456",
        "--servicename", "postgresql-17",
        "--servicepassword", "123456",
        "--serverport", "5432",
        "--datadir", "$DataPath\PostgreSQL\17\data",
        "--enable-components", "server,pgAdmin"
    ) -Wait
    
    Write-Host "✅ PostgreSQL installed as Windows Service" -ForegroundColor Green
    
    # Install pgvector extension
    Write-Host "   Installing pgvector extension..." -ForegroundColor Gray
    $env:PGPASSWORD = "123456"
    & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
    & "C:\Program Files\PostgreSQL\17\bin\createdb.exe" -U postgres legal_ai_db
    & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
    
    Write-Host "✅ pgvector extension installed" -ForegroundColor Green

    # 2. Redis (Memurai - Redis-compatible Windows service)
    Write-Host "🔴 Installing Memurai (Redis for Windows)..." -ForegroundColor Yellow
    
    $MemuraiInstaller = "$env:TEMP\Memurai-Developer-v4.0.6-x64.msi"
    if (!(Test-Path $MemuraiInstaller)) {
        Write-Host "   Downloading Memurai installer..." -ForegroundColor Gray
        Invoke-WebRequest -Uri "https://download.memurai.com/Memurai-Developer-v4.0.6-x64.msi" -OutFile $MemuraiInstaller
    }
    
    Write-Host "   Installing Memurai..." -ForegroundColor Gray
    Start-Process -FilePath "msiexec.exe" -ArgumentList @("/i", $MemuraiInstaller, "/quiet", "/norestart") -Wait
    
    # Configure Memurai
    $MemuraiConfig = @"
# Memurai Configuration for Legal AI Platform
port 6379
bind 0.0.0.0
protected-mode no
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
logfile "$LogPath\memurai.log"
"@
    
    $ConfigPath = "C:\Program Files\Memurai\memurai.conf"
    $MemuraiConfig | Out-File -FilePath $ConfigPath -Encoding utf8
    
    Write-Host "✅ Memurai (Redis) installed as Windows Service" -ForegroundColor Green

    # 3. RabbitMQ
    Write-Host "🐰 Installing RabbitMQ..." -ForegroundColor Yellow
    
    # Install Erlang first (required for RabbitMQ)
    choco install erlang -y
    
    # Install RabbitMQ
    choco install rabbitmq -y
    
    # Enable RabbitMQ Management Plugin
    Write-Host "   Enabling RabbitMQ Management UI..." -ForegroundColor Gray
    & "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.13.0\sbin\rabbitmq-plugins.bat" enable rabbitmq_management
    
    Write-Host "✅ RabbitMQ installed as Windows Service with Management UI" -ForegroundColor Green

    # 4. Neo4j
    Write-Host "🔗 Installing Neo4j Community Edition..." -ForegroundColor Yellow
    
    # Download Neo4j
    $Neo4jZip = "$env:TEMP\neo4j-community-5.15.0-windows.zip"
    if (!(Test-Path $Neo4jZip)) {
        Write-Host "   Downloading Neo4j..." -ForegroundColor Gray
        Invoke-WebRequest -Uri "https://dist.neo4j.org/neo4j-community-5.15.0-windows.zip" -OutFile $Neo4jZip
    }
    
    # Extract Neo4j
    Expand-Archive -Path $Neo4jZip -DestinationPath "C:\Program Files\" -Force
    Rename-Item "C:\Program Files\neo4j-community-5.15.0" "C:\Program Files\Neo4j" -Force
    
    # Configure Neo4j
    $Neo4jConfig = @"
# Neo4j Configuration for Legal AI Platform
server.default_listen_address=0.0.0.0
server.bolt.enabled=true
server.bolt.listen_address=:7687
server.http.enabled=true
server.http.listen_address=:7474
server.https.enabled=false
server.memory.heap.initial_size=512m
server.memory.heap.max_size=1G
server.memory.pagecache.size=512m
server.directories.data=$DataPath\Neo4j\data
server.directories.logs=$LogPath\Neo4j
dbms.security.auth_enabled=false
"@
    
    $Neo4jConfigPath = "C:\Program Files\Neo4j\conf\neo4j.conf"
    $Neo4jConfig | Out-File -FilePath $Neo4jConfigPath -Encoding utf8
    
    # Install Neo4j as Windows Service
    & "C:\Program Files\Neo4j\bin\neo4j.bat" install-service
    
    Write-Host "✅ Neo4j installed as Windows Service" -ForegroundColor Green

    # 5. MinIO
    Write-Host "🪣 Installing MinIO..." -ForegroundColor Yellow
    
    # Download MinIO
    $MinIOExe = "$InstallPath\minio.exe"
    if (!(Test-Path $MinIOExe)) {
        Write-Host "   Downloading MinIO..." -ForegroundColor Gray
        Invoke-WebRequest -Uri "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" -OutFile $MinIOExe
    }
    
    # Create MinIO data directory
    New-Item -ItemType Directory -Path "$DataPath\MinIO" -Force | Out-Null
    
    # Create MinIO service using NSSM
    $MinIOArgs = @(
        "server",
        "$DataPath\MinIO",
        "--console-address", ":9001",
        "--address", ":9000"
    )
    
    & nssm install "${ServicePrefix}-MinIO" $MinIOExe
    & nssm set "${ServicePrefix}-MinIO" AppParameters ($MinIOArgs -join " ")
    & nssm set "${ServicePrefix}-MinIO" AppDirectory $InstallPath
    & nssm set "${ServicePrefix}-MinIO" DisplayName "Legal AI - MinIO Object Storage"
    & nssm set "${ServicePrefix}-MinIO" Description "MinIO object storage service for Legal AI Platform"
    & nssm set "${ServicePrefix}-MinIO" Start SERVICE_AUTO_START
    & nssm set "${ServicePrefix}-MinIO" AppEnvironmentExtra "MINIO_ROOT_USER=minioadmin" "MINIO_ROOT_PASSWORD=minioadmin123" "MINIO_BROWSER_REDIRECT_URL=http://localhost:9001"
    & nssm set "${ServicePrefix}-MinIO" AppStdout "$LogPath\minio-stdout.log"
    & nssm set "${ServicePrefix}-MinIO" AppStderr "$LogPath\minio-stderr.log"
    
    Write-Host "✅ MinIO installed as Windows Service" -ForegroundColor Green

    # 6. Qdrant Vector Database
    Write-Host "🔍 Installing Qdrant Vector Database..." -ForegroundColor Yellow
    
    # Download Qdrant
    $QdrantExe = "$InstallPath\qdrant.exe"
    if (!(Test-Path $QdrantExe)) {
        Write-Host "   Downloading Qdrant..." -ForegroundColor Gray
        Invoke-WebRequest -Uri "https://github.com/qdrant/qdrant/releases/download/v1.7.3/qdrant-x86_64-pc-windows-msvc.zip" -OutFile "$env:TEMP\qdrant.zip"
        Expand-Archive -Path "$env:TEMP\qdrant.zip" -DestinationPath $env:TEMP -Force
        Move-Item "$env:TEMP\qdrant.exe" $QdrantExe -Force
    }
    
    # Create Qdrant configuration
    $QdrantConfig = @"
log_level: INFO
storage:
  storage_path: "$($DataPath.Replace('\', '/'))/Qdrant"
service:
  http_port: 6333
  grpc_port: 6334
  host: 0.0.0.0
cluster:
  enabled: false
"@
    
    $QdrantConfigPath = "$InstallPath\qdrant-config.yaml"
    $QdrantConfig | Out-File -FilePath $QdrantConfigPath -Encoding utf8
    
    # Install Qdrant as Windows Service
    & nssm install "${ServicePrefix}-Qdrant" $QdrantExe
    & nssm set "${ServicePrefix}-Qdrant" AppParameters "--config-path `"$QdrantConfigPath`""
    & nssm set "${ServicePrefix}-Qdrant" AppDirectory $InstallPath
    & nssm set "${ServicePrefix}-Qdrant" DisplayName "Legal AI - Qdrant Vector Database"
    & nssm set "${ServicePrefix}-Qdrant" Description "Qdrant vector database service for Legal AI Platform"
    & nssm set "${ServicePrefix}-Qdrant" Start SERVICE_AUTO_START
    & nssm set "${ServicePrefix}-Qdrant" AppStdout "$LogPath\qdrant-stdout.log"
    & nssm set "${ServicePrefix}-Qdrant" AppStderr "$LogPath\qdrant-stderr.log"
    
    Write-Host "✅ Qdrant installed as Windows Service" -ForegroundColor Green

    # 7. Ollama
    Write-Host "🦙 Installing Ollama..." -ForegroundColor Yellow
    
    # Download and install Ollama
    $OllamaInstaller = "$env:TEMP\OllamaSetup.exe"
    if (!(Test-Path $OllamaInstaller)) {
        Write-Host "   Downloading Ollama..." -ForegroundColor Gray
        Invoke-WebRequest -Uri "https://ollama.com/download/windows" -OutFile $OllamaInstaller
    }
    
    Start-Process -FilePath $OllamaInstaller -ArgumentList "/S" -Wait
    
    # Configure Ollama as service (it auto-installs as service)
    Write-Host "   Pulling required AI models..." -ForegroundColor Gray
    Start-Sleep 10  # Wait for Ollama to start
    
    # Pull required models
    & ollama pull gemma2:2b
    & ollama pull nomic-embed-text:latest
    & ollama pull llama3.1:8b
    
    Write-Host "✅ Ollama installed as Windows Service with AI models" -ForegroundColor Green
}

if ($CompileWorkers -or $InstallAll) {
    Write-Host ""
    Write-Host "=================================================================================" -ForegroundColor Cyan
    Write-Host "⚙️ COMPILING AND INSTALLING WORKER SERVICES" -ForegroundColor Cyan
    Write-Host "=================================================================================" -ForegroundColor Cyan

    # Compile TypeScript workers to JavaScript
    Write-Host "📦 Compiling TypeScript workers..." -ForegroundColor Yellow
    
    $WorkerPath = "$InstallPath\Workers"
    New-Item -ItemType Directory -Path $WorkerPath -Force | Out-Null
    
    # Copy and compile workers
    $SourcePath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
    
    if (Test-Path "$SourcePath\node_modules") {
        Push-Location $SourcePath
        
        Write-Host "   Installing dependencies..." -ForegroundColor Gray
        & npm install
        
        Write-Host "   Compiling TypeScript..." -ForegroundColor Gray
        & npx tsc --project tsconfig.json --outDir "$WorkerPath\compiled"
        
        Pop-Location
        
        # Create worker service scripts
        $Workers = @(
            @{
                Name = "VectorProcessor"
                Script = "vector-processor-worker.js"
                Description = "Vector processing and embedding service"
            },
            @{
                Name = "DocumentProcessor" 
                Script = "document-processor-worker.js"
                Description = "Document ingestion and analysis service"
            },
            @{
                Name = "RAGService"
                Script = "rag-service-worker.js"
                Description = "Retrieval-Augmented Generation service"
            }
        )
        
        foreach ($Worker in $Workers) {
            $ServiceName = "${ServicePrefix}-$($Worker.Name)"
            $ScriptPath = "$WorkerPath\compiled\$($Worker.Script)"
            
            if (Test-Path $ScriptPath) {
                Write-Host "   Installing $($Worker.Name) as Windows Service..." -ForegroundColor Gray
                
                & nssm install $ServiceName "node"
                & nssm set $ServiceName AppParameters "`"$ScriptPath`""
                & nssm set $ServiceName AppDirectory $WorkerPath
                & nssm set $ServiceName DisplayName "Legal AI - $($Worker.Name)"
                & nssm set $ServiceName Description $Worker.Description
                & nssm set $ServiceName Start SERVICE_AUTO_START
                & nssm set $ServiceName AppStdout "$LogPath\$($Worker.Name)-stdout.log"
                & nssm set $ServiceName AppStderr "$LogPath\$($Worker.Name)-stderr.log"
                
                Write-Host "✅ $($Worker.Name) installed as Windows Service" -ForegroundColor Green
            } else {
                Write-Host "⚠️ Worker script not found: $ScriptPath" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "⚠️ Node.js project not found at $SourcePath" -ForegroundColor Yellow
        Write-Host "   Please ensure the SvelteKit project is available and run npm install" -ForegroundColor Gray
    }
}

if ($SetupAdminUI -or $InstallAll) {
    Write-Host ""
    Write-Host "=================================================================================" -ForegroundColor Cyan
    Write-Host "🖥️ SETTING UP ADMIN INTERFACES" -ForegroundColor Cyan
    Write-Host "=================================================================================" -ForegroundColor Cyan

    # Install Redis Commander (Web UI for Redis)
    Write-Host "🔴 Installing Redis Commander..." -ForegroundColor Yellow
    & npm install -g redis-commander
    
    # Create Redis Commander service
    $RedisCommanderScript = "$InstallPath\redis-commander.js"
    $RedisCommanderContent = @"
const redisCommander = require('redis-commander');
const server = redisCommander({
    redis: {
        port: 6379,
        host: 'localhost'
    },
    web: {
        port: 8081,
        host: '0.0.0.0'
    }
});
server.listen();
console.log('Redis Commander started on http://localhost:8081');
"@
    
    $RedisCommanderContent | Out-File -FilePath $RedisCommanderScript -Encoding utf8
    
    & nssm install "${ServicePrefix}-RedisCommander" "node"
    & nssm set "${ServicePrefix}-RedisCommander" AppParameters "`"$RedisCommanderScript`""
    & nssm set "${ServicePrefix}-RedisCommander" DisplayName "Legal AI - Redis Commander"
    & nssm set "${ServicePrefix}-RedisCommander" Description "Web UI for Redis administration"
    & nssm set "${ServicePrefix}-RedisCommander" Start SERVICE_AUTO_START
    & nssm set "${ServicePrefix}-RedisCommander" AppStdout "$LogPath\redis-commander-stdout.log"
    & nssm set "${ServicePrefix}-RedisCommander" AppStderr "$LogPath\redis-commander-stderr.log"
    
    Write-Host "✅ Redis Commander installed as Windows Service" -ForegroundColor Green

    # Create Admin Dashboard HTML
    Write-Host "🖥️ Creating Admin Dashboard..." -ForegroundColor Yellow
    
    $DashboardHTML = @"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legal AI Platform - Admin Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; text-align: center; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h3 { color: #2c3e50; margin-bottom: 15px; }
        .service-link { display: block; padding: 10px; background: #3498db; color: white; text-decoration: none; margin: 5px 0; border-radius: 4px; text-align: center; }
        .service-link:hover { background: #2980b9; }
        .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
        .status-green { background: #27ae60; }
        .status-red { background: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Legal AI Platform - Admin Dashboard</h1>
            <p>Centralized administration for all services and databases</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 Database Administration</h3>
                <a href="http://localhost:5050" class="service-link">
                    <span class="status-indicator status-green"></span>pgAdmin - PostgreSQL
                </a>
                <a href="http://localhost:8081" class="service-link">
                    <span class="status-indicator status-green"></span>Redis Commander
                </a>
                <a href="http://localhost:7474" class="service-link">
                    <span class="status-indicator status-green"></span>Neo4j Browser
                </a>
                <a href="http://localhost:6333/dashboard" class="service-link">
                    <span class="status-indicator status-green"></span>Qdrant Dashboard
                </a>
            </div>
            
            <div class="card">
                <h3>🐰 Message Queue & Storage</h3>
                <a href="http://localhost:15672" class="service-link">
                    <span class="status-indicator status-green"></span>RabbitMQ Management
                </a>
                <a href="http://localhost:9001" class="service-link">
                    <span class="status-indicator status-green"></span>MinIO Console
                </a>
                <a href="http://localhost:4222" class="service-link">
                    <span class="status-indicator status-green"></span>NATS Monitoring
                </a>
            </div>
            
            <div class="card">
                <h3>🤖 AI & Processing Services</h3>
                <a href="http://localhost:11434" class="service-link">
                    <span class="status-indicator status-green"></span>Ollama API
                </a>
                <a href="http://localhost:8094/health" class="service-link">
                    <span class="status-indicator status-green"></span>Enhanced RAG Service
                </a>
                <a href="http://localhost:8093/health" class="service-link">
                    <span class="status-indicator status-green"></span>Upload Service
                </a>
            </div>
            
            <div class="card">
                <h3>🌐 Frontend & API</h3>
                <a href="http://localhost:5173" class="service-link">
                    <span class="status-indicator status-green"></span>SvelteKit Frontend
                </a>
                <a href="http://localhost:5173/api/health" class="service-link">
                    <span class="status-indicator status-green"></span>API Health Check
                </a>
            </div>
            
            <div class="card">
                <h3>🛠️ System Management</h3>
                <a href="#" onclick="checkServices()" class="service-link">
                    <span class="status-indicator status-green"></span>Refresh Service Status
                </a>
                <a href="http://localhost:8080/logs" class="service-link">
                    <span class="status-indicator status-green"></span>Service Logs
                </a>
            </div>
        </div>
    </div>
    
    <script>
        async function checkServices() {
            const services = [
                'http://localhost:5173',
                'http://localhost:8093/health', 
                'http://localhost:8094/health',
                'http://localhost:6379',
                'http://localhost:5432',
                'http://localhost:7474',
                'http://localhost:15672',
                'http://localhost:9001',
                'http://localhost:6333'
            ];
            
            for (const service of services) {
                try {
                    const response = await fetch(service);
                    console.log(\`\${service}: \${response.ok ? 'OK' : 'Error'}\`);
                } catch (error) {
                    console.log(\`\${service}: Offline\`);
                }
            }
        }
        
        // Auto-refresh every 30 seconds
        setInterval(checkServices, 30000);
    </script>
</body>
</html>
"@
    
    $DashboardPath = "$InstallPath\admin-dashboard.html"
    $DashboardHTML | Out-File -FilePath $DashboardPath -Encoding utf8
    
    Write-Host "✅ Admin Dashboard created at $DashboardPath" -ForegroundColor Green
}

# Start all services
Write-Host ""
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "🚀 STARTING ALL SERVICES" -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan

$Services = @(
    "postgresql-17",
    "${ServicePrefix}-MinIO", 
    "${ServicePrefix}-Qdrant",
    "${ServicePrefix}-RedisCommander",
    "RabbitMQ",
    "Neo4j",
    "Ollama"
)

foreach ($ServiceName in $Services) {
    Write-Host "▶️ Starting $ServiceName..." -ForegroundColor Yellow
    try {
        Start-Service -Name $ServiceName -ErrorAction Stop
        Write-Host "✅ $ServiceName started successfully" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Could not start $ServiceName - $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Final summary
Write-Host ""
Write-Host "=================================================================================" -ForegroundColor Green
Write-Host "✅ WINDOWS NATIVE SERVICES INSTALLATION COMPLETE" -ForegroundColor Green
Write-Host "=================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 ADMIN INTERFACES:" -ForegroundColor Cyan
Write-Host "   • Admin Dashboard:     file://$DashboardPath" -ForegroundColor White
Write-Host "   • pgAdmin:            http://localhost:5050" -ForegroundColor White
Write-Host "   • Redis Commander:    http://localhost:8081" -ForegroundColor White
Write-Host "   • RabbitMQ Management: http://localhost:15672 (guest/guest)" -ForegroundColor White
Write-Host "   • Neo4j Browser:      http://localhost:7474" -ForegroundColor White
Write-Host "   • MinIO Console:      http://localhost:9001 (minioadmin/minioadmin123)" -ForegroundColor White
Write-Host "   • Qdrant Dashboard:   http://localhost:6333/dashboard" -ForegroundColor White
Write-Host ""
Write-Host "📊 SERVICE STATUS:" -ForegroundColor Cyan
Write-Host "   Use 'Get-Service *LegalAI*' to check custom service status" -ForegroundColor White
Write-Host "   Use 'services.msc' to manage all Windows services" -ForegroundColor White
Write-Host ""
Write-Host "📁 INSTALLATION PATHS:" -ForegroundColor Cyan
Write-Host "   • Services:    $InstallPath" -ForegroundColor White
Write-Host "   • Data:        $DataPath" -ForegroundColor White
Write-Host "   • Logs:        $LogPath" -ForegroundColor White
Write-Host ""
Write-Host "🎯 NEXT STEPS:" -ForegroundColor Cyan
Write-Host "   1. Open Admin Dashboard to verify all services" -ForegroundColor White
Write-Host "   2. Run your SvelteKit application: npm run dev" -ForegroundColor White
Write-Host "   3. Access the main application at http://localhost:5173" -ForegroundColor White
Write-Host ""

# Open admin dashboard
Write-Host "🚀 Opening Admin Dashboard..." -ForegroundColor Yellow
Start-Process $DashboardPath