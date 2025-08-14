# Legal AI System - Safe Installation Script
# PowerShell 7+ Required
# Run as Administrator for Docker operations

param(
    [switch]$SkipDocker = $false,
    [switch]$SkipModels = $false,
    [switch]$Force = $false,
    [switch]$NoBackup = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'

# Configuration
$script:Config = @{
    ProjectRoot = $PSScriptRoot
    BackupDir = Join-Path $PSScriptRoot "backups"
    RequiredPorts = @(5432, 6379, 6333, 11434, 7474, 7687, 8001)
    RequiredServices = @('postgres', 'redis', 'qdrant', 'ollama', 'neo4j')
    VectorDimensions = 384
    EmbeddingModel = "nomic-embed-text"
    LLMModel = "gemma3-legal"
}

# Logging functions
function Write-Step {
    param([string]$Message, [string]$Type = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch ($Type) {
        "INFO" { "Cyan" }
        "SUCCESS" { "Green" }
        "WARNING" { "Yellow" }
        "ERROR" { "Red" }
        default { "White" }
    }
    Write-Host "[$timestamp] [$Type] $Message" -ForegroundColor $color
}

function Test-ExistingInstallation {
    Write-Step "Checking for existing installation..."
    
    $exists = @{
        Docker = $false
        Database = $false
        Frontend = $false
        Models = $false
    }
    
    # Check Docker containers
    try {
        $containers = docker ps -a --format "{{.Names}}" 2>$null
        if ($containers -match "legal_ai_") {
            $exists.Docker = $true
            Write-Step "Found existing Docker containers" "WARNING"
        }
    } catch {}
    
    # Check database
    if ($exists.Docker) {
        try {
            $tables = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'" 2>$null
            if ($tables -and [int]$tables.Trim() -gt 0) {
                $exists.Database = $true
                Write-Step "Found existing database with $($tables.Trim()) tables" "WARNING"
            }
        } catch {}
    }
    
    # Check frontend
    if (Test-Path (Join-Path $Config.ProjectRoot "sveltekit-frontend\node_modules")) {
        $exists.Frontend = $true
        Write-Step "Found existing frontend installation" "WARNING"
    }
    
    # Check models
    try {
        $models = docker exec legal_ai_ollama ollama list 2>$null
        if ($models) {
            $exists.Models = $true
            Write-Step "Found existing AI models" "WARNING"
        }
    } catch {}
    
    return $exists
}

function Backup-ExistingSystem {
    param([hashtable]$ExistingItems)
    
    if ($NoBackup) {
        Write-Step "Skipping backup (-NoBackup flag set)" "WARNING"
        return $null
    }
    
    Write-Step "Creating backup of existing system..."
    
    # Create backup directory
    if (-not (Test-Path $Config.BackupDir)) {
        New-Item -ItemType Directory -Path $Config.BackupDir -Force | Out-Null
    }
    
    $backupName = "pre-install-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    $backupPath = Join-Path $Config.BackupDir $backupName
    New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
    
    $backupManifest = @{
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Items = @()
    }
    
    try {
        # Backup database if exists
        if ($ExistingItems.Database) {
            Write-Step "Backing up PostgreSQL database..."
            $dbBackup = Join-Path $backupPath "prosecutor_db.sql"
            docker exec legal_ai_postgres pg_dump -U postgres prosecutor_db > $dbBackup
            if (Test-Path $dbBackup) {
                $backupManifest.Items += "Database"
                Write-Step "Database backed up successfully" "SUCCESS"
            }
        }
        
        # Backup Qdrant collections if exist
        if ($ExistingItems.Docker) {
            Write-Step "Backing up Qdrant collections..."
            try {
                $collections = Invoke-RestMethod -Uri "http://localhost:6333/collections" -Method GET
                foreach ($col in $collections.result.collections) {
                    $snapshot = Invoke-RestMethod -Uri "http://localhost:6333/collections/$($col.name)/snapshots" -Method POST
                    $backupManifest.Items += "Qdrant:$($col.name)"
                    Write-Step "Backed up collection: $($col.name)" "SUCCESS"
                }
            } catch {
                Write-Step "Could not backup Qdrant collections: $_" "WARNING"
            }
        }
        
        # Backup configuration files
        Write-Step "Backing up configuration files..."
        $configFiles = @(
            ".env",
            "docker-compose.yml",
            "package.json",
            "package-lock.json",
            "sveltekit-frontend\.env",
            "sveltekit-frontend\package.json",
            "sveltekit-frontend\package-lock.json"
        )
        
        foreach ($file in $configFiles) {
            $fullPath = Join-Path $Config.ProjectRoot $file
            if (Test-Path $fullPath) {
                $destPath = Join-Path $backupPath $file.Replace('\', '_')
                Copy-Item $fullPath $destPath -Force
                $backupManifest.Items += "Config:$file"
            }
        }
        
        # Save backup manifest
        $manifestPath = Join-Path $backupPath "backup-manifest.json"
        $backupManifest | ConvertTo-Json -Depth 5 | Out-File $manifestPath
        
        # Compress backup
        $zipPath = "$backupPath.zip"
        Write-Step "Compressing backup..."
        Compress-Archive -Path $backupPath -DestinationPath $zipPath -Force
        Remove-Item $backupPath -Recurse -Force
        
        Write-Step "Backup completed: $zipPath" "SUCCESS"
        return $zipPath
        
    } catch {
        Write-Step "Backup failed: $_" "ERROR"
        if (-not $Force) {
            throw "Cannot proceed without backup. Use -Force to override (DANGEROUS)"
        }
        return $null
    }
}

function Confirm-Installation {
    param([hashtable]$ExistingItems, [string]$BackupPath)
    
    if ($Force) {
        Write-Step "Force mode enabled - skipping confirmation" "WARNING"
        return $true
    }
    
    $hasExisting = ($ExistingItems.Values | Where-Object { $_ -eq $true }).Count -gt 0
    
    if ($hasExisting) {
        Write-Host "`n⚠️  EXISTING INSTALLATION DETECTED" -ForegroundColor Yellow
        Write-Host "══════════════════════════════════════" -ForegroundColor Yellow
        
        if ($ExistingItems.Database) {
            Write-Host "• Database with existing data" -ForegroundColor White
        }
        if ($ExistingItems.Docker) {
            Write-Host "• Docker containers running" -ForegroundColor White
        }
        if ($ExistingItems.Frontend) {
            Write-Host "• Frontend node_modules" -ForegroundColor White
        }
        if ($ExistingItems.Models) {
            Write-Host "• AI models installed" -ForegroundColor White
        }
        
        if ($BackupPath) {
            Write-Host "`n✅ Backup created at:" -ForegroundColor Green
            Write-Host "   $BackupPath" -ForegroundColor White
        } else {
            Write-Host "`n❌ NO BACKUP CREATED" -ForegroundColor Red
        }
        
        Write-Host "`nThis installation will OVERWRITE existing data!" -ForegroundColor Yellow
        Write-Host "Are you sure you want to continue? (yes/no): " -ForegroundColor Yellow -NoNewline
        
        $response = Read-Host
        if ($response -ne "yes") {
            Write-Step "Installation cancelled by user" "INFO"
            return $false
        }
    }
    
    return $true
}

function Test-Prerequisites {
    Write-Step "Checking prerequisites..."
    
    # Check PowerShell version
    if ($PSVersionTable.PSVersion.Major -lt 7) {
        Write-Step "PowerShell 7+ required. Current version: $($PSVersionTable.PSVersion)" "ERROR"
        Write-Step "Install from: https://github.com/PowerShell/PowerShell/releases" "INFO"
        return $false
    }
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Step "Docker found: $dockerVersion" "SUCCESS"
    } catch {
        Write-Step "Docker not found. Please install Docker Desktop" "ERROR"
        return $false
    }
    
    # Check Node.js
    try {
        $nodeVersion = node --version
        Write-Step "Node.js found: $nodeVersion" "SUCCESS"
    } catch {
        Write-Step "Node.js not found. Please install Node.js 18+" "ERROR"
        return $false
    }
    
    # Check ports
    $blockedPorts = @()
    foreach ($port in $Config.RequiredPorts) {
        $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue -InformationLevel Quiet
        if ($connection) {
            $blockedPorts += $port
        }
    }
    
    if ($blockedPorts.Count -gt 0) {
        Write-Step "Following ports are already in use: $($blockedPorts -join ', ')" "WARNING"
        Write-Step "Run 'docker-compose down' to stop existing services" "INFO"
    }
    
    return $true
}

function Install-RootDependencies {
    Write-Step "Installing root dependencies..."
    
    Push-Location $Config.ProjectRoot
    try {
        # Backup existing package-lock.json if exists
        if (Test-Path "package-lock.json" -and -not $NoBackup) {
            Copy-Item "package-lock.json" "package-lock.json.bak" -Force
        }
        
        # Install root package dependencies
        if (Test-Path "package.json") {
            Write-Step "Installing npm packages..."
            npm install
            if ($LASTEXITCODE -ne 0) {
                # Try with force if normal install fails
                Write-Step "Retrying with --force flag..." "WARNING"
                npm install --force
                if ($LASTEXITCODE -ne 0) {
                    throw "npm install failed"
                }
            }
            Write-Step "Root dependencies installed" "SUCCESS"
        }
        
        # Install frontend dependencies
        $frontendPath = Join-Path $Config.ProjectRoot "sveltekit-frontend"
        if (Test-Path $frontendPath) {
            Push-Location $frontendPath
            
            # Backup frontend package-lock.json
            if (Test-Path "package-lock.json" -and -not $NoBackup) {
                Copy-Item "package-lock.json" "package-lock.json.bak" -Force
            }
            
            Write-Step "Installing frontend dependencies..."
            npm install
            if ($LASTEXITCODE -ne 0) {
                Write-Step "Frontend npm install failed - retrying with force" "WARNING"
                npm install --force
                if ($LASTEXITCODE -ne 0) {
                    Add-TodoItem "Fix frontend npm dependencies" "Check package-lock.json conflicts"
                }
            }
            Pop-Location
        }
    } finally {
        Pop-Location
    }
}

function Start-DockerServices {
    if ($SkipDocker) {
        Write-Step "Skipping Docker services (--SkipDocker flag set)" "WARNING"
        return
    }
    
    Write-Step "Starting Docker services..."
    
    Push-Location $Config.ProjectRoot
    try {
        # Check if we should preserve volumes
        $preserveVolumes = $false
        if (Test-Path ".preserve-volumes") {
            $preserveVolumes = $true
            Write-Step "Preserving Docker volumes (.preserve-volumes file found)" "INFO"
        }
        
        # Stop existing containers
        Write-Step "Stopping existing containers..."
        if ($preserveVolumes) {
            docker-compose stop
        } else {
            docker-compose down --remove-orphans
        }
        
        # Start services
        Write-Step "Starting infrastructure services..."
        docker-compose up -d postgres redis qdrant
        
        # Wait for services to be healthy
        Write-Step "Waiting for services to be healthy..."
        $maxAttempts = 30
        $attempt = 0
        
        while ($attempt -lt $maxAttempts) {
            $healthy = $true
            
            # Check PostgreSQL
            try {
                docker exec legal_ai_postgres pg_isready -U postgres -d prosecutor_db | Out-Null
                if ($LASTEXITCODE -ne 0) { $healthy = $false }
            } catch { $healthy = $false }
            
            # Check Redis
            try {
                docker exec legal_ai_redis redis-cli ping | Out-Null
                if ($LASTEXITCODE -ne 0) { $healthy = $false }
            } catch { $healthy = $false }
            
            # Check Qdrant
            try {
                $qdrantHealth = Invoke-RestMethod -Uri "http://localhost:6333/health" -Method GET
                if ($qdrantHealth.status -ne "ok") { $healthy = $false }
            } catch { $healthy = $false }
            
            if ($healthy) {
                Write-Step "All services are healthy!" "SUCCESS"
                break
            }
            
            $attempt++
            Write-Host "." -NoNewline
            Start-Sleep -Seconds 2
        }
        
        if (-not $healthy) {
            Write-Step "Services failed to start properly" "ERROR"
            docker-compose logs --tail=50
            throw "Docker services unhealthy"
        }
        
        # Initialize database only if needed
        Write-Step "Checking database initialization..."
        $tableCount = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'" 2>$null
        
        if (-not $tableCount -or [int]$tableCount.Trim() -eq 0) {
            Write-Step "Initializing database..."
            Get-Content (Join-Path $Config.ProjectRoot "database\db-init.sql") | docker exec -i legal_ai_postgres psql -U postgres -d prosecutor_db
        } else {
            Write-Step "Database already initialized with $($tableCount.Trim()) tables" "INFO"
        }
        
        # Initialize Qdrant collections
        Write-Step "Initializing Qdrant collections..."
        $qdrantConfig = Get-Content (Join-Path $Config.ProjectRoot "database\qdrant-init.json") | ConvertFrom-Json
        
        foreach ($collection in $qdrantConfig.collections) {
            try {
                # Check if collection exists
                $exists = $false
                try {
                    $existing = Invoke-RestMethod -Uri "http://localhost:6333/collections/$($collection.name)" -Method GET
                    if ($existing.result) {
                        $exists = $true
                        Write-Step "Collection $($collection.name) already exists" "INFO"
                    }
                } catch {}
                
                if (-not $exists) {
                    $body = @{
                        vectors = @{
                            size = $collection.vectors.size
                            distance = $collection.vectors.distance
                        }
                        shard_number = $collection.shard_number
                        replication_factor = $collection.replication_factor
                        optimizers_config = $collection.optimizers_config
                        hnsw_config = $collection.hnsw_config
                    } | ConvertTo-Json -Depth 10
                    
                    Invoke-RestMethod -Uri "http://localhost:6333/collections/$($collection.name)" -Method PUT -Body $body -ContentType "application/json"
                    Write-Step "Created collection: $($collection.name)" "SUCCESS"
                }
            } catch {
                Write-Step "Failed to create collection $($collection.name): $_" "WARNING"
                Add-TodoItem "Fix Qdrant collection" "Collection: $($collection.name), Error: $_"
            }
        }
        
    } finally {
        Pop-Location
    }
}

function Install-AIModels {
    if ($SkipModels) {
        Write-Step "Skipping AI models (--SkipModels flag set)" "WARNING"
        return
    }
    
    Write-Step "Installing AI models..."
    
    # Start Ollama service
    Write-Step "Starting Ollama service..."
    docker-compose up -d ollama
    Start-Sleep -Seconds 5
    
    # Check if Ollama is running locally
    $ollamaLocal = $false
    try {
        $localOllama = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET
        $ollamaLocal = $true
        Write-Step "Using local Ollama instance" "INFO"
    } catch {
        Write-Step "Using Docker Ollama instance" "INFO"
    }
    
    # Check existing models
    $existingModels = @()
    try {
        if ($ollamaLocal) {
            $models = ollama list 2>$null
        } else {
            $models = docker exec legal_ai_ollama ollama list 2>$null
        }
        $existingModels = $models -split "`n" | ForEach-Object { $_.Split()[0] }
    } catch {}
    
    # Pull embedding model if not exists
    if ($existingModels -notcontains "nomic-embed-text") {
        Write-Step "Pulling nomic-embed-text model..."
        if ($ollamaLocal) {
            ollama pull nomic-embed-text
        } else {
            docker exec legal_ai_ollama ollama pull nomic-embed-text
        }
    } else {
        Write-Step "nomic-embed-text model already installed" "INFO"
    }
    
    # Create custom legal model
    $modelfilePath = Join-Path $Config.ProjectRoot "local-models\Modelfile.gemma3-legal"
    if (Test-Path $modelfilePath) {
        if ($existingModels -notcontains "gemma3-legal") {
            Write-Step "Creating gemma3-legal model..."
            if ($ollamaLocal) {
                ollama create gemma3-legal -f $modelfilePath
            } else {
                docker cp $modelfilePath legal_ai_ollama:/tmp/Modelfile.gemma3-legal
                docker exec legal_ai_ollama ollama create gemma3-legal -f /tmp/Modelfile.gemma3-legal
            }
        } else {
            Write-Step "gemma3-legal model already exists" "INFO"
        }
    } else {
        Write-Step "Modelfile not found, using base gemma model" "WARNING"
        if ($existingModels -notcontains "gemma:2b") {
            if ($ollamaLocal) {
                ollama pull gemma:2b
            } else {
                docker exec legal_ai_ollama ollama pull gemma:2b
            }
        }
    }
    
    Write-Step "AI models installed" "SUCCESS"
}

function Initialize-Database {
    Write-Step "Running database migrations..."
    
    Push-Location $Config.ProjectRoot
    try {
        # Check if migrations needed
        $tableCount = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'" 2>$null
        
        if ($tableCount -and [int]$tableCount.Trim() -gt 10) {
            Write-Step "Database appears to be already migrated ($($tableCount.Trim()) tables)" "INFO"
            $response = Read-Host "Run migrations anyway? (yes/no)"
            if ($response -ne "yes") {
                return
            }
        }
        
        # Run Drizzle migrations
        npm run db:push
        if ($LASTEXITCODE -ne 0) {
            Write-Step "Database migration failed" "ERROR"
            Add-TodoItem "Fix database migrations" "Check drizzle config and schema"
        } else {
            Write-Step "Database migrations completed" "SUCCESS"
        }
    } finally {
        Pop-Location
    }
}

function Add-TodoItem {
    param(
        [string]$Title,
        [string]$Description
    )
    
    $todoFile = Join-Path $Config.ProjectRoot "TODO.md"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    if (-not (Test-Path $todoFile)) {
        "# Legal AI System - TODO List`n`n" | Out-File $todoFile
    }
    
    $todoEntry = @"

## [$timestamp] $Title
- **Status**: Pending
- **Description**: $Description
- **Created by**: install.ps1
---
"@
    
    Add-Content -Path $todoFile -Value $todoEntry
    Write-Step "Added TODO: $Title" "WARNING"
}

function Test-SystemHealth {
    Write-Step "Running system health checks..."
    
    $healthResults = @{
        Overall = $true
        Services = @{}
    }
    
    # Test PostgreSQL
    try {
        docker exec legal_ai_postgres pg_isready -U postgres -d prosecutor_db | Out-Null
        $healthResults.Services["PostgreSQL"] = @{
            Status = "Healthy"
            Details = "Database accepting connections"
        }
    } catch {
        $healthResults.Services["PostgreSQL"] = @{
            Status = "Unhealthy"
            Details = $_
        }
        $healthResults.Overall = $false
    }
    
    # Test Redis
    try {
        $redisPing = docker exec legal_ai_redis redis-cli ping
        if ($redisPing -eq "PONG") {
            $healthResults.Services["Redis"] = @{
                Status = "Healthy"
                Details = "Redis responding to ping"
            }
        }
    } catch {
        $healthResults.Services["Redis"] = @{
            Status = "Unhealthy"
            Details = $_
        }
        $healthResults.Overall = $false
    }
    
    # Test Qdrant
    try {
        $qdrantHealth = Invoke-RestMethod -Uri "http://localhost:6333/health" -Method GET
        $healthResults.Services["Qdrant"] = @{
            Status = "Healthy"
            Details = "Version: $($qdrantHealth.version)"
        }
    } catch {
        $healthResults.Services["Qdrant"] = @{
            Status = "Unhealthy"
            Details = $_
        }
        $healthResults.Overall = $false
    }
    
    # Test Ollama
    try {
        $ollamaModels = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET
        $modelCount = $ollamaModels.models.Count
        $healthResults.Services["Ollama"] = @{
            Status = "Healthy"
            Details = "Models loaded: $modelCount"
        }
    } catch {
        $healthResults.Services["Ollama"] = @{
            Status = "Unhealthy"
            Details = $_
        }
        $healthResults.Overall = $false
    }
    
    # Display results
    Write-Host "`n=== System Health Report ===" -ForegroundColor Cyan
    foreach ($service in $healthResults.Services.Keys) {
        $status = $healthResults.Services[$service]
        $color = if ($status.Status -eq "Healthy") { "Green" } else { "Red" }
        Write-Host "$service : $($status.Status) - $($status.Details)" -ForegroundColor $color
    }
    
    if ($healthResults.Overall) {
        Write-Step "All systems operational!" "SUCCESS"
    } else {
        Write-Step "Some services are unhealthy. Check logs with: docker-compose logs" "ERROR"
    }
    
    return $healthResults
}

function Write-RestoreInstructions {
    param([string]$BackupPath)
    
    if (-not $BackupPath) { return }
    
    $restoreScript = @"
# Legal AI System - Restore Instructions
# Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
# Backup: $BackupPath

To restore from this backup:

1. Stop all services:
   docker-compose down

2. Extract backup:
   Expand-Archive -Path "$BackupPath" -DestinationPath ".\restore-temp"

3. Restore database:
   docker-compose up -d postgres
   # Wait for postgres to start
   docker exec -i legal_ai_postgres psql -U postgres -c "DROP DATABASE IF EXISTS prosecutor_db"
   docker exec -i legal_ai_postgres psql -U postgres -c "CREATE DATABASE prosecutor_db"
   Get-Content ".\restore-temp\prosecutor_db.sql" | docker exec -i legal_ai_postgres psql -U postgres -d prosecutor_db

4. Restore configuration files:
   Copy-Item ".\restore-temp\*.json" -Destination "." -Force
   Copy-Item ".\restore-temp\*.env" -Destination "." -Force

5. Restart all services:
   docker-compose up -d

6. Clean up:
   Remove-Item ".\restore-temp" -Recurse -Force
"@
    
    $restoreFile = Join-Path (Split-Path $BackupPath) "RESTORE_INSTRUCTIONS.txt"
    $restoreScript | Out-File $restoreFile
    Write-Step "Restore instructions saved to: $restoreFile" "INFO"
}

# Main execution
function Main {
    Write-Host @"
╔═══════════════════════════════════════════════════════╗
║      Legal AI System - Safe Installation Script        ║
║                   Version 1.0.0                        ║
╚═══════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

    # Check for existing installation
    $existing = Test-ExistingInstallation
    
    # Create backup if needed
    $backupPath = $null
    if (($existing.Values | Where-Object { $_ -eq $true }).Count -gt 0) {
        $backupPath = Backup-ExistingSystem -ExistingItems $existing
        Write-RestoreInstructions -BackupPath $backupPath
    }
    
    # Confirm installation
    if (-not (Confirm-Installation -ExistingItems $existing -BackupPath $backupPath)) {
        return
    }
    
    # Check prerequisites
    if (-not (Test-Prerequisites)) {
        Write-Step "Prerequisites check failed. Please install missing components." "ERROR"
        return
    }
    
    try {
        # Install dependencies
        Install-RootDependencies
        
        # Start Docker services
        Start-DockerServices
        
        # Install AI models
        Install-AIModels
        
        # Initialize database
        Initialize-Database
        
        # Run health checks
        $health = Test-SystemHealth
        
        # Final instructions
        Write-Host "`n=== Installation Complete ===" -ForegroundColor Green
        
        if ($backupPath) {
            Write-Host "`n📦 Backup created at:" -ForegroundColor Yellow
            Write-Host "   $backupPath" -ForegroundColor White
            Write-Host "   See RESTORE_INSTRUCTIONS.txt for recovery steps" -ForegroundColor Gray
        }
        
        Write-Host "`nTo start the development server, run:" -ForegroundColor Yellow
        Write-Host "  cd sveltekit-frontend; npm run dev" -ForegroundColor White
        Write-Host "`nAccess the application at:" -ForegroundColor Yellow
        Write-Host "  http://localhost:5173" -ForegroundColor White
        Write-Host "`nView system health at any time with:" -ForegroundColor Yellow
        Write-Host "  .\health-check.ps1" -ForegroundColor White
        
        if (Test-Path "TODO.md") {
            Write-Host "`nTODO items were created. Review TODO.md for any issues." -ForegroundColor Yellow
        }
        
    } catch {
        Write-Step "Installation failed: $_" "ERROR"
        Write-Step "Check logs: docker-compose logs" "INFO"
        
        if ($backupPath) {
            Write-Step "Restore from backup if needed: $backupPath" "WARNING"
        }
        
        Add-TodoItem "Installation Error" "$_"
        throw
    }
}

# Run main
Main
