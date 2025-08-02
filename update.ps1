# Legal AI System - Update and Maintenance Script
# Handles package updates, Docker image updates, and system maintenance

param(
    [switch]$UpdatePackages = $false,
    [switch]$UpdateDocker = $false,
    [switch]$UpdateModels = $false,
    [switch]$UpdateAll = $false,
    [switch]$Backup = $false,
    [switch]$Optimize = $false,
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"
$script:UpdateLog = @()

# Configuration
$script:Config = @{
    BackupDir = Join-Path $PSScriptRoot "backups"
    MaxBackups = 5
    UpdateBranch = "main"
}

function Write-UpdateLog {
    param(
        [string]$Action,
        [string]$Status,
        [string]$Message,
        [object]$Details = $null
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch ($Status) {
        "SUCCESS" { "Green" }
        "INFO" { "Cyan" }
        "WARNING" { "Yellow" }
        "ERROR" { "Red" }
        "DRYRUN" { "Magenta" }
        default { "White" }
    }
    
    $prefix = if ($DryRun) { "[DRY RUN] " } else { "" }
    Write-Host "$prefix[$timestamp] [$Status] $Action - $Message" -ForegroundColor $color
    
    $script:UpdateLog += @{
        Timestamp = $timestamp
        Action = $Action
        Status = $Status
        Message = $Message
        Details = $Details
        DryRun = $DryRun
    }
}

function Backup-System {
    Write-UpdateLog "Backup" "INFO" "Starting system backup..."
    
    if (-not (Test-Path $Config.BackupDir)) {
        New-Item -ItemType Directory -Path $Config.BackupDir -Force | Out-Null
    }
    
    $backupName = "backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    $backupPath = Join-Path $Config.BackupDir $backupName
    
    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
        
        # Backup database
        Write-UpdateLog "Backup" "INFO" "Backing up PostgreSQL database..."
        $dbBackup = Join-Path $backupPath "prosecutor_db.sql"
        docker exec legal_ai_postgres pg_dump -U postgres prosecutor_db > $dbBackup
        
        # Backup Qdrant collections
        Write-UpdateLog "Backup" "INFO" "Backing up Qdrant collections..."
        $collections = @("legal_documents", "case_embeddings", "evidence_vectors")
        foreach ($collection in $collections) {
            try {
                $snapshot = Invoke-RestMethod -Uri "http://localhost:6333/collections/$collection/snapshots" -Method POST
                Write-UpdateLog "Backup" "SUCCESS" "Created snapshot for $collection"
            } catch {
                Write-UpdateLog "Backup" "WARNING" "Failed to snapshot $collection : $_"
            }
        }
        
        # Backup configuration files
        Write-UpdateLog "Backup" "INFO" "Backing up configuration files..."
        $configFiles = @(
            "docker-compose.yml",
            "package.json",
            "drizzle.config.ts",
            ".env",
            "sveltekit-frontend/.env",
            "sveltekit-frontend/package.json"
        )
        
        foreach ($file in $configFiles) {
            if (Test-Path $file) {
                $dest = Join-Path $backupPath (Split-Path $file -Leaf)
                Copy-Item $file $dest -Force
            }
        }
        
        # Compress backup
        $zipPath = "$backupPath.zip"
        Compress-Archive -Path $backupPath -DestinationPath $zipPath -Force
        Remove-Item $backupPath -Recurse -Force
        
        Write-UpdateLog "Backup" "SUCCESS" "Backup completed: $zipPath"
        
        # Clean old backups
        $backups = Get-ChildItem $Config.BackupDir -Filter "backup-*.zip" | Sort-Object CreationTime -Descending
        if ($backups.Count -gt $Config.MaxBackups) {
            $toDelete = $backups | Select-Object -Skip $Config.MaxBackups
            foreach ($old in $toDelete) {
                Remove-Item $old.FullName -Force
                Write-UpdateLog "Backup" "INFO" "Removed old backup: $($old.Name)"
            }
        }
    } else {
        Write-UpdateLog "Backup" "DRYRUN" "Would create backup at: $backupPath.zip"
    }
}

function Update-NpmPackages {
    Write-UpdateLog "NPM Update" "INFO" "Checking for npm package updates..."
    
    # Update root packages
    Push-Location $PSScriptRoot
    try {
        # Check outdated packages
        Write-UpdateLog "NPM Update" "INFO" "Checking root packages..."
        $outdated = npm outdated --json | ConvertFrom-Json
        
        if ($outdated) {
            Write-UpdateLog "NPM Update" "WARNING" "Outdated packages found" $outdated
            
            if (-not $DryRun) {
                # Update dependencies
                npm update
                
                # Update devDependencies
                npm update --save-dev
                
                # Audit and fix vulnerabilities
                npm audit fix --force
                
                Write-UpdateLog "NPM Update" "SUCCESS" "Root packages updated"
            } else {
                Write-UpdateLog "NPM Update" "DRYRUN" "Would update root packages"
            }
        } else {
            Write-UpdateLog "NPM Update" "INFO" "Root packages are up to date"
        }
        
        # Update frontend packages
        $frontendPath = Join-Path $PSScriptRoot "sveltekit-frontend"
        if (Test-Path $frontendPath) {
            Push-Location $frontendPath
            
            Write-UpdateLog "NPM Update" "INFO" "Checking frontend packages..."
            $frontendOutdated = npm outdated --json | ConvertFrom-Json
            
            if ($frontendOutdated) {
                Write-UpdateLog "NPM Update" "WARNING" "Frontend packages outdated" $frontendOutdated
                
                if (-not $DryRun) {
                    npm update
                    npm update --save-dev
                    npm audit fix --force
                    
                    # Rebuild if needed
                    npm run build
                    
                    Write-UpdateLog "NPM Update" "SUCCESS" "Frontend packages updated"
                } else {
                    Write-UpdateLog "NPM Update" "DRYRUN" "Would update frontend packages"
                }
            } else {
                Write-UpdateLog "NPM Update" "INFO" "Frontend packages are up to date"
            }
            
            Pop-Location
        }
    } catch {
        Write-UpdateLog "NPM Update" "ERROR" "Package update failed: $_"
    } finally {
        Pop-Location
    }
}

function Update-DockerImages {
    Write-UpdateLog "Docker Update" "INFO" "Checking for Docker image updates..."
    
    $images = @(
        "pgvector/pgvector:pg16",
        "redis/redis-stack:latest",
        "qdrant/qdrant:v1.9.0",
        "ollama/ollama:latest",
        "neo4j:5.16"
    )
    
    foreach ($image in $images) {
        Write-UpdateLog "Docker Update" "INFO" "Checking $image..."
        
        if (-not $DryRun) {
            # Pull latest image
            $output = docker pull $image 2>&1
            
            if ($output -match "Status: Downloaded newer image") {
                Write-UpdateLog "Docker Update" "SUCCESS" "Updated $image"
                
                # Restart affected container
                $containerName = switch -Wildcard ($image) {
                    "*pgvector*" { "legal_ai_postgres" }
                    "*redis*" { "legal_ai_redis" }
                    "*qdrant*" { "legal_ai_qdrant" }
                    "*ollama*" { "legal_ai_ollama" }
                    "*neo4j*" { "legal_ai_neo4j" }
                }
                
                if ($containerName) {
                    Write-UpdateLog "Docker Update" "INFO" "Restarting $containerName..."
                    docker-compose restart $containerName
                }
            } elseif ($output -match "Status: Image is up to date") {
                Write-UpdateLog "Docker Update" "INFO" "$image is up to date"
            } else {
                Write-UpdateLog "Docker Update" "WARNING" "Unexpected output for $image"
            }
        } else {
            Write-UpdateLog "Docker Update" "DRYRUN" "Would check and update $image"
        }
    }
    
    # Prune old images
    if (-not $DryRun) {
        Write-UpdateLog "Docker Update" "INFO" "Pruning unused images..."
        docker image prune -f
    }
}

function Update-AIModels {
    Write-UpdateLog "Model Update" "INFO" "Checking for AI model updates..."
    
    $models = @("nomic-embed-text", "gemma:2b", "gemma3-legal")
    
    foreach ($model in $models) {
        Write-UpdateLog "Model Update" "INFO" "Checking $model..."
        
        if (-not $DryRun) {
            try {
                # Check if model exists
                $exists = ollama list | Select-String $model
                
                if ($exists) {
                    # Pull latest version
                    $output = ollama pull $model 2>&1
                    
                    if ($output -match "success" -or $output -match "up to date") {
                        Write-UpdateLog "Model Update" "SUCCESS" "Updated $model"
                    } else {
                        Write-UpdateLog "Model Update" "WARNING" "Could not update $model : $output"
                    }
                } else {
                    Write-UpdateLog "Model Update" "INFO" "Model $model not installed, skipping"
                }
            } catch {
                Write-UpdateLog "Model Update" "ERROR" "Failed to update $model : $_"
            }
        } else {
            Write-UpdateLog "Model Update" "DRYRUN" "Would update model $model"
        }
    }
}

function Optimize-System {
    Write-UpdateLog "Optimization" "INFO" "Starting system optimization..."
    
    # Optimize PostgreSQL
    Write-UpdateLog "Optimization" "INFO" "Optimizing PostgreSQL..."
    if (-not $DryRun) {
        $pgOptimize = @"
VACUUM ANALYZE;
REINDEX DATABASE prosecutor_db;
UPDATE pg_database SET datallowconn = TRUE WHERE datname = 'prosecutor_db';
"@
        
        $pgOptimize | docker exec -i legal_ai_postgres psql -U postgres -d prosecutor_db
        Write-UpdateLog "Optimization" "SUCCESS" "PostgreSQL optimized"
    }
    
    # Optimize Qdrant
    Write-UpdateLog "Optimization" "INFO" "Optimizing Qdrant collections..."
    $collections = @("legal_documents", "case_embeddings", "evidence_vectors")
    
    foreach ($collection in $collections) {
        if (-not $DryRun) {
            try {
                # Trigger optimization
                $body = @{ wait = $true } | ConvertTo-Json
                Invoke-RestMethod -Uri "http://localhost:6333/collections/$collection/optimizer" -Method POST -Body $body -ContentType "application/json"
                Write-UpdateLog "Optimization" "SUCCESS" "Optimized $collection"
            } catch {
                Write-UpdateLog "Optimization" "WARNING" "Failed to optimize $collection : $_"
            }
        } else {
            Write-UpdateLog "Optimization" "DRYRUN" "Would optimize collection $collection"
        }
    }
    
    # Clear Redis cache
    Write-UpdateLog "Optimization" "INFO" "Optimizing Redis..."
    if (-not $DryRun) {
        docker exec legal_ai_redis redis-cli MEMORY DOCTOR
        docker exec legal_ai_redis redis-cli MEMORY PURGE
        Write-UpdateLog "Optimization" "SUCCESS" "Redis memory optimized"
    }
    
    # Clean Docker system
    Write-UpdateLog "Optimization" "INFO" "Cleaning Docker system..."
    if (-not $DryRun) {
        docker system prune -f --volumes
        Write-UpdateLog "Optimization" "SUCCESS" "Docker system cleaned"
    }
}

function Show-UpdateSummary {
    Write-Host "`n=== Update Summary ===" -ForegroundColor Cyan
    
    $summary = $script:UpdateLog | Group-Object Status
    foreach ($group in $summary) {
        $color = switch ($group.Name) {
            "SUCCESS" { "Green" }
            "INFO" { "Cyan" }
            "WARNING" { "Yellow" }
            "ERROR" { "Red" }
            "DRYRUN" { "Magenta" }
            default { "White" }
        }
        Write-Host "$($group.Name): $($group.Count)" -ForegroundColor $color
    }
    
    # Save log
    $logFile = Join-Path $PSScriptRoot "update-log-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    $script:UpdateLog | ConvertTo-Json -Depth 5 | Out-File $logFile
    Write-Host "`nUpdate log saved to: $logFile" -ForegroundColor Gray
    
    # Create TODO for errors
    $errors = $script:UpdateLog | Where-Object { $_.Status -eq "ERROR" }
    if ($errors) {
        $todoPath = Join-Path $PSScriptRoot "TODO.md"
        $todoEntry = "`n## Update Errors - $(Get-Date -Format 'yyyy-MM-dd')`n"
        foreach ($error in $errors) {
            $todoEntry += "- [ ] $($error.Action): $($error.Message)`n"
        }
        Add-Content -Path $todoPath -Value $todoEntry
        Write-Host "`nTODO items created for errors" -ForegroundColor Yellow
    }
}

# Main execution
function Main {
    Write-Host @"
╔═══════════════════════════════════════════════════════╗
║        Legal AI System - Update & Maintenance          ║
║                   Version 1.0.0                        ║
╚═══════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

    if ($DryRun) {
        Write-Host "`n*** DRY RUN MODE - No changes will be made ***`n" -ForegroundColor Magenta
    }
    
    # Backup first if requested
    if ($Backup -or $UpdateAll) {
        Backup-System
    }
    
    # Run updates
    if ($UpdatePackages -or $UpdateAll) {
        Update-NpmPackages
    }
    
    if ($UpdateDocker -or $UpdateAll) {
        Update-DockerImages
    }
    
    if ($UpdateModels -or $UpdateAll) {
        Update-AIModels
    }
    
    if ($Optimize -or $UpdateAll) {
        Optimize-System
    }
    
    # If no specific update requested, show options
    if (-not ($UpdatePackages -or $UpdateDocker -or $UpdateModels -or $UpdateAll -or $Backup -or $Optimize)) {
        Write-Host "`nNo update action specified. Available options:" -ForegroundColor Yellow
        Write-Host "  -UpdatePackages  : Update npm packages"
        Write-Host "  -UpdateDocker    : Update Docker images"
        Write-Host "  -UpdateModels    : Update AI models"
        Write-Host "  -UpdateAll       : Run all updates"
        Write-Host "  -Backup          : Backup system"
        Write-Host "  -Optimize        : Optimize system performance"
        Write-Host "  -DryRun          : Preview changes without applying"
        Write-Host "`nExample: .\update.ps1 -UpdateAll -Backup"
    } else {
        Show-UpdateSummary
    }
}

# Run main
Main
