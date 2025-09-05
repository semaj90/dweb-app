# Legal AI System - Backup and Merge Script
# Creates comprehensive backups before merging configurations

param(
    [switch]$DryRun = $false,
    [switch]$SkipBackup = $false
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupRoot = ".\backups\merge-backup-$timestamp"

# Files to merge
$mergeTargets = @{
    "install.ps1" = @{
        Source = "install-safe.ps1"
        Backup = $true
        MergeStrategy = "replace"  # Safe version should replace unsafe
    }
    "docker-compose.yml" = @{
        Source = $null  # Already updated
        Backup = $true
        MergeStrategy = "keep"
    }
    ".env" = @{
        Source = ".env.example"
        Backup = $true
        MergeStrategy = "merge"  # Preserve existing values
    }
}

function Create-ComprehensiveBackup {
    if ($SkipBackup) {
        Write-Host "⚠️  Skipping backup (-SkipBackup flag)" -ForegroundColor Yellow
        return
    }
    
    Write-Host "📦 Creating comprehensive backup..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
    
    # Backup all critical files
    $criticalFiles = @(
        "*.ps1",
        "*.bat",
        "*.yml",
        "*.yaml",
        "*.json",
        "*.md",
        ".env*",
        "package*.json",
        "docker-compose*",
        "sveltekit-frontend\package*.json",
        "sveltekit-frontend\.env*",
        "sveltekit-frontend\src\lib\server\services\*.ts",
        "database\*.*"
    )
    
    $fileCount = 0
    foreach ($pattern in $criticalFiles) {
        $files = Get-ChildItem -Path . -Filter $pattern -Recurse -ErrorAction SilentlyContinue |
                 Where-Object { $_.FullName -notmatch "node_modules|\.svelte-kit|dist|build|backups" }
        
        foreach ($file in $files) {
            $relativePath = $file.FullName.Substring((Get-Location).Path.Length + 1)
            $destPath = Join-Path $backupRoot $relativePath
            $destDir = Split-Path $destPath -Parent
            
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            
            Copy-Item $file.FullName $destPath -Force
            $fileCount++
        }
    }
    
    Write-Host "✓ Backed up $fileCount files to $backupRoot" -ForegroundColor Green
    
    # Create backup manifest
    $manifest = @{
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        FileCount = $fileCount
        SystemState = @{
            DockerRunning = (docker info 2>$null) -ne $null
            ServicesUp = (docker ps --format "{{.Names}}" | Where-Object { $_ -match "legal_ai" }).Count
        }
    }
    
    $manifest | ConvertTo-Json -Depth 5 | Out-File (Join-Path $backupRoot "backup-manifest.json")
    
    # Backup Docker volumes if running
    if ($manifest.SystemState.DockerRunning -and $manifest.SystemState.ServicesUp -gt 0) {
        Write-Host "📀 Backing up Docker volumes..." -ForegroundColor Cyan
        
        try {
            # PostgreSQL backup
            docker exec legal_ai_postgres pg_dump -U postgres prosecutor_db > (Join-Path $backupRoot "postgres-backup.sql") 2>$null
            
            # Qdrant snapshots
            $collections = @("legal_documents", "case_embeddings", "evidence_vectors")
            foreach ($col in $collections) {
                try {
                    Invoke-RestMethod -Uri "http://localhost:6333/collections/$col/snapshots" -Method POST -ErrorAction SilentlyContinue
                } catch {}
            }
            
            Write-Host "✓ Database backups completed" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Some volume backups failed (non-critical)" -ForegroundColor Yellow
        }
    }
    
    # Compress backup
    Write-Host "📦 Compressing backup..." -ForegroundColor Cyan
    $zipPath = "$backupRoot.zip"
    Compress-Archive -Path $backupRoot -DestinationPath $zipPath -Force
    Remove-Item $backupRoot -Recurse -Force
    
    Write-Host "✅ Backup completed: $zipPath" -ForegroundColor Green
    return $zipPath
}

function Merge-SafeConfigurations {
    Write-Host "`n🔀 Merging configurations..." -ForegroundColor Cyan
    
    foreach ($target in $mergeTargets.GetEnumerator()) {
        $targetFile = $target.Key
        $config = $target.Value
        
        if (-not $config.Source) {
            Write-Host "  → Keeping $targetFile (no source for merge)" -ForegroundColor Gray
            continue
        }
        
        if (-not (Test-Path $config.Source)) {
            Write-Host "  ⚠️  Source not found: $($config.Source)" -ForegroundColor Yellow
            continue
        }
        
        switch ($config.MergeStrategy) {
            "replace" {
                if (Test-Path $targetFile) {
                    # Create timestamped backup of original
                    $backupName = "$targetFile.unsafe-$timestamp.bak"
                    Copy-Item $targetFile $backupName -Force
                    Write-Host "  📄 Backed up unsafe version to $backupName" -ForegroundColor Gray
                }
                
                Copy-Item $config.Source $targetFile -Force
                Write-Host "  ✓ Replaced $targetFile with safe version" -ForegroundColor Green
            }
            
            "merge" {
                if ($targetFile -eq ".env" -and -not (Test-Path $targetFile)) {
                    Copy-Item $config.Source $targetFile -Force
                    Write-Host "  ✓ Created $targetFile from template" -ForegroundColor Green
                } elseif (Test-Path $targetFile) {
                    Write-Host "  → Preserving existing $targetFile" -ForegroundColor Gray
                }
            }
            
            "keep" {
                Write-Host "  → Keeping existing $targetFile" -ForegroundColor Gray
            }
        }
    }
}

function Validate-MergedSystem {
    Write-Host "`n🔍 Validating merged configuration..." -ForegroundColor Cyan
    
    $validationChecks = @{
        "Safe install script" = { Test-Path "install.ps1" -and (Get-Content "install.ps1" -Raw) -match "Test-ExistingInstallation" }
        "Docker compose valid" = { docker-compose config > $null 2>&1; $LASTEXITCODE -eq 0 }
        "Environment configured" = { Test-Path ".env" -or Test-Path ".env.example" }
        "PowerShell scripts" = { (Get-ChildItem *.ps1).Count -ge 5 }
        "Production scripts" = { Test-Path "SETUP-PRODUCTION.bat" -and Test-Path "VALIDATE-SYSTEM.ps1" }
    }
    
    $passed = 0
    $failed = 0
    
    foreach ($check in $validationChecks.GetEnumerator()) {
        try {
            if (& $check.Value) {
                Write-Host "  ✓ $($check.Key)" -ForegroundColor Green
                $passed++
            } else {
                Write-Host "  ✗ $($check.Key)" -ForegroundColor Red
                $failed++
            }
        } catch {
            Write-Host "  ✗ $($check.Key): $_" -ForegroundColor Red
            $failed++
        }
    }
    
    Write-Host "`nValidation: $passed passed, $failed failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
    return $failed -eq 0
}

# Main execution
Write-Host @"
╔═══════════════════════════════════════════════════════╗
║       Legal AI System - Backup & Merge Utility         ║
╚═══════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "`n🔸 DRY RUN MODE - No changes will be made" -ForegroundColor Magenta
}

try {
    # Step 1: Create backup
    $backupPath = Create-ComprehensiveBackup
    
    # Step 2: Merge configurations
    if (-not $DryRun) {
        Merge-SafeConfigurations
    } else {
        Write-Host "`n🔸 Would merge configurations (dry run)" -ForegroundColor Magenta
    }
    
    # Step 3: Validate
    $valid = Validate-MergedSystem
    
    if ($valid) {
        Write-Host "`n✅ System ready for production deployment" -ForegroundColor Green
        Write-Host "📌 Next step: Run .\SETUP-PRODUCTION.bat" -ForegroundColor Yellow
    } else {
        Write-Host "`n⚠️  Some validation checks failed" -ForegroundColor Yellow
        Write-Host "📌 Review errors and run .\health-check.ps1 -Fix" -ForegroundColor Yellow
    }
    
    if ($backupPath) {
        Write-Host "`n💾 Backup location: $backupPath" -ForegroundColor Cyan
    }
    
} catch {
    Write-Host "`n❌ Error during merge: $_" -ForegroundColor Red
    Write-Host "💾 Restore from backup if needed" -ForegroundColor Yellow
    exit 1
}
