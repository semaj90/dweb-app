# pgvector Installation Script for PostgreSQL 17
# Must be run as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "pgvector Installation for PostgreSQL 17" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "❌ ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Running as Administrator" -ForegroundColor Green
Write-Host ""

# Define source and destination paths
$sourcePath = "pgvector-install"
$pgPath = "C:\Program Files\PostgreSQL\17"

# Check if source files exist
if (-not (Test-Path $sourcePath)) {
    Write-Host "❌ ERROR: Source directory '$sourcePath' not found!" -ForegroundColor Red
    Write-Host "Make sure you're running this from the correct directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

try {
    Write-Host "Step 1: Copying vector.dll to PostgreSQL lib directory..." -ForegroundColor Yellow
    Copy-Item "$sourcePath\lib\vector.dll" "$pgPath\lib\" -Force
    Write-Host "✅ vector.dll copied successfully" -ForegroundColor Green

    Write-Host ""
    Write-Host "Step 2: Copying extension files to PostgreSQL share\extension directory..." -ForegroundColor Yellow
    $extensionFiles = Get-ChildItem "$sourcePath\share\extension\*" -File
    foreach ($file in $extensionFiles) {
        Copy-Item $file.FullName "$pgPath\share\extension\" -Force
        Write-Host "  ✅ Copied $($file.Name)" -ForegroundColor Gray
    }
    Write-Host "✅ All extension files copied successfully" -ForegroundColor Green

    Write-Host ""
    Write-Host "Step 3: Verifying installation files..." -ForegroundColor Yellow
    
    if (Test-Path "$pgPath\lib\vector.dll") {
        Write-Host "✅ vector.dll found in lib directory" -ForegroundColor Green
    } else {
        Write-Host "❌ vector.dll not found in lib directory" -ForegroundColor Red
    }
    
    if (Test-Path "$pgPath\share\extension\vector.control") {
        Write-Host "✅ vector.control found in extension directory" -ForegroundColor Green
    } else {
        Write-Host "❌ vector.control not found in extension directory" -ForegroundColor Red
    }
    
    if (Test-Path "$pgPath\share\extension\vector--0.8.0.sql") {
        Write-Host "✅ vector--0.8.0.sql found in extension directory" -ForegroundColor Green
    } else {
        Write-Host "❌ vector--0.8.0.sql not found in extension directory" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "🎉 Installation Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next step: Enable the extension in your database" -ForegroundColor White
    Write-Host "You can now run: CREATE EXTENSION IF NOT EXISTS vector;" -ForegroundColor Yellow
    Write-Host ""

} catch {
    Write-Host ""
    Write-Host "❌ Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Read-Host "Press Enter to continue"