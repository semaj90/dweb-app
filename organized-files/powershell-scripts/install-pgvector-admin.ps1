# pgvector Installation Script - Run as Administrator
# Requires Administrator privileges to copy files to Program Files

Write-Host "=== pgvector Installation for PostgreSQL 17 ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "✅ Running as Administrator" -ForegroundColor Green

# Define paths
$sourcePath = "pgvector-install"
$pgPath = "C:\Program Files\PostgreSQL\17"

# Check if source files exist
if (-not (Test-Path $sourcePath)) {
    Write-Host "ERROR: Source directory '$sourcePath' not found!" -ForegroundColor Red
    Write-Host "Make sure you're running this from the correct directory." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "✅ Source files found" -ForegroundColor Green

try {
    # Copy DLL file
    Write-Host "📦 Copying vector.dll..." -ForegroundColor Yellow
    Copy-Item "$sourcePath\lib\vector.dll" "$pgPath\lib\" -Force
    Write-Host "✅ vector.dll copied" -ForegroundColor Green

    # Copy extension files
    Write-Host "📦 Copying extension files..." -ForegroundColor Yellow
    $extensionFiles = Get-ChildItem "$sourcePath\share\extension\*" -File
    foreach ($file in $extensionFiles) {
        Copy-Item $file.FullName "$pgPath\share\extension\" -Force
        Write-Host "  ✅ $($file.Name)" -ForegroundColor Gray
    }
    Write-Host "✅ All extension files copied" -ForegroundColor Green

    # Copy include files (optional)
    Write-Host "📦 Copying include files..." -ForegroundColor Yellow
    if (Test-Path "$sourcePath\include") {
        Copy-Item "$sourcePath\include\*" "$pgPath\include\" -Recurse -Force
        Write-Host "✅ Include files copied" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "🎉 pgvector installation completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step: Enable the extension in your database" -ForegroundColor Cyan
    Write-Host "Run this command:" -ForegroundColor White
    Write-Host 'node -e "const { Pool } = require(''pg''); const pool = new Pool({connectionString: ''postgresql://postgres:123456@localhost:5432/legal_ai_db''}); pool.query(''CREATE EXTENSION IF NOT EXISTS vector'').then(r => console.log(''✅ pgvector extension enabled!'')).catch(e => console.log(''❌ Error:'', e.message)).finally(() => pool.end());"' -ForegroundColor Yellow

} catch {
    Write-Host ""
    Write-Host "❌ Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")