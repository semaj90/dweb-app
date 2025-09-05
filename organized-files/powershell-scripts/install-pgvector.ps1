# Install pgvector for PostgreSQL 17 on Windows
Write-Host "Installing pgvector for PostgreSQL 17..." -ForegroundColor Green

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "This script requires Administrator privileges. Please run PowerShell as Administrator." -ForegroundColor Red
    exit 1
}

$pgVersion = "17"
$pgPath = "C:\Program Files\PostgreSQL\$pgVersion"

# Method 1: Try downloading pre-built binaries
Write-Host "`nAttempting to download pre-built pgvector binaries..." -ForegroundColor Yellow

$tempDir = "$env:TEMP\pgvector-install"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

try {
    # Download pgvector release for Windows
    $pgvectorVersion = "0.8.0"  # Latest stable version
    $downloadUrl = "https://github.com/pgvector/pgvector/releases/download/v$pgvectorVersion/pgvector-$pgvectorVersion-windows-x64.zip"
    $zipPath = "$tempDir\pgvector.zip"
    
    Write-Host "Downloading pgvector v$pgvectorVersion..."
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -ErrorAction Stop
    
    Write-Host "Extracting files..."
    Expand-Archive -Path $zipPath -DestinationPath $tempDir -Force
    
    # Copy files to PostgreSQL directories
    Write-Host "Installing pgvector extension files..."
    
    $shareExtPath = "$pgPath\share\extension"
    $libPath = "$pgPath\lib"
    
    # Copy control and SQL files
    Copy-Item "$tempDir\*.control" -Destination $shareExtPath -Force
    Copy-Item "$tempDir\*.sql" -Destination $shareExtPath -Force
    
    # Copy DLL files
    Copy-Item "$tempDir\*.dll" -Destination $libPath -Force
    
    Write-Host "pgvector installation completed successfully!" -ForegroundColor Green
    
} catch {
    Write-Host "Pre-built binary download failed. Error: $_" -ForegroundColor Red
    Write-Host "`nAlternative installation methods:" -ForegroundColor Yellow
    Write-Host "1. Use Stack Builder (comes with PostgreSQL installer)" -ForegroundColor Cyan
    Write-Host "2. Build from source (requires Visual Studio)" -ForegroundColor Cyan
    Write-Host "3. Use Docker with PostgreSQL + pgvector image" -ForegroundColor Cyan
    
    Write-Host "`nFor Stack Builder:" -ForegroundColor Yellow
    Write-Host "- Run Stack Builder from Start Menu"
    Write-Host "- Select your PostgreSQL 17 installation"
    Write-Host "- Look for pgvector in the extension list"
    
    Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 1
}

# Cleanup
Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "`nNext steps:" -ForegroundColor Green
Write-Host "1. Restart PostgreSQL service (optional but recommended)"
Write-Host "2. Connect to your database and run: CREATE EXTENSION vector;"
Write-Host "3. Verify installation with: SELECT * FROM pg_extension WHERE extname = 'vector';"