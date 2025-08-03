# Install pgvector using vcpkg (recommended for Windows)
Write-Host "Installing pgvector for PostgreSQL 17 using vcpkg..." -ForegroundColor Green

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "This script requires Administrator privileges. Please run PowerShell as Administrator." -ForegroundColor Red
    exit 1
}

$pgVersion = "17"
$pgPath = "C:\Program Files\PostgreSQL\$pgVersion"
$pgBin = "$pgPath\bin"

# Step 1: Install vcpkg if not present
$vcpkgPath = "C:\vcpkg"
if (-not (Test-Path $vcpkgPath)) {
    Write-Host "Installing vcpkg..." -ForegroundColor Yellow
    git clone https://github.com/Microsoft/vcpkg.git $vcpkgPath
    & "$vcpkgPath\bootstrap-vcpkg.bat"
} else {
    Write-Host "vcpkg already installed at $vcpkgPath" -ForegroundColor Green
}

# Step 2: Install build tools if needed
Write-Host "`nChecking for Visual Studio Build Tools..." -ForegroundColor Yellow
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath
    if ($vsPath) {
        Write-Host "Visual Studio found at: $vsPath" -ForegroundColor Green
    }
} else {
    Write-Host "Visual Studio not found. You may need to install Visual Studio Build Tools." -ForegroundColor Yellow
    Write-Host "Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Cyan
}

# Step 3: Build pgvector from source
Write-Host "`nBuilding pgvector from source..." -ForegroundColor Yellow

$tempDir = "$env:TEMP\pgvector-build"
Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

Push-Location $tempDir

try {
    # Clone pgvector repository
    Write-Host "Cloning pgvector repository..."
    git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
    
    Set-Location pgvector
    
    # Set PostgreSQL paths
    $env:PATH = "$pgBin;$env:PATH"
    $env:PGROOT = $pgPath
    
    # Build using nmake (requires Visual Studio)
    Write-Host "Building pgvector..."
    
    # Create a batch file to run in VS Developer Command Prompt
    @"
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=$pgBin;%PATH%
set PGROOT=$pgPath
nmake /F Makefile.win
nmake /F Makefile.win install
"@ | Out-File -FilePath "build.bat" -Encoding ASCII
    
    # Run the build
    & cmd /c "build.bat"
    
    Write-Host "pgvector built and installed successfully!" -ForegroundColor Green
    
} catch {
    Write-Host "Build failed. Error: $_" -ForegroundColor Red
    
    Write-Host "`nAlternative: Use Docker for development" -ForegroundColor Yellow
    Write-Host "Create a docker-compose.yml with:" -ForegroundColor Cyan
    Write-Host @"
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_password
      POSTGRES_DB: legal_ai_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
"@ -ForegroundColor Gray
}

Pop-Location

# Cleanup
Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "`nNext steps:" -ForegroundColor Green
Write-Host "1. Connect to PostgreSQL and run: CREATE EXTENSION vector;"
Write-Host "2. Verify with: SELECT * FROM pg_extension WHERE extname = 'vector';"