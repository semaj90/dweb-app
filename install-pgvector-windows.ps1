# Install pgvector for PostgreSQL 17 on Windows from GitHub
# This script requires Visual Studio Build Tools to be installed

Write-Host "Installing pgvector for PostgreSQL 17 from GitHub..." -ForegroundColor Green

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "This script requires Administrator privileges. Please run PowerShell as Administrator." -ForegroundColor Red
    exit 1
}

$pgVersion = "17"
$pgPath = "C:\Program Files\PostgreSQL\$pgVersion"
$pgBin = "$pgPath\bin"

# Verify PostgreSQL installation
if (-not (Test-Path $pgPath)) {
    Write-Host "PostgreSQL $pgVersion not found at $pgPath" -ForegroundColor Red
    Write-Host "Please install PostgreSQL 17 first." -ForegroundColor Yellow
    exit 1
}

Write-Host "Found PostgreSQL $pgVersion at $pgPath" -ForegroundColor Green

# Check for Visual Studio Build Tools
Write-Host "`nChecking for Visual Studio Build Tools..." -ForegroundColor Yellow
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$buildToolsPath = $null

if (Test-Path $vsWhere) {
    $installations = & $vsWhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($installations) {
        $buildToolsPath = $installations[0]
        Write-Host "Found Visual Studio Build Tools at: $buildToolsPath" -ForegroundColor Green
    }
}

if (-not $buildToolsPath) {
    Write-Host "Visual Studio Build Tools with C++ support not found." -ForegroundColor Red
    Write-Host "Please install Visual Studio Build Tools with C++ support first:" -ForegroundColor Yellow
    Write-Host "https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Cyan
    Write-Host "`nRequired components:" -ForegroundColor Yellow
    Write-Host "- MSVC v143 - VS 2022 C++ x64/x86 build tools" -ForegroundColor Gray
    Write-Host "- Windows 11 SDK (latest version)" -ForegroundColor Gray
    Write-Host "- CMake tools for Visual Studio" -ForegroundColor Gray
    exit 1
}

# Set up build environment
$vcvarsPath = "$buildToolsPath\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvarsPath)) {
    Write-Host "vcvars64.bat not found. Trying alternative paths..." -ForegroundColor Yellow
    $vcvarsPath = "$buildToolsPath\Common7\Tools\VsDevCmd.bat"
}

# Create temporary directory
$tempDir = "$env:TEMP\pgvector-build-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
Write-Host "`nUsing build directory: $tempDir" -ForegroundColor Gray

Push-Location $tempDir

try {
    # Download pgvector source
    Write-Host "`nDownloading pgvector v0.8.0 source..." -ForegroundColor Yellow
    git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git
    
    if (-not (Test-Path "pgvector")) {
        throw "Failed to clone pgvector repository"
    }
    
    Set-Location pgvector
    
    # Create build script
    Write-Host "`nCreating build script..." -ForegroundColor Yellow
    $buildScript = @"
@echo off
echo Setting up Visual Studio environment...
call "$vcvarsPath"
if errorlevel 1 (
    echo Failed to set up Visual Studio environment
    exit /b 1
)

echo Setting PostgreSQL environment...
set "PGROOT=$pgPath"
set "PATH=$pgBin;%PATH%"

echo Building pgvector...
nmake /F Makefile.win
if errorlevel 1 (
    echo Build failed
    exit /b 1
)

echo Installing pgvector...
nmake /F Makefile.win install
if errorlevel 1 (
    echo Installation failed
    exit /b 1
)

echo pgvector installation completed successfully!
"@

    $buildScript | Out-File -FilePath "build.bat" -Encoding ASCII
    
    # Run the build
    Write-Host "`nBuilding and installing pgvector..." -ForegroundColor Green
    Write-Host "This may take a few minutes..." -ForegroundColor Gray
    
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "build.bat" -Wait -PassThru -NoNewWindow
    
    if ($process.ExitCode -eq 0) {
        Write-Host "`npgvector installation completed successfully!" -ForegroundColor Green
        
        # Verify installation
        Write-Host "`nVerifying installation..." -ForegroundColor Yellow
        $controlFile = "$pgPath\share\extension\vector.control"
        $dllFile = "$pgPath\lib\vector.dll"
        
        if ((Test-Path $controlFile) -and (Test-Path $dllFile)) {
            Write-Host "✓ Extension files installed successfully" -ForegroundColor Green
            Write-Host "  Control file: $controlFile" -ForegroundColor Gray
            Write-Host "  Library file: $dllFile" -ForegroundColor Gray
        } else {
            Write-Host "⚠ Installation verification failed" -ForegroundColor Yellow
            Write-Host "Expected files:" -ForegroundColor Gray
            Write-Host "  $controlFile" -ForegroundColor Gray
            Write-Host "  $dllFile" -ForegroundColor Gray
        }
        
    } else {
        throw "Build process failed with exit code $($process.ExitCode)"
    }
    
} catch {
    Write-Host "`nInstallation failed: $_" -ForegroundColor Red
    Write-Host "`nTroubleshooting tips:" -ForegroundColor Yellow
    Write-Host "1. Ensure Visual Studio Build Tools with C++ support is installed" -ForegroundColor Gray
    Write-Host "2. Try running this script from 'x64 Native Tools Command Prompt for VS'" -ForegroundColor Gray
    Write-Host "3. Check that PostgreSQL 17 is properly installed" -ForegroundColor Gray
    Write-Host "4. Ensure Git is installed and available in PATH" -ForegroundColor Gray
    
    Pop-Location
    Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 1
}

Pop-Location

# Cleanup
Write-Host "`nCleaning up build directory..." -ForegroundColor Gray
Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "`n🎉 pgvector installation completed!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Restart PostgreSQL service (recommended):" -ForegroundColor Gray
Write-Host "   net stop postgresql-x64-17 && net start postgresql-x64-17" -ForegroundColor Gray
Write-Host "2. Connect to your database and create the extension:" -ForegroundColor Gray
Write-Host '   psql -U postgres -d legal_ai_db -c "CREATE EXTENSION vector;"' -ForegroundColor Gray
Write-Host "3. Verify installation:" -ForegroundColor Gray
Write-Host '   psql -U postgres -d legal_ai_db -c "SELECT * FROM pg_extension WHERE extname = ''vector'';"' -ForegroundColor Gray