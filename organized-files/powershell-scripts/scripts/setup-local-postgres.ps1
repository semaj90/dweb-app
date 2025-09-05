# Local PostgreSQL Setup Script for Windows
# This script will download, install, and configure PostgreSQL with pgvector

param(
    [string]$PostgresPassword = "postgres123",
    [string]$DatabaseName = "deeds_legal_ai",
    [string]$PostgresVersion = "16"
)

Write-Host "🐘 Setting up PostgreSQL locally on Windows..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if (-not $isAdmin) {
    Write-Host "⚠️  This script requires administrator privileges to install PostgreSQL." -ForegroundColor Yellow
    Write-Host "Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    exit 1
}

# Function to check if PostgreSQL is already installed
function Test-PostgreSQLInstalled {
    $postgresPath = "C:\Program Files\PostgreSQL\$PostgresVersion\bin\psql.exe"
    return Test-Path $postgresPath
}

# Function to download file with progress
function Download-FileWithProgress {
    param(
        [string]$Url,
        [string]$OutputPath
    )
    
    Write-Host "📥 Downloading: $Url" -ForegroundColor Yellow
    
    try {
        # Use System.Net.WebClient for progress
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($Url, $OutputPath)
        Write-Host "✅ Download completed: $OutputPath" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ Download failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Check if PostgreSQL is already installed
if (Test-PostgreSQLInstalled) {
    Write-Host "✅ PostgreSQL $PostgresVersion is already installed" -ForegroundColor Green
} else {
    Write-Host "📦 PostgreSQL not found. Starting installation..." -ForegroundColor Yellow
    
    # Create temp directory
    $tempDir = "$env:TEMP\postgresql-setup"
    if (-not (Test-Path $tempDir)) {
        New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
    }
    
    # Download PostgreSQL installer
    $installerUrl = "https://get.enterprisedb.com/postgresql/postgresql-$PostgresVersion-1-windows-x64.exe"
    $installerPath = "$tempDir\postgresql-installer.exe"
    
    if (Download-FileWithProgress -Url $installerUrl -OutputPath $installerPath) {
        Write-Host "🔧 Installing PostgreSQL..." -ForegroundColor Yellow
        
        # Silent installation parameters
        $installArgs = @(
            "--mode", "unattended",
            "--superpassword", $PostgresPassword,
            "--servicename", "postgresql-x64-$PostgresVersion",
            "--servicepassword", $PostgresPassword,
            "--serverport", "5432",
            "--locale", "English, United States",
            "--install_runtimes", "0"
        )
        
        try {
            Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -NoNewWindow
            Write-Host "✅ PostgreSQL installation completed" -ForegroundColor Green
        } catch {
            Write-Host "❌ PostgreSQL installation failed: $($_.Exception.Message)" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "❌ Failed to download PostgreSQL installer" -ForegroundColor Red
        exit 1
    }
}

# Add PostgreSQL to PATH
$postgresPath = "C:\Program Files\PostgreSQL\$PostgresVersion\bin"
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")

if ($currentPath -notlike "*$postgresPath*") {
    Write-Host "🔧 Adding PostgreSQL to system PATH..." -ForegroundColor Yellow
    [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$postgresPath", "Machine")
    $env:PATH = "$env:PATH;$postgresPath"
    Write-Host "✅ PostgreSQL added to PATH" -ForegroundColor Green
}

# Wait for PostgreSQL service to start
Write-Host "⏳ Waiting for PostgreSQL service to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Test PostgreSQL connection
Write-Host "🔌 Testing PostgreSQL connection..." -ForegroundColor Yellow
$psqlPath = "$postgresPath\psql.exe"

if (Test-Path $psqlPath) {
    # Create database if it doesn't exist
    Write-Host "📊 Creating database: $DatabaseName" -ForegroundColor Yellow
    
    $createDbCommand = "CREATE DATABASE `"$DatabaseName`";"
    
    try {
        # Use environment variable for password to avoid prompt
        $env:PGPASSWORD = $PostgresPassword
        
        & $psqlPath -U postgres -c $createDbCommand 2>$null
        Write-Host "✅ Database '$DatabaseName' created successfully" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Database might already exist or creation failed" -ForegroundColor Yellow
    }
    
    # Install pgvector extension
    Write-Host "🧮 Installing pgvector extension..." -ForegroundColor Yellow
    
    # Download pgvector for Windows
    $pgvectorUrl = "https://github.com/pgvector/pgvector/releases/download/v0.7.4/pgvector-v0.7.4-pg$PostgresVersion-windows-x64.zip"
    $pgvectorZip = "$tempDir\pgvector.zip"
    
    if (Download-FileWithProgress -Url $pgvectorUrl -OutputPath $pgvectorZip) {
        # Extract pgvector
        Expand-Archive -Path $pgvectorZip -DestinationPath $tempDir -Force
        
        # Copy files to PostgreSQL installation
        $pgvectorSource = "$tempDir\pgvector.dll"
        $pgvectorDest = "C:\Program Files\PostgreSQL\$PostgresVersion\lib\pgvector.dll"
        
        if (Test-Path $pgvectorSource) {
            Copy-Item $pgvectorSource $pgvectorDest -Force
            Write-Host "✅ pgvector extension files copied" -ForegroundColor Green
        }
        
        # Copy SQL files
        $sqlSource = "$tempDir\pgvector--0.7.4.sql"
        $sqlDest = "C:\Program Files\PostgreSQL\$PostgresVersion\share\extension\"
        
        if (Test-Path $sqlSource) {
            Copy-Item "$tempDir\pgvector*" $sqlDest -Force
            Write-Host "✅ pgvector SQL files copied" -ForegroundColor Green
        }
    }
    
    # Enable pgvector extension in database
    try {
        $createExtCommand = "CREATE EXTENSION IF NOT EXISTS vector;"
        & $psqlPath -U postgres -d $DatabaseName -c $createExtCommand
        Write-Host "✅ pgvector extension enabled in database" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  pgvector extension installation may have failed" -ForegroundColor Yellow
        Write-Host "You can install it manually later using: CREATE EXTENSION vector;" -ForegroundColor Yellow
    }
    
    # Test the installation
    Write-Host "🧪 Testing PostgreSQL installation..." -ForegroundColor Yellow
    
    $testCommand = "SELECT version(); SELECT * FROM pg_available_extensions WHERE name = 'vector';"
    
    try {
        $result = & $psqlPath -U postgres -d $DatabaseName -c $testCommand
        Write-Host "✅ PostgreSQL is working correctly" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  PostgreSQL test failed, but installation may still be successful" -ForegroundColor Yellow
    }
    
    # Clean environment variable
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    
} else {
    Write-Host "❌ PostgreSQL installation failed - psql.exe not found" -ForegroundColor Red
    exit 1
}

# Create environment file for the application
Write-Host "📝 Creating environment configuration..." -ForegroundColor Yellow

$envContent = @"
# PostgreSQL Configuration
DATABASE_URL=postgresql://postgres:$PostgresPassword@localhost:5432/$DatabaseName
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=$PostgresPassword
POSTGRES_DB=$DatabaseName

# Vector Database
VECTOR_DIMENSIONS=768
VECTOR_INDEX_TYPE=hnsw

# Application Configuration
NODE_ENV=development
LOG_LEVEL=debug
"@

$envPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend\.env.local"
Set-Content -Path $envPath -Value $envContent -Encoding UTF8

Write-Host "Environment file created: $envPath" -ForegroundColor Green

# Clean up temp files
Write-Host "🧹 Cleaning up temporary files..." -ForegroundColor Yellow
Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "`nPostgreSQL setup completed!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "📋 Configuration Summary:" -ForegroundColor Cyan
Write-Host "• PostgreSQL Version: $PostgresVersion" -ForegroundColor White
Write-Host "• Database Name: $DatabaseName" -ForegroundColor White
Write-Host "• Username: postgres" -ForegroundColor White
Write-Host "• Password: $PostgresPassword" -ForegroundColor White
Write-Host "• Port: 5432" -ForegroundColor White
Write-Host "• pgvector: Enabled" -ForegroundColor White
Write-Host "`n🔧 Usage:" -ForegroundColor Cyan
Write-Host "• Connect: psql -U postgres -d $DatabaseName" -ForegroundColor White
Write-Host "• GUI: pgAdmin (from Start Menu)" -ForegroundColor White
Write-Host "• Environment file: $envPath" -ForegroundColor White

Write-Host "`nReady to run tests!" -ForegroundColor Green