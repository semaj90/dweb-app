# PowerShell script to load .env and configure system
# This properly sets environment variables for VS Code and Windows

Write-Host "🔧 Legal AI System - Environment Configuration" -ForegroundColor Cyan
Write-Host ("=" * 50) -ForegroundColor Cyan
Write-Host ""

# Function to load .env file
function Load-EnvFile {
    param (
        [string]$Path = ".\.env"
    )
    
    if (Test-Path $Path) {
        Write-Host "📁 Loading .env file..." -ForegroundColor Yellow
        
        $envCount = 0
        Get-Content $Path | ForEach-Object {
            if ($_ -match '^([^#][^=]+)=(.*)$') {
                $name = $matches[1].Trim()
                $value = $matches[2].Trim()
                
                # Set for current session
                [System.Environment]::SetEnvironmentVariable($name, $value, [System.EnvironmentVariableTarget]::Process)
                
                # Also set as PowerShell variable
                Set-Variable -Name $name -Value $value -Scope Global -Force
                
                $envCount++
            }
        }
        
        Write-Host "✅ Loaded $envCount environment variables" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ .env file not found at $Path" -ForegroundColor Red
        return $false
    }
}

# Function to set persistent environment variables for VS Code
function Set-VSCodeEnvironment {
    Write-Host "`n🔧 Configuring VS Code environment..." -ForegroundColor Yellow
    
    # Critical variables for VS Code
    $criticalVars = @{
        "DATABASE_URL" = "postgresql://legal_admin:123456@localhost:5432/legal_ai_db"
        "PGPASSWORD" = "123456"
        "GPU_ENABLED" = "true"
        "CGO_ENABLED" = "1"
        "CC" = "C:\Progra~1\LLVM\bin\clang.exe"
        "CXX" = "C:\Progra~1\LLVM\bin\clang++.exe"
        "CUDA_PATH" = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
        "REDIS_URL" = "redis://localhost:6379"
        "API_URL" = "http://localhost:8080"
    }
    
    foreach ($key in $criticalVars.Keys) {
        [System.Environment]::SetEnvironmentVariable($key, $criticalVars[$key], [System.EnvironmentVariableTarget]::User)
    }
    
    Write-Host "✅ VS Code environment variables set" -ForegroundColor Green
    Write-Host "   Restart VS Code to apply changes" -ForegroundColor Yellow
}

# Function to test database connection
function Test-DatabaseConnection {
    Write-Host "`n🔍 Testing database connection..." -ForegroundColor Yellow
    
    $env:PGPASSWORD = "123456"
    $result = & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -h localhost -d legal_ai_db -t -c "SELECT 'Connected' as status" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Database connection successful" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Database connection failed" -ForegroundColor Red
        Write-Host "   Error: $result" -ForegroundColor Red
        return $false
    }
}

# Function to create missing directories
function Ensure-Directories {
    Write-Host "`n📁 Ensuring required directories exist..." -ForegroundColor Yellow
    
    $dirs = @(
        "logs",
        "uploads",
        "documents",
        "evidence",
        "generated_reports"
    )
    
    foreach ($dir in $dirs) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "   Created: $dir" -ForegroundColor Green
        } else {
            Write-Host "   Exists: $dir" -ForegroundColor Gray
        }
    }
}

# Function to start services
function Start-Services {
    Write-Host "`n🚀 Starting services..." -ForegroundColor Yellow
    
    # Check Redis
    $redisPing = & redis-cli ping 2>&1
    if ($redisPing -ne "PONG") {
        Write-Host "   Starting Redis..." -ForegroundColor Cyan
        Start-Process -FilePath "redis-windows\redis-server.exe" -ArgumentList "redis-windows\redis.conf" -WindowStyle Hidden
        Start-Sleep -Seconds 2
    }
    Write-Host "✅ Redis running" -ForegroundColor Green
    
    # Check GPU service
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8080/health" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
        Write-Host "✅ GPU service already running" -ForegroundColor Green
    } catch {
        Write-Host "   Starting GPU service..." -ForegroundColor Cyan
        
        # Build if needed
        Set-Location "go-microservice"
        if (!(Test-Path "legal-processor-enhanced.exe")) {
            Write-Host "   Building GPU processor..." -ForegroundColor Cyan
            & go build -tags=cgo -ldflags="-s -w" -o legal-processor-enhanced.exe enhanced_legal_processor.go
        }
        
        Start-Process -FilePath "legal-processor-enhanced.exe" -WindowStyle Hidden
        Set-Location ..
        Start-Sleep -Seconds 3
    }
}

# Main execution
try {
    # Load .env file
    if (Load-EnvFile) {
        
        # Set VS Code environment
        Set-VSCodeEnvironment
        
        # Ensure directories exist
        Ensure-Directories
        
        # Test database
        if (Test-DatabaseConnection) {
            
            # Start services
            Start-Services
            
            Write-Host "`n" -NoNewline
            Write-Host ("=" * 50) -ForegroundColor Green
            Write-Host "✅ System Configuration Complete!" -ForegroundColor Green
            Write-Host ("=" * 50) -ForegroundColor Green
            
            Write-Host "`n📊 Environment Status:" -ForegroundColor Cyan
            Write-Host "   Database: $env:DATABASE_URL"
            Write-Host "   GPU: $env:GPU_ENABLED"
            Write-Host "   Redis: $env:REDIS_URL"
            Write-Host "   API: $env:API_URL"
            
            Write-Host "`n🎯 Next Steps:" -ForegroundColor Yellow
            Write-Host "   1. Restart VS Code to load environment"
            Write-Host "   2. Run: npm run dev"
            Write-Host "   3. Open: http://localhost:5173"
            
            Write-Host "`n💡 VS Code Terminal Commands:" -ForegroundColor Cyan
            Write-Host "   node check-system-integration.mjs"
            Write-Host "   npm run dev"
            Write-Host "   npm run build"
            
        } else {
            Write-Host "`n⚠️ Database connection failed!" -ForegroundColor Red
            Write-Host "Run: .\FIX-POSTGRES-ADMIN.bat (as Administrator)" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}
