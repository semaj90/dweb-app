# Start MinIO Server on Windows
# Production-ready MinIO setup for Legal AI platform

Write-Host "🚀 Starting MinIO Server for Legal AI Platform" -ForegroundColor Green

# MinIO Configuration
$MINIO_ROOT_USER = "minioadmin"
$MINIO_ROOT_PASSWORD = "minioadmin123"
$MINIO_DATA_DIR = "C:\Users\james\Desktop\deeds-web\deeds-web-app\data\minio"

# Create data directory if it doesn't exist
if (!(Test-Path $MINIO_DATA_DIR)) {
    Write-Host "📁 Creating MinIO data directory: $MINIO_DATA_DIR" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $MINIO_DATA_DIR -Force
}

# Set environment variables
$env:MINIO_ROOT_USER = $MINIO_ROOT_USER
$env:MINIO_ROOT_PASSWORD = $MINIO_ROOT_PASSWORD

# Check if MinIO is already installed
$minioPath = Get-Command minio.exe -ErrorAction SilentlyContinue

if (!$minioPath) {
    Write-Host "❌ MinIO not found. Please install MinIO first:" -ForegroundColor Red
    Write-Host "   1. Download from: https://dl.min.io/server/minio/release/windows-amd64/minio.exe" -ForegroundColor Yellow
    Write-Host "   2. Place in C:\Windows\System32 or add to PATH" -ForegroundColor Yellow
    Write-Host "   3. Run this script again" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ MinIO found at: $($minioPath.Source)" -ForegroundColor Green

# Check if MinIO is already running
$existingProcess = Get-Process -Name "minio" -ErrorAction SilentlyContinue

if ($existingProcess) {
    Write-Host "⚠️ MinIO is already running (PID: $($existingProcess.Id))" -ForegroundColor Yellow
    Write-Host "   Access MinIO Console at: http://localhost:9001" -ForegroundColor Cyan
    Write-Host "   API Endpoint: http://localhost:9000" -ForegroundColor Cyan
    Write-Host "   Credentials: $MINIO_ROOT_USER / $MINIO_ROOT_PASSWORD" -ForegroundColor Cyan
    exit 0
}

# Start MinIO Server
Write-Host "🔧 Starting MinIO server..." -ForegroundColor Blue
Write-Host "   Data Directory: $MINIO_DATA_DIR" -ForegroundColor Gray
Write-Host "   API Port: 9000" -ForegroundColor Gray
Write-Host "   Console Port: 9001" -ForegroundColor Gray

try {
    # Start MinIO in background
    $minioArgs = @(
        "server",
        $MINIO_DATA_DIR,
        "--console-address", ":9001",
        "--address", ":9000"
    )
    
    Write-Host "🚀 Executing: minio $($minioArgs -join ' ')" -ForegroundColor Gray
    
    # Start MinIO process
    $process = Start-Process -FilePath "minio" -ArgumentList $minioArgs -PassThru -WindowStyle Hidden
    
    if ($process) {
        Write-Host "✅ MinIO started successfully (PID: $($process.Id))" -ForegroundColor Green
        Write-Host "" -ForegroundColor White
        Write-Host "🌐 MinIO Access Information:" -ForegroundColor Cyan
        Write-Host "   • Console URL: http://localhost:9001" -ForegroundColor White
        Write-Host "   • API Endpoint: http://localhost:9000" -ForegroundColor White
        Write-Host "   • Username: $MINIO_ROOT_USER" -ForegroundColor White
        Write-Host "   • Password: $MINIO_ROOT_PASSWORD" -ForegroundColor White
        Write-Host "" -ForegroundColor White
        Write-Host "📁 Legal AI Buckets will be auto-created:" -ForegroundColor Yellow
        Write-Host "   • legal-documents" -ForegroundColor Gray
        Write-Host "   • evidence-files" -ForegroundColor Gray
        Write-Host "   • image-assets" -ForegroundColor Gray
        Write-Host "   • thumbnails" -ForegroundColor Gray
        Write-Host "" -ForegroundColor White
        
        # Wait a moment for MinIO to start
        Start-Sleep -Seconds 3
        
        # Test connection
        Write-Host "🧪 Testing MinIO connection..." -ForegroundColor Blue
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:9000/minio/health/live" -Method GET -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Host "✅ MinIO health check passed!" -ForegroundColor Green
            } else {
                Write-Host "⚠️ MinIO responded but health check unclear" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "⚠️ Health check failed, but MinIO may still be starting..." -ForegroundColor Yellow
        }
        
        Write-Host "" -ForegroundColor White
        Write-Host "🎉 MinIO is ready for Legal AI file storage!" -ForegroundColor Green
        
    } else {
        Write-Host "❌ Failed to start MinIO" -ForegroundColor Red
        exit 1
    }
    
} catch {
    Write-Host "❌ Error starting MinIO: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}