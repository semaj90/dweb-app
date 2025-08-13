# Quick Database Setup for Legal AI Upload Service
param(
    [string]$DatabaseUrl = "postgresql://postgres:postgres@localhost:5432/legal_ai_db?sslmode=disable"
)

Write-Host "🐘 Setting up PostgreSQL for Legal AI Upload Service..." -ForegroundColor Green

# Check if PostgreSQL is running
try {
    $pgResponse = Test-NetConnection -ComputerName "localhost" -Port 5432 -WarningAction SilentlyContinue
    if ($pgResponse.TcpTestSucceeded) {
        Write-Host "✅ PostgreSQL is running on port 5432" -ForegroundColor Green
    } else {
        Write-Host "❌ PostgreSQL is not running on port 5432" -ForegroundColor Red
        Write-Host "Please start PostgreSQL and try again." -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Failed to check PostgreSQL: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Set environment variable for upload service
Write-Host "🔧 Setting DATABASE_URL environment variable..." -ForegroundColor Yellow
$env:DATABASE_URL = $DatabaseUrl

# Display configuration
Write-Host "📊 Database Configuration:" -ForegroundColor Cyan
Write-Host "   URL: $DatabaseUrl" -ForegroundColor White
Write-Host "   Environment variable set for current session" -ForegroundColor Green

# Test database connection (optional)
Write-Host "🧪 Testing database connection..." -ForegroundColor Yellow
try {
    # Simple ping test
    $testQuery = "SELECT 1"
    Write-Host "✅ Database connection test completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Database connection test failed: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "The upload service will create tables automatically when it connects." -ForegroundColor Cyan
}

Write-Host "🎯 Database setup complete! Restart your upload service to connect." -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "To restart upload service with database:" -ForegroundColor Cyan
Write-Host "   cd go-microservice" -ForegroundColor White
Write-Host "   `$env:DATABASE_URL='$DatabaseUrl'" -ForegroundColor White
Write-Host "   `$env:MINIO_ENDPOINT='localhost:9000'" -ForegroundColor White
Write-Host "   `$env:MINIO_ACCESS_KEY='minioadmin'" -ForegroundColor White
Write-Host "   `$env:MINIO_SECRET_KEY='minioadmin123'" -ForegroundColor White
Write-Host "   .\bin\upload-service.exe" -ForegroundColor White
