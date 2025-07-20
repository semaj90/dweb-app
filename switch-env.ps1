# Environment Switcher Script
# Easily switch between development environments

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("development", "testing", "production")]
    [string]$Environment
)

$envFile = ".env.$Environment"
$currentEnv = ".env"

if (Test-Path $envFile) {
    if (Test-Path $currentEnv) {
        Remove-Item $currentEnv
    }
    Copy-Item $envFile $currentEnv
    Write-Host "✅ Switched to $Environment environment" -ForegroundColor Green
    
    # Show current database
    $dbUrl = Select-String -Path $currentEnv -Pattern "DATABASE_URL=" | ForEach-Object { $_.Line.Split('=')[1] }
    if ($dbUrl -like "sqlite:*") {
        Write-Host "📊 Using SQLite database" -ForegroundColor Cyan
    } else {
        Write-Host "📊 Using PostgreSQL database" -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "Current configuration:" -ForegroundColor Yellow
    Get-Content $currentEnv | Where-Object { $_ -notmatch "^#" -and $_ -ne "" }
} else {
    Write-Host "❌ Environment file $envFile not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "Available commands:" -ForegroundColor Cyan
Write-Host "  npm run dev          # Start development server" -ForegroundColor White
Write-Host "  npx drizzle-kit push # Push schema changes" -ForegroundColor White
Write-Host "  docker-compose up -d # Start Docker (if using PostgreSQL)" -ForegroundColor White
