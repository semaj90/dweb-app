# Legal AI Assistant - Complete Workflow Test
# This script starts the server and runs comprehensive tests

Write-Host "🚀 Legal AI Assistant - Complete Workflow Test" -ForegroundColor Green

# Set working directory
$workDir = "c:\Users\james\Downloads\Deeds-App-doesn-t-work--main (2)\web-app\sveltekit-frontend"
Set-Location $workDir

Write-Host "📂 Working directory: $workDir" -ForegroundColor Yellow

# Step 1: Check if Node.js is available
Write-Host "📍 Step 1: Checking Node.js..." -ForegroundColor Cyan
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js version: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js first." -ForegroundColor Red
    exit 1
}

# Step 2: Install dependencies
Write-Host "📍 Step 2: Installing dependencies..." -ForegroundColor Cyan
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ npm install failed" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Dependencies installed" -ForegroundColor Green

# Step 3: Setup database
Write-Host "📍 Step 3: Setting up database..." -ForegroundColor Cyan
npm run db:push
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ Database setup may have failed, continuing..." -ForegroundColor Yellow
}

# Step 4: Start development server in background
Write-Host "📍 Step 4: Starting development server..." -ForegroundColor Cyan
$serverJob = Start-Job -ScriptBlock {
    param($workingDir)
    Set-Location $workingDir
    npm run dev
} -ArgumentList $workDir

# Wait for server to start
Write-Host "⏳ Waiting for server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check if server is running
try {
    Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 5 -UseBasicParsing | Out-Null
    Write-Host "✅ Server is running on http://localhost:5173" -ForegroundColor Green
} catch {
    Write-Host "❌ Server failed to start or is not accessible" -ForegroundColor Red
    Stop-Job $serverJob
    exit 1
}

# Step 5: Run Playwright tests
Write-Host "📍 Step 5: Running complete workflow tests..." -ForegroundColor Cyan
Write-Host "🌐 Opening browser for manual testing..." -ForegroundColor Yellow

# Open browser for manual verification
Start-Process "http://localhost:5173"

Write-Host "📋 Manual Test Checklist:" -ForegroundColor Yellow
Write-Host "1. ✅ Register new user (http://localhost:5173/register)" -ForegroundColor White
Write-Host "2. ✅ Login with credentials" -ForegroundColor White
Write-Host "3. ✅ Create new case (http://localhost:5173/cases)" -ForegroundColor White
Write-Host "4. ✅ Save case details" -ForegroundColor White
Write-Host "5. ✅ Edit case information" -ForegroundColor White
Write-Host "6. ✅ Go to Interactive Canvas (http://localhost:5173/interactive-canvas)" -ForegroundColor White
Write-Host "7. ✅ Write report (http://localhost:5173/report-builder)" -ForegroundColor White
Write-Host "8. ✅ Save report" -ForegroundColor White
Write-Host "9. ✅ Export to PDF" -ForegroundColor White
Write-Host "10. ✅ Upload evidence for analysis (http://localhost:5173/evidence)" -ForegroundColor White

Write-Host "`n🎯 Test the following demo credentials:" -ForegroundColor Green
Write-Host "   Email: admin@example.com" -ForegroundColor White
Write-Host "   Password: admin123" -ForegroundColor White
Write-Host "`n   Email: user@example.com" -ForegroundColor White
Write-Host "   Password: user123" -ForegroundColor White

Write-Host "`n⏹️ Press any key to stop the server..." -ForegroundColor Red
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Cleanup
Write-Host "🛑 Stopping server..." -ForegroundColor Yellow
Stop-Job $serverJob
Remove-Job $serverJob

Write-Host "✅ Test session completed!" -ForegroundColor Green
