# PROSECUTOR AI - PowerShell Development Server Startup
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "🎯 PROSECUTOR AI - DEVELOPMENT SERVER STARTUP" -ForegroundColor Yellow
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Set the correct directory
$ProjectPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
Set-Location $ProjectPath

Write-Host "📍 Current Directory: $(Get-Location)" -ForegroundColor Green
Write-Host ""

# Check Node.js installation
Write-Host "🔍 Checking Node.js installation..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found! Please install Node.js first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check npm installation
Write-Host "🔍 Checking npm installation..." -ForegroundColor Yellow
try {
    $npmVersion = npm --version
    Write-Host "✅ npm found: $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ npm not found! Please install npm first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "🚀 Starting Prosecutor AI Development Server..." -ForegroundColor Yellow
Write-Host "📱 Server will be available at: http://localhost:5173" -ForegroundColor Cyan
Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Check if node_modules exists
if (!(Test-Path "node_modules")) {
    Write-Host "📦 node_modules not found. Installing dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ npm install failed!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Try to start the development server
Write-Host "🔥 Attempting to start dev server..." -ForegroundColor Green
npm run dev

# If that fails, try cleaning and rebuilding
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "⚠️ Standard dev command failed. Trying cleanup and restart..." -ForegroundColor Yellow
    Write-Host ""
    
    Write-Host "🧹 Cleaning project cache..." -ForegroundColor Yellow
    npm run clean
    
    Write-Host "🔄 Installing fresh dependencies..." -ForegroundColor Yellow
    npm install
    
    Write-Host "🔄 Starting clean development server..." -ForegroundColor Yellow
    npm run dev:clean
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ Development server failed to start." -ForegroundColor Red
        Write-Host "🔧 Try running these commands manually:" -ForegroundColor Yellow
        Write-Host "   npm run fix:all" -ForegroundColor White
        Write-Host "   npm run dev" -ForegroundColor White
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "🎉 Development server started successfully!" -ForegroundColor Green
Read-Host "Press Enter to exit"