# PROSECUTOR AI - Comprehensive Fix and Startup Script
param(
    [switch]$SkipClean = $false,
    [switch]$Force = $false
)

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "🔧 PROSECUTOR AI - COMPREHENSIVE FIX & STARTUP" -ForegroundColor Yellow
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Set execution policy temporarily
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# Set the correct directory
$ProjectPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
Set-Location $ProjectPath

Write-Host "📍 Working in: $(Get-Location)" -ForegroundColor Green
Write-Host ""

# Function to check if command exists
function Test-Command($command) {
    $null = Get-Command $command -ErrorAction SilentlyContinue
    return $?
}

# Check prerequisites
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow

if (!(Test-Command "node")) {
    Write-Host "❌ Node.js not found! Please install Node.js from https://nodejs.org/" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

if (!(Test-Command "npm")) {
    Write-Host "❌ npm not found! Please install npm." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$nodeVersion = node --version
$npmVersion = npm --version
Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
Write-Host "✅ npm: $npmVersion" -ForegroundColor Green
Write-Host ""

# Clean previous builds if requested
if (!$SkipClean) {
    Write-Host "🧹 Cleaning previous builds..." -ForegroundColor Yellow
    
    $foldersToClean = @(".svelte-kit", "build", "dist", ".vite-temp")
    foreach ($folder in $foldersToClean) {
        if (Test-Path $folder) {
            Remove-Item $folder -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "   🗑️ Removed $folder" -ForegroundColor Gray
        }
    }
    
    Write-Host "✅ Cleanup complete" -ForegroundColor Green
    Write-Host ""
}

# Install or update dependencies
Write-Host "📦 Checking dependencies..." -ForegroundColor Yellow

if (!(Test-Path "node_modules") -or $Force) {
    Write-Host "   Installing fresh dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ npm install failed!" -ForegroundColor Red
        Write-Host "   Try running: npm cache clean --force" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✅ Dependencies already installed" -ForegroundColor Green
}

Write-Host ""

# Run SvelteKit sync
Write-Host "🔄 Running SvelteKit sync..." -ForegroundColor Yellow
npm run prepare
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ SvelteKit sync had issues, but continuing..." -ForegroundColor Yellow
}

Write-Host ""

# Check for common issues and fix them
Write-Host "🔍 Checking for common issues..." -ForegroundColor Yellow

# Check PostCSS config
if (Test-Path "postcss.config.js") {
    $postcssContent = Get-Content "postcss.config.js" -Raw
    if ($postcssContent -like "*tailwindcss*" -and $postcssContent -notlike "*@tailwindcss/postcss*") {
        Write-Host "   🔧 Fixing PostCSS configuration..." -ForegroundColor Yellow
        $newPostCSSConfig = @"
import tailwindcss from "@tailwindcss/postcss";
import autoprefixer from "autoprefixer";

export default {
  plugins: [tailwindcss(), autoprefixer()],
};
"@
        $newPostCSSConfig | Out-File "postcss.config.js" -Encoding UTF8
        Write-Host "   ✅ PostCSS configuration updated" -ForegroundColor Green
    }
}

Write-Host ""

# Start the development server
Write-Host "🚀 Starting Prosecutor AI Development Server..." -ForegroundColor Green
Write-Host "📱 Server will be available at: http://localhost:5173" -ForegroundColor Cyan
Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Try multiple startup approaches
$attempts = @(
    @{ Name = "Standard dev"; Command = "npm run dev" },
    @{ Name = "Clean dev"; Command = "npm run dev:clean" },
    @{ Name = "Safe dev"; Command = "npm run dev:safe" },
    @{ Name = "Local dev"; Command = "npm run dev:local" }
)

$success = $false
foreach ($attempt in $attempts) {
    Write-Host "🔥 Trying: $($attempt.Name)..." -ForegroundColor Yellow
    
    # Start the process and capture output
    try {
        Invoke-Expression $attempt.Command
        $success = $true
        break
    } catch {
        Write-Host "   ❌ $($attempt.Name) failed: $($_.Exception.Message)" -ForegroundColor Red
        continue
    }
}

if (!$success) {
    Write-Host ""
    Write-Host "❌ All startup attempts failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔧 Manual troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "   1. npm run fix:all" -ForegroundColor White
    Write-Host "   2. npm run clean" -ForegroundColor White
    Write-Host "   3. npm install" -ForegroundColor White
    Write-Host "   4. npm run dev" -ForegroundColor White
    Write-Host ""
    Write-Host "📝 Check the error logs above for specific issues." -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "🎉 Development server should be starting!" -ForegroundColor Green
}

Read-Host "Press Enter to exit"