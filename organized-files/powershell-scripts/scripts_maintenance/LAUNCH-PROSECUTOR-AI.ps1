# PROSECUTOR AI - Enhanced Startup Script with Evidence System Info
param(
    [switch]$SkipClean = $false,
    [switch]$Force = $false
)

# Set console colors for better visibility
$Host.UI.RawUI.BackgroundColor = "Black"
$Host.UI.RawUI.ForegroundColor = "Green"
Clear-Host

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "🎯 PROSECUTOR AI - LEGAL CASE MANAGEMENT SYSTEM" -ForegroundColor Yellow
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🏛️  Welcome to Prosecutor AI - Your AI-Powered Legal Assistant" -ForegroundColor White
Write-Host "📁 Advanced Evidence Management & Case Analysis Platform" -ForegroundColor Gray
Write-Host ""

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "📋 EVIDENCE SYSTEM FEATURES" -ForegroundColor Yellow
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The evidence system supports:" -ForegroundColor White
Write-Host "1. 📂 " -NoNewline -ForegroundColor Yellow
Write-Host "Drag files" -NoNewline -ForegroundColor White
Write-Host " from your computer onto the evidence board" -ForegroundColor Gray
Write-Host "2. ➕ " -NoNewline -ForegroundColor Yellow
Write-Host "Click `"ADD EVIDENCE`"" -NoNewline -ForegroundColor White
Write-Host " to open the upload dialog" -ForegroundColor Gray
Write-Host "3. 📄 " -NoNewline -ForegroundColor Yellow
Write-Host "Multiple file types:" -NoNewline -ForegroundColor White
Write-Host " PDF, images (JPG, PNG, GIF), videos (MP4, MOV, AVI), documents" -ForegroundColor Gray
Write-Host "4. 🏷️  " -NoNewline -ForegroundColor Yellow
Write-Host "File metadata:" -NoNewline -ForegroundColor White
Write-Host " Automatic file size, type, and thumbnail generation" -ForegroundColor Gray
Write-Host "5. 📊 " -NoNewline -ForegroundColor Yellow
Write-Host "Evidence organization:" -NoNewline -ForegroundColor White
Write-Host " Categorize and prioritize uploaded evidence" -ForegroundColor Gray
Write-Host ""

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "🚀 STARTING PROSECUTOR AI" -ForegroundColor Yellow
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
    Write-Host "❌ Node.js not found!" -ForegroundColor Red
    Write-Host "📥 Please install Node.js from: https://nodejs.org/" -ForegroundColor Yellow
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
    Write-Host "   📥 Installing fresh dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ npm install failed!" -ForegroundColor Red
        Write-Host "🔧 Try running: npm cache clean --force" -ForegroundColor Yellow
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
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "🚀 LAUNCHING PROSECUTOR AI" -ForegroundColor Yellow
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📱 Server will be available at: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:5173" -ForegroundColor Cyan
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
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host "🎉 PROSECUTOR AI LAUNCHED SUCCESSFULLY!" -ForegroundColor Yellow
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Your application is running at: " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:5173" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📋 QUICK START GUIDE:" -ForegroundColor Yellow
    Write-Host "• Navigate to the Evidence section" -ForegroundColor White
    Write-Host "• Drag files onto the evidence board OR click `"ADD EVIDENCE`"" -ForegroundColor White
    Write-Host "• Supported formats: PDF, JPG, PNG, GIF, MP4, MOV, AVI, DOC, DOCX" -ForegroundColor Gray
    Write-Host "• Files will automatically generate metadata and thumbnails" -ForegroundColor Gray
    Write-Host "• Use the organization tools to categorize your evidence" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🎯 FEATURES AVAILABLE:" -ForegroundColor Yellow
    Write-Host "• Case Management" -ForegroundColor White
    Write-Host "• Evidence Upload & Organization" -ForegroundColor White
    Write-Host "• AI-Powered Legal Analysis" -ForegroundColor White
    Write-Host "• Document Processing" -ForegroundColor White
    Write-Host "• Interactive Evidence Canvas" -ForegroundColor White
    Write-Host "• Real-time Collaboration" -ForegroundColor White
    Write-Host ""
}

Read-Host "Press Enter to exit"