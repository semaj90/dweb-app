#!/usr/bin/env powershell

# Real-time Evidence Management System Startup Script
# Starts Redis, PostgreSQL, WebSocket server, and SvelteKit dev server

param(
    [switch]$StartRedis,
    [switch]$StartDatabase,
    [switch]$StartWebSocket,
    [switch]$StartApp,
    [switch]$All,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Real-time Evidence Management System Startup Script

Usage:
    .\start-realtime.ps1 [options]

Options:
    -StartRedis      Start Redis server (required for real-time features)
    -StartDatabase   Start PostgreSQL database
    -StartWebSocket  Start WebSocket server for real-time updates
    -StartApp        Start SvelteKit development server
    -All             Start all services
    -Help            Show this help message

Examples:
    .\start-realtime.ps1 -All
    .\start-realtime.ps1 -StartRedis -StartWebSocket -StartApp
    .\start-realtime.ps1 -StartApp

Note: Redis and PostgreSQL must be running for full functionality
"@
    exit 0
}

# Set default to start all if no specific flags are provided
if (-not ($StartRedis -or $StartDatabase -or $StartWebSocket -or $StartApp)) {
    $All = $true
}

if ($All) {
    $StartRedis = $true
    $StartDatabase = $true
    $StartWebSocket = $true
    $StartApp = $true
}

Write-Host "🚀 Starting Real-time Evidence Management System..." -ForegroundColor Green

# Check if required tools are available
function Test-Command($command) {
    try {
        Get-Command $command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Start Redis server
if ($StartRedis) {
    Write-Host "📡 Starting Redis server..." -ForegroundColor Yellow
    
    if (Test-Command "redis-server") {
        Start-Process -FilePath "redis-server" -WindowStyle Minimized
        Write-Host "✅ Redis server started" -ForegroundColor Green
    } elseif (Test-Command "docker") {
        Write-Host "🐳 Starting Redis with Docker..." -ForegroundColor Blue
        docker run -d --name redis-realtime -p 6379:6379 redis:alpine
        Write-Host "✅ Redis container started" -ForegroundColor Green
    } else {
        Write-Host "❌ Redis not found. Install Redis or Docker first." -ForegroundColor Red
        Write-Host "Download Redis: https://redis.io/download" -ForegroundColor Yellow
    }
}

# Start PostgreSQL database
if ($StartDatabase) {
    Write-Host "🗄️ Starting PostgreSQL database..." -ForegroundColor Yellow
    
    if (Test-Command "docker-compose") {
        # Use existing docker-compose if available
        if (Test-Path "docker-compose.yml") {
            docker-compose up -d
            Write-Host "✅ Database services started with docker-compose" -ForegroundColor Green
        } else {
            Write-Host "⚠️ docker-compose.yml not found, trying to start individual containers..." -ForegroundColor Yellow
            docker run -d --name postgres-realtime `
                -e POSTGRES_USER=postgres `
                -e POSTGRES_PASSWORD=postgres `
                -e POSTGRES_DB=prosecutor_db `
                -p 5432:5432 `
                postgres:13
            Write-Host "✅ PostgreSQL container started" -ForegroundColor Green
        }
    } else {
        Write-Host "❌ Docker Compose not found. Please install Docker Desktop." -ForegroundColor Red
    }
}

# Wait a moment for services to start
if ($StartRedis -or $StartDatabase) {
    Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}

# Start WebSocket server
if ($StartWebSocket) {
    Write-Host "🔌 Starting WebSocket server..." -ForegroundColor Yellow
    
    if (Test-Command "node") {
        Start-Process -FilePath "node" -ArgumentList "websocket-server.js" -WindowStyle Normal
        Write-Host "✅ WebSocket server started on port 3030" -ForegroundColor Green
    } else {
        Write-Host "❌ Node.js not found. Install Node.js first." -ForegroundColor Red
        Write-Host "Download Node.js: https://nodejs.org/" -ForegroundColor Yellow
    }
}

# Start SvelteKit development server
if ($StartApp) {
    Write-Host "🌐 Starting SvelteKit development server..." -ForegroundColor Yellow
    
    if (Test-Command "npm") {
        # Install dependencies if node_modules doesn't exist
        if (-not (Test-Path "node_modules")) {
            Write-Host "📦 Installing dependencies..." -ForegroundColor Blue
            npm install
        }
        
        # Start the development server
        npm run dev
    } else {
        Write-Host "❌ npm not found. Install Node.js and npm first." -ForegroundColor Red
    }
}

Write-Host @"

🎉 Real-time Evidence Management System is starting!

Services:
- SvelteKit App: http://localhost:5173
- WebSocket Server: ws://localhost:3030
- Redis: localhost:6379
- PostgreSQL: localhost:5432

Real-time Demo: http://localhost:5173/evidence/realtime

To stop services:
- Press Ctrl+C in this terminal to stop SvelteKit
- Use 'docker-compose down' to stop database services
- Use 'taskkill' or close the WebSocket server window

"@ -ForegroundColor Cyan
