# Quick Setup and Test Script for Deeds App
# Run this to start all services and test the application

Write-Host "🚀 Starting Deeds Legal AI Assistant..." -ForegroundColor Green

# Start Docker services
Write-Host "📦 Starting Docker services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for database to be ready
Write-Host "⏳ Waiting for database to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Navigate to frontend and install dependencies
Set-Location "web-app\sveltekit-frontend"

Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
npm install

# Run database migrations
Write-Host "🗄️ Running database migrations..." -ForegroundColor Yellow
npm run db:migrate

# Start the development server
Write-Host "🌐 Starting development server..." -ForegroundColor Green
Write-Host "Application will be available at: http://localhost:5173" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow

npm run dev
