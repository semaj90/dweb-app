#!/usr/bin/env pwsh

Write-Host "🗄️ Fixing Database Migration - 'cases' table already exists" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$webAppPath = "C:\Users\james\Desktop\web-app\sveltekit-frontend"
Set-Location $webAppPath

Write-Host "`n📋 The Problem:" -ForegroundColor Yellow
Write-Host "• Database migration failing because 'cases' table already exists" -ForegroundColor White
Write-Host "• Drizzle trying to CREATE TABLE but table exists from previous runs" -ForegroundColor White
Write-Host "• Need to reset database state or fix migration" -ForegroundColor White

Write-Host "`n🔧 Solution Options:" -ForegroundColor Cyan

Write-Host "`n1️⃣ OPTION 1: Reset Database (Recommended)" -ForegroundColor Green
Write-Host "   This will drop all tables and recreate them fresh" -ForegroundColor Gray

$choice = Read-Host "`nDo you want to reset the database? This will delete all data! (y/N)"

if ($choice -match '^[Yy]') {
    Write-Host "`n🗑️ Resetting database..." -ForegroundColor Yellow
    
    # Stop any running services
    Write-Host "Stopping services..." -ForegroundColor Gray
    try {
        docker-compose down
        Start-Sleep 2
    } catch {
        Write-Host "Services already stopped or not running" -ForegroundColor Gray
    }
    
    # Remove existing database volume
    Write-Host "Removing database volume..." -ForegroundColor Gray
    try {
        docker volume rm web-app_postgres_data -f
        docker volume rm prosecutor_postgres_data -f
        Write-Host "✅ Database volume removed" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Volume removal had issues (may not exist)" -ForegroundColor Yellow
    }
    
    # Start services fresh
    Write-Host "Starting fresh database..." -ForegroundColor Gray
    Set-Location ".."
    docker-compose up -d
    Set-Location "sveltekit-frontend"
    
    # Wait for database to be ready
    Write-Host "Waiting for database to initialize..." -ForegroundColor Gray
    Start-Sleep 10
    
    # Run migrations
    Write-Host "Running fresh migrations..." -ForegroundColor Yellow
    try {
        npm run db:migrate
        Write-Host "✅ Migrations completed successfully!" -ForegroundColor Green
        
        # Seed database
        Write-Host "Seeding database..." -ForegroundColor Yellow
        npm run db:seed
        Write-Host "✅ Database seeded successfully!" -ForegroundColor Green
        
    } catch {
        Write-Host "❌ Migration still failed. Trying alternative approach..." -ForegroundColor Red
        
        # Try using drizzle push instead
        Write-Host "Trying drizzle push..." -ForegroundColor Yellow
        try {
            npm run db:push
            Write-Host "✅ Schema pushed successfully!" -ForegroundColor Green
        } catch {
            Write-Host "❌ Push also failed" -ForegroundColor Red
        }
    }
    
} else {
    Write-Host "`n2️⃣ OPTION 2: Manual Fix (Advanced)" -ForegroundColor Yellow
    Write-Host "If you don't want to reset, try these manual steps:" -ForegroundColor Gray
    Write-Host ""
    Write-Host "A. Connect to database and check tables:" -ForegroundColor Cyan
    Write-Host "   docker exec -it postgres_container psql -U postgres -d prosecutor_db" -ForegroundColor White
    Write-Host "   \dt    (list tables)" -ForegroundColor White
    Write-Host "   \q     (quit)" -ForegroundColor White
    Write-Host ""
    Write-Host "B. Drop specific table if needed:" -ForegroundColor Cyan
    Write-Host "   DROP TABLE IF EXISTS cases CASCADE;" -ForegroundColor White
    Write-Host ""
    Write-Host "C. Use drizzle push instead of migrate:" -ForegroundColor Cyan
    Write-Host "   npm run db:push" -ForegroundColor White
    Write-Host ""
    Write-Host "D. Generate new migration:" -ForegroundColor Cyan
    Write-Host "   npm run db:generate" -ForegroundColor White
}

Write-Host "`n🔍 Verification Steps:" -ForegroundColor Cyan
Write-Host "After fixing, verify with:" -ForegroundColor Gray
Write-Host "• npm run db:studio    (open database admin)" -ForegroundColor White
Write-Host "• npm run dev          (start app)" -ForegroundColor White
Write-Host "• Check browser at http://localhost:5173" -ForegroundColor White

Write-Host "`n💡 Prevention Tips:" -ForegroundColor Cyan
Write-Host "• Use 'npm run db:push' for development" -ForegroundColor White
Write-Host "• Use 'npm run db:migrate' for production" -ForegroundColor White
Write-Host "• Keep migrations in sync with schema" -ForegroundColor White

Write-Host "`n🚀 Alternative Quick Fix:" -ForegroundColor Green
Write-Host "If you just want to get running quickly:" -ForegroundColor Gray
Write-Host "1. npm run db:push    (force schema sync)" -ForegroundColor White
Write-Host "2. npm run db:seed    (add sample data)" -ForegroundColor White
Write-Host "3. npm run dev        (start app)" -ForegroundColor White

Write-Host "`nPress any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
