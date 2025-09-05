Write-Host "🚀 Starting Legal AI Assistant with PostgreSQL..." -ForegroundColor Green

Write-Host "`n1️⃣ Setting environment to PostgreSQL..." -ForegroundColor Yellow
Copy-Item ".env.testing" ".env" -Force

Write-Host "`n2️⃣ Starting Docker services..." -ForegroundColor Yellow
docker-compose up -d postgres redis qdrant

Write-Host "`n3️⃣ Waiting for services to start (10 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "`n4️⃣ Running database migrations..." -ForegroundColor Yellow
try {
    npx drizzle-kit push
    Write-Host "✅ Database migrations completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Migration warning: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`n5️⃣ Initializing database..." -ForegroundColor Yellow
try {
    node init-postgres.js
    Write-Host "✅ Database initialization completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Initialization warning: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`n🎉 Setup complete! Starting development server..." -ForegroundColor Green
Write-Host "`n👤 Demo login credentials:" -ForegroundColor Cyan
Write-Host "   Email: admin@prosecutor.com" -ForegroundColor White
Write-Host "   Password: password" -ForegroundColor White
Write-Host ""

npm run dev
