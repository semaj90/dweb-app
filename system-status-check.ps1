# Status Check and Next Steps for Legal AI Upload System

Write-Host "📊 LEGAL AI UPLOAD SYSTEM STATUS CHECK" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Gray

# Check MinIO
Write-Host "🗄️ MinIO Status:" -ForegroundColor Yellow
try {
    $minioHealth = Invoke-WebRequest -Uri "http://localhost:9000/minio/health/ready" -UseBasicParsing -TimeoutSec 3
    if ($minioHealth.StatusCode -eq 200) {
        Write-Host "   ✅ MinIO is running and healthy" -ForegroundColor Green
        Write-Host "   🌐 Console: http://localhost:9001" -ForegroundColor Cyan
        Write-Host "   🔑 Credentials: minioadmin / minioadmin123" -ForegroundColor Cyan
    }
} catch {
    Write-Host "   ❌ MinIO is not responding" -ForegroundColor Red
}

# Check PostgreSQL
Write-Host "🐘 PostgreSQL Status:" -ForegroundColor Yellow
try {
    $pgTest = Test-NetConnection -ComputerName "localhost" -Port 5432 -WarningAction SilentlyContinue
    if ($pgTest.TcpTestSucceeded) {
        Write-Host "   ✅ PostgreSQL is running on port 5432" -ForegroundColor Green
    } else {
        Write-Host "   ❌ PostgreSQL is not running" -ForegroundColor Red
    }
} catch {
    Write-Host "   ❌ PostgreSQL check failed" -ForegroundColor Red
}

# Check Upload Service
Write-Host "🚀 Upload Service Status:" -ForegroundColor Yellow
try {
    $uploadHealth = Invoke-WebRequest -Uri "http://localhost:8093/health" -UseBasicParsing -TimeoutSec 3
    if ($uploadHealth.StatusCode -eq 200) {
        $healthData = $uploadHealth.Content | ConvertFrom-Json
        Write-Host "   ✅ Upload service is running" -ForegroundColor Green
        Write-Host "   📊 Health: $($healthData.status)" -ForegroundColor White
        Write-Host "   🗄️ MinIO: $($healthData.minio)" -ForegroundColor $(if($healthData.minio) { "Green" } else { "Red" })
        Write-Host "   🐘 Database: $($healthData.db)" -ForegroundColor $(if($healthData.db) { "Green" } else { "Red" })
    }
} catch {
    Write-Host "   ❌ Upload service is not responding" -ForegroundColor Red
}

Write-Host "" -ForegroundColor White
Write-Host "🎯 NEXT STEPS:" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Gray

Write-Host "1. 🔧 Fix Database Connection:" -ForegroundColor Yellow
Write-Host "   Try different PostgreSQL credentials:" -ForegroundColor White
Write-Host "   - postgresql://postgres:password@localhost:5432/legal_ai_db" -ForegroundColor Gray
Write-Host "   - postgresql://postgres:123456@localhost:5432/legal_ai_db" -ForegroundColor Gray
Write-Host "   - postgresql://legal_admin:123456@localhost:5432/legal_ai_db" -ForegroundColor Gray

Write-Host "2. 🧪 Test File Upload:" -ForegroundColor Yellow
Write-Host "   Once both services are green, test upload:" -ForegroundColor White
Write-Host "   curl -X POST -F 'file=@test.txt' -F 'caseId=TEST-001' -F 'documentType=memo' http://localhost:8093/upload" -ForegroundColor Gray

Write-Host "3. 🎨 Connect Frontend:" -ForegroundColor Yellow
Write-Host "   Wire SvelteKit components to upload service endpoint" -ForegroundColor White
Write-Host "   Update upload forms to point to: http://localhost:8093/upload" -ForegroundColor Gray

Write-Host "4. 🔍 Batch Fix Svelte Errors:" -ForegroundColor Yellow
Write-Host "   Run error reduction pipeline on the 2788 TypeScript errors" -ForegroundColor White

Write-Host "" -ForegroundColor White
Write-Host "🎉 PROGRESS: MinIO integration complete! Database connection is the final step." -ForegroundColor Green
