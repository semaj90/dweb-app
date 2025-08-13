# Quick Fix for MinIO Import Issues
Write-Host "🔧 Fixing Go module imports for MinIO integration..." -ForegroundColor Cyan

# Navigate to the go-microservice directory
Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice"

# Fix the import in upload-service/main.go
$uploadServicePath = "cmd\upload-service\main.go"
if (Test-Path $uploadServicePath) {
    Write-Host "Fixing imports in upload-service..." -ForegroundColor Yellow
    $content = Get-Content $uploadServicePath -Raw
    $content = $content -replace 'github\.com/deeds-web/deeds-web-app/go-microservice/pkg/minio', 'microservice/pkg/minio'
    Set-Content -Path $uploadServicePath -Value $content
    Write-Host "✅ Fixed upload-service imports" -ForegroundColor Green
}

# Fix any other files that might have the same issue
$files = Get-ChildItem -Path . -Filter "*.go" -Recurse | Select-String "github.com/deeds-web/deeds-web-app/go-microservice" -List
foreach ($file in $files) {
    Write-Host "Fixing imports in $($file.Path)..." -ForegroundColor Yellow
    $content = Get-Content $file.Path -Raw
    $content = $content -replace 'github\.com/deeds-web/deeds-web-app/go-microservice/', 'microservice/'
    Set-Content -Path $file.Path -Value $content
}

# Run go mod tidy to clean up dependencies
Write-Host "`nCleaning up Go dependencies..." -ForegroundColor Yellow
go mod tidy

# Build the services
Write-Host "`nBuilding services..." -ForegroundColor Yellow
if (!(Test-Path "bin")) {
    New-Item -ItemType Directory -Path "bin" | Out-Null
}

# Build upload service
Write-Host "Building upload-service..." -ForegroundColor Yellow
go build -o .\bin\upload-service.exe .\cmd\upload-service

# Build summarizer service
Write-Host "Building summarizer-service..." -ForegroundColor Yellow
go build -o .\bin\summarizer-service.exe .\cmd\summarizer-service

Write-Host "`n✅ Import fixes complete!" -ForegroundColor Green
Write-Host "You can now run the services with:" -ForegroundColor Cyan
Write-Host "  .\bin\upload-service.exe" -ForegroundColor White
Write-Host "  .\bin\summarizer-service.exe" -ForegroundColor White
