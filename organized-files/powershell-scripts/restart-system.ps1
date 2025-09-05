Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " AI Legal App - System Restart & Health Check" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop all containers and clean volumes
Write-Host "[1/5] Stopping containers and cleaning corrupted data..." -ForegroundColor Yellow
docker-compose -f docker-compose-gpu.yml down 2>$null
docker container prune -f 2>$null
docker volume prune -f 2>$null
Write-Host "   [+] Cleanup complete" -ForegroundColor Green
Write-Host ""

# Step 2: Check for models directory
Write-Host "[2/5] Checking models directory..." -ForegroundColor Yellow
if (Test-Path "models") {
    $modelFiles = Get-ChildItem -Path "models" -Filter "*.gguf" -ErrorAction SilentlyContinue
    if ($modelFiles.Count -gt 0) {
        Write-Host "   [+] Found model files:" -ForegroundColor Green
        $modelFiles | ForEach-Object { Write-Host "     - $($_.Name)" -ForegroundColor Cyan }
    } else {
        Write-Host "   [!] No .gguf model files found in models directory" -ForegroundColor Yellow
    }
} else {
    Write-Host "   [!] Models directory not found" -ForegroundColor Yellow
}
Write-Host ""

# Step 3: Start services
Write-Host "[3/5] Starting Docker services..." -ForegroundColor Yellow
docker-compose -f docker-compose-gpu.yml up -d --build
Write-Host "   [+] Services starting..." -ForegroundColor Green
Write-Host ""

# Step 4: Wait for services to initialize
Write-Host "[4/5] Waiting for services to initialize..." -ForegroundColor Yellow
for ($i = 1; $i -le 30; $i++) {
    Write-Progress -Activity "Starting Services" -Status "Waiting for Ollama..." -PercentComplete ($i * 100 / 30)
    Start-Sleep -Seconds 2

    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction Stop
        Write-Host "   [+] Ollama API is responding!" -ForegroundColor Green
        break
    } catch {
        if ($i -eq 30) {
            Write-Host "   [!] Ollama API took longer than expected to start" -ForegroundColor Yellow
        }
    }
}
Write-Progress -Activity "Starting Services" -Completed
Write-Host ""

# Step 5: Test the system
Write-Host "[5/5] Testing system components..." -ForegroundColor Yellow

# Check containers
$containers = docker ps --format "{{.Names}}"
if ($containers -like "*ollama*") {
    Write-Host "   [+] Ollama container running" -ForegroundColor Green
} else {
    Write-Host "   [X] Ollama container not found" -ForegroundColor Red
}

# Check API
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 10
    Write-Host "   [+] Ollama API accessible" -ForegroundColor Green
    Write-Host "   [+] Available models:" -ForegroundColor Green
    $tags.models | ForEach-Object {
        $sizeGB = [math]::Round($_.size / 1GB, 2)
        Write-Host "     - $($_.name) ($sizeGB GB)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "   [X] Ollama API not responding: $($_.Exception.Message)" -ForegroundColor Red
}

# Check SvelteKit dev server
try {
    $svelteResponse = Invoke-WebRequest -Uri "http://localhost:5174" -Method Get -TimeoutSec 5 -ErrorAction Stop
    Write-Host "   [+] SvelteKit dev server accessible" -ForegroundColor Green
} catch {
    Write-Host "   [!] SvelteKit dev server not accessible (may need manual start)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " System Restart Complete!" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Access your app at: http://localhost:5174/ai-test" -ForegroundColor Cyan
Write-Host "2. If SvelteKit isn't running, navigate to sveltekit-frontend and run 'npm run dev'" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
