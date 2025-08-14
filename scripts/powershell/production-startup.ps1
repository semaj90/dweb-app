#!/bin/bash
# Production System Startup - Fix All Blockers
# Execute: ./production-startup.ps1

Write-Host "🚀 Production System Startup" -ForegroundColor Green

# 1. Force GPU Ollama
Write-Host "Starting Ollama with GPU acceleration..."
$env:CUDA_VISIBLE_DEVICES="0"
Start-Process powershell -ArgumentList "ollama serve" -WindowStyle Minimized

# 2. Fix Modelfile and create model
Write-Host "Creating Gemma3 Legal model..."
Start-Sleep 5
ollama create gemma3-legal -f local-models/Modelfile.gemma3-legal

# 3. Start Docker services (fixes Qdrant connection)
Write-Host "Starting Docker stack..."
docker-compose down
docker-compose up -d postgres redis qdrant

# 4. Wait for services
Write-Host "Waiting for services..."
Start-Sleep 30

# 5. Install UI dependencies
Write-Host "Installing UI dependencies..."
Set-Location sveltekit-frontend
npm install @melt-ui/svelte bits-ui lucide-svelte --force

# 6. Start dev server
Write-Host "Starting SvelteKit dev server..."
Start-Process powershell -ArgumentList "npm run dev" -WindowStyle Normal

# 7. Test endpoints
Write-Host "Testing endpoints..."
Start-Sleep 10
Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET
Invoke-RestMethod -Uri "http://localhost:6333/collections" -Method GET
Invoke-RestMethod -Uri "http://localhost:5173/api/health" -Method GET

Write-Host "✅ Production startup complete" -ForegroundColor Green
Write-Host "Execute: .\manual-validation.ps1" -ForegroundColor Yellow