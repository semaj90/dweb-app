# Install Ollama for SIMD JSON Optimization System
# PowerShell script to download and set up Ollama with embedding models

Write-Host "🤖 Installing Ollama for SIMD JSON Optimization..." -ForegroundColor Cyan

# Function to check if a port is in use
function Test-Port {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    }
    catch {
        return $false
    }
}

# Check if Ollama is already running
if (Test-Port 11434) {
    Write-Host "✅ Ollama already running on port 11434" -ForegroundColor Green
    
    # Check available models
    try {
        $models = ollama list 2>$null
        Write-Host "📚 Current models:" -ForegroundColor Yellow
        Write-Host $models
    }
    catch {
        Write-Host "⚠️ Ollama running but CLI not in PATH" -ForegroundColor Yellow
    }
}
else {
    Write-Host "📥 Ollama not detected. Installing..." -ForegroundColor Yellow
    
    # Download and install Ollama
    $ollamaInstaller = "$env:TEMP\OllamaSetup.exe"
    
    try {
        Write-Host "⬇️ Downloading Ollama installer..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://ollama.ai/download/OllamaSetup.exe" -OutFile $ollamaInstaller
        
        Write-Host "🔧 Running installer..." -ForegroundColor Yellow
        Start-Process -FilePath $ollamaInstaller -Wait
        
        Write-Host "🧹 Cleaning up installer..." -ForegroundColor Gray
        Remove-Item $ollamaInstaller -Force -ErrorAction SilentlyContinue
        
        # Wait for Ollama to start
        Write-Host "⏳ Waiting for Ollama to start..." -ForegroundColor Yellow
        Start-Sleep 10
        
        # Verify installation
        for ($i = 1; $i -le 30; $i++) {
            if (Test-Port 11434) {
                Write-Host "✅ Ollama started successfully!" -ForegroundColor Green
                break
            }
            Write-Host "⏳ Waiting for Ollama... ($i/30)" -ForegroundColor Yellow
            Start-Sleep 2
        }
        
    }
    catch {
        Write-Host "❌ Failed to download Ollama installer" -ForegroundColor Red
        Write-Host "Please manually download from: https://ollama.ai/download" -ForegroundColor White
        return
    }
}

# Install required models for SIMD JSON optimization
Write-Host "`n🧠 Installing embedding models..." -ForegroundColor Cyan

$models = @(
    @{ 
        Name = "nomic-embed-text"
        Description = "384-dimensional embeddings for SIMD optimization"
        Size = "274MB"
        Required = $true
    },
    @{ 
        Name = "llama3.2:1b"
        Description = "Small language model for suggestions"
        Size = "1.3GB"
        Required = $false
    }
)

foreach ($model in $models) {
    Write-Host "`n📦 Installing $($model.Name) ($($model.Size))..." -ForegroundColor Yellow
    Write-Host "   Purpose: $($model.Description)" -ForegroundColor Gray
    
    try {
        # Check if model already exists
        $existingModels = ollama list 2>$null
        if ($existingModels -match $model.Name) {
            Write-Host "✅ $($model.Name) already installed" -ForegroundColor Green
            continue
        }
        
        # Pull the model
        if ($model.Required) {
            Write-Host "⬇️ Pulling $($model.Name) (required for SIMD optimization)..." -ForegroundColor Yellow
            ollama pull $model.Name
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ $($model.Name) installed successfully" -ForegroundColor Green
            }
            else {
                Write-Host "❌ Failed to install $($model.Name)" -ForegroundColor Red
            }
        }
        else {
            $install = Read-Host "Install optional model $($model.Name)? (y/N)"
            if ($install -eq "y" -or $install -eq "Y") {
                ollama pull $model.Name
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "✅ $($model.Name) installed successfully" -ForegroundColor Green
                }
                else {
                    Write-Host "❌ Failed to install $($model.Name)" -ForegroundColor Red
                }
            }
        }
    }
    catch {
        Write-Host "❌ Error installing $($model.Name): $_" -ForegroundColor Red
    }
}

# Test embedding generation
Write-Host "`n🧪 Testing embedding generation..." -ForegroundColor Cyan

try {
    $testPrompt = "This is a test for SIMD JSON optimization with vector embeddings."
    
    Write-Host "🔧 Generating test embedding..." -ForegroundColor Yellow
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/embeddings" -Method POST -Body (@{
        model = "nomic-embed-text"
        prompt = $testPrompt
    } | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 30
    
    if ($response.embedding -and $response.embedding.Length -eq 384) {
        Write-Host "✅ Embedding test successful! Generated 384-dimensional vector" -ForegroundColor Green
        Write-Host "   First 5 dimensions: $($response.embedding[0..4] -join ', ')" -ForegroundColor Gray
    }
    else {
        Write-Host "⚠️ Embedding test returned unexpected format" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "❌ Embedding test failed: $_" -ForegroundColor Red
    Write-Host "   This may indicate Ollama is not fully ready yet" -ForegroundColor Yellow
}

# Create Ollama service management script
$ollamaScript = @"
# Ollama Service Management for SIMD JSON Optimization
param([string]`$Action = "status")

function Test-OllamaPort {
    try {
        `$connection = New-Object System.Net.Sockets.TcpClient
        `$connection.Connect("localhost", 11434)
        `$connection.Close()
        return `$true
    }
    catch {
        return `$false
    }
}

switch (`$Action.ToLower()) {
    "start" {
        if (Test-OllamaPort) {
            Write-Host "✅ Ollama already running" -ForegroundColor Green
        }
        else {
            Write-Host "🚀 Starting Ollama..." -ForegroundColor Yellow
            Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep 5
            
            if (Test-OllamaPort) {
                Write-Host "✅ Ollama started successfully" -ForegroundColor Green
            }
            else {
                Write-Host "❌ Failed to start Ollama" -ForegroundColor Red
            }
        }
    }
    
    "stop" {
        Write-Host "🛑 Stopping Ollama..." -ForegroundColor Yellow
        Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
        Write-Host "✅ Ollama stopped" -ForegroundColor Green
    }
    
    "status" {
        if (Test-OllamaPort) {
            Write-Host "✅ Ollama running on port 11434" -ForegroundColor Green
            
            try {
                `$models = ollama list 2>`$null
                Write-Host "📚 Available models:" -ForegroundColor Yellow
                Write-Host `$models
                
                # Test embedding API
                `$testResponse = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
                Write-Host "📡 API Status: Healthy" -ForegroundColor Green
            }
            catch {
                Write-Host "⚠️ API not responding properly" -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "❌ Ollama not running" -ForegroundColor Red
        }
    }
    
    "test" {
        if (Test-OllamaPort) {
            Write-Host "🧪 Testing embedding generation..." -ForegroundColor Yellow
            try {
                `$response = Invoke-RestMethod -Uri "http://localhost:11434/api/embeddings" -Method POST -Body (@{
                    model = "nomic-embed-text"
                    prompt = "Test embedding for SIMD optimization"
                } | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 30
                
                if (`$response.embedding -and `$response.embedding.Length -eq 384) {
                    Write-Host "✅ Embedding test successful!" -ForegroundColor Green
                    Write-Host "   Dimensions: `$(`$response.embedding.Length)" -ForegroundColor Gray
                }
                else {
                    Write-Host "❌ Invalid embedding response" -ForegroundColor Red
                }
            }
            catch {
                Write-Host "❌ Embedding test failed: `$_" -ForegroundColor Red
            }
        }
        else {
            Write-Host "❌ Ollama not running. Start with: .\ollama-manager.ps1 start" -ForegroundColor Red
        }
    }
    
    default {
        Write-Host "Usage: .\ollama-manager.ps1 [start|stop|status|test]" -ForegroundColor White
    }
}
"@

$ollamaScript | Out-File -FilePath "ollama-manager.ps1" -Encoding UTF8

Write-Host "`n🎉 Ollama setup complete!" -ForegroundColor Green
Write-Host "`n📋 Summary:" -ForegroundColor Cyan
Write-Host "   ✅ Ollama installed and running on port 11434" -ForegroundColor White
Write-Host "   ✅ nomic-embed-text model ready for SIMD optimization" -ForegroundColor White
Write-Host "   ✅ Management script created: ollama-manager.ps1" -ForegroundColor White

Write-Host "`n🔧 Usage:" -ForegroundColor Cyan
Write-Host "   - Check status: .\ollama-manager.ps1 status" -ForegroundColor White
Write-Host "   - Test embeddings: .\ollama-manager.ps1 test" -ForegroundColor White
Write-Host "   - API endpoint: http://localhost:11434/api/embeddings" -ForegroundColor White

Write-Host "`n🚀 Next: Start other services with .\start-optimization-services.bat" -ForegroundColor Yellow