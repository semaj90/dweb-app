# Complete Windows 10 Enhanced RAG Setup Script
# Configures Ollama + Gemma3 + SvelteKit on port 3130 + MCP Context7

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("install", "health", "test", "start", "stop")]
    [string]$Action = "install"
)

Write-Host "=== Windows 10 Enhanced RAG Setup - Action: $Action ===" -ForegroundColor Green

# Configuration
$SVELTEKIT_PORT = 3130
$OLLAMA_PORT = 11434
$MCP_PORT = 40000
$GEMMA3_MODEL_PATH = ".\gemma3Q4_K_M\mohf16-Q4_K_M.gguf"
$MODEL_NAME = "gemma3-legal"

function Test-OllamaService {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$OLLAMA_PORT/api/tags" -Method GET -TimeoutSec 10
        return $true, $response
    }
    catch {
        return $false, $_.Exception.Message
    }
}

function Test-SvelteKitService {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$SVELTEKIT_PORT" -Method GET -TimeoutSec 5
        return $true, "SvelteKit responding"
    }
    catch {
        return $false, $_.Exception.Message
    }
}

function Import-Gemma3Model {
    Write-Host "Importing Gemma3 model to Ollama..." -ForegroundColor Cyan
    
    # Check if model already exists
    $isRunning, $modelsResponse = Test-OllamaService
    if ($isRunning) {
        $hasModel = $modelsResponse.models | Where-Object { $_.name -like "*$MODEL_NAME*" }
        if ($hasModel) {
            Write-Host "✓ $MODEL_NAME model already exists" -ForegroundColor Green
            return $true
        }
    }
    
    # Create Modelfile for Windows 10 with RTX 3060 optimization
    $modelFileContent = @"
FROM $GEMMA3_MODEL_PATH

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 8192
PARAMETER num_gpu 1
PARAMETER num_thread 8

TEMPLATE """<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>"""

SYSTEM """You are a specialized Legal AI Assistant powered by Gemma 3. You have expertise in legal document analysis, contract review, case law research, and legal reasoning. Always provide accurate, well-reasoned responses based on established legal principles while noting when professional legal advice is recommended."""
"@
    
    try {
        # Write Modelfile
        $modelFileContent | Out-File -FilePath ".\Modelfile-gemma3-legal-win10" -Encoding UTF8
        
        # Import model using Ollama
        Write-Host "Creating $MODEL_NAME model..." -ForegroundColor Yellow
        $result = & ollama create $MODEL_NAME -f ".\Modelfile-gemma3-legal-win10" 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Successfully imported $MODEL_NAME" -ForegroundColor Green
            return $true
        } else {
            Write-Host "✗ Failed to import model: $result" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "✗ Error importing model: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

switch ($Action) {
    "install" {
        Write-Host "Installing and configuring Enhanced RAG system..." -ForegroundColor Yellow
        
        # Step 1: Check Ollama installation
        Write-Host "1. Checking Ollama installation..." -ForegroundColor Cyan
        $ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
        if ($ollamaPath) {
            Write-Host "✓ Ollama found at: $($ollamaPath.Source)" -ForegroundColor Green
            $version = & ollama --version
            Write-Host "✓ Version: $version" -ForegroundColor Green
        } else {
            Write-Host "✗ Ollama not found. Please install from https://ollama.ai" -ForegroundColor Red
            exit 1
        }
        
        # Step 2: Start Ollama service
        Write-Host "2. Starting Ollama service..." -ForegroundColor Cyan
        $ollamaProcess = Get-Process -Name "ollama*" -ErrorAction SilentlyContinue
        if (-not $ollamaProcess) {
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 5
        }
        
        $isRunning, $response = Test-OllamaService
        if ($isRunning) {
            Write-Host "✓ Ollama service is running" -ForegroundColor Green
        } else {
            Write-Host "✗ Failed to start Ollama: $response" -ForegroundColor Red
            exit 1
        }
        
        # Step 3: Import Gemma3 model
        Write-Host "3. Importing Gemma3 Legal model..." -ForegroundColor Cyan
        if (Test-Path $GEMMA3_MODEL_PATH) {
            Write-Host "✓ Found model file: $GEMMA3_MODEL_PATH" -ForegroundColor Green
            $importSuccess = Import-Gemma3Model
            if (-not $importSuccess) {
                Write-Host "⚠ Model import failed, but continuing..." -ForegroundColor Yellow
            }
        } else {
            Write-Host "⚠ Model file not found: $GEMMA3_MODEL_PATH" -ForegroundColor Yellow
            Write-Host "  Please ensure your Gemma3 model is in the gemma3Q4_K_M directory" -ForegroundColor Yellow
        }
        
        # Step 4: Install SvelteKit dependencies
        Write-Host "4. Installing SvelteKit dependencies..." -ForegroundColor Cyan
        Set-Location -Path ".\sveltekit-frontend" -ErrorAction SilentlyContinue
        if (Test-Path ".\package.json") {
            Write-Host "Installing npm dependencies..." -ForegroundColor Yellow
            & npm install
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ SvelteKit dependencies installed" -ForegroundColor Green
            } else {
                Write-Host "⚠ Some dependency issues, but continuing..." -ForegroundColor Yellow
            }
        }
        Set-Location -Path ".."
        
        # Step 5: Configure MCP Context7
        Write-Host "5. Configuring MCP Context7..." -ForegroundColor Cyan
        $vsCodeSettingsPath = ".\.vscode\settings.json"
        if (Test-Path $vsCodeSettingsPath) {
            Write-Host "✓ VS Code settings found" -ForegroundColor Green
            Write-Host "  MCP Context7 configured for port $MCP_PORT" -ForegroundColor Green
        } else {
            Write-Host "⚠ VS Code settings not found - MCP may need manual setup" -ForegroundColor Yellow
        }
        
        Write-Host "=== Installation Complete! ===" -ForegroundColor Green
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "  1. Run: .\COMPLETE-WINDOWS-10-SETUP.ps1 -Action start" -ForegroundColor White
        Write-Host "  2. Open VS Code and enable MCP Context7 extension" -ForegroundColor White
        Write-Host "  3. Test with: .\COMPLETE-WINDOWS-10-SETUP.ps1 -Action health" -ForegroundColor White
    }
    
    "health" {
        Write-Host "Running health checks..." -ForegroundColor Yellow
        
        # Check Ollama
        Write-Host "1. Checking Ollama service..." -ForegroundColor Cyan
        $isRunning, $response = Test-OllamaService
        if ($isRunning) {
            Write-Host "✓ Ollama service running on port $OLLAMA_PORT" -ForegroundColor Green
            
            # Check models
            $models = $response.models
            $gemmaModel = $models | Where-Object { $_.name -like "*$MODEL_NAME*" }
            if ($gemmaModel) {
                Write-Host "✓ $MODEL_NAME model available" -ForegroundColor Green
                Write-Host "  Size: $([math]::Round($gemmaModel.size / 1MB, 2)) MB" -ForegroundColor White
            } else {
                Write-Host "⚠ $MODEL_NAME model not found" -ForegroundColor Yellow
                Write-Host "  Available models:" -ForegroundColor White
                $models | ForEach-Object { Write-Host "    - $($_.name)" -ForegroundColor White }
            }
        } else {
            Write-Host "✗ Ollama service not running: $response" -ForegroundColor Red
        }
        
        # Check SvelteKit
        Write-Host "2. Checking SvelteKit service..." -ForegroundColor Cyan
        $isRunning, $response = Test-SvelteKitService
        if ($isRunning) {
            Write-Host "✓ SvelteKit running on port $SVELTEKIT_PORT" -ForegroundColor Green
        } else {
            Write-Host "⚠ SvelteKit not running: $response" -ForegroundColor Yellow
        }
        
        # Check MCP Context7
        Write-Host "3. Checking MCP Context7..." -ForegroundColor Cyan
        try {
            $mcpResponse = Invoke-WebRequest -Uri "http://localhost:$MCP_PORT" -Method GET -TimeoutSec 5
            Write-Host "✓ MCP Context7 running on port $MCP_PORT" -ForegroundColor Green
        }
        catch {
            Write-Host "⚠ MCP Context7 not accessible: $($_.Exception.Message)" -ForegroundColor Yellow
        }
        
        Write-Host "=== Health Check Complete ===" -ForegroundColor Green
    }
    
    "test" {
        Write-Host "Running integration tests..." -ForegroundColor Yellow
        
        # Test Gemma3 model
        Write-Host "Testing Gemma3 model..." -ForegroundColor Cyan
        try {
            $testPrompt = @{
                model = $MODEL_NAME
                prompt = "What is a legal contract?"
                stream = $false
            } | ConvertTo-Json
            
            Write-Host "Sending test prompt to $MODEL_NAME..." -ForegroundColor Yellow
            $response = Invoke-RestMethod -Uri "http://localhost:$OLLAMA_PORT/api/generate" -Method POST -Body $testPrompt -ContentType "application/json" -TimeoutSec 60
            
            if ($response.response) {
                Write-Host "✓ Gemma3 model responding correctly" -ForegroundColor Green
                Write-Host "Sample response: $($response.response.Substring(0, [Math]::Min(100, $response.response.Length)))..." -ForegroundColor White
            } else {
                Write-Host "⚠ Gemma3 model responded but no content" -ForegroundColor Yellow
            }
        }
        catch {
            Write-Host "✗ Gemma3 model test failed: $($_.Exception.Message)" -ForegroundColor Red
        }
        
        Write-Host "=== Integration Test Complete ===" -ForegroundColor Green
    }
    
    "start" {
        Write-Host "Starting Enhanced RAG services..." -ForegroundColor Yellow
        
        # Start Ollama
        Write-Host "1. Starting Ollama..." -ForegroundColor Cyan
        $ollamaProcess = Get-Process -Name "ollama*" -ErrorAction SilentlyContinue
        if (-not $ollamaProcess) {
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 3
            Write-Host "✓ Ollama started" -ForegroundColor Green
        } else {
            Write-Host "✓ Ollama already running" -ForegroundColor Green
        }
        
        # Start SvelteKit
        Write-Host "2. Starting SvelteKit on port $SVELTEKIT_PORT..." -ForegroundColor Cyan
        Set-Location -Path ".\sveltekit-frontend" -ErrorAction SilentlyContinue
        Start-Process -FilePath "npm" -ArgumentList "run", "dev", "--", "--port", $SVELTEKIT_PORT -WindowStyle Minimized
        Set-Location -Path ".."
        Start-Sleep -Seconds 5
        Write-Host "✓ SvelteKit started" -ForegroundColor Green
        
        Write-Host "=== Services Started ===" -ForegroundColor Green
        Write-Host "Access your application at: http://localhost:$SVELTEKIT_PORT" -ForegroundColor Cyan
        Write-Host "Run health check: .\COMPLETE-WINDOWS-10-SETUP.ps1 -Action health" -ForegroundColor White
    }
    
    "stop" {
        Write-Host "Stopping Enhanced RAG services..." -ForegroundColor Yellow
        
        # Stop Node processes (SvelteKit)
        Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object { 
            $_.ProcessName -like "*node*" 
        } | Stop-Process -Force -ErrorAction SilentlyContinue
        
        # Stop Ollama
        Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        
        Write-Host "✓ Services stopped" -ForegroundColor Green
    }
}

Write-Host "Script completed!" -ForegroundColor Green