# PowerShell script to start local LLM services (Ollama and llama.cpp)
# For the SvelteKit legal AI assistant platform

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("both", "ollama", "llamacpp")]
    [string]$Service = "both",
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose,
    
    [Parameter(Mandatory=$false)]
    [switch]$LoadGemma,
    
    [Parameter(Mandatory=$false)]
    [int]$WaitTime = 10
)

# Configuration
$RootPath = "c:\Users\james\Downloads\Deeds-App-doesn-t-work--main (2)"
$OllamaPath = Join-Path $RootPath "Ollama"
$LlamaCppPath = Join-Path $RootPath "llama.cpp"
$GemmaModelPath = Join-Path $RootPath "gemma3Q4_K_M\mo16.gguf"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) { Write-Output $args }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Green "🚀 Starting Local LLM Services for Legal AI Assistant"
Write-ColorOutput Cyan "=================================================="

# Check if paths exist
$pathsValid = $true

if (!(Test-Path $OllamaPath)) {
    Write-ColorOutput Red "❌ Ollama directory not found: $OllamaPath"
    $pathsValid = $false
}

if (!(Test-Path $LlamaCppPath)) {
    Write-ColorOutput Red "❌ llama.cpp directory not found: $LlamaCppPath"
    $pathsValid = $false
}

if (!(Test-Path $GemmaModelPath)) {
    Write-ColorOutput Yellow "⚠️ Gemma model not found: $GemmaModelPath"
}

if (!$pathsValid) {
    Write-ColorOutput Red "❌ Required paths missing. Please check your installation."
    exit 1
}

# Function to check if a port is in use
function Test-Port($Port) {
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

# Function to start Ollama
function Start-Ollama {
    Write-ColorOutput Cyan "🔄 Starting Ollama service..."
    
    $ollamaExe = Join-Path $OllamaPath "ollama.exe"
    if (!(Test-Path $ollamaExe)) {
        Write-ColorOutput Red "❌ Ollama executable not found: $ollamaExe"
        return $false
    }
    
    # Check if Ollama is already running
    if (Test-Port 11434) {
        Write-ColorOutput Yellow "⚠️ Ollama appears to be already running on port 11434"
        return $true
    }
    
    try {
        # Start Ollama in background
        $process = Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden -PassThru
        
        Write-ColorOutput Green "✅ Ollama started (PID: $($process.Id))"
        
        # Wait for service to be ready
        Write-ColorOutput Cyan "⏳ Waiting for Ollama to be ready..."
        $maxAttempts = 30
        $attempt = 0
        
        do {
            Start-Sleep -Seconds 1
            $attempt++
            if ($Verbose) {
                Write-ColorOutput Gray "   Attempt $attempt/$maxAttempts..."
            }
        } while (-not (Test-Port 11434) -and $attempt -lt $maxAttempts)
        
        if (Test-Port 11434) {
            Write-ColorOutput Green "✅ Ollama is ready on http://localhost:11434"
            return $true
        } else {
            Write-ColorOutput Red "❌ Ollama failed to start within timeout"
            return $false
        }
    } catch {
        Write-ColorOutput Red "❌ Failed to start Ollama: $($_.Exception.Message)"
        return $false
    }
}

# Function to load Gemma model into Ollama
function Load-GemmaModel {
    if (!(Test-Path $GemmaModelPath)) {
        Write-ColorOutput Red "❌ Gemma model file not found: $GemmaModelPath"
        return $false
    }
    
    Write-ColorOutput Cyan "🔄 Loading Gemma3 model into Ollama..."
    
    # Create Modelfile
    $modelfilePath = Join-Path $PWD "Modelfile.gemma3"
    $modelfileContent = @"
FROM $GemmaModelPath

# Set parameters for legal AI assistant
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# Gemma3 template
TEMPLATE """<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>"""

# System message for legal assistant
SYSTEM """You are a specialized legal AI assistant with expertise in case law analysis, legal research, and document review. You provide accurate, well-reasoned responses that would be helpful to legal professionals. Always cite your sources when available and clearly state when information is insufficient for a complete answer."""
"@
    
    Set-Content -Path $modelfilePath -Value $modelfileContent
    
    try {
        $ollamaExe = Join-Path $OllamaPath "ollama.exe"
        
        # Import the model
        Write-ColorOutput Cyan "⏳ Importing model (this may take a few minutes)..."
        $importResult = & $ollamaExe create "gemma3-legal" -f $modelfilePath
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput Green "✅ Gemma3 model loaded successfully as 'gemma3-legal'"
            
            # Test the model
            Write-ColorOutput Cyan "🧪 Testing model..."
            $testResult = & $ollamaExe run "gemma3-legal" "Hello, can you help with legal questions?"
            
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput Green "✅ Model test successful"
                return $true
            } else {
                Write-ColorOutput Red "❌ Model test failed"
                return $false
            }
        } else {
            Write-ColorOutput Red "❌ Failed to import Gemma3 model"
            return $false
        }
    } catch {
        Write-ColorOutput Red "❌ Error loading Gemma model: $($_.Exception.Message)"
        return $false
    } finally {
        # Clean up Modelfile
        if (Test-Path $modelfilePath) {
            Remove-Item $modelfilePath -Force
        }
    }
}

# Function to start llama.cpp server
function Start-LlamaCpp {
    Write-ColorOutput Cyan "🔄 Starting llama.cpp server..."
    
    $llamaServerExe = Join-Path $LlamaCppPath "llama-server.exe"
    if (!(Test-Path $llamaServerExe)) {
        Write-ColorOutput Yellow "⚠️ llama-server.exe not found, trying build directory..."
        $llamaServerExe = Join-Path $LlamaCppPath "build\bin\Release\llama-server.exe"
        if (!(Test-Path $llamaServerExe)) {
            Write-ColorOutput Red "❌ llama-server.exe not found in: $LlamaCppPath"
            return $false
        }
    }
    
    # Check if already running
    if (Test-Port 8080) {
        Write-ColorOutput Yellow "⚠️ llama.cpp appears to be already running on port 8080"
        return $true
    }
    
    try {
        # Start llama.cpp server with Gemma model
        $args = @(
            "-m", $GemmaModelPath,
            "--host", "0.0.0.0",
            "--port", "8080",
            "-c", "4096",
            "-t", "4"
        )
        
        $process = Start-Process -FilePath $llamaServerExe -ArgumentList $args -WindowStyle Hidden -PassThru
        
        Write-ColorOutput Green "✅ llama.cpp started (PID: $($process.Id))"
        
        # Wait for service to be ready
        Write-ColorOutput Cyan "⏳ Waiting for llama.cpp to be ready..."
        $maxAttempts = 45  # llama.cpp takes longer to load the model
        $attempt = 0
        
        do {
            Start-Sleep -Seconds 2
            $attempt++
            if ($Verbose) {
                Write-ColorOutput Gray "   Attempt $attempt/$maxAttempts..."
            }
        } while (-not (Test-Port 8080) -and $attempt -lt $maxAttempts)
        
        if (Test-Port 8080) {
            Write-ColorOutput Green "✅ llama.cpp is ready on http://localhost:8080"
            return $true
        } else {
            Write-ColorOutput Red "❌ llama.cpp failed to start within timeout"
            return $false
        }
    } catch {
        Write-ColorOutput Red "❌ Failed to start llama.cpp: $($_.Exception.Message)"
        return $false
    }
}

# Main execution
$success = $true

switch ($Service) {
    "ollama" {
        $ollamaStarted = Start-Ollama
        if ($ollamaStarted -and $LoadGemma) {
            Load-GemmaModel | Out-Null
        }
        $success = $ollamaStarted
    }
    "llamacpp" {
        $success = Start-LlamaCpp
    }
    "both" {
        $ollamaStarted = Start-Ollama
        if ($ollamaStarted -and $LoadGemma) {
            Load-GemmaModel | Out-Null
        }
        
        $llamacppStarted = Start-LlamaCpp
        $success = $ollamaStarted -or $llamacppStarted
    }
}

# Final status report
Write-ColorOutput Cyan "=================================================="
Write-ColorOutput Green "🏁 Local LLM Services Status Report"
Write-ColorOutput Cyan "=================================================="

if (Test-Port 11434) {
    Write-ColorOutput Green "✅ Ollama: Running on http://localhost:11434"
} else {
    Write-ColorOutput Red "❌ Ollama: Not running"
}

if (Test-Port 8080) {
    Write-ColorOutput Green "✅ llama.cpp: Running on http://localhost:8080"
} else {
    Write-ColorOutput Red "❌ llama.cpp: Not running"
}

if ($success) {
    Write-ColorOutput Green "`n🎉 Local LLM services are ready for the SvelteKit application!"
    Write-ColorOutput Cyan "You can now run 'npm run dev' to start the web application."
} else {
    Write-ColorOutput Red "`n❌ Some services failed to start. Check the logs above."
    exit 1
}

Write-ColorOutput Cyan "`n📝 To stop services, use Ctrl+C or close this terminal."
Write-ColorOutput Cyan "To test the integration, visit: http://localhost:5173/test-ai-ask"
