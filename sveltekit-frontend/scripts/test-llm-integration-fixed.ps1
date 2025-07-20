# PowerShell script to test local LLM integration with SvelteKit
# Tests Ollama and llama.cpp services and their integration with the web app

param(
    [Parameter(Mandatory=$false)]
    [switch]$StartServices,
    
    [Parameter(Mandatory=$false)]
    [switch]$LoadGemma,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose
)

# Configuration
$RootPath = "c:\Users\james\Downloads\Deeds-App-doesn-t-work--main (2)"
$FrontendPath = Join-Path $RootPath "web-app\sveltekit-frontend"
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

Write-ColorOutput Green " Testing Local LLM Integration with SvelteKit"
Write-ColorOutput Cyan "=============================================="

# Function to test if a port is in use
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

# Function to test HTTP endpoint
function Test-HttpEndpoint($Url, $Timeout = 5) {
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec $Timeout -UseBasicParsing
        return @{ Success = $true; StatusCode = $response.StatusCode; Response = $response }
    } catch {
        return @{ Success = $false; Error = $_.Exception.Message }
    }
}

# Function to test Ollama API
function Test-OllamaAPI() {
    Write-ColorOutput Cyan " Testing Ollama API..."
    
    $versionTest = Test-HttpEndpoint "http://localhost:11434/api/version"
    if ($versionTest.Success) {
        Write-ColorOutput Green "   Ollama version endpoint accessible"
        
        # Test models endpoint
        $modelsTest = Test-HttpEndpoint "http://localhost:11434/api/tags"
        if ($modelsTest.Success) {
            try {
                $models = $modelsTest.Response.Content | ConvertFrom-Json
                $modelCount = $models.models.Count
                Write-ColorOutput Green "   Found $modelCount models in Ollama"
                
                if ($Verbose) {
                    foreach ($model in $models.models) {
                        Write-ColorOutput Gray "    - $($model.name)"
                    }
                }
                
                # Check for Gemma models
                $gemmaModels = $models.models | Where-Object { $_.name -like "*gemma*" }
                if ($gemmaModels.Count -gt 0) {
                    Write-ColorOutput Green "   Found $($gemmaModels.Count) Gemma model(s)"
                    return @{ Success = $true; HasGemma = $true; Models = $models.models }
                } else {
                    Write-ColorOutput Yellow "   No Gemma models found"
                    return @{ Success = $true; HasGemma = $false; Models = $models.models }
                }
            } catch {
                Write-ColorOutput Red "   Failed to parse models response"
                return @{ Success = $false; Error = "Failed to parse models" }
            }
        } else {
            Write-ColorOutput Red "   Models endpoint failed: $($modelsTest.Error)"
            return @{ Success = $false; Error = $modelsTest.Error }
        }
    } else {
        Write-ColorOutput Red "   Ollama not accessible: $($versionTest.Error)"
        return @{ Success = $false; Error = $versionTest.Error }
    }
}

# Function to test llama.cpp API
function Test-LlamaCppAPI() {
    Write-ColorOutput Cyan " Testing llama.cpp API..."
    
    $healthTest = Test-HttpEndpoint "http://localhost:8080/health"
    if ($healthTest.Success) {
        Write-ColorOutput Green "   llama.cpp health endpoint accessible"
        return @{ Success = $true }
    } else {
        Write-ColorOutput Red "   llama.cpp not accessible: $($healthTest.Error)"
        return @{ Success = $false; Error = $healthTest.Error }
    }
}

# Function to test generation with Ollama
function Test-OllamaGeneration($Model = "gemma3-legal") {
    Write-ColorOutput Cyan " Testing Ollama generation with model: $Model"
    
    $prompt = "What are the key elements of a valid contract? Provide a brief answer."
    $requestBody = @{
        model = $Model
        prompt = $prompt
        stream = $false
        options = @{
            temperature = 0.7
            num_predict = 100
        }
    } | ConvertTo-Json -Depth 3
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method POST -Body $requestBody -ContentType "application/json" -TimeoutSec 30
        
        if ($response.response) {
            Write-ColorOutput Green "   Generation successful"
            if ($Verbose) {
                Write-ColorOutput Gray "  Response: $($response.response.Substring(0, [Math]::Min(100, $response.response.Length)))..."
            }
            return @{ Success = $true; Response = $response.response }
        } else {
            Write-ColorOutput Red "   No response generated"
            return @{ Success = $false; Error = "No response generated" }
        }
    } catch {
        Write-ColorOutput Red "   Generation failed: $($_.Exception.Message)"
        return @{ Success = $false; Error = $_.Exception.Message }
    }
}

# Function to test SvelteKit API endpoints
function Test-SvelteKitAPI() {
    Write-ColorOutput Cyan " Testing SvelteKit API endpoints..."
    
    # Test local health endpoint
    $healthTest = Test-HttpEndpoint "http://localhost:5173/api/ai/health/local"
    if ($healthTest.Success) {
        Write-ColorOutput Green "   SvelteKit health endpoint accessible"
        
        try {
            $healthData = $healthTest.Response.Content | ConvertFrom-Json
            if ($healthData.success) {
                Write-ColorOutput Green "   Health check passed"
                if ($Verbose) {
                    Write-ColorOutput Gray "  Ollama: $($healthData.services.ollama.direct.success)"
                    Write-ColorOutput Gray "  llama.cpp: $($healthData.services.llamaCpp.success)"
                }
            } else {
                Write-ColorOutput Yellow "   Health check reported issues"
            }
        } catch {
            Write-ColorOutput Red "   Failed to parse health response"
        }
    } else {
        Write-ColorOutput Red "   SvelteKit not accessible: $($healthTest.Error)"
        return @{ Success = $false; Error = "SvelteKit not running" }
    }
    
    # Test AI ask endpoint
    $askBody = @{
        query = "What are the basic requirements for forming a contract?"
        context = @()
        includeHistory = $false
        maxSources = 3
        searchThreshold = 0.7
        useCache = $false
    } | ConvertTo-Json -Depth 3
    
    try {
        $askResponse = Invoke-RestMethod -Uri "http://localhost:5173/api/ai/ask" -Method POST -Body $askBody -ContentType "application/json" -TimeoutSec 30
        
        if ($askResponse.success) {
            Write-ColorOutput Green "   AI ask endpoint working"
            if ($Verbose) {
                Write-ColorOutput Gray "  Provider: $($askResponse.data.provider)"
                Write-ColorOutput Gray "  Model: $($askResponse.data.model)"
                Write-ColorOutput Gray "  Answer: $($askResponse.data.answer.Substring(0, [Math]::Min(100, $askResponse.data.answer.Length)))..."
            }
            return @{ Success = $true; Provider = $askResponse.data.provider; Model = $askResponse.data.model }
        } else {
            Write-ColorOutput Yellow "   AI ask endpoint returned error: $($askResponse.error)"
            return @{ Success = $false; Error = $askResponse.error }
        }
    } catch {
        Write-ColorOutput Red "   AI ask endpoint failed: $($_.Exception.Message)"
        return @{ Success = $false; Error = $_.Exception.Message }
    }
}

# Main test execution
Write-ColorOutput Cyan " Pre-flight checks..."

# Check if paths exist
$pathsValid = $true
if (!(Test-Path $OllamaPath)) {
    Write-ColorOutput Red " Ollama directory not found: $OllamaPath"
    $pathsValid = $false
}

if (!(Test-Path $LlamaCppPath)) {
    Write-ColorOutput Red " llama.cpp directory not found: $LlamaCppPath"
    $pathsValid = $false
}

if (!(Test-Path $GemmaModelPath)) {
    Write-ColorOutput Yellow " Gemma model not found: $GemmaModelPath"
}

if (!(Test-Path $FrontendPath)) {
    Write-ColorOutput Red " SvelteKit frontend not found: $FrontendPath"
    $pathsValid = $false
}

if (!$pathsValid) {
    Write-ColorOutput Red " Required paths missing. Please check your installation."
    exit 1
}

# Start services if requested
if ($StartServices) {
    Write-ColorOutput Cyan " Starting services..."
    $startScript = Join-Path $FrontendPath "scripts\start-local-llms.ps1"
    
    if (Test-Path $startScript) {
        $startArgs = @("-Service", "both")
        if ($LoadGemma) { $startArgs += "-LoadGemma" }
        if ($Verbose) { $startArgs += "-Verbose" }
        
        & $startScript @startArgs
        Start-Sleep -Seconds 5
    } else {
        Write-ColorOutput Red " Start script not found: $startScript"
    }
}

# Test services
Write-ColorOutput Cyan "`n Running service tests..."

$ollamaResult = Test-OllamaAPI
$llamaCppResult = Test-LlamaCppAPI

# Test generation if Ollama is available
if ($ollamaResult.Success -and $ollamaResult.HasGemma) {
    $generationResult = Test-OllamaGeneration
} elseif ($ollamaResult.Success -and $ollamaResult.Models.Count -gt 0) {
    $firstModel = $ollamaResult.Models[0].name
    Write-ColorOutput Yellow "  Trying with available model: $firstModel"
    $generationResult = Test-OllamaGeneration $firstModel
}

# Test SvelteKit integration if port 5173 is available
if (Test-Port 5173) {
    $svelteKitResult = Test-SvelteKitAPI
} else {
    Write-ColorOutput Yellow " SvelteKit not running on port 5173. Start with 'npm run dev'"
}

# Final report
Write-ColorOutput Cyan "`n Test Results Summary"
Write-ColorOutput Cyan "========================"

$overallSuccess = $true

if ($ollamaResult.Success) {
    Write-ColorOutput Green " Ollama: Available"
    if ($ollamaResult.HasGemma) {
        Write-ColorOutput Green " Gemma Models: Found"
    } else {
        Write-ColorOutput Yellow " Gemma Models: Not found"
        $overallSuccess = $false
    }
} else {
    Write-ColorOutput Red " Ollama: Not available"
    $overallSuccess = $false
}

if ($llamaCppResult.Success) {
    Write-ColorOutput Green " llama.cpp: Available"
} else {
    Write-ColorOutput Red " llama.cpp: Not available"
}

if ($generationResult.Success) {
    Write-ColorOutput Green " Text Generation: Working"
} else {
    Write-ColorOutput Red " Text Generation: Failed"
    $overallSuccess = $false
}

if ($svelteKitResult.Success) {
    Write-ColorOutput Green " SvelteKit Integration: Working ($($svelteKitResult.Provider))"
} elseif ($svelteKitResult) {
    Write-ColorOutput Red " SvelteKit Integration: Failed"
    $overallSuccess = $false
} else {
    Write-ColorOutput Yellow " SvelteKit Integration: Not tested (app not running)"
}

# Recommendations
Write-ColorOutput Cyan "`nRECOMMENDATIONS"
Write-ColorOutput Cyan "=================="

if (!$ollamaResult.Success) {
    Write-ColorOutput Yellow " Start Ollama: .\scripts\start-local-llms.ps1 -Service ollama"
}

if ($ollamaResult.Success -and !$ollamaResult.HasGemma) {
    Write-ColorOutput Yellow " Load Gemma model: .\scripts\start-local-llms.ps1 -LoadGemma"
}

if (!$svelteKitResult -or !$svelteKitResult.Success) {
    Write-ColorOutput Yellow " Start SvelteKit: npm run dev"
}

if ($overallSuccess) {
    Write-ColorOutput Green "`n All tests passed! Your local LLM integration is ready."
    Write-ColorOutput Cyan "Visit http://localhost:5173/test-ai-ask to test the UI."
} else {
    Write-ColorOutput Red "`n Some tests failed. Please address the issues above."
}

Write-ColorOutput Cyan "`n For more information, check the logs or run with -Verbose flag."
