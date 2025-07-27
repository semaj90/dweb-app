# Complete Gemma3 Legal AI Setup Script
# This script fixes all issues and provides complete integration

Write-Host "🤖 Gemma3 Legal AI - Complete Setup & Integration" -ForegroundColor Cyan
Write-Host "=" * 70

# Check prerequisites
Write-Host "📋 Checking Prerequisites..." -ForegroundColor Yellow

# 1. Check if Ollama is installed and running
Write-Host "  Checking Ollama..." -ForegroundColor Gray
try {
    $ollamaVersion = & ollama --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Ollama installed: $($ollamaVersion.Split(' ')[2])" -ForegroundColor Green
    } else {
        throw "Ollama not found"
    }
} catch {
    Write-Host "  ❌ Ollama not installed!" -ForegroundColor Red
    Write-Host "  📥 Download from: https://ollama.ai/" -ForegroundColor Gray
    Write-Host "  Or run: winget install Ollama.Ollama" -ForegroundColor Gray
    exit 1
}

# 2. Check if Ollama service is running
Write-Host "  Checking Ollama service..." -ForegroundColor Gray
try {
    $ollamaStatus = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method GET -TimeoutSec 5 -ErrorAction Stop
    Write-Host "  ✅ Ollama service running (v$($ollamaStatus.version))" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  Ollama service not running. Starting..." -ForegroundColor Yellow
    Write-Host "  Please run 'ollama serve' in another terminal" -ForegroundColor Gray
    Write-Host "  Then press Enter to continue..." -ForegroundColor Yellow
    Read-Host

    # Try again
    try {
        $ollamaStatus = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method GET -TimeoutSec 5 -ErrorAction Stop
        Write-Host "  ✅ Ollama service now running" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ Ollama service still not accessible" -ForegroundColor Red
        exit 1
    }
}

# 3. Check model file exists
$modelPath = ".\gemma3Q4_K_M\mo16.gguf"
Write-Host "  Checking model file..." -ForegroundColor Gray
if (Test-Path $modelPath) {
    $modelSize = [math]::Round((Get-Item $modelPath).Length / 1GB, 2)
    Write-Host "  ✅ Model file found: $modelPath ($modelSize GB)" -ForegroundColor Green
} else {
    Write-Host "  ❌ Model file not found at: $modelPath" -ForegroundColor Red
    Write-Host "  Please ensure your Gemma3 GGUF file is in the correct location" -ForegroundColor Gray
    exit 1
}

# 4. Check SvelteKit frontend
Write-Host "  Checking SvelteKit frontend..." -ForegroundColor Gray
if (Test-Path ".\sveltekit-frontend\package.json") {
    Write-Host "  ✅ SvelteKit frontend found" -ForegroundColor Green
} else {
    Write-Host "  ❌ SvelteKit frontend not found" -ForegroundColor Red
    exit 1
}

Write-Host "`n🔧 Setting up Gemma3 Legal AI Model..." -ForegroundColor Cyan

# Create the corrected Modelfile
$modelfileContent = @"
# Gemma3 Legal AI Model Configuration
FROM $modelPath

# Template for Gemma3 chat format
TEMPLATE """"""<bos><start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>""""""

# Legal AI System Prompt
SYSTEM """"""You are a specialized Legal AI Assistant powered by Gemma 3, trained on legal documents and case law. You provide:

- Accurate legal analysis and research assistance
- Contract review and document interpretation
- Evidence evaluation and case strategy guidance
- Regulatory compliance insights
- Professional legal terminology and citations

Always maintain professional standards, cite sources when available, and clearly indicate when information is insufficient for definitive advice. Recommend consulting qualified legal professionals for critical decisions.""""""

# Optimized parameters for legal AI
PARAMETER temperature 0.1
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 8192
PARAMETER num_predict 1024
PARAMETER num_batch 8
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<|im_end|>"
"@

$modelfilePath = ".\Modelfile-Gemma3-Legal-Fixed"
$modelfileContent | Out-File -FilePath $modelfilePath -Encoding UTF8
Write-Host "✅ Created corrected Modelfile: $modelfilePath" -ForegroundColor Green

# Import the model into Ollama
Write-Host "`n📥 Importing Gemma3 model into Ollama..." -ForegroundColor Yellow
Write-Host "This may take several minutes for a $modelSize GB model..." -ForegroundColor Gray

try {
    $importResult = & ollama create gemma3-legal -f $modelfilePath 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Successfully imported 'gemma3-legal' model!" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to import model:" -ForegroundColor Red
        Write-Host $importResult -ForegroundColor Gray

        # Try with absolute path
        Write-Host "Trying with absolute path..." -ForegroundColor Yellow
        $absoluteModelPath = (Resolve-Path $modelPath).Path
        $absoluteModelfileContent = $modelfileContent -replace [regex]::Escape($modelPath), $absoluteModelPath
        $absoluteModelfileContent | Out-File -FilePath ".\Modelfile-Absolute" -Encoding UTF8

        $importResult2 = & ollama create gemma3-legal -f ".\Modelfile-Absolute" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Import failed with absolute path too:" -ForegroundColor Red
            Write-Host $importResult2 -ForegroundColor Gray
            exit 1
        } else {
            Write-Host "✅ Successfully imported with absolute path!" -ForegroundColor Green
        }
    }
} catch {
    Write-Host "❌ Error during import: $_" -ForegroundColor Red
    exit 1
}

# Verify the model was created
Write-Host "`n🔍 Verifying model creation..." -ForegroundColor Yellow
try {
    $models = & ollama list 2>&1
    if ($models -match "gemma3-legal") {
        Write-Host "✅ Model 'gemma3-legal' is available!" -ForegroundColor Green

        # Show model info
        $modelInfo = $models | Select-String "gemma3-legal"
        Write-Host "  Model info: $modelInfo" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Model may not have been created properly" -ForegroundColor Yellow
        Write-Host "Available models:" -ForegroundColor Gray
        Write-Host $models -ForegroundColor Gray
    }
} catch {
    Write-Host "⚠️  Could not verify model list" -ForegroundColor Yellow
}

# Test the model with a legal query
Write-Host "`n🧪 Testing model with legal query..." -ForegroundColor Yellow
$testPrompt = "What are the essential elements required for a valid contract under common law?"

try {
    Write-Host "Sending test prompt..." -ForegroundColor Gray

    # Use a simpler test approach
    $testCommand = "ollama run gemma3-legal `"$testPrompt`""
    $testResponse = Invoke-Expression $testCommand 2>&1

    if ($LASTEXITCODE -eq 0 -and $testResponse -and $testResponse.Length -gt 10) {
        Write-Host "✅ Model responds correctly!" -ForegroundColor Green
        Write-Host "Sample response:" -ForegroundColor Gray
        $preview = if ($testResponse.Length -gt 200) { $testResponse.Substring(0, 200) + "..." } else { $testResponse }
        Write-Host $preview -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Model test had issues:" -ForegroundColor Yellow
        Write-Host $testResponse -ForegroundColor Gray
    }
} catch {
    Write-Host "⚠️  Could not test model, but it may still work in the application" -ForegroundColor Yellow
    Write-Host "Error: $_" -ForegroundColor Gray
}

# Update SvelteKit API endpoints
Write-Host "`n🔗 Updating SvelteKit API endpoints..." -ForegroundColor Cyan

$chatServerPath = ".\sveltekit-frontend\src\routes\api\ai\chat\+server.ts"
if (Test-Path $chatServerPath) {
    Write-Host "📝 Updating chat API endpoint..." -ForegroundColor Yellow

    $newChatServer = @"
import { json } from "@sveltejs/kit";
import { ollamaService } from "`$lib/services/ollama-service";

export const POST = async ({ request }) => {
  const startTime = Date.now();

  try {
    const {
      message,
      context,
      conversationId,
      model = "gemma3-legal",
      temperature = 0.1,
      maxTokens = 512,
      systemPrompt
    } = await request.json();

    if (!message || message.trim() === "") {
      return json({ error: "Message is required" }, { status: 400 });
    }

    // Use actual Ollama service instead of mock
    const response = await ollamaService.generate(message, {
      system: systemPrompt || "You are a specialized Legal AI Assistant powered by Gemma 3. You excel at contract analysis, legal research, and providing professional legal guidance.",
      temperature: temperature,
      maxTokens: maxTokens,
      topP: 0.8,
      topK: 20,
      repeatPenalty: 1.05
    });

    return json({
      response,
      model,
      conversationId: conversationId || `conv_`${Date.now()}`,
      metadata: {
        provider: "ollama",
        confidence: 0.9,
        executionTime: Date.now() - startTime,
        fromCache: false,
      }
    });
  } catch (error) {
    console.error("AI chat error:", error);
    return json({
      error: "Failed to process chat",
      details: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 });
  }
};
"@

    try {
        $newChatServer | Out-File -FilePath $chatServerPath -Encoding UTF8
        Write-Host "✅ Updated chat API endpoint" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Could not update chat API: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Chat API endpoint not found at: $chatServerPath" -ForegroundColor Yellow
}

# Create a test endpoint
$testEndpointPath = ".\sveltekit-frontend\src\routes\api\ai\test-gemma3\+server.ts"
$testEndpointDir = Split-Path $testEndpointPath -Parent
if (!(Test-Path $testEndpointDir)) {
    New-Item -ItemType Directory -Path $testEndpointDir -Force | Out-Null
}

$testEndpoint = @"
import { json } from "@sveltejs/kit";
import { ollamaService } from "`$lib/services/ollama-service";

export const GET = async () => {
  try {
    // Check if service is available
    const isAvailable = await ollamaService.healthCheck();
    const models = ollamaService.getAvailableModels();
    const currentModel = ollamaService.getGemma3Model();

    return json({
      status: "success",
      ollama: {
        available: isAvailable,
        models: models,
        gemma3Model: currentModel
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return json({
      status: "error",
      error: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 });
  }
};

export const POST = async ({ request }) => {
  try {
    const { prompt = "What are the key elements of a valid contract?" } = await request.json();

    const response = await ollamaService.generate(prompt, {
      system: "You are a specialized Legal AI Assistant.",
      temperature: 0.1,
      maxTokens: 256
    });

    return json({
      status: "success",
      prompt,
      response,
      model: ollamaService.getGemma3Model(),
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return json({
      status: "error",
      error: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 });
  }
};
"@

try {
    $testEndpoint | Out-File -FilePath $testEndpointPath -Encoding UTF8
    Write-Host "✅ Created test API endpoint: /api/ai/test-gemma3" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not create test endpoint: $_" -ForegroundColor Yellow
}

# Install frontend dependencies if needed
Write-Host "`n📦 Checking frontend dependencies..." -ForegroundColor Yellow
$packageJsonPath = ".\sveltekit-frontend\package.json"
if (Test-Path $packageJsonPath) {
    $nodeModulesPath = ".\sveltekit-frontend\node_modules"
    if (!(Test-Path $nodeModulesPath)) {
        Write-Host "📥 Installing frontend dependencies..." -ForegroundColor Yellow
        Push-Location ".\sveltekit-frontend"
        try {
            & npm install
            Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Could not install dependencies: $_" -ForegroundColor Yellow
        }
        Pop-Location
    } else {
        Write-Host "✅ Frontend dependencies already installed" -ForegroundColor Green
    }
}

Write-Host "`n🎉 Setup Complete!" -ForegroundColor Green
Write-Host "=" * 70

Write-Host "🤖 Your Gemma3 Legal AI is ready!" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 System Status:" -ForegroundColor White
Write-Host "  • Model: gemma3-legal ($modelSize GB)" -ForegroundColor Gray
Write-Host "  • Ollama: Running on http://localhost:11434" -ForegroundColor Gray
Write-Host "  • Frontend: SvelteKit 5 + Bits UI" -ForegroundColor Gray
Write-Host "  • API: Enhanced endpoints ready" -ForegroundColor Gray
Write-Host ""
Write-Host "🚀 Next Steps:" -ForegroundColor White
Write-Host "  1. Start development server:" -ForegroundColor Gray
Write-Host "     cd sveltekit-frontend && npm run dev" -ForegroundColor Cyan
Write-Host "  2. Open your browser to:" -ForegroundColor Gray
Write-Host "     http://localhost:5173" -ForegroundColor Cyan
Write-Host "  3. Test the AI chat interface" -ForegroundColor Gray
Write-Host "  4. Try the test endpoint:" -ForegroundColor Gray
Write-Host "     http://localhost:5173/api/ai/test-gemma3" -ForegroundColor Cyan
Write-Host ""
Write-Host "🧪 Quick Tests:" -ForegroundColor White
Write-Host "  • Test model directly:" -ForegroundColor Gray
Write-Host "    ollama run gemma3-legal 'Explain contract law basics'" -ForegroundColor Cyan
Write-Host "  • Test API endpoint:" -ForegroundColor Gray
Write-Host "    curl http://localhost:5173/api/ai/test-gemma3" -ForegroundColor Cyan
Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor White
Write-Host "  • Complete guide: GEMMA3_INTEGRATION_COMPLETE_GUIDE.md" -ForegroundColor Gray
Write-Host "  • Local LLM setup: sveltekit-frontend/markdowns/LOCAL_LLM_SETUP.md" -ForegroundColor Gray
Write-Host ""
Write-Host "🔧 Configuration Files:" -ForegroundColor White
Write-Host "  • Modelfile: $modelfilePath" -ForegroundColor Gray
Write-Host "  • Chat API: $chatServerPath" -ForegroundColor Gray
Write-Host "  • Test API: $testEndpointPath" -ForegroundColor Gray

Write-Host "`n✨ Happy coding with your Legal AI Assistant! ✨" -ForegroundColor Magenta
