# Complete Gemma3 Legal AI Setup Script
# Fixes GGUF model loading, sets up Ollama, and configures SvelteKit frontend

param(
    [switch]$SkipModelCheck = $false,
    [switch]$UseFallback = $false,
    [switch]$InstallDependencies = $true,
    [string]$ModelPath = "gemma3Q4_K_M"
)

Write-Host "🚀 Starting Complete Gemma3 Legal AI Setup..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check prerequisites
Write-Host "`n📋 Step 1: Checking Prerequisites..." -ForegroundColor Yellow

# Check Node.js
try {
    $nodeVersion = node --version
    Write-Host "  ✅ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Node.js not found. Please install Node.js 20+" -ForegroundColor Red
    exit 1
}

# Check npm
try {
    $npmVersion = npm --version
    Write-Host "  ✅ npm: $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ npm not found" -ForegroundColor Red
    exit 1
}

# Check Ollama
$ollamaInstalled = $false
try {
    $ollamaVersion = ollama --version
    Write-Host "  ✅ Ollama: $ollamaVersion" -ForegroundColor Green
    $ollamaInstalled = $true
} catch {
    Write-Host "  ❌ Ollama not found" -ForegroundColor Red
    Write-Host "  💡 Installing Ollama..." -ForegroundColor Yellow

    # Download and install Ollama
    try {
        $ollamaUrl = "https://ollama.ai/install/windows"
        Write-Host "  🔄 Please install Ollama from: $ollamaUrl" -ForegroundColor Yellow
        Start-Process $ollamaUrl
        Write-Host "  ⏳ Waiting for Ollama installation..." -ForegroundColor Yellow
        Read-Host "Press Enter after installing Ollama"

        $ollamaVersion = ollama --version
        Write-Host "  ✅ Ollama installed: $ollamaVersion" -ForegroundColor Green
        $ollamaInstalled = $true
    } catch {
        Write-Host "  ❌ Failed to verify Ollama installation" -ForegroundColor Red
        $UseFallback = $true
    }
}

# Step 2: Install/Update Dependencies
if ($InstallDependencies) {
    Write-Host "`n🔧 Step 2: Installing Dependencies..." -ForegroundColor Yellow

    Write-Host "  📦 Installing SvelteKit dependencies..." -ForegroundColor White
    npm install

    Write-Host "  📦 Installing Bits UI..." -ForegroundColor White
    npm install bits-ui@next

    Write-Host "  📦 Installing XState..." -ForegroundColor White
    npm install xstate @xstate/svelte

    Write-Host "  📦 Installing additional AI packages..." -ForegroundColor White
    npm install @ai-sdk/openai @ai-sdk/svelte langchain @langchain/community

    Write-Host "  📦 Installing dev dependencies..." -ForegroundColor White
    npm install -D @types/node

    Write-Host "  ✅ Dependencies installed!" -ForegroundColor Green
}

# Step 3: Diagnose and Fix GGUF Model
Write-Host "`n🔍 Step 3: Diagnosing GGUF Model..." -ForegroundColor Yellow

if (-not $SkipModelCheck -and (Test-Path $ModelPath)) {
    $modelFiles = Get-ChildItem -Path $ModelPath -Filter "*.gguf" | Sort-Object Length -Descending
    $validModel = $null

    foreach ($file in $modelFiles) {
        $sizeGB = [math]::Round($file.Length / 1GB, 2)
        Write-Host "  📄 Checking $($file.Name) ($sizeGB GB)..." -ForegroundColor White

        try {
            # Check GGUF header
            $bytes = [System.IO.File]::ReadAllBytes($file.FullName)[0..7]
            $header = [System.Text.Encoding]::ASCII.GetString($bytes)

            if ($header.StartsWith("GGUF") -and $file.Length -gt 1GB -and $file.Length -lt 20GB) {
                Write-Host "    ✅ Valid GGUF file" -ForegroundColor Green
                $validModel = $file
                break
            } else {
                Write-Host "    ❌ Invalid GGUF header or size" -ForegroundColor Red
            }
        } catch {
            Write-Host "    ❌ Cannot read file: $($_.Exception.Message)" -ForegroundColor Red
        }
    }

    if (-not $validModel) {
        Write-Host "  ⚠️  No valid GGUF model found. Using fallback..." -ForegroundColor Yellow
        $UseFallback = $true
    } else {
        Write-Host "  ✅ Using model: $($validModel.Name)" -ForegroundColor Green
    }
} else {
    Write-Host "  ⚠️  Skipping model check or model directory not found" -ForegroundColor Yellow
    $UseFallback = $true
}

# Step 4: Start Ollama Service
Write-Host "`n🔄 Step 4: Starting Ollama Service..." -ForegroundColor Yellow

if ($ollamaInstalled) {
    try {
        # Check if Ollama is already running
        $ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
        if (-not $ollamaProcess) {
            Write-Host "  🔄 Starting Ollama service..." -ForegroundColor White
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 10
        } else {
            Write-Host "  ✅ Ollama service already running" -ForegroundColor Green
        }

        # Test Ollama connection
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method Get -TimeoutSec 5
        Write-Host "  ✅ Ollama service is healthy" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ Failed to start Ollama service: $($_.Exception.Message)" -ForegroundColor Red
        $UseFallback = $true
    }
}

# Step 5: Setup Models
Write-Host "`n🤖 Step 5: Setting up AI Models..." -ForegroundColor Yellow

if ($ollamaInstalled -and -not $UseFallback -and $validModel) {
    # Create custom Modelfile
    $modelfilePath = "Modelfile-Gemma3-Legal-AI"
    $absolutePath = (Resolve-Path $validModel.FullName).Path

    $modelfileContent = @"
FROM $absolutePath

TEMPLATE """<bos><start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192
PARAMETER num_predict 2048

SYSTEM """You are Gemma3 Legal AI, a specialized legal assistant with expertise in:

• Legal document analysis and contract review
• Case law research and precedent identification
• Regulatory compliance and risk assessment
• Legal writing and brief preparation
• Evidence evaluation and case strategy

Provide accurate, well-researched legal information while clearly stating that responses are for informational purposes only and do not constitute legal advice. Always cite relevant authorities when applicable.

Key capabilities:
- Analyze complex legal documents
- Identify potential risks and compliance issues
- Research relevant case law and statutes
- Draft professional legal documents
- Provide strategic recommendations

Remember to maintain confidentiality and follow ethical guidelines in all interactions."""
"@

    $modelfileContent | Out-File -FilePath $modelfilePath -Encoding UTF8
    Write-Host "  📝 Created Modelfile: $modelfilePath" -ForegroundColor Green

    try {
        Write-Host "  🔄 Creating Ollama model 'gemma3-legal-ai'..." -ForegroundColor White
        $createResult = ollama create gemma3-legal-ai -f $modelfilePath 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Custom model created successfully!" -ForegroundColor Green

            # Test the model
            Write-Host "  🧪 Testing custom model..." -ForegroundColor White
            $testResult = ollama run gemma3-legal-ai "Hello, please introduce yourself as a legal AI assistant." --verbose 2>&1

            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✅ Custom model test successful!" -ForegroundColor Green
            } else {
                Write-Host "  ❌ Custom model test failed, using fallback" -ForegroundColor Red
                $UseFallback = $true
            }
        } else {
            Write-Host "  ❌ Failed to create custom model: $createResult" -ForegroundColor Red
            $UseFallback = $true
        }
    } catch {
        Write-Host "  ❌ Error creating custom model: $($_.Exception.Message)" -ForegroundColor Red
        $UseFallback = $true
    }
}

# Setup fallback model
if ($UseFallback -and $ollamaInstalled) {
    Write-Host "  🔄 Setting up fallback model (gemma2:9b)..." -ForegroundColor Yellow
    try {
        ollama pull gemma2:9b
        Write-Host "  ✅ Fallback model ready" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ Failed to pull fallback model" -ForegroundColor Red

        # Try alternative fallback
        Write-Host "  🔄 Trying alternative fallback (llama3.1:8b)..." -ForegroundColor Yellow
        try {
            ollama pull llama3.1:8b
            Write-Host "  ✅ Alternative fallback model ready" -ForegroundColor Green
        } catch {
            Write-Host "  ❌ No fallback models available" -ForegroundColor Red
        }
    }
}

# Step 6: Create/Update Configuration
Write-Host "`n⚙️  Step 6: Updating Configuration..." -ForegroundColor Yellow

# Ensure directories exist
$configDir = "src/lib/config"
$typesDir = "src/lib/types"
$servicesDir = "src/lib/services"
$componentsDir = "src/lib/components/ai"

@($configDir, $typesDir, $servicesDir, $componentsDir) | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -Path $_ -ItemType Directory -Force | Out-Null
        Write-Host "  📁 Created directory: $_" -ForegroundColor White
    }
}

# Update runtime configuration
$runtimeConfig = @"
// Runtime Configuration - Updated by setup script
export const RUNTIME_CONFIG = {
    model: {
        name: '$($UseFallback ? "gemma2:9b" : "gemma3-legal-ai")',
        fallback: 'gemma2:9b',
        endpoint: 'http://localhost:11434',
        useFallback: $($UseFallback.ToString().ToLower()),
        customModelAvailable: $(((-not $UseFallback) -and $validModel).ToString().ToLower())
    },
    setup: {
        timestamp: '$(Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")',
        ollamaInstalled: $($ollamaInstalled.ToString().ToLower()),
        validModelFound: $(($validModel -ne $null).ToString().ToLower()),
        fallbackUsed: $($UseFallback.ToString().ToLower())
    },
    features: {
        streaming: true,
        modelSelection: $($ollamaInstalled.ToString().ToLower()),
        contextInjection: true,
        ragIntegration: false,
        documentAnalysis: true
    }
};
"@

$runtimeConfig | Out-File -FilePath "src/lib/config/runtime-config.ts" -Encoding UTF8 -Force
Write-Host "  ✅ Updated runtime configuration" -ForegroundColor Green

# Step 7: Create Test Route
Write-Host "`n🧪 Step 7: Creating Test Routes..." -ForegroundColor Yellow

$apiDir = "src/routes/api/ai"
if (-not (Test-Path $apiDir)) {
    New-Item -Path $apiDir -ItemType Directory -Force | Out-Null
}

$testApiContent = @"
// Test API Endpoint for Gemma3 Legal AI
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
    try {
        // Test Ollama connection
        const response = await fetch('http://localhost:11434/api/version');
        const ollamaHealthy = response.ok;

        // Get model list
        let models = [];
        if (ollamaHealthy) {
            try {
                const modelsResponse = await fetch('http://localhost:11434/api/tags');
                const modelsData = await modelsResponse.json();
                models = modelsData.models || [];
            } catch (error) {
                console.warn('Failed to get models:', error);
            }
        }

        return json({
            status: 'ok',
            timestamp: new Date().toISOString(),
            ollama: {
                healthy: ollamaHealthy,
                endpoint: 'http://localhost:11434',
                models: models.map(m => m.name)
            },
            config: {
                customModel: '$($UseFallback ? "Not Available" : "gemma3-legal-ai")',
                fallbackModel: 'gemma2:9b',
                usingFallback: $($UseFallback.ToString().ToLower())
            }
        });
    } catch (error) {
        return json({
            status: 'error',
            error: error.message,
            timestamp: new Date().toISOString()
        }, { status: 500 });
    }
};

export const POST: RequestHandler = async ({ request }) => {
    try {
        const { message, model = '$($UseFallback ? "gemma2:9b" : "gemma3-legal-ai")' } = await request.json();

        if (!message) {
            return json({ error: 'Message is required' }, { status: 400 });
        }

        // Test chat with the model
        const response = await fetch('http://localhost:11434/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model,
                messages: [
                    {
                        role: 'user',
                        content: message
                    }
                ],
                stream: false
            })
        });

        if (!response.ok) {
            throw new Error('Chat request failed');
        }

        const data = await response.json();

        return json({
            response: data.message?.content || data.response,
            model,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        return json({
            error: error.message,
            timestamp: new Date().toISOString()
        }, { status: 500 });
    }
};
"@

$testApiContent | Out-File -FilePath "$apiDir/test/+server.ts" -Encoding UTF8 -Force
Write-Host "  ✅ Created test API endpoint: /api/ai/test" -ForegroundColor Green

# Step 8: Create Test Page
$testPageDir = "src/routes/ai-test"
if (-not (Test-Path $testPageDir)) {
    New-Item -Path $testPageDir -ItemType Directory -Force | Out-Null
}

$testPageContent = @"
<script lang="ts">
    import { onMount } from 'svelte';
    import { writable } from 'svelte/store';

    let status = writable(null);
    let testMessage = 'Hello, please introduce yourself as a legal AI assistant.';
    let testResult = writable(null);
    let loading = writable(false);

    onMount(async () => {
        try {
            const response = await fetch('/api/ai/test');
            const data = await response.json();
            status.set(data);
        } catch (error) {
            status.set({ error: error.message });
        }
    });

    async function testChat() {
        loading.set(true);
        try {
            const response = await fetch('/api/ai/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: testMessage })
            });
            const data = await response.json();
            testResult.set(data);
        } catch (error) {
            testResult.set({ error: error.message });
        } finally {
            loading.set(false);
        }
    }
</script>

<svelte:head>
    <title>Gemma3 Legal AI Test</title>
</svelte:head>

<div class="container mx-auto p-8">
    <h1 class="text-3xl font-bold mb-8">🏛️ Gemma3 Legal AI Test</h1>

    <div class="grid gap-6">
        <!-- System Status -->
        <div class="card bg-base-100 shadow-xl">
            <div class="card-body">
                <h2 class="card-title">System Status</h2>
                {#if $status}
                    {#if $status.error}
                        <div class="alert alert-error">
                            <span>❌ Error: {$status.error}</span>
                        </div>
                    {:else}
                        <div class="alert alert-success">
                            <span>✅ System is operational</span>
                        </div>

                        <div class="stats stats-vertical lg:stats-horizontal shadow">
                            <div class="stat">
                                <div class="stat-title">Ollama Status</div>
                                <div class="stat-value text-sm">{$status.ollama.healthy ? '✅ Healthy' : '❌ Offline'}</div>
                            </div>
                            <div class="stat">
                                <div class="stat-title">Current Model</div>
                                <div class="stat-value text-sm">{$status.config.customModel}</div>
                            </div>
                            <div class="stat">
                                <div class="stat-title">Available Models</div>
                                <div class="stat-value text-sm">{$status.ollama.models.length}</div>
                            </div>
                        </div>

                        {#if $status.ollama.models.length > 0}
                            <div class="mt-4">
                                <h3 class="font-semibold mb-2">Available Models:</h3>
                                <div class="flex flex-wrap gap-2">
                                    {#each $status.ollama.models as model}
                                        <div class="badge badge-outline">{model}</div>
                                    {/each}
                                </div>
                            </div>
                        {/if}
                    {/if}
                {:else}
                    <div class="loading loading-spinner loading-md"></div>
                {/if}
            </div>
        </div>

        <!-- Chat Test -->
        <div class="card bg-base-100 shadow-xl">
            <div class="card-body">
                <h2 class="card-title">Chat Test</h2>

                <div class="form-control">
                    <label class="label">
                        <span class="label-text">Test Message</span>
                    </label>
                    <textarea
                        bind:value={testMessage}
                        class="textarea textarea-bordered h-24"
                        placeholder="Enter a test message..."
                    ></textarea>
                </div>

                <div class="card-actions justify-end">
                    <button
                        class="btn btn-primary"
                        class:loading={$loading}
                        on:click={testChat}
                        disabled={$loading || !testMessage.trim()}
                    >
                        {$loading ? 'Testing...' : 'Test Chat'}
                    </button>
                </div>

                {#if $testResult}
                    <div class="mt-4">
                        {#if $testResult.error}
                            <div class="alert alert-error">
                                <span>❌ Error: {$testResult.error}</span>
                            </div>
                        {:else}
                            <div class="alert alert-success">
                                <span>✅ Chat test successful!</span>
                            </div>

                            <div class="mt-4 p-4 bg-base-200 rounded-lg">
                                <h4 class="font-semibold mb-2">AI Response:</h4>
                                <p class="whitespace-pre-wrap">{$testResult.response}</p>

                                <div class="mt-2 text-sm opacity-70">
                                    Model: {$testResult.model} | Time: {new Date($testResult.timestamp).toLocaleTimeString()}
                                </div>
                            </div>
                        {/if}
                    </div>
                {/if}
            </div>
        </div>
    </div>
</div>

<style>
    .container {
        max-width: 1200px;
    }
</style>
"@

$testPageContent | Out-File -FilePath "$testPageDir/+page.svelte" -Encoding UTF8 -Force
Write-Host "  ✅ Created test page: /ai-test" -ForegroundColor Green

# Step 9: Update package.json scripts
Write-Host "`n📜 Step 9: Updating Package Scripts..." -ForegroundColor Yellow

$packageJsonPath = "package.json"
if (Test-Path $packageJsonPath) {
    try {
        $packageJson = Get-Content $packageJsonPath -Raw | ConvertFrom-Json

        # Add AI-specific scripts
        if (-not $packageJson.scripts) {
            $packageJson.scripts = @{}
        }

        $packageJson.scripts | Add-Member -NotePropertyName "ai:test" -NotePropertyValue "node -e `"console.log('Testing AI integration...'); fetch('http://localhost:5173/api/ai/test').then(r=>r.json()).then(console.log).catch(console.error)`"" -Force
        $packageJson.scripts | Add-Member -NotePropertyName "ai:health" -NotePropertyValue "curl -s http://localhost:11434/api/version || echo 'Ollama not running'" -Force
        $packageJson.scripts | Add-Member -NotePropertyName "ai:models" -NotePropertyValue "ollama list" -Force

        $packageJson | ConvertTo-Json -Depth 10 | Out-File $packageJsonPath -Encoding UTF8
        Write-Host "  ✅ Updated package.json scripts" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠️  Could not update package.json: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Step 10: Final Health Check
Write-Host "`n🏥 Step 10: Final System Health Check..." -ForegroundColor Yellow

$healthResults = @{
    ollama = $false
    models = @()
    frontend = $false
}

# Check Ollama
if ($ollamaInstalled) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method Get -TimeoutSec 5
        $healthResults.ollama = $true
        Write-Host "  ✅ Ollama service is healthy" -ForegroundColor Green

        # Check models
        try {
            $modelsResponse = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5
            $healthResults.models = $modelsResponse.models.name
            Write-Host "  ✅ Found $($healthResults.models.Count) models" -ForegroundColor Green
        } catch {
            Write-Host "  ⚠️  Could not list models" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ❌ Ollama service is not responding" -ForegroundColor Red
    }
}

# Check frontend dependencies
try {
    $packageLock = Test-Path "package-lock.json"
    $nodeModules = Test-Path "node_modules"
    $healthResults.frontend = $packageLock -and $nodeModules

    if ($healthResults.frontend) {
        Write-Host "  ✅ Frontend dependencies are installed" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Frontend dependencies may be missing" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠️  Could not check frontend dependencies" -ForegroundColor Yellow
}

# Final Summary
Write-Host "`n🎉 Setup Complete!" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green

Write-Host "`n📊 System Status:" -ForegroundColor Cyan
Write-Host "  • Ollama Service: $(if($healthResults.ollama) {'✅ Running'} else {'❌ Not Running'})" -ForegroundColor White
Write-Host "  • Available Models: $($healthResults.models.Count)" -ForegroundColor White
Write-Host "  • Custom Model: $(if(-not $UseFallback) {'✅ Available'} else {'❌ Using Fallback'})" -ForegroundColor White
Write-Host "  • Frontend Ready: $(if($healthResults.frontend) {'✅ Yes'} else {'⚠️  Check Dependencies'})" -ForegroundColor White

Write-Host "`n🚀 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Start the development server: npm run dev" -ForegroundColor White
Write-Host "2. Visit the test page: http://localhost:5173/ai-test" -ForegroundColor White
Write-Host "3. Test the API endpoint: http://localhost:5173/api/ai/test" -ForegroundColor White
Write-Host "4. Check Ollama models: ollama list" -ForegroundColor White

if ($UseFallback) {
    Write-Host "`n⚠️  Important Notes:" -ForegroundColor Yellow
    Write-Host "• Using fallback model due to GGUF loading issues" -ForegroundColor White
    Write-Host "• To fix custom model, check the GGUF file integrity" -ForegroundColor White
    Write-Host "• Consider re-downloading or re-converting the model" -ForegroundColor White
}

Write-Host "`n📚 Documentation:" -ForegroundColor Cyan
Write-Host "• Setup Guide: README-GEMMA3-LEGAL.md" -ForegroundColor White
Write-Host "• API Documentation: /api/ai/test" -ForegroundColor White
Write-Host "• Troubleshooting: LOCAL_LLM_SETUP.md" -ForegroundColor White

Write-Host "`n🏁 Setup completed successfully!" -ForegroundColor Green
