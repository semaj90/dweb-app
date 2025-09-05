# GEMMA3 INTEGRATION FIX - SIMPLIFIED VERSION
Write-Host "Starting Gemma3 Integration Fix..." -ForegroundColor Cyan

# 1. CHECK MODEL FILE
Write-Host "`nChecking model file..." -ForegroundColor Yellow
$modelPath = ".\gemma3Q4_K_M\mohf16-Q4_K_M.gguf"
if (Test-Path $modelPath) {
    $fileSize = (Get-Item $modelPath).Length / 1MB
    Write-Host "Model file found: $modelPath ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green
} else {
    Write-Host "Model file not found: $modelPath" -ForegroundColor Red
    Write-Host "Available files:" -ForegroundColor Yellow
    Get-ChildItem ".\gemma3Q4_K_M\" | ForEach-Object { Write-Host "  - $($_.Name)" }
    exit 1
}

# 2. START OLLAMA (if not running)
Write-Host "`nStarting Ollama..." -ForegroundColor Yellow
try {
    Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
} catch {
    # Process not running, that's fine
}

Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 5

# Test Ollama connection
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method GET -TimeoutSec 10
    Write-Host "Ollama is running: $($response.version)" -ForegroundColor Green
} catch {
    Write-Host "Failed to connect to Ollama" -ForegroundColor Red
    exit 1
}

# 3. CREATE MODEL
Write-Host "`nCreating Gemma3 model..." -ForegroundColor Yellow
$modelfileContent = @"
FROM .\gemma3Q4_K_M\mohf16-Q4_K_M.gguf

TEMPLATE """<bos><start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>"""

SYSTEM """You are a specialized Legal AI Assistant powered by Gemma 3. You provide accurate legal analysis, contract review, evidence evaluation, and regulatory compliance insights. Always maintain professional standards and recommend consulting qualified legal professionals for critical decisions."""

PARAMETER temperature 0.1
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 8192
"@

Set-Content -Path ".\Modelfile-Gemma3-Legal" -Value $modelfileContent
Write-Host "Updated Modelfile-Gemma3-Legal" -ForegroundColor Green

# Create the model
try {
    & ollama create gemma3-legal -f ".\Modelfile-Gemma3-Legal"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Model created successfully" -ForegroundColor Green
    } else {
        Write-Host "Model creation failed" -ForegroundColor Red
    }
} catch {
    Write-Host "Error creating model: $($_.Exception.Message)" -ForegroundColor Red
}

# 4. TEST MODEL
Write-Host "`nTesting model..." -ForegroundColor Yellow
$models = & ollama list
Write-Host $models

# 5. CREATE API ENDPOINTS
Write-Host "`nCreating API endpoints..." -ForegroundColor Yellow

# Create API directory
$apiPath = ".\sveltekit-frontend\src\routes\api\ai\chat"
if (-not (Test-Path $apiPath)) {
    New-Item -ItemType Directory -Path $apiPath -Force | Out-Null
}

# API content
$apiContent = @'
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const OLLAMA_BASE_URL = 'http://localhost:11434';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { message, conversationId, settings, contextInjection } = await request.json();

    if (!message || message.trim() === '') {
      return json({ error: 'Message is required' }, { status: 400 });
    }

    const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: settings?.model || 'gemma3-legal',
        prompt: message,
        stream: false
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return json({ error: 'AI service unavailable', details: errorText }, { status: 503 });
    }

    const data = await response.json();

    return json({
      response: data.response,
      model: 'gemma3-legal',
      conversationId,
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error('Chat API error:', error);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
};

export const GET: RequestHandler = async () => {
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/version`);
    const isHealthy = response.ok;

    return json({
      status: isHealthy ? 'healthy' : 'unhealthy',
      service: 'ollama',
      model: 'gemma3-legal',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return json({
      status: 'error',
      error: error.message,
      timestamp: new Date().toISOString(),
    }, { status: 503 });
  }
};
'@

Set-Content -Path "$apiPath\+server.ts" -Value $apiContent
Write-Host "Created AI Chat API endpoint" -ForegroundColor Green

Write-Host "`nGemma3 Integration Fix Complete!" -ForegroundColor Green
Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "1. Test: ollama run gemma3-legal 'What is contract law?'" -ForegroundColor White
Write-Host "2. Run: npm run dev" -ForegroundColor White
Write-Host "3. Navigate to: http://localhost:5173/test-gemma3" -ForegroundColor White
