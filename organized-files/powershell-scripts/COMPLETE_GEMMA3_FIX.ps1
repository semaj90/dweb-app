# COMPLETE GEMMA3 INTEGRATION FIX SCRIPT
# This script fixes all known issues with Gemma3 GGUF integration

Write-Host "🚀 Starting Complete Gemma3 Integration Fix..." -ForegroundColor Cyan

# 1. VERIFY MODEL FILE
Write-Host "`n📁 Checking model files..." -ForegroundColor Yellow
$modelPath = ".\gemma3Q4_K_M\mohf16-Q4_K_M.gguf"
if (Test-Path $modelPath) {
    $fileSize = (Get-Item $modelPath).Length / 1MB
    Write-Host "✅ Model file found: $modelPath ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green
} else {
    Write-Host "❌ Model file not found: $modelPath" -ForegroundColor Red
    Write-Host "   Available files in gemma3Q4_K_M:" -ForegroundColor Yellow
    Get-ChildItem ".\gemma3Q4_K_M\" | ForEach-Object { Write-Host "   - $($_.Name)" }
    exit 1
}

# 2. STOP EXISTING OLLAMA (if running)
Write-Host "`n🛑 Stopping existing Ollama processes..." -ForegroundColor Yellow
try {
    Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
} catch {
    Write-Host "   No existing Ollama processes found" -ForegroundColor Gray
}

# 3. START OLLAMA
Write-Host "`n🔄 Starting Ollama..." -ForegroundColor Yellow
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 5

# Test Ollama connection
try {
    $ollamaTest = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method GET -TimeoutSec 10
    Write-Host "✅ Ollama is running: $($ollamaTest.version)" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to connect to Ollama on localhost:11434" -ForegroundColor Red
    Write-Host "   Please ensure Ollama is installed and try again" -ForegroundColor Yellow
    exit 1
}

# 4. CREATE/UPDATE MODEL
Write-Host "`n🤖 Creating Gemma3 model from GGUF..." -ForegroundColor Yellow
$modelfileContent = @"
# Gemma3 Legal AI Modelfile - Fixed Version
FROM .\gemma3Q4_K_M\mohf16-Q4_K_M.gguf

# Template for Gemma3 chat format
TEMPLATE """<bos><start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>"""

# Legal AI System Prompt
SYSTEM """You are a specialized Legal AI Assistant powered by Gemma 3, trained on legal documents and case law. You provide:

- Accurate legal analysis and research assistance
- Contract review and document interpretation
- Evidence evaluation and case strategy guidance
- Regulatory compliance insights
- Professional legal terminology and citations

Always maintain professional standards, cite sources when available, and clearly indicate when information is insufficient for definitive advice. Recommend consulting qualified legal professionals for critical decisions."""

# Optimized parameters for legal AI
PARAMETER temperature 0.1
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 8192
"@

Set-Content -Path ".\Modelfile-Gemma3-Legal" -Value $modelfileContent
Write-Host "✅ Updated Modelfile-Gemma3-Legal" -ForegroundColor Green

# Create the model in Ollama
Write-Host "   Creating model in Ollama..." -ForegroundColor Gray
try {
    $createResult = & ollama create gemma3-legal -f ".\Modelfile-Gemma3-Legal" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Model created successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Model creation failed: $createResult" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Error creating model: $($_.Exception.Message)" -ForegroundColor Red
}

# 5. TEST MODEL
Write-Host "`n🔍 Testing Gemma3 model..." -ForegroundColor Yellow
$listResult = & ollama list 2>&1
Write-Host "Available models:" -ForegroundColor Gray
Write-Host $listResult

# Test a simple prompt
try {
    Write-Host "   Testing with sample prompt..." -ForegroundColor Gray
    $testPrompt = "What is contract law?"
    $testResult = & ollama run gemma3-legal $testPrompt --timeout 30s 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Model responds correctly" -ForegroundColor Green
        Write-Host "   Sample response: $($testResult | Select-Object -First 1)" -ForegroundColor Gray
    } else {
        Write-Host "❌ Model test failed: $testResult" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Error testing model: $($_.Exception.Message)" -ForegroundColor Red
}

# 6. CREATE PROPER API ENDPOINT
Write-Host "`n🌐 Creating AI Chat API endpoint..." -ForegroundColor Yellow
$apiPath = ".\sveltekit-frontend\src\routes\api\ai\chat"
if (-not (Test-Path $apiPath)) {
    New-Item -ItemType Directory -Path $apiPath -Force | Out-Null
}

$apiContent = @"
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const OLLAMA_BASE_URL = 'http://localhost:11434';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { message, conversationId, settings, contextInjection } = await request.json();

    if (!message || message.trim() === '') {
      return json({ error: 'Message is required' }, { status: 400 });
    }

    // Prepare context if injection is enabled
    let systemPrompt = '';
    if (contextInjection?.enabled && contextInjection.documents?.length > 0) {
      systemPrompt = `\n\nContext Information:\n${contextInjection.documents.join('\n')}\n\nPlease use the above context to inform your response when relevant.`;
    }

    const prompt = systemPrompt ? `${systemPrompt}\n\nUser Question: ${message}` : message;

    // Call Ollama API
    const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: settings?.model || 'gemma3-legal',
        prompt,
        stream: false,
        options: {
          temperature: settings?.temperature || 0.1,
          top_p: settings?.top_p || 0.8,
          top_k: settings?.top_k || 20,
          num_ctx: settings?.contextWindow || 8192,
        },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Ollama error:', errorText);
      return json(
        { error: 'AI service unavailable', details: errorText },
        { status: 503 }
      );
    }

    const data = await response.json();

    return json({
      response: data.response,
      model: settings?.model || 'gemma3-legal',
      conversationId,
      tokensUsed: data.eval_count || 0,
      references: contextInjection?.documents || [],
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error('Chat API error:', error);
    return json(
      { error: 'Internal server error', details: error.message },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async () => {
  try {
    // Health check endpoint
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
"@

Set-Content -Path "$apiPath\+server.ts" -Value $apiContent
Write-Host "✅ Created AI Chat API endpoint" -ForegroundColor Green

# 7. UPDATE CHAT STORE ENDPOINT
Write-Host "`n🔧 Updating chat store API endpoint..." -ForegroundColor Yellow
$storeFile = ".\sveltekit-frontend\src\lib\stores\chatStore.ts"
if (Test-Path $storeFile) {
    $storeContent = Get-Content $storeFile -Raw
    $newStoreContent = $storeContent -replace 'fetch\("/api/llm/chat"', 'fetch("/api/ai/chat"'
    Set-Content -Path $storeFile -Value $newStoreContent
    Write-Host "✅ Updated chat store API endpoint" -ForegroundColor Green
} else {
    Write-Host "⚠️  Chat store file not found at expected location" -ForegroundColor Yellow
}

# 8. CREATE STREAMING ENDPOINT
Write-Host "`n🌊 Creating streaming chat endpoint..." -ForegroundColor Yellow
$streamApiContent = @"
import type { RequestHandler } from './$types';

const OLLAMA_BASE_URL = 'http://localhost:11434';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { message, settings } = await request.json();

    if (!message || message.trim() === '') {
      return new Response(JSON.stringify({ error: 'Message is required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Create a ReadableStream for Server-Sent Events
    const stream = new ReadableStream({
      async start(controller) {
        try {
          const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: settings?.model || 'gemma3-legal',
              prompt: message,
              stream: true,
              options: {
                temperature: settings?.temperature || 0.1,
                top_p: settings?.top_p || 0.8,
                top_k: settings?.top_k || 20,
                num_ctx: settings?.contextWindow || 8192,
              },
            }),
          });

          if (!response.ok) {
            throw new Error(`Ollama error: ${response.statusText}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('No response body');
          }

          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.trim()) {
                try {
                  const data = JSON.parse(line);
                  if (data.response) {
                    controller.enqueue(
                      new TextEncoder().encode(`data: ${JSON.stringify({
                        content: data.response,
                        done: data.done || false
                      })}\n\n`)
                    );
                  }
                  if (data.done) {
                    controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
                    controller.close();
                    return;
                  }
                } catch (e) {
                  console.error('Error parsing SSE data:', e);
                }
              }
            }
          }

          controller.close();
        } catch (error) {
          console.error('Streaming error:', error);
          controller.enqueue(
            new TextEncoder().encode(`data: ${JSON.stringify({
              error: error.message
            })}\n\n`)
          );
          controller.close();
        }
      }
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
      },
    });

  } catch (error) {
    console.error('Stream API error:', error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
"@

$streamPath = ".\sveltekit-frontend\src\routes\api\ai\stream"
if (-not (Test-Path $streamPath)) {
    New-Item -ItemType Directory -Path $streamPath -Force | Out-Null
}
Set-Content -Path "$streamPath\+server.ts" -Value $streamApiContent
Write-Host "✅ Created streaming chat endpoint" -ForegroundColor Green

# 9. FINAL VALIDATION
Write-Host "`n✅ Running final validation..." -ForegroundColor Yellow

# Check if Ollama is still running
try {
    $ollamaVersion = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method GET -TimeoutSec 5
    Write-Host "✅ Ollama service: RUNNING ($($ollamaVersion.version))" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama service: NOT RUNNING" -ForegroundColor Red
}

# Check model availability
$modelList = & ollama list 2>&1
if ($modelList -match "gemma3-legal") {
    Write-Host "✅ Gemma3-legal model: LOADED" -ForegroundColor Green
} else {
    Write-Host "❌ Gemma3-legal model: NOT FOUND" -ForegroundColor Red
}

# Check API endpoints
$apiFiles = @(
    ".\sveltekit-frontend\src\routes\api\ai\chat\+server.ts",
    ".\sveltekit-frontend\src\routes\api\ai\stream\+server.ts"
)

foreach ($file in $apiFiles) {
    if (Test-Path $file) {
        Write-Host "✅ API endpoint: $($file.Split('\')[-2]) - CREATED" -ForegroundColor Green
    } else {
        Write-Host "❌ API endpoint: $($file.Split('\')[-2]) - MISSING" -ForegroundColor Red
    }
}

Write-Host "`n🎉 Gemma3 Integration Fix Complete!" -ForegroundColor Cyan
Write-Host "`n📋 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Test the integration: npm run dev" -ForegroundColor White
Write-Host "   2. Navigate to your chat interface" -ForegroundColor White
Write-Host "   3. Send a test message to verify Gemma3 responses" -ForegroundColor White
Write-Host "   4. Check browser DevTools for any remaining errors" -ForegroundColor White

Write-Host "`n🔗 Test URLs:" -ForegroundColor Yellow
Write-Host "   - Chat API Health: http://localhost:5173/api/ai/chat" -ForegroundColor White
Write-Host "   - Ollama Direct: http://localhost:11434/api/version" -ForegroundColor White

Write-Host "`n💡 Troubleshooting:" -ForegroundColor Yellow
Write-Host "   - If model doesn't respond: ollama run gemma3-legal 'test'" -ForegroundColor White
Write-Host "   - If API fails: Check localhost:11434 is accessible" -ForegroundColor White
Write-Host "   - If streaming issues: Check browser network tab" -ForegroundColor White
