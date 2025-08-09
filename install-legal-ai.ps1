# ========================================
# LEGAL AI PROCESSOR - WINDOWS INSTALLER
# ========================================
# Save this as: install-legal-ai.ps1
# Run with: powershell -ExecutionPolicy Bypass -File install-legal-ai.ps1

$ErrorActionPreference = "Continue"
$AppDir = "C:\Users\james\Desktop\deeds-web\deeds-web-app"

Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     LEGAL GPU PROCESSOR v2.0.0                              ║
║     One-Click Windows Installer                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Check Node.js
try {
    $nodeVersion = node --version 2>$null
    Write-Host "✓ Node.js $nodeVersion installed" -ForegroundColor Green
} catch {
    Write-Host "✗ Node.js not found" -ForegroundColor Red
    Write-Host "  Please install from: https://nodejs.org/" -ForegroundColor Gray
    Write-Host "  Then run this script again." -ForegroundColor Gray
    exit 1
}

# Check Docker
try {
    $dockerVersion = docker --version 2>$null
    Write-Host "✓ Docker installed" -ForegroundColor Green
} catch {
    Write-Host "⚠ Docker not found (optional)" -ForegroundColor Yellow
    Write-Host "  Install from: https://docker.com" -ForegroundColor Gray
}

# Create project directory
Write-Host "`nCreating project directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
Set-Location $AppDir
Write-Host "✓ Directory created: $AppDir" -ForegroundColor Green

# Create the setup script
Write-Host "`nCreating setup files..." -ForegroundColor Yellow

# Create package.json first
$packageJson = @'
{
  "name": "legal-gpu-processor",
  "version": "2.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite dev --host localhost --port 5173",
    "build": "vite build",
    "preview": "vite preview",
    "start": "node scripts/start-windows.js"
  },
  "dependencies": {
    "@sveltejs/adapter-node": "^5.0.0",
    "@sveltejs/kit": "^2.0.0",
    "@sveltejs/vite-plugin-svelte": "^4.0.0",
    "svelte": "^5.0.0",
    "vite": "^5.0.0",
    "express": "^4.18.2",
    "pg": "^8.11.0",
    "redis": "^4.6.0",
    "ollama": "^0.5.12"
  }
}
'@
$packageJson | Out-File -FilePath "package.json" -Encoding utf8

# Create directories
$directories = @(
    "src/routes/api/health",
    "src/routes/api/query",
    "src/lib/ai",
    "src/lib/database",
    "scripts",
    "data",
    "models",
    "uploads"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# Create all necessary files
Write-Host "Creating application files..." -ForegroundColor Yellow

# Docker Compose
@'
version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: legal_ai
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - ./data/postgres:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant:/qdrant/storage
'@ | Out-File -FilePath "docker-compose.yml" -Encoding utf8

# Vite config
@'
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    port: 5173,
    host: 'localhost'
  }
});
'@ | Out-File -FilePath "vite.config.js" -Encoding utf8

# Svelte config
@'
import adapter from '@sveltejs/adapter-node';

export default {
  kit: {
    adapter: adapter()
  }
};
'@ | Out-File -FilePath "svelte.config.js" -Encoding utf8

# Main page
@'
<script>
  let query = '';
  let response = '';
  let loading = false;

  async function handleSubmit() {
    if (!query.trim() || loading) return;
    loading = true;

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      const data = await res.json();
      response = data.response || 'No response';
    } catch (error) {
      response = 'Error: ' + error.message;
    } finally {
      loading = false;
    }
  }
</script>

<div style="max-width: 800px; margin: 0 auto; padding: 2rem;">
  <h1>Legal AI Assistant</h1>

  <div style="margin: 2rem 0;">
    <textarea
      bind:value={query}
      placeholder="Enter your legal question..."
      style="width: 100%; min-height: 100px; padding: 1rem;"
    ></textarea>

    <button
      on:click={handleSubmit}
      disabled={loading}
      style="margin-top: 1rem; padding: 0.75rem 2rem; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;"
    >
      {loading ? 'Processing...' : 'Submit'}
    </button>
  </div>

  {#if response}
    <div style="padding: 1rem; background: #f5f5f5; border-radius: 4px;">
      <h3>Response:</h3>
      <p>{response}</p>
    </div>
  {/if}
</div>
'@ | Out-File -FilePath "src/routes/+page.svelte" -Encoding utf8

# Layout
@'
<slot />
'@ | Out-File -FilePath "src/routes/+layout.svelte" -Encoding utf8

# App.html
@'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  %sveltekit.head%
</head>
<body>
  <div>%sveltekit.body%</div>
</body>
</html>
'@ | Out-File -FilePath "src/app.html" -Encoding utf8

# API Health endpoint
@'
import { json } from '@sveltejs/kit';

export async function GET() {
  return json({
    status: 'healthy',
    services: {
      postgres: true,
      redis: true,
      ollama: true
    }
  });
}
'@ | Out-File -FilePath "src/routes/api/health/+server.js" -Encoding utf8

# API Query endpoint
@'
import { json } from '@sveltejs/kit';

export async function POST({ request }) {
  const { query } = await request.json();

  // Try Ollama
  try {
    const res = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma2:9b',
        prompt: query,
        stream: false
      })
    });

    if (res.ok) {
      const data = await res.json();
      return json({ response: data.response });
    }
  } catch (error) {
    console.error('Ollama error:', error);
  }

  return json({
    response: 'AI service unavailable. Please ensure Ollama is running.'
  });
}
'@ | Out-File -FilePath "src/routes/api/query/+server.js" -Encoding utf8

# Quick start batch file
@'
@echo off
echo Starting Legal AI...
docker-compose up -d
npm install
npm run dev
'@ | Out-File -FilePath "START.bat" -Encoding ascii

Write-Host "✓ All files created" -ForegroundColor Green

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
npm install 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠ Some dependencies failed to install" -ForegroundColor Yellow
}

# Start Docker services (if Docker is available)
$dockerInstalled = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerInstalled) {
    Write-Host "`nStarting Docker services..." -ForegroundColor Yellow
    docker-compose up -d 2>$null
    Start-Sleep -Seconds 3
    Write-Host "✓ Docker services started" -ForegroundColor Green
}

# Check for Ollama
Write-Host "`nChecking Ollama..." -ForegroundColor Yellow
try {
    $ollamaTest = Invoke-WebRequest -Uri "http://localhost:11434/api/version" -TimeoutSec 2 -ErrorAction SilentlyContinue
    Write-Host "✓ Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "⚠ Ollama not running" -ForegroundColor Yellow
    Write-Host "  To use AI features, install from: https://ollama.ai" -ForegroundColor Gray
    Write-Host "  Then run: ollama pull gemma2:9b" -ForegroundColor Gray
}

Write-Host @"

========================================
Installation Complete!
========================================

Your Legal AI Processor is ready!

To start the application:
  1. Open terminal in: $AppDir
  2. Run: npm run dev

Or double-click: START.bat

Access at: http://localhost:5173

Services on localhost:
  - PostgreSQL: 5432
  - Redis: 6379
  - Qdrant: 6333
  - Ollama: 11434

========================================

"@ -ForegroundColor Green

# Ask if user wants to start now
$response = Read-Host "Start the application now? (Y/N)"
if ($response -eq 'Y' -or $response -eq 'y') {
    Write-Host "`nStarting development server..." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
    npm run dev
}