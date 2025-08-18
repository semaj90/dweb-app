# 🚀 NATIVE WINDOWS DEPLOYMENT - Complete Fix & Integration Plan

## Current Status Analysis

### ✅ What's Working
- **Go microservices** with GPU acceleration and SOM clustering
- **SvelteKit 2 frontend** with YoRHa theme
- **Ollama** serving AI models locally
- **Infrastructure components** ready (but need native setup)

### ❌ Critical Blockers to Fix
1. **10 TypeScript errors** preventing compilation
2. **Service failures** (HTTP 500, Go APIs on 8084/8085 not responding)
3. **API routing error** (404 for /api/enhanced-document-ingestion)
4. **YoRHa interface** not integrated as homepage
5. **Database CRUD** operations failing

## PHASE 1: Fix Critical Errors (Do This First!)

### Step 1.1: Fix TypeScript Errors
```powershell
# Create automatic TypeScript fixer
Write-Host "Fixing TypeScript errors..." -ForegroundColor Cyan

# Common fixes for your errors:
# 1. Module resolution failures - update tsconfig.json
# 2. Type mismatches - add proper type definitions
# 3. Redeclared variables - rename or remove duplicates
```

### Step 1.2: Fix API Route 404 Error
The issue: You're calling `/api/enhanced-document-ingestion` but it doesn't exist.
The solution: Your actual endpoint is `/api/enhanced-rag/+server.ts`

**Quick Fix:**
```typescript
// In your frontend components, update all API calls:
// WRONG:
fetch('/api/enhanced-document-ingestion', ...)

// CORRECT:
fetch('/api/enhanced-rag', {
  method: 'POST',
  body: JSON.stringify({ 
    action: 'ingest',
    ...data 
  })
})
```

### Step 1.3: Fix Service Startup Failures
```powershell
# Check what's blocking ports
netstat -ano | findstr :8084
netstat -ano | findstr :8085
netstat -ano | findstr :3000

# Kill conflicting processes
taskkill /F /PID [PID_NUMBER]
```

## PHASE 2: Native Windows Services Setup (No Docker!)

### Step 2.1: PostgreSQL Native
```powershell
# Download PostgreSQL Windows installer
Invoke-WebRequest -Uri "https://get.enterprisedb.com/postgresql/postgresql-15.4-1-windows-x64.exe" -OutFile "postgresql-installer.exe"

# Install silently with pgvector
.\postgresql-installer.exe --mode unattended --unattendedmodeui none --prefix "C:\PostgreSQL\15" --serverport 5432 --superpassword postgres --servicename PostgreSQL

# Install pgvector extension manually
cd C:\PostgreSQL\15\bin
psql -U postgres -c "CREATE DATABASE legal_ai_db;"
psql -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Step 2.2: Redis Native Windows
```powershell
# Download Redis for Windows
Invoke-WebRequest -Uri "https://github.com/microsoftarchive/redis/releases/download/win-3.2.100/Redis-x64-3.2.100.msi" -OutFile "Redis-Windows.msi"

# Install Redis
msiexec /i Redis-Windows.msi /quiet

# Start Redis service
net start Redis
```

### Step 2.3: Neo4j Native Windows
```powershell
# Download Neo4j Community
Invoke-WebRequest -Uri "https://neo4j.com/artifact.php?name=neo4j-community-5.23.0-windows.zip" -OutFile "neo4j.zip"

# Extract and install
Expand-Archive -Path neo4j.zip -DestinationPath C:\neo4j
cd C:\neo4j\neo4j-community-5.23.0\bin

# Install as Windows service
.\neo4j.bat install-service
.\neo4j.bat start
```

### Step 2.4: MinIO Native Windows
```powershell
# Download MinIO
Invoke-WebRequest -Uri "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" -OutFile "C:\minio\minio.exe"

# Create service wrapper
New-Service -Name "MinIO" -BinaryPathName "C:\minio\minio.exe server C:\minio\data --console-address :9001" -DisplayName "MinIO Object Storage" -StartupType Automatic

# Start MinIO
Start-Service MinIO
```

### Step 2.5: RabbitMQ Native Windows
```powershell
# Download Erlang (required for RabbitMQ)
Invoke-WebRequest -Uri "https://github.com/erlang/otp/releases/download/OTP-26.0/otp_win64_26.0.exe" -OutFile "erlang.exe"

# Download RabbitMQ
Invoke-WebRequest -Uri "https://github.com/rabbitmq/rabbitmq-server/releases/download/v3.12.14/rabbitmq-server-3.12.14.exe" -OutFile "rabbitmq.exe"

# Install both
.\erlang.exe /S
.\rabbitmq.exe /S

# Enable management plugin
& "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.12.14\sbin\rabbitmq-plugins.bat" enable rabbitmq_management
```

## PHASE 3: Integrate YoRHa Interface

### Step 3.1: Make YoRHa the Homepage
```typescript
// Move YoRHa to main route
// 1. Backup current homepage
// src/routes/+page.svelte.backup

// 2. Replace with YoRHa
// Copy: src/routes/yorha-dashboard/+page.svelte
// To: src/routes/+page.svelte
```

### Step 3.2: Create Unified Navigation
```svelte
<!-- src/lib/components/YoRHaSidebar.svelte -->
<script lang="ts">
  const sections = {
    'Core Features': [
      { name: 'Dashboard', href: '/', icon: '🏠' },
      { name: 'Cases', href: '/cases', icon: '📁' },
      { name: 'Evidence', href: '/evidence', icon: '📎' },
      { name: 'AI Assistant', href: '/ai-assistant', icon: '🤖' },
      { name: 'Enhanced RAG', href: '/rag', icon: '🧠' }
    ],
    'Analysis Tools': [
      { name: 'Search', href: '/search', icon: '🔍' },
      { name: 'Analytics', href: '/analytics', icon: '📊' },
      { name: 'Reports', href: '/reports', icon: '📈' }
    ],
    'Demo Pages': [
      // All 25+ demo pages here
    ],
    'Admin': [
      { name: 'Settings', href: '/admin/settings', icon: '⚙️' },
      { name: 'Users', href: '/admin/users', icon: '👥' }
    ]
  };
</script>

<aside class="yorha-sidebar">
  {#each Object.entries(sections) as [section, pages]}
    <div class="section">
      <h3>{section}</h3>
      {#each pages as page}
        <a href={page.href}>
          <span class="icon">{page.icon}</span>
          {page.name}
        </a>
      {/each}
    </div>
  {/each}
</aside>
```

## PHASE 4: Fix Database CRUD Operations

### Step 4.1: Fix Drizzle ORM Connection
```typescript
// src/lib/server/db.ts
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';

const connectionString = 'postgresql://postgres:postgres@localhost:5432/legal_ai_db?sslmode=disable';
const sql = postgres(connectionString);
export const db = drizzle(sql);
```

### Step 4.2: Fix CRUD Operations
```typescript
// src/routes/api/cases/+server.ts
import { db } from '$lib/server/db';
import { cases } from '$lib/server/schema';

export async function POST({ request }) {
  try {
    const data = await request.json();
    
    // Fix: Ensure proper await and error handling
    const result = await db.insert(cases).values(data).returning();
    
    return new Response(JSON.stringify(result[0]), {
      status: 201,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Database error:', error);
    return new Response(JSON.stringify({ error: 'Database operation failed' }), {
      status: 500
    });
  }
}
```

## PHASE 5: Complete Native Windows Startup Script

```powershell
# Save as: START-NATIVE-WINDOWS-COMPLETE.ps1

Write-Host "🚀 Starting Legal AI Platform (Native Windows)" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Function to check if service is running
function Test-ServiceRunning {
    param($ServiceName, $ProcessName, $Port)
    
    if ($ProcessName) {
        $process = Get-Process $ProcessName -ErrorAction SilentlyContinue
        if ($process) { return $true }
    }
    
    if ($ServiceName) {
        $service = Get-Service $ServiceName -ErrorAction SilentlyContinue
        if ($service.Status -eq 'Running') { return $true }
    }
    
    if ($Port) {
        $connection = Test-NetConnection -ComputerName localhost -Port $Port -ErrorAction SilentlyContinue
        if ($connection.TcpTestSucceeded) { return $true }
    }
    
    return $false
}

# Start PostgreSQL
Write-Host "Starting PostgreSQL..." -ForegroundColor Cyan
if (!(Test-ServiceRunning -ServiceName "PostgreSQL" -Port 5432)) {
    Start-Service PostgreSQL
    Start-Sleep -Seconds 3
}
Write-Host "✓ PostgreSQL running" -ForegroundColor Green

# Start Redis
Write-Host "Starting Redis..." -ForegroundColor Cyan
if (!(Test-ServiceRunning -ServiceName "Redis" -Port 6379)) {
    Start-Service Redis
    Start-Sleep -Seconds 2
}
Write-Host "✓ Redis running" -ForegroundColor Green

# Start Neo4j
Write-Host "Starting Neo4j..." -ForegroundColor Cyan
if (!(Test-ServiceRunning -Port 7474)) {
    & "C:\neo4j\neo4j-community-5.23.0\bin\neo4j.bat" start
    Start-Sleep -Seconds 5
}
Write-Host "✓ Neo4j running" -ForegroundColor Green

# Start MinIO
Write-Host "Starting MinIO..." -ForegroundColor Cyan
if (!(Test-ServiceRunning -ServiceName "MinIO" -Port 9000)) {
    Start-Service MinIO
    Start-Sleep -Seconds 2
}
Write-Host "✓ MinIO running" -ForegroundColor Green

# Start RabbitMQ
Write-Host "Starting RabbitMQ..." -ForegroundColor Cyan
if (!(Test-ServiceRunning -ServiceName "RabbitMQ" -Port 5672)) {
    Start-Service RabbitMQ
    Start-Sleep -Seconds 3
}
Write-Host "✓ RabbitMQ running" -ForegroundColor Green

# Start Ollama
Write-Host "Starting Ollama..." -ForegroundColor Cyan
if (!(Test-ServiceRunning -ProcessName "ollama" -Port 11434)) {
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}
Write-Host "✓ Ollama running" -ForegroundColor Green

# Start Go Services
Write-Host "Starting Go microservices..." -ForegroundColor Cyan

# GPU Orchestrator (8084)
if (!(Test-ServiceRunning -Port 8084)) {
    Start-Process -FilePath "go" -ArgumentList "run", "gpu-orchestrator.go" -WorkingDirectory "." -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# RAG Service (8085)
if (!(Test-ServiceRunning -Port 8085)) {
    Start-Process -FilePath "go" -ArgumentList "run", "enhanced-rag-service.go" -WorkingDirectory "." -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

Write-Host "✓ Go services running" -ForegroundColor Green

# Start SvelteKit
Write-Host "Starting SvelteKit application..." -ForegroundColor Cyan
npm run dev

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "✅ All services running natively!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access points:" -ForegroundColor Cyan
Write-Host "  • Application: http://localhost:3000" -ForegroundColor White
Write-Host "  • PostgreSQL: localhost:5432" -ForegroundColor White
Write-Host "  • Redis: localhost:6379" -ForegroundColor White
Write-Host "  • Neo4j: http://localhost:7474" -ForegroundColor White
Write-Host "  • MinIO: http://localhost:9001" -ForegroundColor White
Write-Host "  • RabbitMQ: http://localhost:15672" -ForegroundColor White
Write-Host "  • Ollama: http://localhost:11434" -ForegroundColor White
Write-Host ""
```

## Immediate Action Items

### 1. Fix TypeScript Errors NOW
```bash
# Run this to see exact errors
npm run check

# Fix each error systematically
# Most common fixes:
# - Add missing type imports
# - Remove duplicate declarations
# - Update import paths
```

### 2. Fix API Route
```bash
# Global find and replace:
# Find: /api/enhanced-document-ingestion
# Replace: /api/enhanced-rag
```

### 3. Test Database Connection
```powershell
# Test PostgreSQL is accessible
psql -U postgres -d legal_ai_db -c "SELECT 1;"
```

### 4. Replace Homepage with YoRHa
```bash
# Backup and replace
cp src/routes/+page.svelte src/routes/+page.svelte.backup
cp src/routes/yorha-dashboard/+page.svelte src/routes/+page.svelte
```

This native Windows approach eliminates Docker completely and gives you full control over each service. Start with Phase 1 to unblock development, then proceed through each phase systematically.
