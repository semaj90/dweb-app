#!/usr/bin/env node

/**
 * 🌐 Legal AI Platform - Complete Environment Setup
 * Windows Native Setup with GPU Acceleration
 */

import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import { execSync } from 'child_process';

console.log('🚀 Legal AI Platform - Environment Setup\n');

// Environment configuration
const envConfig = {
  // Database
  DATABASE_URL: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
  DATABASE_HOST: 'localhost',
  DATABASE_PORT: '5432',
  DATABASE_NAME: 'legal_ai_db',
  DATABASE_USER: 'legal_admin',
  DATABASE_PASSWORD: '123456',
  PGPASSWORD: '123456',

  // Frontend & API
  FRONTEND_PORT: '5173',
  PROXY_PORT: '5180',
  API_PORT: '8080',
  API_URL: 'http://localhost:8080',
  FRONTEND_URL: 'http://localhost:5173',
  PROXY_URL: 'http://localhost:5180',

  // Redis
  REDIS_URL: 'redis://localhost:6379',
  REDIS_HOST: 'localhost',
  REDIS_PORT: '6379',

  // AI & Ollama
  OLLAMA_ENDPOINT: 'http://localhost:11434',
  OLLAMA_API_URL: 'http://localhost:11434/api',
  GEMMA3_MODEL_PATH: 'C:\\Users\\james\\.ollama\\models\\gemma3-legal',

  // Microservices
  UPLOAD_SERVICE_PORT: '8093',
  RAG_SERVICE_PORT: '8094',
  QUIC_GATEWAY_PORT: '8447',
  CLUSTER_MANAGER_PORT: '3099',
  LOAD_BALANCER_PORT: '8099',

  // GPU & CUDA
  GPU_ENABLED: 'true',
  CUDA_PATH: 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0',
  CGO_ENABLED: '1',
  ENABLE_GPU: 'true',

  // File paths
  UPLOADS_DIR: './uploads',
  DOCUMENTS_DIR: './documents',
  EVIDENCE_DIR: './evidence',
  LOGS_DIR: './logs',
  GENERATED_REPORTS_DIR: './generated_reports',

  // Security
  JWT_SECRET: 'your-super-secret-jwt-key-here-change-in-production',
  SESSION_SECRET: 'your-session-secret-key-here',

  // Environment
  NODE_ENV: 'development',
  DEBUG: 'true',
  LOG_LEVEL: 'info'
};

// Create directories
const directories = [
  'uploads',
  'documents', 
  'evidence',
  'logs',
  'generated_reports',
  'temp'
];

console.log('📁 Creating directories...');
directories.forEach(dir => {
  const fullPath = join(process.cwd(), dir);
  if (!existsSync(fullPath)) {
    mkdirSync(fullPath, { recursive: true });
    console.log(`✅ Created: ${dir}`);
  } else {
    console.log(`✅ Exists: ${dir}`);
  }
});

// Create .env file
console.log('\n📝 Creating .env file...');
const envContent = Object.entries(envConfig)
  .map(([key, value]) => `${key}=${value}`)
  .join('\n');

const envPath = join(process.cwd(), '.env');
writeFileSync(envPath, envContent);
console.log('✅ .env file created');

// Create environment loader script
console.log('\n📝 Creating environment loader...');
const envLoaderContent = `#!/usr/bin/env node

/**
 * Environment Loader for Legal AI Platform
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = join(__filename, '..');

// Load .env file
const envPath = join(__dirname, '.env');
try {
  const envContent = readFileSync(envPath, 'utf8');
  const envVars = envContent.split('\\n').filter(line => line.trim() && !line.startsWith('#'));
  
  envVars.forEach(line => {
    const [key, ...valueParts] = line.split('=');
    const value = valueParts.join('=');
    if (key && value) {
      process.env[key.trim()] = value.trim();
    }
  });
  
  console.log('✅ Environment variables loaded');
} catch (error) {
  console.log('⚠️  No .env file found, using defaults');
}

export default process.env;
`;

const envLoaderPath = join(process.cwd(), 'env-loader.mjs');
writeFileSync(envLoaderPath, envLoaderContent);
console.log('✅ Environment loader created');

// Create batch file for Windows
console.log('\n📝 Creating Windows batch file...');
const batchContent = `@echo off
echo 🌐 Legal AI Platform - Environment Setup
echo ======================================
echo.
echo Loading environment variables...
echo.

REM Load .env file
for /f "tokens=1,2 delims==" %%a in (.env) do (
  if not "%%a"=="" (
    if not "%%a:~0,1%"=="#" (
      set "%%a=%%b"
      echo Set: %%a
    )
  )
)

echo.
echo ✅ Environment variables loaded
echo.
echo 🚀 Starting services...
echo.

REM Start frontend
cd sveltekit-frontend
start "Frontend (Port 5173)" cmd /k "npm run dev"

REM Wait for frontend to start
timeout /t 10 /nobreak >nul

REM Start proxy
cd ..
start "Proxy (Port 5180)" cmd /k "node start-port-5180.cjs"

echo.
echo 🌐 Frontend: http://localhost:5173
echo 🌐 Port 5180: http://localhost:5180
echo.
echo 🎉 Setup complete!
pause
`;

const batchPath = join(process.cwd(), 'start-with-env.bat');
writeFileSync(batchPath, batchContent);
console.log('✅ Windows batch file created');

// Create PowerShell script
console.log('\n📝 Creating PowerShell script...');
const psContent = `# Legal AI Platform - Environment Setup (PowerShell)

Write-Host "🌐 Legal AI Platform - Environment Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Load .env file
if (Test-Path ".env") {
    Write-Host "📝 Loading environment variables..." -ForegroundColor Yellow
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^#][^=]+)=(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
            Write-Host "Set: $key" -ForegroundColor Green
        }
    }
    Write-Host "✅ Environment variables loaded" -ForegroundColor Green
} else {
    Write-Host "⚠️  No .env file found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Starting services..." -ForegroundColor Yellow
Write-Host ""

# Start frontend
Set-Location "sveltekit-frontend"
Start-Process -FilePath "cmd" -ArgumentList "/k", "npm run dev" -WindowStyle Normal -Name "Frontend (Port 5173)"

# Wait for frontend to start
Start-Sleep -Seconds 10

# Start proxy
Set-Location ".."
Start-Process -FilePath "cmd" -ArgumentList "/k", "node start-port-5180.cjs" -WindowStyle Normal -Name "Proxy (Port 5180)"

Write-Host ""
Write-Host "🌐 Frontend: http://localhost:5173" -ForegroundColor Green
Write-Host "🌐 Port 5180: http://localhost:5180" -ForegroundColor Green
Write-Host ""
Write-Host "🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to continue"
`;

const psPath = join(process.cwd(), 'start-with-env.ps1');
writeFileSync(psPath, psContent);
console.log('✅ PowerShell script created');

console.log('\n🎉 Environment setup complete!');
console.log('\n📋 Files created:');
console.log('   • .env - Environment variables');
console.log('   • env-loader.mjs - Node.js environment loader');
console.log('   • start-with-env.bat - Windows batch file');
console.log('   • start-with-env.ps1 - PowerShell script');
console.log('\n🚀 To start with environment variables:');
console.log('   • Windows: start-with-env.bat');
console.log('   • PowerShell: .\\start-with-env.ps1');
console.log('   • Node.js: import "./env-loader.mjs"');
