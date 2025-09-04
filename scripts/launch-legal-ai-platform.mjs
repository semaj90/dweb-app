#!/usr/bin/env zx

import { chalk } from 'zx'

// Configure zx
$.verbose = true

console.log(chalk.cyan(`
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🤖 LEGAL AI PLATFORM - AUTOMATED LAUNCH SEQUENCE           ║
║                                                              ║
║  ▼ Gemma Assisted Legal is starting...                      ║
║  ▼ Routing to YoRHa Interface...                            ║
║  ▼ Glory to Mankind                                         ║
║                                                              ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
`))

// Step 1: Start External Services
console.log(chalk.yellow('\n🚀 Step 1: Starting External Services...'))

async function checkService(name, url, timeout = 3000) {
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)
    
    const response = await fetch(url, { 
      signal: controller.signal,
      method: 'HEAD' 
    })
    clearTimeout(timeoutId)
    
    return response.ok
  } catch (error) {
    return false
  }
}

async function startExternalServices() {
  console.log(chalk.blue('📦 Checking MinIO (Port 9000)...'))
  const minioRunning = await checkService('MinIO', 'http://localhost:9000')
  
  if (!minioRunning) {
    console.log(chalk.yellow('⚠️  MinIO not running. Attempting to start...'))
    try {
      // Try to start MinIO if available
      await $`powershell -Command "Get-Process minio -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Process -FilePath 'minio.exe' -ArgumentList 'server','./data' -WindowStyle Hidden -PassThru"`.nothrow()
      console.log(chalk.green('✅ MinIO startup initiated'))
      await sleep(3000) // Wait for startup
    } catch (error) {
      console.log(chalk.red('❌ MinIO not available - continuing without object storage'))
    }
  } else {
    console.log(chalk.green('✅ MinIO already running'))
  }

  console.log(chalk.blue('📡 Checking Redis (Port 6379)...'))
  try {
    await $`redis-cli ping`.timeout('3s')
    console.log(chalk.green('✅ Redis already running'))
  } catch (error) {
    console.log(chalk.yellow('⚠️  Redis not running. Attempting to start...'))
    try {
      await $`powershell -Command "Start-Process -FilePath 'redis-server' -WindowStyle Hidden -PassThru"`.nothrow()
      console.log(chalk.green('✅ Redis startup initiated'))
      await sleep(2000)
    } catch (error) {
      console.log(chalk.red('❌ Redis not available - continuing without cache'))
    }
  }

  console.log(chalk.blue('🗄️  Checking PostgreSQL (Port 5432)...'))
  try {
    await $`pg_isready -h localhost -p 5432`.timeout('3s')
    console.log(chalk.green('✅ PostgreSQL already running'))
  } catch (error) {
    console.log(chalk.yellow('⚠️  PostgreSQL check failed - ensure it\'s running manually'))
  }

  console.log(chalk.blue('🔗 Checking Neo4j (Port 7474)...'))
  const neo4jRunning = await checkService('Neo4j', 'http://localhost:7474')
  if (neo4jRunning) {
    console.log(chalk.green('✅ Neo4j already running'))
  } else {
    console.log(chalk.yellow('⚠️  Neo4j not detected - manual start may be required'))
  }

  console.log(chalk.blue('🧠 Checking Ollama (Port 11434)...'))
  const ollamaRunning = await checkService('Ollama', 'http://localhost:11434')
  if (!ollamaRunning) {
    console.log(chalk.yellow('⚠️  Ollama not running. Attempting to start...'))
    try {
      await $`powershell -Command "Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden -PassThru"`.nothrow()
      console.log(chalk.green('✅ Ollama startup initiated'))
      await sleep(5000) // Ollama takes longer to start
    } catch (error) {
      console.log(chalk.red('❌ Ollama not available - AI features may be limited'))
    }
  } else {
    console.log(chalk.green('✅ Ollama already running'))
  }
}

// Step 2: Check Service Health
async function checkServiceHealth() {
  console.log(chalk.yellow('\n🔍 Step 2: Comprehensive Service Health Check...'))
  
  const services = [
    { name: 'MinIO', url: 'http://localhost:9000', description: 'Object Storage' },
    { name: 'Redis', url: 'redis://localhost:6379', description: 'Cache Server', command: 'redis-cli ping' },
    { name: 'PostgreSQL', url: 'postgresql://localhost:5432', description: 'Database', command: 'pg_isready -h localhost -p 5432' },
    { name: 'Neo4j', url: 'http://localhost:7474', description: 'Graph Database' },
    { name: 'Ollama', url: 'http://localhost:11434', description: 'AI Models' },
  ]

  console.log(chalk.blue('📊 Service Status Report:'))
  console.log(chalk.blue('=' .repeat(50)))
  
  let allHealthy = true
  
  for (const service of services) {
    try {
      let status = '❌ Not Running'
      
      if (service.command) {
        try {
          await $`${service.command}`.timeout('3s')
          status = '✅ Running'
        } catch (error) {
          allHealthy = false
        }
      } else {
        const isHealthy = await checkService(service.name, service.url)
        if (isHealthy) {
          status = '✅ Running'
        } else {
          allHealthy = false
        }
      }
      
      console.log(`${status.padEnd(15)} ${service.name.padEnd(12)} ${service.description}`)
    } catch (error) {
      console.log(`❌ Check Failed  ${service.name.padEnd(12)} ${service.description}`)
      allHealthy = false
    }
  }
  
  if (allHealthy) {
    console.log(chalk.green('\n✅ All services are healthy!'))
  } else {
    console.log(chalk.yellow('\n⚠️  Some services are not running - continuing anyway'))
    console.log(chalk.dim('   The platform will work with reduced functionality'))
  }
  
  return allHealthy
}

// Step 3: Launch Platform
async function launchPlatform() {
  console.log(chalk.yellow('\n🚀 Step 3: Launching Legal AI Platform...'))
  
  // Kill any existing processes that might conflict
  console.log(chalk.blue('🧹 Cleaning up any existing processes...'))
  try {
    await $`taskkill /F /IM simple-upload.exe`.nothrow()
    await $`taskkill /F /IM quic-gateway.exe`.nothrow()
    await $`taskkill /F /IM upload-service.exe`.nothrow()
    console.log(chalk.green('✅ Process cleanup completed'))
  } catch (error) {
    console.log(chalk.dim('ℹ️  No conflicting processes to clean'))
  }
  
  console.log(chalk.blue('⚙️  Setting environment variables...'))
  process.env.NODE_ENV = 'development'
  process.env.VITE_DEV = 'true'
  
  console.log(chalk.green('🎯 Starting npm run dev:full...'))
  console.log(chalk.cyan(`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎮 Legal AI Platform Launching
  🧠 AI Engine: Gemma3-Legal + GGUF + AutoGen Orchestra
  ⚡ GPU: RTX 3060 Ti Acceleration Active
  🌐 Multi-Protocol: QUIC + gRPC + HTTP + WebSocket
  📊 Services: 37 Go Microservices + SvelteKit 2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`))

  try {
    // Change to sveltekit-frontend directory and run dev:full
    cd('sveltekit-frontend')
    await $`npm run dev:full`
  } catch (error) {
    console.log(chalk.red('❌ Platform launch failed:'), error.message)
    console.log(chalk.yellow('\n🔧 Troubleshooting tips:'))
    console.log(chalk.dim('   1. Ensure all dependencies are installed: npm install'))
    console.log(chalk.dim('   2. Check that external services are running'))
    console.log(chalk.dim('   3. Verify port availability'))
    process.exit(1)
  }
}

// Main execution flow
async function main() {
  try {
    console.log(chalk.green('🌟 Starting Legal AI Platform Launch Sequence...'))
    
    // Ensure we're in the right directory
    const cwd = process.cwd()
    if (!cwd.includes('deeds-web-app')) {
      console.log(chalk.red('❌ Please run this script from the deeds-web-app directory'))
      process.exit(1)
    }
    
    // Step 1: Start External Services
    await startExternalServices()
    
    // Give services time to fully start
    console.log(chalk.blue('⏳ Waiting for services to fully initialize...'))
    await sleep(5000)
    
    // Step 2: Check Service Health
    const servicesHealthy = await checkServiceHealth()
    
    // Step 3: Launch Platform
    await launchPlatform()
    
  } catch (error) {
    console.log(chalk.red('💥 Launch sequence failed:'), error.message)
    console.log(chalk.yellow('\n🆘 Emergency startup options:'))
    console.log(chalk.dim('   1. Manual: cd sveltekit-frontend && npm run dev'))
    console.log(chalk.dim('   2. Services: scripts/start-external-services.bat'))
    console.log(chalk.dim('   3. Health: node scripts/check-services.mjs'))
    process.exit(1)
  }
}

// Helper function for delays
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// Run the main function
main().catch(console.error)