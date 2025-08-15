#!/usr/bin/env zx

/**
 * Integrated Legal AI System Startup
 * Technologies: Node.js Cluster + NATS + WebGPU + Go Microservices + GPU Acceleration
 */

import { $, chalk, os } from 'zx'

// Configuration
const config = {
  ports: {
    nats: 4222,
    nats_http: 8222,
    redis: 6379,
    postgres: 5432,
    go_rag: 8094,
    go_upload: 8093,
    go_simd: 8082,
    frontend: 5173,
    node_cluster: 3000
  },
  services: {
    nats: './nats-server.exe',
    redis: './redis-windows/redis-server.exe',
    postgres: '"C:\\Program Files\\PostgreSQL\\17\\bin\\pg_ctl.exe"',
    ollama: 'ollama serve'
  },
  gpu: {
    device: 'RTX 3060 Ti',
    memory: '8GB',
    cuda_version: '12.0'
  }
}

console.log(chalk.blue('🚀 Starting Integrated Legal AI System'))
console.log(chalk.gray(`GPU: ${config.gpu.device} (${config.gpu.memory} VRAM)`))

// Phase 1: Start Infrastructure Services
console.log(chalk.yellow('\n📦 Phase 1: Infrastructure Services'))

async function startInfrastructure() {
  const services = [
    {
      name: 'PostgreSQL',
      cmd: `${config.services.postgres} start -D "C:\\Program Files\\PostgreSQL\\17\\data"`,
      port: config.ports.postgres
    },
    {
      name: 'Redis',
      cmd: `start /min ${config.services.redis}`,
      port: config.ports.redis
    },
    {
      name: 'NATS',
      cmd: `start /min ${config.services.nats} --port ${config.ports.nats} --http_port ${config.ports.nats_http}`,
      port: config.ports.nats
    },
    {
      name: 'Ollama',
      cmd: `start /min ${config.services.ollama}`,
      port: 11434
    }
  ]

  for (const service of services) {
    try {
      console.log(chalk.cyan(`  Starting ${service.name}...`))
      await $`${service.cmd}`
      await sleep(2000) // Give service time to start
      console.log(chalk.green(`  ✅ ${service.name} started on port ${service.port}`))
    } catch (error) {
      console.log(chalk.red(`  ❌ Failed to start ${service.name}: ${error.message}`))
    }
  }
}

// Phase 2: Build and Start Go Microservices
console.log(chalk.yellow('\n🔧 Phase 2: Go Microservices'))

async function startGoServices() {
  const goServices = [
    {
      name: 'Enhanced RAG',
      build: 'go build -o bin/enhanced-rag.exe cmd/enhanced-rag/main.go',
      run: 'start /min bin/enhanced-rag.exe',
      port: config.ports.go_rag
    },
    {
      name: 'Upload Service',
      build: 'go build -o bin/upload-service.exe cmd/upload-service/main.go',
      run: 'start /min bin/upload-service.exe',
      port: config.ports.go_upload
    },
    {
      name: 'SIMD Parser',
      build: 'go build -tags legacy -o bin/simd-parser.exe simd_parser.go',
      run: 'start /min bin/simd-parser.exe',
      port: config.ports.go_simd
    }
  ]

  await $`cd go-microservice`
  
  for (const service of goServices) {
    try {
      console.log(chalk.cyan(`  Building ${service.name}...`))
      await $`cd go-microservice && ${service.build}`
      
      console.log(chalk.cyan(`  Starting ${service.name}...`))
      await $`cd go-microservice && ${service.run}`
      
      console.log(chalk.green(`  ✅ ${service.name} started on port ${service.port}`))
      await sleep(3000)
    } catch (error) {
      console.log(chalk.red(`  ❌ Failed to start ${service.name}: ${error.message}`))
    }
  }
}

// Phase 3: Start Node.js Cluster with PM2
console.log(chalk.yellow('\n🌐 Phase 3: Node.js Cluster'))

async function startNodeCluster() {
  const pm2Config = {
    name: 'legal-ai-cluster',
    script: 'scripts/cluster-manager.mjs',
    instances: os.cpus().length - 1, // Leave one core for system
    exec_mode: 'cluster',
    env: {
      NODE_ENV: 'production',
      NATS_URL: `nats://localhost:${config.ports.nats}`,
      REDIS_URL: `redis://localhost:${config.ports.redis}`,
      DATABASE_URL: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
      GPU_ACCELERATION: 'true',
      WEBGPU_ENABLED: 'true'
    }
  }

  try {
    console.log(chalk.cyan(`  Starting Node.js cluster with ${pm2Config.instances} workers...`))
    
    // Create PM2 ecosystem file
    await $`echo ${JSON.stringify({ apps: [pm2Config] }, null, 2)} > ecosystem.config.json`
    
    // Start with PM2
    await $`pm2 start ecosystem.config.json`
    await $`pm2 logs legal-ai-cluster --lines 10`
    
    console.log(chalk.green(`  ✅ Node.js cluster started with ${pm2Config.instances} workers`))
  } catch (error) {
    console.log(chalk.red(`  ❌ Failed to start Node.js cluster: ${error.message}`))
  }
}

// Phase 4: Start SvelteKit Frontend with WebGPU
console.log(chalk.yellow('\n🎨 Phase 4: SvelteKit Frontend'))

async function startFrontend() {
  try {
    console.log(chalk.cyan('  Starting SvelteKit with WebGPU acceleration...'))
    
    // Set environment variables for WebGPU
    process.env.VITE_WEBGPU_ENABLED = 'true'
    process.env.VITE_GPU_DEVICE = config.gpu.device
    process.env.VITE_NATS_WS_URL = `ws://localhost:${config.ports.nats_http}`
    process.env.VITE_API_BASE_URL = `http://localhost:${config.ports.node_cluster}`
    
    await $`cd sveltekit-frontend`
    await $`cd sveltekit-frontend && npm run dev -- --port ${config.ports.frontend} --host 0.0.0.0`
    
    console.log(chalk.green(`  ✅ SvelteKit frontend started on port ${config.ports.frontend}`))
  } catch (error) {
    console.log(chalk.red(`  ❌ Failed to start frontend: ${error.message}`))
  }
}

// Health Check Function
async function healthCheck() {
  console.log(chalk.yellow('\n🏥 System Health Check'))
  
  const endpoints = [
    { name: 'NATS HTTP', url: `http://localhost:${config.ports.nats_http}/varz` },
    { name: 'Enhanced RAG', url: `http://localhost:${config.ports.go_rag}/health` },
    { name: 'Upload Service', url: `http://localhost:${config.ports.go_upload}/health` },
    { name: 'SIMD Parser', url: `http://localhost:${config.ports.go_simd}/health` },
    { name: 'Frontend', url: `http://localhost:${config.ports.frontend}` }
  ]

  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint.url)
      if (response.ok) {
        console.log(chalk.green(`  ✅ ${endpoint.name}: Healthy`))
      } else {
        console.log(chalk.yellow(`  ⚠️  ${endpoint.name}: Responding but not OK (${response.status})`))
      }
    } catch (error) {
      console.log(chalk.red(`  ❌ ${endpoint.name}: Not responding`))
    }
  }
}

// GPU Status Check
async function checkGPU() {
  console.log(chalk.yellow('\n🎮 GPU Status'))
  
  try {
    // Check NVIDIA GPU
    const nvidiaInfo = await $`nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits`
    console.log(chalk.green(`  ✅ GPU: ${nvidiaInfo.stdout.trim()}`))
    
    // Check CUDA availability
    const cudaVersion = await $`nvcc --version | findstr "release"`
    console.log(chalk.green(`  ✅ CUDA: ${cudaVersion.stdout.trim()}`))
  } catch (error) {
    console.log(chalk.red(`  ❌ GPU not available or NVIDIA tools not installed`))
  }
}

// Recommendation System
function generateRecommendations() {
  console.log(chalk.yellow('\n💡 System Recommendations'))
  
  const recommendations = [
    {
      category: 'Performance',
      items: [
        'Use protobuf for API communication (50-70% size reduction)',
        'Implement WebGPU vertex buffer caching for UI acceleration',
        'Enable Redis clustering for high availability',
        'Use NATS streaming for real-time legal document updates'
      ]
    },
    {
      category: 'Architecture',
      items: [
        'Migrate heavy compute to Go+CUDA microservices',
        'Implement semantic caching with self-organizing maps',
        'Use PostgreSQL JSONB for legal document metadata',
        'Add WebAssembly workers for client-side AI tasks'
      ]
    },
    {
      category: 'Scaling',
      items: [
        'Horizontal scaling with NATS message queues',
        'GPU workload distribution across multiple nodes',
        'Implement predictive prefetching for legal workflows',
        'Use WebGPU compute shaders for client-side operations'
      ]
    }
  ]

  recommendations.forEach(category => {
    console.log(chalk.cyan(`\n  ${category.category}:`))
    category.items.forEach(item => {
      console.log(chalk.gray(`    • ${item}`))
    })
  })
}

// Main execution
async function main() {
  try {
    await checkGPU()
    await startInfrastructure()
    await startGoServices()
    await startNodeCluster()
    
    // Wait a bit for services to stabilize
    await sleep(5000)
    
    await healthCheck()
    generateRecommendations()
    
    console.log(chalk.green('\n🎉 Integrated Legal AI System startup complete!'))
    console.log(chalk.blue(`\n🌐 Access points:`))
    console.log(chalk.gray(`  Frontend: http://localhost:${config.ports.frontend}`))
    console.log(chalk.gray(`  NATS Monitor: http://localhost:${config.ports.nats_http}`))
    console.log(chalk.gray(`  Enhanced RAG: http://localhost:${config.ports.go_rag}/health`))
    
    // Start frontend last (blocking)
    await startFrontend()
    
  } catch (error) {
    console.error(chalk.red(`❌ Startup failed: ${error.message}`))
    process.exit(1)
  }
}

// Utility functions
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error)
}

export { main, healthCheck, generateRecommendations }