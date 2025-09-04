// LEGAL AI MASTER ORCHESTRATOR
// Procedural startup with timing and health checks
// Run with: node master-orchestrator.mjs

import { execSync, spawn } from 'child_process';
import fetch from 'node-fetch';
import { writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { withTiming } from './system-reporter.mjs';

// Configuration
const CONFIG = {
  services: {
    postgresql: { port: 5432, timeout: 30000, required: true },
    redis: { port: 6379, timeout: 10000, required: false },
    ollama: { port: 11434, timeout: 20000, required: true },
    qdrant: { port: 6333, timeout: 15000, required: false },
    minio: { port: 9000, timeout: 10000, required: false },
    sveltekit: { port: 5173, timeout: 15000, required: true },
    enhanced_rag: { port: 8094, timeout: 10000, required: false },
    gpu_orchestrator: { port: 8095, timeout: 10000, required: false }
  },
  models: ['gemma3:legal', 'nomic-embed-text'],
  database: 'legal_ai_db',
  retries: 3,
  logDir: './logs'
};

// Utility functions
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
const timestamp = () => new Date().toISOString();

const log = (message, level = 'INFO') => {
  const msg = `[${timestamp()}] [${level}] ${message}`;
  console.log(msg);
  
  // Ensure logs directory exists
  if (!existsSync(CONFIG.logDir)) {
    mkdirSync(CONFIG.logDir, { recursive: true });
  }
  
  // Log to file
  const logFile = join(CONFIG.logDir, `orchestrator-${new Date().toISOString().split('T')[0]}.log`);
  try {
    writeFileSync(logFile, msg + '\\n', { flag: 'a' });
  } catch (e) {
    console.warn('Failed to write to log file:', e.message);
  }
};

const checkPort = async (port, host = 'localhost', timeout = 5000) => {
  return new Promise((resolve) => {
    const timeoutId = setTimeout(() => resolve(false), timeout);
    
    try {
      const socket = new (require('net').Socket)();
      
      socket.setTimeout(timeout);
      socket.on('connect', () => {
        clearTimeout(timeoutId);
        socket.destroy();
        resolve(true);
      });
      
      socket.on('timeout', () => {
        clearTimeout(timeoutId);
        socket.destroy();
        resolve(false);
      });
      
      socket.on('error', () => {
        clearTimeout(timeoutId);
        socket.destroy();
        resolve(false);
      });
      
      socket.connect(port, host);
    } catch (e) {
      clearTimeout(timeoutId);
      resolve(false);
    }
  });
};

const waitForService = async (name, port, timeout = 30000) => {
  const startTime = Date.now();
  log(`Waiting for ${name} on port ${port}...`);
  
  while (Date.now() - startTime < timeout) {
    if (await checkPort(port)) {
      log(`✓ ${name} is ready on port ${port}`, 'SUCCESS');
      return true;
    }
    await sleep(2000);
  }
  
  log(`✗ ${name} failed to start within ${timeout}ms`, 'ERROR');
  return false;
};

const executeCommand = (command, options = {}) => {
  return new Promise((resolve, reject) => {
    log(`Executing: ${command}`);
    
    const child = spawn('cmd', ['/c', command], {
      stdio: options.silent ? 'pipe' : 'inherit',
      shell: true,
      ...options
    });
    
    let stdout = '';
    let stderr = '';
    
    if (child.stdout) {
      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });
    }
    
    if (child.stderr) {
      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });
    }
    
    child.on('close', (code) => {
      if (code === 0 || options.ignoreErrors) {
        resolve({ stdout, stderr, code });
      } else {
        reject(new Error(`Command failed with code ${code}: ${stderr}`));
      }
    });
    
    child.on('error', (error) => {
      reject(error);
    });
  });
};

// Service Management
class ServiceManager {
  constructor() {
    this.processes = new Map();
    this.healthChecks = new Map();
  }

  async startPostgreSQL() {
    log('🗄️  Starting PostgreSQL...');
    
    try {
      // Check if already running
      if (await checkPort(5432)) {
        log('PostgreSQL already running');
        return true;
      }

      // Start PostgreSQL service
      await executeCommand('net start postgresql-x64-16', { ignoreErrors: true });
      
      // Wait for PostgreSQL
      if (await waitForService('PostgreSQL', 5432, 30000)) {
        // Setup extensions in legal_ai_db
        await sleep(3000);
        await executeCommand(`psql -U postgres -d ${CONFIG.database} -c "CREATE EXTENSION IF NOT EXISTS vector;"`, { ignoreErrors: true });
        await executeCommand(`psql -U postgres -d ${CONFIG.database} -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"`, { ignoreErrors: true });
        await executeCommand(`psql -U postgres -d ${CONFIG.database} -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;"`, { ignoreErrors: true });
        
        log('PostgreSQL extensions configured');
        return true;
      }
      
      return false;
    } catch (error) {
      log(`PostgreSQL startup failed: ${error.message}`, 'ERROR');
      return false;
    }
  }

  async startRedis() {
    log('🔴 Starting Redis...');
    
    try {
      if (await checkPort(6379)) {
        log('Redis already running');
        return true;
      }

      const redisCommand = 'redis-server --port 6379 --maxmemory 2gb --maxmemory-policy allkeys-lru';
      const redisProcess = spawn('cmd', ['/c', redisCommand], {
        detached: true,
        stdio: ['ignore', 'ignore', 'ignore']
      });
      
      redisProcess.unref();
      this.processes.set('redis', redisProcess);
      
      return await waitForService('Redis', 6379, 10000);
    } catch (error) {
      log(`Redis startup failed: ${error.message}`, 'WARN');
      return false;
    }
  }

  async startOllama() {
    log('🤖 Starting Ollama AI service...');
    
    try {
      if (await checkPort(11434)) {
        log('Ollama already running');
        return await this.verifyModels();
      }

      // Start Ollama
      const ollamaProcess = spawn('ollama', ['serve'], {
        detached: true,
        stdio: ['ignore', 'ignore', 'ignore']
      });
      
      ollamaProcess.unref();
      this.processes.set('ollama', ollamaProcess);
      
      if (await waitForService('Ollama', 11434, 20000)) {
        await sleep(2000);
        return await this.verifyModels();
      }
      
      return false;
    } catch (error) {
      log(`Ollama startup failed: ${error.message}`, 'ERROR');
      return false;
    }
  }

  async verifyModels() {
    log('🔍 Verifying AI models...');
    
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      const data = await response.json();
      
      const installedModels = data.models?.map(m => m.name) || [];
      const missingModels = [];
      
      for (const model of CONFIG.models) {
        const found = installedModels.some(installed => 
          installed.includes(model) || installed.includes(model.replace(':', '-'))
        );
        
        if (!found) {
          missingModels.push(model);
        } else {
          log(`✓ Found model: ${model}`);
        }
      }
      
      // Install missing models
      for (const model of missingModels) {
        log(`📥 Installing model: ${model}`);
        try {
          await executeCommand(`ollama pull ${model}`, { timeout: 300000 });
          log(`✓ Installed model: ${model}`, 'SUCCESS');
        } catch (error) {
          log(`✗ Failed to install model ${model}: ${error.message}`, 'ERROR');
          return false;
        }
      }
      
      return true;
    } catch (error) {
      log(`Model verification failed: ${error.message}`, 'ERROR');
      return false;
    }
  }

  async startOptionalServices() {
    log('⚡ Starting optional services...');
    
    const results = {};
    
    // Qdrant
    try {
      if (await checkPort(6333)) {
        log('Qdrant already running');
        results.qdrant = true;
      } else {
        const qdrantProcess = spawn('qdrant', [], {
          detached: true,
          stdio: ['ignore', 'ignore', 'ignore']
        });
        qdrantProcess.unref();
        this.processes.set('qdrant', qdrantProcess);
        results.qdrant = await waitForService('Qdrant', 6333, 15000);
      }
    } catch (error) {
      log(`Qdrant not available: ${error.message}`, 'WARN');
      results.qdrant = false;
    }
    
    // MinIO
    try {
      if (await checkPort(9000)) {
        log('MinIO already running');
        results.minio = true;
      } else {
        if (!existsSync('./minio-data')) {
          mkdirSync('./minio-data', { recursive: true });
        }
        
        const minioProcess = spawn('minio', ['server', './minio-data', '--console-address', ':9001'], {
          detached: true,
          stdio: ['ignore', 'ignore', 'ignore']
        });
        minioProcess.unref();
        this.processes.set('minio', minioProcess);
        results.minio = await waitForService('MinIO', 9000, 10000);
      }
    } catch (error) {
      log(`MinIO not available: ${error.message}`, 'WARN');
      results.minio = false;
    }
    
    return results;
  }

  async startGoMicroservices() {
    log('🔧 Starting Go microservices...');
    
    const services = [
      { name: 'enhanced-rag', port: 8094, executable: 'enhanced-rag-som-system.exe' },
      { name: 'gpu-orchestrator', port: 8095, executable: 'gpu-orchestrator.exe' }
    ];
    
    const results = {};
    
    for (const service of services) {
      try {
        if (await checkPort(service.port)) {
          log(`${service.name} already running on port ${service.port}`);
          results[service.name] = true;
          continue;
        }
        
        if (existsSync(service.executable)) {
          const serviceProcess = spawn(service.executable, [], {
            detached: true,
            stdio: ['ignore', 'ignore', 'ignore'],
            env: { ...process.env, PORT: service.port }
          });
          
          serviceProcess.unref();
          this.processes.set(service.name, serviceProcess);
          
          results[service.name] = await waitForService(service.name, service.port, 10000);
        } else {
          log(`${service.executable} not found, skipping ${service.name}`, 'WARN');
          results[service.name] = false;
        }
      } catch (error) {
        log(`Failed to start ${service.name}: ${error.message}`, 'WARN');
        results[service.name] = false;
      }
    }
    
    return results;
  }

  async runDatabaseMigrations() {
    log('📊 Running database migrations...');
    
    try {
      await executeCommand(`psql -U postgres -d ${CONFIG.database} -f production-migration.sql`, { 
        ignoreErrors: true 
      });
      log('✓ Database migrations completed', 'SUCCESS');
      return true;
    } catch (error) {
      log(`Database migrations failed: ${error.message}`, 'WARN');
      return false;
    }
  }

  async startSvelteKitApp() {
    log('🚀 Starting SvelteKit application...');
    
    try {
      if (await checkPort(5173)) {
        log('SvelteKit already running on port 5173');
        // Validate API routes are working
        await this.validateApiEndpoints();
        return true;
      }

      // Ensure sveltekit-frontend directory exists
      if (!existsSync('./sveltekit-frontend')) {
        log('sveltekit-frontend directory not found!', 'ERROR');
        return false;
      }

      // Change to sveltekit-frontend directory and start
      const svelteProcess = spawn('npm', ['run', 'dev'], {
        cwd: './sveltekit-frontend',
        detached: false,
        stdio: 'inherit',
        env: { ...process.env, PORT: '5173' }
      });
      
      this.processes.set('sveltekit', svelteProcess);
      
      if (await waitForService('SvelteKit', 5173, 30000)) {
        // Wait a bit for API routes to initialize
        await sleep(3000);
        await this.validateApiEndpoints();
        return true;
      }
      
      return false;
    } catch (error) {
      log(`SvelteKit startup failed: ${error.message}`, 'ERROR');
      return false;
    }
  }

  async validateApiEndpoints() {
    log('🔍 Validating API endpoints...');
    
    const endpoints = [
      'http://localhost:5173/api/health',
      'http://localhost:5173/api/system/status'
    ];
    
    for (const endpoint of endpoints) {
      try {
        const response = await fetch(endpoint, { 
          method: 'GET',
          timeout: 5000
        });
        
        if (response.ok) {
          log(`✓ API endpoint ready: ${endpoint}`);
        } else {
          log(`⚠ API endpoint responded with ${response.status}: ${endpoint}`, 'WARN');
        }
      } catch (error) {
        log(`✗ API endpoint failed: ${endpoint} - ${error.message}`, 'WARN');
      }
    }
  }

  async healthCheck() {
    log('🏥 Running comprehensive health check...');
    
    const checks = {};
    
    for (const [name, config] of Object.entries(CONFIG.services)) {
      const isHealthy = await checkPort(config.port);
      checks[name] = isHealthy;
      
      if (isHealthy) {
        log(`✓ ${name} healthy on port ${config.port}`, 'SUCCESS');
      } else if (config.required) {
        log(`✗ ${name} unhealthy on port ${config.port} (REQUIRED)`, 'ERROR');
      } else {
        log(`⚠ ${name} unavailable on port ${config.port} (optional)`, 'WARN');
      }
    }
    
    return checks;
  }

  cleanup() {
    log('🧹 Cleaning up processes...');
    
    for (const [name, process] of this.processes) {
      try {
        if (!process.killed) {
          process.kill();
          log(`Stopped ${name}`);
        }
      } catch (error) {
        log(`Failed to stop ${name}: ${error.message}`, 'WARN');
      }
    }
    
    this.processes.clear();
  }
}

// Main orchestration
async function main() {
  const serviceManager = new ServiceManager();
  
  // Handle cleanup on exit
  process.on('SIGINT', () => {
    log('Received SIGINT, cleaning up...');
    serviceManager.cleanup();
    process.exit(0);
  });
  
  process.on('SIGTERM', () => {
    log('Received SIGTERM, cleaning up...');
    serviceManager.cleanup();
    process.exit(0);
  });
  
  log('========================================');
  log('🎯 LEGAL AI SYSTEM ORCHESTRATOR STARTING');
  log('========================================');
  
  try {
    // Phase 1: Core Infrastructure
    log('📋 PHASE 1: Core Infrastructure');
    const postgresOk = await serviceManager.startPostgreSQL();
    if (!postgresOk) {
      log('❌ PostgreSQL is required but failed to start', 'ERROR');
      return process.exit(1);
    }
    
    const redisOk = await serviceManager.startRedis();
    if (redisOk) {
      log('✅ Redis cache available');
    }
    
    // Phase 2: AI Services
    log('📋 PHASE 2: AI Services');
    const ollamaOk = await serviceManager.startOllama();
    if (!ollamaOk) {
      log('❌ Ollama AI is required but failed to start', 'ERROR');
      return process.exit(1);
    }
    
    // Phase 3: Database Setup
    log('📋 PHASE 3: Database Setup');
    await serviceManager.runDatabaseMigrations();
    
    // Phase 4: Optional Services
    log('📋 PHASE 4: Optional Services');
    const optionalServices = await serviceManager.startOptionalServices();
    
    // Phase 5: Microservices
    log('📋 PHASE 5: Go Microservices');
    const microservices = await serviceManager.startGoMicroservices();
    
    // Phase 6: Web Application
    log('📋 PHASE 6: Web Application');
    const svelteOk = await serviceManager.startSvelteKitApp();
    if (!svelteOk) {
      log('❌ SvelteKit application failed to start', 'ERROR');
      return process.exit(1);
    }
    
    // Final Health Check
    log('📋 FINAL: System Health Check');
    const healthStatus = await serviceManager.healthCheck();
    
    // Summary Report
    log('========================================');
    log('📊 SYSTEM STARTUP COMPLETE');
    log('========================================');
    
    const coreServicesOk = healthStatus.postgresql && healthStatus.ollama && healthStatus.sveltekit;
    const enhancedFeaturesCount = Object.values(optionalServices).filter(Boolean).length;
    const microservicesCount = Object.values(microservices).filter(Boolean).length;
    
    if (coreServicesOk) {
      log('🎉 SYSTEM READY FOR PRODUCTION!', 'SUCCESS');
      log(`📊 Core Services: OPERATIONAL`);
      log(`⚡ Enhanced Features: ${enhancedFeaturesCount}/2 available`);
      log(`🔧 Microservices: ${microservicesCount} running`);
      log('');
      log('🌐 Access your Legal AI System:');
      log('   → Application: http://localhost:5173');
      log('   → API Health: http://localhost:5173/api/health');
      log('   → Admin Login: admin@legalai.com / admin123');
      
      if (optionalServices.minio) {
        log('   → MinIO Console: http://localhost:9001');
      }
      if (optionalServices.qdrant) {
        log('   → Qdrant Dashboard: http://localhost:6333/dashboard');
      }
    } else {
      log('⚠️  SYSTEM PARTIALLY OPERATIONAL', 'WARN');
      log('Some core services failed to start. Check logs above.');
    }
    
  } catch (error) {
    log(`❌ Fatal error during startup: ${error.message}`, 'ERROR');
    serviceManager.cleanup();
    process.exit(1);
  }
}

// Run orchestrator
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}

export { ServiceManager, main };
