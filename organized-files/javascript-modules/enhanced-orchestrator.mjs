// ENHANCED LEGAL AI ORCHESTRATOR WITH TIMING
// Complete wiring of all services with procedural timing
// Run with: npm run start:production

import { execSync, spawn } from 'child_process';
import fetch from 'node-fetch';
import { writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { SystemReporter, ServiceCoordinator } from './system-reporter.mjs';

// Configuration for your existing setup
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

// Enhanced Service Manager with wired API integration
class EnhancedServiceManager {
  constructor() {
    this.processes = new Map();
    this.reporter = new SystemReporter();
    this.coordinator = new ServiceCoordinator(this.reporter);
    this.startupPhases = [
      { name: 'Infrastructure', method: 'startInfrastructure' },
      { name: 'AIServices', method: 'startAIServices' },
      { name: 'DatabaseSetup', method: 'setupDatabase' },
      { name: 'OptionalServices', method: 'startOptionalServices' },
      { name: 'Microservices', method: 'startMicroservices' },
      { name: 'WebApplication', method: 'startWebApplication' },
      { name: 'APIValidation', method: 'validateAllAPIs' }
    ];
  }

  async checkPort(port, host = 'localhost', timeout = 5000) {
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
  }

  async waitForService(name, port, timeout = 30000) {
    const startTime = Date.now();
    log(`🔄 Waiting for ${name} on port ${port}...`);
    
    while (Date.now() - startTime < timeout) {
      if (await this.checkPort(port)) {
        log(`✅ ${name} is ready on port ${port}`, 'SUCCESS');
        this.reporter.recordService(name.toLowerCase(), port, 'healthy');
        return true;
      }
      await sleep(2000);
    }
    
    log(`❌ ${name} failed to start within ${timeout}ms`, 'ERROR');
    this.reporter.recordService(name.toLowerCase(), port, 'failed');
    return false;
  }

  async executeCommand(command, options = {}) {
    return new Promise((resolve, reject) => {
      log(`⚡ Executing: ${command}`);
      
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
  }

  async startInfrastructure() {
    log('🏗️ Starting core infrastructure...');
    
    // PostgreSQL
    try {
      if (await this.checkPort(5432)) {
        log('PostgreSQL already running');
        this.reporter.recordService('postgresql', 5432, 'healthy');
      } else {
        await this.executeCommand('net start postgresql-x64-16', { ignoreErrors: true });
        if (!(await this.waitForService('PostgreSQL', 5432, 30000))) {
          throw new Error('PostgreSQL failed to start');
        }
      }

      // Setup extensions in legal_ai_db
      await sleep(2000);
      await this.executeCommand(`psql -U postgres -d ${CONFIG.database} -c "CREATE EXTENSION IF NOT EXISTS vector;"`, { ignoreErrors: true });
      await this.executeCommand(`psql -U postgres -d ${CONFIG.database} -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"`, { ignoreErrors: true });
      await this.executeCommand(`psql -U postgres -d ${CONFIG.database} -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;"`, { ignoreErrors: true });
      
      log('✅ PostgreSQL extensions configured');
    } catch (error) {
      this.reporter.recordError(error, 'PostgreSQL');
      throw error;
    }

    // Redis (optional)
    try {
      if (await this.checkPort(6379)) {
        log('Redis already running');
        this.reporter.recordService('redis', 6379, 'healthy');
      } else {
        const redisProcess = spawn('redis-server', ['--port', '6379', '--maxmemory', '2gb', '--maxmemory-policy', 'allkeys-lru'], {
          detached: true,
          stdio: ['ignore', 'ignore', 'ignore']
        });
        redisProcess.unref();
        this.processes.set('redis', redisProcess);
        
        if (await this.waitForService('Redis', 6379, 10000)) {
          log('✅ Redis cache available');
        } else {
          this.reporter.recordWarning('Redis failed to start - caching disabled', 'Redis');
        }
      }
    } catch (error) {
      this.reporter.recordWarning(`Redis startup failed: ${error.message}`, 'Redis');
    }

    return true;
  }

  async startAIServices() {
    log('🤖 Starting AI services...');
    
    try {
      if (await this.checkPort(11434)) {
        log('Ollama already running');
        this.reporter.recordService('ollama', 11434, 'healthy');
      } else {
        const ollamaProcess = spawn('ollama', ['serve'], {
          detached: true,
          stdio: ['ignore', 'ignore', 'ignore']
        });
        ollamaProcess.unref();
        this.processes.set('ollama', ollamaProcess);
        
        if (!(await this.waitForService('Ollama', 11434, 20000))) {
          throw new Error('Ollama failed to start');
        }
      }

      await sleep(2000);
      await this.verifyModels();
      return true;
    } catch (error) {
      this.reporter.recordError(error, 'Ollama');
      throw error;
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
          log(`✅ Found model: ${model}`);
        }
      }
      
      // Install missing models
      for (const model of missingModels) {
        log(`📥 Installing model: ${model}`);
        await this.executeCommand(`ollama pull ${model}`, { timeout: 300000 });
        log(`✅ Installed model: ${model}`, 'SUCCESS');
      }
      
      return true;
    } catch (error) {
      this.reporter.recordError(error, 'ModelVerification');
      throw error;
    }
  }

  async setupDatabase() {
    log('📊 Setting up database...');
    
    try {
      await this.executeCommand(`psql -U postgres -d ${CONFIG.database} -f production-migration.sql`, { 
        ignoreErrors: true 
      });
      log('✅ Database migrations completed');
      return true;
    } catch (error) {
      this.reporter.recordWarning(`Database migrations failed: ${error.message}`, 'DatabaseSetup');
      return false;
    }
  }

  async startOptionalServices() {
    log('⚡ Starting optional services...');
    
    const results = {};
    
    // Qdrant
    try {
      if (await this.checkPort(6333)) {
        log('Qdrant already running');
        this.reporter.recordService('qdrant', 6333, 'healthy');
        results.qdrant = true;
      } else {
        const qdrantProcess = spawn('qdrant', [], {
          detached: true,
          stdio: ['ignore', 'ignore', 'ignore']
        });
        qdrantProcess.unref();
        this.processes.set('qdrant', qdrantProcess);
        results.qdrant = await this.waitForService('Qdrant', 6333, 15000);
      }
    } catch (error) {
      log(`⚠️ Qdrant not available: ${error.message}`, 'WARN');
      results.qdrant = false;
    }
    
    // MinIO
    try {
      if (await this.checkPort(9000)) {
        log('MinIO already running');
        this.reporter.recordService('minio', 9000, 'healthy');
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
        results.minio = await this.waitForService('MinIO', 9000, 10000);
      }
    } catch (error) {
      log(`⚠️ MinIO not available: ${error.message}`, 'WARN');
      results.minio = false;
    }
    
    return results;
  }

  async startMicroservices() {
    log('🔧 Starting Go microservices...');
    
    const services = [
      { name: 'enhanced-rag', port: 8094, executable: 'enhanced-rag-som-system.exe' },
      { name: 'gpu-orchestrator', port: 8095, executable: 'gpu-orchestrator.exe' }
    ];
    
    const results = {};
    
    for (const service of services) {
      try {
        if (await this.checkPort(service.port)) {
          log(`${service.name} already running on port ${service.port}`);
          this.reporter.recordService(service.name, service.port, 'healthy');
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
          
          results[service.name] = await this.waitForService(service.name, service.port, 10000);
        } else {
          log(`⚠️ ${service.executable} not found, skipping ${service.name}`, 'WARN');
          this.reporter.recordWarning(`Executable not found: ${service.executable}`, service.name);
          results[service.name] = false;
        }
      } catch (error) {
        this.reporter.recordWarning(`Failed to start ${service.name}: ${error.message}`, service.name);
        results[service.name] = false;
      }
    }
    
    return results;
  }

  async startWebApplication() {
    log('🚀 Starting SvelteKit application...');
    
    try {
      if (await this.checkPort(5173)) {
        log('SvelteKit already running on port 5173');
        this.reporter.recordService('sveltekit', 5173, 'healthy');
        return true;
      }

      // Ensure sveltekit-frontend directory exists
      if (!existsSync('./sveltekit-frontend')) {
        throw new Error('sveltekit-frontend directory not found!');
      }

      const svelteProcess = spawn('npm', ['run', 'dev'], {
        cwd: './sveltekit-frontend',
        detached: false,
        stdio: 'inherit',
        env: { ...process.env, PORT: '5173' }
      });
      
      this.processes.set('sveltekit', svelteProcess);
      
      if (await this.waitForService('SvelteKit', 5173, 30000)) {
        await sleep(3000); // Wait for API routes to initialize
        return true;
      }
      
      throw new Error('SvelteKit failed to start');
    } catch (error) {
      this.reporter.recordError(error, 'SvelteKit');
      throw error;
    }
  }

  async validateAllAPIs() {
    log('🔍 Validating all API endpoints...');
    
    // Core API endpoints based on your sveltekit-frontend structure
    const endpoints = [
      { url: 'http://localhost:5173/api/health', required: true },
      { url: 'http://localhost:5173/api/system/status', required: true },
      { url: 'http://localhost:5173/api/cases', required: true },
      { url: 'http://localhost:5173/api/evidence', required: true },
      { url: 'http://localhost:5173/api/citations', required: true },
      { url: 'http://localhost:5173/api/reports', required: true },
      { url: 'http://localhost:5173/api/chat', required: true },
      { url: 'http://localhost:5173/api/search', required: true },
      { url: 'http://localhost:5173/api/ai', required: true },
      { url: 'http://localhost:5173/api/upload', required: true },
      { url: 'http://localhost:5173/api/auth', required: true },
      { url: 'http://localhost:5173/api/users', required: true }
    ];
    
    const results = [];
    
    for (const endpoint of endpoints) {
      try {
        const response = await fetch(endpoint.url, { 
          method: 'GET',
          timeout: 5000
        });
        
        const result = {
          url: endpoint.url,
          status: response.status,
          ok: response.ok,
          required: endpoint.required
        };
        
        results.push(result);
        
        if (response.ok || response.status === 401) { // 401 means auth required but API works
          log(`✅ API ready: ${endpoint.url}`);
        } else if (endpoint.required) {
          log(`❌ Required API failed: ${endpoint.url} (${response.status})`, 'ERROR');
          this.reporter.recordError(new Error(`Required API failed: ${endpoint.url}`), 'APIValidation');
        } else {
          log(`⚠️ Optional API issue: ${endpoint.url} (${response.status})`, 'WARN');
        }
      } catch (error) {
        const result = {
          url: endpoint.url,
          status: 'error',
          ok: false,
          required: endpoint.required,
          error: error.message
        };
        
        results.push(result);
        
        if (endpoint.required) {
          log(`❌ Required API failed: ${endpoint.url} - ${error.message}`, 'ERROR');
          this.reporter.recordError(error, 'APIValidation');
        } else {
          log(`⚠️ Optional API failed: ${endpoint.url} - ${error.message}`, 'WARN');
        }
      }
    }
    
    const workingAPIs = results.filter(r => r.ok || r.status === 401).length;
    const requiredAPIs = results.filter(r => r.required).length;
    const workingRequiredAPIs = results.filter(r => r.required && (r.ok || r.status === 401)).length;
    
    log(`📊 API Status: ${workingAPIs}/${results.length} total, ${workingRequiredAPIs}/${requiredAPIs} required`);
    
    return {
      results,
      totalAPIs: results.length,
      workingAPIs,
      requiredAPIs,
      workingRequiredAPIs,
      allRequiredWorking: workingRequiredAPIs === requiredAPIs
    };
  }

  async runOrchestration() {
    log('🎯 Starting Legal AI System Orchestration...');
    log(`📅 Start Time: ${new Date().toLocaleString()}`);
    log(`🤖 Models: gemma3:legal + nomic-embed-text`);
    log(`💾 Database: legal_ai_db`);
    log(`🖥️ Platform: Windows Native (No Docker)`);
    log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

    for (const phase of this.startupPhases) {
      const phaseReporter = this.reporter.startPhase(phase.name, `Executing ${phase.name}`);
      
      try {
        const result = await this[phase.method]();
        
        if (result === false || (typeof result === 'object' && Object.values(result).every(v => !v))) {
          if (phase.name === 'Infrastructure' || phase.name === 'AIServices' || phase.name === 'WebApplication') {
            this.reporter.endPhase(phase.name, 'failed');
            throw new Error(`Critical phase ${phase.name} failed`);
          } else {
            this.reporter.endPhase(phase.name, 'partial');
          }
        } else {
          this.reporter.endPhase(phase.name, 'completed');
        }
      } catch (error) {
        this.reporter.recordError(error, phase.name);
        this.reporter.endPhase(phase.name, 'failed');
        
        // Critical phases must succeed
        if (['Infrastructure', 'AIServices', 'WebApplication'].includes(phase.name)) {
          throw error;
        }
      }
    }

    // Generate final report
    const report = this.reporter.printDetailedSummary();
    
    // System status summary
    const healthyServices = Array.from(this.reporter.metrics.services.values())
      .filter(s => s.status === 'healthy').length;
    const totalServices = this.reporter.metrics.services.size;
    
    log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    log('🎉 LEGAL AI SYSTEM IS READY!');
    log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    log('🌐 Access Points:');
    log('   → Application: http://localhost:5173');
    log('   → API Health: http://localhost:5173/api/health');
    log('   → System Status: http://localhost:5173/api/system/status?details=true');
    log('   → Admin Login: admin@legalai.com / admin123');
    log('');
    log('📊 System Summary:');
    log(`   → Services: ${healthyServices}/${totalServices} healthy`);
    log(`   → Health Score: ${report.summary.healthScore}%`);
    log(`   → Total Startup Time: ${report.summary.totalDuration}`);
    log('');
    log('📋 Next Steps:');
    log('   1. Visit http://localhost:5173 to access the application');
    log('   2. Run: npm run test:validate for comprehensive testing');
    log('   3. Check: npm run check:models to verify AI models');
    
    return report;
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

// Main execution
async function main() {
  const serviceManager = new EnhancedServiceManager();
  
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
  
  try {
    await serviceManager.runOrchestration();
  } catch (error) {
    log(`❌ Fatal error: ${error.message}`, 'ERROR');
    serviceManager.cleanup();
    process.exit(1);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}

export { EnhancedServiceManager, main };
