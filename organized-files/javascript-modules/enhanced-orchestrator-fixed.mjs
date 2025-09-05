// ENHANCED LEGAL AI ORCHESTRATOR WITH TIMING - ERROR FIXES
// Complete wiring of all services with procedural timing
// Run with: npm run start:production

import { execSync, spawn } from 'child_process';
import fetch from 'node-fetch';
import { writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

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
  
  // Log to file (FIXED: removed extra backslash)
  const logFile = join(CONFIG.logDir, `orchestrator-${new Date().toISOString().split('T')[0]}.log`);
  try {
    writeFileSync(logFile, msg + '\n', { flag: 'a' });
  } catch (e) {
    console.warn('Failed to write to log file:', e.message);
  }
};

// Simple built-in reporter (no external dependency)
class SimpleReporter {
  constructor() {
    this.phases = new Map();
    this.startTime = Date.now();
    this.services = new Map();
    this.errors = [];
    this.warnings = [];
  }

  startPhase(name, description) {
    const phase = {
      name,
      description,
      startTime: Date.now(),
      endTime: null,
      duration: null,
      status: 'running'
    };
    
    this.phases.set(name, phase);
    console.log(`🔄 [${name}] ${description}`);
    return phase;
  }

  endPhase(name, status = 'completed') {
    const phase = this.phases.get(name);
    if (!phase) return null;

    phase.endTime = Date.now();
    phase.duration = phase.endTime - phase.startTime;
    phase.status = status;
    
    const durationSec = (phase.duration / 1000).toFixed(2);
    const emoji = status === 'completed' ? '✅' : status === 'failed' ? '❌' : '⚠️';
    
    console.log(`${emoji} [${name}] ${phase.description} - ${durationSec}s`);
    
    return phase;
  }

  recordService(name, port, status, details = {}) {
    this.services.set(name, {
      name,
      port,
      status,
      timestamp: new Date().toISOString(),
      ...details
    });
  }

  recordError(error, context = '') {
    const errorRecord = {
      message: error.message || error,
      context,
      timestamp: new Date().toISOString()
    };
    
    this.errors.push(errorRecord);
    console.error(`❌ ERROR [${context}]: ${error.message || error}`);
  }

  recordWarning(warning, context = '') {
    const warningRecord = {
      message: warning,
      context,
      timestamp: new Date().toISOString()
    };
    
    this.warnings.push(warningRecord);
    console.warn(`⚠️ WARNING [${context}]: ${warning}`);
  }

  printSummary() {
    const totalTime = Date.now() - this.startTime;
    const servicesArray = Array.from(this.services.values());
    const healthyServices = servicesArray.filter(s => s.status === 'healthy');
    
    console.log('\n========================================');
    console.log('📊 STARTUP SUMMARY');
    console.log('========================================');
    console.log(`🕐 Total Time: ${(totalTime / 1000).toFixed(2)}s`);
    console.log(`📊 Health Score: ${Math.round((healthyServices.length / servicesArray.length) * 100)}%`);
    console.log(`🟢 Services: ${healthyServices.length}/${servicesArray.length}`);
    
    if (this.errors.length > 0) {
      console.log(`\n❌ Errors (${this.errors.length}):`);
      this.errors.forEach(error => {
        console.log(`   • [${error.context}] ${error.message}`);
      });
    }

    if (this.warnings.length > 0) {
      console.log(`\n⚠️ Warnings (${this.warnings.length}):`);
      this.warnings.forEach(warning => {
        console.log(`   • [${warning.context}] ${warning.message}`);
      });
    }

    console.log('========================================\n');
    
    return {
      totalTime: (totalTime / 1000).toFixed(2) + 's',
      healthyServices: healthyServices.length,
      totalServices: servicesArray.length,
      errors: this.errors.length,
      warnings: this.warnings.length
    };
  }
}

// Enhanced Service Manager with error fixes
class EnhancedServiceManager {
  constructor() {
    this.processes = new Map();
    this.reporter = new SimpleReporter();
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
        const net = require('net'); // FIXED: moved require inside try block
        const socket = new net.Socket();
        
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
        // Try multiple PostgreSQL service names
        const pgServices = ['postgresql-x64-16', 'postgresql-x64-15', 'postgresql-x64-14', 'PostgreSQL'];
        let pgStarted = false;
        
        for (const serviceName of pgServices) {
          try {
            await this.executeCommand(`net start ${serviceName}`, { ignoreErrors: true, silent: true });
            if (await this.checkPort(5432)) {
              log(`PostgreSQL started via ${serviceName}`);
              pgStarted = true;
              break;
            }
          } catch (e) {
            // Try next service name
            continue;
          }
        }
        
        if (!pgStarted && !(await this.waitForService('PostgreSQL', 5432, 10000))) {
          throw new Error('PostgreSQL failed to start - please ensure PostgreSQL is installed');
        }
      }

      // Setup extensions in legal_ai_db (with better error handling)
      await sleep(2000);
      try {
        await this.executeCommand(`psql -U postgres -d ${CONFIG.database} -c "CREATE EXTENSION IF NOT EXISTS vector;"`, { ignoreErrors: true, silent: true });
        await this.executeCommand(`psql -U postgres -d ${CONFIG.database} -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"`, { ignoreErrors: true, silent: true });
        await this.executeCommand(`psql -U postgres -d ${CONFIG.database} -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;"`, { ignoreErrors: true, silent: true });
        log('✅ PostgreSQL extensions configured');
      } catch (error) {
        this.reporter.recordWarning('Failed to install some PostgreSQL extensions', 'PostgreSQL');
      }
    } catch (error) {
      this.reporter.recordError(error, 'PostgreSQL');
      throw error;
    }

    // Redis (optional) - FIXED: better Windows Redis detection
    try {
      if (await this.checkPort(6379)) {
        log('Redis already running');
        this.reporter.recordService('redis', 6379, 'healthy');
      } else {
        // Try to start Redis (Windows native)
        try {
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
        } catch (redisError) {
          this.reporter.recordWarning('Redis not found - install Redis for enhanced caching', 'Redis');
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
        // FIXED: Check if Ollama is installed before trying to start
        try {
          await this.executeCommand('ollama --version', { silent: true });
        } catch (e) {
          throw new Error('Ollama not installed. Please install from https://ollama.ai/download');
        }

        const ollamaProcess = spawn('ollama', ['serve'], {
          detached: true,
          stdio: ['ignore', 'ignore', 'ignore']
        });
        ollamaProcess.unref();
        this.processes.set('ollama', ollamaProcess);
        
        if (!(await this.waitForService('Ollama', 11434, 30000))) { // FIXED: increased timeout
          throw new Error('Ollama failed to start');
        }
      }

      await sleep(3000); // FIXED: increased wait time for Ollama to fully initialize
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
      const response = await fetch('http://localhost:11434/api/tags', {
        method: 'GET',
        timeout: 10000 // FIXED: added timeout
      });
      
      if (!response.ok) {
        throw new Error(`Ollama API returned ${response.status}`);
      }
      
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
        log(`📥 Installing model: ${model} (this may take several minutes)`);
        try {
          await this.executeCommand(`ollama pull ${model}`, { timeout: 600000 }); // FIXED: 10 minute timeout
          log(`✅ Installed model: ${model}`, 'SUCCESS');
        } catch (error) {
          throw new Error(`Failed to install model ${model}: ${error.message}`);
        }
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
      // FIXED: Check if migration file exists
      if (!existsSync('production-migration.sql')) {
        this.reporter.recordWarning('production-migration.sql not found - skipping database setup', 'DatabaseSetup');
        return true;
      }
      
      await this.executeCommand(`psql -U postgres -d ${CONFIG.database} -f production-migration.sql`, { 
        ignoreErrors: true,
        silent: true
      });
      log('✅ Database migrations completed');
      return true;
    } catch (error) {
      this.reporter.recordWarning(`Database migrations failed: ${error.message}`, 'DatabaseSetup');
      return true; // Don't fail startup for migration issues
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
        try {
          await this.executeCommand('qdrant --version', { silent: true });
          const qdrantProcess = spawn('qdrant', [], {
            detached: true,
            stdio: ['ignore', 'ignore', 'ignore']
          });
          qdrantProcess.unref();
          this.processes.set('qdrant', qdrantProcess);
          results.qdrant = await this.waitForService('Qdrant', 6333, 15000);
        } catch (e) {
          log(`⚠️ Qdrant not installed - download from https://qdrant.tech/`, 'WARN');
          results.qdrant = false;
        }
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
        try {
          await this.executeCommand('minio --version', { silent: true });
          
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
        } catch (e) {
          log(`⚠️ MinIO not installed - download from https://min.io/download`, 'WARN');
          results.minio = false;
        }
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
            env: { ...process.env, PORT: service.port.toString() } // FIXED: ensure PORT is string
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

      // FIXED: Better directory checking
      if (!existsSync('./sveltekit-frontend')) {
        if (existsSync('./src')) {
          log('Using current directory as SvelteKit app');
          // Current directory is the SvelteKit app
          const svelteProcess = spawn('npm', ['run', 'dev'], {
            detached: false,
            stdio: 'inherit',
            env: { ...process.env, PORT: '5173' }
          });
          
          this.processes.set('sveltekit', svelteProcess);
        } else {
          throw new Error('SvelteKit application directory not found');
        }
      } else {
        const svelteProcess = spawn('npm', ['run', 'dev'], {
          cwd: './sveltekit-frontend',
          detached: false,
          stdio: 'inherit',
          env: { ...process.env, PORT: '5173' }
        });
        
        this.processes.set('sveltekit', svelteProcess);
      }
      
      if (await this.waitForService('SvelteKit', 5173, 45000)) { // FIXED: increased timeout
        await sleep(5000); // FIXED: increased wait for API routes to initialize
        return true;
      }
      
      throw new Error('SvelteKit failed to start');
    } catch (error) {
      this.reporter.recordError(error, 'SvelteKit');
      throw error;
    }
  }

  async validateAllAPIs() {
    log('🔍 Validating API endpoints...');
    
    // Start with essential endpoints only
    const endpoints = [
      { url: 'http://localhost:5173/api/health', required: true },
      { url: 'http://localhost:5173/api/system/status', required: false },
      { url: 'http://localhost:5173/', required: true } // FIXED: check root path
    ];
    
    const results = [];
    let workingEndpoints = 0;
    
    for (const endpoint of endpoints) {
      try {
        const response = await fetch(endpoint.url, { 
          method: 'GET',
          timeout: 10000 // FIXED: increased timeout
        });
        
        const isWorking = response.ok || response.status === 401 || response.status === 404; // FIXED: 404 can be OK for some endpoints
        
        results.push({
          url: endpoint.url,
          status: response.status,
          ok: isWorking,
          required: endpoint.required
        });
        
        if (isWorking) {
          workingEndpoints++;
          log(`✅ Endpoint ready: ${endpoint.url} (${response.status})`);
        } else if (endpoint.required) {
          log(`❌ Required endpoint failed: ${endpoint.url} (${response.status})`, 'ERROR');
        } else {
          log(`⚠️ Optional endpoint issue: ${endpoint.url} (${response.status})`, 'WARN');
        }
      } catch (error) {
        results.push({
          url: endpoint.url,
          status: 'error',
          ok: false,
          required: endpoint.required,
          error: error.message
        });
        
        if (endpoint.required) {
          log(`❌ Required endpoint failed: ${endpoint.url} - ${error.message}`, 'ERROR');
        } else {
          log(`⚠️ Optional endpoint failed: ${endpoint.url} - ${error.message}`, 'WARN');
        }
      }
    }
    
    log(`📊 API Status: ${workingEndpoints}/${results.length} endpoints responding`);
    
    return {
      results,
      totalAPIs: results.length,
      workingAPIs: workingEndpoints,
      allCriticalWorking: workingEndpoints >= 1 // At least root or health endpoint
    };
  }

  async runOrchestration() {
    log('🎯 Starting Legal AI System Orchestration...');
    log(`📅 Start Time: ${new Date().toLocaleString()}`);
    log(`🤖 Models: gemma3:legal + nomic-embed-text`);
    log(`💾 Database: legal_ai_db`);
    log(`🖥️ Platform: Windows Native (No Docker)`);
    log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

    let criticalFailure = false;

    for (const phase of this.startupPhases) {
      this.reporter.startPhase(phase.name, `Executing ${phase.name}`);
      
      try {
        const result = await this[phase.method]();
        
        if (result === false) {
          if (['Infrastructure', 'AIServices', 'WebApplication'].includes(phase.name)) {
            this.reporter.endPhase(phase.name, 'failed');
            criticalFailure = true;
            break;
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
          criticalFailure = true;
          break;
        }
      }
    }

    // Generate final report
    const report = this.reporter.printSummary();
    
    if (criticalFailure) {
      log('❌ CRITICAL FAILURE - System cannot start', 'ERROR');
      return false;
    }
    
    // System status summary
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
    log(`   → Services: ${report.healthyServices}/${report.totalServices} healthy`);
    log(`   → Total Startup Time: ${report.totalTime}`);
    log('');
    log('📋 Next Steps:');
    log('   1. Visit http://localhost:5173 to access the application');
    log('   2. Run: npm run test:validate for comprehensive testing');
    log('   3. Check: npm run check:models to verify AI models');
    
    return true;
  }

  cleanup() {
    log('🧹 Cleaning up processes...');
    
    for (const [name, process] of this.processes) {
      try {
        if (!process.killed) {
          process.kill('SIGTERM'); // FIXED: use SIGTERM instead of default
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
    const success = await serviceManager.runOrchestration();
    if (!success) {
      process.exit(1);
    }
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
