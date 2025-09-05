#!/usr/bin/env node
// Enhanced Enterprise Development Environment Starter
// Integrates native Windows services with SvelteKit development

import { spawn, exec } from 'child_process';
import { promisify } from 'util';
import { readFileSync, existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const execAsync = promisify(exec);

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

const log = {
  info: (msg) => console.log(`${colors.blue}ℹ${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✓${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}✗${colors.reset} ${msg}`),
  section: (msg) => console.log(`\n${colors.cyan}${colors.bright}=== ${msg} ===${colors.reset}`)
};

class EnterpriseDevEnvironment {
  constructor() {
    this.services = new Map();
    this.healthChecks = new Map();
    this.startTime = Date.now();
  }

  async checkServiceHealth(name, command, successPattern) {
    try {
      const { stdout, stderr } = await execAsync(command, { timeout: 5000 });
      const output = stdout + stderr;
      const isHealthy = successPattern ? output.includes(successPattern) : true;
      
      this.healthChecks.set(name, {
        status: isHealthy ? 'healthy' : 'unhealthy',
        lastCheck: new Date(),
        output: output.trim()
      });
      
      return isHealthy;
    } catch (error) {
      this.healthChecks.set(name, {
        status: 'error',
        lastCheck: new Date(),
        error: error.message
      });
      return false;
    }
  }

  async startPostgreSQL() {
    log.section('PostgreSQL Database Service');
    
    const isHealthy = await this.checkServiceHealth(
      'PostgreSQL', 
      'pg_isready -h localhost -p 5432',
      'accepting connections'
    );
    
    if (isHealthy) {
      log.success('PostgreSQL is already running');
      return true;
    }
    
    log.info('Starting PostgreSQL service...');
    try {
      await execAsync('net start postgresql-x64-17 || net start postgresql-x64-16 || net start postgresql-x64-15');
      
      // Wait for PostgreSQL to be ready
      let attempts = 0;
      while (attempts < 30) {
        if (await this.checkServiceHealth('PostgreSQL', 'pg_isready -h localhost -p 5432', 'accepting connections')) {
          log.success('PostgreSQL started successfully');
          return true;
        }
        await this.sleep(1000);
        attempts++;
      }
      
      log.error('PostgreSQL failed to start within 30 seconds');
      return false;
    } catch (error) {
      log.warning('PostgreSQL service not installed or failed to start');
      log.info('Manual setup required: https://www.postgresql.org/download/windows/');
      return false;
    }
  }

  async startRedis() {
    log.section('Redis Cache Service');
    
    const isHealthy = await this.checkServiceHealth(
      'Redis',
      'redis-cli ping',
      'PONG'
    );
    
    if (isHealthy) {
      log.success('Redis is already running');
      return true;
    }
    
    log.info('Starting Redis service...');
    
    // Try multiple Redis locations
    const redisCommands = [
      'C:\\enterprise-services\\redis\\start-redis.bat',
      'redis-server',
      'C:\\Redis\\redis-server.exe',
      '"C:\\Program Files\\Redis\\redis-server.exe"'
    ];
    
    for (const command of redisCommands) {
      try {
        if (command.includes('.bat') && existsSync(command)) {
          spawn('cmd', ['/c', command], { 
            detached: true, 
            stdio: 'ignore' 
          }).unref();
        } else {
          spawn(command.replace(/"/g, ''), [], { 
            detached: true, 
            stdio: 'ignore' 
          }).unref();
        }
        
        // Wait for Redis to start
        await this.sleep(3000);
        if (await this.checkServiceHealth('Redis', 'redis-cli ping', 'PONG')) {
          log.success('Redis started successfully');
          return true;
        }
      } catch (error) {
        continue;
      }
    }
    
    log.warning('Redis not found or failed to start');
    log.info('Install from: https://github.com/microsoftarchive/redis/releases');
    return false;
  }

  async startRabbitMQ() {
    log.section('RabbitMQ Message Queue');
    
    const isHealthy = await this.checkServiceHealth(
      'RabbitMQ',
      'rabbitmq-diagnostics status',
      'Status of node'
    );
    
    if (isHealthy) {
      log.success('RabbitMQ is already running');
      return true;
    }
    
    log.info('Starting RabbitMQ service...');
    try {
      await execAsync('net start RabbitMQ');
      
      // Wait for RabbitMQ to be ready
      let attempts = 0;
      while (attempts < 20) {
        if (await this.checkServiceHealth('RabbitMQ', 'rabbitmq-diagnostics status', 'Status of node')) {
          log.success('RabbitMQ started successfully');
          return true;
        }
        await this.sleep(2000);
        attempts++;
      }
      
      log.error('RabbitMQ failed to start within 40 seconds');
      return false;
    } catch (error) {
      log.warning('RabbitMQ service not installed or failed to start');
      log.info('Install from: https://www.rabbitmq.com/install-windows.html');
      return false;
    }
  }

  async startEnhancedRAG() {
    log.section('Enhanced RAG Microservice');
    
    const ragPath = path.resolve('..', 'go-microservice', 'bin', 'enhanced-rag.exe');
    
    if (!existsSync(ragPath)) {
      log.error('Enhanced RAG service not found at: ' + ragPath);
      return false;
    }
    
    log.info('Starting Enhanced RAG service...');
    
    const ragProcess = spawn(ragPath, [], {
      cwd: path.dirname(ragPath),
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe']
    });
    
    ragProcess.unref();
    
    // Monitor the process
    ragProcess.stdout?.on('data', (data) => {
      log.info(`RAG: ${data.toString().trim()}`);
    });
    
    ragProcess.stderr?.on('data', (data) => {
      log.warning(`RAG Error: ${data.toString().trim()}`);
    });
    
    this.services.set('enhanced-rag', ragProcess);
    
    // Wait for service to be ready
    await this.sleep(5000);
    
    const isHealthy = await this.checkServiceHealth(
      'Enhanced RAG',
      'curl -s http://localhost:8094/health',
      'status'
    );
    
    if (isHealthy) {
      log.success('Enhanced RAG service started successfully');
      return true;
    } else {
      log.warning('Enhanced RAG service may not be fully ready yet');
      return true; // Continue anyway
    }
  }

  async startUploadService() {
    log.section('Upload Service');
    
    const uploadPath = path.resolve('..', 'go-microservice', 'bin', 'upload-service.exe');
    
    if (!existsSync(uploadPath)) {
      log.warning('Upload service not found at: ' + uploadPath);
      return false;
    }
    
    log.info('Starting Upload service...');
    
    const uploadProcess = spawn(uploadPath, ['--port=8093'], {
      cwd: path.dirname(uploadPath),
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe']
    });
    
    uploadProcess.unref();
    
    this.services.set('upload-service', uploadProcess);
    
    await this.sleep(3000);
    
    const isHealthy = await this.checkServiceHealth(
      'Upload Service',
      'curl -s http://localhost:8093/health',
      'status'
    );
    
    if (isHealthy) {
      log.success('Upload service started successfully');
    } else {
      log.warning('Upload service may not be fully ready yet');
    }
    
    return true;
  }

  async startSvelteKit() {
    log.section('SvelteKit Development Server');
    
    log.info('Starting SvelteKit dev server...');
    
    const svelteProcess = spawn('npm', ['run', 'dev'], {
      stdio: 'inherit',
      shell: true
    });
    
    this.services.set('sveltekit', svelteProcess);
    
    svelteProcess.on('exit', (code) => {
      log.info(`SvelteKit dev server exited with code ${code}`);
      this.shutdown();
    });
    
    return true;
  }

  async displayStatusDashboard() {
    log.section('Enterprise Development Environment Status');
    
    console.log(`${colors.bright}Service Health Dashboard${colors.reset}`);
    console.log('━'.repeat(50));
    
    const services = [
      'PostgreSQL',
      'Redis', 
      'RabbitMQ',
      'Enhanced RAG',
      'Upload Service'
    ];
    
    for (const service of services) {
      const health = this.healthChecks.get(service);
      if (health) {
        const statusIcon = health.status === 'healthy' ? '✓' : 
                          health.status === 'unhealthy' ? '⚠' : '✗';
        const statusColor = health.status === 'healthy' ? colors.green :
                          health.status === 'unhealthy' ? colors.yellow : colors.red;
        
        console.log(`${statusColor}${statusIcon}${colors.reset} ${service.padEnd(20)} ${health.status}`);
      }
    }
    
    console.log('━'.repeat(50));
    console.log(`${colors.blue}Service Endpoints:${colors.reset}`);
    console.log('• SvelteKit Frontend: http://localhost:5173');
    console.log('• Enhanced RAG API:   http://localhost:8094');
    console.log('• Upload Service:     http://localhost:8093');
    console.log('• PostgreSQL:         localhost:5432');
    console.log('• Redis:              localhost:6379');
    console.log('• RabbitMQ:           localhost:5672 (mgmt: :15672)');
    console.log();
    
    const uptime = Math.round((Date.now() - this.startTime) / 1000);
    console.log(`${colors.green}Environment ready! Uptime: ${uptime}s${colors.reset}`);
  }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  shutdown() {
    log.info('Shutting down enterprise development environment...');
    
    for (const [name, process] of this.services) {
      try {
        process.kill();
        log.info(`Stopped ${name}`);
      } catch (error) {
        log.warning(`Failed to stop ${name}: ${error.message}`);
      }
    }
    
    process.exit(0);
  }

  async start() {
    console.log(`${colors.cyan}${colors.bright}`);
    console.log('╔══════════════════════════════════════════════════════════════╗');
    console.log('║             Enhanced Enterprise Development Environment       ║');
    console.log('║                   Legal AI Platform v2.0                     ║');
    console.log('╚══════════════════════════════════════════════════════════════╝');
    console.log(colors.reset);
    
    // Setup graceful shutdown
    process.on('SIGINT', () => this.shutdown());
    process.on('SIGTERM', () => this.shutdown());
    
    // Start all services
    const services = [
      () => this.startPostgreSQL(),
      () => this.startRedis(),
      () => this.startRabbitMQ(),
      () => this.startEnhancedRAG(),
      () => this.startUploadService()
    ];
    
    for (const startService of services) {
      try {
        await startService();
      } catch (error) {
        log.error(`Service startup failed: ${error.message}`);
      }
      await this.sleep(1000); // Stagger service starts
    }
    
    // Display status
    await this.displayStatusDashboard();
    
    // Start SvelteKit (blocks)
    await this.startSvelteKit();
  }
}

// Start the enterprise development environment
const env = new EnterpriseDevEnvironment();
env.start().catch(error => {
  log.error(`Failed to start enterprise environment: ${error.message}`);
  process.exit(1);
});