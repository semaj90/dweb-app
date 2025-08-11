#!/usr/bin/env node

/**
 * YoRHa Legal AI - Production Start Orchestrator
 * 
 * Starts all services for production deployment with:
 * - Health checks and retries
 * - Service dependency management
 * - Performance monitoring
 * - Graceful shutdown handling
 * 
 * @author YoRHa Legal AI Team
 * @version 2.0.0
 */

import 'zx/globals';
import chalk from 'chalk';
import ora from 'ora';
import { WebSocket } from 'ws';
import fetch from 'node-fetch';
import { program } from 'commander';

// Production configuration
const PROD_CONFIG = {
  services: {
    postgresql: {
      name: 'PostgreSQL + pgvector',
      port: 5432,
      healthUrl: null,
      command: 'net start postgresql-x64-17', // Windows service
      priority: 1,
      retries: 3,
      timeout: 30000
    },
    redis: {
      name: 'Redis Cache',
      port: 6379,
      healthUrl: null,
      command: '.\\redis-windows\\redis-server.exe --service-install --service-name redis',
      priority: 2,
      retries: 3,
      timeout: 15000
    },
    ollama: {
      name: 'Ollama LLM',
      port: 11434,
      healthUrl: 'http://localhost:11434/api/version',
      command: 'ollama serve',
      priority: 3,
      retries: 5,
      timeout: 60000,
      warmup: ['llama3.1:8b', 'nomic-embed-text'] // Pre-load models
    },
    qdrant: {
      name: 'Qdrant Vector DB',
      port: 6333,
      healthUrl: 'http://localhost:6333/health',
      command: '.\\qdrant-windows\\qdrant.exe --config-path ./qdrant.toml',
      priority: 4,
      retries: 3,
      timeout: 30000
    },
    goService: {
      name: 'Go Legal AI Service',
      port: 8080,
      grpcPort: 50051,
      healthUrl: 'http://localhost:8080/api/health',
      command: '.\\legal-ai-server.exe --production',
      priority: 5,
      retries: 3,
      timeout: 20000,
      env: {
        'GIN_MODE': 'release',
        'DATABASE_URL': 'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db',
        'REDIS_URL': 'redis://localhost:6379',
        'OLLAMA_URL': 'http://localhost:11434',
        'QDRANT_URL': 'http://localhost:6333'
      }
    },
    sveltekit: {
      name: 'SvelteKit Frontend',
      port: 3000, // Production port
      healthUrl: 'http://localhost:3000/',
      command: 'npm run start', // Built production server
      cwd: path.join(process.cwd(), 'sveltekit-frontend'),
      priority: 6,
      retries: 3,
      timeout: 30000,
      env: {
        'NODE_ENV': 'production',
        'PORT': '3000',
        'HOST': '0.0.0.0'
      }
    }
  },
  monitoring: {
    enabled: true,
    interval: 15000, // 15 seconds
    alertThreshold: 3, // Alert after 3 failed checks
    metrics: {
      cpu: true,
      memory: true,
      disk: true,
      network: true
    }
  }
};

// Global state
const state = {
  services: new Map(),
  metrics: new Map(),
  alerts: new Map(),
  startTime: Date.now()
};

// Enhanced logging with timestamp
const log = {
  timestamp: () => new Date().toISOString(),
  info: (msg) => console.log(`[${log.timestamp()}]`, chalk.blue('ℹ'), msg),
  success: (msg) => console.log(`[${log.timestamp()}]`, chalk.green('✓'), msg),
  error: (msg) => console.log(`[${log.timestamp()}]`, chalk.red('✗'), msg),
  warn: (msg) => console.log(`[${log.timestamp()}]`, chalk.yellow('⚠'), msg),
  debug: (msg) => process.env.DEBUG && console.log(`[${log.timestamp()}]`, chalk.gray('🔍'), msg)
};

// Enhanced health checkers
const healthCheckers = {
  async postgresql() {
    try {
      const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\pg_isready.exe" -h localhost -p 5432`;
      return { healthy: result.exitCode === 0, latency: Date.now() };
    } catch (error) {
      return { healthy: false, error: error.message };
    }
  },

  async redis() {
    const start = Date.now();
    try {
      const result = await $`echo "ping" | .\\redis-windows\\redis-cli.exe -h localhost -p 6379`;
      return { 
        healthy: result.stdout.includes('PONG'), 
        latency: Date.now() - start 
      };
    } catch (error) {
      return { healthy: false, error: error.message };
    }
  },

  async http(url) {
    const start = Date.now();
    try {
      const response = await fetch(url, { 
        timeout: 10000,
        signal: AbortSignal.timeout(10000)
      });
      return { 
        healthy: response.ok, 
        latency: Date.now() - start,
        status: response.status
      };
    } catch (error) {
      return { healthy: false, error: error.message };
    }
  }
};

// Production service management
async function startProductionService(serviceName, config) {
  const spinner = ora(`🚀 Starting ${config.name} (Production)...`).start();
  
  let attempts = 0;
  while (attempts < config.retries) {
    try {
      attempts++;
      
      // Check if already running
      const healthCheck = await checkServiceHealth(serviceName, config);
      if (healthCheck.healthy) {
        spinner.succeed(`${config.name} already running (${healthCheck.latency}ms)`);
        return;
      }

      // Start service with production options
      const options = {
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: { ...process.env, ...config.env }
      };

      if (config.cwd) {
        options.cwd = config.cwd;
      }

      let proc;
      if (config.command.includes('net start')) {
        // Windows service
        proc = await $`${config.command}`;
        if (proc.exitCode === 0) {
          spinner.succeed(`${config.name} service started`);
          return;
        }
      } else if (config.command.includes('.exe')) {
        // Windows executable
        const [cmd, ...args] = config.command.split(' ');
        proc = spawn(cmd, args, options);
      } else {
        // Shell command
        proc = spawn('cmd', ['/c', config.command], { ...options, shell: true });
      }

      if (proc && proc.pid) {
        state.services.set(serviceName, { 
          process: proc, 
          config,
          startTime: Date.now()
        });
        
        // Wait for service with timeout
        const healthCheckPromise = waitForHealthy(serviceName, config, config.timeout);
        const timeoutPromise = new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Timeout')), config.timeout)
        );
        
        await Promise.race([healthCheckPromise, timeoutPromise]);
        
        spinner.succeed(`${config.name} started successfully (${attempts}/${config.retries})`);
        
        // Warmup phase for services that need it
        if (config.warmup) {
          await warmupService(serviceName, config);
        }
        
        return;
      } else {
        throw new Error('Failed to start process');
      }
    } catch (error) {
      if (attempts >= config.retries) {
        spinner.fail(`Failed to start ${config.name} after ${config.retries} attempts: ${error.message}`);
        throw error;
      } else {
        spinner.text = `Retrying ${config.name} (${attempts + 1}/${config.retries})...`;
        await sleep(5000 * attempts); // Exponential backoff
      }
    }
  }
}

async function waitForHealthy(serviceName, config, timeout = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const health = await checkServiceHealth(serviceName, config);
    if (health.healthy) {
      return;
    }
    await sleep(2000);
  }
  throw new Error('Health check timeout');
}

async function warmupService(serviceName, config) {
  if (serviceName === 'ollama' && config.warmup) {
    const spinner = ora(`🔥 Warming up ${config.name} models...`).start();
    
    for (const model of config.warmup) {
      try {
        // Pre-load model
        await $`ollama run ${model} "Hello"`;
        spinner.text = `Warmed up ${model}`;
      } catch (error) {
        log.warn(`Failed to warm up ${model}: ${error.message}`);
      }
    }
    
    spinner.succeed(`${config.name} models warmed up`);
  }
}

async function checkServiceHealth(serviceName, config) {
  try {
    switch (serviceName) {
      case 'postgresql':
        return await healthCheckers.postgresql();
      case 'redis':
        return await healthCheckers.redis();
      default:
        if (config.healthUrl) {
          return await healthCheckers.http(config.healthUrl);
        }
        return { healthy: false, error: 'No health check available' };
    }
  } catch (error) {
    return { healthy: false, error: error.message };
  }
}

async function performSystemHealthCheck() {
  const results = new Map();
  
  for (const [name, config] of Object.entries(PROD_CONFIG.services)) {
    const health = await checkServiceHealth(name, config);
    results.set(name, {
      ...health,
      service: config.name,
      port: config.port
    });
  }
  
  return results;
}

// Production monitoring system
class ProductionMonitor {
  constructor() {
    this.metrics = new Map();
    this.alerts = new Map();
    this.interval = null;
  }

  start() {
    log.info('📊 Starting production monitoring system...');
    
    this.interval = setInterval(async () => {
      await this.collectMetrics();
      await this.checkAlerts();
      await this.logMetrics();
    }, PROD_CONFIG.monitoring.interval);
  }

  stop() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
      log.info('📊 Monitoring system stopped');
    }
  }

  async collectMetrics() {
    const healthResults = await performSystemHealthCheck();
    
    for (const [service, health] of healthResults) {
      const metrics = this.metrics.get(service) || [];
      metrics.push({
        timestamp: Date.now(),
        healthy: health.healthy,
        latency: health.latency || null,
        error: health.error || null
      });
      
      // Keep only last 100 metrics per service
      if (metrics.length > 100) {
        metrics.shift();
      }
      
      this.metrics.set(service, metrics);
    }
  }

  async checkAlerts() {
    for (const [service, metrics] of this.metrics) {
      const recent = metrics.slice(-PROD_CONFIG.monitoring.alertThreshold);
      const failures = recent.filter(m => !m.healthy).length;
      
      if (failures >= PROD_CONFIG.monitoring.alertThreshold) {
        const alertKey = `${service}_failures`;
        if (!this.alerts.has(alertKey)) {
          log.error(`🚨 ALERT: ${service} has failed ${failures} consecutive health checks`);
          this.alerts.set(alertKey, Date.now());
          
          // Attempt to restart service
          await this.attemptServiceRestart(service);
        }
      } else if (failures === 0 && this.alerts.has(`${service}_failures`)) {
        log.success(`✅ RESOLVED: ${service} health checks are passing again`);
        this.alerts.delete(`${service}_failures`);
      }
    }
  }

  async attemptServiceRestart(serviceName) {
    const config = PROD_CONFIG.services[serviceName];
    if (!config) return;
    
    log.warn(`🔄 Attempting to restart ${config.name}...`);
    
    try {
      // Kill existing service if running
      const service = state.services.get(serviceName);
      if (service && service.process) {
        service.process.kill('SIGTERM');
        await sleep(5000);
      }
      
      // Restart service
      await startProductionService(serviceName, config);
      log.success(`🔄 Successfully restarted ${config.name}`);
    } catch (error) {
      log.error(`🔄 Failed to restart ${config.name}: ${error.message}`);
    }
  }

  async logMetrics() {
    // Log summary metrics every 5 minutes
    if (Date.now() % (5 * 60 * 1000) < PROD_CONFIG.monitoring.interval) {
      const summary = this.generateMetricsSummary();
      log.info(`📊 System Metrics: ${summary}`);
    }
  }

  generateMetricsSummary() {
    const summaries = [];
    
    for (const [service, metrics] of this.metrics) {
      const recent = metrics.slice(-20); // Last 20 checks
      const healthy = recent.filter(m => m.healthy).length;
      const avgLatency = recent
        .filter(m => m.latency)
        .reduce((sum, m) => sum + m.latency, 0) / recent.length;
      
      summaries.push(`${service}: ${healthy}/${recent.length} (${avgLatency.toFixed(0)}ms)`);
    }
    
    return summaries.join(', ');
  }
}

// Main production orchestration
async function main() {
  console.log(chalk.cyan.bold('🏭 YoRHa Legal AI - Production Orchestrator\n'));
  
  // Parse command line arguments
  program
    .option('-m, --monitor', 'Enable production monitoring', true)
    .option('--no-monitor', 'Disable monitoring')
    .option('-b, --build', 'Build frontend before starting')
    .option('-d, --daemon', 'Run as daemon process')
    .parse();

  const options = program.opts();
  
  // Build frontend if requested
  if (options.build) {
    const buildSpinner = ora('🔨 Building SvelteKit frontend...').start();
    try {
      await $`cd sveltekit-frontend && npm run build`;
      buildSpinner.succeed('Frontend built successfully');
    } catch (error) {
      buildSpinner.fail(`Frontend build failed: ${error.message}`);
      process.exit(1);
    }
  }
  
  // Check production prerequisites
  await checkProductionPrerequisites();
  
  // Start services in priority order
  const services = Object.entries(PROD_CONFIG.services)
    .sort(([, a], [, b]) => a.priority - b.priority);
  
  log.info(`🚀 Starting ${services.length} production services...`);
  
  for (const [serviceName, config] of services) {
    await startProductionService(serviceName, config);
    await sleep(3000); // Stagger production starts
  }
  
  // Final system health check
  console.log(chalk.cyan('\n🏥 Production Health Check:'));
  const healthResults = await performSystemHealthCheck();
  
  let allHealthy = true;
  for (const [name, result] of healthResults) {
    const status = result.healthy ? 
      chalk.green(`✓ HEALTHY (${result.latency}ms)`) : 
      chalk.red('✗ UNHEALTHY');
    
    console.log(`  ${result.service.padEnd(25)} ${status.padEnd(25)} Port: ${result.port}`);
    if (!result.healthy) allHealthy = false;
  }
  
  if (!allHealthy) {
    log.error('❌ Production startup completed with health issues');
    process.exit(1);
  }
  
  // Show production URLs
  console.log(chalk.cyan('\n🌍 Production URLs:'));
  console.log(`  Frontend:     ${chalk.blue('http://localhost:3000')}`);
  console.log(`  API Gateway:  ${chalk.blue('http://localhost:8080')}`);
  console.log(`  gRPC:         ${chalk.blue('localhost:50051')}`);
  console.log(`  Monitoring:   ${chalk.blue('http://localhost:8080/metrics')}`);
  
  log.success('🚀 Production environment started successfully!');
  
  // Start monitoring system
  if (options.monitor) {
    const monitor = new ProductionMonitor();
    monitor.start();
    
    // Graceful shutdown handler
    const shutdown = async (signal) => {
      log.info(`\n🛑 Received ${signal}, performing graceful shutdown...`);
      monitor.stop();
      
      // Stop all services
      for (const [name, service] of state.services) {
        if (service.process) {
          log.info(`Stopping ${name}...`);
          service.process.kill('SIGTERM');
        }
      }
      
      // Wait for services to stop
      await sleep(10000);
      
      log.success('🛑 Graceful shutdown completed');
      process.exit(0);
    };
    
    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('SIGTERM', () => shutdown('SIGTERM'));
    
    // Keep process alive if not daemon
    if (!options.daemon) {
      log.info('📊 Monitoring active. Press Ctrl+C to stop.');
      await new Promise(() => {});
    }
  }
}

async function checkProductionPrerequisites() {
  const spinner = ora('🔍 Checking production prerequisites...').start();
  
  const checks = [
    { name: 'Built Frontend', check: () => fs.existsSync('./sveltekit-frontend/build') },
    { name: 'Go Service Binary', check: () => fs.existsSync('./legal-ai-server.exe') },
    { name: 'PostgreSQL Service', check: () => $`sc query postgresql-x64-17` },
    { name: 'Production Config', check: () => fs.existsSync('./production.env') }
  ];
  
  const warnings = [];
  
  for (const check of checks) {
    try {
      await check.check();
    } catch {
      warnings.push(check.name);
    }
  }
  
  if (warnings.length > 0) {
    spinner.warn(`Production warnings: ${warnings.join(', ')}`);
    log.warn('🚨 Some production components may not be optimally configured');
  } else {
    spinner.succeed('Production prerequisites verified');
  }
}

// Helper functions
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Run the production orchestrator
main().catch(error => {
  log.error(`Production orchestrator failed: ${error.message}`);
  process.exit(1);
});