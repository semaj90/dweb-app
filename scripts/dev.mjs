#!/usr/bin/env node

/**
 * YoRHa Legal AI - Development Environment Orchestrator
 * 
 * Orchestrates all services for development:
 * - PostgreSQL with pgvector
 * - Redis for caching
 * - Ollama LLM service
 * - Go microservice with gRPC/HTTP2
 * - SvelteKit frontend
 * - Qdrant vector database (optional)
 * 
 * @author YoRHa Legal AI Team
 * @version 2.0.0
 */

import 'zx/globals';
import chalk from 'chalk';
import ora from 'ora';
import { WebSocket } from 'ws';
import fetch from 'node-fetch';

// Configuration
const CONFIG = {
  services: {
    postgresql: {
      name: 'PostgreSQL + pgvector',
      port: 5432,
      healthUrl: null, // Custom health check
      command: '"C:\\Program Files\\PostgreSQL\\17\\bin\\pg_isready.exe" -h localhost -p 5432',
      priority: 1
    },
    redis: {
      name: 'Redis Cache',
      port: 6379,
      healthUrl: null, // Custom health check
      command: '.\\redis-windows\\redis-server.exe',
      cwd: process.cwd(),
      priority: 2
    },
    ollama: {
      name: 'Ollama LLM',
      port: 11434,
      healthUrl: 'http://localhost:11434/api/version',
      command: 'ollama serve',
      priority: 3
    },
    qdrant: {
      name: 'Qdrant Vector DB',
      port: 6333,
      healthUrl: 'http://localhost:6333/health',
      command: '.\\qdrant-windows\\qdrant.exe',
      cwd: process.cwd(),
      priority: 4,
      optional: true
    },
    goService: {
      name: 'Go Microservice',
      port: 8080,
      grpcPort: 50051,
      healthUrl: 'http://localhost:8080/api/health',
      command: '.\\legal-ai-server.exe',
      cwd: process.cwd(),
      priority: 5
    },
    sveltekit: {
      name: 'SvelteKit Frontend',
      port: 5173,
      healthUrl: 'http://localhost:5173/',
      command: 'npm run dev',
      cwd: path.join(process.cwd(), 'sveltekit-frontend'),
      priority: 6
    }
  }
};

// Global state
const state = {
  services: new Map(),
  healthChecks: new Map(),
  startTime: Date.now()
};

// Utility functions
const log = {
  info: (msg) => console.log(chalk.blue('ℹ'), msg),
  success: (msg) => console.log(chalk.green('✓'), msg),
  error: (msg) => console.log(chalk.red('✗'), msg),
  warn: (msg) => console.log(chalk.yellow('⚠'), msg),
  debug: (msg) => process.env.DEBUG && console.log(chalk.gray('🔍'), msg)
};

// Health check functions
const healthCheckers = {
  async postgresql() {
    try {
      const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\pg_isready.exe" -h localhost -p 5432`;
      return result.exitCode === 0;
    } catch {
      return false;
    }
  },

  async redis() {
    try {
      // Try to connect to Redis using redis-cli
      const result = await $`echo "ping" | .\\redis-windows\\redis-cli.exe -h localhost -p 6379`;
      return result.stdout.includes('PONG');
    } catch {
      return false;
    }
  },

  async http(url) {
    try {
      const response = await fetch(url, { 
        timeout: 5000,
        signal: AbortSignal.timeout(5000)
      });
      return response.ok;
    } catch {
      return false;
    }
  }
};

// Service management
async function startService(serviceName, config) {
  const spinner = ora(`Starting ${config.name}...`).start();
  
  try {
    // Check if already running
    if (await isServiceHealthy(serviceName, config)) {
      spinner.succeed(`${config.name} already running`);
      return;
    }

    // Start the service
    const options = {
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe']
    };

    if (config.cwd) {
      options.cwd = config.cwd;
    }

    let proc;
    if (config.command.includes('.exe') || config.command.includes('ollama')) {
      // For Windows executables and Ollama, use spawn
      proc = spawn(config.command.split(' ')[0], config.command.split(' ').slice(1), options);
    } else {
      // For npm commands, use shell
      proc = spawn('cmd', ['/c', config.command], { ...options, shell: true });
    }

    if (proc.pid) {
      state.services.set(serviceName, { process: proc, config });
      
      // Wait for service to be healthy
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds
      
      while (attempts < maxAttempts) {
        await sleep(1000);
        if (await isServiceHealthy(serviceName, config)) {
          spinner.succeed(`${config.name} started successfully on port ${config.port}`);
          return;
        }
        attempts++;
      }
      
      spinner.warn(`${config.name} started but health check failed`);
    } else {
      throw new Error('Failed to start process');
    }
  } catch (error) {
    spinner.fail(`Failed to start ${config.name}: ${error.message}`);
    throw error;
  }
}

async function isServiceHealthy(serviceName, config) {
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
        return false;
    }
  } catch {
    return false;
  }
}

async function checkAllServices() {
  const results = new Map();
  
  for (const [name, config] of Object.entries(CONFIG.services)) {
    if (config.optional && !state.services.has(name)) {
      results.set(name, { status: 'skipped', healthy: false });
      continue;
    }
    
    const healthy = await isServiceHealthy(name, config);
    results.set(name, { 
      status: healthy ? 'healthy' : 'unhealthy', 
      healthy,
      port: config.port
    });
  }
  
  return results;
}

// Main orchestration
async function main() {
  console.log(chalk.cyan.bold('🤖 YoRHa Legal AI - Development Environment Orchestrator\n'));
  
  // Check prerequisites
  await checkPrerequisites();
  
  // Start services in priority order
  const sortedServices = Object.entries(CONFIG.services)
    .filter(([_, config]) => !config.optional || process.argv.includes('--include-optional'))
    .sort(([, a], [, b]) => a.priority - b.priority);
  
  log.info(`Starting ${sortedServices.length} services in sequence...`);
  
  for (const [serviceName, config] of sortedServices) {
    await startService(serviceName, config);
    await sleep(2000); // Stagger starts
  }
  
  // Final health check
  console.log(chalk.cyan('\n📊 System Health Check:'));
  const healthResults = await checkAllServices();
  
  for (const [name, result] of healthResults) {
    const config = CONFIG.services[name];
    const status = result.healthy ? 
      chalk.green('✓ HEALTHY') : 
      result.status === 'skipped' ? chalk.yellow('⊘ SKIPPED') : chalk.red('✗ UNHEALTHY');
    
    console.log(`  ${config.name.padEnd(25)} ${status.padEnd(15)} Port: ${config.port || 'N/A'}`);
  }
  
  // Show URLs
  console.log(chalk.cyan('\n🌐 Service URLs:'));
  console.log(`  Frontend:     ${chalk.blue('http://localhost:5173')}`);
  console.log(`  Go API:       ${chalk.blue('http://localhost:8080')}`);
  console.log(`  Ollama:       ${chalk.blue('http://localhost:11434')}`);
  console.log(`  Qdrant:       ${chalk.blue('http://localhost:6333')}`);
  
  // Development tips
  console.log(chalk.cyan('\n💡 Development Commands:'));
  console.log(`  Status:       ${chalk.gray('npm run status')}`);
  console.log(`  Health:       ${chalk.gray('npm run health')}`);
  console.log(`  Logs:         ${chalk.gray('npm run logs')}`);
  console.log(`  Stop:         ${chalk.gray('npm run stop')}`);
  
  log.success('🚀 Development environment ready!');
  
  // Keep script running and monitor services
  if (!process.argv.includes('--no-monitor')) {
    await monitorServices();
  }
}

async function checkPrerequisites() {
  const spinner = ora('Checking prerequisites...').start();
  
  const checks = [
    { name: 'Node.js', check: () => $.which('node') },
    { name: 'PostgreSQL', check: () => fs.existsSync('C:\\Program Files\\PostgreSQL\\17\\bin\\pg_isready.exe') },
    { name: 'Go Service', check: () => fs.existsSync('./legal-ai-server.exe') },
    { name: 'Redis', check: () => fs.existsSync('./redis-windows/redis-server.exe') },
    { name: 'Ollama', check: () => $.which('ollama') }
  ];
  
  const failed = [];
  
  for (const check of checks) {
    try {
      await check.check();
    } catch {
      failed.push(check.name);
    }
  }
  
  if (failed.length > 0) {
    spinner.fail(`Missing prerequisites: ${failed.join(', ')}`);
    console.log(chalk.yellow('📋 Setup Guide:'));
    console.log('  1. Install PostgreSQL 17 with pgvector extension');
    console.log('  2. Download Redis for Windows');
    console.log('  3. Install Ollama from https://ollama.com');
    console.log('  4. Build Go microservice: go build -o legal-ai-server.exe');
    process.exit(1);
  }
  
  spinner.succeed('Prerequisites check passed');
}

async function monitorServices() {
  log.info('🔍 Starting service monitoring (Ctrl+C to stop)...');
  
  const monitorInterval = setInterval(async () => {
    const healthResults = await checkAllServices();
    const unhealthy = Array.from(healthResults.entries())
      .filter(([, result]) => !result.healthy && result.status !== 'skipped');
    
    if (unhealthy.length > 0) {
      log.warn(`Unhealthy services detected: ${unhealthy.map(([name]) => name).join(', ')}`);
    }
  }, 30000); // Check every 30 seconds
  
  // Graceful shutdown
  process.on('SIGINT', async () => {
    clearInterval(monitorInterval);
    log.info('\n🛑 Shutting down monitoring...');
    
    if (process.argv.includes('--stop-on-exit')) {
      const stopScript = path.join(path.dirname(fileURLToPath(import.meta.url)), 'stop.mjs');
      await $`zx ${stopScript}`;
    }
    
    process.exit(0);
  });
  
  // Keep the process alive
  await new Promise(() => {});
}

// Helper functions
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Handle CLI arguments
if (process.argv.includes('--help')) {
  console.log(`
YoRHa Legal AI Development Orchestrator

Usage: npm run dev [options]

Options:
  --include-optional    Include optional services (Qdrant)
  --no-monitor         Don't monitor services after startup
  --stop-on-exit       Stop all services when script exits
  --help               Show this help message

Examples:
  npm run dev                    # Start core services
  npm run dev --include-optional # Start all services including Qdrant
  npm run dev --no-monitor       # Start services and exit
`);
  process.exit(0);
}

// Run the orchestrator
main().catch(error => {
  log.error(`Orchestrator failed: ${error.message}`);
  process.exit(1);
});