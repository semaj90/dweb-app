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

import chalk from 'chalk';
import fetch from 'node-fetch';
import ora from 'ora';
import { spawn } from 'node:child_process';
import net from 'node:net';
import 'zx/globals';
import { fileURLToPath } from 'url';
import path from 'path';

// Configuration
const CONFIG = {
  services: {
    postgresql: {
      name: 'PostgreSQL + pgvector',
      port: 5432,
      healthUrl: null, // Custom health check
  command: '', // External service (managed by OS); do not attempt to start here
  managed: false,
      priority: 1
    },
    redis: {
      name: 'Redis Cache',
      port: 6379,
      healthUrl: null, // Custom health check
  command: '', // External or optional; skip start by default
  managed: false,
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
    goOllama: {
      name: 'Go Ollama SIMD Service',
      port: 8081,
      healthUrl: 'http://localhost:8081/health',
  command: 'go run ./cmd/go-ollama-simd',
      cwd: path.join(process.cwd(), 'go-microservice'),
      priority: 5
    },
    goService: {
      name: 'Go Microservice',
      port: 8080,
      grpcPort: 50051,
      healthUrl: 'http://localhost:8080/health',
      // Run the enhanced gRPC/HTTP2 server directly in dev
      command: 'go run enhanced-grpc-legal-server.go',
      // Execute from the go-microservice directory
      cwd: path.join(process.cwd(), 'go-microservice'),
      priority: 6
    },
    sveltekit: {
      name: 'SvelteKit Frontend',
      port: 5173,
      healthUrl: 'http://localhost:5173/',
      command: 'npm run dev',
      cwd: path.join(process.cwd(), 'sveltekit-frontend'),
      priority: 7
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
  return await tcpOpen('localhost', 5432, 1000);
  },

  async redis() {
  return await tcpOpen('localhost', 6379, 1000);
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

function tcpOpen(host, port, timeoutMs = 1000) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    const onDone = (ok) => {
      try { socket.destroy(); } catch {}
      resolve(ok);
    };
    socket.setTimeout(timeoutMs);
    socket.once('connect', () => onDone(true));
    socket.once('timeout', () => onDone(false));
    socket.once('error', () => onDone(false));
    socket.connect(port, host);
  });
}

// Service management
async function startService(serviceName, config) {
  const spinner = ora(`Starting ${config.name}...`).start();

  try {
    // Check if already running
    if (await isServiceHealthy(serviceName, config)) {
      spinner.succeed(`${config.name} already running`);
      return;
    }

    // Skip starting for externally managed services
    if (config.managed === false || !config.command) {
      spinner.info(`${config.name} is external or unmanaged; not starting from orchestrator`);
      return;
    }

    // Start the service
    const options = {
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: true
    };

    if (config.cwd) {
      options.cwd = config.cwd;
    }

  const proc = spawn(config.command, options);

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
  let sortedServices = Object.entries(CONFIG.services)
    .filter(([_, config]) => !config.optional || process.argv.includes('--include-optional'))
    .sort(([, a], [, b]) => a.priority - b.priority);

  if (process.argv.includes('--frontend-only')) {
    sortedServices = sortedServices.filter(([name]) => name === 'sveltekit');
  }

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

  const force = process.argv.includes('--force') || process.env.DEV_FORCE === '1';
  const missing = [];
  const warns = [];

  // Node is guaranteed since we're running under Node
  const nodeOk = !!process.version;

  // Ports via TCP checks
  const [pgOk, redisOk] = await Promise.all([
    tcpOpen('localhost', 5432, 500),
    tcpOpen('localhost', 6379, 500)
  ]);

  // Go toolchain (optional)
  let goOk = true;
  try {
    await $`go version`;
  } catch {
    goOk = false;
    warns.push('Go toolchain');
  }

  // Ollama: HTTP check
  let ollamaOk = false;
  try {
    const res = await fetch('http://localhost:11434/api/version', { timeout: 2000, signal: AbortSignal.timeout(2000) });
    ollamaOk = res.ok;
  } catch {
    warns.push('Ollama');
  }

  if (!nodeOk) missing.push('Node.js');
  if (!pgOk) warns.push('PostgreSQL (5432 not reachable)');
  if (!redisOk) warns.push('Redis (6379 not reachable)');

  if (missing.length > 0 && !force) {
    spinner.fail(`Missing prerequisites: ${missing.join(', ')}`);
    console.log(chalk.yellow('📋 Setup Guide:'));
    console.log('  1. Install PostgreSQL 17 with pgvector extension');
    console.log('  2. Download Redis for Windows');
    console.log('  3. Install Ollama from https://ollama.com');
    console.log('  4. Build Go microservice: go build -o legal-ai-server.exe');
    console.log(chalk.gray('    Tip: use --force to bypass checks.'));
    process.exit(1);
  }

  if (warns.length) {
    spinner.warn(`Proceeding with warnings: ${warns.join(', ')}`);
  } else {
    spinner.succeed('Prerequisites check passed');
  }
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