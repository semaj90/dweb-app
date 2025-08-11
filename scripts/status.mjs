#!/usr/bin/env node

/**
 * YoRHa Legal AI - System Status Monitor
 *
 * Real-time monitoring dashboard for all services:
 * - Service health and performance metrics
 * - Resource usage monitoring
 * - Network connectivity status
 * - Interactive dashboard with live updates
 *
 * @author YoRHa Legal AI Team
 * @version 2.0.0
 */

import chalk from 'chalk';
import { program } from 'commander';
import fetch from 'node-fetch';
import ora from 'ora';
import 'zx/globals';

// Status monitoring configuration
const STATUS_CONFIG = {
  services: {
    postgresql: {
      name: 'PostgreSQL + pgvector',
      port: 5432,
      healthUrl: null,
      priority: 'critical',
      category: 'database',
      customCheck: async () => {
        try {
          const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\pg_isready.exe" -h localhost -p 5432`;
          return {
            status: result.exitCode === 0 ? 'healthy' : 'unhealthy',
            latency: result.exitCode === 0 ? await measureLatency('postgresql') : null,
            details: result.stdout.trim()
          };
        } catch (error) {
          return { status: 'error', error: error.message };
        }
      }
    },
    redis: {
      name: 'Redis Cache',
      port: 6379,
      healthUrl: null,
      priority: 'high',
      category: 'cache',
      customCheck: async () => {
        try {
          const start = Date.now();
          const result = await $`echo "ping" | .\\redis-windows\\redis-cli.exe -h localhost -p 6379`;
          const latency = Date.now() - start;

          if (result.stdout.includes('PONG')) {
            // Get additional Redis info
            const infoResult = await $`echo "info memory" | .\\redis-windows\\redis-cli.exe -h localhost -p 6379`;
            return {
              status: 'healthy',
              latency,
              details: 'Connected',
              memory: parseRedisMemory(infoResult.stdout)
            };
          } else {
            return { status: 'unhealthy', error: 'No PONG response' };
          }
        } catch (error) {
          return { status: 'error', error: error.message };
        }
      }
    },
    ollama: {
      name: 'Ollama LLM Service',
      port: 11434,
      healthUrl: 'http://localhost:11434/api/version',
      priority: 'critical',
      category: 'ai',
      extendedCheck: async () => {
        try {
          // Check models
          const modelsResult = await fetch('http://localhost:11434/api/tags');
          const models = modelsResult.ok ? await modelsResult.json() : null;

          // Check GPU status
          const gpuStatus = await checkGPUStatus();

          return { models, gpu: gpuStatus };
        } catch (error) {
          return { error: error.message };
        }
      }
    },
    qdrant: {
      name: 'Qdrant Vector DB',
      port: 6333,
      healthUrl: 'http://localhost:6333/health',
      priority: 'medium',
      category: 'database',
      extendedCheck: async () => {
        try {
          const collectionsResult = await fetch('http://localhost:6333/collections');
          const collections = collectionsResult.ok ? await collectionsResult.json() : null;
          return { collections };
        } catch (error) {
          return { error: error.message };
        }
      }
    },
    goOllama: {
      name: 'Go Ollama SIMD Service',
      port: 8081,
      healthUrl: 'http://localhost:8081/health',
      priority: 'high',
      category: 'ai',
      extendedCheck: async () => {
        try {
          const endpoints = ['/health', '/api/simd/capabilities'];
          const status = {};
          for (const ep of endpoints) {
            const res = await fetch(`http://localhost:8081${ep}`);
            status[ep] = res.ok;
          }
          return { endpoints: status };
        } catch (error) {
          return { error: error.message };
        }
      }
    },
    goService: {
      name: 'Go Legal AI Service',
      port: 8080,
      grpcPort: 50051,
  healthUrl: 'http://localhost:8080/health',
      priority: 'critical',
      category: 'api',
      extendedCheck: async () => {
        try {
          // Check API endpoints
          const endpoints = [
    '/health',
    '/api/v1/documents',
    '/api/v1/search/semantic',
    '/metrics'
          ];

          const endpointStatus = {};
          for (const endpoint of endpoints) {
            try {
              const response = await fetch(`http://localhost:8080${endpoint}`, {
                timeout: 5000,
                signal: AbortSignal.timeout(5000)
              });
              endpointStatus[endpoint] = {
                status: response.status,
                ok: response.ok,
                latency: response.headers.get('x-response-time')
              };
            } catch (error) {
              endpointStatus[endpoint] = { error: error.message };
            }
          }

          return { endpoints: endpointStatus };
        } catch (error) {
          return { error: error.message };
        }
      }
    },
    sveltekit: {
      name: 'SvelteKit Frontend',
      port: process.env.NODE_ENV === 'production' ? 3000 : 5173,
      healthUrl: process.env.NODE_ENV === 'production' ?
        'http://localhost:3000/' : 'http://localhost:5173/',
      priority: 'high',
      category: 'frontend',
      extendedCheck: async () => {
        try {
          // Check key frontend routes
          const routes = ['/', '/health', '/api/health'];
          const routeStatus = {};

          for (const route of routes) {
            try {
              const baseUrl = process.env.NODE_ENV === 'production' ?
                'http://localhost:3000' : 'http://localhost:5173';
              const response = await fetch(`${baseUrl}${route}`, {
                timeout: 10000,
                signal: AbortSignal.timeout(10000)
              });
              routeStatus[route] = {
                status: response.status,
                ok: response.ok,
                size: response.headers.get('content-length')
              };
            } catch (error) {
              routeStatus[route] = { error: error.message };
            }
          }

          return { routes: routeStatus };
        } catch (error) {
          return { error: error.message };
        }
      }
    }
  }
};

// Global monitoring state
const monitorState = {
  metrics: new Map(),
  history: new Map(),
  alerts: new Map(),
  startTime: Date.now()
};

// Enhanced logging with colors and timestamps
const log = {
  timestamp: () => new Date().toISOString(),
  info: (msg) => console.log(`[${log.timestamp()}]`, chalk.blue('ℹ'), msg),
  success: (msg) => console.log(`[${log.timestamp()}]`, chalk.green('✓'), msg),
  error: (msg) => console.log(`[${log.timestamp()}]`, chalk.red('✗'), msg),
  warn: (msg) => console.log(`[${log.timestamp()}]`, chalk.yellow('⚠'), msg),
  debug: (msg) => process.env.DEBUG && console.log(`[${log.timestamp()}]`, chalk.gray('🔍'), msg)
};

// Utility functions
async function measureLatency(service) {
  const start = Date.now();
  try {
    switch (service) {
      case 'postgresql':
        await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT 1;" -t`;
        break;
      default:
        return null;
    }
    return Date.now() - start;
  } catch {
    return null;
  }
}

function parseRedisMemory(infoOutput) {
  const lines = infoOutput.split('\n');
  const memory = {};

  for (const line of lines) {
    if (line.startsWith('used_memory_human:')) {
      memory.used = line.split(':')[1].trim();
    }
    if (line.startsWith('used_memory_peak_human:')) {
      memory.peak = line.split(':')[1].trim();
    }
  }

  return memory;
}

async function checkGPUStatus() {
  try {
    const result = await $`nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`;
    const lines = result.stdout.trim().split('\n');

    return lines.map(line => {
      const [name, temp, util, memUsed, memTotal] = line.split(', ');
      return {
        name: name.trim(),
        temperature: parseInt(temp),
        utilization: parseInt(util),
        memory: {
          used: parseInt(memUsed),
          total: parseInt(memTotal),
          percent: Math.round((parseInt(memUsed) / parseInt(memTotal)) * 100)
        }
      };
    });
  } catch {
    return [{ error: 'NVIDIA GPU not detected or nvidia-smi not available' }];
  }
}

// Health checking functions
async function checkServiceHealth(serviceName, config) {
  const start = Date.now();

  try {
    let basicHealth = { status: 'unknown' };

    // Custom health check
    if (config.customCheck) {
      basicHealth = await config.customCheck();
    }
    // HTTP health check
    else if (config.healthUrl) {
      try {
        const response = await fetch(config.healthUrl, {
          timeout: 10000,
          signal: AbortSignal.timeout(10000)
        });
        basicHealth = {
          status: response.ok ? 'healthy' : 'unhealthy',
          latency: Date.now() - start,
          httpStatus: response.status,
          details: response.ok ? 'HTTP OK' : `HTTP ${response.status}`
        };
      } catch (error) {
        basicHealth = {
          status: 'error',
          error: error.message,
          latency: Date.now() - start
        };
      }
    }

    // Extended health check
    let extended = {};
    if (config.extendedCheck) {
      try {
        extended = await config.extendedCheck();
      } catch (error) {
        extended = { extendedCheckError: error.message };
      }
    }

    return {
      service: serviceName,
      name: config.name,
      priority: config.priority,
      category: config.category,
      port: config.port,
      grpcPort: config.grpcPort,
      timestamp: Date.now(),
      checkDuration: Date.now() - start,
      ...basicHealth,
      ...extended
    };

  } catch (error) {
    return {
      service: serviceName,
      name: config.name,
      status: 'error',
      error: error.message,
      timestamp: Date.now(),
      checkDuration: Date.now() - start
    };
  }
}

// System resource monitoring
async function getSystemResources() {
  try {
    const [cpuResult, memResult, diskResult] = await Promise.allSettled([
      $`wmic cpu get loadpercentage /value`,
      $`wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value`,
      $`wmic logicaldisk get size,freespace,caption /format:csv`
    ]);

    let cpu = null;
    if (cpuResult.status === 'fulfilled') {
      const cpuLine = cpuResult.value.stdout.split('\n').find(line => line.includes('LoadPercentage='));
      if (cpuLine) {
        cpu = parseInt(cpuLine.split('=')[1]);
      }
    }

    let memory = null;
    if (memResult.status === 'fulfilled') {
      const lines = memResult.value.stdout.split('\n');
      const totalLine = lines.find(line => line.includes('TotalVisibleMemorySize='));
      const freeLine = lines.find(line => line.includes('FreePhysicalMemory='));

      if (totalLine && freeLine) {
        const total = parseInt(totalLine.split('=')[1]);
        const free = parseInt(freeLine.split('=')[1]);
        const used = total - free;

        memory = {
          total: Math.round(total / 1024), // MB
          used: Math.round(used / 1024),
          free: Math.round(free / 1024),
          percent: Math.round((used / total) * 100)
        };
      }
    }

    let disks = [];
    if (diskResult.status === 'fulfilled') {
      const lines = diskResult.value.stdout.split('\n').slice(1);
      for (const line of lines) {
        const parts = line.split(',');
        if (parts.length >= 4 && parts[1]) {
          const caption = parts[1].trim();
          const freespace = parseInt(parts[2]) || 0;
          const size = parseInt(parts[3]) || 0;

          if (size > 0) {
            disks.push({
              drive: caption,
              total: Math.round(size / 1024 / 1024 / 1024), // GB
              free: Math.round(freespace / 1024 / 1024 / 1024),
              used: Math.round((size - freespace) / 1024 / 1024 / 1024),
              percent: Math.round(((size - freespace) / size) * 100)
            });
          }
        }
      }
    }

    return { cpu, memory, disks };
  } catch (error) {
    return { error: error.message };
  }
}

// Dashboard rendering functions
function renderServiceStatus(health) {
  const statusColors = {
    healthy: chalk.green,
    unhealthy: chalk.red,
    error: chalk.red,
    unknown: chalk.yellow
  };

  const priorityIcons = {
    critical: '🔴',
    high: '🟡',
    medium: '🔵',
    low: '⚪'
  };

  const statusColor = statusColors[health.status] || chalk.gray;
  const statusText = health.status.toUpperCase();
  const icon = priorityIcons[health.priority] || '⚫';

  let line = `${icon} ${health.name.padEnd(25)} ${statusColor(statusText.padEnd(10))}`;

  if (health.port) {
    line += ` Port:${health.port}`;
    if (health.grpcPort) {
      line += `,${health.grpcPort}`;
    }
  }

  if (health.latency) {
    const latencyColor = health.latency < 100 ? chalk.green :
                        health.latency < 500 ? chalk.yellow : chalk.red;
    line += ` ${latencyColor(`${health.latency}ms`)}`;
  }

  console.log(line);

  // Show additional details for unhealthy services
  if (health.status !== 'healthy' && (health.error || health.details)) {
    const errorMsg = health.error || health.details;
    console.log(`    ${chalk.gray('└─')} ${chalk.red(errorMsg)}`);
  }

  // Show extended information if available
  if (health.models && health.models.models) {
    const modelCount = health.models.models.length;
    console.log(`    ${chalk.gray('└─')} ${chalk.blue(`${modelCount} models loaded`)}`);
  }

  if (health.gpu && health.gpu[0] && !health.gpu[0].error) {
    const gpu = health.gpu[0];
    const tempColor = gpu.temperature < 80 ? chalk.green :
                     gpu.temperature < 90 ? chalk.yellow : chalk.red;
    console.log(`    ${chalk.gray('└─')} GPU: ${tempColor(`${gpu.temperature}°C`)} ${gpu.utilization}% Memory:${gpu.memory.percent}%`);
  }

  if (health.memory) {
    console.log(`    ${chalk.gray('└─')} Memory: ${health.memory.used || 'N/A'} peak:${health.memory.peak || 'N/A'}`);
  }
}

function renderSystemResources(resources) {
  console.log(chalk.cyan('\n📊 System Resources:'));

  if (resources.cpu !== null) {
    const cpuColor = resources.cpu < 70 ? chalk.green :
                    resources.cpu < 90 ? chalk.yellow : chalk.red;
    console.log(`  CPU Usage:        ${cpuColor(`${resources.cpu}%`)}`);
  }

  if (resources.memory) {
    const memColor = resources.memory.percent < 80 ? chalk.green :
                    resources.memory.percent < 95 ? chalk.yellow : chalk.red;
    console.log(`  Memory Usage:     ${memColor(`${resources.memory.percent}%`)} (${resources.memory.used}/${resources.memory.total} MB)`);
  }

  if (resources.disks && resources.disks.length > 0) {
    resources.disks.forEach(disk => {
      const diskColor = disk.percent < 80 ? chalk.green :
                       disk.percent < 95 ? chalk.yellow : chalk.red;
      console.log(`  Disk ${disk.drive}:         ${diskColor(`${disk.percent}%`)} (${disk.used}/${disk.total} GB)`);
    });
  }

  if (resources.error) {
    console.log(`  ${chalk.red('System resource monitoring failed')}`);
  }
}

function renderDashboardHeader() {
  const uptime = Math.round((Date.now() - monitorState.startTime) / 1000);
  const uptimeStr = `${Math.floor(uptime / 60)}m ${uptime % 60}s`;

  console.clear();
  console.log(chalk.cyan.bold('🤖 YoRHa Legal AI - System Status Dashboard'));
  console.log(chalk.gray(`   Monitoring since ${new Date(monitorState.startTime).toLocaleTimeString()}`));
  console.log(chalk.gray(`   Uptime: ${uptimeStr}`));
  console.log(chalk.gray(`   Last update: ${new Date().toLocaleTimeString()}\n`));
}

// Main status monitoring functions
async function performFullHealthCheck() {
  const healthResults = new Map();

  // Check all services in parallel
  const healthPromises = Object.entries(STATUS_CONFIG.services).map(
    async ([serviceName, config]) => {
      const health = await checkServiceHealth(serviceName, config);
      healthResults.set(serviceName, health);
    }
  );

  await Promise.allSettled(healthPromises);

  // Get system resources
  const systemResources = await getSystemResources();

  return { services: healthResults, system: systemResources };
}

async function runInteractiveDashboard() {
  log.info('🖥️  Starting interactive dashboard (Press Ctrl+C to stop)...');

  const refreshInterval = 5000; // 5 seconds

  const updateDashboard = async () => {
    try {
      const { services, system } = await performFullHealthCheck();

      renderDashboardHeader();

      // Group services by category
      const categories = {
        database: [],
        cache: [],
        ai: [],
        api: [],
        frontend: []
      };

      for (const [, health] of services) {
        const category = health.category || 'other';
        if (categories[category]) {
          categories[category].push(health);
        }
      }

      // Render services by category
      for (const [categoryName, categoryServices] of Object.entries(categories)) {
        if (categoryServices.length > 0) {
          console.log(chalk.cyan(`\n📋 ${categoryName.charAt(0).toUpperCase() + categoryName.slice(1)} Services:`));
          categoryServices.forEach(renderServiceStatus);
        }
      }

      // Render system resources
      renderSystemResources(system);

      // Summary
      const healthy = Array.from(services.values()).filter(s => s.status === 'healthy').length;
      const total = services.size;
      const summaryColor = healthy === total ? chalk.green :
                          healthy > total * 0.7 ? chalk.yellow : chalk.red;

      console.log(chalk.cyan('\n📈 Summary:'));
      console.log(`  Service Health:   ${summaryColor(`${healthy}/${total} services healthy`)}`);

      // Store metrics history
      monitorState.metrics.set(Date.now(), { healthy, total, system });

      // Keep only last 100 metrics
      if (monitorState.metrics.size > 100) {
        const oldestKey = Math.min(...monitorState.metrics.keys());
        monitorState.metrics.delete(oldestKey);
      }

    } catch (error) {
      log.error(`Dashboard update failed: ${error.message}`);
    }
  };

  // Initial update
  await updateDashboard();

  // Set up refresh interval
  const interval = setInterval(updateDashboard, refreshInterval);

  // Graceful shutdown
  process.on('SIGINT', () => {
    clearInterval(interval);
    console.log(chalk.cyan('\n📊 Dashboard monitoring stopped'));
    process.exit(0);
  });

  // Keep process alive
  await new Promise(() => {});
}

async function runSingleCheck() {
  const spinner = ora('🔍 Checking system status...').start();

  try {
    const { services, system } = await performFullHealthCheck();

    spinner.stop();
    renderDashboardHeader();

    // Render all services
    console.log(chalk.cyan('🏥 Service Health:'));
    for (const [, health] of services) {
      renderServiceStatus(health);
    }

    renderSystemResources(system);

    // Final summary
    const healthy = Array.from(services.values()).filter(s => s.status === 'healthy').length;
    const total = services.size;

    if (healthy === total) {
      log.success(`🎯 All ${total} services are healthy`);
    } else {
      log.warn(`⚠ ${healthy}/${total} services are healthy`);
    }

  } catch (error) {
    spinner.fail(`Status check failed: ${error.message}`);
    process.exit(1);
  }
}

// Main function
async function main() {
  // Parse command line arguments
  program
    .option('-w, --watch', 'Run interactive dashboard with live updates')
    .option('-j, --json', 'Output results in JSON format')
    .option('-q, --quiet', 'Minimal output')
    .option('--service <name>', 'Check specific service only')
    .parse();

  const options = program.opts();

  // Handle specific service check
  if (options.service) {
    const serviceConfig = STATUS_CONFIG.services[options.service];
    if (!serviceConfig) {
      log.error(`Unknown service: ${options.service}`);
      console.log(`Available services: ${Object.keys(STATUS_CONFIG.services).join(', ')}`);
      process.exit(1);
    }

    const health = await checkServiceHealth(options.service, serviceConfig);

    if (options.json) {
      console.log(JSON.stringify(health, null, 2));
    } else {
      console.log(chalk.cyan(`\n${serviceConfig.name} Status:`));
      renderServiceStatus(health);
    }

    process.exit(health.status === 'healthy' ? 0 : 1);
  }

  // Handle JSON output
  if (options.json) {
    const { services, system } = await performFullHealthCheck();
    const result = {
      timestamp: new Date().toISOString(),
      services: Object.fromEntries(services),
      system,
      summary: {
        healthy: Array.from(services.values()).filter(s => s.status === 'healthy').length,
        total: services.size
      }
    };
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  // Handle interactive mode
  if (options.watch) {
    await runInteractiveDashboard();
  } else {
    await runSingleCheck();
  }
}

// Handle CLI help
if (process.argv.includes('--help')) {
  console.log(`
YoRHa Legal AI System Status Monitor

Usage: npm run status [options]

Options:
  -w, --watch           Run interactive dashboard with live updates
  -j, --json           Output results in JSON format
  -q, --quiet          Minimal output
  --service <name>     Check specific service only
  --help               Show this help message

Examples:
  npm run status                    # Single status check
  npm run status --watch            # Interactive dashboard
  npm run status --json             # JSON output
  npm run status --service ollama   # Check Ollama service only

Available Services:
  postgresql, redis, ollama, qdrant, goService, sveltekit
`);
  process.exit(0);
}

// Helper function
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Run the status monitor
main().catch(error => {
  log.error(`Status monitor failed: ${error.message}`);
  process.exit(1);
});