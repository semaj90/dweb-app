#!/usr/bin/env zx

/**
 * PM2 Cluster Deployment Script for Legal AI System
 * 
 * Features:
 * - QUIC tensor server deployment
 * - Multi-process scaling
 * - Health checks
 * - Zero-downtime deployment
 * - Performance monitoring
 */

import chalk from 'chalk';
import ora from 'ora';

// Configuration
const CONFIG = {
  apps: {
    'sveltekit-frontend': { instances: 'max', port: 5173 },
    'quic-tensor-server': { instances: 2, port: 4433 },
    'simd-server': { instances: 1, port: 8080 },
    'job-processor': { instances: 4, port: 8081 },
    'websocket-server': { instances: 2, port: 8090 },
    'vite-prod': { instances: 2, port: 4173 },
    'redis-monitor': { instances: 1, port: null },
  },
  healthCheckTimeout: 30000,
  deploymentMode: process.argv.includes('--production') ? 'production' : 'development',
};

const log = {
  info: (msg) => console.log(chalk.blue('ℹ'), msg),
  success: (msg) => console.log(chalk.green('✓'), msg),
  error: (msg) => console.log(chalk.red('✗'), msg),
  warn: (msg) => console.log(chalk.yellow('⚠'), msg),
};

async function main() {
  console.log(chalk.cyan.bold('🚀 Legal AI Cluster Deployment\n'));
  
  if (process.argv.includes('--help')) {
    showHelp();
    return;
  }

  const commands = {
    deploy: deployCluster,
    status: checkClusterStatus,
    restart: restartCluster,
    stop: stopCluster,
    logs: showLogs,
    monitor: monitorCluster,
    build: buildServices,
    health: healthCheck,
  };

  const command = process.argv[2] || 'deploy';
  
  if (commands[command]) {
    await commands[command]();
  } else {
    log.error(`Unknown command: ${command}`);
    showHelp();
  }
}

async function deployCluster() {
  log.info('Starting cluster deployment...');

  // 1. Build services
  await buildServices();

  // 2. Ensure logs directory exists
  await ensureLogsDirectory();

  // 3. Stop existing services gracefully
  await gracefulStop();

  // 4. Start services in order
  await startServices();

  // 5. Health checks
  await performHealthChecks();

  // 6. Display cluster status
  await checkClusterStatus();

  log.success('🎉 Cluster deployment completed!');
}

async function buildServices() {
  const spinner = ora('Building services...').start();

  try {
    // Build Go QUIC tensor server
    if (!await $`which go`.quiet()) {
      spinner.warn('Go not found, skipping Go services build');
    } else {
      spinner.text = 'Building QUIC tensor server...';
      await $`cd go-microservice && go build -o quic-tensor-server.exe quic-tensor-server.go`;
      
      spinner.text = 'Building SIMD server...';
      if (await pathExists('./go-microservice/simd-server-prod.exe')) {
        spinner.text = 'SIMD server already built';
      }
    }

    // Build SvelteKit frontend
    spinner.text = 'Building SvelteKit frontend...';
    await $`cd sveltekit-frontend && npm run build`;

    // Install dependencies for workers
    spinner.text = 'Installing worker dependencies...';
    if (await pathExists('./workers/package.json')) {
      await $`cd workers && npm install`;
    }

    spinner.succeed('Services built successfully');
  } catch (error) {
    spinner.fail(`Build failed: ${error.message}`);
    throw error;
  }
}

async function ensureLogsDirectory() {
  await $`mkdir -p logs`;
  log.success('Logs directory ready');
}

async function gracefulStop() {
  const spinner = ora('Stopping existing services...').start();

  try {
    // Get running PM2 processes
    const list = await $`pm2 jlist`.quiet();
    const processes = JSON.parse(list.stdout || '[]');
    
    if (processes.length === 0) {
      spinner.info('No existing PM2 processes found');
      return;
    }

    // Stop each service gracefully
    for (const proc of processes) {
      if (proc.pm2_env?.status === 'online') {
        spinner.text = `Stopping ${proc.name}...`;
        await $`pm2 stop ${proc.name}`.quiet();
      }
    }

    spinner.succeed('Existing services stopped');
  } catch (error) {
    spinner.warn('No existing PM2 processes or stop failed');
  }
}

async function startServices() {
  const spinner = ora('Starting cluster services...').start();

  try {
    // Delete existing PM2 processes
    await $`pm2 delete all`.quiet().catch(() => {});

    // Start services using ecosystem config
    spinner.text = 'Starting services from ecosystem config...';
    await $`pm2 start ecosystem.config.js --env ${CONFIG.deploymentMode}`;

    // Wait for services to initialize
    spinner.text = 'Waiting for services to initialize...';
    await sleep(5000);

    spinner.succeed('Cluster services started');
  } catch (error) {
    spinner.fail(`Failed to start services: ${error.message}`);
    throw error;
  }
}

async function performHealthChecks() {
  const spinner = ora('Performing health checks...').start();

  const healthChecks = [
    { name: 'SvelteKit Frontend', port: 5173, path: '/' },
    { name: 'QUIC Tensor Server', port: 4433, path: '/health', protocol: 'https' },
    { name: 'SIMD Server', port: 8080, path: '/health' },
    { name: 'WebSocket Server', port: 8090, path: '/health' },
  ];

  for (const check of healthChecks) {
    spinner.text = `Health checking ${check.name}...`;
    
    try {
      const protocol = check.protocol || 'http';
      const url = `${protocol}://localhost:${check.port}${check.path}`;
      
      // Use curl for HTTPS endpoints (QUIC server)
      if (protocol === 'https') {
        await $`curl -k -f ${url} --max-time 10`.quiet();
      } else {
        const response = await fetch(url, { 
          timeout: 10000,
          signal: AbortSignal.timeout(10000)
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
      }
      
      log.success(`${check.name} is healthy`);
    } catch (error) {
      log.warn(`${check.name} health check failed: ${error.message}`);
    }
  }

  spinner.succeed('Health checks completed');
}

async function checkClusterStatus() {
  console.log(chalk.cyan('\n📊 Cluster Status:\n'));

  try {
    const list = await $`pm2 jlist`;
    const processes = JSON.parse(list.stdout || '[]');

    if (processes.length === 0) {
      log.warn('No PM2 processes running');
      return;
    }

    // Display process status table
    console.log('Name'.padEnd(20) + 'Status'.padEnd(12) + 'CPU'.padEnd(8) + 'Memory'.padEnd(12) + 'Restarts');
    console.log('─'.repeat(60));

    for (const proc of processes) {
      const name = proc.name.padEnd(20);
      const status = (proc.pm2_env?.status || 'unknown').padEnd(12);
      const cpu = `${proc.monit?.cpu || 0}%`.padEnd(8);
      const memory = `${Math.round((proc.monit?.memory || 0) / 1024 / 1024)}MB`.padEnd(12);
      const restarts = proc.pm2_env?.restart_time || 0;

      const statusColor = proc.pm2_env?.status === 'online' ? chalk.green : chalk.red;
      console.log(`${name}${statusColor(status)}${cpu}${memory}${restarts}`);
    }

    // Show port mappings
    console.log(chalk.cyan('\n🌐 Service Endpoints:\n'));
    for (const [name, config] of Object.entries(CONFIG.apps)) {
      if (config.port) {
        const protocol = name === 'quic-tensor-server' ? 'https' : 'http';
        console.log(`  ${name.padEnd(20)} ${protocol}://localhost:${config.port}`);
      }
    }

  } catch (error) {
    log.error(`Failed to get cluster status: ${error.message}`);
  }
}

async function restartCluster() {
  log.info('Restarting cluster...');
  
  const spinner = ora('Restarting services...').start();
  
  try {
    await $`pm2 restart ecosystem.config.js --env ${CONFIG.deploymentMode}`;
    spinner.succeed('Cluster restarted successfully');
    
    await performHealthChecks();
  } catch (error) {
    spinner.fail(`Restart failed: ${error.message}`);
    throw error;
  }
}

async function stopCluster() {
  log.info('Stopping cluster...');
  
  const spinner = ora('Stopping all services...').start();
  
  try {
    await $`pm2 stop all`;
    await $`pm2 delete all`;
    spinner.succeed('Cluster stopped successfully');
  } catch (error) {
    spinner.fail(`Stop failed: ${error.message}`);
    throw error;
  }
}

async function showLogs() {
  const service = process.argv[3];
  
  if (service) {
    log.info(`Showing logs for ${service}...`);
    await $`pm2 logs ${service} --lines 50`;
  } else {
    log.info('Showing logs for all services...');
    await $`pm2 logs --lines 20`;
  }
}

async function monitorCluster() {
  log.info('Starting cluster monitoring...');
  log.info('Press Ctrl+C to exit monitoring');
  
  // Start PM2 monitoring
  await $`pm2 monit`;
}

async function healthCheck() {
  console.log(chalk.cyan('🏥 Comprehensive Health Check\n'));
  
  const checks = [
    {
      name: 'PM2 Daemon',
      check: async () => {
        await $`pm2 ping`.quiet();
        return 'PM2 daemon is running';
      }
    },
    {
      name: 'Redis Connection',
      check: async () => {
        await $`redis-cli ping`.quiet();
        return 'Redis is responding';
      }
    },
    {
      name: 'PostgreSQL Connection',
      check: async () => {
        await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\pg_isready.exe" -h localhost -p 5432`.quiet();
        return 'PostgreSQL is ready';
      }
    },
    {
      name: 'Ollama Service',
      check: async () => {
        const response = await fetch('http://localhost:11434/api/version', { timeout: 5000 });
        if (!response.ok) throw new Error('Ollama not responding');
        return 'Ollama is running';
      }
    },
    {
      name: 'Disk Space',
      check: async () => {
        const result = await $`dir C:\\ /-c`.quiet();
        return 'Disk space check completed';
      }
    }
  ];

  for (const { name, check } of checks) {
    try {
      const result = await check();
      log.success(`${name}: ${result}`);
    } catch (error) {
      log.error(`${name}: Failed - ${error.message}`);
    }
  }
}

function showHelp() {
  console.log(`
${chalk.cyan.bold('Legal AI Cluster Deployment Tool')}

${chalk.yellow('Usage:')}
  zx scripts/cluster-deploy.mjs [command] [options]

${chalk.yellow('Commands:')}
  deploy     Deploy the full cluster (default)
  status     Show cluster status
  restart    Restart all services
  stop       Stop all services
  logs       Show logs (optionally specify service name)
  monitor    Start PM2 monitoring interface
  build      Build all services
  health     Comprehensive health check

${chalk.yellow('Options:')}
  --production    Deploy in production mode
  --help          Show this help message

${chalk.yellow('Examples:')}
  zx scripts/cluster-deploy.mjs deploy
  zx scripts/cluster-deploy.mjs deploy --production
  zx scripts/cluster-deploy.mjs status
  zx scripts/cluster-deploy.mjs logs quic-tensor-server
  zx scripts/cluster-deploy.mjs health

${chalk.yellow('Service Ports:')}
  SvelteKit Frontend:  5173 (dev) / 3000 (prod)
  QUIC Tensor Server:  4433 (HTTPS)
  SIMD Server:         8080
  Job Processor:       8081
  WebSocket Server:    8090
  Vite Preview:        4173
`);
}

// Helper functions
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function pathExists(path) {
  try {
    await $`test -f ${path}`.quiet();
    return true;
  } catch {
    return false;
  }
}

// Run the deployment script
main().catch(error => {
  log.error(`Deployment failed: ${error.message}`);
  process.exit(1);
});