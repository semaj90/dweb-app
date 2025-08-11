#!/usr/bin/env node

/**
 * YoRHa Legal AI - Service Stop Orchestrator
 * 
 * Gracefully stops all services with:
 * - Process identification and termination
 * - Windows service management
 * - Data integrity preservation
 * - Cleanup procedures
 * 
 * @author YoRHa Legal AI Team
 * @version 2.0.0
 */

import 'zx/globals';
import chalk from 'chalk';
import ora from 'ora';

// Service configuration for shutdown
const STOP_CONFIG = {
  services: [
    {
      name: 'SvelteKit Frontend',
      ports: [3000, 5173],
      processes: ['node', 'vite'],
      gracefulShutdown: true,
      timeout: 10000
    },
    {
      name: 'Go Legal AI Service',
      ports: [8080, 50051],
      processes: ['legal-ai-server.exe'],
      gracefulShutdown: true,
      timeout: 15000
    },
    {
      name: 'Ollama LLM Service',
      ports: [11434],
      processes: ['ollama'],
      gracefulShutdown: true,
      timeout: 30000, // Models need time to unload
      customStop: async () => {
        try {
          await $`ollama stop`;
          return true;
        } catch {
          return false;
        }
      }
    },
    {
      name: 'Qdrant Vector DB',
      ports: [6333],
      processes: ['qdrant.exe'],
      gracefulShutdown: true,
      timeout: 20000
    },
    {
      name: 'Redis Cache',
      ports: [6379],
      processes: ['redis-server.exe'],
      gracefulShutdown: true,
      timeout: 5000,
      customStop: async () => {
        try {
          await $`echo "SHUTDOWN SAVE" | .\\redis-windows\\redis-cli.exe -h localhost -p 6379`;
          return true;
        } catch {
          return false;
        }
      }
    },
    {
      name: 'PostgreSQL Database',
      ports: [5432],
      isWindowsService: true,
      serviceName: 'postgresql-x64-17',
      gracefulShutdown: true,
      timeout: 30000
    }
  ]
};

// Enhanced logging
const log = {
  timestamp: () => new Date().toISOString(),
  info: (msg) => console.log(`[${log.timestamp()}]`, chalk.blue('ℹ'), msg),
  success: (msg) => console.log(`[${log.timestamp()}]`, chalk.green('✓'), msg),
  error: (msg) => console.log(`[${log.timestamp()}]`, chalk.red('✗'), msg),
  warn: (msg) => console.log(`[${log.timestamp()}]`, chalk.yellow('⚠'), msg),
  debug: (msg) => process.env.DEBUG && console.log(`[${log.timestamp()}]`, chalk.gray('🔍'), msg)
};

// Process management utilities
async function findProcessesByPort(port) {
  try {
    const result = await $`netstat -ano | findstr :${port}`;
    const lines = result.stdout.split('\n').filter(line => line.trim());
    const pids = new Set();
    
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 5 && parts[1].includes(`:${port}`)) {
        const pid = parseInt(parts[4]);
        if (!isNaN(pid)) {
          pids.add(pid);
        }
      }
    }
    
    return Array.from(pids);
  } catch {
    return [];
  }
}

async function findProcessesByName(processName) {
  try {
    const result = await $`tasklist /fi "imagename eq ${processName}" /fo csv /nh`;
    const lines = result.stdout.split('\n').filter(line => line.includes(processName));
    const pids = [];
    
    for (const line of lines) {
      const parts = line.split(',');
      if (parts.length >= 2) {
        const pid = parseInt(parts[1].replace(/"/g, ''));
        if (!isNaN(pid)) {
          pids.push(pid);
        }
      }
    }
    
    return pids;
  } catch {
    return [];
  }
}

async function killProcessGracefully(pid, timeout = 10000) {
  try {
    // First attempt: graceful termination
    await $`taskkill /pid ${pid} /t`;
    
    // Wait for process to terminate
    const start = Date.now();
    while (Date.now() - start < timeout) {
      try {
        await $`tasklist /fi "pid eq ${pid}" /fo csv /nh`;
        await sleep(1000);
      } catch {
        // Process no longer exists
        return { success: true, method: 'graceful' };
      }
    }
    
    // Force kill if still running
    await $`taskkill /pid ${pid} /t /f`;
    return { success: true, method: 'forced' };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

async function stopWindowsService(serviceName, timeout = 30000) {
  try {
    // Check if service is running
    const statusResult = await $`sc query ${serviceName}`;
    if (!statusResult.stdout.includes('RUNNING')) {
      return { success: true, method: 'already_stopped' };
    }
    
    // Stop the service
    await $`sc stop ${serviceName}`;
    
    // Wait for service to stop
    const start = Date.now();
    while (Date.now() - start < timeout) {
      try {
        const result = await $`sc query ${serviceName}`;
        if (result.stdout.includes('STOPPED')) {
          return { success: true, method: 'service_stop' };
        }
        await sleep(2000);
      } catch (error) {
        log.debug(`Service query failed: ${error.message}`);
      }
    }
    
    return { success: false, error: 'Timeout waiting for service to stop' };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// Main service stopping logic
async function stopService(serviceConfig) {
  const spinner = ora(`🛑 Stopping ${serviceConfig.name}...`).start();
  
  try {
    // Custom stop procedure if available
    if (serviceConfig.customStop) {
      const customResult = await serviceConfig.customStop();
      if (customResult) {
        spinner.succeed(`${serviceConfig.name} stopped using custom procedure`);
        return { success: true, method: 'custom' };
      }
    }
    
    // Windows service handling
    if (serviceConfig.isWindowsService) {
      const result = await stopWindowsService(serviceConfig.serviceName, serviceConfig.timeout);
      if (result.success) {
        spinner.succeed(`${serviceConfig.name} service stopped (${result.method})`);
        return result;
      } else {
        spinner.fail(`Failed to stop ${serviceConfig.name} service: ${result.error}`);
        return result;
      }
    }
    
    // Process-based service handling
    const allPids = new Set();
    
    // Find processes by port
    if (serviceConfig.ports) {
      for (const port of serviceConfig.ports) {
        const pids = await findProcessesByPort(port);
        pids.forEach(pid => allPids.add(pid));
      }
    }
    
    // Find processes by name
    if (serviceConfig.processes) {
      for (const processName of serviceConfig.processes) {
        const pids = await findProcessesByName(processName);
        pids.forEach(pid => allPids.add(pid));
      }
    }
    
    if (allPids.size === 0) {
      spinner.succeed(`${serviceConfig.name} - no running processes found`);
      return { success: true, method: 'not_running' };
    }
    
    // Stop all found processes
    const results = [];
    for (const pid of allPids) {
      const result = await killProcessGracefully(pid, serviceConfig.timeout);
      results.push({ pid, ...result });
    }
    
    const failed = results.filter(r => !r.success);
    if (failed.length > 0) {
      spinner.fail(`${serviceConfig.name} - failed to stop ${failed.length}/${results.length} processes`);
      return { success: false, failed };
    }
    
    const methods = [...new Set(results.map(r => r.method))];
    spinner.succeed(`${serviceConfig.name} stopped - ${results.length} processes (${methods.join(', ')})`);
    return { success: true, stopped: results.length, methods };
    
  } catch (error) {
    spinner.fail(`${serviceConfig.name} - Error: ${error.message}`);
    return { success: false, error: error.message };
  }
}

// System cleanup procedures
async function performCleanup() {
  const spinner = ora('🧹 Performing system cleanup...').start();
  
  try {
    const cleanupTasks = [
      {
        name: 'Temporary files',
        action: async () => {
          await $`del /q /s %temp%\\yorha-legal-ai-* 2>nul || echo "No temp files found"`;
        }
      },
      {
        name: 'Log rotation',
        action: async () => {
          const logDir = './logs';
          if (await fs.pathExists(logDir)) {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            await fs.move(logDir, `./logs-archive-${timestamp}`);
            await fs.ensureDir(logDir);
          }
        }
      },
      {
        name: 'PID files cleanup',
        action: async () => {
          await $`del /q *.pid 2>nul || echo "No PID files found"`;
        }
      }
    ];
    
    for (const task of cleanupTasks) {
      try {
        await task.action();
        log.debug(`✓ ${task.name} cleanup completed`);
      } catch (error) {
        log.debug(`⚠ ${task.name} cleanup failed: ${error.message}`);
      }
    }
    
    spinner.succeed('System cleanup completed');
  } catch (error) {
    spinner.warn(`Cleanup completed with warnings: ${error.message}`);
  }
}

// System status check
async function checkSystemStatus() {
  console.log(chalk.cyan('\n📊 Final System Status:'));
  
  const statusChecks = [
    {
      name: 'Port Usage',
      check: async () => {
        const ports = [3000, 5173, 8080, 50051, 11434, 6333, 6379, 5432];
        const active = [];
        
        for (const port of ports) {
          const pids = await findProcessesByPort(port);
          if (pids.length > 0) {
            active.push(`${port}:${pids.length}`);
          }
        }
        
        return active.length === 0 ? 'All ports free' : `Active: ${active.join(', ')}`;
      }
    },
    {
      name: 'Running Services',
      check: async () => {
        try {
          const result = await $`sc query type= service state= all | findstr "postgresql\\|redis"`;
          const running = result.stdout.split('\n')
            .filter(line => line.includes('RUNNING')).length;
          return `${running} Windows services running`;
        } catch {
          return '0 Windows services running';
        }
      }
    },
    {
      name: 'System Resources',
      check: async () => {
        try {
          const memResult = await $`wmic computersystem get TotalPhysicalMemory /value`;
          const cpuResult = await $`wmic cpu get loadpercentage /value`;
          
          return 'System resources available';
        } catch {
          return 'Resource check failed';
        }
      }
    }
  ];
  
  for (const check of statusChecks) {
    try {
      const result = await check.check();
      console.log(`  ${check.name.padEnd(20)} ${chalk.green(result)}`);
    } catch (error) {
      console.log(`  ${check.name.padEnd(20)} ${chalk.red('Check failed')}`);
    }
  }
}

// Main stop orchestration
async function main() {
  console.log(chalk.cyan.bold('🛑 YoRHa Legal AI - Stop Orchestrator\n'));
  
  const startTime = Date.now();
  
  // Parse command line arguments
  const forceStop = process.argv.includes('--force');
  const skipCleanup = process.argv.includes('--no-cleanup');
  
  if (forceStop) {
    log.warn('🚨 Force stop mode enabled - services will be terminated immediately');
  }
  
  // Stop services in reverse priority order (frontend first, database last)
  const services = [...STOP_CONFIG.services].reverse();
  const results = [];
  
  log.info(`🛑 Stopping ${services.length} services...`);
  
  for (const service of services) {
    if (forceStop) {
      // Adjust timeout for force mode
      service.timeout = Math.min(service.timeout, 5000);
      service.gracefulShutdown = false;
    }
    
    const result = await stopService(service);
    results.push({ service: service.name, ...result });
    
    // Brief pause between service stops
    await sleep(1000);
  }
  
  // Summary of stop operations
  console.log(chalk.cyan('\n📋 Stop Summary:'));
  const successful = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;
  
  for (const result of results) {
    const status = result.success ? chalk.green('✓ STOPPED') : chalk.red('✗ FAILED');
    console.log(`  ${result.service.padEnd(25)} ${status}`);
  }
  
  console.log(`\n  ${chalk.green(`✓ ${successful} successful`)} | ${failed > 0 ? chalk.red(`✗ ${failed} failed`) : chalk.gray('0 failed')}`);
  
  // Perform cleanup if requested
  if (!skipCleanup) {
    await performCleanup();
  }
  
  // Final system status
  await checkSystemStatus();
  
  const duration = ((Date.now() - startTime) / 1000).toFixed(1);
  
  if (failed === 0) {
    log.success(`🎯 All services stopped successfully in ${duration}s`);
  } else {
    log.warn(`⚠ Stop completed with ${failed} failures in ${duration}s`);
    
    if (failed > 0 && !forceStop) {
      console.log(chalk.yellow('\n💡 Tip: Use --force flag for immediate termination'));
    }
  }
  
  console.log(chalk.cyan('\n🔄 Restart commands:'));
  console.log('  Development: npm run dev');
  console.log('  Production:  npm run start');
}

// Helper functions
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Handle CLI arguments
if (process.argv.includes('--help')) {
  console.log(`
YoRHa Legal AI Stop Orchestrator

Usage: npm run stop [options]

Options:
  --force        Force stop all services immediately (shorter timeouts)
  --no-cleanup   Skip cleanup procedures after stopping services
  --help         Show this help message

Examples:
  npm run stop            # Graceful stop with cleanup
  npm run stop --force    # Force stop all services quickly
  npm run stop --no-cleanup # Stop services but skip cleanup
`);
  process.exit(0);
}

// Run the stop orchestrator
main().catch(error => {
  log.error(`Stop orchestrator failed: ${error.message}`);
  process.exit(1);
});