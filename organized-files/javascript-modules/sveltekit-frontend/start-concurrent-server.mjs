#!/usr/bin/env node

/**
 * Concurrent Server Startup Script
 * Demonstrates enhanced development and production server startup
 */

import { spawn } from 'child_process';
import { cpus, totalmem } from 'os';
import chalk from 'chalk';
import { performance } from 'perf_hooks';

const CPU_COUNT = cpus().length;
const TOTAL_MEMORY_GB = Math.round(totalmem() / 1024 / 1024 / 1024);
const NODE_ENV = process.env.NODE_ENV || 'development';

// Logo and startup banner
const LOGO = `
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                   ║
║   🚀 SvelteKit Concurrent Server Architecture                                     ║
║   ⚡ Enhanced Development with WebGPU + Loki.js + Multi-Core Processing          ║
║                                                                                   ║
║   💻 System: ${CPU_COUNT} cores, ${TOTAL_MEMORY_GB}GB RAM                                                     ║
║   🌍 Environment: ${NODE_ENV.toUpperCase()}                                                  ║
║   📊 Concurrency: Optimized for ${Math.min(CPU_COUNT, 16)} workers                                      ║
║                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
`;

console.log(chalk.cyan(LOGO));

/**
 * Available startup modes
 */
const STARTUP_MODES = {
  'dev': {
    name: 'Development (Single Process)',
    description: 'Standard Vite dev server with HMR',
    command: 'vite',
    args: ['dev', '--config', 'vite.config.concurrent.ts'],
    env: { NODE_ENV: 'development' }
  },
  
  'dev:concurrent': {
    name: 'Development (Concurrent)',
    description: 'Enhanced concurrent server with worker threads',
    command: 'node',
    args: ['server.config.concurrent.mjs'],
    env: { NODE_ENV: 'development', ENABLE_CLUSTERING: 'false' }
  },
  
  'dev:cluster': {
    name: 'Development (Clustered)',
    description: 'Development server with clustering enabled',
    command: 'node',
    args: ['server.config.concurrent.mjs'],
    env: { NODE_ENV: 'development', ENABLE_CLUSTERING: 'true' }
  },
  
  'prod': {
    name: 'Production (Single Process)',
    description: 'Production build with SvelteKit adapter',
    command: 'node',
    args: ['build/index.js'],
    env: { NODE_ENV: 'production' }
  },
  
  'prod:concurrent': {
    name: 'Production (Concurrent)',
    description: 'Production server with concurrent architecture',
    command: 'node',
    args: ['server.config.concurrent.mjs'],
    env: { NODE_ENV: 'production', ENABLE_CLUSTERING: 'false' }
  },
  
  'prod:cluster': {
    name: 'Production (Full Cluster)',
    description: 'Production server with full clustering and load balancing',
    command: 'node',
    args: ['server.config.concurrent.mjs'],
    env: { NODE_ENV: 'production', ENABLE_CLUSTERING: 'true' }
  },
  
  'benchmark': {
    name: 'Performance Benchmark',
    description: 'Run performance benchmarks across all modes',
    command: 'node',
    args: ['scripts/performance-benchmark.mjs'],
    env: { NODE_ENV: 'development' }
  }
};

/**
 * Parse command line arguments
 */
const args = process.argv.slice(2);
const mode = args[0] || 'dev:concurrent';
const selectedMode = STARTUP_MODES[mode];

if (!selectedMode) {
  console.log(chalk.red('❌ Invalid startup mode. Available modes:\n'));
  
  Object.entries(STARTUP_MODES).forEach(([key, config]) => {
    console.log(`  ${chalk.cyan(key.padEnd(15))} - ${config.description}`);
  });
  
  console.log(`\nUsage: ${chalk.yellow('node start-concurrent-server.mjs [mode]')}`);
  process.exit(1);
}

/**
 * System recommendations
 */
function displaySystemRecommendations() {
  console.log(chalk.yellow('💡 System Recommendations:'));
  
  if (TOTAL_MEMORY_GB < 8) {
    console.log(chalk.red(`   ⚠️  Low memory detected (${TOTAL_MEMORY_GB}GB). Consider upgrading to 16GB+ for optimal performance.`));
  } else if (TOTAL_MEMORY_GB >= 16) {
    console.log(chalk.green(`   ✅ Excellent memory (${TOTAL_MEMORY_GB}GB) - All concurrent features enabled.`));
  } else {
    console.log(chalk.yellow(`   ℹ️  Good memory (${TOTAL_MEMORY_GB}GB) - Most features enabled.`));
  }
  
  if (CPU_COUNT < 4) {
    console.log(chalk.red(`   ⚠️  Limited CPU cores (${CPU_COUNT}). Concurrency benefits will be minimal.`));
  } else if (CPU_COUNT >= 8) {
    console.log(chalk.green(`   ✅ Excellent CPU (${CPU_COUNT} cores) - Full concurrent processing enabled.`));
  } else {
    console.log(chalk.yellow(`   ℹ️  Good CPU (${CPU_COUNT} cores) - Moderate concurrent processing enabled.`));
  }
  
  console.log('');
}

/**
 * Environment setup
 */
function setupEnvironment(mode) {
  const envFile = mode.startsWith('prod') ? '.env.production.local' : '.env.development.local';
  
  console.log(chalk.blue(`📋 Loading environment from: ${envFile}`));
  console.log(chalk.blue(`🔧 Mode: ${selectedMode.name}`));
  console.log(chalk.blue(`📝 Description: ${selectedMode.description}`));
  console.log('');
  
  return {
    ...process.env,
    ...selectedMode.env,
    CPU_COUNT: CPU_COUNT.toString(),
    TOTAL_MEMORY_GB: TOTAL_MEMORY_GB.toString(),
    STARTUP_MODE: mode,
    STARTUP_TIME: Date.now().toString()
  };
}

/**
 * Start server with monitoring
 */
async function startServer() {
  displaySystemRecommendations();
  
  const startTime = performance.now();
  const env = setupEnvironment(mode);
  
  console.log(chalk.green(`🚀 Starting ${selectedMode.name}...`));
  console.log(chalk.gray(`   Command: ${selectedMode.command} ${selectedMode.args.join(' ')}`));
  console.log('');
  
  const serverProcess = spawn(selectedMode.command, selectedMode.args, {
    stdio: 'inherit',
    env,
    shell: process.platform === 'win32'
  });
  
  // Server process monitoring
  serverProcess.on('spawn', () => {
    const startupTime = performance.now() - startTime;
    console.log(chalk.green(`✅ Server spawned in ${startupTime.toFixed(2)}ms`));
    
    // Display connection information
    setTimeout(() => {
      console.log(chalk.cyan('🌐 Server Information:'));
      
      if (mode.includes('dev')) {
        console.log(`   📱 Frontend: ${chalk.underline('http://localhost:5173')}`);
        console.log(`   🔥 HMR: ${chalk.underline('ws://localhost:3131')}`);
        console.log(`   🔍 UnoCSS Inspector: ${chalk.underline('http://localhost:5173/__unocss/')}`);
        
        if (mode.includes('concurrent') || mode.includes('cluster')) {
          console.log(`   📊 Performance Monitor: Available in console`);
          console.log(`   💾 Worker Threads: ${Math.min(CPU_COUNT, 8)} active`);
        }
      } else {
        console.log(`   🌍 Production: ${chalk.underline('http://localhost:3000')}`);
        console.log(`   ⚖️  Load Balancer: ${mode.includes('cluster') ? 'Enabled' : 'Disabled'}`);
        console.log(`   🏭 Cluster Workers: ${mode.includes('cluster') ? Math.min(CPU_COUNT, 16) : 1}`);
      }
      
      console.log(`   🏥 Health Check: ${chalk.underline('http://localhost:' + (mode.includes('dev') ? '5173' : '3000') + '/health')}`);
      console.log('');
      
      // Feature status
      console.log(chalk.yellow('🎯 Enabled Features:'));
      console.log('   ✅ WebGPU Acceleration');
      console.log('   ✅ Loki.js In-Memory Database');
      console.log('   ✅ Enhanced PostCSS Processing');
      console.log('   ✅ Concurrent File Processing');
      console.log('   ✅ Memory-Aware Optimizations');
      console.log('   ✅ Multi-Core Build Pipeline');
      
      if (mode.includes('concurrent') || mode.includes('cluster')) {
        console.log('   ✅ Worker Thread Pool');
        console.log('   ✅ Performance Monitoring');
      }
      
      if (mode.includes('cluster')) {
        console.log('   ✅ Process Clustering');
        console.log('   ✅ Load Balancing');
        console.log('   ✅ Auto-Recovery');
      }
      
      console.log('');
      console.log(chalk.magenta('🎮 Quick Commands:'));
      console.log('   Ctrl+C: Graceful shutdown');
      console.log('   Ctrl+R: Restart server');
      
      if (mode.includes('dev')) {
        console.log('   r + Enter: Restart (HMR)');
        console.log('   o + Enter: Open browser');
        console.log('   q + Enter: Quit');
      }
      
      console.log('');
      console.log(chalk.green('🎉 Server is ready! Happy coding!'));
    }, 1000);
  });
  
  serverProcess.on('error', (error) => {
    console.error(chalk.red('❌ Server startup failed:'), error.message);
    process.exit(1);
  });
  
  serverProcess.on('exit', (code, signal) => {
    if (signal === 'SIGINT' || signal === 'SIGTERM') {
      console.log(chalk.yellow('\n👋 Server shutdown gracefully'));
    } else if (code !== 0) {
      console.error(chalk.red(`❌ Server exited with code ${code}`));
      process.exit(code);
    } else {
      console.log(chalk.green('\n✅ Server exited successfully'));
    }
  });
  
  // Handle process signals
  process.on('SIGINT', () => {
    console.log(chalk.yellow('\n🛑 Shutting down server...'));
    serverProcess.kill('SIGINT');
  });
  
  process.on('SIGTERM', () => {
    console.log(chalk.yellow('\n🛑 Terminating server...'));
    serverProcess.kill('SIGTERM');
  });
  
  // Keep the parent process alive
  return new Promise((resolve) => {
    serverProcess.on('exit', resolve);
  });
}

/**
 * Main execution
 */
async function main() {
  try {
    await startServer();
  } catch (error) {
    console.error(chalk.red('❌ Startup failed:'), error.message);
    process.exit(1);
  }
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error(chalk.red('💥 Uncaught Exception:'), error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error(chalk.red('💥 Unhandled Rejection at:'), promise, 'reason:', reason);
  process.exit(1);
});

// Start the server
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}