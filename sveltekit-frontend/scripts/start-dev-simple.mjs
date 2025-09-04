#!/usr/bin/env node
/**
 * 🚀 Simple Development Startup - Core Services Only
 * Reliable startup for development with existing services
 */

import { spawn, exec } from 'child_process';
import { existsSync } from 'fs';
import chalk from 'chalk';
import ora from 'ora';
import boxen from 'boxen';
import { promisify } from 'util';

const execAsync = promisify(exec);

const style = {
  success: (text) => chalk.hex('#51cf66')(text),
  warning: (text) => chalk.hex('#ff6b6b')(text),
  error: (text) => chalk.hex('#ff4757')(text),
  accent: (text) => chalk.hex('#dca561')(text),
  dim: (text) => chalk.dim(text)
};

class SimpleDevStartup {
  constructor() {
    this.processes = [];
  }

  async start() {
    console.log(boxen(
      chalk.bold('🚀 SIMPLE DEV STARTUP\n') +
      chalk.dim('SvelteKit + Core Services'),
      {
        padding: 1,
        margin: 1,
        borderStyle: 'round',
        borderColor: '#dca561'
      }
    ));

    try {
      await this.startSvelteKit();
      await this.startExistingServices();
      this.displaySummary();
      this.setupGracefulShutdown();
    } catch (error) {
      console.error(style.error(`Startup failed: ${error.message}`));
      process.exit(1);
    }
  }

  async startSvelteKit() {
    const spinner = ora('Starting SvelteKit...').start();
    
    // Check if SvelteKit is already running
    const isAlreadyRunning = await this.checkPort(5173);
    if (isAlreadyRunning) {
      spinner.succeed(style.success('SvelteKit already running on port 5173'));
      return;
    }

    try {
      const isWindows = process.platform === 'win32';
      const npmCmd = isWindows ? 'npm.cmd' : 'npm';
      
      const svelteProcess = spawn(npmCmd, ['run', 'dev'], {
        stdio: ['ignore', 'pipe', 'pipe'],
        shell: isWindows,
        env: { 
          ...process.env, 
          NODE_OPTIONS: '--max-old-space-size=4096'
        }
      });
      
      this.processes.push({
        name: 'SvelteKit',
        process: svelteProcess,
        port: 5173
      });

      // Wait for SvelteKit to be ready
      await this.waitForPort(5173, 30000);
      spinner.succeed(style.success('SvelteKit started on port 5173'));
      
    } catch (error) {
      spinner.fail(style.error(`SvelteKit failed: ${error.message}`));
      throw error;
    }
  }

  async startExistingServices() {
    // Core services that might already be running
    const coreServices = [
      { name: 'Enhanced RAG', port: 8094, path: '../go-microservice/bin/enhanced-rag.exe' },
      { name: 'Upload Service', port: 8093, path: '../go-microservice/bin/upload-service.exe' },
      { name: 'Vector Service', port: 8095, path: '../go-microservice/bin/simple-vector-service.exe' },
      { name: 'gRPC Server', port: 50051, path: '../go-microservice/bin/grpc-server.exe' }
    ];

    console.log(style.accent('\n📡 Checking Core Services:'));

    for (const service of coreServices) {
      const spinner = ora(`Checking ${service.name}...`).start();
      
      const isRunning = await this.checkPort(service.port);
      if (isRunning) {
        spinner.succeed(style.success(`${service.name} already running on port ${service.port}`));
      } else if (existsSync(service.path)) {
        try {
          await this.startService(service);
          spinner.succeed(style.success(`${service.name} started on port ${service.port}`));
        } catch (error) {
          spinner.warn(style.warning(`${service.name} failed to start: ${error.message}`));
        }
      } else {
        spinner.warn(style.warning(`${service.name} binary not found at ${service.path}`));
      }
    }
  }

  async startService(service) {
    const serviceProcess = spawn(service.path, [], {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PORT: service.port.toString()
      }
    });

    this.processes.push({
      name: service.name,
      process: serviceProcess,
      port: service.port
    });

    // Wait for service to be ready
    await this.waitForPort(service.port, 10000);
  }

  async checkPort(port) {
    try {
      const response = await fetch(`http://localhost:${port}`, { 
        method: 'HEAD',
        signal: AbortSignal.timeout(1000)
      });
      return response.ok || response.status < 500;
    } catch (error) {
      return false;
    }
  }

  async waitForPort(port, timeout = 10000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      if (await this.checkPort(port)) {
        return true;
      }
      await this.delay(500);
    }
    
    throw new Error(`Port ${port} not ready within ${timeout}ms`);
  }

  displaySummary() {
    const runningServices = this.processes.length + 1; // +1 for potentially existing services
    
    const summary = [
      chalk.bold('🎉 DEVELOPMENT ENVIRONMENT READY'),
      '',
      style.success(`✅ Services Available: ${runningServices}`),
      '',
      style.accent('📡 Access Points:'),
      style.dim('• Frontend: http://localhost:5173'),
      style.dim('• Enhanced RAG: http://localhost:8094'),
      style.dim('• Upload Service: http://localhost:8093'),
      style.dim('• Vector Service: http://localhost:8095'),
      style.dim('• gRPC Server: http://localhost:50051'),
      '',
      style.success('✅ Ready for development')
    ];
    
    console.log(boxen(summary.join('\n'), {
      padding: 1,
      margin: 1,
      borderStyle: 'round',
      borderColor: '#51cf66'
    }));
  }

  setupGracefulShutdown() {
    console.log(style.dim('\n🔍 Development server running. Press Ctrl+C to stop.\n'));
    
    process.on('SIGINT', () => {
      console.log(style.warning('\n⏹️ Shutting down development server...'));
      
      this.processes.forEach(({ process, name }) => {
        try {
          process.kill('SIGTERM');
          console.log(style.dim(`Stopped ${name}`));
        } catch (err) {
          // Process might already be stopped
        }
      });
      
      process.exit(0);
    });
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Start the simple development environment
const startup = new SimpleDevStartup();
startup.start().catch(console.error);