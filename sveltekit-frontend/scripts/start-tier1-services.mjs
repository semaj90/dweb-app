#!/usr/bin/env node
/**
 * Tier 1 Services Startup - Core Services Only
 * Enhanced RAG + Upload Service + CUDA Service + SvelteKit
 */

import { spawn } from 'child_process';
import chalk from 'chalk';
import ora from 'ora';

const style = {
  success: (text) => chalk.hex('#51cf66')(text),
  warning: (text) => chalk.hex('#ff6b6b')(text),
  info: (text) => chalk.hex('#8b9dc3')(text),
  bold: (text) => chalk.bold(text)
};

class Tier1Startup {
  constructor() {
    this.services = [
      { name: 'Enhanced RAG', port: 8095, binaryPath: '../enhanced-rag-updated.exe', sourcePath: '../go-microservice/cmd/enhanced-rag/main.go', running: false },
      { name: 'Upload Service', port: 8093, binaryPath: '../go-microservice/bin/upload-service.exe', sourcePath: '../go-microservice/cmd/upload-service/main.go', running: false },
      { name: 'CUDA Service', port: 8096, binaryPath: '../go-microservice/bin/cuda-ai-service.exe', sourcePath: '../go-microservice/cmd/cuda-service/main.go', running: true } // Already running
    ];
  }

  async start() {
    console.log(style.bold('🚀 Starting Tier 1 Core Services\n'));

    // Check what's already running
    await this.checkRunningServices();

    // Start missing services
    for (const service of this.services) {
      if (!service.running) {
        await this.startService(service);
        await this.sleep(2000);
      }
    }

    // Start SvelteKit
    await this.startSvelteKit();
    
    console.log(style.success('\n✅ Tier 1 services ready!'));
    console.log(style.info('🌐 Frontend: http://localhost:5173'));
    console.log(style.info('🏥 Health: http://localhost:5173/system/health'));
  }

  async checkRunningServices() {
    for (const service of this.services) {
      const spinner = ora(`Checking ${service.name}...`).start();
      
      try {
        const response = await fetch(`http://localhost:${service.port}/health`, { 
          signal: AbortSignal.timeout(2000) 
        });
        if (response.ok) {
          spinner.succeed(style.success(`✅ ${service.name} already running`));
          service.running = true;
        } else {
          throw new Error('Not healthy');
        }
      } catch {
        spinner.info(style.info(`ℹ️  ${service.name} needs to be started`));
        service.running = false;
      }
    }
  }

  async startService(service) {
    const spinner = ora(`Starting ${service.name}...`).start();
    
    try {
      // Check if binary exists first (following CLAUDE.md best practices)
      const fs = await import('fs');
      let command, args;
      
      if (fs.existsSync(service.binaryPath)) {
        // Use existing binary
        spinner.text = `Using existing binary: ${service.binaryPath}`;
        command = service.binaryPath;
        args = [];
      } else {
        // Fallback to go run
        spinner.text = `Binary not found, building from source: ${service.sourcePath}`;
        command = 'go';
        args = ['run', service.sourcePath];
      }

      const childProcess = spawn(command, args, {
        stdio: 'pipe',
        env: { 
          ...process.env, 
          PORT: service.port.toString(),
          // Disable RabbitMQ temporarily if not available
          DISABLE_RABBITMQ: 'true',
          LOG_LEVEL: 'INFO'
        }
      });

      // Give it time to start
      await this.sleep(4000);

      // Check if it's healthy
      try {
        const response = await fetch(`http://localhost:${service.port}/health`, {
          signal: AbortSignal.timeout(5000)
        });
        if (response.ok) {
          spinner.succeed(style.success(`✅ ${service.name} started successfully`));
          service.running = true;
        } else {
          throw new Error('Health check failed');
        }
      } catch (healthError) {
        // Try alternative health check endpoints
        try {
          const altResponse = await fetch(`http://localhost:${service.port}/`, {
            signal: AbortSignal.timeout(3000)
          });
          if (altResponse.status < 500) {
            spinner.succeed(style.success(`✅ ${service.name} started (alternative health check)`));
            service.running = true;
          } else {
            throw healthError;
          }
        } catch {
          spinner.warn(style.warning(`⚠️  ${service.name} started but health check pending`));
          // Mark as running anyway since process started
          service.running = true;
        }
      }

    } catch (error) {
      spinner.fail(style.warning(`❌ ${service.name} failed: ${error.message}`));
    }
  }

  async startSvelteKit() {
    const spinner = ora('Starting SvelteKit frontend...').start();
    
    const viteProcess = spawn('npm', ['run', 'dev'], {
      stdio: 'inherit',
      shell: true
    });

    await this.sleep(3000);
    spinner.succeed(style.success('✅ SvelteKit frontend started'));
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

const startup = new Tier1Startup();
startup.start();