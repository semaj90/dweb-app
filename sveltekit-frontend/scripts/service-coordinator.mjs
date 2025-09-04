#!/usr/bin/env node
/**
 * Service Coordinator for Legal AI Platform
 * Fixes port conflicts and service dependency issues
 * Coordinates startup order with health checks
 */

import { spawn } from 'child_process';
import { createServer } from 'net';
import chalk from 'chalk';
import { setTimeout } from 'timers/promises';

const SERVICES = {
  // Backend infrastructure (must start first)
  postgresql: { port: 5432, required: true, startCommand: null, healthCheck: 'postgresql' },
  redis: { port: 6379, required: true, startCommand: 'redis-server', healthCheck: 'redis' },
  
  // AI/ML services
  ollama: { port: 11434, required: true, startCommand: 'ollama serve', healthCheck: 'ollama' },
  
  // Storage services
  minio: { port: 9000, required: true, startCommand: 'minio server ./minio-data --address :9000 --console-address :9001', healthCheck: 'minio' },
  qdrant: { port: 6333, required: true, startCommand: '.\\qdrant-windows\\qdrant.exe', healthCheck: 'qdrant' },
  
  // Graph database
  neo4j: { port: 7474, required: false, startCommand: null, healthCheck: 'neo4j' },
  
  // Go microservices (start after infrastructure)
  enhancedRAG: { port: 8094, required: true, startCommand: 'cd ../go-microservice && go run cmd/enhanced-rag/main.go', healthCheck: 'http' },
  uploadService: { port: 8093, required: true, startCommand: 'cd ../go-microservice && go run cmd/upload-service/main.go', healthCheck: 'http' },
  
  // QUIC services with dynamic port allocation
  quicGateway: { port: 8447, required: false, startCommand: null, healthCheck: 'quic' },
  
  // Frontend (start last)
  sveltekit: { port: 5173, required: true, startCommand: 'npm run dev', healthCheck: 'http' }
};

const STARTUP_ORDER = [
  ['postgresql', 'redis'],           // Infrastructure first
  ['ollama'],                        // AI service
  ['minio', 'qdrant'],              // Storage services  
  ['neo4j'],                        // Graph database
  ['enhancedRAG', 'uploadService'],  // Go microservices
  ['sveltekit']                     // Frontend last
];

class ServiceCoordinator {
  constructor() {
    this.runningServices = new Map();
    this.failedServices = new Set();
    this.portAllocations = new Map();
  }

  async start() {
    console.log(chalk.cyan('\n🚀 Starting Legal AI Platform Services\n'));
    
    try {
      await this.checkPortAvailability();
      await this.startServicesInOrder();
      await this.verifyAllServices();
      
      console.log(chalk.green('\n✅ All services started successfully!\n'));
      this.displayServiceStatus();
      
    } catch (error) {
      console.error(chalk.red(`\n❌ Service coordination failed: ${error.message}\n`));
      await this.cleanup();
      process.exit(1);
    }
  }

  async checkPortAvailability() {
    console.log(chalk.yellow('🔍 Checking port availability...'));
    
    for (const [serviceName, config] of Object.entries(SERVICES)) {
      const isAvailable = await this.isPortAvailable(config.port);
      
      if (!isAvailable) {
        if (config.required) {
          // Try to find alternative port
          const alternativePort = await this.findAvailablePort(config.port);
          if (alternativePort) {
            console.log(chalk.yellow(`⚠️  ${serviceName}: Port ${config.port} busy, using ${alternativePort}`));
            this.portAllocations.set(serviceName, alternativePort);
          } else {
            throw new Error(`Required service ${serviceName} cannot find available port (tried ${config.port})`);
          }
        } else {
          console.log(chalk.yellow(`⚠️  ${serviceName}: Port ${config.port} busy, will skip`));
          this.failedServices.add(serviceName);
        }
      } else {
        console.log(chalk.green(`✓ ${serviceName}: Port ${config.port} available`));
      }
    }
  }

  async startServicesInOrder() {
    for (const serviceGroup of STARTUP_ORDER) {
      console.log(chalk.cyan(`\n📦 Starting service group: ${serviceGroup.join(', ')}`));
      
      // Start services in parallel within each group
      const startPromises = serviceGroup
        .filter(serviceName => !this.failedServices.has(serviceName))
        .map(serviceName => this.startService(serviceName));
      
      await Promise.allSettled(startPromises);
      
      // Wait for health checks before proceeding to next group
      await setTimeout(3000);
      
      for (const serviceName of serviceGroup) {
        if (!this.failedServices.has(serviceName)) {
          await this.waitForServiceHealth(serviceName);
        }
      }
    }
  }

  async startService(serviceName) {
    const config = SERVICES[serviceName];
    const port = this.portAllocations.get(serviceName) || config.port;
    
    if (!config.startCommand) {
      console.log(chalk.yellow(`⏭️  ${serviceName}: No start command, assuming external management`));
      return;
    }
    
    try {
      console.log(chalk.blue(`🔧 Starting ${serviceName} on port ${port}...`));
      
      // Modify command if port was reassigned
      let command = config.startCommand;
      if (this.portAllocations.has(serviceName)) {
        command = this.adjustCommandForPort(command, port);
      }
      
      const process = spawn('cmd', ['/c', command], {
        stdio: ['ignore', 'pipe', 'pipe'],
        shell: true,
        detached: false
      });
      
      this.runningServices.set(serviceName, { process, port, startTime: Date.now() });
      
      // Handle process events
      process.on('error', (error) => {
        console.error(chalk.red(`❌ ${serviceName} failed to start: ${error.message}`));
        this.failedServices.add(serviceName);
      });
      
      process.on('exit', (code) => {
        if (code !== 0) {
          console.error(chalk.red(`❌ ${serviceName} exited with code ${code}`));
          this.failedServices.add(serviceName);
        }
      });
      
      // Log output for debugging
      if (process.stdout) {
        process.stdout.on('data', (data) => {
          const output = data.toString().trim();
          if (output.includes('error') || output.includes('failed')) {
            console.log(chalk.red(`[${serviceName}] ${output}`));
          }
        });
      }
      
    } catch (error) {
      console.error(chalk.red(`❌ Failed to start ${serviceName}: ${error.message}`));
      this.failedServices.add(serviceName);
    }
  }

  async waitForServiceHealth(serviceName, maxRetries = 10) {
    const config = SERVICES[serviceName];
    const port = this.portAllocations.get(serviceName) || config.port;
    
    console.log(chalk.blue(`🔍 Waiting for ${serviceName} health check...`));
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const isHealthy = await this.checkServiceHealth(serviceName, port, config.healthCheck);
        
        if (isHealthy) {
          console.log(chalk.green(`✅ ${serviceName} is healthy`));
          return true;
        }
        
        if (attempt < maxRetries) {
          console.log(chalk.yellow(`⏳ ${serviceName} not ready (attempt ${attempt}/${maxRetries}), retrying...`));
          await setTimeout(2000);
        }
        
      } catch (error) {
        console.log(chalk.yellow(`⚠️  ${serviceName} health check failed: ${error.message}`));
      }
    }
    
    console.error(chalk.red(`❌ ${serviceName} health check timeout after ${maxRetries} attempts`));
    this.failedServices.add(serviceName);
    return false;
  }

  async checkServiceHealth(serviceName, port, healthCheckType) {
    switch (healthCheckType) {
      case 'http':
        try {
          const response = await fetch(`http://localhost:${port}/health`).catch(() => null);
          return response && response.ok;
        } catch {
          return false;
        }
      
      case 'postgresql':
        return this.isPortActive(port);
      
      case 'redis':
        return this.isPortActive(port);
      
      case 'ollama':
        try {
          const response = await fetch(`http://localhost:${port}/api/tags`).catch(() => null);
          return response && response.ok;
        } catch {
          return false;
        }
      
      case 'minio':
        return this.isPortActive(port);
      
      case 'qdrant':
        try {
          const response = await fetch(`http://localhost:${port}/collections`).catch(() => null);
          return response && response.status < 500;
        } catch {
          return false;
        }
      
      case 'neo4j':
        return this.isPortActive(port);
      
      case 'quic':
        // QUIC health check is more complex, for now just check if port is bound
        return this.isPortActive(port);
      
      default:
        return this.isPortActive(port);
    }
  }

  async isPortAvailable(port) {
    return new Promise((resolve) => {
      const server = createServer();
      
      server.listen(port, '127.0.0.1', () => {
        server.close(() => resolve(true));
      });
      
      server.on('error', () => resolve(false));
    });
  }

  async isPortActive(port) {
    return new Promise((resolve) => {
      const socket = new (require('net').Socket)();
      
      socket.setTimeout(1000);
      socket.on('connect', () => {
        socket.destroy();
        resolve(true);
      });
      
      socket.on('timeout', () => {
        socket.destroy();
        resolve(false);
      });
      
      socket.on('error', () => {
        resolve(false);
      });
      
      socket.connect(port, '127.0.0.1');
    });
  }

  async findAvailablePort(startPort, maxAttempts = 50) {
    for (let i = 0; i < maxAttempts; i++) {
      const port = startPort + i;
      if (await this.isPortAvailable(port)) {
        return port;
      }
    }
    return null;
  }

  adjustCommandForPort(command, port) {
    // Simple port substitution - can be enhanced for specific services
    if (command.includes('--port')) {
      return command.replace(/--port\s+\d+/, `--port ${port}`);
    }
    
    if (command.includes(':')) {
      return command.replace(/:\d+/, `:${port}`);
    }
    
    return command;
  }

  async verifyAllServices() {
    console.log(chalk.cyan('\n🔍 Verifying all services...'));
    
    const requiredServices = Object.entries(SERVICES)
      .filter(([name, config]) => config.required && !this.failedServices.has(name))
      .map(([name]) => name);
    
    const healthyServices = [];
    const unhealthyServices = [];
    
    for (const serviceName of requiredServices) {
      const config = SERVICES[serviceName];
      const port = this.portAllocations.get(serviceName) || config.port;
      
      const isHealthy = await this.checkServiceHealth(serviceName, port, config.healthCheck);
      
      if (isHealthy) {
        healthyServices.push(serviceName);
      } else {
        unhealthyServices.push(serviceName);
      }
    }
    
    if (unhealthyServices.length > 0) {
      throw new Error(`Required services not healthy: ${unhealthyServices.join(', ')}`);
    }
    
    console.log(chalk.green(`✅ All ${healthyServices.length} required services are healthy`));
  }

  displayServiceStatus() {
    console.log(chalk.cyan('📊 Service Status Summary:'));
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    
    for (const [serviceName, config] of Object.entries(SERVICES)) {
      const port = this.portAllocations.get(serviceName) || config.port;
      const service = this.runningServices.get(serviceName);
      const isFailed = this.failedServices.has(serviceName);
      
      let status = '';
      let statusColor = chalk.gray;
      
      if (isFailed) {
        status = '❌ FAILED';
        statusColor = chalk.red;
      } else if (service) {
        const uptime = Math.round((Date.now() - service.startTime) / 1000);
        status = `✅ RUNNING (${uptime}s)`;
        statusColor = chalk.green;
      } else {
        status = '⏭️  SKIPPED';
        statusColor = chalk.yellow;
      }
      
      console.log(`${statusColor(`${serviceName.padEnd(15)} │ Port ${String(port).padEnd(5)} │ ${status}`)}`);
    }
    
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    
    // Display access points
    console.log(chalk.cyan('\n🌐 Access Points:'));
    const runningServices = Array.from(this.runningServices.keys()).filter(name => !this.failedServices.has(name));
    
    if (runningServices.includes('sveltekit')) {
      const port = this.portAllocations.get('sveltekit') || SERVICES.sveltekit.port;
      console.log(`🎯 Frontend: ${chalk.blue(`http://localhost:${port}`)}`);
    }
    
    if (runningServices.includes('enhancedRAG')) {
      const port = this.portAllocations.get('enhancedRAG') || SERVICES.enhancedRAG.port;
      console.log(`🧠 Enhanced RAG: ${chalk.blue(`http://localhost:${port}/api/rag`)}`);
    }
    
    if (runningServices.includes('uploadService')) {
      const port = this.portAllocations.get('uploadService') || SERVICES.uploadService.port;
      console.log(`📤 Upload API: ${chalk.blue(`http://localhost:${port}/upload`)}`);
    }
    
    if (runningServices.includes('minio')) {
      const port = this.portAllocations.get('minio') || 9001;
      console.log(`🗄️  MinIO Console: ${chalk.blue(`http://localhost:${port}`)}`);
    }
    
    console.log('');
  }

  async cleanup() {
    console.log(chalk.yellow('\n🧹 Cleaning up services...'));
    
    for (const [serviceName, serviceInfo] of this.runningServices) {
      try {
        if (serviceInfo.process && !serviceInfo.process.killed) {
          serviceInfo.process.kill();
          console.log(chalk.gray(`🔄 Stopped ${serviceName}`));
        }
      } catch (error) {
        console.log(chalk.yellow(`⚠️  Error stopping ${serviceName}: ${error.message}`));
      }
    }
    
    this.runningServices.clear();
  }

  // Graceful shutdown handler
  setupGracefulShutdown() {
    const shutdown = async (signal) => {
      console.log(chalk.yellow(`\n🛑 Received ${signal}, shutting down gracefully...`));
      await this.cleanup();
      process.exit(0);
    };

    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGHUP', () => shutdown('SIGHUP'));
  }
}

// Main execution
async function main() {
  const coordinator = new ServiceCoordinator();
  coordinator.setupGracefulShutdown();
  
  try {
    await coordinator.start();
    
    // Keep the process running
    console.log(chalk.green('🎯 All services running. Press Ctrl+C to stop.\n'));
    
    // Optional: Start monitoring
    setInterval(async () => {
      // Basic health monitoring could be added here
    }, 30000);
    
  } catch (error) {
    console.error(chalk.red(`💥 Startup failed: ${error.message}`));
    await coordinator.cleanup();
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { ServiceCoordinator };