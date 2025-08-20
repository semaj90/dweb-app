#!/usr/bin/env node

/**
 * Dynamic Port Manager - Vite-style port discovery for Go microservices
 * Implements Context7 best practices for service orchestration
 */

import net from 'net';
import fs from 'fs/promises';
import path from 'path';

const DEFAULT_PORTS = {
  'enhanced-rag': 8094,
  'upload-service': 8093,
  'quic-gateway': 8447,
  'kratos-server': 50051,
  'cluster-manager': 8213,
  'xstate-manager': 8212,
  'load-balancer': 8222,
  'vector-service': 8095,
  'gpu-service': 8096
};

const PORT_RANGE_MAX = 50; // Search up to 50 ports ahead

class DynamicPortManager {
  constructor() {
    this.allocatedPorts = new Map();
    this.configFile = '.vscode/dynamic-ports.json';
    this.lockFile = '.vscode/port-manager.lock';
  }

  /**
   * Check if a port is available (Vite-style)
   */
  async isPortAvailable(port) {
    return new Promise((resolve) => {
      const server = net.createServer();
      
      server.listen(port, (err) => {
        if (err) {
          resolve(false);
        } else {
          server.close(() => resolve(true));
        }
      });
      
      server.on('error', () => resolve(false));
    });
  }

  /**
   * Find next available port starting from preferred port
   */
  async findAvailablePort(serviceName, preferredPort = null) {
    const startPort = preferredPort || DEFAULT_PORTS[serviceName] || 8080;
    
    console.log(`🔍 Finding available port for ${serviceName} (preferred: ${startPort})`);
    
    for (let i = 0; i < PORT_RANGE_MAX; i++) {
      const port = startPort + i;
      
      if (await this.isPortAvailable(port)) {
        console.log(`✅ Port ${port} available for ${serviceName}`);
        this.allocatedPorts.set(serviceName, port);
        await this.savePortConfiguration();
        return port;
      } else {
        console.log(`⚠️  Port ${port} occupied, trying next...`);
      }
    }
    
    throw new Error(`❌ No available port found for ${serviceName} starting from ${startPort}`);
  }

  /**
   * Allocate ports for multiple services in batch
   */
  async allocateServicePorts(services) {
    const portMap = new Map();
    const occupiedPorts = new Set();
    
    console.log(`🚀 Allocating ports for services: ${services.join(', ')}`);
    
    for (const serviceName of services) {
      try {
        const preferredPort = DEFAULT_PORTS[serviceName];
        let allocatedPort = preferredPort;
        
        // Check if preferred port is available and not already allocated
        if (!await this.isPortAvailable(preferredPort) || occupiedPorts.has(preferredPort)) {
          // Find alternative port
          for (let i = 1; i <= PORT_RANGE_MAX; i++) {
            const candidatePort = preferredPort + i;
            if (await this.isPortAvailable(candidatePort) && !occupiedPorts.has(candidatePort)) {
              allocatedPort = candidatePort;
              break;
            }
          }
        }
        
        if (allocatedPort !== preferredPort) {
          console.log(`🔄 ${serviceName}: Port ${preferredPort} occupied, using ${allocatedPort}`);
        } else {
          console.log(`✅ ${serviceName}: Using preferred port ${allocatedPort}`);
        }
        
        portMap.set(serviceName, allocatedPort);
        occupiedPorts.add(allocatedPort);
        this.allocatedPorts.set(serviceName, allocatedPort);
        
      } catch (error) {
        console.error(`❌ Failed to allocate port for ${serviceName}: ${error.message}`);
        throw error;
      }
    }
    
    await this.savePortConfiguration();
    return portMap;
  }

  /**
   * Save current port configuration to file
   */
  async savePortConfiguration() {
    const config = {
      timestamp: new Date().toISOString(),
      ports: Object.fromEntries(this.allocatedPorts),
      metadata: {
        portRangeMax: PORT_RANGE_MAX,
        defaultPorts: DEFAULT_PORTS,
        generator: 'dynamic-port-manager.js'
      }
    };
    
    try {
      await fs.mkdir(path.dirname(this.configFile), { recursive: true });
      await fs.writeFile(this.configFile, JSON.stringify(config, null, 2));
      console.log(`💾 Port configuration saved to ${this.configFile}`);
    } catch (error) {
      console.error(`❌ Failed to save port configuration: ${error.message}`);
    }
  }

  /**
   * Load existing port configuration
   */
  async loadPortConfiguration() {
    try {
      const data = await fs.readFile(this.configFile, 'utf8');
      const config = JSON.parse(data);
      
      // Verify ports are still available
      const validPorts = new Map();
      
      for (const [serviceName, port] of Object.entries(config.ports)) {
        if (await this.isPortAvailable(port)) {
          validPorts.set(serviceName, port);
          console.log(`♻️  Reusing port ${port} for ${serviceName}`);
        } else {
          console.log(`⚠️  Previously allocated port ${port} for ${serviceName} is now occupied`);
        }
      }
      
      this.allocatedPorts = validPorts;
      return validPorts;
      
    } catch (error) {
      console.log(`ℹ️  No existing port configuration found, starting fresh`);
      return new Map();
    }
  }

  /**
   * Generate environment variables for services
   */
  generateEnvVariables() {
    const envVars = {};
    
    for (const [serviceName, port] of this.allocatedPorts) {
      switch (serviceName) {
        case 'enhanced-rag':
          envVars.RAG_HTTP_PORT = port.toString();
          break;
        case 'upload-service':
          envVars.UPLOAD_PORT = port.toString();
          break;
        case 'quic-gateway':
          envVars.QUIC_HTTP_PORT = port.toString();
          envVars.QUIC_UDP_PORT = (port + 1000).toString(); // UDP offset
          break;
        case 'kratos-server':
          envVars.GRPC_PORT = port.toString();
          break;
        case 'cluster-manager':
          envVars.CLUSTER_PORT = port.toString();
          break;
        case 'xstate-manager':
          envVars.XSTATE_PORT = port.toString();
          break;
        case 'load-balancer':
          envVars.LB_PORT = port.toString();
          break;
        case 'vector-service':
          envVars.VECTOR_PORT = port.toString();
          break;
        case 'gpu-service':
          envVars.GPU_PORT = port.toString();
          break;
        default:
          envVars[`${serviceName.toUpperCase().replace('-', '_')}_PORT`] = port.toString();
      }
    }
    
    return envVars;
  }

  /**
   * Generate Vite proxy configuration
   */
  generateViteProxyConfig() {
    const proxyConfig = {};
    
    for (const [serviceName, port] of this.allocatedPorts) {
      switch (serviceName) {
        case 'enhanced-rag':
          proxyConfig['/api/go/enhanced-rag'] = {
            target: `http://localhost:${port}`,
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/api\/go\/enhanced-rag/, '')
          };
          break;
        case 'upload-service':
          proxyConfig['/api/go/upload'] = {
            target: `http://localhost:${port}`,
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/api\/go\/upload/, '')
          };
          break;
        case 'quic-gateway':
          proxyConfig['/api/quic'] = {
            target: `http://localhost:${port}`,
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/api\/quic/, '')
          };
          break;
        case 'kratos-server':
          proxyConfig['/api/grpc'] = {
            target: `http://localhost:${port}`,
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/api\/grpc/, '')
          };
          break;
      }
    }
    
    return proxyConfig;
  }

  /**
   * Kill processes using specific ports (Windows-compatible)
   */
  async killPortProcesses(ports) {
    const { spawn } = await import('child_process');
    
    for (const port of ports) {
      try {
        console.log(`🔫 Attempting to free port ${port}...`);
        
        // Windows: Use netstat and taskkill
        const netstatProcess = spawn('netstat', ['-ano'], { shell: true });
        let output = '';
        
        netstatProcess.stdout.on('data', (data) => {
          output += data.toString();
        });
        
        netstatProcess.on('close', () => {
          const lines = output.split('\n');
          const portLine = lines.find(line => line.includes(`:${port} `));
          
          if (portLine) {
            const parts = portLine.trim().split(/\s+/);
            const pid = parts[parts.length - 1];
            
            if (pid && pid !== '0' && /^\d+$/.test(pid)) {
              console.log(`🎯 Found process ${pid} using port ${port}, terminating...`);
              spawn('taskkill', ['/PID', pid, '/F'], { shell: true });
            }
          }
        });
        
      } catch (error) {
        console.log(`⚠️  Could not kill process on port ${port}: ${error.message}`);
      }
    }
    
    // Wait for processes to terminate
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  /**
   * CLI interface
   */
  async runCLI() {
    const args = process.argv.slice(2);
    const command = args[0];
    
    switch (command) {
      case 'allocate': {
        const services = args.slice(1);
        if (services.length === 0) {
          console.log('Usage: node dynamic-port-manager.js allocate <service1> <service2> ...');
          process.exit(1);
        }
        
        await this.loadPortConfiguration();
        const portMap = await this.allocateServicePorts(services);
        
        console.log('\n🎯 Port Allocation Results:');
        for (const [service, port] of portMap) {
          console.log(`  ${service}: ${port}`);
        }
        
        console.log('\n📋 Environment Variables:');
        const envVars = this.generateEnvVariables();
        for (const [key, value] of Object.entries(envVars)) {
          console.log(`  export ${key}=${value}`);
        }
        break;
      }
      
      case 'kill-ports': {
        const ports = args.slice(1).map(Number);
        if (ports.length === 0) {
          console.log('Usage: node dynamic-port-manager.js kill-ports <port1> <port2> ...');
          process.exit(1);
        }
        
        await this.killPortProcesses(ports);
        break;
      }
      
      case 'status': {
        await this.loadPortConfiguration();
        
        console.log('\n📊 Current Port Allocations:');
        if (this.allocatedPorts.size === 0) {
          console.log('  No ports currently allocated');
        } else {
          for (const [service, port] of this.allocatedPorts) {
            const available = await this.isPortAvailable(port);
            const status = available ? '🟢 Available' : '🔴 In Use';
            console.log(`  ${service}: ${port} (${status})`);
          }
        }
        break;
      }
      
      case 'vite-config': {
        await this.loadPortConfiguration();
        const proxyConfig = this.generateViteProxyConfig();
        
        console.log('\n⚡ Vite Proxy Configuration:');
        console.log(JSON.stringify(proxyConfig, null, 2));
        break;
      }
      
      default:
        console.log(`
🚀 Dynamic Port Manager - Vite-style port discovery for Go microservices

Usage:
  node dynamic-port-manager.js allocate <service1> <service2> ...  # Allocate ports for services
  node dynamic-port-manager.js kill-ports <port1> <port2> ...     # Kill processes using ports
  node dynamic-port-manager.js status                             # Show current allocations
  node dynamic-port-manager.js vite-config                        # Generate Vite proxy config

Services:
  enhanced-rag, upload-service, quic-gateway, kratos-server,
  cluster-manager, xstate-manager, load-balancer, vector-service, gpu-service

Examples:
  node dynamic-port-manager.js allocate enhanced-rag upload-service
  node dynamic-port-manager.js kill-ports 8093 8094 8447
  node dynamic-port-manager.js status
        `);
    }
  }
}

// Run CLI if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const manager = new DynamicPortManager();
  manager.runCLI().catch(console.error);
}

export default DynamicPortManager;