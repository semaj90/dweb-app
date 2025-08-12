// health-check.mjs
// Comprehensive health check for all services

import http from 'http';
import https from 'https';
import net from 'net';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';

const execAsync = promisify(exec);

class HealthChecker {
  constructor() {
    this.results = {
      timestamp: new Date().toISOString(),
      services: {},
      dependencies: {},
      system: {},
      errors: [],
      warnings: []
    };
  }

  async checkPort(host, port, timeout = 1000) {
    return new Promise((resolve) => {
      const socket = new net.Socket();
      socket.setTimeout(timeout);
      
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
      
      socket.connect(port, host);
    });
  }

  async checkHttp(url, options = {}) {
    return new Promise((resolve) => {
      const timeout = options.timeout || 3000;
      const timer = setTimeout(() => resolve({ success: false, error: 'Timeout' }), timeout);
      
      const protocol = url.startsWith('https') ? https : http;
      
      protocol.get(url, (res) => {
        clearTimeout(timer);
        let data = '';
        
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          resolve({
            success: res.statusCode >= 200 && res.statusCode < 400,
            statusCode: res.statusCode,
            data: data.substring(0, 1000) // Limit response size
          });
        });
      }).on('error', (err) => {
        clearTimeout(timer);
        resolve({ success: false, error: err.message });
      });
    });
  }

  async checkFrontend() {
    console.log('🔍 Checking Frontend...');
    const port = 5173;
    const portOpen = await this.checkPort('localhost', port);
    
    if (portOpen) {
      const httpCheck = await this.checkHttp(`http://localhost:${port}`);
      this.results.services.frontend = {
        status: httpCheck.success ? 'healthy' : 'degraded',
        port: port,
        url: `http://localhost:${port}`,
        details: httpCheck
      };
    } else {
      this.results.services.frontend = {
        status: 'down',
        port: port,
        error: 'Port not reachable'
      };
      this.results.errors.push('Frontend service is not running');
    }
  }

  async checkGoAPI() {
    console.log('🔍 Checking Go API...');
    const port = 8084;
    const portOpen = await this.checkPort('localhost', port);
    
    if (portOpen) {
      const healthCheck = await this.checkHttp(`http://localhost:${port}/api/health`);
      const metricsCheck = await this.checkHttp(`http://localhost:${port}/api/metrics`);
      
      this.results.services.goAPI = {
        status: healthCheck.success ? 'healthy' : 'degraded',
        port: port,
        endpoints: {
          health: healthCheck.success,
          metrics: metricsCheck.success
        },
        gpu: false // Will be updated if GPU info is in health response
      };
      
      // Try to parse health response for GPU info
      if (healthCheck.data) {
        try {
          const data = JSON.parse(healthCheck.data);
          this.results.services.goAPI.gpu = data.gpu?.enabled || false;
        } catch {}
      }
    } else {
      this.results.services.goAPI = {
        status: 'down',
        port: port,
        error: 'Port not reachable'
      };
      this.results.warnings.push('Go API service is not running - AI features limited');
    }
  }

  async checkRedis() {
    console.log('🔍 Checking Redis...');
    const port = 6379;
    const portOpen = await this.checkPort('localhost', port);
    
    if (portOpen) {
      try {
        const { stdout } = await execAsync('redis-cli ping');
        this.results.services.redis = {
          status: stdout.trim() === 'PONG' ? 'healthy' : 'degraded',
          port: port
        };
      } catch {
        this.results.services.redis = {
          status: 'degraded',
          port: port,
          error: 'CLI check failed'
        };
      }
    } else {
      this.results.services.redis = {
        status: 'down',
        port: port
      };
      this.results.warnings.push('Redis not running - using memory cache');
    }
  }

  async checkOllama() {
    console.log('🔍 Checking Ollama...');
    const port = 11434;
    const portOpen = await this.checkPort('localhost', port);
    
    if (portOpen) {
      const apiCheck = await this.checkHttp(`http://localhost:${port}/api/tags`);
      
      if (apiCheck.success) {
        try {
          const data = JSON.parse(apiCheck.data);
          const models = data.models || [];
          const hasGemma = models.some(m => m.name?.includes('gemma'));
          
          this.results.services.ollama = {
            status: 'healthy',
            port: port,
            models: models.map(m => m.name),
            hasGemmaModel: hasGemma
          };
          
          if (!hasGemma) {
            this.results.warnings.push('Gemma3-legal model not found - run: ollama pull gemma3-legal:latest');
          }
        } catch {
          this.results.services.ollama = {
            status: 'degraded',
            port: port,
            error: 'API response parse error'
          };
        }
      } else {
        this.results.services.ollama = {
          status: 'degraded',
          port: port,
          error: 'API not responding'
        };
      }
    } else {
      this.results.services.ollama = {
        status: 'down',
        port: port
      };
      this.results.warnings.push('Ollama not running - AI features disabled');
    }
  }

  async checkDependencies() {
    console.log('🔍 Checking Dependencies...');
    
    // Check package.json exists
    const packagePath = path.join(process.cwd(), 'package.json');
    try {
      const packageJson = JSON.parse(await fs.readFile(packagePath, 'utf-8'));
      this.results.dependencies.packageJson = true;
      
      // Check node_modules
      try {
        await fs.access(path.join(process.cwd(), 'node_modules'));
        this.results.dependencies.nodeModules = true;
      } catch {
        this.results.dependencies.nodeModules = false;
        this.results.errors.push('node_modules not found - run: npm install');
      }
      
      // Check for outdated packages
      try {
        const { stdout } = await execAsync('npm outdated --json');
        const outdated = stdout ? JSON.parse(stdout) : {};
        this.results.dependencies.outdated = Object.keys(outdated).length;
        
        if (this.results.dependencies.outdated > 10) {
          this.results.warnings.push(`${this.results.dependencies.outdated} packages are outdated`);
        }
      } catch {
        // npm outdated returns non-zero exit code when packages are outdated
        this.results.dependencies.outdated = 'unknown';
      }
    } catch (error) {
      this.results.dependencies.packageJson = false;
      this.results.errors.push('package.json not found');
    }
  }

  async checkSystem() {
    console.log('🔍 Checking System...');
    
    // Node.js version
    try {
      const { stdout } = await execAsync('node --version');
      this.results.system.nodeVersion = stdout.trim();
      
      const major = parseInt(stdout.match(/v(\d+)/)?.[1] || '0');
      if (major < 18) {
        this.results.errors.push(`Node.js 18+ required (current: ${stdout.trim()})`);
      }
    } catch {
      this.results.system.nodeVersion = 'unknown';
    }
    
    // npm version
    try {
      const { stdout } = await execAsync('npm --version');
      this.results.system.npmVersion = stdout.trim();
    } catch {
      this.results.system.npmVersion = 'unknown';
    }
    
    // Go version
    try {
      const { stdout } = await execAsync('go version');
      this.results.system.goVersion = stdout.trim();
    } catch {
      this.results.system.goVersion = 'not installed';
    }
    
    // GPU availability
    try {
      const { stdout } = await execAsync('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader');
      const [name, driver, memory] = stdout.trim().split(',').map(s => s.trim());
      this.results.system.gpu = {
        available: true,
        name,
        driver,
        memory
      };
    } catch {
      this.results.system.gpu = { available: false };
    }
    
    // Available memory
    if (process.platform === 'win32') {
      try {
        const { stdout } = await execAsync('wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value');
        const lines = stdout.split('\n');
        const free = parseInt(lines.find(l => l.startsWith('FreePhysicalMemory'))?.split('=')[1] || '0') / 1024;
        const total = parseInt(lines.find(l => l.startsWith('TotalVisibleMemorySize'))?.split('=')[1] || '0') / 1024;
        
        this.results.system.memory = {
          free: Math.round(free),
          total: Math.round(total),
          used: Math.round(total - free)
        };
        
        if (free < 1000) {
          this.results.warnings.push('Low system memory available');
        }
      } catch {
        this.results.system.memory = { error: 'Unable to check' };
      }
    }
  }

  printReport() {
    console.log('\n' + '═'.repeat(60));
    console.log('HEALTH CHECK REPORT');
    console.log('═'.repeat(60));
    
    // Services
    console.log('\n📡 SERVICES:');
    for (const [name, info] of Object.entries(this.results.services)) {
      const statusIcon = info.status === 'healthy' ? '✅' : info.status === 'degraded' ? '⚠️' : '❌';
      console.log(`  ${statusIcon} ${name}: ${info.status}`);
      if (info.error) {
        console.log(`     └─ ${info.error}`);
      }
    }
    
    // System
    console.log('\n💻 SYSTEM:');
    console.log(`  Node.js: ${this.results.system.nodeVersion}`);
    console.log(`  npm: ${this.results.system.npmVersion}`);
    console.log(`  Go: ${this.results.system.goVersion}`);
    if (this.results.system.gpu?.available) {
      console.log(`  GPU: ${this.results.system.gpu.name} (${this.results.system.gpu.memory})`);
    } else {
      console.log(`  GPU: Not available`);
    }
    
    // Dependencies
    console.log('\n📦 DEPENDENCIES:');
    console.log(`  package.json: ${this.results.dependencies.packageJson ? '✅' : '❌'}`);
    console.log(`  node_modules: ${this.results.dependencies.nodeModules ? '✅' : '❌'}`);
    if (this.results.dependencies.outdated !== 'unknown') {
      console.log(`  Outdated packages: ${this.results.dependencies.outdated}`);
    }
    
    // Errors and Warnings
    if (this.results.errors.length > 0) {
      console.log('\n❌ ERRORS:');
      this.results.errors.forEach(err => console.log(`  • ${err}`));
    }
    
    if (this.results.warnings.length > 0) {
      console.log('\n⚠️  WARNINGS:');
      this.results.warnings.forEach(warn => console.log(`  • ${warn}`));
    }
    
    // Summary
    const healthyCount = Object.values(this.results.services).filter(s => s.status === 'healthy').length;
    const totalCount = Object.keys(this.results.services).length;
    const healthPercentage = Math.round((healthyCount / totalCount) * 100);
    
    console.log('\n' + '═'.repeat(60));
    if (this.results.errors.length === 0 && healthPercentage === 100) {
      console.log('✅ All systems operational!');
    } else if (this.results.errors.length === 0) {
      console.log(`⚠️  System partially operational (${healthPercentage}% healthy)`);
    } else {
      console.log(`❌ System has critical issues (${this.results.errors.length} errors)`);
    }
    console.log('═'.repeat(60));
    
    // Save report to file
    this.saveReport();
    
    // Exit with appropriate code
    process.exit(this.results.errors.length > 0 ? 1 : 0);
  }

  async saveReport() {
    const reportPath = path.join(process.cwd(), 'health-report.json');
    try {
      await fs.writeFile(reportPath, JSON.stringify(this.results, null, 2));
      console.log(`\n📄 Report saved to: ${reportPath}`);
    } catch (error) {
      console.error('Failed to save report:', error.message);
    }
  }

  async run() {
    console.log('🏥 Starting Health Check...\n');
    
    await this.checkFrontend();
    await this.checkGoAPI();
    await this.checkRedis();
    await this.checkOllama();
    await this.checkDependencies();
    await this.checkSystem();
    
    this.printReport();
  }
}

// Run health check
const checker = new HealthChecker();
checker.run();
