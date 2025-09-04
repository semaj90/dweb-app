#!/usr/bin/env node
/**
 * Legal AI System Startup Progress Monitor
 * Real-time monitoring with progress bars for all services
 */

import { spawn } from 'child_process';
import { createWriteStream } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ANSI color codes for terminal output
const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

class ProgressBar {
  constructor(total, label, width = 40) {
    this.total = total;
    this.current = 0;
    this.label = label;
    this.width = width;
    this.startTime = Date.now();
  }

  update(current, status = '') {
    this.current = current;
    const percentage = Math.round((current / this.total) * 100);
    const filledWidth = Math.round((current / this.total) * this.width);
    const emptyWidth = this.width - filledWidth;
    
    const filled = '█'.repeat(filledWidth);
    const empty = '░'.repeat(emptyWidth);
    const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);
    
    process.stdout.write('\r');
    process.stdout.write(
      `${colors.cyan}${this.label}${colors.reset} [${colors.green}${filled}${colors.white}${empty}${colors.reset}] ${percentage}% (${elapsed}s) ${status}`
    );
    
    if (current >= this.total) {
      console.log(''); // New line when complete
    }
  }

  complete(message = 'Complete') {
    this.update(this.total, `${colors.green}✅ ${message}${colors.reset}`);
  }

  error(message = 'Error') {
    process.stdout.write('\r');
    console.log(`${colors.cyan}${this.label}${colors.reset} [${colors.red}❌ ${message}${colors.reset}]`);
  }
}

class ServiceMonitor {
  constructor() {
    this.services = {
      'PostgreSQL': { port: 5432, health: '/api/health', expected: 'database' },
      'Redis': { port: 6379, health: '/ping', expected: 'cache' },
      'Ollama': { port: 11434, health: '/api/tags', expected: 'ai' },
      'MinIO': { port: 9000, health: '/minio/health/live', expected: 'storage' },
      'Qdrant': { port: 6333, health: '/collections', expected: 'vector' },
      'Enhanced RAG': { port: 8094, health: '/api/health', expected: 'rag' },
      'Upload Service': { port: 8093, health: '/health', expected: 'upload' },
      'SvelteKit': { port: 5173, health: '/api/health', expected: 'frontend' },
      'Neo4j': { port: 7474, health: '/browser', expected: 'graph' }
    };
    
    this.progressBars = {};
    this.serviceStatus = {};
    this.logStream = createWriteStream(join(__dirname, '..', 'logs', 'startup-monitor.log'), { flags: 'a' });
  }

  async checkService(name, config) {
    try {
      const response = await fetch(`http://localhost:${config.port}${config.health}`, {
        method: 'GET',
        timeout: 5000
      });
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  async monitorStartup() {
    console.log(`${colors.bold}${colors.blue}🚀 Legal AI System Startup Monitor${colors.reset}\n`);
    
    // Initialize progress bars
    Object.keys(this.services).forEach(serviceName => {
      this.progressBars[serviceName] = new ProgressBar(100, serviceName.padEnd(15));
      this.serviceStatus[serviceName] = 'starting';
    });

    // Start monitoring
    const monitorInterval = setInterval(async () => {
      let allServicesReady = true;
      
      for (const [serviceName, config] of Object.entries(this.services)) {
        if (this.serviceStatus[serviceName] === 'ready') continue;
        
        const isHealthy = await this.checkService(serviceName, config);
        
        if (isHealthy) {
          this.progressBars[serviceName].complete(`Running on port ${config.port}`);
          this.serviceStatus[serviceName] = 'ready';
          this.log(`✅ ${serviceName} is ready on port ${config.port}`);
        } else {
          this.progressBars[serviceName].update(50, `${colors.yellow}Waiting...${colors.reset}`);
          allServicesReady = false;
        }
      }
      
      if (allServicesReady) {
        clearInterval(monitorInterval);
        await this.performHealthCheck();
      }
    }, 2000);

    // Timeout after 5 minutes
    setTimeout(() => {
      clearInterval(monitorInterval);
      this.showFailedServices();
    }, 300000);
  }

  async performHealthCheck() {
    console.log(`\n${colors.bold}${colors.green}🔍 Performing Comprehensive Health Check${colors.reset}\n`);
    
    const healthResults = {};
    
    for (const [serviceName, config] of Object.entries(this.services)) {
      const healthBar = new ProgressBar(100, `Health: ${serviceName}`.padEnd(20));
      
      try {
        const startTime = Date.now();
        const response = await fetch(`http://localhost:${config.port}${config.health}`, {
          method: 'GET',
          timeout: 10000
        });
        const responseTime = Date.now() - startTime;
        
        if (response.ok) {
          healthBar.complete(`${responseTime}ms - Healthy`);
          healthResults[serviceName] = { status: 'healthy', responseTime };
        } else {
          healthBar.error(`HTTP ${response.status}`);
          healthResults[serviceName] = { status: 'unhealthy', error: `HTTP ${response.status}` };
        }
      } catch (error) {
        healthBar.error(`Connection failed: ${error.message}`);
        healthResults[serviceName] = { status: 'error', error: error.message };
      }
    }
    
    this.generateHealthReport(healthResults);
  }

  generateHealthReport(results) {
    console.log(`\n${colors.bold}${colors.cyan}📊 System Health Report${colors.reset}\n`);
    
    const healthyServices = Object.values(results).filter(r => r.status === 'healthy').length;
    const totalServices = Object.keys(results).length;
    const healthPercentage = Math.round((healthyServices / totalServices) * 100);
    
    console.log(`${colors.bold}Overall System Health: ${healthPercentage}% (${healthyServices}/${totalServices} services)${colors.reset}\n`);
    
    // Service breakdown
    Object.entries(results).forEach(([service, result]) => {
      const statusColor = result.status === 'healthy' ? colors.green : colors.red;
      const statusIcon = result.status === 'healthy' ? '✅' : '❌';
      
      console.log(`${statusIcon} ${colors.bold}${service}${colors.reset}: ${statusColor}${result.status.toUpperCase()}${colors.reset}`);
      
      if (result.responseTime) {
        console.log(`   Response Time: ${result.responseTime}ms`);
      }
      
      if (result.error) {
        console.log(`   Error: ${colors.red}${result.error}${colors.reset}`);
      }
      
      console.log('');
    });
    
    // System recommendations
    this.generateRecommendations(results);
  }

  generateRecommendations(results) {
    console.log(`${colors.bold}${colors.yellow}💡 Recommendations${colors.reset}\n`);
    
    const failedServices = Object.entries(results)
      .filter(([_, result]) => result.status !== 'healthy')
      .map(([service, _]) => service);
    
    if (failedServices.length === 0) {
      console.log(`${colors.green}🎉 All services are running optimally!${colors.reset}`);
      console.log(`${colors.cyan}🌐 Frontend available at: http://localhost:5173${colors.reset}`);
      console.log(`${colors.cyan}📊 MinIO Console at: http://localhost:9001${colors.reset}`);
      console.log(`${colors.cyan}🗄️  Neo4j Browser at: http://localhost:7474${colors.reset}`);
    } else {
      console.log(`${colors.red}⚠️  ${failedServices.length} service(s) need attention:${colors.reset}`);
      
      failedServices.forEach(service => {
        const config = this.services[service];
        console.log(`   • ${service}: Try starting manually or check port ${config.port}`);
      });
      
      console.log(`\n${colors.yellow}🔧 Quick fixes to try:${colors.reset}`);
      console.log(`   • Run: npm run services:status`);
      console.log(`   • Run: npm run fix:ports`);
      console.log(`   • Check logs: npm run logs:all`);
    }
  }

  showFailedServices() {
    console.log(`\n${colors.bold}${colors.red}⏰ Startup Timeout Reached${colors.reset}\n`);
    
    const failedServices = Object.entries(this.serviceStatus)
      .filter(([_, status]) => status !== 'ready')
      .map(([service, _]) => service);
    
    if (failedServices.length > 0) {
      console.log(`${colors.red}❌ Services that failed to start:${colors.reset}`);
      failedServices.forEach(service => {
        this.progressBars[service].error('Startup timeout');
      });
    }
  }

  log(message) {
    const timestamp = new Date().toISOString();
    this.logStream.write(`[${timestamp}] ${message}\n`);
  }
}

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  const monitor = new ServiceMonitor();
  monitor.monitorStartup().catch(console.error);
}

export { ServiceMonitor };