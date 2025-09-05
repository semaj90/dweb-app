#!/usr/bin/env node
/**
 * Comprehensive Legal AI System Health Check
 * Tests all services and provides detailed status report
 */

import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ANSI color codes
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

class HealthChecker {
  constructor() {
    this.services = {
      'SvelteKit Frontend': {
        port: 5183,
        health: '/',
        type: 'frontend',
        priority: 'critical'
      },
      'PostgreSQL Database': {
        port: 5432,
        health: null,
        type: 'database',
        priority: 'critical',
        testCmd: 'PGPASSWORD=123456 psql -U postgres -h localhost -d legal_ai_db -c "SELECT 1;" 2>/dev/null'
      },
      'Redis Cache': {
        port: 6379,
        health: null,
        type: 'cache',
        priority: 'high',
        testCmd: 'redis-cli ping 2>/dev/null'
      },
      'Ollama AI Service': {
        port: 11434,
        health: '/api/tags',
        type: 'ai',
        priority: 'critical'
      },
      'MinIO Storage': {
        port: 9000,
        health: '/minio/health/live',
        type: 'storage',
        priority: 'high'
      },
      'Qdrant Vector DB': {
        port: 6333,
        health: '/collections',
        type: 'vector',
        priority: 'high'
      },
      'Enhanced RAG Service': {
        port: 8094,
        health: '/api/health',
        type: 'rag',
        priority: 'critical'
      },
      'Upload Service': {
        port: 8093,
        health: '/health',
        type: 'upload',
        priority: 'medium'
      },
      'Neo4j Graph DB': {
        port: 7474,
        health: '/browser',
        type: 'graph',
        priority: 'medium'
      }
    };
    
    this.results = {};
    this.startTime = Date.now();
  }

  async checkHttpService(name, config) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch(`http://localhost:${config.port}${config.health}`, {
        signal: controller.signal,
        method: 'GET',
        headers: { 'User-Agent': 'Legal-AI-Health-Check/1.0' }
      });
      
      clearTimeout(timeoutId);
      
      return {
        status: response.ok ? 'healthy' : 'unhealthy',
        responseTime: Date.now() - this.startTime,
        httpStatus: response.status,
        details: response.ok ? 'Service responding' : `HTTP ${response.status}`
      };
    } catch (error) {
      return {
        status: 'error',
        responseTime: null,
        httpStatus: null,
        details: error.name === 'AbortError' ? 'Timeout' : error.message
      };
    }
  }

  async checkCommandService(name, config) {
    return new Promise((resolve) => {
      const startTime = Date.now();
      
      const process = spawn('bash', ['-c', config.testCmd], {
        stdio: ['ignore', 'pipe', 'pipe']
      });
      
      let stdout = '';
      let stderr = '';
      
      process.stdout.on('data', (data) => stdout += data.toString());
      process.stderr.on('data', (data) => stderr += data.toString());
      
      process.on('close', (code) => {
        const responseTime = Date.now() - startTime;
        
        if (code === 0) {
          resolve({
            status: 'healthy',
            responseTime,
            httpStatus: null,
            details: stdout.trim() || 'Command executed successfully'
          });
        } else {
          resolve({
            status: 'error',
            responseTime,
            httpStatus: null,
            details: stderr.trim() || `Command failed with code ${code}`
          });
        }
      });
      
      // Timeout after 10 seconds
      setTimeout(() => {
        process.kill('SIGTERM');
        resolve({
          status: 'error',
          responseTime: Date.now() - startTime,
          httpStatus: null,
          details: 'Command timeout'
        });
      }, 10000);
    });
  }

  async checkService(name, config) {
    console.log(`${colors.cyan}Checking ${name}...${colors.reset}`);
    
    let result;
    if (config.testCmd) {
      result = await this.checkCommandService(name, config);
    } else if (config.health) {
      result = await this.checkHttpService(name, config);
    } else {
      result = {
        status: 'unknown',
        responseTime: null,
        httpStatus: null,
        details: 'No health check method defined'
      };
    }
    
    result.service = name;
    result.port = config.port;
    result.type = config.type;
    result.priority = config.priority;
    
    return result;
  }

  async runHealthChecks() {
    console.log(`${colors.bold}${colors.blue}🔍 Legal AI System Health Check${colors.reset}\n`);
    console.log(`${colors.cyan}Starting comprehensive health assessment...${colors.reset}\n`);
    
    // Run all checks concurrently
    const checkPromises = Object.entries(this.services).map(
      async ([name, config]) => {
        const result = await this.checkService(name, config);
        this.results[name] = result;
        return result;
      }
    );
    
    await Promise.all(checkPromises);
    
    this.generateReport();
  }

  generateReport() {
    console.log(`\n${colors.bold}${colors.cyan}📊 Health Check Results${colors.reset}\n`);
    
    const healthyServices = Object.values(this.results).filter(r => r.status === 'healthy').length;
    const totalServices = Object.keys(this.results).length;
    const healthPercentage = Math.round((healthyServices / totalServices) * 100);
    
    // Overall health status
    let overallStatus;
    let statusColor;
    if (healthPercentage >= 90) {
      overallStatus = 'EXCELLENT';
      statusColor = colors.green;
    } else if (healthPercentage >= 70) {
      overallStatus = 'GOOD';
      statusColor = colors.yellow;
    } else if (healthPercentage >= 50) {
      overallStatus = 'FAIR';
      statusColor = colors.yellow;
    } else {
      overallStatus = 'POOR';
      statusColor = colors.red;
    }
    
    console.log(`${colors.bold}System Health: ${statusColor}${overallStatus} (${healthPercentage}%)${colors.reset}`);
    console.log(`${colors.bold}Services: ${healthyServices}/${totalServices} healthy${colors.reset}\n`);
    
    // Service breakdown by priority
    const priorities = ['critical', 'high', 'medium', 'low'];
    
    priorities.forEach(priority => {
      const services = Object.values(this.results).filter(r => r.priority === priority);
      if (services.length === 0) return;
      
      console.log(`${colors.bold}${priority.toUpperCase()} PRIORITY SERVICES:${colors.reset}`);
      
      services.forEach(result => {
        const statusIcon = result.status === 'healthy' ? '✅' : '❌';
        const statusColor = result.status === 'healthy' ? colors.green : colors.red;
        
        console.log(`${statusIcon} ${colors.bold}${result.service}${colors.reset}`);
        console.log(`   Status: ${statusColor}${result.status.toUpperCase()}${colors.reset}`);
        console.log(`   Port: ${result.port}`);
        console.log(`   Type: ${result.type}`);
        
        if (result.responseTime) {
          console.log(`   Response Time: ${result.responseTime}ms`);
        }
        
        if (result.httpStatus) {
          console.log(`   HTTP Status: ${result.httpStatus}`);
        }
        
        if (result.details) {
          console.log(`   Details: ${result.details}`);
        }
        
        console.log('');
      });
    });
    
    // Service startup recommendations
    this.generateStartupRecommendations();
    
    // Save report to file
    this.saveReport();
  }

  generateStartupRecommendations() {
    const failedServices = Object.values(this.results).filter(r => r.status !== 'healthy');
    
    if (failedServices.length === 0) {
      console.log(`${colors.green}${colors.bold}🎉 All services are running optimally!${colors.reset}`);
      console.log(`${colors.cyan}Frontend: http://localhost:5183${colors.reset}`);
      console.log(`${colors.cyan}MinIO Console: http://localhost:9001${colors.reset}`);
      console.log(`${colors.cyan}Neo4j Browser: http://localhost:7474${colors.reset}`);
      return;
    }
    
    console.log(`${colors.bold}${colors.yellow}🔧 Service Startup Recommendations${colors.reset}\n`);
    
    const criticalFailed = failedServices.filter(s => s.priority === 'critical');
    const highFailed = failedServices.filter(s => s.priority === 'high');
    
    if (criticalFailed.length > 0) {
      console.log(`${colors.red}⚠️  Critical services offline - system functionality limited:${colors.reset}`);
      criticalFailed.forEach(service => {
        console.log(`   • ${service.service}: ${this.getStartupCommand(service)}`);
      });
      console.log('');
    }
    
    if (highFailed.length > 0) {
      console.log(`${colors.yellow}⚠️  High priority services offline - reduced functionality:${colors.reset}`);
      highFailed.forEach(service => {
        console.log(`   • ${service.service}: ${this.getStartupCommand(service)}`);
      });
      console.log('');
    }
    
    console.log(`${colors.cyan}Quick startup commands:${colors.reset}`);
    console.log(`   npm run ollama:start     # Start Ollama AI service`);
    console.log(`   npm run redis:start      # Start Redis cache`);
    console.log(`   npm run neo4j:start      # Start Neo4j graph database`);
    console.log('');
    
    console.log(`${colors.cyan}Full system startup:${colors.reset}`);
    console.log(`   npm run dev:full         # Start all services`);
    console.log(`   START-LEGAL-AI.bat       # Windows batch startup`);
  }

  getStartupCommand(service) {
    const commands = {
      'PostgreSQL Database': 'Start PostgreSQL service',
      'Redis Cache': 'npm run redis:start',
      'Ollama AI Service': 'npm run ollama:start', 
      'MinIO Storage': 'Start MinIO server',
      'Qdrant Vector DB': 'Start Qdrant service',
      'Enhanced RAG Service': 'Start Go RAG microservice',
      'Upload Service': 'Start Go upload microservice',
      'Neo4j Graph DB': 'npm run neo4j:start'
    };
    
    return commands[service.service] || 'Check service documentation';
  }

  async saveReport() {
    const reportData = {
      timestamp: new Date().toISOString(),
      healthPercentage: Math.round((Object.values(this.results).filter(r => r.status === 'healthy').length / Object.keys(this.results).length) * 100),
      services: this.results,
      summary: {
        total: Object.keys(this.results).length,
        healthy: Object.values(this.results).filter(r => r.status === 'healthy').length,
        unhealthy: Object.values(this.results).filter(r => r.status !== 'healthy').length
      }
    };
    
    try {
      const reportsDir = join(__dirname, '..', 'logs');
      await fs.mkdir(reportsDir, { recursive: true });
      
      const reportPath = join(reportsDir, 'health-report.json');
      await fs.writeFile(reportPath, JSON.stringify(reportData, null, 2));
      
      console.log(`${colors.cyan}📁 Health report saved to: logs/health-report.json${colors.reset}\n`);
    } catch (error) {
      console.log(`${colors.yellow}⚠️  Could not save health report: ${error.message}${colors.reset}\n`);
    }
  }
}

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  const checker = new HealthChecker();
  checker.runHealthChecks().catch(console.error);
}

export { HealthChecker };