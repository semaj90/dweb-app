#!/usr/bin/env node
// Complete 38 Microservices Orchestrator - npm run dev:full Integration
// Enhanced Enterprise Development Environment with All Services

import { spawn, exec } from 'child_process';
import { promisify } from 'util';
import { readFileSync, existsSync, writeFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const execAsync = promisify(exec);

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

const log = {
  info: (msg) => console.log(`${colors.blue}ℹ${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✓${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}✗${colors.reset} ${msg}`),
  section: (msg) => console.log(`\n${colors.cyan}${colors.bright}=== ${msg} ===${colors.reset}`)
};

class CompleteMicroservicesOrchestrator {
  constructor() {
    this.services = new Map();
    this.healthChecks = new Map();
    this.startTime = Date.now();
    this.config = {
      maxStartupTime: 300000, // 5 minutes
      healthCheckInterval: 10000, // 10 seconds
      enableMetrics: true,
      enableLogging: true
    };
  }

  // Complete 38 Microservices Configuration
  getServiceConfiguration() {
    return {
      // Tier 1: Core Infrastructure Services (Must Start First)
      tier1: [
        { name: 'PostgreSQL', command: 'pg_isready -h localhost -p 5432', successPattern: 'accepting connections' },
        { name: 'Redis', command: 'redis-cli ping', successPattern: 'PONG' },
        { name: 'RabbitMQ', command: 'rabbitmq-diagnostics status', successPattern: 'Status of node' }
      ],

      // Tier 2: AI/RAG Processing Services (12 services)
      tier2: [
        { name: 'enhanced-rag', binary: 'enhanced-rag.exe', port: 8094, path: '../go-microservice/bin' },
        { name: 'enhanced-rag-service', binary: 'enhanced-rag-service.exe', port: 8195, path: '../go-microservice/bin' },
        { name: 'ai-enhanced', binary: 'ai-enhanced.exe', port: 8096, path: '../ai-summary-service' },
        { name: 'ai-enhanced-final', binary: 'ai-enhanced-final.exe', port: 8097, path: '../ai-summary-service' },
        { name: 'ai-enhanced-fixed', binary: 'ai-enhanced-fixed.exe', port: 8098, path: '../ai-summary-service' },
        { name: 'ai-enhanced-postgresql', binary: 'ai-enhanced-postgresql.exe', port: 8099, path: '../ai-summary-service' },
        { name: 'live-agent-enhanced', binary: 'live-agent-enhanced.exe', port: 8200, path: '../ai-summary-service' },
        { name: 'enhanced-semantic-architecture', binary: 'enhanced-semantic-architecture.exe', port: 8201, path: '../' },
        { name: 'enhanced-legal-ai', binary: 'enhanced-legal-ai.exe', port: 8202, path: '../go-microservice' },
        { name: 'enhanced-legal-ai-clean', binary: 'enhanced-legal-ai-clean.exe', port: 8203, path: '../go-microservice' },
        { name: 'enhanced-legal-ai-fixed', binary: 'enhanced-legal-ai-fixed.exe', port: 8204, path: '../go-microservice' },
        { name: 'enhanced-legal-ai-redis', binary: 'enhanced-legal-ai-redis.exe', port: 8205, path: '../go-microservice' }
      ],

      // Tier 3: File & Upload Services (4 services)
      tier3: [
        { name: 'upload-service', binary: 'upload-service.exe', port: 8093, path: '../go-microservice/bin' },
        { name: 'gin-upload', binary: 'gin-upload.exe', port: 8207, path: '../go-microservice/bin' },
        { name: 'simple-upload', binary: 'simple-upload.exe', port: 8208, path: '../go-microservice/bin' },
        { name: 'document-processor-integrated', binary: 'document-processor-integrated.exe', port: 8081, path: '../ai-summary-service' }
      ],

      // Tier 4: Vector Processing Services (2 services)
      tier4: [
        { name: 'simple-vector-service', binary: 'simple-vector-service.exe', port: 8095, path: '../go-microservice/bin' },
        { name: 'vector-consumer-v2', binary: 'vector-consumer-v2.exe', port: 8096, path: '../go-microservice/bin' }
      ],

      // Tier 5: Network Protocol Services (3 services)
      tier5: [
        { name: 'grpc-server', binary: 'grpc-server.exe', port: 50051, path: '../go-microservice/bin' },
        { name: 'rag-kratos', binary: 'rag-kratos.exe', port: 50052, path: '../go-microservice/bin' },
        { name: 'rag-quic-proxy', binary: 'rag-quic-proxy.exe', port: 8216, path: '../go-microservice/bin' }
      ],

      // Tier 6: Orchestration & State Services (4 services)
      tier6: [
        { name: 'xstate-manager', binary: 'xstate-manager.exe', port: 8212, path: '../go-microservice/bin' },
        { name: 'cluster-http', binary: 'cluster-http.exe', port: 8213, path: '../go-microservice/bin' },
        { name: 'modular-cluster-service', binary: 'modular-cluster-service.exe', port: 8214, path: '../indexing-system' },
        { name: 'modular-cluster-service-production', binary: 'modular-cluster-service-production.exe', port: 8215, path: '../indexing-system' }
      ],

      // Tier 7: Infrastructure & Monitoring Services (13 services)
      tier7: [
        { name: 'simd-health', binary: 'simd-health.exe', port: 8217, path: '../go-microservice/bin' },
        { name: 'simd-parser', binary: 'simd-parser.exe', port: 8218, path: '../go-microservice/bin' },
        { name: 'context7-error-pipeline', binary: 'context7-error-pipeline.exe', port: 8219, path: '../go-microservice/bin' },
        { name: 'gpu-indexer-service', binary: 'gpu-indexer-service.exe', port: 8220, path: '../go-microservice/bin' },
        { name: 'async-indexer', binary: 'async-indexer.exe', port: 8221, path: '../indexing-system' },
        { name: 'recommendation-service', binary: 'recommendation-service.exe', port: 8223, path: '../go-microservice/bin' },
        { name: 'load-balancer', binary: 'load-balancer.exe', port: 8224, path: '../go-microservice/bin' },
        { name: 'simple-server', binary: 'simple-server.exe', port: 8225, path: '../go-microservice/bin' },
        { name: 'test-server', binary: 'test-server.exe', port: 8226, path: '../go-microservice/bin' },
        { name: 'test-build', binary: 'test-build.exe', port: 8227, path: '../go-microservice/bin' },
        { name: 'summarizer-service', binary: 'summarizer-service.exe', port: 8209, path: '../ai-summary-service' },
        { name: 'summarizer-http', binary: 'summarizer-http.exe', port: 8210, path: '../ai-summary-service' },
        { name: 'ai-summary', binary: 'ai-summary.exe', port: 8211, path: '../ai-summary-service' }
      ],

      // Tier 8: Frontend (SvelteKit)
      frontend: [
        { name: 'sveltekit', command: 'npm run dev', cwd: process.cwd() }
      ]
    };
  }

  // Enhanced service health check
  async checkServiceHealth(name, command, successPattern) {
    try {
      const { stdout, stderr } = await execAsync(command, { timeout: 5000 });
      const output = stdout + stderr;
      const isHealthy = successPattern ? output.includes(successPattern) : true;
      
      this.healthChecks.set(name, {
        status: isHealthy ? 'healthy' : 'unhealthy',
        lastCheck: new Date(),
        output: output.trim()
      });
      
      return isHealthy;
    } catch (error) {
      this.healthChecks.set(name, {
        status: 'error',
        lastCheck: new Date(),
        error: error.message
      });
      return false;
    }
  }

  // Start infrastructure services
  async startInfrastructureServices() {
    log.section('Infrastructure Services (PostgreSQL, Redis, RabbitMQ)');
    const services = this.getServiceConfiguration().tier1;
    
    for (const service of services) {
      log.info(`Checking ${service.name}...`);
      const isHealthy = await this.checkServiceHealth(service.name, service.command, service.successPattern);
      
      if (isHealthy) {
        log.success(`${service.name} is already running`);
      } else {
        log.warning(`${service.name} not available - some features may be limited`);
      }
    }
  }

  // Start microservice binary
  async startMicroservice(service) {
    const servicePath = path.resolve(service.path, service.binary);
    
    if (!existsSync(servicePath)) {
      log.warning(`Service binary not found: ${servicePath}`);
      return false;
    }
    
    log.info(`Starting ${service.name} on port ${service.port}...`);
    
    const serviceProcess = spawn(servicePath, [], {
      cwd: path.dirname(servicePath),
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe']
    });
    
    serviceProcess.unref();
    
    // Monitor the process
    if (serviceProcess.stdout) {
      serviceProcess.stdout.on('data', (data) => {
        if (this.config.enableLogging) {
          log.info(`${service.name}: ${data.toString().trim()}`);
        }
      });
    }
    
    if (serviceProcess.stderr) {
      serviceProcess.stderr.on('data', (data) => {
        if (this.config.enableLogging) {
          log.warning(`${service.name} Error: ${data.toString().trim()}`);
        }
      });
    }
    
    this.services.set(service.name, {
      process: serviceProcess,
      port: service.port,
      startTime: Date.now(),
      status: 'starting'
    });
    
    // Wait for service to be ready
    await this.sleep(2000);
    
    // Health check
    const healthUrl = `http://localhost:${service.port}/health`;
    try {
      const response = await fetch(healthUrl, { signal: AbortSignal.timeout(3000) });
      if (response.ok) {
        log.success(`${service.name} started successfully on port ${service.port}`);
        this.services.get(service.name).status = 'running';
        return true;
      }
    } catch (error) {
      log.info(`${service.name} started (health check pending)`);
      this.services.get(service.name).status = 'pending';
    }
    
    return true;
  }

  // Start service tier
  async startServiceTier(tierName, services) {
    if (!services || services.length === 0) {
      return;
    }
    
    log.section(`Starting ${tierName} (${services.length} services)`);
    
    const promises = services.map(async (service) => {
      return await this.startMicroservice(service);
    });
    
    await Promise.all(promises);
    
    // Stagger next tier startup
    await this.sleep(3000);
  }

  // Start SvelteKit frontend
  async startSvelteKit() {
    log.section('SvelteKit Development Server');
    
    log.info('Starting SvelteKit dev server...');
    
    const svelteProcess = spawn('npm', ['run', 'dev'], {
      stdio: 'inherit',
      shell: true,
      cwd: process.cwd()
    });
    
    this.services.set('sveltekit', {
      process: svelteProcess,
      port: 5173,
      startTime: Date.now(),
      status: 'running'
    });
    
    svelteProcess.on('exit', (code) => {
      log.info(`SvelteKit dev server exited with code ${code}`);
      this.shutdown();
    });
    
    return true;
  }

  // Display comprehensive status dashboard
  async displayComprehensiveStatusDashboard() {
    log.section('38 Microservices Development Environment Status');
    
    console.log(`${colors.bright}Complete Service Health Dashboard${colors.reset}`);
    console.log('━'.repeat(80));
    
    const config = this.getServiceConfiguration();
    let totalServices = 0;
    let runningServices = 0;
    
    // Infrastructure services
    console.log(`${colors.cyan}Infrastructure Services (Tier 1):${colors.reset}`);
    for (const service of config.tier1) {
      const health = this.healthChecks.get(service.name);
      if (health) {
        const statusIcon = health.status === 'healthy' ? '✓' : '✗';
        const statusColor = health.status === 'healthy' ? colors.green : colors.red;
        console.log(`  ${statusColor}${statusIcon}${colors.reset} ${service.name.padEnd(20)} ${health.status}`);
        if (health.status === 'healthy') runningServices++;
      }
      totalServices++;
    }
    
    // Microservices tiers
    const tiers = ['tier2', 'tier3', 'tier4', 'tier5', 'tier6', 'tier7'];
    const tierNames = [
      'AI/RAG Processing Services (Tier 2)',
      'File & Upload Services (Tier 3)',
      'Vector Processing Services (Tier 4)',
      'Network Protocol Services (Tier 5)',
      'Orchestration & State Services (Tier 6)',
      'Infrastructure & Monitoring Services (Tier 7)'
    ];
    
    for (let i = 0; i < tiers.length; i++) {
      console.log(`\n${colors.cyan}${tierNames[i]}:${colors.reset}`);
      for (const service of config[tiers[i]]) {
        const serviceData = this.services.get(service.name);
        let status = 'not started';
        let statusColor = colors.red;
        let statusIcon = '✗';
        
        if (serviceData) {
          status = serviceData.status;
          if (status === 'running') {
            statusColor = colors.green;
            statusIcon = '✓';
            runningServices++;
          } else if (status === 'pending') {
            statusColor = colors.yellow;
            statusIcon = '⚠';
            runningServices++;
          }
        }
        
        console.log(`  ${statusColor}${statusIcon}${colors.reset} ${service.name.padEnd(35)} ${status.padEnd(15)} Port: ${service.port}`);
        totalServices++;
      }
    }
    
    // Frontend
    console.log(`\n${colors.cyan}Frontend Services:${colors.reset}`);
    const svelteData = this.services.get('sveltekit');
    if (svelteData) {
      console.log(`  ${colors.green}✓${colors.reset} SvelteKit                         running         Port: 5173`);
      runningServices++;
    }
    totalServices++;
    
    console.log('━'.repeat(80));
    
    // Service endpoints
    console.log(`${colors.blue}Key Service Endpoints:${colors.reset}`);
    console.log('• SvelteKit Frontend:    http://localhost:5173');
    console.log('• Enhanced RAG API:      http://localhost:8094');
    console.log('• Upload Service:        http://localhost:8093');
    console.log('• Vector Service v2.0:   http://localhost:8095');
    console.log('• Legal AI Services:     http://localhost:8202-8205');
    console.log('• Cluster Management:    http://localhost:8213-8215');
    console.log('• GPU Services:          http://localhost:8220');
    console.log('• gRPC Services:         localhost:50051-50052');
    console.log('• QUIC Proxy:            localhost:8216');
    console.log();
    
    const healthPercentage = Math.round((runningServices / totalServices) * 100);
    const overallStatus = healthPercentage >= 80 ? 'EXCELLENT' :
                         healthPercentage >= 60 ? 'GOOD' :
                         healthPercentage >= 40 ? 'FAIR' : 'DEVELOPING';
    
    const overallColor = healthPercentage >= 80 ? colors.green :
                        healthPercentage >= 60 ? colors.blue :
                        healthPercentage >= 40 ? colors.yellow : colors.magenta;
    
    const uptime = Math.round((Date.now() - this.startTime) / 1000);
    console.log(`${colors.green}Environment Status: ${overallColor}${overallStatus}${colors.reset} (${runningServices}/${totalServices} services - ${healthPercentage}%)`);
    console.log(`${colors.blue}Uptime: ${uptime}s | Total Services: ${totalServices} | Architecture: 38 Microservices${colors.reset}`);
    
    // Performance optimization tips
    if (healthPercentage < 80) {
      console.log(`\n${colors.yellow}Performance Optimization:${colors.reset}`);
      console.log('• Some services may take longer to start - this is normal');
      console.log('• Infrastructure services (PostgreSQL, Redis, RabbitMQ) provide enhanced functionality');
      console.log('• Core services (Enhanced RAG, Upload, Vector) provide essential functionality');
      console.log('• Run "npm run services:enterprise:health" for detailed health monitoring');
    }
  }

  // Utility function
  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Graceful shutdown
  shutdown() {
    log.info('Shutting down all microservices...');
    
    for (const [name, serviceData] of this.services) {
      try {
        if (serviceData.process) {
          serviceData.process.kill();
          log.info(`Stopped ${name}`);
        }
      } catch (error) {
        log.warning(`Failed to stop ${name}: ${error.message}`);
      }
    }
    
    process.exit(0);
  }

  // Write service metrics
  writeServiceMetrics() {
    const metrics = {
      timestamp: new Date().toISOString(),
      totalServices: this.services.size,
      runningServices: Array.from(this.services.values()).filter(s => s.status === 'running').length,
      uptime: Date.now() - this.startTime,
      services: Array.from(this.services.entries()).map(([name, data]) => ({
        name,
        port: data.port,
        status: data.status,
        startTime: data.startTime
      }))
    };
    
    try {
      writeFileSync('.vscode/microservices-metrics.json', JSON.stringify(metrics, null, 2));
    } catch (error) {
      // Ignore file write errors
    }
  }

  // Main orchestration method
  async start() {
    console.log(`${colors.cyan}${colors.bright}`);
    console.log('╔══════════════════════════════════════════════════════════════════════════╗');
    console.log('║                  Complete 38 Microservices Orchestrator                 ║');
    console.log('║                     Enhanced Legal AI Platform v2.0                     ║');
    console.log('║    PostgreSQL + Redis + RabbitMQ + 38 Go Services + SvelteKit          ║');
    console.log('╚══════════════════════════════════════════════════════════════════════════╝');
    console.log(colors.reset);
    
    // Setup graceful shutdown
    process.on('SIGINT', () => this.shutdown());
    process.on('SIGTERM', () => this.shutdown());
    
    // Start services in tiers
    const config = this.getServiceConfiguration();
    
    // Tier 1: Infrastructure
    await this.startInfrastructureServices();
    
    // Tier 2-7: Microservices
    await this.startServiceTier('AI/RAG Processing Services', config.tier2);
    await this.startServiceTier('File & Upload Services', config.tier3);
    await this.startServiceTier('Vector Processing Services', config.tier4);
    await this.startServiceTier('Network Protocol Services', config.tier5);
    await this.startServiceTier('Orchestration & State Services', config.tier6);
    await this.startServiceTier('Infrastructure & Monitoring Services', config.tier7);
    
    // Display comprehensive status
    await this.displayComprehensiveStatusDashboard();
    
    // Write metrics
    if (this.config.enableMetrics) {
      this.writeServiceMetrics();
      // Update metrics every 30 seconds
      setInterval(() => this.writeServiceMetrics(), 30000);
    }
    
    // Start SvelteKit (blocks until shutdown)
    await this.startSvelteKit();
  }
}

// Start the complete microservices orchestrator
const orchestrator = new CompleteMicroservicesOrchestrator();
orchestrator.start().catch(error => {
  log.error(`Failed to start microservices environment: ${error.message}`);
  process.exit(1);
});