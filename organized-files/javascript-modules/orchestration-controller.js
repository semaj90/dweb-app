#!/usr/bin/env node

/**
 * Legal AI Orchestration Controller
 * 
 * Master controller that wires up all orchestration components:
 * - Service discovery and registration
 * - Inter-service communication
 * - Health monitoring and alerting
 * - Message routing coordination
 * - Graceful startup and shutdown
 */

const fs = require('fs');
const path = require('path');
const { spawn, exec } = require('child_process');
const http = require('http');
const WebSocket = require('ws');

class LegalAIOrchestrationController {
  constructor() {
    this.config = this.loadConfiguration();
    this.services = new Map();
    this.healthChecks = new Map();
    this.messageRoutes = new Map();
    this.isShuttingDown = false;
    
    // Initialize components
    this.serviceRegistry = new ServiceRegistry();
    this.healthMonitor = new HealthMonitor(this);
    this.messageRouter = new MessageRouter(this);
    this.dependencyManager = new DependencyManager(this);
  }

  loadConfiguration() {
    try {
      const configPath = path.join(__dirname, 'orchestration-config.json');
      const configData = fs.readFileSync(configPath, 'utf8');
      return JSON.parse(configData);
    } catch (error) {
      console.error('[ORCHESTRATOR] Failed to load configuration:', error.message);
      process.exit(1);
    }
  }

  async initialize() {
    console.log('[ORCHESTRATOR] Starting Legal AI Orchestration Controller...');
    console.log(`[ORCHESTRATOR] Environment: ${this.config.orchestration.configuration['environment-variables'].LEGAL_AI_ENV}`);
    
    // Set environment variables
    this.setEnvironmentVariables();
    
    // Initialize orchestration components
    await this.initializeServiceRegistry();
    await this.initializeHealthMonitoring();
    await this.initializeMessageRouting();
    
    // Start management API
    await this.startManagementAPI();
    
    // Set up graceful shutdown
    this.setupGracefulShutdown();
    
    console.log('[ORCHESTRATOR] Orchestration controller initialized successfully');
  }

  setEnvironmentVariables() {
    const envVars = this.config.orchestration.configuration['environment-variables'];
    Object.entries(envVars).forEach(([key, value]) => {
      process.env[key] = value;
    });
    
    console.log('[ORCHESTRATOR] Environment variables configured');
  }

  async initializeServiceRegistry() {
    console.log('[ORCHESTRATOR] Initializing service registry...');
    
    Object.entries(this.config.orchestration.services).forEach(([serviceId, serviceConfig]) => {
      this.serviceRegistry.register(serviceId, serviceConfig);
    });
    
    console.log(`[ORCHESTRATOR] Registered ${this.serviceRegistry.size()} services`);
  }

  async initializeHealthMonitoring() {
    console.log('[ORCHESTRATOR] Initializing health monitoring...');
    
    this.healthMonitor.start();
    console.log('[ORCHESTRATOR] Health monitoring active');
  }

  async initializeMessageRouting() {
    console.log('[ORCHESTRATOR] Initializing message routing...');
    
    const messageConfig = this.config.orchestration.communication['message-routing'];
    Object.entries(messageConfig).forEach(([routeName, routeConfig]) => {
      this.messageRouter.addRoute(routeName, routeConfig);
    });
    
    console.log(`[ORCHESTRATOR] Configured ${Object.keys(messageConfig).length} message routes`);
  }

  async startServices() {
    console.log('[ORCHESTRATOR] Starting services in dependency order...');
    
    const startupOrder = this.config.orchestration.deployment['startup-order'];
    
    for (const serviceId of startupOrder) {
      if (this.isShuttingDown) break;
      
      try {
        await this.startService(serviceId);
        // Wait for service to be healthy before starting next
        await this.waitForServiceHealth(serviceId);
      } catch (error) {
        console.error(`[ORCHESTRATOR] Failed to start ${serviceId}:`, error.message);
        
        if (this.config.orchestration.deployment['restart-policy'] === 'on-failure') {
          console.log(`[ORCHESTRATOR] Retrying ${serviceId} in 5 seconds...`);
          await this.delay(5000);
          await this.startService(serviceId);
        }
      }
    }
    
    console.log('[ORCHESTRATOR] All services started successfully');
  }

  async startService(serviceId) {
    const serviceConfig = this.config.orchestration.services[serviceId];
    if (!serviceConfig) {
      throw new Error(`Service ${serviceId} not found in configuration`);
    }
    
    console.log(`[ORCHESTRATOR] Starting ${serviceConfig.name}...`);
    
    const process = spawn('cmd', ['/c', serviceConfig.startupCommand], {
      stdio: ['ignore', 'pipe', 'pipe'],
      cwd: __dirname,
      env: { ...process.env }
    });
    
    // Store process reference
    this.services.set(serviceId, {
      process,
      config: serviceConfig,
      startTime: Date.now(),
      status: 'starting'
    });
    
    // Handle process output
    process.stdout.on('data', (data) => {
      console.log(`[${serviceId.toUpperCase()}] ${data.toString().trim()}`);
    });
    
    process.stderr.on('data', (data) => {
      console.error(`[${serviceId.toUpperCase()}] ERROR: ${data.toString().trim()}`);
    });
    
    process.on('exit', (code) => {
      if (code !== 0 && !this.isShuttingDown) {
        console.error(`[ORCHESTRATOR] Service ${serviceId} exited with code ${code}`);
        this.handleServiceFailure(serviceId, code);
      }
    });
    
    // Register with service discovery
    this.serviceRegistry.updateStatus(serviceId, 'starting');
    
    console.log(`[ORCHESTRATOR] Started ${serviceConfig.name} (PID: ${process.pid})`);
  }

  async waitForServiceHealth(serviceId, timeout = 60000) {
    const serviceConfig = this.config.orchestration.services[serviceId];
    const startTime = Date.now();
    
    console.log(`[ORCHESTRATOR] Waiting for ${serviceConfig.name} to be healthy...`);
    
    while (Date.now() - startTime < timeout) {
      if (this.isShuttingDown) return;
      
      try {
        const isHealthy = await this.healthMonitor.checkService(serviceId);
        if (isHealthy) {
          this.serviceRegistry.updateStatus(serviceId, 'healthy');
          console.log(`[ORCHESTRATOR] ${serviceConfig.name} is healthy`);
          return;
        }
      } catch (error) {
        // Service not ready yet, continue waiting
      }
      
      await this.delay(2000);
    }
    
    throw new Error(`Service ${serviceId} failed to become healthy within ${timeout}ms`);
  }

  async handleServiceFailure(serviceId, exitCode) {
    console.log(`[ORCHESTRATOR] Handling failure for service ${serviceId}`);
    
    this.serviceRegistry.updateStatus(serviceId, 'failed');
    
    // Check restart policy
    const restartPolicy = this.config.orchestration.deployment['restart-policy'];
    if (restartPolicy === 'on-failure') {
      console.log(`[ORCHESTRATOR] Restarting ${serviceId} due to failure...`);
      
      await this.delay(5000); // Wait before restart
      await this.startService(serviceId);
    }
  }

  async startManagementAPI() {
    const managementServer = http.createServer((req, res) => {
      this.handleManagementRequest(req, res);
    });
    
    // WebSocket server for real-time updates
    const wss = new WebSocket.Server({ server: managementServer });
    wss.on('connection', (ws) => {
      console.log('[ORCHESTRATOR] Management WebSocket connection established');
      
      // Send initial status
      ws.send(JSON.stringify({
        type: 'status',
        data: this.getSystemStatus()
      }));
      
      // Set up periodic updates
      const interval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: 'status',
            data: this.getSystemStatus()
          }));
        }
      }, 5000);
      
      ws.on('close', () => {
        clearInterval(interval);
      });
    });
    
    const port = 8000;
    managementServer.listen(port, () => {
      console.log(`[ORCHESTRATOR] Management API listening on port ${port}`);
    });
  }

  handleManagementRequest(req, res) {
    const url = new URL(req.url, `http://${req.headers.host}`);
    
    // Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }
    
    switch (url.pathname) {
      case '/status':
        this.handleStatusRequest(req, res);
        break;
      case '/services':
        this.handleServicesRequest(req, res);
        break;
      case '/health':
        this.handleHealthRequest(req, res);
        break;
      case '/metrics':
        this.handleMetricsRequest(req, res);
        break;
      case '/restart':
        this.handleRestartRequest(req, res);
        break;
      case '/shutdown':
        this.handleShutdownRequest(req, res);
        break;
      default:
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not Found' }));
    }
  }

  handleStatusRequest(req, res) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(this.getSystemStatus(), null, 2));
  }

  handleServicesRequest(req, res) {
    const services = Array.from(this.services.entries()).map(([id, service]) => ({
      id,
      name: service.config.name,
      status: service.status,
      pid: service.process.pid,
      uptime: Date.now() - service.startTime
    }));
    
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(services, null, 2));
  }

  handleHealthRequest(req, res) {
    const healthStatus = this.healthMonitor.getOverallHealth();
    const statusCode = healthStatus.healthy ? 200 : 503;
    
    res.writeHead(statusCode, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(healthStatus, null, 2));
  }

  getSystemStatus() {
    return {
      orchestrator: {
        status: 'running',
        uptime: process.uptime(),
        memory: process.memoryUsage()
      },
      services: this.serviceRegistry.getAllServices(),
      health: this.healthMonitor.getOverallHealth(),
      messageRoutes: this.messageRouter.getStats(),
      timestamp: new Date().toISOString()
    };
  }

  setupGracefulShutdown() {
    const shutdown = async (signal) => {
      console.log(`[ORCHESTRATOR] Received ${signal}, initiating graceful shutdown...`);
      this.isShuttingDown = true;
      
      await this.stopAllServices();
      
      console.log('[ORCHESTRATOR] Graceful shutdown completed');
      process.exit(0);
    };
    
    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('SIGTERM', () => shutdown('SIGTERM'));
  }

  async stopAllServices() {
    console.log('[ORCHESTRATOR] Stopping all services...');
    
    // Stop services in reverse order
    const shutdownOrder = [...this.config.orchestration.deployment['startup-order']].reverse();
    
    for (const serviceId of shutdownOrder) {
      await this.stopService(serviceId);
    }
  }

  async stopService(serviceId) {
    const service = this.services.get(serviceId);
    if (!service) return;
    
    console.log(`[ORCHESTRATOR] Stopping ${service.config.name}...`);
    
    // Send graceful shutdown signal
    service.process.kill('SIGTERM');
    
    // Wait for graceful shutdown or force kill
    const timeout = this.config.orchestration.deployment['graceful-shutdown-timeout'];
    const timeoutMs = parseInt(timeout) * 1000;
    
    setTimeout(() => {
      if (!service.process.killed) {
        console.log(`[ORCHESTRATOR] Force killing ${service.config.name}`);
        service.process.kill('SIGKILL');
      }
    }, timeoutMs);
    
    this.services.delete(serviceId);
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Service Registry Implementation
class ServiceRegistry {
  constructor() {
    this.services = new Map();
  }

  register(serviceId, config) {
    this.services.set(serviceId, {
      id: serviceId,
      config,
      status: 'registered',
      lastSeen: Date.now()
    });
  }

  updateStatus(serviceId, status) {
    const service = this.services.get(serviceId);
    if (service) {
      service.status = status;
      service.lastSeen = Date.now();
    }
  }

  getAllServices() {
    return Array.from(this.services.values());
  }

  size() {
    return this.services.size;
  }
}

// Health Monitor Implementation
class HealthMonitor {
  constructor(orchestrator) {
    this.orchestrator = orchestrator;
    this.healthResults = new Map();
    this.isRunning = false;
  }

  start() {
    if (this.isRunning) return;
    
    this.isRunning = true;
    this.scheduleHealthChecks();
  }

  scheduleHealthChecks() {
    const interval = this.orchestrator.config.orchestration.monitoring['health-checks'].interval;
    const intervalMs = parseInt(interval) * 1000;
    
    setInterval(async () => {
      if (!this.isRunning) return;
      
      await this.performHealthChecks();
    }, intervalMs);
  }

  async performHealthChecks() {
    const services = this.orchestrator.serviceRegistry.getAllServices();
    
    for (const service of services) {
      try {
        const isHealthy = await this.checkService(service.id);
        this.healthResults.set(service.id, {
          healthy: isHealthy,
          lastCheck: Date.now(),
          responseTime: Date.now() // Simplified
        });
      } catch (error) {
        this.healthResults.set(service.id, {
          healthy: false,
          lastCheck: Date.now(),
          error: error.message
        });
      }
    }
  }

  async checkService(serviceId) {
    const service = this.orchestrator.config.orchestration.services[serviceId];
    if (!service) return false;
    
    // Simple HTTP health check
    return new Promise((resolve) => {
      const healthUrl = `http://${service.host}:${service.port}${service.healthEndpoint}`;
      
      http.get(healthUrl, (res) => {
        resolve(res.statusCode === 200);
      }).on('error', () => {
        resolve(false);
      });
    });
  }

  getOverallHealth() {
    const results = Array.from(this.healthResults.values());
    const healthy = results.filter(r => r.healthy).length;
    const total = results.length;
    
    return {
      healthy: healthy === total,
      services: {
        total,
        healthy,
        unhealthy: total - healthy
      },
      results: Object.fromEntries(this.healthResults)
    };
  }
}

// Message Router Implementation
class MessageRouter {
  constructor(orchestrator) {
    this.orchestrator = orchestrator;
    this.routes = new Map();
  }

  addRoute(routeName, routeConfig) {
    this.routes.set(routeName, routeConfig);
    console.log(`[MESSAGE-ROUTER] Added route: ${routeName}`);
  }

  getStats() {
    return {
      totalRoutes: this.routes.size,
      routes: Array.from(this.routes.keys())
    };
  }
}

// Dependency Manager Implementation
class DependencyManager {
  constructor(orchestrator) {
    this.orchestrator = orchestrator;
  }

  getDependencyOrder() {
    // Implement topological sort for service dependencies
    return this.orchestrator.config.orchestration.deployment['startup-order'];
  }
}

// Main execution
if (require.main === module) {
  const orchestrator = new LegalAIOrchestrationController();
  
  orchestrator.initialize()
    .then(() => orchestrator.startServices())
    .catch((error) => {
      console.error('[ORCHESTRATOR] Fatal error:', error);
      process.exit(1);
    });
}

module.exports = LegalAIOrchestrationController;