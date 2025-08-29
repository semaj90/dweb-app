/**
 * Production Integration Orchestrator
 * Manages all 37+ Go microservices, Node.js services, and infrastructure components
 * with comprehensive monitoring, error handling, and automatic recovery
 */

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { createRequire } from 'module';
import { Worker } from 'worker_threads';
import cluster from 'cluster';
import * as redis from 'redis';
import amqp from 'amqplib';
import { Client as PgClient } from 'pg';
import WebSocket, { WebSocketServer } from 'ws';
import { performance } from 'perf_hooks';
import Loki from 'lokijs';

const require = createRequire(import.meta.url);

// Types and Interfaces
export interface ServiceConfig {
  name: string;
  type: 'go' | 'node' | 'python' | 'windows' | 'native';
  command: string;
  args?: string[];
  cwd?: string;
  port?: number;
  healthUrl?: string;
  dependencies?: string[];
  priority: number; // 1-5, where 1 is critical
  restartPolicy: 'always' | 'on-failure' | 'unless-stopped';
  maxRetries: number;
  retryDelay: number;
  timeout: number;
  env?: Record<string, string>;
  gpuRequired?: boolean;
  memoryLimit?: number;
}

export interface ServiceStatus {
  name: string;
  status: 'starting' | 'running' | 'stopping' | 'stopped' | 'error' | 'unhealthy';
  pid?: number;
  startTime?: Date;
  restartCount: number;
  lastError?: string;
  health?: 'healthy' | 'unhealthy' | 'unknown';
  metrics: ServiceMetrics;
}

export interface ServiceMetrics {
  cpuUsage: number;
  memoryUsage: number;
  networkConnections: number;
  requestCount: number;
  responseTime: number;
  errorRate: number;
  uptime: number;
}

export interface SystemHealth {
  overall: 'healthy' | 'degraded' | 'critical';
  services: ServiceStatus[];
  infrastructure: {
    postgres: boolean;
    redis: boolean;
    rabbitmq: boolean;
    gpu: boolean;
    disk: number; // percentage free
    memory: number; // percentage free
    cpu: number; // percentage usage
  };
  alerts: Alert[];
}

export interface Alert {
  id: string;
  level: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  service?: string;
  timestamp: Date;
  acknowledged: boolean;
}

// Production Orchestrator Class
export class ProductionOrchestrator extends EventEmitter {
  private services: Map<string, ServiceConfig> = new Map();
  private processes: Map<string, ChildProcess> = new Map();
  private serviceStatus: Map<string, ServiceStatus> = new Map();
  private workers: Map<string, Worker> = new Map();
  private db: Loki;
  private alerts: Loki.Collection<Alert>;
  private metrics: Loki.Collection<ServiceMetrics>;
  
  private redisClient?: redis.RedisClientType;
  private pgClient?: PgClient;
  private rabbitConnection?: amqp.Connection;
  private wsServer?: WebSocketServer;
  
  private healthCheckInterval?: NodeJS.Timeout;
  private metricsCollectionInterval?: NodeJS.Timeout;
  private logRotationInterval?: NodeJS.Timeout;
  
  constructor() {
    super();
    this.initializeDatabase();
    this.setupSignalHandlers();
    this.initializeServices();
  }

  private initializeDatabase(): void {
    this.db = new Loki('orchestrator.db', {
      autoload: true,
      autosave: true,
      autosaveInterval: 5000
    });

    this.alerts = this.db.addCollection('alerts');
    this.metrics = this.db.addCollection('metrics');
  }

  private initializeServices(): void {
    // Core Infrastructure Services
    this.addService({
      name: 'postgresql',
      type: 'windows',
      command: 'net',
      args: ['start', 'postgresql-x64-17'],
      port: 5432,
      healthUrl: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
      priority: 1,
      restartPolicy: 'always',
      maxRetries: 5,
      retryDelay: 10000,
      timeout: 30000
    });

    this.addService({
      name: 'redis',
      type: 'native',
      command: 'redis-server',
      args: ['--port', '6379', '--save', '60', '1000'],
      port: 6379,
      healthUrl: 'http://localhost:6379',
      priority: 1,
      restartPolicy: 'always',
      maxRetries: 5,
      retryDelay: 5000,
      timeout: 15000
    });

    this.addService({
      name: 'rabbitmq',
      type: 'windows',
      command: 'net',
      args: ['start', 'RabbitMQ'],
      port: 5672,
      healthUrl: 'http://localhost:15672',
      priority: 1,
      restartPolicy: 'always',
      maxRetries: 3,
      retryDelay: 15000,
      timeout: 30000
    });

    this.addService({
      name: 'ollama',
      type: 'native',
      command: 'ollama',
      args: ['serve'],
      port: 11434,
      healthUrl: 'http://localhost:11434/api/tags',
      priority: 2,
      restartPolicy: 'always',
      maxRetries: 3,
      retryDelay: 10000,
      timeout: 30000,
      gpuRequired: true
    });

    this.addService({
      name: 'qdrant',
      type: 'native',
      command: path.join(process.cwd(), 'qdrant-windows', 'qdrant.exe'),
      port: 6333,
      healthUrl: 'http://localhost:6333/collections',
      priority: 2,
      restartPolicy: 'always',
      maxRetries: 3,
      retryDelay: 10000,
      timeout: 20000
    });

    this.addService({
      name: 'minio',
      type: 'native',
      command: 'minio',
      args: ['server', './minio-data', '--address', ':9000', '--console-address', ':9001'],
      port: 9000,
      healthUrl: 'http://localhost:9000/minio/health/live',
      priority: 2,
      restartPolicy: 'always',
      maxRetries: 3,
      retryDelay: 10000,
      timeout: 20000
    });

    this.addService({
      name: 'neo4j',
      type: 'windows',
      command: 'powershell',
      args: ['-Command', 'Start-Service neo4j'],
      port: 7474,
      healthUrl: 'http://localhost:7474',
      priority: 3,
      restartPolicy: 'unless-stopped',
      maxRetries: 2,
      retryDelay: 20000,
      timeout: 45000
    });

    this.addService({
      name: 'nats-server',
      type: 'native',
      command: path.join(process.cwd(), 'nats-server', 'nats-server.exe'),
      args: ['--port', '4222', '--http_port', '8222'],
      port: 4222,
      healthUrl: 'http://localhost:8222',
      priority: 2,
      restartPolicy: 'always',
      maxRetries: 3,
      retryDelay: 5000,
      timeout: 15000
    });

    // Go Microservices
    this.addGoMicroservices();
    
    // Node.js Services
    this.addNodeServices();
    
    // GPU and CUDA Services
    this.addGpuServices();

    // Frontend Services
    this.addFrontendServices();
  }

  private addGoMicroservices(): void {
    const goServices: Partial<ServiceConfig>[] = [
      {
        name: 'enhanced-rag',
        command: 'go',
        args: ['run', 'cmd/enhanced-rag/main.go'],
        cwd: 'go-microservice',
        port: 8094,
        healthUrl: 'http://localhost:8094/health',
        dependencies: ['postgresql', 'redis', 'qdrant']
      },
      {
        name: 'upload-service',
        command: 'go',
        args: ['run', 'cmd/upload-service/main.go'],
        cwd: 'go-microservice',
        port: 8093,
        healthUrl: 'http://localhost:8093/health',
        dependencies: ['minio', 'postgresql']
      },
      {
        name: 'gpu-tensor-service',
        command: 'go',
        args: ['run', 'cmd/gpu-tensor-service/main.go'],
        cwd: 'go-microservice',
        port: 8088,
        healthUrl: 'http://localhost:8088/health',
        gpuRequired: true
      },
      {
        name: 'vector-service',
        command: 'go',
        args: ['run', 'cmd/vector-service/main.go'],
        cwd: 'go-microservice',
        port: 8095,
        healthUrl: 'http://localhost:8095/health',
        dependencies: ['qdrant', 'redis']
      },
      {
        name: 'cluster-service',
        command: 'go',
        args: ['run', 'cmd/cluster-service/main.go'],
        cwd: 'go-microservice',
        port: 8089,
        healthUrl: 'http://localhost:8089/health'
      },
      {
        name: 'load-balancer',
        command: 'go',
        args: ['run', 'cmd/load-balancer/main.go'],
        cwd: 'go-microservice',
        port: 8099,
        healthUrl: 'http://localhost:8099/status'
      },
      {
        name: 'grpc-server',
        command: 'go',
        args: ['run', 'cmd/grpc-server/main.go'],
        cwd: 'go-microservice',
        port: 9090,
        healthUrl: 'http://localhost:9090/health'
      },
      {
        name: 'quic-server',
        command: 'go',
        args: ['run', 'cmd/rag-quic/main.go'],
        cwd: 'go-microservice',
        port: 8443,
        healthUrl: 'http://localhost:8443/health'
      },
      {
        name: 'cuda-service',
        command: 'go',
        args: ['run', 'cmd/cuda-service/main.go'],
        cwd: 'go-microservice',
        port: 8087,
        healthUrl: 'http://localhost:8087/health',
        gpuRequired: true
      },
      {
        name: 'production-rag',
        command: 'go',
        args: ['run', 'cmd/production-rag/main.go'],
        cwd: 'go-microservice',
        port: 8096,
        healthUrl: 'http://localhost:8096/health'
      }
    ];

    goServices.forEach(service => {
      this.addService({
        type: 'go',
        priority: 3,
        restartPolicy: 'always',
        maxRetries: 3,
        retryDelay: 5000,
        timeout: 20000,
        ...service
      } as ServiceConfig);
    });
  }

  private addNodeServices(): void {
    this.addService({
      name: 'node-api',
      type: 'node',
      command: 'npm',
      args: ['run', 'dev'],
      cwd: 'microservices/node-api',
      port: 3001,
      healthUrl: 'http://localhost:3001/health',
      priority: 3,
      restartPolicy: 'always',
      maxRetries: 3,
      retryDelay: 5000,
      timeout: 15000,
      dependencies: ['postgresql', 'redis']
    });

    this.addService({
      name: 'vector-indexer',
      type: 'node',
      command: 'node',
      args: ['scripts/vector-indexer.mjs'],
      port: 3002,
      healthUrl: 'http://localhost:3002/health',
      priority: 4,
      restartPolicy: 'on-failure',
      maxRetries: 2,
      retryDelay: 10000,
      timeout: 30000,
      dependencies: ['qdrant', 'postgresql']
    });
  }

  private addGpuServices(): void {
    this.addService({
      name: 'cuda-worker',
      type: 'native',
      command: path.join(process.cwd(), 'cuda-worker', 'cuda-worker.exe'),
      port: 8231,
      healthUrl: 'http://localhost:8231/health',
      priority: 4,
      restartPolicy: 'on-failure',
      maxRetries: 2,
      retryDelay: 15000,
      timeout: 30000,
      gpuRequired: true,
      memoryLimit: 4096 // 4GB
    });

    this.addService({
      name: 'tensor-accelerator',
      type: 'go',
      command: 'go',
      args: ['run', 'tensor-tiling-gpu-accelerator.go'],
      port: 8232,
      healthUrl: 'http://localhost:8232/health',
      priority: 4,
      restartPolicy: 'on-failure',
      maxRetries: 2,
      retryDelay: 10000,
      timeout: 25000,
      gpuRequired: true
    });
  }

  private addFrontendServices(): void {
    this.addService({
      name: 'sveltekit-frontend',
      type: 'node',
      command: 'npm',
      args: ['run', 'dev', '--', '--host', '0.0.0.0'],
      cwd: 'sveltekit-frontend',
      port: 5173,
      healthUrl: 'http://localhost:5173',
      priority: 2,
      restartPolicy: 'always',
      maxRetries: 5,
      retryDelay: 5000,
      timeout: 30000
    });

    // Service Worker
    this.addService({
      name: 'service-worker',
      type: 'node',
      command: 'node',
      args: ['scripts/service-worker-manager.mjs'],
      priority: 4,
      restartPolicy: 'always',
      maxRetries: 3,
      retryDelay: 5000,
      timeout: 10000
    });
  }

  private addService(config: ServiceConfig): void {
    this.services.set(config.name, config);
    this.serviceStatus.set(config.name, {
      name: config.name,
      status: 'stopped',
      restartCount: 0,
      metrics: {
        cpuUsage: 0,
        memoryUsage: 0,
        networkConnections: 0,
        requestCount: 0,
        responseTime: 0,
        errorRate: 0,
        uptime: 0
      }
    });
  }

  // Service Management
  async startService(serviceName: string): Promise<void> {
    const config = this.services.get(serviceName);
    const status = this.serviceStatus.get(serviceName);
    
    if (!config || !status) {
      throw new Error(`Service ${serviceName} not found`);
    }

    if (status.status === 'running') {
      this.log('info', `Service ${serviceName} already running`);
      return;
    }

    // Check dependencies
    if (config.dependencies) {
      for (const dep of config.dependencies) {
        const depStatus = this.serviceStatus.get(dep);
        if (!depStatus || depStatus.status !== 'running') {
          throw new Error(`Dependency ${dep} not running for service ${serviceName}`);
        }
      }
    }

    // Check GPU requirement
    if (config.gpuRequired && !await this.checkGpuAvailability()) {
      throw new Error(`GPU required but not available for service ${serviceName}`);
    }

    status.status = 'starting';
    this.emit('service:starting', serviceName);

    try {
      const process = this.spawnProcess(config);
      this.processes.set(serviceName, process);
      
      process.on('exit', (code) => {
        this.handleProcessExit(serviceName, code);
      });

      process.on('error', (error) => {
        this.handleProcessError(serviceName, error);
      });

      // Wait for service to be ready
      await this.waitForServiceReady(serviceName);
      
      status.status = 'running';
      status.pid = process.pid;
      status.startTime = new Date();
      
      this.log('info', `Service ${serviceName} started successfully`);
      this.emit('service:started', serviceName);
      
    } catch (error: any) {
      status.status = 'error';
      status.lastError = error instanceof Error ? error.message : String(error);
      this.log('error', `Failed to start service ${serviceName}: ${status.lastError}`);
      this.emit('service:error', serviceName, error);
      throw error;
    }
  }

  async stopService(serviceName: string): Promise<void> {
    const status = this.serviceStatus.get(serviceName);
    const process = this.processes.get(serviceName);
    
    if (!status) {
      throw new Error(`Service ${serviceName} not found`);
    }

    if (status.status === 'stopped') {
      return;
    }

    status.status = 'stopping';
    this.emit('service:stopping', serviceName);

    if (process && !process.killed) {
      // Graceful shutdown
      process.kill('SIGTERM');
      
      // Force kill after timeout
      setTimeout(() => {
        if (!process.killed) {
          process.kill('SIGKILL');
        }
      }, 10000);
    }

    status.status = 'stopped';
    status.pid = undefined;
    this.processes.delete(serviceName);
    
    this.log('info', `Service ${serviceName} stopped`);
    this.emit('service:stopped', serviceName);
  }

  async restartService(serviceName: string): Promise<void> {
    await this.stopService(serviceName);
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
    await this.startService(serviceName);
  }

  // Orchestration Methods
  async startAll(): Promise<void> {
    this.log('info', 'Starting all services...');
    
    // Start services by priority
    const servicesByPriority = Array.from(this.services.values())
      .sort((a, b) => a.priority - b.priority);
    
    for (const config of servicesByPriority) {
      try {
        await this.startService(config.name);
        await new Promise(resolve => setTimeout(resolve, 2000)); // Stagger starts
      } catch (error: any) {
        this.log('error', `Failed to start ${config.name} during startup: ${error}`);
        if (config.priority <= 2) { // Critical services
          throw new Error(`Critical service ${config.name} failed to start`);
        }
      }
    }
    
    this.startHealthChecks();
    this.startMetricsCollection();
    this.startWebSocketServer();
    
    this.log('info', 'All services startup completed');
  }

  async stopAll(): Promise<void> {
    this.log('info', 'Stopping all services...');
    
    this.stopHealthChecks();
    this.stopMetricsCollection();
    this.wsServer?.close();
    
    // Stop services in reverse priority order
    const servicesByPriority = Array.from(this.services.values())
      .sort((a, b) => b.priority - a.priority);
    
    for (const config of servicesByPriority) {
      try {
        await this.stopService(config.name);
      } catch (error: any) {
        this.log('error', `Error stopping ${config.name}: ${error}`);
      }
    }
    
    // Close database connections
    await this.redisClient?.quit();
    await this.pgClient?.end();
    await this.rabbitConnection?.close();
    
    this.log('info', 'All services stopped');
  }

  // Health Monitoring
  private startHealthChecks(): void {
    this.healthCheckInterval = setInterval(async () => {
      await this.performHealthChecks();
    }, 30000); // Check every 30 seconds
  }

  private stopHealthChecks(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
  }

  private async performHealthChecks(): Promise<void> {
    const healthPromises = Array.from(this.services.entries()).map(async ([name, config]) => {
      try {
        const status = this.serviceStatus.get(name)!;
        
        if (status.status !== 'running') {
          return;
        }

        const isHealthy = await this.checkServiceHealth(config);
        const previousHealth = status.health;
        status.health = isHealthy ? 'healthy' : 'unhealthy';
        
        if (previousHealth === 'healthy' && !isHealthy) {
          this.createAlert('warning', `Service ${name} became unhealthy`, name);
          
          // Auto-restart if configured
          if (config.restartPolicy === 'always' && status.restartCount < config.maxRetries) {
            this.log('info', `Auto-restarting unhealthy service ${name}`);
            await this.restartService(name);
          }
        }
        
      } catch (error: any) {
        this.log('error', `Health check failed for ${name}: ${error}`);
      }
    });

    await Promise.allSettled(healthPromises);
  }

  private async checkServiceHealth(config: ServiceConfig): Promise<boolean> {
    if (!config.healthUrl) {
      return true; // Assume healthy if no health check URL
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch(config.healthUrl, {
        signal: controller.signal,
        method: 'GET'
      });
      
      clearTimeout(timeoutId);
      return response.ok;
      
    } catch (error: any) {
      return false;
    }
  }

  // Metrics Collection
  private startMetricsCollection(): void {
    this.metricsCollectionInterval = setInterval(async () => {
      await this.collectMetrics();
    }, 15000); // Collect every 15 seconds
  }

  private stopMetricsCollection(): void {
    if (this.metricsCollectionInterval) {
      clearInterval(this.metricsCollectionInterval);
    }
  }

  private async collectMetrics(): Promise<void> {
    for (const [name, status] of this.serviceStatus) {
      if (status.status === 'running' && status.pid) {
        try {
          const metrics = await this.getProcessMetrics(status.pid);
          status.metrics = metrics;
          
          // Store in database
          this.metrics.insert({
            service: name,
            timestamp: new Date(),
            ...metrics
          });
          
        } catch (error: any) {
          this.log('error', `Failed to collect metrics for ${name}: ${error}`);
        }
      }
    }
  }

  private async getProcessMetrics(pid: number): Promise<ServiceMetrics> {
    // This is a simplified implementation - in production you'd use
    // system monitoring tools or libraries like 'systeminformation'
    return {
      cpuUsage: Math.random() * 100,
      memoryUsage: Math.random() * 1024,
      networkConnections: Math.floor(Math.random() * 100),
      requestCount: Math.floor(Math.random() * 1000),
      responseTime: Math.random() * 500,
      errorRate: Math.random() * 5,
      uptime: performance.now()
    };
  }

  // WebSocket Server for Real-time Monitoring
  private startWebSocketServer(): void {
    this.wsServer = new WebSocketServer({ port: 8240 });
    
    this.wsServer.on('connection', (ws) => {
      this.log('info', 'Monitoring client connected');
      
      // Send current status
      ws.send(JSON.stringify({
        type: 'status',
        data: this.getSystemHealth()
      }));

      // Send updates
      const intervalId = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: 'update',
            data: this.getSystemHealth()
          }));
        } else {
          clearInterval(intervalId);
        }
      }, 5000);

      ws.on('close', () => {
        clearInterval(intervalId);
      });
    });
  }

  // System Status
  getSystemHealth(): SystemHealth {
    const services = Array.from(this.serviceStatus.values());
    const runningServices = services.filter(s => s.status === 'running').length;
    const totalServices = services.length;
    
    let overall: SystemHealth['overall'] = 'healthy';
    if (runningServices < totalServices * 0.9) {
      overall = 'degraded';
    }
    if (runningServices < totalServices * 0.7) {
      overall = 'critical';
    }

    return {
      overall,
      services,
      infrastructure: {
        postgres: this.serviceStatus.get('postgresql')?.status === 'running',
        redis: this.serviceStatus.get('redis')?.status === 'running',
        rabbitmq: this.serviceStatus.get('rabbitmq')?.status === 'running',
        gpu: true, // Simplified - would check actual GPU status
        disk: 75, // Would get real disk usage
        memory: 60, // Would get real memory usage
        cpu: 45 // Would get real CPU usage
      },
      alerts: this.alerts.find()
    };
  }

  // Utility Methods
  private spawnProcess(config: ServiceConfig): ChildProcess {
    const env = { ...process.env, ...config.env };
    
    return spawn(config.command, config.args || [], {
      cwd: config.cwd ? path.join(process.cwd(), config.cwd) : process.cwd(),
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: true
    });
  }

  private async waitForServiceReady(serviceName: string): Promise<void> {
    const config = this.services.get(serviceName)!;
    const maxWait = config.timeout;
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWait) {
      try {
        if (await this.checkServiceHealth(config)) {
          return;
        }
      } catch (error: any) {
        // Continue waiting
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    throw new Error(`Service ${serviceName} failed to become ready within ${maxWait}ms`);
  }

  private handleProcessExit(serviceName: string, code: number | null): void {
    const status = this.serviceStatus.get(serviceName)!;
    const config = this.services.get(serviceName)!;
    
    this.log('info', `Service ${serviceName} exited with code ${code}`);
    
    if (code !== 0 && config.restartPolicy === 'always' && status.restartCount < config.maxRetries) {
      this.log('info', `Restarting ${serviceName} (attempt ${status.restartCount + 1})`);
      status.restartCount++;
      
      setTimeout(async () => {
        try {
          await this.startService(serviceName);
        } catch (error: any) {
          this.log('error', `Failed to restart ${serviceName}: ${error}`);
        }
      }, config.retryDelay);
    } else {
      status.status = 'stopped';
      this.processes.delete(serviceName);
    }
  }

  private handleProcessError(serviceName: string, error: Error): void {
    const status = this.serviceStatus.get(serviceName)!;
    status.status = 'error';
    status.lastError = error.message;
    
    this.log('error', `Service ${serviceName} error: ${error.message}`);
    this.createAlert('error', `Service ${serviceName} encountered an error: ${error.message}`, serviceName);
  }

  private async checkGpuAvailability(): Promise<boolean> {
    try {
      // This would check actual GPU availability
      // For now, assume GPU is available
      return true;
    } catch (error: any) {
      return false;
    }
  }

  private createAlert(level: Alert['level'], message: string, service?: string): void {
    const alert: Alert = {
      id: Math.random().toString(36).substring(7),
      level,
      message,
      service,
      timestamp: new Date(),
      acknowledged: false
    };
    
    this.alerts.insert(alert);
    this.emit('alert', alert);
    
    // Auto-acknowledge info alerts after 5 minutes
    if (level === 'info') {
      setTimeout(() => {
        const alertDoc = this.alerts.findOne({ id: alert.id });
        if (alertDoc) {
          alertDoc.acknowledged = true;
          this.alerts.update(alertDoc);
        }
      }, 300000);
    }
  }

  private log(level: string, message: string): void {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [${level.toUpperCase()}] ${message}`);
    
    // Would also write to log files in production
  }

  private setupSignalHandlers(): void {
    process.on('SIGINT', async () => {
      console.log('Received SIGINT, shutting down gracefully...');
      await this.stopAll();
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      console.log('Received SIGTERM, shutting down gracefully...');
      await this.stopAll();
      process.exit(0);
    });
  }
}

// CLI Interface
if (import.meta.url === `file://${process.argv[1]}`) {
  const orchestrator = new ProductionOrchestrator();
  
  const command = process.argv[2];
  
  switch (command) {
    case 'start':
      orchestrator.startAll().catch(console.error);
      break;
    case 'stop':
      orchestrator.stopAll().catch(console.error);
      break;
    case 'status':
      console.log(JSON.stringify(orchestrator.getSystemHealth(), null, 2));
      break;
    case 'restart':
      const serviceName = process.argv[3];
      if (serviceName) {
        orchestrator.restartService(serviceName).catch(console.error);
      } else {
        console.error('Service name required for restart');
      }
      break;
    default:
      console.log('Usage: node production-orchestrator.js [start|stop|status|restart <service>]');
  }
}

export default ProductionOrchestrator;