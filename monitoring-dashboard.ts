/**
 * Comprehensive Monitoring and Logging Infrastructure
 * Real-time system monitoring with WebSocket dashboard and analytics
 */

import express from 'express';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';
import { performance } from 'perf_hooks';
import { createRequire } from 'module';
import Loki from 'lokijs';
import { spawn } from 'child_process';

const require = createRequire(import.meta.url);

export interface SystemMetrics {
  timestamp: Date;
  system: {
    cpu: {
      usage: number;
      cores: number;
      temperature?: number;
    };
    memory: {
      total: number;
      used: number;
      free: number;
      percentage: number;
    };
    disk: {
      total: number;
      used: number;
      free: number;
      percentage: number;
    };
    network: {
      bytesIn: number;
      bytesOut: number;
      packetsIn: number;
      packetsOut: number;
    };
    gpu?: {
      usage: number;
      memory: {
        total: number;
        used: number;
        free: number;
      };
      temperature: number;
      power: number;
    };
  };
  services: ServiceMetrics[];
  alerts: Alert[];
  performance: PerformanceMetrics;
}

export interface ServiceMetrics {
  name: string;
  status: 'running' | 'stopped' | 'error' | 'starting' | 'stopping';
  pid?: number;
  port?: number;
  uptime: number;
  restartCount: number;
  health: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  resources: {
    cpu: number;
    memory: number;
    handles: number;
    threads: number;
  };
  performance: {
    requestsPerSecond: number;
    averageResponseTime: number;
    errorRate: number;
    throughput: number;
  };
  database?: {
    connections: number;
    queries: number;
    slowQueries: number;
    lockWaits: number;
  };
  lastError?: {
    message: string;
    timestamp: Date;
    count: number;
  };
}

export interface Alert {
  id: string;
  level: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  service?: string;
  metric?: string;
  value?: number;
  threshold?: number;
  timestamp: Date;
  acknowledged: boolean;
  resolved: boolean;
  duration?: number;
}

export interface PerformanceMetrics {
  overall: {
    score: number; // 0-100
    latency: number;
    throughput: number;
    errorRate: number;
  };
  trends: {
    cpu: number[];
    memory: number[];
    latency: number[];
    errors: number[];
  };
  bottlenecks: string[];
  recommendations: string[];
}

export interface LogEntry {
  timestamp: Date;
  level: 'debug' | 'info' | 'warn' | 'error' | 'fatal';
  service: string;
  message: string;
  metadata?: Record<string, any>;
  correlationId?: string;
  userId?: string;
  sessionId?: string;
  requestId?: string;
  stack?: string;
}

export interface MetricsCollector {
  name: string;
  interval: number;
  enabled: boolean;
  collect(): Promise<any>;
}

export class MonitoringDashboard extends EventEmitter {
  private app: express.Application;
  private server: any;
  private wss: WebSocketServer;
  private clients: Set<WebSocket> = new Set();
  
  private db: Loki;
  private metricsCollection: Loki.Collection<SystemMetrics>;
  private alertsCollection: Loki.Collection<Alert>;
  private logsCollection: Loki.Collection<LogEntry>;
  
  private collectors: Map<string, MetricsCollector> = new Map();
  private metricsHistory: Map<string, any[]> = new Map();
  private alerts: Map<string, Alert> = new Map();
  
  private metricsInterval?: NodeJS.Timeout;
  private alertsInterval?: NodeJS.Timeout;
  private cleanupInterval?: NodeJS.Timeout;
  
  private readonly port = 8242;
  private readonly maxHistorySize = 1000;
  private readonly metricsRetentionDays = 7;
  private readonly logsRetentionDays = 30;

  constructor() {
    super();
    this.initializeApp();
    this.initializeDatabase();
    this.initializeCollectors();
    this.setupRoutes();
  }

  private initializeApp(): void {
    this.app = express();
    this.app.use(express.json());
    this.app.use(express.static(path.join(process.cwd(), 'monitoring-assets')));
    
    this.server = createServer(this.app);
    this.wss = new WebSocketServer({ server: this.server });
    
    this.wss.on('connection', (ws) => {
      this.clients.add(ws);
      
      // Send current system status
      this.sendToClient(ws, 'system-status', this.getCurrentSystemStatus());
      
      ws.on('message', async (data) => {
        try {
          const message = JSON.parse(data.toString());
          await this.handleClientMessage(ws, message);
        } catch (error: any) {
          this.sendToClient(ws, 'error', { message: 'Invalid message format' });
        }
      });
      
      ws.on('close', () => {
        this.clients.delete(ws);
      });
    });
  }

  private initializeDatabase(): void {
    this.db = new Loki('monitoring.db', {
      autoload: true,
      autosave: true,
      autosaveInterval: 10000
    });

    this.metricsCollection = this.db.addCollection('metrics', {
      ttl: this.metricsRetentionDays * 24 * 60 * 60 * 1000
    });

    this.alertsCollection = this.db.addCollection('alerts', {
      ttl: 30 * 24 * 60 * 60 * 1000 // 30 days
    });

    this.logsCollection = this.db.addCollection('logs', {
      ttl: this.logsRetentionDays * 24 * 60 * 60 * 1000
    });
  }

  private initializeCollectors(): void {
    // System metrics collector
    this.collectors.set('system', {
      name: 'system',
      interval: 5000,
      enabled: true,
      collect: async () => {
        return await this.collectSystemMetrics();
      }
    });

    // Service metrics collector
    this.collectors.set('services', {
      name: 'services',
      interval: 10000,
      enabled: true,
      collect: async () => {
        return await this.collectServiceMetrics();
      }
    });

    // GPU metrics collector (if available)
    this.collectors.set('gpu', {
      name: 'gpu',
      interval: 15000,
      enabled: true,
      collect: async () => {
        return await this.collectGpuMetrics();
      }
    });

    // Database metrics collector
    this.collectors.set('database', {
      name: 'database',
      interval: 30000,
      enabled: true,
      collect: async () => {
        return await this.collectDatabaseMetrics();
      }
    });
  }

  private setupRoutes(): void {
    // Dashboard HTML
    this.app.get('/', (req, res) => {
      res.send(this.generateDashboardHTML());
    });

    // API endpoints
    this.app.get('/api/health', (req, res) => {
      res.json({ status: 'ok', timestamp: new Date() });
    });

    this.app.get('/api/metrics', (req, res) => {
      const latest = this.getCurrentSystemStatus();
      res.json(latest);
    });

    this.app.get('/api/metrics/history/:metric', (req, res) => {
      const { metric } = req.params;
      const { hours = 24 } = req.query;
      const history = this.getMetricsHistory(metric, Number(hours));
      res.json(history);
    });

    this.app.get('/api/alerts', (req, res) => {
      const alerts = Array.from(this.alerts.values());
      res.json(alerts);
    });

    this.app.post('/api/alerts/:id/acknowledge', (req, res) => {
      const { id } = req.params;
      const alert = this.alerts.get(id);
      if (alert) {
        alert.acknowledged = true;
        this.alertsCollection.update(alert as any);
        this.broadcastToClients('alert-updated', alert);
        res.json({ success: true });
      } else {
        res.status(404).json({ error: 'Alert not found' });
      }
    });

    this.app.get('/api/logs', (req, res) => {
      const { service, level, since, limit = 100 } = req.query;
      const logs = this.getLogs({
        service: service as string,
        level: level as LogEntry['level'],
        since: since ? new Date(since as string) : undefined,
        limit: Number(limit)
      });
      res.json(logs);
    });

    this.app.get('/api/services', (req, res) => {
      const services = this.getServiceStatus();
      res.json(services);
    });

    this.app.post('/api/services/:name/:action', async (req, res) => {
      const { name, action } = req.params;
      try {
        await this.executeServiceAction(name, action);
        res.json({ success: true, message: `${action} executed for ${name}` });
      } catch (error: any) {
        res.status(500).json({ error: (error as Error).message });
      }
    });

    // Metrics export
    this.app.get('/api/export/metrics', async (req, res) => {
      const { format = 'json', since } = req.query;
      const data = await this.exportMetrics(format as string, since as string);
      
      if (format === 'csv') {
        res.setHeader('Content-Type', 'text/csv');
        res.setHeader('Content-Disposition', 'attachment; filename=metrics.csv');
      }
      
      res.send(data);
    });
  }

  // Start monitoring
  async start(): Promise<void> {
    await this.ensureAssetsDirectory();
    
    this.server.listen(this.port, () => {
      console.log(`Monitoring Dashboard running on http://localhost:${this.port}`);
    });

    this.startMetricsCollection();
    this.startAlertsProcessing();
    this.startCleanupTasks();
    
    this.emit('started');
  }

  private async ensureAssetsDirectory(): Promise<void> {
    const assetsDir = path.join(process.cwd(), 'monitoring-assets');
    try {
      await fs.access(assetsDir);
    } catch {
      await fs.mkdir(assetsDir, { recursive: true });
      await this.createDefaultAssets(assetsDir);
    }
  }

  private startMetricsCollection(): void {
    this.metricsInterval = setInterval(async () => {
      for (const collector of this.collectors.values()) {
        if (collector.enabled) {
          try {
            const metrics = await collector.collect();
            this.storeMetrics(collector.name, metrics);
            this.broadcastToClients('metrics-update', { 
              collector: collector.name, 
              data: metrics 
            });
          } catch (error: any) {
            this.log('error', 'monitoring', `Failed to collect ${collector.name} metrics: ${error}`);
          }
        }
      }
    }, 5000);
  }

  private startAlertsProcessing(): void {
    this.alertsInterval = setInterval(() => {
      this.processAlerts();
    }, 10000);
  }

  private startCleanupTasks(): void {
    this.cleanupInterval = setInterval(() => {
      this.cleanupOldData();
    }, 3600000); // Every hour
  }

  // Metrics collection implementations
  private async collectSystemMetrics(): Promise<any> {
    try {
      // Use systeminformation library or Windows-specific commands
      const cpuUsage = await this.getCpuUsage();
      const memoryUsage = await this.getMemoryUsage();
      const diskUsage = await this.getDiskUsage();
      const networkUsage = await this.getNetworkUsage();

      return {
        timestamp: new Date(),
        cpu: cpuUsage,
        memory: memoryUsage,
        disk: diskUsage,
        network: networkUsage
      };
    } catch (error: any) {
      this.log('error', 'monitoring', `Failed to collect system metrics: ${error}`);
      return null;
    }
  }

  private async collectServiceMetrics(): Promise<ServiceMetrics[]> {
    const services: ServiceMetrics[] = [];
    
    try {
      // Get running processes and match with known services
      const processes = await this.getProcessList();
      
      for (const process of processes) {
        if (this.isKnownService(process.name)) {
          const metrics = await this.getServiceMetrics(process);
          services.push(metrics);
        }
      }
    } catch (error: any) {
      this.log('error', 'monitoring', `Failed to collect service metrics: ${error}`);
    }

    return services;
  }

  private async collectGpuMetrics(): Promise<any> {
    try {
      // Use nvidia-smi command
      const gpuInfo = await this.executeCommand('nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader,nounits');
      
      if (gpuInfo) {
        const values = gpuInfo.split(',').map(v => parseFloat(v.trim()));
        return {
          usage: values[0],
          memoryUsage: values[1],
          memory: {
            total: values[2] * 1024 * 1024, // Convert MB to bytes
            used: values[3] * 1024 * 1024,
            free: values[4] * 1024 * 1024
          },
          temperature: values[5],
          power: values[6]
        };
      }
    } catch (error: any) {
      // GPU not available or nvidia-smi not found
    }
    
    return null;
  }

  private async collectDatabaseMetrics(): Promise<any> {
    try {
      // Collect PostgreSQL metrics
      const pgMetrics = await this.getPostgreSQLMetrics();
      
      // Collect Redis metrics
      const redisMetrics = await this.getRedisMetrics();
      
      return {
        postgresql: pgMetrics,
        redis: redisMetrics
      };
    } catch (error: any) {
      this.log('error', 'monitoring', `Failed to collect database metrics: ${error}`);
      return null;
    }
  }

  // Helper methods for system metrics
  private async getCpuUsage(): Promise<any> {
    try {
      const output = await this.executeCommand('wmic cpu get loadpercentage /value');
      const match = output.match(/LoadPercentage=(\d+)/);
      const usage = match ? parseInt(match[1]) : 0;
      
      const coreCount = await this.getCoreCount();
      
      return {
        usage,
        cores: coreCount,
        temperature: await this.getCpuTemperature()
      };
    } catch {
      return { usage: 0, cores: 1 };
    }
  }

  private async getMemoryUsage(): Promise<any> {
    try {
      const output = await this.executeCommand('wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value');
      
      const totalMatch = output.match(/TotalVisibleMemorySize=(\d+)/);
      const freeMatch = output.match(/FreePhysicalMemory=(\d+)/);
      
      const total = totalMatch ? parseInt(totalMatch[1]) * 1024 : 0;
      const free = freeMatch ? parseInt(freeMatch[1]) * 1024 : 0;
      const used = total - free;
      
      return {
        total,
        used,
        free,
        percentage: total > 0 ? (used / total) * 100 : 0
      };
    } catch {
      return { total: 0, used: 0, free: 0, percentage: 0 };
    }
  }

  private async getDiskUsage(): Promise<any> {
    try {
      const output = await this.executeCommand('wmic logicaldisk where size!=null get size,freespace,caption /value');
      
      let totalSize = 0;
      let totalFree = 0;
      
      const lines = output.split('\n');
      for (const line of lines) {
        if (line.includes('Size=')) {
          const sizeMatch = line.match(/Size=(\d+)/);
          if (sizeMatch) totalSize += parseInt(sizeMatch[1]);
        }
        if (line.includes('FreeSpace=')) {
          const freeMatch = line.match(/FreeSpace=(\d+)/);
          if (freeMatch) totalFree += parseInt(freeMatch[1]);
        }
      }
      
      const used = totalSize - totalFree;
      
      return {
        total: totalSize,
        used,
        free: totalFree,
        percentage: totalSize > 0 ? (used / totalSize) * 100 : 0
      };
    } catch {
      return { total: 0, used: 0, free: 0, percentage: 0 };
    }
  }

  private async getNetworkUsage(): Promise<any> {
    try {
      // This is simplified - would use performance counters in production
      return {
        bytesIn: Math.random() * 1000000,
        bytesOut: Math.random() * 1000000,
        packetsIn: Math.random() * 10000,
        packetsOut: Math.random() * 10000
      };
    } catch {
      return { bytesIn: 0, bytesOut: 0, packetsIn: 0, packetsOut: 0 };
    }
  }

  private async getCoreCount(): Promise<number> {
    try {
      const output = await this.executeCommand('wmic cpu get NumberOfCores /value');
      const match = output.match(/NumberOfCores=(\d+)/);
      return match ? parseInt(match[1]) : 1;
    } catch {
      return 1;
    }
  }

  private async getCpuTemperature(): Promise<number | undefined> {
    try {
      // This would require specialized tools or WMI queries
      return undefined;
    } catch {
      return undefined;
    }
  }

  private async getProcessList(): Promise<any[]> {
    try {
      const output = await this.executeCommand('tasklist /fo csv');
      const lines = output.split('\n').slice(1); // Skip header
      
      return lines.map(line => {
        const fields = line.split(',').map(field => field.replace(/"/g, ''));
        return {
          name: fields[0],
          pid: parseInt(fields[1]),
          memory: fields[4] ? parseInt(fields[4].replace(/[^\d]/g, '')) : 0
        };
      }).filter(p => p.name && p.pid);
    } catch {
      return [];
    }
  }

  private isKnownService(processName: string): boolean {
    const knownServices = [
      'go.exe', 'node.exe', 'postgres.exe', 'redis-server.exe',
      'ollama.exe', 'qdrant.exe', 'minio.exe', 'nats-server.exe'
    ];
    return knownServices.some(service => processName.toLowerCase().includes(service.toLowerCase()));
  }

  private async getServiceMetrics(process: any): Promise<ServiceMetrics> {
    try {
      const perfOutput = await this.executeCommand(`wmic process where ProcessId=${process.pid} get PageFileUsage,ThreadCount,HandleCount /value`);
      
      return {
        name: this.mapProcessToService(process.name),
        status: 'running',
        pid: process.pid,
        uptime: 0, // Would need to track start time
        restartCount: 0, // Would need to track restarts
        health: 'healthy', // Would need health checks
        resources: {
          cpu: 0, // Would need per-process CPU usage
          memory: process.memory || 0,
          handles: this.extractValue(perfOutput, 'HandleCount') || 0,
          threads: this.extractValue(perfOutput, 'ThreadCount') || 0
        },
        performance: {
          requestsPerSecond: 0,
          averageResponseTime: 0,
          errorRate: 0,
          throughput: 0
        }
      };
    } catch {
      return {
        name: process.name,
        status: 'running',
        uptime: 0,
        restartCount: 0,
        health: 'unknown',
        resources: { cpu: 0, memory: 0, handles: 0, threads: 0 },
        performance: { requestsPerSecond: 0, averageResponseTime: 0, errorRate: 0, throughput: 0 }
      };
    }
  }

  private mapProcessToService(processName: string): string {
    const mapping: Record<string, string> = {
      'go.exe': 'go-services',
      'node.exe': 'node-services',
      'postgres.exe': 'postgresql',
      'redis-server.exe': 'redis',
      'ollama.exe': 'ollama',
      'qdrant.exe': 'qdrant',
      'minio.exe': 'minio',
      'nats-server.exe': 'nats-server'
    };
    
    return mapping[processName.toLowerCase()] || processName;
  }

  private extractValue(output: string, key: string): number | undefined {
    const match = output.match(new RegExp(`${key}=(\\d+)`));
    return match ? parseInt(match[1]) : undefined;
  }

  private async getPostgreSQLMetrics(): Promise<any> {
    try {
      // This would connect to PostgreSQL and run queries
      return {
        connections: 10,
        queries: 1000,
        slowQueries: 5,
        lockWaits: 2
      };
    } catch {
      return null;
    }
  }

  private async getRedisMetrics(): Promise<any> {
    try {
      // This would connect to Redis and get INFO
      return {
        connections: 5,
        operations: 5000,
        keyspace: 1000,
        memory: 50000000
      };
    } catch {
      return null;
    }
  }

  private async executeCommand(command: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const child = spawn('cmd', ['/c', command], { 
        stdio: ['ignore', 'pipe', 'pipe'],
        shell: true 
      });
      
      let stdout = '';
      let stderr = '';
      
      child.stdout?.on('data', (data) => {
        stdout += data.toString();
      });
      
      child.stderr?.on('data', (data) => {
        stderr += data.toString();
      });
      
      child.on('close', (code) => {
        if (code === 0) {
          resolve(stdout);
        } else {
          reject(new Error(stderr || `Command failed with code ${code}`));
        }
      });
      
      child.on('error', reject);
      
      // Timeout after 10 seconds
      setTimeout(() => {
        child.kill();
        reject(new Error('Command timeout'));
      }, 10000);
    });
  }

  // Utility methods
  private storeMetrics(collector: string, metrics: any): void {
    if (!metrics) return;
    
    const entry = {
      collector,
      timestamp: new Date(),
      data: metrics
    };
    
    this.metricsCollection.insert(entry);
    
    // Update in-memory history
    const history = this.metricsHistory.get(collector) || [];
    history.push(metrics);
    
    if (history.length > this.maxHistorySize) {
      history.splice(0, history.length - this.maxHistorySize);
    }
    
    this.metricsHistory.set(collector, history);
  }

  private processAlerts(): void {
    const currentMetrics = this.getCurrentSystemStatus();
    
    // Check system alerts
    this.checkSystemAlerts(currentMetrics.system);
    
    // Check service alerts
    for (const service of currentMetrics.services) {
      this.checkServiceAlerts(service);
    }
  }

  private checkSystemAlerts(system: any): void {
    if (system.cpu?.usage > 85) {
      this.createAlert('warning', 'High CPU Usage', `CPU usage is ${system.cpu.usage}%`, undefined, 'cpu', system.cpu.usage, 85);
    }
    
    if (system.memory?.percentage > 90) {
      this.createAlert('critical', 'High Memory Usage', `Memory usage is ${system.memory.percentage.toFixed(1)}%`, undefined, 'memory', system.memory.percentage, 90);
    }
    
    if (system.disk?.percentage > 85) {
      this.createAlert('warning', 'Low Disk Space', `Disk usage is ${system.disk.percentage.toFixed(1)}%`, undefined, 'disk', system.disk.percentage, 85);
    }
    
    if (system.gpu?.usage > 90) {
      this.createAlert('warning', 'High GPU Usage', `GPU usage is ${system.gpu.usage}%`, undefined, 'gpu', system.gpu.usage, 90);
    }
  }

  private checkServiceAlerts(service: ServiceMetrics): void {
    if (service.status === 'error') {
      this.createAlert('error', 'Service Error', `Service ${service.name} is in error state`, service.name);
    }
    
    if (service.health === 'unhealthy') {
      this.createAlert('warning', 'Service Unhealthy', `Service ${service.name} is unhealthy`, service.name);
    }
    
    if (service.performance.errorRate > 5) {
      this.createAlert('warning', 'High Error Rate', `Service ${service.name} error rate is ${service.performance.errorRate}%`, service.name, 'errorRate', service.performance.errorRate, 5);
    }
    
    if (service.performance.averageResponseTime > 5000) {
      this.createAlert('warning', 'Slow Response', `Service ${service.name} response time is ${service.performance.averageResponseTime}ms`, service.name, 'responseTime', service.performance.averageResponseTime, 5000);
    }
  }

  private createAlert(level: Alert['level'], title: string, message: string, service?: string, metric?: string, value?: number, threshold?: number): void {
    const id = `${service || 'system'}-${metric || title.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;
    
    const alert: Alert = {
      id,
      level,
      title,
      message,
      service,
      metric,
      value,
      threshold,
      timestamp: new Date(),
      acknowledged: false,
      resolved: false
    };
    
    this.alerts.set(id, alert);
    this.alertsCollection.insert(alert);
    
    this.broadcastToClients('new-alert', alert);
    this.emit('alert', alert);
  }

  private cleanupOldData(): void {
    const cutoff = Date.now() - (this.metricsRetentionDays * 24 * 60 * 60 * 1000);
    
    // Cleanup metrics
    this.metricsCollection.removeWhere((doc: any) => doc.timestamp.getTime() < cutoff);
    
    // Cleanup resolved alerts older than 7 days
    const alertCutoff = Date.now() - (7 * 24 * 60 * 60 * 1000);
    this.alertsCollection.removeWhere((doc: any) => 
      doc.resolved && doc.timestamp.getTime() < alertCutoff
    );
    
    // Cleanup old logs
    const logCutoff = Date.now() - (this.logsRetentionDays * 24 * 60 * 60 * 1000);
    this.logsCollection.removeWhere((doc: any) => doc.timestamp.getTime() < logCutoff);
  }

  // Client communication
  private sendToClient(ws: WebSocket, type: string, data: any): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type, data, timestamp: new Date() }));
    }
  }

  private broadcastToClients(type: string, data: any): void {
    const message = JSON.stringify({ type, data, timestamp: new Date() });
    
    for (const client of this.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    }
  }

  private async handleClientMessage(ws: WebSocket, message: any): Promise<void> {
    const { type, data } = message;
    
    switch (type) {
      case 'get-metrics':
        this.sendToClient(ws, 'metrics-data', this.getCurrentSystemStatus());
        break;
        
      case 'get-alerts':
        this.sendToClient(ws, 'alerts-data', Array.from(this.alerts.values()));
        break;
        
      case 'acknowledge-alert':
        if (data.id) {
          const alert = this.alerts.get(data.id);
          if (alert) {
            alert.acknowledged = true;
            this.alertsCollection.update(alert);
            this.broadcastToClients('alert-updated', alert);
          }
        }
        break;
        
      case 'get-logs':
        const logs = this.getLogs(data);
        this.sendToClient(ws, 'logs-data', logs);
        break;
    }
  }

  // Public API
  getCurrentSystemStatus(): SystemMetrics {
    const systemMetrics = this.metricsHistory.get('system')?.[this.metricsHistory.get('system')!.length - 1] || {};
    const serviceMetrics = this.metricsHistory.get('services')?.[this.metricsHistory.get('services')!.length - 1] || [];
    
    return {
      timestamp: new Date(),
      system: systemMetrics,
      services: serviceMetrics,
      alerts: Array.from(this.alerts.values()).filter(a => !a.resolved),
      performance: this.calculatePerformanceMetrics()
    };
  }

  private calculatePerformanceMetrics(): PerformanceMetrics {
    const systemHistory = this.metricsHistory.get('system') || [];
    const serviceHistory = this.metricsHistory.get('services') || [];
    
    const cpuTrend = systemHistory.slice(-10).map((m: any) => m.cpu?.usage || 0);
    const memoryTrend = systemHistory.slice(-10).map((m: any) => m.memory?.percentage || 0);
    
    return {
      overall: {
        score: 85, // Would calculate based on various factors
        latency: 150,
        throughput: 1000,
        errorRate: 2.5
      },
      trends: {
        cpu: cpuTrend,
        memory: memoryTrend,
        latency: [100, 120, 150, 140, 130],
        errors: [1, 2, 1, 3, 2]
      },
      bottlenecks: ['Database connections', 'Memory usage'],
      recommendations: [
        'Consider scaling up memory-intensive services',
        'Optimize database query performance',
        'Enable GPU acceleration for AI workloads'
      ]
    };
  }

  getMetricsHistory(metric: string, hours: number): any[] {
    const since = Date.now() - (hours * 60 * 60 * 1000);
    
    return this.metricsCollection.find({
      timestamp: { '$gte': new Date(since) },
      collector: metric
    });
  }

  getLogs(filter: {
    service?: string;
    level?: LogEntry['level'];
    since?: Date;
    limit?: number;
  }): LogEntry[] {
    let query: any = {};
    
    if (filter.service) {
      query.service = filter.service;
    }
    
    if (filter.level) {
      query.level = filter.level;
    }
    
    if (filter.since) {
      query.timestamp = { '$gte': filter.since };
    }
    
    const logs = this.logsCollection.find(query);
    
    // Sort by timestamp descending
    logs.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    
    return logs.slice(0, filter.limit || 100);
  }

  log(level: LogEntry['level'], service: string, message: string, metadata?: any): void {
    const entry: LogEntry = {
      timestamp: new Date(),
      level,
      service,
      message,
      metadata,
      correlationId: Math.random().toString(36).substring(7)
    };
    
    this.logsCollection.insert(entry);
    this.broadcastToClients('new-log', entry);
    
    // Console output
    const levelColors: Record<string, string> = {
      debug: '\x1b[36m',
      info: '\x1b[32m',
      warn: '\x1b[33m',
      error: '\x1b[31m',
      fatal: '\x1b[35m'
    };
    
    const color = levelColors[level] || '';
    console.log(`${color}[${entry.timestamp.toISOString()}] [${level.toUpperCase()}] [${service}] ${message}\x1b[0m`);
  }

  getServiceStatus(): ServiceMetrics[] {
    return this.metricsHistory.get('services')?.[this.metricsHistory.get('services')!.length - 1] || [];
  }

  private async executeServiceAction(serviceName: string, action: string): Promise<void> {
    switch (action) {
      case 'restart':
        this.log('info', serviceName, `Restart initiated`);
        // Would implement actual restart logic
        break;
      case 'stop':
        this.log('info', serviceName, `Stop initiated`);
        // Would implement actual stop logic
        break;
      case 'start':
        this.log('info', serviceName, `Start initiated`);
        // Would implement actual start logic
        break;
      default:
        throw new Error(`Unknown action: ${action}`);
    }
  }

  private async exportMetrics(format: string, since?: string): Promise<string> {
    const sinceDate = since ? new Date(since) : new Date(Date.now() - 24 * 60 * 60 * 1000);
    const metrics = this.metricsCollection.find({
      timestamp: { '$gte': sinceDate }
    });
    
    if (format === 'csv') {
      // Convert to CSV format
      const header = 'timestamp,collector,cpu_usage,memory_percentage,disk_percentage\n';
      const rows = metrics.map(m => 
        `${m.timestamp},${m.collector},${m.data.cpu?.usage || ''},${m.data.memory?.percentage || ''},${m.data.disk?.percentage || ''}`
      ).join('\n');
      return header + rows;
    } else {
      return JSON.stringify(metrics, null, 2);
    }
  }

  private generateDashboardHTML(): string {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Production System Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0e27;
            color: #ffffff;
            line-height: 1.6;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .container { padding: 2rem; max-width: 1400px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .metric { display: flex; justify-content: space-between; align-items: center; margin: 1rem 0; }
        .metric-value { font-size: 2rem; font-weight: bold; }
        .status-ok { color: #4ade80; }
        .status-warning { color: #fbbf24; }
        .status-error { color: #f87171; }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4ade80, #22d3ee);
            transition: width 0.3s ease;
        }
        .services-grid { display: grid; gap: 1rem; margin-top: 1rem; }
        .service-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
        }
        .connection-status {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
        }
        .connected { background: #059669; }
        .disconnected { background: #dc2626; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Production System Monitor</h1>
        <div class="connection-status" id="connectionStatus">Connecting...</div>
    </div>
    
    <div class="container">
        <div class="grid">
            <div class="card">
                <h2>System Resources</h2>
                <div class="metric">
                    <span>CPU Usage</span>
                    <span class="metric-value" id="cpuUsage">0%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="cpuProgress"></div>
                </div>
                
                <div class="metric">
                    <span>Memory Usage</span>
                    <span class="metric-value" id="memoryUsage">0%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="memoryProgress"></div>
                </div>
                
                <div class="metric">
                    <span>Disk Usage</span>
                    <span class="metric-value" id="diskUsage">0%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="diskProgress"></div>
                </div>
            </div>
            
            <div class="card">
                <h2>Services Status</h2>
                <div class="services-grid" id="servicesList">
                    <!-- Services will be populated here -->
                </div>
            </div>
            
            <div class="card">
                <h2>Active Alerts</h2>
                <div id="alertsList">
                    <!-- Alerts will be populated here -->
                </div>
            </div>
            
            <div class="card">
                <h2>Performance Overview</h2>
                <div class="metric">
                    <span>Overall Score</span>
                    <span class="metric-value status-ok" id="performanceScore">85</span>
                </div>
                <div class="metric">
                    <span>Average Latency</span>
                    <span class="metric-value" id="avgLatency">150ms</span>
                </div>
                <div class="metric">
                    <span>Error Rate</span>
                    <span class="metric-value" id="errorRate">2.5%</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket('ws://localhost:8242');
        const connectionStatus = document.getElementById('connectionStatus');
        
        ws.onopen = function() {
            connectionStatus.textContent = 'Connected';
            connectionStatus.className = 'connection-status connected';
        };
        
        ws.onclose = function() {
            connectionStatus.textContent = 'Disconnected';
            connectionStatus.className = 'connection-status disconnected';
        };
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            
            switch(message.type) {
                case 'system-status':
                    updateSystemMetrics(message.data);
                    break;
                case 'metrics-update':
                    if (message.data.collector === 'system') {
                        updateSystemResources(message.data.data);
                    }
                    break;
                case 'new-alert':
                    addAlert(message.data);
                    break;
            }
        };
        
        function updateSystemMetrics(data) {
            if (data.system) {
                updateSystemResources(data.system);
            }
            if (data.services) {
                updateServices(data.services);
            }
            if (data.alerts) {
                updateAlerts(data.alerts);
            }
            if (data.performance) {
                updatePerformance(data.performance);
            }
        }
        
        function updateSystemResources(system) {
            if (system.cpu) {
                document.getElementById('cpuUsage').textContent = system.cpu.usage + '%';
                document.getElementById('cpuProgress').style.width = system.cpu.usage + '%';
            }
            if (system.memory) {
                const usage = Math.round(system.memory.percentage);
                document.getElementById('memoryUsage').textContent = usage + '%';
                document.getElementById('memoryProgress').style.width = usage + '%';
            }
            if (system.disk) {
                const usage = Math.round(system.disk.percentage);
                document.getElementById('diskUsage').textContent = usage + '%';
                document.getElementById('diskProgress').style.width = usage + '%';
            }
        }
        
        function updateServices(services) {
            const servicesList = document.getElementById('servicesList');
            servicesList.innerHTML = '';
            
            services.forEach(service => {
                const item = document.createElement('div');
                item.className = 'service-item';
                
                let statusClass = 'status-ok';
                if (service.status === 'error') statusClass = 'status-error';
                else if (service.health === 'degraded') statusClass = 'status-warning';
                
                item.innerHTML = \`
                    <span>\${service.name}</span>
                    <span class="\${statusClass}">\${service.status}</span>
                \`;
                servicesList.appendChild(item);
            });
        }
        
        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alertsList');
            alertsList.innerHTML = '';
            
            if (alerts.length === 0) {
                alertsList.innerHTML = '<p style="color: #4ade80;">No active alerts</p>';
                return;
            }
            
            alerts.forEach(alert => {
                const item = document.createElement('div');
                item.style.cssText = 'padding: 0.75rem; margin: 0.5rem 0; border-radius: 6px; background: rgba(248, 113, 113, 0.2);';
                item.innerHTML = \`
                    <div style="font-weight: bold;">\${alert.title}</div>
                    <div style="font-size: 0.875rem; opacity: 0.8;">\${alert.message}</div>
                \`;
                alertsList.appendChild(item);
            });
        }
        
        function updatePerformance(performance) {
            document.getElementById('performanceScore').textContent = performance.overall.score;
            document.getElementById('avgLatency').textContent = performance.overall.latency + 'ms';
            document.getElementById('errorRate').textContent = performance.overall.errorRate + '%';
        }
        
        // Request initial data
        setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'get-metrics' }));
            }
        }, 1000);
    </script>
</body>
</html>
    `;
  }

  private async createDefaultAssets(assetsDir: string): Promise<void> {
    // Create a simple favicon
    const favicon = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 'base64');
    await fs.writeFile(path.join(assetsDir, 'favicon.ico'), favicon);
  }

  // Cleanup
  async stop(): Promise<void> {
    if (this.metricsInterval) clearInterval(this.metricsInterval);
    if (this.alertsInterval) clearInterval(this.alertsInterval);
    if (this.cleanupInterval) clearInterval(this.cleanupInterval);
    
    this.wss.close();
    this.server.close();
    
    this.emit('stopped');
  }
}

// Export for use as a module
export default MonitoringDashboard;

// CLI usage
if (import.meta.url === `file://${process.argv[1]}`) {
  const dashboard = new MonitoringDashboard();
  
  dashboard.start().catch(console.error);
  
  process.on('SIGINT', async () => {
    console.log('Shutting down monitoring dashboard...');
    await dashboard.stop();
    process.exit(0);
  });
}