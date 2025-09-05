/**
 * Process Health Monitor for Cached Legal AI Workers
 * Comprehensive monitoring for Node.js processes, Go services, and Python/CUDA workers
 * Integrates with NATS messaging, Redis caching, and existing platform architecture
 */
import { EventEmitter } from 'events';
import { legalAIProcessPool } from './process-pool-manager';
import { legalAIWorkerClient } from './worker-pool-client';
import { legalAIGPUManager } from './gpu-memory-manager';
import Redis from 'ioredis';

export interface HealthMetrics {
  processId: string;
  poolName: string;
  type: 'node' | 'go' | 'python-cuda';
  status: 'healthy' | 'degraded' | 'unhealthy' | 'critical';
  uptime: number;
  memoryUsageMB: number;
  cpuUsagePercent: number;
  responseTime: number;
  errorRate: number;
  requestCount: number;
  lastHealthCheck: number;
  alerts: HealthAlert[];
}

export interface HealthAlert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: number;
  resolved: boolean;
  metadata?: any;
}

export interface SystemHealthSummary {
  overall: 'healthy' | 'degraded' | 'unhealthy' | 'critical';
  processHealth: Record<string, HealthMetrics>;
  gpuHealth: {
    memoryUsage: number;
    temperature: number;
    utilization: number;
    status: 'healthy' | 'degraded' | 'unhealthy' | 'critical';
  };
  poolStats: any;
  activeAlerts: HealthAlert[];
  recommendations: string[];
}

export class ProcessHealthMonitor extends EventEmitter {
  private redis: Redis;
  private healthMetrics: Map<string, HealthMetrics> = new Map();
  private alerts: Map<string, HealthAlert> = new Map();
  private monitoringInterval: NodeJS.Timeout;
  private alertInterval: NodeJS.Timeout;

  // Health thresholds
  private readonly thresholds = {
    memory: {
      warning: 512, // MB
      critical: 1024
    },
    cpu: {
      warning: 70, // %
      critical: 90
    },
    responseTime: {
      warning: 5000, // ms
      critical: 10000
    },
    errorRate: {
      warning: 0.05, // 5%
      critical: 0.15 // 15%
    },
    gpu: {
      memory: {
        warning: 0.80, // 80%
        critical: 0.95 // 95%
      },
      temperature: {
        warning: 75, // °C
        critical: 85
      }
    }
  };

  constructor(redisUrl: string = 'redis://localhost:6379') {
    super();
    this.redis = new Redis(redisUrl);
    
    // Monitor health every 30 seconds
    this.monitoringInterval = setInterval(() => this.performHealthChecks(), 30000);
    
    // Process alerts every 10 seconds
    this.alertInterval = setInterval(() => this.processAlerts(), 10000);
    
    this.initializeMonitoring();
  }

  /**
   * Initialize monitoring system
   */
  private initializeMonitoring(): void {
    // Listen to process pool events
    legalAIProcessPool.on('worker:created', (data) => {
      this.initializeWorkerHealth(data.workerId, data.poolName);
    });

    legalAIProcessPool.on('worker:terminated', (data) => {
      this.removeWorkerHealth(data.workerId);
    });

    // Listen to client events
    legalAIWorkerClient.on('request:completed', (data) => {
      this.updateWorkerMetrics(data.requestId, { 
        responseTime: data.responseTime,
        success: true 
      });
    });

    legalAIWorkerClient.on('request:failed', (data) => {
      this.updateWorkerMetrics(data.requestId, { 
        responseTime: data.responseTime,
        success: false,
        error: data.error 
      });
    });

    // Listen to GPU events
    legalAIGPUManager.on('gpu:high_memory_usage', (data) => {
      this.createAlert('gpu-memory', 'high', 
        `GPU memory usage is high: ${data.usagePercent}%`, data);
    });

    legalAIGPUManager.on('gpu:high_temperature', (data) => {
      this.createAlert('gpu-temperature', 'critical',
        `GPU temperature is critical: ${data.temperature}°C`, data);
    });

    console.log('🏥 Process health monitoring initialized');
  }

  /**
   * Perform comprehensive health checks
   */
  private async performHealthChecks(): Promise<any> {
    try {
      // Check process pool health
      await this.checkProcessPoolHealth();
      
      // Check GPU health  
      await this.checkGPUHealth();
      
      // Check Redis connectivity
      await this.checkRedisHealth();
      
      // Analyze overall system health
      const summary = await this.getSystemHealthSummary();
      this.emit('health:summary', summary);
      
      // Store health snapshot in Redis
      await this.redis.setex('legal_ai:health_snapshot', 300, JSON.stringify(summary));
      
    } catch (error: any) {
      this.createAlert('monitor-error', 'high',
        `Health monitoring error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        { error }
      );
    }
  }

  /**
   * Check process pool health
   */
  private async checkProcessPoolHealth(): Promise<any> {
    const poolStats = legalAIProcessPool.getStats();
    
    for (const [poolName, stats] of Object.entries(poolStats)) {
      const poolHealth = this.assessPoolHealth(poolName, stats as any);
      
      if (poolHealth.status === 'critical') {
        this.createAlert(`pool-${poolName}`, 'critical',
          `Pool ${poolName} is in critical state`, poolHealth);
      } else if (poolHealth.status === 'degraded') {
        this.createAlert(`pool-${poolName}`, 'medium',
          `Pool ${poolName} performance is degraded`, poolHealth);
      }
    }
  }

  /**
   * Check GPU health
   */
  private async checkGPUHealth(): Promise<any> {
    const gpuStats = await legalAIGPUManager.getGPUStats();
    
    const memoryUsage = gpuStats.usedMemoryMB / gpuStats.totalMemoryMB;
    const temperature = gpuStats.temperature;
    const utilization = gpuStats.utilization;
    
    // Check memory usage
    if (memoryUsage > this.thresholds.gpu.memory.critical) {
      this.createAlert('gpu-memory-critical', 'critical',
        `GPU memory critically low: ${(memoryUsage * 100).toFixed(1)}%`, gpuStats);
    } else if (memoryUsage > this.thresholds.gpu.memory.warning) {
      this.createAlert('gpu-memory-warning', 'medium',
        `GPU memory usage high: ${(memoryUsage * 100).toFixed(1)}%`, gpuStats);
    }
    
    // Check temperature
    if (temperature > this.thresholds.gpu.temperature.critical) {
      this.createAlert('gpu-temp-critical', 'critical',
        `GPU temperature critical: ${temperature}°C`, gpuStats);
    } else if (temperature > this.thresholds.gpu.temperature.warning) {
      this.createAlert('gpu-temp-warning', 'medium',
        `GPU temperature high: ${temperature}°C`, gpuStats);
    }
    
    // Update GPU health metrics
    await this.redis.hset('legal_ai:gpu_health', {
      memory_usage: memoryUsage.toFixed(3),
      temperature,
      utilization,
      status: this.getGPUHealthStatus(memoryUsage, temperature),
      last_check: Date.now()
    });
  }

  /**
   * Check Redis health
   */
  private async checkRedisHealth(): Promise<any> {
    try {
      const start = Date.now();
      await this.redis.ping();
      const latency = Date.now() - start;
      
      if (latency > 100) {
        this.createAlert('redis-latency', 'medium',
          `Redis latency high: ${latency}ms`, { latency });
      }
      
      // Check memory usage
      const info = await this.redis.info('memory');
      const memoryMatch = info.match(/used_memory:(\d+)/);
      const memoryUsed = memoryMatch ? parseInt(memoryMatch[1]) : 0;
      const memoryUsedMB = memoryUsed / (1024 * 1024);
      
      if (memoryUsedMB > 512) {
        this.createAlert('redis-memory', 'medium',
          `Redis memory usage high: ${memoryUsedMB.toFixed(0)}MB`, { memoryUsedMB });
      }
      
    } catch (error: any) {
      this.createAlert('redis-connection', 'critical',
        'Redis connection failed', { error: error instanceof Error ? error.message : 'Unknown error' });
    }
  }

  /**
   * Assess individual pool health
   */
  private assessPoolHealth(poolName: string, stats: any): HealthMetrics {
    const healthScore = this.calculateHealthScore(stats);
    
    return {
      processId: `pool-${poolName}`,
      poolName,
      type: this.getPoolType(poolName),
      status: this.getHealthStatus(healthScore),
      uptime: Date.now() - (stats.startTime || Date.now()),
      memoryUsageMB: stats.memoryUsage?.node || 0,
      cpuUsagePercent: 0, // Would be calculated from system metrics
      responseTime: stats.averageResponseTime || 0,
      errorRate: this.calculateErrorRate(poolName),
      requestCount: stats.totalRequests || 0,
      lastHealthCheck: Date.now(),
      alerts: this.getPoolAlerts(poolName)
    };
  }

  /**
   * Calculate health score (0-100)
   */
  private calculateHealthScore(stats: any): number {
    let score = 100;
    
    // Response time impact
    if (stats.averageResponseTime > this.thresholds.responseTime.warning) {
      score -= 20;
    }
    if (stats.averageResponseTime > this.thresholds.responseTime.critical) {
      score -= 30;
    }
    
    // Worker availability impact
    const utilization = stats.busyWorkers / stats.totalWorkers;
    if (utilization > 0.9) score -= 15;
    if (utilization > 0.95) score -= 25;
    
    // Memory usage impact
    if (stats.memoryUsage?.node > this.thresholds.memory.warning) {
      score -= 15;
    }
    if (stats.memoryUsage?.node > this.thresholds.memory.critical) {
      score -= 30;
    }
    
    return Math.max(0, Math.min(100, score));
  }

  /**
   * Get health status from score
   */
  private getHealthStatus(score: number): 'healthy' | 'degraded' | 'unhealthy' | 'critical' {
    if (score >= 90) return 'healthy';
    if (score >= 70) return 'degraded';
    if (score >= 50) return 'unhealthy';
    return 'critical';
  }

  /**
   * Get GPU health status
   */
  private getGPUHealthStatus(memoryUsage: number, temperature: number): 'healthy' | 'degraded' | 'unhealthy' | 'critical' {
    if (temperature > this.thresholds.gpu.temperature.critical || 
        memoryUsage > this.thresholds.gpu.memory.critical) {
      return 'critical';
    }
    if (temperature > this.thresholds.gpu.temperature.warning || 
        memoryUsage > this.thresholds.gpu.memory.warning) {
      return 'degraded';
    }
    return 'healthy';
  }

  /**
   * Create health alert
   */
  private createAlert(id: string, severity: 'low' | 'medium' | 'high' | 'critical', 
                     message: string, metadata?: any): void {
    const alert: HealthAlert = {
      id,
      severity,
      message,
      timestamp: Date.now(),
      resolved: false,
      metadata
    };
    
    this.alerts.set(id, alert);
    this.emit('alert:created', alert);
    
    // Store in Redis for persistence
    this.redis.hset(`legal_ai:alerts:${id}`, {
      severity,
      message,
      timestamp: alert.timestamp,
      resolved: 'false',
      metadata: JSON.stringify(metadata || {})
    });
  }

  /**
   * Process and manage alerts
   */
  private async processAlerts(): Promise<any> {
    const activeAlerts = Array.from(this.alerts.values()).filter(alert => !alert.resolved);
    
    // Auto-resolve alerts older than 5 minutes if condition no longer exists
    for (const alert of activeAlerts) {
      if (Date.now() - alert.timestamp > 300000) { // 5 minutes
        const shouldResolve = await this.shouldAutoResolveAlert(alert);
        if (shouldResolve) {
          this.resolveAlert(alert.id);
        }
      }
    }
    
    // Send critical alerts immediately
    const criticalAlerts = activeAlerts.filter(alert => alert.severity === 'critical');
    if (criticalAlerts.length > 0) {
      this.emit('alerts:critical', criticalAlerts);
    }
  }

  /**
   * Check if alert should be auto-resolved
   */
  private async shouldAutoResolveAlert(alert: HealthAlert): Promise<boolean> {
    try {
      switch (alert.id) {
        case 'gpu-memory-warning':
        case 'gpu-memory-critical':
          const gpuStats = await legalAIGPUManager.getGPUStats();
          const memoryUsage = gpuStats.usedMemoryMB / gpuStats.totalMemoryMB;
          return memoryUsage < this.thresholds.gpu.memory.warning;
          
        case 'gpu-temp-warning':
        case 'gpu-temp-critical':
          const tempStats = await legalAIGPUManager.getGPUStats();
          return tempStats.temperature < this.thresholds.gpu.temperature.warning;
          
        case 'redis-connection':
          try {
            await this.redis.ping();
            return true;
          } catch {
            return false;
          }
          
        default:
          return false;
      }
    } catch {
      return false;
    }
  }

  /**
   * Resolve alert
   */
  resolveAlert(alertId: string): void {
    const alert = this.alerts.get(alertId);
    if (alert) {
      alert.resolved = true;
      this.emit('alert:resolved', alert);
      
      // Update in Redis
      this.redis.hset(`legal_ai:alerts:${alertId}`, 'resolved', 'true');
    }
  }

  /**
   * Get comprehensive system health summary
   */
  async getSystemHealthSummary(): Promise<SystemHealthSummary> {
    const processHealth: Record<string, HealthMetrics> = {};
    const poolStats = legalAIProcessPool.getStats();
    
    // Collect process health
    for (const [poolName, stats] of Object.entries(poolStats)) {
      processHealth[poolName] = this.assessPoolHealth(poolName, stats);
    }
    
    // GPU health
    const gpuStats = await legalAIGPUManager.getGPUStats();
    const gpuMemoryUsage = gpuStats.usedMemoryMB / gpuStats.totalMemoryMB;
    const gpuHealth = {
      memoryUsage: gpuMemoryUsage,
      temperature: gpuStats.temperature,
      utilization: gpuStats.utilization,
      status: this.getGPUHealthStatus(gpuMemoryUsage, gpuStats.temperature)
    };
    
    // Overall system health
    const processStatuses = Object.values(processHealth).map(h => h.status);
    const overall = this.calculateOverallHealth([...processStatuses, gpuHealth.status]);
    
    // Active alerts
    const activeAlerts = Array.from(this.alerts.values()).filter(alert => !alert.resolved);
    
    // Generate recommendations
    const recommendations = this.generateRecommendations(processHealth, gpuHealth, activeAlerts);
    
    return {
      overall,
      processHealth,
      gpuHealth,
      poolStats: legalAIWorkerClient.getPoolStats(),
      activeAlerts,
      recommendations
    };
  }

  /**
   * Calculate overall system health
   */
  private calculateOverallHealth(statuses: string[]): 'healthy' | 'degraded' | 'unhealthy' | 'critical' {
    if (statuses.includes('critical')) return 'critical';
    if (statuses.includes('unhealthy')) return 'unhealthy';
    if (statuses.includes('degraded')) return 'degraded';
    return 'healthy';
  }

  /**
   * Generate system recommendations
   */
  private generateRecommendations(
    processHealth: Record<string, HealthMetrics>,
    gpuHealth: any,
    activeAlerts: HealthAlert[]
  ): string[] {
    const recommendations: string[] = [];
    
    // GPU recommendations
    if (gpuHealth.memoryUsage > 0.8) {
      recommendations.push('Consider reducing GPU model batch sizes or evicting low-priority models');
    }
    if (gpuHealth.temperature > 75) {
      recommendations.push('GPU temperature is elevated - check cooling and workload distribution');
    }
    
    // Process recommendations
    for (const [poolName, health] of Object.entries(processHealth)) {
      if (health.status === 'critical') {
        recommendations.push(`${poolName} pool requires immediate attention - consider restarting workers`);
      }
      if (health.responseTime > 5000) {
        recommendations.push(`${poolName} response times are slow - consider scaling up workers`);
      }
    }
    
    // Alert-based recommendations
    const criticalAlerts = activeAlerts.filter(a => a.severity === 'critical').length;
    if (criticalAlerts > 3) {
      recommendations.push('Multiple critical alerts active - consider system maintenance');
    }
    
    return recommendations;
  }

  /**
   * Utility methods
   */
  private getPoolType(poolName: string): 'node' | 'go' | 'python-cuda' {
    if (poolName.includes('rag')) return 'go';
    if (poolName.includes('vector') || poolName.includes('entity')) return 'python-cuda';
    return 'node';
  }

  private calculateErrorRate(poolName: string): number {
    // This would be calculated from stored metrics
    return 0;
  }

  private getPoolAlerts(poolName: string): HealthAlert[] {
    return Array.from(this.alerts.values()).filter(
      alert => alert.id.includes(poolName) && !alert.resolved
    );
  }

  private initializeWorkerHealth(workerId: string, poolName: string): void {
    // Initialize health tracking for new worker
  }

  private removeWorkerHealth(workerId: string): void {
    // Clean up health tracking for terminated worker
  }

  private updateWorkerMetrics(requestId: string, metrics: any): void {
    // Update worker performance metrics
  }

  /**
   * Shutdown monitoring
   */
  async shutdown(): Promise<any> {
    clearInterval(this.monitoringInterval);
    clearInterval(this.alertInterval);
    await this.redis.quit();
    this.emit('monitor:shutdown');
  }
}

// Global health monitor for legal AI platform
export const legalAIHealthMonitor = new ProcessHealthMonitor();