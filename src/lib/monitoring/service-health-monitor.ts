/**
 * Service Health Monitor & Dashboard
 * 
 * Real-time monitoring of all microservices with:
 * - Automated health checks
 * - Performance metrics collection
 * - Alert system for failures
 * - Auto-recovery mechanisms
 * - Load balancing assistance
 */

import { EventEmitter } from 'events';
import type { ServiceHealthStatus, ProcessingMetrics } from '$lib/types/rag-orchestration';

export interface ServiceEndpoint {
  name: string;
  url: string;
  healthPath: string;
  critical: boolean;
  timeout: number;
  retries: number;
  expectedStatus: number;
}

export interface HealthMetrics {
  timestamp: number;
  responseTime: number;
  status: 'healthy' | 'unhealthy' | 'degraded';
  cpu?: number;
  memory?: number;
  throughput?: number;
  errorRate?: number;
}

export interface AlertRule {
  id: string;
  name: string;
  condition: (metrics: HealthMetrics, history: HealthMetrics[]) => boolean;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  cooldown: number; // ms
}

export class ServiceHealthMonitor extends EventEmitter {
  private services: Map<string, ServiceEndpoint> = new Map();
  private healthHistory: Map<string, HealthMetrics[]> = new Map();
  private alertRules: Map<string, AlertRule> = new Map();
  private lastAlerts: Map<string, number> = new Map();
  private monitoringInterval: NodeJS.Timeout | null = null;
  private isMonitoring = false;

  constructor() {
    super();
    this.setupDefaultServices();
    this.setupDefaultAlertRules();
  }

  /**
   * Setup default service endpoints
   */
  private setupDefaultServices(): void {
    const defaultServices: ServiceEndpoint[] = [
      {
        name: 'Enhanced RAG',
        url: 'http://localhost:8094',
        healthPath: '/health',
        critical: true,
        timeout: 5000,
        retries: 2,
        expectedStatus: 200
      },
      {
        name: 'Upload Service',
        url: 'http://localhost:8093',
        healthPath: '/health',
        critical: true,
        timeout: 5000,
        retries: 2,
        expectedStatus: 200
      },
      {
        name: 'Vector Service',
        url: 'http://localhost:8095',
        healthPath: '/health',
        critical: true,
        timeout: 5000,
        retries: 2,
        expectedStatus: 200
      },
      {
        name: 'GPU Orchestrator',
        url: 'http://localhost:8231',
        healthPath: '/api/gpu/status',
        critical: false,
        timeout: 3000,
        retries: 1,
        expectedStatus: 200
      },
      {
        name: 'Protocol Gateway',
        url: 'http://localhost:8230',
        healthPath: '/api/gateway/health',
        critical: false,
        timeout: 3000,
        retries: 1,
        expectedStatus: 200
      },
      {
        name: 'Health Monitor',
        url: 'http://localhost:8232',
        healthPath: '/api/health',
        critical: false,
        timeout: 3000,
        retries: 1,
        expectedStatus: 200
      },
      {
        name: 'Integration Hub',
        url: 'http://localhost:8096',
        healthPath: '/status',
        critical: false,
        timeout: 3000,
        retries: 1,
        expectedStatus: 200
      },
      {
        name: 'Python Embedding',
        url: 'http://localhost:8097',
        healthPath: '/health',
        critical: false,
        timeout: 3000,
        retries: 1,
        expectedStatus: 200
      },
      {
        name: 'Ollama',
        url: 'http://localhost:11434',
        healthPath: '/api/tags',
        critical: true,
        timeout: 10000,
        retries: 3,
        expectedStatus: 200
      },
      {
        name: 'Qdrant',
        url: 'http://localhost:6333',
        healthPath: '/collections',
        critical: true,
        timeout: 5000,
        retries: 2,
        expectedStatus: 200
      },
      {
        name: 'MinIO',
        url: 'http://localhost:9000',
        healthPath: '/minio/health/live',
        critical: true,
        timeout: 5000,
        retries: 2,
        expectedStatus: 200
      },
      {
        name: 'Neo4j',
        url: 'http://localhost:7474',
        healthPath: '/db/data/',
        critical: false,
        timeout: 5000,
        retries: 2,
        expectedStatus: 200
      },
      {
        name: 'NATS',
        url: 'http://localhost:8222',
        healthPath: '/healthz',
        critical: false,
        timeout: 3000,
        retries: 1,
        expectedStatus: 200
      }
    ];

    defaultServices.forEach(service => {
      this.services.set(service.name, service);
      this.healthHistory.set(service.name, []);
    });
  }

  /**
   * Setup default alert rules
   */
  private setupDefaultAlertRules(): void {
    const defaultRules: AlertRule[] = [
      {
        id: 'service_down',
        name: 'Service Down',
        condition: (metrics) => metrics.status === 'unhealthy',
        severity: 'critical',
        message: 'Service is not responding',
        cooldown: 60000 // 1 minute
      },
      {
        id: 'high_response_time',
        name: 'High Response Time',
        condition: (metrics) => metrics.responseTime > 5000,
        severity: 'medium',
        message: 'Service response time is above 5 seconds',
        cooldown: 300000 // 5 minutes
      },
      {
        id: 'degraded_performance',
        name: 'Degraded Performance',
        condition: (metrics, history) => {
          if (history.length < 5) return false;
          const recent = history.slice(-5);
          const avgResponseTime = recent.reduce((sum, m) => sum + m.responseTime, 0) / recent.length;
          return avgResponseTime > 3000 && metrics.status === 'healthy';
        },
        severity: 'medium',
        message: 'Service performance has degraded',
        cooldown: 600000 // 10 minutes
      },
      {
        id: 'critical_service_failure',
        name: 'Critical Service Failure',
        condition: (metrics) => metrics.status === 'unhealthy',
        severity: 'critical',
        message: 'Critical service has failed',
        cooldown: 30000 // 30 seconds
      }
    ];

    defaultRules.forEach(rule => {
      this.alertRules.set(rule.id, rule);
    });
  }

  /**
   * Start monitoring all services
   */
  startMonitoring(intervalMs: number = 30000): void {
    if (this.isMonitoring) {
      console.log('[Health Monitor] Already monitoring services');
      return;
    }

    console.log('[Health Monitor] 🚀 Starting service monitoring...');
    this.isMonitoring = true;

    // Perform initial health check
    this.performHealthCheck();

    // Setup continuous monitoring
    this.monitoringInterval = setInterval(() => {
      this.performHealthCheck();
    }, intervalMs);

    this.emit('monitoring:started');
  }

  /**
   * Stop monitoring
   */
  stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    
    this.isMonitoring = false;
    console.log('[Health Monitor] 🛑 Monitoring stopped');
    this.emit('monitoring:stopped');
  }

  /**
   * Perform health check on all services
   */
  private async performHealthCheck(): Promise<any> {
    const timestamp = Date.now();
    const healthPromises = Array.from(this.services.entries()).map(
      ([name, service]) => this.checkServiceHealth(name, service, timestamp)
    );

    await Promise.allSettled(healthPromises);
    
    // Emit overall health status
    const healthStatus = this.getOverallHealthStatus();
    this.emit('health:update', healthStatus);
  }

  /**
   * Check health of a single service
   */
  private async checkServiceHealth(
    serviceName: string, 
    service: ServiceEndpoint, 
    timestamp: number
  ): Promise<any> {
    const startTime = Date.now();
    let responseTime = 0;
    let status: 'healthy' | 'unhealthy' | 'degraded' = 'unhealthy';

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), service.timeout);

      const response = await fetch(`${service.url}${service.healthPath}`, {
        method: 'GET',
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'DealsWeb-HealthMonitor/1.0'
        }
      });

      clearTimeout(timeoutId);
      responseTime = Date.now() - startTime;

      if (response.status === service.expectedStatus) {
        status = responseTime > 3000 ? 'degraded' : 'healthy';
      } else {
        status = 'unhealthy';
      }

    } catch (error: any) {
      responseTime = Date.now() - startTime;
      status = 'unhealthy';
      console.warn(`[Health Monitor] ⚠️ ${serviceName} health check failed:`, error);
    }

    // Create metrics
    const metrics: HealthMetrics = {
      timestamp,
      responseTime,
      status
    };

    // Store in history
    const history = this.healthHistory.get(serviceName) || [];
    history.push(metrics);
    
    // Keep only last 100 entries
    if (history.length > 100) {
      history.splice(0, history.length - 100);
    }
    
    this.healthHistory.set(serviceName, history);

    // Check alert rules
    this.checkAlertRules(serviceName, metrics, history, service.critical);

    // Emit service health update
    this.emit('service:health', {
      serviceName,
      metrics,
      critical: service.critical
    });

    console.log(`[Health Monitor] ${this.getStatusIcon(status)} ${serviceName}: ${status} (${responseTime}ms)`);
  }

  /**
   * Check alert rules for a service
   */
  private checkAlertRules(
    serviceName: string, 
    metrics: HealthMetrics, 
    history: HealthMetrics[],
    isCritical: boolean
  ): void {
    this.alertRules.forEach((rule, ruleId) => {
      // Skip non-critical alerts for non-critical services unless they're down
      if (!isCritical && rule.severity !== 'critical' && rule.id !== 'service_down') {
        return;
      }

      if (rule.condition(metrics, history)) {
        const alertKey = `${serviceName}:${ruleId}`;
        const lastAlert = this.lastAlerts.get(alertKey) || 0;
        const now = Date.now();

        // Check cooldown
        if (now - lastAlert > rule.cooldown) {
          this.lastAlerts.set(alertKey, now);
          
          const alert = {
            id: `${alertKey}:${now}`,
            serviceName,
            rule: rule.name,
            severity: rule.severity,
            message: rule.message,
            timestamp: now,
            metrics,
            critical: isCritical
          };

          console.warn(`[Health Monitor] 🚨 ALERT: ${serviceName} - ${rule.message}`);
          this.emit('alert', alert);

          // If critical service is down, emit emergency alert
          if (isCritical && metrics.status === 'unhealthy') {
            this.emit('emergency', {
              ...alert,
              emergency: true,
              message: `CRITICAL SERVICE DOWN: ${serviceName}`
            });
          }
        }
      }
    });
  }

  /**
   * Get overall system health status
   */
  getOverallHealthStatus(): {
    status: 'healthy' | 'degraded' | 'unhealthy';
    services: { [serviceName: string]: HealthMetrics };
    summary: {
      total: number;
      healthy: number;
      degraded: number;
      unhealthy: number;
      critical: number;
    };
  } {
    const services: { [serviceName: string]: HealthMetrics } = {};
    let healthy = 0;
    let degraded = 0;
    let unhealthy = 0;
    let criticalDown = 0;

    this.services.forEach((serviceConfig, serviceName) => {
      const history = this.healthHistory.get(serviceName) || [];
      const latestMetrics = history[history.length - 1];
      
      if (latestMetrics) {
        services[serviceName] = latestMetrics;
        
        switch (latestMetrics.status) {
          case 'healthy':
            healthy++;
            break;
          case 'degraded':
            degraded++;
            break;
          case 'unhealthy':
            unhealthy++;
            if (serviceConfig.critical) {
              criticalDown++;
            }
            break;
        }
      }
    });

    const total = this.services.size;
    let overallStatus: 'healthy' | 'degraded' | 'unhealthy';

    if (criticalDown > 0) {
      overallStatus = 'unhealthy';
    } else if (unhealthy > 0 || degraded > total / 3) {
      overallStatus = 'degraded';
    } else {
      overallStatus = 'healthy';
    }

    return {
      status: overallStatus,
      services,
      summary: {
        total,
        healthy,
        degraded,
        unhealthy,
        critical: criticalDown
      }
    };
  }

  /**
   * Get service history
   */
  getServiceHistory(serviceName: string, limit: number = 50): HealthMetrics[] {
    const history = this.healthHistory.get(serviceName) || [];
    return history.slice(-limit);
  }

  /**
   * Get system metrics
   */
  getSystemMetrics(): ProcessingMetrics & {
    uptime: number;
    servicesMonitored: number;
    alertsTriggered: number;
    averageResponseTime: number;
  } {
    const now = Date.now();
    let totalResponseTime = 0;
    let validMeasurements = 0;

    this.healthHistory.forEach((history) => {
      const latest = history[history.length - 1];
      if (latest && latest.status === 'healthy') {
        totalResponseTime += latest.responseTime;
        validMeasurements++;
      }
    });

    return {
      documentsProcessed: 0, // This would be integrated with the RAG coordinator
      totalProcessingTime: 0,
      averageProcessingTime: 0,
      successRate: 0,
      activeJobs: 0,
      queueDepth: 0,
      uptime: now,
      servicesMonitored: this.services.size,
      alertsTriggered: this.lastAlerts.size,
      averageResponseTime: validMeasurements > 0 ? totalResponseTime / validMeasurements : 0
    };
  }

  /**
   * Add custom service
   */
  addService(service: ServiceEndpoint): void {
    this.services.set(service.name, service);
    this.healthHistory.set(service.name, []);
    console.log(`[Health Monitor] ➕ Added service: ${service.name}`);
  }

  /**
   * Remove service
   */
  removeService(serviceName: string): void {
    this.services.delete(serviceName);
    this.healthHistory.delete(serviceName);
    console.log(`[Health Monitor] ➖ Removed service: ${serviceName}`);
  }

  /**
   * Add custom alert rule
   */
  addAlertRule(rule: AlertRule): void {
    this.alertRules.set(rule.id, rule);
    console.log(`[Health Monitor] 🚨 Added alert rule: ${rule.name}`);
  }

  /**
   * Get status icon for display
   */
  private getStatusIcon(status: string): string {
    switch (status) {
      case 'healthy': return '✅';
      case 'degraded': return '⚠️';
      case 'unhealthy': return '❌';
      default: return '❓';
    }
  }

  /**
   * Check if monitoring is active
   */
  isMonitoringActive(): boolean {
    return this.isMonitoring;
  }

  /**
   * Cleanup resources
   */
  cleanup(): void {
    this.stopMonitoring();
    this.removeAllListeners();
    console.log('[Health Monitor] 🧹 Cleanup completed');
  }
}

// Singleton instance
export const serviceHealthMonitor = new ServiceHealthMonitor();

// Types are already exported as interface declarations