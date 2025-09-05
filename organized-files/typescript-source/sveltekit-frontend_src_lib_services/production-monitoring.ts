// Phase 4: Production Deployment & Monitoring
// Comprehensive monitoring, alerting, and performance optimization

import { writable, derived, type Writable } from 'svelte/store'
import { cacheFirstService } from './cache-first-architecture.js';
import { drizzleCacheWarming } from './drizzle-cache-warming.js';
import { aiGPUService } from './ai-gpu-optimization.js';
import crypto from "crypto";

// ===== MONITORING CONFIGURATION =====

interface MonitoringConfig {
  enabled: boolean;
  reportingInterval: number;
  alertThresholds: {
    responseTime: number;
    errorRate: number;
    cacheHitRate: number;
    memoryUsage: number;
    gpuUtilization: number;
  };
  endpoints: {
    metrics: string;
    alerts: string;
    health: string;
  };
  retention: {
    metrics: number; // days
    logs: number; // days
    traces: number; // days
  };
}

const defaultConfig: MonitoringConfig = {
  enabled: true,
  reportingInterval: 30000, // 30 seconds
  alertThresholds: {
    responseTime: 2000, // 2 seconds
    errorRate: 0.05, // 5%
    cacheHitRate: 0.8, // 80%
    memoryUsage: 0.85, // 85%
    gpuUtilization: 0.95 // 95%
  },
  endpoints: {
    metrics: '/api/v1/metrics',
    alerts: '/api/v1/alerts',
    health: '/api/v1/health'
  },
  retention: {
    metrics: 30,
    logs: 7,
    traces: 3
  }
};

// ===== PERFORMANCE METRICS COLLECTION =====

interface PerformanceMetrics {
  timestamp: number;
  system: {
    memoryUsage: number;
    cpuUsage: number;
    diskSpace: number;
    networkLatency: number;
  };
  application: {
    responseTime: {
      avg: number;
      p50: number;
      p95: number;
      p99: number;
    };
    requestCount: number;
    errorRate: number;
    activeUsers: number;
    sessionDuration: number;
  };
  cache: {
    hitRate: number;
    missRate: number;
    evictionRate: number;
    size: number;
    warmingEffectiveness: number;
  };
  ai: {
    queueSize: number;
    processingTime: number;
    gpuUtilization: number;
    modelAccuracy: number;
    throughput: number;
  };
  database: {
    connectionPool: number;
    queryTime: number;
    lockWaitTime: number;
    transactionRate: number;
  };
}

export class ProductionMonitoringService {
  private config: MonitoringConfig = defaultConfig;
  private metricsBuffer: PerformanceMetrics[] = [];
  private alerts: Alert[] = [];
  private healthStatus: HealthStatus = 'healthy';
  
  // Performance tracking
  private requestTimes: number[] = [];
  private errorCount = 0;
  private totalRequests = 0;
  private activeConnections = new Set<string>();
  
  // Reactive stores
  public metrics = writable<PerformanceMetrics | null>(null);
  public healthScore = writable(100);
  public alertCount = writable(0);
  public systemStatus = writable<SystemStatus>({
    overall: 'healthy',
    components: {
      cache: 'healthy',
      database: 'healthy',
      ai: 'healthy',
      gpu: 'healthy'
    },
    uptime: 0,
    lastUpdate: new Date()
  });

  constructor(config?: Partial<MonitoringConfig>) {
    if (config) {
      this.config = { ...defaultConfig, ...config };
    }
    
    if (this.config.enabled) {
      this.initializeMonitoring();
    }
  }

  // ===== MONITORING INITIALIZATION =====

  private initializeMonitoring(): void {
    console.log('Initializing production monitoring...');
    
    // Start metrics collection
    this.startMetricsCollection();
    
    // Initialize performance observers
    this.initializePerformanceObservers();
    
    // Start health checks
    this.startHealthChecks();
    
    // Initialize alerting
    this.initializeAlerting();
    
    console.log('Production monitoring initialized successfully');
  }

  private startMetricsCollection(): void {
    setInterval(async () => {
      const metrics = await this.collectMetrics();
      this.metricsBuffer.push(metrics);
      this.metrics.set(metrics);
      
      // Keep buffer size manageable
      if (this.metricsBuffer.length > 1000) {
        this.metricsBuffer = this.metricsBuffer.slice(-500);
      }
      
      // Check for alerts
      await this.checkAlertConditions(metrics);
      
    }, this.config.reportingInterval);
  }

  // ===== METRICS COLLECTION =====

  private async collectMetrics(): Promise<PerformanceMetrics> {
    const timestamp = Date.now();
    
    return {
      timestamp,
      system: await this.collectSystemMetrics(),
      application: await this.collectApplicationMetrics(),
      cache: await this.collectCacheMetrics(),
      ai: await this.collectAIMetrics(),
      database: await this.collectDatabaseMetrics()
    };
  }

  private async collectSystemMetrics() {
    // In a real implementation, these would come from actual system APIs
    return {
      memoryUsage: this.getMemoryUsage(),
      cpuUsage: this.getCPUUsage(),
      diskSpace: this.getDiskUsage(),
      networkLatency: await this.measureNetworkLatency()
    };
  }

  private async collectApplicationMetrics() {
    const now = Date.now();
    const recentRequests = this.requestTimes.filter(time => now - time < 60000); // Last minute
    
    return {
      responseTime: this.calculateResponseTimeStats(recentRequests),
      requestCount: recentRequests.length,
      errorRate: this.totalRequests > 0 ? this.errorCount / this.totalRequests : 0,
      activeUsers: this.activeConnections.size,
      sessionDuration: this.calculateAverageSessionDuration()
    };
  }

  private async collectCacheMetrics() {
    const cacheStats = await cacheFirstService.stats.getValue?.() || {};
    
    return {
      hitRate: cacheStats.cacheHitRate || 0,
      missRate: 1 - (cacheStats.cacheHitRate || 0),
      evictionRate: 0, // Would be calculated from cache evictions
      size: (cacheStats.totalCases || 0) + (cacheStats.totalEvidence || 0),
      warmingEffectiveness: await this.calculateWarmingEffectiveness()
    };
  }

  private async collectAIMetrics() {
    const aiStats = await aiGPUService.queueStats.getValue?.() || {};
    
    return {
      queueSize: aiStats.pending || 0,
      processingTime: aiStats.avgProcessingTime || 0,
      gpuUtilization: aiStats.gpuUtilization || 0,
      modelAccuracy: 0.85, // Would be calculated from AI results
      throughput: aiStats.completed || 0
    };
  }

  private async collectDatabaseMetrics() {
    // These would come from database monitoring APIs
    return {
      connectionPool: 8,
      queryTime: 45,
      lockWaitTime: 5,
      transactionRate: 150
    };
  }

  // ===== PERFORMANCE OBSERVERS =====

  private initializePerformanceObservers(): void {
    // Track page load times
    if (typeof window !== 'undefined' && 'PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'navigation') {
            const navEntry = entry as PerformanceNavigationTiming;
            this.recordRequestTime(navEntry.loadEventEnd - navEntry.fetchStart);
          }
        }
      });
      
      observer.observe({ entryTypes: ['navigation'] });
    }

    // Monitor fetch requests
    if (typeof window !== 'undefined') {
      const originalFetch = window.fetch;
      window.fetch = async (...args) => {
        const start = performance.now();
        this.totalRequests++;
        
        try {
          const response = await originalFetch(...args);
          const duration = performance.now() - start;
          this.recordRequestTime(duration);
          
          if (!response.ok) {
            this.errorCount++;
          }
          
          return response;
        } catch (error) {
          this.errorCount++;
          const duration = performance.now() - start;
          this.recordRequestTime(duration);
          throw error;
        }
      };
    }
  }

  // ===== HEALTH CHECKS =====

  private startHealthChecks(): void {
    setInterval(async () => {
      const healthStatus = await this.performHealthCheck();
      this.updateSystemStatus(healthStatus);
    }, 60000); // Check every minute
  }

  private async performHealthCheck(): Promise<ComponentHealth> {
    const checks = await Promise.allSettled([
      this.checkCacheHealth(),
      this.checkDatabaseHealth(),
      this.checkAIHealth(),
      this.checkGPUHealth()
    ]);

    return {
      cache: this.getCheckResult(checks[0]),
      database: this.getCheckResult(checks[1]),
      ai: this.getCheckResult(checks[2]),
      gpu: this.getCheckResult(checks[3])
    };
  }

  private async checkCacheHealth(): Promise<HealthStatus> {
    try {
      const stats = await cacheFirstService.stats.getValue?.();
      return stats?.cacheHitRate > 0.7 ? 'healthy' : 'warning';
    } catch {
      return 'critical';
    }
  }

  private async checkDatabaseHealth(): Promise<HealthStatus> {
    try {
      const response = await fetch('/api/health/database', { timeout: 5000 } as any);
      return response.ok ? 'healthy' : 'warning';
    } catch {
      return 'critical';
    }
  }

  private async checkAIHealth(): Promise<HealthStatus> {
    try {
      const stats = await aiGPUService.queueStats.getValue?.();
      return (stats?.pending || 0) < 10 ? 'healthy' : 'warning';
    } catch {
      return 'critical';
    }
  }

  private async checkGPUHealth(): Promise<HealthStatus> {
    try {
      const stats = await aiGPUService.queueStats.getValue?.();
      const utilization = stats?.gpuUtilization || 0;
      return utilization < 90 ? 'healthy' : 'warning';
    } catch {
      return 'critical';
    }
  }

  // ===== ALERTING SYSTEM =====

  private initializeAlerting(): void {
    console.log('Alert system initialized');
  }

  private async checkAlertConditions(metrics: PerformanceMetrics): Promise<void> {
    const alerts: Alert[] = [];
    
    // Response time alerts
    if (metrics.application.responseTime.avg > this.config.alertThresholds.responseTime) {
      alerts.push({
        id: crypto.randomUUID(),
        type: 'performance',
        severity: 'warning',
        message: `High response time: ${Math.round(metrics.application.responseTime.avg)}ms`,
        timestamp: new Date(),
        component: 'application'
      });
    }
    
    // Error rate alerts
    if (metrics.application.errorRate > this.config.alertThresholds.errorRate) {
      alerts.push({
        id: crypto.randomUUID(),
        type: 'error',
        severity: 'critical',
        message: `High error rate: ${Math.round(metrics.application.errorRate * 100)}%`,
        timestamp: new Date(),
        component: 'application'
      });
    }
    
    // Cache hit rate alerts
    if (metrics.cache.hitRate < this.config.alertThresholds.cacheHitRate) {
      alerts.push({
        id: crypto.randomUUID(),
        type: 'cache',
        severity: 'warning',
        message: `Low cache hit rate: ${Math.round(metrics.cache.hitRate * 100)}%`,
        timestamp: new Date(),
        component: 'cache'
      });
    }
    
    // GPU utilization alerts
    if (metrics.ai.gpuUtilization > this.config.alertThresholds.gpuUtilization) {
      alerts.push({
        id: crypto.randomUUID(),
        type: 'resource',
        severity: 'critical',
        message: `High GPU utilization: ${Math.round(metrics.ai.gpuUtilization)}%`,
        timestamp: new Date(),
        component: 'gpu'
      });
    }
    
    // Process new alerts
    for (const alert of alerts) {
      await this.processAlert(alert);
    }
    
    this.alertCount.set(this.alerts.length);
  }

  private async processAlert(alert: Alert): Promise<void> {
    this.alerts.push(alert);
    
    console.warn(`ALERT [${alert.severity}]: ${alert.message}`);
    
    // Send to external monitoring if configured
    if (this.config.endpoints.alerts) {
      try {
        await fetch(this.config.endpoints.alerts, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(alert)
        });
      } catch (error) {
        console.error('Failed to send alert:', error);
      }
    }
    
    // Auto-remediation for certain alert types
    await this.attemptAutoRemediation(alert);
  }

  // ===== AUTO-REMEDIATION =====

  private async attemptAutoRemediation(alert: Alert): Promise<void> {
    switch (alert.type) {
      case 'cache':
        console.log('Triggering cache warming for low hit rate');
        await drizzleCacheWarming.triggerWarming('low_hit_rate');
        break;
        
      case 'resource':
        if (alert.component === 'gpu') {
          console.log('Optimizing GPU utilization');
          // Could reduce concurrent AI tasks or switch to CPU
        }
        break;
        
      case 'performance':
        console.log('Triggering performance optimization');
        await this.optimizePerformance();
        break;
    }
  }

  private async optimizePerformance(): Promise<void> {
    // Trigger cache warming
    await drizzleCacheWarming.triggerWarming('performance_optimization');
    
    // Clear old cached data
    await cacheFirstService.cleanupCache();
    
    // Optimize AI queue
    // Could implement queue prioritization or load balancing
  }

  // ===== UTILITY FUNCTIONS =====

  private recordRequestTime(duration: number): void {
    this.requestTimes.push(Date.now());
    // Keep only last 1000 requests
    if (this.requestTimes.length > 1000) {
      this.requestTimes = this.requestTimes.slice(-500);
    }
  }

  private calculateResponseTimeStats(times: number[]): {
    avg: number;
    p50: number;
    p95: number;
    p99: number;
  } {
    if (times.length === 0) {
      return { avg: 0, p50: 0, p95: 0, p99: 0 };
    }
    
    const sorted = times.sort((a, b) => a - b);
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    
    return {
      avg,
      p50: sorted[Math.floor(sorted.length * 0.5)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)]
    };
  }

  private getMemoryUsage(): number {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in performance) {
      const memory = (performance as any).memory;
      return memory.usedJSHeapSize / memory.totalJSHeapSize;
    }
    return Math.random() * 0.8; // Fallback simulation
  }

  private getCPUUsage(): number {
    // Would be implemented with actual CPU monitoring
    return Math.random() * 0.6; // Simulation
  }

  private getDiskUsage(): number {
    // Would be implemented with actual disk monitoring
    return Math.random() * 0.7; // Simulation
  }

  private async measureNetworkLatency(): Promise<number> {
    if (typeof window === 'undefined') return 0;
    
    try {
      const start = performance.now();
      await fetch('/api/health/ping', { cache: 'no-cache' });
      return performance.now() - start;
    } catch {
      return 1000; // Fallback high latency
    }
  }

  private calculateAverageSessionDuration(): number {
    // Would track actual session durations
    return 1800000; // 30 minutes simulation
  }

  private async calculateWarmingEffectiveness(): Promise<number> {
    // Would be calculated based on cache warming impact
    return 0.85; // Simulation
  }

  private getCheckResult(result: PromiseSettledResult<HealthStatus>): HealthStatus {
    return result.status === 'fulfilled' ? result.value : 'critical';
  }

  private updateSystemStatus(componentHealth: ComponentHealth): void {
    const components = componentHealth;
    const healthValues = Object.values(components);
    
    let overall: HealthStatus = 'healthy';
    if (healthValues.includes('critical')) overall = 'critical';
    else if (healthValues.includes('warning')) overall = 'warning';
    
    const healthScore = this.calculateHealthScore(components);
    
    this.systemStatus.set({
      overall,
      components,
      uptime: Date.now() - this.startTime,
      lastUpdate: new Date()
    });
    
    this.healthScore.set(healthScore);
  }

  private startTime = Date.now();

  private calculateHealthScore(components: ComponentHealth): number {
    const weights = { cache: 0.3, database: 0.3, ai: 0.2, gpu: 0.2 };
    const scores = {
      healthy: 100,
      warning: 70,
      critical: 30
    };
    
    let totalScore = 0;
    for (const [component, status] of Object.entries(components)) {
      totalScore += weights[component as keyof ComponentHealth] * scores[status];
    }
    
    return Math.round(totalScore);
  }

  // ===== PUBLIC API =====

  async getMetricsReport(): Promise<MetricsReport> {
    const currentMetrics = this.metricsBuffer[this.metricsBuffer.length - 1];
    const historicalMetrics = this.metricsBuffer.slice(-100); // Last 100 data points
    
    return {
      current: currentMetrics,
      historical: historicalMetrics,
      alerts: this.alerts.slice(-50), // Last 50 alerts
      healthScore: await this.healthScore.getValue?.() || 100,
      uptime: Date.now() - this.startTime,
      recommendations: await this.generateRecommendations()
    };
  }

  private async generateRecommendations(): Promise<string[]> {
    const recommendations: string[] = [];
    const currentMetrics = this.metricsBuffer[this.metricsBuffer.length - 1];
    
    if (!currentMetrics) return recommendations;
    
    if (currentMetrics.cache.hitRate < 0.8) {
      recommendations.push('Consider increasing cache warming frequency');
    }
    
    if (currentMetrics.application.responseTime.avg > 1000) {
      recommendations.push('Optimize database queries and enable query caching');
    }
    
    if (currentMetrics.ai.queueSize > 10) {
      recommendations.push('Consider scaling AI processing capacity');
    }
    
    return recommendations;
  }

  async exportMetrics(format: 'json' | 'csv' = 'json'): Promise<string> {
    const report = await this.getMetricsReport();
    
    if (format === 'csv') {
      return this.convertToCSV(report.historical);
    }
    
    return JSON.stringify(report, null, 2);
  }

  private convertToCSV(metrics: PerformanceMetrics[]): string {
    if (metrics.length === 0) return '';
    
    const headers = [
      'timestamp', 'responseTime', 'errorRate', 'cacheHitRate',
      'memoryUsage', 'cpuUsage', 'gpuUtilization'
    ];
    
    const rows = metrics.map(m => [
      m.timestamp,
      m.application.responseTime.avg,
      m.application.errorRate,
      m.cache.hitRate,
      m.system.memoryUsage,
      m.system.cpuUsage,
      m.ai.gpuUtilization
    ]);
    
    return [headers, ...rows].map(row => row.join(',')).join('\n');
  }
}

// ===== TYPE DEFINITIONS =====

type HealthStatus = 'healthy' | 'warning' | 'critical'

interface ComponentHealth {
  cache: HealthStatus;
  database: HealthStatus;
  ai: HealthStatus;
  gpu: HealthStatus;
}

interface SystemStatus {
  overall: HealthStatus;
  components: ComponentHealth;
  uptime: number;
  lastUpdate: Date;
}

interface Alert {
  id: string;
  type: 'performance' | 'error' | 'cache' | 'resource';
  severity: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: Date;
  component: string;
}

interface MetricsReport {
  current: PerformanceMetrics;
  historical: PerformanceMetrics[];
  alerts: Alert[];
  healthScore: number;
  uptime: number;
  recommendations: string[];
}

// ===== GLOBAL MONITORING SERVICE =====

export const productionMonitoring = new ProductionMonitoringService();