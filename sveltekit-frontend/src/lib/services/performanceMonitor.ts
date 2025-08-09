// @ts-nocheck
/**
 * Performance Monitoring and Metrics Collection Service
 * Tracks AI pipeline performance, queue metrics, cache efficiency, and system health
 */

import { writable, derived, readable } from 'svelte/store';
import { browser } from '$app/environment';
import { bullmqService } from './bullmqService';
import { multiLayerCache } from './multiLayerCache';
import { ollamaService } from './ollamaService';
import { langChainService } from '../ai/langchain-ollama-service';

// Types for metrics
export interface AIMetrics {
  // Processing metrics
  documentsProcessed: number;
  averageProcessingTime: number;
  totalProcessingTime: number;
  
  // Embedding metrics
  embeddingsGenerated: number;
  averageEmbeddingTime: number;
  embeddingDimensions: number;
  
  // Query metrics
  totalQueries: number;
  averageQueryTime: number;
  averageConfidence: number;
  
  // Error metrics
  processingErrors: number;
  embeddingErrors: number;
  queryErrors: number;
  errorRate: number;
  
  // Model usage
  modelUsage: Record<string, number>;
  
  // Performance trends
  processingTimeHistory: Array<{ timestamp: number; value: number }>;
  confidenceHistory: Array<{ timestamp: number; value: number }>;
  throughputHistory: Array<{ timestamp: number; value: number }>;
}

export interface QueueMetrics {
  // Queue status for each queue
  queues: Record<string, {
    waiting: number;
    active: number;
    completed: number;
    failed: number;
    delayed: number;
    throughput: number; // jobs per minute
    averageProcessingTime: number;
  }>;
  
  // Overall queue health
  totalJobs: number;
  totalCompleted: number;
  totalFailed: number;
  overallThroughput: number;
  healthScore: number; // 0-100
  
  // Trends
  throughputHistory: Array<{ timestamp: number; value: number }>;
  failureRateHistory: Array<{ timestamp: number; value: number }>;
}

export interface CacheMetrics {
  // Cache performance
  hitRate: number;
  missRate: number;
  evictionRate: number;
  averageAccessTime: number;
  
  // Memory usage
  totalEntries: number;
  totalSize: number;
  memoryUsage: number; // percentage
  
  // Layer performance
  layerStats: {
    memory: { entries: number; size: number; hitRate: number };
    persistent: { entries: number; size: number; hitRate: number };
    search: { entries: number; queries: number };
  };
  
  // Trends
  hitRateHistory: Array<{ timestamp: number; value: number }>;
  memoryUsageHistory: Array<{ timestamp: number; value: number }>;
}

export interface SystemMetrics {
  // Overall system health
  healthScore: number; // 0-100
  uptime: number; // milliseconds
  
  // Component status
  components: {
    ollama: { status: 'healthy' | 'degraded' | 'unhealthy'; responseTime: number };
    database: { status: 'healthy' | 'degraded' | 'unhealthy'; responseTime: number };
    cache: { status: 'healthy' | 'degraded' | 'unhealthy'; responseTime: number };
    queues: { status: 'healthy' | 'degraded' | 'unhealthy'; responseTime: number };
  };
  
  // Resource usage
  memory: { used: number; total: number; percentage: number };
  cpu: { usage: number };
  
  // Network metrics
  networkLatency: number;
  networkThroughput: number;
  
  // Performance indicators
  responseTime: number;
  throughput: number;
  errorRate: number;
  
  // Alerts
  activeAlerts: Array<{
    id: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    message: string;
    timestamp: number;
    component: string;
  }>;
}

export interface PerformanceSnapshot {
  timestamp: number;
  ai: AIMetrics;
  queues: QueueMetrics;
  cache: CacheMetrics;
  system: SystemMetrics;
}

// Performance Monitor Class
class PerformanceMonitor {
  private metricsHistory: PerformanceSnapshot[] = [];
  private startTime: number = Date.now();
  private collectionInterval: NodeJS.Timeout | null = null;
  private alertThresholds = {
    errorRate: 0.05, // 5%
    responseTime: 5000, // 5 seconds
    memoryUsage: 0.9, // 90%
    queueBacklog: 100,
    cacheHitRate: 0.8 // 80%
  };

  // Svelte stores for reactive metrics
  public readonly currentMetrics = writable<PerformanceSnapshot | null>(null);
  public readonly aiMetrics = writable<AIMetrics | null>(null);
  public readonly queueMetrics = writable<QueueMetrics | null>(null);
  public readonly cacheMetrics = writable<CacheMetrics | null>(null);
  public readonly systemMetrics = writable<SystemMetrics | null>(null);
  public readonly alerts = writable<SystemMetrics['activeAlerts']>([]);
  
  // Derived stores
  public readonly healthScore = derived(
    this.currentMetrics,
    ($metrics) => $metrics?.system.healthScore || 0
  );
  
  public readonly isHealthy = derived(
    this.healthScore,
    ($healthScore) => $healthScore >= 80
  );
  
  public readonly criticalAlerts = derived(
    this.alerts,
    ($alerts) => $alerts.filter(alert: any => alert.severity === 'critical')
  );

  constructor() {
    if (browser) {
      this.startMonitoring();
    }
  }

  // Start performance monitoring
  startMonitoring(intervalMs: number = 30000): void {
    if (this.collectionInterval) {
      clearInterval(this.collectionInterval);
    }

    this.collectionInterval = setInterval(async () => {
      try {
        const snapshot = await this.collectMetrics();
        this.updateStores(snapshot);
        this.checkAlerts(snapshot);
        this.metricsHistory.push(snapshot);
        
        // Keep only last 1000 snapshots (about 8 hours at 30s intervals)
        if (this.metricsHistory.length > 1000) {
          this.metricsHistory = this.metricsHistory.slice(-1000);
        }
      } catch (error) {
        console.error('Failed to collect performance metrics:', error);
      }
    }, intervalMs);

    // Initial collection
    this.collectMetrics().then(snapshot: any => {
      this.updateStores(snapshot);
      this.metricsHistory.push(snapshot);
    });
  }

  // Stop monitoring
  stopMonitoring(): void {
    if (this.collectionInterval) {
      clearInterval(this.collectionInterval);
      this.collectionInterval = null;
    }
  }

  // Collect all metrics
  private async collectMetrics(): Promise<PerformanceSnapshot> {
    const timestamp = Date.now();
    
    const [ai, queues, cache, system] = await Promise.all([
      this.collectAIMetrics(),
      this.collectQueueMetrics(),
      this.collectCacheMetrics(),
      this.collectSystemMetrics()
    ]);

    return { timestamp, ai, queues, cache, system };
  }

  // Collect AI processing metrics
  private async collectAIMetrics(): Promise<AIMetrics> {
    try {
      const stats = await langChainService.getProcessingStats(1); // Last 24 hours
      const ollamaHealth = await ollamaService.checkHealth();
      
      // Get historical data from previous snapshots
      const recentSnapshots = this.metricsHistory.slice(-10);
      const processingTimeHistory = recentSnapshots.map(s: any => ({
        timestamp: s.timestamp,
        value: s.ai.averageProcessingTime
      }));
      
      const confidenceHistory = recentSnapshots.map(s: any => ({
        timestamp: s.timestamp,
        value: s.ai.averageConfidence
      }));
      
      const throughputHistory = recentSnapshots.map(s: any => ({
        timestamp: s.timestamp,
        value: s.ai.documentsProcessed
      }));

      return {
        documentsProcessed: stats.documentsProcessed,
        averageProcessingTime: stats.averageProcessingTime,
        totalProcessingTime: stats.documentsProcessed * stats.averageProcessingTime,
        embeddingsGenerated: stats.documentsProcessed * 5, // Estimated chunks per document
        averageEmbeddingTime: stats.averageProcessingTime * 0.3, // Estimated
        embeddingDimensions: 384, // nomic-embed-text dimensions
        totalQueries: stats.totalQueries,
        averageQueryTime: 2000, // Estimated
        averageConfidence: stats.averageConfidence,
        processingErrors: 0, // Would be tracked from error logs
        embeddingErrors: 0,
        queryErrors: 0,
        errorRate: 0.01, // 1% estimated
        modelUsage: stats.modelUsage,
        processingTimeHistory,
        confidenceHistory,
        throughputHistory
      };
    } catch (error) {
      console.error('Failed to collect AI metrics:', error);
      return this.getDefaultAIMetrics();
    }
  }

  // Collect queue metrics
  private async collectQueueMetrics(): Promise<QueueMetrics> {
    try {
      const stats = await bullmqService.getAllQueueStats();
      const queueNames = Object.keys(stats);
      
      let totalJobs = 0;
      let totalCompleted = 0;
      let totalFailed = 0;
      let totalThroughput = 0;
      
      const queues: QueueMetrics['queues'] = {};
      
      for (const queueName of queueNames) {
        const queueStat = stats[queueName];
        if (queueStat.error) continue;
        
        const completed = queueStat.completed || 0;
        const failed = queueStat.failed || 0;
        const active = queueStat.active || 0;
        const waiting = queueStat.waiting || 0;
        const delayed = queueStat.delayed || 0;
        
        totalJobs += completed + failed + active + waiting + delayed;
        totalCompleted += completed;
        totalFailed += failed;
        
        // Estimate throughput (jobs per minute)
        const throughput = completed / (60 * 24); // Assuming 24 hour window
        totalThroughput += throughput;
        
        queues[queueName] = {
          waiting,
          active,
          completed,
          failed,
          delayed,
          throughput,
          averageProcessingTime: 5000 // Estimated 5 seconds
        };
      }
      
      const failureRate = totalJobs > 0 ? totalFailed / totalJobs : 0;
      const healthScore = Math.max(0, 100 - (failureRate * 100) - (totalThroughput < 1 ? 20 : 0));
      
      // Historical data
      const recentSnapshots = this.metricsHistory.slice(-10);
      const throughputHistory = recentSnapshots.map(s: any => ({
        timestamp: s.timestamp,  
        value: s.queues.overallThroughput
      }));
      
      const failureRateHistory = recentSnapshots.map(s: any => ({
        timestamp: s.timestamp,
        value: s.queues.totalFailed / Math.max(1, s.queues.totalJobs)
      }));

      return {
        queues,
        totalJobs,
        totalCompleted,
        totalFailed,
        overallThroughput: totalThroughput,
        healthScore,
        throughputHistory,
        failureRateHistory
      };
    } catch (error) {
      console.error('Failed to collect queue metrics:', error);
      return this.getDefaultQueueMetrics();
    }
  }

  // Collect cache metrics
  private async collectCacheMetrics(): Promise<CacheMetrics> {
    try {
      const stats = multiLayerCache.getStats();
      
      const hitRate = stats.hitRate;
      const missRate = 1 - hitRate;
      const memoryUsage = stats.totalSize / (50 * 1024 * 1024); // 50MB max
      
      // Historical data
      const recentSnapshots = this.metricsHistory.slice(-10); 
      const hitRateHistory = recentSnapshots.map(s: any => ({
        timestamp: s.timestamp,
        value: s.cache.hitRate
      }));
      
      const memoryUsageHistory = recentSnapshots.map(s: any => ({
        timestamp: s.timestamp,
        value: s.cache.memoryUsage
      }));

      return {
        hitRate,
        missRate,
        evictionRate: stats.evictionCount / Math.max(1, stats.totalEntries),
        averageAccessTime: stats.avgAccessTime,
        totalEntries: stats.totalEntries,
        totalSize: stats.totalSize,
        memoryUsage: Math.min(1, memoryUsage),
        layerStats: stats.layerStats,
        hitRateHistory,
        memoryUsageHistory
      };
    } catch (error) {
      console.error('Failed to collect cache metrics:', error);
      return this.getDefaultCacheMetrics();
    }
  }

  // Collect system metrics
  private async collectSystemMetrics(): Promise<SystemMetrics> {
    try {
      const uptime = Date.now() - this.startTime;
      
      // Component health checks
      const [ollamaHealth, langchainHealth] = await Promise.all([
        this.checkOllamaHealth(),
        this.checkLangChainHealth()
      ]);
      
      const databaseHealth = await this.checkDatabaseHealth();
      const cacheHealth = await this.checkCacheHealth();
      const queueHealth = await this.checkQueueHealth();
      
      const components = {
        ollama: ollamaHealth,
        database: databaseHealth,
        cache: cacheHealth,
        queues: queueHealth
      };
      
      // Calculate overall health score
      const componentScores = Object.values(components).map(c: any => 
        c.status === 'healthy' ? 100 : c.status === 'degraded' ? 50 : 0
      );
      const healthScore = componentScores.reduce((a, b) => a + b, 0) / componentScores.length;
      
      // System resource metrics (simplified for browser environment)
      const memory = this.getMemoryMetrics();
      const cpu = { usage: 0.5 }; // Placeholder
      
      // Network metrics (simplified)
      const networkLatency = ollamaHealth.responseTime;
      const networkThroughput = 1000; // Placeholder
      
      // Performance indicators
      const responseTime = (ollamaHealth.responseTime + databaseHealth.responseTime) / 2;
      const throughput = 100; // Placeholder
      const errorRate = 0.01; // 1% placeholder
      
      return {
        healthScore,
        uptime,
        components,
        memory,
        cpu,
        networkLatency,
        networkThroughput,
        responseTime,
        throughput,
        errorRate,
        activeAlerts: this.generateAlerts(components, { memory, responseTime, errorRate })
      };
    } catch (error) {
      console.error('Failed to collect system metrics:', error);
      return this.getDefaultSystemMetrics();
    }
  }

  // Health check methods
  private async checkOllamaHealth(): Promise<SystemMetrics['components']['ollama']> {
    const startTime = Date.now();
    try {
      const health = await ollamaService.checkHealth();
      const responseTime = Date.now() - startTime;
      
      return {
        status: health.status === 'healthy' ? 'healthy' : 'unhealthy',
        responseTime
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: Date.now() - startTime
      };
    }
  }

  private async checkLangChainHealth(): Promise<{ status: 'healthy' | 'degraded' | 'unhealthy'; responseTime: number }> {
    const startTime = Date.now();
    try {
      const health = await langChainService.healthCheck();
      const responseTime = Date.now() - startTime;
      
      return {
        status: health.status === 'healthy' ? 'healthy' : 'unhealthy',
        responseTime
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: Date.now() - startTime
      };
    }
  }

  private async checkDatabaseHealth(): Promise<SystemMetrics['components']['database']> {
    const startTime = Date.now();
    try {
      // Simple health check - would be replaced with actual DB ping
      await new Promise(resolve: any => setTimeout(resolve, 100));
      return {
        status: 'healthy',
        responseTime: Date.now() - startTime
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: Date.now() - startTime
      };
    }
  }

  private async checkCacheHealth(): Promise<SystemMetrics['components']['cache']> {
    const startTime = Date.now();
    try {
      const stats = multiLayerCache.getStats();
      const responseTime = Date.now() - startTime;
      
      return {
        status: stats.hitRate > 0.5 ? 'healthy' : 'degraded',
        responseTime
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: Date.now() - startTime
      };
    }
  }

  private async checkQueueHealth(): Promise<SystemMetrics['components']['queues']> {
    const startTime = Date.now();
    try {
      await bullmqService.getAllQueueStats();
      return {
        status: 'healthy',
        responseTime: Date.now() - startTime
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: Date.now() - startTime
      };
    }
  }

  // Memory metrics (simplified for browser)
  private getMemoryMetrics(): SystemMetrics['memory'] {
    if (typeof performance !== 'undefined' && (performance as any).memory) {
      const memory = (performance as any).memory;
      return {
        used: memory.usedJSHeapSize,
        total: memory.totalJSHeapSize,
        percentage: memory.usedJSHeapSize / memory.totalJSHeapSize
      };
    }
    
    return {
      used: 0,
      total: 0,
      percentage: 0
    };
  }

  // Generate alerts based on metrics
  private generateAlerts(
    components: SystemMetrics['components'],
    metrics: { memory: SystemMetrics['memory']; responseTime: number; errorRate: number }
  ): SystemMetrics['activeAlerts'] {
    const alerts: SystemMetrics['activeAlerts'] = [];
    const timestamp = Date.now();
    
    // Component health alerts
    Object.entries(components).forEach(([name, component]) => {
      if (component.status === 'unhealthy') {
        alerts.push({
          id: `component-${name}-unhealthy`,
          severity: 'critical',
          message: `${name} component is unhealthy`,
          timestamp,
          component: name
        });
      } else if (component.status === 'degraded') {
        alerts.push({
          id: `component-${name}-degraded`,
          severity: 'medium',
          message: `${name} component is degraded`,
          timestamp,
          component: name
        });
      }
      
      if (component.responseTime > this.alertThresholds.responseTime) {
        alerts.push({
          id: `component-${name}-slow`,
          severity: 'medium',
          message: `${name} component response time is high (${component.responseTime}ms)`,
          timestamp,
          component: name
        });
      }
    });
    
    // Memory alerts
    if (metrics.memory.percentage > this.alertThresholds.memoryUsage) {
      alerts.push({
        id: 'memory-high',
        severity: 'high',
        message: `Memory usage is high (${Math.round(metrics.memory.percentage * 100)}%)`,
        timestamp,
        component: 'system'
      });
    }
    
    // Error rate alerts
    if (metrics.errorRate > this.alertThresholds.errorRate) {
      alerts.push({
        id: 'error-rate-high',
        severity: 'high',
        message: `Error rate is high (${Math.round(metrics.errorRate * 100)}%)`,
        timestamp,
        component: 'system'
      });
    }
    
    return alerts;
  }

  // Check for alerts and update store
  private checkAlerts(snapshot: PerformanceSnapshot): void {
    this.alerts.set(snapshot.system.activeAlerts);
  }

  // Update stores with new metrics
  private updateStores(snapshot: PerformanceSnapshot): void {
    this.currentMetrics.set(snapshot);
    this.aiMetrics.set(snapshot.ai);
    this.queueMetrics.set(snapshot.queues);
    this.cacheMetrics.set(snapshot.cache);
    this.systemMetrics.set(snapshot.system);
  }

  // Default metrics for error cases
  private getDefaultAIMetrics(): AIMetrics {
    return {
      documentsProcessed: 0,
      averageProcessingTime: 0,
      totalProcessingTime: 0,
      embeddingsGenerated: 0,
      averageEmbeddingTime: 0,
      embeddingDimensions: 384,
      totalQueries: 0,
      averageQueryTime: 0,
      averageConfidence: 0,
      processingErrors: 0,
      embeddingErrors: 0,
      queryErrors: 0,
      errorRate: 1,
      modelUsage: {},
      processingTimeHistory: [],
      confidenceHistory: [],
      throughputHistory: []
    };
  }

  private getDefaultQueueMetrics(): QueueMetrics {
    return {
      queues: {},
      totalJobs: 0,
      totalCompleted: 0,
      totalFailed: 0,
      overallThroughput: 0,
      healthScore: 0,
      throughputHistory: [],
      failureRateHistory: []
    };
  }

  private getDefaultCacheMetrics(): CacheMetrics {
    return {
      hitRate: 0,
      missRate: 1,
      evictionRate: 0,
      averageAccessTime: 0,
      totalEntries: 0,
      totalSize: 0,
      memoryUsage: 0,
      layerStats: {
        memory: { entries: 0, size: 0, hitRate: 0 },
        persistent: { entries: 0, size: 0, hitRate: 0 },
        search: { entries: 0, queries: 0 }
      },
      hitRateHistory: [],
      memoryUsageHistory: []
    };
  }

  private getDefaultSystemMetrics(): SystemMetrics {
    return {
      healthScore: 0,
      uptime: Date.now() - this.startTime,
      components: {
        ollama: { status: 'unhealthy', responseTime: 0 },
        database: { status: 'unhealthy', responseTime: 0 },
        cache: { status: 'unhealthy', responseTime: 0 },
        queues: { status: 'unhealthy', responseTime: 0 }
      },
      memory: { used: 0, total: 0, percentage: 0 },
      cpu: { usage: 0 },
      networkLatency: 0,
      networkThroughput: 0,
      responseTime: 0,
      throughput: 0,
      errorRate: 1,
      activeAlerts: []
    };
  }

  // Public methods for external usage
  getMetricsHistory(hours: number = 24): PerformanceSnapshot[] {
    const cutoff = Date.now() - (hours * 60 * 60 * 1000);
    return this.metricsHistory.filter(snapshot: any => snapshot.timestamp >= cutoff);
  }

  exportMetrics(): string {
    return JSON.stringify({
      history: this.metricsHistory,
      exported: new Date().toISOString(),
      version: '1.0'
    }, null, 2);
  }

  clearHistory(): void {
    this.metricsHistory = [];
  }
}

// Export singleton instance
export const performanceMonitor = new PerformanceMonitor();

// Export utility functions
export const formatMetricValue = (value: number, type: 'time' | 'percentage' | 'count' | 'size'): string: any => {
  switch (type) {
    case 'time':
      return value < 1000 ? `${Math.round(value)}ms` : `${(value / 1000).toFixed(1)}s`;
    case 'percentage':
      return `${Math.round(value * 100)}%`;
    case 'count':
      return value.toLocaleString();
    case 'size':
      const sizes = ['B', 'KB', 'MB', 'GB'];
      let size = value;
      let i = 0;
      while (size >= 1024 && i < sizes.length - 1) {
        size /= 1024;
        i++;
      }
      return `${size.toFixed(1)} ${sizes[i]}`;
    default:
      return value.toString();
  }
};

export const getHealthStatusColor = (status: 'healthy' | 'degraded' | 'unhealthy'): string: any => {
  switch (status) {
    case 'healthy': return 'text-green-500';
    case 'degraded': return 'text-yellow-500';
    case 'unhealthy': return 'text-red-500';
    default: return 'text-gray-500';
  }
};

export const getAlertSeverityColor = (severity: SystemMetrics['activeAlerts'][0]['severity']): string: any => {
  switch (severity) {
    case 'low': return 'text-blue-500';
    case 'medium': return 'text-yellow-500';
    case 'high': return 'text-orange-500';
    case 'critical': return 'text-red-500';
    default: return 'text-gray-500';
  }
};