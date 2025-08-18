// Performance Monitoring System
// Comprehensive monitoring for production-ready applications

import { EventEmitter } from 'events';

interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  timestamp: Date;
  tags?: Record<string, string>;
  type: 'counter' | 'gauge' | 'histogram' | 'timer';
}

interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  score: number; // 0-100
  checks: {
    cpu: { status: string; value: number };
    memory: { status: string; value: number };
    disk: { status: string; value: number };
    database: { status: string; responseTime: number };
    api: { status: string; responseTime: number };
  };
  timestamp: Date;
}

interface AlertRule {
  id: string;
  metric: string;
  condition: 'gt' | 'lt' | 'eq';
  threshold: number;
  duration: number; // seconds
  severity: 'critical' | 'warning' | 'info';
  enabled: boolean;
}

class PerformanceMonitor extends EventEmitter {
  private metrics: Map<string, PerformanceMetric[]> = new Map();
  private alerts: Map<string, AlertRule> = new Map();
  private timers: Map<string, number> = new Map();
  private counters: Map<string, number> = new Map();
  private gauges: Map<string, number> = new Map();
  private healthChecks: Map<string, () => Promise<any>> = new Map();
  private isMonitoring = false;
  private monitoringInterval?: NodeJS.Timeout;

  constructor() {
    super();
    this.setupDefaultAlerts();
    this.setupDefaultHealthChecks();
  }

  /**
   * Start performance monitoring
   */
  start(): void {
    if (this.isMonitoring) return;

    this.isMonitoring = true;
    console.log('📊 Starting performance monitoring...');

    // Collect metrics every 30 seconds
    this.monitoringInterval = setInterval(() => {
      this.collectSystemMetrics();
      this.evaluateAlerts();
    }, 30000);

    // Initial collection
    this.collectSystemMetrics();
  }

  /**
   * Stop performance monitoring
   */
  stop(): void {
    if (!this.isMonitoring) return;

    this.isMonitoring = false;
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }
    console.log('📊 Performance monitoring stopped');
  }

  /**
   * Record a metric
   */
  recordMetric(metric: Omit<PerformanceMetric, 'timestamp'>): void {
    const fullMetric: PerformanceMetric = {
      ...metric,
      timestamp: new Date()
    };

    const key = metric.name;
    if (!this.metrics.has(key)) {
      this.metrics.set(key, []);
    }

    const metrics = this.metrics.get(key)!;
    metrics.push(fullMetric);

    // Keep only last 1000 metrics per type
    if (metrics.length > 1000) {
      metrics.splice(0, metrics.length - 1000);
    }

    this.emit('metric', fullMetric);
  }

  /**
   * Start a timer
   */
  startTimer(name: string): void {
    this.timers.set(name, Date.now());
  }

  /**
   * End a timer and record the metric
   */
  endTimer(name: string, tags?: Record<string, string>): number {
    const startTime = this.timers.get(name);
    if (!startTime) {
      console.warn(`Timer ${name} was not started`);
      return 0;
    }

    const duration = Date.now() - startTime;
    this.timers.delete(name);

    this.recordMetric({
      name: `timer.${name}`,
      value: duration,
      unit: 'ms',
      type: 'timer',
      tags
    });

    return duration;
  }

  /**
   * Increment a counter
   */
  incrementCounter(name: string, value = 1, tags?: Record<string, string>): void {
    const current = this.counters.get(name) || 0;
    const newValue = current + value;
    this.counters.set(name, newValue);

    this.recordMetric({
      name: `counter.${name}`,
      value: newValue,
      unit: 'count',
      type: 'counter',
      tags
    });
  }

  /**
   * Set a gauge value
   */
  setGauge(name: string, value: number, unit = 'value', tags?: Record<string, string>): void {
    this.gauges.set(name, value);

    this.recordMetric({
      name: `gauge.${name}`,
      value,
      unit,
      type: 'gauge',
      tags
    });
  }

  /**
   * Record API request metrics
   */
  recordApiRequest(endpoint: string, method: string, statusCode: number, duration: number): void {
    const tags = { endpoint, method, status: statusCode.toString() };
    
    this.recordMetric({
      name: 'api.request.duration',
      value: duration,
      unit: 'ms',
      type: 'timer',
      tags
    });

    this.incrementCounter('api.request.total', 1, tags);

    if (statusCode >= 400) {
      this.incrementCounter('api.request.errors', 1, tags);
    }
  }

  /**
   * Record database operation metrics
   */
  recordDatabaseOperation(operation: string, table: string, duration: number, success: boolean): void {
    const tags = { operation, table, success: success.toString() };
    
    this.recordMetric({
      name: 'database.operation.duration',
      value: duration,
      unit: 'ms',
      type: 'timer',
      tags
    });

    this.incrementCounter('database.operation.total', 1, tags);

    if (!success) {
      this.incrementCounter('database.operation.errors', 1, tags);
    }
  }

  /**
   * Record AI/LLM operation metrics
   */
  recordAiOperation(model: string, operation: string, tokensUsed: number, duration: number): void {
    const tags = { model, operation };
    
    this.recordMetric({
      name: 'ai.operation.duration',
      value: duration,
      unit: 'ms',
      type: 'timer',
      tags
    });

    this.recordMetric({
      name: 'ai.operation.tokens',
      value: tokensUsed,
      unit: 'tokens',
      type: 'counter',
      tags
    });

    this.incrementCounter('ai.operation.total', 1, tags);
  }

  /**
   * Get current system health
   */
  async getSystemHealth(): Promise<SystemHealth> {
    const checks = {
      cpu: await this.checkCpuUsage(),
      memory: await this.checkMemoryUsage(),
      disk: await this.checkDiskUsage(),
      database: await this.checkDatabaseHealth(),
      api: await this.checkApiHealth()
    };

    const scores = Object.values(checks).map(check => {
      switch (check.status) {
        case 'healthy': return 100;
        case 'warning': return 60;
        case 'critical': return 20;
        default: return 0;
      }
    });

    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    
    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    if (avgScore < 40) {
      status = 'unhealthy';
    } else if (avgScore < 80) {
      status = 'degraded';
    }

    return {
      status,
      score: Math.round(avgScore),
      checks,
      timestamp: new Date()
    };
  }

  /**
   * Get metrics summary
   */
  getMetricsSummary(): Record<string, any> {
    const summary: Record<string, any> = {};

    // Counters
    summary.counters = Object.fromEntries(this.counters);

    // Gauges
    summary.gauges = Object.fromEntries(this.gauges);

    // Recent metrics (last 100 of each type)
    summary.recentMetrics = {};
    for (const [name, metrics] of this.metrics) {
      summary.recentMetrics[name] = metrics.slice(-100);
    }

    // Performance statistics
    summary.performance = this.calculatePerformanceStats();

    return summary;
  }

  /**
   * Add custom health check
   */
  addHealthCheck(name: string, check: () => Promise<any>): void {
    this.healthChecks.set(name, check);
  }

  /**
   * Add alert rule
   */
  addAlertRule(rule: AlertRule): void {
    this.alerts.set(rule.id, rule);
  }

  /**
   * Remove alert rule
   */
  removeAlertRule(id: string): boolean {
    return this.alerts.delete(id);
  }

  /**
   * Generate performance report
   */
  generateReport(): any {
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);

    const report = {
      timestamp: now,
      period: { start: oneHourAgo, end: now },
      summary: this.getMetricsSummary(),
      health: null as any,
      alerts: this.getRecentAlerts(),
      recommendations: this.generateRecommendations()
    };

    // Add health status
    this.getSystemHealth().then(health => {
      report.health = health;
    });

    return report;
  }

  // Private methods

  private setupDefaultAlerts(): void {
    const defaultAlerts: AlertRule[] = [
      {
        id: 'high_cpu',
        metric: 'system.cpu.usage',
        condition: 'gt',
        threshold: 80,
        duration: 300,
        severity: 'critical',
        enabled: true
      },
      {
        id: 'high_memory',
        metric: 'system.memory.usage',
        condition: 'gt',
        threshold: 85,
        duration: 300,
        severity: 'critical',
        enabled: true
      },
      {
        id: 'slow_api',
        metric: 'api.request.duration',
        condition: 'gt',
        threshold: 5000,
        duration: 60,
        severity: 'warning',
        enabled: true
      },
      {
        id: 'high_error_rate',
        metric: 'api.request.errors',
        condition: 'gt',
        threshold: 10,
        duration: 300,
        severity: 'critical',
        enabled: true
      }
    ];

    defaultAlerts.forEach(alert => this.alerts.set(alert.id, alert));
  }

  private setupDefaultHealthChecks(): void {
    this.addHealthCheck('system', async () => {
      return {
        cpu: process.cpuUsage(),
        memory: process.memoryUsage(),
        uptime: process.uptime()
      };
    });
  }

  private async collectSystemMetrics(): Promise<void> {
    try {
      // System metrics
      const memUsage = process.memoryUsage();
      this.setGauge('system.memory.heap_used', memUsage.heapUsed, 'bytes');
      this.setGauge('system.memory.heap_total', memUsage.heapTotal, 'bytes');
      this.setGauge('system.memory.external', memUsage.external, 'bytes');
      this.setGauge('system.memory.rss', memUsage.rss, 'bytes');

      // Process metrics
      this.setGauge('system.process.uptime', process.uptime(), 'seconds');
      this.setGauge('system.process.pid', process.pid, 'value');

      // Event loop lag
      const start = process.hrtime.bigint();
      setImmediate(() => {
        const lag = Number(process.hrtime.bigint() - start) / 1e6;
        this.setGauge('system.event_loop.lag', lag, 'ms');
      });

      // Custom metrics
      await this.collectCustomMetrics();

    } catch (error) {
      console.error('Error collecting system metrics:', error);
    }
  }

  private async collectCustomMetrics(): Promise<void> {
    // Database connection pool metrics
    try {
      // This would integrate with your actual database pool
      this.setGauge('database.pool.active', 5, 'connections'); // Mock data
      this.setGauge('database.pool.idle', 15, 'connections');
    } catch (error) {
      console.warn('Database pool metrics unavailable');
    }

    // Vector database metrics
    try {
      // This would integrate with Qdrant
      this.setGauge('vectordb.documents', 1250, 'documents'); // Mock data
      this.setGauge('vectordb.collections', 3, 'collections');
    } catch (error) {
      console.warn('Vector database metrics unavailable');
    }
  }

  private async checkCpuUsage(): Promise<{ status: string; value: number }> {
    // Mock CPU usage check
    const cpuUsage = Math.random() * 100;
    
    return {
      status: cpuUsage > 80 ? 'critical' : cpuUsage > 60 ? 'warning' : 'healthy',
      value: Math.round(cpuUsage)
    };
  }

  private async checkMemoryUsage(): Promise<{ status: string; value: number }> {
    const memUsage = process.memoryUsage();
    const usagePercent = (memUsage.heapUsed / memUsage.heapTotal) * 100;
    
    return {
      status: usagePercent > 85 ? 'critical' : usagePercent > 70 ? 'warning' : 'healthy',
      value: Math.round(usagePercent)
    };
  }

  private async checkDiskUsage(): Promise<{ status: string; value: number }> {
    // Mock disk usage check
    const diskUsage = Math.random() * 100;
    
    return {
      status: diskUsage > 90 ? 'critical' : diskUsage > 75 ? 'warning' : 'healthy',
      value: Math.round(diskUsage)
    };
  }

  private async checkDatabaseHealth(): Promise<{ status: string; responseTime: number }> {
    const start = Date.now();
    
    try {
      // Mock database health check
      await new Promise(resolve => setTimeout(resolve, Math.random() * 100));
      const responseTime = Date.now() - start;
      
      return {
        status: responseTime > 1000 ? 'critical' : responseTime > 500 ? 'warning' : 'healthy',
        responseTime
      };
    } catch (error) {
      return {
        status: 'critical',
        responseTime: Date.now() - start
      };
    }
  }

  private async checkApiHealth(): Promise<{ status: string; responseTime: number }> {
    const start = Date.now();
    
    try {
      // Mock API health check
      await new Promise(resolve => setTimeout(resolve, Math.random() * 200));
      const responseTime = Date.now() - start;
      
      return {
        status: responseTime > 2000 ? 'critical' : responseTime > 1000 ? 'warning' : 'healthy',
        responseTime
      };
    } catch (error) {
      return {
        status: 'critical',
        responseTime: Date.now() - start
      };
    }
  }

  private evaluateAlerts(): void {
    // This would evaluate alert conditions against current metrics
    // For now, we'll emit some mock alerts
    if (Math.random() < 0.1) {
      this.emit('alert', {
        id: 'mock_alert',
        severity: 'warning',
        message: 'Mock alert for demonstration',
        timestamp: new Date()
      });
    }
  }

  private getRecentAlerts(): any[] {
    // Return recent alerts (would be stored in database in production)
    return [];
  }

  private calculatePerformanceStats(): any {
    const stats: any = {};

    // Calculate stats for each metric type
    for (const [name, metrics] of this.metrics) {
      if (metrics.length === 0) continue;

      const values = metrics.map(m => m.value);
      stats[name] = {
        count: values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        avg: values.reduce((a, b) => a + b, 0) / values.length,
        p50: this.percentile(values, 0.5),
        p95: this.percentile(values, 0.95),
        p99: this.percentile(values, 0.99)
      };
    }

    return stats;
  }

  private percentile(values: number[], p: number): number {
    const sorted = values.slice().sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * p) - 1;
    return sorted[index] || 0;
  }

  private generateRecommendations(): string[] {
    const recommendations: string[] = [];

    // Analyze metrics and generate recommendations
    const memUsage = process.memoryUsage();
    const heapUsagePercent = (memUsage.heapUsed / memUsage.heapTotal) * 100;

    if (heapUsagePercent > 80) {
      recommendations.push('Consider increasing heap size or optimizing memory usage');
    }

    const errorRate = this.counters.get('api.request.errors') || 0;
    const totalRequests = this.counters.get('api.request.total') || 1;
    
    if ((errorRate / totalRequests) > 0.05) {
      recommendations.push('High API error rate detected - investigate error causes');
    }

    if (recommendations.length === 0) {
      recommendations.push('System performance is within normal parameters');
    }

    return recommendations;
  }
}

// Express middleware for automatic API monitoring
export function createPerformanceMiddleware(monitor: PerformanceMonitor) {
  return (req: any, res: any, next: any) => {
    const start = Date.now();
    
    res.on('finish', () => {
      const duration = Date.now() - start;
      const endpoint = req.route?.path || req.path;
      
      monitor.recordApiRequest(
        endpoint,
        req.method,
        res.statusCode,
        duration
      );
    });
    
    next();
  };
}

// Singleton instance
export const performanceMonitor = new PerformanceMonitor();

// Export types and class
export { PerformanceMonitor, type PerformanceMetric, type SystemHealth, type AlertRule };

// Auto-start monitoring in production
if (process.env.NODE_ENV === 'production') {
  performanceMonitor.start();
}
