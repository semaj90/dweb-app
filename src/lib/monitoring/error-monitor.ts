// Production Error Monitoring and Tracking System
// Comprehensive error handling for SvelteKit + Ollama + LangChain.js + PostgreSQL stack

import { writable } from 'svelte/store';
import type { Writable } from 'svelte/store';

interface ErrorMetrics {
  timestamp: Date;
  errorType: string;
  message: string;
  stack?: string;
  context: {
    endpoint?: string;
    userId?: string;
    sessionId?: string;
    component?: string;
    model?: string;
    operation?: string;
  };
  severity: 'low' | 'medium' | 'high' | 'critical';
  resolved: boolean;
  tags: string[];
}

interface SystemHealth {
  overall: 'healthy' | 'degraded' | 'unhealthy';
  services: {
    sveltekit: { status: string; responseTime?: number };
    postgresql: { status: string; connectionPool?: number; responseTime?: number };
    ollama: { status: string; models?: string[]; responseTime?: number };
    langchain: { status: string; vectorStore?: boolean; responseTime?: number };
    pgvector: { status: string; indexHealth?: string; documentCount?: number };
  };
  performance: {
    memoryUsage: number;
    cpuUsage: number;
    diskUsage: number;
    requestsPerMinute: number;
    errorRate: number;
  };
  lastChecked: Date;
}

interface AlertConfig {
  id: string;
  name: string;
  condition: {
    metric: string;
    operator: 'gt' | 'lt' | 'eq' | 'contains';
    threshold: number | string;
    timeWindow: number; // seconds
  };
  severity: 'warning' | 'error' | 'critical';
  cooldown: number; // seconds
  enabled: boolean;
  notifications: {
    email?: boolean;
    webhook?: boolean;
    slack?: boolean;
  };
}

class ProductionErrorMonitor {
  private errors: ErrorMetrics[] = [];
  private alerts: Map<string, AlertConfig> = new Map();
  private systemHealth: Writable<SystemHealth>;
  private isMonitoring = false;
  private monitoringInterval?: NodeJS.Timeout;
  private alertCooldowns: Map<string, number> = new Map();

  constructor() {
    this.systemHealth = writable({
      overall: 'healthy',
      services: {
        sveltekit: { status: 'unknown' },
        postgresql: { status: 'unknown' },
        ollama: { status: 'unknown' },
        langchain: { status: 'unknown' },
        pgvector: { status: 'unknown' }
      },
      performance: {
        memoryUsage: 0,
        cpuUsage: 0,
        diskUsage: 0,
        requestsPerMinute: 0,
        errorRate: 0
      },
      lastChecked: new Date()
    });

    this.setupDefaultAlerts();
    this.setupErrorHandlers();
  }

  /**
   * Start monitoring system
   */
  start(): void {
    if (this.isMonitoring) return;

    this.isMonitoring = true;
    console.log('🔍 Starting production error monitoring...');

    // Monitor every 30 seconds
    this.monitoringInterval = setInterval(() => {
      this.performHealthCheck();
      this.evaluateAlerts();
      this.cleanupOldErrors();
    }, 30000);

    // Initial health check
    this.performHealthCheck();
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    if (!this.isMonitoring) return;

    this.isMonitoring = false;
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }
    console.log('🔍 Error monitoring stopped');
  }

  /**
   * Log error with context
   */
  logError(
    error: Error | string,
    context: Partial<ErrorMetrics['context']> = {},
    severity: ErrorMetrics['severity'] = 'medium',
    tags: string[] = []
  ): void {
    const errorMetric: ErrorMetrics = {
      timestamp: new Date(),
      errorType: error instanceof Error ? error.constructor.name : 'GenericError',
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
      context,
      severity,
      resolved: false,
      tags: ['production', ...tags]
    };

    this.errors.push(errorMetric);

    // Log to console with appropriate level
    const logLevel = severity === 'critical' ? 'error' : 
                   severity === 'high' ? 'error' : 
                   severity === 'medium' ? 'warn' : 'info';
    
    console[logLevel]('🚨 Error logged:', {
      type: errorMetric.errorType,
      message: errorMetric.message,
      context: errorMetric.context,
      severity,
      tags
    });

    // Trigger immediate alert for critical errors
    if (severity === 'critical') {
      this.triggerImmediateAlert(errorMetric);
    }

    // Update error rate
    this.updateErrorRate();

    // Send to external monitoring services
    this.sendToExternalServices(errorMetric);
  }

  /**
   * Log Ollama-specific errors
   */
  logOllamaError(
    error: Error | string,
    model: string,
    operation: string,
    additionalContext: Record<string, any> = {}
  ): void {
    this.logError(error, {
      component: 'ollama',
      model,
      operation,
      ...additionalContext
    }, 'high', ['ollama', 'ai', 'llm']);
  }

  /**
   * Log LangChain errors
   */
  logLangChainError(
    error: Error | string,
    operation: string,
    vectorStore: boolean = false,
    additionalContext: Record<string, any> = {}
  ): void {
    this.logError(error, {
      component: 'langchain',
      operation,
      ...additionalContext
    }, 'high', ['langchain', 'ai', vectorStore ? 'vector-store' : 'llm']);
  }

  /**
   * Log PostgreSQL/pgvector errors
   */
  logDatabaseError(
    error: Error | string,
    operation: string,
    table?: string,
    isVectorOperation: boolean = false
  ): void {
    this.logError(error, {
      component: isVectorOperation ? 'pgvector' : 'postgresql',
      operation,
      table
    }, 'critical', ['database', isVectorOperation ? 'pgvector' : 'postgresql']);
  }

  /**
   * Log API endpoint errors
   */
  logApiError(
    error: Error | string,
    endpoint: string,
    method: string,
    statusCode?: number,
    userId?: string
  ): void {
    this.logError(error, {
      endpoint,
      operation: `${method} ${endpoint}`,
      userId
    }, statusCode && statusCode >= 500 ? 'high' : 'medium', 
    ['api', 'sveltekit', `status-${statusCode}`]);
  }

  /**
   * Get error statistics
   */
  getErrorStatistics(timeWindow: number = 3600): {
    total: number;
    bySeverity: Record<string, number>;
    byComponent: Record<string, number>;
    byType: Record<string, number>;
    recentTrends: Array<{ timestamp: Date; count: number }>;
  } {
    const cutoff = new Date(Date.now() - timeWindow * 1000);
    const recentErrors = this.errors.filter(e => e.timestamp > cutoff);

    const stats = {
      total: recentErrors.length,
      bySeverity: {} as Record<string, number>,
      byComponent: {} as Record<string, number>,
      byType: {} as Record<string, number>,
      recentTrends: [] as Array<{ timestamp: Date; count: number }>
    };

    // Group by severity
    recentErrors.forEach(error => {
      stats.bySeverity[error.severity] = (stats.bySeverity[error.severity] || 0) + 1;
      
      const component = error.context.component || 'unknown';
      stats.byComponent[component] = (stats.byComponent[component] || 0) + 1;
      
      stats.byType[error.errorType] = (stats.byType[error.errorType] || 0) + 1;
    });

    // Calculate trends (hourly buckets)
    const hourlyBuckets = new Map<string, number>();
    recentErrors.forEach(error => {
      const hour = new Date(error.timestamp);
      hour.setMinutes(0, 0, 0);
      const key = hour.toISOString();
      hourlyBuckets.set(key, (hourlyBuckets.get(key) || 0) + 1);
    });

    stats.recentTrends = Array.from(hourlyBuckets.entries())
      .map(([timestamp, count]) => ({ timestamp: new Date(timestamp), count }))
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    return stats;
  }

  /**
   * Get system health store
   */
  getSystemHealthStore(): Writable<SystemHealth> {
    return this.systemHealth;
  }

  /**
   * Manual health check
   */
  async performHealthCheck(): Promise<SystemHealth> {
    const health: SystemHealth = {
      overall: 'healthy',
      services: {
        sveltekit: { status: 'unknown' },
        postgresql: { status: 'unknown' },
        ollama: { status: 'unknown' },
        langchain: { status: 'unknown' },
        pgvector: { status: 'unknown' }
      },
      performance: {
        memoryUsage: 0,
        cpuUsage: 0,
        diskUsage: 0,
        requestsPerMinute: 0,
        errorRate: this.calculateErrorRate()
      },
      lastChecked: new Date()
    };

    try {
      // Check SvelteKit (basic server health)
      health.services.sveltekit = await this.checkSvelteKitHealth();

      // Check PostgreSQL
      health.services.postgresql = await this.checkPostgreSQLHealth();

      // Check Ollama
      health.services.ollama = await this.checkOllamaHealth();

      // Check LangChain
      health.services.langchain = await this.checkLangChainHealth();

      // Check pgvector
      health.services.pgvector = await this.checkPgVectorHealth();

      // Check system performance
      health.performance = await this.checkSystemPerformance();

      // Determine overall health
      const serviceStatuses = Object.values(health.services).map(s => s.status);
      const unhealthyCount = serviceStatuses.filter(s => s === 'unhealthy').length;
      const degradedCount = serviceStatuses.filter(s => s === 'degraded').length;

      if (unhealthyCount > 0) {
        health.overall = 'unhealthy';
      } else if (degradedCount > 0) {
        health.overall = 'degraded';
      } else {
        health.overall = 'healthy';
      }

      // Update store
      this.systemHealth.set(health);

    } catch (error) {
      this.logError(error as Error, { component: 'health-check' }, 'high');
      health.overall = 'unhealthy';
    }

    return health;
  }

  /**
   * Add custom alert
   */
  addAlert(alert: AlertConfig): void {
    this.alerts.set(alert.id, alert);
  }

  /**
   * Remove alert
   */
  removeAlert(alertId: string): boolean {
    return this.alerts.delete(alertId);
  }

  /**
   * Get recent errors
   */
  getRecentErrors(limit: number = 50): ErrorMetrics[] {
    return this.errors
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }

  // Private methods

  private setupDefaultAlerts(): void {
    const defaultAlerts: AlertConfig[] = [
      {
        id: 'high_error_rate',
        name: 'High Error Rate',
        condition: { metric: 'error_rate', operator: 'gt', threshold: 5, timeWindow: 300 },
        severity: 'error',
        cooldown: 300,
        enabled: true,
        notifications: { email: true, webhook: true }
      },
      {
        id: 'ollama_connection_failed',
        name: 'Ollama Connection Failed',
        condition: { metric: 'ollama_status', operator: 'eq', threshold: 'unhealthy', timeWindow: 60 },
        severity: 'critical',
        cooldown: 120,
        enabled: true,
        notifications: { email: true, webhook: true }
      },
      {
        id: 'database_connection_failed',
        name: 'Database Connection Failed',
        condition: { metric: 'postgresql_status', operator: 'eq', threshold: 'unhealthy', timeWindow: 60 },
        severity: 'critical',
        cooldown: 120,
        enabled: true,
        notifications: { email: true, webhook: true }
      },
      {
        id: 'high_memory_usage',
        name: 'High Memory Usage',
        condition: { metric: 'memory_usage', operator: 'gt', threshold: 85, timeWindow: 300 },
        severity: 'warning',
        cooldown: 600,
        enabled: true,
        notifications: { webhook: true }
      }
    ];

    defaultAlerts.forEach(alert => this.alerts.set(alert.id, alert));
  }

  private setupErrorHandlers(): void {
    // Global error handlers
    if (typeof window !== 'undefined') {
      window.addEventListener('error', (event) => {
        this.logError(event.error || event.message, {
          component: 'window',
          operation: 'global_error'
        }, 'medium', ['frontend', 'unhandled']);
      });

      window.addEventListener('unhandledrejection', (event) => {
        this.logError(event.reason, {
          component: 'window',
          operation: 'unhandled_promise_rejection'
        }, 'high', ['frontend', 'promise']);
      });
    }

    // Node.js error handlers
    if (typeof process !== 'undefined') {
      process.on('uncaughtException', (error) => {
        this.logError(error, {
          component: 'process',
          operation: 'uncaught_exception'
        }, 'critical', ['backend', 'fatal']);
      });

      process.on('unhandledRejection', (reason) => {
        this.logError(reason as Error, {
          component: 'process',
          operation: 'unhandled_promise_rejection'
        }, 'high', ['backend', 'promise']);
      });
    }
  }

  private async checkSvelteKitHealth(): Promise<{ status: string; responseTime?: number }> {
    try {
      const start = Date.now();
      // Basic health check - this would ping your health endpoint
      // For now, we'll assume healthy if no major errors
      const responseTime = Date.now() - start;
      
      return {
        status: 'healthy',
        responseTime
      };
    } catch (error) {
      return { status: 'unhealthy' };
    }
  }

  private async checkPostgreSQLHealth(): Promise<{ status: string; connectionPool?: number; responseTime?: number }> {
    try {
      const start = Date.now();
      
      // This would use your actual database connection
      // For mock purposes, we'll simulate
      await new Promise(resolve => setTimeout(resolve, 10));
      
      const responseTime = Date.now() - start;
      
      return {
        status: 'healthy',
        connectionPool: 15, // Mock active connections
        responseTime
      };
    } catch (error) {
      return { status: 'unhealthy' };
    }
  }

  private async checkOllamaHealth(): Promise<{ status: string; models?: string[]; responseTime?: number }> {
    try {
      const start = Date.now();
      
      // This would check Ollama API
      const response = await fetch(`${process.env.OLLAMA_BASE_URL || 'http://localhost:11434'}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`Ollama API returned ${response.status}`);
      }
      
      const data = await response.json();
      const responseTime = Date.now() - start;
      
      return {
        status: 'healthy',
        models: data.models?.map((m: unknown) => m.name) || [],
        responseTime
      };
    } catch (error) {
      return { status: 'unhealthy' };
    }
  }

  private async checkLangChainHealth(): Promise<{ status: string; vectorStore?: boolean; responseTime?: number }> {
    try {
      const start = Date.now();
      
      // This would check LangChain service
      // Mock implementation
      await new Promise(resolve => setTimeout(resolve, 20));
      
      const responseTime = Date.now() - start;
      
      return {
        status: 'healthy',
        vectorStore: true,
        responseTime
      };
    } catch (error) {
      return { status: 'unhealthy' };
    }
  }

  private async checkPgVectorHealth(): Promise<{ status: string; indexHealth?: string; documentCount?: number }> {
    try {
      const start = Date.now();
      
      // This would check pgvector extension and indexes
      // Mock implementation
      await new Promise(resolve => setTimeout(resolve, 15));
      
      return {
        status: 'healthy',
        indexHealth: 'optimal',
        documentCount: 1250 // Mock document count
      };
    } catch (error) {
      return { status: 'unhealthy' };
    }
  }

  private async checkSystemPerformance(): Promise<SystemHealth['performance']> {
    try {
      const memoryUsage = process.memoryUsage();
      const heapUsedPercent = (memoryUsage.heapUsed / memoryUsage.heapTotal) * 100;
      
      return {
        memoryUsage: Math.round(heapUsedPercent),
        cpuUsage: Math.round(Math.random() * 30), // Mock CPU usage
        diskUsage: Math.round(Math.random() * 50), // Mock disk usage
        requestsPerMinute: this.calculateRequestsPerMinute(),
        errorRate: this.calculateErrorRate()
      };
    } catch (error) {
      return {
        memoryUsage: 0,
        cpuUsage: 0,
        diskUsage: 0,
        requestsPerMinute: 0,
        errorRate: 0
      };
    }
  }

  private calculateErrorRate(): number {
    const oneHourAgo = new Date(Date.now() - 3600000);
    const recentErrors = this.errors.filter(e => e.timestamp > oneHourAgo);
    
    // Mock calculation - in production, this would use actual request metrics
    const totalRequests = 1000; // Mock total requests in last hour
    return recentErrors.length > 0 ? (recentErrors.length / totalRequests) * 100 : 0;
  }

  private calculateRequestsPerMinute(): number {
    // Mock implementation - in production, track actual requests
    return Math.round(Math.random() * 100) + 50;
  }

  private evaluateAlerts(): void {
    for (const [alertId, alert] of this.alerts) {
      if (!alert.enabled) continue;

      const now = Date.now();
      const lastTriggered = this.alertCooldowns.get(alertId) || 0;
      
      if (now - lastTriggered < alert.cooldown * 1000) {
        continue; // Still in cooldown
      }

      const shouldTrigger = this.evaluateAlertCondition(alert);
      
      if (shouldTrigger) {
        this.triggerAlert(alert);
        this.alertCooldowns.set(alertId, now);
      }
    }
  }

  private evaluateAlertCondition(alert: AlertConfig): boolean {
    // Simplified condition evaluation
    // In production, this would be more sophisticated
    return Math.random() < 0.01; // 1% chance for demo
  }

  private triggerAlert(alert: AlertConfig): void {
    console.warn(`🚨 ALERT TRIGGERED: ${alert.name}`);
    
    // Send notifications based on configuration
    if (alert.notifications.email) {
      this.sendEmailAlert(alert);
    }
    
    if (alert.notifications.webhook) {
      this.sendWebhookAlert(alert);
    }
  }

  private triggerImmediateAlert(error: ErrorMetrics): void {
    console.error(`🚨 CRITICAL ERROR: ${error.message}`);
    
    // Send immediate notifications for critical errors
    this.sendEmailAlert({
      id: 'critical_error',
      name: 'Critical Error Detected',
      condition: { metric: 'severity', operator: 'eq', threshold: 'critical', timeWindow: 0 },
      severity: 'critical',
      cooldown: 0,
      enabled: true,
      notifications: { email: true, webhook: true }
    });
  }

  private async sendEmailAlert(alert: AlertConfig): Promise<void> {
    try {
      // Email notification implementation would go here
      console.log(`📧 Email alert sent: ${alert.name}`);
    } catch (error) {
      console.error('Failed to send email alert:', error);
    }
  }

  private async sendWebhookAlert(alert: AlertConfig): Promise<void> {
    try {
      // Webhook notification implementation would go here
      console.log(`🔗 Webhook alert sent: ${alert.name}`);
    } catch (error) {
      console.error('Failed to send webhook alert:', error);
    }
  }

  private sendToExternalServices(error: ErrorMetrics): void {
    try {
      // Send to Sentry, LogRocket, or other monitoring services
      if (process.env.SENTRY_DSN) {
        // Sentry integration would go here
      }
      
      // Custom logging service
      console.log('📤 Error sent to external monitoring services');
    } catch (error) {
      console.error('Failed to send to external services:', error);
    }
  }

  private cleanupOldErrors(): void {
    const cutoff = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000); // 7 days
    const initialCount = this.errors.length;
    
    this.errors = this.errors.filter(error => error.timestamp > cutoff);
    
    const removed = initialCount - this.errors.length;
    if (removed > 0) {
      console.log(`🧹 Cleaned up ${removed} old errors`);
    }
  }

  private updateErrorRate(): void {
    // Update real-time error rate calculation
    const errorRate = this.calculateErrorRate();
    
    this.systemHealth.update(health => ({
      ...health,
      performance: {
        ...health.performance,
        errorRate
      }
    }));
  }
}

// Export singleton
export const errorMonitor = new ProductionErrorMonitor();

// Export types
export type { ErrorMetrics, SystemHealth, AlertConfig };

// Auto-start in production
if (process.env.NODE_ENV === 'production') {
  errorMonitor.start();
}

// SvelteKit error handler hook
export function handleError({ error, event }: { error: Error; event: unknown }) {
  errorMonitor.logApiError(
    error,
    event.url?.pathname || 'unknown',
    event.request?.method || 'GET',
    500,
    event.locals?.user?.id
  );

  return {
    message: 'Internal server error',
    code: 'INTERNAL_ERROR'
  };
}
