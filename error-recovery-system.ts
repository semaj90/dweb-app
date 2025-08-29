/**
 * Error Handling and Recovery System
 * Provides comprehensive error detection, classification, and automated recovery
 */

import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';
import { performance } from 'perf_hooks';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);

export interface ErrorContext {
  serviceName: string;
  errorType: ErrorType;
  severity: ErrorSeverity;
  message: string;
  stack?: string;
  timestamp: Date;
  metadata: Record<string, any>;
  correlationId: string;
}

export interface RecoveryAction {
  id: string;
  name: string;
  description: string;
  automated: boolean;
  condition: (error: ErrorContext) => boolean;
  execute: (error: ErrorContext) => Promise<RecoveryResult>;
  timeout: number;
  retries: number;
  cooldown: number;
}

export interface RecoveryResult {
  success: boolean;
  message: string;
  duration: number;
  nextAction?: string;
  metadata?: Record<string, any>;
}

export interface ErrorPattern {
  id: string;
  name: string;
  pattern: RegExp | string;
  category: ErrorCategory;
  severity: ErrorSeverity;
  commonCauses: string[];
  suggestedActions: string[];
  autoRecoverable: boolean;
}

export interface ServiceHealthSnapshot {
  serviceName: string;
  timestamp: Date;
  status: 'healthy' | 'degraded' | 'unhealthy' | 'critical';
  metrics: {
    cpu: number;
    memory: number;
    connections: number;
    responseTime: number;
    errorRate: number;
    throughput: number;
  };
  errors: ErrorContext[];
  recovery: {
    inProgress: boolean;
    lastAttempt?: Date;
    attemptCount: number;
    success: boolean;
  };
}

enum ErrorType {
  CONNECTION_FAILURE = 'connection_failure',
  TIMEOUT = 'timeout',
  RESOURCE_EXHAUSTED = 'resource_exhausted',
  AUTHENTICATION = 'authentication',
  AUTHORIZATION = 'authorization',
  VALIDATION = 'validation',
  INTERNAL_ERROR = 'internal_error',
  EXTERNAL_DEPENDENCY = 'external_dependency',
  CONFIGURATION = 'configuration',
  STARTUP_FAILURE = 'startup_failure',
  SHUTDOWN_ERROR = 'shutdown_error',
  DATABASE_ERROR = 'database_error',
  NETWORK_ERROR = 'network_error',
  FILE_SYSTEM_ERROR = 'file_system_error',
  GPU_ERROR = 'gpu_error',
  MEMORY_LEAK = 'memory_leak',
  DEADLOCK = 'deadlock'
}

enum ErrorSeverity {
  LOW = 1,
  MEDIUM = 2,
  HIGH = 3,
  CRITICAL = 4,
  CATASTROPHIC = 5
}

enum ErrorCategory {
  INFRASTRUCTURE = 'infrastructure',
  APPLICATION = 'application',
  NETWORK = 'network',
  SECURITY = 'security',
  PERFORMANCE = 'performance',
  DATA = 'data'
}

export class ErrorRecoverySystem extends EventEmitter {
  private errorHistory: Map<string, ErrorContext[]> = new Map();
  private recoveryActions: Map<string, RecoveryAction> = new Map();
  private errorPatterns: Map<string, ErrorPattern> = new Map();
  private serviceHealth: Map<string, ServiceHealthSnapshot> = new Map();
  private activeRecoveries: Set<string> = new Set();
  private recoveryQueue: Array<{ error: ErrorContext; action: RecoveryAction }> = [];
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  
  private readonly maxErrorHistory = 1000;
  private readonly maxRetries = 3;
  private readonly baseRetryDelay = 1000;
  private readonly maxRecoveryTimeout = 300000; // 5 minutes

  constructor() {
    super();
    this.initializeErrorPatterns();
    this.initializeRecoveryActions();
    this.startErrorProcessor();
  }

  private initializeErrorPatterns(): void {
    const patterns: ErrorPattern[] = [
      {
        id: 'connection_refused',
        name: 'Connection Refused',
        pattern: /ECONNREFUSED|connection refused|connection failed/i,
        category: ErrorCategory.NETWORK,
        severity: ErrorSeverity.HIGH,
        commonCauses: ['Service not running', 'Network issues', 'Port blocked'],
        suggestedActions: ['restart_service', 'check_network', 'verify_configuration'],
        autoRecoverable: true
      },
      {
        id: 'timeout_error',
        name: 'Request Timeout',
        pattern: /timeout|ETIMEDOUT|request timed out/i,
        category: ErrorCategory.PERFORMANCE,
        severity: ErrorSeverity.MEDIUM,
        commonCauses: ['High load', 'Network latency', 'Resource contention'],
        suggestedActions: ['scale_up', 'optimize_query', 'increase_timeout'],
        autoRecoverable: true
      },
      {
        id: 'out_of_memory',
        name: 'Out of Memory',
        pattern: /out of memory|OOM|memory exhausted|ENOMEM/i,
        category: ErrorCategory.INFRASTRUCTURE,
        severity: ErrorSeverity.CRITICAL,
        commonCauses: ['Memory leak', 'Insufficient resources', 'Large dataset'],
        suggestedActions: ['restart_service', 'increase_memory', 'garbage_collect'],
        autoRecoverable: true
      },
      {
        id: 'database_connection_lost',
        name: 'Database Connection Lost',
        pattern: /database.*disconnect|connection.*lost|database.*unavailable/i,
        category: ErrorCategory.DATA,
        severity: ErrorSeverity.HIGH,
        commonCauses: ['Database restart', 'Network issues', 'Connection pool exhausted'],
        suggestedActions: ['reconnect_database', 'check_db_health', 'restart_connection_pool'],
        autoRecoverable: true
      },
      {
        id: 'gpu_error',
        name: 'GPU Processing Error',
        pattern: /CUDA|GPU|nvidia|graphics|tensor/i,
        category: ErrorCategory.INFRASTRUCTURE,
        severity: ErrorSeverity.HIGH,
        commonCauses: ['GPU memory exhausted', 'Driver issues', 'Hardware failure'],
        suggestedActions: ['restart_gpu_service', 'check_gpu_memory', 'fallback_to_cpu'],
        autoRecoverable: true
      },
      {
        id: 'port_in_use',
        name: 'Port Already in Use',
        pattern: /EADDRINUSE|port.*already in use|address already in use/i,
        category: ErrorCategory.CONFIGURATION,
        severity: ErrorSeverity.HIGH,
        commonCauses: ['Multiple service instances', 'Port conflict', 'Previous process not cleaned'],
        suggestedActions: ['kill_port_process', 'change_port', 'cleanup_processes'],
        autoRecoverable: true
      },
      {
        id: 'disk_space_full',
        name: 'Disk Space Exhausted',
        pattern: /ENOSPC|no space left|disk.*full|storage.*full/i,
        category: ErrorCategory.INFRASTRUCTURE,
        severity: ErrorSeverity.CRITICAL,
        commonCauses: ['Log files growth', 'Temp files', 'Database growth'],
        suggestedActions: ['cleanup_logs', 'cleanup_temp_files', 'archive_data'],
        autoRecoverable: true
      }
    ];

    for (const pattern of patterns) {
      this.errorPatterns.set(pattern.id, pattern);
    }
  }

  private initializeRecoveryActions(): void {
    const actions: RecoveryAction[] = [
      {
        id: 'restart_service',
        name: 'Restart Service',
        description: 'Performs a graceful service restart',
        automated: true,
        condition: (error) => error.severity >= ErrorSeverity.HIGH,
        execute: async (error) => {
          const startTime = performance.now();
          try {
            await this.restartService(error.serviceName);
            return {
              success: true,
              message: `Service ${error.serviceName} restarted successfully`,
              duration: performance.now() - startTime
            };
          } catch (err: any) {
            return {
              success: false,
              message: `Failed to restart service: ${err}`,
              duration: performance.now() - startTime
            };
          }
        },
        timeout: 60000,
        retries: 2,
        cooldown: 30000
      },
      {
        id: 'reconnect_database',
        name: 'Reconnect Database',
        description: 'Reestablishes database connections',
        automated: true,
        condition: (error) => error.errorType === ErrorType.DATABASE_ERROR,
        execute: async (error) => {
          const startTime = performance.now();
          try {
            await this.reconnectDatabase(error.serviceName);
            return {
              success: true,
              message: 'Database connection reestablished',
              duration: performance.now() - startTime
            };
          } catch (err: any) {
            return {
              success: false,
              message: `Failed to reconnect database: ${err}`,
              duration: performance.now() - startTime,
              nextAction: 'restart_service'
            };
          }
        },
        timeout: 30000,
        retries: 3,
        cooldown: 10000
      },
      {
        id: 'scale_up',
        name: 'Scale Up Service',
        description: 'Increases service resources or instances',
        automated: true,
        condition: (error) => 
          error.errorType === ErrorType.RESOURCE_EXHAUSTED || 
          error.errorType === ErrorType.TIMEOUT,
        execute: async (error) => {
          const startTime = performance.now();
          try {
            await this.scaleUpService(error.serviceName);
            return {
              success: true,
              message: `Scaled up ${error.serviceName}`,
              duration: performance.now() - startTime
            };
          } catch (err: any) {
            return {
              success: false,
              message: `Failed to scale up: ${err}`,
              duration: performance.now() - startTime
            };
          }
        },
        timeout: 120000,
        retries: 1,
        cooldown: 300000 // 5 minutes
      },
      {
        id: 'cleanup_resources',
        name: 'Cleanup Resources',
        description: 'Cleans up temporary files, logs, and unused resources',
        automated: true,
        condition: (error) => 
          error.errorType === ErrorType.RESOURCE_EXHAUSTED ||
          error.message.includes('disk') ||
          error.message.includes('space'),
        execute: async (error) => {
          const startTime = performance.now();
          try {
            const cleaned = await this.cleanupResources(error.serviceName);
            return {
              success: true,
              message: `Cleaned up resources: ${cleaned.join(', ')}`,
              duration: performance.now() - startTime,
              metadata: { cleanedResources: cleaned }
            };
          } catch (err: any) {
            return {
              success: false,
              message: `Failed to cleanup resources: ${err}`,
              duration: performance.now() - startTime
            };
          }
        },
        timeout: 45000,
        retries: 2,
        cooldown: 60000
      },
      {
        id: 'reset_circuit_breaker',
        name: 'Reset Circuit Breaker',
        description: 'Resets circuit breaker to allow traffic',
        automated: true,
        condition: (error) => error.message.includes('circuit') || error.message.includes('breaker'),
        execute: async (error) => {
          const startTime = performance.now();
          try {
            this.resetCircuitBreaker(error.serviceName);
            return {
              success: true,
              message: `Circuit breaker reset for ${error.serviceName}`,
              duration: performance.now() - startTime
            };
          } catch (err: any) {
            return {
              success: false,
              message: `Failed to reset circuit breaker: ${err}`,
              duration: performance.now() - startTime
            };
          }
        },
        timeout: 5000,
        retries: 1,
        cooldown: 30000
      },
      {
        id: 'fallback_to_cpu',
        name: 'Fallback to CPU Processing',
        description: 'Switches GPU workloads to CPU when GPU fails',
        automated: true,
        condition: (error) => error.errorType === ErrorType.GPU_ERROR,
        execute: async (error) => {
          const startTime = performance.now();
          try {
            await this.enableCpuFallback(error.serviceName);
            return {
              success: true,
              message: `Enabled CPU fallback for ${error.serviceName}`,
              duration: performance.now() - startTime
            };
          } catch (err: any) {
            return {
              success: false,
              message: `Failed to enable CPU fallback: ${err}`,
              duration: performance.now() - startTime
            };
          }
        },
        timeout: 30000,
        retries: 1,
        cooldown: 60000
      }
    ];

    for (const action of actions) {
      this.recoveryActions.set(action.id, action);
    }
  }

  private startErrorProcessor(): void {
    // Process recovery queue every 5 seconds
    setInterval(() => {
      this.processRecoveryQueue();
    }, 5000);

    // Clean up old error history every hour
    setInterval(() => {
      this.cleanupErrorHistory();
    }, 3600000);
  }

  // Main error handling entry point
  async handleError(error: ErrorContext): Promise<void> {
    this.emit('errorReceived', error);
    
    // Store error in history
    this.storeError(error);
    
    // Classify and enrich error
    const enrichedError = await this.classifyError(error);
    
    // Update service health
    await this.updateServiceHealth(enrichedError);
    
    // Check circuit breaker
    if (this.shouldTriggerCircuitBreaker(enrichedError)) {
      this.triggerCircuitBreaker(enrichedError.serviceName);
    }
    
    // Find appropriate recovery actions
    const actions = this.findRecoveryActions(enrichedError);
    
    // Execute automated recovery if available
    for (const action of actions) {
      if (action.automated && !this.activeRecoveries.has(enrichedError.serviceName)) {
        this.queueRecoveryAction(enrichedError, action);
      }
    }
    
    this.emit('errorProcessed', enrichedError, actions);
  }

  private storeError(error: ErrorContext): void {
    const history = this.errorHistory.get(error.serviceName) || [];
    history.push(error);
    
    // Limit history size
    if (history.length > this.maxErrorHistory) {
      history.splice(0, history.length - this.maxErrorHistory);
    }
    
    this.errorHistory.set(error.serviceName, history);
  }

  private async classifyError(error: ErrorContext): Promise<ErrorContext> {
    // Find matching pattern
    for (const pattern of this.errorPatterns.values()) {
      const regex = typeof pattern.pattern === 'string' 
        ? new RegExp(pattern.pattern, 'i')
        : pattern.pattern;
      
      if (regex.test(error.message) || (error.stack && regex.test(error.stack))) {
        return {
          ...error,
          metadata: {
            ...error.metadata,
            pattern: pattern.id,
            category: pattern.category,
            commonCauses: pattern.commonCauses,
            suggestedActions: pattern.suggestedActions,
            autoRecoverable: pattern.autoRecoverable
          }
        };
      }
    }
    
    return error;
  }

  private async updateServiceHealth(error: ErrorContext): Promise<void> {
    const current = this.serviceHealth.get(error.serviceName) || {
      serviceName: error.serviceName,
      timestamp: new Date(),
      status: 'healthy',
      metrics: {
        cpu: 0,
        memory: 0,
        connections: 0,
        responseTime: 0,
        errorRate: 0,
        throughput: 0
      },
      errors: [],
      recovery: {
        inProgress: false,
        attemptCount: 0,
        success: true
      }
    };

    current.errors.push(error);
    current.timestamp = new Date();
    
    // Calculate error rate
    const recentErrors = current.errors.filter(
      e => Date.now() - e.timestamp.getTime() < 300000 // Last 5 minutes
    );
    current.metrics.errorRate = recentErrors.length / 5; // Errors per minute
    
    // Determine health status
    if (error.severity >= ErrorSeverity.CRITICAL) {
      current.status = 'critical';
    } else if (error.severity >= ErrorSeverity.HIGH) {
      current.status = 'unhealthy';
    } else if (current.metrics.errorRate > 5) {
      current.status = 'degraded';
    }
    
    this.serviceHealth.set(error.serviceName, current);
    this.emit('healthUpdated', current);
  }

  private shouldTriggerCircuitBreaker(error: ErrorContext): boolean {
    const history = this.errorHistory.get(error.serviceName) || [];
    const recentErrors = history.filter(
      e => Date.now() - e.timestamp.getTime() < 60000 // Last minute
    );
    
    return recentErrors.length >= 5 || error.severity >= ErrorSeverity.CRITICAL;
  }

  private triggerCircuitBreaker(serviceName: string): void {
    let breaker = this.circuitBreakers.get(serviceName);
    if (!breaker) {
      breaker = new CircuitBreaker(serviceName);
      this.circuitBreakers.set(serviceName, breaker);
    }
    
    breaker.trip();
    this.emit('circuitBreakerTripped', serviceName);
  }

  private resetCircuitBreaker(serviceName: string): void {
    const breaker = this.circuitBreakers.get(serviceName);
    if (breaker) {
      breaker.reset();
      this.emit('circuitBreakerReset', serviceName);
    }
  }

  private findRecoveryActions(error: ErrorContext): RecoveryAction[] {
    const actions: RecoveryAction[] = [];
    
    for (const action of this.recoveryActions.values()) {
      if (action.condition(error)) {
        actions.push(action);
      }
    }
    
    // Sort by automated first, then by success rate
    return actions.sort((a, b) => {
      if (a.automated !== b.automated) {
        return a.automated ? -1 : 1;
      }
      return 0;
    });
  }

  private queueRecoveryAction(error: ErrorContext, action: RecoveryAction): void {
    this.recoveryQueue.push({ error, action });
    this.emit('recoveryQueued', error.serviceName, action.id);
  }

  private async processRecoveryQueue(): Promise<void> {
    if (this.recoveryQueue.length === 0) {
      return;
    }

    const { error, action } = this.recoveryQueue.shift()!;
    
    if (this.activeRecoveries.has(error.serviceName)) {
      // Re-queue if service is already being recovered
      this.recoveryQueue.push({ error, action });
      return;
    }

    this.activeRecoveries.add(error.serviceName);
    this.emit('recoveryStarted', error.serviceName, action.id);
    
    try {
      const result = await this.executeRecoveryAction(error, action);
      
      if (result.success) {
        this.emit('recoverySucceeded', error.serviceName, action.id, result);
        await this.markRecoverySuccess(error.serviceName);
      } else {
        this.emit('recoveryFailed', error.serviceName, action.id, result);
        
        if (result.nextAction) {
          const nextAction = this.recoveryActions.get(result.nextAction);
          if (nextAction) {
            this.queueRecoveryAction(error, nextAction);
          }
        }
      }
    } catch (err: any) {
      this.emit('recoveryError', error.serviceName, action.id, err);
    } finally {
      this.activeRecoveries.delete(error.serviceName);
    }
  }

  private async executeRecoveryAction(
    error: ErrorContext, 
    action: RecoveryAction
  ): Promise<RecoveryResult> {
    const timeout = new Promise<RecoveryResult>((_, reject) =>
      setTimeout(() => reject(new Error('Recovery action timeout')), action.timeout)
    );

    const execution = action.execute(error);
    
    try {
      return await Promise.race([execution, timeout]);
    } catch (err: any) {
      return {
        success: false,
        message: `Recovery action failed: ${err}`,
        duration: action.timeout
      };
    }
  }

  private async markRecoverySuccess(serviceName: string): Promise<void> {
    const health = this.serviceHealth.get(serviceName);
    if (health) {
      health.recovery.success = true;
      health.recovery.inProgress = false;
      health.status = 'healthy';
      this.serviceHealth.set(serviceName, health);
    }
  }

  private cleanupErrorHistory(): void {
    const cutoff = Date.now() - 24 * 60 * 60 * 1000; // 24 hours ago
    
    for (const [serviceName, errors] of this.errorHistory) {
      const filtered = errors.filter(e => e.timestamp.getTime() > cutoff);
      this.errorHistory.set(serviceName, filtered);
    }
  }

  // Recovery action implementations
  private async restartService(serviceName: string): Promise<void> {
    this.emit('serviceRestarting', serviceName);
    // Implementation would integrate with actual service manager
    await new Promise(resolve => setTimeout(resolve, 5000));
  }

  private async reconnectDatabase(serviceName: string): Promise<void> {
    this.emit('databaseReconnecting', serviceName);
    // Implementation would handle database reconnection
    await new Promise(resolve => setTimeout(resolve, 3000));
  }

  private async scaleUpService(serviceName: string): Promise<void> {
    this.emit('serviceScaling', serviceName);
    // Implementation would handle service scaling
    await new Promise(resolve => setTimeout(resolve, 10000));
  }

  private async cleanupResources(serviceName: string): Promise<string[]> {
    const cleaned: string[] = [];
    
    // Cleanup logs
    try {
      const logDir = path.join(process.cwd(), 'logs');
      const files = await fs.readdir(logDir);
      const oldLogs = files.filter(f => f.endsWith('.log') && f.includes(serviceName));
      
      for (const log of oldLogs) {
        const stats = await fs.stat(path.join(logDir, log));
        if (Date.now() - stats.mtime.getTime() > 7 * 24 * 60 * 60 * 1000) {
          await fs.unlink(path.join(logDir, log));
          cleaned.push(`log:${log}`);
        }
      }
    } catch (err: any) {
      // Ignore cleanup errors
    }

    // Cleanup temp files
    try {
      const tempDir = path.join(process.cwd(), 'temp');
      const files = await fs.readdir(tempDir);
      
      for (const file of files) {
        if (file.endsWith('.tmp')) {
          await fs.unlink(path.join(tempDir, file));
          cleaned.push(`temp:${file}`);
        }
      }
    } catch (err: any) {
      // Ignore cleanup errors
    }

    return cleaned;
  }

  private async enableCpuFallback(serviceName: string): Promise<void> {
    this.emit('cpuFallbackEnabled', serviceName);
    // Implementation would configure CPU fallback
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  // Public API
  getServiceHealth(serviceName: string): ServiceHealthSnapshot | undefined {
    return this.serviceHealth.get(serviceName);
  }

  getAllServiceHealth(): ServiceHealthSnapshot[] {
    return Array.from(this.serviceHealth.values());
  }

  getErrorHistory(serviceName: string, limit?: number): ErrorContext[] {
    const history = this.errorHistory.get(serviceName) || [];
    return limit ? history.slice(-limit) : history;
  }

  getRecoveryActions(): RecoveryAction[] {
    return Array.from(this.recoveryActions.values());
  }

  async generateErrorReport(): Promise<string> {
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        totalServices: this.serviceHealth.size,
        healthyServices: Array.from(this.serviceHealth.values()).filter(s => s.status === 'healthy').length,
        unhealthyServices: Array.from(this.serviceHealth.values()).filter(s => s.status !== 'healthy').length,
        activeRecoveries: this.activeRecoveries.size,
        queuedRecoveries: this.recoveryQueue.length
      },
      services: Array.from(this.serviceHealth.values()),
      recentErrors: this.getRecentErrors(100),
      circuitBreakers: Array.from(this.circuitBreakers.entries()).map(([name, breaker]) => ({
        service: name,
        state: breaker.getState(),
        failures: breaker.getFailureCount(),
        lastFailure: breaker.getLastFailure()
      }))
    };

    return JSON.stringify(report, null, 2);
  }

  private getRecentErrors(limit: number): ErrorContext[] {
    const allErrors: ErrorContext[] = [];
    
    for (const errors of this.errorHistory.values()) {
      allErrors.push(...errors);
    }
    
    return allErrors
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }
}

class CircuitBreaker {
  private serviceName: string;
  private state: 'closed' | 'open' | 'half-open' = 'closed';
  private failureCount = 0;
  private lastFailure?: Date;
  private nextAttempt?: Date;
  private readonly threshold = 5;
  private readonly timeout = 60000; // 1 minute

  constructor(serviceName: string) {
    this.serviceName = serviceName;
  }

  trip(): void {
    this.failureCount++;
    this.lastFailure = new Date();
    
    if (this.failureCount >= this.threshold) {
      this.state = 'open';
      this.nextAttempt = new Date(Date.now() + this.timeout);
    }
  }

  reset(): void {
    this.state = 'closed';
    this.failureCount = 0;
    this.lastFailure = undefined;
    this.nextAttempt = undefined;
  }

  canProceed(): boolean {
    if (this.state === 'closed') {
      return true;
    }
    
    if (this.state === 'open' && this.nextAttempt && Date.now() > this.nextAttempt.getTime()) {
      this.state = 'half-open';
      return true;
    }
    
    return this.state === 'half-open';
  }

  getState(): string {
    return this.state;
  }

  getFailureCount(): number {
    return this.failureCount;
  }

  getLastFailure(): Date | undefined {
    return this.lastFailure;
  }
}

export { ErrorType, ErrorSeverity, ErrorCategory, ErrorContext, RecoveryAction, ServiceHealthSnapshot };
export default ErrorRecoverySystem;