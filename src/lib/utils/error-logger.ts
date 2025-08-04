import { multiLayerCache } from '../cache/multi-layer-cache.js';
import { rabbitmqService } from '../messaging/rabbitmq-service.js';

/**
 * Comprehensive Error Logging System
 * Integrates with VS Code for real-time error reporting and debugging
 */

export interface ErrorContext {
  userId?: string;
  sessionId?: string;
  requestId?: string;
  userAgent?: string;
  url?: string;
  timestamp: Date;
  environment: 'development' | 'production' | 'test';
  buildVersion?: string;
}

export interface ErrorLog {
  id: string;
  level: 'error' | 'warn' | 'info' | 'debug';
  message: string;
  error?: Error;
  stack?: string;
  context: ErrorContext;
  tags: string[];
  metadata: Record<string, any>;
  resolved?: boolean;
  resolvedAt?: Date;
  resolvedBy?: string;
}

export interface ErrorMetrics {
  totalErrors: number;
  errorsByLevel: Record<string, number>;
  errorsByTag: Record<string, number>;
  topErrors: Array<{ message: string; count: number }>;
  errorRate: number;
  lastHour: number;
  resolved: number;
}

export class ErrorLogger {
  private logs: Map<string, ErrorLog> = new Map();
  private metrics: ErrorMetrics = {
    totalErrors: 0,
    errorsByLevel: {},
    errorsByTag: {},
    topErrors: [],
    errorRate: 0,
    lastHour: 0,
    resolved: 0
  };

  constructor(
    private maxLogs: number = 10000,
    private enableVSCodeIntegration: boolean = true,
    private enableRemoteLogging: boolean = false
  ) {
    this.initializeErrorHandling();
  }

  /**
   * Initialize global error handling
   */
  private initializeErrorHandling(): void {
    // Browser error handling
    if (typeof window !== 'undefined') {
      window.addEventListener('error', (event) => {
        this.logError('Uncaught Error', event.error, {
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno
        }, ['uncaught', 'browser']);
      });

      window.addEventListener('unhandledrejection', (event) => {
        this.logError('Unhandled Promise Rejection', event.reason, {
          promise: event.promise
        }, ['unhandled-rejection', 'promise']);
      });
    }

    // Node.js error handling
    if (typeof process !== 'undefined') {
      process.on('uncaughtException', (error) => {
        this.logError('Uncaught Exception', error, {
          pid: process.pid
        }, ['uncaught', 'node']);
      });

      process.on('unhandledRejection', (reason, promise) => {
        this.logError('Unhandled Rejection', reason as Error, {
          promise: promise.toString()
        }, ['unhandled-rejection', 'node']);
      });
    }
  }

  /**
   * Log an error with full context
   */
  logError(
    message: string,
    error?: Error,
    metadata: Record<string, any> = {},
    tags: string[] = [],
    level: ErrorLog['level'] = 'error'
  ): string {
    const id = this.generateErrorId();
    const context = this.getCurrentContext();

    const errorLog: ErrorLog = {
      id,
      level,
      message,
      error,
      stack: error?.stack,
      context,
      tags: [...tags, level],
      metadata,
      resolved: false
    };

    // Store locally
    this.logs.set(id, errorLog);
    this.updateMetrics(errorLog);

    // Manage log size
    if (this.logs.size > this.maxLogs) {
      const oldestKey = this.logs.keys().next().value;
      this.logs.delete(oldestKey);
    }

    // Cache for persistence
    this.cacheError(errorLog);

    // Send to VS Code if enabled
    if (this.enableVSCodeIntegration) {
      this.sendToVSCode(errorLog);
    }

    // Send to remote logging service
    if (this.enableRemoteLogging) {
      this.sendToRemoteLogger(errorLog);
    }

    // Broadcast error event
    this.broadcastError(errorLog);

    console.error(`[${level.toUpperCase()}] ${message}`, {
      id,
      error,
      metadata,
      tags
    });

    return id;
  }

  /**
   * Log warning
   */
  logWarning(message: string, metadata: Record<string, any> = {}, tags: string[] = []): string {
    return this.logError(message, undefined, metadata, [...tags, 'warning'], 'warn');
  }

  /**
   * Log info
   */
  logInfo(message: string, metadata: Record<string, any> = {}, tags: string[] = []): string {
    return this.logError(message, undefined, metadata, [...tags, 'info'], 'info');
  }

  /**
   * Log debug information
   */
  logDebug(message: string, metadata: Record<string, any> = {}, tags: string[] = []): string {
    return this.logError(message, undefined, metadata, [...tags, 'debug'], 'debug');
  }

  /**
   * Mark error as resolved
   */
  resolveError(errorId: string, resolvedBy?: string): boolean {
    const errorLog = this.logs.get(errorId);
    if (errorLog) {
      errorLog.resolved = true;
      errorLog.resolvedAt = new Date();
      errorLog.resolvedBy = resolvedBy;
      
      this.metrics.resolved++;
      
      // Update cache
      this.cacheError(errorLog);
      
      // Notify VS Code
      if (this.enableVSCodeIntegration) {
        this.notifyVSCodeResolution(errorLog);
      }
      
      return true;
    }
    return false;
  }

  /**
   * Get error by ID
   */
  getError(errorId: string): ErrorLog | undefined {
    return this.logs.get(errorId);
  }

  /**
   * Get errors by filter
   */
  getErrors(filter: {
    level?: ErrorLog['level'];
    tags?: string[];
    resolved?: boolean;
    since?: Date;
    limit?: number;
  } = {}): ErrorLog[] {
    let errors = Array.from(this.logs.values());

    if (filter.level) {
      errors = errors.filter(e => e.level === filter.level);
    }

    if (filter.tags && filter.tags.length > 0) {
      errors = errors.filter(e => 
        filter.tags!.some(tag => e.tags.includes(tag))
      );
    }

    if (filter.resolved !== undefined) {
      errors = errors.filter(e => e.resolved === filter.resolved);
    }

    if (filter.since) {
      errors = errors.filter(e => e.context.timestamp >= filter.since!);
    }

    // Sort by timestamp (newest first)
    errors.sort((a, b) => b.context.timestamp.getTime() - a.context.timestamp.getTime());

    if (filter.limit) {
      errors = errors.slice(0, filter.limit);
    }

    return errors;
  }

  /**
   * Get error metrics and statistics
   */
  getMetrics(): ErrorMetrics {
    return { ...this.metrics };
  }

  /**
   * Clear all errors
   */
  clearErrors(): void {
    this.logs.clear();
    this.metrics = {
      totalErrors: 0,
      errorsByLevel: {},
      errorsByTag: {},
      topErrors: [],
      errorRate: 0,
      lastHour: 0,
      resolved: 0
    };
  }

  /**
   * Export errors to JSON
   */
  exportErrors(filter?: Parameters<typeof this.getErrors>[0]): string {
    const errors = this.getErrors(filter);
    return JSON.stringify({
      exportedAt: new Date().toISOString(),
      totalErrors: errors.length,
      errors: errors.map(e => ({
        ...e,
        error: e.error ? {
          name: e.error.name,
          message: e.error.message,
          stack: e.error.stack
        } : undefined
      }))
    }, null, 2);
  }

  /**
   * Import errors from JSON
   */
  importErrors(jsonData: string): number {
    try {
      const data = JSON.parse(jsonData);
      let imported = 0;

      if (data.errors && Array.isArray(data.errors)) {
        for (const errorData of data.errors) {
          // Reconstruct error object if it exists
          let error: Error | undefined;
          if (errorData.error) {
            error = new Error(errorData.error.message);
            error.name = errorData.error.name;
            error.stack = errorData.error.stack;
          }

          const errorLog: ErrorLog = {
            ...errorData,
            error,
            context: {
              ...errorData.context,
              timestamp: new Date(errorData.context.timestamp)
            },
            resolvedAt: errorData.resolvedAt ? new Date(errorData.resolvedAt) : undefined
          };

          this.logs.set(errorLog.id, errorLog);
          imported++;
        }

        // Recalculate metrics
        this.recalculateMetrics();
      }

      return imported;
    } catch (error) {
      this.logError('Failed to import errors', error as Error, { jsonData });
      return 0;
    }
  }

  // Private helper methods

  private generateErrorId(): string {
    return `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getCurrentContext(): ErrorContext {
    return {
      timestamp: new Date(),
      environment: (process.env.NODE_ENV as any) || 'development',
      buildVersion: process.env.BUILD_VERSION,
      ...(typeof window !== 'undefined' && {
        userAgent: navigator.userAgent,
        url: window.location.href
      }),
      ...(typeof process !== 'undefined' && {
        userId: process.env.USER_ID,
        sessionId: process.env.SESSION_ID
      })
    };
  }

  private updateMetrics(errorLog: ErrorLog): void {
    this.metrics.totalErrors++;
    
    // Count by level
    this.metrics.errorsByLevel[errorLog.level] = 
      (this.metrics.errorsByLevel[errorLog.level] || 0) + 1;
    
    // Count by tags
    errorLog.tags.forEach(tag => {
      this.metrics.errorsByTag[tag] = (this.metrics.errorsByTag[tag] || 0) + 1;
    });

    // Update top errors
    const existing = this.metrics.topErrors.find(e => e.message === errorLog.message);
    if (existing) {
      existing.count++;
    } else {
      this.metrics.topErrors.push({ message: errorLog.message, count: 1 });
    }

    // Sort and limit top errors
    this.metrics.topErrors.sort((a, b) => b.count - a.count);
    this.metrics.topErrors = this.metrics.topErrors.slice(0, 10);

    // Calculate errors in last hour
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
    this.metrics.lastHour = Array.from(this.logs.values())
      .filter(e => e.context.timestamp >= oneHourAgo).length;

    // Calculate error rate (errors per minute over last hour)
    this.metrics.errorRate = this.metrics.lastHour / 60;
  }

  private recalculateMetrics(): void {
    this.metrics = {
      totalErrors: 0,
      errorsByLevel: {},
      errorsByTag: {},
      topErrors: [],
      errorRate: 0,
      lastHour: 0,
      resolved: 0
    };

    Array.from(this.logs.values()).forEach(errorLog => {
      this.updateMetrics(errorLog);
      if (errorLog.resolved) {
        this.metrics.resolved++;
      }
    });
  }

  private async cacheError(errorLog: ErrorLog): Promise<void> {
    try {
      await multiLayerCache.set(
        `error:${errorLog.id}`,
        errorLog,
        {
          type: 'analysis',
          ttl: 86400, // 24 hours
          tags: ['error', 'logging', ...errorLog.tags]
        }
      );

      // Cache error summary for quick access
      await multiLayerCache.set(
        'error:metrics',
        this.metrics,
        {
          type: 'analysis',
          ttl: 300, // 5 minutes
          tags: ['error', 'metrics']
        }
      );
    } catch (error) {
      console.error('Failed to cache error:', error);
    }
  }

  private async sendToVSCode(errorLog: ErrorLog): Promise<void> {
    try {
      // Send error to VS Code extension via WebSocket or HTTP
      if (typeof window !== 'undefined' && (window as any).vscode) {
        (window as any).vscode.postMessage({
          type: 'error',
          data: {
            id: errorLog.id,
            level: errorLog.level,
            message: errorLog.message,
            stack: errorLog.stack,
            context: errorLog.context,
            tags: errorLog.tags,
            metadata: errorLog.metadata
          }
        });
      } else {
        // HTTP fallback for server-side errors
        await fetch('/api/vscode/error', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(errorLog)
        }).catch(() => {}); // Ignore failures
      }
    } catch (error) {
      console.error('Failed to send error to VS Code:', error);
    }
  }

  private async notifyVSCodeResolution(errorLog: ErrorLog): Promise<void> {
    try {
      if (typeof window !== 'undefined' && (window as any).vscode) {
        (window as any).vscode.postMessage({
          type: 'error-resolved',
          data: {
            id: errorLog.id,
            resolvedAt: errorLog.resolvedAt,
            resolvedBy: errorLog.resolvedBy
          }
        });
      }
    } catch (error) {
      console.error('Failed to notify VS Code of resolution:', error);
    }
  }

  private async sendToRemoteLogger(errorLog: ErrorLog): Promise<void> {
    try {
      await fetch('/api/logging/error', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(errorLog)
      });
    } catch (error) {
      console.error('Failed to send to remote logger:', error);
    }
  }

  private async broadcastError(errorLog: ErrorLog): Promise<void> {
    try {
      await rabbitmqService.broadcastUpdate('error_logged', {
        id: errorLog.id,
        level: errorLog.level,
        message: errorLog.message,
        tags: errorLog.tags,
        timestamp: errorLog.context.timestamp
      });
    } catch (error) {
      console.error('Failed to broadcast error event:', error);
    }
  }
}

// Export singleton instance
export const errorLogger = new ErrorLogger(
  10000, // maxLogs
  process.env.NODE_ENV === 'development', // enableVSCodeIntegration
  process.env.NODE_ENV === 'production' // enableRemoteLogging
);

// Helper functions for common error patterns
export function withErrorLogging<T extends any[], R>(
  fn: (...args: T) => R,
  context: string,
  tags: string[] = []
): (...args: T) => R {
  return (...args: T): R => {
    try {
      const result = fn(...args);
      
      // Handle promises
      if (result instanceof Promise) {
        return result.catch((error: Error) => {
          errorLogger.logError(`${context} failed`, error, { args }, tags);
          throw error;
        }) as R;
      }
      
      return result;
    } catch (error) {
      errorLogger.logError(`${context} failed`, error as Error, { args }, tags);
      throw error;
    }
  };
}

export function trackAsyncOperation<T>(
  operation: Promise<T>,
  operationName: string,
  tags: string[] = []
): Promise<T> {
  const startTime = Date.now();
  
  return operation
    .then((result) => {
      errorLogger.logInfo(`${operationName} completed`, {
        duration: Date.now() - startTime,
        success: true
      }, [...tags, 'operation', 'success']);
      return result;
    })
    .catch((error) => {
      errorLogger.logError(`${operationName} failed`, error, {
        duration: Date.now() - startTime
      }, [...tags, 'operation', 'failure']);
      throw error;
    });
}