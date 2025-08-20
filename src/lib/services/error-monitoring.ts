// Error Monitoring and Logging Service
// Production-grade error handling with detailed logging and monitoring capabilities

import { browser } from '$app/environment';
import { writable } from 'svelte/store';

// Types
export interface ErrorLog {
  id: string;
  timestamp: Date;
  level: 'error' | 'warning' | 'info' | 'debug';
  message: string;
  details?: any;
  stack?: string;
  url?: string;
  userAgent?: string;
  userId?: string;
  sessionId?: string;
  context?: Record<string, any>;
  resolved?: boolean;
}

export interface PerformanceMetric {
  id: string;
  timestamp: Date;
  operation: string;
  duration: number;
  success: boolean;
  metadata?: Record<string, any>;
}

export interface DatabaseError {
  id: string;
  timestamp: Date;
  query?: string;
  errorCode?: string;
  message: string;
  stack?: string;
  context?: Record<string, any>;
}

export interface ApiError {
  id: string;
  timestamp: Date;
  endpoint: string;
  method: string;
  statusCode: number;
  message: string;
  requestBody?: any;
  responseBody?: any;
  duration: number;
  userAgent?: string;
  ipAddress?: string;
}

// Stores for reactive error monitoring
export const errorLogs = writable<ErrorLog[]>([]);
export const performanceMetrics = writable<PerformanceMetric[]>([]);
export const isMonitoringEnabled = writable(true);

// Error monitoring service
class ErrorMonitoringService {
  private sessionId: string;
  private userId?: string;
  private errorBuffer: ErrorLog[] = [];
  private performanceBuffer: PerformanceMetric[] = [];
  private flushInterval: NodeJS.Timeout | null = null;
  private maxBufferSize = 100;
  private flushIntervalMs = 30000; // 30 seconds

  constructor() {
    this.sessionId = this.generateSessionId();
    
    if (browser) {
      this.initializeErrorHandlers();
      this.startPeriodicFlush();
    }
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private initializeErrorHandlers(): void {
    // Global error handler
    window.addEventListener('error', (event) => {
      this.logError({
        level: 'error',
        message: event.message || 'Uncaught JavaScript error',
        details: {
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno,
          error: event.error
        },
        stack: event.error?.stack,
        url: window.location.href
      });
    });

    // Unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
      this.logError({
        level: 'error',
        message: 'Unhandled promise rejection',
        details: {
          reason: event.reason,
          promise: event.promise
        },
        stack: event.reason?.stack,
        url: window.location.href
      });
    });

    // Console error override for capturing console.error calls
    const originalConsoleError = console.error;
    console.error = (...args) => {
      this.logError({
        level: 'error',
        message: args.join(' '),
        details: { consoleArgs: args },
        url: window.location.href
      });
      originalConsoleError.apply(console, args);
    };
  }

  private startPeriodicFlush(): void {
    this.flushInterval = setInterval(() => {
      this.flushBuffers();
    }, this.flushIntervalMs);
  }

  public setUserId(userId: string): void {
    this.userId = userId;
  }

  public logError(error: Partial<ErrorLog>): void {
    const errorLog: ErrorLog = {
      id: this.generateId(),
      timestamp: new Date(),
      level: error.level || 'error',
      message: error.message || 'Unknown error',
      details: error.details,
      stack: error.stack,
      url: error.url || (browser ? window.location.href : undefined),
      userAgent: browser ? navigator.userAgent : undefined,
      userId: this.userId,
      sessionId: this.sessionId,
      context: error.context,
      resolved: false
    };

    this.errorBuffer.push(errorLog);
    
    // Update reactive store
    errorLogs.update(logs => [errorLog, ...logs].slice(0, 1000)); // Keep last 1000 errors

    // Flush immediately for critical errors
    if (error.level === 'error') {
      this.flushBuffers();
    }

    // Auto-flush if buffer is full
    if (this.errorBuffer.length >= this.maxBufferSize) {
      this.flushBuffers();
    }
  }

  public logPerformance(metric: Partial<PerformanceMetric>): void {
    const performanceMetric: PerformanceMetric = {
      id: this.generateId(),
      timestamp: new Date(),
      operation: metric.operation || 'unknown',
      duration: metric.duration || 0,
      success: metric.success ?? true,
      metadata: metric.metadata
    };

    this.performanceBuffer.push(performanceMetric);
    
    // Update reactive store
    performanceMetrics.update(metrics => [performanceMetric, ...metrics].slice(0, 1000));

    // Auto-flush if buffer is full
    if (this.performanceBuffer.length >= this.maxBufferSize) {
      this.flushBuffers();
    }
  }

  public logDatabaseError(error: Partial<DatabaseError>): void {
    this.logError({
      level: 'error',
      message: `Database Error: ${error.message}`,
      details: {
        type: 'database',
        query: error.query,
        errorCode: error.errorCode,
        originalError: error
      },
      stack: error.stack,
      context: error.context
    });
  }

  public logApiError(error: Partial<ApiError>): void {
    this.logError({
      level: 'error',
      message: `API Error: ${error.endpoint} ${error.method} - ${error.statusCode}`,
      details: {
        type: 'api',
        endpoint: error.endpoint,
        method: error.method,
        statusCode: error.statusCode,
        requestBody: error.requestBody,
        responseBody: error.responseBody,
        duration: error.duration,
        ipAddress: error.ipAddress
      },
      userAgent: error.userAgent
    });
  }

  public async measurePerformance<T>(
    operation: string,
    fn: () => Promise<T>,
    metadata?: Record<string, any>
  ): Promise<T> {
    const start = performance.now();
    let success = true;
    let result: T;

    try {
      result = await fn();
      return result;
    } catch (error) {
      success = false;
      this.logError({
        level: 'error',
        message: `Performance measurement failed for ${operation}`,
        details: { error, metadata },
        stack: error instanceof Error ? error.stack : undefined
      });
      throw error;
    } finally {
      const duration = performance.now() - start;
      this.logPerformance({
        operation,
        duration,
        success,
        metadata
      });
    }
  }

  public measurePerformanceSync<T>(
    operation: string,
    fn: () => T,
    metadata?: Record<string, any>
  ): T {
    const start = performance.now();
    let success = true;
    let result: T;

    try {
      result = fn();
      return result;
    } catch (error) {
      success = false;
      this.logError({
        level: 'error',
        message: `Performance measurement failed for ${operation}`,
        details: { error, metadata },
        stack: error instanceof Error ? error.stack : undefined
      });
      throw error;
    } finally {
      const duration = performance.now() - start;
      this.logPerformance({
        operation,
        duration,
        success,
        metadata
      });
    }
  }

  private async flushBuffers(): Promise<void> {
    if (this.errorBuffer.length === 0 && this.performanceBuffer.length === 0) {
      return;
    }

    try {
      // Send to monitoring endpoint
      const payload = {
        sessionId: this.sessionId,
        userId: this.userId,
        timestamp: new Date().toISOString(),
        errors: [...this.errorBuffer],
        performance: [...this.performanceBuffer],
        metadata: {
          userAgent: browser ? navigator.userAgent : undefined,
          url: browser ? window.location.href : undefined,
          timestamp: new Date().toISOString()
        }
      };

      // Send to backend monitoring endpoint
      const response = await fetch('/api/monitoring/logs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        // Clear buffers after successful send
        this.errorBuffer = [];
        this.performanceBuffer = [];
      } else {
        console.warn('Failed to send monitoring data:', response.statusText);
      }
    } catch (error) {
      console.warn('Error flushing monitoring buffers:', error);
    }
  }

  public async getErrorSummary(): Promise<{
    totalErrors: number;
    errorsByLevel: Record<string, number>;
    recentErrors: ErrorLog[];
    topErrors: { message: string; count: number }[];
  }> {
    try {
      const response = await fetch('/api/monitoring/summary');
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      this.logError({
        level: 'error',
        message: 'Failed to get error summary',
        details: { error }
      });
    }

    return {
      totalErrors: 0,
      errorsByLevel: {},
      recentErrors: [],
      topErrors: []
    };
  }

  public async getPerformanceMetrics(): Promise<{
    averageResponseTime: number;
    slowestOperations: PerformanceMetric[];
    operationCounts: Record<string, number>;
    successRate: number;
  }> {
    try {
      const response = await fetch('/api/monitoring/performance');
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      this.logError({
        level: 'error',
        message: 'Failed to get performance metrics',
        details: { error }
      });
    }

    return {
      averageResponseTime: 0,
      slowestOperations: [],
      operationCounts: {},
      successRate: 1
    };
  }

  private generateId(): string {
    return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  public destroy(): void {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
      this.flushInterval = null;
    }
    
    // Final flush
    this.flushBuffers();
  }
}

// Export singleton instance
export const errorMonitoring = new ErrorMonitoringService();

// Utility functions for common error patterns
export const handleApiError = (error: any, context?: Record<string, any>) => {
  errorMonitoring.logError({
    level: 'error',
    message: `API Error: ${error.message || 'Unknown API error'}`,
    details: {
      type: 'api',
      error: error,
      status: error.status,
      statusText: error.statusText
    },
    stack: error.stack,
    context
  });
};

export const handleDatabaseError = (error: any, query?: string, context?: Record<string, any>) => {
  errorMonitoring.logDatabaseError({
    message: error.message || 'Database operation failed',
    query,
    errorCode: error.code,
    stack: error.stack,
    context
  });
};

export const handleValidationError = (error: any, data?: any, context?: Record<string, any>) => {
  errorMonitoring.logError({
    level: 'warning',
    message: `Validation Error: ${error.message || 'Invalid data'}`,
    details: {
      type: 'validation',
      error: error,
      data: data,
      validationErrors: error.errors || error.issues
    },
    context
  });
};

export const logInfo = (message: string, details?: any, context?: Record<string, any>) => {
  errorMonitoring.logError({
    level: 'info',
    message,
    details,
    context
  });
};

export const logWarning = (message: string, details?: any, context?: Record<string, any>) => {
  errorMonitoring.logError({
    level: 'warning',
    message,
    details,
    context
  });
};

export const logDebug = (message: string, details?: any, context?: Record<string, any>) => {
  if (process.env.NODE_ENV === 'development') {
    errorMonitoring.logError({
      level: 'debug',
      message,
      details,
      context
    });
  }
};

// Performance monitoring decorators and helpers
export const withPerformanceMonitoring = <T extends (...args: any[]) => Promise<any>>(
  fn: T,
  operationName?: string
): T => {
  return (async (...args: any[]) => {
    const operation = operationName || fn.name || 'anonymous';
    return await errorMonitoring.measurePerformance(operation, () => fn(...args));
  }) as T;
};

export const withErrorBoundary = <T extends (...args: any[]) => any>(
  fn: T,
  fallback?: any,
  context?: Record<string, any>
): T => {
  return ((...args: any[]) => {
    try {
      const result = fn(...args);
      
      // Handle async functions
      if (result && typeof result.catch === 'function') {
        return result.catch((error: any) => {
          errorMonitoring.logError({
            level: 'error',
            message: `Error in ${fn.name || 'anonymous function'}`,
            details: { error, args },
            stack: error.stack,
            context
          });
          return fallback;
        });
      }
      
      return result;
    } catch (error) {
      errorMonitoring.logError({
        level: 'error',
        message: `Error in ${fn.name || 'anonymous function'}`,
        details: { error, args },
        stack: error instanceof Error ? error.stack : undefined,
        context
      });
      return fallback;
    }
  }) as T;
};

// Database query monitoring wrapper
export const monitoredQuery = async <T>(
  queryFn: () => Promise<T>,
  queryDescription: string,
  context?: Record<string, any>
): Promise<T> => {
  return await errorMonitoring.measurePerformance(
    `db_query_${queryDescription}`,
    async () => {
      try {
        return await queryFn();
      } catch (error) {
        handleDatabaseError(error, queryDescription, context);
        throw error;
      }
    },
    { queryDescription, ...context }
  );
};

// Vector search monitoring
export const monitoredVectorSearch = async <T>(
  searchFn: () => Promise<T>,
  query: string,
  searchType: string,
  context?: Record<string, any>
): Promise<T> => {
  return await errorMonitoring.measurePerformance(
    `vector_search_${searchType}`,
    searchFn,
    { query: query.substring(0, 100), searchType, ...context }
  );
};

// Cleanup function for browser environments
if (browser) {
  window.addEventListener('beforeunload', () => {
    errorMonitoring.destroy();
  });
}

export default errorMonitoring;