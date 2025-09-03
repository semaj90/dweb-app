/**
 * Production-Ready Logging Service for Legal AI Platform
 * Enhanced logging with Windows support, structured logging, and performance monitoring
 */

const dev = import.meta.env.NODE_ENV === 'development';
import { getConfig } from '../config/unified-config.js';
import type { LoggingConfig } from '../config/unified-config.js';

// Log levels in order of severity
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogContext {
  userId?: string;
  sessionId?: string;
  requestId?: string;
  userAgent?: string;
  ip?: string;
  endpoint?: string;
  method?: string;
  duration?: number;
  statusCode?: number;
  component?: string;
  service?: string;
}

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  context?: LogContext;
  error?: {
    name: string;
    message: string;
    stack?: string;
    code?: string | number;
  };
  metadata?: Record<string, any>;
  tags?: string[];
  performance?: {
    memoryUsage: NodeJS.MemoryUsage;
    timing: number;
    cpuUsage?: NodeJS.CpuUsage;
  };
}

export interface LogMetrics {
  totalLogs: number;
  logsByLevel: Record<LogLevel, number>;
  errorRate: number;
  averageResponseTime: number;
  memoryTrend: number[];
  windowsSpecific?: {
    gpuMemoryUsage?: number;
    diskIOMetrics?: any;
  };
}

// Windows-specific performance monitoring
class WindowsPerformanceMonitor {
  private isWindows = typeof process !== 'undefined' && process.platform === 'win32';
  private performanceCounters: Map<string, any> = new Map();

  async getWindowsMetrics(): Promise<any> {
    if (!this.isWindows) return null;

    try {
      // Check for Windows-specific GPU info (if nvidia-smi available)
      const gpuInfo = await this.getGPUInfo();
      
      // Get process-specific metrics
      const processMetrics = await this.getProcessMetrics();
      
      return {
        gpu: gpuInfo,
        process: processMetrics,
        platform: {
          osVersion: require('os').release(),
          totalMemory: Math.round(require('os').totalmem() / 1024 / 1024),
          freeMemory: Math.round(require('os').freemem() / 1024 / 1024)
        }
      };
    } catch (error: any) {
      return { error: 'Failed to collect Windows metrics' };
    }
  }

  private async getGPUInfo(): Promise<any> {
    try {
      const { spawn } = await import('child_process');
      
      return new Promise((resolve) => {
        const child = spawn('nvidia-smi', [
          '--query-gpu=memory.total,memory.used,temperature.gpu,utilization.gpu',
          '--format=csv,noheader,nounits'
        ], { stdio: 'pipe', shell: true });

        let output = '';
        child.stdout?.on('data', (data) => {
          output += data.toString();
        });

        child.on('close', (code) => {
          if (code === 0 && output.trim()) {
            const [memTotal, memUsed, temp, util] = output.trim().split(', ').map(Number);
            resolve({
              memoryTotal: memTotal,
              memoryUsed: memUsed,
              temperature: temp,
              utilization: util
            });
          } else {
            resolve(null);
          }
        });

        child.on('error', () => resolve(null));
      });
    } catch {
      return null;
    }
  }

  private async getProcessMetrics(): Promise<any> {
    if (!this.isWindows || typeof process === 'undefined') return null;

    try {
      const memUsage = process.memoryUsage();
      const cpuUsage = process.cpuUsage();
      
      return {
        pid: process.pid,
        uptime: process.uptime(),
        memoryUsage: {
          rss: Math.round(memUsage.rss / 1024 / 1024), // MB
          heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024), // MB
          heapTotal: Math.round(memUsage.heapTotal / 1024 / 1024), // MB
          external: Math.round(memUsage.external / 1024 / 1024) // MB
        },
        cpuUsage: {
          user: cpuUsage.user,
          system: cpuUsage.system
        }
      };
    } catch {
      return null;
    }
  }
}

// Main logging service
export class ProductionLogger {
  private config: LoggingConfig;
  private logBuffer: LogEntry[] = [];
  private metrics: LogMetrics = {
    totalLogs: 0,
    logsByLevel: { debug: 0, info: 0, warn: 0, error: 0 },
    errorRate: 0,
    averageResponseTime: 0,
    memoryTrend: []
  };
  private windowsMonitor: WindowsPerformanceMonitor;
  private flushInterval: NodeJS.Timeout | null = null;
  private metricsInterval: NodeJS.Timeout | null = null;

  constructor() {
    this.config = getConfig().logging;
    this.windowsMonitor = new WindowsPerformanceMonitor();
    this.initializeLogging();
  }

  private initializeLogging(): void {
    // Setup periodic log flushing for file output
    if (this.config.outputs.includes('file') && this.config.file) {
      this.flushInterval = setInterval(() => {
        this.flushBufferedLogs();
      }, 5000); // Flush every 5 seconds
    }

    // Setup metrics collection
    this.metricsInterval = setInterval(async () => {
      await this.collectMetrics();
    }, 30000); // Collect metrics every 30 seconds

    // Graceful shutdown handling
    if (typeof process !== 'undefined') {
      process.on('SIGINT', () => this.shutdown());
      process.on('SIGTERM', () => this.shutdown());
    }
  }

  // Core logging methods
  public debug(message: string, context?: LogContext, metadata?: Record<string, any>): void {
    if (this.shouldLog('debug')) {
      this.log('debug', message, context, metadata);
    }
  }

  public info(message: string, context?: LogContext, metadata?: Record<string, any>): void {
    if (this.shouldLog('info')) {
      this.log('info', message, context, metadata);
    }
  }

  public warn(message: string, context?: LogContext, metadata?: Record<string, any>): void {
    if (this.shouldLog('warn')) {
      this.log('warn', message, context, metadata);
    }
  }

  public error(message: string, error?: Error, context?: LogContext, metadata?: Record<string, any>): void {
    if (this.shouldLog('error')) {
      const errorInfo = error ? {
        name: error.name,
        message: error.message,
        stack: this.config.includeStack ? error.stack : undefined,
        code: (error as any).code
      } : undefined;

      this.log('error', message, context, metadata, errorInfo);
    }
  }

  // Specialized logging methods
  public apiRequest(
    method: string, 
    endpoint: string, 
    statusCode: number, 
    duration: number,
    context?: Partial<LogContext>
  ): void {
    const level: LogLevel = statusCode >= 500 ? 'error' : statusCode >= 400 ? 'warn' : 'info';
    
    this.log(level, `${method} ${endpoint} ${statusCode}`, {
      ...context,
      method,
      endpoint,
      statusCode,
      duration
    }, {
      responseTime: duration,
      httpStatus: statusCode
    });
  }

  public security(event: string, context?: LogContext, metadata?: Record<string, any>): void {
    this.log('warn', `Security Event: ${event}`, context, {
      ...metadata,
      securityEvent: true,
      timestamp: new Date().toISOString()
    }, undefined, ['security']);
  }

  public performance(operation: string, duration: number, context?: LogContext): void {
    const level: LogLevel = duration > 5000 ? 'warn' : duration > 1000 ? 'info' : 'debug';
    
    this.log(level, `Performance: ${operation}`, context, {
      operation,
      duration,
      performanceLog: true
    }, undefined, ['performance']);
  }

  public windowsSpecific(event: string, data?: any, context?: LogContext): void {
    if (typeof process !== 'undefined' && process.platform === 'win32') {
      this.log('info', `Windows: ${event}`, context, {
        ...data,
        windowsEvent: true,
        platform: process.platform
      }, undefined, ['windows']);
    }
  }

  // Core logging implementation
  private log(
    level: LogLevel, 
    message: string, 
    context?: LogContext, 
    metadata?: Record<string, any>,
    error?: LogEntry['error'],
    tags: string[] = []
  ): void {
    const logEntry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      context,
      error,
      metadata,
      tags: tags.length > 0 ? tags : undefined,
      performance: this.includePerformanceData() ? {
        memoryUsage: typeof process !== 'undefined' ? process.memoryUsage() : {} as any,
        timing: Date.now(),
        cpuUsage: typeof process !== 'undefined' ? process.cpuUsage() : undefined
      } : undefined
    };

    // Update metrics
    this.updateMetrics(logEntry);

    // Output to configured destinations
    this.outputLog(logEntry);
  }

  private shouldLog(level: LogLevel): boolean {
    const levels: LogLevel[] = ['debug', 'info', 'warn', 'error'];
    const configLevel = this.config.level;
    return levels.indexOf(level) >= levels.indexOf(configLevel);
  }

  private includePerformanceData(): boolean {
    return dev || this.config.level === 'debug';
  }

  private updateMetrics(entry: LogEntry): void {
    this.metrics.totalLogs++;
    this.metrics.logsByLevel[entry.level]++;

    // Update error rate
    if (entry.level === 'error') {
      this.metrics.errorRate = this.metrics.logsByLevel.error / this.metrics.totalLogs;
    }

    // Update response time (if available)
    if (entry.context?.duration) {
      const currentAvg = this.metrics.averageResponseTime;
      const newCount = this.metrics.totalLogs;
      this.metrics.averageResponseTime = (currentAvg * (newCount - 1) + entry.context.duration) / newCount;
    }

    // Update memory trend (keep last 100 entries)
    if (entry.performance?.memoryUsage?.rss) {
      this.metrics.memoryTrend.push(entry.performance.memoryUsage.rss);
      if (this.metrics.memoryTrend.length > 100) {
        this.metrics.memoryTrend.shift();
      }
    }
  }

  private outputLog(entry: LogEntry): void {
    // Console output
    if (this.config.outputs.includes('console')) {
      this.outputToConsole(entry);
    }

    // File output (buffered)
    if (this.config.outputs.includes('file')) {
      this.logBuffer.push(entry);
      
      // Immediate flush for errors
      if (entry.level === 'error' && this.config.file) {
        this.flushBufferedLogs();
      }
    }

    // Syslog output (Windows Event Log if on Windows)
    if (this.config.outputs.includes('syslog')) {
      this.outputToSyslog(entry);
    }
  }

  private outputToConsole(entry: LogEntry): void {
    const formattedMessage = this.formatLogEntry(entry, 'console');
    
    switch (entry.level) {
      case 'debug':
        console.debug(formattedMessage);
        break;
      case 'info':
        console.info(formattedMessage);
        break;
      case 'warn':
        console.warn(formattedMessage);
        break;
      case 'error':
        console.error(formattedMessage);
        break;
    }
  }

  private formatLogEntry(entry: LogEntry, format: 'console' | 'file' = 'console'): string {
    if (this.config.format === 'json') {
      return JSON.stringify(entry, null, format === 'console' && dev ? 2 : 0);
    } else {
      // Text format
      const parts = [
        entry.timestamp,
        `[${entry.level.toUpperCase().padEnd(5)}]`,
        entry.context?.component ? `[${entry.context.component}]` : '',
        entry.message
      ].filter(Boolean);

      let formatted = parts.join(' ');

      // Add context information
      if (entry.context && Object.keys(entry.context).length > 0) {
        const contextStr = Object.entries(entry.context)
          .filter(([key]) => key !== 'component')
          .map(([key, value]) => `${key}=${value}`)
          .join(' ');
        
        if (contextStr) {
          formatted += ` | ${contextStr}`;
        }
      }

      // Add error information
      if (entry.error) {
        formatted += ` | ERROR: ${entry.error.name}: ${entry.error.message}`;
        if (entry.error.stack && this.config.includeStack) {
          formatted += `\nStack: ${entry.error.stack}`;
        }
      }

      // Add metadata
      if (entry.metadata && Object.keys(entry.metadata).length > 0) {
        formatted += ` | ${JSON.stringify(entry.metadata)}`;
      }

      return formatted;
    }
  }

  private async flushBufferedLogs(): Promise<void> {
    if (this.logBuffer.length === 0 || !this.config.file) return;

    try {
      const fs = await import('fs');
      const path = await import('path');

      // Ensure log directory exists
      const logDir = path.dirname(this.config.file.path);
      if (!fs.existsSync(logDir)) {
        fs.mkdirSync(logDir, { recursive: true });
      }

      // Prepare log entries for writing
      const logEntries = this.logBuffer.splice(0); // Remove all entries from buffer
      const logData = logEntries.map(entry => this.formatLogEntry(entry, 'file')).join('\n') + '\n';

      // Write to log file
      fs.appendFileSync(this.config.file.path, logData, 'utf8');

      // Handle log rotation if needed
      if (this.config.file.rotate) {
        await this.handleLogRotation();
      }

    } catch (error: any) {
      console.error('Failed to flush logs to file:', error);
      // Re-add entries to buffer for retry
      this.logBuffer.unshift(...this.logBuffer);
    }
  }

  private async handleLogRotation(): Promise<void> {
    if (!this.config.file) return;

    try {
      const fs = await import('fs');
      const path = await import('path');

      const stats = fs.statSync(this.config.file.path);
      const maxSizeBytes = this.parseSize(this.config.file.maxSize);

      if (stats.size > maxSizeBytes) {
        const baseName = path.basename(this.config.file.path, path.extname(this.config.file.path));
        const extension = path.extname(this.config.file.path);
        const dirName = path.dirname(this.config.file.path);

        // Rotate existing files
        for (let i = this.config.file.maxFiles - 1; i >= 1; i--) {
          const oldFile = path.join(dirName, `${baseName}.${i}${extension}`);
          const newFile = path.join(dirName, `${baseName}.${i + 1}${extension}`);
          
          if (fs.existsSync(oldFile)) {
            if (i === this.config.file.maxFiles - 1) {
              fs.unlinkSync(oldFile); // Remove oldest
            } else {
              fs.renameSync(oldFile, newFile);
            }
          }
        }

        // Move current file to .1
        const rotatedFile = path.join(dirName, `${baseName}.1${extension}`);
        fs.renameSync(this.config.file.path, rotatedFile);

      }
    } catch (error: any) {
      console.error('Log rotation failed:', error);
    }
  }

  private parseSize(sizeStr: string): number {
    const units: Record<string, number> = { 'B': 1, 'K': 1024, 'M': 1024 * 1024, 'G': 1024 * 1024 * 1024 };
    const match = sizeStr.match(/^(\d+)([BKMG]?)$/i);
    if (!match) return 10 * 1024 * 1024; // Default 10MB
    
    const size = parseInt(match[1]);
    const unit = (match[2] || 'B').toUpperCase();
    return size * (units[unit] || 1);
  }

  private outputToSyslog(entry: LogEntry): void {
    // Windows Event Log integration would go here
    // For now, just output to console with syslog format
    if (typeof process !== 'undefined' && process.platform === 'win32') {
      this.windowsSpecific(`Syslog: ${entry.message}`, { level: entry.level });
    }
  }

  private async collectMetrics(): Promise<void> {
    try {
      // Collect Windows-specific metrics
      if (typeof process !== 'undefined' && process.platform === 'win32') {
        const windowsMetrics = await this.windowsMonitor.getWindowsMetrics();
        if (windowsMetrics) {
          this.metrics.windowsSpecific = windowsMetrics;
        }
      }

      // Log metrics summary (debug level)
      this.debug('Logger metrics collected', undefined, {
        metrics: {
          totalLogs: this.metrics.totalLogs,
          errorRate: Math.round(this.metrics.errorRate * 100) / 100,
          averageResponseTime: Math.round(this.metrics.averageResponseTime),
          memoryTrendSize: this.metrics.memoryTrend.length
        }
      });

    } catch (error: any) {
      this.error('Failed to collect logger metrics', error instanceof Error ? error : undefined);
    }
  }

  // Public API methods
  public getMetrics(): LogMetrics {
    return { ...this.metrics };
  }

  public async getHealthStatus(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    details: any;
  }> {
    const details = {
      totalLogs: this.metrics.totalLogs,
      errorRate: this.metrics.errorRate,
      bufferSize: this.logBuffer.length,
      outputs: this.config.outputs,
      level: this.config.level
    };

    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    
    if (this.metrics.errorRate > 0.1) { // 10% error rate
      status = 'degraded';
    }
    
    if (this.logBuffer.length > 1000) { // Too many buffered logs
      status = 'degraded';
    }
    
    if (this.metrics.errorRate > 0.5) { // 50% error rate
      status = 'unhealthy';
    }

    return { status, details };
  }

  public clearLogs(): void {
    this.logBuffer = [];
    this.metrics = {
      totalLogs: 0,
      logsByLevel: { debug: 0, info: 0, warn: 0, error: 0 },
      errorRate: 0,
      averageResponseTime: 0,
      memoryTrend: []
    };
  }

  public shutdown(): void {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
    }

    // Final flush
    if (this.logBuffer.length > 0) {
      this.flushBufferedLogs();
    }

    this.info('Logger shutdown completed');
  }
}

// Singleton instance
export const logger = new ProductionLogger();

// Convenience exports
export const logDebug = (message: string, context?: LogContext, metadata?: Record<string, any>) =>
  logger.debug(message, context, metadata);

export const logInfo = (message: string, context?: LogContext, metadata?: Record<string, any>) =>
  logger.info(message, context, metadata);

export const logWarn = (message: string, context?: LogContext, metadata?: Record<string, any>) =>
  logger.warn(message, context, metadata);

export const logError = (message: string, error?: Error, context?: LogContext, metadata?: Record<string, any>) =>
  logger.error(message, error, context, metadata);

export const logApiRequest = (method: string, endpoint: string, statusCode: number, duration: number, context?: Partial<LogContext>) =>
  logger.apiRequest(method, endpoint, statusCode, duration, context);

export const logSecurity = (event: string, context?: LogContext, metadata?: Record<string, any>) =>
  logger.security(event, context, metadata);

export const logPerformance = (operation: string, duration: number, context?: LogContext) =>
  logger.performance(operation, duration, context);

// Named and default exports
export const productionLogger = logger;
export default logger;