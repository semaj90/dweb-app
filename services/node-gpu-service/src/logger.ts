/**
 * Logger - Enhanced logging utility with structured output
 */

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3
}

interface LogContext {
  timestamp: string;
  level: string;
  message: string;
  data?: unknown;
  metadata?: {
    service: string;
    version: string;
    pid: number;
  };
}

export class Logger {
  private level: LogLevel;
  private serviceName: string;
  private version: string;

  constructor(logLevel: string = 'info', serviceName: string = 'node-gpu-service') {
    this.level = this.parseLogLevel(logLevel);
    this.serviceName = serviceName;
    this.version = process.env.npm_package_version || '1.0.0';
  }

  private parseLogLevel(level: string): LogLevel {
    switch (level.toLowerCase()) {
      case 'debug': return LogLevel.DEBUG;
      case 'info': return LogLevel.INFO;
      case 'warn': return LogLevel.WARN;
      case 'error': return LogLevel.ERROR;
      default: return LogLevel.INFO;
    }
  }

  private shouldLog(level: LogLevel): boolean {
    return level >= this.level;
  }

  private formatMessage(level: LogLevel, message: string, data?: unknown): string {
    const context: LogContext = {
      timestamp: new Date().toISOString(),
      level: LogLevel[level],
      message,
      metadata: {
        service: this.serviceName,
        version: this.version,
        pid: process.pid
      }
    };

    if (data !== undefined) {
      context.data = data;
    }

    return JSON.stringify(context, null, 2);
  }

  debug(message: string, data?: unknown): void {
    if (!this.shouldLog(LogLevel.DEBUG)) return;
    
    const formatted = this.formatMessage(LogLevel.DEBUG, message, data);
    console.log(`\x1b[36m${formatted}\x1b[0m`); // Cyan
  }

  info(message: string, data?: unknown): void {
    if (!this.shouldLog(LogLevel.INFO)) return;
    
    const formatted = this.formatMessage(LogLevel.INFO, message, data);
    console.log(`\x1b[32m${formatted}\x1b[0m`); // Green
  }

  warn(message: string, data?: unknown): void {
    if (!this.shouldLog(LogLevel.WARN)) return;
    
    const formatted = this.formatMessage(LogLevel.WARN, message, data);
    console.warn(`\x1b[33m${formatted}\x1b[0m`); // Yellow
  }

  error(message: string, error?: unknown): void {
    if (!this.shouldLog(LogLevel.ERROR)) return;
    
    const errorData = error instanceof Error ? {
      name: error.name,
      message: error.message,
      stack: error.stack
    } : error;
    
    const formatted = this.formatMessage(LogLevel.ERROR, message, errorData);
    console.error(`\x1b[31m${formatted}\x1b[0m`); // Red
  }

  setLevel(level: string): void {
    this.level = this.parseLogLevel(level);
  }

  getLevel(): string {
    return LogLevel[this.level];
  }
}