// Logging Types
export interface LogLevel {
  level: 'debug' | 'info' | 'warn' | 'error' | 'fatal';
  timestamp: string;
  message: string;
  context?: Record<string, unknown>;
}

export interface Logger {
  debug(message: string, context?: Record<string, unknown>): void;
  info(message: string, context?: Record<string, unknown>): void;
  warn(message: string, context?: Record<string, unknown>): void;
  error(message: string, context?: Record<string, unknown>): void;
  fatal(message: string, context?: Record<string, unknown>): void;
}

export interface LogEntry {
  id: string;
  level: LogLevel['level'];
  message: string;
  timestamp: string;
  userId?: string;
  sessionId?: string;
  context?: Record<string, unknown>;
  stack?: string;
}

export interface LoggerConfig {
  level: LogLevel['level'];
  enableConsole: boolean;
  enableFile: boolean;
  filePath?: string;
  maxFileSize?: number;
  maxFiles?: number;
}