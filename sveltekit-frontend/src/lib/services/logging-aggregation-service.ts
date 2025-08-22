/**
 * Centralized Logging Aggregation Service
 * Collects, formats, and routes log messages from all system components
 * Supports multiple transports: console, file, remote, database
 */

import { writable, type Writable } from 'svelte/store';
import { browser } from '$app/environment';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'fatal';

export interface LogEntry {
  id: string;
  timestamp: number;
  level: LogLevel;
  category: string;
  message: string;
  data?: unknown;
  error?: Error;
  service?: string;
  userId?: string;
  sessionId?: string;
  requestId?: string;
  tags?: string[];
  meta?: Record<string, any>;
}

export interface LogTransport {
  name: string;
  enabled: boolean;
  minLevel: LogLevel;
  format?: 'json' | 'text' | 'structured';
  send: (entry: LogEntry) => Promise<void>;
}

export interface LogFilter {
  category?: string[];
  level?: LogLevel[];
  service?: string[];
  tags?: string[];
  timeRange?: {
    start: number;
    end: number;
  };
}

export interface LogStats {
  totalEntries: number;
  entriesByLevel: Record<LogLevel, number>;
  entriesByCategory: Record<string, number>;
  entriesByService: Record<string, number>;
  recentErrors: LogEntry[];
  avgLogsPerMinute: number;
  lastEntry?: LogEntry;
}

class LoggingAggregationService {
  private entries: LogEntry[] = [];
  private transports: Map<string, LogTransport> = new Map();
  private maxEntries = 10000; // Keep in memory
  private sessionId: string;
  private userId?: string;
  
  // Stores
  public entriesStore: Writable<LogEntry[]> = writable([]);
  public statsStore: Writable<LogStats> = writable({
    totalEntries: 0,
    entriesByLevel: { debug: 0, info: 0, warn: 0, error: 0, fatal: 0 },
    entriesByCategory: {},
    entriesByService: {},
    recentErrors: [],
    avgLogsPerMinute: 0
  });

  // Configuration
  private config = {
    enableConsoleTransport: true,
    enableFileTransport: false,
    enableRemoteTransport: false,
    enableDatabaseTransport: false,
    minLevel: 'info' as LogLevel,
    bufferSize: 100,
    flushInterval: 5000,
    remoteEndpoint: '/api/logs',
    categories: {
      system: { enabled: true, color: '#007bff' },
      auth: { enabled: true, color: '#28a745' },
      api: { enabled: true, color: '#ffc107' },
      ai: { enabled: true, color: '#6f42c1' },
      database: { enabled: true, color: '#20c997' },
      frontend: { enabled: true, color: '#fd7e14' },
      backend: { enabled: true, color: '#dc3545' },
      security: { enabled: true, color: '#e83e8c' }
    }
  };

  private logBuffer: LogEntry[] = [];
  private flushTimer?: NodeJS.Timeout;

  constructor(config?: Partial<typeof LoggingAggregationService.prototype.config>) {
    this.sessionId = this.generateSessionId();
    
    if (config) {
      Object.assign(this.config, config);
    }

    this.initializeTransports();
    this.startFlushTimer();

    if (browser) {
      this.setupBrowserIntegration();
    }
  }

  private initializeTransports() {
    // Console transport
    if (this.config.enableConsoleTransport) {
      this.addTransport({
        name: 'console',
        enabled: true,
        minLevel: 'debug',
        format: 'structured',
        send: async (entry) => {
          const style = this.getConsoleStyle(entry.level, entry.category);
          const message = this.formatConsoleMessage(entry);
          
          switch (entry.level) {
            case 'debug':
              console.debug(message, entry.data);
              break;
            case 'info':
              console.info(message, entry.data);
              break;
            case 'warn':
              console.warn(message, entry.data);
              break;
            case 'error':
            case 'fatal':
              console.error(message, entry.data, entry.error);
              break;
          }
        }
      });
    }

    // Remote transport
    if (this.config.enableRemoteTransport) {
      this.addTransport({
        name: 'remote',
        enabled: true,
        minLevel: 'info',
        format: 'json',
        send: async (entry) => {
          try {
            await fetch(this.config.remoteEndpoint, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                entries: [entry],
                sessionId: this.sessionId,
                timestamp: Date.now()
              })
            });
          } catch (error) {
            // Fallback to console if remote fails
            console.error('Failed to send log to remote:', error);
          }
        }
      });
    }

    // Database transport (IndexedDB for browser)
    if (browser && this.config.enableDatabaseTransport) {
      this.addTransport({
        name: 'database',
        enabled: true,
        minLevel: 'warn',
        format: 'json',
        send: async (entry) => {
          try {
            await this.saveToIndexedDB(entry);
          } catch (error) {
            console.error('Failed to save log to IndexedDB:', error);
          }
        }
      });
    }
  }

  private setupBrowserIntegration() {
    // Capture unhandled errors
    window.addEventListener('error', (event) => {
      this.error('system', 'Unhandled JavaScript error', {
        message: event.message,
        filename: event.filename,
        line: event.lineno,
        column: event.colno,
        error: event.error
      });
    });

    // Capture unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.error('system', 'Unhandled promise rejection', {
        reason: event.reason
      });
    });

    // Capture console messages
    this.interceptConsole();
  }

  private interceptConsole() {
    const originalConsole = { ...console };

    console.log = (...args) => {
      this.debug('console', args.join(' '), { args });
      originalConsole.log(...args);
    };

    console.warn = (...args) => {
      this.warn('console', args.join(' '), { args });
      originalConsole.warn(...args);
    };

    console.error = (...args) => {
      this.error('console', args.join(' '), { args });
      originalConsole.error(...args);
    };
  }

  // Public logging methods
  public debug(category: string, message: string, data?: unknown, meta?: Record<string, any>) {
    this.log('debug', category, message, data, undefined, meta);
  }

  public info(category: string, message: string, data?: unknown, meta?: Record<string, any>) {
    this.log('info', category, message, data, undefined, meta);
  }

  public warn(category: string, message: string, data?: unknown, meta?: Record<string, any>) {
    this.log('warn', category, message, data, undefined, meta);
  }

  public error(category: string, message: string, data?: unknown, error?: Error, meta?: Record<string, any>) {
    this.log('error', category, message, data, error, meta);
  }

  public fatal(category: string, message: string, data?: unknown, error?: Error, meta?: Record<string, any>) {
    this.log('fatal', category, message, data, error, meta);
  }

  // Structured logging
  public logStructured(entry: Partial<LogEntry>) {
    const fullEntry: LogEntry = {
      id: this.generateEntryId(),
      timestamp: Date.now(),
      level: entry.level || 'info',
      category: entry.category || 'unknown',
      message: entry.message || '',
      sessionId: this.sessionId,
      userId: this.userId,
      ...entry
    };

    this.processLogEntry(fullEntry);
  }

  private log(
    level: LogLevel,
    category: string,
    message: string,
    data?: unknown,
    error?: Error,
    meta?: Record<string, any>
  ) {
    const entry: LogEntry = {
      id: this.generateEntryId(),
      timestamp: Date.now(),
      level,
      category,
      message,
      data,
      error,
      sessionId: this.sessionId,
      userId: this.userId,
      meta
    };

    this.processLogEntry(entry);
  }

  private processLogEntry(entry: LogEntry) {
    // Add to memory store
    this.entries.push(entry);
    
    // Trim if exceeding max entries
    if (this.entries.length > this.maxEntries) {
      this.entries = this.entries.slice(-this.maxEntries);
    }

    // Add to buffer for transport
    this.logBuffer.push(entry);

    // Update stores
    this.updateStores();

    // Send to transports immediately for error/fatal levels
    if (entry.level === 'error' || entry.level === 'fatal') {
      this.flushBufferImmediate();
    }
  }

  private updateStores() {
    this.entriesStore.set([...this.entries]);
    
    const stats = this.calculateStats();
    this.statsStore.set(stats);
  }

  private calculateStats(): LogStats {
    const entriesByLevel: Record<LogLevel, number> = {
      debug: 0,
      info: 0,
      warn: 0,
      error: 0,
      fatal: 0
    };

    const entriesByCategory: Record<string, number> = {};
    const entriesByService: Record<string, number> = {};
    const recentErrors: LogEntry[] = [];

    this.entries.forEach(entry => {
      entriesByLevel[entry.level]++;
      
      entriesByCategory[entry.category] = (entriesByCategory[entry.category] || 0) + 1;
      
      if (entry.service) {
        entriesByService[entry.service] = (entriesByService[entry.service] || 0) + 1;
      }
      
      if ((entry.level === 'error' || entry.level === 'fatal') && recentErrors.length < 10) {
        recentErrors.push(entry);
      }
    });

    // Calculate logs per minute
    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    const recentEntries = this.entries.filter(entry => entry.timestamp > oneMinuteAgo);
    const avgLogsPerMinute = recentEntries.length;

    return {
      totalEntries: this.entries.length,
      entriesByLevel,
      entriesByCategory,
      entriesByService,
      recentErrors: recentErrors.slice(-10),
      avgLogsPerMinute,
      lastEntry: this.entries[this.entries.length - 1]
    };
  }

  // Transport management
  public addTransport(transport: LogTransport) {
    this.transports.set(transport.name, transport);
  }

  public removeTransport(name: string) {
    this.transports.delete(name);
  }

  public enableTransport(name: string) {
    const transport = this.transports.get(name);
    if (transport) {
      transport.enabled = true;
    }
  }

  public disableTransport(name: string) {
    const transport = this.transports.get(name);
    if (transport) {
      transport.enabled = false;
    }
  }

  // Buffer management
  private startFlushTimer() {
    this.flushTimer = setInterval(() => {
      this.flushBuffer();
    }, this.config.flushInterval);
  }

  private async flushBuffer() {
    if (this.logBuffer.length === 0) return;

    const bufferCopy = [...this.logBuffer];
    this.logBuffer = [];

    // Send to all enabled transports
    for (const transport of this.transports.values()) {
      if (!transport.enabled) continue;

      for (const entry of bufferCopy) {
        if (this.shouldSendToTransport(entry, transport)) {
          try {
            await transport.send(entry);
          } catch (error) {
            // Don't log transport errors to avoid infinite loops
            console.error(`Transport ${transport.name} failed:`, error);
          }
        }
      }
    }
  }

  private async flushBufferImmediate() {
    clearTimeout(this.flushTimer!);
    await this.flushBuffer();
    this.startFlushTimer();
  }

  private shouldSendToTransport(entry: LogEntry, transport: LogTransport): boolean {
    const levelOrder: LogLevel[] = ['debug', 'info', 'warn', 'error', 'fatal'];
    const entryLevelIndex = levelOrder.indexOf(entry.level);
    const minLevelIndex = levelOrder.indexOf(transport.minLevel);
    
    return entryLevelIndex >= minLevelIndex;
  }

  // Filtering and querying
  public query(filter: LogFilter = {}): LogEntry[] {
    return this.entries.filter(entry => {
      if (filter.category && !filter.category.includes(entry.category)) {
        return false;
      }
      
      if (filter.level && !filter.level.includes(entry.level)) {
        return false;
      }
      
      if (filter.service && (!entry.service || !filter.service.includes(entry.service))) {
        return false;
      }
      
      if (filter.tags && (!entry.tags || !filter.tags.some(tag => entry.tags!.includes(tag)))) {
        return false;
      }
      
      if (filter.timeRange) {
        if (entry.timestamp < filter.timeRange.start || entry.timestamp > filter.timeRange.end) {
          return false;
        }
      }
      
      return true;
    });
  }

  public search(query: string): LogEntry[] {
    const lowercaseQuery = query.toLowerCase();
    
    return this.entries.filter(entry => 
      entry.message.toLowerCase().includes(lowercaseQuery) ||
      entry.category.toLowerCase().includes(lowercaseQuery) ||
      (entry.service && entry.service.toLowerCase().includes(lowercaseQuery)) ||
      JSON.stringify(entry.data || {}).toLowerCase().includes(lowercaseQuery)
    );
  }

  // Export functionality
  public exportLogs(format: 'json' | 'csv' | 'text' = 'json', filter?: LogFilter): string {
    const entries = filter ? this.query(filter) : this.entries;
    
    switch (format) {
      case 'json':
        return JSON.stringify(entries, null, 2);
      
      case 'csv':
        return this.formatCSV(entries);
      
      case 'text':
        return this.formatText(entries);
      
      default:
        return JSON.stringify(entries, null, 2);
    }
  }

  // Configuration
  public setUserId(userId: string) {
    this.userId = userId;
  }

  public updateConfig(newConfig: Partial<typeof this.config>) {
    Object.assign(this.config, newConfig);
  }

  // Utility methods
  private generateSessionId(): string {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  private generateEntryId(): string {
    return 'log_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  private getConsoleStyle(level: LogLevel, category: string): string {
    const categoryConfig = this.config.categories[category as keyof typeof this.config.categories];
    const color = categoryConfig?.color || '#666';
    
    const levelColors = {
      debug: '#666',
      info: '#007bff',
      warn: '#ffc107',
      error: '#dc3545',
      fatal: '#721c24'
    };
    
    return `color: ${levelColors[level]}; font-weight: bold;`;
  }

  private formatConsoleMessage(entry: LogEntry): string {
    const timestamp = new Date(entry.timestamp).toISOString().substr(11, 12);
    const level = entry.level.toUpperCase().padEnd(5);
    
    return `[${timestamp}] ${level} [${entry.category}] ${entry.message}`;
  }

  private formatCSV(entries: LogEntry[]): string {
    const headers = ['timestamp', 'level', 'category', 'message', 'service', 'userId', 'sessionId'];
    const rows = [headers.join(',')];
    
    entries.forEach(entry => {
      const row = [
        new Date(entry.timestamp).toISOString(),
        entry.level,
        entry.category,
        `"${entry.message.replace(/"/g, '""')}"`,
        entry.service || '',
        entry.userId || '',
        entry.sessionId || ''
      ];
      rows.push(row.join(','));
    });
    
    return rows.join('\n');
  }

  private formatText(entries: LogEntry[]): string {
    return entries.map(entry => {
      const timestamp = new Date(entry.timestamp).toISOString();
      let text = `[${timestamp}] ${entry.level.toUpperCase()} [${entry.category}] ${entry.message}`;
      
      if (entry.service) {
        text += ` (${entry.service})`;
      }
      
      if (entry.data) {
        text += `\n  Data: ${JSON.stringify(entry.data)}`;
      }
      
      if (entry.error) {
        text += `\n  Error: ${entry.error.message}`;
        if (entry.error.stack) {
          text += `\n  Stack: ${entry.error.stack}`;
        }
      }
      
      return text;
    }).join('\n\n');
  }

  private async saveToIndexedDB(entry: LogEntry) {
    // Simple IndexedDB implementation for browser storage
    if (!browser) return;
    
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('LoggingDB', 1);
      
      request.onerror = () => reject(request.error);
      
      request.onsuccess = () => {
        const db = request.result;
        const transaction = db.transaction(['logs'], 'readwrite');
        const store = transaction.objectStore('logs');
        
        const addRequest = store.add(entry);
        addRequest.onsuccess = () => resolve(undefined);
        addRequest.onerror = () => reject(addRequest.error);
      };
      
      request.onupgradeneeded = () => {
        const db = request.result;
        const store = db.createObjectStore('logs', { keyPath: 'id' });
        store.createIndex('timestamp', 'timestamp');
        store.createIndex('level', 'level');
        store.createIndex('category', 'category');
      };
    });
  }

  // Cleanup
  public destroy() {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
    }
    
    // Flush remaining buffer
    this.flushBuffer();
  }
}

// Singleton instance
export const loggingService = new LoggingAggregationService();

// Convenience functions
export function log(level: LogLevel, category: string, message: string, data?: unknown, error?: Error) {
  loggingService.log(level, category, message, data, error);
}

export function debug(category: string, message: string, data?: unknown) {
  loggingService.debug(category, message, data);
}

export function info(category: string, message: string, data?: unknown) {
  loggingService.info(category, message, data);
}

export function warn(category: string, message: string, data?: unknown) {
  loggingService.warn(category, message, data);
}

export function error(category: string, message: string, data?: unknown, errorObj?: Error) {
  loggingService.error(category, message, data, errorObj);
}

export function fatal(category: string, message: string, data?: unknown, errorObj?: Error) {
  loggingService.fatal(category, message, data, errorObj);
}

// Store exports
export const logEntries = loggingService.entriesStore;
export const logStats = loggingService.statsStore;