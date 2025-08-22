// Multi-Protocol Router for Enhanced RAG Pipeline
// Supports QUIC, gRPC, and REST with intelligent fallback

import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

export interface ProtocolConfig {
  endpoint: string;
  timeout: number;
  priority: number;
  healthCheckInterval: number;
  maxRetries: number;
}

export interface MultiProtocolConfig {
  protocols: {
    quic: ProtocolConfig;
    grpc: ProtocolConfig;
    rest: ProtocolConfig;
  };
  fallbackEnabled: boolean;
  healthCheckEnabled: boolean;
  loadBalancing: 'round-robin' | 'least-latency' | 'priority';
}

const defaultConfig: MultiProtocolConfig = {
  protocols: {
    quic: {
      endpoint: 'http://localhost:8094/quic',
      timeout: 2000,
      priority: 1,
      healthCheckInterval: 30000,
      maxRetries: 1
    },
    grpc: {
      endpoint: 'http://localhost:8094/grpc',
      timeout: 5000,
      priority: 2,
      healthCheckInterval: 30000,
      maxRetries: 2
    },
    rest: {
      endpoint: 'http://localhost:8094/api',
      timeout: 10000,
      priority: 3,
      healthCheckInterval: 30000,
      maxRetries: 3
    }
  },
  fallbackEnabled: true,
  healthCheckEnabled: true,
  loadBalancing: 'least-latency'
};

export interface ProtocolStatus {
  protocol: string;
  status: 'healthy' | 'degraded' | 'error' | 'unknown';
  lastCheck: number;
  latency: number;
  errorCount: number;
  successCount: number;
  uptime: number;
}

interface RouterState {
  protocolStatus: Map<string, ProtocolStatus>;
  activeConnections: Map<string, number>;
  metrics: {
    totalRequests: number;
    successfulRequests: number;
    failedRequests: number;
    averageLatency: number;
    protocolUsage: Record<string, number>;
  };
  config: MultiProtocolConfig;
}

const initialState: RouterState = {
  protocolStatus: new Map(),
  activeConnections: new Map(),
  metrics: {
    totalRequests: 0,
    successfulRequests: 0,
    failedRequests: 0,
    averageLatency: 0,
    protocolUsage: { quic: 0, grpc: 0, rest: 0 }
  },
  config: defaultConfig
};

// Reactive stores
export const routerState = writable(initialState);
export const protocolHealth = derived(routerState, $state => 
  Array.from($state.protocolStatus.values())
);
export const routerMetrics = derived(routerState, $state => $state.metrics);

export class MultiProtocolRouter {
  private state: RouterState;
  private workers: Map<string, Worker> = new Map();
  private healthCheckIntervals: Map<string, number> = new Map();
  private unsubscribe: (() => void) | null = null;

  constructor(config?: Partial<MultiProtocolConfig>) {
    this.state = { ...initialState };
    if (config) {
      this.state.config = { ...defaultConfig, ...config };
    }

    // Initialize protocol status
    Object.keys(this.state.config.protocols).forEach(protocol => {
      this.state.protocolStatus.set(protocol, {
        protocol,
        status: 'unknown',
        lastCheck: 0,
        latency: 0,
        errorCount: 0,
        successCount: 0,
        uptime: 0
      });
      this.state.activeConnections.set(protocol, 0);
    });

    // Subscribe to state changes
    this.unsubscribe = routerState.subscribe(state => {
      this.state = state;
    });

    if (browser) {
      this.initializeProtocols();
    }
  }

  private async initializeProtocols(): Promise<void> {
    const protocols = Object.keys(this.state.config.protocols);

    for (const protocol of protocols) {
      try {
        // Initialize protocol worker
        const worker = new Worker(`/workers/protocol-${protocol}-worker.js`);
        worker.onmessage = this.handleWorkerMessage.bind(this);
        worker.onerror = this.handleWorkerError.bind(this);
        this.workers.set(protocol, worker);

        // Start health checking if enabled
        if (this.state.config.healthCheckEnabled) {
          this.startHealthCheck(protocol);
        }

        console.log(`✓ ${protocol.toUpperCase()} protocol initialized`);
      } catch (error) {
        console.error(`Failed to initialize ${protocol} protocol:`, error);
        this.updateProtocolStatus(protocol, 'error');
      }
    }
  }

  private startHealthCheck(protocol: string): void {
    const config = this.state.config.protocols[protocol as keyof typeof this.state.config.protocols];
    
    const checkHealth = async () => {
      const startTime = Date.now();
      
      try {
        const response = await this.executeHealthCheck(protocol);
        const latency = Date.now() - startTime;
        
        if (response.ok) {
          this.updateProtocolStatus(protocol, 'healthy', latency);
        } else {
          this.updateProtocolStatus(protocol, 'degraded', latency);
        }
      } catch (error) {
        this.updateProtocolStatus(protocol, 'error');
      }
    };

    // Initial health check
    checkHealth();

    // Schedule periodic health checks
    const intervalId = setInterval(checkHealth, config.healthCheckInterval);
    this.healthCheckIntervals.set(protocol, intervalId as any);
  }

  private async executeHealthCheck(protocol: string): Promise<{ ok: boolean; data?: unknown }> {
    const worker = this.workers.get(protocol);
    if (!worker) {
      throw new Error(`Worker for ${protocol} not available`);
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Health check timeout for ${protocol}`));
      }, 5000);

      const handleMessage = (event: MessageEvent) => {
        const { type, data, protocol: responseProtocol } = event.data;
        
        if (type === 'HEALTH_CHECK_RESPONSE' && responseProtocol === protocol) {
          clearTimeout(timeout);
          worker.removeEventListener('message', handleMessage);
          resolve({ ok: data.status === 'healthy', data });
        }
      };

      worker.addEventListener('message', handleMessage);
      
      worker.postMessage({
        type: 'HEALTH_CHECK',
        protocol,
        endpoint: this.state.config.protocols[protocol as keyof typeof this.state.config.protocols].endpoint
      });
    });
  }

  private handleWorkerMessage(event: MessageEvent): void {
    const { type, data, protocol, requestId } = event.data;

    switch (type) {
      case 'REQUEST_COMPLETE':
        this.handleRequestComplete(protocol, data, requestId);
        break;
      case 'REQUEST_ERROR':
        this.handleRequestError(protocol, data, requestId);
        break;
      case 'HEALTH_CHECK_RESPONSE':
        // Handled in executeHealthCheck
        break;
    }
  }

  private handleWorkerError(error: ErrorEvent): void {
    console.error('Protocol worker error:', error);
  }

  private updateProtocolStatus(
    protocol: string, 
    status: ProtocolStatus['status'], 
    latency?: number
  ): void {
    routerState.update(state => {
      const protocolStatus = new Map(state.protocolStatus);
      const current = protocolStatus.get(protocol);
      
      if (current) {
        const updated: ProtocolStatus = {
          ...current,
          status,
          lastCheck: Date.now(),
          latency: latency || current.latency
        };

        if (status === 'healthy') {
          updated.successCount++;
          updated.errorCount = Math.max(0, updated.errorCount - 1); // Decay errors
        } else if (status === 'error') {
          updated.errorCount++;
        }

        protocolStatus.set(protocol, updated);
      }

      return {
        ...state,
        protocolStatus
      };
    });
  }

  public async route<T>(
    request: unknown,
    options: {
      preferredProtocol?: string;
      timeout?: number;
      retries?: number;
    } = {}
  ): Promise<T> {
    const startTime = Date.now();
    const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Update metrics
    this.updateMetrics('request_started');

    try {
      // Determine protocol order
      const protocolOrder = this.getProtocolOrder(options.preferredProtocol);
      
      let lastError: Error | null = null;
      
      for (const protocol of protocolOrder) {
        try {
          const result = await this.executeRequest<T>(protocol, request, requestId, options);
          
          // Update success metrics
          const latency = Date.now() - startTime;
          this.updateMetrics('request_success', { protocol, latency });
          
          return result;
        } catch (error) {
          lastError = error as Error;
          console.warn(`Request failed with ${protocol}, trying next...`, error);
          
          this.updateMetrics('request_failed', { protocol });
          
          if (!this.state.config.fallbackEnabled) {
            break;
          }
        }
      }

      throw lastError || new Error('All protocols failed');
    } catch (error) {
      this.updateMetrics('request_error');
      throw error;
    }
  }

  private getProtocolOrder(preferredProtocol?: string): string[] {
    const availableProtocols = Array.from(this.state.protocolStatus.entries())
      .filter(([_, status]) => status.status === 'healthy' || status.status === 'degraded')
      .map(([protocol]) => protocol);

    if (availableProtocols.length === 0) {
      // Fallback to all protocols if none are healthy
      return Object.keys(this.state.config.protocols);
    }

    // Apply load balancing strategy
    switch (this.state.config.loadBalancing) {
      case 'priority':
        return this.orderByPriority(availableProtocols, preferredProtocol);
      
      case 'least-latency':
        return this.orderByLatency(availableProtocols, preferredProtocol);
      
      case 'round-robin':
        return this.orderByRoundRobin(availableProtocols, preferredProtocol);
      
      default:
        return this.orderByPriority(availableProtocols, preferredProtocol);
    }
  }

  private orderByPriority(protocols: string[], preferred?: string): string[] {
    const withPriority = protocols.map(protocol => ({
      protocol,
      priority: this.state.config.protocols[protocol as keyof typeof this.state.config.protocols].priority
    }));

    withPriority.sort((a, b) => a.priority - b.priority);

    const ordered = withPriority.map(item => item.protocol);

    // Move preferred to front if specified and available
    if (preferred && ordered.includes(preferred)) {
      return [preferred, ...ordered.filter(p => p !== preferred)];
    }

    return ordered;
  }

  private orderByLatency(protocols: string[], preferred?: string): string[] {
    const withLatency = protocols.map(protocol => ({
      protocol,
      latency: this.state.protocolStatus.get(protocol)?.latency || Infinity
    }));

    withLatency.sort((a, b) => a.latency - b.latency);

    const ordered = withLatency.map(item => item.protocol);

    if (preferred && ordered.includes(preferred)) {
      return [preferred, ...ordered.filter(p => p !== preferred)];
    }

    return ordered;
  }

  private orderByRoundRobin(protocols: string[], preferred?: string): string[] {
    // Simple round-robin based on usage count
    const withUsage = protocols.map(protocol => ({
      protocol,
      usage: this.state.metrics.protocolUsage[protocol] || 0
    }));

    withUsage.sort((a, b) => a.usage - b.usage);

    const ordered = withUsage.map(item => item.protocol);

    if (preferred && ordered.includes(preferred)) {
      return [preferred, ...ordered.filter(p => p !== preferred)];
    }

    return ordered;
  }

  private async executeRequest<T>(
    protocol: string,
    request: unknown,
    requestId: string,
    options: unknown
  ): Promise<T> {
    const worker = this.workers.get(protocol);
    if (!worker) {
      throw new Error(`Worker for protocol ${protocol} not available`);
    }

    const config = this.state.config.protocols[protocol as keyof typeof this.state.config.protocols];
    const timeout = options.timeout || config.timeout;

    // Increment active connections
    this.incrementActiveConnections(protocol);

    try {
      return await new Promise<T>((resolve, reject) => {
        const timeoutId = setTimeout(() => {
          reject(new Error(`Request timeout for protocol ${protocol}`));
        }, timeout);

        const handleMessage = (event: MessageEvent) => {
          const { type, data, requestId: responseRequestId } = event.data;
          
          if (responseRequestId === requestId) {
            clearTimeout(timeoutId);
            worker.removeEventListener('message', handleMessage);
            
            if (type === 'REQUEST_COMPLETE') {
              resolve(data);
            } else if (type === 'REQUEST_ERROR') {
              reject(new Error(data.message || 'Request failed'));
            }
          }
        };

        worker.addEventListener('message', handleMessage);
        
        worker.postMessage({
          type: 'EXECUTE_REQUEST',
          requestId,
          protocol,
          request,
          endpoint: config.endpoint,
          options
        });
      });
    } finally {
      // Decrement active connections
      this.decrementActiveConnections(protocol);
    }
  }

  private incrementActiveConnections(protocol: string): void {
    routerState.update(state => {
      const activeConnections = new Map(state.activeConnections);
      const current = activeConnections.get(protocol) || 0;
      activeConnections.set(protocol, current + 1);
      
      return {
        ...state,
        activeConnections
      };
    });
  }

  private decrementActiveConnections(protocol: string): void {
    routerState.update(state => {
      const activeConnections = new Map(state.activeConnections);
      const current = activeConnections.get(protocol) || 0;
      activeConnections.set(protocol, Math.max(0, current - 1));
      
      return {
        ...state,
        activeConnections
      };
    });
  }

  private handleRequestComplete(protocol: string, data: unknown, requestId: string): void {
    // Handle successful request completion
    this.updateProtocolStatus(protocol, 'healthy');
  }

  private handleRequestError(protocol: string, error: unknown, requestId: string): void {
    // Handle request error
    this.updateProtocolStatus(protocol, 'error');
  }

  private updateMetrics(
    type: 'request_started' | 'request_success' | 'request_failed' | 'request_error',
    data?: { protocol?: string; latency?: number }
  ): void {
    routerState.update(state => {
      const metrics = { ...state.metrics };

      switch (type) {
        case 'request_started':
          metrics.totalRequests++;
          break;
          
        case 'request_success':
          metrics.successfulRequests++;
          if (data?.protocol) {
            metrics.protocolUsage[data.protocol]++;
          }
          if (data?.latency) {
            metrics.averageLatency = (
              (metrics.averageLatency * (metrics.successfulRequests - 1) + data.latency) / 
              metrics.successfulRequests
            );
          }
          break;
          
        case 'request_failed':
        case 'request_error':
          metrics.failedRequests++;
          break;
      }

      return {
        ...state,
        metrics
      };
    });
  }

  public getProtocolStatus(): Map<string, ProtocolStatus> {
    return new Map(this.state.protocolStatus);
  }

  public getMetrics() {
    return {
      ...this.state.metrics,
      activeConnections: Object.fromEntries(this.state.activeConnections),
      protocolHealth: Object.fromEntries(
        Array.from(this.state.protocolStatus.entries()).map(([protocol, status]) => [
          protocol,
          {
            status: status.status,
            latency: status.latency,
            uptime: status.successCount / (status.successCount + status.errorCount) || 0
          }
        ])
      )
    };
  }

  public async testProtocol(protocol: string): Promise<boolean> {
    try {
      const result = await this.executeHealthCheck(protocol);
      return result.ok;
    } catch (error) {
      return false;
    }
  }

  public destroy(): void {
    // Clean up workers
    this.workers.forEach(worker => worker.terminate());
    this.workers.clear();

    // Clear health check intervals
    this.healthCheckIntervals.forEach(intervalId => clearInterval(intervalId));
    this.healthCheckIntervals.clear();

    // Unsubscribe from state
    if (this.unsubscribe) {
      this.unsubscribe();
    }
  }
}

// Export singleton instance
export const multiProtocolRouter = new MultiProtocolRouter();

// Helper functions for common operations
export const routerHelpers = {
  async ragQuery(query: string, options: unknown = {}) {
    return multiProtocolRouter.route({
      type: 'rag_query',
      query,
      ...options
    }, {
      preferredProtocol: 'quic',
      timeout: 10000
    });
  },

  async documentUpload(document: unknown, options: unknown = {}) {
    return multiProtocolRouter.route({
      type: 'document_upload',
      document,
      ...options
    }, {
      preferredProtocol: 'grpc',
      timeout: 30000
    });
  },

  async semanticSearch(query: string, filters: unknown = {}) {
    return multiProtocolRouter.route({
      type: 'semantic_search',
      query,
      filters
    }, {
      preferredProtocol: 'quic',
      timeout: 5000
    });
  },

  async healthCheck() {
    const protocols = ['quic', 'grpc', 'rest'];
    const results = await Promise.all(
      protocols.map(async protocol => ({
        protocol,
        healthy: await multiProtocolRouter.testProtocol(protocol)
      }))
    );
    
    return Object.fromEntries(results.map(r => [r.protocol, r.healthy]));
  },

  async search(options: unknown = {}) {
    return multiProtocolRouter.route({
      type: 'search',
      ...options
    }, {
      preferredProtocol: 'quic',
      timeout: 8000
    });
  },

  async getSuggestions(query: string, options: unknown = {}) {
    return multiProtocolRouter.route({
      type: 'suggestions',
      query,
      ...options
    }, {
      preferredProtocol: 'rest',
      timeout: 3000
    });
  }
};