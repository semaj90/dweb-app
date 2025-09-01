/**
 * YoRHa API Client for 3D UI Components
 * Provides modular loading of a JSON schema-driven layout
 * and real-time data subscriptions (SSE / WebSocket / polling).
 */

import type { YoRHaButton3DOptions } from '../components/YoRHaButton3D';
import type { YoRHaPanel3DOptions } from '../components/YoRHaPanel3D';
import type { YoRHaInput3DOptions } from '../components/YoRHaInput3D';
import type { YoRHaModal3DOptions } from '../components/YoRHaModal3D';

export interface YoRHaAPIConfig {
  baseURL: string;
  apiKey?: string;
  timeout: number;
  retryAttempts: number;
  enableWebSocket: boolean;
  enableSSE: boolean;
  // Optional callback when any data source pushes new data
  onData?: (id: string, data: any) => void;
}

export interface YoRHaComponentData {
  id: string;
  type: 'button' | 'panel' | 'input' | 'modal' | 'layout';
  config: any;
  data?: {
    loading?: boolean;
    value?: string;
    error?: boolean;
    [key: string]: any;
  };
  metrics?: YoRHaMetrics;
  events?: YoRHaEvent[];
}

export interface YoRHaComponentInstance {
  id: string;
  type: 'button' | 'panel' | 'input' | 'modal' | 'layout';
  config: any;
  data?: unknown;
  metrics?: YoRHaMetrics;
  events?: YoRHaEvent[];
  state?: unknown;
}

export interface YoRHaMetrics {
  interactions: number;
  renderTime: number;
  lastUpdate: string;
  performanceScore: number;
  memoryUsage: number;
}

export interface YoRHaEvent {
  type: string;
  timestamp: string;
  data: any;
  componentId: string;
}

export interface YoRHaSystemStatus {
  database: {
    connected: boolean;
    latency: number;
    activeConnections: number;
    queryCount: number;
  };
  backend: {
    healthy: boolean;
    uptime: number;
    activeServices: number;
    cpuUsage: number;
    memoryUsage: number;
  };
  frontend: {
    renderFPS: number;
    componentCount: number;
    activeComponents: number;
    webGPUEnabled: boolean;
  };
}

export interface YoRHaGraphData {
  nodes: Array<{
    id: string;
    type: 'database' | 'service' | 'component' | 'api';
    label: string;
    position: { x: number; y: number; z: number };
    metrics: any;
    status: 'healthy' | 'warning' | 'error' | 'offline';
  }>;
  edges: Array<{
    from: string;
    to: string;
    type: 'data' | 'api' | 'websocket' | 'grpc';
    traffic: number;
    latency: number;
  }>;
}

export class YoRHaAPIClient {
  private config: YoRHaAPIConfig;
  private websocket?: WebSocket;
  private wsAttempts = 0;
  private eventSource?: EventSource;
  private sseAttempts = 0;
  private cache = new Map<string, any>();
  private subscribers = new Map<string, Set<Function>>();
  private layout: any = null;
  private dataSourceIntervals = new Map<string, ReturnType<typeof setInterval>>();

  constructor(config: Partial<YoRHaAPIConfig> = {}) {
    this.config = {
      baseURL: config.baseURL || 'http://localhost:8443/api/yorha',
      timeout: config.timeout || 5000,
      retryAttempts: config.retryAttempts || 3,
      enableWebSocket: config.enableWebSocket ?? true,
      enableSSE: config.enableSSE ?? true,
      ...config
    };

    if (this.config.enableWebSocket) {
      this.initWebSocket();
    }

    if (this.config.enableSSE) {
      this.initServerSentEvents();
    }
  }

  // ---- Layout + Data Source Lifecycle -------------------------------------------------

  /**
   * Load a JSON layout definition from an endpoint (or absolute URL)
   * and store internally. Existing data streams are stopped before replacing.
   */
  async loadLayout(layoutUrl: string): Promise<any> {
    // Stop existing streams if reloading
    this.stopDataStreams();
    const url = layoutUrl.startsWith('http') ? layoutUrl : layoutUrl;
    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    if (!res.ok) throw new Error(`Failed to load layout: ${res.status}`);
    this.layout = await res.json();
    this.notifySubscribers('layout:loaded', this.layout);
    return this.layout;
  }

  /** Get the active layout object (if loaded). */
  getLayout(): unknown { return this.layout; }

  /** Start polling / mock generation for declared dataSources in the layout. */
  startDataStreams(): void {
    if (!this.layout?.dataSources) return;
    this.stopDataStreams();
    for (const ds of this.layout.dataSources) {
      switch (ds.type) {
        case 'rest': {
          const interval = setInterval(async () => {
            try {
              const res = await fetch(ds.endpoint, { headers: { 'Accept': 'application/json' } });
              if (res.ok) {
                const data = await res.json();
                this.pushData(ds.name, data);
              }
            } catch (err: any) {
              console.warn(`Data source '${ds.name}' fetch failed`, err);
            }
          }, ds.intervalMs ?? 5000);
          this.dataSourceIntervals.set(ds.name, interval);
          break;
        }
        case 'mock': {
          const interval = setInterval(() => {
            const data = {
              value: Math.random(),
              updatedAt: new Date().toISOString(),
              name: ds.name
            };
            this.pushData(ds.name, data);
          }, ds.intervalMs ?? 2000);
          this.dataSourceIntervals.set(ds.name, interval);
          break;
        }
        default:
          console.warn(`Unknown data source type: ${ds.type}`);
      }
    }
  }

  /** Stop all active data source intervals. */
  stopDataStreams(): void {
    Array.from(this.dataSourceIntervals.values()).forEach(interval => clearInterval(interval as any));
    this.dataSourceIntervals.clear();
  }

  /** Push data from a source into cache + notify watchers */
  private pushData(id: string, data: any) {
    this.cache.set(`ds:${id}`, data);
    this.notifySubscribers(`data:${id}`, data);
    if (this.config.onData) {
      try { this.config.onData(id, data); } catch (e: any) { /* swallow */ }
    }
  }

  /** Retrieve last value for a data source */
  getDataSourceValue(id: string): unknown { return this.cache.get(`ds:${id}`); }

  // Component Configuration API
  async getComponentConfig(componentId: string, type: string): Promise<YoRHaComponentData> {
    const cacheKey = `config:${componentId}`;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    const response = await this.apiCall(`/components/${type}/${componentId}`);
    this.cache.set(cacheKey, response);
    return response;
  }

  async updateComponentConfig(componentId: string, config: any): Promise<void> {
    await this.apiCall(`/components/${componentId}`, {
      method: 'PUT',
      body: JSON.stringify(config)
    });

    // Clear cache
    this.cache.delete(`config:${componentId}`);

    // Notify subscribers
    this.notifySubscribers(`component:${componentId}:updated`, config);
  }

  async getComponentMetrics(componentId: string): Promise<YoRHaMetrics> {
    return await this.apiCall(`/metrics/component/${componentId}`);
  }

  // Real-time Data API
  async getSystemStatus(): Promise<YoRHaSystemStatus> {
    return await this.apiCall('/system/status');
  }

  async getGraphData(): Promise<YoRHaGraphData> {
    return await this.apiCall('/system/graph');
  }

  async getDatabaseMetrics(): Promise<any> {
    return await this.apiCall('/metrics/database');
  }

  async getBackendMetrics(): Promise<any> {
    return await this.apiCall('/metrics/backend');
  }

  async getFrontendMetrics(): Promise<any> {
    return await this.apiCall('/metrics/frontend');
  }

  // Component Factory Methods
  async createButtonFromAPI(componentId: string): Promise<YoRHaButton3DOptions> {
    const data = await this.getComponentConfig(componentId, 'button');
    return {
      text: data.config.text,
      variant: data.config.variant,
      size: data.config.size,
      icon: data.config.icon,
      loading: data.data?.loading || false,
      ...data.config
    };
  }

  async createPanelFromAPI(componentId: string): Promise<YoRHaPanel3DOptions> {
    const data = await this.getComponentConfig(componentId, 'panel');
    return {
      title: data.config.title,
      variant: data.config.variant,
      width: data.config.width,
      height: data.config.height,
      scrollable: data.config.scrollable,
      ...data.config
    };
  }

  async createInputFromAPI(componentId: string): Promise<YoRHaInput3DOptions> {
    const data = await this.getComponentConfig(componentId, 'input');
    return {
      placeholder: data.config.placeholder,
      type: data.config.type,
      variant: data.config.variant,
      value: data.data?.value || '',
      error: data.data?.error || false,
      ...data.config
    };
  }

  async createModalFromAPI(componentId: string): Promise<YoRHaModal3DOptions> {
    const data = await this.getComponentConfig(componentId, 'modal');
    return {
      title: data.config.title,
      variant: data.config.variant,
      size: data.config.size,
      closable: data.config.closable,
      ...data.config
    };
  }

  // Event Logging
  async logEvent(event: Omit<YoRHaEvent, 'timestamp'>): Promise<void> {
    const fullEvent = {
      ...event,
      timestamp: new Date().toISOString()
    };

    await this.apiCall('/events', {
      method: 'POST',
      body: JSON.stringify(fullEvent)
    });
  }

  async getEvents(componentId?: string, limit = 100): Promise<YoRHaEvent[]> {
    const params = new URLSearchParams();
    if (componentId) params.set('componentId', componentId);
    params.set('limit', limit.toString());

    return await this.apiCall(`/events?${params.toString()}`);
  }

  // Subscription System
  subscribe<T = any>(event: string, callback: (data: T) => void): () => void {
    if (!this.subscribers.has(event)) {
      this.subscribers.set(event, new Set());
    }

    this.subscribers.get(event)!.add(callback);

    return () => {
      this.subscribers.get(event)?.delete(callback);
    };
  }

  private notifySubscribers(event: string, data: any): void {
    const callbacks = this.subscribers.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }

  // WebSocket Integration with QUIC fallback
  private initWebSocket(): void {
    // Check if we're in browser environment
    if (typeof window === 'undefined' || typeof WebSocket === 'undefined') {
      console.warn('WebSocket not available in SSR environment');
      return;
    }

    const wsUrl = this.config.baseURL.replace(/^https?/, 'wss') + '/ws';

    try {
      this.websocket = new WebSocket(wsUrl);
    } catch (e: any) {
      console.warn('WebSocket init failed', e);
      return;
    }

    this.websocket.onopen = () => {
      this.wsAttempts = 0;
      console.log('YoRHa WebSocket connected');
    };

    this.websocket.onmessage = (event: any) => {
      try {
        const message = JSON.parse(event.data);
        this.handleWebSocketMessage(message);
      } catch (error: any) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.websocket.onclose = () => {
      const delay = Math.min(30000, Math.pow(2, this.wsAttempts) * 1000 + Math.random() * 500);
      this.wsAttempts++;
      console.log(`YoRHa WebSocket disconnected, reconnecting in ${(delay / 1000).toFixed(1)}s (attempt ${this.wsAttempts})`);
      setTimeout(() => this.initWebSocket(), delay);
    };
  }

  private handleWebSocketMessage(message: any): void {
    switch (message.type) {
      case 'component_update':
        this.cache.delete(`config:${message.componentId}`);
        this.notifySubscribers(`component:${message.componentId}:updated`, message.data);
        break;

      case 'system_metrics':
        this.notifySubscribers('system:metrics', message.data);
        break;

      case 'graph_update':
        this.notifySubscribers('system:graph:updated', message.data);
        break;

      default:
        this.notifySubscribers(message.type, message.data);
    }
  }

  // Server-Sent Events
  private initServerSentEvents(): void {
    // Check if we're in browser environment
    if (typeof window === 'undefined' || typeof EventSource === 'undefined') {
      console.warn('EventSource not available in SSR environment');
      return;
    }

    try {
      this.eventSource = new EventSource(`${this.config.baseURL}/events/stream`);
    } catch (e: any) {
      console.warn('SSE init failed', e);
      return;
    }

    this.eventSource.onmessage = (event: any) => {
      try {
        const data = JSON.parse(event.data);
        this.notifySubscribers('sse:message', data);
        if (data && typeof data === 'object' && 'type' in data) {
          const evtType = (data as any).type;
          const payload = (data as any).data ?? data;
          this.notifySubscribers(evtType, payload);
        }
      } catch (error: any) {
        console.error('Failed to parse SSE message:', error);
      }
    };

    this.eventSource.onerror = () => {
      this.eventSource?.close();
      const delay = Math.min(30000, Math.pow(2, this.sseAttempts) * 1000 + Math.random() * 500);
      this.sseAttempts++;
      console.error(`SSE connection error, retrying in ${(delay / 1000).toFixed(1)}s (attempt ${this.sseAttempts})`);
      setTimeout(() => this.initServerSentEvents(), delay);
    };
  }

  // HTTP API Helper
  private async apiCall(endpoint: string, options: RequestInit = {}): Promise<any> {
    const url = `${this.config.baseURL}${endpoint}`;

    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...(this.config.apiKey && { 'Authorization': `Bearer ${this.config.apiKey}` })
      },
      ...options
    };

    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.config.retryAttempts; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

        const response = await fetch(url, {
          ...defaultOptions,
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();

      } catch (error: any) {
        lastError = error as Error;
        if (attempt < this.config.retryAttempts - 1) {
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
        }
      }
    }

    throw lastError;
  }

  // Cleanup
  dispose(): void {
    if (this.websocket) {
      this.websocket.close();
    }

    if (this.eventSource) {
      this.eventSource.close();
    }

    this.stopDataStreams();

    this.cache.clear();
    this.subscribers.clear();
  }
}

// Singleton instance for global use
export const yorhaAPI = new YoRHaAPIClient();
;
// Component Data Validators
export const YoRHaValidators = {
  validateButtonConfig: (config: any): config is YoRHaButton3DOptions => {
    return typeof config === 'object' &&
           (typeof config.text === 'string' || config.text === undefined);
  },

  validatePanelConfig: (config: any): config is YoRHaPanel3DOptions => {
    return typeof config === 'object' &&
           (typeof config.title === 'string' || config.title === undefined);
  },

  validateInputConfig: (config: any): config is YoRHaInput3DOptions => {
    return typeof config === 'object' &&
           (typeof config.placeholder === 'string' || config.placeholder === undefined);
  },

  validateModalConfig: (config: any): config is YoRHaModal3DOptions => {
    return typeof config === 'object' &&
           (typeof config.title === 'string' || config.title === undefined);
  }
};

// YoRHaAPIClient is already exported above as a class