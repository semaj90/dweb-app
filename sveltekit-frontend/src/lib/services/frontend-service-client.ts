/**
 * Frontend Service Client
 * Production-ready client for communicating with backend services
 */

import { writable, derived, type Writable } from 'svelte/store';
import { browser } from '$app/environment';

// Service client configuration
const CONFIG = {
  baseURL: '/api/v1',
  timeout: 30000,
  retries: 3,
  retryDelay: 1000
};

// Types
interface ServiceResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
  requestId: string;
  version: string;
  performance?: {
    executionTime: number;
    cacheHit?: boolean;
    servicesUsed: string[];
  };
}

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: Record<string, {
    status: 'healthy' | 'unhealthy';
    responseTime?: number;
    details?: any;
  }>;
  uptime: number;
  timestamp: string;
}

interface SearchResult {
  id: number;
  title: string;
  score: number;
  type: string;
  content?: string;
}

interface RAGResponse {
  query: string;
  response: string;
  sources: string[];
  confidence: number;
  cacheId?: string;
}

interface UploadResult {
  fileId: string;
  fileName: string;
  size: number;
  url: string;
  status: string;
}

// Service state stores
export const serviceHealth = writable<HealthStatus | null>(null);
export const isConnected = writable<boolean>(true);
export const requestQueue = writable<number>(0);
export const lastError = writable<string | null>(null);

// Derived stores
export const servicesStatus = derived(serviceHealth, ($health) => {
  if (!$health) return 'unknown';
  return $health.status;
});

export const availableServices = derived(serviceHealth, ($health) => {
  if (!$health) return [];
  return Object.entries($health.services)
    .filter(([_, service]) => service.status === 'healthy')
    .map(([name]) => name);
});

class FrontendServiceClient {
  private static instance: FrontendServiceClient;
  private abortController: AbortController = new AbortController();
  private healthCheckInterval: number | null = null;

  constructor() {
    if (browser) {
      this.startHealthChecking();
      this.setupEventListeners();
    }
  }

  static getInstance(): FrontendServiceClient {
    if (!FrontendServiceClient.instance) {
      FrontendServiceClient.instance = new FrontendServiceClient();
    }
    return FrontendServiceClient.instance;
  }

  // ==================== HTTP CLIENT ====================

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ServiceResponse<T>> {
    const url = `${CONFIG.baseURL}${endpoint}`;
    
    // Update request queue
    requestQueue.update(n => n + 1);

    try {
      const response = await fetch(url, {
        ...options,
        signal: this.abortController.signal,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          ...options.headers
        }
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      // Clear last error on success
      lastError.set(null);
      
      return data;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      lastError.set(errorMessage);
      
      throw new Error(errorMessage);
    } finally {
      requestQueue.update(n => Math.max(0, n - 1));
    }
  }

  private async get<T>(endpoint: string): Promise<ServiceResponse<T>> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  private async post<T>(endpoint: string, body: any): Promise<ServiceResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(body)
    });
  }

  // ==================== SERVICE METHODS ====================

  async checkHealth(): Promise<HealthStatus> {
    try {
      const response = await this.get<HealthStatus>('/unified?action=health');
      const healthData = response.data!;
      
      serviceHealth.set(healthData);
      isConnected.set(true);
      
      return healthData;
    } catch (error) {
      isConnected.set(false);
      throw error;
    }
  }

  async search(query: string, type: string = 'mixed'): Promise<SearchResult[]> {
    const response = await this.get<{
      query: string;
      results: SearchResult[];
      total: number;
    }>(`/unified?action=search&query=${encodeURIComponent(query)}&type=${type}`);
    
    return response.data?.results || [];
  }

  async performRAG(query: string, caseId?: string): Promise<RAGResponse> {
    const response = await this.post<RAGResponse>('/unified?action=rag', {
      query,
      caseId,
      userId: 1 // TODO: Get from auth context
    });
    
    return response.data!;
  }

  async uploadFile(file: File, caseId?: string): Promise<UploadResult> {
    // Create FormData for file upload
    const formData = new FormData();
    formData.append('file', file);
    if (caseId) formData.append('caseId', caseId);

    const response = await this.post<UploadResult>('/unified?action=upload', {
      fileName: file.name,
      size: file.size,
      caseId,
      uploadedBy: 1 // TODO: Get from auth context
    });
    
    return response.data!;
  }

  async manageWorkflow(
    type: 'document' | 'case' | 'rag',
    workflowId: string,
    event?: any
  ) {
    const response = await this.post('/unified?action=workflow', {
      type,
      workflowId,
      event
    });
    
    return response.data;
  }

  async cacheOperation(operation: 'get' | 'set' | 'delete', key: string, value?: any, ttl?: number) {
    const response = await this.post('/unified?action=cache', {
      operation,
      key,
      value,
      ttl
    });
    
    return response.data;
  }

  // ==================== REAL-TIME FEATURES ====================

  private setupEventListeners(): void {
    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.stopHealthChecking();
      } else {
        this.startHealthChecking();
      }
    });

    // Handle online/offline events
    window.addEventListener('online', () => {
      isConnected.set(true);
      this.checkHealth();
    });

    window.addEventListener('offline', () => {
      isConnected.set(false);
    });
  }

  private startHealthChecking(): void {
    if (this.healthCheckInterval) return;

    // Initial health check
    this.checkHealth().catch(console.error);

    // Periodic health checks every 30 seconds
    this.healthCheckInterval = window.setInterval(() => {
      this.checkHealth().catch(console.error);
    }, 30000);
  }

  private stopHealthChecking(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }

  // ==================== UTILITY METHODS ====================

  abortAllRequests(): void {
    this.abortController.abort();
    this.abortController = new AbortController();
  }

  getMetrics() {
    return {
      requestQueue: requestQueue,
      isConnected: isConnected,
      lastError: lastError,
      serviceHealth: serviceHealth
    };
  }

  // ==================== CLEANUP ====================

  destroy(): void {
    this.stopHealthChecking();
    this.abortAllRequests();
  }
}

// Singleton instance
export const serviceClient = FrontendServiceClient.getInstance();

// ==================== REACTIVE API FUNCTIONS ====================

/**
 * Reactive search function
 */
export function createSearchStore(initialQuery: string = '') {
  const query = writable(initialQuery);
  const results = writable<SearchResult[]>([]);
  const isLoading = writable(false);
  const error = writable<string | null>(null);

  const search = async (searchQuery?: string) => {
    const q = searchQuery || initialQuery;
    if (!q.trim()) return;

    isLoading.set(true);
    error.set(null);

    try {
      const searchResults = await serviceClient.search(q);
      results.set(searchResults);
    } catch (err) {
      error.set(err instanceof Error ? err.message : 'Search failed');
      results.set([]);
    } finally {
      isLoading.set(false);
    }
  };

  return {
    query,
    results,
    isLoading,
    error,
    search
  };
}

/**
 * Reactive RAG function
 */
export function createRAGStore() {
  const query = writable('');
  const response = writable<RAGResponse | null>(null);
  const isLoading = writable(false);
  const error = writable<string | null>(null);

  const ask = async (ragQuery: string, caseId?: string) => {
    if (!ragQuery.trim()) return;

    isLoading.set(true);
    error.set(null);
    query.set(ragQuery);

    try {
      const ragResponse = await serviceClient.performRAG(ragQuery, caseId);
      response.set(ragResponse);
    } catch (err) {
      error.set(err instanceof Error ? err.message : 'RAG query failed');
      response.set(null);
    } finally {
      isLoading.set(false);
    }
  };

  return {
    query,
    response,
    isLoading,
    error,
    ask
  };
}

/**
 * File upload function
 */
export function createUploadStore() {
  const uploads = writable<Map<string, {
    file: File;
    progress: number;
    status: 'pending' | 'uploading' | 'completed' | 'failed';
    result?: UploadResult;
    error?: string;
  }>>(new Map());

  const upload = async (file: File, caseId?: string) => {
    const uploadId = `${Date.now()}_${file.name}`;
    
    uploads.update(map => {
      map.set(uploadId, {
        file,
        progress: 0,
        status: 'uploading'
      });
      return map;
    });

    try {
      const result = await serviceClient.uploadFile(file, caseId);
      
      uploads.update(map => {
        const upload = map.get(uploadId);
        if (upload) {
          upload.status = 'completed';
          upload.progress = 100;
          upload.result = result;
        }
        return map;
      });

      return result;
    } catch (error) {
      uploads.update(map => {
        const upload = map.get(uploadId);
        if (upload) {
          upload.status = 'failed';
          upload.error = error instanceof Error ? error.message : 'Upload failed';
        }
        return map;
      });
      throw error;
    }
  };

  const removeUpload = (uploadId: string) => {
    uploads.update(map => {
      map.delete(uploadId);
      return map;
    });
  };

  return {
    uploads,
    upload,
    removeUpload
  };
}

// Export for use in cleanup
if (browser) {
  window.addEventListener('beforeunload', () => {
    serviceClient.destroy();
  });
}