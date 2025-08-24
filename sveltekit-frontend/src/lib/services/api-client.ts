/**
 * Modular API Client for Legal AI Platform
 * Integrates with PostgreSQL, pgvector, and Drizzle ORM
 */

import type {
  ApiResponse,
  Case,
  Evidence,
  LegalDocument,
  User,
  CaseCreateRequest,
  CaseUpdateRequest,
  CaseSearchRequest,
  CaseSearchResponse,
  EvidenceCreateRequest,
  EvidenceSearchRequest,
  CommandSearchRequest,
  CommandSearchResponse,
  VectorSearchRequest,
  VectorSearchResult,
  BulkOperationRequest,
  BulkOperationResponse,
  FormSubmissionResult
} from '$lib/types/api';

// Base API client configuration
const BASE_URL = '/api/v1';
const DEFAULT_TIMEOUT = 10000;

interface RequestConfig {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  headers?: Record<string, string>;
  timeout?: number;
  signal?: AbortSignal;
}

class ApiClient {
  private baseUrl: string;
  private defaultHeaders: Record<string, string>;

  constructor(baseUrl = BASE_URL) {
    this.baseUrl = baseUrl;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    };
  }

  private async request<T = any>(
    endpoint: string,
    data?: any,
    config: RequestConfig = {}
  ): Promise<ApiResponse<T>> {
    const {
      method = data ? 'POST' : 'GET',
      headers = {},
      timeout = DEFAULT_TIMEOUT,
      signal
    } = config;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method,
        headers: { ...this.defaultHeaders, ...headers },
        body: data ? JSON.stringify(data) : undefined,
        signal: signal || controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const result: ApiResponse<T> = await response.json();
      
      if (!result.success && result.error) {
        throw new Error(result.error);
      }

      return result;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Unknown API error occurred');
    }
  }

  // === CASE MANAGEMENT ===

  async createCase(data: CaseCreateRequest): Promise<FormSubmissionResult<Case>> {
    return this.request<Case>('/cases', data, { method: 'POST' });
  }

  async updateCase(data: CaseUpdateRequest): Promise<FormSubmissionResult<Case>> {
    return this.request<Case>(`/cases/${data.id}`, data, { method: 'PUT' });
  }

  async getCase(id: string): Promise<ApiResponse<Case>> {
    return this.request<Case>(`/cases/${id}`);
  }

  async deleteCase(id: string): Promise<ApiResponse<void>> {
    return this.request(`/cases/${id}`, null, { method: 'DELETE' });
  }

  async searchCases(params: CaseSearchRequest): Promise<CaseSearchResponse> {
    return this.request<Case[]>('/cases/search', params);
  }

  async getSimilarCases(id: string): Promise<ApiResponse<Array<Case & { similarity: number }>>> {
    return this.request(`/cases/${id}/similar`);
  }

  // === EVIDENCE MANAGEMENT ===

  async createEvidence(data: EvidenceCreateRequest): Promise<FormSubmissionResult<Evidence>> {
    return this.request<Evidence>('/evidence', data, { method: 'POST' });
  }

  async getEvidence(id: string): Promise<ApiResponse<Evidence>> {
    return this.request<Evidence>(`/evidence/${id}`);
  }

  async deleteEvidence(id: string): Promise<ApiResponse<void>> {
    return this.request(`/evidence/${id}`, null, { method: 'DELETE' });
  }

  async searchEvidence(params: EvidenceSearchRequest): Promise<ApiResponse<Evidence[]>> {
    return this.request<Evidence[]>('/evidence/search', params);
  }

  async getEvidenceByCase(caseId: string): Promise<ApiResponse<Evidence[]>> {
    return this.request<Evidence[]>(`/cases/${caseId}/evidence`);
  }

  // === DOCUMENT MANAGEMENT ===

  async uploadDocument(file: File, metadata?: Record<string, any>): Promise<ApiResponse<LegalDocument>> {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    return fetch(`${this.baseUrl}/documents/upload`, {
      method: 'POST',
      body: formData
    }).then(response => response.json());
  }

  async processDocument(id: string): Promise<ApiResponse<LegalDocument>> {
    return this.request(`/documents/${id}/process`, null, { method: 'POST' });
  }

  // === SEARCH & COMMAND ===

  async globalSearch(query: string): Promise<ApiResponse<any[]>> {
    return this.request('/search', { query });
  }

  async vectorSearch(params: VectorSearchRequest): Promise<ApiResponse<VectorSearchResult[]>> {
    return this.request<VectorSearchResult[]>('/search/vector', params);
  }

  async commandSearch(params: CommandSearchRequest): Promise<CommandSearchResponse> {
    return this.request<CommandSearchResponse['results']>('/search/command', params);
  }

  // === AI & ANALYSIS ===

  async analyzeWithAI(content: string, type: 'case' | 'evidence' | 'document'): Promise<ApiResponse<any>> {
    return this.request('/ai/analyze', { content, type });
  }

  async summarizeContent(content: string): Promise<ApiResponse<{ summary: string }>> {
    return this.request('/ai/summarize', { content });
  }

  async extractEntities(content: string): Promise<ApiResponse<any[]>> {
    return this.request('/ai/extract', { content });
  }

  // === BULK OPERATIONS ===

  async bulkOperation(params: BulkOperationRequest): Promise<BulkOperationResponse> {
    return this.request<BulkOperationResponse>('/bulk', params);
  }

  // === HEALTH & METRICS ===

  async getHealth(): Promise<ApiResponse<{ status: string; services: Record<string, boolean> }>> {
    return this.request('/health');
  }

  async getMetrics(): Promise<ApiResponse<Record<string, any>>> {
    return this.request('/metrics');
  }
}

// === REACTIVE DATA STORES ===

interface DataStore<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  lastFetch: number | null;
}

export class ReactiveApiClient extends ApiClient {
  private stores = new Map<string, DataStore<any>>();
  private subscribers = new Map<string, Set<(data: DataStore<any>) => void>>();

  private getStore<T>(key: string): DataStore<T> {
    if (!this.stores.has(key)) {
      this.stores.set(key, {
        data: null,
        loading: false,
        error: null,
        lastFetch: null
      });
    }
    return this.stores.get(key)!;
  }

  private updateStore<T>(key: string, update: Partial<DataStore<T>>) {
    const store = this.getStore<T>(key);
    const newStore = { ...store, ...update };
    this.stores.set(key, newStore);
    
    // Notify subscribers
    const subs = this.subscribers.get(key);
    if (subs) {
      subs.forEach(callback => callback(newStore));
    }
  }

  subscribe<T>(key: string, callback: (data: DataStore<T>) => void): () => void {
    if (!this.subscribers.has(key)) {
      this.subscribers.set(key, new Set());
    }
    
    this.subscribers.get(key)!.add(callback);
    
    // Immediate callback with current data
    callback(this.getStore<T>(key));
    
    // Return unsubscribe function
    return () => {
      const subs = this.subscribers.get(key);
      if (subs) {
        subs.delete(callback);
        if (subs.size === 0) {
          this.subscribers.delete(key);
        }
      }
    };
  }

  async fetchCase(id: string, cache = true): Promise<Case | null> {
    const key = `case:${id}`;
    const store = this.getStore<Case>(key);
    
    // Return cached data if available and fresh
    if (cache && store.data && store.lastFetch && Date.now() - store.lastFetch < 60000) {
      return store.data;
    }

    this.updateStore<Case>(key, { loading: true, error: null });

    try {
      const response = await this.getCase(id);
      const data = response.data!;
      
      this.updateStore<Case>(key, {
        data,
        loading: false,
        error: null,
        lastFetch: Date.now()
      });
      
      return data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.updateStore<Case>(key, {
        loading: false,
        error: errorMessage
      });
      return null;
    }
  }

  async fetchEvidence(caseId: string, cache = true): Promise<Evidence[]> {
    const key = `evidence:${caseId}`;
    const store = this.getStore<Evidence[]>(key);
    
    if (cache && store.data && store.lastFetch && Date.now() - store.lastFetch < 60000) {
      return store.data;
    }

    this.updateStore<Evidence[]>(key, { loading: true, error: null });

    try {
      const response = await this.getEvidenceByCase(caseId);
      const data = response.data!;
      
      this.updateStore<Evidence[]>(key, {
        data,
        loading: false,
        error: null,
        lastFetch: Date.now()
      });
      
      return data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.updateStore<Evidence[]>(key, {
        loading: false,
        error: errorMessage
      });
      return [];
    }
  }

  async searchCasesReactive(params: CaseSearchRequest): Promise<Case[]> {
    const key = `search:cases:${JSON.stringify(params)}`;
    const store = this.getStore<Case[]>(key);
    
    this.updateStore<Case[]>(key, { loading: true, error: null });

    try {
      const response = await this.searchCases(params);
      const data = response.data!;
      
      this.updateStore<Case[]>(key, {
        data,
        loading: false,
        error: null,
        lastFetch: Date.now()
      });
      
      return data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.updateStore<Case[]>(key, {
        loading: false,
        error: errorMessage
      });
      return [];
    }
  }

  // Cache invalidation
  invalidateCache(pattern?: string) {
    if (pattern) {
      // Clear specific cache entries matching pattern
      for (const key of this.stores.keys()) {
        if (key.includes(pattern)) {
          this.stores.delete(key);
        }
      }
    } else {
      // Clear all cache
      this.stores.clear();
    }
  }
}

// Export singleton instances
export const apiClient = new ApiClient();
export const reactiveApiClient = new ReactiveApiClient();

// Export for direct instantiation
export { ApiClient, ReactiveApiClient };

export default apiClient;