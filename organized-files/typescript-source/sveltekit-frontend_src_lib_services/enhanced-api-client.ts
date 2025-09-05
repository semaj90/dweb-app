/**
 * Enhanced API Client with WebSocket Support for GPU Orchestration
 * Provides real-time communication with Go microservices and CUDA orchestrator
 */

import type {
  APIResponse,
  ServiceStatus,
  WorkerStatus,
  AITask,
  WebSocketMessage,
  WebSocketMessageType,
  LegalCase,
  Evidence,
  VectorSearchRequest,
  VectorSearchResult
} from '$lib/types';

import {
  isAPIResponse,
  isWorkerStatus,
  isAITask,
  safeGet
} from '$lib/utils/type-guards';

// Service endpoint configuration
const SERVICE_ENDPOINTS = {
  enhancedRAG: 'http://localhost:8094',
  uploadService: 'http://localhost:8093',
  gpuOrchestrator: 'http://localhost:8231',
  websocketGateway: 'ws://localhost:8232'
} as const;

export class EnhancedApiClient {
  private wsConnection: WebSocket | null = null;
  private wsReconnectAttempts = 0;
  private readonly maxReconnectAttempts = 5;
  private wsMessageHandlers = new Map<string, (message: WebSocketMessage) => void>();

  constructor(private baseUrl: string = '/api') {}

  // --- HTTP API Methods ---

  async get<T>(endpoint: string, params?: Record<string, any>): Promise<APIResponse<T>> {
    try {
      const url = new URL(`${this.baseUrl}${endpoint}`, window.location.origin);
      if (params) {
        Object.entries(params).forEach(([key, value]) => {
          if (value !== undefined) url.searchParams.set(key, String(value));
        });
      }

      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': crypto.randomUUID()
        }
      });

      return await this.handleResponse<T>(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async post<T>(endpoint: string, data?: any): Promise<APIResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': crypto.randomUUID()
        },
        body: data ? JSON.stringify(data) : undefined
      });

      return await this.handleResponse<T>(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async put<T>(endpoint: string, data: any): Promise<APIResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': crypto.randomUUID()
        },
        body: JSON.stringify(data)
      });

      return await this.handleResponse<T>(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async delete<T>(endpoint: string): Promise<APIResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'DELETE',
        headers: {
          'X-Request-ID': crypto.randomUUID()
        }
      });

      return await this.handleResponse<T>(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  // --- Legal Domain API Methods ---

  async getCases(): Promise<APIResponse<LegalCase[]>> {
    return this.get<LegalCase[]>('/cases');
  }

  async createCase(caseData: Partial<LegalCase>): Promise<APIResponse<LegalCase>> {
    return this.post<LegalCase>('/cases', caseData);
  }

  async updateCase(id: string, updates: Partial<LegalCase>): Promise<APIResponse<LegalCase>> {
    return this.put<LegalCase>(`/cases/${id}`, updates);
  }

  async deleteCase(id: string): Promise<APIResponse<void>> {
    return this.delete<void>(`/cases/${id}`);
  }

  async getEvidence(caseId: string): Promise<APIResponse<Evidence[]>> {
    return this.get<Evidence[]>(`/evidence?caseId=${caseId}`);
  }

  async uploadEvidence(file: File, caseId: string, metadata?: Record<string, any>): Promise<APIResponse<Evidence>> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('caseId', caseId);
      if (metadata) {
        formData.append('metadata', JSON.stringify(metadata));
      }

      const response = await fetch(`${this.baseUrl}/evidence/upload`, {
        method: 'POST',
        headers: {
          'X-Request-ID': crypto.randomUUID()
        },
        body: formData
      });

      return await this.handleResponse<Evidence>(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  // --- Vector Search API Methods ---

  async vectorSearch(request: VectorSearchRequest): Promise<APIResponse<VectorSearchResult[]>> {
    return this.post<VectorSearchResult[]>('/search/vector', request);
  }

  async semanticSearch(query: string, filters?: Record<string, any>): Promise<APIResponse<VectorSearchResult[]>> {
    return this.post<VectorSearchResult[]>('/search/semantic', { query, filters });
  }

  // --- AI/ML API Methods ---

  async generateEmbedding(text: string): Promise<APIResponse<number[]>> {
    return this.post<number[]>('/ai/embed', { text });
  }

  async analyzeDocument(documentId: string): Promise<APIResponse<any>> {
    return this.post<any>('/ai/analyze', { documentId });
  }

  async submitAITask(task: Partial<AITask>): Promise<APIResponse<AITask>> {
    return this.post<AITask>('/ai/tasks', task);
  }

  async getWorkerStatus(): Promise<APIResponse<WorkerStatus>> {
    return this.get<WorkerStatus>('/ai/workers/status');
  }

  // --- WebSocket Real-time Communication ---

  connectWebSocket(): Promise<boolean> {
    return new Promise((resolve, reject) => {
      try {
        if (this.wsConnection?.readyState === WebSocket.OPEN) {
          resolve(true);
          return;
        }

        this.wsConnection = new WebSocket(SERVICE_ENDPOINTS.websocketGateway);

        this.wsConnection.onopen = () => {
          console.log('WebSocket connected to GPU orchestrator');
          this.wsReconnectAttempts = 0;
          resolve(true);
        };

        this.wsConnection.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.wsConnection.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.attemptReconnect();
        };

        this.wsConnection.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        // Connection timeout
        setTimeout(() => {
          if (this.wsConnection?.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket connection timeout'));
          }
        }, 5000);

      } catch (error) {
        reject(error);
      }
    });
  }

  subscribeToGPUUpdates(callback: (status: WorkerStatus) => void): string {
    const subscriptionId = crypto.randomUUID();
    
    this.wsMessageHandlers.set(subscriptionId, (message) => {
      if (message.type === 'update' && message.payload && isWorkerStatus(message.payload)) {
        callback(message.payload);
      }
    });

    // Send subscription message
    this.sendWebSocketMessage({
      type: 'subscribe',
      payload: { topic: 'gpu-updates' },
      timestamp: Date.now(),
      requestId: subscriptionId
    });

    return subscriptionId;
  }

  subscribeToTaskUpdates(taskId: string, callback: (task: AITask) => void): string {
    const subscriptionId = crypto.randomUUID();
    
    this.wsMessageHandlers.set(subscriptionId, (message) => {
      if (message.type === 'update' && message.payload && isAITask(message.payload)) {
        if (message.payload.taskId === taskId) {
          callback(message.payload);
        }
      }
    });

    this.sendWebSocketMessage({
      type: 'subscribe',
      payload: { topic: `task-${taskId}` },
      timestamp: Date.now(),
      requestId: subscriptionId
    });

    return subscriptionId;
  }

  unsubscribe(subscriptionId: string): void {
    this.wsMessageHandlers.delete(subscriptionId);
    this.sendWebSocketMessage({
      type: 'unsubscribe',
      timestamp: Date.now(),
      requestId: subscriptionId
    });
  }

  sendWebSocketMessage(message: WebSocketMessage): void {
    if (this.wsConnection?.readyState === WebSocket.OPEN) {
      this.wsConnection.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, message not sent:', message);
    }
  }

  private handleWebSocketMessage(message: WebSocketMessage): void {
    // Handle pong responses
    if (message.type === 'pong') {
      console.debug('WebSocket pong received');
      return;
    }

    // Route message to handlers
    for (const [subscriptionId, handler] of this.wsMessageHandlers) {
      if (message.requestId === subscriptionId || !message.requestId) {
        handler(message);
      }
    }
  }

  private attemptReconnect(): void {
    if (this.wsReconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max WebSocket reconnection attempts reached');
      return;
    }

    this.wsReconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.wsReconnectAttempts), 30000);
    
    console.log(`Attempting WebSocket reconnection in ${delay}ms (attempt ${this.wsReconnectAttempts}/${this.maxReconnectAttempts})`);
    
    setTimeout(() => {
      this.connectWebSocket().catch(console.error);
    }, delay);
  }

  // --- Service Health Monitoring ---

  async getServiceStatus(): Promise<APIResponse<Record<string, ServiceStatus>>> {
    return this.get<Record<string, ServiceStatus>>('/health/services');
  }

  async pingService(serviceName: string): Promise<boolean> {
    try {
      const response = await this.get<{ status: string }>(`/health/${serviceName}`);
      return response.success && response.data?.status === 'ok';
    } catch {
      return false;
    }
  }

  // --- Response Handling ---

  private async handleResponse<T>(response: Response): Promise<APIResponse<T>> {
    const processingTime = Date.now();
    
    try {
      const data = await response.json();
      
      if (!response.ok) {
        return {
          success: false,
          error: {
            code: `HTTP_${response.status}`,
            message: data.message || response.statusText
          },
          metadata: {
            timestamp: new Date().toISOString(),
            processingTimeMs: Date.now() - processingTime
          }
        };
      }

      // Handle wrapped responses
      if (isAPIResponse(data)) {
        return data;
      }

      // Handle direct data responses
      return {
        success: true,
        data: data as T,
        metadata: {
          timestamp: new Date().toISOString(),
          processingTimeMs: Date.now() - processingTime
        }
      };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'PARSE_ERROR',
          message: error instanceof Error ? error.message : 'Failed to parse response'
        },
        metadata: {
          timestamp: new Date().toISOString(),
          processingTimeMs: Date.now() - processingTime
        }
      };
    }
  }

  private handleError<T>(error: unknown): APIResponse<T> {
    return {
      success: false,
      error: {
        code: 'CLIENT_ERROR',
        message: error instanceof Error ? error.message : 'Unknown error occurred'
      },
      metadata: {
        timestamp: new Date().toISOString(),
        processingTimeMs: 0
      }
    };
  }

  // --- Cleanup ---

  disconnect(): void {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
    this.wsMessageHandlers.clear();
  }
}

// Create singleton instance
export const apiClient = new EnhancedApiClient();

// Export for custom instances
export default EnhancedApiClient;