/**
 * Go Service Client - Links agentShellMachine.ts to Go microservices
 * Integrates with Enhanced RAG (8094), Upload Service (8093), and Kratos Server (50051)
 */

interface GoServiceConfig {
  enhancedRagUrl: string;
  uploadServiceUrl: string;
  kratosServerUrl: string;
  timeout: number;
}

interface RAGRequest {
  query: string;
  context?: string[];
  userId?: string;
  caseId?: string;
}

interface RAGResponse {
  response: string;
  confidence: number;
  sources: string[];
  embedding?: number[];
  metadata: {
    model: string;
    processingTime: number;
    tokensUsed: number;
  };
}

interface UploadRequest {
  file: File;
  userId: string;
  caseId?: string;
  metadata?: Record<string, any>;
}

interface UploadResponse {
  fileId: string;
  url: string;
  metadata: {
    size: number;
    mimeType: string;
    uploadTime: number;
  };
}

export class GoServiceClient {
  private config: GoServiceConfig;

  constructor(config?: Partial<GoServiceConfig>) {
    this.config = {
      enhancedRagUrl: 'http://localhost:8094',
      uploadServiceUrl: 'http://localhost:8093',
      kratosServerUrl: 'http://localhost:50051',
      timeout: 30000,
      ...config
    };
  }

  /**
   * Enhanced RAG Service (8094) - AI Analysis and Search
   * Using available endpoints: /health, /ws, /api/gpu/compute, /api/som/train, /api/xstate/event
   */
  async queryRAG(request: RAGRequest): Promise<RAGResponse> {
    try {
      // Use GPU compute endpoint for RAG queries
      const response = await fetch(`${this.config.enhancedRagUrl}/api/gpu/compute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'SvelteKit-Frontend/1.0'
        },
        body: JSON.stringify({
          operation: 'rag_query',
          data: request
        }),
        signal: AbortSignal.timeout(this.config.timeout)
      });

      if (!response.ok) {
        throw new Error(`RAG service error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      
      // Transform GPU compute response to RAG format
      return {
        response: result.output || result.result || 'No response',
        confidence: result.confidence || 0.8,
        sources: result.sources || [],
        embedding: result.embedding,
        metadata: {
          model: result.model || 'enhanced-rag-gpu',
          processingTime: result.processingTime || 0,
          tokensUsed: result.tokensUsed || 0
        }
      };
    } catch (error) {
      console.error('Enhanced RAG service error:', error);
      throw new Error(`RAG query failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Upload Service (8093) - File Processing and Storage
   */
  async uploadFile(request: UploadRequest): Promise<UploadResponse> {
    try {
      const formData = new FormData();
      formData.append('file', request.file);
      formData.append('userId', request.userId);
      
      if (request.caseId) {
        formData.append('caseId', request.caseId);
      }
      
      if (request.metadata) {
        formData.append('metadata', JSON.stringify(request.metadata));
      }

      const response = await fetch(`${this.config.uploadServiceUrl}/upload`, {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(this.config.timeout)
      });

      if (!response.ok) {
        throw new Error(`Upload service error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Upload service error:', error);
      throw new Error(`File upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Semantic Search via Enhanced RAG
   */
  async semanticSearch(query: string, userId: string, options?: {
    limit?: number;
    threshold?: number;
    caseId?: string;
  }): Promise<RAGResponse> {
    return this.queryRAG({
      query: `semantic_search: ${query}`,
      userId,
      caseId: options?.caseId,
      context: options ? [`limit:${options.limit || 10}`, `threshold:${options.threshold || 0.7}`] : undefined
    });
  }

  /**
   * Accept Patch Operation (via RAG service)
   */
  async acceptPatch(patchData: {
    jobId: string;
    userId: string;
    patchContent: string;
    targetFile?: string;
  }): Promise<{ success: boolean; message: string }> {
    try {
      const response = await fetch(`${this.config.enhancedRagUrl}/api/patch/accept`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patchData),
        signal: AbortSignal.timeout(this.config.timeout)
      });

      return await response.json();
    } catch (error) {
      console.error('Patch acceptance error:', error);
      return { success: false, message: `Patch failed: ${error instanceof Error ? error.message : 'Unknown error'}` };
    }
  }

  /**
   * Rate Suggestion (feedback to AI models)
   */
  async rateSuggestion(ratingData: {
    jobId: string;
    rating: number;
    userId: string;
    feedback?: string;
  }): Promise<{ success: boolean }> {
    try {
      const response = await fetch(`${this.config.enhancedRagUrl}/api/feedback/rate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ratingData),
        signal: AbortSignal.timeout(this.config.timeout)
      });

      return await response.json();
    } catch (error) {
      console.error('Rating submission error:', error);
      return { success: false };
    }
  }

  /**
   * Health Check for all Go services
   */
  async checkServiceHealth(): Promise<{
    enhancedRAG: boolean;
    uploadService: boolean;
    kratosServer: boolean;
  }> {
    const checks = await Promise.allSettled([
      fetch(`${this.config.enhancedRagUrl}/health`, { signal: AbortSignal.timeout(5000) }),
      fetch(`${this.config.uploadServiceUrl}/health`, { signal: AbortSignal.timeout(5000) }),
      fetch(`${this.config.kratosServerUrl}/health`, { signal: AbortSignal.timeout(5000) })
    ]);

    return {
      enhancedRAG: checks[0].status === 'fulfilled' && checks[0].value.ok,
      uploadService: checks[1].status === 'fulfilled' && checks[1].value.ok,
      kratosServer: checks[2].status === 'fulfilled' && checks[2].value.ok
    };
  }

  /**
   * WebSocket connection to Enhanced RAG for real-time updates
   */
  connectRAGWebSocket(userId: string, onMessage: (data: any) => void): WebSocket {
    const ws = new WebSocket(`ws://localhost:8094/ws/${userId}`);
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('WebSocket message parse error:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    return ws;
  }
}

// Singleton instance
export const goServiceClient = new GoServiceClient();

// Export types for use in agentShellMachine.ts
export type { RAGRequest, RAGResponse, UploadRequest, UploadResponse };