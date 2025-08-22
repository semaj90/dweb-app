/**
 * Real AI Service - No Mocks
 * Integrates with Ollama + Enhanced RAG services + Vector Search
 */

export interface AIServiceOptions {
  ollamaUrl?: string;
  ragServiceUrl?: string;
  timeout?: number;
}

export interface AIModelInfo {
  name: string;
  size: string;
  family: string;
  available: boolean;
}

export interface ChatRequest {
  message: string;
  sessionId?: string;
  context?: {
    conversationHistory?: any[];
    caseId?: string;
    evidenceId?: string;
  };
  options?: {
    model?: string;
    temperature?: number;
    stream?: boolean;
    useRAG?: boolean;
  };
}

export interface ChatResponse {
  response: string;
  model: string;
  timestamp: string;
  performance: {
    duration: number;
    tokens: number;
    tokensPerSecond: number;
  };
  sources?: any[];
  citations?: any[];
  suggestions?: string[];
  confidence?: number;
  executionTime?: number;
  fromCache?: boolean;
}

export interface AIHealthStatus {
  ollama: boolean;
  ragService: boolean;
  vectorSearch: boolean;
  overall: boolean;
  timestamp: string;
  models: AIModelInfo[];
}

export class RealAIService {
  private ollamaUrl: string;
  private ragServiceUrl: string;
  private timeout: number;

  constructor(options: AIServiceOptions = {}) {
    this.ollamaUrl = options.ollamaUrl || process.env.OLLAMA_URL || 'http://localhost:11434';
    this.ragServiceUrl = options.ragServiceUrl || process.env.RAG_SERVICE_URL || 'http://localhost:8094';
    this.timeout = options.timeout || 30000;
  }

  /**
   * Check if AI services are healthy
   */
  async healthCheck(): Promise<AIHealthStatus> {
    const timestamp = new Date().toISOString();
    const health = {
      ollama: false,
      ragService: false,
      vectorSearch: false,
      overall: false,
      timestamp,
      models: [] as AIModelInfo[]
    };

    // Check Ollama
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`, {
        signal: AbortSignal.timeout(5000)
      });
      if (response.ok) {
        health.ollama = true;
        const data = await response.json();
        health.models = (data.models || []).map((model: any) => ({
          name: model.name,
          size: model.size || 'unknown',
          family: model.details?.family || 'unknown',
          available: true
        }));
      }
    } catch (error) {
      console.warn('Ollama health check failed:', error);
    }

    // Check Enhanced RAG service
    try {
      const response = await fetch(`${this.ragServiceUrl}/health`, {
        signal: AbortSignal.timeout(5000)
      });
      health.ragService = response.ok;
    } catch (error) {
      console.warn('RAG service health check failed:', error);
    }

    // Check Vector Search
    try {
      const response = await fetch('/api/ai/vector-search', {
        signal: AbortSignal.timeout(5000)
      });
      health.vectorSearch = response.ok;
    } catch (error) {
      console.warn('Vector search health check failed:', error);
    }

    health.overall = health.ollama && health.ragService && health.vectorSearch;
    return health;
  }

  /**
   * Connect to AI services and verify availability
   */
  async connect(modelName: string = 'gemma3-legal'): Promise<{
    success: boolean;
    model: string;
    availableModels: string[];
    error?: string;
  }> {
    try {
      const health = await this.healthCheck();
      
      if (!health.overall) {
        const missingServices = [];
        if (!health.ollama) missingServices.push('Ollama');
        if (!health.ragService) missingServices.push('RAG Service');
        if (!health.vectorSearch) missingServices.push('Vector Search');
        
        throw new Error(`Missing services: ${missingServices.join(', ')}`);
      }

      // Verify the requested model is available
      const availableModels = health.models.map(m => m.name);
      const actualModel = availableModels.includes(modelName) ? modelName : availableModels[0];

      return {
        success: true,
        model: actualModel,
        availableModels
      };
    } catch (error) {
      return {
        success: false,
        model: '',
        availableModels: [],
        error: error instanceof Error ? error.message : 'Connection failed'
      };
    }
  }

  /**
   * Send chat message with RAG integration
   */
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const startTime = Date.now();
    
    try {
      // If RAG is enabled, first do a vector search for context
      let ragContext = null;
      if (request.options?.useRAG !== false) {
        try {
          const vectorResponse = await fetch('/api/ai/vector-search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              query: request.message,
              options: {
                maxResults: 5,
                threshold: 0.7,
                includeMetadata: true
              }
            }),
            signal: AbortSignal.timeout(10000)
          });

          if (vectorResponse.ok) {
            const vectorData = await vectorResponse.json();
            ragContext = vectorData.results;
          }
        } catch (error) {
          console.warn('Vector search failed, proceeding without RAG:', error);
        }
      }

      // Send chat request to AI endpoint
      const chatResponse = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: request.message,
          model: request.options?.model || 'gemma3-legal',
          temperature: request.options?.temperature || 0.7,
          stream: request.options?.stream || false,
          context: {
            ...request.context,
            ragContext
          }
        }),
        signal: AbortSignal.timeout(this.timeout)
      });

      if (!chatResponse.ok) {
        throw new Error(`Chat API error: ${chatResponse.status} ${chatResponse.statusText}`);
      }

      const chatData = await chatResponse.json();
      const duration = Date.now() - startTime;

      return {
        response: chatData.response,
        model: chatData.model,
        timestamp: new Date().toISOString(),
        performance: {
          duration,
          tokens: chatData.performance?.tokens || this.estimateTokens(chatData.response),
          tokensPerSecond: chatData.performance?.tokensPerSecond || 0
        },
        sources: ragContext || [],
        citations: chatData.relatedCases || [],
        suggestions: chatData.suggestions || [],
        confidence: 0.85, // Default confidence
        executionTime: duration,
        fromCache: false
      };
    } catch (error) {
      throw new Error(`AI service error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Switch to a different model
   */
  async switchModel(modelName: string): Promise<{ success: boolean; error?: string }> {
    try {
      const health = await this.healthCheck();
      const availableModels = health.models.map(m => m.name);
      
      if (!availableModels.includes(modelName)) {
        throw new Error(`Model '${modelName}' not available. Available models: ${availableModels.join(', ')}`);
      }

      // Test the model with a simple request
      const testResponse = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: 'Test message',
          model: modelName,
          temperature: 0.1
        }),
        signal: AbortSignal.timeout(10000)
      });

      if (!testResponse.ok) {
        throw new Error(`Model switch test failed: ${testResponse.status}`);
      }

      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Model switch failed'
      };
    }
  }

  /**
   * Search for similar documents using vector search
   */
  async searchSimilarDocuments(
    query: string, 
    options: {
      limit?: number;
      threshold?: number;
      collection?: string;
      filter?: Record<string, any>;
    } = {}
  ): Promise<any[]> {
    try {
      const response = await fetch('/api/ai/vector-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          options: {
            maxResults: options.limit || 5,
            threshold: options.threshold || 0.7,
            collection: options.collection || 'legal_documents',
            includeMetadata: true,
            filter: options.filter
          }
        }),
        signal: AbortSignal.timeout(15000)
      });

      if (!response.ok) {
        throw new Error(`Vector search error: ${response.status}`);
      }

      const data = await response.json();
      return data.results || [];
    } catch (error) {
      console.error('Document search failed:', error);
      return [];
    }
  }

  /**
   * Index a document for vector search
   */
  async indexDocument(document: {
    title: string;
    content: string;
    metadata?: Record<string, any>;
  }): Promise<{ success: boolean; error?: string }> {
    try {
      // Use the real vector search service to store the document
      const { vectorSearchService } = await import('./real-vector-search-service');
      
      const documentId = crypto.randomUUID();
      const result = await vectorSearchService.storeDocument(
        documentId,
        document.content,
        {
          title: document.title,
          type: 'legal_document',
          indexed_at: new Date().toISOString(),
          ...document.metadata
        }
      );

      return { success: result };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Indexing failed'
      };
    }
  }

  /**
   * Estimate token count for text
   */
  private estimateTokens(text: string): number {
    // Rough estimation: ~4 characters per token for English
    return Math.ceil(text.length / 4);
  }
}

// Export singleton instance
export const realAIService = new RealAIService();