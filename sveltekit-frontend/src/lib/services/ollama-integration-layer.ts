/**
 * Ollama Integration Layer
 * 
 * Central integration service that connects all Ollama components:
 * - OllamaChatInterface.svelte (UI component)
 * - comprehensive-ollama-summarizer.ts (main service)
 * - langchain-ollama-service.ts (LangChain integration)
 * - All existing Ollama services
 * 
 * Provides unified API for frontend components with proper wiring
 */

import { writable, type Writable } from 'svelte/store';
import { 
  comprehensiveOllamaSummarizer,
  type ComprehensiveSummaryRequest,
  type ComprehensiveSummaryResponse,
  type SummarizerStats
} from './comprehensive-ollama-summarizer';
import { langChainOllamaService } from '$lib/ai/langchain-ollama-service';
import type { ChatRequest, ChatResponse } from '$routes/api/ai/chat/+server';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

export interface IntegratedChatRequest extends ChatRequest {
  // Extended properties for enhanced functionality
  documentContext?: {
    type: 'document' | 'case' | 'evidence' | 'legal-brief' | 'contract';
    content?: string;
    metadata?: Record<string, any>;
  };
  summaryOptions?: {
    includeSummary: boolean;
    includeKeyPoints: boolean;
    includeLegalAnalysis: boolean;
    includeEmbeddings: boolean;
  };
  advancedOptions?: {
    useGPU: boolean;
    enableStreaming: boolean;
    enableCaching: boolean;
    fallbackModel?: string;
  };
}

export interface IntegratedChatResponse extends ChatResponse {
  // Enhanced response with summarization
  summary?: ComprehensiveSummaryResponse;
  integration?: {
    servicesUsed: string[];
    processingPath: string;
    performance: {
      totalTime: number;
      summaryTime?: number;
      chatTime: number;
    };
  };
}

export interface OllamaServiceStatus {
  comprehensive: { status: string; health: number };
  langchain: { status: string; connected: boolean };
  cuda: { status: string; available: boolean };
  gemma3: { status: string; model?: string };
  cluster: { status: string; nodes: number };
  streaming: { status: string; active: number };
}

// ============================================================================
// INTEGRATION LAYER SERVICE
// ============================================================================

class OllamaIntegrationLayer {
  // Stores for reactive state management
  public serviceStatus: Writable<OllamaServiceStatus>;
  public isInitialized: Writable<boolean>;
  public stats: Writable<SummarizerStats | null>;
  public activeRequests: Writable<number>;

  private _initialized = false;
  private _requestCounter = 0;

  constructor() {
    this.serviceStatus = writable({
      comprehensive: { status: 'initializing', health: 0 },
      langchain: { status: 'initializing', connected: false },
      cuda: { status: 'initializing', available: false },
      gemma3: { status: 'initializing' },
      cluster: { status: 'initializing', nodes: 0 },
      streaming: { status: 'initializing', active: 0 }
    });

    this.isInitialized = writable(false);
    this.stats = writable(null);
    this.activeRequests = writable(0);
  }

  // ========================================================================
  // INITIALIZATION
  // ========================================================================

  async initialize(): Promise<void> {
    if (this._initialized) return;

    console.log('🚀 Initializing Ollama Integration Layer...');

    try {
      // Initialize comprehensive summarizer
      await comprehensiveOllamaSummarizer.initialize();
      
      // Test LangChain connection
      const langchainConnected = await langChainOllamaService.testConnection();
      
      // Update service status
      await this.updateServiceStatus();
      
      // Update stats
      const stats = await comprehensiveOllamaSummarizer.getStats();
      this.stats.set(stats);

      this._initialized = true;
      this.isInitialized.set(true);

      console.log('✅ Ollama Integration Layer initialized successfully');

    } catch (error) {
      console.error('❌ Failed to initialize Ollama Integration Layer:', error);
      throw error;
    }
  }

  private async updateServiceStatus(): Promise<void> {
    try {
      const [summarizerHealth, summarizerStats] = await Promise.all([
        comprehensiveOllamaSummarizer.getHealth(),
        comprehensiveOllamaSummarizer.getStats()
      ]);

      const langchainConnected = await langChainOllamaService.testConnection();

      this.serviceStatus.set({
        comprehensive: { 
          status: summarizerHealth.status, 
          health: summarizerHealth.services.length * 20 
        },
        langchain: { 
          status: langchainConnected ? 'healthy' : 'unhealthy', 
          connected: langchainConnected 
        },
        cuda: { 
          status: summarizerStats.services.cuda.status, 
          available: summarizerStats.services.cuda.status === 'healthy' 
        },
        gemma3: { 
          status: summarizerStats.services.gemma3.status,
          model: summarizerStats.models.loaded.find(m => m.includes('gemma3'))
        },
        cluster: { 
          status: summarizerStats.services.cluster.status,
          nodes: summarizerStats.services.cluster.nodes 
        },
        streaming: { 
          status: summarizerStats.services.streaming.status,
          active: summarizerStats.services.streaming.activeStreams 
        }
      });

    } catch (error) {
      console.warn('Failed to update service status:', error);
    }
  }

  // ========================================================================
  // INTEGRATED CHAT FUNCTIONALITY
  // ========================================================================

  async processIntegratedChat(request: IntegratedChatRequest): Promise<IntegratedChatResponse> {
    if (!this._initialized) {
      await this.initialize();
    }

    const startTime = Date.now();
    this._requestCounter++;
    this.activeRequests.update(n => n + 1);

    try {
      // Determine processing strategy
      const strategy = this.determineProcessingStrategy(request);
      
      // Process based on strategy
      let response: IntegratedChatResponse;
      
      switch (strategy) {
        case 'comprehensive-with-summary':
          response = await this.processComprehensiveWithSummary(request);
          break;
        case 'langchain-rag':
          response = await this.processLangChainRAG(request);
          break;
        case 'streaming-enhanced':
          response = await this.processStreamingEnhanced(request);
          break;
        default:
          response = await this.processStandardChat(request);
      }

      // Add integration metadata
      response.integration = {
        servicesUsed: this.getServicesUsed(strategy),
        processingPath: strategy,
        performance: {
          totalTime: Date.now() - startTime,
          summaryTime: response.summary?.processingTime,
          chatTime: Date.now() - startTime - (response.summary?.processingTime || 0)
        }
      };

      return response;

    } finally {
      this.activeRequests.update(n => n - 1);
    }
  }

  private determineProcessingStrategy(request: IntegratedChatRequest): string {
    // Determine best processing strategy based on request
    if (request.documentContext && request.summaryOptions?.includeSummary) {
      return 'comprehensive-with-summary';
    }
    
    if (request.useRAG && request.message.length > 100) {
      return 'langchain-rag';
    }
    
    if (request.stream && request.advancedOptions?.enableStreaming) {
      return 'streaming-enhanced';
    }
    
    return 'standard-chat';
  }

  private async processComprehensiveWithSummary(
    request: IntegratedChatRequest
  ): Promise<IntegratedChatResponse> {
    
    // First, generate comprehensive summary if document context provided
    let summary: ComprehensiveSummaryResponse | undefined;
    
    if (request.documentContext) {
      const summaryRequest: ComprehensiveSummaryRequest = {
        content: request.documentContext.content || request.message,
        type: request.documentContext.type || 'document',
        context: {
          caseId: request.caseId,
          metadata: request.documentContext.metadata
        },
        options: {
          includeEmbeddings: request.summaryOptions?.includeEmbeddings || false,
          enableRAG: request.useRAG || false,
          useGPU: request.advancedOptions?.useGPU || false,
          streamResponse: false,
          cacheResult: request.advancedOptions?.enableCaching !== false,
          model: request.model
        }
      };

      summary = await comprehensiveOllamaSummarizer.generateComprehensiveSummary(summaryRequest);
    }

    // Then process the chat with summary context
    const enhancedMessage = this.enhanceMessageWithSummary(request.message, summary);
    
    const chatResponse = await this.callChatAPI({
      ...request,
      message: enhancedMessage
    });

    return {
      ...chatResponse,
      summary
    };
  }

  private async processLangChainRAG(request: IntegratedChatRequest): Promise<IntegratedChatResponse> {
    try {
      // Process document first if provided
      if (request.documentContext?.content) {
        await langChainOllamaService.processDocument(
          request.documentContext.content,
          request.documentContext.metadata || {}
        );
      }

      // Query with RAG
      const ragResult = await langChainOllamaService.queryDocuments(
        request.message,
        {
          documentTypes: request.documentContext?.type ? [request.documentContext.type] : undefined,
          maxResults: 5,
          relevanceThreshold: 0.7
        }
      );

      // Create enhanced response
      return {
        response: ragResult.answer,
        performance: {
          duration: ragResult.processingTime,
          tokens: ragResult.answer.split(' ').length,
          tokensPerSecond: (ragResult.answer.split(' ').length / ragResult.processingTime) * 1000,
          model: request.model || 'langchain-ollama'
        },
        relatedCases: ragResult.sources.map(s => s.metadata.title).filter(Boolean),
        suggestions: this.extractSuggestions(ragResult.answer)
      };

    } catch (error) {
      // Fallback to standard chat
      console.warn('LangChain RAG failed, falling back to standard chat:', error);
      return await this.processStandardChat(request);
    }
  }

  private async processStreamingEnhanced(
    request: IntegratedChatRequest
  ): Promise<IntegratedChatResponse> {
    // For streaming, we'll return a standard response but indicate streaming capability
    const response = await this.processStandardChat(request);
    
    // Add streaming metadata
    response.streaming = {
      supported: true,
      endpoint: '/api/ollama/comprehensive-summary?stream=true'
    };

    return response;
  }

  private async processStandardChat(request: IntegratedChatRequest): Promise<IntegratedChatResponse> {
    return await this.callChatAPI(request);
  }

  private async callChatAPI(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch('/api/ai/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      throw new Error(`Chat API failed: ${response.status}`);
    }

    return await response.json();
  }

  // ========================================================================
  // STREAMING FUNCTIONALITY
  // ========================================================================

  async *processStreamingChat(
    request: IntegratedChatRequest
  ): AsyncGenerator<Partial<IntegratedChatResponse>, IntegratedChatResponse> {
    
    if (!this._initialized) {
      await this.initialize();
    }

    // For streaming with document context, use comprehensive summarizer
    if (request.documentContext && request.summaryOptions?.includeSummary) {
      const summaryRequest: ComprehensiveSummaryRequest = {
        content: request.documentContext.content || request.message,
        type: request.documentContext.type || 'document',
        options: {
          streamResponse: true,
          enableRAG: request.useRAG || false,
          useGPU: request.advancedOptions?.useGPU || false
        }
      };

      const streamGenerator = comprehensiveOllamaSummarizer.generateStreamingSummary(summaryRequest);
      
      for await (const partialSummary of streamGenerator) {
        yield {
          response: partialSummary.summary || '',
          summary: partialSummary,
          streaming: { active: true }
        };
      }

      return {
        response: 'Streaming completed',
        streaming: { active: false, completed: true }
      } as IntegratedChatResponse;
    }

    // Standard streaming fallback
    throw new Error('Standard streaming not implemented - use comprehensive summary streaming');
  }

  // ========================================================================
  // UTILITY METHODS
  // ========================================================================

  private enhanceMessageWithSummary(
    originalMessage: string, 
    summary?: ComprehensiveSummaryResponse
  ): string {
    if (!summary) return originalMessage;

    return `Context Summary:
${summary.summary}

Key Points:
${summary.keyPoints.join('\n- ')}

User Question: ${originalMessage}

Please answer the question using the provided context.`;
  }

  private extractSuggestions(response: string): string[] {
    // Simple suggestion extraction
    const sentences = response.split(/[.!?]+/).filter(s => s.trim().length > 10);
    return sentences
      .filter(s => s.includes('consider') || s.includes('recommend') || s.includes('suggest'))
      .slice(0, 3)
      .map(s => s.trim());
  }

  private getServicesUsed(strategy: string): string[] {
    const serviceMap = {
      'comprehensive-with-summary': ['comprehensive-summarizer', 'chat-api'],
      'langchain-rag': ['langchain-ollama', 'embeddings', 'vector-search'],
      'streaming-enhanced': ['comprehensive-summarizer', 'streaming'],
      'standard-chat': ['chat-api']
    };

    return serviceMap[strategy as keyof typeof serviceMap] || ['chat-api'];
  }

  // ========================================================================
  // PUBLIC METHODS
  // ========================================================================

  async getServiceHealth(): Promise<OllamaServiceStatus> {
    await this.updateServiceStatus();
    return new Promise((resolve) => {
      this.serviceStatus.subscribe(status => resolve(status))();
    });
  }

  async refreshStats(): Promise<void> {
    if (this._initialized) {
      const stats = await comprehensiveOllamaSummarizer.getStats();
      this.stats.set(stats);
      await this.updateServiceStatus();
    }
  }

  async warmupServices(): Promise<void> {
    if (!this._initialized) {
      await this.initialize();
    }

    console.log('🔥 Warming up all Ollama services...');
    await comprehensiveOllamaSummarizer.warmup();
    console.log('✅ Warmup completed');
  }

  // Method for OllamaChatInterface.svelte to use
  async processChat(request: IntegratedChatRequest): Promise<IntegratedChatResponse> {
    return await this.processIntegratedChat(request);
  }

  // Method for streaming in components
  processStreamingChatForUI(
    request: IntegratedChatRequest
  ): AsyncGenerator<Partial<IntegratedChatResponse>, IntegratedChatResponse> {
    return this.processStreamingChat(request);
  }

  // Get current request count
  getCurrentRequestCount(): number {
    return this._requestCounter;
  }
}

// ============================================================================
// EXPORT SINGLETON
// ============================================================================

export const ollamaIntegrationLayer = new OllamaIntegrationLayer();

// Re-export types for easy import
export type {
  IntegratedChatRequest,
  IntegratedChatResponse,
  OllamaServiceStatus,
  ComprehensiveSummaryRequest,
  ComprehensiveSummaryResponse
};