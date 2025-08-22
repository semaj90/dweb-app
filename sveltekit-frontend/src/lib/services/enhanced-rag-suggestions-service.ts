import { dev } from '$app/environment';

export interface RAGSuggestionRequest {
  content: string;
  reportType: string;
  context?: {
    caseId?: string;
    evidenceIds?: string[];
    userId?: string;
  };
  vectorSearchEnabled?: boolean;
  maxSuggestions?: number;
  confidenceThreshold?: number;
}

export interface RAGSuggestionResponse {
  suggestions: RAGSuggestion[];
  ragContext: RAGContext;
  processingMetrics: RAGProcessingMetrics;
  model: string;
  timestamp: string;
  requestId: string;
}

export interface RAGSuggestion {
  content: string;
  type: string;
  confidence: number;
  reasoning: string;
  supportingContext: string[];
  relevantCitations: string[];
  metadata: {
    source: 'rag_analysis' | 'vector_context' | 'legal_precedent';
    contextDocuments?: string[];
    similarityScores?: number[];
    category: string;
    priority: number;
  };
}

export interface RAGContext {
  retrievedDocuments: RetrievedDocument[];
  vectorSimilarityResults: VectorSimilarityResult[];
  legalPrecedents: LegalPrecedent[];
  contextualFactors: ContextualFactor[];
}

export interface RetrievedDocument {
  id: string;
  title: string;
  content: string;
  documentType: string;
  relevanceScore: number;
  metadata: Record<string, any>;
}

export interface VectorSimilarityResult {
  documentId: string;
  similarity: number;
  snippet: string;
  documentType: string;
}

export interface LegalPrecedent {
  id: string;
  caseTitle: string;
  citation: string;
  relevantLaw: string;
  applicationContext: string;
  confidence: number;
}

export interface ContextualFactor {
  factor: string;
  importance: number;
  reasoning: string;
  supportingEvidence: string[];
}

export interface RAGProcessingMetrics {
  totalProcessingTimeMs: number;
  vectorSearchTimeMs: number;
  documentRetrievalTimeMs: number;
  aiGenerationTimeMs: number;
  documentsRetrieved: number;
  vectorResultsCount: number;
  tokensProcessed: number;
}

/**
 * Enhanced RAG (Retrieval-Augmented Generation) Suggestions Service
 * Integrates with the Go Enhanced RAG microservice for advanced legal analysis
 */
export class EnhancedRAGSuggestionsService {
  private readonly baseUrl: string;
  private readonly timeout: number = 45000; // 45 seconds for complex RAG operations
  private readonly retryAttempts: number = 2;

  constructor(baseUrl: string = 'http://localhost:8094') {
    this.baseUrl = baseUrl;
  }

  /**
   * Generate comprehensive legal suggestions using Enhanced RAG
   */
  async generateRAGSuggestions(request: RAGSuggestionRequest): Promise<RAGSuggestionResponse> {
    const startTime = Date.now();
    let attempt = 0;

    while (attempt < this.retryAttempts) {
      try {
        // Transform request to match Go service format
        const goServiceRequest = {
          query: request.content,
          case_id: request.context?.caseId || '',
          limit: request.maxSuggestions || 5,
          metadata: {
            reportType: request.reportType,
            userId: request.context?.userId,
            evidenceIds: request.context?.evidenceIds,
            vectorSearchEnabled: request.vectorSearchEnabled,
            confidenceThreshold: request.confidenceThreshold
          }
        };
        
        const response = await this.callEnhancedRAGService('/api/rag/search', goServiceRequest);
        
        // Enhance the response with additional processing
        const enhancedResponse = await this.enhanceRAGResponse(response, request);
        
        return enhancedResponse;
      } catch (error) {
        attempt++;
        console.warn(`Enhanced RAG attempt ${attempt} failed:`, error);
        
        if (attempt >= this.retryAttempts) {
          console.error('All Enhanced RAG attempts failed, falling back to local processing');
          return await this.fallbackLocalProcessing(request, startTime);
        }
        
        // Wait before retry (exponential backoff)
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
      }
    }

    throw new Error('Enhanced RAG service unavailable');
  }

  /**
   * Stream suggestions in real-time for immediate feedback
   */
  async *streamRAGSuggestions(request: RAGSuggestionRequest): AsyncGenerator<RAGSuggestion> {
    try {
      const response = await fetch(`${this.baseUrl}/api/rag/suggestions/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`Enhanced RAG streaming failed: ${response.status} ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Unable to read RAG stream');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.substring(6));
                if (data.suggestion) {
                  yield data.suggestion as RAGSuggestion;
                }
              } catch (parseError) {
                console.warn('Failed to parse RAG stream data:', line);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      console.error('RAG streaming failed:', error);
      
      // Fallback to non-streaming
      const response = await this.generateRAGSuggestions(request);
      for (const suggestion of response.suggestions) {
        yield suggestion;
      }
    }
  }

  /**
   * Call the Enhanced RAG microservice
   */
  private async callEnhancedRAGService(endpoint: string, data: any): Promise<any> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Service-Client': 'sveltekit-frontend',
          'X-Request-ID': crypto.randomUUID()
        },
        body: JSON.stringify({
          ...data,
          timestamp: new Date().toISOString(),
          clientVersion: '1.0.0'
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Enhanced RAG service error: ${response.status} - ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new Error('Enhanced RAG request timed out');
      }
      throw error;
    }
  }

  /**
   * Enhance RAG response with additional processing
   */
  private async enhanceRAGResponse(
    response: any,
    request: RAGSuggestionRequest
  ): Promise<RAGSuggestionResponse> {
    // Add confidence scoring and ranking
    const enhancedSuggestions = response.suggestions.map((suggestion: any, index: number) => ({
      ...suggestion,
      metadata: {
        ...suggestion.metadata,
        priority: this.calculatePriority(suggestion, request),
        enhanced: true,
        processingOrder: index + 1
      }
    }));

    // Sort by confidence and priority
    enhancedSuggestions.sort((a: any, b: any) => {
      const scoreA = (a.confidence * 0.7) + (a.metadata.priority * 0.3);
      const scoreB = (b.confidence * 0.7) + (b.metadata.priority * 0.3);
      return scoreB - scoreA;
    });

    return {
      suggestions: enhancedSuggestions.slice(0, request.maxSuggestions || 5),
      ragContext: response.ragContext || this.createDefaultRAGContext(),
      processingMetrics: response.processingMetrics || this.createDefaultMetrics(),
      model: response.model || 'enhanced-rag-v1',
      timestamp: new Date().toISOString(),
      requestId: crypto.randomUUID()
    };
  }

  /**
   * Calculate suggestion priority based on content and context
   */
  private calculatePriority(suggestion: any, request: RAGSuggestionRequest): number {
    let priority = 1;

    // Higher priority for procedural and legal compliance suggestions
    if (suggestion.type?.includes('procedural') || suggestion.type?.includes('compliance')) {
      priority += 2;
    }

    // Higher priority for evidence-related suggestions
    if (suggestion.type?.includes('evidence') || suggestion.content?.toLowerCase().includes('evidence')) {
      priority += 1.5;
    }

    // Higher priority for high-confidence suggestions
    if (suggestion.confidence > 0.8) {
      priority += 1;
    }

    // Higher priority for report-type specific suggestions
    if (request.reportType === 'prosecution_memo' && suggestion.type?.includes('prosecution')) {
      priority += 1;
    }

    return Math.min(priority, 5); // Cap at 5
  }

  /**
   * Fallback local processing when Enhanced RAG service is unavailable
   */
  private async fallbackLocalProcessing(
    request: RAGSuggestionRequest,
    startTime: number
  ): Promise<RAGSuggestionResponse> {
    console.warn('Using fallback local processing for RAG suggestions');

    const suggestions: RAGSuggestion[] = [
      {
        content: 'Consider consulting the Enhanced RAG service for comprehensive legal analysis. Service is currently unavailable - using basic fallback suggestions.',
        type: 'service_unavailable',
        confidence: 0.5,
        reasoning: 'Enhanced RAG microservice is not accessible',
        supportingContext: [],
        relevantCitations: [],
        metadata: {
          source: 'rag_analysis',
          category: 'system_notice',
          priority: 1
        }
      },
      {
        content: 'Ensure all legal analysis includes proper citation of relevant statutes and case law.',
        type: 'legal_citation',
        confidence: 0.7,
        reasoning: 'Standard requirement for legal documents',
        supportingContext: ['Legal writing best practices'],
        relevantCitations: [],
        metadata: {
          source: 'rag_analysis',
          category: 'legal_standards',
          priority: 2
        }
      }
    ];

    return {
      suggestions,
      ragContext: this.createDefaultRAGContext(),
      processingMetrics: {
        totalProcessingTimeMs: Date.now() - startTime,
        vectorSearchTimeMs: 0,
        documentRetrievalTimeMs: 0,
        aiGenerationTimeMs: 0,
        documentsRetrieved: 0,
        vectorResultsCount: 0,
        tokensProcessed: 0
      },
      model: 'fallback-local-v1',
      timestamp: new Date().toISOString(),
      requestId: crypto.randomUUID()
    };
  }

  /**
   * Create default RAG context when service is unavailable
   */
  private createDefaultRAGContext(): RAGContext {
    return {
      retrievedDocuments: [],
      vectorSimilarityResults: [],
      legalPrecedents: [],
      contextualFactors: [
        {
          factor: 'Service Availability',
          importance: 1.0,
          reasoning: 'Enhanced RAG service is currently unavailable',
          supportingEvidence: ['Connection timeout', 'Service unreachable']
        }
      ]
    };
  }

  /**
   * Create default processing metrics
   */
  private createDefaultMetrics(): RAGProcessingMetrics {
    return {
      totalProcessingTimeMs: 0,
      vectorSearchTimeMs: 0,
      documentRetrievalTimeMs: 0,
      aiGenerationTimeMs: 0,
      documentsRetrieved: 0,
      vectorResultsCount: 0,
      tokensProcessed: 0
    };
  }

  /**
   * Check if Enhanced RAG service is healthy and available
   */
  async healthCheck(): Promise<{ 
    available: boolean; 
    version?: string; 
    capabilities?: string[];
    responseTime?: number;
  }> {
    const startTime = Date.now();
    
    try {
      const response = await fetch(`${this.baseUrl}/api/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });

      if (response.ok) {
        const healthData = await response.json();
        return {
          available: true,
          version: healthData.version,
          capabilities: healthData.capabilities || [],
          responseTime: Date.now() - startTime
        };
      } else {
        return { available: false, responseTime: Date.now() - startTime };
      }
    } catch (error) {
      console.error('Enhanced RAG health check failed:', error);
      return { available: false, responseTime: Date.now() - startTime };
    }
  }

  /**
   * Get Enhanced RAG service configuration and status
   */
  getServiceInfo(): {
    baseUrl: string;
    timeout: number;
    retryAttempts: number;
  } {
    return {
      baseUrl: this.baseUrl,
      timeout: this.timeout,
      retryAttempts: this.retryAttempts
    };
  }
}

// Singleton instance
export const enhancedRAGSuggestionsService = new EnhancedRAGSuggestionsService();

/**
 * Convenience function for generating RAG-powered suggestions
 */
export async function generateEnhancedRAGSuggestions(
  content: string,
  reportType: string,
  options: Partial<RAGSuggestionRequest> = {}
): Promise<RAGSuggestionResponse> {
  const request: RAGSuggestionRequest = {
    content,
    reportType,
    vectorSearchEnabled: true,
    maxSuggestions: 5,
    confidenceThreshold: 0.6,
    ...options
  };

  return await enhancedRAGSuggestionsService.generateRAGSuggestions(request);
}

/**
 * Test Enhanced RAG integration
 */
export async function testEnhancedRAGIntegration(): Promise<{
  success: boolean;
  serviceAvailable: boolean;
  version?: string;
  testSuggestion?: RAGSuggestion;
  responseTime?: number;
  error?: string;
}> {
  try {
    const healthCheck = await enhancedRAGSuggestionsService.healthCheck();
    
    if (!healthCheck.available) {
      return {
        success: false,
        serviceAvailable: false,
        responseTime: healthCheck.responseTime,
        error: 'Enhanced RAG service is not available'
      };
    }

    // Test with a simple request
    const testResponse = await enhancedRAGSuggestionsService.generateRAGSuggestions({
      content: 'The defendant was found with stolen property. We need to analyze the evidence and prepare charges.',
      reportType: 'prosecution_memo',
      maxSuggestions: 1,
      confidenceThreshold: 0.5
    });

    return {
      success: true,
      serviceAvailable: true,
      version: healthCheck.version,
      testSuggestion: testResponse.suggestions[0],
      responseTime: testResponse.processingMetrics.totalProcessingTimeMs
    };
  } catch (error) {
    return {
      success: false,
      serviceAvailable: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}