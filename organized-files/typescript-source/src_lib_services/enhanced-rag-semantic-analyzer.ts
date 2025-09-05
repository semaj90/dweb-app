// Enhanced RAG Semantic Analyzer Service
// Provides AI-powered legal document analysis with semantic understanding

import type { 
  RAGQuery, 
  RAGResponse, 
  SearchResult, 
  EnhancedRAGService as IEnhancedRAGService,
  LegalAnalysisRequest,
  LegalAnalysisResponse,
  DocumentAnalysis 
} from '$lib/types/rag';

// Service configuration
const DEFAULT_CONFIG = {
  baseUrl: 'http://localhost:8094',
  maxResults: 10,
  similarityThreshold: 0.75,
  timeout: 30000,
  retryAttempts: 3,
  cacheEnabled: true
};

/**
 * Enhanced RAG Service Implementation
 * Connects to the Go-based Enhanced RAG service running on port 8094
 */
export class EnhancedRAGSemanticAnalyzer implements IEnhancedRAGService {
  private config: typeof DEFAULT_CONFIG;
  private cache: Map<string, any> = new Map();

  constructor(config: Partial<typeof DEFAULT_CONFIG> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Perform a RAG query against legal documents
   */
  async query(request: RAGQuery): Promise<RAGResponse> {
    const startTime = Date.now();
    
    try {
      // Check cache first
      const cacheKey = this.getCacheKey('query', request);
      if (this.config.cacheEnabled && this.cache.has(cacheKey)) {
        const cached = this.cache.get(cacheKey);
        return {
          ...cached,
          cacheHit: true,
          processingTime: Date.now() - startTime
        };
      }

      // Make request to RAG service
      const response = await this.makeRequest('/api/rag/query', {
        method: 'POST',
        body: JSON.stringify(request)
      });

      const result: RAGResponse = {
        query: request.query,
        results: response.results || [],
        response: response.answer || response.response || 'No response generated',
        timestamp: new Date(),
        processingTime: Date.now() - startTime,
        cacheHit: false,
        metadata: {
          totalDocuments: response.totalDocuments || 0,
          averageSimilarity: response.averageSimilarity || 0,
          sources: response.sources || [],
          modelUsed: response.modelUsed || 'gemma3-legal',
          tokenUsage: response.tokenUsage
        }
      };

      // Cache the result
      if (this.config.cacheEnabled) {
        this.cache.set(cacheKey, result);
      }

      return result;

    } catch (error) {
      console.error('RAG query failed:', error);
      
      return {
        query: request.query,
        results: [],
        response: `Error processing query: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
        processingTime: Date.now() - startTime,
        cacheHit: false
      };
    }
  }

  /**
   * Analyze a legal document
   */
  async analyzeDocument(request: LegalAnalysisRequest): Promise<LegalAnalysisResponse> {
    const startTime = Date.now();

    try {
      const response = await this.makeRequest('/api/analyze', {
        method: 'POST',
        body: JSON.stringify(request)
      });

      return {
        analysis: response.analysis || this.getDefaultAnalysis(),
        recommendations: response.recommendations || [],
        precedents: response.precedents || [],
        complianceIssues: response.complianceIssues || [],
        processingTime: Date.now() - startTime,
        timestamp: new Date()
      };

    } catch (error) {
      console.error('Document analysis failed:', error);
      
      return {
        analysis: this.getDefaultAnalysis(),
        recommendations: [{
          type: 'error',
          priority: 'high',
          description: `Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          actionRequired: true
        }],
        processingTime: Date.now() - startTime,
        timestamp: new Date()
      };
    }
  }

  /**
   * Generate embeddings for text
   */
  async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await this.makeRequest('/api/embeddings', {
        method: 'POST',
        body: JSON.stringify({ text })
      });

      return response.embedding || [];
    } catch (error) {
      console.error('Embedding generation failed:', error);
      return [];
    }
  }

  /**
   * Find similar documents
   */
  async findSimilarDocuments(documentId: string, limit = 5): Promise<SearchResult[]> {
    try {
      const response = await this.makeRequest(`/api/similar/${documentId}?limit=${limit}`);
      return response.results || [];
    } catch (error) {
      console.error('Similar document search failed:', error);
      return [];
    }
  }

  /**
   * Update document index
   */
  async updateDocumentIndex(documentId: string, content: string): Promise<void> {
    try {
      await this.makeRequest('/api/index/update', {
        method: 'POST',
        body: JSON.stringify({ documentId, content })
      });
    } catch (error) {
      console.error('Document index update failed:', error);
      throw error;
    }
  }

  /**
   * Get service health status
   */
  async getServiceHealth() {
    try {
      const response = await this.makeRequest('/health');
      
      return {
        status: response.status === 'healthy' ? 'healthy' as const : 'unhealthy' as const,
        services: {
          database: response.database !== 'disconnected',
          vectorStore: response.vectorStore !== false,
          llm: response.llm !== false,
          cache: response.cache !== false
        },
        uptime: response.uptime || 0
      };
    } catch (error) {
      return {
        status: 'unhealthy' as const,
        services: {
          database: false,
          vectorStore: false,
          llm: false,
          cache: false
        },
        uptime: 0
      };
    }
  }

  /**
   * Make HTTP request to RAG service
   */
  private async makeRequest(endpoint: string, options: RequestInit = {}): Promise<any> {
    const url = `${this.config.baseUrl}${endpoint}`;
    
    const defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    };

    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers
      },
      signal: AbortSignal.timeout(this.config.timeout)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Generate cache key for requests
   */
  private getCacheKey(type: string, data: any): string {
    const dataStr = JSON.stringify(data);
    const hash = this.simpleHash(dataStr);
    return `${type}:${hash}`;
  }

  /**
   * Simple hash function for cache keys
   */
  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  /**
   * Get default analysis structure
   */
  private getDefaultAnalysis(): DocumentAnalysis {
    return {
      documentId: '',
      entities: [],
      keyTerms: [],
      sentimentScore: 0,
      complexityScore: 0,
      confidenceLevel: 0,
      extractedDates: [],
      extractedAmounts: [],
      parties: [],
      obligations: [],
      risks: []
    };
  }
}

// Create and export service instance
export const enhancedRAGService = new EnhancedRAGSemanticAnalyzer();

// Export types for convenience
export type { RAGQuery, RAGResponse, SearchResult, LegalAnalysisRequest, LegalAnalysisResponse };