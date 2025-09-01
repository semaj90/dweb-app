/**
 * Unified AI Service - Integration Hub
 * Connects WASM LLM, LangChain + Ollama, NES-GPU Integration, and PostgreSQL
 */

import { browser } from '$app/environment';
import type { LegalDocument } from '$lib/gpu/nes-gpu-integration.js';
import type { LangChainConfig, ProcessingResult, QueryResult } from '$lib/ai/langchain-ollama-service.js';
import type { WASMLLMConfig, WASMLLMResponse } from '$lib/types/vector-jobs.js';

// Lazy imports to avoid SSR issues
let wasmLLMService: any = null;
let langChainOllamaService: any = null; 
let nesGPUIntegration: any = null;
let vectorOps: any = null;

// Load services dynamically - works both server and client side
const loadServices = async () => {
  try {
    if (!wasmLLMService) {
      const wasmModule = await import('$lib/wasm/wasm-llm-service.js');
      wasmLLMService = wasmModule.wasmLLMService;
    }
  } catch (error) {
    console.warn('WASM LLM service not available:', error);
  }
  
  try {
    if (!langChainOllamaService) {
      const langChainModule = await import('$lib/ai/langchain-ollama-service.js');
      langChainOllamaService = langChainModule.langChainOllamaService;
    }
  } catch (error) {
    console.warn('LangChain service not available:', error);
  }
  
  try {
    if (!nesGPUIntegration) {
      const gpuModule = await import('$lib/gpu/nes-gpu-integration.js');
      nesGPUIntegration = gpuModule.nesGPUIntegration;
    }
  } catch (error) {
    console.warn('GPU integration not available:', error);
  }

  // Only try to load vector operations server-side
  if (!browser && !vectorOps) {
    try {
      const dbModule = await import('$lib/server/db/enhanced-vector-operations.js');
      vectorOps = dbModule.vectorOps;
    } catch (error) {
      console.warn('Vector operations not available:', error);
    }
  }
};

export interface UnifiedAIConfig {
  // Service selection
  preferredMode: 'wasm' | 'langchain' | 'gpu' | 'hybrid';
  enableCaching: boolean;
  enableGPUAcceleration: boolean;
  
  // WASM Configuration
  wasmConfig?: Partial<WASMLLMConfig>;
  
  // LangChain Configuration  
  langChainConfig?: Partial<LangChainConfig>;
  
  // GPU Configuration
  gpuConfig?: {
    useNESCache: boolean;
    enableBinaryPipeline: boolean;
    batchSize: number;
  };
  
  // Database Configuration
  dbConfig?: {
    userId: string;
    enableVectorSearch: boolean;
    cacheResults: boolean;
  };
}

export interface UnifiedQueryOptions {
  query: string;
  mode?: 'wasm' | 'langchain' | 'gpu' | 'auto';
  useContext7?: boolean;
  maxResults?: number;
  threshold?: number;
  includeMetadata?: boolean;
}

export interface UnifiedResponse {
  success: boolean;
  response: string;
  mode: string;
  processingTime: number;
  sources?: Array<{
    content: string;
    metadata: Record<string, any>;
    score: number;
  }>;
  metadata?: {
    model: string;
    tokenCount?: number;
    confidence?: number;
    cacheHit?: boolean;
  };
  error?: string;
}

export class UnifiedAIService {
  private config: UnifiedAIConfig;
  private initialized = false;
  private cache = new Map<string, UnifiedResponse>();
  
  constructor(config: Partial<UnifiedAIConfig> = {}) {
    this.config = {
      preferredMode: 'hybrid',
      enableCaching: true,
      enableGPUAcceleration: true,
      wasmConfig: {
        modelPath: 'gemma3-legal',
        maxTokens: 2048,
        temperature: 0.7
      },
      langChainConfig: {
        model: 'gemma3-legal:latest',
        embeddingModel: 'nomic-embed-text:latest',
        temperature: 0.3,
        chunkSize: 1000,
        chunkOverlap: 200,
        useCuda: true
      },
      gpuConfig: {
        useNESCache: true,
        enableBinaryPipeline: true,
        batchSize: 20
      },
      dbConfig: {
        userId: 'system',
        enableVectorSearch: true,
        cacheResults: true
      },
      ...config
    };
  }

  /**
   * Initialize the unified AI service
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    try {
      await loadServices();
      
      // Initialize WASM service
      if (wasmLLMService && (this.config.preferredMode === 'wasm' || this.config.preferredMode === 'hybrid')) {
        const wasmInitialized = await wasmLLMService.initialize();
        if (wasmInitialized && this.config.wasmConfig) {
          await wasmLLMService.loadModel(this.config.wasmConfig);
        }
        console.log('✅ WASM LLM service initialized');
      }
      
      // Test LangChain connection
      if (langChainOllamaService && (this.config.preferredMode === 'langchain' || this.config.preferredMode === 'hybrid')) {
        const connected = await langChainOllamaService.testConnection();
        if (connected) {
          console.log('✅ LangChain + Ollama service initialized');
        } else {
          console.warn('⚠️ LangChain service not available - falling back to WASM');
        }
      }
      
      // Initialize GPU acceleration if available
      if (nesGPUIntegration && this.config.enableGPUAcceleration) {
        console.log('✅ NES-GPU integration available');
      }
      
      this.initialized = true;
      console.log('🚀 Unified AI Service initialized successfully');
      
    } catch (error) {
      console.error('❌ Failed to initialize Unified AI Service:', error);
      throw error;
    }
  }

  /**
   * Process a query using the optimal AI service
   */
  async query(options: UnifiedQueryOptions): Promise<UnifiedResponse> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    const startTime = performance.now();
    const cacheKey = this.generateCacheKey(options);
    
    // Check cache first
    if (this.config.enableCaching && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey)!;
      return {
        ...cached,
        metadata: { ...cached.metadata, cacheHit: true }
      };
    }
    
    try {
      let result: UnifiedResponse;
      const mode = options.mode || this.selectOptimalMode(options);
      
      switch (mode) {
        case 'wasm':
          result = await this.queryWASM(options);
          break;
        case 'langchain':
          result = await this.queryLangChain(options);
          break;
        case 'gpu':
          result = await this.queryGPU(options);
          break;
        default:
          result = await this.queryHybrid(options);
      }
      
      const processingTime = performance.now() - startTime;
      result.processingTime = processingTime;
      
      // Cache successful results
      if (this.config.enableCaching && result.success) {
        this.cache.set(cacheKey, result);
      }
      
      return result;
      
    } catch (error: any) {
      console.error('❌ Query failed:', error);
      return {
        success: false,
        response: '',
        mode: 'error',
        processingTime: performance.now() - startTime,
        error: error.message
      };
    }
  }

  /**
   * Query using WASM LLM service
   */
  private async queryWASM(options: UnifiedQueryOptions): Promise<UnifiedResponse> {
    if (!wasmLLMService) {
      throw new Error('WASM LLM service not available');
    }
    
    try {
      const wasmResponse: WASMLLMResponse = await wasmLLMService.generateText(
        options.query,
        this.config.wasmConfig
      );
      
      return {
        success: true,
        response: wasmResponse.text,
        mode: 'wasm',
        processingTime: wasmResponse.processingTimeMs,
        metadata: {
          model: wasmResponse.metadata?.model || 'wasm-model',
          tokenCount: wasmResponse.tokens,
          confidence: wasmResponse.confidence
        }
      };
    } catch (error: any) {
      throw new Error(`WASM query failed: ${error.message}`);
    }
  }

  /**
   * Query using LangChain + Ollama service
   */
  private async queryLangChain(options: UnifiedQueryOptions): Promise<UnifiedResponse> {
    if (!langChainOllamaService) {
      console.log('Loading LangChain service...');
      await loadServices();
      if (!langChainOllamaService) {
        throw new Error('LangChain service not available after loading');
      }
    }
    
    try {
      const langChainResponse: QueryResult = await langChainOllamaService.queryDocuments(
        options.query,
        {
          maxResults: options.maxResults || 10,
          relevanceThreshold: options.threshold || 0.7
        }
      );
      
      return {
        success: true,
        response: langChainResponse.answer,
        mode: 'langchain',
        processingTime: langChainResponse.processingTime,
        sources: langChainResponse.sources,
        metadata: {
          model: this.config.langChainConfig?.model || 'langchain-model',
          confidence: langChainResponse.confidence
        }
      };
    } catch (error: any) {
      throw new Error(`LangChain query failed: ${error.message}`);
    }
  }

  /**
   * Query using GPU-accelerated search
   */
  private async queryGPU(options: UnifiedQueryOptions): Promise<UnifiedResponse> {
    if (!nesGPUIntegration) {
      throw new Error('GPU integration not available');
    }
    
    try {
      const gpuResults: LegalDocument[] = await nesGPUIntegration.searchLegalDocumentsGPU(
        options.query,
        {
          limit: options.maxResults || 20,
          threshold: options.threshold || 0.7,
          useNESCache: this.config.gpuConfig?.useNESCache,
          enableGPUAcceleration: this.config.enableGPUAcceleration
        }
      );
      
      // Format GPU results as response
      const response = this.formatGPUResults(gpuResults, options.query);
      
      return {
        success: true,
        response,
        mode: 'gpu',
        processingTime: 0, // Will be set by caller
        sources: gpuResults.map(doc => ({
          content: doc.content || doc.title,
          metadata: doc.metadata || {},
          score: doc.score || 0.8
        })),
        metadata: {
          model: 'gpu-accelerated',
          confidence: 0.8
        }
      };
    } catch (error: any) {
      throw new Error(`GPU query failed: ${error.message}`);
    }
  }

  /**
   * Hybrid query using multiple services
   */
  private async queryHybrid(options: UnifiedQueryOptions): Promise<UnifiedResponse> {
    const results: UnifiedResponse[] = [];
    
    // Try GPU first for fast results
    if (nesGPUIntegration && this.config.enableGPUAcceleration) {
      try {
        const gpuResult = await this.queryGPU(options);
        if (gpuResult.success) results.push(gpuResult);
      } catch (error) {
        console.warn('GPU query failed, trying other methods:', error);
      }
    }
    
    // Try LangChain for comprehensive analysis
    if (langChainOllamaService) {
      try {
        const langChainResult = await this.queryLangChain(options);
        if (langChainResult.success) results.push(langChainResult);
      } catch (error) {
        console.warn('LangChain query failed, trying WASM:', error);
      }
    }
    
    // Try WASM as fallback
    if (wasmLLMService && results.length === 0) {
      try {
        const wasmResult = await this.queryWASM(options);
        if (wasmResult.success) results.push(wasmResult);
      } catch (error) {
        console.warn('WASM query failed:', error);
      }
    }
    
    if (results.length === 0) {
      throw new Error('All query methods failed');
    }
    
    // Combine results for best response
    return this.combineResults(results, options);
  }

  /**
   * Ingest documents into the unified system
   */
  async ingestDocuments(documents: LegalDocument[]): Promise<{
    success: boolean;
    processed: number;
    errors: number;
    processingTime: number;
  }> {
    const startTime = performance.now();
    let processed = 0;
    let errors = 0;
    
    try {
      // Process with GPU binary pipeline if available
      if (nesGPUIntegration && this.config.enableGPUAcceleration) {
        await nesGPUIntegration.ingestLegalDocumentsBinary(documents);
        processed = documents.length;
      }
      
      // Process with LangChain for text analysis
      if (langChainOllamaService) {
        for (const doc of documents) {
          try {
            if (doc.content) {
              await langChainOllamaService.processDocument(doc.content, {
                documentId: doc.id,
                title: doc.title,
                type: doc.type
              });
              processed++;
            }
          } catch (error) {
            console.error(`Failed to process document ${doc.id}:`, error);
            errors++;
          }
        }
      }
      
      return {
        success: errors === 0,
        processed,
        errors,
        processingTime: performance.now() - startTime
      };
      
    } catch (error: any) {
      console.error('Document ingestion failed:', error);
      return {
        success: false,
        processed,
        errors: documents.length - processed,
        processingTime: performance.now() - startTime
      };
    }
  }

  /**
   * Get performance statistics
   */
  async getStats() {
    const stats: any = {
      initialized: this.initialized,
      cacheSize: this.cache.size,
      config: this.config
    };
    
    if (wasmLLMService) {
      stats.wasm = wasmLLMService.getStats();
    }
    
    if (langChainOllamaService) {
      stats.langchain = langChainOllamaService.getStats();
    }
    
    if (nesGPUIntegration) {
      stats.gpu = await nesGPUIntegration.getPerformanceStats();
    }
    
    return stats;
  }

  // Helper methods
  private selectOptimalMode(options: UnifiedQueryOptions): 'wasm' | 'langchain' | 'gpu' | 'hybrid' {
    // Simple heuristics for mode selection
    if (options.query.length < 100 && wasmLLMService) return 'wasm';
    if (options.maxResults && options.maxResults > 20 && nesGPUIntegration) return 'gpu';
    if (langChainOllamaService) return 'langchain';
    return 'hybrid';
  }

  private generateCacheKey(options: UnifiedQueryOptions): string {
    return `${options.query}_${options.mode || 'auto'}_${options.maxResults || 10}_${options.threshold || 0.7}`;
  }

  private formatGPUResults(documents: LegalDocument[], query: string): string {
    if (documents.length === 0) {
      return `No relevant legal documents found for query: "${query}"`;
    }
    
    let response = `Found ${documents.length} relevant legal documents:\n\n`;
    
    documents.slice(0, 5).forEach((doc, index) => {
      response += `${index + 1}. ${doc.title}\n`;
      if (doc.summary) {
        response += `   Summary: ${doc.summary}\n`;
      }
      if (doc.riskLevel) {
        response += `   Risk Level: ${doc.riskLevel}\n`;
      }
      response += '\n';
    });
    
    return response;
  }

  private combineResults(results: UnifiedResponse[], options: UnifiedQueryOptions): UnifiedResponse {
    // For now, return the best result based on confidence
    const best = results.reduce((prev, current) => 
      (current.metadata?.confidence || 0) > (prev.metadata?.confidence || 0) ? current : prev
    );
    
    return {
      ...best,
      mode: 'hybrid',
      sources: results.flatMap(r => r.sources || [])
    };
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
    console.log('🧹 Unified AI Service cache cleared');
  }

  /**
   * Dispose of resources
   */
  dispose(): void {
    this.cache.clear();
    
    if (wasmLLMService) {
      wasmLLMService.dispose();
    }
    
    if (langChainOllamaService) {
      langChainOllamaService.reset();
    }
    
    if (nesGPUIntegration) {
      nesGPUIntegration.dispose();
    }
    
    this.initialized = false;
    console.log('🧹 Unified AI Service disposed');
  }
}

// Export singleton instance
export const unifiedAIService = new UnifiedAIService();