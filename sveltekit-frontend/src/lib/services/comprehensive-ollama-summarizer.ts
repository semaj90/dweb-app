/**
 * Comprehensive Ollama Summarizer Service
 * 
 * Unified service that integrates all Ollama components:
 * - LangChain + Ollama integration
 * - CUDA GPU acceleration 
 * - Multiple Ollama services (chat, embeddings, gemma3)
 * - OllamaChatInterface integration
 * - Performance optimization and caching
 * - Multi-model orchestration
 * 
 * Ensures app works with fully linked and wired API endpoints
 */

import type { 
  LangChainConfig, 
  ProcessingResult, 
  QueryResult 
} from '$lib/ai/langchain-ollama-service';
import { LangChainOllamaService } from '$lib/ai/langchain-ollama-service';
import { ollamaCudaService } from './ollama-cuda-service';
import { ollamaGemma3Service } from './ollama-gemma3-service';
import { ollamaService } from './ollama-service';
import { ollamaClusterService } from './ollamaClusterService';
import { ollamaChatStream } from './ollamaChatStream';
import { comprehensiveCachingService } from './comprehensive-caching-service';
import { performanceOptimizationService } from './performance-optimization-service';
import type { ChatRequest, ChatResponse } from '$routes/api/ai/chat/+server';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

export interface SummarizerConfig {
  // Core Configuration
  baseUrl: string;
  primaryModel: string;
  embeddingModel: string;
  fallbackModel?: string;
  
  // Performance Settings
  maxConcurrentRequests: number;
  requestTimeout: number;
  enableCaching: boolean;
  enableGPUAcceleration: boolean;
  
  // Model Settings
  defaultTemperature: number;
  maxTokens: number;
  contextWindow: number;
  
  // Advanced Features
  enableClustering: boolean;
  enableStreaming: boolean;
  enableFallback: boolean;
  enableMetrics: boolean;
}

export interface SummarizerStats {
  services: {
    langchain: { status: string; models: string[] };
    cuda: { status: string; gpuMemory?: number };
    gemma3: { status: string; model?: string };
    cluster: { status: string; nodes: number };
    streaming: { status: string; activeStreams: number };
  };
  performance: {
    requestsProcessed: number;
    averageLatency: number;
    cacheHitRate: number;
    errorRate: number;
  };
  models: {
    loaded: string[];
    available: string[];
    gpu: boolean;
  };
}

export interface ComprehensiveSummaryRequest {
  content: string;
  type: 'document' | 'case' | 'evidence' | 'legal-brief' | 'contract';
  context?: {
    caseId?: string;
    userId?: string;
    metadata?: Record<string, any>;
  };
  options?: {
    includeEmbeddings?: boolean;
    enableRAG?: boolean;
    useGPU?: boolean;
    streamResponse?: boolean;
    cacheResult?: boolean;
    model?: string;
  };
}

export interface ComprehensiveSummaryResponse {
  summary: string;
  keyPoints: string[];
  legalAnalysis?: {
    risks: string[];
    opportunities: string[];
    recommendations: string[];
    precedents?: string[];
  };
  embeddings?: number[][];
  confidence: number;
  processingTime: number;
  model: string;
  sources?: Array<{
    content: string;
    similarity: number;
    metadata: Record<string, any>;
  }>;
  metadata: {
    wordCount: number;
    complexity: 'low' | 'medium' | 'high';
    topKeywords: string[];
    entities: Array<{ text: string; type: string; confidence: number }>;
  };
}

// ============================================================================
// MAIN COMPREHENSIVE OLLAMA SUMMARIZER SERVICE
// ============================================================================

class ComprehensiveOllamaSummarizer {
  private config: SummarizerConfig;
  private langChainService: LangChainOllamaService;
  private isInitialized = false;
  private stats: SummarizerStats;

  constructor(config: Partial<SummarizerConfig> = {}) {
    this.config = {
      baseUrl: 'http://localhost:11434',
      primaryModel: 'gemma3-legal:latest',
      embeddingModel: 'nomic-embed-text:latest',
      fallbackModel: 'gemma2:9b',
      maxConcurrentRequests: 5,
      requestTimeout: 30000,
      enableCaching: true,
      enableGPUAcceleration: true,
      defaultTemperature: 0.3,
      maxTokens: 4096,
      contextWindow: 8192,
      enableClustering: true,
      enableStreaming: true,
      enableFallback: true,
      enableMetrics: true,
      ...config
    };

    this.initializeServices();
    this.initializeStats();
  }

  private initializeServices() {
    // Initialize LangChain service with configuration
    const langChainConfig: Partial<LangChainConfig> = {
      ollamaBaseUrl: this.config.baseUrl,
      model: this.config.primaryModel,
      embeddingModel: this.config.embeddingModel,
      temperature: this.config.defaultTemperature,
      maxTokens: this.config.maxTokens,
      useCuda: this.config.enableGPUAcceleration
    };

    this.langChainService = new LangChainOllamaService(langChainConfig);
  }

  private initializeStats() {
    this.stats = {
      services: {
        langchain: { status: 'initializing', models: [] },
        cuda: { status: 'initializing' },
        gemma3: { status: 'initializing' },
        cluster: { status: 'initializing', nodes: 0 },
        streaming: { status: 'initializing', activeStreams: 0 }
      },
      performance: {
        requestsProcessed: 0,
        averageLatency: 0,
        cacheHitRate: 0,
        errorRate: 0
      },
      models: {
        loaded: [],
        available: [],
        gpu: false
      }
    };
  }

  // ========================================================================
  // INITIALIZATION & HEALTH MANAGEMENT
  // ========================================================================

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      console.log('🚀 Initializing Comprehensive Ollama Summarizer...');

      // Initialize core services
      await this.initializeCoreServices();
      
      // Initialize advanced services (check if they exist and have initialize method)
      if (this.config.enableCaching) {
        try {
          if (typeof comprehensiveCachingService?.initialize === 'function') {
            await comprehensiveCachingService.initialize();
          }
        } catch (error) {
          console.warn('Caching service not available:', error);
        }
      }

      if (this.config.enableMetrics) {
        try {
          if (typeof performanceOptimizationService?.initialize === 'function') {
            await performanceOptimizationService.initialize();
          }
        } catch (error) {
          console.warn('Performance service not available:', error);
        }
      }

      // Test connections
      await this.validateConnections();

      // Update stats
      await this.updateStats();

      this.isInitialized = true;
      console.log('✅ Comprehensive Ollama Summarizer initialized successfully');

    } catch (error) {
      console.error('❌ Failed to initialize Comprehensive Ollama Summarizer:', error);
      throw error;
    }
  }

  private async initializeCoreServices(): Promise<void> {
    const initPromises = [];

    // Test LangChain connection
    initPromises.push(
      this.langChainService.testConnection()
        .then(connected => {
          this.stats.services.langchain.status = connected ? 'healthy' : 'unhealthy';
        })
        .catch(() => {
          this.stats.services.langchain.status = 'error';
        })
    );

    // Check CUDA service if enabled (it auto-initializes as singleton)
    if (this.config.enableGPUAcceleration) {
      initPromises.push(
        Promise.resolve()
          .then(async () => {
            if (ollamaCudaService.isInitialized) {
              this.stats.services.cuda.status = 'healthy';
            } else {
              // Try to get system health as initialization test
              const health = await ollamaCudaService.getSystemHealth();
              this.stats.services.cuda.status = health.status === 'healthy' ? 'healthy' : 'degraded';
            }
          })
          .catch(() => {
            this.stats.services.cuda.status = 'error';
          })
      );
    }

    // Check Gemma3 service (auto-initializes)
    initPromises.push(
      Promise.resolve()
        .then(() => {
          // Check if the service is available by testing a method
          if (typeof ollamaGemma3Service.generateResponse === 'function') {
            this.stats.services.gemma3.status = 'healthy';
          } else {
            this.stats.services.gemma3.status = 'degraded';
          }
        })
        .catch(() => {
          this.stats.services.gemma3.status = 'error';
        })
    );

    // Check cluster service if enabled (auto-initializes)
    if (this.config.enableClustering) {
      initPromises.push(
        Promise.resolve()
          .then(() => {
            // Check if cluster service methods are available
            if (typeof ollamaClusterService.getStats === 'function') {
              this.stats.services.cluster.status = 'healthy';
            } else {
              this.stats.services.cluster.status = 'degraded';
            }
          })
          .catch(() => {
            this.stats.services.cluster.status = 'error';
          })
      );
    }

    await Promise.allSettled(initPromises);
  }

  private async validateConnections(): Promise<void> {
    // Test Ollama API connection
    try {
      const response = await fetch(`${this.config.baseUrl}/api/tags`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      this.stats.models.available = data.models?.map((m: any) => m.name) || [];
    } catch (error) {
      console.warn('Failed to connect to Ollama API:', error);
    }

    // Test primary model
    try {
      await this.langChainService.testConnection();
      this.stats.models.loaded.push(this.config.primaryModel);
    } catch (error) {
      console.warn('Primary model not available:', error);
    }
  }

  // ========================================================================
  // MAIN SUMMARIZATION FUNCTIONALITY
  // ========================================================================

  async generateComprehensiveSummary(
    request: ComprehensiveSummaryRequest
  ): Promise<ComprehensiveSummaryResponse> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const startTime = Date.now();
    this.stats.performance.requestsProcessed++;

    try {
      // Check cache first
      let cacheKey = '';
      if (this.config.enableCaching && request.options?.cacheResult !== false) {
        try {
          if (typeof comprehensiveCachingService?.get === 'function') {
            cacheKey = this.generateCacheKey(request);
            const cached = await comprehensiveCachingService.get(cacheKey);
            if (cached) {
              this.stats.performance.cacheHitRate = 
                (this.stats.performance.cacheHitRate * (this.stats.performance.requestsProcessed - 1) + 1) / 
                this.stats.performance.requestsProcessed;
              return cached;
            }
          }
        } catch (error) {
          console.warn('Cache lookup failed:', error);
        }
      }

      // Process document with embeddings if requested
      let processingResult: ProcessingResult | null = null;
      if (request.options?.includeEmbeddings) {
        processingResult = await this.langChainService.processDocument(
          request.content,
          {
            type: request.type,
            ...request.context?.metadata
          }
        );
      }

      // Perform RAG query if requested
      let ragResult: QueryResult | null = null;
      if (request.options?.enableRAG && processingResult) {
        const query = this.generateQueryFromType(request.type);
        ragResult = await this.langChainService.queryDocuments(query, {
          documentTypes: [request.type],
          maxResults: 5
        });
      }

      // Generate summary using appropriate service
      const summary = await this.generateSummaryWithFallback(request);

      // Extract metadata
      const metadata = this.extractMetadata(request.content);

      // Compile comprehensive response
      const response: ComprehensiveSummaryResponse = {
        summary: summary.content,
        keyPoints: summary.keyPoints,
        legalAnalysis: summary.legalAnalysis,
        embeddings: processingResult?.embeddings,
        confidence: this.calculateConfidence(summary, ragResult),
        processingTime: Date.now() - startTime,
        model: summary.model,
        sources: ragResult?.sources,
        metadata
      };

      // Cache result if enabled
      if (this.config.enableCaching && cacheKey) {
        try {
          if (typeof comprehensiveCachingService?.set === 'function') {
            await comprehensiveCachingService.set(cacheKey, response, {
              ttl: 3600000, // 1 hour
              strategy: 'persistent',
              tags: [request.type, request.context?.caseId].filter(Boolean)
            });
          }
        } catch (error) {
          console.warn('Cache set failed:', error);
        }
      }

      // Update performance stats
      this.updatePerformanceStats(Date.now() - startTime);

      return response;

    } catch (error) {
      this.stats.performance.errorRate = 
        (this.stats.performance.errorRate * (this.stats.performance.requestsProcessed - 1) + 1) / 
        this.stats.performance.requestsProcessed;
      
      console.error('Failed to generate comprehensive summary:', error);
      throw error;
    }
  }

  private async generateSummaryWithFallback(
    request: ComprehensiveSummaryRequest
  ): Promise<{
    content: string;
    keyPoints: string[];
    legalAnalysis?: any;
    model: string;
  }> {
    const model = request.options?.model || this.config.primaryModel;

    try {
      // Primary: Use Gemma3 legal model if available
      if (model.includes('gemma3') && this.stats.services.gemma3.status === 'healthy') {
        return await this.generateWithGemma3Service(request);
      }

      // Secondary: Use CUDA service if GPU enabled
      if (this.config.enableGPUAcceleration && this.stats.services.cuda.status === 'healthy') {
        return await this.generateWithCudaService(request);
      }

      // Tertiary: Use LangChain service
      if (this.stats.services.langchain.status === 'healthy') {
        return await this.generateWithLangChainService(request);
      }

      // Fallback: Use basic Ollama service
      return await this.generateWithBasicService(request);

    } catch (error) {
      console.warn('Primary service failed, trying fallback:', error);
      
      if (this.config.enableFallback) {
        return await this.generateWithBasicService(request);
      }
      
      throw error;
    }
  }

  private async generateWithGemma3Service(request: ComprehensiveSummaryRequest) {
    const prompt = this.buildLegalPrompt(request);
    
    const response = await ollamaGemma3Service.generateResponse(prompt, {
      temperature: this.config.defaultTemperature,
      maxTokens: this.config.maxTokens,
      stream: request.options?.streamResponse || false
    });

    return {
      content: response.content,
      keyPoints: this.extractKeyPoints(response.content),
      legalAnalysis: this.extractLegalAnalysis(response.content),
      model: 'gemma3-legal'
    };
  }

  private async generateWithCudaService(request: ComprehensiveSummaryRequest) {
    const messages = [{
      role: 'user' as const,
      content: this.buildLegalPrompt(request)
    }];

    const response = await ollamaCudaService.chatCompletion(messages, {
      model: this.config.primaryModel,
      temperature: this.config.defaultTemperature,
      maxTokens: this.config.maxTokens,
      streaming: request.options?.streamResponse ? {
        onToken: (token) => console.log('Token:', token),
        onStart: () => console.log('Started CUDA generation'),
        onEnd: () => console.log('Completed CUDA generation')
      } : undefined
    });

    return {
      content: response.content,
      keyPoints: this.extractKeyPoints(response.content),
      legalAnalysis: this.extractLegalAnalysis(response.content),
      model: 'cuda-accelerated'
    };
  }

  private async generateWithLangChainService(request: ComprehensiveSummaryRequest) {
    const query = this.buildLegalPrompt(request);
    
    const result = await this.langChainService.queryDocuments(query, {
      documentTypes: [request.type],
      maxResults: 3
    });

    return {
      content: result.answer,
      keyPoints: this.extractKeyPoints(result.answer),
      legalAnalysis: this.extractLegalAnalysis(result.answer),
      model: 'langchain-ollama'
    };
  }

  private async generateWithBasicService(request: ComprehensiveSummaryRequest) {
    const chatRequest: ChatRequest = {
      message: this.buildLegalPrompt(request),
      model: this.config.fallbackModel || this.config.primaryModel,
      temperature: this.config.defaultTemperature,
      stream: request.options?.streamResponse || false,
      useRAG: request.options?.enableRAG || false,
      caseId: request.context?.caseId
    };

    // Use the existing API endpoint
    const response = await fetch('/api/ai/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(chatRequest)
    });

    if (!response.ok) {
      throw new Error(`Chat API failed: ${response.status}`);
    }

    const chatResponse: ChatResponse = await response.json();

    return {
      content: chatResponse.response,
      keyPoints: this.extractKeyPoints(chatResponse.response),
      legalAnalysis: this.extractLegalAnalysis(chatResponse.response),
      model: 'basic-ollama'
    };
  }

  // ========================================================================
  // STREAMING SUPPORT
  // ========================================================================

  async *generateStreamingSummary(
    request: ComprehensiveSummaryRequest
  ): AsyncGenerator<Partial<ComprehensiveSummaryResponse>, ComprehensiveSummaryResponse> {
    if (!this.config.enableStreaming) {
      throw new Error('Streaming is not enabled');
    }

    const startTime = Date.now();
    let partialContent = '';
    const chunks: string[] = [];

    try {
      // Initialize streaming
      const streamRequest = { ...request, options: { ...request.options, streamResponse: true } };
      
      // Use streaming chat service
      const stream = ollamaChatStream.createStream();
      
      // Process chunks as they arrive
      for await (const chunk of stream.processRequest(streamRequest)) {
        partialContent += chunk.content;
        chunks.push(chunk.content);
        
        // Yield partial response
        yield {
          summary: partialContent,
          keyPoints: this.extractKeyPoints(partialContent),
          confidence: this.calculatePartialConfidence(partialContent),
          processingTime: Date.now() - startTime,
          model: chunk.model || 'streaming'
        };
      }

      // Generate final comprehensive response
      const metadata = this.extractMetadata(request.content);
      
      const finalResponse: ComprehensiveSummaryResponse = {
        summary: partialContent,
        keyPoints: this.extractKeyPoints(partialContent),
        legalAnalysis: this.extractLegalAnalysis(partialContent),
        confidence: this.calculateConfidence({ content: partialContent }, null),
        processingTime: Date.now() - startTime,
        model: 'streaming-ollama',
        metadata
      };

      return finalResponse;

    } catch (error) {
      console.error('Streaming summary failed:', error);
      throw error;
    }
  }

  // ========================================================================
  // UTILITY METHODS
  // ========================================================================

  private buildLegalPrompt(request: ComprehensiveSummaryRequest): string {
    const basePrompt = `Analyze the following ${request.type} and provide a comprehensive summary with key insights:

Content:
${request.content}

Please provide:
1. A concise summary of the main content
2. Key legal points and clauses
3. Risk analysis and recommendations
4. Important dates, parties, and obligations

Format your response as a structured analysis suitable for legal professionals.`;

    // Add context-specific instructions
    switch (request.type) {
      case 'contract':
        return basePrompt + '\n\nFocus on: terms, conditions, obligations, termination clauses, liability, and dispute resolution.';
      case 'legal-brief':
        return basePrompt + '\n\nFocus on: legal arguments, precedents, evidence, and conclusions.';
      case 'case':
        return basePrompt + '\n\nFocus on: facts, legal issues, relevant laws, and potential outcomes.';
      case 'evidence':
        return basePrompt + '\n\nFocus on: relevance, credibility, admissibility, and impact on the case.';
      default:
        return basePrompt;
    }
  }

  private generateQueryFromType(type: string): string {
    const queries = {
      'document': 'What are the key legal concepts and implications in this document?',
      'case': 'What are the legal precedents and case law relevant to this matter?',
      'evidence': 'How does this evidence support or contradict legal arguments?',
      'legal-brief': 'What legal authorities and precedents support these arguments?',
      'contract': 'What are the key contractual obligations and potential risks?'
    };
    
    return queries[type as keyof typeof queries] || queries.document;
  }

  private extractKeyPoints(content: string): string[] {
    // Simple extraction logic - could be enhanced with NLP
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 20);
    const keyIndicators = ['important', 'key', 'significant', 'must', 'shall', 'required', 'critical'];
    
    return sentences
      .filter(sentence => keyIndicators.some(indicator => 
        sentence.toLowerCase().includes(indicator)
      ))
      .slice(0, 5)
      .map(s => s.trim());
  }

  private extractLegalAnalysis(content: string): any {
    // Extract structured legal analysis - could be enhanced with NLP
    const risks = this.extractSection(content, ['risk', 'danger', 'liability', 'exposure']);
    const opportunities = this.extractSection(content, ['opportunity', 'advantage', 'benefit']);
    const recommendations = this.extractSection(content, ['recommend', 'suggest', 'advise', 'should']);

    return {
      risks: risks.slice(0, 3),
      opportunities: opportunities.slice(0, 3),
      recommendations: recommendations.slice(0, 3)
    };
  }

  private extractSection(content: string, keywords: string[]): string[] {
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 15);
    return sentences
      .filter(sentence => keywords.some(keyword => 
        sentence.toLowerCase().includes(keyword)
      ))
      .map(s => s.trim());
  }

  private extractMetadata(content: string): ComprehensiveSummaryResponse['metadata'] {
    const words = content.split(/\s+/).length;
    const complexity = words > 2000 ? 'high' : words > 500 ? 'medium' : 'low';
    
    // Simple keyword extraction
    const wordFreq = new Map<string, number>();
    content.toLowerCase()
      .split(/\W+/)
      .filter(word => word.length > 4)
      .forEach(word => wordFreq.set(word, (wordFreq.get(word) || 0) + 1));
    
    const topKeywords = Array.from(wordFreq.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([word]) => word);

    return {
      wordCount: words,
      complexity,
      topKeywords,
      entities: [] // Could be enhanced with NER
    };
  }

  private calculateConfidence(summary: any, ragResult: QueryResult | null): number {
    let confidence = 0.7; // Base confidence
    
    if (summary.content.length > 200) confidence += 0.1;
    if (ragResult && ragResult.sources.length > 0) confidence += 0.15;
    if (summary.legalAnalysis) confidence += 0.05;
    
    return Math.min(confidence, 0.95);
  }

  private calculatePartialConfidence(content: string): number {
    const length = content.length;
    return Math.min(0.5 + (length / 1000) * 0.3, 0.85);
  }

  private generateCacheKey(request: ComprehensiveSummaryRequest): string {
    const hash = this.simpleHash(request.content);
    const context = request.context?.caseId || 'global';
    const options = JSON.stringify(request.options || {});
    return `summary-${request.type}-${hash}-${context}-${this.simpleHash(options)}`;
  }

  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16);
  }

  private updatePerformanceStats(processingTime: number): void {
    const currentAvg = this.stats.performance.averageLatency;
    const count = this.stats.performance.requestsProcessed;
    
    this.stats.performance.averageLatency = 
      ((currentAvg * (count - 1)) + processingTime) / count;
  }

  private async updateStats(): Promise<void> {
    try {
      // Update model information
      const response = await fetch(`${this.config.baseUrl}/api/tags`);
      if (response.ok) {
        const data = await response.json();
        this.stats.models.available = data.models?.map((m: any) => m.name) || [];
      }

      // Update service health
      const healthChecks = await Promise.allSettled([
        this.langChainService.testConnection(),
        ollamaCudaService.getSystemHealth?.() || Promise.resolve(null),
        ollamaGemma3Service.healthCheck?.() || Promise.resolve(null),
        ollamaClusterService.getStats?.() || Promise.resolve(null)
      ]);

      // Update cluster information
      if (this.config.enableClustering && healthChecks[3].status === 'fulfilled') {
        const clusterStats = healthChecks[3].value as any;
        this.stats.services.cluster.nodes = clusterStats?.activeNodes || 0;
      }

    } catch (error) {
      console.warn('Failed to update stats:', error);
    }
  }

  // ========================================================================
  // PUBLIC METHODS
  // ========================================================================

  async getStats(): Promise<SummarizerStats> {
    await this.updateStats();
    return { ...this.stats };
  }

  async getHealth(): Promise<{ status: string; services: string[] }> {
    const healthyServices = Object.entries(this.stats.services)
      .filter(([_, service]) => service.status === 'healthy')
      .map(([name]) => name);

    const status = healthyServices.length >= 2 ? 'healthy' : 'degraded';
    
    return { status, services: healthyServices };
  }

  async warmup(): Promise<void> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    console.log('🔥 Warming up Comprehensive Ollama Summarizer...');

    // Warm up with a small test request
    const testRequest: ComprehensiveSummaryRequest = {
      content: 'This is a test document for warming up the summarization service.',
      type: 'document',
      options: {
        cacheResult: false,
        streamResponse: false
      }
    };

    try {
      await this.generateComprehensiveSummary(testRequest);
      console.log('✅ Warmup completed successfully');
    } catch (error) {
      console.warn('⚠️ Warmup completed with warnings:', error);
    }
  }

  updateConfig(newConfig: Partial<SummarizerConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Reinitialize services if necessary
    if (newConfig.baseUrl || newConfig.primaryModel) {
      this.isInitialized = false;
    }
  }

  async reset(): Promise<void> {
    this.isInitialized = false;
    this.initializeStats();
    
    // Clear caches
    if (this.config.enableCaching) {
      await comprehensiveCachingService.clearByTags(['summary']);
    }
  }
}

// ============================================================================
// EXPORT SINGLETON INSTANCE
// ============================================================================

export const comprehensiveOllamaSummarizer = new ComprehensiveOllamaSummarizer();

// Export types
export type {
  SummarizerConfig,
  SummarizerStats,
  ComprehensiveSummaryRequest,
  ComprehensiveSummaryResponse
};