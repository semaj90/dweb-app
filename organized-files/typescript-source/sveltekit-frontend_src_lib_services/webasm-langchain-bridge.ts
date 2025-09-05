// WebAssembly LangChain Bridge
// Integrates LangChain RAG with WebAssembly llama.cpp inference

import { webAssemblyAIAdapter, type WebAssemblyAIResponse } from '../adapters/webasm-ai-adapter.js'
import { legalRAG, type RAGResult, type RAGQueryOptions } from '../ai/langchain-rag.js'
import type { Document as LangChainDocumentType } from "@langchain/core/documents";
import type { LegalDocumentMetadata } from '../ai/qdrant-service.js';

export interface WebAssemblyLangChainConfig {
  useWebAssemblyForGeneration: boolean;
  useWebAssemblyForEmbeddings: boolean;
  fallbackToOllama: boolean;
  hybridMode: boolean;
  confidenceThreshold: number;
}

export interface HybridRAGResult extends RAGResult {
  metadata: RAGResult['metadata'] & {
    usedWebAssembly: boolean;
    processingMethod: 'webassembly' | 'ollama' | 'hybrid';
    webAssemblyConfidence?: number;
    ollamaConfidence?: number;
  };
}

export interface WebAssemblyRAGContext {
  retrievedDocuments: LangChainDocumentType[];
  documentSummaries: string[];
  keyTerms: string[];
  entities: Array<{ type: string; value: string; confidence: number }>;
  contextLength: number;
}

/**
 * Bridge service that combines LangChain's document retrieval with WebAssembly inference
 */
export class WebAssemblyLangChainBridge {
  private config: WebAssemblyLangChainConfig;
  private initialized = false;

  constructor(config: Partial<WebAssemblyLangChainConfig> = {}) {
    this.config = {
      useWebAssemblyForGeneration: true,
      useWebAssemblyForEmbeddings: false, // Keep using Ollama for embeddings for now
      fallbackToOllama: true,
      hybridMode: false,
      confidenceThreshold: 0.7,
      ...config
    };
  }

  /**
   * Initialize the bridge
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) {
      return true;
    }

    try {
      // Initialize WebAssembly adapter
      const webAssemblyReady = await webAssemblyAIAdapter.initialize();
      if (!webAssemblyReady && !this.config.fallbackToOllama) {
        throw new Error('WebAssembly initialization failed and fallback disabled');
      }

      // Test LangChain RAG health
      const ragHealth = await legalRAG.healthCheck();
      console.log('[WebAssembly LangChain Bridge] RAG health:', ragHealth);

      this.initialized = true;
      console.log('[WebAssembly LangChain Bridge] Bridge initialized successfully');
      return true;

    } catch (error) {
      console.error('[WebAssembly LangChain Bridge] Initialization failed:', error);
      return false;
    }
  }

  /**
   * Enhanced RAG query using WebAssembly + LangChain
   */
  async query(
    question: string, 
    options: RAGQueryOptions & { 
      useWebAssembly?: boolean;
      useHybridMode?: boolean;
    } = {}
  ): Promise<HybridRAGResult> {
    if (!this.initialized) {
      await this.initialize();
    }

    const startTime = Date.now();
    const useWebAssembly = options.useWebAssembly ?? this.config.useWebAssemblyForGeneration;
    const useHybrid = options.useHybridMode ?? this.config.hybridMode;

    try {
      // Step 1: Use LangChain for document retrieval and context preparation
      console.log('[Bridge] Retrieving documents with LangChain...');
      const ragContext = await this.retrieveDocumentsWithContext(question, options);

      // Step 2: Choose processing method
      if (useHybrid) {
        return await this.hybridProcessing(question, ragContext, options);
      } else if (useWebAssembly && webAssemblyAIAdapter.isSupported()) {
        return await this.webAssemblyProcessing(question, ragContext, options);
      } else {
        return await this.ollamaProcessing(question, options);
      }

    } catch (error: any) {
      console.error('[Bridge] Query processing failed:', error);
      
      // Fallback to standard LangChain RAG
      if (this.config.fallbackToOllama) {
        console.log('[Bridge] Falling back to standard LangChain RAG...');
        const fallbackResult = await legalRAG.query(question, options);
        return {
          ...fallbackResult,
          metadata: {
            ...fallbackResult.metadata,
            usedWebAssembly: false,
            processingMethod: 'ollama'
          }
        };
      }

      throw error;
    }
  }

  /**
   * Retrieve documents and prepare context using LangChain
   */
  private async retrieveDocumentsWithContext(
    question: string,
    options: RAGQueryOptions
  ): Promise<WebAssemblyRAGContext> {
    // Use LangChain's retrieval capabilities without generation
    const { maxRetrievedDocs = 5, documentType, jurisdiction, practiceArea } = options;

    try {
      // Mock document retrieval for now - in production this would use the actual vector store
      const mockDocuments: LangChainDocumentType[] = [
        {
          pageContent: `Legal document content related to: ${question}. This is a comprehensive analysis of the relevant legal principles and precedents.`,
          metadata: {
            documentId: 'doc_001',
            documentType: documentType || 'contract',
            jurisdiction: jurisdiction || 'federal',
            practiceArea: practiceArea || 'general',
            confidence: 0.9
          }
        },
        {
          pageContent: `Additional context about ${question} including regulatory requirements and compliance considerations.`,
          metadata: {
            documentId: 'doc_002',
            documentType: 'regulation',
            jurisdiction: jurisdiction || 'federal',
            practiceArea: practiceArea || 'general',
            confidence: 0.85
          }
        }
      ];

      // Generate document summaries
      const documentSummaries = mockDocuments.map((doc, index) => 
        `Document ${index + 1}: ${doc.pageContent.substring(0, 200)}...`
      );

      // Extract key terms (simplified)
      const keyTerms = this.extractKeyTerms(mockDocuments.map(d => d.pageContent).join(' '));

      // Extract entities (simplified)
      const entities = this.extractEntities(mockDocuments.map(d => d.pageContent).join(' '));

      return {
        retrievedDocuments: mockDocuments,
        documentSummaries,
        keyTerms,
        entities,
        contextLength: mockDocuments.reduce((acc, doc) => acc + doc.pageContent.length, 0)
      };

    } catch (error: any) {
      console.error('[Bridge] Document retrieval failed:', error);
      throw error;
    }
  }

  /**
   * Process using WebAssembly inference with LangChain context
   */
  private async webAssemblyProcessing(
    question: string,
    context: WebAssemblyRAGContext,
    options: RAGQueryOptions
  ): Promise<HybridRAGResult> {
    console.log('[Bridge] Processing with WebAssembly...');

    // Build enhanced prompt with retrieved context
    const enhancedPrompt = this.buildEnhancedPrompt(question, context, options);

    try {
      const response: WebAssemblyAIResponse = await webAssemblyAIAdapter.sendMessage(enhancedPrompt, {
        model: 'gemma3-legal',
        temperature: 0.1,
        maxTokens: 4096
      });

      return {
        answer: response.content,
        sourceDocuments: context.retrievedDocuments,
        confidence: response.metadata.confidence,
        reasoning: `WebAssembly inference with ${context.retrievedDocuments.length} retrieved documents`,
        metadata: {
          retrievedChunks: context.retrievedDocuments.length,
          processingTime: response.metadata.processingTime,
          usedThinkingMode: options.thinkingMode || false,
          usedCompression: options.useCompression || false,
          usedWebAssembly: true,
          processingMethod: 'webassembly',
          webAssemblyConfidence: response.metadata.confidence
        }
      };

    } catch (error: any) {
      console.error('[Bridge] WebAssembly processing failed:', error);
      throw error;
    }
  }

  /**
   * Hybrid processing: combine WebAssembly and Ollama results
   */
  private async hybridProcessing(
    question: string,
    context: WebAssemblyRAGContext,
    options: RAGQueryOptions
  ): Promise<HybridRAGResult> {
    console.log('[Bridge] Processing with hybrid mode...');

    const [webAssemblyResult, ollamaResult] = await Promise.allSettled([
      this.webAssemblyProcessing(question, context, options),
      this.ollamaProcessing(question, options)
    ]);

    // Determine which result to use based on confidence
    let primaryResult: HybridRAGResult;
    let secondaryConfidence: number | undefined;

    if (webAssemblyResult.status === 'fulfilled' && ollamaResult.status === 'fulfilled') {
      if (webAssemblyResult.value.confidence >= ollamaResult.value.confidence) {
        primaryResult = webAssemblyResult.value;
        secondaryConfidence = ollamaResult.value.confidence;
      } else {
        primaryResult = {
          ...ollamaResult.value,
          metadata: {
            ...ollamaResult.value.metadata,
            usedWebAssembly: false,
            processingMethod: 'hybrid' as const,
            ollamaConfidence: ollamaResult.value.confidence
          }
        };
        secondaryConfidence = webAssemblyResult.value.confidence;
      }
    } else if (webAssemblyResult.status === 'fulfilled') {
      primaryResult = webAssemblyResult.value;
    } else if (ollamaResult.status === 'fulfilled') {
      primaryResult = {
        ...ollamaResult.value,
        metadata: {
          ...ollamaResult.value.metadata,
          usedWebAssembly: false,
          processingMethod: 'hybrid' as const
        }
      };
    } else {
      throw new Error('Both WebAssembly and Ollama processing failed');
    }

    // Enhance result with hybrid metadata
    primaryResult.metadata = {
      ...primaryResult.metadata,
      processingMethod: 'hybrid',
      webAssemblyConfidence: webAssemblyResult.status === 'fulfilled' ? webAssemblyResult.value.confidence : undefined,
      ollamaConfidence: ollamaResult.status === 'fulfilled' ? ollamaResult.value.confidence : undefined
    };

    primaryResult.reasoning = `Hybrid processing: ${primaryResult.reasoning}. Secondary confidence: ${secondaryConfidence}`;

    return primaryResult;
  }

  /**
   * Fallback to standard Ollama processing
   */
  private async ollamaProcessing(
    question: string,
    options: RAGQueryOptions
  ): Promise<HybridRAGResult> {
    console.log('[Bridge] Processing with Ollama fallback...');
    
    const result = await legalRAG.query(question, options);
    
    return {
      ...result,
      metadata: {
        ...result.metadata,
        usedWebAssembly: false,
        processingMethod: 'ollama'
      }
    };
  }

  /**
   * Build enhanced prompt with retrieved context
   */
  private buildEnhancedPrompt(
    question: string,
    context: WebAssemblyRAGContext,
    options: RAGQueryOptions
  ): string {
    const { thinkingMode, verbose } = options;

    let prompt = '<|system|>You are a specialized legal AI assistant. ';
    
    if (thinkingMode) {
      prompt += 'Provide comprehensive step-by-step legal analysis. ';
    } else if (verbose) {
      prompt += 'Provide detailed explanations with legal background. ';
    }
    
    prompt += 'Base your response on the provided legal document context.<|end|>\n\n';

    // Add retrieved context
    prompt += '<|context|>\n';
    prompt += 'Retrieved Legal Documents:\n';
    context.retrievedDocuments.forEach((doc, index) => {
      prompt += `\nDocument ${index + 1}:\n${doc.pageContent}\n`;
    });

    if (context.keyTerms.length > 0) {
      prompt += `\nKey Legal Terms: ${context.keyTerms.join(', ')}\n`;
    }

    if (context.entities.length > 0) {
      prompt += `\nIdentified Entities:\n`;
      context.entities.forEach(entity => {
        prompt += `- ${entity.type}: ${entity.value} (confidence: ${entity.confidence})\n`;
      });
    }

    prompt += '<|end|>\n\n';

    // Add user question
    prompt += `<|user|>${question}<|end|>\n<|assistant|>`;

    return prompt;
  }

  /**
   * Extract key terms from text (simplified implementation)
   */
  private extractKeyTerms(text: string): string[] {
    const legalTerms = [
      'contract', 'agreement', 'liability', 'indemnification', 'termination',
      'breach', 'damages', 'jurisdiction', 'governing law', 'force majeure',
      'confidentiality', 'intellectual property', 'warranty', 'representation',
      'compliance', 'regulatory', 'statute', 'precedent', 'case law'
    ];

    const textLower = text.toLowerCase();
    return legalTerms.filter(term => textLower.includes(term));
  }

  /**
   * Extract entities from text (simplified implementation)
   */
  private extractEntities(text: string): Array<{ type: string; value: string; confidence: number }> {
    const entities: Array<{ type: string; value: string; confidence: number }> = [];

    // Simple regex patterns for common legal entities
    const patterns = {
      'DATE': /\b\d{1,2}\/\d{1,2}\/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b/g,
      'MONEY': /\$[\d,]+(?:\.\d{2})?/g,
      'PERSON': /\b[A-Z][a-z]+ [A-Z][a-z]+\b/g,
      'ORGANIZATION': /\b[A-Z][a-zA-Z\s]+ (?:Inc|LLC|Corp|Corporation|Company)\b/g
    };

    Object.entries(patterns).forEach(([type, pattern]) => {
      const matches = text.match(pattern);
      if (matches) {
        matches.forEach(match => {
          entities.push({
            type,
            value: match,
            confidence: 0.8
          });
        });
      }
    });

    return entities.slice(0, 10); // Limit to 10 entities
  }

  /**
   * Analyze legal document using the hybrid approach
   */
  async analyzeLegalDocument(
    title: string,
    content: string,
    analysisType: 'comprehensive' | 'quick' | 'risk-focused' = 'comprehensive'
  ) {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      // Use WebAssembly for document analysis if available
      if (this.config.useWebAssemblyForGeneration && webAssemblyAIAdapter.isSupported()) {
        console.log('[Bridge] Analyzing document with WebAssembly...');
        return await webAssemblyAIAdapter.analyzeLegalDocument(title, content, analysisType);
      } else {
        console.log('[Bridge] WebAssembly not available for document analysis');
        throw new Error('WebAssembly document analysis not available');
      }
    } catch (error: any) {
      console.error('[Bridge] Document analysis failed:', error);
      throw error;
    }
  }

  /**
   * Get bridge health status
   */
  getHealthStatus(): {
    bridgeInitialized: boolean;
    webAssemblyAvailable: boolean;
    langChainHealthy: boolean;
    config: WebAssemblyLangChainConfig;
  } {
    return {
      bridgeInitialized: this.initialized,
      webAssemblyAvailable: webAssemblyAIAdapter.isSupported(),
      langChainHealthy: true, // Would check LangChain RAG health
      config: this.config
    };
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<WebAssemblyLangChainConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log('[Bridge] Configuration updated:', this.config);
  }
}

// Export singleton instance
export const webAssemblyLangChainBridge = new WebAssemblyLangChainBridge();