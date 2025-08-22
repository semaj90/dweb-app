/**
 * LangChain.js Integration with Ollama and llama.cpp GPU Parsing
 * Complete AI pipeline with Go microservice integration
 */

import { Ollama } from '@langchain/ollama';
import { ChatOllama } from '@langchain/ollama';
import { OllamaEmbeddings } from '@langchain/ollama';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RetrievalQAChain } from 'langchain/chains';
import { PromptTemplate } from '@langchain/core/prompts';

export interface LangChainConfig {
  ollamaBaseUrl: string;
  models: {
    chat: 'gemma3-legal';
    embedding: 'nomic-embed-text';
  };
  gpu: {
    enabled: boolean;
    device: 'RTX3060Ti';
    llamaCppConfig: {
      ngl: number; // GPU layers
      contextSize: number;
      batchSize: number;
    };
  };
  goMicroservice: {
    enhancedRAGUrl: string;
    uploadServiceUrl: string;
    quicProxyUrl: string;
  };
}

export interface ProcessingResult {
  text: string;
  embedding: number[];
  summary: string;
  entities: string[];
  confidence: number;
  processingTime: number;
  gpuUtilization?: number;
}

/**
 * Complete LangChain integration with Ollama, llama.cpp, and Go microservices
 */
export class LangChainOllamaIntegration {
  private config: LangChainConfig;
  private chatModel: ChatOllama | null = null;
  private embeddingModel: OllamaEmbeddings | null = null;
  private vectorStore: MemoryVectorStore | null = null;
  private qaChain: RetrievalQAChain | null = null;
  private textSplitter: RecursiveCharacterTextSplitter;
  private isInitialized = false;

  constructor(config: Partial<LangChainConfig> = {}) {
    this.config = {
      ollamaBaseUrl: 'http://localhost:11434',
      models: {
        chat: 'gemma3-legal',
        embedding: 'nomic-embed-text'
      },
      gpu: {
        enabled: true,
        device: 'RTX3060Ti',
        llamaCppConfig: {
          ngl: 35, // Use 35 GPU layers for RTX 3060 Ti
          contextSize: 4096,
          batchSize: 8
        }
      },
      goMicroservice: {
        enhancedRAGUrl: 'http://localhost:8094',
        uploadServiceUrl: 'http://localhost:8093',
        quicProxyUrl: 'http://localhost:8095'
      },
      ...config
    };

    // Initialize text splitter for document processing
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
      separators: ['\n\n', '\n', '.', '!', '?', ';', ',', ' ', '']
    });
  }

  /**
   * Initialize all LangChain components
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('🔗 Initializing LangChain.js with Ollama and GPU acceleration...');

    try {
      // Initialize chat model (gemma3-legal only)
      this.chatModel = new ChatOllama({
        baseUrl: this.config.ollamaBaseUrl,
        model: this.config.models.chat,
        temperature: 0.1, // Low temperature for legal accuracy
        topP: 0.9,
        numCtx: this.config.gpu.llamaCppConfig.contextSize,
        numGpu: this.config.gpu.llamaCppConfig.ngl // GPU layers
      });

      // Initialize embedding model (nomic-embed-text only)
      this.embeddingModel = new OllamaEmbeddings({
        baseUrl: this.config.ollamaBaseUrl,
        model: this.config.models.embedding,
        requestOptions: {
          numGpu: this.config.gpu.llamaCppConfig.ngl
        }
      });

      // Initialize vector store
      this.vectorStore = new MemoryVectorStore(this.embeddingModel);

      // Test GPU connectivity with Go microservices
      await this.testGoMicroserviceIntegration();

      this.isInitialized = true;
      console.log('✅ LangChain integration initialized with GPU acceleration');

    } catch (error) {
      console.error('❌ LangChain initialization failed:', error);
      throw error;
    }
  }

  /**
   * Test integration with Go microservices
   */
  private async testGoMicroserviceIntegration(): Promise<void> {
    const services = [
      { name: 'Enhanced RAG', url: this.config.goMicroservice.enhancedRAGUrl },
      { name: 'Upload Service', url: this.config.goMicroservice.uploadServiceUrl },
      { name: 'QUIC Proxy', url: this.config.goMicroservice.quicProxyUrl }
    ];

    for (const service of services) {
      try {
        const response = await fetch(`${service.url}/health`, { 
          signal: AbortSignal.timeout(3000) 
        });
        if (response.ok) {
          console.log(`  ✅ ${service.name}: Connected`);
        } else {
          console.log(`  ⚠️ ${service.name}: HTTP ${response.status}`);
        }
      } catch (error) {
        console.log(`  🔴 ${service.name}: Not available`);
      }
    }
  }

  /**
   * Process legal document with GPU acceleration
   */
  async processLegalDocument(
    content: string,
    title: string,
    options: {
      generateSummary?: boolean;
      extractEntities?: boolean;
      useGPUParsing?: boolean;
      storeInVector?: boolean;
    } = {}
  ): Promise<ProcessingResult> {
    await this.initialize();

    const startTime = performance.now();
    console.log(`📄 Processing legal document: ${title}`);

    try {
      // Step 1: Split document into chunks
      const chunks = await this.textSplitter.splitText(content);
      console.log(`📑 Document split into ${chunks.length} chunks`);

      // Step 2: Generate embeddings for all chunks
      const embeddings: number[][] = [];
      for (const chunk of chunks) {
        const embedding = await this.embeddingModel!.embedQuery(chunk);
        embeddings.push(embedding);
      }

      // Step 3: Use Go microservice for GPU parsing if enabled
      let gpuParsingResult = null;
      if (options.useGPUParsing) {
        gpuParsingResult = await this.callGoMicroserviceGPUParsing(content, title);
      }

      // Step 4: Generate summary using gemma3-legal
      let summary = '';
      if (options.generateSummary) {
        summary = await this.generateLegalSummary(content);
      }

      // Step 5: Extract legal entities
      let entities: string[] = [];
      if (options.extractEntities) {
        entities = await this.extractLegalEntities(content);
      }

      // Step 6: Store in vector database if requested
      if (options.storeInVector && this.vectorStore) {
        await this.vectorStore.addDocuments(
          chunks.map((chunk, index) => ({
            pageContent: chunk,
            metadata: {
              title,
              chunkIndex: index,
              totalChunks: chunks.length,
              documentId: crypto.randomUUID()
            }
          }))
        );
        console.log(`📚 Stored ${chunks.length} chunks in vector database`);
      }

      const processingTime = performance.now() - startTime;

      const result: ProcessingResult = {
        text: content,
        embedding: embeddings[0] || [], // First chunk embedding as document embedding
        summary,
        entities,
        confidence: gpuParsingResult?.confidence || 0.85,
        processingTime,
        gpuUtilization: gpuParsingResult?.gpuUtilization
      };

      console.log(`✅ Document processing complete (${processingTime.toFixed(2)}ms)`);
      return result;

    } catch (error) {
      console.error('❌ Document processing failed:', error);
      throw error;
    }
  }

  /**
   * Call Go microservice for GPU-accelerated parsing
   */
  private async callGoMicroserviceGPUParsing(
    content: string, 
    title: string
  ): Promise<{ confidence: number; gpuUtilization: number; entities: string[] }> {
    try {
      const response = await fetch(`${this.config.goMicroservice.enhancedRAGUrl}/api/gpu-parse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content,
          title,
          model: this.config.models.chat,
          gpu_config: this.config.gpu.llamaCppConfig
        })
      });

      if (!response.ok) {
        throw new Error(`GPU parsing service error: ${response.status}`);
      }

      const result = await response.json();
      console.log('⚡ GPU parsing complete via Go microservice');
      
      return {
        confidence: result.confidence || 0.8,
        gpuUtilization: result.gpu_utilization || 0,
        entities: result.entities || []
      };

    } catch (error) {
      console.warn('⚠️ GPU parsing via Go microservice failed, using local fallback:', error);
      return {
        confidence: 0.7,
        gpuUtilization: 0,
        entities: []
      };
    }
  }

  /**
   * Generate legal summary using gemma3-legal model
   */
  private async generateLegalSummary(content: string): Promise<string> {
    if (!this.chatModel) {
      throw new Error('Chat model not initialized');
    }

    const summaryPrompt = PromptTemplate.fromTemplate(`
      As a legal AI assistant, provide a concise summary of the following legal document.
      Focus on key legal concepts, obligations, rights, and potential issues.
      
      Document:
      {content}
      
      Legal Summary:
    `);

    try {
      const chain = summaryPrompt.pipe(this.chatModel);
      const result = await chain.invoke({ content: content.slice(0, 4000) }); // Limit for context
      
      return typeof result === 'string' ? result : result.content || 'Summary generation failed';
    } catch (error) {
      console.error('❌ Summary generation failed:', error);
      return 'Summary could not be generated due to processing error.';
    }
  }

  /**
   * Extract legal entities using NLP patterns
   */
  private async extractLegalEntities(content: string): Promise<string[]> {
    const entities: string[] = [];

    // Extract case names (Pattern: Party v. Party)
    const caseNames = content.match(/[A-Z][a-zA-Z\s]+ v\. [A-Z][a-zA-Z\s]+/g) || [];
    entities.push(...caseNames);

    // Extract citations (Pattern: Volume Source Page)
    const citations = content.match(/\d+\s+[A-Z][a-z]+\.?\s*\d+/g) || [];
    entities.push(...citations);

    // Extract people/organizations
    const people = content.match(/(?:Mr\.|Ms\.|Dr\.|Judge|Justice|Attorney)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g) || [];
    entities.push(...people);

    // Extract dates
    const dates = content.match(/\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b/g) || [];
    entities.push(...dates);

    return [...new Set(entities)].slice(0, 20); // Deduplicate and limit
  }

  /**
   * Semantic search using vector similarity
   */
  async semanticSearch(
    query: string, 
    maxResults = 10, 
    threshold = 0.7
  ): Promise<SearchResult[]> {
    await this.initialize();

    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }

    try {
      console.log(`🔍 Semantic search: "${query}"`);

      // Generate query embedding
      const queryEmbedding = await this.embeddingModel!.embedQuery(query);

      // Search vector store
      const results = await this.vectorStore.similaritySearch(query, maxResults);

      // Convert to SearchResult format
      const searchResults = results.map((doc, index) => ({
        item: {
          id: doc.metadata.documentId || `doc-${index}`,
          title: doc.metadata.title || 'Untitled',
          content: doc.pageContent,
          keywords: [],
          metadata: doc.metadata
        },
        score: 1 - (doc.metadata.score || 0.8), // Convert similarity to score
        similarity: doc.metadata.score || 0.8,
        refIndex: index
      }));

      console.log(`📊 Semantic search complete: ${searchResults.length} results`);
      return searchResults;

    } catch (error) {
      console.error('❌ Semantic search failed:', error);
      return [];
    }
  }

  /**
   * RAG query with Go microservice integration
   */
  async ragQuery(
    question: string,
    context: string[] = [],
    useGPUAcceleration = true
  ): Promise<{ answer: string; sources: any[]; confidence: number }> {
    await this.initialize();

    console.log(`🤖 RAG query: "${question}"`);

    try {
      // Option 1: Use Go microservice enhanced RAG (preferred)
      if (useGPUAcceleration) {
        const goResult = await this.callGoMicroserviceRAG(question, context);
        if (goResult.success) {
          return goResult.data;
        }
        console.warn('⚠️ Go microservice RAG failed, using local LangChain');
      }

      // Option 2: Local LangChain RAG fallback
      return await this.localLangChainRAG(question, context);

    } catch (error) {
      console.error('❌ RAG query failed:', error);
      throw error;
    }
  }

  /**
   * Call Go microservice for enhanced RAG with GPU acceleration
   */
  private async callGoMicroserviceRAG(
    question: string, 
    context: string[]
  ): Promise<{ success: boolean; data: any }> {
    try {
      const response = await fetch(`${this.config.goMicroservice.enhancedRAGUrl}/api/rag`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: question,
          context,
          model: this.config.models.chat,
          embedding_model: this.config.models.embedding,
          gpu_config: this.config.gpu.llamaCppConfig,
          use_gpu: this.config.gpu.enabled
        })
      });

      if (!response.ok) {
        throw new Error(`Enhanced RAG service error: ${response.status}`);
      }

      const result = await response.json();
      console.log('⚡ Go microservice RAG complete');

      return {
        success: true,
        data: {
          answer: result.response || result.answer,
          sources: result.sources || [],
          confidence: result.confidence || 0.8
        }
      };

    } catch (error) {
      console.warn('⚠️ Go microservice RAG failed:', error);
      return { success: false, data: null };
    }
  }

  /**
   * Local LangChain RAG processing
   */
  private async localLangChainRAG(
    question: string, 
    context: string[]
  ): Promise<{ answer: string; sources: any[]; confidence: number }> {
    if (!this.chatModel || !this.vectorStore) {
      throw new Error('LangChain models not initialized');
    }

    try {
      // Add context to vector store if provided
      if (context.length > 0) {
        const contextDocs = context.map((text, index) => ({
          pageContent: text,
          metadata: { source: `context-${index}`, type: 'context' }
        }));
        await this.vectorStore.addDocuments(contextDocs);
      }

      // Create QA chain if not exists
      if (!this.qaChain) {
        this.qaChain = RetrievalQAChain.fromLLM(
          this.chatModel,
          this.vectorStore.asRetriever({
            k: 5, // Retrieve top 5 relevant documents
            searchType: 'similarity',
            searchKwargs: { threshold: 0.7 }
          })
        );
      }

      // Execute RAG query
      const result = await this.qaChain.call({ query: question });

      return {
        answer: result.text || 'No answer generated',
        sources: result.sourceDocuments || [],
        confidence: 0.75 // Local processing confidence
      };

    } catch (error) {
      console.error('❌ Local LangChain RAG failed:', error);
      throw error;
    }
  }

  /**
   * Generate embeddings with nomic-embed-text
   */
  async generateEmbedding(text: string): Promise<number[]> {
    await this.initialize();

    if (!this.embeddingModel) {
      throw new Error('Embedding model not initialized');
    }

    try {
      console.log(`🔢 Generating embedding for text (${text.length} chars)`);
      const embedding = await this.embeddingModel.embedQuery(text);
      console.log(`✅ Embedding generated (${embedding.length} dimensions)`);
      return embedding;
    } catch (error) {
      console.error('❌ Embedding generation failed:', error);
      throw error;
    }
  }

  /**
   * Batch process multiple documents
   */
  async batchProcessDocuments(
    documents: Array<{ title: string; content: string; metadata?: unknown }>,
    options: {
      useGPUAcceleration?: boolean;
      generateSummaries?: boolean;
      extractEntities?: boolean;
      chunkSize?: number;
    } = {}
  ): Promise<ProcessingResult[]> {
    await this.initialize();

    console.log(`📚 Batch processing ${documents.length} documents...`);

    const results: ProcessingResult[] = [];
    const batchSize = options.chunkSize || 5;

    // Process in batches to avoid overwhelming the GPU
    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);
      console.log(`⚡ Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(documents.length / batchSize)}`);

      const batchPromises = batch.map(doc => 
        this.processLegalDocument(doc.content, doc.title, {
          generateSummary: options.generateSummaries,
          extractEntities: options.extractEntities,
          useGPUParsing: options.useGPUAcceleration,
          storeInVector: true
        })
      );

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);

      // Small delay between batches to prevent GPU overload
      if (i + batchSize < documents.length) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    console.log(`✅ Batch processing complete: ${results.length} documents processed`);
    return results;
  }

  /**
   * Setup retrieval QA chain with custom prompt
   */
  async setupCustomRAGChain(customPrompt?: string): Promise<void> {
    await this.initialize();

    if (!this.chatModel || !this.vectorStore) {
      throw new Error('Models not initialized');
    }

    const legalPrompt = customPrompt || `
      You are a legal AI assistant specializing in prosecutor and legal analysis.
      Use the provided context to answer questions accurately and professionally.
      
      Context: {context}
      
      Question: {question}
      
      Instructions:
      - Provide accurate legal analysis based on the context
      - Cite relevant sources when possible
      - Highlight any potential legal issues or precedents
      - Use professional legal terminology
      - If uncertain, state limitations clearly
      
      Answer:
    `;

    const prompt = PromptTemplate.fromTemplate(legalPrompt);

    this.qaChain = RetrievalQAChain.fromLLM(
      this.chatModel,
      this.vectorStore.asRetriever({
        k: 8, // Retrieve more documents for legal context
        searchType: 'similarity_score_threshold',
        searchKwargs: { 
          scoreThreshold: 0.6,
          k: 8
        }
      }),
      {
        prompt,
        returnSourceDocuments: true
      }
    );

    console.log('⚖️ Custom legal RAG chain configured');
  }

  /**
   * Get service status and performance metrics
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      models: {
        chat: this.chatModel ? 'ready' : 'not_initialized',
        embedding: this.embeddingModel ? 'ready' : 'not_initialized'
      },
      vectorStore: {
        ready: !!this.vectorStore,
        documentCount: this.vectorStore ? 'unknown' : 0
      },
      gpu: {
        enabled: this.config.gpu.enabled,
        device: this.config.gpu.device,
        layers: this.config.gpu.llamaCppConfig.ngl
      },
      goServices: {
        enhancedRAG: this.config.goMicroservice.enhancedRAGUrl,
        uploadService: this.config.goMicroservice.uploadServiceUrl,
        quicProxy: this.config.goMicroservice.quicProxyUrl
      }
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    this.chatModel = null;
    this.embeddingModel = null;
    this.vectorStore = null;
    this.qaChain = null;
    this.isInitialized = false;
    console.log('🧹 LangChain integration cleaned up');
  }
}

// Global service instance
export const langChainOllamaService = new LangChainOllamaIntegration({
  ollamaBaseUrl: 'http://localhost:11434',
  models: {
    chat: 'gemma3-legal',
    embedding: 'nomic-embed-text'
  },
  gpu: {
    enabled: true,
    device: 'RTX3060Ti',
    llamaCppConfig: {
      ngl: 35, // RTX 3060 Ti optimized
      contextSize: 4096,
      batchSize: 8
    }
  },
  goMicroservice: {
    enhancedRAGUrl: 'http://localhost:8094',
    uploadServiceUrl: 'http://localhost:8093',
    quicProxyUrl: 'http://localhost:8095'
  }
});

// Auto-initialize on import (browser only)
if (typeof window !== 'undefined') {
  langChainOllamaService.initialize().catch(console.warn);
}

// Export types and utilities
export type { LangChainConfig, ProcessingResult, SearchResult };
export { SearchableItem, SearchOptions } from './fuse-lazy-search-indexeddb';