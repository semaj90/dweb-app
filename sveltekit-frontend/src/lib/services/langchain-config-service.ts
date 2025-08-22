import { ollamaCudaService, type ModelMetrics } from "./ollama-cuda-service.js";

/**
 * LangChain Configuration Service
 * Advanced configuration and orchestration for LangChain with local LLMs
 * Supports multiple model providers, chains, and advanced workflows
 */

import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { ConversationChain } from "langchain/chains";
import { BufferMemory, ConversationSummaryMemory } from "langchain/memory";
import { PromptTemplate, ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import type { BaseLanguageModel } from "@langchain/core/language_models/base";
import type { BaseMemory } from 'langchain/memory';
// Orphaned content: import type { BasePromptTemplate

// Import our CUDA-optimized Ollama service

export interface LangChainConfig {
  modelProvider: 'ollama' | 'openai' | 'anthropic' | 'local';
  modelName: string;
  temperature: number;
  maxTokens: number;
  streaming: boolean;
  memoryType: 'buffer' | 'summary' | 'vector' | 'conversation_kg';
  memorySize: number;
  enableCaching: boolean;
  enableLogging: boolean;
  customPrompts?: Record<string, string>;
  chainConfigs?: ChainConfig[];
}

export interface ChainConfig {
  name: string;
  type: 'conversation' | 'rag' | 'analysis' | 'summary' | 'qa' | 'custom';
  prompt: string;
  inputVariables: string[];
  outputParsers?: string[];
  memory?: boolean;
  tools?: string[];
}

export interface ConversationContext {
  sessionId: string;
  userId?: string;
  caseId?: string;
  documentIds?: string[];
  metadata?: Record<string, any>;
}

export interface ChainExecutionResult {
  result: string;
  metadata: {
    executionTime: number;
    tokensUsed: number;
    modelUsed: string;
    confidence?: number;
    sources?: Array<{
      id: string;
      title: string;
      relevance: number;
    }>;
  };
  memory?: {
    summary: string;
    keyPoints: string[];
  };
}

class LangChainConfigService {
  private static instance: LangChainConfigService;
  private config: LangChainConfig;
  private models: Map<string, BaseLanguageModel> = new Map();
  private embeddings: Map<string, OllamaEmbeddings> = new Map();
  private memories: Map<string, BaseMemory> = new Map();
  private chains: Map<string, any> = new Map();
  private prompts: Map<string, BasePromptTemplate> = new Map();
  private initialized = false;

  private constructor() {
    this.config = this.getDefaultConfig();
    this.initializePrompts();
  }

  public static getInstance(): LangChainConfigService {
    if (!LangChainConfigService.instance) {
      LangChainConfigService.instance = new LangChainConfigService();
    }
    return LangChainConfigService.instance;
  }

  private getDefaultConfig(): LangChainConfig {
    return {
      modelProvider: 'ollama',
      modelName: 'gemma2:9b',
      temperature: 0.7,
      maxTokens: 4096,
      streaming: true,
      memoryType: 'buffer',
      memorySize: 10,
      enableCaching: true,
      enableLogging: true,
      customPrompts: {
        legal_analysis: `You are an expert legal AI assistant specialized in analyzing legal documents and cases. 
        Analyze the following content and provide detailed insights, key points, and recommendations.
        
        Content: {content}
        
        Please provide:
        1. Summary of key legal points
        2. Potential risks or concerns
        3. Recommendations for action
        4. Relevant legal precedents or statutes (if applicable)`,
        
        document_qa: `You are a legal document Q&A assistant. Answer questions about the provided document accurately and cite specific sections.
        
        Document: {document}
        Question: {question}
        
        Answer based solely on the document content and cite relevant sections.`,
        
        case_summary: `Summarize the following legal case information in a structured format:
        
        Case Information: {case_info}
        
        Provide:
        1. Case Overview
        2. Key Facts
        3. Legal Issues
        4. Current Status
        5. Next Steps`,
        
        evidence_analysis: `Analyze the following evidence for legal relevance and admissibility:
        
        Evidence: {evidence}
        Context: {context}
        
        Evaluate:
        1. Legal relevance
        2. Admissibility considerations
        3. Strength of evidence
        4. Potential challenges
        5. Recommendations for use`
      }
    };
  }

  /**
   * Initialize the LangChain service with custom configuration
   */
  public async initialize(customConfig?: Partial<LangChainConfig>): Promise<void> {
    try {
      if (customConfig) {
        this.config = { ...this.config, ...customConfig };
      }

      // Initialize models
      await this.initializeModels();
      
      // Initialize memory systems
      this.initializeMemories();
      
      // Initialize chains
      await this.initializeChains();
      
      this.initialized = true;
      console.log('✅ LangChain Configuration Service initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize LangChain service:', error);
      throw error;
    }
  }

  private async initializeModels(): Promise<void> {
    // Initialize primary Ollama model
    const ollamaModel = new ChatOllama({
      baseUrl: import.meta.env.OLLAMA_BASE_URL || 'http://localhost:11434',
      model: this.config.modelName,
      temperature: this.config.temperature,
      streaming: this.config.streaming,
      // CUDA optimizations
      numCtx: 32768,
      numBatch: 512,
      numGpu: 1,
      f16Kv: true,
      useMmap: true,
      useMlock: true
    } as any);

    this.models.set('primary', ollamaModel);

    // Initialize specialized models for different tasks
    const legalModel = new ChatOllama({
      baseUrl: import.meta.env.OLLAMA_BASE_URL || 'http://localhost:11434',
      model: 'gemma2:9b',
      temperature: 0.3, // More deterministic for legal analysis
      numCtx: 65536, // Large context for legal documents
      numGpu: 1
    } as any);

    this.models.set('legal', legalModel);

    // Initialize embedding model
    const embeddingModel = new OllamaEmbeddings({
      baseUrl: import.meta.env.OLLAMA_BASE_URL || 'http://localhost:11434',
      model: 'nomic-embed-text:latest',
      requestOptions: {
        numGpu: 1,
        mainGpu: 0
      }
    });

    this.embeddings.set('primary', embeddingModel);
    
    console.log('✅ LangChain models initialized');
  }

  private initializeMemories(): void {
    // Buffer memory for conversation history
    const bufferMemory = new BufferMemory({
      memoryKey: 'chat_history',
      returnMessages: true,
      inputKey: 'input',
      outputKey: 'output'
    });

    this.memories.set('buffer', bufferMemory);

    // Summary memory for long conversations
    const summaryMemory = new ConversationSummaryMemory({
      llm: this.models.get('primary')!,
      memoryKey: 'chat_history',
      returnMessages: true
    });

    this.memories.set('summary', summaryMemory);

    console.log('✅ LangChain memories initialized');
  }

  private initializePrompts(): void {
    // Legal Analysis Prompt
    const legalAnalysisPrompt = ChatPromptTemplate.fromMessages([
      new SystemMessage(`You are an expert legal AI assistant with deep knowledge of legal principles, statutes, and case law. 
      You provide thorough, accurate, and professional legal analysis while clearly stating that you cannot provide legal advice.`),
      new MessagesPlaceholder('chat_history'),
      new HumanMessage('{input}')
    ]);

    this.prompts.set('legal_analysis', legalAnalysisPrompt);

    // Document Q&A Prompt
    const documentQAPrompt = ChatPromptTemplate.fromMessages([
      new SystemMessage(`You are a legal document analysis assistant. Answer questions about documents accurately, 
      cite specific sections, and provide context. Only answer based on the provided document content.`),
      new HumanMessage(`Document: {document}\n\nQuestion: {question}`)
    ]);

    this.prompts.set('document_qa', documentQAPrompt);

    // Case Summary Prompt
    const caseSummaryPrompt = PromptTemplate.fromTemplate(`
      Create a comprehensive case summary from the following information:
      
      Case Information: {case_info}
      
      Structure your response as:
      ## Case Overview
      ## Key Facts
      ## Legal Issues
      ## Current Status
      ## Recommended Next Steps
      
      Be thorough but concise, focusing on legally relevant information.
    `);

    this.prompts.set('case_summary', caseSummaryPrompt);
  }

  private async initializeChains(): Promise<void> {
    // Legal Analysis Chain
    const legalAnalysisChain = new ConversationChain({
      llm: this.models.get('legal')!,
      prompt: this.prompts.get('legal_analysis')!,
      memory: this.memories.get(this.config.memoryType)!
    });

    this.chains.set('legal_analysis', legalAnalysisChain);

    // Document Q&A Chain
    const documentQAChain = RunnableSequence.from([
      {
        document: (input: any) => input.document,
        question: (input: any) => input.question
      },
      this.prompts.get('document_qa')!,
      this.models.get('primary')!,
      new StringOutputParser()
    ]);

    this.chains.set('document_qa', documentQAChain);

    // Case Summary Chain
    const caseSummaryChain = RunnableSequence.from([
      {
        case_info: (input: any) => input.case_info
      },
      this.prompts.get('case_summary')!,
      this.models.get('legal')!,
      new StringOutputParser()
    ]);

    this.chains.set('case_summary', caseSummaryChain);

    console.log('✅ LangChain chains initialized');
  }

  /**
   * Execute a chain with specified input and context
   */
  public async executeChain(
    chainName: string,
    input: Record<string, any>,
    context?: ConversationContext
  ): Promise<ChainExecutionResult> {
    const startTime = Date.now();
    
    try {
      if (!this.initialized) {
        await this.initialize();
      }

      const chain = this.chains.get(chainName);
      if (!chain) {
        throw new Error(`Chain '${chainName}' not found`);
      }

      // Add context to input if provided
      const chainInput = context ? { ...input, ...context } : input;

      // Execute the chain
      const result = await chain.invoke(chainInput);
      
      const executionTime = Date.now() - startTime;
      const modelUsed = this.config.modelName;

      // Get token usage if available
      const tokensUsed = this.estimateTokens(JSON.stringify(chainInput) + result);

      return {
        result: typeof result === 'string' ? result : result.text || result.output || JSON.stringify(result),
        metadata: {
          executionTime,
          tokensUsed,
          modelUsed,
          confidence: this.calculateConfidence(result)
        }
      };
    } catch (error) {
      console.error(`Failed to execute chain '${chainName}':`, error);
      throw error;
    }
  }

  /**
   * Legal document analysis with RAG integration
   */
  public async analyzeLegalDocument(
    documentContent: string,
    question?: string,
    context?: ConversationContext
  ): Promise<ChainExecutionResult> {
    const input = {
      content: documentContent,
      ...(question && { question })
    };

    const chainName = question ? 'document_qa' : 'legal_analysis';
    return await this.executeChain(chainName, input, context);
  }

  /**
   * Case summary generation
   */
  public async generateCaseSummary(
    caseInfo: Record<string, any>,
    context?: ConversationContext
  ): Promise<ChainExecutionResult> {
    return await this.executeChain('case_summary', { case_info: JSON.stringify(caseInfo) }, context);
  }

  /**
   * Multi-turn conversation with memory
   */
  public async continueConversation(
    message: string,
    sessionId: string,
    context?: ConversationContext
  ): Promise<ChainExecutionResult> {
    const fullContext = {
      sessionId,
      ...context
    };

    return await this.executeChain('legal_analysis', { input: message }, fullContext);
  }

  /**
   * Generate embeddings for semantic search
   */
  public async generateEmbeddings(texts: string[]): Promise<number[][]> {
    try {
      const embeddingModel = this.embeddings.get('primary');
      if (!embeddingModel) {
        throw new Error('Embedding model not initialized');
      }

      return await embeddingModel.embedDocuments(texts);
    } catch (error) {
      console.error('Failed to generate embeddings:', error);
      throw error;
    }
  }

  /**
   * Create custom chain with specific configuration
   */
  public async createCustomChain(config: ChainConfig): Promise<void> {
    try {
      const prompt = PromptTemplate.fromTemplate(config.prompt);
      const model = this.models.get('primary')!;
      
      let chain;
      
      if (config.memory) {
        const memory = this.memories.get(this.config.memoryType)!;
        chain = new ConversationChain({
          llm: model,
          prompt,
          memory
        });
      } else {
        chain = RunnableSequence.from([
          RunnablePassthrough.assign({}),
          prompt,
          model,
          new StringOutputParser()
        ]);
      }

      this.chains.set(config.name, chain);
      console.log(`✅ Custom chain '${config.name}' created`);
    } catch (error) {
      console.error(`Failed to create custom chain '${config.name}':`, error);
      throw error;
    }
  }

  /**
   * Get system performance metrics
   */
  public async getPerformanceMetrics(): Promise<{
    models: Record<string, ModelMetrics>;
    chains: string[];
    memoryUsage: {
      totalSessions: number;
      activeChains: number;
    };
    health: {
      status: 'healthy' | 'degraded' | 'unhealthy';
      lastCheck: string;
    };
  }> {
    try {
      const ollamaHealth = await ollamaCudaService.getSystemHealth();
      
      return {
        models: ollamaHealth.metrics,
        chains: Array.from(this.chains.keys()),
        memoryUsage: {
          totalSessions: this.memories.size,
          activeChains: this.chains.size
        },
        health: {
          status: ollamaHealth.status,
          lastCheck: new Date().toISOString()
        }
      };
    } catch (error) {
      console.error('Failed to get performance metrics:', error);
      return {
        models: {},
        chains: [],
        memoryUsage: { totalSessions: 0, activeChains: 0 },
        health: { status: 'unhealthy', lastCheck: new Date().toISOString() }
      };
    }
  }

  /**
   * Optimize chains for specific use cases
   */
  public async optimizeForUseCase(useCase: 'legal-analysis' | 'document-qa' | 'conversation'): Promise<void> {
    const optimizations = {
      'legal-analysis': {
        temperature: 0.3,
        maxTokens: 8192,
        memoryType: 'summary' as const
      },
      'document-qa': {
        temperature: 0.1,
        maxTokens: 4096,
        memoryType: 'buffer' as const
      },
      'conversation': {
        temperature: 0.7,
        maxTokens: 2048,
        memoryType: 'buffer' as const
      }
    };

    const optimization = optimizations[useCase];
    if (optimization) {
      this.config = { ...this.config, ...optimization };
      
      // Reinitialize with new configuration
      await this.initialize();
      
      console.log(`✅ LangChain optimized for ${useCase}`);
    }
  }

  /**
   * Clear memory for a specific session
   */
  public async clearMemory(sessionId?: string): Promise<void> {
    if (sessionId) {
      // Clear specific session memory
      const memory = this.memories.get(sessionId);
      if (memory && 'clear' in memory) {
        await (memory as any).clear();
      }
    } else {
      // Clear all memories
      for (const memory of this.memories.values()) {
        if ('clear' in memory) {
          await (memory as any).clear();
        }
      }
    }
    
    console.log('✅ Memory cleared');
  }

  // Utility methods
  private estimateTokens(text: string): number {
    // Rough estimation: ~4 characters per token
    return Math.ceil(text.length / 4);
  }

  private calculateConfidence(result: any): number {
    // Simple confidence calculation based on result length and structure
    if (typeof result === 'string') {
      return result.length > 100 ? 0.8 : 0.6;
    }
    return 0.7;
  }

  // Getters
  public get isInitialized(): boolean {
    return this.initialized;
  }

  public get currentConfig(): LangChainConfig {
    return { ...this.config };
  }

  public getAvailableChains(): string[] {
    return Array.from(this.chains.keys());
  }

  public getAvailableModels(): string[] {
    return Array.from(this.models.keys());
  }
}

// Export singleton instance
export const langchainConfigService = LangChainConfigService.getInstance();
export default langchainConfigService;

// Types are already exported at their definitions above