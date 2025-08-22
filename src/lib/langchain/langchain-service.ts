// LangChain Integration Service
// High-level service that orchestrates LangChain operations with event-driven architecture

import { EventEmitter } from 'events';
import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

import LangChainManager, { type LangChainConfig, type ChainExecution } from './langchain-manager';
import { legalTools } from './tools/legal-tools';
import { xstateManager } from '../stores/xstate-store-manager';

export interface LangChainServiceConfig {
  llmProvider: 'ollama' | 'openai' | 'local';
  model: string;
  enableTools: boolean;
  enableMemory: boolean;
  enableStreaming: boolean;
  enableEventLogging: boolean;
  customTools?: unknown[];
}

export interface ConversationSession {
  id: string;
  title: string;
  createdAt: number;
  lastActivity: number;
  messageCount: number;
  context: {
    caseId?: string;
    documentIds?: string[];
    legalContext?: string;
    userRole?: string;
  };
  metadata: {
    totalTokens: number;
    totalCost: number;
    toolsUsed: string[];
  };
}

export interface StreamingResponse {
  sessionId: string;
  executionId: string;
  chunk: string;
  isComplete: boolean;
  metadata?: unknown;
}

/**
 * LangChain Integration Service
 * Provides high-level interface for AI operations with legal specialization
 */
export class LangChainService extends EventEmitter {
  private manager: LangChainManager | null = null;
  private sessions = new Map<string, ConversationSession>();
  private activeStreams = new Map<string, AbortController>();
  private isInitialized = false;
  private config: LangChainServiceConfig;

  constructor(config: LangChainServiceConfig) {
    super();
    this.config = config;
    this.setupEventForwarding();
  }

  async initialize(): Promise<boolean> {
    try {
      console.log('🔄 Initializing LangChain Service...');

      // Build LangChain configuration
      const langchainConfig: LangChainConfig = {
        llm: {
          provider: this.config.llmProvider,
          model: this.config.model,
          temperature: 0.7,
          maxTokens: 2000,
          baseUrl: this.config.llmProvider === 'ollama' ? 'http://localhost:11434' : undefined
        },
        memory: {
          type: this.config.enableMemory ? 'conversation' : 'buffer',
          maxTokens: 4000,
          returnMessages: 10
        },
        tools: this.config.enableTools ? legalTools : [],
        events: {
          enableLogging: this.config.enableEventLogging,
          enableMetrics: true,
          enableStreaming: this.config.enableStreaming
        }
      };

      // Add custom tools if provided
      if (this.config.customTools) {
        langchainConfig.tools.push(...this.config.customTools);
      }

      // Initialize LangChain manager
      this.manager = new LangChainManager(langchainConfig);
      const success = await this.manager.initialize();

      if (!success) {
        throw new Error('LangChain manager initialization failed');
      }

      // Setup event listeners
      this.setupManagerEventListeners();

      this.isInitialized = true;
      this.emit('service:initialized', { config: this.config });

      console.log('✓ LangChain Service initialized successfully');
      langchainServiceStatus.set({ isReady: true, sessions: 0, activeStreams: 0 });

      return true;

    } catch (error) {
      console.error('❌ LangChain Service initialization failed:', error);
      this.emit('service:error', { error: error.message });
      return false;
    }
  }

  // ============ Conversation Management ============

  /**
   * Create a new conversation session
   */
  createSession(
    title?: string,
    context: ConversationSession['context'] = {}
  ): ConversationSession {
    const session: ConversationSession = {
      id: this.generateSessionId(),
      title: title || `Legal AI Session ${new Date().toLocaleString()}`,
      createdAt: Date.now(),
      lastActivity: Date.now(),
      messageCount: 0,
      context,
      metadata: {
        totalTokens: 0,
        totalCost: 0,
        toolsUsed: []
      }
    };

    this.sessions.set(session.id, session);
    this.emit('session:created', { session });

    // Update stores
    this.updateServiceStatus();

    return session;
  }

  /**
   * Get existing session
   */
  getSession(sessionId: string): ConversationSession | undefined {
    return this.sessions.get(sessionId);
  }

  /**
   * List all sessions
   */
  getSessions(): ConversationSession[] {
    return Array.from(this.sessions.values())
      .sort((a, b) => b.lastActivity - a.lastActivity);
  }

  /**
   * Delete a session
   */
  deleteSession(sessionId: string): boolean {
    const deleted = this.sessions.delete(sessionId);
    if (deleted) {
      this.emit('session:deleted', { sessionId });
      this.updateServiceStatus();
    }
    return deleted;
  }

  // ============ AI Interaction Methods ============

  /**
   * Send a message and get a response
   */
  async sendMessage(
    sessionId: string,
    message: string,
    options: {
      chainType?: 'simple' | 'conversation' | 'tool' | 'rag';
      enableTools?: boolean;
      context?: unknown;
    } = {}
  ): Promise<{
    response: string;
    execution: ChainExecution;
    session: ConversationSession;
  }> {
    if (!this.isInitialized || !this.manager) {
      throw new Error('LangChain Service not initialized');
    }

    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    this.emit('message:sending', { sessionId, message });

    try {
      // Execute chain with context
      const execution = await this.manager.executeChain(message, {
        chainType: options.chainType || 'conversation',
        context: { ...session.context, ...options.context },
        memory: this.config.enableMemory,
        tools: options.enableTools !== false && this.config.enableTools ? undefined : []
      });

      // Update session
      session.messageCount++;
      session.lastActivity = Date.now();
      session.metadata.totalTokens += execution.metadata.tokens;
      session.metadata.totalCost += execution.metadata.cost;
      
      // Track tools used
      execution.metadata.tools.forEach(tool => {
        if (!session.metadata.toolsUsed.includes(tool)) {
          session.metadata.toolsUsed.push(tool);
        }
      });

      this.emit('message:received', { 
        sessionId, 
        message, 
        response: execution.output,
        execution 
      });

      // Integrate with XState if available
      if (typeof window !== 'undefined' && xstateManager) {
        xstateManager.sendToMachine('chat', {
          type: 'MESSAGE_DELIVERED',
          messageId: execution.id,
          response: execution.output
        });
      }

      this.updateServiceStatus();

      return {
        response: execution.output,
        execution,
        session
      };

    } catch (error) {
      this.emit('message:error', { sessionId, message, error: error.message });
      throw error;
    }
  }

  /**
   * Send a message with streaming response
   */
  async *sendStreamingMessage(
    sessionId: string,
    message: string,
    options: {
      chainType?: 'simple' | 'conversation' | 'tool' | 'rag';
      enableTools?: boolean;
      context?: unknown;
    } = {}
  ): AsyncGenerator<StreamingResponse, void, unknown> {
    if (!this.isInitialized || !this.manager) {
      throw new Error('LangChain Service not initialized');
    }

    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    // Create abort controller for stream cancellation
    const abortController = new AbortController();
    this.activeStreams.set(sessionId, abortController);

    this.emit('streaming:started', { sessionId, message });

    try {
      const stream = this.manager.executeStreamingChain(message, {
        chainType: options.chainType || 'conversation',
        context: { ...session.context, ...options.context },
        memory: this.config.enableMemory,
        tools: options.enableTools !== false && this.config.enableTools ? undefined : []
      });

      let isComplete = false;

      for await (const { chunk, execution } of stream) {
        // Check if stream was aborted
        if (abortController.signal.aborted) {
          break;
        }

        const response: StreamingResponse = {
          sessionId,
          executionId: execution.id,
          chunk,
          isComplete: false,
          metadata: execution.metadata
        };

        this.emit('streaming:chunk', response);
        yield response;
      }

      // Send completion response
      isComplete = true;
      const finalResponse: StreamingResponse = {
        sessionId,
        executionId: '',
        chunk: '',
        isComplete: true
      };

      // Update session
      session.messageCount++;
      session.lastActivity = Date.now();

      this.emit('streaming:completed', { sessionId, message });
      yield finalResponse;

    } catch (error) {
      this.emit('streaming:error', { sessionId, message, error: error.message });
      throw error;
    } finally {
      this.activeStreams.delete(sessionId);
      this.updateServiceStatus();
    }
  }

  /**
   * Cancel a streaming response
   */
  cancelStream(sessionId: string): boolean {
    const controller = this.activeStreams.get(sessionId);
    if (controller) {
      controller.abort();
      this.activeStreams.delete(sessionId);
      this.emit('streaming:cancelled', { sessionId });
      this.updateServiceStatus();
      return true;
    }
    return false;
  }

  // ============ Tool Operations ============

  /**
   * Execute a specific tool
   */
  async executeTool(
    toolName: string,
    input: string,
    options: unknown = {}
  ): Promise<{
    toolName: string;
    input: string;
    output: string;
    executionTime: number;
  }> {
    if (!this.isInitialized || !this.manager) {
      throw new Error('LangChain Service not initialized');
    }

    const startTime = Date.now();
    this.emit('tool:executing', { toolName, input });

    try {
      const execution = await this.manager.executeChain(input, {
        chainType: 'tool',
        tools: [toolName],
        context: options.context
      });

      const executionTime = Date.now() - startTime;

      const result = {
        toolName,
        input,
        output: execution.output,
        executionTime
      };

      this.emit('tool:executed', result);
      return result;

    } catch (error) {
      this.emit('tool:error', { toolName, input, error: error.message });
      throw error;
    }
  }

  /**
   * Get available tools
   */
  getAvailableTools(): Array<{ name: string; description: string }> {
    if (!this.manager) return [];
    
    return this.manager.getTools().map(tool => ({
      name: tool.name,
      description: tool.description
    }));
  }

  // ============ Advanced Features ============

  /**
   * Analyze legal document using AI
   */
  async analyzeLegalDocument(
    documentContent: string,
    analysisType: 'summary' | 'precedents' | 'facts' | 'holding' | 'reasoning' = 'summary',
    context: unknown = {}
  ): Promise<any> {
    const toolInput = JSON.stringify({
      caseText: documentContent,
      analysisType,
      jurisdiction: context.jurisdiction
    });

    const result = await this.executeTool('case_analysis', toolInput, { context });
    return JSON.parse(result.output);
  }

  /**
   * Search legal database
   */
  async searchLegalDatabase(
    query: string,
    filters: unknown = {},
    options: unknown = {}
  ): Promise<any> {
    const toolInput = JSON.stringify({
      query,
      filters,
      options: { topK: 10, threshold: 0.7, useGPU: true, ...options }
    });

    const result = await this.executeTool('legal_search', toolInput);
    return JSON.parse(result.output);
  }

  /**
   * Generate legal document
   */
  async generateLegalDocument(
    documentType: string,
    parties: unknown,
    terms: unknown,
    options: unknown = {}
  ): Promise<any> {
    const toolInput = JSON.stringify({
      documentType,
      parties,
      terms,
      jurisdiction: options.jurisdiction,
      purpose: options.purpose
    });

    const result = await this.executeTool('legal_drafting', toolInput, { context: options });
    return JSON.parse(result.output);
  }

  // ============ Memory and State Management ============

  /**
   * Save session memory
   */
  saveSessionMemory(sessionId: string): string | null {
    if (!this.manager) return null;
    
    const memory = this.manager.saveMemory();
    this.emit('memory:saved', { sessionId, size: memory.length });
    return memory;
  }

  /**
   * Load session memory
   */
  loadSessionMemory(sessionId: string, memoryData: string): void {
    if (!this.manager) return;
    
    this.manager.loadMemory(memoryData);
    this.emit('memory:loaded', { sessionId, size: memoryData.length });
  }

  /**
   * Clear session memory
   */
  clearSessionMemory(sessionId: string): void {
    if (!this.manager) return;
    
    this.manager.clearMemory();
    this.emit('memory:cleared', { sessionId });
  }

  // ============ Metrics and Monitoring ============

  /**
   * Get service metrics
   */
  getMetrics(): unknown {
    if (!this.manager) return null;
    
    const managerMetrics = this.manager.getMetrics();
    const serviceMetrics = {
      sessions: {
        total: this.sessions.size,
        active: Array.from(this.sessions.values()).filter(s => 
          Date.now() - s.lastActivity < 3600000 // Active in last hour
        ).length
      },
      streaming: {
        activeStreams: this.activeStreams.size
      }
    };

    return { ...managerMetrics, service: serviceMetrics };
  }

  /**
   * Get execution history
   */
  getExecutionHistory(): ChainExecution[] {
    if (!this.manager) return [];
    return this.manager.getExecutions();
  }

  // ============ Private Helper Methods ============

  private setupEventForwarding(): void {
    // Forward important events to consumers
    this.on('session:created', (data) => {
      console.log('📝 Session created:', data.session.id);
    });

    this.on('message:received', (data) => {
      console.log('💬 Message processed:', data.sessionId);
    });

    this.on('streaming:chunk', (data) => {
      // Real-time event for UI updates
    });
  }

  private setupManagerEventListeners(): void {
    if (!this.manager) return;

    // Forward manager events
    this.manager.on('execution:completed', (data) => {
      this.emit('execution:completed', data);
    });

    this.manager.on('execution:failed', (data) => {
      this.emit('execution:failed', data);
    });

    this.manager.on('tool:executed', (data) => {
      this.emit('tool:executed', data);
    });
  }

  private updateServiceStatus(): void {
    langchainServiceStatus.set({
      isReady: this.isInitialized,
      sessions: this.sessions.size,
      activeStreams: this.activeStreams.size
    });
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // ============ Cleanup ============

  async cleanup(): Promise<void> {
    // Cancel all active streams
    for (const controller of this.activeStreams.values()) {
      controller.abort();
    }
    this.activeStreams.clear();

    // Clear sessions
    this.sessions.clear();

    // Cleanup manager
    if (this.manager) {
      this.manager.removeAllListeners();
    }

    this.removeAllListeners();
    this.isInitialized = false;

    console.log('🧹 LangChain Service cleaned up');
  }

  // ============ Getters ============

  get isReady(): boolean {
    return this.isInitialized && this.manager !== null;
  }

  get sessionCount(): number {
    return this.sessions.size;
  }

  get activeStreamCount(): number {
    return this.activeStreams.size;
  }
}

// Svelte stores for reactive access
export const langchainServiceStatus = writable<{
  isReady: boolean;
  sessions: number;
  activeStreams: number;
}>({
  isReady: false,
  sessions: 0,
  activeStreams: 0
});

export const langchainSessions = writable<ConversationSession[]>([]);

export const langchainMetrics = writable<any>({});

// Singleton service instance
let serviceInstance: LangChainService | null = null;

export function createLangChainService(config: LangChainServiceConfig): LangChainService {
  if (serviceInstance) {
    serviceInstance.cleanup();
  }
  
  serviceInstance = new LangChainService(config);
  return serviceInstance;
}

export function getLangChainService(): LangChainService | null {
  return serviceInstance;
}

// Auto-initialize with default config in browser
if (browser) {
  const defaultConfig: LangChainServiceConfig = {
    llmProvider: 'ollama',
    model: 'gemma3:legal-latest',
    enableTools: true,
    enableMemory: true,
    enableStreaming: true,
    enableEventLogging: true
  };

  serviceInstance = createLangChainService(defaultConfig);
  serviceInstance.initialize().catch(console.error);
}

export default LangChainService;