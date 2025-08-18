// NATS + LangChain + RAG Pipeline Integration Service
// Comprehensive integration service connecting all messaging, AI, and search components

import { EventEmitter } from 'events';
import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

// Import core services
import { 
  getNATSService, 
  createNATSService,
  NATS_SUBJECTS,
  type NATSMessage 
} from './nats-messaging-service';
import { getLangChainService } from '../langchain/langchain-service';
import { ragPipeline } from './enhanced-rag-pipeline';
import { multiProtocolRouter } from './multi-protocol-router';

/**
 * Integration Service Configuration
 */
export interface IntegrationConfig {
  // Service enablement
  enableNATSMessaging: boolean;
  enableLangChainIntegration: boolean;
  enableRAGPipeline: boolean;
  enableMultiProtocol: boolean;
  
  // Processing settings
  autoProcessMessages: boolean;
  enableMessagePersistence: boolean;
  enableEventForwarding: boolean;
  enablePerformanceMonitoring: boolean;
  
  // Legal AI specific settings
  enableLegalWorkflows: boolean;
  enableCaseManagement: boolean;
  enableDocumentProcessing: boolean;
  enableRealTimeAnalysis: boolean;
  
  // Performance settings
  maxConcurrentProcessing: number;
  messageRetentionHours: number;
  healthCheckInterval: number;
}

/**
 * Message Processing Result
 */
export interface ProcessingResult {
  messageId: string;
  success: boolean;
  processingTime: number;
  result?: any;
  error?: string;
  serviceUsed: string[];
  timestamp: number;
}

/**
 * Integration Statistics
 */
export interface IntegrationStats {
  messagesProcessed: number;
  successfulProcessing: number;
  failedProcessing: number;
  averageProcessingTime: number;
  serviceHealth: {
    nats: boolean;
    langchain: boolean;
    rag: boolean;
    multiProtocol: boolean;
  };
  lastActivity: number;
  activeWorkflows: number;
}

/**
 * NATS + LangChain + RAG Integration Service
 * Orchestrates communication between all AI services through NATS messaging
 */
export class NATSLangChainIntegration extends EventEmitter {
  private config: IntegrationConfig;
  private natsService: any = null;
  private langchainService: any = null;
  private isInitialized = false;
  private processingQueue: Map<string, NATSMessage> = new Map();
  private activeProcessing = new Set<string>();
  private stats: IntegrationStats = {
    messagesProcessed: 0,
    successfulProcessing: 0,
    failedProcessing: 0,
    averageProcessingTime: 0,
    serviceHealth: {
      nats: false,
      langchain: false,
      rag: false,
      multiProtocol: false
    },
    lastActivity: 0,
    activeWorkflows: 0
  };
  private subscriptionIds: string[] = [];
  private healthCheckTimer: any = null;

  constructor(config: Partial<IntegrationConfig> = {}) {
    super();
    
    this.config = {
      enableNATSMessaging: config.enableNATSMessaging ?? true,
      enableLangChainIntegration: config.enableLangChainIntegration ?? true,
      enableRAGPipeline: config.enableRAGPipeline ?? true,
      enableMultiProtocol: config.enableMultiProtocol ?? true,
      autoProcessMessages: config.autoProcessMessages ?? true,
      enableMessagePersistence: config.enableMessagePersistence ?? true,
      enableEventForwarding: config.enableEventForwarding ?? true,
      enablePerformanceMonitoring: config.enablePerformanceMonitoring ?? true,
      enableLegalWorkflows: config.enableLegalWorkflows ?? true,
      enableCaseManagement: config.enableCaseManagement ?? true,
      enableDocumentProcessing: config.enableDocumentProcessing ?? true,
      enableRealTimeAnalysis: config.enableRealTimeAnalysis ?? true,
      maxConcurrentProcessing: config.maxConcurrentProcessing || 5,
      messageRetentionHours: config.messageRetentionHours || 24,
      healthCheckInterval: config.healthCheckInterval || 30000,
      ...config
    };

    this.setupEventListeners();
  }

  async initialize(): Promise<boolean> {
    try {
      console.log('🔄 Initializing NATS + LangChain + RAG Integration...');

      // Initialize NATS service
      if (this.config.enableNATSMessaging) {
        await this.initializeNATSService();
      }

      // Initialize LangChain service
      if (this.config.enableLangChainIntegration) {
        await this.initializeLangChainService();
      }

      // Initialize RAG pipeline
      if (this.config.enableRAGPipeline) {
        await this.initializeRAGPipeline();
      }

      // Setup message processing workflows
      if (this.config.autoProcessMessages) {
        await this.setupMessageProcessing();
      }

      // Setup legal AI workflows
      if (this.config.enableLegalWorkflows) {
        await this.setupLegalWorkflows();
      }

      // Start health monitoring
      if (this.config.enablePerformanceMonitoring) {
        this.startHealthMonitoring();
      }

      this.isInitialized = true;
      this.emit('integration:initialized', { config: this.config });
      
      console.log('✓ NATS + LangChain + RAG Integration initialized');
      integrationStats.set(this.stats);
      
      return true;

    } catch (error) {
      console.error('❌ Integration initialization failed:', error);
      this.emit('integration:error', { error: error.message });
      return false;
    }
  }

  private async initializeNATSService(): Promise<void> {
    this.natsService = getNATSService();
    
    if (!this.natsService) {
      this.natsService = createNATSService({
        enableLegalChannels: true,
        enableDocumentStreaming: true,
        enableRealTimeAnalysis: true
      });
      await this.natsService.initialize();
    }

    this.stats.serviceHealth.nats = this.natsService.isReady;
    console.log('✓ NATS service initialized');
  }

  private async initializeLangChainService(): Promise<void> {
    this.langchainService = getLangChainService();
    
    if (this.langchainService) {
      this.stats.serviceHealth.langchain = this.langchainService.isReady;
      this.setupLangChainEventForwarding();
      console.log('✓ LangChain service integrated');
    } else {
      console.warn('⚠️ LangChain service not available');
    }
  }

  private async initializeRAGPipeline(): Promise<void> {
    try {
      const ragReady = await ragPipeline.initialize();
      this.stats.serviceHealth.rag = ragReady;
      console.log('✓ RAG pipeline integrated');
    } catch (error) {
      console.warn('⚠️ RAG pipeline initialization failed:', error);
      this.stats.serviceHealth.rag = false;
    }
  }

  private async setupMessageProcessing(): Promise<void> {
    if (!this.natsService) return;

    // Subscribe to all legal AI subjects for processing
    const processingSubjects = [
      NATS_SUBJECTS.SEARCH_QUERY,
      NATS_SUBJECTS.CHAT_MESSAGE,
      NATS_SUBJECTS.DOCUMENT_UPLOADED,
      NATS_SUBJECTS.AI_ANALYSIS_STARTED,
      NATS_SUBJECTS.CASE_CREATED
    ];

    for (const subject of processingSubjects) {
      const subscriptionId = await this.natsService.subscribe(subject, (message: NATSMessage) => {
        this.processMessage(message);
      });
      this.subscriptionIds.push(subscriptionId);
    }

    console.log(`✓ Message processing setup for ${processingSubjects.length} subjects`);
  }

  private async setupLegalWorkflows(): Promise<void> {
    if (!this.natsService) return;

    // Legal case workflow
    await this.natsService.subscribe(NATS_SUBJECTS.CASE_CREATED, async (message: NATSMessage) => {
      await this.handleCaseCreatedWorkflow(message);
    });

    // Document processing workflow
    await this.natsService.subscribe(NATS_SUBJECTS.DOCUMENT_UPLOADED, async (message: NATSMessage) => {
      await this.handleDocumentUploadedWorkflow(message);
    });

    // AI analysis workflow
    await this.natsService.subscribe(NATS_SUBJECTS.AI_ANALYSIS_STARTED, async (message: NATSMessage) => {
      await this.handleAIAnalysisWorkflow(message);
    });

    console.log('✓ Legal AI workflows setup');
  }

  private setupLangChainEventForwarding(): void {
    if (!this.langchainService || !this.natsService) return;

    // Forward LangChain events to NATS
    this.langchainService.on('message:received', async (data: any) => {
      await this.natsService.publish(NATS_SUBJECTS.CHAT_RESPONSE, {
        sessionId: data.sessionId,
        message: data.message,
        response: data.response,
        timestamp: Date.now()
      });
    });

    this.langchainService.on('tool:executed', async (data: any) => {
      await this.natsService.publishAIAnalysisEvent('completed', {
        toolName: data.toolName,
        input: data.input,
        output: data.output,
        executionTime: data.executionTime
      });
    });

    this.langchainService.on('streaming:chunk', async (data: any) => {
      await this.natsService.publish(NATS_SUBJECTS.CHAT_STREAMING, {
        sessionId: data.sessionId,
        chunk: data.chunk,
        isComplete: data.isComplete
      });
    });

    console.log('✓ LangChain event forwarding setup');
  }

  // ============ Message Processing ============

  private async processMessage(message: NATSMessage): Promise<void> {
    if (this.activeProcessing.size >= this.config.maxConcurrentProcessing) {
      this.processingQueue.set(message.messageId, message);
      return;
    }

    this.activeProcessing.add(message.messageId);
    this.stats.activeWorkflows++;
    
    const startTime = Date.now();
    let result: ProcessingResult;

    try {
      result = await this.executeMessageProcessing(message);
      this.stats.successfulProcessing++;
    } catch (error) {
      result = {
        messageId: message.messageId,
        success: false,
        processingTime: Date.now() - startTime,
        error: error.message,
        serviceUsed: [],
        timestamp: Date.now()
      };
      this.stats.failedProcessing++;
    }

    this.stats.messagesProcessed++;
    this.stats.lastActivity = Date.now();
    this.updateAverageProcessingTime(result.processingTime);

    this.activeProcessing.delete(message.messageId);
    this.stats.activeWorkflows--;

    this.emit('message:processed', result);
    integrationStats.set(this.stats);

    // Process next queued message
    this.processNextQueuedMessage();
  }

  private async executeMessageProcessing(message: NATSMessage): Promise<ProcessingResult> {
    const startTime = Date.now();
    const serviceUsed: string[] = [];
    let result: any = null;

    switch (message.subject) {
      case NATS_SUBJECTS.SEARCH_QUERY:
        result = await this.processSearchQuery(message);
        serviceUsed.push('rag-pipeline');
        break;

      case NATS_SUBJECTS.CHAT_MESSAGE:
        result = await this.processChatMessage(message);
        serviceUsed.push('langchain');
        break;

      case NATS_SUBJECTS.DOCUMENT_UPLOADED:
        result = await this.processDocumentUpload(message);
        serviceUsed.push('multi-protocol');
        break;

      case NATS_SUBJECTS.AI_ANALYSIS_STARTED:
        result = await this.processAIAnalysis(message);
        serviceUsed.push('langchain', 'rag-pipeline');
        break;

      default:
        result = await this.processGenericMessage(message);
        serviceUsed.push('generic');
    }

    return {
      messageId: message.messageId,
      success: true,
      processingTime: Date.now() - startTime,
      result,
      serviceUsed,
      timestamp: Date.now()
    };
  }

  private async processSearchQuery(message: NATSMessage): Promise<any> {
    if (!this.stats.serviceHealth.rag) {
      throw new Error('RAG pipeline not available');
    }

    const query = message.data.query || message.data.toString();
    const options = message.data.options || {};

    const searchResult = await ragPipeline.query(query, options);

    // Publish results back to NATS
    if (this.natsService) {
      await this.natsService.publish(NATS_SUBJECTS.SEARCH_RESULTS, {
        queryId: message.messageId,
        query,
        results: searchResult,
        timestamp: Date.now()
      });
    }

    return searchResult;
  }

  private async processChatMessage(message: NATSMessage): Promise<any> {
    if (!this.langchainService || !this.langchainService.isReady) {
      throw new Error('LangChain service not available');
    }

    const { content, sessionId } = message.data;
    
    // Get or create session
    let session = this.langchainService.getSession(sessionId);
    if (!session) {
      session = this.langchainService.createSession(`NATS Session ${sessionId}`);
    }

    const result = await this.langchainService.sendMessage(session.id, content);

    // Response is automatically forwarded via event listeners
    return result;
  }

  private async processDocumentUpload(message: NATSMessage): Promise<any> {
    if (!this.config.enableMultiProtocol) {
      throw new Error('Multi-protocol processing not enabled');
    }

    const documentData = message.data;
    
    // Use multi-protocol router for document processing
    const uploadResult = await multiProtocolRouter.execute('document_upload', documentData, {
      preferredProtocol: 'grpc',
      timeout: 60000
    });

    // Publish processing result
    if (this.natsService) {
      await this.natsService.publishDocumentEvent('processed', {
        documentId: documentData.documentId || documentData.id,
        caseId: documentData.caseId,
        processingResult: uploadResult,
        timestamp: Date.now()
      });
    }

    return uploadResult;
  }

  private async processAIAnalysis(message: NATSMessage): Promise<any> {
    const { documentContent, analysisType, caseId } = message.data;

    let result: any = {};

    // Use LangChain for analysis if available
    if (this.langchainService && this.langchainService.isReady) {
      result.langchainAnalysis = await this.langchainService.analyzeLegalDocument(
        documentContent,
        analysisType
      );
    }

    // Use RAG pipeline for context-aware analysis
    if (this.stats.serviceHealth.rag) {
      result.ragAnalysis = await ragPipeline.query(
        `Analyze this ${analysisType}: ${documentContent.substring(0, 500)}`,
        { caseId }
      );
    }

    // Publish analysis result
    if (this.natsService) {
      await this.natsService.publishAIAnalysisEvent('completed', {
        caseId,
        analysisType,
        result,
        timestamp: Date.now()
      });
    }

    return result;
  }

  private async processGenericMessage(message: NATSMessage): Promise<any> {
    // Generic message processing fallback
    console.log(`📨 Processing generic message: ${message.subject}`, message.data);
    return { processed: true, timestamp: Date.now() };
  }

  // ============ Legal Workflow Handlers ============

  private async handleCaseCreatedWorkflow(message: NATSMessage): Promise<void> {
    const caseData = message.data;
    console.log(`📋 Case created workflow: ${caseData.title || caseData.caseId}`);

    // Trigger initial case analysis
    if (this.natsService) {
      await this.natsService.publishAIAnalysisEvent('started', {
        caseId: caseData.caseId || caseData.id,
        analysisType: 'case_initialization',
        priority: 'high'
      });
    }

    this.emit('workflow:case_created', caseData);
  }

  private async handleDocumentUploadedWorkflow(message: NATSMessage): Promise<void> {
    const documentData = message.data;
    console.log(`📄 Document uploaded workflow: ${documentData.name || documentData.documentId}`);

    // Trigger document processing pipeline
    await this.processDocumentUpload(message);

    this.emit('workflow:document_uploaded', documentData);
  }

  private async handleAIAnalysisWorkflow(message: NATSMessage): Promise<void> {
    const analysisData = message.data;
    console.log(`🧠 AI analysis workflow: ${analysisData.analysisType}`);

    // Process the analysis request
    await this.processAIAnalysis(message);

    this.emit('workflow:ai_analysis', analysisData);
  }

  // ============ Health Monitoring ============

  private startHealthMonitoring(): void {
    this.healthCheckTimer = setInterval(() => {
      this.performHealthCheck();
    }, this.config.healthCheckInterval);
  }

  private async performHealthCheck(): Promise<void> {
    // Check NATS health
    if (this.natsService) {
      this.stats.serviceHealth.nats = this.natsService.isConnected;
    }

    // Check LangChain health
    if (this.langchainService) {
      this.stats.serviceHealth.langchain = this.langchainService.isReady;
    }

    // Check RAG pipeline health
    try {
      const ragHealth = await ragPipeline.getHealthStatus();
      this.stats.serviceHealth.rag = ragHealth.initialized;
    } catch (error) {
      this.stats.serviceHealth.rag = false;
    }

    // Check multi-protocol health
    this.stats.serviceHealth.multiProtocol = this.config.enableMultiProtocol;

    // Publish health status
    if (this.natsService && this.natsService.isConnected) {
      await this.natsService.publish(NATS_SUBJECTS.SYSTEM_HEALTH, {
        integration: {
          initialized: this.isInitialized,
          stats: this.stats,
          timestamp: Date.now()
        }
      });
    }

    integrationStats.set(this.stats);
    this.emit('health:checked', this.stats);
  }

  // ============ Utility Methods ============

  private processNextQueuedMessage(): void {
    if (this.processingQueue.size === 0 || 
        this.activeProcessing.size >= this.config.maxConcurrentProcessing) {
      return;
    }

    const [messageId, message] = this.processingQueue.entries().next().value;
    this.processingQueue.delete(messageId);
    this.processMessage(message);
  }

  private updateAverageProcessingTime(processingTime: number): void {
    const count = this.stats.messagesProcessed;
    this.stats.averageProcessingTime = 
      (this.stats.averageProcessingTime * (count - 1) + processingTime) / count;
  }

  private setupEventListeners(): void {
    this.on('message:processed', (result: ProcessingResult) => {
      console.log(`✓ Message processed: ${result.messageId} (${result.processingTime}ms)`);
    });

    this.on('workflow:case_created', (caseData: any) => {
      console.log(`📋 Case workflow initiated: ${caseData.caseId}`);
    });

    this.on('integration:error', (data: any) => {
      console.error(`❌ Integration error: ${data.error}`);
    });
  }

  // ============ Public API ============

  /**
   * Send a message through the integration system
   */
  async sendMessage(subject: string, data: any, options: any = {}): Promise<void> {
    if (!this.natsService) {
      throw new Error('NATS service not available');
    }

    await this.natsService.publish(subject, data, options);
  }

  /**
   * Execute a search query through the integration
   */
  async executeSearch(query: string, options: any = {}): Promise<any> {
    if (!this.natsService) {
      throw new Error('NATS service not available');
    }

    return await this.natsService.request(NATS_SUBJECTS.SEARCH_QUERY, {
      query,
      options,
      timestamp: Date.now()
    });
  }

  /**
   * Send a chat message through the integration
   */
  async sendChatMessage(content: string, sessionId: string): Promise<any> {
    if (!this.natsService) {
      throw new Error('NATS service not available');
    }

    await this.natsService.publishChatMessage({
      content,
      sessionId,
      timestamp: Date.now()
    }, sessionId);
  }

  /**
   * Trigger AI analysis through the integration
   */
  async triggerAIAnalysis(documentContent: string, analysisType: string, caseId?: string): Promise<void> {
    if (!this.natsService) {
      throw new Error('NATS service not available');
    }

    await this.natsService.publishAIAnalysisEvent('started', {
      documentContent,
      analysisType,
      caseId,
      timestamp: Date.now()
    });
  }

  /**
   * Get integration statistics
   */
  getStats(): IntegrationStats {
    return { ...this.stats };
  }

  /**
   * Get service health status
   */
  getServiceHealth(): IntegrationStats['serviceHealth'] {
    return { ...this.stats.serviceHealth };
  }

  /**
   * Check if integration is ready
   */
  get isReady(): boolean {
    return this.isInitialized && 
           this.stats.serviceHealth.nats &&
           (this.stats.serviceHealth.langchain || this.stats.serviceHealth.rag);
  }

  // ============ Cleanup ============

  async cleanup(): Promise<void> {
    console.log('🧹 Cleaning up NATS + LangChain + RAG Integration...');

    // Stop health monitoring
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }

    // Unsubscribe from all NATS subjects
    if (this.natsService) {
      for (const subscriptionId of this.subscriptionIds) {
        await this.natsService.unsubscribe(subscriptionId);
      }
    }

    // Clear processing queue
    this.processingQueue.clear();
    this.activeProcessing.clear();

    this.isInitialized = false;
    this.removeAllListeners();

    console.log('✓ Integration cleanup complete');
  }
}

// Svelte stores for reactive access
export const integrationStats = writable<IntegrationStats>({
  messagesProcessed: 0,
  successfulProcessing: 0,
  failedProcessing: 0,
  averageProcessingTime: 0,
  serviceHealth: {
    nats: false,
    langchain: false,
    rag: false,
    multiProtocol: false
  },
  lastActivity: 0,
  activeWorkflows: 0
});

export const integrationHealth = derived(
  [integrationStats],
  ([$stats]) => ({
    overallHealth: Object.values($stats.serviceHealth).filter(Boolean).length / 4,
    processingSuccessRate: $stats.messagesProcessed > 0 
      ? $stats.successfulProcessing / $stats.messagesProcessed 
      : 0,
    averageResponseTime: $stats.averageProcessingTime,
    isOperational: $stats.serviceHealth.nats && 
                   ($stats.serviceHealth.langchain || $stats.serviceHealth.rag),
    lastActivity: $stats.lastActivity,
    activeLoad: $stats.activeWorkflows
  })
);

// Singleton instance
let integrationInstance: NATSLangChainIntegration | null = null;

export function createIntegrationService(config?: Partial<IntegrationConfig>): NATSLangChainIntegration {
  if (integrationInstance) {
    integrationInstance.cleanup();
  }
  
  integrationInstance = new NATSLangChainIntegration(config);
  return integrationInstance;
}

export function getIntegrationService(): NATSLangChainIntegration | null {
  return integrationInstance;
}

// Auto-initialize with default config in browser
if (browser) {
  const defaultConfig: Partial<IntegrationConfig> = {
    enableNATSMessaging: true,
    enableLangChainIntegration: true,
    enableRAGPipeline: true,
    enableMultiProtocol: true,
    enableLegalWorkflows: true,
    autoProcessMessages: true,
    enablePerformanceMonitoring: true
  };

  integrationInstance = createIntegrationService(defaultConfig);
  integrationInstance.initialize().catch(console.error);
}

export default NATSLangChainIntegration;