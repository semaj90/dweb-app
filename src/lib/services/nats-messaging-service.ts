// NATS Messaging Service for Legal AI Platform
// High-performance distributed messaging system with real-time event handling

import { EventEmitter } from 'events';
import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

/**
 * NATS Messaging Service Configuration
 */
export interface NATSConfig {
  // Server configuration
  servers: string[];
  name: string;
  reconnect: boolean;
  maxReconnectAttempts: number;
  reconnectTimeWait: number;
  
  // Authentication
  user?: string;
  pass?: string;
  token?: string;
  
  // TLS configuration
  tls?: boolean;
  
  // Performance settings
  pingInterval: number;
  maxPingOut: number;
  
  // Legal AI specific settings
  enableLegalChannels: boolean;
  enableDocumentStreaming: boolean;
  enableRealTimeAnalysis: boolean;
  enableCaseUpdates: boolean;
}

/**
 * NATS Message Interface
 */
export interface NATSMessage {
  subject: string;
  data: any;
  timestamp: number;
  messageId: string;
  correlationId?: string;
  replyTo?: string;
  headers?: Record<string, string>;
  metadata?: {
    source: string;
    userId?: string;
    caseId?: string;
    sessionId?: string;
    priority: 'low' | 'normal' | 'high' | 'critical';
  };
}

/**
 * NATS Subject Patterns for Legal AI
 */
export const NATS_SUBJECTS = {
  // Case management
  CASE_CREATED: 'legal.case.created',
  CASE_UPDATED: 'legal.case.updated',
  CASE_CLOSED: 'legal.case.closed',
  
  // Document events
  DOCUMENT_UPLOADED: 'legal.document.uploaded',
  DOCUMENT_PROCESSED: 'legal.document.processed',
  DOCUMENT_ANALYZED: 'legal.document.analyzed',
  DOCUMENT_INDEXED: 'legal.document.indexed',
  
  // AI analysis events
  AI_ANALYSIS_STARTED: 'legal.ai.analysis.started',
  AI_ANALYSIS_COMPLETED: 'legal.ai.analysis.completed',
  AI_ANALYSIS_FAILED: 'legal.ai.analysis.failed',
  
  // Search and retrieval
  SEARCH_QUERY: 'legal.search.query',
  SEARCH_RESULTS: 'legal.search.results',
  
  // Real-time chat and assistance
  CHAT_MESSAGE: 'legal.chat.message',
  CHAT_RESPONSE: 'legal.chat.response',
  CHAT_STREAMING: 'legal.chat.streaming',
  
  // System events
  SYSTEM_HEALTH: 'system.health',
  SYSTEM_METRICS: 'system.metrics',
  SYSTEM_ALERTS: 'system.alerts',
  
  // Evidence and timeline
  EVIDENCE_ADDED: 'legal.evidence.added',
  EVIDENCE_UPDATED: 'legal.evidence.updated',
  TIMELINE_EVENT: 'legal.timeline.event',
  
  // Collaboration
  USER_ACTIVITY: 'legal.user.activity',
  COLLABORATION_UPDATE: 'legal.collaboration.update',
  
  // Notifications
  NOTIFICATION_SEND: 'legal.notification.send',
  NOTIFICATION_DELIVERED: 'legal.notification.delivered'
} as const;

/**
 * NATS Connection Status
 */
export interface NATSStatus {
  connected: boolean;
  connecting: boolean;
  disconnected: boolean;
  reconnecting: boolean;
  closed: boolean;
  error: string | null;
  lastConnected: number | null;
  reconnectAttempts: number;
  subscriptions: number;
  publishedMessages: number;
  receivedMessages: number;
  bytesIn: number;
  bytesOut: number;
}

/**
 * NATS Messaging Service
 * Provides high-performance distributed messaging for the legal AI platform
 */
export class NATSMessagingService extends EventEmitter {
  private config: NATSConfig;
  private connection: any = null;
  private subscriptions = new Map<string, any>();
  private status: NATSStatus = {
    connected: false,
    connecting: false,
    disconnected: true,
    reconnecting: false,
    closed: false,
    error: null,
    lastConnected: null,
    reconnectAttempts: 0,
    subscriptions: 0,
    publishedMessages: 0,
    receivedMessages: 0,
    bytesIn: 0,
    bytesOut: 0
  };
  private messageQueue: NATSMessage[] = [];
  private messageHistory = new Map<string, NATSMessage[]>();
  private isInitialized = false;

  constructor(config: Partial<NATSConfig> = {}) {
    super();
    
    this.config = {
      servers: config.servers || ['ws://localhost:4222', 'ws://localhost:4223'],
      name: config.name || 'LegalAI-NATS-Client',
      reconnect: config.reconnect ?? true,
      maxReconnectAttempts: config.maxReconnectAttempts || 10,
      reconnectTimeWait: config.reconnectTimeWait || 2000,
      pingInterval: config.pingInterval || 2000,
      maxPingOut: config.maxPingOut || 2,
      enableLegalChannels: config.enableLegalChannels ?? true,
      enableDocumentStreaming: config.enableDocumentStreaming ?? true,
      enableRealTimeAnalysis: config.enableRealTimeAnalysis ?? true,
      enableCaseUpdates: config.enableCaseUpdates ?? true,
      ...config
    };

    this.setupEventListeners();
  }

  async initialize(): Promise<boolean> {
    try {
      console.log('🔄 Initializing NATS messaging service...');
      
      // For browser environments, we'll simulate NATS with WebSocket
      if (browser) {
        await this.initializeBrowserNATS();
      } else {
        await this.initializeServerNATS();
      }

      // Setup default legal AI subscriptions
      if (this.config.enableLegalChannels) {
        await this.setupLegalChannels();
      }

      // Start health monitoring
      this.startHealthMonitoring();

      this.isInitialized = true;
      this.emit('nats:initialized', { config: this.config });
      
      console.log('✓ NATS messaging service initialized');
      natsStatus.set(this.status);
      
      return true;

    } catch (error) {
      console.error('❌ NATS initialization failed:', error);
      this.status.error = error.message;
      this.emit('nats:error', { error: error.message });
      natsStatus.set(this.status);
      return false;
    }
  }

  private async initializeBrowserNATS(): Promise<void> {
    // Browser implementation using WebSocket simulation
    console.log('🌐 Initializing browser NATS simulation...');
    
    // Simulate connection establishment
    this.status.connecting = true;
    this.status.disconnected = false;
    natsStatus.set(this.status);

    // Simulate connection delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Mock successful connection
    this.status.connected = true;
    this.status.connecting = false;
    this.status.lastConnected = Date.now();
    this.status.error = null;
    
    this.emit('nats:connected');
    console.log('✓ Browser NATS simulation connected');
  }

  private async initializeServerNATS(): Promise<void> {
    // Server-side NATS implementation would go here
    // For now, we'll use the browser simulation
    await this.initializeBrowserNATS();
  }

  private async setupLegalChannels(): Promise<void> {
    console.log('📡 Setting up legal AI channels...');
    
    const legalChannels = [
      NATS_SUBJECTS.CASE_CREATED,
      NATS_SUBJECTS.CASE_UPDATED,
      NATS_SUBJECTS.DOCUMENT_UPLOADED,
      NATS_SUBJECTS.DOCUMENT_PROCESSED,
      NATS_SUBJECTS.AI_ANALYSIS_COMPLETED,
      NATS_SUBJECTS.SEARCH_QUERY,
      NATS_SUBJECTS.CHAT_MESSAGE,
      NATS_SUBJECTS.EVIDENCE_ADDED,
      NATS_SUBJECTS.SYSTEM_HEALTH
    ];

    for (const channel of legalChannels) {
      await this.subscribe(channel, (message) => {
        this.handleLegalChannelMessage(channel, message);
      });
    }

    console.log(`✓ ${legalChannels.length} legal channels subscribed`);
  }

  private handleLegalChannelMessage(subject: string, message: NATSMessage): void {
    console.log(`📨 Legal channel message: ${subject}`, message);
    
    // Store message in history
    if (!this.messageHistory.has(subject)) {
      this.messageHistory.set(subject, []);
    }
    
    const history = this.messageHistory.get(subject)!;
    history.push(message);
    
    // Keep only last 100 messages per channel
    if (history.length > 100) {
      this.messageHistory.set(subject, history.slice(-100));
    }

    // Emit specific events based on subject
    this.emit(`message:${subject}`, message);
    
    // Update message stats
    this.status.receivedMessages++;
    natsStatus.set(this.status);
  }

  // ============ Publishing Methods ============

  /**
   * Publish a message to a NATS subject
   */
  async publish(subject: string, data: any, options: {
    correlationId?: string;
    replyTo?: string;
    headers?: Record<string, string>;
    metadata?: NATSMessage['metadata'];
  } = {}): Promise<void> {
    const message: NATSMessage = {
      subject,
      data,
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
      correlationId: options.correlationId,
      replyTo: options.replyTo,
      headers: options.headers,
      metadata: {
        source: 'legal-ai-client',
        priority: 'normal',
        ...options.metadata
      }
    };

    if (!this.status.connected) {
      // Queue message for later delivery
      this.messageQueue.push(message);
      console.log(`📬 Message queued for ${subject} (connection not ready)`);
      return;
    }

    try {
      // Simulate publishing in browser environment
      await this.simulatePublish(message);
      
      this.status.publishedMessages++;
      this.status.bytesOut += JSON.stringify(data).length;
      
      this.emit('nats:published', { subject, messageId: message.messageId });
      natsStatus.set(this.status);
      
      console.log(`📤 Published to ${subject}:`, message.messageId);

    } catch (error) {
      console.error(`❌ Failed to publish to ${subject}:`, error);
      this.emit('nats:publish_failed', { subject, error: error.message });
      throw error;
    }
  }

  private async simulatePublish(message: NATSMessage): Promise<void> {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 10 + Math.random() * 50));
    
    // Simulate potential delivery to subscribers
    const subscription = this.subscriptions.get(message.subject);
    if (subscription && subscription.callback) {
      // Simulate async delivery
      setTimeout(() => {
        subscription.callback(message);
      }, 5);
    }
  }

  /**
   * Publish legal case event
   */
  async publishCaseEvent(eventType: 'created' | 'updated' | 'closed', caseData: any): Promise<void> {
    const subjects = {
      created: NATS_SUBJECTS.CASE_CREATED,
      updated: NATS_SUBJECTS.CASE_UPDATED,
      closed: NATS_SUBJECTS.CASE_CLOSED
    };

    await this.publish(subjects[eventType], caseData, {
      metadata: {
        source: 'case-management',
        caseId: caseData.id,
        priority: 'high'
      }
    });
  }

  /**
   * Publish document processing event
   */
  async publishDocumentEvent(eventType: 'uploaded' | 'processed' | 'analyzed' | 'indexed', documentData: any): Promise<void> {
    const subjects = {
      uploaded: NATS_SUBJECTS.DOCUMENT_UPLOADED,
      processed: NATS_SUBJECTS.DOCUMENT_PROCESSED,
      analyzed: NATS_SUBJECTS.DOCUMENT_ANALYZED,
      indexed: NATS_SUBJECTS.DOCUMENT_INDEXED
    };

    await this.publish(subjects[eventType], documentData, {
      metadata: {
        source: 'document-processor',
        caseId: documentData.caseId,
        priority: 'normal'
      }
    });
  }

  /**
   * Publish AI analysis event
   */
  async publishAIAnalysisEvent(eventType: 'started' | 'completed' | 'failed', analysisData: any): Promise<void> {
    const subjects = {
      started: NATS_SUBJECTS.AI_ANALYSIS_STARTED,
      completed: NATS_SUBJECTS.AI_ANALYSIS_COMPLETED,
      failed: NATS_SUBJECTS.AI_ANALYSIS_FAILED
    };

    await this.publish(subjects[eventType], analysisData, {
      metadata: {
        source: 'ai-analysis-engine',
        caseId: analysisData.caseId,
        priority: eventType === 'failed' ? 'high' : 'normal'
      }
    });
  }

  /**
   * Publish chat message
   */
  async publishChatMessage(messageData: any, sessionId: string): Promise<void> {
    await this.publish(NATS_SUBJECTS.CHAT_MESSAGE, messageData, {
      metadata: {
        source: 'chat-interface',
        sessionId,
        priority: 'normal'
      }
    });
  }

  /**
   * Publish search query
   */
  async publishSearchQuery(queryData: any): Promise<void> {
    await this.publish(NATS_SUBJECTS.SEARCH_QUERY, queryData, {
      metadata: {
        source: 'search-interface',
        priority: 'normal'
      }
    });
  }

  // ============ Subscription Methods ============

  /**
   * Subscribe to a NATS subject
   */
  async subscribe(subject: string, callback: (message: NATSMessage) => void, options: {
    queue?: string;
    maxMessages?: number;
  } = {}): Promise<string> {
    const subscriptionId = this.generateSubscriptionId();
    
    this.subscriptions.set(subject, {
      id: subscriptionId,
      subject,
      callback,
      options,
      createdAt: Date.now(),
      messageCount: 0
    });

    this.status.subscriptions = this.subscriptions.size;
    natsStatus.set(this.status);

    this.emit('nats:subscribed', { subject, subscriptionId });
    console.log(`📡 Subscribed to ${subject} (${subscriptionId})`);

    return subscriptionId;
  }

  /**
   * Unsubscribe from a subject
   */
  async unsubscribe(subscriptionId: string): Promise<void> {
    for (const [subject, subscription] of this.subscriptions.entries()) {
      if (subscription.id === subscriptionId) {
        this.subscriptions.delete(subject);
        this.status.subscriptions = this.subscriptions.size;
        natsStatus.set(this.status);
        
        this.emit('nats:unsubscribed', { subject, subscriptionId });
        console.log(`❌ Unsubscribed from ${subject} (${subscriptionId})`);
        return;
      }
    }
  }

  /**
   * Subscribe to case updates
   */
  async subscribeToCaseUpdates(caseId: string, callback: (message: NATSMessage) => void): Promise<string> {
    return await this.subscribe(`${NATS_SUBJECTS.CASE_UPDATED}.${caseId}`, callback);
  }

  /**
   * Subscribe to document processing for a case
   */
  async subscribeToDocumentProcessing(caseId: string, callback: (message: NATSMessage) => void): Promise<string> {
    return await this.subscribe(`${NATS_SUBJECTS.DOCUMENT_PROCESSED}.${caseId}`, callback);
  }

  /**
   * Subscribe to chat messages for a session
   */
  async subscribeToChatSession(sessionId: string, callback: (message: NATSMessage) => void): Promise<string> {
    return await this.subscribe(`${NATS_SUBJECTS.CHAT_MESSAGE}.${sessionId}`, callback);
  }

  // ============ Request-Reply Pattern ============

  /**
   * Request-reply pattern for synchronous communication
   */
  async request(subject: string, data: any, timeout: number = 5000): Promise<NATSMessage> {
    return new Promise(async (resolve, reject) => {
      const replySubject = this.generateReplySubject();
      const correlationId = this.generateMessageId();
      
      // Setup reply subscription
      const timeoutId = setTimeout(() => {
        this.unsubscribe(replySubject);
        reject(new Error(`Request timeout for subject: ${subject}`));
      }, timeout);

      const subscriptionId = await this.subscribe(replySubject, (message) => {
        if (message.correlationId === correlationId) {
          clearTimeout(timeoutId);
          this.unsubscribe(subscriptionId);
          resolve(message);
        }
      });

      // Send request
      await this.publish(subject, data, {
        correlationId,
        replyTo: replySubject
      });
    });
  }

  /**
   * Reply to a request
   */
  async reply(originalMessage: NATSMessage, responseData: any): Promise<void> {
    if (!originalMessage.replyTo) {
      throw new Error('Cannot reply to message without replyTo subject');
    }

    await this.publish(originalMessage.replyTo, responseData, {
      correlationId: originalMessage.correlationId,
      metadata: {
        source: 'reply-handler',
        priority: 'normal'
      }
    });
  }

  // ============ Streaming Methods ============

  /**
   * Create a stream for document processing
   */
  async createDocumentStream(caseId: string): Promise<{
    publish: (data: any) => Promise<void>;
    subscribe: (callback: (data: any) => void) => Promise<string>;
    close: () => Promise<void>;
  }> {
    const streamSubject = `${NATS_SUBJECTS.DOCUMENT_PROCESSED}.stream.${caseId}`;
    
    return {
      publish: async (data: any) => {
        await this.publish(streamSubject, data, {
          metadata: {
            source: 'document-stream',
            caseId,
            priority: 'normal'
          }
        });
      },
      
      subscribe: async (callback: (data: any) => void) => {
        return await this.subscribe(streamSubject, (message) => {
          callback(message.data);
        });
      },
      
      close: async () => {
        // Close stream logic would go here
        console.log(`📡 Document stream closed for case: ${caseId}`);
      }
    };
  }

  // ============ Health and Monitoring ============

  private startHealthMonitoring(): void {
    setInterval(() => {
      this.publishHealthStatus();
    }, 30000); // Every 30 seconds
  }

  private async publishHealthStatus(): Promise<void> {
    const healthData = {
      timestamp: Date.now(),
      status: this.status,
      uptime: this.isInitialized ? Date.now() - (this.status.lastConnected || Date.now()) : 0,
      subscriptions: this.subscriptions.size,
      queuedMessages: this.messageQueue.length,
      memoryUsage: this.getMemoryUsage()
    };

    try {
      await this.publish(NATS_SUBJECTS.SYSTEM_HEALTH, healthData, {
        metadata: {
          source: 'nats-client',
          priority: 'low'
        }
      });
    } catch (error) {
      console.warn('Failed to publish health status:', error);
    }
  }

  private getMemoryUsage(): any {
    if (typeof window !== 'undefined' && (window as any).performance?.memory) {
      const memory = (window as any).performance.memory;
      return {
        used: memory.usedJSHeapSize,
        total: memory.totalJSHeapSize,
        limit: memory.jsHeapSizeLimit
      };
    }
    return null;
  }

  // ============ Message History and Analytics ============

  getMessageHistory(subject: string, limit: number = 50): NATSMessage[] {
    const history = this.messageHistory.get(subject) || [];
    return history.slice(-limit);
  }

  getSubscriptionStats(): Array<{
    subject: string;
    subscriptionId: string;
    messageCount: number;
    createdAt: number;
  }> {
    return Array.from(this.subscriptions.entries()).map(([subject, sub]) => ({
      subject,
      subscriptionId: sub.id,
      messageCount: sub.messageCount,
      createdAt: sub.createdAt
    }));
  }

  getConnectionMetrics(): NATSStatus {
    return { ...this.status };
  }

  // ============ Utility Methods ============

  private setupEventListeners(): void {
    this.on('nats:connected', () => {
      this.processQueuedMessages();
    });

    this.on('nats:disconnected', () => {
      this.status.connected = false;
      this.status.disconnected = true;
      natsStatus.set(this.status);
    });
  }

  private async processQueuedMessages(): Promise<void> {
    if (this.messageQueue.length === 0) return;

    console.log(`📬 Processing ${this.messageQueue.length} queued messages...`);
    
    const messages = [...this.messageQueue];
    this.messageQueue = [];

    for (const message of messages) {
      try {
        await this.simulatePublish(message);
        this.status.publishedMessages++;
      } catch (error) {
        console.error('Failed to process queued message:', error);
        // Re-queue failed messages
        this.messageQueue.push(message);
      }
    }

    natsStatus.set(this.status);
  }

  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateSubscriptionId(): string {
    return `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateReplySubject(): string {
    return `_INBOX.${Math.random().toString(36).substr(2, 9)}`;
  }

  // ============ Cleanup ============

  async disconnect(): Promise<void> {
    console.log('🔌 Disconnecting NATS...');
    
    // Unsubscribe from all subjects
    for (const [subject] of this.subscriptions.entries()) {
      // Cleanup subscriptions
    }
    this.subscriptions.clear();

    this.status.connected = false;
    this.status.disconnected = true;
    this.status.closed = true;
    
    this.emit('nats:disconnected');
    natsStatus.set(this.status);
    
    console.log('✓ NATS disconnected');
  }

  // ============ Getters ============

  get isConnected(): boolean {
    return this.status.connected;
  }

  get isReady(): boolean {
    return this.isInitialized && this.status.connected;
  }

  get connectionStatus(): NATSStatus {
    return { ...this.status };
  }

  get activeSubscriptions(): number {
    return this.subscriptions.size;
  }

  get messageStats(): { published: number; received: number; queued: number } {
    return {
      published: this.status.publishedMessages,
      received: this.status.receivedMessages,
      queued: this.messageQueue.length
    };
  }
}

// Svelte stores for reactive access
export const natsStatus = writable<NATSStatus>({
  connected: false,
  connecting: false,
  disconnected: true,
  reconnecting: false,
  closed: false,
  error: null,
  lastConnected: null,
  reconnectAttempts: 0,
  subscriptions: 0,
  publishedMessages: 0,
  receivedMessages: 0,
  bytesIn: 0,
  bytesOut: 0
});

export const natsMetrics = derived(
  [natsStatus],
  ([$status]) => ({
    connectionHealth: $status.connected ? 'healthy' : 'disconnected',
    subscriptionCount: $status.subscriptions,
    messageRate: {
      published: $status.publishedMessages,
      received: $status.receivedMessages
    },
    bandwidth: {
      in: $status.bytesIn,
      out: $status.bytesOut
    },
    reconnectionAttempts: $status.reconnectAttempts,
    lastActivity: $status.lastConnected
  })
);

// Singleton instance
let natsInstance: NATSMessagingService | null = null;

export function createNATSService(config?: Partial<NATSConfig>): NATSMessagingService {
  if (natsInstance) {
    natsInstance.removeAllListeners();
  }
  
  natsInstance = new NATSMessagingService(config);
  return natsInstance;
}

export function getNATSService(): NATSMessagingService | null {
  return natsInstance;
}

// Auto-initialize with default config in browser
if (browser) {
  const defaultConfig: Partial<NATSConfig> = {
    servers: ['ws://localhost:4222'],
    name: 'LegalAI-Browser-Client',
    enableLegalChannels: true,
    enableDocumentStreaming: true,
    enableRealTimeAnalysis: true,
    enableCaseUpdates: true
  };

  natsInstance = createNATSService(defaultConfig);
  natsInstance.initialize().catch(console.error);
}

export default NATSMessagingService;