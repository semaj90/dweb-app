/**
 * NATS Messaging Service for Legal AI Platform
 * Real-time messaging integration with NATS Server
 *
 * Features:
 * - WebSocket connection to NATS
 * - Legal AI subject patterns
 * - Real-time case updates
 * - Document processing notifications
 * - AI analysis completion events
 */

// Note: nats.ws types - using generic types for compatibility
// import { connect, StringCodec, JSONCodec } from 'nats.ws';
// import type { NatsConnection, Subscription, Msg } from 'nats.ws';

export interface LegalAIMessage {
  type: 'case.created' | 'document.uploaded' | 'ai.analysis.completed' | 'search.query' | 'chat.message' | 'system.health';
  data: any;
  timestamp: string;
  userId?: string;
  caseId?: string;
  sessionId?: string;
}

export interface MessageHandler {
  (message: LegalAIMessage): void;
}

// Generic types for NATS compatibility
interface NATSConnection {
  publish(subject: string, data: Uint8Array): void;
  subscribe(subject: string): any;
  request(subject: string, data: Uint8Array, options?: { timeout: number }): Promise<any>;
  drain(): Promise<void>;
  closed(): Promise<any>;
  isClosed(): boolean;
  info?: any;
}

interface NATSSubscription {
  unsubscribe(): void;
  [Symbol.asyncIterator](): AsyncIterator<any>;
}

interface NATSCodec {
  encode(data: any): Uint8Array;
  decode(data: Uint8Array): any;
}

// Lightweight EventEmitter (browser + Node)
class EventEmitter {
  private listeners = new Map<string, Set<(...args: any[]) => void>>();
  on(evt: string, fn: (...a: any[]) => void) { if (!this.listeners.has(evt)) this.listeners.set(evt, new Set()); this.listeners.get(evt)!.add(fn); }
  off(evt: string, fn: (...a: any[]) => void) { this.listeners.get(evt)?.delete(fn); }
  once(evt: string, fn: (...a: any[]) => void) { const wrap = (...x: any[]) => { fn(...x); this.off(evt, wrap); }; this.on(evt, wrap); }
  emit(evt: string, ...a: any[]) { this.listeners.get(evt)?.forEach(fn => { try { fn(...a); } catch { } }); }
}

interface NATSMetricsSnapshot {
  connection: { status: 'connected' | 'disconnected'; since: number | null; reconnectAttempts: number };
  messaging: { published: number; received: number; subjects: Record<string, string[]> };
}

export class NATSMessagingService extends EventEmitter {
  private connection: NATSConnection | null = null;
  private subscriptions: Map<string, NATSSubscription> = new Map();
  private messageHandlers: Map<string, Set<MessageHandler>> = new Map();
  private stringCodec: NATSCodec = {
    encode: (data: string) => new TextEncoder().encode(data),
    decode: (data: Uint8Array) => new TextDecoder().decode(data)
  };
  private jsonCodec: NATSCodec = {
    encode: (data: any) => new TextEncoder().encode(JSON.stringify(data)),
    decode: (data: Uint8Array) => JSON.parse(new TextDecoder().decode(data))
  };

  // NATS configuration for Legal AI
  private readonly config = {
    servers: ['ws://localhost:4223'], // WebSocket endpoint
    user: 'legal_ai_client',
    pass: 'legal_ai_2024',
    name: 'Legal AI SvelteKit Client',
    maxReconnectAttempts: 10,
    reconnectTimeWait: 2000,
  };
  private connectedAt: number | null = null;
  private reconnectAttempts = 0;
  private publishedCount = 0;
  private receivedCount = 0;
  private subjectSamples: Map<string, string[]> = new Map();

  // Legal AI subject patterns
  public readonly subjects = {
    // Case management
    CASE_CREATED: 'legal.case.created',
    CASE_UPDATED: 'legal.case.updated',
    CASE_DELETED: 'legal.case.deleted',

    // Document processing
    DOCUMENT_UPLOADED: 'legal.document.uploaded',
    DOCUMENT_PROCESSED: 'legal.document.processed',
    DOCUMENT_ANALYZED: 'legal.document.analyzed',

    // AI analysis
    AI_ANALYSIS_STARTED: 'legal.ai.analysis.started',
    AI_ANALYSIS_COMPLETED: 'legal.ai.analysis.completed',
    AI_SUMMARY_GENERATED: 'legal.ai.summary.generated',

    // Search and chat
    SEARCH_QUERY: 'legal.search.query',
    SEARCH_RESULTS: 'legal.search.results',
    CHAT_MESSAGE: 'legal.chat.message',
    CHAT_RESPONSE: 'legal.chat.response',

    // System events
    SYSTEM_HEALTH: 'system.health',
    SYSTEM_STATUS: 'system.status',

    // Evidence processing
    EVIDENCE_UPLOADED: 'legal.evidence.uploaded',
    EVIDENCE_VALIDATED: 'legal.evidence.validated',
    EVIDENCE_CHAIN_UPDATED: 'legal.evidence.chain.updated',

    // Real-time collaboration
    USER_JOINED: 'legal.collaboration.user.joined',
    USER_LEFT: 'legal.collaboration.user.left',
    DOCUMENT_EDITED: 'legal.collaboration.document.edited',
  };

  /**
   * Connect to NATS Server
   */
  async connect(): Promise<boolean> {
    try {
      console.log('🔌 Connecting to NATS Server...');
      // Mock connection for development - replace with actual NATS connection when available
      this.connection = {
        publish: (subject: string, data: Uint8Array) => {
          console.log(`📤 Mock publish to ${subject}:`, new TextDecoder().decode(data));
        },
        subscribe: (subject: string) => ({
          unsubscribe: () => console.log(`📥 Mock unsubscribe from ${subject}`),
          [Symbol.asyncIterator]: async function* () {
            // Mock async iterator - in real implementation this would yield actual messages
            yield { data: new TextEncoder().encode('{"type":"system.health","data":{},"timestamp":"' + new Date().toISOString() + '"}') };
          }
        }),
        request: async (subject: string, data: Uint8Array, options?: { timeout: number }) => {
          console.log(`📤 Mock request to ${subject}:`, new TextDecoder().decode(data));
          return { data: new TextEncoder().encode('{"type":"system.health","data":{"status":"ok"},"timestamp":"' + new Date().toISOString() + '"}') };
        },
        drain: async () => console.log('🔌 Mock drain'),
        closed: async () => null,
        isClosed: () => false,
        info: { server_name: 'mock-nats' }
      };

      console.log('✅ Connected to NATS Server (Mock)');
      this.connectedAt = Date.now();
      this.emit('connected');
      this.setupConnectionEvents();

      return true;
    } catch (error) {
      console.error('❌ Failed to connect to NATS Server:', error);
      this.emit('error', error);
      return false;
    }
  }

  /**
   * Disconnect from NATS Server
   */
  async disconnect(): Promise<void> {
    if (this.connection) {
      // Unsubscribe from all subjects
      for (const subscription of this.subscriptions.values()) {
        subscription.unsubscribe();
      }
      this.subscriptions.clear();
      this.messageHandlers.clear();

      await this.connection.drain();
      this.connection = null;
      console.log('🔌 Disconnected from NATS Server');
    }
  }

  /**
   * Publish a message to a subject
   */
  async publish(subject: string, data: any, headers?: Record<string, string>): Promise<void> {
    if (!this.connection) {
      throw new Error('Not connected to NATS Server');
    }

    const message: LegalAIMessage = {
      type: this.getMessageType(subject),
      data,
      timestamp: new Date().toISOString(),
      sessionId: this.generateSessionId(),
    };

    try {
      this.connection.publish(subject, this.jsonCodec.encode(message));
      this.publishedCount++;
      this.sampleSubject(subject, message);
      console.log(`📤 Published message to ${subject}:`, message);
      this.emit('publish', { subject, message });
    } catch (error) {
      console.error(`❌ Failed to publish to ${subject}:`, error);
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Subscribe to a subject with message handler
   */
  async subscribe(subject: string, handler: MessageHandler): Promise<void> {
    if (!this.connection) {
      throw new Error('Not connected to NATS Server');
    }

    try {
      // Create subscription if it doesn't exist
      if (!this.subscriptions.has(subject)) {
        const subscription = this.connection.subscribe(subject);
        this.subscriptions.set(subject, subscription);

        // Process messages
        this.processMessages(subscription, subject);
        console.log(`📥 Subscribed to ${subject}`);
      }

      // Add handler to the set
      if (!this.messageHandlers.has(subject)) {
        this.messageHandlers.set(subject, new Set());
      }
      this.messageHandlers.get(subject)!.add(handler);

    } catch (error) {
      console.error(`❌ Failed to subscribe to ${subject}:`, error);
      throw error;
    }
  }

  /**
   * Unsubscribe from a subject
   */
  unsubscribe(subject: string, handler?: MessageHandler): void {
    if (handler) {
      // Remove specific handler
      this.messageHandlers.get(subject)?.delete(handler);
    } else {
      // Remove all handlers and subscription
      const subscription = this.subscriptions.get(subject);
      if (subscription) {
        subscription.unsubscribe();
        this.subscriptions.delete(subject);
      }
      this.messageHandlers.delete(subject);
      console.log(`📥 Unsubscribed from ${subject}`);
    }
  }

  /**
   * Request-Reply pattern
   */
  async request(subject: string, data: any, timeout: number = 5000): Promise<LegalAIMessage> {
    if (!this.connection) {
      throw new Error('Not connected to NATS Server');
    }

    const requestMessage: LegalAIMessage = {
      type: this.getMessageType(subject),
      data,
      timestamp: new Date().toISOString(),
      sessionId: this.generateSessionId(),
    };

    try {
      const response = await this.connection.request(
        subject,
        this.jsonCodec.encode(requestMessage),
        { timeout }
      );

      return this.jsonCodec.decode(response.data) as LegalAIMessage;
    } catch (error) {
      console.error(`❌ Request to ${subject} failed:`, error);
      throw error;
    }
  }

  /**
   * Get connection status
   */
  isConnected(): boolean {
    return this.connection !== null && !this.connection.isClosed();
  }

  /**
   * Get connection info
   */
  getConnectionInfo(): any {
    return this.connection?.info;
  }

  /**
   * High-level API for Legal AI operations
   */

  // Case management
  async notifyCaseCreated(caseData: any): Promise<void> {
    await this.publish(this.subjects.CASE_CREATED, caseData);
  }

  async notifyCaseUpdated(caseId: string, updates: any): Promise<void> {
    await this.publish(this.subjects.CASE_UPDATED, { caseId, updates });
  }

  // Document processing
  async notifyDocumentUploaded(documentData: any): Promise<void> {
    await this.publish(this.subjects.DOCUMENT_UPLOADED, documentData);
  }

  async notifyDocumentProcessed(documentId: string, results: any): Promise<void> {
    await this.publish(this.subjects.DOCUMENT_PROCESSED, { documentId, results });
  }

  // AI analysis
  async notifyAIAnalysisCompleted(analysisId: string, results: any): Promise<void> {
    await this.publish(this.subjects.AI_ANALYSIS_COMPLETED, { analysisId, results });
  }

  // Real-time search
  async sendSearchQuery(query: string, filters?: any): Promise<any> {
    return await this.request(this.subjects.SEARCH_QUERY, { query, filters });
  }

  // Chat messaging
  async sendChatMessage(message: string, caseId?: string): Promise<void> {
    await this.publish(this.subjects.CHAT_MESSAGE, { message, caseId });
  }

  // System health monitoring
  async publishSystemHealth(healthData: any): Promise<void> {
    await this.publish(this.subjects.SYSTEM_HEALTH, healthData);
  }

  // Subscription helpers
  async subscribeToCase(caseId: string, handler: MessageHandler): Promise<void> {
    await this.subscribe(`legal.case.${caseId}.>`, handler);
  }

  async subscribeToDocument(documentId: string, handler: MessageHandler): Promise<void> {
    await this.subscribe(`legal.document.${documentId}.>`, handler);
  }

  async subscribeToAIAnalysis(handler: MessageHandler): Promise<void> {
    await this.subscribe('legal.ai.>', handler);
  }

  async subscribeToSystemEvents(handler: MessageHandler): Promise<void> {
    await this.subscribe('system.>', handler);
  }

  /**
   * Private helper methods
   */

  private setupConnectionEvents(): void {
    if (!this.connection) return;

    this.connection.closed().then((error) => {
      if (error) {
        console.error('🔌 NATS connection closed with error:', error);
      } else {
        console.log('🔌 NATS connection closed gracefully');
      }
    });

    // Handle reconnection
    // Note: nats.ws handles reconnection automatically
  }

  private async processMessages(subscription: NATSSubscription, subject: string): Promise<void> {
    for await (const msg of subscription) {
      try {
        const message = this.jsonCodec.decode(msg.data) as LegalAIMessage;
        console.log(`📨 Received message on ${subject}:`, message);

        // Call all handlers for this subject
        const handlers = this.messageHandlers.get(subject);
        if (handlers) {
          for (const handler of handlers) {
            try {
              handler(message);
            } catch (error) {
              console.error('❌ Error in message handler:', error);
            }
          }
        }
      } catch (error) {
        console.error(`❌ Error processing message on ${subject}:`, error);
      }
    }
  }

  private getMessageType(subject: string): LegalAIMessage['type'] {
    if (subject.includes('case.created')) return 'case.created';
    if (subject.includes('document.uploaded')) return 'document.uploaded';
    if (subject.includes('ai.analysis.completed')) return 'ai.analysis.completed';
    if (subject.includes('search.query')) return 'search.query';
    if (subject.includes('chat.message')) return 'chat.message';
    if (subject.includes('system.health')) return 'system.health';
    return 'system.health'; // default
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2)}`;
  }

  private sampleSubject(subject: string, payload: any) {
    const arr = this.subjectSamples.get(subject) || [];
    const hash = this.hashPayload(payload);
    if (!arr.includes(hash)) {
      if (arr.length < 5) arr.push(hash); else arr[Math.floor(Math.random() * arr.length)] = hash;
      this.subjectSamples.set(subject, arr);
    }
  }

  private hashPayload(obj: any): string {
    try { const s = typeof obj === 'string' ? obj : JSON.stringify(obj); let h = 0; for (let i = 0; i < s.length; i++) { h = (h << 5) - h + s.charCodeAt(i); h |= 0; } return h.toString(16); } catch { return '0'; }
  }

  getMetrics(): NATSMetricsSnapshot {
    return {
      connection: { status: this.isConnected() ? 'connected' : 'disconnected', since: this.connectedAt, reconnectAttempts: this.reconnectAttempts },
      messaging: { published: this.publishedCount, received: this.receivedCount, subjects: Object.fromEntries([...this.subjectSamples.entries()]) }
    };
  }
}

// Singleton instance for global use
export const natsMessaging = new NATSMessagingService();

// Auto-connect when module is imported (in browser environment)
if (typeof window !== 'undefined') {
  natsMessaging.connect().catch(error => {
    console.warn('NATS auto-connect failed:', error);
  });
}