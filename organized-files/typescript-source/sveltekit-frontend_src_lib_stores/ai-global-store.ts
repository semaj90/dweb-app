// XState AI Global Store with Loki.js + IndexedDB + Fuse.js
// Comprehensive AI assistant state management for Legal AI platform

import { createMachine, interpret, type ActorRefFrom } from 'xstate';
import Loki from 'lokijs';
import Fuse from 'fuse.js';
import { browser } from '$app/environment';
import { get, writable, type Writable } from 'svelte/store';
import type { User } from '$lib/server/db/schema-unified-postgres';

// Types for AI interactions
export interface AIMessage {
  id: string;
  sessionId: string;
  userId: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  metadata: {
    model?: string;
    confidence?: number;
    tokens?: number;
    processingTime?: number;
    context?: string[];
    entities?: Array<{
      type: string;
      text: string;
      confidence: number;
    }>;
    embedding?: number[];
    relevantDocs?: string[];
    [key: string]: any;
  };
}

export interface AISession {
  id: string;
  userId: string;
  title: string;
  context: 'legal-research' | 'case-analysis' | 'document-review' | 'general';
  status: 'active' | 'archived' | 'deleted';
  messageCount: number;
  lastMessage?: string;
  createdAt: number;
  updatedAt: number;
  metadata: {
    caseId?: string;
    documentIds?: string[];
    tags?: string[];
    priority?: 'low' | 'medium' | 'high';
    [key: string]: any;
  };
}

export interface AIRecommendation {
  id: string;
  type: 'query-suggestion' | 'document-recommendation' | 'case-insight' | 'workflow-tip';
  title: string;
  description: string;
  actionUrl?: string;
  confidence: number;
  relevance: number;
  createdAt: number;
  metadata: {
    context?: string;
    relatedEntities?: string[];
    requiredActions?: string[];
    [key: string]: any;
  };
}

export interface AIContext {
  user: User | null;
  currentSession: AISession | null;
  activeMessages: AIMessage[];
  recommendations: AIRecommendation[];
  searchHistory: Array<{
    query: string;
    timestamp: number;
    results: number;
    context: string;
  }>;
  settings: {
    model: string;
    temperature: number;
    maxTokens: number;
    streaming: boolean;
    persistHistory: boolean;
    enableRecommendations: boolean;
    autoSave: boolean;
  };
  cache: {
    embeddings: Map<string, number[]>;
    responses: Map<string, string>;
    lastSync: number;
  };
  performance: {
    avgResponseTime: number;
    totalQueries: number;
    cacheHitRate: number;
  };
}

// XState machine for AI assistant
const aiAssistantMachine = createMachine({
  /** @xstate-layout N4IgpgJg5mDOIC5QBkD2A7KAxAZgQwBtUBjAW1QFkIAiWgbQAYBdRUAByqyUQW5Tm4eIUwFnhCuAC2yAAcSc2RGSgAHogDMAFgAcARgCMAJjWKAbGsV60ADnWLNAbTa7diAGKI30rKDm4eISG65vZGFLj4BKQexJgEqAB2sBIMqACOMkF4QWGR0XbRdgnJFmkZhtnZeVGZiAC++XlNRWVNNaVd1Y00LTUVLW0d3f19gwOD0qNYOFNykzNziExRQSHhxJFxcQlJKWkZOzk7+YVFxZ2l5VXVdTXFJS1tDJ2dPX2Dg8MjJvhNqBIABMqT2BBRpFmODm0H2ByS+wMRROxxhFxONzSKCJd1JDzeyMe4je7wGbwGP16IFGpLG02JUzJm3JYNpB0OdKOzMhDOZjyAA */
  types: {
    context: {} as AIContext,
    events: {} as
      | { type: 'START_SESSION'; sessionId?: string; context?: string }
      | { type: 'END_SESSION' }
      | { type: 'SEND_MESSAGE'; content: string; metadata?: Record<string, any> }
      | { type: 'RECEIVE_MESSAGE'; content: string; metadata?: Record<string, any> }
      | { type: 'UPDATE_SETTINGS'; settings: Partial<AIContext['settings']> }
      | { type: 'SYNC_DATA' }
      | { type: 'SEARCH'; query: string; context?: string }
      | { type: 'GET_RECOMMENDATIONS' }
      | { type: 'CLEAR_HISTORY' }
      | { type: 'CACHE_RESPONSE'; key: string; value: string }
      | { type: 'ERROR'; error: string }
  },
  id: 'aiAssistant',
  initial: 'idle',
  context: {
    user: null,
    currentSession: null,
    activeMessages: [],
    recommendations: [],
    searchHistory: [],
    settings: {
      model: 'llama3-legal',
      temperature: 0.7,
      maxTokens: 2048,
      streaming: true,
      persistHistory: true,
      enableRecommendations: true,
      autoSave: true,
    },
    cache: {
      embeddings: new Map(),
      responses: new Map(),
      lastSync: Date.now(),
    },
    performance: {
      avgResponseTime: 0,
      totalQueries: 0,
      cacheHitRate: 0,
    },
  },
  states: {
    idle: {
      on: {
        START_SESSION: 'sessionActive',
        SYNC_DATA: 'syncing',
        UPDATE_SETTINGS: { actions: 'updateSettings' },
      },
    },
    sessionActive: {
      entry: 'startSession',
      on: {
        END_SESSION: 'idle',
        SEND_MESSAGE: 'processing',
        SEARCH: 'searching',
        GET_RECOMMENDATIONS: 'generating',
        SYNC_DATA: 'syncing',
        UPDATE_SETTINGS: { actions: 'updateSettings' },
        CLEAR_HISTORY: { actions: 'clearHistory' },
      },
    },
    processing: {
      entry: 'sendMessage',
      on: {
        RECEIVE_MESSAGE: 'sessionActive',
        ERROR: 'error',
      },
    },
    searching: {
      entry: 'performSearch',
      on: {
        SEARCH_COMPLETE: 'sessionActive',
        ERROR: 'error',
      },
    },
    generating: {
      entry: 'generateRecommendations',
      on: {
        RECOMMENDATIONS_READY: 'sessionActive',
        ERROR: 'error',
      },
    },
    syncing: {
      entry: 'syncData',
      on: {
        SYNC_COMPLETE: 'idle',
        ERROR: 'error',
      },
    },
    error: {
      on: {
        START_SESSION: 'sessionActive',
        SYNC_DATA: 'syncing',
      },
    },
  },
}).provide(/* config */) || machine.withConfig?.({
  actions: {
    startSession: (context, event) => {
      // Implementation in aiService
    },
    sendMessage: (context, event) => {
      // Implementation in aiService
    },
    updateSettings: (context, event) => {
      if (event.type === 'UPDATE_SETTINGS') {
        context.settings = { ...context.settings, ...event.settings };
      }
    },
    clearHistory: (context) => {
      context.activeMessages = [];
      context.searchHistory = [];
    },
    performSearch: (context, event) => {
      // Implementation in aiService
    },
    generateRecommendations: (context) => {
      // Implementation in aiService
    },
    syncData: (context) => {
      // Implementation in aiService
    },
  },
});

// Loki.js database for local storage
class AIDataManager {
  private db: Loki | null = null;
  private messages: Collection<AIMessage> | null = null;
  private sessions: Collection<AISession> | null = null;
  private recommendations: Collection<AIRecommendation> | null = null;
  private fuse: Fuse<AIMessage> | null = null;
  
  async initialize() {
    if (!browser) return;

    this.db = new Loki('legal-ai-assistant.db', {
      autoload: true,
      autoloadCallback: this.onDatabaseLoad.bind(this),
      autosave: true,
      autosaveInterval: 10000, // 10 seconds
      persistenceMethod: 'indexeddb',
      persistenceOptions: {
        autoSave: true,
        autoSaveCallback: () => console.log('💾 AI data saved to IndexedDB'),
      },
    });
  }

  private onDatabaseLoad() {
    // Get or create collections
    this.messages = this.db!.getCollection('messages') || 
                   this.db!.addCollection('messages', { 
                     indices: ['sessionId', 'userId', 'timestamp', 'role'],
                     unique: ['id']
                   });

    this.sessions = this.db!.getCollection('sessions') || 
                   this.db!.addCollection('sessions', {
                     indices: ['userId', 'status', 'updatedAt'],
                     unique: ['id']
                   });

    this.recommendations = this.db!.getCollection('recommendations') || 
                         this.db!.addCollection('recommendations', {
                           indices: ['type', 'confidence', 'createdAt'],
                           unique: ['id']
                         });

    // Initialize Fuse.js for fuzzy search
    this.initializeFuzzySearch();
    
    console.log('🗄️ AI database loaded with', {
      messages: this.messages.count(),
      sessions: this.sessions.count(),
      recommendations: this.recommendations.count(),
    });
  }

  private initializeFuzzySearch() {
    if (!this.messages) return;

    const allMessages = this.messages.find();
    this.fuse = new Fuse(allMessages, {
      keys: [
        { name: 'content', weight: 0.7 },
        { name: 'metadata.entities.text', weight: 0.2 },
        { name: 'metadata.context', weight: 0.1 },
      ],
      threshold: 0.6,
      distance: 1000,
      includeScore: true,
      includeMatches: true,
    });
  }

  // Message operations
  addMessage(message: Omit<AIMessage, 'id'>): AIMessage {
    if (!this.messages) throw new Error('Database not initialized');

    const fullMessage: AIMessage = {
      id: crypto.randomUUID(),
      ...message,
    };

    this.messages.insert(fullMessage);
    this.updateFuzzySearch();
    
    return fullMessage;
  }

  getMessages(sessionId: string): AIMessage[] {
    if (!this.messages) return [];
    return this.messages.find({ sessionId }).sort((a, b) => a.timestamp - b.timestamp);
  }

  searchMessages(query: string, limit = 10): Array<{ item: AIMessage; score: number }> {
    if (!this.fuse || !query.trim()) return [];

    return this.fuse.search(query).slice(0, 20).map(result => ({
      item: result.item,
      score: result.score || 0,
    }));
  }

  // Session operations
  createSession(session: Omit<AISession, 'id'>): AISession {
    if (!this.sessions) throw new Error('Database not initialized');

    const fullSession: AISession = {
      id: crypto.randomUUID(),
      ...session,
    };

    this.sessions.insert(fullSession);
    return fullSession;
  }

  updateSession(sessionId: string, updates: Partial<AISession>): AISession | null {
    if (!this.sessions) return null;

    const session = this.sessions.findOne({ id: sessionId });
    if (!session) return null;

    Object.assign(session, updates, { updatedAt: Date.now() });
    this.sessions.update(session);
    return session;
  }

  getSessions(userId: string, status: AISession['status'] = 'active'): AISession[] {
    if (!this.sessions) return [];
    return this.sessions.find({ userId, status }).sort((a, b) => b.updatedAt - a.updatedAt);
  }

  // Recommendation operations
  addRecommendation(recommendation: Omit<AIRecommendation, 'id'>): AIRecommendation {
    if (!this.recommendations) throw new Error('Database not initialized');

    const fullRecommendation: AIRecommendation = {
      id: crypto.randomUUID(),
      ...recommendation,
    };

    this.recommendations.insert(fullRecommendation);
    return fullRecommendation;
  }

  getRecommendations(limit = 5): AIRecommendation[] {
    if (!this.recommendations) return [];
    return this.recommendations
      .find()
      .sort((a, b) => b.relevance - a.relevance)
      .slice(0, limit);
  }

  // Cache operations
  getCacheStats() {
    return {
      messages: this.messages?.count() || 0,
      sessions: this.sessions?.count() || 0,
      recommendations: this.recommendations?.count() || 0,
      dbSize: (this.db as any)?.serialize?.()?.length || 0 || 0,
    };
  }

  clearData() {
    this.messages?.clear();
    this.sessions?.clear();
    this.recommendations?.clear();
    this.initializeFuzzySearch();
  }

  private updateFuzzySearch() {
    if (!this.messages) return;
    
    const allMessages = this.messages.find();
    this.fuse = new Fuse(allMessages, {
      keys: [
        { name: 'content', weight: 0.7 },
        { name: 'metadata.entities.text', weight: 0.2 },
        { name: 'metadata.context', weight: 0.1 },
      ],
      threshold: 0.6,
      includeScore: true,
      includeMatches: true,
    });
  }

  // Export/Import for backup
  exportData() {
    return {
      messages: this.messages?.find() || [],
      sessions: this.sessions?.find() || [],
      recommendations: this.recommendations?.find() || [],
      timestamp: Date.now(),
    };
  }

  importData(data: any) {
    if (data.messages) {
      this.messages?.clear();
      data.messages.forEach((msg: AIMessage) => this.messages?.insert(msg));
    }
    
    if (data.sessions) {
      this.sessions?.clear();
      data.sessions.forEach((session: AISession) => this.sessions?.insert(session));
    }
    
    if (data.recommendations) {
      this.recommendations?.clear();
      data.recommendations.forEach((rec: AIRecommendation) => this.recommendations?.insert(rec));
    }

    this.initializeFuzzySearch();
  }
}

// Global AI service
export class AIGlobalService {
  private machine: ActorRefFrom<typeof aiAssistantMachine>;
  private dataManager: AIDataManager;
  private eventStream: EventSource | null = null;

  // Svelte stores
  public state: Writable<any>;
  public context: Writable<AIContext>;
  public isProcessing: Writable<boolean>;
  public recommendations: Writable<AIRecommendation[]>;

  constructor() {
    this.dataManager = new AIDataManager();
    this.machine = createActor(aiAssistantMachine).start();
    
    // Initialize stores
    this.state = writable(this.machine.getSnapshot().value);
    this.context = writable(this.machine.getSnapshot().context);
    this.isProcessing = writable(false);
    this.recommendations = writable([]);

    // Subscribe to machine changes
    this.machine.subscribe(snapshot => {
      this.state.set(snapshot.value);
      this.context.set(snapshot.context);
      this.isProcessing.set(snapshot.matches('processing') || snapshot.matches('searching'));
    });
  }

  async initialize() {
    await this.dataManager.initialize();
    this.machine.start();
    this.startEventStream();
    console.log('🤖 AI Global Service initialized');
  }

  // Session management
  async startSession(context?: string, caseId?: string) {
    const sessionId = crypto.randomUUID();
    const userId = get(this.context).user?.id || 'anonymous';

    const session = this.dataManager.createSession({
      userId,
      title: `Session ${new Date().toLocaleTimeString()}`,
      context: (context as AISession['context']) || 'general',
      status: 'active',
      messageCount: 0,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      metadata: { caseId },
    });

    this.machine.send({ type: 'START_SESSION', sessionId: session.id, context });
    return session;
  }

  async endSession() {
    const currentContext = get(this.context);
    if (currentContext.currentSession) {
      this.dataManager.updateSession(currentContext.currentSession.id, { 
        status: 'archived',
        updatedAt: Date.now()
      });
    }
    this.machine.send({ type: 'END_SESSION' });
  }

  // Message handling
  async sendMessage(content: string, metadata: Record<string, any> = {}) {
    const currentContext = get(this.context);
    if (!currentContext.currentSession) {
      await this.startSession();
    }

    const userMessage = this.dataManager.addMessage({
      sessionId: currentContext.currentSession!.id,
      userId: currentContext.user?.id || 'anonymous',
      role: 'user',
      content,
      timestamp: Date.now(),
      metadata,
    });

    this.machine.send({ type: 'SEND_MESSAGE', content, metadata });

    // Call AI service
    try {
      const response = await this.callAIService(content, metadata);
      this.receiveMessage(response.content, response.metadata);
    } catch (error) {
      console.error('AI service error:', error);
      this.machine.send({ type: 'ERROR', error: String(error) });
    }
  }

  private receiveMessage(content: string, metadata: Record<string, any> = {}) {
    const currentContext = get(this.context);
    if (!currentContext.currentSession) return;

    const assistantMessage = this.dataManager.addMessage({
      sessionId: currentContext.currentSession.id,
      userId: currentContext.user?.id || 'anonymous',
      role: 'assistant',
      content,
      timestamp: Date.now(),
      metadata,
    });

    this.machine.send({ type: 'RECEIVE_MESSAGE', content, metadata });
  }

  // Search functionality
  async searchMessages(query: string): Promise<Array<{ item: AIMessage; score: number }>> {
    return this.dataManager.searchMessages(query);
  }

  async searchSemantic(query: string, context?: string): Promise<any> {
    this.machine.send({ type: 'SEARCH', query, context });

    try {
      const response = await fetch('/api/ai/semantic-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, context }),
      });

      if (!response.ok) throw new Error('Search failed');
      return await response.json();
    } catch (error) {
      console.error('Semantic search error:', error);
      this.machine.send({ type: 'ERROR', error: String(error) });
      return { results: [], total: 0 };
    }
  }

  // Recommendations
  async generateRecommendations(): Promise<AIRecommendation[]> {
    this.machine.send({ type: 'GET_RECOMMENDATIONS' });

    try {
      const response = await fetch('/api/ai/recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          context: get(this.context),
          sessionHistory: this.dataManager.getMessages(get(this.context).currentSession?.id || '')
        }),
      });

      if (!response.ok) throw new Error('Recommendations failed');
      const recommendations = await response.json();

      // Store recommendations locally
      recommendations.forEach((rec: Omit<AIRecommendation, 'id'>) => {
        this.dataManager.addRecommendation(rec);
      });

      this.recommendations.set(recommendations);
      return recommendations;
    } catch (error) {
      console.error('Recommendations error:', error);
      return [];
    }
  }

  // Settings
  updateSettings(settings: Partial<AIContext['settings']>) {
    this.machine.send({ type: 'UPDATE_SETTINGS', settings });
  }

  // Data management
  async syncData() {
    this.machine.send({ type: 'SYNC_DATA' });
    
    try {
      // Sync with server if needed
      await this.uploadLocalData();
      await this.downloadServerData();
    } catch (error) {
      console.error('Sync error:', error);
    }
  }

  clearHistory() {
    this.dataManager.clearData();
    this.machine.send({ type: 'CLEAR_HISTORY' });
  }

  exportData() {
    return this.dataManager.exportData();
  }

  importData(data: any) {
    this.dataManager.importData(data);
  }

  // Private methods
  private async callAIService(content: string, metadata: Record<string, any>) {
    const currentContext = get(this.context);
    
    const payload = {
      message: content,
      session_id: currentContext.currentSession?.id,
      user_id: currentContext.user?.id,
      model: currentContext.settings.model,
      temperature: currentContext.settings.temperature,
      max_tokens: currentContext.settings.maxTokens,
      metadata,
    };

    const response = await fetch('/api/ai/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error(`AI service error: ${response.status}`);
    return await response.json();
  }

  private startEventStream() {
    if (!browser || this.eventStream) return;

    this.eventStream = new EventSource('/api/ai/events');
    
    this.eventStream.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleServerEvent(data);
      } catch (error) {
        console.error('Event stream parsing error:', error);
      }
    };

    this.eventStream.onerror = () => {
      console.warn('Event stream disconnected, reconnecting...');
      setTimeout(() => this.startEventStream(), 5000);
    };
  }

  private handleServerEvent(data: any) {
    switch (data.type) {
      case 'message':
        this.receiveMessage(data.content, data.metadata);
        break;
      case 'recommendation':
        this.dataManager.addRecommendation(data.recommendation);
        break;
      case 'sync':
        this.syncData();
        break;
      default:
        console.log('Unknown server event:', data);
    }
  }

  private async uploadLocalData() {
    // Upload local changes to server
    const data = this.dataManager.exportData();
    // Implementation depends on your API
  }

  private async downloadServerData() {
    // Download updates from server
    // Implementation depends on your API
  }

  // Cleanup
  destroy() {
    this.machine.stop();
    if (this.eventStream) {
      this.eventStream.close();
      this.eventStream = null;
    }
  }
}

// Global instance
export const aiService = new AIGlobalService();

// Initialize on browser load
if (browser) {
  aiService.initialize();
}

// Export stores for Svelte components
export const aiState = aiService.state;
export const aiContext = aiService.context;
export const aiProcessing = aiService.isProcessing;
export const aiRecommendations = aiService.recommendations;