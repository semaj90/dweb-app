/**
 * Global AI Assistant Store with XState + Loki.js + IndexedDB Integration
 * Svelte 5 compatible with $state runes and reactive patterns
 */

import { writable, derived, get } from 'svelte/store';
import { createActor, assign, setup, fromPromise } from 'xstate';
import Loki from 'lokijs';
import LokiIndexedAdapter from 'lokijs/src/loki-indexed-adapter';
import Fuse from 'fuse.js';
import { browser } from '$app/environment';

// Types for legal AI interactions
interface AIMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  sessionId: string;
  caseId?: string;
  documentId?: string;
  metadata?: {
    model?: string;
    confidence?: number;
    processingTime?: number;
    cached?: boolean;
    tokens?: number;
    embeddings?: number[];
    entities?: LegalEntity[];
  };
}

interface LegalEntity {
  name: string;
  type: 'case_party' | 'statute' | 'citation' | 'jurisdiction' | 'judge' | 'contract_clause';
  confidence: number;
  startOffset?: number;
  endOffset?: number;
  metadata?: Record<string, any>;
}

interface AISession {
  id: string;
  title: string;
  type: 'chat' | 'analysis' | 'research' | 'drafting';
  caseId?: string;
  createdAt: number;
  lastActiveAt: number;
  messageCount: number;
  metadata: Record<string, any>;
}

interface DocumentAnalysis {
  documentId: string;
  analysisType: 'summary' | 'risks' | 'clauses' | 'compliance';
  result: string;
  confidence: number;
  entities: LegalEntity[];
  timestamp: number;
  caseId?: string;
}

interface UserPreferences {
  preferredModel: string;
  temperature: number;
  maxTokens: number;
  enableEntityExtraction: boolean;
  enableCaching: boolean;
  autoSaveInterval: number;
  theme: 'light' | 'dark' | 'yorha';
}

// Loki.js Database Manager with IndexedDB persistence
class AIDataManager {
  private db: Loki;
  private messages: Collection<AIMessage>;
  private sessions: Collection<AISession>;
  private analyses: Collection<DocumentAnalysis>;
  private preferences: Collection<UserPreferences>;
  private initialized = false;

  constructor() {
    if (!browser) return;

    const adapter = new LokiIndexedAdapter('yorha-ai-db');
    this.db = new Loki('yorha-ai.db', {
      adapter,
      autoload: true,
      autoloadCallback: () => this.initializeCollections(),
      autosave: true,
      autosaveInterval: 30000,
      persistenceMethod: 'adapter'
    });
  }

  private initializeCollections(): void {
    // Messages collection with indexes for fast search
    this.messages = this.db.getCollection('messages') || 
      this.db.addCollection('messages', {
        indices: ['sessionId', 'caseId', 'timestamp', 'role'],
        unique: ['id']
      });

    // Sessions collection
    this.sessions = this.db.getCollection('sessions') || 
      this.db.addCollection('sessions', {
        indices: ['caseId', 'lastActiveAt', 'type'],
        unique: ['id']
      });

    // Document analyses
    this.analyses = this.db.getCollection('analyses') || 
      this.db.addCollection('analyses', {
        indices: ['documentId', 'caseId', 'analysisType', 'timestamp'],
        unique: ['documentId', 'analysisType']
      });

    // User preferences
    this.preferences = this.db.getCollection('preferences') || 
      this.db.addCollection('preferences');

    this.initialized = true;
    console.log('✅ AI Data Manager initialized with IndexedDB persistence');
  }

  // Message operations
  addMessage(message: AIMessage): void {
    if (!this.initialized) return;
    this.messages.insert({ ...message });
  }

  getMessages(sessionId: string): AIMessage[] {
    if (!this.initialized) return [];
    return this.messages.find({ sessionId }).sort((a, b) => a.timestamp - b.timestamp);
  }

  searchMessages(query: string, options: { caseId?: string; limit?: number } = {}): AIMessage[] {
    if (!this.initialized) return [];

    let messages = this.messages.find();
    
    if (options.caseId) {
      messages = messages.filter(m => m.caseId === options.caseId);
    }

    // Use Fuse.js for fuzzy search
    const fuse = new Fuse(messages, {
      keys: ['content', 'metadata.entities.name'],
      threshold: 0.3,
      includeScore: true
    });

    const results = fuse.search(query).slice(0, 20);
    return results.map(r => r.item);
  }

  // Session operations
  createSession(session: Omit<AISession, 'id' | 'createdAt' | 'lastActiveAt' | 'messageCount'>): AISession {
    if (!this.initialized) throw new Error('Database not initialized');

    const newSession: AISession = {
      ...session,
      id: `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: Date.now(),
      lastActiveAt: Date.now(),
      messageCount: 0
    };

    this.sessions.insert(newSession);
    return newSession;
  }

  getRecentSessions(limit: number = 10): AISession[] {
    if (!this.initialized) return [];
    return this.sessions.chain()
      .simplesort('lastActiveAt', true)
      .limit(limit)
      .data();
  }

  updateSession(sessionId: string, updates: Partial<AISession>): void {
    if (!this.initialized) return;
    const session = this.sessions.findOne({ id: sessionId });
    if (session) {
      Object.assign(session, updates, { lastActiveAt: Date.now() });
      this.sessions.update(session);
    }
  }

  // Analysis operations
  saveAnalysis(analysis: DocumentAnalysis): void {
    if (!this.initialized) return;

    const existing = this.analyses.findOne({ 
      documentId: analysis.documentId, 
      analysisType: analysis.analysisType 
    });

    if (existing) {
      Object.assign(existing, analysis);
      this.analyses.update(existing);
    } else {
      this.analyses.insert(analysis);
    }
  }

  getAnalysis(documentId: string, analysisType: string): DocumentAnalysis | null {
    if (!this.initialized) return null;
    return this.analyses.findOne({ documentId, analysisType });
  }

  getCaseAnalyses(caseId: string): DocumentAnalysis[] {
    if (!this.initialized) return [];
    return this.analyses.find({ caseId });
  }

  // Preferences
  getPreferences(): UserPreferences {
    if (!this.initialized) {
      return this.getDefaultPreferences();
    }

    const prefs = this.preferences.findOne({});
    return prefs || this.getDefaultPreferences();
  }

  updatePreferences(updates: Partial<UserPreferences>): void {
    if (!this.initialized) return;

    let prefs = this.preferences.findOne({});
    if (!prefs) {
      prefs = { ...this.getDefaultPreferences(), ...updates };
      this.preferences.insert(prefs);
    } else {
      Object.assign(prefs, updates);
      this.preferences.update(prefs);
    }
  }

  private getDefaultPreferences(): UserPreferences {
    return {
      preferredModel: 'gemma3-legal',
      temperature: 0.7,
      maxTokens: 500,
      enableEntityExtraction: true,
      enableCaching: true,
      autoSaveInterval: 30000,
      theme: 'yorha'
    };
  }

  // Cleanup and maintenance
  cleanupOldData(olderThanDays: number = 30): void {
    if (!this.initialized) return;

    const cutoffTime = Date.now() - (olderThanDays * 24 * 60 * 60 * 1000);
    
    // Remove old messages
    this.messages.removeWhere({ timestamp: { '$lt': cutoffTime } });
    
    // Remove inactive sessions
    this.sessions.removeWhere({ lastActiveAt: { '$lt': cutoffTime } });
    
    // Remove old analyses (keep for longer)
    const analysisCutoff = Date.now() - (90 * 24 * 60 * 60 * 1000); // 90 days
    this.analyses.removeWhere({ timestamp: { '$lt': analysisCutoff } });

    console.log(`Cleaned up AI data older than ${olderThanDays} days`);
  }

  getStats() {
    if (!this.initialized) return null;

    return {
      messages: this.messages.count(),
      sessions: this.sessions.count(),
      analyses: this.analyses.count(),
      lastCleanup: Date.now()
    };
  }
}

// XState machine for AI interactions
const aiMachine = setup({
  types: {
    context: {} as {
      currentSession: AISession | null;
      activeMessages: AIMessage[];
      isProcessing: boolean;
      error: string | null;
      preferences: UserPreferences;
      recentSessions: AISession[];
      searchResults: AIMessage[];
      currentAnalysis: DocumentAnalysis | null;
    },
    events: {} as
      | { type: 'CREATE_SESSION'; sessionData: Omit<AISession, 'id' | 'createdAt' | 'lastActiveAt' | 'messageCount'> }
      | { type: 'LOAD_SESSION'; sessionId: string }
      | { type: 'SEND_MESSAGE'; message: Omit<AIMessage, 'id' | 'timestamp'> }
      | { type: 'MESSAGE_RECEIVED'; message: AIMessage }
      | { type: 'SEARCH_MESSAGES'; query: string; options?: any }
      | { type: 'ANALYZE_DOCUMENT'; documentId: string; analysisType: string; content: string }
      | { type: 'ANALYSIS_COMPLETE'; analysis: DocumentAnalysis }
      | { type: 'UPDATE_PREFERENCES'; preferences: Partial<UserPreferences> }
      | { type: 'ERROR'; error: string }
      | { type: 'RETRY' }
      | { type: 'CLEAR_ERROR' }
  },
  actors: {
    sendMessage: fromPromise(async ({ input }: { input: { message: AIMessage; preferences: UserPreferences } }) => {
      const { message, preferences } = input;
      
      // Call unified inference API
      const response = await fetch('/api/ai/inference-pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'generate',
          data: {
            prompt: message.content,
            model: preferences.preferredModel,
            temperature: preferences.temperature,
            maxTokens: preferences.maxTokens
          },
          options: {
            useCache: preferences.enableCaching,
            caseId: message.caseId,
            userId: 'current-user' // TODO: Get from auth
          }
        })
      });

      if (!response.ok) {
        throw new Error(`AI request failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      return {
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        role: 'assistant' as const,
        content: result.data?.text || result.text || 'No response generated',
        timestamp: Date.now(),
        sessionId: message.sessionId,
        caseId: message.caseId,
        documentId: message.documentId,
        metadata: {
          model: preferences.preferredModel,
          confidence: result.confidence || 0.85,
          processingTime: result.processingTime || 0,
          cached: result.cached || false,
          tokens: result.data?.tokensUsed || 0
        }
      };
    }),

    analyzeDocument: fromPromise(async ({ input }: { input: { documentId: string; analysisType: string; content: string; caseId?: string } }) => {
      const { documentId, analysisType, content, caseId } = input;

      const response = await fetch('/api/ai/inference-pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'legal_analysis',
          data: {
            documentId,
            analysisType,
            content,
            documentType: 'legal_document'
          },
          options: {
            useCache: true,
            caseId,
            userId: 'current-user'
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();
      const analysis = result.data?.analysis || result.analysis;

      return {
        documentId,
        analysisType,
        result: analysis.content || analysis.result || 'Analysis completed',
        confidence: analysis.confidence || 0.85,
        entities: analysis.extractedEntities || [],
        timestamp: Date.now(),
        caseId
      } as DocumentAnalysis;
    })
  }
}).createMachine({
  id: 'aiGlobal',
  initial: 'idle',
  context: {
    currentSession: null,
    activeMessages: [],
    isProcessing: false,
    error: null,
    preferences: {
      preferredModel: 'gemma3-legal',
      temperature: 0.7,
      maxTokens: 500,
      enableEntityExtraction: true,
      enableCaching: true,
      autoSaveInterval: 30000,
      theme: 'yorha'
    },
    recentSessions: [],
    searchResults: [],
    currentAnalysis: null
  },
  states: {
    idle: {
      on: {
        CREATE_SESSION: {
          target: 'active',
          actions: assign(({ context, event }) => {
            const newSession = aiDataManager.createSession(event.sessionData);
            return {
              ...context,
              currentSession: newSession,
              activeMessages: [],
              error: null
            };
          })
        },
        LOAD_SESSION: {
          target: 'active',
          actions: assign(({ context, event }) => {
            const messages = aiDataManager.getMessages(event.sessionId);
            const session = aiDataManager.sessions?.findOne({ id: event.sessionId }) || null;
            return {
              ...context,
              currentSession: session,
              activeMessages: messages,
              error: null
            };
          })
        },
        SEARCH_MESSAGES: {
          actions: assign(({ context, event }) => ({
            ...context,
            searchResults: aiDataManager.searchMessages(event.query, event.options)
          }))
        }
      }
    },
    active: {
      on: {
        SEND_MESSAGE: {
          target: 'processing',
          actions: assign(({ context, event }) => {
            const userMessage: AIMessage = {
              ...event.message,
              id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              timestamp: Date.now(),
              role: 'user'
            };
            
            aiDataManager.addMessage(userMessage);
            
            return {
              ...context,
              activeMessages: [...context.activeMessages, userMessage],
              isProcessing: true,
              error: null
            };
          })
        },
        ANALYZE_DOCUMENT: {
          target: 'analyzing',
          actions: assign(({ context }) => ({
            ...context,
            isProcessing: true,
            error: null
          }))
        }
      }
    },
    processing: {
      invoke: {
        src: 'sendMessage',
        input: ({ context }) => ({
          message: context.activeMessages[context.activeMessages.length - 1],
          preferences: context.preferences
        }),
        onDone: {
          target: 'active',
          actions: assign(({ context, event }) => {
            const assistantMessage = event.output;
            aiDataManager.addMessage(assistantMessage);
            
            // Update session message count
            if (context.currentSession) {
              aiDataManager.updateSession(context.currentSession.id, {
                messageCount: context.currentSession.messageCount + 2 // user + assistant
              });
            }

            return {
              ...context,
              activeMessages: [...context.activeMessages, assistantMessage],
              isProcessing: false
            };
          })
        },
        onError: {
          target: 'active',
          actions: assign(({ context, event }) => ({
            ...context,
            isProcessing: false,
            error: event.error.message
          }))
        }
      }
    },
    analyzing: {
      invoke: {
        src: 'analyzeDocument',
        input: ({ event }) => event.type === 'ANALYZE_DOCUMENT' ? {
          documentId: event.documentId,
          analysisType: event.analysisType,
          content: event.content
        } : {},
        onDone: {
          target: 'active',
          actions: assign(({ context, event }) => {
            const analysis = event.output;
            aiDataManager.saveAnalysis(analysis);

            return {
              ...context,
              currentAnalysis: analysis,
              isProcessing: false
            };
          })
        },
        onError: {
          target: 'active',
          actions: assign(({ context, event }) => ({
            ...context,
            isProcessing: false,
            error: event.error.message
          }))
        }
      }
    }
  },
  on: {
    UPDATE_PREFERENCES: {
      actions: assign(({ context, event }) => {
        const newPreferences = { ...context.preferences, ...event.preferences };
        aiDataManager.updatePreferences(newPreferences);
        return {
          ...context,
          preferences: newPreferences
        };
      })
    },
    ERROR: {
      actions: assign(({ context, event }) => ({
        ...context,
        error: event.error,
        isProcessing: false
      }))
    },
    CLEAR_ERROR: {
      actions: assign(({ context }) => ({
        ...context,
        error: null
      }))
    }
  }
});

// Global instances
export const aiDataManager = new AIDataManager();
export const aiActor = createActor(aiMachine).start();

// Reactive Svelte stores
export const aiState = writable(aiActor.getSnapshot());
export const currentSession = derived(aiState, ($state) => $state.context.currentSession);
export const activeMessages = derived(aiState, ($state) => $state.context.activeMessages);
export const isProcessing = derived(aiState, ($state) => $state.context.isProcessing);
export const aiError = derived(aiState, ($state) => $state.context.error);
export const aiPreferences = derived(aiState, ($state) => $state.context.preferences);
export const searchResults = derived(aiState, ($state) => $state.context.searchResults);
export const currentAnalysis = derived(aiState, ($state) => $state.context.currentAnalysis);

// Update store when actor state changes
aiActor.subscribe((state) => {
  aiState.set(state);
});

// Public API for components
export class AIAssistantService {
  static createSession(sessionData: Omit<AISession, 'id' | 'createdAt' | 'lastActiveAt' | 'messageCount'>): void {
    aiActor.send({ type: 'CREATE_SESSION', sessionData });
  }

  static loadSession(sessionId: string): void {
    aiActor.send({ type: 'LOAD_SESSION', sessionId });
  }

  static sendMessage(content: string, options: { caseId?: string; documentId?: string } = {}): void {
    const session = get(currentSession);
    if (!session) {
      throw new Error('No active session');
    }

    aiActor.send({
      type: 'SEND_MESSAGE',
      message: {
        content,
        sessionId: session.id,
        role: 'user',
        caseId: options.caseId,
        documentId: options.documentId
      }
    });
  }

  static analyzeDocument(documentId: string, content: string, analysisType: 'summary' | 'risks' | 'clauses' | 'compliance'): void {
    aiActor.send({
      type: 'ANALYZE_DOCUMENT',
      documentId,
      analysisType,
      content
    });
  }

  static searchMessages(query: string, options: { caseId?: string; limit?: number } = {}): void {
    aiActor.send({ type: 'SEARCH_MESSAGES', query, options });
  }

  static updatePreferences(preferences: Partial<UserPreferences>): void {
    aiActor.send({ type: 'UPDATE_PREFERENCES', preferences });
  }

  static clearError(): void {
    aiActor.send({ type: 'CLEAR_ERROR' });
  }

  static getRecentSessions(limit: number = 10): AISession[] {
    return aiDataManager.getRecentSessions(limit);
  }

  static getCachedAnalysis(documentId: string, analysisType: string): DocumentAnalysis | null {
    return aiDataManager.getAnalysis(documentId, analysisType);
  }

  static getStats() {
    return aiDataManager.getStats();
  }

  static cleanup(olderThanDays: number = 30): void {
    aiDataManager.cleanupOldData(olderThanDays);
  }
}

// Initialize on browser load
if (browser) {
  // Load recent sessions and preferences on startup
  setTimeout(() => {
    const preferences = aiDataManager.getPreferences();
    aiActor.send({ type: 'UPDATE_PREFERENCES', preferences });
  }, 1000);

  // Periodic cleanup (daily)
  setInterval(() => {
    AIAssistantService.cleanup();
  }, 24 * 60 * 60 * 1000);
}

export default AIAssistantService;