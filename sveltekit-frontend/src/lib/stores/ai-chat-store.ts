/**
 * YoRHa AI Chat Store - Persistent Chat Management
 * Handles conversation history, user preferences, and Enhanced RAG integration
 */

import { writable, derived, get } from "svelte/store";
import { browser } from "$app/environment";
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  type?: 'user' | 'assistant' | 'system' | 'error' | 'command';
  metadata?: {
    confidence?: number;
    sources?: unknown[];
    processing_time?: number;
    model?: string;
    tokens_used?: number;
  };
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  created_at: Date;
  updated_at: Date;
  tags: string[];
  case_id?: string;
  evidence_id?: string;
}

export interface UserPreferences {
  theme: 'yorha-dark' | 'yorha-light';
  auto_save: boolean;
  max_history: number;
  enable_rag: boolean;
  default_model: string;
  notification_sound: boolean;
  export_format: 'json' | 'markdown' | 'txt';
}

export interface ChatState {
  currentSession: ChatSession | null;
  sessions: ChatSession[];
  preferences: UserPreferences;
  isLoading: boolean;
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
  ragServiceUrl: string;
  lastError: string | null;
}

// Default preferences
const defaultPreferences: UserPreferences = {
  theme: 'yorha-dark',
  auto_save: true,
  max_history: 100,
  enable_rag: true,
  default_model: 'enhanced-rag',
  notification_sound: false,
  export_format: 'markdown'
};

// Initialize chat state
const initialState: ChatState = {
  currentSession: null,
  sessions: [],
  preferences: defaultPreferences,
  isLoading: false,
  connectionStatus: 'disconnected',
  ragServiceUrl: 'http://localhost:8093',
  lastError: null
};

// Main chat store
export const chatStore = writable<ChatState>(initialState);
;
// Derived stores for easy access
export const currentSession = derived(chatStore, $state => $state.currentSession);
export const allSessions = derived(chatStore, $state => $state.sessions);
export const userPreferences = derived(chatStore, $state => $state.preferences);
export const isLoading = derived(chatStore, $state => $state.isLoading);
export const connectionStatus = derived(chatStore, $state => $state.connectionStatus);
;
// Storage keys
const STORAGE_KEYS = {
  SESSIONS: 'yorha-ai-chat-sessions',
  PREFERENCES: 'yorha-ai-chat-preferences',
  CURRENT_SESSION: 'yorha-ai-current-session'
};

/**
 * AI Chat Store Manager
 */
class AIChatStore {
  private readonly RAG_SERVICE_URL = 'http://localhost:8093';

  constructor() {
    if (browser) {
      this.loadFromStorage();
      this.checkRAGConnection();
    }
  }

  // Session Management
  createNewSession(title?: string): ChatSession {
    const session: ChatSession = {
      id: `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      title: title || `YoRHa Session ${new Date().toLocaleString()}`,
      messages: [],
      created_at: new Date(),
      updated_at: new Date(),
      tags: ['yorha', 'legal-ai']
    };

    chatStore.update(state => {
      state.currentSession = session;
      state.sessions.unshift(session);

      // Limit sessions based on preferences
      if (state.sessions.length > state.preferences.max_history) {
        state.sessions = state.sessions.slice(0, state.preferences.max_history);
      }

      return state;
    });

    this.saveToStorage();
    return session;
  }

  switchToSession(sessionId: string): void {
    const state = get(chatStore);
    const session = state.sessions.find(s => s.id === sessionId);

    if (session) {
      chatStore.update(state => {
        state.currentSession = session;
        return state;
      });
      this.saveToStorage();
    }
  }

  deleteSession(sessionId: string): void {
    chatStore.update(state => {
      state.sessions = state.sessions.filter(s => s.id !== sessionId);

      // If current session was deleted, switch to most recent
      if (state.currentSession?.id === sessionId) {
        state.currentSession = state.sessions[0] || null;
      }

      return state;
    });
    this.saveToStorage();
  }

  // Message Management
  addMessage(message: Omit<ChatMessage, 'id'>): ChatMessage {
    const fullMessage: ChatMessage = {
      ...message,
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: message.timestamp || new Date()
    };

    chatStore.update(state => {
      if (!state.currentSession) {
        state.currentSession = this.createNewSession();
      }

      state.currentSession.messages.push(fullMessage);
      state.currentSession.updated_at = new Date();

      // Update session in sessions array
      const sessionIndex = state.sessions.findIndex(s => s.id === state.currentSession!.id);
      if (sessionIndex >= 0) {
        state.sessions[sessionIndex] = state.currentSession;
      }

      return state;
    });

    if (get(userPreferences).auto_save) {
      this.saveToStorage();
    }

    return fullMessage;
  }

  updateMessage(messageId: string, updates: Partial<ChatMessage>): void {
    chatStore.update(state => {
      if (state.currentSession) {
        const messageIndex = state.currentSession.messages.findIndex(m => m.id === messageId);
        if (messageIndex >= 0) {
          state.currentSession.messages[messageIndex] = {
            ...state.currentSession.messages[messageIndex],
            ...updates
          };
          state.currentSession.updated_at = new Date();
        }
      }
      return state;
    });

    this.saveToStorage();
  }

  deleteMessage(messageId: string): void {
    chatStore.update(state => {
      if (state.currentSession) {
        state.currentSession.messages = state.currentSession.messages.filter(m => m.id !== messageId);
        state.currentSession.updated_at = new Date();
      }
      return state;
    });

    this.saveToStorage();
  }

  clearCurrentSession(): void {
    chatStore.update(state => {
      if (state.currentSession) {
        state.currentSession.messages = [];
        state.currentSession.updated_at = new Date();
      }
      return state;
    });

    this.saveToStorage();
  }

  // Enhanced RAG Integration
  async sendToRAG(message: string, context?: unknown): Promise<any> {
    chatStore.update(state => {
      state.isLoading = true;
      state.lastError = null;
      return state;
    });

    try {
      const response = await fetch(`${this.RAG_SERVICE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          context: context || 'legal-ai',
          user_id: 'yorha-user',
          session_id: get(currentSession)?.id || 'default',
          include_vector_search: true,
          max_tokens: 1000,
          temperature: 0.7
        })
      });

      if (!response.ok) {
        throw new Error(`RAG service error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      chatStore.update(state => {
        state.connectionStatus = 'connected';
        state.isLoading = false;
        return state;
      });

      return result;
    } catch (error: any) {
      console.error('RAG service error:', error);

      chatStore.update(state => {
        state.connectionStatus = 'disconnected';
        state.isLoading = false;
        state.lastError = error instanceof Error ? error.message : 'Unknown error';
        return state;
      });

      throw error;
    }
  }

  async checkRAGConnection(): Promise<boolean> {
    chatStore.update(state => {
      state.connectionStatus = 'connecting';
      return state;
    });

    try {
      let response: Response;
        try {
          response = await fetch(`${this.RAG_SERVICE_URL}/health`, {
        method: 'GET',
        timeout: 5000
      } as any);
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
        } catch (error: any) {
          console.error('Fetch failed:', error);
          throw error;
        }

      const isHealthy = response.ok;

      chatStore.update(state => {
        state.connectionStatus = isHealthy ? 'connected' : 'disconnected';
        return state;
      });

      return isHealthy;
    } catch (error: any) {
      chatStore.update(state => {
        state.connectionStatus = 'disconnected';
        state.lastError = 'RAG service unavailable';
        return state;
      });
      return false;
    }
  }

  // Storage Management
  private saveToStorage(): void {
    if (!browser) return;

    const state = get(chatStore);

    try {
      localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(
        state.sessions.map(s => ({
          ...s,
          created_at: s.created_at.toISOString(),
          updated_at: s.updated_at.toISOString(),
          messages: s.messages.map(m => ({
            ...m,
            timestamp: m.timestamp.toISOString()
          }))
        }))
      ));

      localStorage.setItem(STORAGE_KEYS.PREFERENCES, JSON.stringify(state.preferences));

      if (state.currentSession) {
        localStorage.setItem(STORAGE_KEYS.CURRENT_SESSION, state.currentSession.id);
      }
    } catch (error: any) {
      console.error('Failed to save chat data:', error);
    }
  }

  private loadFromStorage(): void {
    if (!browser) return;

    try {
      // Load preferences
      const savedPreferences = localStorage.getItem(STORAGE_KEYS.PREFERENCES);
      if (savedPreferences) {
        const preferences = JSON.parse(savedPreferences);
        chatStore.update(state => {
          state.preferences = { ...defaultPreferences, ...preferences };
          return state;
        });
      }

      // Load sessions
      const savedSessions = localStorage.getItem(STORAGE_KEYS.SESSIONS);
      if (savedSessions) {
        const sessions = JSON.parse(savedSessions).map((s: any) => ({
          ...s,
          created_at: new Date(s.created_at),
          updated_at: new Date(s.updated_at),
          messages: s.messages.map((m: any) => ({
            ...m,
            timestamp: new Date(m.timestamp)
          }))
        }));

        const currentSessionId = localStorage.getItem(STORAGE_KEYS.CURRENT_SESSION);
        const currentSession = sessions.find((s: ChatSession) => s.id === currentSessionId) || sessions[0] || null;

        chatStore.update(state => {
          state.sessions = sessions;
          state.currentSession = currentSession;
          return state;
        });
      }
    } catch (error: any) {
      console.error('Failed to load chat data:', error);
    }
  }

  // Utility methods
  exportSession(sessionId: string, format: 'json' | 'markdown' | 'txt' = 'markdown'): string {
    const state = get(chatStore);
    const session = state.sessions.find(s => s.id === sessionId);

    if (!session) return '';

    switch (format) {
      case 'json':
        return JSON.stringify(session, null, 2);

      case 'markdown':
        let md = `# ${session.title}\n\n`;
        md += `**Created:** ${session.created_at.toLocaleString()}\n`;
        md += `**Updated:** ${session.updated_at.toLocaleString()}\n`;
        md += `**Tags:** ${session.tags.join(', ')}\n\n`;

        session.messages.forEach(msg => {
          md += `## ${msg.role === 'user' ? 'User' : 'YoRHa AI'} (${msg.timestamp.toLocaleTimeString()})\n\n`;
          md += `${msg.content}\n\n`;
        });

        return md;

      case 'txt':
        let txt = `${session.title}\n${'='.repeat(session.title.length)}\n\n`;
        txt += `Created: ${session.created_at.toLocaleString()}\n`;
        txt += `Updated: ${session.updated_at.toLocaleString()}\n\n`;

        session.messages.forEach(msg => {
          txt += `[${msg.timestamp.toLocaleTimeString()}] ${msg.role.toUpperCase()}: ${msg.content}\n\n`;
        });

        return txt;

      default:
        return '';
    }
  }

  loadChatHistory(): ChatMessage[] {
    const state = get(chatStore);
    return state.currentSession?.messages || [];
  }

  clearHistory(): void {
    this.clearCurrentSession();
  }

  updatePreferences(updates: Partial<UserPreferences>): void {
    chatStore.update(state => {
      state.preferences = { ...state.preferences, ...updates };
      return state;
    });
    this.saveToStorage();
  }
}

// Export singleton instance
export const aiChatStore = new AIChatStore();
;
// Export convenience functions
export const {
  createNewSession,
  switchToSession,
  deleteSession,
  addMessage,
  updateMessage,
  deleteMessage,
  clearCurrentSession,
  sendToRAG,
  checkRAGConnection,
  exportSession,
  loadChatHistory,
  clearHistory,
  updatePreferences
} = aiChatStore;
