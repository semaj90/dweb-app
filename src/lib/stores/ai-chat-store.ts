// AI Chat Store - Focused chat interface management
import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';
import { aiUnified, aiActions, type ChatMessage } from './ai-unified';

interface ChatSession {
  id: string;
  name: string;
  messages: ChatMessage[];
  createdAt: number;
  lastActivity: number;
  context: string;
  model: string;
}

interface ChatState {
  // Current session
  currentSession: ChatSession | null;
  
  // Session management
  sessions: ChatSession[];
  maxSessions: number;
  
  // UI state
  isTyping: boolean;
  typingIndicator: string;
  showSuggestions: boolean;
  suggestions: string[];
  
  // Input state
  currentInput: string;
  inputHistory: string[];
  historyIndex: number;
  
  // Chat settings
  settings: {
    autoSave: boolean;
    persistHistory: boolean;
    showTimestamps: boolean;
    enableSuggestions: boolean;
    maxMessageLength: number;
  };
  
  // Performance
  lastResponseTime: number;
  averageResponseTime: number;
  responseCount: number;
}

const initialChatState: ChatState = {
  currentSession: null,
  sessions: [],
  maxSessions: 10,
  isTyping: false,
  typingIndicator: '',
  showSuggestions: false,
  suggestions: [],
  currentInput: '',
  inputHistory: [],
  historyIndex: -1,
  settings: {
    autoSave: true,
    persistHistory: true,
    showTimestamps: true,
    enableSuggestions: true,
    maxMessageLength: 4000
  },
  lastResponseTime: 0,
  averageResponseTime: 0,
  responseCount: 0
};

// Main chat store
export const aiChatStore = writable<ChatState>(initialChatState);

// Derived stores
export const currentMessages = derived(
  [aiChatStore, aiUnified],
  ([$chat, $unified]) => {
    // Prefer current session messages, fallback to unified chat state
    return $chat.currentSession?.messages || $unified.chatState.messages;
  }
);

export const isProcessing = derived(
  aiUnified,
  ($unified) => $unified.chatState.isProcessing
);

export const chatPerformance = derived(
  aiChatStore,
  ($chat) => ({
    lastResponseTime: $chat.lastResponseTime,
    averageResponseTime: $chat.averageResponseTime,
    responseCount: $chat.responseCount
  })
);

// Chat actions
export const chatActions = {
  // Initialize chat system
  async initialize() {
    if (!browser) return;
    
    try {
      // Load persisted sessions
      await this.loadPersistedSessions();
      
      // Create default session if none exist
      const sessions = get(aiChatStore).sessions;
      if (sessions.length === 0) {
        await this.createNewSession('Legal Analysis Chat');
      } else {
        // Resume last session
        this.switchToSession(sessions[sessions.length - 1].id);
      }
      
      console.log('✅ Chat system initialized');
    } catch (error) {
      console.error('❌ Chat initialization failed:', error);
    }
  },
  
  // Send message through unified AI system
  async sendMessage(content: string, context?: string) {
    if (!browser || !content.trim()) return;
    
    const chatState = get(aiChatStore);
    const startTime = performance.now();
    
    // Update input history
    const newInputHistory = [content, ...chatState.inputHistory.slice(0, 19)];
    
    aiChatStore.update(state => ({
      ...state,
      currentInput: '',
      inputHistory: newInputHistory,
      historyIndex: -1,
      isTyping: true,
      typingIndicator: 'AI is thinking...'
    }));
    
    try {
      // Send through unified AI system
      await aiActions.sendMessage(content, context);
      
      const responseTime = performance.now() - startTime;
      
      // Update performance metrics
      aiChatStore.update(state => {
        const newCount = state.responseCount + 1;
        const newAverage = (state.averageResponseTime * state.responseCount + responseTime) / newCount;
        
        return {
          ...state,
          isTyping: false,
          typingIndicator: '',
          lastResponseTime: responseTime,
          averageResponseTime: newAverage,
          responseCount: newCount
        };
      });
      
      // Update current session
      this.updateCurrentSession();
      
      // Auto-save if enabled
      if (chatState.settings.autoSave) {
        await this.saveCurrentSession();
      }
      
      // Generate suggestions if enabled
      if (chatState.settings.enableSuggestions) {
        this.generateSuggestions();
      }
      
    } catch (error) {
      console.error('Chat send error:', error);
      
      aiChatStore.update(state => ({
        ...state,
        isTyping: false,
        typingIndicator: ''
      }));
    }
  },
  
  // Create new chat session
  async createNewSession(name?: string) {
    const sessionId = generateSessionId();
    const now = Date.now();
    
    const newSession: ChatSession = {
      id: sessionId,
      name: name || `Chat ${new Date().toLocaleTimeString()}`,
      messages: [],
      createdAt: now,
      lastActivity: now,
      context: 'legal-analysis',
      model: 'gemma3-legal'
    };
    
    aiChatStore.update(state => {
      // Limit number of sessions
      const updatedSessions = [newSession, ...state.sessions.slice(0, state.maxSessions - 1)];
      
      return {
        ...state,
        currentSession: newSession,
        sessions: updatedSessions
      };
    });
    
    // Clear unified chat state for new session
    aiActions.clearChat();
    
    if (browser) {
      await this.saveCurrentSession();
    }
    
    return sessionId;
  },
  
  // Switch to existing session
  switchToSession(sessionId: string) {
    const chatState = get(aiChatStore);
    const session = chatState.sessions.find(s => s.id === sessionId);
    
    if (session) {
      aiChatStore.update(state => ({
        ...state,
        currentSession: session
      }));
      
      // Update unified store with session messages
      aiUnified.update(unified => ({
        ...unified,
        chatState: {
          ...unified.chatState,
          messages: session.messages,
          sessionId: session.id,
          context: session.context,
          currentModel: session.model
        }
      }));
    }
  },
  
  // Delete session
  async deleteSession(sessionId: string) {
    const chatState = get(aiChatStore);
    
    aiChatStore.update(state => {
      const updatedSessions = state.sessions.filter(s => s.id !== sessionId);
      const newCurrentSession = state.currentSession?.id === sessionId 
        ? (updatedSessions[0] || null) 
        : state.currentSession;
      
      return {
        ...state,
        sessions: updatedSessions,
        currentSession: newCurrentSession
      };
    });
    
    if (browser) {
      await this.persistSessions();
    }
  },
  
  // Update current session with latest messages
  updateCurrentSession() {
    const chatState = get(aiChatStore);
    const unifiedState = get(aiUnified);
    
    if (chatState.currentSession) {
      const updatedSession = {
        ...chatState.currentSession,
        messages: unifiedState.chatState.messages,
        lastActivity: Date.now()
      };
      
      aiChatStore.update(state => ({
        ...state,
        currentSession: updatedSession,
        sessions: state.sessions.map(s => 
          s.id === updatedSession.id ? updatedSession : s
        )
      }));
    }
  },
  
  // Save current session to localStorage
  async saveCurrentSession() {
    if (!browser) return;
    
    const chatState = get(aiChatStore);
    if (chatState.currentSession && chatState.settings.persistHistory) {
      try {
        const key = `chat_session_${chatState.currentSession.id}`;
        localStorage.setItem(key, JSON.stringify(chatState.currentSession));
      } catch (error) {
        console.warn('Failed to save session:', error);
      }
    }
  },
  
  // Load persisted sessions
  async loadPersistedSessions() {
    if (!browser) return;
    
    try {
      const sessions: ChatSession[] = [];
      
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key?.startsWith('chat_session_')) {
          const sessionData = localStorage.getItem(key);
          if (sessionData) {
            const session = JSON.parse(sessionData) as ChatSession;
            sessions.push(session);
          }
        }
      }
      
      // Sort by last activity
      sessions.sort((a, b) => b.lastActivity - a.lastActivity);
      
      aiChatStore.update(state => ({
        ...state,
        sessions: sessions.slice(0, state.maxSessions)
      }));
      
    } catch (error) {
      console.warn('Failed to load persisted sessions:', error);
    }
  },
  
  // Persist all sessions
  async persistSessions() {
    if (!browser) return;
    
    const chatState = get(aiChatStore);
    
    // Clear old sessions from localStorage
    const keysToRemove: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith('chat_session_')) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(key => localStorage.removeItem(key));
    
    // Save current sessions
    chatState.sessions.forEach(session => {
      const key = `chat_session_${session.id}`;
      localStorage.setItem(key, JSON.stringify(session));
    });
  },
  
  // Update input
  setInput(value: string) {
    aiChatStore.update(state => ({
      ...state,
      currentInput: value
    }));
  },
  
  // Navigate input history
  navigateHistory(direction: 'up' | 'down') {
    const chatState = get(aiChatStore);
    const { inputHistory, historyIndex } = chatState;
    
    if (inputHistory.length === 0) return;
    
    let newIndex = historyIndex;
    if (direction === 'up' && historyIndex < inputHistory.length - 1) {
      newIndex = historyIndex + 1;
    } else if (direction === 'down' && historyIndex > -1) {
      newIndex = historyIndex - 1;
    }
    
    const newInput = newIndex === -1 ? '' : inputHistory[newIndex];
    
    aiChatStore.update(state => ({
      ...state,
      historyIndex: newIndex,
      currentInput: newInput
    }));
  },
  
  // Generate contextual suggestions
  generateSuggestions() {
    const unifiedState = get(aiUnified);
    const messages = unifiedState.chatState.messages;
    
    if (messages.length === 0) {
      const defaultSuggestions = [
        "What are the key legal considerations for this case?",
        "Can you summarize the relevant statutes?",
        "What precedents apply to this situation?",
        "Help me draft a legal analysis."
      ];
      
      aiChatStore.update(state => ({
        ...state,
        suggestions: defaultSuggestions,
        showSuggestions: true
      }));
      return;
    }
    
    // Generate contextual suggestions based on conversation
    const lastMessage = messages[messages.length - 1];
    const suggestions: string[] = [];
    
    if (lastMessage.role === 'assistant') {
      suggestions.push("Can you elaborate on that point?");
      suggestions.push("What are the potential risks?");
      suggestions.push("Are there any exceptions to consider?");
    }
    
    aiChatStore.update(state => ({
      ...state,
      suggestions,
      showSuggestions: suggestions.length > 0
    }));
  },
  
  // Use suggestion
  useSuggestion(suggestion: string) {
    aiChatStore.update(state => ({
      ...state,
      currentInput: suggestion,
      showSuggestions: false
    }));
  },
  
  // Update settings
  updateSettings(newSettings: Partial<ChatState['settings']>) {
    aiChatStore.update(state => ({
      ...state,
      settings: {
        ...state.settings,
        ...newSettings
      }
    }));
    
    if (browser) {
      localStorage.setItem('chat_settings', JSON.stringify(get(aiChatStore).settings));
    }
  },
  
  // Load settings
  loadSettings() {
    if (!browser) return;
    
    try {
      const savedSettings = localStorage.getItem('chat_settings');
      if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        this.updateSettings(settings);
      }
    } catch (error) {
      console.warn('Failed to load chat settings:', error);
    }
  },
  
  // Clear all data
  async clearAllData() {
    if (!browser) return;
    
    // Clear localStorage
    const keysToRemove: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith('chat_')) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(key => localStorage.removeItem(key));
    
    // Reset store
    aiChatStore.set(initialChatState);
    
    // Clear unified store
    aiActions.clearChat();
  }
};

// Utility functions
function generateSessionId(): string {
  return `chat_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
}

// Auto-initialize if in browser
if (browser) {
  // Load settings first
  chatActions.loadSettings();
  
  // Initialize chat system
  chatActions.initialize();
}