// Chat State Machine with XState
// Manages AI chat conversations, message history, and real-time communication

import { createMachine, assign, type InterpreterFrom } from 'xstate';
import { ragPipeline } from '../services/enhanced-rag-pipeline';
import { multiProtocolRouter } from '../services/multi-protocol-router';

// Types for chat management
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  status: 'sending' | 'sent' | 'delivered' | 'error';
  metadata?: {
    model?: string;
    protocol?: string;
    processingTime?: number;
    sources?: unknown[];
    tokens?: number;
  };
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  context?: {
    caseId?: string;
    documentId?: string;
    legalContext?: string;
  };
}

export interface ChatContext {
  currentSession: ChatSession | null;
  sessions: ChatSession[];
  currentMessage: string;
  isTyping: boolean;
  isConnected: boolean;
  connectionStatus: 'connected' | 'connecting' | 'disconnected' | 'error';
  aiModel: string;
  streamingEnabled: boolean;
  autoSave: boolean;
  settings: {
    maxTokens: number;
    temperature: number;
    topK: number;
    preferredProtocol: 'auto' | 'quic' | 'grpc' | 'rest';
    enableSources: boolean;
    contextWindow: number;
  };
  performance: {
    totalMessages: number;
    totalProcessingTime: number;
    averageResponseTime: number;
    protocolUsage: Record<string, number>;
  };
}

// Chat events
export type ChatEvent =
  | { type: 'CREATE_SESSION'; title?: string; context?: unknown }
  | { type: 'LOAD_SESSION'; sessionId: string }
  | { type: 'DELETE_SESSION'; sessionId: string }
  | { type: 'UPDATE_MESSAGE'; message: string }
  | { type: 'SEND_MESSAGE' }
  | { type: 'CANCEL_MESSAGE' }
  | { type: 'RETRY_MESSAGE'; messageId: string }
  | { type: 'MESSAGE_DELIVERED'; messageId: string; response: unknown }
  | { type: 'MESSAGE_ERROR'; messageId: string; error: string }
  | { type: 'START_TYPING' }
  | { type: 'STOP_TYPING' }
  | { type: 'CONNECTION_STATUS'; status: ChatContext['connectionStatus'] }
  | { type: 'UPDATE_SETTINGS'; settings: Partial<ChatContext['settings']> }
  | { type: 'CLEAR_HISTORY' }
  | { type: 'EXPORT_SESSION'; sessionId: string };

// Default context
const defaultContext: ChatContext = {
  currentSession: null,
  sessions: [],
  currentMessage: '',
  isTyping: false,
  isConnected: true,
  connectionStatus: 'connected',
  aiModel: 'gemma3:legal-latest',
  streamingEnabled: true,
  autoSave: true,
  settings: {
    maxTokens: 2000,
    temperature: 0.7,
    topK: 5,
    preferredProtocol: 'auto',
    enableSources: true,
    contextWindow: 10
  },
  performance: {
    totalMessages: 0,
    totalProcessingTime: 0,
    averageResponseTime: 0,
    protocolUsage: { quic: 0, grpc: 0, rest: 0 }
  }
};

// Services for chat machine
const chatServices = {
  sendMessage: async (context: ChatContext, event: unknown) => {
    const { currentSession, currentMessage, settings } = context;
    if (!currentSession || !currentMessage.trim()) {
      throw new Error('No active session or empty message');
    }

    const startTime = performance.now();
    
    try {
      // Create user message
      const userMessage: ChatMessage = {
        id: `msg_${Date.now()}_user`,
        role: 'user',
        content: currentMessage.trim(),
        timestamp: Date.now(),
        status: 'sent'
      };

      // Get conversation context for RAG
      const conversationContext = currentSession.messages
        .slice(-settings.contextWindow)
        .map(msg => `${msg.role}: ${msg.content}`)
        .join('\n');

      // Send to RAG pipeline
      const result = await ragPipeline.query(currentMessage, {
        context: conversationContext,
        caseId: currentSession.context?.caseId,
        maxTokens: settings.maxTokens,
        temperature: settings.temperature,
        topK: settings.topK,
        protocol: settings.preferredProtocol === 'auto' ? undefined : settings.preferredProtocol,
        stream: context.streamingEnabled
      });

      const processingTime = performance.now() - startTime;

      // Create assistant response
      const assistantMessage: ChatMessage = {
        id: `msg_${Date.now()}_assistant`,
        role: 'assistant',
        content: result.answer || result.response || 'No response generated',
        timestamp: Date.now(),
        status: 'delivered',
        metadata: {
          model: context.aiModel,
          protocol: result.metadata?.protocolUsed,
          processingTime,
          sources: settings.enableSources ? result.sources : undefined,
          tokens: result.metadata?.tokens
        }
      };

      return {
        userMessage,
        assistantMessage,
        processingTime,
        protocol: result.metadata?.protocolUsed || 'unknown'
      };

    } catch (error) {
      throw new Error(`Message failed: ${error.message}`);
    }
  },

  loadSession: async (context: ChatContext, event: unknown) => {
    try {
      // Load session from storage (localStorage, IndexedDB, or API)
      const sessionData = localStorage.getItem(`chat_session_${event.sessionId}`);
      if (!sessionData) {
        throw new Error('Session not found');
      }

      return JSON.parse(sessionData);
    } catch (error) {
      throw new Error(`Failed to load session: ${error.message}`);
    }
  },

  saveSession: async (context: ChatContext) => {
    if (!context.currentSession || !context.autoSave) return;

    try {
      localStorage.setItem(
        `chat_session_${context.currentSession.id}`,
        JSON.stringify(context.currentSession)
      );

      // Update sessions list
      const sessionsIndex = context.sessions.findIndex(s => s.id === context.currentSession!.id);
      const updatedSessions = [...context.sessions];
      if (sessionsIndex >= 0) {
        updatedSessions[sessionsIndex] = context.currentSession;
      } else {
        updatedSessions.push(context.currentSession);
      }

      localStorage.setItem('chat_sessions', JSON.stringify(updatedSessions));
    } catch (error) {
      console.error('Failed to save session:', error);
    }
  },

  loadSessions: async () => {
    try {
      const sessionsData = localStorage.getItem('chat_sessions');
      return sessionsData ? JSON.parse(sessionsData) : [];
    } catch (error) {
      console.error('Failed to load sessions:', error);
      return [];
    }
  }
};

// Chat state machine
export const chatMachine = createMachine({
  id: 'chat',
  initial: 'idle',
  context: defaultContext,
  states: {
    idle: {
      entry: 'loadSessions',
      on: {
        CREATE_SESSION: {
          target: 'active',
          actions: 'createSession'
        },
        LOAD_SESSION: {
          target: 'loading',
          actions: assign({
            connectionStatus: 'connecting'
          })
        },
        UPDATE_SETTINGS: {
          actions: 'updateSettings'
        }
      }
    },

    loading: {
      invoke: {
        id: 'loadSession',
        src: 'loadSession',
        onDone: {
          target: 'active',
          actions: [
            assign({
              currentSession: (_, event) => event.data,
              connectionStatus: 'connected'
            }),
            'addToSessions'
          ]
        },
        onError: {
          target: 'idle',
          actions: [
            assign({
              connectionStatus: 'error'
            }),
            'handleError'
          ]
        }
      }
    },

    active: {
      entry: 'setActiveSession',
      initial: 'ready',
      states: {
        ready: {
          on: {
            UPDATE_MESSAGE: {
              actions: assign({
                currentMessage: (_, event) => event.message
              })
            },
            SEND_MESSAGE: {
              target: 'sending',
              guard: 'hasMessage'
            },
            START_TYPING: {
              actions: assign({ isTyping: true })
            },
            STOP_TYPING: {
              actions: assign({ isTyping: false })
            }
          }
        },

        sending: {
          entry: ['createUserMessage', 'clearCurrentMessage'],
          invoke: {
            id: 'sendMessage',
            src: 'sendMessage',
            onDone: {
              target: 'ready',
              actions: [
                'addAssistantMessage',
                'updatePerformanceMetrics',
                'saveSession'
              ]
            },
            onError: {
              target: 'ready',
              actions: [
                'handleMessageError',
                'saveSession'
              ]
            }
          },
          on: {
            CANCEL_MESSAGE: {
              target: 'ready',
              actions: 'cancelMessage'
            }
          }
        }
      },

      on: {
        CREATE_SESSION: {
          actions: 'createSession'
        },
        DELETE_SESSION: {
          target: 'idle',
          actions: 'deleteSession'
        },
        RETRY_MESSAGE: {
          target: '.sending',
          actions: 'prepareRetry'
        },
        CLEAR_HISTORY: {
          actions: ['clearSessionHistory', 'saveSession']
        },
        EXPORT_SESSION: {
          actions: 'exportSession'
        },
        UPDATE_SETTINGS: {
          actions: 'updateSettings'
        }
      }
    }
  },

  on: {
    CONNECTION_STATUS: {
      actions: assign({
        connectionStatus: (_, event) => event.status,
        isConnected: (_, event) => event.status === 'connected'
      })
    }
  }
}, {
  services: chatServices,
  guards: {
    hasMessage: (context) => context.currentMessage.trim().length > 0
  },
  actions: {
    loadSessions: assign({
      sessions: async () => await chatServices.loadSessions()
    }),

    createSession: assign({
      currentSession: (context, event) => ({
        id: `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        title: event.title || `Chat ${new Date().toLocaleString()}`,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
        context: event.context || {}
      })
    }),

    setActiveSession: assign({
      connectionStatus: 'connected',
      isConnected: true
    }),

    addToSessions: assign({
      sessions: (context) => {
        const existing = context.sessions.find(s => s.id === context.currentSession?.id);
        if (existing) return context.sessions;
        return [...context.sessions, context.currentSession!];
      }
    }),

    createUserMessage: assign({
      currentSession: (context) => {
        if (!context.currentSession) return context.currentSession;

        const userMessage: ChatMessage = {
          id: `msg_${Date.now()}_user`,
          role: 'user',
          content: context.currentMessage.trim(),
          timestamp: Date.now(),
          status: 'sending'
        };

        return {
          ...context.currentSession,
          messages: [...context.currentSession.messages, userMessage],
          updatedAt: Date.now()
        };
      }
    }),

    clearCurrentMessage: assign({
      currentMessage: ''
    }),

    addAssistantMessage: assign({
      currentSession: (context, event) => {
        if (!context.currentSession) return context.currentSession;

        const { userMessage, assistantMessage } = event.data;
        const messages = [...context.currentSession.messages];
        
        // Update user message status
        const userIndex = messages.findIndex(m => m.role === 'user' && m.status === 'sending');
        if (userIndex >= 0) {
          messages[userIndex] = { ...messages[userIndex], status: 'sent' };
        }

        // Add assistant message
        messages.push(assistantMessage);

        return {
          ...context.currentSession,
          messages,
          updatedAt: Date.now()
        };
      }
    }),

    updatePerformanceMetrics: assign({
      performance: (context, event) => {
        const { processingTime, protocol } = event.data;
        const perf = context.performance;
        
        return {
          totalMessages: perf.totalMessages + 1,
          totalProcessingTime: perf.totalProcessingTime + processingTime,
          averageResponseTime: (perf.totalProcessingTime + processingTime) / (perf.totalMessages + 1),
          protocolUsage: {
            ...perf.protocolUsage,
            [protocol]: (perf.protocolUsage[protocol] || 0) + 1
          }
        };
      }
    }),

    handleMessageError: assign({
      currentSession: (context, event) => {
        if (!context.currentSession) return context.currentSession;

        const messages = [...context.currentSession.messages];
        const lastUserMessage = messages.findLast(m => m.role === 'user');
        
        if (lastUserMessage) {
          const index = messages.indexOf(lastUserMessage);
          messages[index] = { ...lastUserMessage, status: 'error' };
        }

        // Add error message
        const errorMessage: ChatMessage = {
          id: `msg_${Date.now()}_error`,
          role: 'system',
          content: `Error: ${event.data.message}`,
          timestamp: Date.now(),
          status: 'delivered'
        };

        return {
          ...context.currentSession,
          messages: [...messages, errorMessage],
          updatedAt: Date.now()
        };
      }
    }),

    deleteSession: assign({
      currentSession: null,
      sessions: (context, event) => {
        const filtered = context.sessions.filter(s => s.id !== event.sessionId);
        
        // Remove from storage
        localStorage.removeItem(`chat_session_${event.sessionId}`);
        localStorage.setItem('chat_sessions', JSON.stringify(filtered));
        
        return filtered;
      }
    }),

    clearSessionHistory: assign({
      currentSession: (context) => {
        if (!context.currentSession) return context.currentSession;
        
        return {
          ...context.currentSession,
          messages: [],
          updatedAt: Date.now()
        };
      }
    }),

    updateSettings: assign({
      settings: (context, event) => ({
        ...context.settings,
        ...event.settings
      })
    }),

    prepareRetry: assign({
      currentMessage: (context, event) => {
        const message = context.currentSession?.messages.find(m => m.id === event.messageId);
        return message?.content || context.currentMessage;
      }
    }),

    cancelMessage: assign({
      currentSession: (context) => {
        if (!context.currentSession) return context.currentSession;

        const messages = context.currentSession.messages.filter(m => m.status !== 'sending');
        
        return {
          ...context.currentSession,
          messages,
          updatedAt: Date.now()
        };
      }
    }),

    saveSession: async (context) => {
      await chatServices.saveSession(context);
    },

    exportSession: (context, event) => {
      const session = context.sessions.find(s => s.id === event.sessionId);
      if (!session) return;

      const exportData = {
        session,
        exportedAt: new Date().toISOString(),
        format: 'json'
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `chat_session_${session.title.replace(/[^a-zA-Z0-9]/g, '_')}.json`;
      a.click();
      URL.revokeObjectURL(url);
    },

    handleError: (context, event) => {
      console.error('Chat machine error:', event.data);
    }
  }
});

// Type for the chat service
export type ChatService = InterpreterFrom<typeof chatMachine>;

// Helper functions for common operations
export const chatActions = {
  createSession: (title?: string, context?: unknown) => ({
    type: 'CREATE_SESSION' as const,
    title,
    context
  }),

  loadSession: (sessionId: string) => ({
    type: 'LOAD_SESSION' as const,
    sessionId
  }),

  deleteSession: (sessionId: string) => ({
    type: 'DELETE_SESSION' as const,
    sessionId
  }),

  updateMessage: (message: string) => ({
    type: 'UPDATE_MESSAGE' as const,
    message
  }),

  sendMessage: () => ({
    type: 'SEND_MESSAGE' as const
  }),

  retryMessage: (messageId: string) => ({
    type: 'RETRY_MESSAGE' as const,
    messageId
  }),

  clearHistory: () => ({
    type: 'CLEAR_HISTORY' as const
  }),

  updateSettings: (settings: Partial<ChatContext['settings']>) => ({
    type: 'UPDATE_SETTINGS' as const,
    settings
  })
};

// Selectors for derived state
export const chatSelectors = {
  isIdle: (state: unknown) => state.matches('idle'),
  isActive: (state: unknown) => state.matches('active'),
  isSending: (state: unknown) => state.matches('active.sending'),
  isReady: (state: unknown) => state.matches('active.ready'),
  
  canSendMessage: (context: ChatContext) => 
    context.currentMessage.trim().length > 0 && context.isConnected,
  
  currentSessionMessages: (context: ChatContext) => 
    context.currentSession?.messages || [],
  
  unreadCount: (context: ChatContext) => 
    context.sessions.reduce((count, session) => 
      count + session.messages.filter(m => m.role === 'assistant' && !m.metadata?.read).length, 0
    ),
  
  averageResponseTime: (context: ChatContext) => 
    Math.round(context.performance.averageResponseTime),
  
  lastMessage: (context: ChatContext) => {
    const messages = context.currentSession?.messages || [];
    return messages[messages.length - 1];
  }
};