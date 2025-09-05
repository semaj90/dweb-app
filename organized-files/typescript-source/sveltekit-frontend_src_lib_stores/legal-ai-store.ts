// Legal AI Store - SvelteKit 2 + Svelte 5 Integration
// Multi-layer caching: Loki.js + Redis + PostgreSQL + pgvector
// Features: XState orchestration, GPU acceleration, real-time updates

import { writable, derived, get } from 'svelte/store';
import { createMachine, interpret, type ActorRefFrom } from 'xstate';
import Loki from 'lokijs';
import Fuse from 'fuse.js';
import { browser } from '$app/environment';

// Import GPU integration
import { 
  gpuService,
  legalDB,
  type Neo4jSearchRequest,
  type Neo4jSearchResponse,
  type GPUEmbeddingRequest,
  type GPUEmbeddingResponse,
  GPU_SERVICE_URLS,
} from '$lib/gpu/nes-gpu-integration';

// Types for Legal AI Store
export interface LegalDocument {
  id: string;
  title: string;
  content: string;
  document_type: 'contract' | 'evidence' | 'brief' | 'citation' | 'case_law';
  practice_area: string;
  case_id?: string;
  embedding?: number[];
  metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface LegalCase {
  id: string;
  title: string;
  description: string;
  status: 'active' | 'closed' | 'pending' | 'archived';
  priority: 'low' | 'medium' | 'high' | 'critical';
  practice_area: string;
  documents: LegalDocument[];
  evidence_count: number;
  ai_summary?: string;
  created_at: string;
  updated_at: string;
}

export interface AIMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    model?: string;
    tokens_used?: number;
    processing_time_ms?: number;
    confidence_score?: number;
    sources?: string[];
    reasoning?: string;
  };
}

export interface SearchQuery {
  id: string;
  query: string;
  practice_area: string;
  document_type: string;
  filters: Record<string, any>;
  results: any[];
  total_results: number;
  processing_info: any;
  timestamp: string;
  cached: boolean;
}

export interface LegalAIState {
  // Authentication
  isAuthenticated: boolean;
  user: {
    id: string;
    name: string;
    email: string;
    role: 'admin' | 'prosecutor' | 'detective' | 'user';
    permissions: string[];
  } | null;

  // Current session
  currentCase: LegalCase | null;
  selectedDocuments: LegalDocument[];
  activeQuery: string;
  
  // Messages and conversation
  messageHistory: AIMessage[];
  isTyping: boolean;
  
  // Search and results
  searchHistory: SearchQuery[];
  currentSearchResults: any[];
  recentDocuments: LegalDocument[];
  
  // Processing states
  isProcessing: boolean;
  processingStage: string;
  progressPercentage: number;
  
  // GPU and performance
  gpuAvailable: boolean;
  tensorCoresActive: boolean;
  performanceMetrics: {
    avg_response_time: number;
    cache_hit_ratio: number;
    gpu_utilization: number;
    total_queries: number;
  };
  
  // Caching stats
  cacheStats: {
    loki_entries: number;
    redis_entries: number;
    postgres_entries: number;
    total_hits: number;
    total_misses: number;
  };
  
  // UI state
  sidebarOpen: boolean;
  activeTab: 'search' | 'cases' | 'documents' | 'chat' | 'settings';
  notifications: Array<{
    id: string;
    type: 'info' | 'success' | 'warning' | 'error';
    message: string;
    timestamp: string;
    dismissed: boolean;
  }>;
  
  // Errors
  errors: string[];
  lastError: string | null;
}

// XState Machine for Legal AI Workflow
export const legalAIMachine = createMachine({
  id: 'legalAI',
  initial: 'initializing',
  context: {
    isAuthenticated: false,
    user: null,
    currentCase: null,
    selectedDocuments: [],
    activeQuery: '',
    messageHistory: [],
    isTyping: false,
    searchHistory: [],
    currentSearchResults: [],
    recentDocuments: [],
    isProcessing: false,
    processingStage: '',
    progressPercentage: 0,
    gpuAvailable: false,
    tensorCoresActive: false,
    performanceMetrics: {
      avg_response_time: 0,
      cache_hit_ratio: 0,
      gpu_utilization: 0,
      total_queries: 0,
    },
    cacheStats: {
      loki_entries: 0,
      redis_entries: 0,
      postgres_entries: 0,
      total_hits: 0,
      total_misses: 0,
    },
    sidebarOpen: true,
    activeTab: 'search' as const,
    notifications: [],
    errors: [],
    lastError: null,
  },
  states: {
    initializing: {
      invoke: {
        src: 'initializeServices',
        onDone: {
          target: 'idle',
          actions: 'setInitialized',
        },
        onError: {
          target: 'error',
          actions: 'setError',
        },
      },
    },
    idle: {
      on: {
        LOGIN: 'authenticating',
        SEARCH: 'searching',
        LOAD_CASE: 'loadingCase',
        CHAT: 'chatting',
        UPLOAD_DOCUMENT: 'uploadingDocument',
        PROCESS_BATCH: 'batchProcessing',
        REFRESH_METRICS: 'refreshingMetrics',
        CLEAR_CACHE: 'clearingCache',
        SET_ACTIVE_TAB: {
          actions: 'setActiveTab',
        },
        TOGGLE_SIDEBAR: {
          actions: 'toggleSidebar',
        },
        DISMISS_NOTIFICATION: {
          actions: 'dismissNotification',
        },
      },
    },
    authenticating: {
      invoke: {
        src: 'authenticate',
        onDone: {
          target: 'idle',
          actions: 'setUser',
        },
        onError: {
          target: 'idle',
          actions: 'setAuthError',
        },
      },
    },
    searching: {
      entry: 'startProcessing',
      invoke: {
        src: 'performSearch',
        onDone: {
          target: 'idle',
          actions: ['setSearchResults', 'stopProcessing', 'addNotification'],
        },
        onError: {
          target: 'idle',
          actions: ['setError', 'stopProcessing'],
        },
      },
    },
    loadingCase: {
      entry: 'startProcessing',
      invoke: {
        src: 'loadCase',
        onDone: {
          target: 'idle',
          actions: ['setCurrentCase', 'stopProcessing'],
        },
        onError: {
          target: 'idle',
          actions: ['setError', 'stopProcessing'],
        },
      },
    },
    chatting: {
      entry: ['startProcessing', 'setTyping'],
      invoke: {
        src: 'processChatMessage',
        onDone: {
          target: 'idle',
          actions: ['addChatMessage', 'stopProcessing', 'stopTyping'],
        },
        onError: {
          target: 'idle',
          actions: ['setError', 'stopProcessing', 'stopTyping'],
        },
      },
    },
    uploadingDocument: {
      entry: 'startProcessing',
      invoke: {
        src: 'uploadDocument',
        onDone: {
          target: 'idle',
          actions: ['addDocument', 'stopProcessing', 'addNotification'],
        },
        onError: {
          target: 'idle',
          actions: ['setError', 'stopProcessing'],
        },
      },
    },
    batchProcessing: {
      entry: 'startProcessing',
      invoke: {
        src: 'processBatch',
        onDone: {
          target: 'idle',
          actions: ['setBatchResults', 'stopProcessing'],
        },
        onError: {
          target: 'idle',
          actions: ['setError', 'stopProcessing'],
        },
      },
    },
    refreshingMetrics: {
      invoke: {
        src: 'refreshMetrics',
        onDone: {
          target: 'idle',
          actions: 'setMetrics',
        },
        onError: {
          target: 'idle',
          actions: 'setError',
        },
      },
    },
    clearingCache: {
      invoke: {
        src: 'clearAllCaches',
        onDone: {
          target: 'idle',
          actions: ['clearCacheStats', 'addNotification'],
        },
        onError: {
          target: 'idle',
          actions: 'setError',
        },
      },
    },
    error: {
      on: {
        RETRY: 'initializing',
        RESET: {
          target: 'idle',
          actions: 'clearError',
        },
      },
    },
  },
}, {
  actions: {
    setInitialized: (context, event) => {
      context.gpuAvailable = event.data.gpuAvailable;
      context.cacheStats = event.data.cacheStats;
    },
    setUser: (context, event) => {
      context.isAuthenticated = true;
      context.user = event.data.user;
      context.lastError = null;
    },
    setAuthError: (context, event) => {
      context.lastError = 'Authentication failed';
      context.errors.push(event.data.message || 'Authentication failed');
    },
    setError: (context, event) => {
      const error = event.data?.message || 'An error occurred';
      context.lastError = error;
      context.errors.push(error);
      context.isProcessing = false;
    },
    clearError: (context) => {
      context.lastError = null;
      context.errors = [];
    },
    startProcessing: (context, event) => {
      context.isProcessing = true;
      context.processingStage = event.stage || 'Processing...';
      context.progressPercentage = 0;
    },
    stopProcessing: (context) => {
      context.isProcessing = false;
      context.processingStage = '';
      context.progressPercentage = 100;
    },
    setTyping: (context) => {
      context.isTyping = true;
    },
    stopTyping: (context) => {
      context.isTyping = false;
    },
    setSearchResults: (context, event) => {
      context.currentSearchResults = event.data.results;
      context.searchHistory.unshift({
        id: Date.now().toString(),
        query: event.data.query,
        practice_area: event.data.practice_area,
        document_type: event.data.document_type,
        filters: event.data.filters || {},
        results: event.data.results,
        total_results: event.data.total_found,
        processing_info: event.data.processing_info,
        timestamp: new Date().toISOString(),
        cached: event.data.processing_info?.cache_operations > 0,
      });
      
      // Keep only last 50 searches
      if (context.searchHistory.length > 50) {
        context.searchHistory = context.searchHistory.slice(0, 50);
      }
      
      // Update performance metrics
      if (event.data.processing_info) {
        context.performanceMetrics.total_queries += 1;
        context.performanceMetrics.avg_response_time = 
          (context.performanceMetrics.avg_response_time * (context.performanceMetrics.total_queries - 1) + 
           event.data.performance_info.total_time_ms) / context.performanceMetrics.total_queries;
      }
    },
    setCurrentCase: (context, event) => {
      context.currentCase = event.data.case;
      context.recentDocuments = event.data.documents || [];
    },
    addChatMessage: (context, event) => {
      const message: AIMessage = {
        id: Date.now().toString(),
        type: 'assistant',
        content: event.data.response,
        timestamp: new Date().toISOString(),
        metadata: event.data.metadata,
      };
      context.messageHistory.push(message);
      
      // Keep only last 100 messages
      if (context.messageHistory.length > 100) {
        context.messageHistory = context.messageHistory.slice(-100);
      }
    },
    addDocument: (context, event) => {
      context.recentDocuments.unshift(event.data.document);
      
      // Keep only last 20 recent documents
      if (context.recentDocuments.length > 20) {
        context.recentDocuments = context.recentDocuments.slice(0, 20);
      }
    },
    setBatchResults: (context, event) => {
      // Handle batch processing results
      context.performanceMetrics.gpu_utilization = event.data.gpu_used ? 100 : 0;
      context.tensorCoresActive = event.data.tensor_cores_used;
    },
    setMetrics: (context, event) => {
      context.performanceMetrics = {
        ...context.performanceMetrics,
        ...event.data.performance,
      };
      context.cacheStats = {
        ...context.cacheStats,
        ...event.data.cache_stats,
      };
    },
    clearCacheStats: (context) => {
      context.cacheStats = {
        loki_entries: 0,
        redis_entries: 0,
        postgres_entries: 0,
        total_hits: 0,
        total_misses: 0,
      };
    },
    setActiveTab: (context, event) => {
      context.activeTab = event.tab;
    },
    toggleSidebar: (context) => {
      context.sidebarOpen = !context.sidebarOpen;
    },
    addNotification: (context, event) => {
      const notification = {
        id: Date.now().toString(),
        type: event.data.type || 'info',
        message: event.data.message || 'Operation completed',
        timestamp: new Date().toISOString(),
        dismissed: false,
      };
      context.notifications.push(notification);
      
      // Auto-dismiss info notifications after 5 seconds
      if (notification.type === 'info') {
        setTimeout(() => {
          const index = context.notifications.findIndex(n => n.id === notification.id);
          if (index !== -1) {
            context.notifications[index].dismissed = true;
          }
        }, 5000);
      }
    },
    dismissNotification: (context, event) => {
      const index = context.notifications.findIndex(n => n.id === event.id);
      if (index !== -1) {
        context.notifications[index].dismissed = true;
      }
    },
  },
  services: {
    initializeServices: async () => {
      // Check GPU availability and initialize services
      const health = await gpuService.getHealth();
      const cacheStats = legalDB.getStats();
      
      return {
        gpuAvailable: health.gpu_service,
        cacheStats: {
          loki_entries: cacheStats.embeddings_count + cacheStats.search_results_count,
          redis_entries: 0, // Would come from Redis ping
          postgres_entries: 0, // Would come from DB query
          total_hits: 0,
          total_misses: 0,
        },
      };
    },
    authenticate: async (context, event) => {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: event.email,
          password: event.password,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Authentication failed');
      }
      
      return await response.json();
    },
    performSearch: async (context, event) => {
      const request: Neo4jSearchRequest = {
        query: event.query,
        practice_area: event.practiceArea || '',
        document_type: event.documentType || '',
        max_results: event.maxResults || 20,
        min_confidence: event.minConfidence || 0.1,
        use_gpu: context.gpuAvailable,
        use_fp16: true,
        use_cache: true,
        batch_optimization: true,
        metadata: event.metadata || {},
      };
      
      const result = await gpuService.enhancedSearch(request);
      
      return {
        ...result,
        query: event.query,
        practice_area: event.practiceArea,
        document_type: event.documentType,
        filters: event.metadata,
      };
    },
    loadCase: async (context, event) => {
      const response = await fetch(`/api/cases/${event.caseId}`, {
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (!response.ok) {
        throw new Error('Failed to load case');
      }
      
      return await response.json();
    },
    processChatMessage: async (context, event) => {
      // Add user message first
      const userMessage: AIMessage = {
        id: Date.now().toString(),
        type: 'user',
        content: event.message,
        timestamp: new Date().toISOString(),
      };
      context.messageHistory.push(userMessage);
      
      // Process with AI
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: event.message,
          context: context.currentCase ? { case_id: context.currentCase.id } : {},
          history: context.messageHistory.slice(-10), // Last 10 messages for context
        }),
      });
      
      if (!response.ok) {
        throw new Error('Chat processing failed');
      }
      
      return await response.json();
    },
    uploadDocument: async (context, event) => {
      const formData = new FormData();
      formData.append('file', event.file);
      formData.append('case_id', context.currentCase?.id || '');
      formData.append('document_type', event.documentType || 'document');
      formData.append('practice_area', event.practiceArea || '');
      
      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Document upload failed');
      }
      
      return await response.json();
    },
    processBatch: async (context, event) => {
      const request: GPUEmbeddingRequest = {
        texts: event.texts,
        use_cache: true,
        normalize: true,
        fp16_precision: context.gpuAvailable,
      };
      
      return await gpuService.generateEmbeddings(event.texts, request);
    },
    refreshMetrics: async () => {
      return await gpuService.getMetrics();
    },
    clearAllCaches: async () => {
      await gpuService.clearCache();
      return { success: true };
    },
  },
});

// Initialize machine (only in browser)
export const legalAIService = browser ? interpret(legalAIMachine).start() : null;

// Reactive stores
export const legalAIState = writable(legalAIService?.getSnapshot()?.value || 'initializing');
export const legalAIContext = writable(legalAIService?.getSnapshot()?.context || {} as LegalAIState);

// Update stores when machine state changes
if (browser && legalAIService) {
  legalAIService.subscribe((state) => {
    legalAIState.set(state.value);
    legalAIContext.set(state.context as LegalAIState);
  });
}

// Derived stores for common UI patterns
export const isAuthenticated = derived(legalAIContext, ($context) => $context.isAuthenticated);
export const currentUser = derived(legalAIContext, ($context) => $context.user);
export const isProcessing = derived(legalAIContext, ($context) => $context.isProcessing);
export const isTyping = derived(legalAIContext, ($context) => $context.isTyping);
export const currentCase = derived(legalAIContext, ($context) => $context.currentCase);
export const searchResults = derived(legalAIContext, ($context) => $context.currentSearchResults);
export const messageHistory = derived(legalAIContext, ($context) => $context.messageHistory);
export const recentDocuments = derived(legalAIContext, ($context) => $context.recentDocuments);
export const gpuAvailable = derived(legalAIContext, ($context) => $context.gpuAvailable);
export const performanceMetrics = derived(legalAIContext, ($context) => $context.performanceMetrics);
export const cacheStats = derived(legalAIContext, ($context) => $context.cacheStats);
export const activeTab = derived(legalAIContext, ($context) => $context.activeTab);
export const sidebarOpen = derived(legalAIContext, ($context) => $context.sidebarOpen);
export const notifications = derived(legalAIContext, ($context) => 
  $context.notifications.filter(n => !n.dismissed)
);
export const hasErrors = derived(legalAIContext, ($context) => $context.errors.length > 0);
export const lastError = derived(legalAIContext, ($context) => $context.lastError);

// Computed metrics
export const cacheHitRatio = derived(cacheStats, ($stats) => {
  const total = $stats.total_hits + $stats.total_misses;
  return total > 0 ? ($stats.total_hits / total) * 100 : 0;
});

export const totalCacheEntries = derived(cacheStats, ($stats) => 
  $stats.loki_entries + $stats.redis_entries + $stats.postgres_entries
);

// Action creators for common operations
export const legalAIActions = {
  // Authentication
  login: (email: string, password: string) => {
    legalAIService?.send({ type: 'LOGIN', email, password });
  },

  // Search
  search: (query: string, options: Partial<Neo4jSearchRequest> = {}) => {
    legalAIService?.send({ 
      type: 'SEARCH', 
      query,
      practiceArea: options.practice_area || '',
      documentType: options.document_type || '',
      maxResults: options.max_results || 20,
      minConfidence: options.min_confidence || 0.1,
      metadata: options.metadata || {},
    });
  },

  // Case management
  loadCase: (caseId: string) => {
    legalAIService?.send({ type: 'LOAD_CASE', caseId });
  },

  // Chat
  sendMessage: (message: string) => {
    legalAIService?.send({ type: 'CHAT', message });
  },

  // Document upload
  uploadDocument: (file: File, documentType: string, practiceArea: string) => {
    legalAIService?.send({ 
      type: 'UPLOAD_DOCUMENT', 
      file, 
      documentType, 
      practiceArea 
    });
  },

  // Batch processing
  processBatch: (texts: string[]) => {
    legalAIService?.send({ type: 'PROCESS_BATCH', texts });
  },

  // UI actions
  setActiveTab: (tab: LegalAIState['activeTab']) => {
    legalAIService?.send({ type: 'SET_ACTIVE_TAB', tab });
  },

  toggleSidebar: () => {
    legalAIService?.send({ type: 'TOGGLE_SIDEBAR' });
  },

  dismissNotification: (id: string) => {
    legalAIService?.send({ type: 'DISMISS_NOTIFICATION', id });
  },

  // System actions
  refreshMetrics: () => {
    legalAIService?.send({ type: 'REFRESH_METRICS' });
  },

  clearCache: () => {
    legalAIService?.send({ type: 'CLEAR_CACHE' });
  },

  // Error handling
  retry: () => {
    legalAIService?.send({ type: 'RETRY' });
  },

  reset: () => {
    legalAIService?.send({ type: 'RESET' });
  },
};

// Reactive database instance
export { legalDB };

// Export everything needed for components
export default {
  // Machine and service
  legalAIMachine,
  legalAIService,
  
  // Stores
  legalAIState,
  legalAIContext,
  
  // Derived stores
  isAuthenticated,
  currentUser,
  isProcessing,
  isTyping,
  currentCase,
  searchResults,
  messageHistory,
  recentDocuments,
  gpuAvailable,
  performanceMetrics,
  cacheStats,
  cacheHitRatio,
  totalCacheEntries,
  activeTab,
  sidebarOpen,
  notifications,
  hasErrors,
  lastError,
  
  // Actions
  legalAIActions,
  
  // Database
  legalDB,
};

// Auto-refresh metrics every 30 seconds
if (browser && legalAIService) {
  setInterval(() => {
    legalAIActions.refreshMetrics();
  }, 30000);
}