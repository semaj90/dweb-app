// XState Store Manager
// Centralized management for all XState machines in the application

import { writable, derived, type Readable } from 'svelte/store';
import { interpret, type InterpreterFrom } from 'xstate';
import { browser } from '$app/environment';

// Import all state machines
import { appStateMachine, type AppService, type AppContext } from '../machines/app-state-machine';
import { documentUploadMachine, type DocumentUploadService, type DocumentUploadContext } from '../machines/document-upload-machine';
import { chatMachine, type ChatService, type ChatContext } from '../machines/chat-machine';
import { searchMachine, type SearchService, type SearchContext } from '../machines/search-machine';

// Store instances for each machine
export const appActor = writable<AppService | null>(null);
export const documentUploadActor = writable<DocumentUploadService | null>(null);
export const chatActor = writable<ChatService | null>(null);
export const searchActor = writable<SearchService | null>(null);

// State stores for reactive access
export const appState = writable<AppContext | null>(null);
export const documentUploadState = writable<DocumentUploadContext | null>(null);
export const chatState = writable<ChatContext | null>(null);
export const searchState = writable<SearchContext | null>(null);

// Machine instances
let appService: AppService | null = null;
let documentUploadService: DocumentUploadService | null = null;
let chatService: ChatService | null = null;
let searchService: SearchService | null = null;

/**
 * XState Store Manager
 * Provides centralized initialization and management of all state machines
 */
export class XStateStoreManager {
  private initialized = false;
  private services = new Map<string, any>();

  /**
   * Initialize all state machines
   */
  async initialize(): Promise<void> {
    if (!browser || this.initialized) return;

    try {
      console.log('🔄 Initializing XState Store Manager...');

      // Initialize app state machine
      await this.initializeAppMachine();
      
      // Initialize document upload machine
      await this.initializeDocumentUploadMachine();
      
      // Initialize chat machine
      await this.initializeChatMachine();
      
      // Initialize search machine
      await this.initializeSearchMachine();

      // Set up cross-machine communication
      this.setupCrossMachineCommunication();

      this.initialized = true;
      console.log('✓ XState Store Manager initialized successfully');

    } catch (error) {
      console.error('❌ XState Store Manager initialization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize application state machine
   */
  private async initializeAppMachine(): Promise<void> {
    appService = interpret(appStateMachine)
      .onTransition((state) => {
        appState.set(state.context);
      })
      .start();

    this.services.set('app', appService);
    appActor.set(appService);

    console.log('✓ App state machine initialized');
  }

  /**
   * Initialize document upload state machine
   */
  private async initializeDocumentUploadMachine(): Promise<void> {
    documentUploadService = interpret(documentUploadMachine)
      .onTransition((state) => {
        documentUploadState.set(state.context);
      })
      .start();

    this.services.set('documentUpload', documentUploadService);
    documentUploadActor.set(documentUploadService);

    console.log('✓ Document upload state machine initialized');
  }

  /**
   * Initialize chat state machine
   */
  private async initializeChatMachine(): Promise<void> {
    chatService = interpret(chatMachine)
      .onTransition((state) => {
        chatState.set(state.context);
      })
      .start();

    this.services.set('chat', chatService);
    chatActor.set(chatService);

    console.log('✓ Chat state machine initialized');
  }

  /**
   * Initialize search state machine
   */
  private async initializeSearchMachine(): Promise<void> {
    searchService = interpret(searchMachine)
      .onTransition((state) => {
        searchState.set(state.context);
      })
      .start();

    this.services.set('search', searchService);
    searchActor.set(searchService);

    console.log('✓ Search state machine initialized');
  }

  /**
   * Set up communication between state machines
   */
  private setupCrossMachineCommunication(): void {
    // Listen for user authentication changes and propagate to other machines
    appService?.onTransition((state) => {
      if (state.changed) {
        const { user, isAuthenticated } = state.context;
        
        // Notify other machines of authentication changes
        if (state.matches('authenticated') && user) {
          this.broadcastUserAuthenticated(user);
        } else if (state.matches('unauthenticated')) {
          this.broadcastUserLoggedOut();
        }
      }
    });

    // Listen for document upload completion and update chat context
    documentUploadService?.onTransition((state) => {
      if (state.matches('completed') && state.context.results.length > 0) {
        // Notify chat machine about new documents for context
        const successfulUploads = state.context.results.filter(r => r.success);
        if (successfulUploads.length > 0) {
          this.broadcastDocumentsUploaded(successfulUploads);
        }
      }
    });

    // Listen for search results and update chat suggestions
    searchService?.onTransition((state) => {
      if (state.matches('results') && state.context.results.length > 0) {
        // Update chat machine with relevant search context
        this.broadcastSearchResults(state.context.query, state.context.results);
      }
    });

    console.log('✓ Cross-machine communication setup complete');
  }

  /**
   * Broadcast user authentication to all machines
   */
  private broadcastUserAuthenticated(user: any): void {
    // Update chat machine with user context
    chatService?.send({
      type: 'UPDATE_SETTINGS',
      settings: {
        userContext: {
          userId: user.id,
          role: user.role,
          preferences: user.preferences
        }
      }
    });

    // Update search machine with user preferences
    searchService?.send({
      type: 'UPDATE_SETTINGS',
      settings: {
        userRole: user.role,
        personalizedResults: true
      }
    });

    console.log('📡 User authentication broadcasted to all machines');
  }

  /**
   * Broadcast user logout to all machines
   */
  private broadcastUserLoggedOut(): void {
    // Clear sensitive data from chat machine
    chatService?.send({ type: 'CLEAR_HISTORY' });

    // Reset search machine to default settings
    searchService?.send({ type: 'CLEAR_SEARCH' });

    console.log('📡 User logout broadcasted to all machines');
  }

  /**
   * Broadcast document upload completion
   */
  private broadcastDocumentsUploaded(documents: any[]): void {
    // Add document context to chat machine
    const documentTitles = documents.map(d => d.documentId || d.fileId).join(', ');
    
    chatService?.send({
      type: 'ADD_NOTIFICATION',
      notification: {
        type: 'success',
        title: 'Documents Uploaded',
        message: `Successfully uploaded: ${documentTitles}`,
        read: false
      }
    });

    console.log('📡 Document upload completion broadcasted');
  }

  /**
   * Broadcast search results for context enhancement
   */
  private broadcastSearchResults(query: string, results: any[]): void {
    // Update chat machine with search context for better responses
    if (results.length > 0) {
      chatService?.send({
        type: 'UPDATE_SETTINGS',
        settings: {
          searchContext: {
            lastQuery: query,
            relevantDocuments: results.slice(0, 3).map(r => ({
              id: r.id,
              title: r.title,
              snippet: r.snippet
            }))
          }
        }
      });
    }

    console.log('📡 Search results context broadcasted to chat machine');
  }

  /**
   * Get service by name
   */
  getService(name: string): any {
    return this.services.get(name);
  }

  /**
   * Send event to specific machine
   */
  sendToMachine(machineName: string, event: any): void {
    const service = this.services.get(machineName);
    if (service) {
      service.send(event);
    } else {
      console.warn(`Machine '${machineName}' not found`);
    }
  }

  /**
   * Broadcast event to all machines
   */
  broadcastEvent(event: any): void {
    this.services.forEach((service, name) => {
      try {
        service.send(event);
      } catch (error) {
        console.warn(`Failed to send event to machine '${name}':`, error);
      }
    });
  }

  /**
   * Get current state of all machines
   */
  getGlobalState(): any {
    const state: any = {};
    
    this.services.forEach((service, name) => {
      state[name] = {
        value: service.state.value,
        context: service.state.context,
        matches: (pattern: string) => service.state.matches(pattern)
      };
    });

    return state;
  }

  /**
   * Stop all machines
   */
  stopAll(): void {
    this.services.forEach((service) => {
      service.stop();
    });
    this.services.clear();
    this.initialized = false;
    
    // Clear stores
    appActor.set(null);
    documentUploadActor.set(null);
    chatActor.set(null);
    searchActor.set(null);
    
    console.log('🛑 All XState machines stopped');
  }

  /**
   * Restart all machines
   */
  async restart(): Promise<void> {
    this.stopAll();
    await this.initialize();
  }

  /**
   * Get performance metrics from all machines
   */
  getPerformanceMetrics(): any {
    const metrics: any = {
      timestamp: Date.now(),
      machines: {}
    };

    this.services.forEach((service, name) => {
      const context = service.state.context;
      
      metrics.machines[name] = {
        state: service.state.value,
        performance: context.performance || {},
        uptime: Date.now() - (context.startTime || Date.now())
      };
    });

    return metrics;
  }

  /**
   * Export machine configurations for debugging
   */
  exportConfigurations(): any {
    const configs: any = {};
    
    this.services.forEach((service, name) => {
      configs[name] = {
        id: service.machine.id,
        initial: service.machine.initial,
        states: Object.keys(service.machine.states),
        context: service.state.context
      };
    });

    return configs;
  }

  /**
   * Health check for all machines
   */
  healthCheck(): any {
    const health: any = {
      overall: 'healthy',
      machines: {},
      issues: []
    };

    this.services.forEach((service, name) => {
      const isHealthy = !service.state.matches('error');
      
      health.machines[name] = {
        status: isHealthy ? 'healthy' : 'error',
        state: service.state.value,
        lastTransition: service.state.changed
      };

      if (!isHealthy) {
        health.issues.push(`Machine '${name}' is in error state`);
        health.overall = 'degraded';
      }
    });

    return health;
  }
}

// Create singleton instance
export const xstateManager = new XStateStoreManager();

// Derived stores for common state combinations
export const globalAppState = derived(
  [appState, chatState, searchState, documentUploadState],
  ([$app, $chat, $search, $upload]) => ({
    app: $app,
    chat: $chat,
    search: $search,
    upload: $upload,
    isAuthenticated: $app?.isAuthenticated || false,
    currentUser: $app?.user || null,
    isProcessing: $chat?.isTyping || $search?.isSearching || $upload?.currentFile !== null,
    unreadNotifications: $app?.notifications?.filter(n => !n.read).length || 0
  })
);

// Reactive performance monitoring
export const performanceMetrics = derived(
  [appState, chatState, searchState, documentUploadState],
  ([$app, $chat, $search, $upload]) => ({
    app: $app?.performance || {},
    chat: $chat?.performance || {},
    search: $search?.performance || {},
    upload: $upload?.performance || {},
    timestamp: Date.now()
  })
);

// Auto-initialize on browser load
if (browser) {
  xstateManager.initialize().catch(console.error);
}

// Helper functions for common operations
export const xstateHelpers = {
  // App helpers
  login: (credentials: { email: string; password: string }) => {
    appService?.send({ type: 'LOGIN', credentials });
  },

  logout: () => {
    appService?.send({ type: 'LOGOUT' });
  },

  addNotification: (notification: any) => {
    appService?.send({ type: 'ADD_NOTIFICATION', notification });
  },

  // Chat helpers
  createChatSession: (title?: string, context?: any) => {
    chatService?.send({ type: 'CREATE_SESSION', title, context });
  },

  sendChatMessage: (message: string) => {
    chatService?.send({ type: 'UPDATE_MESSAGE', message });
    chatService?.send({ type: 'SEND_MESSAGE' });
  },

  // Search helpers
  performSearch: (query: string) => {
    searchService?.send({ type: 'UPDATE_QUERY', query });
    searchService?.send({ type: 'SEARCH' });
  },

  applySearchFilters: (filters: any) => {
    searchService?.send({ type: 'APPLY_FILTERS', filters });
  },

  // Upload helpers
  uploadFiles: (files: File[]) => {
    documentUploadService?.send({ type: 'ADD_FILES', files });
    documentUploadService?.send({ type: 'START_UPLOAD' });
  },

  // Cross-machine operations
  searchAndChat: (query: string) => {
    // First perform search
    searchService?.send({ type: 'UPDATE_QUERY', query });
    searchService?.send({ type: 'SEARCH' });
    
    // Then create chat session with search context
    chatService?.send({ 
      type: 'CREATE_SESSION', 
      title: `Search: ${query}`,
      context: { searchQuery: query }
    });
  }
};

// Debug utilities (only available in development)
if (typeof window !== 'undefined' && import.meta.env.DEV) {
  (window as any).xstateDebug = {
    manager: xstateManager,
    getState: () => xstateManager.getGlobalState(),
    getMetrics: () => xstateManager.getPerformanceMetrics(),
    healthCheck: () => xstateManager.healthCheck(),
    restart: () => xstateManager.restart()
  };
  
  console.log('🔧 XState debug utilities available at window.xstateDebug');
}

export default xstateManager;